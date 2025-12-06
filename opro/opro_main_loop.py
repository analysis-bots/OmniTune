import ast
import os
import re
import json
import random
from typing import Any, List, Dict, Callable, Optional, Tuple, Union

import duckdb
import numpy as np

from opro.opro_components import AssignmentLM, SubspaceLM
from functionality.constraint import AgnosticConstraint, OutputConstraint
from functionality.task import Task, AgnosticTask, RangeQueryRefinementTask
from tools.logging_utils import format_data_for_log, ensure_log_directory, log_to_file

import pandas as pd

from opro.constraint_parser import ConstraintParser, MockParser
from config import LOG_DIR
from functionality.predicate import NumericalPredicate, CategoricalPredicate, NumericalAttribute, Predicate
from functionality.subspace_data_structure import SubspaceStructure, SubspaceDict
from functionality.objectives import get_script_diff_func_sql, get_basic_refinement_validation_function
from functionality.skyline import Skyline
from tools.chat_history import History
from tools.utils import register_duckdb_tables, get_primary_dataframe, \
    build_dataset_schema_context, subsets_containing


class OmniTuneEngine:
    def __init__(
            self,
            task_name: str,
            original_query: str,
            input_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
            constraints: Union[str ,List[Union[AgnosticConstraint, OutputConstraint]]],
            epsilon: float,
            distance_func: Callable[[str], float],
            refineable_predicates_str: str,
            refineable_predicates: List[Predicate],
            max_subspace_iters: int = 5,
            max_refinements: int = 5,
            perform_analysis: bool = True,
            perform_optimization: bool = True,  # Add flag for optimization feature
            max_optimization_attempts: int = 5,  # Control optimization iterations
            max_assignments_per_subspace: int = 2,  # Added from EnhancedRefinementOrchestrator
            max_no_improve_outer_iters: int = 1,  # Early stop in optimization mode after N non-improving iters
            u: int = 0,  # Skyline stability horizon (consecutive outer iterations)
            parse_constraints: bool = True,
            assignment_lm_only_mode: bool = False,
            is_range_task: bool = False,
            subspace_lm_only_random: bool = False,
            random_seed: Optional[int] = None,
            one_shot_mode: bool = False,
            is_having_query: bool = False,
            log_dir: str = LOG_DIR,
            use_history: bool = True,
            use_skyline: bool = True,
            model_provider: Optional[str] = None,
            model_name: Optional[str] = None
    ):
        """
        Initialize the refinement orchestrator.

        Args:
            task_name: Name of the QRP instance
            original_query: The original SQL query to refine
            input_dataset: The input dataset to query against
            constraints: String representation of constraints to satisfy
            epsilon: Threshold for constraint satisfaction
            distance_func: Function to calculate refinement distance
            refineable_predicates_str: String representation of refineable predicates
            refineable_predicates: List of Predicate objects representing refineable predicates
            max_subspace_iters: Maximum number of subspace iterations
            max_refinements: Maximum number of refinements per subspace
            perform_analysis: Whether to perform data analysis
            perform_optimization: Whether to perform query optimization
            max_optimization_attempts: Maximum number of optimization attempts
            max_assignments_per_subspace: Maximum number of attempts for the AssignmentLM per subspace
            assignment_lm_only_mode: If True, runs without the SubspaceLM's subspace selection mechanism.
            subspace_lm_only_random: If True, uses SubspaceLM for subspace selection but samples random assignments (ablation mode).
            random_seed: Optional random seed for reproducible random sampling in subspace_lm_only_random mode.
            one_shot_mode: If True, asks LLM for a single solution from the entire search space without iterations.
            use_history: If False, skip summarized refinement history in prompts.
            use_skyline: If False, disable skyline-based candidate tracking and prompts.
        """
        self.original_query = original_query
        self.subspace_radius_structure = SubspaceStructure(refineable_predicates)
        self.constraints = constraints
        self.epsilon = epsilon
        self.refinement_distance_metric_func = distance_func
        self.refineable_predicates = refineable_predicates
        self.refineable_predicates_str = refineable_predicates_str
        self.max_subspace_iters = max_subspace_iters
        self.max_refinements = max_refinements
        # Normalize dataset and register DuckDB tables (single or multi-table)
        register_duckdb_tables(input_dataset)
        self.input_dataset = get_primary_dataframe(input_dataset)
        self._all_tables = input_dataset if isinstance(input_dataset, dict) else {"df": input_dataset}
        self.assignment_lm_only_mode = assignment_lm_only_mode
        self.is_range_task = is_range_task
        self.subspace_lm_only_random = subspace_lm_only_random
        self._rng = random.Random(random_seed) if random_seed is not None else random
        self.one_shot_mode = one_shot_mode
        self.is_having_query = is_having_query
        self.refineable_predicates_context_json = None
        self.use_history = use_history
        self.use_skyline = use_skyline
        self.is_optimization_mode = False
        self._skyline_history: List[tuple] = []
        try:
            self._skyline_stability_horizon = max(0, int(u))
        except Exception:
            self._skyline_stability_horizon = 0

        if assignment_lm_only_mode:
            self.refineable_predicates_context_json = json.dumps([
                {
                    "id": p.get_id(),
                    "attribute_name": p.attribute.name,
                    "operator": p.operator if isinstance(p, NumericalPredicate) else "IN",
                    "current_value": p.value if isinstance(p, NumericalPredicate) else p.values,
                    "value_type": "numerical" if isinstance(p, NumericalPredicate) else "categorical",
                    "valid_value_range": p.get_valid_value_range()
                }
                for i, p in enumerate(refineable_predicates)
            ])


        if parse_constraints:
            self.constraint_parser = ConstraintParser(input_dataset, constraints, original_query)
            # Parse constraints to get structured objects
            constraints_str = self.constraint_parser.get_constraints_str()
        else:
            # If not parsing, assume constraints are already in string format
            constraints_str = "\n".join([str(c) for c in constraints])
            self.constraint_parser = MockParser(
                parsed_constraints=constraints,
                input_dataset=input_dataset,
                constraints_str=constraints_str,
                original_query=original_query
            )
        self.task_name = task_name

        df = self.input_dataset
        try:
            initial_result = duckdb.query(original_query).df()
        except Exception:
            initial_result = df

        constraints_list = []
        for i, c in enumerate(self.constraint_parser.parsed_constraints):
            status = "unknown"
            if hasattr(c, 'string_evaluation'):
                try:
                    # Try evaluating on the original query result; if it fails, fall back to dataset
                    status = c.string_evaluation(initial_result).strip("• ")
                except Exception:
                    try:
                        status = c.string_evaluation(df).strip("• ")
                    except Exception:
                        status = "unknown"
            constraints_list.append({
                "id": f"C{i + 1}",
                "constraint_str": c.query_str if hasattr(c, 'query_str') else c.get_query_str(),
                "description_concise": str(c).strip("• ").strip("\n"),
                "target_value": getattr(c, "desired_value", "target_value"),
                "current_satisfaction_status": status,
            })

        self.constraints_json = json.dumps(constraints_list)

        # Ensure the task-specific log directory exists
        self.log_dir = ensure_log_directory(log_dir, task_name)

        # Generate refinement objective description from the distance function
        refinement_objective_description = self._generate_refinement_objective_description(
            distance_func, refineable_predicates_str
        )

        # Note: AssignmentLM is still initialized even in subspace_lm_only_random mode to maintain structure
        # consistency, but won't be used for query generation when subspace_lm_only_random=True
        self.assignment_lm = AssignmentLM(
            task_name=f"{task_name}_assignment_lm",
            constraints_json=self.constraints_json,
            input_dataset=self._all_tables if isinstance(input_dataset, dict) else self.input_dataset,
            original_query=original_query,
            epsilon=epsilon,
            refineable_predicates=refineable_predicates,
            refinement_objective_description=refinement_objective_description,
            log_dir=self.log_dir,
            use_history=use_history,
            use_skyline=self.use_skyline,
            model_provider=model_provider,
            model_name=model_name
        )
        if self.assignment_lm_only_mode:
            self.subspace_lm = None
        else:
            self.subspace_lm = SubspaceLM(
                task_name=f"{task_name}_subspace_lm",
                constraints=constraints_str,
                epsilon=epsilon,
                input_dataset=self._all_tables if isinstance(input_dataset, dict) else self.input_dataset,
                parsed_constraints=self.constraint_parser.parsed_constraints,
                original_query=original_query,
                refineable_predicates=refineable_predicates,
                refinement_objective_description=refinement_objective_description,
                log_dir=self.log_dir,
                use_history=use_history,
                use_skyline=self.use_skyline,
                model_provider=model_provider,
                model_name=model_name
            )
        self.skyline = Skyline() if self.use_skyline else None
        # Initialize the basic query validator
        self.basic_query_validator = get_basic_refinement_validation_function(
            self.original_query,
            refineable_predicates
        )

        # Set the maximum number of assignment_lm attempts per subspace
        self.max_assignments_per_subspace = max_assignments_per_subspace

        self.history = History()
        # Initialize memory_df
        pred_cols = []
        for p in refineable_predicates:
            if isinstance(p, NumericalPredicate):
                col = p.attribute.name + ("_min" if ">" in p.operator else "_max")
                pred_cols.append(col)
            else:
                pred_cols.append(f"{p.attribute.name}_set")
        self.pred_columns = pred_cols
        try:
            self.constraint_columns = [c.query_str for c in self.constraint_parser.parsed_constraints]
        except AttributeError:
            self.constraint_columns = [c.get_query_str() for c in self.constraint_parser.parsed_constraints]
        self.subspace_memory_df = pd.DataFrame \
            (columns=["query", *pred_cols, *self.constraint_columns, "distance", "notes", "constraint_score"])

        # Initialize all_refinements_history_df for consolidated logging
        self.all_refinements_history_df = pd.DataFrame \
            (columns=["query", *pred_cols, *self.constraint_columns, "distance", "notes", "constraint_score"])

        self.pred_cols_range = [f"{col_name} range" for col_name in pred_cols]
        self.const_cols_avg = [f"{col_name} average" for col_name in self.constraint_columns]
        self.accumulated_memory_df = pd.DataFrame(columns=[*self.pred_cols_range, *self.const_cols_avg, "distance"])
        self.Q_star = None
        self.d_star = float('inf')
        self.perform_analysis = perform_analysis
        self.perform_optimization = perform_optimization  # Store the optimization flag
        # Early stopping control for optimization mode
        self.max_no_improve_outer_iters = max_no_improve_outer_iters
        self._no_improve_outer_iters = 0

        # Track best satisfying query for optimization mode context
        self._best_satisfying_query = None

        # Create a mapping from LLM predicate ID (e.g., "P1") to actual DataFrame column name (e.g., "LSAT_min")
        self.llm_pid_to_column_name_map = {}
        for i, p_obj in enumerate(self.refineable_predicates):
            llm_pid = p_obj.get_id()
            # Determine the column name as it appears in subspace_memory_df.columns
            # This logic should match how self.pred_columns is populated
            col_name_in_df = ""
            llm_pid_sign = llm_pid
            if isinstance(p_obj, NumericalPredicate):
                col_name_in_df = p_obj.attribute.name + ("_min" if ">" in p_obj.operator else "_max")
                llm_pid_sign = llm_pid + ("_min" if ">" in p_obj.operator else "_max")
            elif isinstance(p_obj, CategoricalPredicate):
                col_name_in_df = f"{p_obj.attribute.name}_set"
                llm_pid_sign = llm_pid + "_set"

            if col_name_in_df:
                self.llm_pid_to_column_name_map[llm_pid] = col_name_in_df
                # Also map the attribute name itself, in case the LLM uses that
                self.llm_pid_to_column_name_map[p_obj.attribute.name] = col_name_in_df
                # And the column name as it appears in the df, in case LLM uses that
                self.llm_pid_to_column_name_map[col_name_in_df] = col_name_in_df
                # Also map the LLM PID with a sign suffix if applicable
                self.llm_pid_to_column_name_map[llm_pid_sign] = col_name_in_df

    def _skyline(self):
        if not self.use_skyline:
            return None
        return self.skyline

    def _insert_into_skyline(self, q: str, d: float, csat: float) -> None:
        if not self.use_skyline or self.skyline is None:
            return
        try:
            self.skyline.insert_query(q, float(d), float(csat))
        except Exception:
            pass

    def _maximize_pareto(self, triples):
        if not self.use_skyline or self.skyline is None:
            if not triples:
                return None
            refinements_that_satisfy_constraints = [t for t in triples if t[2] is not None and t[2] <= self.epsilon]
            if len(refinements_that_satisfy_constraints) > 0:
                triples = refinements_that_satisfy_constraints
            def _sort_key(triple):
                if len(refinements_that_satisfy_constraints) > 0:
                    q, dist, csat = triple
                    dist = dist if dist is not None else float("inf")
                    return (dist,)
                q, dist, csat = triple
                csat_val = csat if csat is not None else float("inf")
                return (csat_val, dist)

            return sorted(triples, key=_sort_key)[0][0]
        try:
            return self.skyline.maximize_pareto_improvement(triples, epsilon=self.epsilon)
        except Exception:
            return None

    def _switch_to_optimization_mode_once(self) -> None:
        if not self.use_skyline or self.skyline is None:
            self.is_optimization_mode = True
        try:
            s = self.skyline
            if hasattr(s, "mode_switched") and not s.mode_switched:
                s.switch_to_optimization_mode(self.epsilon)
        except Exception:
            pass

    def _in_optimization_mode(self) -> bool:
        if not self.use_skyline or self.skyline is None:
            return self.is_optimization_mode
        try:
            s = self.skyline
            return getattr(s, "mode", None) == "optimization"
        except Exception:
            return False

    def _get_optimization_context(self, is_subspace_lm=False) -> Optional[str]:
        """
        Generate optimization context message when in optimization mode.
        
        Returns:
            Optimization context string if in optimization mode with a satisfying query, None otherwise
        """
        if not self.use_skyline:
            return None
        if not self._in_optimization_mode() or self._best_satisfying_query is None:
            return None
        
        context = f"""
**🎯 OPTIMIZATION MODE ACTIVATED 🎯**

We have identified a satisfying query! All constraints are now satisfied.

**Best Current Satisfying Query:**
```sql
{self._best_satisfying_query}
```

**New Objective:**
Now our objective is to utilize the information from the best satisfying query to minimize the refinement distance from the original query while maintaining constraint satisfaction.
"""

        if is_subspace_lm:
            context += """
Suggest a TIGHT subspace based on the best satisfying query that opens the search space towards **the original query**.
"""

        else:
            context += """
Your main target now is to get the refined query as close as possible to the original query.
Focus on keeping all constraints satisfied while improving the distance metric.
"""
        return context

    def _full_range_subspace_payload(self) -> SubspaceDict:
        numeric_predicates = {}
        categorical_predicates = {}
        for p in self.subspace_radius_structure.original_predicates:
            if isinstance(p, NumericalPredicate):
                vr = p.get_valid_value_range() or {}
                if isinstance(vr, dict) and "min" in vr and "max" in vr:
                    numeric_predicates[p.get_id()] = list(np.arange(vr["min"], vr["max"] + vr.get("step", 1), vr.get("step", 1)))
                elif isinstance(vr, dict) and "valid_values" in vr:
                    numeric_predicates[p.get_id()] = vr["valid_values"]
            elif isinstance(p, CategoricalPredicate):

                [contained, containing] = p.get_valid_value_range()['categories']
                categorical_predicates[p.get_id()] = subsets_containing(contained, containing)

        subspace_payload = SubspaceDict(
            numeric_predicates=numeric_predicates,
            categorical_predicates=categorical_predicates
        )
        return subspace_payload

    def _log_clean_line(self, line: str) -> None:
        """Append a single concise line to the task's clean responses log."""
        try:
            clean_log_file = os.path.join(self.log_dir, "clean_responses.log")
            log_to_file(clean_log_file, line.strip().replace("\n", " ") + "\n")
        except Exception:
            pass

    def _get_acc_tokens(self, agent) -> int:
        """Return accumulated total tokens for a given agent (AssignmentLM/SubspaceLM)."""
        try:
            last_usage, acc_usage = agent._get_token_usage()
            return int(acc_usage.get("total_tokens", 0) or 0)
        except Exception:
            return 0

    def _sample_assignment_from_subspace(self, subspace_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample a random predicate assignment from the given subspace.
        
        Args:
            subspace_payload: Dictionary with 'numeric_predicates' and 'categorical_predicates'
                             where values are lists of possible choices
        
        Returns:
            Dictionary mapping predicate IDs to sampled values
        """
        assignment = {}
        num_map = subspace_payload.get("numeric_predicates") or {}
        cat_map = subspace_payload.get("categorical_predicates") or {}
        
        for pid, choices in num_map.items():
            if choices:
                assignment[pid] = self._rng.choice(choices)
        
        for pid, choices in cat_map.items():
            if choices:
                choice = self._rng.choice(choices)
                # Convert tuple to list for consistency with categorical predicate values
                assignment[pid] = list(choice) if isinstance(choice, tuple) else choice
        
        return assignment

    def _construct_query_from_predicates(self, original_query: str, predicates: List[Predicate]) -> Optional[str]:
        """
        Construct SQL query from list of predicates by replacing WHERE and/or HAVING clauses.
        Handles queries with WHERE, HAVING, or both.
        
        Args:
            original_query: The original SQL query
            predicates: List of predicate objects
            
        Returns:
            New SQL query with modified predicates, or None if construction fails
        """
        try:
            # Convert predicates to string representations
            having_predicate_strings = None
            if self.is_having_query:
                having_predicates = [p for p in predicates if isinstance(p.attribute, NumericalAttribute) and '(' in p.attribute.name and ')' in p.attribute.name]
                having_predicate_strings = [f'{p.attribute.name} {p.operator} {p.value}' for p in having_predicates if isinstance(p, NumericalPredicate)]
                numerical_predicate_strings = [f'{p.attribute.name} {p.operator} {p.value}' for p in predicates if isinstance(p, NumericalPredicate) and p not in having_predicates]
                categorical_predicate_strings = [f'{p.attribute.name} IN ({", ".join(map(repr, p.values))})' for p in predicates if isinstance(p, CategoricalPredicate)]
            else:
                numerical_predicate_strings = [f'"{p.attribute.name}" {p.operator} {p.value}' for p in predicates if isinstance(p, NumericalPredicate)]
                categorical_predicate_strings = [f'"{p.attribute.name}" IN ({", ".join(map(repr, p.values))})' for p in predicates if isinstance(p, CategoricalPredicate)]
            predicate_strings = numerical_predicate_strings + categorical_predicate_strings

            if not predicate_strings and not having_predicate_strings:
                return None
            
            # Join predicates with AND
            new_predicates_content = " AND ".join(predicate_strings)
            having_predicate_content = " AND ".join(having_predicate_strings) if having_predicate_strings else ""

            # Normalize whitespace
            normalized_query = re.sub(r'\s+', ' ', original_query.strip())
            
            # Check if original query has WHERE and/or HAVING clauses
            where_match = re.search(r'\bWHERE\b', normalized_query, re.IGNORECASE)
            group_by_match = re.search(r'\bGROUP\s+BY\b', normalized_query, re.IGNORECASE)
            having_match = re.search(r'\bHAVING\b', normalized_query, re.IGNORECASE)

            if where_match and having_match:
                # Both WHERE and HAVING exist - determine which clause to replace based on predicate context
                # For simplicity, replace the first occurrence (WHERE), preserving HAVING

                select_from = normalized_query[:where_match.start()].strip()
                
                # Extract HAVING clause and everything after
                having_start = having_match.start()
                tail_match = re.search(
                    r'\bHAVING\b(.*?)(\b(?:ORDER\s+BY|LIMIT|UNION)\b.*?)?$',
                    normalized_query[having_start:],
                    re.IGNORECASE | re.DOTALL
                )
                
                if tail_match:
                    post_having = tail_match.group(2).strip() if tail_match.group(2) else ""

                    # Add GROUP BY between WHERE and HAVING
                    group_by_clause = normalized_query[group_by_match.start():having_start].strip()
                    
                    new_query = f"{select_from} WHERE {new_predicates_content} {group_by_clause} HAVING {having_predicate_content}"
                    if post_having:
                        new_query += " " + post_having
                    return new_query.strip()
            
            elif where_match:
                # Only WHERE clause exists
                select_from = normalized_query[:where_match.start()].strip()
                
                # Extract everything after WHERE clause (GROUP BY, ORDER BY, etc.)
                after_where_match = re.search(
                    r'\bWHERE\b.*?(\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION)\b.*?)$',
                    normalized_query,
                    re.IGNORECASE | re.DOTALL
                )
                
                tail = (" " + after_where_match.group(1).strip()) if after_where_match else ""
                return f"{select_from} WHERE {new_predicates_content}{tail}".strip()
            
            elif having_match:
                # Only HAVING clause exists
                select_from = normalized_query[:having_match.start()].strip()
                
                # Extract everything after HAVING clause
                after_having_match = re.search(
                    r'\bHAVING\b.*?(\b(?:ORDER\s+BY|LIMIT|UNION)\b.*?)$',
                    normalized_query,
                    re.IGNORECASE | re.DOTALL
                )
                
                tail = (" " + after_having_match.group(1).strip()) if after_having_match else ""
                return f"{select_from} HAVING {having_predicate_content}{tail}".strip()
            
            else:
                # No WHERE or HAVING clause - insert WHERE before tail clauses if they exist
                tail_match = re.search(
                    r'(\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION)\b.*?)$',
                    normalized_query,
                    re.IGNORECASE
                )
                
                if tail_match:
                    select_from = normalized_query[:tail_match.start()].strip()
                    tail = " " + tail_match.group(1).strip()
                    return f"{select_from} WHERE {new_predicates_content}{tail}".strip()
                
                return f"{normalized_query} WHERE {new_predicates_content}".strip()
                
        except Exception as e:
            return None

    def _build_sql_from_assignment(self, assignment: Dict[str, Any]) -> Tuple[Optional[str], List[Predicate]]:
        """
        Reconstruct SQL query and predicate list from a predicate assignment.
        
        Args:
            assignment: Dictionary mapping predicate IDs to their assigned values
            
        Returns:
            Tuple of (reconstructed_sql_query, predicates_list)
        """
        preds_out = []
        
        for orig in self.subspace_radius_structure.original_predicates:
            pid = orig.get_id()
            
            if isinstance(orig, NumericalPredicate):
                val = assignment.get(pid, orig.value)
                preds_out.append(NumericalPredicate(orig.attribute, valid_value_range=orig.get_valid_value_range(), operator=orig.operator, value=val))
            elif isinstance(orig, CategoricalPredicate):
                vals = assignment.get(pid, orig.values)
                # Ensure vals is a list
                if not isinstance(vals, list):
                    vals = list(vals) if hasattr(vals, '__iter__') and not isinstance(vals, str) else [vals]
                preds_out.append(CategoricalPredicate(orig.attribute, valid_value_range=orig.get_valid_value_range(), values=vals))

        sql = self._construct_query_from_predicates(self.original_query, preds_out)
        return sql, preds_out

    def _generate_refinement_objective_description(
            self,
            distance_func: Callable[[str], float],
            refineable_predicates_str: str
    ) -> str:
        """
        Generate a textual description of the refinement objective from the distance function.

        Args:
            distance_func: The refinement distance function
            refineable_predicates_str: String representation of refineable predicates

        Returns:
            A textual description suitable for the LLM system prompt
        """
        # Try to infer the objective from the function name or type
        func_name = getattr(distance_func, '__name__', str(distance_func))

        if 'script_diff' in func_name.lower() or 'textual' in func_name.lower():
            return "Minimize textual differences between the refined query and the original query structure."
        elif 'range_query_distance' in func_name.lower():
            return "Minimize Result based distance by expanding / shrinking numerical predicate ranges."
        elif 'semantic' in func_name.lower():
            return "Minimize semantic differences while maintaining query functionality."
        elif 'distance' in func_name.lower():
            return "Minimize the refinement distance from the original query."
        else:
            # Generic fallback description
            return ("Minimize the refinement distance from the original query by making the smallest "
                    "possible changes to the refineable predicates while satisfying the target constraints.")

    def record_run(
            self,
            query: str,
            predicate_values: Dict[str, Any],
            constraint_values: Dict[str, float],
            distance: float,
            notes: str
    ):
        constraint_score = self.constraint_parser.evaluate_constraint_score(query)
        row = {"query": query, **predicate_values, **constraint_values, "distance": distance, "notes": notes, "constraint_score": constraint_score}
        # Add to subspace memory DataFrame
        self.subspace_memory_df = pd.concat([self.subspace_memory_df, pd.DataFrame([row])], ignore_index=True)

        # Also add to the consolidated refinements history DataFrame
        self.all_refinements_history_df = pd.concat([self.all_refinements_history_df, pd.DataFrame([row])], ignore_index=True)

    def _calculate_refinement_distance(self, q_refined: str) -> float:
        """Calculates the refinement distance using the specified method."""
        # Now calls the function passed during init
        return self.refinement_distance_metric_func(q_refined)

    def _skyline_snapshot(self) -> tuple:
        """
        Deterministic snapshot of the current skyline keyed by (distance, psi_e, query).
        Uses ψ_ε = min(ψ, ε) to align with paper notation.
        """
        if not getattr(self, "skyline", None) or self.skyline.data.empty:
            return tuple()
        try:
            eps = float(self.epsilon)
        except Exception:
            eps = 0.0

        rows = []
        for _, row in self.skyline.data.iterrows():
            try:
                dist = round(float(row.get("distance", 0.0)), 4)
            except Exception:
                dist = 0.0
            try:
                dev_raw = float(row.get("constraint_deviation", 0.0))
            except Exception:
                dev_raw = 0.0
            dev = round(min(dev_raw, eps), 4)
            query_str = str(row.get("query", ""))
            rows.append((dist, dev, query_str))

        return tuple(sorted(rows))

    def _run_one_shot(self) -> tuple[str | None, float, int]:
        """
        Run one-shot LLM baseline with iterative feedback: runs 10 iterations with conversation
        history, providing feedback (constraint score + distance) after each attempt.
        
        Returns:
            Tuple of (best_query, best_distance, total_tokens)
        """
        ONE_SHOT_MAX_ITERATIONS = 10
        
        print("=" * 80)
        print("RUNNING IN ONE-SHOT ITERATIVE MODE (10 iterations)")
        print("LLM will iteratively refine solutions with feedback")
        print("=" * 80)
        
        # Initial evaluation
        initial_constraint_eval_str = self.constraint_parser.evaluation_str(self.original_query)
        initial_is_satisfied = self.constraint_parser.is_satisfied(self.original_query, self.epsilon)
        
        print(f"Initial Query: {self.original_query}")
        print(f"Initial Constraint Eval: {initial_constraint_eval_str}")
        print(f"Initial Constraint Satisfied (Epsilon={self.epsilon}): {initial_is_satisfied}\n{'- ' *50}")
        
        # Get full search space
        subspace_payload = self._full_range_subspace_payload()
        
        # Create initial one-shot prompt
        one_shot_prompt = f"""
You are given a query refinement task. Provide a SINGLE, BEST solution that satisfies all constraints.

**Task Context:**
- Original Query: {self.original_query}
- Constraints to satisfy: {self.constraints_json}
- Refineable Predicates: {self.refineable_predicates_context_json}
- Available Search Space: {json.dumps(subspace_payload.model_dump() if hasattr(subspace_payload, 'model_dump') else subspace_payload)}

**Your Mission:**
Suggest ONE query refinement that:
1. Satisfies ALL output constraints (constraint deviation = 0.0)
2. Minimizes refinement distance from the original query
3. Uses only valid predicate values from the search space

**Response Format:**
```json
{{
    "selected_predicate_values": {{
         "<predicate_id_1>": <value>,  // For numerical predicates
         "<predicate_id_2>": [<values>],  // For categorical predicates
         ...
    }},
    "selected_refinement": "<complete_sql_query>",
    "reasoning_concise": "Brief explanation of why this solution satisfies constraints with minimal distance"
}}
```
"""
        # Initialize conversation with system prompt and first user message
        messages = self.assignment_lm.messages + [{"role": "user", "content": one_shot_prompt}]
        self.assignment_lm.log_message("user", one_shot_prompt, "One-Shot Refinement Request (Iteration 1)")
        
        # Track best results for graceful fallback
        best_satisfying_query = None
        best_satisfying_distance = float('inf')
        best_csat_query = None
        best_csat = float('inf')
        best_csat_distance = float('inf')
        
        # Variables to store previous iteration results for feedback
        prev_response = None
        prev_csat = None
        prev_distance = None
        
        for iteration in range(ONE_SHOT_MAX_ITERATIONS):
            try:
                # For iterations > 0, append feedback to conversation history
                if iteration > 0 and prev_response is not None:
                    feedback_msg = f"""
**Previous Attempt Results (Iteration {iteration}):**
- Constraint Satisfaction Score: {prev_csat:.4f} (target: 0.0)
- Refinement Distance: {prev_distance:.4f}

Please try again with a different refinement that better satisfies the constraints while minimizing distance.
"""
                    messages.append({"role": "assistant", "content": prev_response})
                    messages.append({"role": "user", "content": feedback_msg})
                    self.assignment_lm.log_message("user", feedback_msg, f"One-Shot Feedback (Iteration {iteration + 1})")
                
                # Generate response from LLM
                response = self.assignment_lm.chat_model.generate(messages=messages)
                self.assignment_lm.log_message("assistant", response, f"One-Shot Response (Iteration {iteration + 1})")
                
                # Store response for next iteration's feedback
                prev_response = response
                
                # Parse response
                concise_reason, selected_sql, selected_values = self.assignment_lm.clean_reaponse(response)
                
                if not selected_sql:
                    print(f"Iteration {iteration + 1}: Failed to extract SQL from response")
                    prev_csat = float('inf')
                    prev_distance = float('inf')
                    continue
                
                # Evaluate the solution
                query = selected_sql
                consts = self.constraint_parser.get_results(query)
                csat = self.constraint_parser.evaluate_constraint_score(query)
                distance = float(self._calculate_refinement_distance(query))
                is_satisfied = self.constraint_parser.is_satisfied(query, self.epsilon)
                
                # Store for next iteration's feedback
                prev_csat = csat
                prev_distance = distance
                
                print(f"\nIteration {iteration + 1} Solution Evaluation:")
                print(f"Query: {query}")
                print(f"Constraint Satisfaction Score: {csat:.4f}")
                print(f"Distance: {distance:.4f}")
                print(f"Constraints Satisfied: {is_satisfied}")
                
                # Record the result
                pred_values_dict = selected_values or {}
                transformed_pred_values = {}
                for df_col_name in self.pred_columns:
                    found_val = self.subspace_radius_structure.axis_values.get(df_col_name, None)
                    for llm_key, mapped_df_col in self.llm_pid_to_column_name_map.items():
                        if mapped_df_col == df_col_name and llm_key in pred_values_dict:
                            found_val = pred_values_dict[llm_key]
                            break
                    transformed_pred_values[df_col_name] = found_val
                
                self.record_run(query, transformed_pred_values, consts, distance, f"One-shot iteration {iteration + 1}")
                
                # Track best satisfying query
                if is_satisfied and distance < best_satisfying_distance:
                    best_satisfying_query = query
                    best_satisfying_distance = distance
                    print(f"\n*** New best satisfying query found! Distance: {distance:.4f} ***")
                
                # Track best constraint score (for fallback if no satisfying query found)
                if csat < best_csat:
                    best_csat = csat
                    best_csat_query = query
                    best_csat_distance = distance
                
                # Update Q_star if this is the best satisfying solution
                if is_satisfied and distance < self.d_star:
                    self.Q_star = query
                    self.d_star = distance
                
            except Exception as e:
                # Assume context window exception - return best result found
                print(f"\nException at iteration {iteration + 1} (likely context window): {e}")
                print("Returning best result found so far...")
                break
        
        # Save refinements log
        self._save_refinements_log()
        total_tokens = self._get_acc_tokens(self.assignment_lm)
        
        # Return best satisfying query if found, otherwise best csat query
        if best_satisfying_query is not None:
            self.Q_star = best_satisfying_query
            self.d_star = best_satisfying_distance
            print(f"\n*** Returning best satisfying query with distance: {best_satisfying_distance:.4f} ***")
        elif best_csat_query is not None:
            self.Q_star = best_csat_query
            self.d_star = best_csat_distance
            print(f"\n*** No satisfying query found. Returning best csat query (csat={best_csat:.4f}, distance={best_csat_distance:.4f}) ***")
        
        return self.Q_star, self.d_star, total_tokens

    def run_refinement_loop(self) -> tuple[str | None, float, int]:
        """Runs the main refinement loop according to the algorithm."""

        # Check if one-shot mode is enabled
        if self.one_shot_mode:
            return self._run_one_shot()

        # Run preprocessing before starting the main refinement process
        # self._run_preprocessing()

        # Initial evaluation of the original query
        initial_constraint_eval_str = self.constraint_parser.evaluation_str(self.original_query)
        initial_is_satisfied = self.constraint_parser.is_satisfied(self.original_query, self.epsilon)
        
        # Log mode for clarity
        if self.subspace_lm_only_random:
            print("=" * 80)
            print("RUNNING IN CRITIC-ONLY RANDOM ABLATION MODE")
            print("Critic selects subspaces, AssignmentLM bypassed for random sampling")
            print("=" * 80)

        if self.use_history:
            self.history.add_query(
                query=self.original_query,
                constraint_score=initial_constraint_eval_str,
                refinement_distance=0.0,
                is_satisfied=initial_is_satisfied
            )

        orig_pred_values_num = {
            p.get_id(): p.value
            for p in self.subspace_radius_structure.original_predicates
            if isinstance(p, NumericalPredicate)
        }
        orig_pred_values_cat = {
            p.get_id(): p.values
            for p in self.subspace_radius_structure.original_predicates
            if isinstance(p, CategoricalPredicate)
        }

        orig_pred_values = {**orig_pred_values_num, **orig_pred_values_cat}

        orig_constraint_values = self.constraint_parser.get_results(self.original_query)
        self.record_run(self.original_query, orig_pred_values, orig_constraint_values, 0.0, "Initial query")
        constraint_deviation_score = self.constraint_parser.evaluate_constraint_score(self.original_query)

        print(f"Initial Query: {self.original_query}")
        print(f"Initial Constraint Eval: {initial_constraint_eval_str}")
        print(f"Initial Constraint Satisfied (Epsilon={self.epsilon}): {initial_is_satisfied}\n{'- ' *50}")

        # Ensure Skyline S and global history H are seeded with θ0 before any early return
        # Ensure Skyline epsilon is configured before inserting any records
        if self.use_skyline and self.skyline is not None:
            try:
                if hasattr(self.skyline, "epsilon"):
                    self.skyline.epsilon = float(self.epsilon)
            except Exception:
                pass
        # Seed Skyline and global history with θ0
        try:
            if self.use_history:
                self.history.add_query(
                    query=self.original_query,
                    constraint_score=initial_constraint_eval_str,
                    refinement_distance=0.0,
                    is_satisfied=initial_is_satisfied
                )
            self._insert_into_skyline(self.original_query, 0.0, constraint_deviation_score)
            if self.use_history and self.subspace_lm is not None:
                self.subspace_lm.update_history(self.original_query, orig_constraint_values, constraint_deviation_score, 0.0)
        except Exception:
            pass

        # Check if initial query already satisfies constraints
        if initial_is_satisfied:
            print("Initial query already satisfies constraints. Continuing to optimize distance.")
            self.Q_star = self.original_query
            self.d_star = 0.0
            self._best_satisfying_query = self.original_query
            # Switch Skyline to optimization mode and cleanup once
            self._switch_to_optimization_mode_once()

        # ================= Algorithm-faithful loop (Oracle + Skyline) =================

        def _best_query_json() -> Dict[str, Any]:
            current_q = None
            if self.use_skyline and self.skyline is not None:
                try:
                    current_q = self.skyline.get_best_query()
                except Exception:
                    current_q = None
            if current_q is None:
                current_q = self.Q_star or self.original_query
            current_d = 0.0 if current_q is None else self._calculate_refinement_distance(current_q)
            base_query = current_q if current_q is not None else self.original_query
            current_c = self.constraint_parser.get_results(base_query)
            return {
                "query": current_q,
                "distance": float(current_d),
                "constraints": current_c,
            }

        # Outer loop: t = 1..T
        for t in range(1, self.max_subspace_iters + 1):
            # Oracle-based subspace selection Θ_t ← Oracle(θ_{t-1}, H; S)
            best_q_json = _best_query_json()

            skyline_view = ""
            if self.use_skyline and self.skyline is not None:
                try:
                    skyline_view = self.skyline.get_md_table_view()
                except Exception:
                    skyline_view = ""
            else:
                skyline_view = "Skyline tracking disabled. Prioritize constraint satisfaction and distance minimization."

            if not self.assignment_lm_only_mode:
                try:
                    optimization_context = self._get_optimization_context(is_subspace_lm=True)
                    Theta_t_obj, subspace_concise = self.subspace_lm.get_subspace(skyline=skyline_view,
                                                                optimization_context=optimization_context)
                except Exception as e:
                    Theta_t_obj = None
                    subspace_concise = {}
            else:
                Theta_t_obj = None
                subspace_concise = "AssignmentLM-only mode: no subspace selection performed."


            # Normalize subspace payload for AssignmentLM (dict expected)
            if Theta_t_obj is None:
                Theta_t_obj = self._full_range_subspace_payload()
            if hasattr(Theta_t_obj, "model_dump"):
                subspace_payload = Theta_t_obj.model_dump()
            elif hasattr(Theta_t_obj, "dict"):
                subspace_payload = Theta_t_obj.dict()
            else:
                subspace_payload = Theta_t_obj  # assume dict-like
            # Clean one-liner for subspace (after SubspaceLM response)
            try:
                acc_tokens = self._get_acc_tokens(self.subspace_lm) if self.subspace_lm is not None else 0
                self._log_clean_line(
                    f"[Subspace] t={t} subspace={subspace_concise} acc_tokens_total={acc_tokens}"
                )
                print(f"[Subspace] t={t} subspace={subspace_concise} acc_tokens_total={acc_tokens}")
            except Exception:
                pass

            # Inner loop (sequential): generate exactly K candidates with progressive learning
            candidates = []  # list of dicts: {q, d, csat, consts, preds}
            K = int(self.max_refinements)
            early_stop_counter = 0
            for k in range(K):
                try:
                    if self.subspace_lm_only_random:
                        # Ablation mode: sample random assignment from selected subspace
                        assignment = self._sample_assignment_from_subspace(subspace_payload)
                        q, _ = self._build_sql_from_assignment(assignment)
                        Q_primes = [q] if q else []
                        pred_values_list = [assignment] if q else []
                        reasonings = ["random_sample"]
                    else:
                        # Normal mode: use AssignmentLM
                        optimization_context = self._get_optimization_context()
                        Q_primes, pred_values_list, reasonings = self.assignment_lm.refine_query(
                            subspace=subspace_payload,
                            skyline=skyline_view,
                            single_candidate=True,
                            optimization_context=optimization_context,
                        )
                except Exception as e:
                    Q_primes, pred_values_list, reasonings = [], [], []

                if not Q_primes:
                    continue

                q = Q_primes[0]
                q = q.replace('`', '"')  # normalize backticks to single quotes
                preds = pred_values_list[0] if pred_values_list else {}

                # # Basic structure validation
                # is_valid, info = self.basic_query_validator(q)
                # if not is_valid:
                #     continue

                raw_consts = self.constraint_parser.get_results(q)
                const_json = ast.literal_eval(self.constraints_json)

                consts = {c["id"]: raw_consts[c["constraint_str"]] for c in const_json}
                try:
                    csat = self.constraint_parser.evaluate_constraint_score(q)
                except Exception:
                    csat = 1.0
                d = float(self._calculate_refinement_distance(q))

                self._insert_into_skyline(q, d, csat)

                # Immediate local history update for progressive learning within subspace
                if self.use_history:
                    try:
                        self.assignment_lm.update_history(q, csat, d, constraint_values=consts)
                    except Exception:
                        pass

                # Collect candidate for winner selection
                candidates.append({"q": q, "d": d, "csat": csat, "consts": consts, "preds": preds})
                # Clean one-liner for each evaluated refinement (after we have csat + d)
                try:
                    acc_tokens_a = self._get_acc_tokens(self.assignment_lm)
                    q_to_print = q.replace('\n', ' ')
                    self._log_clean_line(
                        f"[Refinement] k={k} sql={q_to_print} csat={csat:.2f} d={d:.2f} acc_tokens_total={acc_tokens_a}"
                    )
                    print(f"[Refinement] k={k} sql={q_to_print} csat={csat:.2f} d={d:.2f} acc_tokens_total={acc_tokens_a}")
                except Exception:
                    pass


                # ######### Experimental: early stopping #########
                # if csat > self.epsilon:
                #     early_stop_counter += 1
                # else:
                #     early_stop_counter = 0
                # if early_stop_counter >= 3:
                #     print("Three consecutive non-satisfying candidates. Early stopping inner loop.")
                #     break
                # ##############################################

            # Snapshot skyline state after this outer iteration's inner loop
            try:
                if self.use_skyline and self.skyline is not None:
                    self._skyline_history.append(self._skyline_snapshot())
            except Exception:
                pass
            # Algorithm-1 style early stopping: skyline stability over horizon u
            try:
                horizon = getattr(self, "_skyline_stability_horizon", 0)
                if horizon > 0 and t > horizon and len(self._skyline_history) >= (horizon + 1):
                    if self._skyline_history[-1] == self._skyline_history[-1 - horizon]:
                        print(f"Skyline stable over {horizon} iterations. Stopping early.")
                        break
            except Exception:
                pass

            if not candidates:
                continue  # proceed to next outer iteration

            # Compute Improve(θ, S) and select θ_t = argmax Improve
            winner_sql = self._maximize_pareto([(c["q"], c["d"], c["csat"]) for c in candidates])
            if not winner_sql:
                # Fallback: smallest distance
                winner_sql = sorted(candidates, key=lambda x: (x["d"], x["csat"]))[0]["q"]

            winner = next((c for c in candidates if c["q"] == winner_sql), candidates[0])

            # Update global history H (and Skyline S union {θ_t})
            if self.use_history:
                try:
                    if self.subspace_lm is not None:
                        self.subspace_lm.update_history(winner["q"], winner["consts"], winner["csat"], winner["d"])
                except Exception:
                    pass

            # Record refinement in orchestrator structures and update Q* if satisfied
            improved_in_this_iter = self.record_refinement(winner["q"], winner["preds"])  # keeps memory_df in sync

            # Termination check and optimization-mode handling after outer iteration
            is_satisfied_iter = self.constraint_parser.is_satisfied(winner["q"], self.epsilon)
            if is_satisfied_iter:
                # Switch Skyline to optimization mode on first satisfying candidate
                self._switch_to_optimization_mode_once()
                # Update best satisfying query for optimization context
                if self._best_satisfying_query is None or winner["d"] < self.d_star:
                    self._best_satisfying_query = winner["q"]
                # Update global best satisfying query by distance

            # Early stopping: if in optimization mode, stop after N consecutive non-improving iterations
            if self._in_optimization_mode():
                if improved_in_this_iter:
                    self._no_improve_outer_iters = 0
                else:
                    self._no_improve_outer_iters += 1
                # Early stopping check
                if self._no_improve_outer_iters >= self.max_no_improve_outer_iters:
                    print("No distance improvement for consecutive iterations in optimization mode. Stopping early.")
                    break

        # After the algorithm-faithful loop, return current best (backward compatible)
        print("\n===== Refinement Process Finished (Algorithm-Faithful Loop) =====\n\n")
        # Ensure Skyline retains at least one record for observability
        if self.use_skyline and self.skyline is not None:
            try:
                s = self.skyline
                if hasattr(s, "data") and len(s.data) == 0:
                    seed_consts = self.constraint_parser.get_results(self.original_query)
                    self._insert_into_skyline(self.original_query, 0.0, 1.0)
                    if self.use_history and self.subspace_lm is not None:
                        self.subspace_lm.update_history(self.original_query, seed_consts, 1.0, 0.0)
            except Exception:
                pass
        self._save_refinements_log()
        # Get total accumulated tokens for AssignmentLM and SubspaceLM
        total_assignment_lm_tokens = self._get_acc_tokens(self.assignment_lm) if self.assignment_lm is not None else 0
        total_subspace_lm_tokens = self._get_acc_tokens(self.subspace_lm) if self.subspace_lm is not None else 0
        overall_acc_tokens = total_assignment_lm_tokens + total_subspace_lm_tokens
        print(f"Total accumulated tokens for AssignmentLM: {total_assignment_lm_tokens}")
        print(f"Total accumulated tokens for SubspaceLM: {total_subspace_lm_tokens}")
        print(f"Total accumulated tokens overall: {overall_acc_tokens}")

        return self.Q_star, self.d_star, overall_acc_tokens

    def run_opro(self, T: Optional[int] = None, K: Optional[int] = None, u: Optional[int] = None) -> tuple[str | None, float, int]:
        """
        Canonical Algorithm 1 execution: uses SubspaceLM + AssignmentLM with skyline-driven early stopping.
        Ignores ablation/one-shot flags to preserve the reference behavior.
        """
        # Preserve current settings
        orig_max_T = self.max_subspace_iters
        orig_max_K = self.max_refinements
        orig_u = getattr(self, "_skyline_stability_horizon", 0)
        orig_assignment_lm_only = self.assignment_lm_only_mode
        orig_subspace_lm_random = self.subspace_lm_only_random
        orig_one_shot = self.one_shot_mode
        try:
            if T is not None:
                self.max_subspace_iters = int(T)
            if K is not None:
                self.max_refinements = int(K)
            if u is not None:
                try:
                    self._skyline_stability_horizon = max(0, int(u))
                except Exception:
                    self._skyline_stability_horizon = orig_u
            # Disable ablation/one-shot modes for canonical run
            self.assignment_lm_only_mode = False
            self.subspace_lm_only_random = False
            self.one_shot_mode = False
            return self.run_refinement_loop()
        finally:
            # Restore original settings
            self.max_subspace_iters = orig_max_T
            self.max_refinements = orig_max_K
            self._skyline_stability_horizon = orig_u
            self.assignment_lm_only_mode = orig_assignment_lm_only
            self.subspace_lm_only_random = orig_subspace_lm_random
            self.one_shot_mode = orig_one_shot

    def _save_refinements_log(self):
        """Save the consolidated refinements history to refinements.log"""
        # Format the DataFrame as a string for logging
        refinements_log_str = format_data_for_log(self.subspace_memory_df)

        # Save to refinements.log
        refinements_log_path = os.path.join(self.log_dir, "refinements.log")
        log_to_file(refinements_log_path, refinements_log_str, mode='w')

        print(f"Saved consolidated refinements log to {refinements_log_path}")

    def record_refinement(self, Q_prime, pred_values) -> bool:
        """
        Record a valid refinement and evaluate it.

        Args:
            Q_prime: The proposed query
            pred_values: Values for the predicates in the query

        Returns:
            bool: Whether Q_star was updated by this refinement
        """
        # Save the current Q_star and d_star to check if they get updated
        old_Q_star = self.Q_star
        old_d_star = self.d_star

        # Evaluate constraints and distance
        C_Q_prime_eval = self.constraint_parser.evaluation_str(Q_prime)
        is_satisfied = self.constraint_parser.is_satisfied(Q_prime, self.epsilon)
        R_Q_prime = self._calculate_refinement_distance(Q_prime)
        print(f"Refinement Evaluation for Query:\n{Q_prime}\n")
        print(f"Constraint Eval: {C_Q_prime_eval}")
        print(f"Satisfied (Epsilon={self.epsilon}): {is_satisfied}")
        print(f"Refinement Distance: {R_Q_prime}")

        # Record this attempt
        if self.use_history:
            self.history.add_query(
                query=Q_prime,
                constraint_score=C_Q_prime_eval,
                refinement_distance=R_Q_prime,
                is_satisfied=is_satisfied
            )

        const_results = self.constraint_parser.get_results(Q_prime)

        # Transform pred_values from LLM format (using P1, P2 or attribute_name) to DataFrame column format
        transformed_pred_values = {}
        for df_col_name in self.pred_columns: # Iterate over expected columns in memory_df
            # Set the default value to the original predicate value
            found_val = self.subspace_radius_structure.axis_values.get(df_col_name, None)
            # Try to find the value using the mappings, prioritizing llm_pid, then attribute_name, then df_col_name itself
            # This order helps resolve if LLM sometimes returns attribute_name or df_col_name instead of P1/P2
            for llm_key, mapped_df_col in self.llm_pid_to_column_name_map.items():
                if mapped_df_col == df_col_name and llm_key in pred_values:
                    found_val = pred_values[llm_key]
                    break # Found the value for this df_col_name
            transformed_pred_values[df_col_name] = found_val

        # Record in memory_df using transformed predicate values
        self.record_run(Q_prime, transformed_pred_values, const_results, R_Q_prime, "")

        # Check if it's the best satisfying query so far
        if is_satisfied:
            if R_Q_prime < self.d_star:
                print(f"*** New best satisfying query found! Distance: {R_Q_prime} < {self.d_star} ***")
                self.Q_star, self.d_star = Q_prime, R_Q_prime

                # Check if Q_star was updated
                return True

        print(f"\n===== Refinement Loop Finished =====")
        # Check if Q_star was updated (which would only happen if is_satisfied and R_Q_prime < self.d_star)
        return (self.Q_star != old_Q_star or self.d_star != old_d_star)


def run_task(task: Union[Task, AgnosticTask],
             epsilon: float,
             perform_analysis=True,
             perform_optimization=True,
             max_assignments_per_subspace=3,
             max_subspace_iters: int = 5,
             assignment_lm_only_mode: bool = False,
             subspace_lm_only_random: bool = False,
             random_seed: Optional[int] = None,
             one_shot_mode: bool = False,
             is_having: bool =False,
             use_history: bool = True,
             log_dir: Optional[str] = None,
             use_skyline: bool = True,
             model_provider: Optional[str] = None,
             model_name: Optional[str] = None
             ) -> Tuple[str, float, int]:
    """
    Run a refinement task with the given parameters.

    Args:
        task: The refinement task to run
        epsilon: Threshold for constraint satisfaction
        perform_analysis: Whether to perform data analysis during refinement
        perform_optimization: Whether to perform post-satisfaction query optimization
        max_assignments_per_subspace: Maximum number of assignments per subspace
        max_subspace_iters: Maximum number of subspace iterations
        assignment_lm_only_mode: If True, runs without the Critic/subspace selection mechanism.
        subspace_lm_only_random: If True, uses Critic for subspace selection but samples random assignments (ablation mode).
        random_seed: Optional random seed for reproducible random sampling in subspace_lm_only_random mode.
        one_shot_mode: If True, asks LLM for a single solution from the entire search space without iterations.
        use_history: Toggle summarized history usage for AssignmentLM/SubspaceLM prompts.
        use_skyline: Toggle skyline-based candidate tracking and prompts.
    Returns:
        Tuple of (best_query, best_distance, total_tokens)
    """
    dataset = task.df
    constraints = task.output_constraints
    parse_constraints = False  # True if isinstance(task, AgnosticTask) else False

    refineable_predicates = task.refineable_predicates
    refineable_predicates_str_num = "\n".join(
        f"- {p.attribute.name} {p.operator} <refined_{p.attribute.name}_{'min' if '>' in p.operator else 'max'}>"
        for p in refineable_predicates if isinstance(p, NumericalPredicate)
    )
    refineable_predicates_str_cat = "\n".join(
        f"- {p.attribute.name} IN <refined_{p.attribute.name}>"
        for p in refineable_predicates if isinstance(p, CategoricalPredicate)
    )

    refineable_predicates_str = f"{refineable_predicates_str_num}\n{refineable_predicates_str_cat}"
    # Initialize the Orchestrator with all necessary parameters
    orchestrator = OmniTuneEngine(
        task_name=task.name,
        original_query=task.original_query,
        input_dataset=dataset,
        constraints=constraints,
        epsilon=epsilon,
        distance_func=task.refinement_objective,
        refineable_predicates_str=refineable_predicates_str,
        refineable_predicates=refineable_predicates,
        perform_analysis=perform_analysis,
        perform_optimization=perform_optimization,
        max_refinements=max_assignments_per_subspace,
        max_subspace_iters=max_subspace_iters if not one_shot_mode else 1,
        parse_constraints=parse_constraints,
        assignment_lm_only_mode=assignment_lm_only_mode,
        is_range_task=isinstance(task, RangeQueryRefinementTask),
        subspace_lm_only_random=subspace_lm_only_random,
        random_seed=random_seed,
        one_shot_mode=one_shot_mode,
        is_having_query=is_having,
        log_dir=log_dir,
        use_history=use_history,
        use_skyline=use_skyline,
        model_provider=model_provider,
        model_name=model_name
    )

    # Run the refinement loop
    best_query, best_distance, overall_token_use = orchestrator.run_refinement_loop()
    return best_query, best_distance, overall_token_use


# Backward compatibility alias
RefinementOrchestrator = OmniTuneEngine


if __name__ == "__main__":
    # Example Usage (similar to original file)
    ORIGINAL_QUERY = """
SELECT region, AVG(charges), COUNT(*) FROM df
WHERE age >= 45 AND bmi >= 38
GROUP BY region;
"""
    # Load the dataset (adjust path as needed)
    try:
        DATASET = pd.read_csv("../../data/top_k_refinement/medical_insurance.csv")
    except FileNotFoundError:
        print("Error: Dataset file not found. Please adjust the path.")
        DATASET = pd.DataFrame() # Use empty dataframe to avoid crashing

    CONSTRAINTS = """1. Maximum average charges should be at least 25,000
2. Standard deviation between result's region sizes should be at most 25
3. All four regions must be included in the result
"""

    bmi_attribute = NumericalAttribute("bmi", 16.0, 53.0, 0.5)
    age_attribute = NumericalAttribute("age", 18, 64, 1)

    REFINEABLE_PREDICATES = [NumericalPredicate(age_attribute, ">=", 45),
                             NumericalPredicate(bmi_attribute, ">=", 38.0)]

    REFINEABLE_PREDICATES_STR = "- age >= <refined_age>\n- bmi >= <refined_bmi>"

    TASK_NAME = "insurance_refinement_2025_04_24"

    orchestrator = OmniTuneEngine(
        task_name=TASK_NAME,
        original_query=ORIGINAL_QUERY,
        input_dataset=DATASET,
        constraints=CONSTRAINTS,
        epsilon=0.0,
        distance_func=get_script_diff_func_sql(ORIGINAL_QUERY, DATASET),
        refineable_predicates_str=REFINEABLE_PREDICATES_STR,
        refineable_predicates=REFINEABLE_PREDICATES,
        max_subspace_iters=5,  # Example: Limit iterations for testing
        max_refinements=3,  # Limit refinements per subspace for testing
        perform_optimization=True,  # Enable query optimization
        max_assignments_per_subspace=2,  # Enable retry mechanism for invalid queries
        assignment_lm_only_mode=False, # Set to True for ablation test
    )

    # Run the loop
    if not DATASET.empty:
        best_query, best_distance, overall_token_use = orchestrator.run_refinement_loop()
    else:
        print("Skipping refinement loop due to dataset loading error.")
