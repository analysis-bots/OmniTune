import os
import re
import json
from typing import Any, List, Dict, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd

from pydantic import BaseModel, ValidationError, root_validator, validator, model_validator, field_validator

from functionality.skyline import Skyline
from functionality.constraint import AgnosticConstraint
from tools.logging_utils import ensure_log_directory, log_to_file

from chat_model import get_chat_model, ChatModel
from opro.opro_prompt_templates import (
    ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE,
    ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE,
    SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE,
    SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE,
    SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE,
    ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE,
    SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_HISTORY,
    ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_HISTORY,
    ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE,
    ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY,
    SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE,
    SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY,
)
from config import LOG_DIR
from config import ENABLE_AGENT_ATTRIBUTE_STATS, STATS_SAMPLE_SIZE
from tools.attribute_stats import AttributeStatsGenerator
from functionality.predicate import NumericalPredicate, CategoricalPredicate, NumericalAttribute, Predicate
from functionality.subspace_data_structure import SubspaceStructure, SubspaceDict
from tools.utils import subsets_containing, get_primary_dataframe, build_dataset_schema_context


class SubspaceLLMResponse(BaseModel):
    """Structured validator for SubspaceLM responses."""

    selected_predicate_subspace: Dict[str, Any]
    reasoning_concise: Optional[str] = None

    @field_validator("selected_predicate_subspace", mode="before")
    def _validate_subspace(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("selected_predicate_subspace must be an object")
        cleaned: Dict[str, Any] = {}
        for pid, val in v.items():
            # Accept numeric range: [min, max] or {"min": x, "max": y}
            if isinstance(val, dict) and {"min", "max"} <= set(val.keys()):
                lo, hi = val.get("min"), val.get("max")
                cleaned[pid] = cls._validate_numeric_range(lo, hi)
            elif isinstance(val, (list, tuple)):
                if len(val) != 2:
                    raise ValueError(f"{pid} must have exactly two elements")
                # Numeric range
                if all(isinstance(x, (int, float)) for x in val):
                    cleaned[pid] = cls._validate_numeric_range(val[0], val[1])
                # Categorical range: [[contained], [containing]]
                elif all(isinstance(x, (list, tuple)) for x in val):
                    contained_raw, containing_raw = val
                    contained = [str(x) for x in contained_raw]
                    containing = [str(x) for x in containing_raw]
                    # Ensure contained is a subset of containing
                    if not set(contained).issubset(set(containing)):
                        raise ValueError(f"{pid} contained values must be subset of containing values")
                    cleaned[pid] = [contained, containing]
                else:
                    raise ValueError(f"{pid} has unsupported structure")
            else:
                raise ValueError(f"{pid} must be a numeric range or categorical range")
        return cleaned

    @staticmethod
    def _validate_numeric_range(lo: Any, hi: Any) -> List[float]:
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception as exc:
            raise ValueError(f"Numeric range values must be numeric: {exc}")
        if lo_f > hi_f:
            raise ValueError("Numeric range min must be <= max")
        return [lo_f, hi_f]


class AssignmentLLMResponse(BaseModel):
    """Structured validator for AssignmentLM responses."""

    selected_predicate_values: Dict[str, Any] = {}
    selected_refinement: Optional[str] = None
    reasoning_concise: Optional[str] = None

    @field_validator("selected_predicate_values", mode="before")
    def _validate_predicate_values(cls, v: Any) -> Dict[str, Any]:
        if v in (None, "", {}):
            return {}
        if not isinstance(v, dict):
            raise ValueError("selected_predicate_values must be an object")
        cleaned: Dict[str, Any] = {}
        for pid, val in v.items():
            # Categorical -> list/tuple of values
            if isinstance(val, (list, tuple)):
                cleaned[pid] = [str(x) for x in val]
                continue
            # Numeric -> coerce to float if possible
            try:
                cleaned[pid] = float(val)
            except Exception:
                # Fallback to string if numeric coercion fails
                cleaned[pid] = str(val)
        return cleaned

    @model_validator(mode="after")
    def _ensure_refinement_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # selected_refinement can be None during parsing; leave as-is to allow legacy fallback paths.
        return values


class BaseLLMAgent:
    def __init__(
        self,
        task_name: str,
        system_prompt: str,
        log_dir: str = LOG_DIR,
        use_history: bool = True,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.chat_model: ChatModel = get_chat_model(model_provider=model_provider, model_name=model_name)
        self.messages: List[Dict[str, str]] = []
        self.task_name = task_name
        # Extract the base task name without "_assignment_lm" or "_subspace_lm" suffix
        self.base_task_name = task_name.split("_")[0] if "_" in task_name else task_name
        self.agent_type = "assignment_lm" if "_assignment_lm" in task_name else "subspace_lm" if "_subspace_lm" in task_name else "agent"
        # Ensure the task-specific log directory exists
        self.log_dir = ensure_log_directory(log_dir, self.base_task_name)
        self.use_history = use_history
        # Track accumulated token usage across the agent's lifetime
        self.accumulated_token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.append_message("system", system_prompt, "System Prompt")

    def _get_token_usage(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Return (last_usage, accumulated_usage) dictionaries."""
        last_usage = getattr(self.chat_model, "last_usage", None) or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return last_usage, self.accumulated_token_usage

    def _log_clean_line(self, line: str):
        """Write a single concise line to the clean responses log."""
        clean_log_file = os.path.join(self.log_dir, "clean_responses.log")
        try:
            log_to_file(clean_log_file, line.strip().replace("\n", " ") + "\n")
        except Exception:
            pass

    def append_message(self, role: str, content: str, header_description: str = ""):
        """
        Append a message to the conversation history and log it with proper formatting.
        
        Args:
            role: The role of the message sender ("system", "user", or "assistant")
            content: The content of the message
            header_description: A description of the message for the log header
            log_message: Whether to log the message (default True)
        """
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        self.log_message(role, content, header_description)

    def _prepare_message_context(self, prompt: str) -> Tuple[List[Dict[str, str]], Optional[int]]:
        if self.use_history:
            return self.messages + [{"role": "user", "content": prompt}], None
        self.messages.append({"role": "user", "content": prompt})
        return self.messages, len(self.messages) - 1

    def _finalize_assistant_message(self, response: str) -> None:
        if not self.use_history:
            self.messages.append({"role": "assistant", "content": response})

    def _revert_user_message(self, index: Optional[int]) -> None:
        if self.use_history or index is None:
            return
        if index == len(self.messages) - 1:
            self.messages.pop()

    def _reset_to_system_prompt(self) -> None:
        if self.use_history:
            return
        system_msg = next((msg for msg in self.messages if msg.get("role") == "system"), None)
        self.messages = [system_msg] if system_msg else []

    @staticmethod
    def _is_context_overflow_error(error: Exception) -> bool:
        message = str(error).lower()
        keywords = [
            "context length",
            "maximum context",
            "context window",
            "token limit",
            "max tokens",
            "too many tokens",
        ]
        return any(keyword in message for keyword in keywords)

    def log_message(self, role, content, header_description):
        # Format the log entry
        log_role = "USER" if role in ["system", "user"] else self.agent_type.upper()
        header = f"### {log_role} - {header_description}"
        # Include token accounting when assistant responds
        token_log = ""
        if role == "assistant":
            try:
                usage = getattr(self.chat_model, "last_usage", None)
                if usage and isinstance(usage, dict):
                    pt = int(usage.get("prompt_tokens", 0) or 0)
                    ct = int(usage.get("completion_tokens", 0) or 0)
                    tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))
                    # Update accumulated counters
                    self.accumulated_token_usage["prompt_tokens"] += pt
                    self.accumulated_token_usage["completion_tokens"] += ct
                    self.accumulated_token_usage["total_tokens"] += tt
                    token_log = (
                        f"\nToken Usage (this call): prompt={pt}, completion={ct}, total={tt}\n"
                        f"Token Usage (accumulated): "
                        f"prompt={self.accumulated_token_usage['prompt_tokens']}, "
                        f"completion={self.accumulated_token_usage['completion_tokens']}, "
                        f"total={self.accumulated_token_usage['total_tokens']}\n"
                    )
            except Exception:
                token_log = ""
        log_entry = f"{header}\n{content}{token_log}\n---\n"
        # Log to the appropriate file based on agent type
        log_file = os.path.join(self.log_dir, f"{self.agent_type}.log")
        log_to_file(log_file, log_entry)
        # Also log to the conversation file for backward compatibility
        conv_log_file = os.path.join(self.log_dir, "_conversation.log")
        # Add token usage to conversation log for assistant entries
        if role == "assistant" and token_log:
            log_to_file(conv_log_file, f"{log_role}: {content}{token_log}\n{'-' * 50}\n")
        else:
            log_to_file(conv_log_file, f"{log_role}: {content}\n{'-' * 50}\n")
        # Unified dirty log: all prompts + responses with tokens
        dirty_log_file = os.path.join(self.log_dir, "dirty.log")
        log_to_file(dirty_log_file, log_entry)
        # Log only responses from the assistant (assignment_lm/subspace_lm) to a separate file
        if role == "assistant":
            response_log_file = os.path.join(self.log_dir, f"responses.log")
            # Include token usage in responses-only log
            log_to_file(response_log_file, f"{header}\n{content}{token_log}\n\n---\n")


class AssignmentLM(BaseLLMAgent):
    def __init__(
        self,
        task_name: str,
        constraints_json: str,
        input_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        original_query: str,
        epsilon: float,
        refineable_predicates: List[Predicate],
        refinement_objective_description: str,
        log_dir: str = LOG_DIR,
        use_history: bool = True,
        use_skyline: bool = True,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # Generate static context information for system prompt
        dataset_schema = build_dataset_schema_context(input_dataset)
        dataset_schema_context_json = json.dumps(dataset_schema)
        
        refineable_predicates_context_json = json.dumps([
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
        
        # Optionally compute attribute intelligence
        attribute_intelligence_context_json = "{}"
        try:
            if ENABLE_AGENT_ATTRIBUTE_STATS:
                stats_gen = AttributeStatsGenerator(input_dataset, refineable_predicates, sample_size=STATS_SAMPLE_SIZE)
                attribute_intelligence_context_json = json.dumps({
                    "attribute_intelligence": stats_gen.generate_stats()
                })
        except Exception:
            attribute_intelligence_context_json = "{}"

        # Format the system prompt with static context
        prompt_template = ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE if use_skyline else ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE
        formatted_system_prompt = prompt_template.format(
            original_query_context=original_query.strip(),
            refineable_predicates_context_json=refineable_predicates_context_json,
            output_constraints_context_json=constraints_json,
            dataset_schema_context_json=dataset_schema_context_json,
            refinement_objective_description=refinement_objective_description,
            attribute_intelligence_context_json=attribute_intelligence_context_json
        )
        
        super().__init__(task_name, formatted_system_prompt, log_dir, use_history=use_history, model_provider=model_provider, model_name=model_name)
        
        # Store static information as instance variables for internal use
        self.constraints_json = constraints_json
        self.constraint_columns: List[str] = []
        try:
            parsed_constraints = json.loads(constraints_json) if constraints_json else []
            if isinstance(parsed_constraints, list):
                for idx, constraint in enumerate(parsed_constraints):
                    cid = constraint.get("id") or constraint.get("constraint_id")
                    if cid is not None:
                        self.constraint_columns.append(str(cid))
        except (TypeError, ValueError):
            self.constraint_columns = []
        # Keep a primary df for stats and sampling, but allow dict schema
        self.input_dataset = get_primary_dataframe(input_dataset)
        self.original_query = original_query
        self.epsilon = epsilon
        self.refineable_predicates = refineable_predicates
        self.local_history = defaultdict(list)  # h_t in the algorithm inner loop
        self.local_history["constraint_columns"] = list(self.constraint_columns)
        self.use_skyline = use_skyline

        # Map predicate ids (P1, P2, ...) to metadata for validation per Preliminary Definitions
        # Numerical -> single number value; Categorical -> list of categories
        self.predicate_id_to_info: Dict[str, Dict[str, Any]] = {}
        for i, p in enumerate(refineable_predicates):
            pid = p.get_id()
            if isinstance(p, NumericalPredicate):
                self.predicate_id_to_info[pid] = {
                    "type": "numerical",
                    "attribute_name": p.attribute.name,
                    "operator": p.operator,
                }
            else:
                self.predicate_id_to_info[pid] = {
                    "type": "categorical",
                    "attribute_name": p.attribute.name,
                    "operator": "IN",
                }
        self._allowed_predicate_ids = set(self.predicate_id_to_info.keys())

    @staticmethod
    def _md_table_from_local_history(local_history: Dict[str, Any]) -> str:
        try:
            records = local_history.get("attempt_records", []) or []
            if not records:
                queries = local_history.get("queries", []) or []
                distances = local_history.get("refinement_distance", []) or []
                metrics = local_history.get("constraint_satisfaction", []) or []
                constraint_values_seq = local_history.get("constraint_values", []) or []
                satisfied_seq = local_history.get("satisfied_constraints", []) or []
                derived_records: List[Dict[str, Any]] = []
                max_len = min(len(queries), len(distances))
                for idx in range(max_len):
                    record: Dict[str, Any] = {
                        "query": queries[idx],
                        "distance": distances[idx],
                    }
                    if idx < len(metrics) and metrics[idx] is not None:
                        record["constraint_metric"] = metrics[idx]
                    if idx < len(constraint_values_seq):
                        record["constraint_values"] = constraint_values_seq[idx]
                    if idx < len(satisfied_seq):
                        record["satisfied_constraints"] = satisfied_seq[idx]
                    derived_records.append(record)
                records = derived_records

            constraint_columns = list(local_history.get("constraint_columns") or [])
            if not constraint_columns:
                seen = set()
                for rec in records:
                    for key in (rec.get("constraint_values") or {}):
                        if key not in seen:
                            seen.add(key)
                            constraint_columns.append(key)
                constraint_columns.sort()

            header_cells = ["Attempt", "Query"] + constraint_columns + ["Constraint Δ", "Satisfied Constraints", "Distance"]
            align_cells = []
            for name in header_cells:
                if name in {"Attempt", "Constraint Δ", "Distance"}:
                    align_cells.append("---:")
                else:
                    align_cells.append("---")
            header = "| " + " | ".join(header_cells) + " |\n"
            header += "| " + " | ".join(align_cells) + " |\n"

            rows: List[str] = []
            display_records = records[-5:] if records else []
            for idx, rec in enumerate(display_records, 1):
                row_cells: List[str] = [
                    str(idx),
                    rec.get("query", ""),
                ]
                values = rec.get("constraint_values") or {}
                legacy_constraints = rec.get("constraints", {})
                if not values and isinstance(legacy_constraints, dict):
                    values = legacy_constraints
                for col in constraint_columns:
                    val = values.get(col, "—")
                    if isinstance(val, float):
                        val = round(val, 3)
                    row_cells.append(str(val))
                metric = rec.get("constraint_metric")
                row_cells.append("—" if metric is None else str(round(float(metric), 3)))
                satisfied_flag = rec.get("satisfied_constraints", False)
                row_cells.append("True" if satisfied_flag else "False")
                distance = rec.get("distance")
                if isinstance(distance, float):
                    distance = round(distance, 3)
                row_cells.append(str(distance) if distance is not None else "—")
                rows.append("| " + " | ".join(row_cells) + " |")

            if not rows:
                placeholder = ["—"] * len(header_cells)
                rows.append("| " + " | ".join(placeholder) + " |")
            body = "\n".join(rows)
            return header + body
        except Exception:
            fallback_header = "| Attempt | Query | Constraint Δ | Satisfied Constraints | Distance |\n"
            fallback_header += "|---:|---|---:|---|---:|\n"
            fallback_body = "| — | — | — | — | — |"
            return fallback_header + fallback_body

    @staticmethod
    def _clamp_numeric(numeric_region, pid: str, val: Any) -> Optional[float]:
        if pid not in numeric_region:
            # If region not specified, accept as-is if numeric
            try:
                return float(val)
            except Exception:
                return None
        rng = numeric_region.get(pid)
        try:
            v = float(val)
        except Exception:
            return None
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                lo = float(rng[0])
                hi = float(rng[1])
                return min(max(v, lo), hi)
            except Exception:
                return v
        return v

    @staticmethod
    def _project_categorical(categorical_region, pid: str, vals: Any) -> Optional[List[str]]:
        # Ensure vals is list of strings, obey contained ⊆ values ⊆ containing
        try:
            proposed = [str(x) for x in (vals or [])]
        except Exception:
            return None
        if pid not in categorical_region:
            # No region: accept unique values
            return sorted(list(dict.fromkeys(proposed)))
        region_val = categorical_region.get(pid)
        if not (isinstance(region_val, (list, tuple)) and len(region_val) == 2):
            return sorted(list(dict.fromkeys(proposed)))
        contained, containing = region_val
        contained_set = set([str(x) for x in (contained or [])])
        containing_set = set([str(x) for x in (containing or [])])
        # Project: enforce containing and add contained
        projected = (set(proposed) & containing_set) | contained_set
        return sorted(list(projected))

    def refine_query(
        self,
        subspace: Dict[str, Any],
        skyline: str,
        single_candidate: bool = False,
        optimization_context: Optional[str] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Generate refined SQL queries based on the analysis results and subspace.
        
        Args:
            subspace: Current subspace representation
            optimization_context: Optional context about optimization mode and best satisfying query

        Returns:
            Tuple of (queries, predicate_values, reasonings)
        """
        # Format the prompt based on the inputs (static info now in system prompt)
        # Reformat local history into a compact Markdown table for the LLM

        refinement_history_md = self._md_table_from_local_history(self.local_history) if self.use_history else ""
        if self.use_skyline:
            prompt_template = (
                ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE
                if self.use_history
                else ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_HISTORY
            )
        else:
            prompt_template = (
                ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE
                if self.use_history
                else ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY
            )
        prompt = prompt_template.format(
            current_subspace_json=json.dumps(subspace),
            refinement_history_json=refinement_history_md,
            query_skyline=skyline
        )
        
        # Append optimization context if in optimization mode
        if optimization_context:
            prompt += f"\n\n{optimization_context}"

        prompt += "\n\n🎯 **YOUR WINNING MOVE:**\n"
        self.log_message("user", prompt, "Refine Query Request")
        messages, user_index = self._prepare_message_context(prompt)
        try:
            response = self.chat_model.generate(messages=messages)
        except Exception as e:
            if not self.use_history and self._is_context_overflow_error(e):
                self._log_clean_line("[ContextReset] Actor context window exceeded. Resetting conversation.")
                self._reset_to_system_prompt()
                messages, user_index = self._prepare_message_context(prompt)
                try:
                    response = self.chat_model.generate(messages=messages)
                except Exception as retry_error:
                    self._revert_user_message(user_index)
                    raise retry_error
            else:
                self._revert_user_message(user_index)
                raise
        if not response.strip().endswith('```'):
            response = response.strip().split(',\n    "reasoning_concise":')[0] + '\n}\n```'
        self.log_message("assistant", response, "Refine Query Response")
        self._finalize_assistant_message(response)
        # Parse primary JSON block expected by ACTOR_SUGGEST_REFINEMENT_PROMPT_TEMPLATE
        concise_reason, selected_sql, selected_values = self.clean_reaponse(response)
        # Enforce Preliminary Definitions: only refine known predicates and ensure values lie in the provided subspace
        # Subspace may come as {numeric_predicates: {Pid: [min,max]}, categorical_predicates: {Pid: [[contained],[containing]]}}
        numeric_region = {}
        categorical_region = {}
        if isinstance(subspace, dict):
            # Flexible extraction
            numeric_region = subspace.get("numeric_predicates", {}) or subspace.get("numerical_predicates", {}) or {}
            categorical_region = subspace.get("categorical_predicates", {}) or {}

        # Filter and validate selected predicate values per definitions
        validated_selected_values: Dict[str, Any] = {}
        for pid, val in (selected_values or {}).items():
            if pid not in self._allowed_predicate_ids:
                continue
            meta = self.predicate_id_to_info.get(pid, {})
            if meta.get("type") == "numerical":
                v = self._clamp_numeric(numeric_region, pid, val)
                if v is not None:
                    validated_selected_values[pid] = v
            else:
                vlist = self._project_categorical(categorical_region, pid, val)
                if vlist is not None:
                    validated_selected_values[pid] = vlist

        # Finalize outputs in the expected return format
        queries: List[str] = []
        if selected_sql:
            queries.append(selected_sql.strip())

        refined_pred_values_list: List[Dict[str, Any]] = []
        if validated_selected_values:
            refined_pred_values_list.append(validated_selected_values)

        reasonings: List[str] = []
        if concise_reason:
            reasonings.append(str(concise_reason))

        # If primary parsing failed to produce outputs, fall back to legacy extraction
        if not queries or not refined_pred_values_list or not reasonings:
            q2, p2, r2 = self._process_assignment_lm_response(response)
            if not queries:
                queries = q2
            if not refined_pred_values_list:
                refined_pred_values_list = p2
            if not reasonings:
                reasonings = r2

        # Support returning a single candidate when requested to enable sequential refinement
        if single_candidate:
            return queries[:1], refined_pred_values_list[:1], reasonings[:1]

        return queries, refined_pred_values_list, reasonings

    def clean_reaponse(self, response):
        selected_values: Dict[str, Any] = {}
        selected_sql: Optional[str] = None
        concise_reason: Optional[str] = None
        json_candidates: List[str] = []
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            json_candidates.append(json_match.group(1))
        if "```json" in response and not json_candidates:
            json_candidates.append(response.strip("```json"))
        if not json_candidates:
            json_candidates.append(response)

        # Try structured validation first
        for cand in json_candidates:
            try:
                parsed_json = json.loads(cand)
            except Exception:
                continue
            try:
                validated = AssignmentLLMResponse.parse_obj(parsed_json)
                selected_values = validated.selected_predicate_values or {}
                selected_sql = validated.selected_refinement
                concise_reason = validated.reasoning_concise
                break
            except ValidationError as ve:
                self._log_clean_line(f"[AssignmentValidationFailed] {ve}")
                continue

        # Fallbacks for SQL and reasoning from fenced blocks
        if not selected_sql:
            sql_blocks = re.findall(r"```sql(.*?)```", response, re.DOTALL)
            if sql_blocks:
                selected_sql = sql_blocks[0].strip()
        return concise_reason, selected_sql, selected_values

    def update_history(
        self,
        query: str,
        constraint_satisfaction: Union[float, Dict[str, Any], None],
        refinement_distance: float,
        constraint_values: Optional[Dict[str, Any]] = None,
    ):
        """Update the local history h_t with evaluated candidate (Ψ_k, Δ_k) per algorithm inner loop."""
        if not self.use_history:
            return

        sanitized_query = query.replace('\n', ' ')
        constraint_metric: Optional[float] = None
        constraint_values_dict: Dict[str, Any] = {}

        if isinstance(constraint_satisfaction, dict):
            # When the caller supplies a dict in the legacy parameter, treat it as the values mapping
            constraint_values_dict = dict(constraint_satisfaction)
        elif constraint_satisfaction is not None:
            try:
                constraint_metric = round(float(constraint_satisfaction), 3)
            except (TypeError, ValueError):
                constraint_metric = None

        if constraint_values:
            constraint_values_dict = dict(constraint_values)
        if constraint_values_dict:
            self.local_history["constraint_columns"] = list(self.constraint_columns)

        satisfied_constraints = bool(constraint_metric is not None and constraint_metric <= self.epsilon)

        # Maintain backward-compatible lists
        self.local_history["queries"].append(sanitized_query)
        self.local_history["queries"].append(sanitized_query)
        if constraint_metric is not None:
            self.local_history["constraint_satisfaction"].append(round(constraint_metric, 3))
        else:
            self.local_history["constraint_satisfaction"].append(None)
        self.local_history["refinement_distance"].append(round(float(refinement_distance), 3))
        if constraint_values_dict:
            self.local_history.setdefault("constraint_values", [])
            self.local_history["constraint_values"].append(constraint_values_dict)
        self.local_history.setdefault("satisfied_constraints", [])
        self.local_history["satisfied_constraints"].append(satisfied_constraints)

        # Structured attempt records (useful for the Oracle prompt and later aggregation)
        self.local_history.setdefault("attempt_records", [])
        attempt_record: Dict[str, Any] = {
            "query": sanitized_query,
            "distance": round(float(refinement_distance), 3),
        }
        if constraint_metric is not None:
            attempt_record["constraint_metric"] = constraint_metric
        if constraint_values_dict:
            attempt_record["constraint_values"] = constraint_values_dict
            # Preserve backwards compatibility for older log readers
            attempt_record["constraints"] = constraint_values_dict
        elif constraint_metric is not None:
            attempt_record["constraints"] = constraint_metric
        attempt_record["satisfied_constraints"] = satisfied_constraints
        self.local_history["attempt_records"].append(attempt_record)

    def clear_local_history(self):
        # Reset h_t at the start of each outer iteration t
        self.local_history = defaultdict(list)
        self.local_history["constraint_columns"] = list(self.constraint_columns)

    def _format_invalidation_feedback(self, invalidation_feedback: Optional[str]) -> str:
        """Format invalidation feedback for inclusion in the prompt."""
        if not invalidation_feedback or not invalidation_feedback.strip():
            return ""
            
        return f"""**IMPORTANT GUIDANCE FOR QUERY VALIDITY:**
Some of your previous query suggestions were structurally invalid. Please review the following feedback carefully and ensure your new suggestions strictly adhere to the required SQL structure and predicate usage:

{invalidation_feedback}

Avoid repeating these structural mistakes. Ensure all predicates are on refineable attributes and use allowed operators.
"""

    def _process_assignment_lm_response(self, response: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Extract queries, predicate values, and reasonings from the assignment_lm's response."""
        queries = re.findall(r'```sql(.*?)```', response, re.DOTALL)
        refined_predicates_jsons_str = re.findall(r'```json(.*?)```', response, re.DOTALL)
        
        # Parse the JSON objects
        refined_predicates_jsons = []
        for json_str in refined_predicates_jsons_str:
            try:
                refined_predicates_jsons.append(json.loads(json_str))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}. JSON string: {json_str}")
                return [], [], []

        # Extract predicate values with fallback for different formats
        refined_pred_values = []
        for refined_json in refined_predicates_jsons:
            pred_value_dict = {}
            
            # Handle list format (expected from new JSON format)
            if isinstance(refined_json, list):
                for p in refined_json:
                    # New format uses 'predicate_id' and 'new_value'
                    if 'predicate_id' in p and 'new_value' in p:
                        pred_value_dict[p['predicate_id']] = p['new_value']
                    # Old format used 'predicate' and 'value'
                    elif 'predicate' in p and 'value' in p:
                        pred_value_dict[p['predicate']] = p['value']
                    # Alternative format with attribute_name
                    elif 'attribute_name' in p and 'new_value' in p:
                        pred_value_dict[p['attribute_name']] = p['new_value']
            
            # Handle dictionary format (might be used in some responses)
            elif isinstance(refined_json, dict):
                if 'predicates' in refined_json:
                    for key, value in refined_json['predicates'].items():
                        pred_value_dict[key] = value
                elif 'selected_predicate_values' in refined_json:
                    pred_value_dict = refined_json['selected_predicate_values']
                    for key, value in refined_json['selected_predicate_values'].items():
                        pred_value_dict[key] = value
            
            # Add the extracted predicate values
            if pred_value_dict:
                refined_pred_values.append(pred_value_dict)
            else:
                print(f"Warning: Could not extract predicate values from: {refined_json}")
        
        # Extract reasoning sections
        reasonings = re.findall(r'<reasoning_concise_text>(.*?)</reasoning_concise_text>|<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        # Flatten and clean up reasoning tuples from regex groups
        reasonings = [next(r for r in reasoning_tuple if r) for reasoning_tuple in reasonings]
        
        return queries, refined_pred_values, reasonings

    def _parse_constraints_to_json(self, constraints) -> List[Dict[str, str]]:
        """Parse constraints into JSON format, handling both list and string inputs."""
        constraints_list = []
        if isinstance(constraints, list):
            constraint_items = constraints
        else:
            constraint_items = constraints.split('\n') if constraints else []

        for i, constraint in enumerate(constraint_items):
            if isinstance(constraint, str) and constraint.strip():
                constraints_list.append({
                    "id": f"C{i+1}",
                    "description_concise": constraint.strip(),
                    "target_value": "target_value",
                    "current_satisfaction_status": "not_satisfied"
                })
            elif hasattr(constraint, '__str__') and str(constraint).strip():
                constraints_list.append({
                    "id": f"C{i+1}",
                    "description_concise": str(constraint).strip(),
                    "target_value": "target_value",
                    "current_satisfaction_status": "not_satisfied"
                })
        return constraints_list

    def _parse_predicates_to_json(self, refineable_predicates) -> List[Dict[str, str]]:
        """Parse refineable predicates into JSON format, handling both list and string inputs."""
        predicates_list = []
        if isinstance(refineable_predicates, list):
            predicate_items = refineable_predicates
        else:
            predicate_items = refineable_predicates.split('\n') if refineable_predicates else []

        for i, pred in enumerate(predicate_items):
            pred_str = str(pred) if not isinstance(pred, str) else pred
            if pred_str.strip():
                is_numerical = '<refined_' in pred_str and ('_min' in pred_str or '_max' in pred_str)
                pred_parts = pred_str.split()
                attribute_name = pred_parts[1] if len(pred_parts) > 1 else f"attr{i+1}"
                predicates_list.append({
                    "id": pred.get_id(),
                    "attribute_name": attribute_name,
                    "operator": ">=" if is_numerical else "IN",
                    "value_type": "numerical" if is_numerical else "categorical"
                })
        return predicates_list

    def _format_previous_requests_json(self, previous_requests_and_results: Optional[str]) -> str:
        """Format previous requests/results into JSON format."""
        if previous_requests_and_results:
            return json.dumps({"previous_analysis_attempts_json": previous_requests_and_results})
        return "{}"

    def _format_subspace_lm_guidance_json(self, subspace_lm_investigation_suggestions: Optional[List[str]]) -> str:
        """Format subspace_lm investigation suggestions into JSON format."""
        if subspace_lm_investigation_suggestions:
            suggestions_list = []
            for i, sugg in enumerate(subspace_lm_investigation_suggestions):
                suggestions_list.append({
                    "suggestion_id": f"IS{i+1}",
                    "description_concise": sugg
                })
            return json.dumps({"subspace_lm_investigation_suggestions_json": suggestions_list})
        return "{}"

    def _execute_pandas_code(self, analysis_code: str, previous_queries: str):
        """Execute pandas code and return the result."""
        global_vars = {'df': self.input_dataset, 'pd': pd}
        local_vars = {}
        try:
            exec(previous_queries + analysis_code, global_vars, local_vars)
            return local_vars.get('result')
        except Exception as e:
            print(f"Error executing analysis code: {e}")
            return None

    def _parse_json_response(self, analysis_request_nl: str, previous_queries: str) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """Parse JSON format response and execute code."""
        json_match = re.search(r'```json\n(.*?)\n```', analysis_request_nl, re.DOTALL)
        if json_match:
            try:
                analysis_json = json.loads(json_match.group(1))
                analysis_request = analysis_json.get("request_plain_english_concise", "")
                analysis_code = analysis_json.get("pandas_query_implementation_code", "")

                if analysis_request and analysis_code:
                    result = self._execute_pandas_code(analysis_code, previous_queries)
                    if result is not None:
                        return analysis_request, result, analysis_code
            except json.JSONDecodeError:
                print("Failed to parse JSON response from analysis request")
        return None, None, None

    def _parse_legacy_response(self, analysis_request_nl: str, previous_queries: str) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """Parse legacy format response and execute code."""
        analysis_request_extracted = re.findall(r'<request>(.*?)</request>', analysis_request_nl, re.DOTALL)
        if analysis_request_extracted:
            analysis_request = analysis_request_extracted[0].strip().strip("- ").strip('"').strip(".")
            analysis_code_extracted = re.findall(r'<pandas_query_implementation>(.*?)</pandas_query_implementation>',
                                                analysis_request_nl, re.DOTALL)
            
            if analysis_code_extracted:
                result = self._execute_pandas_code(analysis_code_extracted[0], previous_queries)
                if result is not None:
                    return analysis_request, result, analysis_code_extracted[0]
        return None, None, None


class SubspaceLM(BaseLLMAgent):
    def __init__(
        self,
        task_name: str,
        constraints: str,
        epsilon: float,
        input_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parsed_constraints: List[AgnosticConstraint],
        original_query: str,
        refineable_predicates: List[Predicate],
        refinement_objective_description: str,
        log_dir: str = LOG_DIR,
        use_history: bool = True,
        use_skyline: bool = True,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # Store parsed constraints for target formatting
        self.long_term_memory: str = ""
        
        # Convert constraints to JSON format for the formal prompt
        constraints_json = json.dumps([{
            "id": f"C{i+1}",
            "description_concise": str(c),
            "target_value": getattr(c, "desired_value", "target_value"),
            "current_satisfaction_status": "not_satisfied"
        } for i, c in enumerate(parsed_constraints)])

        # Generate static context information for system prompt
        dataset_schema = build_dataset_schema_context(input_dataset)
        dataset_schema_context_json = json.dumps(dataset_schema)
        
        refineable_predicates_context_json = json.dumps([
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

        # Optionally compute attribute intelligence
        attribute_intelligence_context_json = "{}"
        try:
            if ENABLE_AGENT_ATTRIBUTE_STATS:
                stats_gen = AttributeStatsGenerator(input_dataset, refineable_predicates, sample_size=STATS_SAMPLE_SIZE)
                attribute_intelligence_context_json = json.dumps({
                    "attribute_intelligence": stats_gen.generate_stats()
                })
        except Exception:
            attribute_intelligence_context_json = "{}"

        # Initialize with memory-aware system prompt including task context
        prompt_template = SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE if use_skyline else SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE
        formatted_system_prompt = prompt_template.format(
            original_query_context=original_query.strip(),
            refineable_predicates_context_json=refineable_predicates_context_json,
            output_constraints_context_json=constraints_json,
            dataset_schema_context_json=dataset_schema_context_json,
            refinement_objective_description=refinement_objective_description,
            attribute_intelligence_context_json=attribute_intelligence_context_json,
        )
        
        super().__init__(task_name, formatted_system_prompt, log_dir, use_history=use_history, model_provider=model_provider, model_name=model_name)
        
        # Store information as instance variables for internal use
        self.constraints = constraints
        self.constraints_json = constraints_json
        self.epsilon = epsilon
        self.input_dataset = get_primary_dataframe(input_dataset)
        try:
            self.const_targets = {c.query_str: c.desired_value for c in parsed_constraints}
        except AttributeError:
            self.const_targets = {c.get_query_str(): c.desired_value for c in parsed_constraints}
        self.system_prompt = formatted_system_prompt
        
        # Store static context for system prompt updates
        self.original_query = original_query
        self.refineable_predicates_context_json = refineable_predicates_context_json
        self.dataset_schema_context_json = dataset_schema_context_json
        self.refinement_objective_description = refinement_objective_description
        # Structures for algorithmic state (Skyline S, global history H)
        # Import locally to respect modification scope
        self.use_skyline = use_skyline
        self.skyline = Skyline() if use_skyline else None  # \mathbb{S}
        self.global_history = defaultdict(list)  # H: maps serialized subspace -> list of records
        self.subspace_history = []  # list of summary dicts for prompt context
        self._last_suggested_subspace_key = None  # serialized key for associating updates

    @staticmethod
    def _md_table_from_subspace_history(hist: List[Dict[str, Any]]) -> str:
        try:
            rows: List[str] = []
            for item in (hist or [])[-7:]:
                subspace_str = item.get("subspace", "")
                subspace_display = str(subspace_str)
                avg_d = item.get("avg_distance", "")
                avg_c = item.get("avg_constraint_metric", "")
                n = item.get("num_trials", "")
                rows.append(f"| {subspace_display} | {avg_d} | {avg_c} | {n} |")
            header = "| Subspace | AvgDist | AvgDev | Trials |\n|---|---:|---:|---:|\n"
            body = "\n".join(rows) if rows else "| — | — | — | — |"
            return header + body
        except Exception:
            return "| Subspace | AvgDist | AvgDev | Trials |\n|---|---:|---:|---:|\n| — | — | — | — |"

    def get_subspace(self, skyline: str, optimization_context: Optional[str] = None) -> Tuple[SubspaceDict, Dict]:
        """
        Analyze the current subspace and provide feedback on its suitability for query refinement.

        Args:
            skyline: The current skyline description
            best_query_json: The current SQL query + constraint satisfaction + distance, as JSON
            optimization_context: Optional context about optimization mode and best satisfying query

        Returns:
            Updated SubspaceStructure with feedback and suggestions
        """

        # Prepare summarized subspace history for the prompt (averaged performance)
        # Recompute from global_history to keep it consistent
        history_summaries = []
        if self.use_history:
            for subspace_key, records in getattr(self, "global_history", {}).items():
                if not records:
                    continue
                avg_distance = sum(r.get("distance", 0.0) for r in records) / len(records)
                avg_constraint_metric = sum(r.get("agg_constraint_metric", 0.0) for r in records) / len(records)
                history_summaries.append({
                    "subspace": subspace_key,
                    "avg_distance": avg_distance,
                    "avg_constraint_metric": avg_constraint_metric,
                    "num_trials": len(records)
                })
        self.subspace_history = history_summaries if self.use_history else []

        subspace_history_md = self._md_table_from_subspace_history(history_summaries) if self.use_history else ""

        if self.use_skyline:
            prompt_template = (
                SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE
                if self.use_history
                else SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_HISTORY
            )
        else:
            prompt_template = (
                SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE
                if self.use_history
                else SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY
            )
        prompt = prompt_template.format(
            skyline=skyline,
            subspace_history_json=subspace_history_md
        )
        
        # Append optimization context if in optimization mode
        if optimization_context:
            prompt += f"\n\n{optimization_context}"

        prompt += f"\n\n🎯 **YOUR BREAKTHROUGH SUBSPACE:**\n"
        self.log_message("user", prompt, "Subspace Suggestion Request")
        messages, user_index = self._prepare_message_context(prompt)
        try:
            response_text = self.chat_model.generate(messages=messages)
        except Exception as e:
            if not self.use_history and self._is_context_overflow_error(e):
                self._log_clean_line("[ContextReset] Critic context window exceeded. Resetting conversation.")
                self._reset_to_system_prompt()
                messages, user_index = self._prepare_message_context(prompt)
                try:
                    response_text = self.chat_model.generate(messages=messages)
                except Exception as retry_error:
                    self._revert_user_message(user_index)
                    raise retry_error
            else:
                self._revert_user_message(user_index)
                raise

        # Extract JSON block
        json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        raw_json = json_match.group(1) if json_match else response_text.strip()

        try:
            parsed = json.loads(raw_json)
        except Exception:
            # Try to find a JSON object anywhere in the response
            brace_match = re.search(r"\{[\s\S]*\}", response_text)
            if brace_match:
                try:
                    # clean (remove) all comments (// or /* */) before parsing
                    comment_match = re.sub(r"//.*?$|/\*.*?\*/", "", brace_match.group(0), flags=re.DOTALL | re.MULTILINE)
                    parsed = json.loads(comment_match)
                except Exception:
                    parsed = {"selected_predicate_subspace": {}, "reasoning_concise": "parse_failed"}
            else:
                parsed = {"selected_predicate_subspace": {}, "reasoning_concise": "parse_failed"}

        parsed_for_log = parsed
        try:
            validated_payload = SubspaceLLMResponse.parse_obj(parsed)
            sel = validated_payload.selected_predicate_subspace
            parsed_for_log = validated_payload.dict()
        except ValidationError as ve:
            self._log_clean_line(f"[SubspaceValidationFailed] {ve}")
            sel = parsed.get("selected_predicate_subspace", {}) or {}

        refineable_predicate_context = json.loads(self.refineable_predicates_context_json)
        refineable_predicate_context_dict = {pred['id']: pred for pred in refineable_predicate_context}

        #subspace concise: only the [min, max] or [[contained], [containing]] values
        subspace_concise = {}
        for pred_id, val in sel.items():
            corresponding_pred = refineable_predicate_context_dict[pred_id]
            if 'valid_values' in corresponding_pred['valid_value_range'].keys():  # identify range case
                values = corresponding_pred['valid_value_range']['valid_values']
                subspace_concise[str(pred_id)] = sorted(list(set([v for v in values if v >= val[0] and v <= val[1]])))
            else:
                subspace_concise[str(pred_id)] = val

        # Convert into SubspaceDict (per Preliminary Definitions)
        numeric_predicates = {}
        categorical_predicates = {}
        for pred_id, val in sel.items():
            # Numerical: [min, max] of numbers
            if (
                isinstance(val, (list, tuple)) and len(val) == 2 and
                all(isinstance(x, (int, float)) for x in val)
            ):
                corresponding_pred = refineable_predicate_context_dict[pred_id]
                if 'valid_values' in corresponding_pred['valid_value_range'].keys():  # identify range case
                    values = corresponding_pred['valid_value_range']['valid_values']
                    numeric_predicates[str(pred_id)] = sorted(list(set([v for v in values if v >= val[0] and v <= val[1]])))
                else:
                    step = corresponding_pred['valid_value_range']['step']
                    numeric_predicates[str(pred_id)] = np.arange(val[0], val[1] + step, step).round(2).tolist()

            # Categorical: [[contained_set], [containing_set]]
            elif (
                isinstance(val, (list, tuple)) and len(val) == 2 and
                isinstance(val[0], (list, tuple)) and isinstance(val[1], (list, tuple))
            ):
                contained = [str(x) for x in val[0]]
                containing = [str(x) for x in val[1]]
                categorical_predicates[str(pred_id)] = subsets_containing(contained, containing)
            else:
                # Unknown structure; ignore gracefully
                continue

        subspace_obj = SubspaceDict(
            numeric_predicates=numeric_predicates,
            categorical_predicates=categorical_predicates
        )

        # Log assistant response (store raw JSON/response for traceability)
        self._finalize_assistant_message(response_text)
        self.log_message("assistant", json.dumps(parsed_for_log), "Subspace Analysis Response")
        # Keep a stable serialized key for H indexing in update_history
        try:
            self._last_suggested_subspace_key = json.dumps(sel, sort_keys=True)
        except Exception:
            self._last_suggested_subspace_key = json.dumps({"numeric": numeric_predicates, "categorical": categorical_predicates}, sort_keys=True)

        return subspace_obj, subspace_concise

    def update_history(self, query: str, constraint_satisfaction: Dict[str, Union[int, float]], constraint_deviation_score: float, refinement_distance: float):
        """
        Update Skyline (S) and global history (H) with a new evaluated query.

        Args:
            query: SQL of evaluated refinement (theta candidate)
            constraint_satisfaction: Mapping of constraint-id -> deviation/metric (lower is better)
            constraint_deviation_score: Aggregated scalar deviation score (lower is better)
            refinement_distance: Distance to original query (lower is better)
        """

        # Aggregate constraint metrics to a single scalar (mean) for Skyline dominance checks
        # Insert into Skyline
        if not self.use_history:
            return

        if self.use_skyline and self.skyline is not None:
            try:
                self.skyline.insert_query(query, float(refinement_distance), float(constraint_deviation_score))
            except Exception:
                pass

        # Attach to the most recent suggested subspace bucket in global history
        subspace_key = getattr(self, "_last_suggested_subspace_key", None) or json.dumps({"root": True})
        record = {
            "query": query,
            "distance": float(refinement_distance),
            "constraints": constraint_satisfaction,
            "agg_constraint_metric": float(constraint_deviation_score)
        }
        try:
            self.global_history[subspace_key].append(record)
        except Exception:
            # Initialize if needed
            self.global_history = getattr(self, "global_history", {})
            self.global_history.setdefault(subspace_key, []).append(record)

        # Refresh subspace summaries cache for prompt context
        history_summaries = []
        for k, records in self.global_history.items():
            if not records:
                continue
            avg_distance = sum(r.get("distance", 0.0) for r in records) / len(records)
            avg_constraint_metric = sum(r.get("agg_constraint_metric", 0.0) for r in records) / len(records)
            history_summaries.append({
                "subspace": k,
                "avg_distance": avg_distance,
                "avg_constraint_metric": avg_constraint_metric,
                "num_trials": len(records)
            })
        self.subspace_history = history_summaries
