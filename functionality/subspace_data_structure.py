import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Union, Tuple
import ast

import pandas as pd
from pydantic import BaseModel

from chat_model import ChatModel
from functionality.predicate import (
    Predicate,
    NumericalPredicate,
    CategoricalPredicate, NumericalAttribute
)
from tools.utils import construct_predicate_steps_by_jaccard


# subspace_map: {'P1': [p1_value1, p1_value2, ...], 'P2': [p2_value1, p2_value2, ...], ...}
MIDDLE_VAL_JACCARD = 9
NUM_SIDE_STEPS = 3


class SubspaceDict(BaseModel):
    # Define fields based on the *actual* JSON structure returned by the model
    numeric_predicates: Dict[str, List[float]]  # e.g., {"age lower bound": [25, 35]}
    categorical_predicates: Dict[str, List[Tuple[Any, ...]]]  # e.g., {"country": [("United States",), ("Germany",)]}


# class Subspace:


class SubspaceStructure:
    def __init__(
        self,
        original_predicates: List[Predicate]
    ):
        """
        Initialize the RadiusStructure with the original set of predicates.
        """
        self.radius = 0
        # Map attribute names to attribute objects for reconstruction
        self.attribute_map: Dict[str, Any] = {
            pred.attribute.name: pred.attribute for pred in original_predicates
        }
        self.original_predicates = original_predicates
        # Seed both with original predicates
        self.axis_values = {prd.get_id(): prd for prd in original_predicates}
        # initialize selected_predicates with the original predicates
        self.selected_predicates = {k: [v] for k, v in self.axis_values.items()}

    def get_current_predicates_dict(self) -> Dict[str, str]:
        """
        Get a list of all predicates in the selected_predicates.
        :return:
        """
        pred_vals = {}
        for pred_id, preds in self.selected_predicates.items():
            if preds and isinstance(preds[0], NumericalPredicate):
                vals = sorted({p.value for p in preds} | {self.axis_values[pred_id].value})
                vals_str = "{" + ", ".join(str(v) for v in vals) + "}"
            elif preds and isinstance(preds[0], CategoricalPredicate):
                pred_possible_values = self.axis_values[pred_id].attribute.categories
                median_set = set(self.axis_values[pred_id].values)
                pred_possible_values_without_median = set(pred_possible_values) - median_set
                vals_str = f"{{{median_set}}} ⋃ " \
                           f"{{{median_set} \ s | s ∈ {median_set}}} ⋃ " \
                           f"{{{median_set} ⋃ s | s ∈ {pred_possible_values_without_median}}}"
            pred_vals[f"{pred_id} range"] = vals_str
        return pred_vals

    # Note: Orchestrator should call update_radius(history_md, parsed_constraints, chat_model) to expand
    def render_subspace(self, memory_df: pd.DataFrame) -> str:
        """
        Deterministic alternative: take the top-performing row (assumed first) from memory_df,
        and generate a predicate range string in the same format as get_current_predicates,
        using only get_closest_predicates() of the best-performing predicates.
        """
        lines = []
        # For each original predicate, use its best recorded value
        for orig in self.original_predicates:
            col_id = orig.get_id()
            if col_id not in memory_df.columns:
                continue
            
            # Handle empty memory DataFrame by using original values
            if memory_df.empty:
                best_val = orig.value if isinstance(orig, NumericalPredicate) else orig.values
            else:
                try:
                    best_val = memory_df.iloc[0][col_id]
                except IndexError as e:
                    print(f"Error accessing memory_df: {e}")
                    print(f"Memory DataFrame columns: {memory_df.columns}")
                    print(f"Memory DataFrame shape: {memory_df.shape}")
                    print(f"Original predicate ID: {col_id}")
                    print(f"Original predicate: {orig}")
                    # Use original value as fallback
                    best_val = orig.value if isinstance(orig, NumericalPredicate) else orig.values
                    
            # Numerical case
            if isinstance(orig, NumericalPredicate):
                temp_pred = NumericalPredicate(orig.attribute, orig.operator, best_val)
                self.axis_values[col_id] = temp_pred
                neighbors = temp_pred.get_closest_predicates(3)
                self.selected_predicates[col_id] = neighbors
                vals = {best_val} | {p.value for p in neighbors}
                sorted_vals = sorted(vals)
                lines.append(f"• {col_id} ∈ {{{', '.join(str(v) for v in sorted_vals)}}}")
            # Categorical case
            elif isinstance(orig, CategoricalPredicate):
                temp_pred = CategoricalPredicate(orig.attribute, list(best_val))
                median_set = set(temp_pred.values)
                all_possible_values = set(self.axis_values[col_id].attribute.categories) - median_set
                self.axis_values[col_id] = temp_pred
                neighbors = temp_pred.get_closest_predicates(2)
                self.selected_predicates[col_id] = neighbors
                lines.append(f"• {col_id} ∈ {{{median_set}}} ⋃ "
                             f"{{{median_set} \ s | s ∈ {median_set}}} ⋃ "
                             f"{{{median_set} ⋃ s | s ∈ {all_possible_values}}}")
        return "\n".join(lines)

    def get_distance_from_original_experimental(self) -> float:
        """
        Calculate the distance from the original predicates to the current predicates.
        :return:
        """
        distance = 0
        for orig_pred in self.original_predicates:
            pred_id = orig_pred.get_id()
            if pred_id not in self.selected_predicates:
                continue
            # Calculate distance based on the type of predicate
            if isinstance(orig_pred, NumericalPredicate):
                median_val = self.axis_values[pred_id].value
                # Normalize the distance by the original value
                distance += abs(orig_pred.value - median_val) / orig_pred.value
            elif isinstance(orig_pred, CategoricalPredicate):
                median_set = set(self.axis_values[pred_id].values)
                orig_set = set(orig_pred.values)
                # 1 - jaccard_similarity(orig_set, median_set)
                distance += len(orig_set.symmetric_difference(median_set)) / len(orig_set.union(median_set))
        return distance

# DEPRECATED: This class is defined but never instantiated anywhere in the codebase.
# Consider removing if not needed for future use.
class JaccardSubspaceStructure(SubspaceStructure):
    def __init__(
        self,
        original_predicates: List[Predicate],
        input_dataset: pd.DataFrame
    ):
        """
        Initialize the JaccardSubspaceStructure with the original set of predicates
        and the attribute for Jaccard distance calculation.
        """
        super().__init__(original_predicates)
        self.predicate_steps_map = {}
        self.pred_val_to_step_id_map = defaultdict(dict)
        for pred_id in self.axis_values.keys():
            axis_pred = self.axis_values[pred_id]
            pred_steps_arr = construct_predicate_steps_by_jaccard(axis_pred.attribute.name,
                                                                  axis_pred.value, input_dataset)
            if '>' in axis_pred.operator:
                pred_steps_arr = pred_steps_arr[:-1]
            elif '<' in axis_pred.operator:
                pred_steps_arr = pred_steps_arr[1:]
            self.predicate_steps_map[pred_id] = [NumericalPredicate(axis_pred.attribute, axis_pred.operator, val)
                                                 for val in pred_steps_arr]
            self.pred_val_to_step_id_map[pred_id] = {pred_val: i for i, pred_val in enumerate(pred_steps_arr)}

        # TODO: for the special case of Jaccard distance we want that instead of using "steps",
        #  The LLM should be provided with all possible values for each predicate.
        a = 1

    def render_subspace(self, memory_df: pd.DataFrame) -> str:
        """
        Deterministic alternative: take the top-performing row (assumed first) from memory_df,
        and generate a predicate range string in the same format as get_current_predicates,
        using only get_closest_predicates() of the best-performing predicates.
        """
        lines = []
        # For each original predicate, use its best recorded value
        for orig in self.original_predicates:
            col_id = orig.get_id()
            if col_id not in memory_df.columns or not isinstance(orig, NumericalPredicate):
                continue

            # Handle empty memory DataFrame by using original values
            if memory_df.empty:
                best_val = orig.value
            else:
                try:
                    best_val = memory_df.iloc[0][col_id]
                except IndexError as e:
                    print(f"Error accessing memory_df: {e}")
                    print(f"Memory DataFrame columns: {memory_df.columns}")
                    print(f"Memory DataFrame shape: {memory_df.shape}")
                    print(f"Original predicate ID: {col_id}")
                    print(f"Original predicate: {orig}")
                    # Use original value as fallback
                    best_val = orig.value

            # Numerical case
            temp_pred = NumericalPredicate(orig.attribute, orig.operator, best_val)
            self.axis_values[col_id] = temp_pred
            new_axis_id = self.pred_val_to_step_id_map[col_id][best_val]
            neighbors = self.predicate_steps_map[col_id][new_axis_id - NUM_SIDE_STEPS:new_axis_id + NUM_SIDE_STEPS + 1]
            self.selected_predicates[col_id] = neighbors
            vals = {best_val} | {p.value for p in neighbors}
            sorted_vals = sorted(vals)
            lines.append(f"• {col_id} ∈ {{{', '.join(str(v) for v in sorted_vals)}}}")
            # Categorical case
        return "\n".join(lines)

    def get_distance_from_original_experimental(self) -> float:
        """
        Calculate the distance from the original predicates to the current predicates.
        :return:
        """
        distance = 0
        for orig_pred in self.original_predicates:
            pred_id = orig_pred.get_id()
            if pred_id not in self.selected_predicates:
                continue
            # Calculate distance based on the type of predicate
            if isinstance(orig_pred, NumericalPredicate):
                orig_val = orig_pred.value
                median_val = self.axis_values[pred_id].value
                dist_by_indices = abs(self.pred_val_to_step_id_map[pred_id][orig_val] - \
                                  self.pred_val_to_step_id_map[pred_id][median_val])
                distance += dist_by_indices / (MIDDLE_VAL_JACCARD - 1)
            elif isinstance(orig_pred, CategoricalPredicate):
                distance += 1
        return distance / len(self.original_predicates)
