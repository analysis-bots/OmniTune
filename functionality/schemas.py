from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Literal, Any


class ResultStructure(BaseModel):
    lower_bound_columns: list[str]
    upper_bound_columns: list[str]
    category_subset_columns: list[str]


class Condition(BaseModel):
    operator: str
    original: str
    refined: str


class Predicate(BaseModel):
    attribute: str
    conditions: list[Condition]


# --- Parameter Sub-Models ---

class NumericalAttributeParams(BaseModel):
    attribute_name: str
    current_value: Optional[float] = None  # Value used in the query predicate
    explored_range: Optional[List[float]] = None  # e.g., [lower_bound, upper_bound] for sensitivity
    operator: Optional[str] = None  # e.g., ">=", "<", "=="


class CategoricalAttributeParams(BaseModel):
    attribute_name: str
    current_value_set: Optional[List[str]] = None  # Values in IN clause
    explored_category: Optional[str] = None  # Category added/removed/focused on
    operator: Optional[str] = None  # e.g., "IN", "=="


class ConstraintParams(BaseModel):
    constraint_attribute: str
    constraint_target_value: Any  # The specific value checked in the constraint (e.g., 'race2')

# --- Result Data Sub-Models ---


class SensitivityResultData(BaseModel):
    rows_affected: int  # How many rows added/removed by the change
    constraint_rows_affected: int  # How many of the affected rows satisfy the constraint target
    # Optional: Distribution within the affected slice
    affected_slice_constraint_distribution: Optional[Dict[str, int]] = None


class DistributionResultData(BaseModel):
    # For Schema D1/D2: Distribution across bins/categories
    # Key: bin range string or category name
    # Value: Dict {'total_count': int, 'constraint_count': int}
    distribution: Dict[str, Dict[str, int]]


class InteractionResultData(BaseModel):
    # Key: Identifies the slice/facet (e.g., "num_children_below_3", "county_county2")
    # Value: Can be SensitivityResultData or another nested structure if needed
    faceted_results: Dict[str, SensitivityResultData]


class NumericalStatsData(BaseModel):
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None


# --- Main Analysis Result Model ---
class AnalysisResult(BaseModel):
    analysis_type: Literal[
        "numerical_sensitivity_relax",
        "numerical_sensitivity_tighten",
        "categorical_sensitivity_add",
        "categorical_sensitivity_remove",
        "constraint_distribution_numerical",
        "constraint_distribution_categorical",
        "interaction_numerical_numerical",
        "interaction_numerical_categorical",
        "interaction_categorical_numerical",
        "interaction_categorical_categorical",
        "baseline_state"
    ]
    description: str
    primary_attribute: Optional[Union[NumericalAttributeParams, CategoricalAttributeParams]] = None # Optional for baseline
    secondary_attribute: Optional[Union[NumericalAttributeParams, CategoricalAttributeParams]] = None
    constraint_params: Optional[ConstraintParams] = None

    # --- FIX: Ensure these are Optional ---
    other_conditions_applied: Optional[str] = None # Optional: Descriptive string of other conditions
    result_data: Optional[Union[
        SensitivityResultData,
        DistributionResultData,
        InteractionResultData,
        NumericalStatsData,
        Dict[str, Any]
    ]] = None # Optional: Populated AFTER execution


# --- Top-Level Report ---


class AnalysisReport(BaseModel):
    report_summary: Optional[str] = Field(None, description="Optional LLM-generated summary of overall findings")
    analysis_results: List[AnalysisResult]

