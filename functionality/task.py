from abc import ABC
from typing import List

import pandas as pd
from math import exp
from functionality.constraint import OutputConstraint, DiverseTopKSelectionConstraint, RangeQueryFairnessConstraint, \
    DiversityCardinalityConstraint

from enum import Enum
import streamlit as st

from .objectives import get_script_diff_func_sql, get_range_query_distance_function, \
    get_informative_query_validation_function, get_informative_diversity_validation_function, evaluate_constraints, \
    get_pvl_dist_func, get_informative_top_k_validation_function
from tools.sql_engine import SQLEngine, SQLEngineAdvanced
from tools.utils import extract_where_clause, get_alterable_attributes, parse_where_predicates


class TaskType(Enum):
    TOP_K_REFINEMENT = 0
    RANGE_QUERY_REFINEMENT = 1
    DIVERSITY_CONSTRAINT = 2


TASK_TO_REFINEMENT_OBJECTIVE_MAP = {
    TaskType.TOP_K_REFINEMENT: get_script_diff_func_sql,
    TaskType.RANGE_QUERY_REFINEMENT: get_range_query_distance_function,
    TaskType.DIVERSITY_CONSTRAINT: ...
}


class Task(ABC):

    def __init__(self, name: str, df: pd.DataFrame, query: str, constraints: List[OutputConstraint], task_type: TaskType):
        self.name = name
        self.df = df
        self.original_query = query
        self.output_constraints = constraints
        self.task_type = task_type
        self.sql_engine = SQLEngine(df)
        self.alterable_attributes_map = self._get_alterable_attributes_map()
        self.numeric_attributes = self.alterable_attributes_map['numeric']
        self.categorical_attributes = self.alterable_attributes_map['categorical']
        self.constrained_attributes_set = set([c.attribute for c in constraints])
        self.epsilon = 0.0
        self.key_rules = ""
        self.refinement_objective = lambda x: 1
        self.refinement_objective_str = ""
        self.constraints_objective_str = ""
        self.validation_function = lambda x: False
        self.alterable_attributes = list(set(self.alterable_attributes_map['categorical'] + self.alterable_attributes_map['numeric']))

        self.constraints_str = "\n".join([str(c) for c in self.output_constraints])
        self.alterable_attributes_str = "\n".join([f"    - {v} ({str(self.df.dtypes[v])})" for v in self.alterable_attributes])
        self.constrained_attributes = "\n".join(
            [f"    - {v} ({str(self.df.dtypes[v])})" for v in self.constrained_attributes_set])
        self.categorical_attributes_str = "\n".join([f"""    - For categorical predicate of attribute {a}: Replacing / Removing / Adding values to the {a} attribute, 
        either concatenating predictes with {a} attribute with an "OR" operator, or using <attribute> IN (<value1>, ...).""" for a in self.categorical_attributes])
        self.numeric_attributes_str = "\n".join([f"""    - For numeric predicate of attribute {a}: Lowering / Raising values of {a} to create a valid range.""" for a in self.numeric_attributes])

    def generate_requirements(self):
        pass

    def _get_alterable_attributes_map(self):
        where_clause = extract_where_clause(self.original_query)
        alterable_attributes_map = get_alterable_attributes(where_clause)
        return alterable_attributes_map

    def evaluate_constraints_deviation(self, refined_query):
        out_df, is_exc = self.sql_engine.execute(refined_query)
        if is_exc:
            return f'query invalid. exception: {out_df}'
        return evaluate_constraints(out_df, self.output_constraints)


class TopKRefinementTask(Task):
    def __init__(self, name: str, df: pd.DataFrame, query: str, constraints: List[DiverseTopKSelectionConstraint],
                 epsilon=0.0):
        super().__init__(name, df, query, constraints, TaskType.TOP_K_REFINEMENT)
        self.epsilon = epsilon
        self.key_rules = """4. In cases of numerical predicates you can only adjust the values while keeping the operators the same.
5. In cases of categorical predicates you can either add, replace or remove categories that exist in the dataset's column from the predicate.
6. DO NOT under any circumstances repeat an operation that was previously generated!
7."""
        self.refinement_objective = get_script_diff_func_sql(query, df)
        self.refinement_objective_str = """   - For categorical predicates, the distance is based on 1-Jaccard similarity between the original set and the refined set.
        - For numeric predicates, the distance is based on absolute distance between the refined value and original value, divided by the original value."""
        self.constraints_objective_str = "The degree of constraint satisfaction is measured on a scale of 0 to 1."
        max_k = max([c.k for c in constraints])
        self.validation_function = get_informative_top_k_validation_function(query, df, k=max_k)


class RangeQueryRefinementTask(Task):
    def __init__(self, name: str, df: pd.DataFrame, query: str, constraints: List[RangeQueryFairnessConstraint],
                 epsilon=100):
        super().__init__(name, df, query, constraints, TaskType.RANGE_QUERY_REFINEMENT)
        self.epsilon = epsilon
        self.key_rules = """4. You are always allowed to increase the values of both lower and upper bounds of the existing predicates,
    as well as add a new predicate to the WHERE clause as long as it is based on an existing attribute 
    and the opposite operator of the existing predicate, 
    such that there are NO MORE than 2 predicates for each attribute,
    and they must be concatenated with an AND operator between them.
5. Hint: Use the execute_python_code_line_with_dataframe tool to investigate 
    the relations between different values of attributes in the dataframe to find the best refinement.
6. DO NOT under any circumstances repeat an operation that was previously generated!
7."""
        self.refinement_objective = get_range_query_distance_function(query, df)
        self.refinement_objective_str = """      - The refinement distance is based on (1 - Jaccard similarity) between the values in the original query's output dataset
and the refined query's output dataset."""
        self.constraints_objective_str = "To lower the disparity, the refined query's output should satisfy the constraints to the desired extent."
        self.validation_function = get_informative_query_validation_function(query, df)


#TODO - fill in the missing parts
class DiversityConstraintsTask(Task):
    def __init__(self, name: str, df: pd.DataFrame, query: str, constraints: List[DiversityCardinalityConstraint],
                 min_df_length: int=20, max_df_length: int=100):
        super().__init__(name, df, query, constraints, TaskType.DIVERSITY_CONSTRAINT)
        # TODO: Rewrite this whole part according to the new implementation of DiversityCardinalityConstraint
        self.epsilon = 0.0
        self.key_rules = f"""4. For the refined query to be considered valid, the refined query's output dataframe's 
    length MUST be within {min_df_length} and {max_df_length} rows.
5. In cases of categorical predicates you can either add, replace or remove categories. 
    Please, use ONLY existing categorical values that can be found in the original dataset's column.
6. In cases of numerical predicates you can only adjust the values, while keeping the operators the same. 
7. DO NOT under any circumstances repeat an operation that was previously generated!
8."""
        self.refinement_objective = get_pvl_dist_func(query, df)
        self.refinement_objective_str = "    - The distance is (1 - Jaccard similarity) between the original query's " \
                                        "output dataset and the refined query's output dataset."
        self.constraints_objective_str = """The constraints deviation is a score between 0 and 1, relative to the ratio between 
        absolute the distance number of records that satisfy the constraint, to by the required number of rows that must 
        satisfy it. Formally, if we denote the number of rows that satisfy the constraint as n, and the required number
        of rows as k, the constraints deviation score is calculated as max(0, (k - n) / k) for 'AT LEAST' constraints, and
        max(0, (n - k) / k) for 'AT MOST' constraints."""
        self.validation_function = get_informative_diversity_validation_function(query, df)
        self.constraints_map = {c.attribute: c for c in constraints}


TASK_TYPE_TO_CLASS_MAP = {
    "DiverseTopKRefinement": TopKRefinementTask,
    "RangeQueryRefinement": RangeQueryRefinementTask,
    "DiversityConstraints": DiversityConstraintsTask
}

TASK_TYPE_TO_CONSTRAINT_CLASS_MAP = {
    "DiverseTopKRefinement": DiverseTopKSelectionConstraint,
    "RangeQueryRefinement": RangeQueryFairnessConstraint,
    "DiversityConstraints": DiversityCardinalityConstraint
}


class AgnosticTask:
    def __init__(self, name, dataset, query, constraints_str, refinement_objective_str,
                 evaluate_constraints_deviation, refinement_objective, epsilon=0.0):
        self.name = name
        self.df = dataset[self.name]
        self.original_query = query
        self.constraints_str = constraints_str
        self.refinement_objective_str = refinement_objective_str
        self.validation_function = get_informative_query_validation_function(query, self.df)
        self.refinement_objective = refinement_objective
        self.evaluate_constraints_deviation = evaluate_constraints_deviation
        self.alterable_attributes_map = self._get_alterable_attributes_map()
        self.numeric_attributes = self.alterable_attributes_map['numeric']
        self.categorical_attributes = self.alterable_attributes_map['categorical']
        self.alterable_attributes = list(set(self.alterable_attributes_map['categorical'] + self.alterable_attributes_map['numeric']))
        self.alterable_attributes_str = "\n".join([f"    - {v} ({str(self.df.dtypes[v])})" for v in self.alterable_attributes])
        # Make sure the model **understands** the numeric type of each numeric attribute (int, float)!!!!!
        self.categorical_attributes_str = "\n".join([f"""    - For categorical predicate of attribute {a}: Replacing / Removing / Adding values to the {a} attribute, 
        either concatenating predictes with {a} attribute with an "OR" operator, or using <attribute> IN (<value1>, ...).""" for a in self.categorical_attributes])
        self.numeric_attributes_str = "\n".join([f"""    - For numeric predicate of attribute {a}: Lowering / Raising values of {a} to create a valid range.""" for a in self.numeric_attributes])
        self.epsilon = round(epsilon, 2)

    def _get_alterable_attributes_map(self):
        where_clause = extract_where_clause(self.original_query)
        alterable_attributes_map = get_alterable_attributes(where_clause)
        return alterable_attributes_map

def from_json(json_obj):
    task_class = TASK_TYPE_TO_CLASS_MAP[json_obj["task_type"]]
    name = json_obj["name"]
    df = pd.read_csv(json_obj["df_path"])
    query = json_obj["query"]
    constraint_class = TASK_TYPE_TO_CONSTRAINT_CLASS_MAP[json_obj["task_type"]]
    constraints = [constraint_class(**c) for c in json_obj["constraints"]]
    if task_class == TopKRefinementTask:
        epsilons = [json_obj["fairness_epsilon"]]
        print(task_class(name, df, query, constraints, epsilons))
        return task_class(name, df, query, constraints, epsilons)
    if task_class == DiversityConstraintsTask:
        data_size_constraint = json_obj["dataset_size_constraint"]
        if data_size_constraint["operator"] == ">=":
            min_df_length = data_size_constraint["value"]
            max_df_length = 10000
        else:
            min_df_length = 0
            max_df_length = data_size_constraint["value"]
        print(task_class(name, df, query, constraints, min_df_length, max_df_length))
        return task_class(name, df, query, constraints, min_df_length, max_df_length)
    print(task_class(name, df, query, constraints))
    return task_class(name, df, query, constraints)