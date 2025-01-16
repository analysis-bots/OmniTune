FAIRNESS_EVALUATION_MOCK = """
import pandas as pd
from typing import Dict, List

def get_constraints_satisfaction_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
    # Outer function to assess constraints satisfaction
    def constraints_satisfaction_objective(refined_query: str) -> float:
        # Execute the refined query to get the output dataframe
        sql_engine = SQLEngineAdvanced(d_in)
        resp_df, error = sql_engine.execute(refined_query)
        if error:
            return 1.0  # If an error occurs, we consider the quality as the worst (1.0)

        # Initialize deviation scores
        deviations = []

        # Constraint 1: Average grade in Portuguese for female students should be at least 65
        female_students = resp_df[resp_df['sex'] == 'F']
        if not female_students.empty:
            avg_portuguese = female_students['portugese'].mean()
            deviation_1 = max(0, 70 - avg_portuguese) / 70  # Normalize deviation
            deviations.append(deviation_1)
        else:
            deviations.append(1)  # If no female students, treat as unsatisfied

        # Constraint 2: Average math grade for students with a family of more than 3 members should be at least 60
        students_with_large_family = resp_df[resp_df['famsize'] == 'GT3']
        if not students_with_large_family.empty:
            avg_math = students_with_large_family['math'].mean()
            deviation_2 = max(0, 60 - avg_math) / 60  # Normalize deviation
            deviations.append(deviation_2)
        else:
            deviations.append(1)  # If no such students, treat as unsatisfied

        # Calculate final fairness deviation score
        fairness_deviation_score = sum(deviations) / len(deviations)

        return fairness_deviation_score

    return constraints_satisfaction_objective
"""

REFINEMENT_MOCK = """
import pandas as pd
from typing import Dict, List

def get_refinement_distance_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
    # Parse the original query to extract predicates
    original_predicates = parse_where_predicates(original_query)

    def refinement_distance_objective(refined_query: str) -> float:
        # Parse the refined query to extract predicates
        refined_predicates = parse_where_predicates(refined_query)

        # Calculate categorical distance using Jaccard similarity
        original_categorical_values = set()
        refined_categorical_values = set()

        # Extract values for categorical predicates
        for condition in original_predicates['categorical']:
            if condition['attribute_name'] == 'reason':
                original_categorical_values.update(condition['in_values'])

        for condition in refined_predicates['categorical']:
            if condition['attribute_name'] == 'reason':
                refined_categorical_values.update(condition['in_values'])

        intersection = len(original_categorical_values.intersection(refined_categorical_values))
        union = len(original_categorical_values.union(refined_categorical_values))
        jaccard_similarity = intersection / union if union > 0 else 0
        categorical_distance = 1 - jaccard_similarity

        # Calculate numerical distance
        original_numerical_value = None
        refined_numerical_value = None

        # Extract values for numerical predicates
        for condition in original_predicates['numerical']:
            if condition['attribute_name'] == 'goout':
                original_numerical_value = condition['value']

        for condition in refined_predicates['numerical']:
            if condition['attribute_name'] == 'goout':
                refined_numerical_value = condition['value']

        numerical_distance = 0.0
        if original_numerical_value is not None and refined_numerical_value is not None:
            numerical_distance = abs(original_numerical_value - refined_numerical_value) / original_numerical_value

        # Combine the distances with a simple average
        total_distance = (categorical_distance + numerical_distance) / 2.0

        return total_distance

    return refinement_distance_objective
"""

REFINEMENT_QUERY_SPEC_MOCK = """
import pandas as pd
from typing import Dict, List

def get_refinement_distance_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
    # Parse the original query to extract predicates
    original_where_clause = extract_where_clause(original_query)
    parsed_original = parse_where_clause(original_where_clause)

    # Get the min and max values for the 'goout' numerical column from the dataset
    goout_min = 1  # Minimum value for 'goout'
    goout_max = 5  # Maximum value for 'goout'
    goout_range = goout_max - goout_min

    # Get unique values for 'reason' categorical column
    reason_unique_values = ['course', 'other', 'home', 'reputation']

    def refinement_distance_objective(refined_query: str) -> float:
        refined_where_clause = extract_where_clause(refined_query)
        numeric_distance = 0
        categorical_distance = 0
        parsed_refined = parse_where_clause(refined_where_clause)

        for categorical_predicate in parsed_refined.categorical:
            for original_predicate in parsed_original.categorical:
                # Calculate differences for categorical attributes
                if categorical_predicate.name == original_predicate.name:
                    # Calculate the 1 - Jaccard similarity between the two sets of values
                    original_values = set(original_predicate.values)
                    refined_values = set(categorical_predicate.values)
                    intersection = original_values.intersection(refined_values)
                    union = original_values.union(refined_values)
                    predicate_distance = (1 - (len(intersection) / len(union))) if union else (1 if intersection else 0)

                    # Add the distance to the total categorical distance
                    categorical_distance += predicate_distance

        for numerical_predicate in parsed_refined.numerical:
            for original_predicate in parsed_original.numerical:
                refined_name, orig_name = numerical_predicate.name, original_predicate.name
                refined_op, orig_op = numerical_predicate.operator, original_predicate.operator

                # Calculate differences for numerical attributes
                if refined_name == orig_name and refined_op == orig_op:
                    refined_val, orig_val = numerical_predicate.value, original_predicate.value
                    if goout_range != 0:
                        # Calculate the relative difference based on the range of the attribute
                        predicate_distance = abs(refined_val - orig_val) / goout_range
                    else:
                        predicate_distance = 0  # Handle the case where range is zero

                    # Add the distance to the total numerical distance
                    numeric_distance += predicate_distance

        return categorical_distance + numeric_distance

    return refinement_distance_objective
"""
