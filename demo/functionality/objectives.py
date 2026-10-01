import re
import sys
import traceback
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame
from functionality.constraint import OutputConstraint
from tools.sql_engine import SQLEngine
from tools.utils import extract_where_clause, get_column_identifiers, parse_where_clause, \
    categorical_refinement_distance, get_query_pattern, extract_code


########################################## Common Query Validation Objectives ##########################################
def get_query_validation_function(original_query: str, dataset: pd.DataFrame):

    query_pattern = get_query_pattern(original_query)
    allowed_filter_attributes = get_allowed_filter_attributes(original_query)
    original_dataset = dataset

    def validate_query(query: str) -> bool:
        # Clean query
        query = query.replace('\n', ' ').replace('`', '"').strip()

        # Validate that the query fits the pattern
        if not re.fullmatch(query_pattern, query):
            return False

        # Extract the WHERE clause from the query
        where_clause = extract_where_clause(query)

        where_clause = where_clause.replace('`', '"')
        # Extract the column names from the WHERE clause
        attribute_matches = re.findall(r"(['\"]?)(\w+)\1(?=\s*(=|>|<|>=|<=|!=|IN|LIKE|IS))", where_clause, re.IGNORECASE)
        column_names = [match[1] for match in attribute_matches]
        # TODO - Find an alternative way to do this without removing the brackets
        # TODO - Because right now it fails in the columns of <...> OR <...> clauses
        clean_where_clause = where_clause.replace(')', ' ').replace('(', ' ')

        clean_column_names_map = {}

        for col in column_names:
            if col not in original_dataset.columns:
                return False
            # for every column name replace all spaces with underscores
            clean_where_clause = clean_where_clause.replace(col, col.replace(' ', '_'))
            clean_column_names_map[col.replace(' ', '_')] = col

        is_prev_col = False
        is_prev_or = False
        prev_col = None
        for w in clean_where_clause.split(' '):
            if len(w) == 0:
                continue
            for c in clean_column_names_map.keys():
                if c in w:
                    w = clean_column_names_map[c]
                    break
            if w in column_names:
                is_prev_col = True
                prev_col = w
            elif is_prev_col:
                if w not in ['=', '>', '<', '>=', '<=', 'IN']:
                    return False
                is_prev_col = False
            # Check all OR conditions are between the same column predicate and only categorical
            elif w == 'OR':
                if prev_col is None:
                    return False
                is_prev_or = True
                is_prev_col = False
                continue
            if is_prev_or:
                if w not in column_names or w != prev_col or w in original_dataset.select_dtypes(include='number').columns:
                    return False
                else:
                    is_prev_or = False

        # Check if all column names are allowed
        if not all(col in allowed_filter_attributes for col in column_names):
            return False

        # For every column name in the WHERE clause, get its filter values
        for col in column_names:
            # Check if the column is numerical or categorical
            if col in original_dataset.select_dtypes(include='number').columns:
                filter_values = re.findall(rf"{col}\s*[<>]=?\s*(\d+)", where_clause)
                filter_values = [float(v) for v in filter_values]

                # Check if all filter values are between the min and max values of the column
                min_val = original_dataset[col].min()
                max_val = original_dataset[col].max()
                if not all(min_val <= int(val) <= max_val for val in filter_values):
                    return False
            else:
                # TODO: Add possibility of IN (list) in the WHERE clause for categorical columns
                if "IN" in where_clause:
                    try:
                        filter_values = re.findall(rf"[^\(]\"?{col}\"?\s*IN\s*\(((,?\s*\'[^\']+\',?\s*)+)\)[^\)]", where_clause)[0][0].split(',')
                        filter_values = [v.strip().strip("'") for v in filter_values]
                    except IndexError:
                        return False
                else:
                    if "OR" in where_clause:
                        try:
                            match_group = re.match(rf"\((\"?{col}\"?\s*=\s*'([^']+)'(\s+OR)*\s*)+\)", where_clause).group()
                        except AttributeError:
                            info_str = f"OR clause for column {col} must be formatted within parentheses"
                            return False
                    else:
                        match_group = where_clause
                    try:
                        filter_values = re.findall(rf"\"?{col}\"?\s*=\s*'([^']+)'", match_group)
                    except IndexError:
                        return False
                column_values = [v.strip() for v in original_dataset[col][original_dataset[col].notna()].unique()]
                if not all(val.strip() in column_values for val in filter_values):
                    return False

        return True
    return validate_query


def get_informative_query_validation_function(original_query: str, dataset: pd.DataFrame):

    query_pattern = get_query_pattern(original_query)
    allowed_filter_attributes = get_allowed_filter_attributes(original_query)
    original_dataset = dataset

    def validate_query_informative(query: str) -> Tuple[bool, str]:
        # Clean query
        query = query.replace('\n', ' ').replace('`', '"').strip()

        # Validate that the query fits the pattern
        if not re.fullmatch(query_pattern, query):
            info_str = f"Query does not match the pattern: {query_pattern}"
            return False, info_str

        # Extract the WHERE clause from the query
        where_clause = extract_where_clause(query)

        where_clause = where_clause.replace('`', '"')
        # Extract the column names from the WHERE clause
        attribute_matches = re.findall(r"(['\"]?)(\w+)\1(?=\s*(=|>|<|>=|<=|!=|NOT IN|IN|LIKE|IS))", where_clause, re.IGNORECASE)
        column_names = [match[1] for match in attribute_matches]

        clean_where_clause = where_clause.replace(')', ' ').replace('(', ' ')

        clean_column_names_map = {}

        # Check if all column names are allowed
        if not all(col in allowed_filter_attributes for col in column_names):
            invalid_columns = [col for col in column_names if col not in allowed_filter_attributes and col != "NOT"]
            info_str = f"The following attributes are not alterable: {invalid_columns}"
            return False, info_str


        for col in column_names:
            if col not in original_dataset.columns:
                info_str = f"Column {col} not in the original dataset"
                return False, info_str
            # for every column name replace all spaces with underscores
            clean_where_clause = clean_where_clause.replace(col, col.replace(' ', '_'))
            clean_column_names_map[col.replace(' ', '_')] = col


        is_prev_col = False
        is_prev_or = False
        prev_col = None
        for w in clean_where_clause.split(' '):
            if len(w) == 0:
                continue
            for c in clean_column_names_map.keys():
                if c in w:
                    w = clean_column_names_map[c]
                    break
            if w in column_names:
                is_prev_col = True
                prev_col = w
            elif is_prev_col:
                if w not in ['=', '>', '<', '>=', '<=', 'IN', ';']:
                    info_str = f"Operator {w} not allowed after column {prev_col}"
                    return False, info_str
                is_prev_col = False
            # Check all OR conditions are between the same column predicate and only categorical
            elif w == 'OR':
                if prev_col is None:
                    info_str = "OR condition without previous column predicate"
                    return False, info_str
                is_prev_or = True
                is_prev_col = False
                continue
            if is_prev_or:
                if w not in column_names or w != prev_col or w in original_dataset.select_dtypes(include='number').columns:
                    info_str = f"OR condition between different columns or numerical columns"
                    return False, info_str
                else:
                    is_prev_or = False

        # For every column name in the WHERE clause, get its filter values
        for col in column_names:
            # Check if the column is numerical or categorical
            if col in original_dataset.select_dtypes(include='number').columns:
                continue
                # filter_values = re.findall(rf"{col}\s*[<>]=?\s*(\d+)", where_clause)
                # filter_values = [float(v) for v in filter_values]
                #
                # # Check if all filter values are between the min and max values of the column
                # min_val = original_dataset[col].min()
                # max_val = original_dataset[col].max()
                # if not all(min_val <= int(val) <= max_val for val in filter_values):
                #     invalid_values = [val for val in filter_values if not min_val <= int(val) <= max_val]
                #     info_str = f"Filter values for {col} are not within the column range: {invalid_values}.\n" \
                #                f"Valid column range: {min_val} - {max_val}"
                #     return False, info_str
            else:
                # TODO: Add possibility of IN (list) in the WHERE clause for categorical columns
                if "IN" in where_clause:
                    try:
                        filter_values = re.findall(rf"\"?{col}\"?\s*IN\s*\(((,?\s*\'[^\']+\',?\s*)+)\)", where_clause)[0][0].split(',')
                        filter_values = [v.strip().strip("'") for v in filter_values]
                        first_in_index = where_clause.index("IN") + len("IN")
                        where_clause = where_clause[first_in_index:]
                    except IndexError:
                        info_str = f"IN clause not correctly formatted for column {col}"
                        return False, info_str
                else:
                    if "OR" in where_clause:
                        try:
                            match_group = re.match(rf"\((\"?{col}\"?\s*=\s*'([^']+)'(\s+OR)*\s*)+\)", where_clause).group()
                        except AttributeError:
                            info_str = f"OR clause for column {col} must be formatted within parentheses"
                            return False, info_str
                    else:
                        match_group = where_clause
                    try:
                        filter_values = re.findall(rf"\"?{col}\"?\s*=\s*'([^']+)'", match_group)
                    except IndexError:
                        info_str = f"Filter values not correctly formatted for column {col}"
                        return False, info_str
                column_values = [v.strip() for v in original_dataset[col][original_dataset[col].notna()].unique()]
                if not all(val.strip() in column_values for val in filter_values):
                    invalid_values = [val for val in filter_values if val.strip() not in column_values]
                    info_str = f"Filter values for {col} are not valid: {invalid_values}.\n" \
                               f"Valid column values: {column_values}"
                    return False, info_str
        sql_engine = SQLEngine(original_dataset, keep_index=True)
        query_but_with_name_of_dataset_being_df = f'SELECT * FROM df WHERE{query.split("WHERE")[1]}'
        query_results, _ = sql_engine.execute(query_but_with_name_of_dataset_being_df)
        if len(query_results) == 0:
            info_str = f"Query output size is less than 20 records"
            return False, info_str
        return True, ""
    return validate_query_informative


def get_allowed_filter_attributes(original_query: str):
    where_clause = extract_where_clause(original_query)
    return get_column_identifiers(where_clause)


###################################### Diverse Top-K Selection Validation Objective #####################################

def get_informative_top_k_validation_function(original_query: str, input_dataset: pd.DataFrame, k=10):
    sql_engine = SQLEngine(input_dataset, keep_index=True)
    validate_query_func_informative = get_informative_query_validation_function(original_query, input_dataset)
    valid_k = k

    def validate_top_k_informative(refined_query: str):
        """
        Validate the diversity of the refined query by checking if the output dataframe is within a certain percentage
        threshold of the original dataframe size and if the query is valid.
        :param refined_query:
        :return:
        """
        top_k_info_str = ""
        refined_query_output, _ = sql_engine.execute(refined_query)
        refined_df_size = len(refined_query_output)
        is_df_valid = refined_df_size >= valid_k
        is_query_valid, query_valid_info_str = validate_query_func_informative(refined_query)
        if not is_df_valid:
            top_k_info_str = f"Output dataframe size: {refined_df_size:,} " \
                                 f"is less than the minimal allowed size k: {valid_k}"
            if not is_query_valid:
                query_valid_info_str = f"Besides that, query is not valid: {query_valid_info_str}"
        return is_df_valid and is_query_valid, f"{top_k_info_str}\n{query_valid_info_str}"

    return validate_top_k_informative


###################################### Diverse Top-K Selection Distance Objective #####################################

def get_script_diff_func_sql(original_script, dataset):
    """
    Calculate the distance between two where clauses, where the distance is the sum
    of the distances between each pair of numerical and categorical columns.
    """

    original_where_clause = extract_where_clause(original_script)
    parsed_original = parse_where_clause(original_where_clause)

    def calc_diff(query):
        refined_where_clause = extract_where_clause(query)
        numeric_distance = 0
        categorical_distance = 0
        parsed_refined = parse_where_clause(refined_where_clause)
        # Calculate differences for categorical attributes
        for categorical_predicate in parsed_refined.categorical:
            for original_predicate in parsed_original.categorical:
                if categorical_predicate.name == original_predicate.name:
                    predicate_distance = categorical_refinement_distance(original_predicate, categorical_predicate)
                    categorical_distance += predicate_distance
        for numerical_predicate in parsed_refined.numerical:
            for original_predicate in parsed_original.numerical:
                refined_name, orig_name = numerical_predicate.name, original_predicate.name
                refined_op, orig_op = numerical_predicate.operator, original_predicate.operator
                if refined_name == orig_name and refined_op == orig_op:
                    refined_val, orig_val = numerical_predicate.value, original_predicate.value
                    predicate_distance = abs(refined_val - orig_val) / orig_val
                    numeric_distance += predicate_distance
        return categorical_distance + numeric_distance
    return calc_diff


###################################### Range Query Refinement Distance Objective ######################################
def get_range_query_distance_function(original_query: str, input_dataset: pd.DataFrame):
    sql_engine = SQLEngine(input_dataset, keep_index=True)
    original_query_output, _ = sql_engine.execute(original_query)
    original_query_output = original_query_output.set_index('index')

    def range_query_distance(refined_query: str):
        refined_query = extract_code(refined_query)
        refined_query_output, _ = sql_engine.execute(refined_query)
        refined_query_output = refined_query_output.set_index('index')
        # compute 1 - Jaccard Similarity between the output_dataset and the original_query_output
        union = len(set(refined_query_output.index.to_list()).union(set(original_query_output.index.to_list())))
        intersection = len(set(refined_query_output.index.to_list()).intersection(set(original_query_output.index.to_list())))
        return 1 - intersection / union

    return range_query_distance

############################################## Diversity Objectives ####################################################


def sort_by_distance(target_value, values):
    return sorted(values, key=lambda x: abs(x - target_value))

def get_pvl_dist_func(original_query: str, input_dataset: pd.DataFrame):
    """
    Calculate the distance between the original query and a refined query, where the distance is the sum
    of the distances between each pair of numerical and categorical columns.
    - For categorical columns, the distance is the number of values that are not present in the original query
    - For numerical columns, the distance is the index of the refined value in the sorted list of unique values
    :param original_query:
    :param input_dataset:
    :return:
    """
    sql_engine = SQLEngine(df=input_dataset)
    original_results, _ = sql_engine.execute(original_query)
    def calculate_pvl_distance(refined_query: str):
        results, _ = sql_engine.execute(refined_query)
        original_results_strs = [d["id"] for d in original_results.to_dict(orient='records')]
        query_results_strs = [d["id"] for d in results.to_dict(orient='records')]
        union = set(original_results_strs).union(query_results_strs)
        intersection = set(original_results_strs).intersection(query_results_strs)
        if len(union) == 0:
            return 9999999
        return round(1 - (len(intersection) / len(union)), ndigits=2)
    return calculate_pvl_distance


def get_diversity_validation_function(original_query: str, input_dataset: pd.DataFrame,
                                      minimum_df_size=20, maximum_df_size=100):
    sql_engine = SQLEngine(input_dataset, keep_index=True)
    validate_query_func = get_query_validation_function(original_query, input_dataset)

    def validate_diversity(refined_query: str):
        """
        Validate the diversity of the refined query by checking if the output dataframe is within a certain size range
        and if the query is valid.
        :param refined_query:
        :return:
        """
        refined_query_output, _ = sql_engine.execute(refined_query)
        refined_df_size = len(refined_query_output)
        is_df_valid = (minimum_df_size < refined_df_size <= maximum_df_size)
        is_query_valid = validate_query_func(refined_query)
        return is_df_valid and is_query_valid

    return validate_diversity


def get_informative_diversity_validation_function(original_query: str, input_dataset: pd.DataFrame,
                                                  minimum_df_size=20, maximum_df_size=100):
    sql_engine = SQLEngine(input_dataset, keep_index=True)
    validate_query_func_informative = get_informative_query_validation_function(original_query, input_dataset)

    def validate_diversity_informative(refined_query: str):
        """
        Validate the diversity of the refined query by checking if the output dataframe is within a certain percentage
        threshold of the original dataframe size and if the query is valid.
        :param refined_query:
        :return:
        """
        diversity_info_str = ""
        refined_query_output, _ = sql_engine.execute(refined_query)
        refined_df_size = len(refined_query_output)
        is_df_valid = minimum_df_size < refined_df_size <= maximum_df_size
        is_query_valid, query_valid_info_str = validate_query_func_informative(refined_query)
        if not is_df_valid:
            diversity_info_str = f"Output dataframe size: {refined_df_size:,} is not within the valid range: " \
                                    f"between {minimum_df_size} and {maximum_df_size} records"
            if not is_query_valid:
                query_valid_info_str = f"Besides that, query is not valid: {query_valid_info_str}"
        return is_df_valid and is_query_valid, f"{diversity_info_str}\n{query_valid_info_str}"

    return validate_diversity_informative

############################################ Constraint Objectives ##################################################

def get_constraint_evaluation_function(constraints: List[OutputConstraint], epsilon=0.2):

    def constraint_evaluation(output_dataset: DataFrame):
        try:
            output_evaluation = {str(constraint): constraint.evaluate(output_dataset) for constraint in constraints}
            dataset_eval_satisfies = all([score < epsilon for score in output_evaluation.values()])
            if dataset_eval_satisfies:
                return 0, None, False
            else:
                output_str = "\n\n".join([c for c in output_evaluation if output_evaluation[c] > epsilon])
                return sum([max(c - epsilon, 0) for c in output_evaluation.values()]) / len(output_evaluation), \
                    output_str, False
        except Exception as e:
            output_feedback = '\n'.join(traceback.format_exception(*sys.exc_info()))
            return 1, output_feedback, True

    return constraint_evaluation


def evaluate_constraints(d_out, constraint_list):
    sum_score = 0
    for constraint in constraint_list:
        sum_score += constraint.evaluate(d_out)
    return sum_score / len(constraint_list)


if __name__ == '__main__':
    original_query = """SELECT region, AVG(UGPA) as avg_gpa, AVG(LSAT) as avg_lsat, COUNT(*) as size FROM law_students
WHERE UGPA > 3.5 AND LSAT > 38
GROUP BY region;"""

    df = pd.read_csv("../datasets/law_students.csv")
    ref_query = """SELECT region, AVG(UGPA) as avg_gpa, AVG(LSAT) as avg_lsat, COUNT(*) as size FROM law_students
WHERE UGPA > 2.9 AND LSAT > 33
AND region NOT IN ('NE', 'GL', 'FW') 
GROUP BY region;
"""

    validate_query = get_informative_query_validation_function(original_query, df)
    print(validate_query(ref_query))

