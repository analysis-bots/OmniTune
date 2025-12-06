import re
import sys
import traceback
from typing import List, Tuple, Dict, Optional

import duckdb
import pandas as pd
import sqlparse
from sqlparse.sql import Identifier, Comparison, Where, Function
from sqlparse.tokens import Keyword, DML
from pandas import DataFrame
from functionality.constraint import OutputConstraint
from functionality.predicate import Predicate, NumericalPredicate, CategoricalPredicate, NumericalAttribute, \
    CategoricalAttribute
from tools.utils import extract_where_clause, get_column_values, parse_where_clause, \
    categorical_refinement_distance, get_query_pattern, extract_query, extract_having_clause


########################################## Common Query Validation Objectives ##########################################
def get_query_validation_function(original_query: str, dataset: pd.DataFrame):

    query_pattern = get_query_pattern(original_query)
    allowed_filter_attributes = get_allowed_filter_attributes(original_query)  # TODO - correct to use the map of column names and operators
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
        column_names = get_column_values(where_clause)  # TODO - correct to use the map of column names and operators

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
                            return False, info_str
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
        column_names = get_column_values(where_clause)

        clean_where_clause = where_clause.replace(')', ' ').replace('(', ' ')

        clean_column_names_map = {}

        for col in column_names.keys():
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

        # Check if all column names are allowed
        if not all(col in allowed_filter_attributes.keys() for col in column_names.keys()):
            invalid_columns = [col for col in column_names if col not in allowed_filter_attributes]
            info_str = f"The following columns are not alterable: {invalid_columns}"
            return False, info_str

        # For every column name in the WHERE clause, get its filter values
        for col in column_names.keys():
            # Check if the column is numerical or categorical
            if col in original_dataset.select_dtypes(include='number').columns:

                orig_ops = allowed_filter_attributes[col]
                refined_ops = column_names[col]
                for op in refined_ops:
                    if op not in orig_ops:
                        info_str = f"Operator {op} not allowed for column {col}"
                        return False, info_str

                filter_values = re.findall(rf"\"?{col}\"?\s*[<>]=?\s*(\d+)", where_clause)
                filter_values = [float(v) for v in filter_values]

                # Check if all filter values are between the min and max values of the column
                min_val = original_dataset[col].min()
                max_val = original_dataset[col].max()
                if not all(min_val <= int(val) <= max_val for val in filter_values):
                    invalid_values = [val for val in filter_values if not min_val <= int(val) <= max_val]
                    info_str = f"Filter values for {col} are not within the column range: {invalid_values}.\n" \
                               f"Valid column range: {min_val} - {max_val}"
                    return False, info_str
            else:
                # TODO: Add possibility of IN (list) in the WHERE clause for categorical columns
                if "IN" in where_clause:
                    try:
                        filter_values = re.findall(rf"\"?{col}\"?\s*IN\s*\(((,?\s*\'[^\']+\',?\s*)+)\)", where_clause)[0][0].split(',')
                        filter_values = [v.strip().strip("'") for v in filter_values]
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

        return True, ""
    return validate_query_informative


def get_allowed_filter_attributes(original_query: str):
    where_clause = extract_where_clause(original_query)
    return get_column_values(where_clause)


###################################### Diverse Top-K Selection Validation Objective #####################################

def get_informative_top_k_validation_function(original_query: str, input_dataset: pd.DataFrame, k=10):
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
        df = input_dataset
        refined_query_output = duckdb.query(refined_query).to_df()
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

# TODO - work for HAVING as well.
def get_script_diff_func_sql(original_script, dataset):
    """
    Calculate the distance between two where clauses, where the distance is the sum
    of the distances between each pair of numerical and categorical columns.
    """
    try:
        original_where_clause = extract_where_clause(original_script)
        parsed_original = parse_where_clause(original_where_clause)
    except Exception as e:
        original_where_clause = extract_having_clause(original_script)
        parsed_original = parse_where_clause(original_where_clause)


    def calc_diff(query):
        numeric_distance = 0
        categorical_distance = 0
        try:
            refined_where_clause = extract_where_clause(query)
            parsed_refined = parse_where_clause(refined_where_clause)
        except Exception as e:
            refined_where_clause = extract_having_clause(query)
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
                    if orig_val == 0:
                        try:
                            predicate_distance = abs(refined_val - orig_val) / refined_val
                        except ZeroDivisionError:
                            predicate_distance = 0
                    else:
                        predicate_distance = abs(refined_val - orig_val) / orig_val
                    numeric_distance += predicate_distance
        return categorical_distance + numeric_distance
    return calc_diff

def get_script_diff_func_sql_having(original_script, dataset):
    """
    Calculate the distance between two where clauses, where the distance is the sum
    of the distances between each pair of numerical and categorical columns.
    """

    original_where_clause = extract_having_clause(original_script)
    parsed_original = parse_where_clause(original_where_clause)

    def calc_diff(query):
        refined_where_clause = extract_having_clause(query)
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


###################################### HAVING Distance via Predicate Objects ######################################
def get_having_predicate_distance_function(original_query: str, refineable_predicates: List[Predicate]):
    """Return a distance function that measures differences in HAVING-clause predicates
    based on the provided refineable_predicates.

    For numerical predicates, distance is |refined - original| / max(|original|, 1e-9).
    If a BETWEEN clause is used, it is treated as two numerical predicates (>=, <=) as usual.
    Categorical HAVING is uncommon; categorical predicates are ignored by this distance.
    """
    where_dist_func = get_script_diff_func_sql(original_query, None)

    # Build a quick lookup of original numerical predicates by (attribute_name, operator)
    original_num_map: Dict[Tuple[str, str], float] = {}
    original_cat_map: Dict[str, List[str]] = {}
    for p in refineable_predicates:
        if isinstance(p, NumericalPredicate):
            original_num_map[(p.attribute.name, p.operator)] = float(p.value)
        elif isinstance(p, CategoricalPredicate):
            original_cat_map[p.attribute.name] = p.values

    def _extract_numeric_from_having(having_clause: str, attr: str, op: str) -> Optional[float]:
        if not having_clause:
            return None
        attr_escaped = re.escape(attr)
        # 1) Direct comparison: <attr> <op> <number>
        direct_pat = re.compile(rf"{attr_escaped}\s*{re.escape(op)}\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        m = direct_pat.search(having_clause)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        # 2) BETWEEN pattern: <attr> BETWEEN a AND b
        between_pat = re.compile(rf"{attr_escaped}\s+BETWEEN\s+([0-9]+(?:\.[0-9]+)?)\s+AND\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        m2 = between_pat.search(having_clause)
        if m2:
            lo, hi = m2.group(1), m2.group(2)
            try:
                lo_v, hi_v = float(lo), float(hi)
                if op in (">", ">="):
                    return lo_v
                if op in ("<", "<="):
                    return hi_v
            except Exception:
                return None
        return None

    # Cache the original HAVING for potential future logic (not used directly here)
    _ = extract_having_clause(original_query)

    def calc_diff(refined_query: str) -> float:
        refined_having = extract_having_clause(refined_query) or ""
        where_clause_distance = where_dist_func(refined_query)
        total_distance = 0.0
        for (attr, op), orig_val in original_num_map.items():
            refined_val = _extract_numeric_from_having(refined_having, attr, op)
            if refined_val is None:
                continue
            denom = abs(orig_val) if abs(orig_val) > 1e-9 else 1.0
            total_distance += abs(refined_val - orig_val) / denom
        return float(total_distance / len(original_num_map)) + where_clause_distance

    return calc_diff


###################################### Range Query Refinement Distance Objective ######################################
def get_range_query_distance_function(original_query: str, input_dataset: pd.DataFrame):
    df = input_dataset
    con = duckdb.connect()

    # 1) Register the DF with its index materialized as a column
    df_with_idx = df.reset_index()  # keeps MultiIndex too
    # name the index column(s) so you can recognize them later
    df_with_idx = df_with_idx.rename(columns=lambda c: "_idx" if c == "index" else c)

    con.register("df", df_with_idx)

    original_query_output = con.sql(original_query).to_df()
    original_query_output.set_index("_idx", inplace=True)
    def range_query_distance(refined_query: str):
        df = input_dataset
        df_with_idx = df.reset_index()  # keeps MultiIndex too
        # name the index column(s) so you can recognize them later
        df_with_idx = df_with_idx.rename(columns=lambda c: "_idx" if c == "index" else c)

        con.register("df", df_with_idx)

        refined_query = extract_query(refined_query)
        refined_query_output = con.sql(refined_query).to_df()
        refined_query_output.set_index("_idx", inplace=True)
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
    df = input_dataset
    original_results = duckdb.query(original_query).to_df()
    def calculate_pvl_distance(refined_query: str):
        results = duckdb.query(refined_query).to_df()
        original_results_strs = [d["id"] for d in original_results.to_dict(orient='records')]
        query_results_strs = [d["id"] for d in results.to_dict(orient='records')]
        union = set(original_results_strs).union(query_results_strs)
        intersection = set(original_results_strs).intersection(query_results_strs)
        if len(union) == 0:
            return 9999999
        return round(1 - (len(intersection) / len(union)), ndigits=2)
    return calculate_pvl_distance


def get_diversity_distance_function(original_query: str, input_dataset: pd.DataFrame,
                                    predicate_weights: Optional[Dict[str, float]]=None):
    """
    Calculate the distance between the original query and a refined query, where the distance is determined by the
    weighted sum of the distances between each pair of numerical and categorical columns.
    :param original_query:
    :param input_dataset:
    :param predicate_weights: a dictionary containing the weights for each predicate, sum of weights should be 1
    :return:
    """
    original_where_clause = extract_where_clause(original_query)
    parsed_original = parse_where_clause(original_where_clause)

    if predicate_weights is None:
        num__predicates = len(parsed_original.numerical) + len(parsed_original.categorical)
        predicate_weights = {}
        for categorical_predicate in parsed_original.categorical:
            predicate_weights[categorical_predicate.name] = 1 / num__predicates
        for numerical_predicate in parsed_original.numerical:
            predicate_weights[numerical_predicate.name] = 1 / num__predicates

    def calc_diff(query):
        refined_where_clause = extract_where_clause(query)
        numeric_distance = 0
        categorical_distance = 0
        parsed_refined = parse_where_clause(refined_where_clause)
        # Calculate differences for categorical attributes
        for categorical_predicate in parsed_refined.categorical:
            if predicate_weights.get(categorical_predicate.name) == 0:
                continue
            w_predicate = predicate_weights.get(categorical_predicate.name)
            for original_predicate in parsed_original.categorical:
                if categorical_predicate.name == original_predicate.name:
                    predicate_distance = categorical_refinement_distance(original_predicate, categorical_predicate)
                    categorical_distance += w_predicate * predicate_distance
        for numerical_predicate in parsed_refined.numerical:
            if predicate_weights.get(numerical_predicate.name) == 0:
                continue
            w_predicate = predicate_weights.get(numerical_predicate.name)
            for original_predicate in parsed_original.numerical:
                refined_name, orig_name = numerical_predicate.name, original_predicate.name
                refined_op, orig_op = numerical_predicate.operator, original_predicate.operator
                if refined_name == orig_name and refined_op == orig_op:
                    refined_val, orig_val = numerical_predicate.value, original_predicate.value
                    predicate_distance = abs(refined_val - orig_val) / orig_val
                    numeric_distance += w_predicate * predicate_distance
        return categorical_distance + numeric_distance
    return calc_diff


def get_diversity_validation_function(original_query: str, input_dataset: pd.DataFrame,
                                      minimum_df_size=20, maximum_df_size=100):
    validate_query_func = get_query_validation_function(original_query, input_dataset)

    def validate_diversity(refined_query: str):
        """
        Validate the diversity of the refined query by checking if the output dataframe is within a certain size range
        and if the query is valid.
        :param refined_query:
        :return:
        """
        df = input_dataset
        refined_query_output = duckdb.query(refined_query).to_df()
        refined_df_size = len(refined_query_output)
        is_df_valid = (minimum_df_size < refined_df_size <= maximum_df_size)
        is_query_valid = validate_query_func(refined_query)
        return is_df_valid and is_query_valid

    return validate_diversity


def get_informative_diversity_validation_function(original_query: str, input_dataset: pd.DataFrame):
    validate_query_func_informative = get_informative_query_validation_function(original_query, input_dataset)

    def validate_diversity_informative(refined_query: str):
        """
        Validate the diversity of the refined query by checking if the output dataframe is within a certain percentage
        threshold of the original dataframe size and if the query is valid.
        :param refined_query:
        :return:
        """
        diversity_info_str = ""
        df = input_dataset
        refined_query_output = duckdb.query(refined_query).to_df()
        refined_df_size = len(refined_query_output)
        is_df_valid = refined_df_size > 0
        is_query_valid, query_valid_info_str = validate_query_func_informative(refined_query)
        if not is_df_valid:
            diversity_info_str = f"Output dataframe size is 0"
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


def get_basic_refinement_validation_function(
        original_query: str, refineable_predicates: List[Predicate]):
    """Return a simple validation function for refined queries.

    The returned function validates that a given query only modifies
    predicates that appear in ``refineable_predicates`` and that the
    structure of the SQL query matches ``original_query``.  If the query
    is valid the function returns ``(True, "")`` otherwise it returns a
    short string explaining why the query is invalid.
    """

    normalized_original = original_query.replace('\n', ' ').replace('`', '"').strip()
    original_where = extract_where_clause(normalized_original)
    if original_where is None:
        query_pattern = re.escape(normalized_original)
        query_pattern = query_pattern.replace('\n', '\\s+').replace(' ', '\\s*')
    else:
        query_pattern = get_query_pattern(normalized_original)

    query_pattern = query_pattern.strip(";")

    allowed_ops_map: Dict[str, set] = {}
    for pred in refineable_predicates:
        attr_name = pred.attribute.name
        if isinstance(pred, NumericalPredicate):
            allowed_ops_map.setdefault(attr_name, set()).add(pred.operator)
        else:
            allowed_ops_map.setdefault(attr_name, set()).add('=')

    original_had_where = original_where is not None

    def validate(query: str) -> Tuple[bool, str]:
        # Normalize query and make sure the overall structure didn't change
        clean_query = query.replace('\n', ' ').replace('`', '"').strip().replace(';', '')
        if not re.fullmatch(query_pattern, clean_query):
            return False, "query does not match original pattern"

        where_clause = extract_where_clause(clean_query)
        if original_had_where:
            if where_clause is None:
                return False, "WHERE clause removed"
            parsed = parse_where_clause(where_clause)

            for p in parsed.numerical:
                if p.name not in allowed_ops_map:
                    return False, f"predicate {p.name} is not refineable"
                if p.operator not in allowed_ops_map[p.name]:
                    alternative_operator = p.operator.replace('=', '') if '=' in p.operator else f"{p.operator}="
                    return False, (
                        f"operator '{p.operator}' for attribute '{p.name}' is not refineable. did you mean '{alternative_operator}' ?")

            for p in parsed.categorical:
                if p.name not in allowed_ops_map:
                    return False, f"predicate {p.name} is not refineable"
        else:
            if where_clause is not None:
                return False, "WHERE clause added"

        return True, ""

    return validate

def test_get_range_query_distance_function():
    ORIGINAL_QUERY = """SELECT * FROM df
WHERE "region_first" = 'PO' AND "UGPA" >= 3.0 AND "UGPA" <= 3.5
ORDER BY "LSAT" DESC;
"""

    REFINED_QUERY = """SELECT * FROM df
WHERE "region_first" = 'Mt' AND "UGPA" >= 2.8 AND "UGPA" <= 3.3
ORDER BY "LSAT" DESC;
"""

    # Create a dummy DataFrame
    data = {
        'region_first': ['PO', 'Mt', 'PO', 'Mt'],
        'UGPA': [3.1, 2.9, 3.4, 3.2],
        'LSAT': [160, 155, 158, 162]
    }
    df = pd.DataFrame(data)
    # Create the distance function
    distance_function = get_range_query_distance_function(ORIGINAL_QUERY, df)
    # Calculate the distance
    distance = distance_function(REFINED_QUERY)
    print(f"Distance between the original and refined query: {distance}")
    assert distance > 0.5, f"Distance should be greater than 0.5, got {distance}"

def test_validation():
    """
    Tests the following scenario:
    -- Original Query
    SELECT * FROM df
    WHERE "Transaction_Amount" >= 400 AND "Transaction_Amount" <= 800

    -- Invalid Refined Query (with non-refineable "Device_Type" attribute)
    SELECT * FROM df
    WHERE "Transaction_Amount" >= 440 AND "Transaction_Amount" <= 780 AND "Device_Type" = 'Mobile';
    :return:
    """
    original_query = """SELECT * FROM df
WHERE "Transaction_Amount" >= 400 AND "Transaction_Amount" <= 800"""
    # Create a dummy DataFrame that does NOT contain "Device_Type" but has "Transaction_Amount"
    refineable_attribute = NumericalAttribute("Transaction_Amount", min_value=0, max_value=1000, step=10)
    refineable_predicates = [
        NumericalPredicate(refineable_attribute, ">=", 400),
        NumericalPredicate(refineable_attribute, "<=", 800)
    ]
    validate_query = get_basic_refinement_validation_function(original_query, refineable_predicates)

    refined_query_invalid = """SELECT * FROM df
WHERE "Transaction_Amount" >= 440 AND "Transaction_Amount" <= 780 AND "Device_Type" = 'Mobile';"""
    is_valid, info_str = validate_query(refined_query_invalid)
    print(f"Test 1 - Invalid (Device_Type): Valid={is_valid}, Info='{info_str}'")
    assert not is_valid, f"Refined query should be invalid due to Device_Type: {info_str}"

    refined_query_valid_modification = """SELECT * FROM df
WHERE "Transaction_Amount" >= 450 AND "Transaction_Amount" <= 750"""
    is_valid, info_str = validate_query(refined_query_valid_modification)
    print(f"Test 2 - Valid (Transaction_Amount only): Valid={is_valid}, Info='{info_str}'")
    assert is_valid, f"Refined query with only Transaction_Amount should be valid: {info_str}"

    # Test case: Original query with no WHERE clause
    original_query_no_where = """SELECT * FROM df"""
    refineable_attr_y = CategoricalAttribute("Y", categories=['a', 'b', 'c'])
    refineable_preds_y = [CategoricalPredicate(refineable_attr_y, values=['a'])]
    validate_no_where = get_basic_refinement_validation_function(original_query_no_where, refineable_preds_y)

    refined_adds_where_for_refineable = """SELECT * FROM df WHERE "Y" = 'a'"""
    is_valid, info_str = validate_no_where(refined_adds_where_for_refineable)
    print(f"Test 3 - Refined adds WHERE for refineable: Valid={is_valid}, Info='{info_str}'")
    # This should be False because refined query added a WHERE clause original didn't have.
    assert not is_valid, f"Refined query adding WHERE to a no-WHERE original should be invalid: {info_str}"

    original_query_with_having = """SELECT region, AVG(charges) FROM df GROUP BY region HAVING AVG(charges) > 10000"""
    # This test requires the pattern to allow HAVING. Let's make refineable_predicates empty to test adding non-refineable.
    validate_having = get_basic_refinement_validation_function(original_query_with_having, [])

    refined_having_adds_non_refineable_attr = """SELECT region, AVG(charges) FROM df GROUP BY region HAVING AVG(charges) > 10000 AND age > 30"""
    is_valid, info_str = validate_having(refined_having_adds_non_refineable_attr)
    print(f"Test 4 - Refined HAVING adds non-refineable: Valid={is_valid}, Info='{info_str}'")
    assert not is_valid, f"Refined HAVING adding non-refineable attribute 'age' should be invalid: {info_str}"
    original_query_range = """SELECT * FROM df
WHERE "ANNUAL" > 50000 AND "ANNUAL" < 125000;
"""
    refineable_attribute = NumericalAttribute("ANNUAL", min_value=0, max_value=200000, step=1000)
    refineable_predicates = [
        NumericalPredicate(refineable_attribute, ">", 50000),
        NumericalPredicate(refineable_attribute, "<", 125000)
    ]
    # This test requires the pattern to allow HAVING. Let's make refineable_predicates empty to test adding non-refineable.
    validate_range = get_basic_refinement_validation_function(original_query_range, refineable_predicates)

    refined_query_range = """SELECT * FROM df WHERE "ANNUAL" > 60000 AND "ANNUAL" < 115000;
"""
    is_valid, info_str = validate_range(refined_query_range)
    print(f"Test 5 - Refined range query 'ANNUAL': Valid={is_valid}, Info='{info_str}'")
    assert is_valid, ""

    print("Extended validation tests finished.")


if __name__ == '__main__':
    T1a_ORIGINAL_SCRIPT_SQL = """
    SELECT * FROM df
    WHERE "Graduate Major" = 'Aeronautics & Astronautics' AND "Space Walks" >= 8 AND "Space Walks" <= 9
    ORDER BY "Space Flight (hr)" DESC;
    """

    diff_func = get_script_diff_func_sql(T1a_ORIGINAL_SCRIPT_SQL, pd.DataFrame())

    T1a_REFINED_SCRIPT_SQL = """
    SELECT * FROM df
    WHERE "Graduate Major" IN ('Aeronautics & Astronautics', 'Medicine', 'Public Health')
      AND "Space Walks" >= 8 AND "Space Walks" <= 10
    ORDER BY "Space Flight (hr)" DESC;
    """

    distance = diff_func(T1a_REFINED_SCRIPT_SQL)
    print(f"Distance between the original and refined query: {distance}")