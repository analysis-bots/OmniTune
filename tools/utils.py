import traceback
from collections import Counter
from itertools import product
from typing import Dict, Literal, List, Union, Any, Tuple, Optional

import duckdb

import numpy as np
import pandas as pd
import sqlglot
import sqlparse

import re
import pandasql as ps
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
import io
import sys

from functionality.where_clause import CategoricalPredicate, WhereClause, NumericalPredicate

MATCH_STR = "((([\t\s]*\(df\['\w+\s*[\w_\d]*'\]\.isin\(\[.*?\]\)\)\s*&*\s*[\n]*)|([\t\s]*\(df\['\w+\s*[\w_\d]*'\]\s*[<>]=?\s*[-+]?\d\.*\d*\s*\)\s*&*\s*[\n]*))+)"


def extract_query(text, language="sql"):
    # Define the start and end markers
    start_marker = f"```{language}"
    end_marker = "```"
    # Find the start and end positions
    start_pos = text.rfind(start_marker)
    if start_pos == -1:
        start_marker = "\n\n"
        start_pos = text.rfind(start_marker) - 1
        end_pos = len(text)
    else:
        end_pos = text.rfind(end_marker, start_pos + len(start_marker))
        if end_pos == -1:
            return ""

    # Extract the content in between
    content = text[start_pos + len(start_marker):end_pos].strip()

    return content


def execute_operation(operation_str, d_in, language="python", function_name=None):
    """
    Executes the given operation string on the input dataframes and returns the result.
    :param operation_str: The operation string to execute
    :param d_in: A dictionary of dataframes, where the keys are the names of the dataframes
    :param language:
    :return: The result of the operation as a dataframe
    """
    is_exception = False
    if language == "python":
        if "# Example usage" in operation_str:
            operation_str = operation_str.split("# Example usage")[0]
        elif "# Usage example" in operation_str:
            operation_str = operation_str.split("# Usage example")[0]
        env_vars = {'d_in': d_in}
        import_lines = []
        code_lines = []

        for line in operation_str.split("\n"):
            if "import" in line:
                import_lines.append(line)
            else:
                code_lines.append(line)

        # Execute import lines first
        for line in import_lines:
            exec(line.strip(), env_vars, env_vars)

        # Execute the rest of the dev
        code_to_execute = "\n".join(code_lines)
        try:
            exec(code_to_execute, env_vars, env_vars)
            result = eval(f"{function_name}(d_in)", env_vars, env_vars)
        except Exception as e:
            is_exception = True
            result = '\n'.join(traceback.format_exception(*sys.exc_info()))
        return result, is_exception
    elif language == "sql":
        for k, v in d_in.items():
            locals()[k] = v
        return ps.sqldf(operation_str, locals()), is_exception


def get_info_string(df, column_names):
    """
    Returns a string containing information about the given DataFrame.
    :param df: The DataFrame to get information about
    :param column_names: The names of the columns to get information about
    :return: A string containing information about the DataFrame
    """
    info_string = ""
    for column_name in column_names:
        col = df[column_name]
        if is_numeric_dtype(col):
            min_max_tuple = [col.min(), col.max()]
            info_string += f"   - Numerical ({col.dtype}) column '{column_name}' value range: {min_max_tuple}\n"
        else:
            unique_vals = set(col.unique())
            info_string += f"   - Categorical column '{column_name}' unique values: {unique_vals}\n"
    return info_string


def get_info_string_for_df(df):
    """
    Returns a string containing information about the given DataFrame.
    :param df: The DataFrame to get information about
    :param column_names: The names of the columns to get information about
    :return: A string containing information about the DataFrame
    """
    info_string = ""
    column_names = df.columns
    for column_name in column_names:
        col = df[column_name]
        if is_numeric_dtype(col):
            min_max_tuple = [col.min(), col.max()]
            info_string += f"   - Numerical ({col.dtype}) column '{column_name}' value range: {min_max_tuple}\n"
        else:
            unique_vals = set(col.unique())
            info_string += f"   - Categorical column '{column_name}' unique values: {unique_vals}\n"
    return info_string


def get_dataset_info_string(dataset, column_names):
    info_strings = {}
    for k, df in dataset.items():
        info_strings[k] = f"{get_info_string(df, column_names)}\n"
    return info_strings


def get_d_in_info(d_in, column_names):
    d_in_info_strings = get_dataset_info_string(d_in, column_names)
    d_in_info = ""
    for k, df in d_in.items():
        buf = io.StringIO()
        df.info(buf=buf)
        info_lines = buf.getvalue().splitlines()
        info_str = '\n      '.join(info_lines[1:])
        # d_in_info += f" - DataFrame {k} \n    - Info:\n      {info_str}\n\n    - Values:\n{d_in_info_strings[k]}\n"
        d_in_info += f" - DataFrame {k}:\n{d_in_info_strings[k]}\n"
    return d_in_info


def parse_script(script):
    match = re.findall(MATCH_STR, script)[0]
    return match[0]


def parse_line(line) -> (Dict[str, set], Dict[str, list]):
    """ Extracts attributes and their conditions from the given line. """
    # Match categorical conditions
    cat_pattern = r"'(\w+\s*\w*)'\s*=\s*('(.*?)')"
    # Match numerical conditions
    num_pattern = r"'(\w+\s*\w*)'\s*([<>]=?)\s*([+-]?[\d.]+)"

    cat_conditions = re.findall(cat_pattern, line)
    num_conditions = re.findall(num_pattern, line)

    # Process categorical conditions to extract sets of values
    cat_data = {attr: set(eval(f"[{values}]")) for attr, values in cat_conditions}
    # Process numerical conditions to extract operators and thresholds
    num_data = {}
    for attr, op, value in num_conditions:
        if attr not in num_data.keys():
            num_data[attr] = []
        num_data[attr].append((op, float(value)))

    return cat_data, num_data


def create_range_query_code_pattern(filter_attributes):
    allowed_attributes = list(set(attr[0] for attr in filter_attributes))
    pattern_line = "\s*filtered_data\s*=\s*df\[(\s*\(df\['("
    pattern_line += "|".join([attr for attr in allowed_attributes])
    pattern_line += f")'\]\s*([<>]=?)\s*([-*\d.]+)\s*\)\s*&*\s*)+\]\s*"
    return pattern_line


def remove_comments(script: str) -> str:
    # Remove single-line comments
    script_no_single_comments = re.sub(r'#.*', '', script)
    # Remove multi-line comments
    script_no_comments = re.sub(r'\"\"\".*?\"\"\"', '', script_no_single_comments, flags=re.DOTALL)
    return script_no_comments


def extract_code_parts_and_attributes(original_script: str):
    # Remove comments from the original script
    script_no_comments = remove_comments(original_script)

    # Find the prefix up to the line before the first assignment to filtered_data
    prefix_match = re.search(r"(.*?\n)\s+filtered_data\s+=\s+df\[", script_no_comments, re.DOTALL)
    code_prefix = prefix_match.group(1).strip() if prefix_match else ""

    # Find the full assignment line for filtered_data
    alterable_line_match = re.search(r"(\s+filtered_data\s+=\s+df\[.*?\]\n)", script_no_comments, re.DOTALL)
    alterable_line = alterable_line_match.group(1).strip() if alterable_line_match else ""

    # Find the filter attributes and conditions
    filter_match = re.search(r"\s+filtered_data\s+=\s+df\[(.*?)\]\n", script_no_comments, re.DOTALL)
    filter_conditions = filter_match.group(1) if filter_match else ""

    # Extract the filter attributes
    filter_attributes = re.findall(r"df\['(\w+ *\w*)'\]", filter_conditions)
    categorical_filter_attributes = re.findall(r"df\['(\w+ *\w*)'\].isin", filter_conditions)
    numerical_filter_attributes = re.findall(r"df\['(\w+ *\w*)'\] (.*?) \s*-*\d+\.*\d*", filter_conditions)
    filter_attributes = [(c,) for c in categorical_filter_attributes] + numerical_filter_attributes

    # Find the suffix starting from the line after the filtered_data assignment
    suffix_start = alterable_line_match.end() if alterable_line_match else 0
    code_suffix = script_no_comments[suffix_start:]

    return code_prefix, alterable_line, code_suffix, filter_attributes


def format_top_k_refinement_code(code_prefix, alterable_line, code_suffix):
    # Extract the column names from the alterable line
    columns = re.findall(r"df\['(\w+)'\]", alterable_line)

    # Replace specific filter values and threshold with placeholders
    formatted_line = alterable_line
    for column in columns:
        formatted_line = re.sub(rf"\(df\['{column}'\]\.isin\(\['[^']*'\]\)\)",
                                f"(df(['{column}']).isin((['<filter_value_1>', ..., '<filter_value_n>'])))",
                                formatted_line)

    formatted_line = re.sub(r"(=\s*\d+(\.\d+)?)", "= <threshold>", formatted_line)
    code_prefix = '\n'.join([line.strip() for line in code_prefix.split('\n')])

    code_suffix = "\n\n    ".join([line.strip() for line in code_suffix.split("\n\n")])
    # Combine the parts to form the final dev format
    code_format = f"{code_prefix.strip()}\n\n    {formatted_line.strip()}\n\n    {code_suffix.strip()}"
    return code_format


def format_range_query_code(code_prefix, alterable_line, code_suffix):
    # Extract the column names from the alterable line
    columns = re.findall(r"df\['(\w+\s*\w*)'\]", alterable_line)
    cols = []
    for c in columns:
        if c not in cols:
            cols.append(c)

    # Replace specific filter values and threshold with placeholders
    formatted_line = "filtered_data = df["
    formatted_list = []
    for column in cols:
        formatted_list.append(f"(df['{column}'] >= <min_threshold>) & (df['{column}'] <= <max_threshold>)")
    formatted_line += " & ".join(formatted_list) + "]"

    # Combine the parts to form the final dev format
    code_format = f"{code_prefix}\n    {formatted_line.strip()}\n    {code_suffix}"
    return code_format


def create_code_pattern(filter_attributes, df):
    numerical_filter_attributes = [attr for attr in filter_attributes if is_numeric_dtype(df[attr[0]])]
    categorical_filter_attributes = [attr[0] for attr in filter_attributes if not is_numeric_dtype(df[attr[0]])]
    pattern_line = "\s*filtered_data\s*=\s*df\["
    pattern_parts = []
    for attr in categorical_filter_attributes:
        pattern_parts.append(f"\(df\['{attr}'\]\.isin\(\[('[^']*',*\s*)+\]\)\)")
    for attr, op in numerical_filter_attributes:
        pattern_parts.append(f"\(df\['{attr}'\]\s*[<>]=\s*([+-]?\d+(\.\d+)*)\)")
    pattern_line += "\s*&\s*".join(pattern_parts)
    pattern_line += "\s*\]"
    return pattern_line


def extract_filter_conditions(function_string, df):
    # Define the regex pattern to match the filtering condition
    pattern = r"df\['(?P<attribute>\w+)'\]\s*([<>]=?)\s*(?P<value>[-+]?\d*\.\d+|\d+|float\('-inf'\)|float\('inf'\))"

    # Extract matches from the function string
    matches = re.findall(pattern, function_string)

    # Initialize a dictionary to store the min and max values for each attribute
    filter_dict = {}

    for match in matches:
        attribute = match[0]
        operator = match[1]
        value = float(match[2])

        if attribute not in filter_dict:
            filter_dict[attribute] = (None, None)

        # Determine if it's a min or max condition
        if operator == ">=":
            filter_dict[attribute] = (value, filter_dict[attribute][1])
        elif operator == "<=":
            filter_dict[attribute] = (filter_dict[attribute][0], value)
        elif operator == ">":
            filter_dict[attribute] = (
                value + 1e-9, filter_dict[attribute][1])  # small epsilon for strict inequality
        elif operator == "<":
            filter_dict[attribute] = (
                filter_dict[attribute][0], value - 1e-9)  # small epsilon for strict inequality

    # Replace None values with the actual min/max values from the DataFrame
    for attribute, (min_val, max_val) in filter_dict.items():
        if min_val is None:
            filter_dict[attribute] = (df[attribute].min(), max_val)
        if max_val is None:
            filter_dict[attribute] = (min_val, df[attribute].max())

    return filter_dict


def split_code_lines(code):
    split_lines = code.split('\n')
    code_lines = []
    current_line = ""
    inside_continuation = False

    for line in split_lines:
        stripped_line = line.strip()

        # Skip comments, empty lines, and import predicates
        if stripped_line.startswith('#') or not stripped_line or "import" in stripped_line:
            continue

        # Check if the line is part of a multi-line predicate
        if stripped_line.endswith(('&', '|', '\\')) or inside_continuation:
            # Remove any trailing continuation characters
            current_line += stripped_line.rstrip('\\')
            inside_continuation = True

            # Check if the continuation ends
            if line.rstrip().endswith((')', ']', '}')):
                code_lines.append(current_line)
                current_line = ""
                inside_continuation = False
        else:
            # If not a continuation, add the line as is
            if current_line:
                code_lines.append(current_line)
                current_line = ""
            code_lines.append(line)

    # Add the last accumulated line if there's any
    if current_line:
        code_lines.append(current_line)

    return code_lines


def parse_where_clause(where_clause):
    where_clause = where_clause.replace('`', '\"').replace('\n', ' ').replace('\'', '\"')
    # Define regular expressions for matching predicates
    numerical_pattern = re.compile(r'(\"?[^\"]+\"?|\w+)\s*(<=|>=|<|>)\s*([\d]+[\.\d]*)')
    equality_pattern = re.compile(r'(\"?[^\"]+\"?|\w+)\s*=\s*\"?([^\"]+)\"?')
    in_pattern = re.compile(r'(\"?[^\"]+\"?|\w+)\s*IN\s*\(((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*?)\)')

    # Split the WHERE clause by AND
    predicates = where_clause.split(' AND ')

    # result = WhereClause()
    numerical_predicates = []
    categorical_predicates = []
    for predicate in predicates:
        predicate = predicate.strip()

        # Check for numerical predicates
        numerical_match = numerical_pattern.match(predicate)
        if numerical_match:
            attribute_name, operator, value = numerical_match.groups()
            attribute_name = attribute_name.strip('"').strip('`').strip()
            numerical_predicate = NumericalPredicate(attribute_name, operator.strip(), float(value))
            numerical_predicates.append(numerical_predicate)
            continue

        # Check for categorical predicates using equality
        equality_matches = re.findall(equality_pattern, predicate)
        if equality_matches:
            attribute_name = equality_matches[0][0].strip('"').strip()
            values = [match[1] for match in equality_matches]
            categorical_predicates.append(CategoricalPredicate(attribute_name, values))
            continue

        # Check for categorical predicates using IN clause
        in_match = in_pattern.match(predicate)
        if in_match:
            attribute_name, values_str = in_match.groups()
            attribute_name = attribute_name.strip("'").strip('"').strip()
            values = [v.strip().strip("'").strip('"') for v in values_str.split(',')]
            categorical_predicates.append(CategoricalPredicate(attribute_name, values))
            continue

    return WhereClause(numerical_predicates, categorical_predicates)


def get_alterable_attributes(where_clause):
    parsed_where_clause = parse_where_clause(where_clause)
    numerical_attributes = [p.name for p in parsed_where_clause.numerical]
    categorical_attributes = [p.name for p in parsed_where_clause.categorical]
    return {'numeric': numerical_attributes, 'categorical': categorical_attributes}


def parse_where_conditions(sql_query):
    # Parse the SQL query into an AST
    expression = sqlglot.parse_one(sql_query)

    # Initialize the result dictionary
    result = {'categorical': [], 'numerical': []}

    # Traverse the AST to find the WHERE clause
    where_clause = expression.find(sqlglot.exp.Where)

    if where_clause:
        # Split the WHERE clause into individual conditions
        conditions = where_clause.args['this'].flatten()

        for condition in conditions:
            if isinstance(condition, sqlglot.expressions.In):
                # Handle categorical condition (IN clause)
                attribute_name = condition.args['this'].sql()
                in_values = [v.sql().strip("'\"") for v in condition.args['expressions']]
                result['categorical'].append({
                    'attribute_name': attribute_name,
                    'in_values': in_values
                })
            elif isinstance(condition, sqlglot.expressions.Condition):
                # Handle numerical conditions (comparison)
                attribute_name = condition.left.sql()
                operator = condition.sql().split(' ')[1]
                value = condition.right.sql().strip("'\"")

                low_val = high_val = None
                if operator in ['>', '>=']:
                    low_val = int(value) if int(value) == float(value) else float(value)
                elif operator in ['<', '<=']:
                    high_val = int(value) if int(value) == float(value) else float(value)

                result['numerical'].append({
                    'attribute_name': attribute_name,
                    'range': [low_val, high_val]
                })

    return result

def match_query(extracted_query):
    query_pattern = re.compile(r"SELECT[\s\n]+[\w\s,.*()]+[\s\n]+FROM[\s\n]+\w+[\s\n]+WHERE([\s\n]+[\s\S]*)+")
    match = query_pattern.match(extracted_query.strip())
    return match


def extract_where_clause(sql_query):
    # Define the regex pattern to match the WHERE clause
    where_clause_pattern = re.compile(r'\bWHERE\b(.*?)(?:\bORDER\b|\bGROUP\b|\bLIMIT\b|\bUNION\b|$)',
                                      re.IGNORECASE | re.DOTALL)

    # Search for the WHERE clause in the SQL query
    match = where_clause_pattern.search(sql_query)

    # Return the matched WHERE clause or None if not found
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_having_clause(sql_query):
    # Define the regex pattern to match the WHERE clause
    where_clause_pattern = re.compile(r'\bHAVING\b(.*?)(?:\bORDER\b|\bGROUP\b|\bLIMIT\b|\bUNION\b|$)',
                                      re.IGNORECASE | re.DOTALL)

    # Search for the WHERE clause in the SQL query
    match = where_clause_pattern.search(sql_query)

    # Return the matched WHERE clause or None if not found
    if match:
        return match.group(1).strip()
    else:
        return None


def numeric_refinement_distance(a, b, c, d, half_mean_interval=0.5):
    # Address edge cases
    if a == b:
        a -= half_mean_interval
        b += half_mean_interval
    elif c == d:
        c -= half_mean_interval
        d += half_mean_interval
    elif a == d:
        a -= half_mean_interval
        d += half_mean_interval
    elif b == c:
        b += half_mean_interval
        c -= half_mean_interval

    # Calculate intersection
    intersection_length = max(0, min(b, d) - max(a, c))

    # Calculate union
    union_length = max(b, d) - min(a, c)

    # If union length is zero (which shouldn't happen with valid intervals), return 0
    if union_length == 0:
        return 0

    # Compute Jaccard similarity
    jaccard_similarity = intersection_length / union_length

    return 1 - jaccard_similarity


def get_half_mean_interval(df, attribute_name):
    sorted_vals = sorted(df[attribute_name].unique().tolist())[1:-1]
    return np.round(np.mean(np.diff(sorted_vals)), 2) / 2


def categorical_refinement_distance(original_predicate, refined_predicate):
    """
    Jaccard Distance = 1 - Jaccard Similarity
    :param original_predicate:
    :param refined_predicate:
    :return:
    """
    original_values = set(original_predicate.values)
    refined_values = set(refined_predicate.values)
    intersection = original_values.intersection(refined_values)
    union = original_values.union(refined_values)
    return (1 - (len(intersection) / len(union))) if union else (1 if intersection else 0)


def get_interval(predicates, max_val, min_val):
    lower_bound = min_val
    upper_bound = max_val
    for predicate in predicates:
        op = predicate.operator
        val = predicate.value
        if op == '>=' or op == '>':
            lower_bound = max(lower_bound, val)
        elif op == '<=' or op == '<':
            upper_bound = min(upper_bound, val)
    return lower_bound, upper_bound


def get_query_pattern(sql_query):
    where_clause = extract_where_clause(sql_query).strip().replace('`', '"')
    # Create a regex pattern for the query, replacing the WHERE clause with a placeholder that could contain anything
    query_pattern = sql_query.replace('`', '"').replace('*', '\*').replace('(', '\(').replace(')', '\)')
    where_clause = where_clause.replace('(', '\(').replace(')', '\)')
    query_pattern = query_pattern.replace(where_clause, '(.+)').strip().replace('\n', '\s+').replace('`', '"')
    query_pattern = query_pattern.replace('\n', ' ').replace(' ', '\s*')
    return query_pattern.replace('\\\\', '\\')


def get_column_values(where_clause):
    """
    uses sqlparse to tokenize the where clause and extract the column names.
    :param where_clause:
    :return:
    """
    current_col = ""
    column_names_map = {}
    for token in sqlparse.parse(where_clause)[0].tokens:
        if isinstance(token, sqlparse.sql.Identifier):
            current_col = token.value.replace('\"', '')
            if current_col not in column_names_map:
                column_names_map[current_col.replace('\"', '')] = []
        if not token.is_group:
            continue
        for sub_token in token.tokens:
            if isinstance(sub_token, sqlparse.sql.Identifier):
                current_col = sub_token.value.replace('\"', '')
                if current_col not in column_names_map:
                    column_names_map[current_col.replace('\"', '')] = []
            elif isinstance(sub_token, sqlparse.sql.Comparison):
                for sub_sub_token in sub_token.tokens:
                    if isinstance(sub_sub_token, sqlparse.sql.Identifier):
                        current_col = sub_sub_token.value.replace('\"', '')
                        if current_col not in column_names_map:
                            column_names_map[current_col] = []
            elif sub_token.value in ["<", ">", "<=", ">="]:
                column_names_map[current_col].append(sub_token.value)

    return column_names_map


def normalize_weights(w_red, w_blue):
    w_min = min(w_red, w_blue)
    w_max = max(w_red, w_blue)
    w_red = (w_min / w_max) if w_red == w_min else 1
    w_blue = (w_min / w_max) if w_blue == w_min else 1
    return w_red, w_blue


def get_constraint_avg_vals(constraint_results: DataFrame) -> Dict[str, Union[float, int]]:
    """
    Get the average values of the constraints from the DataFrame.
    :param constraint_results: DataFrame containing the results of the constraints
    :param constraint: The name of the constraint to get the average values for
    :return: A dictionary with the average values for each attribute in the constraint
    """
    avg_vals = {}
    for col in constraint_results.columns:
        avg_vals[f"{col} average"] = constraint_results[col].mean()
    return avg_vals


def construct_predicate_steps_by_jaccard(attribute: str, operator: str, orig_val: Union[float, int], df: pd.DataFrame) -> List[Union[float, int]]:
    """
    Generates Jaccard distance-relative steps to construct numeric predicate-subspace.
    :param attribute:
    :param operator:
    :param orig_val:
    :param df:
    :return:
    """
    sorted_df = df.sort_values(by=attribute).reset_index()
    bottom_df = sorted_df[sorted_df[attribute] <= orig_val].reset_index()
    bottom_step = len(bottom_df) // 8
    left_steps = [round(bottom_df.loc[i * bottom_step, attribute] - 0.1, 2) for i in range(8)]
    top_df = sorted_df[sorted_df[attribute] > orig_val].reset_index()
    top_step = len(top_df) // 8
    right_steps = [round(top_df.loc[i * top_step, attribute] - 0.1, 2) for i in range(1, 9)]
    value_span = left_steps + [orig_val] + right_steps
    if '>' in operator:
        value_range = value_span[:-1]
    elif '<' in operator:
        value_range = value_span[::-1][:-1]
    return value_range


def subsets_containing(l1: List[Any], l2: List[Any]) -> List[Tuple[Any]]:
    """
    Return all distinct subsets of l2 that are *multiset* supersets of l1.
    - Order inside each subset doesn't matter for identity, but results are
      returned in a stable, human-friendly order: items in each subset are
      arranged by their first appearance in l2.
    - Duplicates in l2 are respected (multiset logic).

    Examples:
      l1 = [2, 2], l2 = [1, 2, 2, 3]
      -> [[2, 2], [1, 2, 2], [2, 2, 3], [1, 2, 2, 3]]
    """
    c1, c2 = Counter(l1), Counter(l2)

    # If l1 needs something l2 doesn't have enough of, no solutions.
    for k, v in c1.items():
        if c2[k] < v:
            return []

    # Stable ordering of distinct values by first appearance in l2
    first_pos = {}
    for i, x in enumerate(l2):
        first_pos.setdefault(x, i)
    elems = list(c2.keys())
    elems.sort(key=lambda x: (first_pos[x], repr(x)))  # repr tie-breaker for unorderable types

    # For each element, choose how many times it appears in the subset:
    # - if required by l1: from c1[x]..c2[x]
    # - otherwise: 0..c2[x]
    ranges = []
    for x in elems:
        lo = c1.get(x, 0)
        hi = c2[x]
        ranges.append(range(lo, hi + 1))

    results = []
    for counts in product(*ranges):
        subset = []
        for x, k in zip(elems, counts):
            subset.extend([x] * k)
        # Only subsets that (by construction) include l1; no extra check needed
        results.append(tuple(subset))

    # order by size of subset, then lexicographically
    results.sort(key=lambda x: (len(x), x))

    # results = [str(x).replace(",)", ")") for x in results]
    # results = list(set(results))

    return results


# ===================== DuckDB helpers for single/multi table datasets =====================
def register_duckdb_tables(dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
    """Register a pandas DataFrame or a dict of DataFrames as DuckDB tables.

    - Single DataFrame is registered under the name 'df' (backward compatible)
    - Dict is registered with each key as table name
    """
    try:
        if isinstance(dataset, dict):
            for table_name, table_df in dataset.items():
                duckdb.register(str(table_name), table_df)
        else:
            duckdb.register("df", dataset)
    except Exception:
        # Swallow registration errors to preserve legacy behavior
        pass


def get_primary_dataframe(dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Return a primary DataFrame from either a single df or a dict of dfs.

    Preference order for dicts: 'df' key if present, else first inserted key.
    """
    if isinstance(dataset, dict):
        if "df" in dataset:
            return dataset["df"]
        try:
            # Python 3.7+ preserves insertion order
            first_key = next(iter(dataset.keys()))
            return dataset[first_key]
        except Exception:
            # Fallback empty frame
            return pd.DataFrame()
    return dataset


def build_dataset_schema_context(dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
    """Produce a schema context suitable for prompts.

    - Single df -> list of {name, type}
    - Dict      -> list of {table_name, columns: [{name, type}]}
    """
    try:
        if isinstance(dataset, dict):
            return [
                {
                    "table_name": str(table_name),
                    "columns": [
                        {"name": col, "type": str(table_df[col].dtype)}
                        for col in table_df.columns
                    ],
                }
                for table_name, table_df in dataset.items()
            ]
        else:
            return [{"name": col, "type": str(dataset[col].dtype)} for col in dataset.columns]
    except Exception:
        return []


# ===================== Attribute dtype helpers for dict datasets =====================
def get_attribute_dtype(dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], attribute_name: str) -> str:
    """Return a dtype string for an attribute from either a single df or a dict of dfs.

    Supports fully-qualified names like "table.column". If ambiguous (no table prefix),
    finds the first table that contains the column.
    """
    try:
        if isinstance(dataset, dict):
            table_name: Optional[str] = None
            col_name = attribute_name
            if "." in attribute_name:
                # Qualified reference: table.column (support alias or table names)
                parts = attribute_name.split(".")
                # If there are more than 2 segments (e.g., func(table.col)), take the last 2
                if len(parts) >= 2:
                    table_name = parts[-2]
                    col_name = parts[-1]
            if table_name and table_name in dataset and col_name in dataset[table_name].columns:
                return str(dataset[table_name][col_name].dtype)
            # Fallback: search all tables for the column
            for tname, df in dataset.items():
                if col_name in df.columns:
                    return str(df[col_name].dtype)
            return "unknown"
        # Single DataFrame
        return str(dataset[attribute_name].dtype) if attribute_name in dataset.columns else "unknown"
    except Exception:
        return "unknown"


def build_attributes_with_types_str(dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], attribute_names: List[str]) -> str:
    """Return formatted lines: "    - <attr> (<dtype>)" using multi-table aware dtype lookup."""
    lines: List[str] = []
    for attr in attribute_names:
        attr_name = attr.strip()
        dtype_str = get_attribute_dtype(dataset, attr_name)
        lines.append(f"    - {attr_name} ({dtype_str})")
    return "\n".join(lines)

