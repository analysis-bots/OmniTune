from abc import ABC
from dataclasses import dataclass
from typing import Union, Literal, Any, Callable

import numpy as np
import pandas as pd
from pandas import Series
from math import exp

from tools.sql_engine import SQLEngineAdvanced
from tools.utils import normalize_weights

EPSILON = 0.05

OPERATOR_TO_LAMBDA = lambda y: {"<": lambda x: x < y,
                      "<=": lambda x: x <= y,
                      "=": lambda x: x == y,
                      "!=": lambda x: x != y,
                      ">=": lambda x: x >= y,
                      ">": lambda x: x > y,
                      "IN": lambda x: x.isin(y),
                      "NOT IN": lambda x: ~x.isin(y)}

@dataclass
class OutputConstraint(ABC):
    attribute: str

    def evaluate(self, df):
        pass

    def __str__(self):
        pass


def get_value_counts(sensitive_attribute: Series, derived_sensitive_attribute: Series, sensitive_val: Any, derived_val: Any):
    try:
        return derived_sensitive_attribute[sensitive_attribute == sensitive_val].value_counts(normalize=True)[derived_val]
    except KeyError:
        return 0




@dataclass
class DiverseTopKSelectionConstraint(OutputConstraint):
    """
    k: the number of top rows in the dataframe to examine
    n: the number of minimum number of rows that must satisfy the constraint
    attribute: the column to examine
    operator: the comparison operator
    value: the value to compare against
    """
    k: int
    sign: int
    n: int
    operator: Literal['<', '<=', '=', '!=', '>=', '>']
    value: Union[str, float]

    def evaluate(self, df):
        operation = OPERATOR_TO_LAMBDA(self.value)[self.operator]
        df_top_k = df.iloc[:self.k]
        filtered_df = df_top_k[df_top_k[self.attribute].apply(operation)]
        num_valid_values = len(filtered_df)
        return max(self.sign * (self.n - num_valid_values), 0) / self.n

    def __str__(self):
        midfix = "AT LEAST" if self.sign > 0 else "AT MOST"
        return f" - The top {self.k} values in the output should contain {midfix} {self.n} " \
               f"values with '{self.attribute}' {self.operator} {self.value}.\n"


@dataclass
class RangeQueryFairnessConstraint(OutputConstraint):
    """
    attribute: the column to examine
    value:
    w_red:
    w_blue:
    """
    value: Any
    w_red: int = 1
    w_blue: int = 1
    epsilon: int = 0.05
    norm_w_red, norm_w_blue = normalize_weights(w_red, w_blue)

    def evaluate(self, df):
        if len(df) == 0:
            return np.inf
        blue_values_count = len(df[~df[self.attribute] == self.value])
        red_values_count = len(df[~df[self.attribute] != self.value])
        return abs(self.norm_w_red * red_values_count - self.norm_w_blue * blue_values_count) / len(df)

    def __str__(self):
        return f" - The difference between the number of rows with attribute " \
               f"'{self.attribute}' == {self.value} times {self.w_red} and the number of rows with attribute " \
               f"'{self.attribute}' != {self.value} times {self.w_blue} should be less than {self.epsilon}.\n"


# class RangeQueryFairnessConstraint(OutputConstraint):
#     """
#     attribute: the column to examine
#     value:
#     reds:
#     blues:
#     w_red:
#     w_blue:
#     """
#     def __init__(self, attribute, value, reds, blues, w_red, w_blue):
#         super().__init__(attribute)
#         total_vals = reds + blues
#         self.orig_ratio = total_vals / (w_red * reds + w_blue * blues)
#         self.value = value
#         self.w_red = w_red
#         self.w_blue = w_blue
#
#     def evaluate(self, df):
#         if len(df) == 0:
#             return np.inf
#         total_df = len(df)
#         red_values_count = len(df[df[self.attribute] == self.value])
#         blue_values_count = len(df[df[self.attribute] != self.value])
#         out_ratio = (self.w_red * red_values_count - self.w_blue * blue_values_count) / total_df
#         return self.orig_ratio * out_ratio
#
#     def __str__(self):
#         return f" - The ratio between numer of rows '{self.attribute}' = {self.value} : '{self.attribute}' != {self.value} " \
#                   f"should be close to the ratio {self.w_red} : {self.w_blue}.\n"
#

class DiversityCardinalityConstraint(OutputConstraint):
    """
    attribute: the column to examine
    value: the category that must be present
    symbol: the comparison operator
    number: the (minimum / maximum) number of rows that must satisfy the constraint
    original_query_df: the original dataframe
    """
    def __init__(self, attribute, value, symbol, number):
        super().__init__(attribute)
        self.value = value
        self.symbol = symbol
        self.number = number

    def evaluate(self, df):
        """
        A function that evaluates the constraint on the output dataframe.
        The output dataframe's 'attribute' column must contain 'symbol' 'number' of 'required_category' values.
        :param df:
        :return:
        """
        if '<' in self.symbol:
            return max(0, df[self.attribute].value_counts().get(self.value, 0) - self.number) / self.number
        elif '>' in self.symbol:
            return max(0, self.number - df[self.attribute].value_counts().get(self.value, 0)) / self.number
        return 1.0

    def __str__(self):
        symbol_str = "at least" if '>' in self.symbol else "at most"
        return f" - The output dataframe's '{self.attribute}' column should contain {symbol_str} {self.number} " \
               f"of '{self.value}' values."


class AgnosticConstraint:
    """
    query: a callable query that accepts the dataframe as input and returns a numeric value
    term_description: a description of the query in natural language
    symbol: the comparison operator
    desired_value: the (minimum / maximum) value that must satisfy the constraint
    """
    def __init__(self, query: Callable[[pd.DataFrame], Union[int, float]],
                 description: str, symbol: Literal["<", ">", "<=", ">="], desired_value: Union[int, float]):
        self.query = query
        self.description = description
        self.symbol = symbol
        self.symbol_str = "lower or equal to" if '<' in symbol else "greater or equal to"
        self.desired_value = desired_value
        self.satisfaction_str = f"{description} is {self.symbol_str} {desired_value}."

    def evaluate(self, df):
        """
        A function that evaluates the constraint on the output dataframe.
        The output dataframe's 'attribute' column must contain 'symbol' 'number' of 'required_category' values.
        :param df:
        :return:
        """
        if '<' in self.symbol:
            return max(0, self.query(df) - self.desired_value) / self.desired_value
        elif '>' in self.symbol:
            return max(0, self.desired_value - self.query(df)) / self.desired_value
        return 1.0

    def string_evaluation(self, df):
        query_result = self.query(df)
        evaluation = self.evaluate(df)

        if isinstance(query_result, float):
            query_result = round(query_result, 2)
        else:
            query_result = f"{query_result:,}"

        if evaluation > 0:
            much = "much " if evaluation == 1 else ""
            dot = "!" if evaluation == 1 else "."

            lower_or_greater = "greater than" if '<' in self.symbol else "lower than"
            return f"{self.description} is **{query_result}**, which is {much}" \
                   f"{lower_or_greater} the desired value of **{self.desired_value}**{dot}"
        else:
            return f"{self.description} is **{query_result}**, which satisfies the constraint " \
                   f"of being {self.symbol_str} **{self.desired_value}**."

    @classmethod
    def from_dict(cls, constraint_dict):
        query = lambda df: eval(constraint_dict["query"])
        return cls(query, constraint_dict["description"], constraint_dict["symbol"], constraint_dict["desired_value"])


if __name__ == '__main__':
    astronauts = pd.read_csv("../datasets/astronauts_500kb.csv")
    constraint_dict = {
        "query": "sum(df.iloc[:10]['Gender'] == 'Female')",
        "description": "the number of female employees in the top 10 rows",
        "desired_value": 5,
        "symbol": ">=",
    }
    constraint = AgnosticConstraint.from_dict(constraint_dict)
    sql_engine = SQLEngineAdvanced({'astronauts': astronauts})
    q = """SELECT * FROM astronauts
WHERE "Graduate Major" IN ('Aeronautics & Astronautics', 'Aerospace Engineering', 'Mechanical Engineering', 'Engineering Management')
AND "Space Walks" >= 1
AND "Space Walks" <= 9
ORDER BY "Space Flight (hr)" DESC;
"""
    result, _ = sql_engine.execute(q)
    print(f" - {constraint.string_evaluation(result)}")

    constraint_dict = {
        "query": "abs(2 * sum(df['GENDER'] == 'FEMALE') - 3 * sum(df['GENDER'] == 'MALE')) / 5",
        "description": "The absolute distance between 2 fifths of the number of female rows "
                       "and 3 fifths of the number of male rows",
        "symbol": "<=",
        "desired_value": 4000
    }

    texas_tribune = pd.read_csv("../datasets/texas_tribune.csv")

    constraint = AgnosticConstraint.from_dict(constraint_dict)
    sql_engine = SQLEngineAdvanced({'texas_tribune': texas_tribune})
    q = """SELECT * FROM texas_tribune
WHERE "ANNUAL" >= 84999.96 AND "ANNUAL" <= 247804.5;
"""
    result, _ = sql_engine.execute(q)
    print(f" - {constraint.string_evaluation(result)}")
