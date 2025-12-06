from abc import ABC
from dataclasses import dataclass
from typing import Union, Literal, Any, Callable

import numpy as np
import pandas as pd
from pandas import Series

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
    identifier: Any
    desired_value: int

    def evaluate(self, df) -> float:
        pass

    def __str__(self):
        pass
    def get_query_str(self):
        pass
    def get_query_value(self, df):
        pass

    def get_symbol(self):
        pass

    def get_symbol_str(self):
        pass

    def string_evaluation(self, df):
        query_result = self.get_query_value(df)
        evaluation = self.evaluate(df)

        if isinstance(query_result, float):
            query_result = round(query_result, 2)
        else:
            query_result = f"{query_result:,}"

        if evaluation > 0:
            much = "much " if evaluation == 1 else ""
            dot = "!" if evaluation == 1 else "."

            lower_or_greater = "greater than" if '<' in self.get_symbol() else "lower than"
            return f"{self.get_query_str()} is **{query_result}**, which is {much}" \
                   f"{lower_or_greater} the desired value of **{self.desired_value}**{dot}"
        else:
            return f"{self.get_query_str()} is **{query_result}**, which satisfies the constraint " \
                   f"of being {self.get_query_str()} is {self.get_symbol_str()} **{self.desired_value}**."
def get_value_counts(sensitive_attribute: Series, derived_sensitive_attribute: Series, sensitive_val: Any, derived_val: Any):
    try:
        return derived_sensitive_attribute[sensitive_attribute == sensitive_val].value_counts(normalize=True)[derived_val]
    except KeyError:
        return 0




@dataclass
class DiverseTopKSelectionConstraint(OutputConstraint):
    """
    k: the number of top rows in the dataframe to examine
    desired_value: the number of minimum number of rows that must satisfy the constraint
    attribute: the column to examine
    operator: the comparison operator
    identifier: the identifier to compare against
    """
    k: int
    sign: int
    operator: Literal['<', '<=', '=', '!=', '>=', '>']

    def evaluate(self, df):
        operation = OPERATOR_TO_LAMBDA(self.identifier)[self.operator]
        df_top_k = df.iloc[:self.k]
        filtered_df = df_top_k[df_top_k[self.attribute].apply(operation)]
        num_valid_values = len(filtered_df)
        return max(self.sign * (self.desired_value - num_valid_values), 0) / self.desired_value

    def __str__(self):
        midfix = "AT LEAST" if self.sign > 0 else "AT MOST"
        return f" - The top {self.k} values in the output should contain {midfix} {self.desired_value} " \
               f"values with '{self.attribute}' {self.operator} {self.identifier}.\n"

    def get_query_str(self):
        return f" - The number of rows in the top {self.k} of the output where '{self.attribute}' {self.operator} {self.identifier}"

    def get_query_value(self, df):
        operation = OPERATOR_TO_LAMBDA(self.identifier)[self.operator]
        df_top_k = df.iloc[:self.k]
        filtered_df = df_top_k[df_top_k[self.attribute].apply(operation)]
        num_valid_values = len(filtered_df)
        return num_valid_values

    def get_symbol(self):
        return ">=" if self.sign > 0 else "<="

    def get_symbol_str(self):
        return "at least" if self.sign > 0 else "at most"



@dataclass
class RangeQueryFairnessConstraint(OutputConstraint):
    """
    attribute: the column to examine
    w_red:
    w_blue:
    """
    w_red: int = 1
    w_blue: int = 1

    def evaluate(self, df):
        if len(df) == 0:
            return np.inf
        blue_values_count = len(df[df[self.attribute] == self.identifier])
        red_values_count = len(df[df[self.attribute] != self.identifier])
        delta_score = (abs(self.w_red * red_values_count - self.w_blue * blue_values_count) - self.desired_value) / self.desired_value
        delta_score = max(delta_score, 0)
        return min(delta_score, 1)  # TODO - very weak signal!!!!

    def __str__(self):
        return f" The absolute difference |{self.w_blue} * len(result_df['{self.attribute}'] == {self.identifier}) " \
               f"- {self.w_red} * len(result_df['{self.attribute}'] != {self.identifier})| should be below {self.desired_value}.\n"

    def get_query_str(self):
        return f" The absolute difference |{self.w_blue} * len(result_df['{self.attribute}'] == {self.identifier}) " \
               f"- {self.w_red} * len(result_df['{self.attribute}'] != {self.identifier})|"

    def get_query_value(self, df):
        blue_values_count = len(df[df[self.attribute] == self.identifier])
        red_values_count = len(df[df[self.attribute] != self.identifier])
        return abs(self.w_red * red_values_count - self.w_blue * blue_values_count)

    def get_symbol(self):
        return "<="

    def get_symbol_str(self):
        return "at most"

@dataclass
class DiversityCardinalityConstraint(OutputConstraint):
    """
    attribute: the column to examine
    symbol: the comparison operator
    number: the (minimum / maximum) number of rows that must satisfy the constraint
    original_query_df: the original dataframe
    """
    symbol: Literal['<=', '>=', '<', '>']

    def evaluate(self, df):
        """
        A function that evaluates the constraint on the output dataframe.
        The output dataframe's 'attribute' column must contain 'symbol' 'number' of 'required_category' values.
        :param df:
        :return:
        """
        if '<' in self.symbol:
            delta_score = max(0, int(df[self.attribute].value_counts().get(self.identifier, 0)) - self.desired_value) / self.desired_value
            return min(delta_score, 1.0)
        elif '>' in self.symbol:
            delta_score = max(0, self.desired_value - int(df[self.attribute].value_counts().get(self.identifier, 0))) / self.desired_value
            return min(delta_score, 1.0)
        return 1.0

    def __str__(self):
        symbol_str = "at least" if '>' in self.symbol else "at most"
        return f" - The output column '{self.attribute}' should contain {symbol_str} {self.desired_value} " \
               f"of values equal to '{self.identifier}'."

    def get_query_str(self):
        return f" - The number of rows in the output where '{self.attribute}' == {self.identifier}"

    def get_query_value(self, df):
        return int(df[self.attribute].value_counts().get(self.identifier, 0))

    def get_symbol(self):
        return self.symbol

    def get_symbol_str(self):
        return "at least" if '>' in self.symbol else "at most"


class AgnosticConstraint:
    """
    query: a callable query that accepts the dataframe as input and returns a numeric value
    term_description: a description of the query in natural language
    symbol: the comparison operator
    desired_value: the (minimum / maximum) value that must satisfy the constraint
    """
    def __init__(self, query_str: str,
                 query: Callable[[pd.DataFrame], Union[int, float]],
                 description: str,
                 symbol: Literal["<", ">", "<=", ">="],
                 desired_value: Union[int, float]):
        self.query_str = query_str
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
        elif '=' in self.symbol:
            return abs(self.query(df) - self.desired_value) / self.desired_value
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
        query_str = constraint_dict["query"]
        query = lambda df: round(eval(constraint_dict["query"]), 2)
        return cls(query_str, query,
                   constraint_dict["description"], constraint_dict["symbol"], constraint_dict["desired_value"])


    def __str__(self):
        return f"   • {self.query_str} {self.symbol} {self.desired_value}"
