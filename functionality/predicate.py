from dataclasses import dataclass
from typing import Union, Dict, Optional

import numpy as np


@dataclass
class Attribute:
    name: str


@dataclass
class NumericalAttribute(Attribute):
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Union[float, int]

    def __str__(self):
        return f"{self.name} BETWEEN {self.min_value} AND {self.max_value} IN STEPS OF {self.step}"


@dataclass
class CategoricalAttribute(Attribute):
    categories: list[str]

    def __str__(self):
        return f"{self.name} IN ({', '.join(self.categories)})"


@dataclass
class Predicate:
    attribute: Attribute
    valid_value_range: Optional[Dict[str, Union[float, int, list]]]

    def get_closest_predicates(self, radius=1):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_id(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def contract(self, degree):
        """
        Contract the predicate by reducing the range of values.
        For numerical predicates, this means adjusting the value by the step size.
        """
        pass

    def relax(self, value):
        """
        Relax the predicate by expanding the range of values.
        For numerical predicates, this means adjusting the value by the step size.
        """
        pass

    def get_valid_value_range(self):
        pass


@dataclass
class NumericalPredicate(Predicate):
    operator: str
    value: Union[float, int]

    def __str__(self):
        return f"{self.attribute.name} {self.operator} {self.value}"

    def get_closest_predicates(self, radius=1) -> list[Predicate]:
        extended_predicates = []
        for i in range(1, radius+1):
            step = self.attribute.step * i
            if isinstance(self.value, NumericalPredicate):
                self.value = self.value.value  # Ensure value is a number
            # If value > (min_value + step), add a new predicate with the same operator and value - step * i
            if self.value >= self.attribute.min_value + step:
                extended_predicates.append(
                    NumericalPredicate(self.attribute, self.operator, round(self.value - step, 2))
                )
            # If value < (max_value - step), add a new predicate with the same operator and value + step * i
            if self.value <= self.attribute.max_value - step:
                extended_predicates.append(
                    NumericalPredicate(self.attribute, self.operator, round(self.value + step, 2))
                )
        return extended_predicates

    def get_id(self):
        return f"{self.attribute.name} {self.operator}"

    def contract(self, value: int = 1):
        """
        Contract the predicate by reducing the value by the step size.
        """
        if self.operator in ['<', '<=']:
            min_val = self.attribute.min_value if self.operator == '<' else self.attribute.min_value + self.attribute.step
            self.value = max(min_val, self.value - self.attribute.step * value)
        elif self.operator in ['>', '>=']:
            max_val = self.attribute.max_value if self.operator == '>' else self.attribute.max_value - self.attribute.step
            self.value = min(max_val, self.value + self.attribute.step * value)

    def relax(self, value: int = 1):
        """
        Relax the predicate by increasing the value by the step size.
        """
        if self.operator in ['<', '<=']:
            max_val = self.attribute.max_value if self.operator == '<' else self.attribute.max_value - self.attribute.step
            self.value = min(max_val, self.value + self.attribute.step * value)
        elif self.operator in ['>', '>=']:
            min_val = self.attribute.min_value if self.operator == '>' else self.attribute.min_value + self.attribute.step
            self.value = max(min_val, self.value - self.attribute.step * value)

    def get_valid_value_range(self):
        if self.valid_value_range is None:
            self.valid_value_range = {"min": self.attribute.min_value, "max": self.attribute.max_value, "step": self.attribute.step}
        return self.valid_value_range


@dataclass
class CategoricalPredicate(Predicate):
    values: list[str]

    def __str__(self):
        return f"{self.attribute.name} IN ({', '.join(self.values)})"

    def get_closest_predicates(self, radius=1) -> list[Predicate]:
        extended_predicates = []
        # If there are more categories, add a new predicate with the next category
        for category in self.attribute.categories:
            if category not in self.values:
                extended_predicates.append(
                    CategoricalPredicate(self.attribute, list(set(self.values + [category])))
                )

        # If there are fewer categories, create a new predicate with the current categories minus each category
        for category in self.values:
            set_without_category = self.values.copy()
            set_without_category.remove(category)
            extended_predicates.append(
                CategoricalPredicate(self.attribute, set_without_category)
            )

        return extended_predicates

    def get_id(self):
        return f"{self.attribute.name}"

    def contract(self, value: str):
        """
        Contract the predicate by removing the specified value from the set.
        """
        if value in self.values:
            self.values.remove(value)

    def relax(self, value: str):
        """
        Relax the predicate by adding the specified value to the set.
        """
        if value not in self.values:
            self.values.append(value)
            self.values = list(set(self.values))

    def get_valid_value_range(self):
        if self.valid_value_range is None:
            if isinstance(self.attribute.categories, np.ndarray):
                self.attribute.categories = self.attribute.categories.tolist()
            self.valid_value_range = {"categories": [self.values, self.attribute.categories]}
        return self.valid_value_range