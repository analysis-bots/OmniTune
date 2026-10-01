from abc import ABC
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Predicate(ABC):
    name: str


@dataclass
class NumericalPredicate(Predicate):
    operator: str
    value: float


@dataclass
class CategoricalPredicate(Predicate):
    values: List[str]


@dataclass
class WhereClause:
    numerical: List[NumericalPredicate]
    categorical: List[CategoricalPredicate]
