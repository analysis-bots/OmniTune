from dataclasses import dataclass
from typing import List, Optional

# history.py

@dataclass
class QueryHistoryEntry:
    """
    Represents one refinement attempt.
    """
    query: str
    constraint_score: str
    refinement_distance: float
    is_satisfied: bool

@dataclass
class SubspaceHistoryEntry:
    """
    Represents one subspace analysis by the Critic.
    """
    subspace: str
    analysis_report: str

class History:
    """
    Manages both query and subspace histories.
    """

    def __init__(self):
        # Stores QueryHistoryEntry items
        self.query_history: List[QueryHistoryEntry] = []
        # Stores SubspaceHistoryEntry items
        self.subspace_history: List[SubspaceHistoryEntry] = []

    # ---- Query history methods ----

    def add_query(
        self,
        query: str,
        constraint_score: str,
        refinement_distance: float,
        is_satisfied: bool
    ) -> None:
        """
        Append a new query attempt.
        """
        entry = QueryHistoryEntry(
            query=query,
            constraint_score=constraint_score,
            refinement_distance=refinement_distance,
            is_satisfied=is_satisfied
        )
        self.query_history.append(entry)

    def last_queries(self, n: int) -> List[QueryHistoryEntry]:
        """
        Return the last n query entries.
        """
        return self.query_history[-n:]

    def best_query(self) -> Optional[QueryHistoryEntry]:
        """
        Return the satisfied query with the minimal refinement_distance,
        or None if no query satisfied the constraints.
        """
        satisfied = [e for e in self.query_history if e.is_satisfied]
        if not satisfied:
            return None
        return min(satisfied, key=lambda e: e.refinement_distance)
    # ---- Subspace history methods ----

    def add_subspace(self, subspace: str, analysis_report: str) -> None:
        """
        Append a new subspace analysis entry.
        """
        entry = SubspaceHistoryEntry(
            subspace=subspace,
            analysis_report=analysis_report
        )
        self.subspace_history.append(entry)

    def all_subspaces(self) -> List[SubspaceHistoryEntry]:
        """
        Return all recorded subspace analyses.
        """
        return self.subspace_history

    def latest_subspace(self) -> Optional[SubspaceHistoryEntry]:
        """
        Return the most recent subspace analysis, or None if empty.
        """
        if not self.subspace_history:
            return None
        return self.subspace_history[-1]


