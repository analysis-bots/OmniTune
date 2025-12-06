from typing import List, Hashable, Tuple, Optional

import pandas as pd


class Skyline:
    def __init__(self):
        self.data = pd.DataFrame(columns=['query', 'distance', 'constraint_deviation'])
        self.are_constraint_satisfied = False
        # Mode management
        self.mode = "exploration"  # "exploration" or "optimization"
        self.epsilon = 0.0
        self.mode_switched = False
        self._removed_queries_log: List[Tuple[str, float, float]] = []

    def _thresholded_deviation(self, deviation: float) -> float:
        """
        Clamp deviation by epsilon for Pareto comparisons to align with ψ_ε(θ) = min(ψ(θ), ε).
        If epsilon is unset or non-positive, return the raw deviation.
        """
        try:
            dev = float(deviation)
        except Exception:
            return deviation
        try:
            eps = float(self.epsilon)
        except Exception:
            eps = None
        if eps is None or eps <= 0:
            return dev
        return min(dev, eps)

    def insert_query(self, query: str, distance: float, constraint_deviation: float):
        new_row = {
            'query': query,
            'distance': distance,
            'constraint_deviation': constraint_deviation
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

        # Auto-switch trigger: first satisfying query observed
        if (not self.mode_switched) and (constraint_deviation <= self.epsilon):
            self.switch_to_optimization_mode(self.epsilon)

    def switch_to_optimization_mode(self, epsilon: float):
        """Switch to optimization mode and cleanup non-satisfying queries exactly once."""
        if self.mode_switched:
            return
        self.epsilon = float(epsilon)
        # Update flags
        self.mode = "optimization"
        self.mode_switched = True
        self.are_constraint_satisfied = True

    def get_pareto_improvement(self, distance: float, constraint_deviation: float, epsilon: float = 0.0) -> List[Hashable]:
        """
        Returns indices in skyline that are dominated by a candidate under exploration mode.
        Minimizes both distance and constraint_deviation.
        """
        # If in optimization mode, delegate to optimization logic for domination indices
        if self.mode == "optimization":
            return self.get_pareto_improvement_optimization_mode(distance, constraint_deviation)

        dominated_indices = []
        if self.data.empty:
            return dominated_indices

        # Preserve rows that tie on the globally best constraint score
        min_constraint_in_skyline = self.data['constraint_deviation'].apply(self._thresholded_deviation).min()

        # if constraint_deviation == min_constraint_in_skyline:
        #     dominated_indices = self.data.index[self.data['constraint_deviation'] > min_constraint_in_skyline].tolist()
        #     return dominated_indices

        cand_const = self._thresholded_deviation(constraint_deviation)
        for idx, row in self.data.iterrows():
            row_dist = row['distance']
            row_const = self._thresholded_deviation(row['constraint_deviation'])

            dominates_via_distance = (row_dist > distance and row_const >= cand_const)
            dominates_via_constraint = (row_dist >= distance and row_const > cand_const)

            dominated = False
            # if dominates_via_distance or dominates_via_constraint:
            if dominates_via_constraint or dominates_via_distance:
                dominated = True

            # # Do not dominate a row that ties the candidate on the globally best constraint value
            # if dominated and (row_const == constraint_deviation == min_constraint_in_skyline):
            #     dominated = False

            if dominated:
                dominated_indices.append(idx)

        return dominated_indices

    def get_pareto_improvement_optimization_mode(self, distance: float, constraint_deviation: float) -> List[Hashable]:
        """Optimization-mode domination: binary constraint check, then distance-only domination."""
        dominated_indices = []
        if self.data.empty:
            return dominated_indices
        # Reject if candidate doesn't satisfy constraints per epsilon
        if not self.is_candidate_acceptable_optimization_mode(constraint_deviation):
            return dominated_indices
        # Distance-only: dominate rows strictly worse in distance
        for idx, row in self.data.iterrows():
            if row['distance'] > distance:
                dominated_indices.append(idx)
        return dominated_indices

    def is_candidate_acceptable_optimization_mode(self, constraint_deviation: float) -> bool:
        """Accept candidate in optimization mode iff ψ_ε(θ) <= ε using thresholded deviation."""
        return float(self._thresholded_deviation(constraint_deviation)) <= float(self._thresholded_deviation(self.epsilon))

    def maximize_pareto_improvement(self, query_candidates: List[Tuple[str, float, float]], epsilon: float = 0.0) -> str:
        """
        Returns the query that maximizes the Pareto improvement over the existing queries in the skyline,
        after removing all queries that are dominated by it from the skyline. Mode-aware behavior:
        - exploration: dual-objective (distance + constraint_deviation)
        - optimization: accept only constraint-satisfying candidates; distance-only domination
        """
        if not query_candidates:
            return None

        if self.mode == "optimization":
            # Filter candidates by acceptance
            acceptable = []
            for query, distance, constraint_deviation in query_candidates:
                if self.is_candidate_acceptable_optimization_mode(constraint_deviation):
                    dominated_indices = self.get_pareto_improvement_optimization_mode(distance, constraint_deviation)
                    acceptable.append((query, distance, constraint_deviation, len(dominated_indices), dominated_indices))
            if not acceptable:
                return None

            # if any of the candidates satisfy constraints, pick among them ONLY
            if any(x[2] <= epsilon for x in acceptable):
                acceptable = [x for x in acceptable if x[2] <= epsilon]

            # Actually pick max by dominated_count, then min distance
            best = max(acceptable, key=lambda x: (x[3], -x[1]))
            _, _, _, _, to_drop = best
            if to_drop:
                self.data = self.data.drop(index=to_drop).reset_index(drop=True)
            # Ensure skyline retains the winning candidate in optimization mode
            best_query, best_distance, best_constraint = best[0], best[1], best[2]
            # Insert winner if not already present
            if self.data.empty or not (self.data['query'] == best_query).any():
                self.insert_query(best_query, best_distance, best_constraint)
            return best[0]

        # exploration mode (existing behavior with slight refactor)
        min_constraint_in_skyline = (
            self.data['constraint_deviation'].apply(self._thresholded_deviation).min()
            if not self.data.empty else None
        )

        best_tuple = ([], -1, None, None)  # (dominated_indices, improvement_count, query, cand_const)
        for query, distance, constraint_deviation in query_candidates:
            dominated_indices = self.get_pareto_improvement(distance, constraint_deviation, epsilon)
            improvement_count = len(dominated_indices)
            cand_const = self._thresholded_deviation(constraint_deviation)

            # Rank: prioritize candidates that improve the best constraint level when present
            improves_best_constraint = (
                min_constraint_in_skyline is not None and cand_const < min_constraint_in_skyline
            )

            current_best_improves = (
                min_constraint_in_skyline is not None and
                best_tuple[3] is not None and
                best_tuple[3] < min_constraint_in_skyline
            )

            choose = False
            if improves_best_constraint and not current_best_improves:
                choose = True
            elif improves_best_constraint and current_best_improves:
                choose = improvement_count > best_tuple[1]
            elif not improves_best_constraint and not current_best_improves:
                choose = improvement_count > best_tuple[1]

            if choose:
                best_tuple = (dominated_indices, improvement_count, query, cand_const)

        max_dominated_indices, _, max_query, _ = best_tuple
        # Remove dominated queries from the skyline
        if max_dominated_indices:
            self.data = self.data.drop(index=max_dominated_indices).reset_index(drop=True)
        return max_query

    def get_best_query(self) -> Optional[str]:
        return self.data['query'].iloc[0] if not self.data.empty else None

    def get_md_table_view(self) -> str:
        """
        1. clean \n's from query
        2. format distance and constraint_deviation (2 decimal places)
        :return:
        """
        data_clean = self.data.copy()
        data_clean['query'] = data_clean['query'].str.replace('\n', ' ')
        data_clean['distance'] = data_clean['distance'].round(2)
        data_clean['constraint_deviation'] = data_clean['constraint_deviation'].round(2)
        return data_clean.to_markdown(index=False)
