import unittest

from functionality.skyline import Skyline


class TestSkyline(unittest.TestCase):
    def setUp(self):
        self.sky = Skyline()

    def test_initial_state(self):
        self.assertEqual(len(self.sky.data), 0)
        self.assertListEqual(list(self.sky.data.columns), [
            'query', 'distance', 'constraint_deviation'
        ])
        self.assertFalse(self.sky.are_constraint_satisfied)

    def test_insert_query(self):
        self.sky.insert_query("Q1", 1.0, 0.5)
        self.assertEqual(len(self.sky.data), 1)
        row = self.sky.data.iloc[0]
        self.assertEqual(row['query'], "Q1")
        self.assertAlmostEqual(row['distance'], 1.0)
        self.assertAlmostEqual(row['constraint_deviation'], 0.5)

    def test_mode_switch_and_cleanup(self):
        # Set epsilon and insert non-satisfying first
        self.sky.epsilon = 0.3
        self.sky.insert_query("A", 2.0, 0.8)
        self.sky.insert_query("B", 1.8, 0.6)
        self.assertEqual(self.sky.mode, "exploration")
        # Insert satisfying -> triggers switch and cleanup
        self.sky.insert_query("S", 1.5, 0.2)
        self.assertEqual(self.sky.mode, "optimization")
        self.assertTrue(self.sky.mode_switched)
        # Non-satisfying rows removed
        self.assertTrue(all(self.sky.data['constraint_deviation'] <= self.sky.epsilon))
        # In optimization mode, acceptance requires <= epsilon
        dominated = self.sky.get_pareto_improvement_optimization_mode(1.0, 0.1)
        self.assertIsInstance(dominated, list)

    def test_get_pareto_improvement_dominates_some(self):
        # Seed skyline with three points
        self.sky.insert_query("A", 2.0, 0.8)  # dominated by candidate
        self.sky.insert_query("B", 1.8, 0.6)  # not dominated (better constraint)
        self.sky.insert_query("C", 1.2, 0.9)  # dominated by candidate (worse both)

        # Candidate improves distance and not worse on constraint vs A, worse on constraint than B
        dominated = self.sky.get_pareto_improvement(distance=1.5, constraint_deviation=0.8)
        # Expect to dominate A (2.0,0.8) and C (1.2,0.9)?
        # For C: row.distance (1.2) > 1.5 is False and row.constraint (0.9) > 0.8 is True,
        # but also need row.distance >= distance (1.2 >= 1.5) False, so C is NOT dominated.
        # Only A should be dominated here.
        self.assertEqual(dominated, [0])

        # A stronger candidate dominates A and C
        dominated_strong = self.sky.get_pareto_improvement(distance=1.0, constraint_deviation=0.6)
        self.assertCountEqual(dominated_strong, [0, 2])

    def test_maximize_pareto_improvement_and_drop(self):
        # Seed skyline
        self.sky.insert_query("A", 2.0, 0.8)
        self.sky.insert_query("B", 1.8, 0.6)
        self.sky.insert_query("C", 1.2, 0.9)

        # Two candidates; Qx should dominate [A, C]; Qy dominates only A
        candidates = [
            ("Qx", 1.0, 0.6),
            ("Qy", 1.5, 0.8),
        ]
        winner = self.sky.maximize_pareto_improvement(candidates)
        self.assertEqual(winner, "Qx")

        # After choosing Qx, dominated rows (A, C) should be dropped
        remaining_queries = set(self.sky.data['query'].tolist())
        self.assertSetEqual(remaining_queries, {"B"})

    def test_maximize_with_empty_skyline(self):
        # No rows yet; should pick the first candidate and drop none
        candidates = [("Q1", 1.0, 0.5), ("Q2", 0.8, 0.8)]
        winner = self.sky.maximize_pareto_improvement(candidates)
        self.assertEqual(winner, "Q1")
        self.assertEqual(len(self.sky.data), 0)

    def test_optimization_mode_candidate_acceptance_and_selection(self):
        # Switch to optimization mode explicitly
        self.sky.epsilon = 0.3
        self.sky.insert_query("S1", 1.2, 0.2)  # triggers switch & cleanup
        self.assertEqual(self.sky.mode, "optimization")
        # Now only candidates with constraint<=epsilon are considered, pick min distance domination
        candidates = [
            ("Q_bad", 0.5, 0.4),   # rejected (constraint>epsilon)
            ("Q_ok1", 1.1, 0.2),   # acceptable
            ("Q_ok2", 0.9, 0.1),   # acceptable and better distance
        ]
        winner = self.sky.maximize_pareto_improvement(candidates)
        self.assertEqual(winner, "Q_ok2")
        # After selection, skyline should not be empty (observability guarantee)
        self.assertGreaterEqual(len(self.sky.data), 1)


if __name__ == '__main__':
    unittest.main()


