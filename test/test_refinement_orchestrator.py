import unittest
from unittest.mock import patch
import pandas as pd
import duckdb

from opro_main_loop import RefinementOrchestrator
from functionality.predicate import NumericalAttribute, CategoricalAttribute, NumericalPredicate, CategoricalPredicate
from functionality.constraint import DiverseTopKSelectionConstraint, DiversityCardinalityConstraint
from functionality.objectives import get_script_diff_func_sql


class _FakeCriticChatModel:
    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        # Return a deterministic subspace JSON for P1 (numeric) and P2 (categorical)
        return (
            "```json\n"
            "{\n"
            "  \"selected_predicate_subspace\": {\n"
            "    \"P1\": [25, 40],\n"
            "    \"P2\": [[\"Chicago\"], [\"Chicago\", \"New York\"]]\n"
            "  },\n"
            "  \"reasoning_concise\": \"test\"\n"
            "}\n"
            "```"
        )


class _FakeActorChatModel:
    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        # Suggest a refinement within the subspace
        return (
            "```json\n"
            "{\n"
            "  \"selected_predicate_values\": {\n"
            "    \"P1\": 35,\n"
            "    \"P2\": [\"Chicago\", \"New York\"]\n"
            "  },\n"
            "  \"selected_refinement\": \"""SELECT * FROM df WHERE \"age\" >= 35 AND \"city\" IN ('Chicago','New York')\""",\n"
            "  \"reasoning_concise\": \"actor-test\"\n"
            "}\n"
            "```"
        )


class TestRefinementOrchestrator(unittest.TestCase):
    def setUp(self):
        # Minimal dataset and registration for DuckDB as table name 'df'
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "city": ["Chicago", "New York", "Chicago", "New York", "Chicago"],
            "score": [10, 20, 30, 40, 50],
        })
        duckdb.register('df', self.df)

        # Original query + refineable predicates
        self.original_query = "SELECT * FROM df WHERE \"age\" >= 30 AND \"city\" IN ('Chicago')"
        age_attr = NumericalAttribute("age", 0, 100, 1)
        city_attr = CategoricalAttribute("city", ["Chicago", "New York", "LA"])
        self.predicates = [
            NumericalPredicate(age_attr, ">=", 30),
            CategoricalPredicate(city_attr, ["Chicago"]),
        ]
        self.refineable_predicates_str = "- age >= <refined_age_min>\n- city IN <refined_city>"

        # Constraints (easy to satisfy)
        self.constraints = [
            DiversityCardinalityConstraint(attribute="city", identifier="Chicago", desired_value=1, symbol=">="),
            DiverseTopKSelectionConstraint(attribute="score", identifier=30, desired_value=1, k=3, sign=+1, operator=">="),
        ]

    def _make_orchestrator(self, *, use_skyline: bool = True):
        return RefinementOrchestrator(
            task_name="Test_Task",
            original_query=self.original_query,
            input_dataset=self.df,
            constraints=self.constraints,
            epsilon=0.0,
            distance_func=get_script_diff_func_sql(self.original_query, self.df),
            refineable_predicates_str=self.refineable_predicates_str,
            refineable_predicates=self.predicates,
            parse_constraints=False,
            perform_analysis=False,
            max_refinements=3,
            max_subspace_iters=2,
            use_skyline=use_skyline,
        )

    def test_creation(self):
        orch = self._make_orchestrator()
        self.assertIsNotNone(orch.actor)
        self.assertIsNotNone(orch.critic)
        self.assertIn("distance", orch.subspace_memory_df.columns)

    @patch('agents.actor_critic.get_chat_model', side_effect=[_FakeActorChatModel(), _FakeCriticChatModel()])
    def test_run_refinement_loop(self, _):
        orch = self._make_orchestrator()
        best_query, best_distance, token_use = orch.run_refinement_loop()
        self.assertIsInstance(best_query, (str, type(None)))
        self.assertIsInstance(best_distance, float)
        # Ensure critic skyline recorded at least θ0 or a candidate
        self.assertTrue(len(orch.critic.skyline.data) >= 1)

    @patch('agents.actor_critic.get_chat_model', side_effect=[_FakeActorChatModel(), _FakeCriticChatModel()])
    def test_run_refinement_loop_without_skyline(self, _):
        orch = self._make_orchestrator(use_skyline=False)
        best_query, best_distance, token_use = orch.run_refinement_loop()
        self.assertIsInstance(best_query, (str, type(None)))
        self.assertIsInstance(best_distance, float)
        self.assertFalse(orch.use_skyline)
        self.assertIsNone(orch.skyline)
        self.assertIsNone(orch.critic.skyline)

    def test_record_run(self):
        orch = self._make_orchestrator()
        q = self.original_query
        pred_vals = {"age_min": 30, "city_set": ["Chicago"]}
        const_vals = orch.constraint_parser.get_results(q)
        orch.record_run(q, pred_vals, const_vals, distance=0.0, notes="init")
        self.assertEqual(len(orch.subspace_memory_df), 1)
        self.assertIn("constraint_score", orch.subspace_memory_df.columns)


if __name__ == '__main__':
    unittest.main()


