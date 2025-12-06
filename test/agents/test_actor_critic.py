import unittest
from unittest.mock import patch
import pandas as pd
import json

from agents.actor_critic import CriticLLM, ActorLLM
from functionality.constraint import DiverseTopKSelectionConstraint, DiversityCardinalityConstraint
from functionality.predicate import (
    NumericalAttribute, CategoricalAttribute,
    NumericalPredicate, CategoricalPredicate
)


class _FakeChatModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        # Return a deterministic subspace suggestion JSON
        return (
            "```json\n"
            "{\n"
            "  \"selected_predicate_subspace\": {\n"
            "    \"P_age_min\": [25, 35],\n"
            "    \"P_city_set\": [[\"Chicago\"], [\"Chicago\", \"New York\"]]\n"
            "  },\n"
            "  \"reasoning_concise\": \"test\"\n"
            "}\n"
            "```"
        )


class _FakeActorChatModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        # Return an actor refinement JSON with values that will be clamped/projected by ActorLLM
        # P1 numeric will be outside subspace (40); P2 categorical will miss contained element
        payload = {
            "selected_predicate_values": {
                "P1": 40,  # should clamp to max 35 given test subspace
                "P2": ["New York"]  # should project to include contained "Chicago"
            },
            "selected_refinement": "SELECT * FROM t WHERE age >= 35 AND city IN ('Chicago','New York')",
            "reasoning_concise": "test-actor"
        }
        return """```json
{json}
```""".format(json=json.dumps(payload))


class TestCriticLLM(unittest.TestCase):
    def setUp(self):
        # Minimal dataset
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "city": ["Chicago", "New York", "Chicago", "New York", "Chicago"],
            "score": [10, 20, 30, 40, 50],
        })

        # Refineable predicates
        age_attr = NumericalAttribute(name="age", min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name="city", categories=["Chicago", "New York", "LA"]) 
        self.predicates = [
            NumericalPredicate(age_attr, ">=", 30),
            CategoricalPredicate(city_attr, ["Chicago"]),
        ]

        # Constraints (diversity-focused)
        self.div_topk = DiverseTopKSelectionConstraint(
            attribute="score", identifier=30, desired_value=2, k=3, sign=+1, operator=">="
        )
        self.div_card = DiversityCardinalityConstraint(
            attribute="city", identifier="Chicago", desired_value=2, symbol=">="
        )

    def _make_critic(self, parsed_constraints):
        return CriticLLM(
            task_name="Test_Task_critic",
            constraints="",
            epsilon=0.1,
            input_dataset=self.df,
            parsed_constraints=parsed_constraints,
            original_query="SELECT * FROM t WHERE age >= 30 AND city IN ('Chicago')",
            refineable_predicates=self.predicates,
            refinement_objective_description="predicate-based distance"
        )

    def test_creation(self):
        critic = self._make_critic([self.div_topk, self.div_card])
        self.assertIsNotNone(critic.skyline)
        self.assertTrue(hasattr(critic, "global_history"))
        self.assertEqual(critic.subspace_history, [])
        # const_targets should include string keys of constraints
        self.assertTrue(any("top" in k or "rows" in k for k in critic.const_targets.keys()))

    @patch("agents.actor_critic.get_chat_model", return_value=_FakeChatModel())
    def test_get_subspace(self, _):
        critic = self._make_critic([self.div_card])
        best_query_json = {"query_sql": "SELECT * FROM t WHERE age >= 30"}
        subspace = critic.get_subspace(best_query_json)
        self.assertIn("P_age_min", subspace.numeric_predicates)
        self.assertEqual(tuple(subspace.numeric_predicates["P_age_min"]), (25.0, 35.0))
        self.assertIn("P_city_set", subspace.categorical_predicates)
        contained, containing = subspace.categorical_predicates["P_city_set"]
        self.assertIn("Chicago", contained)
        self.assertIn("New York", containing)

    @patch("agents.actor_critic.get_chat_model", return_value=_FakeChatModel())
    def test_update_history(self, _):
        critic = self._make_critic([self.div_topk])
        # Seed a subspace first so _last_suggested_subspace_key is set
        critic.get_subspace({"query_sql": "SELECT * FROM t WHERE age >= 30"})
        critic.update_history(
            query="SELECT * FROM t WHERE age >= 31",
            constraint_satisfaction={"C1": 0.2, "C2": 0.0},
            refinement_distance=0.3,
            constraint_deviation_score=0.5
        )
        # Skyline updated
        self.assertEqual(len(critic.skyline.data), 1)
        # Global history bucketed under last subspace
        self.assertTrue(len(critic.global_history) >= 1)
        total_records = sum(len(v) for v in critic.global_history.values())
        self.assertEqual(total_records, 1)
        # Subspace summary refreshed
        self.assertEqual(len(critic.subspace_history), 1)


class TestActorLLM(unittest.TestCase):
    def setUp(self):
        # Minimal dataset
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "city": ["Chicago", "New York", "Chicago", "New York", "Chicago"],
            "score": [10, 20, 30, 40, 50],
        })

        # Refineable predicates
        age_attr = NumericalAttribute(name="age", min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name="city", categories=["Chicago", "New York", "LA"]) 
        self.predicates = [
            NumericalPredicate(age_attr, ">=", 30),
            CategoricalPredicate(city_attr, ["Chicago"]),
        ]

        # Constraints (diversity-focused)
        self.div_topk = DiverseTopKSelectionConstraint(
            attribute="score", identifier=30, desired_value=2, k=3, sign=+1, operator=">="
        )
        self.div_card = DiversityCardinalityConstraint(
            attribute="city", identifier="Chicago", desired_value=2, symbol=">="
        )
        self.constraints_json = json.dumps([
            {
                "id": "C1",
                "description_concise": "Top-k score constraint",
                "target_value": self.div_topk.desired_value,
                "current_satisfaction_status": "unknown",
            },
            {
                "id": "C2",
                "description_concise": "City diversity constraint",
                "target_value": self.div_card.desired_value,
                "current_satisfaction_status": "unknown",
            },
        ])

    def _make_actor(self):
        return ActorLLM(
            task_name="Test_Task_actor",
            constraints_json=self.constraints_json,
            input_dataset=self.df,
            original_query="SELECT * FROM t WHERE age >= 30 AND city IN ('Chicago')",
            epsilon=0.1,
            refineable_predicates=self.predicates,
            refinement_objective_description="predicate-based distance",
        )

    def test_creation(self):
        actor = self._make_actor()
        # Ensure predicate mapping was created for two predicates P1, P2
        self.assertTrue(hasattr(actor, "predicate_id_to_info"))
        self.assertIn("P1", actor.predicate_id_to_info)
        self.assertIn("P2", actor.predicate_id_to_info)
        self.assertEqual(actor.constraint_columns, ["C1", "C2"])
        table_md = actor._md_table_from_local_history(actor.local_history)
        self.assertIn("Satisfied Constraints", table_md)
        self.assertIn("C1", table_md)

    @patch("agents.actor_critic.get_chat_model", return_value=_FakeActorChatModel())
    def test_refine_query_behavior(self, _):
        actor = self._make_actor()
        # Define subspace: age in [25,35], city contained [Chicago] ⊆ values ⊆ [Chicago, New York]
        subspace = {
            "numeric_predicates": {"P1": [25, 35]},
            "categorical_predicates": {"P2": [["Chicago"], ["Chicago", "New York"]]},
        }
        best_query_json = {"query_sql": "SELECT * FROM t WHERE age >= 30"}
        queries, predicate_values, reasons = actor.refine_query(subspace=subspace, best_query_json=best_query_json)

        # Expect one SQL query
        self.assertEqual(len(queries), 1)
        # Expect one predicate-values dict
        self.assertEqual(len(predicate_values), 1)
        vals = predicate_values[0]
        # Numeric clamped to 35
        self.assertIn("P1", vals)
        self.assertAlmostEqual(vals["P1"], 35.0)
        # Categorical projected to include contained set (Chicago) and within containing
        self.assertIn("P2", vals)
        self.assertIn("Chicago", vals["P2"])  # contained enforced
        self.assertTrue(set(vals["P2"]).issubset({"Chicago", "New York"}))
        # Reason present
        self.assertTrue(len(reasons) >= 1)

    def test_update_history_behavior(self):
        actor = self._make_actor()
        constraint_values = {"C1": 12.3456, "C2": 4}
        actor.update_history(
            query="SELECT * FROM t WHERE age >= 31",
            constraint_satisfaction=0.2,
            refinement_distance=0.3,
            constraint_values=constraint_values,
        )
        # Backwards-compatible lists updated
        self.assertEqual(len(actor.local_history["queries"]), 1)
        self.assertEqual(len(actor.local_history["constraint_satisfaction"]), 1)
        self.assertEqual(len(actor.local_history["refinement_distance"]), 1)
        self.assertIn("constraint_values", actor.local_history)
        self.assertEqual(len(actor.local_history["constraint_values"]), 1)
        self.assertEqual(actor.local_history["constraint_values"][0], constraint_values)
        self.assertIn("satisfied_constraints", actor.local_history)
        self.assertEqual(actor.local_history["satisfied_constraints"][0], False)
        self.assertEqual(actor.local_history["constraint_columns"], ["C1", "C2"])
        # Structured attempt record present
        self.assertIn("attempt_records", actor.local_history)
        self.assertEqual(len(actor.local_history["attempt_records"]), 1)
        record = actor.local_history["attempt_records"][0]
        self.assertEqual(record.get("constraint_metric"), 0.2)
        self.assertEqual(record.get("constraint_values"), constraint_values)
        self.assertFalse(record.get("satisfied_constraints"))

if __name__ == "__main__":
    unittest.main()
