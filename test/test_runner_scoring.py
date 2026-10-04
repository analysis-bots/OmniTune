import unittest
from types import SimpleNamespace

import pandas as pd

from experiments.refinement_scoring import score, validate_sql
from functionality.constraint import DiverseTopKSelectionConstraint
from functionality.predicate import CategoricalAttribute, CategoricalPredicate, NumericalAttribute, NumericalPredicate


class RunnerScoringTest(unittest.TestCase):
    def setUp(self):
        self.task = SimpleNamespace(
            original_query="SELECT * FROM df WHERE age >= 20 AND grp = 'A' ORDER BY age DESC",
            df=pd.DataFrame({'age': [30, 25, 20], 'grp': ['B', 'A', 'A']}),
            refineable_predicates=[
                NumericalPredicate(NumericalAttribute('age', 0, 100, 5), None, '>=', 20),
                CategoricalPredicate(CategoricalAttribute('grp', ['A', 'B']), None, ['A']),
            ],
            output_constraints=[DiverseTopKSelectionConstraint(
                attribute='grp', identifier='B', desired_value=1, k=2, sign=1, operator='=')],
            refinement_objective=lambda sql: 0.5,
        )
        self.config = {'epsilon': 0.0, 'max_dist': pd.Series([3]), 'gt': pd.Series([0.5])}

    def test_categorical_equality_to_in_and_real_success(self):
        sql = "SELECT * FROM df WHERE age >= 20 AND grp IN ('A','B') ORDER BY age DESC"
        result = score(sql, self.task, self.config, 0)
        self.assertTrue(result['success'])
        self.assertEqual(result['constraint_values'], [1.0])
        self.assertEqual(result['predicate_values']['grp'], ['A', 'B'])
        self.assertAlmostEqual(result['optimality'], 1.0)

    def test_finite_distance_is_not_success_when_constraints_fail(self):
        result = score(self.task.original_query, self.task, self.config, 0)
        self.assertEqual(result['distance'], 0.5)
        self.assertFalse(result['success'])
        self.assertEqual(result['optimality'], 0.)

    def test_out_of_domain_value_fails_even_if_constraints_pass(self):
        sql = "SELECT * FROM df WHERE age >= 21 AND grp IN ('A','B') ORDER BY age DESC"
        result = score(sql, self.task, self.config, 0)
        self.assertTrue(result['constraints_satisfied'])
        self.assertFalse(result['success'])
        self.assertEqual(result['domain_issues'], ['age >='])

    def test_rejects_non_predicate_changes(self):
        for sql in [self.task.original_query + ' LIMIT 1',
                    self.task.original_query.replace('>=', '>'),
                    self.task.original_query.replace("grp = 'A'", '1 = 1'),
                    self.task.original_query + '; SELECT 1']:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate_sql(sql, self.task)

    def test_negative_numeric_literal(self):
        self.task.refineable_predicates[0].attribute.min_value = -100
        values, issues = validate_sql(self.task.original_query.replace('>= 20', '>= -5'), self.task)
        self.assertEqual(values['age >='], -5)
        self.assertEqual(issues, [])


if __name__ == '__main__':
    unittest.main()
