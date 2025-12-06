import numpy as np
import pytest
from agents.refinement_preprocessor import RefinementPreprocessor, ConstraintContext
from functionality.predicate import Predicate, NumericalPredicate, CategoricalPredicate, \
    Attribute, NumericalAttribute, CategoricalAttribute

import unittest
import pandas as pd


class TestRefinementWizard(unittest.TestCase):
    def test_extract_constraint_context(self):
        # Sample data
        data = {
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        }
        df = pd.DataFrame(data)

        # Sample attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city', categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        # Test extract_constraint_context
        context = preprocessor.extract_constraint_context('{"query": df["income"].mean() <= 80000, '
                                                    '"description": "average income",'
                                                    '"symbol": "<=", "desired_value": "80000"}')
        self.assertIsNotNone(context)
        self.assertIsInstance(context, ConstraintContext)
        self.assertEqual(context.attribute, 'income')

        self.assertEqual(context.operator, '<=')
        self.assertEqual(context.value, 80000)
        self.assertIsInstance(context.ratio, float)

    def test_extract_subspaces_material(self):
        # Create a larger sample dataset with consistent quantiles and top categories
        np.random.seed(0)  # for reproducibility
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, size=1000),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size=1000,
                                     p=[0.4, 0.3, 0.2, 0.05, 0.05])
        })

        # Define numerical and categorical attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city',
                                         categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Create sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        # Call extract_subspaces_material()
        subspace_material = preprocessor.extract_subspaces_material()

        # Verify the extracted subspace material
        self.assertIsInstance(subspace_material, dict)
        self.assertIn('numerical', subspace_material)
        self.assertIn('categorical', subspace_material)

        # Verify numerical subspace material
        self.assertIn('age', subspace_material['numerical'])
        self.assertIsInstance(subspace_material['numerical']['age'], list)
        self.assertEqual(len(subspace_material['numerical']['age']), 4)  # 4 quantiles

        # Verify categorical subspace material
        self.assertIn('city', subspace_material['categorical'])
        self.assertIsInstance(subspace_material['categorical']['city'], list)
        self.assertEqual(len(subspace_material['categorical']['city']), 5)  # 5 top categories

    def test_generate_subspaces_df(self):
        # Create a larger sample dataset with consistent quantiles and top categories
        np.random.seed(0)  # for reproducibility
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, size=1000),
            'income': np.random.randint(40000, 120000, size=1000),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size=1000,
                                     p=[0.4, 0.3, 0.2, 0.05, 0.05])
        })

        # Define numerical and categorical attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city',
                                         categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Create sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        subspace_df = preprocessor.generate_subspaces_cartesian_df()
        self.assertIsInstance(subspace_df, pd.DataFrame)

        self.assertIn('age > ', subspace_df.columns)
        self.assertIn('city', subspace_df.columns)
        self.assertEqual(len(subspace_df), 20)

        self.assertEqual(len(subspace_df['age > '].unique()), 4)
        self.assertEqual(len(subspace_df['city'].unique()), 5)

    def test_generate_baseline_tracking_structure(self):
        # Create a larger sample dataset with consistent quantiles and top categories
        # Create a larger sample dataset with consistent quantiles and top categories
        np.random.seed(0)  # for reproducibility
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, size=1000),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size=1000,
                                     p=[0.4, 0.3, 0.2, 0.05, 0.05])
        })

        # Define numerical and categorical attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city',
                                         categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Create sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        tracking_structure = preprocessor.generate_baseline_tracking_structure()
        self.assertIsInstance(tracking_structure, pd.DataFrame)
        self.assertIn('age > ', tracking_structure.columns)

    def test_generate_constraint_tracking_structure(self):
        # Create a larger sample dataset with consistent quantiles and top categories
        np.random.seed(0)  # for reproducibility
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, size=1000),
            'income': np.random.randint(40000, 120000, size=1000),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size=1000,
                                     p=[0.4, 0.3, 0.2, 0.05, 0.05])
        })

        # Define numerical and categorical attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city',
                                         categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Create sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        # Test extract_constraint_context
        constraint_json = '{"query": df["income"].mean() <= 80000, ' \
                          '"description": "average income", ' \
                          '"symbol": "<=", "desired_value": "80000"}'

        tracking_structure = preprocessor.generate_constraint_tracking_structure(constraint_json)
        self.assertIsInstance(tracking_structure, pd.DataFrame)
        self.assertIn('age > ', tracking_structure.columns)
        self.assertIn('count', tracking_structure.columns)


    def test_get_marginal_contribution_struct(self):
        # Create a larger sample dataset with consistent quantiles and top categories
        np.random.seed(0)  # for reproducibility
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, size=1000),
            'income': np.random.randint(40000, 120000, size=1000),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size=1000,
                                     p=[0.4, 0.3, 0.2, 0.05, 0.05])
        })

        # Define numerical and categorical attributes
        age_attr = NumericalAttribute(name='age', min_value=0, max_value=100, step=1)
        city_attr = CategoricalAttribute(name='city',
                                         categories=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])

        # Create sample predicates
        predicates = [
            NumericalPredicate(age_attr, operator='>', value=30),
            CategoricalPredicate(attribute=city_attr, values=['Chicago'])
        ]

        # Initialize RefinementWizard
        preprocessor = RefinementPreprocessor(
            task_name="Test Task",
            input_dataset=df,
            original_query="SELECT * FROM table WHERE age > 30 AND city IN ('Chicago')",
            refineable_predicates=predicates,
            refinement_objective_description="predicate-based distance"
        )

        # Test extract_constraint_context
        constraint_json = '{"query": df["income"].mean() <= 80000, ' \
                          '"description": "average income", ' \
                          '"symbol": "<=", "desired_value": "80000"}'

        contrib_struct = preprocessor.get_marginal_contribution_struct(constraint_json)
        self.assertIsInstance(contrib_struct, pd.DataFrame)