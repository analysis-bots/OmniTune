from typing import List

from functionality.predicate import NumericalPredicate, CategoricalPredicate


class SubspaceTools:
    def __init__(self, df):
        self.df = df

    def get_predicate_quantiles(self, predicate: NumericalPredicate) -> List[float]:
        """
        Get the quantiles for a specific attribute in the dataframe.

        :param attribute_name: The name of the attribute to get quantiles for.
        :param quantiles: A list of quantiles to compute (e.g., [0.25, 0.5, 0.75]).
        :return: A list of quantile values for the specified attribute.

        e.g:
         - for predicate "month > 5" it will return the quantiles [1, 3, 6, 9],
         - for predicate "month <= 8" it will return the quantiles [3, 6, 9, 12].

        """
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        if not isinstance(predicate, NumericalPredicate):
            raise ValueError("Predicate must be an instance of NumericalPredicate.")
        attribute_name = predicate.attribute.name

        if attribute_name not in self.df.columns:
            raise ValueError(f"Attribute '{attribute_name}' not found in the dataframe.")

        if predicate.operator in ['<', '<=']:
            quantiles = quantiles[1:]
        else:   # predicate.operator in ['>', '>=']:
            quantiles = quantiles[:-1]

        return self.df[attribute_name].quantile(quantiles).unique().tolist()

    def get_top_n_categories(self, predicate: CategoricalPredicate, n: int = 5) -> List[str]:
        """
        Get the top N categories for a specific categorical attribute in the dataframe.

        :param predicate: The CategoricalPredicate to get top categories for.
        :param n: The number of top categories to return.
        :return: A list of the top N categories for the specified attribute.

        e.g., for predicate "Country IN ('USA', 'China')" it will return the 5 most frequent countries in the dataset.
        """
        if not isinstance(predicate, CategoricalPredicate):
            raise ValueError("Predicate must be an instance of CategoricalPredicate.")

        attribute_name = predicate.attribute.name

        if attribute_name not in self.df.columns:
            raise ValueError(f"Attribute '{attribute_name}' not found in the dataframe.")

        existing_categories = predicate.values
        largest_n_categories = self.df[attribute_name].value_counts().nlargest(n).index.tolist()

        # return largest N categories removing all existing categories
        return list(set(largest_n_categories) - set(existing_categories))[:n]

        # return self.df[attribute_name].value_counts().nlargest(n).index.tolist()

