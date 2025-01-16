import pandas as pd
import sqlite3


class SQLEngine:
    """A facade that holds a dataframe and executes SQL queries on it."""
    def __init__(self, df, keep_index=False):
        """
        Initializes the engine with a dataframe.
        :param df: The dataframe to be used.
        :param keep_index: Whether to keep the index in the dataframe when converting to SQL.
        """
        self.df = df
        self.conn = sqlite3.connect(':memory:')
        self.df.to_sql('df', self.conn, index=keep_index)

    def execute(self, query):
        """
        Executes a SQL query on the dataframe.
        :param query: The query to execute.
        :return: A tuple containing the result of the query or an error message if the query fails,
         along with a boolean indicating if an error occurred.
        """
        try:
            query = self._clean_query(query)
            resp_df = pd.read_sql_query(query, self.conn)
            return resp_df, False
        except Exception as e:
            return str(e), True

    def _clean_query(self, query):
        """
        Cleans the SQL query to fix common syntax errors, such as trailing commas in IN clauses.
        :param query: The query to clean.
        :return: The cleaned query.
        """
        # Use regex to remove trailing commas in IN clauses
        import re
        query = re.sub(r"\bIN\s*\(\s*'[^']*'\s*,\s*\)", lambda m: m.group(0).replace(",", ""), query)
        return query


class SQLEngineAdvanced:
    """A facade that holds a dataframe and executes SQL queries on it."""
    def __init__(self, dataframes: dict[str, pd.DataFrame], keep_index=True):
        """
        Initializes the engine with a dictionary of dataframes.
        :param dataframes: A dictionary of dataframes to be used.
        :param keep_index: Whether to keep the index in the dataframe when converting to SQL.
        """
        self.dataframes = dataframes
        self.conn = sqlite3.connect(':memory:')
        for name, df in dataframes.items():
            df.to_sql(name, self.conn, index=keep_index)

    def execute(self, query):
        """
        Executes a SQL query on the dataframe.
        :param query: The query to execute.
        :return: A tuple containing the result of the query or an error message if the query fails,
         along with a boolean indicating if an error occurred.
        """
        try:
            query = self._clean_query(query)
            resp_df = pd.read_sql_query(query, self.conn)
            resp_df = resp_df.set_index('index')
            return resp_df, False
        except Exception as e:
            return str(e), True

    def _clean_query(self, query):
        """
        Cleans the SQL query to fix common syntax errors, such as trailing commas in IN clauses.
        :param query: The query to clean.
        :return: The cleaned query.
        """
        # Use regex to remove trailing commas in IN clauses
        import re
        query = re.sub(r"\bIN\s*\(\s*'[^']*'\s*,\s*\)", lambda m: m.group(0).replace(",", ""), query)
        return query


if __name__ == '__main__':
    students = pd.read_csv("../datasets/students.csv")
    engine = SQLEngineAdvanced({'students': students})
    result, _ = engine.execute("SELECT * FROM students WHERE reason IN ('home', ) AND guardian = 'mother';")
    print(set(result.index))
