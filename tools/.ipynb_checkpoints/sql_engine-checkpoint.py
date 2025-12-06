import pandas as pd
import sqlite3


class SQLEngine:
    """A facade that holds a dataframe and executes SQL queries on it."""
    def __init__(self, df, keep_index=False, df_name='df'):
        """
        Initializes the engine with a dataframe.
        :param df: The dataframe to be used.
        :param keep_index: Whether to keep the index in the dataframe when converting to SQL.
        """
        self.df = df
        self.conn = sqlite3.connect(':memory:')
        self.df.to_sql(df_name, self.conn, index=keep_index)

    def execute(self, query):
        """
        Executes a SQL query on the dataframe.
        :param query: The query to execute.
        :return: A tuple containing the result of the query or an error message if the query fails,
         along with a boolean indicating if an error occurred.
        """
        try:
            resp_df = pd.read_sql_query(query, self.conn)
            return resp_df, False
        except Exception as e:
            return str(e), True


if __name__ == '__main__':
    engine = SQLEngine("../../data/range_query_refinement/texas_tribune.csv")
    result = engine.execute("SELECT AGENCY,ANNUAL FROM df WHERE MONTHLY < 10000")
    print(result)
