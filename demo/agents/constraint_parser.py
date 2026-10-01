import ast
from typing import Dict, Union

import pandas as pd
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.constants import END
from langgraph.graph import StateGraph
from pandas.core.dtypes.common import is_numeric_dtype

from agents.function_generation import AgentState
from config import OPENAI_API_KEY, MODEL
from functionality.constraint import AgnosticConstraint
from tools.sql_engine import SQLEngineAdvanced
from tools.constants import PARSING_MODEL_SYSTEM_MESSAGE


class ConstraintParser:
    def __init__(self, input_dataset: Dict[str, pd.DataFrame], constraints_str: str, original_query: str = None):
        self.input_dataset = input_dataset
        self.constraints_str = constraints_str
        # Initialize the chat model with streaming enabled
        self.parsing_model = ChatOpenAI(
            model=MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        tools = self._get_tools()
        self.parsing_model = self.parsing_model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        graph = StateGraph(AgentState)
        graph.add_node("query_model", self.call_openai)
        graph.add_node("take_action", self.take_action)

        graph.add_edge("take_action", "query_model")
        graph.add_conditional_edges("query_model", self.exists_action,
                                    {True: "take_action", False: END})
        graph.set_entry_point("query_model")
        self.graph = graph.compile()
        self.sql_engine = SQLEngineAdvanced(self.input_dataset)
        self.query_result, _ = self.sql_engine.execute(original_query)
        self.json_response = None
        self.parsed_constraints = self.parse_constraints()

    def _get_tools(self):
        @tool
        def get_dataset_information():
            """
            Returns A List of result df's column names
            :return: A List of strings of the names of the columns in result df
            """
            df = self.query_result
            column_names_by_df = ""
            for col_name in df.columns:
                col = df[col_name]
                if is_numeric_dtype(col):
                    min_max_tuple = [col.min(), col.max()]
                    column_names_by_df += f" - Numerical ({col.dtype}) column '{col_name}' value range: {min_max_tuple}\n"
                else:
                    unique_vals = set(col.unique())
                    column_names_by_df += f" - Categorical column '{col_name}' unique values: {unique_vals}\n"
            return column_names_by_df

        return [get_dataset_information]

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0 and all([t['name'] in self.tools for t in result.tool_calls])

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in self.tools:  # check for bad tool name from LLM
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {"messages": results}

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        system_message = PARSING_MODEL_SYSTEM_MESSAGE + f"\nFor structural reference," \
                                                        f" here is an example of the original query result:" \
                                                        f"\n{self.query_result}"
        messages = [SystemMessage(content=system_message)] + messages
        response = self.parsing_model.invoke(messages)
        return {"messages": response}

    def parse_constraints(self):
        state = AgentState(messages=[HumanMessage(content=f"Parse the following constraints as a list of dictionaries"
                                                          f" in the JSON format described above.\n"
                                                          f"Ensure the response"
                                                          f" is properly structured JSON.\n"
                                                          f"Implement the query code-lines ONLY using pandas "
                                                          f"functions and operations, any other libraries or "
                                                          f"functions will not be supported.\n"
                                                          f"Constraints:\n"
                                                          f"{self.constraints_str}")])
        response = self.graph.invoke(state)["messages"][-1].content

        if '```json' in response:
            response = response.split('```json')[1]
            response = response.split('```')[0]
        self.json_response = response

        constraint_list = ast.literal_eval(response)
        return [AgnosticConstraint.from_dict(constraint) for constraint in constraint_list]

    def evaluation_list(self, df):
        constraint_bullets = []
        for constraint in self.parsed_constraints:
            if constraint.evaluate(df) == 1:
                constraint_bullets.append(f" ❌ {constraint.string_evaluation(df)}\n")
            elif constraint.evaluate(df) > 0:
                constraint_bullets.append(f" ⚠️ {constraint.string_evaluation(df)}\n")
            else:
                constraint_bullets.append(f" ✅ {constraint.string_evaluation(df)}\n")
        return constraint_bullets

    def evaluation_str(self, refined_query: str):
        df, _ = self.sql_engine.execute(refined_query)
        constraint_bullets = ["Constraint-wise feedback:\n"]
        for constraint in self.parsed_constraints:
            if constraint.evaluate(df) > 0:
                constraint_bullets.append(f"    * {constraint.string_evaluation(df)} (Not satisfied)\n")
            else:
                constraint_bullets.append(f"    * {constraint.string_evaluation(df)} (Satisfied)\n")
        return "\n".join(constraint_bullets)

    def evaluate_query(self, refined_query: str):
        refined_df, _ = self.sql_engine.execute(refined_query)
        return self.evaluation_list(refined_df)

    def get_json_response(self):
        return self.json_response

if __name__ == '__main__':
    law_students = pd.read_csv("../datasets/law_students.csv")
    constr = """the standard deviation of the average UGPA across the regions must be less than 0.2.
(the difference between the region with most students and the region with fewest students) divided by the average group size must be less than 0.2.
every region should have at least 10 law students in the result."""
    query = """SELECT region_first, AVG(UGPA) as avg_gpa, AVG(LSAT) as avg_lsat, COUNT(*) as size FROM law_students
WHERE UGPA > 3.5 AND LSAT > 40
GROUP BY region_first"""
    parser = ConstraintParser({'law_students': law_students}, constr, query)
