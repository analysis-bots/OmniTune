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
from config import OPENAI_API_KEY
from functionality.constraint import AgnosticConstraint
from tools.sql_engine import SQLEngineAdvanced
from tools.constants import PARSING_MODEL_SYSTEM_MESSAGE

class ConstraintParser:
    def __init__(self, input_dataset: Dict[str, pd.DataFrame], constraints_str: str):
        self.input_dataset = input_dataset
        self.constraints_str = constraints_str
        # Initialize the chat model with streaming enabled
        self.parsing_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
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
        self.parsed_constraints = self.parse_constraints()
        self.sql_engine = SQLEngineAdvanced(self.input_dataset)

    def _get_tools(self):
        @tool
        def get_dataset_information():
            """
            Returns A List of df's column names
            :return: A List of strings of the names of the columns in df
            """
            column_names_by_df = ""
            for name, df in self.input_dataset.items():
                column_names_by_df += f"Dataframe: {name}\n"
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
        messages = [SystemMessage(content=PARSING_MODEL_SYSTEM_MESSAGE)] + messages
        response = self.parsing_model.invoke(messages)
        return {"messages": response}

    def parse_constraints(self):
        state = AgentState(messages=[HumanMessage(content=f"Parse the following constraints as a list of dictionaries:"
                                                          f"\n{self.constraints_str}")])
        response = self.graph.invoke(state)["messages"][-1].content
        if '```json' in response:
            response = response.split('```json')[1]
            response = response.split('```')[0]
        constraint_list = ast.literal_eval(response)
        return [AgnosticConstraint.from_dict(constraint) for constraint in constraint_list]

    def evaluation_string(self, df):
        constraint_bullets = []
        for constraint in self.parsed_constraints:
            if constraint.evaluate(df) == 1:
                constraint_bullets.append(f" ❌ {constraint.string_evaluation(df)}\n")
            elif constraint.evaluate(df) > 0:
                constraint_bullets.append(f" ⚠️ {constraint.string_evaluation(df)}\n")
            else:
                constraint_bullets.append(f" ✅ {constraint.string_evaluation(df)}\n")
        return constraint_bullets

    def evaluate_query(self, refined_query: str):
        refined_df, _ = self.sql_engine.execute(refined_query)
        return self.evaluation_string(refined_df)


if __name__ == '__main__':
    students = pd.read_csv("../datasets/students.csv")
    constr = """Average math score for female students should be above 12
    Average Portugese score for students with a family with more than 3 members should be above 13"""
    parser = ConstraintParser({'students': students}, constr)
