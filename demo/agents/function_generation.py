from typing import TypedDict, Annotated, Literal

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph, END
from tools.sql_engine import SQLEngineAdvanced
from config import OPENAI_API_KEY, MODEL
from tools.constants import IMPLEMENTER_SYSTEM_PROMPT, REFINEMENT_QUERY_DESCRIPTION, REFINEMENT_RESULT_DESCRIPTION, \
    REFINEMENT_QUERY_DISTANCE, REFINEMENT_RESULT_DISTANCE

from tools.utils import extract_code, get_d_in_info, get_query_where_columns

REFINEMENT_DESCRIPTIONS = {'Query': REFINEMENT_QUERY_DESCRIPTION, 'Result': REFINEMENT_RESULT_DESCRIPTION}
REFINEMENT_FUNCTIONS = {'Query': REFINEMENT_QUERY_DISTANCE, 'Result': REFINEMENT_RESULT_DISTANCE}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ImplementationAgent:
    """
    An abstract class representing an agent in a multi-agent system.
    """

    def __init__(self, original_query: str, d_in_dict: dict[str, pd.DataFrame]):
        self.model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
        self.system_prompt = SystemMessage(content=IMPLEMENTER_SYSTEM_PROMPT)
        self.original_query = original_query
        self.sql_engine = SQLEngineAdvanced(d_in_dict)
        self.d_in_dict = d_in_dict
        self.original_output = self.execute(original_query)
        columns_of_interest = get_query_where_columns(original_query)
        self.dataset_description = get_d_in_info(d_in_dict, columns_of_interest)
        @tool
        def get_columns_and_types():
            """
            Returns the columns names and their types by dataframe
            :return: A string containing the names of the columns and their types
            """
            return self.get_columns_names_and_types()
        @tool
        def get_specific_column_info(column_name: str):
            """
            Returns a string containing information about the given column name from the result dataframe, such as:
            - unique values for categorical columns
            - min and max values for numerical columns
            :param column_name: The attribute name of the column you would like to get information about
            :return: A string containing information about the column, such as:
            - unique values for categorical columns
            - min and max values for numerical columns
            """
            return self.get_specific_column_info_string(column_name)
        tools = [get_columns_and_types, get_specific_column_info]
        self.tools = {t.name: t for t in tools}
        self.model = self.model.bind_tools(tools)
        memory = MemorySaver()
        graph = StateGraph(AgentState)
        graph.add_node("infer", self.call_openai_infer)
        graph.add_node("infer_again", self.call_openai_infer)
        graph.add_node("take_action", self.take_action)
        graph.add_node("take_action_again", self.take_action)
        graph.add_node("implement", self.call_openai_implement)
        graph.add_conditional_edges("infer", self.exists_action, {True: "take_action", False: "infer"})
        graph.add_edge("take_action", "infer_again")
        graph.add_conditional_edges("infer_again", self.exists_action, {True: "take_action_again", False: "infer_again"})
        graph.add_edge("take_action_again", "implement")
        graph.add_edge("implement", END)
        graph.set_entry_point("infer")
        self.graph = graph.compile(checkpointer=memory)
        self.graph.config = {'recursion_limit': 1000}


    def call_openai_infer(self, state: AgentState):
        messages = state['messages']
        messages.append(HumanMessage(content="Use the tools you are equipped with to get specific columns names, "
                                             "types and values to help you implement the function"))
        message = self.model.invoke(messages)
        self.last_response = message
        return {'messages': [message]}

    def call_openai_implement(self, state: AgentState):
        messages = state['messages']
        messages.append(HumanMessage(content="Implement the function, using the information you have gathered"))
        message = self.model.invoke(messages)
        self.last_response = message
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0 and all([t['name'] in self.tools for t in result.tool_calls])

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in self.tools:  # check for bad tool name from LLM
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    def execute(self, query: str) -> pd.DataFrame:
        resp_df, error = self.sql_engine.execute(query)
        if error:
            return "An error occurred: " + resp_df
        return resp_df

    def get_columns_names_and_types(self):
        """
        Returns A List of column names and their types in the result dataframe
        :return: A string of the names of the columns and their types
        """
        ret_str = " - "
        col_names_to_types = {col: str(self.original_output[col].dtype) for col in self.original_output.columns}
        ret_str += "\n - ".join([f"{col}: {col_type}" for col, col_type in col_names_to_types.items()])
        return ret_str

    def get_specific_column_info_string(self, column_name: str):
        """
        Returns a string containing information about the given column name from result dataframe, such as:
        - unique values for categorical columns
        - min and max values for numerical columns
        :param column_name: The attribute name of the column you would like to get information about
        :return: A string containing information about the column, such as:
        - unique values for categorical columns
        - min and max values for numerical columns
        """

        if self.original_output[column_name].dtype == 'object':
            return f"Unique values for column {column_name} are: {self.original_output[column_name].unique()}"
        elif self.original_output[column_name].dtype == 'int64' or self.original_output[column_name].dtype == 'float64':
            return f"Min value for column {column_name} is: {self.original_output[column_name].min()}\n" \
                   f"Max value for column {column_name} is: {self.original_output[column_name].max()}"
        else:
            return f"Column {column_name} is of type {self.original_output[column_name].dtype}"

    def generate_constraints_objective(self, function_description: str):
        """
        Generates the constraints objective function.
        :return: The constraint objective function.
        """

        msg_text = f"Generate a constraint satisfaction objective function that evaluates the constraint deviation " \
                   f"for the output of the refined query, " \
                   f"based on the following description:\n\n{function_description}\n\n" \
                   f"You are advised to use the following structure of a result dataset:\n\n{self.original_output.head(5)}\n\n" \
                   f"Note that a constraint can be somewhat satisfied, meaning that its deviation score is not necessarily 0 or 1.\n" \
                   "Most Importantly: The Inner function output MUST be a normalized float value, between 0 and 1,\n" \
                   "where 0 indicates that all constraints are satisfied and 1 indicates that " \
                   "none of the constraints are satisfied. To achieve this:" \
                   "1. Each of the constraints deviation score should be normalized to a value between 0 and 1.\n" \
                   "2. The final constraint deviation score should be divided by the number of constraints.\n" \
                   "Ensure that the normalized score is a positive float that does not in any case exceed 1!\n"

        human_msg = HumanMessage(content=msg_text)
        resp = self.graph.invoke(
            {"messages": [self.system_prompt, human_msg]},
            config={"configurable": {"thread_id": 42}})['messages'][-1].content
        code_resp = extract_code(resp, language="python")
        return code_resp

    def generate_refinement_distance_objective(self, function_description: str):
        """
        Generates the refinement distance objective function.
        :param refined_query: The refined query.
        :return: The refinement distance objective function.
        """
        msg_text = f"Generate a refinement distance objective function that calculates the refinement " \
                   f"distance between the original query and the refined query, " \
                   f"based on the following description:\n\n{function_description}\n\n" \
                   f"You are advised to use the following structure of a result dataset:\n\n{self.original_output.head(5)}\n\n" \

        human_msg = HumanMessage(content=msg_text)
        resp = self.graph.invoke(
            {"messages": [self.system_prompt, human_msg]},
            config={"configurable": {"thread_id": 42}})['messages'][-1].content
        code_resp = extract_code(resp, language="python")
        return code_resp

    def edit_refinement_distance_objective(self, refinement_type: Literal['Result', 'Query'], function_description: str):
        """
        Generates the refinement distance objective function.
        :param refined_query: The refined query.
        :return: The refinement distance objective function.
        """
        refinement_type_function = REFINEMENT_FUNCTIONS[refinement_type]

        msg_text = f"Given the following {refinement_type} function, and below description:\n\n" \
                   f"Function implementation:\n\n{refinement_type_function}\n\n" \
                   f"--------------------------------\n\n" \
                   f"Specific description:\n\n{function_description}\n\n" \
                   f"--------------------------------\n\n" \
                   f"Edit the function implementation to match the given description, " \
                   f"while keeping the function signature intact.\n" \
                   f"You are advised to use the following dataset description:\n\n{self.dataset_description}\n\n"
        if refinement_type == 'Query':
            msg_text += """Note that the already implemented function 'parse_where_clause' returns 
            an object with the following structure:
WhereClause(numerical: List[NumericalPredicate], categorical: List[CategoricalPredicate])

In which the predicates are also objects defined as follows:
NumericalPredicate(name: str, operator: str, value: float)
CategoricalPredicate(name: str, values: List[str])
"""
        human_msg = HumanMessage(content=msg_text)
        resp = self.graph.invoke(
            {"messages": [self.system_prompt, human_msg]},
            config={"configurable": {"thread_id": 42}})['messages'][-1].content
        code_resp = extract_code(resp, language="python")
        return code_resp


if __name__ == '__main__':

    original_query = "SELECT * FROM texas_tribune WHERE MONTHLY >= 10000 AND MONTHLY <= 20000"
    dfs = {'texas_tribune': pd.read_csv('../../data/range_query_refinement/texas_tribune.csv')}
    agent = ImplementationAgent(original_query, dfs)

    function_description = "The output dataset should include no less than 40% female employees," \
                           "and no more than 30% of caucasian employees."
    constraints_objective_getter = agent.generate_constraints_objective(function_description)
    exec(constraints_objective_getter)
    evaluate_constraints_deviation = eval("get_constraints_satisfaction_objective(dfs, original_query)")
    print("Fairness deviation score:")
    print(evaluate_constraints_deviation("SELECT * FROM texas_tribune WHERE MONTHLY >= 9000 AND MONTHLY <= 15000"))

    print("\n")

    function_description = "1 - Jaccard similarity between the output records ids of the refined query and the original query"
    refinement_objective_getter = agent.generate_refinement_distance_objective(function_description)
    exec(refinement_objective_getter)
    evaluate_refinement_distance = eval("get_refinement_distance_objective(dfs, original_query)")
    print("Refinement distance score:")
    print(evaluate_refinement_distance("SELECT * FROM texas_tribune WHERE MONTHLY >= 9000 AND MONTHLY <= 15000"))
