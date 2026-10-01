import time
from enum import Enum

from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated, Literal
import re
import operator
from math import exp
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from config import OPENAI_API_KEY, MODEL
from colorama import Fore

from .constraint_parser import ConstraintParser
from .logger import Log
from tools.utils import extract_code, parse_where_predicates, match_query
from tools.sql_engine import SQLEngineAdvanced
import streamlit as st
import sys
from io import StringIO

CRITIC_PROMPT_POSTFIX = "If any previous refinements have been provided above," \
                      " please use them to provide further refinements in as similar 'direction' to the best refinements," \
                      " in terms of constraint satisfaction score.\n" \
                      "DO NOT Suggest any concrete refinements, nor suggest specific values to be used in the refinement." \
                      " Only provide instructions what should be checked from the dataset in order to provide a better refinement."

class ActorCriticAgentState(TypedDict):
    actor_messages: Annotated[list[AnyMessage], add_messages]
    critic_messages: Annotated[list[AnyMessage], add_messages]


RECURSION_LIMIT = 1000


COLOR_MAP = {"Actor": Fore.CYAN, "Critic": Fore.MAGENTA}


class LlmAgent:
    def __init__(self, tools, chat_writer, logger, system_prompt="", name: Literal["Actor", "Critic", "Router"] = "Router",
                 constraint_str=""):
        self.model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL) # reduce inference cost
        self.system_prompt = system_prompt
        self.logger = logger
        self.tools = {t.name: t for t in tools}
        self.model = self.model.bind_tools(tools)
        self.chat_writer = chat_writer
        self.name = name
        self.constraint_str = constraint_str
        self.messages_key = f"{self.name.lower()}_messages"
        self.color = COLOR_MAP.get(self.name, Fore.RESET)
        if name == "Actor":
            self.previous_code_and_results = ""
        self.num_responses = 0

    def exists_action(self, state: ActorCriticAgentState):
        result = state[self.messages_key][-1]
        return len(result.tool_calls) > 0 and all([t['name'] in self.tools for t in result.tool_calls])

    def exists_content(self, state: ActorCriticAgentState):
        result = state[self.messages_key][-1]
        is_ai_message = isinstance(result, AIMessage)
        return is_ai_message and len(result.content) > 0

    def exists_query(self, state: ActorCriticAgentState):
        result = state[self.messages_key][-1]
        if isinstance(result, AIMessage):
            extracted_query = extract_code(result.content)
            if extracted_query is not None and len(extracted_query) > 0:
                matched_query = match_query(extracted_query)
                if matched_query is not None:
                    return True
        return False

    def call_openai(self, state: ActorCriticAgentState):
        messages = state[self.messages_key]
        if self.name == "Actor":
            last_message_with_content = next((m for m in messages[::-1] if m.content), None)
            content = f"Follow the below instructions step by step to optimize your refinement:\n{last_message_with_content.content}" \
                if last_message_with_content else ""
            if len(self.previous_code_and_results) > 0:
                 content = "Below are previous pandas queries on the dataset and their results. " \
                      f"Use them to be nore familiar with the dataset:\n\n{self.previous_code_and_results}\n" \
                           f"{content}"
            messages += [HumanMessage(content=content)]
        elif self.name == "Critic":
            last_message_with_content = next((m for m in messages[::-1] if m.content), None)
            content = f"{last_message_with_content.content}\n\n{CRITIC_PROMPT_POSTFIX}"
            messages += [HumanMessage(content=content)]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        with self.chat_writer.output_tab:
            spinner_text = f"Building new query..." if self.name == "Actor" else \
                f"Analyzing query and constructing feedback from query results..."
            with st.spinner(spinner_text):
                with self.chat_writer.chat_tab:
                    with st.spinner(spinner_text):
                        message = self.model.invoke(messages)
        if len(message.content) > 0:
            self.chat_writer.write_message(self.name, message.content)
        if message.content and len(message.content) > 0:
            log = Log(self.name, f"\n{message.content}")
            self.logger.log(log)
        return {self.messages_key: [message]}

    def take_action(self, state: ActorCriticAgentState):
        tool_calls = state[self.messages_key][-1].tool_calls
        results = []
        for t in tool_calls:
            log = Log(self.name, f"Calling Tool: {t}")
            self.logger.log(log)
            if not t['name'] in self.tools:  # check for bad tool name from LLM
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
                short_result = str(result) if len(str(result)) < 100 else str(result)[:100] + "..."
                log.append_content(self.name, f"Tool Result: {short_result}")
                self.logger.log(log)
                if self.name == "Actor" and t['name'] == "execute_python_code_line_with_dataframe":
                    self.previous_code_and_results += f"\nCode Line:\n{t['args']['code_string']}\nResult:\n{result}\n" + "=" * 50 + "\n"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {self.messages_key: results}


class ActorCriticAgent:

    def __init__(self, actor_tools, critic_tools, original_query, logger,
                 validation_function, evaluation_function, refinement_objective, epsilon,
                 chat_writer, constraint_str, constraint_parser,
                 actor_system="", critic_system=""):

        self.logger = logger
        self.chat_writer = chat_writer
        self.actor_agent = LlmAgent(actor_tools, chat_writer, logger, actor_system, "Actor")
        self.critic_agent = LlmAgent(critic_tools, chat_writer, logger, critic_system, "Critic",
                                     constraint_str=constraint_str)
        self.validation_function = validation_function
        self.evaluate_constraints = evaluation_function
        self.refinement_objective = refinement_objective
        self.epsilon = epsilon
        self.sufficient_var = False
        self.constraint_parser = constraint_parser

        memory = MemorySaver()
        graph = StateGraph(ActorCriticAgentState)
        graph.add_node("actor", self.actor_agent.call_openai)
        graph.add_node("actor_action", self.actor_agent.take_action)
        graph.add_node("critic", self.critic_agent.call_openai)
        graph.add_node("critic_action", self.critic_agent.take_action)
        graph.add_node("is_sufficient", self.is_sufficient)
        graph.add_node("actor_exists_query", self.actor_exists_query)
        graph.add_node("critic_exists_content", self.critic_exists_content)
        graph.add_node("trim_messages", self.trim_messages)

        self.previous_results = ""
        self._handle_query(original_query)

        graph.add_conditional_edges(
            "actor",
            self.actor_agent.exists_query,
            {True: "actor_exists_query", False: "actor_action"}
        )
        graph.add_conditional_edges(
            "critic",
            self.critic_agent.exists_action,
            {True: "critic_action", False: "critic_exists_content"}
        )
        graph.add_edge("actor_action", "actor")
        graph.add_edge("critic_action", "critic")
        graph.add_edge("actor_exists_query", "trim_messages")
        graph.add_edge("trim_messages", "is_sufficient")
        graph.add_conditional_edges(
            "is_sufficient",
            self.is_sufficient_func,
            {True: END, False: "critic"}
        )
        graph.add_conditional_edges(
            "critic_exists_content",
            self.critic_agent.exists_content,
            {True: "actor", False: "critic"}
        )
        graph.set_entry_point("actor")
        self.config = {"configurable": {"thread_id": "42"}}
        self.graph = graph.compile(checkpointer=memory)
        self.last_response = None
        self.refinement_num = 0

    def is_sufficient_func(self, state: ActorCriticAgentState):
        return self.sufficient_var

    def get_last_response(self):
        return self.last_response

    def display_graph(self):
        try:
            png_data = self.graph.get_graph(xray=True).draw_mermaid_png()
            with open('test_output.png', 'wb') as f:
                f.write(png_data)
        except Exception:
            # This requires some extra dependencies and is optional
            pass

    def is_sufficient(self, state: ActorCriticAgentState):
        last_actor_query = None
        for msg in state['actor_messages'][::-1]:
            if isinstance(msg, AIMessage):
                content = msg.content
                extracted_query = extract_code(content)
                if extracted_query is not None:
                    last_actor_query = extracted_query
                    break
        if (last_actor_query is not None) and self.sufficient_var:
            return {"actor_messages": [HumanMessage(content=last_actor_query)],
                    "critic_messages": [HumanMessage(content=last_actor_query)]}
        return {"critic_messages": []}

    def _extract_last_actor_query(self, state):
        last_actor_query = None
        for msg in state['actor_messages'][::-1]:
            if isinstance(msg, AIMessage):
                content = msg.content
                extracted_query = extract_code(content)
                if extracted_query is not None:
                    last_actor_query = extracted_query
                    break
        return last_actor_query

    def actor_exists_query(self, state: ActorCriticAgentState):
        last_actor_message_content = ([next((m.content for m in state['actor_messages'][::-1] if m.content), None)])
        extracted_query = self._extract_last_actor_query(state)
        self.sufficient_var = self._handle_query(extracted_query)
        if len(self.previous_results) > 0:
            res_prompt = f"Below are the previous refinements and scores.\n" \
                         f"Please begin your instructions with comparing the last result to all of them " \
                         f"in terms of constraint satisfaction and refinement distance:\n\n{self.previous_results}\n" \
                         f"Craft a step-by-step plan including what to investigate in the data, " \
                         f"what to adjust according to the result, and etc."
            self.logger.log(Log("Actor", "\n" + "=" * 50 + f"\n\nPrevious results provided here\n\n" + "=" * 50 + "\n"))
            self.logger.log(Log("Actor", f"\n{last_actor_message_content}"))
            last_actor_message_content = f"{res_prompt}\n{last_actor_message_content}"

        # check if the query is a valid SQL query
        if extracted_query is not None and len(extracted_query) > 0:
            is_query_valid, info_str = self.validation_function(extracted_query)
            if is_query_valid:
                constraint_feedback_str = self.constraint_parser.evaluation_str(extracted_query)
                last_actor_message_content += f"\nUse the below constraint-wise feedback to provide more informed instructions:" \
                                              f"\n{constraint_feedback_str}\n"
                self.refinement_num += 1
        human_message = HumanMessage(content=last_actor_message_content)
        return {"critic_messages": [human_message]}

    def critic_exists_content(self, state: ActorCriticAgentState):
        last_critic_message = state["critic_messages"][-1]
        last_critic_message_content = f"Further instructions to follow:\n{last_critic_message.content}"
        if self.previous_results:
            res_prompt = f"Below are the previous refinements and scores:\n" \
                         f"\n\n{self.previous_results}\n" \
                         f"Please use them as a reference, along with the" \
                         f" below instructions to work in a step-by-step way generate a better refinement that adheres " \
                         f"to the requirements.\n"
            self.logger.log(Log("Actor", "\n" + "=" * 50 + f"\n\nPrevious results provided here\n\n" + "=" * 50 + "\n"))
            self.logger.log(Log("Actor", f"\n{last_critic_message_content}"))
            last_critic_message_content = f"{res_prompt}\n{last_critic_message_content}"

        human_message = HumanMessage(content=last_critic_message_content)
        return {"actor_messages": [human_message]}

    def _handle_query(self, extracted_query):
        previous_results_and_scores = ""
        if extracted_query is not None and len(extracted_query) > 0:
            # use regex to check if the query is a valid SQL query
            is_query_valid, info_str = self.validation_function(extracted_query)
            previous_results_and_scores += f"\nQuery:\n{extracted_query}\n\n"
            if is_query_valid:
                constraints_deviation = self.evaluate_constraints(extracted_query)
                new_refinement_dist = self.refinement_objective(extracted_query)

                constraint_score = round(constraints_deviation, 2)
                refinement_dist = round(new_refinement_dist, 2)

                self.chat_writer.show_query(extracted_query, constraint_score, refinement_dist)
                constraint_feedback = self.constraint_parser.evaluate_query(extracted_query)
                evaluation_str = self.constraint_parser.evaluation_str(extracted_query)
                self.chat_writer.show_constraints_feedback(constraint_feedback)
                previous_results_and_scores += f" - Valid: True\n" \
                                               f" - Refinement Distance: {refinement_dist}\n" \
                                               f" - Constraints Deviation: {round(constraints_deviation, 2)}\n" \
                                               f" - {evaluation_str}"
                if constraints_deviation <= self.epsilon:
                    return True
            else:
                previous_results_and_scores += f"Valid: False\nValidation Error: {info_str}\n"
            previous_results_and_scores += "\n" + "=" * 50 + "\n"
            self.previous_results += previous_results_and_scores
        return False

    def trim_messages(self, state: ActorCriticAgentState):
        if len(state["actor_messages"]) > 1:
            # actor_messages_to_delete = [RemoveMessage(id=m.id) if not isinstance(m, ToolMessage)
            #                             else RemoveMessage(id=m.id, kwargs={"tool_call_id": m.tool_call_id})
            #                             for m in state["actor_messages"][1:-1]]
            actor_messages_to_delete = [RemoveMessage(id=m.id, kwargs={"tool_call_id": m.tool_call_id}) for m in state["actor_messages"] if isinstance(m, ToolMessage)]
            actor_messages_to_delete += [RemoveMessage(id=m.id) for m in state["actor_messages"][1:-1] if isinstance(m, AIMessage)]

            # Clean tool calls from last AI Messages
            last_actor_message = state["actor_messages"][-1]
            if isinstance(last_actor_message, AIMessage):
                actor_messages_to_delete.append(RemoveMessage(id=last_actor_message.id))
                last_actor_message_without_tool_calls = AIMessage(content=last_actor_message.content)
                actor_messages_to_delete.append(last_actor_message_without_tool_calls)
        else:
            actor_messages_to_delete = []

        if len(state["critic_messages"]) > 1:
            # critic_messages_to_delete = [RemoveMessage(id=m.id) if not isinstance(m, ToolMessage)
            #                                 else RemoveMessage(id=m.id, kwargs={"tool_call_id": m.tool_call_id})
            #                                 for m in state["critic_messages"][1:-1]]
            critic_messages_to_delete = [RemoveMessage(id=m.id, kwargs={"tool_call_id": m.tool_call_id}) for m in state["critic_messages"] if isinstance(m, ToolMessage)]
            critic_messages_to_delete += [RemoveMessage(id=m.id) for m in state["critic_messages"][1:-1] if isinstance(m, AIMessage)]
        else:
            critic_messages_to_delete = []
        return {"actor_messages": actor_messages_to_delete, "critic_messages": critic_messages_to_delete}

    def get_state(self):
        cur_state = self.graph.get_state(self.config)
        return cur_state.values
