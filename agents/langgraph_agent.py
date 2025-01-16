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
from config import OPENAI_API_KEY
from colorama import Fore

from .constraint_parser import ConstraintParser
from .logger import Log
from tools.utils import extract_code, parse_where_predicates
from tools.sql_engine import SQLEngineAdvanced
import streamlit as st
import sys
from io import StringIO

TODO = f"\nIn case your previous suggestions of refinements did not help lower the refinement distance," \
          f" try to provide different directions for further query refinement. For example: " \
          f"if the predicates include both a lower and upper bound on some numeric attribute, " \
          f"and your previous instructions focused on adjusting the upper bound, " \
          f"you may try to focus on a numerical lower bound instead. If you previously focused on" \
          f" lowering some threshold, you might try to focus on increasing it instead."
class ActorCriticAgentState(TypedDict):
    actor_messages: Annotated[list[AnyMessage], add_messages]
    critic_messages: Annotated[list[AnyMessage], add_messages]


RECURSION_LIMIT = 1000


COLOR_MAP = {"Actor": Fore.CYAN, "Critic": Fore.MAGENTA}


class LlmAgent:
    def __init__(self, tools, chat_writer, logger, system_prompt="", name: Literal["Actor", "Critic", "Router"] = "Router",
                 constraint_str=""):
        self.model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")  # reduce inference cost
        self.count_tokens = self.model.get_num_tokens_from_messages
        self.system_prompt = system_prompt
        self.logger = logger
        self.tools = {t.name: t for t in tools}
        self.model = self.model.bind_tools(tools)
        self.chat_writer = chat_writer
        self.name = name
        self.constraint_str = constraint_str
        self.messages_key = f"{self.name.lower()}_messages"
        self.color = COLOR_MAP.get(self.name, Fore.RESET)
        if self.name == "Critic":
            # Initialize the chat model with streaming enabled
            self.stream_model = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=OPENAI_API_KEY,
                streaming=True,
                verbose=True
            )

    def exists_action(self, state: ActorCriticAgentState):
        result = state[self.messages_key][-1]
        return len(result.tool_calls) > 0 and all([t['name'] in self.tools for t in result.tool_calls])

    def exists_content(self, state: ActorCriticAgentState):
        result = state[self.messages_key][-1]
        is_ai_message = isinstance(result, AIMessage)
        return is_ai_message and len(result.content) > 0

    def call_openai(self, state: ActorCriticAgentState):
        messages = state[self.messages_key]
        if self.name == "Actor":
            last_message_with_content = next((m for m in messages[::-1] if m.content), None)
            content = f"{last_message_with_content.content}\nIf you used any tools, please provide the overall explanation of" \
                      f" the tools used, why you used them, and what information you gained from them, and then provide your result."
            messages += [HumanMessage(content=content)]
        elif self.name == "Critic":
            last_message_with_content = next((m for m in messages[::-1] if m.content), None)
            content = f"{last_message_with_content.content}\nIf any previous refinements have been provided above," \
                      f" please use them to provide further refinements in as similar 'direction' to the best refinements," \
                      f" in terms of constraint satisfaction score.\n" \
                      f"Very important! if you suggest an numerical integer attribute to be adjusted, make sure that the " \
                      f"suggested refined type is ONLY an Integer and not a Float!\n" \
                      f"Also for Categorical attributes, you may try to exclude some of existing values and not only add new ones."
            messages += [HumanMessage(content=content)]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        with self.chat_writer.output_tab:
            spinner_text = f"Building new query..." if self.name == "Actor" else f"Analyzing query and constructing feedback from query results..."
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

    def send_to_critic(self, state: ActorCriticAgentState):
        last_message_with_content = next((m for m in state[self.messages_key][::-1] if m.content), None)
        if last_message_with_content:
            human_message = HumanMessage(content=last_message_with_content.content, id=last_message_with_content.id)
            return {"critic_messages": [human_message]}
        return {"critic_messages": []}

    def send_to_actor(self, state: ActorCriticAgentState):
        last_message_with_content = next((m for m in state[self.messages_key][::-1] if m.content), None)
        if last_message_with_content:
            human_message = HumanMessage(content=last_message_with_content.content, id=last_message_with_content.id)
            return {"actor_messages": [human_message]}
        return {"actor_messages": []}

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
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {self.messages_key: results}


class ActorCriticAgent:

    def __init__(self, actor_tools, critic_tools, original_query, logger,
                 validation_function, evaluation_function, refinement_objective, epsilon,
                 chat_writer, constraint_str, constraint_parser,
                 actor_system="", critic_system="", is_sufficient_func=lambda x: False):

        self.logger = logger
        self.chat_writer = chat_writer
        self.actor_agent = LlmAgent(actor_tools, chat_writer, logger, actor_system, "Actor")
        self.critic_agent = LlmAgent(critic_tools, chat_writer, logger, critic_system, "Critic",
                                     constraint_str=constraint_str)
        self.is_sufficient_func = is_sufficient_func
        self.validation_function = validation_function
        self.evaluate_fairness = evaluation_function
        self.refinement_objective = refinement_objective
        self.epsilon = epsilon
        self.constraint_parser = constraint_parser

        memory = MemorySaver()
        graph = StateGraph(ActorCriticAgentState)
        graph.add_node("actor", self.actor_agent.call_openai)
        graph.add_node("actor_action", self.actor_agent.take_action)
        graph.add_node("critic", self.critic_agent.call_openai)
        graph.add_node("critic_action", self.critic_agent.take_action)
        graph.add_node("send_to_critic", self.actor_agent.send_to_critic)
        graph.add_node("send_to_actor", self.critic_agent.send_to_actor)
        graph.add_node("is_sufficient", self.is_sufficient)
        graph.add_node("actor_exists_content", self.actor_exists_content)
        graph.add_node("critic_exists_content", self.critic_exists_content)
        graph.add_node("trim_messages", self.trim_messages)

        self.previous_results = ""
        self._handle_query(original_query)

        graph.add_conditional_edges(
            "actor",
            self.actor_agent.exists_action,
            {True: "actor_action", False: "actor_exists_content"}
        )
        graph.add_conditional_edges(
            "critic",
            self.critic_agent.exists_action,
            {True: "critic_action", False: "critic_exists_content"}
        )
        graph.add_edge("send_to_critic", "critic")
        graph.add_edge("send_to_actor", "actor")
        graph.add_edge("actor_action", "actor")
        graph.add_edge("critic_action", "critic")
        graph.add_conditional_edges(
            "actor_exists_content",
            self.actor_agent.exists_content,
            {True: "trim_messages", False: "send_to_actor"}
        )
        graph.add_edge("trim_messages", "is_sufficient")
        graph.add_conditional_edges(
            "is_sufficient",
            self.is_sufficient_func,
            {True: END, False: "send_to_critic"}
        )
        graph.add_conditional_edges(
            "critic_exists_content",
            self.critic_agent.exists_content,
            {True: "send_to_actor", False: "send_to_critic"}
        )
        graph.set_entry_point("actor")
        self.config = {"configurable": {"thread_id": "42"}}
        self.graph = graph.compile(checkpointer=memory)
        self.last_response = None

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
                is_query_valid, info_str = self.validation_function(extracted_query)
                if (extracted_query is not None) and is_query_valid:
                    last_actor_query = extracted_query
                    break
        if last_actor_query is not None:
            self._handle_query(last_actor_query)
            if self.is_sufficient_func(state):
                return {"actor_messages": [HumanMessage(content=last_actor_query)],
                        "critic_messages": [HumanMessage(content=last_actor_query)]}
        return {"critic_messages": []}

    def actor_exists_content(self, state: ActorCriticAgentState):
        last_actor_message = state["actor_messages"][-1]
        last_actor_message_content = last_actor_message.content
        if len(self.previous_results) > 0:
            res_prompt = f"Below are the previous refinements and scores:\n\n{self.previous_results}\n"
            self.logger.log(Log("Actor", "\n" + "=" * 50 + f"\n\nPrevious results provided here\n\n" + "=" * 50 + "\n"))
            self.logger.log(Log("Actor", f"\n{last_actor_message_content}"))
            last_actor_message_content = f"{res_prompt}\n{last_actor_message_content}"
            # print("\n" + "=" * 50 + f"\n\nPrevious results provided here\n\n" + "=" * 50 + "\n")
            human_message = HumanMessage(content=last_actor_message_content, id=last_actor_message.id)
            return {"critic_messages": [human_message]}
        return {"critic_messages": []}

    def critic_exists_content(self, state: ActorCriticAgentState):
        last_critic_message = state["critic_messages"][-1]
        last_critic_message_content = last_critic_message.content
        human_message = HumanMessage(content=last_critic_message_content, id=last_critic_message.id)
        return {"actor_messages": [human_message]}

    def _handle_query(self, extracted_query):
        previous_results_and_scores = ""
        if extracted_query is not None and len(extracted_query) > 0:
            # use regex to check if the query is a valid SQL query
            query_pattern = re.compile(r"SELECT[\s\n]+\*\s+FROM[\s\n]+\w+[\s\n]+WHERE[\s\n]+[\s\S]*")
            match = query_pattern.match(extracted_query.strip())
            if match:
                matched_query = match.group()
                is_query_valid, info_str = self.validation_function(matched_query)
                previous_results_and_scores += f"\nQuery:\n{matched_query}\n\n"
                if is_query_valid:
                    if is_query_valid:

                        fairness_deviation = self.evaluate_fairness(matched_query)
                        previous_results_and_scores += f"Valid: True\nFairness Deviation: {round(fairness_deviation, 2)}\n"
                        new_refinement_dist = self.refinement_objective(matched_query)

                        constraint_score = round(fairness_deviation, 2)
                        refinement_dist = round(new_refinement_dist, 2)

                        self.chat_writer.show_query(matched_query, constraint_score, refinement_dist)
                        constraint_feedback = self.constraint_parser.evaluate_query(matched_query)
                        self.chat_writer.show_constraints_feedback(constraint_feedback)
                        if fairness_deviation < self.epsilon + 0.0001:
                            previous_results_and_scores += f"Refinement Distance: {refinement_dist}\n"
                else:
                    previous_results_and_scores += f"Valid: False\nValidation Error: {info_str}\n"
                previous_results_and_scores += "\n" + "=" * 50 + "\n"
                self.previous_results += previous_results_and_scores

    def get_res_prompt(self):
        if len(self.previous_results) == 0:
            return ""
        return f"\nBelow are the previous refinements and scores:\n\n{self.previous_results}\n" \
              f"Give me a valid SPJ Query that is different from all pairs above,\n" \
              f"completely satisfied ALL fairness constraints, as well as refinement distance value lower than " \
              f"any of the above. The output must end with a valid SPJ Query given the above instructions, " \
              f"in the format:\n```sql\n<your refined query>\n```\n"

    def trim_messages(self, state: ActorCriticAgentState):
        if len(state["actor_messages"]) > 1:
            actor_messages_to_delete = [RemoveMessage(id=m.id) if not isinstance(m, ToolMessage)
                                        else RemoveMessage(id=m.id, kwargs={"tool_call_id": m.tool_call_id})
                                        for m in state["actor_messages"][1:-1]]
        else:
            actor_messages_to_delete = []
        return {"actor_messages": actor_messages_to_delete,}
                # "critic_messages": critic_messages_to_delete}

    def get_state(self):
        cur_state = self.graph.get_state(self.config)
        return cur_state.values
