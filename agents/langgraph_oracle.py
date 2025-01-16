import re

from colorama import Fore
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from typing import Dict, Union
from langchain_core.messages import HumanMessage, RemoveMessage
import pandas as pd
from math import exp
import streamlit as st
import sys
from io import StringIO

from agents.constraint_parser import ConstraintParser
from agents.langgraph_agent import ActorCriticAgent, RECURSION_LIMIT, ActorCriticAgentState
from config import OPENAI_API_KEY
from tools.constants import actor_system_prompt_template, critic_system_prompt_template, user_msg_orig, \
    actor_system_prompt_template_epsilon, critic_system_prompt_template_epsilon, user_msg_orig_epsilon, \
    agnostic_actor_system_prompt, agnostic_critic_system_prompt, \
    actor_agnostic_user_msg_orig, actor_agnostic_user_msg_orig_epsilon,\
    critic_agnostic_user_msg_orig, critic_agnostic_user_msg_orig_epsilon
from agents.logger import Logger, Log
from functionality.constraint import DiverseTopKSelectionConstraint
from tools.query_hash_map import QueryHashMap
from tools.utils import extract_code, parse_where_predicates, parse_where_clause, extract_where_clause
from functionality.objectives import evaluate_constraints
from pandas.core.dtypes.common import is_numeric_dtype
from tools.sql_engine import SQLEngine, SQLEngineAdvanced
from langchain.tools import tool
from functionality.task import Task, TopKRefinementTask, AgnosticTask
from tools.utils import highlight_predicate_differences_getter

MAX_ITERS = 1

# Streamlit-specific Chat Writer for unified chat container
class StreamlitChatWriter:
    def __init__(self, original_query: str, output_tab, chat_tab):
        # Initialize chat history
        self.chat_history = []
        self.output_tab = output_tab
        self.previous_feedback = None
        with self.output_tab:
            self.placeholder_header = st.empty()
            self.placeholder_query = st.empty()
            self.placeholder_newline = st.empty()
            self.placeholder_constraint_score = st.empty()
            self.placeholder_refinement_score = st.empty()
            self.placeholder_feedback_header = st.empty()
            self.placeholder_feedback = st.empty()
        self.chat_tab = chat_tab
        with self.chat_tab:
            self.container = st.container(height=600)
        self.highlight_predicate_differences = highlight_predicate_differences_getter(original_query)
        self.refinements_counter = 0

    def show_query(self, query, fairness_deviation, refinement_distance):
        with self.output_tab:
            if query.strip():
                self.refinements_counter += 1
                attempts = "attempts" if self.refinements_counter > 1 else "attempt"
                self.placeholder_header.markdown(f"**Current Refinement (after {self.refinements_counter} {attempts}):**")
                highlighted_diff = self.highlight_predicate_differences(query.strip())
                self.placeholder_query.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap;">{highlighted_diff}</div>',
                    unsafe_allow_html=True,
                )
                self.placeholder_newline.markdown(" ")
                self.placeholder_constraint_score.markdown(f"**Constraints Score:** {round(fairness_deviation, 2)}")
                self.placeholder_refinement_score.markdown(f"**Refinement Distance:** {round(refinement_distance, 2)}")

    def show_constraints_feedback(self, constraint_feedback):
        with self.output_tab:
            self.placeholder_feedback_header.markdown("**Critic Feedback:**")
            self.placeholder_feedback.markdown("\n".join(constraint_feedback))

    def write_message(self, role, message):
        with self.chat_tab:
            with self.container:
                avatar = "🤖" if role == "Actor" else "🔎"
                with st.chat_message(role, avatar=avatar):
                    st.write(f'**{role}:** {message}')
        self.chat_history.append({"role": role, "message": message})

    def render_log(self):
        self.container.empty()
        with self.container:
            for chat in self.chat_history:
                avatar = "🤖" if chat["role"] == "Actor" else "🔎"
                with st.chat_message(chat["role"], avatar=avatar):
                    st.write(f'**{chat["role"]}:** {chat["message"]}')

    def reset_output(self):
        with self.output_tab:
            self.placeholder_header.empty()
            self.placeholder_query.empty()
            self.placeholder_newline.empty()
            self.placeholder_constraint_score.empty()
            self.placeholder_refinement_score.empty()
            self.placeholder_feedback_header.empty()
            self.placeholder_feedback.empty()

    def flush(self):
        pass


def get_system_prompts(task: Task):
    if task.epsilon > 0:
        actor_prompt = actor_system_prompt_template_epsilon.format(key_rules_prompt=task.key_rules, epsilon=task.epsilon)
        critic_prompt = critic_system_prompt_template_epsilon.format(key_rules_prompt=task.key_rules, epsilon=task.epsilon)
    else:
        actor_prompt = actor_system_prompt_template.format(key_rules_prompt=task.key_rules)
        critic_prompt = critic_system_prompt_template.format(key_rules_prompt=task.key_rules)
    return actor_prompt, critic_prompt


def get_user_messages_agnostic(task: AgnosticTask):
    if task.epsilon > 0:
        actor_usr_msg = actor_agnostic_user_msg_orig_epsilon.format(alterable_attributes=task.alterable_attributes_str,
                                                                    original_query=task.original_query,
                                                                    categorical_attributes_str=task.categorical_attributes_str,
                                                                    numeric_attributes_str=task.numeric_attributes_str,
                                                                    epsilon=task.epsilon)

        critic_usr_msg = critic_agnostic_user_msg_orig_epsilon.format(alterable_attributes=task.alterable_attributes_str,
                                                                      original_query=task.original_query,
                                                                      refinement_objective_str=task.refinement_objective_str,
                                                                      constraints_str=task.constraints_str,
                                                                      epsilon=task.epsilon)

    else:
        actor_usr_msg = actor_agnostic_user_msg_orig.format(alterable_attributes=task.alterable_attributes_str,
                                                            original_query=task.original_query,
                                                            categorical_attributes_str=task.categorical_attributes_str,
                                                            numeric_attributes_str=task.numeric_attributes_str)

        critic_usr_msg = critic_agnostic_user_msg_orig.format(alterable_attributes=task.alterable_attributes_str,
                                                              original_query=task.original_query,
                                                              refinement_objective_str=task.refinement_objective_str,
                                                              constraints_str=task.constraints_str)

    return actor_usr_msg, critic_usr_msg


def get_user_message(task: Task):
    if task.epsilon > 0:
        usr_msg = user_msg_orig_epsilon.format(constraints_str=task.constraints_str,
                                                  alterable_attributes=task.alterable_attributes_str,
                                                  constrained_attributes=task.constrained_attributes,
                                                  categorical_attributes_str=task.categorical_attributes_str,
                                                  numeric_attributes_str=task.numeric_attributes_str,
                                                  refinement_objective_str=task.refinement_objective_str,
                                                  fairness_objective_str=task.fairness_objective_str,
                                                  key_rules_prompt=task.key_rules,
                                                  original_query=task.original_query,
                                                  epsilon=task.epsilon)
    else:
        usr_msg = user_msg_orig.format(constraints_str=task.constraints_str,
                                       alterable_attributes=task.alterable_attributes_str,
                                       constrained_attributes=task.constrained_attributes,
                                       categorical_attributes_str=task.categorical_attributes_str,
                                       numeric_attributes_str=task.numeric_attributes_str,
                                       refinement_objective_str=task.refinement_objective_str,
                                       fairness_objective_str=task.fairness_objective_str,
                                       key_rules_prompt=task.key_rules,
                                       original_query=task.original_query)
    return usr_msg, usr_msg


def run_task(task: Union[Task, AgnosticTask],
             output_tab, chat_tab,
             task_logger: Logger = None, is_agnostic: bool = False):
    # Streamlit chat container
    original_query = task.original_query
    chat_writer = StreamlitChatWriter(original_query, output_tab, chat_tab)
    critic_log = Log("Critic", "Starting the iteration")
    main_log = Log("Main", "Starting the iteration")
    if task_logger is None:
        task_logger = Logger(task.name, task.original_query)
    task_logger.log(critic_log)
    task_logger.log(main_log)
    query_hash_map = QueryHashMap()
    query_hash_map.insert(task.original_query)
    previous_results_and_scores = ""

    constraint_parser = ConstraintParser(
        input_dataset={task.name: task.df},
        constraints_str=task.constraints_str
    )

    @tool
    def extract_query(response: str):
        """
        Extracts the query from actor's response and executes it
        :param response: a response string from the actor that may contain a query
        :return: the alleged query extracted from the response
        """
        query = extract_code(response)
        return query

    @tool
    def get_columns_info_string():
        """
        Returns A List of df's column names
        :return: A string containing the columns and their types in the dataframe.
        Very important to distinct between Integer and Float columns to understand the range you should use, and categorical columns.
        """
        critic_log.append_content("Critic", f"Columns in the dataframe: {task.df.columns.tolist()}")
        info_string = "Dataframe columns:\n"
        for col_name in task.df.columns:
            col = task.df[col_name]
            if is_numeric_dtype(col):
                data_type = "Integer" if 'int' in str(col.dtype) else "Float"
                info_string += f" - '{col_name}': Numerical {data_type} column\n"
                critic_log.append_content("Critic", f" - '{col_name}': Numerical {data_type} column")
            else:
                critic_log.append_content("Critic", f" - '{col_name}': Categorical column")
                info_string += f"- '{col_name}': Categorical column\n"
        task_logger.log(critic_log)
        return info_string

    @tool
    def get_specific_column_info_string(column_name):
        """
        Returns a string containing information about the given column name, such as:
        - unique values for categorical columns
        - min and max values for numerical columns
        :param column_name: The attribute name of the column you would like to get information about
        :return: A string containing information about the columns, such as:
        - unique values for categorical columns
        - min and max values for numerical columns
        """
        col = task.df[column_name]
        if is_numeric_dtype(col):
            min_max_tuple = [col.min(), col.max()]
            data_type = "Integer" if 'int' in str(col.dtype) else "Float"
            info_string = f"Numerical {data_type} column '{column_name}' value range: {min_max_tuple}\n"
        else:
            unique_vals = set(col.unique())
            info_string = f"Categorical column '{column_name}' unique values: {unique_vals}\n"
        critic_log.append_content("Critic", info_string)
        task_logger.log(critic_log)
        return info_string

    @tool
    def validate_query(refined_query):
        """
        Validates the refined SQL SPJ query
        :param refined_query: the refined SQL query to validate
        :return: Indication if the refined query is valid
        """
        is_valid, info_str = task.validation_function(refined_query)
        if is_valid:
            critic_log.append_content("Critic", f"Refined query:\n{refined_query}\n\nis valid.")
            task_logger.log(critic_log)
            return f"Refined query:\n{refined_query}\n\nis valid."
        else:
            critic_log.append_content("Critic",
                                      f"Refined query:\n{refined_query}\n\nis invalid.\nValidation error: {info_str}")
            task_logger.log(critic_log)
            return f"Refined query:\n{refined_query}\n\nis invalid.\n" \
                   f"Validation error: {info_str}\n" \
                   f"In case the query is repeatedly invalid, please consider searching for a different path."

    @tool
    def get_refinement_dist(refined_query):
        """
        Evaluates the refinement distance (lower is better) of the refined SQL SPJ query from the original query.
        :param refined_query: the refined SQL query to evaluate the refinement distance for
        :return: Evaluation of refinement diatance
        """
        refinement_dist = task.refinement_objective(refined_query)
        critic_log.append_content("Critic", f"For the refined query:\n{refined_query}\n\n")
        task_logger.log(critic_log)
        return f"For the refined query:\n{refined_query}\n\n" \
               f"The refinement distance is: {round(refinement_dist, 2)}"

    @tool
    def execute_python_code_line_with_dataframe(code_string: str):
        """
        Execute the provided code string in a context where the dataframe is available.
        Use it to investigate the relations between different values of attributes in the dataframe -
        especially between the values of alterable attributes and the constraints to find the best refinement.
        :param code_string: a string of Python code to execute with the dataframe
        :return: the result of the code execution
        """
        # Create a local context with the dataframe as a variable
        df_name = 'df' if isinstance(task, Task) else task.name
        local_context = {df_name: task.df, 'pd': pd}
        code_string = code_string.replace("AND", "and").replace("OR", "or").replace("IN", "in")
        critic_log.append_content("Critic", f"Executing code:\n{code_string}")
        try:
            # Execute the provided code string in the context where 'df' is available
            code_line = f"result = {code_string}"

            # Execute the provided code string in the context where 'df' is available
            exec(code_line, {}, local_context)

            # Extract the result (if any) from the local context after execution
            critic_log.append_content("Critic", f"Code execution result:\n{local_context.get('result', None)}")
            task_logger.log(critic_log)
            return local_context.get('result', None)

        except Exception as e:
            critic_log.append_content("Critic", f"Error executing code: {e}")
            task_logger.log(critic_log)
            return f"Error executing code: {e}"

    @tool
    def get_constraint_deviation(refined_query):
        """
        Evaluates the deviation of constraint satisfaction (lower is better) of the refined query's output.
        :param refined_query: the refined SQL query to evaluate the fainress deviation for
        :return: Evaluation of fairness deviation
        """
        if isinstance(task, AgnosticTask):
            deviation = task.evaluate_fairness(refined_query)
        else:
            sql_engine = SQLEngine(task.df)
            out_df, is_exc = sql_engine.execute(refined_query)
            if is_exc:
                return f'query invalid. exception: {out_df}'
            deviation = evaluate_constraints(out_df, task.output_constraints)
        dev_str = f"For the refined query:\n{refined_query}\n\n" \
                  f"The fairness deviation is: {round(deviation, 2)}"
        if deviation <= task.epsilon:
            dev_str += "\nThe deviation is below epsilon!"
        else:
            dev_str += f"\nThe deviation is still above epsilon = {task.epsilon}!"
        critic_log.append_content("Critic", dev_str)
        task_logger.log(critic_log)
        return dev_str

    @tool
    def was_query_tried_already(refined_query):
        """
        Checks if the refined query was already tried and failed.
        :param refined_query: the refined SQL query to check
        :return: Indication if the refined query was already tried
        """
        if query_hash_map.is_present(refined_query):
            critic_log.append_content("Critic", f"Query:\n{refined_query}\n\nwas already tried before.")
            task_logger.log(critic_log)
            return f"Query:\n{refined_query}\n\nwas already tried before."
        else:
            query_hash_map.insert(refined_query)
            critic_log.append_content("Critic", f"Query:\n{refined_query}\n\nwas not tried yet.")
            task_logger.log(critic_log)
            return f"Query:\n{refined_query}\n\nwas not tried yet."

    def is_query_valid_and_deviation_under_epsilon(state: ActorCriticAgentState):
        for msg in state['actor_messages'][::-1]:
            if isinstance(msg, HumanMessage) or isinstance(msg, RemoveMessage):
                continue
            content = msg.content
            if len(content) == 0:
                tool_calls = msg.tool_calls
                for tool_call in tool_calls:
                    if tool_call['name'] == "was_query_tried_already":
                        content = tool_call['args']['refined_query']
                        break
            extracted_query = extract_query(content)
            if len(content) > 0 and extracted_query is not None:
                refined_query = extracted_query
                is_valid, _ = task.validation_function(refined_query)
                if is_valid:
                    deviation = task.evaluate_fairness(refined_query)
                    return deviation <= task.epsilon
        return False
    if is_agnostic:
        actor_prompt, critic_prompt = agnostic_actor_system_prompt, agnostic_critic_system_prompt
    else:
        actor_prompt, critic_prompt = get_system_prompts(task)
    abot = ActorCriticAgent([get_columns_info_string, get_specific_column_info_string, was_query_tried_already,
                             execute_python_code_line_with_dataframe],
                            [validate_query, get_constraint_deviation, get_refinement_dist,
                             execute_python_code_line_with_dataframe],
                            original_query=task.original_query,
                            validation_function=task.validation_function, evaluation_function=task.evaluate_fairness,
                            refinement_objective=task.refinement_objective, epsilon=task.epsilon,
                            logger=task_logger, chat_writer=chat_writer,
                            constraint_str=task.constraints_str,
                            constraint_parser=constraint_parser,
                            actor_system=actor_prompt, critic_system=critic_prompt,
                            is_sufficient_func=is_query_valid_and_deviation_under_epsilon)
    abot.graph.config = {'recursion_limit': 1000}
    abot.display_graph()
    if is_agnostic:
        actor_user_msg, critic_user_msg = get_user_messages_agnostic(task)
    else:
        actor_user_msg, critic_user_msg = get_user_message(task)
    orig_critic_user_msg = critic_user_msg
    fairness_deviation = float('inf')
    refinement_dist = float('inf')
    best_query = None
    i = 0
    user_log = Log("User", actor_user_msg)
    while i < MAX_ITERS:
        prev_res_prompt_template = lambda \
            previous_results: f"Given the following previous results and scores:\n\n{previous_results}\n" \
                              f"And given the below task description, give me a valid SPJ Query that is different from all pairs above,\n" \
                              f"completley satisfies ALL fairness constraints, as well as refinement distance value lower than " \
                              f"any of the above. The output must end with a valid SPJ Query given the above instructions, " \
                              f"in the format:\n```sql\n<your refined query>\n```\n"
        # print(f"Task: {task.name}, Epsilon: {task.epsilon}, Iteration: {i + 1}\n")
        split_delimiter = "\n" + "=" * 50 + "\n"
        previous_results_and_scores_split = previous_results_and_scores.split(split_delimiter)
        previous_results_and_scores = split_delimiter + split_delimiter.join(
            previous_results_and_scores_split[-10:]) + split_delimiter
        if i == 0:
            abot.previous_results = previous_results_and_scores

        # print(f"User Message:\n{orig_user_msg}")
        # chat_writer.write("Instructions", orig_user_msg)
        current_prev_res_prompt = prev_res_prompt_template(abot.previous_results) if i > 0 else ""
        critic_usr_msg = f"{current_prev_res_prompt}\n\nTask Description:\n{orig_critic_user_msg}"
        user_log.append_content("User", split_delimiter + f"\nPrevious results provided here\n" + split_delimiter)
        task_logger.log(user_log)
        cur_state = abot.get_state()
        actor_messages = [HumanMessage(content=actor_user_msg)]
        critic_messages = [HumanMessage(content=critic_usr_msg)]
        if len(cur_state.keys()) == 0:
            cur_state = {"actor_messages": actor_messages, "critic_messages": critic_messages}
        else:
            cur_state['actor_messages'] += actor_messages
            cur_state['critic_messages'] += critic_messages

        result = abot.graph.invoke(
            cur_state, config={"configurable": {"thread_id": "42"}}
        )['actor_messages'][-1].content
        Q = extract_code(result)
        if not re.match(r'^SELECT.*FROM.*', Q):
            actor_user_msg += f"\n\nThoughts:\n{Q}"
            # print(f"Task: {task.name}, Epsilon: {task.epsilon}")
            # print(f"Thoughts:\n{Q}")
            user_log.append_content("Agent", f"Thoughts:\n{Q}")
            task_logger.log(user_log)
            continue
        i += 1
        fairness_deviation = task.evaluate_fairness(Q)
        previous_results_and_scores += f"Query:\n{Q}\n\n"
        is_query_valid, info_str = task.validation_function(Q)
        if is_query_valid and fairness_deviation <= task.epsilon:
            new_refinement_dist = task.refinement_objective(Q)
            previous_results_and_scores += f"Valid: True\nFairness Deviation: {round(fairness_deviation, 2)}\n" \
                                           f"Refinement Distance: {round(new_refinement_dist, 2)}\n\n" + "=" * 50 + "\n"
            if new_refinement_dist < refinement_dist:
                refinement_dist = new_refinement_dist
                best_query = Q
                critic_usr_msg = orig_critic_user_msg + prev_res_prompt_template(previous_results_and_scores) + \
                          f"\nNow, provide instructions for the next refinement to be slightly closer to the original, " \
                          f"such that it will still be valid and fair."
                user_log.append_content("User", critic_usr_msg)
                task_logger.log(user_log)
                main_log.append_content("Main", f"Refined query:\n{Q}\n\n"
                                                f"Fairness deviation: {round(fairness_deviation, 2)}\n"
                                                f"Refinement distance: {round(refinement_dist, 2)}")
                task_logger.log(main_log)
            else:
                critic_usr_msg = orig_critic_user_msg + prev_res_prompt_template(previous_results_and_scores) + \
                          f"Provide instructions for the next refinement to be slightly closer to the original, such that it will still be valid and fair."
                user_log.append_content("User", critic_usr_msg)
                task_logger.log(user_log)
                main_log.append_content("Main",
                                        f"Refined query:\n{Q}\n\nFairness deviation: {round(fairness_deviation, 2)}"
                                        f"\nRefinement distance: {round(refinement_dist, 2)}")
                task_logger.log(main_log)
        else:
            if is_query_valid:
                previous_results_and_scores += f"Valid: True\nFairness Deviation: {round(fairness_deviation, 2)}\n\n" + "=" * 50 + "\n"
                critic_usr_msg = orig_critic_user_msg + prev_res_prompt_template(previous_results_and_scores) + \
                          f"Provide instructions for the next refinement such that it will" \
                          f" avoid getting stuck in a local minima by exploring different, " \
                          f"diverse paths."
                user_log.append_content("User", critic_usr_msg)
                task_logger.log(user_log)
                main_log.append_content("Main",
                                        f"Refined query:\n{Q}\n\nFairness deviation: {round(fairness_deviation, 2)}")
                task_logger.log(main_log)
            else:
                previous_results_and_scores += f"Valid: False\nValidation Error: {info_str}\n\n" + "=" * 50 + "\n"
                critic_usr_msg = orig_critic_user_msg + prev_res_prompt_template(previous_results_and_scores) + \
                          f"In case the query is repeatedly invalid, " \
                          f"Provide instructions searching for a different path."
                user_log.append_content("User", critic_usr_msg)
                task_logger.log(user_log)
                main_log.append_content("Main", f"Refined query:\n{Q}\n\nInvalid Query.\nValidation error: {info_str}")
                task_logger.log(main_log)

    constraint_feedback = constraint_parser.evaluate_query(best_query)
    main_log.append_content("Main", f"Best query for epsilon = {task.epsilon}:"
                                    f"\n\n{best_query}\n\n"
                                    f"Best refinement distance: {round(refinement_dist, 2)}")
    task_logger.log(main_log)
    # Don't forget to restore stdout after the function completes
    return best_query, constraint_feedback, chat_writer, fairness_deviation, refinement_dist

