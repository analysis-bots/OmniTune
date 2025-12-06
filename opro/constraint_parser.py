from typing import Dict, Union, List, Any, Optional

import duckdb
import pandas as pd
from pydantic import BaseModel

from chat_model import get_chat_model, ChatModel
from tools.utils import register_duckdb_tables, get_primary_dataframe
from functionality.constraint import AgnosticConstraint, OutputConstraint
from opro.opro_prompt_templates import PARSING_MODEL_SYSTEM_MESSAGE

# Pydantic model for structured constraint parsing
class ConstraintDict(BaseModel):
    # Define fields based on the *actual* JSON structure returned by the model
    query: str
    description: str
    symbol: str # This seems to map to the operator
    desired_value: Union[int, float, str, bool] # Renamed from 'value'

class ConstraintList(BaseModel):
    constraints: list[ConstraintDict]

class ConstraintParser:
    def __init__(self, input_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], constraints_str: str, original_query: str = None):
        # Register tables (single or multi) and normalize to a primary df for pandas-based logic
        register_duckdb_tables(input_dataset)
        self.input_dataset = get_primary_dataframe(input_dataset)
        self.constraints_str = constraints_str
        self.chat_model: ChatModel = get_chat_model() # Use the chat model factory
        # Execute original query if provided to get initial result schema for context
        self.original_query_result_info = "" # Store schema info
        self.query_result = None
        if original_query:
            try:
                self.query_result = duckdb.query(original_query).to_df()
            except Exception as e:
                print(f"Warning: Could not execute original query: {e}")
                self.original_query_result_info = None
        
        self.json_response = None  # Keep for potential debugging
        self.parsed_constraints = self.parse_constraints()

    def get_results(self, query: str):
        """
        Executes the provided SQL query on the input dataset and returns the result.
        """
        df = self.input_dataset
        query_result = duckdb.query(query).to_df()
        constraint_vals = {
            c.query_str: c.query(query_result)
            for c in self.parsed_constraints
        }
        return constraint_vals

    def get_constraints_list_as_str(self):
        """
        Returns the parsed constraints as a string.
        """
        if not self.parsed_constraints:
            return "No constraints parsed."

        constraints_list = []
        for constraint in self.parsed_constraints:
            constraints_list.append(str(constraint))
        return "\n".join(constraints_list)


    def parse_constraints(self):
        """Parses the constraints string using the configured chat model's structured output."""
        
        system_message = PARSING_MODEL_SYSTEM_MESSAGE + (
             f"\nFor context, here is the schema of the original query result:\n{self.original_query_result_info}"
             if self.original_query_result_info else ""
        )

        user_message = (
            f"Parse the following constraints into a JSON list of dictionaries, conforming to the required schema.\n"
            f"Ensure the response is a JSON object with a single key 'constraints' containing the list.\n"
            f"**Never** define a constraint query by 'len(df) - len(df[df['attribute'] != value])'!\n"
            f"Instead, **always** use the form 'len(df[df['attribute'] == value]), "
            f"both for lower and upper bounds constraints.\n"
            f"Constraints:\n"
            f"{self.constraints_str}\n"
            f"Desired value must be a non-zero positive numeric value!\n"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        try:
            structured_response = self.chat_model.generate_structured(
                messages=messages,
                response_model=ConstraintList, # Expect a ConstraintList object
                temperature=0.0

            )
            # Store the raw JSON response potentially for debugging
            self.json_response = structured_response.model_dump_json(indent=2) 
            constraint_dicts = structured_response.constraints
            parsed_constraints = [AgnosticConstraint.from_dict(c.model_dump()) for c in constraint_dicts]
            return parsed_constraints
        except ValueError as e:
             print(f"Error parsing constraints using structured output: {e}")
             # Fallback or error handling - maybe try ast.literal_eval on raw response? Risky.
             # For now, return empty list or raise
             print("Returning empty list of constraints due to parsing error.")
             self.json_response = f"Error: {e}" # Store error
             return []
        except Exception as e:
             print(f"An unexpected error occurred during constraint parsing: {e}")
             self.json_response = f"Error: {e}" # Store error
             return []

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
        df = self.input_dataset
        res_df = duckdb.query(refined_query).to_df()
        constraint_bullets = ["Constraint-wise feedback:"]
        for constraint in self.parsed_constraints:
            if constraint.evaluate(res_df) > 0:
                constraint_bullets.append(f"    * {constraint.string_evaluation(res_df)} (Not satisfied)")
            else:
                constraint_bullets.append(f"    * {constraint.string_evaluation(res_df)} (Satisfied)")
        return "\n".join(constraint_bullets)

    def evaluate_query(self, refined_query: str):
        df = self.input_dataset
        refined_df = duckdb.query(refined_query).to_df()
        return self.evaluation_list(refined_df)

    def get_json_response(self):
        return self.json_response

    def evaluate_constraint_score(self, refined_query: str):
        df = self.input_dataset
        refined_df = duckdb.query(refined_query).to_df()
        sum_score = sum([constraint.evaluate(refined_df) for constraint in self.parsed_constraints]) \
                    / len(self.parsed_constraints)
        return sum_score

    def is_satisfied(self, refined_query: str, epsilon=0):
        constraint_score = self.evaluate_constraint_score(refined_query)
        return constraint_score <= epsilon

    def get_constraints_str(self):
        """
        Returns the parsed constraints as a string.
        """
        if not self.parsed_constraints:
            return "No constraints parsed."

        constraints_list = []
        for constraint in self.parsed_constraints:
            constraints_list.append(str(constraint))
        return "\n".join(constraints_list)


class MockParser:
    def __init__(self, parsed_constraints: List[OutputConstraint], input_dataset: pd.DataFrame, constraints_str: str, original_query: Optional[str] = None):
        """
        Mock parser that uses pre-defined constraints instead of parsing them from a string.
        """
        self.input_dataset = input_dataset
        self.constraints_str = constraints_str
        self.original_query_result_info = ""  # Store schema info
        self.query_result = None
        if original_query:
            try:
                df = input_dataset
                self.query_result = duckdb.query(original_query).to_df()
            except Exception as e:
                print(f"Warning: Could not execute original query: {e}")
                self.original_query_result_info = None

        self.parsed_constraints = parsed_constraints
        self.json_response = None

    def evaluate_constraint_score(self, refined_query: str):
        df = self.input_dataset
        refined_df = duckdb.query(refined_query).to_df()
        sum_score = sum([constraint.evaluate(refined_df) for constraint in self.parsed_constraints]) \
                    / len(self.parsed_constraints)
        return sum_score

    def get_results(self, query: str):
        """
        Executes the provided SQL query on the input dataset and returns the result.
        """
        df = self.input_dataset
        try:
            if isinstance(df, dict):
                con = duckdb.connect()
                for table_name, table_df in df.items():
                    con.register(table_name, table_df)
                query_result = con.execute(query).df()
            else:
                query_result = duckdb.query(query).to_df()
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        try:
            constraint_vals = {
                c.get_query_str(): c.get_query_value(query_result)
                for c in self.parsed_constraints
            }
        except AttributeError:
            constraint_vals = {
                c.query_str: c.query(query_result)
                for c in self.parsed_constraints
            }

        return constraint_vals

    def get_constraints_list_as_str(self):
        """
        Returns the parsed constraints as a string.
        """
        if not self.parsed_constraints:
            return "No constraints parsed."

        constraints_list = []
        for constraint in self.parsed_constraints:
            constraints_list.append(str(constraint))
        return "\n".join(constraints_list)

    def is_satisfied(self, refined_query: str, epsilon=0):
        constraint_score = self.evaluate_constraint_score(refined_query)
        return constraint_score <= epsilon

    def evaluation_str(self, refined_query: str):
        df = self.input_dataset
        res_df = duckdb.query(refined_query).to_df()
        constraint_bullets = ["Constraint-wise feedback:"]
        for constraint in self.parsed_constraints:
            if constraint.evaluate(res_df) > 0:
                constraint_bullets.append(f"    * {constraint.string_evaluation(res_df)} (Not satisfied)")
            else:
                constraint_bullets.append(f"    * {constraint.string_evaluation(res_df)} (Satisfied)")
        return "\n".join(constraint_bullets)
