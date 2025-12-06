user_msg_orig = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

The following constraint(s) C that should be satisfied by the output dataset Q'(D):
{constraints_str}

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}
{constrained_attributes}

Given those instructions, the query refinement task is defined as follows:

1. Target Attributes for Refinement:
{categorical_attributes_str}
{numeric_attributes_str}

2. Minimizing Refinement Distance:
{refinement_objective_str}

3. Satisfying Constraints:
    - {constraints_objective_str} 
    - The refined query should completely satisfy all output constraints.

4. Goal:
   - Refine the query so that constraint satisfaction is met while keeping the refinement distance minimal.

Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure that all output constraints are completely satisfied!

Original Query: 
{original_query}

====================================================================================================

Before generating a new refined query, you may utilize the tools you are equipped with, 
especially execute_python_code_line_with_dataframe, to look for patterns in the data 
and the constraints that might be relevant for the refinement.

The overall task is to find a valid refined query as close as possible to the original query (lower refinement distance),
while its output constraints should be completely satisfied!

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.

Your specific role in the refinement process is described in your system prompt. Good Luck!
"""

user_msg_orig_epsilon = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

The following constraint(s) C that should be satisfied by the output dataset Q'(D):
{constraints_str}

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}
{constrained_attributes}

Given those instructions, the query refinement task is defined as follows:

1. Target Attributes for Refinement:
{categorical_attributes_str}
{numeric_attributes_str}

2. Minimizing Refinement Distance:
{refinement_objective_str}

3. Satisfying Constraints:
    - {constraints_objective_str}
    - The refined query should ensure the constraints deviation is less than threshold = {epsilon}.

4. Goal:
   - Refine the query so that constraint satisfaction is met while keeping the refinement distance minimal.

Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure the eventual constraints deviation score is less than threshold = {epsilon}!

Original Query:
{original_query}

====================================================================================================

Before generating a new refined query, you may utilize the tools you are equipped with,
especially execute_python_code_line_with_dataframe, to look for patterns in the data
and the constraints that might be relevant for the refinement.

The overall task is to find a valid refined query as close as possible to the original query (lower refinement distance),
while its constraints deviation should stay lower than {epsilon} - it does not have to be the lowest possible!

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.

Your specific role in the refinement process is described in your system prompt. Good Luck!
"""

system_prompt_template = """You are a smart research assistant. Use the attached tools to solve the Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure that all output constraints are completely satisfied!

Before you calculate the constraints deviation score, you must:
- make sure the query was not tried before, using the 'was_query_tried_already' tool.
- validate the query you have refined, using the 'validate_query' tool.
- if valid, evaluate constraints deviation using `get_constraints_deviation`.

Your task is to find a both valid and completely fair refinement, with as minimal refinement distance as possible.
"""
system_prompt_template_epsilon = """You are a smart research assistant. Use the attached tools to solve the Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure the eventual constraints deviation score is less than threshold = {epsilon}!

Before you calculate the constraints deviation score, you must:
- make sure the query was not tried before, using the 'was_query_tried_already' tool.
- validate the query you have refined, using the 'validate_query' tool.
- if valid, evaluate constraints deviation using `get_constraints_deviation`.

Your task is to find a both valid and fair refinement (constraints deviation < {epsilon}), with as minimal refinement distance as possible.
"""
actor_system_prompt_template = """You are a smart research assistant. Use the attached tools to solve the Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure that all output constraints are completely satisfied!

Before you calculate the constraints deviation score, you must:
- check if the query was tried before, using the 'was_query_tried_already' tool.

Once you found a refinement that has not been tried yet, even if you're not sure about its sufficiency,
YOU MUST return it ENTIRELY (starting by 'SELECT * FROM ...' and so on).
I repeat - return the query ENTIRELY so I could pass it as it is to the critic for evaluation.
"""

actor_system_prompt_template_epsilon = """You are a smart research assistant. Use the attached tools to solve the Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure the eventual constraints deviation score is less than threshold = {epsilon}!

Before you calculate the constraints deviation score, you must:
- check if the query was tried before, using the 'was_query_tried_already' tool.

Once you found a refinement that has not been tried yet, even if you're not sure about its sufficiency,
YOU MUST return it ENTIRELY (starting by 'SELECT * FROM ...' and so on).
I repeat - return the query ENTIRELY so I could pass it as it is to the critic for evaluation.
"""

critic_system_prompt_template = """You are a smart research critic. Use the attached tools to criticize the solution for
 Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. The refinement is ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. The refinement must keep the structure of the query the same.
3. The refinement must minimize changes to the original query.
{key_rules_prompt} Most importantly, make sure that all output constraints are completely satisfied!


Upon receiving a refined query, you must:
- validate the refined query, using the 'validate_query' tool.
- if valid, evaluate constraints deviation using `get_constraints_deviation`.
- if the constraints deviation is less than the threshold, evaluate the refinement distance using `get_refinement_dist`.

Your task is to criticize the refinement for being both valid and fair (constraints deviation < threshold), with as minimal refinement distance as possible.
After assessing the refinement, return a feedback in natural language. Do not suggest any explicit SQL queries, only feedback.
You may suggest general directions for improvement, but do NOT provide the exact solution.
"""

critic_system_prompt_template_epsilon = """You are a smart research critic. Use the attached tools to criticize the solution for
 Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. The refinement is ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. The refinement must keep the structure of the query the same.
3. The refinement must minimize changes to the original query.
 {key_rules_prompt} Most importantly, make sure the eventual constraints deviation score is less than threshold = {epsilon}!


Upon receiving a refined query, you must:
- validate the refined query, using the 'validate_query' tool.
- if valid, evaluate constraints deviation using `get_constraints_deviation`.
- if the constraints deviation score is less than the threshold, evaluate the refinement distance using `get_refinement_dist`.

Your task is to criticize the refinement for being both valid and fair (constraints deviation < threshold), with as minimal refinement distance as possible.
After assessing the refinement, return a feedback in natural language. Do not suggest any explicit SQL queries, only feedback.
You may suggest general directions for improvement, but do NOT provide the exact solution.
"""


# Agnostic Prompt Templates

# parameters to provide: alterable_attributes, categorical_attributes_str, numeric_attributes_str, refinement_objective_str, constraints_str, original_query



actor_agnostic_user_msg_orig = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}
Important: regarding numerical attributes, make sure the refinements you generate use of the same dtype as the original query,
namely, for integer attributes, the refinements should be integers, and for float attributes, the refinements should be floats.

Given those instructions, the query refinement task is defined as follows:

1. Original Query: 
{original_query}

2. Target Attributes for Refinement:
{categorical_attributes_str}
{numeric_attributes_str}

3. For each refinement you generate, you will receive a feedback on the validity, constraint satisfaction and refinement distance.
Your task is to refine the query so that constraint satisfaction is met while keeping the refinement distance minimal.

====================================================================================================

Before generating a new refined query, you must make sure the query was not tried before.

Your primary task is to refine the query iteratively to meet your received feedback. To fulfill your instructions effectively:
a. You are very encouraged to utilize the 'execute_python_code_line_with_dataframe' tool to run exploratory pandas queries on the dataset.
b. When analyzing the dataset, ensure you:
    - Identify patterns, specific values, or ranges that match the critic's suggestions.
    - Validate the inclusion or exclusion of certain values based on the query results.
c. Before finalizing each refinement, use the tool to confirm your reasoning by checking specific conditions or distributions in the data.
d. Document the logic behind your decisions, explaining how the results of your queries influenced the refinement.
For example:
- If the feedback suggests exploring a specific attribute, write and execute a query to inspect its unique values or their distribution.
- Use filtering or aggregation queries to identify subsets of data relevant to the refinement.

The overall task is to find a valid refined query as close as possible to the original query (lower refinement distance),
while its constraints should be completely satisfied!

YOU ARE NOT ALLOWED TO ADD OR REMOVE ANY CONDITIONS FROM THE ORIGINAL QUERY. ONLY MODIFY THE EXISTING CONDITIONS.

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.
"""

actor_agnostic_user_msg_orig_epsilon = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}
Important: regarding numerical attributes, make sure the refinements you generate use of the same dtype as the original query,
namely, for integer attributes, the refinements should be integers, and for float attributes, the refinements should be floats.

Given those instructions, the query refinement task is defined as follows:

1. Original Query: 
{original_query}

2. Target Attributes for Refinement:
{categorical_attributes_str}
{numeric_attributes_str}

3. For each refinement you generate, you will receive a feedback on the validity, constraint satisfaction and refinement distance.
Your task is to refine the query so that constraint satisfaction is met up to threshold = {epsilon}, while keeping the refinement distance minimal.

====================================================================================================

Before generating a new refined query, you must make sure the query was not tried before.

Your primary task is to refine the query iteratively to meet your received feedback. To fulfill your instructions effectively:
a. You are very encouraged to utilize the 'execute_python_code_line_with_dataframe' tool to run exploratory pandas queries on the dataset.
b. When analyzing the dataset, ensure you:
    - Identify patterns, specific values, or ranges that match the critic's suggestions.
    - Validate the inclusion or exclusion of certain values based on the query results.
c. Before finalizing each refinement, use the tool to confirm your reasoning by checking specific conditions or distributions in the data.
d. Document the logic behind your decisions, explaining how the results of your queries influenced the refinement.
For example:
- If the feedback suggests exploring a specific attribute, write and execute a query to inspect its unique values or their distribution.
- Use filtering or aggregation queries to identify subsets of data relevant to the refinement.

The overall task is to find a valid refined query as close as possible to the original query (lower refinement distance),
while its constraint deviation score should stay lower than {epsilon} - it does not have to be the lowest possible!

Based on your findings from the dataset, answer those intriguing questions before providing your final refinement:
For numerical alterable predicates (if exist in the query): 
- [Should we Lower / Raise the value of the predicate?]
- [By how much?]
- [Explain why! based on your findings from the dataset]
For categorical alterable predicates (if exist in the query): 
- [Should we Include / Exclude categories from the predicate?]
- [Which ones?]
- [Explain why! based on your findings from the dataset]

Once you have reached a satisfactory refinement, provide your final output in the following format:
[Provide a brief explanation of which python queries you ran on the dataset and why,
 what new relevant information you found that was interesting, and how it influenced your refinement decisions]

```sql
[Insert your final refined SQL query here]
```

Important: You should usually refine more than one predicate to reach a valid refinement, so try cross-examining
different combinations of values for the allowed attributes to find the best refinement.

YOU ARE NOT ALLOWED TO ADD OR REMOVE ANY CONDITIONS FROM THE ORIGINAL QUERY. ONLY MODIFY THE EXISTING CONDITIONS.

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.
"""

ACTOR_POSTFIX = """Your response should be concise and short and follow the exact following format, chronologically:

Step 1: [Based on the received instructions, summarize which 2-3 correlations between different attributes' values you want to explore and why.
Recall that the dataset is huge and you should focus on the most relevant and concise correlations.]
(Step 1.5: [Optionally call 'get_columns_info_string' and 'get_specific_column_info_string' tools to get more information about the attributes in the dataset and their values])
Step 2: [Based on the correlations mentioned in Step 1, provide a list of specific pandas queries you want to run on the dataset to explore these correlations]
Step 3: [Call the 'execute_python_code_line_with_dataframe' tool with pandas python one-liners to run each of the queries you mentioned in Step 2]
Step 4: [Provide the results and explain what new relevant information you found that was interesting, and how it influenced your refinement decisions]
Step 5: [Provide your final output in the following format]
```sql
[Insert your final refined SQL query here]
```
Step 6: [Explicitly state the reasoning behind your final refinement]

YOU RUN EACH STEP EXACTLY ONE TIME, AND YOU MUST PROVIDE THE OUTPUT OF EACH STEP BEFORE PROCEEDING TO THE NEXT.

Important: You should usually refine more than one predicate to reach a valid refinement, so try cross-examining
different combinations of values for the allowed attributes to find the best refinement.

YOU ARE NOT ALLOWED TO ADD OR REMOVE ANY CONDITIONS FROM THE ORIGINAL QUERY. ONLY MODIFY THE EXISTING CONDITIONS.
"""

critic_agnostic_user_msg_orig = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}
Important: regarding numerical attributes, make sure your corresponding suggested values are of the same dtype as the original query,
namely, for integer attributes, the values should be integers, and for float attributes, the values should be floats.

Given those instructions, the query refinement task is defined as follows:

1. Original Query: 
{original_query}

2. Minimizing Refinement Distance:
{refinement_objective_str}

3. Satisfying Constraints:
The following constraint(s) C that should be satisfied by the output dataset Q'(D):
{constraints_str} 
The refined query should completely satisfy all output constraints.

4. Your task is to provide natural language feedback on the refinement, without suggesting any explicit SQL queries,
   so that the next refinement would most likely satisfy the constraints while keeping the refinement distance minimal.
   
====================================================================================================

Before providing your refinement instructions, you may utilize the tools you are equipped with to 
evaluate the constraint deviation of the refined query using the 'get_constraints_deviation' tool
as well as the refinement distance using the 'get_refinement_distance' tool.

Explicitly discuss previous refinements provided to you and their corresponding scores, 
as a reference to help the actor investigate the dataset and refine the query more effectively.

Then, you should provide feedback on the refinement, focusing on the validity and constraints satisfaction of the refined query,
w.r.t. the constraints and the constraints deviation threshold, in comparison to all previous refinements provided to you.

As a critic, your role is to evaluate the actor's decisions and provide actionable feedback. Remember that the actor has the ability to run simple one-liner pandas queries on the dataset to extract insights. When suggesting improvements for the next refinement:
a. Identify specific aspects of the dataset that the actor should explore further.
b. Suggest concrete pandas queries that the actor can run to discover values to include or exclude.
c. Emphasize using these queries to deduce patterns, ranges, or categories that align with the constraints (even though the actor does not directly receive the constraints).
d. If the actor added a predicate that invalidates the query, tell it plainly that adding predicates is wrong and they must remove it.

Note that this is a multi-objective optimization problem, where a BALANCE between the two objectives is essential.

For example, if the actor tried multiple values above the original for a certain predicate, you must suggest
trying values below the original to minimize the refinement distance, and vice versa.

DO NOT Suggested Refinements!!!!
only instruct what should be checked from the dataset in order to provide a better refinement.
"""

critic_agnostic_user_msg_orig_epsilon = """
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes}

Important: Make it clear to the Actor that the above attributes are the ONLY predicates it is allowed to change their value.
And no other attributes should be added or removed from the query.
 
Important: DO NOT suggest any actual values, only guide the actor to how you suggest to explore the dataset to find the 
right values, given the constraints and requirements on query structure.

Given those instructions, the query refinement task is defined as follows:

1. Original Query: 
{original_query}

2. Minimizing Refinement Distance:
{refinement_objective_str}


3. Satisfying Constraints:
The following constraint(s) C that should be satisfied by the output dataset Q'(D):
{constraints_str} 
- The refined query should ensure the constraint deviation is less than threshold = {epsilon}.

4. Your task is to provide natural language feedback on the refinement, without suggesting any explicit SQL queries,
   so that the next refinement would most likely satisfy the constraints while keeping the refinement distance minimal.


====================================================================================================

Before providing your refinement instructions, you may utilize the tools you are equipped with to 
evaluate the constraint deviation of the refined query using the 'get_constraints_deviation' tool
as well as the refinement distance using the 'get_refinement_distance' tool.

Explicitly discuss previous refinements provided to you and their corresponding scores, as a reference to help the actor
investigate the dataset and refine the query more effectively. if the current refinement satisfied less constraints than 
a previous one, guide the actor to explore refinements closer to the more successful refinement instead!

Use the following format:
[The previously suggested refinement:
<best refinement yet>
Was better in satisfying the <constraints 1, 2, 3....?>. Therefore, let's explore other refinements similar to it.] 

Ask intriguing questions before providing your final refinement:
For numerical alterable predicates (if exist in the query): 
- [Should we Lower / Raise the value of the predicate?]
- [By how much?]
- [Can you explain why, based on your findings from the dataset?]
For categorical alterable predicates (if exist in the query): 
- [Should we Include / Exclude categories from the predicate?]
- [Which ones?]
- [Can you explain why, based on your findings from the dataset?]

Then, you should provide feedback on the refinement, focusing on the validity and constraints satisfaction of the refined query,
w.r.t. the constraints and the constraints deviation threshold, in comparison to all previous refinements provided to you.

As a critic, your role is to evaluate the actor's decisions and provide actionable feedback. Remember that the actor has the ability to run simple one-liner pandas queries on the dataset to extract insights. When suggesting improvements for the next refinement:
a. Identify specific aspects of the dataset that the actor should explore further.
b. Suggest concrete pandas queries that the actor can run to discover values to include or exclude.
c. Emphasize using these queries to deduce patterns, ranges, or categories that align with the constraints (even though the actor does not directly receive the constraints).
d. If the actor added a predicate that invalidates the query, tell it plainly that adding predicates is wrong and they must remove it.

Note that this is a multi-objective optimization problem, where a BALANCE between the two objectives is essential.

For example, if the actor tried multiple values above the original for a certain predicate, you must suggest
trying values below the original to minimize the refinement distance, and vice versa.
Only instruct what should be checked from the dataset in order to provide a better refinement.

DO NOT Suggest any concrete refinements, nor suggest specific values to be used in the refinement.
"""

CRITIC_POSTFIX = """Upon receiving a new refinement, your response should be concise and short and follow the exact following format, chronologically:

Step 1: [Call the 'get_refinement_dist' and 'get_constraints_deviation' tools to evaluate the current refinement]
Step 2: [Compare the current refinement with the previous refinements in the predicate resolution, in terms of constraints satisfaction and refinement distance]
Step 3: [Explain which of the constraints were satisfied and which were not, and what are the possible reasons for that]
Step 4: [Describe and explain 2-3 specific consecutive correlations between the query attributes' and constraint-related attributes in the dataset you suggest exploring to provide a better refinement]
Step 5: [For each of the aspects mentioned in Step 4, explain what the possible results would mean for the next refinement to satisfy the constraints.
Make sure to avoid suggesting actual values, only guide the user on how to explore the dataset to find the right values]
DO NOT Suggest any concrete refinements.

"""

agnostic_actor_system_prompt = """You are a smart research assistant. 
Use the attached tools to solve the Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. You are ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. Keep the structure of the query the same.
3. Minimize changes to the original query.
4. You can do the following:
    - For categorical attributes, you can adjust the existing predicates to either exclude or include certain values.
    - For numeric attributes, you can adjust the existing predicates to either increase or decrease the threshold.
5. The refined query result must contain at least 20 records.
6. Most importantly, make sure that all output constraints are completely satisfied!

Before you calculate the constraints deviation score, you must:
- make sure the query was not tried before.

Once you found a refinement that has not been tried yet, even if you're not sure about its sufficiency,
YOU MUST return it ENTIRELY (starting by 'SELECT * FROM ...' and so on).
I repeat - return the query ENTIRELY so I could pass it as it is to the critic for evaluation.


Your response should be short and concise, focusing on the changes made to the query and the reasoning behind them.

To avoid meaningless refinements, make sure to refine values in a meaningful way, for example:
- if you have a numeric attribute where the values are at least 0.1 far apart, refinements should be in increments of 0.1 at least.
- if you have a numeric attribute where the values are basically integers, refinements should be in integers too (step size of 1 or more).

Do not repeat the same refinement multiple times, try to explore different directions in each refinement.
Remember that you may balance the refinement by RAISING one predicate and LOWERING another, to minimize the refinement distance.
"""

agnostic_critic_system_prompt = """You are a smart research critic. Use the attached tools to criticize the solution for
 Query Refinement task you will be provided with.
Key Rules to Follow for a Valid Refinement:
1. The refinement is ONLY allowed to adjust the existing WHERE clause predicates. Do NOT introduce ANY predicates regarding
 attributes the DO NOT exist in the original query!
2. You can do the following:
    - For categorical attributes, you can adjust the existing predicates to either include or exclude certain values.
    - For numeric attributes, you can adjust the existing predicates to either increase or decrease the threshold.
3. The refinement must keep the structure of the query the same.
4. The refinement must minimize changes to the original query.
5. The refined query result must contain at least 20 records.
6. Most importantly, make sure that all output constraints are completely satisfied!


Upon receiving a refined query, you must:
- validate the refined query, using the 'validate_query' tool.
- if valid, evaluate constraints deviation using `get_constraints_deviation`.
- if the constraints deviation is less than the threshold, evaluate the refinement distance using `get_refinement_dist`.

Your task is to provide feedback about the refinement for being both valid and fair (constraints deviation < threshold), with as minimal refinement distance as possible.
After assessing the refinement, return a feedback in natural language. Do not suggest any explicit SQL queries, only feedback.
You may suggest general directions for improvement, but do NOT provide the exact solution.

Your response should be short and concise, no more than 3-4 sentences, 
focusing on instruction for what to explore and how to interpret the results to the next refinement,
based on the given information.
DO NOT Suggest any concrete refinements, nor suggest specific values to be used in the refinement.
Only provide instructions what should be checked from the dataset in order to provide a better refinement.
"""


REFINEMENT_QUERY_DISTANCE = """
import pandas as pd
from typing import Dict, List

def get_refinement_distance_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
    # Parse the original query to extract predicates
    original_where_clause = extract_where_clause(original_query)
    parsed_original = parse_where_clause(original_where_clause)

    def refinement_distance_objective(refined_query: str) -> float:
        refined_where_clause = extract_where_clause(refined_query)
        numeric_distance = 0
        categorical_distance = 0
        parsed_refined = parse_where_clause(refined_where_clause)
        
        for categorical_predicate in parsed_refined.categorical:
            for original_predicate in parsed_original.categorical:
            
                # Calculate differences for categorical attributes
                if categorical_predicate.name == original_predicate.name:
                    # Calculate the 1 - Jaccard similarity between the two sets of values
                    original_values = set(original_predicate.values)
                    refined_values = set(categorical_predicate.values)
                    intersection = original_values.intersection(refined_values)
                    union = original_values.union(refined_values)
                    predicate_distance = (1 - (len(intersection) / len(union))) if union else (1 if intersection else 0)
                    
                    # Add the distance to the total categorical distance
                    categorical_distance += predicate_distance
                    
        for numerical_predicate in parsed_refined.numerical:
            for original_predicate in parsed_original.numerical:
            
                refined_name, orig_name = numerical_predicate.name, original_predicate.name
                refined_op, orig_op = numerical_predicate.operator, original_predicate.operator
                
                # Calculate differences for numerical attributes
                if refined_name == orig_name and refined_op == orig_op:
                    # Calculate the relative difference between the two values
                    refined_val, orig_val = numerical_predicate.value, original_predicate.value
                    predicate_distance = abs(refined_val - orig_val) / orig_val
                    
                    # Add the distance to the total numerical distance
                    numeric_distance += predicate_distance
        return categorical_distance + numeric_distance

    return refinement_distance_objective
"""

REFINEMENT_QUERY_DESCRIPTION = """
The refinement is calculated based on the differences between the original and refined queries.
The distance is calculated as the sum of the differences between the categorical and numerical predicates.
For categorical predicates, the distance is calculated as 1 - Jaccard similarity between the two sets of values.
For numerical predicates, the distance is calculated as the relative difference between the two values.
The final distance is the sum of the categorical and numerical distances.
"""

REFINEMENT_RESULT_DISTANCE = """
import pandas as pd
from typing import Dict, List

def get_refinement_distance_objective(d_in: Dict[str, pd.DataFrame], original_query: str):
    sql_engine = SQLEngineAdvanced(d_in, keep_index=True)
    original_query_output, _ = sql_engine.execute(original_query)

    def refinement_distance_objective(refined_query: str) -> float:
        refined_query_output, _ = sql_engine.execute(refined_query)
        
        # compute 1 - Jaccard Similarity between the output_dataset and the original_query_output
        union = len(set(refined_query_output.index.to_list()).union(set(original_query_output.index.to_list())))
        intersection = len(set(refined_query_output.index.to_list()).intersection(set(original_query_output.index.to_list())))
        return 1 - intersection / union

    return refinement_distance_objective
"""


REFINEMENT_RESULT_DESCRIPTION = """
The refinement distance is calculated as 1 - Jaccard similarity between the output result of the original query and the refined query.
The Jaccard similarity is calculated as the intersection of the two sets of indices divided by their union.
The final distance is 1 - Jaccard similarity.
"""

FUNCTION_TYPES = """The functions you will be responsible to generate are nested functions, 
that are used to assess the quality of a refined query. Each nested function consists of an outer function and 
within it an inner function. The functions are used to assess the quality of a refined query, as follows:
- The outer function is a function that takes in the input dataset as a pandas dataframe, as well as the original query. 
- The inner function accepts the refined query as input according to the parameters given in the outer function.
Each outer function shall return its inner function as a runnable function.
This way, a task specific inner function can be re-used multiple times with various refined queries along the refinement
process. 

Note that the inner functions must return a float value that represents the quality of the refined query output, 
which is by definition a non-negative float, such that a lower value indicates a better quality.

The nested functions are of two types, where each type has a specific signature:

- Constraint Satisfaction Objective: 
    * Description: A function that calculates the deviation of the output of the refined query from satisfying a set of constraints.
        You will be provided with a list of constraints that the output of the refined query should satisfy.
        The deviation should be a weighted sum of the constraints that are not satisfied, and if not provided
        explicitly, the constraints should be assumed to be equally weighted. Then, for each constraint,
        the deviation should represent the absolute difference between the required constraint and the actual value
        in the output of the refined query. The constraints deviation score should be a non-negative float,
        where a lower value indicates a better quality.
    * Outer function signature: get_constraints_satisfaction_objective(d_in: dict[str, pd.DataFrame], original_query: str)
    * Inner function signature: constraints_satisfaction_objective(refined_query: str) -> float (constraints deviation score)
        The Inner function output MUST be a normalized float value between 0 and 1,
        where 0 indicates that all constraints are satisfied and 1 indicates that none of the constraints are satisfied.

- Refinement Distance Objective: 
    * Description: A function that calculates the distance between the original query and the refined query,
        optionally based on the output of the refined query. You will be provided with a description of the refinement
        in natural language, that states which parameters on either the refined query or the output of the refined query
        should be considered to calculate the distance. The distance should be a non-negative float, where a lower value
        indicates a better quality.
    * Outer function signature: get_refinement_distance_objective(d_in: dict[str, pd.DataFrame], original_query: str)
    * Inner function signature: refinement_distance_objective(refined_query: str) -> float (refinement distance score)
"""

IMPLEMENTER_SYSTEM_PROMPT = f"""You are a python function implementation agent.
The function can use only the pandas library and the below explicitly provided tools. NO NUMPY!
You are responsible for generating functions that assess the quality of a refined query.
You have been provided with a dataset and an original query.

{FUNCTION_TYPES}

The query is a full SELECT-PROJECT-JOIN query.
Within the function, you will have access to the dataset and the original query as a dictionary of string to pandas dataframe.
You are also provided with a class called SQLEngineAdvanced that can be used to execute SQL queries on the dataset 
in the following way:

```python 
    sql_engine = SQLEngineAdvanced(d_in_dict)
    resp_df, error = sql_engine.execute(query)
    if error:
        return "An error occurred: " + resp_df
    return resp_df
```

Using the SQLEngineAdvanced class, you can execute SQL queries on the dataset and get the result as a pandas dataframe,
to further use to assess the quality of the refined query's output.

Assume you are already provided with the following tool, which you highly encouraged to use in your function for parsing the WHERE clause predicates of an SQL query:

```python 
parse_where_predicates(sql_query: str) -> Dict[Literal['categorical', 'numerical'], List[Dict]]
```

When called with an SQL query, this function parses the WHERE clause of an SQL query, extracts the conditions as a dictionary, 
and generates a dictionary with two keys: 'categorical' and 'numerical', each containing a list of conditions.

The dictionary is structured as follows:
{{
    'categorical': [
        {{
            'attribute_name': 'column_name',
            'in_values': ['value1', 'value2', ...]
        }},
        ...
    ],
    'numerical': [
        {{
                'attribute_name': 'column_name',
                'operator': op   # (e.g., '>', '<=', '!='),
                'value': value  # (e.g., 10, 3.14)
        }},
        ...
    ]
}}

Make sure to NOT override the provided tool's signature, as it will be used to evaluate your function.
"""

PARSING_MODEL_SYSTEM_MESSAGE = """You are an agent responsible for parsing constraints from a user query.
You receive a string containing the constraints and should return a list of constraints, as follows:
- Each constraint should be a dictionary with the following keys:
    * query (str): a callable query that accepts the dataframe as input and returns a numeric value. 
                    the query should refer to the dataframe as 'df' and should return a numeric value 
                    that represents the constraint evaluation over the dataframe. 
                    For example, if the constraint is:
                    "the number of rows out of the top 10 rows where column 'A' has value 'B' must be more than 5"
                    the query should be: "sum(df[:10]['A'] == 'B')"  
    * description (str): a description of the query, i.e. the measurable term of the constraint in natural language.
                        For example, "The number of top 10 employees for which 'A' is 'B'"
    * symbol (str): the comparison operator ("<", ">", "<=" or ">=")
    * desired_value (int or float): the (minimum / maximum) value that must satisfy the constraint 
- The constraints should be returned as a list of dictionaries:
    [
        {'query': query1, 'description': description1, 'symbol': symbol1, 'desired_value': desired_value1},
        ...
        {'query': queryN, 'description': descriptionN, 'symbol': symbolN, 'desired_value': desired_valueN}
    ]

Use the tools provided to you to extract the necessary information dataframe: 
- get_dataset_information for precise value names or ranges for each of the relevant specific columns in the result df of the original query.


IMPORTANT: The desired_value cannot be ZERO!!! As it will be used as a denominator in the constraints deviation calculation.
Thus, in case you think the desired value should be zero, consider inverting the symbol and changing the desired_value and 
query accordingly (like 1 - something)...

Also do not use any libraries other than pandas
"""


