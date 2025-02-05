# OmniTune: A universal framework for query refinement via LLMs

![OmniTune](https://github.com/user-attachments/assets/3b4856d1-9c4b-4b79-bee9-b518a3deca58)


### Table of Contents:
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Configurations](#config)
4. [Usage](#usage)
5. [Prompts & Prompt Templates](#prompts)
   - [System Prompts](#system_prompts)
   - [User Prompt Templates](#user_prompts)
   - [Constraint Generation Prompts](#constraint_prompts)

## 1. Introduction: <a name="introduction"></a>
OmniTune is a novel universal framework for query refinement using an LLM multi-agent approach. 
The architecture of OmniTune consists of two main components:
1. The Refinement Problem Wizard; helps users formulate a valid, universal refinement problem.
2. The OmniTune Refinement Engine; builds upon the Actor-Critic paradigm adapted into an LLM multi-agent architecture, and used to effectively refine the input query.

<img width="780" alt="OmniTune Workflow" src="https://github.com/user-attachments/assets/569f4f0d-8973-402d-92c8-7e40f8f97a6d" />

This repository contains the reproducibility package for running our demo.
For any issues regarding reproducibility or anything else using the implementation, please feel free to create a GitHub issue or send an Email.

## 2. Requirements: <a name="requirements"></a>
The system runs on Python >= 3.11.6
To install the required package, cd to main dir, then run:

> pip install -r requirements.txt

The OmniTune UI is built upon the Streamlit[^1] package.
For an optimal UI view, the preferred resultion is 1920 x 1080.

## 3. Configuration: <a name="config"></a>
OmniTune Demo currently supports only OpenAI models.
Before running OmniTune locally, you should get an OpenAI API-Key[^2].
Copy the API-Key paste it inplace of `<YOUR_OPENAI_API_KEY>` in the following line within the file **config.py**:

```
OPENAI_API_KEY = '<YOUR_OPENAI_API_KEY>'
```

Note about pricing: OmniTune defaultly uses GPT-4o-mini model to reduce runtime and costs to minimum.
If you would like to test it with any other model by OpenAI, you may do it using the **config.py** parameter `MODEl`.
More about OpenAI API models pricing is provided [here](https://openai.com/api/pricing/).
 

## 4. Usage: <a name="usage"></a>

### Running OmniTune:
1. Make sure Streamlit is installed successfully:
  > streamlit --version
2. On the main dir, run:
  > streamlit run app.py
3. A new browser tab should open with the system, else go to http://localhost:8501 manually.

### Using OmniTune:
#### Step 1 - Select database and insert query:
The left half of the OmniTune system on Step 1:

<img width="650" alt="Step 1 - Left Half" src="https://github.com/user-attachments/assets/d9da3fe8-e6c9-4ed4-be04-bf866952f3a9" />

- To select a database from the provided list, press on the toggle arrow (1). 
- Alternatively, to upload your own csv file press on the Browse Files button (2).
  A note will be shown in green below the button if successfully uploaded.
- On the right side of the screen you should see a preview of the selected / uploaded database similar to below:
  
  <img width="456" alt="Step 1 - Right Half" src="https://github.com/user-attachments/assets/bd272a14-8e83-439c-b8a2-227d2a97d9b3" />
  
- Insert your initial query to refine in the text box (3).
- Press the Continue button (4)
- On the right side of the screen you can see the results of your initial query on the database:

  <img width="456" alt="Step 1 - Query Results" src="https://github.com/user-attachments/assets/d8f12fea-eb44-4399-a721-c0154a8e63b3" />

#### Step 2 - Constraint satisfaction objective generation:
The left half of the OmniTune system on Step 2:

<img width="590" alt="Step 2" src="https://github.com/user-attachments/assets/dd0a32b1-a677-4fe3-8966-0be37d063e79" />

- Provide your constraint in natural language in (5). 
- Alternatively, to provide your own python implementation for constraint satisfaction press on "click here" (6).
  In that case a new tab would be automatically opened on the right side with corresponding instructions.
- Press the "Generate Constraints Satisfaction Objective" button (7)

#### Step 3 - Refinement distance method selection: 
<img width="490" alt="Step 3" src="https://github.com/user-attachments/assets/0ddb3180-deb5-4ecc-86ae-888d58a3ccaf" />

- Once the message in (8) appears, you shuld see the generated function on the right side of the screen, and you may also edit it using the Edit button.
- Select your preferred refienment distance method using the radio buttons (9).
- Press the "Generate Refinement Distance Method" button (10).

#### Step 4 - Constraint satisfaction threshold selection:
<img width="650" alt="Step 4" src="https://github.com/user-attachments/assets/8f42781f-d8c1-4246-ac9d-203e5fd8978b" />

- On the right side of the screen, you should see the generated function, which you may also edit using the Edit button.
- Move the slider (11) to select your preferred constraint satisfaction threshold.
- Press the "Start Refinement" button (12).

#### Step 5 - Refinement process:
<img width="540" alt="Step 5" src="https://github.com/user-attachments/assets/b130fc18-668f-4cc2-8702-5369752f6374" />

- During the refinement process, on the right half of the screen you should see the current refined query (13).
- Below you can see the current refinement's constraints score (14) and refienemnt distance (15).
- Below that is provided a constraint score explanation (16)

#### Step 6 - Final refinement view:
<img width="540" alt="Step 6" src="https://github.com/user-attachments/assets/51f2d790-1609-4e98-b03c-cae4233e5942" />

- When OmniTune is done refining the query you should see the message in (17), which notes the number attempts took to achieve.
- Query and scores viewed the same as in Step 5.
- Refined query results w.r.t. original query results can be viewed in the table (18).
- To access an in-depth view of the LLM multi agent process step-by-step, press on the "Log & Previous Queries" tab (19)
  
[For a step-by-step tutorial for OmniTune we invite you to watch our tutorial video (Click Here)](tinyurl.com/OmniTune).

## 5. Prompt & Prompt Templates: <a name="prompts"></a>
Below are provided the prompts and templates used by our Actor and Critic LLMs:
### System Prompts: <a name ="system_prompts"></a>
System Prompts are the prompts provided to each LLM prior to all other prompts, and define its overall role.
They are used in the concatenated to the beginning of each messages list sent to each of the LLMs accordingly for context alignment.
The system prompts used in our system are as below:
#### Actor:
```
You are a smart research assistant. 
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
```
#### Critic: 
```
You are a smart research critic. Use the attached tools to criticize the solution for
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
```

### User Prompt Templates: <a name ="user_prompts"></a>

The User Prompt Template is provided to each LLM on each refinement step, prompt-injected with the current task at hand and the current Refinement History.

#### Actor User Prompt Template:

Injected variables:
- The original query provided by user.
- The alterable predicates, which are the predicates within the query allowed to be refined.
- Their correpsonding attributes' information from the database (as data type, along ranges if they are numeric).
- The threshold "epsilon" for constraint satisfation.
- The refinement history which include a short summary of previous refinements and their resulting scores.
- The instructions from the Critic (for any refinement attempt > 1).

```
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{alterable_attributes_info}
Important: regarding numerical attributes, make sure the refinements you generate use of the same dtype as the original query,
namely, for integer attributes, the refinements should be integers, and for float attributes, the refinements should be floats.

Given those instructions, the query refinement task is defined as follows:

1. Original Query: 
{original_query}

2. Target Attributes for Refinement:
{alterable_predicates}

3. For each refinement you generate, you will receive a feedback on the validity, constraint satisfaction and refinement distance.
Your task is to refine the query so that constraint satisfaction is met up to threshold = {epsilon}, while keeping the refinement distance minimal.

4. Refinement History:
{refienment_history}

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

Based on your findings from the dataset, answer the provided questions in the instructions below before providing your final refinement.

Once you have reached a satisfactory refinement, provide your final output in the following format:
[Provide a brief explanation of which python queries you ran on the dataset and why,
 what new relevant information you found that was interesting, and how it influenced your refinement decisions]

\```sql
[Insert your final refined SQL query here]
\```

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.

Provide your refinement using the below instructions:
{critic_instructions}
```

#### Critic User Prompt Template:

Injected variables:
- The original query provided by user.
- Short description of the refinement method objective in natural language.
- The user given constraints in natural language (may be the function if provided only implementation).
- Their corresponding attributes' information from the database (as data type, along ranges if they are numeric).
- The threshold "epsilon" for constraint satisfation.
- The refinement history which include a short summary of previous refinements and their resulting scores.
- The current refinement generated by the Actor.

```
You are an SQL coding assistant.
Below are instructions to refine the query Q below into a Minimal-Refined query Q'.

For your convenience, the Dataset's relevant columns and their types are:
{constrained_attributes_dataset_info}
 
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

4. Refinement History:
{refienment_history}

====================================================================================================

Your task is to provide natural language feedback on the refinement, without suggesting any explicit SQL queries,
so that the next refinement would most likely satisfy the constraints while keeping the refinement distance minimal.

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

Please provide your feedback and instructions based on the below refinement:
{actor_refienment}
```

### Constraint Generation Prompts: <a name ="constraint_prompts"></a>

The prompt templates below receive the constraints description in natural language from the user, and generate components to analyze the refinement according to the constraints accordnigly.
Both are injected with the following variables:
- The natural language description of the constraints, provided by the user.
- The result of the initial query on the databse for structural reference.

#### Constraint Satisfaction Function:
Below is the prompt provided to the LLM to generate the constraints according to the user-provided description:
```
You are a python function implementation agent.
The function can use only the pandas library and the below explicitly provided tools. NO NUMPY!

You are responsible for generating the following nested function to assess the quality of a refined query:
You will be provided with a list of constraints that the output of the refined query should satisfy.
Given this list, you need to implement a function that calculates the deviation of the output of the refined query from satisfying a set of constraints.
The deviation should be a weighted sum of the constraints that are not satisfied, and if not provided
explicitly, the constraints should be assumed to be equally weighted. Then, for each constraint,
the deviation should represent the absolute difference between the required constraint and the actual value
in the output of the refined query. The constraints deviation score should be a non-negative float,
where a lower value indicates a better quality.

The Inner function output MUST be a normalized float value between 0 and 1,
where 0 indicates that all constraints are satisfied and 1 indicates that none of the constraints are satisfied.
The outer function shall return the inner function as a runnable function.
The nested function signatures are to be as follows:
    - Outer function signature: get_constraints_satisfaction_objective(d_in: dict[str, pd.DataFrame], original_query: str)
    - Inner function signature: constraints_satisfaction_objective(refined_query: str) -> float (constraints deviation score)
This way, a task specific inner function can be re-used multiple times with various refined queries along the refinement
process.

Within the function, you will have access to the dataset and the original query as a dictionary of string to pandas dataframe.
You are also provided with a class called SQLEngineAdvanced that can be used to execute SQL queries on the dataset 
in the following way:

\```python 
    sql_engine = SQLEngineAdvanced(d_in_dict)
    resp_df, error = sql_engine.execute(query)
    if error:
        return "An error occurred: " + resp_df
    return resp_df
\```

Using the SQLEngineAdvanced class, you can execute SQL queries on the dataset and get the result as a pandas dataframe,
to further use to assess the quality of the refined query's output.

Given the above instructions, generate a constraint satisfaction objective function, based on the following description:
{user_constraints_description}
You are advised to use the following structure of a result dataset:
{original_query_result_df_head}
Note that a constraint can be somewhat satisfied, meaning that its deviation score is not necessarily 0 or 1.
Most Importantly: The Inner function output MUST be a normalized float value, between 0 and 1,
where 0 indicates that all constraints are satisfied and 1 indicates that none of the constraints are satisfied. To achieve this:
 1. Each of the constraints deviation score should be normalized to a value between 0 and 1.
 2. The final constraint deviation score should be divided by the number of constraints.
Ensure that the normalized score is a positive float that does not in any case exceed 1!
```

#### Constraint-wise Analysis:
Below is the prompt provided to the LLM to generate the constraint wise JSON to present a constraint-wise analysis for each refinement:

```
You are an agent responsible for parsing constraints from a user query.
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

Use the 'get_dataset_information' tool provided to you to extract precise value names or ranges for each of the relevant specific columns in the result df of the original query.
Do not use any libraries other than pandas.

For structural reference, here is an example of the original query result:
{original_query_result_df_head}

Parse the following constraints as a list of dictionaries in the JSON format described above.
Ensure the response is properly structured JSON.
Implement the query code-lines ONLY using pandas functions and operations, any other libraries or functions will not be supported.
Constraints:
{user_constraints_description}
```
[^1]: https://github.com/streamlit/streamlit
[^2]: https://platform.openai.com/api-keys
