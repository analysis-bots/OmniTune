# OmniTune: A universal framework for query refinement via LLMs

![OmniTune](https://github.com/user-attachments/assets/3b4856d1-9c4b-4b79-bee9-b518a3deca58)

This repository contains the reproducibility package for running our demo.
For any issues regarding reproducibility or anything else using the implementation, please feel free to create a GitHub issue or send an Email.

### Table of Contents:
1. Requirements
2. Configurations
3. Usage
4. Prompts & Prompt Templates

## 1. Requirements:
The system runs on Python >= 3.11.6
To install the required package, cd to main dir, then run:

> pip install -r requirements.txt

The OmniTune UI is built upon the Streamlit[^1] package.
For an optimal UI view, the preferred resultion is 1920 x 1080.

## 2. Configuration:
OmniTune Demo currently supports only OpenAI models.
Before running OmniTune locally, you should get an OpenAI API-Key[^2].
Copy the API-Key paste it inplace of `<YOUR_OPENAI_API_KEY>` in the following line within the file **config.py**:

```
OPENAI_API_KEY = '<YOUR_OPENAI_API_KEY>'
```

Note about pricing: OmniTune defaultly uses GPT-4o-mini model to reduce runtime and costs to minimum.
If you would like to test it with any other model by OpenAI, you may do it using the **config.py** parameter `MODEl`.
More about OpenAI API models pricing is provided [here](https://openai.com/api/pricing/).
 

## 3. Usage:

### Running OmniTune:
1. Make sure Streamlit is installed successfully:
  > streamlit --version
2. On the main dir, run:
  > streamlit run app.py
3. A new browser tab should open with the system, else go to http://localhost:8501 manually.

### Using OmniTune:
#### Step 1 - Select database and insert query:
The left half of the OmniTune system on Step 1:

<img width="650" alt="Step 1 - Left Half" src="https://github.com/user-attachments/assets/8ddfb79a-787b-41d3-b323-36390a642b14" />

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

<img width="650" alt="Step 2" src="https://github.com/user-attachments/assets/dd0a32b1-a677-4fe3-8966-0be37d063e79" />

- Provide your constraint in natural language in (5). 
- Alternatively, to provide your own python implementation for constraint satisfaction press on "click here" (6).
  In that case a new tab would be automatically opened on the right side with corresponding instructions.
- Press the "Generate Constraints Satisfaction Objective" button (7)


#### Step 3 - Refinement distance method selection: 
<img width="650" alt="Step 3" src="https://github.com/user-attachments/assets/6c86fd25-708c-49a6-83cc-20d0e79fa143" />

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

## 4. Prompt & Prompt Templates: 
Below are provided the prompts and templates used by our Actor and Critic LLMs:
### System Prompts:
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

### User Prompt Templates:
#### Actor:
```
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

\```sql
[Insert your final refined SQL query here]
\```

Important: You should usually refine more than one predicate to reach a valid refinement, so try cross-examining
different combinations of values for the allowed attributes to find the best refinement.

YOU ARE NOT ALLOWED TO ADD OR REMOVE ANY CONDITIONS FROM THE ORIGINAL QUERY. ONLY MODIFY THE EXISTING CONDITIONS.

Note that this is a multi-objective optimization problem, where you need to BALANCE between the two objectives.
```

#### Critic:
```
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
```

[^1]: https://github.com/streamlit/streamlit
[^2]: https://platform.openai.com/api-keys
