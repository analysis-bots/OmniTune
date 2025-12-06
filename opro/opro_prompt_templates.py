"""
Prompt Templates Library for the Actor-Critic Refinement Framework
Each PROMPT_TEMPLATE below should be used to format
the respective messages sent to the LLM.
"""

PRELIMINARY_DEFINITIONS = """
**=== PRELIMINARY DEFINITIONS ===**

**Predicate:**
A (Refineable) Predicate is the atomic condition in a SQL WHERE/HAVING clause.
A predicate is composed of an attribute, an operator, and a value.
There are two types of predicates:
- Numerical Predicate: Involves numerical attributes with operators like >=, <=, >, <, and values are numbers. 
  Examples for numerical predicates: age >= 30, GPA < 3.5.
- Categorical Predicate: Involves categorical attributes with operators like IN, = and values are categories or sets of categories.
  Note that in case of '=' operator, the value is a single category, and it is also interchangeable with IN operator with a set containing a single value.
  Examples for categorical predicates: region IN ('North', 'East'), status = 'inactive' (interchangeable with: status IN ('inactive')).
Unless stated otherwise, the Refineable Predicates include all and only numerical and categorical predicates from the original query's WHERE and/or HAVING clauses.
You are NOT allowed to add new predicates or remove existing predicates from the original query, as well as change the operators of existing predicates (except for the '=' operator, which is interchangeable with IN operator with a set containing a single value).
Moreover, the predicates defined above are the only part of the query that you are allowed to change.
All other parts of the query (SELECT, FROM, GROUP BY, ORDER BY clauses) must remain unchanged.

**Valid Value Range:**
The Valid Value Range for a Refineable Predicate is the set of all possible values that the predicate can take, based on the dataset schema and the data distribution.
We will define the Valid Value Range for each Refineable Predicate as follows:
- For Numerical Predicates: The Valid Value Range will be provided to you as a json: {{"min": number, "max": number, "step": number}},
  or a JSON with all possible values (if the number of possible values is small enough).
  Examples for numerical predicates: "age lower bound": {{"min": 18, "max": 120, "step": 1}}, "GPA": {{"min": 0, "max": 4, "step": 0.1}}.
- For Categorical Predicates: The Valid Value Range will be provided to you as a json array of strings, representing all possible categories for the attribute.
  Examples for categorical predicates: region: ["North", "South", "East", "West"], status: ["active", "inactive"].
The valid values that are allowed for each Refineable Predicate is defined by its Valid Value Range.
You are not allowed to suggest values outside the Valid Value Range for any Refineable Predicate.

**The Refineable Predicate Space:**
In the context of query refinement, the Refineable Predicate Space is the set of all valid values for each Refineable Predicate, that can be used to modify the original query.
The Refineable Predicate Space is defined as the Cartesian product of the Valid Value Ranges for each Refineable Predicate.
For example, if we have the following Refineable Predicates:
- age >= 30 with Valid Value Range: "age lower bound": {{"min": 18, "max": 99, "step": 1}}
- GPA < 3.5 with Valid Value Range: "GPA upper bound": {{"min": 2.0, "max": 4.0, "step": 0.1}}
- Region IN ('North', 'East') with Valid Value Range: ["North", "South", "East", "West"]
Then the Refineable Predicate Space is the set of all possible combinations of age, GPA, and Region values within their respective Valid Value Ranges, e.g.:
{{"age lower bound": 30, "GPA upper bound": 3.5, "Region set": ["North", "East"]}}
{{"age lower bound": 18, "GPA upper bound": 2.0, "Region set": ["South"]}}
{{"age lower bound": 99, "GPA upper bound": 4.0, "Region set": ["North", "South", "East", "West"]}}
etc...

**Predicate Subspace:**
A Predicate Subspace is a subset of the Refineable Predicate Space, defined by specific ranges or sets of values for one or more Refineable Predicates.
- For Numerical Predicates: A Predicate Subspace can be defined by a list of two numbers [min_value, max_value], representing a range of valid values for the predicate.
  Example: "age lower bound": [30, 40] means, in the predicate age >= X, X can take any value between 30 and 40 (inclusive), according to the step size defined in the Valid Value Range.
- For Categorical Predicates: A Predicate Subspace can be defined by a list of two lists of strings [contained_set, containing_set], representing two sets of categories.
  Example: "Region set": [["North"], ["North", "East", "West"]] means, in the predicate Region IN (X), X must contain "North" and can contain any combination of "North", "East", and "West".
  Note that the first list (contained_set) must be a subset of the second list (containing_set).
- All subspaces must include all refineable predicates.
- For numerical predicates the subspace corresponding range must consist AT LEAST 5 possible values, considering the step size. 

**Constraints Satisfaction:**
A constraint is a condition on the output of the query that must be satisfied.
******Constraint satisfaction is defined as constraint_metric == 0.*******
You will be provided with a list of constraints, and for each constraint you will be provided with informative details about its current satisfaction status.

**Refinement Distance Objective:**
The refinement distance is a measure of how much a refined query deviates from the original query.
The refinement distance can be one of the following:
- Predicate based distance: A numerical value representing the distance between the original query and the refined query, based on the differences in their predicates.
  The query based method is a sum of predicate distances, where each predicate distance is calculated as follows:
    - For Numerical Predicates: The distance is the absolute difference between the original and refined predicate values, divided by the original value (we assume it is non-zero).
    - For Categorical Predicates: The distance is 1 minus the Jaccard similarity between the original and refined predicate sets.
- Normalized Predicate based distance: Like the Predicate based distance, but each predicate distance is divided by the number of Refineable Predicates.
  Note that in this case, the effect of adding / removing categories from Categorical Predicates is much less significant than in the non-normalized version.
- Result based distance: A numerical value representing the distance between the original query and the refined query, based on the differences in their result sets over the original database.
  The result based distance is calculated as 1 minus the Jaccard similarity between the original and refined query result sets.
Below you'll be notified which specific refinement distance method is used in this task.

**=== END OF PRELIMINARY DEFINITIONS ===**

"""

### ASSIGNMENT LM PROMPT TEMPLATES ###
ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE = """
You are the AssignmentLM. You generate candidate predicate assignments within a provided subspace Θ_t.

**PRELIMINARY DEFINITIONS:**
""" + PRELIMINARY_DEFINITIONS + """

**Static task context (fixed for this run):**
- Original Query:
```sql
{original_query_context}
```
- Refineable Predicates:
```json
{refineable_predicates_context_json}
```
- Output Constraints:
```json
{output_constraints_context_json}
```
- Dataset Schema:
```json
{dataset_schema_context_json}
```
- Refinement Distance Objective:
{refinement_objective_description}
- Attribute Intelligence (Data-Informed Stats):
```json
{attribute_intelligence_context_json}
```

**What you will receive in refinement prompts:**
- The current subspace Θ_t (allowed ranges/sets for every predicate).
- Local history for this subspace (recent attempts and outcomes).
- A skyline summary (current (Δ, ψ_ε) trade-offs) to avoid regressions.

**Your job:** Produce assignments consistent with Θ_t that can satisfy all constraints and improve the skyline (reduce distance while keeping constraint deviation near 0). Keep SQL structure unchanged except predicate values and stay within valid ranges.

**Response format (always JSON in a fenced block):**
```json
{{
  "reasoning_concise": "brief rationale",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```
"""

ASSIGNMENT_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE = """

**Objective**: Produce a refined SQL query that satisfies every output constraint while minimizing the refinement distance from the original query.

**Primary Targets**
- Achieve constraint satisfaction (aggregate constraint deviation == 0).
- Once satisfied, continue proposing refinements that preserve satisfaction and further reduce the refinement distance.

**Inputs**:
  
    **Original Query:**
    ```sql
    {original_query_context}
    ```
    
    **Refineable Predicates:**
    ```json
    {refineable_predicates_context_json}
    ```
    
    **Output Constraints:**
    ```json
    {output_constraints_context_json}
    ```
    
    **Dataset Schema:**
    ```json
    {dataset_schema_context_json}
    ```
    
    **Refinement Distance Objective:**
    {refinement_objective_description}
    
    **Attribute Intelligence (Data-Informed Stats):**
    ```json
    {attribute_intelligence_context_json}
    ```
  
**Execution Guardrails**
- Adjust predicate values only; keep SQL structure and operators unchanged.
- Stay strictly within the valid ranges provided.
- Use subspace-local history to avoid repeats and guide improvement.

**Response Format**

```json
{{
  "reasoning_concise": "brief rationale",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```

"""

ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE = """
AssignmentLM: propose a refinement using the current subspace Θ_t. Keep SQL structure unchanged; values must stay within Θ_t.

- Skyline view (current (Δ, ψ_ε) trade-offs):
{query_skyline}
 
- Current subspace Θ_t (allowed ranges/sets):
{current_subspace_json}

- Local history within Θ_t (recent attempts):
{refinement_history_json}

Response format (JSON, fenced):
```json
{{
  "reasoning_concise": "why this assignment can satisfy constraints and improve the skyline",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```
"""

ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_HISTORY = """
AssignmentLM: propose a refinement using the current subspace Θ_t. History summaries are disabled in this mode.

- Skyline view (current (Δ, ψ_ε) trade-offs):
{query_skyline}

- Current subspace Θ_t (allowed ranges/sets):
{current_subspace_json}

Response format (JSON, fenced):
```json
{{
  "reasoning_concise": "why this assignment can satisfy constraints and improve the skyline",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```
"""

ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE = """
AssignmentLM: propose a refinement using the current subspace Θ_t. Skyline summaries are unavailable; rely on the provided local history.

- Current subspace Θ_t (allowed ranges/sets):
{current_subspace_json}

- Recent attempt history in Θ_t:
{refinement_history_json}

Response format (JSON, fenced):
```json
{{
  "reasoning_concise": "why this assignment can satisfy constraints and reduce distance",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```
"""

ASSIGNMENT_LM_SUGGEST_REFINEMENT_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY = """
AssignmentLM: propose a refinement using the current subspace Θ_t. Skyline summaries and history are disabled in this mode.

- Current subspace Θ_t (allowed ranges/sets):
{current_subspace_json}

Response format (JSON, fenced):
```json
{{
  "reasoning_concise": "why this assignment can satisfy constraints and reduce distance",
  "selected_refinement": "<complete_sql_query>",
  "selected_predicate_values": {{
    "<predicate_id_1>": <value>,
    "<predicate_id_2>": [<set_of_values>]
  }}
}}
```
"""


### SUBSPACE LM PROMPT TEMPLATES ###

# --- Critic: System Prompt (Enhanced with Formal Language) ---
SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE = """
You are the SubspaceLM. You propose predicate subspaces that are likely to contain assignments improving the (Δ, ψ_ε) skyline.

**PRELIMINARY DEFINITIONS:**
""" + PRELIMINARY_DEFINITIONS + """

**Static task context (fixed for this run):**
<task_context>
**Original Query:**
```sql
{original_query_context}
```

**Refineable Predicates:**
```json
{refineable_predicates_context_json}
```

**Output Constraints:**
```json
{output_constraints_context_json}
```

**Dataset Schema:**
```json
{dataset_schema_context_json}
```

**Refinement Distance Objective:**
{refinement_objective_description}

**Attribute Intelligence (Data-Informed Stats):**
```json
{attribute_intelligence_context_json}
```
</task_context>

**What you will receive in selection prompts:**
- Skyline summary S⁺(Q) describing current best trade-offs.
- Subspace history summaries (performance of prior subspaces).

**Your job:** Propose a predicate subspace that includes every refineable predicate,
respects valid ranges, and is likely to yield assignments that satisfy constraints and improve the skyline.


**Output format (JSON in a fenced block):**
```json
{{
  "patterns_observed": "<brief signal from prior subspaces>",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  }},
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```

**Validation:** Ensure each numerical range contains ≥5 feasible values (respecting step) and categorical sets obey contained ⊆ values ⊆ containing. Avoid already-exhausted regions and prefer subspaces that can reduce constraint deviation before distance.
"""

SUBSPACE_LM_SYSTEM_PROMPT_TEMPLATE_NO_SKYLINE = """
You are the SubspaceLM. You propose predicate subspaces that are likely to contain assignments improving the (Δ, ψ_ε) skyline when skyline tracking is disabled.

**PRELIMINARY DEFINITIONS:**
""" + PRELIMINARY_DEFINITIONS + """

**Static task context:** same as the primary SubspaceLM prompt (original query, refineable predicates, constraints, schema, distance objective, attribute stats).

**What you will receive in selection prompts:** subspace history summaries (performance of prior subspaces). No skyline table is available in this mode.

**Your job:** Propose a predicate subspace that includes every refineable predicate, respects valid ranges, and is likely to yield assignments that satisfy constraints and reduce refinement distance.

**Output format (JSON in a fenced block):**
```json
{{
  "patterns_observed": "<concise signal from recent subspaces>",
  "reasoning_concise": "short strategy statement",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  }},
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```

**Validation:** Ensure each numerical range contains ≥5 feasible values (respecting step) and categorical sets obey contained ⊆ values ⊆ containing. Cover every refineable predicate and prioritize regions likely to satisfy constraints before distance minimization.
"""

SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE = """
SubspaceLM: propose a predicate subspace likely to contain assignments that improve the (Δ, ψ_ε) skyline.

- Skyline view S⁺(Q):
 {skyline}

- Historical subspace performance (signals from prior subspaces):
 {subspace_history_json}

Guidance:
- Respect valid ranges for every predicate.
- Ensure each numerical range contains at least five feasible values.
- Prefer regions not yet explored that could satisfy constraints and reduce distance.

Response format (JSON, fenced):
```json
{{
  "patterns_observed": "<brief summary of signals from subspace history>",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]     # contained set should be as small as possible!
  }},
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```
"""

SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_HISTORY = """
SubspaceLM: propose a predicate subspace likely to contain assignments that improve the (Δ, ψ_ε) skyline. History summaries are disabled.

- Skyline view S⁺(Q):
{skyline}

Response format (JSON, fenced):
```json
{{
  "patterns_observed": "Reason using skyline signals and current objectives.",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  }},
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```
"""

SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE = """
SubspaceLM: propose a predicate subspace likely to contain assignments that satisfy constraints and reduce distance. Skyline summaries are unavailable.

- Historical subspace performance (signals from prior subspaces):
 {subspace_history_json}

Guidance:
- Respect valid ranges and keep numerical ranges wide enough for at least five feasible values.
- Focus on regions that can flip constraints to satisfied; after that, allow distance reduction.

Response format (JSON, fenced):
```json
{{
  "patterns_observed": "<summary of recent subspace signals>",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  }},
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```
"""

SUBSPACE_LM_SUGGEST_SUBSPACE_PROMPT_TEMPLATE_NO_SKYLINE_NO_HISTORY = """
SubspaceLM: propose a predicate subspace likely to contain assignments that satisfy constraints and reduce distance. Skyline and history summaries are unavailable.

Response format (JSON, fenced):
```json
{{
  "patterns_observed": "Reason about constraint gaps and feasible predicate moves without summaries.",
  "selected_predicate_subspace": {{
    "<predicate_id_1>": [<min_value>, <max_value>],
    "<predicate_id_2>": [<contained_set>, <containing_set>]
  }},
  "reasoning_concise": "short strategy statement",
  "breakthrough_potential": <0.0-1.0>,
  "expected_efficiency": "<success_rate>% trials → constraint satisfaction",
  "novelty_score": <0.0-1.0>
}}
```
"""


PARSING_MODEL_SYSTEM_MESSAGE = """You are an agent responsible for parsing constraints from a user query.
You receive a string containing the constraints and should return a list of constraints, as follows:
- Each constraint should be a dictionary with the following keys:
    * query (str): a callable query that accepts the dataframe as input and returns a numeric value. 
                    the query should refer to Ethe dataframe as 'df' and should return a numeric value 
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
