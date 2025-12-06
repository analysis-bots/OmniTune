SYSTEM_PROMPT_TEMPLATE = """
Understand the following database structure, SQL query and constraints.

SQL query:
{original_query}

Database columns:
{dataset_description}

Constraints:
{output_constraints}

Refinement Distance:
{refinement_distance_metric}

The only predicates allowed in the refined query are:
{allowed_predicates}
in which you are only allowed to change the values within the <> brackets.
"""

TASK_DESCRIPTION_PROMPT_TEMPLATE = """
The task goal is to refine the original SQL query such that the constraints:
{output_constraints}

are satisfied, while the refined query has a minimal refinement distance. 
Please rephrase the task in the most coherent and understandable way possible.
"""

ANALYSIS_REQUEST_PROMPT = """
To refine the query effectively, generate specific analysis requests to understand how changes to the refineable predicates affect the constraints: {output_constraints}.

### Instructions
Generate requests that:
1. Reveal the distribution of data around the current predicate values.
2. Quantify how changes in predicate values impact the number of rows satisfying each constraint.
3. Identify trade-offs between satisfying different constraints.
4. Provide insights to minimize refinement distance.

### Examples
- "Analyze the count of rows with '<attribute>' = '<value>' where '<another_attribute>' ranges from <x> to <y>."
- "Determine the total row count if '<attribute>' >= <x> instead of <y>."
- "Check the distribution of '<attribute>' and total rows when adding '<some_value>' to the <another_attribute> list."

List 3-5 specific analysis requests."""

ANALYSIS_REQUEST_SUBSPACES_PROMPT = """
Given the following suggested subspaces:
{subspaces}

Identify which analysis results can assist in satisfying the constraints.
Focus on possible correlations between combinations of notable values of the refineable attributes and the constraints.
"""

ANALYSIS_QUERY_PROMPT = """
For each of the analysis requests you provided, generate a corresponding JSON object representing the operation.
Use the pandas library conventions for attribute names and operations where applicable.

An operation is recursively defined using the following structure:

### Operation Types and Parameters:

*   **Atomic Condition**: Filter based on a single attribute.
    ```json
    {
      "operation_type": "atomic_condition",
      "parameters": {
        "attribute": "<column_name>",
        "operator": "<one of: '<', '<=', '=', '>', '>=', '!=', 'IN'>",
        "value": "<numeric_or_categorical_value_or_list>"
      }
    }
    ```
*   **Recursive Condition**: Combine conditions with AND/OR.
    ```json
    {
      "operation_type": "recursive_condition",
      "parameters": {
        "connective": "<'AND' or 'OR'>",
        "conditions": [ <list_of_atomic_or_recursive_condition_objects> ]
      }
    }
    ```
*   **Value Counts**: Get counts of unique values for a categorical attribute, optionally filtered.
    ```json
    {
      "operation_type": "value_counts",
      "parameters": {
        "attribute": "<categorical_column_name>",
        "condition": <optional_atomic_or_recursive_condition_object_or_null>
      }
    }
    ```
*   **Bins**: Bin a numeric attribute into intervals, optionally filtered.
    ```json
    {
      "operation_type": "bins",
      "parameters": {
        "attribute": "<numeric_column_name>",
        "condition": <optional_atomic_or_recursive_condition_object_or_null>,
        "step": <numeric_bin_step_size>
      }
    }
    ```
*   **Aggregation**: Calculate max, min, mean, or std for an attribute, optionally filtered (applied *before* aggregation).
    ```json
    {
      "operation_type": "aggregation",
      "parameters": {
        "attribute": "<column_name>",
        "filter": "<'max', 'min', 'mean', or 'std'>",
        "condition": <optional_atomic_or_recursive_condition_object_or_null> 
      }
    }
    ```
*   **Group By**: Group data by an attribute and apply an aggregation, optionally filtered (applied *before* grouping).
    ```json
    {
      "operation_type": "group_by",
      "parameters": {
        "attribute": "<grouping_column_name>",
        "aggregation": {
           "attribute": "<aggregation_column_name>",
           "filter": "<'max', 'min', 'mean', or 'std'>"
        },
        "condition": <optional_atomic_or_recursive_condition_object_or_null>
      }
    }
    ```
*   **Calculation**: Perform simple arithmetic.
    ```json
    {
      "operation_type": "calc",
      "parameters": {
        "value1": <numeric_value>,
        "operator": "<'+', '-', '*', '/'>",
        "value2": <numeric_value>
      }
    }
    ```

### Output Format

Provide a JSON list where each element corresponds to one analysis request. Each element should contain the natural language request and the operation object. Limit the list to 3-5 requests.

```json
[
  {
    "analysis_request": "<Analysis Request 1 in natural language>",
    "operation": <operation_object_1>
  },
  {
    "analysis_request": "<Analysis Request 2 in natural language>",
    "operation": <operation_object_2>
  },
  ...
]
```

Ensure the parameters within the `operation` objects use actual values relevant to the dataset context, not placeholders. If a condition is not needed, set its value to `null`.
"""

ANALYSIS_REPORT_PROMPT_TEMPLATE = """
Based on the below analysis results, generate a concise analysis report with regards to satisfying the constraints:
{analysis_results}
"""

ANALYSIS_SOLUTION_PREFIX_TEMPLATE = """Based on the below analysis report:
 
{analysis_report}

"""

ANALYSIS_SOLUTION_PROMPT_TEMPLATE = """ 
Answer: What values for refinement would you like to suggest? 

---

**Important Note:**

When generating a refined query, follow these steps:

1. **Compare Metrics:**  
   - Compute the constraint violation metric for both the original (or previous refined) query (**M₀**) and the suggested refined query (**M₁**).
   - Define the constraint violation metric such that lower values indicate a query closer to satisfying the constraint.
   - **Important:** If M_refined (M₁) is lower than M_original (M₀), the refinement is an improvement. If M₁ is higher than M₀, the refinement is making the constraint violation worse.

2. **Assess Direction:**
    - If M₁ < M₀, the refinement is beneficial.
    - If M₁ > M₀, then the direction of your change is counterproductive. In this case, it means you have relaxed the query too much. Instead of widening the bounds, you need to **contract** them.

3. **Reverse Adjustment:**  
   - Identify which parameter was changed (for example, if you increased the range by lowering the lower bound or raising the upper bound).
   - Reverse that change. For instance, if widening the range (by lowering the lower bound or raising the upper bound) increased the metric, then try contracting the range (raising the lower bound or lowering the upper bound).

4. **Explain Briefly:**  
   - Include a short explanation such as:  
     > "The refined query increased the violation from M₀ to M₁; therefore, the adjustment on [parameter] is reversed to contract the range."

5. **Propose New Query:**  
   - Only output a refined query that is expected to reduce the violation metric, moving it closer to the target.

6. **Do not repeat the same query or one that most likely worsens the violation metric:**
    - Below are the previous queries and their violation metrics:
    {previous_queries_history}
---

#output format: 
<reasoning_process>
[A step-by-step explanation of how you arrived at the refined values]
</reasoning_process>

<final_result>
 {solution_structure}
</final_result>
"""

ANALYSIS_SOLUTION_SUFFIX_TEMPLATE = """
Your suggested refined values for each predicate must be within the following subspaces:

{subspaces}
"""

ANALYSIS_SUBSPACE_PROMPT_TEMPLATE = """ 
Answer: what subspaces would you like to explore further?

#output format: 
<reasoning_process>
[A reasoning step-by-step looking for the best subspace range to search further for each numerical column, 
and the superset of subsets to explore for each categorical column]
</reasoning_process>

<final_result>
{solution_structure}
</final_result>
"""

LOWER_BOUND_TEMPLATE = "{column_name}: <{column_name} lower bound> # [A short comment explaining why you chose this value, based on the analysis results]"

LOWER_BOUND_RANGE_TEMPLATE = "{column_name} lower bound subspace: [<{column_name} lower bound min value>, <{column_name} lower bound max value>]"

UPPER_BOUND_TEMPLATE = "{column_name}: <{column_name} upper bound> # [A short comment explaining why you chose this value, based on the analysis results]"

UPPER_BOUND_RANGE_TEMPLATE = "{column_name} upper_bound subspace: [<{column_name} upper bound min value>, <{column_name} upper bound max value>]"

CATEGORY_SUBSET_TEMPLATE = "{column_name}: (<{column_name} values subset>) # [A short comment explaining why you chose these values, based on the analysis results]"

CATEGORY_SUPERSET_TEMPLATE = "{column_name}: ((<{column_name} values subset 1>), (<{column_name} values subset 2>), ...)"

STRUCTURED_SOLUTION_PROMPT = """
Based on the refine-able columns, extract each column to its correct category:
- if the column predicate contains a lower bound (>= or >), put it in the lower_bound_columns category.
- if the column predicate contains an upper bound (<= or <), put it in the upper_bound_columns category.
- if the column predicate is a categorical predicate, put it in the category_subset_columns category.
"""

REFINEMENT_RESPONSE_PROMPT = """
Based on these values, provide the corresponding refined SQL query.
"""

CONSTRAINTS_NOT_SATISFIED_TEMPLATE = """
Below is a short report about the current constraints satisfaction status:
{constraints_satisfaction_status}

Which further adjustments, either contractions and/or relaxations, are needed to satisfy the constraints?
"""

