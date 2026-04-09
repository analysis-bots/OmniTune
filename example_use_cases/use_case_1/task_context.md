<task_context>
**Original Query:**
```sql
SELECT segment, customer,
  COUNT(*) AS cnt,
  AVG(price) AS avg_price
FROM Olist joins
GROUP BY segment, customer
HAVING COUNT(*) >= 6
  AND avg_price >= 300
  AND avg_price <= 800
```

**Refineable Predicates:**
```json
[
  {
    "id": "COUNT(*) >=",
    "attribute_name": "COUNT(*)",
    "operator": ">=",
    "current_value": 6,
    "value_type": "numerical",
    "valid_value_range": {"min": 2, "max": 6, "step": 1}
  },
  {
    "id": "avg_price >=",
    "attribute_name": "avg_price",
    "operator": ">=",
    "current_value": 300,
    "value_type": "numerical",
    "valid_value_range": {"min": 200, "max": 500, "step": 1}
  },
  {
    "id": "avg_price <=",
    "attribute_name": "avg_price",
    "operator": "<=",
    "current_value": 800,
    "value_type": "numerical",
    "valid_value_range": {"min": 700, "max": 900, "step": 1}
  }
]
```

**Output Constraints:**
```json
[
  {
    "id": "C1",
    "description_concise": " - The output should contain AT LEAST 4 distinct product segments.\n",
    "target_value": 4,
    "current_satisfaction_status": "not_satisfied"
  },
  {
    "id": "C2",
    "description_concise": " - The average order value across returned segments should be AT LEAST 500.\n",
    "target_value": 500,
    "current_satisfaction_status": "not_satisfied"
  }
]
```

**Dataset Schema:**
```json
[
  {"name": "segment", "type": "object"},
  {"name": "customer", "type": "object"},
  {"name": "cnt", "type": "int64"},
  {"name": "avg_price", "type": "float64"}
]
```

**Refinement Distance Objective:**
Minimize the refinement distance from the original query by making the smallest possible changes to the refineable HAVING predicates while satisfying the target constraints.

**Attribute Intelligence (Data-Informed Stats):**
```json
{
  "COUNT(*)": {"observed_subspace_range": [2, 6]},
  "avg_price_lower_bound": {"observed_subspace_range": [200, 500]},
  "avg_price_upper_bound": {"observed_subspace_range": [700, 900]},
  "initial_result_summary": {
    "num_distinct_segments": 2,
    "cross_segment_average_order_value": 385
  },
  "refined_result_summary": {
    "num_distinct_segments": 4,
    "cross_segment_average_order_value": 540
  }
}
```
</task_context>
