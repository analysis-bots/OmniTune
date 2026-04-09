<task_context>
**Original Query:**
```sql
SELECT city, price,
  sqft, type
FROM HomesForSale
WHERE sqft >= 1500
  AND type IN ('sfr','condo')
  AND city IN (
    SELECT city
    FROM HomesForSale
    GROUP BY city
    HAVING COUNT(*) >= 20
      AND AVG(price) <= 750000
  )
```

**Refineable Predicates:**
```json
[
  {
    "id": "sqft >=",
    "attribute_name": "sqft",
    "operator": ">=",
    "current_value": 1500,
    "value_type": "numerical",
    "valid_value_range": {"min": 1400, "max": 1600, "step": 1}
  },
  {
    "id": "type",
    "attribute_name": "type",
    "operator": "IN",
    "current_value": ["sfr", "condo"],
    "value_type": "categorical",
    "valid_value_range": {
      "categories": [
        ["sfr", "condo"],
        ["sfr", "condo", "townhouse"]
      ]
    }
  },
  {
    "id": "COUNT(*) >=",
    "attribute_name": "COUNT(*)",
    "operator": ">=",
    "current_value": 20,
    "value_type": "numerical",
    "valid_value_range": {"min": 15, "max": 25, "step": 1}
  },
  {
    "id": "AVG(price) <=",
    "attribute_name": "AVG(price)",
    "operator": "<=",
    "current_value": 750000,
    "value_type": "numerical",
    "valid_value_range": {"min": 750000, "max": 850000, "step": 1000}
  }
]
```

**Output Constraints:**
```json
[
  {
    "id": "C1",
    "description_concise": " - The output should contain AT LEAST 6 distinct cities.\n",
    "target_value": 6,
    "current_satisfaction_status": "not_satisfied"
  },
  {
    "id": "C2",
    "description_concise": " - The average price-per-square-foot across returned properties should be AT MOST 520.\n",
    "target_value": 520,
    "current_satisfaction_status": "not_satisfied"
  }
]
```

**Dataset Schema:**
```json
[
  {"name": "city", "type": "object"},
  {"name": "price", "type": "float64"},
  {"name": "sqft", "type": "float64"},
  {"name": "type", "type": "object"}
]
```

**Refinement Distance Objective:**
Minimize the refinement distance from the original query by making the smallest possible changes to the refineable predicates, including both numerical and categorical predicates, while satisfying the target constraints.

**Attribute Intelligence (Data-Informed Stats):**
```json
{
  "sqft": {"observed_subspace_range": [1400, 1600]},
  "COUNT(*)": {"observed_subspace_range": [15, 25]},
  "AVG(price)": {"observed_subspace_range": [750000, 850000]},
  "type": {
    "candidate_category_sets": [
      ["sfr", "condo"],
      ["sfr", "condo", "townhouse"]
    ]
  },
  "initial_result_summary": {
    "num_distinct_cities": 3,
    "average_price_per_sqft": 548
  },
  "refined_result_summary": {
    "num_distinct_cities": 6,
    "average_price_per_sqft": 509
  }
}
```
</task_context>
