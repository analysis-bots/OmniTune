<task_context>
**Original Query:**
```sql
WITH ranked AS (
  SELECT *,
    PERCENT_RANK() OVER (...) AS pr
  FROM TexasTribune
  WHERE HRSWKD > 40
    AND ANNUAL > 50000
)
SELECT ETHNICITY,
  AVG(ANNUAL) AS avg_annual,
  SUM(MONTHLY) AS total_monthly
FROM ranked
WHERE pr >= 0.40 AND pr <= 0.90
GROUP BY ETHNICITY
```

**Refineable Predicates:**
```json
[
  {
    "id": "HRSWKD >",
    "attribute_name": "HRSWKD",
    "operator": ">",
    "current_value": 40,
    "value_type": "numerical",
    "valid_value_range": {"min": 25, "max": 40, "step": 1}
  },
  {
    "id": "ANNUAL >",
    "attribute_name": "ANNUAL",
    "operator": ">",
    "current_value": 50000,
    "value_type": "numerical",
    "valid_value_range": {"min": 45000, "max": 55000, "step": 1}
  },
  {
    "id": "pr >=",
    "attribute_name": "pr",
    "operator": ">=",
    "current_value": 0.40,
    "value_type": "numerical",
    "valid_value_range": {"min": 0.30, "max": 0.85, "step": 0.01}
  },
  {
    "id": "pr <=",
    "attribute_name": "pr",
    "operator": "<=",
    "current_value": 0.90,
    "value_type": "numerical",
    "valid_value_range": {"min": 0.45, "max": 0.95, "step": 0.01}
  }
]
```

**Output Constraints:**
```json
[
  {
    "id": "C1",
    "description_concise": " - The minimum total monthly support across returned ethnicity groups should be AT LEAST 3500000.\n",
    "target_value": 3500000,
    "current_satisfaction_status": "not_satisfied"
  },
  {
    "id": "C2",
    "description_concise": " - The ratio max(avg_annual) / min(avg_annual) across returned groups should be AT MOST 1.15.\n",
    "target_value": 1.15,
    "current_satisfaction_status": "not_satisfied"
  }
]
```

**Dataset Schema:**
```json
[
  {"name": "ETHNICITY", "type": "object"},
  {"name": "avg_annual", "type": "float64"},
  {"name": "total_monthly", "type": "float64"}
]
```

**Refinement Distance Objective:**
Minimize the refinement distance from the original query by making the smallest possible changes to the refineable cohort-selection predicates and percentile-window bounds while satisfying the target constraints.

**Attribute Intelligence (Data-Informed Stats):**
```json
{
  "HRSWKD": {"observed_subspace_range": [25, 40]},
  "ANNUAL": {"observed_subspace_range": [45000, 55000]},
  "pr_lower_bound": {"observed_subspace_range": [0.30, 0.85]},
  "pr_upper_bound": {"observed_subspace_range": [0.45, 0.95]},
  "initial_result_summary": {
    "min_total_monthly": 2000000,
    "max_avg_annual_over_min_avg_annual": 1.65
  },
  "refined_result_summary": {
    "min_total_monthly": 3550000,
    "max_avg_annual_over_min_avg_annual": 1.11
  }
}
```
</task_context>
