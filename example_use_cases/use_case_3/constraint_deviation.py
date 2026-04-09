from __future__ import annotations

from typing import Any, Mapping, Sequence

Row = Mapping[str, Any]


def _safe_float(x: Any) -> float:
    return float(x)


def _normalized_lower_bound_deviation(observed: float, required_min: float) -> float:
    """
    Deviation for constraints of the form observed >= required_min.
    Returns 0 if satisfied, else relative shortfall.
    """
    if required_min <= 0:
        return 0.0 if observed >= required_min else float("inf")
    return max(0.0, (required_min - observed) / required_min)


def _normalized_upper_bound_deviation(observed: float, required_max: float) -> float:
    """
    Deviation for constraints of the form observed <= required_max.
    Returns 0 if satisfied, else relative excess.
    """
    if required_max == 0:
        return 0.0 if observed <= 0 else float("inf")
    return max(0.0, (observed - required_max) / required_max)


def constraint_deviation(
    rows: Sequence[Row],
    avg_annual_col: str = "avg_annual",
    total_monthly_col: str = "total_monthly",
) -> float:
    """
    Case Study 3 constraint deviation.

    Constraints:
      1) min(total_monthly) >= 3,500,000
      2) max(avg_annual) / min(avg_annual) <= 1.15

    Expected result schema (at least):
      - avg_annual
      - total_monthly

    Assumption:
      - Each row corresponds to one ethnicity group after GROUP BY ETHNICITY.
    """
    if not rows:
        return 2.0

    total_monthlies = [_safe_float(row[total_monthly_col]) for row in rows]
    avg_annuals = [_safe_float(row[avg_annual_col]) for row in rows]

    min_total_monthly = min(total_monthlies)
    min_avg_annual = min(avg_annuals)
    max_avg_annual = max(avg_annuals)

    if min_avg_annual <= 0:
        raise ValueError(f"avg_annual must be positive to compute ratio, got {min_avg_annual}")

    salary_ratio = max_avg_annual / min_avg_annual

    dev_support = _normalized_lower_bound_deviation(min_total_monthly, 3_500_000.0)
    dev_ratio = _normalized_upper_bound_deviation(salary_ratio, 1.15)

    return dev_support + dev_ratio


if __name__ == "__main__":
    example_rows = [
        {"ETHNICITY": "g1", "avg_annual": 52000, "total_monthly": 3600000},
        {"ETHNICITY": "g2", "avg_annual": 54000, "total_monthly": 3550000},
        {"ETHNICITY": "g3", "avg_annual": 56000, "total_monthly": 3700000},
    ]
    print(constraint_deviation(example_rows))
