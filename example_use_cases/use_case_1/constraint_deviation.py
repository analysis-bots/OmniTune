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
    segment_col: str = "segment",
    avg_price_col: str = "avg_price",
) -> float:
    """
    Case Study 1 constraint deviation.

    Constraints:
      1) |segs(Q'(D))| >= 4
      2) AVG(ord_val) >= 500

    Expected result schema (at least):
      - segment
      - avg_price

    Assumption:
      - avg_price in each row is the per-group average order value returned by the query.
      - The final cross-segment value statistic is computed as the mean of returned rows' avg_price.
    """
    if not rows:
        return 2.0

    num_segments = len({row[segment_col] for row in rows})
    mean_avg_price = sum(_safe_float(row[avg_price_col]) for row in rows) / len(rows)

    dev_segments = _normalized_lower_bound_deviation(num_segments, 4.0)
    dev_value = _normalized_lower_bound_deviation(mean_avg_price, 500.0)

    return dev_segments + dev_value


if __name__ == "__main__":
    example_rows = [
        {"segment": "beauty", "customer": "c1", "cnt": 3, "avg_price": 520},
        {"segment": "sports", "customer": "c2", "cnt": 4, "avg_price": 540},
        {"segment": "books", "customer": "c3", "cnt": 3, "avg_price": 560},
        {"segment": "toys", "customer": "c4", "cnt": 5, "avg_price": 530},
    ]
    print(constraint_deviation(example_rows))
