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
    city_col: str = "city",
    price_col: str = "price",
    sqft_col: str = "sqft",
) -> float:
    """
    Case Study 2 constraint deviation.

    Constraints:
      1) |cities(Q'(D))| >= 6
      2) average(price / sqft) <= 520

    Expected result schema (at least):
      - city
      - price
      - sqft
    """
    if not rows:
        return 2.0

    num_cities = len({row[city_col] for row in rows})

    price_per_sqft_values = []
    for row in rows:
        sqft = _safe_float(row[sqft_col])
        if sqft <= 0:
            raise ValueError(f"sqft must be positive, got {sqft}")
        price = _safe_float(row[price_col])
        price_per_sqft_values.append(price / sqft)

    mean_price_per_sqft = sum(price_per_sqft_values) / len(price_per_sqft_values)

    dev_cities = _normalized_lower_bound_deviation(num_cities, 6.0)
    dev_cost = _normalized_upper_bound_deviation(mean_price_per_sqft, 520.0)

    return dev_cities + dev_cost


if __name__ == "__main__":
    example_rows = [
        {"city": "A", "price": 700000, "sqft": 1500, "type": "sfr"},
        {"city": "B", "price": 680000, "sqft": 1400, "type": "condo"},
        {"city": "C", "price": 720000, "sqft": 1600, "type": "townhouse"},
        {"city": "D", "price": 690000, "sqft": 1450, "type": "sfr"},
        {"city": "E", "price": 710000, "sqft": 1500, "type": "condo"},
        {"city": "F", "price": 730000, "sqft": 1550, "type": "townhouse"},
    ]
    print(constraint_deviation(example_rows))
