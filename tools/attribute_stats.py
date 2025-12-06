from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from functionality.predicate import Predicate, NumericalPredicate, CategoricalPredicate


class AttributeStatsGenerator:
    def __init__(self, dataset: pd.DataFrame, predicates: List[Predicate], sample_size: int = 3000):
        self.dataset = dataset
        self.predicates = predicates
        self.sample_size = min(int(sample_size or 3000), len(dataset)) if len(dataset) else 0
        self._sample = self._get_sample()

    def _get_sample(self) -> pd.DataFrame:
        if self.sample_size == 0:
            return self.dataset.head(0)
        try:
            return self.dataset.sample(n=self.sample_size, random_state=42, replace=False)
        except Exception:
            return self.dataset.head(self.sample_size)

    def generate_stats(self) -> Dict[str, Any]:
        intelligence: Dict[str, Any] = {}
        for i, p in enumerate(self.predicates):
            pid = p.get_id()
            col = p.attribute.name
            if col not in self._sample.columns:
                continue
            s = self._sample[col].dropna()
            if isinstance(p, NumericalPredicate) and len(s) > 0:
                intelligence[pid] = {
                    "attribute_name": col,
                    "type": "numerical",
                    "current_value": p.value,
                    "stats": self._numerical_stats(s, float(p.value))
                }
            elif isinstance(p, CategoricalPredicate) and len(s) > 0:
                intelligence[pid] = {
                    "attribute_name": col,
                    "type": "categorical",
                    "current_value": list(p.values),
                    "stats": self._categorical_stats(s, set(map(str, p.values)))
                }
        return intelligence

    def _numerical_stats(self, s: pd.Series, current: float) -> Dict[str, Any]:
        desc = s.describe(percentiles=[0.25, 0.5, 0.75])
        q = {"25%": _round(desc.get(0.25)), "50%": _round(desc.get(0.5)), "75%": _round(desc.get(0.75))}
        try:
            counts, bin_edges = np.histogram(s.astype(float), bins=6)
        except Exception:
            counts, bin_edges = np.array([len(s)]), np.array([float(s.min()), float(s.max())])
        density_flag = "unknown"
        if len(bin_edges) > 1:
            idx = np.searchsorted(bin_edges, current, side="right") - 1
            idx = int(np.clip(idx, 0, len(counts) - 1))
            median_cnt = float(np.median(counts)) if len(counts) else 0.0
            density_flag = "high" if (len(counts) and counts[idx] >= median_cnt) else "low"
        # impact zones: top and bottom bins by count
        top_idx = int(np.argmax(counts)) if len(counts) else 0
        low_idx = int(np.argmin(counts)) if len(counts) else 0
        def _range(i: int) -> List[float]:
            if len(bin_edges) < 2:
                return [ _round(desc.get("min")), _round(desc.get("max")) ]
            return [ _round(bin_edges[i]), _round(bin_edges[i+1]) ]
        return {
            "mean": _round(desc.get("mean")),
            "std": _round(desc.get("std")),
            "min": _round(desc.get("min")),
            "max": _round(desc.get("max")),
            "quartiles": q,
            "histogram_bins": [ _round(x) for x in bin_edges.tolist() ],
            "histogram_counts": counts.astype(int).tolist(),
            "density_around_current": density_flag,
            "impact_zones": {"high_impact": _range(top_idx), "low_impact": _range(low_idx)}
        }

    def _categorical_stats(self, s: pd.Series, current_vals: set) -> Dict[str, Any]:
        sv = s.astype(str)
        vc = sv.value_counts(dropna=False)
        total = float(len(sv)) or 1.0
        top = vc.head(10)
        freqs = {k: float(v) / total for k, v in top.to_dict().items()}
        coverage = 0.0
        if current_vals:
            coverage = float(sv.isin(list(current_vals)).mean())
        rare = [k for k, v in vc.items() if (v / total) < 0.01]
        common = [k for k, v in vc.items() if (v / total) >= 0.05]
        return {
            "top_frequencies": freqs,
            "cardinality": int(sv.nunique()),
            "coverage_of_current": _round(coverage),
            "rare_categories": rare[:10],
            "common_categories": common[:10]
        }


def _round(x: Any) -> Any:
    try:
        xv = float(x)
        if np.isnan(xv):
            return None
        return float(np.format_float_positional(xv, precision=3, unique=False, trim='k'))
    except Exception:
        return x


