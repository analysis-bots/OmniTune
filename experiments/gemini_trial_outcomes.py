"""Distinguish a finalized experimental outcome from complete API output."""


def is_finalized(row):
    return bool(row.get("trial_finalized", row.get("trial_complete", True)))


def finalize_output_failure(row):
    """Count an attempted output failure as unsuccessful, preserving evidence."""
    row = dict(row)
    row.setdefault("partial_candidate_score", {
        key: row.get(key) for key in
        ("success", "optimality", "constraints_satisfied", "distance", "sql")
    })
    row.update(trial_complete=False, trial_finalized=True,
               trial_outcome="failed_output_limit", failure_type="output_limit",
               success=False, optimality=0.0, constraints_satisfied=False)
    return row
