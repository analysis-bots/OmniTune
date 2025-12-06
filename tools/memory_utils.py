import pandas as pd
from typing import List, Dict, Any

# memory_utils.py

def init_memory_df(
    predicate_ids: List[str],
    constraint_ids: List[str],
    orig_query: str,
    orig_predicate_values: Dict[str, Any],
    orig_constraint_values: Dict[str, float],
    orig_distance: float,
    orig_notes: str = ""
) -> pd.DataFrame:
    """
    Initialize the memory DataFrame for subspace refinements.

    Parameters:
    - predicate_ids: list of predicate column names (e.g. ['age_min', 'bmi_min'])
    - constraint_ids: list of constraint metric names (e.g. ['max_avg', 'std', 'regions'])
    - orig_query: the original SQL query string
    - orig_predicate_values: dict mapping predicate_ids to their initial values
    - orig_constraint_values: dict mapping constraint_ids to their evaluated values
    - orig_distance: the refinement distance for the original query (typically 0.0)
    - orig_notes: optional initial notes

    Returns:
    - A pandas DataFrame with one row for the original query
    """
    # Build the column list
    columns = ['query'] + predicate_ids + constraint_ids + ['distance', 'notes']
    # Create the initial row dict
    row = {
        'query': orig_query,
        **{pid: orig_predicate_values.get(pid) for pid in predicate_ids},
        **{cid: orig_constraint_values.get(cid) for cid in constraint_ids},
        'distance': orig_distance,
        'notes': orig_notes
    }
    # Construct DataFrame
    df = pd.DataFrame([row], columns=columns)
    return df

def update_memory_df(
    memory_df: pd.DataFrame,
    new_query: str,
    new_predicate_values: Dict[str, Any],
    new_constraint_values: Dict[str, float],
    new_distance: float,
    new_notes: str
) -> pd.DataFrame:
    """
    Append a new refinement attempt to the memory DataFrame.

    Parameters:
    - memory_df: existing DataFrame returned by init_memory_df or prior updates
    - new_query: SQL string of the new refinement
    - new_predicate_values: dict mapping predicate_ids to their values in this run
    - new_constraint_values: dict mapping constraint_ids to their evaluated values
    - new_distance: the computed refinement distance for this new query
    - new_notes: one-sentence LLM-generated summary of the run

    Returns:
    - The updated pandas DataFrame with the new row appended
    """
    # Ensure all required columns exist
    columns = memory_df.columns.tolist()
    # Build the new row dict
    row = {
        'query': new_query,
        **{pid: new_predicate_values.get(pid) for pid in columns if pid in new_predicate_values},
        **{cid: new_constraint_values.get(cid) for cid in columns if cid in new_constraint_values},
        'distance': new_distance,
        'notes': new_notes
    }
    # Append and return
    # return memory_df.append(row, ignore_index=True)
    memory_df = pd.concat([memory_df, pd.DataFrame([row], columns=columns)], ignore_index=True)
    return memory_df

def memory_df_to_markdown(
    memory_df: pd.DataFrame,
    predicate_ids: List[str],
    constraint_ids: List[str],
    include_distance: bool = True,
    include_notes: bool = False
) -> str:
    """
    Convert the memory DataFrame into a markdown table string
    suitable for inclusion in an LLM prompt.

    Parameters:
    - memory_df: DataFrame containing the memory table
    - predicate_ids: list of predicate column names to include (e.g. ['age_min', 'bmi_min'])
    - constraint_ids: list of constraint value column names (e.g. ['max_avg', 'std', 'regions'])
    - include_distance: whether to include the 'distance' column
    - include_notes: whether to include the 'notes' column

    Returns:
    - A markdown-formatted table as a string
    """
    cols = []
    # Only include existing columns
    for col in predicate_ids + constraint_ids:
        if col in memory_df.columns:
            cols.append(col)
    if include_distance and 'distance' in memory_df.columns:
        cols.append('distance')
    if include_notes and 'notes' in memory_df.columns:
        cols.append('notes')
    # Subset the DataFrame
    df_sub = memory_df[cols]
    # Convert to markdown
    markdown_table = df_sub.to_markdown(index=False)
    return markdown_table

def memory_df_to_bullets(
    memory_df: pd.DataFrame,
    predicate_ids: List[str],
    constraint_ids: List[str],
    include_distance: bool = True,
    include_notes: bool = True
) -> str:
    """
    Convert the memory DataFrame into a bullet-list string, one line per row.

    Format:
    - (age_min=X, bmi_min=Y) => max_avg=..., std=..., regions=..., dist=..., note="..."

    Returns:
    - A string with one bullet per memory entry.
    """
    bullets = []
    for _, row in memory_df.iterrows():
        preds = ", ".join(f"{pid}={row[pid]}" for pid in predicate_ids if pid in row)
        cons = ", ".join(f"{cid}={row[cid]}" for cid in constraint_ids if cid in row)
        parts = [preds, cons]
        if include_distance and 'distance' in row:
            parts.append(f"dist={row['distance']}")
        text = "; ".join(parts)
        if include_notes and 'notes' in row:
            text += f"; note=\"{row['notes']}\""
        bullets.append(f"- ({text})")
    return "\n".join(bullets)



# Example usage:
if __name__ == "__main__":
    # Task example
    preds = ['age_min', 'bmi_min']
    consts = ['max_avg', 'std', 'regions']
    orig_q = "SELECT region, AVG(charges), COUNT(*) FROM df WHERE age>=45 AND bmi>=38 GROUP BY region;"
    orig_pred_vals = {'age_min': 45, 'bmi_min': 38.0}
    orig_const_vals = {'max_avg': 20399.29, 'std': 32.47, 'regions': 4}
    mem_df = init_memory_df(preds, consts, orig_q, orig_pred_vals, orig_const_vals, orig_distance=0.0,
                             orig_notes="Original query baseline.")

    # New run example
    new_q = "SELECT region, AVG(charges), COUNT(*) FROM df WHERE age>=43 AND bmi>=40.5 GROUP BY region;"
    new_pred_vals = {'age_min': 43, 'bmi_min': 40.5}
    new_const_vals = {'max_avg': 25703.88, 'std': 21.75, 'regions': 4}
    mem_df = update_memory_df(mem_df, new_q, new_pred_vals, new_const_vals,
                              new_distance=0.09,
                              new_notes="Higher avg, std in range, full coverage.")
    print("### Markdown Table ###")
    print(memory_df_to_markdown(mem_df, preds, consts))
    print("\n### Bullet List ###")
    print(memory_df_to_bullets(mem_df, preds, consts))
