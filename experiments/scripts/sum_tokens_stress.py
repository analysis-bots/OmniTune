# Updated parser to scan all sub-dirs for `clean_responses.log`, write per-file CSVs,
# and optionally produce a single aggregated CSV.
import os.path
import re
from pathlib import Path
import csv
import pandas as pd
from typing import List, Tuple

# --- Config ---
ROOT_DIR = Path("../logs/log_dir_oct_16_gpt_41_mini_stress_test_range")  # change this to your root folder
# ROOT_DIR = Path("../logs/log_dir_oct_16_gpt_41_mini_stress_test_top_k")        # change this to your root folder
TARGET_BASENAME = "clean_responses.log" # the exact file name to discover
WRITE_AGGREGATE = False                  # set False to skip the combined CSV

ITER_START_RE = re.compile(r'^\[Subspace\]\s*t\s*=\s*1\b', re.IGNORECASE)
SUBSPACE_LINE_RE = re.compile(r'^\[Subspace\]', re.IGNORECASE)
REFINEMENT_LINE_RE = re.compile(r'^\[Refinement\]', re.IGNORECASE)

# Token counter appears as "acc_tokens_total=NNNN"
TOKENS_RE = re.compile(r'acc_tokens_total\s*[:=]\s*(\d+)', re.IGNORECASE)

def extract_tokens(line: str):
    m = TOKENS_RE.search(line)
    return int(m.group(1)) if m else None

def process_lines(lines: List[str]) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []

    in_iter = False
    subspace_count = 0
    refinement_count = 0
    last_subspace_tokens = None
    last_refinement_tokens = None

    def flush_iteration():
        nonlocal subspace_count, refinement_count, last_subspace_tokens, last_refinement_tokens
        if not in_iter:
            return
        s_tok = last_subspace_tokens if last_subspace_tokens is not None else 0
        r_tok = last_refinement_tokens if last_refinement_tokens is not None else 0
        rows.append((subspace_count, refinement_count, s_tok + r_tok))

    for raw in lines:
        line = raw.rstrip("\n")

        # Start of new iteration?
        if ITER_START_RE.match(line):
            if in_iter:
                flush_iteration()
                subspace_count = 0
                refinement_count = 0
                last_subspace_tokens = None
                last_refinement_tokens = None
            in_iter = True

            # Count the start line as a Subspace entry
            subspace_count += 1
            tok = extract_tokens(line)
            if tok is not None:
                last_subspace_tokens = tok
            continue

        if not in_iter:
            continue

        if SUBSPACE_LINE_RE.match(line):
            subspace_count += 1
            tok = extract_tokens(line)
            if tok is not None:
                last_subspace_tokens = tok
            continue

        if REFINEMENT_LINE_RE.match(line):
            refinement_count += 1
            tok = extract_tokens(line)
            if tok is not None:
                last_refinement_tokens = tok
            continue

    if in_iter:
        flush_iteration()

    return rows

def discover_inputs(root: Path, target_name: str) -> List[Path]:
    if root.is_file():
        return [root] if root.name == target_name else []
    if root.is_dir():
        return sorted(root.rglob(target_name))
    return []

def write_csv(out_path: Path, rows: List[Tuple[int,int,int]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Subspaces", "Refinements", "Num_Tokens"])
        w.writerows(rows)

suffixes = ["top_k", "range", "diversity", "complex"]

for suffix in suffixes:
    root_dir = Path(f"../logs/log_dir_oct_16_gpt_41_mini_stress_test_{suffix}")
    # --- Run ---
    inputs = discover_inputs(root_dir, TARGET_BASENAME)

    combined_records = []  # (Source_Dir, Iteration, Subspaces, Refinements, Num_Tokens, CSV_Path)
    for log_path in inputs:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rows = process_lines(lines)
        out_csv_path = Path(os.path.join(str(root_dir), f"log_summary_tokens_{log_path.parent.name}.csv"))
        write_csv(out_csv_path, rows)
        for i, (s, r, t) in enumerate(rows, start=1):
            combined_records.append((str(log_path.parent), i, s, r, t, out_csv_path.as_posix()))

    print(f"Processed {len(inputs)} log files.")

# total aggregate CSV
    if WRITE_AGGREGATE and combined_records:
        aggregate_path = root_dir / "aggregate_tokens_stress_test.csv"
        with aggregate_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Source_Dir", "Iteration", "Subspaces", "Refinements", "Num_Tokens", "CSV_Path"])
            w.writerows(combined_records)
        print(f"Wrote aggregate CSV with {len(combined_records)} rows to {aggregate_path}")
    else:
        print("No valid CSVs to aggregate after filtering.")
