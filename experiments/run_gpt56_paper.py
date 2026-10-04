#!/usr/bin/env python3
"""Run the Section 5 GPT-5.6 Luna comparisons and ablations on original tasks.

Each invocation writes one arm to an isolated directory. Use --resume to keep
completed task/trial pairs, and --task-ids for a small pilot or a category shard.
The baseline uses the repository's iterative direct-refinement protocol with
five candidate calls by default (configurable with --baseline-iters); all OmniTune variants use the paper's T=5, K=5 budget.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = {
    "luna_low_baseline": dict(model="gpt-5.6-luna", effort="low", one_shot=True),
    "luna_high_baseline": dict(model="gpt-5.6-luna", effort="high", one_shot=True),
    "luna_low_omnitune": dict(model="gpt-5.6-luna", effort="low"),
    "luna_low_no_subspace": dict(model="gpt-5.6-luna", effort="low", assignment_only=True),
    "luna_low_no_assignment": dict(model="gpt-5.6-luna", effort="low", random_assignment=True),
    "luna_low_complete_history_no_skyline": dict(model="gpt-5.6-luna", effort="low", history=False, skyline=False),
    "luna_low_summary_no_skyline": dict(model="gpt-5.6-luna", effort="low", history=True, skyline=False),
}

PRICES_PER_MILLION = {
    "gpt-5.6-luna": dict(input=0.20, cached=0.02, cache_write=0.25, output=1.20),
}


class TerminalAPIError(BaseException):
    """Abort the trial immediately when model access cannot recover."""


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def install_adapter(model, effort, max_output_tokens, ledger_path, call_context):
    from chat_model import OpenAIChatModel

    totals = dict(api_calls=0, input_tokens=0, output_tokens=0,
                  reasoning_tokens=0, cached_tokens=0, cache_write_tokens=0,
                  estimated_token_cost_usd=0.0, long_context_calls=0,
                  incomplete_calls=0, api_errors=0, quota_errors=0,
                  terminal_api_errors=0)

    def record_event(event):
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a") as stream:
            stream.write(json.dumps({**call_context, **event}, default=str) + "\n")

    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Expected system prompt followed by conversation messages")
        # Preserve every baseline feedback turn and the agents' actual message context.
        for attempt in range(1, 9):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    instructions=messages[0]["content"],
                    input=[{"role": m["role"], "content": m["content"]} for m in messages[1:]],
                    reasoning={"effort": effort},
                    max_output_tokens=max_output_tokens,
                    store=False,
                    service_tier="default",
                )
                break
            except Exception as exc:
                totals["api_errors"] += 1
                is_quota = "insufficient_quota" in str(exc) or "credit_balance_exhausted" in str(exc)
                retryable = isinstance(exc, (RateLimitError, APIConnectionError,
                                             APITimeoutError, InternalServerError)) and not is_quota
                terminal = not retryable or attempt == 8
                totals["quota_errors"] += int(is_quota and terminal)
                totals["terminal_api_errors"] += int(terminal)
                record_event({"event": "api_error", "model": model, "reasoning_effort": effort,
                              "error_type": type(exc).__name__, "quota_error": is_quota,
                              "retryable": retryable, "terminal": terminal,
                              "attempt": attempt})
                if terminal:
                    raise TerminalAPIError(f"{type(exc).__name__}: API request failed") from exc
                time.sleep(min(60, 2 ** attempt))
        usage = getattr(response, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
        cache_write = int(getattr(in_details, "cache_write_tokens", 0) or 0)
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)
        rates = PRICES_PER_MILLION[model]
        multiplier = 2 if inp > 272_000 else 1
        output_multiplier = 1.5 if inp > 272_000 else 1
        cost = (max(0, inp - cached - cache_write) * rates["input"] * multiplier
                + cached * rates["cached"] * multiplier
                + cache_write * rates["cache_write"] * multiplier
                + out * rates["output"] * output_multiplier) / 1_000_000
        totals["api_calls"] += 1
        totals["input_tokens"] += inp
        totals["output_tokens"] += out
        totals["reasoning_tokens"] += reasoning
        totals["cached_tokens"] += cached
        totals["cache_write_tokens"] += cache_write
        totals["estimated_token_cost_usd"] += cost
        totals["long_context_calls"] += int(inp > 272_000)
        totals["incomplete_calls"] += int(response.status != "completed" or not response.output_text)
        record_event({"event": "response", "model": model, "reasoning_effort": effort,
                      "status": response.status, "input_tokens": inp,
                      "output_tokens": out, "reasoning_tokens": reasoning,
                      "cached_tokens": cached, "cache_write_tokens": cache_write,
                      "estimated_token_cost_usd": cost})
        self.last_usage = {"prompt_tokens": inp, "completion_tokens": out,
                           "total_tokens": inp + out}
        self.total_prompt_tokens += inp
        self.total_completion_tokens += out
        self.total_total_tokens += inp + out
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(f"Response status={response.status}; no completed text")
        return response.output_text

    OpenAIChatModel.generate = generate
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--task-ids", nargs="*", help="e.g. top_k_01 range_01")
    parser.add_argument("--subspace-iters", type=int, default=5)
    parser.add_argument("--assignments-per-subspace", type=int, default=5)
    parser.add_argument("--baseline-iters", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-estimated-cost-usd", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.runs, args.subspace_iters, args.assignments_per_subspace,
           args.baseline_iters, args.max_output_tokens) < 1:
        parser.error("budgets and runs must be positive")

    os.chdir(ROOT)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    worker_lock = (out / "worker.lock").open("a+")
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An existing worker owns this output directory")
    config = ARMS[args.arm]
    from experiments.runner_environment import initialize_environment
    initialize_environment(ROOT, config["model"], "openai", out / "logs")
    call_context = {}
    ledger_path = out / "usage_ledger.jsonl"
    totals = install_adapter(config["model"], config["effort"], args.max_output_tokens,
                             ledger_path, call_context)
    from experiments.common import BENCHMARK_CATEGORIES
    from experiments.refinement_scoring import score
    from opro.opro_main_loop import run_task

    tasks = [(f"{category}_{index + 1:02d}", category, index, task, setting)
             for category, setting in BENCHMARK_CATEGORIES.items()
             for index, task in enumerate(setting["tasks"])]
    if args.task_ids:
        selected = set(args.task_ids)
        unknown = selected - {t[0] for t in tasks}
        if unknown:
            parser.error(f"Unknown task IDs: {sorted(unknown)}")
        tasks = [t for t in tasks if t[0] in selected]

    manifest = dict(arm=args.arm, model=config["model"], reasoning_effort=config["effort"],
                    config=config, task_ids=[t[0] for t in tasks], runs_per_task=args.runs,
                    baseline_iters=args.baseline_iters,
                    subspace_iters=args.subspace_iters,
                    assignments_per_subspace=args.assignments_per_subspace,
                    max_output_tokens=args.max_output_tokens, seed=args.seed,
                    pricing_per_million_tokens=PRICES_PER_MILLION[config["model"]],
                    protocol="Repository run_task; same 32 Section 5 tasks and local scorer; baseline receives iterative feedback; OmniTune variants change only documented ablation switches.")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise SystemExit("Output directory has a different manifest; choose a new directory")
    else:
        save_json(manifest_path, manifest)

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if args.resume and results_path.exists() else []
    if results_path.exists() and not args.resume:
        raise SystemExit("Results already exist; pass --resume")
    incomplete = [row for row in results if not row.get("trial_complete", True)]
    if incomplete:
        save_json(out / "incomplete_trials_before_resume.json", incomplete)
        results = [row for row in results if row.get("trial_complete", True)]
    done = {(row["task_id"], int(row["run"])) for row in results}
    spent = (sum(float(json.loads(line).get("estimated_token_cost_usd", 0))
                 for line in ledger_path.read_text().splitlines()) if ledger_path.exists()
             else sum(float(row.get("estimated_token_cost_usd", 0)) for row in results))
    started = time.monotonic()
    expected = len(tasks) * args.runs
    for task_id, category, index, task, setting in tasks:
        for trial in range(1, args.runs + 1):
            if (task_id, trial) in done:
                continue
            if args.max_estimated_cost_usd is not None and spent >= args.max_estimated_cost_usd:
                save_json(out / "stopped_at_cost_cap.json", dict(cap=args.max_estimated_cost_usd,
                           estimated_cost_usd=spent, completed=len(results), expected=expected))
                print(f"Stopped at estimated cost cap after {len(results)}/{expected} trials", flush=True)
                return
            before = totals.copy()
            call_context.clear()
            call_context.update(task_id=task_id, run=trial)
            begun = time.monotonic()
            error = ""
            query = None
            engine_distance = None
            local = {}
            log_dir = out / "logs" / task_id / f"run_{trial}"
            console_path = out / "console" / task_id / f"run_{trial}.txt"
            console_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with console_path.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
                    query, engine_distance, _ = run_task(
                        task, setting["epsilon"], perform_analysis=False,
                        one_shot_mode=config.get("one_shot", False),
                        one_shot_max_iterations=args.baseline_iters,
                        assignment_lm_only_mode=config.get("assignment_only", False),
                        subspace_lm_only_random=config.get("random_assignment", False),
                        use_history=config.get("history", True),
                        use_skyline=config.get("skyline", True),
                        is_having=(category == "complex"),
                        log_dir=str(log_dir), random_seed=args.seed + trial - 1,
                        max_subspace_iters=args.subspace_iters,
                        max_assignments_per_subspace=args.assignments_per_subspace,
                        model_provider="openai", model_name=config["model"],
                    )
                if query:
                    local = score(query, task, setting, index)
            except TerminalAPIError as exc:
                error = f"TerminalAPIError: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            delta = {k: totals[k] - before[k] for k in totals}
            if delta["quota_errors"] and not error:
                error = "OpenAI API insufficient_quota; trial incomplete"
            if delta["incomplete_calls"] and not query and not error:
                error = f"{delta['incomplete_calls']} API response(s) incomplete or without text"
            trial_complete = not (delta["quota_errors"] or delta["terminal_api_errors"]
                                  or delta["incomplete_calls"])
            row = dict(task_id=task_id, benchmark=category, task=task.name,
                       run=trial, seed=args.seed + trial - 1, arm=args.arm,
                       model=config["model"], reasoning_effort=config["effort"],
                       success=bool(local.get("success", False)) and trial_complete,
                       trial_complete=trial_complete,
                       constraints_satisfied=bool(local.get("constraints_satisfied", False)),
                       engine_distance=engine_distance, distance=local.get("distance"),
                       optimality=local.get("optimality", 0.0),
                       constraint_score=local.get("constraint_score"),
                       result_rows=local.get("result_rows"),
                       domain_issues=local.get("domain_issues", []),
                       predicate_values=local.get("predicate_values", {}),
                       sql=query or "", wall_seconds=time.monotonic() - begun,
                       console_path=str(console_path), log_dir=str(log_dir),
                       error=error, **delta)
            results.append(row)
            results.sort(key=lambda item: (item["task_id"], int(item["run"])))
            spent += delta["estimated_token_cost_usd"]
            save_json(results_path, results)
            save_csv(out / "results.csv", results)
            print(f"[{len(results)}/{expected}] {args.arm} {task_id} run {trial}: "
                  f"success={row['success']} calls={row['api_calls']} "
                  f"tokens={row['input_tokens']}+{row['output_tokens']} "
                  f"cost=${delta['estimated_token_cost_usd']:.4f} "
                  f"error={error[:100]}", flush=True)
            if delta["quota_errors"]:
                save_json(out / "stopped_due_to_quota.json", dict(task_id=task_id,
                          run=trial, completed=len(results), expected=expected,
                          estimated_token_cost_usd=spent))
                return
            if delta["terminal_api_errors"]:
                save_json(out / "stopped_due_to_api_error.json", dict(task_id=task_id,
                          run=trial, completed=len(results), expected=expected, error=error))
                return
    save_json(out / "completion.json", dict(completed=len(results), expected=expected,
              wall_seconds=time.monotonic() - started,
              estimated_token_cost_usd=spent, new_call_usage=totals))


if __name__ == "__main__":
    main()
