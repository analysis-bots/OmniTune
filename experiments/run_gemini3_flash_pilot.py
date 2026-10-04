#!/usr/bin/env python3
"""Run Section 5 Gemini 3 Flash comparisons and ablations on the original tasks.

Each invocation writes one arm to an isolated directory. Use --resume to keep
completed task/trial pairs, and --task-ids for a small pilot or a category shard.
The baseline uses the repository's iterative direct-refinement protocol with
five candidate calls; all OmniTune variants use the paper's T=5, K=5 budget.
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

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.gemini_trial_outcomes import is_finalized, finalize_output_failure

ARMS = {
    "flash_minimal_baseline": dict(model="gemini-3-flash-preview", effort="minimal", one_shot=True),
    "flash_high_baseline": dict(model="gemini-3-flash-preview", effort="high", one_shot=True),
    "flash_low_baseline": dict(model="gemini-3-flash-preview", effort="low", one_shot=True),
    "flash_minimal_omnitune": dict(model="gemini-3-flash-preview", effort="minimal"),
    "flash_minimal_no_subspace": dict(model="gemini-3-flash-preview", effort="minimal", assignment_only=True),
    "flash_minimal_no_assignment": dict(model="gemini-3-flash-preview", effort="minimal", random_assignment=True),
    "flash_minimal_complete_history_no_skyline": dict(model="gemini-3-flash-preview", effort="minimal", history=False, skyline=False),
    "flash_minimal_summary_no_skyline": dict(model="gemini-3-flash-preview", effort="minimal", history=True, skyline=False),
}

PRICES_PER_MILLION = {
    "gemini-3-flash-preview": dict(input=0.50, cached=0.05, output=3.00),
}
from experiments.gemini3_pilot_support import BudgetLimitReached, SharedBudget, SharedRateLimiter, install_finite_feedback


class OutputLimitReached(BaseException):
    """End this trial even if the harness catches ordinary model exceptions."""


class TerminalAPIError(BaseException):
    """Stop immediately; local optimization must not swallow API blockers."""


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def verified_transient_auth_error(exc, detail, attempt, client, api_key, model):
    """Retry this observed 403 only while the same key can read the model.

    Model metadata checks do not generate tokens. Persistent permission errors
    still stop the campaign after at most three generation attempts.
    """
    if (not isinstance(exc, httpx.HTTPStatusError)
            or exc.response.status_code != 403
            or detail.strip() != "A valid API key or GCP project is required."
            or attempt >= 3):
        return False
    try:
        check = client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}",
            headers={"x-goog-api-key": api_key}, timeout=30)
        return check.status_code == 200
    except httpx.TransportError:
        return False


def install_adapter(model, effort, max_output_tokens, ledger_path, call_context,
                    api_key, client=None, shared_budget=None, rate_limiter=None):
    """Replace the repository's legacy Gemini adapter for this experiment only.

    Its existing implementation drops messages after the first user turn. The
    baseline must pass all local-scoring feedback back to the API. We also
    preserve thought signatures when Gemini returns them in text responses.
    """
    from chat_model import GoogleChatModel

    totals = dict(api_calls=0, input_tokens=0, output_tokens=0,
                  reasoning_tokens=0, cached_tokens=0,
                  estimated_token_cost_usd=0.0, long_context_calls=0,
                  incomplete_calls=0, output_limit_calls=0, api_errors=0, quota_errors=0,
                  terminal_api_errors=0)
    client = client or httpx.Client(timeout=180)

    def record_event(event):
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a") as stream:
            stream.write(json.dumps({"time_unix": time.time(), **call_context, **event}, default=str) + "\n")

    def generate(self, messages, temperature=0.0, top_p=None, response_format=None):
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Expected system prompt followed by conversation messages")
        parts_by_text = getattr(self, "_gemini_parts_by_text", {})
        contents = []
        for message in messages[1:]:
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(f"Unsupported conversation role: {role}")
            body = message["content"]
            parts = parts_by_text.get(body, [{"text": body}]) if role == "assistant" else [{"text": body}]
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": parts})
        payload = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": contents,
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": effort},
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        reservation = shared_budget.reserve(payload, max_output_tokens) if shared_budget else None
        failed_payload_path = None
        for attempt in range(1, 9):
            try:
                rate_key = rate_limiter.acquire(payload, max_output_tokens) if rate_limiter else None
            except BudgetLimitReached:
                if shared_budget:
                    shared_budget.release(reservation)
                raise
            try:
                response = client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                    headers={"x-goog-api-key": api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                totals["api_errors"] += 1
                if failed_payload_path is None:
                    failed_payload_path = ledger_path.parent / "failed_request_payloads" / f"{call_context['task_id']}_run{call_context['run']}_{time.time_ns()}.json"
                    save_json(failed_payload_path, payload)
                detail = ""
                error_details = []
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        api_error = exc.response.json().get("error", {})
                        detail = str(api_error.get("message", ""))
                        error_details = api_error.get("details", [])
                    except ValueError:
                        detail = exc.response.text[:500]
                lower = detail.lower()
                is_quota = any(term in lower for term in
                               ("billing", "credit", "quota exceeded", "quota exhausted",
                                "free tier limit", "insufficient quota"))
                is_payment = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 402
                auth_recovery = verified_transient_auth_error(
                    exc, detail, attempt, client, api_key, self.model_name)
                retryable = ((isinstance(exc, httpx.TransportError) or
                              isinstance(exc, httpx.HTTPStatusError) and
                              exc.response.status_code in (429, 500, 502, 503, 504)
                              or auth_recovery)
                             and not is_payment)
                terminal = not retryable or attempt == 8
                if rate_limiter:
                    # Failed HTTP requests have no returned generation usage.
                    # Retain their conservative input allowance for this minute.
                    if isinstance(exc, httpx.HTTPStatusError):
                        rate_limiter.settle(rate_key, cost=0.)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        rate_limiter.cooldown(min(180, 45 * attempt))
                totals["quota_errors"] += int(is_quota and terminal)
                totals["terminal_api_errors"] += int(terminal)
                record_event({"event": "api_error", "model": model, "thinking_level": effort,
                              "api_model_name": self.model_name,
                              "request_payload_path": str(failed_payload_path),
                              "error_type": type(exc).__name__, "quota_error": is_quota,
                              "retryable": retryable, "terminal": terminal,
                              "same_key_model_access_verified": auth_recovery,
                              "attempt": attempt, "status_code": getattr(getattr(exc, "response", None), "status_code", None),
                              "detail": detail[:1500]})
                if error_details:
                    record_event({"event": "api_error_details", "attempt": attempt,
                                  "details": error_details})
                if terminal:
                    if shared_budget:
                        shared_budget.release(reservation)
                    raise TerminalAPIError(f"HTTP {getattr(getattr(exc, 'response', None), 'status_code', None)}: {detail or type(exc).__name__}") from exc
                time.sleep(min(60, 2 ** attempt))
        usage = data.get("usageMetadata", {})
        inp = int(usage.get("promptTokenCount", 0) or 0)
        candidates = int(usage.get("candidatesTokenCount", 0) or 0)
        reasoning = int(usage.get("thoughtsTokenCount", 0) or 0)
        out = candidates + reasoning
        cached = int(usage.get("cachedContentTokenCount", 0) or 0)
        rates = PRICES_PER_MILLION[model]
        cost = (max(0, inp - cached) * rates["input"] +
                cached * rates["cached"] + out * rates["output"]) / 1_000_000
        if rate_limiter:
            rate_limiter.settle(rate_key, tokens=inp, cost=cost)
        if shared_budget:
            shared_budget.settle(reservation, cost)
        raw_response_path = ledger_path.parent / "api_responses" / f"{call_context['task_id']}_run{call_context['run']}_{time.time_ns()}.json"
        save_json(raw_response_path, data)
        candidate = (data.get("candidates") or [{}])[0]
        model_parts = candidate.get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in model_parts if not p.get("thought", False)]
        result_text = "".join(text_parts)
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        totals["api_calls"] += 1
        totals["input_tokens"] += inp
        totals["output_tokens"] += out
        totals["reasoning_tokens"] += reasoning
        totals["cached_tokens"] += cached
        totals["estimated_token_cost_usd"] += cost
        totals["incomplete_calls"] += int(finish_reason != "STOP" or not result_text)
        record_event({"event": "response", "model": model, "thinking_level": effort,
                      "status": finish_reason, "input_tokens": inp,
                      "output_tokens": out, "reasoning_tokens": reasoning,
                      "cached_tokens": cached, "candidate_tokens": candidates,
                      "estimated_token_cost_usd": cost, "raw_response_path": str(raw_response_path)})
        self.last_usage = {"prompt_tokens": inp, "completion_tokens": out,
                           "total_tokens": inp + out}
        self.total_prompt_tokens += inp
        self.total_completion_tokens += out
        self.total_total_tokens += inp + out
        if finish_reason == "MAX_TOKENS":
            totals["output_limit_calls"] += 1
            raise OutputLimitReached(f"Gemini response reached {max_output_tokens}-token output cap")
        if finish_reason != "STOP" or not result_text:
            raise RuntimeError(f"Gemini response finishReason={finish_reason}; no completed text")
        # The harness accepts fenced JSON, while Gemini's JSON mode returns a
        # bare object. Preserve the original API parts for thought signatures.
        formatted_text = f"```json\n{result_text}\n```"
        parts_by_text[formatted_text] = model_parts
        self._gemini_parts_by_text = parts_by_text
        return formatted_text

    GoogleChatModel.generate = generate
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
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-cap-usd", type=float, default=8.0)
    parser.add_argument("--campaign-scope", default="Selected original Section 5 tasks; configurable baseline and T/K budgets")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-estimated-cost-usd", type=float, required=True,
                        help="Per-invocation API cost cap based on reported token usage")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write and validate the shard manifest without API calls")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.runs, args.subspace_iters, args.assignments_per_subspace,
           args.baseline_iters, args.max_output_tokens) < 1:
        parser.error("budgets and runs must be positive")
    if args.max_estimated_cost_usd <= 0:
        parser.error("--max-estimated-cost-usd must be positive")

    os.chdir(ROOT)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    worker_lock = (out / 'worker.lock').open('a+')
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An existing worker owns this output directory; no duplicate launched')
    config = ARMS[args.arm]
    from experiments.runner_environment import initialize_environment
    initialize_environment(ROOT, config["model"], "google", out / "logs")
    from chat_model import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        parser.error("GEMINI_API_KEY is required")
    call_context = {}
    ledger_path = out / "usage_ledger.jsonl"
    install_finite_feedback()
    shared_budget = SharedBudget(args.campaign_root.resolve(), args.campaign_cap_usd)
    rate_limiter = (SharedRateLimiter(args.campaign_root.resolve())
                    if (args.campaign_root / 'rate_limit_policy.json').exists() else None)
    totals = install_adapter(config["model"], config["effort"], args.max_output_tokens,
                             ledger_path, call_context, GEMINI_API_KEY, shared_budget=shared_budget,
                             rate_limiter=rate_limiter)
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

    manifest = dict(arm=args.arm, model=config["model"], thinking_level=config["effort"],
                    response_mime_type="application/json",
                    config=config, task_ids=[t[0] for t in tasks], runs_per_task=args.runs,
                    baseline_iters=args.baseline_iters,
                    subspace_iters=args.subspace_iters,
                    assignments_per_subspace=args.assignments_per_subspace,
                    max_output_tokens=args.max_output_tokens, seed=args.seed,
                    epsilon_by_benchmark={category: setting['epsilon'] for _,category,_,_,setting in tasks},
                    pricing_per_million_tokens=PRICES_PER_MILLION[config["model"]],
                    feedback_guard="finite_feedback_v1: empty query results and non-finite raw constraint values fail; applies to all arms",
                    campaign_scope=args.campaign_scope,
                    protocol="Repository run_task; direct baseline uses configured candidate-call budget with local scoring feedback and no API tools; OmniTune uses configured T/K budgets plus subspace calls. Minimal is not guaranteed no-thinking; high uses dynamic thinkingLevel, not a 4096-token thinkingBudget. All arms JSON, configured maxOutputTokens; output cap finalizes unsuccessful. Experiment-local finite-feedback guard applies to all arms.")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise SystemExit("Output directory has a different manifest; choose a new directory")
    else:
        save_json(manifest_path, manifest)
    if args.prepare_only:
        print(f"Prepared {manifest_path}", flush=True)
        return

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if args.resume and results_path.exists() else []
    if results_path.exists() and not args.resume:
        raise SystemExit("Results already exist; pass --resume")
    incomplete = [row for row in results if not is_finalized(row)]
    if incomplete:
        save_json(out / "incomplete_trials_before_resume.json", incomplete)
        results = [row for row in results if is_finalized(row)]
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
                        model_provider="google", model_name=config["model"],
                    )
                if query:
                    local = score(query, task, setting, index)
            except BudgetLimitReached as exc:
                totals["terminal_api_errors"] += 1
                error = f"BudgetLimitReached: {exc}"
            except OutputLimitReached as exc:
                error = f"OutputLimitReached: {exc}"
            except TerminalAPIError as exc:
                error = f"TerminalAPIError: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            delta = {k: totals[k] - before[k] for k in totals}
            if delta["quota_errors"] and not error:
                error = "Gemini API quota or billing error; trial incomplete"
            if delta["incomplete_calls"] and not query and not error:
                error = f"{delta['incomplete_calls']} API response(s) incomplete or without text"
            trial_complete = not (delta["quota_errors"] or delta["terminal_api_errors"]
                                  or delta["incomplete_calls"])
            row = dict(task_id=task_id, benchmark=category, task=task.name,
                       run=trial, seed=args.seed + trial - 1, arm=args.arm,
                       model=config["model"], thinking_level=config["effort"],
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
            row["trial_finalized"] = trial_complete
            if delta["output_limit_calls"]:
                row = finalize_output_failure(row)
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
            if error.startswith("BudgetLimitReached:"):
                save_json(out / "stopped_at_campaign_cap.json", dict(task_id=task_id, run=trial,
                    finalized=len([r for r in results if is_finalized(r)]), error=error,
                    shared_budget=shared_budget.snapshot()))
                return
            if delta["quota_errors"]:
                save_json(out / "stopped_due_to_quota.json", dict(task_id=task_id,
                          run=trial, completed=len(results), expected=expected,
                          estimated_token_cost_usd=spent))
                return
            if delta["terminal_api_errors"]:
                save_json(out / "stopped_due_to_api_error.json", dict(task_id=task_id,
                          run=trial, completed=len(results), expected=expected,
                          error=error[:500]))
                return
    save_json(out / "completion.json", dict(completed=len(results), expected=expected,
              wall_seconds=time.monotonic() - started,
              estimated_token_cost_usd=spent, new_call_usage=totals))


if __name__ == "__main__":
    main()
