"""Experiment-local feedback correction and shared estimated-cost reservations."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import json
import math
import os
import uuid
import time


class BudgetLimitReached(BaseException):
    pass


class SharedRateLimiter:
    """Coordinate input TPM, RPM and rolling spend across campaign workers."""
    def __init__(self, root, clock=time.time, sleeper=time.sleep):
        self.root = Path(root)
        self.policy = json.loads((self.root / 'rate_limit_policy.json').read_text())
        self.path = self.root / 'rate_limit_state.json'
        self.clock, self.sleeper = clock, sleeper

    @contextmanager
    def locked(self):
        with (self.root / 'rate_limit.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else dict(
                events=[], cooldown_until=0., next_request_at=0.)
            yield state
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state, indent=2)+'\n')
            os.replace(tmp, self.path)

    def acquire(self, payload, output_cap):
        tokens = len(json.dumps(payload, ensure_ascii=False).encode('utf8')) + 4096
        cost = (tokens * .5 + int(output_cap) * 3.) / 1_000_000
        if tokens > self.policy['input_tokens_per_minute'] or cost > self.policy['usd_per_ten_minutes']:
            raise BudgetLimitReached('A single request exceeds the local rate allowance')
        while True:
            now = self.clock()
            with self.locked() as state:
                state['events'] = [e for e in state['events'] if e['at'] > now - 600]
                minute = [e for e in state['events'] if e['at'] > now - 60]
                wait = max(0., state['cooldown_until'] - now, state['next_request_at'] - now)
                if (len(minute) >= self.policy['requests_per_minute']
                        or sum(e['tokens'] for e in minute) + tokens > self.policy['input_tokens_per_minute']):
                    wait = max(wait, min(e['at'] + 60 - now for e in minute) + .1)
                if sum(e['cost'] for e in state['events']) + cost > self.policy['usd_per_ten_minutes']:
                    wait = max(wait, min(e['at'] + 600 - now for e in state['events']) + .1)
                if wait <= 0:
                    key = str(uuid.uuid4())
                    state['events'].append(dict(key=key, at=now, tokens=tokens, cost=cost))
                    state['next_request_at'] = now + self.policy['minimum_request_interval_seconds']
                    return key
            self.sleeper(min(5., wait))

    def settle(self, key, tokens=None, cost=None):
        with self.locked() as state:
            for e in state['events']:
                if e['key'] == key:
                    if tokens is not None: e['tokens'] = int(tokens)
                    if cost is not None: e['cost'] = float(cost)
                    break

    def cooldown(self, seconds):
        with self.locked() as state:
            state['cooldown_until'] = max(state['cooldown_until'], self.clock() + seconds)


class SharedBudget:
    def __init__(self, root, cap):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'budget.json'
        self.cap = float(cap)

    @contextmanager
    def locked(self):
        with (self.root / 'budget.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else dict(
                cap_usd=self.cap, estimated_spent_usd=0., reservations={})
            if state['cap_usd'] != self.cap:
                raise ValueError('Campaign budget differs from saved cap')
            yield state
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state, indent=2)+'\n')
            os.replace(tmp, self.path)

    def reserve(self, payload, output_cap):
        # Conservative text-only prompt token allowance: UTF-8 payload bytes
        # plus per-message overhead. No additional paid countTokens calls.
        input_allowance = len(json.dumps(payload, ensure_ascii=False).encode('utf8')) + 4096
        allowance = (input_allowance * .5 + int(output_cap) * 3.) / 1_000_000
        with self.locked() as state:
            available = self.cap - state['estimated_spent_usd'] - sum(state['reservations'].values())
            if available < allowance:
                raise BudgetLimitReached(f'${available:.4f} remaining; next-call reservation ${allowance:.4f}')
            key = str(uuid.uuid4())
            state['reservations'][key] = allowance
        return key

    def settle(self, key, actual):
        with self.locked() as state:
            state['reservations'].pop(key)
            state['estimated_spent_usd'] += actual

    def release(self, key):
        with self.locked() as state:
            state['reservations'].pop(key, None)

    def snapshot(self):
        with self.locked() as state:
            return dict(state)


def install_finite_feedback():
    """Apply only in this worker process; original saved scores are untouched."""
    import duckdb
    from functionality.constraint import AgnosticConstraint
    from opro.constraint_parser import MockParser
    old_evaluate = AgnosticConstraint.evaluate

    def evaluate(self, df):
        value = float(self.query(df))
        if not math.isfinite(value):
            return float('inf')
        return old_evaluate(self, df)

    def constraint_score(self, query):
        df = self.input_dataset
        result = duckdb.query(query).to_df()
        if result.empty:
            return float('inf')
        deviations = [float(c.evaluate(result)) for c in self.parsed_constraints]
        if not deviations or any(not math.isfinite(v) for v in deviations):
            return float('inf')
        return sum(deviations) / len(deviations)

    AgnosticConstraint.evaluate = evaluate
    MockParser.evaluate_constraint_score = constraint_score
