"""Offline adapter and budget checks; no provider API requests are made."""
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from experiments.runner_environment import initialize_environment
from experiments import run_gpt56_paper as gpt
from experiments import run_gemini3_flash_pilot as gemini
from experiments.gemini3_pilot_support import BudgetLimitReached, SharedBudget, SharedRateLimiter
from experiments.gemini_trial_outcomes import finalize_output_failure, is_finalized


def fake_model(cls, name):
    model = cls.__new__(cls)
    model.model_name = name
    model.total_prompt_tokens = model.total_completion_tokens = model.total_total_tokens = 0
    return model


class ModelRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_environment(self.root, 'gpt-5.6-luna', 'openai', self.root / 'logs')
        from chat_model import GoogleChatModel, OpenAIChatModel
        self.google_cls, self.openai_cls = GoogleChatModel, OpenAIChatModel
        for cls in (self.google_cls, self.openai_cls):
            old = cls.generate
            self.addCleanup(setattr, cls, 'generate', old)
        self.messages = [{'role': 'system', 'content': 'system'},
                         {'role': 'user', 'content': 'task'},
                         {'role': 'assistant', 'content': 'candidate'},
                         {'role': 'user', 'content': 'local feedback'}]
        self.context = {'task_id': 'top_k_01', 'run': 1}

    def test_openai_preserves_feedback_and_controls_reasoning(self):
        totals = gpt.install_adapter('gpt-5.6-luna', 'high', 8192,
                                     self.root / 'ledger.jsonl', self.context)
        response = SimpleNamespace(status='completed', output_text='answer', usage=SimpleNamespace(
            input_tokens=100, output_tokens=30,
            input_tokens_details=SimpleNamespace(cached_tokens=20, cache_write_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=10)))
        model = fake_model(self.openai_cls, 'gpt-5.6-luna')
        create = Mock(return_value=response)
        model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        self.assertEqual(model.generate(self.messages), 'answer')
        payload = create.call_args.kwargs
        self.assertEqual(payload['input'], self.messages[1:])
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertEqual(payload['max_output_tokens'], 8192)
        self.assertFalse(payload['store'])
        self.assertNotIn('tools', payload)
        self.assertEqual(totals['reasoning_tokens'], 10)
        self.assertAlmostEqual(totals['estimated_token_cost_usd'], (80*.20 + 20*.02 + 30*1.20)/1e6)

    def test_openai_terminal_errors_abort_without_optimizer_swallowing(self):
        totals = gpt.install_adapter('gpt-5.6-luna', 'low', 8192,
                                     self.root / 'ledger.jsonl', self.context)
        model = fake_model(self.openai_cls, 'gpt-5.6-luna')
        create = Mock(side_effect=ValueError('invalid request'))
        model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with self.assertRaises(gpt.TerminalAPIError):
            model.generate(self.messages)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(totals['terminal_api_errors'], 1)
        self.assertFalse(issubclass(gpt.TerminalAPIError, Exception))

    def gemini_adapter(self, handler, effort='minimal', budget=None):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        totals = gemini.install_adapter('gemini-3-flash-preview', effort, 8192,
                                        self.root / 'ledger.jsonl', self.context,
                                        'unit-test-key', client, shared_budget=budget)
        return fake_model(self.google_cls, 'gemini-3-flash-preview'), totals

    def test_gemini_thinking_history_signatures_and_usage(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                'candidates': [{'content': {'parts': [{'text': '{"value":1}', 'thoughtSignature': 'sig'}]},
                                'finishReason': 'STOP'}],
                'usageMetadata': {'promptTokenCount': 100, 'cachedContentTokenCount': 20,
                                  'candidatesTokenCount': 10, 'thoughtsTokenCount': 5}})
        model, totals = self.gemini_adapter(handler, effort='low')
        first = self.messages[:2]
        result = model.generate(first)
        model.generate(first + [{'role': 'assistant', 'content': result}, self.messages[-1]])
        payload = requests[-1]
        self.assertEqual(payload['contents'][1]['parts'][0]['thoughtSignature'], 'sig')
        self.assertEqual(payload['contents'][-1]['parts'], [{'text': 'local feedback'}])
        self.assertEqual(payload['generationConfig']['thinkingConfig'], {'thinkingLevel': 'low'})
        self.assertEqual(payload['generationConfig']['responseMimeType'], 'application/json')
        self.assertEqual(payload['generationConfig']['maxOutputTokens'], 8192)
        self.assertNotIn('tools', payload)
        self.assertEqual(totals['output_tokens'], 30)
        self.assertEqual(totals['reasoning_tokens'], 10)
        self.assertAlmostEqual(totals['estimated_token_cost_usd'], 2*(80*.5+20*.05+15*3)/1e6)

    def test_output_cap_is_finalized_failure_and_never_retried(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'candidates': [{'finishReason': 'MAX_TOKENS'}],
                'usageMetadata': {'promptTokenCount': 20, 'thoughtsTokenCount': 8192}})
        model, totals = self.gemini_adapter(handler)
        with self.assertRaises(gemini.OutputLimitReached):
            model.generate(self.messages)
        self.assertEqual(len(requests), 1)
        self.assertEqual(totals['output_limit_calls'], 1)
        row = finalize_output_failure({'success': True, 'optimality': .9})
        self.assertTrue(is_finalized(row))
        self.assertFalse(row['success'])
        self.assertEqual(row['optimality'], 0)

    def test_other_incomplete_output_is_not_mislabeled_as_token_cap(self):
        model, totals = self.gemini_adapter(lambda r: httpx.Response(200, json={
            'candidates': [{'finishReason': 'SAFETY'}]}))
        with self.assertRaises(RuntimeError):
            model.generate(self.messages)
        self.assertEqual(totals['incomplete_calls'], 1)
        self.assertEqual(totals['output_limit_calls'], 0)

    def test_payment_error_stops_and_releases_reservation(self):
        budget = SharedBudget(self.root, 1)
        model, totals = self.gemini_adapter(lambda r: httpx.Response(402,
            json={'error': {'message': 'Your prepayment credits are depleted.'}}), budget=budget)
        with self.assertRaises(gemini.TerminalAPIError):
            model.generate(self.messages)
        self.assertEqual(totals['api_errors'], 1)
        self.assertEqual(budget.snapshot()['reservations'], {})
        self.assertEqual(budget.snapshot()['estimated_spent_usd'], 0)

    def test_429_has_bounded_attempts(self):
        model, totals = self.gemini_adapter(lambda r: httpx.Response(429,
            json={'error': {'message': 'Resource has been exhausted'}}))
        with patch.object(gemini.time, 'sleep'), self.assertRaises(gemini.TerminalAPIError):
            model.generate(self.messages)
        self.assertEqual(totals['api_errors'], 8)
        self.assertEqual(totals['terminal_api_errors'], 1)

    def test_direct_attempt_limit_is_optional_and_keeps_old_default(self):
        from opro.opro_main_loop import OmniTuneEngine, run_task
        for function in (OmniTuneEngine.__init__, run_task):
            signature = inspect.signature(function)
            self.assertEqual(signature.parameters['one_shot_max_iterations'].default, 10)
            self.assertEqual(list(signature.parameters)[-1], 'one_shot_max_iterations')

    def test_shared_budget_refuses_overspend(self):
        budget = SharedBudget(self.root, .001)
        with self.assertRaises(BudgetLimitReached):
            budget.reserve({'contents': 'task'}, 8192)
        self.assertEqual(budget.snapshot()['reservations'], {})

    def test_rate_limiter_paces_across_instances(self):
        (self.root / 'rate_limit_policy.json').write_text(json.dumps(dict(
            input_tokens_per_minute=1_200_000, requests_per_minute=120,
            usd_per_ten_minutes=6, minimum_request_interval_seconds=.75)))
        clock = [1000.0]
        def sleep(seconds):
            clock[0] += seconds
        first = SharedRateLimiter(self.root, clock=lambda: clock[0], sleeper=sleep)
        second = SharedRateLimiter(self.root, clock=lambda: clock[0], sleeper=sleep)
        key = first.acquire({'contents': 'task'}, 8192)
        first.settle(key, tokens=10, cost=.00001)
        second.acquire({'contents': 'task'}, 8192)
        self.assertGreaterEqual(clock[0], 1000.75)
        first.cooldown(45)
        second.acquire({'contents': 'task'}, 8192)
        self.assertGreaterEqual(clock[0], 1045.75)


if __name__ == '__main__':
    unittest.main()
