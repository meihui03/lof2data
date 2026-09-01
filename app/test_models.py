"""Tests for models.py's decision logic - which models are offerable, and
whether a cloud model's response means "rate limited" or just "busy".

Stdlib unittest so it runs on the UI venv with no extra dependency:

    .venv/bin/python3 test_models.py

The network calls themselves aren't mocked end-to-end; what's tested is the
pure classification each one feeds into, which is where the actual risk is.
Getting rate_limited vs available wrong either locks the user out of a
working model or lets them waste a full extraction on a dead one.
"""

import unittest

import models


class TestIsCloud(unittest.TestCase):
    def test_cloud_suffix_is_cloud(self):
        self.assertTrue(models.is_cloud("minimax-m3:cloud"))
        self.assertTrue(models.is_cloud("qwen3.5:cloud"))

    def test_everything_else_is_local(self):
        self.assertFalse(models.is_cloud("qwen3-vl:8b-instruct"))
        self.assertFalse(models.is_cloud("gemma4:e4b"))

    def test_cloud_in_the_middle_is_not_cloud(self):
        # A local model that merely has "cloud" in its name must not be
        # probed as a cloud model - only the :cloud tag means hosted.
        self.assertFalse(models.is_cloud("cloudy-vision:7b"))


class TestClassifyProbe(unittest.TestCase):
    def test_200_is_available(self):
        self.assertEqual(models.classify_probe(200, "{}"), models.AVAILABLE)

    def test_429_with_session_usage_limit_is_rate_limited(self):
        body = '{"error":"session usage limit reached, please try again later"}'
        self.assertEqual(models.classify_probe(429, body), models.RATE_LIMITED)

    def test_session_usage_limit_match_is_case_insensitive(self):
        body = '{"error":"Session Usage Limit Reached"}'
        self.assertEqual(models.classify_probe(429, body), models.RATE_LIMITED)

    def test_plain_429_is_available(self):
        # extractor.call_llm retries a transient 429 with backoff and
        # succeeds, so it must not disable the model.
        self.assertEqual(models.classify_probe(429, "Too Many Requests"),
                         models.AVAILABLE)

    def test_402_is_payment_required_not_unreachable(self):
        # Ollama cloud's real response once the free allowance is spent. It
        # used to fall through to UNREACHABLE, telling the user "the daemon
        # did not answer" when the daemon had answered immediately with the
        # actual reason - which hid the one fact they needed.
        body = ('{"error":"this model requires a subscription or extra usage, '
                'upgrade for access at https://ollama.com/upgrade"}')
        self.assertEqual(models.classify_probe(402, body),
                         models.PAYMENT_REQUIRED)

    def test_402_is_payment_required_whatever_the_body(self):
        # The status code alone is unambiguous; don't depend on wording.
        self.assertEqual(models.classify_probe(402, ""), models.PAYMENT_REQUIRED)

    def test_server_error_is_unreachable(self):
        self.assertEqual(models.classify_probe(500, "boom"), models.UNREACHABLE)

    def test_404_is_unreachable(self):
        self.assertEqual(models.classify_probe(404, "model not found"),
                         models.UNREACHABLE)


class TestSupportedModels(unittest.TestCase):
    def test_offers_exactly_the_two_benchmarked_models(self):
        self.assertEqual([m["name"] for m in models.SUPPORTED_MODELS],
                         ["minimax-m3:cloud", "qwen3-vl:8b-instruct"])

    def test_cloud_flag_matches_the_name(self):
        for m in models.SUPPORTED_MODELS:
            self.assertEqual(m["is_cloud"], models.is_cloud(m["name"]))

    def test_cloud_listed_first(self):
        self.assertTrue(models.SUPPORTED_MODELS[0]["is_cloud"])


class TestPartition(unittest.TestCase):
    SUPPORTED = models.SUPPORTED_MODELS

    def test_local_model_is_usable_without_a_probe(self):
        usable, blocked = models.partition(self.SUPPORTED, {})
        self.assertIn("qwen3-vl:8b-instruct", usable)
        self.assertEqual(blocked, [])

    def test_rate_limited_cloud_is_blocked_with_a_reason(self):
        usable, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.RATE_LIMITED})
        self.assertNotIn("minimax-m3:cloud", usable)
        self.assertIn("qwen3-vl:8b-instruct", usable)
        self.assertEqual(len(blocked), 1)
        name, reason = blocked[0]
        self.assertEqual(name, "minimax-m3:cloud")
        self.assertIn("usage limit", reason.lower())

    def test_unreachable_cloud_is_blocked_with_a_different_reason(self):
        _, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.UNREACHABLE})
        self.assertIn("unreachable", blocked[0][1].lower())

    def test_payment_required_reason_is_actionable(self):
        # The whole point of the fix: say what's wrong and where to fix it,
        # rather than "the daemon did not answer".
        _, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.PAYMENT_REQUIRED})
        reason = blocked[0][1].lower()
        self.assertIn("subscription", reason)
        self.assertIn("ollama.com", reason)
        self.assertNotIn("did not answer", reason)

    def test_payment_required_leaves_the_local_model_usable(self):
        usable, _ = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.PAYMENT_REQUIRED})
        self.assertEqual(usable, ["qwen3-vl:8b-instruct"])

    def test_uninstalled_model_is_blocked_with_a_pull_hint(self):
        _, blocked = models.partition(
            self.SUPPORTED, {"qwen3-vl:8b-instruct": models.NOT_INSTALLED})
        self.assertIn("ollama pull", blocked[0][1].lower())

    def test_session_marked_model_is_blocked_even_if_probe_says_ok(self):
        # A model that 429'd during a real extraction must stay out until
        # the user re-checks, even though the cached probe still says fine.
        usable, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.AVAILABLE},
            session_rate_limited={"minimax-m3:cloud"})
        self.assertNotIn("minimax-m3:cloud", usable)
        self.assertIn("minimax-m3:cloud", [n for n, _ in blocked])

    def test_session_marked_local_model_is_also_blocked(self):
        usable, _ = models.partition(
            self.SUPPORTED, {}, session_rate_limited={"qwen3-vl:8b-instruct"})
        self.assertNotIn("qwen3-vl:8b-instruct", usable)

    def test_both_blocked_leaves_nothing_usable(self):
        usable, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.RATE_LIMITED,
                             "qwen3-vl:8b-instruct": models.NOT_INSTALLED})
        self.assertEqual(usable, [])
        self.assertEqual(len(blocked), 2)

    def test_a_model_is_never_both_usable_and_blocked(self):
        usable, blocked = models.partition(
            self.SUPPORTED, {"minimax-m3:cloud": models.RATE_LIMITED})
        self.assertTrue(set(usable).isdisjoint({n for n, _ in blocked}))


class TestPickDefault(unittest.TestCase):
    def test_prefers_the_current_production_model(self):
        self.assertEqual(
            models.pick_default(["qwen3-vl:8b-instruct", "minimax-m3:cloud"]),
            "minimax-m3:cloud")

    def test_falls_back_to_first_usable(self):
        self.assertEqual(
            models.pick_default(["qwen3-vl:8b-instruct", "gemma4:e4b"]),
            "qwen3-vl:8b-instruct")

    def test_none_when_nothing_usable(self):
        self.assertIsNone(models.pick_default([]))


class TestLooksRateLimited(unittest.TestCase):
    def test_detects_the_extractor_error_text(self):
        # What run_hybrid_extraction surfaces when call_llm gives up:
        stderr = ("RuntimeError: Ollama cloud session usage limit reached: "
                  '{"error":"session usage limit"}')
        self.assertTrue(models.looks_rate_limited(stderr))

    def test_detects_a_402_hit_mid_extraction(self):
        # A 402 during a real run must also take the model out of the
        # selector. Matching only "session usage limit" let it through, so
        # the user could pick the same dead model again and lose another
        # full OCR pass to it.
        stderr = ('RuntimeError: Ollama cloud subscription required: '
                  '{"error":"this model requires a subscription or extra usage"}')
        self.assertTrue(models.looks_rate_limited(stderr))

    def test_ignores_unrelated_failures(self):
        self.assertFalse(models.looks_rate_limited(
            "FileNotFoundError: input/missing.pdf"))

    def test_handles_empty_stderr(self):
        self.assertFalse(models.looks_rate_limited(""))
        self.assertFalse(models.looks_rate_limited(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
