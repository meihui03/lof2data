"""Tests for the Claude API path in extractor.py.

    .venv/bin/python3 test_claude.py

No request leaves the machine: the request is built by a pure function, and
the one test that goes through call_llm swaps the SDK client for a stand-in.
That is where the risk is - a malformed request costs a paid call to find out,
and a misread response (a refusal, a reply cut off at max_tokens) would
otherwise surface as "could not parse JSON", three retries later, with the
credit already spent.
"""

import base64
import io
import json
import os
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import extractor as E


def png_b64(width, height, noise=False):
    if noise:
        pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
    else:
        img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def response(*blocks, stop_reason="end_turn", input_tokens=9000, output_tokens=1200):
    return types.SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def text(t):
    return types.SimpleNamespace(type="text", text=t)


class TestIsClaudeModel(unittest.TestCase):
    def test_claude_ids_are_claude(self):
        self.assertTrue(E.is_claude_model("claude-sonnet-5-5"))

    def test_ollama_models_are_not(self):
        for name in ["minimax-m3:cloud", "qwen3-vl:8b-instruct", "qwen3.5:cloud"]:
            self.assertFalse(E.is_claude_model(name), name)


class TestBuildClaudeRequest(unittest.TestCase):
    def test_images_come_before_the_prompt(self):
        req = E.build_claude_request("extract", "claude-sonnet-5-5", [png_b64(10, 10)] * 2)
        content = req["messages"][0]["content"]
        self.assertEqual([b["type"] for b in content], ["image", "image", "text"])
        self.assertEqual(content[-1]["text"], "extract")

    def test_small_page_is_sent_as_png_unchanged(self):
        page = png_b64(10, 10)
        req = E.build_claude_request("x", "claude-sonnet-5-5", [page])
        source = req["messages"][0]["content"][0]["source"]
        self.assertEqual(source, {"type": "base64", "media_type": "image/png", "data": page})

    def test_low_effort_and_no_format_in_prompt_mode(self):
        req = E.build_claude_request("x", "claude-sonnet-5-5", [])
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["output_config"], {"effort": "low"})
        self.assertGreaterEqual(req["max_tokens"], 8000)

    def test_schema_becomes_a_json_schema_format(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        req = E.build_claude_request("x", "claude-sonnet-5-5", [], schema=schema)
        self.assertEqual(req["output_config"]["format"],
                         {"type": "json_schema", "schema": schema})

    def test_every_object_in_the_pydantic_schema_is_closed(self):
        # Claude rejects a structured-output schema with any object left open,
        # and Pydantic's output (nested $defs included) leaves them all open.
        import schema_models
        schema = schema_models.extraction_json_schema()
        sent = E.build_claude_request("x", "claude-sonnet-5-5", [], schema=schema)
        objects = []

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    objects.append(node)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(sent["output_config"]["format"]["schema"])
        self.assertTrue(objects)
        self.assertTrue(all(o.get("additionalProperties") is False for o in objects))
        # The Ollama path is handed the same schema object - left untouched.
        self.assertNotIn("additionalProperties", json.dumps(schema))

    def test_text_only_call_has_just_the_prompt(self):
        req = E.build_claude_request("x", "claude-sonnet-5-5", None)
        self.assertEqual(req["messages"][0]["content"], [{"type": "text", "text": "x"}])


class TestClaudeImageBlock(unittest.TestCase):
    def test_oversized_page_is_reencoded_as_jpeg_under_the_limit(self):
        # Noise defeats PNG compression, the way a phone photo does.
        page = png_b64(1500, 2000, noise=True)
        self.assertGreater(len(base64.b64decode(page)), E.CLAUDE_MAX_IMAGE_BYTES)
        block = E.claude_image_block(page)
        self.assertEqual(block["source"]["media_type"], "image/jpeg")
        self.assertLessEqual(len(base64.b64decode(block["source"]["data"])),
                             E.CLAUDE_MAX_IMAGE_BYTES)


class TestParseClaudeResponse(unittest.TestCase):
    def test_returns_text_and_token_meta(self):
        content, meta = E.parse_claude_response(response(text('{"a": 1}')), elapsed=12.34)
        self.assertEqual(content, '{"a": 1}')
        self.assertEqual(meta, {"prompt_eval_count": 9000, "eval_count": 1200, "time_s": 12.3})

    def test_thinking_blocks_are_not_part_of_the_answer(self):
        thinking = types.SimpleNamespace(type="thinking", thinking="")
        content, _ = E.parse_claude_response(response(thinking, text("{}")), elapsed=1)
        self.assertEqual(content, "{}")

    def test_refusal_raises_instead_of_returning_empty_text(self):
        with self.assertRaisesRegex(RuntimeError, "refused"):
            E.parse_claude_response(response(stop_reason="refusal"), elapsed=1)

    def test_cut_off_reply_raises_instead_of_being_retried_as_bad_json(self):
        with self.assertRaisesRegex(RuntimeError, "max_tokens"):
            E.parse_claude_response(response(text('{"a": '), stop_reason="max_tokens"), elapsed=1)


class TestClaudeClient(unittest.TestCase):
    # An org-level key (not tied to a workspace) is refused with "must
    # include the anthropic-workspace-id header" - ANTHROPIC_WORKSPACE_ID
    # supplies it.
    def test_workspace_id_is_sent_when_set(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_WORKSPACE_ID": "wrkspc_123"}
        with mock.patch.dict(os.environ, env):
            client = E._claude_client(10)
        self.assertEqual(client.default_headers.get("anthropic-workspace-id"), "wrkspc_123")

    def test_no_workspace_header_for_a_workspace_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-x"}):
            os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)
            client = E._claude_client(10)
        self.assertNotIn("anthropic-workspace-id", client.default_headers)


class TestCallLlmRouting(unittest.TestCase):
    def test_claude_model_goes_to_the_sdk_not_ollama(self):
        client = mock.Mock()
        client.messages.create.return_value = response(text("{}"))
        with mock.patch.object(E, "_claude_client", return_value=client), \
             mock.patch.object(E.requests, "post") as ollama:
            content, meta = E.call_llm("x", model="claude-sonnet-5-5", images=[png_b64(10, 10)],
                                       return_meta=True)
        ollama.assert_not_called()
        self.assertEqual(content, "{}")
        self.assertEqual(meta["eval_count"], 1200)
        sent = client.messages.create.call_args.kwargs
        self.assertEqual(sent["model"], "claude-sonnet-5-5")

    def test_without_return_meta_only_the_text_comes_back(self):
        client = mock.Mock()
        client.messages.create.return_value = response(text("{}"))
        with mock.patch.object(E, "_claude_client", return_value=client):
            self.assertEqual(E.call_llm("x", model="claude-sonnet-5-5"), "{}")


if __name__ == "__main__":
    unittest.main()
