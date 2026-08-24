"""三个协议适配器的请求体、端点和 SSE usage 回归测试。"""

from __future__ import annotations

import unittest

from llm_bench import session
from llm_bench.protocols.chat import client as chat
from llm_bench.protocols.messages import client as messages
from llm_bench.protocols.responses import client as responses


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines: list[str]):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def iter_lines(self, **kwargs):
        self.test_case.assertEqual(
            kwargs, {"chunk_size": 256, "decode_unicode": False}
        )
        return iter(line.encode() for line in self.lines)


class ProtocolAdapterTest(unittest.TestCase):
    def setUp(self):
        session.configure("stable-affinity")

    def exercise(self, module, lines, **stream_kwargs):
        captured = {}
        response = FakeResponse(lines)
        response.test_case = self
        original = module.post_stream

        def fake_post(url, api_key, payload, timeout, session_id=None):
            captured.update(
                url=url,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
                session_id=session_id,
            )
            return response

        module.post_stream = fake_post
        try:
            result = module.stream(
                "https://protocol.example",
                "secret",
                "model",
                "system",
                "user",
                32,
                **stream_kwargs,
            )
        finally:
            module.post_stream = original
        return captured, result

    def test_chat(self):
        request, result = self.exercise(
            chat,
            [
                'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}',
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                'data: {"usage":{"prompt_tokens":100,"completion_tokens":2,'
                '"prompt_tokens_details":{"cached_tokens":80}},"choices":[]}',
                "data: [DONE]",
            ],
        )
        self.assertEqual(
            request["url"], "https://protocol.example/v1/chat/completions"
        )
        self.assertEqual(request["payload"]["messages"][0]["role"], "system")
        self.assertEqual(
            request["payload"]["prompt_cache_key"], "stable-affinity"
        )
        self.assertEqual(
            (result["input_tokens"], result["cached_tokens"], result["text"]),
            (100, 80, "ok"),
        )

    def test_responses(self):
        request, result = self.exercise(
            responses,
            [
                "event: response.reasoning_text.delta",
                'data: {"delta":"r"}',
                "",
                'data: {"type":"response.output_text.delta","delta":"ok"}',
                'data: {"type":"response.completed","response":{"usage":'
                '{"input_tokens":100,"output_tokens":2,'
                '"input_tokens_details":{"cached_tokens":80}}}}',
            ],
        )
        self.assertEqual(request["url"], "https://protocol.example/v1/responses")
        self.assertEqual(request["payload"]["max_output_tokens"], 32)
        self.assertEqual(
            request["payload"]["prompt_cache_key"], "stable-affinity"
        )
        self.assertEqual(
            (result["input_tokens"], result["cached_tokens"], result["text"]),
            (100, 80, "ok"),
        )

    def test_messages_nested_usage(self):
        request, result = self.exercise(
            messages,
            [
                'data: {"type":"message_start","message":{"usage":'
                '{"input_tokens":0}}}',
                'data: {"type":"content_block_delta","delta":'
                '{"type":"thinking_delta","thinking":"r"}}',
                'data: {"type":"content_block_delta","delta":'
                '{"type":"text_delta","text":"ok"}}',
                'data: {"type":"message_delta","metadata":{"usage":'
                '{"input_tokens":20,"output_tokens":2,'
                '"cache_read_input_tokens":80}}}',
                'data: {"type":"message_stop"}',
            ],
        )
        self.assertEqual(request["url"], "https://protocol.example/v1/messages")
        self.assertEqual(request["payload"]["system"], "system")
        self.assertNotIn("prompt_cache_key", request["payload"])
        self.assertEqual(
            (result["input_tokens"], result["cached_tokens"], result["text"]),
            (100, 80, "ok"),
        )

    def test_reasoning_effort_is_protocol_native(self):
        chat_req, _ = self.exercise(
            chat,
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ],
            reasoning_effort="xhigh",
        )
        self.assertEqual(chat_req["payload"]["reasoning_effort"], "xhigh")
        self.assertNotIn("reasoning", chat_req["payload"])

        resp_req, _ = self.exercise(
            responses,
            [
                'data: {"type":"response.output_text.delta","delta":"ok"}',
            ],
            reasoning_effort="xhigh",
        )
        self.assertEqual(resp_req["payload"]["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("reasoning_effort", resp_req["payload"])

        msg_req, _ = self.exercise(
            messages,
            [
                'data: {"type":"content_block_delta","delta":'
                '{"type":"text_delta","text":"ok"}}',
                'data: {"type":"message_stop"}',
            ],
            reasoning_effort="high",
        )
        self.assertEqual(msg_req["payload"]["output_config"], {"effort": "high"})

    def test_empty_reasoning_effort_is_omitted(self):
        request, _ = self.exercise(
            responses,
            [
                'data: {"type":"response.output_text.delta","delta":"ok"}',
            ],
        )
        self.assertNotIn("reasoning", request["payload"])
        self.assertNotIn("reasoning_effort", request["payload"])

    def test_chat_sends_appended_history(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        captured = {}
        response = FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        )
        response.test_case = self
        original = chat.post_stream

        def fake_post(url, api_key, payload, timeout, session_id=None):
            captured["payload"] = payload
            captured["session_id"] = session_id
            return response

        chat.post_stream = fake_post
        try:
            chat.stream(
                "https://protocol.example",
                "secret",
                "model",
                "ignored",
                "ignored",
                32,
                messages=history,
            )
        finally:
            chat.post_stream = original
        self.assertEqual(captured["payload"]["messages"], history)

    def test_chat_without_session_omits_cache_key(self):
        captured = {}
        response = FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        )
        response.test_case = self
        original = chat.post_stream

        def fake_post(url, api_key, payload, timeout, session_id=None):
            captured["payload"] = payload
            captured["session_id"] = session_id
            return response

        chat.post_stream = fake_post
        try:
            chat.stream(
                "https://protocol.example",
                "secret",
                "model",
                "system",
                "user",
                32,
                session_id="",
            )
        finally:
            chat.post_stream = original
        self.assertNotIn("prompt_cache_key", captured["payload"])
        self.assertEqual(captured["session_id"], "")


if __name__ == "__main__":
    unittest.main()
