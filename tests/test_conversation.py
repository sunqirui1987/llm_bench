"""通用连续对话：session 亲和、追加、裁剪。"""

from __future__ import annotations

import unittest

from llm_bench.conversation import Conversation
from llm_bench.prompts import (
    DEFAULT_SYSTEM,
    clip_text,
    compose_system,
    compose_user,
    pad_to_tokens,
    trim_messages,
)


class ConversationTest(unittest.TestCase):
    def test_cache_on_keeps_a_stable_session(self):
        first = Conversation(0, system="sys", user="hello", cache=True)
        second = Conversation(1, system="sys", user="hello", cache=True)
        self.assertTrue(first.session_id)
        self.assertTrue(second.session_id)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertNotEqual(first.messages[1]["content"], second.messages[1]["content"])

    def test_cache_off_has_no_session(self):
        conv = Conversation(0, system="sys", user="hello", cache=False)
        self.assertEqual(conv.session_id, "")

    def test_miss_outbound_busts_prefix_every_call(self):
        conv = Conversation(0, system="sys", user="hello", cache=False)
        first = conv.outbound()
        second = conv.outbound()
        self.assertTrue(first[0]["content"].startswith("CACHE_BYPASS"))
        self.assertTrue(second[0]["content"].startswith("CACHE_BYPASS"))
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertEqual(conv.messages[0]["content"], "sys")

    def test_hit_outbound_keeps_stable_system_prefix(self):
        conv = Conversation(0, system="sys", user="hello", cache=True)
        first = conv.outbound()
        conv.commit("ok")
        second = conv.outbound()
        self.assertEqual(first[0]["content"], "sys")
        self.assertEqual(second[0]["content"], "sys")
        self.assertFalse(first[0]["content"].startswith("CACHE_BYPASS"))

    def test_commit_appends_assistant_and_next_user(self):
        conv = Conversation(0, system="sys", user="hello", cache=True)
        self.assertEqual([item["role"] for item in conv.messages], ["system", "user"])
        conv.commit("first answer")
        self.assertEqual(
            [item["role"] for item in conv.messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("CONTINUE_TURN 1", conv.messages[-1]["content"])
        self.assertIn("Continue from where you left off", conv.messages[-1]["content"])
        self.assertEqual(conv.turn, 1)
        self.assertEqual(conv.messages[2]["content"], "first answer")
        outbound = conv.outbound()
        self.assertEqual(len(outbound), 4)
        outbound[0]["content"] = "mutated"
        self.assertEqual(conv.messages[0]["content"], "sys")

    def test_trim_keeps_system_and_latest_user(self):
        messages = [{"role": "system", "content": "S"}]
        for index in range(20):
            messages.append({"role": "user", "content": ("u" * 4000) + str(index)})
            messages.append({"role": "assistant", "content": ("a" * 4000) + str(index)})
        messages.append({"role": "user", "content": "latest"})
        trim_messages(messages, budget=200)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["content"], "latest")
        self.assertLessEqual(len(messages), 5)

    def test_clip_text_keeps_head_and_tail(self):
        text = "HEAD" + ("x" * 400) + "TAIL"
        clipped = clip_text(text, 20)
        self.assertTrue(clipped.startswith("HEAD"))
        self.assertTrue(clipped.endswith("TAIL"))
        self.assertIn("truncated", clipped)

    def test_pad_to_tokens_is_stable_for_same_salt(self):
        first = pad_to_tokens("BASE", 200, salt="same")
        second = pad_to_tokens("BASE", 200, salt="same")
        other = pad_to_tokens("BASE", 200, salt="other")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertTrue(first.startswith("BASE"))
        self.assertIn("CONTEXT PADDING", first)

    def test_custom_followup_is_used_on_commit(self):
        conv = Conversation(
            0,
            system="sys",
            user="hello",
            cache=False,
            followup="请继续写下一章，不要重复。",
        )
        conv.commit("ok")
        self.assertIn("请继续写下一章", conv.messages[-1]["content"])

    def test_compose_system_short_is_plain_default(self):
        text = compose_system(kind="short")
        self.assertEqual(text, DEFAULT_SYSTEM)
        self.assertNotIn("CONTEXT PADDING", text)
        self.assertNotIn("宿命旅途", text)

    def test_compose_system_long_pads_without_builtin_corpus(self):
        text = compose_system(kind="long", input_tokens=400)
        self.assertIn("CONTEXT PADDING", text)
        self.assertNotIn("宿命旅途", text)

    def test_compose_system_uses_custom_text_and_context_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            context = Path(folder) / "ctx.txt"
            system_file = Path(folder) / "sys.txt"
            context.write_text("CONTEXT_BODY\n", encoding="utf-8")
            system_file.write_text("FILE_SYS", encoding="utf-8")
            text = compose_system(
                kind="short",
                text="INLINE_SYS",
                file=str(system_file),
                context_file=str(context),
            )
        self.assertTrue(text.startswith("CONTEXT_BODY"))
        self.assertIn("FILE_SYS", text)
        self.assertIn("INLINE_SYS", text)

    def test_missing_context_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            compose_system(kind="long", context_file="/no/such/context.md")

    def test_compose_user_prefers_file_then_prompt(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "user.txt"
            path.write_text("FROM_FILE", encoding="utf-8")
            self.assertEqual(compose_user("FROM_ARG", str(path)), "FROM_FILE\nFROM_ARG")
            self.assertEqual(compose_user("", str(path)), "FROM_FILE")


if __name__ == "__main__":
    unittest.main()
