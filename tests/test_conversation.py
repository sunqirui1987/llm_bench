"""同一条命令再发 vs 每次都换新命令。"""

from __future__ import annotations

import unittest

from llm_bench.conversation import Conversation
from llm_bench.prompts import (
    DEFAULT_SYSTEM,
    MISS_SCENES,
    clip_text,
    compose_system,
    compose_user,
    pad_to_tokens,
    plan_request,
    trim_messages,
)


class ConversationTest(unittest.TestCase):
    def test_cache_on_keeps_a_stable_session(self):
        first = Conversation(0, system="sys", user="hello", cache=True)
        second = Conversation(1, system="sys", user="hello", cache=True)
        self.assertTrue(first.session_id)
        self.assertTrue(second.session_id)
        self.assertNotEqual(first.session_id, second.session_id)

    def test_hit_outbound_is_byte_identical_every_time(self):
        conv = Conversation(0, system="sys", user="hello", cache=True)
        first = conv.outbound()
        second = conv.outbound()
        self.assertEqual(first, second)
        self.assertEqual(first[0]["content"], "sys")
        self.assertEqual(first[1]["content"], "hello")
        self.assertFalse(first[0]["content"].startswith("CACHE_BYPASS"))

    def test_cache_off_has_no_session(self):
        conv = Conversation(0, system="sys", user="hello", cache=False)
        self.assertEqual(conv.session_id, "")

    def test_miss_outbound_is_a_new_command_every_time(self):
        conv = Conversation(0, system="sys", user="hello", cache=False)
        first = conv.outbound()
        second = conv.outbound()
        self.assertTrue(first[0]["content"].startswith("CACHE_BYPASS"))
        self.assertTrue(second[0]["content"].startswith("CACHE_BYPASS"))
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertNotEqual(first[1]["content"], second[1]["content"])
        self.assertEqual(len(first), 2)

    def test_hit_does_not_put_worker_id_into_the_command(self):
        a = Conversation(0, system="sys", user="", cache=True).outbound()
        b = Conversation(1, system="sys", user="", cache=True).outbound()
        self.assertEqual(a[1]["content"], b[1]["content"])
        self.assertEqual(a[0]["content"], b[0]["content"])

    def test_miss_scenes_cover_different_game_beats(self):
        self.assertGreaterEqual(len(set(MISS_SCENES)), 5)
        seen = set()
        conv = Conversation(0, system="sys", user="", cache=False)
        for _ in range(8):
            seen.add(conv.outbound()[1]["content"].splitlines()[2])
        self.assertGreater(len(seen), 1)

    def test_followup_is_appended_to_the_same_hit_command(self):
        conv = Conversation(
            0,
            system="sys",
            user="hello",
            cache=True,
            followup="请写到首通结算。",
        )
        user = conv.outbound()[1]["content"]
        self.assertIn("请写到首通结算。", user)
        self.assertEqual(user, conv.outbound()[1]["content"])

    def test_output_budget_fits_the_window(self):
        conv = Conversation(
            0,
            system="sys",
            user="hello",
            cache=True,
            max_input=4000,
            max_tokens=500000,
            context_window=8000,
        )
        self.assertGreater(conv.output_tokens_for(), 0)
        self.assertLessEqual(
            conv.input_tokens_for() + conv.output_tokens_for(),
            8000,
        )

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
        first = pad_to_tokens("BASE", 8000, salt="same")
        second = pad_to_tokens("BASE", 8000, salt="same")
        other = pad_to_tokens("BASE", 8000, salt="other")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertIn("BASE", first)
        self.assertIn("CONTEXT PADDING", first)

    def test_pad_blocks_are_destiny_journey_scenes(self):
        text = pad_to_tokens("BASE", 8000, salt="mix", domain="卡琳@雷鸣区")
        self.assertIn("主线坐标", text)
        self.assertIn("终焉神殿", text)
        self.assertIn("首通结算", text)
        self.assertIn("宿命旅途", text)

    def test_plan_request_fits_window(self):
        plan = plan_request(max_input=1000, max_tokens=200, context_window=2000)
        self.assertEqual(plan["input_tokens"], 1000)
        self.assertGreater(plan["max_tokens"], 0)

    def test_compose_system_short_is_plain_default(self):
        text = compose_system(kind="short")
        self.assertEqual(text, DEFAULT_SYSTEM)
        self.assertNotIn("CONTEXT PADDING", text)
        self.assertIn("宿命旅途", text)

    def test_compose_system_long_pads_with_game_bible(self):
        text = compose_system(kind="long", input_tokens=8000)
        self.assertIn("CONTEXT PADDING", text)
        self.assertIn("宿命旅途", text)

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
