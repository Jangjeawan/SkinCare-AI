import json
import unittest
from pathlib import Path

from guide.retrieve import select
from qa.bank_loader import load_bank
from qa.orchestrator import handle_turn
from qa.rules_engine import evaluate, load_rules

ROOT = Path(__file__).resolve().parents[1]


class GoldenRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules()
        cls.cases = json.loads((ROOT / "tests" / "rules" / "golden.json").read_text(encoding="utf-8"))

    def test_catalog_has_fifteen_scored_rules(self):
        self.assertEqual(len(self.rules["rules"]), 15)
        self.assertEqual(self.rules["version"], "2026-10-18")

    def test_golden_vectors(self):
        self.assertEqual(len(self.cases), 15)
        for case in self.cases:
            with self.subTest(case=case["name"]):
                decision = evaluate(case["slots"], self.rules, case.get("vision"))
                if "forbid_rule" in case:
                    self.assertNotIn(case["forbid_rule"], decision["rule_ids"])
                    continue
                self.assertEqual(decision["level"], case["level"])
                if case["level"] is None:
                    self.assertCountEqual(decision["missing_slots"], case["missing"])
                else:
                    self.assertIn(case["rule"], decision["rule_ids"])


class QuestionTurnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_bank()
        cls.rules = load_rules()

    def test_chemical_phrase_finishes_without_llm(self):
        result = handle_turn(
            {"slots": {"scenario": "burn"}, "user_text": "화학물질이 묻었어요"},
            self.bank,
            self.rules,
            llm=lambda _payload: (_ for _ in ()).throw(AssertionError("llm")),
        )
        self.assertEqual(result["action"], "finish")
        self.assertEqual(result["level"], "EMERGENCY")
        self.assertIn("burn.chemical", result["rule_ids"])
        self.assertFalse(result["llm_called"])

    def test_bleeding_phrase_finishes(self):
        result = handle_turn(
            {"slots": {"scenario": "open_wound"}, "user_text": "피가 안 멈춰요"},
            self.bank,
            self.rules,
        )
        self.assertEqual(result["level"], "EMERGENCY")
        self.assertIn("bleed.ongoing", result["rule_ids"])
        self.assertFalse(result["llm_called"])

    def test_blank_text_asks_elapsed_without_llm(self):
        result = handle_turn({"slots": {"systemic": "none"}, "user_text": "  "}, self.bank, self.rules)
        self.assertEqual(result["question_id"], "common.elapsed")
        self.assertFalse(result["llm_called"])
        self.assertFalse(result["overrode_llm"])

    def test_bank_outside_id_falls_back(self):
        def llm(_payload):
            return {"action": "ask", "question_id": "made.up", "utterance": "어디가 아프신가요?", "slots_delta": {}}

        result = handle_turn({"slots": {"systemic": "none"}, "user_text": "긁혔어요"}, self.bank, self.rules, llm)
        self.assertTrue(result["overrode_llm"])
        self.assertEqual(result["question_id"], "common.elapsed")
        self.assertTrue(result["llm_called"])

    def test_drug_utterance_is_discarded(self):
        def llm(_payload):
            return {"action": "ask", "question_id": "common.elapsed", "utterance": "이 연고를 바르셨나요?", "slots_delta": {}}

        result = handle_turn({"slots": {"systemic": "none"}, "user_text": "긁혔어요"}, self.bank, self.rules, llm)
        self.assertTrue(result["overrode_llm"])
        self.assertEqual(result["utterance"], "언제부터인가요?")

    def test_tenth_turn_fills_unknown_and_finishes(self):
        result = handle_turn({"slots": {"scenario": "abrasion"}, "turn": 10, "user_text": ""}, self.bank, self.rules)
        self.assertEqual(result["action"], "finish")
        self.assertEqual(result["slots"]["elapsed"], "unknown")
        self.assertEqual(result["slots"]["site"], "unknown")


class GuideSelectTest(unittest.TestCase):
    def test_null_level_returns_no_cards(self):
        picked = select(None, {}, [], {"cards": [{"id": "x", "body": "원문", "source_url": "https://example.test", "when": {}}]})
        self.assertEqual(picked["reason"], "no_level")
        self.assertEqual(picked["cards"], [])

    def test_unmatched_condition_is_no_card(self):
        cards = {"cards": [{"id": "card.bleed", "body": "원문입니다.", "source_url": "https://example.test", "when": {"rule": "bleed.ongoing"}}]}
        picked = select("WARNING", {"scenario": "abrasion"}, [], cards, labels={}, banned=[])
        self.assertEqual(picked["reason"], "no_card")

    def test_replace_and_redact(self):
        cards = {
            "cards": [
                {
                    "id": "card.burn.heat",
                    "title": "열",
                    "body": "흐르는 물로 식히십시오. 시중 연고를 바릅니다.",
                    "source": "예시",
                    "source_url": "https://example.test/heat",
                    "when": {"scenario": "burn"},
                    "avoid": ["toothpaste"],
                },
                {
                    "id": "card.burn.chemical",
                    "title": "화학",
                    "body": "화학물질 원문입니다.",
                    "source": "예시",
                    "source_url": "https://example.test/chem",
                    "when": {"cause": "chemical"},
                    "replaces": ["card.burn.heat"],
                    "avoid": [],
                },
            ]
        }
        picked = select(
            "EMERGENCY",
            {"scenario": "burn", "cause": "chemical"},
            ["burn.chemical"],
            cards,
            labels={"toothpaste": "치약을 바르지 마세요"},
            banned=["연고"],
        )
        self.assertEqual([card["id"] for card in picked["cards"]], ["card.burn.chemical"])
        self.assertEqual(picked["redactions"], 0)
        self.assertIsNone(picked["reason"])


if __name__ == "__main__":
    unittest.main()
