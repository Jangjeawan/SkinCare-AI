import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from guide.retrieve import load_cards, select
from app.middleware import access_record
from jobs import queue_store
from jobs.feedback_store import enqueue as enqueue_feedback
from jobs.retrain_worker import review
from qa.llm_client import build_messages, parse_content
from scripts.deploy_model import approve
from tracking.compare import compare_images


class CardSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = load_cards()

    def test_official_cards_are_active(self):
        self.assertEqual(len(self.cards["cards"]), 8)
        self.assertTrue(all(card["source_url"].startswith("https://") for card in self.cards["cards"]))

    def test_heat_card_drops_ointment_sentence(self):
        picked = select("WARNING", {"scenario": "burn", "cause": "heat_liquid"}, [], self.cards)
        self.assertEqual(picked["cards"][0]["id"], "card.burn.heat")
        self.assertIn("흐르는 찬물", picked["cards"][0]["body"])
        self.assertNotIn("연고", picked["cards"][0]["body"])
        self.assertGreaterEqual(picked["redactions"], 1)

    def test_chemical_card_replaces_heat(self):
        picked = select("EMERGENCY", {"scenario": "burn", "cause": "chemical"}, ["burn.chemical"], self.cards)
        self.assertEqual([card["id"] for card in picked["cards"]], ["card.burn.chemical"])

    def test_unknown_has_no_self_care_steps(self):
        picked = select("UNKNOWN", {}, [], self.cards)
        self.assertEqual(picked["cards"][0]["id"], "card.unknown")
        self.assertNotIn("세척", picked["cards"][0]["body"])


class TrackingCompareTest(unittest.TestCase):
    def test_same_area_is_similar_and_reports_color_delta(self):
        before = np.full((40, 40, 3), 40, dtype=np.uint8)
        after = np.full((40, 40, 3), 180, dtype=np.uint8)

        def detect(_image):
            return [((0.1, 0.1, 0.4, 0.5), 0.9)]

        result = compare_images(before, after, detect)
        self.assertEqual(result["trend_hint"], "similar")
        self.assertEqual(result["area_ratio_delta"], 0.0)
        self.assertGreater(result["color_delta"]["value"], 0)
        self.assertIn("참고값", result["note"])

    def test_missing_box_is_uncomparable(self):
        image = np.full((40, 40, 3), 80, dtype=np.uint8)
        result = compare_images(image, image, lambda _image: [])
        self.assertEqual(result["trend_hint"], "uncomparable")
        self.assertIsNone(result["area_ratio_delta"])


class LlmContractTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_limits_keys(self):
        messages = build_messages({"question_ids": ["common.elapsed"]})
        self.assertIn("question_id", messages[0]["content"])
        self.assertIn("진단", messages[0]["content"])

    def test_parse_json_object(self):
        parsed = parse_content('{"action":"ask","question_id":"common.elapsed","utterance":"언제부터인가요?","slots_delta":{}}')
        self.assertEqual(parsed["question_id"], "common.elapsed")


class FeedbackAndLogTest(unittest.TestCase):
    def test_feedback_file_has_queue_id_and_log_has_no_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue_id = enqueue_feedback(
                {"request_id": "req-1", "kind": "DISAGREE", "note": "문진 원문", "photo_ref": "photo-1"},
                Path(tmp),
            )
            saved = json.loads((Path(tmp) / f"{queue_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["queue_id"], queue_id)
        record = access_record("req-1", "POST", "/api/v1/ai/feedback", 200, 3, 12)
        self.assertNotIn("note", record)
        self.assertNotIn("user_text", record)
        self.assertEqual(record["text_chars"], 12)


class RetrainGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = queue_store.DIRECTORY
        queue_store.DIRECTORY = Path(self.tmp.name) / "jobs"

    def tearDown(self):
        queue_store.DIRECTORY = self.old
        self.tmp.cleanup()

    def test_low_score_fails_without_deploy(self):
        metrics = Path(self.tmp.name) / "metrics.json"
        metrics.write_text(json.dumps({"macro_f1": 0.2}), encoding="utf-8")
        job = queue_store.enqueue("manual")
        result = review(job["job_id"], metrics)
        self.assertEqual(result["status"], "failed")
        self.assertNotEqual(queue_store.get(job["job_id"])["status"], "deployed")

    def test_approve_copies_only_waiting_job(self):
        metrics = Path(self.tmp.name) / "metrics.json"
        metrics.write_text(json.dumps({"macro_f1": 0.93}), encoding="utf-8")
        job = queue_store.enqueue("manual")
        self.assertEqual(review(job["job_id"], metrics)["status"], "awaiting_deploy")
        self.assertFalse(queue_store.get(job["job_id"])["trained"])
        root = Path(self.tmp.name)
        classify_src = root / "classify.pt"
        detect_src = root / "detect.pt"
        classify_src.write_bytes(b"new-classify")
        detect_src.write_bytes(b"new-detect")
        serving = root / "serving"
        serving.mkdir()
        (serving / "classify.pt").write_bytes(b"old-classify")
        with self.assertRaises(RuntimeError):
            approve("missing", classify_src, detect_src, serving, root / "archive")
        version = approve(job["job_id"], classify_src, detect_src, serving, root / "archive")
        self.assertEqual((serving / "classify.pt").read_bytes(), b"new-classify")
        self.assertTrue(any((root / "archive").glob("*-classify.pt")))
        self.assertEqual(queue_store.get(job["job_id"])["status"], "deployed")
        self.assertIn(job["job_id"][:8], version["classify"])


if __name__ == "__main__":
    unittest.main()
