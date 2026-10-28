import unittest

import numpy as np

from vision.preprocess import expand_xywh
from vision.gate import GateConfig, apply_gate
from vision.quality import QualityConfig, check_image


def _photo(value: int = 120, size: int = 320, noise: int = 30) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.full((size, size, 3), value, dtype=np.int16)
    base = np.clip(base + rng.integers(-noise, noise + 1, base.shape), 0, 255)
    return base.astype(np.uint8)


class QualityTest(unittest.TestCase):
    def test_sharp_photo_passes(self):
        result = check_image(_photo())
        self.assertTrue(result["ok"])
        self.assertEqual(result["reasons"], [])

    def test_small_dark_flat_image_is_rejected(self):
        result = check_image(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertFalse(result["ok"])
        self.assertIn("too_small", result["reasons"])
        self.assertIn("too_dark", result["reasons"])
        self.assertIn("flat", result["reasons"])

    def test_blur_reason(self):
        result = check_image(_photo(noise=0), QualityConfig(blur_var=1e9))
        self.assertIn("blur", result["reasons"])


class GateTest(unittest.TestCase):
    def test_low_top1_is_unknown(self):
        probs = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
        decision = apply_gate(probs, 0.8, 0.1, energy=1.0, energy_max=5.0)
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertIn("low_conf", decision["gate_fail"])

    def test_small_margin_is_unknown(self):
        probs = np.array([0.10, 0.10, 0.60, 0.10, 0.10])
        # burn 0.60, next is 0.10, margin 0.50 — need top1 0.60 and top2 0.50
        probs = np.array([0.50, 0.0, 0.50, 0.0, 0.0])
        decision = apply_gate(probs, 0.8, 0.1, energy=1.0, energy_max=5.0, cfg=GateConfig(top1=0.55))
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertIn("low_margin", decision["gate_fail"])

    def test_pass_returns_class(self):
        probs = np.array([0.02, 0.02, 0.90, 0.04, 0.02])
        decision = apply_gate(probs, 0.8, 0.1, energy=1.0, energy_max=5.0)
        self.assertEqual(decision["label"], "burn")
        self.assertEqual(decision["display"], "화상 의심")
        self.assertEqual(decision["gate_fail"], [])

    def test_missing_energy_file_blocks_label(self):
        probs = np.array([0.02, 0.02, 0.90, 0.04, 0.02])
        decision = apply_gate(probs, 0.8, 0.1, energy=1.0, energy_max=None)
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertIn("ood", decision["gate_fail"])

    def test_small_box_is_no_roi(self):
        probs = np.array([0.02, 0.02, 0.90, 0.04, 0.02])
        decision = apply_gate(probs, 0.9, 0.01, energy=1.0, energy_max=5.0)
        self.assertIn("no_roi", decision["gate_fail"])


class CropTest(unittest.TestCase):
    def test_expand_stays_inside_frame(self):
        x, y, w, h = expand_xywh((0.0, 0.0, 0.2, 0.2), pad=0.15)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 1)
        self.assertLessEqual(y + h, 1)


if __name__ == "__main__":
    unittest.main()
