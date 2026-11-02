# -*- coding: utf-8 -*-
"""Tests for Stage 2 multiclass calibration and ECE."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.stage2_calibration import (  # noqa: E402
    apply_calibrator, calibration_metrics, ece_binary, fit_calibrator,
    select_calibrator_crossfit,
)


def synthetic_probs(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    logits = rng.normal(0, 2.0, size=(n, 3))
    logits[np.arange(n), y] += 1.5
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return p.astype(float), y.astype(int)


class TestStage2Calibration(unittest.TestCase):
    def test_all_methods_return_probabilities(self):
        p, y = synthetic_probs()
        for method in ("none", "temperature", "multinomial_logistic"):
            cal = fit_calibrator(method, p, y)
            q = apply_calibrator(cal, p)
            self.assertEqual(q.shape, p.shape)
            self.assertTrue(np.all(np.isfinite(q)))
            np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-9)

    def test_perfect_calibration_has_zero_ece(self):
        y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        p = y.copy()
        self.assertAlmostEqual(ece_binary(y, p, n_bins=5)["ece"], 0.0)

    def test_ece_positive_for_miscalibrated_confidence(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        self.assertGreater(ece_binary(y, p, n_bins=5)["ece"], 0.1)

    def test_calibration_metrics_reports_class_auc(self):
        p, y = synthetic_probs()
        m = calibration_metrics(p, y)
        self.assertEqual(m["n"], len(y))
        for k in ("0", "1", "2"):
            self.assertIn("ece", m["classwise"][k])
            self.assertIn("auc", m["classwise"][k])

    def test_crossfit_selection_returns_valid_calibrator(self):
        p, y = synthetic_probs(n=900)
        fold = np.repeat(np.arange(3), 300)
        # Make the fold assignment deterministic but non-degenerate.
        rng = np.random.default_rng(5)
        rng.shuffle(fold)
        sel = select_calibrator_crossfit(p, y, fold)
        self.assertIn(sel["method"], ("none", "temperature",
                                      "multinomial_logistic"))
        self.assertIn("calibrator", sel)
        q = apply_calibrator(sel["calibrator"], p)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-9)

    def test_temperature_preserves_argmax(self):
        p, y = synthetic_probs()
        cal = fit_calibrator("temperature", p, y)
        q = apply_calibrator(cal, p)
        np.testing.assert_array_equal(np.argmax(p, axis=1),
                                      np.argmax(q, axis=1))


if __name__ == "__main__":
    unittest.main()
