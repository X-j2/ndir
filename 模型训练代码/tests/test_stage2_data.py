# -*- coding: utf-8 -*-
"""Tests for Stage 2 dataset masks and case-level folds."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from training.stage2_data import (  # noqa: E402
    CACHE_SCHEMA_VERSION, LABEL_CODE, STATE_CODE, _build_case,
    compute_dataset_fingerprint, event_eval_mask, load_dev_ids,
    load_final_test_ids, training_mask, validate_cache_meta,
)
from training.run_stage2 import case_folds  # noqa: E402


def synthetic_meta(n=20):
    ids = list(range(1, n + 1))
    return {"case_meta": {
        str(c): {"reliable_u2_events": [{"start": 0, "end": 10}] * (c % 4),
                 "low_support_u2_events": []} for c in ids}}, ids


def synthetic_data(n=8):
    labels = [LABEL_CODE[x] for x in
              ("U0", "U1", "U2", "U3", "U0", "U2", "U1", "U0")]
    states = [STATE_CODE[x] for x in
              ("VALID", "VALID", "VALID", "FLAT_WITH_HISTORY",
               "FLAT_NO_HISTORY", "VALID", "QUANTILE_ARTIFACT", "VALID")]
    return {
        "label": np.asarray(labels, dtype=np.int8),
        "state": np.asarray(states, dtype=np.int8),
        "ref_valid_frac": np.asarray([1, 1, 1, 1, 1, .5, 1, 1],
                                     dtype=np.float32),
    }


class TestStage2Data(unittest.TestCase):
    def test_dev_ids_are_200_and_disjoint(self):
        ids = load_dev_ids()
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)

    def test_training_mask_is_valid_u0_u1_u2_only(self):
        m = training_mask(synthetic_data())
        np.testing.assert_array_equal(
            m, [True, True, True, False, False, True, False, True])

    def test_event_eval_mask_requires_reference_and_known_state(self):
        m = event_eval_mask(synthetic_data())
        # Row 4 is FLAT_NO_HISTORY (UNKNOWN); row 5 is VALID but ref<0.9;
        # row 6 is QUANTILE_ARTIFACT (UNKNOWN).
        np.testing.assert_array_equal(
            m, [True, True, True, True, False, False, False, True])

    def test_case_folds_are_case_level_and_disjoint(self):
        meta, ids = synthetic_meta()
        folds = case_folds(meta, ids, 5)
        test_cases = [c for _, te in folds for c in te]
        self.assertEqual(sorted(test_cases), sorted(ids))
        seen = set()
        for tr, te in folds:
            self.assertFalse(set(tr) & set(te))
            self.assertFalse(seen & set(te))
            seen |= set(te)

    def test_case_folds_are_deterministic(self):
        meta, ids = synthetic_meta()
        a = case_folds(meta, ids, 5)
        b = case_folds(meta, ids, 5)
        self.assertEqual(a, b)

    def test_no_window_cases_keep_truth_events(self):
        for cid in (7, 106):
            p = _build_case((cid, 15.0, 7.5))
            self.assertEqual(p["rows"], [], cid)
            self.assertGreaterEqual(len(p["meta"]["reliable_u2_events"]), 1)
        p106 = _build_case((106, 15.0, 7.5))
        self.assertGreaterEqual(len(p106["meta"]["low_support_u2_events"]), 1)

    def test_frozen_dev_has_303_and_60_events(self):
        import json
        with open(ROOT / "data_splits" / "cases_200.json",
                  encoding="utf-8") as f:
            dev = set(map(str, json.load(f)["case_ids"]))
        with open(ROOT / "data_splits" / "case_event_counts.json",
                  encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        rel = sum(cases[c]["u2_events"] for c in dev)
        low = sum(cases[c]["u2_low_support_events"] for c in dev)
        self.assertEqual((rel, low), (303, 60))

    def test_cache_validation_rejects_fingerprint_and_split_mismatch(self):
        dev = load_dev_ids()
        final = load_final_test_ids()
        fp = compute_dataset_fingerprint(15.0, 7.5, dev, final)
        meta = {
            "version": CACHE_SCHEMA_VERSION,
            "window_sec": 15.0, "step_sec": 7.5,
            "dataset_fingerprint": fp,
            "dev_case_ids": dev,
        }
        data = {"case_id": np.asarray(dev[:2], dtype=np.int32)}
        self.assertEqual(validate_cache_meta(
            meta, data, fp, dev, final, 15.0, 7.5), [])
        bad = dict(meta, dataset_fingerprint="bad")
        self.assertTrue(validate_cache_meta(
            bad, data, fp, dev, final, 15.0, 7.5))
        bad = dict(meta, dev_case_ids=dev[:-1])
        self.assertTrue(validate_cache_meta(
            bad, data, fp, dev, final, 15.0, 7.5))
        bad = dict(meta, dev_case_ids=dev[:-1] + [final[0]])
        errors = validate_cache_meta(
            bad, data, fp, dev, final, 15.0, 7.5)
        self.assertTrue(any("overlap" in e.lower() for e in errors))

    def test_dataset_fingerprint_is_deterministic_and_window_specific(self):
        dev = load_dev_ids()
        final = load_final_test_ids()
        a = compute_dataset_fingerprint(15.0, 7.5, dev, final)
        b = compute_dataset_fingerprint(15.0, 7.5, dev, final)
        c = compute_dataset_fingerprint(30.0, 15.0, dev, final)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 64)


if __name__ == "__main__":
    unittest.main()
