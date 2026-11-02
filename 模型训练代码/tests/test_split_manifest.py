# -*- coding: utf-8 -*-
"""Verify that split_manifest hashes are explicit and not self-referential."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DATA = ROOT / "data_splits"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSplitManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = DATA / "split_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.sidecar_path = DATA / cls.manifest["manifest_hash_file"]

    def test_manifest_does_not_hash_itself(self):
        self.assertNotIn("split_manifest.json", self.manifest["hashes"])
        self.assertNotIn("split_manifest.json", self.manifest["hash_targets"])

    def test_hash_targets_match_hashes(self):
        self.assertEqual(set(self.manifest["hash_targets"]),
                         set(self.manifest["hashes"].keys()))

    def test_each_recorded_hash_matches_file(self):
        for name in self.manifest["hash_targets"]:
            path = DATA / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(self.manifest["hashes"][name], sha256(path), name)

    def test_sidecar_hashes_manifest(self):
        self.assertTrue(self.sidecar_path.exists())
        text = self.sidecar_path.read_text(encoding="utf-8").strip()
        side_hash, name = text.split()
        self.assertEqual(name, "split_manifest.json")
        self.assertEqual(side_hash, sha256(self.manifest_path))

    def test_split_sizes_and_no_overlap(self):
        test = json.loads((DATA / "final_test_seed42.json").read_text(encoding="utf-8"))
        dev = json.loads((DATA / "cases_200.json").read_text(encoding="utf-8"))
        allc = json.loads((DATA / "cases_250.json").read_text(encoding="utf-8"))
        self.assertEqual(len(test["case_ids"]), 50)
        self.assertEqual(len(dev["case_ids"]), 200)
        self.assertEqual(len(allc["case_ids"]), 250)
        self.assertFalse(set(test["case_ids"]) & set(dev["case_ids"]))


if __name__ == "__main__":
    unittest.main()
