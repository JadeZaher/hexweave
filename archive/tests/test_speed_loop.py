"""Keep timing comparisons separate from numerical equivalence."""

import copy
import unittest

from scripts.speed_loop import compare, deterministic


class SpeedLoopTests(unittest.TestCase):
    def test_only_timing_and_execution_description_are_removed(self):
        self.assertEqual(deterministic({"seconds": 2, "loss": 3}), {"loss": 3})
        self.assertEqual(deterministic({"arithmetic": {"execution": "scalar", "threshold": 1}}),
                         {"arithmetic": {"threshold": 1}})

    def test_any_numerical_change_rejects_speedup(self):
        old = {"cases": [{"mode": "pulse", "seed": 7, "median_seconds": 2.0,
                          "result": {"loss": 1.0, "updates": 3}}]}
        new = copy.deepcopy(old)
        new["cases"][0]["median_seconds"] = 1.0
        self.assertEqual(compare(new, old)[0]["speedup"], 2.0)
        for key in ("loss", "updates"):
            changed = copy.deepcopy(new)
            changed["cases"][0]["result"][key] += 1
            with self.assertRaises(ValueError):
                compare(changed, old)

    def test_missing_case_rejects_comparison(self):
        with self.assertRaises(ValueError):
            compare({"cases": []}, {"cases": [{}]})
