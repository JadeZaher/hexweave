"""T2 diagnostic accounting and frozen-data rejection; no model execution."""

import copy
import unittest
from pathlib import Path

from scripts import text_campaign, text_credit_io, text_profile


def metrics(active=False):
    raw = dict.fromkeys(text_credit_io.COUNTS + text_credit_io.SUMS, 0)
    if active:
        raw.update(frames=1, unique_tile_commits=1, coordinate_opportunities=512)
    return {"raw": raw, "ratios": {"opposite_over_both_nonzero": None,
            "different_over_coordinates": 0.0 if active else None,
            "deficit_over_task_descent": None, "task_route_cosine": None}}


def fixture():
    return {"counts": {"anchor_targets": 1, "returned_frames": 1}, "tile_transitions": {"pulse_context": 0},
            "credit_diagnostics": {"schema": 1, "scope": "frozen anchor unique-tile commits only", "coordinates_per_commit": 512,
                "anchors": 1, "anchors_with_different_results": 0, "anchor_difference_fraction": 0.0,
                "total": metrics(True), "per_tile": [{"tile": index, "metrics": metrics(index == 0)} for index in range(7)]}}


def frozen_data():
    return {"train_bytes": 671, "train_identity": "62ac6e481ad030b9", "valid_bytes": 278,
            "valid_identity": "dcae5afcc74a9d72", "evaluation_examples": 64,
            "evaluation_prefix": {"offset": 0, "byte_count": 72, "identity_fnv1a64": text_profile.T2_PREFIX_FNV}}


class TextCreditTests(unittest.TestCase):
    def test_credit_receipt_fails_closed_on_missing_fields_and_denominators(self):
        options = {"credit_diagnostics": "on", "route_credit": "coupled"}
        text_credit_io.check_credit(fixture(), options)
        raw_faults = {
            "partition": {"different_results": 1, "different_commits": 1},
            "zero-task": {"task_descent": 1, "deficit": 1},
            "zero-task-norm": {"task_nonzero": 1},
            "zero-route-norm": {"route_nonzero": 1},
            "zero-dot": {"task_route_dot": 1},
            "zero-projection": {"coupled_projection": 1, "deficit": 1},
            "zero-transitions": {"task_nonzero": 1, "task_norm_squared": 1, "task_descent": 1, "deficit": 1},
            "opposing-without-coupled-transition": {"task_nonzero": 1, "task_norm_squared": 1,
                "task_transitions": 1, "task_descent": 1, "deficit": 1,
                "different_results": 1, "different_commits": 1, "task_opposing": 1},
            "different-without-transition": {"different_results": 1, "different_commits": 1, "new_where_task_zero": 1}}
        faults = ("missing", "coordinates", "signs", "applied", "ratio", "tiles", "aggregate", "nonfinite", "off")
        for fault in faults + tuple(raw_faults):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                receipt = fixture()
                diagnostic = receipt["credit_diagnostics"]
                raw = diagnostic["total"]["raw"]
                if fault == "missing":
                    raw.pop("deficit")
                elif fault == "coordinates":
                    raw["coordinate_opportunities"] = 511
                elif fault == "signs":
                    raw["both_nonzero"] = 1
                elif fault == "applied":
                    raw["applied_transitions"] = 1
                elif fault == "ratio":
                    diagnostic["total"]["ratios"]["deficit_over_task_descent"] = 0
                elif fault == "tiles":
                    diagnostic["per_tile"].pop()
                elif fault == "aggregate":
                    diagnostic["per_tile"][0]["metrics"]["raw"]["route_nonzero"] = 1
                elif fault == "nonfinite":
                    raw["residual_max"] = float("inf")
                if fault in raw_faults:
                    raw.update(raw_faults[fault])
                    text_credit_io.check_raw(raw, "coupled")
                else:
                    text_credit_io.check_credit(receipt, options | ({"credit_diagnostics": "off"} if fault == "off" else {}))

    def test_t2_profile_cannot_accept_old_validation_identity(self):
        options = {"profile": "t2-credit", "credit_diagnostics": "on", "mode": "pulse"}
        data = frozen_data()
        text_profile.check_profile_receipt({"data": data}, options)
        for change in ({"valid_identity": "old-valid"}, {"train_identity": "other-train"},
                       {"evaluation_prefix": {"offset": 0, "byte_count": 71, "identity_fnv1a64": text_profile.T2_PREFIX_FNV}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                text_profile.check_profile_receipt({"data": data | change}, options)

    def test_campaign_rejects_old_protocol_data_and_missing_diagnostics(self):
        options = {"seed": 7, "route_credit": "coupled", "steps": 256, "anchor_interval": 4,
                   "mode": "pulse", "profile": "t2-credit", "credit_diagnostics": "on"}
        hashes = {"binary": "0" * 64, "train": text_profile.T2_TRAIN_SHA256, "valid": text_profile.T2_DEVELOPMENT_SHA256}
        owner = {"status": "completed", "options": options, "input_sha256": hashes, "input_sha256_after": hashes,
                 "evaluation_input": {"prefix_sha256": text_profile.T2_PREFIX_SHA256},
                 "child": {"returncode": 0, "reaped": True, "terminal_peak_memory_queried": True,
                           "memory": {"peak_working_set_bytes": 1024}, "wall_seconds": 1.0},
                 "result": {"protocol": "text-pulse-v4", "options": options, "data": frozen_data()}}
        for fault in ("protocol", "data", "missing-diagnostics"):
            current = copy.deepcopy(owner)
            if fault == "protocol":
                current["result"]["protocol"] = "text-pulse-v3"
            elif fault == "data":
                current["input_sha256"]["valid"] = "1" * 64
                current["input_sha256_after"]["valid"] = "1" * 64
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                text_campaign.measurement(Path("unused"), current, 7, "coupled")


if __name__ == "__main__":
    unittest.main()
