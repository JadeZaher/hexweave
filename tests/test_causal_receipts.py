"""Causal receipt identities, coverage and cumulative-accounting faults."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import causal_contract, causal_result
from scripts.causal_observer import load, write


def development():
    zero = {"targets": 0, "nll_sum": 0.0}
    return {"dataset": {"split": "development", "manifest_sha256": "a" * 64},
            "before_sha256": "b" * 64, "after_sha256": "b" * 64,
            "teacher_forced": {"total": {"targets": 1, "nll_sum": 2.0},
                               "answer": {"targets": 1, "nll_sum": 2.0}, "citation": zero.copy(), "format": zero.copy()},
            "teacher_coverage": {"kind": "limit"}, "generation_coverage": {"kind": "complete"},
            "cases": [{"generation": {"raw_bytes": [255, 10]}}]}


class CausalReceiptTests(unittest.TestCase):
    def test_evaluation_mutation_mask_and_byte_faults(self):
        limits = {"evaluation_targets": 8, "generation_cases": 1, "generation_bytes": 32}
        causal_result.evaluation(development(), limits, "a" * 64, True)
        for fault in ("state", "split", "manifest", "counts", "loss", "nonfinite", "bytes", "coverage", "missing"):
            with self.subTest(fault=fault):
                value = development()
                if fault == "state":
                    value["after_sha256"] = "c" * 64
                elif fault in ("split", "manifest"):
                    value["dataset"]["split" if fault == "split" else "manifest_sha256"] = "incorrect"
                elif fault == "counts":
                    value["teacher_forced"]["answer"]["targets"] = 2
                elif fault == "loss":
                    value["teacher_forced"]["answer"]["nll_sum"] = 3
                elif fault == "nonfinite":
                    value["teacher_forced"]["answer"]["nll_sum"] = float("nan")
                elif fault == "bytes":
                    value["cases"][0]["generation"]["raw_bytes"] = [True]
                elif fault == "coverage":
                    value["teacher_coverage"]["kind"] = "interrupted"
                else:
                    value = None
                with self.assertRaises(ValueError):
                    causal_result.evaluation(value, limits, "a" * 64, True)

    def test_owner_counter_reset_identity_and_terminal_mismatch(self):
        initial = {"schema": 1, "run_id": "r", "attempt_id": "b", "parent": {"attempt_id": "a"},
                   "budget": {"maximum_opportunities": 100}, "attempt_start_opportunity": 4,
                   "attempt_start_wall_millis": 1000, "consumed": {"opportunities": 4, "wall_millis": 1100, "attempts": 2},
                   "observer_sequence": 8, "stop": "running", "physical_work_unknown": False}
        good = copy.deepcopy(initial)
        good["consumed"].update(opportunities=8, wall_millis=1500)
        causal_result.check_owner(initial, good, good, 4, "completed")
        for fault in ("opportunities", "wall_millis", "attempts", "identity", "observer", "unknown", "stop"):
            with self.subTest(fault=fault):
                value = copy.deepcopy(good)
                if fault in ("opportunities", "wall_millis", "attempts"):
                    value["consumed"][fault] = 0
                elif fault == "identity":
                    value["attempt_id"] = "different"
                elif fault == "observer":
                    value["observer_sequence"] = 7
                elif fault == "unknown":
                    value["physical_work_unknown"] = True
                else:
                    value["stop"] = "user_stop"
                with self.assertRaises(ValueError):
                    causal_result.check_owner(initial, value, value, 4, "completed")

    def test_observer_rejects_duplicate_nonfinite_and_oversized_records(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            for payload in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":"' + "x" * 32 + '"}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    load(path, 24)
            with self.assertRaises(ValueError):
                write(path, {"bad": float("inf")})

    def test_two_hour_attempt_ceiling_is_finite_for_both_operations(self):
        for prompt_only in (False, True):
            value = {"opportunities": 0 if prompt_only else 4, "wall_millis": 7_200_000,
                     "stop_reserve_millis": 60_000, "checkpoint_every": 0 if prompt_only else 4,
                     "evaluation_targets": 0 if prompt_only else 1,
                     "generation_cases": 40 if prompt_only else 24, "generation_bytes": 192}
            for accepted in (3_600_001, 6_600_000, 7_200_000):
                with self.subTest(prompt_only=prompt_only, accepted=accepted):
                    causal_contract.limits(value | {"wall_millis": accepted}, prompt_only)
            for rejected in (7_200_001, 7_200_000.0, True, 60_000):
                with self.subTest(prompt_only=prompt_only, rejected=rejected), self.assertRaises(ValueError):
                    causal_contract.limits(value | {"wall_millis": rejected}, prompt_only)

    def test_protocol_drift_rejected_before_admission(self):
        request = {name: None for name in ("binary", "source_manifest", "build_receipt", "identity", "config", "data_root", "budget", "limits", "resume", "operation")}
        for protocol in ("text-v4", "causal-prefix-v1", "causal-prefix-v2"):
            request.update(schema=1, protocol=protocol)
            with patch.object(causal_contract, "budget") as admission_input, self.assertRaisesRegex(ValueError, "unsupported causal owner request"):
                causal_contract.validate(Path("."), request)
            admission_input.assert_not_called()

    def test_all_numerical_documents_are_required_and_pinned(self):
        files = {name: "a" * 64 for name in ("Cargo.toml", "Cargo.lock", ".cargo/config.toml",
                 "datasets/tiny/train.txt", "datasets/tiny/development-t2.txt")}
        files["docs/CAUSAL-PREFIX-CONTRACT.md"] = causal_contract.BASE_PROTOCOL_SHA256
        files["docs/CAUSAL-PREFIX-V2-CONTRACT.md"] = causal_contract.V2_PROTOCOL_SHA256
        files["docs/CAUSAL-PREFIX-V3-CONTRACT.md"] = causal_contract.PROTOCOL_SHA256
        for changed in ("missing", "base", "attention", "successor"):
            candidate = files.copy()
            if changed == "missing":
                del candidate["docs/CAUSAL-PREFIX-V3-CONTRACT.md"]
            else:
                filename = {"base": "CAUSAL-PREFIX-CONTRACT.md", "attention": "CAUSAL-PREFIX-V2-CONTRACT.md",
                            "successor": "CAUSAL-PREFIX-V3-CONTRACT.md"}[changed]
                candidate["docs/" + filename] = "f" * 64
            with tempfile.TemporaryDirectory() as name, patch.object(causal_contract, "pin", return_value=Path(name)), \
                    patch.object(causal_contract, "load", return_value={"schema": 1, "files": candidate}), self.assertRaises(ValueError):
                causal_contract.source_files(Path(name), {})

    def test_joint_rule_config_has_exact_fields_and_types(self):
        value = {"schema": 1, "protocol": causal_contract.PROTOCOL, "profile": "candidate", "seed": 7,
                 "threads": 1, "anchor_interval": 4, "max_visits": 4, "history_capacity": 8, "context_bytes": 1024,
                 "evidence_rule": "threshold-one-canonical-zero-v1", "route_credit": "detached-u-experimental",
                 "predictor_rule": "joint-adjacent-raw-row-mean-v1", "anchor_batch": "current-anchor-forward-v1",
                 "replay_selection": "newest4-compatible-v1"}
        causal_contract.configuration(value)
        for key, bad in (("predictor_rule", "dense"), ("anchor_batch", "sequential"),
                         ("replay_selection", "newest8"), ("threads", True), ("seed", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                causal_contract.configuration(value | {key: bad})
        old = value.copy()
        del old["predictor_rule"]
        with self.assertRaises(ValueError):
            causal_contract.configuration(old)


if __name__ == "__main__":
    unittest.main()
