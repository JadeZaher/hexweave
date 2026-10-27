"""Authored generic visible-prompt and zero-exposure fault fixtures."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import causal_contract, causal_prepare, causal_prompts, causal_result
from scripts.causal_observer import write
from scripts.pulse_io import digest


def input_file(root, rows=None):
    directory = root / "work/inputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompts.json"
    write(path, {"schema": 1, "prompts": rows or [{"id": "one", "prompt": "visible evidence"}]})
    return {"path": str(path), "sha256": digest(path)}


def prompt_fixture(root):
    spec = input_file(root)
    reference = {"sequence": 1, "manifest_sha256": "a" * 64}
    worker = {"workspace": str(root), "operation": {"kind": "prompt_evaluation", "prompts": spec},
              "limits": {"opportunities": 0, "checkpoint_every": 0, "evaluation_targets": 0,
                         "generation_cases": 1, "generation_bytes": 2, "wall_millis": 10_000, "stop_reserve_millis": 5000},
              "resume": {"checkpoint": reference, "parent_directory": str(root / "parent")}, "identity": {"pinned": True}}
    generation = {"raw_bytes": [255, 10], "valid_utf8": False, "stop_reason": {"kind": "newline"},
                  "before_sha256": "b" * 64, "after_sha256": "b" * 64, "wall_micros": 1,
                  "routes": {"visits": [2, 0, 0, 0, 0, 0, 0], "route_calls": 2, "halts": 2, "forced_exits": 0}}
    value = {"input_sha256": spec["sha256"], "parent_checkpoint": reference, "evaluation_checkpoint": reference,
             "before_sha256": "b" * 64, "after_sha256": "b" * 64, "requested_cases": 1,
             "cases": [{"id": "one", "prompt_sha256": hashlib.sha256(b"visible evidence").hexdigest(), "generation": generation}],
             "coverage": {"kind": "complete"}}
    result = {"physical_opportunities_completed": 0, "status": "completed", "timings": {"training_millis": 0},
              "operation": worker["operation"], "evaluation": {"kind": "prompt_evaluation", "result": value}}
    manifest = {"identity": worker["identity"], "state": {"sha256": "c" * 64},
                "shared": {"sha256": "d" * 64}, "predictor": {"sha256": "e" * 64}, "tiles": [0] * 7}
    return worker, value, result, manifest


class CausalPromptTests(unittest.TestCase):
    def test_exact_prompt_schema_pins_and_bounds(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = [{"id": f"case_{i}", "prompt": "x" * 832} for i in range(40)]
            spec = input_file(root, rows)
            self.assertEqual(len(causal_prompts.read(root, spec, 40, 192)[1]), 40)
            for bad in (rows + [{"id": "extra", "prompt": "x"}], rows[:-1],
                        [rows[0]] * 40, [rows[0] | {"answer": "private"}] + rows[1:],
                        [rows[0] | {"prompt": "x" * 833}] + rows[1:]):
                spec = input_file(root, bad)
                with self.assertRaises(ValueError):
                    causal_prompts.read(root, spec, 40, 192)
            path = Path(spec["path"])
            path.write_text('{"schema":1,"schema":1,"prompts":[]}', encoding="utf-8")
            spec["sha256"] = digest(path)
            with self.assertRaises(ValueError):
                causal_prompts.read(root, spec, 1, 1)
            path.write_bytes(b" " * 65_537)
            spec["sha256"] = digest(path)
            with self.assertRaises(ValueError):
                causal_prompts.read(root, spec, 1, 1)
            spec = input_file(root)
            with self.assertRaises(ValueError):
                causal_prompts.read(root, spec | {"sha256": "0" * 64}, 1, 2)

    def test_zero_training_limits_and_completed_parent_only(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            worker, _, _, _ = prompt_fixture(root)
            causal_contract.limits(worker["limits"], True)
            for field in ("opportunities", "checkpoint_every", "evaluation_targets"):
                with self.assertRaises(ValueError):
                    causal_contract.limits(worker["limits"] | {field: 1}, True)
            with self.assertRaises(ValueError):
                causal_contract.limits(worker["limits"], False)
            with self.assertRaises(ValueError):
                causal_prompts.operation({"kind": "prompt_evaluation", "prompts": worker["operation"]["prompts"], "keys": "hidden"})
            for status in ("user_stop", "resource_stop"):
                inherited = (root, {"status": status}, {"owner": {"stop": "running"}}, {})
                request = {"operation": worker["operation"], "resume": {}, "identity": {}, "budget": {}}
                with patch.object(causal_prepare.causal_checkpoint, "parent", return_value=inherited), self.assertRaises(ValueError):
                    causal_prepare.create(root, request, 0)

    def test_owner_rejects_state_exposure_order_pin_and_output_faults(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            worker, value, result, manifest = prompt_fixture(root)
            causal_prompts.check(value, worker, result, manifest, manifest, manifest)
            for fault in ("state", "physical", "training_time", "missing", "order", "pin", "hash", "utf8", "bytes", "routes"):
                candidate, receipt, final = copy.deepcopy(value), copy.deepcopy(result), copy.deepcopy(manifest)
                if fault == "state":
                    final["state"]["sha256"] = "f" * 64
                elif fault == "physical":
                    receipt["physical_opportunities_completed"] = 1
                elif fault == "training_time":
                    receipt["timings"]["training_millis"] = 1
                elif fault == "missing":
                    candidate["cases"] = []
                elif fault == "order":
                    candidate["cases"][0]["id"] = "other"
                elif fault == "pin":
                    candidate["input_sha256"] = "f" * 64
                elif fault == "hash":
                    candidate["after_sha256"] = "f" * 64
                elif fault == "utf8":
                    candidate["cases"][0]["generation"]["valid_utf8"] = True
                elif fault == "bytes":
                    candidate["cases"][0]["generation"]["raw_bytes"] = [True]
                else:
                    candidate["cases"][0]["generation"]["routes"]["halts"] = 0
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    causal_prompts.check(candidate, worker, receipt, manifest, manifest, final)

    def test_partial_stop_has_no_score_and_never_uses_development_validator(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            worker, value, result, manifest = prompt_fixture(root)
            result["status"] = "user_stop"
            value.update(cases=[], coverage={"kind": "interrupted", "status": "user_stop"})
            with patch.object(causal_result.causal_checkpoint, "checkpoint", return_value=(manifest, {})), \
                    patch.object(causal_result, "evaluation", side_effect=AssertionError("development access")):
                causal_result.check_evaluations(root, worker, result, manifest)
            result["status"] = "completed"
            with self.assertRaises(ValueError):
                causal_prompts.check(value, worker, result, manifest, manifest, manifest)


if __name__ == "__main__":
    unittest.main()
