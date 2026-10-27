"""Authored cross-boundary setup charge and late STOP receipt fixtures."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import causal_contract, causal_prepare, causal_result, causal_runner, causal_terminal
from scripts.causal_observer import Observer, encoded, load, write
from scripts.pulse_io import digest
from test_causal_receipts import development
from test_causal_resume import blob, released_fixture


def committed(directory, identity, owner, sequence):
    directory.mkdir(parents=True, exist_ok=True)
    state = {"boundary": "ready", "global_opportunity": 0, "cursor": {},
             "counters": {key: 0 for key in ("anchors", "local_opportunities", "visits", "local_tile_updates",
                 "anchor_tile_commits", "predictor_anchor_updates", "predictor_replay_updates",
                 "predictor_anchor_pairs", "predictor_replay_pairs", "predictor_transitions")},
             "predictor_version": 0, "predictor_mse_bits": 0, "anchor_loss_bits": 0}
    manifest = {"schema": 1, "sequence": sequence, "identity": identity, "owner": owner,
                "state": blob(directory, "state", state), "shared": blob(directory, "shared", {"packed": 0}),
                "predictor": blob(directory, "predictor", {"packed": 0}),
                "tiles": [{"id": i, "version": 0, "blob": blob(directory, "tile", {"id": i})} for i in range(7)]}
    data = encoded(manifest)
    sha = hashlib.sha256(data).hexdigest()
    (directory / f"manifest-{sha}.bin").write_bytes(data)
    reference = {"sequence": sequence, "manifest_sha256": sha}
    write(directory / f"commit-{sequence:020}-{sha}.json", reference)
    return reference, state


class CausalRevisionTests(unittest.TestCase):
    def test_joint_contribution_conservation_and_finite_metrics(self):
        good = {"anchors": 1, "local_opportunities": 3, "visits": 16, "local_tile_updates": 12,
                "anchor_tile_commits": 2, "predictor_anchor_updates": 1, "predictor_replay_updates": 2,
                "predictor_anchor_pairs": 4, "predictor_replay_pairs": 5, "predictor_transitions": 384,
                "predictor_mse_sum": 0.12345678901234566, "anchor_loss_sum": 1.0}
        causal_result.counters(good, 4, 3)
        for key, bad in (("predictor_anchor_updates", 4), ("predictor_anchor_pairs", 3),
                         ("predictor_replay_pairs", 9), ("predictor_replay_pairs", 1),
                         ("predictor_replay_updates", 4), ("predictor_transitions", 385),
                         ("predictor_anchor_pairs", True), ("predictor_mse_sum", float("nan"))):
            with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                causal_result.counters(good | {key: bad}, 4, 3)
        for key in ("predictor_anchor_pairs", "predictor_replay_pairs"):
            value = good.copy()
            del value[key]
            with self.assertRaises(ValueError):
                causal_result.counters(value, 4, 3)
        with self.assertRaises(ValueError):
            causal_result.counters(good, 4, 4)

    def test_python_handoff_preserves_parent_start_and_positive_setup_charge(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _, identity, budget, _, _, resume = released_fixture(root)
            identity["data"] = {"manifest_sha256": "a" * 64}
            identity["kernel"] = "scalar"
            request = {"resume": resume, "identity": identity, "budget": budget, "operation": {"kind": "train"},
                       "config": {}, "source_manifest": {"sha256": "b" * 64},
                       "limits": {"opportunities": 4, "wall_millis": 10_000, "stop_reserve_millis": 5000,
                                  "checkpoint_every": 4, "evaluation_targets": 8, "generation_cases": 1, "generation_bytes": 2}}
            paths = {"data_root": root, "source_manifest": root / "manifest.json"}
            parent_identity = {key: identity[key] for key in ("source_sha256", "runtime_sha256")}
            with patch.object(causal_prepare.causal_checkpoint, "identity_absent", return_value=True):
                inherited = causal_prepare.causal_checkpoint.parent(root, resume, parent_identity, budget)
            with patch.object(causal_prepare.causal_checkpoint, "parent", return_value=inherited), \
                    patch.object(causal_prepare.time, "monotonic", return_value=1.125):
                context = causal_prepare.create(root, request, 1.0)
                worker, _ = causal_prepare.worker(root, request, paths, context)
            owner = worker["owner"]
            self.assertEqual(owner["attempt_start_wall_millis"], 2000)
            self.assertEqual(owner["consumed"]["wall_millis"], 2125)
            self.assertEqual(owner["consumed"]["attempts"], 2)
            self.assertEqual(owner["budget"], budget)
            self.assertEqual(load(context["directory"] / "worker-request.json")["owner"], owner)

    def test_late_stop_receipt_accepts_retained_evaluation_and_known_exposure_release(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            identity = {"source_sha256": "1" * 64}
            owner = {"schema": 1, "run_id": "r", "attempt_id": "a", "parent": None,
                     "budget": {"maximum_wall_millis": 20_000}, "consumed": {"opportunities": 0, "wall_millis": 100, "attempts": 1},
                     "attempt_start_opportunity": 0, "attempt_start_wall_millis": 0,
                     "observer_sequence": 1, "stop": "running", "physical_work_unknown": False}
            evaluated, _ = committed(directory / "persistence", identity, owner, 1)
            committed(directory / "persistence", identity, owner, 2)
            stopped = copy.deepcopy(owner)
            stopped.update(stop="user_stop", observer_sequence=2)
            stopped["consumed"]["wall_millis"] = 200
            final, state = committed(directory / "persistence", identity, stopped, 3)
            limits = {"opportunities": 4, "evaluation_targets": 8, "generation_cases": 1, "generation_bytes": 2}
            worker = {"owner": owner, "identity": identity, "limits": limits, "operation": {"kind": "train"}, "data_manifest_sha256": "a" * 64}
            result = {"schema": 1, "protocol": causal_contract.PROTOCOL, "attempt_id": "a", "status": "user_stop",
                      "identity": identity, "checkpoint": final, "global_opportunity": 0, "physical_opportunities_completed": 0,
                      "owner": stopped, "cursor": state["cursor"], "counters": state["counters"] | {"predictor_mse_sum": 0.0, "anchor_loss_sum": 0.0},
                      "operation": {"kind": "train"}, "evaluation": {"kind": "train", "result": {"initial": development(), "final": development(),
                          "omitted_reason": None, "final_evaluation_checkpoint": evaluated}}, "timings": {"training_millis": 0},
                      "training_admission_claim": False, "useful_assistant_claim": False}
            write(directory / "result.json", result)
            write(directory / "stdout.log", {"schema": 1, "protocol": causal_contract.PROTOCOL, "attempt_id": "a",
                  "result_sha256": digest(directory / "result.json"), "status": "user_stop", "result_file": "result.json"})
            child = {"reaped": True, "returncode": 0, "job_closed": True, "terminal_peak_memory_queried": True, "stop_requested": "user_stop"}
            accepted = causal_result.check(directory, worker, child)
            context = {"directory": directory, "envelope": owner, "observer": Observer(directory, 3), "started": 0,
                       "deadline": 10, "stop_paths": [directory / "STOP"]}
            report = {"child": child, "physical_work_unknown": True, "consumed": owner["consumed"].copy(),
                      "admission": {"owner": {"pid": 5, "creation_filetime": 6}}, "release_ready": False}
            causal_runner.accept_result(context, report, accepted)
            self.assertFalse(report["physical_work_unknown"])
            with patch.object(causal_terminal.time, "monotonic", return_value=1.0):
                causal_terminal.finish(context, report)
            self.assertTrue(report["release_ready"])
            self.assertEqual(load(directory / "release.json")["status"], "user_stop")
            marker = directory / "persistence" / f"commit-{evaluated['sequence']:020}-{evaluated['manifest_sha256']}.json"
            marker.unlink()
            with self.assertRaises(FileNotFoundError):
                causal_result.check(directory, worker, child)


if __name__ == "__main__":
    unittest.main()
