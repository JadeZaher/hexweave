"""Bounded text-owner fault contracts; test doubles never start training."""

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import text_child, text_runner
from scripts.text_io import read_receipt
from scripts.text_resources import CHILD_LIMIT, GIB, admission, required_cushion


def receipt_fixture(options):
    counts = dict.fromkeys(("revisits", "predictor_replays"), 0)
    counts.update(tile_visits=8, local_updates=6, predictor_predictions=6, anchor_targets=2, returned_frames=2,
                  learned_halts=8, forced_exits=0, route_calls=8, predictor_updates=2, predictor_error_sum=0.2,
                  visits_by_tile=[8, 0, 0, 0, 0, 0, 0], path_lengths=[8, 0, 0], transitions=[0] * 5)
    return {"protocol": "text-pulse-v4", "status": "complete", "options": options, "data": {}, "credit_diagnostics": None,
            "completed_step_opportunities": 8, "counts": counts,
            "residency": {"peak_live_tiles": 1}, "resolved_kernel": "avx2",
            "architecture": {"floating_weight_masters": False,
                             "training_pulse": {"kind": "joint-tile-pulse-v1", "all_coefficients_ternary": True,
                                                "route_credit": options["route_credit"]}},
            "evaluation": {"initial_mean_nll_nats_per_byte": 5.5, "final_mean_nll_nats_per_byte": 5.4,
                           "examples_per_pass": 64, "history_reset_each_example": True, "training_mutation_checked": True},
            "predictor_error": {"update_denominator": 2, "sum_per_update_mean_squared_error": 0.2,
                                "mean_squared_error": 0.1, "includes_replay": True, "components_per_update": 16},
            "tile_transitions": dict.fromkeys(("transform", "curve", "router", "pulse_context", "pulse_heads"), 0),
            "shared_transitions": dict.fromkeys(("embedding", "readout", "predictor"), 0),
            "disk_io": dict.fromkeys(("loads", "commits", "bytes_read", "bytes_written", "initialization_bytes_written"), 1),
            "timings": {"training_seconds": 0.5, "total_seconds": 1},
            "artifact_bounds": {"checkpoint_resumable": False}}


class TextRunnerTests(unittest.TestCase):
    def root(self, name):
        root = Path(name)
        binary = root / "work/target/release/hexweave.exe"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"mock binary never executed")
        data = root / "datasets/tiny"
        data.mkdir(parents=True)
        (data / "train.txt").write_text("train bytes only")
        (data / "valid.txt").write_text("valid bytes only")
        return root

    def test_bounds_reject_before_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            for options in ({"steps": 0}, {"steps": 1025}, {"seed": -1},
                            {"timeout": math.nan}, {"timeout": 121},
                            {"anchor_interval": 17}, {"backend": "gpu"}, {"route_credit": "uncoupled"},
                            {"profile": "t1"}, {"credit_diagnostics": "on"},
                            {"profile": "t2-credit", "mode": "anchor-only"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    text_runner.execute(Path(name), **options)
            self.assertFalse((Path(name) / "work").exists())

    def test_admission_failure_is_durable_and_never_launches(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            with patch.object(text_runner, "admission", side_effect=MemoryError("too little RAM")), \
                    patch.object(text_runner, "run_child") as child:
                directory, report = text_runner.execute(root)
            child.assert_not_called()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(json.loads((directory / "report.json").read_text()), report)
            self.assertEqual(report["input_sha256"], report["input_sha256_after"])

    def test_t2_owner_rejects_unpinned_development_before_launch(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            (root / "datasets/tiny/development-t2.txt").write_text("wrong development bytes")
            with patch.object(text_runner, "run_child") as child:
                _, report = text_runner.execute(root, profile="t2-credit", credit_diagnostics="on")
            child.assert_not_called()
            self.assertEqual(report["status"], "failed")
            self.assertIn("frozen", report["error"])

    def test_preexisting_stop_never_launches_or_clears_stop(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            stop = root / "work/text/STOP"
            stop.parent.mkdir()
            stop.touch()
            with patch.object(text_runner, "run_child") as child:
                _, report = text_runner.execute(root)
            child.assert_not_called()
            self.assertEqual(report["status"], "stopped")
            self.assertTrue(stop.exists())

    def test_single_failure_never_retries(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            with patch.object(text_runner, "admission", return_value={"host_cushion_bytes": 1}), \
                    patch.object(text_runner, "run_child", side_effect=RuntimeError("fault")) as child:
                _, report = text_runner.execute(root)
            child.assert_called_once()
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["automatic_retry"])

    def test_mutated_dataset_invalidates_success(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            def mutate(*_args):
                (root / "datasets/tiny/train.txt").write_text("changed input")
            with patch.object(text_runner, "admission", return_value={"host_cushion_bytes": 1}), \
                    patch.object(text_runner, "run_child", side_effect=mutate):
                _, report = text_runner.execute(root)
            self.assertEqual(report["status"], "failed")
            self.assertIn("identity_error", report)

    def test_final_validation_cannot_outlive_whole_run_deadline(self):
        real_publish = text_runner.publish
        for phase in ("receipt", "receipt-stop", "event", "publish"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as name:
                root, now = self.root(name), [0.0]

                def receipt(*_args):
                    if phase == "receipt":
                        now[0] = 31.0
                    if phase == "receipt-stop":
                        (root / "work/text/STOP").touch()
                    return {}

                def event(directory, kind, **fields):
                    if kind == "run_finished" and phase == "event":
                        now[0] = 31.0

                def publish(directory, report):
                    real_publish(directory, report)
                    if report["status"] == "completed" and phase == "publish":
                        now[0] = 31.0

                with patch.object(text_runner.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(text_runner, "admission", return_value={"host_cushion_bytes": 1}), \
                        patch.object(text_runner, "run_child") as child, patch.object(text_runner, "read_receipt", side_effect=receipt), \
                        patch.object(text_runner, "event", side_effect=event), patch.object(text_runner, "publish", side_effect=publish):
                    directory, report = text_runner.execute(root, timeout=30)
                child.assert_called_once()
                self.assertEqual(report["status"], "stopped" if phase == "receipt-stop" else "failed")
                self.assertEqual(json.loads((directory / "report.json").read_text()), report)

    def test_physical_admission_retains_adaptive_cushion(self):
        self.assertEqual(required_cushion(8 * GIB), 2 * GIB)
        state = {"total_physical_bytes": 32 * GIB, "available_physical_bytes": 3 * GIB}
        with patch("scripts.text_resources.memory_status", return_value=state):
            with self.assertRaises(MemoryError):
                admission()
        state["available_physical_bytes"] = 8 * GIB
        with patch("scripts.text_resources.memory_status", return_value=state):
            self.assertEqual(admission()["child_commit_limit_bytes"], CHILD_LIMIT)

    def test_job_assignment_failure_never_resumes_child(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            process = Mock(pid=42, returncode=-1)
            process.poll.side_effect = [None, -1]
            job = Mock()
            job.assign.side_effect = RuntimeError("job assignment denied")
            with patch.object(text_child, "Job", return_value=job), \
                    patch.object(text_child.subprocess, "Popen", return_value=process), \
                    patch.object(text_child, "process_memory", return_value={}), \
                    patch.object(text_child, "resume_suspended") as resume:
                with self.assertRaisesRegex(RuntimeError, "job assignment denied"):
                    text_child.run_child(root, ["mock"], root, root, 0, 0, "auto", {})
            resume.assert_not_called()
            process.terminate.assert_called_once()
            job.close.assert_called_once()

    def test_cooperative_stop_precedes_termination(self):
        with tempfile.TemporaryDirectory() as name:
            process = Mock()
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("mock", 1), 0]
            text_child.stop_owned(process, Path(name))
            self.assertTrue((Path(name) / "STOP").exists())
            process.terminate.assert_called_once()
            self.assertEqual(process.wait.call_args_list[0].kwargs, {"timeout": 1})

    def test_stop_failure_still_closes_job_and_reaps_owned_handle(self):
        process = Mock(returncode=-1)
        process.poll.return_value = None
        job = Mock()
        child = {}
        with patch.object(text_child, "stop_owned", side_effect=OSError("STOP creation denied")), \
                patch.object(text_child, "process_memory", return_value={"peak_working_set_bytes": 123}):
            errors = text_child.cleanup(process, job, Path("unused"), child)
        job.close.assert_called_once()
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=2)
        self.assertTrue(child["reaped"])
        self.assertIn("STOP creation denied", errors[0])

    def test_terminal_memory_failure_keeps_prior_peak_and_reap_evidence(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        child = {"memory": {"peak_working_set_bytes": 456}}
        with patch.object(text_child, "process_memory", side_effect=OSError("query failed")):
            errors = text_child.cleanup(process, Mock(), Path("unused"), child)
        self.assertTrue(child["reaped"])
        self.assertFalse(child["terminal_peak_memory_queried"])
        self.assertEqual(child["memory"]["peak_working_set_bytes"], 456)
        self.assertIn("query failed", errors[0])

    def test_memory_peak_does_not_shrink_with_terminal_zero(self):
        child = {}
        text_child.record_memory(child, {"peak_working_set_bytes": 456, "working_set_bytes": 123})
        text_child.record_memory(child, {"peak_working_set_bytes": 0, "working_set_bytes": 0})
        self.assertEqual(child["memory"]["peak_working_set_bytes"], 456)

    def test_observer_failure_still_publishes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.root(name)
            with patch.object(text_runner, "event", side_effect=OSError("disk event failure")):
                directory, report = text_runner.execute(root)
            self.assertEqual(report["status"], "failed")
            self.assertTrue((directory / "report.json").is_file())

    def test_receipt_checks_precision_options_and_exposure(self):
        options = {"steps": 8, "seed": 7, "anchor_interval": 4, "mode": "pulse", "route_credit": "coupled",
                   "profile": "fixture", "credit_diagnostics": "off"}
        receipt = receipt_fixture(options)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "result.json"
            snapshot = {key: receipt[key] for key in ("protocol", "status", "options", "data", "completed_step_opportunities")}
            snapshot.update(schema=4, resumable=False, anchor_targets=2, tile_versions=[0] * 7)
            snapshot_path = path.with_name("snapshot.json")
            snapshot_path.write_text(json.dumps(snapshot))
            path.write_text(json.dumps(receipt))
            self.assertEqual(read_receipt(path, options), receipt)
            for change in ({"counts": {"anchor_targets": 3}}, {"status": "stopped"},
                           {"residency": {"peak_live_tiles": True}}, {"evaluation": {}},
                           {"options": options | {"seed": True}}, {"resolved_kernel": "gpu"},
                           {"options": options | {"route_credit": "route-to-u-detached"}},
                           {"counts": receipt["counts"] | {"route_calls": 7}},
                           {"counts": receipt["counts"] | {"returned_frames": 0, "local_updates": 8,
                            "predictor_predictions": 8, "predictor_updates": 0, "predictor_error_sum": 0.0},
                            "predictor_error": receipt["predictor_error"] | {"update_denominator": 0,
                            "sum_per_update_mean_squared_error": 0.0, "mean_squared_error": None}},
                           {"predictor_error": receipt["predictor_error"] | {"update_denominator": 9}},
                           {"protocol": "text-pulse-v3"}, {"credit_diagnostics": {}},
                           {"architecture": {"floating_weight_masters": False}}):
                path.write_text(json.dumps(receipt | change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_receipt(path, options)
            path.write_text(json.dumps(receipt))
            for change in ({"resumable": True}, {"options": options | {"route_credit": "route-to-u-detached"}},
                           {"tile_versions": [0] * 6}):
                snapshot_path.write_text(json.dumps(snapshot | change))
                with self.subTest(snapshot=change), self.assertRaises(ValueError):
                    read_receipt(path, options)



if __name__ == "__main__":
    unittest.main()
