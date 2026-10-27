"""Fault contracts for the bounded fixture owner; no real training is launched."""

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import pulse_runner
from scripts.pulse_io import LOG_LIMIT, read_result


class PulseRunnerTests(unittest.TestCase):
    def test_invalid_bounds_fail_before_creating_work(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for kwargs in ({"steps": 0}, {"steps": 257}, {"seeds": []},
                           {"seeds": [1, 1]}, {"seeds": [1, 2, 3, 4]},
                           {"seeds": [-1]}, {"seeds": [2**64]},
                           {"timeout": math.nan}, {"timeout": 121},
                           {"residents": 4}, {"anchor_interval": 17}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    pulse_runner.execute(root, **kwargs)
                self.assertFalse((root / "work").exists())

    def make_root(self, name):
        root = Path(name)
        binary = root / "work/target/debug" / ("hexweave.exe" if os.name == "nt" else "hexweave")
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"mock executable: never launched")
        return root

    def test_first_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.make_root(name)
            with patch.object(pulse_runner, "run_child", side_effect=RuntimeError("fault")) as child:
                directory, report = pulse_runner.execute(root, seeds=[7, 8, 9])
            self.assertEqual(child.call_count, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(json.loads((directory / "report.json").read_text()), report)
            self.assertFalse(report["native_chat_wakeup"])

    def test_preexisting_stop_never_launches_child(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.make_root(name)
            stop = root / "work/pulse/STOP"
            stop.parent.mkdir()
            stop.touch()
            with patch.object(pulse_runner, "run_child") as child:
                _, report = pulse_runner.execute(root)
            child.assert_not_called()
            self.assertEqual(report["status"], "stopped")
            self.assertTrue(stop.exists())

    def test_success_is_independent_seeds_not_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root = self.make_root(name)
            with patch.object(pulse_runner, "run_child", return_value={"mock": True}) as child:
                _, report = pulse_runner.execute(root, seeds=[7, 8, 9])
            self.assertEqual(child.call_count, 3)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["resume"])
            self.assertIsNone(report["checkpoint"])

    def test_timeout_terminates_only_owned_child(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            process = Mock(pid=42, returncode=-1)
            process.poll.return_value = None
            with patch.object(pulse_runner.subprocess, "Popen", return_value=process), \
                    patch.object(pulse_runner, "watch", side_effect=TimeoutError("timeout")):
                with self.assertRaises(TimeoutError):
                    pulse_runner.run_child(["owned"], root, root / "STOP", 1, 7, 1, "pulse", 1, 4)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
            process.kill.assert_not_called()

    def test_stubborn_owned_child_is_reaped_after_kill(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 5), 1]
        pulse_runner.stop_owned(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_terminal_success_does_not_terminate_exited_child(self):
        process = Mock()
        process.poll.return_value = 0
        pulse_runner.stop_owned(process)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_watch_honors_stop_and_log_limits(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            logs = (root / "stdout", root / "stderr")
            for path in logs:
                path.touch()
            process = Mock()
            process.poll.return_value = None
            (root / "STOP").touch()
            with self.assertRaises(InterruptedError):
                pulse_runner.watch(process, root, root / "STOP", logs, 1)
            (root / "STOP").unlink()
            logs[0].write_bytes(b"x" * (LOG_LIMIT + 1))
            with self.assertRaises(ValueError):
                pulse_runner.watch(process, root, root / "STOP", logs, 1)

    def test_child_nonzero_exit_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            process = Mock(pid=42, returncode=9)
            process.poll.return_value = 9
            with patch.object(pulse_runner.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "exit 9"):
                    pulse_runner.run_child(["owned"], root, root / "STOP", 1, 7, 1, "pulse", 1, 4)
            process.terminate.assert_not_called()

    def test_result_must_match_finite_protocol_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "result.json"
            valid = {"protocol": "ternary-pulse-v1", "completed_steps": 1,
                     "seed": 7, "mode": "pulse", "resident_limit": 1, "anchor_interval": 4,
                     "initial_loss": 2.5, "final_loss": 2.4, "seconds": 0.1,
                     "parameters": 32, "all_learned_weights_ternary": True,
                     "fp32_master_weights": False, "counts": {"anchor_steps": 1},
                     "training_target_exposures": 1,
                     "state_bytes": {"packed_weight_payload": 8, "shared_integer_evidence": 0,
                                     "resident_integer_evidence_current": 64,
                                     "resident_integer_evidence_peak": 64,
                                     "resident_integer_evidence_bound": 64}}
            path.write_text(json.dumps(valid))
            self.assertEqual(read_result(path, 1, 7, "pulse", 1, 4), valid)
            for change in ({"protocol": "other"}, {"completed_steps": 0},
                           {"seed": 8}, {"mode": "anchor-only"}, {"loss": math.nan},
                           {"completed_steps": True}, {"resident_limit": 3},
                           {"anchor_interval": 1}, {"counts": {}}, {"state_bytes": {}},
                           {"initial_loss": None}, {"parameters": True},
                           {"fp32_master_weights": True}):
                path.write_text(json.dumps(valid | change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_result(path, 1, 7, "pulse", 1, 4)
            path.write_text('{"seed":7,"seed":8}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_result(path, 1, 7, "pulse", 1, 4)


if __name__ == "__main__":
    unittest.main()
