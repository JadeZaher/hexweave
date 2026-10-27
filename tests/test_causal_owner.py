"""Authored owner lifecycle and late-publication faults; no numerical work."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import causal_child, causal_contract, causal_terminal
from scripts.causal_observer import Observer, load, write


class Process:
    pid, _handle = 101, 123

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = 1


def terminal_fixture(directory):
    envelope = {"schema": 1, "run_id": "r", "attempt_id": "a", "attempt_start_wall_millis": 0,
                "budget": {"maximum_wall_millis": 20_000}}
    context = {"directory": directory, "envelope": envelope, "observer": Observer(directory, 1),
               "started": 0, "deadline": 10, "stop_paths": [directory / "STOP"]}
    report = {"status": "completed", "physical_work_unknown": False,
              "consumed": {"opportunities": 4, "wall_millis": 0, "attempts": 1},
              "result": {"checkpoint": {"sequence": 1, "manifest_sha256": "a" * 64}},
              "admission": {"owner": {"pid": 5, "creation_filetime": 6}}, "release_ready": False}
    return context, report


class CausalOwnerTests(unittest.TestCase):
    def test_suspended_assignment_precedes_resume_and_exact_reaping(self):
        with tempfile.TemporaryDirectory() as name:
            directory, order, process = Path(name), [], Process()
            owner = SimpleNamespace(pid=1, driver=SimpleNamespace(pid=2))
            job = SimpleNamespace(assign=lambda value: order.append(("assign", value)),
                                  close=lambda: order.append(("close", process)))
            memory = {"working_set_bytes": 100, "peak_working_set_bytes": 100}

            def launched(*_args, **kwargs):
                self.assertTrue(kwargs["creationflags"] & 4)
                self.assertFalse(kwargs["shell"])
                order.append(("create", process))
                return process

            def resumed(value):
                self.assertEqual(order[-1], ("assign", value))
                self.assertTrue(load(directory / "launch.json")["job_assigned"])
                order.append(("resume", value))

            with patch.object(causal_child.resources, "Job", return_value=job), \
                    patch.object(causal_child.subprocess, "Popen", side_effect=launched), \
                    patch.object(causal_child.subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000, create=True), \
                    patch.object(causal_child.resources, "process_identity", side_effect=lambda value: {"pid": value.pid, "creation_filetime": 1}), \
                    patch.object(causal_child.resources, "resume_suspended", side_effect=resumed), \
                    patch.object(causal_child.resources, "process_memory", return_value=memory), \
                    patch.object(causal_child, "watch", side_effect=lambda *_: process.wait()), \
                    patch.object(causal_child.time, "monotonic", return_value=0):
                child = {}
                causal_child.run(directory, directory / "model.exe", directory, "a" * 64, "nonce", owner,
                    {"attempt_id": "a", "budget": {"disk_bytes_per_attempt": 1_048_576}}, [], 10, 5,
                    {"host_cushion_bytes": 1}, "scalar", Observer(directory, 1), child)
            self.assertEqual([event for event, _ in order], ["create", "assign", "resume", "close"])
            self.assertTrue(child["reaped"])
            self.assertTrue(child["terminal_peak_memory_queried"])
            self.assertNotIn("forced_termination", child)

    def test_resource_stop_requests_once_and_hard_deadline_is_terminal(self):
        for forced in (False, True):
            with self.subTest(forced=forced), tempfile.TemporaryDirectory() as name:
                directory, now, polls = Path(name), [0.0], [0]
                for filename in ("stdout.log", "stderr.log"):
                    (directory / filename).touch()
                process, child = Process(), {}

                def poll():
                    polls[0] += 1
                    return None if forced or polls[0] < 3 else 0

                def advance(_seconds):
                    now[0] += 1

                process.poll = poll
                with patch.object(causal_child.resources, "sample", side_effect=MemoryError("cushion")), \
                        patch.object(causal_child.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(causal_child.time, "sleep", side_effect=advance):
                    if forced:
                        with self.assertRaises(TimeoutError):
                            causal_child.watch(process, None, directory, [], 10, 5, 0, 1_048_576, Observer(directory, 1), child)
                    else:
                        causal_child.watch(process, None, directory, [], 10, 5, 0, 1_048_576, Observer(directory, 1), child)
                self.assertEqual(child["stop_requested"], "resource_stop")
                self.assertEqual(load(directory / "worker-stop.json")["reason"], "resource_stop")
                events = (directory / "events.jsonl").read_text().count('"event":"cooperative_stop_requested"')
                self.assertEqual(events, 1)

    def test_late_stop_deadline_and_publication_allowance_fail_closed(self):
        for fault in ("event-deadline", "ready-overrun", "ready-stop", "clean"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                directory, now = Path(name), [1.0]
                context, report = terminal_fixture(directory)
                emit = context["observer"].emit

                def event(*args, **kwargs):
                    result = emit(*args, **kwargs)
                    if fault == "event-deadline":
                        now[0] = 11
                    return result

                def publish(path, value, **kwargs):
                    write(path, value, **kwargs)
                    if path.name == "release-ready.json":
                        if fault == "ready-overrun":
                            now[0] = 3
                        if fault == "ready-stop":
                            (directory / "STOP").touch()

                context["observer"].emit = event
                with patch.object(causal_terminal.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(causal_terminal, "write", side_effect=publish):
                    causal_terminal.finish(context, report)
                self.assertEqual(load(directory / "report.json"), report)
                self.assertEqual(report["status"], "completed" if fault == "clean" else "failed")
                self.assertEqual(report["release_ready"], fault == "clean")
                self.assertFalse((directory / "release-seal.json").exists())
                if fault.startswith("ready-"):
                    self.assertTrue((directory / "release-invalid.json").exists())

    def test_finite_limits_and_external_pin_reject_bad_inputs(self):
        limits = {"opportunities": 4, "wall_millis": 10_000, "stop_reserve_millis": 5000,
                  "checkpoint_every": 4, "evaluation_targets": 8, "generation_cases": 1, "generation_bytes": 32}
        causal_contract.limits(limits)
        for key, value in (("opportunities", True), ("wall_millis", 5000), ("generation_cases", 33), ("checkpoint_every", 5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                causal_contract.limits(limits | {key: value})
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "input.json").write_text("{}")
            with self.assertRaises(ValueError):
                causal_contract.pin(root, {"path": "input.json", "sha256": "0" * 64}, 1024)


if __name__ == "__main__":
    unittest.main()
