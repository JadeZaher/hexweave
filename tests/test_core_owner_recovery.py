"""Deterministic owner recovery budgets, admission and STOP-race contracts."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import core_owner as owner
from scripts import core_owner_checkpoint as check
from scripts import core_owner_recovery as recovery


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CoreOwnerRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.clock = Clock()
        self.host = {"available_physical_bytes": owner.START_FLOOR + 1}
        build = self.root / "work/target/release/hexweave.exe"
        build.parent.mkdir(parents=True)
        build.write_bytes(b"current build")
        for patcher in (mock.patch.object(owner, "ROOT", self.root),
                        mock.patch.object(owner, "memory_status", return_value=self.host),
                        mock.patch.object(recovery.time, "monotonic", self.clock.monotonic),
                        mock.patch.object(recovery.time, "sleep", self.clock.sleep)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.case("run")

    def case(self, name):
        self.out = self.root / "work/core-runs" / name
        self.out.mkdir(parents=True)
        self.path = self.root / f"{name}.json"
        self.config = {"schema": 1, "out": f"work/core-runs/{name}", "resume": False,
                       "wall_seconds": 100, "anchors": 1000, "seed": 7, "connected": {}}
        owner.write(self.path, self.config)

    def recorded_attempt(self, path, config, out, binary, tag, anchors, reason, terminal, code=0, token=None):
        geometry = {"width": 16, "ffn": 48, "heads": 2, "context": 8}
        checkpoint = out / "checkpoints" / f"a{anchors:012d}"
        (checkpoint / "tiles").mkdir(parents=True)
        counters = {"local_updates": anchors * 8}
        owner.write(checkpoint / "connected.json", {"anchors": anchors, "geometry": geometry,
                    "cursor": {"supervised_targets": anchors}, "counters": counters,
                    "options": dict(config["connected"])})
        (checkpoint / "shared.bin").write_bytes(b"shared")
        tiles = []
        for index in range(7):
            tile = checkpoint / "tiles" / f"{index}.bin"
            tile.write_bytes(bytes([index, anchors % 256]))
            tiles.append({"hash": check.sha(tile), "revision": anchors, "version": anchors, "steps": anchors})
        owner.write(checkpoint / "manifest.json", {"schema": 1, "protocol": check.protocols(config)[-1],
                    "identity": "a" * 64, "geometry": geometry, "anchors": anchors, "tiles": tiles,
                    "common": {name: check.sha(checkpoint / name) for name in ("connected.json", "shared.bin")}})
        report = {"protocol": check.protocols(config)[-1], "run_identity": "a" * 64, "anchors": anchors,
                  "checkpoint": str(checkpoint), "terminal": terminal, "mechanism": counters}
        owner.write(out / "report.json", report)
        owner.write(out / f"owner-start-{tag}.json", {"config": str(path), "config_sha256": check.sha(path),
                    "binary_sha256": check.sha(binary), **({"memory_stop_token": token} if token else {})})
        owner.write(out / f"owner-end-{tag}.json", {"returncode": code, "reason": reason, "report": report})
        if reason == "owner_memory_stop":
            owner.request_stop(out, reason, token)
        return code

    def prior(self):
        binary = self.out / "bin/hexweave.exe"
        binary.parent.mkdir()
        binary.write_bytes(b"pinned build")
        source = self.out / "prior-config.json"
        owner.write(source, self.config)
        self.recorded_attempt(source, self.config, self.out, binary, "prior", 10,
                              "owner_memory_stop", "stopped")
        self.config["resume"] = True
        owner.write(self.path, self.config)
        return binary

    def status(self):
        return check.read(next(self.out.glob("owner-recovery-*.json")))

    def test_retry_keeps_lock_binary_options_and_remaining_budget(self):
        attempts = []
        original = self.path.read_bytes()
        def own(path, config, out, binary, host, attempt=None):
            self.assertTrue((out / "OWNER.lock").exists())
            self.assertEqual(binary.read_bytes(), b"current build")
            self.assertEqual(check.read(path), config)
            attempts.append(config)
            self.clock.now += 20.4 if len(attempts) == 1 else 10
            reason, terminal = ("owner_memory_stop", "stopped") if len(attempts) == 1 else ("deadline", "deadline")
            return self.recorded_attempt(path, config, out, binary, attempt, len(attempts) * 10,
                                         reason, terminal, token=attempt)
        with mock.patch.object(owner, "own", side_effect=own):
            self.assertEqual(owner.launch(self.path, False, recovery.Policy()), 0)
        self.assertEqual([config["resume"] for config in attempts], [False, True])
        self.assertEqual([config["wall_seconds"] for config in attempts], [100, 79])
        self.assertTrue(all(config["anchors"] == 1000 for config in attempts))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.status()["wait_seconds"], 30)
        self.assertAlmostEqual(self.status()["remaining_wall_seconds"], 69.6)
        self.assertEqual(len(list(self.out.glob("owner-start-*.json"))), 2)
        self.assertEqual(len(list(self.out.glob("owner-end-*.json"))), 2)
        self.assertEqual(len(list(self.out.glob("STOP.recovered-*"))), 1)
        self.assertEqual(len(list(self.out.glob("report.json.previous-*"))), 1)
        self.assertFalse((self.out / "OWNER.lock").exists())

    def test_initial_resume_refuses_corruption_changed_binary_and_options(self):
        for index, defect in enumerate(("checkpoint", "binary", "options")):
            with self.subTest(defect=defect):
                self.case(f"invalid-{index}")
                binary = self.prior()
                if defect == "checkpoint":
                    (self.out / "checkpoints/a000000000010/tiles/0.bin").write_bytes(b"corrupt")
                elif defect == "binary":
                    binary.unlink()
                else:
                    self.config["seed"] = 9
                    owner.write(self.path, self.config)
                before = (self.out / "STOP").read_bytes(), (self.out / "report.json").read_bytes()
                with mock.patch.object(owner, "own") as own:
                    self.assertRaises(ValueError, owner.launch, self.path, False, recovery.Policy())
                    own.assert_not_called()
                self.assertEqual(before, ((self.out / "STOP").read_bytes(), (self.out / "report.json").read_bytes()))
                self.assertFalse((self.out / "OWNER.lock").exists())

    def test_precision_admission_requires_config_report_manifest_and_state_agreement(self):
        # The fixture writes the newest generation (v5). Stamps override (report, newest manifest, older
        # manifest): "v1".."v4" are pre-bump runs with consistent strings and stay recoverable; "mixed" is a
        # report that disagrees with its newest manifest; "older-vN" puts generation N only in an older
        # checkpoint's manifest (v2-v4 under a v5 run, v5 under a v4 run). All mixed shapes are refused.
        v1, v2, v3, v4, v5 = ("causal-connected-fp32-v1", "causal-connected-fp32-v2", "causal-connected-fp32-v3",
                              "causal-connected-fp32-v4", "causal-connected-fp32-v5")
        ternary = "causal-connected-v5"
        stamps = {"v1": (v1, v1, v1), "v2": (v2, v2, v2), "v3": (v3, v3, v3), "v4": (v4, v4, v4),
                  "mixed": (None, v1, None), "older-v2": (None, None, v2), "older-v3": (None, None, v3),
                  "older-v4": (None, None, v4), "older-v5": (v4, v4, v5),
                  "report": (ternary, None, None), "manifest": (None, ternary, None)}
        admitted = (None, "v1", "v2", "v3", "v4")
        cases = [(None, None), ("ternary", None), ("fp32", None), ("fp32", "v1"), ("fp32", "v2"),
                 ("fp32", "v3"), ("fp32", "v4"), ("fp32", "mixed"), ("fp32", "older-v2"), ("fp32", "older-v3"),
                 ("fp32", "older-v4"), ("fp32", "older-v5"), ("fp32", "config"), ("fp32", "report"),
                 ("fp32", "manifest"), ("fp32", "state"), ("fp32", "payload"), ("fp32", "unknown")]
        for index, (precision, defect) in enumerate(cases):
            with self.subTest(precision=precision, defect=defect):
                self.case(f"precision-{index}")
                if precision is not None:
                    self.config["connected"]["precision"] = precision
                binary = self.prior()
                checkpoint = self.out / "checkpoints/a000000000010"
                report_stamp, manifest_stamp, older_stamp = stamps.get(defect, (None, None, None))
                if defect in ("config", "unknown"):
                    self.config["connected"]["precision"] = "ternary" if defect == "config" else "fp16"
                if report_stamp:
                    report = check.read(self.out / "report.json")
                    report["protocol"] = report_stamp
                    owner.write(self.out / "report.json", report)
                    end = check.read(self.out / "owner-end-prior.json")
                    end["report"] = report
                    owner.write(self.out / "owner-end-prior.json", end)
                manifest = check.read(checkpoint / "manifest.json")
                if defect == "state":
                    state = check.read(checkpoint / "connected.json")
                    state["options"]["precision"] = "ternary"
                    owner.write(checkpoint / "connected.json", state)
                    manifest["common"]["connected.json"] = check.sha(checkpoint / "connected.json")
                manifest["protocol"] = manifest_stamp or manifest["protocol"]
                owner.write(checkpoint / "manifest.json", manifest)
                older = checkpoint.with_name("a000000000005")  # an earlier complete checkpoint of this run
                older.mkdir()
                owner.write(older / "manifest.json", {"identity": "a" * 64,
                            "protocol": older_stamp or manifest["protocol"]})
                if defect == "payload":
                    (checkpoint / "tiles/0.bin").write_bytes(b"corrupt FP32 bundle")
                if defect not in admitted:
                    reason = "foreign checkpoint" if defect.startswith("older") else ""
                    with self.assertRaisesRegex(ValueError, reason):
                        check.inspect(self.config, self.out, binary)
                else:
                    report, _, _ = check.inspect(self.config, self.out, binary)
                    expected = {None: ternary, "ternary": ternary, "fp32": v5}[precision]
                    self.assertEqual(report["protocol"], stamps[defect][0] if defect else expected)

    def test_stable_wait_resets_and_timeout_or_interrupt_never_launches(self):
        samples = [self.host] * 29 + [{"available_physical_bytes": 0}] + [self.host] * 30
        with mock.patch.object(owner, "memory_status", side_effect=samples):
            host, waited = recovery.wait_ready(self.out, None, 100, lambda *_: None)
        self.assertEqual((host, waited), (self.host, 60))
        self.prior()
        stopped = (self.out / "STOP").read_bytes()
        with mock.patch.object(owner, "memory_status", return_value={"available_physical_bytes": 0}), \
                mock.patch.object(owner, "own") as own:
            self.assertEqual(owner.launch(self.path, False, recovery.Policy(wait_seconds=3)), 0)
            own.assert_not_called()
        self.assertEqual(self.status()["reason"], "recovery_wait_exhausted")
        self.assertEqual((self.out / "STOP").read_bytes(), stopped)
        with mock.patch.object(recovery.time, "sleep", side_effect=KeyboardInterrupt), \
                mock.patch.object(owner, "own") as own:
            self.assertEqual(owner.launch(self.path, False, recovery.Policy()), 130)
            own.assert_not_called()
        self.assertEqual((self.out / "STOP").read_bytes(), stopped)
        self.assertFalse((self.out / "OWNER.lock").exists())

    def test_stop_replacement_and_prelaunch_memory_drop_preserve_operator_control(self):
        binary = self.prior()
        expected = recovery.owned_stop(self.out, None)
        def replace_stop(seconds):
            self.clock.sleep(seconds)
            (self.out / "STOP").write_bytes(b"operator pause during wait")
        with mock.patch.object(recovery.time, "sleep", side_effect=replace_stop):
            self.assertRaises(recovery.Cancelled, recovery.wait_ready, self.out, expected, 100, lambda *_: None)
        self.assertEqual((self.out / "STOP").read_bytes(), b"operator pause during wait")
        (self.out / "STOP").write_bytes(expected[1])
        expected = recovery.owned_stop(self.out, None)
        with mock.patch.object(owner, "memory_status", return_value={"available_physical_bytes": 0}), \
                mock.patch.object(owner, "own") as own:
            self.assertRaises(recovery.MemoryUnavailable, recovery.attempt, self.out, "session", 1,
                              self.config, binary, check.sha(binary), expected)
            own.assert_not_called()
        self.assertEqual(recovery.stop_snapshot(self.out / "STOP"), expected)
        archive = owner.archive
        def race(path, label):
            path.write_bytes(b"operator pause")
            moved = archive(path, label)
            path.write_bytes(b"newer operator pause")
            return moved
        with mock.patch.object(owner, "archive", side_effect=race):
            self.assertRaises(recovery.Cancelled, recovery.archive_owned_stop, self.out, expected)
        self.assertEqual((self.out / "STOP").read_bytes(), b"newer operator pause")
        self.assertEqual(next(self.out.glob("STOP.recovered-*")).read_bytes(), b"operator pause")
        owner.request_stop(self.out, "owner_memory_stop", "token")
        self.assertEqual((self.out / "STOP").read_bytes(), b"newer operator pause")

    def test_checkpoint_corruption_during_wait_never_relaunches(self):
        self.prior()
        def sleep(seconds):
            self.clock.sleep(seconds)
            if self.clock.now == 1:
                (self.out / "checkpoints/a000000000010/tiles/0.bin").write_bytes(b"corrupt while waiting")
        with mock.patch.object(recovery.time, "sleep", side_effect=sleep), mock.patch.object(owner, "own") as own:
            self.assertRaises(ValueError, owner.launch, self.path, False, recovery.Policy())
            own.assert_not_called()
        self.assertTrue((self.out / "STOP").exists())
        self.assertTrue((self.out / "report.json").exists())

    def test_only_clean_memory_stops_retry_and_restarts_are_bounded(self):
        cases = [("owner_memory_stop then owner_killed_after_grace", "stopped", 1, 10, 1),
                 ("owner_error: OSError", "stopped", 1, 10, 1),
                 ("owner_interrupted", "stopped", 0, 10, 1),
                 ("completed", "completed", 0, 10, 1),
                 ("owner_memory_stop", "stopped", 0, 101, 1),
                 ("owner_memory_stop", "stopped", 0, 10, 2)]
        for index, (reason, terminal, code, duration, count) in enumerate(cases):
            with self.subTest(reason=reason, duration=duration):
                self.case(f"terminal-{index}")
                calls = []
                def own(path, config, out, binary, host, attempt=None):
                    calls.append(config)
                    self.clock.now += duration
                    anchors = config["anchors"] if terminal == "completed" else len(calls) * 10
                    return self.recorded_attempt(path, config, out, binary, attempt, anchors,
                                                 reason, terminal, code, attempt)
                with mock.patch.object(owner, "own", side_effect=own):
                    self.assertEqual(owner.launch(self.path, False, recovery.Policy(max_restarts=1)), code)
                self.assertEqual(len(calls), count)
                self.assertFalse((self.out / "OWNER.lock").exists())


if __name__ == "__main__":
    unittest.main()
