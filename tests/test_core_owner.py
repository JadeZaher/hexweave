"""Core owner pre-launch safety: run-directory rule, refusals that must leave a run untouched, resume prep."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import core_owner


class CoreOwnerPrepareTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        build = self.root / "work/target/release/hexweave.exe"
        build.parent.mkdir(parents=True)
        build.write_bytes(b"current build")
        self.out = self.root / "work/core-runs/run"
        self.out.mkdir(parents=True)
        patcher = mock.patch.object(core_owner, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_out_must_be_a_named_core_run(self):
        for out, accepted in [("work/core-runs/run", True), ("elsewhere/run", False),
                              ("work/core-runs/../../x", False), ("work/core-runs", False),
                              ("work/core-runs/..", False)]:
            with self.subTest(out=out):
                if accepted:
                    self.assertEqual(core_owner.run_directory({"out": out}), self.out)
                else:
                    self.assertRaises(SystemExit, core_owner.run_directory, {"out": out})

    def test_unsafe_launches_are_refused_without_side_effects(self):
        cases = [
            ("fresh over checkpoints", False, {"checkpoints/a000000000020/manifest.json": "{}"}),
            ("fresh over a pinned binary", False, {"bin/hexweave.exe": "older build"}),
            ("resume past a user STOP", True, {"STOP": "operator pause"}),
        ]
        for index, (name, resume, files) in enumerate(cases):
            with self.subTest(name):
                out = self.root / "work/core-runs" / f"case-{index}"
                for relative, text in files.items():
                    (out / relative).parent.mkdir(parents=True, exist_ok=True)
                    (out / relative).write_text(text, encoding="utf-8")
                before = sorted(p.relative_to(out) for p in out.rglob("*"))
                self.assertRaises(SystemExit, core_owner.prepare, {"resume": resume}, out, False)
                self.assertEqual(sorted(p.relative_to(out) for p in out.rglob("*")), before)

    def test_resume_keeps_pinned_binary_and_sets_aside_owner_stop_and_old_report(self):
        (self.out / "bin").mkdir()
        (self.out / "bin/hexweave.exe").write_bytes(b"pinned build")
        (self.out / "STOP").write_text("owner: owner_memory_stop\n", encoding="utf-8")
        (self.out / "report.json").write_text("{}", encoding="utf-8")
        binary = core_owner.prepare({"resume": True}, self.out, False)
        self.assertEqual(binary.read_bytes(), b"pinned build")
        self.assertFalse((self.out / "STOP").exists() or (self.out / "report.json").exists())
        self.assertEqual(len(list(self.out.glob("STOP.archived-*"))), 1)
        self.assertEqual(len(list(self.out.glob("report.json.previous-*"))), 1)

    def test_owner_error_stops_child_with_grace_and_keeps_the_earlier_reason(self):
        class Child:
            def __init__(self, exits):
                self.exits = exits

            def poll(self):
                return None

            def wait(self, timeout):
                if not self.exits:
                    raise core_owner.subprocess.TimeoutExpired("hexweave", timeout)
                return 0

        for exits, settled in [(True, True), (False, False)]:
            with self.subTest(exits=exits), mock.patch.object(core_owner, "GRACE_SECONDS", 0):
                (self.out / "STOP").unlink(missing_ok=True)
                self.assertEqual(core_owner.settle(Child(exits), self.out, "owner_error: OSError"), settled)
                self.assertEqual(core_owner.stop_reason(self.out), "owner_error: OSError")
        (self.out / "STOP").write_text("owner: owner_memory_stop\n", encoding="utf-8")
        core_owner.settle(Child(True), self.out, "owner_error: OSError")
        self.assertEqual(core_owner.stop_reason(self.out), "owner_memory_stop")
        (self.out / "report.json").write_text('{"terminal": "stop', encoding="utf-8")
        self.assertEqual(core_owner.read_report(self.out / "report.json"), {"unparseable": True})


if __name__ == "__main__":
    unittest.main()
