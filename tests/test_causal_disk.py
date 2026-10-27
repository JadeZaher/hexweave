"""Live publication disappearance remains narrowly bounded and terminal sampling strict."""

import contextlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import causal_disk


class CausalDiskTests(unittest.TestCase):
    def test_external_study_scan_allows_only_live_owner_control_publication(self):
        for owner, filename in ((456, "report.json.pending"), (None, "report.json.pending"), (456, "foreign.pending")):
            with self.subTest(owner=owner, filename=filename), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                path = root / filename
                path.write_bytes(b"owner")
                with self.disappearing(root, path):
                    if owner == 456 and filename == "report.json.pending":
                        self.assertEqual(causal_disk.disk_bytes(root, live_owner_pid=owner), causal_disk.PUBLICATION_CAP)
                    else:
                        with self.assertRaises(FileNotFoundError):
                            causal_disk.disk_bytes(root, live_owner_pid=owner)

    def test_observation_publication_disappearance_requires_exact_live_writer(self):
        for pid in (123, None, 124):
            with self.subTest(pid=pid), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                persistence = self.fixture(root)
                pending = root / f".child-123-assay-observation-{'a' * 64}.json"
                pending.write_bytes(b"payload")
                with self.disappearing(root, pending):
                    if pid == 123:
                        self.assertEqual(causal_disk.disk_bytes(root, pid),
                            (persistence / "OWNED").stat().st_size + causal_disk.PUBLICATION_CAP)
                    else:
                        with self.assertRaises(FileNotFoundError):
                            causal_disk.disk_bytes(root, pid)

    def fixture(self, root: Path) -> Path:
        persistence = root / "persistence"
        persistence.mkdir()
        (persistence / "OWNED").write_text(f"causal-persistence-v1\n{root}\n", newline="")
        return persistence

    def disappearing(self, root: Path, path: Path, error=None):
        original = os.scandir

        def scan(parent):
            with original(parent) as iterator:
                entries = list(iterator)
            if Path(parent) == path.parent and any(Path(e.path) == path for e in entries):
                if error is None:
                    path.unlink()
                fault = FileNotFoundError(str(path)) if error is None else error
                entries = [SimpleNamespace(path=e.path, stat=lambda **_: (_ for _ in ()).throw(fault))
                           if Path(e.path) == path else e for e in entries]
            return contextlib.nullcontext(entries)
        return patch.object(causal_disk.os, "scandir", side_effect=scan)

    def test_vanished_exact_live_temporary_reserves_unenumerated_final_payload(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            persistence = self.fixture(root)
            pending = persistence / ".pending-123-0"
            pending.write_bytes(b"payload")
            baseline = (persistence / "OWNED").stat().st_size
            with self.disappearing(root, pending) as observed:
                self.assertEqual(causal_disk.disk_bytes(root, 123), baseline + causal_disk.PUBLICATION_CAP)
                self.assertEqual(sum(Path(call.args[0]) == root for call in observed.call_args_list), 2)
            self.assertEqual(causal_disk.disk_bytes(root), baseline)

    def test_present_temporary_and_final_hardlink_are_both_charged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            persistence = self.fixture(root)
            pending = persistence / ".pending-123-1"
            pending.write_bytes(b"payload")
            os.link(pending, persistence / "tile-final.bin")
            self.assertEqual(causal_disk.disk_bytes(root, 123), (persistence / "OWNED").stat().st_size + 14)

    def test_retention_removal_requires_stable_second_scan(self):
        names = [f"{kind}-{'a' * 64}.bin" for kind in ("tile", "shared", "predictor", "state", "manifest")]
        names.append(f"commit-{7:020}-{'a' * 64}.json")
        for filename in names:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                persistence = self.fixture(root)
                path = persistence / filename
                path.write_bytes(b"retired")
                with self.disappearing(root, path) as observed:
                    self.assertEqual(causal_disk.disk_bytes(root, 123),
                                     (persistence / "OWNED").stat().st_size + causal_disk.PUBLICATION_CAP)
                    self.assertEqual(sum(Path(call.args[0]) == root for call in observed.call_args_list), 2)
                path.write_bytes(b"retired")
                with self.disappearing(root, path), self.assertRaises(FileNotFoundError):
                    causal_disk.disk_bytes(root)

    def test_second_scan_churn_and_combined_entry_budget_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            persistence = self.fixture(root)
            path = persistence / ".pending-123-0"
            path.write_bytes(b"x")
            with self.disappearing(root, path), patch.object(causal_disk, "ENTRY_CAP", 4), self.assertRaises(ValueError):
                causal_disk.disk_bytes(root, 123)
            info = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0)
            directory = SimpleNamespace(path=str(persistence), stat=lambda **_: info)
            missing = SimpleNamespace(path=str(path), stat=lambda **_: (_ for _ in ()).throw(FileNotFoundError()))
            def scan(parent):
                return contextlib.nullcontext([directory] if Path(parent) == root else [missing])
            with patch.object(causal_disk.os, "scandir", side_effect=scan), self.assertRaises(FileNotFoundError):
                causal_disk.disk_bytes(root, 123)

    def test_wrong_identity_path_name_and_terminal_missing_entries_fail_closed(self):
        cases = [("persistence", ".pending-123-1", None), ("persistence", ".pending-124-1", 123),
                 ("persistence", ".pending-123-x", 123), ("persistence", ".pending-123-01", 123),
                 ("persistence", "tile-final.bin", 123), ("other", ".pending-123-1", 123)]
        for folder, filename, pid in cases:
            with self.subTest(folder=folder, filename=filename, pid=pid), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                self.fixture(root)
                parent = root / folder
                parent.mkdir(exist_ok=True)
                path = parent / filename
                path.write_bytes(b"x")
                with self.disappearing(root, path), self.assertRaises(FileNotFoundError):
                    causal_disk.disk_bytes(root, pid)

    def test_permission_error_and_invalid_marker_are_not_suppressed(self):
        for fault in ("permissions", "marker"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                persistence = self.fixture(root)
                path = persistence / ".pending-123-0"
                path.write_bytes(b"x")
                if fault == "marker":
                    (persistence / "OWNED").write_bytes(b"wrong")
                error = PermissionError("denied") if fault == "permissions" else None
                with self.disappearing(root, path, error), self.assertRaises((PermissionError, ValueError)):
                    causal_disk.disk_bytes(root, 123)

    def test_reparse_entries_and_disappeared_entry_count_still_enforce_bounds(self):
        for fault in ("reparse", "count"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                persistence = self.fixture(root)
                path = persistence / ".pending-123-0"
                path.write_bytes(b"x")
                if fault == "count":
                    with self.disappearing(root, path), patch.object(causal_disk, "ENTRY_CAP", 2), self.assertRaises(ValueError):
                        causal_disk.disk_bytes(root, 123)
                else:
                    info = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
                    entry = SimpleNamespace(path=str(path), stat=lambda **_: info)
                    with patch.object(causal_disk.os, "scandir", return_value=contextlib.nullcontext([entry])), self.assertRaises(ValueError):
                        causal_disk.disk_bytes(root, 123)


if __name__ == "__main__":
    unittest.main()
