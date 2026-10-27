"""Bounded released-parent and external-seal fault fixtures."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import causal_checkpoint, causal_prepare, causal_seal
from scripts.causal_observer import encoded, load, write
from scripts.pulse_io import digest


def blob(directory, kind, payload):
    data = encoded(payload)
    hashed = hashlib.sha256(data).hexdigest()
    name = f"{kind}-{hashed}.bin"
    path = directory / name
    if not path.exists():
        path.write_bytes(data)
    return {"file": name, "sha256": hashed, "bytes": len(data)}


def released_fixture(root, protocol=None):
    directory = root / "work/causal/run/old"
    persistence = directory / "persistence"
    persistence.mkdir(parents=True)
    budget = {"maximum_opportunities": 100, "maximum_wall_millis": 20_000,
              "maximum_attempts": 4, "disk_bytes_per_attempt": 1_048_576}
    identity = {"source_sha256": "1" * 64, "runtime_sha256": "2" * 64}
    if protocol is not None:
        identity["protocol"] = protocol
    owner = {"schema": 1, "run_id": "run", "attempt_id": "old", "parent": None,
             "budget": budget, "consumed": {"opportunities": 4, "wall_millis": 1000, "attempts": 1},
             "attempt_start_opportunity": 0, "attempt_start_wall_millis": 0,
             "observer_sequence": 3, "stop": "running", "physical_work_unknown": False}
    manifest = {"schema": 1, "sequence": 1, "identity": identity, "owner": owner,
                "state": blob(persistence, "state", {"boundary": "ready", "global_opportunity": 4}),
                "shared": blob(persistence, "shared", {"value": 0}),
                "predictor": blob(persistence, "predictor", {"value": 0}), "io": {},
                "tiles": [{"id": index, "version": 1, "blob": blob(persistence, "tile", {"value": index})} for index in range(7)]}
    payload = encoded(manifest)
    hashed = hashlib.sha256(payload).hexdigest()
    (persistence / f"manifest-{hashed}.bin").write_bytes(payload)
    reference = {"sequence": 1, "manifest_sha256": hashed}
    write(persistence / f"commit-{1:020}-{hashed}.json", reference)
    release = {"schema": 1, "run_id": "run", "attempt_id": "old", "owner_identity": {"pid": 10, "creation_filetime": 20},
               "checkpoint": reference, "status": "completed", "consumed": owner["consumed"] | {"wall_millis": 2000},
               "observer_sequence": 4, "child_reaped": True, "observer_acknowledged": True, "physical_work_unknown": False}
    proof = {"schema": 1, "release_sha256": "", "owner_identity": release["owner_identity"], "owner_reaped": True,
             "owner_exit_code": 0, "external_wall_millis": 1500, **identity,
             "finalizer_identity": {"pid": 11, "creation_filetime": 21}}
    resume = {"parent_directory": str(directory), "release_sha256": "", "release_seal_sha256": "",
              "checkpoint": reference, "explicit_user_renewal": False}
    republish(directory, release, proof, resume)
    return directory, identity, budget, release, proof, resume


def republish(directory, release, proof, resume):
    write(directory / "release.json", release)
    resume["release_sha256"] = digest(directory / "release.json")
    write(directory / "release-ready.json", {"schema": 1, "release_sha256": resume["release_sha256"]})
    proof["release_sha256"] = resume["release_sha256"]
    write(directory / "release-seal.json", proof)
    resume["release_seal_sha256"] = digest(directory / "release-seal.json")


class CausalResumeTests(unittest.TestCase):
    def test_release_seal_and_cumulative_evidence_fault_table(self):
        faults = ("missing-seal", "bad-seal-pin", "invalid", "unknown-work", "bad-exit", "over-wall",
                  "source", "old-protocol", "live-owner", "same-finalizer", "budget-stop", "user-stop", "regressed", "blob-change")
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                directory, identity, budget, release, proof, resume = released_fixture(root, "causal-prefix-v2" if fault == "old-protocol" else None)
                if fault == "unknown-work":
                    release["physical_work_unknown"] = True
                elif fault == "bad-exit":
                    proof["owner_exit_code"] = 1
                elif fault == "over-wall":
                    proof["external_wall_millis"] = 2001
                elif fault == "source":
                    proof["source_sha256"] = "3" * 64
                elif fault == "old-protocol":
                    identity = identity | {"protocol": "causal-prefix-v3"}
                elif fault == "same-finalizer":
                    proof["finalizer_identity"] = proof["owner_identity"]
                elif fault in ("budget-stop", "user-stop"):
                    release["status"] = fault.replace("-", "_")
                elif fault == "regressed":
                    release["consumed"]["opportunities"] = 3
                republish(directory, release, proof, resume)
                if fault == "missing-seal":
                    (directory / "release-seal.json").unlink()
                elif fault == "bad-seal-pin":
                    resume["release_seal_sha256"] = "0" * 64
                elif fault == "invalid":
                    write(directory / "release-invalid.json", {"schema": 1})
                elif fault == "blob-change":
                    next((directory / "persistence").glob("tile-*.bin")).write_bytes(b"changed")
                with patch.object(causal_checkpoint, "identity_absent", return_value=fault != "live-owner"), \
                        self.assertRaises((ValueError, FileNotFoundError)):
                    causal_checkpoint.parent(root, resume, identity, budget)

    def test_explicit_renewal_and_single_external_claim_leave_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory, identity, budget, release, proof, resume = released_fixture(root)
            release["status"] = "user_stop"
            resume["explicit_user_renewal"] = True
            republish(directory, release, proof, resume)
            before = {path.relative_to(directory): digest(path) for path in directory.rglob("*") if path.is_file()}
            with patch.object(causal_checkpoint, "identity_absent", return_value=True):
                inherited = causal_checkpoint.parent(root, resume, identity, budget)
                context = {"parent": inherited, "envelope": {"attempt_id": "new"}}
                request = {"resume": resume, "identity": identity, "budget": budget}
                causal_prepare.claim(root, request, context)
                with self.assertRaises(FileExistsError):
                    causal_prepare.claim(root, request, context)
            claim = directory.parent / "claims" / (resume["release_sha256"] + ".json")
            self.assertEqual(load(claim)["new_attempt_id"], "new")
            after = {path.relative_to(directory): digest(path) for path in directory.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_external_seal_requires_reaped_zero_owner_and_charged_wall(self):
        for fault in ("clean", "live", "exit", "wall", "source-change"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                directory, identity, _, release, _, _ = released_fixture(root)
                (directory / "release-seal.json").unlink()
                write(directory / "report.json", {"release_ready": True, "release_sha256": digest(directory / "release.json"),
                      "status": release["status"], "consumed": release["consumed"], "attempt_start_wall_millis": 0,
                      "effective_deadline_millis": 10_000})
                write(directory / "request.json", {"identity": identity})
                process = SimpleNamespace(poll=lambda: None if fault == "live" else 0,
                                          wait=lambda timeout: 1 if fault == "exit" else 0)
                with patch.object(causal_seal.causal_resources, "process_identity", return_value=release["owner_identity"]), \
                        patch.object(causal_seal.causal_contract, "validate", side_effect=ValueError("changed") if fault == "source-change" else None):
                    args = (root, directory, process, release["owner_identity"], 2001 if fault == "wall" else 1500,
                            {"pid": 11, "creation_filetime": 21})
                    if fault == "clean":
                        result = causal_seal.seal(*args)
                        self.assertEqual(load(directory / "release-seal.json"), result)
                    else:
                        with self.assertRaises(ValueError):
                            causal_seal.seal(*args)
                        self.assertFalse((directory / "release-seal.json").exists())


if __name__ == "__main__":
    unittest.main()
