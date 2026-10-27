"""Ready checkpoint and released-parent proof checks; see causal_OWNERSHIP.md."""

from pathlib import Path

from scripts.causal_contract import integer, sha
from scripts.causal_resources import identity_absent
from scripts.causal_observer import contained, load
from scripts.pulse_io import digest


def checkpoint(directory: Path, reference: dict, identity: dict) -> tuple[dict, dict]:
    if (not isinstance(reference, dict) or set(reference) != {"sequence", "manifest_sha256"}
            or not integer(reference["sequence"], 1, 2**64 - 1) or not sha(reference["manifest_sha256"])):
        raise ValueError("invalid checkpoint reference")
    tag = reference["manifest_sha256"]
    marker = contained(directory, f"commit-{reference['sequence']:020}-{tag}.json")
    if load(marker, 1024) != reference:
        raise ValueError("checkpoint lacks exact committed marker")
    manifest_path = contained(directory, f"manifest-{tag}.bin")
    manifest = load(manifest_path, 16_384)
    if digest(manifest_path) != tag or manifest.get("schema") != 1 or manifest.get("sequence") != reference["sequence"] or manifest.get("identity") != identity:
        raise ValueError("checkpoint manifest identity mismatch")
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != 7 or any(tile.get("id") != index for index, tile in enumerate(tiles)):
        raise ValueError("checkpoint requires seven ordered tiles")
    for kind, blob in [(name, manifest[name]) for name in ("state", "shared", "predictor")] + [("tile", tile["blob"]) for tile in tiles]:
        if (set(blob) != {"file", "sha256", "bytes"} or not sha(blob["sha256"])
                or not integer(blob["bytes"], 1, 131_072) or blob["file"] != f"{kind}-{blob['sha256']}.bin"):
            raise ValueError("noncanonical checkpoint blob")
        path = contained(directory, blob["file"])
        if path.stat().st_size != blob["bytes"] or digest(path) != blob["sha256"]:
            raise ValueError("checkpoint blob identity mismatch")
    state = load(contained(directory, manifest["state"]["file"]), 65_536)
    if state.get("boundary") != "ready" or not integer(state.get("global_opportunity"), 0, 2**64 - 1):
        raise ValueError("checkpoint is not a complete Ready boundary")
    from scripts import causal_assay, causal_assay_result
    if identity.get("protocol") == causal_assay.PROTOCOL:
        causal_assay_result.state(state, identity)
    elif "assay" in state:
        raise ValueError("v3 checkpoint contains assay state")
    return manifest, state


def parent(root: Path, resume: dict, identity: dict, budget: dict) -> tuple[Path, dict, dict, dict]:
    if (not isinstance(resume, dict) or set(resume) != {"parent_directory", "release_sha256", "release_seal_sha256", "checkpoint", "explicit_user_renewal"}
            or not sha(resume["release_sha256"]) or not sha(resume["release_seal_sha256"]) or type(resume["explicit_user_renewal"]) is not bool):
        raise ValueError("explicit hash-bound parent resume request required")
    directory = contained(root / "work/causal", resume["parent_directory"])
    release_path = directory / "release.json"
    release = load(release_path, 16_384)
    ready = load(directory / "release-ready.json", 1024)
    if ((directory / "release-invalid.json").exists() or digest(release_path) != resume["release_sha256"]
            or ready != {"schema": 1, "release_sha256": resume["release_sha256"]}):
        raise ValueError("parent release is absent, invalidated or changed")
    if (release.get("schema") != 1 or release.get("child_reaped") is not True or release.get("observer_acknowledged") is not True
            or release.get("physical_work_unknown") is not False or release.get("status") not in ("completed", "user_stop", "resource_stop")
            or release.get("checkpoint") != resume["checkpoint"]):
        raise ValueError("terminal parent cannot resume")
    manifest, state = checkpoint(directory / "persistence", resume["checkpoint"], identity)
    previous = manifest["owner"]
    seal_path = directory / "release-seal.json"
    seal = load(seal_path, 4096)
    if (digest(seal_path) != resume["release_seal_sha256"] or seal.get("schema") != 1
            or seal.get("release_sha256") != resume["release_sha256"] or seal.get("owner_identity") != release.get("owner_identity")
            or seal.get("owner_reaped") is not True or type(seal.get("owner_exit_code")) is not int or seal["owner_exit_code"] != 0
            or seal.get("source_sha256") != identity["source_sha256"] or seal.get("runtime_sha256") != identity["runtime_sha256"]
            or not integer(seal.get("external_wall_millis"), 1, release["consumed"]["wall_millis"] - previous["attempt_start_wall_millis"])):
        raise ValueError("parent lacks pinned external exact-owner release seal")
    if (seal.get("finalizer_identity") == release.get("owner_identity")
            or not identity_absent(release.get("owner_identity")) or not identity_absent(seal.get("finalizer_identity"))):
        raise ValueError("parent owner/finalizer identity is still active")
    if (previous["budget"] != budget or release["run_id"] != previous["run_id"]
            or release["attempt_id"] != previous["attempt_id"] or directory.name != previous["attempt_id"]
            or previous["stop"] in ("failed", "budget_stop") or previous["physical_work_unknown"] is not False):
        raise ValueError("parent checkpoint lineage/budget mismatch")
    if (release["status"] == "user_stop" or previous["stop"] == "user_stop") and not resume["explicit_user_renewal"]:
        raise ValueError("user STOP requires explicit renewal")
    for key in ("opportunities", "wall_millis", "attempts"):
        value = release["consumed"].get(key)
        if not integer(value, previous["consumed"][key], budget["maximum_" + key]):
            raise ValueError("parent cumulative budget evidence regressed")
    if release["consumed"]["attempts"] != previous["consumed"]["attempts"] or release["observer_sequence"] < previous["observer_sequence"]:
        raise ValueError("parent attempt/observer evidence changed")
    if (directory / "release-invalid.json").exists():
        raise ValueError("parent release invalidated during readback")
    return directory, release, manifest, state
