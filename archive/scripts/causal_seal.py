"""External exact-owner reap proof required for resume; see causal_OWNERSHIP.md."""

from scripts import causal_contract, causal_resources
from scripts.causal_observer import contained, load, write
from scripts.pulse_io import digest


def seal(root, directory, process, owner_identity: dict, external_wall_millis: int, finalizer_identity: dict) -> dict:
    root = root.resolve(strict=True)
    directory = contained(root / "work/causal", str(directory))
    if (not causal_contract.integer(external_wall_millis, 1, 86_400_000) or process.poll() is None
            or process.wait(timeout=0) != 0 or causal_resources.process_identity(process) != owner_identity):
        raise ValueError("external finalizer requires exact owner handle reaped with exit zero")
    report = load(directory / "report.json")
    release_path = directory / "release.json"
    release = load(release_path, 16_384)
    release_hash = digest(release_path)
    if (report.get("release_ready") is not True or report.get("release_sha256") != release_hash
            or report.get("status") != release.get("status") or report.get("consumed") != release.get("consumed")
            or release.get("owner_identity") != owner_identity or release.get("child_reaped") is not True
            or release.get("observer_acknowledged") is not True or release.get("physical_work_unknown") is not False):
        raise ValueError("pending owner release lacks final successful evidence")
    if load(directory / "release-ready.json", 1024) != {"schema": 1, "release_sha256": release_hash} or (directory / "release-invalid.json").exists():
        raise ValueError("pending release is absent or invalid")
    charged = release["consumed"]["wall_millis"] - report["attempt_start_wall_millis"]
    if external_wall_millis > charged or external_wall_millis > report["effective_deadline_millis"]:
        raise ValueError("external owner process exceeded charged or hard wall allowance")
    request = load(directory / "request.json", 65_536)
    causal_contract.validate(root, request)
    identity = request["identity"]
    value = {"schema": 1, "release_sha256": release_hash, "owner_identity": owner_identity,
             "owner_reaped": True, "owner_exit_code": 0, "external_wall_millis": external_wall_millis,
             "source_sha256": identity["source_sha256"], "runtime_sha256": identity["runtime_sha256"],
             "finalizer_identity": finalizer_identity}
    write(directory / "release-seal.json", value, immutable=True)
    return value
