"""Read-only admission for automatic connected-run recovery; see scripts/AGENTS.md."""

import hashlib
import json
import re
from pathlib import Path

from scripts import core_owner as owner

# Accepted generations per precision, oldest first. A generation stays admitted while runs of it
# exist; the pinned binary enforces its own protocol.
PROTOCOLS = {"ternary": ("causal-connected-v1", "causal-connected-v2", "causal-connected-v3",
                         "causal-connected-v4", "causal-connected-v5"),
             "fp32": ("causal-connected-fp32-v1", "causal-connected-fp32-v2", "causal-connected-fp32-v3",
                      "causal-connected-fp32-v4", "causal-connected-fp32-v5")}


def protocols(config: dict) -> tuple[str, ...]:
    options = config.get("connected")
    precision = options.get("precision", "ternary") if isinstance(options, dict) else None
    if not isinstance(precision, str) or precision not in PROTOCOLS:
        raise ValueError("unknown connected precision in recovery configuration")
    return PROTOCOLS[precision]


def sha(path: Path, limit: int = 16 * 1024**2) -> str:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized recovery artifact: {path}")
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path, limit: int = 16 * 1024**2) -> dict:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized recovery document: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid recovery document: {path}")
    return value


def unchanged_config(previous: dict, current: dict) -> bool:
    excluded = {"resume", "wall_seconds"}
    return ({key: value for key, value in previous.items() if key not in excluded}
            == {key: value for key, value in current.items() if key not in excluded})


def latest_end(out: Path) -> Path:
    paths = list(out.glob("owner-end-*.json"))
    if not paths:
        raise ValueError("memory recovery requires an owner end receipt")
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))


def checkpoint(out: Path, report: dict, expected_protocol: str) -> Path:
    anchors = report.get("anchors")
    identity = report.get("run_identity")
    if type(anchors) is not int or anchors < 0 or not re.fullmatch(r"[0-9a-f]{64}", identity or ""):
        raise ValueError("invalid recovery report counters or identity")
    path = Path(report["checkpoint"])
    path = (path if path.is_absolute() else owner.ROOT / path).resolve(strict=True)
    root = (out / "checkpoints").resolve(strict=True)
    if path.parent != root or path.name != f"a{anchors:012d}":
        raise ValueError("recovery checkpoint is outside its run or has the wrong counter")
    complete = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"a[0-9]{12}", p.name))
    if not complete or complete[-1].resolve() != path:
        raise ValueError("recovery report does not reference the newest complete checkpoint")
    manifest = read(path / "manifest.json", 1024**2)
    if (manifest.get("schema") != 1 or manifest.get("protocol") != expected_protocol
            or manifest.get("identity") != identity or manifest.get("anchors") != anchors):
        raise ValueError("recovery manifest identity or counters disagree")
    for other in complete:
        older = read(other / "manifest.json", 1024**2)
        if older.get("identity") != identity or older.get("protocol") != manifest["protocol"]:
            raise ValueError("foreign checkpoint in recovery run")
    tiles, common = manifest.get("tiles", []), manifest.get("common", {})
    if len(tiles) != 7 or set(common) != {"connected.json", "shared.bin"}:
        raise ValueError("incomplete recovery checkpoint")
    expected = {f"tiles/{index}.bin": meta["hash"] for index, meta in enumerate(tiles)} | common
    for relative, expected_hash in expected.items():
        artifact = path / relative
        if artifact.resolve(strict=True) != path.resolve().joinpath(relative):
            raise ValueError("recovery payload escapes checkpoint")
        if sha(artifact) != expected_hash:
            raise ValueError(f"corrupt recovery payload: {relative}")
    state = read(path / "connected.json")
    options = state.get("options", {})
    precision = options.get("precision", "ternary") if isinstance(options, dict) else None
    if not isinstance(precision, str) or expected_protocol not in PROTOCOLS.get(precision, ()):
        raise ValueError("recovery common-state precision disagrees with configuration")
    if (state.get("anchors") != anchors or state.get("geometry") != manifest.get("geometry")
            or state.get("cursor", {}).get("supervised_targets") != anchors
            or state.get("counters") != report.get("mechanism")):
        raise ValueError("recovery common state disagrees with report")
    return path


def inspect(config: dict, out: Path, binary: Path, receipt: Path | None = None) -> tuple[dict, str | None, str]:
    accepted = protocols(config)
    end_path = receipt or latest_end(out)
    end = read(end_path)
    report = read(out / "report.json")
    if (end.get("returncode") != 0 or end.get("reason") != "owner_memory_stop"
            or end.get("report") != report or report.get("terminal") != "stopped"
            or report.get("protocol") not in accepted):
        raise ValueError("recovery requires an exact clean owner_memory_stop and stopped report")
    start = read(end_path.with_name(end_path.name.replace("owner-end-", "owner-start-", 1)))
    if sha(binary) != start.get("binary_sha256"):
        raise ValueError("recovery pinned binary is missing or changed")
    source = Path(start["config"])
    if not source.is_absolute():
        source = owner.ROOT / source
    if sha(source, 1024**2) != start.get("config_sha256"):
        raise ValueError("recovery source config changed since its owner receipt")
    if not unchanged_config(read(source, 1024**2), config):
        raise ValueError("recovery changes options beyond resume and remaining wall_seconds")
    # Report, newest manifest, older manifests and saved precision must all name this protocol.
    target = checkpoint(out, report, report["protocol"])
    return report, start.get("memory_stop_token"), sha(target / "manifest.json", 1024**2)
