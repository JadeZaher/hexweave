"""Pinned visible prompts and zero-exposure receipts; see causal_OWNERSHIP.md."""

import hashlib
from pathlib import Path

from scripts.causal_observer import contained, load
from scripts.pulse_io import digest


def operation(value: dict) -> bool:
    if isinstance(value, dict) and value == {"kind": "train"}:
        return False
    if isinstance(value, dict) and value.get("kind") == "retained_assay":
        from scripts.causal_assay_study import operation
        operation(value)
        return False
    if not isinstance(value, dict) or set(value) != {"kind", "prompts"} or value["kind"] != "prompt_evaluation":
        raise ValueError("explicit causal operation required")
    pin = value["prompts"]
    if (not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not isinstance(pin["path"], str)
            or not valid_hash(pin["sha256"])):
        raise ValueError("prompt file pin required")
    return True


def read(root: Path, spec: dict, cases: int, new_bytes: int) -> tuple[Path, list]:
    original = root / spec["path"]
    for ancestor in (original, *original.parents):
        info = ancestor.lstat()
        if ancestor.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("prompt input path has a link/reparse point")
    path = contained(root, spec["path"])
    work = (root / "work").resolve(strict=True)
    if not path.is_relative_to(work) or path.parent == work:
        raise ValueError("prompt file must be in a workspace work subdirectory")
    if not path.is_file() or path.stat().st_size > 65_536 or digest(path) != spec["sha256"]:
        raise ValueError("prompt input pin/size mismatch")
    value = load(path, 65_536)
    rows = value.get("prompts")
    if (set(value) != {"schema", "prompts"} or type(value["schema"]) is not int or value["schema"] != 1
            or type(cases) is not int or not 1 <= cases <= 40 or not isinstance(rows, list) or len(rows) != cases
            or type(new_bytes) is not int or not 1 <= new_bytes <= 192):
        raise ValueError("prompt input schema/count/output bound mismatch")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "prompt"}:
            raise ValueError("prompt row can contain only id and visible prompt")
        key, prompt = row["id"], row["prompt"]
        if (not isinstance(key, str) or not 1 <= len(key) <= 64
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key)
                or key in seen or not isinstance(prompt, str) or not 1 <= len(prompt.encode("utf-8")) <= 832
                or len(prompt.encode("utf-8")) + new_bytes > 1024):
            raise ValueError("invalid prompt ID/bytes/context")
        seen.add(key)
    if digest(path) != spec["sha256"]:
        raise ValueError("prompt input changed during read")
    return path, rows


def same_science(left: dict, right: dict) -> None:
    if any(left.get(key) != right.get(key) for key in ("identity", "state", "shared", "predictor", "tiles")):
        raise ValueError("prompt evaluation changed checkpoint scientific payload")


def check(value: dict, worker: dict, result: dict, parent: dict, evaluated: dict, final: dict) -> None:
    limits = worker["limits"]
    _, rows = read(Path(worker["workspace"]), worker["operation"]["prompts"], limits["generation_cases"], limits["generation_bytes"])
    same_science(parent, evaluated)
    same_science(parent, final)
    expected = {"input_sha256", "parent_checkpoint", "evaluation_checkpoint", "before_sha256", "after_sha256", "requested_cases", "cases", "coverage"}
    if (not isinstance(value, dict) or set(value) != expected
            or value["input_sha256"] != worker["operation"]["prompts"]["sha256"]
            or value["parent_checkpoint"] != worker["resume"]["checkpoint"] or value["requested_cases"] != len(rows)
            or value["before_sha256"] != value["after_sha256"] or not valid_hash(value["before_sha256"])
            or result["physical_opportunities_completed"] != 0 or result["timings"]["training_millis"] != 0):
        raise ValueError("invalid zero-exposure prompt receipt")
    cases, coverage = value["cases"], value["coverage"]
    if not isinstance(cases, list) or len(cases) > len(rows) or not isinstance(coverage, dict):
        raise ValueError("invalid prompt coverage")
    if coverage == {"kind": "complete"}:
        if len(cases) != len(rows):
            raise ValueError("complete prompt receipt omitted cases")
    elif (coverage != {"kind": "interrupted", "status": result["status"]}
            or result["status"] not in ("user_stop", "resource_stop", "budget_stop")):
        raise ValueError("prompt receipt has incompatible partial coverage")
    for index, (case, row) in enumerate(zip(cases, rows)):
        check_case(case, row, value["before_sha256"], limits["generation_bytes"], coverage, index == len(cases) - 1)


def valid_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_case(case, row, state_hash, new_bytes, coverage, last):
    if (not isinstance(case, dict) or set(case) != {"id", "prompt_sha256", "generation"}
            or case["id"] != row["id"] or case["prompt_sha256"] != hashlib.sha256(row["prompt"].encode()).hexdigest()):
        raise ValueError("prompt result order/input identity changed")
    generation = case["generation"]
    if (not isinstance(generation, dict) or generation.get("before_sha256") != state_hash
            or generation.get("after_sha256") != state_hash):
        raise ValueError("prompt case mutated scientific state")
    raw = generation.get("raw_bytes")
    if not isinstance(raw, list) or len(raw) > new_bytes or any(type(byte) is not int or not 0 <= byte <= 255 for byte in raw):
        raise ValueError("invalid/unbounded prompt output bytes")
    try:
        bytes(raw).decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        valid = False
    if generation.get("valid_utf8") is not valid:
        raise ValueError("prompt UTF8 evidence mismatch")
    reason = generation.get("stop_reason")
    if reason == {"kind": "newline"}:
        if not raw or raw[-1] != 10 or 10 in raw[:-1]:
            raise ValueError("invalid newline termination")
    elif reason == {"kind": "byte_budget"}:
        if len(raw) != new_bytes or 10 in raw:
            raise ValueError("invalid byte-budget termination")
    elif not last or coverage.get("kind") != "interrupted" or reason != coverage or 10 in raw:
        raise ValueError("invalid prompt interrupted termination")
    check_routes(generation.get("routes"), len(raw))
    if type(generation.get("wall_micros")) is not int or not 0 <= generation["wall_micros"] <= 86_400_000_000:
        raise ValueError("invalid prompt generation time")


def check_routes(routes, count):
    if not isinstance(routes, dict) or set(routes) != {"visits", "route_calls", "halts", "forced_exits"}:
        raise ValueError("invalid prompt route evidence")
    visits = routes["visits"]
    values = [routes[key] for key in ("route_calls", "halts", "forced_exits")]
    if (not isinstance(visits, list) or len(visits) != 7
            or any(type(value) is not int or not 0 <= value <= 4 * count for value in visits + values)
            or not count <= sum(visits) <= 4 * count or routes["halts"] + routes["forced_exits"] != count
            or routes["route_calls"] + routes["forced_exits"] != sum(visits)):
        raise ValueError("prompt route counters disagree with bytes")
