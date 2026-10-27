"""Reconcile exact assay observations with persisted physical exposure."""

import hashlib
import math
import struct
from pathlib import Path

from scripts import causal_assay as assay
from scripts.causal_contract import integer, sha
from scripts.causal_observer import contained, encoded, load
from scripts.pulse_io import digest


def probability(metrics: dict, n: int, rule: str, transitions: int) -> None:
    keys = {"attempts", "clamped", "transitions", "probability_sum_bits"}
    if not isinstance(metrics, dict) or set(metrics) != keys or any(not integer(metrics[key], 0, 2**64 - 1) for key in keys):
        raise ValueError("invalid assay E ledger encoding")
    value = struct.unpack("<d", struct.pack("<Q", metrics["probability_sum_bits"]))[0]
    if (metrics["attempts"] != metrics["clamped"] + metrics["transitions"]
            or metrics["transitions"] != transitions or metrics["attempts"] > n * 32768
            or not math.isfinite(value) or not 0 <= value <= n * 32768
            or (rule == "dense" and value != metrics["attempts"])):
        raise ValueError("assay E transition/probability conservation")


def config_identity(seed, rule: str, identity: dict) -> None:
    if not integer(seed, 0, 2**64 - 1) or seed not in (7, 17, 29):
        raise ValueError("assay checkpoint seed missing or malformed")
    config = {"schema": 1, "protocol": assay.PROTOCOL, "profile": "retained-" + rule,
              "seed": seed, "threads": 1, "anchor_interval": 1, "max_visits": 4,
              "history_capacity": 8, "context_bytes": 1024,
              "evidence_rule": "threshold-one-canonical-zero-v1", "route_credit": "detached-u-experimental",
              "predictor_rule": "joint-adjacent-raw-row-mean-v1", "anchor_batch": "current-anchor-forward-v1",
              "replay_selection": "none-anchor-only-v1"}
    if hashlib.sha256(encoded(config)).hexdigest() != identity.get("config_sha256"):
        raise ValueError("assay checkpoint seed/rule config identity mismatch")


def state(value: dict, identity: dict) -> None:
    n = value.get("global_opportunity")
    if value.get("schema") != 2 or value.get("cursor") != assay.cursor(n):
        raise ValueError("assay physical cursor/schema mismatch")
    progress = value.get("assay")
    if (not isinstance(progress, dict) or set(progress) != {"schema", "rule", "embedding", "observations"}
            or progress["schema"] != 1 or progress["rule"] not in ("dense", "magnitude")):
        raise ValueError("assay descriptor missing or malformed")
    config_identity(value.get("seed"), progress["rule"], identity)
    counts = value["counters"]
    if counts.get("anchors") != n or any(counts.get(key) != 0 for key in
        ("local_opportunities", "local_prefix_bytes", "local_tile_updates", "predictor_replay_updates", "predictor_replay_pairs")):
        raise ValueError("AO physical ledger contains local work")
    probability(progress["embedding"], n, progress["rule"], counts["shared_transitions"])
    refs = progress["observations"]
    if not isinstance(refs, list) or len(refs) > 8:
        raise ValueError("assay observation ledger count")
    for index, ref in enumerate(refs):
        if (not isinstance(ref, dict) or set(ref) != {"pass", "sha256", "bytes", "complete", "teacher_readouts", "generated_readouts"}
                or ref["pass"] != assay.PASSES[index] or ref["pass"] * 80 > n
                or not sha(ref["sha256"]) or not integer(ref["bytes"], 1, 65_536)
                or type(ref["complete"]) is not bool or not integer(ref["teacher_readouts"], 0, 240)
                or not integer(ref["generated_readouts"], 0, 384)
                or (ref["pass"] not in assay.GENERATIONS and ref["generated_readouts"] != 0)
                or (ref["complete"] and ref["teacher_readouts"] != 240)
                or (not ref["complete"] and index + 1 != len(refs))):
            raise ValueError("assay observation ledger invalid")
    due = sum(p * 80 < n for p in assay.PASSES)
    if len(refs) < due or not all(ref["complete"] for ref in refs[:due]):
        raise ValueError("missing earlier assay observation")
    if identity["data"] != assay.identity():
        raise ValueError("assay checkpoint data identity")


def case(value: dict, row: dict, generate: bool) -> tuple[int, int, bool]:
    fields = {"id", "content_nll", "lf_nll", "first_byte_rank", "generated", "generation_complete", "exact"}
    if not isinstance(value, dict) or set(value) != fields or value["id"] != row["id"]:
        raise ValueError("assay case order/schema mismatch")
    losses, lf = value["content_nll"], value["lf_nll"]
    if not isinstance(losses, list) or len(losses) > 4:
        raise ValueError("assay content coverage")
    for number in losses + ([] if lf is None else [lf]):
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("invalid assay NLL")
    teacher = len(losses) + int(lf is not None)
    if (lf is not None and len(losses) != 4) or (teacher > 0 and not integer(value["first_byte_rank"], 1, 256)):
        raise ValueError("assay teacher order/rank")
    raw = value["generated"]
    if type(value["generation_complete"]) is not bool:
        raise ValueError("assay generation coverage encoding")
    if raw is not None:
        if (not generate or teacher != 5 or not isinstance(raw, list) or len(raw) > 8
                or any(not integer(byte, 0, 255) for byte in raw)
                or (10 in raw and raw.index(10) != len(raw) - 1)):
            raise ValueError("assay raw generation bytes")
    if value["generation_complete"]:
        if raw is None or not raw or (raw[-1] != 10 and len(raw) != 8) or type(value["exact"]) is not bool:
            raise ValueError("assay complete generation missing stop")
        if value["exact"] != (bytes(raw) == row["completion"].encode("ascii")):
            raise ValueError("assay exact match claim changed")
    elif value["exact"] is not None:
        raise ValueError("censored assay generation has exact-match claim")
    return teacher, len(raw or []), teacher == 5 and (not generate or value["generation_complete"])


def observation(directory: Path, reference: dict, worker: dict, rows: list[dict]) -> dict:
    path = contained(directory, f"assay-observation-{reference['sha256']}.json")
    value = load(path, 65_536)
    if path.stat().st_size != reference["bytes"] or digest(path) != reference["sha256"]:
        raise ValueError("assay immutable observation changed")
    fixed = {"schema": 1, "protocol": assay.PROTOCOL, "manifest_sha256": assay.MANIFEST_SHA,
             "seed": worker["config"]["seed"], "rule": worker["config"]["profile"].removeprefix("retained-"),
             "pass": reference["pass"], "physical_anchors": reference["pass"] * 80,
             "numerical_clock": reference["pass"] * 320, "complete": reference["complete"],
             "teacher_readouts": reference["teacher_readouts"], "generated_readouts": reference["generated_readouts"]}
    if (set(value) != set(fixed) | {"before_sha256", "after_sha256", "interrupted", "cases"}
            or any(type(value[key]) is not type(expected) or value[key] != expected for key, expected in fixed.items())
            or not sha(value["before_sha256"]) or value["before_sha256"] != value["after_sha256"]):
        raise ValueError("assay observation identity/coverage mismatch")
    if value["interrupted"] not in (None, "user_stop", "resource_stop", "budget_stop"):
        raise ValueError("invalid assay observation interruption")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) > 48:
        raise ValueError("assay case bound")
    expected_rows = [next(row for row in rows if row["id"] == f"{group}-{index:02}")
                     for group in ("foundation", "paraphrase", "counterfactual") for index in range(16)]
    measured = [case(item, row, value["pass"] in assay.GENERATIONS) for item, row in zip(cases, expected_rows)]
    if (sum(item[0] for item in measured) != value["teacher_readouts"]
            or sum(item[1] for item in measured) != value["generated_readouts"]
            or any(not item[2] for item in measured[:-1])
            or value["complete"] != (len(cases) == 48 and all(item[2] for item in measured) and value["interrupted"] is None)):
        raise ValueError("assay readout conservation mismatch")
    return value


def check(directory: Path, worker: dict, result: dict, checkpoint_state: dict) -> None:
    state(checkpoint_state, worker["identity"])
    if (type(worker["config"].get("seed")) is not int or checkpoint_state["seed"] != worker["config"]["seed"]
            or hashlib.sha256(encoded(worker["config"])).hexdigest() != worker["identity"]["config_sha256"]):
        raise ValueError("assay checkpoint and admitted worker config differ")
    progress = checkpoint_state["assay"]
    if result["evaluation"] != {"kind": "retained_assay", "result": progress}:
        raise ValueError("assay result and checkpoint ledger differ")
    if progress["rule"] != worker["config"]["profile"].removeprefix("retained-"):
        raise ValueError("assay rule changed")
    rows = assay.source(Path(worker["data_root"]))
    for ref in progress["observations"]:
        observation(directory, ref, worker, rows)
    if result["status"] == "completed" and (result["global_opportunity"] != 1600
            or len(progress["observations"]) != 8 or not all(ref["complete"] for ref in progress["observations"])):
        raise ValueError("incomplete assay reported Completed")
