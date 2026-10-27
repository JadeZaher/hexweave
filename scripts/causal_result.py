"""Bind worker output to complete continuation evidence; see causal_OWNERSHIP.md."""

import math
import struct
from pathlib import Path

from scripts import causal_checkpoint, causal_prompts
from scripts.causal_contract import PROTOCOL, integer, sha
from scripts.causal_observer import load
from scripts.pulse_io import digest


def evaluation(value: dict | None, limits: dict, manifest_sha: str, completed: bool) -> None:
    if value is None:
        if completed:
            raise ValueError("completed worker omitted development evaluation")
        return
    if not isinstance(value, dict) or value.get("before_sha256") != value.get("after_sha256") or not sha(value.get("before_sha256")):
        raise ValueError("development evaluation mutated state or lacks identity")
    data = value.get("dataset", {})
    if data.get("split") != "development" or data.get("manifest_sha256") != manifest_sha:
        raise ValueError("development evaluation data pin mismatch")
    losses = value.get("teacher_forced", {})
    for name in ("total", "answer", "citation", "format"):
        row = losses.get(name, {})
        if (not integer(row.get("targets"), 0, limits["evaluation_targets"]) or
                type(row.get("nll_sum")) not in (float, int) or not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0):
            raise ValueError("invalid development loss accounting")
    if (losses["total"]["targets"] != sum(losses[key]["targets"] for key in ("answer", "citation", "format")) or
            not math.isclose(losses["total"]["nll_sum"], sum(losses[key]["nll_sum"] for key in ("answer", "citation", "format")), rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("development mask totals disagree")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) > limits["generation_cases"]:
        raise ValueError("generation case bound exceeded")
    for case in cases:
        raw = case.get("generation", {}).get("raw_bytes")
        if not isinstance(raw, list) or len(raw) > limits["generation_bytes"] or any(not integer(byte, 0, 255) for byte in raw):
            raise ValueError("generation bytes invalid or unbounded")
    for name in ("teacher_coverage", "generation_coverage"):
        coverage = value.get(name, {})
        if coverage.get("kind") not in ("complete", "limit", "interrupted") or (completed and coverage["kind"] == "interrupted"):
            raise ValueError("missing or incompatible evaluation coverage")


def counters(value: dict, opportunity: int, version: int, assay: bool = False) -> None:
    fields = ("anchors", "local_opportunities", "visits", "local_tile_updates", "anchor_tile_commits",
              "predictor_anchor_updates", "predictor_replay_updates", "predictor_anchor_pairs",
              "predictor_replay_pairs", "predictor_transitions")
    if (not isinstance(value, dict) or not integer(version, 0, 2**64 - 1)
            or any(not integer(value.get(key), 0, 2**64 - 1) for key in fields)):
        raise ValueError("joint batch/pair counter encoding invalid")
    a, local = value["anchors"], value["local_opportunities"]
    ab, rb = value["predictor_anchor_updates"], value["predictor_replay_updates"]
    ap, rp = value["predictor_anchor_pairs"], value["predictor_replay_pairs"]
    if (a != (opportunity if assay else (opportunity + 3) // 4) or a + local != opportunity or ab != a
            or not a <= ap <= 4 * a or not rb <= rp <= 4 * rb or rb > local
            or not local <= value["local_tile_updates"] <= 4 * local
            or ap + value["local_tile_updates"] != value["visits"]
            or not a <= value["anchor_tile_commits"] <= ap
            or version > ab + rb or value["predictor_transitions"] > 128 * (ab + rb)):
        raise ValueError("joint batch/pair/visit conservation mismatch")
    for key in ("predictor_mse_sum", "anchor_loss_sum"):
        if type(value.get(key)) not in (float, int) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError("joint aggregate metric invalid")


def check(directory, worker: dict, child: dict) -> dict:
    if (child.get("reaped") is not True or child.get("returncode") != 0 or child.get("job_closed") is not True
            or child.get("terminal_peak_memory_queried") is not True or child.get("cleanup_errors") or child.get("forced_termination")):
        raise ValueError("worker lacks clean exact-handle terminal evidence")
    result = load(directory / "result.json")
    stdout = load(directory / "stdout.log", 16_384)
    wanted = {"schema": 1, "protocol": worker.get("protocol", PROTOCOL), "attempt_id": worker["owner"]["attempt_id"],
              "result_sha256": digest(directory / "result.json"), "status": result.get("status"), "result_file": "result.json"}
    if stdout != wanted or result.get("schema") != 1 or result.get("protocol") != worker.get("protocol", PROTOCOL) or result.get("attempt_id") != wanted["attempt_id"]:
        raise ValueError("worker stdout/result identity mismatch")
    if result.get("identity") != worker["identity"] or result.get("status") not in ("completed", "user_stop", "resource_stop", "budget_stop"):
        raise ValueError("worker identity/terminal status mismatch")
    if result.get("training_admission_claim") is not False or result.get("useful_assistant_claim") is not False:
        raise ValueError("unsupported worker readiness claim")
    physical = result.get("physical_opportunities_completed")
    if not integer(physical, 0, worker["limits"]["opportunities"]):
        raise ValueError("worker physical opportunity accounting invalid")
    expected_global = worker["owner"]["attempt_start_opportunity"] + physical
    manifest, state = causal_checkpoint.checkpoint(directory / "persistence", result["checkpoint"], worker["identity"])
    if result.get("global_opportunity") != expected_global or state["global_opportunity"] != expected_global or result.get("cursor") != state.get("cursor"):
        raise ValueError("worker checkpoint/cursor exposure mismatch")
    if result["status"] == "completed" and physical != worker["limits"]["opportunities"]:
        raise ValueError("completed worker did not finish declared opportunity quota")
    check_owner(worker["owner"], manifest["owner"], result["owner"], physical, result["status"])
    counts = state["counters"].copy()
    for name, bits in (("predictor_mse_sum", "predictor_mse_bits"), ("anchor_loss_sum", "anchor_loss_bits")):
        counts[name] = struct.unpack("<d", struct.pack("<Q", state[bits]))[0]
    assay = worker["operation"]["kind"] == "retained_assay"
    counters(counts, expected_global, state.get("predictor_version"), assay)
    if result.get("counters") != counts or counts["anchors"] != (expected_global if assay else (expected_global + 3) // 4) or counts["local_opportunities"] + counts["anchors"] != expected_global:
        raise ValueError("worker canonical checkpoint counters mismatch")
    if not isinstance(result.get("timings"), dict) or any(not integer(value, 0, 86_400_000) for value in result["timings"].values()):
        raise ValueError("worker timing evidence invalid")
    if assay:
        from scripts import causal_assay_result
        if result.get("operation") != worker["operation"]:
            raise ValueError("assay operation changed")
        causal_assay_result.check(directory, worker, result, state)
    else:
        check_evaluations(directory, worker, result, manifest)
    return result


def check_owner(initial: dict, checkpoint: dict, final: dict, physical: int, status: str) -> None:
    for owner in (checkpoint, final):
        for key in ("schema", "run_id", "attempt_id", "parent", "budget", "attempt_start_opportunity", "attempt_start_wall_millis"):
            if owner.get(key) != initial[key]:
                raise ValueError("worker owner envelope changed")
        consumed = owner["consumed"]
        if (consumed["opportunities"] != initial["consumed"]["opportunities"] + physical or consumed["attempts"] != initial["consumed"]["attempts"]
                or consumed["wall_millis"] < initial["consumed"]["wall_millis"] or owner["observer_sequence"] < initial["observer_sequence"]
                or owner["physical_work_unknown"] is not False or owner["stop"] != ("running" if status == "completed" else status)):
            raise ValueError("worker cumulative owner accounting mismatch")
    if final["consumed"]["wall_millis"] < checkpoint["consumed"]["wall_millis"]:
        raise ValueError("worker final wall regressed")


def check_evaluations(directory, worker, result, manifest):
    if result.get("operation") != worker["operation"]:
        raise ValueError("worker operation changed")
    envelope = result.get("evaluation", {})
    if not isinstance(envelope, dict) or set(envelope) != {"kind", "result"} or envelope["kind"] != worker["operation"]["kind"]:
        raise ValueError("worker evaluation variant mismatch")
    values = envelope["result"]
    if causal_prompts.operation(worker["operation"]):
        parent, _ = causal_checkpoint.checkpoint(Path(worker["resume"]["parent_directory"]), worker["resume"]["checkpoint"], worker["identity"])
        evaluated, _ = causal_checkpoint.checkpoint(directory / "persistence", values["evaluation_checkpoint"], worker["identity"])
        causal_prompts.check(values, worker, result, parent, evaluated, manifest)
    else:
        for name in ("initial", "final"):
            evaluation(values.get(name), worker["limits"], worker["data_manifest_sha256"], result["status"] == "completed")
        causal_checkpoint.checkpoint(directory / "persistence", values["final_evaluation_checkpoint"], worker["identity"])
