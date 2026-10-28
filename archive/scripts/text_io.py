"""Validate byte-experiment receipts and owned paths; see scripts/AGENTS.md."""

import json
import math
from pathlib import Path

from scripts.pulse_io import LOG_LIMIT, finite_tree, unique_object
from scripts.text_credit_io import check_credit
from scripts.text_profile import check_profile_receipt


def owned_parent(root: Path) -> Path:
    for relative in ("work", "work/text"):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("text artifacts must remain in the repository")
        path.mkdir(exist_ok=True)
    return (root / "work/text").resolve(strict=True)


def pinned_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("text executable and data must remain in the repository")
    return path


def read_receipt(path: Path, options: dict) -> dict:
    if path.stat().st_size > LOG_LIMIT:
        raise ValueError("text receipt exceeds 1 MiB")
    receipt = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(receipt, dict) or not finite_tree(receipt):
        raise ValueError("text receipt must be finite JSON")
    if receipt.get("protocol") != "text-pulse-v4" or receipt.get("status") != "complete":
        raise ValueError("text receipt is not a completed protocol result")
    if receipt.get("options") != options:
        raise ValueError("text receipt options mismatch")
    if any(type(receipt["options"].get(key)) is not type(value) for key, value in options.items()):
        raise ValueError("text receipt option types mismatch")
    if type(receipt.get("completed_step_opportunities")) is not int or receipt["completed_step_opportunities"] != options["steps"]:
        raise ValueError("text step accounting mismatch")
    expected = (options["steps"] - 1) // options["anchor_interval"] + 1
    count = receipt.get("counts", {}).get("anchor_targets")
    if type(count) is not int or count != expected:
        raise ValueError("text target accounting mismatch")
    peak = receipt.get("residency", {}).get("peak_live_tiles")
    if type(peak) is not int or peak != 1:
        raise ValueError("text tile residency contract mismatch")
    if receipt.get("architecture", {}).get("floating_weight_masters") is not False:
        raise ValueError("text ternary parameter contract mismatch")
    pulse = receipt.get("architecture", {}).get("training_pulse", {})
    if pulse.get("kind") != "joint-tile-pulse-v1" or pulse.get("all_coefficients_ternary") is not True:
        raise ValueError("missing joint training-pulse contract")
    if options.get("route_credit") not in ("coupled", "route-to-u-detached") or pulse.get("route_credit") != options["route_credit"]:
        raise ValueError("text route-credit identity mismatch")
    if receipt.get("resolved_kernel") not in ("scalar", "avx2"):
        raise ValueError("missing resolved kernel")
    evaluation = receipt.get("evaluation", {})
    for key in ("initial_mean_nll_nats_per_byte", "final_mean_nll_nats_per_byte"):
        if type(evaluation.get(key)) not in (int, float) or evaluation[key] < 0:
            raise ValueError("missing or invalid text evaluation loss")
    check_diagnostics(receipt, options)
    check_profile_receipt(receipt, options)
    check_credit(receipt, options)
    check_snapshot(path.with_name("snapshot.json"), receipt)
    return receipt


def nonnegative_int(value) -> bool:
    return type(value) is int and value >= 0


def check_diagnostics(receipt: dict, options: dict) -> None:
    counts = receipt.get("counts", {})
    names = ("tile_visits", "local_updates", "anchor_targets", "returned_frames",
             "learned_halts", "forced_exits", "revisits", "route_calls",
             "predictor_predictions", "predictor_updates", "predictor_replays")
    if any(not nonnegative_int(counts.get(name)) for name in names):
        raise ValueError("missing text work accounting")
    paths = options["steps"] if options["mode"] == "pulse" else counts["anchor_targets"]
    anchors = counts["anchor_targets"]
    local_opportunities = options["steps"] - anchors if options["mode"] == "pulse" else 0
    if (not anchors <= counts["returned_frames"] <= 3 * anchors or
            not local_opportunities <= counts["local_updates"] <= 3 * local_opportunities or
            counts["predictor_replays"] > local_opportunities):
        raise ValueError("impossible anchor/local work allocation")
    for key, size in (("visits_by_tile", 7), ("path_lengths", 3), ("transitions", 5)):
        values = counts.get(key)
        if not isinstance(values, list) or len(values) != size or not all(map(nonnegative_int, values)):
            raise ValueError("invalid text diagnostic array")
    if (sum(counts["visits_by_tile"]) != counts["tile_visits"] or
            sum(counts["path_lengths"]) != paths or
            sum((index + 1) * count for index, count in enumerate(counts["path_lengths"])) != counts["tile_visits"] or
            counts["learned_halts"] + counts["forced_exits"] != paths or
            counts["route_calls"] + counts["forced_exits"] != counts["tile_visits"] or
            counts["local_updates"] + counts["returned_frames"] != counts["tile_visits"] or
            counts["predictor_predictions"] != counts["local_updates"]):
        raise ValueError("inconsistent text path accounting")
    expected_predictor = counts["returned_frames"] + counts["predictor_replays"] if options["mode"] == "pulse" else 0
    error = receipt.get("predictor_error", {})
    denominator = error.get("update_denominator")
    error_sum = counts.get("predictor_error_sum")
    if (not nonnegative_int(denominator) or denominator != counts["predictor_updates"] or
            denominator != expected_predictor or type(error_sum) not in (int, float) or error_sum < 0 or
            error.get("sum_per_update_mean_squared_error") != error_sum or
            error.get("includes_replay") is not True or error.get("components_per_update") != 16):
        raise ValueError("invalid predictor error denominator")
    mean = error.get("mean_squared_error")
    if (denominator == 0 and mean is not None) or (denominator > 0 and
            (type(mean) not in (int, float) or not math.isclose(mean, error_sum / denominator, rel_tol=1e-6, abs_tol=1e-9))):
        raise ValueError("invalid predictor error mean")
    for key, names in (("tile_transitions", ("transform", "curve", "router", "pulse_context", "pulse_heads")),
                       ("shared_transitions", ("embedding", "readout", "predictor")),
                       ("disk_io", ("loads", "commits", "bytes_read", "bytes_written", "initialization_bytes_written"))):
        values = receipt.get(key, {})
        if any(not nonnegative_int(values.get(name)) for name in names):
            raise ValueError("missing text transition or I/O accounting")
        if key == "tile_transitions" and [values[name] for name in names] != counts["transitions"]:
            raise ValueError("inconsistent tile transition counts")
    for name in ("training_seconds", "total_seconds"):
        value = receipt.get("timings", {}).get(name)
        if type(value) not in (int, float) or value <= 0:
            raise ValueError("invalid text timing")
    evaluation = receipt["evaluation"]
    if (type(evaluation.get("examples_per_pass")) is not int or not 1 <= evaluation["examples_per_pass"] <= 64 or
            evaluation.get("history_reset_each_example") is not True or evaluation.get("training_mutation_checked") is not True or
            receipt.get("artifact_bounds", {}).get("checkpoint_resumable") is not False):
        raise ValueError("missing evaluation or snapshot contract")


def check_snapshot(path: Path, receipt: dict) -> None:
    if path.stat().st_size > LOG_LIMIT:
        raise ValueError("text snapshot exceeds 1 MiB")
    snapshot = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(snapshot, dict) or not finite_tree(snapshot):
        raise ValueError("text snapshot must be finite JSON")
    if type(snapshot.get("schema")) is not int or snapshot["schema"] != 4 or snapshot.get("resumable") is not False:
        raise ValueError("text snapshot schema or resume mismatch")
    for key in ("protocol", "status", "options", "data", "completed_step_opportunities"):
        if snapshot.get(key) != receipt.get(key):
            raise ValueError("text snapshot identity mismatch")
    if any(type(snapshot["options"].get(key)) is not type(value) for key, value in receipt["options"].items()):
        raise ValueError("text snapshot option types mismatch")
    if snapshot.get("anchor_targets") != receipt["counts"]["anchor_targets"]:
        raise ValueError("text snapshot exposure mismatch")
    versions = snapshot.get("tile_versions")
    if not isinstance(versions, list) or len(versions) != 7 or not all(map(nonnegative_int, versions)):
        raise ValueError("text snapshot versions invalid")
