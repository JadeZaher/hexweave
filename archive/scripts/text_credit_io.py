"""Validate observational U-credit denominators and sums; see text/AGENTS.md."""

import math


COUNTS = ("frames", "unique_tile_commits", "repeated_visit_commits", "coordinate_opportunities",
          "task_nonzero", "route_nonzero", "both_nonzero", "opposite_sign", "task_transitions",
          "coupled_transitions", "actual_transitions", "applied_transitions", "different_results",
          "different_commits", "task_opposing", "suppressed_task", "new_where_task_zero")
SUMS = ("task_route_dot", "task_norm_squared", "route_norm_squared", "task_descent",
        "coupled_projection", "deficit", "residual_max")


def close(left, right) -> bool:
    return type(left) in (int, float) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def check_raw(raw: dict, arm: str) -> None:
    if (not isinstance(raw, dict) or set(raw) != set(COUNTS + SUMS) or
            any(type(raw.get(key)) is not int or raw[key] < 0 for key in COUNTS) or
            any(type(raw.get(key)) not in (int, float) or not math.isfinite(raw[key]) for key in SUMS)):
        raise ValueError("missing or nonfinite U-credit raw metrics")
    commits, repeated, coords = raw["unique_tile_commits"], raw["repeated_visit_commits"], raw["coordinate_opportunities"]
    if (coords != 512 * commits or repeated > commits or
            not commits + repeated <= raw["frames"] <= commits + 2 * repeated or
            any(raw[key] > coords for key in COUNTS if key not in ("frames",)) or
            raw["opposite_sign"] > raw["both_nonzero"] or
            raw["both_nonzero"] > min(raw["task_nonzero"], raw["route_nonzero"])):
        raise ValueError("invalid U-credit coordinate/frame denominators")
    selected = "coupled_transitions" if arm == "coupled" else "task_transitions"
    if (raw["actual_transitions"] != raw[selected] or raw["applied_transitions"] != raw["actual_transitions"] or
            raw["task_transitions"] > raw["task_nonzero"] or raw["different_commits"] > commits or
            not raw["different_commits"] <= raw["different_results"] <= 512 * raw["different_commits"] or
            raw["different_results"] != sum(raw[key] for key in ("task_opposing", "suppressed_task", "new_where_task_zero")) or
            not abs(raw["task_transitions"] - raw["coupled_transitions"]) <= raw["different_results"] <= raw["task_transitions"] + raw["coupled_transitions"] or
            raw["task_opposing"] + raw["new_where_task_zero"] > raw["coupled_transitions"] or
            raw["task_opposing"] + raw["suppressed_task"] > raw["task_nonzero"] or
            raw["suppressed_task"] > raw["task_transitions"] or raw["new_where_task_zero"] > coords - raw["task_nonzero"]):
        raise ValueError("invalid U-credit transition reconciliation")
    if (any(raw[key] < 0 for key in ("task_norm_squared", "route_norm_squared", "task_descent", "deficit", "residual_max")) or
            not close(raw["deficit"], raw["coupled_projection"] + raw["task_descent"])):
        raise ValueError("invalid U-credit projection sums")
    if (any((raw[f"{kind}_nonzero"] == 0) != (raw[f"{kind}_norm_squared"] == 0) for kind in ("task", "route")) or
            ((raw["task_transitions"] == 0) != (raw["task_descent"] == 0)) or
            ((raw["task_norm_squared"] == 0 or raw["route_norm_squared"] == 0) and raw["task_route_dot"] != 0) or
            (raw["task_norm_squared"] == 0 and any(raw[key] != 0 for key in ("task_descent", "coupled_projection", "deficit")))):
        raise ValueError("invalid zero-gradient U-credit implications")


def check_metrics(metrics: dict, arm: str) -> dict:
    if not isinstance(metrics, dict):
        raise ValueError("missing U-credit metrics")
    raw = metrics.get("raw")
    check_raw(raw, arm)
    expected = {
        "opposite_over_both_nonzero": (raw["opposite_sign"], raw["both_nonzero"]),
        "different_over_coordinates": (raw["different_results"], raw["coordinate_opportunities"]),
        "deficit_over_task_descent": (raw["deficit"], raw["task_descent"]),
        "task_route_cosine": (raw["task_route_dot"], math.sqrt(raw["task_norm_squared"] * raw["route_norm_squared"]))}
    ratios = metrics.get("ratios", {})
    if set(ratios) != set(expected):
        raise ValueError("missing U-credit ratios")
    for key, (numerator, denominator) in expected.items():
        if (denominator == 0 and ratios[key] is not None) or (denominator > 0 and not close(ratios[key], numerator / denominator)):
            raise ValueError("U-credit ratio denominator mismatch")
    return raw


def check_credit(receipt: dict, options: dict) -> None:
    if "credit_diagnostics" not in receipt:
        raise ValueError("missing explicit credit diagnostic state")
    diagnostic = receipt.get("credit_diagnostics")
    if options["credit_diagnostics"] == "off":
        if diagnostic is not None:
            raise ValueError("unexpected enabled credit diagnostic")
        return
    if (not isinstance(diagnostic, dict) or type(diagnostic.get("schema")) is not int or diagnostic["schema"] != 1 or
            diagnostic.get("scope") != "frozen anchor unique-tile commits only" or
            type(diagnostic.get("coordinates_per_commit")) is not int or diagnostic["coordinates_per_commit"] != 512):
        raise ValueError("missing credit diagnostic contract")
    anchors, different = diagnostic.get("anchors"), diagnostic.get("anchors_with_different_results")
    if type(anchors) is not int or anchors != receipt["counts"]["anchor_targets"] or type(different) is not int or not 0 <= different <= anchors:
        raise ValueError("invalid credit anchor accounting")
    if not close(diagnostic.get("anchor_difference_fraction"), different / anchors):
        raise ValueError("invalid credit anchor difference denominator")
    total = check_metrics(diagnostic.get("total"), options["route_credit"])
    if (total["frames"] != receipt["counts"]["returned_frames"] or
            not anchors <= total["unique_tile_commits"] <= total["frames"] or
            not different <= total["different_commits"] <= 3 * different or
            total["applied_transitions"] > receipt["tile_transitions"]["pulse_context"]):
        raise ValueError("credit anchor/transition totals mismatch")
    tiles = diagnostic.get("per_tile")
    if not isinstance(tiles, list) or len(tiles) != 7:
        raise ValueError("credit diagnostic requires seven tile records")
    rows = []
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict) or type(tile.get("tile")) is not int or tile["tile"] != index:
            raise ValueError("credit tile identity mismatch")
        rows.append(check_metrics(tile.get("metrics"), options["route_credit"]))
    for key in COUNTS + SUMS:
        expected = max(row[key] for row in rows) if key == "residual_max" else sum(row[key] for row in rows)
        if (key in COUNTS and total[key] != expected) or (key in SUMS and not close(total[key], expected)):
            raise ValueError("credit per-tile aggregate mismatch")
