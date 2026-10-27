"""Fixed diagnostic gates; these do not replace science-assistant evaluation."""

from pathlib import Path

from scripts import causal_assay as assay, causal_assay_result
from scripts.causal_observer import load


def undesirable(raw: list[int]) -> bool:
    if not raw or raw[-1] != 10:
        return True
    body = bytes(raw[:-1]).strip()
    if not body:
        return True
    return any(body == word * (len(body) // len(word)) and len(body) >= 2 * len(word)
               for word in (b"soil", b"sand", b"clay", b"silt"))


def summarize(value: dict, rows: list[dict]) -> dict:
    result = {"pass": value["pass"], "complete": value["complete"], "groups": {}, "gate": None}
    for group in ("foundation", "paraphrase", "counterfactual"):
        cases = [case for case in value["cases"] if case["id"].startswith(group + "-")]
        content = [loss for case in cases for loss in case["content_nll"]]
        lf = [case["lf_nll"] for case in cases if case["lf_nll"] is not None]
        exact_by_word = {word: 0 for word in ("soil", "sand", "clay", "silt")}
        for case in cases:
            row = next(row for row in rows if row["id"] == case["id"])
            if case["exact"] is True:
                exact_by_word[row["answer"]] += 1
        result["groups"][group] = {"cases": len(cases), "content_targets": len(content),
            "content_nll_sum": sum(content), "lf_targets": len(lf), "lf_nll_sum": sum(lf),
            "exact": sum(case["exact"] is True for case in cases), "exact_by_word": exact_by_word,
            "undesirable_generations": sum(undesirable(case["generated"]) for case in cases if case["generated"] is not None)}
    if value["pass"] in (8, 16, 17, 20) and value["complete"]:
        groups = result["groups"]
        result["gate"] = (groups["foundation"]["exact"] == 16
            and all(groups[group]["exact"] >= 14 and min(groups[group]["exact_by_word"].values()) >= 3
                    for group in ("paraphrase", "counterfactual"))
            and all(group["undesirable_generations"] == 0 for group in groups.values()))
    return result


def arm(directory: Path) -> dict:
    worker = load(directory / "worker-request.json", 65_536)
    result = load(directory / "result.json")
    rows = assay.source(Path(worker["data_root"]))
    progress = result["evaluation"]["result"]
    summaries = [summarize(causal_assay_result.observation(directory, ref, worker, rows), rows)
                 for ref in progress["observations"]]
    gates = [item["gate"] for item in summaries if item["pass"] in (8, 16, 17, 20)]
    return {"seed": worker["config"]["seed"], "rule": progress["rule"], "observations": summaries,
            "diagnostic_gate": result["status"] == "completed" and len(gates) == 4 and all(gates),
            "useful_assistant_gate": False}
