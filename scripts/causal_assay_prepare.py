"""Prepare six pinned requests only; launching requires an explicit separate command."""

import argparse
import hashlib
import uuid
from pathlib import Path

from scripts import causal_assay as assay, causal_assay_study as study, causal_contract
from scripts.causal_observer import contained, encoded, load, write
from scripts.pulse_io import digest


def prepare(root: Path, template: dict) -> Path:
    root = root.resolve(strict=True)
    required = {"binary", "source_manifest", "build_receipt", "identity", "config"}
    if set(template) != required:
        raise ValueError("template must contain only binary/source/build/identity/config pins")
    common = template["identity"] | {"protocol": assay.PROTOCOL, "protocol_sha256": assay.CONTRACT_SHA,
                                     "data": assay.identity()}
    common.pop("config_sha256", None)
    run_id = uuid.uuid4().hex
    parent = contained(root, "work/causal-studies", exists=False)
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / run_id
    directory.mkdir(exist_ok=False)
    declaration = {"schema": 1, "protocol": assay.PROTOCOL, "run_id": run_id,
                   "common_identity": common, "arms": study.ARMS, "budget": study.BUDGET,
                   "limits": study.LIMITS, "whole_wall_millis": study.WHOLE_MILLIS, "disk_cap_bytes": study.DISK_CAP}
    write(directory / "study.json", declaration, immutable=True)
    spec = {"path": str(directory / "study.json"), "sha256": digest(directory / "study.json")}
    requests = []
    for index, arm in enumerate(study.ARMS):
        config = template["config"] | {"protocol": assay.PROTOCOL, "seed": arm["seed"],
            "profile": "retained-" + arm["rule"], "anchor_interval": 1, "replay_selection": "none-anchor-only-v1"}
        identity = common | {"config_sha256": hashlib.sha256(encoded(config)).hexdigest()}
        request = {"schema": 1, "protocol": assay.PROTOCOL, "binary": template["binary"],
            "source_manifest": template["source_manifest"], "build_receipt": template["build_receipt"],
            "identity": identity, "config": config, "data_root": "datasets/retained-text-assay",
            "budget": study.BUDGET, "limits": study.LIMITS, "resume": None,
            "operation": {"kind": "retained_assay", "study": spec, "arm": index}}
        causal_contract.validate(root, request)
        path = directory / f"arm-{index}.json"
        write(path, request, immutable=True)
        requests.append({"path": str(path), "sha256": digest(path)})
    write(directory / "plan.json", {"schema": 1, "protocol": assay.PROTOCOL, "study": spec,
          "requests": requests, "training_authorized": False}, immutable=True)
    return directory / "plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = causal_contract.pin(root, {"path": str(args.template), "sha256": args.template_sha256}, 16_384)
    result = prepare(root, load(path, 16_384))
    print(f"Prepared only: {result} SHA256 {digest(result)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
