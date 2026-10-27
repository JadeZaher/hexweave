"""Fresh attempts, immutable worker input and resume claims; see causal_OWNERSHIP.md."""

import math
import os
import time
import uuid
from pathlib import Path

from scripts import causal_checkpoint, causal_contract, causal_prompts
from scripts.causal_observer import Observer, contained, load, write
from scripts.pulse_io import digest


def open_request(root: Path, path: Path, expected: str) -> tuple[dict, dict]:
    source = causal_contract.pin(root, {"path": str(path), "sha256": expected}, 65_536)
    request = load(source, 65_536)
    paths = causal_contract.validate(root, request)
    paths["owner_request"] = source
    return request, paths


def owned_parent(root: Path) -> Path:
    for relative in ("work", "work/causal"):
        path = contained(root, relative, exists=False)
        path.mkdir(exist_ok=True)
    return (root / "work/causal").resolve(strict=True)


def create(root: Path, request: dict, started: float) -> dict:
    if request.get("protocol", causal_contract.PROTOCOL) != causal_contract.PROTOCOL:
        from scripts.causal_assay_study import owner_context
        owner_context(root, request)
    parent = owned_parent(root)
    inherited = None
    consumed = {"opportunities": 0, "wall_millis": 0, "attempts": 1}
    run_id, attempt_id = uuid.uuid4().hex, uuid.uuid4().hex
    initial, sequence = 0, 1
    prompt_only = causal_prompts.operation(request["operation"])
    if prompt_only and request["resume"] is None:
        raise ValueError("prompt evaluation requires completed parent")
    if request["resume"] is not None:
        inherited = causal_checkpoint.parent(root, request["resume"], request["identity"], request["budget"])
        previous, release, _, state = inherited
        if prompt_only and (release["status"] != "completed" or inherited[2]["owner"]["stop"] != "running"):
            raise ValueError("prompt evaluation requires clean Completed parent")
        consumed = release["consumed"] | {"attempts": release["consumed"]["attempts"] + 1}
        run_id, initial, sequence = release["run_id"], state["global_opportunity"], release["observer_sequence"] + 1
        if previous.parent.name != run_id:
            raise ValueError("parent is not in its owned run directory")
    bounds = request["budget"]
    if (consumed["attempts"] > bounds["maximum_attempts"] or consumed["opportunities"] >= bounds["maximum_opportunities"] or
            request["limits"]["opportunities"] > bounds["maximum_opportunities"] - consumed["opportunities"]):
        raise ValueError("no admitted cumulative opportunity/attempt budget remains")
    milliseconds = min(request["limits"]["wall_millis"], bounds["maximum_wall_millis"] - consumed["wall_millis"])
    deadline = started + milliseconds / 1000
    if deadline - time.monotonic() <= request["limits"]["stop_reserve_millis"] / 1000:
        raise TimeoutError("no full cooperative stop reserve remains")
    run_directory = contained(parent, run_id, exists=inherited is not None)
    if inherited is None:
        run_directory.mkdir(exist_ok=False)
    directory = run_directory / attempt_id
    directory.mkdir(exist_ok=False)
    envelope = {"schema": 1, "run_id": run_id, "attempt_id": attempt_id,
                "parent": None if inherited is None else {"attempt_id": inherited[1]["attempt_id"], "checkpoint_sha256": request["resume"]["checkpoint"]["manifest_sha256"]},
                "budget": bounds, "consumed": consumed, "attempt_start_opportunity": initial,
                "attempt_start_wall_millis": consumed["wall_millis"], "observer_sequence": sequence,
                "stop": "running", "physical_work_unknown": False}
    observer = Observer(directory, sequence)
    observer.emit("attempt_created", run_id=run_id, attempt_id=attempt_id, parent=envelope["parent"], consumed=consumed)
    if request.get("protocol", causal_contract.PROTOCOL) != causal_contract.PROTOCOL:
        from scripts.causal_assay_study import declaration
        study_directory, _ = declaration(root, request["operation"]["study"])
        write(study_directory / f"arm-{request['operation']['arm']}-attempt.json",
              {"schema": 1, "request_sha256": digest(study_directory / f"arm-{request['operation']['arm']}.json"),
               "owner_pid": os.getpid(), "attempt_directory": str(directory)}, immutable=True)
    stop_paths = [parent / "STOP", run_directory / "STOP", directory / "STOP"]
    if "HEXWEAVE_CAUSAL_DRIVER_STOP" in os.environ:
        stop_paths.append(contained(root / "work/causal-drivers", os.environ["HEXWEAVE_CAUSAL_DRIVER_STOP"], exists=False))
    return {"directory": directory, "parent": inherited, "envelope": envelope, "observer": observer,
            "deadline": deadline, "started": started, "stop_paths": stop_paths}


def worker(root: Path, request: dict, paths: dict, context: dict) -> tuple[dict, str]:
    directory, envelope, observer = context["directory"], context["envelope"], context["observer"]
    envelope["consumed"] = envelope["consumed"] | {"wall_millis": envelope["attempt_start_wall_millis"] + math.ceil((time.monotonic() - context["started"]) * 1000)}
    envelope["observer_sequence"] = observer.sequence + 1
    remaining = min(int((context["deadline"] - time.monotonic()) * 1000), envelope["budget"]["maximum_wall_millis"] - envelope["consumed"]["wall_millis"])
    limits = request["limits"] | {"wall_millis": remaining}
    causal_contract.limits(limits, causal_prompts.operation(request["operation"]), request.get("protocol", causal_contract.PROTOCOL) != causal_contract.PROTOCOL)
    operation = request["operation"]
    if operation["kind"] == "prompt_evaluation":
        operation = operation | {"prompts": operation["prompts"] | {"path": str(paths["prompts"])}}
    if operation["kind"] == "retained_assay":
        operation = operation | {"study": operation["study"] | {"path": str(contained(root, operation["study"]["path"]))}}
    resume = None
    if context["parent"] is not None:
        parent_directory = context["parent"][0]
        resume = request["resume"] | {"parent_directory": str(parent_directory / "persistence"), "release_path": str(parent_directory / "release.json")}
    value = {"schema": 1, "protocol": request.get("protocol", causal_contract.PROTOCOL), "workspace": str(root),
             "attempt_directory": str(directory), "owner": envelope, "identity": request["identity"],
             "config": request["config"], "data_root": str(paths["data_root"]),
             "data_manifest_sha256": request["identity"]["data"]["manifest_sha256"],
             "kernel": request["identity"]["kernel"], "limits": limits,
             "stop_path": str(directory / "worker-stop.json"), "launch_receipt": str(directory / "launch.json"),
             "source_manifest": {"path": str(paths["source_manifest"]), "sha256": request["source_manifest"]["sha256"]}, "resume": resume, "operation": operation}
    write(directory / "worker-request.json", value, immutable=True)
    observer.emit("worker_request_published", sha256=digest(directory / "worker-request.json"))
    return value, digest(directory / "worker-request.json")


def claim(root: Path, request: dict, context: dict) -> None:
    if context["parent"] is None:
        return
    directory, release, _, _ = causal_checkpoint.parent(root, request["resume"], request["identity"], request["budget"])
    if release != context["parent"][1]:
        raise ValueError("parent release changed before claim")
    registry = directory.parent / "claims"
    registry.mkdir(exist_ok=True)
    registry = contained(root / "work/causal", str(registry))
    write(registry / (request["resume"]["release_sha256"] + ".json"), {"schema": 1, "release_sha256": request["resume"]["release_sha256"],
          "new_attempt_id": context["envelope"]["attempt_id"]}, immutable=True)
