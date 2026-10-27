"""The study coordinator is the existing sampled external driver."""

import shutil
import time
from pathlib import Path

from scripts import causal_assay_study as study, causal_disk, causal_resources
from scripts.causal_observer import Observer, contained, load, write
from scripts.pulse_io import digest


class Guard:
    def __init__(self, root: Path, directory: Path, plan: dict, started: float):
        self.root, self.directory, self.plan = root, directory, plan
        self.started = started
        self.deadline = self.started + study.WHOLE_MILLIS / 1000
        self.next_arm, self.stopped, self.last_sample = 0, None, 0.0
        self.drivers, self.attempts, self.records = [], [], []
        self.last_completed, self.owner_free_sample = None, None
        self.observer = Observer(directory, 1)
        self.driver = causal_resources.owner_process()
        self.identity = causal_resources.process_identity(self.driver)
        write(directory / "STARTED.json", {"schema": 1, "driver": self.identity}, immutable=True)
        if (root / "work/causal/STOP").exists():
            raise InterruptedError("program STOP remains terminal; no automatic archival")
        if shutil.disk_usage(root).free < study.DISK_CAP + 256 * 1024**2:
            raise OSError("study disk allowance plus cushion unavailable")
        self.observer.emit("study_started", driver=self.identity, whole_wall_millis=study.WHOLE_MILLIS,
                           fixed_owner_wall_millis_sum=6 * study.LIMITS["wall_millis"])

    def admit(self, request: dict, path: Path, sha: str) -> None:
        self.poll(None, None, {}, force=True, reserve_disk=True)
        if self.stopped or self.next_arm >= 6 or len(self.records) != self.next_arm:
            raise InterruptedError("study is stopped, spent or has an active arm")
        if time.monotonic() + study.LIMITS["wall_millis"] / 1000 + 15 >= self.deadline:
            self.stopped = "budget_stop"
            raise TimeoutError("whole study lacks the next complete fixed arm allowance")
        expected = self.plan["requests"][self.next_arm]
        if (path.resolve(strict=True) != Path(expected["path"]).resolve(strict=True) or sha != expected["sha256"]
                or digest(path) != sha or request["operation"]["arm"] != self.next_arm
                or request["operation"]["study"] != self.plan["study"]):
            raise ValueError("standalone/reordered assay request rejected")
        study.request(self.root, request)
        self.request_sha = sha

    def environment(self, driver: dict, directory: Path) -> dict:
        if driver != self.identity:
            raise ValueError("study driver process identity changed")
        self.drivers.append(directory)
        write(self.directory / f"arm-{self.next_arm}-entered.json",
              {"schema": 1, "study_sha256": self.plan["study"]["sha256"], "arm": self.next_arm,
               "driver": driver, "request_sha256": self.request_sha}, immutable=True)
        self.observer.emit("arm_entered", arm=self.next_arm, request_sha256=self.request_sha, driver_directory=str(directory))
        return {"HEXWEAVE_CAUSAL_STUDY_SHA256": self.plan["study"]["sha256"],
                "HEXWEAVE_CAUSAL_STUDY_ARM": str(self.next_arm)}

    def discover_attempt(self, process) -> None:
        marker = self.directory / f"arm-{self.next_arm}-attempt.json"
        if len(self.attempts) > self.next_arm or not marker.exists():
            return
        value = load(marker, 4096)
        if value.get("request_sha256") != self.request_sha or value.get("owner_pid") != process.pid:
            raise ValueError("study attempt handoff changed")
        path = contained(self.root / "work/causal", value["attempt_directory"])
        self.attempts.append(path)

    def study_bytes(self, process) -> int:
        total = 0
        entries = list(self.directory.iterdir())
        if len(entries) > 128:
            raise ValueError("study control directory entry bound")
        for path in entries:
            try:
                info = path.lstat()
            except FileNotFoundError:
                if (process is None or process.poll() is not None
                        or path.name != f"arm-{self.next_arm}-attempt.json.pending"):
                    raise
                total += causal_disk.PUBLICATION_CAP
                continue
            causal_disk.regular(info, False)
            total += info.st_size
        return total

    def inventory(self, process) -> int:
        total = self.study_bytes(process)
        for directory in self.drivers:
            total += causal_disk.disk_bytes(directory)
        for index, directory in enumerate(self.attempts):
            writer = None
            launch = directory / "launch.json"
            if index == self.next_arm and process is not None and process.poll() is None and launch.exists():
                value = load(launch, 4096)
                if value["owner"]["pid"] != process.pid:
                    raise ValueError("study numerical writer owner mismatch")
                writer = value["child"]["pid"]
            live_owner = process.pid if index == self.next_arm and process is not None and process.poll() is None else None
            total += causal_disk.disk_bytes(directory, writer, live_owner)
        return total

    def resources(self, process, used: int, reserve_disk: bool) -> dict | None:
        remaining = max(0, study.DISK_CAP - used)
        required = 256 * 1024**2 + (remaining if reserve_disk else 0)
        free = shutil.disk_usage(self.root).free
        if used > study.DISK_CAP or free < required:
            self.stopped = self.stopped or "resource_stop"
        if process is not None and process.poll() is None:
            return None
        if causal_resources.process_identity(self.driver) != self.identity or not causal_resources.alive(self.driver):
            raise RuntimeError("study driver identity is not live")
        memory = causal_resources.process_memory(self.driver)
        host = causal_resources.memory_status()
        mode = causal_resources.admission_mode()
        floor = causal_resources.admission_floor(host["total_physical_bytes"], memory["working_set_bytes"], mode)
        self.owner_free_sample = {"scope": "exact live study driver only; adjacent samples with no active owner",
            "driver": self.identity, "driver_memory": memory, "host": host,
            "admission_mode": mode, "owner_admission_floor_bytes": floor,
            "host_cushion_bytes": causal_resources.required_cushion(host["total_physical_bytes"]),
            "sampled_elapsed_millis": int((time.monotonic() - self.started) * 1000),
            "driver_owner_rss_limit_bytes": causal_resources.OWNER_LIMIT, "disk_free_bytes": free,
            "remaining_study_disk_bytes": remaining, "required_free_disk_bytes": required}
        if memory["working_set_bytes"] > causal_resources.OWNER_LIMIT or host["available_physical_bytes"] < floor:
            self.stopped = self.stopped or "resource_stop"
        return self.owner_free_sample

    def poll(self, driver_directory, process, report: dict, force: bool = False, reserve_disk: bool = False) -> None:
        now = time.monotonic()
        if (self.root / "work/causal/STOP").exists() or (self.directory / "STOP").exists():
            self.stopped = self.stopped or "user_stop"
        if now >= self.deadline - 15:
            self.stopped = self.stopped or "budget_stop"
        if force or now - self.last_sample >= 1:
            try:
                if process is not None:
                    self.discover_attempt(process)
                used = self.inventory(process)
                resources = self.resources(process, used, reserve_disk)
            except Exception:
                self.stopped = self.stopped or "failed"
                raise
            self.observer.emit("study_heartbeat", arm=self.next_arm, disk_bytes=used,
                               elapsed_millis=int((now - self.started) * 1000), stopped=self.stopped,
                               owner_free_resources=resources)
            self.last_sample = now
        if self.stopped:
            report["study_stop_reason"] = self.stopped
            if driver_directory is not None:
                from scripts.causal_launch import stop
                stop(driver_directory, self.stopped)
        if now >= self.deadline and process is not None and process.poll() is None:
            raise TimeoutError("whole study hard wall exceeded")

    def record(self, directory: Path, report: dict) -> None:
        try:
            self.poll(None, None, {}, force=True)
        except Exception as error:
            self.stopped = self.stopped or "failed"
            report["study_record_error"] = str(error)
        if self.stopped:
            self.revoke(report)
            write(directory / "driver-report.json", report)
        status = report["status"]
        if status != "completed":
            self.stopped = self.stopped or status
        record = {"arm": self.next_arm, "status": status, "driver_report": str(directory / "driver-report.json"),
                  "driver_report_sha256": digest(directory / "driver-report.json"),
                  "attempt_directory": report.get("attempt_directory"),
                  "execute_interval_millis": report["driver_wall_millis_before_final_receipt"]}
        self.records.append(record)
        self.last_completed = record if status == "completed" else None
        self.observer.emit("arm_terminal", **record)
        self.next_arm += 1

    def revoke_last(self) -> None:
        if self.stopped and self.last_completed is not None:
            record = self.last_completed
            path = Path(record["driver_report"])
            report = load(path)
            self.revoke(report)
            write(path, report)
            record.update(status=self.stopped, driver_report_sha256=digest(path))
            self.last_completed = None
            self.observer.emit("last_arm_revoked", arm=record["arm"], status=self.stopped)

    def revoke(self, report: dict) -> None:
        if not self.stopped:
            return
        if report["status"] == "completed":
            report["status"] = self.stopped
        if report.get("attempt_directory"):
            directory = contained(self.root / "work/causal", report["attempt_directory"])
            invalid = directory / "release-invalid.json"
            if not invalid.exists():
                write(invalid, {"schema": 1, "reason": "late conserved study STOP"}, immutable=True)
