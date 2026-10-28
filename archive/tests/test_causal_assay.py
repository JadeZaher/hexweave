"""Assay owner boundaries and malformed observation records, without model work."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import causal_assay as assay, causal_assay_result as result, causal_assay_study as study
from scripts import causal_assay_campaign as campaign, causal_assay_guard as guard_module
from scripts import causal_contract, causal_launch, causal_resources
from scripts.causal_observer import encoded, load, write
from scripts.pulse_io import digest


def config(seed=7, rule="magnitude"):
    return {"schema": 1, "protocol": assay.PROTOCOL, "profile": "retained-" + rule,
            "seed": seed, "threads": 1, "anchor_interval": 1, "max_visits": 4,
            "history_capacity": 8, "context_bytes": 1024,
            "evidence_rule": "threshold-one-canonical-zero-v1", "route_credit": "detached-u-experimental",
            "predictor_rule": "joint-adjacent-raw-row-mean-v1", "anchor_batch": "current-anchor-forward-v1",
            "replay_selection": "none-anchor-only-v1"}


def guard_fixture(root):
    guard = guard_module.Guard.__new__(guard_module.Guard)
    guard.root, guard.directory = root, root / "study"
    guard.directory.mkdir()
    (root / "work/causal").mkdir(parents=True)
    guard.started, guard.deadline, guard.last_sample = 0.0, 7200.0, 0.0
    guard.next_arm, guard.stopped, guard.last_completed, guard.owner_free_sample = 0, None, None, None
    guard.drivers, guard.attempts, guard.records = [], [], []
    guard.driver, guard.identity = SimpleNamespace(pid=101, _handle=123), {"pid": 101, "creation_filetime": 1}
    guard.observer = Mock()
    return guard


def plan_fixture(guard):
    spec = {"path": str(guard.directory / "study.json"), "sha256": "b" * 64}
    requests = []
    for index in range(6):
        path = guard.directory / f"arm-{index}.json"
        write(path, {"operation": {"arm": index, "study": spec}})
        requests.append({"path": str(path), "sha256": digest(path)})
    guard.plan = {"schema": 1, "protocol": assay.PROTOCOL, "study": spec,
                  "requests": requests, "training_authorized": False}
    path = guard.directory / "plan.json"
    write(path, guard.plan)
    return path


class CausalAssayTests(unittest.TestCase):
    def test_closed_source_order_and_owner_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        rows = assay.source(root / "datasets/retained-text-assay")
        schedule = load(root / "datasets/retained-text-assay/schedule.json", 32_768)
        self.assertEqual(len(rows), 64)
        for passed in range(20):
            order = [assay.cursor(passed * 80 + index * 5)["example_id"] for index in range(16)]
            self.assertEqual(order, schedule["passes"][passed]["case_order"])
        self.assertEqual(assay.cursor(1600)["example_id"], "complete")
        with self.assertRaises(ValueError):
            causal_contract.limits(study.LIMITS)
        causal_contract.limits(study.LIMITS, assay=True)
        for field, wrong in (("generation_cases", 40), ("checkpoint_every", 65), ("generation_bytes", 192)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                causal_contract.limits(study.LIMITS | {field: wrong}, assay=True)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            with patch.object(causal_launch.causal_prepare, "open_request", return_value=({"protocol": assay.PROTOCOL}, {})):
                with self.assertRaisesRegex(ValueError, "conserved live study"):
                    causal_launch.execute(root, root / "request.json", "a" * 64)

    def test_observation_censoring_raw_match_and_ledger_reject_corruption(self):
        row = {"id": "foundation-00", "completion": "soil\n"}
        good = {"id": row["id"], "content_nll": [1.0] * 4, "lf_nll": 0.5, "first_byte_rank": 2,
                "generated": list(b"soil\n"), "generation_complete": True, "exact": True}
        self.assertEqual(result.case(good, row, True), (5, 5, True))
        partial = good | {"generated": list(b"so"), "generation_complete": False, "exact": None}
        self.assertEqual(result.case(partial, row, True), (5, 2, False))
        for field, wrong in (("exact", False), ("generated", list(b"soil\nx")), ("lf_nll", float("nan")),
                             ("first_byte_rank", 0), ("generation_complete", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                result.case(good | {field: wrong}, row, True)
        state = {"schema": 2, "seed": 7, "global_opportunity": 0, "cursor": assay.cursor(0),
                 "assay": {"schema": 1, "rule": "magnitude", "observations": [],
                           "embedding": {"attempts": 0, "clamped": 0, "transitions": 0, "probability_sum_bits": 0}},
                 "counters": {key: 0 for key in ("anchors", "local_opportunities", "local_prefix_bytes",
                    "local_tile_updates", "predictor_replay_updates", "predictor_replay_pairs", "shared_transitions")}}
        identity = {"data": assay.identity(), "config_sha256": hashlib.sha256(encoded(config())).hexdigest()}
        result.state(state, identity)
        for field, wrong in (("schema", 1), ("assay", None), ("global_opportunity", 4)):
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                result.state(state | {field: wrong}, identity)
        changed = copy.deepcopy(state)
        changed["assay"]["embedding"]["attempts"] = 1
        with self.assertRaises(ValueError):
            result.state(changed, identity)
        worker = {"config": config(), "identity": identity, "data_root": "unused"}
        terminal = {"evaluation": {"kind": "retained_assay", "result": state["assay"]}, "status": "resource_stop"}
        with patch.object(assay, "source", return_value=[]) as source:
            result.check(Path("unused"), worker, terminal, state)
            source.assert_called_once()
            source.reset_mock()
            for seed in (None, True, "7", 7.0, 17):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    result.check(Path("unused"), worker, terminal, state | {"seed": seed})
            with self.assertRaises(ValueError):
                result.check(Path("unused"), worker, terminal, {key: value for key, value in state.items() if key != "seed"})
            for changed_config in (config(17), config(rule="dense")):
                wrong_identity = identity | {"config_sha256": hashlib.sha256(encoded(changed_config)).hexdigest()}
                with self.assertRaises(ValueError):
                    result.check(Path("unused"), worker | {"identity": wrong_identity}, terminal, state)
                with self.assertRaises(ValueError):
                    result.check(Path("unused"), worker | {"config": changed_config}, terminal, state)
            source.assert_not_called()

    def test_late_study_stop_revokes_only_last_completed_release(self):
        for boundary, last_status in (("post-loop", "completed"), ("after-report", "completed"),
                                      ("post-loop", "failed"), ("post-loop", "resource_stop")):
            with self.subTest(boundary=boundary, last_status=last_status), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                guard = guard_fixture(root)
                plan = plan_fixture(guard)
                original_record = guard.record

                def launch(*_args, **_kwargs):
                    index = len(guard.records)
                    directory, attempt = root / f"driver-{index}", root / "work/causal" / f"a-{index}"
                    directory.mkdir()
                    attempt.mkdir()
                    value = {"status": last_status if index == 5 else "completed", "attempt_directory": str(attempt),
                             "driver_wall_millis_before_final_receipt": 1}
                    write(attempt / "release.json", {"status": value["status"]})
                    write(directory / "driver-report.json", value)
                    return directory, value

                def record(directory, value):
                    original_record(directory, value)
                    if boundary == "post-loop" and len(guard.records) == 6:
                        (root / "work/causal/STOP").touch()

                def publish(path, value, **kwargs):
                    write(path, value, **kwargs)
                    if boundary == "after-report" and path == guard.directory / "report.json":
                        (root / "work/causal/STOP").touch()

                with patch.object(campaign, "Guard", return_value=guard), \
                        patch.object(causal_contract, "pin", side_effect=lambda _root, spec, _cap: Path(spec["path"])), \
                        patch.object(causal_contract, "validate"), \
                        patch.object(study, "declaration", return_value=(guard.directory, {"run_id": "fixture"})), \
                        patch.object(guard, "inventory", return_value=0), patch.object(guard, "resources", return_value=None), \
                        patch.object(guard, "record", side_effect=record), patch.object(campaign, "write", side_effect=publish), \
                        patch.object(guard_module.time, "monotonic", return_value=1), \
                        patch("scripts.causal_assay_score.arm", side_effect=lambda path: {
                            "rule": "dense" if int(path.name[2:]) % 2 == 0 else "magnitude", "diagnostic_gate": True}), \
                        patch.object(causal_launch, "execute", side_effect=launch) as launched:
                    directory, report = campaign.execute(root, plan, digest(plan))
                expected = "user_stop" if last_status == "completed" else last_status
                self.assertEqual(launched.call_count, 6)
                self.assertEqual(report["status"], expected)
                self.assertEqual(load(directory / "report.json"), report)
                self.assertFalse(report["every_seed_both_arms_gate"])
                self.assertEqual(report["every_seed_gate_by_rule"], {"dense": False, "magnitude": False})
                self.assertEqual([item["status"] for item in report["arms"]], ["completed"] * 5 + [expected])
                for index in range(6):
                    invalid = root / "work/causal" / f"a-{index}" / "release-invalid.json"
                    self.assertEqual(invalid.exists(), index == 5 and last_status == "completed")
                last = report["arms"][-1]
                self.assertEqual(last["driver_report_sha256"], digest(Path(last["driver_report"])))
                self.assertEqual(load(Path(last["driver_report"]))["status"], expected)
                previous = copy.deepcopy(report)
                guard.revoke_last()
                self.assertEqual(report, previous)

    def test_owner_free_admission_conserves_memory_and_remaining_disk(self):
        mib, total = 1024**2, 16 * 1024**3
        for fault in ("rss", "host", "disk", "query", "valid"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as name:
                guard = guard_fixture(Path(name).resolve())
                plan_fixture(guard)
                guard.next_arm, guard.records = 1, [{}]
                request_path = Path(guard.plan["requests"][1]["path"])
                used = 128 * mib
                free = 321 * mib if fault == "disk" else study.DISK_CAP - used + 256 * mib
                rss = causal_resources.OWNER_LIMIT + 1 if fault == "rss" else 32 * mib
                floor = causal_resources.admission_floor(total, rss, "strict")
                host = {"total_physical_bytes": total, "available_physical_bytes": floor - 1 if fault == "host" else floor}
                with patch.object(guard, "inventory", return_value=used), \
                        patch.object(guard_module.shutil, "disk_usage", return_value=SimpleNamespace(free=free)), \
                        patch.object(causal_resources, "process_identity", return_value=guard.identity), \
                        patch.object(causal_resources, "alive", return_value=True), \
                        patch.object(causal_resources, "memory_status", return_value=host), \
                        patch.object(causal_resources, "process_memory", return_value={"working_set_bytes": rss},
                                     side_effect=OSError("sample unavailable") if fault == "query" else None) as sampled, \
                        patch.object(guard_module.time, "monotonic", return_value=1), patch.object(study, "request"):
                    if fault == "valid":
                        guard.admit(load(request_path), request_path, digest(request_path))
                    else:
                        with self.assertRaises((InterruptedError, OSError)):
                            guard.admit(load(request_path), request_path, digest(request_path))
                sampled.assert_called_once_with(guard.driver)
                self.assertEqual(guard.stopped, None if fault == "valid" else "failed" if fault == "query" else "resource_stop")
                if fault != "query":
                    self.assertEqual(guard.owner_free_sample["remaining_study_disk_bytes"], study.DISK_CAP - used)
                    self.assertEqual(guard.owner_free_sample["required_free_disk_bytes"], study.DISK_CAP - used + 256 * mib)
                    self.assertEqual(guard.owner_free_sample["sampled_elapsed_millis"], 1000)
                    self.assertEqual(guard.owner_free_sample["driver"], guard.identity)

    def test_elastic_admission_starts_on_fixed_floor_below_strict_cushion(self):
        mib, total = 1024**2, 16 * 1024**3
        floor = causal_resources.ELASTIC_RUNTIME_FLOOR + causal_resources.CHILD_LIMIT
        self.assertLess(floor, causal_resources.admission_floor(total, 32 * mib, "strict"))
        for available, expected_stop in ((floor, None), (floor - 1, "resource_stop")):
            with self.subTest(available=available), tempfile.TemporaryDirectory() as name, \
                    patch.dict("os.environ", {"HEXWEAVE_CAUSAL_ADMISSION": "elastic"}):
                guard = guard_fixture(Path(name).resolve())
                plan_fixture(guard)
                guard.next_arm, guard.records = 1, [{}]
                request_path = Path(guard.plan["requests"][1]["path"])
                used = 128 * mib
                host = {"total_physical_bytes": total, "available_physical_bytes": available}
                with patch.object(guard, "inventory", return_value=used), \
                        patch.object(guard_module.shutil, "disk_usage",
                                     return_value=SimpleNamespace(free=study.DISK_CAP - used + 256 * mib)), \
                        patch.object(causal_resources, "process_identity", return_value=guard.identity), \
                        patch.object(causal_resources, "alive", return_value=True), \
                        patch.object(causal_resources, "memory_status", return_value=host), \
                        patch.object(causal_resources, "process_memory", return_value={"working_set_bytes": 32 * mib}), \
                        patch.object(guard_module.time, "monotonic", return_value=1), patch.object(study, "request"):
                    if expected_stop is None:
                        guard.admit(load(request_path), request_path, digest(request_path))
                    else:
                        with self.assertRaises(InterruptedError):
                            guard.admit(load(request_path), request_path, digest(request_path))
                self.assertEqual(guard.stopped, expected_stop)
                self.assertEqual(guard.owner_free_sample["admission_mode"], "elastic")
                self.assertEqual(guard.owner_free_sample["owner_admission_floor_bytes"], floor)


if __name__ == "__main__":
    unittest.main()
