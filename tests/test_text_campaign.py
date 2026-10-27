"""Fixed T2 campaign failure and measurement contracts; no training starts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import text_campaign


def measurement(directory, owner, seed, arm):
    return {"seed": seed, "arm": arm, "initial_nll": 5.5, "improvement": 0.05 if arm == "coupled" else 0.1,
            "child_wall_seconds": 1.0, "training_seconds": 0.5, "input_sha256": {}, "resolved_kernel": "avx2",
            "data": {}, "evaluation_examples": 64, "receipt": "mock", "credit_integrity_checked": True,
            "credit_diagnostics": {"anchors_with_different_results": 16,
                "total": {"raw": {"task_descent": 1.0}, "ratios": {"deficit_over_task_descent": 0.1}}}}


class TextCampaignTests(unittest.TestCase):
    def test_campaign_stops_on_failure_stop_and_shared_deadline(self):
        for status in ("failed", "stopped", "deadline", "source-budget", "pair-mismatch", "source-stop"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as name:
                now = [0.0]
                source_calls = [0]

                def source(_root):
                    source_calls[0] += 1
                    if status == "source-budget" and source_calls[0] == 3:
                        now[0] = 211.0
                    if status == "source-stop" and source_calls[0] == 8:
                        (_root / "work/text/STOP").touch()
                    return {}

                def child(root, **options):
                    if status == "deadline":
                        now[0] = 241.0
                    return root, {"status": status if status in ("failed", "stopped") else "completed"}

                def measured(directory, owner, seed, arm):
                    run = measurement(directory, owner, seed, arm)
                    if status == "pair-mismatch" and arm == "route-to-u-detached":
                        run["initial_nll"] = 5.6
                    return run

                with patch.object(text_campaign, "source_identity", side_effect=source), \
                        patch.object(text_campaign.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(text_campaign.text_runner, "execute", side_effect=child) as launch, \
                        patch.object(text_campaign, "measurement", side_effect=measured):
                    directory, report = text_campaign.execute(Path(name), profile="t2-credit")
                self.assertEqual(launch.call_count, {"pair-mismatch": 2, "source-stop": 6}.get(status, 1))
                self.assertEqual(launch.call_args.kwargs["campaign_deadline"], 240)
                self.assertEqual(launch.call_args.kwargs["profile"], "t2-credit")
                self.assertEqual(launch.call_args.kwargs["credit_diagnostics"], "on")
                self.assertEqual(report["status"], "stopped" if status in ("stopped", "source-stop") else "failed")
                self.assertNotIn("comparison", report)
                self.assertEqual(json.loads((directory / "report.json").read_text()), report)

    def test_campaign_alternates_and_decides_only_with_complete_pairs(self):
        with tempfile.TemporaryDirectory() as name, \
                patch.object(text_campaign, "source_identity", return_value={}), \
                patch.object(text_campaign.text_runner, "execute", return_value=(Path(name), {"status": "completed"})) as launch, \
                patch.object(text_campaign, "measurement", side_effect=measurement):
            _, report = text_campaign.execute(Path(name), profile="t2-credit")
        self.assertEqual([(call.kwargs["seed"], call.kwargs["route_credit"]) for call in launch.call_args_list], list(text_campaign.SCHEDULE))
        self.assertEqual(report["comparison"]["decision"], "GO")
        self.assertIn("provisional", report["comparison"]["candidate"])
        self.assertIn("do not estimate incremental", report["comparison"]["timing_scope"])
        runs = report["runs"]
        with self.assertRaises(ValueError):
            text_campaign.paired_decision(runs[:-1])
        runs[0]["credit_diagnostics"]["anchors_with_different_results"] = 15
        inconclusive = text_campaign.paired_decision(runs)
        self.assertEqual(inconclusive["decision"], "INCONCLUSIVE")
        self.assertIn("experimental", inconclusive["candidate"])
        self.assertIn("not validated", inconclusive["candidate"])
        for run in runs:
            if run["arm"] == "route-to-u-detached":
                run["improvement"] = -0.01
        revise = text_campaign.paired_decision(runs)
        self.assertEqual(revise["decision"], "REVISE")
        self.assertIsNone(revise["candidate"])

    def test_terminal_persistence_overrun_is_durably_failed(self):
        real_publish = text_campaign.publish
        for phase in ("event", "publish"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as name:
                now = [0.0]

                def event(directory, kind, **fields):
                    if kind == "campaign_finished" and phase == "event":
                        now[0] = 241.0

                def publish(directory, report):
                    real_publish(directory, report)
                    if report["status"] == "completed" and phase == "publish":
                        now[0] = 241.0

                with patch.object(text_campaign.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(text_campaign, "source_identity", return_value={}), \
                        patch.object(text_campaign.text_runner, "execute", return_value=(Path(name), {"status": "completed"})) as launch, \
                        patch.object(text_campaign, "measurement", side_effect=measurement), \
                        patch.object(text_campaign, "event", side_effect=event), patch.object(text_campaign, "publish", side_effect=publish):
                    directory, report = text_campaign.execute(Path(name), profile="t2-credit")
                self.assertEqual(launch.call_count, 6)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["deadline_exceeded"])
                self.assertNotIn("comparison", report)
                self.assertEqual(json.loads((directory / "report.json").read_text()), report)

    def test_consumed_t1_profile_and_unverified_credit_never_launch(self):
        with tempfile.TemporaryDirectory() as name, patch.object(text_campaign.text_runner, "execute") as launch:
            for profile in ("t1", "fixture", ""):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    text_campaign.execute(Path(name), profile=profile)
            launch.assert_not_called()
        runs = [measurement(None, None, seed, arm) for seed, arm in text_campaign.SCHEDULE]
        runs[0]["credit_integrity_checked"] = False
        with self.assertRaises(ValueError):
            text_campaign.paired_decision(runs)


if __name__ == "__main__":
    unittest.main()
