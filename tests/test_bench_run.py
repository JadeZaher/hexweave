"""Exercise loopback protocol and durable failure recording without networking."""

import http.client
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench.client import complete
from bench.runner import run
from test_bench_manifest import fixture


class ClientTests(unittest.TestCase):
    def test_request_is_nonstreaming_and_measurement_is_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = fixture(temporary)
            response = MagicMock(status=200)
            response.read.return_value = json.dumps({"model": "synthetic-fixture",
                "choices": [{"text": "yes", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
            connection = MagicMock()
            connection.getresponse.return_value = response
            with patch("bench.client.http.client.HTTPConnection", return_value=connection) as factory:
                result = complete(manifest, "Reply yes", 2)
            factory.assert_called_once_with("127.0.0.1", 1234, timeout=2)
            args, kwargs = connection.request.call_args
            self.assertEqual(args, ("POST", "/v1/completions"))
            self.assertFalse(json.loads(kwargs["body"])["stream"])
            self.assertNotIn("Authorization", kwargs["headers"])
            self.assertTrue(result["truncated"])
            self.assertEqual(result["output_utf8_bytes"], 3)
            self.assertEqual(result["reported_native_tokens"]["completion_tokens"], 1)
            self.assertIsNone(result["ttft_seconds"])
            self.assertIsNone(result["peak_vram_bytes"])
            connection.close.assert_called_once()

    def test_redirects_oversized_bodies_wrong_model_and_missing_terminal_fail(self):
        variants = [(302, b"{}"), (200, b"x" * 1025),
                    (200, b'{"model":"different","choices":[{"text":"x","finish_reason":"stop"}]}'),
                    (200, b'{"choices":[{"text":"x","finish_reason":null}]}')]
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = fixture(temporary)
            for status, payload in variants:
                connection, response = MagicMock(), MagicMock(status=status)
                response.read.return_value = payload
                connection.getresponse.return_value = response
                with patch("bench.client.http.client.HTTPConnection", return_value=connection):
                    with self.subTest(status=status, payload=payload[:32]), self.assertRaises(ValueError):
                        complete(manifest, "Reply yes", 2)
                connection.close.assert_called_once()


class RunTests(unittest.TestCase):
    def test_failure_is_persisted_and_no_retry_occurs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = fixture(temporary)
            requester = MagicMock(side_effect=http.client.RemoteDisconnected("fixture disconnect"))
            result = run(path, "work/failure.jsonl", root=temporary, requester=requester)
            records = [json.loads(line) for line in (Path(temporary) / "work/failure.jsonl").read_text().splitlines()]
            self.assertEqual([item["record"] for item in records], ["manifest", "example", "summary"])
            self.assertEqual(records[1]["failure"]["type"], "RemoteDisconnected")
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed_measured_requests"], 0)
            requester.assert_called_once()

    def test_success_records_exact_text_and_warmups_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            manifest["budget"]["warmups"] = 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            requester = MagicMock(return_value={"output": "yes", "finish_reason": "stop", "truncated": False,
                "end_to_end_seconds": 0.5, "output_utf8_bytes": 3, "reported_native_tokens": {
                    "prompt_tokens": None, "completion_tokens": None}, "ttft_seconds": None})
            result = run(path, "work/success.jsonl", root=temporary, requester=requester)
            self.assertEqual(requester.call_count, 2)
            self.assertEqual(result["completed_measured_requests"], 1)
            self.assertEqual(result["exact_match"], 1)
            self.assertEqual(result["output_bytes_per_end_to_end_second"], 6)
            self.assertFalse(result["artifact_endpoint_binding_verified"])
            self.assertFalse(result["leaderboard_eligible"])

    def test_unpinned_manifest_creates_no_output_or_network_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = fixture(temporary)
            manifest["model"]["revision"] = "main"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            requester = MagicMock()
            with self.assertRaises(ValueError):
                run(path, "work/refused.jsonl", root=temporary, requester=requester)
            requester.assert_not_called()
            self.assertFalse((Path(temporary) / "work/refused.jsonl").exists())
