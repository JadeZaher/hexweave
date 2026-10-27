"""Validate benchmark boundaries with synthetic local artifacts only."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bench.files import decode_json, exclusive_output
from bench.manifest import endpoint, load_manifest


def fixture(directory):
    root = Path(directory)
    artifacts = []
    for role in ("weights", "tokenizer", "template"):
        content = f"synthetic {role}; not a model".encode()
        path = root / f"{role}.fixture"
        path.write_bytes(content)
        artifacts.append({"role": role, "path": path.name,
                          "sha256": hashlib.sha256(content).hexdigest()})
    dataset = b'{"id":"fixture-1","prompt":"Reply yes","expected":"yes"}\n'
    (root / "dataset.jsonl").write_bytes(dataset)
    value = {"schema": "hexweave-local-benchmark-v1", "purpose": "software_fixture",
             "model": {"id": "synthetic-fixture", "revision": "a" * 40,
                       "kind": "base", "artifacts": artifacts},
             "backend": {"name": "mock", "version": "1.0.0", "endpoint": "http://127.0.0.1:1234",
                         "model_alias": "synthetic-fixture", "compute_dtype": "fixture",
                         "quantization": "none", "context_tokens": 128,
                         "threads": 1, "kv_cache_dtype": "float32"},
             "dataset": {"id": "synthetic-fixture", "revision": "v1", "split": "fixture",
                         "path": "dataset.jsonl", "sha256": hashlib.sha256(dataset).hexdigest(),
                         "prompt_format": "raw_completion"},
             "generation": {"max_tokens": 8, "temperature": 0, "top_p": 1, "seed": 7, "stop": []},
             "budget": {"examples": 1, "repetitions": 1, "warmups": 0, "timeout_seconds": 2,
                        "max_wall_seconds": 5, "max_response_bytes": 1024}}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, value


class ManifestTests(unittest.TestCase):
    def test_pinned_bytes_load_and_changed_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, value = fixture(temporary)
            loaded = load_manifest(path)
            self.assertEqual(loaded["manifest"], value)
            self.assertEqual(len(loaded["rows"]), 1)
            path.with_name("weights.fixture").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                load_manifest(path)

    def test_malicious_urls_are_refused(self):
        urls = ["https://127.0.0.1:1234", "http://localhost:1234", "http://127.0.0.2:1234",
                "http://example.com:1234", "http://2130706433:1234", "http://127.0.0.1:1234@evil:90",
                "http://user:secret@127.0.0.1:1234", "http://127.0.0.1:1234/?secret=a",
                "http://127.0.0.1:1234/#x", "http://127.0.0.1:1234/v1", "http://127.0.0.1",
                "http://127.0.0.1:65536", "http://[::ffff:127.0.0.1]:1234"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                endpoint(url)
        self.assertEqual(endpoint("http://127.0.0.1:1234"), ("127.0.0.1", 1234))
        self.assertEqual(endpoint("http://[::1]:1234"), ("::1", 1234))

    def test_duplicate_keys_nonfinite_and_bad_bounds_are_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e9999}'):
            with self.assertRaises(ValueError):
                decode_json(raw)
        changes = [("budget", "examples", 0), ("budget", "repetitions", True),
                   ("budget", "max_wall_seconds", 601), ("generation", "temperature", float("inf")),
                   ("generation", "temperature", float("nan")), ("generation", "max_tokens", 513),
                   ("generation", "temperature", 10**1000),
                   ("model", "revision", "main"), ("backend", "version", "REPLACE_VERSION"),
                   ("backend", "threads", 0), ("backend", "threads", 5),
                   ("backend", "threads", True), ("backend", "kv_cache_dtype", "auto")]
        with tempfile.TemporaryDirectory() as temporary:
            path, value = fixture(temporary)
            for group, key, replacement in changes:
                modified = copy.deepcopy(value)
                modified[group][key] = replacement
                path.write_text(json.dumps(modified), encoding="utf-8")
                with self.subTest(field=f"{group}.{key}"), self.assertRaises(ValueError):
                    load_manifest(path)

    def test_json_is_strict_utf8_and_deep_nesting_is_validation_failure(self):
        raw = '{"message":"hello"}'
        self.assertEqual(decode_json(raw.encode("utf-8")), {"message": "hello"})
        for encoding in ("utf-16", "utf-32"):
            with self.subTest(encoding=encoding), self.assertRaisesRegex(ValueError, "UTF-8"):
                decode_json(raw.encode(encoding))
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            decode_json("[" * 10000 + "]" * 10000)

    def test_dataset_duplicate_ids_and_mutated_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, value = fixture(temporary)
            data_path = path.with_name("dataset.jsonl")
            content = data_path.read_bytes() * 2
            data_path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "Dataset hash mismatch"):
                load_manifest(path)
            value["dataset"]["sha256"] = hashlib.sha256(content).hexdigest()
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate dataset"):
                load_manifest(path)

    def test_unknown_auth_fields_and_duplicate_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, value = fixture(temporary)
            value["backend"]["api_key"] = "fixture-secret"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_manifest(path)
            del value["backend"]["api_key"]
            value["model"]["artifacts"].append(value["model"]["artifacts"][0])
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate artifact"):
                load_manifest(path)

    def test_results_require_new_owned_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ("outside.jsonl", "work/../outside.jsonl", str(root.parent / "escape.jsonl")):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    exclusive_output(root, path)
            with exclusive_output(root, "work/new/result.jsonl") as output:
                output.write("owned")
            with self.assertRaises(FileExistsError):
                exclusive_output(root, "work/new/result.jsonl")
