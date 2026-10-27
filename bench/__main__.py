"""Command-line benchmark entry point; see docs/BENCHMARKS.md."""

import argparse
import json

from .manifest import load_manifest
from .runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate or run bounded local completion benchmarks")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="Verify pins and local bytes without networking")
    validate.add_argument("--manifest", required=True)
    execute = subcommands.add_parser("run", help="Query an existing explicitly supplied loopback endpoint")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--output", required=True, help="New JSONL file under this repository's work/")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            snapshot = load_manifest(arguments.manifest)
            result = {"valid": True, "manifest_sha256": snapshot["manifest_sha256"],
                      "examples": len(snapshot["rows"]), "model_loaded": False,
                      "network_contacted": False, "server_artifact_binding_verified": False}
        else:
            result = run(arguments.manifest, arguments.output)
        print(json.dumps(result, allow_nan=False))
        return 0 if arguments.command == "validate" or result["complete"] else 1
    except (OSError, ValueError) as error:
        parser.exit(2, f"Benchmark refused: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
