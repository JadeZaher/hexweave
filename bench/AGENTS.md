# Benchmark implementation

`manifest.py` validates the request before any HTTP request or output creation.
`files.py` owns bounded JSON reading, content hashes and exclusive result paths.
`client.py` owns one direct loopback HTTP request at a time. `runner.py` records
the immutable request and per-example outcomes. `__main__.py` is the CLI.

There are no model downloads, subprocesses, service starts, proxies, redirects,
authentication headers or automatic retries. The caller owns the preexisting
server; this package never terminates it. Requests have a wall-time budget and
bounded bodies. Network failures are recorded and end the run.

The adapter measures complete HTTP request latency. It does not measure first
token latency, pure decode speed, server memory or likelihood. Native token
counts supplied by the server are identified as reported, not independently
verified. The supplied model/artifact identity is checked locally but the generic
HTTP server cannot prove it loaded those bytes. Results preserve that limitation.

Keep source files below 300 lines and functions below 90 lines. Rationale belongs
in this file and docs/BENCHMARKS.md; inline comments describe only non-obvious code.
Tests mock networking and use tiny synthetic files; no benchmark model is loaded.
