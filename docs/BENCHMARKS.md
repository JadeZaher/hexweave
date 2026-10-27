# Local model comparison contract

The scaffold supports bounded, sequential requests to a user-supplied existing
loopback completion server. It hashes declared local artifacts and the exact
dataset bytes, records the run plan and every prompt/output, and measures complete
HTTP request latency. It does not load a model, start a server or fetch artifacts.

Two comparisons answer different questions. A pretrained Granite comparison asks
whether our trained model is useful on this machine at a given quality, latency
and memory cost. Matched F+Hex training ablations ask whether the architecture or
credit rule helps under controlled data, initialization and exposure or compute.
Granite's pretraining and instruction tuning are not matched by parameter count.

## First candidates

The initial local comparison sizes are IBM's
[Granite 4.0 H 350M](https://huggingface.co/ibm-granite/granite-4.0-h-350m),
[Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m), and
[Granite 4.0 H 1B](https://huggingface.co/ibm-granite/granite-4.0-h-1b).
These are candidates, not downloaded or benchmarked models. Select the correct
base/instruct kind from the pinned model card and preserve its prompt template.
The named Granite 4.0 candidates are instruction-tuned; the example manifest
therefore uses `kind: instruct`.
Pin exact repository revisions and actual local file hashes before a run. Include
every weight shard and all tokenizer/template files used by the backend.

## Commands

From the repository root, with Python 3.12 or later:

```powershell
python -m bench validate --manifest benchmarks/granite.template.json
```

This template is intentionally incomplete and must fail. Make a separately named
manifest with actual local paths, immutable revision, hashes and backend version.
Relative input paths resolve against the manifest directory. Validation reads
artifact bytes to verify hashes; it does not allocate model tensors or contact the
endpoint. A raw completion model still needs a pinned template file describing
its raw prompt formatting, even when that template contains no wrapper tokens.

After validation succeeds and the intended local server is already running:

```powershell
python -m bench run --manifest benchmarks/my-pinned-run.json --output work/bench/my-new-run.jsonl
```

Results require a new path under this repository's `work/`; existing files are
never overwritten. The adapter sends one non-streaming `/v1/completions` request
at a time. It accepts only explicit numeric loopback URLs such as
`http://127.0.0.1:1234` and `http://[::1]:1234`. DNS, proxies, credentials, URL
queries, path prefixes and redirects are unsupported. The response must echo the
requested model alias when it includes a `model` field.

The adapter validates an exact schema and rejects unknown fields (including API
keys), non-UTF-8 JSON, duplicate JSON keys/IDs/paths, nonfinite values and invalid bounds. It caps
examples, repetitions, warmups, prompt/body bytes and total/request time. Request
deadlines close only that request's socket. A failed request is persisted and ends
the run; there is no automatic retry. Ctrl-C produces a stopped result when
interrupted during a request. The CLI exits nonzero for failed or stopped runs.

## Data and measurements

The dataset is UTF-8 JSONL. Every row has exactly `id`, `prompt` and `expected`;
`expected` may be null. Prompts are final rendered strings. The strict exact-match
score compares returned text without silently trimming, case-folding or extracting
an answer. A real task needing a richer scorer needs an explicit reviewed adapter.

The result starts with the manifest and SHA256 identities. Per-request rows keep
the full prompt/output, UTF-8 byte counts, server-reported native token counts,
finish reason, truncation and failures. Warmups are labeled and excluded from
the summary. Model and dataset identities are sufficient to recheck the declared
input bytes, but the generic HTTP protocol cannot prove the server loaded those
files or honored the recorded backend/context/dtype settings. Results explicitly
set `artifact_endpoint_binding_verified=false` and `leaderboard_eligible=false`.
Direct backend provenance and runtime telemetry are later integration work.

| Measurement | Current support |
| --- | --- |
| Complete request seconds, mean and median | Measured, including HTTP and server work |
| Output UTF-8 bytes per complete-request second | Measured; not pure decoding throughput |
| Native input/output token counts | Recorded if the server provides them; unverified |
| Exact-match quality | Available only for rows with an expected string |
| Time to first token / pure decode speed | Null: non-streaming adapter cannot measure them |
| Peak process RSS / VRAM / energy | Null: no telemetry adapter yet |
| NLL / bits per byte / perplexity | Null: no likelihood adapter yet |
| p95 latency | Null: inspect raw observations; no small-sample tail claim |

Run batch one, one model at a time, with declared quantization/context/KV and thread
settings. The manifest requires `backend.threads` between one and four and an
explicit `backend.kv_cache_dtype`; these describe the intended existing server
configuration. The generic endpoint cannot verify or set either value, so they
remain unverified metadata in the result. Separate model loading/cold requests
from steady state. Report output
lengths alongside latency because EOS and tokenizer choices change realized work.
The declared context size is metadata; exact tokenizer-aware admission is not
implemented. The caller and server must enforce prompt-plus-output context limits.
The run's wall-time budget applies to HTTP execution after input validation;
streaming artifact hashes before the run are not counted as inference time.

Do not compare raw token perplexity across tokenizers. A future likelihood adapter
should evaluate identical raw continuations and report total negative log
likelihood divided by scored UTF-8 bytes, with explicit context, BOS and boundary
rules. Compare base models with base models; place instruct models in a separate
table with their documented templates. Never call a synthetic fixture a benchmark.

For F+Hex training, record useful completed targets separately from tile visits,
forks, pulse work, reverse corrections and transfers. Compare held-out quality per
training second as well as exposure. The existing CPU fixture and this adapter do
not establish Granite-level quality, lower residency or training-time advantages.
