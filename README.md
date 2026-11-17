
# Hexweave

**Goal:** train a language model whose parameter count is not capped by RAM, under
a memory budget you choose, on commodity hardware, with local updates that can run
continually. Hexweave pages small learned tiles to and from disk and trains each
one from local credit, trading training time and some per-step quality for bounded
residency.

**Where it stands (CPU, one seed, one small corpus):** a 2.05M-coefficient
byte-level model with 6 of 7 tiles resident reached a held-out mean of about 2.65
bits/byte on TinyStories at 1.0–1.2M training positions, in about 50 MB of working
memory and 8.5 GB of SSD writes. A matched bigram scores 3.31 and the project's own
non-paged reference path 2.216. These are early research results, not a trained
assistant; the [paper](paper/main.pdf) states the method, measurements and limits.

Licensed under [Apache-2.0](LICENSE). To cite, see [CITATION.cff](CITATION.cff).
Related GPU work lives in the sibling repository `hexweave-gpu`.

A local-first **F-inspired hex-routed language-model laboratory**. Small learned
blocks live on an axial hex grid; a learned router chooses where computation goes,
continuous activation curves shape the selected transformation, and a learned
pulse carries context between visits.

Start with [the intention guide](docs/GUIDE.md), then the
[current architecture and bottleneck walkthrough](src/causal/STUDY.md).
The [FP32 reference architecture](docs/ARCHITECTURE.md) describes the older
`smoke` fixture. This is a new experiment, not a migrated
C/F checkpoint or a pretrained Granite competitor. The trainable model is the
small ternary byte-level causal candidate (2,051,840 coefficients on seven hex
tiles), trained by the core path below. The FP32 `smoke` fixture is a tiny
numerical reference; the 81M/287M profiles are allocation-free planning inputs.
It is not yet a trained assistant.

The [three performance loops](docs/PERFORMANCE-LOOPS.md) record measured SIMD and
allocation improvements. The earlier ternary-pulse, disk-local text and
joint-pulse experiments, and the v3 owned-study chain, are archived and no longer
build; [the archive guide](archive/README.md) lists what moved and how to restore
it.

## First five minutes

Prerequisites: a recent stable Rust toolchain and Python 3.12 or later. No Python
packages, model weights, datasets, local servers or VS Code extensions are
installed automatically. Cargo may fetch ordinary Rust dependencies when not
using `--offline`; the dependency lockfile is part of this repository.

From this directory in PowerShell:

```powershell
$env:CARGO_TARGET_DIR = Join-Path (Get-Location) 'work/target'
cargo run --locked -- plan --config configs/local-287m.json
cargo run --locked -- smoke --steps 4 --seed 7
cargo run --locked -- causal plan
```

The first command reports geometry and training-state arithmetic without loading
the planned model. The second performs at most 16 tiny CPU updates and exits.
The third prints the causal candidate's configuration and payload ledger without
allocating weights or opening data. These are software checks, not a training
run; training the current candidate is the core path below. The 81M and 287M
profiles stay planning-only.

## Training (core path)

`hexweave train --config <json>` selects a training protocol through its config.
With `"connected": {}`, `causal-connected-v5` trains the tile transformations
using learned correction predictions. Eight logical lanes share structured
memory for routing and activation-curve conditioning, with at most four decoded
tile/Adam bundles and a bounded Lambert W gain scaled by predictor trust (zero
skips the local update). Connected protocol v3 adds one Adam step per visited
tile per anchor, decoupled weight decay with a master-RMS cap
(`connected.weight_decay` 0.01, `connected.master_rms_cap` 1.5), a saturating
activation curve and a learned readout bias
([v3 changes](src/causal/CONNECTED.md#scale-control-saturating-curve-readout-bias-and-per-anchor-local-steps-v3)).
Connected protocol v4 RMS-normalizes every tile's output after the residual sum
and, on a fresh start, initializes the readout bias to the training corpus' byte
log-frequencies; the v4 configs set `connected.weight_decay` to 0.001
([v4 changes](src/causal/CONNECTED.md#residual-normalization-and-readout-prior-v4)).
Connected protocol v5 conditions the correction predictor on the observed target
byte's embedding feature and scales each tile update's decoupled decay by the
multiplier its step received
([v5 changes](src/causal/CONNECTED.md#label-conditioned-credit-predictor-and-gain-scaled-decay-v5)).
v5.1 adds `connected.local_rate` (default 3e-4), the Adam rate of those
predicted-credit tile steps; E, the bias and routes keep `rate`, and the predictor
keeps `connected.predictor_rate`. v5.2 adds `connected.local_decay` (default
`{"kind":"constant"}`); `{"kind":"inverse_sqrt","anchors":N}` scales the local rate
by `1 / sqrt(1 + anchor / N)`. v5.4 adds optional `connected.credit_mode`
(`"lanes"`, `"anchor_exact"`, `"anchor_predicted"`, `"hybrid"`) and
`connected.slice_mode` (`"full"`, `"windowed"`, `"hybrid"`), omitted from run
identity when left at their defaults.
The predictor is active and checkpointed. v1–v4 runs can be repeated from scratch
only under their pinned executables; their weights were removed on 2026-11-07
([retrospectives](docs/CONNECTED-RETROS.md)). See [the connected design](src/causal/CONNECTED.md)
and [verification and run status](docs/STATUS.md).

Set `"connected": {"precision": "fp32"}` for continuous FP32 forward weights
under `causal-connected-fp32-v5`. This keeps the connected learning mechanisms
and four-tile residency bound; it starts from fresh state under its own precision
identity. Omitted precision selects the ternary `causal-connected-v5`. The legacy
top-level `float_forward` option belongs to the anchor-only path.

Configs without `connected` retain the `causal-latent-v2` anchor-only reference,
with all tiles resident. Both protocols use resumable hash-manifested
checkpoints. Run training under the Job-capped owner, which pins the
binary, enforces memory floors and writes receipts:

```powershell
cargo build --release --locked
python -B -m scripts.core_owner --config work/core-runs/<run>.json
```

A config names `out` (`work/core-runs/<name>`), `data`
(`{"kind":"science","root":…,"manifest_sha256":…}`, optionally with
`"redeal_citations":true`, or
`{"kind":"drill","root":"datasets/retained-text-assay"}`), `seed`, `rate`,
`anchors`, `wall_seconds`, `checkpoint_every`, `eval_every`, `eval_cases` and
`resume`. Write `out/STOP` (or press Ctrl+C in the owner) to end a run with a
final checkpoint; set `resume` to continue it from the newest checkpoint that
verifies. V2 science and corpus runs also keep their lowest-held-out-loss checkpoint in `out/best/`.
For connected runs, add `--resume-on-memory` to the owner command for bounded
recovery after clean memory-pressure stops. It retains ownership while waiting
for stable available memory and deducts elapsed attempts from the configured time
budget. Operator STOP cancels recovery. Retry and wait limits, verified-checkpoint
admission and per-attempt receipts are documented in [the owner contract](scripts/AGENTS.md).
For plain-text pretraining, prepare a corpus with `python -B -m
scripts.prepare_corpus --source <stories.txt> --out work/corpora/<name>` and use
`{"kind":"corpus","root":…,"manifest_sha256":…,"window":512}`; in v2 add
`"float_forward":true` to train with f32 weights. Connected mode currently
requires corpus data, history enabled, batch one, constant
base rate and zero v2 exploration. A fresh connected run uses a new `out` and
`"resume":false`; it initializes weights, predictor, Adam, memory and gains
without importing any earlier run state.
Design and invariants: [core path](src/causal/AGENTS.md); owner:
[scripts/AGENTS.md](scripts/AGENTS.md).

To request a bounded story continuation from a specific v2 or connected corpus
checkpoint:

```powershell
cargo run --locked --offline -- complete --config work/core-runs/<run>.json --checkpoint a000000392640 --prompt "Once upon a time" --max-new-bytes 160 --top-p 0.9 --seed 1
```

The checkpoint name must identify a complete checkpoint under that run. The
command verifies its identity and hashes and returns JSON with the generated
text, raw bytes, UTF-8 validity and stop reason. For a v2 checkpoint the prompt
ends a full training window of public held-out text, as training sees its
input; without `--top-p` it decodes greedily, and `--temperature` (default 1.0)
and `--seed` (default 0) apply only with it. Connected completion may run
beside a live training owner: it first snapshots the named checkpoint into
temporary `work/inference/` storage, then copies verified bundles from that
snapshot, retains at most four decoded tile/Adam bundles, and removes its scratch
directory on exit. Check available host memory before running it beside a live
owner, whose memory floor STOP would consume a recovery restart. The run's
existing STOP file remains in place and does not suppress inference. No corpus
is opened.

Connected before/after fingerprints cover the snapshot of the saved files and
the loaded weights,
Adam state, predictor, gains and training memory. Generation uses isolated memory
with frozen parameters, following the same replay policy as connected evaluation.
The v2 fingerprint covers packed model and continuation state, excluding its
in-memory latent optimizer map. Story generation stops before the end-of-story
marker or at the byte limit, using the most recent corpus-window bytes for each
prediction. Connected completion also checks a 120-second deadline between bytes;
final verification and cleanup finish afterward. The prompt
and leading story marker must fit the model context; the output limit is at most
1,023 bytes. The current tiny candidate has not demonstrated
useful completions.

For a bounded connected decoder audit, add `--trace-bytes 16`. The response
includes the actual logits and selected bytes, visible prefixes, space rank,
routes, memory counts and a scalar readout check. Add `--feedback prompt-only`
to compare against frozen post-prompt memory as a labeled diagnostic; normal
completion uses `--feedback generated`. These options read an isolated checkpoint
and do not change training.

For an FP32 connected checkpoint, `--quantize-audit` compares FP32, scaled INT8
and scaled ternary tile/embedding weight views with identical prompt warmup.
The correction predictor and activations remain FP32. Add tracing to
inspect each view's actual scores. The request preserves the original checkpoint
and reports each variant separately, including budget truncation. These are
FP32 computations on rounded weights, not native FP8 kernels or compressed
checkpoint exports; no lower-precision mode is automatically promoted.

For the final integrated checks after a batch of edits:

```powershell
.\scripts\validate.ps1
# Add -Offline when all locked Cargo dependencies are already cached.
# If the default Windows policy blocks this locally authored script:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate.ps1 -Offline
```

Builds and validation logs stay in ignored `work/`. The script uses two build
jobs, bounded backend thread pools, sequential tests and below-normal process
priority. It starts no background training or model server.
The alternative command and editor validation task use a process-only policy
override; they do not change user or machine execution-policy settings.

## Model sizes

| Profile | Hex tiles | Hidden / FFN | Main parameters | Executable status |
| --- | ---: | ---: | ---: | --- |
| Tiny | 7 | 16 / 32 | 22,016 plus auxiliary weights | CPU fixture only |
| Development | 19 | 512 / 1,536 | 81.53M plus auxiliary weights | Plan only |
| Local comparison | 37 | 768 / 2,048 | 287.05M plus auxiliary weights | Plan only |

The 287M target is a useful size class for comparison with small local models,
including IBM Granite's 350M variants. Size proximity does not imply equal
capability, data exposure, inference work or training cost. On this machine's
8 GB RTX 4060, the 16-byte/main-parameter training-state estimate is about
4.28 GiB **before** activations, pulse/controller state, logits, temporary storage
and the 1 GiB device reserve. No full-size training fit has been measured.

## Read the code in this order

| Question | Entry point |
| --- | --- |
| What happens when I run it? | [CLI](src/main.rs) |
| How is size estimated? | [Configuration](src/config.rs) |
| Where can a route go? | [Hex topology](src/topology.rs) |
| How is fanout bounded? | [Metadata planner](src/planner.rs) |
| What really trains? | [Core path](src/causal/AGENTS.md), [FP32 smoke fixture](src/smoke.rs), [model](src/model/AGENTS.md) |
| Who owns a training run? | [Core training owner](scripts/AGENTS.md) |
| Where did pulse, text and the v3 owner chain go? | [Archive guide](archive/README.md), [pulse guide](archive/docs/TERNARY-PULSE.md), [text guide](archive/docs/TEXT-PULSE.md) |
| Which optimizations actually helped? | [Three measured loops](docs/PERFORMANCE-LOOPS.md) |
| How will Granite be compared? | [Benchmark protocol](docs/BENCHMARKS.md) |
| What is still missing? | [Status and gates](docs/STATUS.md) |
| What did each connected version teach? | [Connected retrospectives](docs/CONNECTED-RETROS.md) |
| Is the connected architecture feasible or useful yet? | [Feasibility and utility](docs/CONNECTED-FEASIBILITY.md) |
| What was independently reviewed? | [Review record](docs/REVIEWS.md) |

Open `hexweave.code-workspace` in VS Code for the plan, bounded fixture and
validation tasks. No editor task launches a large model. See
[CONTRIBUTING.md](CONTRIBUTING.md) for review and readability conventions.
