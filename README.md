# Hexweave

A local-first **F-inspired hex-routed language-model laboratory**. Small learned
blocks live on an axial hex grid; a learned router chooses where computation goes,
continuous activation curves shape the selected transformation, and a learned
pulse carries context between visits.

Start with [the intention guide](docs/GUIDE.md), then the
[architecture map](docs/ARCHITECTURE.md). This is a new experiment, not a migrated
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

`hexweave train --config <json>` is the production training path: the same
hex-routed ternary model, trained with latent-weight Adam, eval-matched greedy
routes and an anchor-only schedule, all tiles resident, with resumable
hash-manifested checkpoints. Run it under the Job-capped owner, which pins the
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
verifies. Science runs also keep their lowest-dev-loss checkpoint in `out/best/`.
Design and invariants: [core path](src/causal/AGENTS.md); owner:
[scripts/AGENTS.md](scripts/AGENTS.md).

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
| What was independently reviewed? | [Review record](docs/REVIEWS.md) |

Open `hexweave.code-workspace` in VS Code for the plan, bounded fixture and
validation tasks. No editor task launches a large model. See
[CONTRIBUTING.md](CONTRIBUTING.md) for review and readability conventions.
