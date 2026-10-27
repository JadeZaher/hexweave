# Hexweave

A local-first **F-inspired hex-routed language-model laboratory**. Small learned
blocks live on an axial hex grid; a learned router chooses where computation goes,
continuous activation curves shape the selected transformation, and a learned
pulse carries context between visits.

Start with [the intention guide](docs/GUIDE.md), then the
[architecture map](docs/ARCHITECTURE.md). This is a new experiment, not a migrated
C/F checkpoint or a pretrained Granite competitor. The initial executable is a
tiny CPU numerical fixture. Larger profiles are allocation-free planning inputs.
The [ternary pulse protocol](docs/TERNARY-PULSE.md) is a separate bounded
node-local experiment with real ternary storage and periodic task corrections.
It is not yet the full attention model or a trained assistant.

The [three performance loops](docs/PERFORMANCE-LOOPS.md) add measured SIMD and
allocation improvements. The [disk-local text experiment](docs/TEXT-PULSE.md)
adds real next-byte examples and one loaded hex tile at a time, with shared
weights counted separately. Neither enables full-size training.

**Where the training pulse happens:** [the joint-pulse guide](docs/TRAINING-PULSE.md)
maps the shared ternary context to transform gain, activation-curve output shift
and routing bias, and shows where their training credit joins.

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
cargo run --locked -- pulse --steps 64 --seed 7 --mode pulse
```

The first command reports geometry and training-state arithmetic without loading
the planned model. The second performs at most 16 tiny CPU updates and exits.
The third runs the ternary fixture. These are software checks, not a language-model training campaign or convergence
test. There is deliberately no full-size `train` or `resume` command.

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
| What really trains? | [Numerical fixture](src/smoke.rs), [model](src/model/AGENTS.md) |
| Where is ternary-first learning? | [Pulse guide](docs/TERNARY-PULSE.md), [pulse code map](src/pulse/AGENTS.md) |
| Where is disk-local text training? | [Text guide](docs/TEXT-PULSE.md), [text code map](src/text/AGENTS.md) |
| Where do tile-wide training signals join? | [Joint training pulse](docs/TRAINING-PULSE.md), [implementation](src/text/training_pulse.rs) |
| Which optimizations actually helped? | [Three measured loops](docs/PERFORMANCE-LOOPS.md) |
| How will Granite be compared? | [Benchmark protocol](docs/BENCHMARKS.md) |
| What is still missing? | [Status and gates](docs/STATUS.md) |
| What was independently reviewed? | [Review record](docs/REVIEWS.md) |

Open `hexweave.code-workspace` in VS Code for the plan, bounded fixture and
validation tasks. No editor task launches a large model. See
[CONTRIBUTING.md](CONTRIBUTING.md) for review and readability conventions.
