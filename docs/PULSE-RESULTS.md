# Ternary pulse implementation receipt

Date: 2026-10-23 local / 2026-10-24 04:28 UTC. This records a bounded software
experiment, not a useful pretrained model or full-size training admission.

## Implementation and independent review

The new `pulse` command has 1,464 learned ternary coefficients with packed
two-bit storage and no floating-point parameter masters. Selected nodes receive
predicted local transform/activation updates. Periodic frozen-path task anchors
train embedding/readout, transformations, activation controls, routers and the
correction predictor. Bounded history carries returned corrections with source,
shared and conservative all-tile version stamps.

Authoring and review were separate. Numerical and ownership review findings were
fixed before the integrated software sweep. A GPT-6 Luna monitoring sub-agent
independently inspected the terminal events and executable hashes and reported
back to the active parent conversation. It did not launch extra training.

See [the protocol guide](TERNARY-PULSE.md) and
[exact equations and limitations](../src/pulse/AGENTS.md).

## Integrated verification

One complete sweep after the code/review fix batch passed all nine stages:
structure, formatting, strict all-target Clippy, Rust tests, Python tests,
three allocation-free model plans, and the original FP32 smoke fixture.

- **41 Rust tests passed**, zero failures or ignored tests.
- **22 Python tests passed**, zero failures or skips.
- At sweep time: 40 source files and 12 top-level/docs Markdown files checked.
- Logs: [integrated validation](../work/validation/0f5183685bcb4e30bf7fbdcc1a877279).

The subsequent additions are this observation record and documentation pointers;
no numerical/runtime code was changed after the passing sweep. No model,
dataset, Python package or cloud service was downloaded or started.

## Bounded observations

Both modes used seeds 7, 8 and 9, each with 256 step opportunities, one resident
evidence slot and anchor interval four. Each seed therefore received 64 actual
training target exposures. Anchor-only skipped all 192 non-anchor opportunities.
Pulse performed predicted local work on those opportunities. Seeds were separate
initializations, not checkpoint resumes. Both used the same debug executable.

| Seed | Initial diagnostic loss | Final pulse loss | Final anchor-only loss | Pulse / anchor numerical seconds |
| --- | ---: | ---: | ---: | ---: |
| 7 | 2.828675 | 2.780842 | 2.793771 | 0.021102 / 0.006253 |
| 8 | 2.792794 | 2.714641 | 2.743535 | 0.019650 / 0.005549 |
| 9 | 2.788352 | 2.793887 | 2.763553 | 0.019392 / 0.005156 |

The diagnostic evaluates four heldout input IDs before and after training: eight
target evaluations, not eight distinct IDs. These input embeddings are not
directly trained. This is deliberately a weak synthetic diagnostic, not text
generalization or a basis for a time-to-quality forecast. No tuning followed
these observations.

Pulse had lower final loss on two seeds and higher loss on the third. It performed
1,904 training tile visits versus 490 for anchor-only, and 2,096 predictor forward
calls versus zero. Summed in-process debug runtime was approximately 3.55 times
the control. Tiny, single-shot timings are not reliable throughput benchmarks;
the larger amount of counted work is unambiguous. **No training-efficiency win
is established.** Matched-quality comparison on meaningful data is still needed.

All parameter families made actual trit transitions in pulse mode. Payloads were
366 bytes of packed weights, 928 bytes of shared integer evidence, at most 368
bytes of resident tile evidence, and a peak 672 bytes of history records. Other
payloads are broken out in each receipt. These exclude runtime/allocator and
temporary copies, and must not be presented as measured peak memory.

## Completion and monitoring

Both campaigns ended `completed`, with exactly three child starts and three
matching zero exits each, no retries and no failure events. Executable SHA-256:
`3e29d2a9fe1b2a8ef03ea991c31e74a88d4eddbe09b147734a51fbc683071743`.

- [Pulse receipt](../work/pulse/eaefa619887845be9c9dd4e97b70ead0/report.json)
- [Anchor-only receipt](../work/pulse/24bb9ae7460c40a2a8d67dd54063d5b8/report.json)

The owner processes exited and a final process-name check found no remaining
`hexweave` process. No unrelated process was stopped. Parent C/F training and
the 81M/287M profiles were not launched. Receipts explicitly record no checkpoint,
no resume and `native_chat_wakeup: false`. The lower-cost agent's in-conversation
callback worked; a persistent unattended session or closed-chat alert was not
installed. Healthy fixtures completed naturally at their requested bounds.

## Next research gate

Keep this as the numerical reference. The next useful experiment needs a
meaningful causal text task and a predictor-calibration criterion, with cost
accounted at matched quality. The predicted-routing head, stable evidence across
eviction, out-of-core weight residency and production checkpoints remain separate
implementation/review gates. Do not scale the current result into a long run on
the assumption that local updates have already reduced total training cost.
