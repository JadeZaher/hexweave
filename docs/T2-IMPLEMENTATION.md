# T2 implementation and authored fixture inventory

This is an author handoff, not validation evidence or a run result. The
[predeclared contract](T2-CREDIT-DIAGNOSTIC.md) fixes all data, budgets and decisions.
No optimizer, ternary initialization, forward geometry, actual gradient arithmetic,
default mode or update order was changed. Shadow work adds overhead that must
fit the same bound; no subsequent tuning is implied.

| Area | Changed responsibility |
| --- | --- |
| `src/text/mod.rs`, `src/text_cli.rs` | Explicit fixture/T2 profile and on/off diagnostic identity; defaults retained |
| `src/text/data.rs`, `datasets/tiny/development-t2.txt` | Frozen fresh development bytes, exact first-72-byte evaluation identity; reject old validation under T2 |
| `src/text/credit.rs` | Measure only after repeated frames aggregate and before each original unique-tile apply; reconcile returned transitions |
| `src/text/credit_diagnostic.rs` | Independent task/coupled/route-only shadows, exact actual-gradient parity, per-anchor difference accounting |
| `src/text/credit_metrics.rs` | Frozen-trit deltas, raw f64 sums and per-tile/run coordinate/transition metrics; null ratios |
| `src/text/execution.rs`, `runner.rs`, `report.rs` | Optional observational state and explicit v4 receipt/snapshot metadata; inspection-only snapshots |
| `scripts/text_profile.py`, `text_credit_io.py`, `text_io.py` | Frozen SHA/FNV/prefix and strict raw/ratio/phase/per-tile reconciliation |
| `scripts/text_runner.py`, `text_campaign.py` | Existing supervised ownership reused, explicit T2-only campaign, predeclared mechanism-plus-quality decision, inherited late STOP/deadline persistence guards |

The diagnostic contract includes anchors and anchor differences; per-tile/run
frames, unique commits, repeated commits and 512 coordinate opportunities per
commit; task/route nonzero, intersection/opposition; dot and squared norms;
task/coupled/actual/applied transitions; changed results, opposing, suppressed
and new changes; task descent, coupled projection, deficit and residual maximum.
Ratios expose their denominators, including null for zero task descent/norms.
Local updates remain in the existing full transition counters, outside these
anchor-only diagnostic totals.

The largest tile's actual-update gradient payload is 6,656 bytes. Enabled
diagnostics add three full shadow bundles (19,968 bytes), for 26,624 concurrent
bundle bytes; off reports zero shadow count/bytes. Evidence and other scratch
remain separate and these selected payload bounds do not represent complete heap
or RSS. The existing on/off receipt fixture asserts both scopes.

Authored coverage, not yet executed:

- `src/text/credit_diagnostic_tests.rs`: explicit cancellation and saturation,
  zero-denominator nulls, nonfinite rejection, predicted versus applied counts;
  a forced legal center-neighbor-center anchor path in both arms compares exact
  state, versions, counters and disk I/O with diagnostics on/off.
- `tests/text_contract.rs`: sampled-route on/off equality in both arms, target
  accounting and exact shared state; T2 rejects the old evaluation document.
- `tests/test_text_credit.py`: missing/raw/ratio/per-tile/projection accounting,
  exact differing-result partition, zero-gradient/norm/descent implications and
  impossible transition bounds; old data/protocol and missing diagnostics rejected
  by campaign checks.
- Existing CLI/owner/campaign fixtures extended for explicit profile/on-off,
  frozen-input rejection, consumed-T1 refusal, GO requiring both mechanism and
  transfer quality, and inherited late STOP/deadline/corrective persistence.

Both review cycles have completed; the final resource-accounting correction awaits
targeted readback, then one integrated sweep remains before execution. T2 GO is
provisional; INCONCLUSIVE still permits T3 implementation with an explicitly
experimental detached candidate. Neither outcome validates assistant quality,
changes the default or launches another run. REVISE supports no candidate
selection. Recorded runtime includes diagnostics; incremental instrumentation
overhead is not isolated by the six diagnostic-on runs. Time to a useful small-domain
assistant remains unsupported by this tiny diagnostic.
