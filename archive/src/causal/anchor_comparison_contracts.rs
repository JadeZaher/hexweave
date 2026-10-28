use super::execution::{self, anchor_only_fixture, Step, TrainingSource};
use super::geometry::Geometry;
use super::state::State;
use anyhow::{ensure, Result};
use serde_json::json;
use std::time::Instant;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;
#[path = "anchor_comparison_identity.rs"]
mod identity;
#[path = "anchor_comparison_measurement.rs"]
mod measurement;
#[path = "anchor_comparison_source.rs"]
mod source;
use fixture::Store;
use measurement::{Budget, Observation};
use source::{Source, ANSWERS, HELD_OUT, TRAIN};

const LABEL: &str = "ANCHOR_COMPARISON_V3 ";

struct Arm {
    state: State,
    store: Store,
    source: Source,
    current: bool,
    anchor_losses: Vec<f32>,
    observations: Vec<Observation>,
    learning_cycle_micros: u128,
}

impl Arm {
    fn new(seed: u64, current: bool) -> Result<Self> {
        let source = Source::new();
        Ok(Self {
            state: State::with_cursor(Geometry::candidate(), seed, source.cursor(0))?,
            store: Store::with_geometry(Geometry::candidate(), seed)?,
            source,
            current,
            anchor_losses: Vec::with_capacity(8),
            observations: Vec::with_capacity(3),
            learning_cycle_micros: 0,
        })
    }

    fn observe(&mut self, cycles: usize, budget: &mut Budget) -> Result<()> {
        self.observations.push(measurement::observe(
            &self.state,
            &mut self.store,
            cycles,
            budget,
        )?);
        Ok(())
    }

    fn cycle(&mut self, index: usize, budget: &Budget) -> Result<()> {
        budget.check()?;
        ensure!(self.state.global_opportunity == 4 * index as u64, "clock");
        let started = Instant::now();
        let loss = if self.current {
            match execution::step(&mut self.state, &mut self.store, &self.source, false)? {
                Step::Committed {
                    opportunity,
                    loss: Some(loss),
                    ..
                } if opportunity == 4 * index as u64 => loss,
                other => anyhow::bail!("expected production anchor: {other:?}"),
            }
        } else {
            anchor_only_fixture::step(&mut self.state, &mut self.store, &self.source)?
        };
        ensure!(loss.is_finite(), "anchor pre-update loss");
        self.anchor_losses.push(loss);
        if self.current {
            for local in 1..=3 {
                budget.check()?;
                let cursor = self.state.cursor.clone();
                let reads = self.source.target_reads.get();
                let advances = self.source.advances.get();
                let step = execution::step(&mut self.state, &mut self.store, &self.source, false)?;
                ensure!(
                    matches!(step, Step::Committed { opportunity, loss: None, .. }
                        if opportunity == 4 * index as u64 + local),
                    "expected production local"
                );
                ensure!(
                    self.state.cursor == cursor
                        && self.source.target_reads.get() == reads
                        && self.source.advances.get() == advances,
                    "local accessed target or advanced cursor"
                );
            }
        }
        self.learning_cycle_micros += started.elapsed().as_micros();
        ensure!(
            self.state.global_opportunity == 4 * (index + 1) as u64,
            "clock"
        );
        budget.check()
    }

    fn validate_final(&self) -> Result<()> {
        let (state, counters) = (&self.state, &self.state.counters);
        self.source.validate(&state.cursor)?;
        ensure!(
            state.global_opportunity == 32
                && counters.anchors == 8
                && counters.local_opportunities == if self.current { 24 } else { 0 }
                && self.source.target_reads.get() == 8
                && self.source.advances.get() == 8
                && self.anchor_losses.len() == 8,
            "comparison exposure mismatch"
        );
        ensure!(
            counters.predictor_anchor_updates == 8
                && counters.predictor_anchor_pairs == counters.visits - counters.local_tile_updates
                && counters.predictor_replay_updates <= counters.local_opportunities
                && counters.predictor_replay_pairs <= 4 * counters.predictor_replay_updates
                && counters.predictor_transitions
                    <= 128
                        * (counters.predictor_anchor_updates + counters.predictor_replay_updates),
            "comparison predictor accounting"
        );
        ensure!(
            self.store.updates.iter().sum::<usize>() as u64
                == counters.anchor_tile_commits + counters.local_tile_updates
                && state.shared.version <= 8
                && state.predictor.version
                    <= counters.predictor_anchor_updates + counters.predictor_replay_updates
                && state
                    .tile_versions
                    .iter()
                    .zip(self.store.updates)
                    .all(|(&version, calls)| version <= calls as u64),
            "comparison commit accounting"
        );
        if self.current {
            state.validate_boundary()?;
        } else {
            ensure!(
                counters.predictor_replay_updates == 0
                    && counters.predictor_replay_pairs == 0
                    && counters.local_tile_updates == 0
                    && (8..=32).contains(&counters.visits)
                    && counters.halts + counters.forced_exits == 8
                    && counters.sampled_routes == counters.visits - counters.forced_exits
                    && state.validate_boundary().is_err(),
                "anchor-only ledger fabricated production work"
            );
        }
        Ok(())
    }

    fn report(&self) -> serde_json::Value {
        json!({
            "method": if self.current { "current_anchor_plus_three_locals" } else { "anchor_only" },
            "production_boundary_valid": self.current,
            "virtual_clock": self.state.global_opportunity,
            "actual_learning_opportunities": self.state.counters.anchors + self.state.counters.local_opportunities,
            "cursor": self.state.cursor, "counters": self.state.counters,
            "tile_update_calls_by_id": self.store.updates, "tile_versions": self.state.tile_versions,
            "shared_version": self.state.shared.version, "predictor_version": self.state.predictor.version,
            "target_reads": self.source.target_reads.get(), "cursor_advances": self.source.advances.get(),
            "supervised_sequence": *self.source.sequence.borrow(),
            "anchor_pre_update_losses": self.anchor_losses,
            "learning_cycle_micros": self.learning_cycle_micros,
            "observations": self.observations,
            "digit_retention_final_minus_cycle_four_nll":
                self.observations[2].train_digit_mean_nll - self.observations[1].train_digit_mean_nll,
        })
    }
}

fn payload_bound() -> Result<usize> {
    let geometry = Geometry::candidate();
    let ledger = geometry.ledger(65)?;
    let two_histories = 2 * (8 * (4 * (geometry.features() + geometry.width) + 80) + 16);
    let bounded_reports = 4 * 64 * 1024;
    let scalar_and_prefix_reserve = 128 * 1024;
    let bound = 2 * ledger.stored_weight_bytes
        + two_histories
        + ledger.anchor_core_payload_bound
        + ledger.predictor_phase_payload_bound
        + bounded_reports
        + scalar_and_prefix_reserve;
    ensure!(bound < 16 * 1024 * 1024, "fixture logical payload ceiling");
    ensure!(ledger.coefficients == 2_051_840, "candidate geometry drift");
    ensure!(TRAIN.iter().all(|p| p.len() <= 64), "train prefix cap");
    ensure!(
        HELD_OUT.iter().all(|p| p.len() + 15 <= 65),
        "held-out prefix cap"
    );
    Ok(bound)
}

fn scenario(seed: u64, budget: &mut Budget, payload: usize) -> Result<usize> {
    budget.check()?;
    let before_readouts = (budget.teacher_readouts, budget.generated_readouts);
    let mut anchor = Arm::new(seed, false)?;
    let mut current = Arm::new(seed, true)?;
    ensure!(
        identity::identity(&anchor.state, &mut anchor.store)?
            == identity::identity(&current.state, &mut current.store)?,
        "comparison arms have unequal starts"
    );
    anchor.observe(0, budget)?;
    current.observe(0, budget)?;
    for index in 0..8 {
        anchor.cycle(index, budget)?;
        current.cycle(index, budget)?;
        if index == 3 || index == 7 {
            anchor.observe(index + 1, budget)?;
            current.observe(index + 1, budget)?;
        }
    }
    anchor.validate_final()?;
    current.validate_final()?;
    let expected: Vec<_> = (0..2)
        .flat_map(|offset| (0..4).map(move |case| (case, offset, ANSWERS[case][offset])))
        .collect();
    ensure!(
        *anchor.source.sequence.borrow() == expected
            && *current.source.sequence.borrow() == expected,
        "comparison supervised sequence mismatch"
    );
    let report = json!({
        "schema": 1, "kind": "bounded_anchor_comparison_v3_fixture", "seed": seed,
        "geometry": Geometry::candidate(), "kernel": crate::ternary::kernel_name()?,
        "corpus_sha256": source::corpus_identity(),
        "nll_order": "A_digit,B_digit,C_digit,D_digit,A_LF,B_LF,C_LF,D_LF",
        "measurement_history": "zero", "measurement_routes": "immutable greedy current weights",
        "logical_payload_bound_bytes": payload, "logical_payload_ceiling_bytes": 16 * 1024 * 1024,
        "teacher_readouts": budget.teacher_readouts - before_readouts.0,
        "generated_readouts": budget.generated_readouts - before_readouts.1,
        "payload_scope": "conservative logical payload only; allocator overhead and process RSS unmeasured",
        "anchor_only": anchor.report(), "current_method": current.report(),
        "scope": "test-only equal-supervised-exposure comparison; unequal work; no production schedule, quality, ETA or campaign admission",
    });
    let encoded = serde_json::to_string(&report)?;
    let bytes = LABEL.len() + encoded.len() + 1;
    ensure!(bytes <= 64 * 1024, "seed report ceiling");
    budget.check()?;
    println!("{LABEL}{encoded}");
    Ok(bytes)
}

#[test]
fn fixed_anchor_only_versus_current_method() -> Result<()> {
    let mut budget = Budget::new();
    let payload = payload_bound()?;
    let mut report_bytes = 0;
    for seed in [7, 17, 29] {
        report_bytes += scenario(seed, &mut budget, payload)?;
    }
    ensure!(budget.teacher_readouts == 288, "teacher readout accounting");
    ensure!(
        budget.generated_readouts <= 384,
        "generation readout accounting"
    );
    ensure!(report_bytes <= 192 * 1024, "total report ceiling");
    budget.check()
}
