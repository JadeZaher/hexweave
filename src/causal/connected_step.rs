//! Observed anchor followed by predictor-driven local lanes; see AGENTS.md.

use super::connected_anchor::{self, Credit, Trace};
use super::connected_config::{CreditMode, Options};
use super::connected_lane_apply::Accumulator;
use super::connected_restart::{self as restart, Mark};
use super::connected_signal::{Kind, Record};
use super::connected_state::State;
use super::connected_store::Store;
use super::connected_update;
use super::execution::TrainingSource;
use super::geometry::TILES;
use super::shared::Prefix;
use super::tile_gradients::TileGradients;
use super::tuning;
use anyhow::{ensure, Result};

/// Only a complete anchor-plus-local phase can become checkpointable.
pub(super) fn step<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    threaded: bool,
) -> Result<f32> {
    state.validate()?;
    anyhow::ensure!(
        store.precision == options.precision && state.options == options,
        "connected step precision/options mismatch"
    );
    store.quiescent()?;
    source.validate(&state.cursor)?;
    tuning::set_connected(&options);
    state.ready = false;
    let mark = Mark::of(state);
    // Restarts run once the anchor's updates are all committed (CONNECTED.md §Warm restarts).
    let result = complete(state, store, source, options, rate, threaded)
        .and_then(|loss| restart::finish(state, store, source, &mark, loss).map(|()| loss));
    if result.is_ok() {
        state.ready = true;
        if let Err(error) = state.validate() {
            state.ready = false;
            return Err(error);
        }
    }
    result
}

fn complete<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    threaded: bool,
) -> Result<f32> {
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let (bytes, prefix, trace) = forward(state, store, source, options, &revisions)?;
    // The target is unavailable to every route computation above this boundary.
    let target = source.anchor_target(&state.cursor)?;
    let credit = connected_anchor::gather(&state.shared, store, &prefix, &trace, target.byte)?;
    // Replayed anchors run under the same weights; only their tile gradients survive.
    let mut replayed = Accumulator::default();
    super::connected_replay::run(state, store, source, &mut replayed)?;
    // The label feature under the E that produced these corrections (CONNECTED.md §v5).
    let observed_target = state.shared.target_feature(target.byte)?;
    let tally = Tally::of(&trace, &credit)?;
    super::connected_lane_apply::update_routes(state, store, &trace, credit.loss, options, rate)?;
    step_shared(state, &credit, rate, &options)?;
    observe(
        state,
        &trace,
        &credit,
        &observed_target,
        &revisions,
        options,
    )?;
    // Local lanes consume the newly updated E (fresh prefix inside `apply_tile_credit`) and step
    // tiles at the scheduled `local_rate`; `rate` above drove routes, E and the bias.
    apply_tile_credit(
        state,
        store,
        &bytes,
        &prefix,
        &trace,
        credit.gradients,
        &observed_target,
        target.byte,
        options,
        threaded,
        replayed,
    )?;
    if let Some(replay) = options.replay {
        let position = state.cursor.supervised_targets;
        state.replay.admit(position, credit.loss, replay.capacity);
    }
    advance(state, source, &tally)?;
    Ok(credit.loss)
}

/// One anchor's target-free forward pass under tile `revisions`: prefix bytes, prefix, trace.
pub(super) fn forward<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    revisions: &[u64; TILES],
) -> Result<(Vec<u8>, Prefix, Trace)> {
    let bytes = source.pending_prefix(&state.cursor)?;
    let prefix = state.shared.prefix(&bytes)?;
    let random = tuning::hash(state.seed, &[state.anchors, 0x414e4348]);
    let draw = super::connected_route::draw(&options, state, &[state.anchors], random);
    let trace = connected_anchor::forward(
        store,
        state.geometry,
        &prefix,
        options.visits,
        draw,
        |tile, input| super::connected_reach::anchor_read(state, input, tile, revisions, &options),
    )?;
    Ok((bytes, prefix, trace))
}

/// Per-anchor scalars recorded once the anchor advances.
pub(super) struct Tally {
    pub loss: f32,
    output_rms: f32,
    /// Mean and maximum raw visit RMS.
    visit: (f64, f64),
    knot_overflow: f64,
    route: super::connected_balance::Visit,
}

impl Tally {
    /// The anchor's scalars; the readout input is normalized, so each raw visit output is bounded.
    pub fn of(trace: &Trace, credit: &Credit) -> Result<Self> {
        let visit = trace.visit_rms();
        ensure!(visit.1 <= 1e3, "connected visit output exceeded its bound");
        Ok(Self {
            loss: credit.loss,
            output_rms: credit.output_rms,
            visit,
            knot_overflow: trace.knot_overflow(),
            route: super::connected_balance::Visit::of(trace),
        })
    }
}

/// Steps E and the readout bias on one anchor's credit; both Adam clocks track `anchors`.
pub(super) fn step_shared(
    state: &mut State,
    credit: &Credit,
    rate: f64,
    options: &Options,
) -> Result<()> {
    let embedding = connected_update::apply_embedding(
        &mut state.shared,
        &mut state.shared_latent,
        &credit.embedding,
        rate,
        options,
    )?;
    state.counters.embedding_transitions += embedding.transitions as u64;
    state.counters.change_sum += restart::relative(&embedding);
    if let Some(gradient) = &credit.bias_gradient {
        connected_update::apply_bias(
            &mut state.shared,
            &mut state.bias_latent,
            gradient,
            rate,
            options,
        )?;
    }
    Ok(())
}

/// Advances the cursor and anchor count and folds the anchor into the counters and baseline.
pub(super) fn advance<S: TrainingSource>(
    state: &mut State,
    source: &S,
    tally: &Tally,
) -> Result<()> {
    source.advance_anchor(&mut state.cursor)?;
    state.anchors += 1;
    let counters = &mut state.counters;
    counters.loss_sum += f64::from(tally.loss);
    counters.readout_rms_sum += f64::from(tally.output_rms);
    counters.visit_rms_sum += tally.visit.0;
    counters.visit_rms_max = counters.visit_rms_max.max(tally.visit.1);
    counters.knot_overflow_sum += tally.knot_overflow;
    super::connected_balance::observe(state, &tally.route, tally.loss)?;
    state.baseline = 0.9 * state.baseline + 0.1 * tally.loss;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_tile_credit(
    state: &mut State,
    store: &mut Store,
    bytes: &[u8],
    anchor_prefix: &Prefix,
    trace: &Trace,
    exact_gradients: Vec<TileGradients>,
    observed_target: &[f32],
    target: u8,
    options: Options,
    threaded: bool,
    replayed: Accumulator,
) -> Result<()> {
    match options.credit_mode {
        CreditMode::Lanes => {
            let local_prefix = state.shared.prefix(bytes)?;
            super::connected_lanes::run(state, store, &local_prefix, target, options, threaded)
        }
        CreditMode::AnchorPredicted => super::connected_lanes::run_anchor_predicted(
            state,
            store,
            anchor_prefix,
            trace,
            observed_target,
            options,
        ),
        CreditMode::AnchorExact | CreditMode::Hybrid => {
            let mut gradients = replayed;
            for (frame, gradient) in trace.frames.iter().zip(exact_gradients) {
                state.counters.local_visits[frame.tile] += 1;
                gradients.add_exact(frame.tile, gradient)?;
            }
            if options.credit_mode == CreditMode::Hybrid {
                let local_prefix = state.shared.prefix(bytes)?;
                super::connected_lanes::run_with_accumulator(
                    state,
                    store,
                    &local_prefix,
                    target,
                    options,
                    threaded,
                    gradients,
                    true,
                )
            } else {
                // Non-persisted guard: one mean step per touched tile per anchor, replay or not.
                let (before, tiles) = (state.counters.local_updates, gradients.tiles());
                gradients.apply_with_trust(state, store, &options, false)?;
                ensure!(
                    state.counters.local_updates - before <= tiles,
                    "connected anchor stepped a tile more than once"
                );
                store.quiescent()
            }
        }
    }
}

/// Fits every frame's observed correction, conditioned on the anchor's label feature `target`.
pub(super) fn observe(
    state: &mut State,
    trace: &Trace,
    credit: &Credit,
    target: &[f32],
    revisions: &[u64; TILES],
    options: Options,
) -> Result<()> {
    for (frame, correction) in trace.frames.iter().zip(&credit.corrections) {
        let record = Record {
            input: frame.input.clone(),
            output: frame.output.clone(),
            history: frame.history.clone(),
            correction: correction.clone(),
            tile_id: frame.tile,
            revision: revisions[frame.tile],
            shared_revision: state.anchors,
            epoch: state.epoch,
            kind: Kind::Observed,
            lane: None,
        };
        let metrics = state
            .predictor
            .fit(&record, target, options.predictor_rate)?;
        state.gains[frame.tile].observe(&metrics, state.epoch, revisions[frame.tile], &options)?;
        restart::observe_visit(state, frame.tile, &frame.input, &frame.history)?;
        state.memory.publish(record)?;
        state.epoch += 1;
        state.counters.predictor_fits += 1;
        state.counters.predictor_mse_sum += metrics.mse;
        state.counters.predictor_zero_mse_sum += metrics.zero_mse;
        state.counters.predictor_cosine_sum += metrics.cosine;
        state.counters.observed_records += 1;
        state.counters.anchor_visits[frame.tile] += 1;
    }
    Ok(())
}
