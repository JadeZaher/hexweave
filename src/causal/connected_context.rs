//! EXPERIMENT contextual prefix rows: older rows add their position's detached routed state; see
//! CONNECTED.md §Contextual rows.

use super::connected_anchor::{self, Trace};
use super::connected_config::{CreditMode, Options};
use super::connected_state::State;
use super::connected_store::Store;
use super::execution::TrainingSource;
use super::geometry::Geometry;
use super::shared::{Prefix, Shared};
use super::tuning;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::ops::Range;

/// Serde skip: the default (off) stays out of options identity and serialization.
pub(super) fn is_off(value: &bool) -> bool {
    !*value
}

/// Admission: `anchor_exact`, `anchor_batch` 1, and no reader of earlier anchors' rows.
pub(super) fn validate(options: &Options) -> Result<()> {
    super::connected_context_gate::validate_options(options)?;
    if !options.context_rows {
        return Ok(());
    }
    ensure!(
        options.credit_mode == CreditMode::AnchorExact && options.anchor_batch == 1,
        "connected context_rows requires credit_mode anchor_exact and anchor_batch 1"
    );
    ensure!(
        options.shared.is_none(),
        "connected context_rows refuses connected.shared (keeps the first pilot attributable)"
    );
    ensure!(
        options.replay.is_none() && options.restart.is_none_or(|restart| restart.replay == 0),
        "connected context_rows refuses connected.replay and restart.replay (replayed anchors \
         would read rows without their states)"
    );
    Ok(())
}

/// Row counts of live training prefixes: older rows, rows with a state, summed state/raw RMS.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Tally {
    pub rows: u64,
    pub filled: u64,
    pub ratio_sum: f64,
}

impl Tally {
    /// `filled_share` and mean `state_to_row_rms` since `previous`.
    pub fn report(&self, previous: &Self) -> Value {
        let rows = self.rows - previous.rows;
        let filled = self.filled - previous.filled;
        let ratio = self.ratio_sum - previous.ratio_sum;
        json!({"rows": rows, "filled_share": (rows > 0).then(|| filled as f64 / rows as f64),
            "state_to_row_rms": (filled > 0).then(|| ratio / filled as f64)})
    }

    fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.filled += other.filled;
        self.ratio_sum += other.ratio_sum;
    }
}

/// Session-only routed output states of the newest consecutive positions, keyed by position.
#[derive(Clone, Debug)]
pub(super) struct Context {
    capacity: usize,
    newest: Option<u64>,
    states: VecDeque<Vec<f32>>,
    /// Live training rows only; warm-up fills are not counted.
    pub tally: Tally,
}

impl Context {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            newest: None,
            states: VecDeque::with_capacity(capacity),
            tally: Tally::default(),
        }
    }

    /// Forgets every state (a stream restart).
    pub fn clear(&mut self) {
        self.newest = None;
        self.states.clear();
    }

    /// Appends position `key`'s state; a key that does not follow the newest restarts the buffer.
    pub fn push(&mut self, key: u64, state: Vec<f32>) {
        if self.newest.and_then(|newest| newest.checked_add(1)) != Some(key) {
            self.states.clear();
        }
        self.states.push_back(state);
        if self.states.len() > self.capacity {
            self.states.pop_front();
        }
        self.newest = Some(key);
    }

    /// The buffered state of position `key`, if held.
    pub fn state(&self, key: u64) -> Option<&[f32]> {
        let back = usize::try_from(self.newest?.checked_sub(key)?).ok()?;
        let index = self.states.len().checked_sub(back + 1)?;
        self.states.get(index).map(Vec::as_slice)
    }
}

/// A fresh empty buffer exactly when `options.context_rows`.
pub(super) fn fresh(geometry: Geometry, options: &Options) -> Option<Context> {
    options.context_rows.then(|| Context::new(geometry.context))
}

/// Rows of `bytes` whose newest byte is position `key`: row i is position `key - (L-1-i)` and adds
/// that position's buffered state (scaled by `gate` when present); the newest row and rows before
/// `raw_before` stay raw.
pub(super) fn rows(
    shared: &Shared,
    bytes: &[u8],
    context: &Context,
    key: u64,
    raw_before: usize,
    gate: Option<f32>,
) -> Result<(Prefix, Tally)> {
    ensure!(!bytes.is_empty(), "invalid visible prefix length");
    let (newest, width) = (bytes.len() - 1, shared.geometry.width);
    let mut tally = Tally {
        rows: newest as u64,
        ..Tally::default()
    };
    let strength = super::connected_context_gate::strength(gate);
    let mut gated = gate.map(|_| vec![0.0f32; bytes.len() * width]);
    let mut prefix = shared.prefix_with(bytes, gate, |index, raw| {
        if index >= newest || index < raw_before {
            return None;
        }
        let state = context.state(key.checked_sub((newest - index) as u64)?)?;
        tally.filled += 1;
        let raw_rms = super::connected_update_scale::rms(raw);
        tally.ratio_sum +=
            strength * super::connected_update_scale::rms(state) / raw_rms.max(1e-12);
        if let Some(copy) = gated.as_mut() {
            copy[index * width..(index + 1) * width].copy_from_slice(state);
        }
        Some(state)
    })?;
    prefix.gated_states = gated;
    Ok((prefix, tally))
}

/// The live anchor's prefix, keyed by its anchor index and counted in the buffer's tally.
pub(super) fn anchor_prefix(state: &mut State, bytes: &[u8]) -> Result<Prefix> {
    let Some(context) = state.context.as_mut() else {
        return state.shared.prefix(bytes);
    };
    let gate = super::connected_context_gate::value(&state.gate);
    let (prefix, tally) = rows(&state.shared, bytes, context, state.anchors, 0, gate)?;
    context.tally.add(&tally);
    Ok(prefix)
}

/// Pushes the live anchor's routed output; the stream's START anchor first empties the buffer.
pub(super) fn push_anchor(state: &mut State, trace: &Trace) {
    if let Some(context) = state.context.as_mut() {
        if state.cursor.completion_offset == 0 {
            context.clear();
        }
        context.push(state.anchors, trace.output.clone());
    }
}

/// First warm-up position for a prefix of `length` bytes whose newest byte is position `key`.
pub(super) fn warm_start(key: u64, length: usize) -> u64 {
    key.saturating_sub(length.saturating_sub(1) as u64)
}

/// Forward-only passes over `keys` in order, pushing each position's routed output (the resume
/// refill; frozen replays warm up by observing instead). Nothing is observed, recorded or counted.
pub(super) fn fill(
    context: &mut Context,
    shared: &Shared,
    gate: Option<f32>,
    keys: Range<u64>,
    mut bytes: impl FnMut(u64) -> Result<Vec<u8>>,
    mut forward: impl FnMut(u64, &Prefix) -> Result<Trace>,
) -> Result<()> {
    for key in keys {
        let (prefix, _) = rows(shared, &bytes(key)?, context, key, 0, gate)?;
        let trace = forward(key, &prefix)?;
        context.push(key, trace.output);
    }
    Ok(())
}

/// Resume: refill the positions the next anchor's rows read under the restored weights, each with
/// its anchor's own training draw and memory cut off before its label (not bit-exact; CONNECTED.md).
pub(super) fn refill<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
) -> Result<()> {
    let Some(mut context) = state.context.take() else {
        return Ok(());
    };
    tuning::set_connected(&state.options);
    let frozen = &*state;
    let length = source.pending_prefix(&frozen.cursor)?.len();
    let keys = warm_start(frozen.anchors, length)..frozen.anchors;
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    let forward = |key: u64, prefix: &Prefix| {
        let draw = super::connected_route::anchor(&frozen.options, frozen, key);
        let options = &frozen.options;
        connected_anchor::forward(
            store,
            frozen.geometry,
            prefix,
            options.visits,
            draw,
            |tile, x| {
                let retrieved =
                    (frozen.memory).query_before(x, tile, frozen.epoch, &revisions, key)?;
                Ok(options.history_input(retrieved))
            },
        )
    };
    let bytes = |key| source.pending_prefix(&source.replay_cursor(key)?);
    let gate = super::connected_context_gate::value(&frozen.gate);
    let result = fill(&mut context, &frozen.shared, gate, keys, bytes, forward);
    state.context = Some(context);
    result
}

/// The training buffer's tally (zero without contextual rows).
pub(super) fn tally(state: &State) -> Tally {
    state
        .context
        .as_ref()
        .map(|context| context.tally)
        .unwrap_or_default()
}

/// The log row's `context` object since `previous`; None without contextual rows.
pub(super) fn report(state: &State, previous: &Tally) -> Option<Value> {
    let context = state.context.as_ref()?;
    let mut line = context.tally.report(previous);
    if let Some(gate) = super::connected_context_gate::value(&state.gate) {
        line["gate"] = json!(gate);
    }
    Some(line)
}
