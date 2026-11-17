//! Byte continuation (greedy or nucleus, `decoding`) with local correction replay; see STUDY.md.

use super::decoding::Decoder;
use super::execution::{forward_path, Trace};
use super::immutable;
use super::replay::{Replay, SPAN_WARMUP};
use super::state::{count, State};
use super::store_api::TileStore;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Terminal status a readout poll can report to interrupt evaluation or generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Terminal {
    Completed,
    UserStop,
    ResourceStop,
    BudgetStop,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Routes {
    pub visits: [u64; 7],
    pub route_calls: u64,
    pub halts: u64,
    pub forced_exits: u64,
}

impl Routes {
    pub fn include(&mut self, trace: &Trace, state: &State) -> Result<()> {
        for frame in &trace.frames {
            count(&mut self.visits[frame.tile_id], 1)?;
            if let Some(action) = frame.action {
                count(&mut self.route_calls, 1)?;
                if action + 1 == state.geometry.routes(frame.tile_id)? {
                    count(&mut self.halts, 1)?;
                }
            } else {
                count(&mut self.forced_exits, 1)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "status")]
pub(super) enum Reason {
    Newline,
    EndOfStory,
    ByteBudget,
    Interrupted(Terminal),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Generation {
    pub raw_bytes: Vec<u8>,
    pub valid_utf8: bool,
    pub stop_reason: Reason,
    pub routes: Routes,
    pub wall_micros: u64,
    pub before_sha256: String,
    pub after_sha256: String,
}

pub(super) fn lowest_maximum(values: &[f32]) -> Result<usize> {
    ensure!(!values.is_empty(), "empty greedy readout");
    super::norm::finite(values)?;
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best)
}

/// Clip bytes before embedding; positions restart from zero within this visible slice.
pub(super) fn recent_window(bytes: &[u8], window: usize) -> &[u8] {
    &bytes[bytes.len().saturating_sub(window)..]
}

pub(super) fn generate<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    prompt: &[u8],
    maximum_new_bytes: usize,
    poll: &mut F,
) -> Result<Generation> {
    generate_inner(
        state,
        store,
        prompt,
        maximum_new_bytes,
        poll,
        None,
        state.geometry.context,
        Decoder::Greedy,
    )
}

/// Generate a corpus continuation until its end-of-story marker or byte budget.
pub(super) fn generate_story<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    prompt: &[u8],
    maximum_new_bytes: usize,
    window: usize,
    decoder: Decoder,
    poll: &mut F,
) -> Result<Generation> {
    generate_inner(
        state,
        store,
        prompt,
        maximum_new_bytes,
        poll,
        Some(super::core_corpus::END),
        window,
        decoder,
    )
}

/// Warm local history from up to `SPAN_WARMUP` prompt bytes; this does not update weights.
fn warm_history<S: TileStore>(
    state: &State,
    store: &mut S,
    prompt: &[u8],
    window: usize,
    replay: &mut Replay,
) -> Result<()> {
    for t in prompt.len().saturating_sub(SPAN_WARMUP).max(1)..prompt.len() {
        let prefix = state.shared.prefix(recent_window(&prompt[..t], window))?;
        let history = replay.input(state)?;
        let trace = forward_path(
            store,
            state.geometry,
            &prefix,
            history,
            &mut None,
            state.tile_versions,
        )?;
        replay.observe(state, store, &prefix, &trace, prompt[t])?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_inner<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    prompt: &[u8],
    maximum_new_bytes: usize,
    poll: &mut F,
    end: Option<u8>,
    window: usize,
    decoder: Decoder,
) -> Result<Generation> {
    decoder.validate()?;
    ensure!(
        !prompt.is_empty() && std::str::from_utf8(prompt).is_ok(),
        "generation prompt must be nonempty UTF8"
    );
    ensure!(
        (1..=state.geometry.context).contains(&window),
        "invalid generation window"
    );
    let bounded = if end.is_some() {
        prompt.len() <= state.geometry.context && maximum_new_bytes <= state.geometry.context
    } else {
        prompt
            .len()
            .checked_add(maximum_new_bytes)
            .is_some_and(|n| n <= state.geometry.context)
    };
    ensure!(
        maximum_new_bytes > 0 && bounded,
        "generation budget exceeds context"
    );
    let started = Instant::now();
    let before_sha256 = immutable::identity(state, store)?;
    let mut visible = Vec::with_capacity(prompt.len() + maximum_new_bytes);
    visible.extend_from_slice(prompt);
    let mut raw_bytes = Vec::with_capacity(maximum_new_bytes);
    let mut routes = Routes::default();
    let mut stop_reason = Reason::ByteBudget;
    let mut replay = Replay::default();
    warm_history(state, store, prompt, window, &mut replay)?;
    for _ in 0..maximum_new_bytes {
        if let Some(status) = poll()? {
            stop_reason = Reason::Interrupted(status);
            break;
        }
        // Rebuild the visible byte embeddings for each next byte; there is no prefix/KV cache.
        let prefix = state.shared.prefix(recent_window(&visible, window))?;
        let history = replay.input(state)?;
        let trace = forward_path(
            store,
            state.geometry,
            &prefix,
            history,
            &mut None,
            state.tile_versions,
        )?;
        routes.include(&trace, state)?;
        let readout = state.shared.readout(&trace.output)?;
        let byte = u8::try_from(decoder.choose(&readout.logits, raw_bytes.len())?)?;
        if end == Some(byte) {
            stop_reason = Reason::EndOfStory;
            break;
        }
        // Generated history treats our own chosen byte as the target, not an external label.
        replay.observe(state, store, &prefix, &trace, byte)?;
        raw_bytes.push(byte);
        visible.push(byte);
        if end.is_none() && byte == b'\n' {
            stop_reason = Reason::Newline;
            break;
        }
    }
    let after_sha256 = immutable::identity(state, store)?;
    ensure!(
        before_sha256 == after_sha256,
        "generation mutated model or continuation state"
    );
    Ok(Generation {
        valid_utf8: std::str::from_utf8(&raw_bytes).is_ok(),
        raw_bytes,
        stop_reason,
        routes,
        wall_micros: u64::try_from(started.elapsed().as_micros())?,
        before_sha256,
        after_sha256,
    })
}
