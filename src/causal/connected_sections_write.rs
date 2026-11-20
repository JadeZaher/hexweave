//! The shared wave: its anchors, queue selection and the one section write at its end; see
//! CONNECTED.md §Shared sections.

use super::connected_config::Options;
use super::connected_sections::{Section, Slot};
use super::connected_state::State;
use super::connected_store::Store;
use super::connected_store_codec as codec;
use super::connected_update::{self, Mode};
use super::execution::TrainingSource;
use super::geometry::START_TILE;
use super::tile_gradients::TileGradients;
use anyhow::{ensure, Context, Result};

/// EMA horizon, in writes, of a section's key and write size (running mean before it).
const HORIZON: f64 = 16.0;

/// Match floor of the queue: every section gains at least this much priority per wave.
const MATCH_FLOOR: f64 = 0.05;

/// Relative tolerance under which two priorities (or matches) tie.
const TIE: f64 = 1e-9;

/// Training units (anchors or anchor waves) up to the aligned shared-wave end, then its write;
/// returns the mean anchor loss.
pub(super) fn step<S: TrainingSource>(
    state: &mut State,
    store: &mut Store,
    source: &S,
    options: Options,
    rate: f64,
    threaded: bool,
) -> Result<f32> {
    let shared = options.shared.context("connected shared options missing")?;
    let (start, mut loss) = (state.anchors, 0.0f64);
    loop {
        let before = state.anchors;
        let mean = super::connected_wave::unit(state, store, source, options, rate, threaded)?;
        loss += f64::from(mean) * (state.anchors - before) as f64;
        if state.anchors.is_multiple_of(u64::from(shared.wave)) {
            break;
        }
    }
    finish(state, &options)?;
    Ok((loss / (state.anchors - start).max(1) as f64) as f32)
}

/// The wave-end write between validated boundaries; a failure leaves the state poisoned.
pub(super) fn finish(state: &mut State, options: &Options) -> Result<()> {
    state.validate()?;
    state.ready = false;
    write(state, options)?;
    state.ready = true;
    if let Err(error) = state.validate() {
        state.ready = false;
        return Err(error);
    }
    Ok(())
}

/// Selects the section, takes one local step per anchor of the wave and settles the queue.
fn write(state: &mut State, options: &Options) -> Result<()> {
    let shared = options.shared.context("connected shared options missing")?;
    // Tile steps use the schedule at their anchor's index; the wave's last anchor here.
    let rate = options.local_rate * options.local_decay.factor(state.anchors.saturating_sub(1));
    let width = state.geometry.width;
    let routes = state.geometry.routes(START_TILE)?;
    let local = state.geometry.tile_coefficients(START_TILE)? - 3 * routes * width;
    let sections = state
        .sections
        .as_mut()
        .context("connected shared sections missing")?;
    let items = std::mem::take(&mut sections.pending);
    ensure!(
        items.len() as u64 == u64::from(shared.wave),
        "connected shared wave credit count"
    );
    let mut mean = vec![0.0f64; width];
    for item in &items {
        for (total, value) in mean.iter_mut().zip(&item.base) {
            *total += f64::from(*value) / items.len() as f64;
        }
    }
    let queue = &mut sections.queue;
    // Centred on the earlier waves' mean: the common component would push every match to ~1.
    let centre = queue.centre.as_deref();
    let matches: Vec<f64> = queue
        .slots
        .iter()
        .map(|slot| matching(&mean, slot, centre))
        .collect();
    let last: Vec<_> = queue
        .slots
        .iter()
        .map(|slot| slot.last_write_wave)
        .collect();
    let mut priorities: Vec<f64> = queue.slots.iter().map(|slot| slot.priority).collect();
    let winner = select(&mut priorities, &last, &matches, shared.queue);
    let (size, max_rms) = update(&mut sections.sections[winner], &items, rate, options)?;
    if shared.queue {
        let ratio = ratio(size, queue.slots[winner].update_ema);
        settle(&mut priorities[winner], &matches, ratio);
        // Centring keeps reported priorities bounded; it shifts all equally, so no choice moves.
        let offset = priorities.iter().sum::<f64>() / priorities.len() as f64;
        for (slot, priority) in queue.slots.iter_mut().zip(priorities) {
            slot.priority = priority - offset;
        }
    }
    for (slot, matched) in queue.slots.iter_mut().zip(&matches) {
        slot.last_match = *matched;
    }
    let weight = (1.0 / (queue.waves + 1) as f64).max(1.0 / HORIZON);
    let mut centre = queue.centre.take().unwrap_or_else(|| mean.clone());
    for (value, target) in centre.iter_mut().zip(&mean) {
        *value += weight * (target - *value);
    }
    queue.centre = Some(centre);
    let slot = &mut queue.slots[winner];
    slot.writes += 1;
    slot.last_write_wave = Some(queue.waves);
    let alpha = (1.0 / slot.writes as f64).max(1.0 / HORIZON);
    slot.update_ema = Some(
        slot.update_ema
            .map_or(size, |ema| ema + alpha * (size - ema)),
    );
    let mut key = slot.key.take().unwrap_or_else(|| mean.clone());
    for (value, target) in key.iter_mut().zip(&mean) {
        *value += alpha * (target - *value);
    }
    slot.key = Some(key);
    slot.update_rms = size / (local as f64).sqrt();
    if max_rms.is_nan() || max_rms > slot.master_rms_max {
        slot.master_rms_max = max_rms;
    }
    queue.waves += 1;
    Ok(())
}

/// `clamp((cos(mean - c, key - c) + 1) / 2, 0.05, 1)` with c the centre (0 before any wave); 1
/// for a section with no key yet, 0.5 when either centred vector is zero.
fn matching(mean: &[f64], slot: &Slot, centre: Option<&[f64]>) -> f64 {
    let Some(key) = &slot.key else {
        return 1.0;
    };
    let centred = |values: &[f64]| -> Vec<f64> {
        let shift = |index: usize| centre.map_or(0.0, |centre| centre[index]);
        values
            .iter()
            .enumerate()
            .map(|(index, value)| value - shift(index))
            .collect()
    };
    let (mean, key) = (centred(mean), centred(key));
    let dot: f64 = mean.iter().zip(&key).map(|(a, b)| a * b).sum();
    let norms = mean.iter().map(|a| a * a).sum::<f64>().sqrt()
        * key.iter().map(|b| b * b).sum::<f64>().sqrt();
    let cosine = if norms > 0.0 { dot / norms } else { 0.0 };
    ((cosine + 1.0) / 2.0).clamp(MATCH_FLOOR, 1.0)
}

/// The winning section: with `queue` every priority first gains its match and the highest
/// priority wins, else the highest match; ties go to the section written longest ago (never
/// written first), then the lowest index.
pub(super) fn select(
    priorities: &mut [f64],
    last: &[Option<u64>],
    matches: &[f64],
    queue: bool,
) -> usize {
    if queue {
        for (priority, matched) in priorities.iter_mut().zip(matches) {
            *priority += matched;
        }
    }
    let scores = if queue { &*priorities } else { matches };
    let mut best = 0;
    for (index, &score) in scores.iter().enumerate().skip(1) {
        let leader = scores[best];
        let tied = (score - leader).abs() <= TIE * score.abs().max(leader.abs()).max(1.0);
        // `None < Some(_)`: a never-written section is the one written longest ago.
        if (!tied && score > leader) || (tied && last[index] < last[best]) {
            best = index;
        }
    }
    best
}

/// The winner pays the wave's total match, scaled by its write-size ratio.
pub(super) fn settle(priority: &mut f64, matches: &[f64], ratio: f64) {
    *priority -= matches.iter().sum::<f64>() * ratio;
}

/// `clamp(size / EMA(size), 0.5, 2)`; 1 on a first write (or a zero EMA).
pub(super) fn ratio(size: f64, ema: Option<f64>) -> f64 {
    ema.filter(|&ema| ema > 0.0)
        .map_or(1.0, |ema| (size / ema).clamp(0.5, 2.0))
}

/// One unit-gain local step per item, each gradient taken at the wave-start weights; returns the
/// L2 norm of the applied steps and the largest master RMS after them.
pub(super) fn update(
    section: &mut Section,
    items: &[super::connected_sections_read::Item],
    rate: f64,
    options: &Options,
) -> Result<(f64, f64)> {
    let frozen = section.tile.clone();
    let history = vec![0.0; frozen.geometry.width];
    let (mut squares, mut max_rms) = (0.0, 0.0f64);
    for item in items {
        let mut gradient = TileGradients::zeros(frozen.geometry, frozen.id)?;
        let (base, memory, sink) = (&item.base, &item.memory, Some(&mut gradient));
        frozen.backward(base, memory, &history, &item.upstream, None, sink, None)?;
        let (tile, latents) = (&mut section.tile, &mut section.latents);
        let metrics =
            connected_update::apply(tile, latents, &gradient, rate, 1.0, options, Mode::Local)?;
        squares += metrics.applied_norm.powi(2);
        if metrics.max_rms.is_nan() || metrics.max_rms > max_rms {
            max_rms = metrics.max_rms;
        }
    }
    codec::validate(&section.tile, &section.latents)?;
    if !options.precision.is_ternary() {
        super::connected_precision::validate_masters(&section.tile, &section.latents)?;
    }
    Ok((squares.sqrt(), max_rms))
}
