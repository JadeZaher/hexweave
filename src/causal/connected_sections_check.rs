//! Shared-section validation and reporting; see CONNECTED.md §Shared sections.

use super::super::connected_state::State;
use super::super::connected_update_scale::rms;
use super::super::geometry::{MATRIX_COUNT, START_TILE, TILES};
use super::{Section, Slot};
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};

/// Presence, shapes, wave accounting, queue finiteness and the master RMS cap.
pub(in super::super) fn validate(state: &State) -> Result<()> {
    let (shared, sections) = match (state.options.shared, &state.sections) {
        (None, None) => return Ok(()),
        (Some(shared), Some(sections)) => (shared, sections),
        _ => bail!("connected shared sections present exactly when configured"),
    };
    let (queue, count) = (&sections.queue, shared.sections as usize);
    ensure!(
        sections.sections.len() == count
            && queue.slots.len() == count
            && queue.tile_weight_sum.len() == TILES
            && queue.tile_weight_visits.len() == TILES
            && sections
                .sections
                .iter()
                .all(|section| section.tile.id == START_TILE
                    && section.tile.weights.is_some() != state.options.precision.is_ternary()),
        "connected shared section shape"
    );
    let done = queue.waves.saturating_mul(u64::from(shared.wave));
    ensure!(
        queue.slots.iter().map(|slot| slot.writes).sum::<u64>() == queue.waves
            && done.saturating_add(sections.pending.len() as u64) == state.anchors
            && queue.reads == state.anchors
            && queue.lambert_reads <= queue.reads,
        "connected shared wave accounting"
    );
    let cap = state.options.master_rms_cap * (1.0 + 1e-5);
    for slot in &queue.slots {
        ensure!(
            (shared.queue || slot.priority == 0.0)
                && slot_valid(slot, state.geometry.width, queue.waves, cap),
            "connected shared queue slot"
        );
    }
    ensure!(
        queue.read_delta_sum.is_finite()
            && queue.alignment_sum.abs() <= queue.reads as f64 + 1e-9
            && queue
                .centre
                .as_ref()
                .is_none_or(|centre| centre.len() == state.geometry.width
                    && centre.iter().all(|value| value.is_finite()))
            && queue.centre.is_some() == (queue.waves > 0)
            && queue
                .tile_weight_sum
                .iter()
                .zip(&queue.tile_weight_visits)
                .all(|(sum, &visits)| sum.is_finite()
                    && *sum >= 0.0
                    && *sum <= visits as f64 + 1e-9),
        "connected shared monitors"
    );
    Ok(())
}

/// Finite queue values, a width-`width` key, write bookkeeping that agrees and the RMS cap (a NaN
/// master RMS fails it).
fn slot_valid(slot: &Slot, width: usize, waves: u64, cap: f64) -> bool {
    let written = slot.writes > 0;
    let key = slot.key.as_ref();
    slot.priority.is_finite()
        && key.is_none_or(|key| key.len() == width && key.iter().all(|value| value.is_finite()))
        && written == slot.last_write_wave.is_some()
        && written == slot.update_ema.is_some()
        && slot.last_write_wave.is_none_or(|wave| wave < waves)
        && slot
            .update_ema
            .is_none_or(|ema| ema.is_finite() && ema >= 0.0)
        && slot.update_rms.is_finite()
        && (0.0..=1.0).contains(&slot.last_match)
        && slot.master_rms_max <= cap
}

/// The `shared` block of log rows and the report: queue state and lifetime monitors.
pub(in super::super) fn report(state: &State) -> Result<Option<Value>> {
    let (Some(shared), Some(sections)) = (state.options.shared, &state.sections) else {
        return Ok(None);
    };
    let queue = &sections.queue;
    let slots = &queue.slots;
    let mean_tile_weight: Vec<_> = (queue.tile_weight_sum.iter().zip(&queue.tile_weight_visits))
        .map(|(sum, &visits)| (visits > 0).then(|| sum / visits as f64))
        .collect();
    let weight_rms: Vec<_> = sections.sections.iter().map(master_rms).collect();
    let resident = sections.sections.iter().map(bytes).sum::<usize>();
    let mean = |sum: f64| (queue.reads > 0).then(|| sum / queue.reads as f64);
    Ok(Some(json!({
        "sections": shared.sections, "wave": shared.wave, "queue": shared.queue,
        "teach": shared.teach, "credit": shared.credit, "waves": queue.waves,
        "writes": slots.iter().map(|slot| slot.writes).collect::<Vec<_>>(),
        "priority": slots.iter().map(|slot| slot.priority).collect::<Vec<_>>(),
        "last_write_wave": slots.iter().map(|slot| slot.last_write_wave).collect::<Vec<_>>(),
        "update_rms": slots.iter().map(|slot| slot.update_rms).collect::<Vec<_>>(),
        "match": slots.iter().map(|slot| slot.last_match).collect::<Vec<_>>(),
        "weight_rms": weight_rms, "mean_tile_weight": mean_tile_weight,
        "read_delta": mean(queue.read_delta_sum),
        "credit_alignment": mean(queue.alignment_sum),
        "lambert_share": (queue.reads > 0).then(|| queue.lambert_reads as f64 / queue.reads as f64),
        "resident_bytes": resident, "pending_anchors": sections.pending.len(),
    })))
}

/// RMS over a section's tile-matrix masters; None before a ternary section's first write.
fn master_rms(section: &Section) -> Option<f64> {
    let masters = section
        .latents
        .iter()
        .filter(|((_, slot), _)| *slot < MATRIX_COUNT);
    let (square, count) = masters.fold((0.0, 0usize), |(square, count), (_, latent)| {
        let rms = rms(&latent.weights);
        (
            square + rms * rms * latent.weights.len() as f64,
            count + latent.weights.len(),
        )
    });
    (count > 0).then(|| (square / count as f64).sqrt())
}

/// Decoded bytes a section pins: packed trits, Adam masters and moments, and its FP32 view.
fn bytes(section: &Section) -> usize {
    let packed: usize = section
        .tile
        .packed()
        .iter()
        .map(|matrix| matrix.len())
        .sum();
    let latents = section.latents.iter();
    let adam: usize = latents
        .map(|(_, l)| 4 * (l.weights.len() + l.moments.len()))
        .sum();
    packed
        + adam
        + section
            .tile
            .weights
            .as_ref()
            .map_or(0, |weights| weights.bytes())
}
