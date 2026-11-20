//! EXPERIMENT: always-resident shared sections, read at every route entry and written once per
//! shared wave; see CONNECTED.md §Shared sections.

use super::connected_config::Options;
use super::connected_precision::{self as precision, Precision, View};
use super::connected_state::State;
use super::connected_store::TileMeta;
use super::connected_store_codec as codec;
use super::geometry::{Geometry, START_TILE, TILES};
use super::tile::Tile;
use super::tuning::Latents;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[path = "connected_sections_config.rs"]
mod config;
pub(super) use config::{validate_options, SharedCredit, SharedOptions, MAX_SECTIONS};

/// One section: a tile-shaped bundle (centre-tile shape) and its Adam state, never routed to.
pub(super) struct Section {
    pub tile: Tile,
    pub latents: Latents,
}

/// The pinned sections, their queue (in `connected.json`) and the wave in progress.
pub(super) struct Sections {
    pub sections: Vec<Section>,
    pub queue: Queue,
    /// Session-only write credit of the shared wave in progress; empty at every wave end.
    pub pending: Vec<super::connected_sections_read::Item>,
}

/// Queue state and lifetime monitors; training-side only, never in a bundle.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Queue {
    pub waves: u64,
    pub slots: Vec<Slot>,
    /// Per ordinary tile: summed write weight w_i over its visits, and those visits.
    pub tile_weight_sum: Vec<f64>,
    pub tile_weight_visits: Vec<u64>,
    /// Summed RMS(read - x) / RMS(x) over `reads` anchors (x the route-entry state).
    pub read_delta_sum: f64,
    /// Summed cos(Lambert mix, exact read adjoint) over `reads` anchors.
    pub alignment_sum: f64,
    /// EMA over waves of the wave-mean entry state: the centre keys and means are compared from.
    pub centre: Option<Vec<f64>>,
    pub reads: u64,
    /// Anchors whose write weights came from the Lambert law rather than equal weights.
    pub lambert_reads: u64,
}

/// One section's queue entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Slot {
    pub priority: f64,
    /// EMA of the mean route-entry state of the waves that wrote this section.
    pub key: Option<Vec<f64>>,
    pub writes: u64,
    pub last_write_wave: Option<u64>,
    /// EMA of the write size (L2 norm of the wave's applied steps); None before the first write.
    pub update_ema: Option<f64>,
    /// RMS of the last write: its size over the root of the locally updated coefficient count.
    pub update_rms: f64,
    pub master_rms_max: f64,
    /// This section's match at the last wave end; 0 before the first.
    pub last_match: f64,
}

/// The `connected.json` record: queue plus each section bundle's hash, version and steps.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Saved {
    queue: Queue,
    bundles: Vec<TileMeta>,
}

/// A named checkpoint payload.
type Payload = (String, Vec<u8>);

/// Checkpoint payload name of section `index`.
pub(super) fn payload(index: usize) -> String {
    format!("section-{index}.bin")
}

/// Section `index`'s seed: its matrices take the seed block a tile with id `TILES + index` would.
fn seed(seed: u64, index: usize) -> u64 {
    seed.wrapping_add(16 * (TILES + index - START_TILE) as u64)
}

/// Fresh sections when `options.shared` is configured, initialized like tiles.
pub(super) fn fresh(geometry: Geometry, base: u64, options: &Options) -> Result<Option<Sections>> {
    let Some(shared) = options.shared else {
        return Ok(None);
    };
    let count = shared.sections as usize;
    let mut sections = Vec::with_capacity(count);
    for index in 0..count {
        let mut tile = Tile::new(geometry, START_TILE, seed(base, index))?;
        let latents = if options.precision.is_ternary() {
            Vec::new()
        } else {
            let values = precision::fresh(&tile)?;
            precision::attach_tile(&mut tile, &values, View::Fp32)?;
            values
        };
        sections.push(Section { tile, latents });
    }
    let queue = Queue {
        slots: vec![Slot::default(); count],
        tile_weight_sum: vec![0.0; TILES],
        tile_weight_visits: vec![0; TILES],
        ..Queue::default()
    };
    Ok(Some(Sections {
        sections,
        queue,
        pending: Vec::new(),
    }))
}

/// Encoded section payloads and their record; refused inside a shared wave.
pub(super) fn save(state: &State) -> Result<(Vec<Payload>, Option<Saved>)> {
    let (Some(sections), Some(shared)) = (&state.sections, state.options.shared) else {
        return Ok((Vec::new(), None));
    };
    ensure!(
        sections.pending.is_empty() && state.anchors.is_multiple_of(u64::from(shared.wave)),
        "connected shared sections cannot be saved inside a shared wave"
    );
    let directions = vec![0; state.geometry.tile_coefficients(START_TILE)?];
    let (mut files, mut bundles) = (Vec::new(), Vec::new());
    for (index, section) in sections.sections.iter().enumerate() {
        let (hash, bytes) = codec::encode_bytes(&section.tile, &section.latents, &directions)?;
        let steps = section.latents.iter().map(|(_, latent)| latent.step).max();
        bundles.push(TileMeta {
            hash,
            version: section.tile.version,
            steps: steps.unwrap_or(0),
            revision: 0,
        });
        files.push((payload(index), bytes));
    }
    let queue = sections.queue.clone();
    Ok((files, Some(Saved { queue, bundles })))
}

/// Decodes and hash-checks the sections `saved` describes.
pub(super) fn load(
    payloads: &BTreeMap<String, Vec<u8>>,
    saved: Option<Saved>,
    geometry: Geometry,
    options: &Options,
) -> Result<Option<Sections>> {
    let (shared, saved) = match (options.shared, saved) {
        (None, None) => return Ok(None),
        (Some(shared), Some(saved)) => (shared, saved),
        _ => bail!("connected shared sections and options disagree"),
    };
    ensure!(
        saved.bundles.len() == shared.sections as usize,
        "connected shared section count"
    );
    let mut sections = Vec::with_capacity(saved.bundles.len());
    for (index, meta) in saved.bundles.iter().enumerate() {
        let bytes = payloads
            .get(&payload(index))
            .context("missing connected shared section")?;
        let bundle = super::connected_store_read::load_bytes(bytes, geometry, START_TILE, meta)?;
        ensure!(
            meta.revision == 0 && bundle.last_direction.iter().all(|&value| value == 0),
            "connected shared section record"
        );
        let (mut tile, latents) = (bundle.tile, bundle.latents);
        if !options.precision.is_ternary() {
            precision::attach_tile(&mut tile, &latents, View::Fp32)?;
        }
        sections.push(Section { tile, latents });
    }
    let (queue, pending) = (saved.queue, Vec::new());
    Ok(Some(Sections {
        sections,
        queue,
        pending,
    }))
}

/// Re-attaches every FP32 section view in `mode` (quantized views for comparisons).
pub(super) fn attach_view(
    sections: &mut Option<Sections>,
    precision: Precision,
    mode: View,
) -> Result<()> {
    if precision.is_ternary() {
        return Ok(());
    }
    for section in sections.iter_mut().flat_map(|all| all.sections.iter_mut()) {
        precision::attach_tile(&mut section.tile, &section.latents, mode)?;
    }
    Ok(())
}

/// Section count `options` configures: the extra checkpoint payloads.
pub(super) fn count(options: &Options) -> usize {
    options.shared.map_or(0, |shared| shared.sections as usize)
}

#[path = "connected_sections_check.rs"]
mod check;
pub(super) use check::{report, validate};

#[cfg(test)]
#[path = "connected_sections_contracts.rs"]
mod contracts;
#[cfg(test)]
#[path = "connected_sections_rules_contracts.rs"]
mod rules;
