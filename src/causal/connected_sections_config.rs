//! `connected.shared` options and admission; see CONNECTED.md §Shared sections.

use super::super::connected_config::{CreditMode, Options};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Most sections and the longest shared wave admitted.
pub(in super::super) const MAX_SECTIONS: u32 = 4;
const MAX_WAVE: u32 = 64;

/// `connected.shared`: section count, shared-wave length, queue selection, teaching weight and
/// write credit.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct SharedOptions {
    pub sections: u32,
    #[serde(default = "default_wave", skip_serializing_if = "is_default_wave")]
    pub wave: u32,
    #[serde(default = "default_queue", skip_serializing_if = "is_default_queue")]
    pub queue: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub teach: f64,
    #[serde(default, skip_serializing_if = "SharedCredit::is_default")]
    pub credit: SharedCredit,
}

/// The output adjoint a section's write uses for each anchor.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(in super::super) enum SharedCredit {
    /// The Lambert-weighted mix of the visited tiles' output corrections.
    #[default]
    Mix,
    /// The exact adjoint at the read over the section count, unit-clipped.
    Entry,
}

impl SharedCredit {
    fn is_default(&self) -> bool {
        *self == Self::Mix
    }
}

fn default_wave() -> u32 {
    8
}

fn is_default_wave(value: &u32) -> bool {
    *value == default_wave()
}

fn default_queue() -> bool {
    true
}

fn is_default_queue(value: &bool) -> bool {
    *value
}

fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

/// Admission: 1..=4 sections, a 1..=64 wave that is a multiple of `anchor_batch`, teach in
/// [0, 0.5], `anchor_exact` credit and neither replay (replays do not read the sections).
pub(in super::super) fn validate_options(options: &Options) -> Result<()> {
    let Some(shared) = options.shared else {
        return Ok(());
    };
    ensure!(
        (1..=MAX_SECTIONS).contains(&shared.sections),
        "connected shared sections must be 1..={MAX_SECTIONS}"
    );
    ensure!(
        (1..=MAX_WAVE).contains(&shared.wave) && shared.wave.is_multiple_of(options.anchor_batch),
        "connected shared wave must be 1..={MAX_WAVE} and a multiple of anchor_batch"
    );
    ensure!(
        shared.teach.is_finite() && (0.0..=0.5).contains(&shared.teach),
        "connected shared teach must be in [0, 0.5]"
    );
    ensure!(
        options.credit_mode == CreditMode::AnchorExact,
        "connected shared requires credit_mode anchor_exact"
    );
    ensure!(
        options.replay.is_none() && options.restart.is_none_or(|restart| restart.replay == 0),
        "connected shared refuses connected.replay and restart.replay"
    );
    Ok(())
}
