//! Latent weights, the tuned update rules and the per-call draw-counter stream; see AGENTS.md §core.

use super::tuning::{Knobs, Update};
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// Persistent f32 shadow weights plus two Adam moments per coefficient (12 bytes total).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct Latent {
    pub weights: Vec<f32>,
    pub moments: Vec<f32>,
    pub step: u64,
}

thread_local! {
    static CALLS: Cell<u64> = const { Cell::new(0) };
    // Resident training retains every initialized entry; packed trits do not replace it.
    static LATENTS: RefCell<BTreeMap<(usize, usize), Latent>> = const { RefCell::new(BTreeMap::new()) };
}

/// Reset the draw-counter stream and discard latent state; called when installing fresh knobs.
pub(super) fn reset() {
    CALLS.with(|cell| cell.set(0));
    LATENTS.with(|cell| cell.borrow_mut().clear());
}

/// Run `f` on `key`'s latent weights when float-forward mode is on and they exist.
pub(super) fn with_float<R>(key: (usize, usize), f: impl FnOnce(&[f32]) -> R) -> Option<R> {
    if !super::tuning::current().float_forward {
        return None;
    }
    LATENTS.with(|cell| cell.borrow().get(&key).map(|latent| f(&latent.weights)))
}

fn next_call() -> u64 {
    CALLS.with(|cell| {
        let call = cell.get();
        cell.set(call + 1);
        call
    })
}

/// Latent states keyed (owner, matrix); owner `TILES` is the tied E.
pub(super) type Latents = Vec<((usize, usize), Latent)>;

/// Test helper: off-lattice latents (half the trit plus a sine ripple) for the given matrices.
#[cfg(test)]
pub(super) fn ripple_latents(matrices: &[((usize, usize), &TernaryMatrix)]) -> Result<()> {
    let mut latents = Vec::with_capacity(matrices.len());
    for (key, matrix) in matrices {
        let weights = (0..matrix.len())
            .map(|i| {
                Ok(
                    (0.5 * f32::from(matrix.value(i)?) + 0.3 * (i as f32 * 0.7).sin())
                        .clamp(-1.0, 1.0),
                )
            })
            .collect::<Result<Vec<f32>>>()?;
        let moments = vec![0.0; 2 * weights.len()];
        latents.push((
            *key,
            Latent {
                weights,
                moments,
                step: 0,
            },
        ));
    }
    import(latents, 0);
    Ok(())
}

/// Each tracked latent's Adam step count, keyed like `LATENTS`; no weight or moment data copied.
pub(super) fn steps() -> Vec<((usize, usize), u64)> {
    LATENTS.with(|cell| {
        cell.borrow()
            .iter()
            .map(|(&key, latent)| (key, latent.step))
            .collect()
    })
}

/// Copy of the latent state and draw counter, for checkpoints.
#[cfg(test)]
pub(super) fn export() -> (Latents, u64) {
    let latents = LATENTS.with(|cell| {
        cell.borrow()
            .iter()
            .map(|(key, latent)| (*key, latent.clone()))
            .collect()
    });
    (latents, CALLS.with(Cell::get))
}

/// Borrow the latent map and draw counter without cloning optimizer state.
pub(super) fn with_latents<T>(
    f: impl FnOnce(&BTreeMap<(usize, usize), Latent>, u64) -> Result<T>,
) -> Result<T> {
    LATENTS.with(|cell| f(&cell.borrow(), CALLS.with(Cell::get)))
}

/// Replace the latent state and draw counter from a checkpoint.
pub(super) fn import(latents: Latents, calls: u64) {
    LATENTS.with(|cell| *cell.borrow_mut() = latents.into_iter().collect());
    CALLS.with(|cell| cell.set(calls));
}

/// Remove one tile's latent entries without changing the shared state or draw counter.
pub(super) fn take_tile(owner: usize) -> Latents {
    LATENTS.with(|cell| {
        let mut map = cell.borrow_mut();
        let keys: Vec<_> = map
            .range((owner, 0)..=(owner, usize::MAX))
            .map(|(&key, _)| key)
            .collect();
        keys.into_iter()
            .map(|key| (key, map.remove(&key).unwrap()))
            .collect()
    })
}

/// Attach a tile's latent entries only when all keys are unique and absent.
pub(super) fn put_tile(owner: usize, values: Latents) -> Result<()> {
    ensure!(owner < super::geometry::TILES, "invalid tile latent owner");
    LATENTS.with(|cell| {
        let mut map = cell.borrow_mut();
        ensure!(
            map.range((owner, 0)..=(owner, usize::MAX)).next().is_none(),
            "tile optimizer state is already resident"
        );
        let mut seen = std::collections::BTreeSet::new();
        for (key, _) in &values {
            ensure!(
                key.0 == owner && seen.insert(*key) && !map.contains_key(key),
                "duplicate or foreign tile latent"
            );
        }
        map.extend(values);
        Ok(())
    })
}

/// One tuned update of `matrix` keyed by (owner, matrix index); returns trit transitions.
pub(super) fn update(
    knobs: &Knobs,
    rate: f64,
    key: (usize, usize),
    matrix: &mut TernaryMatrix,
    gradient: &[f32],
) -> Result<usize> {
    ensure!(rate.is_finite() && rate >= 0.0, "invalid tuned rate");
    let stream = || stream(knobs.seed, key.0, key.1);
    match knobs.update {
        Update::Normalized => {
            matrix.apply_stochastic(gradient, effective(rate, gradient), stream())
        }
        Update::Raw => matrix.apply_stochastic(gradient, rate, stream()),
        Update::Latent => {
            // Move this entry out for mutation; other tiles' optimizer state stays resident.
            let existing = LATENTS.with(|cell| cell.borrow_mut().remove(&key));
            let mut latent = match existing {
                Some(latent) => latent,
                None => Latent {
                    weights: (0..matrix.len())
                        .map(|index| matrix.value(index).map(f32::from))
                        .collect::<Result<_>>()?,
                    moments: vec![0.0; 2 * matrix.len()],
                    step: 0,
                },
            };
            // Adam advances even when no shadow weight crosses a ternary threshold.
            latent.step += 1;
            let changed = matrix.apply_latent(
                &mut latent.weights,
                &mut latent.moments,
                gradient,
                rate,
                latent.step,
            );
            LATENTS.with(|cell| cell.borrow_mut().insert(key, latent));
            changed
        }
    }
}

/// Fresh counter-based draw stream per stochastic update call.
pub(super) fn stream(seed: u64, owner: usize, matrix: usize) -> u64 {
    hash(seed, &[owner as u64, matrix as u64, next_call()])
}

/// Pure SplitMix64 fold of a seed and coordinates; equal inputs always give equal output.
pub(super) fn hash(seed: u64, values: &[u64]) -> u64 {
    values
        .iter()
        .fold(mix(seed), |state, &value| mix(state ^ value))
}

/// Rate divided by this update's gradient RMS.
pub(super) fn effective(rate: f64, gradient: &[f32]) -> f64 {
    let square = gradient.iter().map(|g| f64::from(*g).powi(2)).sum::<f64>();
    let rms = (square / gradient.len().max(1) as f64).sqrt();
    if rms > 0.0 {
        rate / rms
    } else {
        0.0
    }
}

// SplitMix64 finalizer: https://prng.di.unimi.it/splitmix64.c
fn mix(value: u64) -> u64 {
    let mut x = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
