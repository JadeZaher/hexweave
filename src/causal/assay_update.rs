use super::shared::Shared;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Rule {
    Dense,
    Magnitude,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metrics {
    pub attempts: u64,
    pub clamped: u64,
    pub transitions: u64,
    #[serde(with = "bits", rename = "probability_sum_bits")]
    pub probability_sum: f64,
}

pub(super) fn base(seed: u64, anchor: u64) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"hexweave/retained-text/embedding-adjacent-v1\0");
    hash.update(seed.to_le_bytes());
    hash.update(anchor.to_le_bytes());
    let digest = hash.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("eight SHA bytes"))
}

// SplitMix64 counter permutation; see the retained-text contract.
pub(super) fn draw(base: u64, coordinate: usize) -> f64 {
    let mut x = base.wrapping_add((coordinate as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Profile E update shared by the assay ledger and the core path; None keeps the pinned rule.
pub(super) fn tuned(shared: &mut Shared, gradient: &[f32]) -> Result<Option<u64>> {
    let knobs = super::tuning::current();
    let Some(rate) = knobs.embed_rate else {
        return Ok(None);
    };
    super::norm::vector(gradient, shared.embedding.len())?;
    let next = shared
        .version
        .checked_add(1)
        .context("E version overflow")?;
    let key = (super::geometry::TILES, 0);
    let moved = super::tuning::update(&knobs, rate, key, &mut shared.embedding, gradient)?;
    if moved > 0 {
        shared.version = next;
    }
    Ok(Some(u64::try_from(moved)?))
}

pub(super) fn apply(
    shared: &mut Shared,
    gradient: &[f32],
    rule: Rule,
    seed: u64,
    anchor: u64,
) -> Result<Metrics> {
    super::norm::vector(gradient, shared.embedding.len())?;
    let next = shared
        .version
        .checked_add(1)
        .context("E version overflow")?;
    if let Some(moved) = tuned(shared, gradient)? {
        return Ok(Metrics {
            attempts: moved,
            clamped: 0,
            transitions: moved,
            probability_sum: moved as f64,
        });
    }
    let base = base(seed, anchor);
    let mut metrics = Metrics::default();
    let mut packed = shared.packed().to_vec();
    for (index, &g) in gradient.iter().enumerate() {
        let probability = match rule {
            Rule::Dense => f64::from(u8::from(g != 0.0)),
            Rule::Magnitude => f64::from(g).abs().min(1.0),
        };
        metrics.probability_sum += probability;
        if probability == 0.0 || draw(base, index) >= probability {
            continue;
        }
        metrics.attempts += 1;
        let old = shared.embedding.value(index)?;
        let new = (old + if g > 0.0 { -1 } else { 1 }).clamp(-1, 1);
        if new == old {
            metrics.clamped += 1;
        } else {
            metrics.transitions += 1;
            let shift = (index % 4) * 2;
            packed[index / 4] = (packed[index / 4] & !(3 << shift)) | (((new + 1) as u8) << shift);
        }
    }
    if rule == Rule::Dense {
        ensure!(
            shared.apply(gradient)? as u64 == metrics.transitions,
            "dense E reconciliation"
        );
        ensure!(shared.packed() == packed, "dense E packed reconciliation");
    } else if metrics.transitions > 0 {
        let replacement = Shared::from_packed(
            shared.geometry,
            next,
            &packed,
            super::geometry::EVIDENCE_RULE,
        )?;
        *shared = replacement;
    }
    Ok(metrics)
}

impl Metrics {
    pub fn add(&mut self, value: &Self) -> Result<()> {
        self.attempts = self
            .attempts
            .checked_add(value.attempts)
            .context("E attempts overflow")?;
        self.clamped = self
            .clamped
            .checked_add(value.clamped)
            .context("E clamps overflow")?;
        self.transitions = self
            .transitions
            .checked_add(value.transitions)
            .context("E transitions overflow")?;
        self.probability_sum += value.probability_sum;
        ensure!(self.probability_sum.is_finite(), "E probability overflow");
        Ok(())
    }
}

mod bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(value.to_bits())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        u64::deserialize(deserializer).map(f64::from_bits)
    }
}
