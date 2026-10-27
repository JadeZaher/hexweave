use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub(super) const VOCAB: usize = 256;
pub(super) const TILES: usize = 7;
pub(super) const VISITS: usize = 4;
pub(super) const MATRIX_COUNT: usize = 11;
pub(super) const EVIDENCE_RULE: &str = "threshold-one-canonical-zero-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Geometry {
    pub width: usize,
    pub ffn: usize,
    pub heads: usize,
    pub context: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct Ledger {
    pub coefficients: usize,
    pub shared_coefficients: usize,
    pub largest_tile_coefficients: usize,
    pub stored_weight_bytes: usize,
    pub prefix_bytes: usize,
    pub replay_scratch_bytes: usize,
    pub anchor_core_payload_bound: usize,
    pub predictor_phase_payload_bound: usize,
    pub predictor_joint_pair_cap: usize,
    pub predictor_joint_signed_candidates_per_batch: usize,
    pub predictor_joint_candidate_pair_terms_per_batch: usize,
    pub tile_io_reserve_bytes: usize,
    pub physical_admission: bool,
}

impl Geometry {
    pub fn fixture() -> Self {
        Self {
            width: 16,
            ffn: 48,
            heads: 2,
            context: 8,
        }
    }

    pub fn candidate() -> Self {
        Self {
            width: 128,
            ffn: 384,
            heads: 4,
            context: 1024,
        }
    }

    pub fn validate(self) -> Result<()> {
        ensure!(
            self == Self::fixture() || self == Self::candidate(),
            "unsupported causal geometry"
        );
        for id in 0..TILES {
            for (rows, cols) in self.shapes(id)? {
                let count = rows
                    .checked_mul(cols)
                    .ok_or_else(|| anyhow::anyhow!("shape overflow"))?;
                ensure!(
                    (1..=1_048_576).contains(&count),
                    "matrix exceeds existing ternary bound"
                );
            }
        }
        Ok(())
    }

    pub fn features(self) -> usize {
        3 * self.width + 2
    }

    pub fn routes(self, id: usize) -> Result<usize> {
        ensure!(id < TILES, "invalid causal tile ID");
        Ok(if id == 3 { 7 } else { 4 })
    }

    pub fn shapes(self, id: usize) -> Result<[(usize, usize); MATRIX_COUNT]> {
        ensure!(
            self == Self::fixture() || self == Self::candidate(),
            "unsupported causal geometry"
        );
        let (d, m, r) = (self.width, self.ffn, self.routes(id)?);
        Ok([
            (d, d),
            (d, d),
            (d, d),
            (d, d),
            (m, d),
            (m, d),
            (d, m),
            (3, m),
            (r, 2 * d),
            (d, 2 * d),
            (2 * d + r, d),
        ])
    }

    pub fn tile_coefficients(self, id: usize) -> Result<usize> {
        self.validate()?;
        Ok(self.shapes(id)?.iter().map(|(r, c)| r * c).sum())
    }

    pub fn ledger(self, length: usize) -> Result<Ledger> {
        self.validate()?;
        ensure!(
            (1..=self.context).contains(&length),
            "invalid ledger prefix length"
        );
        let (d, m, h, l, r) = (self.width, self.ffn, self.heads, length, 7);
        let shared = VOCAB * d + d * self.features();
        let tile = self.tile_coefficients(3)?;
        let mut total = shared;
        for id in 0..TILES {
            total += self.tile_coefficients(id)?;
        }
        let prefix = 4 * l * d + 5 * l;
        let scratch = 4 * (2 * h * d + 2 * h * l + 9 * m + 64 * d + 8 * r + 4);
        let history = 8 * (4 * (self.features() + d) + 80) + 16;
        let anchor = shared / 4
            + 2 * shared
            + tile / 4
            + 6 * tile
            + 4 * VOCAB * d
            + prefix
            + 8 * VISITS * d
            + 32 * VISITS
            + 8 * d
            + 4 * l * d
            + 8 * VOCAB
            + scratch
            + history;
        let predictor_weights = d * self.features();
        let predictor = shared.div_ceil(4)
            + 2 * predictor_weights.div_ceil(4)
            + predictor_weights
            + history
            + 4 * (4 * (self.features() + d) + 80)
            + 16 * d
            + 48 * d
            + 8 * d
            + d
            + 16 * d
            + 4096
            + prefix
            + 8 * VISITS * d
            + 32 * VISITS
            + 8 * d;
        Ok(Ledger {
            coefficients: total,
            shared_coefficients: shared,
            largest_tile_coefficients: tile,
            stored_weight_bytes: total / 4,
            prefix_bytes: prefix,
            replay_scratch_bytes: scratch,
            anchor_core_payload_bound: anchor,
            predictor_phase_payload_bound: predictor,
            predictor_joint_pair_cap: 4,
            predictor_joint_signed_candidates_per_batch: 2 * predictor_weights,
            predictor_joint_candidate_pair_terms_per_batch: 8 * predictor_weights,
            tile_io_reserve_bytes: 2 * (tile / 4 + 2 * tile) + 4096,
            physical_admission: false,
        })
    }
}
