use crate::topology::Mesh;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub(super) const VOCAB: usize = 256;
/// Hex radius of the tile mesh: 1 (7 tiles) by default, 2 (19 tiles) with `tiles19`.
#[cfg(not(feature = "tiles19"))]
pub(super) const RADIUS: usize = 1;
#[cfg(feature = "tiles19")]
pub(super) const RADIUS: usize = 2;
/// Tiles in a radius-`RADIUS` hex; the id layout is in CONNECTED.md §Tile count.
pub(super) const TILES: usize = 3 * RADIUS * (RADIUS + 1) + 1;
/// The centre tile `(q, r) = (0, 0)`: the middle id of the point-symmetric q-then-r order.
pub(super) const START_TILE: usize = TILES / 2;
pub(super) const VISITS: usize = 4;
pub(super) const MATRIX_COUNT: usize = 11;
pub(super) const EVIDENCE_RULE: &str = "threshold-one-canonical-zero-v1";

/// The fixed routing mesh in stable q-then-r axial order, built once per process.
pub(super) fn mesh() -> &'static Mesh {
    static MESH: OnceLock<Mesh> = OnceLock::new();
    MESH.get_or_init(|| Mesh::new(RADIUS).expect("compile-time hex radius is valid"))
}

/// Tile counts some build of this binary has: the default 7 and the `tiles19` 19.
pub(super) const BUILT_TILE_COUNTS: [usize; 2] = [7, 19];

/// Describes `found` tiles against this build and names the build that has them; None if equal.
pub(super) fn tile_mismatch(found: usize) -> Option<String> {
    let build = match found {
        19 => "build with `cargo build --release --features tiles19`",
        7 => "use the default build (`cargo build --release`, no tiles19 feature)",
        _ => "only 7- and 19-tile builds exist",
    };
    (found != TILES).then(|| format!("{found} tiles but this binary is built for {TILES}; {build}"))
}

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

    /// Legal route actions of tile `id`: one per mesh neighbour plus halt.
    pub fn routes(self, id: usize) -> Result<usize> {
        ensure!(id < TILES, "invalid causal tile ID");
        Ok(mesh().neighbors(id)?.len() + 1)
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
        // The centre tile has the most route actions (six neighbours plus halt) at any radius.
        let (d, m, h, l, r) = (
            self.width,
            self.ffn,
            self.heads,
            length,
            self.routes(START_TILE)?,
        );
        let shared = VOCAB * d + d * self.features();
        let tile = self.tile_coefficients(START_TILE)?;
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

#[cfg(test)]
mod tests {
    use super::{mesh, Geometry, RADIUS, START_TILE, TILES};

    #[test]
    fn hex_mesh_is_symmetric_connected_and_centred() -> anyhow::Result<()> {
        let mesh = mesh();
        assert_eq!(mesh.len(), TILES);
        let centre = mesh.coordinates()[START_TILE];
        assert_eq!((centre.q, centre.r), (0, 0));
        let mut reached = [false; TILES];
        let mut frontier = vec![START_TILE];
        reached[START_TILE] = true;
        while let Some(id) = frontier.pop() {
            for next in mesh.neighbors(id)? {
                assert!(
                    mesh.neighbors(next)?.contains(&id),
                    "asymmetric {id}-{next}"
                );
                if !std::mem::replace(&mut reached[next], true) {
                    frontier.push(next);
                }
            }
        }
        assert!(reached.iter().all(|&tile| tile), "unreachable tile");
        for (id, point) in mesh.coordinates().iter().enumerate() {
            // Interior tiles have six neighbours; outer-ring corners three, edges four.
            let ring = [point.q, point.r, -point.q - point.r].map(i32::abs);
            let corner = ring.iter().filter(|&&value| value == RADIUS as i32).count() == 2;
            let expected = match ring.into_iter().max() {
                Some(distance) if distance < RADIUS as i32 => 6,
                _ if corner => 3,
                _ => 4,
            };
            assert_eq!(mesh.neighbors(id)?.len(), expected, "tile {id}");
            assert_eq!(Geometry::candidate().routes(id)?, expected + 1);
        }
        if cfg!(feature = "tiles19") {
            assert_eq!((TILES, START_TILE), (19, 9));
        } else {
            assert_eq!((TILES, START_TILE), (7, 3));
        }
        Ok(())
    }
}
