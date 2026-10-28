//! Float-forward numerics: the input-adjoint and tied-embedding oracles rerun on off-lattice latents,
//! failing outright if float mode does not engage; see AGENTS.md §core.

use super::geometry::{Geometry, TILES};
use super::shared::Shared;
use super::tile::Tile;
use super::tuning::{self, Knobs, Scoped};
use anyhow::{ensure, Result};

/// Float knobs plus off-lattice latents for the fixture tile 3 and E built from `seed`.
fn float_fixture(seed: u64) -> Result<Scoped> {
    let guard = tuning::scoped(Knobs {
        float_forward: true,
        ..Knobs::ORACLE
    });
    let geometry = Geometry::fixture();
    let (shared, tile) = (Shared::new(geometry, seed)?, Tile::new(geometry, 3, seed)?);
    let mut keys: Vec<_> = (0..tile.matrices.len())
        .map(|index| ((3, index), &tile.matrices[index]))
        .collect();
    keys.push(((TILES, 0), &shared.embedding));
    tuning::ripple_latents(&keys)?;
    ensure!(
        shared.embedding_column(usize::from(b'a'))?
            != shared.embedding.column(usize::from(b'a'))?,
        "float mode did not engage"
    );
    Ok(guard)
}

#[test]
fn float_mode_passes_the_input_adjoint_and_tied_embedding_oracles() -> Result<()> {
    {
        let _float = float_fixture(37)?;
        super::numerical_tests::check_conditional_adjoint()?;
    }
    let _float = float_fixture(41)?;
    super::numerical_tests::check_tied_embedding()
}
