use super::checkpoint_owner::Terminal;
use super::evaluation::parse_answer;
use super::generation::{generate, lowest_maximum, Reason};
use super::geometry::{Geometry, VOCAB};
use super::state::State;
use crate::ternary::TernaryMatrix;
use anyhow::Result;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

fn fixed_byte(preferred: Option<u8>) -> Result<(State, fixture::Store)> {
    let source = fixture::Source::new();
    let geometry = Geometry::fixture();
    let mut state = State::with_cursor(geometry, 83, source.cursor())?;
    let mut store = fixture::Store::new(83)?;
    let mut embedding = vec![0i8; geometry.width * VOCAB];
    if let Some(byte) = preferred {
        for row in 0..geometry.width {
            embedding[row * VOCAB + usize::from(byte)] = 1;
        }
    }
    state.shared.embedding = TernaryMatrix::from_values(geometry.width, VOCAB, &embedding)?;
    for tile in &mut store.tiles {
        for (matrix, (rows, cols)) in tile.matrices.iter_mut().zip(geometry.shapes(tile.id)?) {
            *matrix = TernaryMatrix::from_values(rows, cols, &vec![0; rows * cols])?;
        }
    }
    Ok((state, store))
}

#[test]
fn generation_is_immutable_and_preserves_invalid_utf8_without_repair() -> Result<()> {
    let (state, mut store) = fixed_byte(Some(255))?;
    let result = generate(&state, &mut store, b"X", 2, &mut || Ok(None))?;
    assert_eq!(result.raw_bytes, [255, 255]);
    assert!(!result.valid_utf8);
    assert_eq!(result.stop_reason, Reason::ByteBudget);
    assert_eq!(result.before_sha256, result.after_sha256);
    assert_eq!(store.updates, [0; 7]);
    assert_eq!(state.global_opportunity, 0);
    assert_eq!(state.cursor.supervised_targets, 0);
    Ok(())
}

#[test]
fn greedy_ties_newlines_stop_and_context_limits_are_exact() -> Result<()> {
    assert_eq!(lowest_maximum(&[1.0, 1.0, 0.0])?, 0);
    assert!(lowest_maximum(&[f32::NAN]).is_err());
    let (state, mut store) = fixed_byte(None)?;
    assert_eq!(
        generate(&state, &mut store, b"X", 1, &mut || Ok(None))?.raw_bytes,
        [0]
    );
    for (prompt, budget) in [
        (b"".as_slice(), 1),
        (b"X".as_slice(), 0),
        (b"12345678".as_slice(), 1),
        ([255u8].as_slice(), 1),
    ] {
        assert!(generate(&state, &mut store, prompt, budget, &mut || Ok(None)).is_err());
    }
    let stopped = generate(&state, &mut store, b"X", 2, &mut || {
        Ok(Some(Terminal::UserStop))
    })?;
    assert!(stopped.raw_bytes.is_empty());
    assert_eq!(stopped.stop_reason, Reason::Interrupted(Terminal::UserStop));
    let (state, mut store) = fixed_byte(Some(b'\n'))?;
    let newline = generate(&state, &mut store, b"X", 3, &mut || Ok(None))?;
    assert_eq!(newline.raw_bytes, b"\n");
    assert_eq!(newline.stop_reason, Reason::Newline);
    Ok(())
}

#[test]
fn generated_json_scoring_rejects_missing_duplicate_and_extra_fields() {
    assert!(parse_answer(b"{\"answer\":\"UNKNOWN\",\"citation\":null}\n").is_some());
    for invalid in [
        b"{\"answer\":\"UNKNOWN\"}".as_slice(),
        b"{\"answer\":\"X\",\"answer\":\"Y\",\"citation\":null}".as_slice(),
        b"{\"answer\":\"X\",\"citation\":3}".as_slice(),
        b"{\"answer\":\"X\",\"citation\":null,\"extra\":0}".as_slice(),
        b"not json".as_slice(),
    ] {
        assert!(parse_answer(invalid).is_none());
    }
}
