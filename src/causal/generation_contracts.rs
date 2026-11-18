use super::decoding::{
    story_opening,
    Decoder::{self, Greedy},
};
use super::evaluation::parse_answer;
use super::generation::{generate, generate_story, lowest_maximum, Reason, Terminal};
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
    assert_eq!(store.updates, [0; super::geometry::TILES]);
    assert_eq!(state.global_opportunity, 0);
    assert_eq!(state.cursor.supervised_targets, 0);
    Ok(())
}

#[test]
fn greedy_ties_newlines_stop_and_context_limits_are_exact() -> Result<()> {
    assert_eq!(lowest_maximum(&[1.0, 1.0, 0.0])?, 0);
    assert_eq!(lowest_maximum(&[-0.0, 0.0])?, 0);
    assert_eq!(lowest_maximum(&[0.0, -0.0])?, 0);
    assert!(lowest_maximum(&[f32::NAN]).is_err());
    assert!(lowest_maximum(&[f32::INFINITY, 0.0]).is_err());
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
    let multiline = generate_story(&state, &mut store, b"X", 3, 4, Greedy, &mut || Ok(None))?;
    assert_eq!(multiline.raw_bytes, b"\n\n\n");
    assert_eq!(multiline.stop_reason, Reason::ByteBudget);
    let (state, mut store) = fixed_byte(Some(super::core_corpus::END))?;
    let end = generate_story(&state, &mut store, b"X", 3, 4, Greedy, &mut || Ok(None))?;
    assert!(end.raw_bytes.is_empty());
    assert_eq!(end.stop_reason, Reason::EndOfStory);
    assert_eq!(end.before_sha256, end.after_sha256);
    let general = generate(&state, &mut store, b"X", 3, &mut || Ok(None))?;
    assert_eq!(general.raw_bytes, [super::core_corpus::END; 3]);
    assert_eq!(general.stop_reason, Reason::ByteBudget);
    Ok(())
}

#[test]
fn story_generation_uses_the_latest_window_beyond_context() -> Result<()> {
    for (prefix, expected) in [
        (b"abc".as_slice(), b"abc".as_slice()),
        (b"abcd".as_slice(), b"abcd".as_slice()),
        (b"abcde".as_slice(), b"bcde".as_slice()),
        (b"abcdef".as_slice(), b"cdef".as_slice()),
        (b"abcdefg".as_slice(), b"defg".as_slice()),
    ] {
        assert_eq!(super::generation::recent_window(prefix, 4), expected);
    }
    let _guard = super::tuning::scoped(super::tuning::Knobs {
        history: false,
        ..super::tuning::Knobs::PINNED
    });
    let source = fixture::Source::new();
    let state = State::with_cursor(Geometry::fixture(), 83, source.cursor())?;
    let mut store = fixture::Store::new(83)?;
    let long = generate_story(&state, &mut store, b"abcdefg", 3, 4, Greedy, &mut || {
        Ok(None)
    })?;
    let suffix = generate_story(&state, &mut store, b"defg", 3, 4, Greedy, &mut || Ok(None))?;
    assert_eq!(long.raw_bytes, suffix.raw_bytes);
    assert_eq!(long.stop_reason, suffix.stop_reason);
    assert!(generate(&state, &mut store, b"abcdefg", 3, &mut || Ok(None)).is_err());
    Ok(())
}

#[test]
fn story_window_replays_prompt_and_generated_history() -> Result<()> {
    use super::execution::forward_path;
    use super::generation::Routes;
    use super::replay::{Replay, SPAN_WARMUP};

    let _guard = super::tuning::scoped(super::tuning::Knobs {
        history: true,
        ..super::tuning::Knobs::PINNED
    });
    let source = fixture::Source::new();
    let state = State::with_cursor(Geometry::fixture(), 83, source.cursor())?;
    let mut store = fixture::Store::new(83)?;
    let prompt = b"abcdefg";
    let window = 4;
    let budget = 3;
    let actual = generate_story(
        &state,
        &mut store,
        prompt,
        budget,
        window,
        Greedy,
        &mut || Ok(None),
    )?;

    let mut replay = Replay::default();
    let mut visible = prompt.to_vec();
    let mut expected = Vec::new();
    let mut routes = Routes::default();
    let mut reason = Reason::ByteBudget;
    let mut predictions = 0;
    let mut warmup_history_nonzero = false;
    for position in prompt.len().saturating_sub(SPAN_WARMUP).max(1)..prompt.len() + budget {
        let prior = &visible[..position];
        let start = prior.len().saturating_sub(window);
        let prefix = state.shared.prefix(&prior[start..])?;
        let history = replay.input(&state)?;
        if position == prompt.len() {
            warmup_history_nonzero = history.iter().any(|value| *value != 0.0);
        }
        let trace = forward_path(
            &mut store,
            state.geometry,
            &prefix,
            history,
            &mut None,
            state.tile_versions,
        )?;
        let target = if position < prompt.len() {
            prompt[position]
        } else {
            predictions += 1;
            routes.include(&trace, &state)?;
            let readout = state.shared.readout(&trace.output)?;
            let byte = u8::try_from(lowest_maximum(&readout.logits)?)?;
            if byte == super::core_corpus::END {
                reason = Reason::EndOfStory;
                break;
            }
            expected.push(byte);
            visible.push(byte);
            byte
        };
        replay.observe(&state, &mut store, &prefix, &trace, target)?;
    }
    assert!(
        warmup_history_nonzero,
        "fixture must exercise warmup history"
    );
    assert!(
        predictions >= 2,
        "fixture must exercise continuation history"
    );
    assert_eq!(actual.raw_bytes, expected);
    assert_eq!(actual.stop_reason, reason);
    assert_eq!(actual.routes.visits, routes.visits);
    assert_eq!(actual.routes.route_calls, routes.route_calls);
    assert_eq!(actual.routes.halts, routes.halts);
    assert_eq!(actual.routes.forced_exits, routes.forced_exits);
    Ok(())
}

/// Nucleus sampling keeps only the most probable set holding top-p of the tempered mass, draws it
/// in proportion, collapses to greedy at low temperature or a tiny nucleus, and refuses bad settings.
#[test]
fn nucleus_sampling_draws_the_tempered_top_p_set() -> Result<()> {
    let probabilities = [0.5f32, 0.3, 0.15, 0.05];
    let logits = probabilities.map(f32::ln);
    let draws = 20_000;
    let share = |temperature, top_p| -> Result<Vec<f64>> {
        let decoder = Decoder::Nucleus {
            temperature,
            top_p,
            seed: 11,
        };
        let mut counts = [0usize; 4];
        for step in 0..draws {
            counts[decoder.choose(&logits, step)?] += 1;
        }
        Ok(counts.map(|count| count as f64 / draws as f64).to_vec())
    };
    for (temperature, top_p, expected) in [
        (1.0, 1.0, [0.5, 0.3, 0.15, 0.05]),
        (1.0, 0.8, [0.625, 0.375, 0.0, 0.0]),
        (0.01, 1.0, [1.0, 0.0, 0.0, 0.0]),
        (1.0, 0.01, [1.0, 0.0, 0.0, 0.0]),
    ] {
        for (actual, wanted) in share(temperature, top_p)?.iter().zip(expected) {
            assert!(
                (actual - wanted).abs() < 0.015,
                "T {temperature} p {top_p}: {actual} vs {wanted}"
            );
        }
    }
    assert_eq!(share(1.0, 0.8)?[2..], [0.0, 0.0], "the tail is never drawn");
    for (temperature, top_p) in [
        (0.0, 0.9),
        (f64::NAN, 0.9),
        (11.0, 0.9),
        (1.0, 0.0),
        (1.0, 1.01),
    ] {
        let decoder = Decoder::Nucleus {
            temperature,
            top_p,
            seed: 11,
        };
        assert!(
            decoder.validate().is_err(),
            "accepted T {temperature} p {top_p}"
        );
    }
    Ok(())
}

/// Story openings end a full window with the prompt after held-out text ending at a story boundary.
#[test]
fn story_openings_fill_the_window_from_evenly_spaced_boundaries() -> Result<()> {
    // Story ends (0x03) at 3, 13, 19 and 24; window 8 admits 13, 19 and 24.
    let heldout = b"one\x03two story\x03three\x03four\x03";
    let cases: [(usize, usize, &[u8], &[u8]); 4] = [
        (0, 1, b"Hi", b"three\x03Hi"),
        (0, 3, b"Hi", b"story\x03Hi"),
        (2, 3, b"Hi", b"\x03four\x03Hi"),
        (0, 1, b"a long prompt", b"a long prompt"),
    ];
    for (slot, slots, prompt, expected) in cases {
        let opening = story_opening(heldout, 8, slot, slots, prompt)?;
        assert_eq!(opening, expected, "slot {slot} of {slots}");
    }
    assert!(story_opening(heldout, 8, 1, 1, b"Hi").is_err());
    assert!(story_opening(heldout, 26, 0, 1, b"Hi").is_err());
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
