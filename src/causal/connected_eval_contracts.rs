//! Retention visibility and frozen evaluation contracts; see CONNECTED.md.

use super::super::connected_config;
use super::super::connected_predictor::Predictor;
use super::super::execution::TrainingSource;
use super::super::geometry::Geometry;
use super::super::scratch_dir::ScratchDir;
use super::super::storage_paths::digest;
use super::super::tuning::{self, Knobs};
use super::*;

/// The test's scratch directory; dropping the guard removes it, also on failure.
fn root() -> Result<ScratchDir> {
    ScratchDir::new("connected-measurement")
}

fn corpus(path: &Path, train: &[u8]) -> Result<Corpus> {
    let heldout = b"A small story.\x03";
    let entry = |bytes: &[u8]| json!({"bytes":bytes.len(),"sha256":digest(bytes)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry(train),"heldout.txt":entry(heldout)}}))?;
    std::fs::write(path.join("train.txt"), train)?;
    std::fs::write(path.join("heldout.txt"), heldout)?;
    std::fs::write(path.join("manifest.json"), &manifest)?;
    Corpus::open(path, &digest(&manifest), 32)
}

#[test]
fn retention_reads_only_wholly_observed_spans_with_exact_left_context() -> Result<()> {
    let path = root()?;
    let train = (0..1400)
        .map(|index| b'a' + (index % 26) as u8)
        .collect::<Vec<_>>();
    let corpus = corpus(&path, &train)?;
    assert!(corpus.retention_span(128, 128, 255)?.is_none());
    let (bytes, first) = corpus.retention_span(128, 128, 256)?.unwrap();
    assert_eq!(first, 34);
    assert_eq!(bytes, train[94..256]);
    assert_eq!(
        super::super::core_corpus::window_before(&bytes, first, 32),
        train[96..128]
    );
    assert!(corpus.retention_span(1300, 128, 3000)?.is_none());
    assert!(corpus.retention_span(u64::MAX, 128, u64::MAX).is_err());
    let options = Options {
        retention_spans: 2,
        ..Options::default()
    };
    assert!(retention_due(128, options));
    assert!(retention_due(1152, options));
    assert!(!retention_due(4224, options));
    assert_eq!(probe::retention(&corpus, 6, 127, false)?.len(), 0);
    assert_eq!(probe::retention(&corpus, 6, 128, false)?.len(), 1);
    assert_eq!(probe::retention(&corpus, 6, 1152, true)?.len(), 1);
    // Counted in open's digest pass, so windowed reads cannot move them (1400 = 26 * 53 + 22).
    let mut expected = [0; 256];
    for byte in b'a'..=b'z' {
        expected[usize::from(byte)] = if byte <= b'v' { 54 } else { 53 };
    }
    assert_eq!(corpus.byte_counts(), &expected);
    drop(corpus);
    Ok(())
}

#[test]
fn measured_replay_preserves_training_state_and_stop_reports_no_scores() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = root()?;
    let mut corpus = corpus(&path, &[b'a'; 256])?;
    let geometry = Geometry::fixture();
    corpus.window = geometry.context;
    let options = connected_config::Options {
        lanes: 2,
        workers: 2,
        visits: 1,
        ..connected_config::Options::default()
    };
    let state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new(&path.join("tiles"), geometry, 7, 2)?;
    let before = state.payloads()?;
    let assessment = Options {
        budget_seconds: 10,
        retention_spans: 2,
        sample_bytes: 2,
    };
    let evaluated = evaluate(&state, &mut store, &corpus, 1, &path, 1, assessment, false)?;
    assert_eq!(evaluated["heldout_targets"], corpus.heldout.len());
    assert_eq!(evaluated["heldout"]["probes"][0]["byte_offset"], 0);
    assert_eq!(evaluated["retention_eligible_spans"], 0);
    assert!(!evaluated["dev_bits_per_byte"].is_null());
    assert_eq!(before, state.payloads()?);
    corpus.validate(&state.cursor)?;
    std::fs::write(path.join("STOP"), b"fixture")?;
    let stopped = evaluate(&state, &mut store, &corpus, 1, &path, 1, assessment, false)?;
    assert_eq!(stopped["heldout_targets"], 0);
    assert!(stopped["dev_bits_per_byte"].is_null());
    assert_eq!(stopped["partial"], true);
    store.quiescent()?;
    drop(store);
    drop(corpus);
    Ok(())
}

#[test]
fn story_boundaries_trace_real_logits_and_keep_generated_evidence_distinct() -> Result<()> {
    use super::super::connected_decode;
    use super::super::core_corpus::END;
    use super::super::geometry::VOCAB;
    use crate::ternary::TernaryMatrix;

    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = root()?;
    let corpus = corpus(&path, &[b'a'; 256])?;
    let geometry = Geometry::fixture();
    let options = connected_config::Options {
        lanes: 2,
        workers: 2,
        visits: 1,
        ..connected_config::Options::default()
    };
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    state.shared.embedding =
        TernaryMatrix::from_values(geometry.width, VOCAB, &vec![0; geometry.width * VOCAB])?;
    let mut store = Store::new(&path.join("tiles"), geometry, 7, 2)?;
    let before = state.payloads()?;
    let mut generate = |prompt, window, decode| {
        generate_frozen_with_options(
            &state,
            &mut store,
            &[],
            prompt,
            window,
            3,
            Instant::now(),
            Duration::from_secs(10),
            decode,
        )
    };
    for (prompt, window, visible) in [
        ("X", 8, vec![END, b'X']),
        ("abc", 8, vec![END, b'a', b'b', b'c']),
        ("abcdefg", 8, b"\x03abcdefg".to_vec()),
        ("abcdefg", 4, b"defg".to_vec()),
    ] {
        let traced = generate(
            prompt,
            window,
            connected_decode::Options {
                trace_bytes: 2,
                quantize_audit: false,
                generated_feedback: true,
                ..Default::default()
            },
        )?;
        assert_eq!(traced["raw_bytes"], json!([0, 0, 0]));
        assert_eq!(traced["prompt_observations"], prompt.len().min(2));
        assert_eq!(traced["self_generated_publications"], 3);
        assert_eq!(traced["decode_trace"].as_array().unwrap().len(), 2);
        assert_eq!(
            traced["decode_trace"][0]["visible_prefix_bytes"],
            json!(visible)
        );
        for row in traced["decode_trace"].as_array().unwrap() {
            assert_eq!(row["selected_byte"], 0);
            assert_eq!(row["scalar_selected_byte"], 0);
            assert_eq!(row["scalar_readout_max_abs_error"], 0.0);
            assert_eq!(row["logits"].as_array().unwrap().len(), VOCAB);
            assert_eq!(row["maximum_ties"], VOCAB);
            assert_eq!(row["space_rank"], 33);
        }
        let ordinary = generate(prompt, window, connected_decode::Options::default())?;
        assert_eq!(ordinary["raw_bytes"], traced["raw_bytes"]);
        assert!(ordinary.get("decode_trace").is_none());
        let frozen = generate(
            prompt,
            window,
            connected_decode::Options {
                trace_bytes: 2,
                quantize_audit: false,
                generated_feedback: false,
                ..Default::default()
            },
        )?;
        assert_eq!(frozen["self_generated_publications"], 0);
        assert_eq!(
            frozen["decode_trace"][0]["memory_publications"],
            frozen["decode_trace"][1]["memory_publications"]
        );
    }
    assert_eq!(before, state.payloads()?);
    drop(store);
    drop(corpus);
    Ok(())
}

#[test]
fn labels_reach_fits_and_lanes_but_scores_stay_target_free() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = root()?;
    let mut corpus = corpus(&path, &[b'a'; 256])?;
    let geometry = Geometry::fixture();
    corpus.window = geometry.context;
    let options = connected_config::Options {
        lanes: 2,
        workers: 2,
        visits: 2,
        ..connected_config::Options::default()
    };
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new(&path.join("tiles"), geometry, 7, 2)?;
    for _ in 0..8 {
        super::super::connected_step::step(&mut state, &mut store, &corpus, options, 0.003, false)?;
    }
    // Observed fits saw real labels: every row's label block (columns 3d..4d) left its seed.
    let (d, seeded) = (geometry.width, Predictor::new(geometry, 7)?);
    for start in (0..d).map(|row| row * (4 * d + 2)) {
        let block = start + 3 * d..start + 4 * d;
        assert_ne!(
            state.predictor.weights[block.clone()],
            seeded.weights[block]
        );
    }
    // Trusted label-conditioned predictions: replayed lane records depend on the scored byte.
    assert!(state.predictor.trust(super::super::execution::START_TILE) > 0.0);
    // From bit-identical states, the lanes' own predicted corrections follow the label byte.
    let prefix = state.shared.prefix(b"abc")?;
    let mut lanes = |byte| -> Result<Vec<Vec<f32>>> {
        let mut copy = State::restore(&state.payloads()?, geometry, 7)?;
        let start = copy.epoch;
        super::super::connected_lanes::replay(&mut copy, &mut store, &prefix, byte, options)?;
        let published = copy.memory.records.into_iter();
        Ok(published
            .filter(|record| record.epoch >= start)
            .map(|record| record.correction)
            .collect())
    };
    let labelled = lanes(b'c')?;
    assert_ne!(
        labelled,
        lanes(b'x')?,
        "lane predictions ignore the label byte"
    );
    assert_eq!(
        labelled,
        lanes(b'c')?,
        "one byte and state must repeat exactly"
    );
    // The oracle: the same warm-up, then the probe's readout with no target observed at all.
    let readout = {
        let mut reference = replay::Replay::new(&state, &mut store)?;
        reference.score(b"ab", b'c')?;
        reference.readout(b"abc")?
    };
    for target in 0..=u8::MAX {
        let mut replay = replay::Replay::new(&state, &mut store)?;
        // An earlier observed byte leaves label-conditioned memory the probe forward reads.
        replay.score(b"ab", b'c')?;
        // `gather` takes `Readout::loss` of identical logits: any target leak breaks bit equality.
        let scored = replay.score(b"abc", target)?;
        assert_eq!(scored, readout.loss(target)?, "target {target}");
    }
    store.quiescent()?;
    drop(store);
    drop(corpus);
    Ok(())
}
