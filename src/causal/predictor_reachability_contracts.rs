use super::{probe, rules, Predictor, BASELINE_KERNEL, INITIAL_HASH, PAIR_HASH};
use anyhow::Result;
use serde::Serialize;
use serde_json::json;

const LABEL: &str = "PREDICTOR_REACHABILITY ";
const PREFIX_HASH: &str = "f9fd1f11a3fbad57533b99618bc05dd5c6a217a61cfb5192d1b7a0ae23b8b425";
const OBSERVATIONS_HASH: &str = "0c15ccc3d161926853a94f3c5742e732fcc67c31e7956bbf619c5a7fc056c43f";
const PASSES: usize = 16;

#[derive(Serialize)]
struct Delivered {
    mse: f64,
    zero_mse: f64,
    target_dot: f64,
    target_norm: f64,
    correction_norm: f64,
    cosine: Option<f64>,
    gate_passed: bool,
}

#[derive(Serialize)]
pub(super) struct Measurement {
    presentations: usize,
    pub(super) predictor_sha256: String,
    pub(super) version: u64,
    canonical_trit_counts: [usize; 3],
    raw_all_pair_mean_mse: f64,
    pub(super) raw_pair_mean_mse: Vec<f64>,
    delivered: Vec<Delivered>,
    pub(super) gate_passed: bool,
}

#[derive(Default, Serialize)]
struct Counts {
    signed_trials: usize,
    no_move_candidates: usize,
    accepted_rows: usize,
    rejected_rows: usize,
    transitions: usize,
    stalled_presentations: usize,
}

#[derive(Serialize)]
struct Pass {
    pass: usize,
    before_sha256: String,
    counts: Counts,
    after: Measurement,
}

#[test]
fn fixed_predictor_reachability() -> Result<()> {
    assert_eq!(crate::ternary::kernel_name()?, BASELINE_KERNEL);
    let probe::Frozen {
        mut predictor,
        pairs,
        route,
    } = probe::freeze()?;
    assert_eq!(pairs.len(), 4);
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    assert_eq!(probe::predictor_identity(&predictor), INITIAL_HASH);
    assert_eq!(predictor.version, 0);
    let initial = measure(&predictor, &pairs, 0)?;
    let mut passes: Vec<Pass> = Vec::with_capacity(PASSES);
    for pass in 1..=PASSES {
        let next = run_pass(&mut predictor, &pairs, pass)?;
        let previous = passes.last().map_or(&initial, |p| &p.after);
        assert_eq!(next.before_sha256, previous.predictor_sha256);
        assert_eq!(
            next.after.version,
            previous.version + (4 - next.counts.stalled_presentations) as u64
        );
        if pass == 1 {
            assert_eq!(next.after.predictor_sha256, PREFIX_HASH);
            assert_eq!(next.after.version, 4);
        }
        passes.push(next);
    }
    assert_eq!(passes.len(), PASSES);
    let totals = Counts {
        signed_trials: passes.iter().map(|p| p.counts.signed_trials).sum(),
        no_move_candidates: passes.iter().map(|p| p.counts.no_move_candidates).sum(),
        accepted_rows: passes.iter().map(|p| p.counts.accepted_rows).sum(),
        rejected_rows: passes.iter().map(|p| p.counts.rejected_rows).sum(),
        transitions: passes.iter().map(|p| p.counts.transitions).sum(),
        stalled_presentations: passes.iter().map(|p| p.counts.stalled_presentations).sum(),
    };
    assert!(totals.signed_trials <= 6_324_224 && totals.transitions <= 8_192);
    assert_eq!(totals.accepted_rows + totals.rejected_rows, 8_192);
    assert_eq!(totals.no_move_candidates, 8_192);
    let final_measurement = &passes.last().expect("16 fixed passes").after;
    assert_eq!(final_measurement.presentations, 64);
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    let encoded = serde_json::to_string(&json!({
        "schema": 1, "kind": "fixed_predictor_reachability", "seed": 7,
        "geometry": predictor.geometry, "kernel": BASELINE_KERNEL,
        "frozen_pairs_sha256": PAIR_HASH, "initial_predictor_sha256": INITIAL_HASH,
        "frozen_initial_route": route, "rule": "unchanged_adjacent_coordinate",
        "historical_prefix": {
            "source": "f-predictor-update-drive/DIAGNOSTIC-OBSERVATIONS.json",
            "observations_sha256": OBSERVATIONS_HASH, "predictor_sha256_after_four": PREFIX_HASH,
            "reconstructed_presentations": 4, "resumed_weights": false,
        },
        "executed_presentations": 64, "executed_predictor_train_calls": 0,
        "executed_tile_commits": 0, "executed_embedding_commits": 0,
        "executed_local_opportunities": 0, "executed_microchecks": 0,
        "measurement_checkpoints": 17, "production_correction_calls": 68,
        "predictor_matrix_forwards": 396, "totals": totals,
        "final_gate_passed": final_measurement.gate_passed,
        "gate": "at presentation 64 every pair has delivered target_dot > 0 and mse < matched zero_mse",
        "initial": initial, "passes": passes,
        "scope": "fixed fitting-pair reachability only; no production adoption, local-update or assistant-quality result",
    }))?;
    assert!(encoded.len() + LABEL.len() < 32 * 1024);
    println!("{LABEL}{encoded}");
    Ok(())
}

fn run_pass(current: &mut Predictor, pairs: &[probe::Pair], pass: usize) -> Result<Pass> {
    let before_sha256 = probe::predictor_identity(current);
    let mut counts = Counts::default();
    for pair in pairs {
        let before = probe::predictor_identity(current);
        let version = current.version;
        let update = rules::apply(rules::Rule::Adjacent, current, pair)?;
        assert_eq!(update.production_train_calls, 0);
        assert_eq!(update.accepted_rows + update.rejected_rows, 128);
        assert_eq!(update.no_move_candidates, 128);
        assert_eq!(update.changed_trits, update.accepted_rows);
        assert!(update.signed_candidate_trials <= 98_816);
        assert_eq!(
            current.version,
            version + u64::from(update.changed_trits > 0)
        );
        if update.changed_trits == 0 {
            assert_eq!(probe::predictor_identity(current), before);
            counts.stalled_presentations += 1;
        }
        counts.signed_trials += update.signed_candidate_trials;
        counts.no_move_candidates += update.no_move_candidates;
        counts.accepted_rows += update.accepted_rows;
        counts.rejected_rows += update.rejected_rows;
        counts.transitions += update.changed_trits;
        assert_eq!(probe::pairs_identity(pairs), PAIR_HASH);
    }
    assert!(counts.transitions <= 512 && counts.signed_trials <= 395_264);
    Ok(Pass {
        pass,
        before_sha256,
        counts,
        after: measure(current, pairs, pass * 4)?,
    })
}

pub(super) fn measure(
    predictor: &Predictor,
    pairs: &[probe::Pair],
    presentations: usize,
) -> Result<Measurement> {
    let raw = probe::observe(predictor, pairs)?;
    let delivered: Vec<_> = pairs
        .iter()
        .map(|pair| delivered_metrics(predictor, pair))
        .collect::<Result<_>>()?;
    let mut canonical_trit_counts = [0; 3];
    assert_eq!(predictor.matrix.len(), 49_408);
    for index in 0..predictor.matrix.len() {
        let value = predictor.matrix.value(index)?;
        assert!((-1..=1).contains(&value));
        canonical_trit_counts[(value + 1) as usize] += 1;
    }
    assert_eq!(canonical_trit_counts.iter().sum::<usize>(), 49_408);
    assert_eq!(probe::predictor_identity(predictor), raw.predictor_sha256);
    assert_eq!(probe::pairs_identity(pairs), PAIR_HASH);
    Ok(Measurement {
        presentations,
        predictor_sha256: raw.predictor_sha256,
        version: raw.version,
        canonical_trit_counts,
        raw_all_pair_mean_mse: raw.all_pair_coordinate_mean_mse,
        raw_pair_mean_mse: raw.pairs.iter().map(|p| p.coordinate_mean_mse).collect(),
        gate_passed: delivered.iter().all(|p| p.gate_passed),
        delivered,
    })
}

fn delivered_metrics(predictor: &Predictor, pair: &probe::Pair) -> Result<Delivered> {
    let correction = predictor.correction(&pair.features)?;
    assert_eq!((correction.len(), pair.target.len()), (128, 128));
    let (mut error, mut dot, mut p2, mut t2) = (0.0, 0.0, 0.0, 0.0);
    for (&prediction, &target) in correction.iter().zip(&pair.target) {
        assert!(prediction.is_finite() && target.is_finite());
        let (p, t) = (f64::from(prediction), f64::from(target));
        error += (p - t).powi(2);
        dot += p * t;
        p2 += p * p;
        t2 += t * t;
    }
    let (mse, zero_mse) = (error / 128.0, t2 / 128.0);
    let (correction_norm, target_norm) = (p2.sqrt(), t2.sqrt());
    assert!(mse.is_finite() && dot.is_finite() && zero_mse.is_finite());
    assert!((0.0..=1.000002).contains(&correction_norm));
    assert!((0.0..=1.000002).contains(&target_norm));
    let cosine = if correction_norm == 0.0 || target_norm == 0.0 {
        None
    } else {
        let value = dot / (correction_norm * target_norm);
        assert!(value.is_finite() && value.abs() <= 1.000000000001);
        Some(value)
    };
    Ok(Delivered {
        mse,
        zero_mse,
        target_dot: dot,
        target_norm,
        correction_norm,
        cosine,
        gate_passed: dot > 0.0 && mse < zero_mse,
    })
}
