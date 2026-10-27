use super::predictor::Predictor;
use super::predictor_commit_contracts as probe;
use crate::ternary::TernaryMatrix;
use anyhow::Result;
use serde_json::{json, Value};

#[path = "predictor_joint_contracts.rs"]
mod joint;
#[path = "predictor_reachability_contracts.rs"]
mod reachability;
#[path = "predictor_update_rules.rs"]
mod rules;

const LABEL: &str = "PREDICTOR_UPDATE_COMPARISON ";
const PAIR_HASH: &str = "8069a12be8ec4318c27672e376776b9a4a29dab8aca3ed958639fa0af3e33477";
const INITIAL_HASH: &str = "21da6187deb8b3779a04a374f82a333d44e1462cce22b7de44143509b99c2a39";
const BASELINE_HASH: &str = "73ed7b0207d397c4ae8857ddbf25807554da3a902cf12bd10cc46369bd76704b";
const BASELINE_KERNEL: &str = "avx2";
const BASELINE_ALL_AFTER: [f64; 4] = [
    0.9878539720446483,
    1.0207652131765597,
    0.9878555976776191,
    1.0206432379270922,
];

#[test]
fn fixed_predictor_update_comparison() -> Result<()> {
    assert_eq!(crate::ternary::kernel_name()?, BASELINE_KERNEL);
    microchecks()?;
    let probe::Frozen {
        predictor,
        pairs,
        route,
    } = probe::freeze()?;
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    assert_eq!(probe::predictor_identity(&predictor), INITIAL_HASH);
    assert_eq!(pairs.len(), 4);
    let dense = arm(&predictor, &pairs, rules::Rule::DenseGuard)?;
    let adjacent = arm(&predictor, &pairs, rules::Rule::Adjacent)?;
    assert_eq!(probe::predictor_identity(&predictor), INITIAL_HASH);
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    let encoded = serde_json::to_string(&json!({
        "schema": 1, "kind": "fixed_predictor_update_comparison", "seed": 7,
        "geometry": predictor.geometry, "kernel": crate::ternary::kernel_name()?,
        "frozen_pairs_sha256": PAIR_HASH, "initial_predictor_sha256": INITIAL_HASH,
        "frozen_initial_route": route, "arms": [dense, adjacent],
        "executed_fit_presentations": 8, "executed_predictor_train_calls": 4,
        "executed_tile_commits": 0, "executed_embedding_commits": 0,
        "executed_local_opportunities": 0, "executed_baseline_train_calls": 0,
        "pure_nontraining_microchecks": ["no_move_tie", "lowest_column_tie", "nonfinite_rejection", "packed_row_merge"],
        "microcheck_legal_signed_trials": 8,
        "historical_baseline": {
            "source": "f-predictor-commit-drive/DIAGNOSTIC-OBSERVATIONS.json",
            "observations_sha256": BASELINE_HASH, "executed_in_this_test": false,
            "kernel": BASELINE_KERNEL,
            "all_pair_mean_mse_after_each_commit": BASELINE_ALL_AFTER,
        },
        "scope": "test-only update candidates, current-pair acceptance is not aggregate fitting progress or assistant quality",
    }))?;
    assert!(encoded.len() + LABEL.len() < 32 * 1024);
    println!("{LABEL}{encoded}");
    Ok(())
}

fn arm(initial: &Predictor, pairs: &[probe::Pair], rule: rules::Rule) -> Result<Value> {
    let mut current = rules::copy(initial)?;
    assert_eq!(probe::predictor_identity(&current), INITIAL_HASH);
    let mut steps = Vec::with_capacity(4);
    let (mut calls, mut trials, mut changes, mut stalls) = (0, 0, 0, 0);
    let (mut start, mut finish) = (0.0, 0.0);
    for (index, pair) in pairs.iter().enumerate() {
        let before = probe::observe(&current, pairs)?;
        if index == 0 {
            start = before.all_pair_coordinate_mean_mse;
        }
        let update = rules::apply(rule, &mut current, pair)?;
        let after = probe::observe(&current, pairs)?;
        assert_eq!(probe::pairs_identity(pairs), PAIR_HASH);
        assert_eq!(
            after.version,
            before.version + u64::from(update.changed_trits > 0)
        );
        assert_eq!(update.accepted_rows + update.rejected_rows, 128);
        if update.changed_trits == 0 {
            assert_eq!(before.predictor_sha256, after.predictor_sha256);
        }
        assert!(after.pairs[index].coordinate_mean_mse <= before.pairs[index].coordinate_mean_mse);
        calls += update.production_train_calls;
        trials += update.signed_candidate_trials;
        changes += update.changed_trits;
        stalls += usize::from(update.changed_trits == 0);
        finish = after.all_pair_coordinate_mean_mse;
        steps.push(json!({
            "presentation": index + 1, "frozen_pair_index": index,
            "same_pair_mse_before": before.pairs[index].coordinate_mean_mse,
            "same_pair_mse_after": after.pairs[index].coordinate_mean_mse,
            "same_pair_mse_delta": after.pairs[index].coordinate_mean_mse - before.pairs[index].coordinate_mean_mse,
            "all_pair_mse_delta": after.all_pair_coordinate_mean_mse - before.all_pair_coordinate_mean_mse,
            "all_pair_mse_minus_historical_baseline": after.all_pair_coordinate_mean_mse - BASELINE_ALL_AFTER[index],
            "update": update, "before": before, "after": after,
        }));
    }
    let name = match rule {
        rules::Rule::DenseGuard => {
            assert_eq!((calls, trials), (4, 0));
            "dense_row_guard"
        }
        rules::Rule::Adjacent => {
            assert_eq!(calls, 0);
            assert!(trials <= 395_264 && changes <= 512);
            "adjacent_coordinate"
        }
    };
    assert_eq!(steps.len(), 4);
    Ok(json!({
        "rule": name, "presentations": steps, "production_train_calls": calls,
        "signed_candidate_trials": trials, "actual_changed_trits": changes,
        "zero_transition_presentations": stalls, "all_presentations_stalled": stalls == 4,
        "initial_all_pair_mse": start, "final_all_pair_mse": finish,
        "end_to_start_all_pair_mse_delta": finish - start,
        "aggregate_fit_worsened": finish > start,
    }))
}

fn microchecks() -> Result<()> {
    let tie = rules::choose(&[0, 0], &[0.0, 0.0], 0.0)?;
    assert_eq!((tie.movement, tie.trials), (None, 4));
    let first = rules::choose(&[0, 0], &[1.0, 1.0], 1.0)?;
    assert_eq!((first.movement, first.trials), (Some((0, 1)), 4));
    assert!(rules::choose(&[0], &[f32::NAN], 0.0).is_err());
    let original = [-1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0, -1];
    let proposed = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0];
    let expected = [0, 1, 0, 0, -1, 0, 1, 0, -1, 1, 0, -1];
    let a = TernaryMatrix::from_values(2, 6, &original)?;
    let b = TernaryMatrix::from_values(2, 6, &proposed)?;
    let (merged, changed) = rules::merge(&a, &b, &[true, false])?;
    assert_eq!(changed, 6);
    for (index, &value) in expected.iter().enumerate() {
        assert_eq!(merged.value(index)?, value);
        assert_eq!(a.value(index)?, original[index]);
        assert_eq!(b.value(index)?, proposed[index]);
    }
    Ok(())
}
