use super::{probe, reachability, Predictor, BASELINE_KERNEL, INITIAL_HASH, PAIR_HASH};
use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Value};

#[path = "predictor_joint_rules.rs"]
mod joint_rules;

const LABEL: &str = "PREDICTOR_JOINT_FIT ";
const HISTORICAL_HASH: &str = "8f970e14dc9590252f39101a999c6d6f2cc40b7caeede281d47f571cac104609";

#[derive(Default, Serialize)]
struct Counts {
    signed_candidates: usize,
    no_move_candidates: usize,
    accepted_rows: usize,
    rejected_rows: usize,
    transitions: usize,
    stalled_updates: usize,
    raw_pair_regression_updates: [usize; 4],
}

#[derive(Serialize)]
struct Block {
    first_joint_update: usize,
    before_sha256: String,
    counts: Counts,
    raw_pair_mse_delta: [f64; 4],
    after: Value,
}

#[test]
fn fixed_joint_predictor_fit() -> Result<()> {
    assert_eq!(crate::ternary::kernel_name()?, BASELINE_KERNEL);
    aggregation_check()?;
    let probe::Frozen {
        mut predictor,
        pairs,
        route,
    } = probe::freeze()?;
    assert_eq!(pairs.len(), 4);
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    assert_eq!(probe::predictor_identity(&predictor), INITIAL_HASH);
    assert_eq!(predictor.version, 0);
    let mut measured = reachability::measure(&predictor, &pairs, 0)?;
    let initial = checkpoint(&measured)?;
    let mut blocks = Vec::with_capacity(16);
    for block in 0..16 {
        let (report, after) = run_block(&mut predictor, &pairs, block, &measured)?;
        blocks.push(report);
        measured = after;
    }
    assert_eq!(blocks.len(), 16);
    assert_eq!(checkpoint(&measured)?, expected_final());
    let totals = Counts {
        signed_candidates: blocks.iter().map(|b| b.counts.signed_candidates).sum(),
        no_move_candidates: blocks.iter().map(|b| b.counts.no_move_candidates).sum(),
        accepted_rows: blocks.iter().map(|b| b.counts.accepted_rows).sum(),
        rejected_rows: blocks.iter().map(|b| b.counts.rejected_rows).sum(),
        transitions: blocks.iter().map(|b| b.counts.transitions).sum(),
        stalled_updates: blocks.iter().map(|b| b.counts.stalled_updates).sum(),
        raw_pair_regression_updates: std::array::from_fn(|i| {
            blocks
                .iter()
                .map(|b| b.counts.raw_pair_regression_updates[i])
                .sum()
        }),
    };
    assert!(totals.signed_candidates <= 6_324_224 && totals.transitions <= 8_192);
    assert_eq!(totals.accepted_rows + totals.rejected_rows, 8_192);
    assert_eq!(totals.no_move_candidates, 8_192);
    assert_eq!(probe::pairs_identity(&pairs), PAIR_HASH);
    let encoded = serde_json::to_string(&json!({
        "schema": 1, "kind": "fixed_joint_predictor_fit", "seed": 7,
        "geometry": predictor.geometry, "kernel": BASELINE_KERNEL,
        "frozen_pairs_sha256": PAIR_HASH, "initial_predictor_sha256": INITIAL_HASH,
        "frozen_initial_route": route, "rule": "joint_four_pair_raw_row_mean_adjacent",
        "executed_joint_updates": 64, "fitting_pair_contributions": 256,
        "candidate_pair_error_evaluations": totals.signed_candidates * 4,
        "row_base_projections": 32_768, "baseline_pair_error_evaluations": 32_768,
        "executed_predictor_train_calls": 0, "production_joint_calls": 64, "protocol": "causal-prefix-v3", "executed_tile_commits": 0,
        "executed_embedding_commits": 0, "executed_local_opportunities": 0,
        "measurement_checkpoints": 17, "production_correction_calls": 68,
        "predictor_matrix_forwards": 972, "totals": totals,
        "pure_aggregation_check": { "executed": 1, "signed_candidates": 2,
            "candidate_pair_error_evaluations": 8, "baseline_pair_error_evaluations": 4,
            "row_base_projections": 4, "predictor_forwards": 0 },
        "historical_online": { "source": "f-predictor-reachability-drive/DIAGNOSTIC-OBSERVATIONS.json",
            "observations_sha256": HISTORICAL_HASH, "executed_in_this_test": false,
            "online_presentations": 64, "comparison": "equal row-move cap, unequal compute and pair contributions" },
        "final_gate_passed": measured.gate_passed,
        "gate": "at joint update 64 every pair has delivered target_dot > 0 and mse < its matched zero_mse",
        "initial": initial, "blocks": blocks,
        "scope": "fixed joint raw objective; individual raw error may rise; delivered errors measured only at checkpoints; production-kernel parity; no campaign or quality adoption",
    }))?;
    assert!(encoded.len() + LABEL.len() < 32 * 1024);
    println!("{LABEL}{encoded}");
    Ok(())
}

fn checkpoint(measured: &reachability::Measurement) -> Result<Value> {
    let mut report = serde_json::to_value(measured)?;
    let object = report.as_object_mut().expect("measurement object");
    let ordinal = object.remove("presentations").expect("measurement ordinal");
    object.insert("joint_updates".to_owned(), ordinal);
    Ok(report)
}

fn run_block(
    current: &mut Predictor,
    pairs: &[probe::Pair],
    block: usize,
    before: &reachability::Measurement,
) -> Result<(Block, reachability::Measurement)> {
    let before_sha256 = probe::predictor_identity(current);
    assert_eq!(before_sha256, before.predictor_sha256);
    let mut counts = Counts::default();
    for _ in 0..4 {
        let hash = probe::predictor_identity(current);
        let version = current.version;
        let update = joint_rules::apply(current, pairs)?;
        assert!(update.transitions <= 128);
        assert_eq!(current.version, version + u64::from(update.transitions > 0));
        if update.transitions == 0 {
            assert_eq!(probe::predictor_identity(current), hash);
            counts.stalled_updates += 1;
        }
        counts.signed_candidates += update.signed_candidates;
        counts.no_move_candidates += 128;
        counts.accepted_rows += update.transitions;
        counts.rejected_rows += 128 - update.transitions;
        counts.transitions += update.transitions;
        for (count, regressed) in counts
            .raw_pair_regression_updates
            .iter_mut()
            .zip(update.raw_pair_regressed)
        {
            *count += usize::from(regressed);
        }
        assert_eq!(probe::pairs_identity(pairs), PAIR_HASH);
    }
    assert!(counts.transitions <= 512 && counts.signed_candidates <= 395_264);
    let after = reachability::measure(current, pairs, (block + 1) * 4)?;
    assert_eq!(
        after.version,
        before.version + (4 - counts.stalled_updates) as u64
    );
    let report = Block {
        first_joint_update: block * 4 + 1,
        before_sha256,
        counts,
        raw_pair_mse_delta: std::array::from_fn(|i| {
            after.raw_pair_mean_mse[i] - before.raw_pair_mean_mse[i]
        }),
        after: checkpoint(&after)?,
    };
    Ok((report, after))
}

fn aggregation_check() -> Result<()> {
    let feature = [1.0];
    let choice = joint_rules::choose(&[0], &[&feature[..]; 4], &[-1.0, 1.0, 1.0, 1.0])?;
    assert_eq!(choice.movement, Some((0, 1)));
    assert_eq!(choice.signed_candidates, 2);
    Ok(())
}

fn expected_final() -> Value {
    json!({
        "canonical_trit_counts": [
            16546,
            16206,
            16656
        ],
        "delivered": [
            {
                "correction_norm": 0.8538310345008419,
                "cosine": 0.9995851859639721,
                "gate_passed": true,
                "mse": 4.74328753310467e-06,
                "target_dot": 0.7270883035435458,
                "target_norm": 0.8519133244729481,
                "zero_mse": 0.005669971190738676
            },
            {
                "correction_norm": 0.8465011676156723,
                "cosine": 0.9995738135181356,
                "gate_passed": true,
                "mse": 4.772181020074167e-06,
                "target_dot": 0.7155853843526613,
                "target_norm": 0.8457052566356643,
                "zero_mse": 0.0055876357898530845
            },
            {
                "correction_norm": 0.8405904412581867,
                "cosine": 0.9996785719936397,
                "gate_passed": true,
                "mse": 3.551089204947749e-06,
                "target_dot": 0.7066529331861778,
                "target_norm": 0.8409328842755254,
                "zero_mse": 0.005524750905124642
            },
            {
                "correction_norm": 0.8398212684194262,
                "cosine": 0.9996099364138692,
                "gate_passed": true,
                "mse": 4.319990639130308e-06,
                "target_dot": 0.7033342585297064,
                "target_norm": 0.8378076825689814,
                "zero_mse": 0.0054837633825906805
            }
        ],
        "gate_passed": true,
        "joint_updates": 64,
        "predictor_sha256": "b0450a42b72059250e6b267aab77a7c89f0156e00ef7adbadf617c05981fe75f",
        "raw_all_pair_mean_mse": 4.346637099314223e-06,
        "raw_pair_mean_mse": [
            4.74328753310467e-06,
            4.772181020074167e-06,
            3.551089204947749e-06,
            4.319990639130308e-06
        ],
        "version": 39
    })
}
