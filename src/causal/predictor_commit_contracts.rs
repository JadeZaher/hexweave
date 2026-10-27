use super::geometry::{Geometry, TILES};
use super::predictor::Predictor;
use super::state::State;
use super::{credit, execution, immutable, norm};
use crate::topology::Mesh;
use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;
#[path = "predictor_commit_oracle.rs"]
mod oracle;

const PREFIX: &[u8] = b"Evidence: leaves are green.\nAnswer: ";
const LABEL: &str = "PREDICTOR_COMMIT_DIAGNOSTIC ";
const PAIRS: usize = 4;

pub(super) struct Pair {
    pub features: Vec<f32>,
    pub correction: Vec<f32>,
    pub target: Vec<f32>,
}

pub(super) struct Frozen {
    pub predictor: Predictor,
    pub pairs: Vec<Pair>,
    pub route: Vec<(usize, Option<usize>)>,
}

#[derive(Serialize)]
pub(super) struct PairMetrics {
    pub coordinate_mean_mse: f64,
    target_norm: f64,
    prediction_norm: f64,
    exact_saturated_coordinates: usize,
    projection_min: f32,
    projection_max: f32,
}

#[derive(Serialize)]
pub(super) struct Observation {
    pub predictor_sha256: String,
    pub version: u64,
    pub all_pair_coordinate_mean_mse: f64,
    pub pairs: Vec<PairMetrics>,
}

#[test]
fn fixed_predictor_commit_diagnostic() -> Result<()> {
    let Frozen {
        mut predictor,
        pairs,
        route,
    } = freeze()?;
    let frozen = pairs_identity(&pairs);
    let initial = predictor_identity(&predictor);
    let mut commits = Vec::with_capacity(PAIRS);
    for (index, pair) in pairs.iter().enumerate() {
        let before = observe(&predictor, &pairs)?;
        let oracle = oracle::Before::capture(&predictor, pair)?;
        let update = predictor.train(&pair.features, &pair.correction)?;
        assert_eq!(update.mean_squared_error.to_bits(), oracle.error.to_bits());
        let displacement = oracle.displacement(&predictor, pair)?;
        assert_eq!(update.transitions, displacement.changed_trits);
        assert_eq!(
            predictor.version,
            before.version + u64::from(update.transitions > 0)
        );
        let after = observe(&predictor, &pairs)?;
        assert_eq!(pairs_identity(&pairs), frozen);
        commits.push(json!({
            "commit": index + 1, "frozen_pair_index": index,
            "production_precommit_mse_f32": update.mean_squared_error,
            "same_pair_mse_before": before.pairs[index].coordinate_mean_mse,
            "same_pair_mse_after": after.pairs[index].coordinate_mean_mse,
            "same_pair_mse_delta": after.pairs[index].coordinate_mean_mse
                - before.pairs[index].coordinate_mean_mse,
            "all_pair_mse_delta": after.all_pair_coordinate_mean_mse
                - before.all_pair_coordinate_mean_mse,
            "before": before, "after": after,
            "continuous_reference_and_actual_displacement": displacement,
            "central_differences": oracle.finite_differences,
        }));
    }
    assert_eq!(commits.len(), PAIRS);
    let encoded = serde_json::to_string(&json!({
        "schema": 1, "kind": "fixed_predictor_commit_fixture", "seed": 7,
        "geometry": predictor.geometry, "kernel": crate::ternary::kernel_name()?,
        "prefix_utf8": std::str::from_utf8(PREFIX)?, "anchor_target_byte": 123,
        "frozen_initial_route": route, "frozen_pairs_sha256": frozen,
        "initial_predictor_sha256": initial, "commits": commits,
        "executed_predictor_train_calls": PAIRS,
        "executed_tile_commits": 0, "executed_embedding_commits": 0,
        "executed_local_opportunities": 0,
        "finite_difference_epsilon": oracle::EPSILON,
        "comparison": "same ordered initial pairs; correction clipped once before train and again inside train",
        "scope": "conditional continuous derivative and actual finite ternary steps; no learning-gain or quality assertion",
    }))?;
    assert!(encoded.len() + LABEL.len() < 32 * 1024);
    println!("{LABEL}{encoded}");
    Ok(())
}

pub(super) fn freeze() -> Result<Frozen> {
    let geometry = Geometry::candidate();
    assert_eq!((geometry.width, geometry.features()), (128, 386));
    let source = fixture::Source::with_bytes(PREFIX, b"{\"}");
    let state = State::with_cursor(geometry, 7, source.cursor())?;
    let mut store = fixture::Store::with_geometry(geometry, 7)?;
    let before = immutable::identity(&state, &mut store)?;
    let prefix = state.shared.prefix(PREFIX)?;
    let trace = execution::forward_path(
        &mut store,
        geometry,
        &prefix,
        vec![0.0; geometry.width],
        &mut Some(7),
        state.tile_versions,
    )?;
    let route: Vec<_> = trace.frames.iter().map(|f| (f.tile_id, f.action)).collect();
    assert_eq!(route, [(3, Some(3)), (4, Some(0)), (1, Some(2)), (4, None)]);
    let gathered = credit::gather(&state, &mut store, &prefix, &trace, b'{')?;
    let mesh = Mesh::new(1)?;
    let mut pairs = Vec::with_capacity(PAIRS);
    for (index, frame) in trace.frames.iter().enumerate() {
        let output = trace
            .frames
            .get(index + 1)
            .map_or(trace.output.as_slice(), |next| next.input.as_slice());
        let point = mesh.coordinates()[frame.tile_id];
        let correction = norm::unit_clip(&gathered.corrections[index])?;
        pairs.push(Pair {
            features: state.predictor.features(
                &frame.input,
                output,
                &trace.history,
                point.q,
                point.r,
            )?,
            target: norm::unit_clip(&correction)?,
            correction,
        });
    }
    assert_eq!(pairs.len(), PAIRS);
    assert_eq!(before, immutable::identity(&state, &mut store)?);
    assert_eq!(store.updates, [0; TILES]);
    assert_eq!((source.target_reads.get(), source.advances.get()), (0, 0));
    Ok(Frozen {
        predictor: state.predictor,
        pairs,
        route,
    })
}

pub(super) fn observe(predictor: &Predictor, pairs: &[Pair]) -> Result<Observation> {
    let before = predictor_identity(predictor);
    let mut metrics = Vec::with_capacity(PAIRS);
    for pair in pairs {
        let prediction = predictor.predict(&pair.features)?;
        let projection = norm::project(
            &predictor.matrix,
            &pair.features,
            (predictor.geometry.features() as f32).sqrt().recip(),
        )?;
        assert!(prediction
            .iter()
            .zip(&projection)
            .all(|(p, z)| *p == z.tanh()));
        let mse = prediction
            .iter()
            .zip(&pair.target)
            .map(|(&p, &t)| (f64::from(p) - f64::from(t)).powi(2))
            .sum::<f64>()
            / predictor.geometry.width as f64;
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let target_norm = norm(&pair.target);
        let prediction_norm = norm(&prediction);
        assert!(target_norm <= 1.000002);
        assert!(prediction_norm <= (predictor.geometry.width as f64).sqrt());
        assert!(mse.is_finite() && (0.0..=4.00001).contains(&mse));
        norm::finite(&projection)?;
        metrics.push(PairMetrics {
            coordinate_mean_mse: mse,
            target_norm,
            prediction_norm,
            exact_saturated_coordinates: prediction.iter().filter(|p| p.abs() == 1.0).count(),
            projection_min: projection.iter().copied().fold(f32::INFINITY, f32::min),
            projection_max: projection.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        });
    }
    assert_eq!(before, predictor_identity(predictor));
    Ok(Observation {
        predictor_sha256: before,
        version: predictor.version,
        all_pair_coordinate_mean_mse: metrics.iter().map(|m| m.coordinate_mean_mse).sum::<f64>()
            / PAIRS as f64,
        pairs: metrics,
    })
}

pub(super) fn pairs_identity(pairs: &[Pair]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"fixed-predictor-pairs-v1");
    hash.update((pairs.len() as u64).to_le_bytes());
    for pair in pairs {
        for values in [&pair.features, &pair.correction, &pair.target] {
            hash.update((values.len() as u64).to_le_bytes());
            for value in values {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
    }
    format!("{:x}", hash.finalize())
}

pub(super) fn predictor_identity(predictor: &Predictor) -> String {
    let mut hash = Sha256::new();
    hash.update(b"fixed-predictor-state-v1");
    hash.update(predictor.version.to_le_bytes());
    hash.update(predictor.packed());
    format!("{:x}", hash.finalize())
}
