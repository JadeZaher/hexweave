//! Calibrated predictor learning, label conditioning, observed-only fitting and exact
//! continuation; see CONNECTED.md.

use super::connected_predictor::Predictor;
use super::connected_signal::{cosine, length, Kind, Record};
use super::geometry::Geometry;
use super::{norm, tuning};
use anyhow::Result;

const TILE: usize = 3;

/// Fixed unit target direction `u`.
fn direction(width: usize) -> Vec<f32> {
    let values: Vec<f32> = (0..width).map(|j| ((j * 5) % 7) as f32 - 2.5).collect();
    let norm = length(&values) as f32;
    values.iter().map(|value| value / norm).collect()
}

/// Hashed uniform draw in [-1, 1] for one step, stream and coordinate.
fn noise(step: u64, salt: u64, j: usize) -> f32 {
    (tuning::hash(step, &[salt, j as u64]) >> 40) as f32 / 16_777_215.0 * 2.0 - 1.0
}

/// Random RMS-1 label feature for `step`, drawn independently of the stream's other features.
fn label(width: usize, step: u64) -> Result<Vec<f32>> {
    let values: Vec<f32> = (0..width).map(|j| noise(step, 3, j)).collect();
    Ok(norm::rms(&values)?.values)
}

/// Fixed feature stream: a shared pattern plus hashed per-step variation; target `scale * u`.
fn stream(geometry: Geometry, step: u64, scale: f32, kind: Kind) -> Record {
    let d = geometry.width;
    Record {
        input: (0..d)
            .map(|j| (j as f32).cos() + 0.3 * noise(step, 1, j))
            .collect(),
        output: (0..d)
            .map(|j| (j as f32 + 1.0).sin() + 0.3 * noise(step, 2, j))
            .collect(),
        history: vec![0.0; d],
        correction: direction(d).iter().map(|u| scale * u).collect(),
        tile_id: TILE,
        revision: 0,
        shared_revision: 0,
        epoch: step,
        kind,
        lane: (kind == Kind::Predicted).then_some(0),
    }
}

#[test]
fn predictor_uses_only_observed_evidence_and_continues_exactly() -> Result<()> {
    let geometry = Geometry::fixture();
    let d = geometry.width;
    let mut predictor = Predictor::new(geometry, 7)?;
    assert_eq!(predictor.trust(TILE), 0.0);
    let initial = predictor.clone();
    // Unobserved evidence and malformed labels are refused before anything changes.
    let observed = stream(geometry, 0, 0.01, Kind::Observed);
    for (record, target) in [
        (stream(geometry, 0, 0.01, Kind::Predicted), label(d, 0)?),
        (stream(geometry, 0, 0.01, Kind::SelfGenerated), label(d, 0)?),
        (observed.clone(), vec![1.0; d - 1]),
        (observed, vec![f32::NAN; d]),
    ] {
        assert!(predictor.fit(&record, &target, 0.003).is_err());
        assert_eq!(predictor, initial);
    }
    for step in 0..40 {
        predictor.fit(
            &stream(geometry, step, 0.01, Kind::Observed),
            &label(d, step)?,
            0.003,
        )?;
    }
    // A no-evidence target still counts as a fit but changes neither weights nor calibration.
    let before = predictor.clone();
    predictor.fit(
        &stream(geometry, 40, 0.0, Kind::Observed),
        &label(d, 40)?,
        0.003,
    )?;
    assert_eq!(predictor.fits, before.fits + 1);
    let uncounted = Predictor {
        fits: before.fits,
        ..predictor.clone()
    };
    assert_eq!(uncounted, before);
    // (target energy lag, scale): a 1e-14-energy target still fits; a 1e6x lag clips |t_hat| to 4.
    for (lag, scale) in [(1.0, 1e-7), (1e-6, 0.01)] {
        let mut probe = predictor.clone();
        probe.calibration[TILE].target_energy *= lag;
        let old = probe.calibration[TILE];
        let record = stream(geometry, 40, scale, Kind::Observed);
        let metrics = probe.fit(&record, &label(d, 40)?, 0.003)?;
        let new = probe.calibration[TILE];
        // The running-mean identity recovers this fit's alignment p.t_hat and |p|^2.
        let n = new.observations as f64;
        let alignment = n * new.alignment - (n - 1.0) * old.alignment;
        let output_energy = n * new.prediction_energy - (n - 1.0) * old.prediction_energy;
        probe.validate(geometry)?;
        assert!(
            new.observations == old.observations + 1
                && new.target_energy != old.target_energy
                && alignment.abs() <= 4.0 * output_energy.sqrt() * 1.000001
                && (0.0..=1.0).contains(&probe.trust(TILE))
                && metrics.mse.is_finite(),
            "lag {lag}, scale {scale}: alignment {alignment}, |p|^2 {output_energy}"
        );
    }
    let (observed, target) = (stream(geometry, 41, 0.01, Kind::Observed), label(d, 41)?);
    let mut restored: Predictor = serde_json::from_slice(&serde_json::to_vec(&predictor)?)?;
    assert_eq!(predictor, restored);
    assert_eq!(
        predictor.fit(&observed, &target, 0.003)?,
        restored.fit(&observed, &target, 0.003)?
    );
    assert_eq!(predictor, restored);
    // Corrupt calibration: more Adam steps than fits, zero scale after evidence, nonfinite stats.
    let corruptions: [fn(&mut Predictor); 3] = [
        |damaged| damaged.fits = 0,
        |damaged| damaged.calibration[TILE].target_energy = 0.0,
        |damaged| damaged.calibration[TILE].alignment = f64::NAN,
    ];
    for corrupt in corruptions {
        let mut damaged = predictor.clone();
        corrupt(&mut damaged);
        assert!(damaged.validate(geometry).is_err());
    }
    Ok(())
}

#[test]
fn calibrated_predictor_trusts_consistent_targets_and_shrinks_alternating_ones() -> Result<()> {
    let geometry = Geometry::fixture();
    let direction = direction(geometry.width);
    // (odd-step target sign, trust range, prediction-norm range, minimum cosine with u)
    let cases = [
        (1.0, 0.5..=1.0, 0.005..=0.02, 0.9),
        (-1.0, 0.0..=0.1, 0.0..=0.002, -1.0),
    ];
    // One observed byte throughout: its label cannot explain the sign alternation.
    let target = label(geometry.width, 0)?;
    for (odd_sign, trust_range, norm_range, min_cosine) in cases {
        let mut predictor = Predictor::new(geometry, 7)?;
        for step in 0..400 {
            let sign = if step % 2 == 1 { odd_sign } else { 1.0 };
            let record = stream(geometry, step, 0.01 * sign, Kind::Observed);
            predictor.fit(&record, &target, 0.003)?;
        }
        let probe = stream(geometry, 400, 0.0, Kind::Observed);
        let features =
            predictor.features(&probe.input, &probe.output, &probe.history, &target, TILE)?;
        let prediction = predictor.predict(&features, TILE)?;
        let trust = predictor.trust(TILE);
        let (norm, cosine) = (length(&prediction), cosine(&prediction, &direction));
        assert!(
            trust_range.contains(&trust)
                && norm_range.contains(&norm)
                && cosine >= min_cosine
                && norm <= 0.01 * trust * 1.0001,
            "sign {odd_sign}: trust {trust}, norm {norm}, cosine {cosine}"
        );
    }
    Ok(())
}

#[test]
fn predictor_learns_label_conditioned_credit_only_through_the_target_feature() -> Result<()> {
    // Production width: the slope estimate of an uninformed predictor is tight at d = 128.
    let geometry = Geometry::candidate();
    let d = geometry.width;
    // (label visible to the predictor, final trust range, minimum mean prefit cosine of the tail)
    for (visible, trust_range, min_cosine) in [(true, 0.5..=1.0, 0.8), (false, 0.0..=0.1, -1.0)] {
        let mut predictor = Predictor::new(geometry, 7)?;
        let mut tail_cosine = 0.0;
        for step in 0..1000 {
            // The readout's label term: correction `-k * target` plus small independent noise.
            let target = label(d, step)?;
            let mut record = stream(geometry, step, 0.0, Kind::Observed);
            record.correction = (0..d)
                .map(|j| -0.01 * target[j] + 0.001 * noise(step, 4, j))
                .collect();
            let seen = if visible { target } else { vec![0.0; d] };
            let metrics = predictor.fit(&record, &seen, 0.003)?;
            if step >= 900 {
                tail_cosine += metrics.cosine / 100.0;
            }
        }
        let trust = predictor.trust(TILE);
        assert!(
            trust_range.contains(&trust) && tail_cosine >= min_cosine,
            "label visible {visible}: trust {trust}, mean prefit cosine {tail_cosine}"
        );
    }
    Ok(())
}
