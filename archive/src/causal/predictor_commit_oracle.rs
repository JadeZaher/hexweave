use super::super::predictor::Predictor;
use super::Pair;
use anyhow::Result;
use serde::Serialize;

pub(super) const EPSILON: f64 = 1e-4;
const COORDINATES: [(usize, usize); 8] = [
    (0, 0),
    (1, 127),
    (32, 128),
    (63, 255),
    (64, 256),
    (95, 383),
    (126, 384),
    (127, 385),
];

pub(super) struct Before {
    weights: Vec<i8>,
    reference_derivative: Vec<f64>,
    production_derivative: Vec<f32>,
    pub error: f32,
    pub finite_differences: Vec<Difference>,
}

#[derive(Serialize)]
pub(super) struct Difference {
    row: usize,
    column: usize,
    analytic_f64: f64,
    central_difference_f64: f64,
    absolute_error: f64,
}

#[derive(Serialize)]
pub(super) struct Displacement {
    pub changed_trits: usize,
    increasing_trits: usize,
    decreasing_trits: usize,
    blocked_nonzero_gradient_coordinates: usize,
    zero_gradient_coordinates_f32: usize,
    zero_derivative_rows_f32: usize,
    gradient_dot_displacement_f64_reference: f64,
    gradient_dot_displacement_f32_rule: f64,
    reference_gradient_l2: f64,
    maximum_reference_vs_f32_gradient_difference: f64,
}

impl Before {
    pub fn capture(predictor: &Predictor, pair: &Pair) -> Result<Self> {
        let width = predictor.geometry.width;
        let features = predictor.geometry.features();
        assert_eq!((width, features), (128, 386));
        let weights = (0..predictor.matrix.len())
            .map(|i| predictor.matrix.value(i))
            .collect::<Result<Vec<_>>>()?;
        let prediction = predictor.predict(&pair.features)?;
        let mut reference_derivative = Vec::with_capacity(width);
        for (row, (&prediction, &target)) in prediction.iter().zip(&pair.target).enumerate() {
            let projection = reference_projection(
                &weights[row * features..(row + 1) * features],
                &pair.features,
                None,
            );
            let reference_prediction = projection.tanh();
            assert!((reference_prediction - f64::from(prediction)).abs() <= 2e-5);
            reference_derivative.push(
                2.0 / width as f64
                    * (reference_prediction - f64::from(target))
                    * (1.0 - reference_prediction * reference_prediction),
            );
        }
        let production_derivative = prediction
            .iter()
            .zip(&pair.target)
            .map(|(p, t)| 2.0 / width as f32 * (p - t) * (1.0 - p * p))
            .collect();
        let error = prediction
            .iter()
            .zip(&pair.target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / width as f32;
        let finite_differences = differences(&weights, pair, &reference_derivative);
        Ok(Self {
            weights,
            reference_derivative,
            production_derivative,
            error,
            finite_differences,
        })
    }

    pub fn displacement(&self, predictor: &Predictor, pair: &Pair) -> Result<Displacement> {
        let features = pair.features.len();
        let scale32 = (features as f32).sqrt().recip();
        let scale64 = (features as f64).sqrt().recip();
        let mut result = Displacement {
            changed_trits: 0,
            increasing_trits: 0,
            decreasing_trits: 0,
            blocked_nonzero_gradient_coordinates: 0,
            zero_gradient_coordinates_f32: 0,
            zero_derivative_rows_f32: self
                .production_derivative
                .iter()
                .filter(|&&d| d == 0.0)
                .count(),
            gradient_dot_displacement_f64_reference: 0.0,
            gradient_dot_displacement_f32_rule: 0.0,
            reference_gradient_l2: 0.0,
            maximum_reference_vs_f32_gradient_difference: 0.0,
        };
        for (index, &before) in self.weights.iter().enumerate() {
            let (row, column) = (index / features, index % features);
            let gradient32 = (self.production_derivative[row] * scale32) * pair.features[column];
            let gradient64 =
                self.reference_derivative[row] * f64::from(pair.features[column]) * scale64;
            assert!(gradient32.is_finite() && gradient64.is_finite());
            let direction = if gradient32 > 0.0 {
                -1
            } else if gradient32 < 0.0 {
                1
            } else {
                0
            };
            let after = predictor.matrix.value(index)?;
            assert_eq!(
                after,
                (before + direction).clamp(-1, 1),
                "coordinate {index}"
            );
            let delta = after - before;
            assert!(delta.abs() <= 1);
            result.changed_trits += usize::from(delta != 0);
            result.increasing_trits += usize::from(delta > 0);
            result.decreasing_trits += usize::from(delta < 0);
            result.blocked_nonzero_gradient_coordinates +=
                usize::from(gradient32 != 0.0 && delta == 0);
            result.zero_gradient_coordinates_f32 += usize::from(gradient32 == 0.0);
            result.gradient_dot_displacement_f64_reference += gradient64 * f64::from(delta);
            result.gradient_dot_displacement_f32_rule += f64::from(gradient32) * f64::from(delta);
            result.reference_gradient_l2 += gradient64 * gradient64;
            let difference = (gradient64 - f64::from(gradient32)).abs();
            assert!(difference <= 1e-7 + 1e-4 * gradient64.abs());
            result.maximum_reference_vs_f32_gradient_difference = result
                .maximum_reference_vs_f32_gradient_difference
                .max(difference);
        }
        result.reference_gradient_l2 = result.reference_gradient_l2.sqrt();
        assert_eq!(
            result.changed_trits
                + result.blocked_nonzero_gradient_coordinates
                + result.zero_gradient_coordinates_f32,
            self.weights.len()
        );
        assert!(result.gradient_dot_displacement_f64_reference.is_finite());
        assert!(result.gradient_dot_displacement_f32_rule.is_finite());
        assert!(result.reference_gradient_l2.is_finite());
        Ok(result)
    }
}

fn reference_projection(weights: &[i8], features: &[f32], perturb: Option<(usize, f64)>) -> f64 {
    weights
        .iter()
        .zip(features)
        .enumerate()
        .map(|(column, (&weight, &feature))| {
            let delta = perturb
                .filter(|&(index, _)| index == column)
                .map_or(0.0, |(_, delta)| delta);
            (f64::from(weight) + delta) * f64::from(feature)
        })
        .sum::<f64>()
        / (features.len() as f64).sqrt()
}

fn differences(weights: &[i8], pair: &Pair, derivative: &[f64]) -> Vec<Difference> {
    let features = pair.features.len();
    COORDINATES
        .iter()
        .map(|&(row, column)| {
            let row_weights = &weights[row * features..(row + 1) * features];
            let loss = |delta| {
                let prediction =
                    reference_projection(row_weights, &pair.features, Some((column, delta))).tanh();
                (prediction - f64::from(pair.target[row])).powi(2) / pair.target.len() as f64
            };
            let central_difference_f64 = (loss(EPSILON) - loss(-EPSILON)) / (2.0 * EPSILON);
            let analytic_f64 =
                derivative[row] * f64::from(pair.features[column]) / (features as f64).sqrt();
            let absolute_error = (central_difference_f64 - analytic_f64).abs();
            assert!(analytic_f64.is_finite() && central_difference_f64.is_finite());
            assert!(
                absolute_error <= 1e-10 + 1e-5 * analytic_f64.abs(),
                "row {row}, column {column}"
            );
            Difference {
                row,
                column,
                analytic_f64,
                central_difference_f64,
                absolute_error,
            }
        })
        .collect()
}
