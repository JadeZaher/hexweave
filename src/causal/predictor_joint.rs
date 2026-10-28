use super::norm;
use super::predictor::Predictor;
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Context, Result};

pub(super) struct Choice {
    pub movement: Option<(usize, i8)>,
    pub signed_candidates: usize,
}

pub(super) struct Update {
    pub pair_mse: Vec<f32>,
    pub transitions: usize,
}

pub(super) fn choose(weights: &[i8], features: &[&[f32]], targets: &[f32]) -> Result<Choice> {
    ensure!(
        !weights.is_empty() && (1..=4).contains(&features.len()),
        "joint row size"
    );
    ensure!(weights.iter().all(|w| (-1..=1).contains(w)), "joint trit");
    ensure!(
        features.len() == targets.len()
            && features
                .iter()
                .all(|f| f.len() == weights.len() && f.iter().all(|x| x.is_finite()))
            && targets.iter().all(|t| t.is_finite()),
        "joint input shape or finiteness"
    );
    let scale = (weights.len() as f64).sqrt().recip();
    let mut base = [0.0; 4];
    for (pair, input) in features.iter().enumerate() {
        base[pair] = weights
            .iter()
            .zip(*input)
            .map(|(&w, &f)| f64::from(w) * f64::from(f))
            .sum::<f64>()
            * scale;
    }
    ensure!(base.iter().all(|x| x.is_finite()), "joint base finiteness");
    let error = |projections: [f64; 4]| {
        projections
            .iter()
            .zip(targets)
            .map(|(&z, &t)| (z.tanh() - f64::from(t)).powi(2))
            .sum::<f64>()
            / targets.len() as f64
    };
    let mut best = error(base);
    ensure!(best.is_finite(), "joint baseline finiteness");
    let mut choice = Choice {
        movement: None,
        signed_candidates: 0,
    };
    for (column, &weight) in weights.iter().enumerate() {
        for direction in [-1, 1] {
            if !(-1..=1).contains(&(weight + direction)) {
                continue;
            }
            choice.signed_candidates += 1;
            let mut candidate = base;
            for (pair, input) in features.iter().enumerate() {
                candidate[pair] =
                    base[pair] + f64::from(direction) * f64::from(input[column]) * scale;
            }
            let loss = error(candidate);
            ensure!(loss.is_finite(), "joint candidate finiteness");
            if loss < best {
                best = loss;
                choice.movement = Some((column, direction));
            }
        }
    }
    Ok(choice)
}

/// Fit one bounded frozen batch; see AGENTS.md for v3 arithmetic and lifetimes.
pub(super) fn apply(current: &mut Predictor, pairs: &[(&[f32], &[f32])]) -> Result<Update> {
    ensure!(
        (1..=4).contains(&pairs.len()),
        "joint batch must contain 1-4 pairs"
    );
    current.geometry.validate()?;
    let (width, features) = (current.geometry.width, current.geometry.features());
    ensure!(
        current.matrix.rows() == width && current.matrix.cols() == features,
        "joint matrix shape"
    );
    let mut inputs = Vec::with_capacity(pairs.len());
    let mut targets = Vec::with_capacity(pairs.len());
    for &(input, correction) in pairs {
        norm::vector(input, features)?;
        norm::vector(correction, width)?;
        inputs.push(input);
        targets.push(norm::unit_clip(correction)?);
    }
    let mut values = (0..current.matrix.len())
        .map(|i| current.matrix.value(i))
        .collect::<Result<Vec<_>>>()?;
    let (mut signed_candidates, mut proposed_changes) = (0, 0);
    for (row, values) in values.chunks_exact_mut(features).enumerate() {
        let mut row_targets = [0.0; 4];
        for (target, pair) in row_targets.iter_mut().zip(&targets) {
            *target = pair[row];
        }
        let choice = choose(values, &inputs, &row_targets[..pairs.len()])?;
        signed_candidates += choice.signed_candidates;
        if let Some((column, direction)) = choice.movement {
            values[column] += direction;
            proposed_changes += 1;
        }
    }
    ensure!(
        signed_candidates <= width * features * 2 && proposed_changes <= width,
        "joint work bound"
    );
    let matrix = TernaryMatrix::from_values(width, features, &values)?;
    drop(values);
    let proposal = Predictor {
        geometry: current.geometry,
        version: current.version,
        matrix,
    };
    guard(current, &inputs, &targets, proposal, proposed_changes)
}

fn predictions(predictor: &Predictor, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    inputs
        .iter()
        .map(|input| predictor.predict(input))
        .collect()
}

fn row_error(predictions: &[Vec<f32>], targets: &[Vec<f32>], row: usize) -> f64 {
    predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (f64::from(prediction[row]) - f64::from(target[row])).powi(2))
        .sum::<f64>()
        / targets.len() as f64
}

fn guard(
    current: &mut Predictor,
    inputs: &[&[f32]],
    targets: &[Vec<f32>],
    proposal: Predictor,
    proposed_changes: usize,
) -> Result<Update> {
    let width = current.geometry.width;
    let before = predictions(current, inputs)?;
    let proposed = predictions(&proposal, inputs)?;
    let old_errors: Vec<_> = (0..width)
        .map(|row| row_error(&before, targets, row))
        .collect();
    let mut accepted = Vec::with_capacity(width);
    for (row, &old) in old_errors.iter().enumerate() {
        let next = row_error(&proposed, targets, row);
        ensure!(
            next.is_finite() && old.is_finite(),
            "joint guard finiteness"
        );
        accepted.push(next < old);
    }
    let (matrix, transitions) = merge(&current.matrix, &proposal.matrix, &accepted)?;
    let accepted_rows = accepted.iter().filter(|&&x| x).count();
    ensure!(
        transitions == accepted_rows && transitions <= proposed_changes,
        "joint transition accounting"
    );
    drop(proposal);
    let next = Predictor {
        geometry: current.geometry,
        version: current
            .version
            .checked_add(u64::from(transitions > 0))
            .context("joint version overflow")?,
        matrix,
    };
    let actual = predictions(&next, inputs)?;
    for (row, &accept) in accepted.iter().enumerate() {
        for (pair, prediction) in actual.iter().enumerate() {
            let expected = if accept {
                proposed[pair][row]
            } else {
                before[pair][row]
            };
            ensure!(
                prediction[row].to_bits() == expected.to_bits(),
                "joint packed row mismatch"
            );
        }
        ensure!(
            row_error(&actual, targets, row) <= old_errors[row],
            "joint guard regression"
        );
    }
    let pair_mse = metrics(&before, targets, width)?;
    *current = next;
    Ok(Update {
        pair_mse,
        transitions,
    })
}

fn merge(
    original: &TernaryMatrix,
    proposal: &TernaryMatrix,
    accepted: &[bool],
) -> Result<(TernaryMatrix, usize)> {
    ensure!(
        original.rows() == accepted.len()
            && original.rows() == proposal.rows()
            && original.cols() == proposal.cols(),
        "joint row merge shape"
    );
    let mut values = Vec::with_capacity(original.len());
    let mut changed = 0;
    for index in 0..original.len() {
        let before = original.value(index)?;
        let after = if accepted[index / original.cols()] {
            proposal.value(index)?
        } else {
            before
        };
        ensure!((after - before).abs() <= 1, "nonadjacent joint movement");
        changed += usize::from(before != after);
        values.push(after);
    }
    Ok((
        TernaryMatrix::from_values(original.rows(), original.cols(), &values)?,
        changed,
    ))
}

fn metrics(before: &[Vec<f32>], targets: &[Vec<f32>], width: usize) -> Result<Vec<f32>> {
    let values: Vec<f32> = before
        .iter()
        .zip(targets)
        .map(|(prediction, target)| {
            prediction
                .iter()
                .zip(target)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / width as f32
        })
        .collect();
    ensure!(
        values.iter().all(|x| x.is_finite() && *x >= 0.0),
        "joint metric finiteness"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    /// Three of four targets ask for +1, so the joint choice raises the trit (both moves counted).
    #[test]
    fn choice_aggregates_targets_across_pairs() -> anyhow::Result<()> {
        let feature = [1.0];
        let choice = super::choose(&[0], &[&feature[..]; 4], &[-1.0, 1.0, 1.0, 1.0])?;
        assert_eq!(choice.movement, Some((0, 1)));
        assert_eq!(choice.signed_candidates, 2);
        Ok(())
    }
}
