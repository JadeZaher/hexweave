use super::super::geometry::EVIDENCE_RULE;
use super::super::predictor::Predictor;
use super::probe::Pair;
use crate::ternary::TernaryMatrix;
use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Clone, Copy)]
pub(super) enum Rule {
    DenseGuard,
    Adjacent,
}

#[derive(Default, Serialize)]
pub(super) struct Update {
    pub production_train_calls: usize,
    pub signed_candidate_trials: usize,
    pub no_move_candidates: usize,
    reference_selected_rows: Option<usize>,
    proposed_changed_trits: usize,
    pub accepted_rows: usize,
    pub rejected_rows: usize,
    pub changed_trits: usize,
}

pub(super) struct Choice {
    pub movement: Option<(usize, i8)>,
    pub trials: usize,
}

pub(super) fn copy(predictor: &Predictor) -> Result<Predictor> {
    Predictor::from_packed(
        predictor.geometry,
        predictor.version,
        predictor.packed(),
        EVIDENCE_RULE,
    )
}

pub(super) fn apply(rule: Rule, current: &mut Predictor, pair: &Pair) -> Result<Update> {
    match rule {
        Rule::DenseGuard => {
            let mut proposal = copy(current)?;
            let trained = proposal.train(&pair.features, &pair.correction)?;
            guard(
                current,
                pair,
                proposal,
                Update {
                    production_train_calls: 1,
                    proposed_changed_trits: trained.transitions,
                    ..Update::default()
                },
            )
        }
        Rule::Adjacent => adjacent(current, pair),
    }
}

fn adjacent(current: &mut Predictor, pair: &Pair) -> Result<Update> {
    let width = current.geometry.width;
    let features = current.geometry.features();
    let mut values = (0..current.matrix.len())
        .map(|i| current.matrix.value(i))
        .collect::<Result<Vec<_>>>()?;
    let mut update = Update {
        no_move_candidates: width,
        reference_selected_rows: Some(0),
        ..Update::default()
    };
    for (row, values) in values.chunks_exact_mut(features).enumerate() {
        let choice = choose(values, &pair.features, pair.target[row])?;
        update.signed_candidate_trials += choice.trials;
        if let Some((column, direction)) = choice.movement {
            values[column] += direction;
            update.proposed_changed_trits += 1;
        }
    }
    update.reference_selected_rows = Some(update.proposed_changed_trits);
    assert!(update.signed_candidate_trials <= width * features * 2);
    let matrix = TernaryMatrix::from_values(width, features, &values)?;
    drop(values);
    let proposal = Predictor {
        geometry: current.geometry,
        version: current.version,
        matrix,
    };
    guard(current, pair, proposal, update)
}

pub(super) fn choose(weights: &[i8], features: &[f32], target: f32) -> Result<Choice> {
    ensure!(
        !weights.is_empty() && weights.len() == features.len(),
        "candidate row shape"
    );
    ensure!(
        weights.iter().all(|w| (-1..=1).contains(w)),
        "candidate trit"
    );
    ensure!(
        features.iter().all(|f| f.is_finite()) && target.is_finite(),
        "candidate nonfinite input"
    );
    let scale = (features.len() as f64).sqrt().recip();
    let base = weights
        .iter()
        .zip(features)
        .map(|(&w, &f)| f64::from(w) * f64::from(f))
        .sum::<f64>()
        * scale;
    let error = |z: f64| (z.tanh() - f64::from(target)).powi(2);
    let mut best = error(base);
    ensure!(
        base.is_finite() && best.is_finite(),
        "candidate nonfinite base"
    );
    let mut choice = Choice {
        movement: None,
        trials: 0,
    };
    for (column, (&weight, &feature)) in weights.iter().zip(features).enumerate() {
        for direction in [-1, 1] {
            if !(-1..=1).contains(&(weight + direction)) {
                continue;
            }
            choice.trials += 1;
            let loss = error(base + f64::from(direction) * f64::from(feature) * scale);
            ensure!(loss.is_finite(), "candidate nonfinite loss");
            if loss < best {
                best = loss;
                choice.movement = Some((column, direction));
            }
        }
    }
    Ok(choice)
}

fn guard(
    current: &mut Predictor,
    pair: &Pair,
    proposal: Predictor,
    mut update: Update,
) -> Result<Update> {
    let next_version = current
        .version
        .checked_add(1)
        .expect("bounded fixture version");
    let before = current.predict(&pair.features)?;
    let proposed = proposal.predict(&pair.features)?;
    let error = |p: f32, t: f32| (f64::from(p) - f64::from(t)).powi(2);
    let accepted: Vec<_> = before
        .iter()
        .zip(&proposed)
        .zip(&pair.target)
        .map(|((&old, &new), &target)| error(new, target) < error(old, target))
        .collect();
    update.accepted_rows = accepted.iter().filter(|&&accept| accept).count();
    update.rejected_rows = accepted.len() - update.accepted_rows;
    let proposed_changes = (0..current.matrix.len()).try_fold(0, |count, i| {
        Ok::<_, anyhow::Error>(
            count + usize::from(current.matrix.value(i)? != proposal.matrix.value(i)?),
        )
    })?;
    assert_eq!(proposed_changes, update.proposed_changed_trits);
    let (matrix, changed) = merge(&current.matrix, &proposal.matrix, &accepted)?;
    update.changed_trits = changed;
    assert!(changed <= proposed_changes);
    assert_eq!(changed == 0, update.accepted_rows == 0);
    drop(proposal);
    let next = Predictor {
        geometry: current.geometry,
        version: if changed > 0 {
            next_version
        } else {
            current.version
        },
        matrix,
    };
    let actual = next.predict(&pair.features)?;
    for (row, &accept) in accepted.iter().enumerate() {
        let expected = if accept { proposed[row] } else { before[row] };
        assert_eq!(
            actual[row].to_bits(),
            expected.to_bits(),
            "packed row {row}"
        );
        assert!(error(actual[row], pair.target[row]) <= error(before[row], pair.target[row]));
    }
    *current = next;
    Ok(update)
}

pub(super) fn merge(
    original: &TernaryMatrix,
    proposal: &TernaryMatrix,
    accepted: &[bool],
) -> Result<(TernaryMatrix, usize)> {
    ensure!(
        original.rows() == accepted.len()
            && original.rows() == proposal.rows()
            && original.cols() == proposal.cols(),
        "row merge shape"
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
        ensure!((after - before).abs() <= 1, "nonadjacent proposal movement");
        changed += usize::from(before != after);
        values.push(after);
    }
    Ok((
        TernaryMatrix::from_values(original.rows(), original.cols(), &values)?,
        changed,
    ))
}
