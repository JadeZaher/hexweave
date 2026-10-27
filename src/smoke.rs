use std::{collections::BTreeMap, time::Instant};

use anyhow::{ensure, Result};
use candle_core::{Tensor, Var};
use serde_json::{json, Value};

use crate::{
    config::Config,
    model::{cross_entropy, HexModel},
};

type Snapshot = BTreeMap<String, Vec<f32>>;

/// Run a bounded CPU numerical fixture with an owned in-memory model.
pub fn run_smoke(steps: usize, seed: u64) -> Result<Value> {
    ensure!((1..=16).contains(&steps), "smoke steps must be in 1..=16");
    let started = Instant::now();
    let config = Config::smoke();
    let model = HexModel::new(config.clone(), seed)?;
    let parameters = model.parameters();
    let before = snapshot(&parameters)?;
    let initial_loss = evaluate(&model, config.context)?;
    let mut baseline = (config.vocab as f32).ln();
    let mut history = model.empty_history()?;
    let mut visits = 0;
    let mut learned_halts = 0;
    let mut losses = Vec::new();
    for step in 0..steps {
        let (inputs, targets) = example(config.context, (step % 4) as u32);
        let pass = model.forward(&inputs, &history, Some(seed.wrapping_add(step as u64 + 1)))?;
        let task = cross_entropy(&pass.logits, &targets)?;
        let measured = task.to_scalar::<f32>()?;
        ensure!(measured.is_finite(), "nonfinite task loss");
        let objective = policy_objective(&task, &pass.log_probability, baseline)?;
        update(&parameters, &objective)?;
        history = pass.pulse;
        visits += pass.visited.len();
        learned_halts += usize::from(pass.learned_halt);
        losses.push(measured);
        baseline = 0.9 * baseline + 0.1 * measured;
    }
    let changes = changes(&before, &snapshot(&parameters)?);
    let final_loss = evaluate(&model, config.context)?;
    ensure!(
        initial_loss.is_finite() && final_loss.is_finite(),
        "nonfinite evaluation loss"
    );
    Ok(json!({
        "schema": "hexweave-smoke-v1", "backend": "candle-cpu-fp32",
        "steps": steps, "seed": seed, "parameters": before.values().map(Vec::len).sum::<usize>(),
        "initial_loss": initial_loss, "final_loss": final_loss,
        "training_losses": losses, "parameter_l1_change": changes,
        "training_tile_visits": visits, "learned_halts": learned_halts,
        "fixture_target_evaluations": steps * config.context,
        "seconds": started.elapsed().as_secs_f64(),
        "history": "one detached terminal pulse carried between training examples; evaluation resets it",
        "routing": "one sequence route conditioned only on its first position; learned neighbors and halt",
        "fanout": "planner only; absent from numeric fixture",
        "limits": "tiny synthetic IDs, all weights resident, no tokenizer, checkpoint, full F parity or performance claim"
    }))
}

fn policy_objective(task: &Tensor, score: &Tensor, baseline: f32) -> Result<Tensor> {
    // REINFORCE: detached loss advantage times sampled log probability.
    let advantage = task.to_scalar::<f32>()? - baseline;
    Ok(task.add(&score.affine(f64::from(advantage), 0.0)?)?)
}

fn example(length: usize, start: u32) -> (Vec<u32>, Vec<u32>) {
    let tokens: Vec<_> = (0..=length)
        .map(|index| 1 + (start + index as u32 * 3) % 23)
        .collect();
    (tokens[..length].to_vec(), tokens[1..].to_vec())
}

fn evaluate(model: &HexModel, length: usize) -> Result<f32> {
    let (inputs, targets) = example(length, 11);
    let pass = model.forward(&inputs, &model.empty_history()?, None)?;
    Ok(cross_entropy(&pass.logits, &targets)?.to_scalar::<f32>()?)
}

fn update(parameters: &[(String, Var)], objective: &Tensor) -> Result<()> {
    let gradients = objective.backward()?;
    let mut squared_norm = 0_f64;
    for (_, parameter) in parameters {
        if let Some(gradient) = gradients.get(parameter.as_tensor()) {
            for value in gradient.flatten_all()?.to_vec1::<f32>()? {
                ensure!(value.is_finite(), "nonfinite gradient");
                squared_norm += f64::from(value).powi(2);
            }
        }
    }
    let rate = 0.03 / squared_norm.sqrt().max(1.0);
    let mut staged = Vec::new();
    for (_, parameter) in parameters {
        if let Some(gradient) = gradients.get(parameter.as_tensor()) {
            let updated = parameter.sub(&gradient.affine(rate, 0.0)?)?.detach();
            ensure!(
                updated
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| value.is_finite()),
                "nonfinite staged parameter"
            );
            staged.push((parameter, updated));
        }
    }
    for (parameter, updated) in staged {
        parameter.set(&updated)?;
    }
    Ok(())
}

fn snapshot(parameters: &[(String, Var)]) -> Result<Snapshot> {
    parameters
        .iter()
        .map(|(name, value)| Ok((name.clone(), value.flatten_all()?.to_vec1::<f32>()?)))
        .collect()
}

fn changes(before: &Snapshot, after: &Snapshot) -> BTreeMap<&'static str, f64> {
    let mut result = BTreeMap::from([
        ("router", 0.0),
        ("pulse", 0.0),
        ("curve", 0.0),
        ("other", 0.0),
    ]);
    for (name, old) in before {
        let family = if name.starts_with("router") {
            "router"
        } else if name.starts_with("pulse") {
            "pulse"
        } else if name.contains("curve") {
            "curve"
        } else {
            "other"
        };
        let change = old
            .iter()
            .zip(&after[name])
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>();
        *result.get_mut(family).expect("declared parameter family") += change;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::policy_objective;
    use candle_core::{Device, Tensor, Var};

    #[test]
    fn score_gradient_penalizes_higher_loss_selected_action() -> anyhow::Result<()> {
        for (task_loss, expected_sign) in [(3_f32, 1_f32), (1_f32, -1_f32)] {
            let logits = Var::from_vec(vec![0_f32, 0_f32], 2, &Device::Cpu)?;
            let score = logits
                .narrow(0, 0, 1)?
                .reshape(())?
                .sub(&logits.log_sum_exp(0)?)?;
            let objective = policy_objective(&Tensor::new(task_loss, &Device::Cpu)?, &score, 2.0)?;
            let gradients = objective.backward()?;
            let gradient = gradients
                .get(logits.as_tensor())
                .expect("score gradient")
                .to_vec1::<f32>()?;
            assert_eq!(gradient[0].signum(), expected_sign);
            assert_eq!(gradient[1].signum(), -expected_sign);
        }
        Ok(())
    }
}
