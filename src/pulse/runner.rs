use super::credit::{anchor, AnchorRequest};
use super::execution::{evaluate, trajectory, Step};
use super::history::History;
use super::report::{summary, Counts, Report};
use super::residency::Residents;
use super::state::Model;
use super::{Mode, Options, VOCAB, WIDTH};
use anyhow::Result;
use serde_json::Value;
use std::time::Instant;

pub(super) fn run(options: Options) -> Result<Value> {
    let started = Instant::now();
    let mut experiment = Experiment {
        model: Model::new(options.seed)?,
        residents: Residents::new(options.resident_limit),
        history: History::new(),
        counts: Counts::default(),
        baseline: (VOCAB as f32).ln(),
        losses: Vec::new(),
        predictor_error_sum: 0.0,
    };
    let initial = evaluate(&experiment.model, &mut experiment.counts)?;
    for step in 0..options.steps {
        experiment.advance(options, step)?;
    }
    let final_loss = evaluate(&experiment.model, &mut experiment.counts)?;
    experiment.model.validate_weights()?;
    let report = Report {
        options,
        initial,
        final_loss,
        seconds: started.elapsed().as_secs_f64(),
        training_losses: experiment.losses,
        predictor_error_sum: experiment.predictor_error_sum,
    };
    summary(
        &experiment.model,
        &experiment.residents,
        &experiment.history,
        &experiment.counts,
        report,
    )
}

struct Experiment {
    model: Model,
    residents: Residents,
    history: History,
    counts: Counts,
    baseline: f32,
    losses: Vec<f32>,
    predictor_error_sum: f32,
}

impl Experiment {
    fn advance(&mut self, options: Options, step: usize) -> Result<()> {
        self.history.prune(&self.model, step);
        let is_anchor = step.is_multiple_of(options.anchor_interval);
        if !is_anchor && options.mode == Mode::AnchorOnly {
            self.counts.skipped_local_opportunities += 1;
            return Ok(());
        }
        let context = if options.mode == Mode::Pulse {
            self.history.summary()
        } else {
            vec![0.0; WIDTH]
        };
        let local = !is_anchor && options.mode == Mode::Pulse;
        if local {
            self.replay_predictor()?;
        }
        let token = (step / options.anchor_interval + 5 * (step % options.anchor_interval)) % 12;
        let frames = trajectory(
            &mut self.model,
            &mut self.residents,
            &mut self.history,
            Step {
                context: &context,
                token,
                local,
                step,
                seed: options.seed.wrapping_add((step as u64).wrapping_mul(9_001)),
            },
            &mut self.counts,
        )?;
        if is_anchor {
            let result = anchor(
                &mut self.model,
                &mut self.residents,
                &mut self.history,
                AnchorRequest {
                    frames: &frames,
                    context: &context,
                    token,
                    target: (token + 1) % VOCAB,
                    step,
                    baseline: self.baseline,
                    train_predictor: options.mode == Mode::Pulse,
                },
            )?;
            self.baseline = 0.9 * self.baseline + 0.1 * result.loss;
            self.losses.push(result.loss);
            self.counts.anchor_steps += 1;
            self.counts.anchor_targets_per_training_input[token] += 1;
            self.counts.anchor_node_updates += result.node_updates;
            self.counts.anchor_returned_frames += frames.len();
            self.counts.routing_score_updates += result.router_updates;
            self.counts.predictor_anchor_updates += result.predictor_updates;
            self.counts.predictor_forward_calls += result.predictor_updates;
            self.predictor_error_sum += result.predictor_error_sum;
        } else {
            self.counts.predicted_steps += 1;
        }
        Ok(())
    }

    fn replay_predictor(&mut self) -> Result<()> {
        if let Some(record) = self.history.latest() {
            self.predictor_error_sum += self
                .model
                .train_predictor(&record.features, &record.correction)?;
            self.counts.predictor_replay_updates += 1;
            self.counts.predictor_forward_calls += 1;
        }
        Ok(())
    }
}
