use super::super::execution;
use super::super::generation::lowest_maximum;
use super::super::shared::Readout;
use super::super::state::State;
use super::super::store_api::TileStore;
use super::super::{credit, immutable, norm};
use super::fixture::Store;
use crate::topology::Mesh;
use anyhow::{ensure, Result};
use serde::Serialize;

pub(super) struct Probe {
    pub route: Vec<(usize, Option<usize>)>,
    pub generated_byte: u8,
    prefix: &'static [u8],
    frozen: Vec<(Vec<f32>, Vec<f32>)>,
}

#[derive(Serialize)]
pub(super) struct Fit {
    prediction_norm: f64,
    clipped_prediction_norm: f64,
    target_norm: f64,
    coordinate_mean_mse: f64,
    zero_predictor_mse: f64,
    cosine: Option<f64>,
    cosine_undefined: Option<&'static str>,
}

#[derive(Serialize)]
pub(super) struct Observation {
    pub state_sha256: String,
    pub anchor_loss: f32,
    pub pending_loss: f32,
    prefix_open_minus_close_logit: f32,
    generated_prefix_open_minus_close_logit: f32,
    frozen_initial_anchor_predictor_fit: Vec<Fit>,
}

impl Probe {
    pub fn freeze(state: &State, store: &mut Store, prefix: &'static [u8]) -> Result<Self> {
        let before = immutable::identity(state, store)?;
        let visible = state.shared.prefix(prefix)?;
        let trace = execution::forward_path(
            store,
            state.geometry,
            &visible,
            vec![0.0; state.geometry.width],
            &mut Some(state.seed),
            state.tile_versions,
        )?;
        let generated_byte = u8::try_from(lowest_maximum(
            &state.shared.readout(&trace.output)?.logits,
        )?)?;
        let gathered = credit::gather(state, store, &visible, &trace, b'{')?;
        let mesh = Mesh::new(1)?;
        let mut frozen = Vec::with_capacity(trace.frames.len());
        for (index, frame) in trace.frames.iter().enumerate() {
            let output = trace
                .frames
                .get(index + 1)
                .map_or(trace.output.as_slice(), |next| next.input.as_slice());
            let point = mesh.coordinates()[frame.tile_id];
            let correction = norm::unit_clip(&gathered.corrections[index])?;
            frozen.push((
                state
                    .predictor
                    .features(&frame.input, output, &trace.history, point.q, point.r)?,
                norm::unit_clip(&correction)?,
            ));
        }
        let route = trace.frames.iter().map(|f| (f.tile_id, f.action)).collect();
        assert_eq!(before, immutable::identity(state, store)?);
        Ok(Self {
            route,
            generated_byte,
            prefix,
            frozen,
        })
    }

    fn readout(&self, state: &State, store: &mut Store, bytes: &[u8]) -> Result<Readout> {
        let prefix = state.shared.prefix(bytes)?;
        let history = vec![0.0; state.geometry.width];
        let mut input = prefix.initial(state.geometry).to_vec();
        let mesh = Mesh::new(1)?;
        ensure!((1..=4).contains(&self.route.len()), "fixture route bound");
        for (depth, &(id, action)) in self.route.iter().enumerate() {
            let output = store.read(id, |tile| {
                execution::checked_tile(tile, state.geometry, id, state.tile_versions[id])?;
                Ok(tile.forward(&input, &prefix.memory, &history)?.output)
            })?;
            assert_eq!(
                execution::destination(&mesh, id, action)?,
                self.route.get(depth + 1).map(|next| next.0)
            );
            input = output;
        }
        state.shared.readout(&input)
    }

    pub fn observe(&self, state: &State, store: &mut Store) -> Result<Observation> {
        let before = immutable::identity(state, store)?;
        let anchor = self.readout(state, store, self.prefix)?;
        let mut pending = self.prefix.to_vec();
        pending.push(b'{');
        let next = self.readout(state, store, &pending)?;
        let mut generated = self.prefix.to_vec();
        generated.push(self.generated_byte);
        let alternate = self.readout(state, store, &generated)?;
        let observation = Observation {
            state_sha256: before.clone(),
            anchor_loss: anchor.loss(b'{')?,
            pending_loss: next.loss(b'"')?,
            prefix_open_minus_close_logit: anchor.logits[123] - anchor.logits[125],
            generated_prefix_open_minus_close_logit: alternate.logits[123] - alternate.logits[125],
            frozen_initial_anchor_predictor_fit: self
                .frozen
                .iter()
                .map(|(features, target)| {
                    fit(
                        &state.predictor.predict(features)?,
                        target,
                        state.geometry.width,
                    )
                })
                .collect::<Result<_>>()?,
        };
        assert!(observation.anchor_loss.is_finite() && observation.anchor_loss >= 0.0);
        assert!(observation.pending_loss.is_finite() && observation.pending_loss >= 0.0);
        assert!(observation.prefix_open_minus_close_logit.is_finite());
        assert!(observation
            .generated_prefix_open_minus_close_logit
            .is_finite());
        assert_eq!(before, immutable::identity(state, store)?);
        Ok(observation)
    }
}

fn fit(prediction: &[f32], target: &[f32], width: usize) -> Result<Fit> {
    norm::vector(prediction, width)?;
    norm::vector(target, width)?;
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum::<f64>()
    };
    let prediction_norm = dot(prediction, prediction).sqrt();
    let target_square = dot(target, target);
    let target_norm = target_square.sqrt();
    let coordinate_mean_mse = prediction
        .iter()
        .zip(target)
        .map(|(&p, &t)| (f64::from(p) - f64::from(t)).powi(2))
        .sum::<f64>()
        / width as f64;
    let zero_predictor_mse = target_square / width as f64;
    let cosine_undefined = match (prediction_norm == 0.0, target_norm == 0.0) {
        (true, true) => Some("both_zero"),
        (true, false) => Some("prediction_zero"),
        (false, true) => Some("target_zero"),
        (false, false) => None,
    };
    let cosine = if cosine_undefined.is_none() {
        Some(dot(prediction, target) / (prediction_norm * target_norm))
    } else {
        None
    };
    let clipped = norm::unit_clip(prediction)?;
    let clipped_prediction_norm = dot(&clipped, &clipped).sqrt();
    assert!(target_norm <= 1.000002 && clipped_prediction_norm <= 1.000002);
    assert!(zero_predictor_mse <= 1.000004 / width as f64);
    assert!(coordinate_mean_mse.is_finite() && (0.0..=4.00001).contains(&coordinate_mean_mse));
    assert!(cosine.is_none_or(|value| value.is_finite() && value.abs() <= 1.0000001));
    Ok(Fit {
        prediction_norm,
        clipped_prediction_norm,
        target_norm,
        coordinate_mean_mse,
        zero_predictor_mse,
        cosine,
        cosine_undefined,
    })
}
