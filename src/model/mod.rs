mod controller;
mod ops;
mod tile;

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, Var};

use crate::{config::Config, topology::Mesh};
use controller::{Controller, PULSE_WIDTH};
use ops::{log_softmax, Random};
use tile::Tile;

/// Tiny all-resident causal reference; see this directory's AGENTS.md.
pub struct HexModel {
    config: Config,
    mesh: Mesh,
    embedding: Var,
    tiles: Vec<Tile>,
    controller: Controller,
}

pub struct Forward {
    pub logits: Tensor,
    pub log_probability: Tensor,
    pub pulse: Tensor,
    pub visited: Vec<usize>,
    pub learned_halt: bool,
}

impl HexModel {
    pub fn new(config: Config, seed: u64) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.radius == 1
                && config.hidden <= 32
                && config.ffn <= 64
                && config.vocab <= 256
                && config.context <= 16
                && config.max_depth <= 8,
            "numerical backend admits only tiny CPU fixtures"
        );
        ensure!(
            !config.fanout,
            "numeric fanout is not implemented; use the planner"
        );
        let mesh = Mesh::new(config.radius)?;
        let mut random = Random::new(seed);
        let embedding = random.matrix(config.vocab, config.hidden)?;
        let tiles = (0..mesh.len())
            .map(|_| Tile::new(config.hidden, config.ffn, PULSE_WIDTH, &mut random))
            .collect::<candle_core::Result<_>>()?;
        let controller = Controller::new(config.hidden, mesh.len(), &mut random)?;
        Ok(Self {
            config,
            mesh,
            embedding,
            tiles,
            controller,
        })
    }

    pub fn empty_history(&self) -> Result<Tensor> {
        Ok(Tensor::zeros((1, PULSE_WIDTH), DType::F32, &Device::Cpu)?)
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        history: &Tensor,
        sample_seed: Option<u64>,
    ) -> Result<Forward> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.config.context,
            "invalid context length"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.config.vocab),
            "token outside vocabulary"
        );
        ensure!(
            history.dims() == [1, PULSE_WIDTH],
            "invalid pulse history shape"
        );
        ensure!(
            history.dtype() == DType::F32 && history.device().is_cpu(),
            "pulse history requires CPU FP32"
        );
        ensure!(
            history
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()),
            "nonfinite pulse history"
        );
        let ids = Tensor::from_slice(tokens, tokens.len(), &Device::Cpu)?;
        let embedded = self.embedding.index_select(&ids, 0)?;
        let mut hidden = embedded.add(&positions(tokens.len(), self.config.hidden)?)?;
        let mut pulse = history.detach();
        let mut random = sample_seed.map(Random::new);
        let mut score = Tensor::new(0_f32, &Device::Cpu)?;
        let mut visited = Vec::new();
        let mut learned_halt = false;
        let mut current = self
            .mesh
            .coordinates()
            .iter()
            .position(|cell| cell.q == 0 && cell.r == 0)
            .unwrap_or(0);
        for depth in 0..self.config.max_depth.min(self.config.max_visits) {
            pulse = self.controller.pulse(&hidden.narrow(0, 0, 1)?, &pulse)?;
            hidden = self.tiles[current].forward(&hidden, &pulse, self.config.heads)?;
            visited.push(current);
            if depth + 1 == self.config.max_depth.min(self.config.max_visits) {
                break;
            }
            let (next, log_probability) = self.choose(current, &hidden, &pulse, random.as_mut())?;
            score = score.add(&log_probability)?;
            if next == self.mesh.len() {
                learned_halt = true;
                break;
            }
            current = next;
        }
        let logits = hidden.matmul(&self.embedding.t()?)?;
        ensure!(
            logits
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()),
            "nonfinite model logits"
        );
        Ok(Forward {
            logits,
            log_probability: score,
            pulse: pulse.detach(),
            visited,
            learned_halt,
        })
    }

    fn choose(
        &self,
        current: usize,
        hidden: &Tensor,
        pulse: &Tensor,
        random: Option<&mut Random>,
    ) -> Result<(usize, Tensor)> {
        let mut candidates = self.mesh.neighbors(current)?;
        candidates.push(self.mesh.len());
        let position = self.mesh.coordinates()[current];
        let offsets: Vec<_> = candidates
            .iter()
            .flat_map(|&id| {
                if id == self.mesh.len() {
                    [0.0, 0.0]
                } else {
                    let next = self.mesh.coordinates()[id];
                    [(next.q - position.q) as f32, (next.r - position.r) as f32]
                }
            })
            .collect();
        Ok(self.controller.choose(
            &hidden.narrow(0, 0, 1)?,
            pulse,
            &candidates,
            &offsets,
            random,
        )?)
    }

    pub fn parameters(&self) -> Vec<(String, Var)> {
        let mut parameters = vec![("embedding".into(), self.embedding.clone())];
        parameters.extend(
            self.controller
                .parameters()
                .into_iter()
                .map(|(name, value)| (name.to_string(), value)),
        );
        for (index, tile) in self.tiles.iter().enumerate() {
            parameters.extend(
                tile.parameters()
                    .into_iter()
                    .map(|(name, value)| (format!("tile.{index}.{name}"), value)),
            );
        }
        parameters
    }
}

fn positions(length: usize, width: usize) -> candle_core::Result<Tensor> {
    let values: Vec<_> = (0..length)
        .flat_map(|position| {
            (0..width).map(move |coordinate| {
                let angle =
                    position as f32 / 10_000_f32.powf((coordinate / 2 * 2) as f32 / width as f32);
                0.05 * if coordinate % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                }
            })
        })
        .collect();
    Tensor::from_vec(values, (length, width), &Device::Cpu)
}

pub fn cross_entropy(logits: &Tensor, targets: &[u32]) -> Result<Tensor> {
    let (length, vocabulary) = logits.dims2()?;
    ensure!(
        targets.len() == length && targets.iter().all(|&id| (id as usize) < vocabulary),
        "invalid targets"
    );
    let indices = Tensor::from_slice(targets, (length, 1), &Device::Cpu)?;
    Ok(log_softmax(logits, 1)?
        .gather(&indices, 1)?
        .mean_all()?
        .neg()?)
}
