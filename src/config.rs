use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;

pub const MAX_RADIUS: usize = 16;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub name: String,
    pub radius: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub heads: usize,
    pub context: usize,
    pub max_depth: usize,
    pub max_frontier: usize,
    pub max_visits: usize,
    pub fanout: bool,
}

#[derive(Debug, Serialize)]
pub struct Estimate {
    pub profile: String,
    pub tiles: usize,
    pub embedding_parameters: u64,
    pub attention_parameters: u64,
    pub ffn_parameters: u64,
    pub main_parameters: u64,
    pub overhead_parameters: Option<u64>,
    pub fp32_main_weight_bytes: u64,
    pub fp32_adam_main_state_floor_bytes: u64,
    pub accounting: &'static str,
    pub executable: bool,
    pub training_admitted: bool,
}

impl Config {
    /// Fixed synthetic token-ID geometry; never a full-model launch profile.
    pub fn smoke() -> Self {
        Self {
            name: "tiny".into(),
            radius: 1,
            hidden: 16,
            ffn: 32,
            vocab: 256,
            heads: 2,
            context: 8,
            max_depth: 3,
            max_frontier: 2,
            max_visits: 8,
            fanout: false,
        }
    }

    /// Read one bounded configuration document.
    pub fn load(path: &Path) -> Result<Self> {
        const LIMIT: u64 = 65_536;
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut bytes = Vec::new();
        file.take(LIMIT + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() as u64 <= LIMIT, "configuration exceeds 64 KiB");
        let config: Self = serde_json::from_slice(&bytes).context("parse configuration")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.name.len() <= 128, "profile name exceeds 128 bytes");
        ensure!(
            self.radius <= MAX_RADIUS,
            "radius must be at most {MAX_RADIUS}"
        );
        for (name, value, maximum) in [
            ("hidden", self.hidden, 16_384),
            ("ffn", self.ffn, 65_536),
            ("vocab", self.vocab, 1_048_576),
            ("heads", self.heads, 256),
            ("context", self.context, 131_072),
            ("max_depth", self.max_depth, 128),
            ("max_frontier", self.max_frontier, 64),
            ("max_visits", self.max_visits, 8_192),
        ] {
            ensure!(
                (1..=maximum).contains(&value),
                "{name} must be in 1..={maximum}"
            );
        }
        ensure!(
            self.hidden.is_multiple_of(self.heads),
            "hidden must be divisible by heads"
        );
        ensure!(
            self.max_frontier <= self.max_visits,
            "frontier cannot exceed visit budget"
        );
        self.tile_count()?;
        Ok(())
    }

    pub fn tile_count(&self) -> Result<usize> {
        tile_count(self.radius)
    }

    /// Count main matrices without constructing any model tensors.
    pub fn estimate(&self) -> Result<Estimate> {
        self.validate()?;
        let tiles = self.tile_count()?;
        let (n, d, f, v) = (
            tiles as u64,
            self.hidden as u64,
            self.ffn as u64,
            self.vocab as u64,
        );
        let embedding_parameters = product(&[v, d])?;
        let attention_parameters = product(&[n, 4, d, d])?;
        let ffn_parameters = product(&[n, 3, d, f])?;
        let main_parameters = embedding_parameters
            .checked_add(attention_parameters)
            .and_then(|total| total.checked_add(ffn_parameters))
            .context("parameter sum overflow")?;
        Ok(Estimate {
            profile: self.name.clone(),
            tiles,
            embedding_parameters,
            attention_parameters,
            ffn_parameters,
            main_parameters,
            overhead_parameters: None,
            fp32_main_weight_bytes: product(&[main_parameters, 4])?,
            fp32_adam_main_state_floor_bytes: product(&[main_parameters, 16])?,
            accounting: "V*d + tiles*(4*d*d + 3*d*ffn), tied embeddings; excludes norms, curves, controller, pulse, activations, attention workspaces, runtime and checkpoint copies; 16 bytes/main parameter assumes FP32 weights, gradients and two Adam moments; not physical admission or measured memory",
            executable: false,
            training_admitted: false,
        })
    }
}

pub fn tile_count(radius: usize) -> Result<usize> {
    ensure!(radius <= MAX_RADIUS, "radius must be at most {MAX_RADIUS}");
    radius
        .checked_add(1)
        .and_then(|next| radius.checked_mul(next))
        .and_then(|area| area.checked_mul(3))
        .and_then(|area| area.checked_add(1))
        .context("tile count overflow")
}

fn product(factors: &[u64]) -> Result<u64> {
    let mut result = 1_u64;
    for factor in factors {
        match result.checked_mul(*factor) {
            Some(value) => result = value,
            None => bail!("parameter or byte-count product overflow"),
        }
    }
    Ok(result)
}
