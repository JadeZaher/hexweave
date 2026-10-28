use anyhow::{bail, Context, Result};
use hexweave::config::Config;
use std::path::Path;

mod causal_cli;

const USAGE: &str = "hexweave plan --config <path>\nhexweave smoke [--steps 4] [--seed 7]\nhexweave causal plan\nhexweave train --config <core-run.json>   (core path: latent Adam, greedy routes, anchor-only; see src/causal/AGENTS.md)";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..])?,
        Some("smoke") => smoke(&args[1..])?,
        Some("causal") => causal_cli::execute(&args[1..])?,
        Some("help" | "--help" | "-h") | None => {
            println!("{USAGE}");
            return Ok(());
        }
        Some("train") => train(&args[1..])?,
        Some(command) => bail!("unknown command {command:?}\n{USAGE}"),
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn train(args: &[String]) -> Result<serde_json::Value> {
    if args.len() != 2 || args[0] != "--config" {
        bail!("expected train --config <path>");
    }
    hexweave::causal::train(Path::new(&args[1]))
}

fn plan(args: &[String]) -> Result<serde_json::Value> {
    if args.len() != 2 || args[0] != "--config" {
        bail!("expected plan --config <path>");
    }
    let config = Config::load(Path::new(&args[1]))?;
    Ok(serde_json::to_value(config.estimate()?)?)
}

fn smoke(args: &[String]) -> Result<serde_json::Value> {
    let (mut steps, mut seed) = (4_usize, 7_u64);
    let (mut seen_steps, mut seen_seed) = (false, false);
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            bail!("each smoke option needs a value");
        }
        match pair[0].as_str() {
            "--steps" if !seen_steps => {
                steps = pair[1]
                    .parse()
                    .context("steps must be a positive integer")?;
                seen_steps = true;
            }
            "--seed" if !seen_seed => {
                seed = pair[1]
                    .parse()
                    .context("seed must be an unsigned integer")?;
                seen_seed = true;
            }
            flag => bail!("unknown or duplicate smoke option {flag:?}"),
        }
    }
    if !(1..=16).contains(&steps) {
        bail!("smoke steps must be in 1..=16");
    }
    hexweave::smoke::run_smoke(steps, seed)
}
