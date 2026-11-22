use anyhow::{bail, Context, Result};
use hexweave::causal::Decoder;
use hexweave::config::Config;
use std::path::Path;

mod causal_cli;

const USAGE: &str = "hexweave plan --config <path>\nhexweave smoke [--steps 4] [--seed 7]\nhexweave causal plan\nhexweave train --config <core-run.json>\nhexweave complete --config <core-run.json> --checkpoint <aNNNNNNNNNNNN> --prompt <text> [--max-new-bytes 160] [--top-p P [--temperature 1.0] [--seed 0]] [--trace-bytes 0..16] [--feedback generated|prompt-only] [--quantize-audit]\nhexweave reach --config <core-run.json> --checkpoint <aNNNNNNNNNNNN> [--spans 32]";

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
        Some("complete") => complete(&args[1..])?,
        Some("reach") => reach(&args[1..])?,
        Some(command) => bail!("unknown command {command:?}\n{USAGE}"),
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn complete(args: &[String]) -> Result<serde_json::Value> {
    let (mut config, mut checkpoint, mut prompt, mut limit) = (None, None, None, None);
    let (mut trace, mut feedback) = (None, None);
    let (mut top_p, mut temperature, mut seed) = (None, None, None);
    let mut quantize = false;
    let mut at = 0;
    while at < args.len() {
        if args[at] == "--quantize-audit" {
            if quantize {
                bail!("duplicate quantize-audit option");
            }
            quantize = true;
            at += 1;
            continue;
        }
        let pair = args
            .get(at..at + 2)
            .context("each complete option needs a value")?;
        at += 2;
        match pair[0].as_str() {
            "--config" if config.is_none() => config = Some(pair[1].as_str()),
            "--checkpoint" if checkpoint.is_none() => checkpoint = Some(pair[1].as_str()),
            "--prompt" if prompt.is_none() => prompt = Some(pair[1].as_str()),
            "--max-new-bytes" if limit.is_none() => {
                limit = Some(
                    pair[1]
                        .parse::<usize>()
                        .context("max-new-bytes must be a positive integer")?,
                )
            }
            "--trace-bytes" if trace.is_none() => {
                trace = Some(
                    pair[1]
                        .parse::<usize>()
                        .context("trace-bytes must be an integer in 0..=16")?,
                );
            }
            "--feedback" if feedback.is_none() => {
                feedback = Some(match pair[1].as_str() {
                    "generated" => true,
                    "prompt-only" => false,
                    _ => bail!("feedback must be generated or prompt-only"),
                });
            }
            "--top-p" if top_p.is_none() => {
                top_p = Some(pair[1].parse::<f64>().context("top-p must be a number")?)
            }
            "--temperature" if temperature.is_none() => {
                temperature = Some(
                    pair[1]
                        .parse::<f64>()
                        .context("temperature must be a number")?,
                )
            }
            "--seed" if seed.is_none() => {
                seed = Some(
                    pair[1]
                        .parse::<u64>()
                        .context("seed must be an unsigned integer")?,
                )
            }
            flag => bail!("unknown or duplicate complete option {flag:?}"),
        }
    }
    let (Some(config), Some(checkpoint), Some(prompt)) = (config, checkpoint, prompt) else {
        bail!("expected complete --config <path> --checkpoint <name> --prompt <text>");
    };
    let decoder = match top_p {
        Some(top_p) => Decoder::Nucleus {
            temperature: temperature.unwrap_or(1.0),
            top_p,
            seed: seed.unwrap_or(0),
        },
        None if temperature.is_some() || seed.is_some() => {
            bail!("temperature and seed apply only with --top-p")
        }
        None => Decoder::Greedy,
    };
    hexweave::causal::complete_audit(
        Path::new(config),
        checkpoint,
        prompt,
        limit.unwrap_or(160),
        trace.unwrap_or(0),
        feedback.unwrap_or(true),
        quantize,
        decoder,
    )
}

fn reach(args: &[String]) -> Result<serde_json::Value> {
    let (mut config, mut checkpoint, mut spans) = (None, None, None);
    for pair in args.chunks(2) {
        match (pair[0].as_str(), pair.get(1)) {
            ("--config", Some(value)) if config.is_none() => config = Some(value),
            ("--checkpoint", Some(value)) if checkpoint.is_none() => checkpoint = Some(value),
            ("--spans", Some(value)) if spans.is_none() => {
                spans = Some(value.parse().context("invalid spans")?)
            }
            _ => bail!("expected reach --config <path> --checkpoint <name> [--spans N]"),
        }
    }
    let config = config.context("reach needs --config")?;
    let checkpoint = checkpoint.context("reach needs --checkpoint")?;
    hexweave::causal::context_reach(Path::new(config), checkpoint, spans.unwrap_or(32))
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
