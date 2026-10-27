use anyhow::{bail, Context, Result};
use hexweave::pulse::{run_with_options, Mode, Options};
use std::collections::HashSet;

/// Parse only bounded experiment controls; see docs/TERNARY-PULSE.md.
pub fn execute(args: &[String]) -> Result<serde_json::Value> {
    let mut options = Options::default();
    let mut seen = HashSet::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            bail!("each pulse option needs a value");
        }
        let flag = pair[0].as_str();
        if !seen.insert(flag) {
            bail!("duplicate pulse option {flag:?}");
        }
        match flag {
            "--steps" => options.steps = pair[1].parse().context("invalid steps")?,
            "--seed" => options.seed = pair[1].parse().context("invalid seed")?,
            "--residents" => {
                options.resident_limit = pair[1].parse().context("invalid residents")?
            }
            "--anchor-interval" => {
                options.anchor_interval = pair[1].parse().context("invalid anchor interval")?
            }
            "--mode" => {
                options.mode = match pair[1].as_str() {
                    "pulse" => Mode::Pulse,
                    "anchor-only" => Mode::AnchorOnly,
                    mode => bail!("unsupported pulse mode {mode:?}"),
                }
            }
            _ => bail!("unknown pulse option {flag:?}"),
        }
    }
    run_with_options(options)
}
