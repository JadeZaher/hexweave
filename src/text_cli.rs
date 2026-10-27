use anyhow::{bail, Context, Result};
use hexweave::text::{CreditDiagnostics, Mode, Options, Profile, RouteCredit};
use std::collections::HashSet;
use std::path::PathBuf;

/// Parse the bounded text experiment; see docs/TEXT-PULSE.md.
pub fn execute(args: &[String]) -> Result<serde_json::Value> {
    let mut options = Options::default();
    let (mut train, mut valid, mut work) = (None, None, None);
    let mut seen = HashSet::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            bail!("each text option needs a value");
        }
        let flag = pair[0].as_str();
        if !seen.insert(flag) {
            bail!("duplicate text option {flag:?}");
        }
        match flag {
            "--profile" => {
                options.profile = match pair[1].as_str() {
                    "fixture" => Profile::Fixture,
                    "t2-credit" => Profile::T2Credit,
                    value => bail!("unsupported text profile {value:?}"),
                }
            }
            "--credit-diagnostics" => {
                options.credit_diagnostics = match pair[1].as_str() {
                    "off" => CreditDiagnostics::Off,
                    "on" => CreditDiagnostics::On,
                    value => bail!("unsupported credit diagnostics {value:?}"),
                }
            }
            "--steps" => options.steps = pair[1].parse().context("invalid steps")?,
            "--seed" => options.seed = pair[1].parse().context("invalid seed")?,
            "--anchor-interval" => {
                options.anchor_interval = pair[1].parse().context("invalid anchor interval")?
            }
            "--mode" => {
                options.mode = match pair[1].as_str() {
                    "pulse" => Mode::Pulse,
                    "anchor-only" => Mode::AnchorOnly,
                    mode => bail!("unsupported text mode {mode:?}"),
                }
            }
            "--route-credit" => {
                options.route_credit = match pair[1].as_str() {
                    "coupled" => RouteCredit::Coupled,
                    "route-to-u-detached" => RouteCredit::RouteToUDetached,
                    credit => bail!("unsupported route credit {credit:?}"),
                }
            }
            "--train" => train = Some(PathBuf::from(&pair[1])),
            "--valid" => valid = Some(PathBuf::from(&pair[1])),
            "--work-dir" => work = Some(PathBuf::from(&pair[1])),
            _ => bail!("unknown text option {flag:?}"),
        }
    }
    options.validate()?;
    hexweave::text::run(
        &options,
        &train.context("--train is required")?,
        &valid.context("--valid is required")?,
        &work.context("--work-dir is required")?,
    )
}
