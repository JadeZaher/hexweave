use anyhow::{bail, Result};

pub fn execute(args: &[String]) -> Result<serde_json::Value> {
    match args {
        [command] if command == "plan" => hexweave::causal::plan(),
        _ => bail!("expected causal plan"),
    }
}
