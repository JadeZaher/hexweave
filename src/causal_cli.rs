use anyhow::{bail, Result};
use std::path::Path;

pub fn execute(args: &[String]) -> Result<serde_json::Value> {
    match args {
        [command] if command == "plan" => hexweave::causal::plan(),
        [command, flag, request] if command == "train" && flag == "--owner-request" =>
            hexweave::causal::run_owned(Path::new(request)),
        _ => bail!("expected causal plan or causal train --owner-request <owned attempt/worker-request.json>"),
    }
}
