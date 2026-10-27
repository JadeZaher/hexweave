use super::geometry::{Geometry, EVIDENCE_RULE};
use super::storage_paths::digest;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub(super) const PROTOCOL: &str = "causal-prefix-v3";
pub(super) const PROTOCOL_SHA256: &str =
    "f54b9dc8da314cf41e95a4cd40362e8fcad39a547a6ad442d31c2c676edac3a1";
pub(super) const V2_PROTOCOL_SHA256: &str =
    "b6d725b398867675701b9f9bf23484e1f2e72547f3029086d38561859c10fab3";
pub(super) const BASE_PROTOCOL_SHA256: &str =
    "762b6ae6abacae6e047ed5fc2db50cb95471475878ee2a20b8f8f2629fbcd478";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub schema: u32,
    pub protocol: String,
    pub profile: String,
    pub seed: u64,
    pub threads: usize,
    pub anchor_interval: usize,
    pub max_visits: usize,
    pub history_capacity: usize,
    pub context_bytes: usize,
    pub evidence_rule: String,
    pub route_credit: String,
    pub predictor_rule: String,
    pub anchor_batch: String,
    pub replay_selection: String,
}

impl Config {
    pub fn candidate(seed: u64) -> Self {
        Self {
            schema: 1,
            protocol: PROTOCOL.into(),
            profile: "candidate".into(),
            seed,
            threads: 1,
            anchor_interval: 4,
            max_visits: 4,
            history_capacity: 8,
            context_bytes: 1024,
            evidence_rule: EVIDENCE_RULE.into(),
            route_credit: "detached-u-experimental".into(),
            predictor_rule: "joint-adjacent-raw-row-mean-v1".into(),
            anchor_batch: "current-anchor-forward-v1".into(),
            replay_selection: "newest4-compatible-v1".into(),
        }
    }

    pub fn assay(seed: u64, rule: super::assay_update::Rule) -> Self {
        let mut value = Self::candidate(seed);
        value.protocol = super::assay_state::PROTOCOL.into();
        value.profile = match rule {
            super::assay_update::Rule::Dense => "retained-dense",
            super::assay_update::Rule::Magnitude => "retained-magnitude",
        }
        .into();
        value.anchor_interval = 1;
        value.replay_selection = "none-anchor-only-v1".into();
        value
    }

    pub fn assay_rule(&self) -> Option<super::assay_update::Rule> {
        if self.protocol != super::assay_state::PROTOCOL {
            return None;
        }
        match self.profile.as_str() {
            "retained-dense" => Some(super::assay_update::Rule::Dense),
            "retained-magnitude" => Some(super::assay_update::Rule::Magnitude),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(rule) = self.assay_rule() {
            ensure!(
                self == &Self::assay(self.seed, rule) && [7, 17, 29].contains(&self.seed),
                "unsupported assay config"
            );
            return Ok(());
        }
        ensure!(
            self == &Self::candidate(self.seed),
            "unsupported owned causal configuration"
        );
        Ok(())
    }

    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(digest(&serde_json::to_vec(&serde_json::to_value(self)?)?))
    }
}

pub(super) fn protocol_hash(protocol: &str) -> Option<&'static str> {
    match protocol {
        PROTOCOL => Some(PROTOCOL_SHA256),
        super::assay_state::PROTOCOL => Some(super::assay_state::CONTRACT_SHA),
        _ => None,
    }
}

pub(super) fn plan() -> Result<serde_json::Value> {
    let geometry = Geometry::candidate();
    Ok(
        serde_json::json!({"schema":1,"protocol":PROTOCOL,"config":Config::candidate(7),
        "geometry":geometry,"payload_ledger":geometry.ledger(1024)?,"model_allocated":false,
        "physical_admission":false,"training_admitted":false,"useful_assistant_estimate_seconds":null,
        "scope":"experimental byte-prefix candidate; original 81M/287M targets remain separate",
        "required":"independent review, integrated checks, owned physical admission and representative quality trajectory"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_protocol_cannot_be_admitted_as_joint_configuration() -> Result<()> {
        let mut config = Config::candidate(7);
        assert!(config.validate().is_ok());
        for legacy in ["causal-prefix-v1", "causal-prefix-v2"] {
            config.protocol = legacy.into();
            assert!(config.validate().is_err());
            assert!(config.sha256().is_err());
        }
        let mut value = serde_json::to_value(Config::candidate(7))?;
        for key in ["predictor_rule", "anchor_batch", "replay_selection"] {
            let old = value[key].clone();
            value[key] = serde_json::json!("drift");
            assert!(serde_json::from_value::<Config>(value.clone())?
                .validate()
                .is_err());
            value[key] = old;
        }
        Ok(())
    }
}
