//! First complete retention references for one measurement session; see CONNECTED.md.

use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(super) struct References {
    start_anchor: u64,
    scores: BTreeMap<(u64, u64), (u64, f64)>,
}

impl References {
    pub fn new(start_anchor: u64) -> Self {
        Self {
            start_anchor,
            scores: BTreeMap::new(),
        }
    }

    pub fn annotate(&mut self, evaluation: &mut Value, anchors: u64) {
        evaluation["retention_reference_scope"] = json!({
            "kind": "current-measurement-session", "session_start_anchor": self.start_anchor,
            "rule": "first-complete-post-exposure-score-for-identical-offset-and-target-count",
        });
        let Some(probes) = evaluation["retention"]["probes"].as_array_mut() else {
            return;
        };
        for probe in probes {
            let (Some(offset), Some(requested), Some(scored)) = (
                probe["byte_offset"].as_u64(),
                probe["requested_targets"].as_u64(),
                probe["scored_targets"].as_u64(),
            ) else {
                continue;
            };
            let complete = scored == requested && scored > 0 && probe["partial"] == false;
            let value = probe["bits_per_byte"]
                .as_f64()
                .filter(|value| value.is_finite());
            if complete {
                if let Some(bits) = value {
                    self.scores
                        .entry((offset, requested))
                        .or_insert((anchors, bits));
                }
            }
            if let Some(&(reference_anchor, reference_bits)) = self.scores.get(&(offset, requested))
            {
                probe["reference_anchor"] = json!(reference_anchor);
                probe["reference_bits_per_byte"] = json!(reference_bits);
                probe["delta_bits_per_byte"] =
                    json!(value.filter(|_| complete).map(|bits| bits - reference_bits));
                probe["reference_at_first_exposure"] =
                    json!(probe["first_pass_observed_at_anchor"] == reference_anchor);
            } else {
                probe["reference_anchor"] = Value::Null;
                probe["reference_bits_per_byte"] = Value::Null;
                probe["delta_bits_per_byte"] = Value::Null;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluation(scored: u64, bits: f64) -> Value {
        json!({"retention":{"probes":[{"byte_offset":0,"requested_targets":128,
            "scored_targets":scored,"partial":scored != 128,"bits_per_byte":bits,
            "first_pass_observed_at_anchor":128}]}})
    }

    #[test]
    fn retention_deltas_require_complete_identical_target_sets() {
        let mut references = References::new(0);
        let mut partial = evaluation(64, 1.0);
        references.annotate(&mut partial, 128);
        assert!(partial["retention"]["probes"][0]["reference_anchor"].is_null());
        let mut complete = evaluation(128, 2.0);
        references.annotate(&mut complete, 256);
        assert_eq!(complete["retention"]["probes"][0]["reference_anchor"], 256);
        assert_eq!(
            complete["retention"]["probes"][0]["delta_bits_per_byte"],
            0.0
        );
        references.annotate(&mut partial, 512);
        assert!(partial["retention"]["probes"][0]["delta_bits_per_byte"].is_null());
        let mut later = evaluation(128, 2.5);
        references.annotate(&mut later, 1024);
        assert_eq!(later["retention"]["probes"][0]["delta_bits_per_byte"], 0.5);
        assert_eq!(
            later["retention"]["probes"][0]["reference_at_first_exposure"],
            false
        );
    }
}
