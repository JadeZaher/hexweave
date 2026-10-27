use super::data_io::sha256;
use super::data_tests::Fixture;
use serde_json::{json, Value};
use std::fs;

fn generic_fixture() -> Fixture {
    let mut fixture = Fixture::new();
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("manifest.json")).unwrap()).unwrap();
    let sources: Vec<Value> = manifest["sources"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["partition"] != "reserved_evaluation_source")
        .cloned()
        .collect();
    let files: Vec<Value> = sources.iter().map(|s| json!({
        "path":s["original_path"], "sha256":s["sha256"], "bytes":s["bytes"], "family":s["partition"],
    })).collect();
    let mut provenance = serde_json::Map::new();
    for source in &sources {
        provenance.insert(source["source_id"].as_str().unwrap().into(), json!({
            "url":"https://example.test/synthetic-fixture", "title":"Synthetic parser fixture",
            "publisher":"Fixture", "retrieved_at":"2026-10-24T00:00:00Z",
            "license_url":"https://example.test/license", "license_basis":"Synthetic test metadata only",
            "transformation":"Historical engine fixture copied solely to test generic parsing",
        }));
    }
    let pins = json!({"schema":"hexweave-evidence-source-pins-v1", "candidate_files":files, "provenance":provenance});
    let bytes = serde_json::to_vec(&pins).unwrap();
    fs::write(fixture.root.join("source-pins.json"), &bytes).unwrap();
    manifest["sources"] = json!(sources);
    manifest["schema"] = json!("hexweave-evidence-qa-v1");
    manifest["dataset_version"] = json!("generic-parser-fixture-v1");
    manifest["external_sources_fetched"] = json!(true);
    manifest["source_pin_manifest_sha256"] = json!(sha256(&bytes));
    fixture.manifest(|value| *value = manifest);
    fixture
}

#[test]
fn generic_identity_is_distinct_and_retains_span_masks_and_disjointness_checks() {
    let fixture = generic_fixture();
    let dataset = fixture.open().unwrap();
    assert_eq!(dataset.identity().schema, "evidence-causal-data-v1");
    assert_eq!(dataset.len(), 60);
    assert_eq!(dataset.identity().completion_targets_per_epoch, 7111);
    assert!(dataset.record(0).is_ok());
    let historical = Fixture::new().open().unwrap();
    assert_eq!(historical.identity().schema, "project-docs-causal-data-v1");
    assert_ne!(historical.identity(), dataset.identity());
}

#[test]
fn generic_data_rejects_false_provenance_missing_pins_and_legacy_family_labels() {
    for fault in [
        "external",
        "version",
        "missing",
        "family",
        "timestamp",
        "extra",
        "source",
    ] {
        let mut fixture = generic_fixture();
        if fault == "external" {
            fixture.manifest(|value| value["external_sources_fetched"] = json!(false));
        } else if fault == "version" {
            fixture.manifest(|value| value["dataset_version"] = json!(""));
        } else if fault == "source" {
            let source = fixture.root.join("sources/docs/TRAINING-PULSE.md");
            fs::write(source, "changed").unwrap();
        } else {
            let path = fixture.root.join("source-pins.json");
            let mut pins: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            let first = pins["provenance"]
                .as_object()
                .unwrap()
                .keys()
                .next()
                .unwrap()
                .clone();
            match fault {
                "missing" => {
                    pins["provenance"].as_object_mut().unwrap().remove(&first);
                }
                "family" => pins["candidate_files"][0]["family"] = json!("train_ternary_text"),
                "timestamp" => {
                    pins["provenance"][&first]["retrieved_at"] = json!("2026-02-30T00:00:00Z")
                }
                _ => pins["provenance"][&first]["unsupported"] = json!(true),
            }
            let bytes = serde_json::to_vec(&pins).unwrap();
            fs::write(&path, &bytes).unwrap();
            fixture.manifest(|value| value["source_pin_manifest_sha256"] = json!(sha256(&bytes)));
        }
        assert!(fixture.open().is_err(), "accepted {fault}");
    }
}
