use super::data_contracts::{Manifest, SourcePin, SourcePins};
use super::data_io::valid_id;
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

pub(super) const GENERIC: &str = "hexweave-evidence-qa-v1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericPins {
    schema: String,
    candidate_files: Vec<Pin>,
    provenance: BTreeMap<String, Provenance>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: String,
    sha256: String,
    bytes: u64,
    family: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    url: String,
    title: String,
    publisher: String,
    retrieved_at: String,
    license_url: String,
    license_basis: String,
    transformation: String,
}

pub(super) fn identity_schema(manifest: &Manifest) -> Result<&'static str> {
    ensure!(
        !manifest.source_as_of.is_empty() && manifest.source_as_of.len() <= 256,
        "invalid source timestamp description"
    );
    match manifest.schema.as_str() {
        "hexweave-docs-qa-v1" => {
            ensure!(
                manifest.dataset_version == "t1-snapshot-d0-v1"
                    && !manifest.external_sources_fetched,
                "historical engine-fixture provenance changed"
            );
            Ok("project-docs-causal-data-v1")
        }
        GENERIC => {
            ensure!(
                valid_id(&manifest.dataset_version, 64) && manifest.external_sources_fetched,
                "generic evidence data requires version and external provenance"
            );
            Ok("evidence-causal-data-v1")
        }
        _ => anyhow::bail!("unsupported evidence dataset schema"),
    }
}

pub(super) fn family<'a>(manifest: &Manifest, partition: &'a str) -> Result<&'a str> {
    ensure!(
        matches!(
            partition,
            "train" | "development" | "reserved_evaluation_source"
        ),
        "unsupported source partition"
    );
    if manifest.schema == GENERIC {
        return Ok(partition);
    }
    Ok(match partition {
        "train" => "train_ternary_text",
        "development" => "development_benchmark",
        _ => "heldout_fp32_reference",
    })
}

pub(super) fn read_pins(bytes: &[u8], manifest: &Manifest) -> Result<Vec<SourcePin>> {
    if manifest.schema != GENERIC {
        return Ok(serde_json::from_slice::<SourcePins>(bytes)?.candidate_files);
    }
    let pins: GenericPins = serde_json::from_slice(bytes)?;
    ensure!(
        pins.schema == "hexweave-evidence-source-pins-v1"
            && pins.provenance.len() == manifest.sources.len(),
        "generic provenance schema/count mismatch"
    );
    for source in &manifest.sources {
        let entry = pins
            .provenance
            .get(&source.source_id)
            .ok_or_else(|| anyhow::anyhow!("missing source provenance"))?;
        ensure!(
            url(&entry.url)
                && url(&entry.license_url)
                && timestamp(&entry.retrieved_at)
                && bounded(&entry.title, 256)
                && bounded(&entry.publisher, 128)
                && bounded(&entry.license_basis, 1024)
                && bounded(&entry.transformation, 1024),
            "invalid bounded source/license/transformation provenance"
        );
    }
    Ok(pins
        .candidate_files
        .into_iter()
        .map(|p| SourcePin {
            path: p.path,
            sha256: p.sha256,
            bytes: p.bytes,
            family: p.family,
        })
        .collect())
}

fn bounded(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty() && value.len() <= maximum
}

fn url(value: &str) -> bool {
    value.len() > 8
        && value.len() <= 2048
        && value.starts_with("https://")
        && !value.chars().any(char::is_whitespace)
}

fn timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return false;
    }
    let fields = [(0, 4), (5, 7), (8, 10), (11, 13), (14, 16), (17, 19)];
    if fields
        .iter()
        .any(|&(a, b)| !bytes[a..b].iter().all(u8::is_ascii_digit))
    {
        return false;
    }
    let n = |a: usize, b: usize| {
        bytes[a..b]
            .iter()
            .fold(0u32, |n, b| n * 10 + u32::from(b - b'0'))
    };
    let (year, month, day) = (n(0, 4), n(5, 7), n(8, 10));
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    year > 0 && day > 0 && day <= days && n(11, 13) < 24 && n(14, 16) < 60 && n(17, 19) < 60
}
