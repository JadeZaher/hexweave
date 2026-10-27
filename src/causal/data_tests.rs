use super::data::Dataset;
use super::data_contracts::{Lengths, Row, Split};
use super::data_io::sha256;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const PIN: &str = "b7e14d316318a2c44b88051d5f68ef11892bdd576fd3cadcdfe53a11a27d19bd";
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(super) struct Fixture {
    pub root: PathBuf,
    pub pin: String,
    parent: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let project = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let source = project.join("datasets/project-docs");
        let manifest = fs::read(source.join("manifest.json")).unwrap();
        assert_eq!(sha256(&manifest), PIN);
        let value: Value = serde_json::from_slice(&manifest).unwrap();
        fs::create_dir_all(project.join("work")).unwrap();
        let parent = project.join("work").canonicalize().unwrap();
        let root = parent.join(format!(
            "causal-data-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for name in [
            "manifest.json",
            "source-pins.json",
            "train.jsonl",
            "development.jsonl",
        ] {
            fs::copy(source.join(name), root.join(name)).unwrap();
        }
        for item in value["sources"].as_array().unwrap() {
            if item["partition"] == "reserved_evaluation_source" {
                continue;
            }
            let path = item["snapshot_path"].as_str().unwrap();
            fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
            fs::copy(source.join(path), root.join(path)).unwrap();
        }
        Self {
            root,
            pin: PIN.into(),
            parent,
        }
    }

    pub fn open(&self) -> anyhow::Result<Dataset> {
        Dataset::open(&self.root, &self.pin, Split::Train)
    }

    pub fn manifest(&mut self, change: impl FnOnce(&mut Value)) {
        let mut value: Value =
            serde_json::from_slice(&fs::read(self.root.join("manifest.json")).unwrap()).unwrap();
        change(&mut value);
        let bytes = serde_json::to_vec(&value).unwrap();
        self.pin = sha256(&bytes);
        fs::write(self.root.join("manifest.json"), bytes).unwrap();
    }

    pub fn row(&mut self, split: Split, index: usize, change: impl FnOnce(&mut Value)) {
        let text = fs::read_to_string(self.root.join(split.file())).unwrap();
        let mut rows: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        change(&mut rows[index]);
        let mut bytes = Vec::new();
        let mut maximum = Lengths::default();
        let mut targets = 0u64;
        for value in &rows {
            bytes.extend(serde_json::to_vec(value).unwrap());
            bytes.push(b'\n');
            let row: Row = serde_json::from_value(value.clone()).unwrap();
            maximum.include(row.utf8_bytes);
            targets += row.completion.len() as u64;
        }
        fs::write(self.root.join(split.file()), &bytes).unwrap();
        self.manifest(|manifest| {
            let artifact = &mut manifest["artifacts"][split.file()];
            artifact["sha256"] = json!(sha256(&bytes));
            artifact["bytes"] = json!(bytes.len());
            artifact["max_utf8_bytes"] = serde_json::to_value(maximum).unwrap();
            artifact["total_supervised_completion_bytes"] = json!(targets);
        });
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Ok(path) = self.root.canonicalize() {
            if path.parent() == Some(self.parent.as_path())
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("causal-data-test-")
            {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}

#[test]
fn pinned_reader_rejects_corruption_and_keeps_reserved_sources_unopened() {
    let fixture = Fixture::new();
    let dataset = fixture.open().unwrap();
    assert_eq!(dataset.len(), 60);
    assert_eq!(dataset.identity().completion_targets_per_epoch, 7111);
    assert!(!fixture.root.join("sources/docs/ARCHITECTURE.md").exists());
    assert!(Dataset::open(&fixture.root, &"0".repeat(64), Split::Train).is_err());
    let path = fixture.root.join("sources/docs/TRAINING-PULSE.md");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, bytes).unwrap();
    assert!(dataset.record(0).unwrap_err().to_string().contains("hash"));
    assert!(fixture.open().is_err());
}

#[test]
fn trusted_manifest_does_not_bypass_row_and_split_contracts() {
    let changes: [fn(&mut Value); 5] = [
        |row| row["split"] = json!("development"),
        |row| row["source_id"] = json!("E01"),
        |row| row["citation"] = json!("T01:0-1"),
        |row| row["prompt"] = json!("truncated"),
        |row| row["question"] = json!("x".repeat(9000)),
    ];
    for change in changes {
        let mut fixture = Fixture::new();
        fixture.row(Split::Train, 0, change);
        assert!(fixture.open().is_err());
    }
    let mut fixture = Fixture::new();
    fixture.row(Split::Development, 0, |row| {
        row["fact_family_id"] = json!("pulse_entry_points")
    });
    assert!(fixture.open().err().unwrap().to_string().contains("family"));
}

#[test]
fn pinned_paths_and_cached_record_bytes_fail_closed() {
    let mut fixture = Fixture::new();
    fixture.manifest(|manifest| manifest["sources"][0]["snapshot_path"] = json!("../escape.md"));
    assert!(fixture.open().is_err());
    let fixture = Fixture::new();
    let dataset = fixture.open().unwrap();
    let path = fixture.root.join("train.jsonl");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b' ';
    fs::write(path, bytes).unwrap();
    assert!(dataset.record(0).is_err());
    assert!(fixture.open().is_err());
    let mut fixture = Fixture::new();
    fixture.manifest(|manifest| {
        let artifact = manifest["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("train.jsonl")
            .unwrap();
        manifest["artifacts"]["cases.sealed.jsonl"] = artifact;
    });
    assert!(fixture.open().is_err());
}

#[test]
fn source_utf8_is_checked_even_outside_the_selected_span() {
    use super::data_contracts::{Source, STREAM_CAP};
    use super::data_io::source_span;
    let fixture = Fixture::new();
    let mut bytes = vec![b'x'; STREAM_CAP + 1];
    bytes[STREAM_CAP - 1] = 0xc3;
    bytes[STREAM_CAP] = 0xa9;
    let mut source = Source {
        source_id: "probe".into(),
        partition: "train".into(),
        original_path: "probe.md".into(),
        snapshot_path: "sources/probe.md".into(),
        sha256: sha256(&bytes),
        bytes: bytes.len() as u64,
    };
    let path = fixture.root.join(&source.snapshot_path);
    fs::write(&path, &bytes).unwrap();
    assert_eq!(source_span(&fixture.root, &source, 0, 2).unwrap(), b"xx");
    bytes.pop();
    source.bytes = bytes.len() as u64;
    source.sha256 = sha256(&bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(source_span(&fixture.root, &source, 0, 2).is_err());
    bytes[0] = 0xff;
    source.sha256 = sha256(&bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(source_span(&fixture.root, &source, 1, 2).is_err());
}

#[test]
fn cursor_overflow_rejects_atomically_at_a_valid_maximum_count() {
    use super::cursor::Cursor;
    let fixture = Fixture::new();
    let dataset = fixture.open().unwrap();
    let total = dataset.identity().completion_targets_per_epoch;
    let mut cursor = Cursor::start(&dataset).unwrap();
    cursor.epoch = u64::MAX / total;
    let remainder = u64::MAX % total;
    let index = (0..dataset.len())
        .find(|&i| {
            let row = dataset.meta(i).unwrap();
            row.preceding.iter().sum::<u64>() + row.completion_bytes as u64 > remainder
        })
        .unwrap();
    let row = dataset.meta(index).unwrap();
    cursor.row_index = index;
    cursor.example_id = row.id.clone();
    cursor.completion_offset = (remainder - row.preceding.iter().sum::<u64>()) as usize;
    let partial = row.masks.counts(cursor.completion_offset);
    let counts: Vec<_> = (0..3)
        .map(|i| dataset.epoch_counts()[i] * cursor.epoch + row.preceding[i] + partial[i])
        .collect();
    cursor.supervised_targets = u64::MAX;
    cursor.answer_targets = counts[0];
    cursor.citation_targets = counts[1];
    cursor.format_targets = counts[2];
    cursor.validate(&dataset).unwrap();
    let before = cursor.clone();
    assert!(cursor.advance_anchor(&dataset).is_err());
    assert_eq!(cursor, before);
}

#[test]
fn abstention_null_and_json_syntax_have_distinct_target_masks() {
    use super::data_contracts::TargetKind;
    let fixture = Fixture::new();
    let dataset = fixture.open().unwrap();
    let row = (0..dataset.len())
        .map(|i| dataset.record(i).unwrap())
        .find(|row| row.completion.starts_with(b"{\"answer\":\"UNKNOWN\""))
        .unwrap();
    let null = row
        .completion
        .windows(4)
        .position(|bytes| bytes == b"null")
        .unwrap();
    for i in null..null + 4 {
        assert_eq!(row.masks.kind(i), TargetKind::Citation);
    }
    for i in 11..18 {
        assert_eq!(row.masks.kind(i), TargetKind::Answer);
    }
    assert_eq!(row.masks.kind(10), TargetKind::Format);
    assert_eq!(row.masks.kind(row.completion.len() - 1), TargetKind::Format);
    assert_eq!(
        row.masks.counts(row.completion.len()).iter().sum::<u64>(),
        row.completion.len() as u64
    );
}
