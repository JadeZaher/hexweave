use super::data_io::valid_id;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(super) const RECORD_CAP: usize = 8192;
pub(super) const MANIFEST_CAP: usize = 65536;
pub(super) const STREAM_CAP: usize = 8192;
pub(super) const SOURCE_CAP: u64 = 1_048_576;
pub(super) const FILE_CAP: u64 = 4_194_304;
pub(super) const ROW_CAP: usize = 256;
pub(super) const TEMPLATE: &str = "Use evidence only; otherwise UNKNOWN.\nSOURCE {source_id}:{start}-{end}\n{evidence}\nQ: {question}\nA: ";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Split {
    Train,
    Development,
}
impl Split {
    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Development => "development",
        }
    }
    pub fn file(self) -> &'static str {
        match self {
            Self::Train => "train.jsonl",
            Self::Development => "development.jsonl",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub source_id: String,
    pub partition: String,
    pub original_path: String,
    pub snapshot_path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Lengths {
    pub question: usize,
    pub evidence: usize,
    pub completion: usize,
    pub prompt: usize,
    pub envelope: usize,
    pub sequence: usize,
}
impl Lengths {
    pub fn include(&mut self, other: Self) {
        self.question = self.question.max(other.question);
        self.evidence = self.evidence.max(other.evidence);
        self.completion = self.completion.max(other.completion);
        self.prompt = self.prompt.max(other.prompt);
        self.envelope = self.envelope.max(other.envelope);
        self.sequence = self.sequence.max(other.sequence);
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Row {
    pub id: String,
    pub split: String,
    pub fact_family_id: String,
    pub source_id: String,
    pub source_sha256: String,
    pub evidence_start_byte: u64,
    pub evidence_end_byte: u64,
    pub evidence: String,
    pub question: String,
    pub answerable: bool,
    pub answer: String,
    pub citation: Option<String>,
    pub prompt: String,
    pub completion: String,
    pub utf8_bytes: Lengths,
    pub authoring: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TargetKind {
    Answer,
    Citation,
    Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Masks {
    pub answer: (usize, usize),
    pub citation: (usize, usize),
}
impl Masks {
    pub fn kind(self, offset: usize) -> TargetKind {
        if (self.answer.0..self.answer.1).contains(&offset) {
            TargetKind::Answer
        } else if (self.citation.0..self.citation.1).contains(&offset) {
            TargetKind::Citation
        } else {
            TargetKind::Format
        }
    }
    pub fn counts(self, end: usize) -> [u64; 3] {
        let answer = end.min(self.answer.1).saturating_sub(self.answer.0) as u64;
        let citation = end.min(self.citation.1).saturating_sub(self.citation.0) as u64;
        [answer, citation, end as u64 - answer - citation]
    }
}

pub(super) fn validate_row(
    row: &Row,
    source: &Source,
    split: Split,
    index: usize,
) -> Result<Masks> {
    ensure!(
        row.id == format!("{}-{:03}", split.name(), index + 1) && valid_id(&row.id, 48),
        "row order/ID mismatch"
    );
    ensure!(
        row.split == split.name()
            && source.partition == row.split
            && row.source_id == source.source_id,
        "source/split mismatch"
    );
    ensure!(
        valid_id(&row.fact_family_id, 96) && row.source_sha256 == source.sha256,
        "family/source identity mismatch"
    );
    ensure!(
        row.authoring == "handwritten-source-grounded-qa-v1",
        "unknown row provenance"
    );
    let citation = format!(
        "{}:{}-{}",
        row.source_id, row.evidence_start_byte, row.evidence_end_byte
    );
    ensure!(
        row.answerable == row.citation.is_some() && !row.answer.is_empty(),
        "invalid answerability"
    );
    ensure!(
        if row.answerable {
            row.answer != "UNKNOWN" && row.citation.as_deref() == Some(citation.as_str())
        } else {
            row.answer == "UNKNOWN"
        },
        "citation/abstention mismatch"
    );
    let prompt = format!(
        "Use evidence only; otherwise UNKNOWN.\nSOURCE {citation}\n{}\nQ: {}\nA: ",
        row.evidence, row.question
    );
    let answer = serde_json::to_string(&row.answer)?;
    let reference = serde_json::to_string(&row.citation)?;
    let completion = format!("{{\"answer\":{answer},\"citation\":{reference}}}\n");
    ensure!(
        row.prompt == prompt && row.completion == completion,
        "noncanonical prompt/completion"
    );
    let envelope = prompt
        .len()
        .checked_sub(row.evidence.len() + row.question.len())
        .context("envelope underflow")?;
    let lengths = Lengths {
        question: row.question.len(),
        evidence: row.evidence.len(),
        completion: completion.len(),
        prompt: prompt.len(),
        envelope,
        sequence: prompt.len() + completion.len(),
    };
    ensure!(
        lengths == row.utf8_bytes
            && lengths.question > 0
            && lengths.question <= 144
            && lengths.evidence > 0
            && lengths.evidence <= 608
            && lengths.envelope <= 80
            && lengths.completion <= 192
            && lengths.sequence <= 1024,
        "QA byte limits or lengths mismatch"
    );
    let citation_start = 10 + answer.len() + 12;
    let masks = Masks {
        answer: (11, 9 + answer.len()),
        citation: if row.citation.is_some() {
            (citation_start + 1, citation_start + reference.len() - 1)
        } else {
            (citation_start, citation_start + reference.len())
        },
    };
    ensure!(
        masks.answer.1 <= masks.citation.0 && masks.citation.1 < completion.len(),
        "invalid target masks"
    );
    Ok(masks)
}

#[derive(Deserialize)]
pub(super) struct Manifest {
    pub schema: String,
    pub dataset_version: String,
    pub source_as_of: String,
    pub source_pin_manifest_sha256: String,
    pub sources: Vec<Source>,
    pub artifacts: BTreeMap<String, Artifact>,
    pub prompt_template: String,
    pub limits: serde_json::Value,
    pub external_sources_fetched: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Artifact {
    pub sha256: String,
    pub bytes: u64,
    pub rows: usize,
    pub answerable: usize,
    pub abstention: usize,
    pub fact_families: usize,
    pub unique_evidence_spans: usize,
    pub max_utf8_bytes: Lengths,
    pub total_supervised_completion_bytes: u64,
}

#[derive(Deserialize)]
pub(super) struct SourcePins {
    pub candidate_files: Vec<SourcePin>,
}

#[derive(Deserialize)]
pub(super) struct SourcePin {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub family: String,
}
