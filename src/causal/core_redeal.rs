//! Citation re-deal for science training: a fresh same-shape SOURCE value per (row, epoch) in the
//! prompt and its citation, so citing must copy rather than memorize; see AGENTS.md §core.

use super::cursor::{AnchorTarget, Cursor};
use super::data::Dataset;
use super::execution::TrainingSource;
use super::tuning;
use anyhow::{ensure, Context, Result};

const CITED: &[u8] = b"\"citation\":\"";

/// Training view of a QA dataset; cursor, order and target kinds stay the inner dataset's.
pub(super) struct CitationRedeal<'a> {
    pub inner: &'a Dataset,
    pub seed: u64,
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// `Ldd:<digits>-<digits>`, the shape of every SOURCE value and citation in the QA datasets.
fn is_citation(value: &[u8]) -> bool {
    let Some((label, range)) = std::str::from_utf8(value)
        .ok()
        .and_then(|t| t.split_once(':'))
    else {
        return false;
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    label.len() == 3
        && label.as_bytes()[0].is_ascii_uppercase()
        && digits(&label[1..])
        && range
            .split_once('-')
            .is_some_and(|(start, end)| digits(start) && digits(end))
}

/// Same length and punctuation; letters and digits drawn afresh (range numbers keep a nonzero lead).
fn reshaped(value: &[u8], draw: u64) -> Vec<u8> {
    let mut fresh = value.to_vec();
    for (index, byte) in fresh.iter_mut().enumerate() {
        let pick = tuning::hash(draw, &[index as u64]);
        let lead = index > 0 && matches!(value[index - 1], b':' | b'-');
        *byte = match *byte {
            b'A'..=b'Z' => b'A' + (pick % 26) as u8,
            b'0'..=b'9' if lead => b'1' + (pick % 9) as u8,
            b'0'..=b'9' => b'0' + (pick % 10) as u8,
            other => other,
        };
    }
    fresh
}

/// Re-deal the prompt's SOURCE value and the completion's matching citation; a null citation stays.
pub(super) fn redeal_citation(
    prompt: &[u8],
    completion: &[u8],
    draw: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let start = find(prompt, b"\nSOURCE ").context("prompt has no SOURCE line")? + 8;
    let length = prompt[start..]
        .iter()
        .position(|&b| b == b'\n')
        .context("unterminated SOURCE line")?;
    let value = &prompt[start..start + length];
    ensure!(is_citation(value), "unexpected SOURCE value shape");
    let fresh = reshaped(value, draw);
    let mut prompt = prompt.to_vec();
    prompt[start..start + length].copy_from_slice(&fresh);
    let mut completion = completion.to_vec();
    match find(&completion, &[CITED, value, b"\""].concat()) {
        Some(at) => completion[at + CITED.len()..][..length].copy_from_slice(&fresh),
        None => ensure!(
            find(&completion, b"\"citation\":null").is_some(),
            "completion cites neither its SOURCE nor null"
        ),
    }
    Ok((prompt, completion))
}

impl CitationRedeal<'_> {
    fn example(&self, cursor: &Cursor) -> Result<(Vec<u8>, Vec<u8>)> {
        let row = self.inner.record(cursor.row_index)?;
        let draw = tuning::hash(self.seed, &[cursor.row_index as u64, cursor.epoch]);
        redeal_citation(&row.prompt, &row.completion, draw)
    }
}

impl TrainingSource for CitationRedeal<'_> {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        self.inner.validate(cursor)
    }

    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        let original = self.inner.pending_prefix(cursor)?;
        let (mut prefix, completion) = self.example(cursor)?;
        prefix.extend_from_slice(&completion[..cursor.completion_offset]);
        ensure!(
            prefix.len() == original.len(),
            "citation re-deal changed the prefix length"
        );
        Ok(prefix)
    }

    fn anchor_target(&self, cursor: &Cursor) -> Result<AnchorTarget> {
        let mut target = self.inner.anchor_target(cursor)?;
        target.byte = self.example(cursor)?.1[cursor.completion_offset];
        Ok(target)
    }

    fn advance_anchor(&self, cursor: &mut Cursor) -> Result<()> {
        self.inner.advance_anchor(cursor)
    }
}
