//! Drill vocabulary, the key-swap probe, and the test-only harness source; see AGENTS.md §tuning.

use super::assay_source::Row;
#[cfg(test)]
use super::assay_source::Source;
#[cfg(test)]
use super::cursor::{AnchorTarget, Cursor};
#[cfg(test)]
use super::execution::TrainingSource;
#[cfg(test)]
use super::tuning;
#[cfg(test)]
use anyhow::{ensure, Result};

pub(super) const WORDS: [&[u8]; 4] = [b"soil", b"sand", b"clay", b"silt"];

/// True for the 61-byte single-line assay training layout that `redeal` rewrites.
pub(super) fn is_template(prompt: &[u8]) -> bool {
    prompt.len() == 61
        && prompt[11] == b'='
        && prompt[21] == b'='
        && &prompt[38..48] == b"sample at "
}

/// One draw re-deals both source values (distinct words) and the asked key; returns prompt and completion.
pub(super) fn redeal(template: &[u8], draw: u64) -> (Vec<u8>, Vec<u8>) {
    let mut prompt = template.to_vec();
    let first = (draw % 4) as usize;
    let second = (first + 1 + (draw / 4 % 3) as usize) % 4;
    let ask_first = (draw / 12).is_multiple_of(2);
    prompt[12..16].copy_from_slice(WORDS[first]);
    prompt[22..26].copy_from_slice(WORDS[second]);
    prompt[48] = if ask_first { prompt[8] } else { prompt[18] };
    let mut completion = WORDS[if ask_first { first } else { second }].to_vec();
    completion.push(b'\n');
    (prompt, completion)
}

/// Memorization-proof training: each (salt, epoch, row) re-deals both source values and the asked key.
#[cfg(test)]
pub(super) struct Shuffled<'a> {
    pub inner: &'a Source,
    pub salt: u64,
}

#[cfg(test)]
impl Shuffled<'_> {
    /// Template-checked prompt and completion for the cursor's row and epoch.
    fn example(&self, cursor: &Cursor) -> Result<(Vec<u8>, Vec<u8>)> {
        let template = &self.inner.row(&cursor.example_id)?.prompt;
        ensure!(is_template(template), "unexpected training template");
        let draw = tuning::hash(self.salt, &[cursor.row_index as u64, cursor.epoch]);
        Ok(redeal(template, draw))
    }
}

#[cfg(test)]
impl TrainingSource for Shuffled<'_> {
    fn validate(&self, cursor: &Cursor) -> Result<()> {
        self.inner.validate(cursor)
    }

    fn pending_prefix(&self, cursor: &Cursor) -> Result<Vec<u8>> {
        self.inner.pending_prefix(cursor)?;
        let (mut prompt, completion) = self.example(cursor)?;
        prompt.extend_from_slice(&completion[..cursor.completion_offset]);
        Ok(prompt)
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

/// Key-swap probe: flip the asked key (A<->B, C<->D) so retrieval must answer the other source value.
pub(super) fn swap(row: &Row) -> Option<Row> {
    let at = row.prompt.windows(10).position(|w| w == b"sample at ")? + 10;
    let mut prompt = row.prompt.clone();
    prompt[at] = match prompt[at] {
        b'A' => b'B',
        b'B' => b'A',
        b'C' => b'D',
        b'D' => b'C',
        _ => return None,
    };
    Some(Row {
        id: format!("swap-{}", row.id),
        group: row.group.clone(),
        prompt,
        completion: row.completion.clone(),
    })
}
