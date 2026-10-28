//! Citation re-deal contracts through the real dataset cursor; see AGENTS.md §core.

use super::core_redeal::{redeal_citation, CitationRedeal};
use super::cursor::Cursor;
use super::execution::TrainingSource;
use anyhow::{Context, Result};

/// Offset just past `marker`, if it occurs.
fn after(text: &[u8], marker: &[u8]) -> Option<usize> {
    Some(text.windows(marker.len()).position(|w| w == marker)? + marker.len())
}

/// Through epoch 0 and into epoch 1 of the pinned fixture: only the SOURCE value and its citation
/// change, every trained citation byte copies the prompt's SOURCE byte, null citations stay, each
/// row's value changes between epochs and the run seed changes the stream.
#[test]
fn citation_redeal_trains_citing_as_copying_the_prompt_source() -> Result<()> {
    let fixture = super::data_tests::Fixture::new();
    let dataset = fixture.open()?;
    let redeal = CitationRedeal {
        inner: &dataset,
        seed: 7,
    };
    let other = CitationRedeal {
        inner: &dataset,
        seed: 8,
    };
    let start = Cursor::start(&dataset)?;
    assert_ne!(
        other.pending_prefix(&start)?,
        redeal.pending_prefix(&start)?
    );
    let mut first: Vec<Option<Vec<u8>>> = vec![None; dataset.len()];
    let (mut cursor, mut copied, mut nulls, mut redealt) = (start, 0, 0, 0);
    while cursor.epoch == 0 || cursor.row_index < 10 {
        let (prefix, original) = (
            redeal.pending_prefix(&cursor)?,
            dataset.pending_prefix(&cursor)?,
        );
        let prompt = dataset.record(cursor.row_index)?.prompt.len();
        let at = after(&prefix, b"\nSOURCE ").context("SOURCE line")?;
        let length = prefix[at..]
            .iter()
            .position(|&b| b == b'\n')
            .context("SOURCE end")?;
        let source = prefix[at..at + length].to_vec();
        let mut restored = prefix.clone();
        restored[at..at + length].copy_from_slice(&original[at..at + length]);
        if let Some(cited) = after(&prefix[prompt..], b"\"citation\":\"").map(|c| prompt + c) {
            let end = (cited + length).min(prefix.len());
            restored[cited..end].copy_from_slice(&original[cited..end]);
            if prefix.len() - cited < length {
                assert_eq!(
                    redeal.anchor_target(&cursor)?.byte,
                    source[prefix.len() - cited]
                );
                copied += 1;
            }
        }
        assert_eq!(
            restored, original,
            "only the SOURCE value and its citation may change"
        );
        if cursor.completion_offset == 0 {
            let completion = dataset.record(cursor.row_index)?.completion;
            nulls += usize::from(completion.ends_with(b"\"citation\":null}\n"));
            if let Some(earlier) = first[cursor.row_index].replace(source.clone()) {
                assert_ne!(
                    earlier, source,
                    "row {} kept its SOURCE value",
                    cursor.row_index
                );
                redealt += 1;
            }
        }
        redeal.advance_anchor(&mut cursor)?;
    }
    assert_eq!((copied, nulls, redealt), (796, 8, 10));
    assert!(redeal_citation(b"no source line", b"{}", 1).is_err());
    Ok(())
}
