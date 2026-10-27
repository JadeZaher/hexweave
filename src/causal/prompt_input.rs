use super::child_protocol::SourceManifest;
use super::{child_io, storage_paths};
use anyhow::{ensure, Context, Result};
use serde::de::{self, SeqAccess, Visitor};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

pub(super) const INPUT_CAP: u64 = 65_536;
pub(super) const MAX_CASES: usize = 40;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prompt {
    pub id: String,
    pub prompt: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema: u32,
    #[serde(deserialize_with = "rows")]
    pub prompts: Vec<Prompt>,
}

fn rows<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<Prompt>, D::Error> {
    struct Rows;
    impl<'de> Visitor<'de> for Rows {
        type Value = Vec<Prompt>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("at most forty visible prompt rows")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut access: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut rows = Vec::with_capacity(MAX_CASES);
            while let Some(row) = access.next_element()? {
                if rows.len() == MAX_CASES {
                    return Err(de::Error::custom("too many prompt rows"));
                }
                rows.push(row);
            }
            Ok(rows)
        }
    }
    deserializer.deserialize_seq(Rows)
}

impl Input {
    pub fn parse(bytes: &[u8], cases: usize, new_bytes: usize, context: usize) -> Result<Self> {
        ensure!(
            bytes.len() as u64 <= INPUT_CAP,
            "prompt input exceeds bound"
        );
        let input: Self = serde_json::from_slice(bytes)?;
        ensure!(
            input.schema == 1
                && (1..=MAX_CASES).contains(&cases)
                && input.prompts.len() == cases
                && (1..=192).contains(&new_bytes),
            "prompt schema/count/output limit mismatch"
        );
        let mut ids = BTreeSet::new();
        for row in &input.prompts {
            ensure!(
                row.id.len() <= 64
                    && storage_paths::identifier(&row.id)
                    && ids.insert(row.id.as_str()),
                "invalid/duplicate prompt ID"
            );
            ensure!(
                !row.prompt.is_empty()
                    && row.prompt.len() <= 832
                    && row
                        .prompt
                        .len()
                        .checked_add(new_bytes)
                        .is_some_and(|n| n <= context),
                "prompt bytes exceed context"
            );
        }
        Ok(input)
    }
}

pub(super) fn read(workspace: &Path, pin: &SourceManifest) -> Result<Vec<u8>> {
    ensure!(
        pin.path.is_absolute() && storage_paths::hash_valid(&pin.sha256),
        "invalid prompt pin"
    );
    let parent = storage_paths::attempt(
        workspace,
        pin.path.parent().context("missing prompt parent")?,
    )?;
    let name = pin
        .path
        .file_name()
        .and_then(|v| v.to_str())
        .context("invalid prompt filename")?;
    let path = storage_paths::file(&parent, name)?;
    let bytes = child_io::read(&path, INPUT_CAP)?;
    ensure!(
        storage_paths::digest(&bytes) == pin.sha256,
        "prompt input pin changed"
    );
    Ok(bytes)
}
