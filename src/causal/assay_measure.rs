use super::assay_source::Row;
use super::checkpoint_owner::Terminal;
use super::state::State;
use super::store_api::TileStore;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub id: String,
    pub content_nll: Vec<f32>,
    pub lf_nll: Option<f32>,
    pub first_byte_rank: Option<usize>,
    pub generated: Option<Vec<u8>>,
    pub generation_complete: bool,
    pub exact: Option<bool>,
}

fn readout<S: TileStore>(
    state: &State,
    store: &mut S,
    bytes: &[u8],
) -> Result<super::shared::Readout> {
    ensure!(bytes.len() <= 92, "assay readout prefix bound");
    let prefix = state.shared.prefix(bytes)?;
    let trace = super::execution::forward_path(
        store,
        state.geometry,
        &prefix,
        vec![0.0; state.geometry.width],
        &mut None,
        state.tile_versions,
    )?;
    state.shared.readout(&trace.output)
}

pub(super) fn measure<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    row: &Row,
    generate: bool,
    poll: &mut F,
) -> Result<Case> {
    let mut case = Case {
        id: row.id.clone(),
        content_nll: Vec::with_capacity(4),
        lf_nll: None,
        first_byte_rank: None,
        generated: None,
        generation_complete: false,
        exact: None,
    };
    let mut visible = row.prompt.clone();
    for (index, &target) in row.completion.iter().enumerate() {
        if poll()?.is_some() {
            return Ok(case);
        }
        ensure!(visible.len() <= 89, "assay teacher prefix bound");
        let output = readout(state, store, &visible)?;
        let loss = output.loss(target)?;
        ensure!(loss >= 0.0, "negative assay NLL");
        if index == 0 {
            let target_logit = output.logits[usize::from(target)];
            case.first_byte_rank = Some(
                1 + output
                    .logits
                    .iter()
                    .enumerate()
                    .filter(|(id, value)| {
                        **value > target_logit
                            || (**value == target_logit && *id < usize::from(target))
                    })
                    .count(),
            );
        }
        if index == 4 {
            case.lf_nll = Some(loss);
        } else {
            case.content_nll.push(loss);
        }
        visible.push(target);
    }
    if generate {
        visible = row.prompt.clone();
        let mut raw = Vec::with_capacity(8);
        for _ in 0..8 {
            if poll()?.is_some() {
                case.generated = Some(raw);
                return Ok(case);
            }
            let output = readout(state, store, &visible)?;
            let byte = u8::try_from(super::generation::lowest_maximum(&output.logits)?)?;
            raw.push(byte);
            visible.push(byte);
            if byte == b'\n' {
                break;
            }
        }
        case.exact = Some(raw == row.completion);
        case.generated = Some(raw);
        case.generation_complete = true;
    }
    Ok(case)
}

impl Case {
    pub fn teacher_count(&self) -> u64 {
        self.content_nll.len() as u64 + u64::from(self.lf_nll.is_some())
    }
    pub fn generated_count(&self) -> u64 {
        self.generated.as_ref().map_or(0, |raw| raw.len() as u64)
    }
    pub fn complete(&self, generate: bool) -> bool {
        self.teacher_count() == 5 && (!generate || self.generation_complete)
    }
}
