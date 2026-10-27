use super::execution::Counts;
use super::gradients::Gradients;
use super::history::{History, Record};
use super::math::clipped;
use super::node::Frame;
use super::shared::Shared;
use super::storage::Store;
use super::{RouteCredit, WIDTH};
use anyhow::Result;

pub(super) struct AnchorRequest<'a> {
    pub context: &'a [u8],
    pub target: usize,
    pub signal: &'a [f32],
    pub frames: &'a [Frame],
    pub baseline: f32,
    pub step: usize,
    pub predictor: bool,
    pub route_credit: RouteCredit,
}

pub(super) fn anchor(
    shared: &mut Shared,
    store: &mut Store,
    history: &mut History,
    counts: &mut Counts,
    request: AnchorRequest<'_>,
) -> Result<f32> {
    let output = &request.frames.last().unwrap().output;
    let (loss, derivative) = shared.task(output, request.target)?;
    let derivative: Vec<_> = derivative.iter().map(|value| value / 4.0).collect();
    let mut correction = shared.readout.transpose(&derivative)?;
    let mut corrections = vec![vec![0.0; WIDTH]; request.frames.len()];
    for (index, frame) in request.frames.iter().enumerate().rev() {
        corrections[index].clone_from(&correction);
        correction = store.with_node(frame.id, |node| node.returned(frame, &correction))?;
        counts.returned_frames += 1;
    }
    let records: Vec<_> = request
        .frames
        .iter()
        .zip(&corrections)
        .map(|(frame, correction)| {
            Ok(Record {
                features: shared.features(frame, request.signal, &store.mesh),
                correction: clipped(correction)?,
                versions: store.versions,
                shared_version: shared.version,
                step: request.step,
            })
        })
        .collect::<Result<_>>()?;
    commit_tiles(
        store,
        counts,
        request.frames,
        &corrections,
        loss - request.baseline,
        request.route_credit,
    )?;
    shared.update(request.context, &correction, output, &derivative)?;
    if request.predictor {
        for record in records {
            counts.predictor_error_sum +=
                shared.train_predictor(&record.features, &record.correction)?;
            counts.predictor_updates += 1;
            history.push(record);
        }
    }
    counts.anchor_targets += 1;
    Ok(loss)
}

fn commit_tiles(
    store: &mut Store,
    counts: &mut Counts,
    frames: &[Frame],
    corrections: &[Vec<f32>],
    advantage: f32,
    route_credit: RouteCredit,
) -> Result<()> {
    let mut selected: Vec<_> = frames.iter().map(|frame| frame.id).collect();
    selected.sort_unstable();
    selected.dedup();
    let mut anchor_different = false;
    for id in selected {
        let transitions = store.with_node(id, |node| {
            let mut gradients = Gradients::zeros(node);
            for (frame, correction) in frames.iter().zip(corrections) {
                if frame.id == id {
                    anyhow::ensure!(
                        frame.version == node.version,
                        "anchor commit version mismatch"
                    );
                    gradients.add(node, frame, correction, Some(advantage), route_credit)?;
                }
            }
            let observed = if counts.credit.is_some() {
                Some(super::credit_diagnostic::observe(
                    node,
                    frames,
                    corrections,
                    &gradients,
                    (id, advantage, route_credit),
                )?)
            } else {
                None
            };
            let transitions = node.apply(&gradients)?;
            if let (Some(diagnostic), Some(observed)) = (&mut counts.credit, observed) {
                anchor_different |= diagnostic.commit(id, observed, transitions[3])?;
            }
            Ok(transitions)
        })?;
        for (count, changed) in counts.transitions.iter_mut().zip(transitions) {
            *count += changed;
        }
    }
    if let Some(diagnostic) = &mut counts.credit {
        diagnostic.anchors += 1;
        diagnostic.anchors_with_different_results += usize::from(anchor_different);
    }
    Ok(())
}
