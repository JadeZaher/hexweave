use super::{probe::Pair, Predictor};
use anyhow::Result;

pub(super) use super::super::super::predictor_joint::choose;

pub(super) fn apply(
    current: &mut Predictor,
    pairs: &[Pair],
) -> Result<super::super::super::predictor_joint::Update> {
    let borrowed: Vec<_> = pairs
        .iter()
        .map(|pair| (pair.features.as_slice(), pair.correction.as_slice()))
        .collect();
    current.train_joint(&borrowed)
}
