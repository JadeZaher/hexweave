//! A visit's rebuilt activation cache and its post-visit RMS normalization; see STUDY.md.

use super::super::{attention, ffn, norm, pulse};
use anyhow::Result;
use std::borrow::Cow;

pub(super) struct Cache {
    pub(super) attention: attention::Cache,
    pub(super) normalized: norm::Rms,
    pub(super) ffn: ffn::Cache,
    pub(super) pulse: pulse::Cache,
    /// Raw residual-stream output of the visit.
    pub(super) output: Vec<f32>,
    /// Post-visit RMS normalization of the residual stream; Some only under connected `visit_norm`.
    pub(super) output_norm: Option<norm::Rms>,
}

impl Cache {
    /// The output a visit exposes: RMS-normalized under connected `visit_norm`, raw otherwise.
    pub(super) fn visible(&self) -> &[f32] {
        self.output_norm
            .as_ref()
            .map_or(&self.output, |rms| &rms.values)
    }

    /// RMS of the raw output: the normalization's `1/inverse` when normalized, else measured.
    pub(super) fn raw_rms(&self) -> f32 {
        match &self.output_norm {
            Some(rms) => rms.inverse.recip(),
            None => {
                let square: f32 = self.output.iter().map(|v| v * v).sum();
                (square / self.output.len() as f32).sqrt()
            }
        }
    }

    /// Map a visible-output adjoint onto the raw output; the identity without normalization.
    pub(super) fn raw_adjoint<'a>(&self, upstream: &'a [f32]) -> Result<Cow<'a, [f32]>> {
        Ok(match &self.output_norm {
            Some(rms) => Cow::Owned(norm::backward(&rms.values, rms.inverse, upstream)?),
            None => Cow::Borrowed(upstream),
        })
    }
}
