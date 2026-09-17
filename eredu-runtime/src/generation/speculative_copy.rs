//! Closed source-bound host copies for known speculative policies.
use super::*;
use std::mem::{size_of, size_of_val};

/// Exact borrowed sampler copy, including the complete retained history box.
/// Only known grammar-free policies construct this witness. Mutable commitment
/// requires its own independent prepared producer; this copy never certifies it.
/// It supplies no funding or native permission; the caller must pay before copy
/// and retain custody until the returned policy and all enclosing controls retire.
#[derive(Debug)]
pub struct PreparedSpeculativeSamplerCopy<'a, S> {
    source: &'a S,
    copy: fn(&S) -> S,
    bytes: usize,
}
impl<S> PreparedSpeculativeSamplerCopy<'_, S> {
    /// Exact source payload and actual copy-worker controls.
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    /// Copies the same immutable source inspected above after caller admission.
    pub fn copy(self) -> S {
        (self.copy)(self.source)
    }
    /// These concrete policies inherit the ordinary grammar-free behavior.
    /// No caller constraint callback is invoked or certified by this query.
    pub fn grammar_is_complete(&self) -> bool {
        false
    }
}
fn controls<S>(payload: usize, worker: usize) -> Option<usize> {
    let parts = [
        payload,
        worker,
        size_of::<S>(),
        size_of::<PreparedSpeculativeSamplerCopy<'_, S>>(),
        size_of::<Option<PreparedSpeculativeSamplerCopy<'_, S>>>(),
        size_of::<&S>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn default(
    source: &DefaultSampler,
) -> PreparedSpeculativeSamplerCopy<'_, DefaultSampler> {
    PreparedSpeculativeSamplerCopy {
        source,
        copy: |source| *source,
        bytes: controls::<DefaultSampler>(0, 0).expect("fixed sampler controls fit"),
    }
}
pub(super) fn standard(
    source: &GenerationSampler,
) -> Option<PreparedSpeculativeSamplerCopy<'_, GenerationSampler>> {
    let history = source.generated_tokens.prepare_copy()?;
    Some(PreparedSpeculativeSamplerCopy {
        source,
        // The existing Clone uses this same fixed box copy; no history growth.
        copy: Clone::clone,
        bytes: controls::<GenerationSampler>(
            usize::try_from(history.bytes()).ok()?,
            history_controls()?,
        )?,
    })
}
pub(super) fn configured(
    source: &ConfiguredTextSampler,
) -> Option<PreparedSpeculativeSamplerCopy<'_, ConfiguredTextSampler>> {
    let plan = source.prepare_copy().ok()?;
    Some(PreparedSpeculativeSamplerCopy {
        source,
        copy: Clone::clone,
        bytes: controls::<ConfiguredTextSampler>(
            usize::try_from(plan.history_bytes()).ok()?,
            size_of::<SamplerCopyPlan<'_>>()
                .checked_add(size_of::<Result<SamplerCopyPlan<'_>, SamplerCopyError>>())?
                .checked_add(history_controls()?)?,
        )?,
    })
}

pub(super) fn adaptive(source: &MirostatV2Sampler) -> Option<PreparedSpeculativeSamplerCopy<'_, MirostatV2Sampler>> {
    let history = source.penalties.generated_tokens.prepare_copy()?;
    Some(PreparedSpeculativeSamplerCopy { source, copy: Clone::clone,
        bytes: controls::<MirostatV2Sampler>(usize::try_from(history.bytes()).ok()?, history_controls()?)? })
}

fn history_controls() -> Option<usize> {
    let parts = [
        size_of::<history::TokenHistoryCopy<'_>>(),
        size_of::<Option<history::TokenHistoryCopy<'_>>>(),
        size_of::<history::TokenHistory>(),
        size_of::<Box<[u32]>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
