//! Exact retained input for the existing forbidden-trigger controller. This
//! supplies source custody, not model admission or a replacement sampler.
mod controller;
mod startup;
pub(super) use controller::ordinary_error;

use super::{
    ConstraintController, ConstraintRuntime,
    trigger::{self, TriggerPrefix},
    vocabulary::VocabularyLayout,
};
use eredu_core::speculative::{
    ForbiddenControllerError, ForbiddenControllerInputs, ForbiddenControllerSource,
};
use eredu_core::{
    HostPreparationAuthority, SharedTokenFilter, SpeculativeBuffer,
    SpeculativeBufferAllocationError,
    speculative::{PlainControllerError, PlainControllerHistory},
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error(transparent)]
    Selection(#[from] super::selection::Error),
    #[error("this controller requires a separately qualified grammar source")]
    GrammarSource,
    #[error("prepared plan has no validated frozen tokenizer object")]
    TokenizerRecipe,
    #[error(transparent)]
    Tokenizer(#[from] eredu_text::tokenizer_storage::TokenizerSourceError),
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    #[error(transparent)]
    Original(#[from] eredu_runtime::working_memory::OriginalForbiddenSourceError),
    #[error("controller has no exact forbidden-trigger source")]
    Source,
    #[error("forbidden-controller source geometry overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] SpeculativeBufferAllocationError),
    #[error(transparent)]
    History(#[from] PlainControllerError),
    #[error(transparent)]
    Forbidden(#[from] ForbiddenControllerError),
    #[error("forbidden-controller destination capacity changed")]
    Destination,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ForbiddenSourceError {
    #[source]
    cause: Cause,
    // No error allocation is introduced. An escaped partial-copy error keeps
    // the actual cumulative source account after its owned cause retires.
    funding: WorkspaceMetadataFunding,
}

/// An immutable, source-derived vocabulary, trigger and durable history copy.
/// The tokenizer validity alias keeps its independent source owner; downstream
/// admission must authenticate it rather than treating this copy as its grant.
pub(crate) struct PreparedForbiddenSource {
    inputs: ForbiddenControllerInputs,
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
    prefix: TriggerPrefix,
    authority: HostPreparationAuthority,
    funding: WorkspaceMetadataFunding,
}

impl std::fmt::Debug for PreparedForbiddenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedForbiddenSource")
            .field("tokens", &self.inputs.vocabulary_len())
            .field("trigger_bytes", &self.inputs.trigger().len())
            .field("history", &self.history.len())
            .finish_non_exhaustive()
    }
}

impl PreparedForbiddenSource {
    pub(crate) fn source(&self) -> Result<ForbiddenControllerSource<'_>, ForbiddenControllerError> {
        ForbiddenControllerSource::new(&self.history, &self.validity, &self.inputs, self.prefix)
    }
    pub(crate) fn history(&self) -> &[u32] {
        &self.history
    }
    pub(crate) fn validity(&self) -> &SharedTokenFilter {
        &self.validity
    }
    pub(crate) fn vocabulary_len(&self) -> usize {
        self.inputs.vocabulary_len()
    }
    pub(crate) fn token_bytes(&self, token: usize) -> Option<&[u8]> {
        self.inputs.token_bytes(token)
    }
    pub(crate) fn trigger(&self) -> &[u8] {
        self.inputs.trigger()
    }
    pub(crate) fn pending(&self) -> &[u8] {
        self.prefix.bytes(self.inputs.trigger())
    }
    /// The exact ordinary candidate predicate; no filter Vec or activation
    /// payload is constructed by this borrowed inspection.
    pub(crate) fn allows(&self, token: u32) -> bool {
        self.validity.allows(token)
            && self
                .token_bytes(token as usize)
                .is_some_and(|bytes| trigger::find(self.pending(), bytes, self.trigger()).is_none())
    }
}

fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}

/// Fixed source/error transport is admitted before validating or quoting the
/// dynamic payload, so later refusal does not allocate a new error wrapper.
fn controls() -> Option<usize> {
    sum(&[
        size_of::<PreparedForbiddenSource>(),
        size_of::<ForbiddenSourceError>(),
        size_of::<Cause>(),
        size_of::<Result<PreparedForbiddenSource, ForbiddenSourceError>>(),
        size_of::<Result<PreparedForbiddenSource, Cause>>(),
        size_of::<(&ConstraintController, &WorkspaceMetadataFunding)>(),
        size_of::<&WorkspaceMetadataFunding>(), // error-retention closure
        size_of::<Result<(), eredu_core::generation::GenerationError>>(),
        size_of::<Result<HostPreparationAuthority, Cause>>(),
        size_of::<(VocabularyLayout, &[u8])>(),
        size_of::<VocabularyLayout>(),
        size_of::<TriggerPrefix>(),
        size_of::<SharedTokenFilter>(),
        size_of::<WorkspaceMetadataFunding>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<Option<usize>>(),
        size_of::<Result<(), WorkspaceMetadataFundingError>>(),
        size_of::<eredu_core::generation::GenerationError>(),
    ])
}

impl ConstraintController {
    /// Copies only a concrete ToolChoice::None runtime into original paid
    /// destinations. Text/Auto/Active keep their separate source requirements.
    /// This does not change inference_storage, prepared_plain_source or any
    /// public prepared/managed gate.
    pub(crate) fn prepare_forbidden_source(
        &self,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<PreparedForbiddenSource, ForbiddenSourceError> {
        self.prepare_forbidden_source_with_capacity(funding, self.committed_tokens.len())
    }
    fn prepare_forbidden_source_with_capacity(
        &self,
        funding: &WorkspaceMetadataFunding,
        capacity: usize,
    ) -> Result<PreparedForbiddenSource, ForbiddenSourceError> {
        let retain = |cause| ForbiddenSourceError {
            cause,
            funding: funding.clone(),
        };
        funding
            .reserve_metadata(controls().ok_or_else(|| retain(Cause::Overflow))?)
            .map_err(|error| retain(Cause::Funding(error)))?;
        let result = (|| {
            let ConstraintRuntime::Forbidden {
                vocabulary,
                trigger,
                pending,
            } = &self.runtime
            else {
                return Err(Cause::Source);
            };
            if trigger.is_empty() || capacity < self.committed_tokens.len() {
                return Err(Cause::Source);
            }
            let (layout, packed) = vocabulary.source();
            let bytes = sum(&[
                ForbiddenControllerInputs::copy_metadata_bytes(packed.len(), trigger.len())
                    .ok_or(Cause::Overflow)?,
                PlainControllerHistory::copy_metadata_bytes(capacity).ok_or(Cause::Overflow)?,
                HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
                    .ok_or(Cause::Overflow)?,
                size_of::<std::iter::Copied<std::slice::Iter<'_, u8>>>(),
            ])
            .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(bytes)?;
            let host = HostPreparationAuthority::retain(funding.clone());
            let inputs = ForbiddenControllerInputs::copy_prepared(
                packed,
                layout.len(),
                layout.maximum(),
                trigger,
                host.clone(),
            )?;
            let history = self
                .committed_tokens
                .copy_prepared(capacity, host.clone())?;
            Ok(PreparedForbiddenSource {
                inputs,
                history,
                validity: self.validity.clone(),
                prefix: *pending,
                authority: host,
                funding: funding.clone(),
            })
        })();
        result.map_err(retain)
    }
}

#[cfg(test)]
mod tests;
