//! Exact retained input for the existing forbidden-trigger controller. This
//! supplies source custody, not model admission or a replacement sampler.
mod controller;
mod startup;

use super::{trigger::TriggerPrefix, ConstraintController, ConstraintRuntime};
use eredu_core::speculative::{
    ForbiddenControllerError, ForbiddenControllerInputs, ForbiddenControllerSource,
};
use eredu_core::{
    speculative::{PlainControllerError, PlainControllerHistory},
    HostPreparationAuthority, SharedTokenFilter,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error(transparent)]
    Selection(#[from] super::selection::Error),
    #[error("this controller requires a separately qualified grammar source")]
    GrammarSource,
    #[error(transparent)]
    Original(#[from] eredu_runtime::working_memory::OriginalForbiddenSourceError),
    #[error("forbidden-controller source geometry overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    History(#[from] PlainControllerError),
    #[error(transparent)]
    Forbidden(#[from] ForbiddenControllerError),
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ForbiddenSourceError {
    #[source]
    cause: Cause,
    // No error allocation is introduced. An escaped partial-copy error keeps
    // the actual cumulative source account after its owned cause retires.
    funding: HostMetadataFunding,
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
        size_of::<ForbiddenSourceError>(),
        size_of::<Cause>(),
        size_of::<(&ConstraintController, &HostMetadataFunding)>(),
        size_of::<&HostMetadataFunding>(), // error-retention closure
        size_of::<Result<(), eredu_core::generation::GenerationError>>(),
        size_of::<Result<HostPreparationAuthority, Cause>>(),
        size_of::<TriggerPrefix>(),
        size_of::<SharedTokenFilter>(),
        size_of::<HostMetadataFunding>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<Option<usize>>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<eredu_core::generation::GenerationError>(),
    ])
}

#[cfg(test)]
mod tests;
