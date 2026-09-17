//! Allocation-free source construction facts for the same saved-copy driver.
use super::*;
use crate::backend::runtime::cache::state::ResidentDecoderPreparationError;
use eredu_core::{BackendFailure, HostPreparationAuthority, SessionAuthorityError};
use eredu_runtime::generation::{SamplerCopyError, SamplerCopyPlan};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session::model_session::saved_array_copy) enum PendingDescriptorCause
{
    #[error(transparent)]
    Native(#[from] safemlx::ArrayDescriptorError),
    #[error("pending snapshot token must be one integer scalar")]
    Geometry,
}

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session) enum SourcePreparationCause {
    #[error("{0}")]
    Capture(
        #[from]
        #[source]
        eredu_runtime::capture::FundedCaptureCheckpointError,
    ),
    #[error(transparent)]
    Prompt(#[from] super::super::pending_input::PromptCopyCause),
    #[error(transparent)]
    NativeCopy(#[from] original::CopyPreparationCause),
    #[error(transparent)]
    Decoder(#[from] ResidentDecoderPreparationError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Authority(#[from] SessionAuthorityError),
    #[error("snapshot session authority is borrowed")]
    Borrowed(#[from] std::cell::BorrowError),
    #[error(transparent)]
    Sampler(#[from] SamplerCopyError),
    #[error(transparent)]
    Pending(#[from] PendingDescriptorCause),
    #[error("snapshot source does not match this backend target")]
    Target,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourcePreparationFailure {
    #[source]
    cause: SourcePreparationCause,
    // Closed BackendFailure deallocates the source shell before its last token.
    _host: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceOperationFailure {
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}

/// Construction-only preflight. Request/receipt/frontier authorization still
/// runs in prepare_live_inner and copy_inner after independent host admission.
/// This query neither removes stored failures nor acquires submission authority.
pub(super) fn inspect_live(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    sampling: &super::super::super::super::generation::MlxTextSamplingState,
    pending: Option<&MlxTextToken>,
    capture: Option<&eredu_core::capture::SharedCapturePlan>,
) -> Result<(), SourcePreparationCause> {
    let session = runtime.session();
    if session.poison.get() {
        return Err(WorkingMemoryError::ExecutionFenced.into());
    }
    if !runtime
        .backend()
        .matches_prepared_target(&session.payload.target)
    {
        return Err(SourcePreparationCause::Target);
    }
    session.authority.try_borrow()?.require_idle()?;
    sampling
        .quote
        .as_ref()
        .ok_or(WorkingMemoryError::UnknownBound)?
        .validate_paired_capture_copy_with_source(capture)?;
    let _ = sampling.sampler.borrow_funded()?;
    sampling.sampler.as_sampler().prepare_copy()?;
    if let Some(token) = pending {
        if !token.owner.is_healthy() || !token.owner.resources_releasable() {
            return Err(WorkingMemoryError::ExecutionFenced.into());
        }
        TextArraySource::validate_pending_descriptor(&token.value)?;
    }
    Ok(())
}

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        super::super::capture::handoff_control_bytes()?,
        size_of::<SourcePreparationCause>(),
        size_of::<Result<(), SourcePreparationCause>>(),
        size_of::<Result<PreparedResidentDecoderCopy<'_>, ResidentDecoderPreparationError>>(),
        size_of::<Result<u64, ResidentDecoderPreparationError>>(),
        size_of::<Result<SamplerCopyPlan<'_>, SamplerCopyError>>(),
        size_of::<std::cell::Ref<'_, eredu_core::SessionAuthority>>(),
        size_of::<Result<std::cell::Ref<'_, eredu_core::SessionAuthority>, std::cell::BorrowError>>(
        ),
        size_of::<Result<(), SessionAuthorityError>>(),
        size_of::<Result<super::super::BorrowedFundedSampler<'_>, WorkingMemoryError>>(),
        Array::descriptor_control_bytes()?,
        size_of::<safemlx::OwnedArrayDescriptorLoan<'_>>(),
        size_of::<Result<safemlx::OwnedArrayDescriptorLoan<'_>, safemlx::ArrayDescriptorError>>(),
        size_of::<Result<(), PendingDescriptorCause>>(),
        size_of::<Option<HostPreparationAuthority>>(),
        size_of::<Result<PreparedTextComponentsCopy<'_>, Error>>(),
        size_of::<Result<CopiedTextComponents, Error>>(),
        BackendFailure::source_retention_peak_bytes::<SourcePreparationFailure>()?,
        // Finite metadata construction retains a cause before returning it;
        // the public prepared-copy handoff then retains that same error owner.
        BackendFailure::source_retention_peak_bytes::<SourceOperationFailure>()?,
        BackendFailure::source_retention_peak_bytes::<SourceOperationFailure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(in crate::composition::mlx::session) fn retain_failure(
    cause: SourcePreparationCause,
    authority: &HostPreparationAuthority,
) -> Error {
    Error::StorageSource(BackendFailure::from_error(SourcePreparationFailure {
        cause,
        _host: authority.clone(),
    }))
}
#[track_caller]
pub(super) fn retain_operation_failure(
    cause: Error,
    authority: &HostPreparationAuthority,
) -> Error {
    if std::env::var_os("EREDU_HOST_SAVED_SOURCE_DIAGNOSTICS").is_some() {
        eprintln!(
            "HOST_SAVED_OPERATION_FAILURE at {}: {cause:?}",
            std::panic::Location::caller()
        );
    }
    Error::StorageSource(BackendFailure::from_error(SourceOperationFailure {
        cause,
        _host: authority.clone(),
    }))
}

impl PreparedTextComponentsCopy<'_> {
    /// Cold composition of source metadata, tables/accounts, native execution
    /// and destination publication for the same pinned source. The public
    /// original snapshot hook remains a separate qualification decision.
    pub(in crate::composition::mlx::session) fn known_preparation_component_bytes(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::super::super::generation::MlxTextSamplingState,
        pending: Option<&MlxTextToken>,
    ) -> Result<usize, WorkingMemoryError> {
        Self::known_input_preparation_component_bytes(
            runtime,
            sampling,
            pending.map(eredu_core::PendingTextInput::Decode),
        )
    }

    pub(in crate::composition::mlx::session) fn known_input_preparation_component_bytes(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::super::super::generation::MlxTextSamplingState,
        pending: Option<eredu_core::PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Result<usize, WorkingMemoryError> {
        Self::known_input_preparation_with_capture_bytes(runtime, sampling, pending, None)
    }

    pub(in crate::composition::mlx::session) fn known_input_preparation_with_capture_bytes(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::super::super::generation::MlxTextSamplingState,
        pending: Option<eredu_core::PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        capture: Option<&eredu_core::capture::SharedCapturePlan>,
    ) -> Result<usize, WorkingMemoryError> {
        let (decode, prompt) = match pending {
            None => (None, None),
            Some(eredu_core::PendingTextInput::Decode(token)) => (Some(token), None),
            Some(eredu_core::PendingTextInput::Prefill(prompt)) => (
                None,
                Some(
                    super::super::pending_input::PromptCopySource::prepare(sampling, prompt)
                        .map_err(|cause| match cause {
                            super::super::pending_input::PromptCopyCause::Memory(cause) => cause,
                            _ => WorkingMemoryError::UnknownBound,
                        })?,
                ),
            ),
        };
        inspect_live(runtime, sampling, decode, capture).map_err(|cause| {
            diagnose("live-source", &cause);
            match cause {
                SourcePreparationCause::Memory(cause) => cause,
                SourcePreparationCause::Sampler(SamplerCopyError::Overflow) => {
                    WorkingMemoryError::Overflow
                }
                _ => WorkingMemoryError::UnknownBound,
            }
        })?;
        let owner = DecoderCopyOwner::Live(runtime.session().payload.clone());
        let decoder = owner.prepare_fixed().map_err(|cause| {
            diagnose("decoder-source", &cause);
            match cause {
                ResidentDecoderPreparationError::Memory(cause) => cause,
                _ => WorkingMemoryError::UnknownBound,
            }
        })?;
        let mechanisms = runtime
            .session()
            .payload
            .model
            .workspace_mechanisms()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let plan = finite_preparation::SnapshotFinitePreparation::inspect(
            &owner,
            &decoder,
            sampling.prng.as_ref().map(|key| key.as_array()),
            prompt
                .and_then(|prompt| prompt.tokens())
                .or_else(|| decode.map(|token| &token.value)),
            runtime.backend().memory_pool(),
            mechanisms,
            runtime.backend(),
        )
        .map_err(|cause| {
            diagnose("finite-copy", &cause);
            match cause {
                finite_preparation::FinitePreparationCause::Memory(cause) => cause,
                finite_preparation::FinitePreparationCause::Overflow => {
                    WorkingMemoryError::Overflow
                }
                _ => WorkingMemoryError::UnknownBound,
            }
        })?;
        Ok(plan.known_component_bytes())
    }
}

// Temporary opt-in diagnostic while the reached public Host lifecycle is being
// localized. It changes no source/admission decision and runs only on refusal.
fn diagnose(stage: &str, cause: &impl std::fmt::Debug) {
    if std::env::var_os("EREDU_HOST_SAVED_SOURCE_DIAGNOSTICS").is_some() {
        eprintln!("HOST_SAVED_SOURCE_FAILURE {stage}: {cause:?}");
    }
}
