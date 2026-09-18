//! Exact borrowed token input through the existing source-bound I/full-B compiler.
use super::{MlxBackend, MlxModelInput, MlxPreparedInputMaterializer, MlxPreparedModelInputError};
use eredu_core::{BackendFailure, BackendFailureKind, ModelRuntime, TokenInputRejection};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    input::host::{
        HostInputPart, HostInputPlanError, HostTensorValues, HostTensorView, PreparedHostInputPlan,
    },
    working_memory::{
        OriginalPreparedHostInput, OriginalPreparedHostInputError,
        PreparedSemanticSource, WorkingMemoryError,
    },
};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU64,
};

type InitializerError =
    crate::backend::managed_memory::input_allocator::MlxInputAllocatorInitializationError;
type RetiredNativeError = super::original_host_input::RetiredMlxPreparedModelInputError;

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<E: std::error::Error> {
    #[source]
    cause: E,
    // The erased shell and every concrete cause retire before the caller H.
    _funding: HostMetadataFunding,
}
fn failure_controls<E: std::error::Error + Send + Sync + 'static>() -> Option<usize> {
    size_of::<Failure<E>>()
        .checked_add(BackendFailure::source_retention_peak_bytes::<Failure<E>>()?)?
        .checked_add(size_of::<Result<MlxModelInput, BackendFailure>>())
}
fn failure<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    kind: BackendFailureKind,
    funding: &HostMetadataFunding,
) -> BackendFailure {
    BackendFailure::new(
        kind,
        Failure {
            cause,
            _funding: funding.clone(),
        },
    ).with_operation("prepare original speculative prompt")
}
fn accounting_kind(cause: Option<&WorkingMemoryError>) -> BackendFailureKind {
    match cause {
        Some(
            WorkingMemoryError::BudgetExceeded { .. }
            | WorkingMemoryError::CapacityBelowUsage { .. }
            | WorkingMemoryError::Overflow,
        ) => BackendFailureKind::ResourceExhausted,
        Some(
            WorkingMemoryError::IdentityMismatch
            | WorkingMemoryError::ExecutionFenced
            | WorkingMemoryError::Poisoned,
        ) => BackendFailureKind::InvalidSession,
        Some(WorkingMemoryError::ReservedWorkActive) => BackendFailureKind::Busy,
        Some(WorkingMemoryError::UnknownBound) => BackendFailureKind::Unsupported,
        _ => BackendFailureKind::Other,
    }
}
fn control_bytes() -> Option<usize> {
    let parts = [
        failure_controls::<InitializerError>()?,
        failure_controls::<HostInputPlanError>()?,
        failure_controls::<OriginalPreparedHostInputError>()?,
        failure_controls::<WorkingMemoryError>()?,
        failure_controls::<RetiredNativeError>()?,
        MlxPreparedModelInputError::retirement_control_bytes()?,
        size_of::<MlxPreparedInputMaterializer>(),
        size_of::<Result<MlxPreparedInputMaterializer, InitializerError>>(),
        size_of::<[usize; 2]>(),
        size_of::<HostTensorValues<'_>>(),
        size_of::<HostTensorView<'_>>(),
        size_of::<[HostInputPart<'_>; 1]>(),
        size_of::<PreparedHostInputPlan<'_>>(),
        size_of::<Result<PreparedHostInputPlan<'_>, HostInputPlanError>>(),
        size_of::<OriginalPreparedHostInput>(),
        size_of::<Result<OriginalPreparedHostInput, OriginalPreparedHostInputError>>(),
        size_of::<MlxPreparedModelInputError>(),
        size_of::<RetiredNativeError>(),
        size_of::<MlxModelInput>(),
        size_of::<Result<MlxModelInput, BackendFailure>>(),
        size_of::<Option<NonZeroU64>>(),
        size_of::<(&eredu_core::TokenIdsInputPlan<'_>, &[u32])>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(super) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &PreparedSemanticSource,
    input: &eredu_core::TokenIdsInputPlan<'_>,
    chunk: Option<NonZeroU64>,
) -> Result<MlxModelInput, BackendFailure> {
    // These checks lend the actual prepared source, not caller-authorized
    // storage. Text encoding has already authenticated its E source; explicit
    // IDs still need this selected tokenizer's exact generation domain.
    if !runtime.session().matches_healthy_backend(runtime.backend()) {
        return Err(TokenInputRejection::IdentityMismatch.into_backend_failure());
    }
    let model = runtime.session().original_model_source().map_err(|cause| {
        match cause {
            WorkingMemoryError::ReservedWorkActive => TokenInputRejection::Busy,
            WorkingMemoryError::ExecutionFenced => TokenInputRejection::Unavailable,
            _ => TokenInputRejection::IdentityMismatch,
        }
        .into_backend_failure()
    })?;
    let pool = runtime.backend().memory_pool();
    preparation
        .validate(pool, model.erased().inference_execution_identity())
        .map_err(|_| TokenInputRejection::IdentityMismatch.into_backend_failure())?;
    let domain = preparation.tokenizer().generation_domain()
        .ok_or_else(|| TokenInputRejection::IdentityMismatch.into_backend_failure())?;
    let ids = input.tokens();
    if ids.iter().any(|&id| !domain.allows(id)) {
        return Err(TokenInputRejection::InvalidToken.into_backend_failure());
    }
    i32::try_from(ids.len()).map_err(|_| TokenInputRejection::Overflow.into_backend_failure())?;
    let funding = preparation.metadata_funding();
    funding
        .reserve_metadata(
            control_bytes()
                .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?,
        )
        .map_err(HostMetadataFundingError::into_backend_failure)?;

    // This reuses the actual admitted singleton; it never promotes ordinary
    // initialization. The header's existing domain ceiling also constrains I/B.
    let materializer = MlxPreparedInputMaterializer::prepare_admitted(pool)
        .map_err(|cause| failure(cause, BackendFailureKind::Other, funding))?;
    let shape = [1, ids.len()];
    let parts = [HostInputPart {
        modality: eredu_core::InputModality::Text,
        kind: eredu_core::InputPayloadKind::TokenIds,
        payload: HostTensorView {
            shape: &shape,
            values: HostTensorValues::U32(ids),
        },
        metadata: &[],
        extents: &[],
    }];
    let plan = PreparedHostInputPlan::prepare(&parts)
        .map_err(|cause| failure(cause, BackendFailureKind::InvalidInput, funding))?;
    let source = pool.compile_prepared_host_input(plan).map_err(|cause| {
        let kind = accounting_kind(cause.accounting_failure());
        failure(cause, kind, funding)
    })?;
    let plan = materializer.text_input_plan(&source).map_err(|cause| {
        let kind = accounting_kind(Some(&cause));
        failure(cause, kind, funding)
    })?;
    let input = plan.materialize(pool).map_err(|cause| {
        let kind = accounting_kind(cause.accounting_failure());
        // Concrete native prefix Drop may defer to its own retained B; the
        // public cause keeps accounting only and never transports a native arena.
        failure(cause.retire_storage(), kind, funding)
    })?;
    Ok(input.into_prompt(chunk))
}
