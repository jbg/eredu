//! Actual I/A/B compiler selected from the loaded blueprint, with no native readback.
use super::*;
use eredu_architectures::media_plan::{
    MediaSemanticError, OriginalPreparedMediaSemantics, PreparedMediaSemanticCompile,
};
use eredu_core::{BackendFailureKind, TokenInputRejection};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    input::{host::PreparedHostInputPlan, OriginalModelInput, OriginalModelInputPublicationError},
    working_memory::OriginalPreparedHostInputError,
};
use std::mem::{size_of, size_of_val};

type InitializerError = native::MlxInputAllocatorInitializationError;
type SemanticError = eredu_runtime::working_memory::OriginalCompositeSemanticStorageError;
type BindingError = eredu_runtime::replicated_session::OriginalMediaBindingError;
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<E: std::error::Error> {
    #[source]
    cause: E,
    funding: HostMetadataFunding,
}
fn failure<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: &HostMetadataFunding,
) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Other,
        Failure {
            cause,
            funding: funding.clone(),
        },
    )
    .with_operation("prepare original model input")
}
fn error_controls<E: std::error::Error + Send + Sync + 'static>() -> Option<usize> {
    size_of::<Failure<E>>()
        .checked_add(BackendFailure::source_retention_peak_bytes::<Failure<E>>()?)
}
fn controls() -> Option<usize> {
    let parts = [
        error_controls::<InitializerError>()?,
        error_controls::<WorkingMemoryError>()?,
        error_controls::<OriginalPreparedHostInputError>()?,
        error_controls::<MediaSemanticError>()?,
        error_controls::<SemanticError>()?,
        error_controls::<BindingError>()?,
        error_controls::<RetiredMlxPreparedModelInputError>()?,
        error_controls::<RetiredMlxPreparedModelInputBindError>()?,
        error_controls::<HostMetadataFundingError>()?,
        MlxPreparedModelInputError::retirement_control_bytes()?,
        MlxPreparedModelInputBindError::retirement_control_bytes()?,
        size_of::<MlxPreparedInputMaterializer>(),
        size_of::<Result<MlxPreparedInputMaterializer, InitializerError>>(),
        size_of::<PreparedHostInputPlan<'_>>(),
        size_of::<OriginalPreparedHostInput>(),
        size_of::<Result<OriginalPreparedHostInput, OriginalPreparedHostInputError>>(),
        size_of::<PreparedMediaSemanticCompile<'_, '_>>(),
        size_of::<Result<PreparedMediaSemanticCompile<'_, '_>, MediaSemanticError>>(),
        size_of::<OriginalPreparedMediaSemantics<'_>>(),
        size_of::<Result<OriginalPreparedMediaSemantics<'_>, SemanticError>>(),
        size_of::<MlxPreparedModelInputPlan<'_>>(),
        size_of::<Result<MlxPreparedModelInputPlan<'_>, WorkingMemoryError>>(),
        size_of::<MlxOriginalPreparedModelInput>(),
        size_of::<Result<MlxOriginalPreparedModelInput, MlxPreparedModelInputError>>(),
        size_of::<Result<MlxModelInput, MlxPreparedModelInputBindError>>(),
        size_of::<OriginalModelInput<MlxModelInput>>(),
        size_of::<
            Result<
                OriginalModelInput<MlxModelInput>,
                OriginalModelInputPublicationError<MlxModelInput>,
            >,
        >(),
        size_of::<Result<OriginalModelInput<MlxModelInput>, BackendFailure>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(in crate::composition::mlx::session::model_session) fn prepare_model_input(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    plan: PreparedHostInputPlan<'_>,
    capacity: eredu_core::MemoryLimits,
) -> Result<OriginalModelInput<MlxModelInput>, BackendFailure> {
    if !runtime.session().matches_healthy_backend(runtime.backend()) {
        return Err(TokenInputRejection::IdentityMismatch.into_backend_failure());
    }
    let model = runtime
        .session()
        .original_model_source()
        .map_err(|_| TokenInputRejection::Busy.into_backend_failure())?;
    let blueprint = model
        .inference_blueprint()
        .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
    let pool = runtime.backend().memory_ledger();
    let funding = pool
        .prepare_workspace_metadata(model.erased().inference_execution_identity(), capacity)
        .map_err(HostMetadataFundingError::into_backend_failure)?;
    funding
        .reserve_metadata(
            controls().ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?,
        )
        .map_err(HostMetadataFundingError::into_backend_failure)?;
    model
        .erased()
        .prepare_original_media_semantic_binding(&funding)
        .map_err(|cause| failure(cause, &funding))?;
    let materializer = MlxPreparedInputMaterializer::prepare_admitted(pool)
        .map_err(|cause| failure(cause, &funding))?;
    let source = pool
        .compile_prepared_host_input(plan)
        .map_err(|cause| failure(cause, &funding))?;
    let semantics = blueprint
        .plan_original_media_semantics(&source)
        .map_err(|cause| failure(cause, &funding))?
        .compile(pool)
        .map_err(|cause| failure(cause, &funding))?;
    let input = materializer
        .model_input_plan(&semantics)
        .map_err(|cause| failure(cause, &funding))?
        .materialize(pool)
        .map_err(|cause| failure(cause.retire_storage(), &funding))?;
    let input = input
        .bind(runtime, semantics)
        .map_err(|cause| failure(cause.retire_storage(), &funding))?;
    OriginalModelInput::try_new(input, funding).map_err(|cause| {
        let (cause, funding) = cause.retire();
        failure(cause, &funding)
    })
}
