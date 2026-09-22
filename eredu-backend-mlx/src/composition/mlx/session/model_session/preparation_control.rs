//! Original request source joins the same shared startup/readiness worker.
use super::*;
use crate::backend::distributed::MlxTextPreparationControl;
pub(super) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    sequence: &eredu_core::GenerationSequencePreparation<'_, '_>,
) -> Result<Option<MlxTextPreparationControl>, BackendFailure> {
    let session = runtime.session();
    let Some(transport) = session.payload.distributed.as_ref() else {
        return Ok(None);
    };
    let rejected = || eredu_core::TokenInputRejection::IdentityMismatch.into_backend_failure();
    session
        .validate_backend(runtime.backend())
        .map_err(Error::into_backend_failure)?;
    session
        .ensure_no_submission_in_flight()
        .map_err(Error::into_backend_failure)?;
    runtime
        .backend()
        .validate_original_stream_owners()
        .map_err(Error::into_backend_failure)?;
    if sequence.context().attempt() != 0
        || config.sampling().max_new_tokens != Some(sequence.request().max_new_tokens())
    {
        return Err(rejected());
    }
    let blueprint = session
        .payload
        .model
        .inference_blueprint()
        .ok_or_else(rejected)?;
    let manifest = blueprint
        .selected()
        .communication_manifest()
        .ok_or_else(rejected)?;
    let capacity = config
        .inference_policy()
        .memory_limits
        .resolve(runtime.backend().memory_ledger().topology())
        .map_err(BackendFailure::from_error)?;
    if !session
        .payload
        .memory_ledger
        .same_ledger(runtime.backend().memory_ledger())
    {
        return Err(rejected());
    }
    // This is readiness authority, not prepared-media admission. Construct it
    // before the original prepared-source worker validates exact packet, cache,
    // pool and sequence identity. That worker's refusal must participate in the
    // same Admission vote as a token-input refusal on another rank.
    match input {
        TextPreparationInput::OriginalTokenIds(plan)
            if sequence
                .request()
                .token_input()
                .is_some_and(|source| std::ptr::eq(source, *plan)) => {}
        TextPreparationInput::OriginalPrepared(_) => {}
        _ => return Err(eredu_core::TokenInputRejection::Unsupported.into_backend_failure()),
    }
    let source = transport
        .prepare_original_readiness(
            manifest,
            transport.native_world(),
            runtime.backend().memory_ledger(),
            session
                .payload
                .model
                .erased()
                .inference_execution_identity(),
            capacity,
            sequence.context(),
        )
        .map_err(Error::into_backend_failure)?;
    Ok(Some(source))
}
pub(super) fn agree(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    control: Option<&MlxTextPreparationControl>,
    stage: eredu_core::run_preparation::TextPreparationStage,
    status: eredu_core::run_preparation::TextPreparationStatus,
) -> Result<eredu_core::run_preparation::TextPreparationOutcome, BackendFailure> {
    use eredu_core::run_preparation::{
        TextPreparationOutcome as Outcome, TextPreparationStatus as Status,
    };
    let session = runtime.session();
    let valid = session
        .validate_backend(runtime.backend())
        .and_then(|()| session.ensure_no_submission_in_flight());
    #[cfg(all(
        test,
        target_vendor = "apple",
        feature = "metal",
        not(feature = "cuda")
    ))]
    if valid.is_ok()
        && stage == eredu_core::run_preparation::TextPreparationStage::Sampling
        && status == Status::Ready
    {
        saved_array_copy::decoder::resume_driver::failure_tests::checkpoint(
            saved_array_copy::decoder::resume_driver::failure_tests::Point::BeforeSamplingReadiness,
            runtime,
        )
        .map_err(BackendFailure::from_error)?;
    }
    let status = if valid.is_err() {
        Status::Failed
    } else {
        status
    };
    let agreed = match &session.payload.distributed {
        Some(transport) => match control {
            Some(source) => source.agree(
                transport,
                runtime.backend().memory_ledger(),
                session
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                stage,
                status,
            ),
            None => transport.agree_text_preparation(stage, status),
        },
        None if control.is_some() => {
            Err(eredu_core::TokenInputRejection::IdentityMismatch.into_backend_failure())
        }
        None => Ok(match status {
            Status::Ready => Outcome::Ready,
            Status::Cancelled => Outcome::Cancelled,
            Status::Failed => Outcome::Rejected { rank: 0 },
        }),
    };
    valid.map_err(|error| session.lifecycle_failure(error))?;
    agreed
}
