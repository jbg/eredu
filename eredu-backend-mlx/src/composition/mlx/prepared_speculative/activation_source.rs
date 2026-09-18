//! Exact loaded activation source at the existing controlled configuration hook.
use super::*;
use eredu_core::{
    capture::PreparedCapturePlanCopy,
    intervention::PreparedInterventionPlanCopy,
    speculative::{AdmittedSpeculativeActivations, SpeculativeControlError},
};
use eredu_nn::workspace::HostMetadataFundingError;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

pub(super) fn observer(
    plan: &AdmittedSpeculativeActivations,
    request: eredu_core::SpeculativeRequestId,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<
    Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Error>>>,
    SpeculativeControlError,
> {
    let (sources, environment) =
        context
            .original_numerical()
            .ok_or(SpeculativeControlError::Invalid(
                "missing original activation source",
            ))?;
    let result = (|| -> Result<_, Error> {
        sources.validate_environment(environment)?;
        let frames = [
            size_of::<(
                &AdmittedSpeculativeActivations,
                eredu_core::SpeculativeRequestId,
                SpeculativeExecutionStreams<'_>,
            )>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::OriginalCaptureSource,
                    eredu_runtime::working_memory::OriginalCaptureSourceError,
                >,
            >(),
            size_of::<
                Result<
                    Option<
                        Box<
                            dyn eredu_runtime::inspection::SpeculativeActivationObserver<
                                MlxTensor,
                                Error,
                            >,
                        >,
                    >,
                    Error,
                >,
            >(),
            size_of::<SpeculativeControlError>(),
            size_of::<Option<eredu_runtime::working_memory::OriginalInterventionSource>>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::OriginalInterventionSource,
                    eredu_runtime::working_memory::OriginalInterventionSourceError,
                >,
            >(),
            PreparedInterventionPlanCopy::inspection_control_bytes().ok_or(
                Error::WorkspacePlanning(HostMetadataFundingError::Overflow),
            )?,
            PreparedCapturePlanCopy::inspection_control_bytes().ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        sources
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        if context.original_embedded().is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        sources.validate_activations(plan)?;
        if plan.is_empty() {
            return Ok(None);
        }
        let copy = PreparedCapturePlanCopy::inspect(plan.captures())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let source = environment
            .pool()
            .compile_capture_source(copy)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let interventions = if plan.interventions().is_empty() {
            None
        } else {
            let copy = PreparedInterventionPlanCopy::inspect(plan.interventions())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            Some(
                environment
                    .pool()
                    .compile_intervention_source(copy)
                    .map_err(|cause| sources.retain_startup_error(cause))?,
            )
        };
        crate::composition::mlx::session::bounded_capture::original_speculative_capture_with_error(
            source,
            interventions,
            plan.intervention_scopes(),
            plan.capture_scopes(),
            plan.identity(),
            request,
            sources,
        )
    })();
    result.map_err(|cause| {
        SpeculativeControlError::Backend(sources.retain_error(cause).into_backend_failure())
    })
}
