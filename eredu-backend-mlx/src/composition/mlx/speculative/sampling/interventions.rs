//! Atomic static edit installation from the actual loaded declaration source.
use super::capture::OriginalCapture;
use super::*;
use crate::composition::mlx::{
    session::intervention::NativeInterventionEstimator,
    speculative::autoregressive::MlxAutoregressiveMechanisms,
};
use eredu_core::{
    BackendFailure,
    capture::CapturePlanCopyError,
    intervention::*,
    speculative::{SpeculativeCaptureRole, SpeculativeControlError, SpeculativeInterventionPlan},
};
use eredu_runtime::{
    intervention::{StaticInterventionPreflight, StaticInterventionScratchError},
    speculative::autoregressive::AutoregressiveMechanisms,
    working_memory::{
        OriginalInterventionSource, OriginalInterventionSourceError, WorkingMemoryError,
    },
};
use std::mem::{size_of, size_of_val};
/// Fixed source slots retire before their actual installation funding. Cloning
/// aliases existing immutable owners and performs no destination allocation.
#[derive(Clone)]
pub(super) struct OriginalInterventions {
    sources: [Option<OriginalInterventionSource>; 2],
    _host: eredu_core::HostPreparationAuthority,
}
type Slots = Option<OriginalInterventions>;
#[derive(Debug, thiserror::Error)]
enum Refusal {
    #[error("supply at most one intervention plan for each model role")]
    Roles,
    #[error("original static edits require the explicit capture source")]
    Capture,
    #[error("this static source requires the model.logits activation producer")]
    Profile,
}
fn transport(cause: Error) -> SpeculativeControlError {
    SpeculativeControlError::backend_with_retained(cause, Error::take_retained_backend_failure)
}
fn wrong() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn slot(role: SpeculativeCaptureRole) -> usize {
    match role {
        SpeculativeCaptureRole::Target => 0,
        SpeculativeCaptureRole::Draft => 1,
    }
}
pub(super) fn for_placement(
    sources: &Slots,
    placement: SamplingPlacement,
) -> Option<&OriginalInterventionSource> {
    let index = match placement {
        SamplingPlacement::Target => 0,
        SamplingPlacement::Draft => 1,
        _ => return None,
    };
    sources.as_ref()?.sources[index].as_ref()
}
fn controls() -> Option<usize> {
    let frames = [
        size_of::<OriginalInterventions>(),
        size_of::<Slots>(),
        size_of::<Result<Slots, SpeculativeControlError>>(),
        size_of::<&[SpeculativeInterventionPlan]>(),
        size_of::<std::slice::Iter<'_, SpeculativeInterventionPlan>>(),
        size_of::<[bool; 2]>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<Result<(), SpeculativeControlError>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<PreparedInterventionPlanCopy<'_>, CapturePlanCopyError>>(),
        size_of::<Result<OriginalInterventionSource, OriginalInterventionSourceError>>(),
        size_of::<Result<StaticInterventionPreflight, StaticInterventionScratchError>>(),
        PreparedInterventionPlanCopy::inspection_control_bytes()?,
        OriginalInterventionSource::validation_control_bytes()?,
        OriginalCapture::preflight_control_bytes()?,
        NativeInterventionEstimator::prepared_preflight_control_bytes()?,
        BackendFailure::source_retention_peak_bytes::<Error>()?,
        BackendFailure::source_retention_peak_bytes::<Refusal>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
/// The caller DTO is only borrowed here; the retained loaded source supplies
/// declaration identity and selected mechanism support. No source is adopted.
pub(super) fn validate(
    plans: &[SpeculativeInterventionPlan],
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), SpeculativeControlError> {
    let (sources, environment) = context
        .original_numerical()
        .ok_or_else(|| transport(wrong()))?;
    sources
        .validate_environment(environment)
        .map_err(transport)?;
    let _host = MlxAutoregressiveMechanisms::driver_host_metadata(controls(), context)
        .map_err(transport)?;
    if plans.len() > 2 {
        return Err(transport(sources.retain_startup_error(Refusal::Roles)));
    }
    let mut seen = [false; 2];
    for plan in plans {
        let index = slot(plan.role);
        if std::mem::replace(&mut seen[index], true) {
            return Err(transport(sources.retain_startup_error(Refusal::Roles)));
        }
        sources
            .validate_intervention(&plan.plan)
            .map_err(transport)?;
        if plan.plan.points().iter().any(|point| {
            point.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH
                || point.routing.is_some()
                || point.routed_units.is_some()
        }) || plan.plan.plan().operations.iter().any(|operation| {
            !matches!(
                operation.action,
                InterventionAction::Zero { .. }
                    | InterventionAction::Scale { .. }
                    | InterventionAction::Mask { .. }
                    | InterventionAction::MaskComponents { .. }
                    | InterventionAction::Replace { .. }
                    | InterventionAction::Add { .. }
                    | InterventionAction::MaskLogits { .. }
            )
        }) {
            return Err(transport(sources.retain_startup_error(Refusal::Profile)));
        }
    }
    Ok(())
}
/// Build both candidate slots before replacement. Scratch uses fresh H-paid
/// destinations; immutable payloads use the existing C compiler independently.
pub(super) fn prepare(
    plans: &[SpeculativeInterventionPlan],
    capture: Option<&OriginalCapture>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<Slots, SpeculativeControlError> {
    validate(plans, context)?;
    let (sources, environment) = context
        .original_numerical()
        .ok_or_else(|| transport(wrong()))?;
    let host = MlxAutoregressiveMechanisms::driver_host_metadata(
        controls().and_then(|n| n.checked_add(StaticInterventionPreflight::required_bytes()?)),
        context,
    )
    .map_err(transport)?;
    let mut next = [None, None];
    if plans.is_empty() {
        return Ok(None);
    }
    let capture =
        capture.ok_or_else(|| transport(sources.retain_startup_error(Refusal::Capture)))?;
    let mut scratch = StaticInterventionPreflight::prepare(host.clone())
        .map_err(|cause| transport(sources.retain_startup_error(cause)))?;
    for plan in plans {
        let copy = PreparedInterventionPlanCopy::inspect(&plan.plan)
            .map_err(|cause| transport(sources.retain_startup_error(cause)))?;
        let source = environment
            .pool()
            .compile_intervention_source(copy)
            .map_err(|cause| transport(sources.retain_startup_error(cause)))?;
        source
            .validate_pool(environment.pool())
            .map_err(|cause| transport(sources.retain_startup_error(cause)))?;
        // This candidate's exact immutable companions are now paid. Budget
        // refusal drops the candidate and preserves both installed role slots.
        capture
            .preflight_intervention(&source, &mut scratch, context)
            .map_err(transport)?;
        next[slot(plan.role)] = Some(source);
    }
    Ok(Some(OriginalInterventions {
        sources: next,
        _host: host,
    }))
}
