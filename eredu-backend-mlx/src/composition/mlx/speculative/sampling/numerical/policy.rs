//! Actual immutable sampler-policy projection consumed by the numerical phase.
use super::*;
use crate::backend::runtime::generation::MlxSamplingBackend;
use eredu_runtime::generation::{
    PreparedLogitPolicy, PreparedLogitPolicyError, SpeculativeLogitProgram,
};

#[derive(Debug, thiserror::Error)]
enum PolicyCause {
    #[error("prepared logit policy requires the original request's source context")]
    Context,
    #[error("prepared logit policy returned a different numerical value kind")]
    Output,
    #[error("processed controller value differs from the current prepared source")]
    Source,
}

/// Only an actual closed policy projection may reach the native worker. The
/// supplied history remains borrowed through synchronous graph construction;
/// existing penalty constructors own their native uploads before this returns.
pub(crate) fn process_policy<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    history: &[u32],
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalValue, Error> {
    process_policy_at(
        policy,
        logits,
        temperature,
        history,
        SamplingPlacement::Target,
        context,
    )
}
pub(crate) fn process_policy_at<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    history: &[u32],
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<OriginalNumericalValue, Error> {
    let mut failed_capture = None;
    process_policy_inner(
        policy,
        logits,
        temperature,
        history,
        placement,
        context,
        None,
        &mut failed_capture,
    )
    .map(|(value, _)| value)
}
/// The actual published capture source is consumed by the same policy/mask
/// worker and numerical Scope. Success escapes after exact retirement; the
/// separate failed-capture destination receives only sealed host evidence while
/// the native error/recovery path retains unresolved work.
pub(crate) fn process_policy_with_capture<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    history: &[u32],
    context: SpeculativeExecutionStreams<'_>,
    capture: CaptureSource<'_>,
    failed_capture: &mut Option<eredu_core::capture::SharedCapturedStep>,
) -> Result<
    (
        OriginalNumericalValue,
        eredu_core::capture::SharedCapturedStep,
    ),
    Error,
> {
    process_policy_with_capture_at(
        policy,
        logits,
        temperature,
        history,
        SamplingPlacement::Target,
        context,
        capture,
        failed_capture,
    )
}
pub(crate) fn process_policy_with_capture_at<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    history: &[u32],
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
    capture: CaptureSource<'_>,
    failed_capture: &mut Option<eredu_core::capture::SharedCapturedStep>,
) -> Result<
    (
        OriginalNumericalValue,
        eredu_core::capture::SharedCapturedStep,
    ),
    Error,
> {
    let (value, frame) = process_policy_inner(
        policy,
        logits,
        temperature,
        history,
        placement,
        context,
        Some(capture),
        failed_capture,
    )?;
    let frame = frame.ok_or_else(|| {
        retain_planning_error(PolicyCause::Output, logits.value().funding.clone())
    })?;
    Ok((value, frame))
}
fn process_policy_inner<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    logits: &OriginalNumericalValue,
    temperature: f32,
    history: &[u32],
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
    capture: Option<CaptureSource<'_>>,
    failed_capture: &mut Option<eredu_core::capture::SharedCapturedStep>,
) -> Result<
    (
        OriginalNumericalValue,
        Option<eredu_core::capture::SharedCapturedStep>,
    ),
    Error,
> {
    let funding = &logits.value().funding;
    let result = (|| {
        let controls = control_bytes::<S>().ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let (sources, environment) = context
            .original_numerical_for(placement)
            .ok_or_else(|| retain_planning_error(PolicyCause::Context, funding.clone()))?;
        if let Some(controller) = policy.prepared_grammar_controller() {
            if policy.prepared_controller().is_some() || policy.prepared_host_copy().is_some() {
                return Err(sources.retain_startup_error(PolicyCause::Source));
            }
            logits.validate_consumer(sources)?;
            sources.validate_environment(environment)?;
            if logits.controller_choice().is_some_and(|choice| !controller.matches_choice(choice)) {
                return Err(sources.retain_startup_error(PolicyCause::Source));
            }
            let source = controller.controller_source().ok_or_else(|| sources.retain_startup_error(PolicyCause::Source))?;
            super::super::host_owner::validate_grammar_source(source, environment.pool())?;
            let (decision, policy) = controller.logits(history, funding)
                .map_err(|cause| sources.retain_startup_error(cause))?.into_parts();
            let source = decision.source();
            super::super::host_owner::validate_grammar_source(source, environment.pool())?;
            if !source.funding().same_account(funding) {
                return Err(sources.retain_startup_error(PolicyCause::Source));
            }
            let mask = decision.mask_plan(logits.value().array.shape())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let policy = policy.bind(temperature, history.len())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let choice = controller.choice(temperature, funding)
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let domain = decision.capture_domain().map_err(|cause| sources.retain_startup_error(cause))?;
            let output = NumericalProducer::execute_policy_at(context, placement, policy, logits, history,
                Some(mask), capture.map(|capture| capture.with_domain(domain)), failed_capture)?;
            return match output {
                NumericalOutput::Logits(value) => Ok((value.with_controller_choice(Some(choice)), None)),
                NumericalOutput::CapturedLogits { value, capture } => Ok((value.with_controller_choice(Some(choice)), Some(capture))),
                _ => Err(sources.retain_startup_error(PolicyCause::Output)),
            };
        }
        if let Some(controller) = policy.prepared_controller() {
            logits.validate_consumer(sources)?;
            sources.validate_environment(environment)?;
            if logits
                .controller_choice()
                .is_some_and(|choice| !controller.matches_choice(choice))
            {
                return Err(sources.retain_startup_error(PolicyCause::Source));
            }
            let (decision, policy) = controller
                .logits(history)
                .map_err(|cause| sources.retain_startup_error(cause))?
                .into_parts();
            let source = decision.source();
            super::super::host_owner::validate_controller_source(source, environment.pool())?;
            let mask = decision
                .mask_plan(logits.value().array.shape())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let policy = policy
                .bind(temperature, history.len())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let choice = controller
                .choice(temperature)
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let output = NumericalProducer::execute_policy_at(
                context,
                placement,
                policy,
                logits,
                history,
                Some(mask),
                capture.map(|capture| capture.with_domain(decision.capture_domain())),
                failed_capture,
            )?;
            return match output {
                NumericalOutput::Logits(value) => {
                    Ok((value.with_controller_choice(Some(choice)), None))
                }
                NumericalOutput::CapturedLogits { value, capture } => {
                    Ok((value.with_controller_choice(Some(choice)), Some(capture)))
                }
                _ => Err(sources.retain_startup_error(PolicyCause::Output)),
            };
        }
        if logits.controller_choice().is_some() {
            return Err(sources.retain_startup_error(PolicyCause::Source));
        }
        let policy = policy
            .prepared_logit_policy()
            .ok_or_else(|| sources.retain_startup_error(PreparedLogitPolicyError::Unknown))?
            .bind(temperature, history.len())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let output = NumericalProducer::execute_policy_at(
            context,
            placement,
            policy,
            logits,
            history,
            None,
            capture,
            failed_capture,
        )?;
        match output {
            NumericalOutput::Logits(value) => Ok((value, None)),
            NumericalOutput::CapturedLogits { value, capture } => Ok((value, Some(capture))),
            _ => Err(sources.retain_startup_error(PolicyCause::Output)),
        }
    })();
    result.map_err(|cause: Error| match cause.take_retained_backend_failure() {
        Ok(cause) => Error::StorageSource(cause),
        Err(cause) => retain_planning_error(cause, funding.clone()),
    })
}
fn control_bytes<S: SpeculativeSampler<MlxSamplingBackend>>() -> Option<usize> {
    let parts = [
        size_of::<SamplingPlacement>(),
        size_of::<(
            &S,
            &OriginalNumericalValue,
            f32,
            &[u32],
            SamplingPlacement,
            SpeculativeExecutionStreams<'_>,
        )>(),
        size_of::<Result<OriginalNumericalValue, Error>>(),
        size_of::<
            Result<
                (
                    OriginalNumericalValue,
                    eredu_core::capture::SharedCapturedStep,
                ),
                Error,
            >,
        >(),
        super::super::host_owner::grammar_source_control_bytes()?,
        size_of::<Option<eredu_runtime::generation::PreparedGrammarSampler<'_, S, S::PreparedGrammar>>>(),
        size_of::<Result<eredu_runtime::generation::PreparedGrammarLogits<'_, S::PreparedGrammar>, eredu_runtime::generation::PreparedGrammarSamplerError<S::PreparedGrammar>>>(),
        size_of::<Result<eredu_runtime::generation::PreparedControllerChoice, eredu_runtime::generation::PreparedGrammarSamplerError<S::PreparedGrammar>>>(),
        size_of::<eredu_runtime::execution_control::PreparedGrammarChoice<S::PreparedGrammar>>(),
        size_of::<eredu_core::capture::CaptureTokenDomain<'_>>(),
        size_of::<Result<eredu_core::capture::CaptureTokenDomain<'_>, eredu_core::PackedTokenFilterError>>(),
        eredu_runtime::generation::PreparedControllerChoice::metadata_bytes(),
        size_of::<Option<eredu_runtime::generation::PreparedSpeculativeController<'_, S>>>(),
        size_of::<
            Result<
                eredu_runtime::generation::PreparedControlledLogits<'_>,
                eredu_runtime::generation::PreparedControllerError,
            >,
        >(),
        eredu_runtime::generation::TokenMaskPlan::control_bytes(),
        super::super::host_owner::controller_source_control_bytes()?,
        eredu_core::speculative::ForbiddenControllerInputs::operation_control_bytes()?,
        size_of::<&mut ()>(),
        size_of::<&[u32]>(),
        size_of::<Option<PreparedLogitPolicy<'_>>>(),
        size_of::<Result<SpeculativeLogitProgram, PreparedLogitPolicyError>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<NumericalOutput>(),
        size_of::<Result<NumericalOutput, Error>>(),
        size_of::<Result<OriginalNumericalValue, Error>>(),
        size_of::<Option<CaptureSource<'_>>>(),
        size_of::<Option<eredu_core::capture::SharedCapturedStep>>(),
        size_of::<&mut Option<eredu_core::capture::SharedCapturedStep>>(),
        size_of::<
            Result<
                (
                    OriginalNumericalValue,
                    Option<eredu_core::capture::SharedCapturedStep>,
                ),
                Error,
            >,
        >(),
        size_of::<
            Result<
                (
                    OriginalNumericalValue,
                    eredu_core::capture::SharedCapturedStep,
                ),
                Error,
            >,
        >(),
        size_of::<PolicyCause>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

/// Called only for already processed values. Constrained samplers require the
/// private successful-producer witness; ordinary policies cannot consume it.
pub(super) fn bound_controller_choice<'a, S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    value: &'a OriginalNumericalValue,
    sources: &OriginalSpeculativeNumericalSources,
) -> Result<Option<&'a eredu_runtime::generation::PreparedControllerChoice>, Error> {
    let funding = value.validate_consumer(sources)?;
    let parts = [
        size_of::<Option<eredu_runtime::generation::PreparedGrammarSampler<'_, S, S::PreparedGrammar>>>(),
        size_of::<Option<eredu_runtime::generation::PreparedSpeculativeController<'_, S>>>(),
        size_of::<Option<&eredu_runtime::generation::PreparedControllerChoice>>(),
        size_of::<Result<Option<&eredu_runtime::generation::PreparedControllerChoice>, Error>>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    match (policy.prepared_controller(), policy.prepared_grammar_controller(), value.controller_choice()) {
        (None, None, None) => Ok(None),
        (Some(controller), None, Some(choice)) if controller.matches_choice(choice) => Ok(Some(choice)),
        (None, Some(controller), Some(choice)) if controller.matches_choice(choice) => Ok(Some(choice)),
        _ => Err(sources.retain_startup_error(PolicyCause::Source)),
    }
}

pub(crate) fn validate_controller_value<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    value: &OriginalNumericalValue,
    sources: &OriginalSpeculativeNumericalSources,
) -> Result<(), Error> {
    if bound_controller_choice(policy, value, sources)?.is_none() {
        return Err(sources.retain_startup_error(PolicyCause::Source));
    }
    Ok(())
}
