//! Actual prepared grammar ownership through the same original sampler container.
use super::*;
use eredu_core::speculative::{PreparedGrammarController, SpeculativeOutputError};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::generation::{PreparedGrammarSampler, PreparedGrammarSamplerError};
use eredu_runtime::working_memory::WorkingMemoryError;

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure<G: PreparedGrammarController> {
    #[source]
    cause: PreparedGrammarSamplerError<G>,
    host: HostPreparationAuthority,
}
fn failed<G: PreparedGrammarController>(
    cause: PreparedGrammarSamplerError<G>,
    host: &HostPreparationAuthority,
) -> eredu_core::BackendFailure {
    eredu_core::BackendFailure::from_error(Failure {
        cause,
        host: host.clone(),
    })
}
fn missing() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn current<'a, S: SpeculativeSampler<MlxSamplingBackend>>(
    source: &'a S,
    funding: &HostMetadataFunding,
) -> Result<PreparedGrammarSampler<'a, S, S::PreparedGrammar>, Error> {
    let plan = source.prepared_grammar_controller().ok_or_else(missing)?;
    if source.prepared_controller().is_some()
        || source.prepared_host_copy().is_some()
        || !plan
            .controller_source()
            .is_some_and(|source| source.funding().same_account(funding))
    {
        return Err(missing());
    }
    Ok(plan)
}
fn host_controls<S: SpeculativeSampler<MlxSamplingBackend>, E, L>() -> Option<usize> {
    let parts = [
        controls::<S, E, L>(0)?,
        grammar_source_control_bytes()?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            ?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure<S::PreparedGrammar>>()
            ?,
        size_of::<PreparedGrammarSampler<'_, S, S::PreparedGrammar>>(),
        size_of::<Option<PreparedGrammarSampler<'_, S, S::PreparedGrammar>>>(),
        size_of::<Failure<S::PreparedGrammar>>(),
        size_of::<PreparedGrammarSamplerError<S::PreparedGrammar>>(),
        size_of::<Result<S, PreparedGrammarSamplerError<S::PreparedGrammar>>>(),
        size_of::<Result<bool, PreparedGrammarSamplerError<S::PreparedGrammar>>>(),
        size_of::<Result<Policy<S>, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Result<(), SpeculativeControlError>>(),
        size_of::<Result<bool, SpeculativeControlError>>(),
        size_of::<Result<HostPreparationAuthority, HostMetadataFundingError>>(),
        size_of::<Result<f32, Error>>(),
        size_of::<Result<PreparedGrammarSampler<'_, S, S::PreparedGrammar>, Error>>(),
        size_of::<(&S, &HostMetadataFunding)>(),
        size_of::<(&S, SpeculativeExecutionStreams<'_>)>(),
        size_of::<(
            &mut MlxSpeculativeSampling<S, E, L>,
            &numerical::OriginalNumericalValue,
            u32,
            SamplingPlacement,
            SpeculativeExecutionStreams<'_>,
        )>(),
        size_of::<(
            &mut MlxSpeculativeSampling<S, E, L>,
            u32,
            eredu_runtime::TokenDomain,
            usize,
        )>(),
        size_of::<(&MlxSpeculativeSampling<S, E, L>, &[u32])>(),
        size_of::<Option<f32>>(),
    ];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
fn prepare_host<S: SpeculativeSampler<MlxSamplingBackend>, E, L>(
    funding: &HostMetadataFunding,
) -> Result<HostPreparationAuthority, HostMetadataFundingError> {
    funding.reserve_metadata(host_controls::<S, E, L>().ok_or(HostMetadataFundingError::Overflow)?)?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}
fn own<S: SpeculativeSampler<MlxSamplingBackend>>(
    source: S,
    host: HostPreparationAuthority,
    funding: &HostMetadataFunding,
    snapshot: Option<numerical::SnapshotContext>,
) -> Result<Policy<S>, Error> {
    current(&source, funding)?;
    Ok(own_original(
        source,
        host,
        Some(funding.clone()),
        false,
        snapshot,
    ))
}
pub(super) fn prepare<S: SpeculativeSampler<MlxSamplingBackend>>(
    source: &S,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<Policy<S>, Error> {
    let (sources, environment) = context.original_numerical().ok_or_else(missing)?;
    sources.validate_environment(environment)?;
    let funding = sources.metadata_funding();
    let host = prepare_host::<S, Exception, Array>(funding).map_err(Error::WorkspacePlanning)?;
    // Startup consumes the already constructed original request owner. Ordinary
    // or unrelated metadata accounts cannot be adopted while copying it.
    let plan = current(source, funding)?;
    validate_grammar_source(
        plan.controller_source().ok_or_else(missing)?,
        environment.pool(),
    )?;
    let copied = plan
        .copy(funding)
        .map_err(|cause| Error::StorageSource(failed(cause, &host)))?;
    own(
        copied,
        host,
        funding,
        Some(numerical::SnapshotContext::prepare_for_context(context)?),
    )
}
pub(super) fn snapshot_metadata<S: SpeculativeSampler<MlxSamplingBackend>, E, L>(
    source: &Policy<S>,
) -> Option<usize> {
    let funding = source.metadata_funding()?;
    let plan = current(source.source(), funding).ok()?;
    host_controls::<S, E, L>()?
        .checked_add(plan.copy_metadata_bytes()?)?
        .checked_add(eredu_core::HostMetadataFunding::prepaid_control_bytes()?)
}
pub(super) fn copy_snapshot<S: SpeculativeSampler<MlxSamplingBackend>, E, L>(
    source: &Policy<S>, host: &HostPreparationAuthority,
) -> Result<Policy<S>, eredu_core::BackendFailure> {
    let limit = snapshot_metadata::<S, E, L>(source)
        .ok_or_else(|| eredu_core::BackendFailure::from_error(WorkingMemoryError::UnknownBound))?;
    let original_funding = source.metadata_funding()
        .ok_or_else(|| eredu_core::BackendFailure::from_error(WorkingMemoryError::IdentityMismatch))?;
    let plan = current(source.source(), original_funding).map_err(eredu_core::BackendFailure::from_error)?;
    let funding = HostMetadataFunding::from_prepaid(limit, host.clone())
            .map_err(eredu_core::BackendFailure::from_error)?;
    let retained = prepare_host::<S, E, L>(&funding)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let copied = plan.copy(&funding).map_err(|cause| failed(cause, &retained))?;
    own(copied, retained, &funding, source.snapshot_context().cloned())
        .map_err(eredu_core::BackendFailure::from_error)
}
fn control_failure<G: PreparedGrammarController>(
    cause: PreparedGrammarSamplerError<G>,
    host: &HostPreparationAuthority,
) -> SpeculativeControlError {
    use eredu_runtime::{
        execution_control::{PreparedGrammarChoiceCause, TokenChoiceError},
        generation::PreparedGrammarSamplerCause,
    };
    if let PreparedGrammarSamplerCause::Decision(decision) = cause.cause() {
        if let PreparedGrammarChoiceCause::Choice(choice) = decision.cause() {
            return match choice {
                TokenChoiceError::InvalidToken(token) => {
                    SpeculativeControlError::InvalidToken(*token)
                }
                TokenChoiceError::Forbidden(token) => {
                    SpeculativeControlError::ForbiddenToken(*token)
                }
                TokenChoiceError::AlreadyPending => SpeculativeControlError::PendingToken,
                TokenChoiceError::UnexpectedCommit { .. } => {
                    SpeculativeControlError::Invalid("inconsistent forced choice")
                }
                TokenChoiceError::Constraint(never) => match *never {},
            };
        }
    }
    SpeculativeControlError::Backend(failed(cause, host))
}
fn control_missing() -> SpeculativeControlError {
    SpeculativeControlError::Unsupported("sampler has no authenticated prepared grammar owner")
}
fn control_funding(cause: HostMetadataFundingError) -> SpeculativeControlError {
    SpeculativeControlError::Output(SpeculativeOutputError::HostFunding(cause))
}
impl<S, E, L> MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    pub(super) fn commit_grammar_original(
        &mut self,
        value: &numerical::OriginalNumericalValue,
        token: u32,
        placement: SamplingPlacement,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(), Error> {
        self.inner.original_source()?;
        let (sources, environment) = context
            .original_numerical_for(placement)
            .ok_or_else(missing)?;
        let funding = value.validate_consumer(sources)?;
        numerical::validate_commit_value(value, token, placement, context)?;
        numerical::validate_controller_value(self.inner.source(), value, sources)?;
        let source_funding = self.inner.metadata_funding().ok_or_else(missing)?;
        let plan = current(self.inner.source(), source_funding)?;
        validate_grammar_source(
            plan.controller_source().ok_or_else(missing)?,
            environment.pool(),
        )?;
        plan.validate_commit(token)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let host = prepare_host::<S, E, L>(funding).map_err(Error::WorkspacePlanning)?;
        let probability = if plan.requires_commit_probability() {
            Some(numerical::commit_probability(
                value, token, placement, context,
            )?)
        } else {
            None
        };
        let source = plan
            .commit(token, probability, funding)
            .map_err(|cause| Error::StorageSource(failed(cause, &host)))?;
        let replacement = own(
            source,
            host,
            funding,
            self.inner.snapshot_context().cloned(),
        )?;
        self.inner = replacement;
        Ok(())
    }
    pub(super) fn force_grammar_original(
        &mut self,
        token: u32,
        domain: eredu_runtime::TokenDomain,
        position: usize,
    ) -> Result<(), SpeculativeControlError> {
        if !self.inner.is_original() {
            return Err(control_missing());
        }
        let funding = self.inner.metadata_funding().ok_or_else(control_missing)?;
        let plan = current(self.inner.source(), funding).map_err(|_| control_missing())?;
        let host = prepare_host::<S, E, L>(funding).map_err(control_funding)?;
        let source = plan
            .force(token, domain, position, funding)
            .map_err(|cause| control_failure(cause, &host))?;
        let replacement = own(
            source,
            host,
            funding,
            self.inner.snapshot_context().cloned(),
        )
        .map_err(|_| control_missing())?;
        self.inner = replacement;
        Ok(())
    }
    pub(super) fn clear_grammar_original(&mut self) -> Result<bool, SpeculativeControlError> {
        if !self.inner.is_original() {
            return Err(control_missing());
        }
        if self.inner.source().control_pending_forced().is_none() {
            return Ok(false);
        }
        let funding = self.inner.metadata_funding().ok_or_else(control_missing)?;
        let plan = current(self.inner.source(), funding).map_err(|_| control_missing())?;
        let host = prepare_host::<S, E, L>(funding).map_err(control_funding)?;
        let source = plan
            .clear(funding)
            .map_err(|cause| control_failure(cause, &host))?;
        let replacement = own(
            source,
            host,
            funding,
            self.inner.snapshot_context().cloned(),
        )
        .map_err(|_| control_missing())?;
        self.inner = replacement;
        Ok(true)
    }
    pub(super) fn grammar_prefix_original(&self, history: &[u32]) -> Result<bool, Error> {
        self.inner.original_source()?;
        let funding = self.inner.metadata_funding().ok_or_else(missing)?;
        let plan = current(self.inner.source(), funding)?;
        let host = prepare_host::<S, E, L>(funding).map_err(Error::WorkspacePlanning)?;
        plan.prefix_is_complete(history, funding)
            .map_err(|cause| Error::StorageSource(failed(cause, &host)))
    }
    pub(in crate::composition::mlx::speculative::sampling) fn current_grammar_original(
        &self,
    ) -> Result<bool, Error> {
        let funding = self.inner.metadata_funding().ok_or_else(missing)?;
        let plan = current(self.inner.original_source()?, funding)?;
        let history = plan.controller_source().ok_or_else(missing)?.history();
        self.grammar_prefix_original(history)
    }
}
