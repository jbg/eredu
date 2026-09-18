//! One paid provisional adaptive policy through the same immutable owner.
use super::*;
use eredu_runtime::generation::{PreparedAdaptiveCommit, PreparedAdaptiveCommitError};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: PreparedAdaptiveCommitError,
    _host: HostPreparationAuthority,
}
pub(super) fn control_bytes<S>() -> Option<usize> {
    let parts = [
        size_of::<PreparedAdaptiveCommit<'_, S>>(),
        size_of::<Option<PreparedAdaptiveCommit<'_, S>>>(),
        size_of::<Result<S, PreparedAdaptiveCommitError>>(),
        size_of::<Result<(), PreparedAdaptiveCommitError>>(),
        size_of::<Result<f32, Error>>(),
        size_of::<SamplingPlacement>(),
        size_of::<Failure>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl<S, E, L> MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    pub(super) fn commit_adaptive_original(
        &mut self,
        value: &numerical::OriginalNumericalValue,
        token: u32,
        placement: SamplingPlacement,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(), Error> {
        let (sources, environment) = context.original_numerical_for(placement).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        self.inner.original_source()?;
        let funding = value.validate_consumer(sources)?;
        numerical::validate_commit_value(value, token, placement, context)?;
        let controller = self.inner.source().prepared_controller();
        if let Some(controller) = controller {
            numerical::validate_controller_value(self.inner.source(), value, sources)?;
            CopyPlan::Controller(controller).validate_source(environment.pool())?;
        } else if value.controller_choice().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let plan = self
            .inner
            .source()
            .prepared_adaptive_commit()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?;
        plan.validate_token(token)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let bytes = controls::<S, E, L>(plan.metadata_bytes())
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    eredu_nn::workspace::HostMetadataFunding,
                >()?)
            })
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let host = HostPreparationAuthority::retain(funding.clone());
        // Exact successful native observation precedes any provisional mutation.
        let probability = numerical::commit_probability(value, token, placement, context)?;
        let source = plan
            .commit(token, probability, host.clone())
            .map_err(|cause| {
                Error::StorageSource(eredu_core::BackendFailure::from_error(Failure {
                    cause,
                    _host: host.clone(),
                }))
            })?;
        let replacement = own_original(
            source,
            host,
            Some(funding.clone()),
            self.inner.is_fixed_controller(),
            self.inner.snapshot_context().cloned(),
        );
        self.inner = replacement;
        Ok(())
    }
}
