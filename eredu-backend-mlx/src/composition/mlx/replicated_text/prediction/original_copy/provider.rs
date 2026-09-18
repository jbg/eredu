//! Native producers for the neutral prepared embedded cache lifecycle.
mod tensor;
mod external;
use super::*;
use crate::composition::mlx::speculative::{
    OriginalSpeculativeNumericalSources, SpeculativeExecutionStreams,
};
use eredu_architectures::speculative_execution::{
    EmbeddedPredictionCache, PreparedEmbeddedCopy, PreparedEmbeddedCopyError,
    PreparedEmbeddedCopyProvider, PreparedEmbeddedEvidence, PreparedEmbeddedPayload,
    PreparedEmbeddedState,
};
use eredu_core::BackendFailure;
use eredu_nn::workspace::HostMetadataFundingError;
use eredu_runtime::SelectedSpeculativeRealization;
use std::{cell::RefCell, collections::TryReserveError};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Startup(#[from] StartupCause),
    #[error(transparent)]
    Contract(#[from] PreparedEmbeddedCopyError),
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _funding: HostMetadataFunding,
}
fn error_controls() -> Option<usize> {
    let frames = [
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<BackendFailure>(),
        size_of::<Result<(), BackendFailure>>(),
        BackendFailure::source_retention_peak_bytes::<Failure>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
fn failure(context: &OriginalPredictionCopyContext, cause: Cause) -> BackendFailure {
    // Actual native errors may already own unresolved Recovery/source custody.
    // Move that envelope without constructing a second erasure shell.
    let cause = match cause {
        Cause::Startup(StartupCause::Backend(cause)) => match cause.take_retained_backend_failure()
        {
            Ok(cause) => return cause,
            Err(cause) => Cause::Startup(StartupCause::Backend(cause)),
        },
        cause => cause,
    };
    BackendFailure::from_error(Failure {
        cause,
        _funding: context.preparation.metadata_funding().clone(),
    })
}
fn pay(
    context: &OriginalPredictionCopyContext,
    bytes: usize,
) -> Result<HostPreparationAuthority, BackendFailure> {
    let frames = [
        bytes,
        error_controls()
            .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?,
        size_of::<HostPreparationAuthority>(),
        size_of::<Result<HostPreparationAuthority, BackendFailure>>(),
    ];
    let bytes = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?;
    context
        .preparation
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(HostMetadataFundingError::into_backend_failure)?;
    Ok(HostPreparationAuthority::retain(
        context.preparation.metadata_funding().clone(),
    ))
}
fn reject(
    context: &OriginalPredictionCopyContext,
    cause: PreparedEmbeddedCopyError,
) -> BackendFailure {
    match pay(context, size_of::<PreparedEmbeddedCopyError>()) {
        Ok(_controls) => failure(context, cause.into()),
        Err(cause) => cause,
    }
}
fn completed<'a>(
    evidence: Option<&'a PreparedEmbeddedEvidence>,
) -> Result<Option<&'a CompletedResidentSource>, PreparedEmbeddedCopyError> {
    evidence
        .map(|value| {
            value
                .get::<CompletedResidentSource>()
                .ok_or(PreparedEmbeddedCopyError::SourceMismatch)
        })
        .transpose()
}
struct Provider<S> {
    context: OriginalPredictionCopyContext,
    copy: fn(
        &OriginalPredictionCopyContext,
        &S,
        Option<&CompletedResidentSource>,
    ) -> Result<PreparedLane<S>, StartupCause>,
}
impl<S: 'static> PreparedEmbeddedCopyProvider<S> for Provider<S> {
    fn copy_state(
        &self,
        source: &S,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<S>, BackendFailure> {
        let parts = [
            size_of::<PreparedLane<S>>(),
            size_of::<Result<PreparedLane<S>, StartupCause>>(),
            size_of::<Result<PreparedEmbeddedPayload<S>, BackendFailure>>(),
            size_of::<Option<&CompletedResidentSource>>(),
            size_of::<Result<Option<&CompletedResidentSource>, PreparedEmbeddedCopyError>>(),
            size_of_val(&self.copy),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| reject(&self.context, PreparedEmbeddedCopyError::Overflow))?;
        let _controls = pay(&self.context, bytes)?;
        let result = (|| {
            let evidence = completed(evidence).map_err(Cause::Contract)?;
            let copied = (self.copy)(&self.context, source, evidence)?;
            // Independent copy publication installs actual registered custody in
            // copied storage. No prior model-role witness is carried forward.
            Ok(PreparedEmbeddedPayload::new(copied.value, copied.host))
        })();
        result.map_err(|cause| failure(&self.context, cause))
    }
    fn prepare_host(&self, bytes: usize) -> Result<HostPreparationAuthority, BackendFailure> {
        pay(&self.context, bytes)
    }
    fn reject(&self, cause: PreparedEmbeddedCopyError) -> BackendFailure {
        reject(&self.context, cause)
    }
    fn allocation_failure(&self, cause: TryReserveError) -> BackendFailure {
        // Every neutral exact reserve first calls prepare_host, which prepays
        // this retained error destination before the failing allocation attempt.
        failure(&self.context, Cause::Allocation(cause))
    }
}
fn target_provider<S: MlxStateMechanisms + 'static>(
    context: OriginalPredictionCopyContext,
) -> Result<PreparedEmbeddedCopy<S>, BackendFailure> {
    PreparedEmbeddedCopy::prepare(Provider {
        context,
        copy: |context, source, completed| context.target(source, &context.origin, completed),
    })
}
fn prediction_provider<A, P>(
    context: OriginalPredictionCopyContext,
) -> Result<PreparedEmbeddedCopy<P::LaneState>, BackendFailure>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    PreparedEmbeddedCopy::prepare(Provider {
        context,
        copy: |context, source, completed| context.prediction_completed::<A, P>(source, completed),
    })
}
impl OriginalPredictionLane {
    /// Moves the actual paid startup Box payload into its typed cache owner.
    /// Only the closed provider shell is newly constructed; no native copy or
    /// ordinary allocation adoption occurs at this ownership boundary.
    pub(crate) fn into_prepared<A, P>(
        self,
        extension: &P,
        selected: &SelectedSpeculativeRealization,
        preparation: &PreparedSemanticSource,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<PreparedEmbeddedState<P::LaneState>, BackendFailure>
    where
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
    {
        let parts = [
            size_of::<Self>(),
            size_of::<PreparedEmbeddedState<P::LaneState>>(),
            size_of::<PreparedEmbeddedPayload<P::LaneState>>(),
            size_of::<Box<P::LaneState>>(),
            size_of::<Result<Box<P::LaneState>, Box<dyn Any>>>(),
            size_of::<Result<PreparedEmbeddedState<P::LaneState>, BackendFailure>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?;
        let _controls = pay(&self.copy, bytes)?;
        self.copy
            .validate(preparation, origin)
            .map_err(|cause| failure(&self.copy, cause.into()))?;
        if self.class != selected.requirements().strategy().class()
            || self.proposal_capacity
                != selected.requirements().strategy().proposal_capacity().get()
            || self.depth != extension.depth()
            || Some(self.shape) != extension.occurrence_shape()
            || self.alignment != extension.prefill_alignment()
        {
            return Err(failure(
                &self.copy,
                PreparedEmbeddedCopyError::SourceMismatch.into(),
            ));
        }
        let copy = prediction_provider::<A, P>(self.copy.clone())?;
        let state = match self.state.downcast::<P::LaneState>() {
            Ok(state) => state,
            Err(state) => {
                drop(state);
                return Err(failure(
                    &self.copy,
                    PreparedEmbeddedCopyError::SourceMismatch.into(),
                ));
            }
        };
        Ok(PreparedEmbeddedState::new(
            PreparedEmbeddedPayload::new(*state, self._host),
            copy,
        ))
    }
}

/// A single startup handoff. Its independently copied target is consumed once
/// by shared cache preparation; later completion evidence lives on cache states.
/// No request, active Scope or mutable source-state backedge is retained here.
pub(crate) struct OriginalEmbeddedCachePreparation {
    lane: RefCell<Option<OriginalPredictionLane>>,
    target: RefCell<Option<OriginalPredictionTarget>>,
    copy: OriginalPredictionCopyContext,
    identity: eredu_runtime::working_memory::OriginalSpeculativeSourceIdentity,
    _host: HostPreparationAuthority,
}
impl std::fmt::Debug for OriginalEmbeddedCachePreparation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalEmbeddedCachePreparation")
            .finish_non_exhaustive()
    }
}
impl OriginalEmbeddedCachePreparation {
    pub(crate) fn new(
        lane: OriginalPredictionLane,
        target: OriginalPredictionTarget,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<Self, BackendFailure> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, BackendFailure>>(),
            size_of::<Option<OriginalPredictionLane>>(),
            size_of::<Option<OriginalPredictionTarget>>(),
            size_of::<OriginalPredictionCopyContext>(),
            size_of::<eredu_runtime::working_memory::OriginalSpeculativeSourceIdentity>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| reject(&lane.copy, PreparedEmbeddedCopyError::Overflow))?;
        let host = pay(&lane.copy, bytes)?;
        lane.copy
            .preparation
            .validate(sources.pool(), sources.request().execution_identity())
            .map_err(|cause| {
                failure(
                    &lane.copy,
                    StartupCause::Backend(Error::PrefillControl(cause)).into(),
                )
            })?;
        if !lane.origin.same_origin(sources.target_origin()) {
            return Err(failure(
                &lane.copy,
                PreparedEmbeddedCopyError::SourceMismatch.into(),
            ));
        }
        target.validate(&lane.copy.preparation, &lane.origin)
            .map_err(|cause| failure(&lane.copy, cause.into()))?;
        let identity = sources.request().source_identity();
        Ok(Self {
            copy: lane.copy.clone(),
            lane: RefCell::new(Some(lane)),
            target: RefCell::new(Some(target)),
            identity,
            _host: host,
        })
    }
    pub(crate) fn binding_controls(&self) -> Result<HostPreparationAuthority, BackendFailure> {
        let parts = [
            size_of::<SpeculativeExecutionStreams<'_>>(),
            size_of::<Result<SpeculativeExecutionStreams<'_>, BackendFailure>>(),
            size_of::<&Self>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?;
        pay(&self.copy, bytes)
    }
    pub(crate) fn reject_source(&self) -> BackendFailure {
        reject(&self.copy, PreparedEmbeddedCopyError::SourceMismatch)
    }
    pub(crate) fn validate_sources(
        &self,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<(), BackendFailure> {
        if !self.identity.belongs_to_request(sources.request())
            || !self.copy.origin.same_origin(sources.target_origin())
        {
            return Err(reject(
                &self.copy,
                PreparedEmbeddedCopyError::SourceMismatch,
            ));
        }
        self.validate(&self.copy.preparation, sources.target_origin())
    }
    pub(crate) fn validate(
        &self,
        preparation: &PreparedSemanticSource,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<(), BackendFailure> {
        let _controls = pay(&self.copy, size_of::<Result<(), BackendFailure>>())?;
        self.copy
            .validate(preparation, origin)
            .map_err(|cause| failure(&self.copy, cause.into()))
    }
    /// Runs at the shared target-preparation mutation boundary, outside any
    /// numerical Scope. Moves the completed target copy and prepared lane.
    fn prepare<A, P, S>(
        &self,
        source: &S,
        extension: &P,
        selected: &SelectedSpeculativeRealization,
    ) -> Result<EmbeddedPredictionCache<S, P::LaneState>, BackendFailure>
    where
        S: MlxStateMechanisms + 'static,
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
    {
        let parts = [
            size_of::<EmbeddedPredictionCache<S, P::LaneState>>(),
            size_of::<Result<EmbeddedPredictionCache<S, P::LaneState>, BackendFailure>>(),
            size_of::<std::cell::RefMut<'_, Option<OriginalPredictionLane>>>(),
            size_of::<std::cell::RefMut<'_, Option<OriginalPredictionTarget>>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?;
        let _controls = pay(&self.copy, bytes)?;
        let mut lane = self
            .lane
            .try_borrow_mut()
            .map_err(|_| failure(&self.copy, PreparedEmbeddedCopyError::SourceMismatch.into()))?;
        let mut target = self
            .target
            .try_borrow_mut()
            .map_err(|_| failure(&self.copy, PreparedEmbeddedCopyError::SourceMismatch.into()))?;
        let prepared_lane = lane
            .take()
            .ok_or_else(|| failure(&self.copy, PreparedEmbeddedCopyError::SourceMismatch.into()))?;
        drop(lane);
        let target_source = target.take().ok_or_else(||
            failure(&self.copy, PreparedEmbeddedCopyError::SourceMismatch.into()))?;
        drop(target);
        let lane = prepared_lane;
        let target = target_source;
        let target_copy = target.into_prepared(
            source, &self.copy.preparation, &self.copy.origin,
        ).map_err(|cause| failure(&self.copy, cause.into()))?;
        let target = PreparedEmbeddedState::new(
            PreparedEmbeddedPayload::new(target_copy.value, target_copy.host),
            target_provider(self.copy.clone())?,
        );
        let prediction = lane.into_prepared::<A, P>(
            extension,
            selected,
            &self.copy.preparation,
            &self.copy.origin,
        )?;
        EmbeddedPredictionCache::from_prepared(target, prediction)
    }
}

/// Actual shared factory destinations and typed error transport. Callers price
/// their concrete control/payload shapes before invoking a fallible producer;
/// the shared retained-error worker separately pays its concrete error shell.
pub(crate) fn cache_metadata(
    bytes: Option<usize>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<HostPreparationAuthority, BackendFailure> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(HostPreparationAuthority::unmanaged());
    };
    let parts = [
        bytes,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>(),
        Some(size_of::<HostPreparationAuthority>()),
        Some(size_of::<Result<HostPreparationAuthority, BackendFailure>>()),
        Some(size_of::<(Option<usize>, SpeculativeExecutionStreams<'_>)>()),
        Some(size_of::<
            Option<(
                &OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>,
            )>,
        >()),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), |n, p| n.checked_add(p?))
        .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(HostMetadataFundingError::into_backend_failure)?;
    sources
        .validate_environment(environment)
        .map_err(Error::into_backend_failure)?;
    Ok(HostPreparationAuthority::retain(
        sources.metadata_funding().clone(),
    ))
}
pub(crate) fn cache_error<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    context: SpeculativeExecutionStreams<'_>,
) -> BackendFailure {
    match context.original_numerical() {
        Some((sources, _)) => sources.retain_startup_error(cause).into_backend_failure(),
        None => BackendFailure::from_error(cause),
    }
}
pub(crate) fn prepare_cache<A, P, S>(
    source: &S,
    extension: &P,
    selected: &SelectedSpeculativeRealization,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<EmbeddedPredictionCache<S, P::LaneState>, BackendFailure>
where
    S: MlxStateMechanisms + 'static,
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    let (sources, _) = context
        .original_numerical()
        .ok_or_else(|| BackendFailure::from_error(PreparedEmbeddedCopyError::SourceMismatch))?;
    let preparation = context.original_cache_preparation().ok_or_else(|| {
        sources
            .retain_startup_error(WorkingMemoryError::UnknownBound)
            .into_backend_failure()
    })?;
    preparation.validate_sources(sources)?;
    preparation.prepare::<A, P, S>(source, extension, selected)
}
