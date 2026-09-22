//! Retained prerequisites for actual target/prediction cache copies.
mod provider;
mod target;
use super::super::state::MlxStateMechanisms;
use super::startup::{
    OriginalPredictionLane, OriginalPredictionStartupContext, PreparedLane, StartupCause,
};
use super::{MlxEmbeddedPredictionMaterializer, OwnedPredictionCache};
use crate::backend::{
    error::Error,
    managed_memory::NativeMemoryRetention,
    nn::{shared::MlxNeuralBackend, workspace::MlxMetalWorkspaceMechanisms},
    runtime::cache::{
        copy_completed_compressed, copy_completed_pooling,
        kv::CompressedLatentCache,
        state::{
            copy_completed_resident_state, CompletedResidentSource, MlxHybridState,
            MlxPoolingAttentionCache, OriginalResidentState, PreparedResidentDecoderCopy,
        },
        PreparedPredictionCacheCopy,
    },
    OriginalCopyEnvironment, PreparedOriginalCopyEnvironment, PreparedOriginalCopyEnvironmentError,
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionStateCopyFactory,
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::{
    replicated_session::ReplicatedTextControlOrigin,
    working_memory::{MemoryLedger, PreparedSemanticSource, WorkingMemoryError},
};
pub(crate) use provider::{
    cache_error, cache_metadata, prepare_cache, OriginalEmbeddedCachePreparation,
};
use safemlx::PrefillRootsRuntime;
use std::{
    any::Any,
    mem::{size_of, size_of_val},
};
pub(crate) use target::OriginalPredictionTarget;

fn memory(value: WorkingMemoryError) -> StartupCause {
    Error::PrefillControl(value).into()
}
fn add(a: usize, b: usize) -> Result<usize, StartupCause> {
    a.checked_add(b)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

/// Source identity and prepared stream only: no model/module, active state,
/// request slot, or native numerical scope is retained here. A copy opens its
/// existing independent copy scope only at a settled checkpoint boundary.
#[derive(Clone)]
pub(crate) struct OriginalPredictionCopyContext {
    environment: PreparedOriginalCopyEnvironment,
    pool: MemoryLedger,
    roots: PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    origin: ReplicatedTextControlOrigin,
    preparation: PreparedSemanticSource,
    host: HostPreparationAuthority,
}
impl std::fmt::Debug for OriginalPredictionCopyContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalPredictionCopyContext")
            .finish_non_exhaustive()
    }
}
impl OriginalPredictionCopyContext {
    fn preparation_controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, StartupCause>>(),
            size_of::<PreparedOriginalCopyEnvironment>(),
            size_of::<Result<PreparedOriginalCopyEnvironment, PreparedOriginalCopyEnvironmentError>>(
            ),
            size_of::<(
                &OriginalCopyEnvironment<'_>,
                &HostPreparationAuthority,
                &HostMetadataFunding,
            )>(),
            OriginalCopyEnvironment::control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn prepare(
        environment: &OriginalCopyEnvironment<'_>,
        roots: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        preparation: &PreparedSemanticSource,
        origin: &ReplicatedTextControlOrigin,
        host: &HostPreparationAuthority,
    ) -> Result<Self, StartupCause> {
        preparation
            .metadata_funding()
            .reserve_metadata(
                Self::preparation_controls().ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let retained = PreparedOriginalCopyEnvironment::prepare(
            environment,
            host,
            preparation.metadata_funding(),
        )?;
        Ok(Self {
            environment: retained,
            pool: environment.pool().clone(),
            roots: roots.clone(),
            mechanisms,
            origin: origin.clone(),
            preparation: preparation.clone(),
            host: host.clone(),
        })
    }
    pub(super) fn reserve(&self, bytes: usize) -> Result<(), StartupCause> {
        self.preparation
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        Ok(())
    }
    fn loan(&self) -> Result<OriginalCopyEnvironment<'_>, StartupCause> {
        self.preparation
            .tokenizer()
            .validate_pool(&self.pool)
            .map_err(memory)?;
        Ok(self.environment.loan(&self.pool)?)
    }
    pub(super) fn validate(
        &self,
        preparation: &PreparedSemanticSource,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<(), StartupCause> {
        let parts = [
            size_of::<OriginalCopyEnvironment<'_>>(),
            size_of::<Result<OriginalCopyEnvironment<'_>, StartupCause>>(),
            size_of::<
                Result<OriginalCopyEnvironment<'_>, crate::backend::OriginalCopyEnvironmentError>,
            >(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<(), StartupCause>>(),
            self.environment
                .control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ];
        self.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        if !self.preparation.same_preparation(preparation) || !self.origin.same_origin(origin) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        // Loan checks exact retained stream/allocator/domain, not approximate
        // device equality. It provides no target/prediction invocation grant.
        self.loan()?;
        Ok(())
    }
    /// Exact native representation, moved directly out of the shared copy result.
    /// The caller supplies the origin attached to this actual target owner.
    pub(crate) fn target<S: MlxStateMechanisms>(
        &self,
        source: &S,
        origin: &ReplicatedTextControlOrigin,
        completed: Option<&CompletedResidentSource>,
    ) -> Result<PreparedLane<S>, StartupCause> {
        self.validate(&self.preparation, origin)?;
        let parts = [
            size_of::<PreparedResidentDecoderCopy<'_>>(),
            size_of::<
                Result<
                    PreparedResidentDecoderCopy<'_>,
                    crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
                >,
            >(),
            size_of::<Result<S, OriginalResidentState>>(),
            size_of::<PreparedLane<S>>(),
            size_of::<Result<PreparedLane<S>, StartupCause>>(),
        ];
        self.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        let environment = self.loan()?;
        let copied = match source.copy_original_paged_state(
            completed,
            &environment,
            &self.roots,
            self.mechanisms,
            self.preparation.metadata_funding(),
            &self.host,
            self.preparation.limits().clone(),
        )? {
            Some(value) => PreparedLane {
                value,
                host: self.host.clone(),
            },
            None => self.target_source(source.prepare_resident_decoder_copy_fixed()?, completed)?,
        };
        let value = match S::from_original_resident_copy(copied.value) {
            Ok(value) => value,
            Err(unexpected) => {
                drop(unexpected);
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
        };
        Ok(PreparedLane {
            value,
            host: copied.host,
        })
    }
    /// Same independent admitted copy for the actual cold source before typed
    /// cache construction. Registered roots retain their canonical publication;
    /// only a genuine completed-role source can supply optional role evidence.
    fn target_source(
        &self,
        source: PreparedResidentDecoderCopy<'_>,
        completed: Option<&CompletedResidentSource>,
    ) -> Result<PreparedLane<OriginalResidentState>, StartupCause> {
        let parts = [
            size_of::<PreparedResidentDecoderCopy<'_>>(),
            size_of::<Option<&CompletedResidentSource>>(),
            size_of::<OriginalResidentState>(),
            size_of::<Result<OriginalResidentState, Error>>(),
            size_of::<PreparedLane<OriginalResidentState>>(),
            size_of::<Result<PreparedLane<OriginalResidentState>, StartupCause>>(),
            self.environment
                .control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ];
        self.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        let environment = self.loan()?;
        let value = copy_completed_resident_state(
            source,
            completed,
            &environment,
            &self.roots,
            self.mechanisms,
            self.preparation.metadata_funding(),
            &self.host,
            self.preparation.limits(),
        )?;
        Ok(PreparedLane {
            value,
            host: self.host.clone(),
        })
    }
    pub(super) fn prediction<A, P>(
        &self,
        source: &P::LaneState,
    ) -> Result<PreparedLane<P::LaneState>, StartupCause>
    where
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
    {
        self.prediction_completed::<A, P>(source, None)
    }
    pub(super) fn prediction_completed<A, P>(
        &self,
        source: &P::LaneState,
        completed: Option<&CompletedResidentSource>,
    ) -> Result<PreparedLane<P::LaneState>, StartupCause>
    where
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
    {
        self.validate(&self.preparation, &self.origin)?;
        let parts = [
            size_of::<CopyFactory<'_>>(),
            size_of::<PreparedLane<P::LaneState>>(),
            size_of::<Result<PreparedLane<P::LaneState>, StartupCause>>(),
            self.environment
                .control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ];
        self.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        P::prepare_copy_state(
            source,
            &mut CopyFactory {
                context: self,
                completed,
            },
        )
    }
}

struct CopyFactory<'a> {
    context: &'a OriginalPredictionCopyContext,
    completed: Option<&'a CompletedResidentSource>,
}
struct LeafCustody {
    _members: Vec<HostPreparationAuthority>,
    _funding: HostMetadataFunding,
}
impl CopyFactory<'_> {
    fn members<C>(
        &self,
        source: &[OwnedPredictionCache<C>],
        copy: fn(
            &C,
            Option<&CompletedResidentSource>,
            &OriginalCopyEnvironment<'_>,
            &PrefillRootsRuntime,
            MlxMetalWorkspaceMechanisms,
            &HostMetadataFunding,
            &eredu_core::MemoryLimits,
        ) -> Result<PreparedPredictionCacheCopy<C>, Error>,
    ) -> Result<PreparedLane<Vec<OwnedPredictionCache<C>>>, StartupCause> {
        let member_bytes = add(
            size_of::<OwnedPredictionCache<C>>(),
            size_of::<HostPreparationAuthority>(),
        )?;
        let slots = source
            .len()
            .checked_mul(member_bytes)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let parts = [
            slots,
            size_of::<Vec<OwnedPredictionCache<C>>>(),
            size_of::<Vec<HostPreparationAuthority>>(),
            size_of::<PreparedPredictionCacheCopy<C>>(),
            size_of::<Result<PreparedPredictionCacheCopy<C>, Error>>(),
            size_of::<(C, HostPreparationAuthority)>(),
            size_of::<LeafCustody>(),
            size_of::<PreparedLane<Vec<OwnedPredictionCache<C>>>>(),
            size_of::<Result<PreparedLane<Vec<OwnedPredictionCache<C>>>, StartupCause>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<std::slice::Iter<'_, OwnedPredictionCache<C>>>(),
            size_of::<Option<&OwnedPredictionCache<C>>>(),
            size_of_val(&copy),
            HostPreparationAuthority::retention_bytes::<LeafCustody>()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        ];
        self.context
            .reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        // Locals drop in reverse declaration order: provisional numerical rows
        // always retire before the per-member paying authorities on failure.
        let mut members = Vec::new();
        let mut value = Vec::new();
        members.try_reserve_exact(source.len())?;
        value.try_reserve_exact(source.len())?;
        let environment = self.context.loan()?;
        for member in source {
            let (copied, host) = copy(
                member.inner(),
                self.completed,
                &environment,
                &self.context.roots,
                self.context.mechanisms,
                self.context.preparation.metadata_funding(),
                self.context.preparation.limits(),
            )?
            .into_parts();
            members.push(host);
            value.push(OwnedPredictionCache::new(
                copied,
                NativeMemoryRetention::default(),
            ));
        }
        let host = HostPreparationAuthority::retain(LeafCustody {
            _members: members,
            _funding: self.context.preparation.metadata_funding().clone(),
        });
        Ok(PreparedLane { value, host })
    }
}
impl PredictionStateCopyFactory<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
    for CopyFactory<'_>
{
    type Error = StartupCause;
    type Prepared<T> = PreparedLane<T>;
    fn sequential(
        &mut self,
        source: &[OwnedPredictionCache<CompressedLatentCache>],
    ) -> Result<PreparedLane<Vec<OwnedPredictionCache<CompressedLatentCache>>>, StartupCause> {
        self.members(source, copy_completed_compressed)
    }
    fn pooling(
        &mut self,
        source: &[OwnedPredictionCache<MlxPoolingAttentionCache>],
    ) -> Result<PreparedLane<Vec<OwnedPredictionCache<MlxPoolingAttentionCache>>>, StartupCause>
    {
        self.members(source, copy_completed_pooling)
    }
    fn model(
        &mut self,
        source: &MlxHybridState,
    ) -> Result<PreparedLane<MlxHybridState>, StartupCause> {
        self.context
            .target(source, &self.context.origin, self.completed)
    }
}

pub(super) type CopyErased = fn(
    &dyn Any,
    &OriginalPredictionCopyContext,
) -> Result<PreparedLane<Box<dyn Any>>, StartupCause>;
pub(super) fn copy_erased<A, P>(
    source: &dyn Any,
    context: &OriginalPredictionCopyContext,
) -> Result<PreparedLane<Box<dyn Any>>, StartupCause>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    let source = source
        .downcast_ref::<P::LaneState>()
        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
    let parts = [
        size_of::<P::LaneState>(),
        size_of::<Box<dyn Any>>(),
        size_of::<PreparedLane<Box<dyn Any>>>(),
        size_of::<Result<PreparedLane<Box<dyn Any>>, StartupCause>>(),
    ];
    context.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
    let copied = context.prediction::<A, P>(source)?;
    Ok(PreparedLane {
        value: Box::new(copied.value),
        host: copied.host,
    })
}
impl OriginalPredictionLane {
    pub(crate) fn proposal_capacity(&self) -> usize {
        self.proposal_capacity
    }
    pub(crate) fn class(&self) -> eredu_runtime::SpeculativeStrategyClass {
        self.class
    }
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }
    pub(crate) fn occurrence_shape(
        &self,
    ) -> eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape {
        self.shape
    }
    pub(crate) fn initial_frontier(&self) -> u64 {
        self.initial_frontier
    }
    pub(crate) fn prefill_alignment(&self) -> eredu_core::speculative::PredictionPrefillAlignment {
        self.alignment
    }
    pub(crate) fn copy_context(&self) -> &OriginalPredictionCopyContext {
        &self.copy
    }

    /// Performs the actual independent copy without requiring infallible Clone.
    /// This must run at the settled boundary before an invocation opens its Scope.
    pub(crate) fn try_copy(&self) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, StartupCause>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<OriginalPredictionCopyContext>(),
            OriginalPredictionStartupContext::copy_error_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        self.preparation
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let result = (|| {
            self.copy.validate(&self.preparation, &self.origin)?;
            let copied = (self.copy_erased)(self.state.as_ref(), &self.copy)?;
            Ok(Self {
                state: copied.value,
                origin: self.origin.clone(),
                depth: self.depth,
                class: self.class,
                proposal_capacity: self.proposal_capacity,
                preparation: self.preparation.clone(),
                shape: self.shape,
                alignment: self.alignment,
                initial_frontier: self.initial_frontier,
                copy: self.copy.clone(),
                copy_erased: self.copy_erased,
                _host: copied.host,
            })
        })();
        result.map_err(|cause| OriginalPredictionStartupContext::failure(&self.preparation, cause))
    }
    /// Borrows exact typed payload after authenticating the loaded source/header.
    /// No new Box, native handle, invocation, or copy account is created.
    pub(crate) fn state_mut<L: 'static>(
        &mut self,
        preparation: &PreparedSemanticSource,
        origin: &ReplicatedTextControlOrigin,
    ) -> Result<&mut L, StartupCause> {
        self.copy.validate(preparation, origin)?;
        self.state
            .downcast_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))
    }
}
