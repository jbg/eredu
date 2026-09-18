//! Fresh prediction-lane state from the actual loaded extension and source H.
use super::{MlxEmbeddedPredictionMaterializer, OwnedPredictionCache};
use super::original_copy::{OriginalPredictionCopyContext, CopyErased, copy_erased};
use crate::backend::{
    OriginalCopyEnvironment, OriginalCopyEnvironmentError, PreparedOriginalCopyEnvironmentError,
    error::Error,
    managed_memory::NativeMemoryRetention,
    nn::{shared::MlxNeuralBackend, workspace::MlxMetalWorkspaceMechanisms},
    runtime::cache::{
        copy_original_pooling,
        kv::CompressedLatentCache,
        state::{
            MlxHybridState, MlxPoolingAttentionCache, OriginalResidentState,
            PreparedPoolingAttentionCopy, PreparedResidentDecoderCopy,
            ResidentDecoderPreparationError, copy_original_resident_state,
        },
    },
};
use eredu_architectures::prediction_extension::{
    MaterializedPredictionExecutor, PredictionExtensionMaterializer, PredictionStateStartupFactory,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    SelectedSpeculativeRealization, SpeculativeStrategyClass,
    replicated_session::{PreparedControlBindingError, ReplicatedTextControlOrigin},
    working_memory::{PreparedSemanticSource, WorkingMemoryError},
};
use safemlx::{PrefillRootsRuntime, Stream, StreamCopyCause, StreamCopyPlan, StreamCopyError};
use std::{
    any::Any,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

/// A closed destination, deliberately without Clone or an unowned state exit.
/// The erased state/Box retires before exact source provenance and constructor H.
/// Future invocation/checkpoint adapters must borrow this owner, not ordinary-clone it.
pub(crate) struct OriginalPredictionLane {
    pub(super) state: Box<dyn Any>,
    pub(super) origin: ReplicatedTextControlOrigin,
    pub(super) depth: usize,
    pub(super) class: SpeculativeStrategyClass,
    pub(super) proposal_capacity: usize,
    pub(super) preparation: PreparedSemanticSource,
    pub(super) shape: eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape,
    pub(super) alignment: eredu_core::speculative::PredictionPrefillAlignment,
    pub(super) initial_frontier: u64,
    pub(super) copy: OriginalPredictionCopyContext,
    pub(super) copy_erased: CopyErased,
    pub(super) _host: HostPreparationAuthority,
}
impl std::fmt::Debug for OriginalPredictionLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalPredictionLane")
            .field("depth", &self.depth)
            .field("class", &self.class)
            .field("proposal_capacity", &self.proposal_capacity)
            .finish_non_exhaustive()
    }
}

// Provisional destinations retire before the earlier completed leaf accounts
// if a later source/copy fails. This holder never escapes numerical payloads.
struct PoolingRows {
    value: Vec<OwnedPredictionCache<MlxPoolingAttentionCache>>,
    copies: Vec<HostPreparationAuthority>,
}
struct PoolingCustody {
    _copies: Vec<HostPreparationAuthority>,
    _host: HostPreparationAuthority,
}

/// Typed construction result; payload retirement always precedes constructor H.
pub(crate) struct PreparedLane<T> {
    pub(super) value: T,
    pub(super) host: HostPreparationAuthority,
}

impl<T> PreparedLane<T> {
    /// Transfers already completed payload and its paying host custody together.
    /// The destination must destroy payload before the returned authority.
    pub(crate) fn into_parts(self) -> (T, HostPreparationAuthority) {
        (self.value, self.host)
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum StartupCause {
    #[error("{0}")]
    Backend(#[from] Error),
    #[error(transparent)]
    Source(#[from] PreparedControlBindingError),
    #[error(transparent)]
    Stream(#[from] StreamCopyCause),
    #[error(transparent)]
    CopyEnvironment(#[from] OriginalCopyEnvironmentError),
    #[error(transparent)]
    RetainedCopyEnvironment(#[from] PreparedOriginalCopyEnvironmentError),
    #[error(transparent)]
    PreparedStream(#[from] StreamCopyError<HostPreparationAuthority>),
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Decoder(#[from] ResidentDecoderPreparationError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: StartupCause,
    // BackendFailure frees its erasure shell before this retained source/account.
    _funding: HostMetadataFunding,
}
fn memory(cause: WorkingMemoryError) -> StartupCause {
    Error::PrefillControl(cause).into()
}
fn add(a: usize, b: usize) -> Result<usize, StartupCause> {
    a.checked_add(b)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

/// Constructed only by the actual idle MlxModelSession after header authentication.
/// This grants host construction and the existing independent-copy admission only;
/// it grants no target/draft invocation or ordinary native-memory owner.
pub(crate) struct OriginalPredictionStartupContext<'a> {
    environment: &'a OriginalCopyEnvironment<'a>,
    initialized: PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    preparation: &'a PreparedSemanticSource,
    host: HostPreparationAuthority,
}
impl<'a> OriginalPredictionStartupContext<'a> {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<OriginalPredictionLane>(),
            size_of::<Option<OriginalPredictionLane>>(),
            size_of::<Result<Option<OriginalPredictionLane>, Error>>(),
            size_of::<Result<OriginalPredictionLane, StartupCause>>(),
            size_of::<StartupCause>(),
            size_of::<Failure>(),
            size_of::<PreparedSemanticSource>(),
            size_of::<ReplicatedTextControlOrigin>(),
            size_of::<Result<ReplicatedTextControlOrigin, PreparedControlBindingError>>(),
            size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>, StreamCopyCause>>(),
            size_of::<PrefillRootsRuntime>(),
            size_of::<Result<PrefillRootsRuntime, Error>>(),
            size_of::<Option<MlxMetalWorkspaceMechanisms>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?,
            BackendFailure::source_retention_peak_bytes::<Failure>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn copy_error_control_bytes() -> Option<usize> {
        let parts = [size_of::<StartupCause>(), size_of::<Failure>(),
            size_of::<Error>(), size_of::<HostMetadataFunding>(),
            BackendFailure::source_retention_peak_bytes::<Failure>()?];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Caller has already paid control_bytes and authenticated this header/source.
    pub(crate) fn new(
        environment: &'a OriginalCopyEnvironment<'a>,
        initialized: PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        preparation: &'a PreparedSemanticSource,
    ) -> Self {
        Self {
            environment,
            initialized,
            mechanisms,
            preparation,
            host: HostPreparationAuthority::retain(preparation.metadata_funding().clone()),
        }
    }
    pub(crate) fn failure(
        preparation: &PreparedSemanticSource,
        cause: StartupCause,
    ) -> Error {
        Error::StorageSource(BackendFailure::from_error(Failure {
            cause,
            _funding: preparation.metadata_funding().clone(),
        }))
    }
    fn reserve(&self, bytes: usize) -> Result<(), StartupCause> {
        self.preparation
            .metadata_funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        Ok(())
    }
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, StartupCause> {
        let payload = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let frames = [
            size_of::<Vec<T>>(),
            size_of::<PreparedLane<Vec<T>>>(),
            size_of::<Result<PreparedLane<Vec<T>>, StartupCause>>(),
            size_of::<Result<Vec<T>, StartupCause>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<T>(),
            size_of::<std::ops::Range<usize>>(),
        ];
        let bytes = frames.into_iter().try_fold(size_of_val(&frames), add)?;
        self.reserve(add(payload, bytes)?)?;
        let mut result = Vec::new();
        result.try_reserve_exact(count)?;
        Ok(result)
    }
    /// Architecture owns the dispatch and exact source membership. Selected metadata
    /// is borrowed from that same loaded owner; no caller-supplied DTO is accepted.
    pub(crate) fn prepare<A, P>(
        &mut self,
        extension: &P,
        selected: &SelectedSpeculativeRealization,
        origin: Result<ReplicatedTextControlOrigin, PreparedControlBindingError>,
        stream: &Stream,
    ) -> Result<OriginalPredictionLane, StartupCause>
    where
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
    {
        let origin = origin?;
        let strategy = selected.requirements().strategy();
        let depth = extension.depth();
        if !matches!(
            strategy.class(),
            SpeculativeStrategyClass::EmbeddedSequential | SpeculativeStrategyClass::EmbeddedFused
        ) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let stream = StreamCopyPlan::<()>::capture(stream)?;
        self.reserve(
            stream
                .control_bytes()
                .and_then(|n| n.checked_add(stream.source_comparison_control_bytes()?))
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        )?;
        if !stream.matches_source(self.environment.stream()) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        // Reserve the actual erased Box before constructing any destination.
        let frames = [
            size_of::<P::LaneState>(),
            size_of::<PreparedLane<P::LaneState>>(),
            size_of::<Result<PreparedLane<P::LaneState>, StartupCause>>(),
            size_of::<Box<dyn Any>>(),
            size_of::<&P>(),
            size_of::<&SelectedSpeculativeRealization>(),
        ];
        self.reserve(frames.into_iter().try_fold(size_of_val(&frames), add)?)?;
        let shape = extension.occurrence_shape().ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
        let alignment = extension.prefill_alignment();
        let copy = OriginalPredictionCopyContext::prepare(self.environment, &self.initialized,
            self.mechanisms, self.preparation, &origin, &self.host)?;
        let mut prepared = extension.prepare_new_state(self)?;
        let initial_frontier = extension.prefill_frontier(&mut prepared.value).map_err(Error::from)?;
        Ok(OriginalPredictionLane {
            state: Box::new(prepared.value),
            origin,
            depth,
            class: strategy.class(),
            proposal_capacity: strategy.proposal_capacity().get(),
            preparation: self.preparation.clone(),
            shape, alignment, initial_frontier, copy, copy_erased: copy_erased::<A, P>,
            _host: prepared.host,
        })
    }
}
impl PredictionStateStartupFactory<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
    for OriginalPredictionStartupContext<'_>
{
    type Error = StartupCause;
    type Prepared<T> = PreparedLane<T>;

    fn sequential(
        &mut self,
        count: usize,
    ) -> Result<Self::Prepared<Vec<OwnedPredictionCache<CompressedLatentCache>>>, Self::Error> {
        let mut value = self.vector(count)?;
        // The existing constructor contains only inline empty slots/scalars.
        // No native handle, owner, array or paging namespace is born here.
        for _ in 0..count {
            value.push(MlxEmbeddedPredictionMaterializer::sequential_state());
        }
        Ok(PreparedLane {
            value,
            host: self.host.clone(),
        })
    }
    fn pooling(
        &mut self,
        source: &[OwnedPredictionCache<MlxPoolingAttentionCache>],
    ) -> Result<Self::Prepared<Vec<OwnedPredictionCache<MlxPoolingAttentionCache>>>, Self::Error>
    {
        let frames = [
            PreparedPoolingAttentionCopy::preparation_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            HostPreparationAuthority::retention_bytes::<PoolingCustody>()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            size_of::<PoolingRows>(),
            size_of::<PoolingCustody>(),
            size_of::<std::slice::Iter<'_, OwnedPredictionCache<MlxPoolingAttentionCache>>>(),
            size_of::<(MlxPoolingAttentionCache, HostPreparationAuthority)>(),
            size_of::<
                Result<
                    crate::backend::runtime::cache::PreparedPredictionCacheCopy<
                        MlxPoolingAttentionCache,
                    >,
                    Error,
                >,
            >(),
        ];
        self.reserve(frames.into_iter().try_fold(size_of_val(&frames), add)?)?;
        // Preserve source validation before the first destination. Populated
        // resident leaves proceed through the same admitted isolated-copy worker;
        // paged sources still require their distinct manager-copy mechanism.
        for slot in source {
            slot.inner()
                .prepare_isolated_copy_fixed()
                .map_err(|_| memory(WorkingMemoryError::UnknownBound))?;
        }
        let mut rows = PoolingRows {
            value: self.vector(source.len())?,
            copies: self.vector(source.len())?,
        };
        for slot in source {
            let (value, custody) = copy_original_pooling(
                slot.inner(),
                self.environment,
                &self.initialized,
                self.mechanisms,
                self.preparation.metadata_funding(),
                self.preparation.capacity_bytes(),
            )?
            .into_parts();
            rows.value.push(OwnedPredictionCache::new(
                value,
                NativeMemoryRetention::default(),
            ));
            rows.copies.push(custody);
        }
        let host = HostPreparationAuthority::retain(PoolingCustody {
            _copies: rows.copies,
            _host: self.host.clone(),
        });
        Ok(PreparedLane {
            value: rows.value,
            host,
        })
    }

    fn model(
        &mut self,
        source: &MlxHybridState,
    ) -> Result<Self::Prepared<MlxHybridState>, Self::Error> {
        let frames = [
            size_of::<PreparedResidentDecoderCopy<'_>>(),
            size_of::<OriginalResidentState>(),
            size_of::<PreparedLane<MlxHybridState>>(),
            size_of::<Result<PreparedLane<MlxHybridState>, StartupCause>>(),
        ];
        self.reserve(frames.into_iter().try_fold(size_of_val(&frames), add)?)?;
        let plan = PreparedResidentDecoderCopy::hybrid_fixed(source)?;
        let copied = copy_original_resident_state(
            plan,
            self.environment,
            &self.initialized,
            self.mechanisms,
            self.preparation.metadata_funding(),
            &self.host,
            self.preparation.capacity_bytes(),
        )?;
        let OriginalResidentState::Hybrid(value) = copied else {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        };
        Ok(PreparedLane {
            value,
            host: self.host.clone(),
        })
    }
}
