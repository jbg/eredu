//! Bind the portable state lifecycle to the native cache storage of a text session.

use super::{
    ReplicatedTextPrimitives, ReplicatedTextSnapshotPrimitives, ReplicatedTextState,
    SelectedStateRealization,
};
use crate::{
    cache::live::{CacheExport, CacheResidencyManager, CacheRestore, RestoredCache},
    executable_binding::session_erasure::TextStateReport,
    state::{PortableTextState, StateAllocation},
    CacheResidencyReport,
};
use eredu_core::{
    cache::{
        mechanisms::{CacheTensorStorage, LiveCacheBackend},
        CacheRankIdentity, PromptCacheDescriptor, PromptCacheModelIdentity,
    },
    execution_control::SnapshotEstimate,
    BackendFailure, SubmissionBackend,
};
use eredu_nn::{NeuralBackend, Tensor};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;
/// The state type selected by portable route dispatch and the native operations it needs.
pub trait TextStateStorage<B>
where
    B: SubmissionBackend<Executor = Context<B>>,
{
    /// Native cache storage, transfer and tensor operations.
    type CacheStorage: CacheTensorStorage<Value = B::Tensor, Queue = Context<B>>;
    /// Portable state owner selected for the route.
    type Owner: PortableTextState<B, Self::CacheStorage>;
    /// Error at the concrete session boundary.
    type NativeError;

    /// Converts a state operation failure while preserving its cause.
    fn state_failure(error: BackendFailure) -> Self::NativeError;
}

/// Report assembled from the live state and the storage observations it returns.
pub struct PortableTextStateReport {
    offset: i32,
    residency: Option<CacheResidencyReport>,
}

impl TextStateReport for PortableTextStateReport {
    fn offset(&self) -> i32 {
        self.offset
    }
    fn residency(&self) -> Option<CacheResidencyReport> {
        self.residency.clone()
    }
}

impl<B, N> ReplicatedTextState<B> for N
where
    B: SubmissionBackend<Executor = Context<B>>,
    N: TextStateStorage<B>,
    Context<B>: Sized,
{
    type State = N::Owner;
    type StateCheckpoint = N::Owner;
    type StateReport = PortableTextStateReport;
    type Error = N::NativeError;
    type LiveCache = CacheResidencyManager<N::CacheStorage>;
    type CacheRestore = CacheRestore<N::CacheStorage>;
    type CacheExport = CacheExport<N::CacheStorage>;

    fn parameter_storage_error(error: BackendFailure) -> Self::Error {
        N::state_failure(error)
    }

    fn realize_stateless_state(&mut self) -> Option<Self::State> {
        N::Owner::stateless()
    }

    fn allocate_state(
        &mut self,
        selected: &StateAllocation<'_>,
        cache: Option<Self::LiveCache>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
        _: &Context<B>,
    ) -> Result<Self::State, Self::Error> {
        N::Owner::allocate(selected, cache, rank, global_layer_start).map_err(N::state_failure)
    }

    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        _: &Context<B>,
    ) -> Result<Self::StateCheckpoint, Self::Error> {
        state.deep_checkpoint().map_err(N::state_failure)
    }

    fn restore_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        context: &Context<B>,
    ) -> Result<(), Self::Error> {
        state
            .restore_checkpoint(&checkpoint, context)
            .map_err(N::state_failure)
    }

    fn fork_prediction_target_state(
        &mut self,
        state: &Self::State,
        _: &SelectedStateRealization,
        _: Option<&PromptCacheModelIdentity>,
        context: &Context<B>,
    ) -> Result<Self::State, Self::Error> {
        state
            .fork_prediction_target_state(context)
            .map_err(N::state_failure)
    }

    fn install_restored_cache(
        &mut self,
        restored: RestoredCache<N::CacheStorage>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
        prefix: i32,
        layer_frontiers: &[i32],
        selected: &StateAllocation<'_>,
    ) -> Result<Self::State, Self::Error> {
        N::Owner::install_restored_cache(
            selected,
            restored,
            rank,
            global_layer_start,
            prefix,
            layer_frontiers,
        )
        .map_err(N::state_failure)
    }

    fn cache_export(
        &mut self,
        state: &mut Self::State,
        descriptor: &PromptCacheDescriptor,
        layer_frontiers: &[i32],
    ) -> Result<Self::CacheExport, Self::Error> {
        state
            .cache_export(descriptor, layer_frontiers)
            .map_err(N::state_failure)
    }

    fn publish_resident_cache(
        &self,
        state: &mut Self::State,
        store: &dyn eredu_core::cache::PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &eredu_core::cache::PromptCacheOptions,
        context: &Context<B>,
    ) -> Result<eredu_core::cache::PromptCacheManifest, Self::Error> {
        state
            .publish_resident_cache(store, descriptor, prefix, options, context)
            .map_err(N::state_failure)
    }

    fn install_resident_cache(
        &self,
        state: &mut Self::State,
        restore: &crate::PromptCacheRestore<'_>,
        frontiers: &[i32],
        context: &Context<B>,
    ) -> Result<(), Self::Error> {
        state
            .install_resident_cache(restore, frontiers, context)
            .map_err(N::state_failure)
    }

    fn state_roots(state: &Self::State) -> Vec<&B::Tensor> {
        state
            .retained_arrays()
            .into_iter()
            .map(N::CacheStorage::borrow_value)
            .collect()
    }

    fn state_report(&self, state: &Self::State) -> Result<Self::StateReport, Self::Error> {
        Ok(PortableTextStateReport {
            offset: state.offset(),
            residency: state.residency_report().map_err(N::state_failure)?,
        })
    }
}

impl<B, N> ReplicatedTextSnapshotPrimitives<B> for N
where
    B: SubmissionBackend<Executor = Context<B>>,
    N: TextStateStorage<B>
        + ReplicatedTextPrimitives<
            B,
            State = <N as TextStateStorage<B>>::Owner,
            Error = <N as TextStateStorage<B>>::NativeError,
        >,
    Context<B>: Sized,
{
    fn estimate_snapshot_state(&self, state: &N::Owner) -> Option<SnapshotEstimate> {
        state.isolated_snapshot_estimate()
    }

    fn estimate_reset_state(&self, state: &N::Owner) -> Option<SnapshotEstimate> {
        let metadata = state.isolated_snapshot_estimate()?;
        let initial = state.isolated_snapshot_growth(0)?;
        Some(SnapshotEstimate {
            retained_bytes: metadata.retained_bytes.checked_add(initial)?,
            copy_bytes: metadata.copy_bytes.checked_add(initial)?,
        })
    }

    fn estimate_snapshot_growth(&self, state: &N::Owner, additional: u64) -> Option<u64> {
        state.isolated_snapshot_growth(additional)
    }

    fn copy_snapshot_state(
        &mut self,
        state: &N::Owner,
        context: &Context<B>,
    ) -> Result<N::Owner, N::NativeError> {
        let copied = state.isolated_snapshot(context).map_err(N::state_failure)?;
        N::CacheStorage::settle_tensors(&copied.retained_arrays()).map_err(N::state_failure)?;
        Ok(copied)
    }
}
