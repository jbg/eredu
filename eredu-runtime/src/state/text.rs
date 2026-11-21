//! Portable owners of the mutable state used by replicated text execution.

use super::{HybridCacheStorage, HybridState, KeyValueState, PoolingAttentionState};
use crate::{
    cache::live::{CacheExport, CacheResidencyManager, RestoredCache},
    replicated_session::StateSnapshotMemoryCosts,
    CacheResidencyReport, LayerRuntimeState,
};
use eredu_core::{
    cache::{
        mechanisms::{
            CacheTensorStorage, PagedKeyValueStorage, PoolingCacheStorage, ResidentKeyValueStorage,
        },
        PromptCacheDescriptor,
    },
    execution_control::SnapshotEstimate,
    BackendFailure,
};
use eredu_nn::{NeuralBackend, PoolingAttentionCache};

/// Mutable state owned and advanced by the portable text driver.
pub trait PortableTextState<B, C>: LayerRuntimeState<B> + Sized
where
    B: NeuralBackend<Tensor = C::Value>,
    C: CacheTensorStorage,
{
    /// Creates an explicit state-free owner when the profile supports it.
    fn stateless() -> Option<Self> {
        None
    }
    /// Truncates to the requested committed prefix if reversible.
    fn retain_causal_prefix(
        &mut self,
        _length: i32,
        _stream: &C::Queue,
    ) -> Result<bool, BackendFailure> {
        Ok(false)
    }

    /// Whether every state component can be copied independently.
    fn supports_isolated_snapshot(&self) -> bool {
        false
    }
    /// Copies all mutable state for an independently advancing session.
    fn isolated_snapshot(&self, _stream: &C::Queue) -> Result<Self, BackendFailure> {
        Err(BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "complete isolated snapshot is unsupported for this state realization",
        )))
    }
    /// Retained storage outside the tensor roots, including page catalogs.
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        Some(0)
    }
    /// Additional auxiliary storage through the admitted token span.
    fn isolated_snapshot_auxiliary_growth(&self, _additional: u64) -> Option<u64> {
        Some(0)
    }
    /// Bounds the metadata and tensor storage of a complete copy.
    fn isolated_snapshot_estimate(&self) -> Option<SnapshotEstimate> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        let owner = u64::try_from(std::mem::size_of::<Self>()).ok()?;
        let Some(layout) = self.optional_layout() else {
            return Some(SnapshotEstimate {
                retained_bytes: owner,
                copy_bytes: owner,
            });
        };
        let costs = StateSnapshotMemoryCosts::for_tensors(C::snapshot_tensor_costs()?);
        let tensors = self
            .retained_arrays()
            .into_iter()
            .try_fold(0u64, |bytes, array| {
                let payload = C::bytes(array);
                bytes.checked_add(costs.tensor_bytes(payload, C::shape(array).len())?)
            })?;
        costs.estimate(
            layout,
            owner,
            tensors,
            self.isolated_snapshot_auxiliary_bytes()?,
        )
    }
    /// Known retained position capacity through the supplied continuation.
    fn continuation_capacity_bound(&self, _additional: u64) -> Option<u64> {
        None
    }
    /// Bounds state growth through the admitted continuation.
    fn isolated_snapshot_growth(&self, additional: u64) -> Option<u64> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        let Some(layout) = self.optional_layout() else {
            return Some(0);
        };
        let frontier = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        let capacity = frontier.max(self.continuation_capacity_bound(additional)?);
        i32::try_from(capacity).ok()?;
        let costs = StateSnapshotMemoryCosts::for_tensors(C::snapshot_tensor_costs()?);
        costs.growth(
            layout,
            capacity,
            self.isolated_snapshot_auxiliary_growth(additional)?,
        )
    }
    /// Returns the decoder frontier of the realized state.
    fn offset(&self) -> i32;

    /// Returns the frontier of each locally owned layer.
    fn layer_positions(&self) -> Vec<i32>;
    /// Constructs resident state for an exact layout.
    fn device(layout: crate::StateLayout) -> Result<Self, BackendFailure>;
    /// Constructs the selected state around the supplied live cache.
    fn allocate(
        selected: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<C>>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, BackendFailure>;
    /// Installs checkpoint pages and fixed state at the admitted placement.
    fn install_restored_cache(
        selected: &crate::state::StateAllocation<'_>,
        restored: RestoredCache<C>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
        prefix: i32,
        layer_frontiers: &[i32],
    ) -> Result<Self, BackendFailure>;
    /// Finalizes cache tails and collects the inventory for publication.
    fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
        layer_frontiers: &[i32],
    ) -> Result<CacheExport<C>, BackendFailure>;
    /// Reports residency measured by the active portable cache.
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, BackendFailure>;
    /// Borrows all native roots retained by this state.
    fn retained_arrays(&self) -> Vec<&C::Tensor>;
    /// Captures a rollback checkpoint of live mutable state.
    fn deep_checkpoint(&self) -> Result<Self, BackendFailure>;
    /// Forks canonical state with an independent transaction identity.
    fn fork_prediction_target_state(&self, stream: &C::Queue) -> Result<Self, BackendFailure>;
    /// Restores the live state from a compatible rollback checkpoint.
    fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &C::Queue,
    ) -> Result<(), BackendFailure>;
}

impl<B, C> PortableTextState<B, C> for KeyValueState<C>
where
    B: NeuralBackend<Tensor = C::Value>,
    C: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    fn retain_causal_prefix(
        &mut self,
        length: i32,
        stream: &C::Queue,
    ) -> Result<bool, BackendFailure> {
        KeyValueState::<C>::retain_causal_prefix(self, length, stream)
            .map_err(BackendFailure::from_error)
    }

    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }

    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }

    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }

    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }

    fn isolated_snapshot(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        self.isolated_snapshot(stream)
            .map_err(BackendFailure::from_error)
    }

    fn offset(&self) -> i32 {
        KeyValueState::<C>::offset(self)
    }

    fn layer_positions(&self) -> Vec<i32> {
        self.as_ref()
            .iter()
            .map(eredu_nn::AttentionCache::offset)
            .collect()
    }

    fn device(layout: crate::StateLayout) -> Result<Self, BackendFailure> {
        KeyValueState::device(layout).map_err(BackendFailure::from_error)
    }

    fn allocate(
        selected: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<C>>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, BackendFailure> {
        KeyValueState::<C>::allocate_layers(selected, manager, rank, global_layer_start)
            .map_err(BackendFailure::from_error)
    }

    fn install_restored_cache(
        selected: &crate::state::StateAllocation<'_>,
        restored: RestoredCache<C>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
        _prefix: i32,
        _layer_frontiers: &[i32],
    ) -> Result<Self, BackendFailure> {
        let (manager, _) = restored;
        let state =
            KeyValueState::<C>::allocate_layers(selected, Some(manager), rank, global_layer_start)
                .map_err(BackendFailure::from_error)?;
        Ok(state)
    }

    fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
        _layer_frontiers: &[i32],
    ) -> Result<CacheExport<C>, BackendFailure> {
        KeyValueState::<C>::cache_export(self, descriptor).map_err(BackendFailure::from_error)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, BackendFailure> {
        KeyValueState::<C>::residency_report(self).map_err(BackendFailure::from_error)
    }

    fn retained_arrays(&self) -> Vec<&C::Tensor> {
        KeyValueState::<C>::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, BackendFailure> {
        self.deep_clone_state().map_err(BackendFailure::from_error)
    }

    fn fork_prediction_target_state(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        KeyValueState::<C>::fork_prediction_target_state(self, stream)
            .map_err(BackendFailure::from_error)
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &C::Queue,
    ) -> Result<(), BackendFailure> {
        KeyValueState::<C>::restore_checkpoint(self, checkpoint, stream)
            .map_err(BackendFailure::from_error)
    }
}

impl<B, C> PortableTextState<B, C> for HybridState<C>
where
    B: NeuralBackend<Tensor = C::Value>,
    C: HybridCacheStorage,
{
    fn stateless() -> Option<Self> {
        Some(HybridState::<C>::stateless())
    }

    fn retain_causal_prefix(
        &mut self,
        length: i32,
        stream: &C::Queue,
    ) -> Result<bool, BackendFailure> {
        HybridState::<C>::retain_causal_prefix(self, length, stream)
            .map_err(BackendFailure::from_error)
    }

    fn isolated_snapshot_estimate(&self) -> Option<SnapshotEstimate> {
        crate::text_session_driver::TextSnapshotStorage::snapshot_estimate(self)
    }

    fn isolated_snapshot_growth(&self, additional: u64) -> Option<u64> {
        crate::text_session_driver::TextSnapshotStorage::snapshot_growth(self, additional)
    }

    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }

    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }

    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }

    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }

    fn isolated_snapshot(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        self.isolated_snapshot(stream)
            .map_err(BackendFailure::from_error)
    }

    fn offset(&self) -> i32 {
        HybridState::<C>::offset(self)
    }

    fn layer_positions(&self) -> Vec<i32> {
        self.layers().iter().map(|layer| layer.position()).collect()
    }

    fn device(layout: crate::StateLayout) -> Result<Self, BackendFailure> {
        HybridState::device(layout, &[]).map_err(BackendFailure::from_error)
    }

    fn allocate(
        selected: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<C>>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, BackendFailure> {
        HybridState::<C>::allocate_layers(selected, manager, rank, global_layer_start)
            .map_err(BackendFailure::from_error)
    }

    fn install_restored_cache(
        selected: &crate::state::StateAllocation<'_>,
        restored: RestoredCache<C>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
        _prefix: i32,
        layer_frontiers: &[i32],
    ) -> Result<Self, BackendFailure> {
        let (manager, tensors) = restored;
        let mut state =
            HybridState::<C>::allocate_layers(selected, Some(manager), rank, global_layer_start)
                .map_err(BackendFailure::from_error)?;
        state
            .restore_prompt_cache_state(tensors, layer_frontiers)
            .map_err(BackendFailure::from_error)?;
        Ok(state)
    }

    fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
        layer_frontiers: &[i32],
    ) -> Result<CacheExport<C>, BackendFailure> {
        HybridState::<C>::cache_export(self, descriptor, layer_frontiers)
            .map_err(BackendFailure::from_error)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, BackendFailure> {
        HybridState::<C>::residency_report(self).map_err(BackendFailure::from_error)
    }

    fn retained_arrays(&self) -> Vec<&C::Tensor> {
        HybridState::<C>::retained_arrays(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, BackendFailure> {
        self.deep_clone_state().map_err(BackendFailure::from_error)
    }

    fn fork_prediction_target_state(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        HybridState::<C>::fork_prediction_target_state(self, stream)
            .map_err(BackendFailure::from_error)
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &C::Queue,
    ) -> Result<(), BackendFailure> {
        HybridState::<C>::restore_checkpoint(self, checkpoint, stream)
            .map_err(BackendFailure::from_error)
    }
}

impl<B, C> PortableTextState<B, C> for PoolingAttentionState<B, C>
where
    B: NeuralBackend<Tensor = C::Value>,
    C: PoolingCacheStorage,
{
    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }

    fn isolated_snapshot(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        self.isolated_snapshot(stream)
            .map_err(BackendFailure::from_error)
    }

    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }

    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }

    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }

    fn offset(&self) -> i32 {
        self.as_ref().first().map_or(0, |layer| layer.offset())
    }

    fn layer_positions(&self) -> Vec<i32> {
        self.as_ref()
            .iter()
            .map(PoolingAttentionCache::offset)
            .collect()
    }

    fn device(layout: crate::StateLayout) -> Result<Self, BackendFailure> {
        PoolingAttentionState::device(layout).map_err(BackendFailure::from_error)
    }

    fn allocate(
        selected: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<C>>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, BackendFailure> {
        match manager {
            Some(manager) => Self::paged(
                selected.layout().clone(),
                manager,
                global_layer_start,
                0,
                rank,
            ),
            None => Self::device(selected.layout().clone()),
        }
        .map_err(BackendFailure::from_error)
    }

    fn install_restored_cache(
        selected: &crate::state::StateAllocation<'_>,
        restored: RestoredCache<C>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
        prefix: i32,
        layer_frontiers: &[i32],
    ) -> Result<Self, BackendFailure> {
        Self::restore_paged(
            selected.layout().clone(),
            restored,
            global_layer_start,
            prefix,
            rank,
            layer_frontiers,
        )
        .map_err(BackendFailure::from_error)
    }

    fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
        _layer_frontiers: &[i32],
    ) -> Result<CacheExport<C>, BackendFailure> {
        self.cache_export(descriptor)
            .map_err(BackendFailure::from_error)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, BackendFailure> {
        self.residency_report().map_err(BackendFailure::from_error)
    }

    fn retained_arrays(&self) -> Vec<&C::Tensor> {
        self.retained_arrays()
    }

    fn deep_checkpoint(&self) -> Result<Self, BackendFailure> {
        self.checkpoint_clone_state()
            .map_err(BackendFailure::from_error)
    }

    fn fork_prediction_target_state(&self, stream: &C::Queue) -> Result<Self, BackendFailure> {
        self.fork_prediction_target_state(stream)
            .map_err(BackendFailure::from_error)
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &C::Queue,
    ) -> Result<(), BackendFailure> {
        self.restore_checkpoint(checkpoint, stream)
            .map_err(BackendFailure::from_error)
    }
}

/// Opens the selected standalone live cache and realizes its portable state owner.
pub(crate) fn allocate_selected_text_state<B, C, S>(
    selected: &crate::SelectedStateRealization,
) -> Result<S, BackendFailure>
where
    B: NeuralBackend<Tensor = C::Value>,
    C: CacheTensorStorage,
    S: PortableTextState<B, C>,
{
    use crate::cache::live::SelectedLiveCache;
    let allocation = selected.allocation()?;
    let cache = CacheResidencyManager::<C>::open(selected)?;
    S::allocate(&allocation, cache, None, 0)
}

/// Realizes the selected standalone state of a routed predictor.
pub(crate) fn allocate_selected_prediction_state<B, M>(
    selected: &crate::SelectedStateRealization,
) -> Result<M::ModelState, M::Error>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    allocate_selected_text_state::<B, M::ModelCacheStorage, M::ModelState>(selected)
        .map_err(Into::into)
}
