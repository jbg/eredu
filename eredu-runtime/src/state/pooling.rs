//! Portable local attention and compressed pooling state.

use super::{KeyValueLayerState, KeyValueStateError};
use crate::{
    cache::{
        live::{
            CacheExport, CacheResidencyError, CacheResidencyManager, PromptCacheStateArray,
            RestoredCache,
        },
        PagedKeyValueCache, PoolingCache, PoolingCacheState, ResidentKeyValueCache,
    },
    CacheResidencyReport, DeviceState, RuntimeLayerState, RuntimeState, StateLayout,
};
use eredu_core::{
    cache::{
        mechanisms::PoolingCacheStorage, CacheRankIdentity, LayerCachePolicy,
        PoolingStateComponent, PromptCacheDescriptor, StateTensorDimension, StateTensorOwner,
        StateTensorPresence, StateTensorRole,
    },
    BackendFailure,
};
use eredu_nn::{
    AttentionCache, Error as ComputeError, NeuralBackend, PoolingAttentionCache, PoolingOverlap,
    PoolingWindows, Tensor,
};
use std::collections::BTreeMap;

/// Failure to build, advance, or restore a pooling cache.
#[derive(Debug, thiserror::Error)]
pub enum PoolingStateError {
    /// The request does not match the selected pooling layout or frontier.
    #[error("{0}")]
    Invalid(String),
    /// A native tensor operation failed.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// A page catalog operation failed.
    #[error(transparent)]
    Cache(#[from] CacheResidencyError),
    /// A local key/value operation failed.
    #[error(transparent)]
    Local(#[from] KeyValueStateError),
    /// The pooling operator rejected the request.
    #[error(transparent)]
    Compute(#[from] ComputeError),
}

pub(crate) fn invalid(detail: impl Into<String>) -> PoolingStateError {
    PoolingStateError::Invalid(detail.into())
}

fn local_manager<M: PoolingCacheStorage>(
    cache: &KeyValueLayerState<M>,
) -> Option<&CacheResidencyManager<M>> {
    match cache {
        KeyValueLayerState::Paged(cache) => Some(cache.manager()),
        _ => None,
    }
}

fn restore_local_checkpoint<M: PoolingCacheStorage>(
    cache: &mut KeyValueLayerState<M>,
    previous: &KeyValueLayerState<M>,
    queue: &M::Queue,
) -> Result<(), PoolingStateError> {
    if let (KeyValueLayerState::Device(cache), KeyValueLayerState::Device(previous)) =
        (&mut *cache, previous)
    {
        *cache = previous.isolated_snapshot(queue)?;
        Ok(())
    } else {
        Ok(cache.restore_checkpoint(previous, queue)?)
    }
}

/// Portable realization of bounded local keys plus zero, one, or two
/// append-only pooling streams.
#[derive(Debug, Clone)]
pub enum PoolingAttentionLayer<M: PoolingCacheStorage> {
    /// Bounded local keys only.
    Local(KeyValueLayerState<M>),
    /// Local keys plus one compressed stream.
    Compressed {
        /// Bounded local keys.
        local: KeyValueLayerState<M>,
        /// Compressed attention stream.
        pool: PoolingCache<M>,
    },
    /// Local keys, compressed attention, and sparse-index streams.
    Sparse {
        /// Bounded local keys.
        local: KeyValueLayerState<M>,
        /// Compressed attention stream.
        pool: PoolingCache<M>,
        /// Sparse-index stream.
        index_pool: PoolingCache<M>,
    },
}

/// Complete pooling-attention state using the selected neural and storage contracts.
pub type PoolingAttentionState<B, M> = DeviceState<B, PoolingAttentionLayer<M>>;

impl<B, M> DeviceState<B, PoolingAttentionLayer<M>>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: PoolingCacheStorage,
{
    /// Materializes device-resident state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<Self, PoolingStateError> {
        DeviceState::create(layout, PoolingAttentionLayer::resident_from_policy)
    }

    /// Materializes paged local keys plus device pooling streams for every
    /// declared layer.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager<M>,
        global_layer_start: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, PoolingStateError> {
        DeviceState::create(layout, move |layer, policy| {
            let global_layer = global_layer_start
                .checked_add(layer)
                .ok_or_else(|| invalid("pooling-attention global layer index overflowed"))?;
            PoolingAttentionLayer::paged_from_policy(
                layer,
                policy,
                manager.clone(),
                global_layer,
                prefix_tokens,
                rank,
            )
        })
    }

    /// Restores the selected page catalog and fixed pooling streams into a new model state.
    pub fn restore_paged(
        layout: StateLayout,
        restored: RestoredCache<M>,
        global_layer_start: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
        layer_frontiers: &[i32],
    ) -> Result<Self, PoolingStateError> {
        if layer_frontiers.len() != layout.len() {
            return Err(invalid("pooling restore frontiers do not match the layout"));
        }
        let (manager, fixed) = restored;
        let mut tensors = BTreeMap::new();
        for tensor in fixed {
            if tensors
                .insert((tensor.owner, tensor.role), tensor.array)
                .is_some()
            {
                return Err(invalid("pooling prompt cache repeats a fixed-state tensor"));
            }
        }
        let mut state = Self::paged(layout, manager, global_layer_start, prefix_tokens, rank)?;
        for (ordinal, (cache, &frontier)) in
            state.as_mut().iter_mut().zip(layer_frontiers).enumerate()
        {
            cache.restore_prompt_cache_state(
                global_layer_start + ordinal,
                &mut tensors,
                frontier,
            )?;
        }
        if !tensors.is_empty() {
            return Err(invalid(
                "pooling prompt cache contains unexpected fixed-state tensors",
            ));
        }
        Ok(state)
    }

    /// Seals the local caches and exports fixed streams using their absolute layer identities.
    pub fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
    ) -> Result<CacheExport<M>, PoolingStateError> {
        let manager = self
            .as_ref()
            .first()
            .and_then(PoolingAttentionLayer::residency_manager)
            .ok_or_else(|| invalid("prompt-cache persistence requires paged pooling state"))?
            .clone();
        let fixed = self
            .as_ref()
            .iter()
            .try_fold(Vec::new(), |mut fixed, layer| {
                let KeyValueLayerState::Paged(cache) = layer.local() else {
                    return Err(invalid(
                        "prompt-cache persistence requires paged pooling state",
                    ));
                };
                if cache.manager().session_id() != manager.session_id() {
                    return Err(invalid("pooling state has inconsistent paging catalogs"));
                }
                fixed.extend(layer.prompt_cache_state_arrays(cache.global_layer()));
                Ok(fixed)
            })?;
        for layer in self.as_mut() {
            layer.finalize()?;
        }
        Ok(manager.cache_export(descriptor, fixed)?)
    }

    /// Returns aggregate cache telemetry for the current paging catalog.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, PoolingStateError> {
        Ok(self
            .as_ref()
            .iter()
            .find_map(PoolingAttentionLayer::residency_manager)
            .map(CacheResidencyManager::report)
            .transpose()?)
    }

    /// Borrows every array retained by this installed state.
    pub fn retained_arrays(&self) -> Vec<&M::Tensor> {
        self.as_ref()
            .iter()
            .flat_map(PoolingAttentionLayer::retained_arrays)
            .collect()
    }

    /// Checkpoints append-only views and retains all paged sliding history needed for rollback.
    pub fn checkpoint_clone_state(&self) -> Result<Self, PoolingStateError> {
        Self::create(self.layout().clone(), |layer, _| {
            self.as_ref()[layer].checkpoint_clone_state()
        })
    }

    /// Restores a checkpoint made with the same installed layout.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        queue: &M::Queue,
    ) -> Result<(), PoolingStateError> {
        if self.layout() != checkpoint.layout() || self.as_ref().len() != checkpoint.as_ref().len()
        {
            return Err(invalid(
                "pooling-attention checkpoint does not match the installed layout",
            ));
        }
        for (current, previous) in self.as_mut().iter_mut().zip(checkpoint.as_ref()) {
            PoolingAttentionCache::restore(current, previous, queue)?;
        }
        Ok(())
    }

    /// Reports whether all paged layers share one snapshot-compatible catalog.
    pub fn supports_isolated_snapshot(&self) -> bool {
        let mut manager = None;
        self.as_ref().iter().all(|cache| {
            cache.residency_manager().is_none_or(|paging| {
                *manager.get_or_insert(paging.session_id()) == paging.session_id()
            })
        })
    }

    /// Reports catalog storage charged to an isolated snapshot.
    pub fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.as_ref()
            .iter()
            .find_map(PoolingAttentionLayer::residency_manager)
            .map(CacheResidencyManager::isolated_snapshot_bytes)
            .unwrap_or(Some(0))
    }

    /// Bounds retained token capacity after a continuation.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.as_ref().iter().try_fold(0, |bound, cache| {
            Some(bound.max(cache.continuation_capacity_bound(additional)?))
        })
    }

    /// Bounds page catalog and key-only sentinel growth.
    pub fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.as_ref().iter().try_fold(0u64, |bytes, cache| {
            // The neutral key-only geometry has no value tensor. Native local
            // storage uses a one-channel persistence sentinel; bound it as well
            // as every possible new sealed-block catalog entry.
            let absolute = u64::try_from(cache.offset())
                .ok()?
                .checked_add(additional)?;
            let prefix = absolute.max(cache.continuation_capacity_bound(additional)?);
            let costs = crate::replicated_session::StateSnapshotMemoryCosts::for_tensors(
                M::snapshot_tensor_costs()?,
            );
            let sentinel = costs.tensor_bytes(prefix.checked_mul(costs.floating_width)?, 4)?;
            let catalog = if cache.residency_manager().is_some() {
                additional.checked_add(1)?.checked_mul(8192)?
            } else {
                0
            };
            bytes.checked_add(sentinel)?.checked_add(catalog)
        })
    }

    /// Creates an independently writable model snapshot.
    pub fn isolated_snapshot(&self, stream: &M::Queue) -> Result<Self, PoolingStateError> {
        if !self.supports_isolated_snapshot() {
            return Err(invalid("pooling snapshot has inconsistent paging managers"));
        }
        let manager = self
            .as_ref()
            .iter()
            .find_map(PoolingAttentionLayer::residency_manager)
            .map(|manager| {
                manager
                    .isolated_snapshot(stream)
                    .map_err(|error| invalid(error.to_string()))
            })
            .transpose()?;
        DeviceState::create(self.layout().clone(), |layer, _| {
            self.as_ref()[layer].isolated_snapshot_in(manager.as_ref(), stream)
        })
    }

    /// Copies layers and binds them to a forked paging session.
    pub fn fork_prediction_target_state(
        &self,
        stream: &M::Queue,
    ) -> Result<Self, PoolingStateError> {
        let manager = self
            .as_ref()
            .iter()
            .find_map(PoolingAttentionLayer::residency_manager);
        let manager = manager
            .map(|manager| manager.fork_session(stream))
            .transpose()?;
        DeviceState::create(self.layout().clone(), |layer, _| {
            let mut cache = self.as_ref()[layer].deep_clone_state(stream)?;
            if let Some(manager) = manager.as_ref() {
                cache.rebind_paging_manager(manager.clone());
            }
            Ok(cache)
        })
    }
}

impl<M: PoolingCacheStorage> PoolingAttentionLayer<M> {
    /// Retains exact array views plus sealed history needed after a rollback.
    pub fn checkpoint_clone_state(&self) -> Result<Self, PoolingStateError> {
        let mut checkpoint = self.clone();
        if let KeyValueLayerState::Paged(cache) = self.local() {
            *checkpoint.local_mut() = KeyValueLayerState::Paged(
                cache
                    .checkpoint_clone_state()
                    .map_err(PoolingStateError::from)?,
            );
        }
        Ok(checkpoint)
    }
    /// Creates resident pooling-attention state from one architecture-declared
    pub(crate) fn prediction_snapshot_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        use crate::replicated_session::StateSnapshotMemoryCosts;
        use eredu_core::execution_control::SnapshotEstimate;

        let owner = u64::try_from(std::mem::size_of::<Self>()).ok()?;
        let costs = StateSnapshotMemoryCosts::for_tensors(M::snapshot_tensor_costs()?);
        let tensors = self
            .retained_arrays()
            .into_iter()
            .try_fold(0u64, |sum, value| {
                sum.checked_add(costs.tensor_bytes(M::bytes(value), M::shape(value).len())?)
            })?;
        let auxiliary = self
            .residency_manager()
            .map(|manager| manager.isolated_snapshot_bytes())
            .unwrap_or(Some(0))?;
        let bytes = owner.checked_add(tensors)?.checked_add(auxiliary)?;
        Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    pub(crate) fn settled_prediction_snapshot(
        &self,
        queue: &M::Queue,
    ) -> Result<Self, BackendFailure> {
        M::settle_tensors(&self.retained_arrays())?;
        let copy = self
            .isolated_snapshot(queue)
            .map_err(BackendFailure::from_error)?;
        M::settle_tensors(&copy.retained_arrays())?;
        Ok(copy)
    }

    /// layer policy.
    pub fn resident_from_policy(
        layer: usize,
        policy: &LayerCachePolicy,
    ) -> Result<Self, PoolingStateError> {
        let geometry = pooling_attention_geometry(layer, policy)?;
        Self::with_streams(
            KeyValueLayerState::Device(ResidentKeyValueCache::<M>::new_for_sliding_attention(
                geometry.sliding_window,
            )),
            &geometry.stream_ratios,
        )
    }

    /// Creates paged pooling-attention state from one architecture-declared
    /// layer policy.
    pub fn paged_from_policy(
        layer: usize,
        policy: &LayerCachePolicy,
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, PoolingStateError> {
        let geometry = pooling_attention_geometry(layer, policy)?;
        Self::with_streams(
            PagedKeyValueCache::new_key_only_with_layout(
                manager,
                global_layer,
                Some(geometry.sliding_window),
                prefix_tokens,
                rank,
            )
            .map(KeyValueLayerState::Paged)?,
            &geometry.stream_ratios,
        )
    }

    fn with_streams(
        local: KeyValueLayerState<M>,
        ratios: &[i32],
    ) -> Result<Self, PoolingStateError> {
        match ratios {
            [] => Ok(Self::Local(local)),
            [ratio] => Ok(Self::Compressed {
                local,
                pool: PoolingCache::new(*ratio)?,
            }),
            [ratio, index_ratio] => Ok(Self::Sparse {
                local,
                pool: PoolingCache::new(*ratio)?,
                index_pool: PoolingCache::new(*index_ratio)?,
            }),
            _ => Err(invalid(
                "MLX pooling attention supports at most two declared streams",
            )),
        }
    }

    /// Copies all logical local/pooling views on the supplied stream. The
    /// existing paging namespace remains retained until an enclosing fork binds
    /// its independently owned manager.
    pub fn deep_clone_state(&self, stream: &M::Queue) -> Result<Self, PoolingStateError> {
        self.isolated_snapshot_in(self.residency_manager(), stream)
    }

    /// Bounds local history capacity after processing the admitted suffix.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        let absolute = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        i32::try_from(absolute).ok()?;
        match self.local() {
            KeyValueLayerState::Device(cache) => cache.continuation_capacity_bound(additional),
            KeyValueLayerState::Paged(_) => Some(absolute),
            KeyValueLayerState::Stateless => None,
        }
    }

    /// Complete standalone layer copy, including an independent paging namespace.
    pub fn isolated_snapshot(&self, stream: &M::Queue) -> Result<Self, PoolingStateError> {
        let manager = self
            .residency_manager()
            .map(|manager| {
                manager
                    .isolated_snapshot(stream)
                    .map_err(|error| invalid(error.to_string()))
            })
            .transpose()?;
        self.isolated_snapshot_in(manager.as_ref(), stream)
    }

    fn isolated_snapshot_in(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        stream: &M::Queue,
    ) -> Result<Self, PoolingStateError> {
        let local = self.local().copy_candidate(manager, stream, true)?;
        match self {
            Self::Local(_) => Ok(Self::Local(local)),
            Self::Compressed { pool, .. } => Ok(Self::Compressed {
                local,
                pool: pool.isolated_snapshot(stream)?,
            }),
            Self::Sparse {
                pool, index_pool, ..
            } => Ok(Self::Sparse {
                local,
                pool: pool.isolated_snapshot(stream)?,
                index_pool: index_pool.isolated_snapshot(stream)?,
            }),
        }
    }

    /// Rebinds a paged local cache to the selected catalog.
    pub fn rebind_paging_manager(&mut self, manager: CacheResidencyManager<M>) {
        if let KeyValueLayerState::Paged(cache) = self.local_mut() {
            cache.rebind_paging_manager(manager);
        }
    }

    /// Returns the current source-token frontier.
    pub fn offset(&self) -> i32 {
        self.local().offset()
    }

    fn local(&self) -> &KeyValueLayerState<M> {
        match self {
            Self::Local(local) | Self::Compressed { local, .. } | Self::Sparse { local, .. } => {
                local
            }
        }
    }

    fn local_mut(&mut self) -> &mut KeyValueLayerState<M> {
        match self {
            Self::Local(local) | Self::Compressed { local, .. } | Self::Sparse { local, .. } => {
                local
            }
        }
    }

    fn pool(&self, stream: u32) -> Result<&PoolingCache<M>, ComputeError> {
        match (self, stream) {
            (Self::Compressed { pool, .. } | Self::Sparse { pool, .. }, 0) => Ok(pool),
            (Self::Sparse { index_pool, .. }, 1) => Ok(index_pool),
            _ => Err(ComputeError::backend(format!(
                "pooling attention cache has no stream {stream}"
            ))),
        }
    }

    fn pool_mut(&mut self, stream: u32) -> Result<&mut PoolingCache<M>, ComputeError> {
        match (self, stream) {
            (Self::Compressed { pool, .. } | Self::Sparse { pool, .. }, 0) => Ok(pool),
            (Self::Sparse { index_pool, .. }, 1) => Ok(index_pool),
            _ => Err(ComputeError::backend(format!(
                "pooling attention cache has no stream {stream}"
            ))),
        }
    }

    /// Clears all local and pooled state.
    pub fn clear(&mut self) -> Result<(), PoolingStateError> {
        if let Some(manager) = local_manager(self.local()).cloned() {
            manager.clear()?;
        }
        self.reset_local_after_manager_clear();
        Ok(())
    }

    /// Clears local bookkeeping after a shared paging manager was cleared once
    /// by the enclosing runtime cache.
    pub fn reset_local_after_manager_clear(&mut self) {
        match self.local_mut() {
            KeyValueLayerState::Device(cache) => cache.clear(),
            KeyValueLayerState::Paged(cache) => cache.reset_local_after_manager_clear(),
            KeyValueLayerState::Stateless => {}
        }
        match self {
            Self::Local(_) => {}
            Self::Compressed { pool, .. } => pool.clear(),
            Self::Sparse {
                pool, index_pool, ..
            } => {
                pool.clear();
                index_pool.clear();
            }
        }
    }

    /// Returns all arrays retained by local and pooled state.
    pub fn retained_arrays(&self) -> Vec<&M::Tensor> {
        let mut values = self
            .local()
            .retained_values()
            .map(M::storage_ref)
            .collect::<Vec<_>>();
        match self {
            Self::Local(_) => {}
            Self::Compressed { pool, .. } => values.extend(pool.arrays()),
            Self::Sparse {
                pool, index_pool, ..
            } => {
                values.extend(pool.arrays());
                values.extend(index_pool.arrays());
            }
        }
        values
    }

    /// Borrows the shared paging manager when local state is paged.
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager<M>> {
        local_manager(self.local())
    }

    /// Finalizes the local paged tail before durable publication.
    pub fn finalize(&mut self) -> Result<(), PoolingStateError> {
        if let KeyValueLayerState::Paged(cache) = self.local_mut() {
            cache.finalize()?;
        }
        Ok(())
    }

    /// Borrows every architecture-declared pooling component for persistence.
    pub fn prompt_cache_state_arrays(
        &self,
        global_layer: usize,
    ) -> Vec<PromptCacheStateArray<M::Tensor>> {
        let mut arrays = Vec::new();
        match self {
            Self::Local(_) => {}
            Self::Compressed { pool, .. } => {
                append_pooling_state_arrays(&mut arrays, global_layer, 0, pool);
            }
            Self::Sparse {
                pool, index_pool, ..
            } => {
                append_pooling_state_arrays(&mut arrays, global_layer, 0, pool);
                append_pooling_state_arrays(&mut arrays, global_layer, 1, index_pool);
            }
        }
        arrays
    }

    /// Restores every persisted pooling component at one exact source frontier.
    pub fn restore_prompt_cache_state(
        &mut self,
        global_layer: usize,
        tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), M::Tensor>,
        processed_tokens: i32,
    ) -> Result<(), PoolingStateError> {
        match self {
            Self::Local(_) => Ok(()),
            Self::Compressed { pool, .. } => {
                restore_pooling_state(global_layer, 0, pool, tensors, processed_tokens, false)
            }
            Self::Sparse {
                pool, index_pool, ..
            } => {
                restore_pooling_state(global_layer, 0, pool, tensors, processed_tokens, true)?;
                restore_pooling_state(global_layer, 1, index_pool, tensors, processed_tokens, true)
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct PoolingAttentionGeometry {
    sliding_window: i32,
    stream_ratios: Vec<i32>,
}

fn pooling_attention_geometry(
    layer: usize,
    policy: &LayerCachePolicy,
) -> Result<PoolingAttentionGeometry, PoolingStateError> {
    let (attention, tensors) = match policy {
        LayerCachePolicy::KeyOnly { attention, .. } => (attention, &[][..]),
        LayerCachePolicy::KeyOnlyWithState {
            attention, tensors, ..
        } => (attention, tensors.as_slice()),
        _ => {
            return Err(invalid(format!(
                "pooling-attention state requires key-only policy at layer {layer}: {policy:?}"
            )))
        }
    };
    let sliding_window = attention
        .sliding_window_i32()
        .map_err(|error| invalid(error.to_string()))?
        .ok_or_else(|| {
            invalid(format!(
                "pooling-attention state requires a sliding window at layer {layer}"
            ))
        })?;
    let mut streams = BTreeMap::<u32, BTreeMap<PoolingStateComponent, _>>::new();
    for tensor in tensors {
        let StateTensorRole::Pooling { stream, component } = tensor.role else {
            return Err(invalid(format!(
                "pooling-attention state found non-pooling component at layer {layer}: {:?}",
                tensor.role
            )));
        };
        streams.entry(stream).or_default().insert(component, tensor);
    }
    let mut stream_ratios = Vec::with_capacity(streams.len());
    let mut stream_overlaps = Vec::with_capacity(streams.len());
    for (expected_stream, (stream, components)) in streams.into_iter().enumerate() {
        if stream as usize != expected_stream {
            return Err(invalid(format!(
                "pooling-attention streams must be contiguous at layer {layer}, expected {expected_stream}, got {stream}"
            )));
        }
        let (ratio, overlapping) = pooling_stream_geometry(layer, stream, &components)?;
        stream_ratios.push(ratio);
        stream_overlaps.push(overlapping);
    }
    if !matches!(stream_overlaps.as_slice(), [] | [false] | [true, true]) {
        return Err(invalid(format!(
            "pooling-attention stream overlap layout is unsupported at layer {layer}"
        )));
    }
    Ok(PoolingAttentionGeometry {
        sliding_window,
        stream_ratios,
    })
}

fn pooling_stream_geometry(
    layer: usize,
    stream: u32,
    components: &BTreeMap<PoolingStateComponent, &eredu_core::cache::StateTensorPolicy>,
) -> Result<(i32, bool), PoolingStateError> {
    let get = |component| {
        components.get(&component).copied().ok_or_else(|| {
            invalid(format!(
                "MLX pooling stream {stream} at layer {layer} is missing {component:?}"
            ))
        })
    };
    let pooled = get(PoolingStateComponent::Pooled)?;
    let ratio = match (pooled.shape.as_slice(), pooled.presence) {
        (
            [StateTensorDimension::Batch, StateTensorDimension::PrefixTokensDiv(shape_ratio), StateTensorDimension::Fixed(_)],
            StateTensorPresence::PrefixAtLeast(presence_ratio),
        ) if shape_ratio == &presence_ratio => *shape_ratio,
        _ => {
            return Err(invalid(format!(
                "MLX pooling stream {stream} at layer {layer} has invalid pooled geometry"
            )))
        }
    };
    for component in [
        PoolingStateComponent::PendingValues,
        PoolingStateComponent::PendingGates,
    ] {
        let pending = get(component)?;
        match (pending.shape.as_slice(), pending.presence) {
            (
                [StateTensorDimension::Batch, StateTensorDimension::PrefixTokensRem(shape_ratio), StateTensorDimension::Fixed(_)],
                StateTensorPresence::PrefixRemainderNonZero(presence_ratio),
            ) if shape_ratio == &ratio && presence_ratio == ratio => {}
            _ => {
                return Err(invalid(format!(
                "MLX pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
            )))
            }
        }
    }
    let overlap_values = components.get(&PoolingStateComponent::OverlapValues);
    let overlap_gates = components.get(&PoolingStateComponent::OverlapGates);
    if overlap_values.is_some() != overlap_gates.is_some() {
        return Err(invalid(format!(
            "MLX pooling stream {stream} at layer {layer} has incomplete overlap geometry"
        )));
    }
    for (component, overlap) in [
        (PoolingStateComponent::OverlapValues, overlap_values),
        (PoolingStateComponent::OverlapGates, overlap_gates),
    ] {
        let Some(overlap) = overlap else { continue };
        match (overlap.shape.as_slice(), overlap.presence) {
            (
                [StateTensorDimension::Batch, StateTensorDimension::Fixed(shape_ratio), StateTensorDimension::Fixed(_)],
                StateTensorPresence::PrefixAtLeast(presence_ratio),
            ) if shape_ratio == &ratio && presence_ratio == ratio => {}
            _ => {
                return Err(invalid(format!(
                "MLX pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
            )))
            }
        }
    }
    if components.len() != 3 + usize::from(overlap_values.is_some()) * 2 {
        return Err(invalid(format!(
            "MLX pooling stream {stream} at layer {layer} has undeclared components"
        )));
    }
    i32::try_from(ratio.get())
        .map(|ratio| (ratio, overlap_values.is_some()))
        .map_err(|_| invalid("pooling ratio exceeds runtime range"))
}

const POOLING_COMPONENTS: [PoolingStateComponent; 5] = [
    PoolingStateComponent::PendingValues,
    PoolingStateComponent::PendingGates,
    PoolingStateComponent::Pooled,
    PoolingStateComponent::OverlapValues,
    PoolingStateComponent::OverlapGates,
];

fn pooling_role(stream: u32, component: PoolingStateComponent) -> StateTensorRole {
    StateTensorRole::Pooling { stream, component }
}

fn append_pooling_state_arrays<'a, M: PoolingCacheStorage>(
    arrays: &mut Vec<PromptCacheStateArray<M::Tensor>>,
    global_layer: usize,
    stream: u32,
    pool: &'a PoolingCache<M>,
) {
    for (component, array) in POOLING_COMPONENTS.into_iter().zip(pool.state_arrays()) {
        if let Some(array) = array {
            arrays.push(PromptCacheStateArray {
                owner: StateTensorOwner::Layer(global_layer),
                role: pooling_role(stream, component),
                array: array.clone(),
            });
        }
    }
}

fn restore_pooling_state<M: PoolingCacheStorage>(
    global_layer: usize,
    stream: u32,
    pool: &mut PoolingCache<M>,
    tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), M::Tensor>,
    processed_tokens: i32,
    overlapping: bool,
) -> Result<(), PoolingStateError> {
    let mut take = |component| {
        tensors.remove(&(
            StateTensorOwner::Layer(global_layer),
            pooling_role(stream, component),
        ))
    };
    let pending_values = take(PoolingStateComponent::PendingValues);
    let pending_gates = take(PoolingStateComponent::PendingGates);
    let pooled = take(PoolingStateComponent::Pooled);
    let overlap_values = take(PoolingStateComponent::OverlapValues);
    let overlap_gates = take(PoolingStateComponent::OverlapGates);
    let ratio = pool.ratio();
    let expect_pending = processed_tokens % ratio != 0;
    let expect_complete = processed_tokens >= ratio;
    if pending_values.is_some() != expect_pending
        || pending_gates.is_some() != expect_pending
        || pooled.is_some() != expect_complete
        || overlap_values.is_some() != (overlapping && expect_complete)
        || overlap_gates.is_some() != (overlapping && expect_complete)
    {
        return Err(invalid(format!(
            "pooling state for layer {global_layer}, stream {stream} is incomplete"
        )));
    }
    pool.restore_state(
        PoolingCacheState {
            pending_values,
            pending_gates,
            pooled,
            overlap_values,
            overlap_gates,
        },
        processed_tokens,
    )
}

impl<B, M> RuntimeLayerState<B> for PoolingAttentionLayer<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: PoolingCacheStorage,
{
    type RetainedValues<'a> =
        std::iter::Map<std::vec::IntoIter<&'a M::Tensor>, fn(&M::Tensor) -> &M::Value>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_arrays().into_iter().map(M::borrow_value)
    }
}

impl<M: PoolingCacheStorage> PoolingAttentionCache<M::Value> for PoolingAttentionLayer<M> {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset()
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.pool(stream).ok().map(PoolingCache::ratio)
    }

    fn append_local(
        &mut self,
        keys: M::Value,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        let &[batch, tokens, width] = keys.shape() else {
            return Err(ComputeError::backend(
                "local keys must have shape [batch, tokens, width]",
            ));
        };
        let keys = M::into_storage(keys.reshape(&[batch, 1, tokens, width], stream)?);
        let value_width = if matches!(self.local(), KeyValueLayerState::Paged(_)) {
            0
        } else {
            1
        };
        let values = M::zeroed_cache_tensor(&[batch, 1, tokens, value_width], &keys, stream)
            .map_err(ComputeError::backend_source)?;
        let (keys, _) = match self.local_mut() {
            KeyValueLayerState::Device(cache) => cache
                .update_and_fetch(keys, values, stream)
                .map_err(ComputeError::backend_source)?,
            KeyValueLayerState::Paged(cache) => cache
                .update_and_fetch(keys, values, stream)
                .map_err(ComputeError::backend_source)?,
            KeyValueLayerState::Stateless => {
                return Err(ComputeError::backend("local pooling cache has no state"))
            }
        };
        let completed = M::shape(&keys)[2];
        M::into_value(keys).reshape(&[batch, completed, width], stream)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        let window = self
            .local()
            .max_size()
            .ok_or_else(|| ComputeError::backend("local cache has no sliding-window policy"))?;
        let end = offset
            .checked_add(query_tokens)
            .ok_or_else(|| ComputeError::backend("local query extent overflowed"))?;
        let key_offset = offset.saturating_sub(window - 1).max(0);
        M::local_visibility(offset..end, key_offset..end, window, stream)
            .map_err(ComputeError::backend_source)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream_id: u32,
        values: M::Value,
        gates: M::Value,
        absolute_offset: i32,
        stream: &M::Queue,
    ) -> Result<PoolingWindows<M::Value>, ComputeError> {
        self.pool_mut(stream_id)?
            .accumulate_windows(
                M::into_storage(values),
                M::into_storage(gates),
                absolute_offset,
                stream,
            )
            .map(|windows| PoolingWindows {
                values: M::into_value(windows.values),
                gates: M::into_value(windows.gates),
                base_position: windows.base_position,
            })
            .map_err(ComputeError::backend_source)
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: M::Value,
        gates: M::Value,
    ) -> Result<PoolingOverlap<M::Value>, ComputeError> {
        let (values, gates) = self
            .pool_mut(stream)?
            .replace_overlap(M::into_storage(values), M::into_storage(gates));
        Ok(PoolingOverlap {
            values: values.map(M::into_value),
            gates: gates.map(M::into_value),
        })
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: M::Value,
        context: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        self.pool_mut(stream)?
            .update_and_fetch(M::into_storage(values), context)
            .map(M::into_value)
            .map_err(ComputeError::backend_source)
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &M::Queue,
    ) -> Result<Option<M::Value>, ComputeError> {
        self.pool(stream)?
            .make_mask(query_tokens, offset, context)
            .map(|mask| mask.map(M::into_value))
            .map_err(ComputeError::backend_source)
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint, ComputeError> {
        self.checkpoint_clone_state()
            .map_err(ComputeError::backend_source)
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        stream: &M::Queue,
    ) -> Result<(), ComputeError> {
        match (self, checkpoint) {
            (Self::Local(local), Self::Local(previous)) => {
                restore_local_checkpoint(local, previous, stream)
                    .map_err(ComputeError::backend_source)
            }
            (
                Self::Compressed { local, pool },
                Self::Compressed {
                    local: previous_local,
                    pool: previous_pool,
                },
            ) => {
                restore_local_checkpoint(local, previous_local, stream)
                    .map_err(ComputeError::backend_source)?;
                pool.clone_from(previous_pool);
                Ok(())
            }
            (
                Self::Sparse {
                    local,
                    pool,
                    index_pool,
                },
                Self::Sparse {
                    local: previous_local,
                    pool: previous_pool,
                    index_pool: previous_index_pool,
                },
            ) => {
                restore_local_checkpoint(local, previous_local, stream)
                    .map_err(ComputeError::backend_source)?;
                pool.clone_from(previous_pool);
                index_pool.clone_from(previous_index_pool);
                Ok(())
            }
            _ => Err(ComputeError::backend(
                "pooling cache representation changed",
            )),
        }
    }

    fn finalize(&mut self) -> Result<(), ComputeError> {
        PoolingAttentionLayer::finalize(self).map_err(ComputeError::backend_source)
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        self.clear().map_err(ComputeError::backend_source)
    }
}
