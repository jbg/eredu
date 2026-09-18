//! Architecture-declared pooling-attention state realization.

use super::*;

mod prepared_copy;
pub(crate) use prepared_copy::PagedPoolingCopy;
mod resident_copy;
pub(crate) use prepared_copy::PreparedPoolingAttentionCopy;
pub(crate) use resident_copy::ResidentPoolingPreparationError;
pub(crate) use resident_copy::{
    DenseResidentPoolingPublishError, PreparedDenseResidentPoolingState,
    PreparedResidentPoolingCopy, ProjectedDenseResidentPoolingCopy,
    PublishedDenseResidentPoolingState, ResidentPoolingCopyError, SavedResidentPoolingCopy,
};
mod workspace;

/// General MLX realization of bounded local keys plus zero, one, or two
/// append-only pooling streams.
#[derive(Debug, Clone)]
pub enum MlxPoolingAttentionCache {
    /// Bounded local keys only.
    Local(LiveKeyValueCache),
    /// Local keys plus one compressed stream.
    Compressed {
        /// Bounded local keys.
        local: LiveKeyValueCache,
        /// Compressed attention stream.
        pool: PoolingCache,
    },
    /// Local keys, compressed attention, and sparse-index streams.
    Sparse {
        /// Bounded local keys.
        local: LiveKeyValueCache,
        /// Compressed attention stream.
        pool: PoolingCache,
        /// Sparse-index stream.
        index_pool: PoolingCache,
    },
}

/// Complete MLX pooling-attention state realized directly from a neutral
/// architecture layout.
pub type MlxPoolingAttentionState = DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>;

use eredu_runtime::working_memory::InferenceStateRetention;

/// Architecture-independent MLX materializer for pooling-attention layouts.
pub struct MlxPoolingAttentionStateFactory;

impl MlxPoolingAttentionStateFactory {
    /// Materializes device-resident state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<MlxPoolingAttentionState, Exception> {
        DeviceState::create(layout, MlxPoolingAttentionCache::resident_from_policy)
    }

    /// Materializes paged local keys plus device pooling streams for every
    /// declared layer.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager,
        global_layer_start: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<MlxPoolingAttentionState, Exception> {
        DeviceState::create(layout, move |layer, policy| {
            let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                Exception::custom("pooling-attention global layer index overflowed")
            })?;
            MlxPoolingAttentionCache::paged_from_policy(
                layer,
                policy,
                manager.clone(),
                global_layer,
                prefix_tokens,
                rank,
            )
        })
    }

    pub(crate) fn supports_isolated_snapshot(state: &MlxPoolingAttentionState) -> bool {
        let mut manager = None;
        state.as_ref().iter().all(|cache| {
            cache.residency_manager().is_none_or(|paging| {
                *manager.get_or_insert(paging.session_id()) == paging.session_id()
            })
        })
    }

    pub(crate) fn isolated_snapshot_auxiliary_bytes(
        state: &MlxPoolingAttentionState,
    ) -> Option<u64> {
        state
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(CacheResidencyManager::isolated_snapshot_bytes)
            .unwrap_or(Some(0))
    }

    pub(crate) fn original_isolated_snapshot_auxiliary_bytes(
        state: &MlxPoolingAttentionState,
    ) -> Option<u64> {
        state
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(CacheResidencyManager::original_isolated_snapshot_bytes)
            .unwrap_or(Some(0))
    }

    pub(crate) fn continuation_capacity_bound(
        state: &MlxPoolingAttentionState,
        additional: u64,
    ) -> Option<u64> {
        state.as_ref().iter().try_fold(0, |bound, cache| {
            Some(bound.max(cache.continuation_capacity_bound(additional)?))
        })
    }

    pub(crate) fn isolated_snapshot_auxiliary_growth(
        state: &MlxPoolingAttentionState,
        additional: u64,
    ) -> Option<u64> {
        state.as_ref().iter().try_fold(0u64, |bytes, cache| {
            // The neutral key-only geometry has no value tensor. Native local
            // storage uses a one-channel persistence sentinel; bound it as well
            // as every possible new sealed-block catalog entry.
            let absolute = u64::try_from(cache.offset())
                .ok()?
                .checked_add(additional)?;
            let prefix = absolute.max(cache.continuation_capacity_bound(additional)?);
            let sentinel = prefix.checked_mul(16)?.checked_add(4096 + 4 * 16)?;
            let catalog = if cache.residency_manager().is_some() {
                additional.checked_add(1)?.checked_mul(8192)?
            } else {
                0
            };
            bytes.checked_add(sentinel)?.checked_add(catalog)
        })
    }

    pub(crate) fn isolated_snapshot(
        state: &MlxPoolingAttentionState,
        stream: &Stream,
    ) -> Result<MlxPoolingAttentionState, Exception> {
        if !Self::supports_isolated_snapshot(state) {
            return Err(Exception::custom(
                "pooling snapshot has inconsistent paging managers",
            ));
        }
        let manager = state
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(|manager| {
                manager
                    .isolated_snapshot(stream)
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()?;
        let layout = state
            .shared_layout()
            .cloned()
            .ok_or_else(|| Exception::custom("pooling snapshot requires an owned state layout"))?;
        let mut copy = DeviceState::create_with_shared_layout(layout, |layer, _| {
            state.as_ref()[layer].isolated_snapshot_in(manager.as_ref(), stream)
        })?;
        copy.inherit_inference_retention(state);
        Ok(copy)
    }

    pub(crate) fn fork_prediction_target_state(
        state: &MlxPoolingAttentionState,
        stream: &Stream,
    ) -> Result<MlxPoolingAttentionState, Exception> {
        let manager = state
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager);
        let manager = manager
            .map(|manager| manager.fork_session(stream))
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let layout = state
            .shared_layout()
            .cloned()
            .ok_or_else(|| Exception::custom("pooling fork requires an owned state layout"))?;
        let mut fork = DeviceState::create_with_shared_layout(layout, |layer, _| {
            let mut cache = state.as_ref()[layer].deep_clone_state(stream)?;
            if let Some(manager) = manager.as_ref() {
                cache.rebind_paging_manager(manager.clone());
            }
            Ok::<_, Exception>(cache)
        })?;
        fork.inherit_inference_retention(state);
        Ok(fork)
    }
}

impl MlxPoolingAttentionCache {
    pub(crate) fn retained_owner_slot_counts(&self) -> NativeStateSlotCounts {
        let arrays = match self {
            Self::Local(_) => 2,
            Self::Compressed { .. } => 7,
            Self::Sparse { .. } => 12,
        };
        NativeStateSlotCounts::arrays(arrays, usize::from(self.residency_manager().is_some()))
    }

    /// Retains exact array views plus sealed history needed after a rollback.
    pub(crate) fn checkpoint_clone_state(&self) -> Result<Self, Exception> {
        let mut checkpoint = self.clone();
        if let LiveKeyValueCache::Paged(cache) = self.local() {
            *checkpoint.local_mut() = LiveKeyValueCache::Paged(cache.checkpoint_clone_state()?);
        }
        Ok(checkpoint)
    }
    /// Creates resident pooling-attention state from one architecture-declared
    /// layer policy.
    pub fn resident_from_policy(
        layer: usize,
        policy: &LayerCachePolicy,
    ) -> Result<Self, Exception> {
        let geometry = pooling_attention_geometry(layer, policy)?;
        Self::with_streams(
            LiveKeyValueCache::resident(ConcatKeyValueCache::new_for_sliding_attention(
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
        manager: CacheResidencyManager,
        global_layer: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let geometry = pooling_attention_geometry(layer, policy)?;
        Self::with_streams(
            LiveKeyValueCache::paged_key_only(
                manager,
                global_layer,
                Some(geometry.sliding_window),
                prefix_tokens,
                rank,
            )?,
            &geometry.stream_ratios,
        )
    }

    fn with_streams(local: LiveKeyValueCache, ratios: &[i32]) -> Result<Self, Exception> {
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
            _ => Err(Exception::custom(
                "MLX pooling attention supports at most two declared streams",
            )),
        }
    }

    /// Copies all logical local/pooling views on the supplied stream. The
    /// existing paging namespace remains retained until an enclosing fork binds
    /// its independently owned manager.
    pub fn deep_clone_state(&self, stream: &Stream) -> Result<Self, Exception> {
        self.isolated_snapshot_in(self.residency_manager(), stream)
    }

    pub(crate) fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        let absolute = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        i32::try_from(absolute).ok()?;
        match self.local() {
            LiveKeyValueCache::Resident(cache) => cache.continuation_capacity_bound(additional),
            LiveKeyValueCache::Paged(_) => Some(absolute),
        }
    }

    /// Complete standalone layer copy, including an independent paging namespace.
    pub(crate) fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        let manager = self
            .residency_manager()
            .map(|manager| {
                manager
                    .isolated_snapshot(stream)
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()?;
        self.isolated_snapshot_in(manager.as_ref(), stream)
    }

    /// Borrows the closed resident local/pooling copy program. Paged local
    /// storage still requires its independent manager-copy mechanism.
    pub(crate) fn prepare_isolated_copy(
        &self,
    ) -> Result<PreparedPoolingAttentionCopy<'_>, Exception> {
        PreparedPoolingAttentionCopy::new(self)
    }

    pub(crate) fn prepare_isolated_copy_fixed(
        &self,
    ) -> Result<PreparedPoolingAttentionCopy<'_>, prepared_copy::PagedPoolingCopy> {
        PreparedPoolingAttentionCopy::new_fixed(self)
    }

    fn isolated_snapshot_in(
        &self,
        manager: Option<&CacheResidencyManager>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let local = match self.local() {
            LiveKeyValueCache::Resident(_) => {
                return self.prepare_isolated_copy()?.copy(stream);
            }
            LiveKeyValueCache::Paged(cache) => {
                let manager = manager.ok_or_else(|| {
                    Exception::custom("pooling snapshot lacks copied paging manager")
                })?;
                let mut copy = cache.deep_clone_state(stream)?;
                copy.rebind_paging_manager(manager.clone());
                LiveKeyValueCache::Paged(copy)
            }
        };
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

    pub(crate) fn rebind_paging_manager(&mut self, manager: CacheResidencyManager) {
        self.local_mut().rebind_paging_manager(manager);
    }

    /// Returns the current source-token frontier.
    pub fn offset(&self) -> i32 {
        self.local().offset()
    }

    fn local(&self) -> &LiveKeyValueCache {
        match self {
            Self::Local(local) | Self::Compressed { local, .. } | Self::Sparse { local, .. } => {
                local
            }
        }
    }

    fn local_mut(&mut self) -> &mut LiveKeyValueCache {
        match self {
            Self::Local(local) | Self::Compressed { local, .. } | Self::Sparse { local, .. } => {
                local
            }
        }
    }

    fn pool(&self, stream: u32) -> Result<&PoolingCache, ComputeError> {
        match (self, stream) {
            (Self::Compressed { pool, .. } | Self::Sparse { pool, .. }, 0) => Ok(pool),
            (Self::Sparse { index_pool, .. }, 1) => Ok(index_pool),
            _ => Err(PoolingCache::missing_stream(stream)),
        }
    }

    fn pool_mut(&mut self, stream: u32) -> Result<&mut PoolingCache, ComputeError> {
        match (self, stream) {
            (Self::Compressed { pool, .. } | Self::Sparse { pool, .. }, 0) => Ok(pool),
            (Self::Sparse { index_pool, .. }, 1) => Ok(index_pool),
            _ => Err(ComputeError::backend(format!(
                "pooling attention cache has no stream {stream}"
            ))),
        }
    }

    /// Clears all local and pooled state.
    pub fn clear(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self.local().manager().cloned() {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        self.local_mut().reset_local_after_manager_clear();
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
        Ok(())
    }

    /// Clears local bookkeeping after a shared paging manager was cleared once
    /// by the enclosing runtime cache.
    pub fn reset_local_after_manager_clear(&mut self) {
        self.local_mut().reset_local_after_manager_clear();
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
    pub fn retained_arrays(&self) -> Vec<&Array> {
        let mut values = self.local().retained_arrays();
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
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.local().manager()
    }

    /// Finalizes the local paged tail before durable publication.
    pub fn finalize(&mut self) -> Result<(), Exception> {
        self.local_mut().finalize()
    }

    /// Borrows every architecture-declared pooling component for persistence.
    pub fn prompt_cache_state_arrays(&self, global_layer: usize) -> Vec<PromptCacheStateArray<'_>> {
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
        tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), Array>,
        processed_tokens: i32,
    ) -> Result<(), Exception> {
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

use eredu_runtime::state::PoolingAttentionGeometry;

fn pooling_attention_geometry(
    layer: usize,
    policy: &LayerCachePolicy,
) -> Result<PoolingAttentionGeometry, Exception> {
    eredu_runtime::state::pooling_attention_geometry(layer, policy)
        .map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
#[path = "tests/pooling_layout.rs"]
mod pooling_layout_tests;

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

fn append_pooling_state_arrays<'a>(
    arrays: &mut Vec<PromptCacheStateArray<'a>>,
    global_layer: usize,
    stream: u32,
    pool: &'a PoolingCache,
) {
    for (component, array) in POOLING_COMPONENTS.into_iter().zip(pool.state_arrays()) {
        if let Some(array) = array {
            arrays.push(PromptCacheStateArray {
                owner: StateTensorOwner::Layer(global_layer),
                role: pooling_role(stream, component),
                array,
            });
        }
    }
}

fn restore_pooling_state(
    global_layer: usize,
    stream: u32,
    pool: &mut PoolingCache,
    tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), Array>,
    processed_tokens: i32,
    overlapping: bool,
) -> Result<(), Exception> {
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
        return Err(Exception::custom(format!(
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

impl RuntimeLayerState<MlxNeuralBackend> for MlxPoolingAttentionCache {
    type RetainedValues<'a> = RetainedArrayVecIter<'a>;

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) {
        match self.local() {
            LiveKeyValueCache::Resident(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(cache, visitor)
            }
            LiveKeyValueCache::Paged(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(cache, visitor)
            }
        }
        let mut visit_pool = |pool: &PoolingCache| {
            for array in pool.arrays() {
                visitor(retained_tensor(array));
            }
        };
        match self {
            Self::Local(_) => {}
            Self::Compressed { pool, .. } => visit_pool(pool),
            Self::Sparse {
                pool, index_pool, ..
            } => {
                visit_pool(pool);
                visit_pool(index_pool);
            }
        }
    }

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_arrays().into_iter().map(retained_tensor)
    }
}

impl PoolingAttentionCache<MlxTensor> for MlxPoolingAttentionCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset()
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.pool(stream).ok().map(PoolingCache::ratio)
    }

    fn append_local(
        &mut self,
        keys: MlxTensor,
        stream: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let keys = keys.into_array();
        let batch = keys.dim(0);
        let tokens = keys.dim(1);
        let keys = keys
            .try_index_device((.., NewAxis, .., ..), stream)
            .map_err(ComputeError::backend_retained_source)?;
        let dtype = keys.dtype();
        // Paged key-only storage accepts a zero-width logical value and
        // materializes its own one-channel persistence sentinel. Resident KV
        // concatenation needs the sentinel supplied explicitly.
        let value_width = i32::from(!self.local().is_paged());
        let (keys, _) = self
            .local_mut()
            .update_and_fetch(
                keys,
                zeros_dtype(&[batch, 1, tokens, value_width], dtype, stream)
                    .map_err(ComputeError::backend_retained_source)?,
                stream,
            )
            .map_err(ComputeError::backend_retained_source)?;
        keys.try_index_device((.., 0, .., ..), stream)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend_retained_source)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let window = self
            .local()
            .max_size()
            .ok_or_else(|| ComputeError::backend("local cache has no sliding-window policy"))?;
        let key_tokens = (offset + query_tokens).min(window - 1 + query_tokens);
        let key_offset = offset + query_tokens - key_tokens;
        let queries = Array::arange::<i32, i32>(Some(offset), offset + query_tokens, None, stream)
            .and_then(|values| values.try_index_device((.., NewAxis), stream))
            .map_err(ComputeError::backend_retained_source)?;
        let keys =
            Array::arange::<i32, i32>(Some(key_offset), key_offset + key_tokens, None, stream)
                .and_then(|values| values.try_index_device((NewAxis, ..), stream))
                .map_err(ComputeError::backend_retained_source)?;
        let causal = queries
            .ge(&keys, stream)
            .map_err(ComputeError::backend_retained_source)?;
        let recent = keys
            .gt(
                queries
                    .subtract(
                        Array::try_from_int(window).map_err(ComputeError::backend_retained_source)?,
                        stream,
                    )
                    .map_err(ComputeError::backend_retained_source)?,
                stream,
            )
            .map_err(ComputeError::backend_retained_source)?;
        causal
            .logical_and(&recent, stream)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend_retained_source)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream_id: u32,
        values: MlxTensor,
        gates: MlxTensor,
        absolute_offset: i32,
        stream: &Stream,
    ) -> Result<PoolingWindows<MlxTensor>, ComputeError> {
        self.pool_mut(stream_id)?
            .accumulate_windows(
                values.into_array(),
                gates.into_array(),
                absolute_offset,
                stream,
            )
            .map(|windows| PoolingWindows {
                values: MlxTensor::from_array(windows.values),
                gates: MlxTensor::from_array(windows.gates),
                base_position: windows.base_position,
            })
            .map_err(ComputeError::backend_retained_source)
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: MlxTensor,
        gates: MlxTensor,
    ) -> Result<PoolingOverlap<MlxTensor>, ComputeError> {
        let (values, gates) = self
            .pool_mut(stream)?
            .replace_overlap(values.into_array(), gates.into_array());
        Ok(PoolingOverlap {
            values: values.map(MlxTensor::from_array),
            gates: gates.map(MlxTensor::from_array),
        })
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        self.pool_mut(stream)?
            .update_and_fetch(values.into_array(), context)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend_retained_source)
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<Option<MlxTensor>, ComputeError> {
        self.pool(stream)?
            .make_mask(query_tokens, offset, context)
            .map(|mask| mask.map(MlxTensor::from_array))
            .map_err(PoolingCache::mask_error)
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint, ComputeError> {
        self.checkpoint_clone_state()
            .map_err(ComputeError::backend_retained_source)
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        stream: &Stream,
    ) -> Result<(), ComputeError> {
        match (self, checkpoint) {
            (Self::Local(local), Self::Local(previous)) => local
                .restore_checkpoint(previous, stream)
                .map_err(ComputeError::backend_retained_source),
            (
                Self::Compressed { local, pool },
                Self::Compressed {
                    local: previous_local,
                    pool: previous_pool,
                },
            ) => {
                local
                    .restore_checkpoint(previous_local, stream)
                    .map_err(ComputeError::backend_retained_source)?;
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
                local
                    .restore_checkpoint(previous_local, stream)
                    .map_err(ComputeError::backend_retained_source)?;
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
        self.local_mut()
            .finalize()
            .map_err(ComputeError::backend_retained_source)
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        self.clear().map_err(ComputeError::backend_retained_source)
    }
}

#[cfg(test)]
#[path = "tests/pooling_snapshots.rs"]
mod snapshot_tests;
