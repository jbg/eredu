//! Architecture-declared pooling-attention state realization.

use super::*;

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
        DeviceState::create(state.layout().clone(), |layer, _| {
            let mut cache = state.as_ref()[layer].deep_clone_state()?;
            if let Some(manager) = manager.as_ref() {
                cache.rebind_paging_manager(manager.clone());
            }
            Ok(cache)
        })
    }
}

impl MlxPoolingAttentionCache {
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

    /// Copies every state array into an independent MLX graph value.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        match self {
            Self::Local(local) => local.deep_clone_state().map(Self::Local),
            Self::Compressed { local, pool } => Ok(Self::Compressed {
                local: local.deep_clone_state()?,
                pool: pool.deep_clone_state()?,
            }),
            Self::Sparse {
                local,
                pool,
                index_pool,
            } => Ok(Self::Sparse {
                local: local.deep_clone_state()?,
                pool: pool.deep_clone_state()?,
                index_pool: index_pool.deep_clone_state()?,
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
            _ => Err(ComputeError::backend(format!(
                "pooling attention cache has no stream {stream}"
            ))),
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

#[derive(Debug, Eq, PartialEq)]
struct PoolingAttentionGeometry {
    sliding_window: i32,
    stream_ratios: Vec<i32>,
}

fn pooling_attention_geometry(
    layer: usize,
    policy: &LayerCachePolicy,
) -> Result<PoolingAttentionGeometry, Exception> {
    let (attention, tensors) = match policy {
        LayerCachePolicy::KeyOnly { attention, .. } => (attention, &[][..]),
        LayerCachePolicy::KeyOnlyWithFixedState {
            attention, tensors, ..
        } => (attention, tensors.as_slice()),
        _ => {
            return Err(Exception::custom(format!(
                "MLX pooling-attention state requires key-only policy at layer {layer}: {policy:?}"
            )))
        }
    };
    let sliding_window = attention
        .sliding_window_i32()
        .map_err(|error| Exception::custom(error.to_string()))?
        .ok_or_else(|| {
            Exception::custom(format!(
                "MLX pooling-attention state requires a sliding window at layer {layer}"
            ))
        })?;
    let mut streams = BTreeMap::<u32, BTreeMap<PoolingStateComponent, _>>::new();
    for tensor in tensors {
        let StateTensorRole::Pooling { stream, component } = tensor.role else {
            return Err(Exception::custom(format!(
                "MLX pooling-attention state found non-pooling component at layer {layer}: {:?}",
                tensor.role
            )));
        };
        streams.entry(stream).or_default().insert(component, tensor);
    }
    let mut stream_ratios = Vec::with_capacity(streams.len());
    let mut stream_overlaps = Vec::with_capacity(streams.len());
    for (expected_stream, (stream, components)) in streams.into_iter().enumerate() {
        if stream as usize != expected_stream {
            return Err(Exception::custom(format!(
                "MLX pooling-attention streams must be contiguous at layer {layer}, expected {expected_stream}, got {stream}"
            )));
        }
        let (ratio, overlapping) = pooling_stream_geometry(layer, stream, &components)?;
        stream_ratios.push(ratio);
        stream_overlaps.push(overlapping);
    }
    if !matches!(stream_overlaps.as_slice(), [] | [false] | [true, true]) {
        return Err(Exception::custom(format!(
            "MLX pooling-attention stream overlap layout is unsupported at layer {layer}"
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
) -> Result<(i32, bool), Exception> {
    let get = |component| {
        components.get(&component).copied().ok_or_else(|| {
            Exception::custom(format!(
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
            return Err(Exception::custom(format!(
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
                return Err(Exception::custom(format!(
                "MLX pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
            )))
            }
        }
    }
    let overlap_values = components.get(&PoolingStateComponent::OverlapValues);
    let overlap_gates = components.get(&PoolingStateComponent::OverlapGates);
    if overlap_values.is_some() != overlap_gates.is_some() {
        return Err(Exception::custom(format!(
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
                return Err(Exception::custom(format!(
                "MLX pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
            )))
            }
        }
    }
    if components.len() != 3 + usize::from(overlap_values.is_some()) * 2 {
        return Err(Exception::custom(format!(
            "MLX pooling stream {stream} at layer {layer} has undeclared components"
        )));
    }
    i32::try_from(ratio.get())
        .map(|ratio| (ratio, overlap_values.is_some()))
        .map_err(|_| Exception::custom("pooling ratio exceeds MLX runtime range"))
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
            .map_err(ComputeError::backend)?;
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
                    .map_err(ComputeError::backend)?,
                stream,
            )
            .map_err(ComputeError::backend)?;
        keys.try_index_device((.., 0, .., ..), stream)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend)
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
            .map_err(ComputeError::backend)?;
        let keys =
            Array::arange::<i32, i32>(Some(key_offset), key_offset + key_tokens, None, stream)
                .and_then(|values| values.try_index_device((NewAxis, ..), stream))
                .map_err(ComputeError::backend)?;
        let causal = queries.ge(&keys, stream).map_err(ComputeError::backend)?;
        let recent = keys
            .gt(
                queries
                    .subtract(Array::from_int(window), stream)
                    .map_err(ComputeError::backend)?,
                stream,
            )
            .map_err(ComputeError::backend)?;
        causal
            .logical_and(&recent, stream)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend)
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
            .map_err(ComputeError::backend)
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
            .map_err(ComputeError::backend)
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
            .map_err(ComputeError::backend)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        stream: &Stream,
    ) -> Result<(), ComputeError> {
        match (self, checkpoint) {
            (Self::Local(local), Self::Local(previous)) => local
                .restore_checkpoint(previous, stream)
                .map_err(ComputeError::backend),
            (
                Self::Compressed { local, pool },
                Self::Compressed {
                    local: previous_local,
                    pool: previous_pool,
                },
            ) => {
                local
                    .restore_checkpoint(previous_local, stream)
                    .map_err(ComputeError::backend)?;
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
                    .map_err(ComputeError::backend)?;
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
        self.local_mut().finalize().map_err(ComputeError::backend)
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        self.clear().map_err(ComputeError::backend)
    }
}
