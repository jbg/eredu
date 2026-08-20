//! Reusable MLX realization of architecture-declared key/value state.

use std::{collections::BTreeMap, path::Path};

use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheOptions, StateTensorOwner, StateTensorRole,
};
use eredu_nn::{Error as ComputeError, PoolingAttentionCache, PoolingOverlap, PoolingWindows};
use eredu_runtime::{
    CacheResidencyReport, LayerRuntimeState, RuntimeLayerState, RuntimeState, StateError,
    StateLayout,
};
use safemlx::{
    error::Exception,
    ops::{
        indexing::{NewAxis, TryIndexOp},
        zeros_dtype,
    },
    Array, Stream,
};

use crate::{
    backend::mlx::{
        nn::shared::MlxBackend,
        runtime::cache::{
            kv::{
                ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, PoolingCacheState,
                RetainedArrayIter,
            },
            residency::{CacheResidencyManager, PromptCacheStateArray},
            LiveKeyValueCache, PoolingCache,
        },
    },
    core::cache::CacheRankIdentity,
};

/// One concrete MLX key/value layer state selected by runtime residency policy.
#[derive(Debug, Clone)]
pub enum MlxKeyValueLayerState {
    /// Contiguous execution-device keys and values.
    Device(ConcatKeyValueCache),
    /// Block-addressable keys and values managed by finite residency budgets.
    Paged(PagedKeyValueCache),
}

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

impl MlxPoolingAttentionCache {
    /// Creates resident state for a scheduled compression ratio.
    pub fn resident(ratio: i32, sliding_window: i32) -> Result<Self, Exception> {
        Self::with_local(
            ratio,
            LiveKeyValueCache::resident(ConcatKeyValueCache::new_for_sliding_attention(
                sliding_window,
            )),
        )
    }

    /// Creates paged local state under one shared manager.
    pub fn paged(
        ratio: i32,
        sliding_window: i32,
        manager: CacheResidencyManager,
        global_layer: usize,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::with_local(
            ratio,
            LiveKeyValueCache::paged_key_only(
                manager,
                global_layer,
                Some(sliding_window),
                prefix_tokens,
                rank,
            )?,
        )
    }

    fn with_local(ratio: i32, local: LiveKeyValueCache) -> Result<Self, Exception> {
        match ratio {
            0 => Ok(Self::Local(local)),
            4 => Ok(Self::Sparse {
                local,
                pool: PoolingCache::new(4)?,
                index_pool: PoolingCache::new(4)?,
            }),
            ratio if ratio > 0 => Ok(Self::Compressed {
                local,
                pool: PoolingCache::new(ratio)?,
            }),
            _ => Err(Exception::custom("pooling ratio must be nonnegative")),
        }
    }

    pub(crate) fn deep_clone_state(&self) -> Result<Self, Exception> {
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

    /// Returns the complete local key-only snapshot for resident publication.
    pub fn local_snapshot(&self, stream: &Stream) -> Result<Option<(Array, Array)>, Exception> {
        self.local().snapshot_arrays(stream)
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

impl RuntimeLayerState<MlxBackend> for MlxPoolingAttentionCache {
    type RetainedValues<'a> = std::vec::IntoIter<&'a Array>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_arrays().into_iter()
    }
}

impl PoolingAttentionCache<Array> for MlxPoolingAttentionCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset()
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.pool(stream).ok().map(PoolingCache::ratio)
    }

    fn append_local(&mut self, keys: Array, stream: &Stream) -> Result<Array, ComputeError> {
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
            .map_err(ComputeError::backend)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &Stream,
    ) -> Result<Array, ComputeError> {
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
            .map_err(ComputeError::backend)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream_id: u32,
        values: Array,
        gates: Array,
        absolute_offset: i32,
        stream: &Stream,
    ) -> Result<PoolingWindows<Array>, ComputeError> {
        self.pool_mut(stream_id)?
            .accumulate_windows(values, gates, absolute_offset, stream)
            .map(|windows| PoolingWindows {
                values: windows.values,
                gates: windows.gates,
                base_position: windows.base_position,
            })
            .map_err(ComputeError::backend)
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: Array,
        gates: Array,
    ) -> Result<PoolingOverlap<Array>, ComputeError> {
        let (values, gates) = self.pool_mut(stream)?.replace_overlap(values, gates);
        Ok(PoolingOverlap { values, gates })
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: Array,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        self.pool_mut(stream)?
            .update_and_fetch(values, context)
            .map_err(ComputeError::backend)
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<Option<Array>, ComputeError> {
        self.pool(stream)?
            .make_mask(query_tokens, offset, context)
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

impl MlxKeyValueLayerState {
    fn clear(&mut self) -> Result<(), Exception> {
        match self {
            Self::Device(cache) => {
                cache.clear();
                Ok(())
            }
            Self::Paged(cache) => cache.clear(),
        }
    }
}

impl KeyValueCache for MlxKeyValueLayerState {
    fn offset(&self) -> i32 {
        match self {
            Self::Device(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Device(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Device(cache) => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Device(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
            Self::Paged(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Device(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Device(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

impl RuntimeLayerState<MlxBackend> for MlxKeyValueLayerState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        match self {
            Self::Device(cache) => RuntimeLayerState::<MlxBackend>::retained_values(cache),
            Self::Paged(cache) => RuntimeLayerState::<MlxBackend>::retained_values(cache),
        }
    }
}

/// Model-wide MLX key/value state created solely from a neutral layout.
#[derive(Debug, Clone)]
pub struct MlxKeyValueState {
    layout: StateLayout,
    layers: Vec<MlxKeyValueLayerState>,
}

impl MlxKeyValueState {
    /// Creates contiguous execution-device state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = key_value_window(layer, policy)?;
                Ok(MlxKeyValueLayerState::Device(match window {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
                    None => ConcatKeyValueCache::new(),
                }))
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self { layout, layers })
    }

    /// Creates block-addressable state using one shared residency manager.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = key_value_window(layer, policy)?;
                PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                    .map(MlxKeyValueLayerState::Paged)
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self { layout, layers })
    }

    /// Returns the common absolute token offset, or zero for an empty state.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, KeyValueCache::offset)
    }

    /// Clears retained arrays without changing residency or attention windows.
    pub fn clear(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    /// Returns aggregate telemetry when this is paged state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => Some(cache.report()),
                MlxKeyValueLayerState::Device(_) => None,
            })
            .transpose()
    }

    /// Finalizes paged tails and atomically persists a completed prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        let mut manager = None;
        for layer in &mut self.layers {
            let MlxKeyValueLayerState::Paged(cache) = layer else {
                return Err(Exception::custom(
                    "prompt-cache persistence requires explicitly paged state",
                ));
            };
            cache.finalize()?;
            manager.get_or_insert_with(|| cache.manager().clone());
        }
        manager
            .ok_or_else(|| Exception::custom("cannot persist empty paged state"))?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl RuntimeState<MlxBackend> for MlxKeyValueState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        self.layers
            .get(ordinal)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer: ordinal,
                count: self.layers.len(),
            })
    }
}

impl LayerRuntimeState<MlxBackend> for MlxKeyValueState {
    type LayerState = MlxKeyValueLayerState;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }
}

impl AsRef<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_ref(&self) -> &[MlxKeyValueLayerState] {
        &self.layers
    }
}

impl AsMut<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_mut(&mut self) -> &mut [MlxKeyValueLayerState] {
        &mut self.layers
    }
}

fn key_value_window(layer: usize, policy: &LayerCachePolicy) -> Result<Option<i32>, Exception> {
    match policy {
        LayerCachePolicy::KeyValue { attention, .. } => attention
            .sliding_window_i32()
            .map_err(|error| Exception::custom(error.to_string())),
        _ => Err(Exception::custom(format!(
            "MLX key/value state cannot realize non-key/value policy at layer {layer}: {policy:?}"
        ))),
    }
}
