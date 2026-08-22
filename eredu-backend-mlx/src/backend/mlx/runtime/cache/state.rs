//! Reusable MLX realization of architecture-declared key/value state.

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut, Range},
    path::Path,
};

use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheOptions, StateTensorDimension, StateTensorOwner, StateTensorPresence,
    StateTensorRole,
};
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_nn::{
    AttentionCache, AttentionRequest, CompressedAttentionBlock, CompressedAttentionCache,
    CompressedAttentionScan, CompressedAttentionState, CompressedAttentionView,
    Error as ComputeError, PoolingAttentionCache, PoolingOverlap, PoolingWindows,
};
use eredu_runtime::{
    CacheResidencyReport, DeviceState, LayerRuntimeState, ResettableRuntimeLayerState,
    ResettableRuntimeState, RuntimeLayerState, RuntimeState, RuntimeStateComponents, StateError,
    StateLayout, StateSegmentId, StateSegmentSpec,
};
use safemlx::{
    error::Exception,
    ops::{
        indexing::{NewAxis, TryIndexOp},
        zeros_dtype,
    },
    Array, Stream,
};

use crate::backend::mlx::{
    nn::shared::MlxBackend,
    runtime::cache::{
        kv::{
            CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
            PagedKeyValueTransactionCheckpoint, PoolingCacheState, RetainedArrayIter,
        },
        residency::{CacheResidencyManager, LoadedPromptCacheStateTensor, PromptCacheStateArray},
        LiveKeyValueCache, PoolingCache,
    },
};
use eredu_core::cache::CacheRankIdentity;
use ref_cast::RefCast;

use crate::MlxTensor;

type RetainedArrayVecIter<'a> =
    std::iter::Map<std::vec::IntoIter<&'a Array>, fn(&'a Array) -> &'a MlxTensor>;

fn retained_tensor(array: &Array) -> &MlxTensor {
    MlxTensor::ref_cast(array)
}

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

/// Complete MLX pooling-attention state realized directly from a neutral
/// architecture layout.
pub type MlxPoolingAttentionState = DeviceState<MlxBackend, MlxPoolingAttentionCache>;

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
mod pooling_layout_tests {
    use std::num::NonZeroU32;

    use eredu_core::{
        cache::{MutableStateResidency, StateResidencyClass, StateTensorDtype, StateTensorPolicy},
        AttentionPolicy, LayerSchedule,
    };

    use super::*;

    fn pooling_stream(stream: u32, ratio: u32, overlapping: bool) -> Vec<StateTensorPolicy> {
        let ratio = NonZeroU32::new(ratio).unwrap();
        let role = |component| StateTensorRole::Pooling { stream, component };
        let pending = |component| {
            StateTensorPolicy::new(
                role(component),
                vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::PrefixTokensRem(ratio),
                    StateTensorDimension::fixed(8).unwrap(),
                ],
                StateTensorDtype::Floating,
                MutableStateResidency::AlwaysDeviceMutable,
            )
            .unwrap()
            .when_prefix_remainder_nonzero(ratio)
        };
        let mut policies = vec![
            pending(PoolingStateComponent::PendingValues),
            pending(PoolingStateComponent::PendingGates),
            StateTensorPolicy::new_with_residency(
                role(PoolingStateComponent::Pooled),
                vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::PrefixTokensDiv(ratio),
                    StateTensorDimension::fixed(8).unwrap(),
                ],
                StateTensorDtype::Floating,
                StateResidencyClass::SealablePaged,
            )
            .unwrap()
            .when_prefix_at_least(ratio),
        ];
        if overlapping {
            for component in [
                PoolingStateComponent::OverlapValues,
                PoolingStateComponent::OverlapGates,
            ] {
                policies.push(
                    StateTensorPolicy::new(
                        role(component),
                        vec![
                            StateTensorDimension::Batch,
                            StateTensorDimension::Fixed(ratio),
                            StateTensorDimension::fixed(8).unwrap(),
                        ],
                        StateTensorDtype::Floating,
                        MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .unwrap()
                    .when_prefix_at_least(ratio),
                );
            }
        }
        policies
    }

    #[test]
    fn materializes_window_and_pooling_streams_from_layer_policy() {
        let mut tensors = pooling_stream(0, 4, true);
        tensors.extend(pooling_stream(1, 6, true));
        let policy = LayerCachePolicy::key_only_with_fixed_state(
            AttentionPolicy::sliding(37).unwrap(),
            1,
            8,
            tensors,
        )
        .unwrap();

        let geometry = pooling_attention_geometry(7, &policy).unwrap();
        assert_eq!(
            geometry,
            PoolingAttentionGeometry {
                sliding_window: 37,
                stream_ratios: vec![4, 6],
            }
        );
        let state = MlxPoolingAttentionCache::resident_from_policy(7, &policy).unwrap();
        let MlxPoolingAttentionCache::Sparse {
            pool, index_pool, ..
        } = state
        else {
            panic!("two declared streams must create sparse pooling state")
        };
        assert_eq!(pool.ratio(), 4);
        assert_eq!(index_pool.ratio(), 6);

        let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
        let state = MlxPoolingAttentionStateFactory::device(layout.clone()).unwrap();
        assert_eq!(state.layout(), &layout);
    }

    #[test]
    fn materializes_local_only_policy_without_family_arguments() {
        let policy =
            LayerCachePolicy::key_only(AttentionPolicy::sliding(23).unwrap(), 1, 8).unwrap();

        assert_eq!(
            pooling_attention_geometry(2, &policy).unwrap(),
            PoolingAttentionGeometry {
                sliding_window: 23,
                stream_ratios: Vec::new(),
            }
        );
        assert!(matches!(
            MlxPoolingAttentionCache::resident_from_policy(2, &policy).unwrap(),
            MlxPoolingAttentionCache::Local(_)
        ));
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

    fn deep_clone_state(&self) -> Result<Self, Exception> {
        match self {
            Self::Device(cache) => cache.deep_clone_state().map(Self::Device),
            Self::Paged(cache) => cache.deep_clone_state().map(Self::Paged),
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
            (Self::Device(current), Self::Device(previous)) => {
                match previous.snapshot_arrays(stream)? {
                    Some((keys, values)) => {
                        current.restore_resident(keys, values, KeyValueCache::offset(previous))
                    }
                    None => {
                        current.clear();
                        Ok(())
                    }
                }
            }
            (Self::Paged(current), Self::Paged(previous)) => {
                current.restore_checkpoint(previous, stream)
            }
            _ => Err(Exception::custom(
                "key/value checkpoint representation changed",
            )),
        }
    }
}

impl KeyValueCache for MlxKeyValueLayerState {
    fn offset(&self) -> i32 {
        match self {
            Self::Device(cache) => KeyValueCache::offset(cache),
            Self::Paged(cache) => KeyValueCache::offset(cache),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Device(cache) => KeyValueCache::max_size(cache),
            Self::Paged(cache) => KeyValueCache::max_size(cache),
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
            Self::Device(cache) => KeyValueCache::update_for_attention(cache, keys, values, stream),
            Self::Paged(cache) => KeyValueCache::update_for_attention(cache, keys, values, stream),
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

impl ResettableRuntimeLayerState<MlxBackend> for MlxKeyValueLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.clear()
            .map_err(|error| StateError::ResetFailed(error.to_string()))
    }
}

/// Model-wide MLX key/value state created solely from a neutral layout.
#[derive(Debug, Clone)]
pub struct MlxKeyValueState {
    layout: StateLayout,
    layers: Vec<MlxKeyValueLayerState>,
    paged_transaction_branch: bool,
}

/// Unpublished, independently mutable resident MLX key/value state.
///
/// The private wrapper prevents callers from satisfying a semantic transaction
/// with the public shallow [`Clone`] implementation. It can only be created by
/// [`SemanticStateTransaction::branch`], which uses exact deep array clones.
#[derive(Debug)]
pub struct MlxKeyValueTransactionBranch {
    state: MlxKeyValueState,
    paged_rollback: Vec<Option<PagedKeyValueTransactionCheckpoint>>,
}

impl Deref for MlxKeyValueTransactionBranch {
    type Target = MlxKeyValueState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for MlxKeyValueTransactionBranch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
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
        Ok(Self {
            layout,
            layers,
            paged_transaction_branch: false,
        })
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
        Ok(Self {
            layout,
            layers,
            paged_transaction_branch: false,
        })
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

    /// Borrows every native array retained by the complete model state.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(RuntimeLayerState::<MlxBackend>::retained_values)
            .map(MlxTensor::as_array)
            .collect()
    }

    /// Creates an independently advanceable speculative fork.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        Ok(Self {
            layout: self.layout.clone(),
            layers: self
                .layers
                .iter()
                .map(MlxKeyValueLayerState::deep_clone_state)
                .collect::<Result<_, _>>()?,
            paged_transaction_branch: self.paged_transaction_branch,
        })
    }

    fn has_same_transaction_identity(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.layers.len() == other.layers.len()
            && self
                .layers
                .iter()
                .zip(&other.layers)
                .all(|(canonical, branch)| match (canonical, branch) {
                    (MlxKeyValueLayerState::Device(_), MlxKeyValueLayerState::Device(_)) => true,
                    (
                        MlxKeyValueLayerState::Paged(canonical),
                        MlxKeyValueLayerState::Paged(branch),
                    ) => canonical.has_same_transaction_identity(branch),
                    _ => false,
                })
    }

    /// Restores every append-only layer to an exact speculative checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layout != checkpoint.layout || self.layers.len() != checkpoint.layers.len() {
            return Err(Exception::custom(
                "key/value state checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            current.restore_checkpoint(previous, stream)?;
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

impl SemanticStateTransaction for MlxKeyValueState {
    type Branch = MlxKeyValueTransactionBranch;
    type Error = Exception;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        let paged_rollback = self
            .layers
            .iter()
            .map(|layer| match layer {
                MlxKeyValueLayerState::Device(_) => Ok(None),
                MlxKeyValueLayerState::Paged(cache) => cache.transaction_checkpoint().map(Some),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.deep_clone_state().map(|mut state| {
            state.paged_transaction_branch = paged_rollback.iter().any(Option::is_some);
            MlxKeyValueTransactionBranch {
                state,
                paged_rollback,
            }
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        if !self.has_same_transaction_identity(&branch.state) {
            Self::discard_branch(branch)?;
            return Err(Exception::custom(
                "key/value transaction branch identity does not match canonical state",
            ));
        }
        let mut state = branch.state;
        state.paged_transaction_branch = false;
        *self = state;
        Ok(())
    }

    fn discard_branch(mut branch: Self::Branch) -> Result<(), Self::Error> {
        let mut first_error = None;
        for (layer, checkpoint) in branch
            .state
            .layers
            .iter_mut()
            .zip(branch.paged_rollback.iter())
        {
            let (MlxKeyValueLayerState::Paged(cache), Some(checkpoint)) = (layer, checkpoint)
            else {
                continue;
            };
            if let Err(error) = cache.rollback_transaction(checkpoint) {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn permits_parallel_branches(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| matches!(layer, MlxKeyValueLayerState::Device(_)))
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
        let layer = ordinal;
        self.layers
            .get(layer)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
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

impl ResettableRuntimeState<MlxBackend> for MlxKeyValueState {
    fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError> {
        let range = self
            .layout
            .segment(segment)
            .map(StateSegmentSpec::layers)
            .ok_or_else(|| StateError::UnknownSegment {
                segment: segment.clone(),
            })?;
        if self.paged_transaction_branch
            && self.layers[range.clone()]
                .iter()
                .any(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_)))
        {
            return Err(StateError::ResetFailed(
                "paged state segments cannot be reset inside a transaction branch without copy-on-write page ownership"
                    .into(),
            ));
        }
        for layer in &mut self.layers[range] {
            ResettableRuntimeLayerState::<MlxBackend>::reset(layer)?;
        }
        Ok(())
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

/// Append-only attention state selected independently from fixed components.
#[derive(Debug, Clone)]
enum MlxHybridAttentionState {
    KeyValue(MlxKeyValueLayerState),
    Compressed(CompressedLatentCache),
}

impl MlxHybridAttentionState {
    fn deep_clone_state(&self) -> Result<Self, Exception> {
        match self {
            Self::KeyValue(MlxKeyValueLayerState::Device(cache)) => cache
                .deep_clone_state()
                .map(MlxKeyValueLayerState::Device)
                .map(Self::KeyValue),
            Self::KeyValue(MlxKeyValueLayerState::Paged(cache)) => cache
                .deep_clone_state()
                .map(MlxKeyValueLayerState::Paged)
                .map(Self::KeyValue),
            Self::Compressed(cache) => cache.deep_clone_state().map(Self::Compressed),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::KeyValue(cache) => AttentionCache::offset(cache),
            Self::Compressed(cache) => CompressedAttentionCache::offset(cache),
        }
    }

    fn clear(&mut self) -> Result<(), Exception> {
        match self {
            Self::KeyValue(cache) => cache.clear(),
            Self::Compressed(cache) => CompressedAttentionCache::clear(cache)
                .map_err(|error| Exception::custom(error.to_string())),
        }
    }

    fn retained_values(&self) -> Vec<&MlxTensor> {
        match self {
            Self::KeyValue(cache) => {
                RuntimeLayerState::<MlxBackend>::retained_values(cache).collect()
            }
            Self::Compressed(cache) => {
                RuntimeLayerState::<MlxBackend>::retained_values(cache).collect()
            }
        }
    }

    fn manager(&self) -> Option<&CacheResidencyManager> {
        match self {
            Self::KeyValue(MlxKeyValueLayerState::Paged(cache)) => Some(cache.manager()),
            Self::Compressed(cache) => cache.residency_manager(),
            Self::KeyValue(MlxKeyValueLayerState::Device(_)) => None,
        }
    }

    fn finalize(&mut self) -> Result<(), Exception> {
        match self {
            Self::KeyValue(MlxKeyValueLayerState::Paged(cache)) => cache.finalize(),
            Self::Compressed(cache) if cache.is_paged() => cache.finalize(),
            _ => Err(Exception::custom(
                "prompt-cache persistence requires paged attention state",
            )),
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
            (
                Self::KeyValue(MlxKeyValueLayerState::Device(current)),
                Self::KeyValue(MlxKeyValueLayerState::Device(previous)),
            ) => match previous.snapshot_arrays(stream)? {
                Some((keys, values)) => {
                    current.restore_resident(keys, values, KeyValueCache::offset(previous))
                }
                None => {
                    current.clear();
                    Ok(())
                }
            },
            (
                Self::KeyValue(MlxKeyValueLayerState::Paged(current)),
                Self::KeyValue(MlxKeyValueLayerState::Paged(previous)),
            ) => current.restore_checkpoint(previous, stream),
            (Self::Compressed(current), Self::Compressed(previous)) => current
                .restore(&previous.checkpoint(), stream)
                .map_err(|error| Exception::custom(error.to_string())),
            _ => Err(Exception::custom(
                "hybrid attention checkpoint representation changed",
            )),
        }
    }
}

/// One MLX realization of a heterogeneous attention/fixed-state layer.
#[derive(Debug, Clone)]
pub struct MlxHybridLayerState {
    attention: Option<MlxHybridAttentionState>,
    fixed: BTreeMap<StateTensorRole, Option<MlxTensor>>,
    fixed_offset: i32,
}

impl MlxHybridLayerState {
    fn deep_clone_state(&self) -> Result<Self, Exception> {
        let attention = self
            .attention
            .as_ref()
            .map(MlxHybridAttentionState::deep_clone_state)
            .transpose()?;
        let fixed = self
            .fixed
            .iter()
            .map(|(role, value)| {
                value
                    .as_ref()
                    .map(|array| array.as_array().clone().deep_clone())
                    .transpose()
                    .map(|value| (*role, value.map(MlxTensor::from_array)))
            })
            .collect::<Result<_, Exception>>()?;
        Ok(Self {
            attention,
            fixed,
            fixed_offset: self.fixed_offset,
        })
    }

    fn device(layer: usize, policy: &LayerCachePolicy) -> Result<Self, Exception> {
        let attention = hybrid_attention_policy(layer, policy)?.map(|policy| match policy {
            HybridAttentionPolicy::KeyValue { window, key_only } => {
                MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(
                    match (window, key_only) {
                        (Some(window), true) => {
                            ConcatKeyValueCache::new_key_only_for_sliding_attention(window)
                        }
                        (Some(window), false) => {
                            ConcatKeyValueCache::new_for_sliding_attention(window)
                        }
                        (None, true) => ConcatKeyValueCache::new_key_only(),
                        (None, false) => ConcatKeyValueCache::new(),
                    },
                ))
            }
            HybridAttentionPolicy::Compressed => {
                MlxHybridAttentionState::Compressed(CompressedLatentCache::new())
            }
        });
        Ok(Self {
            attention,
            fixed: policy
                .fixed_state()
                .iter()
                .map(|tensor| (tensor.role, None))
                .collect(),
            fixed_offset: 0,
        })
    }

    fn paged(
        layer: usize,
        policy: &LayerCachePolicy,
        manager: &CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let attention = hybrid_attention_policy(layer, policy)?
            .map(|policy| match policy {
                HybridAttentionPolicy::KeyValue { window, key_only } => if key_only {
                    PagedKeyValueCache::new_key_only_with_layout(
                        manager.clone(),
                        layer,
                        window,
                        0,
                        rank,
                    )
                } else {
                    PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                }
                .map(MlxKeyValueLayerState::Paged)
                .map(MlxHybridAttentionState::KeyValue),
                HybridAttentionPolicy::Compressed => {
                    CompressedLatentCache::new_paged(manager.clone(), layer, rank)
                        .map(MlxHybridAttentionState::Compressed)
                }
            })
            .transpose()?;
        Ok(Self {
            attention,
            fixed: policy
                .fixed_state()
                .iter()
                .map(|tensor| (tensor.role, None))
                .collect(),
            fixed_offset: 0,
        })
    }

    /// Clears attention and fixed components while retaining their policies.
    pub fn clear(&mut self) -> Result<(), Exception> {
        if let Some(attention) = &mut self.attention {
            attention.clear()?;
        }
        for component in self.fixed.values_mut() {
            *component = None;
        }
        self.fixed_offset = 0;
        Ok(())
    }
}

impl RuntimeLayerState<MlxBackend> for MlxHybridLayerState {
    type RetainedValues<'a> = std::vec::IntoIter<&'a MlxTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        let mut retained = self
            .attention
            .as_ref()
            .map(MlxHybridAttentionState::retained_values)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        retained.extend(self.fixed.values().filter_map(Option::as_ref));
        retained.into_iter()
    }
}

impl RuntimeStateComponents<MlxBackend> for MlxHybridLayerState {
    fn position(&self) -> i32 {
        self.attention
            .as_ref()
            .map_or(self.fixed_offset, MlxHybridAttentionState::offset)
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<MlxTensor>, StateError> {
        self.fixed
            .get_mut(&role)
            .ok_or(StateError::UnknownComponent { role })
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        if self.attention.is_some() {
            return Err(StateError::InvalidAdvance(
                "attention-backed layers advance through cache append".into(),
            ));
        }
        if tokens <= 0 {
            return Err(StateError::InvalidAdvance(format!(
                "token count must be positive, got {tokens}"
            )));
        }
        self.fixed_offset = self.fixed_offset.checked_add(tokens).ok_or_else(|| {
            StateError::InvalidAdvance("fixed-state token frontier overflowed".into())
        })?;
        Ok(())
    }
}

impl AttentionCache<MlxTensor> for MlxHybridLayerState {
    fn offset(&self) -> i32 {
        self.position()
    }

    fn max_size(&self) -> Option<i32> {
        match self.attention.as_ref() {
            Some(MlxHybridAttentionState::KeyValue(cache)) => AttentionCache::max_size(cache),
            _ => None,
        }
    }

    fn update_for_attention(
        &mut self,
        keys: MlxTensor,
        values: MlxTensor,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor), ComputeError> {
        AttentionCache::update_for_attention(
            match self.attention.as_mut() {
                Some(MlxHybridAttentionState::KeyValue(cache)) => cache,
                _ => {
                    return Err(ComputeError::backend(
                        "layer has no key/value attention cache",
                    ))
                }
            },
            keys,
            values,
            stream,
        )
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        AttentionCache::attention(
            match self.attention.as_mut() {
                Some(MlxHybridAttentionState::KeyValue(cache)) => cache,
                _ => {
                    return Err(ComputeError::backend(
                        "layer has no key/value attention cache",
                    ))
                }
            },
            request,
            stream,
        )
    }
}

impl CompressedAttentionCache<MlxTensor> for MlxHybridLayerState {
    type Checkpoint = CompressedLatentCache;

    fn offset(&self) -> i32 {
        self.position()
    }

    fn is_paged(&self) -> bool {
        matches!(
            self.attention.as_ref(),
            Some(MlxHybridAttentionState::Compressed(cache)) if cache.is_paged()
        )
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<MlxTensor>,
        context: &Stream,
    ) -> Result<CompressedAttentionView<MlxTensor>, ComputeError> {
        match self.attention.as_mut() {
            Some(MlxHybridAttentionState::Compressed(cache)) => cache.append(state, context),
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &Stream,
        visitor: F,
    ) -> Result<CompressedAttentionScan, ComputeError>
    where
        F: FnMut(CompressedAttentionBlock<MlxTensor>) -> Result<u64, ComputeError>,
    {
        match self.attention.as_mut() {
            Some(MlxHybridAttentionState::Compressed(cache)) => {
                cache.visit_blocks(query_tokens, context, visitor)
            }
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        match self.attention.as_ref() {
            Some(MlxHybridAttentionState::Compressed(cache)) => cache.checkpoint(),
            _ => panic!("layer has no compressed-latent attention cache"),
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &Stream,
    ) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(MlxHybridAttentionState::Compressed(cache)) => cache.restore(checkpoint, context),
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn finalize(&mut self) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(MlxHybridAttentionState::Compressed(cache)) => {
                CompressedAttentionCache::finalize(cache)
            }
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(MlxHybridAttentionState::Compressed(cache)) => {
                CompressedAttentionCache::clear(cache)
            }
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }
}

/// MLX state realization for a schedule mixing attention and fixed components.
#[derive(Debug, Clone)]
pub struct MlxHybridState {
    layout: StateLayout,
    global_layer_start: usize,
    layers: Vec<MlxHybridLayerState>,
}

impl MlxHybridState {
    /// Creates device-resident attention and fixed state from a neutral layout.
    pub fn device(layout: StateLayout) -> Result<Self, Exception> {
        Self::device_with_global_layer_start(layout, 0)
    }

    /// Creates device-resident state addressed from an architecture-global layer.
    pub fn device_with_global_layer_start(
        layout: StateLayout,
        global_layer_start: usize,
    ) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| MlxHybridLayerState::device(layer, policy))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
        })
    }

    /// Creates paged attention with device-resident fixed components.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::paged_with_global_layer_start(layout, manager, rank, 0)
    }

    /// Creates paged state addressed from an architecture-global layer.
    pub fn paged_with_global_layer_start(
        layout: StateLayout,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                    Exception::custom("hybrid state global layer index overflowed")
                })?;
                MlxHybridLayerState::paged(global_layer, policy, &manager, rank)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
        })
    }

    /// Mutably borrows the ordinary per-layer states used by neutral units.
    pub fn layers_mut(&mut self) -> &mut [MlxHybridLayerState] {
        &mut self.layers
    }

    /// Borrows the paging manager shared by this state's attention components.
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.layers
            .iter()
            .filter_map(|layer| layer.attention.as_ref())
            .find_map(MlxHybridAttentionState::manager)
    }

    /// Returns the common absolute token frontier.
    pub fn offset(&self) -> i32 {
        self.layers
            .first()
            .map_or(0, RuntimeStateComponents::position)
    }

    /// Clears every heterogeneous component.
    pub fn clear(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    /// Creates an independently advanceable speculative fork.
    ///
    /// Mutable device arrays are copied. Paged forks share only immutable
    /// sealed blocks and the architecture-independent residency manager.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers: self
                .layers
                .iter()
                .map(MlxHybridLayerState::deep_clone_state)
                .collect::<Result<_, _>>()?,
        })
    }

    /// Restores append-only and fixed state to an exact speculative checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layout != checkpoint.layout
            || self.global_layer_start != checkpoint.global_layer_start
            || self.layers.len() != checkpoint.layers.len()
        {
            return Err(Exception::custom(
                "hybrid state checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            match (&mut current.attention, &previous.attention) {
                (Some(current), Some(previous)) => {
                    current.restore_checkpoint(previous, stream)?;
                }
                (None, None) => {}
                _ => {
                    return Err(Exception::custom(
                        "hybrid state checkpoint attention policy changed",
                    ))
                }
            }
            current.fixed.clone_from(&previous.fixed);
            current.fixed_offset = previous.fixed_offset;
        }
        Ok(())
    }

    /// Commits a contiguous architecture-owned execution-group state range.
    pub fn commit_layer_range_from(
        &mut self,
        source: &Self,
        start: usize,
    ) -> Result<(), Exception> {
        if self.layout != source.layout
            || self.global_layer_start != source.global_layer_start
            || start > self.layers.len()
        {
            return Err(Exception::custom(
                "hybrid draft state layout does not match canonical state",
            ));
        }
        self.layers[start..].clone_from_slice(&source.layers[start..]);
        Ok(())
    }

    /// Returns aggregate telemetry when this state contains paged attention.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .filter_map(|layer| layer.attention.as_ref())
            .find_map(MlxHybridAttentionState::manager)
            .map(CacheResidencyManager::report)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Restores all fixed components materialized from a validated manifest.
    pub fn restore_prompt_cache_state(
        &mut self,
        tensors: Vec<LoadedPromptCacheStateTensor>,
        processed_tokens: i32,
        layer_prefix_offsets: &[i32],
    ) -> Result<(), Exception> {
        if processed_tokens < 0 {
            return Err(Exception::custom(format!(
                "prompt-cache token frontier must be non-negative, got {processed_tokens}"
            )));
        }
        if layer_prefix_offsets.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "prompt-cache supplied {} layer frontiers for {} hybrid state layers",
                layer_prefix_offsets.len(),
                self.layers.len()
            )));
        }
        let mut tensors = tensors
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        for (layer, ((state, policy), delta)) in self
            .layers
            .iter_mut()
            .zip(self.layout.layers().iter())
            .zip(layer_prefix_offsets)
            .enumerate()
        {
            let frontier = processed_tokens.checked_add(*delta).ok_or_else(|| {
                Exception::custom(format!(
                    "prompt-cache frontier overflowed at hybrid state layer {layer}"
                ))
            })?;
            if frontier < 0 {
                return Err(Exception::custom(format!(
                    "prompt-cache frontier is negative at hybrid state layer {layer}"
                )));
            }
            for (role, slot) in &mut state.fixed {
                let owner = StateTensorOwner::Layer(self.global_layer_start + layer);
                let restored = tensors.remove(&(owner, *role));
                let state_policy = policy
                    .fixed_state()
                    .iter()
                    .find(|candidate| candidate.role == *role)
                    .expect("realized fixed state comes from its canonical policy");
                if restored.is_none()
                    && frontier != 0
                    && state_policy.is_required_for(frontier as usize)
                {
                    return Err(Exception::custom(format!(
                        "prompt cache is missing required fixed state {owner:?}/{role:?}"
                    )));
                }
                *slot = restored.map(MlxTensor::from_array);
            }
            if state.attention.is_none() {
                state.fixed_offset = frontier;
            }
        }
        if let Some(((owner, role), _)) = tensors.into_iter().next() {
            return Err(Exception::custom(format!(
                "prompt cache contains undeclared fixed state {owner:?}/{role:?}"
            )));
        }
        Ok(())
    }

    /// Restores one architecture-owned state range from a shared prompt catalog.
    pub fn restore_prompt_cache_state_range(
        &mut self,
        tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), Array>,
        range: Range<usize>,
        processed_tokens: i32,
        layer_prefix_offsets: &[i32],
    ) -> Result<(), Exception> {
        if range.end > self.layers.len() || layer_prefix_offsets.len() != range.len() {
            return Err(Exception::custom(
                "prompt-cache state range does not match hybrid state layout",
            ));
        }
        for (relative, layer) in range.enumerate() {
            let state = &mut self.layers[layer];
            let policy = self
                .layout
                .layers()
                .get(layer)
                .expect("validated hybrid state range is inside its layout");
            let frontier = processed_tokens
                .checked_add(layer_prefix_offsets[relative])
                .ok_or_else(|| Exception::custom("prompt-cache layer frontier overflowed"))?;
            if frontier < 0 {
                return Err(Exception::custom(format!(
                    "prompt-cache frontier is negative at hybrid state layer {layer}"
                )));
            }
            for (role, slot) in &mut state.fixed {
                let owner = StateTensorOwner::Layer(self.global_layer_start + layer);
                let restored = tensors.remove(&(owner, *role));
                let state_policy = policy
                    .fixed_state()
                    .iter()
                    .find(|candidate| candidate.role == *role)
                    .expect("realized fixed state comes from its canonical policy");
                if restored.is_none()
                    && frontier != 0
                    && state_policy.is_required_for(frontier as usize)
                {
                    return Err(Exception::custom(format!(
                        "prompt cache is missing required fixed state {owner:?}/{role:?}"
                    )));
                }
                *slot = restored.map(MlxTensor::from_array);
            }
            if state.attention.is_none() {
                state.fixed_offset = frontier;
            }
        }
        Ok(())
    }

    /// Finalizes and catalogs fixed tensors for one architecture-owned range.
    pub fn prompt_cache_state_arrays_range(
        &mut self,
        range: Range<usize>,
        processed_tokens: i32,
        layer_prefix_offsets: &[i32],
    ) -> Result<Vec<PromptCacheStateArray<'_>>, Exception> {
        if range.end > self.layers.len() || layer_prefix_offsets.len() != range.len() {
            return Err(Exception::custom(
                "prompt-cache state range does not match hybrid state layout",
            ));
        }
        for (relative, layer) in range.clone().enumerate() {
            let state = &mut self.layers[layer];
            let frontier = processed_tokens
                .checked_add(layer_prefix_offsets[relative])
                .ok_or_else(|| Exception::custom("prompt-cache layer frontier overflowed"))?;
            if frontier < 0 || state.position() != frontier {
                return Err(Exception::custom(format!(
                    "hybrid state layer {layer} is at {}, expected prefix frontier {frontier}",
                    state.position()
                )));
            }
            if let Some(attention) = &mut state.attention {
                attention.finalize()?;
                if attention.manager().is_none() {
                    return Err(Exception::custom(
                        "prompt persistence requires paged hybrid attention state",
                    ));
                }
            }
        }
        let global_layer_start = self.global_layer_start;
        Ok(range
            .flat_map(|layer| {
                self.layers[layer]
                    .fixed
                    .iter()
                    .filter_map(move |(role, value)| {
                        value.as_ref().map(|array| PromptCacheStateArray {
                            owner: StateTensorOwner::Layer(global_layer_start + layer),
                            role: *role,
                            array: array.as_array(),
                        })
                    })
            })
            .collect())
    }

    /// Finalizes paged attention and persists every declared fixed component.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        let expected = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("prompt-cache prefix length exceeds i32"))?;
        if descriptor.layer_prefix_offsets.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "prompt-cache descriptor supplied {} layer frontiers for {} hybrid state layers",
                descriptor.layer_prefix_offsets.len(),
                self.layers.len()
            )));
        }
        let mut manager = None;
        for (layer, (state, delta)) in self
            .layers
            .iter_mut()
            .zip(&descriptor.layer_prefix_offsets)
            .enumerate()
        {
            let layer_expected = expected.checked_add(*delta).ok_or_else(|| {
                Exception::custom(format!(
                    "prompt-cache frontier overflowed at hybrid state layer {layer}"
                ))
            })?;
            if layer_expected < 0 || state.position() != layer_expected {
                return Err(Exception::custom(format!(
                    "hybrid state layer {layer} is at {}, expected prefix frontier {layer_expected}",
                    state.position(),
                )));
            }
            if let Some(attention) = &mut state.attention {
                attention.finalize()?;
                manager.get_or_insert_with(|| {
                    attention
                        .manager()
                        .expect("finalized paged attention has a manager")
                        .clone()
                });
            }
        }
        let global_layer_start = self.global_layer_start;
        let state_arrays = self
            .layers
            .iter()
            .enumerate()
            .flat_map(|(layer, state)| {
                state.fixed.iter().filter_map(move |(role, value)| {
                    value.as_ref().map(|array| PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(global_layer_start + layer),
                        role: *role,
                        array: array.as_array(),
                    })
                })
            })
            .collect::<Vec<_>>();
        manager
            .ok_or_else(|| Exception::custom("cannot persist hybrid state without attention"))?
            .save_prompt_cache(
                destination,
                descriptor,
                prefix_token_ids,
                &state_arrays,
                options,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl RuntimeState<MlxBackend> for MlxHybridState {
    type RetainedValues<'a> = std::vec::IntoIter<&'a MlxTensor>;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let layer = ordinal;
        self.layers
            .get(layer)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl LayerRuntimeState<MlxBackend> for MlxHybridState {
    type LayerState = MlxHybridLayerState;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }
}

#[derive(Debug, Clone, Copy)]
enum HybridAttentionPolicy {
    KeyValue { window: Option<i32>, key_only: bool },
    Compressed,
}

fn hybrid_attention_policy(
    _layer: usize,
    policy: &LayerCachePolicy,
) -> Result<Option<HybridAttentionPolicy>, Exception> {
    match policy {
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => Ok(None),
        LayerCachePolicy::KeyValue { attention, .. }
        | LayerCachePolicy::KeyValueWithFixedState { attention, .. } => attention
            .sliding_window_i32()
            .map(|window| HybridAttentionPolicy::KeyValue {
                window,
                key_only: false,
            })
            .map(Some)
            .map_err(|error| Exception::custom(error.to_string())),
        LayerCachePolicy::KeyOnly { attention, .. }
        | LayerCachePolicy::KeyOnlyWithFixedState { attention, .. } => attention
            .sliding_window_i32()
            .map(|window| HybridAttentionPolicy::KeyValue {
                window,
                key_only: true,
            })
            .map(Some)
            .map_err(|error| Exception::custom(error.to_string())),
        LayerCachePolicy::CompressedLatentRotary { .. } => {
            Ok(Some(HybridAttentionPolicy::Compressed))
        }
    }
}

#[cfg(test)]
mod semantic_transaction_tests {
    use super::*;
    use eredu_core::{cache::CacheRepresentation, AttentionPolicy, LayerSchedule};
    use eredu_runtime::{PagedCacheOptions, StateSegmentLifetime};

    fn layout(window: Option<i32>) -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(
                    AttentionPolicy::from_sliding_window(window).unwrap(),
                    1,
                    8,
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn manager() -> CacheResidencyManager {
        CacheResidencyManager::new(
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap()
    }

    fn segmented_layout() -> StateLayout {
        let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
        StateLayout::segmented(
            LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap(),
            [
                StateSegmentSpec::new("temporal", 0..1, StateSegmentLifetime::Persistent).unwrap(),
                StateSegmentSpec::new("depth", 1..2, StateSegmentLifetime::FrameLocal).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn device_transaction_uses_an_independent_semantic_branch() {
        let mut canonical = MlxKeyValueState::device(layout(Some(4))).unwrap();
        let branch = canonical.branch().unwrap();

        assert!(canonical.permits_parallel_branches());
        assert!(canonical.has_same_transaction_identity(&branch));
        canonical.commit_branch(branch).unwrap();
    }

    #[test]
    fn transaction_rejects_layout_and_residency_changes() {
        let canonical = MlxKeyValueState::device(layout(Some(4))).unwrap();
        let different_layout = MlxKeyValueState::device(layout(Some(8))).unwrap();
        assert!(!canonical.has_same_transaction_identity(&different_layout));

        let paged = MlxKeyValueState::paged(layout(Some(4)), manager(), None).unwrap();
        assert!(!canonical.has_same_transaction_identity(&paged));
    }

    #[test]
    fn paged_transaction_discard_restores_shared_manager_frontier() {
        let canonical = MlxKeyValueState::paged(layout(None), manager(), None).unwrap();
        assert!(!canonical.permits_parallel_branches());
        let branch = canonical.branch().unwrap();
        let manager = match &branch.state.layers[0] {
            MlxKeyValueLayerState::Paged(cache) => cache.manager().clone(),
            MlxKeyValueLayerState::Device(_) => unreachable!(),
        };
        manager.set_tail_state(0, 0, 3).unwrap();
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 3);

        MlxKeyValueState::discard_branch(branch).unwrap();

        assert_eq!(canonical.offset(), 0);
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 0);
    }

    #[test]
    fn paged_transaction_rejects_segment_reset_before_shared_page_mutation() {
        let canonical = MlxKeyValueState::paged(segmented_layout(), manager(), None).unwrap();
        let mut branch = canonical.branch().unwrap();

        let error = branch
            .reset_segment(&StateSegmentId::new("depth").unwrap())
            .unwrap_err();

        assert!(error.to_string().contains("copy-on-write page ownership"));
        MlxKeyValueState::discard_branch(branch).unwrap();
        assert_eq!(canonical.offset(), 0);
    }

    #[test]
    #[ignore = "requires local MLX execution"]
    fn paged_depth_segment_reset_preserves_temporal_pages_and_later_rollback() {
        let manager = manager();
        let mut canonical =
            MlxKeyValueState::paged(segmented_layout(), manager.clone(), None).unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for (layer, value) in [(0usize, 1.0f32), (1, 2.0)] {
            canonical.layers[layer]
                .update_and_fetch(
                    Array::from_slice(&[value; 5], &[1, 1, 5, 1]),
                    Array::from_slice(&[value + 10.0; 5], &[1, 1, 5, 1]),
                    &stream,
                )
                .unwrap();
        }
        assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
        assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 5);
        assert!(!manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .is_empty());
        assert!(!manager
            .layer_block_ids(1, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .is_empty());

        canonical
            .reset_segment(&StateSegmentId::new("depth").unwrap())
            .unwrap();

        assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
        assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 0);
        assert!(!manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .is_empty());
        assert!(manager
            .layer_block_ids(1, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .is_empty());

        let mut discarded = canonical.branch().unwrap();
        discarded.layers[1]
            .update_and_fetch(
                Array::from_slice(&[3.0f32; 2], &[1, 1, 2, 1]),
                Array::from_slice(&[4.0f32; 2], &[1, 1, 2, 1]),
                &stream,
            )
            .unwrap();
        MlxKeyValueState::discard_branch(discarded).unwrap();
        assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
        assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 0);

        let mut resumed = canonical.branch().unwrap();
        resumed.layers[1]
            .update_and_fetch(
                Array::from_slice(&[5.0f32; 2], &[1, 1, 2, 1]),
                Array::from_slice(&[6.0f32; 2], &[1, 1, 2, 1]),
                &stream,
            )
            .unwrap();
        canonical.commit_branch(resumed).unwrap();
        assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
        assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 2);
    }

    #[test]
    fn paged_sliding_transaction_fails_before_branch_publication() {
        let canonical = MlxKeyValueState::paged(layout(Some(4)), manager(), None).unwrap();
        let error = canonical.branch().unwrap_err();
        assert!(error.to_string().contains("paged sliding attention"));
    }

    #[test]
    #[ignore = "requires local MLX execution"]
    fn mlx_realtime_transaction_paged_rollback_release_resume() {
        let mut canonical = MlxKeyValueState::paged(layout(None), manager(), None).unwrap();
        let mut branch = canonical.branch().unwrap();
        let manager = match &branch.state.layers[0] {
            MlxKeyValueLayerState::Paged(cache) => cache.manager().clone(),
            MlxKeyValueLayerState::Device(_) => unreachable!(),
        };
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let keys = Array::from_slice(&[1.0_f32; 5], &[1, 1, 5, 1]);
        let values = Array::from_slice(&[2.0_f32; 5], &[1, 1, 5, 1]);
        branch.state.layers[0]
            .update_and_fetch(keys, values, &stream)
            .unwrap();
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 5);

        MlxKeyValueState::discard_branch(branch).unwrap();

        let report = manager.report().unwrap();
        assert_eq!(report.logical_cached_tokens, 0);
        assert_eq!(canonical.offset(), 0);

        let mut resumed = canonical.branch().unwrap();
        let keys = Array::from_slice(&[3.0_f32; 2], &[1, 1, 2, 1]);
        let values = Array::from_slice(&[4.0_f32; 2], &[1, 1, 2, 1]);
        resumed.state.layers[0]
            .update_and_fetch(keys, values, &stream)
            .unwrap();
        canonical.commit_branch(resumed).unwrap();
        assert_eq!(canonical.offset(), 2);
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 2);
    }
}
