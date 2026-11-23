//! Mixed attention, fixed components, and named streams for an installed model.

use std::{collections::BTreeMap, ops::Range};

use eredu_core::{
    cache::{
        append_stream::{AppendStreamBacking, AppendStreamBinding, AppendStreamError},
        mechanisms::{CompressedCacheStorage, PagedKeyValueStorage, ResidentKeyValueStorage},
        CacheRankIdentity, LayerCachePolicy, PromptCacheDescriptor, StateTensorOwner,
        StateTensorRole,
    },
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};
use eredu_nn::{
    AttentionCache, AttentionRequest, CompressedAttentionCache, CompressedAttentionState,
    CompressedAttentionView, Error as ComputeError, NeuralBackend,
};

use super::{
    HybridAttentionState, KeyValueLayerState, KeyValueStateError, LayerRuntimeState,
    RuntimeLayerState, RuntimeState, RuntimeStateComponents, StateError, StateLayout,
    StateSegmentId, StateSegmentSpec,
};
use crate::{
    cache::{
        live::{
            CacheExport, CacheResidencyError, CacheResidencyManager, LoadedPromptCacheStateTensor,
            PromptCacheStateArray,
        },
        CompressedLatentCache, LayerAppendStream, LayerAppendStreamError, LayerAppendStreams,
    },
    execution_control::{SnapshotRetention, SnapshotValue},
    CacheResidencyReport, RuntimeAppendStreams,
};

/// Storage primitives used by heterogeneous state.
pub trait HybridCacheStorage:
    ResidentKeyValueStorage + PagedKeyValueStorage + CompressedCacheStorage
{
}
impl<M: ResidentKeyValueStorage + PagedKeyValueStorage + CompressedCacheStorage> HybridCacheStorage
    for M
{
}

/// Failure to assemble, copy, or restore a portable hybrid state.
#[derive(Debug, thiserror::Error)]
pub enum HybridStateError {
    /// The requested operation conflicts with the selected state.
    #[error("{0}")]
    Invalid(String),
    /// An attention layer could not complete a lifecycle operation.
    #[error(transparent)]
    Attention(#[from] KeyValueStateError),
    /// A page catalog operation failed.
    #[error(transparent)]
    Cache(#[from] CacheResidencyError),
    /// The named stream bindings do not match the layout.
    #[error(transparent)]
    StreamBinding(#[from] AppendStreamError),
    /// A named lane could not complete a lifecycle operation.
    #[error(transparent)]
    Streams(#[from] LayerAppendStreamError),
    /// State geometry or component access failed.
    #[error(transparent)]
    Layout(#[from] StateError),
    /// Native storage work did not complete.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

fn invalid(detail: impl Into<String>) -> HybridStateError {
    HybridStateError::Invalid(detail.into())
}

/// One heterogeneous attention and fixed-state layer.
#[derive(Debug, Clone)]
pub struct HybridLayerState<M: HybridCacheStorage> {
    attention: Option<HybridAttentionState<M>>,
    fixed: BTreeMap<StateTensorRole, Option<M::Value>>,
    streams: LayerAppendStreams<M>,
    fixed_offset: i32,
}

impl<M: HybridCacheStorage> HybridLayerState<M> {
    pub(crate) fn resident_key_value(
        &mut self,
    ) -> Option<&mut crate::cache::ResidentKeyValueCache<M>> {
        match &mut self.attention {
            Some(HybridAttentionState::KeyValue(KeyValueLayerState::Device(cache))) => Some(cache),
            _ => None,
        }
    }

    /// Borrows two selected streams in one lane.
    pub fn append_stream_pair(
        &mut self,
        first: u32,
        second: u32,
        lane: u32,
    ) -> Result<(&mut LayerAppendStream<M>, &mut LayerAppendStream<M>), AppendStreamError> {
        self.streams.append_stream_pair(first, second, lane)
    }

    /// Borrows one selected stream lane.
    pub fn append_stream(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<&mut LayerAppendStream<M>, AppendStreamError> {
        self.streams.append_stream(slot, lane)
    }

    /// Advances a fixed-state layer after executing tokens.
    pub fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
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

    /// Replaces all fixed components with the given roles.
    pub fn replace_fixed_components(
        &mut self,
        values: Vec<(StateTensorRole, Option<M::Value>)>,
    ) -> Result<(), StateError> {
        crate::state::replace_fixed_components(
            self.fixed.iter_mut().map(|(role, slot)| (*role, slot)),
            values,
        )
    }

    /// Iterates over declared fixed roles and their retained tensor roots.
    pub fn fixed_components(&self) -> impl Iterator<Item = (StateTensorRole, Option<&M::Value>)> {
        self.fixed
            .iter()
            .map(|(&role, value)| (role, value.as_ref()))
    }

    /// Borrows one fixed component slot.
    pub fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<M::Value>, StateError> {
        self.fixed
            .get_mut(&role)
            .ok_or(StateError::UnknownComponent { role })
    }

    /// Returns the layer’s token frontier.
    pub fn position(&self) -> i32 {
        self.attention
            .as_ref()
            .map_or(self.fixed_offset, HybridAttentionState::offset)
    }

    /// Borrows the exact native roots retained by the layer.
    pub fn retained_values(&self) -> std::vec::IntoIter<&M::Value> {
        let mut retained = self
            .attention
            .as_ref()
            .map(HybridAttentionState::retained_values)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        retained.extend(self.fixed.values().filter_map(Option::as_ref));
        retained.extend(self.streams.retained_values());
        retained.into_iter()
    }

    fn deep_clone_state(&self) -> Result<Self, HybridStateError> {
        let attention = self
            .attention
            .as_ref()
            .map(HybridAttentionState::deep_clone_state)
            .transpose()
            .map_err(HybridStateError::from)?;
        Ok(Self {
            attention,
            fixed: self.fixed.clone(),
            streams: self.streams.clone(),
            fixed_offset: self.fixed_offset,
        })
    }

    fn allocate(
        layer: usize,
        request: &eredu_core::cache::mechanisms::LayerStateAllocation,
        manager: Option<&CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, HybridStateError> {
        use eredu_core::cache::mechanisms::AttentionStorageKind;
        let attention = match request.attention {
            None => None,
            Some((policy, false)) => Some(match policy {
                AttentionStorageKind::KeyValue { window, key_only } => {
                    HybridAttentionState::KeyValue(KeyValueLayerState::resident(window, key_only))
                }
                AttentionStorageKind::CompressedLatent => {
                    HybridAttentionState::Compressed(CompressedLatentCache::new())
                }
            }),
            Some((policy, true)) => {
                let manager = manager.ok_or_else(|| {
                    invalid(format!(
                        "paged hybrid attention has no residency manager at layer {layer}"
                    ))
                })?;
                Some(match policy {
                    AttentionStorageKind::KeyValue { window, key_only } => {
                        HybridAttentionState::KeyValue(
                            KeyValueLayerState::paged(
                                manager.clone(),
                                layer,
                                window,
                                key_only,
                                rank,
                            )
                            .map_err(HybridStateError::from)?,
                        )
                    }
                    AttentionStorageKind::CompressedLatent => HybridAttentionState::Compressed(
                        CompressedLatentCache::new_paged(manager.clone(), layer, rank)
                            .map_err(HybridStateError::from)?,
                    ),
                })
            }
        };
        Ok(Self {
            attention,
            fixed: request.fixed.iter().map(|&role| (role, None)).collect(),
            fixed_offset: 0,
            streams: LayerAppendStreams::new(layer, &request.streams, manager, rank)
                .map_err(HybridStateError::from)?,
        })
    }

    /// Clears attention and fixed components while retaining their policies.
    pub fn clear(&mut self) -> Result<(), HybridStateError> {
        if let Some(attention) = &mut self.attention {
            attention.clear().map_err(HybridStateError::from)?;
        }
        for component in self.fixed.values_mut() {
            *component = None;
        }
        self.streams.clear().map_err(HybridStateError::from)?;
        self.fixed_offset = 0;
        Ok(())
    }
}

impl<B, M> RuntimeLayerState<B> for HybridLayerState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    type RetainedValues<'a> = std::vec::IntoIter<&'a M::Value>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_values()
    }
}

impl<B, M> RuntimeStateComponents<B> for HybridLayerState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    fn position(&self) -> i32 {
        self.position()
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<M::Value>, StateError> {
        self.fixed_component(role)
    }

    fn replace_fixed_components(
        &mut self,
        values: Vec<(eredu_core::cache::StateTensorRole, Option<M::Value>)>,
    ) -> Result<(), StateError> {
        self.replace_fixed_components(values)
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        self.advance_fixed(tokens)
    }
}

impl<M: HybridCacheStorage> AttentionCache<M::Value> for HybridLayerState<M> {
    fn uses_blockwise_attention(&self) -> bool {
        match self.attention.as_ref() {
            Some(HybridAttentionState::KeyValue(cache)) => {
                AttentionCache::uses_blockwise_attention(cache)
            }
            _ => false,
        }
    }
    fn offset(&self) -> i32 {
        self.position()
    }

    fn max_size(&self) -> Option<i32> {
        match self.attention.as_ref() {
            Some(HybridAttentionState::KeyValue(cache)) => AttentionCache::max_size(cache),
            _ => None,
        }
    }

    fn update_for_attention(
        &mut self,
        keys: M::Value,
        values: M::Value,
        stream: &M::Queue,
    ) -> Result<(M::Value, M::Value), ComputeError> {
        AttentionCache::update_for_attention(
            match self.attention.as_mut() {
                Some(HybridAttentionState::KeyValue(cache)) => cache,
                _ => {
                    return Err(ComputeError::backend(
                        "layer has no key/value attention cache",
                    ));
                }
            },
            keys,
            values,
            stream,
        )
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        AttentionCache::attention(
            match self.attention.as_mut() {
                Some(HybridAttentionState::KeyValue(cache)) => cache,
                _ => {
                    return Err(ComputeError::backend(
                        "layer has no key/value attention cache",
                    ));
                }
            },
            request,
            stream,
        )
    }
    fn indexed_attention<N: eredu_nn::NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::IndexedAttentionInput<'_, M::Value>,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::KeyValue(cache)) => {
                cache.indexed_attention::<N>(request, stream)
            }
            _ => Err(ComputeError::backend(
                "layer has no key/value attention cache",
            )),
        }
    }
    fn relative_attention<N: eredu_nn::NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::RelativeAttentionInput<'_, M::Value>,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::KeyValue(cache)) => {
                cache.relative_attention::<N>(request, stream)
            }
            _ => Err(ComputeError::backend(
                "layer has no key/value attention cache",
            )),
        }
    }
}

impl<M: HybridCacheStorage> eredu_nn::ReadOnlyAttentionCache<M::Value> for HybridLayerState<M> {
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        stream: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::KeyValue(cache)) => {
                eredu_nn::ReadOnlyAttentionCache::attention_with_transient(cache, request, stream)
            }
            _ => Err(ComputeError::backend(
                "layer has no key/value attention cache",
            )),
        }
    }
}

impl<M: HybridCacheStorage> CompressedAttentionCache<M::Value> for HybridLayerState<M> {
    type Checkpoint = CompressedLatentCache<M>;
    type History = CompressedLatentCache<M>;

    fn history(&self) -> Option<&Self::History> {
        match self.attention.as_ref() {
            Some(HybridAttentionState::Compressed(cache)) => Some(cache),
            _ => None,
        }
    }

    fn offset(&self) -> i32 {
        self.position()
    }

    fn is_paged(&self) -> bool {
        matches!(
            self.attention.as_ref(),
            Some(HybridAttentionState::Compressed(cache)) if cache.is_paged()
        )
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<M::Value>,
        context: &M::Queue,
    ) -> Result<CompressedAttentionView<M::Value>, ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::Compressed(cache)) => cache.append(state, context),
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        match self.attention.as_ref() {
            Some(HybridAttentionState::Compressed(cache)) => cache.checkpoint(),
            _ => panic!("layer has no compressed-latent attention cache"),
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &M::Queue,
    ) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::Compressed(cache)) => cache.restore(checkpoint, context),
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn finalize(&mut self) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::Compressed(cache)) => {
                CompressedAttentionCache::finalize(cache)
            }
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        match self.attention.as_mut() {
            Some(HybridAttentionState::Compressed(cache)) => CompressedAttentionCache::clear(cache),
            _ => Err(ComputeError::backend(
                "layer has no compressed-latent attention cache",
            )),
        }
    }
}

/// State realization for a schedule mixing attention and fixed components.
#[derive(Debug, Clone)]
pub struct HybridState<M: HybridCacheStorage> {
    layout: Option<StateLayout>,
    global_layer_start: usize,
    layers: Vec<HybridLayerState<M>>,
    manager: Option<CacheResidencyManager<M>>,
}

impl<M: HybridCacheStorage> SnapshotValue for HybridState<M>
where
    M::Value: SnapshotValue,
{
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        if !self.has_immutable_roots() {
            return Err(BackendFailure::from_error(invalid(
                "hybrid value snapshots require immutable device-local roots",
            )));
        }
        let overflow = || BackendFailure::from_error(ExecutionControlError::Overflow);
        let mut metadata = std::mem::size_of::<Self>() as u64;
        if let Some(layout) = &self.layout {
            metadata = metadata
                .checked_add(layout.logical_metadata_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        for layer in &self.layers {
            metadata = metadata
                .checked_add(std::mem::size_of::<HybridLayerState<M>>() as u64)
                .and_then(|bytes| bytes.checked_add((layer.fixed.len() as u64).checked_mul(256)?))
                .ok_or_else(overflow)?;
        }
        let retained = metadata
            .checked_add(
                self.isolated_snapshot_auxiliary_bytes()
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        let retained_bytes =
            self.retained_arrays()
                .into_iter()
                .try_fold(retained, |bytes, array| {
                    bytes
                        .checked_add(
                            M::borrow_value(array)
                                .estimate_snapshot_value()?
                                .retained_bytes,
                        )
                        .ok_or_else(overflow)
                })?;
        Ok(SnapshotEstimate {
            retained_bytes,
            copy_bytes: metadata,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotRetention) -> Result<Self, BackendFailure> {
        self.estimate_snapshot_value()?;
        // Device append caches detach reserved suffixes before a divergent write.
        // Fixed components are replaced wholesale. No native copy is submitted.
        self.deep_clone_state().map_err(BackendFailure::from_error)
    }
}

impl<M: HybridCacheStorage> HybridState<M> {
    /// Borrows a physical layer by ordinal.
    pub fn layer(&mut self, layer: usize) -> Result<&mut HybridLayerState<M>, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }

    /// Borrows the layout when this owner has state.
    pub fn optional_layout(&self) -> Option<&StateLayout> {
        self.layout.as_ref()
    }

    /// Returns the installed layout. Panics on a stateless owner.
    pub fn layout(&self) -> &StateLayout {
        self.layout
            .as_ref()
            .expect("layout requires stateful hybrid storage")
    }

    /// Explicit state-free owner; no sentinel layers or native cache allocation.
    pub fn stateless() -> Self {
        Self {
            layout: None,
            global_layer_start: 0,
            layers: Vec::new(),
            manager: None,
        }
    }
    /// Checks that two owners have the same installed layout and storage classes.
    pub fn has_same_snapshot_layout(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.global_layer_start == other.global_layer_start
            && self.layers.len() == other.layers.len()
            && self.layers.iter().zip(&other.layers).all(|(a, b)| {
                std::mem::discriminant(&a.attention) == std::mem::discriminant(&b.attention)
                    && match (&a.attention, &b.attention) {
                        (
                            Some(HybridAttentionState::KeyValue(a)),
                            Some(HybridAttentionState::KeyValue(b)),
                        ) => std::mem::discriminant(a) == std::mem::discriminant(b),
                        (
                            Some(HybridAttentionState::Compressed(_)),
                            Some(HybridAttentionState::Compressed(_)),
                        )
                        | (None, None) => true,
                        _ => false,
                    }
            })
    }

    /// Bounds retained positions after an admitted continuation.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.layers
            .iter()
            .try_fold(0, |bound, layer| match &layer.attention {
                None | Some(HybridAttentionState::KeyValue(KeyValueLayerState::Stateless)) => {
                    Some(bound)
                }
                Some(HybridAttentionState::KeyValue(KeyValueLayerState::Device(cache))) => {
                    Some(bound.max(cache.continuation_capacity_bound(additional)?))
                }
                Some(HybridAttentionState::KeyValue(KeyValueLayerState::Paged(cache))) => Some(
                    bound.max(
                        u64::try_from(cache.offset())
                            .ok()?
                            .checked_add(additional)?,
                    ),
                ),
                Some(HybridAttentionState::Compressed(cache)) => {
                    Some(bound.max(cache.continuation_capacity_bound(additional)?))
                }
            })
    }

    /// Reports whether every layer can fork a coherent page catalog.
    pub fn supports_isolated_snapshot(&self) -> bool {
        self.layers.iter().all(|layer| {
            layer.streams.consistent_manager(self.manager.as_ref())
                && match &layer.attention {
                    None
                    | Some(HybridAttentionState::KeyValue(
                        KeyValueLayerState::Stateless | KeyValueLayerState::Device(_),
                    )) => true,
                    Some(HybridAttentionState::KeyValue(KeyValueLayerState::Paged(cache))) => {
                        self.manager.as_ref().is_some_and(|manager| {
                            manager.session_id() == cache.manager().session_id()
                        })
                    }
                    Some(HybridAttentionState::Compressed(cache)) => {
                        cache.residency_manager().is_none_or(|paging| {
                            self.manager
                                .as_ref()
                                .is_some_and(|manager| manager.session_id() == paging.session_id())
                        })
                    }
                }
        })
    }

    /// Reports live catalog and spare capacity bytes charged to a snapshot.
    pub fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        let manager = self
            .manager
            .as_ref()
            .map(|manager| manager.isolated_snapshot_bytes())
            .unwrap_or(Some(0))?;
        self.layers.iter().try_fold(manager, |bytes, layer| {
            let spare = match &layer.attention {
                Some(HybridAttentionState::KeyValue(KeyValueLayerState::Device(cache))) => {
                    cache.spare_capacity_bytes()?
                }
                _ => 0,
            };
            bytes
                .checked_add(layer.streams.metadata_bytes()?)?
                .checked_add(spare)
        })
    }

    /// Bounds auxiliary storage after continuing from a snapshot.
    pub fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        let paged = self
            .layers
            .iter()
            .map(|layer| layer.streams.paged_count() as u64)
            .sum::<u64>()
            + self
                .layers
                .iter()
                .filter(|layer| {
                    layer
                        .attention
                        .as_ref()
                        .and_then(HybridAttentionState::manager)
                        .is_some()
                })
                .count() as u64;
        let pages = additional
            .checked_add(1)?
            .checked_mul(paged)?
            .checked_mul(8192)?;
        self.layers.iter().try_fold(pages, |bytes, layer| {
            bytes.checked_add(layer.streams.snapshot_growth(additional)?)
        })
    }

    /// Reports whether future appends preserve the retained storage.
    pub fn has_immutable_roots(&self) -> bool {
        self.manager.is_none()
            && self.layers.iter().all(|layer| {
                layer.streams.consistent_manager(None)
                    && match &layer.attention {
                        None
                        | Some(HybridAttentionState::KeyValue(KeyValueLayerState::Stateless)) => {
                            true
                        }
                        Some(HybridAttentionState::KeyValue(KeyValueLayerState::Device(cache))) => {
                            cache.has_immutable_roots()
                        }
                        _ => false,
                    }
            })
    }

    /// Creates a candidate with independent page ownership.
    pub fn transaction_candidate(&self, stream: &M::Queue) -> Result<Self, HybridStateError> {
        self.copy_candidate(stream, false)
    }

    /// Deeply copies attention and every architecture-declared fixed tensor.
    /// Fixed-state handles must not inherit the transaction checkpoint's sharing.
    pub fn isolated_snapshot(&self, stream: &M::Queue) -> Result<Self, HybridStateError> {
        self.copy_candidate(stream, true)
    }

    fn copy_candidate(&self, stream: &M::Queue, isolated: bool) -> Result<Self, HybridStateError> {
        if !self.supports_isolated_snapshot() {
            return Err(invalid(
                "isolated snapshots require consistent attention paging managers",
            ));
        }
        let manager = self
            .manager
            .as_ref()
            .map(|manager| {
                (if isolated {
                    manager.isolated_snapshot(stream)
                } else {
                    manager.transaction_candidate(stream)
                })
                .map_err(|e| invalid(e.to_string()))
            })
            .transpose()?;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let attention = layer
                    .attention
                    .as_ref()
                    .map(|state| state.copy_candidate(manager.as_ref(), stream, isolated))
                    .transpose()
                    .map_err(HybridStateError::from)?;
                let fixed = if !isolated {
                    layer.fixed.clone()
                } else {
                    layer
                        .fixed
                        .iter()
                        .map(|(role, value)| {
                            Ok((
                                *role,
                                value
                                    .as_ref()
                                    .map(|value| {
                                        M::copy_tensor(M::storage_ref(value), stream)
                                            .map(M::into_value)
                                            .map_err(HybridStateError::from)
                                    })
                                    .transpose()?,
                            ))
                        })
                        .collect::<Result<_, HybridStateError>>()?
                };
                Ok(HybridLayerState {
                    streams: if isolated {
                        layer
                            .streams
                            .fork(manager.as_ref(), stream)
                            .map_err(HybridStateError::from)?
                    } else {
                        layer
                            .streams
                            .transaction_candidate(manager.as_ref(), stream)
                            .map_err(HybridStateError::from)?
                    },
                    attention,
                    fixed,
                    fixed_offset: layer.fixed_offset,
                })
            })
            .collect::<Result<_, HybridStateError>>()?;
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers,
            manager,
        })
    }

    /// Creates device-resident attention and fixed state from a neutral layout.
    pub fn device(
        layout: StateLayout,
        bindings: &[AppendStreamBinding],
    ) -> Result<Self, HybridStateError> {
        Self::device_with_global_layer_start(layout, 0, bindings)
    }

    /// Creates device-resident state addressed from an architecture-global layer.
    pub fn device_with_global_layer_start(
        layout: StateLayout,
        global_layer_start: usize,
        bindings: &[AppendStreamBinding],
    ) -> Result<Self, HybridStateError> {
        Self::allocate_uniform(layout, global_layer_start, bindings, None, None)
    }

    /// Creates paged attention with device-resident fixed components.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager<M>,
        rank: Option<CacheRankIdentity>,
        bindings: &[AppendStreamBinding],
    ) -> Result<Self, HybridStateError> {
        Self::paged_with_global_layer_start(layout, manager, rank, 0, bindings)
    }

    /// Creates paged state addressed from an architecture-global layer.
    pub fn paged_with_global_layer_start(
        layout: StateLayout,
        manager: CacheResidencyManager<M>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
        bindings: &[AppendStreamBinding],
    ) -> Result<Self, HybridStateError> {
        Self::allocate_uniform(layout, global_layer_start, bindings, Some(manager), rank)
    }

    fn allocate_uniform(
        layout: StateLayout,
        global_layer_start: usize,
        bindings: &[AppendStreamBinding],
        manager: Option<CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, HybridStateError> {
        crate::validate_append_stream_layout(&layout, bindings)?;
        let paged = manager.is_some();
        let backing = if paged {
            AppendStreamBacking::Paged
        } else {
            AppendStreamBacking::Resident
        };
        let mut requests = layout
            .layers()
            .iter()
            .map(|policy| {
                let attention = super::allocation::attention_storage_kind(policy)
                    .map_err(|error| invalid(error.to_string()))?;
                Ok(eredu_core::cache::mechanisms::LayerStateAllocation {
                    attention: attention.map(|kind| (kind, paged)),
                    fixed: policy
                        .fixed_state()
                        .iter()
                        .map(|tensor| tensor.role)
                        .collect(),
                    streams: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, HybridStateError>>()?;
        for binding in bindings {
            requests[binding.layer]
                .streams
                .push((binding.clone(), backing));
        }
        let layers = Self::realize_layers(&requests, manager.as_ref(), rank, global_layer_start)?;
        Ok(Self {
            layout: Some(layout),
            global_layer_start,
            layers,
            manager,
        })
    }

    /// Realizes the exact attention, fixed-state, and stream storage requests.
    pub fn allocate_layers(
        allocation: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, HybridStateError> {
        let layers = Self::realize_layers(
            allocation.layers(),
            manager.as_ref(),
            rank,
            global_layer_start,
        )?;
        Ok(Self {
            layout: Some(allocation.layout().clone()),
            global_layer_start,
            layers,
            manager,
        })
    }

    fn realize_layers(
        requests: &[eredu_core::cache::mechanisms::LayerStateAllocation],
        manager: Option<&CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Vec<HybridLayerState<M>>, HybridStateError> {
        requests
            .iter()
            .enumerate()
            .map(|(layer, request)| {
                let global_layer = global_layer_start
                    .checked_add(layer)
                    .ok_or_else(|| invalid("hybrid state global layer index overflowed"))?;
                HybridLayerState::allocate(global_layer, request, manager, rank)
            })
            .collect()
    }

    /// Borrows ordinary layer state without changing native storage or frontiers.
    pub fn layers(&self) -> &[HybridLayerState<M>] {
        &self.layers
    }

    /// Mutably borrows the ordinary per-layer states used by neutral units.
    pub fn layers_mut(&mut self) -> &mut [HybridLayerState<M>] {
        &mut self.layers
    }

    /// Borrows the paging manager shared by this state's attention components.
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager<M>> {
        self.manager.as_ref().or_else(|| {
            self.layers
                .iter()
                .filter_map(|layer| layer.attention.as_ref())
                .find_map(HybridAttentionState::manager)
        })
    }

    /// Borrows every native array retained by the complete hybrid state.
    pub fn retained_arrays(&self) -> Vec<&M::Tensor> {
        self.layers
            .iter()
            .flat_map(HybridLayerState::retained_values)
            .map(M::storage_ref)
            .collect()
    }

    /// Returns the common absolute token frontier.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, HybridLayerState::position)
    }

    /// Clears every heterogeneous component.
    pub fn clear(&mut self) -> Result<(), HybridStateError> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    /// Captures an exact sequential transaction checkpoint.
    /// Immutable append roots remain shared; independently advancing branches
    /// use the model-wide fork operation to receive a separate paging manager.
    pub fn deep_clone_state(&self) -> Result<Self, HybridStateError> {
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers: self
                .layers
                .iter()
                .map(HybridLayerState::deep_clone_state)
                .collect::<Result<_, _>>()?,
            manager: self.manager.clone(),
        })
    }

    /// Forks the attention and named streams for a prediction target.
    pub fn fork_prediction_target_state(
        &self,
        stream: &M::Queue,
    ) -> Result<Self, HybridStateError> {
        let Some(manager) = self.manager.as_ref() else {
            return self.deep_clone_state();
        };
        let manager = manager.fork_session(stream)?;
        let mut fork = self.deep_clone_state()?;
        for layer in &mut fork.layers {
            layer.streams = layer
                .streams
                .fork(Some(&manager), stream)
                .map_err(HybridStateError::from)?;
            if let Some(attention) = &mut layer.attention {
                attention.rebind_paging_manager(manager.clone());
            }
        }
        fork.manager = Some(manager);
        Ok(fork)
    }

    /// Retains only verified full-attention K/V, without replaying target equations.
    pub fn retain_causal_prefix(
        &mut self,
        length: i32,
        stream: &M::Queue,
    ) -> Result<bool, HybridStateError> {
        // Stateless media owners retain no decoder positions and therefore do
        // not constrain another pipeline owner's accepted causal frontier.
        if self.layout.as_ref().is_none_or(|layout| {
            layout
                .layers()
                .iter()
                .all(|policy| matches!(policy, LayerCachePolicy::NoState))
        }) {
            return Ok(true);
        }
        if !self.layout().layers().iter().all(|policy| {
            policy.fixed_state().is_empty()
                && policy.append_streams().is_empty()
                && match policy {
                    LayerCachePolicy::NoState => true,
                    LayerCachePolicy::KeyValue { attention, .. } => attention
                        .sliding_window_i32()
                        .is_ok_and(|window| window.is_none()),
                    _ => false,
                }
        }) {
            return Ok(false);
        }
        if length < 0 || length > self.offset() {
            return Err(invalid("causal prefix is outside the verified frontier"));
        }
        let saved = self.deep_clone_state()?;
        for layer in &mut self.layers {
            if let Some(HybridAttentionState::KeyValue(cache)) = &mut layer.attention {
                if let Err(error) = cache.truncate_prefix(length, stream) {
                    self.restore_checkpoint(&saved, stream)?;
                    return Err(HybridStateError::from(error));
                }
            }
        }
        for value in self.retained_arrays() {
            if let Err(error) = M::settle_tensors(&[value]) {
                self.restore_checkpoint(&saved, stream)?;
                return Err(HybridStateError::from(error));
            }
        }
        Ok(true)
    }

    /// Restores append-only and fixed state to an exact speculative checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &M::Queue,
    ) -> Result<(), HybridStateError> {
        if !self.has_same_snapshot_layout(checkpoint) {
            return Err(invalid(
                "hybrid state checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            match (&mut current.attention, &previous.attention) {
                (Some(current), Some(previous)) => {
                    current
                        .restore_checkpoint(previous, stream)
                        .map_err(HybridStateError::from)?;
                }
                (None, None) => {}
                _ => {
                    return Err(invalid("hybrid state checkpoint attention policy changed"));
                }
            }
            current
                .streams
                .restore(&previous.streams, stream)
                .map_err(HybridStateError::from)?;
            current.fixed.clone_from(&previous.fixed);
            current.fixed_offset = previous.fixed_offset;
        }
        Ok(())
    }

    /// Commits exactly one architecture-declared state segment.
    pub fn commit_segment_from(
        &mut self,
        source: &Self,
        segment: &str,
    ) -> Result<(), HybridStateError> {
        if self.layout != source.layout
            || self.global_layer_start != source.global_layer_start
            || self.layers.len() != source.layers.len()
        {
            return Err(invalid(
                "hybrid draft state layout does not match canonical state",
            ));
        }
        let range = self.segment_range(segment)?;
        self.layers[range.clone()].clone_from_slice(&source.layers[range]);
        Ok(())
    }

    /// Resolves one named architecture-owned state segment.
    pub fn segment_range(&self, segment: &str) -> Result<Range<usize>, HybridStateError> {
        let segment = StateSegmentId::new(segment).map_err(|error| invalid(error.to_string()))?;
        self.layout
            .as_ref()
            .and_then(|layout| layout.segment(&segment))
            .map(StateSegmentSpec::layers)
            .ok_or_else(|| invalid(format!("unknown hybrid state segment {segment:?}")))
    }

    /// Returns aggregate telemetry when this state contains paged attention.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, HybridStateError> {
        self.residency_manager()
            .map(CacheResidencyManager::report)
            .transpose()
            .map_err(|error| invalid(error.to_string()))
    }

    /// Restores all fixed components materialized from a validated manifest.
    pub fn restore_prompt_cache_state(
        &mut self,
        tensors: Vec<LoadedPromptCacheStateTensor<M::Tensor>>,
        layer_frontiers: &[i32],
    ) -> Result<(), HybridStateError> {
        if layer_frontiers.len() != self.layers.len() {
            return Err(invalid(format!(
                "prompt-cache supplied {} layer frontiers for {} hybrid state layers",
                layer_frontiers.len(),
                self.layers.len()
            )));
        }
        let mut retained = BTreeMap::new();
        for tensor in tensors {
            let owner = tensor.owner;
            let role = tensor.role;
            let StateTensorOwner::Layer(index) = owner;
            let declared = index
                .checked_sub(self.global_layer_start)
                .and_then(|index| self.layers.get(index))
                .is_some_and(|layer| layer.fixed.contains_key(&role));
            if !declared {
                return Err(invalid(format!(
                    "prompt cache contains undeclared fixed state {owner:?}/{role:?}"
                )));
            }
            if retained.insert((owner, role), tensor.array).is_some() {
                return Err(invalid(format!(
                    "prompt cache repeats fixed state {owner:?}/{role:?}"
                )));
            }
        }
        for (layer, (state, &frontier)) in self.layers.iter_mut().zip(layer_frontiers).enumerate() {
            for (role, slot) in &mut state.fixed {
                let owner = StateTensorOwner::Layer(self.global_layer_start + layer);
                *slot = retained.remove(&(owner, *role)).map(M::into_value);
            }
            if state.attention.is_none() && !state.fixed.is_empty() {
                state.fixed_offset = frontier;
            }
        }
        Ok(())
    }

    /// Restores one architecture-owned state range from a shared prompt catalog.
    pub fn restore_prompt_cache_state_range(
        &mut self,
        tensors: &mut BTreeMap<(StateTensorOwner, StateTensorRole), M::Tensor>,
        range: Range<usize>,
        processed_tokens: i32,
        layer_prefix_offsets: &[i32],
    ) -> Result<(), HybridStateError> {
        if range.end > self.layers.len() || layer_prefix_offsets.len() != range.len() {
            return Err(invalid(
                "prompt-cache state range does not match hybrid state layout",
            ));
        }
        for (relative, layer) in range.enumerate() {
            let state = &mut self.layers[layer];
            let policy = self
                .layout
                .as_ref()
                .ok_or_else(|| invalid("stateless owner has no prompt cache"))?
                .layers()
                .get(layer)
                .expect("validated hybrid state range is inside its layout");
            let frontier = processed_tokens
                .checked_add(layer_prefix_offsets[relative])
                .ok_or_else(|| invalid("prompt-cache layer frontier overflowed"))?;
            if frontier < 0 {
                return Err(invalid(format!(
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
                    return Err(invalid(format!(
                        "prompt cache is missing required fixed state {owner:?}/{role:?}"
                    )));
                }
                *slot = restored.map(M::into_value);
            }
            if state.attention.is_none() && !state.fixed.is_empty() {
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
    ) -> Result<Vec<PromptCacheStateArray<M::Tensor>>, HybridStateError> {
        if range.end > self.layers.len() || layer_prefix_offsets.len() != range.len() {
            return Err(invalid(
                "prompt-cache state range does not match hybrid state layout",
            ));
        }
        for (relative, layer) in range.clone().enumerate() {
            let state = &mut self.layers[layer];
            if state.attention.is_none() && state.fixed.is_empty() {
                continue;
            }
            let frontier = processed_tokens
                .checked_add(layer_prefix_offsets[relative])
                .ok_or_else(|| invalid("prompt-cache layer frontier overflowed"))?;
            if frontier < 0 || state.position() != frontier {
                return Err(invalid(format!(
                    "hybrid state layer {layer} is at {}, expected prefix frontier {frontier}",
                    state.position()
                )));
            }
            if let Some(attention) = &mut state.attention {
                attention.finalize().map_err(HybridStateError::from)?;
                state.streams.finalize().map_err(HybridStateError::from)?;
                if attention.manager().is_none() {
                    return Err(invalid(
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
                            array: M::storage_ref(array).clone(),
                        })
                    })
            })
            .collect())
    }

    /// Finalizes paged attention and exposes the native fixed-state tensors.
    pub fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
        layer_frontiers: &[i32],
    ) -> Result<CacheExport<M>, HybridStateError> {
        if layer_frontiers.len() != self.layers.len() {
            return Err(invalid(format!(
                "prompt-cache descriptor supplied {} layer frontiers for {} hybrid state layers",
                layer_frontiers.len(),
                self.layers.len()
            )));
        }
        for layer in &self.layers {
            layer
                .streams
                .validate_batch(descriptor.batch_size())
                .map_err(HybridStateError::from)?;
        }
        let mut manager = self.manager.clone();
        for (layer, (state, &layer_expected)) in
            self.layers.iter_mut().zip(layer_frontiers).enumerate()
        {
            if state.attention.is_none() && state.fixed.is_empty() {
                continue;
            }
            if state.position() != layer_expected {
                return Err(invalid(format!(
                    "hybrid state layer {layer} is at {}, expected prefix frontier {layer_expected}",
                    state.position(),
                )));
            }
            if let Some(attention) = &mut state.attention {
                attention.finalize().map_err(HybridStateError::from)?;
                state.streams.finalize().map_err(HybridStateError::from)?;
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
                        array: M::storage_ref(array).clone(),
                    })
                })
            })
            .collect::<Vec<_>>();
        manager
            .ok_or_else(|| invalid("cannot persist hybrid state without attention"))?
            .cache_export(descriptor, state_arrays)
            .map_err(|error| invalid(error.to_string()))
    }
}

impl<B, M> RuntimeState<B> for HybridState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    type RetainedValues<'a> = std::vec::IntoIter<&'a M::Value>;

    fn layout(&self) -> &StateLayout {
        self.layout()
    }

    fn optional_layout(&self) -> Option<&StateLayout> {
        self.optional_layout()
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: crate::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let layer = ordinal;
        self.layers
            .get(layer)
            .map(HybridLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl<B, M> LayerRuntimeState<B> for HybridState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    type LayerState = HybridLayerState<M>;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        self.layer(layer)
    }
}

impl<M: HybridCacheStorage> eredu_nn::RetainedTensorRoots<M::Value> for HybridState<M> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a M::Value)) {
        self.layers
            .iter()
            .flat_map(HybridLayerState::retained_values)
            .for_each(visitor);
    }
}

impl<B, M> crate::DecoderPipelineState<B> for HybridState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    fn allocate_pipeline_state(
        layout: StateLayout,
        first_global_layer: usize,
    ) -> Result<Self, eredu_core::BackendFailure> {
        Self::device_with_global_layer_start(layout, first_global_layer, &[])
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn fork_pipeline_state(&self, context: &M::Queue) -> Result<Self, eredu_core::BackendFailure> {
        self.fork_prediction_target_state(context)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn pipeline_offset(&self) -> i32 {
        self.offset()
    }
}

impl<B, M> RuntimeAppendStreams<B> for HybridLayerState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    type Stream = LayerAppendStream<M>;

    fn append_stream(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<&mut Self::Stream, AppendStreamError> {
        self.append_stream(slot, lane)
    }

    fn append_stream_pair(
        &mut self,
        first: u32,
        second: u32,
        lane: u32,
    ) -> Result<(&mut Self::Stream, &mut Self::Stream), AppendStreamError> {
        self.append_stream_pair(first, second, lane)
    }
}

impl<M: HybridCacheStorage> eredu_nn::AuxiliaryConvolutionState<M::Value> for HybridLayerState<M> {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<M::Value>, ComputeError> {
        self.fixed_component(StateTensorRole::Convolution { slot })
            .map_err(ComputeError::backend)
    }
}

impl<M: HybridCacheStorage> crate::text_session_driver::TextSnapshotStorage<M::Queue>
    for HybridState<M>
{
    fn snapshot_estimate(&self) -> Option<SnapshotEstimate> {
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
        let costs = crate::replicated_session::StateSnapshotMemoryCosts::for_tensors(
            M::snapshot_tensor_costs()?,
        );
        let tensors = self
            .retained_arrays()
            .into_iter()
            .try_fold(0u64, |bytes, tensor| {
                bytes.checked_add(costs.tensor_bytes(M::bytes(tensor), M::shape(tensor).len())?)
            })?;
        costs.estimate(
            layout,
            owner,
            tensors,
            self.isolated_snapshot_auxiliary_bytes()?,
        )
    }

    fn snapshot_growth(&self, additional: u64) -> Option<u64> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        let Some(layout) = self.optional_layout() else {
            return Some(0);
        };
        let frontier = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        let capacity = frontier.max(self.continuation_capacity_bound(additional)?);
        i32::try_from(capacity).ok()?;
        let costs = crate::replicated_session::StateSnapshotMemoryCosts::for_tensors(
            M::snapshot_tensor_costs()?,
        );
        costs.growth(
            layout,
            capacity,
            self.isolated_snapshot_auxiliary_growth(additional)?,
        )
    }

    fn snapshot_copy(&self, queue: &M::Queue) -> Result<Self, BackendFailure> {
        self.isolated_snapshot(queue)
            .map_err(BackendFailure::from_error)
    }
}

impl<B, M> crate::PredictionModelState<B> for HybridState<M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + NeuralBackend<Tensor = M::Value>,
    M: HybridCacheStorage,
{
    type LayerState = HybridLayerState<M>;

    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.layers_mut()
    }
}
