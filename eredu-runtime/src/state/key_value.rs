//! Model-wide key/value state realization and transactions.

use std::ops::{Deref, DerefMut};

use eredu_core::{
    cache::{
        mechanisms::{PagedKeyValueStorage, ResidentKeyValueStorage},
        CacheRankIdentity, LayerCachePolicy, PromptCacheDescriptor,
    },
    scheduler::SemanticStateTransaction,
    BackendFailure,
};
use eredu_nn::{
    AttentionCache, AttentionRequest, Error as ComputeError, NeuralBackend, ReadOnlyAttentionCache,
};

use crate::{
    cache::{
        live::{CacheExport, CacheResidencyError, CacheResidencyManager, CacheStreamCheckpoint},
        PagedKeyValueCache, ResidentKeyValueCache,
    },
    CacheResidencyReport, LayerRuntimeState, ResettableRuntimeLayerState, ResettableRuntimeState,
    RuntimeLayerState, RuntimeState, StateError, StateLayout, StateSegmentId, StateSegmentSpec,
};

type RetainedValues<'a, M> = std::iter::Flatten<
    std::array::IntoIter<
        Option<&'a <M as eredu_core::cache::mechanisms::CacheTensorStorage>::Value>,
        2,
    >,
>;

/// Failures during portable key/value state assembly and lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum KeyValueStateError {
    /// The layout or requested transaction conflicts with retained state.
    #[error("{0}")]
    InvalidState(String),
    /// A page operation failed.
    #[error(transparent)]
    Cache(#[from] CacheResidencyError),
    /// The backend could not complete a storage operation.
    #[error(transparent)]
    Native(#[from] BackendFailure),
}

pub(super) fn invalid_state(message: impl Into<String>) -> KeyValueStateError {
    KeyValueStateError::InvalidState(message.into())
}

/// Key/value storage for one physical invocation.
#[derive(Debug, Clone)]
pub enum KeyValueLayerState<M: ResidentKeyValueStorage + PagedKeyValueStorage> {
    /// A physical invocation with no mutable attention state.
    Stateless,
    /// Contiguous execution-device keys and values.
    Device(ResidentKeyValueCache<M>),
    /// Block-addressable keys and values managed by finite residency budgets.
    Paged(PagedKeyValueCache<M>),
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> KeyValueLayerState<M> {
    /// Creates append-only key/value storage on the execution device.
    pub fn resident(window: Option<i32>, key_only: bool) -> Self {
        Self::Device(match (window, key_only) {
            (Some(window), true) => {
                ResidentKeyValueCache::new_key_only_append_only_for_sliding_attention(window)
            }
            (Some(window), false) => {
                ResidentKeyValueCache::new_append_only_for_sliding_attention(window)
            }
            (None, true) => ResidentKeyValueCache::new_key_only_append_only(),
            (None, false) => ResidentKeyValueCache::new_append_only(),
        })
    }

    /// Opens an exact key/value stream in the selected paging catalog.
    pub fn paged(
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        window: Option<i32>,
        key_only: bool,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, CacheResidencyError> {
        if key_only {
            PagedKeyValueCache::new_key_only_with_layout(manager, global_layer, window, 0, rank)
        } else {
            PagedKeyValueCache::new_with_layout(manager, global_layer, window, 0, rank)
        }
        .map(Self::Paged)
    }

    pub(super) fn copy_candidate(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        queue: &M::Queue,
        isolated: bool,
    ) -> Result<Self, KeyValueStateError> {
        match self {
            Self::Stateless => Ok(Self::Stateless),
            Self::Device(cache) if !isolated => Ok(Self::Device(cache.checkpoint_clone_state()?)),
            Self::Device(cache) => Ok(Self::Device(cache.isolated_snapshot(queue)?)),
            Self::Paged(cache) => {
                let manager = manager.ok_or_else(|| {
                    invalid_state("copied paged attention requires a paging catalog")
                })?;
                let mut copy = cache.deep_clone_state(queue)?;
                copy.rebind_paging_manager(manager.clone());
                Ok(Self::Paged(copy))
            }
        }
    }

    /// Borrows the retained roots of the selected storage.
    pub fn retained_values(&self) -> RetainedValues<'_, M> {
        let mut values = [None, None];
        match self {
            Self::Stateless => {}
            Self::Device(cache) => {
                for (slot, tensor) in values.iter_mut().zip(cache.arrays()) {
                    *slot = Some(M::borrow_value(tensor));
                }
            }
            Self::Paged(cache) => {
                for (slot, tensor) in values.iter_mut().zip(cache.arrays()) {
                    *slot = Some(M::borrow_value(tensor));
                }
            }
        }
        values.into_iter().flatten()
    }

    /// Truncates the selected cache to the exact token frontier.
    pub fn truncate_prefix(
        &mut self,
        length: i32,
        stream: &M::Queue,
    ) -> Result<(), KeyValueStateError> {
        match self {
            Self::Stateless => Ok(()),
            Self::Device(cache) => cache
                .truncate(length, stream)
                .map_err(KeyValueStateError::from),
            Self::Paged(cache) => cache
                .truncate(i64::from(length), stream)
                .map_err(KeyValueStateError::from),
        }
    }

    /// Clears retained keys and values without changing storage placement.
    pub fn clear(&mut self) -> Result<(), KeyValueStateError> {
        match self {
            Self::Stateless => Ok(()),
            Self::Device(cache) => {
                cache.clear();
                Ok(())
            }
            Self::Paged(cache) => cache.clear().map_err(KeyValueStateError::from),
        }
    }

    pub(super) fn deep_clone_state(&self) -> Result<Self, KeyValueStateError> {
        match self {
            Self::Stateless => Ok(Self::Stateless),
            Self::Device(cache) => cache
                .checkpoint_clone_state()
                .map(Self::Device)
                .map_err(KeyValueStateError::from),
            Self::Paged(cache) => cache
                .checkpoint_clone_state()
                .map(Self::Paged)
                .map_err(KeyValueStateError::from),
        }
    }

    pub(super) fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &M::Queue,
    ) -> Result<(), KeyValueStateError> {
        match (self, checkpoint) {
            (Self::Stateless, Self::Stateless) => Ok(()),
            (Self::Device(current), Self::Device(previous)) => {
                *current = previous
                    .checkpoint_clone_state()
                    .map_err(KeyValueStateError::from)?;
                Ok(())
            }
            (Self::Paged(current), Self::Paged(previous)) => current
                .restore_checkpoint(previous, stream)
                .map_err(KeyValueStateError::from),
            _ => Err(invalid_state("key/value checkpoint representation changed")),
        }
    }
}

impl<B, M> RuntimeLayerState<B> for KeyValueLayerState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    type RetainedValues<'a> = RetainedValues<'a, M>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_values()
    }
}

impl<B, M> ResettableRuntimeLayerState<B> for KeyValueLayerState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    fn reset(&mut self) -> Result<(), StateError> {
        self.clear()
            .map_err(|error| StateError::ResetFailed(error.to_string()))
    }
}

/// Model-wide key/value state created from a selected portable layout.
#[derive(Debug, Clone)]
pub struct KeyValueState<M: ResidentKeyValueStorage + PagedKeyValueStorage> {
    layout: StateLayout,
    global_layer_start: usize,
    layers: Vec<KeyValueLayerState<M>>,
    paged_transaction_branch: bool,
}

/// Unpublished branch of a model-wide key/value state.
///
/// The private wrapper prevents callers from satisfying a semantic transaction
/// with the public shallow [`Clone`] implementation. The branch records the
/// paged rollback frontier and checkpoints resident state.
#[derive(Debug)]
pub struct KeyValueTransactionBranch<M: ResidentKeyValueStorage + PagedKeyValueStorage> {
    state: KeyValueState<M>,
    paged_rollback: Vec<Option<CacheStreamCheckpoint>>,
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> Deref for KeyValueTransactionBranch<M> {
    type Target = KeyValueState<M>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> DerefMut for KeyValueTransactionBranch<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> KeyValueState<M> {
    /// Returns the installed layer and segment layout.
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Borrows one mutable physical layer.
    pub fn layer(&mut self, layer: usize) -> Result<&mut KeyValueLayerState<M>, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }

    /// Clears the layers of an admitted state segment.
    pub fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError> {
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
                .any(|layer| matches!(layer, KeyValueLayerState::Paged(_)))
        {
            return Err(StateError::ResetFailed(
                "paged state segments cannot be reset inside a transaction branch without copy-on-write page ownership"
                    .into(),
            ));
        }
        for layer in &mut self.layers[range] {
            layer
                .clear()
                .map_err(|error| StateError::ResetFailed(error.to_string()))?;
        }
        Ok(())
    }

    /// Bounds token storage after an admitted continuation.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.layers.iter().try_fold(0, |bound, layer| match layer {
            KeyValueLayerState::Stateless => Some(bound),
            KeyValueLayerState::Device(cache) => {
                Some(bound.max(cache.continuation_capacity_bound(additional)?))
            }
            KeyValueLayerState::Paged(cache) => Some(
                bound.max(
                    u64::try_from(cache.offset())
                        .ok()?
                        .checked_add(additional)?,
                ),
            ),
        })
    }

    /// Creates contiguous execution-device state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<Self, KeyValueStateError> {
        Self::device_with_global_layer_start(layout, 0)
    }

    /// Creates contiguous state addressed from an architecture-global layer.
    pub fn device_with_global_layer_start(
        layout: StateLayout,
        global_layer_start: usize,
    ) -> Result<Self, KeyValueStateError> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                if *policy == LayerCachePolicy::NoState {
                    return Ok(KeyValueLayerState::Stateless);
                }
                let window = key_value_window(layer, policy)?;
                Ok(KeyValueLayerState::resident(window, false))
            })
            .collect::<Result<Vec<_>, KeyValueStateError>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
            paged_transaction_branch: false,
        })
    }

    /// Creates block-addressable state using one shared residency manager.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager<M>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, KeyValueStateError> {
        Self::paged_with_global_layer_start(layout, manager, rank, 0)
    }

    /// Creates block-addressable state addressed from an architecture-global layer.
    pub fn paged_with_global_layer_start(
        layout: StateLayout,
        manager: CacheResidencyManager<M>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, KeyValueStateError> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                if *policy == LayerCachePolicy::NoState {
                    return Ok(KeyValueLayerState::Stateless);
                }
                let window = key_value_window(layer, policy)?;
                let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                    invalid_state("key/value state global layer index overflowed")
                })?;
                KeyValueLayerState::paged(manager.clone(), global_layer, window, false, rank)
                    .map_err(KeyValueStateError::from)
            })
            .collect::<Result<Vec<_>, KeyValueStateError>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
            paged_transaction_branch: false,
        })
    }

    /// Realizes the exact key/value storage requested for each layer.
    pub fn allocate_layers(
        allocation: &crate::state::StateAllocation<'_>,
        manager: Option<CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, KeyValueStateError> {
        use eredu_core::cache::mechanisms::AttentionStorageKind;
        let layers = allocation
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, request)| {
                let window = match request.attention {
                    None if request.fixed.is_empty() && request.streams.is_empty() => {
                        return Ok(KeyValueLayerState::Stateless)
                    }
                    Some((
                        AttentionStorageKind::KeyValue {
                            window,
                            key_only: false,
                        },
                        _,
                    )) if request.fixed.is_empty() && request.streams.is_empty() => window,
                    _ => {
                        return Err(invalid_state(format!(
                            "expected ordinary key/value storage at layer {layer}"
                        )))
                    }
                };
                if matches!(request.attention, Some((_, true))) {
                    let manager = manager.as_ref().ok_or_else(|| {
                        invalid_state(format!(
                            "paged key/value state has no residency manager at layer {layer}"
                        ))
                    })?;
                    let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                        invalid_state("key/value state global layer index overflowed")
                    })?;
                    KeyValueLayerState::paged(manager.clone(), global_layer, window, false, rank)
                        .map_err(KeyValueStateError::from)
                } else {
                    Ok(KeyValueLayerState::resident(window, false))
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            layout: allocation.layout().clone(),
            global_layer_start,
            layers,
            paged_transaction_branch: false,
        })
    }

    /// Returns the common absolute token offset, or zero for an empty state.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, AttentionCache::offset)
    }

    /// Retains a full-attention causal prefix atomically. Sliding histories use replay.
    pub fn retain_causal_prefix(
        &mut self,
        length: i32,
        stream: &M::Queue,
    ) -> Result<bool, KeyValueStateError> {
        if !self.layout.layers().iter().all(|policy| match policy {
            LayerCachePolicy::NoState => true,
            LayerCachePolicy::KeyValue { attention, .. } => attention
                .sliding_window_i32()
                .is_ok_and(|window| window.is_none()),
            _ => false,
        }) {
            return Ok(false);
        }
        if length < 0 || length > self.offset() {
            return Err(invalid_state(
                "causal prefix is outside the verified frontier",
            ));
        }
        let saved = self.deep_clone_state()?;
        for layer in &mut self.layers {
            if let Err(error) = layer.truncate_prefix(length, stream) {
                self.restore_checkpoint(&saved, stream)?;
                return Err(error);
            }
        }
        for value in self.retained_arrays() {
            M::settle_tensors(&[value])?;
        }
        Ok(true)
    }

    /// Clears retained arrays without changing residency or attention windows.
    pub fn clear(&mut self) -> Result<(), KeyValueStateError> {
        for layer in &mut self.layers {
            layer.clear()?;
        }
        Ok(())
    }

    /// Borrows every native array retained by the complete model state.
    pub fn retained_arrays(&self) -> Vec<&M::Tensor> {
        self.layers
            .iter()
            .flat_map(KeyValueLayerState::retained_values)
            .map(M::storage_ref)
            .collect()
    }

    /// Creates an independently advanceable speculative fork.
    pub fn deep_clone_state(&self) -> Result<Self, KeyValueStateError> {
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers: self
                .layers
                .iter()
                .map(KeyValueLayerState::deep_clone_state)
                .collect::<Result<_, _>>()?,
            paged_transaction_branch: self.paged_transaction_branch,
        })
    }

    /// Reports whether this state can fork a single coherent page catalog.
    pub fn supports_isolated_snapshot(&self) -> bool {
        let mut manager = None;
        self.layers.iter().all(|layer| match layer {
            KeyValueLayerState::Stateless | KeyValueLayerState::Device(_) => true,
            KeyValueLayerState::Paged(cache) => {
                let id = cache.manager().session_id();
                *manager.get_or_insert(id) == id
            }
        })
    }

    /// Reports the bytes charged for backing and catalog metadata.
    pub fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        let paged = self
            .layers
            .iter()
            .find_map(|layer| match layer {
                KeyValueLayerState::Paged(cache) => Some(cache.manager().isolated_snapshot_bytes()),
                _ => None,
            })
            .unwrap_or(Some(0))?;
        self.layers
            .iter()
            .try_fold(paged, |bytes, layer| match layer {
                KeyValueLayerState::Device(cache) => {
                    bytes.checked_add(cache.spare_capacity_bytes()?)
                }
                _ => Some(bytes),
            })
    }

    /// Bounds additional paging metadata after advancement.
    pub fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        // At most one new block per token per layer. This conservative catalog
        // bound also covers partial tails becoming sealed during advancement.
        let paged = self
            .layers
            .iter()
            .filter(|layer| matches!(layer, KeyValueLayerState::Paged(_)))
            .count() as u64;
        additional
            .checked_add(1)?
            .checked_mul(paged)?
            .checked_mul(8192)
    }

    /// Reports whether later operations preserve all current device roots.
    pub fn has_immutable_roots(&self) -> bool {
        self.layers.iter().all(|layer| match layer {
            KeyValueLayerState::Stateless => true,
            KeyValueLayerState::Device(cache) => cache.has_immutable_roots(),
            KeyValueLayerState::Paged(_) => false,
        })
    }

    /// Creates a transaction candidate with independent page ownership.
    pub fn transaction_candidate(&self, stream: &M::Queue) -> Result<Self, KeyValueStateError> {
        self.copy_candidate(stream, false)
    }

    /// Copies device arrays, mutable tails and sealed blocks with independent
    /// namespace ownership. Every paged copy remains in the same finite pool.
    pub fn isolated_snapshot(&self, stream: &M::Queue) -> Result<Self, KeyValueStateError> {
        self.copy_candidate(stream, true)
    }

    fn copy_candidate(
        &self,
        stream: &M::Queue,
        isolated: bool,
    ) -> Result<Self, KeyValueStateError> {
        if !self.supports_isolated_snapshot() {
            return Err(invalid_state(
                "isolated state contains inconsistent paging managers",
            ));
        }
        let manager = self
            .layers
            .iter()
            .find_map(|layer| match layer {
                KeyValueLayerState::Paged(cache) => Some(cache.manager()),
                _ => None,
            })
            .map(|manager| {
                (if isolated {
                    manager.isolated_snapshot(stream)
                } else {
                    manager.transaction_candidate(stream)
                })
                .map_err(KeyValueStateError::from)
            })
            .transpose()?;
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.copy_candidate(manager.as_ref(), stream, isolated))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            paged_transaction_branch: self.paged_transaction_branch,
            layers,
        })
    }

    /// Forks the prediction target and its shared paging session.
    pub fn fork_prediction_target_state(
        &self,
        stream: &M::Queue,
    ) -> Result<Self, KeyValueStateError> {
        let manager = self.layers.iter().find_map(|layer| match layer {
            KeyValueLayerState::Stateless | KeyValueLayerState::Device(_) => None,
            KeyValueLayerState::Paged(cache) => Some(cache.manager()),
        });
        let Some(manager) = manager else {
            return self.deep_clone_state();
        };
        if self.layers.iter().any(|layer| match layer {
            KeyValueLayerState::Stateless | KeyValueLayerState::Device(_) => false,
            KeyValueLayerState::Paged(cache) => {
                cache.manager().session_id() != manager.session_id()
            }
        }) {
            return Err(invalid_state(
                "key/value state contains more than one paging session",
            ));
        }
        let manager = manager
            .fork_session(stream)
            .map_err(|error| invalid_state(error.to_string()))?;
        let mut fork = self.deep_clone_state()?;
        for layer in &mut fork.layers {
            if let KeyValueLayerState::Paged(cache) = layer {
                cache.rebind_paging_manager(manager.clone());
            }
        }
        Ok(fork)
    }

    /// Checks that a branch is eligible to publish into this state.
    pub fn has_same_transaction_identity(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.global_layer_start == other.global_layer_start
            && self.layers.len() == other.layers.len()
            && self
                .layers
                .iter()
                .zip(&other.layers)
                .all(|(canonical, branch)| match (canonical, branch) {
                    (KeyValueLayerState::Stateless, KeyValueLayerState::Stateless) => true,
                    (KeyValueLayerState::Device(_), KeyValueLayerState::Device(_)) => true,
                    (KeyValueLayerState::Paged(canonical), KeyValueLayerState::Paged(branch)) => {
                        canonical.has_same_transaction_identity(branch)
                    }
                    _ => false,
                })
    }

    /// Checks whether a snapshot can realize the same layer layout.
    pub fn has_same_snapshot_layout(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.global_layer_start == other.global_layer_start
            && self.layers.len() == other.layers.len()
            && self
                .layers
                .iter()
                .zip(&other.layers)
                .all(|(a, b)| std::mem::discriminant(a) == std::mem::discriminant(b))
    }

    /// Restores every append-only layer to an exact speculative checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &M::Queue,
    ) -> Result<(), KeyValueStateError> {
        if !self.has_same_transaction_identity(checkpoint) {
            return Err(invalid_state(
                "key/value state checkpoint layout does not match canonical state",
            ));
        }
        for (current, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            current.restore_checkpoint(previous, stream)?;
        }
        Ok(())
    }

    /// Returns aggregate telemetry when this is paged state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, KeyValueStateError> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                KeyValueLayerState::Paged(cache) => Some(cache.report()),
                KeyValueLayerState::Stateless | KeyValueLayerState::Device(_) => None,
            })
            .transpose()
            .map_err(KeyValueStateError::from)
    }

    /// Finalizes paged tails and exposes native storage for the portable publisher.
    pub fn cache_export(
        &mut self,
        descriptor: &PromptCacheDescriptor,
    ) -> Result<CacheExport<M>, KeyValueStateError> {
        let mut manager = None;
        for layer in &mut self.layers {
            if matches!(layer, KeyValueLayerState::Stateless) {
                continue;
            }
            let KeyValueLayerState::Paged(cache) = layer else {
                return Err(invalid_state(
                    "prompt-cache persistence requires explicitly paged state",
                ));
            };
            cache.finalize().map_err(KeyValueStateError::from)?;
            manager.get_or_insert_with(|| cache.manager().clone());
        }
        manager
            .ok_or_else(|| invalid_state("cannot persist empty paged state"))?
            .cache_export(descriptor, Vec::new())
            .map_err(|error| invalid_state(error.to_string()))
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> SemanticStateTransaction
    for KeyValueState<M>
{
    type Branch = KeyValueTransactionBranch<M>;
    type Error = KeyValueStateError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        let paged_rollback = self
            .layers
            .iter()
            .map(|layer| match layer {
                KeyValueLayerState::Stateless | KeyValueLayerState::Device(_) => Ok(None),
                KeyValueLayerState::Paged(cache) => cache
                    .transaction_checkpoint()
                    .map(Some)
                    .map_err(KeyValueStateError::from),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.deep_clone_state().map(|mut state| {
            state.paged_transaction_branch = paged_rollback.iter().any(Option::is_some);
            KeyValueTransactionBranch {
                state,
                paged_rollback,
            }
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        if !self.has_same_transaction_identity(&branch.state) {
            Self::discard_branch(branch)?;
            return Err(invalid_state(
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
            let (KeyValueLayerState::Paged(cache), Some(checkpoint)) = (layer, checkpoint) else {
                continue;
            };
            if let Err(error) = cache.rollback_transaction(checkpoint) {
                first_error.get_or_insert_with(|| KeyValueStateError::from(error));
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn permits_parallel_branches(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer,
                KeyValueLayerState::Stateless | KeyValueLayerState::Device(_)
            )
        })
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> eredu_nn::RetainedTensorRoots<M::Value>
    for KeyValueState<M>
{
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a M::Value)) {
        self.layers
            .iter()
            .flat_map(KeyValueLayerState::retained_values)
            .for_each(visitor);
    }
}

impl<B, M> RuntimeState<B> for KeyValueState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    type RetainedValues<'a> = RetainedValues<'a, M>;

    fn layout(&self) -> &StateLayout {
        self.layout()
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: crate::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let layer = ordinal;
        self.layers
            .get(layer)
            .map(KeyValueLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl<B, M> LayerRuntimeState<B> for KeyValueState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    type LayerState = KeyValueLayerState<M>;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        self.layer(layer)
    }
}

impl<B, M> ResettableRuntimeState<B> for KeyValueState<M>
where
    B: NeuralBackend<Tensor = M::Value>,
    M: ResidentKeyValueStorage + PagedKeyValueStorage,
{
    fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError> {
        self.reset_segment(segment)
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> AsRef<[KeyValueLayerState<M>]>
    for KeyValueState<M>
{
    fn as_ref(&self) -> &[KeyValueLayerState<M>] {
        &self.layers
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> AsMut<[KeyValueLayerState<M>]>
    for KeyValueState<M>
{
    fn as_mut(&mut self) -> &mut [KeyValueLayerState<M>] {
        &mut self.layers
    }
}

fn key_value_window(
    layer: usize,
    policy: &LayerCachePolicy,
) -> Result<Option<i32>, KeyValueStateError> {
    match policy {
        LayerCachePolicy::KeyValue { attention, .. } => attention
            .sliding_window_i32()
            .map_err(|error| invalid_state(error.to_string())),
        _ => Err(invalid_state(format!(
            "key/value state cannot realize non-key/value policy at layer {layer}: {policy:?}"
        ))),
    }
}

fn stateless_attention() -> ComputeError {
    ComputeError::backend("stateless invocation has no key/value attention state")
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> AttentionCache<M::Value>
    for KeyValueLayerState<M>
{
    fn memory_retained_positions(&self) -> Option<u64> {
        match self {
            Self::Stateless => None,
            Self::Device(cache) => cache.memory_retained_positions(),
            Self::Paged(cache) => cache.memory_retained_positions(),
        }
    }

    fn update_memory_contract(
        &self,
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => cache.update_memory_contract(invocation),
            Self::Paged(cache) => cache.update_memory_contract(invocation),
        }
    }

    fn uses_blockwise_attention(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Stateless => 0,
            Self::Device(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Stateless => None,
            Self::Device(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: M::Value,
        values: M::Value,
        queue: &M::Queue,
    ) -> Result<(M::Value, M::Value), ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => AttentionCache::update_for_attention(cache, keys, values, queue),
            Self::Paged(cache) => AttentionCache::update_for_attention(cache, keys, values, queue),
        }
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        queue: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => cache.attention(request, queue),
            Self::Paged(cache) => cache.attention(request, queue),
        }
    }

    fn indexed_attention<B: NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::IndexedAttentionInput<'_, M::Value>,
        queue: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => cache.indexed_attention::<B>(request, queue),
            Self::Paged(cache) => cache.indexed_attention::<B>(request, queue),
        }
    }

    fn relative_attention<B: NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: eredu_nn::RelativeAttentionInput<'_, M::Value>,
        queue: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => cache.relative_attention::<B>(request, queue),
            Self::Paged(cache) => cache.relative_attention::<B>(request, queue),
        }
    }
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage> ReadOnlyAttentionCache<M::Value>
    for KeyValueLayerState<M>
{
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        queue: &M::Queue,
    ) -> Result<M::Value, ComputeError> {
        match self {
            Self::Stateless => Err(stateless_attention()),
            Self::Device(cache) => cache.attention_with_transient(request, queue),
            Self::Paged(cache) => cache.attention_with_transient(request, queue),
        }
    }
}
