//! Model-wide key/value state realization and transactions.

use super::*;

mod original_paged_copy;
mod realtime_branch;
pub(crate) use realtime_branch::RealtimeKvBranchPlan;
mod prepared_copy;
pub(crate) use prepared_copy::ResidentKvPreparationError;
pub(in crate::backend::runtime::cache::state) use prepared_copy::{PagedWork, PreparedPagedStorageCopy};
pub(in crate::backend::runtime::cache::state) use prepared_copy::copy_resident_kv_layer_retained;
pub(crate) use prepared_copy::{
    DenseResidentKvPublishError, InitializedPagedDenseCopy, PreparedDenseResidentKvState,
    PreparedResidentKvCopy, PublishedDenseResidentKvState, ResidentKvCopyError,
    SavedResidentKvCopy,
};
mod workspace;
pub(crate) use workspace::CompleteStateProjectionFailure;
pub(in crate::backend::runtime::cache::state) use workspace::project_layers;
pub(in crate::backend::runtime::cache::state) use workspace::{SourceCounts, project_complete};

#[cfg(test)]
mod slot_tests;

/// One concrete MLX key/value layer state selected by runtime residency policy.
#[derive(Debug, Clone)]
pub enum MlxKeyValueLayerState {
    /// A physical invocation with no mutable attention state.
    Stateless,
    /// Contiguous execution-device keys and values.
    Device(ConcatKeyValueCache),
    /// Block-addressable keys and values managed by finite residency budgets.
    Paged(PagedKeyValueCache),
}

impl MlxKeyValueLayerState {
    pub(super) fn retained_owner_slot_counts(&self) -> NativeStateSlotCounts {
        match self {
            Self::Stateless => NativeStateSlotCounts::default(),
            Self::Device(_) => NativeStateSlotCounts::arrays(2, 0),
            Self::Paged(_) => NativeStateSlotCounts::arrays(2, 1),
        }
    }

    pub(super) fn clear(&mut self) -> Result<(), Exception> {
        match self {
            Self::Stateless => Ok(()),
            Self::Device(cache) => {
                cache.clear();
                Ok(())
            }
            Self::Paged(cache) => cache.clear(),
        }
    }

    fn deep_clone_state(&self) -> Result<Self, Exception> {
        match self {
            Self::Stateless => Ok(Self::Stateless),
            Self::Device(cache) => cache.checkpoint_clone_state().map(Self::Device),
            Self::Paged(cache) => cache.checkpoint_clone_state().map(Self::Paged),
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
            (Self::Stateless, Self::Stateless) => Ok(()),
            (Self::Device(current), Self::Device(previous)) => {
                *current = previous.checkpoint_clone_state()?;
                Ok(())
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
    fn paged_relative_attention(
        &mut self,
        input: &eredu_nn::RelativeAttentionInput<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Stateless => Err(Exception::custom(
                "stateless invocation cannot execute relative attention",
            )),
            Self::Device(cache) => cache.paged_relative_attention(input, stream),
            Self::Paged(cache) => cache.paged_relative_attention(input, stream),
        }
    }
    fn offset(&self) -> i32 {
        match self {
            Self::Stateless => 0,
            Self::Device(cache) => KeyValueCache::offset(cache),
            Self::Paged(cache) => KeyValueCache::offset(cache),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Stateless => None,
            Self::Device(cache) => KeyValueCache::max_size(cache),
            Self::Paged(cache) => KeyValueCache::max_size(cache),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Stateless => Vec::new(),
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
        softcap: Option<f32>,
        arithmetic: eredu_nn::AttentionArithmetic,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Stateless => Err(Exception::custom(
                "stateless invocation cannot execute cached attention",
            )),
            Self::Device(cache) => {
                cache.paged_attention(queries, scale, mask, sinks, softcap, arithmetic, stream)
            }
            Self::Paged(cache) => {
                cache.paged_attention(queries, scale, mask, sinks, softcap, arithmetic, stream)
            }
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Stateless => Err(Exception::custom(
                "stateless invocation cannot acquire key/value state",
            )),
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
            Self::Stateless => Err(Exception::custom(
                "stateless invocation cannot acquire key/value state",
            )),
            Self::Device(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

impl RuntimeLayerState<MlxNeuralBackend> for MlxKeyValueLayerState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) {
        match self {
            Self::Stateless => {}
            Self::Device(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(cache, visitor)
            }
            Self::Paged(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(cache, visitor)
            }
        }
    }

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        match self {
            Self::Stateless => [None, None].into_iter().flatten(),
            Self::Device(cache) => RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache),
            Self::Paged(cache) => RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache),
        }
    }
}

impl ResettableRuntimeLayerState<MlxNeuralBackend> for MlxKeyValueLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.clear()
            .map_err(|error| StateError::ResetFailed(error.to_string()))
    }
}

/// Model-wide MLX key/value state created solely from a neutral layout.
#[derive(Debug)]
pub struct MlxKeyValueState {
    layout: eredu_runtime::SharedStateLayout,
    global_layer_start: usize,
    pub(super) layers: eredu_runtime::HostSlotTable<MlxKeyValueLayerState>,
    paged_transaction_branch: bool,
    inference_retention: eredu_runtime::working_memory::InferenceRetention,
}

impl Clone for MlxKeyValueState {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers: eredu_runtime::HostSlotTable::new(
                self.layers.slots().to_vec().into_boxed_slice(),
            ),
            paged_transaction_branch: self.paged_transaction_branch,
            inference_retention: self.inference_retention.clone(),
        }
    }
}

/// Unpublished, independently mutable resident MLX key/value state.
///
/// The private wrapper prevents callers from satisfying a semantic transaction
/// with the public shallow [`Clone`] implementation. It can only be created by
/// [`SemanticStateTransaction::branch`], which uses exact deep array clones.
#[derive(Debug)]
pub struct MlxKeyValueTransactionBranch {
    pub(super) state: MlxKeyValueState,
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

pub(super) fn selected_state_layers(
    selected: &SelectedStateRealization,
) -> Result<Vec<&[SelectedStateComponentRealization]>, Exception> {
    let layout = selected.layout();
    let mut cursor: usize = 0;
    let mut layers = Vec::with_capacity(layout.len());
    for layer in 0..layout.len() {
        let expected = layout.components(layer).ok_or_else(|| {
            Exception::custom(format!(
                "selected state layout has no component contract for layer {layer}"
            ))
        })?;
        let end = cursor
            .checked_add(expected.len())
            .ok_or_else(|| Exception::custom("selected state component count overflowed"))?;
        let realized = selected.components().get(cursor..end).ok_or_else(|| {
            Exception::custom(format!(
                "selected state omits declared components at layer {layer}"
            ))
        })?;
        for (component, expected) in realized.iter().zip(expected) {
            if component.layer() != layer || component.component() != expected {
                return Err(Exception::custom(format!(
                    "selected state component contract differs from the layout at layer {layer}"
                )));
            }
        }
        layers.push(realized);
        cursor = end;
    }
    if cursor != selected.components().len() {
        return Err(Exception::custom(format!(
            "selected state contains {} components beyond its layout",
            selected.components().len() - cursor
        )));
    }
    Ok(layers)
}

pub(super) fn common_selected_placement(
    layer: usize,
    kind: &str,
    components: &[SelectedStateComponentRealization],
) -> Result<Option<StateComponentPlacement>, Exception> {
    let Some(first) = components.first() else {
        return Ok(None);
    };
    let placement = first.placement();
    if components
        .iter()
        .any(|component| component.placement() != placement)
    {
        return Err(Exception::custom(format!(
            "MLX {kind} components select incompatible placements at layer {layer}"
        )));
    }
    Ok(Some(placement))
}

fn empty_resident_kv_layer(window: Option<i32>) -> MlxKeyValueLayerState {
    MlxKeyValueLayerState::Device(match window {
        Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
        None => ConcatKeyValueCache::new(),
    })
}

impl eredu_runtime::working_memory::ResidentTableResetState for MlxKeyValueState {
    type Layer = MlxKeyValueLayerState;
    type ResetPlan = original_reset::MlxPagedResetPlan;
    type ResetContext = original_reset::MlxPagedResetContext;
    fn resident_reset_plan(
        &self,
    ) -> Result<(Self::ResetPlan, usize), eredu_runtime::working_memory::WorkingMemoryError> {
        self.original_reset_plan()
    }
    fn prepare_resident_reset_context(
        &self,
        plan: &Self::ResetPlan,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Self::ResetContext, eredu_core::BackendFailure> {
        self.prepare_original_reset(plan, funding)
    }
    fn validate_resident_reset_placement(
        layer: &Self::Layer,
        _role: eredu_core::cache::StateComponentRole,
        placement: eredu_runtime::StateComponentPlacement,
    ) -> bool {
        matches!(
            (layer, placement),
            (
                MlxKeyValueLayerState::Paged(_),
                eredu_runtime::StateComponentPlacement::Paged
            ) | (
                MlxKeyValueLayerState::Device(_) | MlxKeyValueLayerState::Stateless,
                eredu_runtime::StateComponentPlacement::Device
            )
        )
    }
    fn empty_resident_reset_layer_prepared(
        context: &mut Self::ResetContext,
        source: &Self::Layer,
        policy: &LayerCachePolicy,
        child: Option<eredu_runtime::HostSlotTable<()>>,
    ) -> Result<Self::Layer, eredu_core::BackendFailure> {
        Self::empty_original_reset_layer(context, source, policy, child)
    }
    type Child = ();
    fn resident_reset_layers(&self) -> &eredu_runtime::HostSlotTable<Self::Layer> {
        &self.layers
    }
    fn resident_reset_layout(&self) -> &eredu_runtime::SharedStateLayout {
        &self.layout
    }
    fn resident_reset_global_start(&self) -> usize {
        self.global_layer_start
    }
    fn resident_fork_is_empty(&self) -> bool {
        !self.paged_transaction_branch
            && self.layers.slots().iter().all(|layer| match layer {
                MlxKeyValueLayerState::Stateless => true,
                MlxKeyValueLayerState::Device(cache) => cache.resident_fork_is_empty(),
                _ => false,
            })
    }
    fn validate_resident_reset_layer(layer: &Self::Layer, policy: &LayerCachePolicy) -> bool {
        match (layer, policy) {
            (MlxKeyValueLayerState::Stateless, LayerCachePolicy::NoState) => true,
            (
                MlxKeyValueLayerState::Device(cache),
                LayerCachePolicy::KeyValue { attention, .. },
            ) => attention
                .sliding_window_i32()
                .is_ok_and(|window| cache.matches_resident_reset_policy(window)),
            (MlxKeyValueLayerState::Paged(cache), LayerCachePolicy::KeyValue { attention, .. }) => {
                attention
                    .sliding_window_i32()
                    .is_ok_and(|window| cache.matches_original_reset_policy(window))
            }
            _ => false,
        }
    }
    fn empty_resident_reset_layer(policy: &LayerCachePolicy) -> Self::Layer {
        match policy {
            LayerCachePolicy::NoState => MlxKeyValueLayerState::Stateless,
            LayerCachePolicy::KeyValue { attention, .. } => empty_resident_kv_layer(
                attention.sliding_window_i32().expect("validated KV window"),
            ),
            _ => unreachable!("validated resident KV policy"),
        }
    }
    fn from_resident_reset(
        _context: &mut Self::ResetContext,
        layout: eredu_runtime::SharedStateLayout,
        global_layer_start: usize,
        layers: eredu_runtime::HostSlotTable<Self::Layer>,
    ) -> Self {
        Self {
            layout,
            global_layer_start,
            layers,
            paged_transaction_branch: false,
            inference_retention: Default::default(),
        }
    }
}

impl MlxKeyValueState {
    /// Includes absent future KV fields and the actual immutable host tables.
    /// Manager catalogs and external metadata are separate unknown domains.
    pub(crate) fn retained_owner_slot_counts(&self) -> Option<NativeStateSlotCounts> {
        self.layers.slots().iter().try_fold(
            NativeStateSlotCounts {
                layouts: 1,
                slot_tables: 1,
                ..Default::default()
            },
            |counts, layer| counts.checked_add(layer.retained_owner_slot_counts()),
        )
    }

    /// Prepares a borrowed resident decoder copy without acquiring or granting
    /// construction authority. Paged storage requires a separate closed plan.
    pub(crate) fn prepare_resident_copy(
        &self,
    ) -> Result<PreparedResidentKvCopy<'_>, ResidentKvCopyError> {
        PreparedResidentKvCopy::prepare(self)
    }

    /// Borrows the actual fixed layer-table extent and accounting identity.
    /// Nested native arrays and manager resources require separate inventory.
    /// Cloning this token grants neither slot access nor allocation permission.
    pub(crate) fn layer_slot_metadata(&self) -> &eredu_runtime::HostSlotMetadata {
        self.layers.metadata()
    }

    pub(crate) fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.layers
            .slots()
            .iter()
            .try_fold(0, |bound, layer| match layer {
                MlxKeyValueLayerState::Stateless => Some(bound),
                MlxKeyValueLayerState::Device(cache) => {
                    Some(bound.max(cache.continuation_capacity_bound(additional)?))
                }
                MlxKeyValueLayerState::Paged(cache) => Some(
                    bound.max(
                        u64::try_from(KeyValueCache::offset(cache))
                            .ok()?
                            .checked_add(additional)?,
                    ),
                ),
            })
    }

    /// Creates contiguous execution-device state for every declared layer.
    pub fn device(layout: StateLayout) -> Result<Self, Exception> {
        Self::device_with_global_layer_start(layout, 0)
    }

    /// Creates contiguous state addressed from an architecture-global layer.
    pub fn device_with_global_layer_start(
        layout: StateLayout,
        global_layer_start: usize,
    ) -> Result<Self, Exception> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                if *policy == LayerCachePolicy::NoState {
                    return Ok(MlxKeyValueLayerState::Stateless);
                }
                let window = key_value_window(layer, policy)?;
                Ok(empty_resident_kv_layer(window))
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self {
            layout: eredu_runtime::SharedStateLayout::new(layout),
            global_layer_start,
            inference_retention: Default::default(),
            layers: eredu_runtime::HostSlotTable::new(layers.into_boxed_slice()),
            paged_transaction_branch: false,
        })
    }

    /// Creates block-addressable state using one shared residency manager.
    pub fn paged(
        layout: StateLayout,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::paged_with_global_layer_start(layout, manager, rank, 0)
    }

    /// Creates block-addressable state addressed from an architecture-global layer.
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
                if *policy == LayerCachePolicy::NoState {
                    return Ok(MlxKeyValueLayerState::Stateless);
                }
                let window = key_value_window(layer, policy)?;
                let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                    Exception::custom("key/value state global layer index overflowed")
                })?;
                PagedKeyValueCache::new_with_layout(manager.clone(), global_layer, window, 0, rank)
                    .map(MlxKeyValueLayerState::Paged)
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self {
            layout: eredu_runtime::SharedStateLayout::new(layout),
            global_layer_start,
            inference_retention: Default::default(),
            layers: eredu_runtime::HostSlotTable::new(layers.into_boxed_slice()),
            paged_transaction_branch: false,
        })
    }

    /// Creates the exact component placements selected before allocation.
    ///
    /// A paging manager is required when any selected component is paged. The
    /// rank identifies the local cache shard when one is present.
    pub fn from_selected(
        selected: &SelectedStateRealization,
        manager: Option<CacheResidencyManager>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::from_selected_with_global_layer_start(selected, manager, rank, 0)
    }

    /// Creates selected state addressed from an architecture-global layer.
    pub fn from_selected_with_global_layer_start(
        selected: &SelectedStateRealization,
        manager: Option<CacheResidencyManager>,
        rank: Option<CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Exception> {
        let selected_layers = selected_state_layers(selected)?;
        let layout = selected.layout().clone();
        let layers = layout
            .layers()
            .iter()
            .zip(selected_layers)
            .enumerate()
            .map(|(layer, (policy, components))| {
                if *policy == LayerCachePolicy::NoState {
                    if !components.is_empty() { return Err(Exception::custom("stateless invocation has selected state components")); }
                    return Ok(MlxKeyValueLayerState::Stateless);
                }
                let window = key_value_window(layer, policy)?;
                match common_selected_placement(layer, "key/value", components)? {
                    Some(StateComponentPlacement::Device) => {
                        Ok(empty_resident_kv_layer(window))
                    }
                    Some(StateComponentPlacement::Paged) => {
                        let manager = manager.as_ref().ok_or_else(|| {
                            Exception::custom(format!(
                                "MLX paged key/value state has no residency manager at layer {layer}"
                            ))
                        })?;
                        let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                            Exception::custom("key/value state global layer index overflowed")
                        })?;
                        PagedKeyValueCache::new_with_layout(
                            manager.clone(),
                            global_layer,
                            window,
                            0,
                            rank,
                        )
                        .map(MlxKeyValueLayerState::Paged)
                    }
                    Some(placement) => Err(Exception::custom(format!(
                        "MLX key/value state does not support selected placement {placement:?} at layer {layer}"
                    ))),
                    None => Err(Exception::custom(format!(
                        "MLX key/value state has no selected components at layer {layer}"
                    ))),
                }
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self {
            layout: eredu_runtime::SharedStateLayout::new(layout),
            global_layer_start,
            inference_retention: Default::default(),
            layers: eredu_runtime::HostSlotTable::new(layers.into_boxed_slice()),
            paged_transaction_branch: false,
        })
    }

    /// Returns the common absolute token offset, or zero for an empty state.
    pub fn offset(&self) -> i32 {
        self.layers.slots().first().map_or(0, KeyValueCache::offset)
    }

    /// Clears retained arrays without changing residency or attention windows.
    pub fn clear(&mut self) -> Result<(), Exception> {
        for layer in self.layers.slots_mut() {
            layer.clear()?;
        }
        Ok(())
    }

    /// Inventories layer arrays and all retained paged managers together.
    pub(crate) fn retained_storage(
        &self,
    ) -> Result<
        crate::backend::runtime::residency::storage::RetainedStorage,
        crate::backend::runtime::residency::manager::ResidencyError,
    > {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub(crate) fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
        storage
            .include_retained_values::<crate::backend::runtime::residency::manager::ResidencyError>(
                |visitor| {
                    for layer in self.layers.slots() {
                        RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(
                            layer, visitor,
                        );
                    }
                    Ok(true)
                },
            )?;
        for manager in self.layers.slots().iter().filter_map(|layer| match layer {
            MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
            _ => None,
        }) {
            manager.collect_retained_storage(storage)?;
        }
        Ok(())
    }

    /// Borrows native layer arrays; sealed paged blocks remain with their managers.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        self.layers
            .slots()
            .iter()
            .flat_map(RuntimeLayerState::<MlxNeuralBackend>::retained_values)
            .map(MlxTensor::as_array)
            .collect()
    }

    /// Creates an independently advanceable speculative fork.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            inference_retention: self.inference_retention.clone(),
            layers: eredu_runtime::HostSlotTable::new(
                self.layers
                    .slots()
                    .iter()
                    .map(MlxKeyValueLayerState::deep_clone_state)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            ),
            paged_transaction_branch: self.paged_transaction_branch,
        })
    }

    pub(crate) fn supports_isolated_snapshot(&self) -> bool {
        let mut manager = None;
        self.layers.slots().iter().all(|layer| match layer {
            MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_) => true,
            MlxKeyValueLayerState::Paged(cache) => {
                let id = cache.manager().session_id();
                *manager.get_or_insert(id) == id
            }
        })
    }

    pub(crate) fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.layers
            .slots()
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => {
                    Some(cache.manager().isolated_snapshot_bytes())
                }
                _ => None,
            })
            .unwrap_or(Some(0))
    }

    pub(crate) fn original_isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.layers
            .slots()
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => {
                    Some(cache.manager().original_isolated_snapshot_bytes())
                }
                _ => None,
            })
            .unwrap_or(Some(0))
    }

    pub(crate) fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        // At most one new block per token per layer. This conservative catalog
        // bound also covers partial tails becoming sealed during advancement.
        let paged = self
            .layers
            .slots()
            .iter()
            .filter(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_)))
            .count() as u64;
        additional
            .checked_add(1)?
            .checked_mul(paged)?
            .checked_mul(8192)
    }

    /// Copies device arrays, mutable tails and sealed blocks with independent
    /// namespace ownership. Every paged copy remains in the same finite pool.
    pub(crate) fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        if !self.supports_isolated_snapshot() {
            return Err(Exception::custom(
                "isolated state contains inconsistent paging managers",
            ));
        }
        let manager = self
            .layers
            .slots()
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
                _ => None,
            })
            .map(|manager| {
                manager
                    .isolated_snapshot(stream)
                    .map_err(|e| Exception::custom(e.to_string()))
            })
            .transpose()?;
        let layers = self
            .layers
            .slots()
            .iter()
            .map(|layer| match layer {
                MlxKeyValueLayerState::Stateless => Ok(MlxKeyValueLayerState::Stateless),
                MlxKeyValueLayerState::Device(cache) => cache
                    .isolated_snapshot(stream)
                    .map(MlxKeyValueLayerState::Device),
                MlxKeyValueLayerState::Paged(cache) => {
                    let mut copy = cache.deep_clone_state(stream)?;
                    copy.rebind_paging_manager(
                        manager
                            .as_ref()
                            .expect("paged state has copied manager")
                            .clone(),
                    );
                    Ok(MlxKeyValueLayerState::Paged(copy))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            inference_retention: self.inference_retention.clone(),
            paged_transaction_branch: self.paged_transaction_branch,
            layers: eredu_runtime::HostSlotTable::new(layers.into_boxed_slice()),
        })
    }

    pub(crate) fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        let manager = self.layers.slots().iter().find_map(|layer| match layer {
            MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_) => None,
            MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
        });
        let Some(manager) = manager else {
            return self.deep_clone_state();
        };
        if self.layers.slots().iter().any(|layer| match layer {
            MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_) => false,
            MlxKeyValueLayerState::Paged(cache) => {
                cache.manager().session_id() != manager.session_id()
            }
        }) {
            return Err(Exception::custom(
                "key/value state contains more than one paging session",
            ));
        }
        let manager = manager
            .fork_session(stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut fork = self.deep_clone_state()?;
        for layer in fork.layers.slots_mut() {
            if let MlxKeyValueLayerState::Paged(cache) = layer {
                cache.rebind_paging_manager(manager.clone());
            }
        }
        Ok(fork)
    }

    pub(super) fn has_same_transaction_identity(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.global_layer_start == other.global_layer_start
            && self.layers.len() == other.layers.len()
            && self.layers.slots().iter().zip(other.layers.slots()).all(
                |(canonical, branch)| match (canonical, branch) {
                    (MlxKeyValueLayerState::Stateless, MlxKeyValueLayerState::Stateless) => true,
                    (MlxKeyValueLayerState::Device(_), MlxKeyValueLayerState::Device(_)) => true,
                    (
                        MlxKeyValueLayerState::Paged(canonical),
                        MlxKeyValueLayerState::Paged(branch),
                    ) => canonical.has_same_transaction_identity(branch),
                    _ => false,
                },
            )
    }

    /// Restores every append-only layer to an exact speculative checkpoint.
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
                "key/value state checkpoint layout does not match canonical state",
            ));
        }
        self.inference_retention
            .restore_admission(&checkpoint.inference_retention);
        for (current, previous) in self
            .layers
            .slots_mut()
            .iter_mut()
            .zip(checkpoint.layers.slots())
        {
            current.restore_checkpoint(previous, stream)?;
        }
        Ok(())
    }

    /// Returns aggregate telemetry when this is paged state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .slots()
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => Some(cache.report()),
                MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_) => None,
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
        for layer in self.layers.slots_mut() {
            if matches!(layer, MlxKeyValueLayerState::Stateless) {
                continue;
            }
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
            .slots()
            .iter()
            .map(|layer| match layer {
                MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_) => Ok(None),
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
        state
            .inference_retention
            .extend_from(&self.inference_retention);
        state.paged_transaction_branch = false;
        *self = state;
        Ok(())
    }

    fn discard_branch(mut branch: Self::Branch) -> Result<(), Self::Error> {
        let mut first_error = None;
        for (layer, checkpoint) in branch
            .state
            .layers
            .slots_mut()
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
        self.layers.slots().iter().all(|layer| {
            matches!(
                layer,
                MlxKeyValueLayerState::Stateless | MlxKeyValueLayerState::Device(_)
            )
        })
    }
}

impl eredu_runtime::working_memory::InferenceStateRetention for MlxKeyValueState {
    fn inference_retention(&self) -> &eredu_runtime::working_memory::InferenceRetention {
        &self.inference_retention
    }

    fn inference_retention_mut(
        &mut self,
    ) -> &mut eredu_runtime::working_memory::InferenceRetention {
        &mut self.inference_retention
    }

    fn retain_inference(&mut self, request: &eredu_runtime::working_memory::InferenceRequest) {
        self.inference_retention.retain(request);
    }
}

impl RuntimeState<MlxNeuralBackend> for MlxKeyValueState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn layout(&self) -> &StateLayout {
        self.layout.layout()
    }

    fn shared_layout(&self) -> Option<&eredu_runtime::SharedStateLayout> {
        Some(&self.layout)
    }

    fn visit_all_retained_values(
        &self,
        visitor: &mut dyn FnMut(&MlxTensor),
    ) -> Result<(), StateError> {
        for layer in self.layers.slots() {
            RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(layer, visitor);
        }
        Ok(())
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        let layer = ordinal;
        self.layers
            .slots()
            .get(layer)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

impl LayerRuntimeState<MlxNeuralBackend> for MlxKeyValueState {
    type LayerState = MlxKeyValueLayerState;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .slots_mut()
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }
}

impl ResettableRuntimeState<MlxNeuralBackend> for MlxKeyValueState {
    fn reset_segment(&mut self, segment: &StateSegmentId) -> Result<(), StateError> {
        let range = self
            .layout
            .layout()
            .segment(segment)
            .map(StateSegmentSpec::layers)
            .ok_or_else(|| StateError::UnknownSegment {
                segment: segment.clone(),
            })?;
        if self.paged_transaction_branch
            && self.layers.slots()[range.clone()]
                .iter()
                .any(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_)))
        {
            return Err(StateError::ResetFailed(
                "paged state segments cannot be reset inside a transaction branch without copy-on-write page ownership"
                    .into(),
            ));
        }
        for layer in &mut self.layers.slots_mut()[range] {
            ResettableRuntimeLayerState::<MlxNeuralBackend>::reset(layer)?;
        }
        Ok(())
    }
}

impl AsRef<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_ref(&self) -> &[MlxKeyValueLayerState] {
        self.layers.slots()
    }
}

impl AsMut<[MlxKeyValueLayerState]> for MlxKeyValueState {
    fn as_mut(&mut self) -> &mut [MlxKeyValueLayerState] {
        self.layers.slots_mut()
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

#[path = "key_value/original_reset.rs"]
mod original_reset;

#[path = "key_value/snapshot_source.rs"]
mod snapshot_source;
pub(crate) use snapshot_source::PagedSnapshotSource;
pub(in crate::backend::runtime::cache::state) use snapshot_source::{PagedSnapshotLayer, PagedSnapshotState};

pub(crate) use prepared_copy::{
    InitializedPagedKvCopy, PagedKvPreparationError, PreparedPagedKvCopy, PreparedPagedKvHostCopy,
    SavedPagedKvCopy,
};
