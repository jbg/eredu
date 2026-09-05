//! Model-wide key/value state realization and transactions.

use super::*;

/// One concrete MLX key/value layer state selected by runtime residency policy.
#[derive(Debug, Clone)]
pub enum MlxKeyValueLayerState {
    /// Contiguous execution-device keys and values.
    Device(ConcatKeyValueCache),
    /// Block-addressable keys and values managed by finite residency budgets.
    Paged(PagedKeyValueCache),
}

impl MlxKeyValueLayerState {
    pub(super) fn clear(&mut self) -> Result<(), Exception> {
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
            Self::Device(cache) => cache.checkpoint_clone_state().map(Self::Device),
            Self::Paged(cache) => Ok(Self::Paged(cache.checkpoint_clone_state())),
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
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

impl RuntimeLayerState<MlxNeuralBackend> for MlxKeyValueLayerState {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        match self {
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
#[derive(Debug, Clone)]
pub struct MlxKeyValueState {
    layout: StateLayout,
    global_layer_start: usize,
    pub(super) layers: Vec<MlxKeyValueLayerState>,
    paged_transaction_branch: bool,
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

impl MlxKeyValueState {
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
                let window = key_value_window(layer, policy)?;
                Ok(MlxKeyValueLayerState::Device(match window {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
                    None => ConcatKeyValueCache::new(),
                }))
            })
            .collect::<Result<Vec<_>, Exception>>()?;
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
                let window = key_value_window(layer, policy)?;
                let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                    Exception::custom("key/value state global layer index overflowed")
                })?;
                PagedKeyValueCache::new_with_layout(manager.clone(), global_layer, window, 0, rank)
                    .map(MlxKeyValueLayerState::Paged)
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
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
                let window = key_value_window(layer, policy)?;
                match common_selected_placement(layer, "key/value", components)? {
                    Some(StateComponentPlacement::Device) => {
                        Ok(MlxKeyValueLayerState::Device(match window {
                            Some(window) => {
                                ConcatKeyValueCache::new_for_sliding_attention(window)
                            }
                            None => ConcatKeyValueCache::new(),
                        }))
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
            layout,
            global_layer_start,
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
            .flat_map(RuntimeLayerState::<MlxNeuralBackend>::retained_values)
            .map(MlxTensor::as_array)
            .collect()
    }

    /// Creates an independently advanceable speculative fork.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers: self
                .layers
                .iter()
                .map(MlxKeyValueLayerState::deep_clone_state)
                .collect::<Result<_, _>>()?,
            paged_transaction_branch: self.paged_transaction_branch,
        })
    }

    pub(crate) fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        let manager = self.layers.iter().find_map(|layer| match layer {
            MlxKeyValueLayerState::Device(_) => None,
            MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
        });
        let Some(manager) = manager else {
            return self.deep_clone_state();
        };
        if self.layers.iter().any(|layer| match layer {
            MlxKeyValueLayerState::Device(_) => false,
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
        for layer in &mut fork.layers {
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
        if self.layout != checkpoint.layout
            || self.global_layer_start != checkpoint.global_layer_start
            || self.layers.len() != checkpoint.layers.len()
        {
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

impl RuntimeState<MlxNeuralBackend> for MlxKeyValueState {
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

impl LayerRuntimeState<MlxNeuralBackend> for MlxKeyValueState {
    type LayerState = MlxKeyValueLayerState;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }
}

impl ResettableRuntimeState<MlxNeuralBackend> for MlxKeyValueState {
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
            ResettableRuntimeLayerState::<MlxNeuralBackend>::reset(layer)?;
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
