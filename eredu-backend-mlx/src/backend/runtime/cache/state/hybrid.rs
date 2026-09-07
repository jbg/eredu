//! Mixed attention and fixed-component state realization.

use super::*;

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
                .checkpoint_clone_state()
                .map(MlxKeyValueLayerState::Device)
                .map(Self::KeyValue),
            Self::KeyValue(MlxKeyValueLayerState::Paged(cache)) => Ok(Self::KeyValue(
                MlxKeyValueLayerState::Paged(cache.checkpoint_clone_state()),
            )),
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
                RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache).collect()
            }
            Self::Compressed(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache).collect()
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
            ) => {
                *current = previous.checkpoint_clone_state()?;
                Ok(())
            }
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
        Ok(Self {
            attention,
            fixed: self.fixed.clone(),
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

    fn from_selected(
        layer: usize,
        policy: &LayerCachePolicy,
        components: &[SelectedStateComponentRealization],
        manager: Option<&CacheResidencyManager>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let attention_policy = hybrid_attention_policy(layer, policy)?;
        let attention_components = match policy {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => 0,
            LayerCachePolicy::KeyOnly { .. } | LayerCachePolicy::KeyOnlyWithFixedState { .. } => 1,
            LayerCachePolicy::KeyValue { .. }
            | LayerCachePolicy::KeyValueWithFixedState { .. }
            | LayerCachePolicy::CompressedLatentRotary { .. } => 2,
        };
        let (attention_components, fixed_components) = components
            .split_at_checked(attention_components)
            .ok_or_else(|| {
                Exception::custom(format!(
                    "selected hybrid state omits attention components at layer {layer}"
                ))
            })?;
        if fixed_components.len() != policy.fixed_state().len() {
            return Err(Exception::custom(format!(
                "selected hybrid state fixed-component count differs at layer {layer}"
            )));
        }
        for component in fixed_components {
            if !matches!(component.component().role(), StateComponentRole::Fixed(_)) {
                return Err(Exception::custom(format!(
                    "selected hybrid fixed-state contract differs at layer {layer}"
                )));
            }
            if component.placement() != StateComponentPlacement::Device {
                return Err(Exception::custom(format!(
                    "MLX hybrid fixed state does not support selected placement {:?} at layer {layer}",
                    component.placement()
                )));
            }
        }
        let placement = common_selected_placement(layer, "hybrid attention", attention_components)?;
        let attention = match (attention_policy, placement) {
            (None, None) => None,
            (Some(policy), Some(StateComponentPlacement::Device)) => Some(match policy {
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
            }),
            (Some(policy), Some(StateComponentPlacement::Paged)) => {
                let manager = manager.ok_or_else(|| {
                    Exception::custom(format!(
                        "MLX paged hybrid attention has no residency manager at layer {layer}"
                    ))
                })?;
                Some(match policy {
                    HybridAttentionPolicy::KeyValue { window, key_only } => {
                        let cache = if key_only {
                            PagedKeyValueCache::new_key_only_with_layout(
                                manager.clone(),
                                layer,
                                window,
                                0,
                                rank,
                            )
                        } else {
                            PagedKeyValueCache::new_with_layout(
                                manager.clone(),
                                layer,
                                window,
                                0,
                                rank,
                            )
                        }?;
                        MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))
                    }
                    HybridAttentionPolicy::Compressed => MlxHybridAttentionState::Compressed(
                        CompressedLatentCache::new_paged(manager.clone(), layer, rank)?,
                    ),
                })
            }
            (Some(_), Some(placement)) => {
                return Err(Exception::custom(format!(
                    "MLX hybrid attention does not support selected placement {placement:?} at layer {layer}"
                )))
            }
            (None, Some(_)) | (Some(_), None) => {
                return Err(Exception::custom(format!(
                    "selected hybrid attention contract differs from the layout at layer {layer}"
                )))
            }
        };
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

impl RuntimeLayerState<MlxNeuralBackend> for MlxHybridLayerState {
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

impl RuntimeStateComponents<MlxNeuralBackend> for MlxHybridLayerState {
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
    manager: Option<CacheResidencyManager>,
}

impl MlxHybridState {
    pub(crate) fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.layers
            .iter()
            .try_fold(0, |bound, layer| match &layer.attention {
                None => Some(bound),
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) => {
                    Some(bound.max(cache.continuation_capacity_bound(additional)?))
                }
                _ => None,
            })
    }

    pub(crate) fn supports_isolated_snapshot(&self) -> bool {
        self.manager.is_none()
            && self.layers.iter().all(|layer| {
                matches!(
                    layer.attention,
                    None | Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Device(_)
                    ))
                )
            })
    }

    /// Deeply copies attention and every architecture-declared fixed tensor.
    /// Fixed-state handles must not inherit the transaction checkpoint's sharing.
    pub(crate) fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        if !self.supports_isolated_snapshot() {
            return Err(Exception::custom(
                "isolated snapshots of paged/compressed state are unsupported",
            ));
        }
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let attention = match &layer.attention {
                    None => None,
                    Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(
                        cache,
                    ))) => Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Device(cache.isolated_snapshot(stream)?),
                    )),
                    _ => unreachable!("validated device attention and fixed state"),
                };
                let fixed = layer
                    .fixed
                    .iter()
                    .map(|(role, value)| {
                        Ok((
                            *role,
                            value
                                .as_ref()
                                .map(|value| {
                                    value
                                        .as_array()
                                        .contiguous(false, stream)?
                                        .deep_clone()
                                        .map(MlxTensor::from_array)
                                })
                                .transpose()?,
                        ))
                    })
                    .collect::<Result<_, Exception>>()?;
                Ok(MlxHybridLayerState {
                    attention,
                    fixed,
                    fixed_offset: layer.fixed_offset,
                })
            })
            .collect::<Result<_, Exception>>()?;
        Ok(Self {
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            layers,
            manager: None,
        })
    }

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
            manager: None,
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
            manager: Some(manager),
        })
    }

    /// Creates the exact component placements selected before allocation.
    ///
    /// A paging manager is required when any selected attention component is
    /// paged. Fixed components are accepted only with explicit device
    /// placement.
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
                let global_layer = global_layer_start.checked_add(layer).ok_or_else(|| {
                    Exception::custom("hybrid state global layer index overflowed")
                })?;
                MlxHybridLayerState::from_selected(
                    global_layer,
                    policy,
                    components,
                    manager.as_ref(),
                    rank,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            global_layer_start,
            layers,
            manager,
        })
    }

    /// Mutably borrows the ordinary per-layer states used by neutral units.
    pub fn layers_mut(&mut self) -> &mut [MlxHybridLayerState] {
        &mut self.layers
    }

    /// Borrows the paging manager shared by this state's attention components.
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.manager.as_ref().or_else(|| {
            self.layers
                .iter()
                .filter_map(|layer| layer.attention.as_ref())
                .find_map(MlxHybridAttentionState::manager)
        })
    }

    /// Borrows every native array retained by the complete hybrid state.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(RuntimeLayerState::<MlxNeuralBackend>::retained_values)
            .map(MlxTensor::as_array)
            .collect()
    }

    /// Returns the common absolute token frontier.
    pub fn offset(&self) -> i32 {
        self.layers
            .first()
            .map_or(0, RuntimeStateComponents::position)
    }

    #[cfg(test)]
    pub(crate) fn semantic_snapshot(&self) -> Vec<(i32, Vec<(StateTensorRole, bool)>)> {
        self.layers
            .iter()
            .map(|layer| {
                (
                    layer.position(),
                    layer
                        .fixed
                        .iter()
                        .map(|(role, value)| (*role, value.is_some()))
                        .collect(),
                )
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn fixed_numeric_snapshot(
        &self,
    ) -> Result<Vec<(usize, StateTensorRole, Vec<i32>, Vec<f32>)>, Exception> {
        let mut snapshot = Vec::new();
        for (layer, state) in self.layers.iter().enumerate() {
            for (role, value) in &state.fixed {
                let Some(value) = value else {
                    continue;
                };
                let evaluated = value.as_array().evaluated()?;
                snapshot.push((
                    layer,
                    *role,
                    value.as_array().shape().to_vec(),
                    evaluated.as_slice::<f32>().to_vec(),
                ));
            }
        }
        Ok(snapshot)
    }

    #[cfg(test)]
    pub(crate) fn retained_numeric_snapshot(&self) -> Result<Vec<(Vec<i32>, Vec<f32>)>, Exception> {
        self.retained_arrays()
            .into_iter()
            .map(|array| {
                let evaluated = array.evaluated()?;
                Ok((array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec()))
            })
            .collect()
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
            manager: self.manager.clone(),
        })
    }

    pub(crate) fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        let Some(manager) = self.manager.as_ref() else {
            return self.deep_clone_state();
        };
        let manager = manager
            .fork_session(stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut fork = self.deep_clone_state()?;
        for layer in &mut fork.layers {
            match layer.attention.as_mut() {
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
                    cache.rebind_paging_manager(manager.clone());
                }
                Some(MlxHybridAttentionState::Compressed(cache)) => {
                    cache.rebind_paging_manager(manager.clone());
                }
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(_)))
                | None => {}
            }
        }
        fork.manager = Some(manager);
        Ok(fork)
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

    /// Commits exactly one architecture-declared state segment.
    pub fn commit_segment_from(&mut self, source: &Self, segment: &str) -> Result<(), Exception> {
        if self.layout != source.layout
            || self.global_layer_start != source.global_layer_start
            || self.layers.len() != source.layers.len()
        {
            return Err(Exception::custom(
                "hybrid draft state layout does not match canonical state",
            ));
        }
        let range = self.segment_range(segment)?;
        self.layers[range.clone()].clone_from_slice(&source.layers[range]);
        Ok(())
    }

    /// Resolves one named architecture-owned state segment.
    pub fn segment_range(&self, segment: &str) -> Result<Range<usize>, Exception> {
        let segment =
            StateSegmentId::new(segment).map_err(|error| Exception::custom(error.to_string()))?;
        self.layout
            .segment(&segment)
            .map(StateSegmentSpec::layers)
            .ok_or_else(|| Exception::custom(format!("unknown hybrid state segment {segment:?}")))
    }

    /// Returns aggregate telemetry when this state contains paged attention.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.residency_manager()
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
            if state.attention.is_none() && !state.fixed.is_empty() {
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
    ) -> Result<Vec<PromptCacheStateArray<'_>>, Exception> {
        if range.end > self.layers.len() || layer_prefix_offsets.len() != range.len() {
            return Err(Exception::custom(
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
        if descriptor.layer_prefix_offsets().len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "prompt-cache descriptor supplied {} layer frontiers for {} hybrid state layers",
                descriptor.layer_prefix_offsets().len(),
                self.layers.len()
            )));
        }
        let mut manager = self.manager.clone();
        for (layer, (state, delta)) in self
            .layers
            .iter_mut()
            .zip(descriptor.layer_prefix_offsets())
            .enumerate()
        {
            if state.attention.is_none() && state.fixed.is_empty() {
                continue;
            }
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

impl RuntimeState<MlxNeuralBackend> for MlxHybridState {
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

impl LayerRuntimeState<MlxNeuralBackend> for MlxHybridState {
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
#[path = "tests/semantic_transactions.rs"]
mod semantic_transaction_tests;

#[cfg(test)]
#[path = "tests/isolated_snapshots.rs"]
mod isolated_snapshot_tests;
