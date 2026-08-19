//! Backend-neutral mutable model-state contracts.
//!
//! Architectures declare state geometry through [`StateLayout`]. Runtime
//! policies select a residency realization, while concrete backends retain
//! their native layer-state and tensor types.

use std::marker::PhantomData;

use eredu_core::{
    cache::{LayerCachePolicy, PromptCacheError, PromptCacheModelIdentity, PromptCacheTopology},
    LayerSchedule,
};
use eredu_nn::NeuralBackend;

/// Complete ordered mutable-state geometry owned by one model instance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateLayout {
    layers: LayerSchedule<LayerCachePolicy>,
}

impl StateLayout {
    /// Creates and validates an ordered per-layer state layout.
    pub fn new(layers: LayerSchedule<LayerCachePolicy>) -> Result<Self, StateError> {
        if layers.is_empty() {
            return Err(StateError::EmptyLayout);
        }
        for (layer, policy) in layers.iter().enumerate() {
            policy
                .validate()
                .map_err(|error| StateError::InvalidLayer {
                    layer,
                    reason: error.to_string(),
                })?;
        }
        Ok(Self { layers })
    }

    /// Returns the number of architecture-global layers represented here.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether this layout has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns one layer's exact state policy.
    pub fn layer(&self, layer: usize) -> Option<&LayerCachePolicy> {
        self.layers.get(layer)
    }

    /// Borrows the portable ordered layer schedule.
    pub const fn layers(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layers
    }
}

/// Runtime policy selecting how architecture-declared mutable state is held.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StateResidencyPlan {
    /// Keep every mutable layer state in backend execution memory.
    Device,
    /// Seal append-only state into independently pageable blocks.
    Paged(PagedStatePlan),
}

/// Backend-independent controls for a paged mutable-state realization.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PagedStatePlan {
    block_tokens: usize,
    device_bytes: u64,
    host_bytes: u64,
}

impl PagedStatePlan {
    /// Creates finite paged-state limits.
    pub fn new(
        block_tokens: usize,
        device_bytes: u64,
        host_bytes: u64,
    ) -> Result<Self, StateError> {
        if block_tokens == 0 {
            return Err(StateError::InvalidResidency(
                "paged-state block size must be nonzero".into(),
            ));
        }
        if device_bytes == 0 {
            return Err(StateError::InvalidResidency(
                "paged-state device budget must be nonzero".into(),
            ));
        }
        Ok(Self {
            block_tokens,
            device_bytes,
            host_bytes,
        })
    }

    /// Returns the number of token positions sealed into one block.
    pub const fn block_tokens(self) -> usize {
        self.block_tokens
    }

    /// Returns the finite execution-device budget.
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }

    /// Returns the finite host budget; zero disables host residency.
    pub const fn host_bytes(self) -> u64 {
        self.host_bytes
    }
}

/// Concrete layer state capable of exposing backend-native retained tensors.
pub trait RuntimeLayerState<B: NeuralBackend> {
    /// Allocation-free iterator returned for one layer.
    type RetainedValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;

    /// Borrows tensors that must remain alive through this layer's submission.
    fn retained_values(&self) -> Self::RetainedValues<'_>;
}

/// Mutable state realization consumed by generic resident and layerwise engines.
pub trait RuntimeState<B: NeuralBackend> {
    /// Concrete monomorphized state passed to one architecture unit.
    type LayerState: RuntimeLayerState<B>;

    /// Returns the exact layout used to create this realization.
    fn layout(&self) -> &StateLayout;

    /// Mutably borrows one architecture-global layer state.
    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError>;

    /// Borrows tensors retained by one layer without cloning their handles.
    fn retained_values(
        &self,
        layer: usize,
    ) -> Result<<Self::LayerState as RuntimeLayerState<B>>::RetainedValues<'_>, StateError>;
}

/// Fully device-resident state with one concrete value per architecture layer.
#[derive(Debug)]
pub struct DeviceState<B: NeuralBackend, L> {
    layout: StateLayout,
    layers: Vec<L>,
    backend: PhantomData<fn() -> B>,
}

impl<B: NeuralBackend, L> DeviceState<B, L> {
    /// Realizes every layer through a backend-specific construction closure.
    pub fn create<E>(
        layout: StateLayout,
        mut create: impl FnMut(usize, &LayerCachePolicy) -> Result<L, E>,
    ) -> Result<Self, E> {
        let layers = layout
            .layers()
            .iter()
            .enumerate()
            .map(|(layer, policy)| create(layer, policy))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout,
            layers,
            backend: PhantomData,
        })
    }
}

impl<B, L> RuntimeState<B> for DeviceState<B, L>
where
    B: NeuralBackend,
    L: RuntimeLayerState<B>,
{
    type LayerState = L;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, StateError> {
        let count = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(StateError::UnknownLayer { layer, count })
    }

    fn retained_values(
        &self,
        layer: usize,
    ) -> Result<<Self::LayerState as RuntimeLayerState<B>>::RetainedValues<'_>, StateError> {
        self.layers
            .get(layer)
            .map(RuntimeLayerState::retained_values)
            .ok_or(StateError::UnknownLayer {
                layer,
                count: self.layers.len(),
            })
    }
}

/// Architecture and placement identity used to derive persistence identity.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModelStateIdentity {
    /// Stable architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Cache-relevant architecture fingerprint.
    pub architecture_fingerprint: String,
    /// Total architecture layer count.
    pub layer_count: usize,
    /// Inclusive first global layer owned by this runtime instance.
    pub global_layer_start: usize,
    /// Attention sink or pinned-prefix token count.
    pub sink_tokens: usize,
    /// Per-owned-layer processed-token deltas.
    pub layer_prefix_offsets: Vec<i32>,
    /// Rank-local distributed placement.
    pub topology: PromptCacheTopology,
}

impl ModelStateIdentity {
    /// Combines architecture identity, placement, and exact state geometry.
    pub fn prompt_cache_identity(
        &self,
        layout: &StateLayout,
    ) -> Result<PromptCacheModelIdentity, PromptCacheError> {
        let global_layer_end = self
            .global_layer_start
            .checked_add(layout.len())
            .ok_or_else(|| PromptCacheError::Malformed("owned layer range overflowed".into()))?;
        let identity = PromptCacheModelIdentity {
            model_family: self.model_family.clone(),
            effective_model_type: self.effective_model_type.clone(),
            architecture_fingerprint: self.architecture_fingerprint.clone(),
            layer_count: self.layer_count,
            global_layer_start: self.global_layer_start,
            global_layer_end,
            sink_tokens: self.sink_tokens,
            topology: self.topology.clone(),
            layer_layout: layout.layers().clone(),
            layer_prefix_offsets: self.layer_prefix_offsets.clone(),
        };
        identity.validate()?;
        Ok(identity)
    }
}

/// Invalid architecture state geometry or runtime access.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum StateError {
    /// A model declared no state-bearing layer slots.
    #[error("runtime state layout must contain at least one layer")]
    EmptyLayout,
    /// One layer supplied an invalid portable state policy.
    #[error("invalid runtime state policy for layer {layer}: {reason}")]
    InvalidLayer {
        /// Invalid global layer index.
        layer: usize,
        /// Validation detail.
        reason: String,
    },
    /// A residency plan violates a finite-resource invariant.
    #[error("invalid runtime state residency plan: {0}")]
    InvalidResidency(String),
    /// A runtime requested a layer outside the realized layout.
    #[error("runtime state layer {layer} is outside the {count}-layer layout")]
    UnknownLayer {
        /// Requested layer.
        layer: usize,
        /// Realized layer count.
        count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{AttentionPolicy, LayerSchedule};

    fn layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                2,
                vec![
                    LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
                    LayerCachePolicy::key_value(
                        AttentionPolicy::from_sliding_window(Some(16)).unwrap(),
                        2,
                        8,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn prompt_identity_is_derived_from_layout_and_placement() {
        let layout = layout();
        let identity = ModelStateIdentity {
            model_family: "fixture".into(),
            effective_model_type: "fixture-v1".into(),
            architecture_fingerprint: "geometry-1".into(),
            layer_count: 4,
            global_layer_start: 1,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0, 0],
            topology: PromptCacheTopology::default(),
        }
        .prompt_cache_identity(&layout)
        .unwrap();
        assert_eq!(identity.global_layer_start, 1);
        assert_eq!(identity.global_layer_end, 3);
        assert_eq!(identity.layer_layout, *layout.layers());
    }

    #[test]
    fn prompt_identity_rejects_offsets_that_do_not_match_layout() {
        let error = ModelStateIdentity {
            model_family: "fixture".into(),
            effective_model_type: "fixture-v1".into(),
            architecture_fingerprint: "geometry-1".into(),
            layer_count: 2,
            global_layer_start: 0,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            topology: PromptCacheTopology::default(),
        }
        .prompt_cache_identity(&layout())
        .unwrap_err();
        assert!(matches!(error, PromptCacheError::Incompatible(_)));
    }
}
