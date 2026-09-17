//! Exact empty hybrid tables; no numerical copy or new native owner.
use super::*;
use eredu_runtime::working_memory::ResidentTableResetState;

impl ResidentTableResetState for MlxHybridState {
    type Layer = MlxHybridLayerState;
    type ResetPlan = ();
    type ResetContext = ();
    type Child = (StateTensorRole, Option<MlxTensor>);

    fn resident_reset_layers(&self) -> &eredu_runtime::HostSlotTable<Self::Layer> {
        &self.layers
    }
    fn resident_reset_layout(&self) -> &eredu_runtime::SharedStateLayout {
        &self.layout
    }
    fn resident_reset_global_start(&self) -> usize {
        self.global_layer_start
    }
    fn validate_resident_reset_state(&self) -> bool {
        self.manager.is_none()
    }

    fn resident_fork_is_empty(&self) -> bool {
        self.manager.is_none()
            && self.layers.slots().iter().all(|layer| {
                layer.fixed_offset == 0
                    && layer
                        .fixed
                        .table()
                        .slots()
                        .iter()
                        .all(|(_, value)| value.is_none())
                    && match layer.attention.as_ref() {
                        None => true,
                        Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(
                            cache,
                        ))) => cache.resident_fork_is_empty(),
                        _ => false,
                    }
            })
    }

    fn validate_resident_reset_layer(layer: &Self::Layer, policy: &LayerCachePolicy) -> bool {
        let attention = match (policy, layer.attention.as_ref()) {
            (LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }, None) => true,
            (
                LayerCachePolicy::KeyValue { attention, .. }
                | LayerCachePolicy::KeyValueWithFixedState { attention, .. },
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))),
            ) => attention
                .sliding_window_i32()
                .is_ok_and(|window| cache.matches_resident_reset_policy(window)),
            // Key-only/compressed/paged variants need their own empty owner proof.
            _ => false,
        };
        let slots = layer.fixed.table().slots();
        let declared = policy.fixed_state();
        attention
            && slots.len() == declared.len()
            && slots.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && slots
                .iter()
                .all(|(role, _)| declared.iter().any(|tensor| tensor.role == *role))
    }
    fn resident_reset_child(
        layer: &Self::Layer,
    ) -> Option<&eredu_runtime::HostSlotTable<Self::Child>> {
        Some(layer.fixed.table())
    }
    fn empty_resident_reset_child(source: &Self::Child) -> Self::Child {
        (source.0, None)
    }
    fn empty_resident_reset_layer(_: &LayerCachePolicy) -> Self::Layer {
        unreachable!("hybrid reset always supplies its actual funded child table")
    }
    fn empty_resident_reset_layer_with_child(
        policy: &LayerCachePolicy,
        child: Option<eredu_runtime::HostSlotTable<Self::Child>>,
    ) -> Self::Layer {
        let attention = match policy {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => None,
            LayerCachePolicy::KeyValue { attention, .. }
            | LayerCachePolicy::KeyValueWithFixedState { attention, .. } => {
                Some(MlxHybridAttentionState::KeyValue(
                    MlxKeyValueLayerState::Device(empty_resident_attention_cache(
                        attention.sliding_window_i32().expect("validated window"),
                    )),
                ))
            }
            _ => unreachable!("validated resident hybrid policy"),
        };
        MlxHybridLayerState {
            attention,
            fixed: FixedStateSlots::from_published_slots(
                child.expect("one child per hybrid layer"),
            ),
            fixed_offset: 0,
        }
    }
    fn from_resident_reset(
        layout: eredu_runtime::SharedStateLayout,
        global_layer_start: usize,
        layers: eredu_runtime::HostSlotTable<Self::Layer>,
    ) -> Self {
        Self {
            layout,
            global_layer_start,
            layers,
            manager: None,
            inference_retention: Default::default(),
        }
    }
}
