//! Exact empty hybrid tables and source-qualified paged manager; no numerical copy.
use super::super::paged_reset::{MlxPagedResetContext, MlxPagedResetPlan};
use super::*;
use eredu_core::BackendFailure;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{ResidentTableResetState, WorkingMemoryError};
use std::mem::size_of;

fn frames() -> usize {
    size_of::<(
        &MlxHybridState,
        &MlxPagedResetPlan,
        Option<&HostMetadataFunding>,
        std::slice::Iter<'static, MlxHybridLayerState>,
        Option<&'static CacheResidencyManager>,
        (usize, &'static MlxHybridLayerState),
        Option<&'static MlxHybridAttentionState>,
        Option<&'static LayerCachePolicy>,
        &mut MlxPagedResetContext,
        Option<eredu_runtime::HostSlotTable<(StateTensorRole, Option<MlxTensor>)>>,
        Result<MlxHybridLayerState, BackendFailure>,
    )>()
}
impl MlxHybridState {
    fn reset_manager(&self) -> Result<Option<&CacheResidencyManager>, WorkingMemoryError> {
        for (index, layer) in self.layers.slots().iter().enumerate() {
            if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) =
                &layer.attention
            {
                let manager = self
                    .manager
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                let policy = self
                    .layout
                    .layout()
                    .layer(index)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                if !manager.same_catalog(cache.manager())
                    || self.global_layer_start.checked_add(index) != Some(cache.global_layer())
                    || !Self::validate_resident_reset_layer(layer, policy)
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
        }
        Ok(self.manager.as_ref())
    }
}

impl ResidentTableResetState for MlxHybridState {
    type Layer = MlxHybridLayerState;
    type ResetPlan = MlxPagedResetPlan;
    type ResetContext = MlxPagedResetContext;
    type Child = (StateTensorRole, Option<MlxTensor>);

    fn resident_reset_plan(&self) -> Result<(Self::ResetPlan, usize), WorkingMemoryError> {
        MlxPagedResetPlan::inspect(self.reset_manager()?, frames())
    }
    fn prepare_resident_reset_context(
        &self,
        plan: &Self::ResetPlan,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self::ResetContext, BackendFailure> {
        plan.prepare(
            self.reset_manager().map_err(BackendFailure::from_error)?,
            frames(),
            funding,
        )
    }
    fn validate_resident_reset_placement(
        layer: &Self::Layer,
        role: StateComponentRole,
        placement: eredu_runtime::StateComponentPlacement,
    ) -> bool {
        use eredu_runtime::StateComponentPlacement::{Device, Paged};
        match role {
            StateComponentRole::Fixed(role) => {
                placement == Device
                    && layer
                        .fixed
                        .table()
                        .slots()
                        .iter()
                        .any(|(actual, _)| *actual == role)
            }
            StateComponentRole::AttentionKeys | StateComponentRole::AttentionValues => matches!(
                (layer.attention.as_ref(), placement),
                (
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Paged(_)
                    )),
                    Paged
                ) | (
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Device(_)
                    )),
                    Device
                )
            ),
            _ => false,
        }
    }
    fn empty_resident_reset_layer_prepared(
        context: &mut Self::ResetContext,
        source: &Self::Layer,
        policy: &LayerCachePolicy,
        child: Option<eredu_runtime::HostSlotTable<Self::Child>>,
    ) -> Result<Self::Layer, BackendFailure> {
        if !Self::validate_resident_reset_layer(source, policy) || child.is_none() {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) =
            &source.attention
        {
            let manager = context
                .manager()
                .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::IdentityMismatch))?;
            Ok(MlxHybridLayerState {
                attention: Some(MlxHybridAttentionState::KeyValue(
                    MlxKeyValueLayerState::Paged(cache.empty_original_reset(manager.clone())),
                )),
                fixed: FixedStateSlots::from_published_slots(child.expect("validated child")),
                fixed_offset: 0,
            })
        } else {
            Ok(Self::empty_resident_reset_layer_with_child(policy, child))
        }
    }
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
        self.reset_manager().is_ok()
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
            (
                LayerCachePolicy::KeyValue { attention, .. }
                | LayerCachePolicy::KeyValueWithFixedState { attention, .. },
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))),
            ) => attention
                .sliding_window_i32()
                .is_ok_and(|window| cache.matches_original_reset_policy(window)),
            // Key-only/compressed variants need their own empty owner proof.
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
        context: &mut Self::ResetContext,
        layout: eredu_runtime::SharedStateLayout,
        global_layer_start: usize,
        layers: eredu_runtime::HostSlotTable<Self::Layer>,
    ) -> Self {
        Self {
            layout,
            global_layer_start,
            layers,
            manager: context.take_manager(),
            inference_retention: Default::default(),
        }
    }
}

#[cfg(test)]
#[path = "resident_reset/tests.rs"]
mod tests;
