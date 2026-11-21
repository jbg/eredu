//! Projection from admitted semantic state to exact layer storage requests.

use eredu_core::{
    cache::{
        mechanisms::{AttentionStorageKind, LayerStateAllocation},
        LayerCachePolicy, StateComponentRole,
    },
    BackendFailure, BackendFailureKind,
};

use crate::{SelectedStateRealization, StateComponentPlacement, StateLayout};

/// Prepared storage requests and the layout retained by the portable executor.
#[derive(Debug)]
pub struct StateAllocation<'a> {
    layout: &'a StateLayout,
    layers: Vec<LayerStateAllocation>,
}

impl<'a> StateAllocation<'a> {
    /// Retained semantic layout.
    pub fn layout(&self) -> &StateLayout {
        self.layout
    }

    /// Exact requests in layer order.
    pub fn layers(&self) -> &[LayerStateAllocation] {
        &self.layers
    }
}

impl SelectedStateRealization {
    /// Validates component placement and projects the exact storage for each
    /// layer before assembling the portable state.
    pub fn allocation(&self) -> Result<StateAllocation<'_>, BackendFailure> {
        fn invalid(detail: impl Into<String>) -> BackendFailure {
            BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::new(std::io::ErrorKind::InvalidInput, detail.into()),
            )
            .with_operation("project state allocation")
        }

        let layout = self.layout();
        let mut selected = self.components();
        let mut layers = Vec::with_capacity(layout.len());
        for (layer, policy) in layout.layers().iter().enumerate() {
            let expected = layout.components(layer).expect("validated layer layout");
            let (components, remainder) =
                selected.split_at_checked(expected.len()).ok_or_else(|| {
                    invalid(format!("selected state omits components at layer {layer}"))
                })?;
            selected = remainder;
            let mut attention_placement = None;
            let mut fixed = Vec::new();
            for (component, expected) in components.iter().zip(expected) {
                if component.layer() != layer || component.component() != expected {
                    return Err(invalid(format!(
                        "selected state differs from layout at layer {layer}"
                    )));
                }
                match expected.role() {
                    StateComponentRole::Fixed(role) => {
                        if component.placement() != StateComponentPlacement::Device {
                            return Err(invalid(format!(
                                "fixed state requires device backing at layer {layer}"
                            )));
                        }
                        fixed.push(role);
                    }
                    StateComponentRole::AppendStream { .. } => {}
                    _ => {
                        let placement = component.placement();
                        if attention_placement
                            .replace(placement)
                            .is_some_and(|previous| previous != placement)
                        {
                            return Err(invalid(format!(
                                "attention state requires common backing at layer {layer}"
                            )));
                        }
                    }
                }
            }
            let attention = attention_storage_kind(policy).map_err(BackendFailure::from_error)?;
            layers.push(LayerStateAllocation {
                attention: attention
                    .zip(attention_placement)
                    .map(|(kind, placement)| (kind, placement == StateComponentPlacement::Paged)),
                fixed,
                streams: Vec::new(),
            });
        }
        if !selected.is_empty() {
            return Err(invalid(
                "selected state has components beyond the declared layers",
            ));
        }
        for (binding, backing) in self.append_stream_allocations() {
            let layer = layers
                .get_mut(binding.layer)
                .ok_or_else(|| invalid("selected append stream is outside the declared layers"))?;
            layer.streams.push((binding.clone(), *backing));
        }
        Ok(StateAllocation { layout, layers })
    }
}

pub(super) fn attention_storage_kind(
    policy: &LayerCachePolicy,
) -> Result<Option<AttentionStorageKind>, eredu_core::LayerScheduleError> {
    Ok(match policy {
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => None,
        LayerCachePolicy::CompressedLatentRotary { .. } => {
            Some(AttentionStorageKind::CompressedLatent)
        }
        LayerCachePolicy::KeyValue { attention, .. }
        | LayerCachePolicy::KeyValueWithState { attention, .. }
        | LayerCachePolicy::KeyOnly { attention, .. }
        | LayerCachePolicy::KeyOnlyWithState { attention, .. } => {
            Some(AttentionStorageKind::KeyValue {
                window: attention.sliding_window_i32()?,
                key_only: matches!(
                    policy,
                    LayerCachePolicy::KeyOnly { .. } | LayerCachePolicy::KeyOnlyWithState { .. }
                ),
            })
        }
    })
}
