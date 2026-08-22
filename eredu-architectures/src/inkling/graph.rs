//! Typed Inkling component and runtime-state graphs.

use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    },
    LayerSchedule,
};
use eredu_runtime::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec, StateError, StateLayout, StateSegmentLifetime, StateSegmentSpec,
};

use super::ModelArgs;

/// Stable segment identity for the target decoder state.
pub const TARGET_STATE_SEGMENT: &str = "target";
/// Stable segment identity for checkpoint-embedded prediction state.
pub const PREDICTION_STATE_SEGMENT: &str = "prediction";

/// Builds the text/media/prediction graph from normalized family config.
pub fn component_graph(args: &ModelArgs) -> Result<ComponentGraph, ComponentGraphError> {
    let mut units = vec![ComponentSpec {
        id: "embedding".into(),
        kind: ComponentKind::StaticText,
        external_inputs: vec![ComponentDomain::TokenIds],
        dependencies: vec![],
        dependency_inputs: vec![],
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    }];
    if args.audio_config.is_some() {
        units.push(ComponentSpec {
            id: "audio".into(),
            kind: ComponentKind::Audio,
            external_inputs: vec![ComponentDomain::AudioFeatures],
            dependencies: vec![],
            dependency_inputs: vec![],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Media,
        });
    }
    if args.vision_config.is_some() {
        units.push(ComponentSpec {
            id: "vision".into(),
            kind: ComponentKind::Vision,
            external_inputs: vec![ComponentDomain::PatchMatrix],
            dependencies: vec![],
            dependency_inputs: vec![],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Media,
        });
    }
    let media = args.audio_config.is_some() as usize + args.vision_config.is_some() as usize;
    units.push(ComponentSpec {
        id: "assembly".into(),
        kind: ComponentKind::Assembly,
        external_inputs: vec![],
        dependencies: std::iter::once("embedding".into())
            .chain(args.audio_config.is_some().then(|| "audio".into()))
            .chain(args.vision_config.is_some().then(|| "vision".into()))
            .collect(),
        dependency_inputs: vec![ComponentDomain::HiddenStates; media + 1],
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    });
    let mut previous = "assembly".to_owned();
    for layer in 0..args.text_config.num_hidden_layers as usize {
        let id = format!("decoder.{layer}");
        units.push(ComponentSpec {
            id: id.clone(),
            kind: ComponentKind::Decoder,
            external_inputs: vec![],
            dependencies: vec![previous],
            dependency_inputs: vec![ComponentDomain::HiddenStates],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Decoder,
        });
        previous = id;
    }
    units.push(ComponentSpec {
        id: "output".into(),
        kind: ComponentKind::OutputProjection,
        external_inputs: vec![],
        dependencies: vec![previous.clone()],
        dependency_inputs: vec![ComponentDomain::HiddenStates],
        output: ComponentDomain::Logits,
        residency: ComponentResidencyClass::Static,
    });
    let mut outputs = vec!["output".to_owned()];
    if args
        .mtp_config
        .as_ref()
        .is_some_and(|config| config.num_nextn_predict_layers > 0)
    {
        units.push(ComponentSpec {
            id: "prediction".into(),
            kind: ComponentKind::Prediction,
            external_inputs: vec![ComponentDomain::TokenIds],
            dependencies: vec![previous],
            dependency_inputs: vec![ComponentDomain::HiddenStates],
            output: ComponentDomain::Logits,
            residency: ComponentResidencyClass::Draft,
        });
        outputs.push("prediction".into());
    }
    ComponentGraph::new(units, outputs)
}

/// Declares key/value state plus all four bounded convolution histories from
/// the same normalized layer schedule consumed by execution.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, StateError> {
    text_state_layout(&args.text_config)
}

/// Declares rank-local text state from the per-layer geometries produced by
/// tensor-parallel placement.
pub fn parallel_state_layout(
    args: &ModelArgs,
    local_layers: &[super::TextArgs],
) -> Result<StateLayout, StateError> {
    if local_layers.len() != args.text_config.layer_schedule.len() {
        return Err(StateError::InvalidResidency(format!(
            "Inkling parallel state has {} local layers, expected {}",
            local_layers.len(),
            args.text_config.layer_schedule.len()
        )));
    }
    let policies = local_layers
        .iter()
        .zip(args.text_config.layer_schedule.iter())
        .map(|(text, policy)| text_state_policy(text, *policy))
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(policies.len(), policies)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}

/// Declares the embedded predictor's KV and convolution state using the same
/// ordinary decoder-layer geometry as execution.
pub fn mtp_state_layout(args: &ModelArgs) -> Result<Option<StateLayout>, StateError> {
    let Some(config) = args
        .mtp_config
        .as_ref()
        .filter(|config| config.num_nextn_predict_layers > 0)
    else {
        return Ok(None);
    };
    let count = usize::try_from(config.num_nextn_predict_layers)
        .map_err(|_| StateError::InvalidResidency("Inkling MTP depth is negative".into()))?;
    let mut policies = Vec::with_capacity(count);
    for depth in 0..count {
        let attention = if config.local_layer_ids.contains(&depth) {
            args.text_config
                .layer_schedule
                .iter()
                .find_map(|policy| {
                    policy
                        .attention
                        .window()
                        .map(|window| eredu_core::AttentionPolicy::Sliding { window })
                })
                .ok_or_else(|| {
                    StateError::InvalidResidency(
                        "Inkling MTP requests local attention without a sliding window".into(),
                    )
                })?
        } else {
            eredu_core::AttentionPolicy::Full
        };
        let text = super::mtp::mtp_text_args(&args.text_config, config, attention)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?;
        policies.push(text_state_policy(
            &text,
            *text
                .layer_schedule
                .get(0)
                .expect("one-layer MTP geometry has one policy"),
        )?);
    }
    StateLayout::new(
        LayerSchedule::new(policies.len(), policies)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
    .map(Some)
}

/// Declares the complete target plus checkpoint-embedded prediction layout.
///
/// Both segments persist across accepted generation steps, but remain
/// independently identified so persistence and future lifecycle policy consume
/// architecture-owned boundaries rather than a flattened backend schedule.
pub fn composite_state_layout(
    target: &StateLayout,
    prediction: Option<&StateLayout>,
) -> Result<StateLayout, StateError> {
    let prediction_len = prediction.map_or(0, StateLayout::len);
    let total_len = target.len().checked_add(prediction_len).ok_or_else(|| {
        StateError::InvalidResidency("Inkling composite state layer count overflowed".into())
    })?;
    let layers = target
        .layers()
        .iter()
        .cloned()
        .chain(
            prediction
                .into_iter()
                .flat_map(|layout| layout.layers().iter().cloned()),
        )
        .collect::<Vec<_>>();
    let mut segments = vec![StateSegmentSpec::new(
        TARGET_STATE_SEGMENT,
        0..target.len(),
        StateSegmentLifetime::Persistent,
    )?];
    if prediction_len > 0 {
        segments.push(StateSegmentSpec::new(
            PREDICTION_STATE_SEGMENT,
            target.len()..total_len,
            StateSegmentLifetime::Persistent,
        )?);
    }
    StateLayout::segmented(
        LayerSchedule::new(total_len, layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
        segments,
    )
}

fn text_state_layout(text: &super::TextArgs) -> Result<StateLayout, StateError> {
    let layers = text
        .layer_schedule
        .iter()
        .map(|policy| text_state_policy(text, *policy))
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}

fn text_state_policy(
    text: &super::TextArgs,
    policy: super::LayerPolicy,
) -> Result<LayerCachePolicy, StateError> {
    let history = text.sconv_kernel_size - 1;
    let fixed = |value| {
        StateTensorDimension::fixed(value)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))
    };
    let local = policy.attention.window().is_some();
    let kv_width = text
        .key_value_heads(local)
        .checked_mul(text.attention_head_dim(local))
        .ok_or_else(|| StateError::InvalidResidency("Inkling KV width overflow".into()))?;
    let tensors = [kv_width, kv_width, text.hidden_size, text.hidden_size]
        .into_iter()
        .enumerate()
        .map(|(slot, channels)| {
            StateTensorPolicy::new(
                StateTensorRole::Convolution { slot: slot as u32 },
                vec![
                    StateTensorDimension::Batch,
                    fixed(history)?,
                    fixed(channels)?,
                ],
                StateTensorDtype::Floating,
                MutableStateResidency::AlwaysDeviceMutable,
            )
            .map_err(|error| StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerCachePolicy::key_value_with_fixed_state(
        policy.attention,
        text.key_value_heads(local),
        text.attention_head_dim(local),
        tensors,
    )
    .map_err(|error| StateError::InvalidResidency(error.to_string()))
}

#[cfg(test)]
mod tests {
    use eredu_core::cache::StateTensorRole;

    use super::*;

    #[test]
    fn every_layer_declares_four_convolution_slots() {
        let args = ModelArgs::from_hf_json(
            br#"{"text_config":{"hidden_size":16,"num_hidden_layers":1,"vocab_size":32,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"d_rel":2,
            "intermediate_size":24,"n_routed_experts":2,"num_experts_per_tok":1,
            "n_shared_experts":1}}"#,
        )
        .unwrap();
        let layout = state_layout(&args).unwrap();
        let slots = layout
            .layers()
            .get(0)
            .expect("one layer")
            .fixed_state()
            .iter()
            .filter(|tensor| matches!(tensor.role, StateTensorRole::Convolution { .. }))
            .count();
        assert_eq!(slots, 4);
    }

    #[test]
    fn mtp_state_uses_declared_override_geometry_and_four_convolutions() {
        let args = ModelArgs::from_hf_json(
            br#"{"text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":32,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"d_rel":2,
            "intermediate_size":24,"n_routed_experts":2,"num_experts_per_tok":1,
            "n_shared_experts":1,"layer_types":["full_attention","sliding_attention"],
            "sliding_window":8},"mtp_config":{"num_nextn_predict_layers":2,
            "local_layer_ids":[1],"num_key_value_heads":2,"head_dim":4,
            "swa_num_key_value_heads":1,"swa_head_dim":8,"sconv_kernel_size":3}}"#,
        )
        .unwrap();
        let layout = mtp_state_layout(&args).unwrap().expect("MTP layout");
        assert_eq!(layout.len(), 2);
        assert!(layout
            .layers()
            .get(0)
            .unwrap()
            .attention()
            .unwrap()
            .window()
            .is_none());
        assert_eq!(
            layout
                .layers()
                .get(1)
                .unwrap()
                .attention()
                .unwrap()
                .window()
                .unwrap()
                .get(),
            8
        );
        for layer in layout.layers().iter() {
            assert_eq!(layer.fixed_state().len(), 4);
            assert_eq!(
                layer.fixed_state()[0].shape[1],
                StateTensorDimension::fixed(2).unwrap()
            );
        }

        let target = state_layout(&args).unwrap();
        let composite = composite_state_layout(&target, Some(&layout)).unwrap();
        assert_eq!(composite.len(), 4);
        assert_eq!(composite.segments().len(), 2);
        assert_eq!(composite.segments()[0].id().as_str(), TARGET_STATE_SEGMENT);
        assert_eq!(composite.segments()[0].layers(), 0..2);
        assert_eq!(
            composite.segments()[0].lifetime(),
            StateSegmentLifetime::Persistent
        );
        assert_eq!(
            composite.segments()[1].id().as_str(),
            PREDICTION_STATE_SEGMENT
        );
        assert_eq!(composite.segments()[1].layers(), 2..4);
        assert_eq!(
            composite.segments()[1].lifetime(),
            StateSegmentLifetime::Persistent
        );
    }
}
