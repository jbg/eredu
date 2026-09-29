//! Typed component and mutable-state graphs for Gemma 4.

use eredu_core::cache::{
    LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_core::LayerSchedule;
use eredu_nn::AttentionStateSource;
use eredu_runtime::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec, StateError, StateLayout,
};

use super::ModelArgs;

/// Optional component topology surrounding the text decoder.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ComponentOptions {
    /// Include a vision tower and projector.
    pub vision: bool,
    /// Include an audio tower and projector.
    pub audio: bool,
    /// Include embedded multi-token prediction.
    pub prediction: bool,
    /// Include an external assistant output.
    pub assistant: bool,
}

/// Builds the architecture's storage-independent component graph.
pub fn component_graph(
    args: &ModelArgs,
    options: ComponentOptions,
) -> Result<ComponentGraph, ComponentGraphError> {
    let mut units = vec![ComponentSpec {
        id: "text.embedding".into(),
        kind: ComponentKind::StaticText,
        external_inputs: vec![ComponentDomain::TokenIds],
        dependencies: vec![],
        dependency_inputs: vec![],
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    }];
    let mut assembly_dependencies = vec!["text.embedding".to_owned()];
    if options.vision {
        units.push(ComponentSpec {
            id: "vision".into(),
            kind: ComponentKind::Vision,
            external_inputs: vec![ComponentDomain::PatchMatrix],
            dependencies: vec![],
            dependency_inputs: vec![],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Media,
        });
        assembly_dependencies.push("vision".into());
    }
    if options.audio {
        units.push(ComponentSpec {
            id: "audio".into(),
            kind: ComponentKind::Audio,
            external_inputs: vec![ComponentDomain::AudioFeatures],
            dependencies: vec![],
            dependency_inputs: vec![],
            output: ComponentDomain::HiddenStates,
            residency: ComponentResidencyClass::Media,
        });
        assembly_dependencies.push("audio".into());
    }
    units.push(ComponentSpec {
        id: "assembly".into(),
        kind: ComponentKind::Assembly,
        external_inputs: vec![],
        dependency_inputs: vec![ComponentDomain::HiddenStates; assembly_dependencies.len()],
        dependencies: assembly_dependencies,
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    });
    let mut previous = "assembly".to_owned();
    for layer in 0..args.num_hidden_layers() {
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
    if options.prediction {
        units.push(ComponentSpec {
            id: "prediction".into(),
            kind: ComponentKind::Prediction,
            external_inputs: vec![ComponentDomain::TokenIds],
            dependencies: vec![previous.clone()],
            dependency_inputs: vec![ComponentDomain::HiddenStates],
            output: ComponentDomain::Logits,
            residency: ComponentResidencyClass::Draft,
        });
        outputs.push("prediction".into());
    }
    if options.assistant {
        units.push(ComponentSpec {
            id: "assistant".into(),
            kind: ComponentKind::Assistant,
            external_inputs: vec![ComponentDomain::TokenIds],
            dependencies: vec![previous],
            dependency_inputs: vec![ComponentDomain::HiddenStates],
            output: ComponentDomain::Logits,
            residency: ComponentResidencyClass::Draft,
        });
        outputs.push("assistant".into());
    }
    ComponentGraph::new(units, outputs)
}

/// Builds exact key/value, shared, and prefix state geometry from the
/// same layer schedule used by execution.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, StateError> {
    let prefix = StateTensorPolicy::new(
        StateTensorRole::PrefixEmbedding,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::PrefixTokens,
            StateTensorDimension::fixed(args.hidden_size)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
        ],
        StateTensorDtype::Floating,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .map_err(|error| StateError::InvalidResidency(error.to_string()))?
    .optional();
    let layers = args
        .layer_schedule
        .iter()
        .enumerate()
        .map(|(layer, policy)| {
            if policy.key_value == AttentionStateSource::Shared {
                return Ok(LayerCachePolicy::NoState);
            }
            let kv_heads = policy.num_key_value_heads.get() as i32;
            let head_dim = policy.head_dim.get() as i32;
            // Sharing the K/V projection does not share its cache payload:
            // learned key normalization and RoPE differ from value normalization.
            if layer == 0 {
                LayerCachePolicy::key_value_with_state(
                    policy.attention,
                    kv_heads,
                    head_dim,
                    vec![prefix.clone()],
                    Vec::new(),
                )
            } else {
                LayerCachePolicy::key_value(policy.attention, kv_heads, head_dim)
            }
            .map_err(|error| StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}
