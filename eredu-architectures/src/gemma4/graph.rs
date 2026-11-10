//! Typed component and mutable-state graphs for Gemma 4.

use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
    StateTensorRole,
};
use eredu_core::LayerSchedule;
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
pub fn state_layout(config: &super::FamilyConfig) -> Result<StateLayout, StateError> {
    declared_state_layout(&config.text, config.attention_scopes(), None)
}

pub(crate) fn tensor_state_layout(
    config: &super::FamilyConfig,
    rank: eredu_core::ParallelRankTopology,
) -> Result<StateLayout, StateError> {
    declared_state_layout(&config.text, config.attention_scopes(), Some(rank))
}

pub(crate) fn local_state_layout(
    config: &super::FamilyConfig,
    blocks: &[ModelArgs],
) -> Result<StateLayout, StateError> {
    if blocks.is_empty() || blocks.len() != config.text.num_hidden_layers() {
        return Err(StateError::InvalidResidency(
            "local attention scopes do not cover the declared stack".into(),
        ));
    }
    let attention = blocks
        .iter()
        .enumerate()
        .map(|(layer, args)| {
            config
                .attention(layer)
                .and_then(|spec| spec.for_invocation(args, layer))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| StateError::InvalidResidency(e.to_string()))?;
    declared_state_layout(&config.text, &attention, None)
}

fn declared_state_layout(
    args: &ModelArgs,
    attention: &[super::text::specs::AttentionSpec],
    rank: Option<eredu_core::ParallelRankTopology>,
) -> Result<StateLayout, StateError> {
    let layers = args
        .layer_schedule
        .iter()
        .enumerate()
        .map(|(layer, _policy)| {
            args.attention_state_owner(layer)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))?;
            let spec = attention.get(layer).ok_or_else(|| {
                StateError::InvalidResidency("missing declared attention state scope".into())
            })?;
            let tensors = if layer == 0 {
                vec![StateTensorPolicy::new(
                    StateTensorRole::PrefixEmbedding,
                    vec![
                        StateTensorDimension::Batch,
                        StateTensorDimension::PrefixTokens,
                        StateTensorDimension::fixed(spec.query.input)
                            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(|error| StateError::InvalidResidency(error.to_string()))?
                .optional()]
            } else {
                Vec::new()
            };
            let range = rank
                .map(|rank| {
                    eredu_core::balanced_contiguous_range(
                        spec.geometry.key_value_heads() as usize,
                        rank.tensor_parallel_size(),
                        rank.tensor_parallel_rank(),
                        false,
                    )
                })
                .transpose()
                .map_err(|error| StateError::InvalidResidency(error.to_string()))?;
            spec.state_policy(range, tensors)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}
