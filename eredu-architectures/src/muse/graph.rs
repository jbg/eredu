//! Typed Muse-Glimmer component and mutable-state graphs.

use eredu_core::{cache::LayerCachePolicy, LayerSchedule};
use eredu_runtime::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec, StateError, StateLayout,
};

use super::DecoderConfig;

/// Canonical group identities retain vision zero and decoder one. Waveform
/// units form an independent root feeding the same ordered decoder assembly.
pub(crate) fn execution_graph(
    args: &DecoderConfig,
) -> Result<eredu_runtime::ExecutionGraph, eredu_nn::Error> {
    use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};
    if args.audio_config.is_none() {
        return ExecutionGraph::chain([
            super::model::VISION_EXECUTION_GROUP,
            super::model::TEXT_EXECUTION_GROUP,
        ])
        .map_err(eredu_nn::Error::backend);
    }
    ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(super::model::VISION_EXECUTION_GROUP),
            ExecutionGroupSpec::with_dependencies(
                super::model::TEXT_EXECUTION_GROUP,
                [
                    super::model::VISION_EXECUTION_GROUP,
                    super::audio::execution::AUDIO_EXECUTION_GROUP,
                ],
            ),
            ExecutionGroupSpec::root(super::audio::execution::AUDIO_EXECUTION_GROUP),
        ],
        super::model::TEXT_EXECUTION_GROUP,
    )
    .map_err(eredu_nn::Error::backend)
}

/// Builds the vision/assembly/decoder/output graph.
pub fn component_graph(args: &DecoderConfig) -> Result<ComponentGraph, ComponentGraphError> {
    let mut units = vec![ComponentSpec {
        id: "embedding".into(),
        kind: ComponentKind::StaticText,
        external_inputs: vec![ComponentDomain::TokenIds],
        dependencies: vec![],
        dependency_inputs: vec![],
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    }];
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
    units.push(ComponentSpec {
        id: "assembly".into(),
        kind: ComponentKind::Assembly,
        external_inputs: vec![],
        dependencies: std::iter::once("embedding".into())
            .chain(args.vision_config.is_some().then(|| "vision".into()))
            .chain(args.audio_config.is_some().then(|| "audio".into()))
            .collect(),
        dependency_inputs: vec![
            ComponentDomain::HiddenStates;
            1 + usize::from(args.vision_config.is_some())
                + usize::from(args.audio_config.is_some())
        ],
        output: ComponentDomain::HiddenStates,
        residency: ComponentResidencyClass::Static,
    });
    let mut previous = "assembly".to_owned();
    for layer in 0..args.num_hidden_layers as usize {
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
        dependencies: vec![previous],
        dependency_inputs: vec![ComponentDomain::HiddenStates],
        output: ComponentDomain::Logits,
        residency: ComponentResidencyClass::Static,
    });
    ComponentGraph::new(units, ["output"])
}

/// Declares the exact per-layer full/sliding key/value state geometry.
pub fn state_layout(args: &DecoderConfig) -> Result<StateLayout, StateError> {
    let mut layers = args
        .attention_schedule
        .iter()
        .map(|policy| {
            LayerCachePolicy::key_value(*policy, args.num_key_value_heads, args.head_dim)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(audio) = &args.audio_config {
        let audio = super::audio::execution::state_layout(audio)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?;
        layers.extend(audio.layers().iter().cloned());
    }
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}
