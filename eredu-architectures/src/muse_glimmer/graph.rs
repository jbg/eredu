//! Typed Muse-Glimmer component and mutable-state graphs.

use eredu_core::{cache::LayerCachePolicy, LayerSchedule};
use eredu_runtime::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec, StateError, StateLayout,
};

use super::DecoderConfig;

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
    units.push(ComponentSpec {
        id: "assembly".into(),
        kind: ComponentKind::Assembly,
        external_inputs: vec![],
        dependencies: std::iter::once("embedding".into())
            .chain(args.vision_config.is_some().then(|| "vision".into()))
            .collect(),
        dependency_inputs: vec![
            ComponentDomain::HiddenStates;
            1 + usize::from(args.vision_config.is_some())
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
    let layers = args
        .attention_schedule
        .iter()
        .map(|policy| {
            LayerCachePolicy::key_value(*policy, args.num_key_value_heads, args.head_dim)
                .map_err(|error| StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers)
            .map_err(|error| StateError::InvalidResidency(error.to_string()))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_schedule_preserves_full_and_sliding_layers() {
        let value = serde_json::json!({
            "model_type":"muse_glimmer","image_token_id":22,"video_token_id":23,
            "out_hidden_size":32,"projector_hidden_size":16,
            "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":2,
              "intermediate_size":24,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
              "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
              "rope_theta":10000.0,"layer_types":["sliding_attention","full_attention"],
              "layer_rope_theta":[10000.0,0.0],"sliding_window":8,"tie_word_embeddings":false,
              "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
            "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
              "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,
              "merge_size":2,"pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,
              "layer_norm_eps":0.00001,"hidden_act":"gelu","layer_types":["full_attention"],
              "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        });
        let args = DecoderConfig::from_hf_value(&value).unwrap();
        let layout = state_layout(&args).unwrap();
        assert!(layout
            .layers()
            .get(0)
            .unwrap()
            .attention()
            .unwrap()
            .window()
            .is_some());
        assert!(layout
            .layers()
            .get(1)
            .unwrap()
            .attention()
            .unwrap()
            .window()
            .is_none());
    }
}
