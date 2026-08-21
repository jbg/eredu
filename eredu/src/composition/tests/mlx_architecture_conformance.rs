//! MLX composition conformance for backend-neutral architecture operators.

use eredu_runtime::{LayeredArchitecture, LayeredForwardState};
use safemlx::{Array, Device, DeviceType, ExecutionContext};

use crate::backend::mlx::{
    nn::shared::MlxBackend,
    runtime::cache::state::{MlxHybridState, MlxKeyValueState},
};

#[test]
#[ignore = "initializes MLX and executes a neutral family forward"]
fn neutral_gemma4_text_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::gemma4::LayeredModel<MlxBackend>;
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"gemma4","tie_word_embeddings":true,
          "text_config":{"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
            "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":true,
            "attention_k_eq_v":false,"layer_types":["full_attention"]}
        }))
        .unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::gemma4::state_layout(&args.text).unwrap())
            .unwrap();
    let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [eredu_architectures::gemma4::DecoderInputPart::Text(&tokens)];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_forward(
        &mut architecture,
        eredu_architectures::gemma4::ModelInput {
            parts: &parts,
            vision: None,
            audio: None,
            per_layer_tokens: None,
            mask: None,
        },
        &mut state,
        stream,
    )
    .unwrap();
    let mut hidden =
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_execution_group(
            &mut architecture,
            2,
            &initial,
            &[],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::build_unit(
        &architecture,
        2,
        0,
        stream,
    )
    .unwrap();
    hidden = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::forward_unit(
        &mut architecture,
        2,
        0,
        &mut unit,
        &hidden,
        &mut state,
        &mut context,
        stream,
    )
    .unwrap();
    let logits = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::finish_forward(
        &mut architecture,
        &hidden,
        &mut state,
        &context,
        stream,
    )
    .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 32]);
    logits.evaluated().unwrap();
}

#[test]
#[ignore = "initializes MLX and executes a neutral family forward"]
fn neutral_inkling_text_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::inkling::LayeredModel<MlxBackend>;
    let args = eredu_architectures::inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"inkling_mm_model","image_token_id":60,"audio_token_id":61,
          "text_config":{"hidden_size":16,"num_hidden_layers":1,"vocab_size":64,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "layer_types":["full_attention"],"mlp_layer_types":["dense"],
            "sconv_kernel_size":2,"d_rel":2,"rel_extent":16,
            "intermediate_size":32,"dense_intermediate_size":32,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1}
        }))
        .unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::inkling::state_layout(&args).unwrap()).unwrap();
    let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [eredu_architectures::inkling::DecoderInputPart::Text(
        &tokens,
    )];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_forward(
        &mut architecture,
        eredu_architectures::inkling::ModelInput {
            parts: &parts,
            vision_patches: None,
            audio: None,
        },
        &mut state,
        stream,
    )
    .unwrap();
    let mut hidden =
        <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_execution_group(
            &mut architecture,
            1,
            &initial,
            &[&initial],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::build_unit(
        &architecture,
        1,
        0,
        stream,
    )
    .unwrap();
    hidden = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::forward_unit(
        &mut architecture,
        1,
        0,
        &mut unit,
        &hidden,
        &mut state,
        &mut context,
        stream,
    )
    .unwrap();
    let logits = <Architecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::finish_forward(
        &mut architecture,
        &hidden,
        &mut state,
        &context,
        stream,
    )
    .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 64]);
    logits.evaluated().unwrap();
}

#[test]
#[ignore = "initializes MLX and executes a neutral family forward"]
fn neutral_muse_glimmer_text_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::muse_glimmer::LayeredModel<MlxBackend>;
    let args = eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],
            "model_type":"muse_glimmer","image_token_id":30,"video_token_id":29,
            "out_hidden_size":16,"projector_hidden_size":8,
            "vision_config":{"model_type":"muse_glimmer_vision","hidden_act":"gelu",
              "hidden_size":4,"intermediate_size":8,"num_attention_heads":1,
              "num_hidden_layers":1,"patch_size":2,"patch_temporal":2,"merge_size":2,
              "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,
              "layer_norm_eps":1e-5,"layer_types":["full_attention"],
              "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}},
            "text_config":{"model_type":"muse_glimmer_text","hidden_size":8,
              "num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,
              "num_key_value_heads":1,"head_dim":4,"vocab_size":32,
              "max_position_embeddings":64,"rms_norm_eps":1e-5,"post_norm_eps":1e-8,
              "rope_theta":10000.0,"layer_types":["full_attention"],
              "layer_rope_theta":[0.0],"sliding_window":8,"tie_word_embeddings":false,
              "hidden_act":"silu","attention_dropout":0.0,"attention_bias":false,
              "mlp_bias":false,"qk_scale_factor":1.0,"output_multiplier":1.0,
              "final_logit_softcapping":20.0}
        }))
        .unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::muse_glimmer::state_layout(&args).unwrap())
            .unwrap();
    let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [eredu_architectures::muse_glimmer::DecoderInputPart::Text(
        &tokens,
    )];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::begin_forward(
        &mut architecture,
        eredu_architectures::muse_glimmer::ModelInput {
            parts: &parts,
            vision: None,
            mask: None,
        },
        &mut state,
        stream,
    )
    .unwrap();
    let mut hidden =
        <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::begin_execution_group(
            &mut architecture,
            1,
            &initial,
            &[&initial],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit = <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::build_unit(
        &architecture,
        1,
        0,
        stream,
    )
    .unwrap();
    hidden = <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::forward_unit(
        &mut architecture,
        1,
        0,
        &mut unit,
        &hidden,
        &mut state,
        &mut context,
        stream,
    )
    .unwrap();
    let logits =
        <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::finish_forward(
            &mut architecture,
            &hidden,
            &mut state,
            &context,
            stream,
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 32]);
    logits.evaluated().unwrap();
}
