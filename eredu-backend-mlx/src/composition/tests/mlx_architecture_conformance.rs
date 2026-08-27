//! MLX composition conformance for backend-neutral architecture operators.

use eredu_nn::Tensor;
use eredu_runtime::{DeviceState, LayeredArchitecture, LayeredForwardState};
use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType, ExecutionContext};
use std::sync::OnceLock;

use crate::backend::{
    nn::{shared::MlxNeuralBackend, tensor::TokenValidationScope},
    runtime::cache::{
        kv::CompressedLatentCache,
        state::{MlxHybridState, MlxKeyValueState, MlxPoolingAttentionStateFactory},
    },
};
use crate::MlxTensor;

#[test]
fn every_layered_family_exposes_authoritative_geometry_and_parameter_binding() {
    fn assert_bindable<A: eredu_runtime::ArchitectureParameters<MlxNeuralBackend>>() {}

    assert_bindable::<eredu_architectures::llama::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::muse_glimmer::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>>(
    );
    assert_bindable::<eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::deepseek::v3::Model<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>>();
    assert_bindable::<eredu_architectures::moshi::LayeredModel<MlxNeuralBackend>>();
}

fn mlx_execution() -> Option<ExecutionContext> {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    let available = AVAILABLE.get_or_init(|| {
        #[cfg(feature = "metal")]
        {
            match safemlx::metal::is_available() {
                Ok(true) => {}
                Ok(false) => return false,
                Err(error) => panic!("MLX Metal availability probe failed: {error}"),
            }
        }
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        match safemlx::ops::zeros::<f32>(&[1], execution.stream())
            .and_then(|probe| probe.evaluated().map(|_| ()))
        {
            Ok(()) => true,
            Err(error) if error.what().contains("No Metal device available") => false,
            Err(error) => panic!("MLX execution probe failed: {error}"),
        }
    });
    if !available {
        eprintln!("skipping MLX execution conformance: native device initialization failed");
        return None;
    }
    Some(ExecutionContext::new(Device::new(DeviceType::Cpu, 0)))
}

macro_rules! execute_group {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $context:expr, $group:expr, $initial:expr, $dependencies:expr, $stream:expr) => {{
        let mut hidden = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_execution_group(
            &mut $architecture,
            $group,
            $initial,
            $dependencies,
            &mut $state,
            &mut $context,
            $stream,
        )
        .unwrap();
        let unit_count =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::group_unit_count(
                &$architecture,
                $group,
            )
            .unwrap();
        for index in 0..unit_count {
            let mut unit =
                <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::build_unit(
                    &$architecture,
                    $group,
                    index,
                    $stream,
                )
                .unwrap();
            hidden =
                <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::forward_unit(
                    &mut $architecture,
                    $group,
                    index,
                    &mut unit,
                    &hidden,
                    &mut $state,
                    &mut $context,
                    $stream,
                )
                .unwrap();
        }
        hidden = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::complete_execution_group(
            &mut $architecture,
            $group,
            &hidden,
            &mut $state,
            &mut $context,
            $stream,
        )
        .unwrap();
        hidden
    }};
}

macro_rules! execute_target_group {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $input:expr, $shape:expr, $stream:expr) => {{
        let token_validation_scope = TokenValidationScope::begin().unwrap();
        let LayeredForwardState {
            hidden: initial,
            mut context,
        } = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_forward(
            &mut $architecture,
            $input,
            &mut $state,
            $stream,
        )
        .unwrap();
        let hidden = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            0,
            &initial,
            &[],
            $stream
        );
        let logits =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::finish_forward(
                &mut $architecture,
                &hidden,
                &mut $state,
                &context,
                $stream,
            )
            .unwrap();
        assert_eq!(logits.shape(), $shape);
        let token_validations = token_validation_scope.finish();
        async_eval_with_event(std::iter::once(logits.as_array()).chain(token_validations.arrays()))
            .unwrap()
            .synchronize()
            .unwrap();
        token_validations.validate_completed().unwrap();
    }};
}

macro_rules! execute_vision_text_groups {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $input:expr, $shape:expr, $stream:expr) => {{
        let token_validation_scope = TokenValidationScope::begin().unwrap();
        let LayeredForwardState {
            hidden: initial,
            mut context,
        } = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_forward(
            &mut $architecture,
            $input,
            &mut $state,
            $stream,
        )
        .unwrap();
        let vision = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            0,
            &initial,
            &[],
            $stream
        );
        let hidden = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            1,
            &initial,
            &[&vision],
            $stream
        );
        let logits =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::finish_forward(
                &mut $architecture,
                &hidden,
                &mut $state,
                &context,
                $stream,
            )
            .unwrap();
        assert_eq!(logits.shape(), $shape);
        let token_validations = token_validation_scope.finish();
        async_eval_with_event(std::iter::once(logits.as_array()).chain(token_validations.arrays()))
            .unwrap()
            .synchronize()
            .unwrap();
        token_validations.validate_completed().unwrap();
    }};
}

#[test]
fn neutral_gemma4_text_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>;
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
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::gemma4::state_layout(&args.text).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let parts = [eredu_architectures::gemma4::DecoderInputPart::Text(&tokens)];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_forward(
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
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_execution_group(
            &mut architecture,
            2,
            &initial,
            &[],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::build_unit(
            &architecture,
            2,
            0,
            stream,
        )
        .unwrap();
    hidden = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::forward_unit(
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
    let logits =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::finish_forward(
            &mut architecture,
            &hidden,
            &mut state,
            &context,
            stream,
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 32]);
    logits.as_array().evaluated().unwrap();
}

#[test]
fn neutral_inkling_text_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>;
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
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::inkling::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let parts = [eredu_architectures::inkling::DecoderInputPart::Text(
        &tokens,
    )];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_forward(
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
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_execution_group(
            &mut architecture,
            1,
            &initial,
            &[&initial],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::build_unit(
            &architecture,
            1,
            0,
            stream,
        )
        .unwrap();
    hidden = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::forward_unit(
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
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::finish_forward(
            &mut architecture,
            &hidden,
            &mut state,
            &context,
            stream,
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 64]);
    logits.as_array().evaluated().unwrap();
}

#[test]
fn neutral_muse_glimmer_text_only_forward_monomorphizes_on_mlx() {
    type Architecture = eredu_architectures::muse_glimmer::LayeredModel<MlxNeuralBackend>;
    let mut args = eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(
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
    args.vision_config = None;
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::muse_glimmer::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let parts = [eredu_architectures::muse_glimmer::DecoderInputPart::Text(
        &tokens,
    )];
    let LayeredForwardState {
        hidden: initial,
        mut context,
    } = <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::begin_forward(
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
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::begin_execution_group(
            &mut architecture,
            1,
            &initial,
            &[&initial],
            &mut state,
            &mut context,
            stream,
        )
        .unwrap();
    let mut unit =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::build_unit(
            &architecture,
            1,
            0,
            stream,
        )
        .unwrap();
    hidden =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::forward_unit(
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
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::finish_forward(
            &mut architecture,
            &hidden,
            &mut state,
            &context,
            stream,
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 2, 32]);
    logits.as_array().evaluated().unwrap();
}

#[test]
fn neutral_llama_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::llama::model_args_from_config_value(&serde_json::json!({
        "model_type":"llama","hidden_size":16,"num_hidden_layers":1,
        "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":4,"rms_norm_eps":1e-5,"vocab_size":32,
        "max_position_embeddings":64,"rope_theta":10000.0,"tie_word_embeddings":true
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::llama::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::llama::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
        "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":4,"rms_norm_eps":1e-6,"vocab_size":32,
        "max_position_embeddings":64,"rope_theta":1000000.0,"tie_word_embeddings":true
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::qwen::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::qwen::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen3_next_hybrid_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>;
    let args =
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_next","vocab_size":32,"hidden_size":16,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
            "num_experts":0,"linear_conv_kernel_dim":2,
            "linear_key_head_dim":4,"linear_value_head_dim":4,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "layer_types":["linear_attention","full_attention"],
            "rope_theta":1000000.0,"partial_rotary_factor":0.5
        }))
        .unwrap()
        .text;
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::qwen::hybrid::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::qwen::hybrid::EmbeddedInput::Target {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen35_conditional_forward_executes_on_mlx() {
    type Architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>;
    let args =
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_5","image_token_id":30,"video_token_id":31,
            "text_config":{
                "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":16,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
                "linear_conv_kernel_dim":2,"linear_key_head_dim":4,
                "linear_value_head_dim":4,"linear_num_key_heads":2,
                "linear_num_value_heads":4,
                "layer_types":["linear_attention","full_attention"],
                "tie_word_embeddings":false
            },
            "vision_config":{
                "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
                "num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16
            }
        }))
        .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = MlxHybridState::device(
        eredu_architectures::qwen::hybrid::state_layout(&args.text).unwrap(),
    )
    .unwrap();
    let text_tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let image_tokens = MlxTensor::from_array(Array::from_slice(&[30_u32], &[1, 1]));
    let projected_image_tokens = MlxTensor::from_array(Array::from_slice(&[0_u32], &[1, 1]));
    let projected_image = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 16], &[1, 1, 16]));
    let projected_video_tokens = MlxTensor::from_array(Array::from_slice(&[0_u32], &[1, 1]));
    let projected_video = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 16], &[1, 1, 16]));
    let grid = [(1, 2, 2)];
    let pixels = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 96], &[4, 24]));
    let parts = [
        eredu_architectures::qwen::vl::InputPart::Text(&text_tokens),
        eredu_architectures::qwen::vl::InputPart::Projected {
            tokens: &projected_image_tokens,
            embeddings: &projected_image,
        },
        eredu_architectures::qwen::vl::InputPart::Projected {
            tokens: &projected_video_tokens,
            embeddings: &projected_video,
        },
        eredu_architectures::qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    execute_vision_text_groups!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::qwen::hybrid::ConditionalInput::Target {
            parts: &parts,
            pixels: Some(&pixels),
            mask: None,
        },
        &[1, 5, 32],
        stream
    );
}

#[test]
fn neutral_qwen3_vl_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::qwen::vl::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3_vl","image_token_id":30,"video_token_id":31,
        "tie_word_embeddings":false,
        "text_config":{
            "model_type":"qwen3_vl_text","hidden_size":16,
            "num_hidden_layers":1,"intermediate_size":32,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
            "rms_norm_eps":0.000001,"vocab_size":32,
            "max_position_embeddings":64,"rope_theta":1000000.0,
            "rope_scaling":{"mrope_section":[1,1,2],"mrope_interleaved":true}
        },
        "vision_config":{
            "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
            "num_position_embeddings":16,"in_channels":3,"patch_size":2,
            "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16,
            "deepstack_visual_indexes":[0]
        }
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::qwen::vl::state_layout(&args).unwrap())
            .unwrap();
    let text_tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let image_tokens = MlxTensor::from_array(Array::from_slice(&[30_u32], &[1, 1]));
    let grid = [(1, 2, 2)];
    let pixels = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 96], &[4, 24]));
    let parts = [
        eredu_architectures::qwen::vl::InputPart::Text(&text_tokens),
        eredu_architectures::qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    execute_vision_text_groups!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::qwen::vl::ModelInput {
            parts: &parts,
            pixels: Some(&pixels),
            mask: None,
        },
        &[1, 3, 32],
        stream
    );
}

#[test]
fn neutral_gpt_oss_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type":"gpt_oss","hidden_size":32,"intermediate_size":32,
        "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":8,"vocab_size":32,"num_local_experts":2,
        "num_experts_per_tok":1,"rms_norm_eps":1e-5,"sliding_window":8,
        "max_position_embeddings":64,"rope_theta":10000.0,
        "quantization_config":{"quant_method":"mxfp4"},"swiglu_limit":7.0,
        "layer_types":["full_attention"]
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture =
        eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(args.clone(), stream)
            .unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::gpt_oss::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_kimi_linear_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::kimi_linear::model_args_from_config_value(&serde_json::json!({
        "model_type":"kimi_linear","vocab_size":32,"hidden_size":12,
        "num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,
        "intermediate_size":17,"head_dim":4,"model_max_length":64,
        "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],
            "num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
        "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,
        "qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
        "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
        "routed_scaling_factor":1.0,"first_k_dense_replace":1,
        "num_expert_group":1,"topk_group":1
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::kimi_linear::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_lfm2_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type":"lfm2","vocab_size":32,"hidden_size":16,
        "intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,
        "num_key_value_heads":2,"max_position_embeddings":64,
        "layer_types":["conv","full_attention"],"conv_L_cache":3,
        "block_multiple_of":8,"block_ffn_dim_multiplier":1.0,
        "block_auto_adjust_ff_dim":true,"tie_word_embeddings":false
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::lfm2::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_nemotron_h_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h","vocab_size":32,"hidden_size":16,
        "intermediate_size":24,"num_hidden_layers":4,
        "hybrid_override_pattern":"M*-E","num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":4,"mamba_num_heads":4,
        "n_groups":2,"mamba_head_dim":4,"ssm_state_size":3,"conv_kernel":3,
        "n_routed_experts":4,"n_shared_experts":1,"moe_intermediate_size":8,
        "moe_shared_expert_intermediate_size":8,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*E"
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::nemotron_h::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::nemotron_h::EmbeddedInput::target(&tokens, None),
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_deepseek_v3_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::deepseek::v3::Model<MlxNeuralBackend>;
    type State = DeviceState<MlxNeuralBackend, CompressedLatentCache>;
    let args = eredu_architectures::deepseek::parse_v3_config(&serde_json::json!({
        "hidden_size":8,"intermediate_size":16,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":31,
        "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,
        "qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = State::create(
        eredu_architectures::deepseek::v3::state_layout(&args).unwrap(),
        |_, _| Ok::<_, std::convert::Infallible>(CompressedLatentCache::new()),
    )
    .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        State,
        architecture,
        state,
        eredu_architectures::deepseek::mtp::EmbeddedInput::target(&tokens, None),
        &[1, 2, 31],
        stream
    );
}

#[test]
fn neutral_deepseek_v4_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxPoolingAttentionState;
    let args = eredu_architectures::deepseek::parse_v4_config(&serde_json::json!({
        "hidden_size":8,"moe_intermediate_size":8,"num_hidden_layers":3,
        "num_attention_heads":2,"head_dim":4,"qk_rope_head_dim":2,
        "q_lora_rank":4,"o_lora_rank":2,"o_groups":2,"vocab_size":31,
        "max_position_embeddings":64,"sliding_window":8,
        "compress_ratios":[0,4,128,0],"index_n_heads":2,"index_head_dim":4,
        "index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":4,"num_experts_per_tok":2,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
        "norm_topk_prob":true,"num_nextn_predict_layers":1
    }))
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = MlxPoolingAttentionStateFactory::device(
        eredu_architectures::deepseek::v4::state_layout(&args).unwrap(),
    )
    .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        State,
        architecture,
        state,
        eredu_architectures::deepseek::mtp::EmbeddedInput::target(&tokens, None),
        &[1, 2, 31],
        stream
    );
}

#[test]
fn deepseek_unit_factories_use_installed_expert_realizations() {
    use eredu_nn::GatedProductExpertBankOperator;

    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 2, 1).unwrap(),
        0,
    )
    .unwrap();

    let v3_args = eredu_architectures::deepseek::parse_v3_config(&serde_json::json!({
        "hidden_size":8,"intermediate_size":16,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":31,
        "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,
        "qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "tie_word_embeddings":false
    }))
    .unwrap();
    let mut expected_v3_target =
        eredu_architectures::deepseek::v3::expert_bank_spec(&v3_args, 1).unwrap();
    expected_v3_target.expert_count = 2;
    expected_v3_target.intermediate_dimensions = 3;
    let mut expected_v3_mtp =
        eredu_architectures::deepseek::v3::expert_bank_spec(&v3_args, 2).unwrap();
    expected_v3_mtp.expert_count = 2;
    expected_v3_mtp.intermediate_dimensions = 3;
    let v3_plan = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        topology,
        std::collections::BTreeMap::from([
            (
                (eredu_runtime::ExecutionGroupId::new("target").unwrap(), 1),
                expected_v3_target.clone(),
            ),
            (
                (eredu_runtime::ExecutionGroupId::new("mtp.0").unwrap(), 0),
                expected_v3_mtp.clone(),
            ),
        ]),
    )
    .unwrap();
    let mut v3 =
        eredu_architectures::deepseek::v3::Model::<MlxNeuralBackend>::new(v3_args, stream).unwrap();
    v3.install_expert_realization(v3_plan);
    let v3_target = v3.construct_unit(0, 1, stream).unwrap();
    let eredu_architectures::deepseek::v3::Unit::Target(v3_target) = v3_target else {
        panic!("V3 target factory returned a prediction unit")
    };
    let eredu_architectures::deepseek::block::V3FeedForward::Routed(v3_target) =
        v3_target.feed_forward
    else {
        panic!("V3 sparse target factory returned a dense MLP")
    };
    assert_eq!(
        v3_target.experts.spec().expert_count,
        expected_v3_target.expert_count
    );
    assert_eq!(
        v3_target.experts.spec().intermediate_dimensions,
        expected_v3_target.intermediate_dimensions
    );
    let v3_mtp = v3.construct_unit(1, 0, stream).unwrap();
    assert_eq!(
        v3_mtp.expert_bank_spec().unwrap().expert_count,
        expected_v3_mtp.expert_count
    );

    let v4_args = eredu_architectures::deepseek::parse_v4_config(&serde_json::json!({
        "hidden_size":8,"moe_intermediate_size":8,"num_hidden_layers":3,
        "num_attention_heads":2,"head_dim":4,"qk_rope_head_dim":2,
        "q_lora_rank":4,"o_lora_rank":2,"o_groups":2,"vocab_size":31,
        "max_position_embeddings":64,"sliding_window":8,
        "compress_ratios":[0,4,128,0],"index_n_heads":2,"index_head_dim":4,
        "index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":4,"num_experts_per_tok":2,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
        "norm_topk_prob":true,"num_nextn_predict_layers":1
    }))
    .unwrap();
    let mut expected_v4 = eredu_architectures::deepseek::v4::expert_bank_spec(&v4_args, 0).unwrap();
    expected_v4.expert_count = 2;
    expected_v4.intermediate_dimensions = 3;
    let v4_plan = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        topology,
        std::collections::BTreeMap::from([(
            (eredu_runtime::ExecutionGroupId::new("target").unwrap(), 0),
            expected_v4.clone(),
        )]),
    )
    .unwrap();
    let mut v4 =
        eredu_architectures::deepseek::v4::Model::<MlxNeuralBackend>::new(v4_args, stream).unwrap();
    v4.install_expert_realization(v4_plan);
    let v4_target = v4.construct_unit(0, 0, stream).unwrap();
    let eredu_architectures::deepseek::v4::Unit::Target(v4_target) = v4_target else {
        panic!("V4 target factory returned a prediction unit")
    };
    assert_eq!(
        v4_target.feed_forward.experts.spec().expert_count,
        expected_v4.expert_count
    );
    assert_eq!(
        v4_target
            .feed_forward
            .experts
            .spec()
            .intermediate_dimensions,
        expected_v4.intermediate_dimensions
    );
}

#[test]
fn neutral_moshi_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::moshi::LayeredModel<MlxNeuralBackend>;
    let config = eredu_architectures::moshi::MoshiConfig::from_json(
        r#"{
            "model_type":"moshi","dim":16,"text_card":31,
            "n_q":4,"dep_q":3,"generated_audio_codebooks":2,"card":32,
            "num_heads":4,"num_layers":1,"dim_feedforward":24,
            "causal":true,"context":7,"max_period":10000.0,
            "positional_embedding":"rope","depformer_dim":16,
            "depformer_dim_feedforward":24,"depformer_num_heads":4,
            "depformer_num_layers":1,"depformer_context":3,
            "depformer_max_period":10000.0,"depformer_pos_emb":"none",
            "delays":[0,0,1,2,1]
        }"#,
    )
    .unwrap();
    let Some(execution) = mlx_execution() else {
        return;
    };
    let stream = execution.stream();
    let mut architecture = Architecture::new(config.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::moshi::state_layout(&config).unwrap())
            .unwrap();
    let text = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let audio_tokens = (0..4)
        .map(|_| MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2])))
        .collect::<Vec<_>>();
    let audio = audio_tokens.iter().collect::<Vec<_>>();
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::moshi::Input {
            text: &text,
            audio: &audio,
            mask: None,
        },
        &[1, 2, 31],
        stream
    );
}
