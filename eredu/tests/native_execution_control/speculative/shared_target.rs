//! Prepared shared-target admission and prospective adapter workspace.
use super::*;
use eredu_core::decoding::{DecodingMode, SpeculationOptionsRequest};

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn shared_target_prepared_modes_preserve_output_and_pending_adapter_forecasts() {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"nemotron_labs_diffusion", "hidden_size":6, "head_dim":4,
        "num_attention_heads":2, "num_key_value_heads":1, "num_hidden_layers":2,
        "intermediate_size":7, "vocab_size":64, "rms_norm_eps":0.00001,
        "block_size":3, "mask_token_id":0, "hidden_act":"silu", "dlm_paradigm":"bidirectional",
        "tie_word_embeddings":false, "attention_bias":false, "mlp_bias":false,
        "max_position_embeddings":64, "sliding_window":null,
        "rope_parameters":{"rope_type":"yarn","factor":2.0,"rope_theta":10000.0,
            "original_max_position_embeddings":4,"beta_fast":32.0,"beta_slow":1.0,
            "mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}
    });
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    let adapter = fixture(false);
    let mut declaration: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::DRAFT_ADAPTER_CONFIG_JSON,
    )
    .unwrap();
    declaration["r"] = 2.into();
    declaration["lora_alpha"] = 4.into();
    let bytes = serde_json::to_vec(&declaration).unwrap();
    std::fs::write(adapter.0.join("adapter_config.json"), &bytes).unwrap();
    let family =
        eredu_architectures::nemotron_labs_diffusion::FamilyConfig::from_hf_value(&config).unwrap();
    let declaration =
        eredu_architectures::nemotron_labs_diffusion::adapter::DraftAdapterConfig::from_json(
            &bytes,
            "nvidia/Nemotron-Labs-Diffusion-3B",
        )
        .unwrap();
    let mut schema = eredu_architectures::nemotron_labs_diffusion::adapter::safetensors_plan(
        &family,
        &declaration,
    )
    .unwrap();
    for tensor in &mut schema.common_tensors {
        tensor.key = tensor.aliases[0].clone();
    }
    write_tensor_plan(&adapter.0, &schema);
    std::fs::rename(
        adapter.0.join("model.safetensors"),
        adapter.0.join("adapter_model.safetensors"),
    )
    .unwrap();
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_draft_adapter(eredu_core::DraftAdapterSource::new(
            &adapter.0,
            "nvidia/Nemotron-Labs-Diffusion-3B",
        ));
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &plan).unwrap();
    let (model, _) = loaded.parts_mut();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"left right"})],
            add_generation_prompt: true,
            tool_choice: ToolChoice::None,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(7),
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let request = |settings| PreparedChatSpeculativeGenerationRequest {
        control_handle: None,
        input: PreparedChatInput::token_ids(&chat, vec![1, 4, 2]),
        drafting: eredu_core::SpeculativeDraft::Target,
        settings,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_| {},
    };
    let expected = model
        .generate_prepared_text_speculative(request(settings))
        .unwrap();
    let speculation = DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
            draft_adapter: true,
            ..Default::default()
        }
        .try_into()
        .unwrap(),
    );
    let mut returned_to_ar = false;
    let actual = model
        .with_controlled_text_speculative(request(settings), Default::default(), |session| {
            assert!(session.step()?.is_some());
            assert_eq!(
                session.decoding_state().unwrap().active,
                DecodingMode::Autoregressive
            );
            let spent = session.snapshot_usage();
            let initial = session
                .forecast_remaining_generation(3, &Default::default())
                .unwrap();
            assert!(
                initial
                    .speculative
                    .shared_backbone
                    .as_ref()
                    .unwrap()
                    .adapter
                    .is_none()
            );
            session.request_decoding_mode(speculation)?;
            let pending = session
                .forecast_remaining_generation(3, &Default::default())
                .unwrap();
            assert_eq!(pending.speculative.max_draft_tokens, 2);
            assert!(
                pending
                    .speculative
                    .shared_backbone
                    .as_ref()
                    .unwrap()
                    .adapter
                    .is_some()
            );
            assert_eq!(session.snapshot_usage(), spent);
            assert_eq!(session.decoding_state().unwrap().pending, Some(speculation));
            while session.step()?.is_some() {
                let state = session.decoding_state().unwrap();
                if !returned_to_ar
                    && state.active == speculation
                    && state.pending.is_none()
                    && session.token_ids().len() > 1
                {
                    session.request_decoding_mode(DecodingMode::Autoregressive)?;
                    returned_to_ar = true;
                }
            }
            Ok(())
        })
        .unwrap();
    assert!(returned_to_ar);
    assert_eq!(actual.token_ids(), expected.token_ids());
    let diffusion = PreparedChatGenerationSettings {
        decoding_mode: Some(DecodingMode::Diffusion { policy: None }),
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..settings.overrides
        },
        ..settings
    };
    let expected = model
        .generate_prepared_text_speculative(request(diffusion))
        .unwrap();
    let actual = model
        .with_controlled_text_speculative(request(diffusion), Default::default(), |session| {
            while session.step()?.is_some() {}
            Ok(())
        })
        .unwrap();
    assert_eq!(actual.token_ids(), expected.token_ids());
    assert_eq!(actual.token_ids().len(), 2);
}
