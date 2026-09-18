//! Public original raw-media construction and shared automatic/controlled delivery.
use super::*;
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView};

pub(super) fn fixture() -> Fixture {
    let root = super::fixture(false);
    let config = serde_json::json!({
        "model_type":"gemma4_unified", "tie_word_embeddings":false,
        "image_token_id":30, "audio_token_id":31,
        "text_config":{
            "model_type":"gemma4_text", "hidden_size":16, "num_hidden_layers":2,
            "intermediate_size":16, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":64,
            "max_position_embeddings":128, "attention_k_eq_v":false,
            "layer_types":["full_attention","sliding_attention"], "sliding_window":8
        },
        "vision_config":{
            "hidden_size":16, "intermediate_size":16, "num_hidden_layers":1,
            "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
            "patch_size":4, "pooling_kernel_size":2, "position_embedding_size":4,
            "rms_norm_eps":0.00001
        },
        "audio_config":{
            "hidden_size":16, "num_hidden_layers":1, "num_attention_heads":4,
            "output_proj_dims":8, "conv_kernel_size":3, "attention_chunk_size":4,
            "attention_context_left":5, "attention_context_right":0,
            "attention_invalid_logits_value":-1000000000.0, "attention_logit_cap":50.0,
            "residual_weight":0.5, "rms_norm_eps":0.00001,
            "subsampling_conv_channels":[4,8]
        }
    });
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan_with_values(&root.0, resolved.architecture.checkpoint(), |key, _| {
        if key.ends_with("input_min") || key.ends_with("output_min") {
            Some(-4.0)
        } else if key.ends_with("input_max") || key.ends_with("output_max") {
            Some(4.0)
        } else {
            None
        }
    });
    managed_fixture(root)
}

pub(super) fn with_parts<R>(run: impl FnOnce(&[HostInputPart<'_>]) -> R) -> R {
    let patches = std::array::from_fn::<_, 192, _>(|i| (i as f32 - 93.0) / 193.0);
    let audio = std::array::from_fn::<_, 512, _>(|i| (i as f32 - 251.0) / 513.0);
    let image_metadata = [
        (
            InputMetadataKey::PatchGrid,
            HostTensorView {
                shape: &[1, 3],
                values: HostTensorValues::I32(&[1, 2, 2]),
            },
        ),
        (
            InputMetadataKey::PatchPositions,
            HostTensorView {
                shape: &[1, 4, 2],
                values: HostTensorValues::I32(&[0, 0, 0, 1, 1, 0, 1, 1]),
            },
        ),
    ];
    let audio_metadata = [(
        InputMetadataKey::AudioMask,
        HostTensorView {
            shape: &[1, 4],
            values: HostTensorValues::Bool(&[true, true, true, true]),
        },
    )];
    run(&[
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::U32(&[1, 2]),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[1, 4, 48],
                values: HostTensorValues::F32(&patches),
            },
            metadata: &image_metadata,
            extents: &[InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        },
        HostInputPart {
            modality: InputModality::Audio,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[1, 4, 128],
                values: HostTensorValues::F32(&audio),
            },
            metadata: &audio_metadata,
            extents: &[InputExtent::AudioValidFrames(4)],
        },
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 1],
                values: HostTensorValues::U32(&[3]),
            },
            metadata: &[],
            extents: &[],
        },
    ])
}
// The same three-token pager used by the public plain and TP cases. Pages
// seal within/across the actual media 2/2/1 prompt chunks; finite Device
// capacity covers this source without claiming forced Host/Disk transfers.
fn paged_state() -> eredu_runtime::CacheResidencyPolicy {
    eredu_runtime::CacheResidencyPolicy::Paged(
        eredu_runtime::PagedCacheOptions::new(3, 8 << 20, 0, 1)
            .unwrap()
            .with_full_attention(true),
    )
}

fn load_with_state(
    root: &Fixture,
    execution: &ExecutionPlan,
    state: eredu_runtime::CacheResidencyPolicy,
) -> LoadedModel<eredu_backend_mlx::backend::MlxBackend<'static>> {
    LoadedModel::load_execution_plan(
        &MlxBackendFactory::default().with_state_residency(state),
        &root.0,
        execution,
    )
    .unwrap_or_else(report_failure)
    .into_parts()
    .0
}

fn run(mode: &str) -> serde_json::Value {
    let root = fixture();
    let execution =
        ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Accelerator(0)).unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let mut model = load_with_state(
        &root,
        &execution,
        eredu_runtime::CacheResidencyPolicy::Device,
    );
    let settings = settings(0.0);
    if mode == "ordinary" {
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"hello"})],
                    tools: vec![],
                    tool_choice: ToolChoice::None,
                    add_generation_prompt: true,
                    ..Default::default()
                },
                8 << 30,
            )
            .unwrap_or_else(|error| panic!("source chat preparation: {error:#}"));
        let mut ordinary = settings;
        ordinary.inference = TextInferencePolicy {
            prefill_chunk_positions: NonZeroU64::new(2),
            ..Default::default()
        };
        let cancellation = GenerationCancellationToken::new();
        let input = with_parts(|parts| model.prepare_chat_input(&chat, parts, &cancellation))
            .unwrap_or_else(report_failure)
            .expect("live media preparation");
        let mut prepared = PreparedChatRequest::new(&chat, chat_settings(&chat, ordinary));
        prepared.input = PreparedChatPrompt::Media(input);
        prepared.output_mode = PreparedChatOutputMode::Text;
        let mut session = model
            .start_controlled_chat(
                prepared,
                TraceLimits {
                    per_record_bytes: 1 << 20,
                    total_bytes: 64 << 20,
                },
                Default::default(),
                |_| ControlFlow::Continue(()),
            )
            .unwrap_or_else(report_failure)
            .expect("live control");
        assert_eq!(session.prompt_attribution().decoder_positions, 5);
        session
            .run(|_| ControlFlow::Continue(()))
            .unwrap_or_else(report_failure);
        let ids = session.token_ids().to_vec();
        drop(session);
        assert_eq!(ids.len(), 4);
        return serde_json::json!({"ids":ids,"text":model.decode(&ids,true).unwrap()});
    }
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let input = with_parts(|parts| model.prepare_managed_model_input(parts, 8 << 30))
        .unwrap_or_else(report_failure);
    let request = ManagedPreparedInputRequest::from_original(input, settings);
    let cancel = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(text) = event {
            visible.push_str(text)
        }
    };
    let output = if mode == "controlled" {
        let mut session = model
            .start_managed_prepared_input(&source, request, &cancel)
            .unwrap_or_else(report_failure)
            .unwrap();
        let report = session.preparation_report().unwrap();
        assert_eq!(report.geometry.input_positions, 5);
        assert_eq!(report.geometry.prefill_chunk_positions, 2);
        while session.finish_reason().is_none() {
            session = session
                .advance(&cancel, &mut emit)
                .unwrap_or_else(report_failure);
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("terminal media request"))
    } else {
        model
            .generate_managed_prepared_input(&source, request, &cancel, &mut emit)
            .unwrap_or_else(report_failure)
            .unwrap()
    };
    assert_eq!(output.token_ids.as_ref().len(), 4);
    assert_eq!(output.text.as_str(), visible);
    let address = output.text.as_str().as_ptr();
    drop(source);
    drop(model);
    assert_eq!(output.text.as_str().as_ptr(), address);
    serde_json::json!({"ids":output.token_ids.as_ref(),"text":output.text.as_str()})
}
#[test]
#[ignore = "requires an accessible Metal device and original media input sources"]
fn native_gemma_raw_original_media_matches_ordinary_managed_and_controlled() {
    verify_mode_results(
        "managed_plain::original_media::native_gemma_raw_original_media_matches_ordinary_managed_and_controlled",
        "EREDU_PUBLIC_GEMMA_ORIGINAL_MEDIA_MODE",
        run,
    );
}

#[path = "original_media/capture.rs"]
mod capture;
