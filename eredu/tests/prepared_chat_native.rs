//! Public ordinary tool generation, native capture and committed semantic copies.
//! A small deterministic neural fixture makes every generated token observable.
use eredu::{
    api::*,
    runtime::chat::{ChatTemplateRequest, ToolChoice},
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    DevicePlan, ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides,
    SemanticEvent, TextInferencePolicy, TextPreparationOptions, capture::*,
    execution_control::SnapshotLimits,
};
use eredu_runtime::{execution_control::SnapshotBudget, working_memory::WorkspaceCopyLimits};
use std::{io::Write, num::NonZeroU64};
use tokenizers::{AddedToken, Tokenizer, decoders::byte_level::ByteLevel, models::bpe::BPE};

const CAPACITY: u64 = 8 << 30;
const PIECES: [&str; 5] = [
    "<tool_call>",
    "\n{\"name\":\"reading\",\"arguments\":{\"value\":",
    "17",
    "}}\n",
    "</tool_call>",
];
#[path = "prepared_chat_native/media.rs"]
mod media;
#[path = "prepared_chat_native/audio.rs"]
mod audio;
#[path = "prepared_chat_native/sampling.rs"]
mod sampling;
#[path = "prepared_chat_native/capture_lifecycle.rs"]
mod capture_lifecycle;

fn fixture(multimodal: bool) -> (tempfile::TempDir, Tokenizer, Vec<u32>) {
    let root = tempfile::tempdir().unwrap();
    let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
    alphabet.sort_unstable();
    let vocab: tokenizers::models::bpe::Vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), id as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens(
            ["<|im_start|>", "<|im_end|>"]
                .map(|value| AddedToken::from(value, true).normalized(false)),
        )
        .unwrap();
    tokenizer
        .add_tokens(PIECES.map(|value| AddedToken::from(value, false).normalized(false)))
        .unwrap();
    if multimodal {
        tokenizer.add_special_tokens(
            ["<|vision_start|>", "<|image_pad|>", "<|video_pad|>", "<|vision_end|>"]
                .map(|value| AddedToken::from(value, true).normalized(false)),
        ).unwrap();
    }
    let mut script = PIECES
        .map(|text| tokenizer.token_to_id(text).unwrap())
        .to_vec();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    script.push(eos);
    let newline = tokenizer.encode("\n", false).unwrap().get_ids()[0];
    let width = tokenizer.get_vocab_size(true);
    let hidden = 16;
    let config = if multimodal {
        serde_json::json!({
            "architectures":["Qwen3VLForConditionalGeneration"], "model_type":"qwen3_vl",
            "image_token_id":tokenizer.token_to_id("<|image_pad|>").unwrap(),
            "video_token_id":tokenizer.token_to_id("<|video_pad|>").unwrap(),
            "vision_start_token_id":tokenizer.token_to_id("<|vision_start|>").unwrap(),
            "vision_end_token_id":tokenizer.token_to_id("<|vision_end|>").unwrap(),
            "tie_word_embeddings":false,"eos_token_id":eos,
            "text_config":{"model_type":"qwen3_vl_text","hidden_size":hidden,"num_hidden_layers":2,
                "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
                "rms_norm_eps":0.000001,"vocab_size":width,"max_position_embeddings":2048,
                "rope_theta":1000000.0,"rope_scaling":{"mrope_section":[1,1,2],"mrope_interleaved":true}},
            "vision_config":{"depth":2,"hidden_size":8,"intermediate_size":16,"num_heads":2,
                "num_position_embeddings":16,"in_channels":3,"patch_size":2,"spatial_merge_size":2,
                "temporal_patch_size":1,"out_hidden_size":hidden,"deepstack_visual_indexes":[0,1]}
        })
    } else { serde_json::json!({"model_type":"qwen2", "hidden_size":hidden,
        "num_hidden_layers":2, "intermediate_size":32, "num_attention_heads":4,
        "num_key_value_heads":2, "rms_norm_eps":0.00001, "vocab_size":width,
        "eos_token_id":eos, "max_position_embeddings":2048, "tie_word_embeddings":false}) };
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let template = include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja");
    std::fs::write(
        root.path().join("tokenizer_config.json"),
        serde_json::to_vec(&serde_json::json!({"chat_template":template})).unwrap(),
    )
    .unwrap();
    tokenizer
        .save(root.path().join("tokenizer.json"), false)
        .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let checkpoint = resolved.architecture.checkpoint();
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    let previous = std::iter::once(newline)
        .chain(script[..script.len() - 1].iter().copied())
        .collect::<Vec<_>>();
    for tensor in checkpoint.common_tensors.iter().chain(
        checkpoint
            .layout_groups
            .iter()
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| &variant.tensors),
    ) {
        let start = data.len();
        for index in 0..tensor.shape.iter().product::<usize>() {
            let value: f32 = if tensor.key.contains("norm") {
                1.0
            } else if tensor.key.ends_with(".embed_tokens.weight") {
                let row = index / hidden;
                let col = index % hidden;
                if previous.get(col).copied() == Some(row as u32) {
                    1.0
                } else {
                    0.0
                }
            } else if tensor.key.ends_with("lm_head.weight") {
                let row = index / hidden;
                let col = index % hidden;
                if script.get(col).copied() == Some(row as u32) {
                    10.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(tensor.key.clone(), serde_json::json!({"dtype":"F32", "shape":tensor.shape, "data_offsets":[start,data.len()]}));
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(root.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    (root, tokenizer, script)
}

fn fail<T>(error: impl std::error::Error + 'static) -> T {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(error) = cause {
        eprintln!("{error:?}");
        cause = error.source();
    }
    panic!("public prepared chat failed; source chain printed above")
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn ordinary_tool_capture_snapshot_restore_and_fork_preserve_committed_semantics() {
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        check(choice, LocalDevice::Accelerator(0), true);
    }
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run explicitly with native device access")]
fn cpu_tools_preserve_ordinary_and_controlled_snapshot_semantics() {
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        check(choice, LocalDevice::Cpu, false);
    }
}

fn check(choice: ToolChoice, device: LocalDevice, capture: bool) {
    let (root, tokenizer, script) = fixture(false);
    let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), root.path(), &execution)
            .unwrap_or_else(fail)
            .into_parts();
    let cancellation = GenerationCancellationToken::new();
    let tokenizer_source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.path().join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(fail);
    let source = model
        .compile_managed_chat_source(
            &tokenizer_source,
            std::fs::File::open(root.path().join("tokenizer_config.json")).unwrap(),
            true,
            &cancellation,
        )
        .unwrap_or_else(fail)
        .unwrap();
    let policy = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"Read value 17."})],
        tools: vec![
            serde_json::json!({"type":"function","function":{"name":"reading",
            "parameters":{"$schema":"http://json-schema.org/draft-07/schema#", "type":"object",
                "properties":{"value":{"enum":[17]}}, "required":["value"], "additionalProperties":false}}}),
        ],
        tool_choice: choice,
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, CAPACITY, &cancellation)
        .unwrap_or_else(fail)
        .unwrap();
    let prompt = tokenizer.encode(chat.rendered_prompt(), false).unwrap();
    let count = prompt.len() as u64;
    let chunk = if count % 17 == 0 { 19 } else { 17 };
    assert!(count > chunk && count % chunk != 0);
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(CAPACITY),
            prefill_chunk_positions: NonZeroU64::new(chunk),
            ..Default::default()
        },
        seed: 37,
        ..Default::default()
    };
    if !capture {
        let mut events = Vec::new();
        let output = model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings.clone()), &cancellation)
            .unwrap_or_else(fail).unwrap()
            .run(&cancellation, &mut |event| events.push(event)).unwrap_or_else(fail);
        assert!(output.token_ids.starts_with(&script[..5]));
        assert_eq!(output.finish_reason, eredu_core::FinishReason::GrammarComplete);
        assert_eq!(events.iter().filter(|event| matches!(event,
            SemanticEvent::ToolCallStart { name, .. } if name == "reading")).count(), 1);
        let arguments: String = events.iter().filter_map(|event| match event {
            SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => Some(json_fragment.as_str()),
            _ => None,
        }).collect();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&arguments).unwrap(), serde_json::json!({"value":17}));
        let ids = output.token_ids.to_vec();
        let finish = output.finish_reason;
        drop(output);
        model.prepare_reset_ordinary().unwrap_or_else(fail)
            .reset_admitted(eredu_core::SessionResetLimits::new(CAPACITY)).unwrap_or_else(fail);
        model.synchronize().unwrap_or_else(fail);
        sampling::check_recorded_child(&mut model, &chat, settings, &script, &ids, finish, &events);
        return;
    }
    let capture_lifecycle::Proof {
        events, expected_ids, expected_finish, budget, spent, terminal_budget, terminal_spent,
    } = capture_lifecycle::check(&mut model, &chat, &settings, &script, count, &cancellation);
    model.prepare_reset_ordinary().unwrap_or_else(fail)
        .reset_admitted(eredu_core::SessionResetLimits::new(CAPACITY)).unwrap_or_else(fail);
    model.synchronize().unwrap_or_else(fail);
    sampling::check_recorded_child(
        &mut model, &chat, settings, &script, &expected_ids, expected_finish, &events,
    );
    drop((
        events,
        source,
        tokenizer_source,
        chat,
        model,
    ));
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while budget.usage().retained_bytes != 0 || terminal_budget.usage().retained_bytes != 0 {
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        assert!(
            std::time::Instant::now() < until,
            "saved copy custody did not retire"
        );
        std::thread::yield_now();
    }
    assert_eq!(budget.usage().cumulative_copy_bytes, spent);
    assert_eq!(terminal_budget.usage().cumulative_copy_bytes, terminal_spent);
}
