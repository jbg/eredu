//! Small native CPU fixture through only the selected public facade.
use eredu::{
    api::*,
    runtime::chat::{ChatTemplateRequest, ToolChoice},
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    capture::*, execution_control::*, ExecutionPlan, GenerationConfigOverrides, SemanticEvent,
    SessionCapabilities,
};
use std::{
    io::Write,
    ops::ControlFlow,
    path::{Path, PathBuf},
};
use tokenizers::{
    decoders::byte_level::ByteLevel, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture(fragments: bool) -> Fixture {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "eredu-control-native-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let fixture = Fixture(root);
    write_weights(&fixture.0);
    let mut vocabulary = (0..64)
        .map(|id| (format!("word{id}"), id))
        .collect::<std::collections::HashMap<_, _>>();
    for (id, token) in [
        (0, "[UNK]"),
        (1, if fragments { "Ã" } else { "left" }),
        (2, if fragments { "©" } else { "right" }),
        (63, "<|im_end|>"),
    ] {
        vocabulary.remove(&format!("word{id}"));
        vocabulary.insert(token.into(), id);
    }
    let words = WordLevel::builder()
        .vocab(vocabulary.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    tokenizer
        .save(fixture.0.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        fixture.0.join("chat_template.jinja"),
        include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja"),
    )
    .unwrap();
    fixture
}
fn write_weights(root: &Path) {
    let config = serde_json::json!({"model_type":"qwen2", "hidden_size":16,
        "num_hidden_layers":2, "intermediate_size":32, "num_attention_heads":4,
        "num_key_value_heads":2, "rms_norm_eps":0.00001, "vocab_size":64,
        "eos_token_id":63, "max_position_embeddings":1024, "tie_word_embeddings":false});
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(root, resolved.architecture.checkpoint());
}

fn write_tensor_plan(
    root: &Path,
    checkpoint: &eredu_checkpoint::schema::SafetensorsCheckpointPlan,
) {
    let mut data = vec![];
    let mut header = serde_json::Map::new();
    for tensor in checkpoint.common_tensors.iter().chain(
        checkpoint
            .layout_groups
            .iter()
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| &variant.tensors),
    ) {
        let start = data.len();
        for index in 0..tensor.shape.iter().product::<usize>() {
            let value = if tensor.key.ends_with(".layer_scalar") {
                1.15f32
            } else if tensor.key.contains("norm") {
                1.0f32
            } else {
                (((index * 17 + tensor.key.len() * 7) % 101) as f32 - 50.0) * 0.003
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            tensor.key.clone(),
            serde_json::json!({"dtype":"F32", "shape":tensor.shape,
            "data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut weights = std::fs::File::create(root.join("model.safetensors")).unwrap();
    weights
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    weights.write_all(&header).unwrap();
    weights.write_all(&data).unwrap();
}
fn collect(
    records: &mut Vec<ControlledGenerationRecord>,
) -> impl FnMut(ControlledGenerationRecord) -> ControlFlow<()> + '_ {
    |record| {
        records.push(record);
        ControlFlow::Continue(())
    }
}
fn semantics(records: &[ControlledGenerationRecord]) -> Vec<SemanticEvent> {
    records
        .iter()
        .filter_map(|record| match &record.generation.event {
            ObservedGenerationEvent::Semantic { event, .. } => Some(event.clone()),
            _ => None,
        })
        .collect()
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run the CPU fixture with --no-default-features --features mlx; Metal-enabled MLX initialization needs an accessible GPU"
)]
fn native_cpu_facade_restores_and_forks_sampled_partial_text_without_reopening_artifacts() {
    native_facade(LocalDevice::Cpu, false);
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn native_cpu_controlled_text_restores_and_forks_without_semantic_support() {
    native_facade(LocalDevice::Cpu, true);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_metal_facade_restores_and_forks_sampled_partial_text() {
    native_facade(LocalDevice::Accelerator(0), false);
}

fn native_facade(device: LocalDevice, text: bool) {
    let root = fixture(true);
    if text {
        std::fs::write(
            root.0.join("chat_template.jinja"),
            "{% for m in messages %}{{ m.content }}{% endfor %} reply: ",
        )
        .unwrap();
    }
    let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            tools: if text {
                vec![]
            } else {
                vec![
                    serde_json::json!({"type":"function", "function":{"name":"lookup",
            "parameters":{"type":"object", "properties":{}, "additionalProperties":false}}}),
                ]
            },
            tool_choice: ToolChoice::None,
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        // Fixed fixture seed keeps later unconstrained draws in printable tokens;
        // byte-fragment choices above are intentional and checked separately.
        seed: 17,
        ..Default::default()
    };
    let trace = TraceLimits {
        per_record_bytes: 16384,
        total_bytes: 65536,
    };
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
        .unwrap();
    // Every subsequent action must use the resident prepared model and tokenizer.
    std::fs::remove_file(root.0.join("model.safetensors")).unwrap();
    std::fs::remove_file(root.0.join("tokenizer.json")).unwrap();
    let mut records = vec![];
    let mut run = if text {
        assert!(matches!(
            chat.semantic_support(),
            eredu::runtime::chat::SemanticSupport::Unsupported { .. }
        ));
        model.start_controlled(
            prepared,
            &[],
            Default::default(),
            collect(&mut records),
            eredu::api::OutputMode::Text,
        )
    } else {
        model.start_controlled(
            prepared,
            &[],
            Default::default(),
            collect(&mut records),
            eredu::api::OutputMode::Semantic,
        )
    }
    .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 64 << 20,
        cumulative_copy_bytes: 256 << 20,
    })
    .unwrap();
    run.force_next_token(1).unwrap();
    run.step(collect(&mut records)).unwrap();
    run.force_next_token(2).unwrap();
    let partial = run.snapshot(collect(&mut records)).unwrap();
    assert_eq!(partial.metadata().pending_forced_token, Some(2));
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    let baseline = run.token_ids().to_vec();
    let text = semantics(&records[before..]);
    assert!(text.contains(&SemanticEvent::TextDelta("é".into())));
    let copied_before_restore = run.snapshot_usage().unwrap().cumulative_copy_bytes;
    run.restore(&partial, collect(&mut records)).unwrap();
    assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > copied_before_restore);
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    assert_eq!(run.token_ids(), baseline);
    assert_eq!(semantics(&records[before..]), text);
    let mut branch = run
        .fork(
            &partial,
            GenerationBranchOptions {
                trace_limits: trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            collect(&mut records),
        )
        .unwrap();
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    assert_eq!(run.token_ids(), baseline);
    assert_eq!(semantics(&records[before..]), text);
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    assert_eq!(run.token_ids(), baseline);
    let copied_before_drop = run.snapshot_usage().unwrap().cumulative_copy_bytes;
    drop(branch);
    assert_eq!(run.snapshot_usage().unwrap().branches, 0);
    assert_eq!(
        run.snapshot_usage().unwrap().cumulative_copy_bytes,
        copied_before_drop
    );
}

#[allow(dead_code)]
#[path = "../examples/controlled_generate.rs"]
mod controlled_example;

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn complete_controlled_example_verifies_native_capture_restore_and_modified_branch() {
    let root = fixture(false);
    controlled_example::run_example(
        &root.0,
        "Explain gravity briefly.",
        eredu_core::intervention::InterventionDtype::Float32,
    )
    .unwrap();
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn native_text_matches_ordinary_sampling_with_checkpoint_defaults_and_padded_logits() {
    text_matches_ordinary_sampling();
}

fn text_matches_ordinary_sampling() {
    text_matches_ordinary_sampling_with_policy(
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::WeightTransformationPlan::PreserveCheckpoint,
    );
}

fn text_matches_ordinary_sampling_with_policy(
    residency: eredu_core::ResidencyPlan,
    transformation: eredu_core::WeightTransformationPlan,
) {
    let root = fixture(false);
    // Keep the checkpoint's 64 logits positions but only 32 mapped tokenizer IDs.
    let path = root.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    tokenizer["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .retain(|_, id| id.as_u64().unwrap() < 32);
    tokenizer["added_tokens"] = serde_json::json!([]);
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    // No EOS: compare the complete bounded continuation, including repeated draws.
    let path = root.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    std::fs::write(root.0.join("generation_config.json"), r#"{"do_sample":true,"temperature":1.7,"top_k":12,"top_p":0.9,"repetition_penalty":1.1,"max_new_tokens":12}"#).unwrap();
    std::fs::write(root.0.join("chat_template.jinja"), "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %} word3 word7{% endif %}").unwrap();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_residency(residency)
        .with_weight_transformation(transformation)
        .with_prepared_checkpoint_cache(
            eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy {
                directory: root.0.join("prepared"),
                max_bytes: 64 * 1024 * 1024,
            },
        );
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    for temperature in [0.0, 1.7] {
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":"word4 right"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(temperature),
                ..Default::default()
            },
            seed: 827,
            ..Default::default()
        };
        let trace = TraceLimits {
            per_record_bytes: 16384,
            total_bytes: 65536,
        };
        let prepared = model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
            .unwrap();
        let prompt = prepared.prompt_token_ids().to_vec();
        assert_eq!(prompt, [4, 2, 3, 7]);
        let resolved = prepared.generation_config();
        assert_eq!(resolved.temperature, temperature);
        assert_eq!(resolved.top_k, 12);
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.repetition_penalty, 1.1);
        assert_eq!(resolved.max_new_tokens, Some(12));
        let baseline: Vec<_> = model
            .generate_tokens(
                prompt.clone(),
                eredu_core::TextGenerationConfig::new(resolved).with_seed(settings.seed),
            )
            .unwrap()
            .take(12)
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        assert!(baseline.iter().all(|id| *id < 32));
        model.reset().unwrap();
        let mut records = vec![];
        let mut run = model
            .start_controlled(
                prepared,
                &[],
                Default::default(),
                collect(&mut records),
                eredu::api::OutputMode::Text,
            )
            .unwrap();
        assert!(run.force_next_token(32).is_err());
        assert!(run.force_next_token(63).is_err());
        run.run(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), baseline);
        assert_eq!(
            run.finish_reason(),
            Some(eredu_core::FinishReason::MaxTokens)
        );
        drop(run);
        let expected = model.decode(&baseline, true).unwrap();
        let actual: String = semantics(&records)
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(actual, expected);
        model.reset().unwrap();
    }
}

#[path = "native_execution_control/speculative.rs"]
mod speculative;

#[path = "native_execution_control/candidate_domain.rs"]
mod candidate_domain;

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with accessible Metal initialization"
)]
fn prompt_cache_stores_share_generation_identity_cancellation_and_budget_contracts() {
    let root = fixture(false);
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_paged_state(eredu_core::PagedStatePlan::new(3, 4 << 20, 4 << 20).unwrap())
        .with_prompt_cache_persistence(true);
    let load = || {
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts()
            .0
    };
    let mut cold = load();
    let mut model = load();
    let chat=model.prepare_chat(ChatTemplateRequest {
        messages:vec![serde_json::json!({"role":"user","content":"left right left right left right left right"})],
        add_generation_prompt:true,..Default::default()
    }).unwrap();
    let tokens = model.encode(chat.rendered_prompt(), false).unwrap();
    let settings = PreparedChatGenerationSettings {
        prefill: PrefillChunkPolicy::Bounded(std::num::NonZeroUsize::new(3).unwrap()),
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        ..Default::default()
    };
    let generate = |model: &mut LoadedModel<_>, ids: Vec<u32>, cancellation| {
        model.reset().unwrap();
        model
            .session()
            .generate(GenerationRequest {
                input: PreparedChatInput::token_ids(&chat, ids),
                output_mode: OutputMode::Text,
                settings,
                caller_stop_sequences: &[],
                cancellation,
                control_handle: None,
                on_event: |_| {},
            })
            .unwrap()
    };
    let stores: Vec<std::sync::Arc<dyn PromptCacheStore>> = vec![
        std::sync::Arc::new(MemoryPromptCacheStore::new(4 << 20)),
        std::sync::Arc::new(DirectoryPromptCacheStore::new(
            root.0.join("cache"),
            4 << 20,
        )),
    ];
    for store in stores {
        model
            .set_prompt_cache_store("fixture-checkpoint", Box::new(store.clone()))
            .unwrap();
        let first = generate(&mut model, tokens.clone(), Default::default());
        assert_eq!(
            first.stats().prompt_cache.as_ref().unwrap().reused_tokens,
            0
        );
        assert!(
            model.prompt_cache_store().unwrap().retained_bytes() > 0,
            "stats={:?}, prefill={:?}",
            first.stats(),
            model.prefill_chunking_support()
        );
        let pinned = model.prompt_cache_store().unwrap().open().unwrap().unwrap();
        let saved_manifest = pinned.manifest().clone();
        let warm = generate(&mut model, tokens.clone(), Default::default());
        assert!(warm.stats().prompt_cache.as_ref().unwrap().reused_tokens > 0);
        assert_eq!(warm.token_ids(), first.token_ids());
        assert_eq!(
            warm.token_ids(),
            generate(&mut cold, tokens.clone(), Default::default()).token_ids()
        );
        // The existing checkpoint identity permits reuse across model loads.
        cold.set_prompt_cache_store("fixture-checkpoint", Box::new(store.clone()))
            .unwrap();
        let reloaded = generate(&mut cold, tokens.clone(), Default::default());
        assert!(
            reloaded
                .stats()
                .prompt_cache
                .as_ref()
                .unwrap()
                .reused_tokens
                > 0
        );
        assert_eq!(reloaded.token_ids(), warm.token_ids());
        cold.remove_prompt_cache_store();
        // Reject a mismatched digest atomically through both store implementations.
        let validation: Box<dyn PromptCacheStore> = if store.directory().is_some() {
            Box::new(DirectoryPromptCacheStore::new(
                root.0.join("invalid-cache"),
                4 << 20,
            ))
        } else {
            Box::new(MemoryPromptCacheStore::new(4 << 20))
        };
        let mut writer = validation
            .begin(
                1 << 20,
                &eredu_core::cache::PromptCacheOptions::new(None, true).unwrap(),
            )
            .unwrap();
        for name in saved_manifest
            .blocks
            .iter()
            .map(|b| &b.shard)
            .chain(saved_manifest.state_tensors.iter().map(|s| &s.shard))
        {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut pinned.open_shard(name).unwrap(), &mut bytes).unwrap();
            writer.put_shard(name, bytes).unwrap();
        }
        let mut invalid = saved_manifest.clone();
        invalid.blocks[0].payload_sha256 = "0".repeat(64);
        assert!(writer.commit(&invalid).is_err());
        assert!(validation.open().unwrap().is_none());
        assert_eq!(validation.retained_bytes(), 0);
        // A reader admitted before replacement remains usable, even after clear.
        model.prompt_cache_store().unwrap().clear().unwrap();
        if model.prompt_cache_store().unwrap().directory().is_none() {
            assert!(model.prompt_cache_store().unwrap().retained_bytes() > 0);
        }
        for block in &saved_manifest.blocks {
            let source = eredu_runtime::RetainedCacheShard::from_reader(
                std::path::Path::new(&block.shard),
                pinned.open_shard(&block.shard).unwrap(),
                block.logical_bytes,
            )
            .unwrap();
            let mut first = vec![
                0;
                source
                    .metadata()
                    .info(&block.first_array)
                    .unwrap()
                    .data_offsets
                    .1
                    - source
                        .metadata()
                        .info(&block.first_array)
                        .unwrap()
                        .data_offsets
                        .0
            ];
            let mut second = vec![
                0;
                source
                    .metadata()
                    .info(&block.second_array)
                    .unwrap()
                    .data_offsets
                    .1
                    - source
                        .metadata()
                        .info(&block.second_array)
                        .unwrap()
                        .data_offsets
                        .0
            ];
            source
                .read_into(
                    &mut [
                        (&block.first_array, &mut first),
                        (&block.second_array, &mut second),
                    ],
                    Some(&block.payload_sha256),
                )
                .unwrap();
        }
        drop(pinned);
        assert_eq!(model.prompt_cache_store().unwrap().retained_bytes(), 0);
        let _ = generate(&mut model, tokens.clone(), Default::default());
        let mut extended = tokens.clone();
        extended.extend([1, 2, 1, 2]);
        let warm = generate(&mut model, extended.clone(), Default::default());
        assert!(warm.stats().prompt_cache.as_ref().unwrap().reused_tokens > 0);
        assert_eq!(
            warm.token_ids(),
            generate(&mut cold, extended.clone(), Default::default()).token_ids()
        );
        extended[0] = if extended[0] == 1 { 2 } else { 1 };
        let mismatch = generate(&mut model, extended.clone(), Default::default());
        assert_eq!(
            mismatch
                .stats()
                .prompt_cache
                .as_ref()
                .unwrap()
                .reused_tokens,
            0
        );
        assert_eq!(
            mismatch.token_ids(),
            generate(&mut cold, extended.clone(), Default::default()).token_ids()
        );
        model
            .set_prompt_cache_store("different-checkpoint", Box::new(store.clone()))
            .unwrap();
        let changed_identity = generate(&mut model, extended.clone(), Default::default());
        assert_eq!(
            changed_identity
                .stats()
                .prompt_cache
                .as_ref()
                .unwrap()
                .reused_tokens,
            0
        );
        assert_eq!(changed_identity.token_ids(), mismatch.token_ids());
        let cancellation = eredu_core::GenerationCancellationToken::default();
        let signal = cancellation.clone();
        let cancelled = model
            .session()
            .generate(GenerationRequest {
                input: PreparedChatInput::token_ids(&chat, extended),
                output_mode: OutputMode::Text,
                settings,
                caller_stop_sequences: &[],
                cancellation,
                control_handle: None,
                on_event: move |event| {
                    if matches!(event, SemanticEvent::TextDelta(_)) {
                        signal.cancel();
                    }
                },
            })
            .unwrap();
        assert_eq!(
            cancelled.finish_reason(),
            eredu_core::FinishReason::Cancelled
        );
        assert_eq!(model.prompt_cache_store().unwrap().retained_bytes(), 0);
    }
    model
        .set_prompt_cache_store(
            "fixture-checkpoint",
            Box::new(MemoryPromptCacheStore::new(1)),
        )
        .unwrap();
    let denied = generate(&mut model, tokens.clone(), Default::default());
    assert_eq!(model.prompt_cache_store().unwrap().retained_bytes(), 0);
    assert_eq!(
        denied.token_ids(),
        generate(&mut cold, tokens, Default::default()).token_ids()
    );
}
