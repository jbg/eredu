//! Small native CPU fixture through only the selected public facade.
use eredu::{
    api::*,
    runtime::chat::{ChatTemplateRequest, ToolChoice},
};
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
    let checkpoint = resolved.architecture.checkpoint();
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
            let value = if tensor.key.contains("norm") {
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
    while header.len() % 8 != 0 {
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
    native_facade(LocalDevice::Cpu);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_metal_facade_restores_and_forks_sampled_partial_text() {
    native_facade(LocalDevice::Accelerator(0));
}

fn native_facade(device: LocalDevice) {
    let root = fixture(true);
    let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            tools: vec![
                serde_json::json!({"type":"function", "function":{"name":"lookup",
            "parameters":{"type":"object", "properties":{}, "additionalProperties":false}}}),
            ],
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
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
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
    run.restore(&partial, collect(&mut records)).unwrap();
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
    drop(branch);
    assert_eq!(run.snapshot_usage().unwrap().branches, 0);
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
