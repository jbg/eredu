//! Native CLI reads and facade pinning share the ordinary operation driver.
#![cfg(feature = "mlx")]
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{capture::CapturePlan, refinement::*, ExecutionPlan, GenerationConfigOverrides};
use serde_json::{json, Value};
use std::{
    io::Write,
    ops::ControlFlow,
    path::Path,
    process::{Command, Stdio},
};

fn fixture(root: &Path) {
    let source = json!({"model_type":"diffusion_gemma","canvas_length":32,"text_config":{"model_type":"diffusion_gemma_text","hidden_size":4,"num_hidden_layers":1,"intermediate_size":6,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":0.00001,"vocab_size":64,"max_position_embeddings":512,"enable_moe_block":false,"layer_types":["full_attention"]}});
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&source).unwrap(),
    )
    .unwrap();
    let config = eredu_architectures::diffusion_gemma::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&source).unwrap(),
    )
    .unwrap();
    let plan = eredu_architectures::diffusion_gemma::safetensors_plan(&config).unwrap();
    let mut data = vec![];
    let mut header = serde_json::Map::new();
    for tensor in plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter(|g| g.required)
                .flat_map(|g| &g.variants[0].tensors),
        )
        .filter(|t| t.requirement == eredu_checkpoint::schema::TensorRequirement::Required)
    {
        let start = data.len();
        for i in 0..tensor.shape.iter().product::<usize>() {
            let value = if tensor.key.contains("norm") || tensor.key.ends_with("layer_scalar") {
                1.0f32
            } else {
                (((i * 7 + tensor.key.len() * 13) % 31) as f32 - 15.0) * 0.01
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            tensor.key.clone(),
            json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    let mut vocab =
        serde_json::Map::from_iter([("[UNK]".into(), json!(0)), (":".into(), json!(1))]);
    for (i, letter) in ('A'..='Z').enumerate() {
        vocab.insert(letter.to_string(), json!(i + 2));
    }
    let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
    std::fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    std::fs::write(root.join("chat_template.jinja"), "{% if enable_thinking is defined and enable_thinking %}thought{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}").unwrap();
}

#[test]
fn cli_reads_choice_noul_and_score_with_resident_and_bounded_weights() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let request = json!({"state":{"ticket":"duplicate charge"},"steps":2,"questions":[{"id":"route","type":"choice","question":"Choose the department","options":["billing","technical"]},{"id":"urgent","type":"noul","question":"Is it urgent?"},{"id":"priority","type":"score","question":"Rate the priority","levels":[{"label":"low","value":10,"description":"routine"},{"label":"high","value":30,"description":"urgent"}]}]});
    let path = root.path().join("request.json");
    std::fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    for residency in [None, Some("--layerwise-host"), Some("--dense-disk-stream")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_eredu"));
        command
            .args(["--no-auto", "--timing", "--device", "cpu", "--model"])
            .arg(root.path())
            .args(["--seed", "42", "--decisions"]);
        if residency == Some("--dense-disk-stream") {
            command.arg("-").stdin(Stdio::piped());
        } else {
            command.arg(&path);
        }
        if let Some(flag) = residency {
            command.arg(flag);
        }
        let output = if residency == Some("--dense-disk-stream") {
            let mut child = command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(&serde_json::to_vec(&request).unwrap())
                .unwrap();
            child.wait_with_output().unwrap()
        } else {
            command.output().unwrap()
        };
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        for field in [
            "canvas_prepare_seconds=",
            "chat_prepare_seconds=",
            "discovery_seconds=",
            "admission_seconds=",
            "prepare_seconds=",
            "context_seconds=",
            "refinement_seconds=",
            "finish_seconds=",
            "prompt_tokens=",
            "canvas_tokens=32",
        ] {
            assert!(
                stderr.contains(field),
                "missing timing field {field}: {stderr}"
            );
        }
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["steps"], 2);
        assert_eq!(result["seed"], 42);
        assert_eq!(result["normalization"], "allowed_labels");
        let answers = result["answers"].as_object().unwrap();
        assert_eq!(answers.len(), 3);
        for row in answers.values() {
            let total: f64 = row["probabilities"]
                .as_object()
                .unwrap()
                .values()
                .map(|p| p.as_f64().unwrap())
                .sum();
            assert!((total - 1.0).abs() < 1e-12);
            assert!(row["confidence"].as_f64().unwrap() >= 0.5);
            assert!((0.0..=1.0).contains(&row["label_mass"].as_f64().unwrap()));
        }
        assert!(matches!(
            answers["route"]["choice"].as_str(),
            Some("billing" | "technical")
        ));
        assert!(answers["urgent"]["value"].is_boolean());
        let score = answers["priority"]["score"].as_f64().unwrap();
        assert!((10.0..=30.0).contains(&score));
    }
}

#[test]
fn ordinary_and_controlled_facade_preserve_pins_and_canonical_commitment() {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path());
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), root.path(), &plan)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role":"user","content":"evidence"})],
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        seed: 7,
        draft_events: true,
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        refinement: Some(
            RefinementOptions {
                max_denoising_steps: 2,
                stability_threshold: 2,
                ..Default::default()
            }
            .try_into()
            .unwrap(),
        ),
        ..Default::default()
    };
    let mut seed = vec![None; 32];
    seed[0] = Some(2);
    seed[2] = Some(3);
    let layout = RefinementCanvas {
        seed_tokens: seed,
        pinned_positions: vec![0, 2],
    };
    let trace = TraceLimits {
        per_record_bytes: 1 << 20,
        total_bytes: 16 << 20,
    };
    // A model-domain token absent from the tokenizer is rejected before execution.
    let mut invalid = layout.clone();
    invalid.seed_tokens[0] = Some(63);
    let rejected = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
        .unwrap()
        .with_refinement_canvas(invalid);
    assert!(matches!(
        model.generate_observed_text(
            rejected,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(())
        ),
        Err(PreparedChatError::Generation(
            eredu_core::generation::GenerationError::RefinementCanvas(_)
        ))
    ));
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
        .unwrap()
        .with_refinement_canvas(layout.clone());
    let ordinary = model
        .generate_observed_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!((ordinary.token_ids[0], ordinary.token_ids[2]), (2, 3));
    model.reset().unwrap();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
        .unwrap()
        .with_refinement_canvas(layout);
    let mut controlled = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    controlled.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(controlled.token_ids(), ordinary.token_ids);
}
