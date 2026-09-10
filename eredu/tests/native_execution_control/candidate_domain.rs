use super::*;

/// ByteLevel decodes Ċ to a newline, completing Qwen's forbidden tool trigger.
/// Also leave one unmapped logit ID to distinguish validity from semantics.
pub(super) fn install_vocabulary(root: &Path) {
    let path = root.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocabulary = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    vocabulary.remove("word61");
    vocabulary.remove("word62");
    vocabulary.insert("<tool_call>Ċ".into(), 61.into());
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
}

fn candidates(event: &ObservedGenerationEvent) -> Option<(u32, u64, bool, &CaptureCandidates)> {
    let ObservedGenerationEvent::Token {
        token_id,
        prediction_index,
        forced,
        captures: Some(step),
        ..
    } = event
    else {
        return None;
    };
    assert_eq!(step.records.len(), 1);
    assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
    let Some(CapturePayload::Candidates(candidates)) = &step.records[0].payload else {
        panic!()
    };
    Some((*token_id, *prediction_index, *forced, candidates))
}

pub(super) fn assert_domain(candidates: &CaptureCandidates) {
    assert_eq!(
        candidates.stage,
        CandidateScoreStage::RawLogitsBeforeSampling
    );
    assert_eq!(
        candidates.domain,
        Some(CandidateDomain {
            allowed_tokens: 62,
            vocabulary: 64,
            constrained: true,
        })
    );
    assert_eq!(candidates.candidates.len(), 64);
    for candidate in &candidates.candidates {
        assert!(candidate.score.is_finite());
        assert_eq!(candidate.allowed, ![61, 62].contains(&candidate.token_id));
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn forbidden_tool_candidates_match_ordinary_and_controlled_domains_before_forcing() {
    let root = fixture(false);
    install_vocabulary(&root.0);
    let path = root.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        tools: vec![serde_json::json!({"type":"function", "function": {
            "name":"lookup", "parameters":{"type":"object", "properties":{}, "additionalProperties":false}
        }})],
        tool_choice: ToolChoice::None,
        enable_thinking: Some(false),
        add_generation_prompt: true,
        ..Default::default()
    }).unwrap();
    let usage = CaptureUsage {
        captures: 32,
        retained_bytes: 16 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "candidates".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::TopCandidates { count: 64 },
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let trace = TraceLimits {
        per_record_bytes: 64 << 10,
        total_bytes: 1 << 20,
    };
    let ordinary_prepared = model
        .prepare_observed_chat(&chat, settings, capture.clone(), trace)
        .unwrap();
    let controlled_prepared = model
        .prepare_observed_chat(&chat, settings, capture.clone(), trace)
        .unwrap();
    let prepared = model
        .prepare_observed_chat(&chat, settings, capture, trace)
        .unwrap();
    let uncaptured = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
        .unwrap();
    let expected = model
        .generate_observed_chat(uncaptured, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    model.reset().unwrap();
    let mut ordinary = Vec::new();
    let observed = model
        .generate_observed_chat(ordinary_prepared, &[], Default::default(), |record| {
            ordinary.push(record);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(
        observed.token_ids(),
        expected.token_ids(),
        "capture must not change sampling"
    );
    model.reset().unwrap();
    let mut records = Vec::new();
    {
        let mut run = model
            .start_controlled_chat(
                controlled_prepared,
                &[],
                Default::default(),
                collect(&mut records),
            )
            .unwrap();
        run.run(collect(&mut records)).unwrap();
    }
    let ordinary = ordinary
        .iter()
        .filter_map(|record| candidates(&record.event))
        .collect::<Vec<_>>();
    let controlled = records
        .iter()
        .filter_map(|record| candidates(&record.generation.event))
        .collect::<Vec<_>>();
    assert_eq!(
        ordinary, controlled,
        "capture must preserve sampling and match controlled execution"
    );
    assert!(controlled
        .iter()
        .any(|(_, prediction, _, _)| *prediction > 0));
    for (token, _, forced, capture) in controlled {
        assert!(!forced);
        assert_eq!(capture.source, CandidateLogitsSource::Original);
        assert_domain(capture);
        assert!(
            capture
                .candidates
                .iter()
                .find(|candidate| candidate.token_id == token)
                .unwrap()
                .allowed
        );
    }
    records.clear();
    model.reset().unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    for token in [1, 2] {
        run.force_next_token(token).unwrap();
        run.step(collect(&mut records)).unwrap();
    }
    assert_eq!(
        records
            .iter()
            .filter_map(|record| candidates(&record.generation.event))
            .count(),
        2
    );
    for record in &records {
        if let Some((token, _, forced, capture)) = candidates(&record.generation.event) {
            assert!(forced);
            assert_domain(capture);
            assert!(
                capture
                    .candidates
                    .iter()
                    .find(|candidate| candidate.token_id == token)
                    .unwrap()
                    .allowed
            );
            assert!(
                capture
                    .candidates
                    .iter()
                    .find(|candidate| candidate.token_id == 3)
                    .unwrap()
                    .allowed,
                "a forced choice must not mark other domain members forbidden"
            );
        }
    }
}
