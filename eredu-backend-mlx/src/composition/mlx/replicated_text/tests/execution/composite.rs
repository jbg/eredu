#[test]
fn invalid_token_failure_requires_restoration_proof_or_fences_session_mutation() {
    let (stream, weights_stream) = execution_streams();
    let root = tiny_artifact("llama", true);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::default(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(
        plan,
        crate::MlxLoadRequest::default(),
        &stream,
        &weights_stream,
    )
    .unwrap();
    let backend = crate::backend::MlxBackend::new(&stream, &weights_stream);
    let mut session = crate::composition::mlx::MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let first = eredu_core::BackendSession::decode(
        &mut session,
        &backend,
        Array::from_slice(&[1_u32], &[1, 1]),
    )
    .unwrap();
    let overlap = eredu_core::BackendSession::decode(
        &mut session,
        &backend,
        Array::from_slice(&[2_u32], &[1, 1]),
    )
    .err()
    .expect("an unresolved completion must gate the next state mutation");
    assert!(overlap
        .to_string()
        .contains("unresolved submission completion"));
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert!(session.reset().is_err());
    assert!(session.speculative_model_mut().is_err());
    assert!(session.neutral_prediction_target_mut().is_err());
    assert!(session.submit_token_decode(&backend, 3).is_err());
    let empty_prompt = crate::composition::mlx::MlxModelInput::from(
        crate::backend::runtime::media::input::ModelInput::new(&[]),
    );
    assert!(
        eredu_core::BackendSession::prefill(&mut session, &backend, empty_prompt.clone()).is_err()
    );
    assert!(eredu_core::InspectableBackendSession::inspect_decode(
        &mut session,
        &backend,
        Array::from_slice(&[3_u32], &[1, 1]),
        &eredu_core::ObservationRequest::all(),
    )
    .is_err());
    assert!(session
        .submit_prefill_with_observer(
            &backend,
            empty_prompt.clone(),
            &mut eredu_runtime::NoopObserver,
        )
        .is_err());
    assert!(session
        .install_embedded_prediction_observers(
            eredu_runtime::NoopObserver,
            eredu_runtime::NoopObserver
        )
        .is_err());
    let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
        session.prompt_cache_model_identity().unwrap(),
        "fixture-checkpoint",
        "tokens:1",
        1,
    )
    .unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let destination = cache_root.path().join("must-not-be-created");
    assert!(session
        .save_prompt_cache(
            &backend,
            &destination,
            descriptor.clone(),
            &[1],
            &eredu_core::cache::PromptCacheOptions::default(),
        )
        .is_err());
    assert!(session
        .load_prompt_cache(&backend, &destination, &descriptor, &[1])
        .is_err());
    assert!(session
        .load_prompt_cache_for_input(&backend, &destination, &descriptor, &[1], &empty_prompt,)
        .is_err());
    assert!(!destination.exists());
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        counts
    );
    eredu_core::Completion::wait(&first.completion).unwrap();
    let next = eredu_core::BackendSession::decode(
        &mut session,
        &backend,
        Array::from_slice(&[2_u32], &[1, 1]),
    )
    .expect("successful completion must release the submission gate");
    eredu_core::Completion::wait(&next.completion).unwrap();
    let before = session.neutral_prediction_target_mut().unwrap().state_snapshot();
    let before_numeric = session
        .neutral_prediction_target_mut()
        .unwrap()
        .fixed_numeric_state_snapshot()
        .unwrap();

    let error = eredu_core::BackendSession::decode(
        &mut session,
        &backend,
        Array::from_slice(&[-1_i32], &[1, 1]),
    )
    .err()
    .expect("out-of-domain token must fail exact mechanism completion");
    assert!(error.to_string().contains("outside 0..64"));
    if let Ok(target) = session.neutral_prediction_target_mut() {
        assert!(error.model_state_preserved());
        assert_eq!(target.state_snapshot(), before);
        assert_eq!(target.fixed_numeric_state_snapshot().unwrap(), before_numeric);
        let recovered = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[2_u32], &[1, 1]),
        )
        .expect("proven restoration permits retry");
        eredu_core::Completion::wait(&recovered.completion).unwrap();
    } else {
        assert!(session.reset().is_err());
        let retry = eredu_core::BackendSession::decode(
            &mut session,
            &backend,
            Array::from_slice(&[2_u32], &[1, 1]),
        )
        .err()
        .expect("unresolved or failed native validation must fence later mutation");
        assert!(retry.to_string().contains("fenced after prior operation failure"));
    }
}

#[test]
fn public_handoff_executes_ordinary_and_routed_qwen_with_repeated_decode() {
    crate::tests::support::path_instrumentation::reset();
    let (stream, weights_stream) = execution_streams();
    for (model_type, tied) in [
        ("llama", true),
        ("mistral", false),
        ("qwen2", false),
        ("qwen3", true),
        ("qwen3_moe", false),
        ("gpt_oss", false),
    ] {
        let root = tiny_artifact(model_type, tied);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let policy = eredu_core::PreparationPolicy::default();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            policy,
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap_or_else(|error| panic!("{model_type}: {error}"));
        let mut executable = model.into_executable();
        let executable = executable.erased_mut();
        for token in [1_u32, 2] {
            let logits = executable
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap();
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
        assert!(executable.parameter_bank_report().unwrap().is_none());
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts {
            architecture_constructions: 6,
            state_allocations: 6,
            payload_opens: 6,
            constructors: 6,
            unit_constructions: 6,
            materializations: 0,
            local_static_bindings: 0,
            excluded_local_static_parameters: 0,
            forwards: 12,
            state_publications: 12,
            completions: 12,
        }
    );
}

#[test]
fn gguf_requirements_retain_shard_and_multi_output_provenance() {
    let (stream, _) = execution_streams();
    let gguf = tiny_llama_gguf("llama", Some(eredu_gguf::GgmlType::MxFp4), &stream);
    let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
    let shard = inspection.gguf_checkpoint().unwrap().shards()[0]
        .path()
        .to_path_buf();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let output = requirements
        .parameters()
        .iter()
        .find(|parameter| parameter.name() == "lm_head.weight")
        .expect("tied GGUF output remains explicit in the logical topology");
    assert!(matches!(
        output.presence(),
        eredu_runtime::ReplicatedTextParameterPresence::Tied { target }
            if target == "model.embed_tokens.weight"
    ));
    let derived = requirements
        .parameters()
        .iter()
        .find(|parameter| {
            matches!(
                parameter.presence(),
                eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
            ) && parameter
                .physical_sources()
                .iter()
                .any(|source| source.output().ends_with(".scales"))
        })
        .expect("MXFP4 requirements include a derived scales output");
    let source = &derived.physical_sources()[0];
    assert_eq!(source.shard(), shard);
    assert!(source.tensor().ends_with(".weight"));
    assert!(source.output().ends_with(".scales"));
    let direct = requirements
        .parameters()
        .iter()
        .find(|parameter| {
            parameter
                .physical_sources()
                .iter()
                .any(|candidate| candidate.tensor() == source.tensor())
                && parameter.presence().has_physical_source()
        })
        .expect("the same MXFP4 tensor includes its direct weight output");
    assert_eq!(direct.physical_sources()[0].shard(), source.shard());
    assert_ne!(direct.physical_sources()[0].output(), source.output());
}
