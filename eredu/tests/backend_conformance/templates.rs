use super::*;
use eredu::api::{LoadedTextModelOptions, TextModelError};
use eredu_text::tokenizer::ChatTemplateIdentity;

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
        add_generation_prompt: true,
        extra_template_kwargs: serde_json::Map::from_iter([(
            "application_label".into(),
            serde_json::json!("goose"),
        )]),
        ..Default::default()
    }
}

fn text_options() -> LoadedTextModelOptions {
    LoadedTextModelOptions {
        chat_template: Some(ModelChatTemplate::Named(BTreeMap::from([(
            "default".into(),
            concat!(
                "{{ bos_token }}{{ application_label }}:{{ messages[0].content }}",
                "{% if add_generation_prompt %}|assistant{% endif %}",
            )
            .into(),
        )]))),
    }
}

#[test]
fn standard_loaders_apply_template_override_and_preserve_checkpoint_metadata() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    // A malformed template field must not block an explicit replacement.
    std::fs::write(
        artifact.path().join("tokenizer_config.json"),
        r#"{"chat_template":42,"bos_token":"<bos>"}"#,
    )
    .unwrap();
    std::fs::write(
        artifact.path().join("generation_config.json"),
        r#"{"do_sample":true,"temperature":0.25}"#,
    )
    .unwrap();
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap());
    assert!(LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &plan).is_err());
    let automatic = AutomaticPlanRequest::new(artifact.path(), plan.device().clone());
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let (automatically_loaded, report) = LoadedModel::plan_and_load_with_text_options(
        &MockBackend,
        &AutomaticPlanner::default(),
        &automatic,
        text_options(),
    )
    .unwrap();
    assert_eq!(automatically_loaded.drafting_plan(), report.plan.drafting());
    let models = [
        LoadedModel::load_execution_plan_with_text_options(
            &MockBackend,
            artifact.path(),
            &plan,
            text_options(),
        )
        .unwrap()
        .into_parts()
        .0,
        automatically_loaded.into_parts().0,
        LoadedModel::load_inspected_execution_plan_with_text_options(
            &MockBackend,
            inspection,
            &plan,
            text_options(),
        )
        .unwrap()
        .into_parts()
        .0,
        LoadedModel::load_with_text_options(MockBackend, artifact.path(), (), text_options())
            .unwrap(),
    ];
    for mut model in models {
        assert_eq!(model.model_family(), ModelKind::Llama);
        assert_eq!(model.effective_model_type(), "mistral");
        assert_eq!(model.eos_token_ids(), &[0]);
        assert_eq!(model.encode("hello", false).unwrap(), vec![1]);
        assert_eq!(
            model
                .resolve_generation_config(Default::default())
                .unwrap()
                .temperature,
            0.25,
        );
        assert_eq!(
            model.chat_template_kwargs().unwrap(),
            vec!["application_label"]
        );
        let prepared = model.prepare_chat(request()).unwrap();
        assert_eq!(
            prepared.template_identity(),
            &ChatTemplateIdentity::Named("default".into())
        );
        assert_eq!(prepared.rendered_prompt(), "<bos>goose:hello|assistant");
        assert_eq!(prepared.generation_prompt(), "|assistant");
    }
}

#[test]
fn missing_template_requires_explicit_choice_and_keeps_raw_generation_available() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap());
    let automatic = AutomaticPlanRequest::new(artifact.path(), plan.device().clone());
    let mut models = [
        LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &plan)
            .unwrap()
            .into_parts()
            .0,
        LoadedModel::plan_and_load(&MockBackend, &AutomaticPlanner::default(), &automatic)
            .unwrap()
            .0
            .into_parts()
            .0,
    ];
    for model in &mut models {
        assert!(!model.has_chat_template());
        assert!(matches!(
            model.prepare_chat(request()),
            Err(TextModelError::MissingChatTemplate)
        ));
        assert_eq!(model.selected_chat_template_identity(None).unwrap(), None);
        assert!(model.chat_template_kwargs().unwrap().is_empty());
        assert!(!client_code(model).is_empty());
        model.set_chat_template(Some("{{ messages[0].content }}".into()));
        assert_eq!(
            model.prepare_chat(request()).unwrap().rendered_prompt(),
            "hello"
        );
    }
}

#[test]
fn replacement_updates_named_selection_kwargs_and_protocol_preparation() {
    let mut model = unicode_model(None);
    let original = model.prepare_chat(request()).unwrap();
    assert!(original.format_profile_identity().is_some());
    let fingerprint = *model.tokenizer_fingerprint();
    model.set_chat_template(Some(ModelChatTemplate::Named(BTreeMap::from([
        (
            "default".into(),
            "{{ application_label }}:{{ messages[0].content }}".into(),
        ),
        ("tool_use".into(), QWEN_TEMPLATE.into()),
    ]))));
    assert_eq!(
        model.selected_chat_template_identity(None).unwrap(),
        Some(ChatTemplateIdentity::Named("default".into()))
    );
    assert_eq!(
        model
            .selected_chat_template_identity(Some(&[serde_json::json!({})]))
            .unwrap(),
        Some(ChatTemplateIdentity::Named("tool_use".into())),
    );
    assert_eq!(
        model.chat_template_kwargs().unwrap(),
        vec!["application_label"]
    );
    let replaced = model.prepare_chat(request()).unwrap();
    assert_eq!(replaced.rendered_prompt(), "goose:hello");
    assert_eq!(
        replaced.template_identity(),
        &ChatTemplateIdentity::Named("default".into())
    );
    assert_eq!(replaced.format_profile_identity(), None);

    model.set_chat_template(Some(ModelChatTemplate::Named(BTreeMap::from([(
        "default".into(),
        "second".into(),
    )]))));
    assert_eq!(
        model.prepare_chat(request()).unwrap().rendered_prompt(),
        "second"
    );

    // Application builtin templates use the same recognition and validation.
    model.set_chat_template(Some(QWEN_TEMPLATE.into()));
    assert_eq!(model.prepare_chat(request()).unwrap(), original);
    assert_eq!(model.tokenizer_fingerprint(), &fingerprint);
    model.set_chat_template(Some("{% invalid %}".into()));
    assert!(matches!(
        model.prepare_chat(request()),
        Err(TextModelError::Template(_))
    ));
    model.set_chat_template(None);
    assert!(!model.has_chat_template());
    assert!(matches!(
        model.prepare_chat(request()),
        Err(TextModelError::MissingChatTemplate)
    ));
    assert!(original.format_profile_identity().is_some());
}

fn write_gguf(root: &Path, template: Option<eredu_gguf::MetadataValue>) -> PathBuf {
    use eredu_gguf::{GgmlType, MetadataValue as V, TensorInput, Writer};
    let mut metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("llama".into())),
        ("llama.embedding_length".into(), V::Uint32(2)),
        ("llama.attention.head_count".into(), V::Uint32(1)),
        ("llama.block_count".into(), V::Uint32(1)),
        ("llama.feed_forward_length".into(), V::Uint32(2)),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            V::Float32(1e-5),
        ),
        ("llama.vocab_size".into(), V::Uint32(2)),
        ("llama.context_length".into(), V::Uint32(32)),
    ]);
    if let Some(template) = template {
        metadata.insert("tokenizer.chat_template".into(), template);
    }
    let tensors = [
        ("token_embd.weight", vec![2, 2]),
        ("output_norm.weight", vec![2]),
        ("blk.0.attn_norm.weight", vec![2]),
        ("blk.0.ffn_norm.weight", vec![2]),
        ("blk.0.attn_q.weight", vec![2, 2]),
        ("blk.0.attn_k.weight", vec![2, 2]),
        ("blk.0.attn_v.weight", vec![2, 2]),
        ("blk.0.attn_output.weight", vec![2, 2]),
        ("blk.0.ffn_gate.weight", vec![2, 2]),
        ("blk.0.ffn_up.weight", vec![2, 2]),
        ("blk.0.ffn_down.weight", vec![2, 2]),
    ];
    let inputs = tensors
        .iter()
        .map(|(name, dimensions)| TensorInput {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data: &([0; 16])[..dimensions.iter().product::<u64>() as usize * 4],
        })
        .collect::<Vec<_>>();
    let path = root.join("model.gguf");
    Writer::default()
        .write(std::fs::File::create(&path).unwrap(), &metadata, &inputs)
        .unwrap();
    path
}

#[test]
fn gguf_template_precedence_is_override_then_embedded_then_sidecar() {
    use eredu_gguf::MetadataValue;
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    std::fs::write(artifact.path().join("chat_template.jinja"), "sidecar").unwrap();
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap());
    for (embedded, expected) in [
        (
            Some(MetadataValue::String("embedded".into())),
            Some("embedded"),
        ),
        (None, Some("sidecar")),
        (Some(MetadataValue::Uint32(42)), None),
    ] {
        let path = write_gguf(artifact.path(), embedded);
        let loaded = LoadedModel::load_execution_plan(&MockBackend, &path, &plan);
        if let Some(expected) = expected {
            assert_eq!(
                loaded
                    .unwrap()
                    .model_mut()
                    .prepare_chat(request())
                    .unwrap()
                    .rendered_prompt(),
                expected
            );
        } else {
            assert!(matches!(
                loaded,
                Err(eredu::api::PlannedModelLoadError::Loading(
                    eredu::api::LoadedModelLoadError::Metadata(
                        eredu::api::TextMetadataError::GgufTokenizer(_)
                    )
                ))
            ));
        }
        let mut overridden = LoadedModel::load_execution_plan_with_text_options(
            &MockBackend,
            &path,
            &plan,
            LoadedTextModelOptions {
                chat_template: Some("override".into()),
            },
        )
        .unwrap();
        assert_eq!(
            overridden
                .model_mut()
                .prepare_chat(request())
                .unwrap()
                .rendered_prompt(),
            "override"
        );
    }
    let path = write_gguf(
        artifact.path(),
        Some(MetadataValue::String("embedded".into())),
    );
    std::fs::write(
        artifact.path().join("tokenizer_config.json"),
        r#"{"chat_template":42}"#,
    )
    .unwrap();
    let mut loaded = LoadedModel::load_execution_plan(&MockBackend, &path, &plan).unwrap();
    assert_eq!(
        loaded
            .model_mut()
            .prepare_chat(request())
            .unwrap()
            .rendered_prompt(),
        "embedded"
    );
}
