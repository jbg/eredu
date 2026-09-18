use super::*;
use eredu::api::{TextInspectionOptions, TextModelError, TextModelOptions, inspect_text_model};
use eredu_core::{
    InspectionIssueCode, InspectionReadiness, InspectionSeverity, ModelInspectionReport,
};
use eredu_text::tokenizer::ChatTemplateIdentity;

fn retained_artifacts(
    model: LoadedModel<MockBackend>,
    root: &Path,
    template: Option<ModelChatTemplate>,
) -> original_sources::Fixture<MockBackend> {
    // All GGUF fixtures in this module deliberately use this same tokenizer sidecar.
    let bytes = std::fs::read(root.join("tokenizer.json")).unwrap();
    let mut kwargs = std::fs::read(root.join("tokenizer_config.json"))
        .ok()
        .map(|bytes| {
            serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&bytes).unwrap()
        })
        .unwrap_or_default();
    kwargs.remove("chat_template");
    original_sources::Fixture::from_loaded(model, bytes, template, kwargs)
}

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

fn text_options() -> TextModelOptions {
    TextModelOptions {
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
    for model in models {
        let mut model = retained_artifacts(model, artifact.path(), text_options().chat_template);
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
        let prepared = {
            let request = request();
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model
                .chat_source(
                    !request.tools.is_empty(),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            model
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        };
        assert_eq!(
            prepared.template_identity(),
            &ChatTemplateIdentity::Named("default".into())
        );
        assert_eq!(prepared.rendered_prompt(), "<bos>goose:hello|assistant");
        assert_eq!(prepared.generation_prompt(), "|assistant");
    }
    let options = TextModelOptions {
        chat_template: Some(
            concat!(
                "{% if bos_token != '<bos>' or application_label != 'goose' %}",
                "{{ raise_exception('template variables were not preserved') }}{% endif %}",
                "{{ messages[0].content }}",
            )
            .into(),
        ),
    };
    let report = inspect_text_model(
        ModelInspectionReport::unverified(artifact.path(), ArtifactFormat::SafeTensors),
        &options,
        TextInspectionOptions {
            chat_request: Some(request()),
        },
    );
    assert_eq!(report.chat_template, InspectionReadiness::Ready);
    assert!(
        report.issues.iter().all(|issue| {
            issue.code != InspectionIssueCode::MissingChatTemplate
                && issue.severity != InspectionSeverity::Error
        }),
        "{:?}",
        report.issues
    );
    let template = options.chat_template.clone();
    let mut loaded = retained_artifacts(
        LoadedModel::load_with_text_options(MockBackend, artifact.path(), (), options).unwrap(),
        artifact.path(),
        template,
    );
    assert_eq!(
        {
            let request = request();
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = loaded
                .chat_source(
                    !request.tools.is_empty(),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            loaded
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        }
        .rendered_prompt(),
        "hello"
    );
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
    ]
    .map(|model| retained_artifacts(model, artifact.path(), None));
    for model in &mut models {
        assert!(!model.has_chat_template());
        assert!(matches!(
            model.chat_source(false, &Default::default()),
            Err(original_sources::FixtureSourceError::Chat(
                eredu::api::ManagedChatSourceError::Validation(_)
            ))
        ));
        assert_eq!(model.selected_chat_template_identity(None).unwrap(), None);
        assert!(model.chat_template_kwargs().unwrap().is_empty());
        assert!(!client_code(model).is_empty());
        model.replace_template(Some("{{ messages[0].content }}".into()));
        assert_eq!(
            {
                let request = request();
                let cancellation = eredu_core::GenerationCancellationToken::new();
                let source = model
                    .chat_source(
                        !request.tools.is_empty(),
                        &cancellation,
                    )
                    .unwrap()
                    .unwrap();
                model
                    .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                    .unwrap()
                    .unwrap()
            }
            .rendered_prompt(),
            "hello"
        );
    }
}

#[test]
fn replacement_updates_named_selection_kwargs_and_protocol_preparation() {
    let mut model = unicode_model(None);
    let original = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(
                !request.tools.is_empty(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    assert!(original.format_profile_identity().is_some());
    let fingerprint = *model.tokenizer_fingerprint();
    model.replace_template(Some(ModelChatTemplate::Named(BTreeMap::from([
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
    let replaced = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(
                !request.tools.is_empty(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    assert_eq!(replaced.rendered_prompt(), "goose:hello");
    assert_eq!(
        replaced.template_identity(),
        &ChatTemplateIdentity::Named("default".into())
    );
    assert_eq!(replaced.format_profile_identity(), None);

    model.replace_template(Some(ModelChatTemplate::Named(BTreeMap::from([(
        "default".into(),
        "second".into(),
    )]))));
    assert_eq!(
        {
            let request = request();
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model
                .chat_source(
                    !request.tools.is_empty(),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            model
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        }
        .rendered_prompt(),
        "second"
    );

    // Application builtin templates use the same recognition and validation.
    model.replace_template(Some(QWEN_TEMPLATE.into()));
    let replay = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(
                !request.tools.is_empty(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    assert_eq!(replay.rendered_prompt(), original.rendered_prompt());
    assert_eq!(replay.generation_prompt(), original.generation_prompt());
    assert_eq!(replay.template_identity(), original.template_identity());
    assert_eq!(
        replay.format_profile_identity(),
        original.format_profile_identity()
    );
    assert_eq!(model.tokenizer_fingerprint(), &fingerprint);
    model.replace_template(Some("{% invalid %}".into()));
    assert!(matches!(
        model.chat_source(false, &Default::default()),
        Err(original_sources::FixtureSourceError::Chat(
            eredu::api::ManagedChatSourceError::Source(_)
        ))
    ));
    model.replace_template(None);
    assert!(!model.has_chat_template());
    assert!(matches!(
        model.chat_source(false, &Default::default()),
        Err(original_sources::FixtureSourceError::Chat(
            eredu::api::ManagedChatSourceError::Validation(_)
        ))
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
            let mut model = retained_artifacts(
                loaded.unwrap().into_parts().0,
                artifact.path(),
                Some(expected.into()),
            );
            assert_eq!(
                {
                    let request = request();
                    let cancellation = eredu_core::GenerationCancellationToken::new();
                    let source = model
                        .chat_source(
                            !request.tools.is_empty(),
                            &cancellation,
                        )
                        .unwrap()
                        .unwrap();
                    model
                        .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                        .unwrap()
                        .unwrap()
                }
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
            TextModelOptions {
                chat_template: Some("override".into()),
            },
        )
        .unwrap();
        let mut overridden = retained_artifacts(
            overridden.into_parts().0,
            artifact.path(),
            Some("override".into()),
        );
        assert_eq!(
            {
                let request = request();
                let cancellation = eredu_core::GenerationCancellationToken::new();
                let source = overridden
                    .chat_source(
                        !request.tools.is_empty(),
                        &cancellation,
                    )
                    .unwrap()
                    .unwrap();
                overridden
                    .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                    .unwrap()
                    .unwrap()
            }
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
    let loaded = LoadedModel::load_execution_plan(&MockBackend, &path, &plan).unwrap();
    let mut loaded = retained_artifacts(
        loaded.into_parts().0,
        artifact.path(),
        Some("embedded".into()),
    );
    assert_eq!(
        {
            let request = request();
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = loaded
                .chat_source(
                    !request.tools.is_empty(),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            loaded
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        }
        .rendered_prompt(),
        "embedded"
    );
}

#[test]
fn text_inspection_uses_template_overrides_for_both_artifact_formats() {
    use eredu_gguf::MetadataValue;

    for format in [ArtifactFormat::SafeTensors, ArtifactFormat::Gguf] {
        for checkpoint_template in [
            None,
            Some(serde_json::json!("ordinary chat")),
            Some(serde_json::json!(42)),
        ] {
            let artifact = TestDirectory::new();
            write_loadable_text_artifact(artifact.path());
            let tokenizer_path = artifact.path().join("tokenizer.json");
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
            tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
            tokenizer.with_decoder(Some(ByteLevel::default()));
            tokenizer
                .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
                .unwrap();
            tokenizer.save(&tokenizer_path, false).unwrap();
            std::fs::write(
                artifact.path().join("generation_config.json"),
                serde_json::json!({"eos_token_id": tokenizer.token_to_id("<|im_end|>").unwrap()})
                    .to_string(),
            )
            .unwrap();
            let path = if format == ArtifactFormat::SafeTensors {
                std::fs::write(
                    artifact.path().join("tokenizer_config.json"),
                    serde_json::json!({"chat_template": checkpoint_template}).to_string(),
                )
                .unwrap();
                artifact.path().to_path_buf()
            } else {
                // Both embedded and sidecar templates must be superseded.
                if checkpoint_template.is_some() {
                    std::fs::write(
                        artifact.path().join("chat_template.jinja"),
                        "ordinary sidecar chat",
                    )
                    .unwrap();
                }
                write_gguf(
                    artifact.path(),
                    checkpoint_template.as_ref().map(|value| {
                        value.as_str().map_or(MetadataValue::Uint32(42), |text| {
                            MetadataValue::String(text.into())
                        })
                    }),
                )
            };
            let baseline = inspect_text_model(
                ModelInspectionReport::unverified(&path, format),
                &TextModelOptions::default(),
                TextInspectionOptions::default(),
            );
            assert_eq!(
                baseline.chat_template,
                match &checkpoint_template {
                    None => InspectionReadiness::Missing,
                    Some(value) if value.is_string() => InspectionReadiness::Ready,
                    Some(_) => InspectionReadiness::Invalid,
                }
            );
            assert_ne!(baseline.semantic_streaming, InspectionReadiness::Ready);
            assert_ne!(baseline.native_tools, InspectionReadiness::Ready);

            for template in [
                QWEN_TEMPLATE.into(),
                ModelChatTemplate::Named(BTreeMap::from([
                    ("default".into(), QWEN_TEMPLATE.into()),
                    ("tool_use".into(), QWEN_TEMPLATE.into()),
                ])),
            ] {
                let options = TextModelOptions {
                    chat_template: Some(template),
                };
                for chat_request in [None, Some(request())] {
                    let report = inspect_text_model(
                        ModelInspectionReport::unverified(&path, format),
                        &options,
                        TextInspectionOptions { chat_request },
                    );
                    assert_eq!(report.tokenizer, InspectionReadiness::Ready);
                    assert_eq!(report.chat_template, InspectionReadiness::Ready);
                    assert_eq!(
                        report.semantic_streaming,
                        InspectionReadiness::Ready,
                        "{:?}",
                        report.issues
                    );
                    assert_eq!(
                        report.native_tools,
                        InspectionReadiness::Ready,
                        "{:?}",
                        report.issues
                    );
                    assert!(
                        report.issues.iter().all(|issue| {
                            issue.code == InspectionIssueCode::RequestSpecificValidation
                        }),
                        "{:?}",
                        report.issues
                    );
                }
                let template = options.chat_template.clone();
                let mut loaded = retained_artifacts(
                    LoadedModel::load_with_text_options(MockBackend, &path, (), options).unwrap(),
                    artifact.path(),
                    template,
                );
                let prepared = {
                    let request = request();
                    let cancellation = eredu_core::GenerationCancellationToken::new();
                    let source = loaded
                        .chat_source(
                            !request.tools.is_empty(),
                            &cancellation,
                        )
                        .unwrap()
                        .unwrap();
                    loaded
                        .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                        .unwrap()
                        .unwrap()
                };
                assert!(matches!(
                    prepared.semantic_support(),
                    eredu::runtime::chat::SemanticSupport::Supported
                ));
                assert!(prepared.native_tool_support().is_supported());
            }
        }
    }
}

#[test]
fn text_inspection_validates_the_override_instead_of_falling_back() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    std::fs::write(artifact.path().join("chat_template.jinja"), "valid chat").unwrap();
    let gguf = write_gguf(
        artifact.path(),
        Some(eredu_gguf::MetadataValue::String("valid chat".into())),
    );
    for (path, format) in [
        (artifact.path(), ArtifactFormat::SafeTensors),
        (gguf.as_path(), ArtifactFormat::Gguf),
    ] {
        for template in [
            ModelChatTemplate::Single("{% invalid %}".into()),
            ModelChatTemplate::Named(BTreeMap::from([("tool_use".into(), QWEN_TEMPLATE.into())])),
        ] {
            let options = TextModelOptions {
                chat_template: Some(template),
            };
            let report = inspect_text_model(
                ModelInspectionReport::unverified(path, format),
                &options,
                TextInspectionOptions {
                    chat_request: Some(request()),
                },
            );
            assert_eq!(report.semantic_streaming, InspectionReadiness::Unsupported);
            assert_eq!(report.native_tools, InspectionReadiness::Unsupported);
            assert!(report.issues.iter().any(|issue| {
                issue.code == InspectionIssueCode::UnsupportedSemanticProtocol
                    && issue.severity == InspectionSeverity::Error
            }));
            let template = options.chat_template.clone();
            let mut loaded = retained_artifacts(
                LoadedModel::load_with_text_options(MockBackend, path, (), options).unwrap(),
                artifact.path(),
                template,
            );
            assert!(matches!(
                loaded.chat_source(false, &Default::default()),
                Err(original_sources::FixtureSourceError::Chat(
                    eredu::api::ManagedChatSourceError::Source(_)
                ))
            ));
        }
    }
}
