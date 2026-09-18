use super::*;
const INKLING: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_templates/inkling-small-8cc5877b.jinja"
));
#[test]
fn explicit_profile_uses_actual_named_effort_and_retires_after_shared_chat_execution() {
    for (manual, enabled) in [(false, false), (true, true)] {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let cancellation = GenerationCancellationToken::new();
        // This tiny decoder fixture deliberately has ID holes. Adding special
        // tokens uses vocabulary length, so first fill those unused entries to
        // preserve the existing generated Ġhi token at ID 8.
        let mut source: serde_json::Value = serde_json::from_str(TOKENIZER).unwrap();
        let vocabulary = source["model"]["vocab"].as_object_mut().unwrap();
        for (id, spelling) in [(5, "<unused-5>"), (6, "<unused-6>"), (7, "<unused-7>")] {
            vocabulary.insert(spelling.into(), serde_json::json!(id));
        }
        let source = serde_json::to_vec(&source).unwrap();
        let mut raw = tokenizers::Tokenizer::from_bytes(&source).unwrap();
        let structural = crate::api::request::probes::inkling::STRUCTURAL
            .iter()
            .chain(crate::api::request::probes::inkling::TOOL_STRUCTURAL.iter())
            .map(|spelling| tokenizers::AddedToken::from(*spelling, true).normalized(false))
            .collect::<Vec<_>>();
        raw.add_special_tokens(structural).unwrap();
        assert_eq!(raw.id_to_token(8).as_deref(), Some("Ġhi"));
        assert_eq!(raw.decode(&[0, 8, 0], true).unwrap(), "h hih");
        let json = raw.to_string(false).unwrap();
        let config = serde_json::json!({"chat_template": INKLING}).to_string();
        let (mut model, tokenizer, source) = loaded_sources(
            runtime,
            json.as_bytes(),
            &config,
            "unrelated-profile-id",
            ModelKind::Inkling,
            &cancellation,
        );
        let mut request = request();
        request.enable_thinking = Some(enabled);
        request.allow_unparsed_reasoning = true;
        request.extra_template_kwargs.insert(
            "reasoning_effort".into(),
            serde_json::json!(if enabled { "none" } else { "high" }),
        );
        let mut ordinary = eredu_text::tokenizer::Tokenizer::from_tokenizer(raw.clone());
        let mut kwargs = request.extra_template_kwargs.clone();
        kwargs.insert(
            "reasoning_effort".into(),
            serde_json::json!(if enabled { "high" } else { "none" }),
        );
        let expected = ordinary
            .apply_chat_template_json(
                eredu_text::tokenizer::ModelChatTemplate::Single(INKLING.into()),
                [request.messages.clone()],
                None,
                "unrelated-profile-id",
                true,
                Some(&kwargs),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = model
            .prepare_chat(&source, &request, u64::MAX, &cancellation)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.rendered_prompt(), expected);
        assert!(expected.contains(if enabled {
            "Thinking effort level: 0.9"
        } else {
            "Thinking effort level: 0"
        }));
        assert_eq!(
            model.token_validity.as_ref(),
            prepared.tokenizer_source().generation_domain().unwrap(),
            "loaded and original generation domains"
        );
        let escaped = prepared.render().clone();
        facts.borrow_mut().ids.clear();
        let mut session = model
            .start_prepared_chat(literal_request(&prepared, chat_settings()), &cancellation)
            .unwrap()
            .unwrap();
        assert_eq!(
            facts.borrow().ids,
            raw.encode(expected.as_str(), false).unwrap().get_ids()
        );
        let mut visible = String::new();
        let mut sink = |event: SemanticEvent| {
            if let SemanticEvent::TextDelta(text) = event {
                visible.push_str(&text);
            }
        };
        let output = if manual {
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut sink).unwrap();
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal explicit-profile session"))
        } else {
            session.run(&cancellation, &mut sink).unwrap()
        };
        assert_eq!(visible, "h hih");
        assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
        drop((
            prepared, request, ordinary, raw, json, source, tokenizer, model,
        ));
        assert_eq!(escaped.prompt(true), expected);
        assert!(pool.used_bytes().unwrap() > 0);
        drop((escaped, output));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn default_profile_without_controls_preserves_generation_policy_and_shared_cursor() {
    use crate::api::PreparedChatGenerationSettings;
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/muse-glimmer-30b-97c77dff.jinja"
    ));
    for manual in [false, true] {
        let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let mut source: serde_json::Value = serde_json::from_str(TOKENIZER).unwrap();
        let vocabulary = source["model"]["vocab"].as_object_mut().unwrap();
        for (id, spelling) in [(5, "<unused-5>"), (6, "<unused-6>"), (7, "<unused-7>")] {
            vocabulary.insert(spelling.into(), serde_json::json!(id));
        }
        let mut raw =
            tokenizers::Tokenizer::from_bytes(&serde_json::to_vec(&source).unwrap()).unwrap();
        raw.add_special_tokens(
            crate::api::request::probes::muse::STRUCTURAL
                .iter()
                .map(|spelling| tokenizers::AddedToken::from(*spelling, true).normalized(false))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(raw.id_to_token(8).as_deref(), Some("Ġhi"));
        let json = raw.to_string(false).unwrap();
        let selected = eredu_text::tokenizer::ModelChatTemplate::Single(TEMPLATE.into());
        let mut model = LoadedModel::from_runtime(
            runtime,
            ChatTokenizer::from_tokenizer(raw.clone()),
            LoadedTextModelConfig {
                model_family: ModelKind::MuseGlimmer,
                effective_model_type: "muse_glimmer".into(),
                model_id: "unrelated-profile-id".into(),
                chat_template: Some(selected),
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        let cancellation = GenerationCancellationToken::new();
        let mut chat = request();
        chat.add_generation_prompt = false;
        chat.allow_unparsed_reasoning = true;
        assert!(chat.enable_thinking.is_none() && chat.reasoning_effort.is_none());
        assert!(chat.extra_template_kwargs.is_empty());
        // Cold inspection includes the shared behavioral profile selector.
        // A raw false-toggle template render omits this suffix.
        let compiler = crate::runtime::chat::constraints::ConstraintCompiler::from_tokenizer(
            &model.tokenizer,
            &model.eos_token_ids,
        );
        let ordinary = crate::api::request::inspect_chat_from_parts(
            &mut model.tokenizer,
            model.chat_template.clone().unwrap(),
            &model.model_id,
            &model.eos_token_ids,
            Some(&compiler),
            chat.clone(),
        )
        .unwrap();
        assert_eq!(
            ordinary.format_profile_identity(),
            Some("muse-glimmer.atem.v1")
        );
        let prompt = ordinary.rendered_prompt.clone();
        assert!(prompt.ends_with("<|start|>assistant"));
        assert!(prompt.contains("Reasoning strength: high."));
        let expected = raw.encode(prompt.as_str(), false).unwrap();
        facts.borrow_mut().maximum_context = Some(expected.len() as u64 + 3);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(json.as_bytes()).unwrap();
        let tokenizer = model.compile_managed_plain_text_source(file).unwrap();
        let config = serde_json::json!({"chat_template":TEMPLATE}).to_string();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(config.as_bytes()).unwrap();
        let source = model
            .compile_managed_chat_source(&tokenizer, file, false, &cancellation)
            .unwrap()
            .unwrap();
        let cold = pool.used_bytes().unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                managed_memory_capacity_bytes: Some(u64::MAX),
                ..Default::default()
            },
            ..Default::default()
        };
        let prepared = model
            .prepare_chat(&source, &chat, u64::MAX, &cancellation)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.rendered_prompt(), prompt);
        assert_eq!(
            model.token_validity.as_ref(),
            prepared.tokenizer_source().generation_domain().unwrap(),
            "loaded and original generation domains"
        );
        let prepared_bytes = pool.used_bytes().unwrap();
        facts.borrow_mut().short = true;
        let error =
            match model.start_prepared_chat(literal_request(&prepared, settings), &cancellation) {
                Err(error) => error,
                Ok(_) => panic!("short source must refuse"),
            };
        assert!(pool.used_bytes().unwrap() > cold);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), prepared_bytes);
        facts.borrow_mut().short = false;
        facts.borrow_mut().ids.clear();
        let mut visible = String::new();
        let mut emit = |event: SemanticEvent| {
            if let SemanticEvent::TextDelta(text) = event {
                visible.push_str(&text);
            }
        };
        let output = if manual {
            let mut session = model
                .start_prepared_chat(literal_request(&prepared, settings), &cancellation)
                .unwrap()
                .unwrap();
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut emit).unwrap();
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal default-profile chat"))
        } else {
            model
                .start_prepared_chat(literal_request(&prepared, settings), &cancellation)
                .unwrap()
                .unwrap()
                .run(&cancellation, &mut emit)
                .unwrap()
        };
        assert_eq!(facts.borrow().ids, expected.get_ids());
        assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
        assert_eq!(visible, "h hih");
        let escaped = output.token_ids.clone();
        drop((
            output, prepared, source, tokenizer, model, ordinary, raw, json, chat,
        ));
        assert!(pool.used_bytes().unwrap() > 0);
        assert_eq!(escaped.as_ref(), &[0, 8, 0]);
        drop(escaped);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
