use super::*;
use crate::api::PreparedChatGenerationSettings;
use crate::memory_fixture::{LedgerFixture as _, StorageFixture as _};

#[test]
fn public_prepared_chat_shares_cursor_authenticates_templates_and_retains_early_budget() {
    let mut previous: Option<(String, Vec<u32>)> = None;
    for manual in [false, true] {
        let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let mut config: serde_json::Value = serde_json::from_str(CHAT_CONFIG).unwrap();
        config["chat_template"] = serde_json::Value::String(format!(
            "{{{{ bos_token }}}}{{{{ caller_value.nested[-1] }}}}{}",
            config["chat_template"].as_str().unwrap()
        ));
        let config = config.to_string();
        let selected = eredu_text::tokenizer::load_model_chat_template_from_str(&config)
            .unwrap()
            .unwrap();
        let mut loaded_tokenizer = ChatTokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
        let defaults = serde_json::json!({"bos_token":"<s>","caller_value":"default", "messages":[{"role":"user","content":"default messages"}],"add_generation_prompt":false});
        loaded_tokenizer.set_template_kwargs(defaults.as_object().unwrap().clone());
        let mut model = LoadedModel::from_runtime(
            runtime,
            loaded_tokenizer,
            LoadedTextModelConfig {
                model_family: ModelKind::Llama,
                effective_model_type: "llama".into(),
                model_id: "smol".into(),
                chat_template: Some(selected.clone()),
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        let cancellation = GenerationCancellationToken::new();
        let mut tokenizer_file = tempfile::tempfile().unwrap();
        tokenizer_file.write_all(TOKENIZER.as_bytes()).unwrap();
        let tokenizer = model
            .compile_managed_plain_text_source(tokenizer_file)
            .unwrap();
        let mut template_file = tempfile::tempfile().unwrap();
        template_file.write_all(config.as_bytes()).unwrap();
        let source = model
            .compile_managed_chat_source(&tokenizer, template_file, false, &cancellation)
            .unwrap()
            .unwrap();
        let cold = pool.live_charge_bytes().unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                    "host".into(),
                    eredu_core::MemoryLimit::Finite(u64::MAX),
                )]),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut chat = request();
        // These permissions alone do not change ordinary text rendering.
        // Exercise both ordinary-driver and manually advanced public sessions.
        chat.parallel_tool_calls = crate::runtime::chat::ParallelToolCallPolicy::Enabled {
            max_calls: if manual {
                std::num::NonZeroUsize::new(2)
            } else {
                None
            },
        };
        chat.allow_unparsed_reasoning = true;
        assert!(crate::api::request::text_chat_eligibility(&chat).is_ok());
        chat.extra_template_kwargs=serde_json::json!({"caller_value":{"nested":[17,-0.0]},"messages":[{"role":"user","content":"caller 界"}],"add_generation_prompt":true}).as_object().unwrap().clone();
        let mut oracle = ChatTokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
        oracle.set_template_kwargs(defaults.as_object().unwrap().clone());
        let rendered = oracle
            .apply_chat_template_json(
                selected.clone(),
                [chat.messages.clone()],
                None,
                "smol",
                true,
                Some(&chat.extra_template_kwargs),
            )
            .unwrap()
            .pop()
            .unwrap();
        let expected_ids = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes())
            .unwrap()
            .encode(rendered, false)
            .unwrap();
        // Size the mock model from the actual ordinary rendered prompt plus
        // this request's output allowance, including all template overrides.
        facts.borrow_mut().maximum_context = Some(
            u64::try_from(expected_ids.len())
                .unwrap()
                .checked_add(u64::try_from(settings.overrides.max_new_tokens.unwrap()).unwrap())
                .unwrap(),
        );
        assert!(model
            .prepare_chat(
                &source,
                &chat,
                &crate::memory_fixture::limits(1),
                &cancellation
            )
            .is_err());
        assert_eq!(facts.borrow().chat_renders, 0);
        assert_eq!(facts.borrow().encodes, 0);
        assert_eq!(pool.live_charge_bytes().unwrap(), cold);

        let cancelled = GenerationCancellationToken::new();
        cancelled.cancel();
        assert!(model
            .prepare_chat(
                &source,
                &chat,
                &crate::memory_fixture::limits(u64::MAX),
                &cancelled
            )
            .unwrap()
            .is_none());
        assert_eq!(facts.borrow().chat_renders, 0);
        model.set_chat_template(Some(eredu_text::tokenizer::ModelChatTemplate::Single(
            "changed".into(),
        )));
        let error = model
            .prepare_chat(
                &source,
                &chat,
                &crate::memory_fixture::limits(u64::MAX),
                &cancellation,
            )
            .unwrap_err();
        assert_eq!(
            error.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        );
        drop(error);
        model.set_chat_template(Some(selected.clone()));
        assert_eq!(facts.borrow().chat_renders, 0);

        let mut reasoning = chat.clone();
        reasoning.enable_thinking = Some(true);
        reasoning.allow_unparsed_reasoning = false;
        let error = model
            .prepare_chat(
                &source,
                &reasoning,
                &crate::memory_fixture::limits(u64::MAX),
                &cancellation,
            )
            .unwrap_err();
        assert_eq!(facts.borrow().encodes, 0);
        assert!(!facts.borrow().order.contains(&"submit"));
        drop(error);
        assert_eq!(pool.live_charge_bytes().unwrap(), cold);

        let prepared = model
            .prepare_chat(
                &source,
                &chat,
                &crate::memory_fixture::limits(u64::MAX),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        let prepared_bytes = pool.live_charge_bytes().unwrap();
        assert!(prepared_bytes > cold);
        assert!(model
            .start_prepared_chat(literal_request(&prepared, settings.clone()), &cancelled)
            .unwrap()
            .is_none());
        assert_eq!(facts.borrow().encodes, 0);
        model.set_chat_template(Some(eredu_text::tokenizer::ModelChatTemplate::Single(
            "changed".into(),
        )));
        let error = match model
            .start_prepared_chat(literal_request(&prepared, settings.clone()), &cancellation)
        {
            Err(error) => error,
            Ok(_) => panic!("changed template must reject at execution too"),
        };
        assert_eq!(
            error.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        );
        drop(error);
        model.set_chat_template(Some(selected));
        assert_eq!(pool.live_charge_bytes().unwrap(), prepared_bytes);

        facts.borrow_mut().short = true;
        let error = match model
            .start_prepared_chat(literal_request(&prepared, settings.clone()), &cancellation)
        {
            Err(error) => error,
            Ok(_) => panic!("exact request one-short admission must refuse"),
        };
        assert!(pool.live_charge_bytes().unwrap() > prepared_bytes);
        assert!(!facts.borrow().order.contains(&"submit"));
        drop(error);
        assert_eq!(pool.live_charge_bytes().unwrap(), prepared_bytes);
        facts.borrow_mut().short = false;
        facts.borrow_mut().ids.clear();

        let mut visible = String::new();
        let mut emit = |event: SemanticEvent| {
            if let SemanticEvent::TextDelta(text) = event {
                visible.push_str(&text);
            }
        };
        let mut session = model
            .start_prepared_chat(literal_request(&prepared, settings.clone()), &cancellation)
            .unwrap()
            .unwrap();
        let output = if manual {
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut emit).unwrap();
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal prepared chat"))
        } else {
            session.run(&cancellation, &mut emit).unwrap()
        };
        assert_eq!(facts.borrow().ids, expected_ids.get_ids());
        assert_eq!(visible, "h hih");
        assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
        if let Some((text, ids)) = previous.take() {
            assert_eq!(visible, text);
            assert_eq!(output.token_ids.as_ref(), ids.as_slice());
        }
        previous = Some((visible, output.token_ids.as_ref().to_vec()));
        drop((prepared, source, tokenizer, model));
        assert!(
            pool.live_charge_bytes().unwrap() > 0,
            "terminal token output retains original owners"
        );
        assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
        drop(output);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}
