//! Public ordinary semantic sessions with an ordinary-only neutral backend.
use super::*;
use crate::api::{ManagedPlainTextSource, PreparedChatGenerationSettings, PreparedChatRequest};
use crate::runtime::chat::{ChatTemplateRequest, ToolChoice};
use std::io::Write;

const CAPACITY: u64 = 512 << 20;
const TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja"
));

#[test]
fn original_prompt_attribution_retains_its_producer_after_session_and_sources_retire() {
    let (mut model, source, facts, pool) = fixture("Seventeen.");
    let mut policy = request(ToolChoice::None);
    policy.tools.clear();
    let chat = prepare_chat(&model, &source, policy);
    let cancel = GenerationCancellationToken::new();
    let session = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel).unwrap().unwrap();
    let attribution = session.prompt_attribution().unwrap().clone();
    let ids = facts.borrow().ids.clone();
    assert_eq!(attribution.attribution().complete_token_ids(), Some(ids.as_slice()));
    assert_eq!(attribution.attribution().input_range(0).unwrap(), [0,ids.len() as u64]);
    assert_eq!(attribution.attribution().input_range(2).unwrap(), [ids.len() as u64+1,ids.len() as u64+2]);
    let output = session.run(&cancel,&mut |_|{}).unwrap();
    drop((output,chat,source,model));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(attribution.attribution().complete_token_ids(), Some(ids.as_slice()));
    drop(attribution);
    assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn prepared_pause_resume_preserves_choices_and_matches_uninterrupted_events() {
    use eredu_core::execution_control::{GenerationControlHandle, GenerationStatus};
    for choice in [ToolChoice::None, ToolChoice::Required, ToolChoice::Auto] {
        let text = if choice == ToolChoice::None { "Seventeen." } else {
            "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>"
        };
        let (mut model, source, facts, pool) = fixture(text);
        let mut policy = request(choice);
        policy.tools[0]["function"]["parameters"]["$schema"] = serde_json::json!("http://json-schema.org/draft-07/schema#");
        let chat = prepare_chat(&model, &source, policy);
        let cancel = GenerationCancellationToken::new();
        let control = GenerationControlHandle::new(cancel.clone());
        control.request_pause();
        let session = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel).unwrap().unwrap();
        assert_eq!(session.status(), GenerationStatus::Prepared);
        let mut events = Vec::new();
        let mut session = session.run_until_paused(&control, &mut |event| events.push(event)).unwrap();
        assert_eq!(session.status(), GenerationStatus::Paused);
        assert_eq!(session.next_prediction(), 0);
        assert!(events.is_empty());
        assert!(!facts.borrow().order.contains(&"submit"));
        let first = facts.borrow().prediction_sequence[0];
        session.force_next_token(first).unwrap();
        session.pause().unwrap();
        assert_eq!(session.pending_forced_token(), Some(first));
        session = session.advance(&cancel, &mut |event| events.push(event)).unwrap();
        assert_eq!(session.next_prediction(), 1);
        assert_eq!(session.status(), GenerationStatus::Paused);
        let prefix = session.token_ids().to_vec();
        session = session.run_until_paused(&control, &mut |event| events.push(event)).unwrap();
        assert_eq!(session.token_ids(), &prefix);
        let mut session = session.resume(&control, &mut |event| events.push(event)).unwrap();
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(session.next_prediction(), session.token_ids().len() as u64);
        assert!(session.timing().time_to_first_token().is_some());
        assert!(session.pause().is_err());
        let output = session.into_output().unwrap_or_else(|_| panic!("completed"));
        let expected = (output.token_ids.to_vec(), output.finish_reason, serde_json::to_value(&events).unwrap());
        drop((output, events));
        let mut events = Vec::new();
        let output = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel).unwrap().unwrap()
            .run(&cancel, &mut |event| events.push(event)).unwrap();
        assert_eq!(output.token_ids.as_ref(), expected.0);
        assert_eq!(output.finish_reason, expected.1);
        assert_eq!(serde_json::to_value(&events).unwrap(), expected.2);
        drop((output, events, chat, source, model));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn prepared_control_cancellation_wins_over_pause_without_inventing_a_prediction() {
    use eredu_core::execution_control::{GenerationControlHandle, GenerationStatus};
    let (mut model, source, facts, pool) = fixture("Seventeen.");
    let mut policy = request(ToolChoice::None);
    policy.tools.clear();
    let chat = prepare_chat(&model, &source, policy);
    let cancel = GenerationCancellationToken::new();
    let control = GenerationControlHandle::new(cancel.clone());
    let session = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel).unwrap().unwrap();
    control.request_pause();
    control.cancel();
    let mut events = Vec::new();
    let session = session.run_until_paused(&control, &mut |event| events.push(event)).unwrap();
    assert_eq!(session.status(), GenerationStatus::Cancelled);
    assert_eq!(session.next_prediction(), 0);
    assert_eq!(session.finish_reason(), Some(FinishReason::Cancelled));
    assert!(!facts.borrow().order.contains(&"submit"));
    assert!(session.token_ids().is_empty());
    drop(session);
    drop((events, chat, source, model));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn prepared_choices_validate_sources_without_advancing_and_commit_once() {
    for mode in 0..4 {
        let output = if mode >= 2 { "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>" } else { "Seventeen." };
        let (mut model, source, facts, pool) = if mode == 0 {
            fixture_with_template(output, "{{ messages[0].content }}", ModelKind::Qwen2)
        } else { fixture(output) };
        let mut policy = request(match mode { 0 | 1 => ToolChoice::None, 2 => ToolChoice::Required, _ => ToolChoice::Auto });
        if mode == 0 { policy.tools.clear(); }
        else { policy.tools[0]["function"]["parameters"]["$schema"] = serde_json::json!("http://json-schema.org/draft-07/schema#"); }
        let chat = prepare_chat(&model, &source, policy);
        let mut request = PreparedChatRequest::new(&chat, settings());
        if mode == 0 { request.output_mode = crate::api::PreparedChatOutputMode::Text; }
        let cancel = GenerationCancellationToken::new();
        let mut session = model.start_prepared_chat(request, &cancel).unwrap().unwrap();
        let predicted = facts.borrow().prediction_sequence.clone();
        let failed = session.force_next_token(u32::MAX).expect_err("invalid canonical ID");
        assert!(matches!(failed.token_choice_rejection(), Some(eredu_runtime::execution_control::TokenChoiceError::InvalidToken(u32::MAX))));
        assert!(session.token_ids().is_empty());
        assert!(session.pending_forced_token().is_none());
        assert!(!facts.borrow().order.contains(&"submit"));
        drop(failed);
        let mut events = Vec::new();
        for token in &predicted {
            if session.finish_reason().is_some() { break; }
            let before = session.token_ids().len();
            if before == 0 {
                session.force_next_token(*token).unwrap();
                assert_eq!(session.pending_forced_token(), Some(*token));
                let rejected = session.force_next_token(*token).expect_err("pending choice cannot be overwritten");
                assert!(matches!(rejected.token_choice_rejection(), Some(eredu_runtime::execution_control::TokenChoiceError::AlreadyPending)));
                drop(rejected);
                assert_eq!(session.token_ids().len(), before);
                assert!(session.clear_forced_token().unwrap());
                assert!(!session.clear_forced_token().unwrap());
                session.force_next_token(*token).unwrap();
            }
            session = session.advance(&cancel, &mut |event| events.push(event)).unwrap();
            assert_eq!(session.token_ids().len(), before + 1);
            assert_eq!(session.token_ids().last(), Some(token));
            assert!(session.pending_forced_token().is_none());
        }
        assert!(session.finish_reason().is_some());
        assert!(session.force_next_token(predicted[0]).is_err());
        let output = session.into_output().unwrap_or_else(|_| panic!("terminal session"));
        let forced_ids = output.token_ids.to_vec();
        let forced_finish = output.finish_reason;
        let forced_events = serde_json::to_value(&events).unwrap();
        drop((output, events));
        let mut request = PreparedChatRequest::new(&chat, settings());
        if mode == 0 { request.output_mode = crate::api::PreparedChatOutputMode::Text; }
        let mut events = Vec::new();
        let ordinary = model.start_prepared_chat(request, &cancel).unwrap().unwrap()
            .run(&cancel, &mut |event| events.push(event)).unwrap();
        assert_eq!(ordinary.token_ids.as_ref(), forced_ids.as_slice());
        assert_eq!(ordinary.finish_reason, forced_finish);
        assert_eq!(serde_json::to_value(&events).unwrap(), forced_events);
        drop((ordinary, events, source, chat, model));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn public_source_prepared_observer_uses_committed_delivery_before_semantic_events() {
    for manual in [false, true] {
        let (mut model, source, facts, pool) = fixture("Seventeen.");
        let mut policy = request(ToolChoice::None);
        policy.tools.clear();
        let chat = prepare_chat(&model, &source, policy);
        let cancellation = GenerationCancellationToken::new();
        let observed = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&observed);
        let mut observer = move |token: Option<u32>, frame: Option<eredu_core::capture::SharedCapturedStep>, seconds: f64| {
            assert!(frame.is_none());
            assert!(seconds >= 0.0);
            seen.borrow_mut().push(token.expect("unobserved committed token"));
        };
        let mut request = PreparedChatRequest::new(&chat, settings());
        request.options = Some(TextPreparationOptions { capture: None, interventions: None });
        let session = model.start_prepared_chat(request, &cancellation).unwrap().unwrap()
            .with_capture_observer(&mut observer);
        let mut events = Vec::new();
        let mut emit = |event| {
            assert!(!observed.borrow().is_empty(), "observer precedes semantic publication");
            events.push(event);
        };
        let output = if manual {
            let mut session = session;
            let mut scoped_tokens = Vec::new();
            while session.finish_reason().is_none() {
                let expected_forced = scoped_tokens.is_empty();
                if expected_forced {
                    session.force_next_token(facts.borrow().prediction_sequence[0]).unwrap();
                }
                let mut delivered_timing = None;
                let mut record = |token: Option<u32>, capture: Option<eredu_core::capture::SharedCapturedStep>, delivery: crate::api::request::TokenDeliveryFacts| {
                    assert!(capture.is_none());
                    assert_eq!(delivery.forced, expected_forced);
                    assert!(delivery.step_seconds >= 0.0);
                    assert!(delivery.timing.time_to_first_token().is_some());
                    delivered_timing = Some(delivery.timing);
                    let token = token.expect("committed scoped delivery");
                    assert_eq!(observed.borrow().last(), Some(&token), "persistent observer precedes scoped delivery");
                    scoped_tokens.push(token);
                };
                session = session.advance_with_delivery(&cancellation, Some(&mut record), None, &mut emit).unwrap();
                assert_eq!(delivered_timing, Some(session.timing()));
            }
            let count = observed.borrow().len();
            let mut unexpected = |_, _, _| panic!("terminal state cannot redeliver");
            session = session.advance_with_delivery(&cancellation, Some(&mut unexpected), None, &mut emit).unwrap();
            assert_eq!(observed.borrow().len(), count);
            assert_eq!(&*observed.borrow(), &scoped_tokens);
            session.into_output().unwrap_or_else(|_| panic!("terminal session"))
        } else {
            session.run(&cancellation, &mut emit).unwrap()
        };
        assert_eq!(&*observed.borrow(), output.token_ids.as_ref());
        assert_eq!(facts.borrow().empty_preparation_options, 1);
        drop((output, events, source, chat, model));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn prepared_controller_snapshot_aliases_preserve_independent_committed_prefixes() {
    use eredu_runtime::execution_control::SnapshotTokenController;
    fn history(controller: &crate::runtime::chat::constraints::ConstraintController) -> Vec<u32> {
        use eredu_core::speculative::PreparedGrammarController;
        if let Some(grammar) = controller.prepared_grammar() {
            grammar.prepared_grammar_source().history().to_vec()
        } else if let Some(source) = controller.prepared_forbidden_source() {
            source.history().to_vec()
        } else {
            controller.prepared_plain_source().unwrap().history().to_vec()
        }
    }
    for mode in 0..4 {
        let output = if mode >= 2 { "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>" } else { "Seventeen." };
        let (model, source, facts, pool) = if mode == 0 {
            fixture_with_template(output, "{{ messages[0].content }}", ModelKind::Qwen2)
        } else { fixture(output) };
        let mut policy = request(match mode { 0 | 1 => ToolChoice::None, 2 => ToolChoice::Required, _ => ToolChoice::Auto });
        if mode == 0 { policy.tools.clear(); }
        else { policy.tools[0]["function"]["parameters"]["$schema"] = serde_json::json!("http://json-schema.org/draft-07/schema#"); }
        let chat = prepare_chat(&model, &source, policy);
        let preparation = model.prepare_semantic_source(chat.tokenizer_source(), CAPACITY).unwrap();
        let funding = preparation.metadata_funding().clone();
        let domain = eredu_runtime::TokenDomain::new(preparation.tokenizer().generation_domain().unwrap().allowed_mask().unwrap().len());
        let output_mode = if mode == 0 { crate::api::PreparedChatOutputMode::Text } else { crate::api::PreparedChatOutputMode::Semantic };
        let mut prepared = model.prepare_chat_semantics(preparation, &chat, 256, std::num::NonZeroUsize::MIN, &[], true, output_mode).unwrap();
        let tokens = facts.borrow().prediction_sequence.clone();
        let controller = prepared.controller_mut();
        let split = tokens.len() / 2;
        for token in &tokens[..split] { controller.commit_token(*token).unwrap(); }
        let before = pool.used_bytes().unwrap();
        let saved = controller.fork_original_snapshot().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before, "saving immutable source aliases allocates nothing");
        assert_eq!(history(&saved), tokens[..split]);
        for token in &tokens[split..tokens.len()-1] { controller.commit_token(*token).unwrap(); }
        assert_eq!(history(&saved), tokens[..split]);
        let mut branch = saved.fork_original_snapshot().unwrap();
        for token in &tokens[split..tokens.len()-1] { branch.commit_token(*token).unwrap(); }
        assert_eq!(history(&branch), history(controller));
        assert_eq!(history(&saved), tokens[..split]);
        assert_eq!(branch.is_complete().unwrap(), controller.is_complete().unwrap());
        assert!(!matches!(branch.inference_storage(), eredu_core::TextControllerStorage::Unknown));
        let mut choices = eredu_runtime::execution_control::TokenChoiceController::new(saved, domain);
        choices.force_prepared_next(tokens[split], &funding).unwrap();
        let before_copy = pool.used_bytes().unwrap();
        let saved_choices = choices.fork_original_snapshot().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before_copy, "choice snapshots only copy scalar state and immutable source aliases");
        assert_eq!(saved_choices.pending_forced(), Some(tokens[split]));
        assert!(choices.clear_forced());
        assert_eq!(saved_choices.pending_forced(), Some(tokens[split]));
        assert_eq!(history(saved_choices.inner()), tokens[..split]);
        drop((branch, prepared, source, chat, model));
        assert!(pool.used_bytes().unwrap() > 0, "saved prefix retains its original source accounts");
        drop((choices, saved_choices, funding));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

fn fixture(
    output: &str,
) -> (
    LoadedModel<Backend>,
    ManagedPlainTextSource,
    Rc<RefCell<Facts>>,
    WorkingMemoryPool,
) {
    fixture_with_template(output, TEMPLATE, ModelKind::Qwen2)
}
fn fixture_with_template(
    output: &str,
    template: &str,
    family: ModelKind,
) -> (
    LoadedModel<Backend>,
    ManagedPlainTextSource,
    Rc<RefCell<Facts>>,
    WorkingMemoryPool,
) {
    use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};
    let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
    alphabet.sort_unstable();
    let vocabulary: tokenizers::models::bpe::Vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, ch)| (ch.to_string(), index as u32))
        .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens(
            ["<|im_start|>", "<|im_end|>"]
                .map(|text| AddedToken::from(text, true).normalized(false)),
        )
        .unwrap();
    tokenizer
        .add_tokens(
            ["<tool_call>", "</tool_call>", "<think>", "</think>"]
                .map(|text| AddedToken::from(text, false).normalized(false)),
        )
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut predictions = tokenizer.encode(output, false).unwrap().get_ids().to_vec();
    predictions.push(eos);
    let json = tokenizer.to_string(false).unwrap();
    let (runtime, facts, pool) = bare_runtime_with_capacity(CAPACITY);
    {
        let mut facts = facts.borrow_mut();
        facts.prediction_sequence = predictions;
        facts.output_width = Some(tokenizer.get_vocab_size(true));
        facts.maximum_context = Some(16_384);
        facts.shared_admission_capacity = Some(CAPACITY);
    }
    let model = LoadedModel::from_runtime(
        runtime,
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: family,
            effective_model_type: "qwen".into(),
            model_id: "ordinary-semantic-fixture".into(),
            chat_template: Some(template.into()),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(json.as_bytes()).unwrap();
    let source = model.compile_managed_plain_text_source(file).unwrap();
    (model, source, facts, pool)
}

fn prepare_chat(
    model: &LoadedModel<Backend>,
    tokenizer: &ManagedPlainTextSource,
    request: ChatTemplateRequest,
) -> crate::runtime::chat::PreparedChat {
    let template = match model.chat_template.as_ref().unwrap() {
        eredu_text::tokenizer::ModelChatTemplate::Single(template) => serde_json::json!(template),
        eredu_text::tokenizer::ModelChatTemplate::Named(templates) => serde_json::Value::Array(
            templates
                .iter()
                .map(|(name, template)| serde_json::json!({"name":name,"template":template}))
                .collect(),
        ),
    };
    let config = serde_json::json!({"chat_template":template}).to_string();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(config.as_bytes()).unwrap();
    let cancel = GenerationCancellationToken::new();
    let source = model
        .compile_managed_chat_source(tokenizer, file, !request.tools.is_empty(), &cancel)
        .unwrap()
        .unwrap();
    model
        .prepare_chat(&source, &request, CAPACITY, &cancel)
        .unwrap()
        .unwrap()
}

#[test]
fn public_prepared_chat_retains_original_policy_render_and_compilation_without_registration() {
    for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
        let (model, source, facts, pool) = fixture("Seventeen.");
        let mut request = request(choice);
        request.tools[0]["function"]["parameters"]["$schema"] =
            serde_json::json!("http://json-schema.org/draft-07/schema#");
        let chat = prepare_chat(&model, &source, request);
        assert_eq!(
            chat.generation_runtime_plan().unwrap().tool_choice(),
            choice
        );
        assert!(chat.rendered_prompt().contains("Read value 17."));
        assert_eq!(chat.capacity(), CAPACITY);
        assert_eq!(chat.eos_token_ids(), model.eos_token_ids);
        let sources = chat.controller_sources();
        assert!(
            pool.validate_shared_controller_source(eredu_core::SharedControllerSource::Bytes(
                sources.recipe.unwrap()
            ))
            .is_err()
        );
        chat.compilation().validate_sources(sources, &pool).unwrap();
        let used = pool.used_bytes().unwrap();
        let alias = chat.clone();
        assert_eq!(alias, chat);
        assert_eq!(
            alias.rendered_prompt().as_ptr(),
            chat.rendered_prompt().as_ptr()
        );
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert!(!facts.borrow().order.contains(&"submit"));
        drop((chat, source, model));
        assert!(pool.used_bytes().unwrap() > 0);
        assert!(alias.rendered_prompt().contains("Read value 17."));
        drop(alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn public_source_prepared_text_uses_one_manual_and_uninterrupted_cursor() {
    for manual in [false, true] {
        let (mut model, source, facts, pool) = fixture("Seventeen.");
        let mut request = request(ToolChoice::None);
        request.tools[0]["function"]["parameters"]["$schema"] =
            serde_json::json!("http://json-schema.org/draft-07/schema#");
        let chat = prepare_chat(&model, &source, request);
        let cancellation = GenerationCancellationToken::new();
        let mut events = Vec::new();
        let session = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancellation)
            .unwrap().unwrap();
        let output = if manual {
            let mut session = session;
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut |event| events.push(event)).unwrap();
            }
            session.into_output().unwrap_or_else(|_| panic!("terminal ordinary session"))
        } else {
            session.run(&cancellation, &mut |event| events.push(event)).unwrap()
        };
        let text = events.iter().filter_map(|event| match event {
            SemanticEvent::TextDelta(text) => Some(text.as_str()), _ => None,
        }).collect::<String>();
        assert_eq!(text, "Seventeen.");
        assert_eq!(events.iter().filter(|event| matches!(event, SemanticEvent::Finished { .. })).count(), 1);
        assert!(facts.borrow().speculative_seeds.is_empty());
        assert!(facts.borrow().speculative_prompts.is_empty());
        drop((model, source, chat));
        assert!(pool.used_bytes().unwrap() > 0);
        drop((output, events));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn public_source_prepared_plain_template_uses_the_same_semantic_cursor() {
    for manual in [false, true] {
        let (mut model, source, facts, pool) = fixture_with_template(
            "Seventeen.", "{{ messages[0].content }}", ModelKind::Qwen2);
        let mut request = request(ToolChoice::None);
        request.tools.clear();
        let chat = prepare_chat(&model, &source, request);
        assert!(chat.generation_runtime_plan().is_none());
        let cancellation = GenerationCancellationToken::new();
        let error = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancellation)
            .err().expect("unrecognized protocol requires explicit literal output");
        assert_eq!(error.semantic_output_rejection().as_ref(), Some(chat.semantic_support()));
        assert!(!facts.borrow().order.contains(&"admit"));
        drop(error);
        let mut generation = PreparedChatRequest::new(&chat, settings());
        generation.output_mode = crate::api::PreparedChatOutputMode::Text;
        let session = model.start_prepared_chat(generation, &cancellation)
            .unwrap().unwrap();
        let mut events = Vec::new();
        let output = if manual {
            let mut session = session;
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut |event| events.push(event)).unwrap();
            }
            session.into_output().unwrap_or_else(|_| panic!("terminal session"))
        } else {
            session.run(&cancellation, &mut |event| events.push(event)).unwrap()
        };
        let text = events.iter().filter_map(|event| match event {
            SemanticEvent::TextDelta(text) => Some(text.as_str()), _ => None,
        }).collect::<String>();
        assert_eq!(text, "Seventeen.");
        assert_eq!(events.iter().filter(|event| matches!(event, SemanticEvent::Finished { .. })).count(), 1);
        assert!(facts.borrow().speculative_seeds.is_empty());
        drop((source, chat, model));
        assert!(pool.used_bytes().unwrap() > 0);
        drop((events, output));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn public_chat_source_file_selects_named_tool_template_and_rejects_wrong_entry() {
    let (mut model, tokenizer, facts, pool) = fixture("Seventeen.");
    let config = serde_json::json!({"chat_template": [
        {"name":"default", "template":"plain:{{ messages[0].content }}"},
        {"name":"tool_use", "template":"tools:{{ messages[0].content }}|{{ tools|tojson }}"},
    ]})
    .to_string();
    model.chat_template =
        eredu_text::tokenizer::load_model_chat_template_from_str(&config).unwrap();
    let cancellation = GenerationCancellationToken::new();
    let file = |config: &str| {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(config.as_bytes()).unwrap();
        file
    };
    for has_tools in [false, true] {
        let source = model
            .compile_managed_chat_source(&tokenizer, file(&config), has_tools, &cancellation)
            .unwrap()
            .unwrap();
        drop(source);
    }
    let wrong = serde_json::json!({"chat_template": [
        {"name":"default", "template":"plain:{{ messages[0].content }}"},
    ]})
    .to_string();
    let failure = model
        .compile_managed_chat_source(&tokenizer, file(&wrong), true, &cancellation)
        .unwrap_err();
    assert!(failure.to_string().contains("identity"), "{failure}");
    assert_eq!(facts.borrow().chat_renders, 0);
    assert!(!facts.borrow().order.contains(&"submit"));
    drop((model, tokenizer, failure));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_bound_semantic_render_and_policy_use_real_tools_and_shared_sources() {
    use crate::api::portable::original_token_input::chat::prepare_original_chat;
    use eredu_text::chat_storage::ChatTemplatePlan;
    for (choice, recognized) in [
        (ToolChoice::None, true),
        (ToolChoice::Auto, true),
        (ToolChoice::Required, true),
        (ToolChoice::None, false),
    ] {
        let (mut model, source, facts, pool) = fixture_with_template(
            "Seventeen.",
            if recognized {
                TEMPLATE
            } else {
                "{{ messages[0].content }}"
            },
            ModelKind::Qwen2,
        );
        let mut request = request(choice);
        if !recognized {
            request.tools.clear();
        }
        let template = model.chat_template.as_ref().unwrap().clone();
        let compiled = pool
            .compile_chat_template(
                ChatTemplatePlan::prepare_model(
                    &template,
                    &model.model_id,
                    !request.tools.is_empty(),
                )
                .unwrap(),
            )
            .unwrap();
        let expected = model
            .tokenizer
            .apply_chat_template_json(
                template,
                [request.messages.clone()],
                Some(if choice == ToolChoice::None {
                    &[]
                } else {
                    &request.tools
                }),
                &model.model_id,
                true,
                Some(&request.extra_template_kwargs),
            )
            .unwrap()
            .remove(0);
        let (prepared, (policy, compilation)) = prepare_original_chat(
            &model.runtime,
            &compiled,
            source.original(),
            &request,
            Some(model.tokenizer.template_kwargs()),
            &model.eos_token_ids,
            CAPACITY,
            &GenerationCancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(prepared.prompt(true), expected);
        assert_eq!(prepared.profile().tool_dialect.is_some(), recognized);
        assert!(prepared.render().has_sources(&compiled, source.original()));
        use eredu_runtime::working_memory::ControllerCompilationOutput;
        compilation
            .validate_sources(policy.controller_sources(), &pool)
            .unwrap();
        assert_eq!(policy.metadata().eos_token_ids, model.eos_token_ids);
        assert_eq!(
            policy
                .metadata()
                .generation_runtime_plan
                .as_ref()
                .map(|plan| plan.tool_choice()),
            recognized.then_some(choice)
        );
        assert_eq!(
            policy
                .metadata()
                .selection
                .native_tool_support
                .is_supported(),
            recognized
        );
        let policy_alias = policy.clone();
        assert!(!facts.borrow().order.contains(&"submit"));
        let render = prepared.render().clone();
        drop((
            prepared,
            policy,
            source,
            model,
            compiled,
            request,
            compilation,
        ));
        assert!(pool.used_bytes().unwrap() > 0);
        assert_eq!(render.prompt(true), expected);
        drop(render);
        assert!(pool.used_bytes().unwrap() > 0);
        drop(policy_alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn source_bound_policy_receipts_keep_the_actual_compiler_outputs() {
    use crate::api::portable::original_token_input::chat::prepare_original_chat;
    use eredu_runtime::working_memory::ControllerCompilationOutput;
    use eredu_text::chat_storage::ChatTemplatePlan;
    for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
        let (model, source, facts, pool) = fixture("Seventeen.");
        let mut request = request(choice);
        request.tools[0]["function"]["parameters"]["$schema"] =
            serde_json::json!("http://json-schema.org/draft-07/schema#");
        let template = pool
            .compile_chat_template(
                ChatTemplatePlan::prepare_model(
                    model.chat_template.as_ref().unwrap(),
                    &model.model_id,
                    true,
                )
                .unwrap(),
            )
            .unwrap();
        let (rendered, (policy, compilation)) = prepare_original_chat(
            &model.runtime,
            &template,
            source.original(),
            &request,
            Some(model.tokenizer.template_kwargs()),
            &model.eos_token_ids,
            CAPACITY,
            &GenerationCancellationToken::new(),
        )
        .unwrap()
        .unwrap();
        compilation
            .validate_sources(policy.controller_sources(), &pool)
            .unwrap();
        assert!(
            compilation
                .metadata_funding()
                .same_account(rendered.metadata_funding())
        );
        let owners = policy.controller_sources();
        assert!(
            owners
                .recipe
                .unwrap()
                .retains_funding(compilation.metadata_funding())
        );
        assert!(
            owners
                .grammar
                .unwrap()
                .retains_funding(compilation.metadata_funding())
        );
        assert_eq!(owners.validation.is_some(), choice != ToolChoice::None);
        if let Some(validation) = owners.validation {
            assert!(validation.retains_funding(compilation.metadata_funding()));
        }
        assert!(
            pool.validate_shared_controller_source(eredu_core::SharedControllerSource::Bytes(
                owners.recipe.unwrap(),
            ))
            .is_err()
        );
        let used = pool.used_bytes().unwrap();
        let alias = policy.clone();
        let receipt_alias = compilation.clone();
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert!(!facts.borrow().order.contains(&"submit"));
        drop((
            model,
            source,
            request,
            rendered,
            template,
            policy,
            compilation,
        ));
        receipt_alias
            .validate_sources(alias.controller_sources(), &pool)
            .unwrap();
        assert!(pool.used_bytes().unwrap() > 0);
        drop((alias, receipt_alias));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn cancellation_after_policy_compilation_prevents_actual_request_rendering() {
    use crate::api::portable::original_token_input::chat::prepare_with_policy;
    use eredu_text::chat_storage::ChatTemplatePlan;
    let (model, source, facts, pool) = fixture("Seventeen.");
    let mut request = request(ToolChoice::Required);
    request.tools[0]["function"]["parameters"]["$schema"] =
        serde_json::json!("http://json-schema.org/draft-07/schema#");
    let template = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_model(
                model.chat_template.as_ref().unwrap(),
                &model.model_id,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let cold = pool.used_bytes().unwrap();
    let cancellation = GenerationCancellationToken::new();
    let mut probe_renders = None;
    let prepared = prepare_with_policy(
        &model.runtime,
        &template,
        source.original(),
        &request,
        Some(model.tokenizer.template_kwargs()),
        CAPACITY,
        &cancellation,
        |profile| {
            let outputs = crate::api::request::policy::compile_original(
                &model.runtime,
                profile.profile(),
                &request,
                &model.eos_token_ids,
                profile.preparation(),
            )?;
            probe_renders = Some(facts.borrow().chat_renders);
            cancellation.cancel();
            Ok(outputs)
        },
    )
    .unwrap();
    assert!(prepared.is_none());
    assert_eq!(Some(facts.borrow().chat_renders), probe_renders);
    assert!(!facts.borrow().order.contains(&"submit"));
    assert_eq!(pool.used_bytes().unwrap(), cold);
    drop((model, source, template, request));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_bound_schema_rejection_precedes_actual_request_rendering() {
    use crate::api::portable::original_token_input::chat::prepare_original_chat;
    use eredu_text::chat_storage::ChatTemplatePlan;
    use std::error::Error;
    let template_text = format!(
        "{{% if messages[0].content == 'reject actual request render' %}}{{{{ raise_exception('actual request reached renderer') }}}}{{% endif %}}{TEMPLATE}"
    );
    let (model, source, facts, pool) =
        fixture_with_template("Seventeen.", &template_text, ModelKind::Qwen2);
    let template = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_model(
                model.chat_template.as_ref().unwrap(),
                &model.model_id,
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let mut request = request(ToolChoice::Required);
    request.messages[0]["content"] = serde_json::json!("reject actual request render");
    request.tools[0]["function"]["parameters"] = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#", "type": 17,
    });
    let failure = prepare_original_chat(
        &model.runtime,
        &template,
        source.original(),
        &request,
        Some(model.tokenizer.template_kwargs()),
        &model.eos_token_ids,
        CAPACITY,
        &GenerationCancellationToken::new(),
    )
    .unwrap_err();
    let mut cause: &(dyn Error + 'static) = &failure;
    let mut compiled = false;
    let mut rendered = false;
    loop {
        compiled |= cause.is::<crate::runtime::chat::tool_schema::original::CompilationFailure>();
        rendered |= cause.is::<eredu_runtime::working_memory::OriginalChatRenderOperationError>();
        let Some(next) = cause.source() else { break };
        cause = next;
    }
    assert!(
        compiled,
        "invalid schema must fail in policy compilation: {failure}"
    );
    assert!(!rendered, "the actual request must not reach rendering");
    let probe_renders = facts.borrow().chat_renders;
    assert!(probe_renders > 0, "behavioral profile probes still run");
    assert!(!facts.borrow().order.contains(&"submit"));
    drop(failure);
    // The same request with valid declarations reaches the deliberately rejected
    // render, proving the earlier result was not a profile/template refusal.
    request.tools[0]["function"]["parameters"] = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": false,
    });
    let failure = prepare_original_chat(
        &model.runtime,
        &template,
        source.original(),
        &request,
        Some(model.tokenizer.template_kwargs()),
        &model.eos_token_ids,
        CAPACITY,
        &GenerationCancellationToken::new(),
    )
    .unwrap_err();
    assert_eq!(
        facts.borrow().chat_renders,
        2 * probe_renders + 1,
        "valid declarations must reach the one actual request render: {failure}"
    );
    drop((model, source, request, template));
    assert!(
        pool.used_bytes().unwrap() > 0,
        "failed render retains original sources"
    );
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn public_ordinary_semantic_policies_preserve_text_reasoning_and_automatic_tools() {
    check_public_semantic_policies(None);
}

#[test]
fn public_source_prepared_draft7_policies_preserve_text_reasoning_and_automatic_tools() {
    check_public_semantic_policies(Some("http://json-schema.org/draft-07/schema#"));
}

fn check_public_semantic_policies(schema: Option<&str>) {
    const QWEN3: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja"
    ));
    let call = "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>";
    for (tools, choice, output, reasoning) in [
        (false, ToolChoice::None, "Seventeen.", false),
        (true, ToolChoice::None, "Seventeen.", false),
        (true, ToolChoice::Auto, "Seventeen.", false),
        (true, ToolChoice::Auto, call, false),
        (
            false,
            ToolChoice::None,
            "<think>\nCheck the value.\n</think>Seventeen.",
            true,
        ),
    ] {
        let mut baseline = None;
        for manual in [false, true] {
            let (mut model, source, _, pool) = fixture_with_template(
                output,
                if reasoning { QWEN3 } else { TEMPLATE },
                if reasoning {
                    ModelKind::Qwen3
                } else {
                    ModelKind::Qwen2
                },
            );
            let mut request = request(choice);
            if !tools {
                request.tools.clear();
            } else if let Some(schema) = schema {
                request.tools[0]["function"]["parameters"]["$schema"] = serde_json::json!(schema);
            }
            if reasoning {
                request.enable_thinking = Some(true);
                request.add_generation_prompt = false;
            }
            let chat = prepare_chat(&model, &source, request);
            let cancel = GenerationCancellationToken::new();
            let mut events = Vec::new();
            let mut session = model
                .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
                .unwrap()
                .unwrap();
            let result = if manual {
                while session.finish_reason().is_none() {
                    session = session
                        .advance(&cancel, &mut |event| events.push(event))
                        .unwrap();
                }
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal session"))
            } else {
                session
                    .run(&cancel, &mut |event| events.push(event))
                    .unwrap()
            };
            let text = events
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::TextDelta(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            let thought = events
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            if output == call {
                assert!(events.iter().any(|event| matches!(event, SemanticEvent::ToolCallStart { name, .. } if name.as_str() == "reading")));
                assert!(text.is_empty());
            } else {
                assert_eq!(text, "Seventeen.");
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, SemanticEvent::ToolCallStart { .. }))
                );
            }
            assert_eq!(thought, if reasoning { "Check the value." } else { "" });
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                    .count(),
                1
            );
            let normalized = events
                .iter()
                .map(|event| match event {
                    SemanticEvent::ToolCallStart { index, name, .. } => {
                        format!("start:{index}:{name}")
                    }
                    event => format!("{event:?}"),
                })
                .collect::<Vec<_>>();
            if let Some(expected) = &baseline {
                assert_eq!(&normalized, expected);
            } else {
                baseline = Some(normalized);
            }
            drop((result, events, chat, model, source));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

fn request(choice: ToolChoice) -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user","content":"Read value 17."})],
        tools: vec![serde_json::json!({"type":"function", "function": {
            "name":"reading", "description":"Return the observed integer.",
            "parameters":{"type":"object", "properties":{"value":{"type":"integer"}},
                "required":["value"], "additionalProperties":false}
        }})],
        tool_choice: choice,
        add_generation_prompt: true,
        ..Default::default()
    }
}
fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(96),
            temperature: Some(0.0),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(CAPACITY),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn public_ordinary_tool_session_commits_split_arguments_and_matches_manual_progress() {
    check_public_ordinary_tool_session(None, ToolChoice::Required);
}

#[test]
fn public_source_prepared_draft7_tool_session_preserves_split_arguments_and_manual_parity() {
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        check_public_ordinary_tool_session(Some("http://json-schema.org/draft-07/schema#"), choice);
    }
}

fn check_public_ordinary_tool_session(schema: Option<&str>, choice: ToolChoice) {
    let mut baseline = None;
    for manual in [false, true] {
        let (mut model, source, facts, pool) = fixture(
            "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>",
        );
        let mut request = request(choice);
        if let Some(schema) = schema {
            request.tools[0]["function"]["parameters"]["$schema"] = serde_json::json!(schema);
        }
        let chat = prepare_chat(&model, &source, request);
        let cancel = GenerationCancellationToken::new();
        let mut events = Vec::new();
        let session = model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
            .unwrap()
            .unwrap();
        let output = if manual {
            let mut session = session;
            while session.finish_reason().is_none() {
                session = session
                    .advance(&cancel, &mut |event| events.push(event))
                    .unwrap();
            }
            let count = events.len();
            session = session
                .advance(&cancel, &mut |event| events.push(event))
                .unwrap();
            assert_eq!(events.len(), count, "terminal publication occurs once");
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal session"))
        } else {
            session
                .run(&cancel, &mut |event| events.push(event))
                .unwrap()
        };
        assert!(
            matches!(&events[0], SemanticEvent::ToolCallStart { index: 0, name, .. }
            if name.as_str() == "reading")
        );
        let fragments = events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment,
                } => Some(json_fragment.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            fragments.len() > 1,
            "arguments arrive across committed tokens"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fragments.concat()).unwrap(),
            serde_json::json!({"value":17})
        );
        assert!(matches!(
            events[events.len() - 2],
            SemanticEvent::ToolCallEnd
        ));
        assert!(matches!(
            events.last(),
            Some(SemanticEvent::Finished { .. })
        ));
        let normalized = events
            .iter()
            .map(|event| match event {
                SemanticEvent::ToolCallStart { index, name, .. } => format!("start:{index}:{name}"),
                event => format!("{event:?}"),
            })
            .collect::<Vec<_>>();
        if let Some(expected) = &baseline {
            assert_eq!(&normalized, expected);
        } else {
            baseline = Some(normalized);
        }
        assert!(facts.borrow().speculative_seeds.is_empty());
        assert!(facts.borrow().speculative_prompts.is_empty());
        drop((model, source, chat));
        assert!(
            pool.used_bytes().unwrap() > 0,
            "escaped output/events keep funding"
        );
        drop((output, events));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn public_ordinary_session_cancellation_and_foreign_origin_precede_preparation() {
    let (mut model, source, facts, _pool) = fixture("hello");
    let chat = prepare_chat(&model, &source, request(ToolChoice::None));
    let cancel = GenerationCancellationToken::new();
    cancel.cancel();
    let before = facts.borrow().encodes;
    assert!(
        model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
            .unwrap()
            .is_none()
    );
    assert_eq!(facts.borrow().encodes, before);
    let mut short = settings();
    short.inference.managed_memory_capacity_bytes = Some(1);
    assert!(
        model
            .start_prepared_chat(PreparedChatRequest::new(&chat, short), &GenerationCancellationToken::new())
            .is_err()
    );
    assert_eq!(
        facts.borrow().encodes,
        before,
        "refuse before prompt encoding"
    );
    assert!(!facts.borrow().order.contains(&"submit"));
    let (mut foreign, _foreign_source, foreign_facts, _other_pool) = fixture("hello");
    let error = foreign
        .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &GenerationCancellationToken::new())
        .err()
        .expect("foreign loaded origin");
    assert_eq!(
        error.input_rejection(),
        Some(TokenInputRejection::IdentityMismatch)
    );
    assert_eq!(foreign_facts.borrow().encodes, 0);
    model.set_chat_template(Some("changed template".into()));
    let error = model
        .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &GenerationCancellationToken::new())
        .err()
        .expect("changed selected template");
    assert_eq!(
        error.input_rejection(),
        Some(TokenInputRejection::IdentityMismatch)
    );
    assert_eq!(facts.borrow().encodes, before);
}

#[test]
fn public_ordinary_tool_cancellation_preserves_committed_partial_arguments() {
    let mut baseline = None;
    for manual in [false, true] {
        let (mut model, source, facts, pool) = fixture(
            "<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>",
        );
        let chat = prepare_chat(&model, &source, request(ToolChoice::Required));
        let cancel = GenerationCancellationToken::new();
        let mut events = Vec::new();
        let mut emit = |event| {
            if matches!(&event, SemanticEvent::ToolArgumentsDelta { .. }) {
                cancel.cancel();
            }
            events.push(event);
        };
        let session = model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
            .unwrap()
            .unwrap();
        let output = if manual {
            let mut session = session;
            while session.finish_reason().is_none() {
                session = session.advance(&cancel, &mut emit).unwrap();
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("cancelled terminal state"))
        } else {
            session.run(&cancel, &mut emit).unwrap()
        };
        assert!(matches!(
            events.last(),
            Some(SemanticEvent::Finished {
                reason: FinishReason::Cancelled
            })
        ));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolArgumentsDelta { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                .count(),
            1
        );
        let normalized = events
            .iter()
            .map(|event| match event {
                SemanticEvent::ToolCallStart { index, name, .. } => format!("start:{index}:{name}"),
                event => format!("{event:?}"),
            })
            .collect::<Vec<_>>();
        if let Some(expected) = &baseline {
            assert_eq!(&normalized, expected);
        } else {
            baseline = Some(normalized);
        }
        assert!(facts.borrow().next_prediction < facts.borrow().prediction_sequence.len());
        drop((model, source, chat));
        assert!(pool.used_bytes().unwrap() > 0);
        drop((output, events));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn failed_chat_cursor_exposes_its_committed_prefix_without_detaching_funding() {
    let (mut model, source, facts, pool) = fixture("Seventeen.");
    let mut policy = request(ToolChoice::None);
    policy.tools.clear();
    let chat = prepare_chat(&model, &source, policy);
    let cancel = GenerationCancellationToken::new();
    let mut events = Vec::new();
    let session = model.start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
        .unwrap().unwrap().advance(&cancel, &mut |event| events.push(event)).unwrap();
    let prefix = session.token_ids().to_vec();
    assert_eq!(prefix.len(), 1);
    facts.borrow_mut().fail_step = true;
    let error = session.advance(&cancel, &mut |event| events.push(event)).err().unwrap();
    assert_eq!(error.committed_token_ids(), Some(prefix.as_slice()));
    drop((events, chat, source, model));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(error.committed_token_ids(), Some(prefix.as_slice()));
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn canonical_explicit_prefix_uses_exact_ids_without_reencoding_chat_render() {
    let (mut model, source, facts, pool) = fixture("Seventeen.");
    let mut policy = request(ToolChoice::None);
    policy.tools.clear();
    let chat = prepare_chat(&model, &source, policy);
    let cancel = GenerationCancellationToken::new();
    // A nontext prefix intentionally differs from the rendered user prompt.
    let prefix = [13, 29, 13, 41];
    let mut request = PreparedChatRequest::new(&chat, settings());
    request.input = crate::api::PreparedChatPrompt::TokenIds(&prefix);
    let session = model.start_prepared_chat(request, &cancel).unwrap().unwrap();
    assert_eq!(facts.borrow().ids, prefix);
    let attribution = session.prompt_attribution().unwrap().clone();
    assert_eq!(attribution.attribution().complete_token_ids(), Some(prefix.as_slice()));
    let output = session.run(&cancel, &mut |_| {}).unwrap();
    assert!(!output.token_ids.is_empty());
    drop((output, chat, source, model));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(attribution.attribution().complete_token_ids(), Some(prefix.as_slice()));
    drop(attribution);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn explicit_prefix_refuses_unknown_ids_before_generation_admission() {
    let (mut model, source, facts, _) = fixture("Seventeen.");
    let mut policy = request(ToolChoice::None);
    policy.tools.clear();
    let chat = prepare_chat(&model, &source, policy);
    let mut request = PreparedChatRequest::new(&chat, settings());
    request.input = crate::api::PreparedChatPrompt::TokenIds(&[u32::MAX]);
    let error = model.start_prepared_chat(request, &GenerationCancellationToken::new()).err().unwrap();
    assert_eq!(error.input_rejection(), Some(eredu_core::TokenInputRejection::InvalidToken));
    assert!(facts.borrow().order.is_empty());
}

#[path = "prepared_semantic/recorded.rs"]
mod recorded;
