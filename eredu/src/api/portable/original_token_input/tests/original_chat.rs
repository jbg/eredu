use super::super::chat::{
    compile_original_chat_file, prepare_original_text_chat, start_original_text_chat,
};
use super::*;
use crate::runtime::chat::ChatTemplateRequest;
use eredu_runtime::working_memory::OriginalTextSourceBudget;
use eredu_text::chat_storage::{ChatMessages, ChatRenderContext, ChatTemplatePlan};
use std::io::Write as _;

impl OriginalChatBackend for Backend {
    fn prepare_original_chat_profile(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        capacity: u64,
    ) -> Result<OriginalChatProfilePreparation, OriginalChatProfileError> {
        OriginalChatProfilePreparation::new(
            template,
            tokenizer,
            &runtime.backend().execution,
            capacity,
        )
    }

    fn validate_original_chat_sources(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
    ) -> Result<(), TokenInputRejection> {
        template
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)?;
        tokenizer
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    fn validate_original_chat_render(
        runtime: &ModelRuntime<Self>,
        render: &OriginalRenderedChat,
    ) -> Result<(), TokenInputRejection> {
        render
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    fn compile_original_chat_template(
        runtime: &ModelRuntime<Self>,
        plan: ChatTemplatePlan<'_>,
    ) -> Result<OriginalChatTemplate, OriginalChatOperationError> {
        runtime.backend().facts.borrow_mut().chat_sources += 1;
        runtime
            .backend()
            .pool
            .compile_chat_template(plan)
            .map_err(Into::into)
    }
    fn compile_original_chat_template_file(
        runtime: &ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
        model_id: &str,
    ) -> Result<OriginalChatTemplate, OriginalChatOperationError> {
        runtime.backend().facts.borrow_mut().chat_sources += 1;
        runtime
            .backend()
            .pool
            .compile_chat_template_file(read, model_id)
            .map_err(Into::into)
    }
    fn render_original_chat(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        messages: ChatMessages<'_>,
        consumer: GenerationSequenceConsumerLayout,
    ) -> Result<OriginalRenderedChat, OriginalChatOperationError> {
        runtime.backend().facts.borrow_mut().chat_renders += 1;
        let rendered = runtime
            .backend()
            .pool
            .render_original_chat(template, tokenizer, messages, consumer)
            .map_err(OriginalChatOperationError::from)?;
        if let Some(cancel) = &runtime.backend().facts.borrow().cancel_after_render {
            cancel.cancel();
        }
        Ok(rendered)
    }
    fn render_original_chat_with_context(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'_>,
        consumer: GenerationSequenceConsumerLayout,
    ) -> Result<OriginalRenderedChat, OriginalChatOperationError> {
        runtime.backend().facts.borrow_mut().chat_renders += 1;
        let rendered = runtime
            .backend()
            .pool
            .render_original_chat_with_context(template, tokenizer, context, consumer)?;
        if let Some(cancel) = &runtime.backend().facts.borrow().cancel_after_render {
            cancel.cancel();
        }
        Ok(rendered)
    }
}

const CHAT_CONFIG: &str = include_str!("chat_fixtures/tokenizer_config.json");
// Actual small HF model with an explicit unknown token. The released full-file
// oracle uses its complete vocabulary; this neutral driver fixture preserves
// the existing h / Ġhi predictions while exercising full rendered prompt bytes.
const TOKENIZER: &str = include_str!("chat_fixtures/tokenizer.json");
fn config() -> TextGenerationConfig {
    super::config().with_inference_policy(eredu_core::TextInferencePolicy {
        managed_memory_capacity_bytes: Some(u64::MAX),
        ..Default::default()
    })
}
fn source_budget_bytes() -> u64 {
    OriginalTextSourceBudget::storage_bytes().unwrap()
}
fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![
            serde_json::json!({"role":"system","content":"hi"}),
            serde_json::json!({"role":"user","content":"hi 世界"}),
        ],
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    }
}
fn sources(
    runtime: &ModelRuntime<Backend>,
    cancellation: &GenerationCancellationToken,
) -> (OriginalTokenizer, OriginalChatTemplate) {
    let tokenizer = runtime
        .backend()
        .pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(TOKENIZER.as_bytes())
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(CHAT_CONFIG.as_bytes()).unwrap();
    let template = compile_original_chat_file(runtime, file, "smol", cancellation)
        .unwrap()
        .unwrap();
    (tokenizer, template)
}

#[test]
fn actual_private_chat_file_render_encode_and_shared_cursor_preserve_cached_outputs() {
    for manual in [false, true] {
        // Positive shared ceiling, with each J/H/S/E/I/R still admitted from its
        // actual plan. Exact-minus-one admission is checked separately.
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let cancellation = GenerationCancellationToken::new();
        let (tokenizer, template) = sources(&runtime, &cancellation);
        let cold = tokenizer.original_bytes() + template.original_bytes();
        assert_eq!(
            pool.used_bytes().unwrap(),
            cold,
            "config-file I retired after fresh J"
        );
        let request = request();
        let reference = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
        let mut ordinary = eredu_text::tokenizer::Tokenizer::from_tokenizer(reference.clone());
        let selected = eredu_text::tokenizer::load_model_chat_template_from_str(CHAT_CONFIG)
            .unwrap()
            .unwrap();
        let mut previous = None;
        let mut previous_held = 0;
        for generation in [false, true] {
            let expected = ordinary
                .apply_chat_template_json(
                    selected.clone(),
                    [request.messages.clone()],
                    None,
                    "smol",
                    generation,
                    None,
                )
                .unwrap()
                .pop()
                .unwrap();
            let expected_ids = reference.encode(expected.as_str(), false).unwrap();
            assert!(!expected_ids.get_ids().is_empty());
            let render = prepare_original_text_chat(
                &runtime,
                &template,
                &tokenizer,
                &request,
                config(),
                &cancellation,
            )
            .unwrap()
            .unwrap();
            assert_eq!(render.prompt(generation), expected);
            let h = render.original_bytes();
            assert_eq!(
                pool.used_bytes().unwrap(),
                cold + previous_held + h + source_budget_bytes()
            );
            let retained_render = render.clone_render_for_test();
            facts.borrow_mut().ids.clear();
            let mut session = start_original_text_chat(
                &mut runtime,
                &template,
                &tokenizer,
                render,
                generation,
                config(),
                &[],
                &[],
                true,
                &cancellation,
            )
            .unwrap()
            .unwrap();
            assert_eq!(facts.borrow().ids, expected_ids.get_ids());
            let mut visible = String::new();
            let mut event = |event: GenerationPlainTextEvent<'_>| {
                if let GenerationPlainTextEvent::TextDelta(text) = event {
                    visible.push_str(text);
                }
            };
            let output = if manual {
                while session.finish_reason().is_none() {
                    session = session.advance(&cancellation, &mut event).unwrap();
                }
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal chat session"))
            } else {
                session.run(&cancellation, &mut event).unwrap()
            };
            assert_eq!(output.text.as_str(), visible);
            assert_eq!(output.text.as_str(), "h hih");
            assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
            let r = facts.borrow().held;
            assert_eq!(pool.used_bytes().unwrap(), cold + previous_held + h + r);
            drop(retained_render);
            assert_eq!(
                pool.used_bytes().unwrap(),
                cold + previous_held + r,
                "last H alias retires after prompt encoding"
            );
            drop(previous.take());
            previous_held = r;
            previous = Some(output);
        }
        assert_eq!(facts.borrow().chat_sources, 1);
        // Each request performs the two shared default-profile recognition
        // renders before its real prompt render; encoding still occurs once.
        assert_eq!(facts.borrow().chat_renders, 6);
        assert_eq!(facts.borrow().encodes, 2);
        assert_eq!(facts.borrow().stops, 2);
        let output = previous.unwrap();
        let text = output.text.clone();
        let ids = output.token_ids.clone();
        drop((output, template, tokenizer, runtime));
        assert_eq!(pool.used_bytes().unwrap(), previous_held);
        drop(text);
        assert_eq!(pool.used_bytes().unwrap(), previous_held);
        drop(ids);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn private_chat_cancellation_and_late_startup_failures_keep_exact_owner_lifetimes() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    cancellation.cancel();
    assert!(
        compile_original_chat_file(
            &runtime,
            tempfile::tempfile().unwrap(),
            "smol",
            &cancellation
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(facts.borrow().chat_sources, 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(runtime);
    for phase in [
        "render",
        "stop",
        "encode",
        "eos",
        "short",
        "changed_capacity",
    ] {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        let cancellation = GenerationCancellationToken::new();
        let (tokenizer, template) = sources(&runtime, &cancellation);
        let cold = tokenizer.original_bytes() + template.original_bytes();
        if phase == "render" {
            facts.borrow_mut().cancel_after_render = Some(cancellation.clone());
        }
        let render = prepare_original_text_chat(
            &runtime,
            &template,
            &tokenizer,
            &request(),
            config(),
            &cancellation,
        )
        .unwrap();
        if phase == "render" {
            assert!(render.is_none());
            assert_eq!(pool.used_bytes().unwrap(), cold);
            assert_eq!(facts.borrow().stops, 0);
            assert_eq!(facts.borrow().encodes, 0);
            continue;
        }
        let render = render.unwrap();
        let h = render.original_bytes();
        if phase == "stop" {
            facts.borrow_mut().cancel_after_stops = Some(cancellation.clone());
        }
        if phase == "encode" {
            facts.borrow_mut().cancel_after_encode = Some(cancellation.clone());
        }
        if phase == "short" {
            facts.borrow_mut().short = true;
        }
        let startup_config = if phase == "changed_capacity" {
            config().with_inference_policy(eredu_core::TextInferencePolicy {
                managed_memory_capacity_bytes: Some(u64::MAX - 1),
                ..Default::default()
            })
        } else {
            config()
        };
        let eos: &[u32] = if phase == "eos" { &[u32::MAX] } else { &[] };
        let result = start_original_text_chat(
            &mut runtime,
            &template,
            &tokenizer,
            render,
            true,
            startup_config,
            eos,
            &["hi"],
            true,
            &cancellation,
        );
        match phase {
            "stop" | "encode" => {
                assert!(result.unwrap().is_none());
                assert_eq!(facts.borrow().stops, 1);
                assert_eq!(facts.borrow().encodes, usize::from(phase == "encode"));
                assert_eq!(pool.used_bytes().unwrap(), cold);
            }
            "eos" | "short" | "changed_capacity" => {
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!("expected chat startup rejection"),
                };
                assert_eq!(error.retained_render_bytes(), h);
                assert!(pool.used_bytes().unwrap() >= cold + h + source_budget_bytes());
                if phase == "changed_capacity" {
                    assert!(matches!(
                        error.input_rejection(),
                        Some(TokenInputRejection::IdentityMismatch)
                    ));
                }
                if matches!(phase, "eos" | "changed_capacity") {
                    assert_eq!(facts.borrow().stops, 0);
                    assert_eq!(facts.borrow().encodes, 0);
                    assert_eq!(pool.used_bytes().unwrap(), cold + h + source_budget_bytes());
                } else {
                    assert_eq!(facts.borrow().stops, 1);
                    assert_eq!(facts.borrow().encodes, 1);
                    assert!(!facts.borrow().order.contains(&"submit"));
                }
                drop(error);
                assert_eq!(pool.used_bytes().unwrap(), cold);
            }
            _ => unreachable!(),
        }
        drop((template, tokenizer, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn chat_caller_ceiling_precedes_render_and_survives_preparation_failure() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    let (tokenizer, template) = sources(&runtime, &cancellation);
    let cold = pool.used_bytes().unwrap();
    // A caller unable to retain existing sources never starts rendering. A
    // caller able to retain only the ceiling fails before profile preparation.
    for capacity in [1, cold + source_budget_bytes()] {
        let config = config().with_inference_policy(eredu_core::TextInferencePolicy {
            managed_memory_capacity_bytes: Some(capacity),
            ..Default::default()
        });
        let error = prepare_original_text_chat(
            &runtime,
            &template,
            &tokenizer,
            &request(),
            config,
            &cancellation,
        )
        .unwrap_err();
        assert_eq!(error.retained_render_bytes(), 0);
        assert_eq!(facts.borrow().chat_renders, 0);
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(facts.borrow().encodes, 0);
        assert!(facts.borrow().order.is_empty());
        if capacity != 1 {
            assert_eq!(pool.used_bytes().unwrap(), capacity);
            // Retaining the failure keeps the same domain ceiling active even
            // against a second source producer asking for a larger ceiling.
            assert!(
                Backend::prepare_original_text_source_budget(&runtime, &tokenizer, u64::MAX)
                    .is_err()
            );
        } else {
            assert_eq!(pool.used_bytes().unwrap(), cold);
        }
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), cold);
    }
    drop((template, tokenizer, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unsupported_chat_request_and_equal_content_foreign_sources_reject_before_work() {
    use crate::runtime::chat::ToolChoice;
    let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    let (tokenizer, template) = sources(&runtime, &cancellation);
    let cold = pool.used_bytes().unwrap();
    for case in [0, 1, 2, 6, 7] {
        let mut input = request();
        match case {
            0 => input.tool_choice = ToolChoice::Required,
            1 => {
                input.tool_choice = ToolChoice::None;
                input.tools.push(serde_json::json!({"name":"tool"}));
            }
            2 => input.enable_thinking = Some(true),
            6 => {
                input
                    .extra_template_kwargs
                    .insert("messages".to_owned(), serde_json::json!([null]));
            }
            // Rich content and tool history are supported borrowed inputs.
            // A message itself must still be an object before rendering.
            7 => input.messages[0] = serde_json::json!("not a message object"),
            _ => unreachable!(),
        }
        let error = prepare_original_text_chat(
            &runtime,
            &template,
            &tokenizer,
            &input,
            config(),
            &cancellation,
        )
        .unwrap_err();
        assert_eq!(error.retained_render_bytes(), 0);
        assert_eq!(pool.used_bytes().unwrap(), cold, "request case {case}");
        assert_eq!(facts.borrow().chat_renders, 0);
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(facts.borrow().encodes, 0);
        drop(error);
    }
    let (other_tokenizer, other_template) = sources(&runtime, &cancellation);
    assert!(!other_tokenizer.same_source(&tokenizer));
    assert!(!other_template.same_source(&template));
    let cold = pool.used_bytes().unwrap();
    for wrong_j in [false, true] {
        let render = prepare_original_text_chat(
            &runtime,
            &template,
            &tokenizer,
            &request(),
            config(),
            &cancellation,
        )
        .unwrap()
        .unwrap();
        let h = render.original_bytes();
        let error = match start_original_text_chat(
            &mut runtime,
            if wrong_j { &other_template } else { &template },
            if wrong_j {
                &tokenizer
            } else {
                &other_tokenizer
            },
            render,
            true,
            config(),
            &[],
            &[],
            true,
            &cancellation,
        ) {
            Err(error) => error,
            Ok(_) => panic!("equal-content sources cannot replace the actual H bindings"),
        };
        assert!(matches!(
            error.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        ));
        assert_eq!(error.retained_render_bytes(), h);
        assert_eq!(pool.used_bytes().unwrap(), cold + h + source_budget_bytes());
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(facts.borrow().encodes, 0);
        assert!(facts.borrow().order.is_empty());
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), cold);
    }
    drop((
        other_template,
        other_tokenizer,
        template,
        tokenizer,
        runtime,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
#[ignore = "requires exact pinned complete tokenizer and config paths; required serial chat oracle"]
fn released_complete_config_and_tokenizer_use_original_private_chat_request() {
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Input {
        origin: String,
        sha256: String,
    }
    #[derive(serde::Deserialize)]
    struct Inputs {
        tokenizer: Input,
        config: Input,
    }
    let manifest = std::env::var_os("EREDU_ORIGINAL_CHAT_ARTIFACTS").expect("pinned chat inputs");
    let inputs: Inputs = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let read = |input: &Input| {
        let bytes = std::fs::read(&input.origin).unwrap();
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(digest, input.sha256);
        bytes
    };
    let tokenizer_bytes = read(&inputs.tokenizer);
    let config_bytes = read(&inputs.config);
    let reference = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    let selected = eredu_text::tokenizer::load_model_chat_template_from_str(
        std::str::from_utf8(&config_bytes).unwrap(),
    )
    .unwrap()
    .unwrap();
    let mut ordinary = eredu_text::tokenizer::Tokenizer::from_tokenizer(reference.clone());
    let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
    // The complete byte/Unicode corpus exceeds the small default mock context.
    facts.borrow_mut().maximum_context = Some(1024);
    let source = crate::api::tokenizer::compile_original_text_tokenizer_file(
        &runtime,
        std::fs::File::open(&inputs.tokenizer.origin).unwrap(),
    )
    .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let template = compile_original_chat_file(
        &runtime,
        std::fs::File::open(&inputs.config.origin).unwrap(),
        "smol",
        &cancellation,
    )
    .unwrap()
    .unwrap();
    let cold = source.original_bytes() + template.original_bytes();
    assert_eq!(pool.used_bytes().unwrap(), cold);
    let a = source.token_id("a").unwrap();
    let b = source.token_id("b").unwrap();
    assert!(!source.is_special("a") && !source.is_special("b"));
    let domain = source.generation_domain().unwrap().allowed_mask().unwrap();
    facts.borrow_mut().prediction_ids = Some([a, b, a]);
    facts.borrow_mut().output_width = Some(domain.len());
    let scalar_bytes = (0..=255)
        .map(|n| char::from_u32(n).unwrap())
        .collect::<String>();
    let cases = [
        vec![],
        vec![serde_json::json!({"role":"system","content":""})],
        vec![serde_json::json!({"role":"user","content":" \n e\u{301} 世界🙂\0\t<|im_end|> "})],
        vec![
            serde_json::json!({"role":"system","content":"Rules"}),
            serde_json::json!({"role":"user","content":"Question"}),
            serde_json::json!({"role":"assistant","content":"Answer"}),
        ],
        vec![
            serde_json::json!({"role":"","content":""}),
            serde_json::json!({"role":"custom rôle","content":"nonempty"}),
        ],
        vec![serde_json::json!({"role":"user","content":scalar_bytes})],
    ];
    let mut rows = Vec::new();
    for (case, messages) in cases.into_iter().enumerate() {
        let request = ChatTemplateRequest {
            messages,
            ..ChatTemplateRequest::default()
        };
        let mut last_ids = None;
        for manual in [false, true] {
            for generation in [false, true] {
                for skip in [false, true] {
                    let expected = ordinary
                        .apply_chat_template_json(
                            selected.clone(),
                            [request.messages.clone()],
                            None,
                            "smol",
                            generation,
                            None,
                        )
                        .unwrap()
                        .pop()
                        .unwrap();
                    let expected_ids = reference.encode(expected.as_str(), false).unwrap();
                    let render = prepare_original_text_chat(
                        &runtime,
                        &template,
                        &source,
                        &request,
                        config(),
                        &cancellation,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(render.prompt(generation), expected);
                    assert_eq!(render.generation_suffix(), "<|im_start|>assistant\n");
                    facts.borrow_mut().ids.clear();
                    let h = render.original_bytes();
                    let prior_actions = facts.borrow().order.len();
                    let started = start_original_text_chat(
                        &mut runtime,
                        &template,
                        &source,
                        render,
                        generation,
                        config(),
                        &[],
                        &[],
                        skip,
                        &cancellation,
                    );
                    if expected_ids.get_ids().is_empty() {
                        assert!(expected.is_empty());
                        let error = match started {
                            Err(error) => error,
                            Ok(_) => panic!("empty encoded prompt must reject before I"),
                        };
                        assert!(matches!(
                            error.input_rejection(),
                            Some(TokenInputRejection::Empty)
                        ));
                        assert_eq!(error.retained_render_bytes(), h);
                        assert_eq!(pool.used_bytes().unwrap(), cold + h + source_budget_bytes());
                        assert_eq!(facts.borrow().order.len(), prior_actions);
                        drop(error);
                        assert_eq!(pool.used_bytes().unwrap(), cold);
                        last_ids = Some(Vec::new());
                        rows.push(serde_json::json!({"case":case,"manual":manual,"generation_prompt":generation,"skip_special":skip,"prompt_bytes":0,"input_ids":0,"outcome":"typed_empty_before_I"}));
                        continue;
                    }
                    let mut session = started.unwrap().unwrap();
                    assert_eq!(facts.borrow().ids, expected_ids.get_ids());
                    let mut visible = String::new();
                    let mut emit = |event: GenerationPlainTextEvent<'_>| {
                        if let GenerationPlainTextEvent::TextDelta(text) = event {
                            visible.push_str(text);
                        }
                    };
                    let output = if manual {
                        while session.finish_reason().is_none() {
                            session = session.advance(&cancellation, &mut emit).unwrap();
                        }
                        session
                            .into_output()
                            .unwrap_or_else(|_| panic!("terminal chat"))
                    } else {
                        session.run(&cancellation, &mut emit).unwrap()
                    };
                    assert_eq!(output.token_ids.as_ref(), &[a, b, a]);
                    assert_eq!(
                        output.text.as_str(),
                        reference.decode(&[a, b, a], skip).unwrap()
                    );
                    assert_eq!(output.text.as_str(), visible);
                    let held = facts.borrow().held;
                    assert_eq!(pool.used_bytes().unwrap(), cold + held);
                    let ids = output.token_ids.clone();
                    let text = output.text.clone();
                    drop(output);
                    drop(text);
                    assert_eq!(pool.used_bytes().unwrap(), cold + held);
                    drop(ids);
                    assert_eq!(pool.used_bytes().unwrap(), cold);
                    last_ids = Some(expected_ids.get_ids().to_vec());
                    rows.push(serde_json::json!({"case":case,"manual":manual,"generation_prompt":generation,"skip_special":skip,"prompt_bytes":expected.len(),"input_ids":expected_ids.get_ids().len(),"output_ids":3}));
                }
            }
        }
        assert!(last_ids.is_some());
    }
    assert_eq!(rows.len(), 48);
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("outcome").is_some())
            .count(),
        4
    );
    assert_eq!(facts.borrow().chat_renders, 48);
    assert_eq!(facts.borrow().encodes, 48);
    assert_eq!(facts.borrow().stops, 48);
    drop((template, source, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    println!(
        "ORIGINAL_CHAT_ORACLE_JSON={}",
        serde_json::json!({"scope":"complete pinned config/tokenizer original private J/H/C/S/E/I/R chat","config_sha256":inputs.config.sha256,"tokenizer_sha256":inputs.tokenizer.sha256,"private_requests":48,"completed_requests":44,"empty_rejections":4,"rows":rows})
    );
}

mod public_managed;

mod qwen_tools;

mod inkling;
mod released_template;

mod muse_nemotron;

mod deepseek;

mod lfm;

mod llama;

mod gemma;

mod mistral;

mod clock_templates;

mod nanbeige_kimi;

#[path = "original_chat/profile.rs"]
mod profile;
