use super::*;
use eredu::api::{
    PreparedChatError, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, TextModelError, TraceLimits,
};
use eredu::runtime::chat::{ChatTemplateRequest, PreparedChat, SemanticSupport, ToolChoice};
use eredu_core::{
    capture::CapturePlan, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool,
};
use serde_json::json;
use std::{cell::RefCell, ops::ControlFlow, rc::Rc};
use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

fn fixture(
    pool: &WorkingMemoryPool,
    template: bool,
) -> (
    LoadedModel<MockBackend>,
    Rc<RefCell<BackendCalls>>,
    Vec<u32>,
) {
    let vocabulary: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, character)| (character.to_string(), id as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut answer = tokenizer.encode("okay", false).unwrap().get_ids().to_vec();
    answer.push(eos);
    let backend = MockBackend::default();
    let calls = backend.calls.clone();
    calls.borrow_mut().pool = Some(pool.clone());
    calls.borrow_mut().scripted_tokens = answer.iter().copied().collect();
    let model = LoadedModel::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "frozen-recipe-host-custody".into(),
            chat_template: template.then(|| {
                include_str!("../fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja").into()
            }),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, calls, answer)
}

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "Say okay."})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(16),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn limits() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 65_536,
        total_bytes: 1_048_576,
    }
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(cause) = error.downcast_ref::<T>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

fn registered_recipe(pool: &WorkingMemoryPool, calls: &Rc<RefCell<BackendCalls>>) -> u64 {
    let calls = calls.borrow();
    assert_eq!(calls.shared_bytes_attempts, 1);
    assert_eq!(calls.shared_bytes_factories, 1);
    assert!(calls.shared_bytes_payload > 0);
    assert!(calls.shared_bytes_capacity >= calls.shared_bytes_payload as u64);
    assert_eq!(pool.used_bytes().unwrap(), calls.shared_bytes_capacity);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    calls.shared_bytes_capacity
}

#[test]
fn host_rejection_precedes_missing_template_or_invalid_grammar_and_keeps_typed_cause() {
    for template in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut model, calls, _) = fixture(&pool, template);
        calls.borrow_mut().reject_host = true;
        let mut invalid = request();
        invalid.tools = vec![json!({"type":"function", "function": {
            "name":"broken", "parameters":{"type":17}
        }})];
        let error = model.prepare_chat(invalid.clone()).unwrap_err();
        assert!(matches!(error, TextModelError::Backend(_)));
        assert!(cause::<MockError>(&error).is_some());
        assert_eq!(calls.borrow().host_attempts, 1);
        assert_eq!(calls.borrow().shared_bytes_attempts, 0);
        assert_eq!(calls.borrow().prompts, 0);
        assert!(calls.borrow().configs.is_empty());
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), 0);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        calls.borrow_mut().reject_host = false;
        let error = model.prepare_chat(invalid).unwrap_err();
        if template {
            assert!(matches!(error, TextModelError::ToolConstraint(_)));
        } else {
            assert!(matches!(error, TextModelError::MissingChatTemplate));
        }
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn recipe_registration_rejection_is_backend_failure_and_retires_opaque_compiler() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, calls, _) = fixture(&pool, true);
    calls.borrow_mut().reject_shared_bytes = true;
    let error = model.prepare_chat(request()).unwrap_err();
    assert!(matches!(error, TextModelError::Backend(_)));
    assert!(cause::<MockError>(&error).is_some());
    assert_eq!(calls.borrow().host_attempts, 1);
    assert_eq!(calls.borrow().shared_bytes_attempts, 1);
    assert_eq!(calls.borrow().shared_bytes_factories, 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    calls.borrow_mut().reject_shared_bytes = false;
    let prepared = model.prepare_chat(request()).unwrap();
    assert!(matches!(
        prepared.semantic_support(),
        SemanticSupport::Supported
    ));
    let bytes = calls.borrow().shared_bytes_capacity;
    assert!(bytes > 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((model, prepared));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn recognized_prepared_chat_clones_keep_exact_source_charge_after_model_drop() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, calls, _) = fixture(&pool, true);
    let prepared = model.prepare_chat(request()).unwrap();
    assert!(matches!(
        prepared.semantic_support(),
        SemanticSupport::Supported
    ));
    let bytes = registered_recipe(&pool, &calls);
    let alias = prepared.clone();
    let last = alias.clone();
    drop((model, prepared));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert!(!last.rendered_prompt().is_empty());
    drop(last);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

// This zero-byte finite reservation tests exclusion only. The mock backend does
// not supply or claim a finite native inference/parser workspace quote.
fn exclusion_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "portable host exclusion fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

fn controlled_output(
    model: &mut LoadedModel<MockBackend>,
    chat: &PreparedChat,
    prompt: u32,
    text: bool,
) -> Vec<u32> {
    let prepared = model
        .prepare_observed_token_ids(
            chat,
            vec![prompt],
            settings(),
            CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut session = if text {
        model
            .start_controlled_text(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
    } else {
        model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
    };
    while session.finish_reason().is_none() {
        session.step(|_| ControlFlow::Continue(())).unwrap();
    }
    session.token_ids().to_vec()
}

#[test]
fn zero_finite_reservation_blocks_prepare_and_both_semantic_entries_before_work() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, calls, answer) = fixture(&pool, true);
    let chat = model.prepare_chat(request()).unwrap();
    let bytes = registered_recipe(&pool, &calls);
    let controlled = model
        .prepare_observed_token_ids(
            &chat,
            vec![answer[0]],
            settings(),
            CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let reservation = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &exclusion_admission(),
            bytes,
        )
        .unwrap();
    let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
    let attempts = calls.borrow().shared_bytes_attempts;
    let factories = calls.borrow().shared_bytes_factories;
    let error = model.prepare_chat(request()).unwrap_err();
    assert!(matches!(error, TextModelError::Backend(_)));
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    let error = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![answer[0]]),
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| panic!("rejected semantic setup emitted an event"),
        })
        .unwrap_err();
    assert!(matches!(error, PreparedChatError::Backend(_)));
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    let error = model
        .start_controlled_chat(
            controlled,
            &[],
            Default::default(),
            |_| -> ControlFlow<()> {
                panic!("rejected semantic setup emitted a controlled record")
            },
        )
        .err()
        .expect("controlled semantic setup must reject");
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(calls.borrow().shared_bytes_attempts, attempts);
    assert_eq!(calls.borrow().shared_bytes_factories, factories);
    assert_eq!(calls.borrow().prompts, 0);
    assert!(calls.borrow().configs.is_empty());
    assert!(calls.borrow().filters.is_empty());
    assert_eq!(
        calls
            .borrow()
            .scripted_tokens
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        answer
    );
    assert_eq!(
        (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
        before
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(reservation);
    let output = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![answer[0]]),
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    assert_eq!(output.token_ids, answer);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    calls.borrow_mut().scripted_tokens = answer.iter().copied().collect();
    assert_eq!(
        controlled_output(&mut model, &chat, answer[0], false),
        answer
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop((chat, model));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn explicit_text_ordinary_and_controlled_skip_host_reconstruction_with_recipe_aliases_live() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, calls, answer) = fixture(&pool, true);
    let chat = model.prepare_chat(request()).unwrap();
    let alias = chat.clone();
    let bytes = registered_recipe(&pool, &calls);
    let host_attempts = calls.borrow().host_attempts;
    let byte_attempts = calls.borrow().shared_bytes_attempts;
    calls.borrow_mut().reject_host = true;
    calls.borrow_mut().reject_shared_bytes = true;
    let output = model
        .generate_prepared_text(PreparedChatGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![answer[0]]),
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    assert_eq!(output.token_ids, answer);
    calls.borrow_mut().scripted_tokens = answer.iter().copied().collect();
    assert_eq!(
        controlled_output(&mut model, &chat, answer[0], true),
        answer
    );
    assert_eq!(calls.borrow().host_attempts, host_attempts);
    assert_eq!(calls.borrow().shared_bytes_attempts, byte_attempts);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((model, chat));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "frozen_recipe/error_conversion.rs"]
mod error_conversion;
