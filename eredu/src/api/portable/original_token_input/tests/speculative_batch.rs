//! Actual original source/lane preparation stops at a neutral dispatch sentinel.
use super::*;
use crate::api::{
    ManagedPlainTextRequest, ManagedPlainTextSpeculativeBatchLane,
    ManagedPlainTextSpeculativeBatchRequest, PreparedChatGenerationSettings,
};
use std::{io::Write, num::NonZeroUsize};

#[derive(Clone)]
struct Speculative;
type Backend = super::Backend<Speculative>;

impl SpeculativeGenerationBackend for Backend {
    type Drafter = ();
    fn speculative_capability(_: &ModelRuntime<Self>) -> SpeculativeCapability {
        SpeculativeCapability::Unavailable
    }
    fn with_speculative_execution<
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    >(
        runtime: &mut ModelRuntime<Self>,
        mut request: SpeculativeGenerationBatchRequest<'_, Self, (), C>,
        _: V,
    ) -> Result<SpeculativeGenerationBatchOutput, WorkingMemoryError> {
        let lanes = request.take_lanes();
        assert_eq!(lanes.len(), 2);
        let sources: Vec<_> = lanes
            .iter()
            .map(|lane| {
                let source = lane
                    .semantic()
                    .prepared_source()
                    .unwrap()
                    .downcast_ref::<PreparedSemanticState>()
                    .unwrap()
                    .preparation();
                source
                    .validate(&runtime.backend().pool, &runtime.backend().execution)
                    .unwrap();
                source.validate_configuration(lane.configuration()).unwrap();
                source.validate_callback(lane.event_callback()).unwrap();
                source
            })
            .collect();
        assert!(
            !sources[0].same_preparation(sources[1]),
            "each lane retains its own source header"
        );
        assert!(
            sources[0]
                .validate_configuration(lanes[1].configuration())
                .is_err()
        );
        assert!(
            sources[0]
                .validate_callback(lanes[1].event_callback())
                .is_err()
        );
        assert_eq!(
            (lanes[0].config().max_tokens, lanes[1].config().max_tokens),
            (3, 4)
        );
        assert_eq!(
            (
                lanes[0].config().max_draft_tokens,
                lanes[1].config().max_draft_tokens
            ),
            (1, 2)
        );
        runtime.backend().facts.borrow_mut().speculative_seeds =
            lanes.iter().map(|lane| lane.generation().seed()).collect();
        // Refuse before any executor/model work, retaining the caller's original
        // failure path rather than fabricating successful generated output.
        Err(WorkingMemoryError::UnknownBound)
    }
}
// Reuse the exact admitted Digits+ByteLevel/added-token fixture exercised by
// original_plain, rather than reconstructing its finite regex profile here.
const CAPACITY: u64 = 128 << 20;
const JSON: &str = super::original_plain::JSON;
fn fixture() -> (LoadedModel<Backend>, Rc<RefCell<Facts>>, WorkingMemoryPool) {
    let (runtime, facts, pool) = bare_runtime_with_capacity_for::<Speculative>(CAPACITY);
    let model = LoadedModel::from_runtime(
        runtime,
        ChatTokenizer::from_bytes(JSON.as_bytes()).unwrap(),
        LoadedTextModelConfig {
            model_family: ModelKind::Llama,
            effective_model_type: "llama".into(),
            model_id: "original-batch-input".into(),
            chat_template: Some("{{ messages[0].content }}".into()),
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, facts, pool)
}
fn source(model: &LoadedModel<Backend>) -> crate::api::ManagedPlainTextSource {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(JSON.as_bytes()).unwrap();
    model.compile_managed_plain_text_source(file).unwrap()
}
fn lane(
    text: &'static str,
    seed: u64,
    maximum: usize,
    draft: usize,
    managed: bool,
) -> ManagedPlainTextSpeculativeBatchLane<'static, fn(SemanticEvent)> {
    let settings = PreparedChatGenerationSettings {
        seed,
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(maximum),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: managed.then_some(CAPACITY),
            ..Default::default()
        },
        ..Default::default()
    };
    ManagedPlainTextSpeculativeBatchLane {
        text: ManagedPlainTextRequest::new(text, settings),
        max_draft_tokens: NonZeroUsize::new(draft).unwrap(),
        cancellation: Default::default(),
        on_event: |_| {},
    }
}
#[test]
fn original_speculative_batch_qualifies_distinct_sources_and_rejects_late_policy_before_prompts() {
    for invalid in [false, true] {
        let (mut model, facts, pool) = fixture();
        let source = source(&model);
        let cold = pool.used_bytes().unwrap();
        let error = model
            .generate_managed_plain_text_speculative_batch(
                &source,
                ManagedPlainTextSpeculativeBatchRequest {
                    drafting: SpeculativeDraft::Embedded,
                    lanes: vec![
                        lane("hi", 17, 3, 1, true),
                        lane("hi hi<S>", 29, 4, 2, !invalid),
                    ],
                    scheduler: Default::default(),
                },
            )
            .err()
            .expect("neutral dispatch sentinel or late policy rejection");
        if invalid {
            assert_eq!(
                error.input_rejection(),
                Some(TokenInputRejection::Unsupported)
            );
            assert_eq!(facts.borrow().speculative_sources, 0);
            assert!(facts.borrow().speculative_prompts.is_empty());
            assert!(facts.borrow().speculative_seeds.is_empty());
            assert_eq!(pool.used_bytes().unwrap(), cold);
        } else {
            assert!(error.backend_failure().is_some());
            assert_eq!(facts.borrow().speculative_sources, 2);
            let expected: Vec<_> = ["hi", "hi hi<S>"]
                .into_iter()
                .map(|text| model.encode(text, true).unwrap())
                .collect();
            assert_eq!(expected, [vec![2], vec![2, 8, 4]]);
            assert_eq!(facts.borrow().speculative_prompts, expected);
            assert_eq!(facts.borrow().speculative_seeds, [17, 29]);
            assert!(
                pool.used_bytes().unwrap() > cold,
                "escaped batch error retains its paid source prefix"
            );
        }
        drop((source, model));
        if !invalid {
            assert!(pool.used_bytes().unwrap() > 0);
        }
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn prepared_chat_speculative_batch_uses_the_same_paid_lanes_and_host_vote() {
    use crate::api::{
        PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest,
    };
    use crate::runtime::chat::ChatTemplateRequest;
    for invalid in [false, true] {
        let (mut model, facts, pool) = fixture();
        let tokenizer = source(&model);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(br#"{"chat_template":"{{ messages[0].content }}"}"#)
            .unwrap();
        let cancellation = GenerationCancellationToken::new();
        let source = model
            .compile_managed_chat_source(&tokenizer, file, false, &cancellation)
            .unwrap()
            .unwrap();
        let chats = ["hi", "hi hi<S>"].map(|text| {
            model
                .prepare_chat(
                    &source,
                    &ChatTemplateRequest {
                        messages: vec![serde_json::json!({"role":"user", "content":text})],
                        ..Default::default()
                    },
                    CAPACITY,
                    &cancellation,
                )
                .unwrap()
                .unwrap()
        });
        let cold = pool.used_bytes().unwrap();
        let make = |index: usize, seed, maximum, draft, funded| {
            let text = lane("", seed, maximum, draft, funded);
            PreparedChatSpeculativeBatchLane {
                chat: &chats[index],
                input: crate::api::PreparedChatPrompt::Rendered,
                output_mode: crate::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                settings: text.text.settings,
                max_draft_tokens: text.max_draft_tokens,
                caller_stop_sequences: &[],
                cancellation: text.cancellation,
                on_event: text.on_event,
            }
        };
        let error = model
            .generate_prepared_chat_speculative_batch(
                PreparedChatSpeculativeBatchRequest {
                    drafting: SpeculativeDraft::Embedded,
                    lanes: vec![make(0, 17, 3, 1, true), make(1, 29, 4, 2, !invalid)],
                    scheduler: Default::default(),
                },
            )
            .err()
            .expect("neutral dispatch sentinel or failed policy");
        if invalid {
            assert_eq!(
                error.input_rejection(),
                Some(TokenInputRejection::Unsupported)
            );
            assert_eq!(facts.borrow().speculative_sources, 0);
            assert!(facts.borrow().speculative_prompts.is_empty());
            assert!(facts.borrow().speculative_seeds.is_empty());
            assert_eq!(pool.used_bytes().unwrap(), cold);
        } else {
            assert!(error.backend_failure().is_some());
            assert_eq!(facts.borrow().speculative_sources, 2);
            assert_eq!(facts.borrow().speculative_prompts, [vec![2], vec![2, 8, 4]]);
            assert_eq!(facts.borrow().speculative_seeds, [17, 29]);
            assert!(pool.used_bytes().unwrap() > cold);
        }
        drop((chats, source, tokenizer, model));
        if !invalid {
            assert!(pool.used_bytes().unwrap() > 0);
        }
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
