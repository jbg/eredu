use super::*;
use eredu::api::{
    PreparedChatGenerationSettings, PreparedChatSpeculativeBatchLane,
    PreparedChatSpeculativeBatchRequest, PreparedChatSpeculativeRequest, TextSamplingStrategy,
};
use eredu::runtime::chat::{ChatTemplateRequest, PreparedChat};
use eredu_core::generation::{CheckpointGenerationConfig, GenerationError};
use eredu_core::{
    SpeculativeCapability, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest,
    SpeculativeGenerationVisitor, SpeculativeTokenFilterController,
};

impl SpeculativeGenerationBackend for MockBackend {
    type Drafter = ();

    fn speculative_capability(_: &ModelRuntime<Self>) -> SpeculativeCapability {
        SpeculativeCapability::Unavailable
    }

    fn with_speculative_execution<C, V>(
        runtime: &mut ModelRuntime<Self>,
        mut request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
        _: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Self::Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let mut calls = runtime.backend().calls.borrow_mut();
        calls.speculative += 1;
        for mut lane in request.take_lanes() {
            calls.speculative_prompts.push(lane.prompt().to_vec());
            assert_eq!(
                lane.config().temperature,
                lane.generation().sampling().temperature
            );
            assert_eq!(lane.config().max_tokens, 2);
            calls.configs.push(lane.take_generation());
            calls
                .filters
                .push(lane.take_constraint().filter_at(&[]).unwrap());
        }
        Err(MockError::Injected)
    }
}

fn chat(model: &mut original_sources::Fixture<MockBackend>) -> PreparedChat {
    {
        let request = ChatTemplateRequest {
            messages: vec![serde_json::json!({"role": "user", "content": "a"})],
            add_generation_prompt: true,
            ..Default::default()
        };
        let cancel = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancel)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancel)
            .map(|chat| chat.expect("active preparation"))
    }
    .unwrap()
}

fn mirostat() -> PreparedChatGenerationSettings {
    original_sources::settings(PreparedChatGenerationSettings {
        strategy: TextSamplingStrategy::MirostatV2 { tau: 4.0, eta: 0.2 },
        overrides: GenerationConfigOverrides {
            temperature: Some(0.8),
            max_new_tokens: Some(2),
            ..Default::default()
        },
        seed: 73,
        ..Default::default()
    })
}

#[test]
fn prepared_sampling_preserves_strategy_resolved_controls_and_vocabulary_masks() {
    for strategy in [TextSamplingStrategy::Standard, mirostat().strategy] {
        let backend = MockBackend {
            logits: vec![0.0, 100.0, 10.0, 200.0, 300.0, 1.0, 400.0],
            ..Default::default()
        };
        let calls = backend.calls.clone();
        let mut model = sparse_vocabulary_model_with_backend(
            backend,
            Some(CheckpointGenerationConfig {
                do_sample: Some(true),
                temperature: Some(0.6),
                top_k: Some(17),
                top_p: Some(0.85),
                min_p: Some(0.05),
                repetition_penalty: Some(1.2),
                repeat_last_n: Some(32),
                frequency_penalty: Some(0.3),
                presence_penalty: Some(0.4),
                max_new_tokens: Some(2),
            }),
        );
        let chat = chat(&mut model);
        let original_source = model.tokenizer_source().clone();
        let settings = PreparedChatGenerationSettings {
            strategy,
            overrides: GenerationConfigOverrides {
                presence_penalty: Some(0.7),
                ..Default::default()
            },
            seed: 73,
            ..Default::default()
        };
        let expected = model.resolve_generation_config(settings.overrides).unwrap();
        // The observed path shares the same strategy resolution and constraints.
        for observed in [false, true] {
            let cancel = eredu_core::GenerationCancellationToken::new();
            let request =
                eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings));
            let mut delivered = Vec::new();
            let mut observer =
                |token, capture: Option<eredu_core::capture::SharedCapturedStep>, _| {
                    assert!(capture.is_none());
                    delivered.push(token);
                };
            let run = model
                .start_prepared_chat(request, &cancel)
                .unwrap()
                .unwrap();
            let output = if observed {
                run.with_capture_observer(&mut observer)
                    .run(&cancel, &mut |_| {})
                    .unwrap()
            } else {
                run.run(&cancel, &mut |_| {}).unwrap()
            };
            if observed {
                assert_eq!(delivered, vec![Some(2), Some(2)]);
            }
            assert_eq!(output.token_ids.as_ref(), [2, 2]);
            assert_eq!(output.finish_reason, eredu_core::FinishReason::MaxTokens);
        }
        let calls = calls.borrow();
        assert_eq!(calls.configs.len(), 2);
        for config in &calls.configs {
            assert_eq!(config.strategy(), strategy);
            assert_eq!(config.seed(), 73);
            assert_eq!(config.sampling(), expected);
        }
        assert_eq!(calls.filters.len(), 4);
        for filter in &calls.filters {
            assert_eq!(
                filter.allowed_mask_for(7).unwrap().unwrap().as_ref(),
                [true, false, true, false, false, true, false]
            );
        }
    }
    assert_eq!(
        PreparedChatGenerationSettings::default().strategy,
        TextSamplingStrategy::Standard
    );
}

#[test]
fn invalid_mirostat_is_rejected_before_backend_work() {
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for tau_is_invalid in [true, false] {
            let backend = MockBackend::default();
            let calls = backend.calls.clone();
            let mut model = sparse_vocabulary_model_with_backend(backend, None);
            let chat = chat(&mut model);
            let original_source = model.tokenizer_source().clone();
            let settings = PreparedChatGenerationSettings {
                strategy: if tau_is_invalid {
                    TextSamplingStrategy::MirostatV2 {
                        tau: invalid,
                        eta: 0.2,
                    }
                } else {
                    TextSamplingStrategy::MirostatV2 {
                        tau: 4.0,
                        eta: invalid,
                    }
                },
                ..mirostat()
            };
            assert_invalid_speculative_settings(None, settings, |error| {
                assert!(matches!(
                    (error, tau_is_invalid),
                    (GenerationError::InvalidMirostatTau(_), true)
                        | (GenerationError::InvalidMirostatEta(_), false)
                ));
            });
            let error = {
                let cancel = Default::default();
                let mut request = eredu::api::PreparedChatRequest::new(
                    &chat,
                    original_sources::settings(settings),
                );
                request.stop_sequences = &[];
                model
                    .start_prepared_chat(request, &cancel)
                    .and_then(|session| {
                        session.expect("active request").run(
                            &cancel,
                            &mut (|_| panic!("invalid settings must not emit events")),
                        )
                    })
            }
            .unwrap_err();
            assert!(matches!(
                (
                    frozen_recipe::cause::<GenerationError>(&error),
                    tau_is_invalid
                ),
                (Some(GenerationError::InvalidMirostatTau(_)), true)
                    | (Some(GenerationError::InvalidMirostatEta(_)), false)
            ));
            let calls = calls.borrow();
            assert!(calls.configs.is_empty());
            assert_eq!(calls.prompts, 0);
            assert!(calls.filters.is_empty());
        }
    }
}

#[test]
fn mirostat_requires_positive_effective_temperature_after_resolution() {
    for (checkpoint, overrides) in [
        (
            None,
            GenerationConfigOverrides {
                temperature: Some(0.0),
                ..Default::default()
            },
        ),
        (
            None,
            GenerationConfigOverrides {
                do_sample: Some(false),
                temperature: Some(0.8),
                ..Default::default()
            },
        ),
        (
            Some(CheckpointGenerationConfig {
                do_sample: Some(false),
                temperature: Some(0.8),
                ..Default::default()
            }),
            GenerationConfigOverrides::default(),
        ),
        (
            Some(CheckpointGenerationConfig {
                temperature: Some(0.0),
                ..Default::default()
            }),
            GenerationConfigOverrides::default(),
        ),
    ] {
        let backend = MockBackend::default();
        let calls = backend.calls.clone();
        assert_invalid_speculative_settings(
            checkpoint.clone(),
            PreparedChatGenerationSettings {
                overrides,
                ..mirostat()
            },
            |error| {
                assert!(matches!(
                    error,
                    GenerationError::InvalidMirostatTemperature(0.0)
                ))
            },
        );
        let mut model = sparse_vocabulary_model_with_backend(backend, checkpoint);
        let chat = chat(&mut model);
        let original_source = model.tokenizer_source().clone();
        let error = {
            let cancel = Default::default();
            let mut request = eredu::api::PreparedChatRequest::new(
                &chat,
                original_sources::settings(PreparedChatGenerationSettings {
                    overrides,
                    ..mirostat()
                }),
            );
            request.stop_sequences = &[];
            model
                .start_prepared_chat(request, &cancel)
                .and_then(|session| {
                    session.expect("active request").run(
                        &cancel,
                        &mut (|_| panic!("invalid settings must not emit events")),
                    )
                })
        }
        .unwrap_err();
        assert!(matches!(
            frozen_recipe::cause::<GenerationError>(&error),
            Some(GenerationError::InvalidMirostatTemperature(0.0))
        ));
        let calls = calls.borrow();
        assert!(calls.configs.is_empty());
        assert_eq!(calls.prompts, 0);
    }
}

#[test]
fn speculative_mirostat_preserves_single_and_mixed_batch_settings() {
    for lookahead in [false, true] {
        for embedded in [false, true] {
            let backend = MockBackend::default();
            let calls = backend.calls.clone();
            let mut model = sparse_vocabulary_model_with_backend(
                backend,
                Some(CheckpointGenerationConfig {
                    repetition_penalty: Some(1.2),
                    repeat_last_n: Some(32),
                    frequency_penalty: Some(0.3),
                    presence_penalty: Some(0.4),
                    ..Default::default()
                }),
            );
            let chat = chat(&mut model);
            let original_source = model.tokenizer_source().clone();
            let expected = model
                .resolve_generation_config(mirostat().overrides)
                .unwrap();
            let mut drafter = ();
            let error = model
                .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                    output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                    skip_special_tokens: true,
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::Rendered,
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    settings: mirostat(),
                    options: eredu::api::PreparedChatSpeculativeGenerationOptions {
                        scheduler: eredu_core::SpeculativeSchedulerOptions::default()
                            .with_lookahead(lookahead),
                        ..Default::default()
                    },
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| panic!("mock backend must not emit events"),
                })
                .expect_err("mock backend returns its sentinel error");
            assert!(error.backend_failure().is_some(), "{error:?}");
            let error = model
                .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    lanes: [TextSamplingStrategy::Standard, mirostat().strategy]
                        .into_iter()
                        .map(|strategy| PreparedChatSpeculativeBatchLane {
                            output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                            skip_special_tokens: true,
                            chat: &chat,
                            input: eredu::api::PreparedChatPrompt::Rendered,
                            settings: PreparedChatGenerationSettings {
                                strategy,
                                ..mirostat()
                            },
                            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                            caller_stop_sequences: &[],
                            cancellation: Default::default(),
                            on_event: Box::new(|_| panic!("mock backend must not emit events")),
                        })
                        .collect(),
                    scheduler: eredu_core::SpeculativeSchedulerOptions::default()
                        .with_lookahead(lookahead),
                })
                .err()
                .expect("mock backend returns its sentinel error");
            assert!(error.backend_failure().is_some(), "{error:?}");
            let calls = calls.borrow();
            assert_eq!(calls.prompts, 3);
            assert_eq!(calls.speculative, 2);
            assert_eq!(calls.configs.len(), 3);
            for (config, strategy) in calls.configs.iter().zip([
                mirostat().strategy,
                TextSamplingStrategy::Standard,
                mirostat().strategy,
            ]) {
                assert_eq!(config.strategy(), strategy);
                assert_eq!(config.seed(), 73);
                assert_eq!(config.sampling(), expected);
            }
            assert_eq!(calls.filters.len(), 3);
            for filter in &calls.filters {
                assert_eq!(
                    filter.allowed_mask_for(7).unwrap().unwrap().as_ref(),
                    [true, false, true, false, false, true, false]
                );
            }
        }
    }
}

fn assert_invalid_speculative_settings(
    checkpoint: Option<CheckpointGenerationConfig>,
    settings: PreparedChatGenerationSettings,
    check_error: impl Fn(GenerationError),
) {
    let backend = MockBackend::default();
    let calls = backend.calls.clone();
    let mut model = sparse_vocabulary_model_with_backend(backend, checkpoint);
    let chat = chat(&mut model);
    let original_source = model.tokenizer_source().clone();
    for lookahead in [false, true] {
        for embedded in [false, true] {
            let mut drafter = ();
            let error = model
                .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                    output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                    skip_special_tokens: true,
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::Rendered,
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    settings: original_sources::settings(settings),
                    options: eredu::api::PreparedChatSpeculativeGenerationOptions {
                        scheduler: eredu_core::SpeculativeSchedulerOptions::default()
                            .with_lookahead(lookahead),
                        ..Default::default()
                    },
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| panic!("invalid settings must not emit events"),
                })
                .expect_err("invalid settings must fail");
            let error =
                frozen_recipe::cause::<GenerationError>(&error).expect("typed generation error");
            check_error(error.clone());
            let error = model
                .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    lanes: [
                        original_sources::settings(PreparedChatGenerationSettings::default()),
                        original_sources::settings(settings),
                    ]
                    .into_iter()
                    .map(|settings| PreparedChatSpeculativeBatchLane {
                        output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                        skip_special_tokens: true,
                        chat: &chat,
                        input: eredu::api::PreparedChatPrompt::Rendered,
                        settings,
                        max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                        caller_stop_sequences: &[],
                        cancellation: Default::default(),
                        on_event: Box::new(|_| panic!("invalid settings must not emit events")),
                    })
                    .collect(),
                    scheduler: eredu_core::SpeculativeSchedulerOptions::default()
                        .with_lookahead(lookahead),
                })
                .err()
                .expect("invalid batch settings must fail");
            let error =
                frozen_recipe::cause::<GenerationError>(&error).expect("typed generation error");
            check_error(error.clone());
        }
    }
    let calls = calls.borrow();
    assert_eq!(calls.prompts, 0);
    assert_eq!(calls.speculative, 0);
    assert!(calls.configs.is_empty());
    assert!(calls.filters.is_empty());
}

#[test]
fn speculative_single_batch_and_prompt_failures_use_provider_hook() {
    for fail_prompt in [false, true] {
        for batch in [false, true] {
            let backend = MockBackend::default();
            let calls = backend.calls.clone();
            let mut model = sparse_vocabulary_model_with_backend(backend, None);
            let chat = chat(&mut model);
            let original_source = model.tokenizer_source().clone();
            calls.borrow_mut().reject_prompt = fail_prompt;
            let error = if batch {
                model
                    .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
                        drafting: SpeculativeDraft::Embedded,
                        lanes: vec![PreparedChatSpeculativeBatchLane {
                            output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                            skip_special_tokens: true,
                            chat: &chat,
                            input: eredu::api::PreparedChatPrompt::Rendered,
                            settings: mirostat(),
                            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                            caller_stop_sequences: &[],
                            cancellation: Default::default(),
                            on_event: Box::new(|_| panic!("rejected backend must not emit")),
                        }],
                        scheduler: Default::default(),
                    })
                    .err()
                    .unwrap()
            } else {
                model
                    .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                        output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                        skip_special_tokens: true,
                        chat: &chat,
                        input: eredu::api::PreparedChatPrompt::Rendered,
                        drafting: SpeculativeDraft::Embedded,
                        settings: mirostat(),
                        options: Default::default(),
                        caller_stop_sequences: &[],
                        cancellation: Default::default(),
                        on_event: |_| panic!("rejected backend must not emit"),
                    })
                    .unwrap_err()
            };
            let error = error
                .backend_failure()
                .expect("backend error retains typed branch");
            assert_eq!(error.kind(), eredu_core::BackendFailureKind::Busy);
            assert_eq!(error.operation(), "portable-provider-hook");
            assert!(std::error::Error::source(&error).unwrap().is::<MockError>());
            assert_eq!(calls.borrow().speculative, usize::from(!fail_prompt));
            assert_eq!(calls.borrow().prompts, usize::from(!fail_prompt));
        }
    }
}

#[test]
fn speculative_exact_ids_preserve_the_prefix_and_refuse_tokenizer_holes_before_preparation() {
    for prefix in [&[5, 2, 0, 2][..], &[2, 1][..]] {
        let backend = MockBackend::default();
        let calls = backend.calls.clone();
        let mut model = sparse_vocabulary_model_with_backend(backend, None);
        let chat = chat(&mut model);
        let error = model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(prefix),
                drafting: SpeculativeDraft::Embedded,
                settings: mirostat(),
                output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                skip_special_tokens: true,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| panic!("the inspection backend does not publish events"),
            })
            .unwrap_err();
        let calls = calls.borrow();
        if prefix.contains(&1) {
            assert_eq!(
                error.input_rejection(),
                Some(eredu_core::TokenInputRejection::InvalidToken)
            );
            assert_eq!(calls.prompts, 0);
            assert!(calls.speculative_prompts.is_empty());
        } else {
            assert!(error.backend_failure().is_some());
            assert_eq!(calls.prompts, 1);
            assert_eq!(calls.speculative_prompts, [prefix]);
        }
    }
}
