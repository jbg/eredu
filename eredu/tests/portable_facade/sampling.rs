use super::*;
use eredu::api::{
    PreparedChatError, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest,
    PreparedChatSpeculativeError, PreparedChatSpeculativeGenerationRequest, TextSamplingStrategy,
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
        Err(MockError)
    }
}

fn chat(model: &mut LoadedModel<MockBackend>) -> PreparedChat {
    model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role": "user", "content": "a"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap()
}

fn mirostat() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        strategy: TextSamplingStrategy::MirostatV2 { tau: 4.0, eta: 0.2 },
        overrides: GenerationConfigOverrides {
            temperature: Some(0.8),
            max_new_tokens: Some(2),
            ..Default::default()
        },
        seed: 73,
    }
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
        let settings = PreparedChatGenerationSettings {
            strategy,
            overrides: GenerationConfigOverrides {
                presence_penalty: Some(0.7),
                ..Default::default()
            },
            seed: 73,
        };
        let expected = model.resolve_generation_config(settings.overrides).unwrap();
        // The observed path shares the same strategy resolution and constraints.
        for observed in [false, true] {
            let output = if observed {
                let prepared = model
                    .prepare_observed_chat(
                        &chat,
                        settings,
                        eredu_core::capture::CapturePlan::none(),
                        eredu::api::TraceLimits {
                            per_record_bytes: 65536,
                            total_bytes: 1024 * 1024,
                        },
                    )
                    .unwrap();
                model
                    .generate_observed_chat(prepared, &[], Default::default(), |_| {
                        std::ops::ControlFlow::Continue(())
                    })
                    .unwrap()
            } else {
                model
                    .generate_prepared_chat(PreparedChatGenerationRequest {
                        input: PreparedChatInput::rendered_prompt(&chat),
                        settings,
                        caller_stop_sequences: &[],
                        cancellation: Default::default(),
                        on_event: |_| {},
                    })
                    .unwrap()
            };
            assert_eq!(output.token_ids, [2, 2]);
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
            let error = model
                .generate_prepared_chat(PreparedChatGenerationRequest {
                    input: PreparedChatInput::rendered_prompt(&chat),
                    settings,
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| panic!("invalid settings must not emit events"),
                })
                .unwrap_err();
            assert!(matches!(
                (error, tau_is_invalid),
                (
                    PreparedChatError::Generation(GenerationError::InvalidMirostatTau(_)),
                    true
                ) | (
                    PreparedChatError::Generation(GenerationError::InvalidMirostatEta(_)),
                    false
                )
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
        let error = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings: PreparedChatGenerationSettings {
                    overrides,
                    ..mirostat()
                },
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| panic!("invalid settings must not emit events"),
            })
            .unwrap_err();
        assert!(matches!(
            error,
            PreparedChatError::Generation(GenerationError::InvalidMirostatTemperature(0.0))
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
            let expected = model
                .resolve_generation_config(mirostat().overrides)
                .unwrap();
            let mut drafter = ();
            let error = model
                .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::rendered_prompt(&chat),
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
                .err()
                .expect("mock backend returns its sentinel error");
            assert!(matches!(error, PreparedChatSpeculativeError::Backend(_)));
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
                            input: PreparedChatInput::rendered_prompt(&chat),
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
            assert!(matches!(error, PreparedChatSpeculativeError::Backend(_)));
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
    for lookahead in [false, true] {
        for embedded in [false, true] {
            let mut drafter = ();
            let error = model
                .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::rendered_prompt(&chat),
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    settings,
                    options: eredu::api::PreparedChatSpeculativeGenerationOptions {
                        scheduler: eredu_core::SpeculativeSchedulerOptions::default()
                            .with_lookahead(lookahead),
                        ..Default::default()
                    },
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| panic!("invalid settings must not emit events"),
                })
                .err()
                .expect("invalid settings must fail");
            let PreparedChatSpeculativeError::Generation(error) = error else {
                panic!("expected portable generation error: {error}");
            };
            check_error(error);
            let error = model
                .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut drafter)
                    },
                    lanes: [PreparedChatGenerationSettings::default(), settings]
                        .into_iter()
                        .map(|settings| PreparedChatSpeculativeBatchLane {
                            input: PreparedChatInput::rendered_prompt(&chat),
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
            let PreparedChatSpeculativeError::Generation(error) = error else {
                panic!("expected portable generation error: {error}");
            };
            check_error(error);
        }
    }
    let calls = calls.borrow();
    assert_eq!(calls.prompts, 0);
    assert_eq!(calls.speculative, 0);
    assert!(calls.configs.is_empty());
    assert!(calls.filters.is_empty());
}
