#[test]
fn native_cancellation_is_rejected_before_speculative_submission() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let mut backend = scripted_backend();
    let options = SpeculativeSchedulerOptions::default()
        .with_completion_wait(
            Duration::from_millis(1),
            eredu_core::CompletionCancellationMode::NativeCancel,
        )
        .unwrap();

    let error = MlxSpeculativeScheduler::<_, UniformSampler>::new(
        &mut backend,
        SpeculativeExecutionStreams::single(context.stream()),
        options,
    )
    .err()
    .expect("MLX must reject unsupported native cancellation");

    assert!(error.to_string().contains("does not support NativeCancel"));
    assert!(backend.routes.is_empty());
}

#[test]
fn scheduler_sizes_new_draft_rounds_to_the_available_proposals() {
    fn capacities(max_tokens: usize) -> Vec<usize> {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::single(context.stream()),
            SpeculativeSchedulerOptions::default().with_lookahead(false),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                SpeculativeConfig {
                    max_tokens,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                Box::new(TestSemanticState::default()),
                |_| {},
            )
            .unwrap();
        scheduler.run().unwrap();
        scheduler.finish().unwrap();
        backend.draft_capacities
    }

    assert_eq!(capacities(2), [1]);
    assert_eq!(capacities(3), [2]);
}

#[test]
fn ordinary_canonical_and_speculative_lookahead_have_identical_semantics() {
    fn run_speculative(lookahead: bool) -> (Vec<u32>, Vec<SemanticEvent>, FinishReason) {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let events = Rc::new(RefCell::new(Vec::new()));
        let callback = Rc::clone(&events);
        let mut cache = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
            SpeculativeSchedulerOptions::default().with_lookahead(lookahead),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                SpeculativeConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                CountingSampler::default(),
                Box::new(TestSemanticState::default()),
                move |event| callback.borrow_mut().push(event),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        let finish_reason = request.finish_reason.unwrap();
        let events = events.borrow().clone();
        (request.token_ids, events, finish_reason)
    }

    let canonical = run_speculative(false);
    let lookahead = run_speculative(true);
    assert_eq!(lookahead, canonical);

    let mut ordinary = TestSemanticState::default();
    let mut ordinary_events = Vec::new();
    for &token in &canonical.0 {
        assert!(!ordinary.push_token(token).unwrap());
        ordinary_events.extend(ordinary.take_events());
    }
    ordinary.finish(canonical.2).unwrap();
    ordinary_events.extend(ordinary.take_events());
    assert_eq!(ordinary_events, canonical.1);
    assert_eq!(
        canonical.1.last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::MaxTokens
        })
    );
}

#[test]
fn optimistic_semantic_stop_truncates_an_accepted_block_transactionally() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = target.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut cache = 0;
    let events = Rc::new(RefCell::new(Vec::new()));
    let callback_events = Rc::clone(&events);
    let mut backend = scripted_backend();
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(stream, draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit_with_semantics(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            CountingSampler::default(),
            Box::new(TestSemanticState {
                stop: vec![1, 2],
                ..TestSemanticState::default()
            }),
            move |event| callback_events.borrow_mut().push(event),
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 2]);
    assert_eq!(request.finish_reason, Some(FinishReason::StopSequence));
    assert_eq!(request.stats.accept_lens(), vec![1]);
    assert_eq!(request.sampler.process_calls, 2);
    assert_eq!(request.stats.optimistic_draft_blocks(), 1);
    assert_eq!(request.stats.discarded_optimistic_blocks(), 1);
    assert_eq!(cache, 2);
    assert_eq!(
        events.borrow().as_slice(),
        [
            SemanticEvent::TextDelta("1".into()),
            SemanticEvent::TextDelta("2".into()),
            SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            }
        ],
        "draft and optimistic tokens must never publish semantic events"
    );
    assert_eq!(
        events.borrow().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence
        })
    );
}

#[test]
fn grammar_completion_mid_block_matches_the_committed_prefix() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut cache = 0;
    let mut backend = scripted_backend();
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::single(stream),
        SpeculativeSchedulerOptions::default().with_lookahead(false),
    )
    .unwrap();
    scheduler
        .submit_with_semantics(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            GrammarCountingSampler {
                inner: CountingSampler::default(),
                complete_after: 2,
            },
            Box::new(TestSemanticState::default()),
            |_| {},
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 2]);
    assert_eq!(request.finish_reason, Some(FinishReason::GrammarComplete));
    assert_eq!(request.sampler.inner.process_calls, 2);
    assert_eq!(request.sampler.inner.committed, request.token_ids);
    assert_eq!(request.stats.draft_tokens(), 1);
    assert_eq!(
        backend.draft_storage.len(),
        1,
        "drafting must stop as soon as the logical prefix completes the grammar"
    );
    assert_eq!(cache, 2);

    let mut ordinary = GrammarCountingSampler {
        inner: CountingSampler::default(),
        complete_after: 2,
    };
    let mut ordinary_tokens = Vec::new();
    for expected in [1u32, 2, 0] {
        let mut values = [0.0f32; 3];
        values[expected as usize] = 10.0;
        let logits = MlxTensor::from_array(Array::from_slice(&values, &[1, 3]));
        let processed = ordinary
            .process_logits(&logits, 0.0, &ordinary_tokens, stream)
            .unwrap();
        let chosen = ordinary
            .sample_processed(&processed, 0.0, None, stream)
            .unwrap()
            .into_array()
            .item::<u32>(stream);
        ordinary.commit_token(&processed, chosen, stream).unwrap();
        ordinary_tokens.push(chosen);
        if ordinary.grammar_is_complete().unwrap() {
            break;
        }
    }
    assert_eq!(request.token_ids, ordinary_tokens);
}

#[test]
fn terminal_grammar_bonus_discards_matching_optimistic_work() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut cache = 0;
    let mut backend = scripted_backend();
    backend.bonus_token = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit_with_semantics(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 8,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            GrammarCountingSampler {
                inner: CountingSampler::default(),
                complete_after: 4,
            },
            Box::new(TestSemanticState::default()),
            |_| {},
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 2, 0, 0]);
    assert_eq!(request.finish_reason, Some(FinishReason::GrammarComplete));
    assert_eq!(request.stats.optimistic_target_bonus_tokens(), 1);
    assert_eq!(request.stats.discarded_optimistic_tokens(), 1);
    assert_eq!(request.stats.reused_optimistic_tokens(), 0);
    assert_eq!(request.stats.consumed_optimistic_tokens(), 0);
    assert_eq!(request.sampler.inner.committed, request.token_ids);
}

#[test]
fn cancellation_discards_only_the_affected_request_runtime() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
    let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
    let parts_a = [InputPart::text_token_ids(&prompt_a)];
    let parts_b = [InputPart::text_token_ids(&prompt_b)];
    let events_a = Rc::new(RefCell::new(Vec::new()));
    let events_b = Rc::new(RefCell::new(Vec::new()));
    let callback_a = Rc::clone(&events_a);
    let callback_b = Rc::clone(&events_b);
    let cancellation_a = GenerationCancellationToken::new();
    let cancellation_b = GenerationCancellationToken::new();
    let mut cache_a = 0;
    let mut cache_b = 0;
    let mut backend = scripted_backend();
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let config = SpeculativeConfig {
        max_tokens: 5,
        max_draft_tokens: 2,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let first = scheduler
        .submit_with_semantics_cancellable(
            &mut cache_a,
            ModelInput::new(&parts_a),
            config.clone(),
            None,
            GrammarCountingSampler {
                inner: CountingSampler::default(),
                complete_after: usize::MAX,
            },
            Box::new(TestSemanticState::default()),
            cancellation_a.clone(),
            move |event| callback_a.borrow_mut().push(event),
        )
        .unwrap();
    scheduler
        .submit_with_semantics_cancellable(
            &mut cache_b,
            ModelInput::new(&parts_b),
            config,
            None,
            GrammarCountingSampler {
                inner: CountingSampler::default(),
                complete_after: 3,
            },
            Box::new(TestSemanticState::default()),
            cancellation_b.clone(),
            move |event| callback_b.borrow_mut().push(event),
        )
        .unwrap();

    scheduler.step().unwrap();
    scheduler.step().unwrap();
    assert!(scheduler
        .requests
        .request(first)
        .unwrap()
        .has_pending_verification());
    cancellation_a.cancel();
    assert!(!cancellation_b.is_cancelled());
    scheduler.run().unwrap();
    let output = scheduler.finish().unwrap();

    assert!(output.requests[0].cancelled);
    assert_eq!(output.requests[0].token_ids, vec![1]);
    assert_eq!(
        output.requests[0].finish_reason,
        Some(FinishReason::Cancelled)
    );
    assert_eq!(output.requests[0].sampler.inner.committed, vec![1]);
    assert_eq!(
        events_a.borrow().as_slice(),
        &[
            SemanticEvent::TextDelta("1".into()),
            SemanticEvent::Finished {
                reason: FinishReason::Cancelled
            }
        ]
    );
    assert!(!output.requests[1].cancelled);
    assert_eq!(output.requests[1].token_ids, vec![1, 2, 0]);
    assert_eq!(
        output.requests[1].finish_reason,
        Some(FinishReason::GrammarComplete)
    );
    assert_eq!(
        output.requests[1].sampler.inner.committed,
        output.requests[1].token_ids
    );
    assert_eq!(
        events_b.borrow().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::GrammarComplete
        })
    );
}

#[test]
fn semantic_callback_token_cancels_at_the_committed_prefill_boundary() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let events = Rc::new(RefCell::new(Vec::new()));
    let callback_events = Rc::clone(&events);
    let cancellation = GenerationCancellationToken::new();
    let callback_cancellation = cancellation.clone();
    let mut cache = 0;
    let mut backend = scripted_backend();
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::single(context.stream()),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();

    scheduler
        .submit_with_semantics_cancellable(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            GrammarCountingSampler {
                inner: CountingSampler::default(),
                complete_after: usize::MAX,
            },
            Box::new(TestSemanticState::default()),
            cancellation,
            move |event| {
                callback_events.borrow_mut().push(event.clone());
                if matches!(event, SemanticEvent::TextDelta(_)) {
                    callback_cancellation.cancel();
                }
            },
        )
        .unwrap();

    assert!(scheduler.is_finished());
    scheduler.run().unwrap();
    let output = scheduler.finish().unwrap();
    let request = &output.requests[0];
    assert!(request.cancelled);
    assert_eq!(request.token_ids, vec![1]);
    assert_eq!(request.sampler.inner.committed, request.token_ids);
    assert_eq!(request.finish_reason, Some(FinishReason::Cancelled));
    assert_eq!(cache, 1);
    assert!(backend.draft_storage.is_empty());
    assert_eq!(
        events.borrow().as_slice(),
        &[
            SemanticEvent::TextDelta("1".into()),
            SemanticEvent::Finished {
                reason: FinishReason::Cancelled
            }
        ]
    );
}

#[test]
fn no_lookahead_full_acceptance_bonus_eos_and_max_tokens_use_safe_boundaries() {
    fn run(
        max_tokens: usize,
        eos_token_ids: Vec<u32>,
    ) -> (Vec<u32>, FinishReason, SpeculativeStats, usize) {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut cache = 0;
        let mut backend = scripted_backend();
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::single(stream),
            SpeculativeSchedulerOptions::default().with_lookahead(false),
        )
        .unwrap();
        scheduler
            .submit_with_semantics(
                &mut cache,
                ModelInput::new(&parts),
                SpeculativeConfig {
                    max_tokens,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids,
                },
                None,
                CountingSampler::default(),
                Box::new(TestSemanticState::default()),
                |_| {},
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        (
            request.token_ids,
            request.finish_reason.unwrap(),
            request.stats,
            cache,
        )
    }

    let eos = run(6, vec![2]);
    assert_eq!(eos.0, vec![1, 2]);
    assert_eq!(eos.1, FinishReason::Eos);
    assert_eq!(eos.2.accept_lens(), vec![1]);
    assert_eq!(eos.3, 2);

    let max = run(2, Vec::new());
    assert_eq!(max.0, vec![1, 2]);
    assert_eq!(max.1, FinishReason::MaxTokens);
    assert_eq!(max.2.accept_lens(), vec![1]);
    assert_eq!(max.3, 2);

    let bonus = run(4, Vec::new());
    assert_eq!(bonus.0, vec![1, 2, 0, 1]);
    assert_eq!(bonus.1, FinishReason::MaxTokens);
    assert_eq!(bonus.2.accept_lens(), vec![2]);
    assert_eq!(bonus.2.accepted_tokens(), 2);
    assert_eq!(bonus.3, 4);
}

#[test]
fn no_lookahead_residual_replacement_commits_only_the_replacement() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut cache = 0;
    let mut backend = scripted_backend();
    backend.reject_first = true;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::single(stream),
        SpeculativeSchedulerOptions::default().with_lookahead(false),
    )
    .unwrap();
    scheduler
        .submit_with_semantics(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 2,
                max_draft_tokens: 2,
                temperature: 1.0,
                eos_token_ids: Vec::new(),
            },
            Some(safemlx::random::key(11).unwrap()),
            CountingSampler::default(),
            Box::new(TestSemanticState::default()),
            |_| {},
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 1]);
    assert_eq!(request.finish_reason, Some(FinishReason::MaxTokens));
    assert_eq!(request.stats.accept_lens(), vec![0]);
    assert_eq!(request.sampler.committed, request.token_ids);
    assert_eq!(cache, 2);
}

#[test]
fn commit_failure_does_not_publish_or_advance_canonical_output_state() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut cache = 0;
    let events = Rc::new(RefCell::new(Vec::new()));
    let callback_events = Rc::clone(&events);
    let mut backend = CommitFailBackend {
        inner: scripted_backend(),
    };
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::single(stream),
        SpeculativeSchedulerOptions::default().with_lookahead(false),
    )
    .unwrap();
    let id = scheduler
        .submit_with_semantics(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            CountingSampler::default(),
            Box::new(TestSemanticState::default()),
            move |event| callback_events.borrow_mut().push(event),
        )
        .unwrap();
    let published_after_prefill = events.borrow().len();
    assert!(scheduler.run().is_err());
    let request = scheduler.requests.request(id).unwrap();

    assert_eq!(events.borrow().len(), published_after_prefill);
    assert_eq!(request.sequence().tokens(), &[1]);
    assert_eq!(request.sampler().inner().committed, vec![1]);
    assert_eq!(request.sampler().inner().process_calls, 1);
}
