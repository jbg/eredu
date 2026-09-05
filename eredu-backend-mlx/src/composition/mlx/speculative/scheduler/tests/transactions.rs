#[test]
fn full_acceptance_promotes_shared_optimistic_branch() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler.run().unwrap();
    let output = scheduler.finish().unwrap();
    let stats = &output.requests[0].stats;

    assert_eq!(stats.optimistic_draft_blocks(), 1);
    assert_eq!(stats.reused_optimistic_blocks(), 1);
    assert_eq!(stats.reused_optimistic_tokens(), 1);
    assert_eq!(stats.consumed_optimistic_tokens(), 1);
    assert_eq!(stats.optimistic_target_bonus_tokens(), 1);
    assert_eq!(stats.optimistic_bonus_matches(), 1);
    assert_eq!(output.scheduler.peak_optimistic_branches(), 1);
    assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
    let first_commit = backend
        .routes
        .iter()
        .position(|(operation, _)| *operation == "commit_target")
        .unwrap();
    assert!(
        backend.routes[..first_commit]
            .iter()
            .all(|(operation, _)| *operation != "cache_truncate"),
        "a fully accepted bonus-emitting verification must not truncate"
    );
}

#[test]
fn matching_bonus_is_emitted_and_consumes_one_paired_proposal() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let id = scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();

    for _ in 0..4 {
        scheduler.step().unwrap();
    }

    let request = scheduler.requests.request(id).unwrap();
    assert_eq!(request.sequence().tokens(), &[1, 2, 0, 0]);
    assert_eq!(request.status(), SpeculativeRequestStatus::ReadyToDraft);
    let retained = request.block().unwrap();
    assert_eq!(retained.proposals().len(), 1);
    assert_eq!(retained.proposals()[0].token(), 1);
    assert_eq!(retained.state().step, 4);
    assert_eq!(
        retained.proposals()[0]
            .distribution()
            .try_index_device((0, 0, 1), draft.stream())
            .unwrap()
            .item::<f32>(draft.stream()),
        10.0
    );
    assert_eq!(request.stats().consumed_optimistic_tokens(), 1);
    assert_eq!(request.stats().reused_optimistic_tokens(), 1);
    assert_eq!(request.stats().optimistic_bonus_matches(), 1);
    scheduler.step().unwrap();
    assert_eq!(
        scheduler
            .requests
            .request(id)
            .unwrap()
            .block()
            .unwrap()
            .proposals()
            .len(),
        2,
        "the consumed optimistic token must be topped back up before submission"
    );
    scheduler.step().unwrap();
    assert_eq!(
        scheduler
            .backend
            .routes
            .iter()
            .filter(|(operation, _)| *operation == "verify_input_3")
            .count(),
        2
    );

    scheduler.cancel(id).unwrap();
    scheduler.run().unwrap();
    let output = scheduler.finish().unwrap();
    assert_eq!(output.requests[0].token_ids, vec![1, 2, 0, 0]);
    assert_eq!(backend.draft_storage.len(), 5);
}

#[test]
fn mismatching_bonus_restores_exact_non_lookahead_state() {
    #[allow(clippy::type_complexity)]
    fn run(
        options: SpeculativeSchedulerOptions,
    ) -> (Vec<u32>, Vec<u32>, usize, SpeculativeStats, CountingSampler) {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 1,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut callback_tokens = Vec::new();
        let request;
        {
            let mut scheduler = MlxSpeculativeScheduler::new(
                &mut backend,
                SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
                options,
            )
            .unwrap();
            scheduler
                .submit(
                    &mut cache,
                    ModelInput::new(&parts),
                    SpeculativeConfig {
                        max_tokens: 7,
                        max_draft_tokens: 2,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                    None,
                    CountingSampler::default(),
                    |tokens| {
                        callback_tokens.extend_from_slice(tokens);
                        Ok(())
                    },
                )
                .unwrap();
            scheduler.run().unwrap();
            request = scheduler.finish().unwrap().requests.pop().unwrap();
        }
        (
            request.token_ids,
            callback_tokens,
            cache,
            request.stats,
            request.sampler,
        )
    }

    let without = run(SpeculativeSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 0,
        lookahead_blocks: 0,
        ..SpeculativeSchedulerOptions::default()
    });
    let with = run(SpeculativeSchedulerOptions::default());

    assert_eq!(with.0, without.0);
    assert_eq!(with.1, without.1);
    assert_eq!(with.2, without.2);
    assert_eq!(with.3.target_tokens(), without.3.target_tokens());
    assert_eq!(with.3.draft_tokens(), without.3.draft_tokens());
    assert_eq!(with.3.accepted_tokens(), without.3.accepted_tokens());
    assert_eq!(with.3.accept_lens(), without.3.accept_lens());
    assert_eq!(with.3.emitted_tokens(), without.3.emitted_tokens());
    assert_eq!(with.4.process_calls, without.4.process_calls);
    assert_eq!(with.4.histories, without.4.histories);
    assert_eq!(with.4.committed, without.4.committed);
    assert!(with.3.optimistic_bonus_mismatches() > 0);
    assert!(with.3.discarded_optimistic_tokens() > 0);
    assert_eq!(with.3.consumed_optimistic_tokens(), 0);
}

#[test]
fn promoted_round_leaves_last_emitted_token_out_of_target_cache() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let id = scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();

    scheduler.step().unwrap();
    scheduler.step().unwrap();
    scheduler.step().unwrap();
    scheduler.step().unwrap();
    assert_eq!(
        scheduler.status(id),
        Some(SpeculativeRequestStatus::ReadyToDraft)
    );
    scheduler.cancel(id).unwrap();
    let output = scheduler.finish().unwrap();

    assert_eq!(output.requests[0].token_ids, vec![1, 2, 0, 0]);
    // Prefill retained one token. The fully accepted verification evaluated
    // `[first, proposal_1, proposal_2]`. The matching target bonus is
    // emitted immediately but remains outside the target cache.
    assert_eq!(cache, 4);
}

#[test]
fn rejection_discards_branch_sampler_prng_history_and_cache_state() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: true,
        accept_second: false,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let id = scheduler
        .submit(
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
            |_| Ok(()),
        )
        .unwrap();
    scheduler.step().unwrap();
    scheduler.step().unwrap();
    scheduler.step().unwrap();
    assert_eq!(
        scheduler.status(id),
        Some(SpeculativeRequestStatus::OptimisticDraftReady)
    );
    scheduler.step().unwrap();
    scheduler.cancel(id).unwrap();
    let output = scheduler.finish().unwrap();
    let request = &output.requests[0];

    assert_eq!(request.token_ids, vec![1, 1]);
    assert_eq!(request.sampler.process_calls, 2);
    assert_eq!(request.stats.discarded_optimistic_blocks(), 1);
    assert_eq!(request.stats.discarded_optimistic_tokens(), 2);
    assert_eq!(cache, 2);
    assert_eq!(backend.draft_storage[0], backend.draft_storage[2]);
}

#[test]
fn rejection_matches_execution_with_lookahead_disabled() {
    fn run(options: SpeculativeSchedulerOptions) -> (Vec<u32>, usize, SpeculativeStats) {
        let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let prompt = Array::from_slice(&[7u32], &[1, 1]);
        let parts = [InputPart::text_token_ids(&prompt)];
        let mut backend = ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: true,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        };
        let mut cache = 0;
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
            options,
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                SpeculativeConfig {
                    max_tokens: 5,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let mut requests = scheduler.finish().unwrap().requests;
        let output = requests.pop().unwrap();
        (output.token_ids, cache, output.stats)
    }

    let without = run(SpeculativeSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 0,
        lookahead_blocks: 0,
        ..SpeculativeSchedulerOptions::default()
    });
    let with = run(SpeculativeSchedulerOptions::default());
    assert_eq!(with.0, without.0);
    assert_eq!(with.1, without.1);
    assert_eq!(with.2.accept_lens(), without.2.accept_lens());
    assert!(with.2.discarded_optimistic_tokens() > 0);
    assert_eq!(without.2.optimistic_draft_tokens(), 0);
}

#[test]
fn target_eos_discards_in_flight_lookahead() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 0,
        reject_first: true,
        accept_second: false,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 1,
                temperature: 0.0,
                eos_token_ids: vec![0],
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();
    assert_eq!(request.token_ids, vec![1, 0]);
    assert_eq!(request.stats.optimistic_draft_tokens(), 1);
    assert_eq!(request.stats.discarded_optimistic_tokens(), 1);
    assert_eq!(request.stats.reused_optimistic_tokens(), 0);
}

#[test]
fn bonus_eos_completes_only_its_request_and_discards_continuation() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt_a = Array::from_slice(&[7u32], &[1, 1]);
    let prompt_b = Array::from_slice(&[8u32], &[1, 1]);
    let parts_a = [InputPart::text_token_ids(&prompt_a)];
    let parts_b = [InputPart::text_token_ids(&prompt_b)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache_a = 0;
    let mut cache_b = 0;
    let mut callback_a = Vec::new();
    let mut callback_b = Vec::new();
    let output;
    {
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
            SpeculativeSchedulerOptions::default(),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache_a,
                ModelInput::new(&parts_a),
                SpeculativeConfig {
                    max_tokens: 6,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: vec![0],
                },
                None,
                DefaultSampler,
                |tokens| {
                    callback_a.extend_from_slice(tokens);
                    Ok(())
                },
            )
            .unwrap();
        scheduler
            .submit(
                &mut cache_b,
                ModelInput::new(&parts_b),
                SpeculativeConfig {
                    max_tokens: 5,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                DefaultSampler,
                |tokens| {
                    callback_b.extend_from_slice(tokens);
                    Ok(())
                },
            )
            .unwrap();
        scheduler.run().unwrap();
        output = scheduler.finish().unwrap();
    }

    assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
    assert_eq!(callback_a, output.requests[0].token_ids);
    assert_eq!(output.requests[0].stats.optimistic_target_bonus_tokens(), 1);
    assert_eq!(output.requests[0].stats.optimistic_bonus_matches(), 0);
    assert_eq!(output.requests[0].stats.discarded_optimistic_tokens(), 1);
    assert_eq!(output.requests[1].token_ids.len(), 5);
    assert_eq!(callback_b, output.requests[1].token_ids);
    assert!(output.requests[1].stats.rounds() > output.requests[0].stats.rounds());
    assert!(output.scheduler.cross_request_draft_opportunities() > 0);
}

#[test]
fn consumed_one_token_branch_never_submits_empty_verification() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 1,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 2, 0, 2, 0]);
    assert_eq!(request.stats.consumed_optimistic_tokens(), 1);
    assert_eq!(request.stats.reused_optimistic_tokens(), 0);
    assert_eq!(request.stats.reused_optimistic_blocks(), 0);
    assert_eq!(
        backend
            .routes
            .iter()
            .filter(|(operation, _)| *operation == "verify")
            .count(),
        2
    );
}

#[test]
fn max_token_boundary_does_not_draft_unusable_lookahead() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;
    let mut scheduler = MlxSpeculativeScheduler::new(
        &mut backend,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit(
            &mut cache,
            ModelInput::new(&parts),
            SpeculativeConfig {
                max_tokens: 3,
                max_draft_tokens: 1,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler.run().unwrap();
    let request = scheduler.finish().unwrap().requests.pop().unwrap();

    assert_eq!(request.token_ids, vec![1, 2, 0]);
    assert_eq!(request.stats.optimistic_draft_tokens(), 0);
    assert_eq!(request.stats.optimistic_draft_blocks(), 0);
    assert_eq!(backend.draft_storage.len(), 1);
}

#[test]
fn adaptive_sampler_does_not_claim_exact_optimistic_promotion() {
    assert!(
        SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(
            &GenerationSampler::new()
        )
    );
    assert!(
        !SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(
            &MirostatV2Sampler::default()
        )
    );
}

#[test]
fn stale_optimistic_prefix_is_an_error_not_a_fallback() {
    let branch = SpeculativeOptimisticBranch::new(
        SpeculativeDraftBlock::new(
            ScriptedDraftState {
                step: 1,
                storage: Arc::new(()),
            },
            vec![SpeculativeProposal::new(
                0,
                Array::from_slice(&[1.0f32, 0.0, 0.0], &[1, 1, 3]),
            )],
        ),
        vec![1, 2],
    );
    let mut stats = SpeculativeStats::default();
    let error = resolve_optimistic_branch(Some(branch), &[1, 0], Some(0), false, &mut stats)
        .err()
        .unwrap();

    assert!(error
        .to_string()
        .contains("diverged from the canonical committed prefix"));
    assert_eq!(stats.optimistic_target_bonus_tokens(), 0);
    assert_eq!(stats.optimistic_bonus_matches(), 0);
    assert_eq!(stats.optimistic_bonus_mismatches(), 0);
    assert_eq!(stats.consumed_optimistic_tokens(), 0);
    assert_eq!(stats.discarded_optimistic_tokens(), 0);
}
