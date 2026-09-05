#[test]
fn history_derived_sampler_matches_non_lookahead_execution() {
    fn run(options: SpeculativeSchedulerOptions) -> (Vec<u32>, Vec<usize>) {
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
        let sampler = GenerationSampler::new()
            .top_k(0)
            .top_p(1.0)
            .min_p(0.0)
            .penalties(1.2, -1, 0.1, 0.1);
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
                sampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        (request.token_ids, request.stats.accept_lens().to_vec())
    }

    let without = run(SpeculativeSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 0,
        lookahead_blocks: 0,
        ..SpeculativeSchedulerOptions::default()
    });
    let with = run(SpeculativeSchedulerOptions::default());
    assert_eq!(with, without);
}

#[test]
fn stochastic_match_and_mismatch_ignore_interleaving_and_branch_slots() {
    fn run(
        seed: u64,
        options: SpeculativeSchedulerOptions,
        with_peer: bool,
    ) -> (Vec<u32>, Vec<usize>, SpeculativeStats) {
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
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
            options,
        )
        .unwrap();
        let config = SpeculativeConfig {
            max_tokens: 8,
            max_draft_tokens: 2,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        scheduler
            .submit(
                &mut cache_a,
                ModelInput::new(&parts_a),
                config.clone(),
                Some(safemlx::random::key(seed).unwrap()),
                UniformSampler,
                |_| Ok(()),
            )
            .unwrap();
        if with_peer {
            scheduler
                .submit(
                    &mut cache_b,
                    ModelInput::new(&parts_b),
                    config,
                    Some(safemlx::random::key(seed + 1000).unwrap()),
                    UniformSampler,
                    |_| Ok(()),
                )
                .unwrap();
        }
        scheduler.run().unwrap();
        let request = scheduler.finish().unwrap().requests.remove(0);
        (
            request.token_ids,
            request.stats.accept_lens().to_vec(),
            request.stats,
        )
    }

    let no_lookahead = SpeculativeSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 0,
        lookahead_blocks: 0,
        ..SpeculativeSchedulerOptions::default()
    };
    let mut saw_match = false;
    let mut saw_mismatch = false;
    for seed in 0..64 {
        let with = run(seed, SpeculativeSchedulerOptions::default(), false);
        if with.2.optimistic_bonus_matches() == 0 && with.2.optimistic_bonus_mismatches() == 0 {
            continue;
        }
        let without = run(seed, no_lookahead, false);
        let interleaved = run(seed, SpeculativeSchedulerOptions::default(), true);
        assert_eq!((&with.0, &with.1), (&without.0, &without.1));
        assert_eq!((&with.0, &with.1), (&interleaved.0, &interleaved.1));
        saw_match |= with.2.optimistic_bonus_matches() > 0;
        saw_mismatch |= with.2.optimistic_bonus_mismatches() > 0;
        if saw_match && saw_mismatch {
            break;
        }
    }
    assert!(saw_match, "scripted seeds did not exercise a bonus match");
    assert!(
        saw_mismatch,
        "scripted seeds did not exercise a bonus mismatch"
    );
}

#[test]
fn stochastic_request_is_reproducible_across_scheduler_interleavings() {
    fn run(
        with_peer: bool,
    ) -> (
        Vec<u32>,
        Vec<usize>,
        Vec<SemanticEvent>,
        Option<FinishReason>,
    ) {
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
        let events = Rc::new(RefCell::new(Vec::new()));
        let callback = Rc::clone(&events);
        let mut scheduler = MlxSpeculativeScheduler::new(
            &mut backend,
            SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
            SpeculativeSchedulerOptions::default(),
        )
        .unwrap();
        let config = SpeculativeConfig {
            max_tokens: 5,
            max_draft_tokens: 2,
            temperature: 1.0,
            eos_token_ids: Vec::new(),
        };
        scheduler
            .submit_with_semantics(
                &mut cache_a,
                ModelInput::new(&parts_a),
                config.clone(),
                Some(safemlx::random::key(7).unwrap()),
                MirostatV2Sampler::default(),
                Box::new(TestSemanticState::default()),
                move |event| callback.borrow_mut().push(event),
            )
            .unwrap();
        if with_peer {
            scheduler
                .submit(
                    &mut cache_b,
                    ModelInput::new(&parts_b),
                    config,
                    Some(safemlx::random::key(99).unwrap()),
                    MirostatV2Sampler::default(),
                    |_| Ok(()),
                )
                .unwrap();
        }
        scheduler.run().unwrap();
        let output = scheduler.finish().unwrap();
        let events = events.borrow().clone();
        (
            output.requests[0].token_ids.clone(),
            output.requests[0].stats.accept_lens().to_vec(),
            events,
            output.requests[0].finish_reason,
        )
    }

    assert_eq!(run(false), run(true));
}
