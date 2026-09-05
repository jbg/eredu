#[test]
fn independent_requests_progress_fairly_and_preserve_output_order() {
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
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    scheduler
        .submit(
            &mut cache_a,
            ModelInput::new(&parts_a),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: vec![0],
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler
        .submit(
            &mut cache_b,
            ModelInput::new(&parts_b),
            SpeculativeConfig {
                max_tokens: 6,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: vec![2],
            },
            None,
            DefaultSampler,
            |_| Ok(()),
        )
        .unwrap();
    scheduler.run().unwrap();
    let output = scheduler.finish().unwrap();

    assert_eq!(output.requests[0].token_ids, vec![1, 2, 0]);
    assert_eq!(output.requests[1].token_ids, vec![1, 2]);
    assert!(output.scheduler.cross_request_draft_opportunities() > 0);
    let verify = backend
        .routes
        .iter()
        .position(|(operation, _)| *operation == "verify")
        .unwrap();
    let cross_draft = backend.routes[verify + 1..]
        .iter()
        .position(|(operation, _)| *operation == "draft")
        .map(|offset| verify + 1 + offset)
        .unwrap();
    let resolve = backend
        .routes
        .iter()
        .position(|(operation, _)| *operation == "commit_target")
        .unwrap();
    assert!(verify < cross_draft && cross_draft < resolve);
}

#[test]
fn scheduler_limits_bound_retained_transactions_and_branches() {
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
        SpeculativeSchedulerOptions {
            max_in_flight_verifications: 2,
            max_optimistic_branches: 1,
            lookahead_blocks: 1,
            ..SpeculativeSchedulerOptions::default()
        },
    )
    .unwrap();
    for (cache, parts) in [(&mut cache_a, &parts_a), (&mut cache_b, &parts_b)] {
        scheduler
            .submit(
                cache,
                ModelInput::new(parts),
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
    }
    scheduler.run().unwrap();
    let stats = scheduler.finish().unwrap().scheduler;
    assert!(stats.peak_in_flight_verifications() <= 2);
    assert!(stats.peak_optimistic_branches() <= 1);
    assert_eq!(stats.peak_optimistic_branches(), 1);
}

#[test]
fn adaptive_disabling_stops_unproductive_branches_without_changing_output() {
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
                    max_tokens: 10,
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
        let request = scheduler.finish().unwrap().requests.pop().unwrap();
        (request.token_ids, cache, request.stats)
    }

    let adaptive = run(SpeculativeSchedulerOptions {
        adaptive_lookahead_min_blocks: 2,
        ..SpeculativeSchedulerOptions::default()
    });
    let disabled = run(SpeculativeSchedulerOptions::default().with_lookahead(false));

    assert_eq!(adaptive.0, disabled.0);
    assert_eq!(adaptive.1, disabled.1);
    assert_eq!(adaptive.2.accept_lens(), disabled.2.accept_lens());
    assert_eq!(adaptive.2.optimistic_draft_blocks(), 2);
    assert!(adaptive.2.adaptive_lookahead_disabled());
    assert_eq!(disabled.2.optimistic_draft_blocks(), 0);
}
