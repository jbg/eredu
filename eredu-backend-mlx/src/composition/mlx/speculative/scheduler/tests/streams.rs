#[test]
fn greedy_engine_commits_only_the_accepted_prefix() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let input = ModelInput::new(&parts);
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 2,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let mut cache = 0;
    let mut emitted = Vec::new();
    let (tokens, stats) = generate_tokens(
        &mut ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        },
        &mut cache,
        input,
        &config,
        None,
        &mut DefaultSampler,
        SpeculativeExecutionStreams::single(stream),
        SpeculativeSchedulerOptions::default(),
        |tokens| {
            emitted.extend_from_slice(tokens);
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(tokens, vec![1, 2, 1]);
    assert_eq!(emitted, tokens);
    assert_eq!(stats.accept_lens(), vec![1]);
    assert_eq!(stats.accepted_tokens(), 1);
    assert_eq!(cache, 3);
}

#[test]
#[ignore = "requires an MLX Metal device"]
fn cpu_draft_gpu_target_split_stream_routes_and_commits() {
    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let input = ModelInput::new(&parts);
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 2,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let mut backend = ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: false,
        bonus_token: 0,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    };
    let mut cache = 0;

    let (tokens, _) = generate_with_streams(
        &mut backend,
        &mut cache,
        input,
        &config,
        None,
        &mut DefaultSampler,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
    )
    .unwrap();

    assert_eq!(tokens, vec![1, 2, 1]);
    for (operation, device) in backend.routes {
        let expected =
            if operation == "begin_draft" || operation == "draft" || operation == "commit_draft" {
                DeviceType::Cpu
            } else {
                DeviceType::Gpu
            };
        assert_eq!(device, expected, "{operation}");
    }
}

#[test]
#[ignore = "requires an MLX Metal device"]
fn split_stream_engine_preserves_stochastic_acceptance() {
    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let input = ModelInput::new(&parts);
    let config = SpeculativeConfig {
        max_tokens: 4,
        max_draft_tokens: 4,
        temperature: 1.0,
        eos_token_ids: Vec::new(),
    };
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
    let mut sampler = MirostatV2Sampler::default();
    let key = safemlx::random::key(7).unwrap();

    let (tokens, stats) = generate_with_streams(
        &mut backend,
        &mut cache,
        input,
        &config,
        Some(key),
        &mut sampler,
        SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap(),
    )
    .unwrap();

    assert_eq!(tokens, vec![1, 2, 0, 0]);
    assert_eq!(stats.accepted_tokens(), 2);
    assert_eq!(sampler.generated_tokens(), tokens);
}

#[test]
fn mirostat_v2_speculative_commits_accepted_target_distributions() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let input = ModelInput::new(&parts);
    let config = SpeculativeConfig {
        max_tokens: 4,
        max_draft_tokens: 4,
        temperature: 1.0,
        eos_token_ids: Vec::new(),
    };
    let mut cache = 0;
    let mut sampler = MirostatV2Sampler::default();
    let key = safemlx::random::key(7).unwrap();

    let (tokens, stats) = generate(
        &mut ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: false,
            accept_second: true,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        },
        &mut cache,
        input,
        &config,
        Some(key),
        &mut sampler,
        stream,
    )
    .unwrap();

    assert_eq!(tokens, vec![1, 2, 0, 0]);
    assert_eq!(stats.draft_tokens(), 2);
    assert_eq!(stats.accepted_tokens(), 2);
    assert_eq!(sampler.generated_tokens(), tokens);
    assert!((sampler.mu() - 12.0).abs() < 1e-4);
}

#[test]
fn mirostat_v2_speculative_commits_replacement_not_rejected_draft() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let input = ModelInput::new(&parts);
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 4,
        temperature: 1.0,
        eos_token_ids: Vec::new(),
    };
    let mut cache = 0;
    let mut sampler = MirostatV2Sampler::default();
    let key = safemlx::random::key(11).unwrap();

    let (tokens, stats) = generate(
        &mut ScriptedBackend {
            first_token: 1,
            rejection_token: 1,
            reject_first: true,
            accept_second: false,
            bonus_token: 0,
            routes: Vec::new(),
            draft_storage: Vec::new(),
            draft_capacities: Vec::new(),
        },
        &mut cache,
        input,
        &config,
        Some(key),
        &mut sampler,
        stream,
    )
    .unwrap();

    assert_eq!(tokens, vec![1, 1]);
    assert_eq!(stats.draft_tokens(), 1);
    assert_eq!(stats.accepted_tokens(), 0);
    assert_eq!(sampler.generated_tokens(), tokens);
    assert!((sampler.mu() - 11.0).abs() < 1e-4);
}

#[test]
fn execution_stream_topologies_classify_on_cpu_devices() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let second_stream = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let second_device = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));

    assert_eq!(
        SpeculativeExecutionStreams::single(target.stream()).topology(),
        SpeculativeExecutionTopology::Single
    );
    assert_eq!(
        SpeculativeExecutionStreams::for_test(target.stream(), second_stream.stream())
            .unwrap()
            .topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert_eq!(
        SpeculativeExecutionStreams::for_test(target.stream(), second_device.stream())
            .unwrap()
            .topology(),
        SpeculativeExecutionTopology::CrossDeviceSplit
    );
}

#[test]
fn same_device_event_handoffs_order_both_cpu_stream_directions() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let streams = SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap();
    assert_eq!(
        streams.topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );

    let lhs = Array::ones::<f32>(&[1024, 1024], target.stream()).unwrap();
    let rhs = Array::ones::<f32>(&[1024, 1024], target.stream()).unwrap();
    let target_output = lhs.matmul(&rhs, target.stream()).unwrap();
    // CPU work may finish before it can be observed as incomplete. The
    // final value below proves ordering; the explicit Metal test proves
    // that a same-device handoff does not synchronize its producer.
    let target_handoff = streams.wait_for_target_outputs([&target_output]).unwrap();

    let draft_output = target_output
        .add(Array::from_f32(1.0), draft.stream())
        .unwrap();
    let draft_handoff = streams.wait_for_draft_outputs([&draft_output]).unwrap();
    let consumed = draft_output
        .add(Array::from_f32(1.0), target.stream())
        .unwrap();
    let consumed_completion = async_eval_with_event([&consumed]).unwrap();

    // Queued waits retain both handoffs after their public handles drop.
    drop(target_handoff);
    drop(draft_handoff);
    consumed_completion.synchronize().unwrap();
    assert_eq!(
        consumed
            .try_index_device((0, 0), target.stream())
            .unwrap()
            .item::<f32>(target.stream()),
        1026.0
    );
}

#[test]
#[ignore = "requires an MLX Metal device"]
fn execution_streams_classify_single_same_device_and_cross_device_topologies() {
    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let second_gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));

    let single = SpeculativeExecutionStreams::single(target.stream());
    let same_device =
        SpeculativeExecutionStreams::for_test(target.stream(), second_gpu.stream()).unwrap();
    let cross_device =
        SpeculativeExecutionStreams::for_test(target.stream(), cpu.stream()).unwrap();

    assert_eq!(single.topology(), SpeculativeExecutionTopology::Single);
    assert!(!single.is_split());
    assert!(!single.crosses_devices());
    assert_eq!(
        same_device.topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert!(same_device.is_split());
    assert!(!same_device.crosses_devices());
    assert_eq!(
        cross_device.topology(),
        SpeculativeExecutionTopology::CrossDeviceSplit
    );
    assert!(cross_device.is_split());
    assert!(cross_device.crosses_devices());
}

#[test]
#[ignore = "requires an MLX Metal device"]
fn same_gpu_split_stream_runs_exact_optimistic_lookahead() {
    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    assert_ne!(
        target.stream().get_index().unwrap(),
        draft.stream().get_index().unwrap()
    );
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

    assert_eq!(
        output.scheduler.execution_topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert_eq!(
        stats.execution_topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert!(stats.optimistic_draft_blocks() > 0);
    assert!(stats.optimistic_target_bonus_tokens() > 0);
    assert_eq!(
        stats.optimistic_bonus_matches() + stats.optimistic_bonus_mismatches(),
        stats.optimistic_target_bonus_tokens()
    );
    assert!(backend
        .routes
        .iter()
        .all(|(_, device)| *device == DeviceType::Gpu));
    let verify = backend
        .routes
        .iter()
        .position(|(operation, _)| *operation == "verify")
        .unwrap();
    let optimistic_draft = backend.routes[verify + 1..]
        .iter()
        .position(|(operation, _)| *operation == "draft")
        .map(|offset| verify + 1 + offset)
        .unwrap();
    let resolve = backend
        .routes
        .iter()
        .position(|(operation, _)| *operation == "commit_target")
        .unwrap();
    assert!(verify < optimistic_draft && optimistic_draft < resolve);
}

#[test]
#[ignore = "explicit Metal speculative event handoff test; run on a Metal host"]
fn same_gpu_speculative_handoff_does_not_synchronize_the_producer_stream() {
    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let streams = SpeculativeExecutionStreams::for_test(target.stream(), draft.stream()).unwrap();
    assert_eq!(
        streams.topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert_ne!(
        target.stream().get_index().unwrap(),
        draft.stream().get_index().unwrap()
    );

    let lhs = Array::ones::<f32>(&[4096, 4096], target.stream()).unwrap();
    let rhs = Array::ones::<f32>(&[4096, 4096], target.stream()).unwrap();
    let produced = lhs.matmul(&rhs, target.stream()).unwrap();
    let handoff = streams.wait_for_target_outputs([&produced]).unwrap();
    assert!(
        !handoff.is_complete().unwrap(),
        "the same-device speculative handoff waited for target completion on the host"
    );

    let consumed = produced.square(draft.stream()).unwrap();
    let completion = async_eval_with_event([&consumed]).unwrap();
    drop(handoff);
    completion.synchronize().unwrap();
}

#[test]
fn same_device_split_preserves_stochastic_lookahead() {
    let target = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let prompt = Array::from_slice(&[7u32], &[1, 1]);
    let parts = [InputPart::text_token_ids(&prompt)];
    let run = |lookahead| {
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
            SpeculativeSchedulerOptions::default().with_lookahead(lookahead),
        )
        .unwrap();
        scheduler
            .submit(
                &mut cache,
                ModelInput::new(&parts),
                SpeculativeConfig {
                    max_tokens: 8,
                    max_draft_tokens: 2,
                    temperature: 1.0,
                    eos_token_ids: Vec::new(),
                },
                Some(safemlx::random::key(7).unwrap()),
                UniformSampler,
                |_| Ok(()),
            )
            .unwrap();
        scheduler.run().unwrap();
        scheduler.finish().unwrap().requests.pop().unwrap()
    };
    let request = run(true);
    let canonical = run(false);

    assert_eq!(
        request.stats.execution_topology(),
        SpeculativeExecutionTopology::SameDeviceSplit
    );
    assert_eq!(request.token_ids, canonical.token_ids);
    assert_eq!(request.token_ids.len(), 8);
    assert_eq!(request.stats.emitted_tokens(), 8);
    assert!(request.stats.optimistic_draft_blocks() > 0);
    assert_eq!(canonical.stats.optimistic_draft_blocks(), 0);
}
