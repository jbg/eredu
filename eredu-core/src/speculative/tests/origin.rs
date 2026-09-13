use super::*;

#[test]
fn scheduler_attributes_interleaved_requests_and_optimistic_prefixes() {
    for full_acceptance in [false, true] {
        let mut executor = MockExecutor {
            capture_origins: true,
            full_acceptance,
            ..Default::default()
        };
        let mut first_cache = Vec::new();
        let mut second_cache = Vec::new();
        let mut table = SpeculativeRequestTable::new(
            SpeculativeSchedulerOptions::default(),
            SpeculativeExecutionTopology::SameDeviceSplit,
        )
        .unwrap();
        for (cache, prompt) in [(&mut first_cache, vec![4, 8]), (&mut second_cache, vec![9])] {
            table
                .submit(
                    &mut executor,
                    cache,
                    prompt,
                    SpeculativeConfig {
                        max_tokens: 8,
                        max_draft_tokens: 2,
                        temperature: 0.7,
                        eos_token_ids: Vec::new(),
                    },
                    empty_mock_runtime(8, GenerationCancellationToken::new()),
                    SpeculativeRandomness {
                        target: Some(0),
                        draft: Some(0),
                    },
                    false,
                    (),
                )
                .unwrap();
            assert!(executor.origin.is_none());
        }
        while !table.is_finished() {
            let start = executor.activations.len();
            let prefixes: Vec<_> = (0..2)
                .map(|index| {
                    table
                        .request(SpeculativeRequestId::new(index))
                        .unwrap()
                        .sequence()
                        .tokens()
                        .to_vec()
                })
                .collect();
            table.step(&mut executor, true, ()).unwrap();
            assert!(executor.origin.is_none());
            for (stage, origin) in &executor.activations[start..] {
                let origin = origin.expect("every scheduled native operation has an origin");
                let committed = &prefixes[origin.request.index()];
                assert_eq!(origin.committed_tokens, committed.len());
                if origin.optimistic {
                    assert_eq!(*stage, "proposal");
                    assert!(origin.prediction > origin.committed_tokens);
                }
                if *stage != "proposal" {
                    assert!(!origin.optimistic);
                    assert_eq!(origin.prediction, committed.len());
                    assert_eq!(
                        origin.prefix_digest,
                        SpeculativeActivationOrigin::new(origin.request, committed, false)
                            .prefix_digest
                    );
                }
            }
        }
        let output = table.finish().unwrap();
        assert_eq!(output.requests.len(), 2);
        assert!(executor
            .activations
            .iter()
            .any(|(_, o)| o.unwrap().optimistic));
        assert!(executor
            .activations
            .iter()
            .any(|(s, o)| *s == "replay" && o.unwrap().committed_tokens > 1));
        for request in 0..2 {
            let first: Vec<_> = executor
                .activations
                .iter()
                .filter_map(|(stage, origin)| {
                    let origin = origin.unwrap();
                    (origin.request.index() == request
                        && *stage == "proposal"
                        && origin.committed_tokens == 1)
                        .then_some(origin)
                })
                .collect();
            assert!(first.len() >= 4);
            for (offset, origin) in first[..4].iter().enumerate() {
                assert_eq!(origin.prediction, offset + 1);
                assert_eq!(origin.optimistic, offset >= 2);
                let prefix = vec![1; offset + 1];
                assert_eq!(
                    origin.prefix_digest,
                    SpeculativeActivationOrigin::new(origin.request, &prefix, false).prefix_digest
                );
                assert_eq!(
                    serde_json::from_value::<SpeculativeActivationOrigin>(
                        serde_json::to_value(origin).unwrap()
                    )
                    .unwrap(),
                    *origin
                );
            }
        }
    }
}

#[test]
fn operation_origin_clears_after_errors_and_unwind() {
    let mut executor = MockExecutor::default();
    let origin = SpeculativeActivationOrigin::new(SpeculativeRequestId::new(7), &[2, 5], true);
    let result = with_activation_origin(&mut executor, Some(origin), |e| {
        assert_eq!(e.origin, Some(origin));
        Err::<(), _>("injected native failure")
    });
    assert!(result.is_err());
    assert!(executor.origin.is_none());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_activation_origin(&mut executor, Some(origin), |e| {
            assert_eq!(e.origin, Some(origin));
            panic!("injected unwind");
        });
    }))
    .is_err());
    assert!(executor.origin.is_none());
    // Direct low-level proposal calls have no invented request identity.
    propose_block(
        &mut executor,
        &MockSampling::default(),
        &mut vec![2],
        2,
        1,
        &[2],
        0.7,
        &[],
        None,
        (),
    )
    .unwrap();
    assert_eq!(executor.activations.last().unwrap(), &("proposal", None));
}
