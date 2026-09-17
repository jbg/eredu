use super::*;
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::PreparedTextComponentsCopy;

#[test]
fn saved_frontier_survives_scalar_sampler_and_paired_copy_after_source_retirement() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    // Loading still precedes every finite account in this actual fixture.
    let (mut destination, destination_artifact) = runtime(&pool);
    let (mut source, source_artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut source);
    let mut state = start(&mut driver, u64::MAX);
    let opening = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let source = TextArraySource::inspect(runtime, &generation.sampling, None).unwrap();
        assert_eq!((source.next_prediction, source.frontier), (0, 0));
        PreparedTextArrayCopy::prepare(runtime, &generation.sampling, None)
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    assert_eq!(
        (opening.source.next_prediction, opening.source.frontier),
        (0, 0)
    );
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (arrays, sampling, pair) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let live = &generation.sampling;
        let quote = live.quote.as_ref().unwrap();
        assert_eq!(live.next_prediction, 2);
        assert_eq!(quote.local_prediction(live.next_prediction).unwrap(), 2);
        assert_eq!(quote.prediction_frontier(live.next_prediction).unwrap(), 6);
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .admission()
                .unwrap()
                .position(),
            6
        );
        let before = source_state(live);
        assert_eq!(before.1.len(), 2);
        assert!(before.1.iter().any(|token| *token != 0));
        let arrays = PreparedTextArrayCopy::prepare(runtime, live, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap();
        let sampling = PreparedTextArrayCopy::prepare(runtime, live, outputs.last())
            .unwrap()
            .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap();
        let pair = PreparedTextComponentsCopy::prepare(runtime, live, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap();
        assert_eq!(source_state(live), before);
        (arrays, sampling, pair)
    };
    assert_eq!(
        (arrays.source.next_prediction, arrays.source.frontier),
        (2, 6)
    );
    assert_eq!((sampling.next_prediction(), sampling.frontier()), (2, 6));
    assert_eq!(
        (
            pair.sampling().next_prediction(),
            pair.sampling().frontier()
        ),
        (2, 6)
    );
    let saved_key = words(sampling.arrays.key.as_ref().unwrap());
    let saved_token = words(sampling.arrays.pending.as_ref().unwrap());
    let third = advance(&mut driver, &mut state);
    {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let current =
            TextArraySource::inspect(runtime, &generation.sampling, Some(&third)).unwrap();
        assert_eq!((current.next_prediction, current.frontier), (3, 7));
        assert_eq!((sampling.next_prediction(), sampling.frontier()), (2, 6));
    }
    drop((opening, third, outputs, state));
    drop(driver);
    // The status probe neither retains the payload nor affects exclusive access.
    // Its observation is active-alias retirement, not semantic destruction.
    let retired = source.session().test_payload_retirement_probe();
    drop((source, source_artifact));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        retired()
    });
    // Saved duplication validates its own custody, not the retired live branch.
    let sampling_copy = PreparedTextArrayCopy::prepare_saved(&destination, &sampling)
        .unwrap()
        .copy_sampling(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    let pair_copy = PreparedTextComponentsCopy::prepare_saved(&destination, &pair)
        .unwrap()
        .copy(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_eq!(
        (sampling_copy.next_prediction(), sampling_copy.frontier()),
        (2, 6)
    );
    assert_eq!(
        (
            pair_copy.sampling().next_prediction(),
            pair_copy.sampling().frontier()
        ),
        (2, 6)
    );
    assert_eq!(words(sampling_copy.arrays.key.as_ref().unwrap()), saved_key);
    assert_eq!(
        words(sampling_copy.arrays.pending.as_ref().unwrap()),
        saved_token
    );
    assert_ne!(
        identity(sampling_copy.arrays.key.as_ref().unwrap()),
        identity(sampling.arrays.key.as_ref().unwrap())
    );
    assert_eq!(
        format!("{:?}", pair_copy.sampling().sampler.as_sampler()),
        format!("{:?}", pair.sampling().sampler.as_sampler())
    );
    drop((
        arrays,
        sampling,
        pair,
        sampling_copy,
        pair_copy,
        destination,
        destination_artifact,
    ));
    settle(&pool, 0);
}

#[test]
fn substituted_live_capture_coordinates_reject_without_copy_or_accounting_change() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, u64::MAX);
    let output = advance(&mut driver, &mut state);
    {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let live = &generation.sampling;
        let source = TextArraySource::inspect(runtime, live, Some(&output)).unwrap();
        assert_eq!((source.next_prediction, source.frontier), (1, 5));
        let before = (
            accounting(&pool),
            paths::snapshot(),
            copies(),
            source_state(live),
        );
        let mut changed_frontier = source.clone();
        changed_frontier.frontier += 1;
        let error = changed_frontier
            .validate(runtime, live, Some(&output))
            .unwrap_err();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        let mut changed_prediction = source.clone();
        changed_prediction.next_prediction += 1;
        let error = changed_prediction
            .validate(runtime, live, Some(&output))
            .unwrap_err();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        source.validate(runtime, live, Some(&output)).unwrap();
        assert_eq!(
            (
                accounting(&pool),
                paths::snapshot(),
                copies(),
                source_state(live)
            ),
            before
        );
    }
    drop((output, state));
    drop(driver);
    drop((runtime, artifact));
    settle(&pool, 0);
}
