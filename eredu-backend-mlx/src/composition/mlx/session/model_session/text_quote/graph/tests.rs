use super::super::sequence::fixture::{Probe, tests as fixture};
use super::*;
use crate::backend::submission_recovery::prediction::test_counts::Calls;
use crate::composition::mlx::session::model_session::disk_layerwise_tests as disk;
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, TextGeneration, TextGenerationInput,
    TextPreparationOptions, TokenOutput,
};

// An explicit fixture ceiling, not a selected production default or fit report.
const GRAPH: u64 = 4 << 20;
fn config(total: u64, graph_metadata: u64) -> TextGenerationConfig {
    let config = fixture::config(4, total);
    let mut policy = config.inference_policy();
    policy.graph_metadata_capacity_bytes = NonZeroU64::new(graph_metadata);
    // Original prefill scopes require the same admitted Record arena.
    policy.submission_tracking_capacity_bytes = NonZeroU64::new(1 << 20);
    config.with_inference_policy(policy)
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a T {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return value;
        }
        error = error.source().expect("original typed source");
    }
}

#[test]
fn one_original_arena_follows_real_ordinary_controlled_capture_sequence_and_opening_execution() {
    let stream = fixture::stream();
    for residency in 0..3 {
        let mut reference = None;
        // None baseline, ordinary Q, capture-only, R-only, C+R, C+R+opening.
        for mode in 0..6 {
            if mode == 5 && residency != 0 {
                continue;
            }
            for controlled in [false, true] {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let (mut runtime, _artifact) = fixture::load(&stream, &pool, residency);
                let source = matches!(mode, 2 | 4 | 5).then(|| fixture::source(&runtime));
                let options = TextPreparationOptions {
                    interventions: None, capture: source.clone(),
                };
                let probe = (mode != 0).then(|| Probe::new(&runtime, source.as_ref(), mode == 5));
                let calls = Calls::new();
                let config = if mode == 0 {
                    fixture::config(4, u64::MAX)
                } else {
                    config(u64::MAX, GRAPH)
                };
                let input = TextGenerationInput::TokenIds(vec![2, 5, 7]);
                let outputs = if controlled {
                    let mut run = if mode >= 3 {
                        ControlledTextGeneration::from_input_with_sequence(
                            &mut runtime,
                            input,
                            config,
                            disk::Controller::default(),
                            Some(options),
                            GenerationSequenceRequest::new(4, &[]),
                        )
                        .unwrap()
                    } else {
                        ControlledTextGeneration::from_input_with_options(
                            &mut runtime,
                            input,
                            config,
                            disk::Controller::default(),
                            options,
                        )
                        .unwrap()
                    };
                    let sequence = run
                        .take_prepared_sequence()
                        .map(|s| s.prepare_storage().unwrap());
                    assert_eq!(sequence.is_some(), mode >= 3);
                    let mut outputs = Vec::new();
                    while let Some(token) = run.next() {
                        outputs.push(token.unwrap().into_output());
                        let delivery = run.take_captured_delivery().unwrap();
                        assert_eq!(delivery.is_some(), source.is_some());
                        drop(delivery);
                    }
                    drop((run, sequence));
                    outputs
                } else {
                    let mut run = if mode >= 3 {
                        TextGeneration::from_input_with_sequence(
                            &mut runtime,
                            input,
                            config,
                            TokenFilter::All,
                            Some(options),
                            GenerationSequenceRequest::new(4, &[]),
                        )
                        .unwrap()
                    } else {
                        TextGeneration::from_input_with_options(
                            &mut runtime,
                            input,
                            config,
                            options,
                        )
                        .unwrap()
                    };
                    let sequence = run
                        .take_prepared_sequence()
                        .map(|s| s.prepare_storage().unwrap());
                    assert_eq!(sequence.is_some(), mode >= 3);
                    let mut outputs = Vec::new();
                    while let Some(token) = run.next() {
                        outputs.push(token.unwrap());
                        let delivery = run.take_captured_delivery().unwrap();
                        assert_eq!(delivery.is_some(), source.is_some());
                        drop(delivery);
                    }
                    drop((run, sequence));
                    outputs
                };
                assert_eq!(outputs.len(), 4);
                let values = outputs
                    .iter()
                    .map(|t| t.token_id().unwrap())
                    .collect::<Vec<_>>();
                // Repeated reads reuse the token's one observation; they do not
                // replenish either original role or physical storage allowance.
                assert_eq!(
                    values,
                    outputs
                        .iter()
                        .map(|t| t.token_id().unwrap())
                        .collect::<Vec<_>>()
                );
                let state = runtime.session().payload.model.erased().state_snapshot();
                assert!(state.iter().all(|(position, _)| *position == 6));
                if let Some(expected) = &reference {
                    assert_eq!(&(values, state), expected);
                } else {
                    reference = Some((values, state));
                }
                assert_eq!(calls.original(), if mode == 0 { [0; 5] } else { [4; 5] });
                let retained = probe.as_ref().map(|probe| {
                    let preparation = probe.take();
                    let quote = preparation.quote.as_ref().unwrap();
                    let arena = quote.graph_quota.as_ref().unwrap().clone();
                    assert!(arena.same_arena(quote.graph_quota.as_ref().unwrap()));
                    assert!(arena.occupied_bytes() <= GRAPH as usize);
                    assert_eq!(probe.facts().source, source.is_some());
                    assert_eq!(probe.facts().r > 0, mode >= 3);
                    assert!(quote.original_controls().is_some());
                    assert!(quote.record_quota.is_some());
                    let held = probe.facts().held;
                    assert!(held > GRAPH);
                    drop(preparation);
                    (arena, held)
                });
                drop((outputs, probe, source, calls));
                fixture::finish(runtime, &stream);
                if let Some((arena, held)) = retained {
                    fixture::settle(&pool, held);
                    assert_eq!(
                        arena.occupied_bytes(),
                        0,
                        "all graph owners have retired, arena remains"
                    );
                    drop(arena);
                }
                fixture::settle(&pool, 0);
            }
        }
    }
}

#[test]
fn ordinary_graph_metadata_capacity_is_in_the_first_quote_exact_and_minus_one_before_work() {
    let stream = fixture::stream();
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = fixture::load(&stream, &pool, route);
        let ids = vec![2, 5, 7];
        let input = disk::evidence(&ids);
        let controller = disk::Controller::default();
        let baseline = pool.used_bytes().unwrap();
        let (preparation, quote) =
            super::super::admit(&runtime, &input, config(u64::MAX, GRAPH), &controller).unwrap();
        let required = preparation.request().memory_reservation().unwrap().bytes();
        assert!(required > GRAPH);
        let arena = quote.graph_quota.as_ref().unwrap();
        assert_eq!(
            arena.occupied_bytes(),
            0,
            "cold arena construction has no graph allocation"
        );
        assert!(!quote.has_capture());
        assert!(
            quote.original_controls().is_some(),
            "real source-free control promotion"
        );
        let state = runtime.session().payload.model.erased().state_snapshot();
        drop((preparation, quote));
        fixture::settle(&pool, baseline);
        let short = super::super::admit(
            &runtime,
            &input,
            config(baseline + required - 1, GRAPH),
            &controller,
        )
        .unwrap_err();
        assert!(matches!(
            cause::<WorkingMemoryError>(&short),
            WorkingMemoryError::BudgetExceeded { .. }
        ));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            state
        );
        let (preparation, quote) = super::super::admit(
            &runtime,
            &input,
            config(baseline + required, GRAPH),
            &controller,
        )
        .unwrap();
        assert_eq!(
            preparation.request().memory_reservation().unwrap().bytes(),
            required
        );
        assert!(quote.graph_quota.is_some());
        drop((preparation, quote));
        fixture::finish(runtime, &stream);
        fixture::settle(&pool, 0);
    }
}

#[test]
fn ordinary_exhaustion_preserves_typed_source_and_original_arena_without_refill() {
    let stream = fixture::stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    // The minimum valid actual object layout can be insufficient for its graph.
    // This is a fixture search over cold layouts, never a production fit policy.
    let capacity = (1..=GRAPH)
        .find(|n| {
            PreparedSubmissionGraphQuota::<OriginalGraphMetadata>::layout(*n as usize).is_ok()
        })
        .unwrap();
    let result = TextGeneration::new(&mut runtime, vec![2, 5, 7], config(u64::MAX, capacity));
    let error = match result {
        Err(error) => error,
        Ok(mut run) => {
            let error = run
                .next()
                .unwrap()
                .err()
                .expect("the accepted minimum cannot fit the actual graph");
            drop(run);
            error
        }
    };
    assert_eq!(
        *cause::<safemlx::error::GraphMetadataFailure>(&error),
        safemlx::error::GraphMetadataFailure::Exhausted
    );
    let preparation = probe.take();
    let quote = preparation.quote.as_ref().unwrap();
    let arena = quote.graph_quota.as_ref().unwrap().clone();
    assert!(arena.same_arena(quote.graph_quota.as_ref().unwrap()));
    assert!(arena.occupied_bytes() <= capacity as usize);
    // Refusal never changes the admitted limit, original bank or quote owner.
    assert_eq!(
        quote
            .config()
            .inference_policy()
            .graph_metadata_capacity_bytes
            .unwrap()
            .get(),
        capacity
    );
    let held = probe.facts().held;
    drop((error, preparation, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, held);
    assert_eq!(arena.occupied_bytes(), 0);
    drop(arena);
    fixture::settle(&pool, 0);
}

#[test]
fn original_graph_metadata_prompt_busy_preserves_the_same_input_work_node_and_arena() {
    use super::super::sequence::fixture::Mode;
    use eredu_core::TextGenerationBackend;
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };
    let stream = fixture::stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::Defer);
    let rejected = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        config(u64::MAX, GRAPH),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    );
    assert!(
        rejected.is_err(),
        "only actual sequence extraction is deferred"
    );
    drop(rejected);
    let preparation = probe.take();
    let quote = preparation.quote.as_ref().unwrap();
    let arena = quote.graph_quota.as_ref().unwrap().clone();
    let scopes = quote.preparation_scopes.as_ref().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(released) = safemlx::try_with_submission_retirement(|| {
                let _ = ready_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
            }) {
                return released;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    });
    struct Release(
        Option<mpsc::Sender<()>>,
        Option<std::thread::JoinHandle<bool>>,
    );
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
            if let Some(worker) = self.1.take() {
                let _ = worker.join();
            }
        }
    }
    let mut release = Release(Some(release_tx), Some(worker));
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let ids = vec![2, 5, 7];
    let pointer = ids.as_ptr() as usize;
    let error =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), ids, &preparation).unwrap_err();
    assert!(matches!(
        error,
        Error::PreparationScope(safemlx::SubmissionScopeOwnerCause::RuntimeBusy)
    ));
    let identities = scopes.pending_prompt_identities().unwrap();
    assert_eq!(identities.0, pointer);
    let used = pool.used_bytes().unwrap();
    for _ in 0..3 {
        let error = MlxBackend::prepare_text_prompt_admitted(
            runtime.backend(),
            vec![2, 5, 7],
            &preparation,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::PreparationScope(safemlx::SubmissionScopeOwnerCause::RuntimeBusy)
        ));
        assert_eq!(scopes.pending_prompt_identities(), Some(identities));
        assert!(arena.same_arena(quote.graph_quota.as_ref().unwrap()));
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert_eq!(arena.occupied_bytes(), 0, "Busy has constructed no graph");
    }
    release.0.take().unwrap().send(()).unwrap();
    assert!(release.1.take().unwrap().join().unwrap());
    let prompt =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), vec![2, 5, 7], &preparation)
            .unwrap();
    let repeated =
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), vec![2, 5, 7], &preparation)
            .unwrap_err();
    assert!(matches!(repeated, Error::PreparationScopeUnavailable));
    let held = probe.facts().held;
    drop((prompt, preparation, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, held);
    drop(arena);
    fixture::settle(&pool, 0);
}

#[test]
fn graph_refusal_keeps_native_source_across_portable_tensor_projection() {
    use eredu_nn::Tensor;
    use std::time::{Duration, Instant};
    let stream = fixture::stream();
    // Mechanism-only projection test; actual original-account coverage is above.
    let mut prepared = PreparedSubmissionGraphQuota::try_new(128, ()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let arena = loop {
        match prepared.try_allocate() {
            Ok(arena) => break arena,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    let mut prepared = safemlx::PreparedSubmissionScopeOwner::try_new(())
        .unwrap()
        .with_graph_quota(arena.clone());
    let mut scope = loop {
        match safemlx::SubmissionScope::try_begin_retaining(prepared) {
            Ok(scope) => break scope,
            Err(error) => {
                assert_eq!(
                    error.cause(),
                    safemlx::SubmissionScopeOwnerCause::RuntimeBusy
                );
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    let error = crate::MlxTensor::full_f32(3.25, &[2, 3], &stream).unwrap_err();
    assert_eq!(
        cause::<safemlx::error::GraphMetadataFailure>(&error),
        &safemlx::error::GraphMetadataFailure::Exhausted
    );
    scope.seal();
    assert_eq!(arena.occupied_bytes(), 0);
    drop((scope, arena));
    safemlx::reclaim_allocation_owners();
}
