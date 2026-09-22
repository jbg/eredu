use super::*;

fn config() -> eredu_core::TextGenerationConfig {
    eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
}

fn start(
    request: &InferenceRequest,
    id: &InferenceExecutionIdentity,
) -> Result<PrefillDriver<Vec<f64>, NativeCompletion>, WorkingMemoryError> {
    PrefillDriver::new(
        id,
        request,
        request.geometry(),
        GenerationCancellationToken::new(),
    )
}

#[test]
fn preparation_claims_once_before_running_the_shared_prefill_driver() {
    for budgeted in [false, true] {
        let id = InferenceExecutionIdentity::default();
        let g = geometry(3, OutputDemand::LastPosition);
        let pool = if budgeted {
            memory::host_ledger(65536, 0).unwrap()
        } else {
            memory::unlimited_ledger(0)
        };
        let request: InferenceRequest = pool.reserve(&id, &admission(g)).unwrap().into();
        assert!(matches!(
            request.prepare_text(&InferenceExecutionIdentity::default(), g, config()),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        for maximum in [None, Some(2), Some(4)] {
            let mut sampling = config().sampling();
            sampling.max_new_tokens = maximum;
            assert!(matches!(
                request.prepare_text(&id, g, eredu_core::TextGenerationConfig::new(sampling)),
                Err(WorkingMemoryError::PreparationConfigurationMismatch)
            ));
        }
        let preparation = request.prepare_text(&id, g, config()).unwrap();
        let clone = preparation.clone();
        assert!(matches!(
            request.prepare_text(&id, g, config()),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        assert!(
            matches!(
                start(&request, &id),
                Err(WorkingMemoryError::IdentityMismatch)
            ),
            "original retention cannot bypass preparation"
        );
        assert!(matches!(
            start(preparation.request(), &id),
            Err(WorkingMemoryError::PreparationNotReady)
        ));
        let prompt = preparation.claim_prompt().unwrap();
        assert!(
            clone.bind_prompt().is_err(),
            "cannot bind while construction is unresolved"
        );
        assert!(clone.claim_prompt().is_err());
        prompt.finish().unwrap();
        assert!(
            matches!(
                start(preparation.request(), &id),
                Err(WorkingMemoryError::PreparationNotReady)
            ),
            "construction alone does not bind the prompt"
        );
        clone.bind_prompt().unwrap();
        assert!(matches!(
            clone.claim_sampling(config().with_seed(20)),
            Err(WorkingMemoryError::PreparationConfigurationMismatch)
        ));
        preparation
            .claim_sampling(config())
            .unwrap()
            .finish()
            .unwrap();
        assert!(clone.claim_sampling(config()).is_err());
        let mut driver = start(preparation.request(), &id).unwrap();
        assert!(matches!(
            start(clone.request(), &id),
            Err(WorkingMemoryError::AlreadyStarted)
        ));
        let mut executor = RecurrentExecutor::new();
        let mut scores = Vec::new();
        assert_eq!(
            driver
                .run(&mut executor, |_, output| scores
                    .extend(output.into_iter().flatten()))
                .unwrap(),
            PrefillOutcome::Complete
        );
        assert!(scores.iter().any(|score| *score != 0.));
        assert_eq!(executor.submitted.len(), 3);
        drop((driver, executor, preparation, clone, request));
        assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    }
}

#[test]
fn failed_preparation_never_refunds_startup_and_stage_tickets_retain_the_charge() {
    let id = InferenceExecutionIdentity::default();
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = memory::host_ledger(65536, 0).unwrap();
    let request: InferenceRequest = pool.reserve(&id, &admission(g)).unwrap().into();
    let preparation = request.prepare_text(&id, g, config()).unwrap();
    preparation.bind_prompt().unwrap(); // Existing prepared native input.
    let failed = preparation.claim_sampling(config()).unwrap();
    let bytes = pool.funded_used_bytes().unwrap();
    drop(failed);
    assert!(matches!(
        preparation.claim_sampling(config()),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    assert!(matches!(
        start(preparation.request(), &id),
        Err(WorkingMemoryError::PreparationNotReady)
    ));
    assert!(request.prepare_text(&id, g, config()).is_err());
    assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
    drop((preparation, request));
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);

    let request: InferenceRequest = pool.reserve(&id, &admission(g)).unwrap().into();
    let preparation = request.prepare_text(&id, g, config()).unwrap();
    let outstanding = preparation.claim_prompt().unwrap();
    drop((preparation, request));
    assert_eq!(
        pool.funded_used_bytes().unwrap(),
        bytes,
        "move-only stage retains its resource authority"
    );
    drop(outstanding);
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_peak_bytes().unwrap(), bytes);
}

#[test]
fn independent_reservation_wrappers_share_one_concurrent_preparation_claim() {
    let id = InferenceExecutionIdentity::default();
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = memory::host_ledger(65536, 0).unwrap();
    let reservation = pool.reserve(&id, &admission(g)).unwrap();
    let bytes = reservation
        .requirements()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let workers = (0..8)
        .map(|_| {
            let reservation = reservation.clone();
            let barrier = barrier.clone();
            let id = id.clone();
            std::thread::spawn(move || {
                let request = InferenceRequest::from(reservation);
                barrier.wait();
                request.prepare_text(&id, g, config())
            })
        })
        .collect::<Vec<_>>();
    let mut winners = Vec::new();
    for worker in workers {
        match worker.join().unwrap() {
            Ok(preparation) => winners.push(preparation),
            Err(error) => assert_eq!(error, WorkingMemoryError::PreparationAlreadyStarted),
        }
    }
    assert_eq!(winners.len(), 1);
    assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
    let winner = winners.pop().unwrap();
    winner.bind_prompt().unwrap();
    winner.claim_sampling(config()).unwrap().finish().unwrap();
    drop(start(winner.request(), &id).unwrap());
    drop((winner, reservation));
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);
}

#[test]
fn preparation_and_direct_prefill_cannot_both_win_the_same_start() {
    let g = geometry(3, OutputDemand::LastPosition);
    for _ in 0..16 {
        let id = InferenceExecutionIdentity::default();
        let request = unlimited_request(&id, g).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let handles = [false, true].map(|prepare| {
            let request = request.clone();
            let id = id.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                if prepare {
                    request.prepare_text(&id, g, config()).is_ok()
                } else {
                    start(&request, &id).is_ok()
                }
            })
        });
        let winners = handles
            .into_iter()
            .map(|handle| usize::from(handle.join().unwrap()))
            .sum::<usize>();
        assert_eq!(winners, 1);
    }
}
