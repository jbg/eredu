use super::super::sequence::fixture::tests as fixture;
use super::super::sequence::fixture::{Mode, Probe};
use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{
    GenerationSequenceRequest, TextGeneration, TextGenerationBackend, TextGenerationInput,
    TextPreparationOptions, TokenFilter,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

pub(in crate::composition::mlx::session::model_session::text_quote) fn with_foreign_runtime(
    f: impl FnOnce(),
) {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(held) = safemlx::try_with_submission_retirement(|| {
                let _ = ready_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
            }) {
                return held;
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
            if let Some(t) = self.1.take() {
                let _ = t.join();
            }
        }
    }
    let mut release = Release(Some(release_tx), Some(worker));
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    f();
    release.0.take().unwrap().send(()).unwrap();
    assert!(
        release.1.take().unwrap().join().unwrap(),
        "callback waited for the foreign runtime loan"
    );
}
fn admitted(
    runtime: &mut ModelRuntime<MlxBackend<'static>>,
    capture: Option<&eredu_core::capture::SharedCapturePlan>,
    temperature: f32,
) -> (Probe, MlxTextPreparation) {
    let probe = Probe::new(runtime, capture, false);
    probe.mode(Mode::Defer);
    let base = fixture::config(4, u64::MAX);
    let mut sampling = base.sampling();
    sampling.temperature = temperature;
    let config = TextGenerationConfig::new(sampling)
        .with_seed(19)
        .with_inference_policy(base.inference_policy().clone());
    let options = capture.map(|source| TextPreparationOptions {
        interventions: None,
        capture: Some(source.clone()),
    });
    let result = TextGeneration::from_input_with_sequence(
        runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        config,
        TokenFilter::All,
        options,
        GenerationSequenceRequest::new(4, &[]),
    );
    assert!(
        result.is_err(),
        "only the genuine sequence extraction was deferred"
    );
    drop(result);
    let preparation = probe.take();
    (probe, preparation)
}
fn prompt_state(scopes: &PreparationScopes) -> (usize, usize, usize) {
    let loan = scopes.prompt.borrow();
    let Slot::Pending(p) = &*loan else {
        panic!("same pending prompt")
    };
    let Some(NativePreparation::Prepared(native)) = &p.native else {
        panic!("same allocated native preparation")
    };
    (
        p.ids.tokens().as_ptr() as usize,
        p.work.identity(),
        native.allocation_identity(),
    )
}
fn sampling_state(scopes: &PreparationScopes) -> (usize, usize, usize) {
    let loan = scopes.sampling.borrow();
    let Slot::Pending(p) = &*loan else {
        panic!("same pending sampler")
    };
    let Some(NativePreparation::Prepared(native)) = &p.native else {
        panic!("same allocated native preparation")
    };
    (
        p.work.identity(),
        native.allocation_identity(),
        match p.sampler.as_ref().unwrap().as_sampler() {
            eredu_runtime::generation::ConfiguredTextSampler::Standard(sampler) => {
                sampler.generated_tokens().len()
            }
            eredu_runtime::generation::ConfiguredTextSampler::MirostatV2(sampler) => {
                sampler.generated_tokens().len()
            }
        },
    )
}
fn busy(error: Error) {
    assert!(matches!(
        error,
        Error::PreparationScope(SubmissionScopeOwnerCause::RuntimeBusy)
    ));
}

#[test]
fn original_pair_retries_the_same_input_sampler_work_and_node_then_spends_once() {
    let stream = fixture::stream();
    for capture in [false, true] {
        for temperature in [0.0, 0.65] {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
            let source = capture.then(|| fixture::source(&runtime));
            let (probe, preparation) = admitted(&mut runtime, source.as_ref(), temperature);
            let quote = preparation.quote.as_ref().unwrap();
            let scopes = quote.preparation_scopes.as_ref().unwrap();
            let ids = vec![2, 5, 7];
            let first_ptr = ids.as_ptr() as usize;
            with_foreign_runtime(|| {
                busy(
                    MlxBackend::prepare_text_prompt_admitted(runtime.backend(), ids, &preparation)
                        .unwrap_err(),
                );
                let same = prompt_state(scopes);
                assert_eq!(same.0, first_ptr);
                let used = pool.fixture_host_charge().unwrap();
                for _ in 0..3 {
                    busy(
                        MlxBackend::prepare_text_prompt_admitted(
                            runtime.backend(),
                            vec![2, 5, 7],
                            &preparation,
                        )
                        .unwrap_err(),
                    );
                    assert_eq!(prompt_state(scopes), same);
                    assert_eq!(pool.fixture_host_charge().unwrap(), used);
                }
                assert!(matches!(
                    MlxBackend::prepare_text_prompt_admitted(
                        runtime.backend(),
                        vec![2, 5, 6],
                        &preparation
                    ),
                    Err(Error::PreparationScopeInputMismatch)
                ));
                assert_eq!(prompt_state(scopes), same);
            });
            let prompt = MlxBackend::prepare_text_prompt_admitted(
                runtime.backend(),
                vec![2, 5, 7],
                &preparation,
            )
            .unwrap();
            assert_eq!(
                prompt.parts[0]
                    .payload()
                    .value()
                    .evaluated()
                    .unwrap()
                    .as_slice::<u32>(),
                &[2, 5, 7]
            );
            assert!(matches!(
                MlxBackend::prepare_text_prompt_admitted(
                    runtime.backend(),
                    vec![2, 5, 7],
                    &preparation
                ),
                Err(Error::PreparationScopeUnavailable)
            ));
            let prompt =
                MlxBackend::bind_text_prompt_preparation(runtime.backend(), prompt, &preparation)
                    .unwrap();
            with_foreign_runtime(|| {
                busy(
                    MlxBackend::start_text_generation_admitted(
                        runtime.backend(),
                        quote.config(),
                        &preparation,
                    )
                    .err()
                    .unwrap(),
                );
                let same = sampling_state(scopes);
                let used = pool.fixture_host_charge().unwrap();
                for _ in 0..3 {
                    busy(
                        MlxBackend::start_text_generation_admitted(
                            runtime.backend(),
                            quote.config(),
                            &preparation,
                        )
                        .err()
                        .unwrap(),
                    );
                    assert_eq!(sampling_state(scopes), same);
                    assert_eq!(pool.fixture_host_charge().unwrap(), used);
                }
            });
            let state = MlxBackend::start_text_generation_admitted(
                runtime.backend(),
                quote.config(),
                &preparation,
            )
            .unwrap();
            assert_eq!(state.sampling.prng.is_some(), temperature != 0.0);
            assert!(matches!(*scopes.prompt.borrow(), Slot::Spent));
            assert!(matches!(*scopes.sampling.borrow(), Slot::Spent));
            // Direct role entry keeps the spent check independent of the outer
            // funding-run transfer that also rejects subsequent public setup.
            assert!(matches!(
                scopes.sampling(quote, quote.config(), preparation.request.as_ref().unwrap()),
                Err(Error::PreparationScopeUnavailable)
            ));
            drop((prompt, state, preparation, probe, source));
            fixture::finish(runtime, &stream);
            fixture::settle(&pool, 0);
        }
    }
}

#[test]
fn original_scope_capsules_outlive_quiescence_probe_drop_and_native_child_retirement() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let (probe, preparation) = admitted(&mut runtime, None, 0.0);
    let held = probe.facts().held;
    let quote = preparation.quote.as_ref().unwrap();
    let scopes = quote.preparation_scopes.as_ref().unwrap();
    let p = preparation.request.as_ref().unwrap();
    let (prompt_stage, prompt_custody) = scopes.bank.borrow_mut().claim_prompt(p).unwrap();
    let (sampling_stage, sampling_custody) = scopes
        .bank
        .borrow_mut()
        .claim_sampling(p, quote.config())
        .unwrap();
    fn begin(c: OriginalPreparationScopeCustody) -> safemlx::SubmissionScope {
        let mut pending = safemlx::PreparedSubmissionScopeOwner::try_new(c).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match safemlx::SubmissionScope::try_begin_retaining(pending) {
                Ok(scope) => return scope,
                Err(e) => {
                    assert_eq!(e.cause(), SubmissionScopeOwnerCause::RuntimeBusy);
                    pending = e.into_parts().1;
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
        }
    }
    let mut parent = begin(prompt_custody);
    let mut child = begin(sampling_custody);
    parent.seal();
    child.seal();
    assert!(parent.status().is_settled());
    assert!(child.status().is_settled());
    // No Work was created by this allocation-lifetime fixture. The actual
    // original bank/stages supply both capsules, with no native certificate.
    drop((prompt_stage, sampling_stage, preparation, probe));
    fixture::finish(runtime, &stream);
    // Retiring model owners may span several guarded queue batches. Drain to
    // the exact original capsule charge before testing parent/child deletion.
    fixture::settle(&pool, held);
    assert_eq!(pool.fixture_host_charge().unwrap(), held);
    drop(parent);
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        held,
        "live child retains its native parent allocation"
    );
    drop(child);
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        held,
        "post-delete owner nodes still await ordinary unlocked reclamation"
    );
    let copy = pool.clone();
    std::thread::spawn(move || {
        safemlx::reclaim_allocation_owners();
        assert_eq!(copy.fixture_host_charge().unwrap(), 0);
    })
    .join()
    .unwrap();
}

#[test]
fn same_original_pair_rejects_reentry_and_unwind_cannot_refill_the_role() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let (probe, preparation) = admitted(&mut runtime, None, 0.0);
    let quote = preparation.quote.as_ref().unwrap();
    let scopes = quote.preparation_scopes.as_ref().unwrap();
    let marker = Arc::new(());
    let expected = marker.clone();
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let (checkout, pending) = Checkout::enter(&scopes.prompt).unwrap();
        assert!(pending.is_none());
        assert!(matches!(
            MlxBackend::prepare_text_prompt_admitted(
                runtime.backend(),
                vec![2, 5, 7],
                &preparation
            ),
            Err(Error::PreparationScopeReentrant)
        ));
        let _checkout = checkout;
        std::panic::panic_any(marker);
    }))
    .unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &expected
    ));
    assert!(matches!(
        MlxBackend::prepare_text_prompt_admitted(runtime.backend(), vec![2, 5, 7], &preparation),
        Err(Error::PreparationScopeUnavailable)
    ));
    assert!(matches!(*scopes.sampling.borrow(), Slot::Unclaimed));
    drop((preparation, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, 0);
}

mod unused_work;
