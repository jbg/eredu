use super::sequence::fixture::{Probe, tests as fixture};
use super::*;
use crate::backend::submission_recovery::prefill::{
    PrefillBankOwner,
    test_trace::{Event, Trace},
};
use crate::composition::mlx::session::model_session::disk_layerwise_tests as disk;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, PendingTextInput, TextGeneration,
    TextGenerationInput, TokenFilter, TokenOutput,
};
use eredu_runtime::prefill::{PrefillControlPlan, PrefillControlRole};
use std::num::NonZeroU64;
#[path = "prefill_tests/activation.rs"]
mod activation;
#[path = "prefill_tests/cpu_paged.rs"]
mod cpu_paged;
#[path = "prefill_tests/persistent_disk.rs"]
mod persistent_disk;
#[path = "prefill_tests/prompt_cache.rs"]
mod prompt_cache;

fn config(original: bool) -> TextGenerationConfig {
    let mut config = fixture::config(4, u64::MAX);
    if !original {
        config = TextGenerationConfig::new(config.sampling()).with_seed(19);
    }
    if original {
        fixture::chunked_original(config)
    } else {
        let mut policy = config.inference_policy().clone();
        policy.prefill_chunk_positions = NonZeroU64::new(2);
        config.with_inference_policy(policy.clone())
    }
}
fn original_source_config() -> TextGenerationConfig {
    // The borrowed token source supplies the actual input producer. Derive
    // both native arenas from that complete source rather than old test caps.
    let config = config(true);
    let mut policy = config.inference_policy().clone();
    policy.submission_tracking_capacity_bytes = None;
    policy.graph_metadata_capacity_bytes = None;
    config.with_inference_policy(policy.clone())
}
#[test]
fn c1_actual_original_roles_preserve_ordinary_controlled_nonzero_three_residency_results() {
    if !crate::tests::support::native_process::enter("physical-limit-parity") {
        return;
    }
    let environment = fixture::PreparedResidencyFixture::new();
    let stream = environment.stream();
    let pool = &environment.pool;
    for residency in 0..3 {
        let mut expected = None;
        for finite in [false, true] {
            for controlled in [false, true] {
                let (mut runtime, _artifact, baseline) = environment.load(residency, None);
                let disk_reads = environment.disk_read_bytes(&runtime, residency);
                let probe = Some(Probe::new(&runtime, None, false));
                let trace = Trace::new();
                let input = vec![2, 5, 7, 3, 11];
                let mut request_config = config(finite);
                {
                    let mut policy = request_config.inference_policy().clone();
                    policy.submission_tracking_capacity_bytes = None;
                    policy.graph_metadata_capacity_bytes = None;
                    request_config = request_config.with_inference_policy(policy.clone());
                }
                let outputs = if controlled {
                    let mut generation = ControlledTextGeneration::from_token_ids_with_sequence(
                        &mut runtime,
                        eredu_core::TokenIdsInputPlan::new(&input).unwrap(),
                        request_config,
                        disk::Controller::default(),
                        None,
                        GenerationSequenceRequest::new(4, &[]),
                    )
                    .unwrap();
                    let mut sequence = Some({
                        generation
                            .take_prepared_sequence()
                            .unwrap()
                            .prepare_storage()
                            .unwrap()
                    });
                    let mut outputs = Vec::new();
                    for next in &mut generation {
                        let token = next.unwrap_or_else(|error| panic!(
                            "residency={residency}, finite={finite}, controlled={controlled}: {error:?}"));
                        if let Some(sequence) = &mut sequence {
                            sequence
                                .commit(
                                    token.token_id(),
                                    eredu_core::TokenTerminalSignals::default(),
                                )
                                .unwrap();
                        }
                        outputs.push(token.into_output());
                    }
                    if let Some(sequence) = sequence {
                        assert_eq!(sequence.tokens().len(), 4);
                    }
                    outputs
                } else {
                    let mut generation = TextGeneration::from_token_ids_with_sequence(
                        &mut runtime,
                        eredu_core::TokenIdsInputPlan::new(&input).unwrap(),
                        request_config,
                        TokenFilter::All,
                        None,
                        GenerationSequenceRequest::new(4, &[]),
                    )
                    .unwrap();
                    let mut sequence = Some({
                        generation
                            .take_prepared_sequence()
                            .unwrap()
                            .prepare_storage()
                            .unwrap()
                    });
                    let mut outputs = Vec::new();
                    for next in &mut generation {
                        let token = next.unwrap_or_else(|error| panic!(
                            "residency={residency}, finite={finite}, controlled={controlled}: {error:?}"));
                        if let Some(sequence) = &mut sequence {
                            sequence
                                .commit(
                                    token.token_id().unwrap(),
                                    eredu_core::TokenTerminalSignals::default(),
                                )
                                .unwrap();
                        }
                        outputs.push(token);
                    }
                    if let Some(sequence) = sequence {
                        assert_eq!(sequence.tokens().len(), 4);
                    }
                    outputs
                };
                assert_eq!(outputs.len(), 4);
                let tokens = outputs
                    .iter()
                    .map(|t| t.token_id().unwrap())
                    .collect::<Vec<_>>();
                let numeric = runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .resident_reset_source()
                    .unwrap()
                    .state()
                    .retained_arrays()
                    .into_iter()
                    .map(|a| {
                        (
                            a.shape().to_vec(),
                            a.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                assert!(!numeric.is_empty());
                assert!(numeric.iter().flat_map(|(_, v)| v).all(|x| x.is_finite()));
                assert!(numeric.iter().flat_map(|(_, v)| v).any(|x| *x != 0.0));
                let state = runtime.session().payload.model.erased().state_snapshot();
                assert!(state.iter().all(|(position, _)| *position == 8));
                if let Some(reference) = &expected {
                    assert_eq!(&(tokens, numeric), reference);
                } else {
                    expected = Some((tokens, numeric));
                }
                let events = trace.events();
                environment.assert_disk_read_progress(&runtime, residency, disk_reads);
                let prepared = events
                    .iter()
                    .filter_map(|e| match e {
                        Event::Prepared { plan, roots, graph } => Some((*plan, *roots, *graph)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let completed = events
                    .iter()
                    .filter_map(|e| match e {
                        Event::Completed { roots, original } => Some((*roots, *original)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let on_stream = events
                    .iter()
                    .filter_map(|event| match event {
                        Event::ModelCompleted { roots } => Some(*roots),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                // Finite and unlimited limits use the same prepared scopes.
                assert_eq!(
                    completed.len(),
                    0,
                    "residency={residency}, finite={finite}, controlled={controlled}: {events:?}"
                );
                assert_eq!(
                    on_stream.len(),
                    6,
                    "residency={residency}, finite={finite}, controlled={controlled}: {events:?}"
                );
                assert!(completed.iter().all(|(roots, _)| *roots > 0));
                assert!(on_stream.iter().all(|roots| *roots > 0));
                {
                    assert_eq!(prepared.len(), 1);
                    let (plan, roots, graph) = prepared[0];
                    assert_eq!(plan.span_count(), 3);
                    assert_eq!(plan.scope_count(), 17);
                    assert!(roots > 0);
                    assert!(graph > 0);
                    assert!(matches!(events.first(), Some(Event::Prepared { .. })));
                    let actual = events
                        .iter()
                        .filter_map(|e| match e {
                            Event::Started(role) => Some(*role),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual,
                        (0..plan.scope_count())
                            .map(|n| plan.role(n).unwrap())
                            .collect::<Vec<_>>()
                    );
                    assert!(completed.iter().all(|(_, original)| *original));
                }
                if let Some(probe) = &probe {
                    let preparation = probe.take();
                    assert!(preparation.quote.as_ref().unwrap().record_quota.is_some());
                }
                drop((outputs, probe, trace));
                fixture::finish(runtime, stream);
                fixture::settle(pool, baseline);
            }
        }
    }
}
#[test]
fn c1_actual_prepared_bank_rejects_foreign_request_role_and_reissue_without_advancing() {
    let environment = fixture::PreparedResidencyFixture::new();
    let pool = &environment.pool;
    let (mut runtime, _artifact, first_source_baseline) = environment.load(0, None);
    let stream = runtime.backend().stream().clone();
    // Construct both model identities before the original request reserves work.
    let before_foreign_source = pool.fixture_host_charge().unwrap();
    let (foreign_runtime, _other_artifact, foreign_target_bytes) = environment.load(0, None);
    let foreign_stream = foreign_runtime.backend().stream().clone();
    let source_baseline = first_source_baseline
        .checked_add(
            foreign_target_bytes
                .checked_sub(before_foreign_source)
                .unwrap(),
        )
        .unwrap();
    runtime.backend().validate_original_stream_owners().unwrap();
    let native_runtime = runtime
        .session()
        .payload
        .model
        .erased()
        .prefill_roots_runtime()
        .unwrap();
    let probe = Probe::new(&runtime, None, false);
    let run = TextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        eredu_core::TokenIdsInputPlan::new(&[2, 5, 7, 3, 11]).unwrap(),
        original_source_config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .unwrap();
    let preparation = probe.take();
    let held = probe.facts().held;
    let prepared = preparation.request.as_ref().unwrap();
    let step = prepared
        .claim_step(&probe.context(), PendingTextInput::Prefill(()))
        .unwrap();
    let quote = preparation.quote.as_ref().unwrap();
    let mut original = quote.prefill_scopes.as_ref().unwrap().borrow_mut();
    let set = original.claim(&step).unwrap();
    assert!(original.claim(&step).is_err());
    drop(original);
    let plan = set.facts().plan();
    let (owner, view) = PrefillBankOwner::new(
        set,
        step.request(),
        &native_runtime,
        quote.record_quota.as_ref(),
        quote.graph_quota.as_ref().unwrap(),
        quote.original_controls().unwrap(),
    )
    .unwrap();
    assert!(
        view.begin(step.request(), PrefillControlRole::FinalIndex)
            .is_err()
    );
    let first = plan.role(0).unwrap();
    let foreign = crate::memory_fixture::empty_admitted_request(
        &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
        plan.geometry(),
    )
    .unwrap();
    assert!(view.begin(&foreign, first).is_err());
    with_foreign_runtime(|| {
        assert!(matches!(
            view.begin(step.request(), first),
            Err(Error::PrefillScope(
                safemlx::SubmissionScopeOwnerCause::RuntimeBusy
            ))
        ));
    });
    let (mut guard, roots) = view.begin(step.request(), first).unwrap();
    assert!(roots.is_none());
    assert!(view.begin(step.request(), first).is_err());
    guard.seal();
    assert!(guard.finish().unwrap().settled);
    let (mut next, roots) = view.begin(step.request(), plan.role(1).unwrap()).unwrap();
    assert!(roots.is_none());
    next.seal();
    assert!(next.finish().unwrap().settled);
    drop(run); // releases the core mutable runtime loan; original bank remains live
    let executable = runtime.session().payload.model.erased();
    assert_eq!(
        executable.prefill_status_for_test().unwrap(),
        (false, false)
    );
    executable
        .install_prefill_controls(Some(
            crate::backend::submission_recovery::prefill::PrefillControlProjection::Prefill(
                view.clone(),
            ),
        ))
        .unwrap();
    assert_eq!(executable.prefill_status_for_test().unwrap(), (true, true));
    executable.retire_expired_opening_rows().unwrap();
    assert_eq!(executable.prefill_status_for_test().unwrap(), (true, true));
    let error = foreign_runtime
        .session()
        .payload
        .model
        .erased()
        .install_prefill_controls(Some(
            crate::backend::submission_recovery::prefill::PrefillControlProjection::Prefill(
                view.clone(),
            ),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .is::<WorkingMemoryError>()
    );
    assert!(view.is_live());
    assert_eq!(executable.prefill_status_for_test().unwrap(), (true, true));
    fixture::finish(foreign_runtime, &foreign_stream);
    assert!(pool.fixture_host_charge().unwrap() >= held);
    runtime.synchronize().unwrap();
    assert!(view.is_live());
    assert_eq!(executable.prefill_status_for_test().unwrap(), (true, true));
    drop((step, preparation, probe));
    // The first ordinary snapshot retires only the outer node. The bank stays
    // live through the initial idle/expired-view check, then the fixed-point
    // cleanup destroys its last strong owner. Synchronize must revisit the
    // installed weak after that destruction; the external view remains paid.
    use crate::backend::ordinary_retirement::OrdinaryRetirement;
    drop(OrdinaryRetirement::new(OrdinaryRetirement::new(owner)));
    assert!(view.is_live());
    assert_eq!(executable.prefill_status_for_test().unwrap(), (true, true));
    runtime.synchronize().unwrap();
    assert!(!view.is_live());
    assert_eq!(
        executable.prefill_status_for_test().unwrap(),
        (false, false)
    );
    fixture::finish(runtime, &stream);
    // The escaped view retains Q and its actual parent planning/source accounts.
    // The pre-request factory source baseline remains independently owned.
    assert!(pool.fixture_host_charge().unwrap() >= source_baseline.checked_add(held).unwrap());
    drop(view);
    fixture::settle(pool, source_baseline);
}
#[test]
fn c1_smaller_chunks_recompute_all_collector_occupancy_and_keep_unknown_producers_explicit() {
    let g = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 4,
        prefill_chunk_positions: 5,
        output: OutputDemand::LastPosition,
    };
    let mut prior = 0;
    for chunk in [5, 4, 3, 2, 1] {
        let facts = crate::backend::submission_recovery::prefill::facts(
            InferenceGeometry {
                prefill_chunk_positions: chunk,
                ..g
            },
            19,
        )
        .unwrap();
        let plan = PrefillControlPlan::new(
            InferenceGeometry {
                prefill_chunk_positions: chunk,
                ..g
            },
            true,
        )
        .unwrap();
        assert_eq!(facts.plan(), plan);
        assert!(facts.graph_bytes().unwrap() >= prior);
        prior = facts.graph_bytes().unwrap();
        assert_eq!(facts.graph_residual_ceiling(prior + 1).unwrap(), 1);
        assert!(facts.graph_residual_ceiling(prior - 1).is_err());
        assert!(facts.total_bytes().unwrap().unwrap() > prior);
    }
}

fn with_foreign_runtime(f: impl FnOnce()) {
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(ok) = safemlx::try_with_submission_retirement(|| {
                ready_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
            }) {
                return ok;
            }
            assert!(Instant::now() < end);
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
    f();
    release.0.take().unwrap().send(()).unwrap();
    assert!(release.1.take().unwrap().join().unwrap());
}

#[test]
fn c3_original_bank_missing_record_refuses_before_role_or_carrier_construction() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let native_runtime = runtime
        .session()
        .payload
        .model
        .erased()
        .prefill_roots_runtime()
        .unwrap();
    let probe = Probe::new(&runtime, None, false);
    let run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7, 3, 11]),
        config(true),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .unwrap();
    let preparation = probe.take();
    let prepared = preparation.request.as_ref().unwrap();
    let step = prepared
        .claim_step(&probe.context(), PendingTextInput::Prefill(()))
        .unwrap();
    let quote = preparation.quote.as_ref().unwrap();
    let set = quote
        .prefill_scopes
        .as_ref()
        .unwrap()
        .borrow_mut()
        .claim(&step)
        .unwrap();
    let trace = Trace::new();
    let before = crate::tests::support::path_instrumentation::snapshot();
    let result = PrefillBankOwner::new(
        set,
        step.request(),
        &native_runtime,
        None,
        quote.graph_quota.as_ref().unwrap(),
        quote.original_controls().unwrap(),
    );
    assert!(matches!(
        result,
        Err(Error::OriginalNativeControl(
            safemlx::OriginalNativeControlError::MissingRecord
        ))
    ));
    assert!(trace.events().is_empty());
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        before
    );
    assert!(
        quote
            .prefill_scopes
            .as_ref()
            .unwrap()
            .borrow_mut()
            .claim(&step)
            .is_err()
    );
    drop((result, trace, step, preparation, run, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, 0);
}

#[test]
fn c4_genuine_original_source_span_and_final_index_roles_evaluate_with_retained_busy_failure() {
    use safemlx::error::ScopedEvaluationCause;
    let stream = fixture::stream();
    for residency in 0..3 {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = fixture::load(&stream, &pool, residency);
        let native_runtime = runtime
            .session()
            .payload
            .model
            .erased()
            .prefill_roots_runtime()
            .unwrap();
        let probe = Probe::new(&runtime, None, false);
        let run = TextGeneration::from_input_with_sequence(
            &mut runtime,
            TextGenerationInput::TokenIds(vec![2, 5, 7, 3, 11]),
            config(true),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap();
        let preparation = probe.take();
        let prepared = preparation.request.as_ref().unwrap();
        let step = prepared
            .claim_step(&probe.context(), PendingTextInput::Prefill(()))
            .unwrap();
        let quote = preparation.quote.as_ref().unwrap();
        let set = quote
            .prefill_scopes
            .as_ref()
            .unwrap()
            .borrow_mut()
            .claim(&step)
            .unwrap();
        let plan = set.facts().plan();
        let (bank, view) = PrefillBankOwner::new(
            set,
            step.request(),
            &native_runtime,
            quote.record_quota.as_ref(),
            quote.graph_quota.as_ref().unwrap(),
            quote.original_controls().unwrap(),
        )
        .unwrap();
        let mut escaped = None;
        for index in 0..plan.scope_count() {
            let role = plan.role(index).unwrap();
            let (mut guard, roots) = view.begin(step.request(), role).unwrap();
            {
                let source = safemlx::Array::try_from_slice(&[2.0f32, -3.0, 7.0], &[3]).unwrap();
                let output = source.add(&source, &stream).unwrap();
                if index == 0 {
                    with_foreign_runtime(|| {
                        let error = output.evaluated().unwrap_err();
                        assert_eq!(
                            error.scoped_evaluation_cause(),
                            Some(ScopedEvaluationCause::RuntimeBusy)
                        );
                        escaped = Some(error);
                    });
                }
                let evaluated = output.evaluated().unwrap();
                assert_eq!(evaluated.try_as_slice::<f32>().unwrap(), &[4.0, -6.0, 14.0]);
                drop(evaluated);
                if let Some(roots) = roots {
                    roots.append(&output).unwrap();
                    roots.complete().unwrap();
                }
            }
            guard.seal();
            assert!(guard.finish().unwrap().settled, "role {role:?}");
        }
        assert!(escaped.is_some());
        drop((run, step, bank, view, preparation, probe));
        fixture::finish(runtime, &stream);
        // Exact role carrier still owns real Q custody after runtime and bank.
        assert!(pool.fixture_host_charge().unwrap() > 0);
        assert_eq!(
            escaped.as_ref().unwrap().scoped_evaluation_cause(),
            Some(ScopedEvaluationCause::RuntimeBusy)
        );
        drop(escaped);
        fixture::settle(&pool, 0);
    }
}

/// Lend the real original Source role to a backend-private component fixture.
/// This is test setup, not proof that additional prepared slots fit original Q.
/// The callback must settle its operations before returning. Its result may
/// retain actual custody, so teardown does not assume that the pool is empty.
fn with_original_operation_controls<T>(
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
    ) -> T,
) -> T {
    with_original_operation_controls_and_pool(|controls, observer, _pool| {
        operation(controls, observer)
    })
}

fn with_original_operation_controls_and_pool<T>(
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
    ) -> T,
) -> T {
    with_original_operation_controls_and_destinations(None, |controls, observer, pool, bank| {
        assert!(bank.is_none());
        operation(controls, observer, pool)
    })
}

thread_local! {
    static POINTWISE_CONTROLS: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static POINTWISE_ACCEPTED_CONTROLS: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static HOST_DESTINATION_FACTS: std::cell::Cell<Option<eredu_runtime::working_memory::HostDestinationFacts>> = const { std::cell::Cell::new(None) };
    static HOST_DESTINATION_BANK: RefCell<Option<eredu_runtime::working_memory::OriginalHostDestinationBank>> = const { RefCell::new(None) };
}
// Add the exact local producer requirement before the genuine prefill component
// is sealed into Q. This creates no new allowance or post-admission adjustment.
pub(super) fn pointwise_prefill_controls(
    facts: eredu_runtime::working_memory::TextPrefillScopeFacts,
) -> Result<eredu_runtime::working_memory::TextPrefillScopeFacts, Error> {
    POINTWISE_CONTROLS.with(|slot| {
        let Some(extra) = slot.get() else {
            return Ok(facts);
        };
        let old = facts
            .operation_control_bytes()
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
        let added = old
            .checked_add(extra)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let result = facts.with_operation_controls(Some(added)).map_err(memory)?;
        assert_eq!(
            result.total_bytes().unwrap().unwrap() - facts.total_bytes().unwrap().unwrap(),
            extra
        );
        POINTWISE_ACCEPTED_CONTROLS.with(|accepted| accepted.set(Some(added)));
        Ok(result)
    })
}
struct PointwiseControlsReset;
impl Drop for PointwiseControlsReset {
    fn drop(&mut self) {
        POINTWISE_CONTROLS.with(|slot| slot.set(None));
        POINTWISE_ACCEPTED_CONTROLS.with(|slot| slot.set(None));
    }
}
pub(super) fn capture_host_destinations(
    sequence: Option<&mut SequenceQuotation>,
) -> Result<(), Error> {
    if HOST_DESTINATION_FACTS.with(|facts| facts.get().is_some()) {
        // Consume the genuine accepted bank while the quote is still unique.
        // The fixture retains this owner; shared quote aliases grant no mutation.
        let bank = sequence
            .expect("destination fixture uses sequence admission")
            .take_host_destinations()?
            .expect("declared original destination bank");
        HOST_DESTINATION_BANK.with(|slot| assert!(slot.replace(Some(bank)).is_none()));
    }
    Ok(())
}
pub(super) fn host_destination_controls(
    controls: eredu_runtime::working_memory::PreparedTextControlWorkspace,
) -> Result<eredu_runtime::working_memory::PreparedTextControlWorkspace, Error> {
    HOST_DESTINATION_FACTS.with(|facts| match facts.get() {
        Some(facts) => controls.with_host_destinations(facts).map_err(memory),
        None => Ok(controls),
    })
}
struct HostFactsReset(Option<eredu_runtime::working_memory::HostDestinationFacts>);
impl Drop for HostFactsReset {
    fn drop(&mut self) {
        let unused = HOST_DESTINATION_BANK.with(|slot| slot.borrow_mut().take());
        drop(unused);
        HOST_DESTINATION_FACTS.with(|facts| facts.set(self.0));
    }
}
fn with_registered_original_operation_controls<T>(
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
    ) -> T,
) -> T {
    with_original_operation_controls_and_destinations_mode(
        None,
        true,
        |controls, observer, pool, bank| {
            assert!(bank.is_none());
            operation(controls, observer, pool)
        },
    )
}
fn with_original_operation_controls_and_destinations<T>(
    facts: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
        Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> T,
) -> T {
    with_original_operation_controls_and_destinations_mode(facts, false, operation)
}
struct OriginalOperationFixture {
    runtime: ModelRuntime<MlxBackend<'static>>,
    artifact: tempfile::TempDir,
    pool: MemoryLedger,
    stream: safemlx::Stream,
    tokenizer: Option<eredu_runtime::working_memory::OriginalTokenizer>,
    stops: Option<eredu_runtime::working_memory::OriginalStopSource>,
}
impl OriginalOperationFixture {
    fn prepare(registered_domain: bool) -> Self {
        let (runtime, artifact, pool, stream) = if registered_domain {
            // Follow the same retained selection and native creator as the public
            // path. Ordinary stream wrappers cannot authenticate original arenas.
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let artifact =
                crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
            let inspection =
                eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(
                    artifact.path(),
                )
                .unwrap();
            let factory = crate::MlxBackendFactory::default();
            let plan = eredu_core::ExecutionPlan::fully_resident(
                eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
            );
            let selected =
                eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
            let runtime = eredu_core::realize_execution_plan_target(&factory, &plan, selected)
                .unwrap()
                .into_runtime()
                .unwrap();
            let stream = runtime.backend().stream().clone();
            (runtime, artifact, pool, stream)
        } else {
            let stream = fixture::stream();
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let (runtime, artifact) = fixture::load(&stream, &pool, 0);
            (runtime, artifact, pool, stream)
        };
        use eredu_runtime::working_memory::OriginalTokenizerBackend;
        use eredu_text::{stop_storage::StopCompilePlan, tokenizer_storage::TokenizerPlan};
        use fixture::token_input::original_plain::json;
        let tokenizer = registered_domain.then(|| {
            MlxBackend::compile_original_tokenizer(
                &runtime,
                TokenizerPlan::prepare_json(json().as_bytes())
                    .unwrap()
                    .with_generation_domain()
                    .unwrap(),
            )
            .unwrap()
        });
        let stops = registered_domain.then(|| {
            MlxBackend::compile_original_text_stop_source(
                &runtime,
                StopCompilePlan::prepare_refs(&[]).unwrap(),
            )
            .unwrap()
        });
        Self {
            runtime,
            artifact,
            pool,
            stream,
            tokenizer,
            stops,
        }
    }
    fn finish(self) {
        let Self {
            runtime,
            artifact,
            pool,
            stream,
            tokenizer,
            stops,
        } = self;
        fixture::finish(runtime, &stream);
        drop((tokenizer, stops, artifact, pool));
    }
}

fn with_original_operation_controls_and_destinations_mode<T>(
    facts: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    registered_domain: bool,
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
        Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> T,
) -> T {
    let mut prepared = OriginalOperationFixture::prepare(registered_domain);
    let result = with_prepared_original_operation_controls(facts, &mut prepared, operation);
    prepared.finish();
    result
}
fn with_prepared_original_operation_controls<T>(
    facts: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    prepared: &mut OriginalOperationFixture,
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
        Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> T,
) -> T {
    with_prepared_original_operation_reservation(
        facts,
        prepared,
        |controls, observer, pool, bank, _reservation| operation(controls, observer, pool, bank),
    )
}
fn with_prepared_original_operation_reservation<T>(
    facts: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    prepared: &mut OriginalOperationFixture,
    operation: impl FnOnce(
        &eredu_runtime::working_memory::OriginalTextControlGuard,
        &safemlx::OriginalScopeObserver,
        &MemoryLedger,
        Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
        &eredu_runtime::working_memory::WorkingMemoryReservation,
    ) -> T,
) -> T {
    // Test-only addition occurs at the genuine pre-seal controls constructor.
    let reset = HostFactsReset(HOST_DESTINATION_FACTS.with(|slot| slot.replace(facts)));
    let OriginalOperationFixture {
        runtime,
        pool,
        stream,
        tokenizer,
        stops,
        ..
    } = prepared;
    let native_runtime = runtime
        .session()
        .payload
        .model
        .erased()
        .prefill_roots_runtime()
        .unwrap();
    use eredu_runtime::working_memory::AggregateGenerationDecoderInput;
    use fixture::token_input::original_plain::Domain;
    let decoder = tokenizer
        .as_ref()
        .zip(stops.as_ref())
        .map(|(tokenizer, stops)| {
            AggregateGenerationDecoderInput::new_plain_text(tokenizer, stops, 4, true).unwrap()
        });
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
        eredu_core::RetainedGenerationSequence,
        Error,
        eredu_core::BackendFailure,
    >()
    .unwrap();
    let probe = Probe::new(runtime, None, false);
    let run = if let Some(tokenizer) = tokenizer.as_ref() {
        (
            None,
            Some(
                ControlledTextGeneration::from_token_ids_with_sequence(
                    runtime,
                    eredu_core::TokenIdsInputPlan::new(&[2, 5, 7, 3, 11]).unwrap(),
                    original_source_config(),
                    Domain(tokenizer.clone()),
                    None,
                    GenerationSequenceRequest::new(4, &[])
                        .with_decoder(decoder.as_ref().unwrap())
                        .with_consumer(&consumer),
                )
                .unwrap(),
            ),
        )
    } else {
        (
            Some(
                TextGeneration::from_token_ids_with_sequence(
                    runtime,
                    eredu_core::TokenIdsInputPlan::new(&[2, 5, 7, 3, 11]).unwrap(),
                    original_source_config(),
                    TokenFilter::All,
                    None,
                    GenerationSequenceRequest::new(4, &[]),
                )
                .unwrap(),
            ),
            None,
        )
    };
    let host_destinations = HOST_DESTINATION_BANK.with(|slot| slot.borrow_mut().take());
    assert_eq!(host_destinations.is_some(), facts.is_some());
    drop(reset);
    let preparation = probe.take();
    let prepared = preparation.request.as_ref().unwrap();
    let step = prepared
        .claim_step(&probe.context(), PendingTextInput::Prefill(()))
        .unwrap();
    let quote = preparation.quote.as_ref().unwrap();
    let original = quote
        .prefill_scopes
        .as_ref()
        .unwrap()
        .borrow_mut()
        .claim(&step)
        .unwrap();
    assert_eq!(
        original.facts().plan().role(0),
        Some(PrefillControlRole::SourcePreparation)
    );
    if POINTWISE_CONTROLS.with(|slot| slot.get().is_some()) {
        let admitted = POINTWISE_ACCEPTED_CONTROLS
            .with(|slot| slot.get())
            .expect("cold plan joined before Q");
        assert_eq!(original.facts().operation_control_bytes(), Some(admitted));
    }
    let controls = quote.original_controls().unwrap();
    let (bank, view) = PrefillBankOwner::new(
        original,
        step.request(),
        &native_runtime,
        quote.record_quota.as_ref(),
        quote.graph_quota.as_ref().unwrap(),
        controls.clone(),
    )
    .unwrap();
    let (mut source, roots) = view
        .begin(step.request(), PrefillControlRole::SourcePreparation)
        .unwrap();
    assert!(roots.is_none());
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();

    // All real driver, request, quote, bank and current-role owners stay live.
    // The borrowed guard and observer cannot lend a reference beyond this call;
    // any explicit owned aliases in T still retain their actual original Q.
    let result = operation(
        &controls,
        &observer,
        &pool,
        host_destinations,
        step.request().memory_reservation(),
    );

    source.seal();
    assert!(
        source.finish().unwrap().settled,
        "fixture Source role remains pending"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let retired = observer.retire_completed_records().unwrap();
        if retired == safemlx::SubmissionRetirement::CompleteSnapshot {
            break;
        }
        assert_eq!(retired, safemlx::SubmissionRetirement::Busy);
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(observer);
    drop((step, bank, view, preparation, run, probe, controls));
    drop(decoder);
    runtime.synchronize().unwrap();
    stream.synchronize().unwrap();
    // T may contain an actual retained native cause, so no pool-zero wait is
    // valid here. The worker below returns () after retiring all of its work.
    result
}

#[test]
fn original_residency_capacity_retry_preserves_cause_and_source_retirement() {
    let _pool = crate::tests::support::test_utils::initialize_original_sources();
    // These are the actual manager streams. Prepare their persistent workers
    // before the genuine Source role; a Stream constructor alone is not enough.
    let cpu = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let source_stream = safemlx::Stream::new_with_device(&cpu);
    let device_stream = safemlx::Stream::new_with_device(&cpu);
    let native_runtime =
        safemlx::PrefillRootsRuntime::prepare_for_stream(&device_stream, &source_stream).unwrap();
    with_registered_original_operation_controls(|controls, observer, _pool| {
        crate::backend::runtime::residency::manager::exercise_original_capacity_retry(
            controls,
            observer,
            source_stream,
            device_stream,
        );
    });
    // The facts token survives the actual Source settlement and teardown.
    drop(native_runtime);
}

#[test]
fn original_gguf_miss_hit_and_later_miss_preserve_values_same_lease_and_weak_cache() {
    let source = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
    with_original_operation_controls(|controls, observer| {
        source.cache_miss_hit_and_retry(controls, observer);
    });
}

#[test]
fn original_gguf_affine_companions_share_one_nonzero_conversion_and_retire_last_owner() {
    let source = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new_affine();
    with_original_operation_controls(|controls, observer| {
        source.affine_companions(controls, observer);
    });
}

#[test]
fn original_gguf_read_failure_outlives_source_role_with_actual_box_and_custody() {
    let source = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
    let (failure, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (source.failed_read(controls, observer), pool.clone())
    });
    drop(source);
    assert!(failure.cause().store_error().is_some());
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(failure);
    fixture::settle(&pool, 0);
}

#[test]
fn original_gguf_unpublished_box_unwind_safely_retires_empty_pending_owner() {
    let source = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
    with_original_operation_controls(|controls, observer| {
        source.unwind_unpublished(controls, observer);
    });
}

#[test]
fn original_neural_two_parent_waits_and_evaluated_identity_boundaries_retire_exact_owners() {
    // Native stream owners and the nonzero eventless input exist before Source
    // entry. This test measures ordering/retirement, not cold allocation fit.
    let streams = std::array::from_fn(|_| fixture::stream());
    let input = crate::MlxTensor::from_array(
        safemlx::Array::try_from_slice(&[2.0f32, -3.5, 7.25], &[3]).unwrap(),
    );
    assert_eq!(
        input.as_array().evaluated().unwrap().as_slice::<f32>(),
        &[2.0, -3.5, 7.25]
    );
    with_original_operation_controls(|controls, observer| {
        crate::backend::nn::shared::PreparedNeuralSubmission::exercise_original_identity_join(
            controls, observer, &input, &streams,
        );
    });
    assert_eq!(
        input.as_array().evaluated().unwrap().as_slice::<f32>(),
        &[2.0, -3.5, 7.25]
    );
}

#[test]
fn original_named_destinations_preserve_cycle_aliases_refusals_and_cross_request_custody() {
    let environment = fixture::PreparedResidencyFixture::new();
    let pool = &environment.pool;
    let (runtime, artifact, source_baseline) = environment.load(0, None);
    let stream = runtime.backend().stream().clone();
    runtime.backend().validate_original_stream_owners().unwrap();
    let mut first = OriginalOperationFixture {
        runtime,
        artifact,
        pool: pool.clone(),
        stream,
        tokenizer: None,
        stops: None,
    };
    // Both loading authorities must retire before A deliberately escapes.
    // A's canonical cells keep their reserved account alive across B; opening
    // a new model-loading authority at that point correctly refuses.
    let before_second_load = pool.fixture_host_charge().unwrap();
    let (runtime, artifact, second_source_baseline) = environment.load(0, None);
    let final_source_baseline = source_baseline
        + second_source_baseline
            .checked_sub(before_second_load)
            .unwrap();
    let stream = runtime.backend().stream().clone();
    runtime.backend().validate_original_stream_owners().unwrap();
    let mut second = OriginalOperationFixture {
        runtime,
        artifact,
        pool: pool.clone(),
        stream,
        tokenizer: None,
        stops: None,
    };
    // Prime the genuine managed source before this component creates ordinary
    // numerical references. These extra named tables exercise the mechanism;
    // they are not a model quote-fit claim.
    let mut tables = crate::backend::runtime::residency::manager::NamedArraysFixture::new();
    let old =
        with_prepared_original_operation_controls(None, &mut first, |controls, _, _, bank| {
            assert!(bank.is_none());
            tables.first_request(controls)
        });
    first.finish();
    assert!(pool.fixture_host_charge().unwrap() > source_baseline);
    let collected = old.collect_plain_before_canonical();
    with_prepared_original_operation_controls(None, &mut second, |controls, _, _, bank| {
        assert!(bank.is_none());
        tables.second_request(old, collected, controls, pool);
    });
    second.finish();
    drop(tables);
    fixture::settle(pool, final_source_baseline);
}

#[test]
fn original_lease_return_moves_prepared_ids_on_miss_and_warm_and_keeps_request_custody() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let mut first_fixture = OriginalOperationFixture::prepare(true);
    let mut second_fixture = OriginalOperationFixture::prepare(true);
    let baseline = pool.fixture_host_charge().unwrap();
    let baseline_unquoted = pool.unquoted_owner_count().unwrap();
    let leases = crate::backend::runtime::residency::manager::LeaseReturnFixture::new(&pool);
    let (first, old_pool) = with_prepared_original_operation_controls(
        None,
        &mut first_fixture,
        |controls, observer, pool, bank| {
            assert!(bank.is_none());
            (leases.first_request(controls, observer), pool.clone())
        },
    );
    let (current, current_pool) = with_prepared_original_operation_controls(
        None,
        &mut second_fixture,
        |controls, observer, pool, bank| {
            assert!(bank.is_none());
            (
                leases.second_request(first, controls, observer),
                pool.clone(),
            )
        },
    );
    leases.settle_old_collection();
    // The second collection outlives its manager and both lexical requests.
    // Its arrays still own the first request's physical cells, while its own
    // final Vec/IDs/deferred node keep the second request's custody.
    drop(leases);
    assert!(old_pool.same_ledger(&current_pool));
    assert!(current_pool.fixture_host_charge().unwrap() > baseline);
    assert_eq!(current.leases().len(), 2);
    assert_eq!(
        current.leases()[0]
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    drop(current);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        disk::reclaim();
        pool.fixture_host_charge().unwrap() == baseline
            && pool.unquoted_owner_count().unwrap() == baseline_unquoted
    });
    first_fixture.finish();
    second_fixture.finish();
}

#[test]
fn original_resident_cached_decode_preserves_selected_model_scope_and_nonzero_state() {
    let stream = fixture::stream();
    let mut reference = None;
    for original in [false, true] {
        for controlled in [false, true] {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
            let trace = Trace::new();
            let input = vec![2, 5, 7, 3, 11];
            let outputs = if controlled {
                ControlledTextGeneration::from_input(
                    &mut runtime,
                    TextGenerationInput::TokenIds(input),
                    config(original),
                    disk::Controller::default(),
                )
                .unwrap()
                .map(|token| token.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::new(&mut runtime, input, config(original))
                    .unwrap()
                    .map(Result::unwrap)
                    .collect::<Vec<_>>()
            };
            assert_eq!(outputs.len(), 4);
            let tokens = outputs
                .iter()
                .map(|output| output.token_id().unwrap())
                .collect::<Vec<_>>();
            let values = runtime
                .session()
                .payload
                .model
                .erased()
                .resident_reset_source()
                .unwrap()
                .state()
                .retained_arrays()
                .into_iter()
                .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
                .collect::<Vec<_>>();
            assert!(!values.is_empty());
            assert!(values.iter().flatten().all(|value| value.is_finite()));
            assert!(values.iter().flatten().any(|value| *value != 0.0));
            assert!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .state_snapshot()
                    .iter()
                    .all(|(position, _)| *position == 8)
            );
            let actual = (tokens, values);
            if let Some(reference) = &reference {
                assert_eq!(&actual, reference);
            } else {
                reference = Some(actual);
            }
            let completed = trace
                .events()
                .into_iter()
                .filter_map(|event| match event {
                    Event::ModelCompleted { roots } => Some(roots),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(completed.len(), if original { 3 } else { 0 });
            assert!(completed.iter().all(|roots| *roots > 0));
            drop(outputs);
            fixture::finish(runtime, &stream);
            fixture::settle(&pool, 0);
        }
    }
}

#[test]
fn original_model_projection_refuses_foreign_authority_and_retains_outer_scope_after_inner_finish()
{
    use crate::backend::submission_recovery::prefill::ModelExecutionPreparation;
    use crate::backend::submission_recovery::{Retention, Status, prediction};
    struct Ticket {
        _request: InferenceRequest,
        original: Option<eredu_runtime::working_memory::OriginalPredictionRecoveryCustody>,
    }
    impl Retention for Ticket {
        fn observe(&self, _: Status) {}
    }
    impl prediction::PredictionRetention for Ticket {
        fn install_prediction_custody(
            &mut self,
            value: eredu_runtime::working_memory::OriginalPredictionRecoveryCustody,
        ) {
            self.original = Some(value);
        }
    }
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let native = runtime
        .session()
        .payload
        .model
        .erased()
        .prefill_roots_runtime()
        .unwrap();
    let probe = Probe::new(&runtime, None, false);
    let run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7, 3, 11]),
        config(true),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .unwrap();
    let preparation = probe.take();
    let prepared = preparation.request.as_ref().unwrap();
    let step = prepared
        .claim_step(&probe.context(), PendingTextInput::Prefill(()))
        .unwrap();
    let quote = preparation.quote.as_ref().unwrap();
    let facts = quote
        .prefill_scopes
        .as_ref()
        .unwrap()
        .borrow()
        .facts()
        .unwrap();
    let ready = ModelExecutionPreparation::new(
        step.request(),
        facts,
        native,
        quote.original_controls().unwrap(),
    )
    .unwrap();
    let mut roles = quote.claim_prediction_scopes(&step).unwrap().unwrap();
    let role = roles.take_model_execution().unwrap();
    let (mut recovery, owner) = prediction::begin_model(
        Some(role),
        Ticket {
            _request: step.request().clone(),
            original: None,
        },
        Some(ready),
    )
    .unwrap();
    let owner = owner.unwrap();
    let view = owner.projection();
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
    assert!(prediction::owns_observer(&recovery, &observer));
    let foreign_execution = eredu_runtime::working_memory::InferenceExecutionIdentity::default();
    let foreign = crate::memory_fixture::empty_admitted_request(
        &foreign_execution,
        step.request().geometry(),
    )
    .unwrap();
    assert!(view.validate_execution(&foreign_execution).is_err());
    assert!(view.begin(&foreign).is_err());
    {
        let mut child = safemlx::SubmissionScope::try_begin().unwrap();
        assert!(view.begin(step.request()).is_err());
        child.seal();
    }
    let (inner, roots) = view.begin(step.request()).unwrap();
    assert!(view.begin(step.request()).is_err());
    let source = safemlx::Array::try_from_slice(&[2.0f32, -3.0, 7.0], &[3]).unwrap();
    let result = source.add(&source, &stream).unwrap();
    roots.append(&result).unwrap();
    roots.complete_on_stream(&stream).unwrap();
    view.mark_completed().unwrap();
    inner.finish().unwrap();
    // Inner completion neither sealed nor replaced the outer original scope.
    let current = safemlx::OriginalScopeObserver::require_current().unwrap();
    assert!(observer.same_scope(&current));
    assert!(prediction::owns_observer(&recovery, &current));
    let later = result.add(&source, &stream).unwrap();
    assert_eq!(
        later.evaluated().unwrap().as_slice::<f32>(),
        &[6.0, -9.0, 21.0]
    );
    drop((later, result, source, roots, current));
    recovery.seal();
    assert!(recovery.finish().unwrap().settled);
    drop((view, observer, owner, roles, step, preparation, run, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, 0);
}

#[test]
fn speculative_capture_pair_keeps_actual_source_custody_after_request_in_both_drop_orders() {
    use std::sync::atomic::Ordering;
    for control_first in [false, true] {
        let (execution, control, drops, held, pool) =
            with_original_operation_controls_and_pool(|controls, _observer, pool| {
                let (execution, control, drops, held) =
                    crate::composition::mlx::speculative::capture_error_transport::fixture_pair(
                        controls,
                        pool,
                        control_first,
                    );
                (execution, control, drops, held, pool.clone())
            });
        // Both aliases outlive the genuine Source role, bank and request.
        // This checks error custody only, not an additional source admission.
        assert!(pool.fixture_host_charge().unwrap() > 0);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        if control_first {
            drop(control);
            assert!(pool.fixture_host_charge().unwrap() > 0);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(execution);
        } else {
            drop(execution);
            assert!(pool.fixture_host_charge().unwrap() > 0);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(control);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(held.load(Ordering::SeqCst));
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_speculative_completion_keeps_nonzero_duplicate_roots_and_exact_iterator_prefix() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let inputs = [
        safemlx::Array::from_slice(&[1.0_f32], &[1]),
        safemlx::Array::from_slice(&[3.0_f32], &[1]),
    ];
    for input in &inputs {
        input.evaluated().unwrap();
    }
    with_registered_original_operation_controls(|controls, observer, _pool| {
        crate::backend::nn::shared::PreparedNeuralSubmission::exercise_exact_root_iterator(
            controls, observer, &inputs[0], &stream,
        );
        Prepared::exercise_roots(controls, observer, &inputs, &stream);
    });
    drop(inputs);
    drop(native);
}

#[test]
fn original_speculative_busy_pair_retains_actual_custody_in_both_final_drop_orders() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = safemlx::Array::from_slice(&[7.0_f32], &[1]);
    root.evaluated().unwrap();
    let (pairs, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_busy_pair(controls, observer, &root, &stream, |work| {
                with_foreign_runtime(work)
            }),
            pool.clone(),
        )
    });
    assert!(pool.fixture_host_charge().unwrap() > 0);
    let [(execution_a, control_a), (execution_b, control_b)] = pairs;
    drop(control_a);
    drop(execution_b);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(execution_a);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(control_b);
    fixture::settle(&pool, 0);
    drop(root);
    drop(native);
}

#[test]
fn original_speculative_snapshot_publication_keeps_old_busy_and_three_snapshot_keys() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = safemlx::Array::from_slice(&[13.0_f32], &[1]);
    root.evaluated().unwrap();
    // Exact same configuration used by the genuine Source fixture below.
    let graph = config(true)
        .inference_policy()
        .graph_metadata_capacity_bytes
        .unwrap()
        .get();
    let count = usize::try_from(graph).unwrap().checked_add(1).unwrap();
    let (errors, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_publication(controls, observer, &root, &stream, count),
            pool.clone(),
        )
    });
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(errors);
    fixture::settle(&pool, 0);
    drop(root);
    drop(native);
}

#[test]
fn original_speculative_iterator_unwind_keeps_payload_until_unlocked_retirement() {
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = safemlx::Array::from_slice(&[17.0_f32], &[1]);
    root.evaluated().unwrap();
    let dropped = with_original_operation_controls(|controls, observer| {
        crate::backend::nn::shared::PreparedNeuralSubmission::exercise_root_iterator_unwind(
            controls, observer, &root, &stream,
        )
    });
    assert!(dropped.get());
    drop(root);
    drop(native);
}

#[test]
fn original_speculative_foreign_role_refusal_keeps_both_actual_custodies() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let (foreign, foreign_pool) = with_original_operation_controls_and_pool(|_, observer, pool| {
        (observer.clone(), pool.clone())
    });
    let (error, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        assert!(!observer.same_scope(&foreign));
        (
            Prepared::exercise_wrong_owner(controls, &foreign, &stream),
            pool.clone(),
        )
    });
    drop(foreign);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert!(foreign_pool.fixture_host_charge().unwrap() > 0);
    drop(error);
    fixture::settle(&pool, 0);
    fixture::settle(&foreign_pool, 0);
}

#[test]
fn original_speculative_terminal_source_keeps_first_offending_cause_after_request() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let offending = safemlx::error::Exception::custom("retained first speculative contract source");
    let (errors, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_terminal_source(controls, observer, offending),
            pool.clone(),
        )
    });
    assert!(pool.fixture_host_charge().unwrap() > 0);
    let [first, later] = errors;
    drop(first);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(later);
    fixture::settle(&pool, 0);
}

#[test]
fn original_speculative_deadline_overflow_quarantines_active_owner_and_retains_error() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = safemlx::Array::from_slice(&[23.0_f32], &[1]);
    root.evaluated().unwrap();
    let (error, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_deadline_overflow(controls, observer, &root, &stream),
            pool.clone(),
        )
    });
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(error);
    fixture::settle(&pool, 0);
    drop(root);
    drop(native);
}

#[test]
fn original_speculative_exact_native_root_reserve_failure_retains_source_after_request() {
    use crate::composition::mlx::speculative::original_completion::PreparedOriginalSpeculativeCompletion as Prepared;
    let stream = fixture::stream();
    let native = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = safemlx::Array::from_slice(&[19.0_f32], &[1]);
    root.evaluated().unwrap();
    let graph_capacity = usize::try_from(
        config(true)
            .inference_policy()
            .graph_metadata_capacity_bytes
            .unwrap()
            .get(),
    )
    .unwrap();
    let (error, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_exact_native_root_reserve_failure(
                controls,
                observer,
                &root,
                &stream,
                graph_capacity,
            ),
            pool.clone(),
        )
    });
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert_eq!(
        root.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[19.0]
    );
    drop(error);
    fixture::settle(&pool, 0);
    drop(root);
    drop(native);
}

#[test]
fn original_neural_clone_preparation_error_keeps_cause_and_custody_after_request() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    #[derive(Debug)]
    struct Source {
        pool: MemoryLedger,
        dropped: Arc<AtomicBool>,
        funded_on_drop: Arc<AtomicBool>,
    }
    impl std::fmt::Display for Source {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("retained cold native clone cause")
        }
    }
    impl std::error::Error for Source {}
    impl Drop for Source {
        fn drop(&mut self) {
            // The final native source child still sees its original admission.
            self.funded_on_drop.store(
                self.pool.fixture_host_charge().unwrap() > 0,
                Ordering::SeqCst,
            );
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let funded_on_drop = Arc::new(AtomicBool::new(false));
    let (error, pool, message) = with_original_operation_controls_and_pool(|controls, _, pool| {
        let cause = safemlx::error::Exception::from_source(Source {
            pool: pool.clone(),
            dropped: Arc::clone(&dropped),
            funded_on_drop: Arc::clone(&funded_on_drop),
        });
        assert_eq!(cause.scoped_evaluation_cause(), None);
        let message = cause.what().as_ptr() as usize;
        let error = crate::backend::runtime::execution::generic::OriginalOperationPlan::<()>::exercise_neural_clone_preparation_failure(controls, cause);
        (error, pool.clone(), message)
    });
    // Never-started bank/pending payloads use OrdinaryRetirement. Explicitly
    // destroy that queue before testing that only the escaping source keeps Q.
    assert!(safemlx::can_reclaim_submission_resources());
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert!(!dropped.load(Ordering::SeqCst));
    // Public neutral conversion must keep the same original Exception and text.
    let error = error.into_backend_failure();
    let source = std::error::Error::source(&error).unwrap();
    let cause = std::error::Error::source(source)
        .unwrap()
        .downcast_ref::<safemlx::error::Exception>()
        .unwrap();
    assert_eq!(cause.what(), "retained cold native clone cause");
    assert_eq!(cause.what().as_ptr() as usize, message);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(error);
    assert!(dropped.load(Ordering::SeqCst));
    assert!(funded_on_drop.load(Ordering::SeqCst));
    fixture::settle(&pool, 0);
}

#[test]
fn original_source_acquisition_exhaustion_preserves_typed_cause_without_new_custody() {
    use crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitions;
    use eredu_checkpoint::store::PreparedAcquisitionRefusal;
    let (error, pool) = with_original_operation_controls_and_pool(|controls, _, pool| {
        let mut bank = PreparedSourceAcquisitions::new(0, controls.clone()).unwrap();
        bank.seal().unwrap();
        // A second seal is a real finite-population refusal, with no pending
        // destination or payload whose guard could mask the escaping error.
        (bank.seal().unwrap_err(), pool.clone())
    });
    assert!(safemlx::can_reclaim_submission_resources());
    crate::backend::ordinary_retirement::reclaim_all();
    fixture::settle(&pool, 0);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<PreparedAcquisitionRefusal>()
        .expect("the exact nonowning category remains typed");
    assert!(matches!(cause, PreparedAcquisitionRefusal::Population));
    assert_eq!(error.to_string(), cause.to_string());
    drop(error);
    fixture::settle(&pool, 0);
}

#[test]
fn original_gguf_host_copy_busy_preserves_typed_source_after_role() {
    let gguf = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
    let (error, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (gguf.host_copy_busy(controls, observer), pool.clone())
    });
    drop(gguf);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert!(matches!(
        error.cause(),
        crate::backend::runtime::checkpoint::store::GgufHostCopyCause::Copy {
            cause: safemlx::OwnedHostCopyCause::Busy,
            native: None,
        }
    ));
    drop(error);
    fixture::settle(&pool, 0);
}

#[test]
fn original_gguf_host_copy_foreign_owner_failure_retains_both_final_drop_orders() {
    let (foreign, foreign_custody, foreign_pool) =
        with_original_operation_controls_and_pool(|controls, observer, pool| {
            (observer.clone(), controls.clone(), pool.clone())
        });
    for error_first in [true, false] {
        let gguf = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
        let (error, pool) =
            with_original_operation_controls_and_pool(|controls, observer, pool| {
                (
                    gguf.host_copy_failure(controls, observer, &foreign, 0, error_first),
                    pool.clone(),
                )
            });
        drop(gguf);
        if !error_first {
            assert!(pool.fixture_host_charge().unwrap() > 0);
        }
        drop(error);
        fixture::settle(&pool, 0);
    }
    drop(foreign);
    drop(foreign_custody);
    fixture::settle(&foreign_pool, 0);
}

#[test]
fn original_gguf_host_copy_affine_prefix_failure_keeps_actual_failed_input() {
    let (foreign, foreign_custody, foreign_pool) =
        with_original_operation_controls_and_pool(|controls, observer, pool| {
            (observer.clone(), controls.clone(), pool.clone())
        });
    let gguf = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new_affine();
    let (error, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            gguf.host_copy_failure(controls, observer, &foreign, 1, false)
                .unwrap(),
            pool.clone(),
        )
    });
    drop(gguf);
    assert_eq!(error.output_ordinal(), 1);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(error);
    fixture::settle(&pool, 0);
    drop(foreign);
    drop(foreign_custody);
    fixture::settle(&foreign_pool, 0);
}

#[test]
fn original_gguf_immutable_cached_alias_outlives_role_and_retires_on_other_thread() {
    let gguf = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
    let (array, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            gguf.host_copy_escaping_alias(controls, observer),
            pool.clone(),
        )
    });
    drop(gguf);
    assert!(pool.fixture_host_charge().unwrap() > 0);
    assert_eq!(
        array.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[1.5, -2.5, 3.5, 4.5, -5.5, 6.5, 7.5, -8.5]
    );
    std::thread::spawn(move || drop(array)).join().unwrap();
    safemlx::reclaim_allocation_owners();
    fixture::settle(&pool, 0);
}

#[test]
fn original_gguf_typed_transform_refusals_retain_actual_input_and_source_in_both_drop_orders() {
    use crate::backend::runtime::checkpoint::store::{GgufHostCopyCause, OriginalGgufMissFixture};
    for wrong_binding in [false, true] {
        for error_first in [false, true] {
            let fixture = OriginalGgufMissFixture::new();
            let (error, pool) =
                with_original_operation_controls_and_pool(|controls, observer, pool| {
                    (
                        fixture.host_transform_refusal(
                            controls,
                            observer,
                            wrong_binding,
                            error_first,
                        ),
                        pool.clone(),
                    )
                });
            drop(fixture);
            if let Some(error) = error {
                assert!(pool.fixture_host_charge().unwrap() > 0);
                if wrong_binding {
                    assert!(matches!(error.cause(), GgufHostCopyCause::TypedBinding));
                } else {
                    assert!(matches!(
                        error.cause(),
                        GgufHostCopyCause::Conversion(safemlx::error::IoError::InvalidFormat(_))
                    ));
                }
                drop(error);
            }
            fixture::settle(&pool, 0);
        }
    }
}

#[test]
fn original_gguf_admitted_g1_and_typed_refusals_retain_same_source_and_accepted_hold() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    use eredu_runtime::working_memory::{HostDestinationFacts, WorkingMemoryError};
    for after_read in [false, true] {
        let fixture = OriginalGgufMissFixture::new();
        let layouts = fixture.admitted_dense_layouts();
        let bytes = if after_read {
            layouts.iter().sum::<u64>() - 1
        } else {
            layouts[0] - 1
        };
        let facts = match HostDestinationFacts::new(bytes, layouts.len()) {
            Ok(facts) => facts,
            Err(WorkingMemoryError::UnknownBound) => return,
            Err(cause) => panic!("qualified destination: {cause}"),
        };
        let (error, pool) = with_original_operation_controls_and_destinations(
            Some(facts),
            |controls, observer, pool, bank| {
                (
                    fixture.admitted_refusal(controls, observer, bank.unwrap(), after_read),
                    pool.clone(),
                )
            },
        );
        assert!(pool.fixture_host_charge().unwrap() > 0);
        drop(error);
        fixture::settle(&pool, 0);
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    }
}

#[test]
fn original_gguf_admitted_exact_miss_retains_output_and_warm_hit_spends_no_destination() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    use eredu_runtime::working_memory::{HostDestinationFacts, WorkingMemoryError};
    let fixture = OriginalGgufMissFixture::new();
    let layouts = fixture.admitted_dense_layouts();
    let facts = match HostDestinationFacts::new(layouts.iter().sum(), layouts.len()) {
        Ok(facts) => facts,
        Err(WorkingMemoryError::UnknownBound) => return,
        Err(cause) => panic!("qualified destination: {cause}"),
    };
    let (output, pool) = with_original_operation_controls_and_destinations(
        Some(facts),
        |controls, observer, pool, bank| {
            (
                fixture.admitted_miss(controls, observer, bank.unwrap()),
                pool.clone(),
            )
        },
    );
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(output);
    fixture::settle(&pool, 0);
    let warm = OriginalGgufMissFixture::new();
    with_original_operation_controls_and_destinations(
        Some(HostDestinationFacts::new(0, 1).unwrap()),
        |controls, observer, _pool, bank| {
            warm.admitted_warm_hit(controls, observer, bank.unwrap());
        },
    );
}

#[test]
fn original_gguf_admitted_g2_g3_prefix_refusals_keep_same_source_and_release_exact_hold() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    use eredu_runtime::working_memory::{HostDestinationFacts, WorkingMemoryError};
    for index in 1..8 {
        let fixture = OriginalGgufMissFixture::new();
        let layouts = fixture.admitted_dense_layouts();
        let bytes = layouts[..=index].iter().sum::<u64>() - 1;
        let facts = match HostDestinationFacts::new(bytes, layouts.len()) {
            Ok(v) => v,
            Err(WorkingMemoryError::UnknownBound) => return,
            Err(e) => panic!("qualified destination: {e}"),
        };
        let (error, pool) = with_original_operation_controls_and_destinations(
            Some(facts),
            |controls, observer, pool, bank| {
                (
                    fixture.admitted_metadata_refusal(
                        controls,
                        observer,
                        bank.unwrap(),
                        layouts[index],
                    ),
                    pool.clone(),
                )
            },
        );
        assert!(pool.fixture_host_charge().unwrap() > 0);
        drop(error);
        fixture::settle(&pool, 0);
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    }
}

fn source_arena_facts(
    bytes: u64,
    attempts: usize,
    partitions: usize,
) -> Option<eredu_runtime::working_memory::HostDestinationFacts> {
    use eredu_runtime::working_memory::{HostDestinationFacts, HostSourceConstructionFacts};
    match HostDestinationFacts::new(0, 0) {
        Ok(facts) => Some(
            facts
                .with_source_constructions(
                    HostSourceConstructionFacts::new(bytes, attempts, partitions).unwrap(),
                )
                .unwrap(),
        ),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(cause) => panic!("source facts: {cause}"),
    }
}

#[test]
fn original_gguf_funded_source_arenas_charge_actual_misses_and_keep_escaped_aliases() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    for (constructor, outputs) in [
        (
            OriginalGgufMissFixture::new as fn() -> OriginalGgufMissFixture,
            1,
        ),
        (OriginalGgufMissFixture::new_affine, 3),
        (OriginalGgufMissFixture::new_mxfp4, 2),
        (OriginalGgufMissFixture::new_packed, 1),
    ] {
        let gguf = constructor();
        let layouts = gguf.source_storage_layouts();
        assert_eq!(layouts.len(), outputs);
        assert!(layouts.iter().all(|n| *n > 0));
        let Some(facts) = source_arena_facts(layouts.iter().sum::<u64>() * 3, layouts.len() * 3, 3)
        else {
            return;
        };
        let (array, pool) = with_original_operation_controls_and_destinations(
            Some(facts),
            |controls, observer, pool, bank| {
                let source = bank.unwrap().take_source_constructions().unwrap();
                (
                    gguf.funded_source_miss_hit_miss(controls, observer, source),
                    pool.clone(),
                )
            },
        );
        drop(gguf);
        assert!(pool.fixture_host_charge().unwrap() > 0);
        std::thread::spawn(move || drop(array)).join().unwrap();
        safemlx::reclaim_allocation_owners();
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_gguf_funded_source_refuses_dense_one_short_and_affine_second_arena() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    for affine in [false, true] {
        let gguf = if affine {
            OriginalGgufMissFixture::new_affine()
        } else {
            OriginalGgufMissFixture::new()
        };
        let layouts = gguf.source_storage_layouts();
        let ordinal = usize::from(affine);
        let bytes = if affine { layouts[0] } else { layouts[0] - 1 };
        let Some(facts) = source_arena_facts(bytes, ordinal + 1, 0) else {
            return;
        };
        let (error, pool) = with_original_operation_controls_and_destinations(
            Some(facts),
            |controls, observer, pool, bank| {
                (
                    gguf.funded_source_refusal(
                        controls,
                        observer,
                        bank.unwrap().take_source_constructions().unwrap(),
                        ordinal,
                    ),
                    pool.clone(),
                )
            },
        );
        drop(gguf);
        assert!(pool.fixture_host_charge().unwrap() > 0);
        drop(error);
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_gguf_source_bank_refuses_foreign_request_before_pending_construction() {
    use crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture;
    let gguf = OriginalGgufMissFixture::new();
    let bytes = gguf.source_storage_layouts()[0];
    let Some(facts) = source_arena_facts(bytes, 1, 0) else {
        return;
    };
    let (bank, origin_pool) =
        with_original_operation_controls_and_destinations(Some(facts), |_, _, pool, bank| {
            (
                bank.unwrap().take_source_constructions().unwrap(),
                pool.clone(),
            )
        });
    let bank = with_original_operation_controls(|controls, _| {
        gguf.reject_foreign_source_bank(controls, bank)
    });
    assert!(origin_pool.fixture_host_charge().unwrap() > 0);
    drop(bank);
    fixture::settle(&origin_pool, 0);
}

#[test]
fn original_pointwise_workspace_producer_consumes_prepared_eval_traversal_with_casts_and_broadcasts()
 {
    use crate::backend::nn::shared::PreparedNeuralSubmission as Prepared;
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (plan, unsupported) = Prepared::pointwise_declarations();
    let left = crate::MlxTensor::from_array(
        safemlx::Array::from_slice(&[2.0_f32, 4.0], &[2, 1])
            .as_dtype(safemlx::Dtype::Float16, &stream)
            .unwrap(),
    );
    let right =
        crate::MlxTensor::from_array(safemlx::Array::from_slice(&[1.0_f32, 3.0, 5.0], &[1, 3]));
    let scale = crate::MlxTensor::from_array(safemlx::Array::from_slice(&[2.0_f32; 6], &[2, 3]));
    for input in [&left, &right, &scale] {
        input.as_array().evaluated().unwrap();
    }
    let lazy = crate::MlxTensor::from_array(scale.as_array().square(&stream).unwrap());
    // The exact cold requirement enters the existing admission constructor;
    // the prepared slot and leaf loans are created only after that comparison.
    POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(plan.control_bytes())).is_none()));
    let _reset = PointwiseControlsReset;
    let (output, pool) = with_original_operation_controls_and_pool(|controls, observer, pool| {
        (
            Prepared::exercise_pointwise_traversal(
                controls,
                observer,
                &[&left, &right, &scale],
                &stream,
                plan,
                unsupported,
                &lazy,
            ),
            pool.clone(),
        )
    });
    assert_eq!(
        output
            .as_array()
            .evaluated()
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[6.0, 10.0, 14.0, 10.0, 14.0, 18.0]
    );
    drop(output);
    fixture::settle(&pool, 0);
    drop((runtime, left, right, scale));
}

#[path = "prefill_tests/grouped_linear_tests.rs"]
mod grouped_linear_tests;

#[path = "prefill_tests/router_tests.rs"]
mod router_tests;

#[path = "prefill_tests/foreground_source_capacity_tests.rs"]
mod foreground_source_capacity_tests;

#[path = "prefill_tests/operation_component_tests.rs"]
mod operation_component_tests;

#[path = "prefill_tests/rotary_component_tests.rs"]
mod rotary_component_tests;

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[path = "prefill_tests/embedding_component_tests.rs"]
mod embedding_component_tests;

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[path = "prefill_tests/prepared_rotary_component_tests.rs"]
mod prepared_rotary_component_tests;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[path = "prefill_tests/fp8_component_tests.rs"]
mod fp8_component_tests;

#[test]
fn inactive_parallel_slot_retires_only_its_alias_and_preserves_escaped_original_custody() {
    use crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection;
    let environment = fixture::PreparedResidencyFixture::new();
    let pool = &environment.pool;
    let (runtime, artifact, source_baseline) = environment.load(0, None);
    let stream = runtime.backend().stream().clone();
    runtime.backend().validate_original_stream_owners().unwrap();
    let mut prepared = OriginalOperationFixture {
        runtime,
        artifact,
        pool: pool.clone(),
        stream,
        tokenizer: None,
        stops: None,
    };
    let raw =
        with_prepared_original_operation_controls(None, &mut prepared, |controls, _, _, bank| {
            assert!(bank.is_none());
            controls.metadata_custody()
        });
    prepared.finish();
    assert!(
        pool.fixture_host_charge().unwrap() > source_baseline,
        "actual accepted span remains held"
    );
    let escaped = OriginalParallelControlProjection::exercise_inactive_retirement(raw);
    assert!(
        pool.fixture_host_charge().unwrap() > source_baseline,
        "removing the closed installed slot cannot release its escaped alias"
    );
    drop(escaped);
    fixture::settle(pool, source_baseline);
}
