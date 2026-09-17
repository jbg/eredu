use super::*;
mod ordinary_output;
use eredu_core::{GenerationSequenceConsumerLayout, TextGenerationDriver};
fn layout() -> GenerationSequenceConsumerLayout {
    // Neutral actual fixture types; no claim that this is the facade's layout.
    GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
}

#[test]
fn retained_provider_copy_has_independent_budget_state_and_output_lifetime() {
    use crate::execution_control::{PreparedTextHostCopy, TextHostCopyError};

    let (mut actual, state) = runtime(Mode {
        copy_headroom: 65_536,
        ..Mode::default()
    });
    let mut sequence = with_consumer(&mut actual, 1, None, &layout())
        .unwrap()
        .prepare_storage()
        .unwrap();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let pool = state.borrow().pool.clone();
    let used = pool.used_bytes().unwrap();
    let bytes = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            u64::MAX,
        )
        .unwrap()
        .storage_bytes()
        .unwrap();
    assert!(bytes > 0);
    let short = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            used + bytes - 1,
        )
        .unwrap()
        .copy(bytes);
    assert!(matches!(
        short,
        Err(TextHostCopyError::Admission(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(sequence.tokens(), &[5]);
    assert_eq!(sequence.finish_reason(), None);

    let mut copied = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &sequence,
            used + bytes,
        )
        .unwrap()
        .copy(bytes)
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), used + bytes);
    assert_ne!(sequence.tokens().as_ptr(), copied.tokens().as_ptr());
    assert_eq!(copied.tokens(), &[5]);
    assert_eq!(
        copied
            .commit(7, TokenTerminalSignals::default())
            .unwrap()
            .finish_reason,
        Some(eredu_core::FinishReason::Eos)
    );
    assert_eq!(
        sequence
            .commit(6, TokenTerminalSignals::default())
            .unwrap()
            .finish_reason,
        Some(eredu_core::FinishReason::MaxTokens)
    );
    drop(sequence);
    retire_request(&state);
    drop(actual);
    assert_eq!(pool.used_bytes().unwrap(), bytes);

    // Resume's logical branch lease follows the actual independent provider,
    // including aliases that survive both its cursor and its machine H handle.
    use crate::execution_control::{PendingSnapshotResumeRetention, SnapshotBudget};
    use eredu_core::execution_control::{SnapshotEstimate, SnapshotLimits, SnapshotResourceKind};
    let logical = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0,
        max_branches: 1,
        retained_bytes: 40,
        cumulative_copy_bytes: 120,
    });
    let cost = SnapshotEstimate {
        retained_bytes: 40,
        copy_bytes: 40,
    };
    let reserve = || {
        PendingSnapshotResumeRetention::reserve(&logical, SnapshotResourceKind::Branch, cost)
            .unwrap()
    };
    let resume_bytes = pool
        .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            u64::MAX,
        )
        .unwrap()
        .storage_bytes()
        .unwrap();
    let short = pool
        .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            bytes + resume_bytes - 1,
        )
        .unwrap()
        .copy_original_resume(40, reserve());
    assert!(matches!(
        short,
        Err(TextHostCopyError::Admission(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(logical.usage().branches, 0);
    assert_eq!(logical.usage().retained_bytes, 0);
    assert_eq!(logical.usage().cumulative_copy_bytes, 40);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    for host_last in [true, false] {
        let (branch, host) = pool
            .prepare_resume_provider_for_test::<RetainedGenerationSequence, TextHostCopyError>(
                &copied,
                bytes + resume_bytes,
            )
            .unwrap()
            .copy_original_resume(40, reserve())
            .unwrap();
        let ids = branch.into_token_ids();
        assert_eq!(&*ids, &[5, 7]);
        if host_last {
            drop(ids);
            assert_eq!(logical.usage().branches, 1);
            assert_eq!(pool.used_bytes().unwrap(), bytes + resume_bytes);
            drop(host);
        } else {
            let alias = ids.clone();
            drop(ids);
            drop(host);
            assert_eq!(logical.usage().branches, 1);
            assert_eq!(logical.usage().retained_bytes, 40);
            assert!(
                PendingSnapshotResumeRetention::reserve(
                    &logical,
                    SnapshotResourceKind::Branch,
                    cost
                )
                .is_err()
            );
            std::thread::spawn(move || drop(alias)).join().unwrap();
        }
        assert_eq!(logical.usage().branches, 0);
        assert_eq!(logical.usage().retained_bytes, 0);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
    }
    assert_eq!(logical.usage().cumulative_copy_bytes, 120);

    let plan = pool
        .prepare_generation_host_copy::<RetainedGenerationSequence, TextHostCopyError>(
            &copied,
            u64::MAX,
        )
        .unwrap();
    let later_bytes = plan.storage_bytes().unwrap();
    let later = plan.copy(later_bytes).unwrap();
    assert_eq!(later.tokens(), &[5, 7]);
    assert_eq!(later.finish_reason(), Some(eredu_core::FinishReason::Eos));
    drop(copied);
    let ids = later.into_token_ids();
    let alias = ids.clone();
    drop(ids);
    assert_eq!(&*alias, &[5, 7]);
    assert_eq!(pool.used_bytes().unwrap(), later_bytes);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
fn with_consumer(
    runtime: &mut ModelRuntime<Backend>,
    route: usize,
    options: Option<TextPreparationOptions>,
    consumer: &GenerationSequenceConsumerLayout,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let request = GenerationSequenceRequest::new(2, &[9, 7, 9]).with_consumer(consumer);
    let input = TextGenerationInput::TokenIds(vec![2]);
    let map = |e| match e {
        eredu_core::ControlledTextGenerationError::Preparation(e) => e,
        _ => panic!("unexpected preparation failure"),
    };
    let sequence = match route {
        0 => {
            let mut run = TextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(2),
                TokenFilter::All,
                options,
                request,
            )?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            sequence
        }
        1 => {
            let mut run = ControlledTextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(2),
                Controller,
                options,
                request,
            )
            .map_err(map)?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            sequence
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut run = driver
                .start_input_with_sequence(input, config(2), Controller, options, request)
                .map_err(map)?;
            let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
            assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
            sequence
        }
    };
    Ok(sequence)
}

#[test]
fn original_consumer_controls_add_exact_r_and_preserve_admission_phases_for_all_drivers() {
    let consumer = layout();
    for route in 0..3 {
        for capture in [false, true] {
            let mode = Mode {
                capture,
                ..Mode::default()
            };
            let options = || {
                capture.then(|| TextPreparationOptions {
                    interventions: None, capture: Some(capture_source()),
                })
            };
            let (mut baseline, base) = runtime(mode);
            let ordinary = extract(&mut baseline, 2, &[9, 7, 9], false, false, options()).unwrap();
            let base_r = base.borrow().r;
            drop(ordinary);
            retire_request(&base);
            drop(baseline);
            let (mut actual, state) = runtime(mode);
            let sequence = with_consumer(&mut actual, route, options(), &consumer).unwrap();
            assert_eq!(sequence.consumer_layout(), Some(&consumer));
            let facts = state.borrow();
            assert_eq!(facts.r - base_r, consumer.retention_peak_bytes() as u64);
            assert_eq!(facts.held, facts.p + 51 + facts.r);
            assert_eq!(
                facts.order,
                ["admit", "bind", "extract", "prompt", "sampling"]
            );
            assert_eq!(
                facts.votes,
                if capture {
                    vec![
                        (Stage::Admission, Status::Ready),
                        (Stage::Prompt, Status::Ready),
                        (Stage::Sampling, Status::Ready),
                        (Stage::Instrumentation, Status::Ready),
                    ]
                } else {
                    vec![
                        (Stage::Admission, Status::Ready),
                        (Stage::Prompt, Status::Ready),
                        (Stage::Sampling, Status::Ready),
                    ]
                }
            );
            assert!(
                facts
                    .active
                    .as_ref()
                    .unwrap()
                    .owner
                    .borrow_mut()
                    .take_generation_sequence_bank()
                    .is_none()
            );
            drop(facts);
            let (pool, held) = retire_request(&state);
            drop(actual);
            assert_eq!(pool.used_bytes().unwrap(), held);
            assert!(matches!(
                pool.pin_registered_storage([(1u32, 64)]),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            drop(sequence);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn original_consumer_exact_minus_one_rejects_original_width_one_before_binding_or_work() {
    let consumer = layout();
    for capture in [false, true] {
        let (mut actual, state) = runtime(Mode {
            capture,
            short: true,
            ..Mode::default()
        });
        let options = capture.then(|| TextPreparationOptions {
            interventions: None, capture: Some(capture_source()),
        });
        let error = with_consumer(&mut actual, 0, options, &consumer).unwrap_err();
        // The harness's constant fixture has only width one, so there is no stale
        // geometry return on fallback and this must be the actual budget cause.
        let mut source: &(dyn std::error::Error + 'static) = &error;
        let mut budget = false;
        loop {
            if let Some(WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: available,
            }) = source.downcast_ref::<WorkingMemoryError>()
            {
                assert_eq!(*required, *available + 1);
                budget = true;
            }
            match source.source() {
                Some(next) => source = next,
                None => break,
            }
        }
        assert!(
            budget,
            "exact minus one must preserve its typed budget cause: {error:?}"
        );
        let facts = state.borrow();
        assert_eq!(facts.order, ["admit"]);
        assert_eq!(facts.votes, [(Stage::Admission, Status::Failed)]);
        assert_eq!(facts.pool.used_bytes().unwrap(), 64);
        assert!(facts.active.is_none());
    }
}

#[test]
fn original_consumer_result_and_partial_iterator_keep_same_explicit_source_transition() {
    let consumer = layout();
    let (mut actual, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let sequence = with_consumer(&mut actual, 1, None, &consumer).unwrap();
    let mut sequence = sequence.prepare_storage().unwrap();
    let pointer = sequence.tokens().as_ptr();
    sequence.commit(4, TokenTerminalSignals::default()).unwrap();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let (pool, held) = retire_request(&state);
    drop(actual);
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    let tokens = sequence.into_token_ids();
    assert_eq!(tokens.as_ptr(), pointer);
    assert_eq!(tokens.as_slice(), [4, 5]);
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let alias = tokens.clone();
    let mut iter = tokens.into_iter();
    assert_eq!(iter.next(), Some(4));
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(iter);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
