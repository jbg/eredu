use super::*;
use crate::{prefill::PrefillChunk, ActivationObserver};

type Error = FundedCaptureError<MechanismError>;
type Observer<'a> = dyn ActivationObserver<Tensor, Error> + 'a;
fn metadata_source() -> SharedCapturePlan {
    let mut raw = raw();
    for selection in &mut raw.selections {
        selection.schedule.prefill = false;
    }
    admit(raw, point(), 4, false)
}
fn chunk(start: u64, end: u64) -> PrefillChunk {
    PrefillChunk {
        input: start..end,
        position: 2 + start,
        output: OutputDemand::LastPosition.for_chunk(end == 3),
    }
}
fn epoch(n: u64) -> DistributedCommitEpoch {
    DistributedCommitEpoch::new(n).unwrap()
}
fn enter(o: &mut Observer<'_>, index: u64) {
    o.begin_prefill_chunk(&chunk(index, index + 1)).unwrap();
    o.prepare_transaction(epoch(index + 1), ExpertPass::Prefill)
        .unwrap();
}
fn commit(o: &mut Observer<'_>, index: u64) {
    o.complete_transaction(epoch(index + 1)).unwrap();
    o.finish_transaction(epoch(index + 1), true);
}
fn skipped(step: &SharedCapturedStep, outcome: CaptureStepOutcome) {
    assert_eq!(step.phase(), CapturePhase::Prefill);
    assert_eq!(step.prediction_index(), 0);
    assert_eq!(step.outcome(), outcome);
    assert_eq!(step.step_usage().captures, 0);
    assert!(step.records().iter().all(|r| matches!(
        r.outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    )));
}

#[test]
fn chunks_spend_one_p0_row_and_metadata_then_preserve_nonzero_decode() {
    let source = metadata_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    let before = ledger(&pool);
    let allocations = CLAIM_ALLOCATIONS.get();
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            assert!(!o.requires_sequence_readout());
            for i in 0..3 {
                enter(o, i);
                o.observe("block.output", &tensor(0)).unwrap();
                commit(o, i);
                assert_eq!(ledger(&pool), before);
            }
            o.finish_prefill(true);
            o.finish_prefill(false); // a repeated terminal cannot replace delivery
        })
        .unwrap();
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    assert_eq!(funded.spent_steps(), 1);
    assert_eq!(backend.calls, 0);
    let p0 = funded.take_shared_step().unwrap().unwrap();
    skipped(&p0, CaptureStepOutcome::Committed);
    let mut expected = CaptureLedger::new(source.admission());
    expected.begin_step();
    crate::capture::CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
        .unwrap()
        .reserve_metadata(&mut expected)
        .unwrap();
    assert_eq!(p0.step_usage(), expected.step());
    assert_eq!(p0.cumulative_usage(), expected.total());
    funded
        .with_observer(&mut backend, 1, &|e| e, |o| {
            assert!(matches!(
                o.prepare_transaction(epoch(3), ExpertPass::Decode),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Undrained
                ))
            ));
        })
        .unwrap();
    assert!(!funded.has_pending_step());
    assert_eq!(funded.spent_steps(), 1);
    // Chunk epochs 1..3 were actually consumed. The first decode uses epoch 4,
    // while its capture coordinate is still request-local prediction one.
    funded
        .with_observer(&mut backend, 1, &|e| e, |o| {
            o.prepare_transaction(epoch(4), ExpertPass::Decode).unwrap();
            o.observe("block.output", &tensor(1)).unwrap();
            o.complete_transaction(epoch(4)).unwrap();
            o.finish_transaction(epoch(4), true);
        })
        .unwrap();
    let p1 = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(p1.prediction_index(), 1);
    assert_eq!(p1.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(p1.step_usage().captures, 3);
    assert_eq!(funded.spent_steps(), 2);
    let data = p1.records()[0]
        .payload
        .as_ref()
        .unwrap()
        .as_tensor()
        .unwrap();
    assert_eq!(
        data,
        &TensorObservation::new(vec![6, 2], TensorObservationData::F32(tensor(1).values)).unwrap()
    );
}

#[test]
fn outer_cancel_is_the_only_terminal_even_after_final_chunk_commit() {
    for entered in [0, 1, 3] {
        let source = metadata_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        funded
            .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
                for i in 0..entered {
                    enter(o, i);
                    commit(o, i);
                }
                o.finish_prefill(false);
                o.finish_prefill(true);
            })
            .unwrap();
        assert_eq!(funded.spent_steps(), usize::from(entered != 0));
        if entered == 0 {
            assert!(!funded.has_pending_step());
            assert_eq!(funded.usage(), CaptureUsage::default());
            assert!(funded.take_shared_step().unwrap().is_none());
        } else {
            skipped(
                &funded.take_shared_step().unwrap().unwrap(),
                CaptureStepOutcome::Aborted,
            );
        }
    }
}

#[test]
fn chunk_rejection_late_error_and_unwind_keep_one_aborted_frame() {
    for entered in [1, 3] {
        for mode in 0..3 {
            let source = metadata_source();
            let h = plan(&source).initialization_peak_bytes();
            let pool = WorkingMemoryPool::new(h, 0).unwrap();
            let (r, run) = fresh(&pool, h);
            let mut funded = session(&source, &pool, &r, &run);
            let result = catch_unwind(AssertUnwindSafe(|| {
                funded
                    .with_observer(
                        &mut Backend::default(),
                        0,
                        &|e| e,
                        |o| -> Result<(), Error> {
                            for i in 0..entered {
                                enter(o, i);
                                o.complete_transaction(epoch(i + 1)).unwrap();
                                o.finish_transaction(epoch(i + 1), mode != 0 || i + 1 != entered);
                            }
                            if mode == 1 {
                                return Err(FundedCaptureError::Backend(MechanismError::Sentinel(
                                    73,
                                )));
                            }
                            assert!(
                                mode != 2,
                                "after actual chunk commit, before outer completion"
                            );
                            o.finish_prefill(true); // cannot revive a rejected per-chunk commit
                            Ok(())
                        },
                    )
                    .unwrap()
            }));
            match mode {
                0 => result.unwrap().unwrap(),
                1 => assert!(matches!(
                    result.unwrap(),
                    Err(FundedCaptureError::Backend(MechanismError::Sentinel(73)))
                )),
                _ => assert!(result.is_err()),
            }
            assert_eq!(funded.spent_steps(), 1);
            let step = funded.take_shared_step().unwrap().unwrap();
            skipped(&step, CaptureStepOutcome::Aborted);
            assert_eq!(funded.usage(), step.cumulative_usage());
            let clone = step.clone();
            drop(step);
            drop(funded);
            drop(run);
            drop(r);
            assert!(pool.used_bytes().unwrap() >= h);
            drop(clone);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn missing_outer_success_never_promotes_an_intermediate_or_sealed_frame() {
    for entered in [1, 3] {
        let source = metadata_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        funded
            .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
                for i in 0..entered {
                    enter(o, i);
                    commit(o, i);
                }
                // Lexical Drop is the backstop when no outer terminal was sent.
            })
            .unwrap();
        skipped(
            &funded.take_shared_step().unwrap().unwrap(),
            CaptureStepOutcome::Aborted,
        );
        assert_eq!(funded.spent_steps(), 1);
    }
}

#[test]
fn bad_coordinates_reject_before_claim_and_later_gaps_abort_without_refund() {
    let source = metadata_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    for bad in [
        PrefillChunk {
            input: 1..2,
            ..chunk(0, 1)
        },
        PrefillChunk {
            input: 0..0,
            ..chunk(0, 1)
        },
        PrefillChunk {
            input: 0..4,
            ..chunk(0, 1)
        },
        PrefillChunk {
            position: 0,
            ..chunk(0, 1)
        },
        PrefillChunk {
            position: u64::MAX,
            ..chunk(0, 1)
        },
        PrefillChunk {
            output: OutputDemand::LastPosition,
            ..chunk(0, 1)
        },
    ] {
        funded
            .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
                assert!(matches!(
                    o.begin_prefill_chunk(&bad),
                    Err(FundedCaptureError::Protocol(CaptureProtocolError::Geometry))
                ));
                o.finish_prefill(false);
            })
            .unwrap();
        assert_eq!(funded.spent_steps(), 0);
        assert!(!funded.has_pending_step());
        assert_eq!(funded.usage(), CaptureUsage::default());
    }
    funded
        .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
            enter(o, 0);
            commit(o, 0);
            assert!(matches!(
                o.begin_prefill_chunk(&chunk(2, 3)),
                Err(FundedCaptureError::Protocol(CaptureProtocolError::Geometry))
            ));
            o.finish_prefill(true);
        })
        .unwrap();
    skipped(
        &funded.take_shared_step().unwrap().unwrap(),
        CaptureStepOutcome::Aborted,
    );
    assert_eq!(funded.spent_steps(), 1);
}

#[test]
fn stale_or_wrong_chunk_epochs_and_repeated_completion_cannot_commit() {
    for mode in 0..5 {
        let source = metadata_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        funded
            .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
                enter(o, 0);
                match mode {
                    0 => {
                        assert!(matches!(
                            o.complete_transaction(epoch(2)),
                            Err(FundedCaptureError::Protocol(
                                CaptureProtocolError::Transaction
                            ))
                        ));
                    }
                    1 | 2 => {
                        commit(o, 0);
                        o.begin_prefill_chunk(&chunk(1, 2)).unwrap();
                        assert!(matches!(
                            o.prepare_transaction(
                                if mode == 1 { epoch(1) } else { epoch(2) },
                                if mode == 1 {
                                    ExpertPass::Prefill
                                } else {
                                    ExpertPass::Decode
                                }
                            ),
                            Err(FundedCaptureError::Protocol(
                                CaptureProtocolError::Transaction
                            ))
                        ));
                    }
                    3 => {
                        o.complete_transaction(epoch(1)).unwrap();
                        assert!(matches!(
                            o.complete_transaction(epoch(1)),
                            Err(FundedCaptureError::Protocol(
                                CaptureProtocolError::Transaction
                            ))
                        ));
                    }
                    _ => o.finish_transaction(epoch(1), true), // no complete: cannot promote
                }
                o.finish_prefill(true);
            })
            .unwrap();
        skipped(
            &funded.take_shared_step().unwrap().unwrap(),
            CaptureStepOutcome::Aborted,
        );
        assert_eq!(funded.spent_steps(), 1);
    }
}

struct Uncalled;
impl eredu_nn::RetainedGeneratedTensorFactory<Tensor, Error> for Uncalled {
    fn program(&self) -> eredu_nn::GeneratedTensorProgram<'_> {
        panic!("no factory inspection between chunks")
    }
    fn visit_sources(
        &mut self,
        _: &mut dyn FnMut(eredu_nn::GeneratedTensorSourceRole, &Tensor) -> Result<(), Error>,
    ) -> Result<(), Error> {
        panic!("no source visitation between chunks")
    }
    fn generate(
        &mut self,
        _: &mut dyn FnMut(&Tensor) -> Result<(), Error>,
    ) -> Result<Tensor, Error> {
        panic!("no generation between chunks")
    }
}
#[test]
fn value_and_both_generated_hooks_require_a_current_chunk_transaction() {
    let source = metadata_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    let descriptor = crate::capture::generated_capture_source(
        &eredu_nn::BlockFp8InputReconstructionPlan::new(&[1, 2])
            .unwrap()
            .logical_capture_source()
            .unwrap(),
    );
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            enter(o, 0);
            commit(o, 0);
            assert!(matches!(
                o.observe("block.output", &tensor(0)),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Transaction
                ))
            ));
            assert!(matches!(
                o.observe_generated("block.output", &tensor(0), &descriptor, &mut || panic!(
                    "bare factory"
                )),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Transaction
                ))
            ));
            assert!(matches!(
                o.observe_generated_retained(
                    "block.output",
                    &tensor(0),
                    &descriptor,
                    &mut Uncalled
                ),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Transaction
                ))
            ));
            o.finish_prefill(false);
        })
        .unwrap();
    skipped(
        &funded.take_shared_step().unwrap().unwrap(),
        CaptureStepOutcome::Aborted,
    );
    assert_eq!(backend.calls, 0);
}

#[test]
fn selected_split_prefill_stays_rejected_but_full_sequence_and_unannotated_match() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            assert!(o.requires_sequence_readout());
            assert!(matches!(
                o.begin_prefill_chunk(&chunk(0, 1)),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::PrefillAttribution
                ))
            ));
        })
        .unwrap();
    assert_eq!(funded.spent_steps(), 0);
    assert!(!funded.has_pending_step());
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            o.begin_prefill_chunk(&PrefillChunk {
                output: OutputDemand::Sequence,
                ..chunk(0, 3)
            })
            .unwrap();
            o.prepare_transaction(epoch(1), ExpertPass::Prefill)
                .unwrap();
            o.observe("block.output", &tensor(0)).unwrap();
            o.complete_transaction(epoch(1)).unwrap();
            o.finish_transaction(epoch(1), true);
            o.finish_prefill(true);
        })
        .unwrap();
    let actual = funded.take_shared_step().unwrap().unwrap();
    let other_pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (other_r, other_run) = fresh(&other_pool, h);
    let mut ordinary = session(&source, &other_pool, &other_r, &other_run);
    forward(&mut ordinary, &mut Backend::default(), 0, true).unwrap();
    let expected = ordinary.take_shared_step().unwrap().unwrap();
    assert_eq!(actual.outcome(), expected.outcome());
    assert_eq!(actual.records(), expected.records());
    assert_eq!(actual.step_usage(), expected.step_usage());
    assert_eq!(backend.calls, 2);
}

#[test]
fn empty_plan_chunks_keep_mandatory_terminal_drain_without_claims() {
    let mut raw = raw();
    raw.selections.clear();
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    funded
        .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
            for i in 0..3 {
                enter(o, i);
                commit(o, i);
            }
            o.finish_prefill(true);
        })
        .unwrap();
    assert!(funded.has_pending_step());
    assert!(funded
        .with_observer(&mut Backend::default(), 1, &|e| e, |_| ())
        .is_err());
    assert!(funded.take_shared_step().unwrap().is_none());
    assert!(!funded.has_pending_step());
    funded
        .with_observer(&mut Backend::default(), 1, &|e| e, |o| {
            o.prepare_transaction(epoch(4), ExpertPass::Decode).unwrap();
            o.complete_transaction(epoch(4)).unwrap();
            o.finish_transaction(epoch(4), true);
        })
        .unwrap();
    assert!(funded.has_pending_step());
    assert!(funded.take_shared_step().unwrap().is_none());
    assert_eq!(funded.spent_steps(), 0);
    assert_eq!(funded.usage(), CaptureUsage::default());
}

#[test]
fn final_sealing_failure_keeps_original_partial_owner_and_all_charges() {
    let source = metadata_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    funded
        .with_observer(&mut Backend::default(), 0, &|e| e, |o| {
            for i in 0..2 {
                enter(o, i);
                commit(o, i);
            }
            enter(o, 2);
            drop(run);
            assert!(matches!(
                o.complete_transaction(epoch(3)),
                Err(FundedCaptureError::Host(CaptureRunHostError::Step(_)))
            ));
            o.finish_transaction(epoch(3), false);
            o.finish_prefill(true);
        })
        .unwrap();
    assert_eq!(funded.spent_steps(), 1);
    let held = ledger(&pool);
    assert!(funded.take_shared_step().is_err());
    assert!(funded.has_pending_step());
    assert_eq!(ledger(&pool), held);
    drop(funded);
    drop(r);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
