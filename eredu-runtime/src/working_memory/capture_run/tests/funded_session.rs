mod generated;
mod prefill;
use super::*;
use crate::ExpertPass;
use crate::capture::{
    CaptureProtocolError, CaptureSession, FundedCaptureDrainError, FundedCaptureError,
    FundedCaptureSession, ScheduledCaptureBackend,
};

struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}
fn tensor(p: u64) -> Tensor {
    // The shared fixture retains two cached positions plus three prompt positions.
    let rows = 5 + p as usize;
    Tensor {
        shape: vec![rows, 2],
        values: (0..rows * 2).map(|n| (n as f32 - 2.0) * 0.25).collect(),
    }
}
#[derive(Debug, thiserror::Error)]
enum MechanismError {
    #[error("actual source differs")]
    Shape,
    #[error("original sentinel {0}")]
    Sentinel(u32),
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
}
#[derive(Default)]
struct Backend {
    calls: usize,
    fail: bool,
    panic: bool,
}
fn usage(n: usize) -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: n as u64 * 4,
        host_bytes: n as u64 * 4,
        encoded_bytes: 8192,
    }
}
impl ScheduledCaptureBackend for Backend {
    type Tensor = Tensor;
    type Error = MechanismError;
    fn validate_source(
        &self,
        value: &Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        if value.shape != geometry.source_shape() {
            return Err(MechanismError::Shape);
        }
        Ok(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Tensor,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(geometry.elements()))
    }
    fn transform(
        &mut self,
        value: &Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        self.calls += 1;
        let mut output = claim.prepare()?;
        if output.len() > 0 {
            output.push_f32(value.values[0]).unwrap();
        }
        assert!(!self.panic, "after actual partial host construction");
        if self.fail {
            return Err(MechanismError::Sentinel(73));
        }
        while output.initialized_count() < output.len() {
            output
                .push_f32(value.values[output.initialized_count()])
                .unwrap();
        }
        Ok(output.finish().unwrap())
    }
}
impl CaptureBackend for Backend {
    type Tensor = Tensor;
    type Error = MechanismError;
    fn shape(&self, value: &Tensor) -> Result<Vec<u64>, Self::Error> {
        Ok(value.shape.iter().map(|&n| n as u64).collect())
    }
    fn source_dtype(&self, _: &Tensor) -> Option<TensorDtype> {
        Some(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let n = slice.shape.iter().product::<u64>();
        let n = match selection.transform {
            CaptureTransform::Preview { max_elements } => n.min(max_elements),
            _ => n,
        };
        Ok(usage(n as usize))
    }
    fn transform(
        &mut self,
        value: &Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.calls += 1;
        let (shape, n) = match selection.transform {
            CaptureTransform::Preview { max_elements } => {
                let n = value.values.len().min(max_elements as usize);
                (vec![n], n)
            }
            _ => (
                slice.shape.iter().map(|&n| n as usize).collect(),
                value.values.len(),
            ),
        };
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                shape,
                TensorObservationData::F32(value.values[..n].to_vec()),
            )
            .unwrap(),
        ))
    }
}
fn session(
    source: &SharedCapturePlan,
    pool: &WorkingMemoryPool,
    r: &WorkingMemoryReservation,
    run: &WorkingMemoryFundingRun,
) -> FundedCaptureSession {
    let s = run
        .prepare_capture_run(r, plan(source))
        .unwrap()
        .into_capture_session()
        .unwrap();
    assert!(s.source().same_storage(source));
    assert_eq!(ledger(pool).2, plan(source).initialization_peak_bytes());
    s
}
fn forward(
    s: &mut FundedCaptureSession,
    b: &mut Backend,
    p: u64,
    committed: bool,
) -> Result<(), FundedCaptureError<MechanismError>> {
    let epoch = DistributedCommitEpoch::new(p + 1).unwrap();
    s.with_observer(b, p, &|e| e, |o| {
        assert!(o.transactional());
        assert!(o.requires_prepared_traversal());
        let guard = crate::inspection::ObservationTransactionGuard::new(o, epoch);
        guard.observer.prepare_transaction(
            epoch,
            if p == 0 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            },
        )?;
        guard.observer.observe("block.output", &tensor(p))?;
        guard.observer.complete_transaction(epoch)?;
        guard.finish(committed);
        Ok(())
    })
    .unwrap()
}
#[test]
fn exact_factory_and_legacy_prefill_decode_wire_quota_parity() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (short_r, short_run) = fresh(&pool, h - 1);
    let before = ledger(&pool);
    let allocations = CLAIM_ALLOCATIONS.get();
    assert!(
        short_run
            .prepare_capture_run(&short_r, plan(&source))
            .is_err()
    );
    assert_eq!(ledger(&pool), before);
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    drop(short_r);
    drop(short_run);
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut legacy = CaptureSession::from_shared_plan(source.clone());
    let mut fb = Backend::default();
    let mut lb = Backend::default();
    for p in 0..4 {
        forward(&mut funded, &mut fb, p, true).unwrap();
        assert!(funded.has_pending_step());
        assert!(
            funded
                .with_observer(&mut fb, p + 1, &|e| e, |_| ())
                .is_err()
        );
        let epoch = DistributedCommitEpoch::new(p + 1).unwrap();
        legacy
            .prepare_step_transaction(
                epoch,
                if p == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                },
                p,
            )
            .unwrap();
        legacy.observe(&mut lb, "block.output", &tensor(p)).unwrap();
        legacy.complete_transaction(epoch).unwrap();
        legacy.finish_transaction(epoch, true);
        let actual = funded.take_shared_step().unwrap().unwrap();
        let expected = legacy.take_step().unwrap();
        let mut a = serde_json::to_value(actual).unwrap();
        let mut b = serde_json::to_value(expected).unwrap();
        a.as_object_mut().unwrap().remove("capture_seconds");
        b.as_object_mut().unwrap().remove("capture_seconds");
        assert_eq!(a, b);
        assert!(!funded.has_pending_step());
        assert_eq!(funded.spent_steps(), p as usize + 1);
        assert_eq!(pool.used_bytes().unwrap(), h);
    }
    assert_eq!(fb.calls, lb.calls);
    assert_eq!(funded.usage().captures, 10);
}
#[test]
fn exact_source_and_payload_aliases_outlive_closed_session() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    let pointer = funded.source().admission() as *const _;
    forward(&mut funded, &mut backend, 0, true).unwrap();
    let step = funded.take_shared_step().unwrap().unwrap();
    let alias = step.clone();
    assert!(alias.same_storage(&step));
    let CapturePayload::SharedTensor(tensor) = step.records()[0].payload.as_ref().unwrap() else {
        panic!("shared actual tensor")
    };
    let tensor = tensor.clone();
    assert_eq!(funded.source().admission() as *const _, pointer);
    assert_eq!(tensor.shape(), &[5, 2]);
    drop(step);
    drop(funded);
    drop(source);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(tensor);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_error_and_unwind_keep_aborted_frame_and_spent_coordinates() {
    for panic in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        let mut backend = Backend {
            fail: !panic,
            panic,
            ..Default::default()
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            forward(&mut funded, &mut backend, 0, true)
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(matches!(
                result.unwrap(),
                Err(FundedCaptureError::Backend(MechanismError::Sentinel(73)))
            ));
        }
        assert!(funded.has_pending_step());
        assert_eq!(funded.spent_steps(), 1);
        let step = funded.take_shared_step().unwrap().unwrap();
        assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
        assert_eq!(step.cumulative_usage().captures, 1);
        assert_eq!(backend.calls, 1);
        if !panic {
            assert!(matches!(
                step.records()[0].outcome,
                CaptureOutcome::Failed {
                    reason: CaptureFailureReason::Native,
                    ..
                }
            ));
        }
        drop(step);
        backend.fail = false;
        backend.panic = false;
        forward(&mut funded, &mut backend, 1, true).unwrap();
        assert_eq!(
            funded
                .take_shared_step()
                .unwrap()
                .unwrap()
                .cumulative_usage()
                .captures,
            4
        );
    }
}
#[test]
fn sealed_delivery_survives_close_but_aborted_drain_keeps_failed_owner() {
    for seal in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        let mut backend = Backend::default();
        funded
            .with_observer(&mut backend, 0, &|e| e, |o| {
                let e = DistributedCommitEpoch::FIRST;
                o.prepare_transaction(e, ExpertPass::Prefill).unwrap();
                o.observe("block.output", &tensor(0)).unwrap();
                if seal {
                    o.complete_transaction(e).unwrap();
                }
                drop(run);
                o.finish_transaction(e, seal);
            })
            .unwrap();
        if seal {
            let step = funded.take_shared_step().unwrap().unwrap();
            assert_eq!(step.outcome(), CaptureStepOutcome::Committed);
            drop(step);
        } else {
            let before = ledger(&pool);
            assert!(matches!(
                funded.take_shared_step(),
                Err(FundedCaptureDrainError::Delivery(_))
            ));
            assert!(funded.has_pending_step());
            assert_eq!(ledger(&pool), before);
            assert!(funded.take_shared_step().is_err());
            assert_eq!(funded.spent_steps(), 1);
        }
        drop(funded);
        drop(r);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn skipped_quota_and_schedule_do_not_call_native_worker() {
    let mut raw = raw();
    raw.limits.per_step.captures = 0;
    raw.limits.on_limit = CaptureLimitPolicy::Skip;
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    forward(&mut funded, &mut backend, 0, true).unwrap();
    let step = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(backend.calls, 0);
    assert_eq!(step.cumulative_usage().captures, 0);
    assert!(matches!(
        step.records()[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit { .. }
        }
    ));
    assert!(matches!(
        step.records()[1].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    assert_eq!(step.records()[0].source_dtype, Some(TensorDtype::F32));
}
#[test]
fn failed_metadata_start_is_terminal_without_unpriced_frame_or_retry() {
    let mut raw = raw();
    raw.limits.per_step.host_bytes = 1;
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    assert!(matches!(
        forward(&mut funded, &mut backend, 0, true),
        Err(FundedCaptureError::Admission(CaptureError::Limit { .. }))
    ));
    assert!(funded.has_pending_step());
    assert_eq!(funded.spent_steps(), 1);
    assert_eq!(backend.calls, 0);
    assert!(funded.take_shared_step().unwrap().is_none());
    assert!(!funded.has_pending_step());
    let before = funded.usage();
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            assert!(matches!(
                o.prepare_transaction(DistributedCommitEpoch::FIRST, ExpertPass::Prefill),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Undrained
                ))
            ));
        })
        .unwrap();
    assert_eq!(funded.usage(), before);
    assert!(!funded.has_pending_step());
}
#[test]
fn missing_duplicate_and_repeated_terminal_follow_same_transaction_rules() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            assert!(o.observe("block.output", &tensor(0)).is_err());
            o.prepare_transaction(DistributedCommitEpoch::FIRST, ExpertPass::Prefill)
                .unwrap();
            o.complete_transaction(DistributedCommitEpoch::FIRST)
                .unwrap();
            o.finish_transaction(DistributedCommitEpoch::FIRST, true);
            o.finish_transaction(DistributedCommitEpoch::FIRST, false);
        })
        .unwrap();
    let first = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(first.outcome(), CaptureStepOutcome::Committed);
    assert!(matches!(
        first.records()[0].outcome,
        CaptureOutcome::Missing
    ));
    assert_eq!(backend.calls, 0);
    funded
        .with_observer(&mut backend, 1, &|e| e, |o| {
            let e = DistributedCommitEpoch::new(2).unwrap();
            o.prepare_transaction(e, ExpertPass::Decode).unwrap();
            o.observe("block.output", &tensor(1)).unwrap();
            assert!(matches!(
                o.observe("block.output", &tensor(1)),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::Duplicate
                ))
            ));
        })
        .unwrap();
    let second = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(second.outcome(), CaptureStepOutcome::Aborted);
    assert_eq!(backend.calls, 3);
    assert_eq!(second.cumulative_usage().captures, 3);
}
#[test]
fn factory_rejects_spent_or_closed_bank() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    drop(bank.begin_step(CapturePhase::Prefill, 0).unwrap());
    assert!(matches!(
        bank.into_capture_session(),
        Err(CaptureRunHostError::Coordinate)
    ));
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let (r, run) = fresh(&pool, h);
    let bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    drop(run);
    assert!(matches!(
        bank.into_capture_session(),
        Err(CaptureRunHostError::Memory(_))
    ));
    drop(r);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn encoding_rejection_and_failed_sealing_leave_original_charged_delivery() {
    let mut large = point();
    large.axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(4096);
    let source = admit(raw(), large, 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    let value = Tensor {
        shape: vec![5, 4096],
        values: vec![1.25; 5 * 4096],
    };
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            let epoch = DistributedCommitEpoch::FIRST;
            o.prepare_transaction(epoch, ExpertPass::Prefill).unwrap();
            assert!(matches!(
                o.observe("block.output", &value),
                Err(FundedCaptureError::Host(CaptureRunHostError::Step(
                    CaptureStepError::EncodedSize { index: 0 }
                )))
            ));
        })
        .unwrap();
    let step = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
    assert!(matches!(
        step.records()[0].outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Invalid,
            ..
        }
    ));
    assert!(step.records()[0].payload.is_none());
    assert_eq!(step.cumulative_usage().captures, 1);
    drop(step);
    funded
        .with_observer(&mut backend, 1, &|e| e, |o| {
            let epoch = DistributedCommitEpoch::new(2).unwrap();
            o.prepare_transaction(epoch, ExpertPass::Decode).unwrap();
            drop(run);
            assert!(matches!(
                o.complete_transaction(epoch),
                Err(FundedCaptureError::Host(CaptureRunHostError::Step(_)))
            ));
        })
        .unwrap();
    assert!(funded.take_shared_step().is_err());
    assert!(funded.has_pending_step());
    drop(funded);
    drop(r);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_mismatch_and_generated_source_reject_before_worker_or_factory() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            o.prepare_transaction(DistributedCommitEpoch::FIRST, ExpertPass::Prefill)
                .unwrap();
            assert!(matches!(
                o.observe("block.output", &tensor(1)),
                Err(FundedCaptureError::Backend(MechanismError::Shape))
            ));
        })
        .unwrap();
    let step = funded.take_shared_step().unwrap().unwrap();
    assert_eq!(step.cumulative_usage().captures, 0);
    assert_eq!(backend.calls, 0);
    drop(step);
    let calls = Cell::new(0);
    funded
        .with_observer(&mut backend, 1, &|e| e, |o| {
            let epoch = DistributedCommitEpoch::new(2).unwrap();
            o.prepare_transaction(epoch, ExpertPass::Decode).unwrap();
            let source = GeneratedCaptureSource {
                creation_bytes: 64,
                source_dtype: Some(TensorDtype::F32),
            };
            let mut generate = || {
                calls.set(calls.get() + 1);
                Ok(tensor(1))
            };
            o.observe_generated("unselected", &tensor(1), &source, &mut generate)
                .unwrap();
            assert!(matches!(
                o.observe_generated("block.output", &tensor(1), &source, &mut generate),
                Err(FundedCaptureError::Protocol(
                    CaptureProtocolError::GeneratedSource
                ))
            ));
        })
        .unwrap();
    assert_eq!(calls.get(), 0);
    assert_eq!(backend.calls, 0);
    assert_eq!(
        funded.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
}

#[test]
fn empty_plan_uses_no_frame_but_still_requires_terminal_drain_and_fresh_phase() {
    let mut raw = raw();
    raw.selections.clear();
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    forward(&mut funded, &mut backend, 0, true).unwrap();
    assert!(funded.has_pending_step());
    assert!(funded.take_shared_step().unwrap().is_none());
    forward(&mut funded, &mut backend, 1, true).unwrap();
    assert!(funded.take_shared_step().unwrap().is_none());
    assert_eq!(backend.calls, 0);
    assert_eq!(funded.spent_steps(), 0);
    assert_eq!(funded.usage(), CaptureUsage::default());
}

#[test]
fn schedule_skipped_generated_hook_is_lazy_and_delivers_metadata_only() {
    let mut raw = raw();
    let skipped = raw.selections.remove(1);
    raw.selections = vec![skipped];
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut funded = session(&source, &pool, &r, &run);
    let mut backend = Backend::default();
    let calls = Cell::new(0);
    funded
        .with_observer(&mut backend, 0, &|e| e, |o| {
            assert!(
                !o.requires_sequence_readout(),
                "skipped prefill is known before prepare"
            );
            let e = DistributedCommitEpoch::FIRST;
            o.prepare_transaction(e, ExpertPass::Prefill).unwrap();
            let source = GeneratedCaptureSource {
                creation_bytes: 64,
                source_dtype: Some(TensorDtype::F32),
            };
            let mut generate = || {
                calls.set(calls.get() + 1);
                Ok(tensor(0))
            };
            o.observe_generated("block.output", &tensor(0), &source, &mut generate)
                .unwrap();
            o.complete_transaction(e).unwrap();
            o.finish_transaction(e, true);
        })
        .unwrap();
    assert_eq!(calls.get(), 0);
    assert_eq!(backend.calls, 0);
    let step = funded.take_shared_step().unwrap().unwrap();
    assert!(matches!(
        step.records()[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    assert_eq!(step.step_usage().captures, 0);
}

#[test]
fn shared_step_policy_matches_funded_delivery_and_preparation_readout() {
    use crate::capture::{CaptureObservationStep, CaptureRecordStatus};
    for mode in [CaptureLimitPolicy::Fail, CaptureLimitPolicy::Skip] {
        let mut raw = raw();
        raw.limits.on_limit = mode;
        // Metadata captures cost zero; this permits two actual outputs total.
        raw.limits.cumulative.captures = 2;
        let source = admit(raw, point(), 4, false);
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut funded = session(&source, &pool, &r, &run);
        let mut backend = Backend::default();
        let mut cold = CaptureLedger::new(source.admission());
        for p in 0..4 {
            let phase = if p == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            };
            let policy = CaptureObservationStep::new(source.admission(), phase, p).unwrap();
            funded
                .with_observer(&mut backend, p, &|e| e, |observer| {
                    // Available before prepare_transaction, without a frame/epoch.
                    assert_eq!(
                        observer.requires_sequence_readout(),
                        policy.requires_sequence_readout()
                    );
                    assert_eq!(observer.requires_sequence_readout(), p == 0);
                })
                .unwrap();
            cold.begin_step();
            policy.reserve_metadata(&mut cold).unwrap();
            let mut expected_error = false;
            for index in 0..source.admission().plan().selections.len() {
                let initial = policy.initial_status(index).unwrap();
                if !policy.select(index, initial, "block.output").unwrap() {
                    continue;
                }
                let geometry = policy.tensor_geometry(index).unwrap();
                match policy.reserve_value(&mut cold, usage(geometry.elements())) {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        assert!(matches!(
                            policy.select(index, CaptureRecordStatus::Consumed, "block.output"),
                            Err(CaptureProtocolError::Duplicate)
                        ));
                    }
                    Err(CaptureError::Limit { .. }) => {
                        expected_error = true;
                        break;
                    }
                    Err(e) => panic!("unexpected {e}"),
                }
            }
            let result = forward(&mut funded, &mut backend, p, true);
            assert_eq!(result.is_err(), expected_error);
            assert_eq!(funded.usage(), cold.total());
            let delivered = funded.take_shared_step().unwrap().unwrap();
            assert_eq!(delivered.cumulative_usage(), cold.total());
            assert_eq!(delivered.step_usage(), cold.step());
            drop(delivered);
        }
    }
}

#[test]
fn funded_checkpoint_inherits_quota_and_fresh_absolute_claims_without_rewinding_source() {
    use crate::capture::FundedCaptureCheckpointError;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Drops(Arc<AtomicUsize>);
    impl Drop for Drops {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.limits.cumulative.captures = 1;
    raw.limits.on_limit = CaptureLimitPolicy::Skip;
    let source = admit(raw.clone(), point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, funding) = fresh(&pool, h);
    let mut parent = session(&source, &pool, &reservation, &funding);
    let mut backend = Backend::default();
    forward(&mut parent, &mut backend, 0, true).unwrap();
    assert!(matches!(
        parent.prepare_checkpoint(),
        Err(FundedCaptureCheckpointError::Boundary)
    ));
    drop(parent.take_shared_step().unwrap());
    let drops = Arc::new(AtomicUsize::new(0));
    let host = HostPreparationAuthority::retain(Drops(drops.clone()));
    let checkpoint = parent.prepare_checkpoint().unwrap();
    assert!(checkpoint.required_bytes().unwrap() > 0);
    let checkpoint = checkpoint.construct(&host).unwrap();
    drop(host);
    assert_eq!(checkpoint.next_prediction(), 1);
    assert_eq!(checkpoint.inherited_usage().captures, 1);
    assert_eq!(parent.spent_steps(), 1);
    assert!(checkpoint.continuation_host_plan(5).is_err());

    // Every attempt needs actual independent funding. Refusal precedes the
    // claim table allocation; saved state and original spent rows do not change.
    let short_bytes = checkpoint
        .continuation_host_plan(4)
        .unwrap()
        .initialization_peak_bytes();
    let child_pool = WorkingMemoryPool::new(short_bytes, 0).unwrap();
    let (short_reservation, short_funding) = fresh(&child_pool, short_bytes - 1);
    let allocations = CLAIM_ALLOCATIONS.get();
    assert!(
        short_funding
            .prepare_capture_run(
                &short_reservation,
                checkpoint.continuation_host_plan(4).unwrap(),
            )
            .is_err()
    );
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    assert_eq!(checkpoint.next_prediction(), 1);
    drop(short_funding);
    drop(short_reservation);

    let (child_reservation, child_funding) = fresh(&child_pool, short_bytes);
    let bank = child_funding
        .prepare_capture_run(
            &child_reservation,
            checkpoint.continuation_host_plan(4).unwrap(),
        )
        .unwrap();
    assert_eq!(bank.first_prediction(), 1);
    assert_eq!(bank.spent_steps(), 0);
    let mut child = checkpoint.into_continuation(bank).unwrap();
    let mut child_backend = Backend::default();
    forward(&mut child, &mut child_backend, 1, true).unwrap();
    let child_frame = child.take_shared_step().unwrap().unwrap();
    assert_eq!(child_frame.as_ref().prediction_index, 1);
    assert!(matches!(
        child_frame.as_ref().records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit { .. }
        }
    ));
    assert_eq!(
        child_backend.calls, 0,
        "saved cumulative quota prevents another transfer"
    );
    assert_eq!(child.usage().captures, 1);
    assert_eq!(child.spent_steps(), 1);
    assert_eq!(parent.spent_steps(), 1);
    assert_eq!(checkpoint.inherited_usage().captures, 1);
    assert!(
        forward(&mut child, &mut child_backend, 1, true).is_err(),
        "repeating a coordinate does not restore its spent claim"
    );

    // Identical declarations are not the same physical capture source.
    let other_source = admit(raw, point(), 4, false);
    let other_h = plan(&other_source).initialization_peak_bytes();
    let other_pool = WorkingMemoryPool::new(other_h, 0).unwrap();
    let (other_reservation, other_funding) = fresh(&other_pool, other_h);
    let mut other = session(
        &other_source,
        &other_pool,
        &other_reservation,
        &other_funding,
    );
    forward(&mut other, &mut Backend::default(), 0, true).unwrap();
    drop(other.take_shared_step().unwrap());
    let other_checkpoint = other
        .prepare_checkpoint()
        .unwrap()
        .construct(&HostPreparationAuthority::default())
        .unwrap();
    let other_plan = other_checkpoint.continuation_host_plan(4).unwrap();
    let other_child_h = other_plan.initialization_peak_bytes();
    let other_child_pool = WorkingMemoryPool::new(other_child_h, 0).unwrap();
    let (other_child_reservation, other_child_funding) = fresh(&other_child_pool, other_child_h);
    let other_bank = other_child_funding
        .prepare_capture_run(&other_child_reservation, other_plan)
        .unwrap();
    assert_eq!(other_bank.first_prediction(), checkpoint.next_prediction());
    assert!(matches!(
        checkpoint.into_continuation(other_bank),
        Err(FundedCaptureCheckpointError::Host(
            CaptureRunHostError::Coordinate
        ))
    ));
    drop(parent);
    drop(funding);
    drop(reservation);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(checkpoint);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(
        child.usage().captures,
        1,
        "child keeps its own paid bank and inherited ledger"
    );
}

#[test]
fn funded_capture_restore_keeps_post_snapshot_spending_and_fork_is_independent() {
    use crate::capture::FundedCaptureCheckpoint;
    fn destination(
        saved: &FundedCaptureCheckpoint,
        restore: bool,
    ) -> (
        WorkingMemoryPool,
        WorkingMemoryReservation,
        WorkingMemoryFundingRun,
        FundedCaptureSession,
    ) {
        let plan = saved.continuation_host_plan(4).unwrap();
        let h = plan.initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (reservation, funding) = fresh(&pool, h);
        let bank = funding.prepare_capture_run(&reservation, plan).unwrap();
        let session = if restore {
            saved.into_restoration(bank).unwrap()
        } else {
            saved.into_continuation(bank).unwrap()
        };
        (pool, reservation, funding, session)
    }
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.limits.cumulative.captures = 2;
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (reservation, funding) = fresh(&pool, h);
    let mut parent = session(&source, &pool, &reservation, &funding);
    let mut backend = Backend::default();
    forward(&mut parent, &mut backend, 0, true).unwrap();
    drop(parent.take_shared_step().unwrap());
    let saved = parent
        .prepare_checkpoint()
        .unwrap()
        .construct(&HostPreparationAuthority::default())
        .unwrap();
    assert_eq!(saved.inherited_usage().captures, 1);
    forward(&mut parent, &mut backend, 1, true).unwrap();
    drop(parent.take_shared_step().unwrap());
    assert_eq!(saved.current_usage().unwrap().captures, 2);
    assert_eq!(
        saved.inherited_usage().captures,
        1,
        "saved frontier is immutable"
    );

    let (restore_pool, restore_reservation, restore_funding, mut restored) =
        destination(&saved, true);
    let mut restore_backend = Backend::default();
    assert!(matches!(
        forward(&mut restored, &mut restore_backend, 1, true),
        Err(FundedCaptureError::Admission(CaptureError::Limit {
            cumulative: true,
            ..
        }))
    ));
    assert_eq!(
        restore_backend.calls, 0,
        "post-snapshot spending refuses before transfer"
    );
    drop(restored.take_shared_step().unwrap());
    let (again_pool, again_reservation, again_funding, mut again) = destination(&saved, true);
    assert!(forward(&mut again, &mut restore_backend, 1, true).is_err());
    assert_eq!(
        restore_backend.calls, 0,
        "serial restoration cannot refund the remaining allowance"
    );
    drop(again.take_shared_step().unwrap());

    // Explicit independent child admission inherits the immutable saved minimum,
    // matching the existing ordinary fork contract rather than a global policy.
    let (fork_pool, fork_reservation, fork_funding, mut forked) = destination(&saved, false);
    let mut fork_backend = Backend::default();
    forward(&mut forked, &mut fork_backend, 1, true).unwrap();
    drop(forked.take_shared_step().unwrap());
    assert_eq!(fork_backend.calls, 1);
    assert_eq!(forked.usage().captures, 2);
    assert_eq!(saved.current_usage().unwrap().captures, 2);

    drop(parent);
    drop(funding);
    drop(reservation);
    assert!(
        ledger(&pool).0 > 0,
        "closed cumulative source keeps its original paying H"
    );
    drop(saved);
    assert!(
        ledger(&pool).0 > 0,
        "restored aliases retain the same cumulative source"
    );
    drop(restored);
    drop(again);
    assert_eq!(
        ledger(&pool).0,
        0,
        "last source alias retires H after its fixed owner"
    );
    drop((restore_funding, restore_reservation, restore_pool));
    drop((again_funding, again_reservation, again_pool));
    drop((forked, fork_funding, fork_reservation, fork_pool));
}

mod invocation;
