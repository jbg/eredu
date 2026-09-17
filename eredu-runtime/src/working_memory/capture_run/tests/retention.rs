//! Scalar original-account fixtures; no native byte or completion claim.
use super::*;
use crate::inspection::{PrefillChunkRetentionContext, PreparedPrefillChunkRetention};
use crate::prefill::PrefillChunk;

fn selected() -> SharedCapturePlan {
    let mut p = point();
    p.axes.as_mut().unwrap()[0].dimension = SymbolicDimension::Sequence;
    admit(raw(), p, 4, false)
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn chunk(index: u64) -> PrefillChunk {
    PrefillChunk {
        input: index..index + 1,
        position: 2 + index,
        output: OutputDemand::LastPosition.for_chunk(index == 2),
    }
}
fn epoch(n: u64) -> DistributedCommitEpoch {
    DistributedCommitEpoch::new(n).unwrap()
}
fn original(pool: &WorkingMemoryPool, bytes: u64) -> (InferenceRequest, WorkingMemoryFundingRun) {
    let geometry = geometry();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "neutral scalar original capture account");
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(5),
        4,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(bytes),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
    })
    .unwrap();
    let (r, run) = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &Admission {
                state,
                requested_positions: 9,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            pool.effective_capacity().unwrap(),
        )
        .unwrap()
        .into_funding()
        .unwrap();
    (r.into(), run)
}
fn bank(
    request: &InferenceRequest,
    run: &WorkingMemoryFundingRun,
    source: &SharedCapturePlan,
) -> PreparedCaptureRun {
    run.prepare_capture_run(request.memory_reservation().unwrap(), plan(source))
        .unwrap()
}
fn context<'a>(
    request: &'a InferenceRequest,
    chunk: &'a PrefillChunk,
    n: u64,
) -> PrefillChunkRetentionContext<'a> {
    PrefillChunkRetentionContext::new(request, chunk, epoch(n))
}
#[test]
fn bootstrap_before_claim_uses_exact_h_without_another_hold_and_short_h_rejects() {
    let source = selected();
    let h = plan(&source).initialization_peak_bytes();
    for bytes in [h - 1, h] {
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let (request, run) = original(&pool, bytes);
        let mut native = run.scope().unwrap();
        let before = ledger(&pool);
        let allocations = CLAIM_ALLOCATIONS.get();
        if bytes < h {
            assert!(matches!(
                run.prepare_capture_run(request.memory_reservation().unwrap(), plan(&source)),
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::BudgetExceeded { .. }
                ))
            ));
            assert_eq!(ledger(&pool), before);
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
        } else {
            let bank = bank(&request, &run, &source);
            let before = ledger(&pool);
            let claims = CLAIM_ALLOCATIONS.get();
            let span = chunk(0);
            let cx = context(&request, &span, 7);
            let (segment, registration) = bank
                .prefill_source_bootstrap()
                .unwrap()
                .begin_segment(&mut native, &cx)
                .unwrap();
            assert_eq!(bank.spent_steps(), 0);
            assert_eq!(ledger(&pool), before);
            assert_eq!(CLAIM_ALLOCATIONS.get(), claims);
            registration.validate_context(&cx).unwrap();
            segment.validate_native_scope(&native).unwrap();
            drop((registration, segment));
            assert!(
                bank.prefill_source_bootstrap()
                    .unwrap()
                    .begin_segment(&mut native, &cx)
                    .is_err(),
                "dropping the claim-free association never clears scope custody"
            );
            drop(bank);
        }
        native.certify().unwrap();
        drop((request, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn bootstrap_rejects_foreign_request_pool_and_wrong_fixed_chunk_before_allocation() {
    let source = selected();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(2 * h, 0).unwrap();
    let other_pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (request, run) = original(&pool, h);
    let (other, other_run) = original(&pool, h);
    let (foreign_request, foreign_run) = original(&other_pool, h);
    let bank = bank(&request, &run, &source);
    let mut native = run.scope().unwrap();
    let mut foreign = foreign_run.scope().unwrap();
    let before = ledger(&pool);
    let allocations = CLAIM_ALLOCATIONS.get();
    let span = chunk(0);
    assert!(bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&other, &span, 3))
        .is_err());
    assert!(bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut foreign, &context(&request, &span, 3))
        .is_err());
    for bad in [
        PrefillChunk {
            input: 0..2,
            ..span.clone()
        },
        PrefillChunk {
            position: 0,
            ..span.clone()
        },
        PrefillChunk {
            output: OutputDemand::Sequence,
            ..span.clone()
        },
    ] {
        assert!(bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &context(&request, &bad, 3))
            .is_err());
    }
    let unbudgeted =
        InferenceRequest::without_memory_budget(&InferenceExecutionIdentity::default(), geometry())
            .unwrap();
    assert!(bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&unbudgeted, &span, 3))
        .is_err());
    assert_eq!(ledger(&pool), before);
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    assert_eq!(bank.spent_steps(), 0);
    let accepted = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&request, &span, 3))
        .unwrap();
    let foreign_bank = foreign_run
        .prepare_capture_run(foreign_request.memory_reservation().unwrap(), plan(&source))
        .unwrap();
    let foreign_accepted = foreign_bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut foreign, &context(&foreign_request, &span, 3))
        .unwrap();
    drop((accepted, foreign_accepted, foreign_bank));
    native.certify().unwrap();
    foreign.certify().unwrap();
    drop((
        bank,
        request,
        run,
        other,
        other_run,
        foreign_request,
        foreign_run,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(other_pool.used_bytes().unwrap(), 0);
}
#[test]
fn actual_source_chunk_epoch_and_slot_cannot_be_replaced_by_equal_geometry() {
    let source = selected();
    let equal = selected();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(2 * h, 0).unwrap();
    let (request, run) = original(&pool, 2 * h);
    let mut a = bank(&request, &run, &source);
    let b = bank(&request, &run, &equal);
    let mut native = run.scope().unwrap();
    let mut sibling = run.scope().unwrap();
    let span = chunk(0);
    let cx = context(&request, &span, 5);
    let (mut segment, registration) = a
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let (other, other_registration) = b
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut sibling, &cx)
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let equal_assembly =
        CapturePrefillRowAssembly::prepare(equal.admission(), 0, geometry()).unwrap();
    segment
        .validate_prefill_fragment(&assembly.fragment(0).unwrap())
        .unwrap();
    assert!(segment
        .validate_prefill_fragment(&assembly.fragment(1).unwrap())
        .is_err());
    assert!(segment
        .validate_prefill_fragment(&equal_assembly.fragment(0).unwrap())
        .is_err());
    assert!(segment.validate_native_scope(&sibling).is_err());
    assert!(registration
        .validate_context(&context(&request, &span, 6))
        .is_err());
    for outcome in [
        None,
        Some(DistributedCommitOutcome::Committed(epoch(6))),
        Some(DistributedCommitOutcome::Aborted(epoch(5))),
    ] {
        assert!(registration.validate_commit(&cx, outcome).is_err());
    }
    registration
        .validate_commit(&cx, Some(DistributedCommitOutcome::Committed(epoch(5))))
        .unwrap();
    // Internal fixture validates association only; canonical issuance is covered
    // by the actual SessionPrefill companion, not manufactured native evidence.
    let ticket = registration.into_settled();
    let other_ticket = other_registration.into_settled();
    segment.validate_settled_ticket(&native, &ticket).unwrap();
    assert!(segment
        .validate_settled_ticket(&native, &other_ticket)
        .is_err());
    let frame = a
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    assert!(frame
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&request, &chunk(1), 6))
        .is_err());
    // The existing private foundation retirement is used only for this scalar
    // no-native fixture. Canonical A exposes no release operation.
    segment.retire_after_settled_boundary(&mut native).unwrap();
    drop((ticket, segment));
    let next = chunk(1);
    let next_context = context(&request, &next, 6);
    let (next_segment, next_registration) = frame
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &next_context)
        .unwrap();
    next_registration.validate_context(&next_context).unwrap();
    next_segment
        .validate_prefill_fragment(&assembly.fragment(1).unwrap())
        .unwrap();
    drop((frame, next_registration, next_segment, other_ticket, other));
    assert_eq!(
        a.spent_steps(),
        1,
        "later chunk bootstrap does not claim another frame"
    );
    drop((a, b));
    native.certify().unwrap();
    sibling.certify().unwrap();
    drop((request, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn escaped_registration_retains_original_h_and_channel_drop_quarantines_every_origin() {
    let source = selected();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h + 11, 0).unwrap();
    let values = pool.register_storage([(1u32, 11), (2u32, 0)]).unwrap();
    let (request, run) = original(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = bank(&request, &run, &source);
    let span = chunk(0);
    let (mut segment, registration) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&request, &span, 1))
        .unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_tensor(0).unwrap();
    let mut transfer = claim
        .prepare_with_segment_source(&mut native, &mut segment, values)
        .unwrap();
    while transfer.initialized_count() < transfer.len() {
        transfer.push_f32(0.75).unwrap();
    }
    drop(transfer.finish().unwrap());
    drop(step);
    drop((bank, segment));
    assert_eq!(
        ledger(&pool).2,
        h,
        "registration and unresolved native source slot retain scheduled H"
    );
    drop(native);
    drop((registration, request, run));
    assert_eq!(pool.used_bytes().unwrap(), h + 11);
    assert!(
        pool.pin_registered_storage([(1u32, 11), (2u32, 0)]).is_ok(),
        "unfunded source roots remain physically pinned, including the zero-byte key"
    );
}
#[test]
fn source_health_is_rechecked_before_bootstrap_and_after_committed_association() {
    let source = selected();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (request, run) = original(&pool, h);
    let bank = bank(&request, &run, &source);
    let mut native = run.scope().unwrap();
    let span = chunk(0);
    let cx = context(&request, &span, 2);
    let (segment, registration) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let bad = run.scope().unwrap();
    drop(bad);
    assert!(bank.prefill_source_bootstrap().is_err());
    assert!(registration.validate_context(&cx).is_err());
    assert!(segment.validate_native_scope(&native).is_err());
    drop((segment, registration, bank, native, request, run));
    assert_eq!(pool.used_bytes().unwrap(), h);
}

#[test]
fn last_registration_retains_original_h_after_certified_scope_teardown() {
    let source = selected();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (request, run) = original(&pool, h);
    let mut native = run.scope().unwrap();
    let bank = bank(&request, &run, &source);
    let span = chunk(0);
    let (segment, registration) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &context(&request, &span, 1))
        .unwrap();
    drop((bank, segment, source));
    // This scalar fixture submitted no native work. Positive certification drops
    // the native source slot, so it cannot be the remaining stamped-H owner.
    native.certify().unwrap();
    drop((request, run));
    let retained = ledger(&pool);
    assert_eq!(retained.0, h);
    assert_eq!(
        retained.2, h,
        "the escaped registration is the last scheduled-H owner"
    );
    assert_eq!(
        retained.3, 1,
        "only the original scheduled host scope remains"
    );
    drop(registration);
    let released = ledger(&pool);
    assert_eq!(released.0, 0);
    assert_eq!(released.2, 0);
    assert_eq!(released.3, 0);
    assert_eq!(released.4, 0);
    assert_eq!(released.5, 0);
}
