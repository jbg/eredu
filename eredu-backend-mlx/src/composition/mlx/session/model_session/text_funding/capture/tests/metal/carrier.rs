//! Real SessionPrefill, original quote/bank and canonical ticket. The sole
//! external hook uses an already settled/registered F32 source and the existing
//! whole tensor claim; this does not exercise or activate fragment transfer.
use super::super::super::observer::NativeScheduledCapture;
use super::*;
use crate::composition::mlx::replicated_text::prefill_retention_fixture::{
    PrefillFixtureResult, PrefillRetentionFixture,
};
use crate::composition::mlx::session::model_session::text_funding::{
    capture_carrier_control_bytes, copy_limits_with_work_controls, text_work_control_bytes,
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::checkpoint::TensorDtype;
use eredu_runtime::{
    capture::{FundedCaptureError, FundedCaptureSession, ScheduledCaptureBackend},
    inspection::{
        PrefillChunkRetentionContext, PreparedPrefillChunkRetention, SettledPrefillChunkRetention,
    },
    prefill::{PrefillChunk, PrefillOutcome},
    ActivationObserver, ExpertPass,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const VALUES: [f32; 6] = [1.25, -2., 3.5, -4.75, 5.25, -6.];
thread_local! {
    static DROPPING_WORK: RefCell<Option<FundedWorkOwner>> = const { RefCell::new(None) };
}
// Test-only strong probe is scoped to an already owned work value. It is
// cleared before the fixture's own work field on normal exit and unwind.
struct DroppingWork;
impl DroppingWork {
    fn new(work: &FundedWorkOwner) -> Self {
        DROPPING_WORK.with(|slot| {
            assert!(
                slot.borrow().is_none(),
                "no stale probe from another fixture"
            );
            *slot.borrow_mut() = Some(work.clone());
        });
        Self
    }
}
impl Drop for DroppingWork {
    fn drop(&mut self) {
        let owner = DROPPING_WORK.with(|slot| slot.borrow_mut().take());
        // Payload retirement must never run under the TLS RefCell loan.
        drop(owner);
    }
}
struct NativeDrop(Arc<AtomicBool>);
impl Drop for NativeDrop {
    fn drop(&mut self) {
        // This sidecar owns no Array/work. If the original work is still live,
        // native payload destruction must happen after all its RefCell loans.
        DROPPING_WORK.with(|slot| {
            if let Some(work) = slot.borrow().as_ref() {
                assert!(work.scope.try_borrow_mut().is_ok());
                assert!(work.capture.try_borrow_mut().is_ok());
            }
        });
        self.0.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug, thiserror::Error)]
#[error("original external capture failure")]
struct Original(Arc<()>);
#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    AfterTransfer,
    RetirementBusy,
}
struct Native<'a> {
    work: &'a FundedWork,
    stream: &'a Stream,
    failure: Failure,
    identity: Arc<()>,
    prepared: usize,
    transformed: usize,
    retired: usize,
}
impl ScheduledCaptureBackend for Native<'_> {
    type Tensor = Array;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        assert_eq!(context.chunk().input, 0..3);
        self.prepared += 1;
        self.work
            .prepare_capture_chunk(bootstrap, context)
            .map(Some)
            .map_err(FundedCaptureError::Backend)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        assert_eq!(ticket.chunk().input, 0..3);
        assert_eq!(
            self.work.capture_state_for_test().unwrap(),
            Some((1, 1, false))
        );
        self.retired += 1;
        if self.failure == Failure::RetirementBusy {
            FundedWork::force_capture_retirement_busy_for_test();
        }
        self.work
            .retire_capture_chunk(ticket)
            .map_err(FundedCaptureError::Backend)
    }
    fn validate_source(
        &self,
        value: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Error> {
        NativeScheduledCapture::for_test(self.work, self.stream).validate_source(value, geometry)
    }
    fn estimate(
        &self,
        value: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        NativeScheduledCapture::for_test(self.work, self.stream).estimate(value, geometry)
    }
    fn transform(
        &mut self,
        value: &Array,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        self.transformed += 1;
        let tensor = self
            .work
            .capture_settled_external(value, claim, self.stream)?;
        assert_eq!(
            self.work.capture_state_for_test().unwrap(),
            Some((1, 1, false))
        );
        if self.failure == Failure::AfterTransfer {
            return Err(Error::Other(Box::new(Original(self.identity.clone()))));
        }
        Ok(tensor)
    }
}

// A real external fixture hook, injected at the existing post-model callback.
// Its source handle is retired immediately after use, leaving only the actual
// carrier and any deliberately escaped native alias. All transaction/ticket
// methods forward to the real funded observer, never minting evidence here.
struct External<'a> {
    inner: &'a mut dyn ActivationObserver<Array, Error>,
    source: Option<Array>,
}
impl ActivationObserver<Array, Error> for External<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        self.inner.requires_prepared_traversal()
    }
    fn requires_sequence_readout(&self) -> bool {
        self.inner.requires_sequence_readout()
    }
    fn transactional(&self) -> bool {
        self.inner.transactional()
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.inner.begin_prefill_chunk(chunk)
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.inner.finish_prefill(committed);
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        self.inner.prepare_prefill_chunk_retention(context)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), Error> {
        self.inner.retire_prefill_chunk_retention(ticket)
    }
    fn prepare_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: ExpertPass,
    ) -> Result<(), Error> {
        self.inner.prepare_transaction(epoch, pass)
    }
    fn coordinate_transaction(&mut self, epoch: DistributedCommitEpoch) -> Result<(), Error> {
        self.inner.coordinate_transaction(epoch)
    }
    fn complete_transaction(&mut self, epoch: DistributedCommitEpoch) -> Result<(), Error> {
        let source = self.source.take().expect("one actual full Sequence chunk");
        let result = self.inner.observe("block.output", &source);
        drop(source);
        result?;
        self.inner.complete_transaction(epoch)
    }
    fn finish_transaction(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        self.inner.finish_transaction(epoch, committed);
    }
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Error> {
        self.inner.observe(path, value)
    }
}

struct Fixture {
    session: PrefillRetentionFixture,
    source: Option<Array>,
    plan: SharedCapturePlan,
    tokens: Arc<[i32]>,
    pool: MemoryLedger,
    reservation: WorkingMemoryReservation,
    run: WorkingMemoryFundingRun,
    request: InferenceRequest,
    _dropping_work: DroppingWork,
    work: FundedWorkOwner,
    bank: FundedCaptureSession,
    sources: RetainedStoragePublication,
    source_bytes: u64,
    required: u64,
    source_dropped: Arc<AtomicBool>,
}
fn fixture() -> Fixture {
    let stream = stream();
    let bootstrap = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let bootstrap_owner = NativeMemoryOwner::acquire(&bootstrap).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let session = PrefillRetentionFixture::load(artifact.path(), &stream).unwrap();
    let source = Array::from_slice(&VALUES, &[3, 2]);
    source.evaluated().unwrap();
    stream.synchronize().unwrap();
    let backing = source.allocation_info().unwrap().unwrap();
    let source_bytes = u64::try_from(backing.bytes())
        .unwrap()
        .checked_add(u64::try_from(backing.host_control_bytes()).unwrap())
        .unwrap();
    let source_dropped = Arc::new(AtomicBool::new(false));
    source
        .retain_allocation_owner(NativeDrop(source_dropped.clone()))
        .unwrap_or_else(|_| panic!("attach actual source lifetime probe"));
    let plan = SharedCapturePlan::new(admitted(
        vec![SymbolicDimension::Sequence, SymbolicDimension::Known(2)],
        CaptureTransform::FullTensor,
        vec![],
    ));
    let tokens: Arc<[i32]> = Arc::from([1, 2, 3]);
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 3,
        output: OutputDemand::Sequence,
    };
    let (mut state, closing) = session.quote(geometry, &plan, &tokens).unwrap();
    let mut initial = session.inventory().unwrap();
    initial.include_array(&source).unwrap();
    initial.include_capture_plan(plan.clone()).unwrap();
    let initial_bytes = initial.byte_bound().unwrap().unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let ordinary = OrdinaryPublicationPlan::for_closing_population(
        initial.generic_publication_rows(&pool).unwrap(),
        closing,
        1,
    )
    .unwrap();
    let mut workspace = state.execution_workspace.take().unwrap();
    let controls = text_work_control_bytes(geometry.max_output_tokens)
        .unwrap()
        .checked_add(capture_carrier_control_bytes(&plan, None).unwrap())
        .unwrap()
        .checked_add(ordinary.additional_bytes())
        .unwrap();
    workspace.retained = WorkspaceBound::bounded(
        workspace
            .retained
            .bytes()
            .unwrap()
            .checked_add(controls)
            .unwrap(),
        "actual scheduled host plan plus measured original work/carrier controls",
    );
    workspace
        .physical_domains
        .as_mut()
        .unwrap()
        .retained
        .add_allocation(controls, &bootstrap.host_placement_handle())
        .unwrap();
    let state = state.with_execution_workspace(workspace).unwrap();
    let admission = match eredu_core::apply_admission_policy(
        session.capabilities(),
        AdmissionRequest {
            input: InputTokenCount::text(3),
            max_output_tokens: 4,
            batch_size: 1,
            additional_headroom: crate::memory_fixture::headroom(0),
            memory_limits: Default::default(),
        },
        state,
    )
    .unwrap()
    {
        AdmissionResult::Admitted(admission) => admission,
        AdmissionResult::Rejected(reason) => {
            panic!("actual native equation admission rejected: {reason:?}")
        }
    };
    let required = admission.incremental_required_bytes.unwrap();
    assert!(
        required
            > CaptureRunHostPlan::prepare(&plan)
                .unwrap()
                .initialization_peak_bytes()
    );

    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let sources = initial.publish_unquoted(&owner).unwrap();
    drop((owner, bootstrap_owner));
    let before = pool.snapshot().unwrap();
    let existing = pool.fixture_host_current().unwrap();
    assert!(pool.fixture_host_charge().unwrap() >= initial_bytes);
    let increment = pool
        .reservation_requirements(&admission, None)
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let capacity = existing.checked_add(increment).unwrap();
    assert!(matches!(
        pool.reserve_with_capacity(
            session.identity(),
            &admission,
            crate::memory_fixture::physical_host_limits(&pool, capacity - 1)
        ),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    session.validate_frontier(0).unwrap();
    let (reservation, run) = pool
        .reserve_with_capacity(
            session.identity(),
            &admission,
            crate::memory_fixture::physical_host_limits(&pool, capacity),
        )
        .unwrap()
        .into_funding()
        .unwrap();
    // Host capture records can retire independently after failure. The native
    // envelope remains charged until its own completion has been established.
    let required = required
        .checked_sub(
            CaptureRunHostPlan::prepare(&plan)
                .unwrap()
                .initialization_peak_bytes(),
        )
        .unwrap();
    let request = InferenceRequest::from(&reservation);
    let bank = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&plan).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let work = FundedWork::new_ordinary(run.scope().unwrap(), ordinary).unwrap();
    let dropping_work = DroppingWork::new(&work);
    bank.validate_native_scope(work.scope.borrow().as_ref().unwrap())
        .unwrap();
    Fixture {
        session,
        source: Some(source),
        plan,
        tokens,
        pool,
        reservation,
        run,
        request,
        _dropping_work: dropping_work,
        work,
        bank,
        sources,
        source_bytes,
        required,
        source_dropped,
    }
}
fn execute(
    f: &mut Fixture,
    failure: Failure,
    cancelled: bool,
) -> (
    Result<PrefillFixtureResult, Error>,
    (usize, usize, usize),
    Arc<()>,
) {
    let stream = stream();
    let mut native = Native {
        work: &f.work,
        stream: &stream,
        failure,
        identity: Arc::new(()),
        prepared: 0,
        transformed: 0,
        retired: 0,
    };
    let cancellation = GenerationCancellationToken::new();
    if cancelled {
        cancellation.cancel();
    }
    let source = f.source.take();
    let result = f
        .bank
        .with_observer(
            &mut native,
            0,
            &|error| Error::Other(Box::new(error)),
            |observer| {
                let mut external = External {
                    inner: observer,
                    source,
                };
                f.session.run(
                    f.request.clone(),
                    f.tokens.clone(),
                    cancellation,
                    &mut external,
                )
            },
        )
        .unwrap();
    stream.synchronize().unwrap();
    (
        result,
        (native.prepared, native.transformed, native.retired),
        native.identity,
    )
}
fn publish(f: &Fixture, scores: Option<&Array>) {
    let mut storage = f.work.prepare_inventory().unwrap();
    f.session.collect_inventory(&mut storage).unwrap();
    if let Some(scores) = scores {
        scores.evaluated().unwrap();
        storage.include_array(scores).unwrap();
    }
    f.work.publish(storage).unwrap();
}
fn captured(step: &SharedCapturedStep) {
    assert_eq!(step.phase(), CapturePhase::Prefill);
    assert_eq!(step.prediction_index(), 0);
    assert_eq!(step.records().len(), 1);
    let record = &step.records()[0];
    assert_eq!(record.outcome, CaptureOutcome::Captured);
    assert_eq!(record.source_shape.as_deref(), Some([3, 2].as_slice()));
    assert_eq!(record.source_dtype, Some(TensorDtype::F32));
    assert_eq!(
        record.payload.as_ref().unwrap().as_tensor().unwrap().data(),
        &TensorObservationData::F32(VALUES.to_vec())
    );
}
fn original<'a>(mut error: &'a (dyn std::error::Error + 'static)) -> Option<&'a Original> {
    loop {
        if let Some(error) = error.downcast_ref() {
            return Some(error);
        }
        error = error.source()?;
    }
}

#[test]
fn canonical_native_prefill_retires_source_outside_loans_before_whole_work_certification() {
    let mut f = fixture();
    let (result, calls, _) = execute(&mut f, Failure::None, false);
    let result = result.unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Complete);
    assert_eq!(calls, (1, 1, 1));
    assert_eq!(f.bank.spent_steps(), 1);
    f.session.validate_frontier(3).unwrap();
    assert!(f.work.capture_state_for_test().unwrap().is_none());
    assert!(
        f.work.scope.borrow().is_some(),
        "ticket never certifies whole native work"
    );
    reclaim(&f.pool);
    assert!(
        f.source_dropped.load(Ordering::SeqCst),
        "carrier was the final native source owner"
    );
    let step = f.bank.take_shared_step().unwrap().unwrap();
    assert_eq!(step.outcome(), CaptureStepOutcome::Committed);
    captured(&step);
    publish(&f, result.scores.as_ref());
    f.work.certify().unwrap();
    assert!(f.work.scope.borrow().is_none());
    assert!(f.pool.fixture_host_peak().unwrap() <= f.pool.fixture_host_limit().unwrap());
    let pool = f.pool.clone();
    drop((result, step, f));
    DROPPING_WORK.with(|slot| assert!(slot.borrow().is_none()));
    settled_bytes(&pool, 0);
}

#[test]
fn escaped_native_alias_and_shared_frame_retain_their_actual_original_owners() {
    let mut f = fixture();
    let alias = f.source.as_ref().unwrap().clone();
    let (result, calls, _) = execute(&mut f, Failure::None, false);
    let result = result.unwrap();
    assert_eq!(calls, (1, 1, 1));
    assert!(!f.source_dropped.load(Ordering::SeqCst));
    assert_eq!(
        alias
            .evaluated()
            .unwrap()
            .try_iter::<f32>()
            .unwrap()
            .collect::<Vec<_>>(),
        VALUES
    );
    let frame = f.bank.take_shared_step().unwrap().unwrap();
    let second = frame.clone();
    assert!(frame.same_storage(&second));
    captured(&frame);
    publish(&f, result.scores.as_ref());
    f.work.certify().unwrap();
    let pool = f.pool.clone();
    let bytes = f.source_bytes;
    let dropped = f.source_dropped.clone();
    let host = CaptureRunHostPlan::prepare(&f.plan)
        .unwrap()
        .initialization_peak_bytes();
    drop((result, f, frame));
    // The completed frame retains H, and the independently escaped native
    // source retains exactly its already measured physical allocation charge.
    // This fixture creates no original text-control guard; transient native
    // workspace and work/carrier controls retire with the completed work.
    settled_bytes(&pool, host.checked_add(bytes).unwrap());
    assert!(!dropped.load(Ordering::SeqCst));
    captured(&second);
    drop(second);
    settled_bytes(&pool, bytes);
    assert!(!dropped.load(Ordering::SeqCst));
    drop(alias);
    settled_bytes(&pool, 0);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn initial_cancellation_creates_no_carrier_frame_or_source_claim() {
    let mut f = fixture();
    let (result, calls, _) = execute(&mut f, Failure::None, true);
    let result = result.unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Cancelled);
    assert_eq!(calls, (0, 0, 0));
    assert!(result.scores.is_none());
    f.session.validate_frontier(0).unwrap();
    assert!(f.work.capture_state_for_test().unwrap().is_none());
    assert!(f.bank.take_shared_step().unwrap().is_none());
    assert_eq!(f.bank.spent_steps(), 0);
    publish(&f, None);
    f.work.certify().unwrap();
    let pool = f.pool.clone();
    drop((result, f));
    settled_bytes(&pool, 0);
}

#[test]
fn late_native_hook_failure_preserves_original_cause_and_uncertified_source_custody() {
    let mut f = fixture();
    let (result, calls, identity) = execute(&mut f, Failure::AfterTransfer, false);
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("late hook must fail"),
    };
    assert!(Arc::ptr_eq(&original(&error).unwrap().0, &identity));
    assert_eq!(calls, (1, 1, 0));
    assert_eq!(
        f.work.capture_state_for_test().unwrap(),
        Some((1, 1, false))
    );
    assert!(!f.source_dropped.load(Ordering::SeqCst));
    let step = f.bank.take_shared_step().unwrap().unwrap();
    assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
    publish(&f, None);
    assert!(
        !f.work.published.get(),
        "active carrier cannot be marked fully published"
    );
    assert!(f.work.certify().is_err());
    assert!(f.work.scope.borrow().is_some());
    let pool = f.pool.clone();
    let required = f.required;
    let source_bytes = f.source_bytes;
    drop((step, error, f));
    reclaim(&pool);
    assert!(
        pool.fixture_host_current().unwrap() >= required + source_bytes,
        "unresolved carrier preserves its full native bound and source registration"
    );
}

#[test]
fn retirement_busy_aborts_sealed_frame_and_keeps_complete_original_carrier() {
    let mut f = fixture();
    let (result, calls, _) = execute(&mut f, Failure::RetirementBusy, false);
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("Busy must reject retirement"),
    };
    assert!(caused_by::<super::super::super::carrier::CaptureCarrierError>(&error));
    assert_eq!(calls, (1, 1, 1));
    assert_eq!(f.bank.spent_steps(), 1);
    f.session.validate_frontier(3).unwrap();
    assert_eq!(
        f.work.capture_state_for_test().unwrap(),
        Some((1, 1, false))
    );
    assert!(!f.source_dropped.load(Ordering::SeqCst));
    let step = f.bank.take_shared_step().unwrap().unwrap();
    assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
    captured(&step);
    publish(&f, None);
    assert!(
        !f.work.published.get(),
        "active carrier cannot be marked fully published"
    );
    assert!(f.work.certify().is_err());
    let pool = f.pool.clone();
    let required = f.required;
    let source_bytes = f.source_bytes;
    drop((step, error, f));
    reclaim(&pool);
    assert!(pool.fixture_host_current().unwrap() >= required + source_bytes);
}

#[test]
fn measured_carrier_controls_reject_arithmetic_overflow_without_native_work() {
    let before = EVALUATIONS.get();
    let mut limits = WorkspaceCopyLimits::new(crate::memory_fixture::limits(u64::MAX));
    limits.additional_host_metadata_bytes = u64::MAX;
    for error in [
        text_work_control_bytes(u64::MAX).unwrap_err(),
        copy_limits_with_work_controls(limits).unwrap_err(),
    ] {
        let Error::Other(cause) = error else {
            panic!("typed working-memory error")
        };
        assert!(matches!(
            cause.downcast_ref::<WorkingMemoryError>(),
            Some(WorkingMemoryError::Overflow)
        ));
    }
    assert_eq!(EVALUATIONS.get(), before);
}

#[path = "carrier/rows.rs"]
mod rows;

#[test]
fn scoped_closed_work_probe_resets_on_unwind_before_final_real_work_alias() {
    let mut f = fixture();
    let (result, calls, _) = execute(&mut f, Failure::None, false);
    let result = result.unwrap();
    assert_eq!(calls, (1, 1, 1));
    reclaim(&f.pool);
    assert!(f.source_dropped.load(Ordering::SeqCst));
    let frame = f.bank.take_shared_step().unwrap().unwrap();
    captured(&frame);
    publish(&f, result.scores.as_ref());
    f.work.certify().unwrap();
    let work = f.work.clone();
    let pool = f.pool.clone();
    let marker = Arc::new(());
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let marker = marker.clone();
        move || {
            let _owners = (result, frame, f);
            std::panic::panic_any(marker);
        }
    }))
    .unwrap_err();
    assert!(Arc::ptr_eq(
        payload.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    DROPPING_WORK.with(|slot| assert!(slot.borrow().is_none()));
    assert!(work.scope.try_borrow_mut().unwrap().is_none());
    assert!(work.capture.try_borrow_mut().unwrap().is_none());
    drop(work);
    settled_bytes(&pool, 0);
}

#[test]
fn closed_failed_carrier_survives_work_alias_unwind_without_refunding_original_scope() {
    for failure in [Failure::AfterTransfer, Failure::RetirementBusy] {
        let mut f = fixture();
        let (result, calls, identity) = execute(&mut f, failure, false);
        let error = result.err().expect("actual late capture failure");
        match failure {
            Failure::AfterTransfer => {
                assert_eq!(calls, (1, 1, 0));
                assert!(Arc::ptr_eq(&original(&error).unwrap().0, &identity));
            }
            Failure::RetirementBusy => {
                assert_eq!(calls, (1, 1, 1));
                assert!(caused_by::<super::super::super::carrier::CaptureCarrierError>(&error));
            }
            Failure::None => unreachable!(),
        }
        let frame = f.bank.take_shared_step().unwrap().unwrap();
        assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
        match failure {
            Failure::AfterTransfer => {
                assert_eq!(frame.records().len(), 1);
                assert!(matches!(
                    frame.records()[0].outcome,
                    CaptureOutcome::Failed {
                        reason: CaptureFailureReason::Native,
                        ..
                    }
                ));
                assert!(frame.records()[0].payload.is_none());
                assert_eq!(
                    frame.records()[0].source_shape.as_deref(),
                    Some([3, 2].as_slice())
                );
            }
            Failure::RetirementBusy => captured(&frame),
            Failure::None => unreachable!(),
        }
        publish(&f, None);
        assert!(!f.work.published.get());
        let work = f.work.clone();
        let pool = f.pool.clone();
        let required = f.required + f.source_bytes;
        let dropped = f.source_dropped.clone();
        let marker = Arc::new(());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let marker = marker.clone();
            move || {
                let _owners = (f, frame, error);
                std::panic::panic_any(marker);
            }
        }))
        .unwrap_err();
        assert!(Arc::ptr_eq(
            panic.downcast_ref::<Arc<()>>().unwrap(),
            &marker
        ));
        DROPPING_WORK.with(|slot| assert!(slot.borrow().is_none()));
        assert_eq!(work.capture_state_for_test().unwrap(), Some((1, 1, false)));
        assert!(!dropped.load(Ordering::SeqCst));
        assert!(work.certify().is_err());
        assert!(work.scope.try_borrow_mut().unwrap().is_some());
        assert!(pool.fixture_host_current().unwrap() >= required);
        drop(work);
        reclaim(&pool);
        // Closing the carrier allocation cannot certify its aborted source
        // segment or turn failed native work into a reusable/refunded account.
        assert!(pool.fixture_host_current().unwrap() >= required);
    }
}
