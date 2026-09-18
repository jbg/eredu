//! Real native SessionPrefill row composition. Gateway/readiness stays disabled.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity as Key;
use crate::composition::mlx::replicated_text::{NativeOpeningRows, NativeOpeningRowsOwner};
use crate::composition::mlx::session::model_session::text_error;
use eredu_runtime::layered::PreparedCaptureSelection;
use std::{collections::BTreeMap, mem::size_of};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cut {
    None,
    CancelFinal,
    NativeBusy,
    AttachmentPrefix,
    HandoffBusy,
}
struct Joined {
    session: PrefillRetentionFixture,
    selected: PreparedCaptureSelection,
    source: SharedCapturePlan,
    work: FundedWorkOwner,
    error_allowance: Option<text_error::OriginalErrorAllowance>,
    rows: NativeOpeningRowsOwner,
    bank: FundedCaptureSession,
    owned: OwnedTextSpanWorkspace,
    reservation: WorkingMemoryReservation,
    run: WorkingMemoryFundingRun,
    sources: RetainedStoragePublication,
    pool: WorkingMemoryPool,
    tokens: Arc<[i32]>,
    exact: u64,
    // Streamed weights reopen this test-owned artifact during actual execution.
    _artifact: tempfile::TempDir,
}
fn geometry(chunk: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 1,
        prefill_chunk_positions: chunk,
        output: OutputDemand::LastPosition,
    }
}
fn options(kind: usize) -> crate::MlxLoadRequest {
    let weights = match kind {
        0 => eredu_runtime::WeightResidency::fully_resident(),
        1 => eredu_runtime::WeightResidency::layerwise_host(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 1)
                    .unwrap(),
            ),
        ),
        _ => eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 28, 0, 0, 0).unwrap(),
        ),
    };
    crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(weights),
    )
}
fn joined(kind: usize, chunk: u64) -> Joined {
    build(kind, chunk, true)
}
fn build(kind: usize, chunk: u64, install: bool) -> Joined {
    build_probe(kind, chunk, install, false).0
}
fn build_probe(kind: usize, chunk: u64, install: bool, foreign: bool) -> (Joined, Option<Error>) {
    let stream = stream();
    let bootstrap = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let bootstrap_owner = NativeMemoryOwner::acquire(&bootstrap).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let session =
        PrefillRetentionFixture::load_with_options(artifact.path(), &stream, options(kind))
            .unwrap();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let initial = session.inventory().unwrap();
    let initial_bytes = initial.byte_bound().unwrap().unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    // Original C is deliberately absent. Only real loaded owners are registered.
    let sources = initial.publish_unquoted(&owner).unwrap();
    drop((owner, bootstrap_owner));
    assert_eq!(pool.used_bytes().unwrap(), initial_bytes);
    let g = geometry(chunk);
    let tokens: Arc<[i32]> = Arc::from([1, 2, 3, 4, 5]);
    let source = session.opening_source(g);
    let selected = session.opening_selection(&source).unwrap();
    let bound = selected.bind_geometry(g).unwrap();
    let proposal = session.opening_rows_proposal(bound).unwrap();
    let q = session
        .opening_rows_quote(&pool, g, &source, &tokens)
        .unwrap();
    let c = PreparedCapturePlanPublication::prepare(
        &pool,
        q.span_workspace().plan(),
        &source,
        Key::CapturePlan(source.storage_identity().clone()),
        None,
    )
    .unwrap();
    let (sealed, q) = proposal
        .seal(
            q,
            TextHostControlFacts::new(
                Some(
                    ((size_of::<Joined>() + size_of::<PreparedCaptureSelection>()) as u64)
                        .checked_add(text_error::control_peak_bytes().unwrap())
                        .unwrap(),
                ),
                Some(capture_carrier_control_bytes(&source, None).unwrap()),
                Some(text_work_control_bytes(g.max_output_tokens).unwrap()),
            ),
            c,
        )
        .unwrap();
    let rejected_plan = if foreign {
        // A fresh actual equation report has equal numeric facts but a distinct
        // unpromoted SpanPlan. It must not borrow the accepted plan's attachment.
        let other_quote = session
            .opening_rows_quote(&pool, g, &source, &tokens)
            .unwrap();
        let other_c = PreparedCapturePlanPublication::prepare(
            &pool,
            other_quote.span_workspace().plan(),
            &source,
            Key::CapturePlan(source.storage_identity().clone()),
            None,
        )
        .unwrap();
        let other = session.opening_rows_proposal(bound).unwrap();
        let (other, other_quote) = other
            .seal(
                other_quote,
                TextHostControlFacts::new(
                    Some(
                        ((size_of::<Joined>() + size_of::<PreparedCaptureSelection>()) as u64)
                            .checked_add(text_error::control_peak_bytes().unwrap())
                            .unwrap(),
                    ),
                    Some(capture_carrier_control_bytes(&source, None).unwrap()),
                    Some(text_work_control_bytes(g.max_output_tokens).unwrap()),
                ),
                other_c,
            )
            .unwrap();
        assert_eq!(other_quote.incremental_bytes(), q.incremental_bytes());
        drop(other_quote);
        Some(other)
    } else {
        None
    };
    let exact = initial_bytes.checked_add(q.incremental_bytes()).unwrap();
    let request = AdmissionRequest {
        input: InputTokenCount::text(5),
        max_output_tokens: 1,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    // First prove this exact quote passes core policy. The domain-capacity
    // probe below must then reject it through the original residual binding,
    // including registered preparation/source credit on bounded weight routes.
    let incremental =
        WorkspaceBound::bounded(q.incremental_bytes(), "actual complete original quote");
    let admission = match eredu_core::apply_admission_policy_with_incremental(
        session.capabilities(),
        request.clone(),
        q.state().clone(),
        &incremental,
        None,
    )
    .unwrap()
    {
        AdmissionResult::Admitted(a) => a,
        other => panic!("actual joined quote: {other:?}"),
    };
    assert_eq!(admission.incremental_required_bytes, q.incremental_bytes());
    let mut candidates = Vec::new();
    let rejection = plan_prefill_incremental_with_capacity(
        session.identity(),
        &pool,
        session.capabilities(),
        request.clone(),
        g,
        exact - 1,
        |candidate| {
            candidates.push(candidate);
            if candidate == g {
                Ok(q.clone())
            } else {
                // The planner reached its next candidate only after rejecting
                // the valid original quote for capacity. Stop before quoting a
                // replacement: this bank belongs to the original geometry.
                assert_eq!(
                    candidate.prefill_chunk_positions + 1,
                    g.prefill_chunk_positions
                );
                assert_eq!(pool.used_bytes().unwrap(), initial_bytes);
                Err(WorkingMemoryError::PreparedSourceUnavailable.into())
            }
        },
    );
    if g.prefill_chunk_positions == 1 {
        assert_eq!(candidates, [g]);
        assert!(matches!(
            rejection,
            Err(PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded {
                required_bytes, available_bytes,
            })) if required_bytes == q.incremental_bytes() && available_bytes + 1 == required_bytes
        ));
    } else {
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0], g);
        assert!(matches!(
            rejection,
            Err(PrefillPlanningError::Reservation(
                WorkingMemoryError::PreparedSourceUnavailable
            ))
        ));
    }
    assert_eq!(pool.used_bytes().unwrap(), initial_bytes);
    let (reservation, accepted) = plan_prefill_incremental_with_capacity(
        session.identity(),
        &pool,
        session.capabilities(),
        request,
        g,
        exact,
        |candidate| {
            assert_eq!(candidate, g, "the original candidate fits exactly");
            Ok(q.clone())
        },
    )
    .unwrap();
    drop(q);
    let (reservation, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    let (mut owned, witness) = accepted
        .begin_capture_plan_publication::<Key>(&run, &reservation, &source)
        .unwrap()
        .publish_and_finish(&scope)
        .unwrap();
    drop(witness);
    let rejection = rejected_plan.map(|other| {
        other
            .allocate(&mut owned)
            .err()
            .expect("equal facts are not the accepted binding")
    });
    // Success proves the foreign rejection did not extract either one-use bank.
    let rows = sealed.allocate(&mut owned).unwrap();
    let error_allowance =
        Some(text_error::OriginalErrorAllowance::for_rows_fixture(&owned, &rows, &source).unwrap());
    assert!(
        text_error::OriginalErrorAllowance::for_rows_fixture(&owned, &rows, &source).is_err(),
        "no second envelope from the same original row bank"
    );
    assert!(owned.take_prefill_storage_pins::<Key>().is_err());
    assert!(owned.take_prefill_storage_publications::<Key>().is_err());
    if install {
        session.install_opening_rows(&rows).unwrap();
        assert!(
            session.install_opening_rows(&rows).is_err(),
            "one installation only"
        );
    }
    let bank = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&source).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let work =
        FundedWork::new_with_opening_rows(scope, Some(owned.control_guard()), Some(rows.clone()))
            .unwrap();
    assert_eq!(pool.effective_capacity().unwrap(), exact);
    (
        Joined {
            session,
            selected,
            source,
            work,
            error_allowance,
            rows,
            bank,
            owned,
            reservation,
            run,
            sources,
            pool,
            tokens,
            exact,
            _artifact: artifact,
        },
        rejection,
    )
}
fn unique(entries: Vec<(Key, u64)>) -> BTreeMap<Key, u64> {
    let mut out = BTreeMap::new();
    for (key, n) in entries {
        if let Some(old) = out.insert(key, n) {
            assert_eq!(old, n);
        }
    }
    out
}
struct NativeRows<'a> {
    work: &'a FundedWork,
    rows: &'a NativeOpeningRows,
    pool: &'a WorkingMemoryPool,
    stream: &'a Stream,
    cancel: GenerationCancellationToken,
    cut: Cut,
    chunks: Vec<std::ops::Range<u64>>,
    completed: Vec<std::ops::Range<u64>>,
    new_native: usize,
    transformed: usize,
}
impl ScheduledCaptureBackend for NativeRows<'_> {
    type Tensor = Array;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        let (_, phase, opening, _) = self.rows.snapshot_for_test();
        assert_eq!(phase, Some("opening"));
        assert!(!opening.is_empty());
        for (key, bytes) in unique(opening) {
            // In later chunks these include the new KV published by the prior
            // end hook. Initial loaded owners are existing-only as well.
            assert!(self.pool.pin_registered_storage([(key, bytes)]).is_ok());
        }
        self.chunks.push(context.chunk().input.clone());
        self.work
            .prepare_capture_chunk(bootstrap, context)
            .map(Some)
            .map_err(FundedCaptureError::Backend)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        let (_, phase, opening, end) = self.rows.snapshot_for_test();
        assert_eq!(
            phase,
            Some("end"),
            "same-session end callback precedes observer retirement"
        );
        let opening = unique(opening);
        for (key, bytes) in unique(end) {
            if matches!(&key, Key::Native(_)) && !opening.contains_key(&key) {
                assert!(
                    self.pool.pin_registered_storage([(key, bytes)]).is_err(),
                    "new end backing has no early registry publication"
                );
                self.new_native += 1;
            }
        }
        let chunk = ticket.chunk().input.clone();
        match self.cut {
            Cut::NativeBusy => FundedWork::force_capture_retirement_busy_for_test(),
            Cut::AttachmentPrefix => NativeOpeningRows::force_attachment_busy_after_for_test(1),
            Cut::HandoffBusy => NativeOpeningRows::force_handoff_busy_for_test(),
            _ => {}
        }
        self.work
            .retire_capture_chunk(ticket)
            .map_err(FundedCaptureError::Backend)?;
        assert!(self.work.capture_state_for_test().unwrap().is_none());
        assert_eq!(self.rows.snapshot_for_test().1, None);
        self.completed.push(chunk.clone());
        if self.cut == Cut::CancelFinal && chunk.end == 5 {
            self.cancel.cancel();
        }
        Ok(())
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
        NativeScheduledCapture::for_test(self.work, self.stream).transform(value, claim)
    }
    fn validate_prefill_source(
        &self,
        value: &Array,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Error>> {
        NativeScheduledCapture::for_test(self.work, self.stream)
            .validate_prefill_source(value, fragment)
    }
    fn estimate_prefill(
        &self,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        NativeScheduledCapture::for_test(self.work, self.stream).estimate_prefill(geometry)
    }
    fn transform_prefill_fragment(
        &mut self,
        value: &Array,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.transformed += 1;
        NativeScheduledCapture::for_test(self.work, self.stream)
            .transform_prefill_fragment(value, claim)
    }
}
struct Observed {
    result: Result<PrefillFixtureResult, Error>,
    openings: Vec<std::ops::Range<u64>>,
    completed: Vec<std::ops::Range<u64>>,
    new_native: usize,
    transforms: usize,
}
fn execute(f: &mut Joined, cut: Cut, initially_cancelled: bool) -> Observed {
    let allowance = f.error_allowance.take().expect("one fixture operation");
    let stream = stream();
    let cancel = GenerationCancellationToken::new();
    if initially_cancelled {
        cancel.cancel();
    }
    let mut native = NativeRows {
        work: &f.work,
        rows: &f.rows,
        pool: &f.pool,
        stream: &stream,
        cancel: cancel.clone(),
        cut,
        chunks: vec![],
        completed: vec![],
        new_native: 0,
        transformed: 0,
    };
    let request = InferenceRequest::from(&f.reservation);
    let result = f
        .bank
        .with_prefill_observer(
            &mut native,
            f.selected.bind_geometry(request.geometry()).unwrap(),
            &|e| Error::Other(Box::new(e)),
            |observer| f.session.run(request, f.tokens.clone(), cancel, observer),
        )
        .unwrap();
    stream.synchronize().unwrap();
    let NativeRows {
        chunks: openings,
        completed,
        new_native,
        transformed: transforms,
        ..
    } = native;
    // Low-level fixture entry has no core permit; its sole original sealed
    // allowance encloses the same complete actual runtime/observer error.
    Observed {
        result: allowance.finish(result),
        openings,
        completed,
        new_native,
        transforms,
    }
}
fn finish(f: &Joined, result: &PrefillFixtureResult) {
    let mut inventory = f.session.inventory().unwrap();
    if let Some(scores) = &result.scores {
        scores.evaluated().unwrap();
        inventory.include_array(scores).unwrap();
    }
    f.work.publish(inventory).unwrap();
    f.work.certify().unwrap();
    assert!(f.work.scope.borrow().is_none());
    assert_eq!(f.pool.effective_capacity().unwrap(), f.exact);
    assert!(f.pool.peak_bytes().unwrap() <= f.exact);
}
fn close_numeric(a: &[(Vec<i32>, Vec<f32>)], b: &[(Vec<i32>, Vec<f32>)]) {
    assert_eq!(a.len(), b.len());
    for ((shape, a), (other_shape, b)) in a.iter().zip(b) {
        assert_eq!(shape, other_shape);
        assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 2e-5 + 2e-4 * x.abs(), "{x} != {y}");
        }
    }
}
fn numerical(kind: usize, chunk: u64) -> (Vec<(Vec<i32>, Vec<f32>)>, Vec<f32>, Vec<f32>) {
    let mut f = joined(kind, chunk);
    let observed = execute(&mut f, Cut::None, false);
    let result = observed.result.unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Complete);
    let expected = (0..5)
        .step_by(usize::try_from(chunk).unwrap())
        .map(|start| start..(start + chunk).min(5))
        .collect::<Vec<_>>();
    assert_eq!(observed.openings, expected);
    assert_eq!(observed.completed, expected);
    assert_eq!(observed.transforms, expected.len());
    assert!(
        observed.new_native > 0,
        "actual new KV must pass end publication"
    );
    assert_eq!(f.bank.spent_steps(), 1);
    let frame = f.bank.take_shared_step().unwrap().unwrap();
    assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(frame.records().len(), 1);
    let record = &frame.records()[0];
    assert_eq!(record.outcome, CaptureOutcome::Captured);
    assert_eq!(record.source_shape.as_deref(), Some([1, 5, 32].as_slice()));
    let values = match record.payload.as_ref().unwrap().as_tensor().unwrap().data() {
        TensorObservationData::F32(values) => values.clone(),
        _ => panic!("actual full F32 target"),
    };
    assert_eq!(values.len(), 160);
    assert!(values.iter().any(|v| *v != 0.));
    let state = f.session.numeric_state().unwrap();
    assert!(!state.is_empty());
    assert!(state.iter().flat_map(|(_, v)| v).any(|v| *v != 0.));
    let scores = result
        .scores
        .as_ref()
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    finish(&f, &result);
    let pool = f.pool.clone();
    drop((result, frame, f));
    settled_bytes(&pool, 0);
    (state, scores, values)
}
#[test]
fn actual_nonzero_two_two_one_native_rows_match_full_span_for_all_weight_routes() {
    let full = numerical(0, 5);
    for kind in 0..3 {
        // Width one also reaches the planner's terminal exact-minus-one
        // BudgetExceeded directly on each actual weight-residency route.
        for chunk in [1, 2] {
            let chunked = numerical(kind, chunk);
            close_numeric(&full.0, &chunked.0);
            close_numeric(&[(vec![], full.1.clone())], &[(vec![], chunked.1)]);
            close_numeric(&[(vec![], full.2.clone())], &[(vec![], chunked.2)]);
        }
    }
}
#[test]
fn final_cancellation_retires_all_rows_and_aborts_only_the_provisional_frame() {
    let mut f = joined(0, 2);
    let observed = execute(&mut f, Cut::CancelFinal, false);
    let result = observed.result.unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Cancelled);
    assert_eq!(observed.completed, [0..2, 2..4, 4..5]);
    assert_eq!(f.rows.snapshot_for_test().0, 3);
    assert_eq!(
        f.bank.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
    finish(&f, &result);
    let pool = f.pool.clone();
    drop((result, f));
    settled_bytes(&pool, 0);
}
#[test]
fn initial_cancellation_visits_no_row_or_native_source() {
    let mut f = joined(0, 2);
    let observed = execute(&mut f, Cut::None, true);
    let result = observed.result.unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Cancelled);
    assert!(observed.openings.is_empty());
    assert!(observed.completed.is_empty());
    assert_eq!(observed.transforms, 0);
    assert_eq!(f.rows.snapshot_for_test().0, 0);
    assert!(f.bank.take_shared_step().unwrap().is_none());
    finish(&f, &result);
    let pool = f.pool.clone();
    drop((result, f));
    settled_bytes(&pool, 0);
}
fn failed(cut: Cut, phase: &str) {
    let mut f = joined(0, 2);
    let observed = execute(&mut f, cut, false);
    assert!(observed.result.is_err());
    assert_eq!(observed.openings, [0..2]);
    assert!(observed.completed.is_empty());
    assert_eq!(observed.transforms, 1);
    let (next, actual, opening, end) = f.rows.snapshot_for_test();
    assert_eq!(next, 0);
    assert_eq!(actual, Some(phase));
    assert!(!opening.is_empty());
    assert!(!end.is_empty());
    if phase != "end" {
        for (key, n) in unique(end) {
            assert!(f.pool.pin_registered_storage([(key, n)]).is_ok());
        }
    }
    assert!(
        f.work.capture_state_for_test().unwrap().is_some(),
        "old canonical marker survives failed handoff"
    );
    assert!(f.work.certify().is_err());
    f.work.publish(f.session.inventory().unwrap()).unwrap();
    assert!(!f.work.published.get());
    assert!(f.work.certify().is_err());
    assert_eq!(
        f.bank.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
    assert_eq!(f.pool.effective_capacity().unwrap(), f.exact);
    let error = observed.result.err().expect("actual retained row failure");
    assert!(matches!(error, Error::OriginalControl(_)));
    assert!(!error.model_state_preserved());
    let snapshot = f.rows.snapshot_for_test();
    drop(error);
    assert_eq!(
        f.rows.snapshot_for_test(),
        snapshot,
        "error retirement cannot take the published prefix or canonical marker"
    );
    assert!(f.work.capture_state_for_test().unwrap().is_some());
    // Quarantine is intentional; failure/drop is never fabricated completion.
    f.owned
        .control_guard()
        .validate_native_scope(f.work.scope.borrow().as_ref().unwrap())
        .unwrap();
    f.run.validate_reservation(&f.reservation).unwrap();
}
#[test]
fn native_busy_keeps_completed_capsule_and_original_marker() {
    failed(Cut::NativeBusy, "end");
}
#[test]
fn attached_prefix_failure_keeps_all_origins_and_original_marker() {
    failed(Cut::AttachmentPrefix, "publishing");
}
#[test]
fn late_handoff_busy_preserves_published_prefix_and_original_marker() {
    failed(Cut::HandoffBusy, "published");
}
#[test]
fn equal_model_foreign_session_cannot_install_original_rows() {
    let f = joined(0, 2);
    let bootstrap = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&bootstrap).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let other = PrefillRetentionFixture::load(artifact.path(), &stream()).unwrap();
    assert!(other.install_opening_rows(&f.rows).is_err());
    assert_eq!(f.rows.snapshot_for_test().0, 0);
    assert!(f.rows.snapshot_for_test().1.is_none());
    drop((other, owner));
    // No model operation began. Existing ordinary work finalization suffices.
    let result = PrefillFixtureResult {
        outcome: PrefillOutcome::Cancelled,
        scores: None,
    };
    finish(&f, &result);
    let pool = f.pool.clone();
    drop(f);
    settled_bytes(&pool, 0);
}

struct Quiet;
impl ActivationObserver<Array, Error> for Quiet {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn observe(&mut self, _: &str, _: &Array) -> Result<(), Error> {
        Ok(())
    }
}
#[test]
fn actual_unprepared_forward_and_same_source_rebind_cannot_freshen_original_rows() {
    let mut f = build(0, 5, false);
    let result = f
        .session
        .run(
            InferenceRequest::from(&f.reservation),
            f.tokens.clone(),
            GenerationCancellationToken::new(),
            &mut Quiet,
        )
        .unwrap();
    assert_eq!(result.outcome, PrefillOutcome::Complete);
    assert!(
        f.session.install_opening_rows(&f.rows).is_err(),
        "current runtime token was invalidated"
    );
    f.session.rebind_after_unprepared().unwrap();
    let allowance = f
        .error_allowance
        .take()
        .expect("one consumed fixture allowance");
    let installation = f.session.install_opening_rows(&f.rows);
    let error = allowance
        .finish(installation)
        .err()
        .expect("same-source new generation cannot replace original binding");
    assert!(f.error_allowance.is_none());
    assert_eq!(f.rows.snapshot_for_test().0, 0);
    assert!(f.rows.snapshot_for_test().1.is_none());
    assert_eq!(f.bank.spent_steps(), 0);
    finish(&f, &result);
    let protected = f.owned.protected_host_bytes();
    let source_bytes = f.source.capacity_bytes().unwrap();
    let pool = f.pool.clone();
    drop((result, f));
    // The escaped error's validation guard retains original P+Q+finite S and
    // the completed C registration witness. Unlike a raw backing sidecar, that
    // guard must preserve source-origin validation; it owns no model/path/bank.
    settled_bytes(&pool, protected.checked_add(source_bytes).unwrap());
    drop(error);
    settled_bytes(&pool, 0);
}

#[test]
fn inactive_prefill_selection_cannot_construct_an_orphan_native_row_bank() {
    let f = build(0, 2, false);
    let before = f.pool.used_bytes().unwrap();
    for mode in 0..3 {
        let source = f.session.inactive_opening_source(geometry(2), mode);
        let selected = f.session.opening_selection(&source).unwrap();
        assert!(f
            .session
            .opening_rows_proposal(selected.bind_geometry(geometry(2)).unwrap())
            .is_err());
        assert_eq!(f.pool.used_bytes().unwrap(), before);
    }
    assert_eq!(f.rows.snapshot_for_test().0, 0);
    assert!(f.work.capture_state_for_test().unwrap().is_none());
    let result = PrefillFixtureResult {
        outcome: PrefillOutcome::Cancelled,
        scores: None,
    };
    finish(&f, &result);
    let pool = f.pool.clone();
    drop(f);
    settled_bytes(&pool, 0);
}
#[test]
fn escaped_actual_kv_alias_retains_exact_original_controls_after_terminal_trim() {
    let mut f = joined(0, 2);
    let result = execute(&mut f, Cut::None, false).result.unwrap();
    let mut aliases = f.session.state_aliases().unwrap();
    let alias = aliases.pop().expect("actual nonzero KV owner");
    drop(aliases);
    let fact = alias.try_allocation_info().unwrap().unwrap();
    let bytes = fact.bytes() as u64;
    let values = alias.evaluated().unwrap().try_to_vec::<f32>().unwrap();
    assert!(values.iter().any(|v| *v != 0.));
    let protected = f.owned.protected_host_bytes();
    let frame = f.bank.take_shared_step().unwrap().unwrap();
    let frame_alias = frame.clone();
    assert!(frame.same_storage(&frame_alias));
    finish(&f, &result);
    let pool = f.pool.clone();
    drop((frame, frame_alias, result, f));
    // The attached opaque allocation owns raw aggregate custody, not the plan,
    // model or row bank. Only this backing and the original P+Q+finite S remain.
    settled_bytes(&pool, protected.checked_add(bytes).unwrap());
    assert_eq!(
        alias.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        values
    );
    drop(alias);
    settled_bytes(&pool, 0);
}

#[test]
fn unwind_after_actual_native_attachment_keeps_prefix_and_fences_same_session() {
    let mut f = joined(0, 2);
    NativeOpeningRows::force_attachment_panic_for_test();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute(&mut f, Cut::None, false)
    }))
    .err()
    .expect("actual first attachment panic");
    assert_eq!(*panic.downcast::<u32>().unwrap(), 173);
    assert_eq!(f.rows.snapshot_for_test().1, Some("publishing"));
    assert!(f.work.capture_state_for_test().unwrap().is_some());
    assert!(f.work.certify().is_err());
    assert_eq!(
        f.bank.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
    let before = f.rows.snapshot_for_test();
    let result = f.session.run(
        InferenceRequest::from(&f.reservation),
        f.tokens.clone(),
        GenerationCancellationToken::new(),
        &mut Quiet,
    );
    assert!(
        result.is_err(),
        "same-session retry must fence before any new row"
    );
    assert_eq!(f.rows.snapshot_for_test(), before);
    f.owned
        .control_guard()
        .validate_native_scope(f.work.scope.borrow().as_ref().unwrap())
        .unwrap();
}

#[test]
fn rejected_equal_fact_plan_preserves_actual_accepted_custody_after_all_native_owners_drop() {
    let (f, rejection) = build_probe(0, 2, false, true);
    let rejection = rejection.expect("separate unpromoted binding was rejected");
    let protected = f.owned.protected_host_bytes();
    // The rejected plan intentionally retains its original shared source/path
    // payloads as well as the explicit actual accepted control guard.
    let source_bytes = f.source.capacity_bytes().unwrap();
    let path_bytes = f.selected.paths().capacity_bytes().unwrap();
    let result = PrefillFixtureResult {
        outcome: PrefillOutcome::Cancelled,
        scores: None,
    };
    finish(&f, &result);
    let pool = f.pool.clone();
    drop(f);
    settled_bytes(
        &pool,
        protected
            .checked_add(source_bytes)
            .unwrap()
            .checked_add(path_bytes)
            .unwrap(),
    );
    drop(rejection);
    settled_bytes(&pool, 0);
}

#[test]
fn original_envelope_preserves_native_witness_source_and_core_outer_retirement() {
    for preserved in [false, true] {
        let mut f = joined(0, 2);
        let native = f.session.install_opening_rows(&f.rows).err().unwrap();
        // Exercise both existing native classifications with the same actual
        // stateless duplicate-install cause. This is a wrapper/retirement test,
        // not a new proof of distributed agreement or native completion.
        let native = if preserved {
            Error::before_model_mutation(native)
        } else {
            native
        };
        let display = native.to_string();
        let allowance = f.error_allowance.take().unwrap();
        let error = allowance.finish::<()>(Err(native)).unwrap_err();
        assert_eq!(error.model_state_preserved(), preserved);
        assert_eq!(error.to_string(), display);
        let outer = eredu_core::BackendFailure::from_error(error);
        let native = std::error::Error::source(&outer)
            .unwrap()
            .downcast_ref::<Error>()
            .unwrap();
        assert_eq!(native.model_state_preserved(), preserved);
        assert_eq!(native.to_string(), display);
        let mut replay = Vec::new();
        for _ in 0..32 {
            replay.push(
                text_error::OriginalErrorAllowance::for_rows_fixture(&f.owned, &f.rows, &f.source)
                    .err()
                    .expect("the one fixture envelope was consumed"),
            );
        }
        let result = PrefillFixtureResult {
            outcome: PrefillOutcome::Cancelled,
            scores: None,
        };
        finish(&f, &result);
        let held = f
            .owned
            .protected_host_bytes()
            .checked_add(f.source.capacity_bytes().unwrap())
            .unwrap();
        let pool = f.pool.clone();
        drop((result, f));
        settled_bytes(&pool, held);
        drop(outer);
        settled_bytes(&pool, 0);
        assert_eq!(replay.len(), 32);
        drop(replay);
    }
}
