//! Actual prepared sessions and canonical callbacks. Each pin run deliberately
//! rejects before source preparation; no bounded native execution is claimed.
use super::super::super::mechanisms::{OpeningPinFailure, OpeningPinSetup, PreparedOpeningPins};
use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use crate::backend::runtime::residency::storage::StorageIdentity;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{capture::*, *};
use eredu_runtime::{
    capture::{FundedCaptureError, ScheduledCaptureBackend},
    inspection::{PrefillChunkRetentionContext, PreparedPrefillChunkRetention},
    working_memory::*,
};
use std::collections::BTreeMap;

fn geometry(cached: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: cached,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
pub(super) fn source(f: &PrefillRetentionFixture, g: InferenceGeometry) -> SharedCapturePlan {
    // A real selected prefill point is required to enter canonical retention.
    // Empty plans intentionally bypass the backend bootstrap callback.
    const PATH: &str = "readout.embedding";
    let declaration = f.session.opening_paths().prefill_observation(PATH).unwrap();
    assert_eq!(
        declaration.readout_stage(),
        eredu_runtime::layered::PrefillReadoutStage::BeforeReadout
    );
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "opening".into(),
        path: PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    plan.limits.per_step = unlimited;
    plan.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &f.discovery.catalog,
            &f.discovery.support,
            &f.discovery.support.capture,
            CaptureRequestShape {
                batch: g.batch_size,
                prompt_tokens: g.input_positions,
                max_predictions: g.max_output_tokens,
            },
            CaptureTextOrigin {
                cached_positions: g.cached_positions,
            },
        )
        .unwrap(),
    )
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
fn quote(
    f: &PrefillRetentionFixture,
    pool: &MemoryLedger,
    g: InferenceGeometry,
    source: &SharedCapturePlan,
) -> IncrementalInferenceQuote {
    // Actual family/state equations remain a conservative envelope. The tested
    // callback stops before preparing input; this is not a claim that omitted
    // layerwise materialization is covered for a successful host/disk forward.
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let storage = RegisteredWorkspaceStorage::<u32>::bind(
        pool,
        &context,
        std::iter::empty::<eredu_runtime::working_memory::RegisteredWorkspaceStorageRow<u32>>(),
    )
    .unwrap();
    let equations = f.session.quote(&f.blueprint, g, &context).unwrap();
    let state = f.session.state_estimate(g, f.dtype).unwrap();
    let state = equations.refine_state_backing(state).unwrap();
    let bound = || {
        WorkspaceBound::bounded(
            0,
            "pin callback rejects before input preparation/model/native capture work",
        )
    };
    let outside = crate::memory_fixture::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: g,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: WorkspaceBound::bounded(
            CaptureRunHostPlan::prepare(source)
                .unwrap()
                .initialization_peak_bytes(),
            "actual original scheduled bank",
        ),
    });
    let outside = quote_text_prompt_workspace(g, Some(4), &context)
        .unwrap()
        .compose(outside)
        .unwrap();
    ResidualInferenceQuote::compose(&equations, state, outside, &storage)
        .unwrap()
        .into_incremental()
}
fn unique(snapshot: &PreparedOpeningPins) -> BTreeMap<StorageIdentity, u64> {
    let mut unique = BTreeMap::new();
    for (key, bytes) in snapshot.captured_keys_for_test() {
        if let Some(old) = unique.insert(key, bytes) {
            assert_eq!(old, bytes);
        }
    }
    unique
}
fn cause<'a, T: std::error::Error + 'static>(
    mut e: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = e.downcast_ref::<T>() {
            return Some(found);
        }
        e = e.source()?;
    }
}
#[derive(Debug, thiserror::Error)]
#[error("intentional stop after snapshot pin, before input preparation")]
struct BeforeInput;

struct Pin<'a> {
    snapshot: &'a mut PreparedOpeningPins,
    scope: &'a mut WorkingMemoryFundingScope,
    result: Option<Result<(), OpeningPinFailure>>,
    segment: Option<CaptureSourceSegment>,
    panic_after: bool,
}
impl ScheduledCaptureBackend for Pin<'_> {
    type Tensor = Array;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        let (segment, _registration) = bootstrap
            .begin_segment(self.scope, context)
            .map_err(|e| FundedCaptureError::Backend(Error::Other(Box::new(e))))?;
        self.segment = Some(segment);
        self.result = Some(self.snapshot.pin_registered(
            context,
            self.scope,
            self.segment.as_ref().unwrap(),
        ));
        assert!(
            self.snapshot
                .pin_registered(context, self.scope, self.segment.as_ref().unwrap(),)
                .is_err(),
            "a completed or failed snapshot cannot issue another pin row"
        );
        if self.panic_after {
            std::panic::panic_any(17_u32);
        }
        Err(FundedCaptureError::Backend(Error::Other(Box::new(
            BeforeInput,
        ))))
    }
    fn validate_source(
        &self,
        _: &Array,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<eredu_core::checkpoint::TensorDtype, Error> {
        panic!("no tensor source hook")
    }
    fn estimate(
        &self,
        _: &Array,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("no tensor estimate")
    }
    fn transform(
        &mut self,
        _: &Array,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        panic!("no tensor transform")
    }
}
fn pin_in_real_callback(
    f: &mut PrefillRetentionFixture,
    setup: &mut OpeningPinSetup,
    source: &SharedCapturePlan,
    panic_after: bool,
) -> (Option<Result<(), OpeningPinFailure>>, CaptureSourceSegment) {
    let mut bank = setup
        .run
        .prepare_capture_run(
            &setup.reservation,
            CaptureRunHostPlan::prepare(source).unwrap(),
        )
        .unwrap()
        .into_capture_session()
        .unwrap();
    let mut native = Pin {
        snapshot: &mut setup.snapshot,
        scope: &mut setup.scope,
        result: None,
        segment: None,
        panic_after,
    };
    let request = InferenceRequest::from(&setup.reservation);
    let paths = f.session.opening_paths().clone();
    let selected = paths.prepare_capture_selection(source).unwrap();
    let bound = selected.bind_geometry(request.geometry()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bank.with_prefill_observer(
            &mut native,
            bound,
            &|e| Error::Other(Box::new(e)),
            |observer| {
                f.run(
                    request,
                    Arc::from([1]),
                    GenerationCancellationToken::new(),
                    observer,
                )
            },
        )
    }));
    if panic_after {
        let payload = result.err().expect("callback must unwind");
        assert_eq!(*payload.downcast::<u32>().unwrap(), 17);
    } else {
        let error = result
            .unwrap()
            .unwrap()
            .err()
            .expect("callback must reject before model work");
        assert!(cause::<BeforeInput>(&error).is_some());
    }
    assert_eq!(bank.spent_steps(), 0, "bootstrap/pins do not claim p0");
    let result = native.result.take();
    let segment = native.segment.take().unwrap();
    drop(native);
    drop(bank);
    (result, segment)
}
fn settle_empty_callback(stream: &Stream, scope: WorkingMemoryFundingScope) {
    // Separate native completion evidence for test cleanup: the callback error
    // itself proves nothing. No source preparation/model work was submitted.
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim();
        matches!(
            safemlx::try_retire_completed_submissions().unwrap(),
            safemlx::SubmissionRetirement::CompleteSnapshot
        )
    });
    scope.certify().unwrap();
}

#[test]
fn actual_resident_host_disk_first_opening_pins_loaded_owners_without_new_registration() {
    for kind in 0..3 {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let artifact =
            crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
        let mut f =
            PrefillRetentionFixture::load_with_options(artifact.path(), &stream, options(kind))
                .unwrap();
        let source = source(&f, geometry(0));
        let initial = f.inventory().unwrap();
        let publication = initial.publish_unquoted(&loading).unwrap();
        drop(loading);
        let q = quote(&f, &pool, geometry(0), &source);
        let mut setup = f
            .session
            .prepare_pin_snapshot(&source, q, &pool, None)
            .unwrap();
        let captured = setup.snapshot.captured_keys_for_test();
        let unique = unique(&setup.snapshot);
        assert!(
            captured.len() > unique.len(),
            "actual static/manager roles alias physical storage"
        );
        let bytes = unique.values().sum::<u64>();
        assert!(bytes > 0);
        let registered_before = pool.fixture_host_charge().unwrap();
        let (result, segment) = pin_in_real_callback(&mut f, &mut setup, &source, false);
        result.unwrap().unwrap();
        assert_eq!(setup.snapshot.registered_bytes(), Some(bytes));
        assert_eq!(setup.snapshot.issued_rows_for_test(), 1);
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            registered_before,
            "existing pins make no physical charge"
        );
        f.validate_frontier(0).unwrap();
        let OpeningPinSetup {
            snapshot,
            scope,
            owned,
            reservation,
            run,
        } = setup;
        settle_empty_callback(&stream, scope);
        drop(segment);
        let held = owned.protected_host_bytes();
        drop((owned, reservation, run));
        assert!(pool.fixture_host_charge().unwrap() >= held);
        drop((snapshot, publication, source, f, captured, unique));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            crate::backend::ordinary_retirement::reclaim();
            // The final native drops enqueue accounting owners. No later native
            // entry is expected, so reclaim its separate unlocked host queue.
            safemlx::reclaim_allocation_owners();
            pool.fixture_host_charge().unwrap() == 0
        });
    }
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
fn old_snapshot_survives_actual_later_state_and_new_kv_snapshot_rejects_missing_keys() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let mut f = PrefillRetentionFixture::load(artifact.path(), &stream).unwrap();
    let first = source(&f, geometry(0));
    let next = source(&f, geometry(1));
    let initial = f.inventory().unwrap();
    let publication = initial.publish_unquoted(&loading).unwrap();
    drop(loading);
    let q = quote(&f, &pool, geometry(0), &first);
    let old = f
        .session
        .prepare_pin_snapshot(&first, q, &pool, Some(u64::MAX))
        .unwrap();
    let old_keys = old.snapshot.captured_keys_for_test();
    let request = InferenceRequest::from(&old.reservation);
    let output = f
        .run(
            request,
            Arc::from([1]),
            GenerationCancellationToken::new(),
            &mut Quiet,
        )
        .unwrap();
    assert_eq!(output.outcome, PrefillOutcome::Complete);
    stream.synchronize().unwrap();
    f.validate_frontier(1).unwrap();
    assert_eq!(
        old.snapshot.captured_keys_for_test(),
        old_keys,
        "snapshot retains old owners, never refreshes itself"
    );
    // Ordinary execution preserves the runtime binding independently of the
    // snapshot's retained owner identity and the later KV publication check.
    f.session.validate_opening_paths_after_forward().unwrap();
    // The first original native scope continues to retain its unregistered
    // successful state. A separately admitted pin-only request must not infer
    // publication from that native completion.
    let q = quote(&f, &pool, geometry(1), &next);
    let mut setup = f
        .session
        .prepare_pin_snapshot(&next, q, &pool, Some(u64::MAX))
        .unwrap();
    assert!(
        unique(&setup.snapshot)
            .keys()
            .any(|key| !old_keys.iter().any(|(old, _)| old == key)),
        "real forward retained new backing"
    );
    let registered = pool.fixture_host_charge().unwrap();
    let (result, segment) = pin_in_real_callback(&mut f, &mut setup, &next, false);
    let error = result.unwrap().unwrap_err();
    assert!(matches!(
        cause::<BoundedPinError>(&error),
        Some(BoundedPinError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(setup.snapshot.registered_bytes(), None);
    assert_eq!(setup.snapshot.issued_rows_for_test(), 1);
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        registered,
        "missing late key commits no prefix"
    );
    let OpeningPinSetup {
        snapshot,
        scope,
        owned,
        reservation,
        run,
    } = setup;
    settle_empty_callback(&stream, scope);
    drop(segment);
    // Failed attempts retain their canonical stamp as well as control custody.
    // The stamp owns the original H and the actual shared source (with C).
    let held = owned.protected_host_bytes()
        + CaptureRunHostPlan::prepare(&next)
            .unwrap()
            .initialization_peak_bytes()
        + next.capacity_bytes().unwrap();
    drop((snapshot, owned, reservation, run, next));
    let before_error_drop = pool.fixture_host_charge().unwrap();
    assert!(before_error_drop >= held);
    drop(error);
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        before_error_drop - held,
        "escaped failure owns original P+Q+S, stamped H and source C"
    );
    // Publish actual changed state only after testing its missing opening key.
    let final_publication = f.inventory().unwrap().publish_funded(&old.scope).unwrap();
    let OpeningPinSetup {
        snapshot,
        scope,
        owned,
        reservation,
        run,
    } = old;
    settle_empty_callback(&stream, scope);
    drop((
        snapshot,
        owned,
        reservation,
        run,
        output,
        final_publication,
        publication,
        first,
        f,
    ));
}

#[test]
fn callback_panic_keeps_pinned_snapshot_outside_unwind_and_never_reissues() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let mut f = PrefillRetentionFixture::load(artifact.path(), &stream).unwrap();
    let source = source(&f, geometry(0));
    let initial = f.inventory().unwrap();
    let publication = initial.publish_unquoted(&loading).unwrap();
    drop(loading);
    let q = quote(&f, &pool, geometry(0), &source);
    let mut setup = f
        .session
        .prepare_pin_snapshot(&source, q, &pool, None)
        .unwrap();
    let expected = setup.snapshot.captured_keys_for_test();
    let (result, segment) = pin_in_real_callback(&mut f, &mut setup, &source, true);
    result.unwrap().unwrap();
    assert_eq!(setup.snapshot.captured_keys_for_test(), expected);
    assert!(setup.snapshot.registered_bytes().unwrap() > 0);
    assert_eq!(setup.snapshot.issued_rows_for_test(), 1);
    // The actual shared session is fenced by its outer callback unwind. The
    // snapshot remains merely retained evidence and cannot clear that fence.
    let request = crate::memory_fixture::empty_admitted_request(f.identity(), geometry(0)).unwrap();
    assert!(f
        .run(
            request,
            Arc::from([1]),
            GenerationCancellationToken::new(),
            &mut Quiet
        )
        .is_err());
    let OpeningPinSetup {
        snapshot,
        scope,
        owned,
        reservation,
        run,
    } = setup;
    settle_empty_callback(&stream, scope);
    drop(segment);
    let held = owned.protected_host_bytes();
    drop((owned, reservation, run));
    assert!(pool.fixture_host_charge().unwrap() >= held);
    drop((snapshot, publication, source, f));
}
