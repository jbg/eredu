//! Private original admission only: no capture installation or model execution.
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
};
use crate::tests::support::path_instrumentation;
use eredu_core::{TextGenerationBackend, capture::*};
use eredu_runtime::PreparedSessionObservationError;

type Runtime = ModelRuntime<MlxBackend<'static>>;

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))
}
fn load(stream: &Stream, pool: &WorkingMemoryPool, route: usize) -> (Runtime, tempfile::TempDir) {
    match route {
        0 => host::runtime(stream, pool, None),
        1 => host::runtime(stream, pool, Some(1)),
        2 => disk::load_runtime(stream, pool, true),
        _ => unreachable!(),
    }
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn config(capacity: u64) -> TextGenerationConfig {
    disk::config(0.0, 1, capacity)
}
fn source(runtime: &Runtime, cached_positions: u64) -> SharedCapturePlan {
    // Public cold discovery and caller-owned source creation precede admission.
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "original-logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule {
            prefill: false,
            ..Default::default()
        },
        slices: vec![],
        transform: CaptureTransform::Preview { max_elements: 3 },
    });
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin { cached_positions },
        )
        .unwrap(),
    )
}
fn candidate(
    runtime: &Runtime,
    source: &SharedCapturePlan,
    ids: &Vec<u32>,
    controller: &disk::Controller,
) -> (u64, u64, u64) {
    let capture = CaptureAdmission::new(runtime.session(), geometry(), source, eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary()).unwrap();
    let h = capture.host.initialization_peak_bytes();
    let c = capture.new_source_bytes;
    let (_, _, mut outside) = super::super::enclosing_components(
        runtime.session(),
        geometry(),
        (ids.capacity() * 4) as u64,
    )
    .unwrap();
    let before = outside.retained.bytes().unwrap();
    capture.add_enclosing(&mut outside).unwrap();
    assert_eq!(outside.retained.bytes().unwrap(), before + h);
    let storage = ControllerStorageContract::inspect(controller).unwrap();
    let registered = storage
        .pin_registered(controller, runtime.backend().memory_pool())
        .unwrap();
    let candidate = super::super::quote_incremental(
        runtime.session(),
        geometry(),
        (ids.capacity() * 4) as u64,
        config(u64::MAX),
        controller.inference_workspace(4).unwrap(),
        &storage,
        Some(&registered),
        false,
        Some(&capture),
    )
    .unwrap();
    let quote = candidate.quote.unwrap();
    assert_eq!(quote.geometry(), geometry());
    assert!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .is_some()
    );
    (quote.incremental_bytes(), h, c)
}
fn admit(
    runtime: &Runtime,
    source: &SharedCapturePlan,
    ids: &Vec<u32>,
    controller: &disk::Controller,
    capacity: u64,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure> {
    super::super::admit_with_capture(
        runtime,
        &disk::evidence(ids),
        config(capacity),
        controller,
        source,
    )
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a T {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return found;
        }
        error = error.source().expect("original typed failure source");
    }
}
fn pending(
    quote: &TextExecutionQuote,
    source: &SharedCapturePlan,
    h: u64,
) -> *const PreparedCaptureRun {
    let bank = quote.capture.as_ref().unwrap().pending.borrow();
    let bank = &bank.as_ref().expect("still pending").bank;
    assert!(bank.source().same_storage(source));
    assert_eq!(bank.protected_bytes(), h);
    assert_eq!(bank.spent_steps(), 0);
    bank as *const _
}
fn finish_runtime(runtime: Runtime, stream: &Stream) {
    runtime.synchronize().unwrap();
    drop(runtime);
    stream.synchronize().unwrap();
    // Existing terminal graph retirement, only after all cold assertions.
    safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
        .unwrap()
        .synchronize()
        .unwrap();
}

fn settle_terminal(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        // Native graph retirement visits a bounded batch. Every retired record
        // still requires its own terminal proof; this empty completion submits
        // no tensor work and lets subsequent passes reach the remaining owners.
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        disk::reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}

#[test]
fn original_capture_admission_prices_new_source_once_on_resident_host_and_disk() {
    let stream = stream();
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = load(&stream, &pool, route);
        let ids = vec![2, 5, 7];
        let source = source(&runtime, 0);
        let early_alias = source.clone();
        let controller = disk::Controller::default();
        let baseline = pool.used_bytes().unwrap();
        let peak = pool.peak_bytes().unwrap();
        let native = path_instrumentation::snapshot();
        let inputs = path_instrumentation::session_input_creation_attempts();
        let resets = path_instrumentation::session_reset_attempts();
        let state = runtime.session().payload.model.erased().state_snapshot();
        let residency = runtime.session().residency_report().unwrap();
        let (required, h, c) = candidate(&runtime, &source, &ids, &controller);
        assert!(h > 0 && c > 0);
        assert_eq!(c, source.capacity_bytes().unwrap());
        assert!(required >= h + c);
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(pool.peak_bytes().unwrap(), peak);
        let exact = baseline.checked_add(required).unwrap();
        let short = admit(&runtime, &source, &ids, &controller, exact - 1).unwrap_err();
        assert!(matches!(
            cause::<WorkingMemoryError>(&short),
            WorkingMemoryError::BudgetExceeded { .. }
        ));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(pool.peak_bytes().unwrap(), peak);
        assert!(matches!(
            pool.pin_registered_storage([(
                StorageIdentity::CapturePlan(source.storage_identity().clone()),
                c,
            )]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let (preparation, quote) = admit(&runtime, &source, &ids, &controller, exact).unwrap();
        assert_eq!(
            preparation.request().memory_reservation().unwrap().bytes(),
            required
        );
        assert_eq!(pool.used_bytes().unwrap(), exact);
        pending(&quote, &source, h);
        let bank = quote
            .capture
            .as_ref()
            .unwrap()
            .take_pending(runtime.session(), &source, geometry())
            .unwrap();
        assert!(bank.bank.source().same_storage(&early_alias));
        assert_eq!(bank.bank.protected_bytes(), h);
        let terminal_tail = bank.span.protected_host_bytes() + c;
        assert!(terminal_tail < required);
        assert_eq!(bank.bank.spent_steps(), 0);
        // No Prompt/Sampling/native work has run, yet the entire bank already
        // exists under the original request and exact source is published.
        assert_eq!(path_instrumentation::snapshot(), native);
        assert_eq!(
            path_instrumentation::session_input_creation_attempts(),
            inputs
        );
        assert_eq!(path_instrumentation::session_reset_attempts(), resets);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            state
        );
        let admitted_residency = runtime.session().residency_report().unwrap();
        match (&residency, &admitted_residency) {
            (Some(before), Some(after)) => {
                assert_eq!(before.initialized(), after.initialized());
                assert_eq!(before.offload(), after.offload());
                assert_eq!(before.active_window(), after.active_window());
                assert_eq!(before.weight_store(), after.weight_store());
                assert_eq!(before.unit_sources(), after.unit_sources());
                assert_eq!(before.materialization(), after.materialization());
                assert_eq!(before.units().len(), after.units().len());
                for (before, after) in before.units().iter().zip(after.units()) {
                    let added_pin = u64::from(
                        route == 1
                            && before.policy() == eredu_core::residency::ResidencyPolicy::Windowed,
                    );
                    assert_eq!(
                        (
                            after.id(),
                            after.planned_tier(),
                            after.policy(),
                            after.expected_bytes(),
                            after.host_allocated_bytes(),
                            after.device_allocated_bytes(),
                            after.host_resident(),
                            after.device_resident(),
                            after.host_pins(),
                            after.device_pins(),
                            after.active_window()
                        ),
                        (
                            before.id(),
                            before.planned_tier(),
                            before.policy(),
                            before.expected_bytes(),
                            before.host_allocated_bytes(),
                            before.device_allocated_bytes(),
                            before.host_resident(),
                            before.device_resident(),
                            before.host_pins() + added_pin,
                            before.device_pins(),
                            before.active_window()
                        ),
                    );
                }
            }
            _ => assert_eq!(admitted_residency, residency),
        }
        assert_eq!(controller.0.get(), (0, 0));
        drop((bank, quote, preparation));
        disk::settle(&pool, baseline + terminal_tail);
        assert_eq!(
            runtime.session().residency_report().unwrap(),
            residency,
            "original host source leases retire with the unused quote"
        );
        let (existing_required, existing_h, existing_c) =
            candidate(&runtime, &source, &ids, &controller);
        assert_eq!(existing_h, h);
        assert_eq!(existing_c, 0);
        assert_eq!(existing_required + c, required);
        // Only P+Q+S+C survives, with the original ceiling unchanged. The
        // second request does not fit that ceiling. Raising it also rejects:
        // the session retains its actual handoff, but live original host
        // custody makes that predecessor ineligible for a ceiling increase.
        let next_exact = baseline + terminal_tail + existing_required;
        assert_eq!(pool.used_bytes().unwrap() + existing_required, next_exact);
        assert!(next_exact > exact);
        assert_eq!(pool.effective_capacity().unwrap(), exact);
        let at_original = admit(&runtime, &source, &ids, &controller, exact).unwrap_err();
        assert!(
            matches!(
                cause::<WorkingMemoryError>(&at_original),
                WorkingMemoryError::BudgetExceeded { .. }
            ),
            "route {route}, original ceiling: {at_original:?}"
        );
        let rejected = admit(&runtime, &source, &ids, &controller, next_exact).unwrap_err();
        assert!(
            matches!(
                cause::<WorkingMemoryError>(&rejected),
                WorkingMemoryError::ExecutionFenced
            ),
            "route {route}, retained host custody: {rejected:?}"
        );
        assert_eq!(pool.effective_capacity().unwrap(), exact);
        disk::settle(&pool, baseline + terminal_tail);
        assert_eq!(path_instrumentation::snapshot(), native);
        assert_eq!(controller.0.get(), (0, 0));
        drop(source);
        finish_runtime(runtime, &stream);
        settle_terminal(&pool, terminal_tail);
        // Earlier aliases retain the exact original host and C charges, without
        // retaining unused equation/controller headroom.
        assert_eq!(early_alias.capacity_bytes(), Some(c));
        drop(early_alias);
        settle_terminal(&pool, 0);
    }
}

#[test]
fn original_capture_bank_rejects_wrong_source_and_coordinates_then_moves_once() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let equal_independent = self::source(&runtime, 0);
    assert_eq!(
        source.admission().identity(),
        equal_independent.admission().identity()
    );
    assert!(!source.same_storage(&equal_independent));
    let wrong_origin = self::source(&runtime, 1);
    let ids = vec![2, 5, 7];
    let controller = disk::Controller::default();
    let before = pool.used_bytes().unwrap();
    let rejected = admit(&runtime, &wrong_origin, &ids, &controller, u64::MAX).unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&rejected),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let (_, h, c) = candidate(&runtime, &source, &ids, &controller);
    let (preparation, quote) = admit(&runtime, &source, &ids, &controller, u64::MAX).unwrap();
    let ptr = pending(&quote, &source, h);
    let used = pool.used_bytes().unwrap();
    let capture = quote.capture.as_ref().unwrap();
    let error = capture
        .take_pending(runtime.session(), &equal_independent, geometry())
        .unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pending(&quote, &source, h), ptr);
    let mut wrong = geometry();
    wrong.cached_positions = 1;
    let error = capture
        .take_pending(runtime.session(), &source, wrong)
        .unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pending(&quote, &source, h), ptr);
    assert_eq!(pool.used_bytes().unwrap(), used);
    let bank = capture
        .take_pending(runtime.session(), &source, geometry())
        .unwrap();
    assert!(capture.pending.borrow().is_none());
    assert!(bank.bank.source().same_storage(&source));
    let terminal_tail = bank.span.protected_host_bytes() + c;
    let error = capture
        .take_pending(runtime.session(), &source, geometry())
        .unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::PreparationAlreadyStarted
    ));
    assert_eq!(controller.0.get(), (0, 0));
    drop((bank, quote, preparation));
    // Only the original protected controls and published source outlive work.
    disk::settle(&pool, before + terminal_tail);
    drop((source, equal_independent, wrong_origin));
    finish_runtime(runtime, &stream);
    disk::settle(&pool, 0);
}

#[test]
fn original_capture_bank_checks_actual_current_path_token_before_consumption() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let ids = vec![2, 5, 7];
    let controller = disk::Controller::default();
    let (_, h, _) = candidate(&runtime, &source, &ids, &controller);
    let (preparation, quote) = admit(&runtime, &source, &ids, &controller, u64::MAX).unwrap();
    let ptr = pending(&quote, &source, h);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        disk::reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
    let payload = runtime.session_mut().payload.get_mut().unwrap();
    assert!(
        payload
            .model
            .erased_mut()
            .publish_parameter_replacements(&Default::default(), false)
            .unwrap()
    );
    let before = path_instrumentation::snapshot();
    let used = pool.used_bytes().unwrap();
    let error = quote
        .capture
        .as_ref()
        .unwrap()
        .take_pending(runtime.session(), &source, geometry())
        .unwrap_err();
    assert_eq!(
        *cause::<PreparedSessionObservationError>(&error),
        PreparedSessionObservationError::BindingMismatch
    );
    assert_eq!(pending(&quote, &source, h), ptr);
    assert_eq!(path_instrumentation::snapshot(), before);
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(controller.0.get(), (0, 0));
    drop((quote, preparation, source));
    finish_runtime(runtime, &stream);
    disk::settle(&pool, 0);
}

#[test]
fn late_source_quarantine_rejects_install_witness_and_preserves_original_pending_bank() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let alias = source.clone();
    let ids = vec![2, 5, 7];
    let controller = disk::Controller::default();
    let (old_preparation, old_quote) =
        admit(&runtime, &source, &ids, &controller, u64::MAX).unwrap();
    // Actual source attachment comes from the first original request. Keep one
    // of its real native scopes open so its later abandonment quarantines that
    // inherited origin after the second request's admission health check.
    let abandoned = old_quote.funding_scope().unwrap();
    let (_, h, c) = candidate(&runtime, &source, &ids, &controller);
    assert_eq!(c, 0);
    let (preparation, quote) = admit(&runtime, &source, &ids, &controller, u64::MAX).unwrap();
    let ptr = pending(&quote, &source, h);
    let capture = quote.capture.as_ref().unwrap();
    capture
        .pending
        .borrow()
        .as_ref()
        .unwrap()
        .witness
        .validate(&pool)
        .unwrap();
    drop(abandoned);
    let used = pool.used_bytes().unwrap();
    let before = path_instrumentation::snapshot();
    assert!(matches!(
        capture
            .control_guard()
            .validate_reservation(preparation.request().memory_reservation().unwrap()),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    for _ in 0..2 {
        let error = capture
            .take_pending(runtime.session(), &source, geometry())
            .unwrap_err();
        assert!(matches!(
            cause::<WorkingMemoryError>(&error),
            WorkingMemoryError::ExecutionFenced
        ));
        assert_eq!(pending(&quote, &source, h), ptr);
        assert_eq!(pool.used_bytes().unwrap(), used);
    }
    // The same inherited origin must also reject a later admission; no new H
    // bank or partial source publication can disguise the quarantined account.
    let rejected = admit(&runtime, &source, &ids, &controller, u64::MAX).unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&rejected),
        WorkingMemoryError::ExecutionFenced
    ));
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(path_instrumentation::snapshot(), before);
    assert_eq!(controller.0.get(), (0, 0));
    {
        let pending = capture.pending.borrow();
        assert!(alias.same_storage(pending.as_ref().unwrap().bank.source()));
    }
    drop((
        quote,
        preparation,
        old_quote,
        old_preparation,
        source,
        alias,
    ));
    finish_runtime(runtime, &stream);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "abandonment must not refund the old source envelope"
    );
}

mod span_install;

mod controls;
