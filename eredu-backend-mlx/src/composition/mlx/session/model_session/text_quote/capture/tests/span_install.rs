//! Actual original admission and custody, without enabling native execution.
use super::*;
use eredu_runtime::working_memory::InferenceWorkspaceSpan;

fn active_source(runtime: &Runtime, path: &str, preview: u64) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "original-prefill".into(),
        path: path.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::Preview {
            max_elements: preview,
        },
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap(),
    )
}

pub(super) fn quoted(
    runtime: &Runtime,
    source: &SharedCapturePlan,
    ids: &Vec<u32>,
    controller: &disk::Controller,
) -> IncrementalInferenceQuote {
    let capture = CaptureAdmission::new(runtime.session(), geometry(), source, eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary()).unwrap();
    let mut g = geometry();
    g.output = capture.physical_output(g.output);
    let storage = ControllerStorageContract::inspect(controller).unwrap();
    let registered = storage
        .pin_registered(controller, runtime.backend().memory_pool())
        .unwrap();
    super::super::super::quote_incremental(
        runtime.session(),
        g,
        (ids.capacity() * 4) as u64,
        config(u64::MAX),
        controller.inference_workspace(4).unwrap(),
        &storage,
        Some(&registered),
        false,
        Some(&capture),
    )
    .unwrap()
    .quote
    .unwrap()
}

#[test]
fn original_bound_capture_admits_exact_physical_readout_and_plan_on_all_weight_routes() {
    let stream = stream();
    for route in 0..3 {
        for (path, preview, output) in [
            ("readout.embedding", 5, OutputDemand::LastPosition),
            (
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                0,
                OutputDemand::Sequence,
            ),
        ] {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (runtime, _artifact) = load(&stream, &pool, route);
            let source = active_source(&runtime, path, preview);
            let ids = vec![2, 5, 7];
            let controller = disk::Controller::default();
            let baseline = pool.used_bytes().unwrap();
            let before = path_instrumentation::snapshot();
            let frontier = runtime.session().payload.model.erased().state_snapshot();
            let quote = quoted(&runtime, &source, &ids, &controller);
            let required = quote.incremental_bytes();
            let p = quote.span_workspace().retention_peak_bytes().unwrap();
            assert!(p > 0 && required > p);
            assert_eq!(quote.geometry().output, output);
            drop(quote);
            let exact = baseline + required;
            let error = admit(&runtime, &source, &ids, &controller, exact - 1).unwrap_err();
            assert!(matches!(
                cause::<WorkingMemoryError>(&error),
                WorkingMemoryError::BudgetExceeded { .. }
            ));
            assert_eq!(pool.used_bytes().unwrap(), baseline);
            let (preparation, quote) = admit(&runtime, &source, &ids, &controller, exact).unwrap();
            let request = preparation.request();
            assert_eq!(request.geometry().output, output);
            assert_eq!(request.memory_reservation().unwrap().bytes(), required);
            let installed = quote
                .take_capture_installation(runtime.session(), &source)
                .unwrap();
            let bound = installed.prefill_selection().unwrap();
            assert_eq!(bound.geometry(), request.geometry());
            assert!(bound.selection().source().same_storage(&source));
            let span = installed.span_workspace();
            span.reservation().validate_domain(&pool).unwrap();
            assert_eq!(span.reservation().geometry(), request.geometry());
            assert_eq!(span.reservation().bytes(), required);
            assert_eq!(span.workspace().retention_peak_bytes(), Some(p));
            let q = span
                .workspace()
                .text_controls()
                .unwrap()
                .facts()
                .total_bytes()
                .unwrap()
                .unwrap();
            assert!(q > 0);
            let publication = span
                .workspace()
                .text_controls()
                .unwrap()
                .capture_publication_control_bytes();
            assert_eq!(span.protected_host_bytes(), p + q + publication);
            let chunks: Vec<_> = span
                .workspace()
                .plan()
                .records()
                .iter()
                .filter_map(|r| match r.span() {
                    InferenceWorkspaceSpan::Prefill(c) => Some(c),
                    _ => None,
                })
                .collect();
            assert_eq!(chunks.len(), 3);
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(c.input, i as u64..i as u64 + 1);
                assert_eq!(c.output, output.for_chunk(i == 2));
            }
            assert!(
                span.workspace()
                    .plan()
                    .records()
                    .iter()
                    .all(|r| r.new_allocation_bytes().is_some())
            );
            assert_eq!(pool.used_bytes().unwrap(), exact);
            assert_eq!(path_instrumentation::snapshot(), before);
            assert_eq!(
                runtime.session().payload.model.erased().state_snapshot(),
                frontier
            );
            assert_eq!(controller.0.get(), (0, 0));
            drop((installed, quote, preparation, source));
            finish_runtime(runtime, &stream);
            settle_terminal(&pool, 0);
        }
    }
}

#[test]
fn original_selection_rejects_readout_downgrade_and_accepts_only_actual_shared_host_source() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = active_source(&runtime, eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 0);
    let foreign = active_source(&runtime, eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 0);
    let capture = CaptureAdmission::new(runtime.session(), geometry(), &source, eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary()).unwrap();
    let before = (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        path_instrumentation::snapshot(),
    );
    assert!(matches!(
        cause::<eredu_runtime::layered::PreparedCaptureSelectionError>(
            &capture.bind_geometry(geometry()).err().unwrap()
        ),
        eredu_runtime::layered::PreparedCaptureSelectionError::Readout
    ));
    let alias = CaptureRunHostPlan::prepare(capture.selection.source()).unwrap();
    assert!(!std::ptr::eq(alias.source(), &source));
    assert!(alias.source().same_storage(&source));
    capture.validate_host(&alias).unwrap();
    let other = CaptureRunHostPlan::prepare(&foreign).unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&capture.validate_host(&other).unwrap_err()),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(
        (
            pool.used_bytes().unwrap(),
            pool.peak_bytes().unwrap(),
            path_instrumentation::snapshot()
        ),
        before
    );
    drop(alias);
    drop(other);
    drop(capture);
    drop((source, foreign));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, 0);
}

#[test]
fn installed_span_alias_keeps_original_plan_charge_after_quote_and_collector_retire() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = active_source(&runtime, "readout.embedding", 5);
    let c = source.capacity_bytes().unwrap();
    let baseline = pool.used_bytes().unwrap();
    let ids = vec![2, 5, 7];
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &ids,
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let protected = installed.span_workspace().protected_host_bytes();
    let alias = installed.span_workspace().workspace().plan().clone();
    let p = installed
        .span_workspace()
        .workspace()
        .retention_peak_bytes()
        .unwrap();
    let before = path_instrumentation::snapshot();
    drop((quote, preparation));
    installed.validate_sources(&pool).unwrap();
    assert!(
        installed
            .span_workspace()
            .workspace()
            .plan()
            .same_plan(&alias)
    );
    drop(installed);
    assert!(pool.used_bytes().unwrap() >= baseline + c + p);
    assert_eq!(alias.geometry().output, OutputDemand::LastPosition);
    assert!(!alias.records().is_empty());
    assert_eq!(path_instrumentation::snapshot(), before);
    drop(alias);
    // The source-side raw scope retains exactly original P+Q+S, plus registered
    // C, after native/run closure; the rest of the original envelope retires.
    disk::settle(&pool, baseline + protected + c);
    drop(source);
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, 0);
}
