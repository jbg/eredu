//! Exercise the executable boundary used by admission, not a second equation driver.
use super::*;
use eredu_core::TokenFilter;
use eredu_runtime::layered::PreparedCaptureSelectionError;
use std::cell::Cell;

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn has_cause<T: std::error::Error + 'static>(
    error: &(dyn std::error::Error + 'static),
    check: impl Fn(&T) -> bool,
) -> bool {
    let mut next = Some(error);
    while let Some(error) = next {
        if error.downcast_ref::<T>().is_some_and(&check) {
            return true;
        }
        // The existing generic prepared-execution wrapper retains E in its
        // Backend variant, not through std::error::Error::source.
        next = match error.downcast_ref::<
            eredu_architectures::prepared_execution::PreparedExecutionError<eredu_nn::Error>,
        >() {
            Some(eredu_architectures::prepared_execution::PreparedExecutionError::Backend(inner)) => {
                Some(inner)
            }
            _ => error.source(),
        };
    }
    false
}
fn same_quote(
    left: &eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
    right: &eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
) {
    assert_eq!(left.equations.geometry(), right.equations.geometry());
    assert_eq!(
        left.equations.completed_spans(),
        right.equations.completed_spans()
    );
    assert_eq!(
        left.equations.transient().bytes(),
        right.equations.transient().bytes()
    );
    assert_eq!(
        left.equations.retained_peak_bytes(),
        right.equations.retained_peak_bytes()
    );
    assert_eq!(
        left.equations.tensor_transient_peak_bytes(),
        right.equations.tensor_transient_peak_bytes()
    );
    assert_eq!(
        left.equations.host_peak_bytes(),
        right.equations.host_peak_bytes()
    );
    assert_eq!(left.sampling.steps, right.sampling.steps);
    assert_eq!(left.sampling.output_width, right.sampling.output_width);
    assert_eq!(left.sampling.peak.bytes(), right.sampling.peak.bytes());
    assert_eq!(
        left.sampling.final_history_bytes,
        right.sampling.final_history_bytes
    );
}

#[test]
fn model_bound_routes_keep_exact_source_and_sample_final_row_for_resident_host_disk() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = match route {
            0 => host::runtime(&stream, &pool, None),
            1 => host::runtime(&stream, &pool, Some(1)),
            _ => disk::load_runtime(&stream, &pool, true),
        };
        for cached in [0, 8] {
            crate::backend::submission_recovery::wait_for_retirement(|| {
                reclaim();
                runtime.session().payload.active_owner_count() == 1
            });
            runtime.session().ensure_no_submission_in_flight().unwrap();
            for path in [
                "readout.embedding",
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
            ] {
                let source = source(&runtime, path, 5, 1);
                let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
                let source = SharedCapturePlan::new(
                    source
                        .admission()
                        .plan()
                        .clone()
                        .admit_with_text_origin(
                            &discovery.catalog,
                            &discovery.support,
                            &discovery.support.capture,
                            source.admission().request(),
                            CaptureTextOrigin {
                                cached_positions: cached,
                            },
                        )
                        .unwrap(),
                );
                let actual_paths = paths(&runtime);
                let selected = actual_paths.prepare_capture_selection(&source).unwrap();
                let source_alias = source.clone();
                drop(source);
                let g = eredu_core::InferenceGeometry {
                    cached_positions: cached,
                    ..geometry(&selected)
                };
                let bound = selected.bind_geometry(g).unwrap();
                let executable = &runtime.session().payload.model;
                let config = disk::config(0.0, 2, u64::MAX);
                let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
                let frontier = executable.erased().state_snapshot();
                let cold = ColdGuard::new();
                let (quote, h) = executable
                    .quote_replicated_resident_text_with_sampling_and_prefill_capture(
                        g,
                        config,
                        &TokenFilter::All,
                        bound,
                    )
                    .unwrap();
                let (registered, pin, rh) = executable
                    .quote_registered_resident_text_with_sampling_and_prefill_capture(
                        g,
                        config,
                        &TokenFilter::All,
                        &pool,
                        bound,
                    )
                    .unwrap();
                assert!(quote.equations.transient().bytes().is_some());
                assert!(registered
                    .equations
                    .residual_workspace()
                    .unwrap()
                    .peak_bytes()
                    .is_some());
                assert_eq!(quote.equations.completed_spans(), 6);
                assert_eq!(quote.sampling.steps, 4);
                assert_eq!(
                    quote.sampling.output_width,
                    registered.sampling.output_width
                );
                assert_eq!(
                    quote.sampling.peak.bytes(),
                    registered.sampling.peak.bytes()
                );
                assert_eq!(quote.equations.geometry(), g);
                assert_eq!(
                    g.output,
                    if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                        eredu_core::OutputDemand::Sequence
                    } else {
                        eredu_core::OutputDemand::LastPosition
                    }
                );
                assert!(std::ptr::eq(h.source(), selected.source()));
                assert!(std::ptr::eq(rh.source(), selected.source()));
                assert!(h.source().same_storage(&source_alias));
                assert_eq!(
                    h.initialization_peak_bytes(),
                    rh.initialization_peak_bytes()
                );
                assert!(h.initialization_peak_bytes() > 0);
                assert_eq!(executable.erased().state_snapshot(), frontier);
                assert_eq!(
                    (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
                    before
                );
                drop((quote, registered, pin, h, rh));
                cold.assert_cold();
            }
            if cached == 0 {
                let ids = disk::tokens();
                let (capacity, _) = disk::exact_capacity(&runtime, &pool, &ids, 0.0, 2);
                let outputs =
                    disk::outputs(&mut runtime, ids, disk::config(0.0, 2, capacity), false);
                assert_eq!(disk::token_ids(&outputs).len(), 4);
                drop(outputs);
            }
        }
        retire(runtime, &pool, &stream);
    }
}

#[test]
fn model_bound_routes_reject_changed_candidate_and_equal_content_foreign_path_owner() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let other_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (other, _other_artifact) = host::runtime(&stream, &other_pool, None);
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = match route {
            0 => host::runtime(&stream, &pool, None),
            1 => host::runtime(&stream, &pool, Some(1)),
            _ => disk::load_runtime(&stream, &pool, true),
        };
        {
            let source = source(&runtime, "readout.embedding", 5, 1);
            let actual = paths(&runtime);
            let foreign = paths(&other);
            assert_eq!(strings(&actual), strings(&foreign));
            assert!(!actual.same_storage(&foreign));
            let selected = actual.prepare_capture_selection(&source).unwrap();
            let foreign_selected = foreign.prepare_capture_selection(&source).unwrap();
            let g = geometry(&selected);
            let executable = &runtime.session().payload.model;
            let config = disk::config(0.0, 2, u64::MAX);
            let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
            let frontier = executable.erased().state_snapshot();
            let cold = ColdGuard::new();
            for (candidate, bound) in [
                (
                    eredu_core::InferenceGeometry {
                        prefill_chunk_positions: 1,
                        ..g
                    },
                    selected.bind_geometry(g).unwrap(),
                ),
                // The binding rejection also precedes invalid native batch projection.
                (
                    eredu_core::InferenceGeometry { batch_size: 0, ..g },
                    selected.bind_geometry(g).unwrap(),
                ),
                (g, foreign_selected.bind_geometry(g).unwrap()),
            ] {
                let error = executable
                    .quote_replicated_resident_text_with_sampling_and_prefill_capture(
                        candidate,
                        config,
                        &TokenFilter::All,
                        bound,
                    )
                    .unwrap_err();
                assert!(has_cause::<PreparedCaptureSelectionError>(
                    &error,
                    |e| matches!(e, PreparedCaptureSelectionError::Identity)
                ));
                let error = executable
                    .quote_registered_resident_text_with_sampling_and_prefill_capture(
                        candidate,
                        config,
                        &TokenFilter::All,
                        &pool,
                        bound,
                    )
                    .unwrap_err();
                assert!(has_cause::<PreparedCaptureSelectionError>(
                    &error,
                    |e| matches!(e, PreparedCaptureSelectionError::Identity)
                ));
            }
            assert_eq!(executable.erased().state_snapshot(), frontier);
            assert_eq!(
                (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
                before
            );
            cold.assert_cold();
        }
        retire(runtime, &pool, &stream);
    }
    retire(other, &other_pool, &stream);
}

#[test]
fn model_bound_decode_only_agrees_with_raw_and_keeps_ordinary_and_raw_prefill_behavior() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = host::runtime(&stream, &pool, None);
    {
        let active = source(&runtime, eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 5, 1);
        let mut raw = active.admission().plan().clone();
        raw.selections[0].schedule.prefill = false;
        let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
        let source = SharedCapturePlan::new(
            raw.admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                active.admission().request(),
            )
            .unwrap(),
        );
        let actual = paths(&runtime);
        let selected = actual.prepare_capture_selection(&source).unwrap();
        let g = geometry(&selected);
        assert_eq!(g.output, eredu_core::OutputDemand::LastPosition);
        let executable = &runtime.session().payload.model;
        let config = disk::config(0.0, 2, u64::MAX);
        let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
        let cold = ColdGuard::new();
        let plain = executable
            .quote_replicated_resident_text_with_sampling(g, config, &TokenFilter::All)
            .unwrap();
        let (raw, raw_h) = executable
            .quote_replicated_resident_text_with_sampling_and_capture(
                g,
                config,
                &TokenFilter::All,
                &source,
            )
            .unwrap();
        let (bound, bound_h) = executable
            .quote_replicated_resident_text_with_sampling_and_prefill_capture(
                g,
                config,
                &TokenFilter::All,
                selected.bind_geometry(g).unwrap(),
            )
            .unwrap();
        // Bound p0 runs the prepared borrowed hook to advance fragment
        // progress even when every selection is decode-only. Raw p0 skips
        // that hook. The extra host traversal cost is therefore real; compare
        // its span-local delta while requiring identical numerical domains.
        assert_eq!(raw.equations.geometry(), bound.equations.geometry());
        assert_eq!(
            raw.equations.completed_spans(),
            bound.equations.completed_spans()
        );
        assert_eq!(
            raw.equations.retained_peak_bytes(),
            bound.equations.retained_peak_bytes()
        );
        assert_eq!(
            raw.equations.tensor_transient_peak_bytes(),
            bound.equations.tensor_transient_peak_bytes()
        );
        assert_eq!(raw.sampling.steps, bound.sampling.steps);
        assert_eq!(raw.sampling.output_width, bound.sampling.output_width);
        assert_eq!(raw.sampling.peak.bytes(), bound.sampling.peak.bytes());
        assert_eq!(
            raw.sampling.final_history_bytes,
            bound.sampling.final_history_bytes
        );
        let raw_spans = raw.equations.span_workspace_plan().records();
        let bound_spans = bound.equations.span_workspace_plan().records();
        assert_eq!(raw_spans.len(), bound_spans.len());
        let hook = bound_spans[0]
            .new_allocation_bytes()
            .unwrap()
            .checked_sub(raw_spans[0].new_allocation_bytes().unwrap())
            .unwrap();
        assert!(hook > 0);
        let mut prefill_spans = 0;
        let mut decode_spans = 0;
        for (plain, selected) in raw_spans.iter().zip(bound_spans) {
            assert_eq!(plain.span(), selected.span());
            let extra = match plain.span() {
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Sampling(_) => panic!("model scheduler emitted a sampling phase"),
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Prefill(_) => {
                    prefill_spans += 1;
                    hook
                }
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Decode { .. } => {
                    decode_spans += 1;
                    0
                }
            };
            assert_eq!(
                selected.new_allocation_bytes(),
                Some(
                    plain
                        .new_allocation_bytes()
                        .unwrap()
                        .checked_add(extra)
                        .unwrap()
                )
            );
        }
        assert_eq!((prefill_spans, decode_spans), (2, 4));
        assert_eq!(raw.equations.peak_span(), bound.equations.peak_span());
        assert!(matches!(
            raw.equations.peak_span(),
            Some(eredu_runtime::working_memory::InferenceWorkspaceSpan::Prefill(_))
        ));
        assert_eq!(
            bound.equations.transient().bytes(),
            raw.equations
                .transient()
                .bytes()
                .and_then(|bytes| bytes.checked_add(hook))
        );
        // Decode already required this host hook on the raw route, so its
        // independent host peak is unchanged even though the largest combined
        // tensor-plus-host span (prefill) grows by the hook cost.
        assert_eq!(
            bound.equations.host_peak_bytes(),
            raw.equations.host_peak_bytes()
        );
        assert_eq!(
            raw_h.initialization_peak_bytes(),
            bound_h.initialization_peak_bytes()
        );
        assert!(std::ptr::eq(raw_h.source(), &source));
        assert!(std::ptr::eq(bound_h.source(), selected.source()));
        let error = executable
            .quote_replicated_resident_text_with_sampling_and_capture(
                g,
                config,
                &TokenFilter::All,
                &active,
            )
            .unwrap_err();
        assert!(has_cause::<eredu_runtime::capture::CaptureProtocolError>(
            &error,
            |e| matches!(
                e,
                eredu_runtime::capture::CaptureProtocolError::PrefillAttribution
            )
        ));
        let after = executable
            .quote_replicated_resident_text_with_sampling(g, config, &TokenFilter::All)
            .unwrap();
        same_quote(&plain, &after);
        assert_eq!(
            (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
            before
        );
        drop((plain, raw, raw_h, bound, bound_h, after));
        cold.assert_cold();
    }
    retire(runtime, &pool, &stream);
}
