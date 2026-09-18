//! Real private admission -> original core context -> installation -> prediction.
//! Native payloads are never prepared by this fixture outside that admitted path.
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
};
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    TextGenerationBackend, TextGenerationDriver, TextGenerationInput, TextPreparationOptions,
    capture::*, observation::TensorObservationData,
};
use eredu_runtime::working_memory::CaptureRunHostPlan;

type Runtime = ModelRuntime<MlxBackend<'static>>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Active,
    Sequence,
    Empty,
    DecodeOnly,
    SkipPrefill,
}
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
fn source(runtime: &Runtime, mode: Mode) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut raw = CapturePlan::none();
    if mode != Mode::Empty {
        raw.selections.push(CaptureSelection {
            id: "entry-readout".into(),
            path: if mode == Mode::Sequence {
                "model.logits"
            } else {
                "readout.embedding"
            }
            .into(),
            schedule: CaptureSchedule {
                prefill: mode != Mode::DecodeOnly,
                first_prediction: u64::from(mode == Mode::SkipPrefill),
                ..Default::default()
            },
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        });
    }
    raw.limits.per_step = CaptureUsage {
        captures: 1,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.cumulative = CaptureUsage {
        captures: 4,
        ..raw.limits.per_step
    };
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
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a T {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return found;
        }
        error = error.source().expect("preserved original error");
    }
}
fn finish_runtime(runtime: Runtime, stream: &Stream) {
    runtime.synchronize().unwrap();
    drop(runtime);
    stream.synchronize().unwrap();
    safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
        .unwrap()
        .synchronize()
        .unwrap();
}
fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        disk::reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn values(frame: &SharedCapturedStep) -> &[f32] {
    let Some(CapturePayload::SharedTensor(tensor)) = &frame.records()[0].payload else {
        panic!("same original host bank must own the tensor");
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("F32 capture");
    };
    assert!(!values.is_empty());
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
    values
}
fn quoted(runtime: &Runtime, source: &SharedCapturePlan, controller: &disk::Controller) -> u64 {
    let mut geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let capture = CaptureAdmission::new(runtime.session(), geometry, source, eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary())
        .unwrap()
        .with_opening_rows();
    geometry.output = capture.physical_output(geometry.output);
    let storage = ControllerStorageContract::inspect(controller).unwrap();
    let registered = storage
        .pin_registered(controller, runtime.backend().memory_pool())
        .unwrap();
    let ids = vec![2_u32, 5, 7];
    let quote = quote_incremental(
        runtime.session(),
        geometry,
        (ids.capacity() * std::mem::size_of::<u32>()) as u64,
        disk::config(0.0, 1, u64::MAX),
        controller.inference_workspace(4).unwrap(),
        &storage,
        Some(&registered),
        false,
        Some(&capture),
    )
    .unwrap()
    .quote
    .unwrap();
    assert_eq!(quote.geometry(), geometry);
    quote.incremental_bytes()
}

/// Real entry/install path; vectors copied here are test result comparisons,
/// never original admitted native/source inventories or replacement accounts.
fn run(route: usize, chunk: u64, mode: Mode, exact_check: bool) -> (Vec<u32>, Vec<Vec<f32>>) {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, route);
    let source = source(&runtime, mode);
    let alias = source.clone();
    let h = CaptureRunHostPlan::prepare(&source)
        .unwrap()
        .initialization_peak_bytes();
    let c = source.capacity_bytes().unwrap();
    let controller = disk::Controller::default();
    let selector = Selection::new(&runtime, &source);
    let probe = CaptureFundingProbe::new(&source);
    let capacity = if exact_check {
        assert_eq!(chunk, 1);
        let baseline = pool.used_bytes().unwrap();
        let native = paths::snapshot();
        let inputs = paths::session_input_creation_attempts();
        let frontier = runtime.session().payload.model.erased().state_snapshot();
        let required = quoted(&runtime, &source, &controller);
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(paths::snapshot(), native);
        assert_eq!(paths::session_input_creation_attempts(), inputs);
        let exact = baseline.checked_add(required).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let error = driver
            .start_input_with_options(
                TextGenerationInput::TokenIds(vec![2, 5, 7]),
                disk::config(0.0, chunk, exact - 1),
                controller.clone(),
                TextPreparationOptions {
                    interventions: None, capture: Some(source.clone()),
                },
            )
            .err()
            .expect("one byte short must fail before prompt/controller work");
        assert!(matches!(
            cause::<WorkingMemoryError>(&error),
            WorkingMemoryError::BudgetExceeded { .. }
        ));
        assert_eq!(controller.0.get(), (0, 0));
        assert_eq!(paths::snapshot(), native);
        assert_eq!(paths::session_input_creation_attempts(), inputs);
        assert_eq!(
            driver
                .runtime()
                .session()
                .payload
                .model
                .erased()
                .state_snapshot(),
            frontier
        );
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert!(probe.is_empty());
        assert!(selector.0.quote.borrow().is_none());
        assert_eq!(selector.0.calls.get(), 1);
        exact
    } else {
        u64::MAX
    };
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start_input_with_options(
            TextGenerationInput::TokenIds(vec![2, 5, 7]),
            disk::config(0.0, chunk, capacity),
            controller.clone(),
            TextPreparationOptions {
                interventions: None, capture: Some(source.clone()),
            },
        )
        .unwrap();
    let quote = selector.take_quote();
    let original = probe.take();
    assert_eq!(selector.0.calls.get(), 1 + usize::from(exact_check));
    assert_eq!(
        original.reservation,
        quote.request().memory_reservation().unwrap().bytes()
    );
    assert_eq!(original.capture, h);
    assert_eq!(original.source, c);
    assert_eq!(quote.request().geometry().prefill_chunk_positions, chunk);
    assert_eq!(
        quote.request().geometry().output,
        if mode == Mode::Sequence {
            OutputDemand::Sequence
        } else {
            OutputDemand::LastPosition
        }
    );
    let rows = quote.capture.as_ref().unwrap().opening_rows();
    assert_eq!(
        rows.is_some(),
        matches!(mode, Mode::Active | Mode::Sequence)
    );
    if let Some(rows) = &rows {
        assert_eq!(rows.snapshot_for_test(), (0, None, vec![], vec![]));
        assert!(std::ptr::eq(
            rows.as_ref(),
            quote
                .capture
                .as_ref()
                .unwrap()
                .opening_rows()
                .unwrap()
                .as_ref()
        ));
    }
    {
        // This is the real machine's read-only quiescent view, after actual
        // core context binding and text_capture::install completed.
        let boundary = driver.quiescent(&mut continuation).unwrap();
        let (runtime, state, _) = boundary.parts();
        let installed = state
            .funded_capture
            .as_ref()
            .expect("actual installed original bank");
        assert!(installed.source().same_storage(&source));
        installed.validate_sources(&pool).unwrap();
        installed
            .span_workspace()
            .control_guard()
            .validate_reservation(quote.request().memory_reservation().unwrap())
            .unwrap();
        assert_eq!(
            installed.span_workspace().protected_host_bytes(),
            original.protected
        );
        assert_eq!(installed.collector().usage(), CaptureUsage::default());
        let again = quote
            .take_capture_installation(runtime.session(), &source)
            .err()
            .unwrap();
        assert!(matches!(
            cause::<WorkingMemoryError>(&again),
            WorkingMemoryError::PreparationAlreadyStarted
        ));
    }
    assert_eq!(controller.0.get(), (0, 0));
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    let mut numerical = Vec::new();
    let mut captures = 0;
    for prediction in 0..4_u64 {
        let token = driver.advance(&mut continuation).unwrap().unwrap();
        tokens.push(token.token_id());
        drop(token);
        assert!(driver.capture_pending(&continuation).unwrap());
        let delivery = driver.take_completed_delivery(&mut continuation).unwrap();
        if mode == Mode::Empty {
            assert!(
                delivery.is_none(),
                "empty source has no frame or frame claim"
            );
            assert!(!driver.capture_pending(&continuation).unwrap());
            let boundary = driver.quiescent(&mut continuation).unwrap();
            let (_, state, _) = boundary.parts();
            let collector = state.funded_capture.as_ref().unwrap().collector();
            assert_eq!(collector.spent_steps(), 0);
            assert_eq!(collector.usage(), CaptureUsage::default());
            assert_eq!(
                controller.0.get(),
                ((prediction + 1) as usize, (prediction + 1) as usize)
            );
            continue;
        }
        let frame = delivery.unwrap();
        assert!(!driver.capture_pending(&continuation).unwrap());
        assert!(
            driver
                .take_completed_delivery(&mut continuation)
                .unwrap()
                .is_none()
        );
        assert_eq!(frame.prediction_index(), prediction);
        assert_eq!(
            frame.phase(),
            if prediction == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            }
        );
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        assert_eq!(frame.records().len(), usize::from(mode != Mode::Empty));
        let selected = matches!(mode, Mode::Active | Mode::Sequence)
            || (mode != Mode::Empty && prediction > 0);
        if selected {
            captures += 1;
            assert_eq!(frame.records()[0].selection_id, "entry-readout");
            assert_eq!(frame.records()[0].outcome, CaptureOutcome::Captured);
            numerical.push(values(&frame).to_vec());
        } else if mode != Mode::Empty {
            assert_eq!(
                frame.records()[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            );
            assert!(frame.records()[0].payload.is_none());
        }
        assert_eq!(frame.step_usage().captures, u64::from(selected));
        assert_eq!(frame.cumulative_usage().captures, captures);
        let frame_alias = frame.clone();
        assert!(frame_alias.same_storage(&frame));
        assert_eq!(frame_alias.records().as_ptr(), frame.records().as_ptr());
        drop(frame_alias);
        if let Some(rows) = &rows {
            assert_eq!(
                rows.snapshot_for_test(),
                (3_usize.div_ceil(chunk as usize), None, vec![], vec![])
            );
        }
        frames.push(frame);
        {
            let boundary = driver.quiescent(&mut continuation).unwrap();
            let (_, state, _) = boundary.parts();
            let collector = state.funded_capture.as_ref().unwrap().collector();
            assert_eq!(collector.spent_steps(), (prediction + 1) as usize);
            assert_eq!(collector.usage().captures, captures);
        }
        assert_eq!(
            controller.0.get(),
            ((prediction + 1) as usize, (prediction + 1) as usize)
        );
    }
    if matches!(mode, Mode::Active | Mode::Sequence) {
        assert_eq!(numerical[0].len(), numerical[1].len() * 3);
    }
    let inputs = paths::session_input_creation_attempts();
    assert!(driver.advance(&mut continuation).unwrap().is_none());
    assert!(driver.advance(&mut continuation).unwrap().is_none());
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (4, 4));
    assert_eq!(selector.0.calls.get(), 1 + usize::from(exact_check));
    if exact_check {
        assert!(pool.peak_bytes().unwrap() <= capacity);
    }
    drop((continuation, driver, rows, quote, probe, selector));
    finish_runtime(runtime, &stream);
    // Original frame H and original P+Q+finite-S+C have distinct final owners.
    settle(
        &pool,
        if frames.is_empty() {
            original.source_tail()
        } else {
            h + original.source_tail()
        },
    );
    drop(frames);
    settle(&pool, original.source_tail());
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), original.source_tail());
    assert_eq!(alias.capacity_bytes(), Some(c));
    drop(alias);
    settle(&pool, 0);
    (tokens, numerical)
}

#[test]
fn private_entry_installs_its_one_original_bank_and_captures_nonzero_prefill_plus_three_decodes() {
    let (expected_tokens, expected_values) = run(0, 3, Mode::Active, false);
    for route in 0..3 {
        let (tokens, values) = run(route, 1, Mode::Active, true);
        assert_eq!(tokens, expected_tokens, "route {route}");
        assert_eq!(values.len(), expected_values.len());
        for (got, expected) in values.iter().zip(&expected_values) {
            assert_eq!(got.len(), expected.len());
            for (a, b) in got.iter().zip(expected) {
                assert!(
                    (a - b).abs() <= 1e-5 + 1e-5 * b.abs(),
                    "route {route}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn private_entry_empty_decode_only_and_skipped_prefill_install_without_an_opening_row_bank() {
    for route in 0..3 {
        for mode in [Mode::Empty, Mode::DecodeOnly, Mode::SkipPrefill] {
            let (tokens, values) = run(route, 1, mode, true);
            assert_eq!(tokens.len(), 4);
            assert_eq!(values.len(), if mode == Mode::Empty { 0 } else { 3 });
        }
    }
}

#[test]
fn private_entry_selection_is_exact_source_and_session_scoped_and_resets_on_unwind() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let other_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (other, _other_artifact) = load(&stream, &other_pool, 0);
    let source = source(&runtime, Mode::Active);
    let equal_source = self::source(&runtime, Mode::Active);
    let controller = disk::Controller::default();
    let native = paths::snapshot();
    let used = pool.used_bytes().unwrap();
    let input = disk::evidence(&vec![2, 5, 7]);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _selection = Selection::new(&runtime, &source);
        assert!(selected(&runtime, &source).is_some());
        assert!(
            admit_if_selected(
                &runtime,
                &input,
                disk::config(0.0, 1, u64::MAX),
                &controller,
                &equal_source
            )
            .is_none()
        );
        assert!(
            admit_if_selected(
                &other,
                &input,
                disk::config(0.0, 1, u64::MAX),
                &controller,
                &source
            )
            .is_none()
        );
        panic!("fixture unwind");
    }));
    assert!(panic.is_err());
    assert!(selected(&runtime, &source).is_none());
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(paths::snapshot(), native);
    assert_eq!(pool.used_bytes().unwrap(), used);
    drop(Selection::new(&runtime, &source));
    drop((source, equal_source));
    finish_runtime(runtime, &stream);
    finish_runtime(other, &stream);
    settle(&pool, 0);
    settle(&other_pool, 0);
}

#[test]
fn private_sequence_gateway_captures_full_logits_and_preserves_ordinary_tokens_across_weight_routes()
 {
    let (expected_tokens, expected_values) = run(0, 3, Mode::Sequence, false);
    let (body_tokens, _) = run(0, 3, Mode::Active, false);
    assert_eq!(expected_tokens, body_tokens);
    for route in 0..3 {
        for chunk in [1, 2] {
            let (tokens, values) = run(route, chunk, Mode::Sequence, chunk == 1);
            assert_eq!(tokens, expected_tokens, "route {route}, chunk {chunk}");
            assert_eq!(values.len(), expected_values.len());
            for (got, expected) in values.iter().zip(&expected_values) {
                assert_eq!(got.len(), expected.len());
                for (a, b) in got.iter().zip(expected) {
                    assert!(
                        (a - b).abs() <= 1e-5 + 1e-5 * b.abs(),
                        "route {route}, chunk {chunk}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
