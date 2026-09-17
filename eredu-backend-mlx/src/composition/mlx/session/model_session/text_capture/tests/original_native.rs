//! Real public core original-input options, including full prefill row capture.
//! Compares ordinary capture with original managed and controlled entry points.
use super::*;
use eredu_core::observation::TensorObservationData;
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, TextGeneration, TextPreparationOptions,
    TokenIdsInputPlan, TokenTerminalSignals,
};

const CASE: &str = "composition::mlx::session::model_session::text_capture::tests::original_native::original_input_capture_matches_full_prefill_and_controlled_decode";
const MODE: &str = "EREDU_ORIGINAL_CAPTURE_CORE_MODE";
const PREFIX: &str = "ORIGINAL_CAPTURE_CORE_RESULT:";

fn full_source(runtime: &Runtime) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let usage = CaptureUsage {
        captures: 4,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "original-full-logits".into(),
        path: "model.logits".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    raw.limits.per_step = usage;
    raw.limits.cumulative = usage;
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
fn failure<T>(error: impl std::error::Error + 'static) -> T {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(error) = cause {
        eprintln!("ORIGINAL_CAPTURE_FAILURE: {error}");
        cause = error.source();
    }
    panic!("original capture source chain above");
}
fn frame(delivery: CapturedStepDelivery, prediction: u64) -> CapturedStepDelivery {
    let frame = delivery.as_step();
    assert_eq!(frame.prediction_index, prediction);
    assert_eq!(
        frame.phase,
        if prediction == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        }
    );
    assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
    assert_eq!(frame.records.len(), 1);
    let record = &frame.records[0];
    assert_eq!(record.selection_id, "original-full-logits");
    let shape = record.selected_shape.as_ref().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 1);
    assert_eq!(
        shape[1],
        if prediction == 0 { 3 } else { 1 },
        "full capture keeps every logical prompt row across chunk boundaries"
    );
    let payload = record.payload.as_ref().unwrap();
    let TensorObservationData::F32(values) = payload.as_tensor().unwrap().data() else {
        panic!("expected F32 host values")
    };
    assert_eq!(values.len() as u64, shape.iter().product::<u64>());
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 1e-6));
    delivery
}
fn run(mode: &str) -> serde_json::Value {
    // Original input entry points require the same admitted stream owners as
    // the public factory; the older borrowed-stream fixture cannot grant them.
    let pool = crate::backend::managed_memory::domain();
    let streams =
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_factory(&pool)
            .unwrap()
            .expect("admitted native stream factory");
    let identity = crate::backend::MlxDeviceIdentity::from_realized_device(
        &safemlx::Device::new(safemlx::DeviceType::Gpu, 0),
        Some(crate::backend::MlxAcceleratorFamily::Metal),
    )
    .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(streams, identity);
    let _artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, _artifact.path(), crate::MlxLoadRequest::default())
            .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let source = full_source(&runtime);
    let options = || TextPreparationOptions {
        interventions: None, capture: Some(source.clone()),
    };
    let config = disk::config(0.0, if mode == "ordinary" { 3 } else { 2 }, 8 << 30);
    let config = if mode == "ordinary" {
        let mut policy = config.inference_policy();
        policy.managed_memory_capacity_bytes = None;
        config.with_inference_policy(policy)
    } else {
        config
    };
    let mut ids = [2, 5, 7];
    let mut output = Vec::new();
    let mut frames = Vec::new();
    if mode == "ordinary" {
        let mut run = ControlledTextGeneration::new(
            &mut runtime,
            ids.to_vec(),
            config,
            disk::Controller::default(),
        )
        .unwrap_or_else(failure);
        run.enable_capture(source.admission().clone())
            .unwrap_or_else(failure);
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap_or_else(failure);
            output.push(token.token_id());
            drop(token);
            frames.push(frame(
                run.take_captured_delivery().unwrap().unwrap(),
                prediction,
            ));
        }
        assert!(run.next().is_none());
    } else if mode == "managed" {
        let mut run = TextGeneration::from_token_ids_with_sequence(
            &mut runtime,
            TokenIdsInputPlan::new(&ids).unwrap(),
            config,
            TokenFilter::All,
            Some(options()),
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap_or_else(failure);
        ids.fill(1); // The original I destination has already copied the actual source.
        let mut sequence = run
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap_or_else(failure);
            let id = token.token_id().unwrap();
            drop(token);
            sequence
                .commit(id, TokenTerminalSignals::default())
                .unwrap();
            output.push(id);
            frames.push(frame(
                run.take_captured_delivery().unwrap().unwrap(),
                prediction,
            ));
        }
        assert_eq!(sequence.tokens(), output);
        assert!(run.next().is_none());
    } else {
        assert_eq!(mode, "controlled");
        let mut run = ControlledTextGeneration::from_token_ids_with_sequence(
            &mut runtime,
            TokenIdsInputPlan::new(&ids).unwrap(),
            config,
            disk::Controller::default(),
            Some(options()),
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap_or_else(failure);
        ids.fill(1);
        let mut sequence = run
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap_or_else(failure);
            let id = token.token_id();
            drop(token);
            sequence
                .commit(id, TokenTerminalSignals::default())
                .unwrap();
            output.push(id);
            frames.push(frame(
                run.take_captured_delivery().unwrap().unwrap(),
                prediction,
            ));
        }
        assert_eq!(sequence.tokens(), output);
        assert!(run.next().is_none());
    }
    runtime.synchronize().unwrap();
    drop((runtime, source));
    // Inspect escaped host values only after the source/session/native owners retire.
    let values: Vec<Vec<f32>> = frames
        .iter()
        .map(|frame| {
            let TensorObservationData::F32(values) = frame.as_step().records[0]
                .payload
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap()
                .data()
            else {
                unreachable!()
            };
            values.to_vec()
        })
        .collect();
    if mode != "ordinary" {
        assert!(frames.iter().all(|frame| frame.shared().is_some()));
        assert!(
            pool.used_bytes().unwrap() > 0,
            "escaped frames retain original custody"
        );
    }
    let with_frames = pool.used_bytes().unwrap();
    drop(frames);
    if mode != "ordinary" {
        // Process initialization has independent persistent custody. The escaped
        // frame owners must release their own original request hold.
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            pool.used_bytes().unwrap() < with_frames
        });
    }
    serde_json::json!({"ids":output,"values":values})
}

#[test]
fn original_input_capture_matches_full_prefill_and_controlled_decode() {
    if let Ok(mode) = std::env::var(MODE) {
        println!("{PREFIX}{}", run(&mode));
        return;
    }
    let mut reference: Option<serde_json::Value> = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "{mode}:\n{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        let stdout = String::from_utf8(child.stdout).unwrap();
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.split_once(PREFIX).map(|(_, value)| value))
                .expect("child result"),
        )
        .unwrap();
        if let Some(expected) = &reference {
            assert_eq!(actual["ids"], expected["ids"], "{mode}");
            let a: Vec<Vec<f32>> = serde_json::from_value(actual["values"].clone()).unwrap();
            let b: Vec<Vec<f32>> = serde_json::from_value(expected["values"].clone()).unwrap();
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(&b) {
                assert_eq!(a.len(), b.len());
                for (&a, &b) in a.iter().zip(b) {
                    assert!(
                        (a - b).abs() <= 1e-5 + 1e-5 * b.abs(),
                        "{mode}: {a} versus {b}"
                    );
                }
            }
        } else {
            reference = Some(actual);
        }
    }
}
