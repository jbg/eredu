//! Real sparse provider rows across full prefill, bounded prefill and cached decode.
use super::*;
use eredu_core::capture::{CapturePayload, RoutedUnitCapture};

const CASE: &str =
    "managed_plain::capture::routed::native_routed_capture_matches_ordinary_managed_and_controlled";
const MODE: &str = "EREDU_ORIGINAL_ROUTED_CAPTURE_MODE";
const RESULT: &str = "ORIGINAL_ROUTED_CAPTURE_RESULT:";

#[derive(Clone, Copy)]
enum Residency { Resident, Host, ForegroundDisk }
impl Residency {
    fn plan(self) -> eredu_core::ResidencyPlan {
        match self {
            Self::Resident => eredu_core::ResidencyPlan::FullyResident,
            Self::Host => eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            Self::ForegroundDisk => eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20, host_budget_bytes: 0,
                host_lookahead: 0, background_queue: 0,
            },
        }
    }
    fn fixture(self) -> Fixture {
        let fixture = super::super::super::routed_components::routed_fixture();
        if !matches!(self, Self::Resident) {
            // Three actual nonzero layers exceed Host window1 and disk window2.
            let path = fixture.0.join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            config["num_hidden_layers"] = 3.into();
            std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
            let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
            write_tensor_plan(&fixture.0, resolved.architecture.checkpoint());
        }
        managed_fixture(fixture)
    }
}


fn capture_plan(paths: &[String]) -> CapturePlan {
    let mut plan = CapturePlan::none();
    for (index, path) in paths.iter().enumerate() {
        plan.selections.push(CaptureSelection {
            id: format!("sparse-{index}"),
            path: path.clone(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::RoutedUnits,
        });
    }
    let usage = CaptureUsage {
        captures: 8,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    plan
}
fn sparse(records: &[CaptureRecord], prediction: usize) -> Vec<RoutedUnitCapture> {
    assert_eq!(records.len(), 2);
    records
        .iter()
        .map(|record| {
            assert!(matches!(record.outcome, CaptureOutcome::Captured));
            assert_eq!(record.charged.captures, 1);
            let Some(CapturePayload::RoutedUnits(value)) = &record.payload else {
                panic!("sparse original/effective payload");
            };
            assert_eq!(value.rows.len(), if prediction == 0 { 20 } else { 4 });
            assert!(value.rows.iter().any(|row| {
                let TensorObservationData::F32(values) = row.values.data() else {
                    panic!("float units");
                };
                values.iter().any(|value| value.abs() > 1e-6)
            }));
            value.clone()
        })
        .collect()
}
fn run(mode: &str, residency: Residency) -> serde_json::Value {
    let root = residency.fixture();
    let graph = inspect_architecture(&root.0).unwrap();
    let bank = &graph.routed_components[0];
    let paths = [bank.activation.clone(), bank.effective_activation.clone()];
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_residency(residency.plan());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    if mode == "ordinary" {
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let mut options = settings(0.0);
        options.inference.managed_memory_capacity_bytes = None;
        options.inference.prefill_chunk_positions = None;
        let prepared = model
            .prepare_observed_token_ids(
                &chat,
                vec![0, 1, 2, 3, 4],
                options,
                capture_plan(&paths),
                TraceLimits {
                    per_record_bytes: 1 << 20,
                    total_bytes: 8 << 20,
                },
            )
            .unwrap();
        let mut ids = Vec::new();
        let mut frames = Vec::new();
        model
            .generate_observed_text(prepared, &[], Default::default(), |record| {
                if let ObservedGenerationEvent::Token {
                    token_id,
                    captures: Some(frame),
                    ..
                } = record.event
                {
                    ids.push(token_id);
                    frames.push(frame);
                }
                ControlFlow::Continue(())
            })
            .unwrap_or_else(report_failure);
        drop((model, root));
        assert_eq!(frames.len(), 4);
        let rows: Vec<_> = frames
            .iter()
            .enumerate()
            .map(|(prediction, frame)| sparse(&frame.records, prediction))
            .collect();
        return serde_json::json!({"ids": ids, "rows": rows});
    }
    assert!(matches!(mode, "managed" | "controlled"));
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let discovery = model.capture_discovery().unwrap();
    let capture = SharedCapturePlan::new(
        capture_plan(&paths)
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 5,
                    max_predictions: 4,
                },
            )
            .unwrap(),
    );
    let mut ids = Vec::new();
    let mut frames = Vec::new();
    let mut observer = |token: Option<u32>, frame: Option<CapturedStepDelivery>, seconds: f64| {
        assert!(seconds >= 0.0);
        ids.push(token.expect("committed token"));
        let Some(CapturedStepDelivery::Shared(frame)) = frame else {
            panic!("original shared frame");
        };
        assert_eq!(frame.prediction_index() as usize, frames.len());
        frames.push(frame);
    };
    let cancellation = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(delta) = event {
            visible.push_str(delta);
        }
    };
    let output = if mode == "managed" {
        model
            .generate_observed_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
                capture,
                &cancellation,
                &mut observer,
                &mut emit,
            )
            .unwrap_or_else(report_failure)
            .expect("original request")
    } else {
        let mut session = model
            .start_observed_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
                capture,
                &cancellation,
                &mut observer,
            )
            .unwrap_or_else(report_failure)
            .expect("original request");
        while session.finish_reason().is_none() {
            session = session
                .advance(&cancellation, &mut emit)
                .unwrap_or_else(report_failure);
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("completed session"))
    };
    drop(observer);
    assert_eq!(ids, output.token_ids.as_ref());
    assert_eq!(visible, output.text.as_str());
    assert_eq!(frames.len(), 4);
    drop((source, model, root));
    let rows: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(prediction, frame)| {
            assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
            assert!(frame.clone().same_storage(frame));
            sparse(frame.records(), prediction)
        })
        .collect();
    serde_json::json!({"ids": ids, "rows": rows})
}

fn verify_residency(case: &str, residency: Residency) {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(&mode, residency));
        return;
    }
    let mut reference: Option<serde_json::Value> = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .expect("routed result"),
        )
        .unwrap();
        if mode == "ordinary" {
            reference = Some(actual);
            continue;
        }
        let reference = reference.as_ref().unwrap();
        assert_eq!(actual["ids"], reference["ids"]);
        let actual: Vec<Vec<RoutedUnitCapture>> =
            serde_json::from_value(actual["rows"].clone()).unwrap();
        let expected: Vec<Vec<RoutedUnitCapture>> =
            serde_json::from_value(reference["rows"].clone()).unwrap();
        for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.rows.len(), expected.rows.len());
            for (actual, expected) in actual.rows.iter().zip(&expected.rows) {
                assert_eq!(
                    (
                        actual.token,
                        actual.slot,
                        actual.expert,
                        actual.unit_start,
                        actual.unit_stride
                    ),
                    (
                        expected.token,
                        expected.slot,
                        expected.expert,
                        expected.unit_start,
                        expected.unit_stride
                    )
                );
                assert!((actual.coefficient - expected.coefficient).abs() < 5e-5);
                let (TensorObservationData::F32(actual), TensorObservationData::F32(expected)) =
                    (actual.values.data(), expected.values.data())
                else {
                    panic!("float rows");
                };
                assert_eq!(actual.len(), expected.len());
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 5e-5));
            }
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_routed_capture_matches_ordinary_managed_and_controlled() {
    verify_residency(CASE, Residency::Resident);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_routed_host_capture_matches_ordinary_managed_and_controlled() {
    verify_residency("managed_plain::capture::routed::native_routed_host_capture_matches_ordinary_managed_and_controlled",
        Residency::Host);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_routed_disk_capture_matches_ordinary_managed_and_controlled() {
    verify_residency("managed_plain::capture::routed::native_routed_disk_capture_matches_ordinary_managed_and_controlled",
        Residency::ForegroundDisk);
}

fn check_routed_saved(after_commit: bool) {
    let root = managed_fixture(super::super::super::routed_components::routed_fixture());
    let graph = inspect_architecture(&root.0).unwrap();
    let bank = &graph.routed_components[0];
    let execution = ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(), &root.0, &execution).unwrap().into_parts();
    super::saved::check_saved_capture_loaded_with_transform(
        model, root, after_commit, &bank.activation, 6, false,
        CaptureTransform::RoutedUnits, None);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_routed_capture_snapshot_restore_and_fork_preserve_spending() {
    check_routed_saved(true);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_routed_pending_capture_restore_keeps_uneven_prefill_coordinates() {
    check_routed_saved(false);
}
