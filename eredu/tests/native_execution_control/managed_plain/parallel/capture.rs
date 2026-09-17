//! Public observation capture uses the same real Ring loader and bounded process runner.
use super::*;
use eredu_core::observation::TensorObservationData;

const HISTOGRAM_EDGES: [f32; 6] = [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0];

const CASE: &str = "managed_plain::parallel::capture::native_parallel_logits_receipts_match_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_MANAGED_TP_CAPTURE_MODE";
const RESULT: &str = "PUBLIC_MANAGED_TP_CAPTURE_RESULT:";
const IDS: [&str; 4] = ["global logits λ", "strided vocabulary 雪", "prefix preview", "global summary Ω"];
const SUMMARY_CASE: &str = "managed_plain::parallel::capture::native_parallel_summary_receipts_match_ordinary_and_controlled";
const SUMMARY_MODE: &str = "EREDU_PUBLIC_MANAGED_TP_SUMMARY_MODE";
const SUMMARY_RESULT: &str = "PUBLIC_MANAGED_TP_SUMMARY_RESULT:";
const UNIT_CASE: &str = "managed_plain::parallel::capture::native_parallel_unit_output_receipts_match_ordinary_and_controlled";
const UNIT_MODE: &str = "EREDU_PUBLIC_MANAGED_TP_UNIT_CAPTURE_MODE";
const UNIT_RESULT: &str = "PUBLIC_MANAGED_TP_UNIT_CAPTURE_RESULT:";

fn capture_path<const UNIT: bool>() -> &'static str {
    if UNIT { "model.layers.0.output" } else { eredu_core::MODEL_LOGITS_OBSERVATION_PATH }
}
fn capture_width<const UNIT: bool>() -> u64 { if UNIT { 16 } else { 64 } }
fn capture_ids<const UNIT: bool>() -> [&'static str; 4] {
    if UNIT { ["unit output λ", "strided hidden 雪", "unit prefix preview", "unit summary Ω"] } else { IDS }
}

fn plan<const REDUCTION: u8, const UNIT: bool>() -> CapturePlan {
    let mut plan = CapturePlan::none();
    for (index, transform) in [CaptureTransform::FullTensor, CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 7 }].into_iter().enumerate() {
        plan.selections.push(CaptureSelection {
            id: capture_ids::<UNIT>()[index].into(), path: capture_path::<UNIT>().into(),
            schedule: CaptureSchedule::default(),
            slices: if index == 1 { vec![CaptureSlice {
                axis: if UNIT { "hidden" } else { "vocabulary" }.into(),
                start: 1, end: capture_width::<UNIT>() - 1, stride: 3,
            }] } else { vec![] },
            transform,
        });
    }
    if REDUCTION != 0 {
        plan.selections.push(CaptureSelection {
            id: capture_ids::<UNIT>()[3].into(), path: capture_path::<UNIT>().into(),
            schedule: CaptureSchedule::default(), slices: vec![], transform: match REDUCTION {
                1 => CaptureTransform::Summary,
                2 => CaptureTransform::Histogram { edges: HISTOGRAM_EDGES.to_vec() },
                _ => panic!("unknown fixture reduction"),
            },
        });
    }
    // The source-derived receipt bound is below 64 KiB for each of these
    // selections (at most 320 F32 values, rank three, and bounded context labels).
    // Two receivers * three selections * (128 * 64 KiB parser storage), plus
    // gather/framing/evidence and one producer, fit in 64 MiB of host allowance.
    // Gather retention is below 8 MiB; all receiver/producer encodings and
    // evidence fit in 1 MiB. These are logical protocol limits, independent of
    // the managed request's native/metadata permission. Four committed frames
    // consume four steps; unused credits are never refunded by the fixture.
    plan.limits.per_step = CaptureUsage { captures: 16, retained_bytes: 8 << 20,
        host_bytes: 64 << 20, encoded_bytes: 1 << 20 };
    plan.limits.cumulative = CaptureUsage { captures: 64, retained_bytes: 32 << 20,
        host_bytes: 256 << 20, encoded_bytes: 4 << 20 };
    if REDUCTION != 0 {
        // One additional receipt/receiver row; the same source-bound equations
        // apply, while the numerical summary has only fixed scalar output.
        plan.limits.per_step.host_bytes = 96 << 20;
        plan.limits.per_step.retained_bytes = 12 << 20;
        plan.limits.per_step.encoded_bytes = 2 << 20;
        plan.limits.cumulative.host_bytes = 384 << 20;
        plan.limits.cumulative.retained_bytes = 48 << 20;
        plan.limits.cumulative.encoded_bytes = 8 << 20;
    }
    plan
}

fn run<const REDUCTION: u8, const UNIT: bool>(mode: &str) -> serde_json::Value {
    run_topology::<REDUCTION, UNIT>(mode, eredu_core::ParallelTopology::new(2,1,1,1).unwrap())
}
fn run_topology<const REDUCTION: u8, const UNIT: bool>(mode: &str,
    topology: eredu_core::ParallelTopology) -> serde_json::Value {
    if mode == "serial" {
        let root = managed_fixture(fixture(false));
        let execution = ExecutionPlan::fully_resident(
            eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
        let (model, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(), &root.0, &execution).unwrap().into_parts();
        return capture_loaded::<REDUCTION, UNIT>("ordinary", model, root, false);
    }
    let worker: fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, Fixture)
        -> serde_json::Value = match (mode, topology.world_size()) {
        ("ordinary", 2) => ordinary::<REDUCTION, UNIT>,
        ("managed", 2) => managed::<REDUCTION, UNIT>,
        ("controlled", 2) => controlled::<REDUCTION, UNIT>,
        ("ordinary", 4) => ordinary_four::<REDUCTION, UNIT>,
        ("managed", 4) => managed_four::<REDUCTION, UNIT>,
        ("controlled", 4) => controlled_four::<REDUCTION, UNIT>,
        _ => panic!("unknown capture mode or fixture world"),
    };
    run_partitioned_with_lifecycle(mode, topology, Some(worker))
}
fn ordinary<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded::<REDUCTION, UNIT>("ordinary", model, root, true) }
fn managed<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded::<REDUCTION, UNIT>("managed", model, root, true) }
fn controlled<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded::<REDUCTION, UNIT>("controlled", model, root, true) }

fn ordinary_four<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded_world::<REDUCTION, UNIT, 4>("ordinary", model, root, true) }
fn managed_four<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded_world::<REDUCTION, UNIT, 4>("managed", model, root, true) }
fn controlled_four<const REDUCTION: u8, const UNIT: bool>(model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture)
    -> serde_json::Value { capture_loaded_world::<REDUCTION, UNIT, 4>("controlled", model, root, true) }

fn plan_world<const REDUCTION: u8, const UNIT: bool, const WORLD: usize>() -> CapturePlan {
    let mut plan = plan::<REDUCTION, UNIT>();
    if WORLD > 2 {
        // One gather has W retained outputs and the existing global equation
        // reserves its work for W participants. Scale the two-rank host/wire
        // allowance by this exact quadratic population; capture count is fixed.
        let factor = u64::try_from(WORLD.checked_mul(WORLD).unwrap() / 4).unwrap();
        for usage in [&mut plan.limits.per_step, &mut plan.limits.cumulative] {
            usage.host_bytes = usage.host_bytes.checked_mul(factor).unwrap();
            usage.retained_bytes = usage.retained_bytes.checked_mul(factor).unwrap();
            usage.encoded_bytes = usage.encoded_bytes.checked_mul(factor).unwrap();
        }
    }
    plan
}
fn capture_loaded<const REDUCTION: u8, const UNIT: bool>(mode: &str, model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture, partitioned: bool) -> serde_json::Value {
    capture_loaded_world::<REDUCTION, UNIT, 2>(mode, model, root, partitioned)
}
fn capture_loaded_world<const REDUCTION: u8, const UNIT: bool, const WORLD: usize>(mode: &str,
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture, partitioned: bool) -> serde_json::Value {
    capture_loaded_plan(mode,model,root,partitioned,|_|plan_world::<REDUCTION,UNIT,WORLD>(),
        evidence_and_values::<REDUCTION,UNIT>)
}
fn capture_loaded_plan(mode:&str,
    model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture,partitioned:bool,
    make_plan:impl FnOnce(&eredu_core::capture::CaptureDiscovery)->CapturePlan,
    evaluate:impl FnOnce(&[u32],&str,&[&CapturedStep],bool)->serde_json::Value)->serde_json::Value {
    capture_loaded_plan_with_capacity(mode,model,root,partitioned,None,make_plan,evaluate)
}
fn capture_loaded_plan_with_capacity(mode:&str,
    mut model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture,partitioned:bool,
    managed_capacity:Option<u64>,
    make_plan:impl FnOnce(&eredu_core::capture::CaptureDiscovery)->CapturePlan,
    evaluate:impl FnOnce(&[u32],&str,&[&CapturedStep],bool)->serde_json::Value)->serde_json::Value {
    // This public discovery also retains the loaded artifact/execution labels.
    let discovery = model.capture_discovery().unwrap_or_else(report_failure);
    let selected=make_plan(&discovery);
    if mode == "ordinary" {
        let chat = model.prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
            add_generation_prompt: true, ..Default::default()
        }).unwrap();
        let mut settings = settings(0.0);
        settings.inference.managed_memory_capacity_bytes = None;
        let prepared = model.prepare_observed_token_ids(&chat, vec![0,1,2,3,4], settings,
            selected, TraceLimits { per_record_bytes: 1 << 20, total_bytes: 4 << 20 })
            .unwrap_or_else(report_failure);
        let mut ids = Vec::new();
        let mut frames = Vec::new();
        model.generate_observed_text(prepared, &[], Default::default(), |record| {
            if let ObservedGenerationEvent::Token { token_id, captures: Some(frame), .. } = record.event {
                ids.push(token_id); frames.push(frame);
            }
            ControlFlow::Continue(())
        }).unwrap_or_else(report_failure);
        let text = model.decode(&ids, true).unwrap();
        drop((model, root));
        let frames: Vec<_> = frames.iter().collect();
        return evaluate(&ids, &text, &frames, partitioned);
    }
    let source = model.compile_managed_plain_text_source(
        std::fs::File::open(root.0.join("tokenizer.json")).unwrap()).unwrap_or_else(report_failure);
    let capture = SharedCapturePlan::new(selected.admit(&discovery.catalog, &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape { batch: 1, prompt_tokens: 5, max_predictions: 4 }).unwrap());
    let mut deliveries = Vec::new();
    let mut observer = |token: Option<u32>, frame: Option<CapturedStepDelivery>, seconds: f64| {
        if token.is_none() {
            eprintln!("PUBLIC_CAPTURE_NO_TOKEN_DELIVERY: outcome={:?} records={:?}",
                frame.as_ref().map(|frame| frame.as_step().outcome),
                frame.as_ref().map(|frame| frame.as_step().records.len()));
        }
        // Failure/cancellation can deliver its terminal frame without a token.
        // Keep its exact owner and let the operation report the original error
        // before applying successful-run fixture assertions.
        deliveries.push((token, frame, seconds));
    };
    let cancellation = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut finishes = Vec::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| match event {
        GenerationPlainTextEvent::TextDelta(text) => visible.push_str(text),
        GenerationPlainTextEvent::Finished { reason } => finishes.push(reason),
    };
    let mut settings = settings(0.0);
    if let Some(capacity) = managed_capacity {
        settings.inference.managed_memory_capacity_bytes = Some(capacity);
    }
    let output = if mode == "managed" {
        model.generate_observed_managed_plain_text(&source,
            ManagedPlainTextRequest::new(PROMPT, settings), capture, &cancellation,
            &mut observer, &mut emit).unwrap_or_else(report_failure).expect("live captured request")
    } else {
        let mut session = model.start_observed_managed_plain_text(&source,
            ManagedPlainTextRequest::new(PROMPT, settings), capture, &cancellation,
            &mut observer).unwrap_or_else(report_failure).expect("live captured session");
        while session.finish_reason().is_none() {
            session = session.advance(&cancellation, &mut emit).unwrap_or_else(report_failure);
        }
        session.into_output().unwrap_or_else(|_| panic!("terminal captured session"))
    };
    drop(observer);
    let mut frames = Vec::new();
    let mut ids = Vec::new();
    for (token, frame, seconds) in deliveries {
        assert!(seconds >= 0.0);
        ids.push(token.expect("successful captured request delivered a committed token"));
        let Some(CapturedStepDelivery::Shared(frame)) = frame else { panic!("paid shared receipt frame") };
        assert_eq!(frame.prediction_index() as usize, frames.len());
        frames.push(frame);
    }
    assert_eq!(ids, output.token_ids.as_ref());
    assert_eq!(output.finish_reason, FinishReason::MaxTokens);
    assert_eq!(finishes, [FinishReason::MaxTokens]);
    assert_eq!(visible, output.text.as_str());
    drop(source); drop((model, root));
    // Read original payload owners after the loaded model and source retire.
    // The aliases exercise the same receipt custody without cloning its payload.
    for frame in &frames { assert!(frame.clone().same_storage(frame)); }
    let frames: Vec<_> = frames.iter().map(SharedCapturedStep::as_step).collect();
    evaluate(&ids, output.text.as_str(), &frames, partitioned)
}

fn evidence_and_values<const REDUCTION: u8, const UNIT: bool>(ids: &[u32], text: &str, frames: &[&CapturedStep], partitioned: bool)
    -> serde_json::Value {
    assert_eq!(ids.len(), 4);
    assert_eq!(frames.len(), 4);
    let mut prior: Option<&PartitionCaptureContext> = None;
    let mut result = Vec::new();
    for (prediction, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
        assert_eq!(frame.prediction_index as usize, prediction);
        let phase = if prediction == 0 { CapturePhase::Prefill } else { CapturePhase::Decode };
        assert_eq!(frame.phase, phase);
        assert_eq!(frame.records.len(), if REDUCTION != 0 { 4 } else { 3 });
        assert_eq!(frame.partitions.len(), if partitioned { if REDUCTION != 0 { 4 } else { 3 } } else { 0 });
        for (index, evidence) in frame.partitions.iter().enumerate() {
            evidence.context.validate().unwrap();
            assert_eq!(evidence.schema_version, PARTITION_CAPTURE_SCHEMA_VERSION);
            assert_eq!(evidence.context.selection_index, index);
            assert_eq!(evidence.context.prediction as usize, prediction);
            assert_eq!(evidence.context.phase, phase);
            assert_eq!(evidence.producers, [0]);
            assert!(!evidence.receipt_plan_identity.is_empty());
            assert!(!evidence.contributions.is_empty());
            assert!(evidence.contributions.iter().all(|part| part.producer_rank == 0));
            if let Some(first) = frame.partitions.first() {
                assert_eq!(evidence.context.run_identity, first.context.run_identity);
                assert_eq!(evidence.context.forward_epoch, first.context.forward_epoch);
            }
            if let Some(prior) = prior {
                assert_eq!(evidence.context.artifact_identity, prior.artifact_identity);
                assert_eq!(evidence.context.execution_identity, prior.execution_identity);
                assert_eq!(evidence.context.run_identity, prior.run_identity);
                assert_eq!(evidence.context.capture_plan_identity, prior.capture_plan_identity);
                assert!(evidence.context.forward_epoch > prior.forward_epoch);
            }
        }
        prior = frame.partitions.first().map(|evidence| &evidence.context);
        let sequence = if prediction == 0 { 5 } else { 1 };
        let width = capture_width::<UNIT>();
        let mut rows = Vec::new();
        let mut full: Option<&[f32]> = None;
        for (index, record) in frame.records.iter().enumerate() {
            assert_eq!(record.selection_id, capture_ids::<UNIT>()[index]);
            assert_eq!(record.path, capture_path::<UNIT>());
            assert_eq!(record.source_shape.as_deref(), Some([1, sequence, width].as_slice()));
            let columns = if index == 1 { (width - 2).div_ceil(3) } else { width };
            assert_eq!(record.selected_shape.as_deref(), Some([1, sequence, columns].as_slice()));
            if index == 3 && REDUCTION == 2 {
                let CapturePayload::Histogram(histogram) = record.payload.as_ref().unwrap() else {
                    panic!("histogram receipt");
                };
                let values = full.expect("full-tensor oracle before histogram");
                assert_eq!(histogram.edges, HISTOGRAM_EDGES);
                let mut counts = [0u64; HISTOGRAM_EDGES.len() - 1];
                let (mut below, mut above, mut non_finite) = (0u64, 0u64, 0u64);
                for &value in values {
                    if !value.is_finite() { non_finite += 1; }
                    else if value < HISTOGRAM_EDGES[0] { below += 1; }
                    else if value > *HISTOGRAM_EDGES.last().unwrap() { above += 1; }
                    else {
                        let bin = HISTOGRAM_EDGES.partition_point(|edge| *edge <= value)
                            .saturating_sub(1).min(counts.len() - 1);
                        counts[bin] += 1;
                    }
                }
                assert_eq!(histogram.counts, counts);
                assert_eq!((histogram.below, histogram.above, histogram.non_finite),
                    (below, above, non_finite));
                assert_eq!(counts.iter().sum::<u64>() + below + above + non_finite,
                    values.len() as u64);
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":null,"outcome":record.outcome,"values":histogram.edges,
                    "counts":[histogram.counts, [below,above,non_finite]]}));
                continue;
            }
            if index == 3 {
                let CapturePayload::Summary(summary) = record.payload.as_ref().unwrap() else { panic!("summary receipt") };
                let values = full.expect("full-tensor oracle before summary");
                assert_eq!((summary.elements, summary.finite, summary.non_finite, summary.nan,
                    summary.positive_infinity, summary.negative_infinity),
                    (values.len() as u64, values.len() as u64, 0, 0, 0, 0));
                let expected = [values.iter().copied().fold(f32::INFINITY, f32::min) as f64,
                    values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64,
                    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64,
                    (values.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>() / values.len() as f64).sqrt()];
                let actual = [summary.min.unwrap(), summary.max.unwrap(), summary.mean.unwrap(), summary.rms.unwrap()];
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 3e-5, "summary {actual} versus raw oracle {expected}");
                }
                rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                    "payload_shape":null,"outcome":record.outcome,"values":actual,
                    "counts":[summary.elements,summary.finite,summary.non_finite,summary.nan,
                        summary.positive_infinity,summary.negative_infinity]}));
                continue;
            }
            let tensor = record.payload.as_ref().unwrap().as_tensor().unwrap();
            let TensorObservationData::F32(values) = tensor.data() else { panic!("floating captured activation") };
            assert_eq!(values.len(), if index == 2 { 7 } else { (sequence * columns) as usize });
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().any(|value| value.abs() > 1e-6));
            if index == 0 { full = Some(values); }
            if index == 1 {
                let selected: Vec<_> = full.unwrap().chunks_exact(width as usize)
                    .flat_map(|row| (1..width as usize - 1).step_by(3).map(|column| row[column])).collect();
                assert_eq!(values.as_slice(), selected.as_slice());
            }
            if index == 2 { assert_eq!(values.as_slice(), &full.unwrap()[..7]); }
            rows.push(serde_json::json!({"id":record.selection_id,"shape":record.selected_shape,
                "payload_shape":tensor.shape(),"outcome":record.outcome,"values":values}));
        }
        result.push(serde_json::json!({"prediction":prediction,"rows":rows}));
    }
    serde_json::json!({"ids":ids,"text":text,"frames":result})
}

fn compare(actual: &serde_json::Value, expected: &serde_json::Value, mode: &str, rank: usize) {
    assert_eq!(actual["ids"], expected["ids"], "{mode} rank{rank} ids");
    assert_eq!(actual["text"], expected["text"], "{mode} rank{rank} text");
    let actual = actual["frames"].as_array().unwrap();
    let expected = expected["frames"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual["prediction"], expected["prediction"]);
        let actual = actual["rows"].as_array().unwrap();
        let expected = expected["rows"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            for field in ["id", "shape", "payload_shape", "outcome", "counts"] {
                assert_eq!(actual[field], expected[field], "{mode} rank{rank} {field}");
            }
            let actual = actual["values"].as_array().unwrap();
            let expected = expected["values"].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let (actual, expected) = (actual.as_f64().unwrap(), expected.as_f64().unwrap());
                assert!((actual - expected).abs() <= 2e-5 + 1e-5 * expected.abs(),
                    "{mode} rank{rank} value {index}: {actual} versus {expected}");
            }
        }
    }
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_parallel_logits_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(CASE, MODE, RESULT, "TP capture", 2,
        &["ordinary", "managed", "controlled"], run::<0, false>, compare)
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_parallel_summary_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(SUMMARY_CASE, SUMMARY_MODE, SUMMARY_RESULT, "TP summary capture", 2,
        &["ordinary", "managed", "controlled"], run::<1, false>, compare)
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_parallel_unit_output_receipts_match_ordinary_and_controlled() {
    // The declared first decoder unit is complete on both TP ranks. The same
    // serial observation is the numerical oracle, including all five prefill
    // rows across three chunks and the subsequent cached decode frontiers.
    compare_selected_modes_by(UNIT_CASE, UNIT_MODE, UNIT_RESULT, "TP unit capture", 2,
        &["ordinary", "managed", "controlled"], run::<0, true>, compare)
}

fn saved_unit_capture(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
) -> serde_json::Value {
    super::super::capture::check_saved_capture_loaded(model, root, true,
        capture_path::<true>(), capture_width::<true>() as usize, true)
}
fn run_saved_unit(mode: &str) -> serde_json::Value {
    assert!(matches!(mode, "serial" | "lifecycle"));
    run_partitioned_with_lifecycle(mode, eredu_core::ParallelTopology::new(2,1,1,1).unwrap(),
        Some(saved_unit_capture))
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_parallel_unit_capture_restore_and_fork_preserve_sources_and_spending() {
    compare_selected_modes(
        "managed_plain::parallel::capture::native_parallel_unit_capture_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_MANAGED_TP_UNIT_SAVED_MODE", "PUBLIC_MANAGED_TP_UNIT_SAVED_RESULT:",
        "TP saved unit capture", 2, &["lifecycle"], run_saved_unit,
    )
}


fn run_pipeline_unit(mode: &str) -> serde_json::Value {
    run_topology::<1, true>(mode, eredu_core::ParallelTopology::new(1,2,1,1).unwrap())
}
fn run_combined_unit(mode: &str) -> serde_json::Value {
    run_topology::<1, true>(mode, eredu_core::ParallelTopology::new(2,2,1,1).unwrap())
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_pipeline_unit_capture_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(
        "managed_plain::parallel::capture::native_pipeline_unit_capture_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_PP_UNIT_CAPTURE_MODE", "PUBLIC_MANAGED_PP_UNIT_CAPTURE_RESULT:",
        "PP unit capture", 2, &["ordinary", "managed", "controlled"], run_pipeline_unit, compare,
    )
}
#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_combined_unit_capture_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(
        "managed_plain::parallel::capture::native_combined_unit_capture_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_COMBINED_UNIT_CAPTURE_MODE", "PUBLIC_MANAGED_COMBINED_UNIT_CAPTURE_RESULT:",
        "combined unit capture", 4, &["ordinary", "managed", "controlled"], run_combined_unit, compare,
    )
}

// Fixed-bin receipts reuse the existing Ring process, ordinary raw oracle,
// bounded chunk driver and controlled advancement fixture.
fn run_pipeline_histogram(mode: &str) -> serde_json::Value {
    run_topology::<2, true>(mode, eredu_core::ParallelTopology::new(1,2,1,1).unwrap())
}
fn run_combined_histogram(mode: &str) -> serde_json::Value {
    run_topology::<2, true>(mode, eredu_core::ParallelTopology::new(2,2,1,1).unwrap())
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_parallel_histogram_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(
        "managed_plain::parallel::capture::native_parallel_histogram_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_TP_HISTOGRAM_MODE", "PUBLIC_MANAGED_TP_HISTOGRAM_RESULT:",
        "TP histogram capture", 2, &["ordinary", "managed", "controlled"], run::<2, false>, compare,
    )
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_pipeline_histogram_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(
        "managed_plain::parallel::capture::native_pipeline_histogram_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_PP_HISTOGRAM_MODE", "PUBLIC_MANAGED_PP_HISTOGRAM_RESULT:",
        "PP histogram capture", 2, &["ordinary", "managed", "controlled"], run_pipeline_histogram, compare,
    )
}
#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_combined_histogram_receipts_match_ordinary_and_controlled() {
    compare_selected_modes_by(
        "managed_plain::parallel::capture::native_combined_histogram_receipts_match_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_COMBINED_HISTOGRAM_MODE", "PUBLIC_MANAGED_COMBINED_HISTOGRAM_RESULT:",
        "combined histogram capture", 4, &["ordinary", "managed", "controlled"], run_combined_histogram, compare,
    )
}

#[path = "capture/projected.rs"]
mod projected;

#[path = "capture/routed.rs"]
mod routed;

#[path = "capture/interventions.rs"]
mod interventions;
