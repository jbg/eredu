use super::*;

pub(super) fn configuration(hybrid: bool) -> serde_json::Value {
    let mut config = if hybrid {
        conditional_qwen_partition_config(false)
    } else {
        qwen_vl_partition_config(false)
    };
    if hybrid {
        config["text_config"]["linear_conv_kernel_dim"] = 4.into();
    }
    // Both actual vision blocks contribute retained DeepStack values, including
    // values needed only by decoder spans after the first text-only span.
    config["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([0, 1]);
    config
}
pub(super) fn fixture(config: &serde_json::Value) -> tempfile::TempDir {
    prepared_adapter::payload_fixture_config_with(config, 1.0, |name, shape| {
        let scalar = if name.ends_with("A_log") {
            Some(-0.4)
        } else if name.ends_with("dt_bias") {
            Some(-1.2)
        } else if name.ends_with("input_min") || name.ends_with("output_min") {
            // Gemma checkpoint clipping endpoints must form a valid interval.
            Some(-4.0)
        } else if name.ends_with("input_max") || name.ends_with("output_max") {
            Some(4.0)
        } else {
            None
        };
        scalar.map(|v| {
            NumericTensor::new(
                shape.to_vec(),
                vec![v; shape.iter().map(|&n| n as usize).product()],
            )
        })
    })
    .0
}
pub(super) fn input() -> Input {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload};
    Input::new(
        vec![
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[1, 2])),
                [],
            )
            .unwrap(),
            PreparedInputPart::new_with_extents(
                InputModality::Image,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [16, 24],
                    (0..384).map(|i| ((i % 37) as f32 - 18.0) / 80.0).collect(),
                )),
                [(
                    InputMetadataKey::PatchGrid,
                    NumericTensor::new([1, 3], vec![1., 4., 4.]),
                )],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 4,
                    width: 4,
                }],
            )
            .unwrap(),
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[4])),
                [],
            )
            .unwrap(),
        ],
        |tensor| eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor),
    )
    .unwrap()
}
pub(super) fn residencies() -> [eredu_core::ResidencyPlan; 3] {
    [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 20),
            host_budget_bytes: Some(1 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}
fn vision(projections: &[(String, Vec<i32>)]) -> Vec<(String, Vec<i32>)> {
    projections
        .iter()
        .filter(|(name, _)| name.starts_with("model.visual."))
        .cloned()
        .collect()
}
fn vocabulary(projections: &[(String, Vec<i32>)]) -> Vec<Vec<i32>> {
    projections
        .iter()
        .filter(|(name, _)| name == "lm_head.weight")
        .map(|(_, shape)| shape.clone())
        .collect()
}
pub(super) fn assert_trace(
    report: &Report,
    schedule: Schedule,
    owns_output: bool,
    owns_ingress: bool,
    hidden: i32,
) {
    let end = schedule.cancel_after.unwrap_or(report.input_positions);
    let expected = (0..end)
        .step_by(schedule.chunk as usize)
        .map(|start| {
            let next = (start + schedule.chunk).min(report.input_positions);
            PrefillChunk {
                input: start..next,
                position: start,
                output: schedule.output.for_chunk(next == report.input_positions),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(report.trace.announced, expected);
    assert_eq!(report.trace.delivered, expected);
    assert!(!report.trace.observations.is_empty());
    assert_eq!(
        report.locally_requested,
        schedule.cancel_after.is_some() && schedule.locally_cancel
    );
    assert_eq!(
        report.outcome,
        if schedule.cancel_after.is_some() {
            PrefillOutcome::Cancelled
        } else {
            PrefillOutcome::Complete
        }
    );
    assert_eq!(report.output.is_none(), schedule.cancel_after.is_some());
    let first_vision = vision(&report.trace.projections[0]);
    assert!(
        !first_vision.is_empty(),
        "actual selected encoder projections must execute"
    );
    let mut expected_heads = Vec::new();
    for (span, projections) in expected.iter().zip(&report.trace.projections) {
        assert_eq!(
            vision(projections),
            first_vision,
            "later spans must not repeat encoder work"
        );
        if owns_output && span.output != OutputDemand::StateOnly {
            let rows = if span.output == OutputDemand::Sequence {
                (span.input.end - span.input.start) as i32
            } else {
                1
            };
            expected_heads.push(vec![1, rows, hidden]);
        }
        assert_eq!(
            vocabulary(projections),
            expected_heads,
            "project exactly each demanded hidden interval before vocabulary readout"
        );
    }
    if owns_ingress {
        // A text-only first span still settles all complete future media roots.
        let future = report.trace.roots[0]
            .iter()
            .filter(|(shape, _)| shape == &[1, (report.input_positions - 3) as i32, hidden])
            .collect::<Vec<_>>();
        assert!(
            future.len() >= 3,
            "compact projected vision and both DeepStack roots"
        );
        for (_, values) in future {
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| v.abs() > 1e-9));
        }
    }
    for layer in report.state.as_ref() {
        if layer.attention.is_some() || !layer.fixed.is_empty() {
            assert_eq!(layer.position(), end as i32);
        }
    }
}

fn run_partition_case(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    stepped: bool,
    cancel_after: Option<u64>,
) -> Vec<Report> {
    run_partition_case_with_chunk(
        config,
        artifact,
        topology,
        residency,
        stepped,
        cancel_after,
        2,
    )
}

fn run_partition_case_with_chunk(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    stepped: bool,
    cancel_after: Option<u64>,
    chunk: u64,
) -> Vec<Report> {
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let description = numeric_composite_parameter_description(config);
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_residency(residency);
    let world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        let workers = (0..topology.world_size()).map(|rank| {
            let (inspection, description, plan, config) = (&inspection, &description, &plan, config);
            let world = Arc::clone(&world);
            scope.spawn(move || {
                let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description, rank_topology).unwrap();
                let mut context = NumericContext::with_partition(layout, rank, world);
                context.bind_checkpoint_values = true;
                let prepare = || {
                    let source = partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(
                        inspection, plan, rank, std::time::Duration::from_secs(30), None, 16).unwrap();
                    assert!(matches!((plan.residency(), source.selected().text_realization().residency()),
                        (eredu_core::ResidencyPlan::FullyResident, LayerWeightResidency::FullyResident) |
                        (eredu_core::ResidencyPlan::LayerwiseHost { .. }, LayerWeightResidency::LayerwiseHost(_)) |
                        (eredu_core::ResidencyPlan::DenseDiskStream { .. }, LayerWeightResidency::DenseDiskStream(_))));
                    source
                };
                let input = input();
                let mut reference = partitioned_adapter::composite(prepare(), &context).unwrap();
                let expected = if cancel_after == Some(2) {
                    Some(reference.forward(&numeric_text_prepared_input(&[1, 2]), true).unwrap())
                } else if cancel_after == Some(4) {
                    // Independently execute the same original media coordinate
                    // system with one four-position prefix instead of two spans.
                    let report = reference.media_prefill.as_mut().expect("typed selected media dispatch")(&input,
                        Schedule {
                            output: OutputDemand::LastPosition, chunk: 4, stepped: false, cancel_after: Some(4), locally_cancel: rank == 0, follow_decode: false }).unwrap();
                    assert_eq!(report.outcome, PrefillOutcome::Cancelled);
                    None
                } else { Some(reference.forward(&input, true).unwrap()) };
                if let Some(expected) = &expected { nonzero(expected); }
                let reference_state = reference.snapshot().unwrap();
                let original_vision = vision(&context.projections.lock().unwrap());
                let mut actual = partitioned_adapter::composite(prepare(), &context).unwrap();
                context.projections.lock().unwrap().clear();
                context.media_completions.lock().unwrap().clear();
                let schedule = Schedule {
                            output: OutputDemand::LastPosition, chunk, stepped, cancel_after, locally_cancel: rank == 0, follow_decode: true };
                let report = actual.media_prefill.as_mut().expect("typed selected media dispatch")(&input, schedule).unwrap();
                let hidden = config["text_config"]["hidden_size"].as_i64().unwrap() as i32;
                assert_trace(&report, schedule, rank_topology.owns_output_head(), rank_topology.owns_embedding(), hidden);
                same_state(&report.state, &reference_state);
                if cancel_after != Some(2) {
                    assert_eq!(vision(&report.trace.projections[0]), original_vision, "one actual encoder invocation per selected owner");
                }
                if let Some(output) = &report.output {
                    nonzero(output);
                    assert_tensor_close(output, expected.as_ref().unwrap(), "selected media full versus shared spans");
                }
                for (token, actual) in [2, 6, 1].into_iter().zip(&report.cached) {
                    let expected = if token == 2 && cancel_after.is_some() {
                        (reference.restart_after_cancel)(token, cancel_after.unwrap()).unwrap()
                    } else { reference.forward(&numeric_text_prepared_input(&[token]), false).unwrap() };
                    nonzero(actual);
                    assert_tensor_close(actual, &expected, "ordinary selected media prefix cached decode");
                }
                same_state(&report.final_state, &reference.snapshot().unwrap());
                report
            })
        }).collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    })
}

fn matrix(hybrid: bool, cancel_after: Option<u64>) {
    let config = configuration(hybrid);
    let artifact = fixture(&config);
    for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
        for residency in residencies() {
            let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
            let run = run_partition_case(
                &config,
                &artifact,
                topology,
                residency.clone(),
                false,
                cancel_after,
            );
            let step =
                run_partition_case(&config, &artifact, topology, residency, true, cancel_after);
            for (run, step) in run.iter().zip(&step) {
                assert_eq!(run.trace, step.trace);
                same_state(&run.state, &step.state);
                same_state(&run.final_state, &step.final_state);
                match (&run.output, &step.output) {
                    (Some(a), Some(b)) => assert_tensor_close(a, b, "media run/step"),
                    (None, None) => {}
                    _ => panic!("media output presence"),
                }
            }
        }
    }
}
#[test]
fn qwen_vl_selected_media_encoder_once_spans_match_full_tp_pp_residency_and_restore() {
    matrix(false, None);
}
#[test]
fn conditional_qwen_selected_media_encoder_once_spans_match_full_tp_pp_residency_and_restore() {
    matrix(true, None);
}
#[test]
fn qwen_vl_selected_media_cancellation_preserves_prefix_before_and_inside_media() {
    for end in [2, 4] {
        matrix(false, Some(end));
    }
}
#[test]
fn conditional_qwen_selected_media_cancellation_preserves_prefix_before_and_inside_media() {
    for end in [2, 4] {
        matrix(true, Some(end));
    }
}

#[test]
fn conditional_decoder_deepstack_boundary_matches_single_rank_for_mixed_span_widths() {
    let config = configuration(true);
    let artifact = fixture(&config);
    // Independent full-prompt execution has no decoder-to-decoder transfer.
    // Both DeepStack layers contribute, and PP=2 cuts after the first layer.
    let full = ordinary::independent_full_report(&config, &artifact);
    assert_eq!(full.input_positions, 7);
    assert_eq!(full.state.as_ref().len(), 2);
    for residency in residencies() {
        for chunk in [2, 3, 4] {
            let ranks = run_partition_case_with_chunk(
                &config,
                &artifact,
                ParallelTopology::new(1, 2, 1, 1).unwrap(),
                residency.clone(),
                true,
                None,
                chunk,
            );
            let output = ranks.last().unwrap();
            assert_tensor_close(
                output.output.as_ref().unwrap(),
                full.output.as_ref().unwrap(),
                "conditional normalized decoder wire versus independent single rank",
            );
            let state = ranks
                .iter()
                .flat_map(|rank| rank.state.as_ref().iter().cloned())
                .collect::<Vec<_>>();
            same_layers(&state, full.state.as_ref());
            assert_eq!(output.cached.len(), 3);
            assert_eq!(full.cached.len(), 3);
            for (actual, expected) in output.cached.iter().zip(&full.cached) {
                assert_tensor_close(
                    actual,
                    expected,
                    "conditional decoder wire cached continuation",
                );
            }
            let final_state = ranks
                .iter()
                .flat_map(|rank| rank.final_state.as_ref().iter().cloned())
                .collect::<Vec<_>>();
            same_layers(&final_state, full.final_state.as_ref());
        }
    }
}
