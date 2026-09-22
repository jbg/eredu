//! Muse uses the same selected source, scheduler, completion and state checks.
use super::*;

fn config(routed: bool) -> serde_json::Value {
    let mut config = if routed {
        routed_muse_partition_fixture()
    } else {
        dense_muse_partition_fixture()
    };
    config["vision_config"]["num_hidden_layers"] = 2.into();
    config["vision_config"]["layer_types"] =
        serde_json::json!(["full_attention", "window_attention"]);
    config
}
fn input(mixed: bool) -> Input {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload};
    let mut parts = vec![PreparedInputPart::new(
        InputModality::Text,
        PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[1, 2])),
        [],
    )
    .unwrap()];
    for modality in if mixed {
        vec![InputModality::Image, InputModality::Video]
    } else {
        vec![InputModality::Image]
    } {
        let edge = if mixed { 2 } else { 4 };
        let patches = edge * edge;
        parts.push(
            PreparedInputPart::new_with_extents(
                modality,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [patches, 12],
                    (0..patches * 12)
                        .map(|i| ((i % 37) as f32 - 18.0) / 80.0)
                        .collect(),
                )),
                [(
                    InputMetadataKey::PatchGrid,
                    NumericTensor::new([1, 3], vec![1., edge as f32, edge as f32]),
                )],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: edge as usize,
                    width: edge as usize,
                }],
            )
            .unwrap(),
        );
    }
    parts.push(
        PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[4])),
            [],
        )
        .unwrap(),
    );
    Input::new(parts, |tensor| {
        eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
    })
    .unwrap()
}
fn vision(projections: &[(String, Vec<i32>)]) -> Vec<(String, Vec<i32>)> {
    projections
        .iter()
        .filter(|(name, _)| name.starts_with("model.vision_"))
        .cloned()
        .collect()
}
fn assert_trace(report: &Report, schedule: Schedule, owns_output: bool, owns_ingress: bool) {
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
        schedule.locally_cancel && schedule.cancel_after.is_some()
    );
    assert_eq!(
        report.outcome,
        if schedule.cancel_after.is_some() {
            PrefillOutcome::Cancelled
        } else {
            PrefillOutcome::Complete
        }
    );
    let first = vision(&report.trace.projections[0]);
    assert!(
        !first.is_empty(),
        "actual selected vision equations must execute"
    );
    let mut heads = Vec::new();
    for (span, projections) in expected.iter().zip(&report.trace.projections) {
        assert_eq!(
            vision(projections),
            first,
            "later spans repeat no encoder/projector work"
        );
        if owns_output && span.output != OutputDemand::StateOnly {
            let rows = if span.output == OutputDemand::Sequence {
                (span.input.end - span.input.start) as i32
            } else {
                1
            };
            heads.push(vec![1, rows, 8]);
        }
        let actual = projections
            .iter()
            .filter(|(name, _)| name == "lm_head.weight")
            .map(|(_, shape)| shape.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            actual, heads,
            "select demanded rows before vocabulary projection"
        );
    }
    if owns_ingress {
        let rows = (report.input_positions - 3) as i32;
        let roots = report.trace.roots[0]
            .iter()
            .filter(|(shape, _)| shape == &[rows, 8])
            .collect::<Vec<_>>();
        assert!(
            !roots.is_empty(),
            "first text-only span completes all flattened normalized future media rows"
        );
        assert!(roots
            .iter()
            .all(|(_, values)| values.iter().all(|v| v.is_finite())
                && values.iter().any(|v| v.abs() > 1e-8)));
    }
    for layer in report.state.as_ref() {
        if layer.attention.is_some() {
            assert_eq!(layer.position(), end as i32);
        }
    }
}

#[test]
fn muse_selected_retained_media_matches_local_full_rows_and_mixed_image_video_in_all_residencies() {
    let config = config(false);
    let artifact = selected::fixture(&config);
    for mixed in [false, true] {
        let count = if mixed { 5 } else { 7 };
        for residency in selected::residencies() {
            for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
                let reference = ordinary::run_with_input(
                    &config,
                    &artifact,
                    residency.clone(),
                    None,
                    false,
                    output,
                    input(mixed),
                );
                for width in [1, 2, 3, count] {
                    for stepped in [false, true] {
                        let schedule = Schedule {
                            output,
                            chunk: width,
                            stepped,
                            cancel_after: None,
                            locally_cancel: false,
                            follow_decode: true,
                        };
                        let actual = ordinary::run_with_input(
                            &config,
                            &artifact,
                            residency.clone(),
                            Some(schedule),
                            false,
                            output,
                            input(mixed),
                        );
                        assert_eq!(actual.outputs.len(), 4);
                        for (actual, expected) in actual.outputs.iter().zip(&reference.outputs) {
                            nonzero(actual);
                            assert_tensor_close(
                                actual,
                                expected,
                                "Muse normalized media and cached decode",
                            );
                        }
                        same_state(&actual.state, &reference.state);
                        let report = actual.media.as_ref().unwrap();
                        assert_trace(report, schedule, true, true);
                        assert_eq!(
                            report.output.as_ref().unwrap().shape,
                            [
                                1,
                                if output == OutputDemand::Sequence {
                                    count as i32
                                } else {
                                    1
                                },
                                7
                            ]
                        );
                        if mixed && width == 2 {
                            assert_eq!(
                                report
                                    .trace
                                    .delivered
                                    .iter()
                                    .map(|c| c.input.end - c.input.start)
                                    .collect::<Vec<_>>(),
                                [2, 2, 1]
                            );
                        }
                    }
                }
            }
        }
    }
}

fn partition_case(
    config: &serde_json::Value,
    artifact: &tempfile::TempDir,
    topology: ParallelTopology,
    residency: eredu_core::ResidencyPlan,
    stepped: bool,
    cancel_after: Option<u64>,
) -> Vec<Report> {
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let description = numeric_composite_parameter_description(config);
    let plan = prepared_adapter::plan(None)
        .with_topology(topology)
        .with_residency(residency);
    let world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        let workers = (0..topology.world_size()).map(|rank| {
            let (inspection, description, plan) = (&inspection, &description, &plan);
            let world = Arc::clone(&world);
            scope.spawn(move || {
                let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(description, rank_topology).unwrap();
                let mut context = NumericContext::with_partition(layout, rank, world);
                context.bind_checkpoint_values = true;
                let prepare = || {
                    let source = partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection, plan, rank,
                        std::time::Duration::from_secs(30), None, 16).unwrap();
                    assert!(matches!((plan.residency(), source.selected().text_realization().residency()),
                        (eredu_core::ResidencyPlan::FullyResident, LayerWeightResidency::FullyResident) |
                        (eredu_core::ResidencyPlan::LayerwiseHost { .. }, LayerWeightResidency::LayerwiseHost(_)) |
                        (eredu_core::ResidencyPlan::DenseDiskStream { .. }, LayerWeightResidency::DenseDiskStream(_))));
                    source
                };
                let input = input(false);
                let mut reference = partitioned_adapter::composite(prepare(), &context).unwrap();
                let expected = match cancel_after {
                    Some(2) => Some(reference.forward(&numeric_text_prepared_input(&[1,2]), true).unwrap()),
                    Some(4) => {
                        let prefix = reference.media_prefill.as_mut().unwrap()(&input, Schedule { output: OutputDemand::LastPosition,
                            chunk: 4, stepped: false, cancel_after: Some(4), locally_cancel: rank==0, follow_decode: false }).unwrap();
                        assert_eq!(prefix.outcome, PrefillOutcome::Cancelled); None
                    }
                    None => Some(reference.forward(&input, true).unwrap()),
                    _ => unreachable!(),
                };
                let reference_state = reference.snapshot().unwrap();
                let original_vision = vision(&context.projections.lock().unwrap());
                let mut actual = partitioned_adapter::composite(prepare(), &context).unwrap();
                context.projections.lock().unwrap().clear();context.media_completions.lock().unwrap().clear();
                let schedule = Schedule { output: OutputDemand::LastPosition, chunk: 2, stepped, cancel_after,
                    locally_cancel: rank==0, follow_decode: true };
                let report = actual.media_prefill.as_mut().unwrap()(&input, schedule).unwrap();
                assert_trace(&report, schedule, rank_topology.owns_output_head(), rank_topology.owns_embedding());
                same_state(&report.state, &reference_state);
                let patch = "model.vision_tower.patch_embedder.patch_embedding.weight";
                let actual_vision = vision(&report.trace.projections[0]);
                assert_eq!(actual_vision.iter().filter(|(name, _)| name == patch).count(),
                    usize::from(rank_topology.owns_embedding()),
                    "the actual ingress owner embeds patches once; receivers reuse the transported embedded boundary");
                if cancel_after != Some(2) {
                    let mut expected_vision = original_vision;
                    if !rank_topology.owns_embedding() {
                        // Ordinary full input reconstructs this work on each stage.
                        // The retained path receives already embedded patches.
                        expected_vision.retain(|(name, _)| name != patch);
                    }
                    assert_eq!(actual_vision, expected_vision);
                }
                if let Some(output) = &report.output {
                    nonzero(output);assert_tensor_close(output, expected.as_ref().unwrap(), "Muse selected shared prefill");
                }
                assert_eq!(report.cached.len(), 3);
                for (token, output) in [2,6,1].into_iter().zip(&report.cached) {
                    let expected = if token == 2 && cancel_after.is_some() {
                        (reference.restart_after_cancel)(token, cancel_after.unwrap()).unwrap()
                    } else { reference.forward(&numeric_text_prepared_input(&[token]), false).unwrap() };
                    nonzero(output);assert_tensor_close(output, &expected, "Muse ordinary committed-prefix cached decode");
                }
                same_state(&report.final_state, &reference.snapshot().unwrap());
                report
            })
        }).collect::<Vec<_>>();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    })
}

#[test]
fn muse_selected_retained_media_and_cancelled_prefix_match_tp_pp_residency_and_controlled_steps() {
    for routed in [false, true] {
        let config = config(routed);
        let artifact = selected::fixture(&config);
        for (tp, pp) in [(2, 1), (1, 2), (2, 2)] {
            for residency in selected::residencies() {
                for cancel_after in [None, Some(2), Some(4)] {
                    let topology = ParallelTopology::new(tp, pp, 1, 1).unwrap();
                    let run = partition_case(
                        &config,
                        &artifact,
                        topology,
                        residency.clone(),
                        false,
                        cancel_after,
                    );
                    let step = partition_case(
                        &config,
                        &artifact,
                        topology,
                        residency.clone(),
                        true,
                        cancel_after,
                    );
                    for (run, step) in run.iter().zip(&step) {
                        assert_eq!(run.trace, step.trace);
                        same_state(&run.state, &step.state);
                        same_state(&run.final_state, &step.final_state);
                        match (&run.output, &step.output) {
                            (Some(a), Some(b)) => {
                                assert_tensor_close(a, b, "Muse selected run/step")
                            }
                            (None, None) => (),
                            _ => panic!("output presence"),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn external_whole_prepared_media_preserves_nonzero_readout_and_cached_state() {
    let config = config(false);
    let artifact = selected::fixture(&config);
    for mixed in [false, true] {
        for residency in selected::residencies() {
            for demand in [OutputDemand::LastPosition, OutputDemand::Sequence] {
                let expected = ordinary::run_with_input(
                    &config,
                    &artifact,
                    residency.clone(),
                    None,
                    false,
                    demand,
                    input(mixed),
                );
                let actual = ordinary::run_external_with_input(
                    &config,
                    &artifact,
                    residency.clone(),
                    demand,
                    input(mixed),
                );
                assert_eq!(actual.outputs.len(), 4);
                for (actual, expected) in actual.outputs.iter().zip(&expected.outputs) {
                    assert!(actual.data.iter().any(|x| x.abs() > 1e-8));
                    assert_tensor_close(actual, expected, "unchanged external whole media source");
                }
                same_state(&actual.state, &expected.state);
            }
        }
    }
}
