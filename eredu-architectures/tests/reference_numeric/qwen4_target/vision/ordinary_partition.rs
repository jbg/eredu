//! Ordinary source selection, media projection and distributed cached execution together.
use super::*;
use eredu_architectures::processor_plan::{GgufSpecialTokenIds, QwenMediaTokenIds};

fn long_prompt(inspector: &Inspector) -> PreparedModelInput<NumericTensor> {
    let mut parts = vec![text(&[3; 29])];
    parts.extend(prompt(inspector).into_parts());
    // The selected 32-token chunk ends inside the four-position image span.
    // Vision is projected once; later chunks retain its rows, original IDs and
    // media rotary positions rather than re-encoding the request.
    parts.push(text(&[3; 3]));
    input(parts, inspector)
}

fn exact_layout(
    target: &PreparedTarget,
    ingress: &MediaIngress,
    rank: ParallelRankTopology,
) -> LocalModelLayout {
    let context = NumericContext::default();
    let tensor = target
        .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())
        .unwrap();
    let global =
        TargetModel::<NumericBackend>::new(tensor.source_bound_spec().clone(), &context).unwrap();
    let mut local = TargetModel::<NumericBackend>::new_tensor_parallel(
        tensor.source_bound_spec().clone(),
        tensor.partition().clone(),
        &context,
    )
    .unwrap();
    let tensor_parameters = local.parameter_description(&context).unwrap();
    local
        .set_expert_realization(
            &tensor
                .partition()
                .local_spec()
                .expert_realization(rank)
                .unwrap(),
        )
        .unwrap();
    let target_layout = tensor
        .partition()
        .local_expert_layout(
            &global.parameter_description(&context).unwrap(),
            &tensor_parameters,
            &local.parameter_description(&context).unwrap(),
            rank.expert_parallel_rank(),
            rank.expert_parallel_size(),
        )
        .unwrap();
    let conditional = ConditionalModel::<NumericBackend>::new(
        target.bound_spec().unwrap(),
        ingress.clone(),
        ingress.vision().config().clone(),
        &context,
    )
    .unwrap();
    let mut layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &conditional.parameter_description(&context).unwrap(),
        rank,
    )
    .unwrap();
    for (name, tensor) in target_layout.tensors() {
        layout.insert(name.to_owned(), tensor.clone());
    }
    layout
}

#[allow(clippy::too_many_arguments)]
fn run(
    path: &std::path::Path,
    target: &PreparedTarget,
    ingress: &MediaIngress,
    tp: usize,
    pp: usize,
    ep: usize,
    residency: LayerWeightResidency,
    expected: &[NumericTensor],
) {
    eprintln!(
        "ordinary media {} TP{tp} PP{pp} EP{ep} {residency:?}",
        path.display()
    );
    let mut inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    if path
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        inspection
            .architecture_plan_mut()
            .bind_gguf_special_token_ids(GgufSpecialTokenIds::Qwen(QwenMediaTokenIds {
                image_token_id: media().image,
                video_token_id: media().video,
                vision_start_token_id: media().start,
                vision_end_token_id: media().end,
            }))
            .unwrap();
    }
    let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let outputs = std::thread::scope(|scope| {
        (0..topology.world_size())
            .map(|rank| {
                let inspection = inspection.clone();
                let world = Arc::clone(&world);
                let residency = residency.clone();
                std::thread::Builder::new()
                    .name(format!("qwen4-media-rank-{rank}"))
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let request =
                            crate::qwen4_target::partition_selection::request(tp, pp, ep, rank)
                                .with_media_execution(MediaLoadRequest::Required(
                                    MediaExecutionPolicy::new(processor_request()).unwrap(),
                                ))
                                .with_weight_residency(WeightResidency::with_layers(residency));
                        let selected = eredu_architectures::select_preparation(
                            &inspection,
                            &request,
                            &prepared_adapter::NumericPreparationProvider { addressable: true },
                        )
                        .unwrap_or_else(|e| {
                            panic!("media TP{tp} PP{pp} EP{ep} rank {rank} selection: {e}")
                        });
                        assert!(selected.execution().processor().is_some());
                        let admitted = eredu_core::ModelPreparationPlan::from_retained_admission(
                            inspection,
                            selected.admission(),
                        )
                        .unwrap();
                        let sources = eredu_architectures::prepared_sources::prepare_model_sources(
                            admitted, selected,
                        )
                        .unwrap();
                        let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                        let mut context = NumericContext::with_partition(
                            exact_layout(target, ingress, rank_topology),
                            rank,
                            world,
                        );
                        context.bind_checkpoint_values = true;
                        let mut executable = partitioned_adapter::composite(sources, &context)
                            .unwrap_or_else(|e| {
                                panic!("media TP{tp} PP{pp} EP{ep} rank {rank} construction: {e}")
                            });
                        let inspector = Inspector(0.into());
                        // One rank fails admission while peers hold a valid chunked
                        // request. Every participant rejects before mutation.
                        let malformed = if rank == 0 {
                            input(vec![text(&[99])], &inspector)
                        } else {
                            long_prompt(&inspector)
                        };
                        assert!(executable.forward(&malformed, true).is_err());
                        assert!(executable
                            .positions()
                            .unwrap()
                            .iter()
                            .all(|position| *position == 0));
                        // Valid requests disagree on extent and cursor necessity.
                        let mismatched = if rank == 0 {
                            input(vec![text(&[3])], &inspector)
                        } else {
                            long_prompt(&inspector)
                        };
                        assert!(executable.forward(&mismatched, true).is_err());
                        assert!(executable
                            .positions()
                            .unwrap()
                            .iter()
                            .all(|position| *position == 0));
                        let mut inputs = vec![long_prompt(&inspector)];
                        inputs.extend(
                            (0..16).map(|step| input(vec![text(&[step % 12])], &inspector)),
                        );
                        let actual: Vec<_> = inputs
                            .iter()
                            .enumerate()
                            .map(|(step, input)| {
                                executable.forward(input, step == 0).unwrap_or_else(|e| {
                                    panic!(
                                        "media TP{tp} PP{pp} EP{ep} rank {rank} step {step}: {e}"
                                    )
                                })
                            })
                            .collect();
                        assert!(executable
                            .positions()
                            .unwrap()
                            .iter()
                            .all(|position| *position == 66));
                        executable.reset().unwrap();
                        assert!(executable
                            .positions()
                            .unwrap()
                            .iter()
                            .all(|position| *position == 0));
                        for (step, input) in inputs.iter().enumerate() {
                            let replay = executable.forward(input, step == 0).unwrap();
                            assert_eq!(
                                replay.data, actual[step].data,
                                "media reset replay rank {rank} step {step}"
                            );
                        }
                        actual
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    for actual in outputs {
        assert_eq!(actual.len(), expected.len());
        for (step, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            // Shared sessions publish the final position with its singleton time axis collapsed.
            assert_eq!(actual.data.len(), expected.data.len());
            let actual = NumericTensor::new(expected.shape.clone(), actual.data.clone());
            assert_tensor_close(
                &actual,
                expected,
                &format!("ordinary media TP{tp} PP{pp} EP{ep} step {step}"),
            );
        }
    }
}

#[test]
fn qwen4_ordinary_media_safetensors_gguf_tp_pp_cached_and_residency_match_reference() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_vision_weights(&fixture.safetensors_path, 3);
    let projector = directory.path().join("mmproj.gguf");
    eredu_evaluation::qwen4_exp::write_vision_projector(&projector, 3);
    let vision = gguf_vision(
        &fixture.gguf,
        &eredu_gguf::Checkpoint::open(&projector).unwrap(),
        media(),
    )
    .unwrap();
    let ingress = fixture.gguf.media_ingress(vision).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let source = long_prompt(&inspector);
    let prepared = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert_eq!(prepared.token_ids().dim(1), 50);
    let (expected, _) = reference(&fixture.gguf, &ingress, &prepared, &context);
    assert!(expected
        .iter()
        .flat_map(|value| &value.data)
        .any(|value| value.abs() > 0.01));
    for (path, target) in [
        (&fixture.safetensors_path, &fixture.safetensors),
        (&fixture.gguf_path, &fixture.gguf),
    ] {
        for (tp, pp, ep) in [
            (2, 1, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ] {
            for residency in [
                LayerWeightResidency::FullyResident,
                LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1)
                        .unwrap(),
                )),
                LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
                ),
            ] {
                run(path, target, &ingress, tp, pp, ep, residency, &expected);
            }
        }
    }
}
