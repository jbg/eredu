//! Ordinary artifact selection, retained rank binding and shared sessions execute together.
use super::*;
use eredu_architectures::qwen4_exp::prepared::PreparedTarget;
use eredu_runtime::*;

fn inputs() -> Vec<Vec<u64>> {
    vec![
        vec![3, 4, 0],
        vec![5, 8, 9, 11],
        vec![12],
        vec![14],
        vec![2],
    ]
}

fn baseline(target: &PreparedTarget) -> Vec<NumericTensor> {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let local = target.tensor_partition(0, 1).unwrap();
    let mut runtime = prepared_tensor_parallel::runtime(target, &local, false, &context);
    let mut state = state(target.spec());
    let rows = local
        .row_lookups(
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 1 << 16,
                output_bytes: 32768,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: rows
            .bind(
                rows.entries()
                    .iter()
                    .map(|(id, row)| (id.clone(), super::super::row_bank::SourceRows::new(row)))
                    .collect(),
            )
            .unwrap(),
    };
    inputs()
        .into_iter()
        .enumerate()
        .map(|(step, ids)| {
            let output = runtime
                .forward_with_provider_and_observer(
                    TargetInput {
                        ids: Some(OriginalTokenIds::Host(&ids)),
                        batch: 1,
                        tokens: ids.len() as i32,
                        embeddings: None,
                        visible: None,
                        rotary: None,
                        position_delta: None,
                    },
                    &mut state,
                    if step < 2 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    },
                    &mut provider,
                    &context,
                    &mut NoopObserver,
                )
                .unwrap();
            numeric_text_output(output).unwrap()
        })
        .collect()
}

fn run(
    path: &std::path::Path,
    target: &PreparedTarget,
    tp: usize,
    pp: usize,
    ep: usize,
    expected: &[NumericTensor],
) {
    let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let outputs = std::thread::scope(|scope| {
        (0..topology.world_size())
            .map(|rank| {
                let world = Arc::clone(&world);
                let inspection = inspection.clone();
                // Full prepared visitors have large debug-build stack frames.
                std::thread::Builder::new()
                    .name(format!("qwen4-ordinary-rank-{rank}"))
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let request = partition_selection::request(tp, pp, ep, rank);
                        let selected = eredu_architectures::select_preparation(
                            &inspection,
                            &request,
                            &prepared_adapter::NumericPreparationProvider { addressable: true },
                        )
                        .unwrap();
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
                        let tensor_rank = rank_topology.tensor_parallel_rank();
                        let tensor = target.tensor_partition(tensor_rank, tp).unwrap();
                        let context = NumericContext::default();
                        let global = TargetModel::<NumericBackend>::new(
                            tensor.source_bound_spec().clone(),
                            &context,
                        )
                        .unwrap();
                        let mut local = TargetModel::<NumericBackend>::new_tensor_parallel(
                            tensor.source_bound_spec().clone(),
                            tensor.partition().clone(),
                            &context,
                        )
                        .unwrap();
                        let tensor_parameters = local.parameter_description(&context).unwrap();
                        let experts = tensor.partition().local_spec().expert_realization(rank_topology).unwrap();
                        local.set_expert_realization(&experts).unwrap();
                        let layout = tensor
                            .partition()
                            .local_expert_layout(
                                &global.parameter_description(&context).unwrap(),
                                &tensor_parameters,
                                &local.parameter_description(&context).unwrap(),
                                rank_topology.expert_parallel_rank(),
                                rank_topology.expert_parallel_size(),
                            )
                            .unwrap();
                        let mut context = NumericContext::with_partition(layout, rank, world);
                        context.bind_checkpoint_values = true;
                        let mut executable = partitioned_adapter::routed(
                            sources,
                            &context,
                            Arc::new(AtomicUsize::new(0)),
                            None,
                        )
                        .unwrap_or_else(|e| {
                            panic!("ordinary TP{tp} PP{pp} EP{ep} rank {rank} construction: {e}")
                        });
                        let actual: Vec<_> = inputs()
                            .into_iter()
                            .enumerate()
                            .map(|(step, ids)| {
                                let tokens = NumericTensor::from_i32_slice(
                                    &ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
                                    &[1, ids.len() as i32],
                                    &context,
                                )
                                .unwrap();
                                let output = executable.forward(&tokens, step < 2).unwrap_or_else(|e| {
                                    panic!("ordinary TP{tp} PP{pp} EP{ep} rank {rank} step {step}: {e}")
                                });
                                output
                            })
                            .collect();
                        assert!(executable
                            .positions()
                            .unwrap()
                            .iter()
                            .all(|position| *position == 10));
                        executable.reset().unwrap();
                        assert!(executable.positions().unwrap().iter().all(|position| *position == 0));
                        for (step, ids) in inputs().into_iter().enumerate() {
                            let tokens = NumericTensor::from_i32_slice(
                                &ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
                                &[1, ids.len() as i32], &context,
                            ).unwrap();
                            let replay = executable.forward(&tokens, step < 2).unwrap();
                            assert_eq!(replay.data, actual[step].data, "reset EP replay rank {rank} step {step}");
                        }
                        actual
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for actual in outputs {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_tensor_close(actual, expected, "ordinary prepared TP/PP/EP cached logits");
        }
    }
}

#[test]
fn qwen4_ordinary_prepared_safetensors_and_gguf_tp_pp_sessions_match_nonzero_reference() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::Fixture::new()
        .prepare(directory.path())
        .unwrap();
    let expected = baseline(&fixture.safetensors);
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
            (2, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ] {
            run(path, target, tp, pp, ep, &expected);
        }
    }
}

#[test]
fn qwen4_other_rank_parameter_and_component_projection_matches_authored_tp_pp_ep_geometry() {
    use eredu_core::capture::{CaptureError, CaptureReservation, CaptureSkipReason, CaptureUsage};
    struct Budget;
    impl CaptureReservation for Budget {
        fn reserve(&mut self, _: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
            Ok(None)
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::Fixture::new()
        .prepare(directory.path())
        .unwrap();
    let context = NumericContext::default();
    for pipeline in [2, 3] {
        let topology = ParallelTopology::new(2, pipeline, 2, 1).unwrap();
        for (path, target) in [
            (&fixture.safetensors_path, &fixture.safetensors),
            (&fixture.gguf_path, &fixture.gguf),
        ] {
            let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
            let ranks: Vec<_> = (0..topology.world_size())
                .map(|rank| {
                    let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                    let selected = eredu_architectures::select_preparation(
                        &inspection,
                        &partition_selection::request(2, pipeline, 2, rank),
                        &prepared_adapter::NumericPreparationProvider { addressable: true },
                    )
                    .unwrap();
                    let admitted = eredu_core::ModelPreparationPlan::from_retained_admission(
                        inspection.clone(),
                        selected.admission(),
                    )
                    .unwrap();
                    let sources = eredu_architectures::prepared_sources::prepare_model_sources(
                        admitted, selected,
                    )
                    .unwrap();
                    let tensor = target
                        .tensor_partition(rank_topology.tensor_parallel_rank(), 2)
                        .unwrap();
                    let global = TargetModel::<NumericBackend>::new(
                        tensor.source_bound_spec().clone(),
                        &context,
                    )
                    .unwrap();
                    let mut local = TargetModel::<NumericBackend>::new_tensor_parallel(
                        tensor.source_bound_spec().clone(),
                        tensor.partition().clone(),
                        &context,
                    )
                    .unwrap();
                    let tensor_parameters = local.parameter_description(&context).unwrap();
                    let experts = tensor
                        .partition()
                        .local_spec()
                        .expert_realization(rank_topology)
                        .unwrap();
                    local.set_expert_realization(&experts).unwrap();
                    let global_parameters = global.parameter_description(&context).unwrap();
                    let layout = tensor
                        .partition()
                        .local_expert_layout(
                            &global_parameters,
                            &tensor_parameters,
                            &local.parameter_description(&context).unwrap(),
                            rank_topology.expert_parallel_rank(),
                            rank_topology.expert_parallel_size(),
                        )
                        .unwrap();
                    let parameters = global_parameters
                        .with_partition_layout(rank_topology, layout)
                        .unwrap();
                    (sources, parameters)
                })
                .collect();
            let (origin, parameters) = &ranks[0];
            let descriptor = origin.architecture().architecture_descriptor();
            let decoder = parameters
                .groups()
                .iter()
                .find(|group| group.logical_name() == "model.layers.1")
                .unwrap();
            assert!(
                matches!(
                    decoder.owner(),
                    eredu_runtime::ParameterGroupOwner::ExecutionUnit { global_unit: 2, .. }
                ),
                "the second decoder ordinal includes the earlier lexical injection"
            );
            let lexical = parameters
                .groups()
                .iter()
                .find(|group| group.logical_name() == "model.layers.0.ple")
                .unwrap();
            assert!(matches!(
                lexical.owner(),
                eredu_runtime::ParameterGroupOwner::ExecutionUnit { global_unit: 0, .. }
            ));
            let first_decoder = parameters
                .groups()
                .iter()
                .find(|group| group.logical_name() == "model.layers.0")
                .unwrap();
            assert!(matches!(
                first_decoder.owner(),
                eredu_runtime::ParameterGroupOwner::ExecutionUnit { global_unit: 1, .. }
            ));
            let mut invalid = descriptor.clone();
            invalid
                .observations
                .points
                .iter_mut()
                .find(|point| point.path == "model.layers.1.output")
                .unwrap()
                .axes
                .as_mut()
                .unwrap()[3]
                .dimension = eredu_core::SymbolicDimension::Known(999);
            assert!(origin
                .selected()
                .execution()
                .component_partition_layout_for_rank(&invalid, parameters, 0)
                .is_err());
            let mut unknown = descriptor.clone();
            unknown
                .observations
                .points
                .iter_mut()
                .find(|point| point.path == "model.layers.1.output")
                .unwrap()
                .path = "model.layers.1.undeclared_seam".into();
            assert!(origin
                .selected()
                .execution()
                .component_partition_layout_for_rank(&unknown, parameters, 0)
                .unwrap()
                .unwrap()
                .observation("model.layers.1.undeclared_seam")
                .is_none());
            for (rank, (local, local_parameters)) in ranks.iter().enumerate() {
                // The local route validates the projection against independently constructed
                // local modules. Other-rank inspection must produce exactly that declaration.
                let actual = origin
                    .selected()
                    .execution()
                    .component_partition_layout_for_rank(&descriptor, parameters, rank)
                    .unwrap()
                    .unwrap();
                let expected = local
                    .selected()
                    .execution()
                    .component_partition_layout(&descriptor, local_parameters)
                    .unwrap()
                    .unwrap();
                assert_eq!(actual, expected, "component projection rank {rank}");
                let topology = ParallelRankTopology::new(topology, rank).unwrap();
                for layer in 0..2 {
                    for seam in [
                        eredu_core::UnitObservation::Input,
                        eredu_core::UnitObservation::Output,
                    ] {
                        let path = seam.path(&format!("model.layers.{layer}"));
                        let point = descriptor.observations.get(&path).unwrap();
                        assert_eq!(point.axes.as_ref().unwrap().len(), 4);
                        let placement = actual.observation(&path).unwrap();
                        // PP3 cuts immediately after lexical unit 0 and before its
                        // decoder unit 1. PP2 keeps those together on the first stage.
                        let owner = if layer == 0 {
                            usize::from(pipeline == 3)
                        } else {
                            pipeline - 1
                        };
                        assert_eq!(
                            placement.coordinates().is_some(),
                            topology.pipeline_parallel_rank() == owner
                        );
                        assert_eq!(
                            placement.exports(),
                            topology.pipeline_parallel_rank() == owner
                        );
                        assert_eq!(
                            placement.site(),
                            eredu_runtime::inspection::ObservationHookSite::Unit
                        );
                    }
                }
                for member in parameters.groups().iter().flat_map(|group| group.members()) {
                    let actual = origin
                        .selected()
                        .execution()
                        .parameter_partition_layout_for_rank(
                            parameters,
                            member.target(),
                            rank,
                            &mut Budget,
                        )
                        .unwrap()
                        .unwrap();
                    let expected = local
                        .selected()
                        .execution()
                        .parameter_partition_layout_for_rank(
                            local_parameters,
                            member.target(),
                            rank,
                            &mut Budget,
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        actual,
                        expected,
                        "parameter {} rank {rank}",
                        member.target()
                    );
                }
            }
        }
    }
}
