//! Native predictor cuts preserve local LSTM cells and the published final readout.
use super::*;
use eredu_architectures::voicechat::{
    rnnt::RnntPredictionState,
    rnnt_predictor::{rnnt_predictor_parameter_description, RnntPredictorArchitecture},
    rnnt_predictor_distribution::{
        PreparedRnntPredictorPipelineBindings, RnntPredictorPipelinePreparationError,
    },
    rnnt_predictor_partition::{RnntPredictorBoundarySchema, RnntPredictorPipelineCuts},
    rnnt_predictor_preparation::PreparedRnntPredictorBindings,
};
use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    native::materialize_voicechat_rnnt_predictor_pipeline,
};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
    DistributedCommitOutcome, ParallelRankTopology, ParallelTopology,
};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, BoundaryDimensionContract,
    BoundaryRoleContract, BoundaryTensorDimension, BoundaryTensorDtype,
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationRouteDescriptor, CommunicationRouteId, CommunicationTensorLimits,
    LayeredPartitionDriver, PartitionBoundaryRoute, PartitionOutputPublication, PartitionOwnership,
    PartitionedExecutionPlan, PipelineActivationDtype, PipelineWireContract,
    RoleExactBoundaryContract,
};

const WORKER: &str = "EREDU_RNNT_PREDICTOR_PIPELINE_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_RNNT_PREDICTOR_PIPELINE_DIRECTORY";

pub(super) fn roots(state: &RnntPredictionState<MlxTensor>, stream: &Stream) -> Retained {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        values.push((
            tensor.shape().to_vec(),
            tensor.element_type().unwrap(),
            tensor.to_f32_vec(stream).unwrap(),
        ))
    });
    values
}
pub(super) fn compare_prediction(
    state: &RnntPredictionState<MlxTensor>,
    expected: &Value,
    fixture: &Value,
    stream: &Stream,
) {
    for (global, layer) in state.global_layers().zip(&state.local_state().layers) {
        for (name, tensor) in [("h", &layer.hidden), ("c", &layer.cell)] {
            let values = expected[name].as_array().unwrap()[global * 2..global * 2 + 2].to_vec();
            compare(tensor, &Value::Array(values), fixture, stream);
        }
    }
}
fn snapshot(state: &RnntPredictionState<MlxTensor>, stream: &Stream) {
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::{SnapshotBudget, SnapshotValue};
    let before = roots(state, stream);
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let estimate = state.estimate_snapshot_value().unwrap();
    let reservation = budget
        .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
        .unwrap();
    let copied = state.copy_snapshot_value(&reservation).unwrap();
    assert_eq!(copied.global_layers(), state.global_layers());
    assert_eq!(copied.dimensions(), state.dimensions());
    assert_eq!(roots(&copied, stream), before);
    assert_eq!(
        copied.output().to_f32_vec(stream).unwrap(),
        state.output().to_f32_vec(stream).unwrap()
    );
    assert_eq!(roots(state, stream), before);
}

#[test]
fn native_rnnt_predictor_pipeline_worker() {
    run_worker(false, false, false);
}

#[test]
fn native_rnnt_greedy_pipeline_worker() {
    run_worker(true, false, false);
}

#[test]
fn native_perception_rnnt_pipeline_worker() {
    run_worker(false, true, false);
}

#[test]
fn native_pcm_pipeline_worker() {
    run_worker(false, false, true);
}

fn run_worker(greedy: bool, coupled: bool, pcm: bool) {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    let size = size.parse::<usize>().unwrap();
    let tensor_ranks = size / 2;
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), size);
    let rank = world.rank();
    let stage = rank / tensor_ranks;
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value = serde_json::from_str(if coupled || pcm {
        eredu_evaluation::voicechat::PERCEPTION_RNNT_REFERENCE_JSON
    } else {
        eredu_evaluation::voicechat::RNNT_REFERENCE_JSON
    })
    .unwrap();
    let dimensions = dimensions(&fixture);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let source = checkpoint(
        &fixture,
        false,
        &directory.path().join("predictor.safetensors"),
    );
    for mode in 0..3 {
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let description = rnnt_predictor_parameter_description(dimensions).unwrap();
        let architecture = RnntPredictorArchitecture::<Backend>::new(dimensions).unwrap();
        let range = stage..stage + 1;
        let partition = ArchitecturePartition::from_description(
            &description,
            [(description.graph().groups()[0].id(), range.clone())],
            PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
            &architecture.state_layout().unwrap(),
            &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..2,
            )]),
            (),
            RnntPredictorBoundarySchema::new(dimensions).unwrap(),
        )
        .unwrap();
        let selected = PreparedRnntPredictorBindings::from_rnnt(
            prepared(dimensions, source.clone()),
            &partition,
        )
        .unwrap();
        let selected = if tensor_ranks > 1 {
            selected
                .select_parallel(
                    ParallelRankTopology::new(
                        ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                        rank % tensor_ranks,
                    )
                    .unwrap(),
                )
                .unwrap()
                .0
        } else {
            selected
        };
        assert_eq!(selected.source().source_keys().len(), 4);
        assert!(selected
            .source()
            .source_keys()
            .iter()
            .all(|key| key.ends_with(&format!("_l{stage}"))));
        assert!(matches!(
            selected.source().source_metadata(&format!(
                "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l{}",
                1 - stage
            )),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let sizes = selected
            .unit_bindings()
            .iter()
            .map(|(_, bindings)| binding_bytes(bindings).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            sizes,
            [192],
            "each stage retains one complete recurrent layer per TP replica"
        );
        let residency = match mode {
            0 => LayerWeightResidency::FullyResident,
            1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(192), Some(1 << 20), 1).unwrap(),
            )),
            _ => LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(192, 0, 0, 0).unwrap(),
            ),
        };
        let (manifest, routes, session, tensor_group) = communication(
            dimensions,
            tensor_ranks,
            rank,
            1,
            dimensions.predictor_width as usize,
        );
        let cuts =
            RnntPredictorPipelineCuts::new(dimensions, routes.iter().map(|route| (route.route, 1)))
                .unwrap();
        let driver =
            LayeredPartitionDriver::new_with_state_ownership(&partition, 0, range.clone(), true)
                .unwrap();
        let plan = PartitionedExecutionPlan::new(
            description.graph().clone(),
            vec![(ArchitectureGroupKind::Recurrent, false)],
            vec![Some(driver)],
            routes,
            Some(PartitionOutputPublication {
                group: session,
                owner_rank: tensor_ranks,
            }),
            Some(session),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        if mode == 0 && !greedy && !coupled && !pcm {
            let narrow = PreparedRnntPredictorBindings::from_rnnt(
                prepared(dimensions, source.clone()),
                &partition,
            )
            .unwrap();
            let narrow = if tensor_ranks > 1 {
                narrow
                    .select_parallel(
                        ParallelRankTopology::new(
                            ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                            rank % tensor_ranks,
                        )
                        .unwrap(),
                    )
                    .unwrap()
                    .0
            } else {
                narrow
            };
            let (narrow_manifest, _, _, _) = communication(dimensions, tensor_ranks, rank, 1, 1);
            assert!(matches!(
                PreparedRnntPredictorPipelineBindings::new(
                    narrow,
                    partition.clone(),
                    cuts.clone(),
                    narrow_manifest,
                    plan.clone(),
                    tensor_group,
                    PipelineActivationDtype::Float32,
                ),
                Err(RnntPredictorPipelinePreparationError::Capability(
                    eredu_runtime::CommunicationCapabilityError::InsufficientLimits {
                        operation: CommunicationOperation::Broadcast,
                    }
                ))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "undersized final readout publication fails during cold admission"
            );
        }
        let selected = PreparedRnntPredictorPipelineBindings::new(
            selected,
            partition,
            cuts,
            manifest,
            plan,
            tensor_group,
            PipelineActivationDtype::Float32,
        )
        .unwrap();
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        if pcm {
            super::pcm::exercise(
                selected,
                residency,
                directory.path(),
                source.clone(),
                &fixture,
                mode,
                &world,
                &weights,
                &stream,
            );
            continue;
        }
        let (mut model, metadata) = materialize_voicechat_rnnt_predictor_pipeline(
            selected, &world, residency, &weights, &stream,
        )
        .unwrap();
        assert_eq!(metadata.static_device_bytes(), 0);
        assert_eq!(metadata.layer_parameter_bytes(), 192);
        assert_eq!(
            model.prediction_state_bytes().unwrap(),
            if stage == 0 { 24 } else { 16 }
        );
        let carry_bytes = model
            .state_layout()
            .layer(0)
            .unwrap()
            .fixed_state()
            .iter()
            .map(|tensor| {
                tensor
                    .resolved_shape(1, 1)
                    .unwrap()
                    .into_iter()
                    .map(|extent| extent as u64)
                    .product::<u64>()
                    * 4
            })
            .sum::<u64>();
        assert_eq!(
            carry_bytes, 16,
            "phase layout contains only this stage's F32 h and c"
        );
        if coupled {
            super::perception::exercise(
                &mut model,
                directory.path(),
                source.clone(),
                &fixture,
                mode,
                &world,
                &weights,
                &stream,
            );
            continue;
        }
        if greedy {
            super::greedy::exercise(
                &mut model,
                directory.path(),
                source.clone(),
                &fixture,
                mode,
                &world,
                &weights,
                &stream,
            );
            continue;
        }
        let embedding = floats(
            &fixture["parameters"]["stt_model.rnnt_decoder.prediction.embed.weight"]["values"],
        );
        let mut state = None;
        let mut epoch = 0;
        for (step, token) in [None, None, Some(0), Some(2), Some(1)]
            .into_iter()
            .enumerate()
        {
            let values = token.map_or(vec![0.; 2], |token| {
                embedding[token * 2..token * 2 + 2].to_vec()
            });
            let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2]));
            let descriptor = [91, mode, step as u64, 0, 0, 0, 0];
            if step == 0 && mode == 2 {
                let reads = model
                    .residency_report()
                    .unwrap()
                    .weight_store()
                    .physical_reads;
                let mut bad = descriptor;
                bad[6] = rank as u64;
                assert!(model
                    .forward(
                        &input,
                        None,
                        step as u64,
                        bad,
                        DistributedCommitEpoch::new(1).unwrap(),
                        &stream
                    )
                    .is_err());
                assert!(
                    model
                        .forward(
                            &input,
                            None,
                            step as u64 + rank as u64,
                            descriptor,
                            DistributedCommitEpoch::new(1).unwrap(),
                            &stream
                        )
                        .is_err(),
                    "the explicit recurrent step participates in initial agreement"
                );
                assert_eq!(
                    model
                        .residency_report()
                        .unwrap()
                        .weight_store()
                        .physical_reads,
                    reads
                );
            }
            let before = state.as_ref().map(|state| roots(state, &stream));
            epoch += 1;
            let commit = DistributedCommitEpoch::new(epoch).unwrap();
            let (candidate, outcome) = model
                .forward(
                    &input,
                    state.as_ref(),
                    step as u64,
                    descriptor,
                    commit,
                    &stream,
                )
                .unwrap();
            assert_eq!(outcome, DistributedCommitOutcome::Committed(commit));
            assert_eq!(candidate.global_layers(), range);
            assert_eq!(candidate.local_state().layers.len(), 1);
            assert_eq!(
                roots(&candidate, &stream).len(),
                if stage == 0 { 3 } else { 2 }
            );
            if step == 0 {
                compare(
                    candidate.output(),
                    &Value::Array(
                        fixture["predictor_steps"][0]["output"].as_array().unwrap()[..2].to_vec(),
                    ),
                    &fixture,
                    &stream,
                );
            } else {
                let reference = &fixture["predictor_steps"][step - 1];
                let output = reference["output"].as_array().unwrap();
                compare(
                    candidate.output(),
                    &Value::Array(output[output.len() - 2..].to_vec()),
                    &fixture,
                    &stream,
                );
                compare_prediction(&candidate, reference, &fixture, &stream);
                epoch += 1;
                let commit = DistributedCommitEpoch::new(epoch).unwrap();
                let (replay, outcome) = model
                    .forward(
                        &input,
                        state.as_ref(),
                        step as u64,
                        descriptor,
                        commit,
                        &stream,
                    )
                    .unwrap();
                assert_eq!(outcome, DistributedCommitOutcome::Committed(commit));
                assert_eq!(roots(&replay, &stream), roots(&candidate, &stream));
                assert_eq!(
                    replay.output().to_f32_vec(&stream).unwrap(),
                    candidate.output().to_f32_vec(&stream).unwrap()
                );
            }
            assert_eq!(state.as_ref().map(|state| roots(state, &stream)), before);
            if step == 1 && mode == 0 {
                snapshot(&candidate, &stream);
            }
            state = Some(candidate);
        }
        if mode > 0 {
            assert!(
                model
                    .residency_report()
                    .unwrap()
                    .offload()
                    .peak_resident_bytes()
                    .get(MemoryTier::Device)
                    <= 192
            );
        }
        if let Some(report) = model.dense_stream_report().unwrap() {
            assert_eq!(report.prefill_forwards() + report.decode_forwards(), 9);
        }
    }
}

fn boundary_contract(config: RnntDimensions, cut: usize) -> RoleExactBoundaryContract {
    let geometry = RnntPredictorBoundarySchema::new(config)
        .unwrap()
        .at_cut(cut)
        .unwrap();
    let resolved = geometry.resolve(1, 1).unwrap();
    let schema = geometry.wire_schema().unwrap();
    let roles = std::iter::once(schema.primary())
        .chain(schema.auxiliary())
        .zip(std::iter::once(resolved.primary()).chain(resolved.auxiliary()))
        .map(|(symbolic, resolved)| {
            let dtype = match symbolic.dtype() {
                BoundaryTensorDtype::Activation | BoundaryTensorDtype::Float32 => TensorDtype::F32,
                BoundaryTensorDtype::Int32 => TensorDtype::I32,
                other => panic!("unexpected RNNT predictor wire dtype {other:?}"),
            };
            let shape = symbolic
                .shape()
                .iter()
                .zip(resolved.shape())
                .map(|(dim, extent)| match dim {
                    BoundaryTensorDimension::Batch | BoundaryTensorDimension::Sequence => {
                        BoundaryDimensionContract::Variable {
                            maximum: *extent as usize,
                        }
                    }
                    BoundaryTensorDimension::Fixed(value) => {
                        BoundaryDimensionContract::Fixed(*value as usize)
                    }
                    other => panic!("unexpected RNNT predictor wire dimension {other:?}"),
                })
                .collect();
            BoundaryRoleContract::symbolic(symbolic.role(), dtype, shape).unwrap()
        });
    RoleExactBoundaryContract::new(schema.identity(), roles).unwrap()
}

fn communication(
    config: RnntDimensions,
    tensor_ranks: usize,
    rank: usize,
    cut: usize,
    publication_elements: usize,
) -> (
    CommunicationManifest,
    Vec<PartitionBoundaryRoute>,
    CollectiveGroupId,
    Option<CollectiveGroupId>,
) {
    let world = tensor_ranks * 2;
    let session = CollectiveGroupId::new(1);
    let tensor_requirement = |operation, dtypes| {
        let count = if operation == CommunicationOperation::SendReceive {
            // Hidden activation, original zero-state prototype, and step/cut metadata.
            3
        } else {
            1
        };
        let limits = CommunicationTensorLimits::new(count, 3, 8192, None).unwrap();
        CommunicationOperationRequirement::tensors(operation, dtypes, limits, true).unwrap()
    };
    let mut groups = vec![CommunicationGroupDescriptor::new(
        session,
        0,
        (0..world).collect(),
        Some(rank),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 2, publication_elements, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap()];
    let mut local_tensor = None;
    if tensor_ranks > 1 {
        let stage = rank / tensor_ranks;
        let id = CollectiveGroupId::new(10 + stage as u32);
        groups.push(
            CommunicationGroupDescriptor::new(
                id,
                1,
                (stage * tensor_ranks..(stage + 1) * tensor_ranks).collect(),
                Some(rank % tensor_ranks),
                CommunicationGroupRequirements::new([
                    tensor_requirement(
                        CommunicationOperation::AllReduceSum,
                        vec![TensorDtype::F32],
                    ),
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap(),
        );
        local_tensor = Some(id);
    }
    let mut routes = Vec::new();
    let mut descriptors = Vec::new();
    for lane in 0..tensor_ranks {
        let route = CommunicationRouteId::new(100 + lane as u64);
        routes.push(PartitionBoundaryRoute {
            source_group: 0,
            destination_group: 0,
            source_rank: lane,
            destination_rank: tensor_ranks + lane,
            route,
        });
        descriptors.push(
            CommunicationRouteDescriptor::new(
                route,
                lane,
                lane,
                tensor_ranks + lane,
                tensor_requirement(
                    CommunicationOperation::SendReceive,
                    vec![TensorDtype::F32, TensorDtype::I32],
                ),
            )
            .unwrap()
            .with_boundary_contract(boundary_contract(config, cut))
            .unwrap(),
        );
    }
    let manifest = CommunicationManifest::new(world, rank, groups, descriptors)
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(30),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    (manifest, routes, session, local_tensor)
}

pub(super) fn launch(size: usize, worker: &str) {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let sockets = (0..size)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = Children(Vec::new());
    for rank in 0..size {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", worker, "--nocapture", "--test-threads=1"])
                .env(WORKER, size.to_string())
                .env(DIRECTORY, directory.path())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if Instant::now() >= deadline || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                let _ = child.kill();
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // Reap every worker before asserting so one failed rank cannot abandon its peers.
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    let mut success = true;
    let mut diagnostics = String::new();
    for (rank, output) in outputs.into_iter().enumerate() {
        match output {
            Ok(output) => {
                success &= output.status.success();
                diagnostics.push_str(&format!(
                    "world {size} rank {rank}: {}\n{}\n{}\n",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ));
            }
            Err(error) => {
                success = false;
                diagnostics.push_str(&format!(
                    "world {size} rank {rank}: failed to reap: {error}\n"
                ));
            }
        }
    }
    assert!(success, "RNNT pipeline worker failures:\n{diagnostics}");
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_rnnt_predictor_pp2_and_tp2_pp2_preserve_nvidia_readout_and_local_carry() {
    launch(2, "pipeline::native_rnnt_predictor_pipeline_worker");
    launch(4, "pipeline::native_rnnt_predictor_pipeline_worker");
}
