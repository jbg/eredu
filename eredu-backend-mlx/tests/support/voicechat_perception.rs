//! Exact retained perception parameters execute through native residency policies.
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission, checkpoint::Component, conformer::ConformerDimensions,
        perception::PerceptionDimensions, perception_preparation::PreparedPerceptionBindings,
        subsampling::SubsamplingDimensions,
    },
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes,
    native::{
        materialize_voicechat_perception_pipeline, MlxPerceptionPipeline,
        MlxPerceptionPipelineState,
    },
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_core::residency::OffloadConfig;
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions};
use safemlx::{Array, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::Value;

use eredu_architectures::voicechat::{
    perception_distribution::PreparedPerceptionPipelineBindings,
    perception_layered::{
        perception_parameter_description, PerceptionArchitecture, PerceptionBoundarySchema,
    },
};
use eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend;
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, ParallelRankTopology,
    ParallelTopology,
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

pub fn floats(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
pub fn tensor(value: &Value) -> MlxTensor {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_array(Array::from_slice(&floats(value), &shape))
}
pub fn dimensions(fixture: &Value) -> PerceptionDimensions {
    let geometry =
        |part: &str, name: &str| fixture[part]["geometry"][name].as_i64().unwrap() as i32;
    PerceptionDimensions {
        subsampling: SubsamplingDimensions {
            features: geometry("subsampling", "features"),
            channels: geometry("subsampling", "channels"),
            width: geometry("subsampling", "width"),
        },
        block: ConformerDimensions {
            width: geometry("conformer", "width"),
            heads: geometry("conformer", "heads"),
            ff_hidden: geometry("conformer", "feed_forward"),
            conv_kernel: geometry("conformer", "kernel"),
            left_context: geometry("conformer", "left_context"),
        },
        layers: 1,
        output_width: fixture["perception"]["output_width"].as_i64().unwrap() as i32,
    }
}
pub fn checkpoint(fixture: &Value, f16: bool, path: &std::path::Path) -> SharedCheckpointSource {
    let mut values: Vec<_> = ["subsampling", "conformer", "perception"]
        .into_iter()
        .flat_map(|part| {
            fixture[part]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| {
                    let shape: Vec<usize> = value["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let bytes: Vec<u8> = if f16 {
                        floats(value)
                            .into_iter()
                            .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                            .collect()
                    } else {
                        floats(value)
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect()
                    };
                    (name.clone(), shape, bytes)
                })
        })
        .collect();
    values.push((
        "unrelated.weight".into(),
        vec![1],
        vec![0; if f16 { 2 } else { 4 }],
    ));
    let dtype = if f16 { Dtype::F16 } else { Dtype::F32 };
    serialize_to_file(
        values.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        path,
    )
    .unwrap();
    Arc::new(SafetensorsWeightStore::open(path).unwrap())
}
pub fn prepared(
    dimensions: PerceptionDimensions,
    source: SharedCheckpointSource,
) -> PreparedPerceptionBindings {
    let parameters: BTreeMap<_, _> = source
        .source_keys()
        .into_iter()
        .map(|name| {
            let metadata = source.source_metadata(&name).unwrap();
            let recipe = DerivedWeightRecipe::Source {
                key: name.clone(),
                selection: TensorSelection::Full,
            };
            let output = recipe.infer(source.as_ref()).unwrap();
            let peak = recipe.peak_materialization_bytes(source.as_ref()).unwrap()
                + metadata.encoded_byte_len;
            (
                name.clone(),
                ParameterAdmission {
                    name,
                    component: Component::Perception,
                    artifact: PhysicalArtifactId::Primary,
                    output,
                    stored_bytes: metadata.encoded_byte_len,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: peak,
                    recipe,
                },
            )
        })
        .collect();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let bindings =
        PreparedPerceptionBindings::from_parameters(dimensions, source.clone(), &parameters)
            .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold binding projection reads no payload"
    );
    bindings
}

pub struct PipelineFixture {
    pub model: MlxPerceptionPipeline,
    pub state: MlxPerceptionPipelineState,
    pub largest: u64,
    pub owns_block: bool,
}

pub struct PreparedPipelineFixture {
    pub prepared: PreparedPerceptionPipelineBindings,
    pub largest: u64,
    pub total: u64,
    pub owns_block: bool,
    pub residency: LayerWeightResidency,
}

pub fn prepare_pipeline(
    dimensions: PerceptionDimensions,
    source: SharedCheckpointSource,
    cut: usize,
    mode: u64,
    world: &safemlx::distributed::Group,
) -> PreparedPipelineFixture {
    let rank = world.rank();
    let tensor_ranks = world.size() / 2;
    let stage = rank / tensor_ranks;
    let global = prepared(dimensions, source.clone());
    let (global, local) = if tensor_ranks > 1 {
        let topology = ParallelRankTopology::new(
            ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
            rank % tensor_ranks,
        )
        .unwrap();
        let (bindings, layout) = global.select_parallel(topology).unwrap();
        (bindings, Some(layout))
    } else {
        (global, None)
    };
    let description = perception_parameter_description(dimensions).unwrap();
    let complete = if let Some(layout) = &local {
        PerceptionArchitecture::<MlxNeuralBackend>::new_parallel(dimensions, layout).unwrap()
    } else {
        PerceptionArchitecture::<MlxNeuralBackend>::new(dimensions).unwrap()
    };
    let owns_block = (cut == 0 && stage == 1) || (cut == 1 && stage == 0);
    let mut owned = Vec::new();
    if stage == 0 {
        owned.push((description.graph().groups()[0].id(), 0..1));
    }
    if owns_block {
        owned.push((description.graph().groups()[1].id(), 0..1));
    }
    if stage == 1 {
        owned.push((description.graph().groups()[2].id(), 0..1));
    }
    let partition = ArchitecturePartition::from_description(
        &description,
        owned,
        PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
        &complete.state_layout().unwrap(),
        &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            1,
            0..1,
        )]),
        (),
        PerceptionBoundarySchema::new(dimensions),
    )
    .unwrap();
    let selected = global.into_partition(&partition).unwrap();
    let sizes = selected
        .unit_bindings()
        .iter()
        .map(|(_, unit)| binding_bytes(unit).unwrap())
        .collect::<Vec<_>>();
    let largest = *sizes.iter().max().unwrap();
    let total: u64 = sizes.iter().sum();
    let residency = match mode {
        0 => LayerWeightResidency::FullyResident,
        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(largest), Some(1 << 20), 1).unwrap(),
        )),
        _ => LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(largest * 2, 0, 0, 0).unwrap(),
        ),
    };
    let (manifest, routes, session, tensor_group) =
        communication(dimensions, tensor_ranks, rank, cut, cut + 1);
    let mut drivers = vec![None, None, None];
    for group in partition.groups() {
        drivers[group.group_index()] = Some(
            LayeredPartitionDriver::new_with_state_ownership(
                &partition,
                group.group_index(),
                group.global_units(),
                group.group_index() == 1,
            )
            .unwrap(),
        );
    }
    let plan = PartitionedExecutionPlan::new(
        description.graph().clone(),
        vec![
            (ArchitectureGroupKind::AudioEncoder, false),
            (ArchitectureGroupKind::AudioEncoder, false),
            (ArchitectureGroupKind::Projector, false),
        ],
        drivers,
        routes,
        Some(PartitionOutputPublication {
            group: session,
            owner_rank: tensor_ranks,
        }),
        Some(session),
        PipelineWireContract::new(PipelineActivationDtype::Float32),
    )
    .unwrap();
    let prepared = PreparedPerceptionPipelineBindings::new(
        selected,
        partition,
        manifest,
        plan,
        tensor_group,
        PipelineActivationDtype::Float32,
    )
    .unwrap();
    PreparedPipelineFixture {
        prepared,
        largest,
        total,
        owns_block,
        residency,
    }
}

pub fn pipeline(
    dimensions: PerceptionDimensions,
    source: SharedCheckpointSource,
    cut: usize,
    mode: u64,
    world: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
) -> PipelineFixture {
    let PreparedPipelineFixture {
        prepared,
        largest,
        total,
        owns_block,
        residency,
    } = prepare_pipeline(dimensions, source, cut, mode, world);
    let (model, state, metadata) =
        materialize_voicechat_perception_pipeline(prepared, &world, residency, &weights, &stream)
            .unwrap();

    assert_eq!(metadata.static_device_bytes(), 0);
    assert_eq!(metadata.layer_parameter_bytes(), total);
    assert_eq!(metadata.residency(), residency.execution_residency());
    PipelineFixture {
        model,
        state,
        largest,
        owns_block,
    }
}
fn boundary_contract(dimensions: PerceptionDimensions) -> RoleExactBoundaryContract {
    let schema = PerceptionBoundarySchema::new(dimensions)
        .wire_schema()
        .unwrap();
    let resolved = schema.resolve(2, 16).unwrap();
    let roles = std::iter::once(schema.primary())
        .chain(schema.auxiliary())
        .zip(std::iter::once(resolved.primary()).chain(resolved.auxiliary()))
        .map(|(symbolic, resolved)| {
            let dtype = match symbolic.dtype() {
                BoundaryTensorDtype::Activation | BoundaryTensorDtype::Float32 => TensorDtype::F32,
                BoundaryTensorDtype::Int32 => TensorDtype::I32,
                other => panic!("unexpected fixture wire type {other:?}"),
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
                    other => panic!("unexpected fixture wire dimension {other:?}"),
                })
                .collect();
            BoundaryRoleContract::symbolic(symbolic.role(), dtype, shape).unwrap()
        });
    RoleExactBoundaryContract::new(schema.identity(), roles).unwrap()
}

fn communication(
    dimensions: PerceptionDimensions,
    tensor_ranks: usize,
    rank: usize,
    source_group: usize,
    destination_group: usize,
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
            3
        } else {
            1
        };
        let limits = CommunicationTensorLimits::new(count, 3, 2048, None).unwrap();
        CommunicationOperationRequirement::tensors(operation, dtypes, limits, true).unwrap()
    };
    let mut groups = vec![CommunicationGroupDescriptor::new(
        session,
        0,
        (0..world).collect(),
        Some(rank),
        CommunicationGroupRequirements::new([
            tensor_requirement(CommunicationOperation::Broadcast, vec![TensorDtype::F32]),
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap()];
    let mut local_tensor = None;
    if tensor_ranks > 1 {
        for stage in 0..2 {
            let members = (stage * tensor_ranks..(stage + 1) * tensor_ranks).collect::<Vec<_>>();
            let local = members.iter().position(|&member| member == rank);
            let group = CollectiveGroupId::new(10 + stage as u32);
            if local.is_none() {
                continue;
            }
            local_tensor = Some(group);
            groups.push(
                CommunicationGroupDescriptor::new(
                    group,
                    1,
                    members,
                    local,
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
        }
    }
    let mut routes = Vec::new();
    let mut descriptors = Vec::new();
    for lane in 0..tensor_ranks {
        let route = CommunicationRouteId::new(100 + lane as u64);
        routes.push(PartitionBoundaryRoute {
            source_group,
            destination_group,
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
            .with_boundary_contract(boundary_contract(dimensions))
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
