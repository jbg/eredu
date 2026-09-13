//! Dense MLA partitions consume the direct selection and ordinary parameter tasks.
use super::*;

fn architecture<E>(error: impl std::fmt::Display) -> DenseDecoderPartitionedDispatchError<E> {
    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
}

pub(super) fn prepare<B, S, V>(
    args: &crate::deepseek::V3Args,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    if args.has_sparse_moe_layers() || args.num_nextn_predict_layers != 0 {
        return Err(architecture(
            "direct V3 partition requires a dense prediction-free schedule",
        ));
    }
    let selected_args = crate::replicated_text::selected_deepseek_v3_args(args, selected.base())
        .map_err(architecture)?;
    let parameters = crate::deepseek::parallel::v3_parameter_description(&selected_args)
        .map_err(architecture)?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank).map_err(architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(architecture("dense V3 partition must own one target group"));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(architecture(
            "dense V3 admission names another execution group",
        ));
    }
    let start = owned.units().start;
    let geometry = crate::deepseek::parallel::v3_local_geometry(&selected_args, &layout)
        .map_err(architecture)?;
    let complete_state = geometry.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry.clone(),
        crate::deepseek::v3::TargetBoundarySchema::from_args(&selected_args),
    )
    .map_err(architecture)?;
    validate_partitioned_binding(selected.requirements(), &partition).map_err(architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(architecture)?;
    let source_architecture = if selected.base().parameters().iter().any(|parameter| {
        matches!(
            parameter.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        )
    }) {
        let source_args = crate::replicated_text::source_deepseek_v3_args(args, selected.base())
            .map_err(architecture)?;
        let source_parameters = crate::deepseek::parallel::v3_parameter_description(&source_args)
            .map_err(architecture)?;
        validate_transform_parameter_space(&source_parameters, &parameters)
            .map_err(architecture)?;
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(architecture)?;
        let source_geometry =
            crate::deepseek::parallel::v3_local_geometry(&source_args, &source_layout)
                .map_err(architecture)?;
        if source_geometry.state_layout() != &complete_state {
            return Err(architecture(
                "dense V3 transform source changed local state geometry",
            ));
        }
        let source_partition = ArchitecturePartition::from_description(
            &source_parameters,
            [(owned.group().as_str(), owned.units())],
            selected.requirements().ownership().clone(),
            &complete_state,
            &state_plan,
            source_geometry.clone(),
            crate::deepseek::v3::TargetBoundarySchema::from_args(&source_args),
        )
        .map_err(architecture)?;
        validate_partitioned_binding(selected.requirements(), &source_partition)
            .map_err(architecture)?;
        if source_partition.units().ne(partition.units()) {
            return Err(architecture(
                "dense V3 transform source changed local unit addresses",
            ));
        }
        let mut model =
            crate::deepseek::v3::Model::<B>::new_parallel(source_args, source_geometry, context)
                .map_err(architecture)?;
        model.set_partition_target_start(start);
        Some((model, source_layout))
    } else {
        None
    };
    let capability_estimate =
        crate::capability::deepseek_v3(&selected_args).map_err(architecture)?;
    let effective_model_type = selected_args.model_type.clone();
    let mut model = crate::deepseek::v3::Model::<B>::new_parallel(selected_args, geometry, context)
        .map_err(architecture)?;
    model.set_partition_target_start(start);
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(model, selected, partition)
        .map_err(architecture)?;
    visitor
        .visit(
            PreparedPartitionedArchitecture {
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                backend: PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}
