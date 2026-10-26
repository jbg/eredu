//! Qwen hybrid partitions consume retained selection and exact prepared tasks.
use super::*;

fn architecture<E>(error: impl std::fmt::Display) -> DenseDecoderPartitionedDispatchError<E> {
    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
}

pub(super) fn prepare<B, S, V>(
    args: &crate::qwen::hybrid::HybridConfig,
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
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    prepare_with::<B, S, _, _, _>(args, selected, store, context, |prepared, store| {
        visitor.visit(prepared, store)
    })
}

fn prepare_with<B, S, F, O, E>(
    args: &crate::qwen::hybrid::HybridConfig,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    finish: F,
) -> Result<O, DenseDecoderPartitionedDispatchError<E>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    F: FnOnce(
        PreparedPartitionedArchitecture<
            B,
            crate::qwen::hybrid::LayeredModel<B>,
            crate::qwen::hybrid::LocalGeometry,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<O, E>,
{
    if args.is_moe() || args.mtp_num_hidden_layers != 0 {
        return Err(architecture(
            "direct Qwen hybrid partition requires a dense prediction-free schedule",
        ));
    }
    let selected_args = crate::replicated_text::selected_qwen_hybrid_args(args, selected.base())
        .map_err(architecture)?;
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| architecture("Qwen hybrid selection has no retained definition"))?;
    let definition =
        crate::qwen::hybrid::project_parameters(&selected_args, &retained).map_err(architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank).map_err(architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(architecture(
            "dense Qwen hybrid partition must own one target group",
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(architecture(
            "dense Qwen hybrid admission names another execution group",
        ));
    }
    let geometry =
        crate::qwen::hybrid::local_geometry(&selected_args, &layout).map_err(architecture)?;
    let complete_state = geometry.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry.clone(),
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(architecture)?;
    selected.requirements().validate_binding(&partition).map_err(architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(architecture)?;
    let source_architecture = if selected.base().uses_weight_transform() {
        let source_args = args.clone();
        let source_definition = crate::qwen::hybrid::project_parameters(&source_args, &retained)
            .map_err(architecture)?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|error| architecture(error.to_string()))?;
        let source_geometry = crate::qwen::hybrid::local_geometry(&source_args, &source_layout)
            .map_err(architecture)?;
        if source_geometry.state_layout() != &complete_state {
            return Err(architecture(
                "dense Qwen hybrid transform source changed local state geometry",
            ));
        }
        let source_partition = ArchitecturePartition::from_description(
            &source_parameters,
            [(owned.group().as_str(), owned.units())],
            selected.requirements().ownership().clone(),
            &complete_state,
            &state_plan,
            source_geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
        )
        .map_err(architecture)?;
        selected.requirements().validate_binding(&source_partition)
            .map_err(architecture)?;
        if source_partition.units().ne(partition.units()) {
            return Err(architecture(
                "dense Qwen hybrid transform source changed local unit addresses",
            ));
        }
        let model = crate::qwen::hybrid::QwenKernel::<B>::from_partition(
            source_args,
            source_definition,
            &source_partition,
            context,
        )
        .map_err(architecture)?;
        Some((model, source_layout))
    } else {
        None
    };
    let capability_estimate =
        crate::capability::qwen_hybrid_text(&selected_args).map_err(architecture)?;
    let effective_model_type = selected_args.model_type.clone();
    let model = crate::qwen::hybrid::QwenKernel::<B>::from_partition(
        selected_args,
        definition,
        &partition,
        context,
    )
    .map_err(architecture)?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(model, selected, partition)
        .map_err(architecture)?;
    finish(
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

pub(super) fn prepare_prediction<B, S, M, V>(
    args: &crate::qwen::hybrid::HybridConfig,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: PartitionedPredictionTargetVisitor<B, S, M>,
{
    let extension = <crate::qwen::hybrid::LayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension).map_err(architecture)?;
    prepare_with::<B, S, _, _, _>(args, selected, store, context, |prepared, store| {
        visitor.visit(prepared, extension, store)
    })
}

pub(super) fn prepare_routed<B, S, V>(
    args: &crate::qwen::hybrid::HybridConfig,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: FamilyRoutedPartitionVisitor<B, S, crate::qwen::hybrid::LayeredModel<B>>,
{
    if !args.is_moe() || args.mtp_num_hidden_layers != 0 {
        return Err(architecture(
            "routed Qwen hybrid requires a prediction-free target",
        ));
    }
    let selected_args =
        crate::replicated_text::selected_qwen_hybrid_args(args, selected.base().text())
            .map_err(architecture)?;
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| architecture("Qwen hybrid selection has no retained definition"))?;
    let definition =
        crate::qwen::hybrid::project_parameters(&selected_args, &retained).map_err(architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank).map_err(architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(architecture(
            "routed Qwen hybrid partition must own one target group",
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(architecture(
            "routed Qwen hybrid admission names another execution group",
        ));
    }
    let geometry =
        crate::qwen::hybrid::local_geometry(&selected_args, &layout).map_err(architecture)?;
    let plan =
        crate::qwen::hybrid::partition_expert_realization_plan(&selected_args, &geometry, rank)
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
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(architecture)?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let source_definition =
            crate::qwen::hybrid::project_parameters(args, &retained).map_err(architecture)?;
        let source_parameters = source_definition.parameters().clone();
        let source_layout =
            derive_partitioned_transform_source_layout(&source_parameters, &parameters, rank)
                .map_err(|error| architecture(error.to_string()))?;
        let source_geometry =
            crate::qwen::hybrid::local_geometry(args, &source_layout).map_err(architecture)?;
        if source_geometry.state_layout() != &complete_state {
            return Err(architecture(
                "routed Qwen hybrid transform source changed state geometry",
            ));
        }
        let source_plan =
            crate::qwen::hybrid::partition_expert_realization_plan(args, &source_geometry, rank)
                .map_err(architecture)?;
        let source_partition = ArchitecturePartition::from_description(
            &source_parameters,
            [(owned.group().as_str(), owned.units())],
            selected.requirements().ownership().clone(),
            &complete_state,
            &state_plan,
            source_geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size),
        )
        .map_err(architecture)?;
        selected.requirements().validate_binding(&source_partition)
            .map_err(architecture)?;
        if source_partition.units().ne(partition.units()) {
            return Err(architecture(
                "routed Qwen hybrid transform source changed local unit addresses",
            ));
        }
        let mut model = crate::qwen::hybrid::QwenKernel::<B>::from_partition(
            args.clone(),
            source_definition,
            &source_partition,
            context,
        )
        .map_err(architecture)?;
        model.kernel_mut().install_expert_realization(source_plan);
        Some((model, source_layout))
    } else {
        None
    };
    let capability_estimate =
        crate::capability::qwen_hybrid_text(&selected_args).map_err(architecture)?;
    let effective_model_type = selected_args.model_type.clone();
    let mut model = crate::qwen::hybrid::QwenKernel::<B>::from_partition(
        selected_args,
        definition,
        &partition,
        context,
    )
    .map_err(architecture)?;
    model.kernel_mut().install_expert_realization(plan.clone());
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        model,
        source_architecture,
        selected,
        partition,
        parameters,
        layout,
        None,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}
