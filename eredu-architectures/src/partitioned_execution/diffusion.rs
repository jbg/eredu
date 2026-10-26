//! Shared iterative partitions consume exact cold selection and prepared tasks.
use super::*;

fn architecture<E>(error: impl std::fmt::Display) -> DenseDecoderPartitionedDispatchError<E> {
    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
}

pub(super) fn prepare_routed<B, S, V>(
    args: &crate::diffusion_gemma::FamilyConfig,
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
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    V: FamilyRoutedPartitionVisitor<B, S, crate::diffusion_gemma::model::TextModel<B>>,
{
    let mut selected_args = args.clone();
    selected_args.text.weight_quantization = None;
    selected_args.text.quantized_weights = None;
    let formats = selected.base().text().quantized_formats();
    selected_args.text.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
    let retained = selected
        .base()
        .text()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| architecture("shared iterative selection has no retained definition"))?;
    let definition = crate::diffusion_gemma::model::project_parameters(&selected_args, &retained)
        .map_err(architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank).map_err(architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(architecture(
            "shared iterative partition must own one target group",
        ));
    };
    if owned.group().as_str() != crate::diffusion_gemma::model::TEXT_EXECUTION_GROUP {
        return Err(architecture(
            "shared iterative admission names another execution group",
        ));
    }
    let geometry =
        crate::diffusion_gemma::parallel::local_geometry(&selected_args, &layout, owned.units())
            .map_err(architecture)?;
    let plan = crate::diffusion_gemma::model::expert_realization_plan(
        &selected_args,
        Some(&geometry),
        rank,
    )
    .map_err(architecture)?
    .ok_or_else(|| architecture("selected shared stack has no routed bank"))?;
    let complete_state = geometry
        .state_layout()
        .cloned()
        .ok_or_else(|| architecture("shared iterative partition has no context slots"))?;
    let partition = crate::diffusion_gemma::parallel::partition(
        &selected_args,
        definition.clone(),
        &layout,
        owned.units(),
        selected.requirements().ownership().clone(),
    )
    .map_err(architecture)?;
    let source_architecture = if selected.base().text().uses_weight_transform() {
        let source_parameters = retained.parameters();
        let source_layout =
            derive_partitioned_transform_source_layout(source_parameters, &parameters, rank)
                .map_err(|error| architecture(error.to_string()))?;
        let source_geometry =
            crate::diffusion_gemma::parallel::local_geometry(args, &source_layout, owned.units())
                .map_err(architecture)?;
        if source_geometry.state_layout() != Some(&complete_state) {
            return Err(architecture(
                "shared iterative transform source changed state geometry",
            ));
        }
        let source_plan = crate::diffusion_gemma::model::expert_realization_plan(
            args,
            Some(&source_geometry),
            rank,
        )
        .map_err(architecture)?
        .ok_or_else(|| architecture("source shared stack has no routed bank"))?;
        let source_partition = crate::diffusion_gemma::parallel::partition(
            args,
            retained.clone(),
            &source_layout,
            owned.units(),
            selected.requirements().ownership().clone(),
        )
        .map_err(architecture)?;
        validate_partitioned_binding(selected.requirements(), &source_partition)
            .map_err(architecture)?;
        if source_partition.units().ne(partition.units()) {
            return Err(architecture(
                "transform source changed shared unit addresses",
            ));
        }
        let mut model = crate::diffusion_gemma::model::TextKernel::<B>::from_partition(
            args.clone(),
            retained.clone(),
            &source_partition,
            context,
        )
        .map_err(architecture)?;
        model
            .kernel_mut()
            .set_expert_realization(source_plan)
            .map_err(architecture)?;
        Some((model, source_layout))
    } else {
        None
    };
    let capability_estimate =
        crate::capability::diffusion_gemma(&selected_args).map_err(architecture)?;
    let effective_model_type = "diffusion_gemma".to_owned();
    let mut model = crate::diffusion_gemma::model::TextKernel::<B>::from_partition(
        selected_args,
        definition,
        &partition,
        context,
    )
    .map_err(architecture)?;
    model
        .kernel_mut()
        .set_expert_realization(plan.clone())
        .map_err(architecture)?;
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

pub(super) fn prepare_dense<B, S, V>(
    args: &crate::diffusion_gemma::FamilyConfig,
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
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let mut selected_args = args.clone();
    selected_args.text.weight_quantization = None;
    selected_args.text.quantized_weights = None;
    let formats = selected.base().quantized_formats();
    selected_args.text.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| architecture("shared iterative selection has no retained definition"))?;
    let definition = crate::diffusion_gemma::model::project_parameters(&selected_args, &retained)
        .map_err(architecture)?;
    let parameters = definition.parameters().clone();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank).map_err(architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(architecture(
            "shared iterative partition must own one target group",
        ));
    };
    if owned.group().as_str() != crate::diffusion_gemma::model::TEXT_EXECUTION_GROUP {
        return Err(architecture(
            "shared iterative admission names another execution group",
        ));
    }
    let geometry =
        crate::diffusion_gemma::parallel::local_geometry(&selected_args, &layout, owned.units())
            .map_err(architecture)?;
    let complete_state = geometry
        .state_layout()
        .cloned()
        .ok_or_else(|| architecture("shared iterative partition has no context slots"))?;
    let partition = crate::diffusion_gemma::parallel::partition(
        &selected_args,
        definition.clone(),
        &layout,
        owned.units(),
        selected.requirements().ownership().clone(),
    )
    .map_err(architecture)?;
    let source_architecture = if selected.base().uses_weight_transform() {
        let source_parameters = retained.parameters();
        let source_layout =
            derive_partitioned_transform_source_layout(source_parameters, &parameters, rank)
                .map_err(|error| architecture(error.to_string()))?;
        let source_geometry =
            crate::diffusion_gemma::parallel::local_geometry(args, &source_layout, owned.units())
                .map_err(architecture)?;
        if source_geometry.state_layout() != Some(&complete_state) {
            return Err(architecture(
                "shared iterative transform source changed state geometry",
            ));
        }
        let source_partition = crate::diffusion_gemma::parallel::partition(
            args,
            retained.clone(),
            &source_layout,
            owned.units(),
            selected.requirements().ownership().clone(),
        )
        .map_err(architecture)?;
        validate_partitioned_binding(selected.requirements(), &source_partition)
            .map_err(architecture)?;
        if source_partition.units().ne(partition.units()) {
            return Err(architecture(
                "transform source changed shared unit addresses",
            ));
        }
        let model = crate::diffusion_gemma::model::TextKernel::<B>::from_partition(
            args.clone(),
            retained.clone(),
            &source_partition,
            context,
        )
        .map_err(architecture)?;
        Some((model, source_layout))
    } else {
        None
    };
    let capability_estimate =
        crate::capability::diffusion_gemma(&selected_args).map_err(architecture)?;
    let effective_model_type = "diffusion_gemma".to_owned();
    let model = crate::diffusion_gemma::model::TextKernel::<B>::from_partition(
        selected_args,
        definition,
        &partition,
        context,
    )
    .map_err(architecture)?;
    validate_partitioned_binding(selected.requirements(), &partition).map_err(architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(architecture)?;
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
