//! Total construction of independently owned shared-backbone text partitions.
use super::*;

pub(super) fn prepare<B, S, V>(
    args: &crate::nemotron_labs_diffusion::FamilyConfig,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    use crate::nemotron_labs_diffusion::{parallel, partitioned::PartitionedTextKernel};
    let error = |message: String| DenseDecoderPartitionedDispatchError::Architecture(message);
    let target = args
        .with_checkpoint_formats(crate::replicated_text::selected_formats(selected.base()))
        .map_err(|e| error(e.to_string()))?;
    let retained = selected
        .base()
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| error("shared-backbone selection has no retained definition".into()))?;
    let definition =
        crate::nemotron_labs_diffusion::definition::project_text_parameters(&target, &retained)
            .map_err(|e| error(e.to_string()))?;
    let parameters = definition.parameters();
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(parameters, rank).map_err(error)?;
    let [owned] = selected.requirements().groups() else {
        return Err(error(
            "shared-backbone text partition must own one decoder group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TEXT_DECODER_EXECUTION_GROUP {
        return Err(error(
            "shared-backbone text ownership names another execution group".into(),
        ));
    }
    let partition = parallel::partition(
        &target,
        &definition,
        &layout,
        owned.units(),
        selected.requirements().ownership().clone(),
    )
    .map_err(|e| error(e.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition).map_err(error)?;
    let source_architecture = if selected.base().uses_weight_transform() {
        let source = args
            .with_checkpoint_formats(crate::replicated_text::requirement_formats(
                selected.base().requirements(),
            ))
            .map_err(|e| error(e.to_string()))?;
        let description = retained.parameters();
        let source_layout =
            derive_partitioned_transform_source_layout(description, parameters, rank)
                .map_err(|failure| error(failure.to_string()))?;
        let source_partition = parallel::partition(
            &source,
            &retained,
            &source_layout,
            owned.units(),
            selected.requirements().ownership().clone(),
        )
        .map_err(|e| error(e.to_string()))?;
        validate_partitioned_binding(selected.requirements(), &source_partition).map_err(error)?;
        let model = PartitionedTextKernel::<B>::from_partition(
            source,
            retained.clone(),
            &source_partition,
            context,
        )
        .map_err(|e| error(e.to_string()))?;
        Some((model, source_layout))
    } else {
        None
    };
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        parameters,
        &partition,
    )
    .map_err(|e| error(e.to_string()))?;
    let capability_estimate = crate::capability::nemotron_labs_diffusion(&target, false)
        .map_err(|e| error(e.to_string()))?;
    let effective_model_type = crate::decoder::Config::model_identity(&target).to_owned();
    let model = PartitionedTextKernel::<B>::from_partition(target, definition, &partition, context)
        .map_err(|e| error(e.to_string()))?;
    let prepared =
        prepare_partitioned::<B, S, _, _, _, _, _>(model, selected, partition).map_err(error)?;
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
