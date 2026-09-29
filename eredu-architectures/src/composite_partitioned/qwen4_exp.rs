//! Flash-Next binds retained media sources to exact target and encoder placement.
use super::*;
use crate::qwen4_exp::{
    conditional::ConditionalModel,
    media::MediaIngress,
    prepared::PreparedConditionalPartition,
    target::{BoundTargetSpec, TargetModel, TargetTensorPartition},
};
use eredu_runtime::ArchitectureParameters;

type GatedPlan = crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>;

fn invalid<E>(error: impl std::fmt::Display) -> CompositePartitionPreparationError<E> {
    CompositePartitionPreparationError::Architecture(error.to_string())
}

/// The target authors exact recurrent/GQA/expert selections; shared vision
/// placement supplies its independent heads and merger channels. The resulting
/// description retains this authority rather than reconstructing it at binding.
fn local_geometry<B>(
    bound: &BoundTargetSpec,
    ingress: &MediaIngress,
    vision: &crate::qwen::vision::VisionConfig,
    rank: eredu_core::ParallelRankTopology,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (
        TargetTensorPartition,
        eredu_runtime::ArchitectureParameterDescription,
        LocalModelLayout,
        GatedPlan,
    ),
    String,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    let error = |error: eredu_nn::Error| error.to_string();
    let tensor = bound
        .geometry()
        .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())
        .map_err(error)?;
    let global = TargetModel::<B>::new(bound.clone(), context).map_err(error)?;
    let target_parameters = global.parameter_description(context).map_err(error)?;
    let mut local = TargetModel::<B>::new_tensor_parallel(bound.clone(), tensor.clone(), context)
        .map_err(error)?;
    let tensor_parameters = local.parameter_description(context).map_err(error)?;
    let experts = tensor
        .local_spec()
        .expert_realization(rank)
        .map_err(error)?;
    local.set_expert_realization(&experts).map_err(error)?;
    let target_layout = tensor
        .local_expert_layout(
            &target_parameters,
            &tensor_parameters,
            &local.parameter_description(context).map_err(error)?,
            rank.expert_parallel_rank(),
            rank.expert_parallel_size(),
        )
        .map_err(error)?;
    let global =
        ConditionalModel::<B>::new(bound.clone(), ingress.clone(), vision.clone(), context)
            .map_err(error)?;
    let parameters = global.parameter_description(context).map_err(error)?;
    let mut layout = derive_partitioned_local_layout(&parameters, rank)?;
    for (name, placement) in target_layout.tensors() {
        layout.insert(name.to_owned(), placement.clone());
    }
    let parameters = parameters
        .with_partition_layout(rank, layout.clone())
        .map_err(|error| error.to_string())?;
    Ok((tensor, parameters, layout, experts))
}

impl PreparedConditionalPartition {
    /// Builds the selected conditional graph and delegates only generic native
    /// materialization, row/bank acquisition, collectives and session mechanisms.
    pub fn visit<B, S, V>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Result<V::Output, CompositePartitionPreparationError<V::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>
            + RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: AuthoritativeCompositePartitionVisitor<B, S>,
    {
        let prepared = self.prepare::<B, S, V::Error>(context)?;
        visitor
            .visit(prepared)
            .map_err(CompositePartitionPreparationError::Visitor)
    }

    fn prepare<B, S, E>(
        self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        PreparedCompositePartition<
            ConditionalModel<B>,
            TargetTensorPartition,
            <ConditionalModel<B> as PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        CompositePartitionPreparationError<E>,
    >
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>
            + RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
    {
        let target = self.prepared_target().clone();
        let ingress = self.ingress().clone();
        let row_sources = self.row_sources().clone();
        let capability = self.capability_estimate().clone();
        let source_bound = self.tensor_target().source_bound_spec().clone();
        let selected = self.selected().clone();
        let rank = selected.requirements().topology();
        let SelectedCompositeTextRealization::Routed { execution, .. } = selected.base() else {
            return Err(invalid(
                "Flash-Next conditional partition lost its routed selection",
            ));
        };
        let bound = target.selected_bound_spec(execution).map_err(invalid)?;
        let mut vision = ingress.vision().config().clone();
        vision.linear_formats.clear();
        for (name, format) in crate::replicated_text::selected_matrix_formats(
            execution.text().requirements(),
            execution.text(),
        ) {
            if name.starts_with("model.visual.") {
                vision.linear_formats.insert(name, format);
            }
        }
        let (tensor, parameters, layout, experts) =
            local_geometry::<B>(&bound, &ingress, &vision, rank, context).map_err(invalid)?;
        let state = selected
            .requirements()
            .state()
            .ok_or_else(|| invalid("Flash-Next conditional partition has no target state"))?;
        let mut architecture = ConditionalModel::<B>::new_partitioned(
            bound,
            tensor.clone(),
            ingress.clone(),
            vision,
            rank,
            &layout,
            state,
            context,
        )
        .map_err(invalid)?;
        architecture
            .set_expert_realization(&experts)
            .map_err(invalid)?;
        let boundary =
            <ConditionalModel<B> as PartitionedLayeredArchitecture<B, S>>::boundary_schema(
                &architecture,
            )
            .map_err(invalid)?;
        let partition = ArchitecturePartition::from_architecture::<B, S, _, _>(
            &architecture,
            selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str(), group.units())),
            selected.requirements().ownership().clone(),
            tensor,
            boundary,
            &parameters,
        )
        .map_err(invalid)?;
        if partition.state() != selected.requirements().state() {
            return Err(invalid(
                "constructed Flash-Next conditional state differs from cold admission",
            ));
        }
        let source_architecture =
            if crate::replicated_text::selected_uses_transform(execution.text()) {
                let source_vision = ingress.vision().config().clone();
                let (source_tensor, source_parameters, source_layout, source_experts) =
                    local_geometry::<B>(&source_bound, &ingress, &source_vision, rank, context)
                        .map_err(invalid)?;
                if crate::partitioned_execution::derive_partitioned_transform_source_layout(
                    &source_parameters,
                    &parameters,
                    rank,
                )
                .map_err(invalid)?
                    != source_layout
                {
                    return Err(invalid(
                        "Flash-Next conditional transform changed exact source placement",
                    ));
                }
                let mut source = ConditionalModel::<B>::new_partitioned(
                    source_bound,
                    source_tensor,
                    ingress,
                    source_vision,
                    rank,
                    &source_layout,
                    state,
                    context,
                )
                .map_err(invalid)?;
                source
                    .set_expert_realization(&source_experts)
                    .map_err(invalid)?;
                if source.state_layout().map_err(invalid)?
                    != architecture.state_layout().map_err(invalid)?
                {
                    return Err(invalid(
                        "Flash-Next conditional transform source changed local state geometry",
                    ));
                }
                Some((Box::new(source), source_layout))
            } else {
                None
            };
        let owned = selected
            .requirements()
            .groups()
            .iter()
            .find(|group| group.group().as_str() == crate::decoder::TARGET_EXECUTION_GROUP)
            .ok_or_else(|| invalid("Flash-Next conditional partition has no target group"))?
            .units();
        let rows = match execution.row_lookups() {
            Some(plan) => {
                let sources = eredu_runtime::PreparedRowLookups::new(
                    row_sources
                        .entries()
                        .values()
                        .filter(|row| owned.contains(&row.spec().unit))
                        .cloned(),
                    target.spec().units.len(),
                )
                .map_err(invalid)?;
                Some(
                    plan.clone()
                        .for_units(owned, target.spec().units.len())
                        .map_err(invalid)?
                        .bind(sources)
                        .map_err(invalid)?,
                )
            }
            None if row_sources.entries().is_empty() => None,
            None => {
                return Err(invalid(
                    "Flash-Next conditional partition lost row lookup selection",
                ));
            }
        };
        let owner_units = experts
            .unit_specs()
            .keys()
            .map(|(_, unit)| (*unit, *unit))
            .collect();
        let routed = prepared_gated_composite_execution::<B, S, _>(
            &selected,
            &architecture,
            &layout,
            experts,
            owner_units,
            target.spec().units.len(),
            target.spec().configuration().hidden_size as usize,
        )
        .map_err(invalid)?;
        let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
            selected.materialization_tasks(),
            &parameters,
            &partition,
        )
        .map_err(invalid)?;
        let publication = crate::partitioned_execution::PublicationValueDescriptor::new(
            target.spec().configuration().vocabulary,
        )
        .map_err(invalid)?;
        prepare_composite_partition::<B, S, _, _, _>(
            architecture,
            source_architecture,
            selected,
            partition,
            CompositePartitionDetails {
                layout,
                tasks,
                capability_estimate: capability,
                effective_model_type: "qwen4_exp".into(),
                publication,
                routed: Some(routed),
                rows,
            },
        )
        .map_err(invalid)
    }
}

impl PreparedConditionalPartition {
    /// Pairs the selected conditional partition and retained prediction owners.
    pub fn visit_prediction<B, S, V>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        visitor: V,
        binding: crate::prepared_execution::PredictionBinding,
    ) -> Result<V::Output, CompositePartitionPreparationError<V::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::HyperNeuralBackend
            + 'static,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>
            + RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: AuthoritativeCompositePartitionVisitor<B, S>,
    {
        let prediction = self
            .prediction_weights::<B>(context)
            .map_err(invalid)?
            .with_discovery_binding(&binding)
            .map_err(invalid)?;
        let spec = self.prediction_spec().map_err(invalid)?;
        let prediction_layout = prediction
            .local_layout()
            .cloned()
            .ok_or_else(|| invalid("prediction has no retained tensor layout"))?;
        let selected = self.selected().clone();
        let SelectedCompositeTextRealization::Routed { execution, .. } = selected.base() else {
            return Err(invalid("prediction requires a routed target"));
        };
        let source = self.prepared_target().artifact().clone();
        let target_source = crate::replicated_text::restrict_store_handoff(
            execution.text().requirements(),
            source.clone(),
            crate::replicated_text::StoreHandoffScope::Primary,
        )
        .map_err(invalid)?;
        let provider_source = crate::partitioned_execution::qwen4_exp::prediction_provider_source(
            execution.text().requirements(),
            source,
            prediction.source().clone(),
        )
        .map_err(invalid)?;
        let mut prepared = self.prepare::<B, S, V::Error>(context)?;
        let mut provider_layout = prepared.layout.clone();
        for (name, layout) in prediction_layout.tensors() {
            provider_layout.insert(name.to_owned(), layout.clone());
        }
        let banks = crate::partitioned_execution::qwen4_exp::prediction_banks(
            execution,
            &spec,
            &provider_layout,
            execution.text().auxiliary_materialization_tasks(),
            selected.requirements().topology(),
        )
        .map_err(invalid)?;
        let retained = prepared
            .banks
            .as_mut()
            .ok_or_else(|| invalid("prediction target has no selected parameter banks"))?;
        retained.extend_banks(banks).map_err(invalid)?;
        prepared.provider_layout = Some(provider_layout);
        visitor.visit_prediction(
            prepared,
            prediction,
            target_source,
            provider_source,
            binding,
        )
    }
}
