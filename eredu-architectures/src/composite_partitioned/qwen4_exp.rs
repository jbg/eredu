//! Flash-Next binds retained media sources to exact target and encoder placement.
use super::*;
use crate::qwen4_exp::{
    conditional::{ConditionalKernel, ConditionalModel},
    media::MediaIngress,
    prepared::PreparedConditionalPartition,
    target::{BoundTargetSpec, TargetTensorPartition},
};

type GatedPlan = crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>;

fn invalid<E>(error: impl std::fmt::Display) -> CompositePartitionPreparationError<E> {
    CompositePartitionPreparationError::Architecture(error.to_string())
}

/// The target authors exact recurrent/GQA/expert selections; shared vision
/// placement supplies its independent heads and merger channels. The resulting
/// description retains this authority rather than reconstructing it at binding.
fn local_geometry(
    bound: &BoundTargetSpec,
    ingress: &MediaIngress,
    vision: &crate::qwen::vision::VisionConfig,
    rank: eredu_core::ParallelRankTopology,
) -> Result<
    (
        TargetTensorPartition,
        eredu_runtime::ArchitectureParameterDescription,
        LocalModelLayout,
        GatedPlan,
    ),
    String,
> {
    let error = |error: eredu_nn::Error| error.to_string();
    let tensor = bound
        .geometry()
        .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())
        .map_err(error)?;
    let definition = bound.geometry().definition().map_err(error)?;
    let target_parameters = definition.parameters();
    let local = tensor.local_spec();
    let tensor_parameters = local.parameter_description().map_err(error)?;
    let experts = local.expert_realization(rank).map_err(error)?;
    let expert_parameters = local
        .clone()
        .with_expert_realization(&experts)
        .and_then(|spec| spec.parameter_description())
        .map_err(error)?;
    let target_layout = tensor
        .local_expert_layout(
            target_parameters,
            &tensor_parameters,
            &expert_parameters,
            rank.expert_parallel_rank(),
            rank.expert_parallel_size(),
        )
        .map_err(error)?;
    crate::qwen4_exp::conditional::validate_ingress(bound.geometry(), ingress, vision)
        .map_err(error)?;
    let parameters =
        crate::qwen4_exp::conditional::parameter_description(&definition, vision).map_err(error)?;
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
            local_geometry(&bound, &ingress, &vision, rank).map_err(invalid)?;
        let state = selected
            .requirements()
            .state()
            .ok_or_else(|| invalid("Flash-Next conditional partition has no target state"))?;
        let mut architecture = ConditionalKernel::<B>::new_partitioned(
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
        architecture = architecture
            .with_expert_realization(&experts)
            .map_err(invalid)?;
        let boundary = architecture.target_boundary_schema().map_err(invalid)?;
        let partition = ArchitecturePartition::from_description(
            &parameters,
            selected
                .requirements()
                .groups()
                .iter()
                .map(|group| (group.group().as_str(), group.units())),
            selected.requirements().ownership().clone(),
            architecture.definition().state_layout(),
            architecture.definition().state_partition_plan(),
            tensor,
            boundary,
        )
        .map_err(invalid)?;
        if partition.state() != selected.requirements().state() {
            return Err(invalid(
                "constructed Flash-Next conditional state differs from cold admission",
            ));
        }
        let source_architecture =
            if execution.text().uses_weight_transform() {
                let source_vision = ingress.vision().config().clone();
                let (source_tensor, source_parameters, source_layout, source_experts) =
                    local_geometry(&source_bound, &ingress, &source_vision, rank)
                        .map_err(invalid)?;
                if eredu_runtime::derive_partitioned_transform_source_layout(
                    &source_parameters,
                    &parameters,
                    rank,
                )
                .map_err(|error| invalid(error.to_string()))?
                    != source_layout
                {
                    return Err(invalid(
                        "Flash-Next conditional transform changed exact source placement",
                    ));
                }
                let mut source = ConditionalKernel::<B>::new_partitioned(
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
                source = source
                    .with_expert_realization(&source_experts)
                    .map_err(invalid)?;
                if source.definition().state_layout() != architecture.definition().state_layout() {
                    return Err(invalid(
                        "Flash-Next conditional transform source changed local state geometry",
                    ));
                }
                Some((
                    Box::new(source.into_model().map_err(invalid)?),
                    source_layout,
                ))
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
        let architecture = architecture
            .into_partition_model(&partition)
            .map_err(invalid)?;
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
        let target_source = eredu_runtime::restrict_checkpoint_handoff(
            execution.text().requirements(),
            source.clone(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| invalid(handoff.to_string()))?;
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
