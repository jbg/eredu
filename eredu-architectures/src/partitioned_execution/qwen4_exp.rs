//! Flash-Next binds retained canonical tasks to exact authored rank placement.
use super::*;
use crate::qwen4_exp::{prepared::PreparedTargetPartition, target::TargetModel};
use eredu_runtime::ArchitectureParameters;

fn invalid<E>(error: impl std::fmt::Display) -> DenseDecoderPartitionedDispatchError<E> {
    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
}

impl PreparedTargetPartition {
    /// Constructs the selected stream-state target and delegates native binding
    /// through the ordinary family-neutral partition visitor.
    pub fn visit<B, S, V>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        visitor: V,
    ) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: RoutedPartitionedProductionVisitor<B, S>,
    {
        self.visit_with::<B, S, _>(context, OrdinaryFamilyRoutedPartitionVisitor(visitor))
    }

    fn visit_with<B, S, V>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        visitor: V,
    ) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: FamilyRoutedPartitionVisitor<B, S, TargetModel<B>>,
    {
        let target = self.prepared_target().clone();
        let rows = self.row_sources().clone();
        let capability = self.capability_estimate().clone();
        let (_, source_partition, selected) = self.into_parts();
        let rank = selected.requirements().topology();
        let [owned] = selected.requirements().groups() else {
            return Err(invalid(
                "Flash-Next partition must own one target execution group",
            ));
        };
        if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
            return Err(invalid(
                "Flash-Next partition selected another execution group",
            ));
        }
        let owned_units = owned.units();

        // The selected formats may differ from checkpoint storage. Both plans
        // retain their own physical companions, packing and exact TP selections.
        let bound = target
            .selected_bound_spec(selected.base())
            .map_err(invalid)?;
        let tensor = bound
            .geometry()
            .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())
            .map_err(invalid)?;
        let global_model = TargetModel::<B>::new(bound.clone(), context).map_err(invalid)?;
        let mut parameters = global_model
            .parameter_description(context)
            .map_err(invalid)?;
        let mut model = TargetModel::<B>::new_tensor_parallel(bound, tensor.clone(), context)
            .map_err(invalid)?;
        let local_parameters = model.parameter_description(context).map_err(invalid)?;
        let experts = tensor
            .local_spec()
            .expert_realization(rank)
            .map_err(invalid)?;
        model.set_expert_realization(&experts).map_err(invalid)?;
        let layout = tensor
            .local_expert_layout(
                &parameters,
                &local_parameters,
                &model.parameter_description(context).map_err(invalid)?,
                rank.expert_parallel_rank(),
                rank.expert_parallel_size(),
            )
            .map_err(invalid)?;
        retained_layout::validate(&parameters, &layout).map_err(invalid)?;
        parameters = parameters
            .with_partition_layout(rank, layout.clone())
            .map_err(invalid)?;
        let complete_state = model.state_layout().map_err(invalid)?;
        let state_plan = crate::transport::pipeline_state(0, &complete_state);
        let partition = ArchitecturePartition::from_description(
            &parameters,
            [(crate::decoder::TARGET_EXECUTION_GROUP, owned_units.clone())],
            selected.requirements().ownership().clone(),
            &complete_state,
            &state_plan,
            tensor.clone(),
            target.spec().boundary_schema().map_err(invalid)?,
        )
        .map_err(invalid)?;
        validate_partitioned_binding(selected.requirements(), &partition).map_err(invalid)?;
        let state = partition
            .state()
            .ok_or_else(|| invalid("Flash-Next target partition has no state"))?;
        model.set_partition_state(state).map_err(invalid)?;
        model.retain_global_parameters(parameters.clone());

        let source_architecture =
            if crate::replicated_text::selected_uses_transform(selected.base().text()) {
                let source_bound = source_partition.source_bound_spec().clone();
                let source_global =
                    TargetModel::<B>::new(source_bound.clone(), context).map_err(invalid)?;
                let source_parameters = source_global
                    .parameter_description(context)
                    .map_err(invalid)?;
                validate_transform_parameter_space(&source_parameters, &parameters)
                    .map_err(invalid)?;
                let mut source_model = TargetModel::<B>::new_tensor_parallel(
                    source_bound,
                    source_partition.partition().clone(),
                    context,
                )
                .map_err(invalid)?;
                let source_tensor_parameters = source_model
                    .parameter_description(context)
                    .map_err(invalid)?;
                let source_experts = source_partition
                    .partition()
                    .local_spec()
                    .expert_realization(rank)
                    .map_err(invalid)?;
                source_model
                    .set_expert_realization(&source_experts)
                    .map_err(invalid)?;
                let source_layout = source_partition
                    .partition()
                    .local_expert_layout(
                        &source_parameters,
                        &source_tensor_parameters,
                        &source_model
                            .parameter_description(context)
                            .map_err(invalid)?,
                        rank.expert_parallel_rank(),
                        rank.expert_parallel_size(),
                    )
                    .map_err(invalid)?;
                if source_model.state_layout().map_err(invalid)? != complete_state {
                    return Err(invalid(
                        "Flash-Next transform source changed local state geometry",
                    ));
                }
                source_model.set_partition_state(state).map_err(invalid)?;
                source_model.retain_global_parameters(
                    source_parameters
                        .with_partition_layout(rank, source_layout.clone())
                        .map_err(invalid)?,
                );
                Some((source_model, source_layout))
            } else {
                None
            };

        let rows = match selected.base().row_lookups() {
            Some(plan) => {
                let sources = eredu_runtime::PreparedRowLookups::new(
                    rows.entries()
                        .values()
                        .filter(|row| owned_units.contains(&row.spec().unit))
                        .cloned(),
                    target.spec().units.len(),
                )
                .map_err(invalid)?;
                Some(
                    plan.clone()
                        .for_units(owned_units, target.spec().units.len())
                        .map_err(invalid)?
                        .bind(sources)
                        .map_err(invalid)?,
                )
            }
            None if rows.entries().is_empty() => None,
            None => {
                return Err(invalid(
                    "Flash-Next partition has no retained row selection",
                ));
            }
        };
        let source = crate::replicated_text::restrict_store_handoff(
            selected.base().text().requirements(),
            target.artifact().clone(),
            crate::replicated_text::StoreHandoffScope::Primary,
        )
        .map_err(invalid)?;
        // Tasks contain global canonical recipes; the native loader applies the
        // retained layout once. Passing already sliced recipes here would select twice.
        prepare_family_routed_partition::<B, S, _, _, _, _>(
            model,
            source_architecture,
            selected,
            partition,
            parameters,
            layout,
            rows,
            experts,
            source,
            visitor,
            capability,
            "qwen4_exp_text".into(),
        )
    }
}

/// Keeps target-owned shared parameters while retaining a separate predictor's
/// non-overlapping physical source keys in the joint provider view.
pub(crate) fn prediction_provider_source(
    requirements: &eredu_runtime::ReplicatedTextRequirements,
    target: eredu_checkpoint::store::SharedCheckpointSource,
    prediction: eredu_checkpoint::store::SharedCheckpointSource,
) -> Result<eredu_checkpoint::store::SharedCheckpointSource, String> {
    let target_keys = target.source_keys().into_iter().collect::<BTreeSet<_>>();
    let prediction_keys = prediction
        .source_keys()
        .into_iter()
        .filter(|key| !target_keys.contains(key))
        .collect::<BTreeSet<_>>();
    let complete = if prediction_keys.is_empty() {
        target
    } else {
        let prediction: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                prediction,
                "prediction-only physical parameters",
                prediction_keys,
            )
            .map_err(|error| error.to_string())?,
        );
        std::sync::Arc::new(
            eredu_checkpoint::store::CompositeCheckpointSource::new([target, prediction])
                .map_err(|error| error.to_string())?,
        )
    };
    crate::replicated_text::restrict_store_handoff(
        requirements,
        complete,
        crate::replicated_text::StoreHandoffScope::Complete,
    )
}

/// Prediction banks replicate expert ownership across stage and expert groups;
/// their matrix widths retain the independently selected tensor partition.
pub(crate) fn prediction_banks(
    selected: &SelectedRoutedTextRealization,
    spec: &crate::qwen4_exp::mtp::PredictionSpec,
    layout: &eredu_runtime::LocalModelLayout,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    rank: eredu_core::ParallelRankTopology,
) -> Result<BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>, String> {
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(rank.tensor_parallel_size(), 1, 1, 1)
            .map_err(|error| error.to_string())?,
        rank.tensor_parallel_rank(),
    )
    .map_err(|error| error.to_string())?;
    let plan = spec
        .expert_realization(topology)
        .map_err(|error| error.to_string())?;
    spec.units
        .iter()
        .map(|unit| unit.feed_forward.feed_forward.bank)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|id| {
            let bank = selected
                .bank(id)
                .ok_or_else(|| "prediction bank is absent from retained selection".to_owned())?;
            let members = crate::routed_text::project_addressable_members_with_tasks(
                bank.catalog(),
                selected.text(),
                tasks,
            )
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|member| member.with_owner_rank(rank.global_rank()))
            .collect();
            let bank = bank
                .with_partition_geometry(
                    plan.clone().into(),
                    bank.catalog().clone(),
                    members,
                    layout,
                )
                .map_err(|error| error.to_string())?;
            Ok((id, bank))
        })
        .collect()
}

impl PreparedTargetPartition {
    /// Pairs retained prediction owners with the same target partition constructor.
    pub fn visit_prediction<B, S, V>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        visitor: V,
        binding: crate::prepared_execution::PredictionBinding,
    ) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::HyperNeuralBackend
            + 'static,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: RoutedPartitionedProductionVisitor<B, S>,
    {
        let prediction = self
            .prediction_weights::<B>(context)
            .map_err(invalid)?
            .with_discovery_binding(&binding)
            .map_err(invalid)?;
        let spec = self.prediction_spec().map_err(invalid)?;
        let layout = prediction
            .local_layout()
            .cloned()
            .ok_or_else(|| invalid("prediction has no retained tensor layout"))?;
        let source = prediction_provider_source(
            self.selected().base().text().requirements(),
            self.prepared_target().artifact().clone(),
            prediction.source().clone(),
        )
        .map_err(invalid)?;
        self.visit_with::<B, S, _>(
            context,
            RetainedPredictionVisitor {
                visitor,
                prediction,
                spec,
                layout,
                source,
                binding,
            },
        )
        .map_err(|error| match error {
            DenseDecoderPartitionedDispatchError::Architecture(error) => invalid(error),
            DenseDecoderPartitionedDispatchError::Visitor(error) => error,
        })
    }
}

struct RetainedPredictionVisitor<V, P> {
    visitor: V,
    prediction: P,
    spec: crate::qwen4_exp::mtp::PredictionSpec,
    layout: eredu_runtime::LocalModelLayout,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    binding: crate::prepared_execution::PredictionBinding,
}
impl<B, S, V, P> FamilyRoutedPartitionVisitor<B, S, TargetModel<B>>
    for RetainedPredictionVisitor<V, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    V: RoutedPartitionedProductionVisitor<B, S>,
    P: crate::prediction_extension::PreparedRoutedPrediction<B, TargetModel<B>>,
{
    type Output = V::Output;
    type Error = DenseDecoderPartitionedDispatchError<V::Error>;
    fn visit<G>(
        self,
        mut prepared: PreparedRoutedPartitionedArchitecture<
            B,
            TargetModel<B>,
            G,
            <TargetModel<B> as eredu_runtime::PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        source: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static,
    {
        let mut provider_layout = prepared.layout.clone();
        for (name, layout) in self.layout.tensors() {
            provider_layout.insert(name.to_owned(), layout.clone());
        }
        let selected = prepared.prepared.selected();
        let banks = prediction_banks(
            selected.base(),
            &self.spec,
            &provider_layout,
            selected.base().text().auxiliary_materialization_tasks(),
            selected.topology(),
        )
        .map_err(invalid)?;
        if banks.keys().any(|id| prepared.banks.contains_key(id)) {
            return Err(invalid("prediction and target parameter banks overlap"));
        }
        prepared.banks.extend(banks);
        prepared.provider_layout = Some(provider_layout);
        self.visitor
            .visit_prediction(prepared, self.prediction, source, self.source, self.binding)
    }
}
