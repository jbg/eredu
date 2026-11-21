//! Typed routed partition and its selected storage and execution authority.

mod assembly;
mod visitor;
use assembly::{construct_routed_partition, BoundRoutedPartition};
pub use visitor::*;

use std::collections::BTreeSet;

use crate::{ArchitecturePartition, LocalModelLayout, PipelinePartitionArchitecture};
use eredu_core::communication::CommunicationManifest;

use super::{
    dense_partition::DenseDecoderPartitionedDispatchError,
    partition_executors::PreparedRoutedExecutionHandoff,
    routed_partition::{
        PartitionBankBinding, PartitionBankPlan, PartitionBankSelection, PreparedPartitionBanks,
    },
    PreparedPartitionSessionFacts,
};

/// One checked routed partition with source preparation, physical placement and native binding inputs.
pub struct PreparedRoutedPartitionedArchitecture<B, A, G, W, Bank> {
    architecture: A,
    base: crate::SelectedReplicatedTextRealization,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    source_architecture: Option<(A, LocalModelLayout)>,
    layout: LocalModelLayout,
    provider_layout: Option<LocalModelLayout>,
    tasks: Vec<crate::ReplicatedTextMaterializationTask>,
    facts: Result<PreparedPartitionSessionFacts, String>,
    tensor_rank: usize,
    banks: PreparedPartitionBanks<Bank>,
    rows: Option<crate::SelectedRowLookups>,
    unowned_sources: BTreeSet<String>,
    excluded: BTreeSet<String>,
    execution: PreparedRoutedExecutionHandoff,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<B, A, G, W, Bank> PreparedRoutedPartitionedArchitecture<B, A, G, W, Bank>
where
    B: eredu_nn::GroupedNeuralBackend,
    Bank: PartitionBankPlan,
{
    /// Retains the complete typed partition and its projected binding inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_admission(
        architecture: A,
        base: crate::SelectedReplicatedTextRealization,
        partition: ArchitecturePartition<G, W>,
        communication: CommunicationManifest,
        source_architecture: Option<(A, LocalModelLayout)>,
        layout: LocalModelLayout,
        provider_layout: Option<LocalModelLayout>,
        tasks: Vec<crate::ReplicatedTextMaterializationTask>,
        facts: Result<PreparedPartitionSessionFacts, String>,
        tensor_rank: usize,
        banks: PreparedPartitionBanks<Bank>,
        rows: Option<crate::SelectedRowLookups>,
        unowned_sources: BTreeSet<String>,
        excluded: BTreeSet<String>,
        execution: PreparedRoutedExecutionHandoff,
    ) -> Self {
        Self {
            architecture,
            base,
            partition,
            communication,
            source_architecture,
            layout,
            provider_layout,
            tasks,
            facts,
            tensor_rank,
            banks,
            rows,
            unowned_sources,
            excluded,
            execution,
            backend: std::marker::PhantomData,
        }
    }

    /// Returns the retained session facts from portable admission.
    fn session_facts(&self) -> Result<PreparedPartitionSessionFacts, String> {
        self.facts.clone()
    }

    /// Returns this rank's coordinate within the selected tensor group.
    pub const fn tensor_rank(&self) -> usize {
        self.tensor_rank
    }

    /// Returns the row policies assigned to this stage.
    pub fn row_lookups(&self) -> Option<&crate::SelectedRowLookups> {
        self.rows
            .as_ref()
            .filter(|rows| !rows.prepared().entries().is_empty())
    }

    /// Returns the tensor rank responsible for row acquisition.
    pub fn row_lookup_owner(&self) -> Option<usize> {
        self.row_lookups().map(|_| 0)
    }

    /// Returns inputs for allocating native bank storage and row acquisition.
    pub fn bank_binding(&self) -> PartitionBankBinding<'_> {
        self.banks.binding(
            self.provider_layout.as_ref().unwrap_or(&self.layout),
            self.row_lookups()
                .filter(|_| Some(self.tensor_rank) == self.row_lookup_owner()),
        )
    }

    /// Returns checkpoint keys materialized by other expert owners.
    pub fn unowned_expert_checkpoint_sources(&self) -> BTreeSet<String> {
        self.unowned_sources.clone()
    }

    /// Returns logical targets owned by the independent bank providers.
    fn addressable_logical_targets(&self) -> BTreeSet<String> {
        self.banks.addressable_logical_targets()
    }

    /// Consumes this exact routed admission into a rank-local runtime factory handoff.
    fn prepare_session_runtime<S, R, FactoryError, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        factory: F,
    ) -> Result<
        crate::PreparedPartitionedSessionRuntime<R, S>,
        crate::PartitionedSessionPreparationError<FactoryError>,
    >
    where
        S: crate::LayerRuntimeState<B>,
        A: crate::LayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        F: FnOnce(
            crate::PartitionedSessionFactoryInput<'_, A, G, W>,
            Option<(A, crate::LocalModelLayout)>,
            crate::LocalModelLayout,
            &crate::SelectedReplicatedTextRealization,
            PreparedRoutedExecutionHandoff,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<(R, S), FactoryError>,
    {
        let Self {
            architecture,
            base,
            partition,
            communication,
            excluded,
            source_architecture,
            layout,
            provider_layout: _,
            tasks,
            facts: _,
            tensor_rank: _,
            unowned_sources: _,
            banks: _,
            rows: _,
            execution,
            backend: _,
        } = self;
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                crate::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        crate::validate_retained_parameter_layout(&parameters, &layout).map_err(|error| {
            crate::PartitionedSessionPreparationError::Contract(error.to_string())
        })?;
        if let Some((source, source_layout)) = source_architecture.as_ref() {
            let source_parameters = source.parameter_description(context).map_err(|error| {
                crate::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
            crate::validate_transform_parameter_space(&source_parameters, &parameters).map_err(
                |error| crate::PartitionedSessionPreparationError::Contract(error.to_string()),
            )?;
            crate::validate_retained_parameter_layout(&source_parameters, source_layout).map_err(
                |error| crate::PartitionedSessionPreparationError::Contract(error.to_string()),
            )?;
        }
        let excluded = excluded.iter().map(String::as_str).collect();
        crate::prepare_partitioned_session_runtime_with_exclusions::<
            _,
            B,
            _,
            S,
            _,
            _,
            FactoryError,
            _,
        >(
            architecture,
            base,
            partition,
            communication,
            Some(&tasks),
            &excluded,
            topology,
            crate::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| {
                factory(
                    input,
                    source_architecture,
                    layout,
                    selected,
                    execution,
                    context,
                )
            },
        )
    }
}

impl<B, A, G, W, Bank: PartitionBankPlan> PartitionBankSelection
    for PreparedRoutedPartitionedArchitecture<B, A, G, W, Bank>
{
    type Bank = Bank;

    fn partition_bank_selection(&self) -> &PreparedPartitionBanks<Bank> {
        &self.banks
    }
}

/// Family-blind backend visitor for an exact routed decoder partition.
pub trait RoutedPartitionedProductionVisitor<B, S, Bank>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: crate::RuntimeState<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Binds independently prepared prediction owners with the selected target
    /// partition and their retained source authorities.
    fn visit_prediction<A, G, P>(
        self,
        _prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
        _prediction: P,
        _target_source: eredu_checkpoint::store::SharedCheckpointSource,
        _provider_source: eredu_checkpoint::store::SharedCheckpointSource,
        _binding: super::PredictionBinding,
    ) -> Result<Self::Output, DenseDecoderPartitionedDispatchError<Self::Error>>
    where
        Self: Sized,
        B: eredu_nn::BlockwiseAttentionBackend + eredu_nn::HyperNeuralBackend,
        A: PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
        P: crate::speculative::PreparedRoutedPrediction<B, A>,
    {
        Err(DenseDecoderPartitionedDispatchError::Architecture(
            "retained partitioned prediction binding is unavailable".into(),
        ))
    }

    /// Receives one immutable architecture-selected routed partition.
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            B,
            A,
            G,
            <A as crate::PartitionedLayeredArchitecture<B, S>>::Boundary,
            Bank,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

/// Family-blind backend visitor for an exact routed partitioned prediction target.
pub trait RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, Bank>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::RuntimeState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives one routed target after architecture-owned extension pairing.
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            B,
            A,
            G,
            <A as crate::PartitionedLayeredArchitecture<B, S>>::Boundary,
            Bank,
        >,
        extension: <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}
