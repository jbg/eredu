use super::super::*;
use super::*;

pub(crate) struct PartitionedCompositeBindingVisitor<'a> {
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

pub(crate) trait CompositeExecutableFinalizer<A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static;
}

impl<A> CompositeExecutableFinalizer<A> for OrdinaryReplicatedFinalizer
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        Ok(Box::new(completed))
    }
}

impl<A, P> CompositeExecutableFinalizer<A> for PredictionReplicatedFinalizer<P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn finish<D>(
        self,
        completed: CompletedComposite<A, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
                MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        completed
            .with_prediction(self.prediction, self.capability)
            .map(|completed| Box::new(completed) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedCompositeBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

pub(crate) struct PartitionedCompositePredictionBindingVisitor<'a> {
    pub store: Arc<dyn CheckpointSource>,
    pub distributed: crate::backend::distributed::MlxDistributedSession,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionPredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PartitionedCompositePredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            PredictionReplicatedFinalizer {
                prediction: SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                capability: self.capability,
            },
        )
    }
}

pub(crate) fn bind_prepared_partitioned_composite<A, G, W, F>(
    prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    F: CompositeExecutableFinalizer<A>,
{
    let execution_plan = prepared
        .execution_plan()
        .map_err(Error::ArchitectureModel)?;
    let publication_authority = execution_plan
        .publication_authority(prepared.communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "composite partition has no selected output publication authority".into(),
            )
        })?;
    let selected_residency = prepared
        .prepared()
        .selected()
        .base()
        .execution()
        .residency();
    let session_group = prepared
        .prepared()
        .selected()
        .session_group()
        .ok_or_else(|| {
            Error::ArchitectureModel("composite partition has no selected session group".into())
        })?;
    let prompt_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let local_state = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| {
            Error::ArchitectureModel("composite partition has no selected local state".into())
        })?;
    let prompt_cache_identity = local_state
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let admission = prepared.prepared().architecture().admission_config();
    let processor = prepared.prepared().selected().base().processor().clone();
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let mut mechanisms: MlxReplicatedTextMechanisms<
        PreparedCompositeArchitecture<A>,
        MlxHybridState,
    > = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    #[cfg(test)]
    {
        crate::tests::support::path_instrumentation::constructor();
        crate::tests::support::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime::<MlxNeuralBackend, _, _, _, _>(
            prompt_topology.clone(),
            stream,
            |input, physical_layout, executor_plan, selected, context| {
                let tensor_group = executor_plan.communication_tensor_group();
                let prepared = eredu_runtime::prepare_default_partitioned_runtime(
                    input,
                    None,
                    physical_layout,
                    None,
                    selected,
                    &prompt_topology,
                    eredu_runtime::PartitionedUnitScope::Owned,
                    &[],
                    &mut mechanisms,
                    context,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let (architecture, _partition, manifest, execution_policy, bounded_policy, state) =
                    prepared.into_parts();
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("composite communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(manifest, tensor_group, session_group)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor = executor_plan.bind(
                    architecture.into_inner(),
                    execution_policy,
                    parallel,
                    MlxPartitionTensorAllocator,
                    super::super::distributed::expert::MlxExpertRouteTensorMovement::new(stream),
                )?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        eredu_runtime::PartitionedTextExecution::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    finalizer.finish(
        CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
            session,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            selected_residency,
            partition_sampling_group,
            partition_communication_authority,
            Some(publication_authority.owner_group_rank()),
            publication_authority.local_public_output(),
            stream,
        ),
    )
}

pub(crate) fn bind_partitioned_composite(
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        eredu_architectures::replicated_text::CompositeTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::composite_partitioned::visit_authoritative_composite_partition::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        selected,
        stream,
        PartitionedCompositeBindingVisitor {
            store,
            distributed,
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(crate) fn bind_partitioned_dense_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        SelectedReplicatedTextRealization,
        ReplicatedTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::visit_resident_partitioned_architecture::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        inspection,
        selected,
        store,
        stream,
        PartitionedDenseDecoderBindingVisitor {
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(crate) fn bind_partitioned_routed_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::dispatch_routed_partitioned_production(
        inspection,
        selected,
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        ),
        |(store, distributed, additional, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxHybridState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedRoutedDecoderBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(store, distributed, additional, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_relu2_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxHybridState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedRoutedDecoderBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(store, distributed, _, stream, weights_stream), inspection, selected| {
            eredu_architectures::partitioned_execution::visit_pooling_routed_partitioned_production::<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                _,
            >(
                inspection,
                selected,
                store,
                stream,
                PartitionedPoolingRoutedDecoderBindingVisitor {
                    distributed,
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_partitioned_routed_prediction_decoder(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    extension: MaterializedEmbeddedPrediction,
    realization: eredu_runtime::SelectedSpeculativeRealization,
    capability: eredu_architectures::capability::CapabilityEstimate,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    eredu_architectures::partitioned_execution::dispatch_routed_partitioned_production(
        inspection,
        selected,
        (
            extension,
            realization,
            capability,
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
        ),
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxHybridState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_relu2_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxHybridState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |(
            extension,
            realization,
            capability,
            store,
            distributed,
            additional,
            stream,
            weights_stream,
        ),
         inspection,
         selected| {
            eredu_architectures::partitioned_execution::visit_pooling_routed_partitioned_prediction_target_production::<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                MlxEmbeddedPredictionMaterializer,
                _,
            >(
                inspection,
                selected,
                extension,
                store,
                stream,
                PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: additional,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
    )
}
