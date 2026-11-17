use super::super::*;

use crate::backend::{
    distributed::MlxDistributedSession,
    nn::shared::MlxCommunicationTensorMetadata,
    runtime::distributed::{Group, topology::CommunicationRouteRealization},
};
use eredu_architectures::prepared_execution::{
    PreparedPartitionSessionFacts, PreparedTextSessionFacts,
};
use eredu_core::{CollectiveGroupId, cache::PromptCacheTopology};
use eredu_runtime::{
    LocalModelLayout, PartitionCommunication, PartitionCommunicationAuthority,
    PartitionOutputAuthority, PartitionedExecutionPlan, PartitionedSessionFactoryInput,
    PreparedPartitionedSessionRuntime, SelectedReplicatedTextRealization,
};
use std::collections::BTreeSet;

pub(super) struct PartitionRankPreparation<'a, A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    facts: PreparedPartitionSessionFacts,
    mechanisms: MlxReplicatedTextMechanisms<A, S>,
    distributed: Option<MlxDistributedSession>,
    sampling: Option<Group>,
    authority: Option<PartitionCommunicationAuthority>,
    stream: &'a Stream,
}

pub(super) struct BoundPartitionRank<A, G, W, S, P> {
    pub architecture: A,
    pub partition: eredu_runtime::ArchitecturePartition<G, W>,
    pub policy: P,
    pub bounded_policy: Option<P>,
    pub state: S,
    pub communication: PartitionCommunication<
        MlxNeuralBackend,
        Group,
        CommunicationRouteRealization,
        MlxCommunicationTensorMetadata,
    >,
    pub parallel: Option<Group>,
    pub communication_executor: Stream,
    pub row_status: Option<Group>,
}

/// A bound session and the retained publication metadata shared by final adapters.
pub(super) struct BoundPartitionSession<'a, A, S, D>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    pub session: ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    pub text: PreparedTextSessionFacts,
    pub publication: PartitionOutputAuthority,
    pub sampling: Option<Group>,
    pub authority: Option<PartitionCommunicationAuthority>,
    pub stream: &'a Stream,
}

impl<A, S, D> BoundPartitionSession<'_, A, S, D>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    pub(super) fn into_text(self) -> CompletedReplicatedText<A, S, D> {
        let (identity, capability, model_type, residency) = self.text.into_parts();
        CompletedReplicatedText::from_session(
            self.session,
            identity,
            capability,
            model_type,
            residency,
            self.sampling,
            self.authority,
            Some(self.publication.owner_group_rank()),
            self.publication.local_public_output(),
            self.stream,
        )
    }
}

impl<'a, A, S> PartitionRankPreparation<'a, A, S>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::Unit: 'static,
{
    pub(super) fn new(
        facts: PreparedPartitionSessionFacts,
        store: Arc<dyn CheckpointSource>,
        distributed: MlxDistributedSession,
        claimed_sources: BTreeSet<String>,
        stream: &'a Stream,
        weights_stream: &Stream,
        prediction_residency: super::super::super::prediction::parameters::PredictionResidency,
    ) -> Self {
        let mut mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
        mechanisms.set_prediction_residency(prediction_residency);
        mechanisms.set_ignored_checkpoint_sources(claimed_sources);
        #[cfg(test)]
        {
            crate::tests::support::path_instrumentation::constructor();
            crate::tests::support::path_instrumentation::neutral_partitioned_construction();
        }
        Self {
            facts,
            mechanisms,
            distributed: Some(distributed),
            sampling: None,
            authority: None,
            stream,
        }
    }

    pub(super) fn topology(&self) -> PromptCacheTopology {
        self.facts.prompt_cache_topology().clone()
    }

    pub(super) fn execution_plan(&self) -> PartitionedExecutionPlan {
        self.facts.execution_plan().clone()
    }

    pub(super) fn execution_residency(&self) -> eredu_runtime::ExecutionResidency {
        self.facts.text().residency().execution_residency()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare<G, W>(
        &mut self,
        input: PartitionedSessionFactoryInput<A, G, W>,
        source: Option<(A, LocalModelLayout)>,
        layout: LocalModelLayout,
        selected: &SelectedReplicatedTextRealization,
        addressable_parameters: &[String],
        groups: (
            Option<CollectiveGroupId>,
            CollectiveGroupId,
            Option<CollectiveGroupId>,
        ),
        context: &Stream,
    ) -> Result<BoundPartitionRank<A, G, W, S, MlxArchitectureLayerwisePolicy<A, S>>, Error> {
        let (source, source_layout) = source.map_or((None, None), |(architecture, layout)| {
            (Some(architecture), Some(layout))
        });
        let prepared = eredu_runtime::prepare_default_partitioned_runtime(
            input,
            source,
            layout,
            source_layout,
            selected,
            self.facts.prompt_cache_topology(),
            self.facts.unit_scope(),
            addressable_parameters,
            &mut self.mechanisms,
            context,
        )?;
        let (architecture, partition, manifest, policy, bounded_policy, state) =
            prepared.into_parts();
        let distributed = self.distributed.take().ok_or_else(|| {
            Error::Parallel("partition communication was already consumed".into())
        })?;
        let (tensor_group, sampling_group, row_status_group) = groups;
        let row_status = row_status_group
            .map(|id| {
                distributed
                    .selected_group(id)
                    .cloned()
                    .ok_or_else(|| Error::Parallel("selected row status group is not bound".into()))
            })
            .transpose()?;
        let (communication, parallel, sampling, communication_executor) =
            distributed.into_partition_communication(manifest, tensor_group, sampling_group)?;
        self.authority = Some(communication.authority());
        self.sampling = Some(sampling);
        Ok(BoundPartitionRank {
            architecture,
            partition,
            policy,
            bounded_policy,
            state,
            communication,
            parallel,
            communication_executor,
            row_status,
        })
    }

    pub(super) fn finish<R, D>(
        self,
        binding: PreparedPartitionedSessionRuntime<R, S>,
        strategy: D,
    ) -> Result<BoundPartitionSession<'a, A, S, D>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
                Runtime = R,
            >,
    {
        let (text, _, _, publication) = self.facts.into_parts();
        let session = eredu_runtime::construct_replicated_text_session_with_runtime(
            binding,
            self.mechanisms,
            strategy,
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        Ok(BoundPartitionSession {
            session,
            text,
            publication,
            sampling: self.sampling,
            authority: self.authority,
            stream: self.stream,
        })
    }
}
