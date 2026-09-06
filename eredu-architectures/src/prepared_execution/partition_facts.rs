//! Final partition metadata projected once from the retained architecture admission.

use eredu_core::{cache::PromptCacheTopology, CollectiveGroupId};
use eredu_runtime::{
    ArchitectureParameters, LayerWeightResidency, PartitionOutputAuthority,
    PartitionedExecutionPlan, PipelineActivationDtype,
};

use super::PreparedTextSessionFacts;
use crate::partitioned_execution::{
    PreparedPartitionedAdmission, PreparedPartitionedArchitecture,
    PreparedRoutedPartitionedArchitecture,
};

/// Exact publication, prompt-cache, and immutable session facts for a selected partition.
pub struct PreparedPartitionSessionFacts {
    text: PreparedTextSessionFacts,
    topology: PromptCacheTopology,
    execution: PartitionedExecutionPlan,
    publication: PartitionOutputAuthority,
    tensor_group: Option<CollectiveGroupId>,
    session_group: CollectiveGroupId,
    activation_dtype: PipelineActivationDtype,
}

impl PreparedPartitionSessionFacts {
    /// Optional tensor group already selected by architecture placement.
    pub const fn tensor_group(&self) -> Option<CollectiveGroupId> {
        self.tensor_group
    }

    /// Exact tensor group required by the architecture's direct-partition route.
    pub fn required_tensor_group(&self) -> Result<CollectiveGroupId, String> {
        self.tensor_group
            .ok_or_else(|| "selected direct partition has no tensor group".into())
    }

    /// Exact opaque session group selected for communication and publication.
    pub const fn session_group(&self) -> CollectiveGroupId {
        self.session_group
    }

    /// Selected boundary activation dtype, without native dtype interpretation.
    pub const fn activation_dtype(&self) -> PipelineActivationDtype {
        self.activation_dtype
    }

    /// Immutable text-session facts retained across local materialization.
    pub const fn text(&self) -> &PreparedTextSessionFacts {
        &self.text
    }

    /// Exact prompt-cache topology already resolved from the retained admission.
    pub const fn prompt_cache_topology(&self) -> &PromptCacheTopology {
        &self.topology
    }

    /// Architecture-selected complete execution plan.
    pub const fn execution_plan(&self) -> &PartitionedExecutionPlan {
        &self.execution
    }

    /// Exact output ownership in world and selected group-local coordinates.
    pub const fn publication_authority(&self) -> PartitionOutputAuthority {
        self.publication
    }

    /// Consumes the facts for native runtime assembly and final adaptation.
    pub fn into_parts(
        self,
    ) -> (
        PreparedTextSessionFacts,
        PromptCacheTopology,
        PartitionedExecutionPlan,
        PartitionOutputAuthority,
    ) {
        (self.text, self.topology, self.execution, self.publication)
    }

    fn from_admission<B, A, R, Q, G, W>(
        prepared: &PreparedPartitionedAdmission<A, R, Q, G, W>,
        execution: PartitionedExecutionPlan,
        capability: &crate::capability::CapabilityEstimate,
        model_type: &str,
        residency: LayerWeightResidency,
    ) -> Result<Self, String>
    where
        B: eredu_nn::NeuralBackend,
        A: ArchitectureParameters<B>,
        A::DefinitionError: std::fmt::Display,
    {
        let selected = prepared.selected();
        let publication = execution
            .publication_authority(selected.communication())
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "selected partition has no output publication authority".to_owned())?;
        let topology = selected.prompt_cache_topology()?;
        let identity = selected
            .partition()
            .prompt_cache_identity::<B, A>(prepared.architecture(), topology.clone())
            .map_err(|error| error.to_string())?;
        let session_group = selected
            .session_group()
            .ok_or_else(|| "selected partition has no session group".to_owned())?;
        Ok(Self {
            text: PreparedTextSessionFacts::from_parts(
                identity,
                capability.clone(),
                model_type.to_owned(),
                residency,
            ),
            topology,
            execution,
            publication,
            tensor_group: selected.tensor_group(),
            session_group,
            activation_dtype: selected.activation_dtype(),
        })
    }
}

impl<B, A, G, W> PreparedPartitionedArchitecture<B, A, G, W>
where
    B: eredu_nn::NeuralBackend,
    A: ArchitectureParameters<B>,
    A::DefinitionError: std::fmt::Display,
{
    /// Projects publication/cache/session facts through this exact dense partition selection.
    pub fn session_facts(&self) -> Result<PreparedPartitionSessionFacts, String> {
        let selected = self.prepared().selected();
        let execution = if selected.topology().pipeline_parallel_size() > 1 {
            selected.pipeline_execution_plan()?
        } else {
            selected.direct_execution_plan()?
        };
        PreparedPartitionSessionFacts::from_admission::<B, _, _, _, _, _>(
            self.prepared(),
            execution,
            self.capability_estimate(),
            self.effective_model_type(),
            selected.base().residency(),
        )
    }
}

impl<B, A, G, W, E> PreparedRoutedPartitionedArchitecture<B, A, G, W, E>
where
    B: eredu_nn::GroupedNeuralBackend,
    A: ArchitectureParameters<B>,
    A::DefinitionError: std::fmt::Display,
{
    /// Projects publication/cache/session facts without reselecting the routed execution recipe.
    pub fn session_facts(&self) -> Result<PreparedPartitionSessionFacts, String> {
        PreparedPartitionSessionFacts::from_admission::<B, _, _, _, _, _>(
            self.prepared(),
            self.execution_handoff().execution_plan().clone(),
            self.capability_estimate(),
            self.effective_model_type(),
            self.prepared().selected().base().text().residency(),
        )
    }
}

impl<A, G, W> crate::composite_partitioned::PreparedCompositePartition<A, G, W> {
    /// Projects publication/cache/session facts from the complete retained composite admission.
    pub fn session_facts<B>(&self) -> Result<PreparedPartitionSessionFacts, String>
    where
        B: eredu_nn::NeuralBackend,
        A: ArchitectureParameters<B>,
        A::DefinitionError: std::fmt::Display,
    {
        PreparedPartitionSessionFacts::from_admission::<B, _, _, _, _, _>(
            self.prepared(),
            self.execution_plan()?,
            self.capability_estimate(),
            self.effective_model_type(),
            self.prepared().selected().base().execution().residency(),
        )
    }
}
