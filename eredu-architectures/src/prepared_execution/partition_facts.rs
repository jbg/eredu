//! Final partition metadata projected once from the retained architecture admission.

use eredu_runtime::{
    ArchitectureParameters, LayerWeightResidency, PartitionedExecutionPlan, PartitionedUnitScope,
};

use super::PreparedTextSessionFacts;
use crate::partitioned_execution::{AdmittedRoutedPartition, PreparedPartitionedAdmission};

pub use eredu_runtime::executable_binding::PreparedPartitionSessionFacts;

fn from_admission<B, A, R, Q, G, W>(
    prepared: &PreparedPartitionedAdmission<A, R, Q, G, W>,
    execution: PartitionedExecutionPlan,
    capability: &crate::capability::CapabilityEstimate,
    model_type: &str,
    residency: LayerWeightResidency,
    unit_scope: PartitionedUnitScope,
) -> Result<PreparedPartitionSessionFacts, String>
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
    let session_group = selected
        .session_group()
        .ok_or_else(|| "selected partition has no session group".to_owned())?;
    Ok(PreparedPartitionSessionFacts::new(
        PreparedTextSessionFacts::from_parts(
            capability
                .clone()
                .for_selected_state(selected.partition().state().map(|state| state.layout()))
                .map_err(|error| error.to_string())?,
            model_type.to_owned(),
            residency,
        ),
        topology,
        execution,
        publication,
        selected.tensor_group(),
        session_group,
        selected.activation_dtype(),
        unit_scope,
    ))
}

pub(crate) fn dense_session_facts<B, A, G, W>(
    prepared: &PreparedPartitionedAdmission<
        A,
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
        G,
        W,
    >,
    capability: &crate::capability::CapabilityEstimate,
    model_type: &str,
    pipeline: bool,
) -> Result<PreparedPartitionSessionFacts, String>
where
    B: eredu_nn::NeuralBackend,
    A: ArchitectureParameters<B>,
    A::DefinitionError: std::fmt::Display,
{
    let selected = prepared.selected();
    let (execution, unit_scope) = if pipeline {
        (
            selected.pipeline_execution_plan()?,
            PartitionedUnitScope::Owned,
        )
    } else {
        (selected.direct_execution_plan()?, PartitionedUnitScope::All)
    };
    from_admission::<B, _, _, _, _, _>(
        prepared,
        execution,
        capability,
        model_type,
        selected.base().residency(),
        unit_scope,
    )
}

impl<B, A, G, W> AdmittedRoutedPartition<B, A, G, W>
where
    B: eredu_nn::GroupedNeuralBackend,
    A: ArchitectureParameters<B>,
    A::DefinitionError: std::fmt::Display,
{
    /// Projects publication/cache/session facts without reselecting the routed execution recipe.
    pub fn session_facts(&self) -> Result<PreparedPartitionSessionFacts, String> {
        from_admission::<B, _, _, _, _, _>(
            self.prepared(),
            self.execution_handoff().execution_plan().clone(),
            self.capability_estimate(),
            self.effective_model_type(),
            self.prepared().selected().base().text().residency(),
            PartitionedUnitScope::Owned,
        )
    }
}

impl<A, G, W> crate::composite_partitioned::AdmittedCompositePartition<A, G, W> {
    /// Projects publication/cache/session facts from the complete retained composite admission.
    pub fn session_facts<B>(&self) -> Result<PreparedPartitionSessionFacts, String>
    where
        B: eredu_nn::NeuralBackend,
        A: ArchitectureParameters<B>,
        A::DefinitionError: std::fmt::Display,
    {
        from_admission::<B, _, _, _, _, _>(
            self.prepared(),
            self.execution_plan()?,
            self.capability_estimate(),
            self.effective_model_type(),
            self.prepared().selected().base().execution().residency(),
            PartitionedUnitScope::Owned,
        )
    }
}
