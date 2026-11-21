//! Exact publication and cache facts handed to the rank binder.

use eredu_core::{cache::PromptCacheTopology, CollectiveGroupId};

use crate::{PartitionOutputAuthority, PartitionedExecutionPlan, PartitionedUnitScope};
use eredu_core::communication::PipelineActivationDtype;

use super::PreparedTextSessionFacts;

/// Exact publication, prompt-cache, and immutable session facts for a selected partition.
#[derive(Clone)]
pub struct PreparedPartitionSessionFacts {
    text: PreparedTextSessionFacts,
    topology: PromptCacheTopology,
    execution: PartitionedExecutionPlan,
    publication: PartitionOutputAuthority,
    tensor_group: Option<CollectiveGroupId>,
    session_group: CollectiveGroupId,
    activation_dtype: PipelineActivationDtype,
    unit_scope: PartitionedUnitScope,
}

impl PreparedPartitionSessionFacts {
    /// Couples the selected publication, communication and cache facts for a bound rank.
    pub fn new(
        text: PreparedTextSessionFacts,
        topology: PromptCacheTopology,
        execution: PartitionedExecutionPlan,
        publication: PartitionOutputAuthority,
        tensor_group: Option<CollectiveGroupId>,
        session_group: CollectiveGroupId,
        activation_dtype: PipelineActivationDtype,
        unit_scope: PartitionedUnitScope,
    ) -> Self {
        Self {
            text,
            topology,
            execution,
            publication,
            tensor_group,
            session_group,
            activation_dtype,
            unit_scope,
        }
    }

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

    /// Selected ownership of rank-local execution units.
    pub const fn unit_scope(&self) -> PartitionedUnitScope {
        self.unit_scope
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
}
