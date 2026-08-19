//! Backend-neutral model execution contracts and algorithms.
//!
//! This crate orchestrates opaque backend-native values. It deliberately has
//! no dependency on an architecture implementation or execution backend.

#![warn(missing_docs)]

/// Backend execution, parameter, transfer, and collective capabilities.
pub mod backend;
/// Portable execution-group topology and scheduling state.
pub mod execution;
/// Backend-neutral immutable-weight residency declarations and orchestration.
pub mod residency;

pub use backend::{CollectiveBackend, ParameterBackend, SubmissionBackend, TransferBackend};
pub use execution::{
    ExecutionGraph, ExecutionGraphError, ExecutionGroupId, ExecutionGroupReadySet,
    ExecutionGroupSpec, ReadyGroupState,
};
pub use residency::{OffloadUnit, ResidencyDeclarationError, WeightBinding};

/// Inspectable architecture/runtime topology without backend-native values.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RuntimeMetadata {
    model_identity: String,
    execution_graph: ExecutionGraph,
    execution_unit_count: usize,
}

impl RuntimeMetadata {
    /// Creates validated runtime metadata for one concrete architecture instance.
    pub fn new(
        model_identity: impl Into<String>,
        execution_graph: ExecutionGraph,
        execution_unit_count: usize,
    ) -> Result<Self, RuntimeMetadataError> {
        let model_identity = model_identity.into();
        if model_identity.trim().is_empty() {
            return Err(RuntimeMetadataError::EmptyModelIdentity);
        }
        if execution_unit_count == 0 {
            return Err(RuntimeMetadataError::EmptyExecution);
        }
        Ok(Self {
            model_identity,
            execution_graph,
            execution_unit_count,
        })
    }

    /// Returns the architecture-provided compatibility identity.
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    /// Returns the validated architecture execution graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }

    /// Returns the total number of ordered execution units across all groups.
    pub const fn execution_unit_count(&self) -> usize {
        self.execution_unit_count
    }
}

/// Invalid backend-neutral runtime metadata.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum RuntimeMetadataError {
    /// The architecture supplied no compatibility identity.
    #[error("runtime model identity must not be empty")]
    EmptyModelIdentity,
    /// The architecture supplied no executable units.
    #[error("runtime must contain at least one execution unit")]
    EmptyExecution,
}

/// Error produced by backend-neutral runtime orchestration.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Invalid execution-group topology or scheduling transition.
    #[error(transparent)]
    ExecutionGraph(#[from] ExecutionGraphError),
    /// Invalid runtime metadata.
    #[error(transparent)]
    Metadata(#[from] RuntimeMetadataError),
    /// A concrete backend capability failed.
    #[error("runtime backend operation failed: {0}")]
    Backend(String),
}
