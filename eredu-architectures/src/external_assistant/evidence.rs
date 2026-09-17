//! Per-output evidence of the actual completed external invocation.
use crate::{composite_execution::ExternalPredictionTargetCapture,
    speculative_execution::PreparedEmbeddedEvidence};

/// Target output and its exact completed source. Ordinary execution has no
/// original source witness; the same raw values preserve ordinary semantics.
pub struct ExternalTargetResult<T> {
    /// Actual selected target scores.
    pub logits: T,
    /// Actual architecture-declared hidden/shared/ordered target values.
    pub capture: ExternalPredictionTargetCapture<T>,
    /// The source produced by completion of these particular native values.
    pub evidence: Option<PreparedEmbeddedEvidence>,
}
impl<T> ExternalTargetResult<T> {
    /// Moves an ordinary result without allocating another owner.
    pub fn ordinary(logits: T, capture: ExternalPredictionTargetCapture<T>) -> Self {
        Self { logits, capture, evidence: None }
    }
}
/// Existing assistant result plus its actual completion/source witness.
pub struct ExternalOperationResult<T> {
    /// Unchanged result of the architecture operation.
    pub output: T,
    /// Exact completed inventory, independently retained by state/output owners.
    pub evidence: Option<PreparedEmbeddedEvidence>,
}
impl<T> ExternalOperationResult<T> {
    /// Moves the ordinary result without inventing completion evidence.
    pub fn ordinary(output: T) -> Self { Self { output, evidence: None } }
}
