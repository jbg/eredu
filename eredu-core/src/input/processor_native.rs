//! Native tensor allocation, completed transfers, and inspection for prepared input.

use super::PreparedInputInspector;

/// Optional primitive failure distinguishing capability denial from execution failure.
#[derive(Debug, thiserror::Error)]
pub enum OptionalProcessorMechanism<E: std::fmt::Display> {
    /// The backend did not implement this additive primitive.
    #[error("processor mechanism {0} is unavailable")]
    Unavailable(&'static str),
    /// The implemented primitive failed.
    #[error("processor mechanism failed: {0}")]
    Backend(E),
}

/// Backend-native tensor mechanisms used after portable processing succeeds.
pub trait ProcessorMechanisms {
    /// Native tensor handle.
    type Tensor;
    /// Mechanism failure.
    type Error: std::fmt::Display;

    /// Inspects native tensors after binding and observation.
    fn inspector(&self) -> &dyn PreparedInputInspector<Self::Tensor>;

    // Constructors must own copied inputs and retain asynchronous resources
    // through completion, including on failure.
    /// Constructs a native unsigned token tensor.
    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native floating-point tensor.
    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native signed metadata tensor.
    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native Boolean metadata tensor when implemented.
    fn tensor_bool(
        &mut self,
        _values: &[bool],
        _shape: &[usize],
    ) -> Result<Self::Tensor, OptionalProcessorMechanism<Self::Error>> {
        Err(OptionalProcessorMechanism::Unavailable("Boolean tensor"))
    }
}

/// Native tensor construction and recovery retained during portable processing.
pub trait ProcessorPreparationScope:
    ProcessorMechanisms<Error: std::error::Error + 'static> + Sized
{
    /// Establishes completion or retains unresolved native work for recovery.
    fn finish(self) -> Result<(), Self::Error>;
}

/// Native input lowering and inspection for an already selected portable processor.
pub trait MediaInputLowering<T> {
    /// Native tensor created by the selected processor.
    type ProcessorTensor: Clone;
    /// Native error reported by processor tensor mechanisms.
    type ProcessorError: std::error::Error + Send + Sync + From<crate::BackendFailure> + 'static;
    /// Preparation lifetime tracking the native allocations and submissions.
    type Scope: ProcessorPreparationScope<
        Tensor = Self::ProcessorTensor,
        Error = Self::ProcessorError,
    >;

    /// Opens native recovery before allocating tensors.
    fn begin(&self) -> Result<Self::Scope, Self::ProcessorError>;
    /// Borrows native metadata during portable prompt admission.
    fn inspector(&self) -> &impl PreparedInputInspector<T>;
    /// Converts one prepared native tensor to the graph tensor representation.
    fn convert_tensor(&self, tensor: &Self::ProcessorTensor) -> T;
}
