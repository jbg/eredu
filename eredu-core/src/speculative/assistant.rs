//! Exact native tensor operations for externally conditioned prediction.

use crate::{
    backend::NativeTextValues,
    capability::NativeAllocatorObservations,
    capture::SpeculativeCapturePrimitives,
    checkpoint_mechanisms::{CheckpointParameterBackend, ExactParameterMaterializer},
    speculative::SpeculativeQueueMechanisms,
    BackendFailure, BoundedCompletion, SpeculativeTelemetry,
};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::{NeuralBackend, Tensor};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

type Queue<M> = <<M as ExternalAssistantOperators>::Tensor as Tensor>::Context;
type Transport<M> =
    <<M as ExternalAssistantOperators>::Capture as SpeculativeCapturePrimitives>::Transport;

/// Native operations used by the portable external-assistant driver.
pub trait ExternalAssistantOperators: Sized + 'static {
    /// Ordinary target handles.
    type Text: NativeTextValues<
        Tensor = Self::Tensor,
        Neural = Self::NeuralBackend,
        Context = Queue<Self>,
    >;
    /// Native tensor operators and weights.
    type NeuralBackend: eredu_nn::GroupedNeuralBackend<Tensor = Self::Tensor>
        + eredu_nn::DistributedNeuralBackend
        + Clone;
    /// Checkpoint materialization resources.
    type Binder;
    /// Tensor storage used to construct the assistant's resident attention cache.
    type Cache: crate::cache::mechanisms::ResidentKeyValueStorage<Value = Self::Tensor>;
    /// Native capture and communication primitives.
    type Capture: SpeculativeCapturePrimitives<
        Tensor = Self::Tensor,
        InvocationError = Self::Error,
        Queues: SpeculativeQueueMechanisms<Queue = Queue<Self>>,
    >;
    /// Native allocator facts.
    type Memory: NativeAllocatorObservations;
    /// Native tensor handle.
    type Tensor: Tensor + Clone;
    /// Native scores consumed by sampling.
    type Logits: Clone;
    /// Native submission completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Native operation failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Preserves an owned cause at the numerical boundary.
    fn failure(error: impl std::error::Error + Send + Sync + 'static) -> Self::Error;
    /// Reports a rejected geometry or native input.
    fn invalid(message: String) -> Self::Error;
    /// Retains the input representation for an exact target token tensor.
    fn target_input(value: &Self::Tensor) -> <Self::Text as NativeTextValues>::InputTensor;
    /// Returns exact tensor extents.
    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error>;
    /// Returns the width of the sequence axis.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;
    /// Exposes native sampling logits.
    fn into_logits(value: Self::Tensor) -> Self::Logits;
    /// Reads complete backing sizes without evaluating tensors.
    fn tensor_backing_bytes(values: &[&Self::Tensor]) -> Result<Option<u64>, BackendFailure>;
    /// Estimates retained and copied logits storage.
    fn control_logits_estimate(
        logits: &Self::Logits,
    ) -> Option<crate::execution_control::SnapshotEstimate>;
    /// Bounds one copy of the supplied tensor.
    fn control_tensor_bytes(value: &Self::Tensor) -> Option<u64>;

    /// Borrows the native communicator for the exact admitted manifest and group.
    fn bound_assistant_group<'a>(
        transport: &'a Transport<Self>,
        group: crate::CollectiveGroupId,
        manifest: &crate::communication::CommunicationManifest,
    ) -> Result<&'a <Self::NeuralBackend as NeuralBackend>::ParallelContext, Self::Error>;
    /// Reads an exact sequence row.
    fn sequence_row(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Reads the bounded suffix on the selected queue.
    fn sequence_suffix(
        value: &Self::Tensor,
        maximum: i32,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Slices the supplied shared attention tensor.
    fn shared_prefix(
        value: &Self::Tensor,
        length: i32,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Slices the supplied token tensor.
    fn token_prefix(
        value: &Self::Tensor,
        end: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Materializes exact token values, with optional transfer to the nominated queue.
    fn target_tokens(
        tokens: &[u32],
        destination: Option<&Queue<Self>>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Borrows the native value used for queue dependency submission.
    fn tensor_root(value: &Self::Tensor) -> &<<Self::Capture as SpeculativeCapturePrimitives>::Queues as SpeculativeQueueMechanisms>::Root;
    /// Settles a copy on the specified destination.
    fn copy_tensor(
        value: &Self::Tensor,
        destination: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Submits the exact selected tensor roots.
    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a;
}

/// Native operations for binding an admitted assistant.
pub trait ExternalAssistantBinder: Sized {
    /// Neural operations used by portable assistant construction.
    type Backend: CheckpointParameterBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + Clone;
    /// Parameter allocation and transfer mechanisms on the assistant streams.
    type Storage: crate::checkpoint_mechanisms::unit_binding::NativeParameterUnitBinding<
        Self::Backend,
    >;
    /// Retained score rows supplied to the assistant observer.
    type Logits: Clone;
    /// Failure at the assistant execution and observation boundary.
    type ObservationError: 'static;
    /// Binding failure.
    type Error: From<BackendFailure> + std::error::Error + Send + Sync + 'static;

    /// Context used for module construction and execution.
    fn context(&self) -> &Context<Self::Backend>;

    /// Supplies the ordinary parameter storage operations on these streams.
    fn storage(&self) -> Self::Storage;

    /// Supplies exact native parameter metadata and encoding.
    type Parameters: ExactParameterMaterializer<Self::Backend>;

    /// Upper bound for this native recipe's evaluation workspace.
    fn recipe_workspace(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<u64, Self::Error>;

    /// Native read and conversion scope used for fully resident parameters.
    type ParameterReads: crate::checkpoint_mechanisms::ParameterReadScope<
        Value = <Self::Backend as NeuralBackend>::Tensor,
    >;

    /// Opens a read scope on the selected checkpoint and execution contexts.
    fn parameter_reads(&self) -> Self::ParameterReads;
}
