//! Native storage, placement, and submission for prepared parameter units.

use super::{bounded_quantization::BoundedQuantizationBackend, CheckpointParameterBackend};
use crate::{
    residency::{
        observations::DenseResidencyMechanisms, parameter_binding::WeightBinding,
        ResidencyLeaseStorage, ResidencyStorage, ResidencyStorageMechanisms, UnitRetirement,
    },
    resources::ResourceIdentity,
    BackendFailure, SubmissionBackend,
};
use eredu_nn::NeuralBackend;

/// Native recovery for preparation driven by the caller.
pub trait ComponentPreparationScope: Sized {
    /// Seals native work and retains pending resources, preserving any preparation error.
    fn finish<T, E>(self, result: Result<T, E>) -> Result<T, E>
    where
        E: From<BackendFailure>;
}

/// Provides native context and recovery for retained component preparation.
pub trait ComponentPreparation {
    /// Execution context used by the bound components.
    type Context: ?Sized;
    /// Exact native scope covering this preparation.
    type Scope: ComponentPreparationScope;

    /// Provides the execution context without transferring control of preparation.
    fn component_context(&self) -> &Self::Context;

    /// Supplies optional observation and initialization of native allocator defaults.
    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn crate::capability::NativeAllocatorCacheDefaults> {
        None
    }

    /// Opens native retirement before the caller constructs an aggregate.
    fn begin_component_preparation(&self) -> Result<Self::Scope, BackendFailure>;
}

/// Retained identity of immutable parameters on an admitted execution context.
pub trait ImmutableUnitStorage<C: ?Sized> {
    /// Confirms the certificate is valid on this exact context.
    fn identity(&self, context: &C) -> Option<&ResourceIdentity>;
}

/// Native allocation, transfer and population for admitted unit storage.
pub trait ParameterStorageBinding<B: NeuralBackend> {
    /// Native operations that retain storage through completion.
    type UnitOperations: crate::residency::NativeBoundedUnits<
            Tensor = B::Tensor,
            Context = UnitContext<B>,
            Transfer = UnitTransfer<B, Self>,
        > + 'static;
    /// Native certificate for a resident parameter set on its execution queue.
    type ImmutableParameters: ImmutableUnitStorage<UnitContext<B>> + 'static;
    /// Native storage, transfer, and observation mechanisms for the portable residency owner.
    type ResidencyMechanisms: DenseResidencyMechanisms<Storage: Send + 'static>
        + Clone
        + Send
        + Sync
        + 'static;

    /// Opens native quantization mechanisms on the unit execution context.
    fn unit_quantizer(&self) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure>;

    /// Exact host transfer allocation capacity for the proposed bindings.
    fn unit_host_capacity(&self, bindings: &[WeightBinding]) -> Result<u64, BackendFailure>;

    /// Provides native streams at the selected transfer placement.
    fn unit_residency_context(
        &self,
        distinct_transfer_stream: bool,
    ) -> Result<<Self::ResidencyMechanisms as ResidencyStorageMechanisms>::Context, BackendFailure>;

    /// Native handle validation and publication primitives.
    type Parameters: CheckpointParameterBackend<
        Parameter = B::Tensor,
        MaterializedWeight = B::Tensor,
    >;

    /// Retains a native device value as a parameter without copying its storage.
    fn retain_parameter(
        value: &<<<Self::ResidencyMechanisms as ResidencyStorageMechanisms>::Storage as ResidencyStorage>::Copy as ResidencyLeaseStorage>::DeviceValue,
    ) -> B::Tensor;

    /// Supplies native queue ordering and completion operations.
    fn unit_operations(&self) -> Self::UnitOperations;
}

type UnitContext<B> = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context;
type UnitTransfer<B, M> =
    <<M as ParameterStorageBinding<B>>::ResidencyMechanisms as ResidencyStorageMechanisms>::Transfer;

/// Native materialization and submission for independently addressable units.
pub trait NativeParameterUnitBinding<B: NeuralBackend>:
    ParameterStorageBinding<
    B,
    UnitOperations: crate::residency::NativeResidentUnits<B, UnitTransfer<B, Self>>
                        + UnitRetirement<Error = Self::UnitError>
                        + crate::residency::NativeBoundedUnits<
        Tensor = B::Tensor,
        Context = UnitContext<B>,
        Transfer = UnitTransfer<B, Self>,
        Error = Self::UnitError,
    >,
>
{
    /// Native recovery for the binding operation, including unit construction.
    type Scope: ComponentPreparationScope;
    /// Failure at the native execution or observation boundary.
    type UnitError: std::error::Error + Send + Sync + From<BackendFailure> + 'static;

    /// Starts tracking work from storage preparation through final materialization.
    fn begin_unit_binding(&self) -> Result<Self::Scope, BackendFailure>;

    /// Supplies the execution queue for resident transfer ordering.
    fn unit_executor(&self) -> &UnitContext<B>;

    /// Certifies immutable parameter handles on the exact resident queue.
    fn resident_parameter_certificate(&self) -> Result<Self::ImmutableParameters, BackendFailure>;
}

/// Native storage, executor and retirement mechanisms for local components.
pub trait LocalComponentBinding<B: SubmissionBackend>: NativeParameterUnitBinding<B> {
    /// Retains an executor for all invocations of the bound local component.
    fn local_executor(&self) -> B::OwnedExecutor;
}
