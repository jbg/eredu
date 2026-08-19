//! Narrow capability contracts implemented by execution backends.

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointLease, CheckpointSource},
};
use eredu_core::Completion;
use eredu_nn::NeuralBackend;

/// Submits backend-native work and retains values through exact completion.
pub trait SubmissionBackend: NeuralBackend {
    /// Backend executor, queue, stream, or equivalent submission context.
    type Executor: ?Sized;
    /// Exact completion object for one submission.
    type Completion: Completion;

    /// Retains an owned value until `completion` has completed exactly.
    fn retain_until_complete<T: Send + 'static>(
        executor: &Self::Executor,
        completion: &Self::Completion,
        value: T,
    ) -> Result<(), <Self::Completion as Completion>::Error>;
}

/// Materializes and binds checkpoint data to backend-native parameter slots.
pub trait ParameterBackend: NeuralBackend {
    /// One backend-native parameter slot.
    type Parameter;
    /// Materialized backend-native checkpoint weight.
    type MaterializedWeight;
    /// Backend context used only while realizing checkpoint parameters.
    type MaterializationContext: ?Sized;
    /// In-flight guard retaining encoded sources through exact realization completion.
    type Materialization;
    /// Backend-specific loading failure.
    type ParameterError: std::error::Error + Send + Sync + 'static;

    /// Lowers one format-preserving encoded lease into a native weight.
    fn materialize(
        lease: CheckpointLease,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Lowers a validated neutral recipe directly into a native weight.
    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Borrows the native weight retained by an in-flight materialization.
    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight;

    /// Binds one materialized weight to its destination parameter.
    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError>;
}

/// Promotes and demotes backend-native storage without changing its semantics.
pub trait TransferBackend: SubmissionBackend + ParameterBackend {
    /// Backend-owned host representation.
    type HostBuffer;
    /// In-flight transfer guard retaining all source and destination storage.
    type Transfer: Completion<Error = Self::TransferError>;
    /// Backend-specific transfer failure.
    type TransferError: std::error::Error + Send + Sync + 'static;

    /// Promotes host storage into a materialized execution weight.
    fn promote(
        executor: &Self::Executor,
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Self::Transfer), Self::TransferError>;

    /// Demotes a materialized execution weight into backend-owned host storage.
    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError>;
}

/// Collective operations available to distributed runtime policies.
pub trait CollectiveBackend: SubmissionBackend {
    /// Backend-native collective group.
    type Group: ?Sized;
    /// Backend-specific collective failure.
    type CollectiveError: std::error::Error + Send + Sync + 'static;

    /// Reduces a tensor across the selected group.
    fn all_reduce(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Gathers a tensor across the selected group.
    fn all_gather(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Exchanges tensor partitions across the selected group.
    fn all_to_all(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;
}
