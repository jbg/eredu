//! Native cache, checkpoint and completion mechanisms for retained prediction.

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_nn::{NeuralBackend, Tensor};

use super::ExactParameterMaterializer;
use crate::BackendFailure;

/// Materializes the exact parameter and cache storage requested by the caller.
pub trait NativePredictionMaterialization<B>: Sized
where
    B: NeuralBackend,
{
    /// Backend materialization failure.
    type Error: std::error::Error + Send + Sync + From<BackendFailure> + 'static;
    /// Supplies native parameter metadata and exact encoders.
    type Parameters: ExactParameterMaterializer<B>;
    /// Native storage operations for a portable pooling-attention layer.
    type PoolingStorage: crate::cache::mechanisms::PoolingCacheStorage<
        Value = B::Tensor,
        Queue = <B::Tensor as Tensor>::Context,
    >;
    /// Native tensor storage for sequential compressed prediction.
    type SequentialStorage: crate::cache::mechanisms::CompressedCacheStorage<
        Value = B::Tensor,
        Queue = <B::Tensor as Tensor>::Context,
    >;
    /// Native storage used by the predictor's portable model state.
    type ModelCacheStorage: crate::cache::mechanisms::CacheTensorStorage<
        Value = B::Tensor,
        Queue = <B::Tensor as Tensor>::Context,
    >;
    /// Borrowed resources used during one materialization pass.
    type Context<'a>;

    /// Binds native resources to the exact source admitted by portable preparation.
    fn bind_source<'a>(
        source: eredu_checkpoint::store::SharedCheckpointSource,
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
    ) -> Self::Context<'a>;

    /// Parameter validation and publication for the exact loaned tensor handles.
    type ModuleParameters: crate::checkpoint_mechanisms::CheckpointParameterBackend<
        Parameter = B::Tensor,
    >;
    /// Native storage and transfer mechanisms for the selected checkpoint unit.
    type ResidencyMechanisms: crate::residency::ResidencyStorageMechanisms + Clone;
    /// Native transfer and completion authority for one invocation.
    type ModuleScope;

    /// Native library queue retained with an installed parameter module.
    type ModuleContext: Clone + std::borrow::Borrow<<B::Tensor as Tensor>::Context>;

    /// Retains the selected execution queue.
    fn retain_module_context(context: &<B::Tensor as Tensor>::Context) -> Self::ModuleContext;

    /// Orders and retains an already acquired parameter transfer.
    fn begin_module(
        transfer: <Self::ResidencyMechanisms as crate::residency::ResidencyStorageMechanisms>::Transfer,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::ModuleScope, Self::Error>;

    /// Retains the exact immutable values protected by the module's transfer.
    fn module_weights(
        scope: &mut Self::ModuleScope,
    ) -> Result<Vec<(String, <Self::ModuleParameters as crate::checkpoint_mechanisms::CheckpointParameterBackend>::MaterializedWeight)>, Self::Error>;

    /// Settles the native scope, including roots changed before an equation failure.
    fn finish_module(scope: Self::ModuleScope, roots: Vec<B::Tensor>) -> Result<(), Self::Error>;

    /// Settles the exact retained prediction state and output dependencies before
    /// observation publication. Native implementations include deferred validation
    /// dependencies and retain the original error source on failure.
    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), crate::BackendFailure>
    where
        B::Tensor: 'a;

    /// Provides the exact retained source and execution context for this pass.
    fn parameter_source<'a>(
        context: &'a Self::Context<'_>,
    ) -> (
        &'a SharedCheckpointSource,
        &'a <B::Tensor as Tensor>::Context,
    );
}
