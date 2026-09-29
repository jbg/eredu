//! Typed prediction owners whose routed providers are shared with target preparation.
use super::*;

/// Architecture-owned preparation consumed without family dispatch by a backend.
/// Immutable modules and state are bound before the target installs their residency;
/// separately owned providers are then moved into the executor without cloning them.
pub trait PreparedRoutedPrediction<B, A>: Sized
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Materialized prediction strategy, parameterized by its native module owners and provider.
    type Executor<M, P>: MaterializedPredictionExecutor<A, B, M> + 'static
    where
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B> + 'static,
        P::Error: std::fmt::Display;

    /// Exact retained source for prediction module materialization.
    fn source(&self) -> &eredu_checkpoint::store::SharedCheckpointSource;
    /// Bank identities to move out of the joint provider collection.
    fn banks(&self) -> Vec<eredu_runtime::RoutedBankId>;
    /// Binds immutable owners and their independently selected mutable state.
    fn materialize<M>(
        self,
        context: &mut M::Context<'_>,
        realize_state: impl FnOnce(
            &mut M::Context<'_>,
            &eredu_runtime::SelectedStateRealization,
        ) -> Result<M::ModelState, M::Error>,
    ) -> Result<
        Self::Executor<M, eredu_runtime::ResidentExpertProvider>,
        PredictionConstructionError<M::Error>,
    >
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>;

    /// Replaces the construction-time resident provider with the selected moved banks.
    fn with_provider<M, P>(
        executor: Self::Executor<M, eredu_runtime::ResidentExpertProvider>,
        provider: P,
    ) -> Self::Executor<M, P>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        M: PredictionExtensionMaterializer<B> + 'static,
        <M::ModelState as PredictionModelState<B>>::LayerState:
            eredu_runtime::RuntimeAppendStreams<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B> + 'static,
        P::Error: std::fmt::Display;
}
