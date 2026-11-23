//! Native mechanisms consumed by portable realtime startup.

use eredu_core::sampling::DenseSamplingBackend;

use crate::state::{HybridCacheStorage, HybridState};
use eredu_core::BackendFailure;
use eredu_core::{
    backend::LocalComponentStorage, realtime_mechanisms::RealtimeGraphResources,
    sampling::SamplingBackend,
};
use eredu_nn::{AttentionCache, NeuralBackend, Tensor};

use crate::{
    execution_control::SnapshotRetainedGrowth, LayerRuntimeState, ParameterUnitMechanisms,
    RuntimeStateComponents, StateLayout,
};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Portable state ownership and component binding for a realtime layered graph.
pub trait RealtimeGraphMechanisms<B: NeuralBackend>: Sized {
    /// Creates portable state from the exact selected layout.
    fn realize_state(&self, layout: &StateLayout) -> Result<Self::State, BackendFailure>;
    /// State used for every declared neural layout.
    type State: LayerRuntimeState<
        B,
        LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    >;
    /// Materialization mechanisms shared with other model drivers.
    type Components: ParameterUnitMechanisms<B>;
    /// Borrows the component binder.
    fn components(&self) -> &Self::Components;

    /// Native sampling failure retained as a source.
    type SamplingError: std::error::Error + Send + Sync + 'static;
    /// Existing sampling mechanism on the same context.
    type Sampling: DenseSamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = Context<B>,
        Error = Self::SamplingError,
    >;

    /// Selected execution context.
    fn context(&self) -> &Context<B>;
    /// Branches sequential state, isolating all mutable storage.
    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure>;
    /// Collects state roots that must complete before canonical publication.
    fn retained_values(state: &Self::State) -> Vec<&B::Tensor>;

    /// Opens native recovery before portable streamed input work.
    fn open_input_recovery(
        context: &Context<B>,
    ) -> Result<impl Sized + 'static + use<Self, B>, BackendFailure>;

    /// Keeps branch accounting alive through any native work started in this scope.
    fn hold_fork_retention(
        context: &Context<B>,
        retention: SnapshotRetainedGrowth,
    ) -> Result<impl Sized + 'static + use<Self, B>, BackendFailure>;
}

/// Portable cache owner used by categorical startup.
pub trait CategoricalRealtimeState: Sized {
    /// Constructs resident state for the exact selected model layout.
    fn resident_categorical_state(layout: &StateLayout) -> Result<Self, BackendFailure>;
}

impl<C> CategoricalRealtimeState for crate::state::KeyValueState<C>
where
    C: eredu_core::cache::mechanisms::ResidentKeyValueStorage
        + eredu_core::cache::mechanisms::PagedKeyValueStorage,
{
    fn resident_categorical_state(layout: &StateLayout) -> Result<Self, BackendFailure> {
        Self::device(layout.clone()).map_err(BackendFailure::from_error)
    }
}

impl<B, M> RealtimeGraphMechanisms<B> for M
where
    B: LocalComponentStorage<CacheStorage: HybridCacheStorage>,
    M: RealtimeGraphResources<Backend = B> + ParameterUnitMechanisms<B>,
    <M::Sampling as SamplingBackend>::Error: 'static,
{
    type State = HybridState<B::CacheStorage>;
    type Components = Self;
    type SamplingError = <M::Sampling as SamplingBackend>::Error;
    type Sampling = M::Sampling;

    fn realize_state(&self, layout: &StateLayout) -> Result<Self::State, BackendFailure> {
        HybridState::device(layout.clone(), &[]).map_err(BackendFailure::from_error)
    }

    fn components(&self) -> &Self {
        self
    }

    fn context(&self) -> &Context<B> {
        self.graph_context()
    }

    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure> {
        state.deep_clone_state().map_err(BackendFailure::from_error)
    }

    fn retained_values(state: &Self::State) -> Vec<&B::Tensor> {
        state
            .layers()
            .iter()
            .flat_map(|layer| layer.retained_values())
            .collect()
    }

    fn open_input_recovery(
        context: &Context<B>,
    ) -> Result<impl Sized + 'static + use<B, M>, BackendFailure> {
        M::open_input_recovery(context)
    }

    fn hold_fork_retention(
        context: &Context<B>,
        retention: SnapshotRetainedGrowth,
    ) -> Result<impl Sized + 'static + use<B, M>, BackendFailure> {
        M::hold_fork_retention(context, retention.authority())
    }
}
