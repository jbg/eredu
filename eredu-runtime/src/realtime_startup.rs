//! Native mechanisms consumed by portable realtime startup.

use eredu_core::{BackendFailure, Completion};
use eredu_nn::{AttentionCache, NeuralBackend, RetainedTensorRoots, Tensor};

use crate::{
    execution_control::SnapshotRetainedGrowth, DenseSamplingBackend, LayerRuntimeState,
    ParameterUnitMechanisms, RuntimeStateComponents, StateLayout,
};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Realizes an exact selected state plan.
pub trait RealtimeStateMaterializer<P, S> {
    /// Allocates independent state from the retained selection.
    fn realize_state(&self, selected: &P) -> Result<S, BackendFailure>;
}

/// Finite copy limits for a joint state transaction.
#[derive(Debug, Clone, Copy)]
pub struct RealtimeCopyLimits {
    /// Maximum storage and temporary copy bytes for one branch.
    pub max_branch_bytes: u64,
    /// Cumulative copy allowance shared by isolated descendants.
    pub max_total_copy_bytes: u64,
}

/// Groups independently realized state under one copy budget and transaction.
pub trait RealtimeStateGroup<L, R> {
    /// Combined state owner.
    type State;

    /// Adopts both states into one transactional owner.
    fn group_state(
        &self,
        left: L,
        right: R,
        limits: RealtimeCopyLimits,
    ) -> Result<Self::State, BackendFailure>;
}

/// Disjoint access to two parts of one unpublished state transaction.
pub trait RealtimeJointStateParts {
    /// First state part.
    type Left: ?Sized;
    /// Second state part.
    type Right: ?Sized;

    /// Borrows the parts while the branch keeps transaction ownership.
    fn parts_mut(&mut self) -> (&mut Self::Left, &mut Self::Right);
}

/// Native RNG and exact completion types for realtime generation.
pub trait RealtimeStartupMechanisms {
    /// Explicit random state adopted by the generation transaction.
    type Random;
    /// Exact completion attached to candidate execution.
    type Completion: Completion;

    /// Realizes an explicitly selected random seed.
    fn realize_random(&self, seed: u64) -> Result<Self::Random, BackendFailure>;
}

/// Runs startup neural work and completes its retained outputs before returning.
pub trait RealtimeStartupPreparation<C, T: Tensor> {
    /// Enters native recovery and completes an architecture-owned operation.
    fn prepare_startup<O: RetainedTensorRoots<T>>(
        &self,
        operation: impl FnOnce(&mut C, &T::Context) -> Result<O, BackendFailure>,
    ) -> Result<O, BackendFailure>;
}

/// Opens the transport for a retained pipeline route and reports its native world identity.
pub trait PipelineTransportProvider {
    /// Native transport passed to the generic component binder.
    type Transport;

    /// Opens the selected native transport alongside its process identity.
    fn open(&self) -> Result<(crate::PlacementRank, Self::Transport), BackendFailure>;
}

/// Native state ownership and component binding for a realtime layered graph.
pub trait RealtimeGraphMechanisms<B: NeuralBackend>:
    Sized + RealtimeStateMaterializer<StateLayout, Self::State>
{
    /// State used for every declared neural layout.
    type State: LayerRuntimeState<
        B,
        LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    >;
    /// Native materialization mechanisms shared with other model adapters.
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

    /// Retains native work started by one streamed input operation through recovery.
    fn with_input_recovery<T, E>(operation: impl FnOnce() -> Result<T, E>) -> Result<T, E>
    where
        E: From<BackendFailure>;

    /// Holds retained branch accounting while native work can outlive an advance.
    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure>;
}
