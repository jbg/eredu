//! Portable Audio graph construction from retained routes and native mechanisms.

use eredu_core::BackendFailure;
use eredu_nn::{
    DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, RetainedTensorRoots, Tensor,
    TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    execution_control::SnapshotRetainedGrowth, ComponentRankMechanisms, LocalComponentMechanisms,
    PlacementRank, RealtimeGraphMechanisms, RealtimeStateMaterializer, SamplingBackend,
    StateLayout, SubmissionBackend,
};

use super::{
    AudioLanguagePlacement, AudioPlacement, PipelineAudio, PreparedResidentAudio, ReplicatedAudio,
    ResidentAudio, ResidentAudioBinding, ResidentAudioError, TensorParallelAudio,
};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Native mechanisms paired with an architecture-selected language route.
pub struct PlacedAudio<M, P> {
    mechanisms: M,
    placement: P,
}

impl<B, M, P> RealtimeGraphMechanisms<B> for PlacedAudio<M, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    M: RealtimeGraphMechanisms<B>,
{
    type State = M::State;
    type Components = M::Components;
    type SamplingError = M::SamplingError;
    type Sampling = M::Sampling;

    fn components(&self) -> &Self::Components {
        self.mechanisms.components()
    }

    fn context(&self) -> &Context<B> {
        self.mechanisms.context()
    }

    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure> {
        M::branch_state(state)
    }

    fn retained_values(state: &Self::State) -> Vec<&B::Tensor> {
        M::retained_values(state)
    }

    fn with_input_recovery<T, E>(operation: impl FnOnce() -> Result<T, E>) -> Result<T, E>
    where
        E: From<BackendFailure>,
    {
        M::with_input_recovery(operation)
    }

    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure> {
        M::with_fork_retention(retention, operation)
    }
}

impl<M, P, S> RealtimeStateMaterializer<StateLayout, S> for PlacedAudio<M, P>
where
    M: RealtimeStateMaterializer<StateLayout, S>,
{
    fn realize_state(&self, layout: &StateLayout) -> Result<S, BackendFailure> {
        self.mechanisms.realize_state(layout)
    }
}

impl<B, M, P> ResidentAudioBinding<B> for PlacedAudio<M, P>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    M: RealtimeGraphMechanisms<B>,
    P: AudioLanguagePlacement<B, M::State, M::Components>,
{
    type Placement = P;

    fn placement(&self) -> &P {
        &self.placement
    }
}

impl PreparedResidentAudio {
    /// Assembles the retained local graph using generic realtime mechanisms.
    pub fn bind_local<B, M>(
        self,
        mechanisms: M,
    ) -> Result<ResidentAudio<B, PlacedAudio<M, ReplicatedAudio>>, ResidentAudioError>
    where
        B: GroupedNeuralBackend
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = Context<B>>,
        M: RealtimeGraphMechanisms<B, Components: LocalComponentMechanisms<B, M::State>>,
        <M::Sampling as SamplingBackend>::RandomState: Clone,
    {
        ResidentAudio::from_prepared(
            self,
            PlacedAudio {
                mechanisms,
                placement: ReplicatedAudio,
            },
        )
    }

    /// Assembles the retained tensor rank after comparing the transport identity.
    pub fn bind_tensor<B, M>(
        self,
        mechanisms: M,
        world: PlacementRank,
        collective: B::ParallelContext,
    ) -> Result<
        ResidentAudio<B, PlacedAudio<M, TensorParallelAudio<B::ParallelContext>>>,
        ResidentAudioError,
    >
    where
        B: TensorParallelGroupedNeuralBackend
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = Context<B>>,
        B::ParallelContext: Clone,
        M: RealtimeGraphMechanisms<B, Components: LocalComponentMechanisms<B, M::State>>,
        <M::Sampling as SamplingBackend>::RandomState: Clone,
    {
        self.validate_world(world, AudioPlacement::Tensor)?;
        ResidentAudio::from_prepared(
            self,
            PlacedAudio {
                mechanisms,
                placement: TensorParallelAudio::new(collective),
            },
        )
    }

    /// Assembles the retained pipeline rank using its native transport.
    pub fn bind_pipeline<B, M, C>(
        self,
        mechanisms: M,
        world: PlacementRank,
        transport: C,
    ) -> Result<ResidentAudio<B, PlacedAudio<M, PipelineAudio<C>>>, ResidentAudioError>
    where
        B: TensorParallelGroupedNeuralBackend
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = Context<B>>,
        M: RealtimeGraphMechanisms<
            B,
            State: RetainedTensorRoots<B::Tensor>,
            Components: ComponentRankMechanisms<B, M::State, C>,
        >,
        <M::Sampling as SamplingBackend>::RandomState: Clone,
    {
        self.validate_world(world, AudioPlacement::Pipeline)?;
        ResidentAudio::from_prepared(
            self,
            PlacedAudio {
                mechanisms,
                placement: PipelineAudio::new(transport),
            },
        )
    }

    fn validate_world(
        &self,
        world: PlacementRank,
        route: impl FnOnce(eredu_core::ParallelRankTopology) -> AudioPlacement,
    ) -> Result<(), ResidentAudioError> {
        if self.placement().rank().is_some_and(|rank| {
            route(rank) == self.placement()
                && rank.world_size() == world.world_size()
                && rank.global_rank() == world.global_rank()
        }) {
            Ok(())
        } else {
            Err(ResidentAudioError::Selection(
                "transport world or route differs from the retained Audio rank".into(),
            ))
        }
    }
}
