//! Typed state membership and native lifecycle delegation for prediction lanes.
use super::{snapshot, PredictionExtensionMaterializer, PredictionModelState};
use eredu_core::{
    execution_control::SnapshotEstimate, speculative::SpeculativePredictionMemoryObservation,
    BackendFailure,
};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Tensor,
};

/// Exact lane-state shape and lifecycle mechanisms selected by an executor.
/// A profile enumerates state roots; the materializer settles native dependencies.
pub trait PredictionStateLifecycle<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Existing mutable lane owner, without an additional storage wrapper.
    type State: Clone + 'static;

    /// Settles all retained state roots and actual phase outputs.
    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure>;
    /// Native capacity/frontier observation, preserving unavailable facts.
    fn memory_observation(
        state: &Self::State,
        additional: u64,
    ) -> Option<SpeculativePredictionMemoryObservation>;
    /// Complete copy bound, including host container ownership.
    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate>;
    /// Copies after reservation and settles native copy dependencies.
    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure>;
}

/// Lifecycle for architecture-ordered model state.
pub struct ModelPredictionStateProfile;

impl<B, M> PredictionStateLifecycle<B, M> for ModelPredictionStateProfile
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type State = M::ModelState;

    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        M::complete_prediction_values(
            state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }

    fn memory_observation(
        state: &Self::State,
        additional: u64,
    ) -> Option<SpeculativePredictionMemoryObservation> {
        M::model_memory_observation(state, additional)
    }

    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure> {
        M::model_snapshot(state, context)
    }
}

/// Lifecycle for an ordered sequential cache list.
pub struct SequentialPredictionStateProfile;

impl<B, M> PredictionStateLifecycle<B, M> for SequentialPredictionStateProfile
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type State = Vec<M::SequentialState>;

    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::sequential_snapshot_estimate)
    }

    fn memory_observation(
        state: &Self::State,
        additional: u64,
    ) -> Option<SpeculativePredictionMemoryObservation> {
        snapshot::sequence_memory(state, |state| {
            M::sequential_memory_observation(state, additional)
        })
    }

    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure> {
        snapshot::sequence_copy(state, |state| M::sequential_snapshot(state, context))
    }
}

/// Lifecycle for an ordered pooling cache list.
pub struct PoolingPredictionStateProfile;

impl<B, M> PredictionStateLifecycle<B, M> for PoolingPredictionStateProfile
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type State = Vec<M::PoolingState>;

    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::pooling_snapshot_estimate)
    }

    fn memory_observation(
        state: &Self::State,
        additional: u64,
    ) -> Option<SpeculativePredictionMemoryObservation> {
        snapshot::sequence_memory(state, |state| {
            M::pooling_memory_observation(state, additional)
        })
    }

    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure> {
        snapshot::sequence_copy(state, |state| M::pooling_snapshot(state, context))
    }
}
