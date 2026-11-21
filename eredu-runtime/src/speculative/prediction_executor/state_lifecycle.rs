//! Typed state membership and native lifecycle delegation for prediction lanes.
use super::{snapshot, PredictionExtensionMaterializer};
use crate::speculative::SequentialPredictionCache;
use crate::PredictionModelState;
use eredu_core::{
    cache::mechanisms::CompressedCacheStorage, execution_control::SnapshotEstimate,
    speculative::SpeculativePredictionMemoryObservation, BackendFailure,
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
                .flat_map(crate::RuntimeLayerState::<B>::retained_values)
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
        use crate::state::PortableTextState;

        let retained = state
            .isolated_snapshot_estimate()
            .map(|estimate| estimate.retained_bytes);
        Some(SpeculativePredictionMemoryObservation {
            layer_positions: state
                .layer_positions()
                .into_iter()
                .map(|position| u64::try_from(position).ok())
                .collect::<Option<_>>()?,
            current_state_bytes: retained
                .and_then(|bytes| bytes.checked_add(state.isolated_snapshot_growth(0)?)),
            peak_state_bytes: retained
                .and_then(|bytes| bytes.checked_add(state.isolated_snapshot_growth(additional)?)),
        })
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
    type State = Vec<SequentialPredictionCache<B, M>>;

    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(crate::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate> {
        snapshot::sequence_estimate(state, |cache| {
            if cache.is_paged() {
                return None;
            }
            let host = std::mem::size_of::<SequentialPredictionCache<B, M>>() as u64;
            cache.retained_arrays().into_iter().try_fold(
                SnapshotEstimate {
                    retained_bytes: host,
                    copy_bytes: host,
                },
                |sum, tensor| {
                    let estimate = M::SequentialStorage::snapshot_tensor_estimate(tensor)?;
                    Some(SnapshotEstimate {
                        retained_bytes: sum.retained_bytes.checked_add(estimate.retained_bytes)?,
                        copy_bytes: sum.copy_bytes.checked_add(estimate.copy_bytes)?,
                    })
                },
            )
        })
    }

    fn memory_observation(
        state: &Self::State,
        _additional: u64,
    ) -> Option<SpeculativePredictionMemoryObservation> {
        snapshot::sequence_memory(state, |cache| {
            Some(SpeculativePredictionMemoryObservation {
                layer_positions: vec![u64::try_from(cache.offset()).ok()?],
                current_state_bytes: None,
                peak_state_bytes: None,
            })
        })
    }

    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure> {
        use eredu_core::cache::mechanisms::LiveCacheBackend;
        snapshot::sequence_copy(state, |cache| {
            if cache.is_paged() {
                return Ok(None);
            }
            if let Some((latent, rotary)) = cache.arrays() {
                M::SequentialStorage::settle_tensors(&[latent, rotary])?;
            }
            let copy = cache.isolated_snapshot(context)?;
            if let Some((latent, rotary)) = copy.arrays() {
                M::SequentialStorage::settle_tensors(&[latent, rotary])?;
            }
            Ok(Some(copy))
        })
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
    type State = Vec<crate::speculative::PredictionPoolingState<B, M>>;

    fn complete_state(
        state: &mut Self::State,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(crate::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
        )
    }

    fn snapshot_estimate(state: &Self::State) -> Option<SnapshotEstimate> {
        snapshot::sequence_estimate(state, |layer| layer.prediction_snapshot_estimate())
    }

    fn memory_observation(
        state: &Self::State,
        _: u64,
    ) -> Option<SpeculativePredictionMemoryObservation> {
        snapshot::sequence_memory(state, |layer| {
            Some(SpeculativePredictionMemoryObservation {
                layer_positions: vec![u64::try_from(eredu_nn::PoolingAttentionCache::offset(
                    layer,
                ))
                .ok()?],
                current_state_bytes: None,
                peak_state_bytes: None,
            })
        })
    }

    fn snapshot(
        state: &Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::State>, BackendFailure> {
        snapshot::sequence_copy(state, |layer| {
            layer.settled_prediction_snapshot(context).map(Some)
        })
    }
}
