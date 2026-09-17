use super::{
    EmbeddedPredictionStrategy, EmbeddedPredictionTargetState, SpeculativeControlError,
    SpeculativeTensorMechanisms,
};
use eredu_core::execution_control::SnapshotEstimate;

pub(super) fn copy<'a, S, M>(
    strategy: &S,
    cache: &S::TargetCache,
    state: &EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>,
    context: M::Context<'a>,
) -> Result<
    Option<(
        S::TargetCache,
        EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>,
    )>,
    SpeculativeControlError,
>
where
    M: SpeculativeTensorMechanisms,
    S: EmbeddedPredictionStrategy<M>,
{
    if sum([
        strategy.control_target_estimate(cache),
        strategy.control_prediction_estimate(&state.prediction_cache),
        M::control_tensor_packet_estimate(&state.capture),
    ])
    .is_none()
    {
        return Ok(None);
    }
    let Some(cache) = strategy.control_target_snapshot(cache, context)? else {
        return Ok(None);
    };
    let Some(prediction_cache) =
        strategy.control_prediction_snapshot(&state.prediction_cache, context)?
    else {
        return Ok(None);
    };
    let Some(capture) = M::control_tensor_packet_snapshot(&state.capture, context)? else {
        return Ok(None);
    };
    Ok(Some((
        cache,
        EmbeddedPredictionTargetState {
            capture,
            prediction_cache,
        },
    )))
}

pub(super) fn sum<const N: usize>(
    parts: [Option<SnapshotEstimate>; N],
) -> Option<SnapshotEstimate> {
    parts.into_iter().try_fold(
        SnapshotEstimate {
            retained_bytes: 0,
            copy_bytes: 0,
        },
        |sum, part| {
            let part = part?;
            Some(SnapshotEstimate {
                retained_bytes: sum.retained_bytes.checked_add(part.retained_bytes)?,
                copy_bytes: sum.copy_bytes.checked_add(part.copy_bytes)?,
            })
        },
    )
}

pub(super) fn host<T>(
    identity: Option<&eredu_runtime::SpeculativeIdentity>,
) -> Option<SnapshotEstimate> {
    let bytes = (std::mem::size_of::<T>() as u64)
        .checked_add(identity.map_or(0, |id| id.as_str().len() as u64))?;
    Some(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}
