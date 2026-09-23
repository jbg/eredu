use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};

pub(super) fn sequence_estimate<T>(
    states: &[T],
    estimate: impl Fn(&T) -> Option<SnapshotEstimate>,
) -> Option<SnapshotEstimate> {
    let host = (std::mem::size_of::<Vec<T>>() as u64)
        .checked_add((states.len() as u64).checked_mul(std::mem::size_of::<T>() as u64)?)?;
    states.iter().try_fold(
        SnapshotEstimate {
            retained_bytes: host,
            copy_bytes: host,
        },
        |sum, state| {
            let item = estimate(state)?;
            Some(SnapshotEstimate {
                retained_bytes: sum.retained_bytes.checked_add(item.retained_bytes)?,
                copy_bytes: sum.copy_bytes.checked_add(item.copy_bytes)?,
            })
        },
    )
}

pub(super) fn sequence_copy<T>(
    states: &[T],
    copy: impl Fn(&T) -> Result<Option<T>, BackendFailure>,
) -> Result<Option<Vec<T>>, BackendFailure> {
    states.iter().map(copy).collect()
}

/// Composes ordinary state membership; native producers own capacity facts.
pub(super) fn sequence_memory<T>(
    states: &[T],
    observe: impl Fn(&T) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation>,
) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
    let mut result = eredu_core::speculative::SpeculativePredictionMemoryObservation {
        layer_positions: Vec::new(),
        current_state_bytes: Some(0),
        peak_state_bytes: Some(0),
    };
    for state in states {
        let observation = observe(state)?;
        result.layer_positions.extend(observation.layer_positions);
        result.current_state_bytes = result
            .current_state_bytes
            .zip(observation.current_state_bytes)
            .and_then(|(a, b)| a.checked_add(b));
        result.peak_state_bytes = result
            .peak_state_bytes
            .zip(observation.peak_state_bytes)
            .and_then(|(a, b)| a.checked_add(b));
    }
    Some(result)
}
