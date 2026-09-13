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
