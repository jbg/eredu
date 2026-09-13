//! Native settled-copy primitives shared by speculative state adapters.
use eredu_core::execution_control::SnapshotEstimate;
use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Stream};

pub(crate) fn estimate_array(array: &Array) -> Option<SnapshotEstimate> {
    let bytes = u64::try_from(array.nbytes())
        .ok()?
        .checked_mul(2)?
        .checked_add(4096)?
        .checked_add((array.shape().len() as u64).checked_mul(16)?)?;
    Some(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

pub(crate) fn estimate_state<'a, S>(
    arrays: impl IntoIterator<Item = &'a Array>,
) -> Option<SnapshotEstimate> {
    let host = std::mem::size_of::<S>() as u64;
    arrays.into_iter().try_fold(
        SnapshotEstimate {
            retained_bytes: host,
            copy_bytes: host,
        },
        |sum, array| {
            let array = estimate_array(array)?;
            Some(SnapshotEstimate {
                retained_bytes: sum.retained_bytes.checked_add(array.retained_bytes)?,
                copy_bytes: sum.copy_bytes.checked_add(array.copy_bytes)?,
            })
        },
    )
}

pub(crate) fn settle(arrays: Vec<&Array>) -> Result<(), Exception> {
    if !arrays.is_empty() {
        async_eval_with_event(arrays)?.synchronize()?;
    }
    Ok(())
}

pub(crate) fn copy_array(array: &Array, stream: &Stream) -> Result<Array, Exception> {
    settle(vec![array])?;
    let copy = array.copy(stream)?;
    settle(vec![&copy])?;
    Ok(copy)
}
