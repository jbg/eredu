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
