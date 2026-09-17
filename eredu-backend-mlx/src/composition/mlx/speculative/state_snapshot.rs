//! Native settled-copy primitives shared by speculative state adapters.
use eredu_core::execution_control::SnapshotEstimate;
use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Stream};

pub(crate) fn estimate_array(array: &Array) -> Option<SnapshotEstimate> {
    let descriptor=array.try_descriptor().ok()?;
    let facts=descriptor.facts();
    let bytes=crate::backend::runtime::cache::state::snapshot_estimate::array_bytes(
        facts.logical_bytes(),facts.rank(),
    )?;
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

/// Same logical estimate through an allocation-free exact source visitor.
pub(crate) fn estimate_visited<S>(visit:impl FnOnce(&mut dyn FnMut(&Array)))
    ->Option<SnapshotEstimate> {
    let host=std::mem::size_of::<S>() as u64;
    let mut estimate=Some(SnapshotEstimate{retained_bytes:host,copy_bytes:host});
    visit(&mut |array|{
        estimate=estimate.and_then(|sum|{
            let item=estimate_array(array)?;
            Some(SnapshotEstimate{
                retained_bytes:sum.retained_bytes.checked_add(item.retained_bytes)?,
                copy_bytes:sum.copy_bytes.checked_add(item.copy_bytes)?,
            })
        });
    });
    estimate
}

pub(crate) fn settle(arrays: Vec<&Array>) -> Result<(), Exception> {
    if !arrays.is_empty() {
        async_eval_with_event(arrays)?.synchronize()?;
    }
    Ok(())
}

pub(crate) fn copy_array(array: &Array, stream: &Stream) -> Result<Array, Exception> {
    settle(vec![array])?;
    // MLX's copy primitive can preserve the source backing even after exact
    // completion. Durable snapshots require independent storage. The bounded
    // estimate includes contiguous staging plus this explicit destination copy.
    let copy = array.contiguous(false, stream)?.deep_clone()?;
    // Establish completion on the exact returned handle before attaching its
    // allocation-lifetime authority. Cold allocation inspection never does so.
    copy.evaluated()?;
    Ok(copy)
}
