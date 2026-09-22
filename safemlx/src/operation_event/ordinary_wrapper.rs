//! Fixed controls for the ordinary asynchronous completion wrapper.
use super::*;
use crate::utils::{guard::Guarded, VectorArray};
use std::mem::{size_of, size_of_val};

impl Event {
    /// Fixed transports of a host wait on an existing ordinary event. This
    /// does not submit a consumer wait record or allocate another event.
    pub fn ordinary_synchronize_control_bytes() -> Option<usize> {
        let fields = [
            size_of::<&Self>(),
            size_of::<Result<()>>(),
            size_of::<<() as Guarded>::Guard>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_event>(),
            size_of::<i32>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}

impl OperationEvent {
    /// Fixed transports of the ordinary consumer wait wrapper. Native wait
    /// records and backend tasks have their own observed-control source query.
    pub fn ordinary_wait_wrapper_control_bytes() -> Option<usize> {
        let fields = [
            size_of::<(&Self, &Stream)>(),
            size_of::<(&Event, &Stream)>(),
            size_of::<Result<()>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_stream>(),
            size_of::<safemlx_sys::mlx_event>(),
            size_of::<i32>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
    /// C vector/completion shells and Rust call transports used by the ordinary
    /// asynchronous submission wrapper. Native root-vector buffers, graph
    /// construction, evaluation and dispatch are separate contributions.
    pub fn ordinary_submission_wrapper_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query, with no runtime access or allocation.
        let native = unsafe { safemlx_sys::mlx_ordinary_event_wrapper_controls() };
        let fields = [
            native,
            size_of::<VectorArray>(),
            size_of::<<VectorArray as Guarded>::Guard>(),
            size_of::<Result<VectorArray>>(),
            size_of::<Event>(),
            size_of::<<Event as Guarded>::Guard>(),
            size_of::<Result<Event>>(),
            size_of::<Self>(),
            size_of::<Result<Self>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_vector_array>(),
            size_of::<safemlx_sys::mlx_event>(),
            size_of::<&Array>(),
            size_of::<i32>() * 2,
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}

impl Stream {
    /// The ordinary clone's native Stream shell and fixed call transports.
    /// This clones a selected stream value, without creating a native stream.
    pub fn ordinary_clone_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query, with no native execution.
        let native = unsafe { safemlx_sys::mlx_ordinary_stream_clone_wrapper_controls() };
        let fields = [
            native,
            size_of::<&Self>(),
            size_of::<Self>(),
            size_of::<<Self as Guarded>::Guard>(),
            size_of::<Result<Self>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}
