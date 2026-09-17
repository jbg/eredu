use super::*;
use std::{ffi::c_void, marker::PhantomData, ptr};

/// Qualified host constructor envelope for generic Add/Multiply. This covers
/// descriptor/primitive/vector Graph requests, including temporary overlap.
/// It does not cover Eval, workers, Data, native buffers or outer C array handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointwiseGraphLayout {
    native: safemlx_sys::mlx_pointwise_graph_layout,
}
impl PointwiseGraphLayout {
    /// Number of operations admitted by this envelope.
    pub fn operations(self) -> usize {
        self.native.operations
    }
    /// Maximum declared input/output rank across the actual operation stream.
    pub fn maximum_rank(self) -> usize {
        self.native.maximum_rank
    }
    /// Maximum physical constructor blocks, excluding header and slots.
    pub fn blocks(self) -> usize {
        self.native.blocks
    }
    /// Requested bytes including the Graph-owned header and pointer slots.
    pub fn requested_bytes(self) -> usize {
        self.native.requested_bytes
    }
    /// Sum of per-request allocator extents. Tail absorption and fragmentation
    /// mean this is not a sufficient free-byte test; preparation reserves blocks.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// (Maximum bytes, actual alignment, count) per actual constructor class.
    /// Physical reserved blocks all use reserved_alignment(), allowing smaller
    /// requests to take the smallest adequate block without an alignment hazard.
    pub fn classes(&self) -> impl ExactSizeIterator<Item = (usize, usize, usize)> + '_ {
        self.native
            .request_bytes
            .iter()
            .copied()
            .zip(self.native.request_alignments.iter().copied())
            .zip(self.native.request_counts.iter().copied())
            .map(|((bytes, alignment), count)| (bytes, alignment, count))
    }
    /// Alignment of each physically reserved constructor block.
    pub fn reserved_alignment(self) -> usize {
        self.native.reserved_alignment
    }
    /// Exact header and finite pointer-slot allocation requests.
    pub fn owner_requests(self) -> [(usize, usize); 2] {
        [
            (self.native.header_bytes, self.native.header_alignment),
            (self.native.slots_bytes, self.native.slots_alignment),
        ]
    }
    /// Named native/C/safe query, preparation and retirement transports.
    /// The Graph-owned header/slots are reported separately in requested_bytes.
    pub fn control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_pointwise_graph_layout>(),
            size_of::<PreparedPointwiseGraph<'static>>(),
            size_of::<Result<PreparedPointwiseGraph<'static>>>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<*mut c_void>(),
            size_of::<u32>(),
            size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}

/// Once-owned physical host reservation bound to the borrowed exact role.
/// It must be dropped before Eval. Drop clears the host association and frees
/// unused blocks/slots/header; consumed blocks remain with their real owners.
#[derive(Debug)]
pub struct PreparedPointwiseGraph<'a> {
    owner: *mut c_void,
    _observer: PhantomData<&'a OriginalScopeObserver>,
}
impl Drop for PreparedPointwiseGraph<'_> {
    fn drop(&mut self) {
        // SAFETY: unique opaque owner, no payload/native resource in unused
        // slots. GraphQuota serializes links, and frees outside its loan. The
        // borrowed observer retains source custody through actual header free.
        unsafe { safemlx_sys::mlx_operation_event_finish_pointwise_graph(self.owner) };
    }
}
impl OperationEvent {
    /// Pure qualified recipe; invalid/overflow/unqualified input stays unknown.
    pub fn pointwise_graph_layout(
        operations: usize,
        maximum_rank: usize,
    ) -> Option<PointwiseGraphLayout> {
        let mut native = safemlx_sys::mlx_pointwise_graph_layout::default();
        // SAFETY: scalar query, no allocation or authority, output written only
        // when the owning native layout is qualified.
        unsafe {
            safemlx_sys::mlx_operation_event_pointwise_graph_layout(
                &mut native,
                operations,
                maximum_rank,
            )
        }
        .then_some(PointwiseGraphLayout { native })
    }
    /// Reserve the actual header, slots and constructor blocks before any
    /// tensor operation. Failure leaves the role, source and caller plan intact.
    /// A second active host reservation refuses; it cannot borrow/refill one.
    pub fn prepare_pointwise_graph<'a>(
        layout: PointwiseGraphLayout,
        observer: &'a OriginalScopeObserver,
    ) -> Result<PreparedPointwiseGraph<'a>> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut owner = ptr::null_mut();
        // SAFETY: null unique destination, live exact observer, immutable scalar
        // recipe. Native code recomputes every byte/count from the two inputs.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_prepare_pointwise_graph(
                &mut owner,
                observer.raw,
                layout.operations(),
                layout.maximum_rank(),
            )
        };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(PreparedPointwiseGraph {
            owner,
            _observer: PhantomData,
        })
    }
}
