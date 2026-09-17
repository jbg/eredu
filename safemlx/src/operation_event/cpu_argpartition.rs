use super::OperationEvent;
use std::mem::size_of;

/// Concrete rank-two, last-axis CPU partition task and its Eval destinations.
/// Buffer payloads, cross-stream receipts and process worker startup are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuArgPartitionLayout {
    native: safemlx_sys::mlx_cpu_argpartition_layout,
}
impl CpuArgPartitionLayout {
    /// Eight owning request classes as (bytes, alignment, count).
    pub fn requests(self) -> [(usize, usize, usize); 8] {
        std::array::from_fn(|i| {
            (
                self.native.request_bytes[i],
                self.native.request_alignments[i],
                self.native.request_counts[i],
            )
        })
    }
    /// The shared Graph bank's header and finite pointer-slot requests.
    pub fn owner_requests(self) -> [(usize, usize); 2] {
        [
            (self.native.header_bytes, self.native.header_alignment),
            (self.native.slots_bytes, self.native.slots_alignment),
        ]
    }
    /// Requested Graph bytes; these are not a second allocation account.
    pub fn requested_bytes(self) -> usize {
        self.native.requested_bytes
    }
    /// Worst-alignment extents. Actual physical preflight still handles fragmentation.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Named query, host invocation and concrete worker controls.
    pub fn control_bytes(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_argpartition_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_argpartition_layout>(),
            size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}
impl OperationEvent {
    /// Same last-axis F32 worker for a positive rank-one through rank-three
    /// source. Actual primitive, shape, stride span and kth validation occur at
    /// the native Eval boundary; this query grants no execution authority.
    pub fn cpu_argpartition_source_layout(rank:usize,elements:usize,tracer:bool)->Option<CpuArgPartitionLayout> {
        let mut native=safemlx_sys::mlx_cpu_argpartition_layout::default();
        // SAFETY: initialized fixed output and scalar source geometry only.
        let known=unsafe {safemlx_sys::mlx_operation_event_cpu_argpartition_source_layout(
            &mut native,rank,elements,tracer)};
        if !known {return None;}
        let controls=size_of::<(usize,usize,bool)>()+size_of::<*mut safemlx_sys::mlx_cpu_argpartition_layout>();
        native.named_control_bytes=native.named_control_bytes.checked_add(controls)?;
        Some(CpuArgPartitionLayout {native})
    }
    /// Pure owning query; does not create a stream, worker, task or arena.
    pub fn cpu_argpartition_layout(tracer: bool) -> Option<CpuArgPartitionLayout> {
        let mut native = safemlx_sys::mlx_cpu_argpartition_layout::default();
        // SAFETY: initialized scalar output; the native query publishes only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_argpartition_layout(&mut native, tracer) }
            .then_some(CpuArgPartitionLayout { native })
    }
}

/// Baseline input/output crossing owners, including slow Event and fast Fence
/// alternatives. Repeated consumers and within-Eval waits need their separate
/// worker population; this baseline alone is not a complete traversal fit.
/// No new arena or source account is represented here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterReceiptLayout {
    native: safemlx_sys::mlx_router_receipt_layout,
}
impl RouterReceiptLayout {
    /// Requested baseline Graph payloads for one input/output crossing.
    pub fn graph_requested_bytes(self) -> usize {
        self.native.graph_requested_bytes
    }
    /// Worst-alignment Graph extents; actual allocation still handles fragmentation.
    pub fn graph_allocation_extents(self) -> usize {
        self.native.graph_allocation_extents
    }
    /// Maximum platform Event creations, separate from managed Graph requests.
    pub fn platform_events(self) -> usize {
        self.native.platform_events
    }
    /// Actual successful fast-Fence payload bytes before allocator capacity rounding.
    pub fn fast_backing_bytes(self) -> usize {
        self.native.fast_backing_bytes
    }
    /// Physical fast-Fence generations; the slow Event alternative has zero.
    pub fn fast_backing_births(self) -> usize {
        self.native.fast_backing_births
    }
    /// Named owning-module, C and safe query/dispatch controls.
    pub fn control_bytes(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_router_receipt_layout>(),
            size_of::<*mut safemlx_sys::mlx_router_receipt_layout>(),
            size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}
impl OperationEvent {
    /// Pure mixed Metal/CPU receipt query. Other backends report unknown.
    pub fn router_receipt_layout() -> Option<RouterReceiptLayout> {
        let mut native = safemlx_sys::mlx_router_receipt_layout::default();
        // SAFETY: fixed scalar output; no stream, event or task is constructed.
        unsafe { safemlx_sys::mlx_operation_event_router_receipt_layout(&mut native) }
            .then_some(RouterReceiptLayout { native })
    }
}
