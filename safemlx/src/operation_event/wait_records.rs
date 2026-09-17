use super::OperationEvent;

/// Fixed native consumer-wait Record requests on the qualified implementation.
/// Existing arenas already fund these bytes. Backend tasks/handlers and arena
/// block metadata, tail absorption and fragmentation are separate contributions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationWaitRecordLayout {
    native: safemlx_sys::mlx_operation_wait_record_layout,
}
impl OperationWaitRecordLayout {
    /// Maximum once-only event-backed waits represented by the caller.
    pub fn wait_count(self) -> usize {
        self.native.wait_count
    }
    /// Per-wait concrete WaitRecord object request, including its contained owners.
    pub fn object_bytes(self) -> usize {
        self.native.object_bytes
    }
    /// Alignment of each concrete WaitRecord object.
    pub fn object_alignment(self) -> usize {
        self.native.object_alignment
    }
    /// Per-wait initial empty capture slots allocated by the Record base.
    pub fn capture_slots(self) -> usize {
        self.native.capture_slots
    }
    /// Per-wait initial capture backing request.
    pub fn capture_bytes(self) -> usize {
        self.native.capture_bytes
    }
    /// Alignment of the capture backing.
    pub fn capture_alignment(self) -> usize {
        self.native.capture_alignment
    }
    /// Per-wait stream receipts; one for this producer.
    pub fn stream_receipts(self) -> usize {
        self.native.stream_receipts
    }
    /// Per-wait stream-receipt backing request.
    pub fn stream_bytes(self) -> usize {
        self.native.stream_bytes
    }
    /// Alignment of the stream-receipt backing.
    pub fn stream_alignment(self) -> usize {
        self.native.stream_alignment
    }
    /// Nonempty Record allocation requests per wait.
    pub fn allocations_per_wait(self) -> usize {
        self.native.allocations_per_wait
    }
    /// Sum of the three per-wait byte requests, excluding arena block overhead.
    pub fn requested_bytes_per_wait(self) -> usize {
        self.native.requested_bytes_per_wait
    }
    /// Checked allocation request count across all represented waits.
    pub fn total_record_allocations(self) -> usize {
        self.native.total_record_allocations
    }
    /// Checked total requested bytes; not a successful-workspace or arena-fit bound.
    pub fn total_record_requested_bytes(self) -> usize {
        self.native.total_record_requested_bytes
    }
    /// All attempted native Record extents, including allocator headers and
    /// alignment, without early-retirement credit or the final arena split tail.
    pub fn record_allocation_extents(self) -> Option<usize> {
        [
            (self.object_bytes(), self.object_alignment()),
            (self.capture_bytes(), self.capture_alignment()),
            (self.stream_bytes(), self.stream_alignment()),
        ]
        .into_iter()
        .try_fold(0usize, |sum, (bytes, alignment)| {
            sum.checked_add(crate::SubmissionRecordQuota::allocation_extent(
                bytes, alignment,
            )?)
        })?
        .checked_mul(self.wait_count())
    }

    /// Named native query/factory/wait transports, not extra heap requests or a compiler stack bound.
    pub fn named_control_bytes(self) -> usize {
        self.native.named_control_bytes
    }
}
impl OperationEvent {
    /// Pure pre-work recipe query. Unknown compiler/library qualification or
    /// count overflow returns None before constructing any native owner.
    /// Counts must come from the actual finite caller, not a progress snapshot.
    pub fn wait_record_layout(waits: usize) -> Option<OperationWaitRecordLayout> {
        let mut native = safemlx_sys::mlx_operation_wait_record_layout::default();
        // SAFETY: the scalar output is valid; the pure native query retains nothing.
        unsafe { safemlx_sys::mlx_operation_event_wait_record_layout(&mut native, waits) }
            .then_some(OperationWaitRecordLayout { native })
    }
}
