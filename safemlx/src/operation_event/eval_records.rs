use super::OperationEvent;

/// Six requested Record allocations for a selected untimed asynchronous Eval.
/// Earlier traversal, later capture growth, other Graph/native owners and allocator
/// block overhead are separate. This does not certify cold admission or fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationEvalRecordLayout {
    native: safemlx_sys::mlx_operation_eval_record_layout,
}
impl OperationEvalRecordLayout {
    /// Actual tape entries, including the selected Synchronizer.
    pub fn tape_entries(self) -> usize {
        self.native.tape_entries
    }
    /// Distinct full stream identities in the selected untimed tape.
    pub fn stream_count(self) -> usize {
        self.native.stream_count
    }
    /// Sum of one plus sibling count per tape entry.
    pub fn output_slots(self) -> usize {
        self.native.output_slots
    }
    /// Concrete Eval Record object request.
    pub fn object_bytes(self) -> usize {
        self.native.object_bytes
    }
    /// Alignment of the Eval Record object.
    pub fn object_alignment(self) -> usize {
        self.native.object_alignment
    }
    /// Initial empty Data capture slots; later growth is separate.
    pub fn capture_slots(self) -> usize {
        self.native.capture_slots
    }
    /// Initial capture backing request.
    pub fn capture_bytes(self) -> usize {
        self.native.capture_bytes
    }
    /// Alignment of initial capture backing.
    pub fn capture_alignment(self) -> usize {
        self.native.capture_alignment
    }
    /// Fresh derived stream-state vector request.
    pub fn stream_state_bytes(self) -> usize {
        self.native.stream_state_bytes
    }
    /// Alignment of stream-state entries.
    pub fn stream_state_alignment(self) -> usize {
        self.native.stream_state_alignment
    }
    /// Fresh base stream-receipt vector request.
    pub fn stream_receipt_bytes(self) -> usize {
        self.native.stream_receipt_bytes
    }
    /// Alignment of stream receipts.
    pub fn stream_receipt_alignment(self) -> usize {
        self.native.stream_receipt_alignment
    }
    /// Fresh primitive-owner vector request.
    pub fn primitive_owner_bytes(self) -> usize {
        self.native.primitive_owner_bytes
    }
    /// Alignment of shared primitive owners.
    pub fn primitive_owner_alignment(self) -> usize {
        self.native.primitive_owner_alignment
    }
    /// Fresh physical output-pin vector request.
    pub fn output_pin_bytes(self) -> usize {
        self.native.output_pin_bytes
    }
    /// Alignment of physical output pins.
    pub fn output_pin_alignment(self) -> usize {
        self.native.output_pin_alignment
    }
    /// Nonempty requests in this six-request slice.
    pub fn record_allocations(self) -> usize {
        self.native.record_allocations
    }
    /// Sum of requested bytes; excludes arena overhead and earlier traversal.
    pub fn record_requested_bytes(self) -> usize {
        self.native.record_requested_bytes
    }
    /// Four physical constructor requests: Synchronizer, raw descriptor,
    /// descriptor shared control and root Event shared control. Their actual
    /// type alignments are reported here; reservation uses the common maximum.
    pub fn host_graph_requests(self) -> [(usize, usize); 4] {
        std::array::from_fn(|i| {
            (
                self.native.host_graph_request_bytes[i],
                self.native.host_graph_request_alignments[i],
            )
        })
    }
    /// Exact owning header and pointer slots of the shared Graph reservation.
    pub fn host_graph_owner_requests(self) -> [(usize, usize); 2] {
        [
            (
                self.native.host_graph_header_bytes,
                self.native.host_graph_header_alignment,
            ),
            (
                self.native.host_graph_slots_bytes,
                self.native.host_graph_slots_alignment,
            ),
        ]
    }
    /// Four consumed constructor blocks, separate from header/slot storage.
    pub fn host_graph_blocks(self) -> usize {
        self.native.host_graph_blocks
    }
    /// Common physical alignment of all four reserved constructor blocks.
    pub fn host_graph_reserved_alignment(self) -> usize {
        self.native.host_graph_reserved_alignment
    }
    /// All six owning requests, without a duplicate charge against the Graph arena.
    pub fn host_graph_requested_bytes(self) -> usize {
        self.native.host_graph_requested_bytes
    }
    /// Per-request physical extents; fragmentation and absorbed tails still
    /// require actual physical preflight before any Synchronizer construction.
    pub fn host_graph_allocation_extents(self) -> usize {
        self.native.host_graph_allocation_extents
    }
    /// Event constructor transports already included in query_control_bytes().
    pub fn host_graph_event_controls(self) -> usize {
        self.native.host_graph_event_controls
    }
    /// Controlled platform Event population, not opaque driver-private bytes
    /// or a promise that platform creation succeeds after Graph reservation.
    pub fn host_graph_platform_events(self) -> usize {
        self.native.host_graph_platform_events
    }
    /// Named query/native factory transports, including this Rust query's
    /// scalar output/result. Not additional heap or a compiler stack-frame bound.
    pub fn query_control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<safemlx_sys::mlx_operation_eval_record_layout>(),
            size_of::<Option<Self>>(),
            3 * size_of::<usize>(),
            size_of::<bool>(),
            size_of::<*mut safemlx_sys::mlx_operation_eval_record_layout>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}
impl OperationEvent {
    /// Pure qualified layout of the four fresh post-tape reservations, Eval
    /// object and initial capture backing. No runtime loan or native owner is
    /// constructed. Invalid shape, overflow or unknown qualification returns None.
    ///
    /// Inputs must come from the actual selected producer: tape entries include
    /// the Synchronizer, streams count distinct full identities, and output slots
    /// sum one plus sibling count per tape entry. Discovering these counts after
    /// traversal cannot certify admission before that traversal. The separate
    /// unselected eventless fast return has no Eval and is not a zero-shaped Eval.
    pub fn eval_record_layout(
        tape_entries: usize,
        streams: usize,
        output_slots: usize,
    ) -> Option<OperationEvalRecordLayout> {
        let mut native = safemlx_sys::mlx_operation_eval_record_layout::default();
        // SAFETY: valid scalar output; the pure native query retains nothing.
        unsafe {
            safemlx_sys::mlx_operation_event_eval_record_layout(
                &mut native,
                tape_entries,
                streams,
                output_slots,
            )
        }
        .then_some(OperationEvalRecordLayout { native })
    }
}
