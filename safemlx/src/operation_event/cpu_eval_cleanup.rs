use super::OperationEvent;

/// Exact CPU Eval cleanup task and empty Data-reference storage. These blocks
/// use the existing Graph arena; primitive tasks and invocation vectors are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuEvalCleanupLayout {
    native: safemlx_sys::mlx_cpu_eval_cleanup_layout,
}
impl CpuEvalCleanupLayout {
    /// Payload requests: the concrete cleanup TaskNode and optional Data slots.
    pub fn requests(self) -> [(usize, usize); 2] {
        std::array::from_fn(|i| {
            (
                self.native.request_bytes[i],
                self.native.request_alignments[i],
            )
        })
    }
    /// Shared physical bank header and pointer-slot requests.
    pub fn owner_requests(self) -> [(usize, usize); 2] {
        [
            (self.native.header_bytes, self.native.header_alignment),
            (self.native.slots_bytes, self.native.slots_alignment),
        ]
    }
    /// Nonempty payload allocations, excluding the bank header and slots.
    pub fn blocks(self) -> usize {
        self.native.blocks
    }
    /// Requested Graph bytes, including the temporary bank header and slots.
    pub fn requested_bytes(self) -> usize {
        self.native.requested_bytes
    }
    /// Worst-alignment extents; physical preflight still handles fragmentation.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Named query and host transports, excluding Graph backing and shared startup.
    pub fn control_bytes(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_eval_cleanup_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_eval_cleanup_layout>(),
            2 * size_of::<usize>(),
            size_of::<bool>(),
            size_of::<CpuEvalCleanupPopulation>(),
            size_of::<Option<CpuEvalCleanupPopulation>>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
    /// Checked selected CPU populations. Include a CPU Synchronizer's root
    /// edges once. Counts describe storage; they grant neither admission nor fit.
    pub fn population(
        self,
        evaluations: usize,
        input_edges: usize,
        sibling_slots: usize,
    ) -> Option<CpuEvalCleanupPopulation> {
        if input_edges > evaluations.checked_mul(self.native.inputs)?
            || sibling_slots > evaluations.checked_mul(self.native.siblings)?
        {
            return None;
        }
        let slots = input_edges.checked_add(sibling_slots)?;
        Some(CpuEvalCleanupPopulation {
            maximum: self,
            evaluations,
            input_edges,
            sibling_slots,
            task_bytes: evaluations.checked_mul(self.native.request_bytes[0])?,
            data_slot_bytes: slots.checked_mul(self.native.data_slot_bytes)?,
            data_allocations: evaluations.min(slots),
        })
    }
}

/// Finite cumulative CPU cleanup storage. Nodes and slots may overlap until
/// accepted FIFO work retires; no second queue node or accounting owner exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuEvalCleanupPopulation {
    maximum: CpuEvalCleanupLayout,
    evaluations: usize,
    input_edges: usize,
    sibling_slots: usize,
    task_bytes: usize,
    data_slot_bytes: usize,
    data_allocations: usize,
}
impl CpuEvalCleanupPopulation {
    /// Maximum per-entry layout, with the actual temporary preflight owner shapes.
    pub fn maximum(self) -> CpuEvalCleanupLayout {
        self.maximum
    }
    /// Actual selected CPU evaluation ceiling, including its Synchronizer.
    pub fn evaluations(self) -> usize {
        self.evaluations
    }
    /// Cumulative input-edge slots, including CPU root edges once.
    pub fn input_edges(self) -> usize {
        self.input_edges
    }
    /// Cumulative sibling slots from those same CPU evaluations.
    pub fn sibling_slots(self) -> usize {
        self.sibling_slots
    }
    /// All concrete cleanup TaskNode backing, including its inline callable.
    pub fn task_bytes(self) -> usize {
        self.task_bytes
    }
    /// All empty Data-reference backing across the selected entries.
    pub fn data_slot_bytes(self) -> usize {
        self.data_slot_bytes
    }
    /// At most one nonempty Data allocation per selected entry.
    pub fn data_allocations(self) -> usize {
        self.data_allocations
    }
}
impl OperationEvent {
    /// Pure CPU-producer query available independently of the GPU adapter. The
    /// owning native types supply their layouts; no worker or runtime is created.
    pub fn cpu_eval_cleanup_layout(inputs: usize, siblings: usize) -> Option<CpuEvalCleanupLayout> {
        let mut native = safemlx_sys::mlx_cpu_eval_cleanup_layout::default();
        // SAFETY: scalar query with a valid output; no runtime access or ownership effects.
        unsafe {
            safemlx_sys::mlx_operation_event_cpu_eval_cleanup_layout(&mut native, inputs, siblings)
        }
        .then_some(CpuEvalCleanupLayout { native })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_cleanup_population_keeps_sparse_edges_and_zero_entry_distinct() {
        let layout = OperationEvent::cpu_eval_cleanup_layout(2, 1).expect("owning CPU layout");
        let selected = layout.population(4, 3, 1).expect("four bounded entries");
        assert_eq!(selected.task_bytes(), 4 * layout.requests()[0].0);
        // Four cumulative edges, not the maximum three slots times four entries.
        let one_slot = OperationEvent::cpu_eval_cleanup_layout(1, 0).unwrap();
        assert_eq!(selected.data_slot_bytes(), 4 * one_slot.requests()[1].0);
        assert_eq!(selected.data_allocations(), 4);
        assert!(layout.population(4, 9, 0).is_none());
        assert!(layout.population(4, 0, 5).is_none());
        assert!(layout.population(usize::MAX, 0, 0).is_none());
        let empty = layout.population(0, 0, 0).unwrap();
        assert_eq!(empty.task_bytes(), 0);
        assert_eq!(empty.data_slot_bytes(), 0);
        let zero = OperationEvent::cpu_eval_cleanup_layout(0, 0).unwrap();
        let one_empty_entry = zero.population(1, 0, 0).unwrap();
        assert!(one_empty_entry.task_bytes() > 0);
        assert_eq!(one_empty_entry.data_slot_bytes(), 0);
        assert_eq!(one_empty_entry.data_allocations(), 0);
        assert!(OperationEvent::cpu_eval_cleanup_layout(usize::MAX, 1).is_none());
    }
}
