use super::OperationEvent;

/// Qualified GPU host prologue and empty invocation-vector destinations.
/// Backing remains in the existing role Graph arena. Output Array handles are
/// included; output Data, other primitive work and JIT remain separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuEvalPrologueLayout {
    native: safemlx_sys::mlx_gpu_eval_prologue_layout,
}
impl GpuEvalPrologueLayout {
    /// Actual requests: counted control, handler, Data slots, output Array slots,
    /// and optional tracer Array slots. Empty entries have zero bytes.
    pub fn requests(self) -> [(usize, usize); 5] {
        std::array::from_fn(|i| {
            (
                self.native.request_bytes[i],
                self.native.request_alignments[i],
            )
        })
    }
    /// Exact shared bank header and pointer-slot requests.
    pub fn owner_requests(self) -> [(usize, usize); 2] {
        [
            (self.native.header_bytes, self.native.header_alignment),
            (self.native.slots_bytes, self.native.slots_alignment),
        ]
    }
    /// Number of payload requests, excluding the bank header and pointer slots.
    pub fn blocks(self) -> usize {
        self.native.blocks
    }
    /// Total requested Graph storage, including the bank header and pointer slots.
    pub fn requested_bytes(self) -> usize {
        self.native.requested_bytes
    }
    /// Per-request extents are descriptive; actual preflight handles fragmentation.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Named query/control transports, excluding the separately owned Graph blocks.
    pub fn control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_gpu_eval_prologue_layout>(),
            size_of::<*mut safemlx_sys::mlx_gpu_eval_prologue_layout>(),
            4 * size_of::<usize>(), // both descriptive query entry points
            2 * size_of::<bool>(),
            size_of::<GpuEvalProloguePopulation>(),
            size_of::<Option<GpuEvalProloguePopulation>>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
    /// Finite selected populations, including Synchronizer root edges. Fixed
    /// controls can survive every entry until callbacks finish; vector payload
    /// uses the cumulative edge/sibling count, not max_slots * evaluations.
    pub fn population(
        self,
        evaluations: usize,
        input_edges: usize,
        sibling_slots: usize,
    ) -> Option<GpuEvalProloguePopulation> {
        if input_edges > evaluations.checked_mul(self.native.inputs)?
            || sibling_slots > evaluations.checked_mul(self.native.siblings)?
        {
            return None;
        }
        let data_slots = input_edges.checked_add(sibling_slots)?;
        let fixed_payload_bytes = self.native.request_bytes[0]
            .checked_add(self.native.request_bytes[1])?
            .checked_mul(evaluations)?;
        let data_slot_bytes = data_slots.checked_mul(self.native.data_slot_bytes)?;
        let output_array_slots = evaluations.checked_add(sibling_slots)?;
        let tracer_array_slots = if self.native.tracer_inputs == self.native.inputs {
            input_edges
        } else {
            0
        };
        let output_array_bytes = output_array_slots.checked_mul(self.native.array_slot_bytes)?;
        let tracer_array_bytes = tracer_array_slots.checked_mul(self.native.array_slot_bytes)?;
        Some(GpuEvalProloguePopulation {
            maximum: self,
            evaluations,
            input_edges,
            sibling_slots,
            fixed_payload_bytes,
            data_slot_bytes,
            vector_allocations: evaluations.min(data_slots),
            output_array_slots,
            tracer_array_slots,
            output_array_bytes,
            tracer_array_bytes,
            invocation_vector_allocations: evaluations
                .checked_add(evaluations.min(tracer_array_slots))?,
        })
    }
}

/// Descriptive finite overlap; neither arena authority nor a free-byte fit test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuEvalProloguePopulation {
    maximum: GpuEvalPrologueLayout,
    evaluations: usize,
    input_edges: usize,
    sibling_slots: usize,
    fixed_payload_bytes: usize,
    data_slot_bytes: usize,
    vector_allocations: usize,
    output_array_slots: usize,
    tracer_array_slots: usize,
    output_array_bytes: usize,
    tracer_array_bytes: usize,
    invocation_vector_allocations: usize,
}
impl GpuEvalProloguePopulation {
    /// Maximum per-invocation layout used to bound this selected population.
    pub fn maximum(self) -> GpuEvalPrologueLayout {
        self.maximum
    }
    /// Number of selected primitive evaluations, including the Synchronizer.
    pub fn evaluations(self) -> usize {
        self.evaluations
    }
    /// Cumulative input edges, including root edges exactly once.
    pub fn input_edges(self) -> usize {
        self.input_edges
    }
    /// Cumulative sibling slots across the selected evaluations.
    pub fn sibling_slots(self) -> usize {
        self.sibling_slots
    }
    /// Counted-completion and handler payload storage across all evaluations.
    pub fn fixed_payload_bytes(self) -> usize {
        self.fixed_payload_bytes
    }
    /// Cumulative Data-reference slot storage, excluding allocator bookkeeping.
    pub fn data_slot_bytes(self) -> usize {
        self.data_slot_bytes
    }
    /// At most one nonempty vector per entry. Actual per-entry requests are
    /// recomputed from the genuine current array before physical reservation.
    pub fn vector_allocations(self) -> usize {
        self.vector_allocations
    }
    /// One output handle per entry, plus actual sibling handles.
    pub fn output_array_slots(self) -> usize {
        self.output_array_slots
    }
    /// Conservative selected tracer-copy slots, distinct from retained Data.
    pub fn tracer_array_slots(self) -> usize {
        self.tracer_array_slots
    }
    /// Cumulative output Array-handle backing, separate from output Data.
    pub fn output_array_bytes(self) -> usize {
        self.output_array_bytes
    }
    /// Cumulative tracer Array-handle backing under the selected tracing ceiling.
    pub fn tracer_array_bytes(self) -> usize {
        self.tracer_array_bytes
    }
    /// Output-vector attempts plus the finite nonempty tracer-vector ceiling.
    pub fn invocation_vector_allocations(self) -> usize {
        self.invocation_vector_allocations
    }
}
impl OperationEvent {
    /// Pure compiled-adapter query. CPU/no-GPU/CUDA return None for this Metal
    /// producer; their existing evaluation and ordinary paths are unchanged.
    pub fn gpu_eval_prologue_layout(
        inputs: usize,
        siblings: usize,
    ) -> Option<GpuEvalPrologueLayout> {
        Self::gpu_eval_prologue_layout_with_tracing(inputs, siblings, false)
    }
    /// Pure variant for a possible tracer invocation. This flag prices a
    /// destination; the native fixed producer independently reads its array.
    pub fn gpu_eval_prologue_layout_with_tracing(
        inputs: usize,
        siblings: usize,
        tracer: bool,
    ) -> Option<GpuEvalPrologueLayout> {
        let mut native = safemlx_sys::mlx_gpu_eval_prologue_layout::default();
        // SAFETY: fixed scalar query; valid output, no runtime or owner effects.
        unsafe {
            safemlx_sys::mlx_operation_event_gpu_eval_prologue_layout_with_tracing(
                &mut native,
                inputs,
                siblings,
                tracer,
            )
        }
        .then_some(GpuEvalPrologueLayout { native })
    }
}
