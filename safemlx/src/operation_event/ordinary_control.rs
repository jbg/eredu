//! Ordinary control allowances from the linked native allocation sources.
use super::{OperationEvalTraversalLimits, OperationEvent};

/// Descriptive allowance for the ordinary physical observer. It creates no
/// native scope, graph, original authority or reservation. The observer charges
/// actual live allocations; unused allowance belongs to the enclosing scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinaryControlPopulation {
    observed_host_controls: usize,
    control_allocations: usize,
    platform_events: usize,
}
impl From<safemlx_sys::mlx_ordinary_control_population> for OrdinaryControlPopulation {
    fn from(value: safemlx_sys::mlx_ordinary_control_population) -> Self {
        Self {
            observed_host_controls: value.observed_host_controls,
            control_allocations: value.control_allocations,
            platform_events: value.platform_events,
        }
    }
}
fn empty() -> safemlx_sys::mlx_ordinary_control_population {
    safemlx_sys::mlx_ordinary_control_population {
        observed_host_controls: 0,
        control_allocations: 0,
        platform_events: 0,
    }
}
impl OrdinaryControlPopulation {
    /// Compose independently selected sources without dropping overflow checks.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            observed_host_controls: self
                .observed_host_controls
                .checked_add(other.observed_host_controls)?,
            control_allocations: self
                .control_allocations
                .checked_add(other.control_allocations)?,
            platform_events: self.platform_events.checked_add(other.platform_events)?,
        })
    }
    /// Host controls to reserve, including each actual allocation's observer
    /// header and its selected constructor transports. This is a prospective
    /// source envelope; it is not measured live residency.
    pub fn observed_host_control_bytes(self) -> usize {
        self.observed_host_controls
    }
    /// Source-derived maximum control-allocation occurrences. Numerical payload
    /// births and caller C shells remain separate.
    pub fn control_allocations(self) -> usize {
        self.control_allocations
    }
    /// Selected platform Event occurrences, whose opaque implementation overhead
    /// remains outside the native allocation capacities reported here.
    pub fn platform_events(self) -> usize {
        self.platform_events
    }
}
impl OperationEvent {
    /// Metal consumer WaitRecord and its retained capture/stream buffers only.
    /// Its event handler and any encoder closure must be included in the actual
    /// GPU dispatch source's consumer-wait population. The Event is borrowed.
    pub fn ordinary_metal_wait_record_control_layout() -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure checked layout query writes one initialized result.
        unsafe { safemlx_sys::mlx_ordinary_metal_wait_record_control_layout(&mut native) }
            .then(|| native.into())
    }

    /// Actual event-backed CPU consumer wait: WaitRecord, its initial capture
    /// and stream buffers, and the selected CPU wait task. The producer Event
    /// is borrowed; an eventless completion incurs none of these allocations.
    pub fn ordinary_cpu_wait_control_layout() -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure checked layout query writes only this initialized value.
        unsafe { safemlx_sys::mlx_ordinary_cpu_wait_control_layout(&mut native) }
            .then(|| native.into())
    }

    /// Shared native descriptor/primitive/operand constructors. Includes neither
    /// original handle slots/routing nor caller-owned ordinary C shells. The
    /// shared constructor reserves at least four operand slots; pass the actual
    /// maximum operand count with that minimum capacity.
    pub fn ordinary_frontend_control_layout(
        entries: usize,
        seeds: usize,
        rank: usize,
        operands: usize,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure checked source arithmetic; only the initialized output is written.
        unsafe {
            safemlx_sys::mlx_ordinary_frontend_control_layout(
                &mut native,
                entries,
                seeds,
                rank,
                operands,
            )
        }
        .then(|| native.into())
    }
    /// Ordinary Eval containers over the selected CPU streams, including actual
    /// cross-stream Fence owners and their wait/update tasks. Metal-feature CPU
    /// producers use the Fence's existing Event worker; no GPU or mixed-device
    /// source is implied. The roots copied into core Eval are included; the C
    /// caller's append-built vector is separate. Ready roots may take the
    /// unchanged no-Eval fast return.
    pub fn ordinary_cpu_eval_control_layout(
        p: OperationEvalTraversalLimits,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure profile/checked population query, without an execution object.
        unsafe {
            safemlx_sys::mlx_ordinary_cpu_eval_control_layout(
                &mut native,
                p.roots,
                p.arrays,
                p.tape_entries,
                p.input_edges,
                p.output_slots,
                p.streams,
                p.captures,
            )
        }
        .then(|| native.into())
    }
    /// Shared ordinary Eval containers and its Synchronizer Event on one Metal
    /// stream. The selected GPU dispatch source must separately cover its async
    /// Event, handlers, encoder, fences and receipts. Caller C roots are separate.
    /// Platform-event count includes both Synchronizer and async Events; their
    /// allocation bytes remain split between these two source queries.
    pub fn ordinary_metal_eval_control_layout(
        p: OperationEvalTraversalLimits,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure checked layout query writes only the initialized output.
        unsafe {
            safemlx_sys::mlx_ordinary_metal_eval_control_layout(
                &mut native,
                p.roots,
                p.arrays,
                p.tape_entries,
                p.input_edges,
                p.output_slots,
                p.streams,
                p.captures,
            )
        }
        .then(|| native.into())
    }
    /// Shared ordinary Eval containers for one Metal model stream and one
    /// source-qualified CPU router stream. Other stream populations refuse.
    /// The selected mixed GPU worker query separately owns the async Event,
    /// both Fence alternatives, CPU tasks, GPU handlers, encoders and receipts.
    /// This query counts their platform Events once, including both slow Fence
    /// alternatives; it neither qualifies the numerical workers nor grants
    /// execution authority.
    pub fn ordinary_metal_router_eval_control_layout(
        p: OperationEvalTraversalLimits,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure checked layout query writes only the initialized output.
        unsafe {
            safemlx_sys::mlx_ordinary_metal_router_eval_control_layout(
                &mut native,
                p.roots,
                p.arrays,
                p.tape_entries,
                p.input_edges,
                p.output_slots,
                p.streams,
                p.captures,
            )
        }
        .then(|| native.into())
    }
    /// Conservative physical-observer allowance from an already qualified
    /// worker's Graph allocation extents. This is the shared allocator mapping:
    /// it does not select a device, qualify a worker or grant Original authority.
    pub fn ordinary_dispatch_control_envelope(
        graph_extents: usize,
    ) -> Option<OrdinaryControlPopulation> {
        Self::ordinary_cpu_dispatch_envelope(graph_extents)
    }
    /// Conservative ordinary allowance from the SAME selected CPU worker's
    /// finite Graph request extent sum. Minimum block extent bounds the number
    /// of observer headers. Unused original framing only enlarges this allowance;
    /// no original source, arena or execution permission follows from it.
    pub fn ordinary_cpu_dispatch_envelope(
        graph_extents: usize,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: checked source arithmetic writes one initialized result.
        unsafe { safemlx_sys::mlx_ordinary_cpu_dispatch_envelope(&mut native, graph_extents) }
            .then(|| native.into())
    }
    /// One fresh core ArrayVector `reserve(elements)` allocation in the linked
    /// libc++ profile. Actual callers must establish this exact reserve worker;
    /// append-built vectors require their separate growth query.
    pub fn ordinary_reserved_array_vector_control_layout(
        elements: usize,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: pure size/profile query writes only the initialized output.
        unsafe {
            safemlx_sys::mlx_ordinary_reserved_array_vector_control_layout(&mut native, elements)
        }
        .then(|| native.into())
    }

    /// The actual C ArrayVector's append growth in the linked libc++ profile.
    /// Excludes its C shell and the distinct exact-size copy into core Eval.
    pub fn ordinary_array_vector_control_layout(
        elements: usize,
    ) -> Option<OrdinaryControlPopulation> {
        let mut native = empty();
        // SAFETY: no vector is constructed; only scalar source arithmetic runs.
        unsafe { safemlx_sys::mlx_ordinary_array_vector_control_layout(&mut native, elements) }
            .then(|| native.into())
    }
}
