//! Finite constructor and evaluation storage for one CPU MXFP4 submission.
use super::{
    CpuBinaryOperation, CpuCopyEvalLayout, CpuMxFp4QuantizeConstructionLayout,
    CpuMxFp4QuantizePayloadLayout, CpuUnaryOperation, OperationEvalTraversalLayout,
    OperationEvalTraversalLimits, OperationEvent,
};
use crate::{Dtype, SubmissionGraphQuota, SubmissionRecordQuota};
use std::mem::{size_of, size_of_val};

/// Storage for quantizing one detached, completed input into two roots on a CPU stream.
/// The input may require compaction. Tracing and retained graph transforms are
/// excluded. Native execution still authenticates the actual input and scope.
///
/// These capacities cover fresh Graph and Record arenas, including constructor,
/// evaluation, exact roots and completion. Arena owners, input custody, physical
/// buffers, runtime/stream/cache setup, failure ownership and encoded host output
/// remain separate contributions. This layout creates no execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuMxFp4QuantizeSubmissionLayout {
    construction: CpuMxFp4QuantizeConstructionLayout,
    payload: CpuMxFp4QuantizePayloadLayout,
    traversal: OperationEvalTraversalLayout,
    graph_capacity: usize,
    record_capacity: usize,
    controls: usize,
}

impl CpuMxFp4QuantizeSubmissionLayout {
    /// Constructor bank to prepare before calling the ordinary quantizer.
    pub fn construction(self) -> CpuMxFp4QuantizeConstructionLayout {
        self.construction
    }
    /// Per-request payloads to price with the selected physical allocator.
    pub fn payload(self) -> CpuMxFp4QuantizePayloadLayout {
        self.payload
    }
    /// Finite traversal for both output roots, including the Synchronizer.
    pub fn traversal(self) -> OperationEvalTraversalLayout {
        self.traversal
    }
    /// Capacity of a fresh Graph arena for this single submission.
    pub fn graph_capacity(self) -> usize {
        self.graph_capacity
    }
    /// Capacity of a fresh Record arena for this single submission.
    pub fn record_capacity(self) -> usize {
        self.record_capacity
    }
    /// Named query and execution transports, separate from arena/buffer owners.
    /// This inventory is not an enforceable whole-stack or process ceiling.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
}

#[derive(Default)]
struct WorkerMaximum {
    graph: usize,
    controls: usize,
}
impl WorkerMaximum {
    fn copy(&mut self, layout: CpuCopyEvalLayout) -> Option<()> {
        self.include(
            layout
                .graph_allocation_extents()
                .checked_add(layout.worker_graph_allocation_extents())?
                .checked_add(layout.signal_graph_allocation_extents())?,
            layout.control_bytes()?,
        );
        Some(())
    }
    fn include(&mut self, graph: usize, controls: usize) {
        self.graph = self.graph.max(graph);
        self.controls = self.controls.max(controls);
    }
}

impl OperationEvent {
    /// Compose existing CPU worker queries with the quantizer's constructor
    /// population. Rows is the product of all leading dimensions. Unsupported
    /// geometry, unavailable native layouts and overflow return `None` before
    /// any array, device, stream, arena or ordinary heap allocation is created.
    pub fn cpu_mxfp4_quantize_submission_layout(
        dtype: Dtype,
        rank: usize,
        rows: usize,
        columns: usize,
    ) -> Option<CpuMxFp4QuantizeSubmissionLayout> {
        let construction =
            Self::cpu_mxfp4_quantize_construction_layout(dtype, rank, rows, columns)?;
        let payload = Self::cpu_mxfp4_quantize_payload_layout(dtype, rows, columns)?;
        let elements = rows.checked_mul(columns)?;
        let distances = elements.checked_mul(16)?;
        let groups = elements / 32;
        let mut worker = WorkerMaximum::default();
        // Only the input and final output reshapes carry the caller's rank.
        // Arithmetic runs on flattened groups or rank-three distance/packing
        // tables. Price compaction without assuming the input is contiguous.
        for layout in [
            Self::cpu_reshape_copy_layout(rank, rank.max(3), false)?,
            Self::cpu_reshape_alias_layout(rank.max(3), rank, false)?,
            Self::cpu_broadcast_alias_layout(3, 3, false)?,
            Self::cpu_expand_dims_alias_layout(2, 3, false)?,
            Self::cpu_squeeze_layout(3, false)?,
            Self::cpu_typed_arg_reduce_layout(dtype, 3, 16, elements, false)?,
            Self::cpu_u32_row_sum_layout(3, 8, elements / 8, false)?,
            Self::cpu_arange_int_layout(Dtype::Uint32, 8, false)?,
        ] {
            worker.copy(layout)?;
        }
        worker.copy(if dtype == Dtype::Float32 {
            Self::cpu_row_max_layout(2, 32, groups, false)?
        } else {
            Self::cpu_half_row_max_layout(dtype, 2, 32, groups, false)?
        })?;
        worker.copy(Self::cpu_typed_select_broadcast_layout(
            dtype, 2, groups, false,
        )?)?;
        for (source, destination) in [
            (Dtype::Float32, dtype),
            (dtype, Dtype::Int32),
            (Dtype::Int32, dtype),
            (dtype, Dtype::Uint8),
        ] {
            worker.copy(Self::cpu_cast_layout(
                source,
                destination,
                3,
                distances,
                false,
            )?)?;
        }
        for operation in [
            CpuBinaryOperation::Add,
            CpuBinaryOperation::Subtract,
            CpuBinaryOperation::Multiply,
            CpuBinaryOperation::Divide,
            CpuBinaryOperation::Power,
            CpuBinaryOperation::Equal,
        ] {
            let layout = Self::cpu_binary_layout(operation, dtype, 3, distances, false)?;
            worker.include(
                layout
                    .graph_allocation_extents()
                    .checked_add(layout.worker_graph_allocation_extents())?,
                layout.control_bytes()?,
            );
        }
        for operation in [CpuBinaryOperation::Multiply, CpuBinaryOperation::Power] {
            let layout = Self::cpu_binary_layout(operation, Dtype::Uint32, 3, elements, false)?;
            worker.include(
                layout
                    .graph_allocation_extents()
                    .checked_add(layout.worker_graph_allocation_extents())?,
                layout.control_bytes()?,
            );
        }
        for operation in [
            CpuUnaryOperation::Log2,
            CpuUnaryOperation::Absolute,
            CpuUnaryOperation::Round,
        ] {
            let layout = Self::cpu_unary_layout(operation, dtype, 3, false)?;
            worker.include(
                layout
                    .graph_allocation_extents()
                    .checked_add(layout.worker_graph_allocation_extents())?,
                layout.control_bytes()?,
            );
        }
        // Count frontend candidates even when identity casts/views are elided.
        // Select has at most three inputs; each primitive has a single output.
        let candidates = construction.graph().primitives();
        let tape = candidates.checked_add(1)?;
        let traversal = Self::eval_traversal_layout(OperationEvalTraversalLimits {
            roots: 2,
            arrays: candidates
                .checked_add(construction.graph().seeds())?
                .checked_add(2)?,
            tape_entries: tape,
            input_edges: candidates.checked_mul(3)?.checked_add(2)?,
            output_slots: tape,
            streams: 1,
            // Eight base slots plus three input/one output weak Data owners,
            // one fresh output and one General-copy's three Data owners.
            captures: 16,
        })?;
        let completion = Self::cpu_completion_layout(2)?;
        let synchronizer = Self::resident_graph_layout(1, 0, 3)?;
        let roots = Self::root_storage_layout(2)?;
        let graph_extents = construction
            .graph()
            .allocation_extents()
            .checked_add(synchronizer.allocation_extents())?
            .checked_add(roots.graph_request_extent())?
            .checked_add(worker.graph.checked_mul(candidates)?)?
            .checked_add(completion.graph_allocation_extents())?
            .checked_add(completion.worker_graph_allocation_extents())?
            .checked_add(completion.signal_graph_allocation_extents())?;
        let graph_capacity = SubmissionGraphQuota::fresh_capacity_for_extents(graph_extents)?;
        let record_capacity = SubmissionRecordQuota::fresh_capacity_for_extents(
            traversal.record_allocation_extents()?,
        )?;
        let frames = [
            size_of::<CpuMxFp4QuantizeSubmissionLayout>(),
            size_of::<Option<CpuMxFp4QuantizeSubmissionLayout>>(),
            size_of::<WorkerMaximum>(),
            size_of::<[CpuCopyEvalLayout; 8]>(),
            size_of::<[(Dtype, Dtype); 4]>(),
            size_of::<[CpuBinaryOperation; 6]>(),
            size_of::<[CpuUnaryOperation; 3]>(),
            size_of::<usize>() * 12,
            construction.control_bytes()?,
            payload.control_bytes()?,
            worker.controls.checked_mul(candidates)?,
            completion.control_bytes()?,
            synchronizer.control_bytes()?,
            traversal.query_control_bytes()?,
            Self::non_object_control_bytes()?,
            SubmissionGraphQuota::fit_query_control_bytes()?,
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?;
        Some(CpuMxFp4QuantizeSubmissionLayout {
            construction,
            payload,
            traversal,
            graph_capacity,
            record_capacity,
            controls,
        })
    }
}
