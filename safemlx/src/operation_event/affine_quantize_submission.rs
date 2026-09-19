//! Composed CPU affine quantization and optional companion conversion storage.
use super::{
    AffineQuantizeConstructionLayout, CpuCopyEvalLayout, OperationEvalTraversalLayout,
    OperationEvalTraversalLimits, OperationEvent, ResidentGraphLayout,
};
use crate::{
    Dtype, OriginalBufferBudget, OriginalBufferCause, PreparedInputRuntime, SubmissionGraphQuota,
    SubmissionRecordQuota,
};
use std::mem::{size_of, size_of_val};

/// Fresh storage for one CPU affine quantization of a completed, detached input.
/// Scales and biases may be converted to another floating-point dtype. Possible
/// input compaction is included without assuming donation or early retirement.
///
/// Input custody, arena owners, runtime/stream/cache preparation, failure owners
/// and encoded host destinations are separate. This query creates no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuAffineQuantizeSubmissionLayout {
    construction: AffineQuantizeConstructionLayout,
    companions: Option<ResidentGraphLayout>,
    traversal: OperationEvalTraversalLayout,
    requests: [usize; 6],
    request_count: usize,
    outputs: [usize; 3],
    graph_capacity: usize,
    record_capacity: usize,
    controls: usize,
}
impl CpuAffineQuantizeSubmissionLayout {
    /// Constructor bank for packed weights and their native scale/bias siblings.
    pub fn construction(self) -> AffineQuantizeConstructionLayout {
        self.construction
    }
    /// Two cast constructors when the requested companion dtype differs.
    /// Prepare after dropping the affine bank; drop before evaluation.
    pub fn companion_construction(self) -> Option<ResidentGraphLayout> {
        self.companions
    }
    /// Finite traversal for all three final roots, including the Synchronizer.
    pub fn traversal(self) -> OperationEvalTraversalLayout {
        self.traversal
    }
    /// Individual possible allocations: input compaction, three native outputs,
    /// followed by both converted companions when their dtype changes.
    pub fn request_bytes(&self) -> &[usize] {
        &self.requests[..self.request_count]
    }
    /// Final packed weights, scales and biases, in that order.
    pub fn output_bytes(self) -> [usize; 3] {
        self.outputs
    }
    /// Fresh Graph arena capacity for construction, workers, roots and completion.
    pub fn graph_capacity(self) -> usize {
        self.graph_capacity
    }
    /// Fresh Record arena capacity for the selected traversal.
    pub fn record_capacity(self) -> usize {
        self.record_capacity
    }
    /// Price every potential allocation using the selected physical allocator.
    pub fn physical_capacity(
        &self,
        runtime: &PreparedInputRuntime,
    ) -> Result<usize, OriginalBufferCause> {
        self.request_bytes()
            .iter()
            .try_fold(0usize, |total, &request| {
                total
                    .checked_add(OriginalBufferBudget::request_layout(runtime, request)?.capacity())
                    .ok_or(OriginalBufferCause::InvalidLayout)
            })
    }
    /// Named transports for the query and its selected native operations.
    /// This inventory is not a dependency-wide or process-wide ceiling.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
}
fn floating_bytes(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::Float16 | Dtype::Bfloat16 => Some(2),
        Dtype::Float32 => Some(4),
        _ => None,
    }
}
fn eval_extents(layout: CpuCopyEvalLayout) -> Option<usize> {
    layout
        .graph_allocation_extents()
        .checked_add(layout.worker_graph_allocation_extents())?
        .checked_add(layout.signal_graph_allocation_extents())
}
impl OperationEvent {
    /// Compose affine construction, optional companion casts and existing CPU
    /// workers. Rows is the product of leading dimensions. Scalar validation
    /// rejects unsupported geometry and overflow before creating native resources.
    pub fn cpu_affine_quantize_submission_layout(
        dtype: Dtype,
        companion_dtype: Dtype,
        rank: usize,
        rows: usize,
        columns: usize,
        group_size: i32,
        bits: i32,
    ) -> Option<CpuAffineQuantizeSubmissionLayout> {
        let scalar = floating_bytes(dtype)?;
        let companion_scalar = floating_bytes(companion_dtype)?;
        let construction = Self::affine_quantize_construction_layout(rank)?;
        let quantize = Self::cpu_affine_quantize_layout(
            dtype, rank, rows, columns, group_size, bits, true, false,
        )?;
        // The native worker validated nonzero groups and all packing geometry.
        let elements = rows.checked_mul(columns)?;
        let groups = elements.checked_div(usize::try_from(group_size).ok()?)?;
        let packed = elements
            .checked_mul(usize::try_from(bits).ok()?)?
            .checked_div(8)?;
        let native_companion = groups.checked_mul(scalar)?;
        let final_companion = groups.checked_mul(companion_scalar)?;
        let mut requests = [0; 6];
        requests[..4].copy_from_slice(&[
            elements.checked_mul(scalar)?,
            packed,
            native_companion,
            native_companion,
        ]);
        let casts = usize::from(dtype != companion_dtype).checked_mul(2)?;
        let (companions, cast_graph, cast_controls) = if casts == 0 {
            (None, 0, 0)
        } else {
            requests[4..].fill(final_companion);
            let bank = Self::resident_graph_layout(casts, 0, rank)?;
            let worker = Self::cpu_cast_layout(dtype, companion_dtype, rank, groups, false)?;
            (
                Some(bank),
                bank.allocation_extents()
                    .checked_add(eval_extents(worker)?.checked_mul(casts)?)?,
                bank.control_bytes()?
                    .checked_add(worker.control_bytes()?.checked_mul(casts)?)?,
            )
        };
        let traversal = Self::eval_traversal_layout(OperationEvalTraversalLimits {
            roots: 3,
            arrays: 5 + casts, // input, three siblings, casts, Synchronizer
            tape_entries: 2 + casts,
            input_edges: 4 + casts,  // affine input, cast inputs, three roots
            output_slots: 4 + casts, // three siblings, casts, Synchronizer
            streams: 1,
            // Base captures, weak input/three siblings and possible compaction.
            captures: 16,
        })?;
        let completion = Self::cpu_completion_layout(3)?;
        let synchronizer = Self::resident_graph_layout(1, 0, 3)?;
        let roots = Self::root_storage_layout(3)?;
        let graph_extents = construction
            .graph_allocation_extents()
            .checked_add(eval_extents(quantize)?)?
            .checked_add(cast_graph)?
            .checked_add(synchronizer.allocation_extents())?
            .checked_add(roots.graph_request_extent())?
            .checked_add(eval_extents(completion)?)?;
        let graph_capacity = SubmissionGraphQuota::fresh_capacity_for_extents(graph_extents)?;
        let record_capacity = SubmissionRecordQuota::fresh_capacity_for_extents(
            traversal.record_allocation_extents()?,
        )?;
        let frames = [
            size_of::<CpuAffineQuantizeSubmissionLayout>(),
            size_of::<Option<CpuAffineQuantizeSubmissionLayout>>(),
            size_of::<[usize; 6]>(),
            size_of::<[usize; 4]>(),
            size_of::<Dtype>() * 2,
            size_of::<i32>() * 2,
            size_of::<usize>() * 16,
            size_of::<Result<usize, OriginalBufferCause>>(),
            construction.control_bytes()?,
            quantize.control_bytes()?,
            cast_controls,
            completion.control_bytes()?,
            synchronizer.control_bytes()?,
            traversal.query_control_bytes()?,
            Self::non_object_control_bytes()?,
            SubmissionGraphQuota::fit_query_control_bytes()?,
            OriginalBufferBudget::request_layout_control_bytes()?.checked_mul(4 + casts)?,
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?;
        Some(CpuAffineQuantizeSubmissionLayout {
            construction,
            companions,
            traversal,
            requests,
            request_count: 4 + casts,
            outputs: [packed, final_companion, final_companion],
            graph_capacity,
            record_capacity,
            controls,
        })
    }
}
