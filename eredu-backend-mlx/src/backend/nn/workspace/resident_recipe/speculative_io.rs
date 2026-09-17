//! Actual independent output views reduced through the ordinary native recorder.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AutoregressiveReadoutRecipe {
    completion: ResidentCompletionRecipe,
    storage: CertifiedSpanStorage,
    controls: usize,
}
impl AutoregressiveReadoutRecipe {
    pub(crate) fn inspect(
        report: &WorkspaceTraceReport,
        rows: usize,
        positions: usize,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("speculative readout trace is incomplete"));
        // Each row is exactly the shared static Index (Slice plus axis removal).
        // No initializer represents another allocation of the logits source.
        if rows == 0
            || report.operations.len() != rows
            || report
                .operations
                .iter()
                .any(|op| !matches!(op.kind, WorkspaceOperationKind::Index { selected_axes: 1 }))
        {
            return Err(invalid());
        }
        let recorder = ResidentRecipeRecorder::with_context(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: u64::try_from(positions).map_err(|_| invalid())?,
                max_output_tokens: 0,
                prefill_chunk_positions: u64::try_from(positions).map_err(|_| invalid())?,
                output: eredu_core::OutputDemand::Sequence,
            },
            mechanism,
            context,
        )?;
        let reduced = recorder.reduce_trace(report, None, 0, rows)?;
        if reduced.first_missing_operation.is_some()
            || reduced.unqualified_kernel_owner.is_some()
            || reduced.validation_roots != 0
            || reduced.nested_completions != 0
        {
            return Err(invalid());
        }
        let completion = ResidentCompletionRecipe {
            validation_roots: 0,
            grouped_outputs: reduced.grouped_outputs,
            traversal: reduced.traversal.ok_or_else(invalid)?,
            graph: reduced.graph.ok_or_else(invalid)?,
            dispatch: Some(reduced.dispatch.ok_or_else(invalid)?),
            nested_completions: 0,
            nested_root_capacity: 0,
        };
        let controls = reduced.query_controls.ok_or_else(invalid)?;
        Ok(Self {
            completion,
            storage: reduced.mutable_storage.ok_or_else(invalid)?,
            controls,
        })
    }
    pub(crate) fn completion(self) -> ResidentCompletionRecipe {
        self.completion
    }
    pub(crate) fn mutable_storage(self) -> CertifiedSpanStorage {
        self.storage
    }
    pub(crate) fn control_bytes(self) -> usize {
        self.controls
    }
}

impl ResidentCompletionRecipe {
    /// Union two actual programs executed on the same selected GPU stream.
    /// Counts remain additive, including both Synchronizer alternatives and
    /// root lists; this is neither a lifetime maximum nor a new role authority.
    /// CPU entries, when present, remain in the original single CPU stream.
    pub(crate) fn checked_union(self, other: Self) -> Option<Self> {
        self.union_with_controls(other).map(|(value, _)| value)
    }
    fn union_with_controls(self, other: Self) -> Option<(Self, usize)> {
        let a = self.traversal.limits();
        let b = other.traversal.limits();
        if a.streams > 2 || b.streams > 2 {
            return None;
        }
        let graph = safemlx::OperationEvent::resident_graph_layout_with_shells(
            self.graph
                .primitives()
                .checked_add(other.graph.primitives())?,
            self.graph.seeds().checked_add(other.graph.seeds())?,
            self.graph.maximum_rank().max(other.graph.maximum_rank()),
            self.graph
                .maximum_operands()
                .max(other.graph.maximum_operands()),
            self.graph
                .additional_shells()
                .checked_add(other.graph.additional_shells())?,
        )?;
        let traversal = safemlx::OperationEvent::eval_traversal_layout(
            safemlx::OperationEvalTraversalLimits {
                roots: a.roots.checked_add(b.roots)?,
                arrays: a.arrays.checked_add(b.arrays)?,
                tape_entries: a.tape_entries.checked_add(b.tape_entries)?,
                input_edges: a.input_edges.checked_add(b.input_edges)?,
                output_slots: a.output_slots.checked_add(b.output_slots)?,
                streams: a.streams.max(b.streams),
                captures: a.captures.checked_add(b.captures)?,
            },
        )?;
        let (a, b) = (self.dispatch?, other.dispatch?);
        if a.cpu_model.is_some() || b.cpu_model.is_some() { return None; }
        let gpu_entries = a.gpu_entries.checked_add(b.gpu_entries)?;
        let gpu_edges = a.gpu_input_edges.checked_add(b.gpu_input_edges)?;
        let gpu_siblings = a.gpu_siblings.checked_add(b.gpu_siblings)?;
        let gpu_births = a.gpu_births.checked_add(b.gpu_births)?;
        let sorts = a
            .additional_sort_kernels
            .checked_add(b.additional_sort_kernels)?;
        let cpu_entries = a.cpu_entries.checked_add(b.cpu_entries)?;
        let parallel_entries=a.parallel_entries.checked_add(b.parallel_entries)?;
        let parallel_graph_extents=a.parallel_graph_extents.checked_add(b.parallel_graph_extents)?;
        let worker_rank = a.worker_rank.max(b.worker_rank);
        let copy_rank_extents = a.copy_rank_extents.checked_add(b.copy_rank_extents)?;
        let worker = safemlx::OperationEvent::resident_gpu_worker_layout_with_router(
            gpu_entries,
            gpu_edges,
            gpu_siblings,
            traversal.limits().arrays,
            gpu_births,
            worker_rank,
            graph.maximum_operands(),
            sorts,
            cpu_entries.checked_sub(parallel_entries)?,
        )?;
        let dispatch = ResidentDispatchPopulation {
                cpu_model: None,
            gpu_entries: a.gpu_entries.checked_add(b.gpu_entries)?,
            gpu_input_edges: a.gpu_input_edges.checked_add(b.gpu_input_edges)?,
            gpu_siblings,
            gpu_births,
            additional_sort_kernels: sorts,
            cpu_entries: a.cpu_entries.checked_add(b.cpu_entries)?,
            cpu_input_edges: a.cpu_input_edges.checked_add(b.cpu_input_edges)?,
            cpu_siblings: a.cpu_siblings.checked_add(b.cpu_siblings)?,
            parallel_entries,
            parallel_graph_extents,
            worker_graph_extents: worker.allocation_extents().checked_add(copy_rank_extents)?.checked_add(parallel_graph_extents)?,
            worker_rank,
            copy_rank_extents,
            kernel_attempts: worker.kernel_attempts(),
        };
        let controls = [
            graph.control_bytes()?,
            traversal.query_control_bytes()?,
            worker.control_bytes()?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Option<Self>>(),
            std::mem::size_of::<ResidentDispatchPopulation>(),
            std::mem::size_of::<safemlx::OperationEvalTraversalLimits>() * 2,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        Some((
            Self {
                validation_roots: self.validation_roots.checked_add(other.validation_roots)?,
                grouped_outputs: self.grouped_outputs.merge(other.grouped_outputs)?,
                traversal,
                graph,
                dispatch: Some(dispatch),
                nested_completions: self
                    .nested_completions
                    .checked_add(other.nested_completions)?,
                nested_root_capacity: self.nested_root_capacity.max(other.nested_root_capacity),
            },
            controls,
        ))
    }
}

/// Append the already-reduced readout program to the sole actual equation.
/// All fallible combinations precede mutation, preserving the recorded source
/// plan and its exact placeholder credit on refusal.
pub(super) fn bind(
    recipe: &mut ResidentNativeRecipe,
    readout: Option<&AutoregressiveReadoutRecipe>,
) -> Result<(), crate::backend::error::Error> {
    use crate::backend::error::Error as NativeError;
    let invalid = || {
        NativeError::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
    };
    if recipe.records.len() != 1 {
        return Err(invalid());
    }
    let Some(readout) = readout else {
        return Ok(());
    };
    bind_row(&mut recipe.records[0], readout)
}

fn bind_row(
    row: &mut ResidentSpanRecipe,
    readout: &AutoregressiveReadoutRecipe,
) -> Result<(), crate::backend::error::Error> {
    let invalid = || crate::backend::error::Error::PrefillControl(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    );
    if row.first_missing_operation.is_some() || row.unqualified_kernel_owner.is_some() {
        return Err(invalid());
    }
    let current = ResidentCompletionRecipe {
        validation_roots: row.validation_roots,
        grouped_outputs: row.grouped_outputs,
        traversal: row.traversal.ok_or_else(invalid)?,
        graph: row.graph.ok_or_else(invalid)?,
        dispatch: Some(row.dispatch.ok_or_else(invalid)?),
        nested_completions: row.nested_completions,
        nested_root_capacity: 0,
    };
    let (union, union_controls) = current
        .union_with_controls(readout.completion)
        .ok_or_else(invalid)?;
    let storage = row.mutable_storage.ok_or_else(invalid)?;
    let storage = CertifiedSpanStorage {
        mutable_bytes: storage
            .mutable_bytes
            .checked_add(readout.storage.mutable_bytes)
            .ok_or_else(invalid)?,
        maximum_births: storage
            .maximum_births
            .checked_add(readout.storage.maximum_births)
            .ok_or_else(invalid)?,
    };
    let controls = row
        .query_controls
        .and_then(|n| n.checked_add(readout.controls))
        .and_then(|n| n.checked_add(union_controls))
        .ok_or_else(invalid)?;
    let nodes = row
        .host_primitive_nodes
        .checked_add(readout.completion.graph.primitives())
        .ok_or_else(invalid)?;
    row.mutable_storage = Some(storage);
    row.query_controls = Some(controls);
    row.maximum_rank = row
        .maximum_rank
        .max(readout.completion.graph.maximum_rank());
    row.host_primitive_nodes = nodes;
    row.traversal = Some(union.traversal);
    row.graph = Some(union.graph);
    row.dispatch = union.dispatch;
    row.grouped_outputs = union.grouped_outputs;
    row.validation_roots = union.validation_roots;
    row.nested_completions = union.nested_completions;
    Ok(())
}


pub(super) fn bind_prefill_input(
    row: &mut ResidentSpanRecipe,
    report: &WorkspaceTraceReport,
    geometry: InferenceGeometry,
    mechanism: MlxMetalWorkspaceMechanisms,
    context: &WorkspaceContext,
) -> Result<(), crate::backend::error::Error> {
    let invalid = || context.metadata_error(format_args!("speculative prefill input trace is incomplete"));
    if report.operations.len() != 1 || !matches!(report.operations[0].kind,
        WorkspaceOperationKind::Index { selected_axes: 0 }) {
        return Err(invalid().into());
    }
    let recorder = ResidentRecipeRecorder::with_context(geometry, mechanism, context)?;
    let reduced = recorder.reduce_trace(report, None, 0, 1)?;
    if reduced.first_missing_operation.is_some() || reduced.unqualified_kernel_owner.is_some()
        || reduced.validation_roots != 0 || reduced.nested_completions != 0 {
        return Err(invalid().into());
    }
    let fragment = AutoregressiveReadoutRecipe {
        completion: ResidentCompletionRecipe {
            validation_roots: 0,
            grouped_outputs: reduced.grouped_outputs,
            traversal: reduced.traversal.ok_or_else(invalid)?,
            graph: reduced.graph.ok_or_else(invalid)?,
            dispatch: Some(reduced.dispatch.ok_or_else(invalid)?),
            nested_completions: 0,
            nested_root_capacity: 0,
        },
        storage: reduced.mutable_storage.ok_or_else(invalid)?,
        controls: reduced.query_controls.ok_or_else(invalid)?,
    };
    bind_row(row, &fragment)
}
