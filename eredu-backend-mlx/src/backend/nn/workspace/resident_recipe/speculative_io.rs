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
        mechanism: super::super::ResidentExecutionMechanisms,
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
        let recorder = mechanism.recorder(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: u64::try_from(positions).map_err(|_| invalid())?,
                max_output_tokens: 0,
                prefill_chunk_positions: u64::try_from(positions).map_err(|_| invalid())?,
                output: eredu_core::OutputDemand::Sequence,
            },
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
    /// Construction facts for the actual captured final-row Index, without
    /// inventing another trace or a second completion frontier.
    pub(crate) fn index_construction(
        operation: &WorkspaceOperation,
        mechanism: super::super::ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<safemlx::ResidentGraphLayout, Error> {
        let invalid =
            || context.metadata_error(format_args!("captured readout Index source is incomplete"));
        if !matches!(
            operation.kind,
            WorkspaceOperationKind::Index { selected_axes: 1 }
        ) || operation.inputs.len() != 1
            || operation.outputs.len() != 1
            || operation.inputs[0].shape().len() != 3
            || operation.outputs[0].shape().len() != 2
        {
            return Err(invalid());
        }
        let (entries, seeds, rank, operands, shells, source_controls) = match mechanism {
            super::super::ResidentExecutionMechanisms::Cpu { cpu, .. } => {
                let plan = cpu
                    .plan(operation.as_view())
                    .map_err(|cause| context.metadata_source(cause))?
                    .ok_or_else(invalid)?;
                (
                    plan.population.construction_entries,
                    plan.seeds,
                    plan.rank,
                    plan.population.maximum_operands.max(4),
                    plan.parameter_shells,
                    plan.population.controls,
                )
            }
            super::super::ResidentExecutionMechanisms::Metal(_) => {
                let plan = lowering(operation.as_view()).ok_or_else(invalid)?;
                let rank = operation
                    .inputs
                    .iter()
                    .chain(&operation.outputs)
                    .map(|value| value.shape().len())
                    .max()
                    .unwrap_or(0)
                    .max(plan.intermediate_rank);
                (
                    plan.primitives,
                    plan.seeds,
                    rank,
                    plan.maximum_operands.max(4),
                    backend_handle_shells(operation.as_view(), plan).ok_or_else(invalid)?,
                    0,
                )
            }
        };
        let graph = safemlx::OperationEvent::resident_graph_layout_with_shells(
            entries, seeds, rank, operands, shells,
        )
        .ok_or_else(invalid)?;
        context.charge_metadata(
            std::mem::size_of::<(
                safemlx::ResidentGraphLayout,
                Option<safemlx::ResidentGraphLayout>,
                Result<safemlx::ResidentGraphLayout, Error>,
                [usize; 6],
            )>()
            .checked_add(source_controls)
            .and_then(|n| n.checked_add(graph.control_bytes()?))
            .ok_or_else(invalid)?,
        )?;
        Ok(graph)
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
    /// Compose programs for the same selected stream and enclosing completion.
    /// Native workers and root lists remain additive; the CPU census excludes
    /// each fragment's private Synchronizer and adds the one combined frontier.
    /// Mixing CPU model and GPU model populations is not a valid source loan.
    pub(crate) fn checked_union(self, other: Self) -> Option<Self> {
        self.union_with_controls(other).map(|(value, _)| value)
    }
    fn union_with_controls(self, other: Self) -> Option<(Self, usize)> {
        let a = self.traversal.limits();
        let b = other.traversal.limits();
        if a.streams > 2 || b.streams > 2 {
            return None;
        }
        let cpu = match (self.dispatch?.cpu_model, other.dispatch?.cpu_model) {
            (Some(mut a), Some(b)) => {
                a.add(b)?;
                Some(a)
            }
            (None, None) => None,
            _ => return None,
        };
        let shared_synchronizer = usize::from(cpu.is_some());
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
                tape_entries: a
                    .tape_entries
                    .checked_add(b.tape_entries)?
                    .checked_sub(shared_synchronizer)?,
                input_edges: a.input_edges.checked_add(b.input_edges)?,
                output_slots: a
                    .output_slots
                    .checked_add(b.output_slots)?
                    .checked_sub(shared_synchronizer)?,
                streams: a.streams.max(b.streams),
                captures: a.captures.checked_add(b.captures)?,
            },
        )?;
        let (a, b) = (self.dispatch?, other.dispatch?);
        if let Some(cpu) = cpu {
            if a.completion_streams()? != self.traversal.limits().streams
                || b.completion_streams()? != other.traversal.limits().streams
            {
                return None;
            }
            let completion = safemlx::OperationEvent::cpu_completion_layout(traversal.roots())?;
            if completion.backing_births() != 0 || completion.worker_graph_allocation_extents() != 0
            {
                return None;
            }
            let dispatch = ResidentDispatchPopulation {
                cpu_model: Some(cpu),
                cpu_entries: a.cpu_entries.checked_add(b.cpu_entries)?.checked_sub(1)?,
                cpu_input_edges: a.cpu_input_edges.checked_add(b.cpu_input_edges)?,
                cpu_siblings: a.cpu_siblings.checked_add(b.cpu_siblings)?.checked_sub(1)?,
                parallel_entries: a.parallel_entries.checked_add(b.parallel_entries)?,
                parallel_graph_extents: a
                    .parallel_graph_extents
                    .checked_add(b.parallel_graph_extents)?,
                worker_rank: a.worker_rank.max(b.worker_rank),
                ..a
            };
            if dispatch.completion_streams()? != traversal.limits().streams {
                return None;
            }
            let controls = [
                graph.control_bytes()?,
                traversal.query_control_bytes()?,
                completion.control_bytes()?,
                size_of::<super::super::cpu::CpuPopulation>() * 3,
                size_of::<Self>(),
                size_of::<Option<Self>>(),
                size_of::<ResidentDispatchPopulation>(),
                size_of::<safemlx::OperationEvalTraversalLimits>() * 2,
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)?;
            return Some((
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
            ));
        }
        let gpu_entries = a.gpu_entries.checked_add(b.gpu_entries)?;
        let gpu_edges = a.gpu_input_edges.checked_add(b.gpu_input_edges)?;
        let gpu_siblings = a.gpu_siblings.checked_add(b.gpu_siblings)?;
        let gpu_births = a.gpu_births.checked_add(b.gpu_births)?;
        let sorts = a
            .additional_sort_kernels
            .checked_add(b.additional_sort_kernels)?;
        let cpu_entries = a.cpu_entries.checked_add(b.cpu_entries)?;
        let parallel_entries = a.parallel_entries.checked_add(b.parallel_entries)?;
        let parallel_graph_extents = a
            .parallel_graph_extents
            .checked_add(b.parallel_graph_extents)?;
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
            worker_graph_extents: worker
                .allocation_extents()
                .checked_add(copy_rank_extents)?
                .checked_add(parallel_graph_extents)?,
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
    let invalid = || {
        crate::backend::error::Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )
    };
    if row.first_missing_operation.is_some() || row.unqualified_kernel_owner.is_some() {
        #[cfg(test)]
        eprintln!(
            "AR source row missing={:?} detail={:?} kernel={:?}",
            row.first_missing_operation, row.missing_operation_detail, row.unqualified_kernel_owner
        );
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
    mechanism: super::super::ResidentExecutionMechanisms,
    context: &WorkspaceContext,
) -> Result<(), crate::backend::error::Error> {
    let invalid = || {
        context.metadata_error(format_args!(
            "speculative prefill input trace is incomplete"
        ))
    };
    if report.operations.len() != 1
        || !matches!(
            report.operations[0].kind,
            WorkspaceOperationKind::StaticSlice { .. }
        )
    {
        return Err(invalid().into());
    }
    let recorder = mechanism.recorder(geometry, context)?;
    let reduced = recorder.reduce_trace(report, None, 0, 1)?;
    if reduced.first_missing_operation.is_some()
        || reduced.unqualified_kernel_owner.is_some()
        || reduced.validation_roots != 0
        || reduced.nested_completions != 0
    {
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

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "speculative_io_tests.rs"]
mod tests;
