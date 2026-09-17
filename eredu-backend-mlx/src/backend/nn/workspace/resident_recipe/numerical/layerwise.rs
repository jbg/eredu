//! Shared original Host-copy and replaced-unit constructor populations.
use super::*;
use super::super::host_copies::PreparedSourceCopies;
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;
use crate::backend::runtime::residency::manager::WindowPopulation;
use safemlx::OperationEvent;

impl SpeculativeNumericalRecipe {
    pub(super) fn with_prepared_source_copies(self, source:PreparedSourceCopies,
        context:&WorkspaceContext)->Result<Self,Error> {
        let invalid=||context.metadata_error(format_args!(
            "retained transfer source differs from its actual numerical population"));
        context.charge_metadata(std::mem::size_of::<(
            Self,PreparedSourceCopies,ResidentCompletionRecipe,ResidentDispatchPopulation,
            safemlx::OperationEvalTraversalLimits,Result<Self,Error>,[usize;8]
        )>())?;
        let mut result = self;
        let mut completion = result.completion;
        let copies = source.copies;
        let transfers = source.transfers;
        let count = copies.per_forward;
        let mut limits = completion.traversal.limits();
        limits.arrays = limits.arrays.checked_add(count.checked_mul(2).ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        limits.input_edges = limits.input_edges.checked_add(count).ok_or_else(invalid)?;
        completion.traversal = OperationEvent::eval_traversal_layout(limits).ok_or_else(invalid)?;
        let graph = completion.graph;
        completion.graph = OperationEvent::resident_graph_layout_with_shells(
            graph.primitives().checked_add(count).ok_or_else(invalid)?,
            graph.seeds().checked_add(count).ok_or_else(invalid)?,
            graph.maximum_rank().max(source.rank), graph.maximum_operands().max(4),
            graph.additional_shells().checked_add(transfers.binding_shells).ok_or_else(invalid)?,
        ).ok_or_else(invalid)?;
        let mut dispatch = completion.dispatch.ok_or_else(invalid)?;
        if dispatch.cpu_model.is_some() { return Err(invalid()); }
        dispatch.worker_rank = dispatch.worker_rank.max(source.rank);
        let worker = OperationEvent::resident_gpu_worker_layout_with_router(
            dispatch.gpu_entries, dispatch.gpu_input_edges, dispatch.gpu_siblings,
            limits.arrays, dispatch.gpu_births, dispatch.worker_rank,
            completion.graph.maximum_operands(), dispatch.additional_sort_kernels,
            dispatch.cpu_entries.checked_sub(dispatch.parallel_entries).ok_or_else(invalid)?,
        ).ok_or_else(invalid)?;
        dispatch.worker_graph_extents = worker.allocation_extents()
            .checked_add(dispatch.copy_rank_extents)
            .and_then(|n|n.checked_add(dispatch.parallel_graph_extents)).ok_or_else(invalid)?;
        dispatch.kernel_attempts = worker.kernel_attempts();
        completion.dispatch = Some(dispatch);
        // Equation completions retain their own finite root distribution.
        // Transfers have separate exact Eval/Wait producers in the same arena.
        let mut graph = graph_capacity::ResidentGraphStorage::for_completion(completion)
            .ok_or_else(invalid)?;
        graph.include_source_copies(copies, transfers, dispatch).ok_or_else(invalid)?;
        let records = record_capacity::ResidentRecordStorage::for_completion_sources(
            completion, 0, Some(source)).ok_or_else(invalid)?;
        let equation_frontiers = if dispatch.cpu_entries == 0 { 1 } else {
            completion.nested_completions.checked_add(1).ok_or_else(invalid)? };
        completion.nested_completions = completion.nested_completions.checked_add(count)
            .and_then(|n|n.checked_add(transfers.per_forward)).ok_or_else(invalid)?;
        completion.nested_root_capacity = completion.nested_root_capacity.max(transfers.roots);
        result.completion = completion;
        result.storage.mutable_bytes = result.storage.mutable_bytes.checked_add(transfers.bytes)
            .ok_or_else(invalid)?;
        result.storage.maximum_births = result.storage.maximum_births.checked_add(count)
            .ok_or_else(invalid)?;
        result.graph_capacity = usize::try_from(graph.full_capacity.ok_or_else(invalid)?)
            .map_err(|_|invalid())?;
        result.record_capacity = usize::try_from(records.full_capacity.ok_or_else(invalid)?)
            .map_err(|_|invalid())?;
        result.kernels = dispatch.kernel_attempts
            .checked_mul(equation_frontiers)
            .and_then(|n|n.checked_add(copies.dispatch.kernel_attempts.checked_mul(count)?))
            .and_then(|n|n.checked_add(copies.aggregate_dispatch.kernel_attempts.checked_mul(transfers.per_forward)?))
            .ok_or_else(invalid)?;
        result.controls = result.controls.checked_add(u64::try_from(source.controls)
            .map_err(|_|invalid())?).and_then(|n|n.checked_add(graph.control_bytes()?))
            .and_then(|n|n.checked_add(u64::try_from(worker.control_bytes()?).ok()?))
            .ok_or_else(invalid)?;
        Ok(result)
    }

    /// The source is the same retained selected parameter/copy inventory used
    /// by the operation bank. Windows describe actual finite warm/missing
    /// attempts; no text-forward geometry or synthetic parameter provider is
    /// introduced for a realtime frame.
    pub(crate) fn with_layerwise_source(self,source:&LayerwiseWorkspace,
        windows:&[WindowPopulation],context:&WorkspaceContext)->Result<Self,Error> {
        let invalid=||context.metadata_error(format_args!("retained layerwise source is incomplete"));
        context.charge_metadata(PreparedSourceCopies::inspection_control_bytes().ok_or_else(invalid)?)?;
        let copies=PreparedSourceCopies::inspect_layerwise(source,windows)
            .map_err(|cause|context.metadata_source(cause))?;
        let (constructors,query)=super::super::parameter_construction::constructor_source(
            source,context.metadata_funding().as_ref()).map_err(|cause|context.metadata_source(cause))?;
        let (additional,controls)=constructors.native_source(
            if context.metadata_funding().is_some(){0}else{query})
            .map_err(|cause|context.metadata_source(cause))?;
        let mut result=self;
        let graph=result.completion.graph;
        result.completion.graph=OperationEvent::resident_graph_layout_with_shells(
            graph.primitives().checked_add(additional.primitives()).ok_or_else(invalid)?,
            graph.seeds().checked_add(additional.seeds()).ok_or_else(invalid)?,
            graph.maximum_rank().max(additional.maximum_rank()),
            graph.maximum_operands().max(additional.maximum_operands()),graph.additional_shells()
        ).ok_or_else(invalid)?;
        result.storage.mutable_bytes=result.storage.mutable_bytes.checked_add(constructors.scalar_bytes)
            .ok_or_else(invalid)?;
        result.storage.maximum_births=result.storage.maximum_births.checked_add(constructors.slots)
            .ok_or_else(invalid)?;
        result.controls=result.controls.checked_add(u64::try_from(controls).map_err(|_|invalid())?)
            .ok_or_else(invalid)?;
        // Constructors are replaced before Eval. Only the common copy reducer
        // extends the real traversal, native copy workers and completion bank.
        result.with_prepared_source_copies(copies,context)
    }
}
