//! Native lowering populations consumed by the prepared Eval destination.
//! The source is the same borrowed operation stream used by workspace facts.
use super::*;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PointwiseTraversalFactError {
    #[error(transparent)]
    Fact(#[from] MlxWorkspaceFactError),
    #[error("selected operation has no closed pointwise traversal producer")]
    UnknownOperation,
    #[error("pointwise traversal population overflow")]
    Overflow,
    #[error("native prepared traversal layout is not qualified")]
    UnknownQualification,
}

impl MlxMetalWorkspaceMechanisms {
    pub(crate) fn pointwise_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<facts::Emitter<'static>>(),
            size_of::<PointwiseTraversalFactError>(),
            size_of::<WorkspaceOperationView<'static>>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<Option<safemlx::OperationEvalRecordLayout>>(),
            size_of::<Option<safemlx::OperationEvalTraversalLayout>>(),
            size_of::<
                Result<
                    (
                        safemlx::OperationEvalTraversalLayout,
                        safemlx::PointwiseGraphLayout,
                        safemlx::OperationEvalRecordLayout,
                        safemlx::GpuEvalProloguePopulation,
                        usize,
                    ),
                    PointwiseTraversalFactError,
                >,
            >(),
            size_of::<safemlx::PointwiseGraphLayout>(),
            size_of::<Option<safemlx::PointwiseGraphLayout>>(),
            size_of::<[usize; 5]>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// No input/source authority is created here. The owning submission module
    /// supplies actual borrowed completed leaves and binds this result before
    /// graph construction. No iterator size hint controls allocation or credit.
    pub(crate) fn pointwise_traversal_layout<'a>(
        &self,
        operations: impl IntoIterator<Item = WorkspaceOperationView<'a>>,
        leaves: usize,
        roots: usize,
    ) -> Result<
        (
            safemlx::OperationEvalTraversalLayout,
            safemlx::PointwiseGraphLayout,
            safemlx::OperationEvalRecordLayout,
            safemlx::GpuEvalProloguePopulation,
            usize,
        ),
        PointwiseTraversalFactError,
    > {
        use PointwiseTraversalFactError as E;
        let mut arrays = leaves.checked_add(1).ok_or(E::Overflow)?;
        let mut tape = 1usize;
        let mut edges = roots;
        let mut operations_count = 0usize;
        let mut maximum_rank = 0usize;
        for operation in operations {
            if !matches!(
                operation.kind,
                WorkspaceOperationKindView::Elementwise("add" | "multiply")
            ) {
                return Err(E::UnknownOperation);
            }
            // Reuse the actual selected worker's complete geometry/dtype
            // validation, including its unknown branches. Nothing is written.
            if self
                .emit(operation, &mut facts::Emitter::count())?
                .is_none()
            {
                return Err(E::UnknownOperation);
            }
            operations_count = operations_count.checked_add(1).ok_or(E::Overflow)?;
            for layout in operation.inputs.iter().chain(operation.outputs.iter()) {
                maximum_rank = maximum_rank.max(layout.shape().len());
            }
            // ops.cpp Add/Multiply: up to two casts, two Broadcast descriptors,
            // and one binary descriptor, with 1+1+1+1+2 ordered input edges.
            // No siblings are produced. Float32 metadata erases F16/BF16, so
            // matching metadata dtypes do not remove either possible cast.
            arrays = arrays.checked_add(5).ok_or(E::Overflow)?;
            tape = tape.checked_add(5).ok_or(E::Overflow)?;
            edges = edges.checked_add(6).ok_or(E::Overflow)?;
        }
        // Broadcast aliases Data. The selected cast/copy and binary workers
        // each publish at most one Data; their ordinary base capture backing is
        // larger and remains part of the actual prepared constructor recipe.
        let base = safemlx::OperationEvent::eval_record_layout(tape, 1, tape)
            .ok_or(E::UnknownQualification)?;
        let traversal =
            safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots,
                arrays,
                tape_entries: tape,
                input_edges: edges,
                output_slots: tape,
                streams: 1,
                captures: base.capture_slots().max(1),
            })
            .ok_or(E::UnknownQualification)?;
        // The native queries execute in this cold producer. Their named
        // transports enter its control requirement, not Graph/Record arena bytes.
        let graph = safemlx::OperationEvent::pointwise_graph_layout(operations_count, maximum_rank)
            .ok_or(E::UnknownQualification)?;
        // Every lowered primitive has <= 2 inputs; the Synchronizer has the
        // actual root count. No selected producer has siblings. Include its
        // edges once in the cumulative vector population, never once per entry.
        // RetainGraph can make an invocation a tracer after initial leaf
        // validation. Price the same finite E input handles conservatively;
        // native creation allocates only when the actual array requests them.
        let gpu_layout =
            safemlx::OperationEvent::gpu_eval_prologue_layout_with_tracing(roots.max(2), 0, true)
                .ok_or(E::UnknownQualification)?;
        let gpu_prologue = gpu_layout.population(tape, edges, 0).ok_or(E::Overflow)?;
        let query_controls = base
            .query_control_bytes()
            .ok_or(E::Overflow)?
            .checked_add(traversal.query_control_bytes().ok_or(E::Overflow)?)
            .and_then(|bytes| bytes.checked_add(graph.control_bytes()?))
            .and_then(|bytes| bytes.checked_add(gpu_layout.control_bytes()?))
            .ok_or(E::Overflow)?;
        Ok((traversal, graph, base, gpu_prologue, query_controls))
    }
}
