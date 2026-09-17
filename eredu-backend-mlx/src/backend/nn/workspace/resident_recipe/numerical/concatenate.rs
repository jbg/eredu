//! The exact shifted captured-prefill join through the shared CPU concat plan.
use super::*;
use std::mem::{size_of, size_of_val};
impl SpeculativeNumericalRecipe {
    pub(crate) fn inspect_cpu_concatenate(
        report: &WorkspaceTraceReport,
        ordinary: MlxMetalWorkspaceMechanisms,
        cpu: MlxCpuWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        use eredu_nn::workspace::WorkspaceOperationKindView as K;
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU capture concatenation source differs from its axis-one two-input equation"
            ))
        };
        let [operation] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let operation = operation.as_view();
        if !matches!(operation.kind, K::Concatenate)
            || operation.inputs.len() != 2
            || operation.outputs.len() != 1
        {
            return Err(invalid());
        }
        let left = operation.inputs.get(0).ok_or_else(invalid)?;
        let right = operation.inputs.get(1).ok_or_else(invalid)?;
        let output = operation.outputs.get(0).ok_or_else(invalid)?;
        let rank = left.shape().len();
        if !(3..=4).contains(&rank)
            || right.shape().len() != rank
            || output.shape().len() != rank
            || left
                .shape()
                .iter()
                .chain(right.shape())
                .chain(output.shape())
                .any(|&n| n <= 0)
            || left.shape()[0] != 1
            || right.shape()[0] != 1
            || output.shape()[0] != 1
            || left.shape()[2..] != right.shape()[2..]
            || left.shape()[2..] != output.shape()[2..]
            || left.shape()[1].checked_add(right.shape()[1]) != Some(output.shape()[1])
        {
            return Err(invalid());
        }
        // Exact dtype/stride restrictions, frontend casts, General copy jobs,
        // output backing and native controls come from the ordinary CPU plan.
        let recorder = ResidentRecipeRecorder::with_cpu_context(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: output.shape()[1] as u64,
                max_output_tokens: 0,
                prefill_chunk_positions: output.shape()[1] as u64,
                output: eredu_core::OutputDemand::Sequence,
            },
            ordinary,
            cpu,
            context,
        )?;
        let reduced = recorder.reduce_trace(report, None, 0, 1)?;
        if reduced.first_missing_operation.is_some()
            || reduced.unqualified_kernel_owner.is_some()
            || reduced.validation_roots != 0
            || reduced.nested_completions != 0
        {
            return Err(invalid());
        }
        let dispatch = reduced.dispatch.ok_or_else(invalid)?;
        if dispatch.cpu_model.is_none()
            || dispatch.gpu_entries != 0
            || dispatch.parallel_entries != 0
            || dispatch.kernel_attempts != 0
        {
            return Err(invalid());
        }
        let completion = ResidentCompletionRecipe {
            validation_roots: 0,
            grouped_outputs: reduced.grouped_outputs,
            traversal: reduced.traversal.ok_or_else(invalid)?,
            graph: reduced.graph.ok_or_else(invalid)?,
            dispatch: Some(dispatch),
            nested_completions: 0,
            nested_root_capacity: 0,
        };
        let graph =
            graph_capacity::ResidentGraphStorage::for_completion(completion).ok_or_else(invalid)?;
        let record = record_capacity::ResidentRecordStorage::for_completion(completion)
            .ok_or_else(invalid)?;
        let frames = [
            size_of::<(
                &WorkspaceTraceReport,
                MlxMetalWorkspaceMechanisms,
                MlxCpuWorkspaceMechanisms,
                &WorkspaceContext,
            )>(),
            size_of::<WorkspaceOperationView<'_>>(),
            size_of::<WorkspaceLayoutView<'_>>() * 3,
            size_of::<Option<WorkspaceLayoutView<'_>>>(),
            size_of::<&WorkspaceContext>(),
            size_of::<
                std::iter::Chain<
                    std::iter::Chain<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>,
                    std::slice::Iter<'_, i32>,
                >,
            >(),
            size_of::<Option<i32>>(),
            size_of::<InferenceGeometry>(),
            size_of::<usize>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<ReducedTrace>(),
            size_of::<ResidentDispatchPopulation>(),
            size_of::<ResidentCompletionRecipe>(),
            size_of::<graph_capacity::ResidentGraphStorage>(),
            size_of::<record_capacity::ResidentRecordStorage>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Self>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(reduced.query_controls?))
            .ok_or_else(invalid)?;
        let controls = u64::try_from(controls)
            .map_err(|_| invalid())?
            .checked_add(graph.control_bytes().ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        Ok(Self {
            completion,
            storage: reduced.mutable_storage.ok_or_else(invalid)?,
            graph_capacity: usize::try_from(graph.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?,
            record_capacity: usize::try_from(record.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?,
            kernels: 0,
            controls,
        })
    }
}
