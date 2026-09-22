//! A child that owns both the ordinary equation and its validation collector.
//! Ordinary numerical inspection keeps its existing no-validation predicate.
use super::*;
use std::mem::{size_of, size_of_val};
impl SpeculativeNumericalRecipe {
    /// Output roots are completed by the caller's existing root collector;
    /// only completions inside the selected equation consume nested slots.
    pub(crate) fn inspect_completed_outputs(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::inspect_owned_child_sources(
            report,
            outputs,
            mechanism,
            context,
            crate::backend::array_copy::CaptureNativePopulation::default(),
            None,
            None,
            false,
        )
    }
    pub(crate) fn inspect_owned_child(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::inspect_owned_child_with_capture(
            report,
            outputs,
            mechanism,
            context,
            crate::backend::array_copy::CaptureNativePopulation::default(),
        )
    }
    pub(crate) fn inspect_owned_child_with_capture(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
        capture: crate::backend::array_copy::CaptureNativePopulation,
    ) -> Result<Self, Error> {
        Self::inspect_owned_child_with_sources(report, outputs, mechanism, context, capture, None)
    }
    pub(crate) fn inspect_owned_child_with_sources(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
        capture: crate::backend::array_copy::CaptureNativePopulation,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ) -> Result<Self, Error> {
        Self::inspect_owned_child_sources(
            report, outputs, mechanism, context, capture, layerwise, None, true,
        )
    }
    /// The supplied occurrence source was prepared from this exact trace and
    /// selected group. Local constructor sources remain independently bound.
    pub(crate) fn inspect_owned_parallel_child_with_sources(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
        capture: crate::backend::array_copy::CaptureNativePopulation,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        parallel:&crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation,
    ) -> Result<Self, Error> {
        Self::inspect_owned_child_sources(
            report,
            outputs,
            mechanism,
            context,
            capture,
            layerwise,
            Some(parallel),
            true,
        )
    }
    fn inspect_owned_child_sources(
        report: &WorkspaceTraceReport,
        outputs: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
        capture: crate::backend::array_copy::CaptureNativePopulation,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        parallel:Option<&crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation>,
        complete_child: bool,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "retained numerical child has an incomplete native producer"
            ))
        };
        if outputs == 0 || report.operations.is_empty() {
            return Err(invalid());
        }
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 1,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: eredu_core::OutputDemand::Sequence,
        };
        let mut recorder = match mechanism {
            ResidentExecutionMechanisms::Metal(ordinary) => {
                ResidentRecipeRecorder::with_context(geometry, ordinary, context)?
            }
            ResidentExecutionMechanisms::Cpu { ordinary, cpu } => {
                ResidentRecipeRecorder::with_cpu_context(geometry, ordinary, cpu, context)?
            }
        };
        if let Some(source) = layerwise {
            recorder.bind_layerwise_constructor_source(source)?;
        }
        // This child owns a complete model equation, including construction of
        // its unloaded slots. Sampling spans on the enclosing recorder do not;
        // keep this comparison at equation entries, before shared reduction.
        recorder.validate_layerwise_constructor_trace(report)?;
        let reduced = recorder.reduce_trace_with_parallel(report, None, 0, outputs, parallel)?;
        if reduced.first_missing_operation.is_some() || reduced.unqualified_kernel_owner.is_some() {
            return Err(context.metadata_error(format_args!(
                "retained numerical child native producer missing at operation {:?}; kernel source {:?}; source {:?}",
                reduced.first_missing_operation,reduced.unqualified_kernel_owner,reduced.missing_operation_detail)));
        }
        let dispatch = reduced.dispatch.ok_or_else(|| {
            context.metadata_error(format_args!(
                "retained numerical child has no dispatch population"
            ))
        })?;
        let roots = outputs
            .checked_add(reduced.validation_roots)
            .and_then(|n| n.checked_add(capture.retained_roots))
            .ok_or_else(invalid)?;
        let nested = reduced
            .nested_completions
            .checked_add(usize::from(complete_child))
            .and_then(|n| n.checked_add(capture.completions))
            .ok_or_else(invalid)?;
        let graph = reduced.graph.ok_or_else(|| {
            context.metadata_error(format_args!(
                "retained numerical child has no graph population"
            ))
        })?;
        let traversal = reduced.traversal.ok_or_else(|| {
            context.metadata_error(format_args!(
                "retained numerical child has no traversal population"
            ))
        })?;
        let (traversal, dispatch, capture_controls) = super::super::capture::extend_frontier(
            traversal,
            graph,
            dispatch,
            capture.retained_roots,
        )
        .ok_or_else(invalid)?;
        let completion = ResidentCompletionRecipe {
            validation_roots: reduced.validation_roots,
            grouped_outputs: reduced.grouped_outputs,
            traversal,
            graph,
            dispatch: Some(dispatch),
            nested_completions: nested,
            nested_root_capacity: roots,
        };
        let graph =
            graph_capacity::ResidentGraphStorage::for_completion(completion).ok_or_else(invalid)?;
        let records = record_capacity::ResidentRecordStorage::for_completion(completion)
            .ok_or_else(invalid)?;
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<ReducedTrace>(),
            size_of_val(&layerwise),
            size_of_val(&parallel),
            size_of_val(&complete_child),
            size_of::<InferenceGeometry>(),
            size_of::<ResidentCompletionRecipe>(),
            size_of::<ResidentDispatchPopulation>(),
            size_of::<graph_capacity::ResidentGraphStorage>(),
            size_of::<record_capacity::ResidentRecordStorage>(),
            size_of::<[usize; 3]>(),
            size_of::<(
                &WorkspaceTraceReport,
                usize,
                ResidentExecutionMechanisms,
                &WorkspaceContext,
                crate::backend::array_copy::CaptureNativePopulation,
            )>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(reduced.query_controls?))
            .and_then(|n| n.checked_add(capture.controls))
            .and_then(|n| n.checked_add(capture_controls))
            .ok_or_else(invalid)?;
        let controls = u64::try_from(controls)
            .map_err(|_| invalid())?
            .checked_add(graph.control_bytes().ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        Self {
            completion,
            storage: reduced.mutable_storage.ok_or_else(|| {
                context.metadata_error(format_args!(
                    "retained numerical child has no mutable storage"
                ))
            })?,
            graph_capacity: usize::try_from(graph.full_capacity.ok_or_else(|| {
                context.metadata_error(format_args!(
                    "retained numerical child has no graph capacity"
                ))
            })?)
            .map_err(|_| invalid())?,
            record_capacity: usize::try_from(records.full_capacity.ok_or_else(|| {
                context.metadata_error(format_args!(
                    "retained numerical child has no record capacity"
                ))
            })?)
            .map_err(|_| invalid())?,
            kernels: dispatch.kernel_attempts,
            controls,
            ordinary_calls: None,
        }
        .with_ordinary_calls(report, mechanism, context)
    }
}
