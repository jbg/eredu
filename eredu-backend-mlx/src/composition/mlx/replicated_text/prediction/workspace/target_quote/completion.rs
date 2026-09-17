//! Exact state/score census at the existing target equation trace boundary.
use super::*;
use eredu_architectures::prepared_execution::InferenceEquationTraceObserver;
use eredu_nn::workspace::{WorkspaceStoragePopulation, WorkspaceTraceReport};
use eredu_runtime::working_memory::InferenceWorkspaceSpan;
pub(in crate::composition::mlx::replicated_text::prediction::workspace) struct CompletionTrace<'a> {
    recorder: &'a mut ResidentRecipeRecorder,
    context: &'a WorkspaceContext,
    roots: Option<usize>,
}
impl<'a> CompletionTrace<'a> {
    pub(in crate::composition::mlx::replicated_text::prediction::workspace) fn new(
        recorder: &'a mut ResidentRecipeRecorder,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        context
            .charge_metadata(
                size_of::<Self>() + size_of::<Option<usize>>() + size_of::<(usize, usize)>(),
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        Ok(Self {
            recorder,
            context,
            roots: None,
        })
    }
    fn record(&mut self, state: usize, output: usize) -> Result<(), eredu_nn::Error> {
        if self.roots.is_some() {
            return Err(self
                .context
                .metadata_source(WorkingMemoryError::IdentityMismatch));
        }
        self.roots = Some(
            state
                .checked_add(output)
                .ok_or_else(|| self.context.metadata_source(WorkingMemoryError::Overflow))?,
        );
        Ok(())
    }
    pub(in crate::composition::mlx::replicated_text::prediction::workspace) fn finish(self) -> Result<usize, Error> {
        self.roots.ok_or_else(|| {
            Error::from(
                self.context
                    .metadata_source(WorkingMemoryError::UnknownBound),
            )
        })
    }
}
impl InferenceEquationTraceObserver for CompletionTrace<'_> {
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
    ) -> Result<(), eredu_nn::Error> {
        self.record(retained_roots, output_roots)?;
        self.recorder
            .observe(span, report, retained_roots, output_roots, input_operation)
    }
    fn observe_prepared_with_storage(&mut self,span:&InferenceWorkspaceSpan,
        report:&WorkspaceTraceReport,retained_roots:usize,output_roots:usize,
        input_operation:Option<usize>,output:Option<WorkspaceStoragePopulation>)->Result<(),eredu_nn::Error>{
        self.record(retained_roots,output_roots)?;
        self.recorder.observe_prepared_with_storage(span,report,retained_roots,output_roots,input_operation,output)
    }
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), eredu_nn::Error> {
        self.record(retained_roots, output_roots)?;
        self.recorder.observe_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
        )
    }
    fn observe_target_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<WorkspaceStoragePopulation>,
        capture: &WorkspaceTensor,
        scores: Option<&WorkspaceTensor>,
    ) -> Result<(), eredu_nn::Error> {
        // The shared session completes state plus vocabulary output here.
        // Hidden capture has its own owner and joins the outer phase completion.
        self.record(retained_roots, usize::from(scores.is_some()))?;
        self.recorder.observe_target_with_storage(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
            capture,
            scores,
        )
    }
    fn observe_external_target_with_storage(
        &mut self, span: &InferenceWorkspaceSpan, report: &WorkspaceTraceReport,
        retained_roots: usize, output_roots: usize, input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
        capture: &eredu_architectures::composite_execution::ExternalPredictionTargetCapture<WorkspaceTensor>,
        scores: Option<&WorkspaceTensor>,
    ) -> Result<(), eredu_nn::Error> {
        // As in the ordinary external target worker, the inner session settles
        // state/scores; the complete semantic capture remains in the outer role.
        self.record(retained_roots, usize::from(scores.is_some()))?;
        self.recorder.observe_external_target_with_storage(
            span, report, retained_roots, output_roots, input_operation, output, capture, scores,
        )
    }

}
