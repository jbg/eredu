//! Exact source companion for the existing resident equation recorder.
use super::*;
pub(super) use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation;
pub(super) use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource;

pub(crate) struct ParallelRecipeRecorder {
    recorder: ResidentRecipeRecorder,
    source: OriginalParallelSource,
    mechanism:ResidentExecutionMechanisms,
}
impl ParallelRecipeRecorder {
    pub(in crate::backend::nn::workspace) fn new(
        geometry: eredu_core::InferenceGeometry,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
        source: &OriginalParallelSource,
    ) -> Result<Self, Error> {
        context.charge_metadata(
            std::mem::size_of::<Self>() + std::mem::size_of::<Result<Self, Error>>(),
        )?;
        Ok(Self {
            recorder: mechanism.recorder(geometry, context)?,
            source: source.clone(),
            mechanism,
        })
    }
    pub(crate) fn bind_addressable_sources(&mut self,source:AddressableSources)->Result<(),Error>{self.recorder.bind_addressable_sources(source)}
    pub(crate) fn bind_layerwise_constructor_source(&mut self,
        source: &crate::backend::runtime::execution::generic::LayerwiseWorkspace,
    ) -> Result<(), Error> {
        self.recorder.bind_layerwise_constructor_source(source)
    }
    /// Actual local capture callbacks extend the same recorded model span.
    /// Receiver receipt bookkeeping adds no fictitious native completion.
    pub(crate) fn record_capture_population(&mut self,
        population:crate::backend::array_copy::CaptureNativePopulation)->Result<(),Error> {
        self.recorder.record_capture_population(population)
    }
    pub(crate) fn record_capture_scalars(&mut self,
        scalars:&[std::cell::Cell<Option<eredu_nn::workspace::WorkspaceFloatingType>>])->Result<(),Error> {
        self.recorder.record_capture_scalars(scalars)
    }
    pub(crate) fn finish(
        self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<ResidentNativeRecipe, Error> {
        let mut recipe = self.recorder.finish(plan)?;
        // Control placement is still required when the genuine equation range
        // is empty, for example restoring an already terminal parallel branch.
        recipe.parallel_source = Some(self.source);
        Ok(recipe)
    }
    fn record(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained: usize,
        outputs: usize,
        input: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
        prepared: bool,
    ) -> Result<(), Error> {
        let invocation = self
            .source
            .prepare_invocation_with_boundary(&report.operations,Some(self.mechanism),self.recorder.addressable_sources.as_ref())
            .map_err(|cause| self.source.neural_error(cause))?;
        let population=invocation.expert_capture_population().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        self.recorder.record_equation_with_parallel(
            span,report,retained,outputs,input,output,prepared,Some(invocation),
        )?;
        if population.publications!=0||population.completions!=0||population.controls!=0 {
            self.recorder.record_capture_population(population)?;
        }
        Ok(())
    }
}
impl InferenceEquationTraceObserver for ParallelRecipeRecorder {
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
    ) -> Result<(), Error> {
        self.record(
            span,
            report,
            retained_roots,
            output_roots,
            Some(input_operation),
            None,
            false,
        )
    }
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        self.record(
            span,
            report,
            retained_roots,
            output_roots,
            Some(input_operation),
            output,
            false,
        )
    }
    fn observe_prepared_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        self.record(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
            true,
        )
    }
    fn observe_sampling_input(&mut self, input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan) -> Result<(), Error> {
        self.recorder.record_sampling_input(input)
    }
    fn observe_sampling(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.recorder.record_sampling(phase, report, None)
    }
    fn observe_sampling_with_storage(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing: WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.recorder.record_sampling(phase, report, Some(closing))
    }
}

impl eredu_runtime::working_memory::SamplingWorkspaceObserver for ParallelRecipeRecorder {
    fn observe_input(&mut self, input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan) -> Result<(), Error> {
        self.recorder.record_sampling_input(input)
    }
    fn observe(&mut self, phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport) -> Result<(), Error> {
        self.recorder.record_sampling(phase, report, None)
    }
    fn observe_with_storage(&mut self, phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport, closing: WorkspaceStoragePopulation) -> Result<(), Error> {
        self.recorder.record_sampling(phase, report, Some(closing))
    }
}
