//! One retained sampling trace program for initial and revised native policies.
use super::*;
use eredu_runtime::working_memory::{
    SamplingWorkspaceObserver, SamplingWorkspacePhase, SamplingWorkspacePlanCollector,
    SamplingWorkspaceReport,
};
use std::mem::{size_of, size_of_val};

/// Immutable source rows in actual preparation/step order. Numerical state and
/// request/attempt permission belong to the consuming admission, never this plan.
#[derive(Debug)]
pub(crate) struct ResidentSamplingProgram {
    pub(super) rows: Vec<ResidentSamplingRecipe>,
    input: Option<eredu_runtime::working_memory::SamplingWorkspaceInputPlan>,
    // The original recorder paid every row before growth. Keep that account
    // after the rows when this program leaves the enclosing model quote.
    _planning: Option<eredu_core::HostMetadataFunding>,
}
impl ResidentSamplingProgram {
    pub(crate) fn planning_metadata(&self) -> Option<&eredu_core::HostMetadataFunding> {
        self._planning.as_ref()
    }
    pub(crate) fn input(&self) -> Option<eredu_runtime::working_memory::SamplingWorkspaceInputPlan> {
        self.input
    }
    pub(crate) fn kernel_attempts(&self) -> Option<usize> {
        self.rows.iter().try_fold(0usize, |n, row| {
            if row.phase == eredu_runtime::working_memory::SamplingWorkspacePhase::Preparation
                && row.preparation.is_some()
                && row.completion.is_none()
            {
                Some(n)
            } else {
                let completion = row.completion?;
                let dispatch = completion.dispatch?;
                let frontiers = if dispatch.cpu_entries == 0 {
                    1
                } else {
                    completion.nested_completions.checked_add(1)?
                };
                n.checked_add(dispatch.kernel_attempts.checked_mul(frontiers)?)
            }
        })
    }
    pub(crate) fn steps(&self) -> Option<usize> {
        self.rows.len().checked_sub(1)
    }
    pub(crate) fn rows(&self) -> &[ResidentSamplingRecipe] {
        &self.rows
    }
    pub(crate) fn preparation_graph(&self) -> Option<Option<safemlx::ResidentGraphLayout>> {
        match self.rows.first()?.preparation? {
            ResidentSamplingPreparation::Empty => Some(None),
            ResidentSamplingPreparation::EagerKey(graph) => Some(Some(graph)),
        }
    }
    pub(crate) fn completion(&self, index: usize) -> Option<ResidentCompletionRecipe> {
        self.rows.get(index.checked_add(1)?)?.completion
    }
}
impl ResidentRecipeRecorder {
    pub(super) fn record_sampling_input(&mut self, input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan) -> Result<(), Error> {
        if self.sampling_input.is_some() || !self.sampling.is_empty() {
            return Err(self.metadata_error("sampling source was already recorded"));
        }
        self.sampling_input = Some(input);
        Ok(())
    }
    /// Both consumers see the original report before it retires: the neutral
    /// collector keeps phase storage facts, and native lowering keeps the
    /// selected graph, completion and allocation populations.
    pub(crate) fn sampling_collector(
        self,
        steps: u64,
        context: &WorkspaceContext,
    ) -> Result<SamplingRecipeCollector<'_>, Error> {
        context.charge_metadata(size_of::<SamplingRecipeCollector<'_>>()
            .checked_add(size_of::<Result<SamplingRecipeCollector<'_>, Error>>())
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        let portable = SamplingWorkspacePlanCollector::new(self.geometry, steps, context)?;
        Ok(SamplingRecipeCollector {
            native: self,
            portable,
            steps,
        })
    }
    pub(super) fn take_sampling_program(
        &mut self,
        expected: Option<usize>,
    ) -> Result<ResidentSamplingProgram, Error> {
        if expected != Some(self.sampling.len()) {
            return Err(self.metadata_error("resident recipe is missing actual sampling phases"));
        }
        if let Some(context) = &self.context {
            let controls = [
                size_of::<ResidentSamplingProgram>(),
                size_of::<Result<ResidentSamplingProgram, Error>>(),
                size_of::<Option<eredu_core::HostMetadataFunding>>(),
            ];
            context.charge_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?;
        }
        Ok(ResidentSamplingProgram {
            rows: std::mem::take(&mut self.sampling),
            input: self.sampling_input,
            _planning: self
                .context
                .as_ref()
                .and_then(WorkspaceContext::metadata_funding),
        })
    }
    /// Completes the same sampler trace without constructing unrelated model
    /// equations. The actual borrowed sampler's caller supplies the finite count
    /// already used by quote_sampling_workspace_with_observer.
    pub(crate) fn finish_sampling(mut self, steps: u64) -> Result<ResidentSamplingProgram, Error> {
        if !self.records.is_empty() || steps != self.geometry.max_output_tokens {
            return Err(self.metadata_error("sampling program differs from its traced request"));
        }
        self.take_sampling_program(usize::try_from(steps).ok().and_then(|n| n.checked_add(1)))
    }
}

pub(crate) struct SamplingRecipeCollector<'a> {
    native: ResidentRecipeRecorder,
    portable: SamplingWorkspacePlanCollector<'a>,
    steps: u64,
}
impl SamplingRecipeCollector<'_> {
    pub(crate) fn finish(
        mut self,
        report: &SamplingWorkspaceReport,
    ) -> Result<(ResidentSamplingProgram, InferenceSpanWorkspacePlan), Error> {
        if !self.native.records.is_empty() || report.steps != self.steps {
            return Err(self
                .native
                .metadata_error("sampling program differs from its traced request"));
        }
        let plan = self.portable.finish(report)?;
        let program = self.native.take_sampling_program(
            usize::try_from(self.steps)
                .ok()
                .and_then(|n| n.checked_add(1)),
        )?;
        Ok((program, plan))
    }
}
impl SamplingWorkspaceObserver for SamplingRecipeCollector<'_> {
    fn observe_input(&mut self, input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan) -> Result<(), Error> {
        self.portable.observe_input(input)?;
        self.native.record_sampling_input(input)
    }
    fn observe(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.portable.observe(phase, report)?;
        self.native.record_sampling(phase, report, None)
    }
    fn observe_with_storage(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing: WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.portable.observe_with_storage(phase, report, closing)?;
        self.native.record_sampling(phase, report, Some(closing))
    }
}
impl SamplingWorkspaceObserver for ResidentRecipeRecorder {
    fn observe_input(&mut self, input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan) -> Result<(), Error> {
        self.record_sampling_input(input)
    }
    fn observe(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.record_sampling(phase, report, None)
    }
    fn observe_with_storage(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing: WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.record_sampling(phase, report, Some(closing))
    }
}
