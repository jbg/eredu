//! The actual sampling observer produces the same immutable span record owner.
use super::*;
use crate::working_memory::{
    SamplingWorkspaceObserver, SamplingWorkspacePhase, SamplingWorkspaceReport,
};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

/// Prospective collector for the actual preparation and finite sampling phases.
/// It records the shared sampling worker's reports before they retire; callers
/// cannot fill records from detached byte estimates. This is descriptive plan
/// construction, not extension admission or permission to submit native work.
pub struct SamplingWorkspacePlanCollector<'a> {
    geometry: InferenceGeometry,
    steps: u64,
    records: Vec<InferenceSpanWorkspaceRecord>,
    context: &'a WorkspaceContext,
}
#[derive(Debug, thiserror::Error)]
#[error("sampling workspace phases do not match the declared finite program")]
struct PhaseMismatch;
impl<'a> SamplingWorkspacePlanCollector<'a> {
    /// Prepare the exact row destination under the actual planning context.
    /// Geometry describes the retained original run; `steps` is the number of
    /// remaining sampler calls, independently rechecked at extension admission.
    pub fn new(
        geometry: InferenceGeometry,
        steps: u64,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(
            [
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<(InferenceGeometry, u64, &'a WorkspaceContext)>(),
                size_of::<usize>(),
                size_of::<Vec<InferenceSpanWorkspaceRecord>>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        geometry
            .validate_fixed()
            .map_err(|error| context.metadata_source(error))?;
        if steps > geometry.max_output_tokens {
            return Err(context.metadata_source(PhaseMismatch));
        }
        let capacity = usize::try_from(steps)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let records = context.metadata_vec(capacity)?;
        Ok(Self {
            geometry,
            steps,
            records,
            context,
        })
    }

    /// Close the same actual phase traversal. Unknown row bounds remain unknown;
    /// this method does not promote them to complete admission facts.
    pub fn finish(
        self,
        report: &SamplingWorkspaceReport,
    ) -> Result<InferenceSpanWorkspacePlan, Error> {
        self.context.charge_metadata(
            size_of::<Self>()
                + size_of::<Result<InferenceSpanWorkspacePlan, Error>>()
                + size_of::<(&SamplingWorkspaceReport, usize)>(),
        )?;
        let expected = usize::try_from(self.steps)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if report.steps != self.steps || self.records.len() != expected {
            return Err(self.context.metadata_source(PhaseMismatch));
        }
        InferenceSpanWorkspacePlan::new_metadata(self.geometry, self.records, Some(self.context))
    }
}
impl SamplingWorkspaceObserver for SamplingWorkspacePlanCollector<'_> {
    fn observe(
        &mut self,
        phase: SamplingWorkspacePhase,
        trace: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.context.charge_metadata(
            size_of::<InferenceSpanWorkspaceRecord>()
                + size_of::<(SamplingWorkspacePhase, usize)>()
                + size_of::<Result<(), Error>>(),
        )?;
        let expected = if self.records.is_empty() {
            SamplingWorkspacePhase::Preparation
        } else {
            SamplingWorkspacePhase::Step {
                index: u64::try_from(self.records.len() - 1)
                    .map_err(|_| WorkspaceMetadataError::Overflow)?,
            }
        };
        if phase != expected
            || matches!(phase, SamplingWorkspacePhase::Step { index } if index >= self.steps)
        {
            return Err(self.context.metadata_source(PhaseMismatch));
        }
        self.records.push(InferenceSpanWorkspaceRecord::new(
            InferenceWorkspaceSpan::Sampling(phase),
            trace,
            crate::working_memory::WorkspaceReportMetadata::new(self.context),
        )?);
        Ok(())
    }
}
