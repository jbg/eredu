//! Closed original whole-request compiler. Native completeness remains rejected;
//! complete neutral producers exercise this same comparison/construction seam.
use super::*;
use eredu_core::{
    AdmissionObservation, AdmissionPolicyError, AdmissionRequest, AdmissionRequirements,
    BorrowedAdmissionRejection, BorrowedAdmissionResult,
};
mod diagnostics;
mod report_workspace;
use diagnostics::{ConstructionFailure, StateReport};

// These concrete existing source aliases confer no request grant. Their own
// original source accounts remain live without registration or byte credit.
#[derive(Clone, Debug)]
struct Sources {
    selected_model: OriginalPreparedHostInput,
    input: OriginalPreparedHostInput,
    // Last so these exact source wrappers retire before the report-held Q.
    report: Option<report_workspace::ReportOwner>,
}

// Only a complete mechanism producer in this module can construct this recipe.
// Native A/B/B3/C have no such producer yet; they retain their fixed early gap.
// No public bytes/flags/closure constructor or ordinary-report adoption exists.
struct Recipe<'a> {
    pool: &'a MemoryLedger,
    execution: &'a InferenceExecutionIdentity,
    selected_model: &'a OriginalPreparedHostInput,
    input: &'a OriginalPreparedHostInput,
    initialized_revision: u64,
    current_revision: u64,
    current_frontier: u64,
    maximum_context: u64,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: Option<eredu_core::MemoryLimits>,
}
impl Recipe<'_> {
    fn validate(&self, sources: &Sources) -> Result<(), WorkingMemoryError> {
        sources.selected_model.validate_pool(self.pool)?;
        sources.input.validate_pool(self.pool)?;
        if !sources.selected_model.same_source(self.selected_model)
            || !sources.input.same_source(self.input)
            || self.initialized_revision != self.current_revision
            || self.current_frontier != self.geometry.cached_positions
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.geometry
            .validate_fixed()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        if self.geometry.batch_size != self.request.batch_size
            || self
                .geometry
                .cached_positions
                .checked_add(self.geometry.input_positions)
                != Some(self.request.input.model_positions)
            || self.geometry.max_output_tokens != self.request.max_output_tokens
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Failure {
    Source(WorkingMemoryError),
    Policy(AdmissionPolicyError),
    Rejected(BorrowedAdmissionRejection<'static>),
    Construction(ConstructionFailure<Sources>),
}
impl From<WorkingMemoryError> for Failure {
    fn from(value: WorkingMemoryError) -> Self {
        Self::Source(value)
    }
}

// Report production is a closed pre-admission operation. The actual selected
// neutral consumer calls it for each candidate; no callback runs after Q. This
// is private so a downstream caller cannot assert completeness with raw bytes.
fn admit<'source>(
    recipe: Recipe<'_>,
    sources: Sources,
    mut report: impl FnMut(InferenceGeometry) -> Result<StateReport<'source>, WorkingMemoryError>,
) -> Result<(Sources, WorkingMemoryReservation), Failure> {
    recipe.validate(&sources)?;
    // Same cheap context check before candidate construction as the legacy route.
    // A complete source has a concrete maximum, so no allocated missing reason.
    let request = recipe.request.clone();
    if let Some(rejection) = eredu_core::check_admission_context_borrowed(
        AdmissionObservation::Available(recipe.maximum_context),
        request.clone(),
    )
    .map_err(Failure::Policy)?
    {
        return Err(Failure::Rejected(rejection));
    }
    let (report, requirements, decision, ticket) =
        select_prefill_candidate(recipe.geometry, |g| {
            let report = report(g).map_err(|e| CandidateFailure::Terminal(Failure::Source(e)))?;
            if report.geometry != g {
                return Err(CandidateFailure::Terminal(Failure::Source(
                    WorkingMemoryError::IdentityMismatch,
                )));
            }
            let floor = report
                .control_bytes::<Sources>()
                .map_err(|e| CandidateFailure::Terminal(Failure::Source(e)))?;
            if report.components[5] < floor {
                return Err(CandidateFailure::Terminal(Failure::Source(
                    WorkingMemoryError::UnknownBound,
                )));
            }
            let requirements = report
                .requirements()
                .map_err(|e| CandidateFailure::Terminal(Failure::Source(e)))?;
            let decision = match eredu_core::apply_admission_requirements(
                request.clone(),
                AdmissionRequirements {
                    maximum_context: AdmissionObservation::Available(recipe.maximum_context),
                    state: requirements,
                    incremental: None,
                    incremental_per_domain: false,
                },
            )
            .map_err(|e| CandidateFailure::Terminal(Failure::Policy(e)))?
            {
                BorrowedAdmissionResult::Admitted(decision) => decision,
                BorrowedAdmissionResult::Rejected(rejection) => {
                    return Err(CandidateFailure::Terminal(Failure::Rejected(rejection)));
                }
            };
            let pending = {
                let mut usage = recipe.pool.0.usage.lock().map_err(|_| {
                    CandidateFailure::Terminal(Failure::Source(WorkingMemoryError::Poisoned))
                })?;
                let commit = PreparedAccountCommit::prepare(
                    recipe.pool,
                    recipe.execution,
                    &usage,
                    decision
                        .incremental_required_bytes
                        .expect("host fixture requirement"),
                    recipe.capacity.as_ref(),
                    &[],
                )
                .map_err(|e| match e {
                    e @ (WorkingMemoryError::Domain(MemoryDomainError::BudgetExceeded {
                        ..
                    })) => CandidateFailure::SmallerChunk(Failure::Source(e)),
                    e => CandidateFailure::Terminal(Failure::Source(e)),
                })?;
                funding::PendingAccount::accept(
                    recipe.pool,
                    recipe.execution,
                    &mut usage,
                    commit,
                    decision
                        .incremental_required_bytes
                        .expect("host fixture requirement"),
                    recipe.capacity.clone(),
                    floor,
                )
                .map_err(|e| CandidateFailure::Terminal(Failure::Source(e)))?
            };
            // Publish the fixed node before any recoverable diagnostic constructor.
            Ok((report, requirements, decision, pending.publish()))
        })?;
    let (mut sources, reservation, report_storage) = diagnostics::construct(
        ticket,
        sources,
        recipe.execution,
        report,
        requirements,
        decision,
        recipe.capacity,
    )
    .map_err(Failure::Construction)?;
    if let Some(storage) = report_storage {
        sources.report = Some(report_workspace::ReportOwner::new(
            storage,
            &sources,
            reservation.clone(),
        ));
    }
    Ok((sources, reservation))
}

#[cfg(test)]
pub(super) mod tests;
