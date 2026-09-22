//! A lexical pair of actual completed runs; it grants no model or native work.
use super::*;
use eredu_core::HostMetadataFunding;

#[derive(Debug)]
struct Guard {
    request: InferenceRequest,
    context: TextStepContext,
    issue: u64,
}
impl Guard {
    fn prepare(
        request: &InferenceRequest,
        context: &TextStepContext,
    ) -> Result<Self, WorkingMemoryError> {
        let issue = control::validate(request, context, |run| {
            if run.control_pending.is_some() {
                return Err(WorkingMemoryError::PreparationNotReady);
            }
            let issue = run
                .control_issue
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?;
            run.control_issue = issue;
            run.control_pending = Some(issue);
            Ok(issue)
        })?;
        Ok(Self {
            request: request.clone(),
            context: context.clone(),
            issue,
        })
    }
    fn validate(&self) -> Result<(), WorkingMemoryError> {
        control::validate(&self.request, &self.context, |run| {
            if run.control_pending == Some(self.issue) {
                Ok(())
            } else {
                Err(WorkingMemoryError::IdentityMismatch)
            }
        })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(authority) = self.request.preparation.as_ref() {
            if let Ok(mut state) = authority.state.lock() {
                if let Some(run) = state.run.as_mut() {
                    if run.control_pending == Some(self.issue) {
                        run.control_pending = None;
                    }
                }
            }
        }
    }
}

/// Holds the two actual completed-run gates throughout an atomic native exchange.
/// No attempt, sampler epoch, observation/copy allowance or admission is reset.
/// The typed state worker must separately validate both actual placement sources.
#[derive(Debug)]
pub struct PendingTextBranchExchange {
    installed: Guard,
    incoming: Guard,
    // Requests and fixed controls retire before their independently paid account.
    _funding: HostMetadataFunding,
}
impl InferenceRequest {
    /// Validates both exact completed contexts without allocating or claiming.
    pub fn validate_branch_exchange(
        &self,
        context: &TextStepContext,
        incoming: &Self,
        incoming_context: &TextStepContext,
    ) -> Result<(), WorkingMemoryError> {
        if self.validate_same_request(incoming).is_ok() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for (request, context) in [(self, context), (incoming, incoming_context)] {
            control::validate(request, context, |run| {
                if run.control_pending.is_some() {
                    Err(WorkingMemoryError::PreparationNotReady)
                } else {
                    Ok(())
                }
            })?;
        }
        Ok(())
    }
    /// Pays fixed control storage, then gates both actual runs before planning.
    /// Failure abandons any first gate but never refunds its monotone issue.
    pub fn begin_branch_exchange(
        &self,
        context: &TextStepContext,
        incoming: &Self,
        incoming_context: &TextStepContext,
        funding: &HostMetadataFunding,
    ) -> Result<PendingTextBranchExchange, WorkingMemoryError> {
        self.validate_branch_exchange(context, incoming, incoming_context)?;
        funding
            .reserve_metadata(std::mem::size_of::<(
                PendingTextBranchExchange,
                Guard,
                Result<PendingTextBranchExchange, WorkingMemoryError>,
                Result<Guard, WorkingMemoryError>,
            )>())
            .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        let installed = Guard::prepare(self, context)?;
        let incoming = Guard::prepare(incoming, incoming_context)?;
        Ok(PendingTextBranchExchange {
            installed,
            incoming,
            _funding: funding.clone(),
        })
    }
}
impl PendingTextBranchExchange {
    /// Revalidates the exact gates immediately before the typed state vote.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.installed.validate()?;
        self.incoming.validate()
    }
    pub(crate) fn source_request(&self, incoming: bool) -> (&InferenceRequest, &TextStepContext) {
        let guard = if incoming {
            &self.incoming
        } else {
            &self.installed
        };
        (&guard.request, &guard.context)
    }
}
