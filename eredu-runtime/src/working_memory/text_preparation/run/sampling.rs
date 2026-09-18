//! A prospective sampler replacement bound to one quiescent original run.
use super::*;
use super::control::validate;
use eredu_core::HostMetadataFunding;
use std::mem::{size_of, size_of_val};

#[derive(Debug, Clone)]
pub(in crate::working_memory) struct SamplingExtensionBinding {
    request: InferenceRequest,
    context: TextStepContext,
    issue: u64,
    first: u64,
    end: u64,
    // All descriptor aliases retire before their original planning payer.
    funding: HostMetadataFunding,
}

/// One non-repeatable sampler replacement attempt at an authenticated boundary.
/// While this owner is pending, the same run cannot issue a prediction. Dropping
/// it abandons this attempt without modifying the installed sampler or refunding
/// constructor spending. It is neither a native scope nor a byte allowance.
#[derive(Debug)]
pub struct PendingSamplingExtension {
    pub(in crate::working_memory) binding: SamplingExtensionBinding,
    committed: bool,
}

impl InferenceRequest {
    /// Validates the actual retained preparation and core boundary before any
    /// planning context is constructed. The result is a count, never authority.
    pub fn sampling_extension_remaining(
        &self,
        context: &TextStepContext,
    ) -> Result<u64, WorkingMemoryError> {
        validate(self, context, |run| {
            if run.control_pending.is_some() {
                return Err(WorkingMemoryError::PreparationNotReady);
            }
            self.geometry()
                .max_output_tokens
                .checked_sub(run.next_attempt)
                .ok_or(WorkingMemoryError::Overflow)
        })
    }

    /// Claims a fresh extension issue after paying its fixed original controls.
    /// The core context must be the actual installed machine's current boundary.
    /// A retained old context, active prediction or unrelated request cannot
    /// create this owner. Native work requires separate complete span admission.
    pub fn begin_sampling_extension(
        &self,
        context: &TextStepContext,
        funding: &HostMetadataFunding,
    ) -> Result<PendingSamplingExtension, WorkingMemoryError> {
        self.sampling_extension_remaining(context)?;
        funding
            .reserve_metadata(control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        let issue = validate(self, context, |run| {
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
        Ok(PendingSamplingExtension {
            binding: SamplingExtensionBinding {
                request: self.clone(),
                context: context.clone(),
                issue,
                first: context.attempt(),
                end: self.geometry().max_output_tokens,
                funding: funding.clone(),
            },
            committed: false,
        })
    }
}

impl PendingSamplingExtension {
    /// Exact original run geometry; no model invocation is added by this plan.
    pub fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.binding.request.geometry()
    }
    /// First original prediction that may consume the newly installed sampler.
    pub fn first_attempt(&self) -> u64 {
        self.binding.first
    }
    /// Exact remaining interval; the original request's output ceiling is fixed.
    pub fn remaining_steps(&self) -> u64 {
        self.binding.end - self.binding.first
    }
    pub(in crate::working_memory) fn commit(
        mut self,
    ) -> Result<SamplingExtensionBinding, WorkingMemoryError> {
        self.binding.validate_pending()?;
        validate(&self.binding.request, &self.binding.context, |run| {
            if run.control_pending != Some(self.binding.issue) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            run.sampling_epoch = self.binding.issue;
            run.control_pending = None;
            Ok(())
        })?;
        self.committed = true;
        Ok(self.binding.clone())
    }
}
impl Drop for PendingSamplingExtension {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(actual) = self.binding.request.preparation.as_ref() {
            if let Ok(mut state) = actual.state.lock() {
                if let Some(run) = &mut state.run {
                    if run.control_pending == Some(self.binding.issue) {
                        run.control_pending = None;
                    }
                }
            }
        }
    }
}
impl SamplingExtensionBinding {
    pub(in crate::working_memory) fn request(&self) -> &InferenceRequest {
        &self.request
    }
    pub(in crate::working_memory) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(in crate::working_memory) fn first(&self) -> u64 {
        self.first
    }
    pub(in crate::working_memory) fn end(&self) -> u64 {
        self.end
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        self.issue == other.issue
            && self.context == other.context
            && self.request.validate_same_request(&other.request).is_ok()
    }
    pub(in crate::working_memory) fn validate_pending(&self) -> Result<(), WorkingMemoryError> {
        validate(&self.request, &self.context, |run| {
            if run.control_pending == Some(self.issue) {
                Ok(())
            } else {
                Err(WorkingMemoryError::IdentityMismatch)
            }
        })
    }
    pub(in crate::working_memory) fn validate_step(
        &self,
        step: &InferenceTextStep,
    ) -> Result<u64, WorkingMemoryError> {
        self.request.validate_same_request(step.request())?;
        let reservation = self
            .request
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let (actual, attempt) = step.original_scope_identity(reservation)?;
        if !Arc::ptr_eq(
            &actual,
            self.request
                .preparation
                .as_ref()
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        ) || attempt < self.first
            || attempt >= self.end
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if run.sampling_epoch != self.issue || run.control_pending.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(attempt)
    }
    pub(in crate::working_memory) fn validate_policy(
        &self,
        graph: Option<std::num::NonZeroU64>,
        tracking: Option<std::num::NonZeroU64>,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pending()?;
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let policy = actual.config.inference_policy();
        if graph != policy.graph_metadata_capacity_bytes
            || tracking != policy.submission_tracking_capacity_bytes
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}
fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PendingSamplingExtension>(),
        size_of::<SamplingExtensionBinding>(),
        size_of::<Result<PendingSamplingExtension, WorkingMemoryError>>(),
        size_of::<InferenceRequest>(),
        size_of::<TextStepContext>(),
        size_of::<HostMetadataFunding>(),
        size_of::<(u64, u64, bool)>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        crate::working_memory::text_preparation::synchronization_control_bytes()?,
        HostMetadataFunding::reservation_control_bytes(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

impl InferenceTextStep {
    // Original model roles remain valid after a sampler replacement, while the
    // old sampling/event roles must never be minted alongside the new account.
    pub(in crate::working_memory) fn original_sampling_is_current(
        &self,
    ) -> Result<bool, WorkingMemoryError> {
        let actual = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let state = actual
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if self.finished
            || run.fenced
            || run.active != Some(self.attempt)
            || run.control_pending.is_some()
        {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(run.sampling_epoch == 0)
    }
}
