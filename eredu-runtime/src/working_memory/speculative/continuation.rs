//! Paid monotonic issuance extension for an authenticated restored future.
use super::*;
use crate::speculative::autoregressive::AutoregressiveContinuation;
use eredu_core::SpeculativeBufferAllocationError;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};

/// A failed extension cannot change the issuance limit or rewind attempts.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativeContinuationError {
    /// Fixed request/geometry/accounting refusal before any owned destination.
    #[error(transparent)]
    Request(#[from] WorkingMemoryError),
    /// Exact debit refused before allocation.
    #[error(transparent)]
    Metadata(#[from] HostMetadataFundingError),
    /// Destination allocation failed while retaining its actual paying account.
    #[error(transparent)]
    Allocation(#[from] SpeculativeBufferAllocationError),
}
impl OriginalSpeculativeRequest {
    /// Extend only this schedule's issuance limit. The existing source-pair
    /// host account pays the fixed controls; no native grant is constructed.
    /// The caller installs cursor limits only after success.
    pub fn prepare_continuation(
        &self,
        continuation: &AutoregressiveContinuation,
        funding: &HostMetadataFunding,
    ) -> Result<(), SpeculativeContinuationError> {
        self.prepare_continuation_slots(
            ScheduleIdentity::Autoregressive(continuation.identity()),
            continuation.previous_slots(),
            continuation.next_slots(),
            continuation.control_bytes().and_then(|n| {
                n.checked_add(size_of::<(
                    &Self,
                    &AutoregressiveContinuation,
                    &HostMetadataFunding,
                )>())
            }),
            funding,
        )
    }
    /// Grow only this exact Embedded cursor's issuance limit. Existing roles
    /// retain their own accounts; spent ordinals remain unchanged.
    pub fn prepare_embedded_continuation(
        &self,
        continuation: &crate::speculative::embedded_occurrence::EmbeddedContinuation,
        funding: &HostMetadataFunding,
    ) -> Result<(), SpeculativeContinuationError> {
        use crate::speculative::embedded_occurrence::EmbeddedContinuation;
        self.prepare_continuation_slots(
            ScheduleIdentity::Embedded(continuation.identity()),
            continuation.previous_slots(),
            continuation.next_slots(),
            continuation.control_bytes().and_then(|n| {
                n.checked_add(size_of::<(
                    &Self,
                    &EmbeddedContinuation,
                    &HostMetadataFunding,
                )>())
            }),
            funding,
        )
    }
    pub(super) fn prepare_continuation_slots(
        &self,
        identity: ScheduleIdentity,
        previous_slots: usize,
        next_slots: usize,
        caller_controls: Option<usize>,
        funding: &HostMetadataFunding,
    ) -> Result<(), SpeculativeContinuationError> {
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed
            || identity != self.identity
            || previous_slots != slots.limit
            || next_slots < slots.limit
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        self.ticket.status()?;
        let parts = [
            caller_controls,
            Some(size_of::<std::sync::MutexGuard<'_, RoleSlots>>()),
            Some(size_of::<Result<(), SpeculativeContinuationError>>()),
            Some(size_of::<SpeculativeContinuationError>()),
            Some(size_of::<(
                &Self,
                ScheduleIdentity,
                usize,
                usize,
                Option<usize>,
                &HostMetadataFunding,
            )>()),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), |n, part| n.checked_add(part?))
            .ok_or(WorkingMemoryError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        // Only fixed issuance limits change. Physical accounts are owned by
        // actual operations and their escaped values, never by this counter.
        slots.limit = next_slots;
        Ok(())
    }
}
