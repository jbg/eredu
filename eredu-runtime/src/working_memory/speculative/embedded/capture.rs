//! Early source-bound cumulative capture spending under the actual request.
use super::*;
use crate::working_memory::{
    CaptureRunLedger, OriginalCaptureSource, capture_run::embedded::Cumulative,
};
use eredu_core::{SharedStorageIdentity, capture::CaptureUsage};

/// Closed cumulative capture source retained outside snapshots and model roles.
///
/// Construction authenticates the actual request and immutable source. This
/// owner supplies no native scope, role, source publication or capture claim.
/// Its fixed host account and spending survive all aliases without reset.
#[derive(Debug, Clone)]
pub struct OriginalEmbeddedCaptureLineage {
    source: SharedStorageIdentity,
    ledger: CaptureRunLedger,
}
impl OriginalEmbeddedCaptureLineage {
    /// Compare the exact source storage, without accepting equal declarations.
    pub fn validate_source(
        &self,
        source: &OriginalCaptureSource,
    ) -> Result<(), WorkingMemoryError> {
        if &self.source == source.plan().storage_identity() {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(crate) fn ledger(&self) -> &CaptureRunLedger {
        &self.ledger
    }
    pub(in crate::working_memory) fn validate_current(
        &self,
        source: &OriginalCaptureSource,
        current: Option<&Cumulative>,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_source(source)?;
        match current {
            Some(current)
                if current.source == self.source && current.ledger.same_storage(&self.ledger) =>
            {
                Ok(())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}
impl OriginalSpeculativeRequest {
    /// Create the exact capture ledger before the first quote, or borrow its
    /// existing owner. Repeated calls allocate nothing and never refund usage.
    /// The fixed ledger account is admitted by the same actual request capacity
    /// and original account worker as its model roles, with no role consumed.
    pub fn prepare_embedded_capture_lineage(
        &self,
        source: &OriginalCaptureSource,
    ) -> Result<OriginalEmbeddedCaptureLineage, SpeculativeRequestError> {
        source.validate_pool(self.ticket.pool())?;
        self.ticket.status()?;
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed || !matches!(self.identity, ScheduleIdentity::Embedded(_)) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(current) = &slots.model_capture {
            if &current.source != source.plan().storage_identity() {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            current.ledger.inspect_usage()?;
            return Ok(OriginalEmbeddedCaptureLineage {
                source: current.source.clone(),
                ledger: current.ledger.clone(),
            });
        }
        let controls = [
            size_of::<OriginalEmbeddedCaptureLineage>(),
            size_of::<Cumulative>(),
            size_of::<Result<OriginalEmbeddedCaptureLineage, SpeculativeRequestError>>(),
            size_of::<(&Self, &OriginalCaptureSource)>(),
            size_of::<std::sync::MutexGuard<'_, RoleSlots>>(),
            size_of::<Result<CaptureUsage, WorkingMemoryError>>(),
            OriginalCaptureSource::validation_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            CaptureRunLedger::inspection_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = controls
            .checked_add(CaptureRunLedger::control_bytes()?)
            .and_then(|n| n.checked_add(account_control_bytes().ok()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        let ticket = accept(
            self.ticket.pool(),
            &self.execution,
            self.capacity,
            bytes,
            bytes,
        )?;
        if let Err(cause) = ticket.status() {
            return Err(SpeculativeRequestError {
                cause,
                ticket: Some(ticket),
            });
        }
        let lineage = OriginalEmbeddedCaptureLineage {
            source: source.plan().storage_identity().clone(),
            ledger: CaptureRunLedger::new_request(ticket),
        };
        slots.model_capture = Some(Cumulative {
            source: lineage.source.clone(),
            ledger: lineage.ledger.clone(),
        });
        Ok(lineage)
    }

    /// Read the settled usage only when the exact retained request and source
    /// still own this same lineage. This starts no claim or ledger mutation.
    pub fn inspect_embedded_capture_lineage_usage(
        &self,
        source: &OriginalCaptureSource,
        lineage: &OriginalEmbeddedCaptureLineage,
    ) -> Result<CaptureUsage, WorkingMemoryError> {
        source.validate_pool(self.ticket.pool())?;
        let slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed || !matches!(self.identity, ScheduleIdentity::Embedded(_)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        lineage.validate_current(source, slots.model_capture.as_ref())?;
        lineage.ledger.inspect_usage()
    }
}
