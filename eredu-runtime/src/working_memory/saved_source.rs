//! Full request reservation while retaining an authenticated saved component.

use super::{
    Admission, FundedSamplerCopy, InferenceExecutionIdentity, MemoryLedger, Usage,
    WorkingMemoryCapacityHandoff, WorkingMemoryError, WorkingMemoryReservation,
    WorkingMemoryStorage, WorkspaceCopyCustody, residual::RegisteredStoragePin,
};

/// Actual immutable sampler and native-copy custody from one saved account,
/// together with already registered physical source roots. Neither an execution
/// identity, scalar byte count nor caller-edited report can create these owners.
///
/// This proves accounting custody only. A native provider must additionally
/// bind its complete actual source inventory and establish settled access. The
/// supplied registrations remain charged separately; this witness gives no
/// source credit, execution permission, completion proof or copy allocation.
#[derive(Debug)]
pub struct RegisteredSavedSamplingSource<'a, K: Ord + Send + 'static> {
    sampler: &'a FundedSamplerCopy,
    native: &'a WorkspaceCopyCustody,
    registered: WorkingMemoryStorage<K>,
}

impl WorkspaceCopyCustody {
    /// Associates two actual owners from the same saved account. Sources from
    /// different accounts are rejected even when their pool and bytes match.
    /// Health is checked again under the destination reservation commit lock.
    pub fn bind_saved_sampling_source<'a, K: Clone + Ord + Send + Sync + 'static>(
        &'a self,
        sampler: &'a FundedSamplerCopy,
        registered: WorkingMemoryStorage<K>,
    ) -> Result<RegisteredSavedSamplingSource<'a, K>, WorkingMemoryError> {
        let borrowed = sampler.borrow_funded();
        if !self.funding_source().same_account(borrowed.source())
            || !std::sync::Arc::ptr_eq(&self.execution().0, &borrowed.execution().0)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let source = RegisteredSavedSamplingSource {
            sampler,
            native: self,
            registered,
        };
        {
            let usage = self
                .pool()
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            source.validate(self.pool(), &usage)?;
        }
        Ok(source)
    }
}

// The trait and module are private. External callers cannot provide a callback
// into Usage or replace either source-owner/origin validation at the commit.
pub(super) trait SavedSourceValidation {
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError>;
    fn pin_control_bytes(&self) -> Result<usize, WorkingMemoryError>;
    fn pin(&self) -> RegisteredStoragePin;
}

impl<K: Clone + Ord + Send + Sync + 'static> SavedSourceValidation
    for RegisteredSavedSamplingSource<'_, K>
{
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        let sampler = self.sampler.borrow_funded();
        let native = self.native.funding_source();
        if !pool.same_ledger(self.native.pool())
            || !native.same_account(sampler.source())
            || !std::sync::Arc::ptr_eq(&self.native.execution().0, &sampler.execution().0)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        native.validate(usage, self.native.execution())?;
        sampler.source().validate(usage, sampler.execution())?;
        self.registered.validate_copy_source(pool, usage)
    }

    fn pin(&self) -> RegisteredStoragePin {
        RegisteredStoragePin::new(self.registered.clone())
    }
    fn pin_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        RegisteredStoragePin::single_control_bytes::<K>(self.registered.has_source_preparation())
    }
}

impl MemoryLedger {
    /// Quotes the complete reservation, including the authenticated saved-source
    /// pin and the fixed join used by an incremental source-credit reservation.
    /// This descriptor grants no reservation or source execution authority.
    pub fn saved_source_reservation_requirements<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        admission: &Admission,
        incremental: Option<&eredu_core::DomainMemoryRequirements>,
        source: &RegisteredSavedSamplingSource<'_, K>,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        if !self.same_ledger(source.native.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut requirements = self.reservation_requirements(admission, incremental)?;
        let pin = source
            .pin_control_bytes()?
            .checked_add(if incremental.is_some() {
                RegisteredStoragePin::pair_control_bytes(true)?
            } else {
                0
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        requirements.add_allocation(
            u64::try_from(pin).map_err(|_| WorkingMemoryError::Overflow)?,
            &self.host_placement_handle(),
        )?;
        Ok(requirements)
    }
    /// Reserves a complete full request bound, retaining its actual saved source
    /// registrations without discounting any source-inclusive term. Both saved
    /// owners and every supplied registration origin are checked atomically
    /// before capacity handoffs or reservation counters change.
    ///
    /// `admission` has the ordinary complete full-quotation contract of
    /// `reserve_with_capacity`; this does not turn it into a sealed incremental
    /// quote. The native composition must bind that quote to the exact saved
    /// data, request and selected mechanism. Source pins flow into work scopes
    /// and remain quarantined if their work is not certified.
    pub fn reserve_saved_source_with_capacity_handoff<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: eredu_core::MemoryLimits,
        handoffs: &[WorkingMemoryCapacityHandoff],
        source: &RegisteredSavedSamplingSource<'_, K>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_source(
            execution,
            admission,
            Some(capacity),
            None,
            handoffs,
            Some(source),
        )
    }
}
