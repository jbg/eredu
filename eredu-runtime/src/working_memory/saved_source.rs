//! Full request reservation while retaining an authenticated saved component.

use super::{
    residual::RegisteredStoragePin, Admission, FundedSamplerCopy, InferenceExecutionIdentity,
    Usage, WorkingMemoryCapacityHandoff, WorkingMemoryError, WorkingMemoryPool,
    WorkingMemoryReservation, WorkingMemoryStorage, WorkspaceCopyCustody,
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
    fn validate(&self, pool: &WorkingMemoryPool, usage: &Usage) -> Result<(), WorkingMemoryError>;
    fn pin(&self) -> RegisteredStoragePin;
}

impl<K: Clone + Ord + Send + Sync + 'static> SavedSourceValidation
    for RegisteredSavedSamplingSource<'_, K>
{
    fn validate(&self, pool: &WorkingMemoryPool, usage: &Usage) -> Result<(), WorkingMemoryError> {
        let sampler = self.sampler.borrow_funded();
        let native = self.native.funding_source();
        if !pool.same_domain(self.native.pool())
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
}

impl WorkingMemoryPool {
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
        capacity: u64,
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
