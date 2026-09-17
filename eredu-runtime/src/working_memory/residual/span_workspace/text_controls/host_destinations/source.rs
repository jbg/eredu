//! Finite source-construction custody from the same accepted host component.
use super::*;
use crate::working_memory::OriginalHostSourceCustody;
mod program;
pub use program::{HostSourceConstructionProgram,OriginalHostSourceProgramBanks,OriginalHostSourceProgramError};

/// Disjoint cumulative source storage. Describes a request, never an allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSourceConstructionFacts {
    capacity: u64,
    attempts: usize,
    partitions: usize,
    protected: u64,
    program: Option<u64>,
    peak: Option<HostSourcePeakSelection>,
}
impl HostSourceConstructionFacts {
    /// Count actual producer layouts and finite child banks before acceptance.
    /// The containing producer supplies its complete managed layout, including
    /// receipt-bearing native/Rust ownership and its construction transports.
    pub fn new(
        capacity: u64,
        maximum_attempts: usize,
        maximum_partitions: usize,
    ) -> Result<Self, WorkingMemoryError> {
        let per_attempt = size_of::<OriginalHostSourceReceipt>()
            .checked_add(size_of::<OriginalHostSourceError>())
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<OriginalHostSourceReceipt, OriginalHostSourceError>,
                >())
            })
            .and_then(|n| n.checked_add(size_of::<HostDestinationCause>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        // Root plus actual split banks; a refused split never creates custody.
        let banks = maximum_partitions
            .checked_add(1)
            .and_then(|n| n.checked_mul(size_of::<OriginalHostSourceBank>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = per_attempt
            .checked_mul(maximum_attempts)
            .and_then(|n| n.checked_add(banks))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<Option<OriginalHostSourceBank>>()))
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<OriginalHostSourceBank, HostDestinationCause>,
                >())
            })
            // Exhaustion is a nonowning result, even with zero attempts.
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<OriginalHostSourceReceipt, OriginalHostSourceError>,
                >())
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        let protected = capacity
            .checked_add(u64::try_from(controls).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            capacity,
            attempts: maximum_attempts,
            partitions: maximum_partitions,
            protected,
            program: None,
            peak: None,
        })
    }
    /// Select reusable physical backing separately from cumulative constructor
    /// storage. The opaque selection is retained by the actual producer plan.
    /// The capacity passed to new must contain C only in this mode; this method
    /// adds the one physical peak B. Each constructor still reports C+B, and
    /// must consume its authentic split permit before allocation.
    pub fn with_peak_backing(
        mut self,
        selection: HostSourcePeakSelection,
    ) -> Result<Self, WorkingMemoryError> {
        if self.peak.is_some() || self.program.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.protected = self
            .protected
            .checked_add(selection.bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.peak = Some(selection);
        Ok(self)
    }
    /// Cumulative managed producer bytes, with no intermediate refund.
    pub const fn capacity_bytes(self) -> u64 {
        self.capacity
    }
    /// Reached construction attempts, including an owning refusal.
    pub const fn maximum_attempts(self) -> usize {
        self.attempts
    }
    /// Maximum one-level child banks; splitting performs no allocation.
    pub const fn maximum_partitions(self) -> usize {
        self.partitions
    }
    /// Complete disjoint contribution, including this bank's named transports.
    pub const fn protected_bytes(self) -> u64 {
        self.protected
    }
}

/// Allocation-free identity of one retained cold source-capacity selection.
/// Equal byte counts never compare as the same selection. This is descriptive;
/// only an accepted source bank may bind it to an actual retirement owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSourcePeakSelection {
    identity: u64,
    bytes: u64,
}
impl HostSourcePeakSelection {
    /// Create one identity in the actual retained producer, without granting bytes.
    pub fn new(bytes: u64) -> Result<Self, WorkingMemoryError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let identity = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(Self { identity, bytes })
    }
    /// Retain the same producer identity while describing one request's peak.
    pub const fn with_backing_bytes(self, bytes: u64) -> Self {
        Self {
            identity: self.identity,
            bytes,
        }
    }
    /// Whether two requirements came from that exact retained producer.
    pub const fn same_producer(self, other: Self) -> bool {
        self.identity == other.identity
    }
    /// Selected live physical backing requirement.
    pub const fn backing_bytes(self) -> u64 {
        self.bytes
    }
}
/// The retained backend factory's actual capacity owner. Identity names the
/// live owner (whose handle remains retained), never a capacity snapshot.
pub trait OriginalHostSourcePeakCapacity {
    /// Identity and peak retained by the actual selected producer.
    fn selection(&self) -> HostSourcePeakSelection;
    /// Stable identity while the actual retirement owner is retained.
    fn owner_identity(&self) -> usize;
    /// Immutable capacity of that owner.
    fn backing_bytes(&self) -> u64;
    /// The accepted request custody accompanying the owner.
    fn source_custody(&self) -> OriginalHostSourceCustody;
}

/// Move-only source-construction authority extracted from accepted host storage.
/// Child banks cannot split again, refill, merge, or refund their parent.
#[derive(Debug)]
pub struct OriginalHostSourceBank {
    program: Option<u64>,
    remaining: u64,
    peak: Option<HostSourcePeakSelection>,
    peak_owner: Option<usize>,
    attempts: usize,
    partitions: usize,
    reservation: Option<WorkingMemoryReservation>,
    controls: OriginalHostSourceCustody,
}
impl OriginalHostSourceBank {
    pub(super) fn new(
        facts: HostSourceConstructionFacts,
        reservation: WorkingMemoryReservation,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self::new_with_custody(facts, Some(reservation), controls.into())
    }
    pub(in crate::working_memory) fn new_with_custody(
        facts: HostSourceConstructionFacts,
        reservation: Option<WorkingMemoryReservation>,
        controls: OriginalHostSourceCustody,
    ) -> Self {
        Self {
            program: facts.program,
            remaining: facts.capacity,
            peak: facts.peak,
            peak_owner: None,
            attempts: facts.attempts,
            partitions: facts.partitions,
            reservation,
            controls,
        }
    }
    /// Actual accepted-owner identity, independent of equal scalar capacities.
    pub fn belongs_to(&self, controls: &OriginalTextControlGuard) -> bool {
        self.belongs_to_source(&controls.clone().into())
    }
    /// Exact accepted source account, including speculative role identity.
    pub fn belongs_to_source(&self, controls: &OriginalHostSourceCustody) -> bool {
        self.controls.same_source(controls)
    }
    /// Compare the complete unspent source population accepted by this bank.
    pub fn matches_facts(&self, facts: HostSourceConstructionFacts) -> bool {
        self.program == facts.program
            && self.remaining == facts.capacity
            && self.peak == facts.peak
            && self.attempts == facts.attempts
            && self.partitions == facts.partitions
    }
    /// Validate selection and request before allocating its actual capacity owner.
    pub fn validate_peak_selection(
        &self,
        selection: HostSourcePeakSelection,
        controls: &OriginalTextControlGuard,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_source_peak_selection(selection, &controls.clone().into())
    }
    /// Authenticate the same accepted account before allocating its capacity owner.
    pub fn validate_source_peak_selection(
        &self, selection: HostSourcePeakSelection, controls: &OriginalHostSourceCustody,
    ) -> Result<(), WorkingMemoryError> {
        self.controls.validate_account(self.reservation.as_ref())?;
        if self.peak != Some(selection) || self.peak_owner.is_some() || !self.belongs_to_source(controls) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Bind once, before splitting, to the exact selected native capacity owner.
    /// The caller retains that owner through every split and source allocation.
    pub fn bind_peak_capacity<C: OriginalHostSourcePeakCapacity>(
        &mut self,
        capacity: &C,
    ) -> Result<(), WorkingMemoryError> {
        let controls = capacity.source_custody();
        self.validate_source_peak_selection(capacity.selection(), &controls)?;
        if self.peak != Some(capacity.selection())
            || self.peak_owner.is_some()
            || !self.belongs_to_source(&controls)
            || capacity.backing_bytes() != capacity.selection().backing_bytes()
            || capacity.owner_identity() == 0
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.peak_owner = Some(capacity.owner_identity());
        Ok(())
    }
    /// Unspent cumulative source bytes.
    pub fn remaining_bytes(&self) -> u64 {
        self.remaining
    }
    /// Unspent finite construction attempts.
    pub fn remaining_attempts(&self) -> usize {
        self.attempts
    }
    /// Unissued child-bank population.
    pub fn remaining_partitions(&self) -> usize {
        self.partitions
    }
    /// Move a finite slice into one pending operation before it is activated.
    /// Validation/refusal is nonowning and changes no counters.
    pub fn split(&mut self, bytes: u64, attempts: usize) -> Result<Self, HostDestinationCause> {
        self.validate_split(bytes, attempts)?;
        Ok(self.split_validated(bytes, attempts))
    }
    pub(super) fn validate_split(
        &self,
        bytes: u64,
        attempts: usize,
    ) -> Result<(), HostDestinationCause> {
        self.controls
            .validate_account(self.reservation.as_ref())
            .map_err(HostDestinationCause::Memory)?;
        // All children must inherit the one accepted physical owner. Splitting
        // before binding could otherwise bind independent full-peak counters.
        if self.peak.is_some() && self.peak_owner.is_none() {
            return Err(HostDestinationCause::Memory(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        if self.program.is_some() {
            return Err(HostDestinationCause::Memory(WorkingMemoryError::IdentityMismatch));
        }
        if self.partitions == 0 || attempts > self.attempts {
            return Err(HostDestinationCause::Attempts);
        }
        if bytes > self.remaining {
            return Err(HostDestinationCause::Capacity {
                required: bytes,
                remaining: self.remaining,
            });
        }
        Ok(())
    }
    pub(super) fn split_validated(&mut self, bytes: u64, attempts: usize) -> Self {
        self.partitions -= 1;
        self.remaining -= bytes;
        self.attempts -= attempts;
        Self {
            program: None,
            remaining: bytes,
            peak: self.peak,
            peak_owner: self.peak_owner,
            attempts,
            partitions: 0,
            reservation: self.reservation.clone(),
            controls: self.controls.clone(),
        }
    }
    /// Debit before entering the actual constructor. No callback or allocator
    /// runs under the account health loan. The returned receipt must be moved
    /// into the producer's retirement owner before its first allocation.
    pub fn try_debit(
        &mut self,
        managed_bytes: u64,
    ) -> Result<OriginalHostSourceReceipt, OriginalHostSourceError> {
        if self.program.is_some() {
            return Err(OriginalHostSourceError {cause:HostDestinationCause::Memory(WorkingMemoryError::IdentityMismatch),receipt:None});
        }
        if self.attempts == 0 {
            return Err(OriginalHostSourceError {
                cause: HostDestinationCause::Attempts,
                receipt: None,
            });
        }
        self.attempts -= 1;
        let receipt = OriginalHostSourceReceipt {
            controls: self.controls.clone(),
        };
        let result = (|| {
            self.controls
                .validate_account(self.reservation.as_ref())
                .map_err(HostDestinationCause::Memory)?;
            if managed_bytes > self.remaining {
                return Err(HostDestinationCause::Capacity {
                    required: managed_bytes,
                    remaining: self.remaining,
                });
            }
            self.remaining -= managed_bytes;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(receipt),
            Err(cause) => Err(OriginalHostSourceError {
                cause,
                receipt: Some(receipt),
            }),
        }
    }
}

/// Accounting custody only. No payload, source, native owner or refill API.
/// The actual constructor keeps this after all charged storage in drop order.
#[derive(Debug)]
pub struct OriginalHostSourceReceipt {
    controls: OriginalHostSourceCustody,
}
impl OriginalHostSourceReceipt {
    /// Consume already debited source custody into metadata-only retention.
    /// No payload/source/native owner is retained, and no capacity is granted,
    /// refunded or exposed. The constructor must retain this after its storage.
    pub fn into_metadata(self) -> super::OriginalHostMetadataCustody {
        let metadata = self.controls.accounting();
        drop(self);
        metadata
    }

    /// Authenticate the same accepted request before native owner construction.
    pub fn belongs_to(&self, controls: &OriginalTextControlGuard) -> bool {
        self.controls.same_source(&controls.clone().into())
    }
}

/// Fixed cause followed by finite attempt custody. Exhaustion has no owner.
#[derive(Debug)]
pub struct OriginalHostSourceError {
    cause: HostDestinationCause,
    receipt: Option<OriginalHostSourceReceipt>,
}
impl OriginalHostSourceError {
    /// Exact refusal cause; no formatted native handler is involved.
    pub fn cause(&self) -> &HostDestinationCause {
        &self.cause
    }
    /// Whether this reached attempt retains original accounting custody.
    pub fn retains_receipt(&self) -> bool {
        self.receipt.is_some()
    }
}
impl std::fmt::Display for OriginalHostSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for OriginalHostSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

mod publication;
pub use publication::{
    OriginalHostSourceConstruction, OriginalHostSourceFailure, OriginalHostSourceFailureCause,
    OriginalHostSourcePending, OriginalHostSourceRefusal,
};
