//! Fixed host destinations funded separately before the original span is sealed.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

use crate::working_memory::qualified_storage::{self, ReserveError};
mod metadata;
pub use metadata::{OriginalHostMetadataCustody, OriginalHostMetadataVec};
mod source;
pub use source::{
    HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError, OriginalHostSourceBank, OriginalHostSourceConstruction,
    OriginalHostSourceError, OriginalHostSourceFailure, OriginalHostSourceFailureCause,
    OriginalHostSourceReceipt, OriginalHostSourceRefusal,
};
pub use source::{
    HostSourcePeakSelection, OriginalHostSourcePeakCapacity, OriginalHostSourcePending,
};

/// Finite additional managed storage. This is a request, not an allocation grant.
/// It excludes allocator-private bookkeeping, not exposed Vec spare capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostDestinationFacts {
    capacity: u64,
    attempts: usize,
    partitions: usize,
    protected: u64,
    source: Option<HostSourceConstructionFacts>,
}
impl HostDestinationFacts {
    /// Describe a disjoint host component using the qualified fresh-Vec producer.
    /// Existing Q/native/source components must not already contain these bytes.
    pub fn new(capacity: u64, maximum_attempts: usize) -> Result<Self, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let reserve_controls = qualified_storage::reserve_control_bytes::<u8>()?;
        let per_attempt = size_of::<OriginalHostVec<u8>>()
            .checked_add(size_of::<OriginalHostVecError<u8>>())
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<OriginalHostVec<u8>, OriginalHostVecError<u8>>,
                >())
            })
            .and_then(|n| n.checked_add(size_of::<HostDestinationCause>()))
            .and_then(|n| n.checked_add(size_of::<Layout>()))
            .and_then(|n| n.checked_add(reserve_controls))
            .ok_or(WorkingMemoryError::Overflow)?;
        // A nonowning exhaustion result exists even for a zero-attempt bank.
        // It neither allocates nor clones custody; price its fixed call frame.
        let refusal = size_of::<OriginalHostVecError<u8>>()
            .checked_add(size_of::<
                Result<OriginalHostVec<u8>, OriginalHostVecError<u8>>,
            >())
            .and_then(|n| n.checked_add(size_of::<HostDestinationCause>()))
            .and_then(|n| n.checked_add(size_of::<&mut OriginalHostDestinationBank>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = per_attempt
            .checked_mul(maximum_attempts)
            .and_then(|n| n.checked_add(size_of::<OriginalHostDestinationBank>()))
            .and_then(|n| n.checked_add(size_of::<Option<OriginalHostDestinationBank>>()))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(refusal))
            .ok_or(WorkingMemoryError::Overflow)?;
        let protected = capacity
            .checked_add(u64::try_from(controls).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            capacity,
            attempts: maximum_attempts,
            partitions: 0,
            protected,
            source: None,
        })
    }
    /// Price finite one-level destination tickets before acceptance. No payload
    /// is allocated by a split. Existing source sub-banks remain disjoint.
    pub fn with_partitions(
        mut self,
        maximum_partitions: usize,
    ) -> Result<Self, WorkingMemoryError> {
        if self.partitions != 0 {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let child = size_of::<OriginalHostDestinationBank>()
            .checked_add(size_of::<
                Result<OriginalHostDestinationBank, HostDestinationCause>,
            >())
            .ok_or(WorkingMemoryError::Overflow)?;
        let extra = child
            .checked_mul(maximum_partitions)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.protected = self
            .protected
            .checked_add(u64::try_from(extra).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.partitions = maximum_partitions;
        Ok(self)
    }
    /// Number of one-level child tickets permitted by the actual accepted bank.
    pub const fn maximum_partitions(self) -> usize {
        self.partitions
    }
    /// Add a separately derived finite source-construction component once.
    /// It must not already be counted in the vector or original work component.
    pub fn with_source_constructions(
        mut self,
        facts: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.source.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.protected = self
            .protected
            .checked_add(facts.protected_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.source = Some(facts);
        Ok(self)
    }
    /// Replace an exactly identified cold source description when its retained
    /// producer composes an aggregate program. This adjusts descriptive protected
    /// bytes only; it creates no accepted bank, source authority or spending reset.
    pub fn replace_source_constructions(
        self,
        expected: Option<HostSourceConstructionFacts>,
        next: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.source != expected {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let without = match expected {
            Some(expected) => self.without_source(expected)?,
            None => self,
        };
        without.with_source_constructions(next)
    }
    // Describes only the component left after a checked move of its actual
    // SourceBank. It changes no accepted funding or cumulative spending.
    pub(super) fn without_source(
        mut self,
        expected: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.source != Some(expected) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.protected = self
            .protected
            .checked_sub(expected.protected_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.source = None;
        Ok(self)
    }
    /// Actual disjoint source-construction facts, if selected before acceptance.
    pub fn source_constructions(self) -> Option<HostSourceConstructionFacts> {
        self.source
    }
    /// Additional protected host bytes, including the finite bank/owner controls.
    pub const fn protected_bytes(self) -> u64 {
        self.protected
    }
    /// Maximum cumulative exposed payload allocations; no intermediate refund.
    pub const fn capacity_bytes(self) -> u64 {
        self.capacity
    }
    /// Maximum allocation attempts, including a reached reserve/refusal.
    pub const fn maximum_attempts(self) -> usize {
        self.attempts
    }
}

impl PreparedTextControlWorkspace {
    pub(super) fn with_host_source_constructions(
        mut self,
        facts: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        let Some(previous) = self.binding.host_destinations else {
            return self.with_host_destinations(
                HostDestinationFacts::new(0, 0)?.with_source_constructions(facts)?,
            );
        };
        let next = previous.with_source_constructions(facts)?;
        let added = next
            .protected_bytes()
            .checked_sub(previous.protected_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.binding.facts.work = self
            .binding
            .facts
            .work
            .map(|n| n.checked_add(added).ok_or(WorkingMemoryError::Overflow))
            .transpose()?;
        self.binding.host_destinations = Some(next);
        Ok(self)
    }

    /// Add this disjoint contribution before genuine seal/admission, once only.
    pub fn with_host_destinations(
        mut self,
        facts: HostDestinationFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.host_destinations.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.binding.facts.work = self
            .binding
            .facts
            .work
            .map(|n| {
                n.checked_add(facts.protected)
                    .ok_or(WorkingMemoryError::Overflow)
            })
            .transpose()?;
        self.binding.host_destinations = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Extract the separately protected bank once. A cloned control guard cannot
    /// construct, refill or replace it.
    pub fn take_host_destinations(
        &mut self,
    ) -> Result<Option<OriginalHostDestinationBank>, WorkingMemoryError> {
        let facts = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .host_destinations;
        let Some(facts) = facts else {
            return Ok(None);
        };
        if self.host_destinations_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalHostDestinationBank {
            sources: facts.source.map(|facts| {
                OriginalHostSourceBank::new(
                    facts,
                    self.reservation().clone(),
                    self.controls.clone(),
                )
            }),
            remaining: facts.capacity,
            attempts: facts.attempts,
            partitions: facts.partitions,
            reservation: self.reservation().clone(),
            controls: self.controls.clone(),
        };
        self.host_destinations_taken = true;
        Ok(Some(bank))
    }
}

/// Move-only allocation authority from one actual accepted request.
#[derive(Debug)]
pub struct OriginalHostDestinationBank {
    sources: Option<OriginalHostSourceBank>,
    remaining: u64,
    attempts: usize,
    partitions: usize,
    reservation: WorkingMemoryReservation,
    controls: OriginalTextControlGuard,
}
impl OriginalHostDestinationBank {
    /// Move the actual accepted source component out once, without touching Vec
    /// capacity or creating source/native allocations. Absence grants nothing.
    pub fn take_source_constructions(&mut self) -> Option<OriginalHostSourceBank> {
        self.sources.take()
    }

    // Paired move used only by the original prefill source handoff. Empty
    // wrapper recreation carries zero vector capacity and the accepted account.
    pub(super) fn restore_source_component(
        bank: &mut Option<Self>, source: &mut Option<OriginalHostSourceBank>,
        expected: HostSourceConstructionFacts, remaining: Option<HostDestinationFacts>,
        controls: &OriginalTextControlGuard, reservation: &WorkingMemoryReservation,
    ) -> Result<HostDestinationFacts, WorkingMemoryError> {
        controls.validate_reservation(reservation)?;
        let actual = source.as_ref().ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !actual.belongs_to(controls) || !actual.matches_facts(expected) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let facts = match (bank.as_ref(), remaining) {
            (Some(bank), Some(facts)) if bank.belongs_to(controls) && bank.matches_facts(facts)
                && facts.source_constructions().is_none() => facts,
            (None, None) => HostDestinationFacts::new(0, 0)?,
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        };
        let restored = facts.with_source_constructions(expected)?;
        // No fallible step remains after either accepted owner is moved.
        let actual = source.take().expect("validated original source component");
        match bank {
            Some(bank) => bank.sources = Some(actual),
            None => *bank = Some(Self { sources: Some(actual), remaining: 0, attempts: 0,
                partitions: 0, reservation: reservation.clone(), controls: controls.clone() }),
        }
        Ok(restored)
    }
    /// Compare the actual accepted host owner; equal scalar capacities do not match.
    pub fn belongs_to(&self, controls: &OriginalTextControlGuard) -> bool {
        self.controls.custody.same(&controls.custody)
    }
    /// Validate the entire unconsumed selected population, including its source
    /// component. Equal facts alone issue no authority; callers also check owner.
    pub fn matches_facts(&self, facts: HostDestinationFacts) -> bool {
        self.remaining == facts.capacity
            && self.attempts == facts.attempts
            && self.partitions == facts.partitions
            && match (self.sources.as_ref(), facts.source) {
                (Some(bank), Some(facts)) => bank.matches_facts(facts),
                (None, None) => true,
                _ => false,
            }
    }
    /// Remaining cumulative payload capacity; not a grant or allocator telemetry.
    pub fn remaining_bytes(&self) -> u64 {
        self.remaining
    }
    /// Remaining calls admitted by this bank's finite population.
    pub fn remaining_attempts(&self) -> usize {
        self.attempts
    }
    /// Remaining one-level child tickets. Retiring a child never replenishes it.
    pub fn remaining_partitions(&self) -> usize {
        self.partitions
    }
    /// Split both actual components atomically before constructing a pending row.
    /// All refusal/health checks precede either mutation; returned children cannot
    /// split again. `source` must match the actual retained component's presence.
    pub fn split(
        &mut self,
        bytes: u64,
        attempts: usize,
        source: Option<(u64, usize)>,
    ) -> Result<Self, HostDestinationCause> {
        self.controls
            .validate_reservation(&self.reservation)
            .map_err(HostDestinationCause::Memory)?;
        if self.partitions == 0 || attempts > self.attempts {
            return Err(HostDestinationCause::Attempts);
        }
        if bytes > self.remaining {
            return Err(HostDestinationCause::Capacity {
                required: bytes,
                remaining: self.remaining,
            });
        }
        match (self.sources.as_ref(), source) {
            (Some(bank), Some((bytes, attempts))) => bank.validate_split(bytes, attempts)?,
            (None, None) => {}
            _ => {
                return Err(HostDestinationCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
        }
        // No fallible call/callback remains between paired counter transitions.
        let sources = match (self.sources.as_mut(), source) {
            (Some(bank), Some((bytes, attempts))) => Some(bank.split_validated(bytes, attempts)),
            (None, None) => None,
            _ => unreachable!("validated component presence"),
        };
        self.partitions -= 1;
        self.remaining -= bytes;
        self.attempts -= attempts;
        Ok(Self {
            sources,
            remaining: bytes,
            attempts,
            partitions: 0,
            reservation: self.reservation.clone(),
            controls: self.controls.clone(),
        })
    }
    /// Admit before the single reserve. Owning results retain the original hold,
    /// with the Vec destroyed before its receipt. Exhaustion creates no owner.
    pub fn try_vec<T: Copy>(
        &mut self,
        elements: usize,
    ) -> Result<OriginalHostVec<T>, OriginalHostVecError<T>> {
        // Refuse before constructing an owner, cloning custody or checking
        // health. Every branch that can return an owner spends a finite attempt.
        if self.attempts == 0 {
            return Err(OriginalHostVecError {
                cause: HostDestinationCause::Attempts,
                destination: None,
            });
        }
        self.attempts -= 1;
        let mut destination = OriginalHostVec {
            values: Vec::new(),
            limit: elements,
            started: false,
            _receipt: HostDestinationReceipt {
                _controls: self.controls.clone(),
            },
        };
        let result = (|| {
            if !qualified_storage::qualified() {
                return Err(HostDestinationCause::UnknownProducer);
            }
            self.controls
                .validate_reservation(&self.reservation)
                .map_err(HostDestinationCause::Memory)?;
            if size_of::<T>() == 0
                || size_of::<OriginalHostVec<T>>() != size_of::<OriginalHostVec<u8>>()
            {
                return Err(HostDestinationCause::Layout);
            }
            let layout = Layout::array::<T>(elements).map_err(|_| HostDestinationCause::Layout)?;
            let bytes = u64::try_from(layout.size()).map_err(|_| HostDestinationCause::Layout)?;
            if bytes > self.remaining {
                return Err(HostDestinationCause::Capacity {
                    required: bytes,
                    remaining: self.remaining,
                });
            }
            self.remaining -= bytes;
            qualified_storage::reserve_empty(&mut destination.values, elements).map_err(
                |cause| match cause {
                    ReserveError::Unqualified => HostDestinationCause::UnknownProducer,
                    ReserveError::Layout => HostDestinationCause::Layout,
                    ReserveError::Reserve(cause) => HostDestinationCause::Reserve(cause),
                    ReserveError::Contract => HostDestinationCause::ProducerContract,
                },
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(destination),
            Err(cause) => Err(OriginalHostVecError {
                cause,
                destination: Some(destination),
            }),
        }
    }
}

#[derive(Debug)]
struct HostDestinationReceipt {
    _controls: OriginalTextControlGuard,
}

/// Owning fixed destination. Only slices escape; its Vec and receipt never split.
#[derive(Debug)]
pub struct OriginalHostVec<T> {
    values: Vec<T>,
    limit: usize,
    started: bool,
    // Last. Enclosing charged owners must also free their shell before this owner.
    _receipt: HostDestinationReceipt,
}
impl<T> OriginalHostVec<T> {
    /// Actual initialized extent, which can be partial after failed fill.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    /// Borrow initialized storage without providing a growing Vec interface.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
    /// Actual exposed capacity, separate from length and allocator-private bytes.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }
    /// Requested no-growth element limit.
    pub fn requested_elements(&self) -> usize {
        self.limit
    }
    /// Actual initialized element count.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Whether initialized storage is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Fill once. The source supplies expected independently of the iterator. Both over-
    /// and under-production retain their actual prefix; iterator size hints are ignored.
    pub fn try_fill<I: IntoIterator<Item = T>>(
        &mut self,
        expected: usize,
        input: I,
    ) -> Result<(), HostDestinationCause> {
        if self.started {
            return Err(HostDestinationCause::Consumed);
        }
        self.started = true;
        if expected > self.limit {
            return Err(HostDestinationCause::Extent {
                expected: self.limit,
                actual: expected,
            });
        }
        for value in input {
            if self.values.len() == expected || self.values.len() == self.values.capacity() {
                return Err(HostDestinationCause::Extent {
                    expected,
                    actual: self.values.len().saturating_add(1),
                });
            }
            self.values.push(value);
        }
        if self.values.len() != expected {
            return Err(HostDestinationCause::Extent {
                expected,
                actual: self.values.len(),
            });
        }
        Ok(())
    }
}
impl<T> AsRef<[T]> for OriginalHostVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T> AsMut<[T]> for OriginalHostVec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Typed preflight, capacity or actual reserve/fill cause.
#[derive(Debug, thiserror::Error)]
pub enum HostDestinationCause {
    /// The actual build has no qualified fresh-Vec implementation.
    #[error("fresh host Vec producer is unqualified")]
    UnknownProducer,
    /// Original request/account health validation failed.
    #[error("{0}")]
    Memory(#[source] WorkingMemoryError),
    /// Finite allocation-attempt population is spent.
    #[error("host destination attempts exhausted")]
    Attempts,
    /// Requested typed extent is not representable/supported.
    #[error("host destination layout is unsupported")]
    Layout,
    /// Protected bytes are insufficient before allocator entry.
    #[error("host destination needs {required} bytes, {remaining} remain")]
    Capacity {
        /// Requested managed bytes.
        required: u64,
        /// Protected bytes still available.
        remaining: u64,
    },
    /// The real reserve failed after admission.
    #[error("{0}")]
    Reserve(#[source] TryReserveError),
    /// Actual storage contradicts the qualified implementation.
    #[error("fresh host Vec producer contract failed")]
    ProducerContract,
    /// Fill is once-only, including a failed fill.
    #[error("host destination fill already attempted")]
    Consumed,
    /// A checked iterator/source extent differs.
    #[error("host destination expected {expected} elements, received {actual}")]
    Extent {
        /// Checked expected element count.
        expected: usize,
        /// Actual encountered/advertised count.
        actual: usize,
    },
}

/// Fixed exhaustion refusal or the actual finite attempt's owned failed prefix.
pub struct OriginalHostVecError<T> {
    cause: HostDestinationCause,
    destination: Option<OriginalHostVec<T>>,
}
impl<T> OriginalHostVecError<T> {
    /// Exact fixed or allocator cause.
    pub fn cause(&self) -> &HostDestinationCause {
        &self.cause
    }
    /// Move the cause and optional intact destination; the Vec and receipt never split.
    pub fn into_parts(self) -> (HostDestinationCause, Option<OriginalHostVec<T>>) {
        (self.cause, self.destination)
    }
    /// The actual partial destination under its original hold. Exhaustion has none.
    pub fn destination(&self) -> Option<&OriginalHostVec<T>> {
        self.destination.as_ref()
    }
}

impl<T> std::fmt::Debug for OriginalHostVecError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalHostVecError")
            .field("cause", &self.cause)
            .field(
                "length",
                &self.destination.as_ref().map(OriginalHostVec::len),
            )
            .field(
                "capacity",
                &self.destination.as_ref().map(OriginalHostVec::capacity),
            )
            .finish()
    }
}
impl<T> std::fmt::Display for OriginalHostVecError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for OriginalHostVecError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl PreparedTextControlWorkspace {
    /// Price a separately owned finite immutable-output source component before
    /// acceptance. It does not split or replace a cache/Host source allowance.
    pub fn with_output_source_constructions(mut self, facts: HostSourceConstructionFacts) -> Result<Self, WorkingMemoryError> {
        if !qualified_storage::qualified() { return Err(WorkingMemoryError::UnknownBound); }
        if self.binding.output_sources.is_some() { return Err(WorkingMemoryError::AlreadyStarted); }
        self.binding.facts.work = self.binding.facts.work
            .map(|bytes| bytes.checked_add(facts.protected_bytes()).ok_or(WorkingMemoryError::Overflow)).transpose()?;
        self.binding.output_sources = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Consume the independent immutable-output source bank once. Custody aliases
    /// and saved model state cannot recreate the bank or refund its attempts.
    pub fn take_output_source_constructions(&mut self) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        let facts = self.workspace().control_binding().ok_or(WorkingMemoryError::IdentityMismatch)?.output_sources;
        let Some(facts) = facts else { return Ok(None); };
        if self.output_sources_taken { return Err(WorkingMemoryError::AlreadyStarted); }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalHostSourceBank::new(facts, self.reservation().clone(), self.controls.clone());
        self.output_sources_taken = true;
        Ok(Some(bank))
    }
}
