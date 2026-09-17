//! Transfers a reserved envelope to independently retiring physical storage.

use super::{
    InferenceExecutionIdentity, PreparedAccountCommit, Usage, WorkingMemoryError,
    WorkingMemoryPool, WorkingMemoryReservation, WorkingMemoryStorage,
    residual::RegisteredStoragePin, text_preparation::RequestStart,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};

mod accounts;
pub(super) use accounts::{
    AccountLedger, AccountNode, AccountTicket, PendingAccount, PendingOriginal, RetiringAccount,
};
mod capture_source;
use capture_source::{ActiveCaptureSpan, CaptureSourceSlot};
pub use capture_source::{
    CaptureSourceSegment, PreparedPrefillChunkRetention, SettledCaptureSourceParcel,
    SettledPrefillChunkRetention,
};
pub(super) fn capture_source_control_bytes() -> usize {
    capture_source::control_bytes()
}
// Closed short-fragment construction keeps rollback alive while preparing its
// target writer. This exposes representation sizing only, never hold authority.
pub(super) fn capture_source_rollback_control_bytes() -> usize {
    std::mem::size_of::<capture_tensor::CaptureSourceRollback<'_>>()
}

mod capture_tensor;
pub(super) use capture_tensor::CaptureTensorCustody;

mod capacity_handoff;
pub(in crate::working_memory) mod native_partition;
mod no_decoder_prompt;
mod prepared_scope;
pub use prepared_scope::PreparedWorkingMemoryFundingScope;
mod span_workspace;
pub(super) use capacity_handoff::CapacityHandoffPlan;
pub use capacity_handoff::WorkingMemoryCapacityHandoff;
pub(in crate::working_memory) use capture_source::CapturePinIdentity;
pub(super) use span_workspace::{
    RawSpanHostOwner, SpanHostCustody, SpanHostOwner,
    retirement_control_bytes as span_retirement_control_bytes,
};
#[cfg(test)]
pub(super) use span_workspace::{after_span_attachment, quarantine_source_after_next_attachment};

#[cfg(test)]
pub(in crate::working_memory) mod tests;

#[derive(Debug)]
pub(super) struct FundingState {
    pub(super) remaining: u64,
    // Protected subset of host_held, released only after the canonical node shell.
    control_floor: u64,
    // A subset of remaining retained for closed host payloads and their
    // construction/replacement overlap. Native publication cannot spend it.
    pub(super) host_held: u64,
    // One protected request-wide partition; Some(0) remains a live owner.
    native_held: Option<u64>,
    native_issued: bool,
    // Weak identity only: never a new owner of the stamp, custody or pool.
    // An expired marker remains closed until exact canonical removal.
    active_span: Option<ActiveCaptureSpan>,
    pub(super) allocations: usize,
    pub(super) registrations: usize,
    pub(super) scopes: usize,
    // Only scopes that can submit native work retain transient headroom.
    pub(super) native_scopes: usize,
    run_open: bool,
    quarantined: bool,
    metadata_live: bool,
    capacity: Option<u64>,
    execution: Weak<()>,
    // Every uncertified scope contributes its complete borrowed inventory.
    // Scopes can add distinct source pins beyond the original run's bundle.
    // Permanent quarantine keeps these accounting-only owners (and therefore
    // possibly the pool) alive until a future certified cleanup mechanism.
    quarantined_borrowed: Option<Box<QuarantinedStoragePins>>,
    // Source lifetime only. The enclosing prepared-copy H already priced these
    // constructor controls; this grants no B bytes or publication permission.
    // Kept through the final execution Weak and AccountNode shell retirement.
    preparation: Option<eredu_core::HostPreparationAuthority>,
}

// A node is prepared before taking the usage lock. Linking only moves owners:
// no provider clone, pin destruction, or allocation occurs under that lock.
#[derive(Debug)]
struct QuarantinedStoragePins {
    _pin: RegisteredStoragePin,
    next: Option<Box<Self>>,
}

impl FundingState {
    pub(in crate::working_memory) fn accepted_capacity(&self) -> Option<u64> {
        self.capacity
    }
    fn empty() -> Self {
        Self {
            remaining: 0,
            control_floor: 0,
            host_held: 0,
            native_held: None,
            native_issued: false,
            active_span: None,
            allocations: 0,
            registrations: 0,
            scopes: 0,
            native_scopes: 0,
            run_open: false,
            quarantined: false,
            metadata_live: false,
            capacity: None,
            execution: Weak::new(),
            quarantined_borrowed: None,
            preparation: None,
        }
    }

    pub(super) fn new(
        remaining: u64,
        capacity: Option<u64>,
        execution: &InferenceExecutionIdentity,
        metadata_live: bool,
        scopes: usize,
        native_scopes: usize,
    ) -> Self {
        Self {
            remaining,
            control_floor: 0,
            host_held: 0,
            native_held: None,
            native_issued: false,
            active_span: None,
            allocations: 0,
            registrations: 0,
            scopes,
            native_scopes,
            run_open: true,
            quarantined: false,
            metadata_live,
            capacity,
            execution: Arc::downgrade(&execution.0),
            quarantined_borrowed: None,
            // Closed authority Clone is allocation-free and cannot run payload
            // callbacks under Usage. Every retirement happens after unlock.
            preparation: execution.1.clone(),
        }
    }

    pub(super) fn protected_remaining(&self) -> Result<u64, WorkingMemoryError> {
        self.host_held
            .checked_add(self.native_held.unwrap_or(0))
            .ok_or(WorkingMemoryError::Poisoned)
    }

    pub(super) fn spendable_remaining(&self) -> Result<u64, WorkingMemoryError> {
        self.remaining
            .checked_sub(self.protected_remaining()?)
            .ok_or(WorkingMemoryError::Poisoned)
    }

    fn retains_workspace(&self) -> bool {
        self.run_open || self.native_scopes != 0 || self.quarantined || self.active_span.is_some()
    }

    // Compute both counters before a mint commits any host balance or owners.
    fn scope_counts_after(
        &self,
        total: usize,
        native: usize,
    ) -> Result<(usize, usize), WorkingMemoryError> {
        let scopes = self
            .scopes
            .checked_add(total)
            .ok_or(WorkingMemoryError::Overflow)?;
        let native_scopes = self
            .native_scopes
            .checked_add(native)
            .ok_or(WorkingMemoryError::Overflow)?;
        if native > total || native_scopes > scopes {
            return Err(WorkingMemoryError::Poisoned);
        }
        Ok((scopes, native_scopes))
    }

    fn close_scope(&mut self, purpose: ScopePurpose) {
        self.scopes = self.scopes.checked_sub(1).expect("live scope count");
        if purpose == ScopePurpose::Native {
            self.native_scopes = self
                .native_scopes
                .checked_sub(1)
                .expect("live native scope count");
        }
    }

    // Source health is deliberately separate from exclusive account spending.
    // Existing held-host operations and copying FROM this source into another
    // independently admitted account do not spend this account's headroom.
    pub(super) fn validate_span_spend(
        &self,
        scope: Option<&WorkingMemoryFundingScope>,
    ) -> Result<(), WorkingMemoryError> {
        if self
            .active_span
            .as_ref()
            .is_some_and(|span| !span.matches(scope))
        {
            Err(WorkingMemoryError::ExecutionFenced)
        } else {
            Ok(())
        }
    }

    pub(super) fn validate_native_publication(
        &self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        scope.validate_native_purpose()?;
        if self.quarantined || self.native_scopes == 0 {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        self.validate_span_spend(Some(scope))
    }

    pub(super) fn validate_registered_copy_origin(&self) -> Result<(), WorkingMemoryError> {
        if self.quarantined {
            Err(WorkingMemoryError::ExecutionFenced)
        } else {
            Ok(())
        }
    }
}

impl WorkingMemoryReservation {
    /// Converts a fresh, uniquely owned reservation into immutable request
    /// evidence and a separate owner of its execution funding. Conversion
    /// changes neither the charge nor the historical peak. Existing clones or
    /// begun preparation reject conversion.
    ///
    /// The returned metadata retains the original identity, byte bound and
    /// domain exclusion, but does not keep unused workspace charged after the
    /// run closes and every work scope is certified. Native work must retain a
    /// scope, and every surviving allocation must be published through it.
    pub fn into_funding(mut self) -> Result<(Self, WorkingMemoryFundingRun), WorkingMemoryError> {
        let run = self.start_funding()?;
        Ok((self, run))
    }

    /// Starts the same fresh unique funding account while retaining the actual
    /// reservation in its caller on every rejection or unwind. A second start
    /// rejects. This permits an enclosing source owner to retire its payload
    /// before the accepted reservation/run; it grants no new bytes or account.
    /// All uniqueness, freshness, accounting and borrowed-root transfers match
    /// `into_funding`. The returned run must retain every subsequent scope.
    pub fn start_funding(&mut self) -> Result<WorkingMemoryFundingRun, WorkingMemoryError> {
        let reservation = self
            .0
            .get_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if reservation.funding.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let start = reservation
            .start
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if !matches!(*start, RequestStart::Fresh) {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let pool = reservation.pool.clone();
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let id = reservation.account_id;
        usage.funding.start(id)?;
        reservation.funding = Some(id);
        drop(usage);
        drop(start);
        // Historical request metadata must not pin borrowed allocations after
        // funding closes. Future work and independent scopes own this pin.
        let borrowed_storage = reservation.borrowed_storage.take();
        Ok(WorkingMemoryFundingRun {
            pool,
            id,
            open: true,
            handoff_taken: false,
            borrowed_storage,
        })
    }
}

/// Borrowed only from an owned concrete host component. No metadata-only
/// reservation or raw source plan can construct that component's public proof.
#[derive(Debug, Clone, Copy)]
pub(super) enum FundingSource<'a> {
    HostScope(&'a WorkingMemoryFundingScope),
    NativeScope(&'a WorkingMemoryFundingScope),
    CopyRun(&'a WorkingMemoryFundingRun),
}

impl FundingSource<'_> {
    pub(super) fn same_account(&self, other: FundingSource<'_>) -> bool {
        let id = |source: FundingSource<'_>| match source {
            FundingSource::HostScope(scope) | FundingSource::NativeScope(scope) => scope.id,
            FundingSource::CopyRun(run) => run.id,
        };
        self.pool().same_domain(other.pool()) && id(*self) == id(other)
    }

    pub(super) fn pool(&self) -> &WorkingMemoryPool {
        match self {
            Self::HostScope(scope) | Self::NativeScope(scope) => &scope.pool,
            Self::CopyRun(run) => &run.pool,
        }
    }

    pub(super) fn validate(
        &self,
        usage: &Usage,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        let (id, active) = match self {
            Self::HostScope(scope) | Self::NativeScope(scope) => (scope.id, scope.active),
            Self::CopyRun(run) => (run.id, run.open),
        };
        let state = usage
            .funding
            .get(&id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !Weak::ptr_eq(&state.execution, &Arc::downgrade(&execution.0)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let retained = match self {
            Self::HostScope(_) => state.scopes != 0,
            Self::NativeScope(scope) => {
                scope.purpose == ScopePurpose::Native && state.native_scopes != 0
            }
            Self::CopyRun(_) => state.run_open,
        };
        if !active || !retained || state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }
}

impl WorkingMemoryPool {
    // The sole caller derives `bytes` from the actual sealed sampler source.
    // Create a normal funding account directly: no fabricated inference
    // geometry, request metadata, or second accounting ledger is involved.
    pub(super) fn open_sampler_copy_account(
        &self,
        source: FundingSource<'_>,
        execution: &InferenceExecutionIdentity,
        bytes: u64,
        capacity: u64,
    ) -> Result<(WorkingMemoryFundingRun, WorkingMemoryFundingScope), WorkingMemoryError> {
        if !self.same_domain(source.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut node = Some(AccountNode::empty());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        source.validate(&usage, execution)?;
        let id = commit_copy_account(
            self,
            &mut usage,
            &mut node,
            execution,
            bytes,
            capacity,
            CopyHostHolds::None,
        )?;
        drop(usage);
        Ok((
            WorkingMemoryFundingRun {
                pool: self.clone(),
                id,
                open: true,
                handoff_taken: false,
                borrowed_storage: None,
            },
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.clone(),
                id,
                active: true,
                borrowed_storage: None,
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }

    // Only the sealed workspace-copy route supplies this byte requirement and
    // its matching existing-only pin. Source validation and numeric admission
    // share one lock, including quarantine changes after cold preparation.
    pub(super) fn open_workspace_copy_account<K: Ord + Send + 'static>(
        &self,
        source: &WorkingMemoryStorage<K>,
        pin: RegisteredStoragePin,
        execution: &InferenceExecutionIdentity,
        bytes: u64,
        capacity: u64,
    ) -> Result<(WorkingMemoryFundingRun, WorkingMemoryFundingScope), WorkingMemoryError> {
        self.open_workspace_copy_validated(pin, execution, bytes, capacity, |usage| {
            source.validate_copy_source(self, usage)
        })
    }
    pub(super) fn open_completed_workspace_copy_account<K: Ord + Send + 'static>(
        &self,
        source: &super::workspace_copy::completed::CompletedSourceCustody,
        registered: Option<&WorkingMemoryStorage<K>>,
        pin: RegisteredStoragePin,
        execution: &InferenceExecutionIdentity,
        bytes: u64,
        capacity: u64,
    ) -> Result<(WorkingMemoryFundingRun, WorkingMemoryFundingScope), WorkingMemoryError> {
        self.open_workspace_copy_validated(pin, execution, bytes, capacity, |usage| {
            source.validate(self, usage)?;
            if let Some(registered) = registered {
                registered.validate_copy_source(self, usage)?;
            }
            Ok(())
        })
    }
    fn open_workspace_copy_validated(
        &self,
        pin: RegisteredStoragePin,
        execution: &InferenceExecutionIdentity,
        bytes: u64,
        capacity: u64,
        validate: impl FnOnce(&Usage) -> Result<(), WorkingMemoryError>,
    ) -> Result<(WorkingMemoryFundingRun, WorkingMemoryFundingScope), WorkingMemoryError> {
        let mut node = Some(AccountNode::empty());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        validate(&usage)?;
        let id = commit_copy_account(
            self,
            &mut usage,
            &mut node,
            execution,
            bytes,
            capacity,
            CopyHostHolds::None,
        )?;
        drop(usage);
        Ok((
            WorkingMemoryFundingRun {
                pool: self.clone(),
                id,
                open: true,
                handoff_taken: false,
                borrowed_storage: None,
            },
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.clone(),
                id,
                active: true,
                borrowed_storage: Some(pin),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }
    // The sealed aggregate derives host_bytes from its borrowed sampler plan.
    // Every supplied source proof is checked under the same lock as this one commit.
    pub(super) fn open_sampling_copy_account<K: Ord + Send + 'static>(
        &self,
        sampler: FundingSource<'_>,
        sampler_execution: &InferenceExecutionIdentity,
        arrays: &WorkingMemoryStorage<K>,
        complete_source: Option<&WorkingMemoryStorage<K>>,
        pin: RegisteredStoragePin,
        destination: &InferenceExecutionIdentity,
        bytes: u64,
        host_bytes: u64,
        capacity: u64,
    ) -> Result<
        (
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        if !self.same_domain(sampler.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut node = Some(AccountNode::empty());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        sampler.validate(&usage, sampler_execution)?;
        arrays.validate_copy_source(self, &usage)?;
        if let Some(source) = complete_source {
            source.validate_copy_source(self, &usage)?;
        }
        let id = commit_copy_account(
            self,
            &mut usage,
            &mut node,
            destination,
            bytes,
            capacity,
            CopyHostHolds::Sampler(host_bytes),
        )?;
        drop(usage);
        Ok((
            WorkingMemoryFundingRun {
                pool: self.clone(),
                id,
                open: true,
                handoff_taken: false,
                borrowed_storage: None,
            },
            WorkingMemorySamplerScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.clone(),
                    id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                }),
                // The matching ledger hold was installed atomically above.
                held: host_bytes,
            },
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.clone(),
                id,
                active: true,
                borrowed_storage: Some(pin),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }
}

// The concrete routes, not byte callers, select the exact initial custody.
// Zero-byte host payloads still have one independent host scope.
enum CopyHostHolds {
    None,
    // A closed fresh host destination has no native scope or public run.
    HostOnly(u64),
    Sampler(u64),
    Paired {
        sampler: u64,
        decoder: u64,
    },
    Grouped {
        sampler: u64,
        decoder: u64,
        tables: usize,
    },
}

impl CopyHostHolds {
    fn total_and_scopes(self) -> Result<(u64, usize), WorkingMemoryError> {
        match self {
            Self::Grouped {
                sampler,
                decoder,
                tables,
            } => Ok((
                sampler
                    .checked_add(decoder)
                    .ok_or(WorkingMemoryError::Overflow)?,
                tables.checked_add(2).ok_or(WorkingMemoryError::Overflow)?,
            )),
            Self::None => Ok((0, 1)),
            Self::HostOnly(bytes) => Ok((bytes, 1)),
            Self::Sampler(bytes) => Ok((bytes, 2)),
            Self::Paired { sampler, decoder } => Ok((
                sampler
                    .checked_add(decoder)
                    .ok_or(WorkingMemoryError::Overflow)?,
                3,
            )),
        }
    }
}

// Created only alongside a plan borrowing the actual table or funded slots.
// Neither a public metadata token nor a caller-provided ID creates this proof.
pub(super) enum DecoderCopySource<'a, K: Ord + Send + 'static> {
    Registered(&'a WorkingMemoryStorage<K>),
    Funded {
        scope: &'a WorkingMemoryDecoderHostScope,
        execution: &'a InferenceExecutionIdentity,
    },
}

impl<K: Ord + Send + 'static> DecoderCopySource<'_, K> {
    fn validate(&self, pool: &WorkingMemoryPool, usage: &Usage) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Registered(source) => source.validate_copy_source(pool, usage),
            Self::Funded { scope, execution } => scope.validate(pool, usage, execution),
        }
    }
}

impl WorkingMemoryPool {
    // All inputs come from the sealed sampler/slot/copy programs. The complete
    // source pin bundle is staged outside the lock and enters native custody only.
    pub(super) fn open_text_components_copy_account<K: Ord + Send + 'static>(
        &self,
        sampler: FundingSource<'_>,
        sampler_execution: &InferenceExecutionIdentity,
        decoder: DecoderCopySource<'_, K>,
        operands: &WorkingMemoryStorage<K>,
        complete_source: &WorkingMemoryStorage<K>,
        pin: RegisteredStoragePin,
        destination: &InferenceExecutionIdentity,
        bytes: u64,
        sampler_hold: u64,
        decoder_hold: u64,
        capacity: u64,
    ) -> Result<
        (
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            WorkingMemoryDecoderHostScope,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        if !self.same_domain(sampler.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut node = Some(AccountNode::empty());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        sampler.validate(&usage, sampler_execution)?;
        decoder.validate(self, &usage)?;
        operands.validate_copy_source(self, &usage)?;
        complete_source.validate_copy_source(self, &usage)?;
        let id = commit_copy_account(
            self,
            &mut usage,
            &mut node,
            destination,
            bytes,
            capacity,
            CopyHostHolds::Paired {
                sampler: sampler_hold,
                decoder: decoder_hold,
            },
        )?;
        drop(usage);
        Ok((
            WorkingMemoryFundingRun {
                pool: self.clone(),
                id,
                open: true,
                handoff_taken: false,
                borrowed_storage: None,
            },
            WorkingMemorySamplerScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.clone(),
                    id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                }),
                held: sampler_hold,
            },
            WorkingMemoryDecoderHostScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.clone(),
                    id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                }),
                held: decoder_hold,
            },
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.clone(),
                id,
                active: true,
                borrowed_storage: Some(pin),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }
}

// Concrete copy routes validate their own authenticated sources before
// reaching this private numeric commit. No public scalar admission is added.
fn commit_copy_account(
    pool: &WorkingMemoryPool,
    usage: &mut Usage,
    node: &mut Option<Box<AccountNode>>,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    capacity: u64,
    host_holds: CopyHostHolds,
) -> Result<u64, WorkingMemoryError> {
    let native_scopes = usize::from(!matches!(host_holds, CopyHostHolds::HostOnly(_)));
    let (held, scopes) = host_holds.total_and_scopes()?;
    if held > bytes {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    let commit =
        PreparedAccountCommit::prepare(pool, execution, usage, bytes, Some(capacity), &[])?;
    let id = usage.next_funding;
    let next = id.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
    let mut state = FundingState::new(
        bytes,
        Some(capacity),
        execution,
        false,
        scopes,
        native_scopes,
    );
    state.host_held = held;
    // All source, capacity and counter checks precede this one commit.
    commit.commit(usage);
    usage
        .funding
        .publish(node.take().expect("prepared copy node"), id, state, true);
    usage.next_funding = next;
    Ok(id)
}

/// Move-only authority for a converted reservation's future native work.
/// Drop closes future work and releases only unassigned funding, after all work
/// scopes settle. Retain this owner until quoted host/controller/sampler payloads
/// have retired or received independent, complete storage accounting.
#[derive(Debug)]
#[must_use = "retain the run through all quoted payloads and future native work"]
pub struct WorkingMemoryFundingRun {
    pool: WorkingMemoryPool,
    id: u64,
    open: bool,
    handoff_taken: bool,
    // Drop after the run's locked accounting transition, never under that lock.
    borrowed_storage: Option<RegisteredStoragePin>,
}

impl WorkingMemoryFundingRun {
    // Only the consumed prompt stage reaches this helper. All sources and the
    // exact destination reservation are checked with the P hold and two scopes
    // under one usage lock; no new account or general allocation API is created.
    pub(super) fn open_dense_prompt_scopes<K: Ord + Send + 'static>(
        &self,
        reservation: &WorkingMemoryReservation,
        source: DecoderCopySource<'_, K>,
        complete_source: &WorkingMemoryStorage<K>,
        pins: RegisteredStoragePin,
        bytes: u64,
    ) -> Result<
        (
            InferenceExecutionIdentity,
            WorkingMemoryDecoderHostScope,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Retain any original residual exclusion as well as the actual decoder
        // inventory. The entire bundle belongs to native scope/quarantine.
        let pins = if complete_source.source_preparation().is_some() {
            // The source constructor H prices this fixed pair. Its children
            // already exist; no variable Vec or additional single-pin wrapper.
            match &self.borrowed_storage {
                Some(borrowed) => RegisteredStoragePin::pair(pins, borrowed.clone()),
                None => pins,
            }
        } else {
            RegisteredStoragePin::aggregate(
                [Some(pins), self.borrowed_storage.clone()]
                    .into_iter()
                    .flatten(),
            )
        };
        let execution = reservation.0.execution.clone();
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &execution)?;
        source.validate(&self.pool, &usage)?;
        complete_source.validate_copy_source(&self.pool, &usage)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let (scopes, native_scopes) = state.scope_counts_after(2, 1)?;
        state.host_held = held;
        state.scopes = scopes;
        state.native_scopes = native_scopes;
        drop(usage);
        Ok((
            execution,
            WorkingMemoryDecoderHostScope {
                scope: Some(WorkingMemoryFundingScope {
                    purpose: ScopePurpose::Host,
                    pool: self.pool.clone(),
                    id: self.id,
                    active: true,
                    borrowed_storage: None,
                    capture_source: None,
                    native_publication_identity: None,
                }),
                held: bytes,
            },
            WorkingMemoryFundingScope {
                purpose: ScopePurpose::Native,
                pool: self.pool.clone(),
                id: self.id,
                active: true,
                borrowed_storage: Some(pins),
                capture_source: None,
                native_publication_identity: None,
            },
        ))
    }

    // Closed pending-token construction derives its own P; callers cannot
    // choose a byte allowance or obtain this scope for native work.
    pub(super) fn open_pending_input_scope(
        &self,
        reservation: &WorkingMemoryReservation,
        bytes: u64,
    ) -> Result<WorkingMemoryDecoderHostScope, WorkingMemoryError> {
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &reservation.0.execution)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let scopes = state
            .scopes
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        state.host_held = held;
        state.scopes = scopes;
        drop(usage);
        Ok(WorkingMemoryDecoderHostScope {
            scope: Some(WorkingMemoryFundingScope {
                purpose: ScopePurpose::Host,
                pool: self.pool.clone(),
                id: self.id,
                active: true,
                borrowed_storage: None,
                capture_source: None,
                native_publication_identity: None,
            }),
            held: bytes,
        })
    }

    /// Creates custody exclusively for the original host sampler. This type
    /// exposes no native work, adoption or certification methods. Its internal
    /// scope is distinct from every PRNG/submission/recovery scope.
    pub fn sampler_scope(&self) -> Result<WorkingMemorySamplerScope, WorkingMemoryError> {
        Ok(WorkingMemorySamplerScope {
            scope: Some(self.scope_with_purpose(ScopePurpose::Host)?),
            held: 0,
        })
    }

    /// Creates an owned scope before native work starts. A scope has no implicit
    /// successful completion: dropping it uncertified quarantines the envelope.
    /// An active stamped span excludes new scopes on this same account until
    /// its canonical source-parcel transition; independent accounts are separate.
    pub fn scope(&self) -> Result<WorkingMemoryFundingScope, WorkingMemoryError> {
        self.scope_with_purpose(ScopePurpose::Native)
    }

    fn scope_with_purpose(
        &self,
        purpose: ScopePurpose,
    ) -> Result<WorkingMemoryFundingScope, WorkingMemoryError> {
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let state = usage.funding.get_mut(&self.id).expect("live funding run");
        if !state.run_open || state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        let (scopes, native_scopes) =
            state.scope_counts_after(1, usize::from(purpose == ScopePurpose::Native))?;
        state.scopes = scopes;
        state.native_scopes = native_scopes;
        Ok(WorkingMemoryFundingScope {
            purpose,
            pool: self.pool.clone(),
            id: self.id,
            active: true,
            borrowed_storage: self.borrowed_storage.clone(),
            capture_source: None,
            native_publication_identity: None,
        })
    }

    /// Rechecks this live run against its exact original reservation account.
    ///
    /// Reservation clones share the same account. Equal geometry, execution or
    /// pool capacity cannot substitute a different reservation, including one
    /// from another account in this same pool. The original metadata must remain
    /// live and the run must be open and healthy at this point in time.
    ///
    /// This reads existing accounting only: no scope, hold, pin, allocation or
    /// grant is created. It neither certifies native work nor checks independent
    /// source origins; callers retain and validate those witnesses separately.
    pub fn validate_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &reservation.0.execution)?;
        let state = usage
            .funding
            .get(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }

    /// The exact managed domain covered by this run.
    pub fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }

    /// Closes future scopes. Outstanding scopes and retained allocations retain
    /// their coverage; metadata and descendants retain the account's ceiling and
    /// unquoted-work exclusion even after unused workspace is released. The
    /// ceiling changes only through explicitly delegated successor policy.
    pub fn close(mut self) -> Result<(), WorkingMemoryError> {
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        usage
            .funding
            .get_mut(&self.id)
            .expect("live funding run")
            .run_open = false;
        self.open = false;
        settle(&mut usage, self.id);
        drop(usage);
        accounts::drain(&self.pool);
        Ok(())
    }
}

/// Closed host-only custody for one sampler construction. It cannot be created
/// from an existing native work scope or used to certify native completion.
/// Dropping unused custody settles only this independently minted empty scope.
#[derive(Debug)]
pub struct WorkingMemorySamplerScope {
    scope: Option<WorkingMemoryFundingScope>,
    held: u64,
}

impl WorkingMemorySamplerScope {
    pub(super) fn source(&self) -> FundingSource<'_> {
        FundingSource::HostScope(self.scope.as_ref().expect("live sampler custody"))
    }

    pub(super) fn validate_source(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.scope
            .as_ref()
            .expect("live sampler custody")
            .validate_sampler_source(execution)
    }

    pub(super) fn validate_stage(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<InferenceExecutionIdentity, WorkingMemoryError> {
        self.scope
            .as_ref()
            .expect("live sampler custody")
            .validate_sampler_stage(reservation)
    }

    // Only the consuming canonical stage constructor can claim this hold.
    pub(super) fn hold_sampler_payload(&mut self, bytes: u64) -> Result<(), WorkingMemoryError> {
        self.hold_sampler_payload_checked(bytes, None)
    }

    // The resumed constructor validates its actual funded source and installs
    // the new host hold at one accounting boundary, before copying history.
    pub(super) fn hold_resumed_sampler_payload(
        &mut self,
        bytes: u64,
        source: FundingSource<'_>,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.hold_sampler_payload_checked(bytes, Some((source, execution)))
    }

    fn hold_sampler_payload_checked(
        &mut self,
        bytes: u64,
        source: Option<(FundingSource<'_>, &InferenceExecutionIdentity)>,
    ) -> Result<(), WorkingMemoryError> {
        if self.held != 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let scope = self.scope.as_ref().expect("live sampler custody");
        if source.is_some_and(|(source, _)| !scope.pool.same_domain(source.pool())) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut usage = scope
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if let Some((source, execution)) = source {
            source.validate(&usage, execution)?;
        }
        let state = usage
            .funding
            .get_mut(&scope.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !scope.active || state.scopes == 0 || state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        state.host_held = held;
        self.held = bytes;
        Ok(())
    }
}

impl Drop for WorkingMemorySamplerScope {
    fn drop(&mut self) {
        if let Some(scope) = self.scope.take() {
            {
                let mut usage = lock_for_retirement(&scope.pool);
                let state = usage
                    .funding
                    .get_mut(&scope.id)
                    .expect("live sampler custody");
                state.host_held -= self.held;
            }
            // This scope was never exposed for native work. A bound owner's
            // host payload must retire before this final custody field.
            let _ = scope.certify();
        }
    }
}

/// Closed host-only custody minted by paired/dense or pending-input construction.
/// It exposes no native work, general scope conversion, or independent factory.
#[derive(Debug)]
pub(super) struct WorkingMemoryDecoderHostScope {
    scope: Option<WorkingMemoryFundingScope>,
    held: u64,
}

impl WorkingMemoryDecoderHostScope {
    pub(super) fn pool(&self) -> &WorkingMemoryPool {
        &self.scope.as_ref().expect("live decoder host custody").pool
    }

    // The split pending-input worker rechecks this immediately before filling.
    // An already closed run may retain payloads but cannot construct new ones.
    pub(super) fn validate_pending_input_allocation(
        &self,
        reservation: &WorkingMemoryReservation,
        expected: u64,
    ) -> Result<(), WorkingMemoryError> {
        let scope = self
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if reservation.0.funding != Some(scope.id)
            || !scope.pool.same_domain(&reservation.0.pool)
            || self.held != expected
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = scope
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate(&scope.pool, &usage, &reservation.0.execution)?;
        let state = usage
            .funding
            .get(&scope.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }

    // The closed table transfer holds this exact pool's usage lock and uses
    // the returned identity/held amount only for its checked atomic commit.
    pub(super) fn validate_host_transfer(
        &self,
        usage: &Usage,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(u64, u64), WorkingMemoryError> {
        self.validate(self.pool(), usage, execution)?;
        Ok((
            self.scope.as_ref().expect("validated host scope").id,
            self.held,
        ))
    }

    // Called only after the closed transfer checks bytes <= held and stages
    // every fallible action, while committing the matching ledger decrement.
    pub(super) fn transfer_host_hold(&mut self, bytes: u64) {
        self.held -= bytes;
    }

    fn validate(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        let scope = self
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !pool.same_domain(&scope.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        FundingSource::HostScope(scope).validate(usage, execution)?;
        let state = usage
            .funding
            .get(&scope.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if self.held > state.host_held || state.protected_remaining()? > state.remaining {
            return Err(WorkingMemoryError::Poisoned);
        }
        Ok(())
    }
}

impl Drop for WorkingMemoryDecoderHostScope {
    fn drop(&mut self) {
        if let Some(scope) = self.scope.take() {
            {
                let mut usage = lock_for_retirement(&scope.pool);
                let state = usage
                    .funding
                    .get_mut(&scope.id)
                    .expect("live decoder host custody");
                state.host_held -= self.held;
            }
            // The fixed slot payload retires before this final custody field.
            // No native operation can have used this internally minted scope.
            let _ = scope.certify();
        }
    }
}

impl Drop for WorkingMemoryFundingRun {
    fn drop(&mut self) {
        if self.open {
            let mut usage = lock_for_retirement(&self.pool);
            usage
                .funding
                .get_mut(&self.id)
                .expect("live funding run")
                .run_open = false;
            settle(&mut usage, self.id);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopePurpose {
    Native,
    Host,
}

/// Owned funding retained through native submission, completion and publication.
/// An error, unwind or ordinary Drop does not prove completion: it permanently
/// retains the envelope's unassigned balance. Exact allocation retirement returns
/// credit to that balance, including handles dropped after partial publication.
#[derive(Debug)]
#[must_use = "certify only after settlement and complete retained-storage publication"]
pub struct WorkingMemoryFundingScope {
    purpose: ScopePurpose,
    pool: WorkingMemoryPool,
    pub(super) id: u64,
    active: bool,
    // Independent lifetime for excluded registered storage. Every uncertified
    // scope transfers its entire bundle into permanent account custody.
    borrowed_storage: Option<RegisteredStoragePin>,
    // Independent, lazily installed capture-only channel. An abandoned handle
    // never removes this slot; uncertified Drop quarantines both channels.
    // Absent ordinary scopes hold one pointer; channel storage is priced in H.
    capture_source: Option<Box<CaptureSourceSlot>>,
    // Allocated only by the selected native publisher. No scope/budget/source
    // backedge; lets a retained attempt reject a different actual work scope.
    pub(in crate::working_memory) native_publication_identity:
        Option<native_partition::NativePublicationScopeIdentity>,
}

impl WorkingMemoryFundingScope {
    pub(in crate::working_memory) fn validate_native_purpose(
        &self,
    ) -> Result<(), WorkingMemoryError> {
        if self.active && self.purpose == ScopePurpose::Native {
            Ok(())
        } else {
            Err(WorkingMemoryError::ExecutionFenced)
        }
    }

    pub(super) fn validate_sampler_source(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::HostScope(self).validate(&usage, execution)
    }

    pub(super) fn validate_sampler_stage(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<InferenceExecutionIdentity, WorkingMemoryError> {
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::HostScope(self).validate(&usage, &reservation.0.execution)?;
        Ok(reservation.0.execution.clone())
    }

    /// The exact managed domain covered by this scope.
    pub fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }

    /// Checks domain identity without submitting work or changing accounting.
    pub fn validate_domain(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.pool.same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Publishes exact backing identities from this scope's reserved envelope.
    /// This has the same physical-lifetime contract as pool storage registration.
    /// Returned map keys that pin storage must retire before their handles.
    pub fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<BTreeMap<K, WorkingMemoryStorage<K>>, WorkingMemoryError> {
        self.pool.adopt_storage_individually(self, storage)
    }

    /// Certifies that native work has settled and every surviving managed
    /// allocation has complete independent storage accounting. This is an
    /// explicit provider assertion, not a logical-step or polling-success signal.
    /// Leave the scope uncertified on incomplete publication or unknown escape.
    /// An active stamped span must first consume its matching canonical ticket
    /// through source-parcel handoff. Certification cannot clear its exclusion.
    pub fn certify(mut self) -> Result<(), WorkingMemoryError> {
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let state = usage.funding.get_mut(&self.id).expect("live funding scope");
        if state
            .active_span
            .as_ref()
            .is_some_and(|span| span.matches(Some(&self)))
        {
            // Error Drop retains/quarantines this exact channel. Certification
            // cannot replace the canonical ticket/parcel transition.
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.close_scope(self.purpose);
        self.active = false;
        settle(&mut usage, self.id);
        drop(usage);
        accounts::drain(&self.pool);
        Ok(())
    }
}

impl Drop for WorkingMemoryFundingScope {
    fn drop(&mut self) {
        if self.active {
            // Allocation and ownership preparation precede the lock, including
            // poisoned-account cleanup. Successful certification takes no node.
            let capture = self.capture_source.take().map(|slot| (*slot).into_pin());
            let pins = match (self.borrowed_storage.take(), capture) {
                (None, None) => None,
                (Some(pin), None) | (None, Some(pin)) => Some(pin),
                (Some(a), Some(b)) => Some(RegisteredStoragePin::aggregate([a, b])),
            };
            let pins = pins.map(|pin| {
                Box::new(QuarantinedStoragePins {
                    _pin: pin,
                    next: None,
                })
            });
            let mut usage = lock_for_retirement(&self.pool);
            let state = usage.funding.get_mut(&self.id).expect("live funding scope");
            state.quarantined = true;
            if let Some(mut pins) = pins {
                pins.next = state.quarantined_borrowed.take();
                state.quarantined_borrowed = Some(pins);
            }
            state.close_scope(self.purpose);
            settle(&mut usage, self.id);
        }
    }
}

pub(super) fn retire_metadata(usage: &mut Usage, id: u64) {
    usage
        .funding
        .get_mut(&id)
        .expect("live funding metadata")
        .metadata_live = false;
    settle(usage, id);
}

pub(super) fn retire_allocation(usage: &mut Usage, id: u64, bytes: u64) {
    let state = usage.funding.get_mut(&id).expect("live funded allocation");
    state.allocations -= 1;
    usage.registered -= bytes;
    if state.retains_workspace() {
        state.remaining += bytes;
        usage.reserved += bytes;
    }
    settle(usage, id);
}

pub(super) fn retire_registration(usage: &mut Usage, id: u64) {
    usage
        .funding
        .get_mut(&id)
        .expect("live funded registration")
        .registrations -= 1;
    settle(usage, id);
}

// Poison is global to Usage. Cleanup may destroy real payloads, but cannot
// derive reusable transient headroom from that possibly inconsistent ledger.
// Mark existing accounts before any host decrement, retirement or settlement;
// this neither creates custody nor clears the mutex poison.
pub(super) fn lock_for_retirement(pool: &WorkingMemoryPool) -> accounts::RetirementGuard<'_> {
    accounts::lock(pool)
}

fn settle(usage: &mut Usage, id: u64) {
    let state = usage.funding.get_mut(&id).expect("live funding account");
    if !state.retains_workspace() {
        // Host and native partitions retain their exact protected holds.
        // Recognized native rows never add a second registered byte charge.
        if let Some(reserved) = state
            .spendable_remaining()
            .ok()
            .and_then(|released| usage.reserved.checked_sub(released))
        {
            usage.reserved = reserved;
            state.remaining = state
                .protected_remaining()
                .expect("validated protected hold");
        } else {
            // Never fabricate credit after an accounting invariant failure.
            state.quarantined = true;
        }
    }
    if !state.metadata_live
        && !state.retains_workspace()
        && state.scopes == 0
        && state.native_held.is_none()
        && state.remaining == state.control_floor
        && state.host_held == state.control_floor
        && state.allocations == 0
        && state.registrations == 0
    {
        usage.funding.mark_terminal(id);
    }
}

mod decoder_group;

// Named private representation used by original admission, never a byte grant.
pub(super) fn span_workspace_raw_payload_bytes() -> usize {
    std::mem::size_of::<span_workspace::RawSpanHostCustody>()
}

mod generation_copy;
pub(in crate::working_memory) use generation_copy::{
    GenerationCopyAdmission, GenerationCopyCustody, GenerationCopySource,
};

mod copy_controls;
pub(in crate::working_memory) use copy_controls::copy_account_control_bytes;
