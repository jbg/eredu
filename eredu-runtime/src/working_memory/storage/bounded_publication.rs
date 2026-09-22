//! Fixed new-key publication through one original, already established C namespace.
use super::*;
use crate::working_memory::{
    CaptureSourceSegment, InferenceSpanWorkspacePlan, InferenceWorkspaceSpan,
    OriginalTextControlGuard,
    funding::{CapturePinIdentity, RawSpanHostOwner},
};
use crate::{inspection::PrefillChunkRetentionContext, prefill::PrefillChunk};
use std::{marker::PhantomData, mem::size_of, sync::TryLockError};

#[derive(Debug)]
struct PublicationRow {
    chunk: PrefillChunk,
    slots: usize,
}
#[derive(Debug)]
pub(in crate::working_memory) struct BatchPublicationLayout {
    key: TypeId,
    rows: Box<[PublicationRow]>,
    bytes: u64,
    pub(in crate::working_memory) original: Arc<capture_publication::PublicationLayout>,
}
impl BatchPublicationLayout {
    pub(in crate::working_memory) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(in crate::working_memory) fn matches<K: 'static>(&self) -> bool {
        self.key == TypeId::of::<K>()
    }
}
/// Cold finite slot ceilings for the actual span schedule. These are provider
/// facts, never a byte grant. Bind after the exact same-key C publication plan
/// and before accepting the original quote. Provider key payloads are separate.
#[derive(Debug)]
pub struct PreparedPrefillStoragePublicationPlan<K: CapturePlanStorageKey> {
    plan: InferenceSpanWorkspacePlan,
    rows: Box<[PublicationRow]>,
    bytes: u64,
    marker: PhantomData<fn() -> K>,
}
impl<K: CapturePlanStorageKey> PreparedPrefillStoragePublicationPlan<K> {
    /// Unknown ceilings and checked arithmetic overflow reject cold preparation.
    pub fn prepare(
        plan: &InferenceSpanWorkspacePlan,
        mut slots: impl FnMut(&PrefillChunk) -> Option<usize>,
    ) -> Result<Self, WorkingMemoryError> {
        let count = plan
            .records()
            .iter()
            .filter(|r| matches!(r.span(), InferenceWorkspaceSpan::Prefill(_)))
            .count();
        let mut rows = Vec::with_capacity(count);
        let mut bytes = 0usize;
        for record in plan.records() {
            if let InferenceWorkspaceSpan::Prefill(chunk) = record.span() {
                let slots = slots(chunk).ok_or(WorkingMemoryError::UnknownBound)?;
                bytes = bytes
                    .checked_add(row_peak::<K>(slots)?)
                    .ok_or(WorkingMemoryError::Overflow)?;
                rows.push(PublicationRow {
                    chunk: chunk.clone(),
                    slots,
                });
            }
        }
        bytes = bytes
            .checked_add(
                count
                    .checked_mul(2 * size_of::<PublicationRow>())
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .and_then(|b| {
                b.checked_add(size_of::<BatchPublicationLayout>() + 2 * size_of::<usize>())
            })
            .and_then(|b| {
                b.checked_add(
                    3 * (size_of::<Self>()
                        + size_of::<OriginalPrefillStoragePublicationSlots<K>>()),
                )
            })
            // The canonical namespace wrapper gains one link. Its ordinary map
            // metadata remains the pre-existing registry accounting exclusion.
            .and_then(|b| b.checked_add(size_of::<Option<Box<RegistryBatch<K>>>>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            plan: plan.clone(),
            rows: rows.into_boxed_slice(),
            bytes: u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)?,
            marker: PhantomData,
        })
    }
    /// Cumulative S: every issued row's output/error may escape and coexist.
    pub fn control_peak_bytes(&self) -> u64 {
        self.bytes
    }
    pub(in crate::working_memory) fn into_layout(
        self,
        plan: &InferenceSpanWorkspacePlan,
        original: &Arc<capture_publication::PublicationLayout>,
    ) -> Result<Arc<BatchPublicationLayout>, WorkingMemoryError> {
        if !self.plan.same_plan(plan) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        original.key::<K>()?;
        Ok(Arc::new(BatchPublicationLayout {
            key: TypeId::of::<K>(),
            rows: self.rows,
            bytes: self.bytes,
            original: original.clone(),
        }))
    }
}
fn row_peak<K: CapturePlanStorageKey>(slots: usize) -> Result<usize, WorkingMemoryError> {
    // Actual fixed vectors, per-unique provider key Arc, singleton registration
    // Vec+Arc, output handle, and fixed node slots. Vec->Box peak is included.
    let per = size_of::<(K, u64, Arc<eredu_core::MemoryPlacement>)>()
        .checked_add(size_of::<StagedAllocation<K>>())
        .and_then(|n| n.checked_add(2 * size_of::<Option<(RegistryKey<K>, Entry)>>()))
        .and_then(|n| n.checked_add(2 * size_of::<K>() + 2 * size_of::<usize>()))
        .and_then(|n| n.checked_add(size_of::<Registration<K>>() + 2 * size_of::<usize>()))
        .and_then(|n| {
            n.checked_add(
                3 * (size_of::<BoundedPublishedAllocation<K>>()
                    + size_of::<Registration<K>>()
                    + size_of::<StagedAllocation<K>>()
                    + size_of::<K>()),
            )
        })
        .ok_or(WorkingMemoryError::Overflow)?;
    per.checked_mul(slots)
        .and_then(|n| {
            n.checked_add(
                3 * (size_of::<BoundedPublicationAttempt<K>>()
                    + size_of::<RegistryBatch<K>>()
                    + size_of::<registry::RetiredEntry<K>>()),
            )
        })
        .and_then(|n| n.checked_add(CapturePinIdentity::retained_control_bytes()))
        .ok_or(WorkingMemoryError::Overflow)
}
/// The one finite bank extracted from the actual promoted original owner.
#[derive(Debug)]
#[must_use]
pub struct OriginalPrefillStoragePublicationSlots<K: CapturePlanStorageKey> {
    layout: Arc<BatchPublicationLayout>,
    next: usize,
    controls: OriginalTextControlGuard,
    marker: PhantomData<fn() -> K>,
}
impl<K: CapturePlanStorageKey> OriginalPrefillStoragePublicationSlots<K> {
    pub(in crate::working_memory) fn new(
        layout: Arc<BatchPublicationLayout>,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self {
            layout,
            next: 0,
            controls,
            marker: PhantomData,
        }
    }
    /// Includes abandoned, failed and panicking issued attempts.
    pub fn spent_rows(&self) -> usize {
        self.next
    }
    /// The actual canonical chunk, account, original C witness and stamped slot
    /// must match under one Usage loan. Busy before issuance spends no row;
    /// issuance consumes it before any fixed scratch construction.
    pub fn begin(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<BoundedPublicationAttempt<K>, BoundedPublicationError> {
        let row = self
            .layout
            .rows
            .get(self.next)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if row.chunk.input != context.chunk().input
            || row.chunk.position != context.chunk().position
            || row.chunk.output != context.chunk().output
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let identity = segment.pin_identity();
        {
            let usage = native.pool().0.usage.try_lock().map_err(lock_error)?;
            self.controls
                .custody
                .validate_batch_publication_locked::<K>(
                    native.pool(),
                    &usage,
                    native,
                    &self.layout,
                    context.request(),
                )?;
            segment.validate_pin_locked(
                native,
                &usage,
                &identity,
                Some(context),
                self.controls.custody.pin_source()?,
            )?;
        }
        let slots = row.slots;
        self.next += 1;
        let controls = self.controls.clone();
        Ok(BoundedPublicationAttempt {
            inputs: Vec::with_capacity(slots),
            staged: Vec::with_capacity(slots),
            node: Some(RegistryBatch::prepare(
                slots,
                controls.custody.raw().clone(),
            )),
            identity,
            layout: self.layout.clone(),
            slots,
            terminal: false,
            published: false,
            controls,
        })
    }
}
/// Nonblocking registry contention or the original typed accounting failure.
#[derive(Debug, thiserror::Error)]
pub enum BoundedPublicationError {
    #[error("bounded publication is busy")]
    /// No accounting mutation occurred. The issued row remains terminal.
    Busy,
    #[error("bounded publication rejected: {0}")]
    /// Original identity, origin, capacity, overflow or budget cause.
    Storage(#[from] WorkingMemoryError),
}
fn lock_error<T>(error: TryLockError<T>) -> BoundedPublicationError {
    match error {
        TryLockError::WouldBlock => BoundedPublicationError::Busy,
        TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
    }
}
struct StagedAllocation<K: CapturePlanStorageKey> {
    first_input: usize,
    key: Option<Arc<K>>,
    bytes: u64,
    placement: Arc<eredu_core::MemoryPlacement>,
    funding_allowance_bytes: u64,
    locator: Option<EntryLocator>,
    registry_key: Option<RegistryKey<K>>,
    output: Option<BoundedPublishedAllocation<K>>,
    activation_pool: Option<MemoryLedger>,
}
/// Caller-owned terminal attempt: publish borrows it, so typed failure and
/// provider unwind leave all original/staged keys and original custody here.
/// There is no retry, replacement row, raw registration or byte-based grant.
#[must_use]
pub struct BoundedPublicationAttempt<K: CapturePlanStorageKey> {
    inputs: Vec<(K, u64, Arc<eredu_core::MemoryPlacement>)>,
    staged: Vec<StagedAllocation<K>>,
    node: Option<Box<RegistryBatch<K>>>,
    identity: CapturePinIdentity,
    layout: Arc<BatchPublicationLayout>,
    slots: usize,
    terminal: bool,
    published: bool,
    controls: OriginalTextControlGuard,
}
impl<K: CapturePlanStorageKey> std::fmt::Debug for BoundedPublicationAttempt<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedPublicationAttempt")
            .field("inputs", &self.inputs.len())
            .field("published", &self.published)
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}
impl<K: CapturePlanStorageKey> BoundedPublicationAttempt<K> {
    /// Move a descriptor into a preallocated slot, returning rejected ownership.
    pub fn push_owned(
        &mut self,
        key: K,
        bytes: u64,
        placement: Arc<eredu_core::MemoryPlacement>,
    ) -> Result<(), K> {
        if self.terminal || self.inputs.len() == self.slots {
            return Err(key);
        }
        self.inputs.push((key, bytes, placement));
        Ok(())
    }
    /// Number of retained inputs, including aliases and rejected-batch prefixes.
    pub fn retained_input_count(&self) -> usize {
        self.inputs.len()
    }
    /// Publish all unique descriptors atomically. Newly charged entries consume
    /// only this original account's unheld remainder; aliases retain their old
    /// origin. No key work, allocation or destruction follows the first
    /// mutation inside the Usage loan.
    pub fn publish(
        &mut self,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), BoundedPublicationError> {
        if self.terminal {
            return Err(WorkingMemoryError::PreparationAlreadyStarted.into());
        }
        self.terminal = true;
        // Keep every original input throughout provider comparison/clone panic.
        for (first_input, (key, bytes, placement)) in self.inputs.iter().enumerate() {
            placement
                .validate(native.pool().topology())
                .map_err(WorkingMemoryError::from)?;
            if let Some(prior) = self
                .staged
                .iter()
                .find(|s| s.key.as_ref().expect("staged key").as_ref().cmp(key) == Ordering::Equal)
            {
                same_capacity(prior.bytes, *bytes)?;
                if prior.placement != *placement {
                    return Err(WorkingMemoryError::IdentityMismatch.into());
                }
                continue;
            }
            let key = Arc::new(key.clone());
            let output = BoundedPublishedAllocation {
                storage: WorkingMemoryStorage::pending(vec![key.as_ref().clone()], *bytes),
                raw: self.controls.custody.raw().clone(),
            };
            self.staged.push(StagedAllocation {
                first_input,
                registry_key: Some(RegistryKey::Shared(key.clone())),
                key: Some(key),
                bytes: *bytes,
                placement: Arc::clone(placement),
                funding_allowance_bytes: 0,
                locator: None,
                output: Some(output),
                activation_pool: Some(native.pool().clone()),
            });
        }
        for staged in &mut self.staged {
            Arc::get_mut(&mut staged.output.as_mut().expect("private output").storage.0)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
        }
        let mut usage = native.pool().0.usage.try_lock().map_err(lock_error)?;
        self.controls
            .custody
            .validate_batch_publication_locked::<K>(
                native.pool(),
                &usage,
                native,
                &self.layout,
                self.identity.request()?,
            )?;
        segment.validate_pin_locked(
            native,
            &usage,
            &self.identity,
            None,
            self.controls.custody.pin_source()?,
        )?;
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|r| r.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let mut allocations = 0usize;
        for staged in &mut self.staged {
            if let Some((locator, entry)) =
                registry.locate(staged.key.as_ref().expect("staged key").as_ref())
            {
                same_capacity(entry.bytes, staged.bytes)?;
                if entry.placement != staged.placement {
                    return Err(WorkingMemoryError::IdentityMismatch.into());
                }
                validate_entry_origin(entry, &usage)?;
                entry
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                staged.locator = Some(locator);
            } else {
                allocations = allocations
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        for (i, staged) in self.staged.iter().enumerate() {
            if staged.locator.is_some()
                && self.staged[..i].iter().any(|s| s.locator == staged.locator)
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
        native.pool().0.check_host_increment(&usage, 0)?;
        let state = usage
            .funding
            .get(&native.id)
            .ok_or(WorkingMemoryError::ExecutionFenced)?;
        state.validate_span_spend(Some(native))?;
        for (slot, (domain, _)) in native.pool().topology().domains().enumerate() {
            let incremental = self
                .staged
                .iter()
                .filter(|row| row.locator.is_none() && row.placement.domains().contains(&domain))
                .try_fold(0u64, |sum, row| {
                    sum.checked_add(row.bytes)
                        .ok_or(WorkingMemoryError::Overflow)
                })?;
            let available = if slot == native.pool().0.host_slot {
                state.spendable_remaining()?
            } else {
                state.domains[slot]
                    .remaining
                    .checked_sub(state.domains[slot].native_held.unwrap_or(0))
                    .ok_or(WorkingMemoryError::Poisoned)?
            };
            if incremental > available {
                return Err(WorkingMemoryError::DomainAllowanceExceeded {
                    domain,
                    required_bytes: incremental,
                    available_bytes: available,
                }
                .into());
            }
            state.domains[slot]
                .remaining
                .checked_sub(incremental)
                .ok_or(WorkingMemoryError::Poisoned)?;
            let placement_allowance = self
                .staged
                .iter()
                .filter(|row| {
                    row.locator.is_none()
                        && row.placement.domains().contains(&domain)
                        && matches!(
                            row.placement.kind(),
                            eredu_core::MemoryPlacementKind::Possible { .. }
                        )
                })
                .map(|row| row.bytes)
                .sum::<u64>();
            let fixed_bytes = incremental
                .checked_sub(placement_allowance)
                .ok_or(WorkingMemoryError::Overflow)?;
            let converted =
                state.allocation_allowance(slot, fixed_bytes, placement_allowance, 0)?;
            state.domains[slot]
                .remaining_charge
                .placement_allowance_bytes
                .checked_sub(placement_allowance)
                .and_then(|bytes| bytes.checked_sub(converted))
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            usage.domains[slot]
                .placement_allowances
                .checked_sub(converted)
                .ok_or(WorkingMemoryError::Poisoned)?;
            let mut remaining_conversion = converted;
            for row in self.staged.iter_mut().filter(|row| {
                row.locator.is_none()
                    && row.placement.domains().contains(&domain)
                    && matches!(
                        row.placement.kind(),
                        eredu_core::MemoryPlacementKind::Fixed(_)
                    )
            }) {
                row.funding_allowance_bytes = row.bytes.min(remaining_conversion);
                remaining_conversion -= row.funding_allowance_bytes;
            }
            usage.domains[slot]
                .reserved
                .checked_sub(incremental)
                .ok_or(WorkingMemoryError::Poisoned)?;
            usage.domains[slot]
                .registered
                .checked_add(incremental)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        let count = state
            .allocations
            .checked_add(allocations)
            .ok_or(WorkingMemoryError::Overflow)?;
        let registrations = state
            .registrations
            .checked_add(self.staged.len())
            .ok_or(WorkingMemoryError::Overflow)?;
        let node = self
            .node
            .as_mut()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?;
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .and_then(|r| r.downcast_mut::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        // Commit frontier. Every locator, counter and private owner was checked.
        // New entries move into empty slots; old entries only increment owners.
        for (i, staged) in self.staged.iter_mut().enumerate() {
            if let Some(locator) = staged.locator {
                registry.at_mut(locator).owners += 1;
            } else {
                node.entries[i] = Some((
                    staged.registry_key.take().expect("prepared key"),
                    Entry {
                        reset_layout_id: None,
                        funding_allowance_bytes: staged.funding_allowance_bytes,
                        native_retired: false,
                        pending_allocation: false,
                        prepaid: None,
                        bytes: staged.bytes,
                        placement: Arc::clone(&staged.placement),
                        owners: 1,
                        funding: Some(native.id),
                    },
                ));
            }
        }
        if allocations != 0 {
            registry.link(self.node.take().expect("prepared node"));
        }
        let state = usage
            .funding
            .get_mut(&native.id)
            .expect("validated account");
        state.allocations = count;
        state.registrations = registrations;
        for (slot, (domain, _)) in native.pool().topology().domains().enumerate() {
            let incremental = self
                .staged
                .iter()
                .filter(|row| row.locator.is_none() && row.placement.domains().contains(&domain))
                .map(|row| row.bytes)
                .sum::<u64>();
            let placement_allowance = self
                .staged
                .iter()
                .filter(|row| {
                    row.locator.is_none()
                        && row.placement.domains().contains(&domain)
                        && matches!(
                            row.placement.kind(),
                            eredu_core::MemoryPlacementKind::Possible { .. }
                        )
                })
                .map(|row| row.bytes)
                .sum::<u64>();
            let converted = self
                .staged
                .iter()
                .filter(|row| row.locator.is_none() && row.placement.domains().contains(&domain))
                .map(|row| row.funding_allowance_bytes)
                .sum::<u64>();
            let balance = &mut usage
                .funding
                .get_mut(&native.id)
                .expect("validated account")
                .domains[slot];
            balance.remaining -= incremental;
            balance.remaining_charge.placement_allowance_bytes -= placement_allowance + converted;
            usage.domains[slot].placement_allowances -= converted;
            usage.domains[slot].reserved -= incremental;
            usage.domains[slot].registered += incremental;
        }
        for staged in &mut self.staged {
            let registration =
                Arc::get_mut(&mut staged.output.as_mut().expect("private output").storage.0)
                    .expect("validated private registration");
            registration.pool = staged.activation_pool.take();
            registration.funding = Some(native.id);
        }
        drop(usage);
        // Provider keys may themselves own payloads. Retire all duplicate and
        // staging aliases while the private output registrations still cover
        // them, before any independent output can escape. A destructor panic
        // leaves the attempt terminal and its registrations retained.
        self.inputs.clear();
        for staged in &mut self.staged {
            drop(staged.registry_key.take());
            drop(staged.key.take());
        }
        self.published = true;
        Ok(())
    }
    /// Move one unique allocation's opaque owner after whole-batch success.
    /// Slots correspond to first occurrence order; duplicates have no extra row.
    pub fn take_allocation(&mut self, index: usize) -> Option<BoundedPublishedAllocation<K>> {
        if !self.published {
            return None;
        }
        self.staged.get_mut(index)?.output.take()
    }
    /// First original input ordinal for one successful canonical allocation.
    /// Available only after the entire batch committed. This lends no key,
    /// registration or mutable destination and performs no new comparison.
    pub fn published_input_index(&self, unique: usize) -> Option<usize> {
        self.published.then_some(())?;
        Some(self.staged.get(unique)?.first_input)
    }

    /// Unique successful allocation count, including aliases and zero-byte keys.
    pub fn published_count(&self) -> usize {
        if self.published { self.staged.len() } else { 0 }
    }
}
/// One canonical allocation registration plus its actual original host custody.
/// Clones share registration; ordinary aliases also keep the registry node's
/// raw custody until its last entry. No source/full-witness back-edge is stored.
#[must_use]
pub struct BoundedPublishedAllocation<K: CapturePlanStorageKey> {
    storage: WorkingMemoryStorage<K>,
    raw: RawSpanHostOwner,
}
impl<K: CapturePlanStorageKey> Clone for BoundedPublishedAllocation<K> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            raw: self.raw.clone(),
        }
    }
}
impl<K: CapturePlanStorageKey> std::fmt::Debug for BoundedPublishedAllocation<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedPublishedAllocation")
            .field("bytes", &self.storage.bytes())
            .finish_non_exhaustive()
    }
}
impl<K: CapturePlanStorageKey> BoundedPublishedAllocation<K> {
    /// Full physical capacity of this unique key, not a new execution allowance.
    pub fn bytes(&self) -> Option<u64> {
        self.storage.bytes()
    }
    /// Same-pool canonical origin and original raw-custody health, under one lock.
    pub fn validate_source(&self, pool: &MemoryLedger) -> Result<(), BoundedPublicationError> {
        let usage = pool.0.usage.try_lock().map_err(lock_error)?;
        self.raw.validate_origin_locked(pool, &usage)?;
        self.storage.validate_copy_source(pool, &usage)?;
        Ok(())
    }
}

pub(in crate::working_memory) fn validate_original_namespace<K: CapturePlanStorageKey>(
    _pool: &MemoryLedger,
    usage: &crate::working_memory::Usage,
    original: &capture_publication::PublicationLayout,
) -> Result<(), WorkingMemoryError> {
    let key = original.key::<K>()?;
    if key.capture_plan_identity() != Some(&original.source) {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    let registry = usage
        .storage
        .get(&TypeId::of::<K>())
        .and_then(|r| r.downcast_ref::<Registry<K>>())
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    let entry = registry
        .get(key)
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    same_capacity(original.capacity, entry.bytes)?;
    validate_entry_origin(entry, usage)
}
