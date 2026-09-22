//! Existing-only grouped pins with original, finite control custody.
use super::*;
use crate::working_memory::{
    CaptureSourceSegment, InferenceSpanWorkspacePlan, InferenceWorkspaceSpan,
    OriginalTextControlGuard, Usage, funding::CapturePinIdentity, residual::RegisteredStoragePin,
};
use crate::{inspection::PrefillChunkRetentionContext, prefill::PrefillChunk};
use std::{marker::PhantomData, mem::size_of, sync::TryLockError};

#[derive(Debug)]
pub(in crate::working_memory) struct PinLayout {
    key: TypeId,
    rows: Box<[PinRow]>,
    bytes: u64,
}
#[derive(Debug)]
struct PinRow {
    chunk: PrefillChunk,
    slots: usize,
}
impl PinLayout {
    pub(in crate::working_memory) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(in crate::working_memory) fn matches<K: 'static>(&self) -> bool {
        self.key == TypeId::of::<K>()
    }
}

/// Cold typed slot ceilings for the exact immutable equation schedule. This is
/// a provider storage/control fact, not an allocation or publication authority.
/// Key payload allocations and physical-owner controls remain provider facts.
#[derive(Debug)]
pub struct PreparedPrefillStoragePinPlan<K: Ord + Send + 'static> {
    plan: InferenceSpanWorkspacePlan,
    layout: Arc<PinLayout>,
    marker: PhantomData<fn() -> K>,
}
impl<K: Ord + Send + 'static> PreparedPrefillStoragePinPlan<K> {
    /// The callback supplies a bound for every actual prefill row. Unknown is
    /// rejected; decode rows are outside this mechanism. Diagnostic construction
    /// belongs to original cold preparation, not its later protected use.
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
                    .checked_add(attempt_peak::<K>(slots)?)
                    .ok_or(WorkingMemoryError::Overflow)?;
                rows.push(PinRow {
                    chunk: chunk.clone(),
                    slots,
                });
            }
        }
        // Sum every row: all successful groups AND all failed attempts may
        // coexist. Fixed bank/layout/diagnostic moves and Vec->Box overlap are
        // separate from each row's final core, scratch and constructor moves.
        bytes = bytes
            .checked_add(
                count
                    .checked_mul(size_of::<PinRow>())
                    .and_then(|b| b.checked_mul(2))
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .and_then(|b| b.checked_add(size_of::<PinLayout>() + 2 * size_of::<usize>()))
            .and_then(|b| {
                b.checked_add(
                    3 * (size_of::<Self>() + size_of::<OriginalPrefillStoragePinSlots<K>>()),
                )
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(Self {
            plan: plan.clone(),
            layout: Arc::new(PinLayout {
                key: TypeId::of::<K>(),
                rows: rows.into_boxed_slice(),
                bytes,
            }),
            marker: PhantomData,
        })
    }
    /// Full finite S contribution, added once beside original P and Q.
    pub fn control_peak_bytes(&self) -> u64 {
        self.layout.bytes
    }
    pub(in crate::working_memory) fn into_layout(
        self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<Arc<PinLayout>, WorkingMemoryError> {
        if !self.plan.same_plan(plan) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(self.layout)
    }
}
fn attempt_peak<K: Ord + Send + 'static>(slots: usize) -> Result<usize, WorkingMemoryError> {
    // Inputs (including duplicates), unique final keys, capacity/ordinal staging.
    let payload = size_of::<(Option<K>, u64)>()
        .checked_add(size_of::<K>())
        .and_then(|n| n.checked_add(size_of::<PinOrdinal>()))
        .and_then(|n| n.checked_mul(slots))
        .ok_or(WorkingMemoryError::Overflow)?;
    // Both row Arc allocations and the original retained stamp allocation have
    // their actual payload/counters. Successful stamps can coexist across rows.
    let heap = size_of::<Registration<K>>()
        .checked_add(size_of::<BoundedPinCore<K>>())
        .and_then(|n| n.checked_add(CapturePinIdentity::retained_control_bytes()))
        .and_then(|n| n.checked_add(4 * size_of::<usize>()))
        .ok_or(WorkingMemoryError::Overflow)?;
    let moves = size_of::<BoundedPinAttempt<K>>()
        .checked_add(size_of::<FailedBoundedPinAttempt<K>>())
        .and_then(|n| n.checked_add(size_of::<BoundedRegisteredStorage<K>>()))
        .and_then(|n| n.checked_add(size_of::<Registration<K>>() + size_of::<BoundedPinCore<K>>()))
        .and_then(|n| n.checked_mul(3))
        .ok_or(WorkingMemoryError::Overflow)?;
    payload
        .checked_add(heap)
        .and_then(|n| n.checked_add(moves))
        .and_then(|n| n.checked_add(retirement_control_bytes::<K>()?))
        .ok_or(WorkingMemoryError::Overflow)
}

// One typed payload extraction can pass through the erased slot/quarantine
// carriers. Sum these concrete temporary controls for every finite row in S.
fn retirement_control_bytes<K: Ord + Send + 'static>() -> Option<usize> {
    size_of::<Option<BoundedPinCore<K>>>()
        .checked_add(size_of::<Arc<BoundedPinCore<K>>>())?
        .checked_add(size_of::<OpeningPinOwner>())?
        .checked_add(size_of::<RegisteredStoragePin>())
}

/// One bank extracted from the consuming original accepted owner. Each row may
/// issue one attempt; failures, panic and Drop never replace or refund it.
#[derive(Debug)]
#[must_use]
pub struct OriginalPrefillStoragePinSlots<K: Ord + Send + 'static> {
    layout: Arc<PinLayout>,
    next: usize,
    controls: OriginalTextControlGuard,
    marker: PhantomData<fn() -> K>,
}
impl<K: Ord + Send + 'static> OriginalPrefillStoragePinSlots<K> {
    pub(in crate::working_memory) fn new(
        layout: Arc<PinLayout>,
        controls: OriginalTextControlGuard,
    ) -> Self {
        Self {
            layout,
            next: 0,
            controls,
            marker: PhantomData,
        }
    }
    /// Rows already issued, including failed/abandoned attempts.
    pub fn spent_rows(&self) -> usize {
        self.next
    }
    /// Validate the exact canonical chunk, current stamped slot and original
    /// account under one Usage lock. Consume the row before allocating scratch.
    /// Busy before issuance leaves this bank unchanged; an issued attempt cannot
    /// be retried or replaced after a failed commit.
    pub fn begin(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<BoundedPinAttempt<K>, BoundedPinError> {
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
        let pool = native.pool();
        {
            let usage = pool.0.usage.try_lock().map_err(lock_error)?;
            self.controls.custody.validate_pin_locked(
                pool,
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
        // Custody is present before scratch construction and remains with a
        // partial attempt during unwinding. No later operation grows these Vecs.
        let controls = self.controls.clone();
        Ok(BoundedPinAttempt {
            inputs: Vec::with_capacity(slots),
            ordinals: Vec::with_capacity(slots),
            output: BoundedRegisteredStorage(Some(Arc::new(BoundedPinCore {
                storage: WorkingMemoryStorage::pending(Vec::with_capacity(slots), 0),
                identity,
                layout: self.layout.clone(),
                controls,
            }))),
            slots,
            _controls: self.controls.clone(),
        })
    }
}

/// Nonblocking inspection rejection or original typed accounting cause.
#[derive(Debug, thiserror::Error)]
pub enum BoundedPinError {
    #[error("storage accounting inspection is busy")]
    /// Usage is already borrowed; no owner count or balance changed.
    Busy,
    #[error(transparent)]
    /// Original identity, health, capacity, overflow or poison rejection.
    Storage(#[from] WorkingMemoryError),
}
fn lock_error<T>(e: TryLockError<T>) -> BoundedPinError {
    match e {
        TryLockError::WouldBlock => BoundedPinError::Busy,
        TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
    }
}
struct PinOrdinal {
    bytes: u64,
    ordinal: EntryLocator,
}
/// A consumed row's fixed input and staging owners. It grants no new capacity.
#[must_use]
pub struct BoundedPinAttempt<K: Ord + Send + 'static> {
    inputs: Vec<(Option<K>, u64)>,
    ordinals: Vec<PinOrdinal>,
    output: BoundedRegisteredStorage<K>,
    slots: usize,
    _controls: OriginalTextControlGuard,
}
impl<K: Ord + Send + 'static> std::fmt::Debug for BoundedPinAttempt<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedPinAttempt")
            .field("inputs", &self.inputs.len())
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}
impl<K: Ord + Send + 'static> BoundedPinAttempt<K> {
    /// Move an existing key into its prepriced slot. A rejected key is returned
    /// untouched; its caller retains its own payload/control obligations. No
    /// comparisons, cloning, registration or allocation occur here.
    pub fn push_owned(&mut self, key: K, bytes: u64) -> Result<(), K> {
        if self.inputs.len() == self.slots {
            return Err(key);
        }
        self.inputs.push((Some(key), bytes));
        Ok(())
    }
    /// Atomically pin only already registered keys. Errors retain every input,
    /// staged key and original controls in a terminal owner; there is no retry.
    pub fn pin_registered(
        mut self,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<BoundedRegisteredStorage<K>, FailedBoundedPinAttempt<K>> {
        match self.commit(native, segment) {
            Ok(()) => Ok(self.output),
            Err(cause) => Err(FailedBoundedPinAttempt {
                attempt: self,
                cause,
            }),
        }
    }
    fn commit(
        &mut self,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), BoundedPinError> {
        let core = self
            .output
            .core_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let registration =
            Arc::get_mut(&mut core.storage.0).ok_or(WorkingMemoryError::IdentityMismatch)?;
        // Provider comparisons occur while self still owns every key, before
        // Usage and before any owner increment. Duplicates remain in inputs.
        for (input, bytes) in &mut self.inputs {
            let key = input.as_ref().ok_or(WorkingMemoryError::IdentityMismatch)?;
            if let Some(i) = registration
                .keys
                .iter()
                .position(|old| old.cmp(key) == Ordering::Equal)
            {
                same_capacity(self.ordinals[i].bytes, *bytes)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            } else {
                registration.bytes = registration
                    .bytes
                    .and_then(|total| total.checked_add(*bytes));
                registration
                    .keys
                    .push(input.take().ok_or(WorkingMemoryError::IdentityMismatch)?);
                self.ordinals.push(PinOrdinal {
                    bytes: *bytes,
                    ordinal: EntryLocator::Fixed { batch: 0, slot: 0 },
                });
            }
        }
        let pool = native.pool().clone();
        // Prepare the final pool owner before commit, never after it.
        let registered_pool = pool.clone();
        let mut usage = pool.0.usage.try_lock().map_err(lock_error)?;
        core.controls.custody.validate_pin_locked(
            &pool,
            &usage,
            native,
            &core.layout,
            core.identity.request()?,
        )?;
        segment.validate_pin_locked(
            native,
            &usage,
            &core.identity,
            None,
            core.controls.custody.pin_source()?,
        )?;
        if !registration.keys.is_empty() {
            let registry = usage
                .storage
                .get(&TypeId::of::<K>())
                .and_then(|r| r.downcast_ref::<Registry<K>>())
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            for (key, staged) in registration.keys.iter().zip(self.ordinals.iter_mut()) {
                let (ordinal, entry) = registry
                    .locate(key)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                same_capacity(entry.bytes, staged.bytes)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
                validate_entry_origin(entry, &usage)?;
                entry
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                staged.ordinal = ordinal;
            }
            for (i, staged) in self.ordinals.iter().enumerate() {
                if self.ordinals[..i]
                    .iter()
                    .any(|prior| prior.ordinal == staged.ordinal)
                {
                    return Err(WorkingMemoryError::IdentityMismatch.into());
                }
            }
            let registry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .and_then(|r| r.downcast_mut::<Registry<K>>())
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            // No provider comparisons/clones/drop, allocation or fallible work
            // from this point through activation. All ordinals are distinct.
            for staged in &self.ordinals {
                registry.at_mut(staged.ordinal).owners += 1;
            }
        }
        registration.pool = Some(registered_pool);
        drop(usage);
        Ok(())
    }
}
/// Terminal failed row. Retains keys before controls; exposes only its cause.
#[must_use]
pub struct FailedBoundedPinAttempt<K: Ord + Send + 'static> {
    attempt: BoundedPinAttempt<K>,
    cause: BoundedPinError,
}
impl<K: Ord + Send + 'static> FailedBoundedPinAttempt<K> {
    /// Original typed failure, without releasing any retained input.
    pub fn cause(&self) -> &BoundedPinError {
        &self.cause
    }
    /// Number of accepted input slots, including retained duplicate keys.
    pub fn retained_input_count(&self) -> usize {
        self.attempt.inputs.len()
    }
}
impl<K: Ord + Send + 'static> std::fmt::Debug for FailedBoundedPinAttempt<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailedBoundedPinAttempt")
            .field("cause", &self.cause)
            .field("attempt", &self.attempt)
            .finish()
    }
}
impl<K: Ord + Send + 'static> std::fmt::Display for FailedBoundedPinAttempt<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<K: Ord + Send + 'static> std::error::Error for FailedBoundedPinAttempt<K> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct BoundedPinCore<K: Ord + Send + 'static> {
    // Existing grouped registration drops all provider keys before custody.
    storage: WorkingMemoryStorage<K>,
    // The successful core keeps the original stamp, never a newly minted one.
    // Keys retire first; stamp/layout retire before the original P+Q+S custody.
    identity: CapturePinIdentity,
    layout: Arc<PinLayout>,
    controls: OriginalTextControlGuard,
}
/// Escaped grouped pin retaining its actual original S (and P+Q) custody.
/// Clones share the same core and original chunk stamp (including bank H).
/// No raw registration/guard extraction is exposed.
#[must_use]
pub struct BoundedRegisteredStorage<K: Ord + Send + 'static>(Option<Arc<BoundedPinCore<K>>>);
impl<K: Ord + Send + 'static> BoundedPinCore<K> {
    fn retire(self: Arc<Self>) {
        // All typed and erased strong owners use this exact concrete operation.
        // No Weak escapes. Its winner owns the payload after Arc deallocation.
        drop(Arc::into_inner(self));
    }
}
impl<K: Ord + Send + 'static> Drop for BoundedRegisteredStorage<K> {
    fn drop(&mut self) {
        if let Some(core) = self.0.take() {
            core.retire();
        }
    }
}
impl<K: Ord + Send + 'static> Clone for BoundedRegisteredStorage<K> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.core())))
    }
}
impl<K: Ord + Send + 'static> std::fmt::Debug for BoundedRegisteredStorage<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedRegisteredStorage")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}
impl<K: Ord + Send + 'static> BoundedRegisteredStorage<K> {
    fn core(&self) -> &Arc<BoundedPinCore<K>> {
        self.0.as_ref().expect("live bounded pin owner")
    }
    fn core_mut(&mut self) -> Option<&mut BoundedPinCore<K>> {
        Arc::get_mut(self.0.as_mut()?)
    }

    /// Unique physical capacity pinned, not a new charge or scalar allowance.
    pub fn bytes(&self) -> Option<u64> {
        self.core().storage.bytes()
    }
    /// Same namespace and every existing allocation origin's current health.
    /// Closing the original run alone does not invalidate healthy source pins.
    pub fn validate_source(&self, pool: &MemoryLedger) -> Result<(), BoundedPinError> {
        let usage = pool.0.usage.try_lock().map_err(lock_error)?;
        self.core()
            .controls
            .custody
            .validate_pin_origin_locked(pool, &usage)?;
        self.core().storage.validate_copy_source(pool, &usage)?;
        Ok(())
    }
}

// Closed erasure of an already allocated successful core. No caller-provided
// validator or new wrapper allocation can enter a capture slot.
pub(in crate::working_memory) trait OpeningPinGroup: Send + Sync {
    fn validate_origins(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError>;
    fn validate_activation(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
        segment: &CaptureSourceSegment,
        context: &PrefillChunkRetentionContext<'_>,
        controls: &OriginalTextControlGuard,
    ) -> Result<(), WorkingMemoryError>;
    fn retire(self: Arc<Self>);
}
impl<K: Ord + Send + Sync + 'static> OpeningPinGroup for BoundedPinCore<K> {
    fn validate_origins(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.controls
            .custody
            .validate_pin_origin_locked(pool, usage)?;
        self.storage.validate_copy_source(pool, usage)
    }
    fn validate_activation(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
        segment: &CaptureSourceSegment,
        context: &PrefillChunkRetentionContext<'_>,
        controls: &OriginalTextControlGuard,
    ) -> Result<(), WorkingMemoryError> {
        // The caller already validated this exact receipt's complete controls
        // under Usage. Compare the actual owner, not equal diagnostic amounts.
        if !self.controls.custody.same(&controls.custody) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        segment.validate_pin_association_locked(
            native,
            usage,
            &self.identity,
            Some(context),
            self.controls.custody.pin_source()?,
        )?;
        self.storage.validate_copy_source(native.pool(), usage)
    }
    fn retire(self: Arc<Self>) {
        BoundedPinCore::retire(self);
    }
}
impl<K: Ord + Send + Sync + 'static> BoundedRegisteredStorage<K> {
    pub(in crate::working_memory) fn validate_install_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
        segment: &CaptureSourceSegment,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.core().controls.custody.validate_pin_locked(
            native.pool(),
            usage,
            native,
            &self.core().layout,
            context.request(),
        )?;
        segment.validate_pin_association_locked(
            native,
            usage,
            &self.core().identity,
            Some(context),
            self.core().controls.custody.pin_source()?,
        )?;
        self.core()
            .storage
            .validate_copy_source(native.pool(), usage)
    }
    pub(in crate::working_memory) fn into_opening(mut self) -> OpeningPinOwner {
        OpeningPinOwner(Some(self.0.take().expect("live bounded pin owner")))
    }
}

// Closed, allocation-free erasure of the same successful concrete core. No
// validator implementation or raw strong/Weak handle can enter from a caller.
pub(in crate::working_memory) struct OpeningPinOwner(Option<Arc<dyn OpeningPinGroup>>);
impl Clone for OpeningPinOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live opening pin owner"),
        )))
    }
}
impl Drop for OpeningPinOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
impl std::ops::Deref for OpeningPinOwner {
    type Target = dyn OpeningPinGroup;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live opening pin owner")
    }
}
impl OpeningPinOwner {
    pub(in crate::working_memory) fn into_pin(self) -> RegisteredStoragePin {
        RegisteredStoragePin::from_opening(self)
    }
}
