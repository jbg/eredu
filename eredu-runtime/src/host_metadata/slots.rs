//! Fixed mutable host slots with independently retained accounting metadata.

use super::{HostMetadataIdentity, MetadataCustody};
use eredu_core::{SharedStorageAccountingId, SharedStorageAttachmentError};
use std::{
    fmt,
    mem::size_of,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

/// A rejected attachment to one fixed host slot source.
#[derive(Debug, thiserror::Error)]
pub enum HostSlotAttachmentError<E> {
    /// An originally admitted table has one fixed domain; attaching another
    /// account would require an independently admitted destination operation.
    #[error("original reset table belongs to a different domain")]
    OriginalDomainMismatch,

    /// The slot table began retirement. An escaped token retains old custody
    /// only; it cannot acquire a new attachment or restore access to payload.
    #[error("host slot source has retired")]
    Retired,
    /// Existing per-domain accounting attachment failed. Poison remains an
    /// error, including after retirement; it is never interpreted as settlement.
    #[error("{0}")]
    Attachment(#[from] SharedStorageAttachmentError<E>),
}

/// Move-only ownership of an actual fixed boxed host payload.
///
/// Only inline slot storage (`len * size_of::<T>()`) is described. Nested native
/// arrays, vectors, strings and other owners in `T` require separate inventory.
/// This constructor neither prices those resources nor grants funding for the
/// original allocation. Callers must already hold its construction authority.
///
/// The payload is outside Arc and locks. Mutation uses an exclusive slice borrow
/// and cannot grow the table. No allocating Clone or owning slice export exists;
/// an independently copied table needs a new owner and prior copy authority.
/// A copy operation must borrow this actual table, never just its metadata token.
pub struct HostSlotTable<T> {
    // Retirement is marked in Drop, before either field is destroyed. Elements
    // and all their nested payloads then retire before this table's token.
    slots: Box<[T]>,
    metadata: HostSlotMetadata,
}

impl<T> HostSlotTable<T> {
    /// Exact metadata constructors around an already-paid boxed slot payload.
    /// Slot contents and the boxed extent require their own source inventory.
    pub fn host_source_control_bytes() -> Option<usize> {
        use eredu_core::{BackendFailure, HostMetadataFunding, HostPreparationAuthority};
        let parts = [
            HostSlotMetadata::prepared_host_control_bytes()?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?,
            size_of::<(
                Box<[T]>,
                &HostMetadataFunding,
                HostPreparationAuthority,
                Self,
            )>(),
            size_of::<Result<Self, BackendFailure>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Installs the existing prepared-host metadata owner under actual funding.
    /// The input extent is moved, never copied or resized. Its caller must have
    /// paid the extent and every nested child before this metadata handoff.
    pub fn from_boxed_with_host_source(
        slots: Box<[T]>,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, eredu_core::BackendFailure> {
        let bytes = Self::host_source_control_bytes()
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        let authority = eredu_core::HostPreparationAuthority::retain(funding.clone());
        let identity = HostMetadataIdentity::prepared_host(&authority)
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Self::new_prepared_host(slots, identity, &authority))
    }

    /// Transfers an existing exact boxed extent without cloning its elements.
    /// Ownership/attachment metadata is created under the caller's authority.
    pub fn new(slots: Box<[T]>) -> Self {
        let len = slots.len();
        let slot_size = size_of::<T>();
        let capacity = len
            .checked_mul(slot_size)
            .and_then(|bytes| u64::try_from(bytes).ok());
        Self {
            slots,
            metadata: HostSlotMetadata(Some(Arc::new(SlotMetadata {
                live: SlotLiveness::Ordinary(Mutex::new(true)),
                len,
                slot_size,
                capacity,
                custody: MetadataCustody::new(),
            }))),
        }
    }

    /// The ordinary slot owner with both metadata mutexes initialized under
    /// the participating workspace constructor's allowance.
    pub(crate) fn new_workspace(
        slots: Box<[T]>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        use eredu_nn::workspace::WorkspaceMetadataError;
        if !context.uses_checked_metadata() {
            return Ok(Self::new(slots));
        }
        let bytes = HostSlotMetadata::workspace_control_bytes()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        context.charge_metadata(bytes)?;
        let len = slots.len();
        let slot_size = size_of::<T>();
        let capacity = len
            .checked_mul(slot_size)
            .and_then(|bytes| u64::try_from(bytes).ok());
        let live = Mutex::new(true);
        drop(live.lock().expect("private new workspace table mutex"));
        Ok(Self {
            slots,
            metadata: HostSlotMetadata(Some(Arc::new(SlotMetadata {
                live: SlotLiveness::Ordinary(live),
                len,
                slot_size,
                capacity,
                custody: MetadataCustody::new_text(),
            }))),
        })
    }

    pub(crate) fn new_prepared_host(
        slots: Box<[T]>,
        identity: HostMetadataIdentity,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Self {
        let len = slots.len();
        let slot_size = size_of::<T>();
        let capacity = len
            .checked_mul(slot_size)
            .and_then(|n| u64::try_from(n).ok());
        let live = Mutex::new(true);
        // Initialize while exclusively owned: one actual PAL allocation, with
        // no competing lazy candidates after metadata is shared.
        drop(live.lock().expect("private new table mutex"));
        Self {
            slots,
            metadata: HostSlotMetadata(Some(Arc::new(SlotMetadata {
                live: SlotLiveness::Ordinary(live),
                len,
                slot_size,
                capacity,
                custody: MetadataCustody::new_prepared_host(identity, authority),
            }))),
        }
    }

    pub(crate) fn original_reset(
        slots: Box<[T]>,
        identity: HostMetadataIdentity,
        custody: crate::working_memory::resident_reset::ResetCustody,
    ) -> Self {
        let len = slots.len();
        let slot_size = size_of::<T>();
        let capacity = len
            .checked_mul(slot_size)
            .and_then(|n| u64::try_from(n).ok());
        Self {
            slots,
            metadata: HostSlotMetadata(Some(Arc::new(SlotMetadata {
                live: SlotLiveness::Original(AtomicBool::new(true)),
                len,
                slot_size,
                capacity,
                custody: MetadataCustody::original_reset(custody, identity),
            }))),
        }
    }

    /// Exact immutable slot borrow. This borrow, not a token, establishes access
    /// to the source contents for any future closed copy plan.
    pub fn slots(&self) -> &[T] {
        &self.slots
    }

    /// Exclusive fixed-extent mutation. This grants no authority to allocate
    /// nested resources, replace their inventory, or resize the table.
    pub fn slots_mut(&mut self) -> &mut [T] {
        &mut self.slots
    }

    /// Number of inline slots, including absent or zero-sized values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the table has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Payload-free accounting token. Cloning this handle does not retain slot
    /// contents and cannot create a copy or execution permission.
    pub fn metadata(&self) -> &HostSlotMetadata {
        &self.metadata
    }
}

impl<T> Drop for HostSlotTable<T> {
    fn drop(&mut self) {
        self.metadata.retire();
        // No lock remains held when Rust drops slots, followed by metadata.
    }
}

impl<T> fmt::Debug for HostSlotTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotTable")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

enum SlotLiveness {
    Ordinary(Mutex<bool>),
    // This original mode never installs an attachment. A live query has no
    // side effect and may linearize at its atomic load; no native mutex exists.
    Original(AtomicBool),
}

struct SlotMetadata {
    // Ordinary cold attachment/retirement serializes with its metadata mutex.
    // Original fixed-domain tables only load/store an atomic liveness flag.
    live: SlotLiveness,
    len: usize,
    slot_size: usize,
    capacity: Option<u64>,
    custody: MetadataCustody,
}

/// Owned, type-erased accounting identity for a fixed host slot table.
///
/// The sealed extent comes only from an actual `HostSlotTable<T>`. The token
/// contains neither its elements nor a pointer through which they can be read.
/// Escaped tokens may conservatively retain existing charges after elements
/// retire. Even a live token is not source-data evidence: closed copy plans must
/// also borrow the exact table and retain the relevant execution/source guard.
///
/// Identity and per-domain attachments use the existing metadata protocol.
/// The table marks retirement before destroying payload. Attachments serialize
/// with that mark; payload and attached-handle destruction occurs outside locks.
pub struct HostSlotMetadata(Option<Arc<SlotMetadata>>);

impl Clone for HostSlotMetadata {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live metadata"))))
    }
}
impl Drop for HostSlotMetadata {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl HostSlotMetadata {
    fn inner(&self) -> &SlotMetadata {
        self.0.as_deref().expect("live metadata")
    }
    pub(crate) fn original_control_bytes() -> Option<usize> {
        let allocation = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<SlotMetadata>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        allocation
            .checked_add(size_of::<SlotMetadata>())?
            .checked_add(size_of::<Option<SlotMetadata>>())?
            .checked_add(size_of::<HostSlotMetadata>())
    }

    pub(crate) fn workspace_control_bytes() -> Option<usize> {
        use crate::working_memory::{qualified_shared_bytes, OriginalHostMetadataCustody};
        let mutex =
            usize::try_from(OriginalHostMetadataCustody::initialized_mutex_bytes().ok()?).ok()?;
        [
            Self::original_control_bytes()?,
            usize::try_from(qualified_shared_bytes::<()>().ok()?).ok()?,
            MetadataCustody::text_control_bytes()?,
            mutex.checked_mul(2)?,
            size_of::<Mutex<bool>>(),
            size_of::<std::sync::MutexGuard<'static, bool>>(),
            size_of::<
                Result<
                    std::sync::MutexGuard<'static, bool>,
                    std::sync::PoisonError<std::sync::MutexGuard<'static, bool>>,
                >,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn prepared_host_control_bytes() -> Option<usize> {
        use crate::working_memory::OriginalHostMetadataCustody;
        let mutex =
            usize::try_from(OriginalHostMetadataCustody::initialized_mutex_bytes().ok()?).ok()?;
        [
            Self::original_control_bytes()?,
            HostMetadataIdentity::original_control_bytes()?,
            MetadataCustody::text_control_bytes()?,
            mutex.checked_mul(2)?,
            size_of::<Mutex<bool>>(),
            size_of::<std::sync::MutexGuard<'static, bool>>(),
            size_of::<
                Result<
                    std::sync::MutexGuard<'static, bool>,
                    std::sync::PoisonError<std::sync::MutexGuard<'static, bool>>,
                >,
            >(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
            size_of::<Result<HostMetadataIdentity, crate::working_memory::WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn original_reset_custody(
        &self,
    ) -> Option<&crate::working_memory::resident_reset::ResetCustody> {
        self.inner().custody.reset.as_ref()
    }

    /// Process-local identity, independent of payload and token lifetime.
    pub fn identity(&self) -> &HostMetadataIdentity {
        self.inner().custody.identity()
    }

    /// Exact number of slots in the original boxed source.
    pub fn len(&self) -> usize {
        self.inner().len
    }

    /// Whether the original source has no slots.
    pub fn is_empty(&self) -> bool {
        self.inner().len == 0
    }

    /// Inline size of one slot. Zero-sized values may have a nonzero count.
    pub fn slot_size(&self) -> usize {
        self.inner().slot_size
    }

    /// Checked inline boxed extent only. No nested payload, table/Arc/custody
    /// metadata or allocator overhead is included. Overflow remains unknown.
    pub fn capacity_bytes(&self) -> Option<u64> {
        self.inner().capacity
    }

    /// Whether both tokens name the same original table and attachment custody.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live metadata"),
            other.0.as_ref().expect("live metadata"),
        )
    }

    pub(crate) fn original_source_is_live(&self) -> bool {
        match &self.inner().live {
            SlotLiveness::Original(live) => live.load(Ordering::Acquire),
            SlotLiveness::Ordinary(_) => false,
        }
    }

    /// Attaches one accounting handle per domain while the table is live.
    ///
    /// An existing live attachment returns false without calling the provider.
    /// Retirement rejects even an already attached domain. The provider runs
    /// under the cold lifecycle gate and existing metadata attachment lock;
    /// perform only closed accounting, with no table/token reentry, nested owner
    /// locks, native work or user callbacks. Returned handles must retain only
    /// independent identity/domain keys, never this token or its table, which
    /// would form a custody cycle. Provider errors leave earlier attachments.
    ///
    /// Retired or poisoned rejection does not invoke the provider. A provider
    /// already admitted by the gate completes before retirement destroys any
    /// element. No alive check exposed here can replace an actual table borrow.
    /// Read-only existing attachment or original fixed-table domain check.
    pub fn validate_original_attachment(
        &self,
        domain: &SharedStorageAccountingId,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        if let Some(reset) = &self.inner().custody.reset {
            return if reset.same_accounting_owner(domain) {
                Ok(())
            } else {
                Err(crate::working_memory::WorkingMemoryError::IdentityMismatch)
            };
        }
        self.inner().custody.original_attachment_ready(domain)
    }

    /// Exact single attachment slot and fixed controls for a copied host table.
    /// The enclosing copy host plan must include this before construction.
    #[doc(hidden)]
    pub fn copy_attachment_control_bytes() -> Option<usize> {
        [
            MetadataCustody::text_attachment_control_bytes()?,
            size_of::<(&Self, &SharedStorageAccountingId)>(),
            size_of::<std::sync::MutexGuard<'_, bool>>(),
            size_of::<Result<(), crate::working_memory::WorkingMemoryError>>(),
            size_of::<
                Result<bool, HostSlotAttachmentError<crate::working_memory::WorkingMemoryError>>,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Prepares one actual first-domain attachment after the enclosing H grant.
    /// Only metadata created by the prepared host constructor is eligible; no
    /// source table is promoted and no native/accounting authority is created.
    #[doc(hidden)]
    pub fn prepare_copy_attachment(
        &self,
        domain: &SharedStorageAccountingId,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        use crate::working_memory::WorkingMemoryError as E;
        let SlotLiveness::Ordinary(live) = &self.inner().live else {
            return Err(E::IdentityMismatch);
        };
        let live = live.lock().map_err(|_| E::Poisoned)?;
        if !*live {
            return Err(E::ExecutionFenced);
        }
        self.inner().custody.prepare_copy_attachment(domain)
    }

    /// Attaches a concrete accounting owner after funding its prospective node.
    /// The liveness gate and metadata lock are released before unused captures
    /// or the returned error can retire. No native authority is created.
    pub(crate) fn try_attach_owned_prepared<T: eredu_core::SharedStorageRetirement, E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(
            eredu_core::SharedStorageAttachmentLayout,
        ) -> Result<eredu_core::SharedStorageOwner<T>, E>,
    ) -> Result<bool, HostSlotAttachmentError<E>> {
        let SlotLiveness::Ordinary(live) = &self.inner().live else {
            return Err(HostSlotAttachmentError::OriginalDomainMismatch);
        };
        let mut acquire = Some(acquire);
        let result = (|| {
            let live = live.lock().map_err(|_| {
                HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Poisoned)
            })?;
            if !*live {
                return Err(HostSlotAttachmentError::Retired);
            }
            self.inner()
                .custody
                .try_attach_owned_prepared(owner, |layout| {
                    acquire.take().expect("single provider")(layout)
                })
                .map_err(HostSlotAttachmentError::Attachment)
        })();
        drop(acquire);
        result
    }

    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageAccountingId,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, HostSlotAttachmentError<E>> {
        self.try_attach_mode(domain, acquire, false)
    }

    /// Publishes into a previously prepared single attachment destination.
    /// Missing capacity rejects before invoking the provider; it never grows.
    #[doc(hidden)]
    pub fn try_attach_prepared_copy<E>(
        &self,
        domain: &SharedStorageAccountingId,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, HostSlotAttachmentError<E>> {
        self.try_attach_mode(domain, acquire, true)
    }
    fn try_attach_mode<E>(
        &self,
        domain: &SharedStorageAccountingId,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
        prepared_copy: bool,
    ) -> Result<bool, HostSlotAttachmentError<E>> {
        let ordinary = match &self.inner().live {
            SlotLiveness::Original(live) => {
                let result = if !live.load(Ordering::Acquire) {
                    Err(HostSlotAttachmentError::Retired)
                } else if self
                    .inner()
                    .custody
                    .reset
                    .as_ref()
                    .expect("original liveness has custody")
                    .same_accounting_owner(domain)
                {
                    Ok(false)
                } else {
                    Err(HostSlotAttachmentError::OriginalDomainMismatch)
                };
                // No attachment publication, mutex, or provider invocation is
                // possible in this mode. Captures retire outside all loans.
                drop(acquire);
                return result;
            }
            SlotLiveness::Ordinary(live) => live,
        };
        let live = ordinary.lock().map_err(|_| {
            HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Poisoned)
        })?;
        if !*live {
            return Err(HostSlotAttachmentError::Retired);
        }
        // Keep an uninvoked provider's captures here. The inner duplicate path
        // must not drop a captured preacquired charge under the outer gate.
        let mut acquire = Some(acquire);
        let invoke = || acquire.take().expect("provider invoked at most once")();
        let result = if prepared_copy {
            self.inner()
                .custody
                .try_attach_prepared_copy(domain, invoke)
        } else {
            self.inner().custody.try_attach(domain, invoke)
        }
        .map_err(HostSlotAttachmentError::Attachment);
        // Returned errors and skipped providers can own reentrant destructors.
        drop(live);
        drop(acquire);
        result
    }

    fn retire(&self) {
        // Poison is preserved. This only fences future attachment before payload
        // destruction; it neither releases old custody nor certifies any work.
        match &self.inner().live {
            SlotLiveness::Original(live) => live.store(false, Ordering::Release),
            SlotLiveness::Ordinary(live) => {
                let mut live = live.lock().unwrap_or_else(|poison| poison.into_inner());
                *live = false;
            }
        }
    }
}

impl fmt::Debug for HostSlotMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotMetadata")
            .field("identity", self.identity())
            .field("len", &self.len())
            .field("slot_size", &self.slot_size())
            .field("capacity_bytes", &self.capacity_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests;
