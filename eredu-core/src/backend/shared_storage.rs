//! Closed immutable sources and their per-domain accounting custody.

use super::{HostPreparationAuthority, SharedTokenFilter};
use std::{
    cmp::Ordering,
    collections::TryReserveError,
    fmt,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

mod declaration;
mod owned;
pub use declaration::{ControllerDeclarationData, SharedControllerDeclaration};
pub use owned::{ErasedSharedStorageOwner, SharedStorageOwner, SharedStorageRetirement};

/// Process-local identity of one immutable source allocation owner.
///
/// Value keys allocate nothing and retain neither source nor accounting owner.
/// Their monotonically assigned IDs are never reused, including after the source
/// retires. They are not persistent identities or execution permission.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SharedStorageIdentity(u64);
impl SharedStorageIdentity {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| next.checked_add(1))
            .expect("process-local source identity space exhausted"))
    }
    /// Identity keys have no independently allocated shell.
    pub const fn source_shell_bytes() -> Option<usize> { Some(0) }
}

/// Process-local identity of one accounting domain.
///
/// A new default value creates a different domain. This payload-free key
/// retains no runtime pool or native resources.
#[derive(Clone, Default)]
pub struct SharedStorageDomain(Arc<DomainPayload>);
type DomainPayload = ();

macro_rules! identity_traits {
    ($identity:ty) => {
        impl PartialEq for $identity {
            fn eq(&self, other: &Self) -> bool {
                self.as_ptr() == other.as_ptr()
            }
        }
        impl Eq for $identity {}
        impl PartialOrd for $identity {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $identity {
            fn cmp(&self, other: &Self) -> Ordering {
                self.as_ptr().cmp(&other.as_ptr())
            }
        }
        impl Hash for $identity {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.as_ptr().hash(state);
            }
        }
        impl fmt::Debug for $identity {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($identity))
                    .field(&self.as_ptr())
                    .finish()
            }
        }
    };
}
identity_traits!(SharedStorageDomain);

impl SharedStorageDomain {
    fn as_ptr(&self) -> *const DomainPayload { Arc::as_ptr(&self.0) }
    /// Layout of the payload in this identity's one shared allocation.
    /// The caller must separately qualify and include its Arc header. This
    /// creates no identity and grants no storage or execution authority.
    pub fn shared_payload_layout() -> std::alloc::Layout {
        std::alloc::Layout::new::<DomainPayload>()
    }

    /// Whether two keys identify exactly the same accounting domain.
    pub fn same_identity(&self, other: &Self) -> bool {
        self == other
    }
}

/// Failure to attach accounting custody to an immutable shared source.
#[derive(Debug, thiserror::Error)]
pub enum SharedStorageAttachmentError<E> {
    /// The domain already has opaque custody or a different typed owner.
    #[error("shared storage accounting attachment type does not match")]
    AttachmentMismatch,
    /// A nonblocking attachment found an acquisition already in progress.
    #[error("shared storage accounting custody is busy")]
    Busy,
    /// An earlier accounting acquisition panicked while holding custody.
    #[error("shared storage accounting custody is poisoned")]
    Poisoned,
    /// Custody metadata could not reserve a slot before provider acquisition.
    #[error("shared storage accounting metadata allocation failed: {0}")]
    Allocation(#[source] TryReserveError),
    /// The accounting provider rejected the attachment, preserving its cause.
    #[error("shared storage accounting provider rejected attachment: {0}")]
    Provider(#[source] E),
}

pub(super) struct Attachment {
    domain: SharedStorageDomain,
    _custody: AttachmentCustody,
}
enum AttachmentCustody {
    Opaque(#[allow(dead_code)] Box<dyn Send + Sync>),
    Typed(Arc<dyn std::any::Any + Send + Sync>),
    Owned(ErasedSharedStorageOwner),
}

/// Private shared mechanism. Each closed payload owner destroys its numerical
/// allocation before this field drops. No public arbitrary-payload promise is
/// exposed by this mechanism: filters, bytes, independently constructed immutable
/// declarations and admitted capture-plan owners use this custody.
pub(crate) struct SharedStorageCustody {
    identity: SharedStorageIdentity,
    pub(super) attachments: Mutex<Vec<Attachment>>,
}

impl SharedStorageCustody {
    pub(crate) fn new() -> Self {
        Self {
            identity: SharedStorageIdentity::new(),
            attachments: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn identity(&self) -> &SharedStorageIdentity {
        &self.identity
    }

    pub(crate) fn has_accounting_custody(
        &self,
        domain: &SharedStorageDomain,
    ) -> Result<bool, SharedStorageAttachmentError<std::convert::Infallible>> {
        let custody = self.attachments.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => SharedStorageAttachmentError::Busy,
            std::sync::TryLockError::Poisoned(_) => SharedStorageAttachmentError::Poisoned,
        })?;
        Ok(custody.iter().any(|entry| entry.domain == *domain))
    }

    pub(crate) fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        let mut custody = self
            .attachments
            .lock()
            .map_err(|_| SharedStorageAttachmentError::Poisoned)?;
        Self::attach_locked(&mut custody, domain, acquire)
    }

    pub(crate) fn try_attach_nonblocking<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        let mut custody = self.attachments.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => SharedStorageAttachmentError::Busy,
            std::sync::TryLockError::Poisoned(_) => SharedStorageAttachmentError::Poisoned,
        })?;
        Self::attach_locked(&mut custody, domain, acquire)
    }

    pub(crate) fn try_attach_typed_nonblocking<T: Send + Sync + 'static, E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, SharedStorageAttachmentError<E>> {
        let mut custody = self.attachments.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => SharedStorageAttachmentError::Busy,
            std::sync::TryLockError::Poisoned(_) => SharedStorageAttachmentError::Poisoned,
        })?;
        if let Some(entry) = custody.iter().find(|entry| entry.domain == *domain) {
            let AttachmentCustody::Typed(owner) = &entry._custody else {
                return Err(SharedStorageAttachmentError::AttachmentMismatch);
            };
            if !owner.is::<T>() {
                return Err(SharedStorageAttachmentError::AttachmentMismatch);
            }
            // Cloning/downcasting the existing Arc allocates nothing. The
            // attachment remains an owner throughout this borrowed lookup.
            return Ok(owner
                .clone()
                .downcast::<T>()
                .unwrap_or_else(|_| unreachable!("checked attachment type")));
        }
        custody
            .try_reserve(1)
            .map_err(SharedStorageAttachmentError::Allocation)?;
        let owner = acquire().map_err(SharedStorageAttachmentError::Provider)?;
        custody.push(Attachment {
            domain: domain.clone(),
            _custody: AttachmentCustody::Typed(owner.clone()),
        });
        Ok(owner)
    }

    pub(crate) fn owned_attachment_control_bytes<T: SharedStorageRetirement>() -> Option<usize> {
        owned::attachment_control_bytes::<T>()
    }

    pub(crate) fn try_attach_owned_nonblocking<T: SharedStorageRetirement, E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<SharedStorageOwner<T>, E>,
    ) -> Result<SharedStorageOwner<T>, SharedStorageAttachmentError<E>> {
        // Keep an unused provider in this outer frame: early reuse/mismatch or
        // lock failure must drop captures only after the local guard is gone.
        let mut acquire = Some(acquire);
        let result = (|| {
            let mut custody = self.attachments.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => SharedStorageAttachmentError::Busy,
                std::sync::TryLockError::Poisoned(_) => SharedStorageAttachmentError::Poisoned,
            })?;
            if let Some(entry) = custody.iter().find(|entry| entry.domain == *domain) {
                let AttachmentCustody::Owned(owner) = &entry._custody else {
                    return Err(SharedStorageAttachmentError::AttachmentMismatch);
                };
                return owner
                    .clone_typed::<T>()
                    .ok_or(SharedStorageAttachmentError::AttachmentMismatch);
            }
            // Preserved source metadata allocation, before provider acquisition.
            // This is not an allocation-free custody Vec redesign.
            custody
                .try_reserve(1)
                .map_err(SharedStorageAttachmentError::Allocation)?;
            let owner = acquire.take().expect("single acquisition")()
                .map_err(SharedStorageAttachmentError::Provider)?;
            custody.push(Attachment {
                domain: domain.clone(),
                _custody: AttachmentCustody::Owned(owner.clone().erase()),
            });
            Ok(owner)
        })();
        drop(acquire);
        result
    }

    fn attach_locked<E>(
        custody: &mut Vec<Attachment>,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        if custody.iter().any(|entry| entry.domain == *domain) {
            return Ok(false);
        }
        custody
            .try_reserve(1)
            .map_err(SharedStorageAttachmentError::Allocation)?;
        let handle = acquire().map_err(SharedStorageAttachmentError::Provider)?;
        custody.push(Attachment {
            domain: domain.clone(),
            _custody: AttachmentCustody::Opaque(handle),
        });
        Ok(true)
    }
}

impl Drop for SharedStorageCustody {
    fn drop(&mut self) {
        // Exclusive final access needs no lock. Provider destructors can reenter
        // accounting, including after a poisoned acquisition. No user-defined
        // destructor runs while the owner mutex is held.
        let attachments = self
            .attachments
            .get_mut()
            .unwrap_or_else(|p| p.into_inner());
        drop(std::mem::take(attachments));
    }
}

struct BytesInner {
    bytes: Vec<u8>,
    custody: SharedStorageCustody,
    authority: HostPreparationAuthority,
    #[cfg(test)]
    payload_retired: Option<Arc<std::sync::atomic::AtomicBool>>,
}
impl SharedStorageRetirement for BytesInner {
    fn retire(self: Arc<Self>) { drop(Arc::into_inner(self)); }
}

impl Drop for BytesInner {
    fn drop(&mut self) {
        // The closed Vec payload has no custom element destructors. Retire its
        // entire allocation before automatic field drop releases any charge.
        drop(std::mem::take(&mut self.bytes));
        #[cfg(test)]
        if let Some(retired) = &self.payload_retired {
            retired.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// One immutable byte allocation shared across controller source owners.
///
/// Clones made before attachment preserve later accounting custody until the
/// final alias retires. Borrowed slices do not escape their owner's lifetime;
/// there is no mutable or consuming raw-payload export. Copying bytes creates
/// independent caller-owned storage and does not inherit this owner's coverage.
#[derive(Clone)]
pub struct SharedControllerBytes(SharedStorageOwner<BytesInner>);

impl SharedControllerBytes {
    /// Exact source-owner and identity allocation requests, excluding the byte
    /// destination and later domain attachments. This grants no permission.
    pub fn source_shell_bytes() -> Option<usize> {
        use std::{alloc::Layout, sync::atomic::AtomicUsize};
        Layout::new::<[AtomicUsize; 2]>().extend(Layout::new::<BytesInner>()).ok()?
            .0.pad_to_align().size().checked_add(SharedStorageIdentity::source_shell_bytes()?)
    }
    /// Transfers the vector and retains its existing construction authority.
    /// The caller must pay for the destination and source shells before this
    /// call; supplying authority alone establishes no finite storage bound.
    pub fn new(bytes: Vec<u8>, authority: HostPreparationAuthority) -> Self {
        Self(SharedStorageOwner::new(BytesInner {
            bytes,
            custody: SharedStorageCustody::new(),
            authority,
            #[cfg(test)]
            payload_retired: None,
        }))
    }

    /// Borrows the immutable byte contents.
    pub fn as_ref(&self) -> &[u8] {
        &self.0.bytes
    }

    /// Exact allocation owner identity, independent of its byte contents.
    pub fn identity(&self) -> &SharedStorageIdentity {
        self.0.custody.identity()
    }

    /// Retained byte allocation capacity, including spare elements. Identity
    /// and custody metadata are not numerical storage; overflow remains unknown.
    pub fn capacity_bytes(&self) -> Option<u64> {
        u64::try_from(self.0.bytes.capacity()).ok()
    }

    /// Whether both handles share exactly this allocation owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.0.same_owner(&other.0)
    }

    /// Whether the retained construction authority holds this exact metadata
    /// account. This checks custody only; it does not certify prior allocations.
    pub fn retains_funding(&self, funding: &super::HostMetadataFunding) -> bool {
        self.0.authority.is_funded_by(funding)
    }

    /// Attaches one accounting handle per domain, including to earlier clones.
    ///
    /// Returns false without invoking the provider for an already attached
    /// domain. A slot is reserved before acquisition, so publishing a successful
    /// handle cannot fail. Rejection preserves all earlier attachments.
    ///
    /// The provider runs under owner custody and may perform only closed
    /// accounting operations: no owner reentry, other source locks, native work
    /// or user callbacks. Its handle must not retain this owner indirectly or
    /// directly. Register only payload-free identity/domain keys. Provider
    /// handles and errors retire outside the custody lock; attached handles
    /// outlive the byte allocation.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.custody.try_attach(domain, acquire)
    }
}

impl AsRef<[u8]> for SharedControllerBytes {
    fn as_ref(&self) -> &[u8] {
        SharedControllerBytes::as_ref(self)
    }
}
impl fmt::Debug for SharedControllerBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedControllerBytes")
            .field("len", &self.0.bytes.len())
            .field("capacity_bytes", &self.capacity_bytes())
            .field("identity", self.identity())
            .finish()
    }
}

/// Borrowed closed source in a controller's immutable shared inventory.
///
/// This descriptor owns no numerical payload and grants no storage credit.
/// Consumers preserve its kind when checking tokenizer-filter provenance.
#[derive(Debug, Clone, Copy)]
pub enum SharedControllerSource<'a> {
    /// Immutable tokenizer/filter mask storage.
    Filter(&'a SharedTokenFilter),
    /// Immutable non-mask source bytes.
    Bytes(&'a SharedControllerBytes),
    /// Immutable independently constructed declaration data. This supplies no
    /// tokenizer provenance or authority for executing from the declaration.
    Declaration(&'a SharedControllerDeclaration),
}

impl SharedControllerSource<'_> {
    /// Reads the exact existing attachment without registering or adopting data.
    pub fn has_accounting_custody(&self, domain: &SharedStorageDomain)
        -> Result<bool, SharedStorageAttachmentError<std::convert::Infallible>> {
        match self {
            Self::Filter(value) => value.has_accounting_custody(domain),
            Self::Bytes(value) => value.0.custody.has_accounting_custody(domain),
            Self::Declaration(value) => value.has_accounting_custody(domain),
        }
    }

    /// Exact source allocation owner identity.
    pub fn identity(&self) -> &SharedStorageIdentity {
        match self {
            Self::Filter(value) => value.identity(),
            Self::Bytes(value) => value.identity(),
            Self::Declaration(value) => value.identity(),
        }
    }

    /// Exact retained capacity, or unknown on conversion overflow.
    pub fn capacity_bytes(&self) -> Option<u64> {
        match self {
            Self::Filter(value) => value.capacity_bytes(),
            Self::Bytes(value) => value.capacity_bytes(),
            Self::Declaration(value) => value.capacity_bytes(),
        }
    }

    /// Planning allowance for the source. Byte buffers and filters report exact
    /// capacity; opaque declaration storage may use a configured estimate.
    pub fn admission_bytes(&self) -> Option<u64> {
        match self {
            Self::Declaration(value) => value.admission_bytes(),
            _ => self.capacity_bytes(),
        }
    }

    /// Uses the closed owner's accounting protocol. The provider restrictions
    /// and destruction guarantees of `SharedControllerBytes::try_attach` apply.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        match self {
            Self::Filter(value) => value.try_attach(domain, acquire),
            Self::Bytes(value) => value.try_attach(domain, acquire),
            Self::Declaration(value) => value.try_attach(domain, acquire),
        }
    }
}

#[cfg(test)]
mod tests;
