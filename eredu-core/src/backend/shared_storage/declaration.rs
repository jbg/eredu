//! Immutable declaration sources with the shared accounting/retirement worker.
use super::{
    ErasedSharedStorageOwner, SharedStorageAttachmentError, SharedStorageCustody,
    SharedStorageDomain, SharedStorageIdentity, SharedStorageOwner, SharedStorageRetirement,
};
use std::{fmt, sync::Arc};

/// Trusted immutable declaration producer used by neutral composition.
///
/// Implementations must own all reported allocations independently: no mutable
/// execution state, ordinary compiler/factory aliases, account ownership, or
/// hidden independently escaping numerical payload. The capacity is the exact
/// completed allocation population from the producer's checked constructor,
/// including spare capacity, and must remain fixed for this owner's lifetime.
/// Constructor scratch and fixed call frames are excluded. Reporting bytes
/// grants no allocation permission, original-domain qualification or execution.
pub trait ControllerDeclarationData: Send + Sync + 'static {
    /// Exact independently owned retained allocation bytes, or unknown.
    /// This query must be allocation-free and must not inspect external state.
    fn owned_capacity_bytes(&self) -> Option<u64>;
}
struct Inner<T> {
    value: T,
    custody: SharedStorageCustody,
}
impl<T: ControllerDeclarationData> SharedStorageRetirement for Inner<T> {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
fn custody<T: ControllerDeclarationData>(
    owner: &ErasedSharedStorageOwner,
) -> &SharedStorageCustody {
    &owner
        .downcast_ref::<Inner<T>>()
        .expect("closed declaration type")
        .custody
}
fn capacity<T: ControllerDeclarationData>(owner: &ErasedSharedStorageOwner) -> Option<u64> {
    owner
        .downcast_ref::<Inner<T>>()
        .expect("closed declaration type")
        .value
        .owned_capacity_bytes()
}
/// Shared immutable declaration data with exact identity and attached storage
/// accounting. Payload destruction precedes custody on every typed/erased exit.
/// No mutable, raw-Arc, Weak or consuming payload export exists.
#[derive(Clone)]
pub struct SharedControllerDeclaration {
    owner: ErasedSharedStorageOwner,
    custody: fn(&ErasedSharedStorageOwner) -> &SharedStorageCustody,
    capacity: fn(&ErasedSharedStorageOwner) -> Option<u64>,
}
impl SharedControllerDeclaration {
    /// Exact initial declaration-owner and identity allocation requests. The
    /// declaration's own reachable backings and later accounting attachments
    /// are separate. This query creates no owner, attachment or permission.
    pub fn source_shell_bytes<T: ControllerDeclarationData>() -> Option<usize> {
        use std::{alloc::Layout, sync::atomic::AtomicUsize};
        let header = Layout::new::<[AtomicUsize; 2]>();
        let owner = header.extend(Layout::new::<Inner<T>>()).ok()?.0.pad_to_align();
        let identity = header.extend(Layout::new::<()>()).ok()?.0.pad_to_align();
        owner.size().checked_add(identity.size())
    }

    /// Fixed borrowed type/identity/capacity inspection transports. This query
    /// performs no inspection and supplies no source or allocation permission.
    pub fn inspection_control_bytes<T: ControllerDeclarationData>() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<ErasedSharedStorageOwner>(),
            size_of::<Option<&Inner<T>>>(),
            size_of::<Option<&T>>(),
            size_of::<&(dyn std::any::Any + Send + Sync)>(),
            size_of::<fn(&ErasedSharedStorageOwner) -> &SharedStorageCustody>(),
            size_of::<fn(&ErasedSharedStorageOwner) -> Option<u64>>(),
            size_of::<Option<u64>>(),
            size_of::<&SharedStorageIdentity>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Transfers an independently constructed declaration without cloning it.
    /// Callers must establish their applicable constructor permission first;
    /// this constructor neither adopts managed work nor certifies its producer.
    pub fn new<T: ControllerDeclarationData>(value: T) -> Self {
        Self {
            owner: SharedStorageOwner::new(Inner {
                value,
                custody: SharedStorageCustody::new(),
            })
            .erase(),
            custody: custody::<T>,
            capacity: capacity::<T>,
        }
    }
    /// Borrows the original concrete declaration with no owning or mutable escape.
    pub fn declaration<T: ControllerDeclarationData>(&self) -> Option<&T> {
        self.owner
            .downcast_ref::<Inner<T>>()
            .map(|owner| &owner.value)
    }
    /// Exact allocation-owner identity, independent of declaration contents.
    pub fn identity(&self) -> &SharedStorageIdentity {
        (self.custody)(&self.owner).identity()
    }
    /// Actual completed capacity supplied by the closed declaration producer.
    pub fn capacity_bytes(&self) -> Option<u64> {
        (self.capacity)(&self.owner)
    }
    /// Whether both handles retain the same declaration allocation owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
    pub(super) fn has_accounting_custody(&self, domain: &SharedStorageDomain)
        -> Result<bool, SharedStorageAttachmentError<std::convert::Infallible>> {
        (self.custody)(&self.owner).has_accounting_custody(domain)
    }
    /// Same closed accounting attachment protocol as filters and byte sources.
    /// Providers must retain no declaration/payload alias; attached custody
    /// outlives payloads, and provider callbacks run under the custody lock.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        (self.custody)(&self.owner).try_attach(domain, acquire)
    }
}
impl fmt::Debug for SharedControllerDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedControllerDeclaration")
            .field("identity", self.identity())
            .field("capacity_bytes", &self.capacity_bytes())
            .finish()
    }
}
