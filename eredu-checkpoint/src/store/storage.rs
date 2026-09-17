//! Source-owned host payload bounds, preserving physical ownership across views.

use super::{CheckpointSource, StoreError};
use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Weak},
};

/// Identity of a source-owned host payload or immutable reader-storage ceiling.
/// Clones retain a weak reference to the original Arc allocation, preventing
/// address reuse even after its [`SourceStorage`] and source value retire.
/// Equality therefore remains valid for deferred accounting without retaining
/// the value strongly. The source value's destructor still runs normally;
/// the Arc allocation, including its inline storage, remains reserved until
/// the last weak reference retires.
///
/// This process-local token is not persistent or serializable. Its namespace
/// is distinct from native buffers and independently owned byte slices.
#[derive(Clone)]
pub struct SourceStorageIdentity(Weak<dyn Any + Send + Sync>, Option<SourceControl>);

impl SourceStorageIdentity {
    /// Borrow only this token's actual built-in source-constructor custody.
    /// Ordinary Arc identities have none. No amount, owner, or custody clone
    /// escapes; the runtime recognizes its own private concrete origin type.
    pub fn constructor_control_owner<C: Any>(&self) -> Option<&C> {
        self.1.as_ref()?.origin()
    }

    /// Retain the actual source allocation without inventing a payload-byte fact.
    pub(crate) fn for_owner<T: Any + Send + Sync>(owner: &Arc<T>) -> Self {
        let identity: Weak<T> = Arc::downgrade(owner);
        Self(identity, None)
    }

    fn address(&self) -> usize {
        // Compare only the Arc allocation's data address, not erased metadata.
        // The retained weak reference keeps this address unavailable for reuse.
        self.0.as_ptr() as *const () as usize
    }
}

impl std::fmt::Debug for SourceStorageIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SourceStorageIdentity")
            .field(&self.address())
            .finish()
    }
}

impl PartialEq for SourceStorageIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.address() == other.address()
    }
}

impl Eq for SourceStorageIdentity {}

impl PartialOrd for SourceStorageIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SourceStorageIdentity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.address().cmp(&other.address())
    }
}

impl std::hash::Hash for SourceStorageIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.address(), state);
    }
}

// Implemented directly on the existing Arc value. Borrowing this adapter does
// not allocate an erased box, clone the strong owner, or expose its payload.
trait BorrowedSourceOwner: Send + Sync {
    fn retain_owner(&self) -> Arc<dyn Any + Send + Sync>;
    fn storage_identity(&self) -> SourceStorageIdentity;
    fn control(&self) -> Option<SourceControl> {
        None
    }
}

impl<T: Any + Send + Sync> BorrowedSourceOwner for Arc<T> {
    fn retain_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.clone()
    }

    fn storage_identity(&self) -> SourceStorageIdentity {
        SourceStorageIdentity::for_owner(self)
    }
}

impl BorrowedSourceOwner for Arc<dyn Any + Send + Sync> {
    fn retain_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.clone()
    }

    fn storage_identity(&self) -> SourceStorageIdentity {
        SourceStorageIdentity(Arc::downgrade(self), None)
    }
}

/// Borrowed physical source owner and its complete payload capacity or immutable
/// reader-storage ceiling. Construction does not clone the Arc or allocate.
///
/// This is a neutral source fact, not a reservation or an execution grant.
/// Repeated visits may describe the same owner. Callers must check consistent
/// capacities, deduplicate identities and use checked arithmetic themselves.
/// Callback slots, retained owners and later publication/pins need their own
/// original-account host bound; this reference does not provide that budget.
#[derive(Clone, Copy)]
pub struct SourceStorageRef<'a> {
    owner: &'a dyn BorrowedSourceOwner,
    bytes: u64,
}

impl<'a> SourceStorageRef<'a> {
    /// Declares the same physical-owner contract as [`SourceStorage::insert`].
    /// The Arc must retain the entire reported payload or identify its immutable
    /// reader ceiling. Independent payloads require distinct owners or one owner
    /// covering their sum. No payload, file, or catalog is inspected here.
    pub fn new<T: Any + Send + Sync>(owner: &'a Arc<T>, bytes: u64) -> Self {
        Self { owner, bytes }
    }

    /// Complete physical-owner bound, including any spare payload capacity.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Existing weak source-storage identity; never retains the payload strongly.
    /// The existing Arc allocation supplies the weak count without a new allocation.
    pub fn identity(&self) -> SourceStorageIdentity {
        self.owner.storage_identity()
    }

    /// Retains this same Arc allocation without copying payload, constructing a
    /// map, or allocating an erasure wrapper. Caller custody survives the source
    /// callback and any later source error/unwind. Final owner Drop is ordinary
    /// payload destruction and must run outside accounting/manager locks.
    pub fn retain(&self) -> SourceStorageOwner {
        SourceStorageOwner {
            _owner: self.owner.retain_owner(),
            bytes: self.bytes,
            control: self.owner.control(),
        }
    }
}

impl std::fmt::Debug for SourceStorageRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceStorageRef")
            .field("identity", &self.identity())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// One retained physical source Arc and its capacity. Cloning retains the same
/// allocation; neither construction through [`SourceStorageRef::retain`] nor
/// cloning allocates a wrapper or copies payload. This grants no byte access,
/// source-health certification, reservation, or publication authority.
#[derive(Clone)]
pub struct SourceStorageOwner {
    _owner: Arc<dyn Any + Send + Sync>,
    bytes: u64,
    // Arc/Weak storage always retires before its independent constructor hold.
    control: Option<SourceControl>,
}

impl SourceStorageOwner {
    /// Complete physical-owner bound supplied by the source.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Weak identity in the same namespace as [`SourceStorage::capacities`].
    pub fn identity(&self) -> SourceStorageIdentity {
        SourceStorageIdentity(Arc::downgrade(&self._owner), self.control.clone())
    }

    /// Borrows these same retained facts without cloning the owner.
    pub fn as_storage_ref(&self) -> SourceStorageRef<'_> {
        SourceStorageRef {
            owner: self,
            bytes: self.bytes,
        }
    }
}

impl std::fmt::Debug for SourceStorageOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceStorageOwner")
            .field("identity", &self.identity())
            .field("bytes", &self.bytes)
            .finish()
    }
}

// Fixed source-constructor custody. Only this concrete wrapper can supply a
// retention callback; providers cannot implement the erased interface. C is
// never extracted. Runtime authenticates its own private C through a borrow.
trait ControlStorage: std::fmt::Debug + Send + Sync {
    fn origin(&self) -> &dyn Any;
    fn retire(self: Arc<Self>);
}
#[derive(Debug)]
struct ControlBody<C>(C);
impl<C: Any + std::fmt::Debug + Send + Sync> ControlStorage for ControlBody<C> {
    fn origin(&self) -> &dyn Any {
        &self.0
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
pub(crate) struct SourceControl(Option<Arc<dyn ControlStorage>>);
impl Clone for SourceControl {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for SourceControl {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
impl SourceControl {
    pub(crate) fn new<C: Any + std::fmt::Debug + Send + Sync>(custody: C) -> Self {
        Self(Some(Arc::new(ControlBody(custody))))
    }
    pub(crate) fn origin<C: Any>(&self) -> Option<&C> {
        self.0
            .as_ref()
            .expect("source control")
            .origin()
            .downcast_ref()
    }
    pub(crate) const fn body_layout<C>() -> std::alloc::Layout {
        std::alloc::Layout::new::<ControlBody<C>>()
    }
    pub(crate) fn owner_control_bytes<C>() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<ControlBody<C>>(),
            size_of::<Option<ControlBody<C>>>(),
            size_of::<Arc<ControlBody<C>>>(),
            size_of::<Arc<dyn ControlStorage>>(),
            size_of::<Option<Arc<dyn ControlStorage>>>(),
            size_of::<&SourceControl>(),
            size_of::<&dyn Any>(),
            size_of::<Option<&C>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
}

/// Crate-private source allocation owner. No Arc/Weak or control is exported.
/// Its alias and identity projections preserve the same original control.
#[derive(Debug)]
pub(crate) struct SourceHandle<T> {
    allocation: Arc<T>,
    control: Option<SourceControl>,
}
impl<T> Clone for SourceHandle<T> {
    fn clone(&self) -> Self {
        Self {
            allocation: self.allocation.clone(),
            control: self.control.clone(),
        }
    }
}
impl<T> std::ops::Deref for SourceHandle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.allocation
    }
}
impl<T: Any + Send + Sync> SourceHandle<T> {
    pub(crate) fn owner_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<Self>(),
            size_of::<Arc<T>>(),
            size_of::<Weak<T>>(),
            size_of::<SourceStorageIdentity>(),
            size_of::<SourceStorageOwner>(),
            size_of::<SourceStorageRef<'static>>(),
            size_of::<Arc<dyn Any + Send + Sync>>(),
            size_of::<&Self>(),
            size_of::<Option<SourceControl>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
    pub(crate) fn new(value: T, control: Option<SourceControl>) -> Self {
        Self {
            allocation: Arc::new(value),
            control,
        }
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
    }
    pub(crate) fn identity(&self) -> SourceStorageIdentity {
        let weak: Weak<T> = Arc::downgrade(&self.allocation);
        SourceStorageIdentity(weak, self.control.clone())
    }
    pub(crate) fn storage_ref(&self, bytes: u64) -> SourceStorageRef<'_> {
        SourceStorageRef { owner: self, bytes }
    }
    pub(crate) fn origin<C: Any>(&self) -> Option<&C> {
        self.control.as_ref()?.origin()
    }
    #[cfg(test)]
    pub(crate) fn ordinary_weak(&self) -> Weak<T> {
        assert!(self.control.is_none(), "funded identities retain control");
        Arc::downgrade(&self.allocation)
    }
    #[cfg(test)]
    pub(crate) fn into_ordinary(self) -> T
    where
        T: std::fmt::Debug,
    {
        assert!(self.control.is_none());
        Arc::try_unwrap(self.allocation).unwrap()
    }
}
/// Erases only source access. The private opaque identity keeps the original
/// allocation and its constructor control through every strong/weak alias.
#[derive(Clone)]
pub(crate) struct CheckpointSourceHandle {
    allocation: Arc<dyn CheckpointSource>,
    identity: SourceStorageIdentity,
}
impl<T: CheckpointSource + 'static> SourceHandle<T> {
    pub(crate) fn into_checkpoint_source(self) -> CheckpointSourceHandle {
        let Self {
            allocation,
            control,
        } = self;
        let weak: Weak<T> = Arc::downgrade(&allocation);
        let identity = SourceStorageIdentity(weak, control);
        CheckpointSourceHandle {
            allocation,
            identity,
        }
    }
}
impl CheckpointSourceHandle {
    pub(crate) fn source(&self) -> &(dyn CheckpointSource + 'static) {
        self.allocation.as_ref()
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
    pub(crate) fn identity(&self) -> SourceStorageIdentity {
        self.identity.clone()
    }
    pub(crate) fn origin<C: Any>(&self) -> Option<&C> {
        self.identity.constructor_control_owner()
    }
    pub(crate) fn erasure_control_bytes<T>() -> Option<usize> {
        use std::mem::size_of;
        let controls = [
            size_of::<Self>(),
            size_of::<Arc<T>>(),
            size_of::<Weak<T>>(),
            size_of::<Arc<dyn CheckpointSource>>(),
            size_of::<Weak<dyn Any + Send + Sync>>(),
            size_of::<SourceStorageIdentity>(),
            size_of::<Option<SourceControl>>(),
            size_of::<&Self>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
}

impl<T: Any + Send + Sync> BorrowedSourceOwner for SourceHandle<T> {
    fn retain_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.allocation.clone()
    }
    fn storage_identity(&self) -> SourceStorageIdentity {
        self.identity()
    }
    fn control(&self) -> Option<SourceControl> {
        self.control.clone()
    }
}
impl BorrowedSourceOwner for SourceStorageOwner {
    fn retain_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self._owner.clone()
    }
    fn storage_identity(&self) -> SourceStorageIdentity {
        self.identity()
    }
    fn control(&self) -> Option<SourceControl> {
        self.control.clone()
    }
}

/// Complete host payload capacity owned by the inspected checkpoint sources.
///
/// Includes encoded memory stores and a source's bounded retained reader
/// buffers. Catalog/recipe metadata, operating-system page cache, externally
/// owned leases and later conversion/materialization workspace are separate.
/// This is a storage bound, not a native allocation or budget reservation.
///
/// Owners are retained so identities cannot be reused while bounds are merged.
/// Restricted logical views therefore cannot discount physically retained bytes.
#[derive(Clone, Default)]
pub struct SourceStorage {
    owners: BTreeMap<usize, SourceStorageOwner>,
}

impl std::fmt::Debug for SourceStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceStorage")
            .field("owners", &self.owners.len())
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl SourceStorage {
    /// Declares one distinct owner's complete host payload capacity. Aliases
    /// must use the same Arc owner and bound; separate payloads need distinct
    /// owners or one owner covering their sum. The owner must keep that payload
    /// alive, or identify an immutable ceiling for source-owned reader storage.
    pub fn insert<T: Any + Send + Sync>(
        &mut self,
        owner: Arc<T>,
        bytes: u64,
    ) -> Result<(), StoreError> {
        self.insert_owner(SourceStorageOwner {
            _owner: owner,
            bytes,
            control: None,
        })
    }

    /// Retain an existing borrowed source with the ordinary shared insertion
    /// worker. This neither copies payload nor certifies constructor custody.
    pub fn include_ref(&mut self, source: SourceStorageRef<'_>) -> Result<(), StoreError> {
        self.insert_owner(source.retain())
    }

    pub(crate) fn insert_retained(&mut self, owner: SourceStorageOwner) -> Result<(), StoreError> {
        self.insert_owner(owner)
    }

    fn insert_owner(&mut self, owner: SourceStorageOwner) -> Result<(), StoreError> {
        let identity = Arc::as_ptr(&owner._owner) as *const () as usize;
        if let Some(existing) = self.owners.get(&identity) {
            if existing.bytes != owner.bytes {
                return Err(StoreError::Internal(
                    "checkpoint storage aliases report different capacity bounds".into(),
                ));
            }
        } else {
            self.owners.insert(identity, owner);
        }
        Ok(())
    }

    /// Joins physical storage, counting each retained owner once across sources.
    pub fn merge(&mut self, other: Self) -> Result<(), StoreError> {
        for owner in other.owners.into_values() {
            self.insert_owner(owner)?;
        }
        Ok(())
    }

    /// Complete sum of unique source-owned host payload capacities.
    pub fn bytes(&self) -> Result<u64, StoreError> {
        self.owners.values().try_fold(0u64, |sum, owner| {
            sum.checked_add(owner.bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: "checkpoint source storage capacity".into(),
                })
        })
    }

    /// Number of distinct physical storage owners in this bound.
    pub fn owner_count(&self) -> usize {
        self.owners.len()
    }

    /// Complete per-owner capacities for shared accounting across inventories.
    /// Identity tokens remain comparable after this inventory retires, without
    /// extending the source value's lifetime. Retain the inventory separately
    /// when the physical payload itself must remain available.
    pub fn capacities(&self) -> impl ExactSizeIterator<Item = (SourceStorageIdentity, u64)> + '_ {
        self.owners
            .values()
            .map(|owner| (owner.identity(), owner.bytes))
    }

    /// Inspects retained sources without payload reads or source reconstruction.
    /// A missing source contract keeps the combined bound unknown.
    pub fn collect<'a>(
        sources: impl IntoIterator<Item = &'a dyn CheckpointSource>,
    ) -> Result<Option<Self>, StoreError> {
        let mut storage = Self::default();
        for source in sources {
            let Some(owned) = source.source_storage()? else {
                return Ok(None);
            };
            storage.merge(owned)?;
        }
        storage.bytes()?;
        Ok(Some(storage))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod visit_tests;
