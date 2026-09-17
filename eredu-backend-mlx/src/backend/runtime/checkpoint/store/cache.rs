//! One weak cache for ordinary and admitted physical GGUF groups.
use super::*;
use eredu_checkpoint::{
    gguf_store::{GgufCacheIdentityView, GgufLeaseIdentity, StoredGgufCacheIdentity},
    prepared_index::{PreparedIndex, PreparedIndexNode},
};
use eredu_gguf::{InitializedStorage, StorageFamily, StorageProvider, SuppliedStorageError};
use eredu_runtime::working_memory::{
    HostDestinationCause, OriginalHostDestinationBank, OriginalHostMetadataCustody,
    OriginalHostMetadataVec, OriginalHostSourceBank, WorkingMemoryError,
};
use std::{alloc::Layout, cmp::Ordering, ops::Deref};

#[derive(Debug)]
pub(super) struct MetadataFamily;
#[derive(Debug)]
pub(super) struct MetadataBuffer<T: Copy>(OriginalHostMetadataVec<T>);
impl<T: Copy> AsRef<[T]> for MetadataBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}
impl<T: Copy> AsMut<[T]> for MetadataBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}
impl<T: Copy + std::fmt::Debug + 'static> InitializedStorage<T> for MetadataBuffer<T> {
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
}
impl StorageFamily for MetadataFamily {
    type Buffer<T: Copy + std::fmt::Debug + 'static> = MetadataBuffer<T>;
    type Error = HostDestinationCause;
}
struct Provider<'a>(&'a mut OriginalHostDestinationBank);
impl StorageProvider for Provider<'_> {
    type Family = MetadataFamily;
    fn prepare<T: Copy + std::fmt::Debug + 'static>(
        &mut self,
        elements: usize,
        initializer: T,
    ) -> Result<MetadataBuffer<T>, (HostDestinationCause, Option<MetadataBuffer<T>>)> {
        let mut values = match self.0.try_vec(elements) {
            Ok(values) => values,
            Err(error) => {
                let (cause, values) = error.into_parts();
                return Err((cause, values.map(|v| MetadataBuffer(v.into_metadata()))));
            }
        };
        if let Err(cause) = values.try_fill(elements, std::iter::repeat(initializer).take(elements))
        {
            return Err((cause, Some(MetadataBuffer(values.into_metadata()))));
        }
        Ok(MetadataBuffer(values.into_metadata()))
    }
}
#[derive(Debug)]
pub(super) enum Key {
    Ordinary(GgufLeaseIdentity),
    Supplied(StoredGgufCacheIdentity<MetadataFamily>),
}
impl Key {
    pub(super) fn view(&self) -> GgufCacheIdentityView<'_> {
        match self {
            Self::Ordinary(v) => v.cache_view(),
            Self::Supplied(v) => v.cache_view(),
        }
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.view() == other.view()
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.view().cmp(&other.view())
    }
}

/// Shared block first, accounting-only custody last: a stale weak alias may
/// keep the physical Arc allocation alive after its arrays/names retire.
#[derive(Clone, Debug)]
pub(super) struct Group {
    value: Arc<CachedGgufGroup>,
    custody: Option<OriginalHostMetadataCustody>,
}
impl Deref for Group {
    type Target = CachedGgufGroup;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl Group {
    pub(super) fn ordinary(arrays: CachedGgufArrays) -> Self {
        Self {
            value: Arc::new(CachedGgufGroup { arrays }),
            custody: None,
        }
    }
    #[cfg(test)]
    pub(super) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
    pub(super) fn downgrade(&self) -> WeakGroup {
        WeakGroup {
            value: Arc::downgrade(&self.value),
            _custody: self.custody.clone(),
        }
    }
}
#[derive(Debug)]
pub(super) struct WeakGroup {
    value: Weak<CachedGgufGroup>,
    _custody: Option<OriginalHostMetadataCustody>,
}
impl WeakGroup {
    pub(super) fn stale(&self) -> bool {
        self.value.strong_count() == 0
    }
    pub(super) fn upgrade(&self) -> Option<Group> {
        self.value.upgrade().map(|value| Group {
            value,
            custody: self._custody.clone(),
        })
    }
}
pub(super) type Index = PreparedIndex<Key, WeakGroup, Option<OriginalHostMetadataCustody>>;
pub(super) type Node = PreparedIndexNode<Key, WeakGroup, Option<OriginalHostMetadataCustody>>;

/// Actual successful/failed current-miss prefix; the enclosing pending recovery
/// keeps this across producer/commit refusal. No source/manager is in raw custody.
pub(super) struct PreparedEntry {
    key: Option<Key>,
    failed_key: Option<StoredGgufCacheIdentity<MetadataFamily>>,
    arrays: Option<CachedGgufArrays>,
    group: Option<Group>,
    pub(super) node: Option<Node>,
    custody: OriginalHostMetadataCustody,
}
impl PreparedEntry {
    pub(super) fn prepare(
        identity: &GgufLeaseIdentity,
        destinations: &mut OriginalHostDestinationBank,
        source: &mut OriginalHostSourceBank,
    ) -> Result<Self, (GgufHostCopyCause, Option<Self>)> {
        let bytes =
            control_bytes().map_err(|cause| (GgufHostCopyCause::SourcePublication(cause), None))?;
        let custody = source
            .try_debit(bytes)
            .map_err(|cause| (GgufHostCopyCause::SourceFunding(cause), None))?
            .into_metadata();
        let mut out = Self {
            key: None,
            failed_key: None,
            arrays: None,
            group: None,
            node: None,
            custody,
        };
        match StoredGgufCacheIdentity::prepare(identity, &mut Provider(destinations)) {
            Ok(key) => {
                out.key = Some(Key::Supplied(key));
                Ok(out)
            }
            Err(error) => {
                let (cause, key) = error.into_parts();
                out.failed_key = Some(key);
                Err((GgufHostCopyCause::CacheStorage(cause), Some(out)))
            }
        }
    }
    /// Called only after successful conversion; the entire named allocation
    /// contribution was debited before key construction. No fallible/provider
    /// observation occurs while moving the arrays into these two owners.
    pub(super) fn complete(&mut self, arrays: CachedGgufArrays) {
        self.arrays = Some(arrays);
        let group = Group {
            value: Arc::new(CachedGgufGroup {
                arrays: self.arrays.take().expect("current group"),
            }),
            custody: Some(self.custody.clone()),
        };
        self.group = Some(group);
        self.node = Some(Node::new(
            self.key.take().expect("prepared key"),
            self.group.as_ref().expect("current group").downgrade(),
            Some(self.custody.clone()),
        ));
    }
    #[cfg(test)]
    pub(super) fn retained_group(&self) -> Option<&Group> {
        self.group.as_ref()
    }
    pub(super) fn take_group(&mut self) -> Group {
        self.group.take().expect("completed group")
    }
}
/// New per-miss owners only; key buffer extents are separately in destination
/// facts. The shared cache context and weak source shell are existing owners.
pub(super) fn control_bytes() -> Result<u64, WorkingMemoryError> {
    let node = OriginalHostMetadataCustody::boxed_storage_bytes(Node::storage_layout())?;
    let group =
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<CachedGgufGroup>())?;
    let fixed = [
        std::mem::size_of::<Provider<'static>>(),
        StoredGgufCacheIdentity::<MetadataFamily>::preparation_control_bytes()
            .ok_or(WorkingMemoryError::Overflow)?,
        std::mem::size_of::<PreparedEntry>(),
        std::mem::size_of::<Result<PreparedEntry, (GgufHostCopyCause, Option<PreparedEntry>)>>(),
        std::mem::size_of::<Node>(),
        std::mem::size_of::<Group>(),
        std::mem::size_of::<Option<Group>>(),
        std::mem::size_of::<Option<Node>>(),
        std::mem::size_of::<Result<Option<Node>, ()>>(),
        std::mem::size_of::<Result<(Group, Option<Node>), PreparedGgufHostCopyFailure>>(),
        std::mem::size_of::<eredu_gguf::StorageRequestBound>(),
        std::mem::size_of::<SuppliedStorageError<HostDestinationCause>>(),
        std::mem::size_of::<
            eredu_checkpoint::gguf_store::StoredGgufCacheIdentityFailure<MetadataFamily>,
        >(),
        Index::worker_control_layout()
            .ok_or(WorkingMemoryError::Overflow)?
            .size(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)?;
    node.checked_add(group)
        .and_then(|n| n.checked_add(u64::try_from(fixed).ok()?))
        .ok_or(WorkingMemoryError::Overflow)
}
/// Shared existing context requested layout, with exactly one PAL initialization
/// before the context is shared. This is never added once per cache miss.
pub(super) fn context_storage_bytes() -> Result<u64, WorkingMemoryError> {
    super::CacheHandle::retained_storage_bytes()
}

#[cfg(test)]
thread_local! { static AFTER_SWEEP: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = std::cell::RefCell::new(None); }
#[cfg(test)]
pub(super) fn after_sweep_once(action: impl FnOnce() + 'static) {
    AFTER_SWEEP.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some(Box::new(action));
    });
}
#[cfg(test)]
pub(super) fn after_sweep() {
    let action = AFTER_SWEEP.with(|slot| slot.borrow_mut().take());
    if let Some(action) = action {
        action();
    }
}
