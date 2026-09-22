//! Actual retained communicator storage, including exact owner identity.
use super::{Group, GroupStorageDomain, GroupStorageKind, GroupStorageUnavailable, NativeGroup};
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
};
/// The native group and all of its fixed owners remain borrowed. Known C++
/// requests do not substitute for missing managed or inherited source obligations.
/// OS-private resources follow the existing native runtime boundary.
pub struct GroupPersistentStorage<'a> {
    group: &'a Group,
    value: safemlx_sys::mlx_distributed_persistent_storage,
}
impl fmt::Debug for GroupPersistentStorage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupPersistentStorage")
            .field("kind", &self.kind())
            .field("unresolved", &self.value.unresolved)
            .finish_non_exhaustive()
    }
}
impl Group {
    /// Query only actual initialized owner fields and the existing loaded
    /// thread-runtime qualifier. No init, queue read, task, polling or allocation.
    pub fn persistent_storage(
        &self,
    ) -> Result<GroupPersistentStorage<'_>, GroupStorageUnavailable> {
        let mut value = safemlx_sys::mlx_distributed_persistent_storage::default();
        // SAFETY: Rc retains actual Group wrapper; native source query is read-only.
        if !unsafe {
            safemlx_sys::mlx_distributed_group_persistent_storage(&mut value, self.native.c_group)
        } {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupPersistentStorage { group: self, value })
    }
    /// Complete fixed source-query frames, payable before persistent_storage.
    pub fn persistent_storage_control_bytes(&self) -> Option<usize> {
        // SAFETY: immutable group, fixed source/ABI controls only.
        let native = unsafe {
            safemlx_sys::mlx_distributed_group_persistent_storage_controls(self.native.c_group)
        };
        let controls = [
            size_of::<GroupPersistentStorage<'_>>(),
            size_of::<&Self>(),
            size_of::<Result<GroupPersistentStorage<'_>, GroupStorageUnavailable>>(),
            size_of::<Layout>() * 3,
            size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
            size_of::<usize>() * 4,
        ];
        controls.into_iter().try_fold(
            native.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )
    }
}
impl GroupPersistentStorage<'_> {
    /// Actual implementation type, independent of any requested backend string.
    pub fn kind(&self) -> GroupStorageKind {
        match self.value.kind {
            1 => GroupStorageKind::Empty,
            2 => GroupStorageKind::Ring,
            _ => GroupStorageKind::Unknown,
        }
    }
    /// Whether an actual retained domain still lacks a complete source.
    pub fn is_unqualified(&self, domain: GroupStorageDomain) -> bool {
        self.value.unresolved & domain as u32 != 0
    }
    /// True while any required retained domain is still unresolved.
    pub fn has_unqualified_storage(&self) -> bool {
        self.value.unresolved != 0
    }
    /// Actual C++ allocator requests, including its C wrapper and shared owner.
    pub fn retained_cpp_bytes(&self) -> usize {
        self.value.retained_cpp_bytes
    }
    /// Same actual owner plus the safe wrapper's Rc allocation. The padded
    /// header/payload layout handles NativeGroup alignment explicitly.
    pub fn retained_owner_bytes(&self) -> Option<usize> {
        let (layout, _) = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<NativeGroup>())
            .ok()?;
        self.value
            .retained_cpp_bytes
            .checked_add(layout.pad_to_align().size())
    }
    /// Native shared allocation including the implementation, not an extra charge.
    pub fn shared_owner_bytes(&self) -> usize {
        self.value.shared_owner_bytes
    }
    /// Actual immutable socket node and bucket allocator requests.
    pub fn socket_map_bytes(&self) -> (usize, usize, usize) {
        (
            self.value.socket_nodes,
            self.value.socket_node_bytes,
            self.value.socket_bucket_bytes,
        )
    }
    /// Retained worker count and qualified C++ thread-runtime requests. OS-private
    /// storage follows the existing runtime boundary; this does not size it.
    pub fn thread_cpp_bytes(&self) -> (usize, usize) {
        (self.value.thread_count, self.value.thread_cpp_runtime_bytes)
    }
    /// Actual retained scratch, socket-vector and thread-vector capacities.
    pub fn vector_bytes(&self) -> (usize, usize, usize) {
        (
            self.value.communication_buffer_bytes,
            self.value.socket_vector_bytes,
            self.value.pool_vector_bytes,
        )
    }
    /// Exact borrowed wrapper identity.
    pub fn is_for(&self, group: &Group) -> bool {
        std::ptr::eq(self.group, group)
    }
    /// Same native implementation across distinct safe/native wrappers.
    pub fn same_implementation(&self, other: &Self) -> bool {
        // SAFETY: both borrowed Group owners remain live; native comparison reads immutable shared targets.
        unsafe {
            safemlx_sys::mlx_distributed_group_same_implementation(
                self.group.native.c_group,
                other.group.native.c_group,
            )
        }
    }
    /// Fixed native query and retained-storage description controls.
    pub fn control_bytes(&self) -> Option<usize> {
        self.value.controls.checked_add(size_of::<Self>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistent_source_counts_actual_shared_owner_and_preserves_native_identity() {
        let first = Group::init(false, super::super::Backend::Ring).unwrap();
        let second = Group::init(false, super::super::Backend::Ring).unwrap();
        let source = first.persistent_storage().unwrap();
        let alias = second.persistent_storage().unwrap();
        assert_eq!(source.kind(), GroupStorageKind::Empty);
        assert!(!source.has_unqualified_storage());
        assert!(source.is_for(&first));
        assert!(!source.is_for(&second));
        assert!(source.same_implementation(&alias));
        let retained = source.retain_buffer().unwrap();
        let aliased = alias.retain_buffer().unwrap();
        assert_eq!(retained.identity(), aliased.identity());
        assert!(retained.is_for(&second));
        assert_eq!(retained.bytes(), 0);
        assert_eq!(source.retained_cpp_bytes(), alias.retained_cpp_bytes());
        assert!(source.shared_owner_bytes() > 0);
        assert!(source.retained_owner_bytes().unwrap() > source.retained_cpp_bytes());
        assert_eq!(source.socket_map_bytes(), (0, 0, 0));
        assert_eq!(source.thread_cpp_bytes(), (0, 0));
        assert_eq!(source.vector_bytes(), (0, 0, 0));
        assert!(first.persistent_storage_control_bytes().unwrap() > 0);
    }
}

/// Process-unique native implementation identity; distinct wrappers of the same
/// shared communicator retain the same key. It grants no access or ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupBufferIdentity(std::num::NonZeroU64);

/// Sealed fixed Ring communication buffer source. The actual native Group keeps
/// the allocation alive; this is neither a tensor nor a mutable payload loan.
#[derive(Clone)]
pub struct RetainedGroupBuffer {
    group: Group,
    identity: GroupBufferIdentity,
    bytes: usize,
}
impl fmt::Debug for RetainedGroupBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedGroupBuffer")
            .field("identity", &self.identity)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
impl RetainedGroupBuffer {
    /// Physical placement of the retained ring transport's native host vector.
    pub fn allocation_placement(&self) -> crate::AllocationPlacement {
        crate::AllocationPlacement::Host
    }
    /// Exact fixed allocation key minted by the native implementation itself.
    pub fn identity(&self) -> GroupBufferIdentity {
        self.identity
    }
    /// Actual retained vector capacity, independent of tensor shapes or requests.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Confirms the same native implementation even through another wrapper.
    pub fn is_for(&self, group: &Group) -> bool {
        // SAFETY: both safe owners keep their immutable native implementations live.
        unsafe {
            safemlx_sys::mlx_distributed_group_same_implementation(
                self.group.native.c_group,
                group.native.c_group,
            )
        }
    }
    /// Alias construction has no native query, allocation, or runtime entry.
    pub const fn clone_control_bytes() -> usize {
        size_of::<Self>()
            + size_of::<Group>()
            + size_of::<GroupBufferIdentity>()
            + size_of::<&Self>()
    }
}
impl GroupPersistentStorage<'_> {
    /// Retain the actual buffer owner only from a completely qualified loaded
    /// implementation. Empty sources retain their zero-sized identity as proof.
    /// The initialized Ring buffer never changes capacity after construction.
    pub fn retain_buffer(&self) -> Result<RetainedGroupBuffer, GroupStorageUnavailable> {
        if self.has_unqualified_storage() {
            return Err(GroupStorageUnavailable);
        }
        let identity = std::num::NonZeroU64::new(self.value.storage_identity)
            .ok_or(GroupStorageUnavailable)?;
        Ok(RetainedGroupBuffer {
            group: self.group.clone(),
            identity: GroupBufferIdentity(identity),
            bytes: self.value.communication_buffer_bytes,
        })
    }
}
