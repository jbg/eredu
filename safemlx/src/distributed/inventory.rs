//! Exact retained native implementation fields, with unresolved storage explicit.
use super::Group;
use std::{fmt, mem::size_of};

/// Actual implementation type, independent of the backend requested at init.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupStorageKind {
    /// The implementation has no qualified type inventory.
    Unknown,
    /// The actual singleton fallback; no Ring transport was created.
    Empty,
    /// Native socket Ring implementation.
    Ring,
    /// Native MPI implementation.
    Mpi,
    /// Native JACCL implementation.
    Jaccl,
    /// Native NCCL implementation.
    Nccl,
}

/// Storage whose actual producer has not supplied a complete native bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GroupStorageDomain {
    /// Shared-pointer control allocations remain unqualified.
    SharedControl = 1,
    /// Associative container bucket/node storage remains unqualified.
    AssociativeStorage = 2,
    /// Worker thread and operating-system runtime storage remains unqualified.
    WorkerRuntime = 4,
    /// Submission task, queue and result owners remain unqualified.
    SubmissionStorage = 8,
    /// Backend-private resources remain unqualified.
    BackendPrivate = 16,
    /// A separate terminal ancestor remains retained by this implementation.
    InheritedOwner = 32,
    /// The actual implementation does not supply the inventory contract.
    UnknownImplementation = 64,
}

/// The retained native handle cannot supply even a partial immutable inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("native communicator storage inventory is unavailable")]
pub struct GroupStorageUnavailable;

/// A read-only inventory borrowed from the exact native owner. The known fields
/// are not a total allocation bound, queue capacity, or submission authority.
pub struct GroupStorageInventory<'group> {
    group: &'group Group,
    value: safemlx_sys::mlx_distributed_storage_inventory,
}
impl fmt::Debug for GroupStorageInventory<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupStorageInventory").field("kind", &self.kind())
            .field("unresolved_domains", &self.value.unresolved).finish_non_exhaustive()
    }
}
impl Group {
    /// No allocation, runtime lock, backend initialization, rank query, error
    /// formatting, polling, or work submission occurs. The immutable owner stays
    /// borrowed for the complete lifetime of the returned inventory.
    pub fn storage_inventory(&self) -> Result<GroupStorageInventory<'_>, GroupStorageUnavailable> {
        let mut value = safemlx_sys::mlx_distributed_storage_inventory {
            kind: 0, unresolved: 0, wrapper_bytes: 0, implementation_bytes: 0,
            socket_handles: 0, socket_capacity_bytes: 0, buffer_bytes: 0,
            pool_workers: 0, pool_worker_capacity_bytes: 0,
            socket_workers: 0, socket_map_buckets: 0,
        };
        // SAFETY: the Rc owns the initialized wrapper; output is a live fixed
        // struct. The shim reads only immutable construction-time fields.
        if !unsafe { safemlx_sys::mlx_distributed_group_storage_inventory(&mut value, self.native.c_group) } {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupStorageInventory { group: self, value })
    }

    /// Exact named frame/control bytes; no communicator storage is credited.
    pub fn storage_inventory_control_bytes() -> Option<usize> {
        // SAFETY: pure fixed-layout query has no native inputs or side effects.
        let native = unsafe { safemlx_sys::mlx_distributed_group_storage_inventory_controls() };
        let controls = [size_of::<GroupStorageInventory<'_>>(), size_of::<Result<GroupStorageInventory<'_>, GroupStorageUnavailable>>(),
            size_of::<(&Self, &Self)>(), size_of::<bool>(), size_of::<GroupStorageUnavailable>()];
        controls.into_iter().try_fold(native.checked_add(std::mem::size_of_val(&controls))?, usize::checked_add)
    }
}
impl GroupStorageInventory<'_> {
    /// Actual native implementation type, independent of the requested backend.
    pub fn kind(&self) -> GroupStorageKind {
        match self.value.kind { 1=>GroupStorageKind::Empty, 2=>GroupStorageKind::Ring,
            3=>GroupStorageKind::Mpi, 4=>GroupStorageKind::Jaccl, 5=>GroupStorageKind::Nccl,
            _=>GroupStorageKind::Unknown }
    }
    /// Whether the actual source leaves this storage domain unqualified.
    pub fn is_unqualified(&self, domain: GroupStorageDomain) -> bool {
        self.value.unresolved & domain as u32 != 0
    }
    /// Whether any source storage domain still lacks a qualified bound.
    pub fn has_unqualified_storage(&self) -> bool { self.value.unresolved != 0 }
    /// Native C++ wrapper object only; excludes its allocator's bookkeeping.
    pub fn wrapper_bytes(&self) -> usize { self.value.wrapper_bytes }
    /// Inline implementation fields only; excludes shared-control allocation.
    pub fn implementation_bytes(&self) -> usize { self.value.implementation_bytes }
    /// Number of native socket handles retained by the source.
    pub fn socket_handles(&self) -> usize { self.value.socket_handles }
    /// Payload capacity of the actual immutable socket vectors.
    pub fn socket_capacity_bytes(&self) -> usize { self.value.socket_capacity_bytes }
    /// Payload capacity of the actual retained communication buffer vector.
    pub fn buffer_bytes(&self) -> usize { self.value.buffer_bytes }
    /// Number of retained pool workers, excluding their runtime allocation size.
    pub fn pool_workers(&self) -> usize { self.value.pool_workers }
    /// Actual std::thread vector payload, excluding thread runtime storage.
    pub fn pool_worker_capacity_bytes(&self) -> usize { self.value.pool_worker_capacity_bytes }
    /// Number of retained socket workers, excluding their runtime allocation size.
    pub fn socket_workers(&self) -> usize { self.value.socket_workers }
    /// Logical bucket count only; no guessed std-library bucket/node layout.
    pub fn socket_map_buckets(&self) -> usize { self.value.socket_map_buckets }
    /// Whether both loans refer to the same exact native implementation.
    pub fn same_implementation(&self, other: &Self) -> bool {
        // SAFETY: both inventories retain a borrow of their live native owner;
        // comparison reads only immutable shared-pointer targets, not handles.
        unsafe { safemlx_sys::mlx_distributed_group_same_implementation(self.group.native.c_group, other.group.native.c_group) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_inventory_identifies_actual_fallback_and_shared_native_incarnation() {
        let first = Group::init(false, super::super::Backend::Ring).unwrap();
        let second = Group::init(false, super::super::Backend::Ring).unwrap();
        assert!(!first.shares_native_handle(&second));
        let inventory = first.storage_inventory().unwrap();
        let other = second.storage_inventory().unwrap();
        // This isolated singleton fixture requests Ring but retains EmptyGroup.
        assert_eq!(inventory.kind(), GroupStorageKind::Empty);
        assert!(inventory.same_implementation(&other));
        assert!(inventory.wrapper_bytes() > 0 && inventory.implementation_bytes() > 0);
        assert_eq!((inventory.socket_handles(), inventory.buffer_bytes(), inventory.pool_workers(), inventory.socket_workers()), (0,0,0,0));
        assert!(inventory.is_unqualified(GroupStorageDomain::SharedControl));
        assert!(inventory.has_unqualified_storage());
        assert!(!inventory.is_unqualified(GroupStorageDomain::SubmissionStorage));
    }
}
