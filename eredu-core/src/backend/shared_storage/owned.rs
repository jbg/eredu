//! Closed strong-owner population for concrete shared accounting payloads.
use super::*;
use std::{any::Any, mem::size_of, ops::Deref};
type AnyOwner = Arc<dyn Any + Send + Sync>;

/// Concrete shared-owner retirement supplied by the payload's implementation.
///
/// Providers whose original custody covers the Arc allocation must route every
/// strong exit through `Arc::into_inner`, then drop its returned concrete value.
/// They must not retain or export Weak owners of that allocation. This contract
/// does not itself establish funding, completion, or correct provider custody.
pub trait SharedStorageRetirement: Any + Send + Sync {
    /// Consumes one strong owner. A strict provider frees its Arc allocation
    /// before the returned payload can release its last accounting custody.
    fn retire(self: Arc<Self>);
}

/// Typed immutable owner with closed concrete retirement on every strong exit.
/// No existing Arc constructor, raw owning export, mutation or Weak API exists.
pub struct SharedStorageOwner<T: SharedStorageRetirement>(Option<Arc<T>>);
impl<T: SharedStorageRetirement> SharedStorageOwner<T> {
    /// Allocates one Arc for the supplied concrete payload. Its caller must
    /// establish any original admission before construction.
    pub fn new(value: T) -> Self {
        Self(Some(Arc::new(value)))
    }
    /// Whether both handles retain the exact same allocation.
    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live owner"),
            other.0.as_ref().expect("live owner"),
        )
    }
    /// Erases the same owner without allocating or permitting raw Arc escape.
    pub fn erase(mut self) -> ErasedSharedStorageOwner {
        ErasedSharedStorageOwner {
            owner: self.0.take().map(|owner| owner as AnyOwner),
            retire: retire_erased::<T>,
        }
    }
}
impl<T: SharedStorageRetirement> Clone for SharedStorageOwner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T: SharedStorageRetirement> Deref for SharedStorageOwner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_deref().expect("live owner")
    }
}
impl<T: SharedStorageRetirement> fmt::Debug for SharedStorageOwner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedStorageOwner").finish_non_exhaustive()
    }
}
impl<T: SharedStorageRetirement> Drop for SharedStorageOwner<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}

/// Erased alias of the same closed shared owner, with borrowed type inspection.
/// Its concrete retirement function is fixed by core when erasing the owner.
pub struct ErasedSharedStorageOwner {
    owner: Option<AnyOwner>,
    retire: fn(AnyOwner),
}
impl ErasedSharedStorageOwner {
    /// Borrows the original concrete payload without creating an owning escape.
    pub fn downcast_ref<T: SharedStorageRetirement>(&self) -> Option<&T> {
        self.owner.as_ref().expect("live owner").downcast_ref::<T>()
    }
    pub(super) fn clone_typed<T: SharedStorageRetirement>(&self) -> Option<SharedStorageOwner<T>> {
        self.downcast_ref::<T>()?;
        let owner = self
            .owner
            .as_ref()
            .expect("live owner")
            .clone()
            .downcast::<T>()
            .unwrap_or_else(|_| unreachable!("checked concrete owner"));
        Some(SharedStorageOwner(Some(owner)))
    }
}
impl Clone for ErasedSharedStorageOwner {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            retire: self.retire,
        }
    }
}
impl fmt::Debug for ErasedSharedStorageOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErasedSharedStorageOwner")
            .finish_non_exhaustive()
    }
}
impl Drop for ErasedSharedStorageOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            (self.retire)(owner);
        }
    }
}
fn retire_erased<T: SharedStorageRetirement>(owner: AnyOwner) {
    let owner = owner
        .downcast::<T>()
        .unwrap_or_else(|_| unreachable!("original concrete retirement"));
    owner.retire();
}

// Actual attachment element plus typed/erased construction, lookup and concrete
// retirement controls. Arc payload/header allocation is priced by its provider.
pub(super) fn attachment_control_bytes<T: SharedStorageRetirement, E>() -> Option<usize> {
    [
        attachments::node_bytes(),
        attachments::maximum_insertion_controls::<T, E>()?,
        size_of::<SharedStorageOwner<T>>(),
        size_of::<ErasedSharedStorageOwner>(),
        size_of::<Option<T>>(),
        size_of::<Arc<T>>(),
        size_of::<AnyOwner>(),
        size_of::<Result<Arc<T>, AnyOwner>>(),
        size_of::<Option<AnyOwner>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests;
