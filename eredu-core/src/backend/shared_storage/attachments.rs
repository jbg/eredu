//! One exact node per accounting owner, funded before node and owner construction.
use super::*;
use eredu_collections::ordered_map::{Map, TryInsertError};
#[cfg(test)]
use std::sync::PoisonError;
use std::{
    alloc::Layout,
    mem::{replace, size_of},
    sync::{MutexGuard, TryLockError},
};
type Entries = Map<SharedStorageAccountingId, Option<AttachmentCustody>>;

/// Exact new attachment node and reached insertion controls. The provider adds
/// its concrete owner's allocation/control layout before admitting construction.
#[derive(Clone, Copy, Debug)]
pub struct SharedStorageAttachmentLayout {
    node: Layout,
    controls: usize,
}
impl SharedStorageAttachmentLayout {
    /// Allocation requested by the single new ordered-map node.
    pub fn node_layout(self) -> Layout {
        self.node
    }
    /// Node backing and the reached lookup/insertion/retirement controls.
    pub fn requested_bytes(self) -> usize {
        self.controls
    }
}

/// Attachment nodes for a source that supplies its own exclusive access.
///
/// Construction allocates no backing. A provider must admit the supplied node
/// layout and its own concrete owner before returning that owner. Reuse never
/// calls the provider. Final retirement frees each node and identity key before
/// its attached custody. Providers must not retain the source itself.
pub struct SharedStorageAttachmentTable {
    entries: Entries,
}
impl Default for SharedStorageAttachmentTable {
    fn default() -> Self {
        Self::new()
    }
}
impl SharedStorageAttachmentTable {
    /// Empty source controls with no attached node.
    pub fn new() -> Self {
        Self {
            entries: Entries::new(),
        }
    }
    /// One node and addressable insertion/retirement controls, excluding the
    /// concrete owner allocation and source/lock controls.
    pub fn owned_attachment_control_bytes<T: SharedStorageRetirement, E>() -> Option<usize> {
        owned::attachment_control_bytes::<T, E>()
    }

    /// Whether this exact accounting owner already retains the source.
    pub fn has_accounting_custody(&self, owner: &SharedStorageAccountingId) -> bool {
        self.entries.contains_key(owner)
    }
    /// Attaches opaque custody after prospective node and owner admission.
    pub fn try_attach<E>(
        &mut self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        if self.entries.contains_key(owner) {
            return Ok(false);
        }
        insert(&mut self.entries, owner, |layout| {
            Ok((AttachmentCustody::Opaque(acquire(layout)?), true))
        })
    }
    /// Retrieves the same typed owner or admits and attaches a new one.
    pub fn try_attach_typed<T: Send + Sync + 'static, E>(
        &mut self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, SharedStorageAttachmentError<E>> {
        if let Some(entry) = self.entries.get(owner) {
            let Some(AttachmentCustody::Typed(prior)) = entry else {
                return Err(SharedStorageAttachmentError::AttachmentMismatch);
            };
            return prior
                .clone()
                .downcast::<T>()
                .map_err(|_| SharedStorageAttachmentError::AttachmentMismatch);
        }
        insert(&mut self.entries, owner, |layout| {
            let value = acquire(layout)?;
            Ok((AttachmentCustody::Typed(value.clone()), value))
        })
    }
    /// Retrieves a closed owner or admits its node before constructing it.
    pub fn try_attach_owned<T: SharedStorageRetirement, E>(
        &mut self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<SharedStorageOwner<T>, E>,
    ) -> Result<SharedStorageOwner<T>, SharedStorageAttachmentError<E>> {
        if let Some(entry) = self.entries.get(owner) {
            let Some(AttachmentCustody::Owned(prior)) = entry else {
                return Err(SharedStorageAttachmentError::AttachmentMismatch);
            };
            return prior
                .clone_typed::<T>()
                .ok_or(SharedStorageAttachmentError::AttachmentMismatch);
        }
        insert(&mut self.entries, owner, |layout| {
            let value = acquire(layout)?;
            Ok((AttachmentCustody::Owned(value.clone().erase()), value))
        })
    }
}
impl Drop for SharedStorageAttachmentTable {
    fn drop(&mut self) {
        for (key, owner) in replace(&mut self.entries, Entries::new()) {
            drop(key);
            drop(owner);
        }
    }
}

/// Per-accounting-owner custody with allocation-free lookup and reuse.
///
/// Providers run under this source's lock and must perform only closed
/// accounting operations: no source reentry, native work, or user callbacks.
/// Errors and unused closures retire after releasing the lock. The source
/// constructor owns its mutex controls; providers own prospective node backing.
pub struct SharedStorageAttachments {
    entries: Mutex<SharedStorageAttachmentTable>,
}
impl Default for SharedStorageAttachments {
    fn default() -> Self {
        Self::new()
    }
}
impl SharedStorageAttachments {
    /// Empty caller-funded source control. No attachment node is allocated.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(SharedStorageAttachmentTable::new()),
        }
    }
    #[cfg(test)]
    pub(crate) fn lock(
        &self,
    ) -> Result<
        MutexGuard<'_, SharedStorageAttachmentTable>,
        PoisonError<MutexGuard<'_, SharedStorageAttachmentTable>>,
    > {
        self.entries.lock()
    }
    #[cfg(test)]
    pub(crate) fn try_lock(
        &self,
    ) -> Result<
        MutexGuard<'_, SharedStorageAttachmentTable>,
        TryLockError<MutexGuard<'_, SharedStorageAttachmentTable>>,
    > {
        self.entries.try_lock()
    }
    /// Whether this exact accounting owner already retains the source.
    pub fn has_accounting_custody(
        &self,
        owner: &SharedStorageAccountingId,
    ) -> Result<bool, SharedStorageAttachmentError<std::convert::Infallible>> {
        Ok(self
            .entries
            .try_lock()
            .map_err(lock_error)?
            .has_accounting_custody(owner))
    }
    /// Blocking opaque attachment with prospective node admission.
    pub fn try_attach<E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.attach_opaque(owner, acquire, false)
    }
    /// Nonblocking opaque attachment with prospective node admission.
    pub fn try_attach_nonblocking<E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.attach_opaque(owner, acquire, true)
    }
    fn attach_opaque<E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Box<dyn Send + Sync>, E>,
        nonblocking: bool,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        let mut acquire = Some(acquire);
        let result = (|| {
            let mut entries = if nonblocking {
                self.entries.try_lock().map_err(lock_error)?
            } else {
                self.entries
                    .lock()
                    .map_err(|_| SharedStorageAttachmentError::Poisoned)?
            };
            entries.try_attach(owner, |layout| {
                acquire.take().expect("single provider")(layout)
            })
        })();
        drop(acquire);
        result
    }
    /// Retrieves the same typed owner or invokes the paid provider once.
    pub fn try_attach_typed_nonblocking<T: Send + Sync + 'static, E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, SharedStorageAttachmentError<E>> {
        let mut acquire = Some(acquire);
        let result = (|| {
            let mut entries = self.entries.try_lock().map_err(lock_error)?;
            entries.try_attach_typed(owner, |layout| {
                acquire.take().expect("single provider")(layout)
            })
        })();
        drop(acquire);
        result
    }
    /// Typed attachment whose final strong exit uses concrete retirement.
    pub fn try_attach_owned_nonblocking<T: SharedStorageRetirement, E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<SharedStorageOwner<T>, E>,
    ) -> Result<SharedStorageOwner<T>, SharedStorageAttachmentError<E>> {
        let mut acquire = Some(acquire);
        let result = (|| {
            let mut entries = self.entries.try_lock().map_err(lock_error)?;
            entries.try_attach_owned(owner, |layout| {
                acquire.take().expect("single provider")(layout)
            })
        })();
        drop(acquire);
        result
    }
}
fn lock_error<E>(
    error: TryLockError<MutexGuard<'_, SharedStorageAttachmentTable>>,
) -> SharedStorageAttachmentError<E> {
    match error {
        TryLockError::WouldBlock => SharedStorageAttachmentError::Busy,
        TryLockError::Poisoned(_) => SharedStorageAttachmentError::Poisoned,
    }
}
fn insert<O, E>(
    entries: &mut Entries,
    owner: &SharedStorageAccountingId,
    acquire: impl FnOnce(SharedStorageAttachmentLayout) -> Result<(AttachmentCustody, O), E>,
) -> Result<O, SharedStorageAttachmentError<E>> {
    let controls = entries
        .insertion_control_bytes(owner)
        .and_then(|n| {
            n.checked_add(size_of::<(
                Option<AttachmentCustody>,
                Option<O>,
                SharedStorageAttachmentLayout,
                Result<O, SharedStorageAttachmentError<E>>,
            )>())
        })
        .ok_or(SharedStorageAttachmentError::Overflow)?;
    let mut acquired = None;
    entries
        .try_insert_with(owner.clone(), None, |node| {
            let controls = controls
                .checked_add(node.size())
                .ok_or(SharedStorageAttachmentError::Overflow)?;
            acquired = Some(
                acquire(SharedStorageAttachmentLayout { node, controls })
                    .map_err(SharedStorageAttachmentError::Provider)?,
            );
            Ok(())
        })
        .map_err(|error| match error {
            TryInsertError::Funding(error) => error,
            TryInsertError::SizeOverflow => SharedStorageAttachmentError::Overflow,
        })?;
    let (custody, result) = acquired.expect("new node acquired its owner");
    *entries.get_mut(owner).expect("inserted node") = Some(custody);
    Ok(result)
}
pub(super) fn node_bytes() -> usize {
    Entries::node_allocation_layout().size()
}
pub(super) fn maximum_insertion_controls<T: SharedStorageRetirement, E>() -> Option<usize> {
    Entries::maximum_insertion_control_bytes()?.checked_add(size_of::<(
        Option<AttachmentCustody>,
        Option<SharedStorageOwner<T>>,
        SharedStorageAttachmentLayout,
        Result<SharedStorageOwner<T>, SharedStorageAttachmentError<E>>,
    )>())
}

#[cfg(test)]
mod tests;
