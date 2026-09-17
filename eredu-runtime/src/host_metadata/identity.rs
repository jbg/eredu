//! Escaping identity custody is distinct from payload-free registry keys.
use crate::working_memory::{resident_reset::ResetCustody, WorkingMemoryError};
use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering as AtomicOrdering},
        Arc,
    },
};

#[derive(Clone)]
enum Key {
    Ordinary(Arc<()>),
    Original(u64),
}
/// Registration identity containing no payload, account, or source backedge.
/// Original-reset keys are non-repeating integers; ordinary keys preserve the
/// existing independent identity allocation. Neither grants access or funding.
#[derive(Clone)]
pub struct HostMetadataKey(Key);
impl HostMetadataKey {
    /// Maximum managed allocation retained by a cloned registry key. Ordinary
    /// keys retain the existing `Arc<()>`; original keys are inline integers.
    /// This prices key lifetime, not a new identity or metadata allocation.
    /// Unsupported managed-layout qualification returns `None`.
    pub fn maximum_clone_storage_bytes() -> Option<u64> {
        crate::working_memory::qualified_shared_bytes::<()>().ok()
    }

    fn value(&self) -> (u8, u64) {
        match &self.0 {
            Key::Ordinary(p) => (0, Arc::as_ptr(p) as usize as u64),
            Key::Original(id) => (1, *id),
        }
    }
}
impl PartialEq for HostMetadataKey {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}
impl Eq for HostMetadataKey {}
impl PartialOrd for HostMetadataKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HostMetadataKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().cmp(&other.value())
    }
}
impl Hash for HostMetadataKey {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.value().hash(h)
    }
}
impl fmt::Debug for HostMetadataKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HostMetadataKey")
            .field(&self.value())
            .finish()
    }
}

enum OriginalIdentityCustody {
    Reset(ResetCustody),
    HostPreparation(eredu_core::HostPreparationAuthority),
    Prepared(crate::working_memory::PreparedInputHostCustody),
}
struct OriginalIdentity {
    _custody: OriginalIdentityCustody,
}
fn next_original_identity() -> Result<u64, WorkingMemoryError> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |n| {
        n.checked_add(1)
    })
    .map_err(|_| WorkingMemoryError::Overflow)
}
/// Owning diagnostic identity of actual metadata. An original-reset identity
/// retains its original allowance until the final identity allocation retires.
/// Use registry_key for accounting keys: storing this owner in its own account
/// registry would create a custody cycle.
pub struct HostMetadataIdentity {
    key: HostMetadataKey,
    original: Option<Arc<OriginalIdentity>>,
}
impl HostMetadataIdentity {
    pub(super) fn ordinary() -> Self {
        Self {
            key: HostMetadataKey(Key::Ordinary(Arc::new(()))),
            original: None,
        }
    }
    pub(crate) fn original_reset(custody: ResetCustody) -> Result<Self, WorkingMemoryError> {
        let id = next_original_identity()?;
        Ok(Self {
            key: HostMetadataKey(Key::Original(id)),
            original: Some(Arc::new(OriginalIdentity {
                _custody: OriginalIdentityCustody::Reset(custody),
            })),
        })
    }
    pub(crate) fn original_prepared(
        custody: crate::working_memory::PreparedInputHostCustody,
    ) -> Result<Self, WorkingMemoryError> {
        let id = next_original_identity()?;
        Ok(Self {
            key: HostMetadataKey(Key::Original(id)),
            original: Some(Arc::new(OriginalIdentity {
                _custody: OriginalIdentityCustody::Prepared(custody),
            })),
        })
    }
    pub(crate) fn prepared_host(
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, WorkingMemoryError> {
        let id = next_original_identity()?;
        Ok(Self {
            key: HostMetadataKey(Key::Original(id)),
            original: Some(Arc::new(OriginalIdentity {
                _custody: OriginalIdentityCustody::HostPreparation(authority.clone()),
            })),
        })
    }
    /// Payload-free key with no reset custody. Original keys do not allocate or
    /// retain the identity control block, and are never reused after retirement.
    pub fn registry_key(&self) -> &HostMetadataKey {
        &self.key
    }
    pub(crate) fn original_control_bytes() -> Option<usize> {
        let allocation = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<OriginalIdentity>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        allocation
            .checked_add(std::mem::size_of::<OriginalIdentity>())?
            .checked_add(std::mem::size_of::<Option<OriginalIdentity>>())?
            .checked_add(std::mem::size_of::<HostMetadataIdentity>())
    }
}
impl Clone for HostMetadataIdentity {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            original: self.original.as_ref().map(Arc::clone),
        }
    }
}
impl Drop for HostMetadataIdentity {
    fn drop(&mut self) {
        // All strong handles use into_inner; no Weak/raw Arc escapes. The
        // returned custody survives deallocation of this identity allocation.
        if let Some(owner) = self.original.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl PartialEq for HostMetadataIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for HostMetadataIdentity {}
impl PartialOrd for HostMetadataIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HostMetadataIdentity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}
impl Hash for HostMetadataIdentity {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.key.hash(h)
    }
}
impl fmt::Debug for HostMetadataIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HostMetadataIdentity")
            .field(&self.key)
            .finish()
    }
}
