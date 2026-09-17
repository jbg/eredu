//! Fixed canonical-request identity with closed host custody.
use crate::HostPreparationAuthority;
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{atomic::AtomicUsize, Arc},
};
struct Payload {
    _authority: HostPreparationAuthority,
}
/// Clone-only identity for snapshots of one exact request. Identity aliases
/// retain their own shell custody, not the request/model/native payload.
pub struct SpeculativeRequestIdentity(Option<Arc<Payload>>);
impl SpeculativeRequestIdentity {
    /// Ordinary construction, with no managed funding claim.
    pub fn new() -> Self {
        Self::with_authority(HostPreparationAuthority::unmanaged())
    }
    /// Concrete shared shell and constructor/retirement controls. The caller
    /// separately prices the construction of its actual host authority.
    pub fn retained_control_bytes() -> Option<usize> {
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            shared,
            size_of::<Self>(),
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<Arc<Payload>>(),
            size_of::<HostPreparationAuthority>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Allocates this actual shell only after the caller has admitted its query.
    pub fn with_authority(authority: HostPreparationAuthority) -> Self {
        Self(Some(Arc::new(Payload {
            _authority: authority,
        })))
    }
    /// Tests exact identity without exposing a raw shared owner or Weak handle.
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live request identity"),
            other.0.as_ref().expect("live request identity"),
        )
    }
}
impl Default for SpeculativeRequestIdentity {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for SpeculativeRequestIdentity {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live request identity"),
        )))
    }
}
impl fmt::Debug for SpeculativeRequestIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeRequestIdentity")
            .finish_non_exhaustive()
    }
}
impl Drop for SpeculativeRequestIdentity {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
