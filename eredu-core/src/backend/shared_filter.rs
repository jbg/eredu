//! Immutable filter payloads with independent, per-domain accounting custody.

use super::{
    shared_storage::SharedStorageCustody, SharedStorageAttachmentError, SharedStorageDomain,
    SharedStorageIdentity, TokenFilter,
};
use std::{fmt, ops::Deref, sync::Arc};

struct Inner {
    filter: TokenFilter,
    custody: SharedStorageCustody,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Payload retirement must precede the last physical accounting refund.
        // Replacing with All performs no allocation and leaves no mask to be
        // destroyed later by automatic field drop.
        drop(std::mem::replace(&mut self.filter, TokenFilter::All));
        // The shared custody field retires next, without acquiring its mutex.
    }
}

/// One immutable filter payload shared across controllers, decisions and clones.
///
/// Accounting attachments belong to the shared owner, so even clones created
/// before attachment preserve coverage until the final alias retires. Values
/// compare by filter contents; storage identity is available separately.
#[derive(Clone)]
pub struct SharedTokenFilter(Arc<Inner>);

impl SharedTokenFilter {
    /// Owns the supplied filter without copying or shrinking its allocation.
    pub fn new(filter: TokenFilter) -> Self {
        Self(Arc::new(Inner {
            filter,
            custody: SharedStorageCustody::new(),
        }))
    }

    /// Borrows the immutable filter without copying its payload.
    pub fn as_ref(&self) -> &TokenFilter {
        &self.0.filter
    }

    /// Exact owner identity, independent of its filter contents.
    pub fn identity(&self) -> &SharedStorageIdentity {
        self.0.custody.identity()
    }

    /// Retained mask allocation capacity, including spare elements.
    ///
    /// `All` owns no numerical payload. Identity and custody metadata are not
    /// numerical storage. Conversion or arithmetic overflow remains unknown.
    pub fn capacity_bytes(&self) -> Option<u64> {
        match self.as_ref() {
            TokenFilter::All => Some(0),
            TokenFilter::Allowed(mask) => u64::try_from(mask.capacity())
                .ok()?
                .checked_mul(std::mem::size_of::<bool>() as u64),
        }
    }

    /// Whether two handles share this exact filter allocation owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Checks the existing attachment without allocating, acquiring new custody
    /// or invoking a provider. The accounting domain must independently verify
    /// its canonical registration and exact capacity. This is not a grant.
    pub fn has_accounting_custody(&self,domain:&SharedStorageDomain)
        ->Result<bool,SharedStorageAttachmentError<std::convert::Infallible>> {
        self.0.custody.has_accounting_custody(domain)
    }

    /// Acquires accounting custody once per exact domain.
    ///
    /// Returns `false` without calling `acquire` when the domain is already
    /// attached, or `true` after publishing its new handle. The vector slot is
    /// reserved before acquisition; publishing the returned handle cannot fail.
    /// A rejected acquisition leaves all earlier attachments intact.
    ///
    /// The provider runs under the custody lock and must perform only closed
    /// accounting operations: no owner reentry, native work or user callbacks.
    /// Its returned handle must not retain this shared owner, directly or
    /// indirectly. Use the payload-free identity/domain keys for registration
    /// to avoid an owner-registration cycle. Provider handles are never dropped
    /// under the custody lock and outlive the filter's numerical payload.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.custody.try_attach(domain, acquire)
    }
}

impl AsRef<TokenFilter> for SharedTokenFilter {
    fn as_ref(&self) -> &TokenFilter {
        SharedTokenFilter::as_ref(self)
    }
}
impl Deref for SharedTokenFilter {
    type Target = TokenFilter;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
impl PartialEq for SharedTokenFilter {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl Eq for SharedTokenFilter {}
impl fmt::Debug for SharedTokenFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedTokenFilter")
            .field("filter", self.as_ref())
            .field("identity", self.identity())
            .finish()
    }
}

#[cfg(test)]
mod tests;
