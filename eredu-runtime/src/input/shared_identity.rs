use super::PreparedInputCacheIdentity;
use crate::host_metadata::{HostMetadataIdentity, MetadataCustody};
use eredu_core::{PreparedInputIdentity, SharedStorageAccountingId, SharedStorageAttachmentError};
use std::{fmt, sync::Arc};

/// Shared immutable prepared-input/cache description with exact payload custody.
///
/// Cloning shares its existing identity, descriptions and fingerprints; no tensor
/// is retained and no content is re-encoded or hashed. Equality compares semantic
/// values, while `identity`/`same_storage` distinguish physical ownership.
pub struct SharedPreparedInputCacheIdentity(Option<Arc<Inner>>);

struct Inner {
    // Payload must retire before any attached accounting owner.
    payload: PreparedInputCacheIdentity,
    custody: MetadataCustody,
}

impl SharedPreparedInputCacheIdentity {
    fn inner(&self) -> &Inner {
        self.0.as_deref().expect("live prepared cache identity")
    }
    pub(crate) fn new_original(
        payload: PreparedInputCacheIdentity,
        custody: crate::working_memory::PreparedInputHostCustody,
        source: crate::working_memory::OriginalPreparedHostInput,
    ) -> Result<Self, crate::working_memory::WorkingMemoryError> {
        let custody = MetadataCustody::original_prepared(custody, source)?;
        Ok(Self(Some(Arc::new(Inner { payload, custody }))))
    }
    pub(crate) fn original_control_bytes() -> Option<usize> {
        let allocation = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Inner>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            std::mem::size_of::<Inner>(),
            std::mem::size_of::<Option<Inner>>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, crate::working_memory::WorkingMemoryError>>(),
            MetadataCustody::original_prepared_control_bytes()?,
        ]
        .into_iter()
        .try_fold(allocation, usize::checked_add)
    }
    /// Consumes an already constructed identity without cloning its allocations.
    /// Construction needs the caller's host authority; measurement/attachment
    /// below does not retroactively admit the payload's original allocation.
    pub fn new(payload: PreparedInputCacheIdentity) -> Self {
        Self(Some(Arc::new(Inner {
            payload,
            custody: MetadataCustody::new(),
        })))
    }

    // Only the fixed text-identity compiler uses this privately initialized
    // ordinary owner. It grants no source/custody certificate or attachment.
    pub(super) fn new_text(payload: PreparedInputCacheIdentity) -> Self {
        Self(Some(Arc::new(Inner {
            payload,
            custody: MetadataCustody::new_text(),
        })))
    }
    pub(super) fn new_original_text(
        payload: PreparedInputCacheIdentity,
        custody: crate::working_memory::OriginalTextMetadataCustody,
    ) -> Result<Self, crate::working_memory::WorkingMemoryError> {
        let input = (payload, custody);
        crate::working_memory::qualified_shared_bytes::<()>()?;
        let mut inner = Inner {
            payload: input.0,
            custody: MetadataCustody::new_original_text(input.1),
        };
        inner.custody.prepare_original_text()?;
        Ok(Self(Some(Arc::new(inner))))
    }
    // Payload layouts stay in their owning module; the prompt quote qualifies
    // the actual shared headers and adds the one pre-share PAL allocation.
    // PreparedInputCacheIdentity's inline extent is already in identity.peak.
    pub(crate) fn text_control_request() -> Option<(std::alloc::Layout, std::alloc::Layout, usize)>
    {
        use std::{alloc::Layout, mem::size_of};
        let controls = [
            size_of::<Inner>(),
            size_of::<Option<Inner>>(),
            size_of::<Self>(),
            size_of::<Arc<Inner>>(),
            size_of::<Result<Self, super::TextInputIdentityError>>(),
        ]
        .into_iter()
        .try_fold(
            MetadataCustody::text_control_bytes()?
                .checked_add(MetadataCustody::text_attachment_control_bytes()?)?,
            usize::checked_add,
        )?;
        Some((Layout::new::<Inner>(), Layout::new::<()>(), controls))
    }

    /// Genuine original source retained by this cache, when constructed inside B.
    /// Ordinary cache creation cannot install this source certificate.
    pub fn original_source(&self) -> Option<&crate::working_memory::OriginalPreparedHostInput> {
        self.inner().custody.original_prepared_source()
    }
    pub(crate) fn matches_original_account(
        &self,
        expected: &crate::working_memory::PreparedInputHostCustody,
    ) -> bool {
        self.inner().custody.original_prepared_matches(expected)
    }
    /// Authenticates already-paid residence without registration or credit.
    /// None denotes an ordinary cache; foreign original sources are errors.
    pub fn original_residence(
        &self,
        pool: &crate::working_memory::MemoryLedger,
    ) -> Option<Result<(), crate::working_memory::WorkingMemoryError>> {
        self.inner().custody.original_prepared_residence(pool)
    }
    /// Fixed controls for lending a publication witness from this identity.
    /// The worker only shares existing closed accounts and allocates no storage.
    pub fn original_publication_source_control_bytes() -> Option<usize> {
        use super::OriginalPreparedWorkspaceSource;
        use crate::working_memory::{PreparedInputHostCustody, WorkingMemoryError};
        [
            std::mem::size_of::<OriginalPreparedWorkspaceSource>(),
            std::mem::size_of::<PreparedInputHostCustody>(),
            std::mem::size_of::<Option<&PreparedInputHostCustody>>(),
            std::mem::size_of::<Result<(), WorkingMemoryError>>(),
            std::mem::size_of::<Option<Result<OriginalPreparedWorkspaceSource, WorkingMemoryError>>>(),
            std::mem::size_of::<Option<OriginalPreparedWorkspaceSource>>(),
        ].into_iter().try_fold(0usize, usize::checked_add)
    }
    /// Account-only source evidence for publication of this already-paid cache.
    /// It contains no projected tensor roots and cannot supply workspace credit
    /// or authorize media execution. Ordinary identities return None.
    pub fn original_publication_source(
        &self,
        pool: &crate::working_memory::MemoryLedger,
    ) -> Option<
        Result<super::OriginalPreparedWorkspaceSource, crate::working_memory::WorkingMemoryError>,
    > {
        let residence = self.original_residence(pool)?;
        let custody = self.inner().custody.original_prepared_account()?;
        Some(
            residence
                .map(|()| super::OriginalPreparedWorkspaceSource::cache_residence(custody.share())),
        )
    }
    /// Exact attachment-domain check for an existing original source witness.
    /// This cannot create an attachment or permit an original request.
    pub fn original_domain_matches(&self, domain: &SharedStorageAccountingId) -> Option<bool> {
        self.inner().custody.original_prepared_domain(domain)
    }
    /// Opaque process-local storage identity, independent of the payload lifetime.
    pub fn identity(&self) -> &HostMetadataIdentity {
        self.inner().custody.identity()
    }

    /// Exact retained payload capacity, including its inline representation.
    /// Excludes shared ownership/custody metadata and allocator bookkeeping.
    pub fn capacity_bytes(&self) -> Option<u64> {
        self.inner().payload.capacity_bytes()
    }

    /// Whether both aliases share the same physical payload and custody.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live cache"),
            other.0.as_ref().expect("live cache"),
        )
    }

    /// Borrows the exact ordered, payload-free prepared description.
    pub fn prepared(&self) -> &PreparedInputIdentity {
        self.inner().payload.prepared()
    }

    /// Borrows the original semantic-content fingerprint.
    pub fn semantic_content_fingerprint(&self) -> &str {
        self.inner().payload.semantic_content_fingerprint()
    }

    /// Borrows the already computed canonical prefix fingerprint.
    pub fn prefix_content_fingerprint(&self) -> &str {
        self.inner().payload.prefix_content_fingerprint()
    }

    /// Attaches one accounting owner per exact domain. Existing aliases share
    /// custody. The closed provider must perform accounting only: no owner
    /// reentry, native work or user callbacks while custody is locked. A failed
    /// provider does not publish an attachment or change the retained payload.
    pub(crate) fn original_attachment_ready(
        &self,
        domain: &SharedStorageAccountingId,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        self.inner().custody.original_attachment_ready(domain)
    }

    /// Funds the new attachment node and closed owner before construction.
    pub(crate) fn try_attach_owned_prepared<T: eredu_core::SharedStorageRetirement, E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(
            eredu_core::SharedStorageAttachmentLayout,
        ) -> Result<eredu_core::SharedStorageOwner<T>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.inner()
            .custody
            .try_attach_owned_prepared(owner, acquire)
    }

    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageAccountingId,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.inner().custody.try_attach(domain, acquire)
    }
}

impl AsRef<PreparedInputCacheIdentity> for SharedPreparedInputCacheIdentity {
    fn as_ref(&self) -> &PreparedInputCacheIdentity {
        &self.inner().payload
    }
}

impl From<PreparedInputCacheIdentity> for SharedPreparedInputCacheIdentity {
    fn from(payload: PreparedInputCacheIdentity) -> Self {
        Self::new(payload)
    }
}

impl PartialEq for SharedPreparedInputCacheIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.inner().payload == other.inner().payload
    }
}
impl Eq for SharedPreparedInputCacheIdentity {}

impl fmt::Debug for SharedPreparedInputCacheIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPreparedInputCacheIdentity")
            .field("identity", self.identity())
            .field("capacity_bytes", &self.capacity_bytes())
            .field("parts", &self.prepared().len())
            .finish()
    }
}

#[cfg(test)]
mod tests;

impl Clone for SharedPreparedInputCacheIdentity {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live cache"))))
    }
}
impl Drop for SharedPreparedInputCacheIdentity {
    fn drop(&mut self) {
        // No Weak/raw Arc escapes. The last strong exit deallocates the control
        // before destroying payload and original/ordinary custody.
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
