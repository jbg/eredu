//! Closed immutable model/input metadata and per-domain accounting custody.

use crate::{input::SharedPreparedInputCacheIdentity, state::SharedStateLayout};
use eredu_core::{SharedStorageAttachmentError, SharedStorageDomain};
#[cfg(test)]
use std::sync::Arc;
use std::sync::Mutex;

mod funded_vec;
pub(crate) use funded_vec::{funded_vec, funded_vec_bytes};

mod slot_initialization;
pub use slot_initialization::{
    DenseHostSlotFinishError, DenseHostSlotInitialization, DenseHostSlotInitializationBuilder,
    HostSlotFinishError, HostSlotInitialization, HostSlotInitializationBuilder,
    HostSlotInitializationError, HostSlotPushError, InitializedDenseHostSlots,
    InitializedHostSlots, PreparedDenseHostCopyError,
};

mod slots;
pub use slots::{HostSlotAttachmentError, HostSlotMetadata, HostSlotTable};

mod identity;
pub use identity::{HostMetadataIdentity, HostMetadataKey};

enum MetadataAttachments {
    Ordinary(Mutex<Vec<Attachment>>),
    OriginalText {
        entries: Mutex<Vec<Attachment>>,
        // Vec and every boxed registration retire before raw control custody.
        custody: crate::working_memory::OriginalTextMetadataCustody,
    },
    OriginalPrepared {
        source: crate::working_memory::OriginalPreparedHostInput,
        custody: crate::working_memory::PreparedInputHostCustody,
    },
}

struct Attachment {
    domain: SharedStorageDomain,
    _custody: Box<dyn Send + Sync>,
}

/// Closed layout/input owners and the fixed slot metadata token construct this custody.
/// Closed payload fields retire before this custody. Fixed slot tokens cover
/// only their inline extent, never arbitrary transitive element payloads.
pub(crate) struct MetadataCustody {
    identity: HostMetadataIdentity,
    initialized: std::sync::atomic::AtomicBool,
    attachments: MetadataAttachments,
    reset: Option<crate::working_memory::resident_reset::ResetCustody>,
    // Identity may escape independently; this second loan protects metadata/PAL
    // storage until attachments and both metadata mutexes have retired.
    preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl MetadataCustody {
    pub(crate) fn new() -> Self {
        Self {
            identity: HostMetadataIdentity::ordinary(),
            initialized: std::sync::atomic::AtomicBool::new(false),
            attachments: MetadataAttachments::Ordinary(Mutex::new(Vec::new())),
            reset: None,
            preparation: None,
        }
    }

    pub(crate) fn new_prepared_host(
        identity: HostMetadataIdentity,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Self {
        let result = Self {
            identity,
            initialized: std::sync::atomic::AtomicBool::new(true),
            attachments: MetadataAttachments::Ordinary(Mutex::new(Vec::new())),
            reset: None,
            preparation: Some(authority.clone()),
        };
        if let MetadataAttachments::Ordinary(entries) = &result.attachments {
            drop(entries.lock().expect("private new attachment mutex"));
        }
        result
    }

    // The closed text identity worker has a known first-attachment population.
    // Initialize its mutex before sharing so exactly one PAL owner is possible.
    // Generic ordinary metadata retains its existing lazy behavior.
    pub(crate) fn new_text() -> Self {
        let result = Self::new();
        if let MetadataAttachments::Ordinary(storage) = &result.attachments {
            drop(
                storage
                    .lock()
                    .expect("private new metadata mutex cannot be poisoned"),
            );
        }
        result
            .initialized
            .store(true, std::sync::atomic::Ordering::Release);
        result
    }
    pub(crate) fn new_original_text(
        custody: crate::working_memory::OriginalTextMetadataCustody,
    ) -> Self {
        Self {
            identity: HostMetadataIdentity::ordinary(),
            initialized: std::sync::atomic::AtomicBool::new(false),
            attachments: MetadataAttachments::OriginalText {
                entries: Mutex::new(Vec::new()),
                custody,
            },
            reset: None,
            preparation: None,
        }
    }
    pub(crate) fn prepare_original_text(
        &mut self,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        let MetadataAttachments::OriginalText { entries, .. } = &mut self.attachments else {
            unreachable!()
        };
        let mut entries = entries.lock().expect("private metadata lock");
        crate::working_memory::qualified_shared_bytes::<()>()?;
        entries
            .try_reserve_exact(1)
            .map_err(crate::working_memory::WorkingMemoryError::ControlStorageReserve)?;
        if entries.capacity() != 1 {
            return Err(crate::working_memory::WorkingMemoryError::IdentityMismatch);
        }
        self.initialized
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }
    pub(crate) fn original_attachment_ready(
        &self,
        domain: &SharedStorageDomain,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        use crate::working_memory::WorkingMemoryError as E;
        match &self.attachments {
            MetadataAttachments::OriginalPrepared { custody, .. } => {
                if custody.pool().shared_storage_domain().same_identity(domain) {
                    Ok(())
                } else {
                    Err(E::IdentityMismatch)
                }
            }
            MetadataAttachments::OriginalText { custody, .. } => {
                if custody.matches_domain(domain) {
                    Ok(())
                } else {
                    Err(E::IdentityMismatch)
                }
            }
            MetadataAttachments::Ordinary(storage) => {
                // Never initialize an ordinary PAL owner while qualifying it.
                if !self.initialized.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(E::UnknownBound);
                }
                let entries = storage.try_lock().map_err(|_| E::UnknownBound)?;
                if entries.iter().any(|e| e.domain == *domain) {
                    Ok(())
                } else {
                    Err(E::UnknownBound)
                }
            }
        }
    }
    pub(crate) fn text_attachment_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        // Exactly one requested final slot; original construction never grows it.
        [
            size_of::<Attachment>(),
            size_of::<usize>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<(), crate::working_memory::WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn text_control_bytes() -> Option<usize> {
        use std::{
            mem::size_of,
            sync::{MutexGuard, PoisonError},
        };
        [
            size_of::<Self>(),
            size_of::<HostMetadataIdentity>(),
            size_of::<Mutex<Vec<Attachment>>>(),
            size_of::<MutexGuard<'static, Vec<Attachment>>>(),
            size_of::<
                Result<
                    MutexGuard<'static, Vec<Attachment>>,
                    PoisonError<MutexGuard<'static, Vec<Attachment>>>,
                >,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn original_reset(
        custody: crate::working_memory::resident_reset::ResetCustody,
        identity: HostMetadataIdentity,
    ) -> Self {
        Self {
            identity,
            initialized: std::sync::atomic::AtomicBool::new(false),
            attachments: MetadataAttachments::Ordinary(Mutex::new(Vec::new())),
            reset: Some(custody),
            preparation: None,
        }
    }

    pub(crate) fn original_prepared(
        custody: crate::working_memory::PreparedInputHostCustody,
        source: crate::working_memory::OriginalPreparedHostInput,
    ) -> Result<Self, crate::working_memory::WorkingMemoryError> {
        let identity = HostMetadataIdentity::original_prepared(custody.share())?;
        Ok(Self {
            identity,
            initialized: std::sync::atomic::AtomicBool::new(false),
            attachments: MetadataAttachments::OriginalPrepared { custody, source },
            reset: None,
            preparation: None,
        })
    }
    pub(crate) fn original_prepared_control_bytes() -> Option<usize> {
        HostMetadataIdentity::original_control_bytes()?
            .checked_add(std::mem::size_of::<
                Result<Self, crate::working_memory::WorkingMemoryError>,
            >())?
            .checked_add(std::mem::size_of::<
                crate::working_memory::PreparedInputHostCustody,
            >())
    }
    pub(crate) fn original_prepared_account(
        &self,
    ) -> Option<&crate::working_memory::PreparedInputHostCustody> {
        match &self.attachments {
            MetadataAttachments::OriginalPrepared { custody, .. } => Some(custody),
            _ => None,
        }
    }
    pub(crate) fn original_prepared_source(
        &self,
    ) -> Option<&crate::working_memory::OriginalPreparedHostInput> {
        match &self.attachments {
            MetadataAttachments::OriginalPrepared { source, .. } => Some(source),
            _ => None,
        }
    }
    pub(crate) fn original_prepared_matches(
        &self, expected: &crate::working_memory::PreparedInputHostCustody,
    ) -> bool {
        matches!(&self.attachments, MetadataAttachments::OriginalPrepared { custody, .. }
            if custody.same_account(expected))
    }
    pub(crate) fn original_prepared_residence(
        &self,
        pool: &crate::working_memory::WorkingMemoryPool,
    ) -> Option<Result<(), crate::working_memory::WorkingMemoryError>> {
        match &self.attachments {
            MetadataAttachments::OriginalPrepared { custody, source } => {
                Some(if custody.pool().same_domain(pool) {
                    source.validate_pool(pool)
                } else {
                    Err(crate::working_memory::WorkingMemoryError::IdentityMismatch)
                })
            }
            _ => None,
        }
    }
    pub(crate) fn original_prepared_domain(&self, domain: &SharedStorageDomain) -> Option<bool> {
        match &self.attachments {
            MetadataAttachments::OriginalPrepared { custody, .. } => {
                Some(custody.pool().shared_storage_domain().same_identity(domain))
            }
            _ => None,
        }
    }
    pub(crate) fn identity(&self) -> &HostMetadataIdentity {
        &self.identity
    }

    pub(crate) fn prepare_copy_attachment(
        &self,
        domain: &SharedStorageDomain,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        use crate::working_memory::WorkingMemoryError as E;
        if self.preparation.is_none() {
            return Err(E::IdentityMismatch);
        }
        let MetadataAttachments::Ordinary(storage) = &self.attachments else {
            return Err(E::IdentityMismatch);
        };
        let mut entries = storage.lock().map_err(|_| E::Poisoned)?;
        if entries.iter().any(|entry| entry.domain == *domain) {
            return Ok(());
        }
        // One actual first attachment of the prepared destination. Ordinary
        // cross-domain attachment remains on its existing growth path below.
        if !entries.is_empty() || entries.capacity() > 1 {
            return Err(E::IdentityMismatch);
        }
        if entries.capacity() == 0 {
            entries
                .try_reserve_exact(1)
                .map_err(E::ControlStorageReserve)?;
        }
        if entries.capacity() != 1 {
            return Err(E::IdentityMismatch);
        }
        Ok(())
    }
    pub(crate) fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.try_attach_mode(domain, acquire, false)
    }
    pub(crate) fn try_attach_prepared_copy<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        if self.preparation.is_none() {
            return Err(SharedStorageAttachmentError::AttachmentMismatch);
        }
        self.try_attach_mode(domain, acquire, true)
    }
    fn try_attach_mode<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
        prepared_copy: bool,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        // Every original alias, including SharedHostMetadata::Input, takes this
        // branch before any lazy lock, Vec reserve or provider invocation.
        let storage = match &self.attachments {
            MetadataAttachments::OriginalPrepared { custody, .. } => {
                return if custody.pool().shared_storage_domain().same_identity(domain) {
                    Ok(false) // existing source residence, never a registration proof
                } else {
                    Err(SharedStorageAttachmentError::AttachmentMismatch)
                };
            }
            MetadataAttachments::Ordinary(storage) => storage,
            MetadataAttachments::OriginalText { entries, custody } => {
                if !custody.matches_domain(domain) {
                    return Err(SharedStorageAttachmentError::AttachmentMismatch);
                }
                entries
            }
        };
        let mut attachments = storage
            .lock()
            .map_err(|_| SharedStorageAttachmentError::Poisoned)?;
        self.initialized
            .store(true, std::sync::atomic::Ordering::Release);
        if attachments.iter().any(|entry| entry.domain == *domain) {
            return Ok(false);
        }
        // Acquisition may create a charge. Reserve its infallible publication
        // slot first, so no later allocation can lose that returned handle.
        if prepared_copy || matches!(&self.attachments, MetadataAttachments::OriginalText { .. }) {
            if attachments.len() == attachments.capacity() {
                return Err(SharedStorageAttachmentError::AttachmentMismatch);
            }
        } else {
            attachments
                .try_reserve(1)
                .map_err(SharedStorageAttachmentError::Allocation)?;
        }
        let handle = acquire().map_err(SharedStorageAttachmentError::Provider)?;
        attachments.push(Attachment {
            domain: domain.clone(),
            _custody: handle,
        });
        Ok(true)
    }
}

impl Drop for MetadataCustody {
    fn drop(&mut self) {
        // Final exclusive access does not lock or treat poison as settlement.
        // Provider destructors run outside the owner mutex, including after a
        // failed acquisition, and may reenter other accounting owners safely.
        if let MetadataAttachments::Ordinary(storage)
        | MetadataAttachments::OriginalText {
            entries: storage, ..
        } = &mut self.attachments
        {
            let attachments = storage
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner());
            drop(std::mem::take(attachments));
        }
    }
}

/// One of the closed immutable host-metadata payloads retained by a model/input.
///
/// Clones preserve the actual shared allocation owner, including attachments
/// acquired after an earlier clone was created. This is not a generic payload
/// wrapper, an arbitrary byte proof, or allocation permission.
#[derive(Clone, Debug)]
pub enum SharedHostMetadata {
    /// Exact ordered state layout and its nested policy/segment storage.
    Layout(SharedStateLayout),
    /// Exact prepared-input description and semantic cache fingerprints.
    Input(SharedPreparedInputCacheIdentity),
    /// Precomputed unit/group observation paths from the actual architecture.
    ObservationPaths(crate::SharedLayeredObservationPaths),
}

impl SharedHostMetadata {
    /// The payload-free identity of this exact owner, independent of contents.
    pub fn identity(&self) -> &HostMetadataIdentity {
        match self {
            Self::Layout(layout) => layout.identity(),
            Self::Input(input) => input.identity(),
            Self::ObservationPaths(paths) => paths.identity(),
        }
    }

    /// Complete retained payload capacity reported by the concrete owner.
    /// Unknown or overflowing storage remains `None`, never a zero-byte proof.
    pub fn capacity_bytes(&self) -> Option<u64> {
        match self {
            Self::Layout(layout) => layout.capacity_bytes(),
            Self::Input(input) => input.capacity_bytes(),
            Self::ObservationPaths(paths) => paths.capacity_bytes(),
        }
    }

    /// Requires either this domain's existing attachment or the closed original
    /// text constructor's unused fixed slot. No allocator or provider runs.
    pub fn validate_original_attachment(
        &self,
        domain: &SharedStorageDomain,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        match self {
            Self::Layout(v) => v.original_attachment_ready(domain),
            Self::Input(v) => v.original_attachment_ready(domain),
            Self::ObservationPaths(v) => v.original_attachment_ready(domain),
        }
    }
    /// Attaches accounting custody once per exact domain. Returns `false`
    /// without calling the provider when that domain is already attached.
    /// The publication slot is reserved before provider acquisition; rejection
    /// preserves every earlier attachment and alias.
    ///
    /// The provider runs under the owner lock and must perform only closed
    /// accounting operations, without owner reentry, native work or callbacks.
    /// Its handle must not retain this payload owner directly or indirectly;
    /// use the separate identity/domain keys to avoid a registration cycle.
    /// Handles retire after the payload and outside the owner lock. An earlier
    /// provider panic poisons custody and rejects subsequent attachment.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        match self {
            Self::Layout(layout) => layout.try_attach(domain, acquire),
            Self::Input(input) => input.try_attach(domain, acquire),
            Self::ObservationPaths(paths) => paths.try_attach(domain, acquire),
        }
    }
}

#[cfg(test)]
mod tests;
