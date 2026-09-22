//! Immutable prompt-cache metadata and its retained construction and estimate accounts.
use super::PromptCacheManifest;
use crate::{HostMetadataFunding, HostMetadataFundingError};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

struct Inner {
    manifest: PromptCacheManifest,
    funding: HostMetadataFunding,
    dependency: Option<HostMetadataFunding>,
}

/// Shared immutable ownership of one funded prompt-cache manifest.
///
/// Cloning shares the actual backing, its construction account and any separately
/// admitted dependency estimate account. Serialization has the same representation
/// as [`PromptCacheManifest`]. This owner supplies neither file authenticity nor
/// cache restoration or native execution authority.
/// An explicit clone of the borrowed plain manifest is a separate caller-owned
/// allocation and does not inherit this account.
#[derive(Clone)]
pub struct SharedPromptCacheManifest(Option<Arc<Inner>>);

/// Prepaid shared shell for one manifest built by the existing persistence worker.
/// Payload constructors must reserve their own allocations on this same account
/// before allocating. This token supplies no native or file operation authority.
pub struct PreparedPromptCacheManifest {
    funding: HostMetadataFunding,
    dependency: Option<HostMetadataFunding>,
}

impl PreparedPromptCacheManifest {
    /// Shared shell and fixed construction transports, excluding manifest fields.
    pub fn control_bytes() -> Option<usize> {
        let allocation = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Inner>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let frames = [
            allocation,
            size_of::<Self>(),
            size_of::<Inner>(),
            size_of::<SharedPromptCacheManifest>(),
            size_of::<Result<Self, HostMetadataFundingError>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Reserves the shared shell before the persistence worker builds its payload.
    pub fn prepare(funding: HostMetadataFunding) -> Result<Self, HostMetadataFundingError> {
        funding
            .reserve_metadata(Self::control_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        Ok(Self {
            funding,
            dependency: None,
        })
    }

    /// Retains a separately admitted dependency-overhead allowance alongside the
    /// exact metadata account. The producer reserves that estimate before calling
    /// this method; this shell neither sets nor enlarges the estimate allowance.
    pub fn prepare_with_dependency(
        funding: HostMetadataFunding,
        dependency: HostMetadataFunding,
    ) -> Result<Self, HostMetadataFundingError> {
        let mut prepared = Self::prepare(funding)?;
        prepared.dependency = Some(dependency);
        Ok(prepared)
    }

    /// Loans the actual account to the manifest's metadata constructors.
    pub fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }

    /// Moves the already funded manifest into its prepaid shared shell.
    /// No field is copied and no additional allocation allowance is inferred.
    pub fn publish(self, manifest: PromptCacheManifest) -> SharedPromptCacheManifest {
        SharedPromptCacheManifest(Some(Arc::new(Inner {
            manifest,
            funding: self.funding,
            dependency: self.dependency,
        })))
    }
}

impl SharedPromptCacheManifest {
    /// Borrows metadata without detaching its account or authenticating a file.
    pub fn as_manifest(&self) -> &PromptCacheManifest {
        &self
            .0
            .as_ref()
            .expect("live prompt-cache manifest")
            .manifest
    }
    /// Whether these aliases retain the exact same manifest backing.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live manifest"),
            other.0.as_ref().expect("live manifest"),
        )
    }
}
impl Drop for SharedPromptCacheManifest {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            if let Some(inner) = Arc::into_inner(owner) {
                // The closed Arc shell is freed before its payload and account.
                let Inner {
                    manifest,
                    funding,
                    dependency,
                } = inner;
                drop(manifest);
                drop(funding);
                drop(dependency);
            }
        }
    }
}
impl std::ops::Deref for SharedPromptCacheManifest {
    type Target = PromptCacheManifest;
    fn deref(&self) -> &Self::Target {
        self.as_manifest()
    }
}
impl AsRef<PromptCacheManifest> for SharedPromptCacheManifest {
    fn as_ref(&self) -> &PromptCacheManifest {
        self.as_manifest()
    }
}
impl std::fmt::Debug for SharedPromptCacheManifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_manifest().fmt(f)
    }
}
impl PartialEq for SharedPromptCacheManifest {
    fn eq(&self, other: &Self) -> bool {
        self.as_manifest() == other.as_manifest()
    }
}
impl Eq for SharedPromptCacheManifest {}
impl std::hash::Hash for SharedPromptCacheManifest {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.as_manifest(), state);
    }
}
impl serde::Serialize for SharedPromptCacheManifest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.as_manifest(), serializer)
    }
}
