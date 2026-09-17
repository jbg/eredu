//! Finite destination publication under an actual independently admitted copy.
use super::*;
use crate::working_memory::{WorkspaceCopyCustody, WorkspaceCopyRetention};
use eredu_core::HostPreparationAuthority;
use std::{marker::PhantomData, mem::size_of};

pub(super) enum PublicationOrigin {
    Prepaid(PrepaidStorageOrigin),
    Copy(WorkspaceCopyRetention),
}
impl PublicationOrigin {
    pub(super) fn same_source_account(&self, value: &crate::working_memory::OriginalHostMetadataCustody) -> bool {
        matches!(self, Self::Prepaid(PrepaidStorageOrigin::Immutable(origin)) if origin.raw().same(value))
    }
    pub(super) fn immutable(&self) -> bool {
        matches!(self, Self::Prepaid(origin) if origin.immutable())
    }
    pub(super) fn same_origin(&self, other: &PrepaidStorageOrigin) -> bool {
        matches!(self, Self::Prepaid(origin) if origin.same_origin(other))
    }
    pub(super) fn prepaid(&self) -> Result<&PrepaidStorageOrigin, WorkingMemoryError> {
        match self {
            Self::Prepaid(origin) => Ok(origin),
            Self::Copy(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(super) fn validate_capacity(&self, bytes: u64) -> Result<(), WorkingMemoryError> {
        self.prepaid()?.validate_capacity(bytes)
    }
    pub(super) fn validate_publisher(
        &self,
        scope: &WorkingMemoryFundingScope,
        usage: &crate::working_memory::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Prepaid(origin) => origin.validate_publisher(scope, usage),
            Self::Copy(copy) => copy.validate_publication(scope, usage),
        }
    }
    pub(super) fn preparation(&self) -> Option<&HostPreparationAuthority> {
        match self {
            Self::Copy(copy) => copy.preparation(),
            Self::Prepaid(_) => None,
        }
    }
}

/// Source-count layout for one terminal finite publication. These are host
/// constructor facts; neither the layout nor H can issue or increase B funding.
#[must_use]
pub struct WorkspaceCopyPublicationPlan<K: Clone + Ord + Send + Sync + 'static> {
    slots: usize,
    bytes: usize,
    marker: PhantomData<fn() -> K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> WorkspaceCopyPublicationPlan<K> {
    /// `nested_key_bytes` is the owning provider's bound for one actual key
    /// clone, excluding K's inline representation. Count every publication row.
    pub fn new(slots: usize, nested_key_bytes: u64) -> Result<Self, WorkingMemoryError> {
        let rows =
            PreparedNativePublication::<K>::qualified_control_bytes(slots, nested_key_bytes)?;
        let bytes =
            [
                usize::try_from(rows).map_err(|_| WorkingMemoryError::Overflow)?,
                size_of::<Self>(),
                size_of::<PreparedWorkspaceCopyPublication<K>>(),
                size_of::<Result<Self, WorkingMemoryError>>(),
                size_of::<Result<PreparedWorkspaceCopyPublication<K>, WorkingMemoryError>>(),
                size_of::<PublicationOrigin>(),
                size_of::<WorkspaceCopyRetention>(),
                size_of::<HostPreparationAuthority>(),
                size_of::<(&WorkspaceCopyCustody, &WorkingMemoryFundingScope)>(),
                size_of::<std::sync::MutexGuard<'_, crate::working_memory::Usage>>(),
                size_of::<
                    std::sync::LockResult<std::sync::MutexGuard<'_, crate::working_memory::Usage>>,
                >(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            slots,
            bytes,
            marker: PhantomData,
        })
    }
    /// Add this constructor contribution to the independently admitted host plan.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }

    /// Consumes the fixed row population after its host preparation admission.
    /// Only the actual same-account native copy scope can publish through it.
    /// This does not confer the original-text role or a prepaid native origin.
    pub fn prepare(
        self,
        copy: &WorkspaceCopyCustody,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<PreparedWorkspaceCopyPublication<K>, WorkingMemoryError> {
        let copy = copy.retention();
        let host = copy.preparation().ok_or(WorkingMemoryError::UnknownBound)?;
        {
            let usage = scope
                .pool()
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            copy.validate_publication(scope, &usage)?;
        }
        // The retained copy covers all partial constructor prefixes. Canonical
        // nodes and output registrations independently retain their host token.
        let inputs = crate::working_memory::qualified_storage::vector(self.slots, true)?;
        let rows = crate::working_memory::qualified_storage::vector(self.slots, true)?;
        let node = RegistryBatch::prepare_copy_exact(self.slots, host)?;
        let namespace = PreparedNamespace::prepare_copy::<K>(host);
        Ok(PreparedWorkspaceCopyPublication {
            inner: PreparedNativePublication {
                inputs,
                rows,
                node: Some(node),
                namespace: Some(namespace),
                terminal: false,
                published: false,
                slots: self.slots,
                exact_storage: true,
                failure_site: "registry copy preparation",
                partition: PublicationOrigin::Copy(copy),
            },
        })
    }
}

/// One actual B-funded transaction through the shared canonical publication
/// worker. Failed keys/rows and partial output shells stay here; no retry/reset.
#[must_use]
pub struct PreparedWorkspaceCopyPublication<K: Clone + Ord + Send + Sync + 'static> {
    inner: PreparedNativePublication<K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> PreparedWorkspaceCopyPublication<K> {
    /// Supply a completed physical allocation's exact identity and capacity.
    /// Backend inspection is responsible for that fact, as on ordinary funded
    /// publication. Every new canonical byte is debited from this copy's scope.
    pub fn push(&mut self, key: K, bytes: u64) -> Result<usize, WorkingMemoryError> {
        self.inner
            .push_observation(crate::working_memory::NativeStorageObservation::Ordinary(
                key, bytes,
            ))
    }
    /// Atomic validation and publication. A refusal consumes the attempt while
    /// preserving every staged owner and the original typed cause.
    pub fn publish(&mut self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.inner.publish(scope)
    }
    /// Independently retain an already published input for its physical owner.
    /// Duplicate inputs have no second output or charge.
    pub fn input(&self, index: usize) -> Option<&WorkingMemoryStorage<K>> {
        self.inner.input(index)
    }
    /// Move the existing registration to its final attachment after publication.
    pub fn take_input(&mut self, index: usize) -> Option<WorkingMemoryStorage<K>> {
        self.inner.take_input(index)
    }
}
