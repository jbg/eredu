//! Existing immutable Host allocation aliases, without source or birth authority.
use super::{OriginalBufferCause, OriginalBufferError, PreparedAllocationOwner};
use crate::{AllocationInfo, Array, utils::runtime_lock};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

/// A borrowed completed Array's actual Host-transfer allocation. Publication
/// preserves positive ordinary/prepared provenance from the exact Host owner.
/// This witness never certifies a new mutable birth or grants source funding.
pub struct HostTransferArrayAliasWitness<'a> {
    facts: safemlx_sys::mlx_immutable_host_transfer_info,
    array: &'a Array,
}
impl fmt::Debug for HostTransferArrayAliasWitness<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostTransferArrayAliasWitness")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl HostTransferArrayAliasWitness<'_> {
    /// Actual full capacity and opaque identity in the Host-transfer namespace.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(
            self.facts.backing.identity,
            self.facts.backing.charged_bytes,
            true,
        )
    }
    /// Positive provenance shared with the immutable Host buffer witness.
    pub fn is_prepared_source(&self) -> bool {
        self.facts.prepared_source
    }
    /// Revalidate the same completed Host allocation and append one prepared
    /// owner under a single native loan. Refusal returns the unchanged owner.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_host_array(self.array, self.facts)
    }
    /// Fixed inspection and attachment transport; no allocation or source grant.
    pub fn inspection_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query, without runtime or source work.
        let native = unsafe { safemlx_sys::mlx_host_transfer_array_alias_control_bytes() };
        let frames = [
            native,
            size_of::<Self>(),
            size_of::<AllocationInfo>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, OriginalBufferCause>>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<safemlx_sys::mlx_immutable_host_transfer_info>(),
            size_of::<bool>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<u32>(),
            size_of::<&Array>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl Array {
    /// Observe a settled Host-transfer backing without evaluation, polling,
    /// registration, ordinary fallback or new source authorization.
    pub fn inspect_host_transfer_alias(
        &self,
    ) -> Result<Option<HostTransferArrayAliasWitness<'_>>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut source = safemlx_sys::mlx_immutable_host_transfer_info {
            backing: safemlx_sys::mlx_original_buffer_info {
                known: false,
                identity: 0,
                charged_bytes: 0,
            },
            prepared_source: false,
        };
        // SAFETY: actual Array remains borrowed under one runtime loan; the
        // fixed native protocol neither consumes custody nor allocates.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_host_transfer_array_alias_info(&mut source, self.as_ptr())
        })?;
        let facts = source.backing;
        match (facts.known, facts.identity) {
            (true, identity) if identity != 0 => Ok(Some(HostTransferArrayAliasWitness {
                facts: source,
                array: self,
            })),
            (false, 0) if facts.charged_bytes == 0 => Ok(None),
            _ => Err(OriginalBufferCause::InvalidLayout),
        }
    }
}

/// One completed native Data view of certified immutable Host-transfer backing.
/// Clones, shared-buffer views and native pins of this Data retain an attached
/// owner. An independent reload or the immutable Host buffer does not. This is
/// Device-view residency custody; it grants no source or mutable birth authority.
#[derive(Debug)]
pub struct HostTransferArrayViewWitness<'a> {
    facts: safemlx_sys::mlx_host_transfer_view_info,
    array: &'a Array,
}
impl HostTransferArrayViewWitness<'_> {
    /// Revalidate Host backing and this nonrecycled native Data generation, then
    /// append the already prepared owner to that Data without allocation.
    /// Failure returns its unchanged owner; no native progress is performed.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_host_view(self.array, self.facts)
    }
    /// Exact fixed inspection and attachment transports, excluding owner nodes.
    pub fn control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query, no runtime/source interaction.
        let native = unsafe { safemlx_sys::mlx_host_transfer_array_view_control_bytes() };
        let frames = [
            native,
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, OriginalBufferCause>>(),
            size_of::<safemlx_sys::mlx_host_transfer_view_info>() * 2,
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<u32>(),
            size_of::<&Array>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl Array {
    /// Inspect this completed Host-backed native Data, without granting source
    /// authority, evaluating, polling, registering or selecting a fallback.
    pub fn inspect_host_transfer_view(
        &self,
    ) -> Result<Option<HostTransferArrayViewWitness<'_>>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut facts = safemlx_sys::mlx_host_transfer_view_info {
            backing: safemlx_sys::mlx_original_buffer_info {
                known: false,
                identity: 0,
                charged_bytes: 0,
            },
            view_identity: 0,
        };
        // SAFETY: the Array remains borrowed under one native runtime loan.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_host_transfer_array_view_info(&mut facts, self.as_ptr())
        })?;
        match (
            facts.backing.known,
            facts.backing.identity,
            facts.view_identity,
        ) {
            (true, backing, view) if backing != 0 && view != 0 => {
                Ok(Some(HostTransferArrayViewWitness { facts, array: self }))
            }
            (false, 0, 0) if facts.backing.charged_bytes == 0 => Ok(None),
            _ => Err(OriginalBufferCause::InvalidLayout),
        }
    }
}

#[cfg(all(test, feature = "metal"))]
#[path = "host_alias/tests.rs"]
mod tests;
