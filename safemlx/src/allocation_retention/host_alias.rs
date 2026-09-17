//! Existing immutable Host allocation aliases, without source or birth authority.
use super::{OriginalBufferCause, OriginalBufferError, PreparedAllocationOwner};
use crate::{AllocationInfo, Array, utils::runtime_lock};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

/// A borrowed completed Array's actual Host-transfer allocation. Publication
/// requires an independently authenticated existing prepaid immutable source.
/// This witness never certifies a new mutable birth or ordinary allocation.
pub struct HostTransferArrayAliasWitness<'a> {
    facts: safemlx_sys::mlx_original_buffer_info,
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
        AllocationInfo::from_native(self.facts.identity, self.facts.charged_bytes, true)
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
        let mut facts = safemlx_sys::mlx_original_buffer_info {
            known: false,
            identity: 0,
            charged_bytes: 0,
        };
        // SAFETY: actual Array remains borrowed under one runtime loan; the
        // fixed native protocol neither consumes custody nor allocates.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_host_transfer_array_alias_info(&mut facts, self.as_ptr())
        })?;
        match (facts.known, facts.identity) {
            (true, identity) if identity != 0 => {
                Ok(Some(HostTransferArrayAliasWitness { facts, array: self }))
            }
            (false, 0) if facts.charged_bytes == 0 => Ok(None),
            _ => Err(OriginalBufferCause::InvalidLayout),
        }
    }
}
