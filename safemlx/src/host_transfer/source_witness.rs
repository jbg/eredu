//! Borrowed exact immutable Host backing, with positive constructor provenance.
use super::ImmutableHostTransferBuffer;
use crate::{
    utils::runtime_lock, AllocationInfo, OriginalBufferCause, OriginalBufferError,
    PreparedAllocationOwner,
};
use std::mem::{size_of, size_of_val};

/// Exact retained Host allocation. This is observation and attachment authority,
/// never a grant to create a new prepaid source or numerical allocation.
pub struct ImmutableHostTransferWitness<'a> {
    pub(super) facts: safemlx_sys::mlx_immutable_host_transfer_info,
    buffer: &'a ImmutableHostTransferBuffer,
}
impl std::fmt::Debug for ImmutableHostTransferWitness<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImmutableHostTransferWitness")
            .field("allocation", &self.allocation())
            .field("prepared_source", &self.is_prepared_source())
            .finish_non_exhaustive()
    }
}
impl ImmutableHostTransferWitness<'_> {
    /// Nonrecycled native Host identity and full physical backing capacity.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(
            self.facts.backing.identity,
            self.facts.backing.charged_bytes,
            self.facts.backing.placement,
        )
        .with_host_controls(self.facts.backing.host_control_bytes)
    }
    /// Positive native constructor classification; false denotes actual ordinary
    /// Host allocation, not an unknown descriptor or a missing prepared tag.
    pub fn is_prepared_source(&self) -> bool {
        self.facts.prepared_source
    }
    /// Compare the same immutable owner, provenance and physical facts before
    /// attaching one prepaid node. Refusal preserves the entire unattached owner.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_immutable_host(self.buffer.buffer.raw, self.facts)
    }
    /// Fixed inspection/attachment transports; excludes separately priced nodes.
    /// No runtime entry, source work or allocation occurs while querying.
    pub fn inspection_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query.
        let native = unsafe { safemlx_sys::mlx_immutable_host_transfer_control_bytes() };
        let frames = [
            native,
            size_of::<Self>(),
            size_of::<AllocationInfo>(),
            size_of::<Result<Self, OriginalBufferCause>>(),
            size_of::<safemlx_sys::mlx_immutable_host_transfer_info>() * 2,
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<&ImmutableHostTransferBuffer>(),
            size_of::<u32>(),
            size_of::<Result<(), OriginalBufferCause>>(),
            size_of::<bool>(),
            size_of::<(
                safemlx_sys::mlx_host_transfer_buffer,
                safemlx_sys::mlx_immutable_host_transfer_info,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl ImmutableHostTransferBuffer {
    /// Positively inspect this retained immutable native Host owner without
    /// fabricating an Array, polling, evaluating, allocating or reclaiming.
    pub fn inspect_original_source(
        &self,
    ) -> Result<ImmutableHostTransferWitness<'_>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut facts = safemlx_sys::mlx_immutable_host_transfer_info {
            backing: safemlx_sys::mlx_original_buffer_info {
                host_control_bytes: 0,
                known: false,
                identity: 0,
                charged_bytes: 0,
                placement: safemlx_sys::mlx_memory_placement {
                    kind: 0,
                    device: -1,
                    device_count: 0,
                },
            },
            prepared_source: false,
        };
        // SAFETY: this immutable owner remains borrowed under one runtime loan;
        // fixed native inspection neither changes source state nor invokes callbacks.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_immutable_host_transfer_inspect(&mut facts, self.buffer.raw)
        })?;
        if !facts.backing.known || facts.backing.identity == 0 {
            return Err(OriginalBufferCause::UncertifiedBacking);
        }
        Ok(ImmutableHostTransferWitness {
            facts,
            buffer: self,
        })
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests;
