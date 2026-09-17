//! Descriptor observation and handle retention without native housekeeping.

use super::{runtime_lock, Array, ArrayAllocationInfo, Dtype};

mod descriptor;
pub use descriptor::{
    ArrayDescriptorError, ArrayDescriptorFacts, ArrayDescriptorLoan, OwnedArrayDescriptorLoan,
};

/// Ordinary, serialized entry for an inventory of already retained arrays.
///
/// Entry waits for the existing process runtime owner and performs the same
/// housekeeping as an ordinary native call. Subsequent cold allocation queries
/// and inspection clones on this thread reuse this owner; they still do not
/// evaluate arrays or turn unknown backing into completed storage.
///
/// Use only at an authorized ordinary boundary, before borrowing its inventory.
/// Do not hold this guard across submission waits, worker joins, or acquisition
/// of a resource mutex whose owner may itself be waiting for the runtime. Drop
/// it before those operations. It supplies no submission, allocation-budget or
/// original-execution authority. Cold and original callers must keep using the
/// immediate, fallible inspection APIs instead of entering this guard.
#[must_use = "retain the guard through the ordinary inventory"]
pub struct OrdinaryArrayMetadataGuard {
    _guard: runtime_lock::RuntimeLockGuard,
}

impl OrdinaryArrayMetadataGuard {
    /// Enters once using ordinary runtime serialization, before any inspection.
    /// No query is attempted or retried by this constructor. An active
    /// original-required scope returns a fixed domain refusal before entry;
    /// housekeeping cannot change that domain check into original authority.
    pub fn enter() -> Result<Self, crate::OriginalNativeControlError> {
        // SAFETY: this existing TLS-only check reads the actual current Scope's
        // original-required flag, including inherited/unconfigured children.
        // Refuse before ordinary lock waiting, reclamation or housekeeping.
        if !unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
            return Err(crate::OriginalNativeControlError::ForeignDomain);
        }
        let guard = runtime_lock::enter();
        // An ordinary housekeeping callback may have changed current scope.
        if !unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
            return Err(crate::OriginalNativeControlError::ForeignDomain);
        }
        Ok(Self { _guard: guard })
    }
}

impl std::fmt::Debug for OrdinaryArrayMetadataGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OrdinaryArrayMetadataGuard")
            .finish()
    }
}

/// Immutable descriptor facts with no tensor, allocation owner or payload pin.
/// The facts describe the instant of observation and do not certify completion
/// of an unknown backing or authorize reuse after the array changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayMetadataSnapshot {
    shape: Vec<i32>,
    dtype: Dtype,
    nbytes: usize,
    allocation: Option<ArrayAllocationInfo>,
}

impl ArrayMetadataSnapshot {
    /// Logical shape of the retained descriptor.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Physical element representation.
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Logical byte count; never a substitute for unknown backing capacity.
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }
    /// Completed certified backing, or unknown without evaluation or polling.
    pub const fn allocation(&self) -> Option<ArrayAllocationInfo> {
        self.allocation
    }
}

/// A descriptor observation could not be completed without native work.
#[derive(Debug, thiserror::Error)]
pub enum ArrayMetadataError {
    /// Another thread owns the runtime lock; nothing was observed or executed.
    #[error("native runtime is busy during array metadata inspection")]
    RuntimeBusy,
    /// The existing native metadata query failed.
    #[error("array metadata inspection failed: {0}")]
    Native(#[source] crate::error::Exception),
}

impl Array {
    /// Native heap-object bytes created by one successful inspection clone.
    ///
    /// Returns the linked C++ implementation's size of the single array handle.
    /// The clone shares its existing `ArrayDesc`, shape, graph and backing; those
    /// allocations are retained, not copied. This excludes allocator bookkeeping,
    /// Rust wrappers and guards, source storage, and error-reporting allocations.
    /// Callers must account for those separately where applicable.
    ///
    /// This is a cold representation fact: it needs no array, device or runtime
    /// lock and does not allocate, evaluate, poll, reclaim or run housekeeping.
    /// It supplies no storage identity, completion or execution authority.
    pub fn inspection_clone_handle_bytes() -> usize {
        // SAFETY: The C shim returns sizeof its linked array type. It takes no
        // pointers, accesses no runtime state, and cannot throw or allocate.
        unsafe { safemlx_sys::mlx_array_clone_handle_bytes() }
    }

    /// Reads completed backing identity and full capacity without copying shape
    /// metadata or allocating a descriptor snapshot. The successful result has
    /// fixed size and contains no allocation owner or payload pin.
    ///
    /// Does not evaluate, poll, reclaim owners or run native housekeeping.
    /// Returns immediately with `RuntimeBusy` on runtime-lock contention.
    /// Unfinished or uncertified backing remains `None`; logical tensor bytes
    /// never substitute for a missing physical bound. The caller must retain
    /// the source and keep it settled and unchanged when using these facts.
    pub fn try_allocation_info(&self) -> Result<Option<ArrayAllocationInfo>, ArrayMetadataError> {
        runtime_lock::try_retire(|| self.allocation_info())
            .ok_or(ArrayMetadataError::RuntimeBusy)?
            .map_err(ArrayMetadataError::Native)
    }

    /// Shares this exact array descriptor for retained inspection without
    /// evaluating, polling, copying numerical payload, reclaiming owners or
    /// invoking runtime housekeeping. Returns immediately with `RuntimeBusy`
    /// when another thread owns the native runtime lock. Native handle-creation
    /// failures retain their original exception as the error source.
    ///
    /// Unlike `try_metadata_snapshot`, this retains the actual descriptor and
    /// its existing graph/backing. It creates no completion, immutable-content
    /// or allocation authority. Callers using a separately captured snapshot
    /// must keep the source settled and unchanged between observation and
    /// retention; unrelated aliases may otherwise change its native state.
    pub fn try_clone_for_inspection(&self) -> Result<Self, ArrayMetadataError> {
        // The purpose-specific public operation retains one existing handle.
        // This private lock primitive supplies no generic callback authority or
        // permission to retire an unresolved submission.
        runtime_lock::try_retire(|| self.try_clone_handle())
            .ok_or(ArrayMetadataError::RuntimeBusy)?
            .map_err(ArrayMetadataError::Native)
    }

    /// Captures descriptor facts without evaluating, polling, reading payload,
    /// reclaiming owners or invoking native housekeeping. Returns immediately
    /// when another thread owns the runtime lock. Copying shape metadata does
    /// not create numerical storage or retain this array's backing allocation.
    pub fn try_metadata_snapshot(&self) -> Result<ArrayMetadataSnapshot, ArrayMetadataError> {
        // This internal lock primitive suppresses reentrant housekeeping. It
        // grants no terminal-resource or completion authority to this query.
        runtime_lock::try_retire(|| {
            Ok(ArrayMetadataSnapshot {
                shape: self.shape().to_vec(),
                dtype: self.dtype(),
                nbytes: self.nbytes(),
                allocation: self.allocation_info()?,
            })
        })
        .ok_or(ArrayMetadataError::RuntimeBusy)?
        .map_err(ArrayMetadataError::Native)
    }
}

#[cfg(test)]
mod tests;
