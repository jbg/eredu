//! Borrowed host plans consumed by the shared snapshot transaction.

/// Host-copy failure preserving either an existing ordinary diagnostic or an
/// already-owned neutral source, including its partial destination custody.
#[derive(Debug, thiserror::Error)]
pub enum TextHostCopyError {
    /// Fixed original account refusal before any destination/error allocation.
    #[error("{0}")]
    Admission(#[source] crate::working_memory::WorkingMemoryError),
    /// Existing ordinary parser/decoder diagnostic; no new formatting occurs.
    #[error("{0}")]
    Message(String),
    /// Exact provider cause and any retained partial-copy storage account.
    #[error("{0}")]
    Source(#[source] eredu_core::BackendFailure),
}

/// Complete host-state copy prepared without destination allocation.
///
/// The concrete plan borrows the actual source and reports its complete logical
/// copied storage. The shared driver acquires ordinary host preparation and
/// reserves the existing nonrefundable SnapshotBudget before calling `copy`.
/// Neither this trait nor that logical reservation grants original physical
/// storage, native submission, source replacement, or execution authority.
///
/// Implementations preserve their borrowed source on failure. Output aliases and
/// errors that escape the enclosing continuation need their own source/host
/// custody, just as the existing snapshot callback contracts require.
pub trait PreparedTextHostCopy {
    /// Independently mutable destination host state, with no source mutation.
    type Copied;

    /// Complete logical host contribution, including decoder/stop/cursor and
    /// delivery state selected by the concrete provider. Unknown remains unknown.
    fn storage_bytes(&self) -> Option<u64>;

    /// Consumes this plan once after budget reservation. `retained_bytes` is the
    /// complete transaction's logical amount, for paired public metadata only;
    /// it is not a physical allocation grant or permission to refresh a budget.
    fn copy(self, retained_bytes: u64) -> Result<Self::Copied, TextHostCopyError>;

    /// Named enclosing snapshot controls included in an original physical plan.
    /// Ordinary callbacks remain unqualified; a numerical logical size is not proof.
    fn original_control_bytes(&self) -> Option<usize> {
        None
    }
    /// Exact native planner contribution admitted by this same host destination.
    /// The shared driver compares it with the backend's current borrowed query.
    fn original_preparation_bytes(&self) -> Option<u64> {
        None
    }

    /// Uses an independently authenticated original destination account and
    /// returns its payload-free lifetime custody for the enclosing snapshot.
    /// This is called only after logical SnapshotBudget reservation. A source
    /// account or request grant must never be returned as destination custody.
    fn copy_original(
        self,
        _retained_bytes: u64,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError>
    where
        Self: Sized,
    {
        Err(TextHostCopyError::Admission(
            crate::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }
    /// Publishes the logical lease only after independent host admission, before
    /// constructing any copied provider. Every surviving output alias and the
    /// returned host authority must retain that same lease. Implementations that
    /// cannot couple those lifetimes remain unqualified; ordinary copy is not a
    /// fallback. This reservation grants no physical storage or native authority.
    fn copy_original_resume(
        self,
        _retained_bytes: u64,
        _reservation: super::PendingSnapshotResumeRetention,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError>
    where
        Self: Sized,
    {
        Err(TextHostCopyError::Admission(
            crate::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }
}

/// Compatibility callbacks keep their existing caller-owned estimate and scope.
/// New concrete providers bind their actual borrowed source in their own plan.
pub(super) struct CallbackHostCopy<F> {
    bytes: Option<u64>,
    copy: F,
}
impl<F> CallbackHostCopy<F> {
    pub(super) fn new(bytes: Option<u64>, copy: F) -> Self {
        Self { bytes, copy }
    }
}
impl<T, F: FnOnce(u64) -> Result<T, String>> PreparedTextHostCopy for CallbackHostCopy<F> {
    type Copied = T;
    fn storage_bytes(&self) -> Option<u64> {
        self.bytes
    }
    fn copy(self, retained_bytes: u64) -> Result<T, TextHostCopyError> {
        (self.copy)(retained_bytes).map_err(TextHostCopyError::Message)
    }
}
