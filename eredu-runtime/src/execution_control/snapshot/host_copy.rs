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

/// Borrowed additional host state copied inside the original semantic snapshot
/// transaction. The source remains immutable until this plan is consumed.
///
/// Implementations report the actual complete producer: destination backing,
/// transient controls, partial failure and error storage. `copy` executes only
/// after these bytes have joined the same physical admission as the cursor and
/// parser. It must retain the supplied destination authority in every escaping
/// payload or error; an input/source authority cannot substitute for it.
pub trait PreparedTextHostJournal {
    /// Independently owned journal copied from this exact borrowed source.
    type Copied;
    /// Logical retained/copy bytes charged once to the shared SnapshotBudget.
    fn storage_bytes(&self) -> Option<u64>;
    /// Exact prospective destination, scratch, control and failure producer bytes.
    fn preparation_bytes(&self) -> Option<usize>;
    /// Consumes the plan after original physical and logical admission.
    fn copy(
        self,
        retained_bytes: u64,
        destination: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self::Copied, TextHostCopyError>;
}
impl PreparedTextHostJournal for () {
    type Copied = ();
    fn storage_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn preparation_bytes(&self) -> Option<usize> {
        Some(0)
    }
    fn copy(
        self,
        _: u64,
        _: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), TextHostCopyError> {
        Ok(())
    }
}
