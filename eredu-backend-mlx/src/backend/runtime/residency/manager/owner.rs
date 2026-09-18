//! Shared manager identity whose strong and weak aliases retain source custody.
use super::{ManagerInner, MemoryTier, OffloadUnitId, ResidencyLeaseOwner};
use eredu_runtime::working_memory::{
    OriginalHostMetadataCustody, SharedNativeInitializationCustody, WorkingMemoryError,
};
use std::{
    alloc::Layout,
    fmt,
    ops::Deref,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

// No raw Arc or Weak of this control escapes. The final shared block is freed
// before its account retires; the account contains no manager/source backedge.
#[derive(Clone, Default, Debug)]
pub(crate) struct ManagerCustody(Option<Arc<SharedNativeInitializationCustody>>);
impl Drop for ManagerCustody {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl ManagerCustody {
    pub(crate) fn is_source_funded(&self) -> bool { self.0.is_some() }
    pub(super) fn new(custody: SharedNativeInitializationCustody) -> Self {
        Self(Some(Arc::new(custody)))
    }
    pub(super) fn validate_pool(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.0
            .as_deref()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .validate_pool(pool)
    }
    pub(super) fn validate_operation_custody(
        &self,
        custody: &eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Result<(), WorkingMemoryError> {
        custody.validate_initialization(self.0.as_deref().ok_or(WorkingMemoryError::UnknownBound)?)
    }
    pub(super) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<
            SharedNativeInitializationCustody,
        >())
    }
}

/// Only this strong handle constructs or clones the manager's Arc. Weak leases
/// retain the same independent account without keeping manager state alive.
#[derive(Clone)]
pub(crate) struct ManagerOwner {
    value: Arc<ManagerInner>,
    custody: ManagerCustody,
}
impl ManagerOwner {
    pub(super) fn new(value: ManagerInner, custody: ManagerCustody) -> Self {
        Self {
            value: Arc::new(value),
            custody,
        }
    }
    pub(crate) fn source_custody(&self) -> Option<ManagerCustody> {
        self.custody.0.as_ref().map(|_| self.custody.clone())
    }
    pub(crate) fn downgrade(&self) -> ManagerWeak {
        ManagerWeak {
            value: Arc::downgrade(&self.value),
            custody: self.custody.clone(),
        }
    }
    pub(crate) fn as_ptr(&self) -> *const ManagerInner {
        Arc::as_ptr(&self.value)
    }
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
    pub(super) fn validate_pool(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_pool(pool)
    }
    pub(super) fn validate_operation_custody(
        &self,
        custody: &eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_operation_custody(custody)
    }
    pub(super) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<ManagerInner>())?
            .checked_add(ManagerCustody::storage_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)
    }
}
impl Deref for ManagerOwner {
    type Target = ManagerInner;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl fmt::Debug for ManagerOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ManagerOwner").field(&self.as_ptr()).finish()
    }
}

/// Opaque manager identity retained by residency leases and operation sources.
/// It exports neither a raw weak/strong owner nor source or admission authority.
#[derive(Clone, Default)]
pub struct ManagerWeak {
    value: Weak<ManagerInner>,
    custody: ManagerCustody,
}
impl ManagerWeak {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn as_ptr(&self) -> *const ManagerInner {
        self.value.as_ptr()
    }
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        self.value.ptr_eq(&other.value)
    }
    pub(crate) fn upgrade(&self) -> Option<ManagerOwner> {
        self.value.upgrade().map(|value| ManagerOwner {
            value,
            custody: self.custody.clone(),
        })
    }
}
impl fmt::Debug for ManagerWeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ManagerWeak").field(&self.as_ptr()).finish()
    }
}
impl eredu_runtime::residency::ResidencyLeaseHandle<ManagerInner> for ManagerWeak {
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Some(owner) = self.upgrade() {
            owner.release_residency_pin(id, tier);
        }
    }
}

/// Transfer recovery can outlive the manager while retaining its poison flag.
/// The flag shares source custody, and never retains a manager strong handle.
#[derive(Clone)]
pub(super) struct FailureFlag {
    value: Arc<AtomicBool>,
    custody: ManagerCustody,
}
impl FailureFlag {
    pub(super) fn new(custody: ManagerCustody) -> Self {
        Self {
            value: Arc::new(AtomicBool::new(false)),
            custody,
        }
    }
    pub(super) fn load(&self, ordering: Ordering) -> bool {
        self.value.load(ordering)
    }
    pub(super) fn store(&self, value: bool, ordering: Ordering) {
        self.value.store(value, ordering);
    }
    pub(super) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<AtomicBool>())
    }
}

/// Manager operations only borrow their streams. Original construction reuses
/// the already admitted immutable pair rather than allocating extra C wrappers.
pub(super) enum ManagerStream {
    Ordinary(safemlx::Stream),
    Prepared {
        streams: crate::backend::runtime::checkpoint::store::PreparedMaterializationStreams,
        source: bool,
    },
}
impl Deref for ManagerStream {
    type Target = safemlx::Stream;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(stream) => stream,
            Self::Prepared {
                streams,
                source: true,
            } => streams.source_stream(),
            Self::Prepared {
                streams,
                source: false,
            } => streams.execution_stream(),
        }
    }
}
impl AsRef<safemlx::Stream> for ManagerStream {
    fn as_ref(&self) -> &safemlx::Stream {
        self
    }
}

// Request-born host results use the same opaque alias boundary as cold source
// buffers. Every alias keeps its account after the Arc field, so the last shared
// control deallocates before either source or request custody retires.
#[derive(Clone)]
enum HostCustody {
    Source(ManagerCustody),
    Request(eredu_runtime::working_memory::OriginalOperationMetadataCustody),
}

/// Immutable transfer-buffer owner preserving its source account through every
/// inventory and snapshot alias. No raw Arc/Weak extraction is available.
#[derive(Clone)]
pub struct RetainedHostBuffer {
    value: HostBufferValue,
    custody: HostCustody,
}
#[derive(Clone)]
enum HostBufferValue {
    Ordinary(Arc<safemlx::ImmutableHostTransferBuffer>),
    Original(Arc<PreparedHostBuffer>),
}
struct PreparedHostBuffer {
    buffer: safemlx::ImmutableHostTransferBuffer,
    attachment: Option<super::super::storage::PublishedAllocation>,
}
impl RetainedHostBuffer {
    pub(super) fn original(
        buffer: safemlx::ImmutableHostTransferBuffer,
        custody: ManagerCustody,
    ) -> Self {
        Self {
            value: HostBufferValue::Original(Arc::new(PreparedHostBuffer { buffer, attachment: None })),
            custody: HostCustody::Source(custody),
        }
    }
    pub(crate) fn request(
        source: crate::backend::runtime::residency::storage::filled_host::PublishedHostSource,
    ) -> Self {
        let (buffer, proof, custody) = source.into_parts();
        Self {
            value: HostBufferValue::Original(Arc::new(PreparedHostBuffer {
                buffer,
                attachment: proof,
            })),
            custody: HostCustody::Request(custody),
        }
    }
    pub(crate) fn attachment_receipt_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<&Self>(),
            size_of::<(&HostBufferValue, &HostCustody)>(),
            size_of::<Option<super::super::storage::PublishedAllocation>>(),
            size_of::<safemlx::AllocationInfo>(),
            // The map closure captures only this existing custody reference.
            size_of::<&eredu_runtime::working_memory::OriginalOperationMetadataCustody>(),
            size_of::<super::super::storage::RetainedAllocationReceipt<'_>>(),
            size_of::<Option<super::super::storage::RetainedAllocationReceipt<'_>>>(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn attachment_receipt(&self) -> Option<super::super::storage::RetainedAllocationReceipt<'_>> {
        match (&self.value, &self.custody) {
            (HostBufferValue::Original(value), HostCustody::Request(custody)) =>
                value.attachment.map(|proof| proof.borrow(custody)),
            _ => None,
        }
    }
    pub(crate) fn prepared_metadata(&self) -> Option<&safemlx::HostTransferMetadataSnapshot> {
        match &self.value {
            HostBufferValue::Original(value) => value.buffer.prepared_metadata(),
            _ => None,
        }
    }
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.value, &other.value) {
            (HostBufferValue::Ordinary(a), HostBufferValue::Ordinary(b)) => Arc::ptr_eq(a, b),
            (HostBufferValue::Original(a), HostBufferValue::Original(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    pub(crate) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<PreparedHostBuffer>())
    }
}
impl From<Arc<safemlx::ImmutableHostTransferBuffer>> for RetainedHostBuffer {
    fn from(value: Arc<safemlx::ImmutableHostTransferBuffer>) -> Self {
        Self {
            value: HostBufferValue::Ordinary(value),
            custody: HostCustody::Source(ManagerCustody::default()),
        }
    }
}
impl Deref for RetainedHostBuffer {
    type Target = safemlx::ImmutableHostTransferBuffer;
    fn deref(&self) -> &Self::Target {
        match &self.value {
            HostBufferValue::Ordinary(value) => value,
            HostBufferValue::Original(value) => &value.buffer,
        }
    }
}
impl AsRef<safemlx::ImmutableHostTransferBuffer> for RetainedHostBuffer {
    fn as_ref(&self) -> &safemlx::ImmutableHostTransferBuffer {
        self
    }
}
impl fmt::Debug for RetainedHostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedHostBuffer").finish_non_exhaustive()
    }
}

/// Named host storage retains the same source custody even after its manager
/// and all leases have retired. Only borrowed contents are exposed.
#[derive(Clone)]
pub struct ResidentHostOwner {
    value: Arc<super::ResidentHostBuffers>,
    custody: HostCustody,
    // Publication names/alias rows retire before their exact preparation H.
    _metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}
impl ResidentHostOwner {
    pub(super) fn original(value: super::ResidentHostBuffers, custody: ManagerCustody) -> Self {
        Self {
            value: Arc::new(value),
            custody: HostCustody::Source(custody),
            _metadata: None,
        }
    }
    pub(super) fn request(
        value: super::ResidentHostBuffers,
        custody: eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Self {
        Self {
            value: Arc::new(value),
            custody: HostCustody::Request(custody),
            _metadata: None,
        }
    }
    pub(super) fn request_with_metadata(
        value: super::ResidentHostBuffers,
        custody: eredu_runtime::working_memory::OriginalOperationMetadataCustody,
        metadata: eredu_nn::workspace::HostMetadataFunding,
    ) -> Self {
        Self {
            value: Arc::new(value),
            custody: HostCustody::Request(custody),
            _metadata: Some(metadata),
        }
    }
    pub(super) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(
            Layout::new::<super::ResidentHostBuffers>(),
        )
    }
}
impl From<Arc<super::ResidentHostBuffers>> for ResidentHostOwner {
    fn from(value: Arc<super::ResidentHostBuffers>) -> Self {
        Self {
            value,
            custody: HostCustody::Source(ManagerCustody::default()),
            _metadata: None,
        }
    }
}
impl Deref for ResidentHostOwner {
    type Target = super::ResidentHostBuffers;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl fmt::Debug for ResidentHostOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ResidentHostOwner")
            .field(&Arc::as_ptr(&self.value))
            .finish()
    }
}

// Test-only retirement observation keeps even its weak backing paired with
// source custody. Neither a raw Arc nor a raw Weak leaves this private owner.
#[cfg(test)]
pub(super) struct RetainedHostObserver {
    value: ObservedHostBuffer,
    custody: HostCustody,
}
#[cfg(test)]
enum ObservedHostBuffer {
    Ordinary(Weak<safemlx::ImmutableHostTransferBuffer>),
    Original(Weak<PreparedHostBuffer>),
}
#[cfg(test)]
impl RetainedHostBuffer {
    pub(super) fn observe_for_test(&self) -> RetainedHostObserver {
        RetainedHostObserver {
            value: match &self.value {
                HostBufferValue::Ordinary(value) => {
                    ObservedHostBuffer::Ordinary(Arc::downgrade(value))
                }
                HostBufferValue::Original(value) => {
                    ObservedHostBuffer::Original(Arc::downgrade(value))
                }
            },
            custody: self.custody.clone(),
        }
    }
}
#[cfg(test)]
impl RetainedHostObserver {
    pub(super) fn upgrade(&self) -> Option<RetainedHostBuffer> {
        let value = match &self.value {
            ObservedHostBuffer::Ordinary(value) => HostBufferValue::Ordinary(value.upgrade()?),
            ObservedHostBuffer::Original(value) => HostBufferValue::Original(value.upgrade()?),
        };
        Some(RetainedHostBuffer {
            value,
            custody: self.custody.clone(),
        })
    }
}
