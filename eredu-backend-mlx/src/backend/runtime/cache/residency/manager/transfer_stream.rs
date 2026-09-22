//! One admitted transfer stream shared by a manager and its state branches.
use crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams;
use eredu_core::BackendFailure;
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use safemlx::{DeviceType, Stream, StreamCopyPlan};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, Mutex},
};

impl super::CacheResidencyManager {
    /// Establishes the transfer constructor before workspace description. The
    /// manager's closed source slot selects one constructor for all branches.
    pub(crate) fn prepare_transfer_stream(
        &self,
        ledger: &MemoryLedger,
        execution: &Stream,
    ) -> Result<(), BackendFailure> {
        let mut slot = self
            .inner
            .transfer_stream
            .lock()
            .map_err(|_| BackendFailure::from_error(CacheTransferStreamError::Poisoned))?;
        if let Some(source) = &*slot {
            return source
                .with_stream(ledger, execution, |_| ())
                .map_err(BackendFailure::from_error);
        }
        *slot = Some(PreparedCacheTransferStream::prepare(ledger, execution)?);
        Ok(())
    }

    /// Installs the actual cold construction owner after model loading begins.
    /// This handoff creates no stream, worker, reservation or native allocation.
    pub(crate) fn install_transfer_stream(
        &self,
        source: &PreparedCacheTransferStream,
        ledger: &MemoryLedger,
        execution: &Stream,
    ) -> Result<(), CacheTransferStreamError> {
        source.with_stream(ledger, execution, |_| ())?;
        let mut slot = self
            .inner
            .transfer_stream
            .try_lock()
            .map_err(|cause| match cause {
                std::sync::TryLockError::WouldBlock => CacheTransferStreamError::Busy,
                std::sync::TryLockError::Poisoned(_) => CacheTransferStreamError::Poisoned,
            })?;
        match &*slot {
            Some(installed)
                if !Arc::ptr_eq(
                    installed.0.as_ref().expect("retained transfer owner"),
                    source.0.as_ref().expect("retained transfer owner"),
                ) =>
            {
                Err(CacheTransferStreamError::ForeignSource)
            }
            Some(_) => Ok(()),
            None => {
                *slot = Some(source.clone());
                Ok(())
            }
        }
    }

    /// Pure ownership loan. It does not create a stream or initialize workers.
    pub(crate) fn prepared_transfer_stream(
        &self,
    ) -> Result<PreparedCacheTransferStream, CacheTransferStreamError> {
        self.inner
            .transfer_stream
            .try_lock()
            .map_err(|cause| match cause {
                std::sync::TryLockError::WouldBlock => CacheTransferStreamError::Busy,
                std::sync::TryLockError::Poisoned(_) => CacheTransferStreamError::Poisoned,
            })?
            .as_ref()
            .cloned()
            .ok_or(CacheTransferStreamError::Unavailable)
    }
}

type Resource = InitializedSharedNative<Mutex<PreparedExecutionStreams>>;

/// Closed sharing frees the wrapper allocation before retiring its resources
/// and host account. No weak reference or mutable native stream escapes.
pub(crate) struct PreparedCacheTransferStream(Option<Arc<Resource>>);
impl Clone for PreparedCacheTransferStream {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for PreparedCacheTransferStream {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for PreparedCacheTransferStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCacheTransferStream")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CacheTransferStreamError {
    #[error("cache manager already retains a different transfer source")]
    ForeignSource,
    #[error("cache transfer stream constructor source is unavailable")]
    Unavailable,
    #[error("cache transfer stream target differs from the execution device")]
    Device,
    #[error("cache transfer stream source is poisoned")]
    Poisoned,
    #[error("cache transfer stream source is busy")]
    Busy,
    #[error(transparent)]
    Accounting(#[from] WorkingMemoryError),
    #[error(transparent)]
    Snapshot(#[from] safemlx::StreamCopyCause),
    #[error(transparent)]
    Native(#[from] BackendFailure),
}

#[derive(Debug)]
struct Initializer {
    ledger: MemoryLedger,
    device: DeviceType,
    caller_control_bytes: usize,
}
impl SharedNativeInitializer for Initializer {
    type Output = Mutex<PreparedExecutionStreams>;
    type Error = CacheTransferStreamError;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // The supported Rust Arc representation uses two usize counters, then
        // the aligned payload. Each stream/worker constructor has its own
        // existing native initializer and independently retained account.
        let (layout, _) = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<Resource>())
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let controls = [
            layout.pad_to_align().size(),
            size_of::<Self>(),
            size_of::<PreparedCacheTransferStream>(),
            size_of::<Resource>(),
            size_of::<Option<PreparedExecutionStreams>>(),
            size_of::<Result<Self::Output, Self::Error>>(),
            size_of::<Result<Resource, SharedNativeInitializationError<Self>>>(),
            size_of::<SharedNativeInitializationError<Self>>(),
            size_of::<CacheTransferStreamError>(),
            size_of::<BackendFailure>(),
            self.caller_control_bytes,
            size_of::<(&MemoryLedger, &Stream)>(),
            size_of::<StreamCopyPlan<()>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        _custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let streams = PreparedExecutionStreams::for_device_factory(&self.ledger, self.device)
            .map_err(|cause| CacheTransferStreamError::Native(cause.into_backend_failure()))?
            .ok_or(CacheTransferStreamError::Unavailable)?;
        Ok(Mutex::new(streams))
    }
}

impl PreparedCacheTransferStream {
    #[cfg(test)]
    pub(crate) fn registry_owner_account(&self) -> Option<(usize, u64)> {
        let owner = self.0.as_ref()?;
        let resources = owner.output().lock().ok()?;
        Some((
            Arc::as_ptr(owner) as usize,
            resources.permanent_registry_bytes()?,
        ))
    }

    /// Native state construction calls this before inference tracing. A pure
    /// workspace traversal only borrows the completed resource below.
    pub(crate) fn prepare(
        ledger: &MemoryLedger,
        execution: &Stream,
    ) -> Result<Self, BackendFailure> {
        Self::prepare_for_caller::<()>(ledger, execution)
    }

    /// The caller's concrete cold handoff remains charged with the shared owner.
    pub(crate) fn prepare_for_caller<C>(
        ledger: &MemoryLedger,
        execution: &Stream,
    ) -> Result<Self, BackendFailure> {
        let caller_control_bytes = size_of::<C>()
            .checked_add(size_of::<Option<C>>())
            .and_then(|bytes| bytes.checked_add(size_of::<Result<C, BackendFailure>>()))
            .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::Overflow))?;
        let source =
            StreamCopyPlan::<()>::capture(execution).map_err(BackendFailure::from_error)?;
        // This selected factory exposes the backend's registered default target.
        // An ordinal without its own admitted constructor cannot borrow it.
        if source.device_index() != 0 {
            return Err(BackendFailure::from_error(CacheTransferStreamError::Device));
        }
        let initialized = ledger
            .initialize_shared_native(Initializer {
                ledger: ledger.clone(),
                device: source.device_type(),
                caller_control_bytes,
            })
            .map_err(BackendFailure::from_error)?;
        Ok(Self(Some(Arc::new(initialized))))
    }

    /// Borrows only this actual native resource. The caller supplies separately
    /// admitted copying, transfer, completion and destination controls.
    pub(crate) fn with_stream<R>(
        &self,
        ledger: &MemoryLedger,
        execution: &Stream,
        run: impl FnOnce(&Stream) -> R,
    ) -> Result<R, CacheTransferStreamError> {
        let owner = self.0.as_ref().expect("live closed transfer stream");
        owner.validate_pool(ledger)?;
        let resources = owner
            .output()
            .lock()
            .map_err(|_| CacheTransferStreamError::Poisoned)?;
        resources
            .validate_pool(ledger)
            .map_err(|cause| CacheTransferStreamError::Native(BackendFailure::from_error(cause)))?;
        let expected = StreamCopyPlan::<()>::capture(execution)?;
        let actual = StreamCopyPlan::<()>::capture(resources.execution())?;
        if expected.device_type() != actual.device_type()
            || expected.device_index() != actual.device_index()
        {
            return Err(CacheTransferStreamError::Device);
        }
        Ok(run(resources.execution()))
    }
}

#[cfg(test)]
#[path = "transfer_stream/tests.rs"]
mod tests;
