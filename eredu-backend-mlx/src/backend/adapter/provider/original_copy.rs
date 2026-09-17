//! Immutable native prerequisites for a separately admitted saved-state copy.

use super::{BackendStreams, MlxBackend};
use crate::backend::managed_memory::{gpu_stream::MlxStreamOwnershipError, input_allocator};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use safemlx::{
    InitializedInputAllocator, InputAllocatorCause, PreparedInputRuntime, PreparedStreamCopy,
    Stream, StreamCopyCause, StreamCopyPlan,
};

/// No scope, tensor, byte reservation, or owned native handle is created here.
/// The backend loan keeps the actual admitted stream owners alive; the process
/// allocator independently retains its admitted initializer account.
pub(crate) struct OriginalCopyEnvironment<'a> {
    stream: &'a Stream,
    pool: &'a WorkingMemoryPool,
    allocator: &'static InitializedInputAllocator,
}

/// Fixed pre-grant refusal. Constructing this value allocates no diagnostics.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalCopyEnvironmentError {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Streams(#[from] MlxStreamOwnershipError),
    #[error(transparent)]
    Allocator(#[from] InputAllocatorCause),
    #[error(transparent)]
    StreamValue(#[from] StreamCopyCause),
}

impl MlxBackend<'_> {
    /// Borrow only prerequisites born in this backend's managed domain. This
    /// does not initialize ordinary streams/runtime or infer ownership from an
    /// equivalent device or available capacity. Source/target compatibility,
    /// source quiescence, and fresh copy admission remain the caller's checks.
    pub(crate) fn original_copy_environment(
        &self,
    ) -> Result<OriginalCopyEnvironment<'_>, OriginalCopyEnvironmentError> {
        let streams = match &self.streams {
            BackendStreams::Prepared(streams) => streams,
            BackendStreams::Ordinary { .. } => return Err(WorkingMemoryError::UnknownBound.into()),
        };
        streams.validate_pool(&self.memory_pool)?;
        let allocator = input_allocator::admitted_initializer(&self.memory_pool)?;
        Ok(OriginalCopyEnvironment {
            stream: streams.execution(),
            pool: &self.memory_pool,
            allocator,
        })
    }
}

impl OriginalCopyEnvironment<'_> {
    pub(crate) fn stream(&self) -> &Stream {
        self.stream
    }

    pub(crate) fn pool(&self) -> &WorkingMemoryPool {
        self.pool
    }

    /// Revalidate the actual singleton when constructing copy storage. The
    /// returned thread-local witness owns no allocator and enters no new scope.
    pub(crate) fn input_runtime(
        &self,
    ) -> Result<PreparedInputRuntime, OriginalCopyEnvironmentError> {
        Ok(self.allocator.try_borrow_runtime()?)
    }

    pub(crate) fn allocator(&self) -> &'static InitializedInputAllocator {
        self.allocator
    }

    /// Named scalar/reference/result transport plus the real allocator loan.
    /// Stream/device/allocator births remain in their existing shared accounts.
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, OriginalCopyEnvironmentError>>(),
            size_of::<OriginalCopyEnvironmentError>(),
            size_of::<Result<PreparedInputRuntime, OriginalCopyEnvironmentError>>(),
            InitializedInputAllocator::borrow_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// The same initialized copy environment, retaining only pool identity and the
/// immutable native initializer. Stream ownership remains in the supplied closed
/// prepared-stream owner; no new device, queue or initializer is created.
#[derive(Clone)]
pub(crate) struct RetainedOriginalCopyEnvironment {
    stream: StreamCopyPlan<()>,
    pool: WorkingMemoryPool,
    allocator: &'static InitializedInputAllocator,
}
impl OriginalCopyEnvironment<'_> {
    pub(crate) fn retain_prerequisites(
        &self,
    ) -> Result<RetainedOriginalCopyEnvironment, OriginalCopyEnvironmentError> {
        Ok(RetainedOriginalCopyEnvironment {
            stream: StreamCopyPlan::capture(self.stream)?,
            pool: self.pool.clone(),
            allocator: self.allocator,
        })
    }
}
impl RetainedOriginalCopyEnvironment {
    pub(crate) fn control_bytes(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, OriginalCopyEnvironmentError>>(),
            OriginalCopyEnvironment::control_bytes()?,
            self.stream.control_bytes()?,
            self.stream.source_comparison_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn loan<'a, C: Send + Sync + 'static>(
        &'a self,
        stream: &'a PreparedStreamCopy<C>,
        pool: &WorkingMemoryPool,
    ) -> Result<OriginalCopyEnvironment<'a>, OriginalCopyEnvironmentError> {
        if !self.pool.same_domain(pool) || !self.stream.matches_source(stream.as_stream()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        // The same admitted singleton must still validate this pool; equal
        // stream values alone never substitute for its retained initializer.
        if !std::ptr::eq(
            input_allocator::admitted_initializer(&self.pool)?,
            self.allocator,
        ) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(OriginalCopyEnvironment {
            stream: stream.as_stream(),
            pool: &self.pool,
            allocator: self.allocator,
        })
    }
}

mod owned;
pub(crate) use owned::{PreparedOriginalCopyEnvironment, PreparedOriginalCopyEnvironmentError};
