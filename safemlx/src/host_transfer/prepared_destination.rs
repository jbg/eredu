//! Exclusive paid Host destination for the shared CopyToHostTransfer worker.
use super::{HostTransferBuffer, ImmutableHostTransferBuffer};
use crate::{
    Array, Dtype, OperationEvent, OriginalScopeObserver, Stream, error::Exception,
    operation_event::ScopedOperation, utils::runtime_lock,
};
use std::{mem::size_of, ptr};

/// A fixed refusal or the original typed native copy/completion failure.
#[derive(Debug, thiserror::Error)]
pub enum PreparedHostCopyError {
    /// The destination was reused, unpublished, or not successfully completed.
    #[error("prepared host destination is not in the required state")]
    State,
    /// Original native failure with unchanged completion/source information.
    #[error(transparent)]
    Native(#[from] Exception),
}

/// One-use destination retaining its native prefix before its actual Host owner.
/// No public method exposes writable bytes or uncompleted Host storage.
pub struct PreparedHostCopyDestination {
    output: Option<Array>,
    event: Option<OperationEvent>,
    buffer: Option<HostTransferBuffer>,
    attempted: bool,
    completed: bool,
    failed: bool,
}
impl std::fmt::Debug for PreparedHostCopyDestination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedHostCopyDestination")
            .field("attempted", &self.attempted)
            .field("completed", &self.completed)
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}
impl PreparedHostCopyDestination {
    pub(super) fn new(buffer: HostTransferBuffer) -> Self {
        Self {
            output: None,
            event: None,
            buffer: Some(buffer),
            attempted: false,
            completed: false,
            failed: false,
        }
    }
    /// Fixed Rust controls; Host backing, source-arena and native graph/record
    /// populations are separately returned by their actual producers.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, crate::PreparedInputCause>>(),
            size_of::<ScopedOperation>(),
            size_of::<Option<ScopedOperation>>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<u32>(),
            size_of::<Result<(), PreparedHostCopyError>>(),
            size_of::<Result<&Array, PreparedHostCopyError>>(),
            size_of::<Result<ImmutableHostTransferBuffer, PreparedHostCopyError>>(),
            size_of::<(&mut Self, &Array, &Stream, &OriginalScopeObserver)>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)?
            .checked_add(OperationEvent::control_bytes()?)
    }
    /// Shared store worker controls and one direct C output-handle Graph extent.
    /// The worker may compact a completed strided source before writing Host.
    pub fn original_layout(rank: usize, dtype: Dtype) -> Option<(usize, usize)> {
        let (mut controls, mut extent) = (0, 0);
        let valid = unsafe {
            safemlx_sys::mlx_original_host_store_layout(
                rank,
                dtype.into(),
                &mut controls,
                &mut extent,
            )
        };
        valid.then_some((controls, extent))
    }
    /// Submit exactly once under the supplied active original role. The caller
    /// authenticates the canonical source and retains the returned root in its
    /// existing final completion union before any subsequent fallible work.
    pub fn submit(
        &mut self,
        source: &Array,
        stream: &Stream,
        observer: &OriginalScopeObserver,
    ) -> Result<&Array, PreparedHostCopyError> {
        if self.attempted {
            return Err(PreparedHostCopyError::State);
        }
        self.attempted = true;
        let buffer = self.buffer.as_ref().ok_or(PreparedHostCopyError::State)?;
        let destination = buffer
            .prepared_source
            .as_ref()
            .ok_or(PreparedHostCopyError::State)?;
        let event = ScopedOperation::for_observer(observer.clone())?;
        let mut raw = safemlx_sys::mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        let result = runtime_lock::try_retire(|| {
            event.check(unsafe {
                safemlx_sys::mlx_copy_to_prepared_host_operation(
                    &mut raw,
                    event.raw,
                    source.as_ptr(),
                    destination.raw(),
                    stream.as_ptr(),
                )
            })
        })
        .unwrap_or_else(|| Err(event.observer.error(10)));
        // Keep even a failed native attempt. Native recovery retains submitted
        // work; the destination cannot be retried or exposed after failure.
        self.event = Some(event.into());
        if !raw.ctx.is_null() {
            // SAFETY: the native producer transfers this unique prepared shell.
            self.output = Some(unsafe { Array::from_ptr(raw) });
        }
        if let Err(cause) = result {
            self.failed = true;
            return Err(cause.into());
        }
        self.output.as_ref().ok_or(PreparedHostCopyError::State)
    }
    /// Host-wait the same copy completion without publishing its destination.
    /// A polling/waiting error never certifies completion or releases a prefix.
    pub fn synchronize(&mut self) -> Result<(), PreparedHostCopyError> {
        if self.failed || self.output.is_none() || self.buffer.is_none() {
            return Err(PreparedHostCopyError::State);
        }
        if let Err(cause) = self
            .event
            .as_ref()
            .ok_or(PreparedHostCopyError::State)?
            .synchronize()
        {
            self.failed = true;
            return Err(cause.into());
        }
        self.completed = true;
        Ok(())
    }
    /// Move completed immutable Host storage to canonical publication. The
    /// native output/event remain retained here until the enclosing role retires.
    pub fn take_completed(&mut self) -> Result<ImmutableHostTransferBuffer, PreparedHostCopyError> {
        if self.failed || !self.completed {
            return Err(PreparedHostCopyError::State);
        }
        self.buffer
            .take()
            .map(HostTransferBuffer::freeze)
            .ok_or(PreparedHostCopyError::State)
    }
    /// Borrow the actual output descriptor without transferring its authority.
    pub fn output(&self) -> Option<&Array> {
        self.output.as_ref()
    }
}
