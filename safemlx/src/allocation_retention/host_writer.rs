//! Exclusive, unsubmitted host payload lending to a file-I/O worker.
use super::{OwnedNode, RetiredOwner, destroy, enqueue, take_owner};
use crate::{HostTransferBuffer, PreparedInputCause};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    ptr,
    thread::{self, ThreadId},
};

/// An exclusive prepared host buffer that may cross a worker boundary before
/// publication. Only byte access is available away from its creating thread.
/// A worker-side drop queues the already-paid owner for ordinary host retirement.
pub struct PreparedHostTransferWriter {
    node: Option<Box<OwnedNode<HostTransferBuffer>>>,
    creator: ThreadId,
    bytes: *mut u8,
    len: usize,
}

// SAFETY: construction uses either the prepared-source constructor with zero
// Array handles or a fresh ordinary buffer created inside the writer factory.
// The cached slice is uniquely owned, no native work has been submitted,
// and the worker can only mutate these bytes. Native publication is creator-only;
// a foreign-thread Drop transfers the preallocated owner to the existing queue.
// The arena's erased owner was required to be Send at its construction boundary.
unsafe impl Send for PreparedHostTransferWriter {}

impl fmt::Debug for PreparedHostTransferWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedHostTransferWriter")
            .field("bytes", &self.len)
            .finish_non_exhaustive()
    }
}
/// Typed failure while creating a fresh ordinary exclusive Host destination.
#[derive(Debug, thiserror::Error)]
pub enum OrdinaryHostWriterError {
    /// The existing ordinary constructor or mutable-byte inspection failed.
    #[error(transparent)]
    Native(#[from] crate::error::Exception),
    /// The destination's actual prepaid owner was absent.
    #[error("ordinary Host writer requires its prepared allocation owner")]
    MissingOwner,
    /// Attachment refused; the caller's slot retains the entire prepared owner.
    #[error(transparent)]
    Attachment(crate::PreparedAllocationOwnerCause),
    /// The paid retirement node could not be allocated.
    #[error("ordinary Host writer retirement allocation failed")]
    AllocationFailed,
}
impl PreparedHostTransferWriter {
    /// Creates a fresh exclusive destination and attaches its actual prepaid
    /// owner before any worker can receive writable bytes. A failed attachment
    /// restores the whole prepared owner to `owner`; success empties that slot.
    /// The owner must not retain this destination. It retires after its backing,
    /// including a worker-side panic or cancellation before publication.
    pub fn ordinary_with_prepared_owner<T: Send + 'static>(
        shape: &[i32],
        dtype: crate::Dtype,
        owner: &mut Option<crate::PreparedAllocationOwner<T>>,
    ) -> Result<Self, OrdinaryHostWriterError> {
        if owner.is_none() {
            return Err(OrdinaryHostWriterError::MissingOwner);
        }
        let writer = Self::ordinary(shape, dtype)?;
        let prepared = owner.take().expect("checked prepared destination owner");
        let raw = writer
            .node
            .as_ref()
            .expect("fresh exclusive writer")
            .owner
            .raw;
        match prepared.attach_host(raw) {
            Ok(()) => Ok(writer),
            Err(error) => {
                let (cause, prepared) = error.into_parts();
                *owner = Some(prepared);
                Err(OrdinaryHostWriterError::Attachment(cause))
            }
        }
    }
    /// Same constructor and nonallocating attachment call transports. The two
    /// prepared owner nodes and physical Host storage are separate sources.
    pub fn ordinary_with_prepared_owner_control_bytes<T: Send + 'static>(
        rank: usize,
    ) -> Option<usize> {
        let frames = [
            Self::ordinary_control_bytes(rank)?,
            crate::PreparedAllocationOwner::<T>::host_attachment_control_bytes()?,
            size_of::<(
                &[i32],
                crate::Dtype,
                &mut Option<crate::PreparedAllocationOwner<T>>,
            )>(),
            size_of::<Result<Self, OrdinaryHostWriterError>>(),
            size_of::<crate::PreparedAllocationOwner<T>>(),
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Creates a new ordinary Transfer buffer with no aliases or submitted work,
    /// then lends its exclusive bytes to the existing worker handoff. The caller
    /// supplies ordinary allocation admission; this method grants no source role.
    /// Publication and native destruction remain confined to the creating thread.
    pub fn ordinary(shape: &[i32], dtype: crate::Dtype) -> Result<Self, OrdinaryHostWriterError> {
        let mut buffer =
            HostTransferBuffer::new(shape, dtype, crate::HostTransferPolicy::Transfer)?;
        let bytes = buffer.as_bytes_mut()?;
        bytes.fill(0);
        let (data, len) = (bytes.as_mut_ptr(), bytes.len());
        Self::from_exclusive_buffer(buffer, data, len)
            .map_err(|_| OrdinaryHostWriterError::AllocationFailed)
    }
    /// Same fresh constructor, byte inspection and retirement-node transports.
    /// Observed native constructor controls and physical backing are separate.
    pub fn ordinary_control_bytes(rank: usize) -> Option<usize> {
        let frames = [
            Self::control_bytes()?,
            HostTransferBuffer::ordinary_constructor_control_bytes(rank)?,
            HostTransferBuffer::ordinary_mutable_bytes_control_bytes()?,
            size_of::<(&[i32], crate::Dtype)>(),
            size_of::<Result<Self, OrdinaryHostWriterError>>(),
            size_of::<OrdinaryHostWriterError>(),
            size_of::<(*mut u8, usize)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Node allocation and the exact exclusive handoff/control representations.
    /// The caller also pays the underlying prepared host constructor's census.
    pub fn control_bytes() -> Option<usize> {
        let controls = [
            Layout::new::<OwnedNode<HostTransferBuffer>>().size(),
            size_of::<Self>(),
            size_of::<Option<Box<OwnedNode<HostTransferBuffer>>>>(),
            size_of::<Result<Self, PreparedInputCause>>(),
            size_of::<Result<HostTransferBuffer, Self>>(),
            size_of::<thread::Thread>(),
            size_of::<ThreadId>(),
            size_of::<(*mut u8, usize)>(),
            size_of::<&mut [u8]>(),
            size_of::<Layout>(),
            size_of::<*mut OwnedNode<HostTransferBuffer>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
    // Only PreparedHostTransferPlan::construct_writer calls this after checking
    // zero array handles. No public mutable HostTransferBuffer can enter here.
    pub(crate) fn new(mut buffer: HostTransferBuffer) -> Result<Self, PreparedInputCause> {
        let bytes = buffer
            .prepared_source
            .as_mut()
            .ok_or(PreparedInputCause::Invalid)?
            .bytes_mut();
        let (data, len) = (bytes.as_mut_ptr(), bytes.len());
        Self::from_exclusive_buffer(buffer, data, len)
    }
    // Both callers have established fresh exclusive ownership before obtaining
    // this exact live allocation's bytes. No general buffer conversion is public.
    fn from_exclusive_buffer(
        buffer: HostTransferBuffer,
        data: *mut u8,
        len: usize,
    ) -> Result<Self, PreparedInputCause> {
        let layout = Layout::new::<OwnedNode<HostTransferBuffer>>();
        // SAFETY: this exact nonzero layout is initialized before Box ownership.
        let raw = unsafe { std::alloc::alloc(layout) }.cast::<OwnedNode<HostTransferBuffer>>();
        if raw.is_null() {
            return Err(PreparedInputCause::AllocationFailed);
        }
        let node = unsafe {
            raw.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<HostTransferBuffer>,
                },
                owner: buffer,
            });
            Box::from_raw(raw)
        };
        Ok(Self {
            node: Some(node),
            creator: thread::current().id(),
            bytes: data,
            len,
        })
    }
    /// Borrow the final destination without native calls, locks or registration.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the private node keeps the exact allocation alive; only this
        // exclusive writer can access it, and the returned loan cannot escape.
        unsafe { std::slice::from_raw_parts_mut(self.bytes, self.len) }
    }
    /// Return native ownership only on the creating thread. A wrong-thread
    /// attempt returns the unchanged writer, including all buffer custody.
    pub fn try_return(mut self) -> Result<HostTransferBuffer, Self> {
        if thread::current().id() != self.creator {
            return Err(self);
        }
        Ok(take_owner(self.node.take().expect("exclusive host writer")))
    }
}
impl Drop for PreparedHostTransferWriter {
    fn drop(&mut self) {
        let Some(node) = self.node.take() else {
            return;
        };
        if thread::current().id() == self.creator {
            // Normal ready eviction really retires backing before the caller
            // admits another window under the shared live-capacity counter.
            drop(take_owner(node));
        } else {
            // SAFETY: transfer this unique, fully initialized preallocated node
            // exactly once. No native destructor or arbitrary owner Drop runs
            // on the I/O worker, including cancellation, panic and detachment.
            unsafe { enqueue(Box::into_raw(node).cast()) };
        }
    }
}

#[cfg(test)]
mod tests;
