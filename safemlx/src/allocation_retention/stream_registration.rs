//! Exact CPU stream registration; thread/event/queue readiness is separate.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::{utils::runtime_lock, Stream};
use std::{
    alloc::Layout,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
};

/// Fixed constructor refusal, with no formatted native error allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum StreamRegistrationCause {
    /// The actual constructor/layout ABI is not qualified.
    #[error("stream registration layout is unqualified")]
    UnknownLayout,
    /// The initialization context or arguments are invalid.
    #[error("invalid stream registration context")]
    Invalid,
    /// Another native operation holds the required loan.
    #[error("stream registration is busy")]
    Busy,
    /// An actual owner-node, wrapper or registration allocation failed.
    #[error("stream registration allocation failed")]
    AllocationFailed,
    /// The wrapper no longer names this exact admitted constructor token.
    #[error("stream registration source identity mismatch")]
    IdentityMismatch,
    /// No further globally unique scalar stream index fits.
    #[error("stream index supply exhausted")]
    IdentityExhausted,
    /// The registry synchronization constructor/loan failed.
    #[error("stream registry construction failed")]
    ConstructionFailed,
    /// The retained layout changed before construction.
    #[error("stream registration layout changed")]
    LayoutChanged,
}
fn cause(status: u32) -> StreamRegistrationCause {
    use StreamRegistrationCause::*;
    match status {
        1 => UnknownLayout,
        3 => Busy,
        4 => AllocationFailed,
        8 => IdentityMismatch,
        9 => IdentityExhausted,
        10 => ConstructionFailed,
        11 => LayoutChanged,
        _ => Invalid,
    }
}
/// The fixed refusal followed by the exact untransferred owner.
pub struct StreamRegistrationError<T> {
    cause: StreamRegistrationCause,
    owner: T,
}
impl<T> StreamRegistrationError<T> {
    /// The actual refusal stage.
    pub fn cause(&self) -> StreamRegistrationCause {
        self.cause
    }
    /// The exact retained owner; no new source allowance is minted.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the original owner after disposing any prepared header first.
    pub fn into_parts(self) -> (StreamRegistrationCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for StreamRegistrationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamRegistrationError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for StreamRegistrationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for StreamRegistrationError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// One module registry and its publication/guard storage. This is priced once,
/// independently of each new entry and of loaded-constructor qualification.
#[derive(Clone, Copy, Debug)]
pub struct StreamRegistrationStaticLayout {
    /// Exact registry object, compiler guard and publication extents.
    pub bytes: usize,
    /// Whether the compiled ABI qualifies those static extents.
    pub qualified: bool,
}
/// Pure static facts; no stream, registry or thread is initialized.
pub fn stream_registration_static_layout() -> StreamRegistrationStaticLayout {
    let mut native = safemlx_sys::mlx_stream_registration_static_layout::default();
    // SAFETY: initialized fixed output; query inspects owning types only.
    unsafe { safemlx_sys::mlx_stream_registration_static_layout_for(&mut native) };
    StreamRegistrationStaticLayout {
        bytes: native.bytes,
        qualified: native.qualified == 1,
    }
}
/// Exact producer layout retained before the caller's source comparison.
pub struct CpuStreamRegistrationLayout<T> {
    native: safemlx_sys::mlx_stream_registration_layout,
    controls: usize,
    owner: PhantomData<fn() -> T>,
}
impl<T> Copy for CpuStreamRegistrationLayout<T> {}
impl<T> Clone for CpuStreamRegistrationLayout<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for CpuStreamRegistrationLayout<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuStreamRegistrationLayout")
            .field("native", &self.native)
            .field("controls", &self.controls)
            .finish()
    }
}
impl<T> CpuStreamRegistrationLayout<T> {
    /// Exact native entry including its inline global CPU encoder.
    pub fn registration_bytes(&self) -> usize {
        self.native.object_bytes
    }
    /// Exact separately owned C Stream wrapper.
    pub fn wrapper_bytes(&self) -> usize {
        self.native.wrapper_bytes
    }
    /// Fixed query/constructor/result/retirement controls.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Dynamic constructor contribution; excludes the separately priced module
    /// statics, CPU worker startup, later task storage, events and GPU queues.
    pub fn required_bytes(&self) -> Option<usize> {
        self.native
            .object_bytes
            .checked_add(self.native.wrapper_bytes)?
            .checked_add(mem::size_of::<OwnedNode<T>>())?
            .checked_add(self.controls)
    }
}
/// A real new CPU stream whose registration/encoder owns original birth custody.
/// It does not certify a prepared CPU worker or grant submission authority.
pub struct RegisteredCpuStream {
    stream: ManuallyDrop<Stream>,
    pub(super) owner: *mut c_void,
}
// SAFETY: the independent Stream wrapper may move threads. The constructor
// token is compared only under native synchronization and never dereferenced;
// its native process registration retains the actual allocation independently.
unsafe impl Send for RegisteredCpuStream {}
impl fmt::Debug for RegisteredCpuStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredCpuStream")
            .finish_non_exhaustive()
    }
}
impl RegisteredCpuStream {
    /// Borrow for existing operations. The independent C wrapper is immutable
    /// while borrowed; unsafe code must not mutate it through Stream::as_ptr.
    pub fn as_stream(&self) -> &Stream {
        &self.stream
    }
    /// Authenticate the actual registration's retained constructor token. This
    /// still grants neither worker readiness nor a new source allowance.
    pub fn try_borrow(&self) -> Result<(), StreamRegistrationCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(StreamRegistrationCause::Busy);
        };
        // SAFETY: the token came from this constructor and remains native-owned.
        let status = unsafe {
            safemlx_sys::mlx_stream_registration_borrow(self.stream.c_stream, self.owner)
        };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
}
impl Drop for RegisteredCpuStream {
    fn drop(&mut self) {
        // SAFETY: exact independent C Stream allocation. Scalar deletion has
        // no runtime entry. The process node still retains its source custody.
        unsafe { safemlx_sys::mlx_stream_copy_free(self.stream.c_stream) };
    }
}
/// One preallocated custody node, transferred only with actual native creation.
pub struct PreparedCpuStream<T: Send + 'static> {
    layout: CpuStreamRegistrationLayout<T>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedCpuStream<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCpuStream")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedCpuStream<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedCpuStream<T> {
    /// Query before admission or any owner allocation. An unsupported producer
    /// returns UnknownLayout; observed capacity and free memory are irrelevant.
    pub fn layout() -> Result<CpuStreamRegistrationLayout<T>, StreamRegistrationCause> {
        let mut native = safemlx_sys::mlx_stream_registration_layout::default();
        // SAFETY: initialized fixed transport; query performs no initialization.
        let status = unsafe { safemlx_sys::mlx_stream_registration_layout_for(&mut native) };
        if status != 0 {
            return Err(cause(status));
        }
        let controls = [
            native.controls,
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<CpuStreamRegistrationLayout<T>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Result<CpuStreamRegistrationLayout<T>, StreamRegistrationCause>>(),
            mem::size_of::<safemlx_sys::mlx_stream_registration_layout>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<RegisteredCpuStream>(),
            mem::size_of::<Result<Self, StreamRegistrationError<T>>>(),
            mem::size_of::<Result<RegisteredCpuStream, StreamRegistrationError<Self>>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<safemlx_sys::mlx_stream>(),
            mem::size_of::<u32>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            mem::size_of::<T>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(mem::size_of_val(&controls), usize::checked_add)
            .ok_or(StreamRegistrationCause::UnknownLayout)?;
        let layout = CpuStreamRegistrationLayout {
            native,
            controls,
            owner: PhantomData,
        };
        layout
            .required_bytes()
            .ok_or(StreamRegistrationCause::UnknownLayout)?;
        Ok(layout)
    }
    /// Allocate the exact owner node after the caller protects the retained
    /// layout with its source admission. Failure returns the intact source.
    pub fn with_layout(
        layout: CpuStreamRegistrationLayout<T>,
        owner: T,
    ) -> Result<Self, StreamRegistrationError<T>> {
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: exact allocation, initialized before Box adoption.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(StreamRegistrationError {
                cause: StreamRegistrationCause::AllocationFailed,
                owner,
            });
        }
        let node = unsafe {
            pointer.write(OwnedNode {
                retired: RetiredOwner {
                    next: std::ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(pointer)
        };
        Ok(Self {
            layout,
            node: Some(node),
        })
    }
    /// Borrow the exact untransferred owner.
    pub fn owner(&self) -> &T {
        &self.node.as_ref().expect("live stream initializer").owner
    }
    /// Dispose the outer node before returning its original owner.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("live stream initializer"))
    }
    /// Construct one new registration. Every refusal keeps this exact node;
    /// success keeps custody in the native process node through its final free.
    pub fn try_initialize(mut self) -> Result<RegisteredCpuStream, StreamRegistrationError<Self>> {
        let mut raw = safemlx_sys::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        let status;
        let owner;
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(StreamRegistrationError {
                    cause: StreamRegistrationCause::Busy,
                    owner: self,
                });
            };
            owner = (&mut **self.node.as_mut().expect("live stream initializer")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: failure does not adopt owner or change output. Success
            // owns both native entry and independent wrapper, without callbacks.
            status = unsafe {
                safemlx_sys::mlx_stream_register_cpu(
                    &mut raw,
                    self.layout.native,
                    owner,
                    Some(retire),
                )
            };
            if status == 0 {
                let _ = Box::into_raw(self.node.take().expect("live stream initializer"));
            }
        }
        if status == 0 {
            Ok(RegisteredCpuStream {
                stream: ManuallyDrop::new(Stream { c_stream: raw }),
                owner,
            })
        } else {
            Err(StreamRegistrationError {
                cause: cause(status),
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests;
