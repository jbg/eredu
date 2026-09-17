//! Actual admitted GPU stream/encoder birth; later work and platform allocation remain separate.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::{utils::runtime_lock, InitializedMetalDevice, InitializedScheduler, Stream};
use std::{
    alloc::Layout,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
};

/// Fixed constructor refusal, with no formatted native error allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GpuStreamRegistrationCause {
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
    /// Metal refused the real command queue constructor.
    #[error("GPU command queue creation failed")]
    QueueFailed,
    /// Metal refused the real initial command buffer constructor.
    #[error("GPU command buffer creation failed")]
    BufferFailed,
    /// No further globally unique scalar stream index fits.
    #[error("stream index supply exhausted")]
    IdentityExhausted,
    /// The registry synchronization constructor/loan failed.
    #[error("stream registry construction failed")]
    ConstructionFailed,
    /// The actual worker is failed, blocked or stopping; readiness is unavailable.
    #[error("native stream execution is fenced")]
    ExecutionFailed,
    /// The retained layout changed before construction.
    #[error("stream registration layout changed")]
    LayoutChanged,
}
fn cause(status: u32) -> GpuStreamRegistrationCause {
    use GpuStreamRegistrationCause::*;
    match status {
        1 => UnknownLayout,
        3 => Busy,
        4 => AllocationFailed,
        5 => QueueFailed,
        6 => BufferFailed,
        8 => IdentityMismatch,
        9 => IdentityExhausted,
        10 => ConstructionFailed,
        11 => LayoutChanged,
        12 => ExecutionFailed,
        _ => Invalid,
    }
}
/// The fixed refusal followed by the exact untransferred owner.
pub struct GpuStreamRegistrationError<T> {
    cause: GpuStreamRegistrationCause,
    owner: T,
}
impl<T> GpuStreamRegistrationError<T> {
    /// The actual refusal stage.
    pub fn cause(&self) -> GpuStreamRegistrationCause {
        self.cause
    }
    /// The exact retained owner; no new source allowance is minted.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the original owner after disposing any prepared header first.
    pub fn into_parts(self) -> (GpuStreamRegistrationCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for GpuStreamRegistrationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuStreamRegistrationError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for GpuStreamRegistrationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for GpuStreamRegistrationError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Private scalar identities of admitted Device and Scheduler owners.
#[derive(Debug, Clone, Copy)]
pub struct GpuStreamTarget {
    native: safemlx_sys::mlx_gpu_stream_target,
}
impl GpuStreamTarget {
    /// Authenticate actual constructors before source admission. No stream is created.
    pub fn for_initialized(
        device: &InitializedMetalDevice,
        scheduler: &InitializedScheduler,
    ) -> Result<Self, GpuStreamRegistrationCause> {
        device.try_borrow().map_err(|cause| {
            if cause == crate::MetalDeviceCause::Busy {
                GpuStreamRegistrationCause::Busy
            } else {
                GpuStreamRegistrationCause::IdentityMismatch
            }
        })?;
        scheduler.try_borrow().map_err(|cause| {
            if cause == crate::SchedulerCause::Busy {
                GpuStreamRegistrationCause::Busy
            } else {
                GpuStreamRegistrationCause::IdentityMismatch
            }
        })?;
        Ok(Self {
            native: safemlx_sys::mlx_gpu_stream_target {
                device_identity: device.identity,
                scheduler_identity: scheduler.identity,
            },
        })
    }
}
/// Exact producer layout retained before the caller's source comparison.
pub struct GpuStreamRegistrationLayout<T> {
    native: safemlx_sys::mlx_gpu_stream_registration_layout,
    controls: usize,
    owner: PhantomData<fn() -> T>,
}
impl<T> Copy for GpuStreamRegistrationLayout<T> {}
impl<T> Clone for GpuStreamRegistrationLayout<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for GpuStreamRegistrationLayout<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuStreamRegistrationLayout")
            .field("native", &self.native)
            .field("controls", &self.controls)
            .finish()
    }
}
impl<T> GpuStreamRegistrationLayout<T> {
    /// Six exact native owning request bytes and alignments.
    pub fn requests(&self) -> [(usize, usize); 6] {
        std::array::from_fn(|i| (self.native.bytes[i], self.native.alignments[i]))
    }
    /// Opaque platform constructor populations, not claimed as host byte bounds.
    pub fn platform_objects(&self) -> (usize, usize, usize) {
        (
            self.native.platform_queues,
            self.native.platform_buffers,
            self.native.autorelease_pools,
        )
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
    /// statics, later task/receipt storage and platform-private queue/buffer allocation.
    pub fn required_bytes(&self) -> Option<usize> {
        self.native
            .bytes
            .into_iter()
            .try_fold(0usize, usize::checked_add)?
            .checked_add(self.native.wrapper_bytes)?
            .checked_add(mem::size_of::<OwnedNode<T>>())?
            .checked_add(self.controls)
    }
}
/// A real new GPU stream whose encoder/control retains original birth custody.
/// Later work, platform resources and mutable diagnostics remain separate.
pub struct RegisteredGpuStream {
    stream: ManuallyDrop<Stream>,
    owner: *mut c_void,
    target: GpuStreamTarget,
}
// SAFETY: the independent Stream wrapper may move threads. The constructor
// token is compared only under native synchronization and never dereferenced;
// its native process registration retains the actual allocation independently.
unsafe impl Send for RegisteredGpuStream {}
impl fmt::Debug for RegisteredGpuStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredGpuStream")
            .finish_non_exhaustive()
    }
}
impl RegisteredGpuStream {
    /// Observe this exact admitted worker's terminal frontier without creating,
    /// submitting, progressing or waiting for work. `Busy` preserves all owners.
    /// This snapshot is not a session readiness grant: callers must separately
    /// hold exclusive session state and exclude further producers of that state.
    pub fn try_observe_idle(&self) -> Result<(), GpuStreamRegistrationCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(GpuStreamRegistrationCause::Busy);
        };
        // SAFETY: private original tokens are authenticated under native loans;
        // the runtime loan excludes host encoder mutation during observation.
        let status = unsafe {
            safemlx_sys::mlx_stream_gpu_registration_observe_idle(
                self.stream.c_stream,
                self.owner,
                self.target.native,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }

    /// Borrow for existing operations. The independent C wrapper is immutable
    /// while borrowed; unsafe code must not mutate it through Stream::as_ptr.
    pub fn as_stream(&self) -> &Stream {
        &self.stream
    }
    /// Authenticate the actual registration's retained constructor token. This
    /// still grants neither worker readiness nor a new source allowance.
    pub fn try_borrow(&self) -> Result<(), GpuStreamRegistrationCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(GpuStreamRegistrationCause::Busy);
        };
        // SAFETY: the token came from this constructor and remains native-owned.
        let status = unsafe {
            safemlx_sys::mlx_stream_gpu_registration_borrow(
                self.stream.c_stream,
                self.owner,
                self.target.native,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
}
impl Drop for RegisteredGpuStream {
    fn drop(&mut self) {
        // SAFETY: exact independent C Stream allocation. Scalar deletion has
        // no runtime entry. The process node still retains its source custody.
        unsafe { safemlx_sys::mlx_stream_copy_free(self.stream.c_stream) };
    }
}
/// One preallocated custody node, transferred only with actual native creation.
pub struct PreparedGpuStream<T: Send + 'static> {
    layout: GpuStreamRegistrationLayout<T>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedGpuStream<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedGpuStream")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedGpuStream<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedGpuStream<T> {
    /// Query before admission or any owner allocation. An unsupported producer
    /// returns UnknownLayout; observed capacity and free memory are irrelevant.
    pub fn layout() -> Result<GpuStreamRegistrationLayout<T>, GpuStreamRegistrationCause> {
        let mut native = safemlx_sys::mlx_gpu_stream_registration_layout::default();
        // SAFETY: initialized fixed transport; query performs no initialization.
        let status = unsafe { safemlx_sys::mlx_gpu_stream_registration_layout_for(&mut native) };
        if status != 0 {
            return Err(cause(status));
        }
        let controls = [
            native.controls,
            mem::size_of::<Self>(),
            mem::size_of::<GpuStreamTarget>(),
            mem::size_of::<Result<GpuStreamTarget, GpuStreamRegistrationCause>>(),
            mem::size_of::<T>(),
            mem::size_of::<GpuStreamRegistrationLayout<T>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Result<GpuStreamRegistrationLayout<T>, GpuStreamRegistrationCause>>(),
            mem::size_of::<safemlx_sys::mlx_gpu_stream_registration_layout>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<RegisteredGpuStream>(),
            mem::size_of::<Result<Self, GpuStreamRegistrationError<T>>>(),
            mem::size_of::<Result<RegisteredGpuStream, GpuStreamRegistrationError<Self>>>(),
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
            .ok_or(GpuStreamRegistrationCause::UnknownLayout)?;
        let layout = GpuStreamRegistrationLayout {
            native,
            controls,
            owner: PhantomData,
        };
        layout
            .required_bytes()
            .ok_or(GpuStreamRegistrationCause::UnknownLayout)?;
        Ok(layout)
    }
    /// Allocate the exact owner node after the caller protects the retained
    /// layout with its source admission. Failure returns the intact source.
    pub fn with_layout(
        layout: GpuStreamRegistrationLayout<T>,
        owner: T,
    ) -> Result<Self, GpuStreamRegistrationError<T>> {
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: exact allocation, initialized before Box adoption.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(GpuStreamRegistrationError {
                cause: GpuStreamRegistrationCause::AllocationFailed,
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
    pub fn try_initialize(
        mut self,
        target: GpuStreamTarget,
    ) -> Result<RegisteredGpuStream, GpuStreamRegistrationError<Self>> {
        let mut raw = safemlx_sys::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        let status;
        let owner;
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(GpuStreamRegistrationError {
                    cause: GpuStreamRegistrationCause::Busy,
                    owner: self,
                });
            };
            owner = (&mut **self.node.as_mut().expect("live stream initializer")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: failure does not adopt owner or change output. Success
            // owns both native entry and independent wrapper, without callbacks.
            status = unsafe {
                safemlx_sys::mlx_stream_register_gpu(
                    &mut raw,
                    self.layout.native,
                    target.native,
                    owner,
                    Some(retire),
                )
            };
            if status == 0 {
                let _ = Box::into_raw(self.node.take().expect("live stream initializer"));
            }
        }
        if status == 0 {
            Ok(RegisteredGpuStream {
                stream: ManuallyDrop::new(Stream { c_stream: raw }),
                owner,
                target,
            })
        } else {
            Err(GpuStreamRegistrationError {
                cause: cause(status),
                owner: self,
            })
        }
    }
}
