//! One admitted allocator constructor, using the existing unlocked owner queue.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::{utils::runtime_lock, PreparedInputRuntime};
use std::{alloc::Layout, ffi::c_void, fmt, mem, ptr};

/// Fixed refusal. None of these outcomes creates allocator coverage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum InputAllocatorCause {
    /// The owning producer layout is not qualified for this implementation.
    #[error("input allocator layout is unqualified")]
    UnknownLayout,
    /// Invalid native output or an active incompatible submission context.
    #[error("invalid input allocator initialization context")]
    Invalid,
    /// No wait, housekeeping, constructor retry or ownership handoff occurred.
    #[error("input allocator initialization is busy")]
    Busy,
    /// A concrete owner/object constructor failed; its prefix remains owned.
    #[error("input allocator initialization allocation failed")]
    AllocationFailed,
    /// This operation never creates the required native Device.
    #[error("input allocator initialization requires an existing Device")]
    MissingDevice,
    /// The existing allocator was born without this admission.
    #[error("input allocator has an ordinary predecessor")]
    OrdinaryPredecessor,
    /// A different admitted initializer already owns the actual singleton.
    #[error("input allocator already has an admitted owner")]
    AlreadyInitialized,
    /// The private completed witness does not match the actual native singleton.
    #[error("input allocator initialization identity mismatch")]
    IdentityMismatch,
    /// The checked nonreusing initializer identity supply is exhausted.
    #[error("input allocator initialization identity supply exhausted")]
    IdentityExhausted,
}
fn cause(status: u32) -> InputAllocatorCause {
    match status {
        1 => InputAllocatorCause::UnknownLayout,
        3 => InputAllocatorCause::Busy,
        4 => InputAllocatorCause::AllocationFailed,
        5 => InputAllocatorCause::MissingDevice,
        6 => InputAllocatorCause::OrdinaryPredecessor,
        7 => InputAllocatorCause::AlreadyInitialized,
        8 => InputAllocatorCause::IdentityMismatch,
        9 => InputAllocatorCause::IdentityExhausted,
        _ => InputAllocatorCause::Invalid,
    }
}
/// Cause followed by exact unchanged preparation/custody.
pub struct InputAllocatorError<T> {
    cause: InputAllocatorCause,
    owner: T,
}
impl<T> InputAllocatorError<T> {
    /// Converts the retained owner while preserving the exact refusal cause.
    /// A caller may cancel an unadopted preparation to free its shell and keep
    /// the same source custody in the returned error.
    pub fn map_owner<U>(self, map: impl FnOnce(T) -> U) -> InputAllocatorError<U> {
        InputAllocatorError {
            cause: self.cause,
            owner: map(self.owner),
        }
    }

    /// Fixed refusal, without a formatted native handler.
    pub fn cause(&self) -> InputAllocatorCause {
        self.cause
    }
    /// Borrows the actual retained preparation.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Returns the refusal and the intact owner.
    pub fn into_parts(self) -> (InputAllocatorCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for InputAllocatorError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputAllocatorError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for InputAllocatorError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for InputAllocatorError<T> {}

/// Exact disjoint constructor and fixed-storage contributions. This contains no
/// Device/accounting certificate, runtime Rc, source payload or request grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputAllocatorLayout {
    /// Actual singleton coordination/object storage with static lifetime.
    pub static_bytes: usize,
    /// Actual heap object size (zero for the CPU in-place object).
    pub native_object_bytes: usize,
    /// Actual retained queue node, including its accounting payload.
    pub rust_node_bytes: usize,
    /// Named constructor/result/failure/retirement controls.
    pub control_bytes: usize,
    /// Metal borrows an existing Device; its ownership is a separate prerequisite.
    pub requires_device: bool,
}
impl InputAllocatorLayout {
    /// Dynamic constructor requirement. Static storage belongs to the baseline.
    pub fn required_bytes(self) -> Option<usize> {
        self.native_object_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}
struct Construction {
    raw: safemlx_sys::mlx_prepared_input_runtime,
    identity: u64,
    status: u32,
}
fn empty_runtime() -> safemlx_sys::mlx_prepared_input_runtime {
    safemlx_sys::mlx_prepared_input_runtime {
        allocator: ptr::null_mut(),
        page_size: 0,
        maximum: 0,
        storage_kind: 0,
        controls: 0,
        placement: safemlx_sys::mlx_memory_placement {
            kind: 0,
            device: -1,
            device_count: 0,
        },
    }
}

/// Private successful initializer identity. It contains no raw allocator pointer
/// or neutral funding authority. Borrowing never initializes an ordinary fallback.
/// The native singleton retains the actual owner independently of this token.
#[derive(Debug)]
pub struct InitializedInputAllocator {
    identity: u64,
    requires_device: bool,
}
impl InitializedInputAllocator {
    /// Device lifetime/accounting is still a separate prerequisite on Metal.
    pub fn requires_device(&self) -> bool {
        self.requires_device
    }
    /// Exact per-call native and safe loan controls, without entering the runtime
    /// or constructing an allocator. Singleton ownership is priced separately.
    pub fn borrow_control_bytes() -> Option<usize> {
        // SAFETY: pure sizeof query; no runtime entry, owner or initialization.
        let native = unsafe { safemlx_sys::mlx_input_allocator_borrow_controls() };
        if native == 0 {
            return None;
        }
        let parts = [
            native,
            mem::size_of::<&Self>(),
            mem::size_of::<safemlx_sys::mlx_prepared_input_runtime>(),
            mem::size_of::<u32>(),
            mem::size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<PreparedInputRuntime>(),
            mem::size_of::<Result<PreparedInputRuntime, InputAllocatorCause>>(),
            mem::size_of::<InputAllocatorCause>(),
        ];
        parts
            .into_iter()
            .try_fold(mem::size_of_val(&parts), usize::checked_add)
    }
    /// One no-housekeeping immediate loan creates a current-thread borrower,
    /// including inside an admitted role. Native validation only try-locks the
    /// existing singleton and matches its immutable admitted birth identity;
    /// initialization still requires an empty submission context. This loan
    /// supplies no buffer grant and cannot initialize an ordinary fallback.
    pub fn try_borrow_runtime(&self) -> Result<PreparedInputRuntime, InputAllocatorCause> {
        let mut raw = empty_runtime();
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(InputAllocatorCause::Busy);
        };
        // SAFETY: private identity came only from successful initialization. The
        // native singleton outlives this token and verifies its completed birth.
        let status = unsafe { safemlx_sys::mlx_input_allocator_borrow(&mut raw, self.identity) };
        if status == 0 {
            Ok(PreparedInputRuntime::from_initialized(raw))
        } else {
            Err(cause(status))
        }
    }
}

/// One prepared retained owner. The caller supplies genuine accepted custody;
/// this low-level mechanism does not itself issue a memory grant. No native
/// constructor runs before try_initialize, and every refusal returns this owner.
pub struct PreparedInputAllocator<T: Send + 'static> {
    layout: InputAllocatorLayout,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedInputAllocator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedInputAllocator")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedInputAllocator<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedInputAllocator<T> {
    /// Pure owning layout; no Device, queue drain, runtime entry or construction.
    pub fn layout() -> Result<InputAllocatorLayout, InputAllocatorCause> {
        let mut native = safemlx_sys::mlx_input_allocator_layout::default();
        // SAFETY: pure query writes one initialized repr(C) structure.
        unsafe { safemlx_sys::mlx_input_allocator_layout_for(&mut native) };
        if native.qualified != 1 {
            return Err(InputAllocatorCause::UnknownLayout);
        }
        let parts = [
            // The layout query is reached twice sequentially. The first call's
            // frames retire before admission; no two queries are live together.
            mem::size_of::<InputAllocatorLayout>(),
            mem::size_of::<Result<InputAllocatorLayout, InputAllocatorCause>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<Construction>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<InitializedInputAllocator>(),
            mem::size_of::<PreparedInputRuntime>(),
            mem::size_of::<Result<Self, InputAllocatorError<T>>>(),
            mem::size_of::<Result<InitializedInputAllocator, InputAllocatorError<Self>>>(),
            mem::size_of::<Result<PreparedInputRuntime, InputAllocatorCause>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<T>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            native.controls,
        ];
        let control_bytes = parts
            .into_iter()
            .try_fold(mem::size_of_val(&parts), usize::checked_add)
            .ok_or(InputAllocatorCause::UnknownLayout)?;
        Ok(InputAllocatorLayout {
            static_bytes: native.static_bytes,
            native_object_bytes: native.object_bytes,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
            requires_device: native.requires_device != 0,
        })
    }
    /// Allocates only the exact queue node, after the owning caller's comparison.
    pub fn try_new(owner: T) -> Result<Self, InputAllocatorError<T>> {
        let layout = match Self::layout() {
            Ok(layout) => layout,
            Err(cause) => return Err(InputAllocatorError { cause, owner }),
        };
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: matching layout, initialized before Box construction.
        let node = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(InputAllocatorError {
                cause: InputAllocatorCause::AllocationFailed,
                owner,
            });
        }
        let node = unsafe {
            node.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(node)
        };
        Ok(Self {
            layout,
            node: Some(node),
        })
    }
    /// The same unconsumed owner; no node address is exposed.
    pub fn owner(&self) -> &T {
        &self.node.as_ref().expect("live initializer").owner
    }
    /// Cancel preparation, freeing its actual shell before returning custody.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("live initializer"))
    }
    /// Success alone transfers custody. No polling, ordinary hooks or retries.
    pub fn try_initialize(
        mut self,
    ) -> Result<InitializedInputAllocator, InputAllocatorError<Self>> {
        let mut construction = Construction {
            raw: empty_runtime(),
            identity: 0,
            status: 2,
        };
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(InputAllocatorError {
                    cause: InputAllocatorCause::Busy,
                    owner: self,
                });
            };
            let owner =
                (&mut **self.node.as_mut().expect("live initializer") as *mut OwnedNode<T>).cast();
            // SAFETY: native consumes the preallocated node only on success;
            // every other status preserves the exact owner and zero output.
            construction.status = unsafe {
                safemlx_sys::mlx_input_allocator_initialize(
                    &mut construction.raw,
                    &mut construction.identity,
                    owner,
                    Some(retire),
                )
            };
            if construction.status == 0 {
                let _ = Box::into_raw(self.node.take().expect("live initializer"));
            }
        }
        if construction.status == 0 {
            Ok(InitializedInputAllocator {
                identity: construction.identity,
                requires_device: self.layout.requires_device,
            })
        } else {
            Err(InputAllocatorError {
                cause: cause(construction.status),
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    #[derive(Debug)]
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn shared_input_initializer_busy_preserves_the_exact_prepared_node_and_owner() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let owner = Owner(dropped.clone());
        let layout = PreparedInputAllocator::<Owner>::layout();
        if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
            assert!(layout.is_ok(), "{layout:?}");
        }
        if matches!(layout, Err(InputAllocatorCause::UnknownLayout)) {
            let error = PreparedInputAllocator::try_new(owner).unwrap_err();
            assert_eq!(error.cause(), InputAllocatorCause::UnknownLayout);
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            drop(error);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            return;
        }
        let prepared = PreparedInputAllocator::try_new(owner).unwrap();
        let pointer = &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>;
        let (entered, wait_entered) = mpsc::channel();
        let (release, wait_release) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _loan = runtime_lock::enter();
            entered.send(()).unwrap();
            let _ = wait_release.recv();
        });
        wait_entered.recv().unwrap();
        let result = prepared.try_initialize();
        // Always release before assertions so a test panic cannot strand a loan.
        release.send(()).unwrap();
        worker.join().unwrap();
        let error = result.unwrap_err();
        assert_eq!(error.cause(), InputAllocatorCause::Busy);
        assert_eq!(
            &**error.owner().node.as_ref().unwrap() as *const OwnedNode<Owner>,
            pointer
        );
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        let (_, prepared) = error.into_parts();
        drop(prepared.into_owner());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
