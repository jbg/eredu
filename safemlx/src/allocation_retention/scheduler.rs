//! One actual process Scheduler birth, independent of worker/stream construction.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::utils::runtime_lock;
use std::{alloc::Layout, ffi::c_void, fmt, mem};

/// Fixed initialization/refusal cause; no allocated native diagnostic text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SchedulerCause {
    /// The compiled layout or actual loaded constructor is not qualified.
    #[error("Scheduler producer layout is unqualified")]
    UnknownLayout,
    /// The supplied initialization context is invalid.
    #[error("invalid Scheduler initialization context")]
    Invalid,
    /// Another initializer or native operation holds the required loan.
    #[error("Scheduler initialization is busy")]
    Busy,
    /// The actual initialization node or Scheduler allocation failed.
    #[error("Scheduler initialization allocation failed")]
    AllocationFailed,
    /// An ordinary singleton already exists without admitted birth custody.
    #[error("Scheduler has an ordinary predecessor")]
    OrdinaryPredecessor,
    /// A singleton already has an admitted birth owner.
    #[error("Scheduler already has an admitted owner")]
    AlreadyInitialized,
    /// The retained identity does not name the published singleton.
    #[error("Scheduler initialization identity mismatch")]
    IdentityMismatch,
    /// No further initialization identity can be issued.
    #[error("Scheduler initialization identity supply exhausted")]
    IdentityExhausted,
    /// Native subobject construction failed; its unpublished storage was freed.
    #[error("Scheduler native constructor failed")]
    ConstructionFailed,
    /// Revalidation differs from the exact retained cold layout.
    #[error("Scheduler initialization layout changed")]
    LayoutChanged,
}
fn cause(status: u32) -> SchedulerCause {
    use SchedulerCause::*;
    match status {
        1 => UnknownLayout,
        3 => Busy,
        4 => AllocationFailed,
        6 => OrdinaryPredecessor,
        7 => AlreadyInitialized,
        8 => IdentityMismatch,
        9 => IdentityExhausted,
        10 => ConstructionFailed,
        11 => LayoutChanged,
        _ => Invalid,
    }
}
/// Fixed cause followed by the exact untransferred source/accounting owner.
pub struct SchedulerError<T> {
    cause: SchedulerCause,
    owner: T,
}
impl<T> SchedulerError<T> {
    /// Converts the retained owner while preserving the exact refusal cause.
    /// A caller may cancel an unadopted preparation to free its shell and keep
    /// the same source custody in the returned error.
    pub fn map_owner<U>(self, map: impl FnOnce(T) -> U) -> SchedulerError<U> {
        SchedulerError {
            cause: self.cause,
            owner: map(self.owner),
        }
    }

    /// The fixed initialization stage or refusal cause.
    pub fn cause(&self) -> SchedulerCause {
        self.cause
    }
    /// Borrow the original source owner retained by this error.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the fixed cause and the exact owner that was not transferred.
    pub fn into_parts(self) -> (SchedulerCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for SchedulerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for SchedulerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for SchedulerError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Fixed module and existing first-caller identity storage, priced once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerStaticLayout {
    /// Actual module objects, instruction constant and native thread-id guard.
    pub bytes: usize,
    /// Constant/static ABI qualification, independent of the loaded constructor.
    pub qualified: bool,
}
/// Read only owning static/type facts. No native thread identity or runtime is
/// initialized, and this does not check or create the dynamic Scheduler object.
pub fn scheduler_static_layout() -> SchedulerStaticLayout {
    let mut native = safemlx_sys::mlx_scheduler_initialization_static_layout::default();
    // SAFETY: initialized repr(C) output; pure fixed-static native query.
    unsafe { safemlx_sys::mlx_scheduler_initialization_static_layout_for(&mut native) };
    SchedulerStaticLayout {
        bytes: native.bytes,
        qualified: native.qualified == 1,
    }
}
/// Exact selected constructor and owner-node layout retained before comparison.
/// This describes one empty Scheduler, not worker/map/thread growth or workspace.
pub struct SchedulerLayout<T> {
    owner_type: std::marker::PhantomData<fn() -> T>,
    native: safemlx_sys::mlx_scheduler_initialization_layout,
    rust_node_bytes: usize,
    control_bytes: usize,
}
impl<T> Copy for SchedulerLayout<T> {}
impl<T> Clone for SchedulerLayout<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for SchedulerLayout<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerLayout")
            .field("native", &self.native)
            .field("rust_node_bytes", &self.rust_node_bytes)
            .field("control_bytes", &self.control_bytes)
            .finish()
    }
}
impl<T> SchedulerLayout<T> {
    /// Requested native Scheduler object bytes, including its empty containers.
    pub fn object_bytes(&self) -> usize {
        self.native.object_bytes
    }
    /// Alignment of the actual native Scheduler object.
    pub fn object_alignment(&self) -> usize {
        self.native.object_alignment
    }
    /// Requested storage for the exact retained Rust owner node.
    pub fn rust_node_bytes(&self) -> usize {
        self.rust_node_bytes
    }
    /// Named query, constructor, result and retirement controls.
    pub fn control_bytes(&self) -> usize {
        self.control_bytes
    }
    /// Complete checked contribution of this one dynamic constructor/owner.
    /// Module statics and neutral source-account storage are charged separately.
    pub fn required_bytes(&self) -> Option<usize> {
        self.native
            .object_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}
/// Private immutable process-singleton identity, without a raw mutable pointer.
#[derive(Debug)]
pub struct InitializedScheduler {
    pub(super) identity: u64,
}
impl InitializedScheduler {
    /// Validate the existing admitted singleton without creating any runtime.
    pub fn try_borrow(&self) -> Result<(), SchedulerCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(SchedulerCause::Busy);
        };
        // SAFETY: only successful construction creates this private identity.
        let status = unsafe { safemlx_sys::mlx_scheduler_initialized_borrow(self.identity) };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
}
/// One prepared source-custody node; success alone transfers it permanently.
/// The owner supplies custody, not allocation or submission authority.
pub struct PreparedScheduler<T: Send + 'static> {
    layout: SchedulerLayout<T>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedScheduler<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedScheduler")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedScheduler<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedScheduler<T> {
    /// Query the actually loaded constructor without taking the runtime lock,
    /// initializing native identity, loading a library or creating a Scheduler.
    /// Unsupported ABI/code returns UnknownLayout before an owner allocation.
    pub fn layout() -> Result<SchedulerLayout<T>, SchedulerCause> {
        let mut native = safemlx_sys::mlx_scheduler_initialization_layout::default();
        // SAFETY: initialized fixed repr(C) output. Native reads its fixed type
        // facts and qualified, directly bound, readable libc++ code only.
        let status = unsafe { safemlx_sys::mlx_scheduler_initialization_layout_for(&mut native) };
        if status != 0 {
            return Err(cause(status));
        }
        let controls = [
            native.controls,
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<SchedulerLayout<T>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Result<SchedulerLayout<T>, SchedulerCause>>(),
            mem::size_of::<safemlx_sys::mlx_scheduler_initialization_layout>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<InitializedScheduler>(),
            mem::size_of::<Result<Self, SchedulerError<T>>>(),
            mem::size_of::<Result<InitializedScheduler, SchedulerError<Self>>>(),
            mem::size_of::<Result<(), SchedulerCause>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<u64>(),
            mem::size_of::<u32>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            mem::size_of::<T>(),
        ];
        let control_bytes = controls
            .into_iter()
            .try_fold(mem::size_of_val(&controls), usize::checked_add)
            .ok_or(SchedulerCause::UnknownLayout)?;
        let layout = SchedulerLayout {
            owner_type: std::marker::PhantomData,
            native,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
        };
        layout
            .required_bytes()
            .ok_or(SchedulerCause::UnknownLayout)?;
        Ok(layout)
    }
    /// Query then allocate one node, preserving the exact owner on refusal.
    /// The caller must already protect this allocation with its admission policy.
    pub fn try_new(owner: T) -> Result<Self, SchedulerError<T>> {
        let layout = match Self::layout() {
            Ok(value) => value,
            Err(cause) => return Err(SchedulerError { cause, owner }),
        };
        Self::with_layout(layout, owner)
    }
    /// Consume the retained owner-typed layout after its source comparison.
    /// Native construction revalidates the loaded implementation before malloc.
    pub fn with_layout(layout: SchedulerLayout<T>, owner: T) -> Result<Self, SchedulerError<T>> {
        if layout.rust_node_bytes != mem::size_of::<OwnedNode<T>>() {
            return Err(SchedulerError {
                cause: SchedulerCause::Invalid,
                owner,
            });
        }
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: matching allocation; initialized before Box adopts storage.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(SchedulerError {
                cause: SchedulerCause::AllocationFailed,
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
    /// Borrow the exact original custody retained in this prepared node.
    pub fn owner(&self) -> &T {
        &self
            .node
            .as_ref()
            .expect("live Scheduler initializer")
            .owner
    }
    /// Free the unused node's outer allocation before returning its custody.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("live Scheduler initializer"))
    }
    /// Construct the singleton once, retaining every refusal in this same node.
    /// The native winner is process-lived; success never refunds its birth owner.
    pub fn try_initialize(mut self) -> Result<InitializedScheduler, SchedulerError<Self>> {
        let mut identity = 0;
        let status;
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(SchedulerError {
                    cause: SchedulerCause::Busy,
                    owner: self,
                });
            };
            let owner = (&mut **self.node.as_mut().expect("live Scheduler initializer")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: every refusal leaves node/output intact. Success retains
            // this exact preallocated node with the permanent native singleton.
            status = unsafe {
                safemlx_sys::mlx_scheduler_initialize(
                    &mut identity,
                    self.layout.native,
                    owner,
                    Some(retire),
                )
            };
            if status == 0 {
                let _ = Box::into_raw(self.node.take().expect("live Scheduler initializer"));
            }
        }
        if status == 0 {
            Ok(InitializedScheduler { identity })
        } else {
            Err(SchedulerError {
                cause: cause(status),
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
    fn scheduler_busy_retains_exact_node_then_extracts_original_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let layout = PreparedScheduler::<Owner>::layout();
        if std::env::var_os("EREDU_REQUIRE_SCHEDULER_INITIALIZATION_QUALIFICATION").is_some() {
            assert!(layout.is_ok(), "{layout:?}");
        }
        let layout = match layout {
            Ok(value) => value,
            Err(SchedulerCause::UnknownLayout) => {
                let error = PreparedScheduler::try_new(Owner(drops.clone())).unwrap_err();
                assert_eq!(error.cause(), SchedulerCause::UnknownLayout);
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                drop(error);
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                return;
            }
            Err(error) => panic!("unexpected Scheduler query: {error}"),
        };
        assert!(layout.required_bytes().unwrap() > layout.object_bytes());
        let prepared = PreparedScheduler::with_layout(layout, Owner(drops.clone())).unwrap();
        let pointer = &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>;
        let (entered, wait_entered) = mpsc::channel();
        let (release, wait_release) = mpsc::channel();
        // A disconnected channel also releases the loan during assertion unwind.
        let worker = std::thread::spawn(move || {
            let _loan = runtime_lock::enter();
            entered.send(()).unwrap();
            let _ = wait_release.recv();
        });
        wait_entered.recv().unwrap();
        let error = prepared.try_initialize().unwrap_err();
        assert_eq!(error.cause(), SchedulerCause::Busy);
        let (_, prepared) = error.into_parts();
        assert_eq!(
            &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>,
            pointer
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        worker.join().unwrap();
        let owner = prepared.into_owner();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn scheduler_static_storage_is_stable_across_dynamic_query_and_runtime_loan() {
        let fixed = scheduler_static_layout();
        let dynamic = PreparedScheduler::<()>::layout();
        if std::env::var_os("EREDU_REQUIRE_SCHEDULER_INITIALIZATION_QUALIFICATION").is_some() {
            assert!(fixed.qualified && dynamic.is_ok(), "{fixed:?}; {dynamic:?}");
        }
        assert!(fixed.bytes > 0);
        let _loan = runtime_lock::enter();
        let other = std::thread::spawn(|| {
            let same = scheduler_static_layout();
            let query = PreparedScheduler::<()>::layout();
            (
                same,
                query.map(|layout| (layout.object_bytes(), layout.control_bytes())),
            )
        })
        .join()
        .unwrap();
        assert_eq!(other.0, fixed);
        assert_eq!(
            other.1,
            dynamic.map(|layout| (layout.object_bytes(), layout.control_bytes()))
        );
    }
}
