//! Qualified process-worker startup custody; later execution remains separate.
use super::{
    destroy, retire, take_owner, InitializedScheduler, OwnedNode, RegisteredCpuStream, RetiredOwner,
};
use crate::utils::runtime_lock;
use std::{alloc::Layout, ffi::c_void, fmt, marker::PhantomData, mem};

/// Fixed startup refusal, without retaining an allocated native diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CpuWorkerCause {
    /// The actual loaded constructor or compiled layout is not qualified.
    #[error("CPU worker startup layout is unqualified")]
    UnknownLayout,
    /// The supplied initialization context is invalid.
    #[error("invalid CPU worker startup context")]
    Invalid,
    /// A native or safe runtime loan is held elsewhere.
    #[error("CPU worker startup is busy")]
    Busy,
    /// An actual owner, worker or constructor-prefix allocation failed.
    #[error("CPU worker startup allocation failed")]
    AllocationFailed,
    /// An ordinary worker already exists; it cannot acquire retrospective custody.
    #[error("CPU worker has an ordinary predecessor")]
    OrdinaryPredecessor,
    /// This stream already has an admitted worker.
    #[error("CPU worker already has an admitted owner")]
    AlreadyInitialized,
    /// The exact Scheduler, stream or worker constructor token differs.
    #[error("CPU worker source identity mismatch")]
    IdentityMismatch,
    /// Native construction failed after disposing its unpublished prefix.
    #[error("CPU worker native constructor failed")]
    ConstructionFailed,
    /// The loaded producer differs from the retained cold layout.
    #[error("CPU worker startup layout changed")]
    LayoutChanged,
    /// The actual worker is failed, blocked or stopping; readiness is unavailable.
    #[error("native stream execution is fenced")]
    ExecutionFailed,
    /// Exact native system error code, copied before its diagnostic retires.
    #[error("CPU worker system failure {value} in category {category}")]
    SystemFailure {
        /// Original error_code value (for example the pthread_create result).
        value: i32,
        /// One means the qualified generic category; zero is another category.
        category: u32,
    },
}
fn cause(result: safemlx_sys::mlx_cpu_worker_result) -> CpuWorkerCause {
    use CpuWorkerCause::*;
    match result.cause {
        1 => UnknownLayout,
        3 => Busy,
        4 => AllocationFailed,
        6 => OrdinaryPredecessor,
        7 => AlreadyInitialized,
        8 => IdentityMismatch,
        10 => ConstructionFailed,
        11 => LayoutChanged,
        13 => ExecutionFailed,
        12 => SystemFailure {
            value: result.system_value,
            category: result.system_category,
        },
        _ => Invalid,
    }
}
/// Fixed refusal followed by the same untransferred accounting/source owner.
pub struct CpuWorkerError<T> {
    cause: CpuWorkerCause,
    owner: T,
}
impl<T> CpuWorkerError<T> {
    /// Actual startup refusal, preserving native system code where available.
    pub fn cause(&self) -> CpuWorkerCause {
        self.cause
    }
    /// Borrow the intact original owner.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the cause and exact owner, without minting a new allowance.
    pub fn into_parts(self) -> (CpuWorkerCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for CpuWorkerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuWorkerError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for CpuWorkerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for CpuWorkerError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Fixed matcher data and libc++ key/guard storage, priced once per process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuWorkerStaticLayout {
    /// Exact compiled fixed extents, separate from each worker's dynamic source.
    pub bytes: usize,
    /// Static ABI qualification, independent of runtime-loaded code matching.
    pub qualified: bool,
}
/// Side-effect-free static facts; does not initialize the libc++ TLS key.
pub fn cpu_worker_static_layout() -> CpuWorkerStaticLayout {
    let mut native = safemlx_sys::mlx_cpu_worker_static_layout::default();
    // SAFETY: initialized fixed C output; no constructor or loader query.
    unsafe { safemlx_sys::mlx_cpu_worker_static_layout_for(&mut native) };
    CpuWorkerStaticLayout {
        bytes: native.bytes,
        qualified: native.qualified == 1,
    }
}
/// Scalar target minted from actual admitted Scheduler and stream identities.
/// It owns no Stream wrapper and grants no submission authority.
#[derive(Debug)]
pub struct CpuWorkerTarget {
    native: safemlx_sys::mlx_cpu_worker_target,
}
// SAFETY: the opaque token is compared, never dereferenced by Rust. Successful
// registration has process lifetime; native authenticates all scalars again.
unsafe impl Send for CpuWorkerTarget {}
unsafe impl Sync for CpuWorkerTarget {}
impl CpuWorkerTarget {
    /// Capture the exact registered stream while its immutable wrapper is borrowed.
    /// No worker, wrapper copy or source node is created by this operation.
    pub fn for_stream(
        scheduler: &InitializedScheduler,
        stream: &RegisteredCpuStream,
    ) -> Result<Self, CpuWorkerCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(CpuWorkerCause::Busy);
        };
        let mut native = safemlx_sys::mlx_cpu_worker_target {
            scheduler_identity: 0,
            stream_index: -1,
            device_index: -1,
            stream_birth: std::ptr::null(),
        };
        // SAFETY: both private identities came from successful original constructors.
        // The independent wrapper cannot mutate through this shared borrow.
        let result = unsafe {
            safemlx_sys::mlx_cpu_worker_target_for(
                &mut native,
                scheduler.identity,
                stream.as_stream().c_stream,
                stream.owner,
            )
        };
        if result.cause == 0 {
            Ok(Self { native })
        } else {
            Err(cause(result))
        }
    }
}
/// Exact startup requests and finite failure prefixes for one source owner type.
pub struct CpuWorkerLayout<T> {
    native: safemlx_sys::mlx_cpu_worker_layout,
    controls: usize,
    owner: PhantomData<fn() -> T>,
}
impl<T> Copy for CpuWorkerLayout<T> {}
impl<T> Clone for CpuWorkerLayout<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for CpuWorkerLayout<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuWorkerLayout")
            .field("native", &self.native)
            .field("controls", &self.controls)
            .finish()
    }
}
impl<T> CpuWorkerLayout<T> {
    /// One directory node including its inline worker; counted once only.
    pub fn entry_bytes(&self) -> usize {
        self.native.entry_bytes
    }
    /// Sum of constructor-failure payload/String/refstring requests, including overlap.
    pub fn failure_request_bytes(&self) -> usize {
        self.native.failure_requests
    }
    /// Named native/C/safe preparation, result and retirement controls.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete dynamic startup contribution. Excludes separately priced statics,
    /// OS-private thread resources and later tasks/diagnostic String growth.
    pub fn required_bytes(&self) -> Option<usize> {
        [
            self.native.entry_bytes,
            self.native.fallback_control_bytes,
            self.native.fallback_char_bytes,
            self.native.thread_handle_bytes,
            self.native.thread_implementation_bytes,
            self.native.thread_packet_bytes,
            self.native.failure_requests,
            mem::size_of::<OwnedNode<T>>(),
            self.controls,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}
/// Birth authentication for an admitted process worker. This is not a health,
/// completion, task-allocation or mutable diagnostic-capacity certificate.
#[derive(Debug)]
pub struct InitializedCpuWorker {
    target: CpuWorkerTarget,
    birth: *const c_void,
}
// SAFETY: only immutable scalar/token comparisons occur; native synchronizes
// directory access and the process worker permanently retains the actual token.
unsafe impl Send for InitializedCpuWorker {}
unsafe impl Sync for InitializedCpuWorker {}
impl InitializedCpuWorker {
    /// Observe this exact admitted worker's terminal frontier without creating,
    /// submitting, progressing or waiting for work. `Busy` preserves all owners.
    /// This snapshot is not a session readiness grant: callers must separately
    /// hold exclusive session state and exclude further producers of that state.
    pub fn try_observe_idle(&self) -> Result<(), CpuWorkerCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(CpuWorkerCause::Busy);
        };
        // SAFETY: private original tokens are authenticated under native loans;
        // the runtime loan excludes host encoder mutation during observation.
        let result =
            unsafe { safemlx_sys::mlx_cpu_worker_observe_idle(self.target.native, self.birth) };
        if result.cause == 0 {
            Ok(())
        } else {
            Err(cause(result))
        }
    }

    /// Authenticate the same worker source without creating a worker or task.
    /// This immutable identity check is valid inside either execution mode; it
    /// grants no submission, completion, or allocation authority.
    pub fn try_borrow(&self) -> Result<(), CpuWorkerCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(CpuWorkerCause::Busy);
        };
        // SAFETY: private identities remain process owned and are rechecked natively.
        let result = unsafe { safemlx_sys::mlx_cpu_worker_borrow(self.target.native, self.birth) };
        if result.cause == 0 {
            Ok(())
        } else {
            Err(cause(result))
        }
    }
}
/// One admitted custody node, adopted only after actual std::thread startup.
pub struct PreparedCpuWorker<T: Send + 'static> {
    layout: CpuWorkerLayout<T>,
    target: Option<CpuWorkerTarget>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedCpuWorker<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCpuWorker")
            .field("layout", &self.layout)
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedCpuWorker<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedCpuWorker<T> {
    /// Query the actual loaded producer before source admission. No loader call,
    /// thread key, worker, runtime lock or source owner is initialized.
    pub fn layout() -> Result<CpuWorkerLayout<T>, CpuWorkerCause> {
        let mut native = safemlx_sys::mlx_cpu_worker_layout::default();
        // SAFETY: initialized fixed transport; native performs read-only qualification.
        let result = unsafe { safemlx_sys::mlx_cpu_worker_layout_for(&mut native) };
        if result.cause != 0 {
            return Err(cause(result));
        }
        let controls = [
            native.controls,
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<CpuWorkerLayout<T>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Result<CpuWorkerLayout<T>, CpuWorkerCause>>(),
            mem::size_of::<safemlx_sys::mlx_cpu_worker_layout>(),
            mem::size_of::<safemlx_sys::mlx_cpu_worker_result>() * 3,
            mem::size_of::<CpuWorkerTarget>() * 2,
            mem::size_of::<Result<CpuWorkerTarget, CpuWorkerCause>>(),
            mem::size_of::<&InitializedScheduler>(),
            mem::size_of::<&RegisteredCpuStream>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<InitializedCpuWorker>(),
            mem::size_of::<Result<Self, CpuWorkerError<(CpuWorkerTarget, T)>>>(),
            mem::size_of::<Result<InitializedCpuWorker, CpuWorkerError<Self>>>(),
            mem::size_of::<Result<(), CpuWorkerCause>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<*const c_void>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            mem::size_of::<T>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(mem::size_of_val(&controls), usize::checked_add)
            .ok_or(CpuWorkerCause::UnknownLayout)?;
        let layout = CpuWorkerLayout {
            native,
            controls,
            owner: PhantomData,
        };
        layout
            .required_bytes()
            .ok_or(CpuWorkerCause::UnknownLayout)?;
        Ok(layout)
    }
    /// Allocate the exact source node only after the caller admits this layout.
    /// The scalar target needs no allocation and remains in each owning error.
    pub fn with_layout(
        layout: CpuWorkerLayout<T>,
        target: CpuWorkerTarget,
        owner: T,
    ) -> Result<Self, CpuWorkerError<(CpuWorkerTarget, T)>> {
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: exact nonzero node layout, initialized before Box adoption.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(CpuWorkerError {
                cause: CpuWorkerCause::AllocationFailed,
                owner: (target, owner),
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
            target: Some(target),
            node: Some(node),
        })
    }
    /// Borrow the same custody retained since admission.
    pub fn owner(&self) -> &T {
        &self.node.as_ref().expect("live worker initializer").owner
    }
    /// Dispose the outer node before returning its source and scalar target.
    pub fn into_parts(mut self) -> (CpuWorkerTarget, T) {
        let target = self.target.take().expect("live worker target");
        let owner = take_owner(self.node.take().expect("live worker initializer"));
        (target, owner)
    }
    /// Start exactly one process worker. All failures retain this same node;
    /// native construction and exception prefixes retire before the result returns.
    pub fn try_initialize(mut self) -> Result<InitializedCpuWorker, CpuWorkerError<Self>> {
        let mut birth = std::ptr::null();
        let result;
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(CpuWorkerError {
                    cause: CpuWorkerCause::Busy,
                    owner: self,
                });
            };
            let owner = (&mut **self.node.as_mut().expect("live worker initializer")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: native rechecks layout/identities; failure cannot adopt owner.
            // Success permanently owns the node, without running its destructor.
            result = unsafe {
                safemlx_sys::mlx_cpu_worker_initialize(
                    &mut birth,
                    self.layout.native,
                    self.target.as_ref().expect("live worker target").native,
                    owner,
                    Some(retire),
                )
            };
            if result.cause == 0 {
                let _ = Box::into_raw(self.node.take().expect("live worker initializer"));
            }
        }
        if result.cause == 0 {
            Ok(InitializedCpuWorker {
                target: self.target.take().expect("live worker target"),
                birth,
            })
        } else {
            Err(CpuWorkerError {
                cause: cause(result),
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
    fn worker_busy_preserves_exact_node_target_and_source_until_extraction() {
        let result = PreparedCpuWorker::<Owner>::layout();
        if std::env::var_os("EREDU_REQUIRE_CPU_WORKER_STARTUP_QUALIFICATION").is_some() {
            assert!(result.is_ok(), "{result:?}");
        }
        let layout = match result {
            Ok(value) => value,
            Err(CpuWorkerCause::UnknownLayout) => return,
            Err(other) => panic!("{other:?}"),
        };
        assert!(layout.failure_request_bytes() > 0);
        assert!(layout.required_bytes().unwrap() > layout.entry_bytes());
        let drops = Arc::new(AtomicUsize::new(0));
        // Inert private target: the occupied runtime loan must refuse before C
        // inspects any target. No admitted Scheduler/worker is fabricated.
        let target = CpuWorkerTarget {
            native: safemlx_sys::mlx_cpu_worker_target {
                scheduler_identity: 17,
                stream_index: 19,
                device_index: 0,
                stream_birth: std::ptr::null(),
            },
        };
        let prepared =
            PreparedCpuWorker::with_layout(layout, target, Owner(drops.clone())).unwrap();
        let original = &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>;
        let (entered, ready) = mpsc::channel();
        let (release, waiting) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _loan = runtime_lock::enter();
            entered.send(()).unwrap();
            let _ = waiting.recv(); // disconnect also releases on assertion unwind
        });
        ready.recv().unwrap();
        let error = prepared.try_initialize().unwrap_err();
        assert_eq!(error.cause(), CpuWorkerCause::Busy);
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<CpuWorkerCause>());
        assert_eq!(
            &**error.owner().node.as_ref().unwrap() as *const _,
            original
        );
        assert_eq!(
            error.owner().target.as_ref().unwrap().native.stream_index,
            19
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        worker.join().unwrap();
        let (_, prepared) = error.into_parts();
        let (target, owner) = prepared.into_parts();
        assert_eq!(target.native.scheduler_identity, 17);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
