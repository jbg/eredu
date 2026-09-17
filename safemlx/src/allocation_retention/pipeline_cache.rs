//! Request-owned destinations for actual precompiled Metal pipeline lookups.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner, SubmissionGraphQuota};
use std::{alloc::Layout, fmt, marker::PhantomData, mem::size_of, ptr};

/// Fixed constructor/attachment refusal, without an allocated diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineCacheCause {
    /// The actual compiler/library/device combination lacks qualified coverage.
    UnknownLayout,
    /// A malformed constructor input was refused.
    Invalid,
    /// Checked layout arithmetic overflowed.
    Overflow,
    /// An actual constructor allocation failed.
    AllocationFailed,
    /// The cache or destination arena was already shared, used, or configured.
    AlreadyBound,
    /// The finite selector destination population was exhausted.
    Exhausted,
}
impl PipelineCacheCause {
    fn status(status: u32) -> Self {
        match status {
            1 => Self::UnknownLayout,
            2 => Self::Invalid,
            3 => Self::Overflow,
            4 => Self::AllocationFailed,
            5 => Self::AlreadyBound,
            6 => Self::Exhausted,
            _ => Self::Invalid,
        }
    }
}
impl fmt::Display for PipelineCacheCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownLayout => "unqualified precompiled Metal pipeline layout",
            Self::Invalid => "invalid prepared Metal pipeline destination",
            Self::Overflow => "prepared Metal pipeline layout overflow",
            Self::AllocationFailed => "prepared Metal pipeline allocation failed",
            Self::AlreadyBound => "prepared Metal pipeline destination already bound or used",
            Self::Exhausted => "prepared Metal pipeline destinations exhausted",
        })
    }
}
impl std::error::Error for PipelineCacheCause {}

/// Constructor failure preserving the unchanged supplied custody.
pub struct PipelineCacheError<T> {
    cause: PipelineCacheCause,
    owner: T,
}
impl<T> PipelineCacheError<T> {
    /// The concrete nonowning refusal.
    pub fn cause(&self) -> PipelineCacheCause {
        self.cause
    }
    /// Recover the cause and unchanged owner after all constructor storage freed.
    pub fn into_parts(self) -> (PipelineCacheCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for PipelineCacheError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCacheError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for PipelineCacheError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<T> std::error::Error for PipelineCacheError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Actual requested native slab, unlocked-retirement node and fixed controls.
#[derive(Clone, Copy, Debug)]
pub struct PipelineCacheLayout {
    native: usize,
    node: usize,
    controls: usize,
}
impl PipelineCacheLayout {
    /// All managed storage requested by this constructor and its fixed worker.
    /// Platform compiler, allocator and driver internals remain excluded.
    pub fn required_bytes(self) -> Option<usize> {
        self.native
            .checked_add(self.node)?
            .checked_add(self.controls)
    }
}

/// Finite destination count from the selected native worker population.
#[derive(Clone, Copy, Debug)]
pub struct PreparedPipelineCachePlan {
    attempts: usize,
}
impl PreparedPipelineCachePlan {
    /// Record the actual certified lookup count; this grants no memory credit.
    pub fn new(attempts: usize) -> Self {
        Self { attempts }
    }
    /// Query exact layouts without creating a Device, source or pipeline.
    pub fn layout<T: Send + 'static>(self) -> Result<PipelineCacheLayout, PipelineCacheCause> {
        let mut native = safemlx_sys::mlx_pipeline_cache_layout::default();
        // SAFETY: the pure native query writes only this fixed local.
        let status =
            unsafe { safemlx_sys::mlx_pipeline_cache_layout_for(&mut native, self.attempts) };
        if status != 0 {
            return Err(PipelineCacheCause::status(status));
        }
        let controls = [
            size_of::<Self>(),
            size_of::<PipelineCacheLayout>(),
            size_of::<safemlx_sys::mlx_pipeline_cache_layout>(),
            size_of::<Result<PipelineCacheLayout, PipelineCacheCause>>(),
            size_of::<Result<(), PipelineCacheCause>>(),
            size_of::<Option<usize>>(),
            size_of::<Layout>(),
            size_of::<*mut OwnedNode<T>>(),
            size_of::<Box<OwnedNode<T>>>(),
            size_of::<T>(),
            size_of::<super::RetirementBatch>(),
            size_of::<*mut RetiredOwner>(),
            size_of::<unsafe fn(*mut RetiredOwner)>(),
            size_of::<safemlx_sys::mlx_pipeline_cache>(),
            size_of::<safemlx_sys::mlx_submission_graph_quota>(),
            size_of::<&SubmissionGraphQuota>(),
            size_of::<PreparedPipelineCache<T>>(),
            size_of::<PipelineCacheError<T>>(),
            size_of::<Result<PreparedPipelineCache<T>, PipelineCacheError<T>>>(),
            size_of::<u32>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(
                native
                    .control_bytes
                    .checked_add(std::mem::size_of_val(&controls))
                    .ok_or(PipelineCacheCause::Overflow)?,
                usize::checked_add,
            )
            .ok_or(PipelineCacheCause::Overflow)?;
        Ok(PipelineCacheLayout {
            native: native.allocation_bytes,
            node: size_of::<OwnedNode<T>>(),
            controls,
        })
    }
    /// Construct after admission; native failure frees its prefix before returning
    /// the unchanged custody. No shader compiler runs during preparation.
    pub fn realize<T: Send + 'static>(
        self,
        owner: T,
    ) -> Result<PreparedPipelineCache<T>, PipelineCacheError<T>> {
        if let Err(cause) = self.layout::<T>() {
            return Err(PipelineCacheError { cause, owner });
        }
        // SAFETY: concrete nonzero node layout; failure does not consume owner.
        let storage =
            unsafe { std::alloc::alloc(Layout::new::<OwnedNode<T>>()) }.cast::<OwnedNode<T>>();
        if storage.is_null() {
            return Err(PipelineCacheError {
                cause: PipelineCacheCause::AllocationFailed,
                owner,
            });
        }
        // SAFETY: exclusive aligned storage initialized exactly once.
        unsafe {
            storage.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
        }
        // SAFETY: the initialized allocation has this exact Box type.
        let node = unsafe { Box::from_raw(storage) };
        let mut raw = safemlx_sys::mlx_pipeline_cache {
            ctx: ptr::null_mut(),
        };
        // SAFETY: native consumes this queue node only on success; callback only
        // enqueues after its slab and all retained framework handles retire.
        let status = unsafe {
            safemlx_sys::mlx_pipeline_cache_new_retaining(
                &mut raw,
                self.attempts,
                storage.cast(),
                Some(retire),
            )
        };
        if status != 0 {
            return Err(PipelineCacheError {
                cause: PipelineCacheCause::status(status),
                owner: take_owner(node),
            });
        }
        let _ = Box::into_raw(node);
        Ok(PreparedPipelineCache {
            raw,
            owner: PhantomData,
        })
    }
}

/// Native pipeline rows and their independent raw metadata custody. The arena
/// keeps this cache alive; supplied custody must not own that arena or its scopes.
pub struct PreparedPipelineCache<T: Send + 'static> {
    raw: safemlx_sys::mlx_pipeline_cache,
    owner: PhantomData<T>,
}
impl<T: Send + 'static> PreparedPipelineCache<T> {
    /// Attach once to a newly constructed, unshared arena before any graph birth.
    /// Success retains the same cache; failure leaves both owners unchanged.
    pub fn install(&self, arena: &SubmissionGraphQuota) -> Result<(), PipelineCacheCause> {
        // SAFETY: both opaque owners remain borrowed; native checks freshness
        // and transfers only an additional reference on successful installation.
        let status = unsafe { safemlx_sys::mlx_pipeline_cache_install(self.raw, arena.raw()) };
        if status == 0 {
            Ok(())
        } else {
            Err(PipelineCacheCause::status(status))
        }
    }
}
impl<T: Send + 'static> fmt::Debug for PreparedPipelineCache<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedPipelineCache")
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedPipelineCache<T> {
    fn drop(&mut self) {
        // SAFETY: this is the one safe native reference, independent of arena
        // aliases and fixed native errors. Final callback only queues T.
        unsafe { safemlx_sys::mlx_pipeline_cache_free(self.raw) };
    }
}
// SAFETY: native reference/row claims are atomic; T retires only through the
// existing Send queue. Sharing the safe owner additionally requires T: Sync.
unsafe impl<T: Send + 'static> Send for PreparedPipelineCache<T> {}
unsafe impl<T: Send + Sync + 'static> Sync for PreparedPipelineCache<T> {}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::{
        reclaim_allocation_owners, MetalDeviceCause, PreparedMetalDevice,
        PreparedSubmissionGraphQuota,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            assert!(crate::can_reclaim_submission_resources());
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn graph() -> SubmissionGraphQuota {
        PreparedSubmissionGraphQuota::try_new(4096, ())
            .unwrap()
            .try_allocate()
            .unwrap()
    }
    #[test]
    fn pipeline_cache_failed_attach_preserves_owner_and_final_graph_alias_retires_it() {
        // Production authenticates and retains this separate shared baseline.
        // Other selected fixtures may already own the same native singleton.
        let prepared = PreparedMetalDevice::try_new(()).unwrap();
        if let Err(error) = prepared.try_initialize() {
            assert!(matches!(
                error.cause(),
                MetalDeviceCause::AlreadyInitialized | MetalDeviceCause::OrdinaryPredecessor
            ), "Device prefix: {error:?}");
        }
        let plan = PreparedPipelineCachePlan::new(2);
        let qualified = plan.layout::<Owner>().is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_PIPELINE_CACHE").is_some() {
            assert!(
                qualified,
                "actual precompiled library/selector layout must qualify"
            );
        }
        if !qualified {
            return;
        }
        let retired = Arc::new(AtomicUsize::new(0));
        let cache = plan.realize(Owner(retired.clone())).unwrap();
        let shared = graph();
        let shared_alias = shared.clone();
        assert_eq!(
            cache.install(&shared),
            Err(PipelineCacheCause::AlreadyBound)
        );
        drop((shared, shared_alias));
        reclaim_allocation_owners();
        assert_eq!(retired.load(Ordering::SeqCst), 0);

        let actual = graph();
        cache.install(&actual).unwrap();
        let foreign = graph();
        assert_eq!(
            cache.install(&foreign),
            Err(PipelineCacheCause::AlreadyBound)
        );
        assert_eq!(
            cache.install(&actual),
            Err(PipelineCacheCause::AlreadyBound)
        );
        drop(foreign);
        let alias = actual.clone();
        drop((cache, actual));
        reclaim_allocation_owners();
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        reclaim_allocation_owners();
        assert_eq!(retired.load(Ordering::SeqCst), 1);

        let refused_retired = Arc::new(AtomicUsize::new(0));
        let refused = PreparedPipelineCachePlan::new(usize::MAX)
            .realize(Owner(refused_retired.clone()))
            .unwrap_err();
        assert_eq!(refused_retired.load(Ordering::SeqCst), 0);
        drop(refused);
        assert_eq!(refused_retired.load(Ordering::SeqCst), 1);
    }
}
