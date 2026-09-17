//! Constructor-owned finite storage for native graph metadata.
use super::{OwnedNode, RetiredOwner, destroy, retire, take_owner};
use crate::utils::runtime_lock;
use std::{alloc::Layout, ffi::c_void, fmt, marker::PhantomData, mem, ptr, rc::Rc};

/// Fixed construction refusal. It neither consumes custody nor proves completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SubmissionGraphQuotaCause {
    /// Native checked layout arithmetic rejected the component ceiling.
    #[error("invalid graph metadata capacity")]
    InvalidCapacity,
    /// The prepared Rust owner could not be allocated.
    #[error("graph metadata owner allocation failed")]
    AllocationFailed,
    /// No native constructor ran; the identical preparation is returned.
    #[error("native runtime is busy during graph metadata construction")]
    RuntimeBusy,
    /// Native allocation failed before owner handoff.
    #[error("graph metadata arena allocation failed")]
    NativeAllocationFailed,
}

/// Cause first and unchanged original owner last.
pub struct SubmissionGraphQuotaError<T> {
    cause: SubmissionGraphQuotaCause,
    owner: T,
}
impl<T> SubmissionGraphQuotaError<T> {
    /// Fixed cause, with no formatting or owner clone.
    pub fn cause(&self) -> SubmissionGraphQuotaCause {
        self.cause
    }
    /// Borrow the unchanged prepared owner.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Consume this error, keeping the original owner.
    pub fn into_parts(self) -> (SubmissionGraphQuotaCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for SubmissionGraphQuotaError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionGraphQuotaError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for SubmissionGraphQuotaError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for SubmissionGraphQuotaError<T> {}

/// Actual allocated and named construction/retirement representations for one T.
/// The arena includes its allocator headers. Task/device/error and unmigrated primitive payloads
/// are separate mechanisms; this is not a prediction of evaluation success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionGraphQuotaLayout {
    /// Explicit physical arena ceiling, including its in-arena block headers.
    pub capacity: usize,
    /// Complete native resource plus arena allocation.
    pub native_allocation_bytes: usize,
    /// Native allocation alignment.
    pub native_alignment: usize,
    /// Preallocated Rust queue node including T.
    pub rust_node_bytes: usize,
    /// Actual named preparation, begin, extraction, failure and retirement values.
    pub control_bytes: usize,
}
impl SubmissionGraphQuotaLayout {
    /// Checked original contribution, before any arena is constructed.
    pub fn total_bytes(self) -> Option<usize> {
        self.native_allocation_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}

mod retirement_capacity;
pub use retirement_capacity::{
    RetirementCapacity, RetirementCapacityCause, RetirementCapacityOwner, RetirementCapacityPermit,
};

struct Construction {
    raw: safemlx_sys::mlx_submission_graph_quota,
    status: i32,
}

/// Closed native strong owner. Clones share exactly one fixed arena.
/// No raw owner/Weak/reset/grow/refill operation is exposed. Native Scope and
/// Record, allocation and empty-container references retain the allocation independently.
pub struct SubmissionGraphQuota {
    raw: safemlx_sys::mlx_submission_graph_quota,
    _origin: PhantomData<Rc<()>>,
}
impl Clone for SubmissionGraphQuota {
    fn clone(&self) -> Self {
        // SAFETY: this live closed handle pins the same native atomic owner.
        unsafe { safemlx_sys::mlx_submission_graph_quota_retain(self.raw) };
        Self {
            raw: self.raw,
            _origin: PhantomData,
        }
    }
}
impl Drop for SubmissionGraphQuota {
    fn drop(&mut self) {
        // SAFETY: release returns native storage before enqueueing the Send
        // capsule. Arbitrary owner destruction uses the existing unlocked queue.
        unsafe { safemlx_sys::mlx_submission_graph_quota_release(self.raw) };
    }
}
impl fmt::Debug for SubmissionGraphQuota {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionGraphQuota")
            .finish_non_exhaustive()
    }
}
impl SubmissionGraphQuota {
    /// Conservative physical extent sum for positive allocation requests.
    /// The caller supplies the sum of their requested bytes and a finite attempt
    /// ceiling. Actual block headers and maximum supported alignment/rounding
    /// come from the native allocator; no arena or runtime is initialized.
    pub fn allocation_population_extent(bytes: usize, attempts: usize) -> Option<usize> {
        let mut out = 0;
        // SAFETY: pure checked arithmetic writes only this scalar.
        (unsafe {
            safemlx_sys::mlx_submission_graph_quota_population_extent(&mut out, bytes, attempts)
        } == 0)
            .then_some(out)
    }
    /// Fresh-arena capacity for a complete sum of all attempted block extents.
    /// Adds the allocator's actual minimum split tail once. Population coverage
    /// belongs to the calling producer; this query grants no storage authority.
    pub fn fresh_capacity_for_extents(extents: usize) -> Option<usize> {
        let mut out = 0;
        // SAFETY: pure checked arithmetic writes only this scalar.
        (unsafe { safemlx_sys::mlx_submission_graph_quota_fresh_capacity(&mut out, extents) } == 0)
            .then_some(out)
    }
    /// Named fixed controls for the pure population and fresh-capacity queries.
    /// These are query representations, not a second arena allocation.
    pub fn fit_query_control_bytes() -> Option<usize> {
        [
            mem::size_of::<safemlx_sys::mlx_submission_graph_quota_layout>(),
            mem::size_of::<(usize, usize, usize)>(), // request/count/output
            mem::size_of::<(usize, usize, usize, usize)>(), // native checks
            mem::size_of::<(*mut usize, i32)>(),
            mem::size_of::<Option<usize>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn raw(&self) -> safemlx_sys::mlx_submission_graph_quota {
        self.raw
    }
    /// Exact owner comparison; no layout equality can substitute for identity.
    pub fn same_arena(&self, other: &Self) -> bool {
        self.raw.ctx == other.raw.ctx
    }
    /// Physical occupied blocks, for diagnostics only; never completion evidence.
    pub fn occupied_bytes(&self) -> usize {
        // SAFETY: the native arena is live and its short arithmetic loan is safe.
        unsafe { safemlx_sys::mlx_submission_graph_quota_occupied(self.raw) }
    }
}

/// One exclusive preparation. T: Send allows its existing owned-node queue to
/// retire custody on another host thread; no unsafe Send assertion is used.
/// T must not own this future arena or a Scope/Record which retains it.
pub struct PreparedSubmissionGraphQuota<T: Send + 'static> {
    layout: SubmissionGraphQuotaLayout,
    node: Option<Box<OwnedNode<T>>>,
    retire: unsafe extern "C" fn(*mut c_void),
}
impl<T: Send + 'static> fmt::Debug for PreparedSubmissionGraphQuota<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSubmissionGraphQuota")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedSubmissionGraphQuota<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedSubmissionGraphQuota<T> {
    pub(crate) fn capacity(&self) -> usize {
        self.layout.capacity
    }
    /// Cold concrete layout. Invokes no allocator, runtime loan or housekeeping.
    pub fn layout(
        capacity: usize,
    ) -> Result<SubmissionGraphQuotaLayout, SubmissionGraphQuotaCause> {
        let mut facts = safemlx_sys::mlx_submission_graph_quota_layout {
            capacity: 0,
            allocation_bytes: 0,
            alignment: 0,
            retirement_controls: 0,
        };
        // SAFETY: checked native layout writes this valid output or rejects it.
        if unsafe { safemlx_sys::mlx_submission_graph_quota_layout_for(&mut facts, capacity) } != 0
        {
            return Err(SubmissionGraphQuotaCause::InvalidCapacity);
        }
        let parts = [
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<Construction>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<SubmissionGraphQuota>(),
            mem::size_of::<Result<Self, SubmissionGraphQuotaError<T>>>(),
            mem::size_of::<Result<SubmissionGraphQuota, SubmissionGraphQuotaError<Self>>>(),
            // Same existing queue's concrete unbox and detached-batch controls.
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<T>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            facts.retirement_controls,
        ];
        let control_bytes = parts
            .into_iter()
            .try_fold(mem::size_of_val(&parts), |sum, value| {
                sum.checked_add(value)
            })
            .ok_or(SubmissionGraphQuotaCause::InvalidCapacity)?;
        Ok(SubmissionGraphQuotaLayout {
            capacity,
            native_allocation_bytes: facts.allocation_bytes,
            native_alignment: facts.alignment,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
        })
    }
    /// Prepare only the queue node. Failure returns the unchanged T.
    pub fn try_new(capacity: usize, owner: T) -> Result<Self, SubmissionGraphQuotaError<T>> {
        let layout = match Self::layout(capacity) {
            Ok(value) => value,
            Err(cause) => return Err(SubmissionGraphQuotaError { cause, owner }),
        };
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: checked matching Layout, initialized before constructing Box.
        let node = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(SubmissionGraphQuotaError {
                cause: SubmissionGraphQuotaCause::AllocationFailed,
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
            retire,
        })
    }
    /// Cancel this never-attached preparation, unboxing before returning T.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("unconsumed arena preparation"))
    }
    /// Borrow custody without exporting its queue node or allocation owner.
    pub fn owner(&self) -> &T {
        &self
            .node
            .as_ref()
            .expect("unconsumed arena preparation")
            .owner
    }
    /// Allocate the fixed arena once; Busy/failure returns this same preparation.
    pub fn try_allocate(mut self) -> Result<SubmissionGraphQuota, SubmissionGraphQuotaError<Self>> {
        let mut controls = Construction {
            raw: safemlx_sys::mlx_submission_graph_quota {
                ctx: ptr::null_mut(),
            },
            status: -1,
        };
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(SubmissionGraphQuotaError {
                    cause: SubmissionGraphQuotaCause::RuntimeBusy,
                    owner: self,
                });
            };
            let payload = (&mut **self.node.as_mut().expect("unconsumed arena preparation")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: success alone consumes this exclusive initialized node.
            controls.status = unsafe {
                safemlx_sys::mlx_submission_graph_quota_new_retaining(
                    &mut controls.raw,
                    self.layout.capacity,
                    payload,
                    Some(self.retire),
                )
            };
            if controls.status == 0 {
                let _ = Box::into_raw(self.node.take().expect("unconsumed arena preparation"));
            }
        }
        if controls.status == 0 {
            Ok(SubmissionGraphQuota {
                raw: controls.raw,
                _origin: PhantomData,
            })
        } else {
            Err(SubmissionGraphQuotaError {
                cause: SubmissionGraphQuotaCause::NativeAllocationFailed,
                owner: self,
            })
        }
    }
}

// Focused original-account and native allocation tests are added before freeze.

#[cfg(test)]
mod tests;
