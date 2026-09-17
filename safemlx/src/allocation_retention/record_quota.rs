//! Constructor-owned finite storage for native submission bookkeeping.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::utils::runtime_lock;
use std::{alloc::Layout, ffi::c_void, fmt, marker::PhantomData, mem, ptr, rc::Rc};

/// Fixed construction refusal. It neither consumes custody nor proves completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SubmissionRecordQuotaCause {
    /// Native checked layout arithmetic rejected the component ceiling.
    #[error("invalid submission tracking capacity")]
    InvalidCapacity,
    /// The prepared Rust owner could not be allocated.
    #[error("submission tracking owner allocation failed")]
    AllocationFailed,
    /// No native constructor ran; the identical preparation is returned.
    #[error("native runtime is busy during submission tracking construction")]
    RuntimeBusy,
    /// Native allocation failed before owner handoff.
    #[error("submission tracking arena allocation failed")]
    NativeAllocationFailed,
}

/// Cause first and unchanged original owner last.
pub struct SubmissionRecordQuotaError<T> {
    cause: SubmissionRecordQuotaCause,
    owner: T,
}
impl<T> SubmissionRecordQuotaError<T> {
    /// Fixed cause, with no formatting or owner clone.
    pub fn cause(&self) -> SubmissionRecordQuotaCause {
        self.cause
    }
    /// Borrow the unchanged prepared owner.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Consume this error, keeping the original owner.
    pub fn into_parts(self) -> (SubmissionRecordQuotaCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for SubmissionRecordQuotaError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionRecordQuotaError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for SubmissionRecordQuotaError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for SubmissionRecordQuotaError<T> {}

/// Actual allocated and named construction/retirement representations for one T.
/// The arena includes its allocator headers. Graph/task/device/error payloads
/// are separate mechanisms; this is not a prediction of evaluation success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionRecordQuotaLayout {
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
impl SubmissionRecordQuotaLayout {
    /// Checked original contribution, before any arena is constructed.
    pub fn total_bytes(self) -> Option<usize> {
        self.native_allocation_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}

struct Construction {
    raw: safemlx_sys::mlx_submission_record_quota,
    status: i32,
}

/// Closed native strong owner. Clones share exactly one fixed arena.
/// No raw owner/Weak/reset/grow/refill operation is exposed. Native Scope and
/// Record references retain the allocation independently of this Rust handle.
pub struct SubmissionRecordQuota {
    raw: safemlx_sys::mlx_submission_record_quota,
    _origin: PhantomData<Rc<()>>,
}
impl Clone for SubmissionRecordQuota {
    fn clone(&self) -> Self {
        // SAFETY: this live closed handle pins the same native atomic owner.
        unsafe { safemlx_sys::mlx_submission_record_quota_retain(self.raw) };
        Self {
            raw: self.raw,
            _origin: PhantomData,
        }
    }
}
impl Drop for SubmissionRecordQuota {
    fn drop(&mut self) {
        // SAFETY: release returns native storage before enqueueing the Send
        // capsule. Arbitrary owner destruction uses the existing unlocked queue.
        unsafe { safemlx_sys::mlx_submission_record_quota_release(self.raw) };
    }
}
impl fmt::Debug for SubmissionRecordQuota {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionRecordQuota")
            .finish_non_exhaustive()
    }
}
impl SubmissionRecordQuota {
    /// Exact native physical block storage for one allocation request, including
    /// headers, back-offset, padding and rounding. Unsupported alignment or
    /// overflow returns unknown. This cold query performs no allocation and
    /// does not account for other live blocks or external fragmentation.
    pub fn allocation_extent(requested: usize, alignment: usize) -> Option<usize> {
        let mut out = 0;
        // SAFETY: pure checked native arithmetic writes this initialized scalar
        // only on success, retaining no pointer or owner.
        (unsafe {
            safemlx_sys::mlx_submission_record_quota_allocation_extent(
                &mut out, requested, alignment,
            )
        } == 0)
            .then_some(out)
    }

    /// Sufficient capacity for a fresh arena when `extents` includes every actual
    /// finite allocation attempt, computed by [`Self::allocation_extent`].
    /// Native split-tail storage makes this insensitive to retirement order and
    /// reuse. This pure query does not certify the supplied population or an
    /// already occupied arena; unsupported alignment and overflow return None.
    pub fn fresh_capacity_for_extents(extents: usize) -> Option<usize> {
        let mut out = 0;
        // SAFETY: pure checked scalar query; no allocation or pointer retention.
        (unsafe { safemlx_sys::mlx_submission_record_quota_fresh_capacity(&mut out, extents) } == 0)
            .then_some(out)
    }

    pub(crate) fn raw(&self) -> safemlx_sys::mlx_submission_record_quota {
        self.raw
    }
    /// Exact owner comparison; no layout equality can substitute for identity.
    pub fn same_arena(&self, other: &Self) -> bool {
        self.raw.ctx == other.raw.ctx
    }
    /// Physical occupied blocks, for diagnostics only; never completion evidence.
    pub fn occupied_bytes(&self) -> usize {
        // SAFETY: the native arena is live and its short arithmetic loan is safe.
        unsafe { safemlx_sys::mlx_submission_record_quota_occupied(self.raw) }
    }
}

/// One exclusive preparation. T: Send allows its existing owned-node queue to
/// retire custody on another host thread; no unsafe Send assertion is used.
/// T must not own this future arena or a Scope/Record which retains it.
pub struct PreparedSubmissionRecordQuota<T: Send + 'static> {
    layout: SubmissionRecordQuotaLayout,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedSubmissionRecordQuota<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSubmissionRecordQuota")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedSubmissionRecordQuota<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedSubmissionRecordQuota<T> {
    /// Smallest physical arena accepted by the actual native allocator.
    /// This is not a Record count or a successful submission-workspace bound.
    pub fn minimum_layout() -> Result<SubmissionRecordQuotaLayout, SubmissionRecordQuotaCause> {
        let mut capacity = 0;
        // SAFETY: pure checked arithmetic writes only this initialized output.
        if unsafe { safemlx_sys::mlx_submission_record_quota_minimum_capacity(&mut capacity) } != 0
        {
            return Err(SubmissionRecordQuotaCause::InvalidCapacity);
        }
        Self::layout(capacity)
    }
    /// Cold concrete layout. Invokes no allocator, runtime loan or housekeeping.
    pub fn layout(
        capacity: usize,
    ) -> Result<SubmissionRecordQuotaLayout, SubmissionRecordQuotaCause> {
        let mut facts = safemlx_sys::mlx_submission_record_quota_layout {
            capacity: 0,
            allocation_bytes: 0,
            alignment: 0,
            retirement_controls: 0,
        };
        // SAFETY: checked native layout writes this valid output or rejects it.
        if unsafe { safemlx_sys::mlx_submission_record_quota_layout_for(&mut facts, capacity) } != 0
        {
            return Err(SubmissionRecordQuotaCause::InvalidCapacity);
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
            mem::size_of::<SubmissionRecordQuota>(),
            mem::size_of::<Result<Self, SubmissionRecordQuotaError<T>>>(),
            mem::size_of::<Result<SubmissionRecordQuota, SubmissionRecordQuotaError<Self>>>(),
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
            .ok_or(SubmissionRecordQuotaCause::InvalidCapacity)?;
        Ok(SubmissionRecordQuotaLayout {
            capacity,
            native_allocation_bytes: facts.allocation_bytes,
            native_alignment: facts.alignment,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
        })
    }
    /// Prepare only the queue node. Failure returns the unchanged T.
    pub fn try_new(capacity: usize, owner: T) -> Result<Self, SubmissionRecordQuotaError<T>> {
        let layout = match Self::layout(capacity) {
            Ok(value) => value,
            Err(cause) => return Err(SubmissionRecordQuotaError { cause, owner }),
        };
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: checked matching Layout, initialized before constructing Box.
        let node = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(SubmissionRecordQuotaError {
                cause: SubmissionRecordQuotaCause::AllocationFailed,
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
    pub fn try_allocate(
        mut self,
    ) -> Result<SubmissionRecordQuota, SubmissionRecordQuotaError<Self>> {
        let mut controls = Construction {
            raw: safemlx_sys::mlx_submission_record_quota {
                ctx: ptr::null_mut(),
            },
            status: -1,
        };
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(SubmissionRecordQuotaError {
                    cause: SubmissionRecordQuotaCause::RuntimeBusy,
                    owner: self,
                });
            };
            let payload = (&mut **self.node.as_mut().expect("unconsumed arena preparation")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: success alone consumes this exclusive initialized node.
            controls.status = unsafe {
                safemlx_sys::mlx_submission_record_quota_new_retaining(
                    &mut controls.raw,
                    self.layout.capacity,
                    payload,
                    Some(retire),
                )
            };
            if controls.status == 0 {
                let _ = Box::into_raw(self.node.take().expect("unconsumed arena preparation"));
            }
        }
        if controls.status == 0 {
            Ok(SubmissionRecordQuota {
                raw: controls.raw,
                _origin: PhantomData,
            })
        } else {
            Err(SubmissionRecordQuotaError {
                cause: SubmissionRecordQuotaCause::NativeAllocationFailed,
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests;
