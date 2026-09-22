//! Ownership tied to native allocations and Scope metadata, with unlocked retirement.

use crate::{Array, error::Exception, utils::guard::Guarded, utils::runtime_lock};
use std::{
    cell::Cell,
    ffi::c_void,
    fmt, ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

/// Failed allocation attachment, preserving the original owner's ownership.
pub struct AllocationOwnerError<T> {
    error: Exception,
    owner: T,
}

impl<T> AllocationOwnerError<T> {
    /// Native failure or unavailable certified backing.
    pub fn error(&self) -> &Exception {
        &self.error
    }

    /// Returns the failure and the owner that was never attached.
    pub fn into_parts(self) -> (Exception, T) {
        (self.error, self.owner)
    }
}

impl<T> fmt::Debug for AllocationOwnerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocationOwnerError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<T> fmt::Display for AllocationOwnerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl<T> std::error::Error for AllocationOwnerError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// A preallocated intrusive node lets the native destructor hand ownership to
// ordinary Rust code without allocating, locking or invoking arbitrary Drop.
#[repr(C)]
struct RetiredOwner {
    next: *mut RetiredOwner,
    destroy: unsafe fn(*mut RetiredOwner),
}

#[repr(C)]
struct OwnedNode<T> {
    retired: RetiredOwner,
    owner: T,
}

static RETIRED: AtomicPtr<RetiredOwner> = AtomicPtr::new(ptr::null_mut());

pub(crate) fn static_storage_bytes() -> usize {
    std::mem::size_of_val(&RETIRED)
}

thread_local! {
    static RECLAIMING: Cell<bool> = const { Cell::new(false) };
}

// Returning through this boundary deallocates the moved-out Box before the
// caller can drop T, which may release the accounting for the Rust node itself.
fn take_owner<T>(node: Box<OwnedNode<T>>) -> T {
    let OwnedNode { owner, .. } = *node;
    owner
}

unsafe fn destroy<T>(node: *mut RetiredOwner) {
    // SAFETY: the repr(C) first field has the allocation's original address;
    // this callback is paired with exactly OwnedNode<T> at construction.
    let owner = take_owner(unsafe { Box::from_raw(node.cast::<OwnedNode<T>>()) });
    drop(owner);
}

unsafe fn enqueue(node: *mut RetiredOwner) {
    let mut head = RETIRED.load(Ordering::Relaxed);
    loop {
        // SAFETY: a node is published exactly once per exclusive ownership
        // transfer. Readers cannot access it until the release CAS succeeds.
        unsafe { (*node).next = head };
        match RETIRED.compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return,
            Err(current) => head = current,
        }
    }
}

unsafe extern "C" fn retire(payload: *mut c_void) {
    // SAFETY: successful native construction/attachment owns this preallocated node and
    // calls back once, only after the final covered native allocation retires.
    // This path never runs arbitrary Rust destructors or allocates memory.
    unsafe { enqueue(payload.cast()) };
}

struct RetirementBatch(*mut RetiredOwner);

impl Drop for RetirementBatch {
    fn drop(&mut self) {
        // A user's destructor may panic. Return all unvisited nodes to the
        // global queue so the next ordinary host entry can reclaim them.
        while !self.0.is_null() {
            let node = self.0;
            // SAFETY: this batch exclusively owns its detached list.
            unsafe {
                self.0 = (*node).next;
                enqueue(node);
            }
        }
    }
}

// Shared ordinary/specific retirement exclusion. Acquiring this guard never
// enters the native runtime or progresses an unrelated owner.
struct ReclamationGuard;
fn begin_reclamation() -> Option<ReclamationGuard> {
    if !runtime_lock::can_reclaim_submission_resources()
        || RECLAIMING
            .try_with(|running| running.replace(true))
            .unwrap_or(true)
    {
        return None;
    }
    Some(ReclamationGuard)
}
impl Drop for ReclamationGuard {
    fn drop(&mut self) {
        let _ = RECLAIMING.try_with(|running| running.set(false));
    }
}

/// Drops owners whose covered native allocation has retired (backing or Scope metadata).
///
/// Returns the number reclaimed from the current queue. This never waits for
/// native work. It does nothing during unwinding, reentrant reclamation, or
/// while this thread holds the native runtime lock. Ordinary runtime entry also
/// calls this before acquiring that lock. Call explicitly after final native
/// cleanup if no subsequent runtime operation is expected.
///
/// Owners can retire on native worker threads; their Rust destructors run only
/// here on an ordinary host thread. A queued owner continues retaining its
/// resources until reclaimed. A panicking destructor leaves other owners queued.
pub fn reclaim_allocation_owners() -> usize {
    let Some(_reset) = begin_reclamation() else {
        return 0;
    };
    let mut batch = RetirementBatch(RETIRED.swap(ptr::null_mut(), Ordering::Acquire));
    let mut reclaimed = 0usize;
    while !batch.0.is_null() {
        let node = batch.0;
        // SAFETY: the acquire exchange exclusively receives initialized nodes
        // published by retire(). T: Send permits destruction on this host thread.
        unsafe {
            batch.0 = (*node).next;
            let destroy = (*node).destroy;
            destroy(node);
        }
        reclaimed += 1;
    }
    reclaimed
}

impl Array {
    /// Retains payload-free authority without evaluating, polling or waiting.
    ///
    /// Existing clones share the native descriptor. Materialization publishes
    /// its owner to certified physical backing before kernels use that backing;
    /// views, donated storage, native submission pins and certified host aliases
    /// then retain it until their backing retires. An unevaluated graph retains
    /// it until its final descriptor retires. Empty values may retain authority
    /// with their descriptor even though they have no physical allocation.
    ///
    /// This does not propagate ownership to independently allocated derived
    /// results. Attach those separately. Materialization into unrecognized
    /// borrowed/custom backing fails instead of silently losing ownership.
    /// Existing completed-only [`Self::retain_allocation_owner`] semantics and
    /// cold allocation identity queries remain unchanged.
    ///
    /// The owner must not retain this array, its graph, or covered storage,
    /// directly or indirectly: that would create an ownership cycle. Charge
    /// handles without native/source roots are suitable. Retirement uses the
    /// same unlocked host queue as [`Self::retain_allocation_owner`]. Failure
    /// before attachment returns the original owner.
    pub fn retain_deferred_allocation_owner<T: Send + Sync + 'static>(
        &self,
        owner: T,
    ) -> Result<(), AllocationOwnerError<T>> {
        retain_owner(owner, |attached, payload, release| unsafe {
            // SAFETY: the shared descriptor remains live through this handoff;
            // the native side arms the callback after all fallible publication.
            safemlx_sys::mlx_array_retain_deferred_allocation_owner(
                attached,
                self.as_ptr(),
                payload,
                release,
            )
        })
    }

    /// Retains an owner until this completed physical backing and every native
    /// alias retire, including views, lazy children, submission pins and donated
    /// backing. Certified host-transfer storage also retains it across separate
    /// native wrappers and surviving host-buffer owners.
    ///
    /// This does not evaluate, poll, copy or replace storage. Unfinished,
    /// unrecognized and allocation-free values return the original owner in an
    /// error. Successful attachment leaves allocation identity/capacity unchanged.
    /// An independently allocated result requires its own attachment.
    ///
    /// Native retirement queues the owner for [`reclaim_allocation_owners`],
    /// which drops it outside the native runtime lock. Owners must not retain
    /// this backing themselves, directly or through an ancestor graph, because
    /// doing so would create an ownership cycle.
    pub fn retain_allocation_owner<T: Send + 'static>(
        &self,
        owner: T,
    ) -> Result<(), AllocationOwnerError<T>> {
        retain_owner(owner, |attached, payload, release| unsafe {
            // SAFETY: the helper owns all callback state and outputs. This
            // borrowed array keeps completed backing alive through attachment.
            safemlx_sys::mlx_array_retain_allocation_owner(
                attached,
                self.as_ptr(),
                payload,
                release,
            )
        })
    }
}

// Shared node handoff for audited native array and host-buffer producers.
pub(crate) fn retain_owner<T: Send + 'static>(
    owner: T,
    attach: impl FnOnce(*mut bool, *mut c_void, Option<unsafe extern "C" fn(*mut c_void)>) -> i32,
) -> Result<(), AllocationOwnerError<T>> {
    let _guard = runtime_lock::enter();
    let mut node = Box::new(OwnedNode {
        retired: RetiredOwner {
            next: ptr::null_mut(),
            destroy: destroy::<T>,
        },
        owner,
    });
    let mut attached = false;
    // The closure binds one audited native producer. Both producers consume
    // the node only on attached=true, after every fallible allocation.
    let result = <() as Guarded>::try_from_op(|_| {
        attach(
            &mut attached,
            ptr::from_mut(&mut node.retired).cast(),
            Some(retire),
        )
    });
    if attached {
        // The borrowed array/buffer keeps its backing alive through attachment,
        // so native retirement cannot race this ownership handoff.
        let _ = Box::into_raw(node);
        debug_assert!(result.is_ok());
        Ok(())
    } else {
        let OwnedNode { owner, .. } = *node;
        Err(AllocationOwnerError {
            error: result.err().unwrap_or_else(|| {
                Exception::custom("allocation ownership requires completed certified backing")
            }),
            owner,
        })
    }
}

#[cfg(test)]
mod deferred_tests;
#[cfg(test)]
mod tests;

mod prepared;
pub(crate) mod scope;
pub use prepared::{
    AllocationOwnerLayout, PreparedAllocationOwner, PreparedAllocationOwnerCause,
    PreparedAllocationOwnerError, PreparedAllocationRetirement,
};

mod record_quota;
pub use record_quota::{
    PreparedSubmissionRecordQuota, SubmissionRecordQuota, SubmissionRecordQuotaCause,
    SubmissionRecordQuotaError, SubmissionRecordQuotaLayout,
};

#[cfg(test)]
mod descriptor_tests;

mod graph_quota;
mod host_alias;
mod original_buffer;
pub use graph_quota::{
    PreparedSubmissionGraphQuota, RetirementCapacity, RetirementCapacityCause,
    RetirementCapacityOwner, RetirementCapacityPermit, SubmissionGraphQuota,
    SubmissionGraphQuotaCause, SubmissionGraphQuotaError, SubmissionGraphQuotaLayout,
};
pub use host_alias::{HostTransferArrayAliasWitness, HostTransferArrayViewWitness};
pub use original_buffer::{
    ImmutableSourceInspection, ImmutableSourceWitness, OrdinaryBufferInspection,
    OrdinaryBufferWitness, OriginalBufferAliasWitness, OriginalBufferBudget,
    OriginalBufferBudgetLayout, OriginalBufferCause, OriginalBufferError, OriginalBufferInspection,
    OriginalBufferLifetimeObserver, OriginalBufferPopulationLayout, OriginalBufferWitness,
    PreparedOriginalBufferBudget, SharedOriginalBufferInspection,
};

mod prefill_failure;
pub use prefill_failure::{
    PrefillFailureCause, PrefillFailureError, PrefillFailureLayout, PrefillNativeError,
    PrefillNativeFailureKind, PrefillNativeText, PreparedPrefillFailure, RetainedPrefillFailure,
};

mod input_allocator;
pub use input_allocator::*;

mod metal_device;
pub use metal_device::*;

mod scheduler;
pub use scheduler::*;

mod stream_copy;
pub use stream_copy::{
    CpuMatmulFacts, CpuMatmulKernel, PreparedStreamCopy, StreamCopyCause, StreamCopyError,
    StreamCopyPlan,
};

mod cpu_worker;
pub use cpu_worker::*;

mod stream_registration;
pub use stream_registration::*;

mod gpu_stream_registration;
pub use gpu_stream_registration::{
    GpuStreamRegistrationCause, GpuStreamRegistrationError, GpuStreamRegistrationLayout,
    GpuStreamTarget, PreparedGpuStream, RegisteredGpuStream,
};

pub(crate) mod kernel_family;

mod pipeline_cache;
pub use pipeline_cache::{
    PipelineCacheCause, PipelineCacheError, PipelineCacheLayout, PreparedPipelineCache,
    PreparedPipelineCachePlan,
};

mod host_writer;
pub use host_writer::{OrdinaryHostWriterError, PreparedHostTransferWriter};

/// A prepaid, payload-free owner handed to one physical backing observer.
/// Retirement uses the same deferred queue as tensor allocation attachments.
#[derive(Debug)]
pub struct PhysicalBackingCustody {
    node: *mut c_void,
    publish: safemlx_sys::mlx_physical_backing_publish,
}
// SAFETY: construction requires a Send + Sync payload, and the node is move-only.
unsafe impl Send for PhysicalBackingCustody {}
unsafe impl Sync for PhysicalBackingCustody {}
impl PhysicalBackingCustody {
    /// Exact Rust allocation for the retained owner. Reserve before `new`.
    pub const fn control_bytes<T: Send + Sync + 'static>() -> usize {
        std::mem::size_of::<OwnedNode<T>>()
    }
    /// Retain a previously funded owner without entering native execution.
    pub fn new<T: Send + Sync + 'static>(owner: T) -> Self {
        let node = Box::new(OwnedNode {
            retired: RetiredOwner {
                next: ptr::null_mut(),
                destroy: destroy::<T>,
            },
            owner,
        });
        Self {
            node: Box::into_raw(node).cast(),
            publish: None,
        }
    }
    /// Retains a prepaid allocation-only grant that must be published after
    /// native allocation. The callback never grants numerical execution.
    pub fn new_pending<T: PhysicalBackingPublication>(owner: T) -> Self {
        let mut custody = Self::new(owner);
        custody.publish = Some(publish_backing::<T>);
        custody
    }
    pub(crate) fn into_raw(self) -> (*mut c_void, safemlx_sys::mlx_physical_backing_publish) {
        let result = (self.node, self.publish);
        std::mem::forget(self);
        result
    }
    pub(crate) unsafe extern "C" fn release(payload: *mut c_void) {
        // SAFETY: native ownership is transferred once and retired once.
        unsafe { retire(payload) };
    }
    pub(crate) unsafe fn borrowed_owner<'a, T: Send + Sync + 'static>(
        payload: *mut c_void,
    ) -> &'a T {
        // SAFETY: the caller authenticates the concrete constructor type and
        // retains native ownership of this node through the returned borrow.
        &unsafe { &*payload.cast::<OwnedNode<T>>() }.owner
    }
    pub(crate) unsafe fn borrowed_error(payload: *mut c_void) -> std::sync::Arc<Exception> {
        // SAFETY: only a failed physical observer creates this exact node type;
        // the currently caught native exception retains it through this clone.
        unsafe { &*payload.cast::<OwnedNode<std::sync::Arc<Exception>>>() }
            .owner
            .clone()
    }
}
impl Drop for PhysicalBackingCustody {
    fn drop(&mut self) {
        // SAFETY: this exclusive unpublished node has not crossed native ownership.
        unsafe { retire(self.node) };
    }
}

/// Producer-owned reservation conversion after actual backing allocation.
pub trait PhysicalBackingPublication: Send + Sync + 'static {
    /// Publish already accepted physical capacity without allocating or native work.
    fn publish(&mut self) -> Result<(), Exception>;
}
unsafe extern "C" fn publish_backing<T: PhysicalBackingPublication>(
    payload: *mut c_void,
    failure: *mut *mut c_void,
    release: *mut Option<unsafe extern "C" fn(*mut c_void)>,
) -> bool {
    // SAFETY: the observer transferred this exact exclusive node; native calls
    // publication once before exposing any tensor or cache reference.
    let owner = unsafe { &mut (*payload.cast::<OwnedNode<T>>()).owner };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.publish()));
    let error = match result {
        Ok(Ok(())) => return true,
        Ok(Err(error)) => error,
        Err(_) => Exception::custom("physical backing publication unwound"),
    };
    let custody = PhysicalBackingCustody::new(std::sync::Arc::new(error));
    unsafe {
        *failure = custody.into_raw().0;
        *release = Some(PhysicalBackingCustody::release);
    }
    false
}
