//! A private completion mailbox for one already-paid attachment node.
use super::*;
use std::{
    marker::PhantomData,
    ptr,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};

pub(super) struct Slot {
    // null: callback pending; CLOSED: receiver abandoned/consumed; otherwise
    // the unique node whose native backing has actually retired.
    node: AtomicPtr<RetiredOwner>,
}
const CLOSED: *mut RetiredOwner = ptr::without_provenance_mut(1);

/// Owner-thread receiver for one prepared attachment's actual retirement.
///
/// This neither waits for native work nor scans the global retirement queue.
/// Dropping the receiver transfers an already-retired node, or its future
/// callback, to the ordinary queue. No drop can release a live native alias.
/// The receiver retains no array, graph, native object, or caller payload.
pub struct PreparedAllocationRetirement {
    slot: Arc<Slot>,
    // Native callbacks may run anywhere; payload reclamation remains on the
    // thread that prepared this receiver, outside the native runtime lock.
    _thread: PhantomData<Rc<()>>,
}
impl fmt::Debug for PreparedAllocationRetirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedAllocationRetirement")
            .finish_non_exhaustive()
    }
}
impl PreparedAllocationRetirement {
    /// Reclaim only this attachment if native backing has retired. Returns true
    /// only after its exact node and payload are destroyed. Pending callbacks,
    /// consumed receivers, runtime entry, recursion and unwinding return false.
    /// A panicking payload consumes this node once and leaves other owners alone.
    pub fn try_reclaim(&mut self) -> bool {
        let Some(_guard) = super::super::begin_reclamation() else {
            return false;
        };
        let node = self.slot.node.load(Ordering::Acquire);
        if node.is_null() || node == CLOSED {
            return false;
        }
        // &mut self excludes another receiver operation. The native callback
        // publishes once and never accesses its node after the successful CAS.
        self.slot.node.store(CLOSED, Ordering::Release);
        // SAFETY: the acquire load receives the exclusive, initialized node
        // only after native retirement. Its matching destructor frees the node
        // before invoking T::drop, under the shared reentrancy exclusion.
        unsafe { ((*node).destroy)(node) };
        true
    }
}
impl Drop for PreparedAllocationRetirement {
    fn drop(&mut self) {
        let node = self.slot.node.swap(CLOSED, Ordering::AcqRel);
        if !node.is_null() && node != CLOSED {
            // SAFETY: swap exclusively removes the one completed node. Drop
            // only queues it, even during unwinding or under the native lock.
            unsafe { super::super::enqueue(node) };
        }
    }
}

pub(super) unsafe extern "C" fn retire<T: Send + 'static>(payload: *mut c_void) {
    let node = payload.cast::<OwnedNode<T>>();
    // SAFETY: all prepared attach routes install this callback with this exact
    // initialized node. Native final retirement owns it until publication.
    let slot = unsafe { (*node).retirement.as_ref().map(Arc::as_ptr) };
    let Some(slot) = slot else {
        unsafe { super::super::retire(payload) };
        return;
    };
    // Read only the slot pointer before publishing. A successful receiver may
    // immediately free the node and its Arc; no node/slot access follows success.
    // The callback neither clones/drops an Arc nor runs an arbitrary destructor.
    let result = unsafe {
        (*slot).node.compare_exchange(
            ptr::null_mut(),
            payload.cast(),
            Ordering::Release,
            Ordering::Relaxed,
        )
    };
    match result {
        Ok(_) => {}
        Err(value) if value == CLOSED => unsafe { super::super::retire(payload) },
        // A second callback is an internal ownership violation; fail closed
        // rather than enqueue the same node twice or unwind across native code.
        Err(_) => std::process::abort(),
    }
}

impl<T: Send + 'static> PreparedAllocationOwner<T> {
    /// Prepare the same checked attachment with a private retirement receiver.
    /// Add `retirement_control_bytes()` to the ordinary `layout()` facts before
    /// construction. Failure returns the unchanged unattached owner.
    pub fn try_new_with_retirement(
        owner: T,
    ) -> Result<(Self, PreparedAllocationRetirement), PreparedAllocationOwnerError<T>> {
        let mut prepared = Self::try_new(owner)?;
        let slot = Arc::new(Slot {
            node: AtomicPtr::new(ptr::null_mut()),
        });
        prepared.node.as_mut().unwrap().retirement = Some(Arc::clone(&slot));
        Ok((
            prepared,
            PreparedAllocationRetirement {
                slot,
                _thread: PhantomData,
            },
        ))
    }

    /// Additional actual shared allocation and named constructor, callback,
    /// receiver, failure, reclaim and Drop controls. Node storage, checked
    /// attachment and T itself remain covered by the ordinary layout query.
    pub fn retirement_control_bytes() -> Option<usize> {
        let allocation = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Slot>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            size_of::<Slot>(), size_of::<Arc<Slot>>(),
            size_of::<Option<Arc<Slot>>>(), size_of::<PreparedAllocationRetirement>(),
            size_of::<(T, Self, Arc<Slot>, PreparedAllocationRetirement)>(),
            size_of::<Result<(Self, PreparedAllocationRetirement), PreparedAllocationOwnerError<T>>>(),
            size_of::<(&mut PreparedAllocationRetirement, *mut RetiredOwner)>(),
            size_of::<Option<super::super::ReclamationGuard>>(),
            size_of::<Result<bool, std::thread::AccessError>>(),
            size_of::<(*mut c_void, *mut OwnedNode<T>, Option<*const Slot>)>(),
            size_of::<Result<*mut RetiredOwner, *mut RetiredOwner>>(),
            size_of::<(Box<OwnedNode<T>>, Option<Arc<Slot>>, T)>(),
        ].into_iter().try_fold(allocation, usize::checked_add)
    }
}

#[cfg(test)]
mod tests;
