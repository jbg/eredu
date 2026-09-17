//! Fixed byte reuse at final native quota retirement, independently of queued T.
use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct State {
    capacity: usize,
    occupied: AtomicUsize,
}

/// A closed native-storage capacity counter. This is not a funding authority:
/// callers must reserve its capacity and controls, and keep their actual custody
/// after every handle/permit. No native object or arbitrary callback is retained.
#[derive(Clone)]
pub struct RetirementCapacity {
    state: Arc<State>,
}
impl fmt::Debug for RetirementCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetirementCapacity")
            .field("capacity", &self.state.capacity)
            .finish_non_exhaustive()
    }
}
/// A refused acquisition or split that leaves the native capacity unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RetirementCapacityCause {
    /// This acquisition/split took no charge and allocated no native buffer.
    #[error("native source capacity exhausted: requested {requested}, available {available}")]
    Exhausted {
        /// Bytes requested by the acquisition or split.
        requested: usize,
        /// Unoccupied capacity or unsplit permit bytes available.
        available: usize,
    },
    /// A competing acquisition increased the observed occupied amount.
    #[error("native source capacity is busy")]
    Busy,
}
impl RetirementCapacity {
    /// One actual Arc/state allocation, after the caller accepts its layout.
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Arc::new(State {
                capacity,
                occupied: AtomicUsize::new(0),
            }),
        }
    }
    /// The accepted fixed capacity, without observing or progressing native work.
    pub fn capacity(&self) -> usize {
        self.state.capacity
    }
    /// Identity of this retained capacity owner; descriptive, not authority.
    pub fn owner_identity(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }
    /// Diagnostic snapshot only; never use this as native completion proof.
    pub fn occupied_bytes(&self) -> usize {
        self.state.occupied.load(Ordering::Acquire)
    }
    /// Checked acquisition; exhaustion takes no charge and allocates no buffer.
    /// A failed strong CAS retries only after strictly lower occupied bytes:
    /// refunds therefore cannot cause spurious Busy, and every retry descends a
    /// finite usize amount. Competing increases still refuse immediately.
    /// Success owns exactly one non-Clone permit and no new heap node.
    pub fn try_acquire(
        &self,
        bytes: usize,
    ) -> Result<RetirementCapacityPermit, RetirementCapacityCause> {
        let mut occupied = self.state.occupied.load(Ordering::Acquire);
        loop {
            let available = self.state.capacity - occupied;
            if bytes > available {
                return Err(RetirementCapacityCause::Exhausted {
                    requested: bytes,
                    available,
                });
            }
            if bytes == 0 {
                break;
            }
            match self.state.occupied.compare_exchange(
                occupied,
                occupied + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                // Strong CAS has no spurious failure. Retrying only strict
                // decreases cannot spin on an unchanged/increasing counter,
                // even if other clients also acquire or return capacity.
                Err(current) if current < occupied => occupied = current,
                Err(_) => return Err(RetirementCapacityCause::Busy),
            }
        }
        Ok(RetirementCapacityPermit {
            state: Arc::clone(&self.state),
            bytes,
        })
    }
    /// Requested Arc/state backing and the concrete new/acquire/refund/Drop
    /// frames. Per-quota owner/node storage is priced by that quota's T layout.
    pub fn control_bytes() -> Option<usize> {
        let allocation = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<State>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            mem::size_of::<Self>(),
            mem::size_of::<State>(),
            mem::size_of::<Arc<State>>(),
            mem::size_of::<RetirementCapacityPermit>(),
            mem::size_of::<Result<RetirementCapacityPermit, RetirementCapacityCause>>(),
            mem::size_of::<RetirementCapacityCause>(),
            mem::size_of::<Result<usize, usize>>(),
            mem::size_of::<usize>() * 4,
            mem::size_of::<&mut RetirementCapacityPermit>(),
            mem::size_of::<Result<RetirementCapacityPermit, RetirementCapacityCause>>(),
        ]
        .into_iter()
        .try_fold(allocation, usize::checked_add)
    }
}

/// One charged amount. Cancellation refunds once; successful quota handoff
/// refunds at its final native release and retains this handle until T retires.
/// No Clone, reset, raw owner, or manual early-refund operation is exposed.
/// Splitting transfers an existing charge without increasing the total.
pub struct RetirementCapacityPermit {
    state: Arc<State>,
    bytes: usize,
}
impl fmt::Debug for RetirementCapacityPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetirementCapacityPermit")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
impl RetirementCapacityPermit {
    /// Exact retained owner and unconsumed byte charge, without changing either.
    pub fn belongs_to(&self, capacity: &RetirementCapacity) -> bool {
        Arc::ptr_eq(&self.state, &capacity.state)
    }
    /// Unconsumed charge held by this move-only permit.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Move part of this already charged allowance to an independently retired
    /// native allocation. No counter change, new capacity or heap allocation.
    pub fn try_split(&mut self, bytes: usize) -> Result<Self, RetirementCapacityCause> {
        if bytes > self.bytes {
            return Err(RetirementCapacityCause::Exhausted {
                requested: bytes,
                available: self.bytes,
            });
        }
        let state = Arc::clone(&self.state);
        self.bytes -= bytes;
        Ok(Self { state, bytes })
    }
    fn retire_bytes(&mut self) {
        let bytes = mem::replace(&mut self.bytes, 0);
        if bytes != 0 && self.state.occupied.fetch_sub(bytes, Ordering::AcqRel) < bytes {
            // An internal ownership violation cannot cross the C callback by
            // unwinding, nor turn underflow into fresh allocation permission.
            std::process::abort();
        }
    }
}
impl Drop for RetirementCapacityPermit {
    fn drop(&mut self) {
        self.retire_bytes();
    }
}

/// Native retirement consumes only the fixed permit's byte charge. Its Arc and
/// the caller's arbitrary T still retire through the existing unlocked queue.
/// T must retain actual reserved custody and contain no backedge to this quota.
pub struct RetirementCapacityOwner<T: Send + 'static> {
    permit: RetirementCapacityPermit,
    owner: T,
}
impl<T: Send + 'static> RetirementCapacityOwner<T> {
    /// Borrow the unchanged custody without exposing its permit.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Cancel an unattached or returned preparation. Permit/Arc retire before T
    /// is returned, so its custody covers the complete failed constructor.
    pub fn into_owner(self) -> T {
        let Self { permit, owner } = self;
        drop(permit);
        owner
    }
}
impl<T: Send + 'static> fmt::Debug for RetirementCapacityOwner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetirementCapacityOwner")
            .field("permit", &self.permit)
            .finish_non_exhaustive()
    }
}

unsafe extern "C" fn retire_with_capacity<T: Send + 'static>(payload: *mut c_void) {
    // SAFETY: this callback is installed only with this exact initialized node.
    // Successful native handoff owns it exclusively until enqueue. GraphQuota
    // calls after occupied==0 and after freeing the native arena. In a prepared
    // host source, HostTransferStorage first frees PreparedInputAllocation's
    // backing; its shared allocator control then returns the final quota block.
    // Therefore the source backing is already gone before these bytes recycle.
    // No T, Arc, arbitrary destructor, allocation, or user callback runs here.
    unsafe {
        (*payload.cast::<OwnedNode<RetirementCapacityOwner<T>>>())
            .owner
            .permit
            .retire_bytes();
        retire(payload);
    }
}
impl<T: Send + 'static> PreparedSubmissionGraphQuota<RetirementCapacityOwner<T>> {
    /// Attach a precharged fixed permit to this quota's actual native lifetime.
    /// Failure owns the unchanged T+permit. Native construction takes the node
    /// only on success; Busy/failure cannot invoke the installed release callback.
    pub fn try_new_with_retirement(
        capacity: usize,
        owner: T,
        permit: RetirementCapacityPermit,
    ) -> Result<Self, SubmissionGraphQuotaError<RetirementCapacityOwner<T>>> {
        let mut prepared = Self::try_new(capacity, RetirementCapacityOwner { permit, owner })?;
        prepared.retire = retire_with_capacity::<T>;
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests;
