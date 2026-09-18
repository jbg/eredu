//! One immutable admitted plan and its exact retained host payload.
use super::*;
use crate::HostPreparationAuthority;
use crate::{
    backend::SharedStorageCustody, ObservationRequirement, ObservationValueType,
    SharedStorageAttachmentError, SharedStorageDomain, SharedStorageIdentity, SharedStorageOwner,
    SharedStorageRetirement, TensorAxis,
};
use std::{
    fmt,
    mem::size_of,
    sync::{Arc, Mutex},
};

struct Inner {
    // Automatic field retirement destroys all nested plan payload before custody.
    plan: AdmittedCapturePlan,
    // Only the closed limit-revision compiler installs this exact source alias.
    predecessor: Option<SharedCapturePlan>,
    #[cfg(test)]
    retired: Option<tests::PayloadRetired>,
    custody: SharedStorageCustody,
    #[cfg(test)]
    shell_retired: Option<Arc<std::sync::atomic::AtomicBool>>,
    #[cfg(test)]
    custody_retired: Option<tests::PayloadRetired>,
    // Final, independent host/account tokens only. No plan/session/native owner.
    ordinary: Mutex<Option<OrdinaryHostOwner>>,
}

/// Shared immutable ownership of one actually admitted capture plan.
///
/// Construction moves the existing plan without copying or shrinking any of its
/// buffers. Aliases created before accounting attachment retain that attachment.
/// This owner supplies source identity and exact retained capacity, not a native
/// quote, allocation permission, invocation or logical capture quota. Borrowed
/// plan clones are independent caller allocations and do not inherit custody.
#[derive(Clone)]
pub struct SharedCapturePlan(PlanOwner);

#[derive(Clone)]
struct PlanOwner(Option<Arc<Inner>>);
impl PlanOwner {
    fn get(&self) -> &Inner {
        self.0.as_ref().expect("live capture source")
    }
}
impl Drop for PlanOwner {
    fn drop(&mut self) {
        let mut pending = self.0.take().map(|owner| PlanOwner(Some(owner)));
        while let Some(mut tail) = pending.take() {
            if let Some(mut inner) = Arc::into_inner(tail.0.take().expect("live capture source")) {
                pending = inner.predecessor.take().map(|source| source.0);
                // All strong exits consume their Arc; no Weak/raw owner escapes.
                // Arc/control deallocation precedes payload, attachment Vec and
                // finally ordinary custody retirement.
                #[cfg(test)]
                if let Some(flag) = &inner.shell_retired {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                drop(inner);
            }
        }
    }
}
#[derive(Clone)]
struct OrdinaryHostOwner(Option<Arc<OrdinaryHostNode>>);
struct OrdinaryHostNode {
    previous: Option<OrdinaryHostOwner>,
    _incoming: HostPreparationAuthority,
    #[cfg(test)]
    allocation_retired: Option<Arc<std::sync::atomic::AtomicBool>>,
}
impl Drop for OrdinaryHostOwner {
    fn drop(&mut self) {
        let mut pending = self.0.take().map(|owner| OrdinaryHostOwner(Some(owner)));
        while let Some(mut tail) = pending.take() {
            let owner = tail.0.take().expect("live pending custody");
            if let Some(mut node) = Arc::into_inner(owner) {
                // Keep the remaining population in its closed wrapper: if this
                // incoming token unwinds, pending still consumes every Arc.
                // Normal retirement remains iterative and does not grow stack.
                pending = node.previous.take();
                #[cfg(test)]
                if let Some(flag) = &node.allocation_retired {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                drop(node);
            }
        }
    }
}

impl SharedCapturePlan {
    /// Move an existing semantic admission into shared ownership.
    pub fn new(plan: AdmittedCapturePlan) -> Self {
        Self(PlanOwner(Some(Arc::new(Inner {
            plan,
            predecessor: None,
            #[cfg(test)]
            retired: None,
            custody: SharedStorageCustody::new(),
            #[cfg(test)]
            shell_retired: None,
            #[cfg(test)]
            custody_retired: None,
            ordinary: Mutex::new(None),
        }))))
    }

    pub(super) fn from_prepared_copy(plan: AdmittedCapturePlan, host: HostPreparationAuthority,
        predecessor: Option<SharedCapturePlan>) -> Self {
        Self(PlanOwner(Some(Arc::new(Inner {
            plan,
            predecessor,
            #[cfg(test)] retired: None,
            custody: SharedStorageCustody::new(),
            #[cfg(test)] shell_retired: None,
            #[cfg(test)] custody_retired: None,
            ordinary: Mutex::new(Some(OrdinaryHostOwner(Some(Arc::new(OrdinaryHostNode {
                previous: None,
                _incoming: host,
                #[cfg(test)] allocation_retired: None,
            }))))),
        }))))
    }

    /// Concrete storage added when an already priced admitted payload moves
    /// into this owner: outer Arc/control/custody with an inline identity key.
    /// The inline admitted payload is excluded because capacity_bytes and
    /// existing ordinary checkpoint plans already count it. Attachments and
    /// allocator overhead remain separate; this is not a construction grant.
    pub fn new_owner_control_bytes() -> Option<u64> {
        let header = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>();
        let (outer, _) = header.extend(std::alloc::Layout::new::<Inner>()).ok()?;
        let bytes = outer
            .pad_to_align()
            .size()
            .checked_sub(std::mem::size_of::<AdmittedCapturePlan>())?;
        u64::try_from(bytes).ok()
    }

    /// One concrete closed host-retention node, allocated only by the ordinary
    /// bridge below. The source shell's inline field is included separately by
    /// new_owner_control_bytes; no payload byte is counted again.
    pub fn ordinary_host_control_bytes() -> Option<u64> {
        let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<OrdinaryHostNode>())
            .ok()?;
        u64::try_from(layout.pad_to_align().size()).ok()
    }

    /// Retains existing ordinary host custody on every alias of this actual
    /// source. The token must contain only independent host/account lifetimes,
    /// never this source, a session or a native owner whose metadata refers back.
    /// This grants no memory bound, execution, registration or completion.
    pub fn retain_host_preparation(
        &self,
        host: &HostPreparationAuthority,
    ) -> Result<(), CaptureError> {
        let mut slot =
            self.0.get().ordinary.lock().map_err(|_| {
                CaptureError::Invalid("capture source host custody is poisoned".into())
            })?;
        let next = OrdinaryHostOwner(Some(Arc::new(OrdinaryHostNode {
            previous: slot.clone(),
            _incoming: host.clone(),
            #[cfg(test)]
            allocation_retired: None,
        })));
        let previous = slot.replace(next);
        drop(slot);
        drop(previous);
        Ok(())
    }

    /// Borrow the exact retained admission without allocating another plan.
    pub fn admission(&self) -> &AdmittedCapturePlan {
        &self.0.get().plan
    }

    /// Process-local payload-owner identity, distinct from the semantic digest.
    pub fn storage_identity(&self) -> &SharedStorageIdentity {
        self.0.get().custody.identity()
    }

    /// Whether both handles retain the very same physical plan owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0 .0.as_ref().expect("live source"),
            other.0 .0.as_ref().expect("live source"),
        )
    }

    /// This owner was built by the checked limit-revision compiler from this
    /// exact predecessor. Equal declarations or digests provide no such proof.
    pub fn is_limit_revision_of(&self, source: &Self) -> bool {
        self.0.get().predecessor.as_ref().is_some_and(|parent| parent.same_storage(source))
    }

    /// Exact predecessor retained by the closed limit-revision compiler.
    /// This borrow supplies source lineage, never publication or execution credit.
    pub fn limit_revision_source(&self) -> Option<&Self> {
        self.0.get().predecessor.as_ref()
    }

    /// Complete retained plan payload, including inline DTOs and spare capacity.
    ///
    /// Counts all nested selections, points, axes, requirements, strings and
    /// transform-owned vectors. Custody/Arc/allocator bookkeeping and compiler
    /// stack are outside this retained payload. Overflow remains unknown. This
    /// reports existing storage; it is not a construction peak or funding grant.
    pub fn capacity_bytes(&self) -> Option<u64> {
        capacity(&self.0.get().plan)
    }

    /// Attach custody once per exact accounting domain, including earlier aliases.
    ///
    /// An existing domain returns false without invoking the provider. Otherwise
    /// the publication slot is reserved before acquisition. Rejection preserves
    /// earlier attachments. Provider errors retain their concrete cause.
    /// The provider runs under the custody lock and must perform only closed
    /// accounting: no owner reentry, native work or user callbacks. Its handle
    /// must not retain this plan; use the separate payload-free identity key to
    /// avoid a source/registration cycle. Custody retires after all plan payload
    /// and outside its lock. An acquisition panic poisons subsequent attachment.
    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.get().custody.try_attach(domain, acquire)
    }

    /// Nonblocking form of [`Self::try_attach`]. Busy never invokes the provider
    /// or changes attachments. All provider/lifetime rules are identical.
    /// The existing per-domain custody Vec may reserve metadata before provider
    /// acquisition; this method does not make that bookkeeping allocation-free.
    pub fn try_attach_nonblocking<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.get().custody.try_attach_nonblocking(domain, acquire)
    }
    /// Retain or retrieve a closed owned attachment for this exact domain.
    /// Opaque, raw-typed and differently typed owned custody reject. The raw
    /// typed API likewise rejects owned custody, preserving its closed exits.
    ///
    /// The provider obeys the closed-accounting restrictions of try_attach and
    /// must borrow staged resources; consumed closure captures may not run user
    /// destructors under custody. Unused closures and returned errors retire
    /// outside the lock. Existing metadata may reserve before acquisition; owner
    /// erasure/lookup/clone add no allocation. No funding or completion is implied.
    pub fn try_attach_owned_nonblocking<T: SharedStorageRetirement, E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<SharedStorageOwner<T>, E>,
    ) -> Result<SharedStorageOwner<T>, SharedStorageAttachmentError<E>> {
        self.0
            .get()
            .custody
            .try_attach_owned_nonblocking(domain, acquire)
    }

    /// Actual fixed attachment element and owned-dispatch/retirement controls.
    /// The concrete payload/header, metadata Vec capacity, source outer owner
    /// and allocator internals remain separate original facts. This checked
    /// diagnostic allocates nothing and grants no memory or execution authority.
    pub fn owned_attachment_control_bytes<T: SharedStorageRetirement>() -> Option<usize> {
        SharedStorageCustody::owned_attachment_control_bytes::<T>()
    }

    /// Retain or retrieve the exact typed owner attached in this domain.
    /// Opaque or differently typed prior custody rejects without invoking the
    /// provider. This authenticates owner identity/type only; the closed owner
    /// must validate its own physical key, capacity, domain and current health.
    /// Provider/lifetime constraints are those of [`Self::try_attach`]. Only Arc
    /// aliases are added; no typed wrapper is allocated inside the custody lock.
    pub fn try_attach_typed_nonblocking<T: Send + Sync + 'static, E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, SharedStorageAttachmentError<E>> {
        self.0
            .get()
            .custody
            .try_attach_typed_nonblocking(domain, acquire)
    }
}

impl std::ops::Deref for SharedCapturePlan {
    type Target = AdmittedCapturePlan;
    fn deref(&self) -> &Self::Target {
        self.admission()
    }
}
impl AsRef<AdmittedCapturePlan> for SharedCapturePlan {
    fn as_ref(&self) -> &AdmittedCapturePlan {
        self.admission()
    }
}
impl fmt::Debug for SharedCapturePlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedCapturePlan")
            .field("identity", &self.admission().identity())
            .field("storage_identity", self.storage_identity())
            .field("capacity_bytes", &self.capacity_bytes())
            .finish_non_exhaustive()
    }
}

fn bytes<T>(capacity: usize) -> Option<u64> {
    u64::try_from(capacity)
        .ok()?
        .checked_mul(size_of::<T>() as u64)
}
fn text_bytes(text: &String) -> Option<u64> {
    bytes::<u8>(text.capacity())
}
fn capacity(source: &AdmittedCapturePlan) -> Option<u64> {
    let AdmittedCapturePlan {
        plan,
        points,
        identity,
        request: _,
        invocation_bounds: _,
        text_origin: _,
    } = source;
    let CapturePlan {
        selections,
        schema_version: _,
        limits: _,
    } = plan;
    let mut total = (size_of::<AdmittedCapturePlan>() as u64)
        .checked_add(bytes::<CaptureSelection>(selections.capacity())?)?
        .checked_add(bytes::<ObservationPoint>(points.capacity())?)?
        .checked_add(text_bytes(identity)?)?;
    for selection in selections {
        let CaptureSelection {
            id,
            path,
            slices,
            transform,
            schedule: _,
        } = selection;
        total = total
            .checked_add(text_bytes(id)?)?
            .checked_add(text_bytes(path)?)?
            .checked_add(bytes::<CaptureSlice>(slices.capacity())?)?;
        for CaptureSlice {
            axis,
            start: _,
            end: _,
            stride: _,
        } in slices
        {
            total = total.checked_add(text_bytes(axis)?)?;
        }
        total = total.checked_add(match transform {
            CaptureTransform::Histogram { edges } => bytes::<f32>(edges.capacity())?,
            CaptureTransform::TokenScores { token_ids } => bytes::<u32>(token_ids.capacity())?,
            CaptureTransform::RoutedUnits
            | CaptureTransform::Preview { .. }
            | CaptureTransform::Slice
            | CaptureTransform::FullTensor
            | CaptureTransform::Summary
            | CaptureTransform::TopCandidates { .. } => 0,
        })?;
    }
    for point in points {
        let ObservationPoint {
            path,
            node_id,
            meaning,
            value_type,
            axes,
            requirements,
            dtype: _,
            prefill: _,
            decode: _,
            position: _,
            retained_bytes: _,
            host_bytes: _,
        } = point;
        total = total
            .checked_add(text_bytes(path)?)?
            .checked_add(text_bytes(node_id)?)?
            .checked_add(text_bytes(meaning)?)?
            .checked_add(bytes::<ObservationRequirement>(requirements.capacity())?)?;
        total = total.checked_add(match value_type {
            ObservationValueType::Tensor => 0,
            ObservationValueType::RoutedUnits {
                routing,
                geometry: _,
            } => text_bytes(routing)?,
        })?;
        if let Some(axes) = axes {
            total = total.checked_add(bytes::<TensorAxis>(axes.capacity())?)?;
            for TensorAxis { name, dimension } in axes {
                total = total.checked_add(text_bytes(name)?)?;
                // All current symbolic dimensions are inline. Exhaustive match
                // makes a future payload-bearing dimension require review here.
                match dimension {
                    SymbolicDimension::Known(_)
                    | SymbolicDimension::Batch
                    | SymbolicDimension::Sequence
                    | SymbolicDimension::TokenRows
                    | SymbolicDimension::Context
                    | SymbolicDimension::MediaPositions
                    | SymbolicDimension::Unknown => {}
                }
            }
        }
    }
    Some(total)
}

#[cfg(test)]
mod tests;
