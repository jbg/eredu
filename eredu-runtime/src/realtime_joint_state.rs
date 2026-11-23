//! Portable atomic transaction and copy admission for coupled realtime caches.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use eredu_core::{
    cache::mechanisms::{CacheCopyScope, CacheCopySubmission},
    execution_control::SnapshotEstimate,
    realtime_mechanisms::RealtimeJointStateParts,
    scheduler::SemanticStateTransaction,
    BackendFailure,
};

use crate::{
    execution_control::{
        ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork, SnapshotReservation,
    },
    replicated_session::StateSnapshotMemoryCosts,
    state::{
        HybridCacheStorage, HybridState, HybridStateError, KeyValueLayerState, KeyValueState,
        KeyValueStateError,
    },
};

/// Finite copy limits for a joint state transaction.
#[derive(Debug, Clone, Copy)]
pub struct RealtimeCopyLimits {
    /// Maximum storage and temporary copy bytes for one branch.
    pub max_branch_bytes: u64,
    /// Cumulative copy allowance shared by isolated descendants.
    pub max_total_copy_bytes: u64,
}

/// Failure of an atomic realtime state transaction.
#[derive(Debug, thiserror::Error)]
pub enum JointStateError {
    /// Native error retains its original source.
    #[error(transparent)]
    Native(#[from] BackendFailure),
    /// The backbone state could not be copied into an independent owner.
    #[error(transparent)]
    Backbone(#[from] HybridStateError),
    /// The speech state could not be copied into an independent owner.
    #[error(transparent)]
    Speech(#[from] KeyValueStateError),
    /// A state cannot be independently copied with its selected realization.
    #[error("joint state has inconsistent isolated-copy storage")]
    UnsupportedStorage,
    /// A previous branch or unresolved copy retains the exclusive admission.
    #[error("joint state already has an outstanding branch or copy")]
    Busy,
    /// Admission would exceed the configured finite copy allowance.
    #[error("joint state copy exceeds its configured budget")]
    Budget,
    /// Accounting or generation identity cannot be represented.
    #[error("joint state accounting or generation overflow")]
    Overflow,
    /// A branch does not originate at this canonical boundary.
    #[error("joint state branch identity or layout does not match")]
    Identity,
}

struct Admission {
    active: AtomicBool,
    accounting: Arc<CopyAccounting>,
    // Fork ownership follows CopyPermit into execution and native recovery.
    // It is deliberately independent of accounting shared with the source.
    _fork_retention: Option<SnapshotReservation>,
}

// Shared by isolated descendants. Pointer identity is also the originating
// cache owner's proof; independent construction never grants restore authority.
struct CopyAccounting {
    consumed: AtomicU64,
    limits: RealtimeCopyLimits,
}

#[derive(Clone)]
struct CopyPermit(Arc<CopyPermitOwner>);
struct CopyPermitOwner(Arc<Admission>);
impl Drop for CopyPermitOwner {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}

struct PendingStateCopy<M: HybridCacheStorage, A> {
    source_backbone: HybridState<M>,
    source_speech: KeyValueState<M>,
    backbone: Option<HybridState<M>>,
    speech: Option<KeyValueState<M>>,
    _authority: A,
}

/// One canonical owner of heterogeneous and key/value cache families.
///
/// Candidates share immutable roots and copy mutable storage and paging namespaces in the same
/// finite process pools. They remain exclusive until commit/discard, including
/// after a failed copy with unresolved native work. The enclosing scheduler
/// establishes execution completion before committing or discarding a branch.
pub struct JointRealtimeState<M: HybridCacheStorage + CacheCopySubmission> {
    left: HybridState<M>,
    right: KeyValueState<M>,
    copy_queue: M::Queue,
    admission: Arc<Admission>,
    generation: u64,
}

/// Unpublished joint cache candidate. Native execution completion is caller-owned.
pub struct JointRealtimeBranch<M: HybridCacheStorage + CacheCopySubmission> {
    left: HybridState<M>,
    right: KeyValueState<M>,
    permit: CopyPermit,
    generation: u64,
    identity: Vec<Option<u64>>,
}

impl<M: HybridCacheStorage + CacheCopySubmission> JointRealtimeBranch<M> {
    /// Borrows the native state values retained by the branch.
    pub fn parts(&self) -> (&HybridState<M>, &KeyValueState<M>) {
        (&self.left, &self.right)
    }

    /// Shares lifetime authority and native roots with the submission owner.
    pub fn retention(&self) -> impl Sized + 'static {
        (self.left.clone(), self.right.clone(), self.permit.clone())
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> RealtimeJointStateParts
    for JointRealtimeBranch<M>
{
    type Left = HybridState<M>;
    type Right = [KeyValueLayerState<M>];

    fn parts_mut(&mut self) -> (&mut Self::Left, &mut Self::Right) {
        (&mut self.left, self.right.as_mut())
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> JointRealtimeState<M> {
    /// Adopts the selected state and a finite copy budget.
    pub fn new(
        left: HybridState<M>,
        right: KeyValueState<M>,
        copy_queue: M::Queue,
        limits: RealtimeCopyLimits,
    ) -> Result<Self, JointStateError> {
        if M::snapshot_tensor_costs().is_none() || !supports_copy(&left, &right) {
            return Err(JointStateError::UnsupportedStorage);
        }
        Ok(Self {
            left,
            right,
            copy_queue,
            admission: Arc::new(Admission {
                active: AtomicBool::new(false),
                accounting: Arc::new(CopyAccounting {
                    consumed: AtomicU64::new(0),
                    limits,
                }),
                _fork_retention: None,
            }),
            generation: 0,
        })
    }

    /// Borrows the committed components.
    pub fn parts(&self) -> (&HybridState<M>, &KeyValueState<M>) {
        (&self.left, &self.right)
    }

    /// Cumulative admitted copying, shared by descendants.
    pub fn copy_bytes_consumed(&self) -> u64 {
        self.admission.accounting.consumed.load(Ordering::Acquire)
    }

    fn estimate(
        left: &HybridState<M>,
        right: &KeyValueState<M>,
    ) -> Result<SnapshotEstimate, JointStateError> {
        let storage = Self::storage_estimate(left, right)?;
        let bookkeeping = std::mem::size_of::<Self>()
            .max(std::mem::size_of::<JointRealtimeBranch<M>>()) as u64
            + std::mem::size_of::<Admission>() as u64
            + (2 * std::mem::size_of::<usize>()) as u64;
        Ok(SnapshotEstimate {
            retained_bytes: storage
                .retained_bytes
                .checked_add(bookkeeping)
                .ok_or(JointStateError::Overflow)?,
            copy_bytes: storage
                .copy_bytes
                .checked_add(bookkeeping)
                .ok_or(JointStateError::Overflow)?,
        })
    }

    fn storage_estimate(
        left: &HybridState<M>,
        right: &KeyValueState<M>,
    ) -> Result<SnapshotEstimate, JointStateError> {
        let costs = StateSnapshotMemoryCosts::for_tensors(
            M::snapshot_tensor_costs().ok_or(JointStateError::UnsupportedStorage)?,
        );
        let mut bytes = 0u64;
        for layout in left.optional_layout().into_iter().chain([right.layout()]) {
            bytes = bytes
                .checked_add(
                    layout
                        .logical_metadata_bytes()
                        .ok_or(JointStateError::Overflow)?,
                )
                .ok_or(JointStateError::Overflow)?;
            for layer in 0..layout.len() {
                let components = layout
                    .components(layer)
                    .ok_or(JointStateError::Overflow)?
                    .len();
                bytes = bytes
                    .checked_add(costs.layer_metadata)
                    .and_then(|n| {
                        n.checked_add((components as u64).checked_mul(costs.slot_metadata)?)
                    })
                    .ok_or(JointStateError::Overflow)?;
            }
        }
        for tensor in left
            .retained_arrays()
            .into_iter()
            .chain(right.retained_arrays())
        {
            bytes = bytes
                .checked_add(
                    costs
                        .tensor_bytes(M::bytes(tensor), M::shape(tensor).len())
                        .ok_or(JointStateError::Overflow)?,
                )
                .ok_or(JointStateError::Overflow)?;
        }
        bytes = bytes
            .checked_add(
                left.isolated_snapshot_auxiliary_bytes()
                    .ok_or(JointStateError::Overflow)?,
            )
            .and_then(|n| n.checked_add(right.isolated_snapshot_auxiliary_bytes()?))
            .ok_or(JointStateError::Overflow)?;
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn admit(&self, estimate: SnapshotEstimate) -> Result<CopyPermit, JointStateError> {
        if estimate.retained_bytes > self.admission.accounting.limits.max_branch_bytes {
            return Err(JointStateError::Budget);
        }
        if self.admission.active.swap(true, Ordering::AcqRel) {
            return Err(JointStateError::Busy);
        }
        let permit = CopyPermit(Arc::new(CopyPermitOwner(self.admission.clone())));
        self.admission
            .accounting
            .consumed
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(estimate.copy_bytes)
                    .filter(|total| *total <= self.admission.accounting.limits.max_total_copy_bytes)
            })
            .map_err(|_| JointStateError::Budget)?;
        Ok(permit)
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> JointRealtimeState<M> {
    fn copy_candidate(
        &self,
        retention: Option<&SnapshotReservation>,
        isolated: bool,
    ) -> Result<JointRealtimeBranch<M>, JointStateError> {
        M::reclaim_copies();
        let immutable =
            !isolated && self.left.has_immutable_roots() && self.right.has_immutable_roots();
        let mut estimate = Self::estimate(&self.left, &self.right)?;
        if immutable {
            estimate.copy_bytes = 0;
        }
        let permit = self.admit(estimate)?;
        let (left, right) = if immutable {
            (self.left.clone(), self.right.clone())
        } else {
            self.copy_components(isolated, (permit.clone(), retention.cloned()))?
        };
        let identity = paging_identity(&left, &right);
        Ok(JointRealtimeBranch {
            left,
            right,
            identity,
            permit,
            generation: self.generation,
        })
    }

    fn copy_components(
        &self,
        isolated: bool,
        authority: impl Sized + 'static,
    ) -> Result<(HybridState<M>, KeyValueState<M>), JointStateError> {
        // The scope keeps source catalogs and any partial candidates live
        // until every submitted copy settles, including a failed second copy.
        let mut scope = M::begin_copy(
            &self.copy_queue,
            PendingStateCopy {
                source_backbone: self.left.clone(),
                source_speech: self.right.clone(),
                backbone: None,
                speech: None,
                _authority: authority,
            },
        )?;
        let source = &scope.resources().source_backbone;
        let backbone = if isolated {
            source.isolated_snapshot(&self.copy_queue)?
        } else {
            source.transaction_candidate(&self.copy_queue)?
        };
        scope.resources_mut().backbone = Some(backbone);
        let source = &scope.resources().source_speech;
        let speech = if isolated {
            source.isolated_snapshot(&self.copy_queue)?
        } else {
            source.transaction_candidate(&self.copy_queue)?
        };
        scope.resources_mut().speech = Some(speech);

        let resources = scope.resources();
        let roots = resources
            .backbone
            .as_ref()
            .expect("copied backbone")
            .retained_arrays()
            .into_iter()
            .chain(
                resources
                    .speech
                    .as_ref()
                    .expect("copied speech")
                    .retained_arrays(),
            )
            .cloned()
            .collect::<Vec<_>>();
        scope.settle(&roots)?;
        let resources = scope.resources_mut();
        let copied = (
            resources.backbone.take().expect("settled backbone"),
            resources.speech.take().expect("settled speech"),
        );
        scope.finish();
        Ok(copied)
    }

    fn copy_owner(
        &self,
        retention: &SnapshotReservation,
        fork_retention: Option<SnapshotReservation>,
    ) -> Result<Self, JointStateError> {
        let JointRealtimeBranch {
            left,
            right,
            generation,
            ..
        } = self.copy_candidate(Some(retention), true)?;
        Ok(Self {
            left,
            right,
            copy_queue: self.copy_queue.clone(),
            admission: Arc::new(Admission {
                active: AtomicBool::new(false),
                accounting: self.admission.accounting.clone(),
                _fork_retention: fork_retention,
            }),
            generation,
        })
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> ForkStateEstimate for JointRealtimeState<M> {
    fn estimate_fork_branch(branch: &Self::Branch) -> Result<u64, BackendFailure> {
        Self::estimate(&branch.left, &branch.right)
            .map(|estimate| estimate.retained_bytes)
            .map_err(BackendFailure::from_error)
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> SemanticStateTransaction
    for JointRealtimeState<M>
{
    type Branch = JointRealtimeBranch<M>;
    type Error = JointStateError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        self.copy_candidate(None, false)
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        let next = self
            .generation
            .checked_add(1)
            .ok_or(JointStateError::Overflow)?;
        if !Arc::ptr_eq(&self.admission, &branch.permit.0 .0)
            || self.generation != branch.generation
            || !same_layout(self.parts(), (&branch.left, &branch.right))
            || !supports_copy(&branch.left, &branch.right)
            || branch.identity != paging_identity(&branch.left, &branch.right)
        {
            return Err(JointStateError::Identity);
        }
        self.left = branch.left;
        self.right = branch.right;
        self.generation = next;
        Ok(())
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> IsolatedStateCopy for JointRealtimeState<M> {
    type Error = JointStateError;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
        Self::estimate(&self.left, &self.right)
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_owner(retention, None)
    }

    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
        let copy_active = || {
            self.admission.active.load(Ordering::Acquire)
                || saved.admission.active.load(Ordering::Acquire)
        };
        if copy_active() {
            // Native retirement can hold a completed invocation's copy permit.
            M::reclaim_copies();
            if copy_active() {
                return Err(JointStateError::Busy);
            }
        }
        self.generation
            .checked_add(1)
            .ok_or(JointStateError::Overflow)?;
        if !Arc::ptr_eq(&self.admission.accounting, &saved.admission.accounting)
            || !same_layout(self.parts(), (&saved.left, &saved.right))
            || !supports_copy(&saved.left, &saved.right)
        {
            return Err(JointStateError::Identity);
        }
        Ok(())
    }

    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
        self.validate_isolated_restore(&saved)?;
        // Every fallible check precedes installation. The live owner's identity,
        // stream and shared cumulative accounting survive this replacement.
        self.left = saved.left;
        self.right = saved.right;
        self.generation += 1;
        Ok(())
    }
}

impl<M: HybridCacheStorage + CacheCopySubmission> IsolatedStateFork for JointRealtimeState<M> {
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_owner(retention, Some(retention.clone()))
    }
}

fn supports_copy<M: HybridCacheStorage>(left: &HybridState<M>, right: &KeyValueState<M>) -> bool {
    left.supports_isolated_snapshot() && right.supports_isolated_snapshot()
}

fn same_layout<M: HybridCacheStorage>(
    left: (&HybridState<M>, &KeyValueState<M>),
    right: (&HybridState<M>, &KeyValueState<M>),
) -> bool {
    left.0.has_same_snapshot_layout(right.0) && left.1.has_same_snapshot_layout(right.1)
}

fn paging_identity<M: HybridCacheStorage>(
    left: &HybridState<M>,
    right: &KeyValueState<M>,
) -> Vec<Option<u64>> {
    std::iter::once(left.residency_manager().map(|manager| manager.session_id()))
        .chain(right.as_ref().iter().map(|layer| match layer {
            KeyValueLayerState::Paged(cache) => Some(cache.manager().session_id()),
            _ => None,
        }))
        .collect()
}
