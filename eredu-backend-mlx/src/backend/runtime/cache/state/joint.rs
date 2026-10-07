//! Atomic ownership of independently copied heterogeneous and attention state.

use super::*;
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork, SnapshotReservation,
};
use safemlx::{transforms::async_eval_with_event, Event};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

/// Copy admission limits. Execution growth is admitted by the enclosing session.
#[derive(Debug, Clone, Copy)]
pub struct JointCopyLimits {
    /// Maximum conservative storage and temporary-copy estimate for one branch.
    pub max_branch_bytes: u64,
    /// Cumulative copy allowance, including discarded and failed copies.
    pub max_total_copy_bytes: u64,
}

/// Joint native transaction or copy admission failure.
#[derive(Debug, thiserror::Error)]
pub enum JointStateError {
    /// Native error retains its original source.
    #[error(transparent)]
    Native(#[from] Exception),
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
    limits: JointCopyLimits,
}

#[derive(Clone)]
struct CopyPermit(Arc<CopyPermitOwner>);
struct CopyPermitOwner(Arc<Admission>);
impl Drop for CopyPermitOwner {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::Release);
    }
}

/// One canonical owner of heterogeneous and key/value cache families.
///
/// Branch copies have independent arrays and paging namespaces in the same
/// finite process pools. They remain exclusive until commit/discard, including
/// after a failed copy with unresolved native work. The enclosing scheduler
/// establishes execution completion before committing or discarding a branch.
pub struct MlxJointState {
    hybrid: MlxHybridState,
    key_value: MlxKeyValueState,
    stream: Stream,
    admission: Arc<Admission>,
    generation: u64,
}

/// Unpublished joint cache candidate. Native execution completion is caller-owned.
pub struct MlxJointStateBranch {
    hybrid: MlxHybridState,
    key_value: MlxKeyValueState,
    permit: CopyPermit,
    generation: u64,
    paging_identity: Vec<Option<u64>>,
}

impl MlxJointStateBranch {
    /// Shares lifetime authority, without making another transaction or copying
    /// native values. Completion retains this before any eager execution work.
    pub(crate) fn retention(&self) -> impl Sized + 'static {
        JointRetention {
            _hybrid: self.hybrid.clone(),
            _key_value: self.key_value.clone(),
            _permit: self.permit.clone(),
        }
    }
    /// Borrows the two independent cache families for one coupled invocation.
    pub fn parts_mut(&mut self) -> (&mut MlxHybridState, &mut [MlxKeyValueLayerState]) {
        (&mut self.hybrid, self.key_value.as_mut())
    }

    /// Native values retained by the candidate, including every batch lane.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        arrays(&self.hybrid, &self.key_value)
    }
}

impl MlxJointState {
    /// Adopts selected cache realizations and finite copy limits without copying.
    pub fn new(
        hybrid: MlxHybridState,
        key_value: MlxKeyValueState,
        stream: &Stream,
        limits: JointCopyLimits,
    ) -> Result<Self, JointStateError> {
        if !hybrid.supports_isolated_snapshot() || !key_value.supports_isolated_snapshot() {
            return Err(JointStateError::UnsupportedStorage);
        }
        Ok(Self {
            hybrid,
            key_value,
            stream: stream.clone(),
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

    /// Committed component owners for observation and resource accounting.
    pub fn parts(&self) -> (&MlxHybridState, &MlxKeyValueState) {
        (&self.hybrid, &self.key_value)
    }

    /// Native values retained by both canonical cache families.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        arrays(&self.hybrid, &self.key_value)
    }

    /// Cumulative admitted copying; discard and failures do not refund it.
    pub fn copy_bytes_consumed(&self) -> u64 {
        self.admission.accounting.consumed.load(Ordering::Acquire)
    }

    /// Actual logical bytes in currently retained native tensors.
    /// Paged host/disk catalogs are additionally priced by `copy_estimate`.
    pub fn retained_tensor_bytes(&self) -> Result<u64, JointStateError> {
        self.retained_arrays()
            .into_iter()
            .try_fold(0u64, |sum, array| {
                sum.checked_add(
                    u64::try_from(array.nbytes()).map_err(|_| JointStateError::Overflow)?,
                )
                .ok_or(JointStateError::Overflow)
            })
    }

    /// Conservative copy reservation using actual tensor shapes, including batch
    /// lanes, contiguous-copy temporaries, and complete paged catalog storage.
    pub fn copy_estimate(&self) -> Result<SnapshotEstimate, JointStateError> {
        component_copy_estimate(&self.hybrid, &self.key_value)
    }

    fn admit(&self) -> Result<CopyPermit, JointStateError> {
        let estimate = self.copy_estimate()?;
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

fn component_copy_estimate(
    hybrid: &MlxHybridState,
    key_value: &MlxKeyValueState,
) -> Result<SnapshotEstimate, JointStateError> {
    let mut bytes = std::mem::size_of::<MlxJointStateBranch>()
        .max(std::mem::size_of::<MlxJointState>()) as u64
        + std::mem::size_of::<Admission>() as u64
        + (2 * std::mem::size_of::<usize>()) as u64;
    for layout in [hybrid.layout(), key_value.layout()] {
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
                .checked_add(1024)
                .and_then(|n| n.checked_add((components as u64).checked_mul(256)?))
                .ok_or(JointStateError::Overflow)?;
        }
    }
    for array in arrays(hybrid, key_value) {
        bytes = bytes
            .checked_add(
                (array.nbytes() as u64)
                    .checked_mul(2)
                    .ok_or(JointStateError::Overflow)?,
            )
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| n.checked_add((array.shape().len() as u64).checked_mul(16)?))
            .ok_or(JointStateError::Overflow)?;
    }
    bytes = bytes
        .checked_add(
            hybrid
                .isolated_snapshot_auxiliary_bytes()
                .ok_or(JointStateError::Overflow)?,
        )
        .and_then(|n| n.checked_add(key_value.isolated_snapshot_auxiliary_bytes()?))
        .ok_or(JointStateError::Overflow)?;
    Ok(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

impl ForkStateEstimate for MlxJointState {
    fn estimate_fork_branch(branch: &Self::Branch) -> Result<u64, BackendFailure> {
        component_copy_estimate(&branch.hybrid, &branch.key_value)
            .map(|estimate| estimate.retained_bytes)
            .map_err(BackendFailure::from_error)
    }
}

struct CopyResources {
    source_hybrid: MlxHybridState,
    source_key_value: MlxKeyValueState,
    hybrid: Option<MlxHybridState>,
    key_value: Option<MlxKeyValueState>,
    permit: Option<CopyPermit>,
    event: Option<Event>,
    _stream: Stream,
    _snapshot_retention: Option<SnapshotReservation>,
}
struct CopyRetention(OrdinaryRetirement<CopyResources>);
impl Deref for CopyRetention {
    type Target = CopyResources;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CopyRetention {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Retention for CopyRetention {
    fn observe(&self, _: Status) {}
}

impl MlxJointState {
    fn copy_candidate(
        &self,
        retention: Option<&SnapshotReservation>,
    ) -> Result<MlxJointStateBranch, JointStateError> {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        let permit = self.admit()?;
        let mut recovery =
            Recovery::begin(CopyRetention(OrdinaryRetirement::new(CopyResources {
                source_hybrid: self.hybrid.clone(),
                source_key_value: self.key_value.clone(),
                hybrid: None,
                key_value: None,
                permit: Some(permit),
                event: None,
                _stream: self.stream.clone(),
                _snapshot_retention: retention.cloned(),
            })))?;
        let hybrid = recovery
            .retention()
            .source_hybrid
            .isolated_snapshot(&self.stream)?;
        recovery.retention_mut().hybrid = Some(hybrid);
        let key_value = recovery
            .retention()
            .source_key_value
            .isolated_snapshot(&self.stream)?;
        recovery.retention_mut().key_value = Some(key_value);
        let values = arrays(
            recovery.retention().hybrid.as_ref().expect("copied hybrid"),
            recovery
                .retention()
                .key_value
                .as_ref()
                .expect("copied attention"),
        );
        let event = async_eval_with_event(values)?;
        recovery.retention_mut().event = Some(event);
        recovery.seal();
        let status = recovery.wait();
        if status.failed || status.blocked {
            return Err(Exception::custom(
                "joint state copy failed; unresolved resources remain retained",
            )
            .into());
        }
        recovery
            .retention()
            .event
            .as_ref()
            .expect("copy event")
            .synchronize()?;
        let resources = recovery.retention_mut();
        let branch = MlxJointStateBranch {
            paging_identity: paging_identity(
                resources.hybrid.as_ref().expect("copied hybrid"),
                resources.key_value.as_ref().expect("copied attention"),
            ),
            hybrid: resources.hybrid.take().expect("copied hybrid"),
            key_value: resources.key_value.take().expect("copied attention"),
            permit: resources.permit.take().expect("copy admission"),
            generation: self.generation,
        };
        recovery.finish();
        ordinary_retirement::reclaim();
        Ok(branch)
    }
}

impl SemanticStateTransaction for MlxJointState {
    type Branch = MlxJointStateBranch;
    type Error = JointStateError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        self.copy_candidate(None)
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        let next = self
            .generation
            .checked_add(1)
            .ok_or(JointStateError::Overflow)?;
        if !Arc::ptr_eq(&self.admission, &branch.permit.0 .0)
            || self.generation != branch.generation
            || !self.hybrid.has_same_snapshot_layout(&branch.hybrid)
            || !self.key_value.has_same_snapshot_layout(&branch.key_value)
            || !branch.hybrid.supports_isolated_snapshot()
            || !branch.key_value.supports_isolated_snapshot()
            || branch.paging_identity != paging_identity(&branch.hybrid, &branch.key_value)
        {
            return Err(JointStateError::Identity);
        }
        self.hybrid = branch.hybrid;
        self.key_value = branch.key_value;
        self.generation = next;
        Ok(())
    }
}

impl IsolatedStateCopy for MlxJointState {
    type Error = JointStateError;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
        self.copy_estimate()
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_owner(retention, None)
    }

    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
        if self.admission.active.load(Ordering::Acquire)
            || saved.admission.active.load(Ordering::Acquire)
        {
            return Err(JointStateError::Busy);
        }
        self.generation
            .checked_add(1)
            .ok_or(JointStateError::Overflow)?;
        if !Arc::ptr_eq(&self.admission.accounting, &saved.admission.accounting)
            || !self.hybrid.has_same_snapshot_layout(&saved.hybrid)
            || !self.key_value.has_same_snapshot_layout(&saved.key_value)
            || !saved.hybrid.supports_isolated_snapshot()
            || !saved.key_value.supports_isolated_snapshot()
        {
            return Err(JointStateError::Identity);
        }
        Ok(())
    }

    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
        self.validate_isolated_restore(&saved)?;
        // Every fallible check precedes installation. The live owner's identity,
        // stream and shared cumulative accounting survive this replacement.
        self.hybrid = saved.hybrid;
        self.key_value = saved.key_value;
        self.generation += 1;
        Ok(())
    }
}

struct JointRetention {
    _hybrid: MlxHybridState,
    _key_value: MlxKeyValueState,
    _permit: CopyPermit,
}

fn arrays<'a>(hybrid: &'a MlxHybridState, key_value: &'a MlxKeyValueState) -> Vec<&'a Array> {
    hybrid
        .retained_arrays()
        .into_iter()
        .chain(key_value.retained_arrays())
        .collect()
}

fn paging_identity(hybrid: &MlxHybridState, key_value: &MlxKeyValueState) -> Vec<Option<u64>> {
    std::iter::once(
        hybrid
            .residency_manager()
            .map(CacheResidencyManager::session_id),
    )
    .chain(key_value.as_ref().iter().map(|layer| match layer {
        MlxKeyValueLayerState::Paged(cache) => Some(cache.manager().session_id()),
        _ => None,
    }))
    .collect()
}

impl MlxJointState {
    fn copy_owner(
        &self,
        retention: &SnapshotReservation,
        fork_retention: Option<SnapshotReservation>,
    ) -> Result<Self, JointStateError> {
        let MlxJointStateBranch {
            hybrid,
            key_value,
            permit,
            generation,
            ..
        } = self.copy_candidate(Some(retention))?;
        let copied = Self {
            hybrid,
            key_value,
            stream: self.stream.clone(),
            admission: Arc::new(Admission {
                active: AtomicBool::new(false),
                accounting: Arc::clone(&self.admission.accounting),
                _fork_retention: fork_retention,
            }),
            generation,
        };
        // The copy is complete. Keeping the saved state must not retain the
        // source owner's exclusive transition admission.
        drop(permit);
        Ok(copied)
    }
}

impl IsolatedStateFork for MlxJointState {
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_owner(retention, Some(retention.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_core::{AttentionPolicy, LayerSchedule};
    use eredu_nn::AttentionCache;
    use eredu_runtime::execution_control::SnapshotBudget;
    use safemlx::{Device, DeviceType};

    fn canonical() -> MlxJointState {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let hybrid_layout =
            StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap())
                .unwrap();
        let key_value_layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let hybrid = MlxHybridState::device(hybrid_layout, &[]).unwrap();
        let mut key_value = MlxKeyValueState::device(key_value_layout).unwrap();
        AttentionCache::update_for_attention(
            &mut key_value.as_mut()[0],
            MlxTensor::from_array(Array::from_slice(&[1_f32, 2.], &[2, 1, 1, 1])),
            MlxTensor::from_array(Array::from_slice(&[-1_f32, -2.], &[2, 1, 1, 1])),
            &stream,
        )
        .unwrap();
        MlxJointState::new(
            hybrid,
            key_value,
            &stream,
            JointCopyLimits {
                max_branch_bytes: 1 << 20,
                max_total_copy_bytes: 1 << 20,
            },
        )
        .unwrap()
    }

    #[test]
    fn joint_retention_keeps_branch_admission_until_completion_owner_releases_it() {
        let canonical = canonical();
        let cost = canonical.copy_estimate().unwrap().copy_bytes;
        let branch = canonical.branch().unwrap();
        let completion_retention = branch.retention();
        drop(branch);

        // A submission can outlive its candidate handle after failure. Its
        // retained lifetime authority still prevents a second cache branch.
        assert!(matches!(canonical.branch(), Err(JointStateError::Busy)));
        assert_eq!(canonical.copy_bytes_consumed(), cost);
        drop(completion_retention);
        assert_eq!(canonical.copy_bytes_consumed(), cost);

        let next = canonical.branch().unwrap();
        assert_eq!(canonical.copy_bytes_consumed(), 2 * cost);
        drop(next);
        assert_eq!(canonical.copy_bytes_consumed(), 2 * cost);
    }

    #[test]
    fn fork_budget_remains_owned_by_native_completion_after_child_and_branch_drop() {
        let parent = canonical();
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 1 << 20,
            cumulative_copy_bytes: 1 << 20,
        });
        let estimate = parent.estimate_isolated_copy().unwrap();
        let lease = budget
            .reserve(SnapshotResourceKind::Branch, Some(estimate))
            .unwrap();
        let child = parent.fork_isolated(&lease).unwrap();
        drop(lease);
        let branch = child.branch().unwrap();
        let completion_retention = branch.retention();
        drop(child);
        drop(branch);
        assert_eq!(budget.usage().branches, 1);
        assert_eq!(budget.usage().retained_bytes, estimate.retained_bytes);
        // The parent shares cumulative native copying, not its child's lease.
        assert_eq!(parent.copy_bytes_consumed(), 2 * estimate.copy_bytes);
        let source_branch = parent.branch().unwrap();
        drop(source_branch);
        drop(completion_retention);
        assert_eq!(budget.usage().branches, 0);
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(budget.usage().cumulative_copy_bytes, estimate.copy_bytes);
    }
}
