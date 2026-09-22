//! Receipt work bound to the live capture owner, epoch and nonrefundable ledger.
use super::*;
use eredu_core::{Completion, DistributedCommitEpoch, checkpoint::TensorDtype};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

mod coordination;
mod funded_allowance;
mod funded_coordination;
mod funded_interventions;
pub use funded_interventions::{
    PartitionInterventionInvocationSource, PartitionInterventionLocalAllowance,
    PartitionInterventionMemberSource, PartitionInterventionSourceError,
    PreparedPartitionInterventionSource,
};
pub(crate) use funded_interventions::{
    PartitionInterventionOutcome, PreparedPartitionIntervention,
};
mod funded_fragments;
pub use coordination::SessionPartitionCoordination;
pub(crate) use funded_allowance::PreparedPartitionRemoteCharge;
pub use funded_allowance::{PartitionCaptureAllowanceError, PreparedPartitionCaptureAllowance};
pub use funded_coordination::{
    PartitionCaptureCoordinationError, PreparedPartitionCaptureCoordination,
};
pub use funded_fragments::{
    PartitionCaptureFragmentAllowanceError, PartitionCaptureFragmentGeometry,
    PartitionCaptureFragmentSource, PartitionCaptureRoutedFragmentGeometry,
    PartitionCaptureRoutedFragmentSource, PreparedPartitionFragmentAllowance,
    PreparedPartitionFragmentLoan,
};
pub(crate) use funded_fragments::{PreparedPartitionFragmentSourceAllowance, SourceBindingError};
mod hook;
pub use hook::{PartitionCaptureHookTransport, SessionPartitionHook};
mod intervention;
mod intervention_evidence;
pub use intervention::SessionPartitionIntervention;
mod source;
pub use source::SessionPartitionSource;
pub(super) use source::routed_input_rows;
#[cfg(test)]
mod identity_tests;
mod routed;

/// Retained common identities supplied by loaded-session composition. These
/// labels describe an already coordinated run; constructing them does not prove
/// that arbitrary caller labels belong to a loaded model or to peer sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionCaptureIdentity {
    artifact: String,
    execution: String,
    run: String,
    setup: Option<crate::CommunicationSessionIdentity>,
    first_epoch: Option<DistributedCommitEpoch>,
    overlay: Option<String>,
    world_size: usize,
}

// The same canonical writer serves ordinary and paid run identities.
pub(in crate::capture::partition) struct SessionRunName(
    pub(in crate::capture::partition) crate::CommunicationSessionIdentity,
    pub(in crate::capture::partition) DistributedCommitEpoch,
);
impl std::fmt::Display for SessionRunName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "capture:{}:{}", self.0, self.1.value())
    }
}

impl PartitionCaptureIdentity {
    /// Validates bounded identity storage and an exact nonempty participant world.
    pub fn new(
        artifact: String,
        execution: String,
        run: String,
        overlay: Option<String>,
        world_size: usize,
    ) -> Result<Self, CaptureError> {
        if world_size == 0 || u32::try_from(world_size).is_err() {
            return Err(invalid("invalid partition capture world"));
        }
        for identity in [&artifact, &execution, &run]
            .into_iter()
            .chain(overlay.iter())
        {
            if identity.is_empty() || identity.len() > 256 {
                return Err(invalid(
                    "partition capture identity must contain 1..=256 bytes",
                ));
            }
        }
        Ok(Self {
            artifact,
            execution,
            run,
            setup: None,
            first_epoch: None,
            overlay,
            world_size,
        })
    }

    /// Binds loaded capture to an agreed setup and its first actual forward epoch.
    ///
    /// No collective or native work occurs here. The shared observer preparation
    /// claims the first epoch once, including failed capture admissions. Later
    /// steps and restores retain it; a new run receives a later session epoch.
    /// Native composition must supply this setup's actual model and overlay facts.
    pub fn for_session(
        artifact: String,
        execution: String,
        setup: crate::CommunicationSessionIdentity,
        overlay: Option<String>,
    ) -> Result<Self, CaptureError> {
        let mut identity = Self::new(
            artifact,
            execution,
            setup.to_string(),
            overlay,
            setup.participant_count(),
        )?;
        identity.setup = Some(setup);
        Ok(identity)
    }

    fn bind_epoch(&mut self, epoch: DistributedCommitEpoch) {
        if self.first_epoch.is_none() {
            if let Some(setup) = self.setup {
                self.run = SessionRunName(setup, epoch).to_string();
                self.first_epoch = Some(epoch);
            }
        }
    }

    pub(in crate::capture) fn child_identity(&self) -> Option<Self> {
        let setup = self.setup?;
        let mut child = self.clone();
        child.run = setup.to_string();
        child.first_epoch = None;
        Some(child)
    }

    pub(in crate::capture) fn heap_bytes(&self) -> Option<u64> {
        [&self.artifact, &self.execution, &self.run]
            .into_iter()
            .chain(self.overlay.iter())
            .try_fold(0u64, |sum, value| {
                sum.checked_add(u64::try_from(value.len()).ok()?)
            })
    }

    fn context(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        epoch: DistributedCommitEpoch,
        invocation: Option<CaptureInvocationShape>,
    ) -> PartitionCaptureContext {
        PartitionCaptureContext {
            invocation,
            artifact_identity: self.artifact.clone(),
            execution_identity: self.execution.clone(),
            run_identity: self.run.clone(),
            overlay_identity: self.overlay.clone(),
            capture_plan_identity: plan.identity().into(),
            selection_index: index,
            phase,
            prediction,
            forward_epoch: epoch.value(),
            invocation_window: None,
        }
    }
}

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

/// Side-effect-free native bounds for one producer fragment. The creation bound
/// covers a deferred input factory in addition to the ordinary capture estimate;
/// unused credits are not refunded when this point uses an existing tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionCaptureNativeEstimate {
    /// Same conservative source/transform/host/record bound as `CaptureBackend::estimate`.
    pub capture: CaptureUsage,
    /// Maximum separately declared native work for a deferred source factory.
    pub generated_creation_bytes: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PartitionCaptureKey {
    Observation(usize),
    InterventionEvidence { operation: usize, evidence: usize },
}
impl PartitionCaptureKey {
    fn hash(self, digest: &mut Sha256) {
        match self {
            Self::Observation(index) => {
                digest.update([0]);
                digest.update((index as u64).to_le_bytes());
            }
            Self::InterventionEvidence {
                operation,
                evidence,
            } => {
                digest.update([1]);
                digest.update((operation as u64).to_le_bytes());
                digest.update((evidence as u64).to_le_bytes());
            }
        }
    }
}

pub(in crate::capture) struct PartitionCaptureRun {
    identity: PartitionCaptureIdentity,
    claimed: BTreeMap<PartitionCaptureKey, [u8; 32]>,
    intervention_claimed: BTreeMap<usize, [u8; 32]>,
    intervention_completed: BTreeMap<usize, (DistributedCommitEpoch, CaptureUsage)>,
    skipped: BTreeMap<usize, (DistributedCommitEpoch, CaptureSkipReason, CaptureUsage)>,
    coordination_reserved: bool,
    coordination_prepared: bool,
    coordinated: bool,
    staged: Vec<(
        DistributedCommitEpoch,
        PartitionCaptureKey,
        CaptureRecord,
        PartitionCaptureEvidence,
    )>,
    published: Vec<PartitionCaptureEvidence>,
}

impl PartitionCaptureRun {
    pub(in crate::capture) fn child_identity(&self) -> Option<PartitionCaptureIdentity> {
        self.identity.child_identity()
    }
    pub(in crate::capture) fn identity_heap_bytes(&self) -> Option<u64> {
        self.identity.heap_bytes()
    }
    pub(in crate::capture) fn matches_artifact(&self, artifact: &str) -> bool {
        self.identity.artifact == artifact
    }
    pub(in crate::capture) fn delivery_complete(&self, epoch: DistributedCommitEpoch) -> bool {
        self.coordinated
            && self.intervention_claimed.len() == self.intervention_completed.len()
            && self.claimed.len()
                == self
                    .staged
                    .iter()
                    .filter(|(active, ..)| *active == epoch)
                    .count()
    }
    pub(crate) fn intervention_completed(
        &self,
        index: usize,
        epoch: DistributedCommitEpoch,
    ) -> bool {
        self.intervention_completed
            .get(&index)
            .is_some_and(|(active, _)| *active == epoch)
    }
    pub(in crate::capture) fn world_size(&self) -> usize {
        self.identity.world_size
    }
    pub(in crate::capture) fn begin_step(&mut self) {
        self.claimed.clear();
        self.intervention_claimed.clear();
        self.intervention_completed.clear();
        self.skipped.clear();
        self.coordination_reserved = false;
        self.coordination_prepared = false;
        self.coordinated = false;
        self.staged.clear();
        self.published.clear();
    }
    pub(in crate::capture) fn finish(
        &mut self,
        epoch: DistributedCommitEpoch,
        committed: bool,
        records: Option<&mut Vec<CaptureRecord>>,
        mut interventions: Option<&mut Vec<eredu_core::intervention::InterventionRecord>>,
    ) {
        if committed {
            if let Some(records) = interventions.as_mut() {
                for (index, (completed_epoch, charged)) in &self.intervention_completed {
                    if *completed_epoch != epoch {
                        continue;
                    }
                    records[*index].outcome = if records[*index]
                        .routed_units
                        .is_some_and(|receipt| receipt.affected_values == 0)
                    {
                        eredu_core::intervention::InterventionOutcome::Unmatched
                    } else {
                        eredu_core::intervention::InterventionOutcome::Applied
                    };
                    // Charge was checked during preparation; no new reservation at commit.
                    records[*index].charged = *charged;
                }
            }
            if let Some(records) = records {
                for (index, (skipped_epoch, reason, charged)) in std::mem::take(&mut self.skipped) {
                    if skipped_epoch != epoch {
                        continue;
                    }
                    records[index].outcome = CaptureOutcome::Skipped { reason };
                    records[index].charged = charged;
                }
                for (staged_epoch, index, record, evidence) in self.staged.drain(..) {
                    if staged_epoch == epoch {
                        match index {
                            PartitionCaptureKey::Observation(index) => records[index] = record,
                            PartitionCaptureKey::InterventionEvidence {
                                operation,
                                evidence,
                            } => {
                                if let Some(interventions) = interventions.as_mut() {
                                    interventions[operation].evidence[evidence] = record;
                                }
                            }
                        }
                        // Capacity was reserved while delivery could still fail.
                        self.published.push(evidence);
                    }
                }
            }
        } else {
            self.skipped.clear();
            self.staged
                .retain(|(staged_epoch, ..)| *staged_epoch != epoch);
        }
    }
    pub(in crate::capture) fn take_evidence(&mut self) -> Vec<PartitionCaptureEvidence> {
        std::mem::take(&mut self.published)
    }
}

/// Move-only authority for one selection in one live capture transaction. It
/// cannot be replayed after restore, transferred to another run, or supplied a
/// replacement ledger. Every producer and receiver cost was prepaid globally;
/// this rank can spend only its local allowance.
pub struct SessionPartitionCapture<'a, T: PartitionCaptureTransport> {
    epoch: DistributedCommitEpoch,
    pub(super) index: usize,
    key: PartitionCaptureKey,
    plan: SharedCapturePlan,
    rank: usize,
    exchange: PartitionCaptureExchange<'a, T>,
    quota: CaptureQuota,
    global_reserved: CaptureUsage,
    observed: bool,
    source_dtype: Option<TensorDtype>,
    fragments: Option<Vec<CapturedPartitionFragment>>,
    hook_accepted: Option<bool>,
    source_accepted: Option<bool>,
    routed: Option<routed::RoutedInvocation>,
    // Last: all retained plan/work payloads retire before preparation custody.
    owner: Arc<crate::capture::CaptureHostOwner>,
}

impl<T: PartitionCaptureTransport> SessionPartitionCapture<'_, T> {
    /// Existing immutable shared source retained by this exact partition work.
    /// No source payload copy or fresh source registration is manufactured.
    pub fn shared_plan_source(&self) -> &SharedCapturePlan {
        &self.plan
    }
    /// Full nonrefundable global charge for preparation and working credits.
    pub const fn global_reserved(&self) -> CaptureUsage {
        self.global_reserved
    }
    /// This rank's prepaid allowance, including transport and receiver work.
    pub fn local_reserved(&self) -> CaptureUsage {
        self.quota.limit()
    }
    /// Credits spent locally so far. Unused credits remain charged globally.
    pub fn local_used(&self) -> CaptureUsage {
        self.quota.used()
    }
}

impl CaptureSession {
    /// Checks this run against the actual loaded setup without replacing its
    /// first forward epoch. Fresh runs may be bound; an already-bound parent or
    /// re-admitted child must match every retained model/overlay fact.
    pub fn ensure_partition_capture(
        &mut self,
        identity: PartitionCaptureIdentity,
    ) -> Result<(), CaptureError> {
        if identity.setup.is_none() || identity.first_epoch.is_some() {
            return Err(invalid(
                "loaded capture binding requires a fresh setup scope",
            ));
        }
        if let Some(run) = &self.partition {
            let current = &run.identity;
            if current.setup != identity.setup
                || current.artifact != identity.artifact
                || current.execution != identity.execution
                || current.overlay != identity.overlay
                || current.world_size != identity.world_size
            {
                return Err(invalid(
                    "capture run belongs to another loaded setup or parameter version",
                ));
            }
            return Ok(());
        }
        self.configure_partition_capture(identity)
    }

    /// Binds a newly created capture owner to retained, coordinated model/run
    /// identity. It cannot be replaced after binding or after a step starts.
    /// Snapshot forks need fresh child composition; restore keeps this binding.
    pub fn configure_partition_capture(
        &mut self,
        identity: PartitionCaptureIdentity,
    ) -> Result<(), CaptureError> {
        if self.partition.is_some()
            || self.has_step
            || self.records.is_some()
            || self.transaction.is_some()
        {
            return Err(invalid(
                "partition capture identity requires an unstarted run",
            ));
        }
        self.partition = Some(PartitionCaptureRun {
            identity,
            claimed: BTreeMap::new(),
            intervention_claimed: BTreeMap::new(),
            intervention_completed: BTreeMap::new(),
            skipped: BTreeMap::new(),
            coordination_reserved: false,
            coordination_prepared: false,
            coordinated: false,
            staged: Vec::new(),
            published: Vec::new(),
        });
        Ok(())
    }

    pub(in crate::capture) fn bind_partition_run_epoch(&mut self, epoch: DistributedCommitEpoch) {
        if let Some(run) = &mut self.partition {
            run.identity.bind_epoch(epoch);
        }
    }

    fn partition_epoch(&self) -> Result<DistributedCommitEpoch, CaptureError> {
        match self.transaction {
            Some((epoch, CaptureTransactionStatus::Pending)) if self.records.is_some() => Ok(epoch),
            _ => Err(invalid(
                "partition capture requires a pending live transaction",
            )),
        }
    }

    /// Cancels only locally prepared work, before any collective or source hook.
    /// Already spent preparation credits remain consumed. The outcome is sealed
    /// into common coordination and becomes visible only after final commit.
    pub(super) fn skip_partition_selection(
        &mut self,
        index: usize,
        budget: CaptureBudget,
        cumulative: bool,
        before: CaptureUsage,
    ) -> Result<(), CaptureError> {
        let epoch = self.partition_epoch()?;
        let run = self
            .partition
            .as_mut()
            .ok_or_else(|| invalid("unbound partition"))?;
        if !run.coordination_reserved
            || run.coordination_prepared
            || run.skipped.contains_key(&index)
            || self.plan.plan().limits.on_limit != CaptureLimitPolicy::Skip
        {
            return Err(invalid("partition skip requires fresh prepaid preparation"));
        }
        let record = self
            .records
            .as_ref()
            .and_then(|records| records.get(index))
            .ok_or_else(|| invalid("partition skip selection is absent"))?;
        if record.outcome != CaptureOutcome::Missing {
            return Err(invalid("partition skip selection is not scheduled"));
        }
        let after = self.ledger.step();
        let difference = |after: u64, before: u64| {
            after
                .checked_sub(before)
                .ok_or_else(|| invalid("partition preparation accounting moved backwards"))
        };
        let charged = record.charged.checked_add(CaptureUsage {
            captures: difference(after.captures, before.captures)?,
            retained_bytes: difference(after.retained_bytes, before.retained_bytes)?,
            host_bytes: difference(after.host_bytes, before.host_bytes)?,
            encoded_bytes: difference(after.encoded_bytes, before.encoded_bytes)?,
        })?;
        run.claimed.remove(&PartitionCaptureKey::Observation(index));
        run.skipped.insert(
            index,
            (
                epoch,
                CaptureSkipReason::Limit { budget, cumulative },
                charged,
            ),
        );
        Ok(())
    }

    fn validate_partition_work<T: PartitionCaptureTransport>(
        &self,
        work: &SessionPartitionCapture<'_, T>,
    ) -> Result<(), CaptureError> {
        if !Arc::ptr_eq(&self.owner, &work.owner)
            || self.partition_epoch()? != work.epoch
            || !self
                .partition
                .as_ref()
                .is_some_and(|run| run.coordinated && run.claimed.contains_key(&work.key))
        {
            return Err(invalid(
                "partition capture authority belongs to another owner or forward",
            ));
        }
        Ok(())
    }

    /// Prices every expected producer plus every receiver before native work,
    /// then grants this rank a sealed local allowance from the live run ledger.
    /// Admission is local only. The enclosing observer must agree all admissions
    /// and common receipt identities before allowing producer hooks to execute.
    pub fn prepare_partition_capture<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
        index: usize,
        producers: Vec<PartitionCaptureProducer>,
        limits: PartitionCaptureReceiptLimits,
        estimate: impl FnMut(
            &[u64],
            &CaptureSelection,
            &ResolvedCaptureSlice,
        ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    ) -> Result<SessionPartitionCapture<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.prepare_partition_capture_combined(
            transport,
            index,
            producers,
            PartitionCaptureCombination::Disjoint,
            limits,
            estimate,
        )
    }

    /// Prices a retained assembly equation through the same live, move-only
    /// capture authority, including raw summands and complete receiver storage.
    pub fn prepare_partition_capture_combined<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
        index: usize,
        producers: Vec<PartitionCaptureProducer>,
        combination: PartitionCaptureCombination,
        limits: PartitionCaptureReceiptLimits,
        estimate: impl FnMut(
            &[u64],
            &CaptureSelection,
            &ResolvedCaptureSlice,
        ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    ) -> Result<SessionPartitionCapture<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.prepare_partition_selection(
            transport,
            self.plan.clone(),
            PartitionCaptureKey::Observation(index),
            index,
            producers,
            combination,
            limits,
            estimate,
        )
    }

    fn prepare_partition_selection<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
        plan: SharedCapturePlan,
        key: PartitionCaptureKey,
        index: usize,
        producers: Vec<PartitionCaptureProducer>,
        combination: PartitionCaptureCombination,
        limits: PartitionCaptureReceiptLimits,
        mut estimate: impl FnMut(
            &[u64],
            &CaptureSelection,
            &ResolvedCaptureSlice,
        ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    ) -> Result<SessionPartitionCapture<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let native_selection = super::sum::reserve_native_selection(
            &plan,
            index,
            combination,
            transport.participant_count(),
            &mut self.ledger,
        )?;
        self.prepare_partition_work(
            transport,
            plan.clone(),
            key,
            index,
            match combination {
                PartitionCaptureCombination::Disjoint => routed::Producers::Dense(producers),
                PartitionCaptureCombination::SumF64ToF32 => routed::Producers::Sum(producers),
            },
            limits,
            |receipt, rank, fragment| {
                let projection = receipt.producer(rank).expect("retained producer");
                estimate(
                    projection.local_shape(),
                    &native_selection,
                    projection.fragments()[fragment].local(),
                )
            },
        )
    }

    fn prepare_partition_work<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
        plan: SharedCapturePlan,
        key: PartitionCaptureKey,
        index: usize,
        producers: routed::Producers,
        limits: PartitionCaptureReceiptLimits,
        mut estimate: impl FnMut(
            &PartitionCaptureReceiptPlan,
            usize,
            usize,
        ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    ) -> Result<SessionPartitionCapture<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let epoch = self.partition_epoch()?;
        let run = self
            .partition
            .as_ref()
            .ok_or_else(|| invalid("partition capture identity is not bound"))?;
        let world_size = run.identity.world_size;
        let rank = transport.capture_rank();
        if transport.participant_count() != world_size
            || rank >= world_size
            || run.coordination_prepared
            || run.claimed.contains_key(&key)
            || matches!(key, PartitionCaptureKey::Observation(index) if run.skipped.contains_key(&index))
        {
            return Err(
                invalid("partition capture topology or selection authority disagrees").into(),
            );
        }
        if producers.len() == 0 || producers.len() > limits.max_producers {
            return Err(invalid("partition capture producer count exceeds its bound").into());
        }
        let preparation = producers.preparation_usage()?.checked_add(CaptureUsage {
            host_bytes: 2048,
            ..Default::default()
        })?;
        let global_preparation = preparation.checked_mul(world_size as u64)?;
        let mut preparation_quota = self
            .ledger
            .reserve_quota(global_preparation)?
            .reserve_quota(preparation)
            .map_err(prepaid_bound_error)?;
        let context = run.identity.context(
            &plan,
            index,
            self.phase,
            self.prediction,
            epoch,
            self.invocation,
        );
        let mut receipt = producers
            .admit(
                plan.clone(),
                context,
                world_size,
                limits,
                &mut preparation_quota,
            )
            .map_err(|error| match error {
                PartitionCaptureMergeError::Capture(source)
                    if matches!(source.cause(), CaptureError::Limit { .. }) =>
                {
                    prepaid_bound_error(source)
                }
                other => other.into(),
            })?;
        let point = &plan.points()[index];
        let mut sparse_costs = Vec::new();
        let costs = receipt_work_costs(transport, &mut receipt, &mut estimate, |usage| {
            sparse_costs.push(usage);
            Ok(())
        })?;
        let global = costs.global;
        let local = costs.local;
        let mut quota = self
            .ledger
            .reserve_quota(global)?
            .reserve_quota(local)
            .map_err(prepaid_bound_error)?;
        let exchange =
            PartitionCaptureExchange::admit(transport, receipt, &mut quota).map_err(|error| {
                match error {
                    PartitionCaptureExchangeError::Capture(error) => prepaid_bound_error(error),
                    other => other,
                }
            })?;
        self.partition
            .as_mut()
            .expect("checked binding")
            .claimed
            .insert(key, costs.descriptor);
        let sparse = matches!(
            point.value_type,
            eredu_core::ObservationValueType::RoutedUnits { .. }
        );
        let mut work = SessionPartitionCapture {
            owner: Arc::clone(&self.owner),
            epoch,
            index,
            key,
            plan,
            rank,
            exchange,
            quota,
            global_reserved: global_preparation.checked_add(global)?,
            observed: false,
            source_dtype: None,
            fragments: None,
            hook_accepted: None,
            source_accepted: None,
            routed: sparse.then_some(routed::RoutedInvocation::Pending),
        };
        if sparse {
            routed::prepare_fragments(&mut work, self.phase, self.prediction, sparse_costs)?;
        }
        Ok(work)
    }

    /// Transforms this rank's one actual source tensor through prepaid fragment
    /// reservations. Nonproducers perform no tensor inspection or native work.
    /// The enclosing TP hook must agree a local failure before a row collective.
    pub fn observe_partition<B: CaptureBackend, T: PartitionCaptureTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        backend: &mut B,
        tensor: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)?;
        if work.observed || work.routed.is_some() {
            return Err(invalid("partition source was already observed").into());
        }
        work.observed = true;
        if work.source_accepted == Some(false) {
            return Err(invalid("partition dependencies have not completed").into());
        }
        let Some(projection) = work.exchange.receipt_plan().producer(work.rank) else {
            return Ok(());
        };
        if backend
            .shape(tensor)
            .map_err(CaptureExecutionError::Backend)?
            != projection.local_shape()
        {
            return Err(invalid("partition source shape differs from its live authority").into());
        }
        work.source_dtype = backend.source_dtype(tensor);
        let mut fragments = Vec::with_capacity(projection.fragments().len());
        for fragment_index in 0..projection.fragments().len() {
            fragments.push(capture_fragment_combined(
                backend,
                tensor,
                PartitionCaptureRequest {
                    invocation: self.invocation,
                    plan: &work.plan,
                    selection_index: work.index,
                    phase: self.phase,
                    prediction: self.prediction,
                    projection,
                    fragment_index,
                    producer_rank: work.rank,
                },
                &mut work.quota,
                work.exchange.receipt_plan().combination(),
            )?);
        }
        work.fragments = Some(fragments);
        Ok(())
    }

    /// Captures a deferred source through the same reserve-before-factory path.
    /// Actual fragment precision is retained; a nonproducer never calls the factory.
    pub fn observe_generated_partition<B: CaptureBackend, T: PartitionCaptureTransport, E>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        backend: &mut B,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
        map_error: &impl Fn(CaptureExecutionError<B::Error>) -> E,
    ) -> Result<(), E>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)
            .map_err(|error| map_error(error.into()))?;
        if work.observed || work.routed.is_some() {
            return Err(map_error(
                invalid("partition source was already observed").into(),
            ));
        }
        work.observed = true;
        if work.source_accepted == Some(false) {
            return Err(map_error(
                invalid("partition dependencies have not completed").into(),
            ));
        }
        let Some(projection) = work.exchange.receipt_plan().producer(work.rank) else {
            return Ok(());
        };
        if backend
            .shape(prototype)
            .map_err(|error| map_error(CaptureExecutionError::Backend(error)))?
            != projection.local_shape()
        {
            return Err(map_error(
                invalid("partition source shape differs from its live authority").into(),
            ));
        }
        let mut fragments = Vec::with_capacity(projection.fragments().len());
        for fragment_index in 0..projection.fragments().len() {
            fragments.push(capture_generated_fragment_combined(
                backend,
                prototype,
                source,
                generate,
                PartitionCaptureRequest {
                    invocation: self.invocation,
                    plan: &work.plan,
                    selection_index: work.index,
                    phase: self.phase,
                    prediction: self.prediction,
                    projection,
                    fragment_index,
                    producer_rank: work.rank,
                },
                &mut work.quota,
                map_error,
                work.exchange.receipt_plan().combination(),
            )?);
        }
        work.source_dtype = match fragments.first() {
            Some(fragment) => fragment.record().source_dtype.clone(),
            None => source.source_dtype.clone(),
        };
        work.fragments = Some(fragments);
        Ok(())
    }

    /// Common all-rank delivery boundary after exact forward completion. This
    /// consumes the authority once and stages global records in their live owner.
    /// Nothing is published until its shared transaction receives final commit.
    pub fn complete_partition_capture<T: PartitionCaptureTransport>(
        &mut self,
        mut work: SessionPartitionCapture<'_, T>,
    ) -> Result<(), PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(&work)?;
        let receipt = work.exchange.receipt_plan();
        let evidence_cost = evidence_usage(receipt)?;
        let local = if work.hook_accepted == Some(false) || work.source_accepted == Some(false) {
            Err(invalid("partition hook has not agreed successful capture").into())
        } else if receipt.producer(work.rank).is_some() {
            work.fragments
                .take()
                .filter(|_| {
                    work.observed
                        && matches!(work.routed, None | Some(routed::RoutedInvocation::Complete))
                })
                .ok_or_else(|| invalid("partition producer supplied no completed source").into())
                .and_then(|fragments| {
                    receipt
                        .encode_producer(work.rank, work.source_dtype, fragments, &mut work.quota)
                        .map(Some)
                        .map_err(Into::into)
                })
        } else {
            Ok(None)
        };
        let received = work.exchange.exchange(local, &mut work.quota)?;
        if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
            work.quota.reserve(evidence_cost)?
        {
            return Err(CaptureError::Limit { budget, cumulative }.into());
        }
        let (context, producers, capture, receipt_plan_identity) = received.into_parts();
        let combination = capture.combination();
        let (record, contributions) = capture.into_parts();
        let evidence = PartitionCaptureEvidence {
            schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
            combination,
            context,
            receipt_plan_identity,
            producers,
            contributions: contributions
                .into_iter()
                .map(|contribution| PartitionCaptureContributionRecord {
                    producer_rank: contribution.producer_rank(),
                    local: region(contribution.geometry().local()),
                    destination: region(contribution.geometry().destination()),
                    charged: contribution.charged(),
                    routed: contribution.routed().cloned(),
                })
                .collect(),
        };
        let run = self.partition.as_mut().expect("validated live owner");
        run.published.reserve(run.staged.len() + 1);
        run.staged.push((work.epoch, work.key, record, evidence));
        Ok(())
    }
}

fn hash_usage(digest: &mut Sha256, usage: CaptureUsage) {
    for value in [
        usage.captures,
        usage.retained_bytes,
        usage.host_bytes,
        usage.encoded_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
}

fn region(slice: &ResolvedCaptureSlice) -> PartitionCaptureRegion {
    PartitionCaptureRegion {
        starts: slice.starts.clone(),
        ends: slice.ends.clone(),
        strides: slice.strides.clone(),
        shape: slice.shape.clone(),
    }
}

pub(super) fn evidence_usage(
    receipt: &PartitionCaptureReceiptPlan,
) -> Result<CaptureUsage, CaptureError> {
    let mut usage = CaptureUsage {
        host_bytes: 8192,
        encoded_bytes: 8192,
        ..Default::default()
    };
    for (_, projection) in receipt.producers() {
        usage = usage.checked_add(CaptureUsage {
            host_bytes: 32,
            encoded_bytes: 32,
            ..Default::default()
        })?;
        usage = usage.checked_add(
            CaptureUsage {
                host_bytes: add(1024, mul(projection.global_shape().len() as u64, 512)?)?,
                encoded_bytes: add(1024, mul(projection.global_shape().len() as u64, 512)?)?,
                ..Default::default()
            }
            .checked_mul(projection.fragments().len() as u64)?,
        )?;
    }
    let routed = super::routed::provenance_bytes(receipt)?;
    usage.checked_add(CaptureUsage {
        host_bytes: routed,
        encoded_bytes: routed,
        ..Default::default()
    })
}

/// A child allowance has already been charged. Exceeding it indicates an invalid
/// bound and must not become an ordinary application-budget skip.
fn prepaid_bound_error(error: CaptureError) -> PartitionCaptureExchangeError {
    if matches!(error.cause(), CaptureError::Limit { .. }) {
        PartitionCaptureExchangeError::PrepaidBound { source: error }
    } else {
        error.into()
    }
}

struct ReceiptWorkCosts {
    global: CaptureUsage,
    local: CaptureUsage,
    descriptor: [u8; 32],
}
fn receipt_work_costs<T: PartitionCaptureTransport>(
    transport: &T,
    receipt: &mut PartitionCaptureReceiptPlan,
    mut estimate: impl FnMut(
        &PartitionCaptureReceiptPlan,
        usize,
        usize,
    ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    mut sparse: impl FnMut(CaptureUsage) -> Result<(), CaptureError>,
) -> Result<ReceiptWorkCosts, CaptureError>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    let world_size = receipt.world_size();
    let rank = transport.capture_rank();
    let index = receipt.context().selection_index;
    let selection = &receipt.plan.plan().selections[index];
    let point = &receipt.plan.points()[index];
    let mut producers_global = CaptureUsage::default();
    let mut producers_local = CaptureUsage::default();
    let mut producer_descriptor = Sha256::new();
    let mut maximum = 0u64;
    let mut all_bounded = true;
    for (producer, projection) in receipt.producers() {
        let mut usage = CaptureUsage::default();
        let mut record_bytes = receipt.producer_envelope_bytes(producer)?;
        let mut dtype_bytes = 0u64;
        // A nonempty fragment's admitted complete-record bound also covers the
        // repeated source dtype, including a backend's encoded format string.
        // Empty producers have no such source; keep their explicit caller cap.
        all_bounded &= !projection.fragments().is_empty();
        for fragment in 0..projection.fragments().len() {
            let native = estimate(receipt, producer, fragment)?;
            let fragment_usage =
                fragment_metadata_usage(selection, point, projection.global_shape().len())?
                    .checked_add(
                        if receipt.combination == PartitionCaptureCombination::SumF64ToF32 {
                            metadata_reservation(selection, point)?
                        } else {
                            CaptureUsage::default()
                        },
                    )?
                    .checked_add(native.capture)?
                    .checked_add(CaptureUsage {
                        retained_bytes: native.generated_creation_bytes,
                        ..Default::default()
                    })?;
            record_bytes = add(
                record_bytes,
                PartitionCaptureReceiptPlan::fragment_envelope_bytes(fragment)?,
            )?;
            record_bytes = add(record_bytes, fragment_usage.encoded_bytes)?;
            dtype_bytes = dtype_bytes.max(fragment_usage.encoded_bytes);
            usage = usage.checked_add(fragment_usage)?;
            if producer == rank && receipt.routed_producer(rank).is_some() {
                sparse(fragment_usage)?;
            }
        }
        maximum = maximum.max(add(record_bytes, dtype_bytes)?);
        producers_global = producers_global.checked_add(usage)?;
        if producer == rank {
            producers_local = producers_local.checked_add(usage)?;
        }
        producer_descriptor.update((producer as u64).to_le_bytes());
        hash_usage(&mut producer_descriptor, usage);
    }
    if all_bounded {
        receipt.restrict_record_bytes(maximum)?;
    }
    // Transport and decoding use the source-derived record bound, not the
    // complete per-step encoded budget from which all ranks' work is charged.
    let common = PartitionCaptureExchange::<T>::estimate_usage(transport, receipt)?
        .checked_add(receipt.delivery_usage()?)?
        .checked_add(evidence_usage(receipt)?)?;
    let mut global = common
        .checked_mul(world_size as u64)?
        .checked_add(producers_global)?;
    let mut local = common.checked_add(producers_local)?;
    let mut descriptor = Sha256::new();
    descriptor.update(receipt.identity().as_bytes());
    hash_usage(&mut descriptor, common);
    descriptor.update(producer_descriptor.finalize());
    for (producer, _) in receipt.producers() {
        let encoding = receipt.encoding_usage(producer)?;
        global = global.checked_add(encoding)?;
        if producer == rank {
            local = local.checked_add(encoding)?;
        }
        descriptor.update((producer as u64).to_le_bytes());
        hash_usage(&mut descriptor, encoding);
    }
    Ok(ReceiptWorkCosts {
        global,
        local,
        descriptor: descriptor.finalize().into(),
    })
}

pub(crate) use funded_fragments::PreparedPartitionAssemblyCharge;
