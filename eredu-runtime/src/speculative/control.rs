//! Scoped controlled access to the same speculative scheduler used by generation.

use super::*;
use crate::execution_control::{SnapshotBudget, SnapshotReservation, TraceBudget, TraceLimits};
use eredu_core::{HostPreparationAuthority, SpeculativeRequestIdentity, SpeculativeValues};
use eredu_core::{
    execution_control::{
        ExecutionControlError, SnapshotEstimate, SnapshotLimits, SnapshotResourceKind,
        SnapshotUsage,
    },
    generation::{SpeculativeRequestId, SpeculativeRequestStatus},
    speculative::{SpeculativeControlError, SpeculativeControlSnapshot, SpeculativeProposalView},
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
mod views;
use crate::execution_control::PendingSnapshotReservation;
mod tables;
use tables::{ControlTable, PreparedSave, SavedOwner};

mod branch;
pub use branch::*;

// Scheduler Backend errors combine executor and sampler failures of the same
// associated type. Neither provider identity nor an origin tag is available.
// An owner which refuses extraction must return the unchanged error to the next
// owner; both defaults therefore preserve the exact ordinary driver wrapper.
fn driver_failure<E, S>(error: SpeculativeDriverError<E::Error>) -> SpeculativeControlError
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Error = E::Error>,
{
    SpeculativeControlError::driver_with_retained(error, |error| {
        E::take_retained_failure(error).or_else(S::take_retained_failure)
    })
}

/// Explicit resource limits for controlled speculative inspection.
#[derive(Debug, Clone)]
pub struct ControlledSpeculativeOptions {
    /// JSON transport limits shared with observed and ordinary controlled runs.
    pub trace_limits: TraceLimits,
    /// Admitted raw-logit selections, captured independently for draft and target.
    /// Other activation paths require a backend with phase-aware capture support.
    pub capture: Option<eredu_core::capture::AdmittedCapturePlan>,
    /// Internal forward observations and interventions under one cumulative
    /// invocation budget. Physical sequence positions are independent of the
    /// scheduler's prediction coordinate and the sampler's one-row captures.
    pub activations: Option<eredu_core::speculative::AdmittedSpeculativeActivations>,
    /// Snapshot retention and copying limits. `None` disables snapshots.
    pub snapshots: Option<SnapshotLimits>,
}

/// One already charged internal record drained after cancellation or a failed
/// action. Forward completion is separate from speculative token acceptance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlledSpeculativeActivation {
    /// Wire version for the delivery envelope.
    pub schema_version: u32,
    /// Shares the step stream's monotone delivery sequence.
    pub sequence: u64,
    /// Logical run which performed the invocation.
    pub run_id: u64,
    /// Restore epoch which performed the invocation.
    pub epoch: u64,
    /// Original invocation evidence, including aborted forward status.
    pub activation: eredu_core::speculative::SpeculativeActivationCapture,
}

impl Default for ControlledSpeculativeOptions {
    fn default() -> Self {
        Self {
            trace_limits: TraceLimits {
                per_record_bytes: 1 << 20,
                total_bytes: 64 << 20,
            },
            snapshots: None,
            capture: None,
            activations: None,
        }
    }
}

/// One proposal's disposition at an exact verification boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeProposalDisposition {
    /// The target accepted this proposal.
    Accepted,
    /// The target rejected it and selected a replacement.
    Rejected,
    /// The proposal was not reached (rejection, termination or cancellation).
    Discarded,
}

/// Resolved or cancelled block with target decisions separate from tentative tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeVerificationRecord {
    /// Whether acceptance decisions were committed; false means cancellation
    /// discarded the block without applying target acceptance decisions.
    pub resolved: bool,
    /// Original proposed block, including failed and unvisited proposals.
    pub proposals: SpeculativeProposalView,
    /// One disposition per proposal.
    pub dispositions: SpeculativeValues<SpeculativeProposalDisposition>,
    /// Exact newly committed target tokens (accepted prefix plus replacement/bonus).
    pub committed_token_ids: SpeculativeValues<u32>,
    /// Optimistic work resolved with this block, if any.
    pub optimistic: Option<SpeculativeProposalView>,
    /// Number of optimistic proposals reused as the next block.
    pub optimistic_reused: usize,
    /// Number consumed as a target bonus.
    pub optimistic_consumed: usize,
    /// Number discarded after the canonical prefix changed or terminated.
    pub optimistic_discarded: usize,
}

/// One scheduler action. A verification commits a whole block atomically.
/// Proposal coordinates are generated-token positions, never prompt positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlledSpeculativeStep {
    /// Wire version for this host-only record.
    pub schema_version: u32,
    /// Monotone delivery sequence, never rewound.
    pub sequence: u64,
    /// Logical run currently installed in this scope.
    #[serde(default)]
    pub run_id: u64,
    /// Restoration epoch, incremented after each successful restore or exchange.
    pub epoch: u64,
    /// Scheduler status after this action.
    pub status: SpeculativeRequestStatus,
    /// Proposal block when created, extended or promoted from optimistic work.
    /// A promoted block can contain previously reported, reused proposals.
    pub drafted: Option<SpeculativeProposalView>,
    /// Completed verification with explicit acceptance and rejection.
    pub verification: Option<SpeculativeVerificationRecord>,
    /// Newly committed tokens; proposals are excluded.
    pub committed_token_ids: SpeculativeValues<u32>,
    /// Forced canonical token consumed by this action, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forced_token: Option<u32>,
    /// Effective prospective sampling policy after this action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<crate::execution_control::SamplingStateFacts>,
    /// Bounded raw-logit captures, including tentative draft and target rows.
    pub captures: SpeculativeValues<eredu_core::speculative::SpeculativePredictionCapture>,
    /// Internal component evidence from actual forwards in this action. Physical
    /// rows, scheduler coordinates and tentative phases remain distinct. These
    /// records share this step's run/restore identity and transport accounting.
    #[serde(default, skip_serializing_if = "SpeculativeValues::is_empty")]
    pub activations: SpeculativeValues<eredu_core::speculative::SpeculativeActivationCapture>,
    /// Active time to the first committed target token, before its callbacks.
    pub timing: GenerationTiming,
    /// Active host duration of this action, including synchronous publication.
    pub step_seconds: f64,
}

/// Session-local immutable snapshot handle. Retention ends on explicit release
/// or when the scoped session exits; copying handles does not copy native state.
#[derive(Debug, Clone)]
pub struct SpeculativeSnapshotHandle {
    owner: SpeculativeRequestIdentity,
    id: u64,
}
impl SpeculativeSnapshotHandle {
    /// Stable local identifier suitable for an Inspector snapshot list.
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Object-safe Inspector surface while the backend lends its execution resources.
/// No new scheduler action starts between calls. Each `step` performs one existing
/// action; submitted native verification may complete while the worker is paused,
/// with its exact completion still owned by the session. Snapshots
/// require `can_snapshot`; committed blocks are never split into fake token steps.
/// Returning from the controlling closure cancels and safely settles unfinished work.
pub trait ControlledSpeculativeSession {
    /// Replaces future internal edits using authority prepared for this loaded
    /// execution. Capture selections, geometry and allowances remain fixed.
    fn readmit_activation_interventions(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError>;
    /// Current scheduler phase, or `Prefill` before the first step.
    fn status(&self) -> SpeculativeRequestStatus;
    /// Canonical generated-token prefix.
    fn token_ids(&self) -> &[u32];
    /// Active TTFT, unaffected by pauses, snapshots or inspection work.
    fn timing(&self) -> GenerationTiming;
    /// Identity of the active logical run; zero identifies the root.
    fn run_id(&self) -> u64;
    /// Creates an inactive isolated child from a snapshot in this scope.
    fn fork(
        &mut self,
        snapshot: &SpeculativeSnapshotHandle,
    ) -> Result<SpeculativeBranchHandle, SpeculativeControlError>;
    /// Serial copy-based exchange; the handle then retains the previously active run.
    fn exchange(
        &mut self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError>;
    /// Prefix and identity currently retained in an inactive branch slot.
    fn branch_info(
        &self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError>;
    /// Releases an inactive branch slot without refunding copying or observation.
    fn release_branch(
        &mut self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<(), SpeculativeControlError>;
    /// Current native sampling compatibility, or None for unsupported samplers.
    fn sampling_state(&self) -> Option<crate::execution_control::SamplingStateFacts>;
    /// Validates and installs a prospective temperature/reseed at a settled boundary.
    fn override_sampling(
        &mut self,
        request: crate::execution_control::SamplingOverride,
    ) -> Result<crate::execution_control::SamplingStateFacts, SpeculativeControlError>;
    /// Stages one canonical token for the next target decision.
    fn force_next_token(&mut self, token: u32) -> Result<(), SpeculativeControlError>;
    /// Clears a staged choice at a settled boundary.
    fn clear_forced_token(&mut self) -> Result<bool, SpeculativeControlError>;
    /// Replaces future role-specific tensor edits; an empty list removes them.
    fn intervene(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    ) -> Result<(), SpeculativeControlError>;
    /// Current restore epoch.
    fn epoch(&self) -> u64;
    /// Performs one action; `None` means the request is already terminal.
    fn step(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError>;
    /// Drains one remaining host record after cancellation or failure. This is
    /// allowed on failed sessions and uses the same cumulative transport budget
    /// as ordinary steps. It never launches or recovers native work. A rejected
    /// delivery is consumed, without refunding capture or invocation allowances.
    fn take_activation_evidence(
        &mut self,
    ) -> Result<Option<ControlledSpeculativeActivation>, SpeculativeControlError>;
    /// Cancels and settles any retained exact completion before returning.
    fn cancel(&mut self) -> Result<(), SpeculativeControlError>;
    /// Whether complete snapshot mechanisms and costs exist at this boundary.
    /// Resource limits are still enforced when `snapshot` reserves its storage.
    fn can_snapshot(&self) -> bool;
    /// Snapshot capability with an explicit reason for unavailable state or mechanisms.
    fn snapshot_support(&self) -> eredu_core::execution_control::ControlSupport;
    /// Saves canonical model, sampler, random and semantic state under the budget.
    fn snapshot(&mut self) -> Result<SpeculativeSnapshotHandle, SpeculativeControlError>;
    /// Restores an immutable snapshot; sequence and resource usage never rewind.
    fn restore(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<(), SpeculativeControlError>;
    /// Releases retained snapshot resources; cumulative copying is not refunded.
    fn release_snapshot(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<(), SpeculativeControlError>;
    /// Current non-rewindable snapshot accounting.
    fn snapshot_usage(&self) -> SnapshotUsage;
}

struct Saved<E: SpeculativeExecutor, S: SpeculativeSampling, C> {
    state: SpeculativeControlSnapshot<E, S, C>,
    run_id: u64,
    estimate: SnapshotEstimate,
    _reservation: SnapshotReservation,
    // The Rc shell, complete state and logical lease all retire first.
    _host: HostPreparationAuthority,
}

struct Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    scheduler: SpeculativeScheduler<'a, E, S, C, P>,
    lane: Option<PreparedSpeculativeLane<'a, E, S, C, P>>,
    id: Option<SpeculativeRequestId>,
    owner: SpeculativeRequestIdentity,
    snapshots: ControlTable<SavedOwner<E, S, C>>,
    branches: ControlTable<branch::Branch<E, S, C>>,
    next_branch: u64,
    run_id: u64,
    next_snapshot: u64,
    budget: Option<SnapshotBudget>,
    trace: TraceBudget,
    retained_record_funding: bool,
    sequence: u64,
    epoch: u64,
    preparation: Duration,
    timing: GenerationTiming,
    failed: bool,
    vocabulary: usize,
    intervention_discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    activation_discovery: Option<eredu_core::speculative::SpeculativeActivationDiscovery>,
}

impl<'a, E, S, C, P> Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    fn charge_record(&mut self, record: &impl Serialize) -> Result<(), SpeculativeControlError> {
        let _host = if self.retained_record_funding {
            self.scheduler
                .executor
                .driver_host_metadata(
                    TraceBudget::counting_control_bytes(),
                    self.scheduler.context,
                )
                .map_err(|e| {
                    SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
                })?
        } else {
            HostPreparationAuthority::unmanaged()
        };
        // Only the concrete closed Step/Activation records call this helper;
        // no caller-defined serializer is admitted by the prepared mode.
        self.trace.charge(record).map_err(Into::into)
    }

    fn request(&self) -> Option<&eredu_core::SpeculativeRequest<'a, E, S, C, P>> {
        self.id.and_then(|id| self.scheduler.requests.request(id))
    }
    fn healthy(&self) -> Result<(), SpeculativeControlError> {
        if self.failed {
            Err(SpeculativeControlError::Failed)
        } else {
            Ok(())
        }
    }

    fn agree_delivery<T>(
        &mut self,
        local: Result<T, SpeculativeControlError>,
    ) -> Result<T, SpeculativeControlError> {
        let stage = eredu_core::run_preparation::TextPreparationStage::Delivery;
        let result = eredu_core::run_preparation::finish_preparation(
            stage,
            local,
            |status| {
                self.scheduler.executor.agree_text_preparation(
                    stage,
                    status,
                    self.scheduler.context,
                )
            },
            SpeculativeControlError::backend,
        );
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn owned(&self, handle: &SpeculativeSnapshotHandle) -> Result<(), SpeculativeControlError> {
        if !self.owner.same(&handle.owner) || !self.snapshots.contains_key(&handle.id) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        Ok(())
    }
    fn prepare_save(
        &self, kind: SnapshotResourceKind, table_bytes: Result<usize,SpeculativeControlError>,
    ) -> Result<PreparedSave, SpeculativeControlError> {
        self.healthy()?;
        let request = self
            .request()
            .ok_or(SpeculativeControlError::NotQuiescent)?;
        if !request.is_control_boundary() {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        let budget = self
            .budget
            .as_ref()
            .ok_or(SpeculativeControlError::Unsupported(
                "snapshot limits were not supplied",
            ))?;
        let estimate = request
            .control_snapshot_estimate(self.scheduler.executor)
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let controls = SavedOwner::<E,S,C>::control_bytes()
            .and_then(|n| n.checked_add(PendingSnapshotReservation::control_bytes()?))
            .ok_or(ExecutionControlError::Overflow)?;
        let overhead = controls.checked_add(table_bytes?).and_then(|n| u64::try_from(n).ok())
            .ok_or(ExecutionControlError::Overflow)?;
        let estimate = SnapshotEstimate {
            retained_bytes: estimate.retained_bytes.checked_add(overhead).ok_or(ExecutionControlError::Overflow)?,
            copy_bytes: estimate.copy_bytes.checked_add(overhead).ok_or(ExecutionControlError::Overflow)?,
        };
        // Logical policy wins before allocating any owner; a later refusal
        // refunds retained bytes while cumulative copy attempts stay consumed.
        let reservation = budget.reserve_pending(kind, Some(estimate))?;
        let host = self
            .scheduler
            .executor
            .driver_host_metadata(Some(controls), self.scheduler.context)
            .map_err(|e| {
                SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
            })?;
        Ok(PreparedSave {
            estimate,
            reservation,
            host,
        })
    }
    fn finish_save(
        &self,
        prepared: PreparedSave,
    ) -> Result<SavedOwner<E, S, C>, SpeculativeControlError> {
        let PreparedSave {
            estimate,
            reservation,
            host,
        } = prepared;
        let reservation = reservation.publish_with_host(host.clone());
        let state = self.request().ok_or(SpeculativeControlError::NotQuiescent)?
            .control_snapshot(self.scheduler.executor, self.scheduler.context)?;
        Ok(SavedOwner::new(Saved {
            state, run_id: self.run_id, estimate, _reservation: reservation, _host: host,
        }))
    }
    fn save_state(
        &self,
        kind: SnapshotResourceKind,
    ) -> Result<SavedOwner<E, S, C>, SpeculativeControlError> {
        self.finish_save(self.prepare_save(kind, Ok(0))?)
    }
    fn reserve_control(
        &self,
        kind: SnapshotResourceKind,
        estimate: SnapshotEstimate,
    ) -> Result<SnapshotReservation, SpeculativeControlError> {
        let pending = self
            .budget
            .as_ref()
            .expect("saved state budget")
            .reserve_pending(kind, Some(estimate))?;
        let host = self
            .scheduler
            .executor
            .driver_host_metadata(
                PendingSnapshotReservation::control_bytes(),
                self.scheduler.context,
            )
            .map_err(|e| {
                SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
            })?;
        Ok(pending.publish_with_host(host))
    }
    fn step_inner(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError> {
        self.healthy()?;
        if self.id.is_some() && self.scheduler.is_finished() {
            return Ok(None);
        }
        let forced = self
            .request()
            .and_then(|r| r.sampler().control_pending_forced());
        let before = self
            .request()
            .map(|r| {
                Ok::<_, SpeculativeControlError>((
                    r.sequence().tokens().len(),
                    r.proposals_with_metadata(self.scheduler.executor, self.scheduler.context)
                        .map_err(driver_failure::<E, S>)?,
                    r.optimistic_proposals_with_metadata(
                        self.scheduler.executor,
                        self.scheduler.context,
                    )
                    .map_err(driver_failure::<E, S>)?,
                    r.stats().counters(),
                ))
            })
            .transpose();
        let before = self.agree_delivery(before)?;
        let started = std::time::Instant::now();
        let was_prefill = self.lane.is_some();
        if let Some(lane) = self.lane.take() {
            self.id = Some(
                self.scheduler
                    .submit(lane)
                    .map_err(driver_failure::<E, S>)?,
            );
            self.timing = GenerationTiming::new(
                self.request()
                    .and_then(|r| r.stats().submission_to_first_token())
                    .map(|t| self.preparation + t),
            );
        } else {
            self.scheduler.step().map_err(driver_failure::<E, S>)?;
        }
        let elapsed = started.elapsed();
        let local = (|| {
            let request = self.request().expect("submitted request");
            let old_len = before.as_ref().map_or(0, |b| b.0);
            let committed = views::collect(
                request.sequence().tokens()[old_len..].iter().copied(),
                self.scheduler.executor,
                self.scheduler.context,
                std::mem::size_of::<ControlledSpeculativeStep>()
                    + std::mem::size_of::<
                        Result<Option<ControlledSpeculativeStep>, SpeculativeControlError>,
                    >(),
            )?;
            let after_proposals = request
                .proposals_with_metadata(self.scheduler.executor, self.scheduler.context)
                .map_err(driver_failure::<E, S>)?;
            let after_optimistic = request
                .optimistic_proposals_with_metadata(self.scheduler.executor, self.scheduler.context)
                .map_err(driver_failure::<E, S>)?;
            let drafted = if after_optimistic.as_ref() != before.as_ref().and_then(|b| b.2.as_ref())
                && after_optimistic.is_some()
            {
                after_optimistic
            } else if after_proposals.as_ref() != before.as_ref().and_then(|b| b.1.as_ref())
                && after_proposals.is_some()
            {
                after_proposals
            } else {
                None
            };
            let verification = match before {
                Some((_, Some(proposals), optimistic, stats))
                    if request.stats().rounds() != stats.rounds()
                        || request.status() == SpeculativeRequestStatus::Cancelled =>
                {
                    let accepted = request.stats().accepted_tokens() - stats.accepted_tokens();
                    let dispositions = views::collect(
                        (0..proposals.token_ids.len()).map(|i| {
                            if i < accepted {
                                SpeculativeProposalDisposition::Accepted
                            } else if i == accepted && committed.len() > accepted {
                                SpeculativeProposalDisposition::Rejected
                            } else {
                                SpeculativeProposalDisposition::Discarded
                            }
                        }),
                        self.scheduler.executor,
                        self.scheduler.context,
                        std::mem::size_of::<SpeculativeVerificationRecord>(),
                    )?;
                    Some(SpeculativeVerificationRecord {
                        resolved:request.stats().rounds()>stats.rounds(),proposals,dispositions,
                        committed_token_ids:committed.clone(),optimistic,
                        optimistic_reused:request.stats().reused_optimistic_tokens()-stats.reused_optimistic_tokens(),
                        optimistic_consumed:request.stats().consumed_optimistic_tokens()-stats.consumed_optimistic_tokens(),
                        optimistic_discarded:request.stats().discarded_optimistic_tokens()-stats.discarded_optimistic_tokens(),
                    })
                }
                _=>None,
            };
            let mut record=ControlledSpeculativeStep {
                schema_version:1,sequence:self.sequence,epoch:self.epoch,run_id:self.run_id,status:request.status(),
                drafted:if was_prefill {None}else {drafted},verification,
                forced_token:forced.filter(|_|!committed.is_empty()),sampling:self.sampling_state(),
                committed_token_ids:committed,captures:SpeculativeValues::default(),activations:SpeculativeValues::default(),
                timing:self.timing,step_seconds:elapsed.as_secs_f64(),
            };
            // Preserve the producer's exact buffer and frame custody. The
            // shared freezer prices only a retained buffer's new immutable shell;
            // ordinary buffers remain ordinary without post-hoc adoption.
            let captures = self
                .scheduler
                .requests
                .request_mut(self.id.expect("submitted request"))
                .expect("submitted request")
                .sampler_mut()
                .take_control_captures();
            record.captures =
                views::freeze(captures, self.scheduler.executor, self.scheduler.context)?;
            let mut activations = self
                .scheduler
                .executor
                .driver_buffer(0, self.scheduler.context)
                .map_err(|e| {
                    SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
                })?;
            while let Some(capture) = self.scheduler.executor.take_activation_capture() {
                views::push(
                    &mut activations,
                    capture,
                    self.scheduler.executor,
                    self.scheduler.context,
                )?;
            }
            record.activations=views::freeze(activations,self.scheduler.executor,self.scheduler.context)?;
            self.charge_record(&record)?;
            self.sequence=self.sequence.checked_add(1).ok_or(ExecutionControlError::Overflow)?;
            Ok(Some(record))
        })();
        self.agree_delivery(local)
    }
}

impl<'a, E, S, C, P> ControlledSpeculativeSession for Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    fn readmit_activation_interventions(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        if self.lane.is_none() {
            self.request()
                .ok_or(SpeculativeControlError::NotQuiescent)?
                .validate_control_edit()?;
        }
        // The selected executor always authenticates the candidate. An original
        // observer owns its exact declaration source; ordinary observers require
        // the controller's loaded discovery. Neither path skips validation.
        self.scheduler.executor.validate_activation_readmission(
            &plan, self.activation_discovery.as_ref(),
        )?;
        self.scheduler
            .executor
            .readmit_activation_interventions(plan)
    }
    fn status(&self) -> SpeculativeRequestStatus {
        self.request()
            .map_or(SpeculativeRequestStatus::Prefill, |r| r.status())
    }
    fn token_ids(&self) -> &[u32] {
        self.request().map_or(&[], |r| r.sequence().tokens())
    }
    fn timing(&self) -> GenerationTiming {
        self.timing
    }
    fn run_id(&self) -> u64 {
        self.run_id
    }
    fn fork(
        &mut self,
        snapshot: &SpeculativeSnapshotHandle,
    ) -> Result<SpeculativeBranchHandle, SpeculativeControlError> {
        self.fork_inner(snapshot)
    }
    fn exchange(
        &mut self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError> {
        self.exchange_inner(branch)
    }
    fn branch_info(
        &self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<SpeculativeBranchInfo, SpeculativeControlError> {
        self.branch_info_inner(branch)
    }
    fn release_branch(
        &mut self,
        branch: &SpeculativeBranchHandle,
    ) -> Result<(), SpeculativeControlError> {
        self.owned_branch(branch)?;
        self.branches.remove(&branch.id);
        Ok(())
    }
    fn sampling_state(&self) -> Option<crate::execution_control::SamplingStateFacts> {
        let (temperature, requires_positive_temperature, has_rng) =
            self.request()?.control_sampling_facts()?;
        Some(crate::execution_control::SamplingStateFacts {
            temperature,
            requires_positive_temperature,
            has_rng,
        })
    }
    fn override_sampling(
        &mut self,
        request: crate::execution_control::SamplingOverride,
    ) -> Result<crate::execution_control::SamplingStateFacts, SpeculativeControlError> {
        self.healthy()?;
        let facts = self
            .sampling_state()
            .ok_or(SpeculativeControlError::Unsupported(
                "sampler has no prospective sampling control",
            ))?;
        let action = crate::execution_control::validate_sampling_override::<
            eredu_core::BackendFailure,
        >(facts, request)
        .map_err(|e| match e {
            crate::execution_control::SamplingOverrideError::Invalid(reason) => {
                SpeculativeControlError::Invalid(reason)
            }
            crate::execution_control::SamplingOverrideError::Backend(e) => {
                SpeculativeControlError::backend(e)
            }
        })?;
        self.scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request")
            .control_override_sampling(
                action.temperature(),
                action.reseed(),
                self.scheduler.context,
            )?;
        Ok(self.sampling_state().expect("supported sampler"))
    }
    fn force_next_token(&mut self, token: u32) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        self.scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request")
            .control_force_next(token, self.vocabulary)
    }
    fn clear_forced_token(&mut self) -> Result<bool, SpeculativeControlError> {
        self.healthy()?;
        let request = self
            .scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request");
        request.validate_control_edit()?;
        request.sampler_mut().try_control_clear_forced()
    }
    fn intervene(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    ) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        let controls=eredu_core::intervention::AdmittedInterventionPlan::discovery_validation_control_bytes()
            .and_then(|n|n.checked_mul(plans.len()))
            .and_then(|n|n.checked_add(std::mem::size_of::<Result<(),SpeculativeControlError>>()
                +std::mem::size_of::<Vec<eredu_core::speculative::SpeculativeInterventionPlan>>()));
        let _host=self.scheduler.executor.driver_host_metadata(controls,self.scheduler.context)
            .map_err(|e|SpeculativeControlError::backend_with_retained(e,E::take_retained_failure))?;
        if let Some(discovery) = self.intervention_discovery.as_ref() {
            for plan in &plans {
                plan.plan.validate_discovery(discovery).map_err(|_|SpeculativeControlError::Invalid(
                    "intervention belongs to another loaded source or session"))?;
            }
        }
        if let Some(lane) = self.lane.as_mut() {
            let sampler=lane.runtime_mut().sampler_mut();
            if self.intervention_discovery.is_none() {
                sampler.validate_control_interventions_prepared(&plans,self.scheduler.context)?;
            }
            return sampler.control_intervene_prepared(plans,self.scheduler.context);
        }
        let request = self
            .scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request");
        request.validate_control_edit()?;
        let sampler=request.sampler_mut();
        if self.intervention_discovery.is_none() {
            sampler.validate_control_interventions_prepared(&plans,self.scheduler.context)?;
        }
        sampler.control_intervene_prepared(plans,self.scheduler.context)
    }
    fn epoch(&self) -> u64 {
        self.epoch
    }
    fn step(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.step_inner())) {
            Ok(mut result) => {
                if result.is_err() {
                    self.failed = true;
                    if let Some(error) = self.scheduler.executor.take_activation_error() {
                        result = Err(error);
                    }
                }
                result
            }
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn take_activation_evidence(
        &mut self,
    ) -> Result<Option<ControlledSpeculativeActivation>, SpeculativeControlError> {
        let Some(activation) = self.scheduler.executor.take_activation_capture() else {
            return Ok(None);
        };
        let next = self
            .sequence
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let record = ControlledSpeculativeActivation {
            schema_version: 1,
            sequence: self.sequence,
            run_id: self.run_id,
            epoch: self.epoch,
            activation,
        };
        self.charge_record(&record)?;
        self.sequence = next;
        Ok(Some(record))
    }

    fn cancel(&mut self) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        // Submit a pre-cancelled lane to retain the ordinary cancellation semantics
        // without prefill, then settle pending verification through the same driver.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(lane) = self.lane.as_mut() {
                lane.runtime_mut().cancellation().cancel();
            }
            if let Some(id) = self.id {
                self.scheduler.cancel(id).map_err(driver_failure::<E, S>)?;
            }
            // Use the same action and bounded-record boundary as peers calling
            // step, including cancellation of a retained verification.
            while self.step_inner()?.is_some() {}
            Ok(())
        }))
        .unwrap_or_else(|payload| {
            self.failed = true;
            std::panic::resume_unwind(payload)
        });
        if result.is_err() {
            self.failed = true;
            if let Some(error) = self.scheduler.executor.take_activation_error() {
                return Err(error);
            }
        }
        result
    }
    fn can_snapshot(&self) -> bool {
        matches!(
            self.snapshot_support(),
            eredu_core::execution_control::ControlSupport::Supported
        )
    }
    fn snapshot_support(&self) -> eredu_core::execution_control::ControlSupport {
        use eredu_core::execution_control::ControlSupport;
        let reason = if self.failed {
            Some("session failed")
        } else if self.budget.is_none() {
            Some("snapshot limits were not supplied")
        } else if self.request().is_none_or(|r| !r.is_control_boundary()) {
            Some(
                "snapshots require a canonical boundary after prefill, with no retained proposals or verification",
            )
        } else if self
            .request()
            .is_none_or(|r| r.status() == SpeculativeRequestStatus::Cancelled)
        {
            Some("cancelled sessions cannot be restored")
        } else {
            self.request()
                .and_then(|r| r.control_snapshot_unavailable_reason(self.scheduler.executor))
        };
        reason.map_or(ControlSupport::Supported, |reason| {
            ControlSupport::Unsupported {
                reason: reason.into(),
            }
        })
    }
    fn snapshot(&mut self) -> Result<SpeculativeSnapshotHandle, SpeculativeControlError> {
        let next = self
            .next_snapshot
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let table_bytes = self.snapshots.growth_bytes(self.scheduler.executor);
        let prepared = self.prepare_save(SnapshotResourceKind::Snapshot, table_bytes)?;
        self.snapshots.prepare_insert(self.scheduler.executor, self.scheduler.context)?;
        let saved = self.finish_save(prepared)?;
        let id = self.next_snapshot;
        self.snapshots.insert(id, saved)?;
        self.next_snapshot = next;
        Ok(SpeculativeSnapshotHandle {
            owner: self.owner.clone(),
            id,
        })
    }
    fn restore(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        self.owned(handle)?;
        if self.snapshots[&handle.id].run_id != self.run_id {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        if !self.request().is_some_and(|r| r.is_control_boundary()) {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        let saved = &self.snapshots[&handle.id];
        let _reservation = self.reserve_control(SnapshotResourceKind::Restore, saved.estimate)?;
        let request = self
            .scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request");
        if !request.is_control_boundary() {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        self.failed = true;
        let result = request.restore_control_snapshot(
            self.scheduler.executor,
            &saved.state,
            self.scheduler.context,
        );
        if result.is_ok() {
            self.failed = false;
            self.epoch = epoch;
        }
        result
    }
    fn release_snapshot(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<(), SpeculativeControlError> {
        self.owned(handle)?;
        self.snapshots.remove(&handle.id);
        Ok(())
    }
    fn snapshot_usage(&self) -> SnapshotUsage {
        self.budget
            .as_ref()
            .map_or_else(SnapshotUsage::default, SnapshotBudget::usage)
    }
}

/// Scoped driver selected by the facade after ordinary speculative preflight.
pub struct DriveControlledSpeculation<'f, F> {
    run: RunSpeculativeGeneration,
    options: ControlledSpeculativeOptions,
    drive: F,
    failure: &'f mut Option<SpeculativeControlError>,
    vocabulary: usize,
    intervention_discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    activation_discovery: Option<eredu_core::speculative::SpeculativeActivationDiscovery>,
}
impl<'f, F> DriveControlledSpeculation<'f, F> {
    /// Starts active preparation timing before facade and backend preparation.
    pub fn new(
        scheduler: eredu_core::generation::SpeculativeSchedulerOptions,
        options: ControlledSpeculativeOptions,
        drive: F,
        failure: &'f mut Option<SpeculativeControlError>,
    ) -> Self {
        Self {
            run: RunSpeculativeGeneration::new(scheduler),
            options,
            drive,
            failure,
            vocabulary: 0,
            intervention_discovery: None,
            activation_discovery: None,
        }
    }
}
impl<F> DriveControlledSpeculation<'_, F> {
    /// Retains authoritative loaded internal discovery for prospective edits.
    pub fn with_activation_discovery(
        mut self,
        discovery: Option<eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Self {
        self.activation_discovery = discovery;
        self
    }
    /// Retains exact loaded capabilities for re-admission of prospective edits.
    pub fn with_intervention_discovery(
        mut self,
        discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    ) -> Self {
        self.intervention_discovery = discovery;
        self
    }
    /// Supplies the canonical tokenizer ID ceiling; the sampler also checks its active grammar.
    pub fn with_vocabulary(mut self, vocabulary: usize) -> Self {
        self.vocabulary = vocabulary;
        self
    }
}
impl<F> SpeculativeGenerationVisitor for DriveControlledSpeculation<'_, F>
where
    F: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
{
    fn scheduler_options(&self) -> Option<eredu_core::generation::SpeculativeSchedulerOptions> {
        self.run.scheduler_options()
    }

    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: impl Into<SpeculativeBuffer<PreparedSpeculativeLane<'a, E, S, C, P>>>,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>
    where
        E: SpeculativeExecutor + 'a,
        S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>>
            + 'a,
        C: SpeculativeConstraint,
        P: SpeculativePublisher<C>,
    {
        let mut lanes = lanes.into();
        // The shared provider prices this actual scope and final error bridge
        // before either is constructed. Custody remains live through both return
        // paths; the exact typed cause stays in the caller's failure destination.
        let controls = [std::mem::size_of::<Self>(), std::mem::size_of::<Session<'a, E, S, C, P>>(),
            std::mem::size_of::<HostPreparationAuthority>(), std::mem::size_of::<SpeculativeControlError>(),
            std::mem::size_of::<eredu_core::SpeculativeOutputError>(),
            std::mem::size_of::<
                Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>,
            >(),
        ];
        let publication_host = executor
            .driver_host_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .and_then(|n| {
                        n.checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                            SpeculativeDriverError<E::Error>,
                        >()?)
                    }),
                context,
            )
            .map_err(SpeculativeDriverError::Backend)?;
        let retained_bridge = !publication_host.is_unmanaged();
        let result = (|| {
            let local = if lanes.len() != 1 {
                Err(SpeculativeControlError::Unsupported(
                    "controlled speculation requires exactly one lane",
                ))
            } else {
                Ok(())
            };
            let stage = eredu_core::run_preparation::TextPreparationStage::Request;
            eredu_core::run_preparation::finish_preparation(
                stage,
                local,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeControlError::backend,
            )?;
            let mut lane = lanes.pop().expect("checked single lane");
            let local = (|| {
                if let Some(plan) = self.options.activations {
                    // A fresh single-lane scheduler assigns its first request index zero.
                    executor.configure_activation_capture(
                        plan,
                        SpeculativeRequestId::new(0),
                        context,
                    )?;
                }
                if let Some(plan) = self.options.capture {
                    if plan.request().batch != 1 || plan.request().prompt_tokens != 1
                        || plan.request().max_predictions < lane.config().max_tokens as u64 {
                        return Err(SpeculativeControlError::Invalid("capture admission does not cover the selected one-row speculative request"));
                    }
                    lane.runtime_mut()
                        .sampler_mut()
                        .enable_control_capture_prepared(plan, context)?;
                }
                Ok(())
            })();
            let stage = eredu_core::run_preparation::TextPreparationStage::Instrumentation;
            eredu_core::run_preparation::finish_preparation(
                stage,
                local,
                |status| executor.agree_text_preparation(stage, status, context),
                SpeculativeControlError::backend,
            )?;
            let mut scheduler = SpeculativeScheduler::new(
                executor,
                self.run.options,
                topology,
                optimistic_execution_available,
                component_timings_collected,
                context,
            )
            .map_err(driver_failure::<E, S>)?;
            scheduler.reserve_requests(1).map_err(driver_failure::<E, S>)?;
            let outputs = scheduler.executor.driver_buffer(1, context)
                .map_err(SpeculativeDriverError::Backend);
            let outputs = eredu_core::run_preparation::finish_preparation(
                stage,
                outputs,
                |status| {
                    scheduler
                        .executor
                        .agree_text_preparation(stage, status, context)
                },
                SpeculativeDriverError::Preparation,
            )
            .map_err(driver_failure::<E, S>)?;
            let prepared = (|| {
                let owner = scheduler.executor.driver_identity(context).map_err(|e| {
                    SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
                })?;
                let budget = self
                    .options
                    .snapshots
                    .map(|limits| {
                        let host = scheduler
                            .executor
                            .driver_host_metadata(SnapshotBudget::construction_bytes(), context)
                            .map_err(|e| {
                                SpeculativeControlError::backend_with_retained(
                                    e,
                                    E::take_retained_failure,
                                )
                            })?;
                        Ok::<_, SpeculativeControlError>(SnapshotBudget::new_with_host(
                            limits, host,
                        ))
                    })
                    .transpose()?;
                Ok::<_, SpeculativeControlError>((owner, budget))
            })();
            let (owner, budget) = eredu_core::run_preparation::finish_preparation(
                stage,
                prepared,
                |status| {
                    scheduler
                        .executor
                        .agree_text_preparation(stage, status, context)
                },
                SpeculativeControlError::backend,
            )?;
            let mut session = Session {
                scheduler,
                lane: Some(lane),
                id: None,
                owner,
                snapshots: ControlTable::default(),
                branches: ControlTable::default(),
                next_branch: 1,
                run_id: 0,
                next_snapshot: 0,
                budget,
                trace: TraceBudget::new(self.options.trace_limits),
                retained_record_funding: retained_bridge,
                sequence: 0,
                epoch: 0,
                preparation: self.run.started.elapsed(),
                timing: GenerationTiming::default(),
                failed: false,
                vocabulary: self.vocabulary,
                intervention_discovery: self.intervention_discovery,
                activation_discovery: self.activation_discovery,
            };
            let driven = (self.drive)(&mut session);
            if session.failed {
                // The failing step already agreed its outcome. Even callers
                // that ignore it cannot start an unmatched final exchange.
                driven?;
                return Err(SpeculativeControlError::Failed);
            }
            if driven.is_err() {
                // A caller-owned failure aligns with a peer's next readiness
                // check or its final controlling-closure disposition.
                session.agree_delivery(driven)?;
            }
            if session.lane.is_some() || !session.scheduler.is_finished() {
                session.cancel()?;
            }
            session.agree_delivery(Ok(()))?;
            let timing = session.timing;
            let completed = session.scheduler.finish().map_err(driver_failure::<E, S>)?;
            completed_output::<_, E::Error>(completed, outputs, |_| timing).map_err(driver_failure::<E, S>)
        })();
        result.map_err(|error| {
            let message = if retained_bridge {
                eredu_core::SpeculativeOutputError::Storage("controlled speculative driver failed")
            } else {
                eredu_core::SpeculativeOutputError::publication(error.to_string())
            };
            *self.failure = Some(error);
            SpeculativeDriverError::Output(message)
        })
    }
}
