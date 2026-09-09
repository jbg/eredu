//! Scoped controlled access to the same speculative scheduler used by generation.

use super::*;
use crate::execution_control::{SnapshotBudget, SnapshotReservation, TraceBudget, TraceLimits};
use eredu_core::{
    execution_control::{
        ExecutionControlError, SnapshotEstimate, SnapshotLimits, SnapshotResourceKind,
        SnapshotUsage,
    },
    generation::{SpeculativeRequestId, SpeculativeRequestStatus},
    speculative::{SpeculativeControlError, SpeculativeControlSnapshot, SpeculativeProposalView},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

/// Explicit resource limits for controlled speculative inspection.
#[derive(Debug, Clone)]
pub struct ControlledSpeculativeOptions {
    /// JSON transport limits shared with observed and ordinary controlled runs.
    pub trace_limits: TraceLimits,
    /// Admitted raw-logit selections, captured independently for draft and target.
    /// Other activation paths require a backend with phase-aware capture support.
    pub capture: Option<eredu_core::capture::AdmittedCapturePlan>,
    /// Snapshot retention and copying limits. `None` disables snapshots.
    pub snapshots: Option<SnapshotLimits>,
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
    pub dispositions: Vec<SpeculativeProposalDisposition>,
    /// Exact newly committed target tokens (accepted prefix plus replacement/bonus).
    pub committed_token_ids: Vec<u32>,
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
    /// Restoration epoch, incremented after each successful restore.
    pub epoch: u64,
    /// Scheduler status after this action.
    pub status: SpeculativeRequestStatus,
    /// Proposal block when created, extended or promoted from optimistic work.
    /// A promoted block can contain previously reported, reused proposals.
    pub drafted: Option<SpeculativeProposalView>,
    /// Completed verification with explicit acceptance and rejection.
    pub verification: Option<SpeculativeVerificationRecord>,
    /// Newly committed tokens; proposals are excluded.
    pub committed_token_ids: Vec<u32>,
    /// Bounded raw-logit captures, including tentative draft and target rows.
    pub captures: Vec<eredu_core::speculative::SpeculativePredictionCapture>,
    /// Active time to the first committed target token, before its callbacks.
    pub timing: GenerationTiming,
    /// Active host duration of this action, including synchronous publication.
    pub step_seconds: f64,
}

/// Session-local immutable snapshot handle. Retention ends on explicit release
/// or when the scoped session exits; copying handles does not copy native state.
#[derive(Debug, Clone)]
pub struct SpeculativeSnapshotHandle {
    owner: Arc<()>,
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
    /// Current scheduler phase, or `Prefill` before the first step.
    fn status(&self) -> SpeculativeRequestStatus;
    /// Canonical generated-token prefix.
    fn token_ids(&self) -> &[u32];
    /// Active TTFT, unaffected by pauses, snapshots or inspection work.
    fn timing(&self) -> GenerationTiming;
    /// Current restore epoch.
    fn epoch(&self) -> u64;
    /// Performs one action; `None` means the request is already terminal.
    fn step(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError>;
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
    estimate: SnapshotEstimate,
    _reservation: SnapshotReservation,
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
    owner: Arc<()>,
    snapshots: BTreeMap<u64, Saved<E, S, C>>,
    next_snapshot: u64,
    budget: Option<SnapshotBudget>,
    trace: TraceBudget,
    sequence: u64,
    epoch: u64,
    preparation: Duration,
    timing: GenerationTiming,
    failed: bool,
}

impl<'a, E, S, C, P> Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
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
    fn owned(&self, handle: &SpeculativeSnapshotHandle) -> Result<(), SpeculativeControlError> {
        if !Arc::ptr_eq(&self.owner, &handle.owner) || !self.snapshots.contains_key(&handle.id) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        Ok(())
    }
    fn step_inner(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError> {
        self.healthy()?;
        if self.id.is_some() && self.scheduler.is_finished() {
            return Ok(None);
        }
        let before = self.request().map(|r| {
            (
                r.sequence().tokens().len(),
                r.proposals(),
                r.optimistic_proposals(),
                r.stats().clone(),
            )
        });
        let started = std::time::Instant::now();
        let was_prefill = self.lane.is_some();
        if let Some(lane) = self.lane.take() {
            self.id = Some(
                self.scheduler
                    .submit(lane)
                    .map_err(SpeculativeControlError::backend)?,
            );
            self.timing = GenerationTiming::new(
                self.request()
                    .and_then(|r| r.stats().submission_to_first_token())
                    .map(|t| self.preparation + t),
            );
        } else {
            self.scheduler
                .step()
                .map_err(SpeculativeControlError::backend)?;
        }
        let elapsed = started.elapsed();
        let request = self.request().expect("submitted request");
        let old_len = before.as_ref().map_or(0, |b| b.0);
        let committed = request.sequence().tokens()[old_len..].to_vec();
        let after_proposals = request.proposals();
        let after_optimistic = request.optimistic_proposals();
        let drafted = if after_optimistic != before.as_ref().and_then(|b| b.2.clone())
            && after_optimistic.is_some()
        {
            after_optimistic
        } else if after_proposals != before.as_ref().and_then(|b| b.1.clone())
            && after_proposals.is_some()
        {
            after_proposals
        } else {
            None
        };
        let verification = before.and_then(|(_, proposals, optimistic, stats)| {
            let proposals = proposals?;
            if request.stats().rounds() == stats.rounds()
                && request.status() != SpeculativeRequestStatus::Cancelled
            {
                return None;
            }
            let accepted = request.stats().accepted_tokens() - stats.accepted_tokens();
            let dispositions = (0..proposals.token_ids.len())
                .map(|i| {
                    if i < accepted {
                        SpeculativeProposalDisposition::Accepted
                    } else if i == accepted && committed.len() > accepted {
                        SpeculativeProposalDisposition::Rejected
                    } else {
                        SpeculativeProposalDisposition::Discarded
                    }
                })
                .collect();
            Some(SpeculativeVerificationRecord {
                resolved: request.stats().rounds() > stats.rounds(),
                proposals,
                dispositions,
                committed_token_ids: committed.clone(),
                optimistic,
                optimistic_reused: request.stats().reused_optimistic_tokens()
                    - stats.reused_optimistic_tokens(),
                optimistic_consumed: request.stats().consumed_optimistic_tokens()
                    - stats.consumed_optimistic_tokens(),
                optimistic_discarded: request.stats().discarded_optimistic_tokens()
                    - stats.discarded_optimistic_tokens(),
            })
        });
        let mut record = ControlledSpeculativeStep {
            schema_version: 1,
            sequence: self.sequence,
            epoch: self.epoch,
            status: request.status(),
            drafted: if was_prefill { None } else { drafted },
            verification,
            committed_token_ids: committed,
            captures: Vec::new(),
            timing: self.timing,
            step_seconds: elapsed.as_secs_f64(),
        };
        record.captures = self
            .scheduler
            .requests
            .request_mut(self.id.expect("submitted request"))
            .expect("submitted request")
            .sampler_mut()
            .take_control_captures();
        self.trace.charge(&record)?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        Ok(Some(record))
    }
}

impl<'a, E, S, C, P> ControlledSpeculativeSession for Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
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
    fn epoch(&self) -> u64 {
        self.epoch
    }
    fn step(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.step_inner())) {
            Ok(result) => {
                if result.is_err() {
                    self.failed = true;
                }
                result
            }
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn cancel(&mut self) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        // Submit a pre-cancelled lane to retain the ordinary cancellation semantics
        // without prefill, then settle pending verification through the same driver.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(lane) = self.lane.as_mut() {
                lane.runtime_mut().cancellation().cancel();
            }
            if self.lane.is_some() {
                self.step_inner()?;
            }
            if let Some(id) = self.id {
                self.scheduler
                    .cancel(id)
                    .map_err(SpeculativeControlError::backend)?;
                self.scheduler
                    .run()
                    .map_err(SpeculativeControlError::backend)?;
            }
            Ok(())
        }))
        .unwrap_or_else(|payload| {
            self.failed = true;
            std::panic::resume_unwind(payload)
        });
        if result.is_err() {
            self.failed = true;
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
            Some("snapshots require a canonical boundary after prefill, with no retained proposals or verification")
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
        // Include the scope's saved-state envelope and conservative map-node
        // storage as well as the core/native snapshot payload.
        let overhead = (std::mem::size_of::<Saved<E, S, C>>() as u64)
            .checked_add(256)
            .ok_or(ExecutionControlError::Overflow)?;
        let estimate = SnapshotEstimate {
            retained_bytes: estimate
                .retained_bytes
                .checked_add(overhead)
                .ok_or(ExecutionControlError::Overflow)?,
            copy_bytes: estimate
                .copy_bytes
                .checked_add(overhead)
                .ok_or(ExecutionControlError::Overflow)?,
        };
        let next = self
            .next_snapshot
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let reservation = budget.reserve(SnapshotResourceKind::Snapshot, Some(estimate))?;
        let state = request.control_snapshot(self.scheduler.executor, self.scheduler.context)?;
        let id = self.next_snapshot;
        self.next_snapshot = next;
        self.snapshots.insert(
            id,
            Saved {
                state,
                estimate,
                _reservation: reservation,
            },
        );
        Ok(SpeculativeSnapshotHandle {
            owner: Arc::clone(&self.owner),
            id,
        })
    }
    fn restore(
        &mut self,
        handle: &SpeculativeSnapshotHandle,
    ) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        self.owned(handle)?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        if !self.request().is_some_and(|r| r.is_control_boundary()) {
            return Err(SpeculativeControlError::NotQuiescent);
        }
        let saved = &self.snapshots[&handle.id];
        let _reservation = self
            .budget
            .as_ref()
            .expect("saved snapshot budget")
            .reserve(SnapshotResourceKind::Restore, Some(saved.estimate))?;
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
        }
    }
}
impl<F> SpeculativeGenerationVisitor for DriveControlledSpeculation<'_, F>
where
    F: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
{
    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        mut lanes: Vec<PreparedSpeculativeLane<'a, E, S, C, P>>,
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
        let result = (|| {
            if lanes.len() != 1 {
                return Err(SpeculativeControlError::Unsupported(
                    "controlled speculation requires exactly one lane",
                ));
            }
            let mut lane = lanes.pop().expect("checked single lane");
            if let Some(plan) = self.options.capture.clone() {
                lane.runtime_mut()
                    .sampler_mut()
                    .enable_control_capture(plan)?;
            }
            let scheduler = SpeculativeScheduler::new(
                executor,
                self.run.options,
                topology,
                optimistic_execution_available,
                component_timings_collected,
                context,
            )
            .map_err(SpeculativeControlError::backend)?;
            let mut session = Session {
                scheduler,
                lane: Some(lane),
                id: None,
                owner: Arc::new(()),
                snapshots: BTreeMap::new(),
                next_snapshot: 0,
                budget: self.options.snapshots.map(SnapshotBudget::new),
                trace: TraceBudget::new(self.options.trace_limits),
                sequence: 0,
                epoch: 0,
                preparation: self.run.started.elapsed(),
                timing: GenerationTiming::default(),
                failed: false,
            };
            (self.drive)(&mut session)?;
            session.healthy()?;
            if session.lane.is_some() || !session.scheduler.is_finished() {
                session.cancel()?;
            }
            let timing = session.timing;
            let completed = session
                .scheduler
                .finish()
                .map_err(SpeculativeControlError::backend)?;
            completed_output::<_, E::Error>(completed, |_| timing)
                .map_err(SpeculativeControlError::backend)
        })();
        result.map_err(|error| {
            let message = error.to_string();
            *self.failure = Some(error);
            SpeculativeDriverError::Output(eredu_core::SpeculativeOutputError::publication(message))
        })
    }
}
