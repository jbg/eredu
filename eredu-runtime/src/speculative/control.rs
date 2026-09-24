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
use std::{collections::BTreeMap, rc::Rc, sync::Arc, time::Duration};

mod branch;
mod forecast;
pub use branch::*;

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
    pub committed_token_ids: Vec<u32>,
    /// Forced canonical token consumed by this action, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forced_token: Option<u32>,
    /// Effective prospective sampling policy after this action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<crate::execution_control::SamplingStateFacts>,
    /// Bounded raw-logit captures, including tentative draft and target rows.
    pub captures: Vec<eredu_core::speculative::SpeculativePredictionCapture>,
    /// Internal component evidence from actual forwards in this action. Physical
    /// rows, scheduler coordinates and tentative phases remain distinct. These
    /// records share this step's run/restore identity and transport accounting.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activations: Vec<eredu_core::speculative::SpeculativeActivationCapture>,
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
    /// Releases optional conversions only at a canonical boundary. Pending work
    /// is rejected without polling, settling, advancing or consuming budgets.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads role-labelled retention budgets, including while native work is pending.
    /// Does not poll, settle, reserve, evaluate, or advance the controlled run.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<eredu_core::residency::ExecutionConversionRetentionReport, SpeculativeControlError>
    {
        Err(SpeculativeControlError::Unsupported(
            "conversion retention observation",
        ))
    }

    /// Read-only outlook at an advanced canonical boundary without proposals or
    /// pending verification. Does not poll, settle, copy, reserve or advance work.
    /// The what-if horizon does not change the configured generation limit.
    fn forecast_remaining_generation(
        &self,
        additional_tokens: u64,
        options: &crate::memory_forecast::GenerationForecastOptions,
    ) -> Result<
        crate::memory_forecast::SpeculativeContinuationForecast,
        crate::memory_forecast::GenerationForecastError,
    > {
        let _ = (additional_tokens, options);
        Err(
            crate::memory_forecast::GenerationForecastError::UnsupportedContinuation(
                "controlled session has no speculative continuation projection".into(),
            ),
        )
    }
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
    snapshots: BTreeMap<u64, Rc<Saved<E, S, C>>>,
    branches: BTreeMap<u64, branch::Branch<E, S, C>>,
    next_branch: u64,
    run_id: u64,
    next_snapshot: u64,
    budget: Option<SnapshotBudget>,
    trace: TraceBudget,
    sequence: u64,
    epoch: u64,
    preparation: Duration,
    timing: GenerationTiming,
    failed: bool,
    vocabulary: usize,
    intervention_discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    activation_discovery: Option<eredu_core::speculative::SpeculativeActivationDiscovery>,
    forecast_profiles: Option<(
        crate::memory_forecast::LoadedMemoryProfile,
        crate::memory_forecast::SpeculativeMemoryProfile,
    )>,
    forecast_instrumented: bool,
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
        if !Arc::ptr_eq(&self.owner, &handle.owner) || !self.snapshots.contains_key(&handle.id) {
            return Err(SpeculativeControlError::IncompatibleSnapshot);
        }
        Ok(())
    }
    fn save_state(
        &self,
        kind: SnapshotResourceKind,
    ) -> Result<Rc<Saved<E, S, C>>, SpeculativeControlError> {
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
        let reservation = budget.reserve(kind, Some(estimate))?;
        let state = request.control_snapshot(self.scheduler.executor, self.scheduler.context)?;
        Ok(Rc::new(Saved {
            state,
            run_id: self.run_id,
            estimate,
            _reservation: reservation,
        }))
    }
    fn step_inner(&mut self) -> Result<Option<ControlledSpeculativeStep>, SpeculativeControlError> {
        self.healthy()?;
        if self.id.is_some() && self.scheduler.is_finished() {
            return Ok(None);
        }
        let forced = self
            .request()
            .and_then(|r| r.sampler().control_pending_forced());
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
            run_id: self.run_id,
            status: request.status(),
            drafted: if was_prefill { None } else { drafted },
            verification,
            forced_token: forced.filter(|_| !committed.is_empty()),
            sampling: self.sampling_state(),
            committed_token_ids: committed,
            captures: Vec::new(),
            activations: Vec::new(),
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
        while let Some(capture) = self.scheduler.executor.take_activation_capture() {
            record.activations.push(capture);
        }
        let local = (|| {
            self.trace.charge(&record)?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or(ExecutionControlError::Overflow)?;
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
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        if self.failed
            || !self
                .request()
                .is_some_and(|request| request.is_control_boundary())
        {
            return Err(eredu_core::residency::ParameterConversionTrimError::NotQuiescent);
        }
        self.scheduler.executor.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<eredu_core::residency::ExecutionConversionRetentionReport, SpeculativeControlError>
    {
        self.scheduler
            .executor
            .parameter_conversion_retention()
            .map_err(SpeculativeControlError::Backend)?
            .ok_or(SpeculativeControlError::Unsupported(
                "conversion retention observation",
            ))
    }

    fn forecast_remaining_generation(
        &self,
        additional_tokens: u64,
        options: &crate::memory_forecast::GenerationForecastOptions,
    ) -> Result<
        crate::memory_forecast::SpeculativeContinuationForecast,
        crate::memory_forecast::GenerationForecastError,
    > {
        self.forecast_inner(additional_tokens, options)
    }
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
        plan.validate(self.activation_discovery.as_ref().ok_or(
            SpeculativeControlError::Unsupported(
                "loaded execution has no internal activation discovery",
            ),
        )?)?;
        let instrumented = !plan.is_empty();
        self.scheduler
            .executor
            .readmit_activation_interventions(plan)?;
        self.forecast_instrumented |= instrumented;
        Ok(())
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
        Ok(request.sampler_mut().control_clear_forced())
    }
    fn intervene(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    ) -> Result<(), SpeculativeControlError> {
        self.healthy()?;
        if !plans.is_empty() {
            let discovery = self.intervention_discovery.as_ref().ok_or(
                SpeculativeControlError::Unsupported(
                    "loaded execution has no speculative intervention discovery",
                ),
            )?;
            for plan in &plans {
                let checked = plan.plan.plan().clone().admit(
                    discovery,
                    plan.plan.request(),
                    plan.plan.session_id(),
                )?;
                if checked.identity() != plan.plan.identity() {
                    return Err(SpeculativeControlError::Invalid(
                        "intervention belongs to another loaded source or session",
                    ));
                }
            }
        }
        let instrumented = plans.iter().any(|p| !p.plan.plan().operations.is_empty());
        if let Some(lane) = self.lane.as_mut() {
            lane.runtime_mut().sampler_mut().control_intervene(plans)?;
            self.forecast_instrumented |= instrumented;
            return Ok(());
        }
        let request = self
            .scheduler
            .requests
            .request_mut(self.id.ok_or(SpeculativeControlError::NotQuiescent)?)
            .expect("submitted request");
        request.validate_control_edit()?;
        request.sampler_mut().control_intervene(plans)?;
        self.forecast_instrumented |= instrumented;
        Ok(())
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
        self.trace.charge(&record)?;
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
                self.scheduler
                    .cancel(id)
                    .map_err(SpeculativeControlError::backend)?;
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
        let next = self
            .next_snapshot
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let saved = self.save_state(SnapshotResourceKind::Snapshot)?;
        let id = self.next_snapshot;
        self.next_snapshot = next;
        self.snapshots.insert(id, saved);
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
    vocabulary: usize,
    intervention_discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    activation_discovery: Option<eredu_core::speculative::SpeculativeActivationDiscovery>,
    forecast_profiles: Option<(
        crate::memory_forecast::LoadedMemoryProfile,
        crate::memory_forecast::SpeculativeMemoryProfile,
    )>,
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
            forecast_profiles: None,
        }
    }
}
impl<F> DriveControlledSpeculation<'_, F> {
    /// Supplies immutable selected geometry before execution resources are borrowed.
    /// Native residency/capacity are refreshed from the active executor on observation.
    pub fn with_forecast_profiles(
        mut self,
        profiles: Option<(
            crate::memory_forecast::LoadedMemoryProfile,
            crate::memory_forecast::SpeculativeMemoryProfile,
        )>,
    ) -> Self {
        self.forecast_profiles = profiles;
        self
    }
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
                if let Some(plan) = self.options.activations.clone() {
                    // A fresh single-lane scheduler assigns its first request index zero.
                    executor.configure_activation_capture(
                        plan,
                        SpeculativeRequestId::new(0),
                        context,
                    )?;
                }
                if let Some(plan) = self.options.capture.clone() {
                    lane.runtime_mut()
                        .sampler_mut()
                        .enable_control_capture(plan)?;
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
                branches: BTreeMap::new(),
                next_branch: 1,
                run_id: 0,
                next_snapshot: 0,
                budget: self.options.snapshots.map(SnapshotBudget::new),
                trace: TraceBudget::new(self.options.trace_limits),
                sequence: 0,
                epoch: 0,
                preparation: self.run.started.elapsed(),
                timing: GenerationTiming::default(),
                failed: false,
                vocabulary: self.vocabulary,
                intervention_discovery: self.intervention_discovery,
                activation_discovery: self.activation_discovery,
                forecast_profiles: self.forecast_profiles,
                forecast_instrumented: self
                    .options
                    .capture
                    .as_ref()
                    .is_some_and(|c| !c.plan().selections.is_empty())
                    || self
                        .options
                        .activations
                        .as_ref()
                        .is_some_and(|a| !a.is_empty()),
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
