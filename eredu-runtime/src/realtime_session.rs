//! Backend-neutral realtime session ownership over the singular fair scheduler.

use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use eredu_core::{
    consensus::{validate_ranked_identity_bounded, BoundedConsensusTransport},
    scheduler::{
        RequestId, RequestStatus, Scheduler, SchedulerCapabilities, SchedulerError,
        SchedulerLimits, SchedulerProgress, SchedulerReport, SemanticStateTransaction,
        TransitionOutput, WorkId,
    },
    BoundedCompletionWait, Completion, ParallelTopology, RealtimeInputFrame, RealtimeSampling,
    RealtimeSpeechConfig,
};
use sha2::{Digest, Sha256};

use crate::{
    CommunicationCompletionPolicy, LayerWeightResidency, RealtimeGenerationBranch,
    RealtimeGenerationState, RealtimeGenerationTransactionError, RealtimeIdentity,
    RealtimeIngressContract, RealtimePayloadContract, RealtimePayloadGeneration,
    RealtimePayloadOwnerIdentity, SelectedRealtimeRealization,
};

static NEXT_REALTIME_INCARNATION: AtomicU64 = AtomicU64::new(1);

/// Exact selected model identity bound to every session in one scheduler.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeModelSessionIdentity {
    selected: Option<Box<SelectedRealtimeRealization>>,
    architecture: RealtimeIdentity,
    source: RealtimeIdentity,
    execution: RealtimeIdentity,
    schedule_identity: RealtimeIdentity,
    schedule: RealtimeSpeechConfig,
    state_layout: RealtimeIdentity,
    topology_policy: RealtimeIdentity,
    topology: ParallelTopology,
    rank: usize,
    residency: LayerWeightResidency,
    completion: CommunicationCompletionPolicy,
}

impl RealtimeModelSessionIdentity {
    /// Derives exact session identity exclusively from one neutral selected realization.
    pub fn from_selected(selected: &SelectedRealtimeRealization) -> Self {
        Self {
            selected: Some(Box::new(selected.clone())),
            architecture: selected.requirements().architecture().clone(),
            source: selected.source().clone(),
            execution: selected.execution().clone(),
            schedule_identity: selected.requirements().speech_schedule_identity().clone(),
            schedule: selected.requirements().speech_schedule().clone(),
            state_layout: selected.requirements().state_layout_identity().clone(),
            topology_policy: selected.requirements().topology().identity().clone(),
            topology: selected.topology(),
            rank: selected.rank(),
            residency: selected.residency(),
            completion: selected.completion(),
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        architecture: RealtimeIdentity,
        source: RealtimeIdentity,
        execution: RealtimeIdentity,
        schedule_identity: RealtimeIdentity,
        schedule: RealtimeSpeechConfig,
        state_layout: RealtimeIdentity,
        topology_policy: RealtimeIdentity,
        topology: ParallelTopology,
        rank: usize,
        residency: LayerWeightResidency,
        completion: CommunicationCompletionPolicy,
    ) -> Self {
        Self {
            selected: None,
            architecture,
            source,
            execution,
            schedule_identity,
            schedule,
            state_layout,
            topology_policy,
            topology,
            rank,
            residency,
            completion,
        }
    }

    /// Exact portable schedule selected for this model.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Exact selected execution topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Selected bounded completion policy for topology-wide coordination.
    pub const fn completion(&self) -> CommunicationCompletionPolicy {
        self.completion
    }

    /// Derives a typed model-owner identity without backend names or objects.
    pub fn model_owner(&self) -> RealtimeModelOwnerIdentity {
        RealtimeModelOwnerIdentity(self.clone())
    }

    fn distributed_consensus_identity(&self) -> [u32; 8] {
        let mut digest = Sha256::new();
        for identity in [
            &self.architecture,
            &self.source,
            &self.execution,
            &self.schedule_identity,
            &self.topology_policy,
        ] {
            let bytes = identity.as_str().as_bytes();
            digest.update(
                u64::try_from(bytes.len())
                    .expect("identity byte length fits u64")
                    .to_le_bytes(),
            );
            digest.update(bytes);
        }
        for value in [
            format!("{:?}", self.schedule),
            format!("{:?}", self.topology),
            format!("{:?}", self.residency),
            format!("{:?}", self.completion),
        ] {
            digest.update(
                u64::try_from(value.len())
                    .expect("identity component length fits u64")
                    .to_le_bytes(),
            );
            digest.update(value.as_bytes());
        }
        if let Some(selected) = &self.selected {
            for value in [
                format!("{:?}", selected.state()),
                format!("{:?}", selected.observations()),
            ] {
                digest.update(
                    u64::try_from(value.len())
                        .expect("selected identity component length fits u64")
                        .to_le_bytes(),
                );
                digest.update(value.as_bytes());
            }
        }
        let digest = digest.finalize();
        std::array::from_fn(|index| {
            let start = index * 4;
            u32::from_le_bytes(
                digest[start..start + 4]
                    .try_into()
                    .expect("SHA-256 has eight complete u32 words"),
            )
        })
    }
}

/// Typed model owner derived from an exact neutral selected contract.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeModelOwnerIdentity(RealtimeModelSessionIdentity);

impl RealtimeModelOwnerIdentity {
    /// Exact neutral model/session identity represented by this owner.
    pub const fn session_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.0
    }
}

/// Monotonically allocated identity for one newly registered session.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimeSessionIncarnation(u64);

impl RealtimeSessionIncarnation {
    /// Monotonic process-local incarnation value.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// History-generation identity preserved across release and resume.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimeHistoryGeneration(u64);

impl RealtimeHistoryGeneration {
    /// Monotonic process-local history-generation value.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Canonical state for one fair-scheduler realtime request.
pub struct RealtimeSessionState<M, S, R, C> {
    model: RealtimeModelSessionIdentity,
    owner: RealtimeModelOwnerIdentity,
    incarnation: RealtimeSessionIncarnation,
    history_generation: RealtimeHistoryGeneration,
    committed_batch: Option<NonZeroUsize>,
    generation: RealtimeGenerationState<M, S, R, C>,
}

impl<M, S, R, C> RealtimeSessionState<M, S, R, C> {
    /// Exact selected model identity.
    pub const fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Typed model owner derived from the selected contract.
    pub const fn model_owner(&self) -> &RealtimeModelOwnerIdentity {
        &self.owner
    }

    /// Session incarnation allocated at first registration.
    pub const fn incarnation(&self) -> RealtimeSessionIncarnation {
        self.incarnation
    }

    /// Coordinate-history generation preserved by release/resume.
    pub const fn history_generation(&self) -> RealtimeHistoryGeneration {
        self.history_generation
    }

    /// Batch committed by the first successful frame, if any.
    pub const fn committed_batch(&self) -> Option<NonZeroUsize> {
        self.committed_batch
    }

    /// Derives the exact payload contract for the committed session batch.
    pub fn payload_contract(
        &self,
        ingress: &RealtimeIngressContract,
    ) -> Result<RealtimePayloadContract, RealtimeSessionExecutionError> {
        if ingress.schedule() != self.model.schedule() {
            return Err(RealtimeSessionExecutionError::IngressScheduleMismatch);
        }
        let batch = self
            .committed_batch
            .ok_or(RealtimeSessionExecutionError::BatchNotAdmitted)?;
        Ok(RealtimePayloadContract::new(
            ingress.schedule().clone(),
            batch.get(),
            ingress.text_domain(),
            ingress.audio_domain(),
            RealtimePayloadGeneration::new(self.history_generation.value())
                .expect("scheduler history generations are nonzero"),
            RealtimePayloadOwnerIdentity::new(self.incarnation.value())
                .expect("scheduler session incarnations are nonzero"),
        )
        .expect("scheduler-admitted payload contract has a positive batch"))
    }

    /// Canonical atomic generation state.
    pub const fn generation(&self) -> &RealtimeGenerationState<M, S, R, C> {
        &self.generation
    }

    /// Mutably borrows generation state while the scheduler proves the request idle.
    pub fn generation_mut(&mut self) -> &mut RealtimeGenerationState<M, S, R, C> {
        &mut self.generation
    }
}

/// Unpublished realtime session branch owned by the core scheduler.
pub struct RealtimeSessionBranch<MB, S, R, C> {
    model: RealtimeModelSessionIdentity,
    owner: RealtimeModelOwnerIdentity,
    incarnation: RealtimeSessionIncarnation,
    history_generation: RealtimeHistoryGeneration,
    committed_batch: Option<NonZeroUsize>,
    generation: RealtimeGenerationBranch<MB, S, R, C>,
}

impl<MB, S, R, C> RealtimeSessionBranch<MB, S, R, C> {
    /// Exact selected model identity.
    pub const fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Typed model owner used by payload contracts.
    pub const fn model_owner(&self) -> &RealtimeModelOwnerIdentity {
        &self.owner
    }

    /// Exact session incarnation.
    pub const fn incarnation(&self) -> RealtimeSessionIncarnation {
        self.incarnation
    }

    /// Exact coordinate-history generation.
    pub const fn history_generation(&self) -> RealtimeHistoryGeneration {
        self.history_generation
    }

    /// Batch admitted by this unpublished branch, including its current frame.
    pub const fn committed_batch(&self) -> Option<NonZeroUsize> {
        self.committed_batch
    }

    /// Derives the exact payload contract after scheduler batch admission.
    pub fn payload_contract(
        &self,
        ingress: &RealtimeIngressContract,
    ) -> Result<RealtimePayloadContract, RealtimeSessionExecutionError> {
        if ingress.schedule() != self.model.schedule() {
            return Err(RealtimeSessionExecutionError::IngressScheduleMismatch);
        }
        let batch = self
            .committed_batch
            .ok_or(RealtimeSessionExecutionError::BatchNotAdmitted)?;
        Ok(RealtimePayloadContract::new(
            ingress.schedule().clone(),
            batch.get(),
            ingress.text_domain(),
            ingress.audio_domain(),
            RealtimePayloadGeneration::new(self.history_generation.value())
                .expect("scheduler history generations are nonzero"),
            RealtimePayloadOwnerIdentity::new(self.incarnation.value())
                .expect("scheduler session incarnations are nonzero"),
        )
        .expect("scheduler-admitted payload contract has a positive batch"))
    }

    /// Unpublished generation branch passed to the injected executor.
    pub fn generation_mut(&mut self) -> &mut RealtimeGenerationBranch<MB, S, R, C> {
        &mut self.generation
    }

    fn admit_batch(&mut self, batch: usize) -> Result<(), RealtimeSessionExecutionError> {
        let batch = NonZeroUsize::new(batch).ok_or(RealtimeSessionExecutionError::EmptyBatch)?;
        match self.committed_batch {
            Some(committed) if committed != batch => Err(RealtimeSessionExecutionError::Batch {
                committed: committed.get(),
                submitted: batch.get(),
            }),
            Some(_) => Ok(()),
            None => {
                self.committed_batch = Some(batch);
                Ok(())
            }
        }
    }
}

impl<M, S, R, C> SemanticStateTransaction for RealtimeSessionState<M, S, R, C>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    S: Clone,
    R: Clone,
    C: Completion,
{
    type Branch = RealtimeSessionBranch<M::Branch, S, R, C>;
    type Error = RealtimeSessionTransactionError<M::Error, C::Error>;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(RealtimeSessionBranch {
            model: self.model.clone(),
            owner: self.owner.clone(),
            incarnation: self.incarnation,
            history_generation: self.history_generation,
            committed_batch: self.committed_batch,
            generation: self
                .generation
                .branch()
                .map_err(RealtimeSessionTransactionError::Generation)?,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        if self.model != branch.model
            || self.owner != branch.owner
            || self.incarnation != branch.incarnation
            || self.history_generation != branch.history_generation
        {
            Self::discard_branch(branch)?;
            return Err(RealtimeSessionTransactionError::IdentityMismatch);
        }
        if let (Some(committed), Some(submitted)) = (self.committed_batch, branch.committed_batch) {
            if committed != submitted {
                Self::discard_branch(branch)?;
                return Err(RealtimeSessionTransactionError::BatchMismatch);
            }
        }
        self.generation
            .commit_branch(branch.generation)
            .map_err(RealtimeSessionTransactionError::Generation)?;
        self.committed_batch = branch.committed_batch;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        RealtimeGenerationState::<M, S, R, C>::discard_branch(branch.generation)
            .map_err(RealtimeSessionTransactionError::Generation)
    }

    fn permits_parallel_branches(&self) -> bool {
        false
    }
}

/// Released canonical state which can resume only under the exact model identity.
pub struct ReleasedRealtimeSession<M, S, R, C> {
    state: RealtimeSessionState<M, S, R, C>,
}

impl<M, S, R, C> ReleasedRealtimeSession<M, S, R, C> {
    /// Exact selected model identity required for resumption.
    pub const fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        self.state.model_identity()
    }

    /// Preserved session incarnation.
    pub const fn incarnation(&self) -> RealtimeSessionIncarnation {
        self.state.incarnation()
    }

    /// Batch committed by the first successful frame, if any.
    pub const fn committed_batch(&self) -> Option<NonZeroUsize> {
        self.state.committed_batch()
    }
}

/// Singular fair scheduler for one exact selected realtime model.
pub struct RealtimeSessionScheduler<M, S, R, C, O>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    S: Clone,
    R: Clone,
    C: Completion,
    O: TransitionOutput,
{
    model: RealtimeModelSessionIdentity,
    scheduler: Scheduler<RealtimeInputFrame, RealtimeSessionState<M, S, R, C>, O>,
}

impl<M, S, R, C, O> RealtimeSessionScheduler<M, S, R, C, O>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    S: Clone,
    R: Clone,
    C: Completion,
    O: TransitionOutput,
{
    /// Exact selected model identity shared by every admitted request.
    pub const fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Creates one scheduler for both single-request and concurrent production use.
    pub fn new(
        model: RealtimeModelSessionIdentity,
        limits: SchedulerLimits,
    ) -> Result<Self, SchedulerError> {
        Ok(Self {
            model,
            scheduler: Scheduler::new(limits)?,
        })
    }

    /// Registers a new canonical session with a fresh monotonic incarnation.
    pub fn register(
        &mut self,
        request: RequestId,
        generation: RealtimeGenerationState<M, S, R, C>,
    ) -> Result<RealtimeSessionIncarnation, RealtimeSessionError> {
        if generation.schedule_state().schedule() != self.model.schedule() {
            return Err(RealtimeSessionError::ScheduleMismatch);
        }
        self.scheduler.validate_registration(request)?;
        let incarnation = allocate_incarnation()?;
        let state = RealtimeSessionState {
            model: self.model.clone(),
            owner: self.model.model_owner(),
            incarnation,
            history_generation: RealtimeHistoryGeneration(incarnation.0),
            committed_batch: None,
            generation,
        };
        self.scheduler.register(request, state)?;
        Ok(incarnation)
    }

    /// Resumes released state only under the exact selected model identity.
    pub fn resume(
        &mut self,
        request: RequestId,
        released: ReleasedRealtimeSession<M, S, R, C>,
    ) -> Result<(), RealtimeSessionResumeError<M, S, R, C>> {
        if released.state.model != self.model {
            return Err(RealtimeSessionResumeError {
                reason: RealtimeSessionError::ModelIdentityMismatch,
                released: Box::new(released),
            });
        }
        if let Err(error) = self.scheduler.validate_registration(request) {
            return Err(RealtimeSessionResumeError {
                reason: RealtimeSessionError::Scheduler(error),
                released: Box::new(released),
            });
        }
        self.scheduler
            .register(request, released.state)
            .expect("prevalidated realtime resumption cannot fail registration");
        Ok(())
    }

    /// Enqueues one portable frame on the singular fair path.
    pub fn enqueue(
        &mut self,
        request: RequestId,
        frame: RealtimeInputFrame,
    ) -> Result<WorkId, SchedulerError> {
        self.scheduler.enqueue(request, frame)
    }

    /// Enqueues one portable frame with an absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        request: RequestId,
        frame: RealtimeInputFrame,
        deadline: Option<Instant>,
    ) -> Result<WorkId, SchedulerError> {
        self.scheduler
            .enqueue_with_deadline(request, frame, deadline)
    }

    /// Atomically enqueues ordered frames on one request.
    pub fn enqueue_batch(
        &mut self,
        request: RequestId,
        frames: Vec<RealtimeInputFrame>,
    ) -> Result<Vec<WorkId>, SchedulerError> {
        self.scheduler.enqueue_batch(request, frames)
    }

    /// Runs one fair local turn using an injected family/backend-independent submission closure.
    pub fn run_local_turn<E>(
        &mut self,
        now: Instant,
        mut execute: impl FnMut(
            WorkId,
            &RealtimeInputFrame,
            &mut RealtimeSessionBranch<M::Branch, S, R, C>,
        ) -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame, O>, SchedulerError>
    where
        E: std::error::Error,
    {
        self.ensure_local_topology()?;
        self.scheduler.run_local_turn(now, |id, frame, branch| {
            branch
                .admit_batch(frame.batch())
                .map_err(RealtimeSessionSubmissionError::<E>::Session)?;
            execute(id, frame, branch).map_err(RealtimeSessionSubmissionError::Execution)
        })
    }

    /// Runs one local turn while admitting at most `maximum_frames` new transitions.
    pub fn run_local_bounded<E>(
        &mut self,
        now: Instant,
        maximum_frames: usize,
        mut execute: impl FnMut(
            WorkId,
            &RealtimeInputFrame,
            &mut RealtimeSessionBranch<M::Branch, S, R, C>,
        ) -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame, O>, SchedulerError>
    where
        E: std::error::Error,
    {
        self.ensure_local_topology()?;
        let mut progress = self.scheduler.poll_completions(now);
        self.scheduler.prepare_bounded(maximum_frames, now)?;
        progress.newly_submitted = self.scheduler.submit_prepared(now, |id, frame, branch| {
            branch
                .admit_batch(frame.batch())
                .map_err(RealtimeSessionSubmissionError::<E>::Session)?;
            execute(id, frame, branch).map_err(RealtimeSessionSubmissionError::Execution)
        })?;
        let completed = self.scheduler.poll_completions(now);
        progress.committed.extend(completed.committed);
        progress.failed.extend(completed.failed);
        Ok(progress)
    }

    /// Runs one fair turn with mandatory topology-wide schedule and completion consensus.
    pub fn run_distributed_turn<T, E>(
        &mut self,
        protocol: u64,
        transport: &T,
        now: Instant,
        mut execute: impl FnMut(
            WorkId,
            &RealtimeInputFrame,
            &mut RealtimeSessionBranch<M::Branch, S, R, C>,
        ) -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame, O>, SchedulerError>
    where
        T: BoundedConsensusTransport,
        <T::Completion as Completion>::Error: std::fmt::Display,
        E: std::error::Error,
        O: eredu_core::scheduler::DistributedTransitionOutput,
    {
        if self.model.topology().is_replicated() {
            return Err(SchedulerError::Consensus(
                "distributed realtime turns require a non-replicated selected topology".into(),
            ));
        }
        let participants = self.model.topology().world_size();
        if transport.participant_count() != participants {
            return Err(SchedulerError::Consensus(format!(
                "distributed realtime transport has {} participants; selected topology requires {participants}",
                transport.participant_count(),
            )));
        }
        let wait: BoundedCompletionWait = self.model.completion().bounded_wait();
        validate_ranked_identity_bounded(
            transport,
            protocol,
            &self.model.distributed_consensus_identity(),
            self.model.rank,
            wait,
        )
        .map_err(|error| SchedulerError::Consensus(error.to_string()))?;
        self.scheduler
            .run_distributed_turn(protocol, transport, wait, now, |id, frame, branch| {
                branch
                    .admit_batch(frame.batch())
                    .map_err(RealtimeSessionSubmissionError::<E>::Session)?;
                execute(id, frame, branch).map_err(RealtimeSessionSubmissionError::Execution)
            })
    }

    fn ensure_local_topology(&self) -> Result<(), SchedulerError> {
        if self.model.topology().is_replicated() {
            Ok(())
        } else {
            Err(SchedulerError::Consensus(
                "rank-local realtime turns are forbidden for a non-replicated selected topology"
                    .into(),
            ))
        }
    }

    /// Atomically replaces sampling only when the request owns no queued or branched work.
    pub fn replace_sampling<E>(
        &mut self,
        request: RequestId,
        sampling: RealtimeSampling,
        realize: impl FnOnce(RealtimeSampling) -> Result<(Vec<S>, Option<R>), E>,
    ) -> Result<(), RealtimeSamplingUpdateError<E, M::Error, C::Error>> {
        if self.scheduler.queued_for_request(request) != 0 {
            return Err(RealtimeSamplingReplacementError::QueuedWork);
        }
        let state = self
            .scheduler
            .request_state_mut(request)
            .map_err(RealtimeSamplingReplacementError::Scheduler)?;
        let (samplers, random) =
            realize(sampling).map_err(RealtimeSamplingReplacementError::Realization)?;
        state
            .generation_mut()
            .replace_sampling(sampling, samplers, random)
            .map_err(RealtimeSamplingReplacementError::Generation)
    }

    /// Cancels queued/prepared/submitted work using core scheduler semantics.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        self.scheduler.cancel(request)
    }

    /// Marks a request finished using core scheduler semantics.
    pub fn finish(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        self.scheduler.finish(request)
    }

    /// Releases an idle canonical session for exact later resumption.
    pub fn release(
        &mut self,
        request: RequestId,
    ) -> Result<ReleasedRealtimeSession<M, S, R, C>, SchedulerError> {
        self.scheduler
            .release(request)
            .map(|state| ReleasedRealtimeSession { state })
    }

    /// Active or terminal request status.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.scheduler.request_status(request)
    }

    /// Number of portable frames still queued for one active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.scheduler.queued_for_request(request)
    }

    /// Removes a terminal identity so the caller may explicitly reuse it.
    pub fn forget_terminal(&mut self, request: RequestId) -> Result<RequestStatus, SchedulerError> {
        self.scheduler.forget_terminal(request)
    }

    /// Immutable canonical session state, when active.
    pub fn request_state(&self, request: RequestId) -> Option<&RealtimeSessionState<M, S, R, C>> {
        self.scheduler.request_state(request)
    }

    /// Current scheduler telemetry.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }

    /// Configured and observed scheduler capabilities.
    pub fn capabilities(&self) -> SchedulerCapabilities {
        self.scheduler.capabilities()
    }
}

fn allocate_incarnation() -> Result<RealtimeSessionIncarnation, RealtimeSessionError> {
    NEXT_REALTIME_INCARNATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map(RealtimeSessionIncarnation)
        .map_err(|_| RealtimeSessionError::IncarnationExhausted)
}

/// Stable session ownership failure before scheduler submission.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeSessionError {
    /// Core scheduler lifecycle failure.
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    /// Generation schedule differs from the exact selected model schedule.
    #[error("realtime generation schedule differs from selected model identity")]
    ScheduleMismatch,
    /// Released state belongs to another exact selected model.
    #[error("released realtime session model identity does not match")]
    ModelIdentityMismatch,
    /// Process-local monotonic incarnation space is exhausted.
    #[error("realtime session incarnation identity space is exhausted")]
    IncarnationExhausted,
}

/// Batch admission failure before an injected frame executor is called.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeSessionExecutionError {
    /// Payload-contract derivation was attempted before scheduler batch admission.
    #[error("realtime session frame batch has not been admitted")]
    BatchNotAdmitted,
    /// Portable frames must have a positive batch.
    #[error("realtime session frame batch must be positive")]
    EmptyBatch,
    /// Every frame in an incarnation must use the first committed batch.
    #[error("realtime session batch is {submitted}, committed batch is {committed}")]
    Batch {
        /// Batch already committed by this incarnation.
        committed: usize,
        /// Batch carried by the rejected frame.
        submitted: usize,
    },
    /// Frame ingress schedule differs from the exact selected session schedule.
    #[error("realtime ingress schedule differs from selected session schedule")]
    IngressScheduleMismatch,
}

/// Failed resumption retaining full released state for retry or disposal.
pub struct RealtimeSessionResumeError<M, S, R, C> {
    reason: RealtimeSessionError,
    released: Box<ReleasedRealtimeSession<M, S, R, C>>,
}

impl<M, S, R, C> RealtimeSessionResumeError<M, S, R, C> {
    /// Stable reason resumption was rejected.
    pub const fn reason(&self) -> &RealtimeSessionError {
        &self.reason
    }

    /// Recovers the unchanged released state.
    pub fn into_released(self) -> ReleasedRealtimeSession<M, S, R, C> {
        *self.released
    }
}

impl<M, S, R, C> std::fmt::Debug for RealtimeSessionResumeError<M, S, R, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealtimeSessionResumeError")
            .field("reason", &self.reason)
            .finish_non_exhaustive()
    }
}

impl<M, S, R, C> std::fmt::Display for RealtimeSessionResumeError<M, S, R, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.reason.fmt(formatter)
    }
}

impl<M, S, R, C> std::error::Error for RealtimeSessionResumeError<M, S, R, C> {}

#[derive(Debug, thiserror::Error)]
enum RealtimeSessionSubmissionError<E: std::error::Error> {
    #[error(transparent)]
    Session(RealtimeSessionExecutionError),
    #[error(transparent)]
    Execution(E),
}

/// Failure while publishing or discarding complete session state.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeSessionTransactionError<M, C>
where
    M: std::error::Error,
    C: std::error::Error,
{
    /// Atomic generation-state transaction failed.
    #[error(transparent)]
    Generation(RealtimeGenerationTransactionError<M, C>),
    /// An unpublished branch did not retain its exact session identity.
    #[error("realtime session branch identity does not match canonical state")]
    IdentityMismatch,
    /// An unpublished branch attempted to change committed batch.
    #[error("realtime session branch batch does not match canonical state")]
    BatchMismatch,
}

/// Sampling replacement failure while preserving queued-work ordering.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeSamplingReplacementError<E, G> {
    /// Queued frames retain the sampling policy under which they were accepted.
    #[error("cannot replace realtime sampling while frames are queued")]
    QueuedWork,
    /// Core scheduler did not admit mutable idle-state access.
    #[error(transparent)]
    Scheduler(SchedulerError),
    /// Backend-neutral sampler/RNG realization failed.
    #[error("realtime sampling realization failed")]
    Realization(E),
    /// New sampler/RNG state did not match generation geometry.
    #[error("realtime generation rejected replacement sampling")]
    Generation(G),
}

/// Sampling replacement failure specialized to one generation transaction.
pub type RealtimeSamplingUpdateError<E, M, C> =
    RealtimeSamplingReplacementError<E, RealtimeGenerationTransactionError<M, C>>;

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible, rc::Rc, time::Duration};

    use eredu_core::{
        consensus::{BoundedConsensusTransport, ConsensusTransport},
        scheduler::{RequestStatus, TransitionOutput},
        BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
        CompletionCancellationMode, RealtimeFrameConvention, Submission,
    };

    use super::*;
    use crate::TokenDomain;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct ModelState(usize);

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("model transaction failed")]
    struct ModelError;

    impl SemanticStateTransaction for ModelState {
        type Branch = Self;
        type Error = ModelError;

        fn branch(&self) -> Result<Self::Branch, Self::Error> {
            Ok(self.clone())
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
            *self = branch;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct MockCompletion(Rc<Cell<bool>>);

    #[derive(Debug, Clone, Copy, thiserror::Error)]
    #[error("completion failed")]
    struct CompletionError;

    impl Completion for MockCompletion {
        type Error = CompletionError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(self.0.get())
        }

        fn wait(&self) -> Result<(), Self::Error> {
            self.0.get().then_some(()).ok_or(CompletionError)
        }
    }

    #[derive(Debug)]
    struct MockOutput {
        completion: MockCompletion,
    }

    impl TransitionOutput for MockOutput {
        type Error = CompletionError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            self.completion.is_complete()
        }

        fn retained_resources(&self) -> usize {
            1
        }
    }

    impl eredu_core::scheduler::DistributedTransitionOutput for MockOutput {
        fn encode_distributed_output(&self, output: &mut Vec<u32>) -> Result<(), String> {
            output.push(0);
            Ok(())
        }
    }

    type Sessions = RealtimeSessionScheduler<ModelState, (), (), MockCompletion, MockOutput>;

    fn schedule() -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            2,
            1,
            1,
            1,
            9,
            8,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, 0, 1],
        )
        .unwrap()
    }

    fn identity(suffix: &str) -> RealtimeModelSessionIdentity {
        identity_with_topology(suffix, ParallelTopology::new(1, 1, 1, 1).unwrap())
    }

    fn identity_with_topology(
        suffix: &str,
        topology: ParallelTopology,
    ) -> RealtimeModelSessionIdentity {
        let id = |prefix: &str| RealtimeIdentity::new(format!("{prefix}-{suffix}")).unwrap();
        RealtimeModelSessionIdentity::from_parts(
            id("architecture"),
            id("source"),
            id("execution"),
            id("schedule"),
            schedule(),
            id("state"),
            id("topology"),
            topology,
            0,
            LayerWeightResidency::FullyResident,
            CommunicationCompletionPolicy::new(
                Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        )
    }

    #[derive(Debug, Clone, Copy)]
    struct ReadyConsensusCompletion;

    impl Completion for ReadyConsensusCompletion {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl BoundedCompletion for ReadyConsensusCompletion {
        fn wait_bounded(
            self,
            _: BoundedCompletionWait,
        ) -> Result<BoundedCompletionOutcome, Self::Error> {
            Ok(BoundedCompletionOutcome::Completed)
        }
    }

    struct DisagreeingTransport {
        calls: Cell<usize>,
        disagree_on: usize,
    }

    impl DisagreeingTransport {
        fn model_identity() -> Self {
            Self {
                calls: Cell::new(0),
                disagree_on: 1,
            }
        }

        fn schedule() -> Self {
            Self {
                calls: Cell::new(0),
                disagree_on: 4,
            }
        }
    }

    impl ConsensusTransport for DisagreeingTransport {
        type Error = Infallible;

        fn participant_count(&self) -> usize {
            2
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            let mut gathered = local.repeat(2);
            if call == 1 {
                *gathered.last_mut().expect("identity frame has a rank word") = 1;
            }
            if call == self.disagree_on {
                gathered[local.len()] ^= 1;
            }
            Ok(gathered)
        }
    }

    impl BoundedConsensusTransport for DisagreeingTransport {
        type Completion = ReadyConsensusCompletion;
        type GatherOutput = Vec<u32>;

        fn submit_all_gather_words(
            &self,
            local: &[u32],
        ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Self::Error> {
            Ok(Submission {
                output: self.all_gather_words(local)?,
                completion: ReadyConsensusCompletion,
            })
        }

        fn resolve_all_gather_words(
            &self,
            output: Self::GatherOutput,
        ) -> Result<Vec<u32>, Self::Error> {
            Ok(output)
        }
    }

    fn generation() -> RealtimeGenerationState<ModelState, (), (), MockCompletion> {
        RealtimeGenerationState::new(
            ModelState(0),
            schedule(),
            RealtimeSampling::greedy(),
            vec![(), ()],
            None,
        )
        .unwrap()
    }

    fn ingress() -> RealtimeIngressContract {
        RealtimeIngressContract::new(schedule(), TokenDomain::new(32), TokenDomain::new(16))
            .unwrap()
    }

    fn frame(batch: usize) -> RealtimeInputFrame {
        RealtimeInputFrame::new(batch, vec![1; batch])
    }

    fn limits(submissions: usize) -> SchedulerLimits {
        SchedulerLimits::with_execution_bounds(8, 32, submissions, 8, 1, usize::MAX).unwrap()
    }

    fn execute_immediately(
        _id: WorkId,
        _frame: &RealtimeInputFrame,
        branch: &mut RealtimeSessionBranch<ModelState, (), (), MockCompletion>,
    ) -> Result<MockOutput, Infallible> {
        branch.generation_mut().model_state_mut().0 += 1;
        let completion = MockCompletion(Rc::new(Cell::new(true)));
        branch
            .generation_mut()
            .attach_submission_completion(completion.clone())
            .unwrap();
        Ok(MockOutput { completion })
    }

    #[test]
    fn one_scheduler_round_robins_single_lane_sessions() {
        let mut sessions = Sessions::new(identity("a"), limits(2)).unwrap();
        let first = RequestId::new(1);
        let second = RequestId::new(2);
        let first_incarnation = sessions.register(first, generation()).unwrap();
        let second_incarnation = sessions.register(second, generation()).unwrap();
        assert_ne!(first_incarnation, second_incarnation);
        sessions
            .enqueue_batch(first, vec![frame(1), frame(1)])
            .unwrap();
        sessions
            .enqueue_batch(second, vec![frame(1), frame(1)])
            .unwrap();

        let progress = sessions
            .run_local_turn(Instant::now(), execute_immediately)
            .unwrap();
        assert_eq!(progress.committed.len(), 2);
        assert_eq!(progress.committed[0].0.request(), first);
        assert_eq!(progress.committed[1].0.request(), second);
        assert_eq!(
            sessions
                .request_state(first)
                .unwrap()
                .generation()
                .model_state()
                .0,
            1
        );
        assert_eq!(
            sessions
                .request_state(second)
                .unwrap()
                .generation()
                .model_state()
                .0,
            1
        );
    }

    #[test]
    fn non_replicated_identity_rejects_local_turn_before_submission() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut sessions =
            Sessions::new(identity_with_topology("tp", topology), limits(1)).unwrap();
        let request = RequestId::new(90);
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        let error = sessions
            .run_local_turn(Instant::now(), move |id, frame, branch| {
                observed.set(observed.get() + 1);
                execute_immediately(id, frame, branch)
            })
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert_eq!(calls.get(), 0);
        assert_eq!(sessions.queued_for_request(request), 1);

        let error = sessions
            .run_local_bounded(Instant::now(), 1, |id, frame, branch| {
                calls.set(calls.get() + 1);
                execute_immediately(id, frame, branch)
            })
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert_eq!(calls.get(), 0);
        assert_eq!(sessions.queued_for_request(request), 1);
    }

    #[test]
    fn distributed_model_identity_disagreement_submits_and_publishes_nothing() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut sessions = Sessions::new(
            identity_with_topology("tp-disagreement", topology),
            limits(1),
        )
        .unwrap();
        let request = RequestId::new(91);
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        let error = sessions
            .run_distributed_turn(
                17,
                &DisagreeingTransport::model_identity(),
                Instant::now(),
                move |id, frame, branch| {
                    observed.set(observed.get() + 1);
                    execute_immediately(id, frame, branch)
                },
            )
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert_eq!(calls.get(), 0);
        assert_eq!(
            sessions
                .request_state(request)
                .unwrap()
                .generation()
                .model_state()
                .0,
            0
        );
        assert_eq!(sessions.report().completed_work, 0);
    }

    #[test]
    fn distributed_schedule_disagreement_submits_and_publishes_nothing() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut sessions =
            Sessions::new(identity_with_topology("tp-schedule", topology), limits(1)).unwrap();
        let request = RequestId::new(92);
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        let error = sessions
            .run_distributed_turn(
                18,
                &DisagreeingTransport::schedule(),
                Instant::now(),
                move |id, frame, branch| {
                    observed.set(observed.get() + 1);
                    execute_immediately(id, frame, branch)
                },
            )
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert_eq!(calls.get(), 0);
        assert_eq!(sessions.report().completed_work, 0);
    }

    #[test]
    fn scheduler_branch_derives_payload_identity_before_native_execution() {
        let mut sessions = Sessions::new(identity("payload"), limits(1)).unwrap();
        let request = RequestId::new(20);
        let incarnation = sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(2)).unwrap();

        sessions
            .run_local_turn(Instant::now(), |id, submitted, branch| {
                let contract = branch.payload_contract(&ingress()).unwrap();
                assert_eq!(contract.batch().get(), 2);
                assert_eq!(contract.owner().value(), incarnation.value());
                assert_eq!(
                    contract.generation().value(),
                    branch.history_generation().value()
                );
                execute_immediately(id, submitted, branch)
            })
            .unwrap();
    }

    #[test]
    fn committed_batch_rejects_a_different_batch_before_execution() {
        let mut sessions = Sessions::new(identity("a"), limits(1)).unwrap();
        let request = RequestId::new(3);
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        sessions
            .run_local_turn(Instant::now(), execute_immediately)
            .unwrap();
        assert_eq!(
            sessions
                .request_state(request)
                .unwrap()
                .committed_batch()
                .unwrap()
                .get(),
            1
        );

        sessions.enqueue(request, frame(2)).unwrap();
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        assert!(sessions
            .run_local_turn(Instant::now(), move |id, frame, branch| {
                observed.set(observed.get() + 1);
                execute_immediately(id, frame, branch)
            })
            .is_err());
        assert_eq!(calls.get(), 0);
        assert_eq!(
            sessions.request_status(request),
            Some(RequestStatus::Failed)
        );
    }

    #[test]
    fn release_resume_preserves_incarnation_and_wrong_model_keeps_state_recoverable() {
        let request = RequestId::new(4);
        let mut first = Sessions::new(identity("a"), limits(1)).unwrap();
        let incarnation = first.register(request, generation()).unwrap();
        let released = first.release(request).unwrap();
        assert_eq!(released.incarnation(), incarnation);

        let mut wrong = Sessions::new(identity("b"), limits(1)).unwrap();
        let error = wrong.resume(request, released).unwrap_err();
        assert!(matches!(
            error.reason(),
            RealtimeSessionError::ModelIdentityMismatch
        ));
        let released = error.into_released();
        first.resume(request, released).unwrap();
        assert_eq!(
            first.request_state(request).unwrap().incarnation(),
            incarnation
        );
    }

    #[test]
    fn queued_work_blocks_sampling_replacement_before_realization() {
        let request = RequestId::new(5);
        let mut sessions = Sessions::new(identity("a"), limits(1)).unwrap();
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let realized = Rc::new(Cell::new(false));
        let observed = realized.clone();
        let result = sessions.replace_sampling(
            request,
            RealtimeSampling::greedy(),
            move |_| -> Result<_, Infallible> {
                observed.set(true);
                Ok((vec![(), ()], None))
            },
        );
        assert!(matches!(
            result,
            Err(RealtimeSamplingReplacementError::QueuedWork)
        ));
        assert!(!realized.get());
    }

    #[test]
    fn cancellation_and_deadline_use_core_terminal_semantics() {
        let mut sessions = Sessions::new(identity("a"), limits(1)).unwrap();
        let cancelled = RequestId::new(6);
        sessions.register(cancelled, generation()).unwrap();
        sessions.enqueue(cancelled, frame(1)).unwrap();
        sessions.cancel(cancelled).unwrap();
        assert_eq!(
            sessions.request_status(cancelled),
            Some(RequestStatus::Cancelled)
        );

        let expired = RequestId::new(7);
        sessions.register(expired, generation()).unwrap();
        sessions
            .enqueue_with_deadline(
                expired,
                frame(1),
                Some(Instant::now() - Duration::from_secs(1)),
            )
            .unwrap();
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        sessions
            .run_local_turn(Instant::now(), move |id, frame, branch| {
                observed.set(observed.get() + 1);
                execute_immediately(id, frame, branch)
            })
            .unwrap();
        assert_eq!(calls.get(), 0);
        assert_eq!(
            sessions.request_status(expired),
            Some(RequestStatus::DeadlineExceeded)
        );
    }

    #[test]
    fn cancelling_submitted_work_retains_it_until_exact_completion() {
        let request = RequestId::new(8);
        let mut sessions = Sessions::new(identity("a"), limits(1)).unwrap();
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let completion = Rc::new(Cell::new(false));
        let submitted = completion.clone();
        sessions
            .run_local_turn(Instant::now(), move |_, _, branch| {
                let completion = MockCompletion(submitted.clone());
                branch
                    .generation_mut()
                    .attach_submission_completion(completion.clone())
                    .unwrap();
                Ok::<_, Infallible>(MockOutput { completion })
            })
            .unwrap();
        sessions.cancel(request).unwrap();
        assert_eq!(sessions.report().abandoned_in_flight_work, 1);
        completion.set(true);
        sessions
            .run_local_turn(Instant::now(), execute_immediately)
            .unwrap();
        assert_eq!(sessions.report().abandoned_in_flight_work, 0);
    }
}
