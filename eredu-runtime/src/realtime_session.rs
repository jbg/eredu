//! Backend-neutral realtime session ownership over the singular fair scheduler.

use eredu_core::BackendFailure;
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
pub struct RealtimeSessionState<M, S, R, C, P = ()> {
    model: std::sync::Arc<RealtimeModelSessionIdentity>,
    owner: std::sync::Arc<RealtimeModelOwnerIdentity>,
    incarnation: RealtimeSessionIncarnation,
    history_generation: RealtimeHistoryGeneration,
    committed_batch: Option<NonZeroUsize>,
    generation: RealtimeGenerationState<M, S, R, C>,
    preparation:std::marker::PhantomData<fn()->P>,
}

impl<M, S, R, C, P> RealtimeSessionState<M, S, R, C, P> {
    /// Exact selected model identity.
    pub fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Typed model owner derived from the selected contract.
    pub fn model_owner(&self) -> &RealtimeModelOwnerIdentity {
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

    /// Derives the first or later frame contract from canonical scheduler
    /// identity before any state branch or native input construction.
    pub fn payload_contract_for_frame(&self,ingress:&RealtimeIngressContract,batch:usize,
        funding:&eredu_core::HostMetadataFunding)->Result<RealtimePayloadContract,RealtimeSessionExecutionError> {
        let occurrence=self.frame_occurrence_for(batch)?;
        if ingress.schedule()!=self.model.schedule() {
            return Err(RealtimeSessionExecutionError::IngressScheduleMismatch);
        }
        funding.reserve_metadata(std::mem::size_of::<RealtimePayloadContract>()
            +std::mem::size_of::<Result<RealtimePayloadContract,RealtimeSessionExecutionError>>())?;
        let schedule=ingress.schedule().try_clone_with_host_source(funding)?;
        Ok(RealtimePayloadContract::new(schedule,occurrence.batch().get(),
            ingress.text_domain(),ingress.audio_domain(),
            RealtimePayloadGeneration::new(occurrence.history_generation().value()).expect("nonzero scheduler history"),
            RealtimePayloadOwnerIdentity::new(occurrence.incarnation().value()).expect("nonzero scheduler incarnation"))
            .expect("validated ingress domains and positive frame batch"))
    }

    /// Borrows the actual pre-branch frame occurrence after positive batch
    /// validation. This identity grants no source construction or allocation.
    pub fn frame_occurrence_for(&self,batch:usize)->Result<RealtimeFrameOccurrence,RealtimeSessionExecutionError> {
        let batch=NonZeroUsize::new(batch).ok_or(RealtimeSessionExecutionError::EmptyBatch)?;
        if let Some(committed)=self.committed_batch {
            if committed!=batch {return Err(RealtimeSessionExecutionError::Batch {
                committed:committed.get(),submitted:batch.get()});}
        }
        Ok(RealtimeFrameOccurrence{incarnation:self.incarnation,history:self.history_generation,
            frontier:self.generation.schedule_state().frontier(),batch})
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

/// Actual session/frame coordinates from canonical scheduler state or its
/// unpublished branch. This descriptor contains no allocation or submission authority.
#[derive(Debug,Clone,Copy,Eq,PartialEq)]
pub struct RealtimeFrameOccurrence {
    incarnation:RealtimeSessionIncarnation,
    history:RealtimeHistoryGeneration,
    frontier:usize,
    batch:NonZeroUsize,
}
impl RealtimeFrameOccurrence {
    /// Actual session incarnation, preserved by released state.
    pub const fn incarnation(self)->RealtimeSessionIncarnation {self.incarnation}
    /// Actual history generation, distinct across session restoration policies.
    pub const fn history_generation(self)->RealtimeHistoryGeneration {self.history}
    /// Next input frame of the actual canonical schedule.
    pub const fn frontier(self)->usize {self.frontier}
    /// Batch already admitted by the scheduler for this frame.
    pub const fn batch(self)->NonZeroUsize {self.batch}
}

/// Unpublished realtime session branch owned by the core scheduler.
pub struct RealtimeSessionBranch<MB, S, R, C, P = ()> {
    model: std::sync::Arc<RealtimeModelSessionIdentity>,
    owner: std::sync::Arc<RealtimeModelOwnerIdentity>,
    incarnation: RealtimeSessionIncarnation,
    history_generation: RealtimeHistoryGeneration,
    committed_batch: Option<NonZeroUsize>,
    generation: RealtimeGenerationBranch<MB, S, R, C>,
    // Last field: source/account custody outlives every branch-owned payload.
    preparation:Option<P>,
    preparation_issued:bool,
}

impl<MB, S, R, C, P> RealtimeSessionBranch<MB, S, R, C, P> {
    /// Installs one move-only source/plan after admission and before scheduler
    /// submission. Rejection returns the unchanged source to its owner.
    pub fn install_preparation(&mut self,source:P)->Result<(),P> {
        if self.preparation_issued {return Err(source);}
        self.preparation_issued=true;self.preparation=Some(source);Ok(())
    }
    /// Borrows the exact preparation retained by this unpublished branch.
    pub fn preparation(&self)->Option<&P> {self.preparation.as_ref()}
    /// Consumes this branch's preparation exactly once for native activation.
    pub fn take_preparation(&mut self)->Option<P> {self.preparation.take()}

    /// Exact selected model identity.
    pub fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Typed model owner used by payload contracts.
    pub fn model_owner(&self) -> &RealtimeModelOwnerIdentity {
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

    /// Borrows the scheduler's actual frame coordinates before owned preparation.
    pub fn frame_occurrence(&self)->Result<RealtimeFrameOccurrence,RealtimeSessionExecutionError> {
        Ok(RealtimeFrameOccurrence{incarnation:self.incarnation,history:self.history_generation,
            frontier:self.generation.schedule_state().frontier(),
            batch:self.committed_batch.ok_or(RealtimeSessionExecutionError::BatchNotAdmitted)?})
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

impl<M,S,R,C,P> RealtimeSessionState<M,S,R,C,P>
where M:SemanticStateTransaction,M::Error:Send+Sync+'static,C:Completion {
    /// Actual session/generation branch shells and owned schedule/directory.
    /// Child payload, model, sampler and random producers are counted separately.
    pub fn branch_host_bytes<F,G,H>(&self)->Option<usize> {
        Self::branch_shell_bytes::<F,G,H>()?.checked_add(self.generation.branch_host_bytes::<F,G,H>()?)
    }
    fn branch_shell_bytes<F,G,H>()->Option<usize> {
        let parts=[std::mem::size_of::<F>(),std::mem::size_of::<G>(),std::mem::size_of::<H>(),
            std::mem::size_of::<RealtimeSessionBranch<M::Branch,S,R,C,P>>(),
            std::mem::size_of::<Result<RealtimeSessionBranch<M::Branch,S,R,C,P>,BackendFailure>>(),
            eredu_core::HostMetadataFunding::reservation_control_bytes()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
    }

    /// Creates the scheduler branch from explicit paid child sources while
    /// sharing the canonical immutable selected identity.
    pub fn branch_with_host_source<F,G,H>(&self,funding:&eredu_core::HostMetadataFunding,
        branch_model:F,clone_sampler:G,clone_random:H)
        ->Result<RealtimeSessionBranch<M::Branch,S,R,C,P>,BackendFailure>
    where F:FnOnce(&M,&eredu_core::HostMetadataFunding)->Result<M::Branch,BackendFailure>,
        G:FnMut(&S,&eredu_core::HostMetadataFunding)->Result<S,BackendFailure>,
        H:FnOnce(&R,&eredu_core::HostMetadataFunding)->Result<R,BackendFailure> {
        funding.reserve_metadata(Self::branch_shell_bytes::<F,G,H>()
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?)?;
        let generation=self.generation.branch_with_host_source(funding,branch_model,clone_sampler,clone_random)?;
        Ok(RealtimeSessionBranch{model:self.model.clone(),owner:self.owner.clone(),
            incarnation:self.incarnation,history_generation:self.history_generation,
            committed_batch:self.committed_batch,generation,preparation:None,preparation_issued:false})
    }
}

impl<M, S, R, C, P> SemanticStateTransaction for RealtimeSessionState<M, S, R, C, P>
where
    M: SemanticStateTransaction,
    M::Error: Send + Sync + 'static,
    S: Clone,
    R: Clone,
    C: Completion,
{
    type Branch = RealtimeSessionBranch<M::Branch, S, R, C, P>;
    type Error = RealtimeSessionTransactionError;

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
            preparation:None,
            preparation_issued:false,
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
pub struct ReleasedRealtimeSession<M, S, R, C, P = ()> {
    state: RealtimeSessionState<M, S, R, C, P>,
}

impl<M, S, R, C, P> ReleasedRealtimeSession<M, S, R, C, P> {
    /// Exact selected model identity required for resumption.
    pub fn model_identity(&self) -> &RealtimeModelSessionIdentity {
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
pub struct RealtimeSessionScheduler<M, S, R, C, O, P = ()>
where
    M: SemanticStateTransaction,
    M::Error: Send + Sync + 'static,
    S: Clone,
    R: Clone,
    C: Completion,
    O: TransitionOutput,
{
    model: std::sync::Arc<RealtimeModelSessionIdentity>,
    scheduler: Scheduler<RealtimeInputFrame, RealtimeSessionState<M, S, R, C, P>, O>,
}

impl<M,S,R,C,O> RealtimeSessionScheduler<M,S,R,C,O,()>
where M:SemanticStateTransaction,M::Error:Send+Sync+'static,S:Clone,R:Clone,
    C:Completion,O:TransitionOutput {
    /// Creates the ordinary shared scheduler without a preparation payload.
    pub fn new(model:RealtimeModelSessionIdentity,limits:SchedulerLimits)->Result<Self,SchedulerError> {
        Self::new_with_preparation(model,limits)
    }
}

impl<M, S, R, C, O, P> RealtimeSessionScheduler<M, S, R, C, O, P>
where
    M: SemanticStateTransaction,
    M::Error: Send + Sync + 'static,
    S: Clone,
    R: Clone,
    C: Completion,
    O: TransitionOutput,
{
    /// Exact selected model identity shared by every admitted request.
    pub fn model_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.model
    }

    /// Creates a scheduler whose prepared branches retain the explicit P
    /// source type. The ordinary constructor continues to select P=().
    pub fn new_with_preparation(
        model: RealtimeModelSessionIdentity,
        limits: SchedulerLimits,
    ) -> Result<Self, SchedulerError> {
        Ok(Self {
            model:std::sync::Arc::new(model),
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
            owner: std::sync::Arc::new(self.model.model_owner()),
            incarnation,
            history_generation: RealtimeHistoryGeneration(incarnation.0),
            committed_batch: None,
            generation,
            preparation:std::marker::PhantomData,
        };
        self.scheduler.register(request, state)?;
        Ok(incarnation)
    }

    /// Resumes released state only under the exact selected model identity.
    /// The caller retains the slot unchanged on failure; success takes its state.
    pub fn resume(
        &mut self,
        request: RequestId,
        released: &mut Option<ReleasedRealtimeSession<M, S, R, C, P>>,
    ) -> Result<(), RealtimeSessionError> {
        let state = released
            .as_ref()
            .ok_or(RealtimeSessionError::MissingReleasedState)?;
        if state.state.model != self.model {
            return Err(RealtimeSessionError::ModelIdentityMismatch);
        }
        self.scheduler.validate_registration(request)?;
        self.scheduler
            .register(
                request,
                released.take().expect("validated released state").state,
            )
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
            &mut RealtimeSessionBranch<M::Branch, S, R, C, P>,
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
        execute: impl FnMut(
            WorkId,
            &RealtimeInputFrame,
            &mut RealtimeSessionBranch<M::Branch, S, R, C, P>,
        ) -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame, O>, SchedulerError>
    where
        E: std::error::Error,
    {
        self.run_local_bounded_with_preparation(now,maximum_frames,
            |_,_,state|state.branch(),execute)
    }

    /// Runs the existing local fair turn with source admission before any
    /// branch clone. A typed preparation payload travels inside the returned
    /// branch through queued, submitted, failed and retired ownership.
    pub fn run_local_bounded_with_preparation<A,E>(
        &mut self,now:Instant,maximum_frames:usize,
        prepare:impl FnMut(WorkId,&RealtimeInputFrame,&RealtimeSessionState<M,S,R,C,P>)
            ->Result<RealtimeSessionBranch<M::Branch,S,R,C,P>,A>,
        mut execute:impl FnMut(WorkId,&RealtimeInputFrame,&mut RealtimeSessionBranch<M::Branch,S,R,C,P>)
            ->Result<O,E>,
    )->Result<SchedulerProgress<RealtimeInputFrame,O>,SchedulerError>
    where A:std::error::Error,E:std::error::Error {
        self.ensure_local_topology()?;
        let mut progress = self.scheduler.poll_completions(now);
        self.scheduler.prepare_bounded_with(maximum_frames,now,prepare)?;
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
            &mut RealtimeSessionBranch<M::Branch, S, R, C, P>,
        ) -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame, O>, SchedulerError>
    where
        T: BoundedConsensusTransport,
        <T::Completion as Completion>::Error: std::fmt::Display,
        E: std::error::Error,
        O: eredu_core::scheduler::DistributedTransitionOutput,
    {
        let wait = self.distributed_turn_wait(protocol, transport)?;
        self.scheduler
            .run_distributed_turn(protocol, transport, wait, now, |id, frame, branch| {
                branch
                    .admit_batch(frame.batch())
                    .map_err(RealtimeSessionSubmissionError::<E>::Session)?;
                execute(id, frame, branch).map_err(RealtimeSessionSubmissionError::Execution)
            })
    }

    /// Runs the existing distributed turn with a source compiler before the
    /// actual state branch. Admission refusal is agreed before any model call;
    /// completion, publication, cancellation and epoch rules remain unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn run_distributed_bounded_with_preparation<T, A, E>(
        &mut self, protocol: u64, transport: &T, now: Instant, maximum_frames: usize,
        prepare: impl FnMut(WorkId, &RealtimeInputFrame, &RealtimeSessionState<M,S,R,C,P>)
            -> Result<RealtimeSessionBranch<M::Branch,S,R,C,P>, A>,
        execute: impl FnMut(WorkId, &RealtimeInputFrame, &mut RealtimeSessionBranch<M::Branch,S,R,C,P>)
            -> Result<O, E>,
    ) -> Result<SchedulerProgress<RealtimeInputFrame,O>, SchedulerError>
    where T: BoundedConsensusTransport,
        <T::Completion as Completion>::Error: std::fmt::Display,
        A: std::error::Error, E: std::error::Error,
        O: eredu_core::scheduler::DistributedTransitionOutput {
        self.run_distributed_bounded_with_preparation_and_errors(protocol,transport,now,
            maximum_frames,prepare,execute,|_,_|{})
    }

    /// Preserves each exact completion or host-observation source before the
    /// shared distributed protocol resolves its original frame output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_distributed_bounded_with_preparation_and_errors<T, A, E>(
        &mut self, protocol: u64, transport: &T, now: Instant, maximum_frames: usize,
        prepare: impl FnMut(WorkId, &RealtimeInputFrame, &RealtimeSessionState<M,S,R,C,P>)
            -> Result<RealtimeSessionBranch<M::Branch,S,R,C,P>, A>,
        mut execute: impl FnMut(WorkId, &RealtimeInputFrame, &mut RealtimeSessionBranch<M::Branch,S,R,C,P>)
            -> Result<O, E>,
        observe_failure: impl FnMut(WorkId,O::Error),
    ) -> Result<SchedulerProgress<RealtimeInputFrame,O>, SchedulerError>
    where T: BoundedConsensusTransport,
        <T::Completion as Completion>::Error: std::fmt::Display,
        A: std::error::Error, E: std::error::Error,
        O: eredu_core::scheduler::DistributedTransitionOutput {
        let wait = self.distributed_turn_wait(protocol, transport)?;
        self.scheduler.run_distributed_bounded_with_preparation_and_errors(protocol, transport, wait,
            now, maximum_frames, prepare, |id, frame, branch| {
                branch.admit_batch(frame.batch())
                    .map_err(RealtimeSessionSubmissionError::<E>::Session)?;
                execute(id, frame, branch).map_err(RealtimeSessionSubmissionError::Execution)
            },observe_failure)
    }

    fn distributed_turn_wait<T: BoundedConsensusTransport>(
        &self, protocol: u64, transport: &T,
    ) -> Result<BoundedCompletionWait, SchedulerError>
    where <T::Completion as Completion>::Error: std::fmt::Display {
        if self.model.topology().is_replicated() {
            return Err(SchedulerError::with_host_source(transport.metadata_funding(),SchedulerError::Consensus,format_args!("distributed realtime turns require a non-replicated selected topology")));
        }
        let participants = self.model.topology().world_size();
        if transport.participant_count() != participants {
            return Err(SchedulerError::with_host_source(transport.metadata_funding(),SchedulerError::Consensus,format_args!(
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
        .map_err(|error| SchedulerError::from_consensus_with_host_source(transport.metadata_funding(),error))?;
        Ok(wait)
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
    pub fn replace_sampling<E: std::error::Error + Send + Sync + 'static>(
        &mut self,
        request: RequestId,
        sampling: RealtimeSampling,
        realize: impl FnOnce(RealtimeSampling) -> Result<(Vec<S>, Option<R>), E>,
    ) -> Result<(), RealtimeSamplingUpdateError> {
        if self.scheduler.queued_for_request(request) != 0 {
            return Err(RealtimeSamplingReplacementError::QueuedWork);
        }
        let state = self
            .scheduler
            .request_state_mut(request)
            .map_err(RealtimeSamplingReplacementError::Scheduler)?;
        let (samplers, random) = realize(sampling).map_err(|error| {
            RealtimeSamplingReplacementError::Realization(BackendFailure::from_error(error))
        })?;
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
    ) -> Result<ReleasedRealtimeSession<M, S, R, C, P>, SchedulerError> {
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
    pub fn request_state(&self, request: RequestId) -> Option<&RealtimeSessionState<M, S, R, C, P>> {
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
    /// The caller supplied an empty released-state slot.
    #[error("no released realtime session is available to resume")]
    MissingReleasedState,
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
    /// The admitted frame's host metadata source refused contract construction.
    #[error(transparent)]
    HostMetadata(#[from] eredu_core::HostMetadataFundingError),
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

#[derive(Debug, thiserror::Error)]
enum RealtimeSessionSubmissionError<E: std::error::Error> {
    #[error(transparent)]
    Session(RealtimeSessionExecutionError),
    #[error(transparent)]
    Execution(E),
}

/// Failure while publishing or discarding complete session state.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeSessionTransactionError {
    /// Atomic generation-state transaction failed.
    #[error(transparent)]
    Generation(RealtimeGenerationTransactionError),
    /// An unpublished branch did not retain its exact session identity.
    #[error("realtime session branch identity does not match canonical state")]
    IdentityMismatch,
    /// An unpublished branch attempted to change committed batch.
    #[error("realtime session branch batch does not match canonical state")]
    BatchMismatch,
}

/// Sampling replacement failure while preserving queued-work ordering.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeSamplingReplacementError {
    /// Queued frames retain the sampling policy under which they were accepted.
    #[error("cannot replace realtime sampling while frames are queued")]
    QueuedWork,
    /// Core scheduler did not admit mutable idle-state access.
    #[error(transparent)]
    Scheduler(SchedulerError),
    /// Backend-neutral sampler/RNG realization failed.
    #[error("realtime sampling realization failed")]
    Realization(#[source] BackendFailure),
    /// New sampler/RNG state did not match generation geometry.
    #[error("realtime generation rejected replacement sampling")]
    Generation(#[source] RealtimeGenerationTransactionError),
}

/// Sampling replacement failure specialized to one generation transaction.
pub type RealtimeSamplingUpdateError = RealtimeSamplingReplacementError;

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
    #[cfg(all(target_arch="aarch64",target_os="macos"))]
    fn original_frame_claims_and_retained_outputs_share_the_actual_pool_account() {
        use crate::working_memory::{WorkingMemoryPool,WorkingMemoryError,InferenceExecutionIdentity,
            RealtimeFrameRequirements,HostSourceConstructionFacts};
        let mut sessions=Sessions::new(identity("frame-account"),limits(1)).unwrap();
        let request=RequestId::new(41);
        sessions.register(request,generation()).unwrap();
        sessions.enqueue(request,frame(1)).unwrap();
        let pool=WorkingMemoryPool::new(1<<24,0).unwrap();
        let execution=InferenceExecutionIdentity::default();
        let mut observed=false;
        sessions.run_local_turn(Instant::now(),|id,frame,branch| {
            observed=true;
            let occurrence=branch.frame_occurrence().unwrap();
            let requirements=||RealtimeFrameRequirements::new(occurrence,Some(128),Some(256),
                Some(64),Some(32),Some(4096),HostSourceConstructionFacts::new(128,2,0).unwrap()).unwrap();
            let required=requirements().required_bytes().unwrap();
            let mut accepted=pool.reserve_realtime_frame(&execution,requirements(),required,None).unwrap();
            assert_eq!(pool.used_bytes().unwrap(),required);
            assert!(matches!(accepted.validate(&InferenceExecutionIdentity::default(),occurrence),
                Err(WorkingMemoryError::IdentityMismatch)));
            accepted.validate(&execution,occurrence).unwrap();
            let native=accepted.claim_native().unwrap();
            assert!(matches!(accepted.claim_native(),Err(WorkingMemoryError::AlreadyStarted)));
            let source=accepted.take_source_constructions().unwrap();
            assert!(matches!(accepted.take_source_constructions(),Err(WorkingMemoryError::AlreadyStarted)));
            let funding=accepted.metadata_funding().unwrap();
            funding.reserve_metadata(128).unwrap();
            let alias=funding.clone();
            assert!(alias.reserve_metadata(usize::MAX).is_err());
            funding.reserve_metadata(128).unwrap();
            assert!(accepted.metadata_funding().is_err());
            let output_custody=native.budget_custody();
            let retained_origin: crate::working_memory::OriginalOperationMetadataCustody =
                output_custody.clone().into();
            retained_origin.validate_retained_origin(&pool).unwrap();
            let foreign = WorkingMemoryPool::new(1 << 24, 0).unwrap();
            assert_eq!(retained_origin.validate_retained_origin(&foreign),
                Err(WorkingMemoryError::IdentityMismatch));
            drop((accepted,native,source,funding,alias));
            retained_origin.validate_retained_origin(&pool).unwrap();
            drop(retained_origin);
            assert_eq!(pool.used_bytes().unwrap(),required);
            assert!(matches!(pool.reserve_realtime_frame(&execution,requirements(),required,None)
                .unwrap_err().cause(),WorkingMemoryError::BudgetExceeded{..}));
            drop(output_custody);
            assert_eq!(pool.used_bytes().unwrap(),0);
            let retry=pool.reserve_realtime_frame(&execution,requirements(),required,None).unwrap();
            drop(retry);
            assert_eq!(pool.used_bytes().unwrap(),0);
            let fenced = pool.reserve_realtime_frame(&execution, requirements(), required, None).unwrap();
            let source = fenced.budget_custody();
            drop(fenced);
            source.assert_closed_metadata_origin_refusal(&pool);
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), required, "quarantine cannot refund the original account");
            execute_immediately(id,frame,branch)
        }).unwrap();
        assert!(observed);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn original_realtime_source_admits_startup_under_its_accepted_capacity() {
        use crate::{
            prefetch::BackgroundPrefetchWorker,
            working_memory::{HostSourceConstructionFacts, InferenceExecutionIdentity,
                OriginalHostSourceCustody, RealtimeFrameRequirements, WorkingMemoryError,
                WorkingMemoryPool},
        };
        fn operation(_: &eredu_core::residency::OffloadUnitId) -> Result<(), String> { Ok(()) }
        let operation = operation as fn(&eredu_core::residency::OffloadUnitId) -> Result<(), String>;
        let plan = match BackgroundPrefetchWorker::thread_startup_plan(
            &operation, "realtime-source-startup") {
            Ok(plan) => plan,
            Err(WorkingMemoryError::UnknownBound) => {
                assert!(std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_none());
                return;
            }
            Err(cause) => panic!("unexpected startup source refusal: {cause}"),
        };
        let mut sessions = Sessions::new(identity("startup-account"), limits(1)).unwrap();
        let request = RequestId::new(42);
        sessions.register(request, generation()).unwrap();
        sessions.enqueue(request, frame(1)).unwrap();
        let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let mut observed = false;
        sessions.run_local_turn(Instant::now(), |id, frame, branch| {
            observed = true;
            let occurrence = branch.frame_occurrence().unwrap();
            let requirements = || RealtimeFrameRequirements::new(occurrence, Some(128),
                Some(256), Some(64), Some(32), Some(4096),
                HostSourceConstructionFacts::new(128, 2, 0).unwrap()).unwrap();
            let frame_bytes = requirements().required_bytes().unwrap();
            let capacity = frame_bytes.checked_add(plan.required_bytes()).unwrap();
            let small = pool.reserve_realtime_frame(
                &execution, requirements(), capacity - 1, None).unwrap();
            let source = OriginalHostSourceCustody::from(small.budget_custody());
            assert!(matches!(plan.prepare_for_source(&source, None),
                Err(WorkingMemoryError::BudgetExceeded { .. })));
            assert_eq!(pool.used_bytes().unwrap(), frame_bytes);
            drop((source, small));
            assert_eq!(pool.used_bytes().unwrap(), 0);
            let accepted = pool.reserve_realtime_frame(
                &execution, requirements(), capacity, None).unwrap();
            let source = OriginalHostSourceCustody::from(accepted.budget_custody());
            let startup = plan.prepare_for_source(&source, None).unwrap();
            assert_eq!(pool.used_bytes().unwrap(), capacity);
            // A separate, unstarted host hold does not keep the original frame
            // alive or acquire any of its native/source construction claims.
            drop((source, accepted));
            assert_eq!(pool.used_bytes().unwrap(), plan.required_bytes());
            drop(startup);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            execute_immediately(id, frame, branch)
        }).unwrap();
        assert!(observed);
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
        let mut released = Some(first.release(request).unwrap());
        assert_eq!(released.as_ref().unwrap().incarnation(), incarnation);

        let mut wrong = Sessions::new(identity("b"), limits(1)).unwrap();
        let error = wrong.resume(request, &mut released).unwrap_err();
        assert!(matches!(error, RealtimeSessionError::ModelIdentityMismatch));
        assert_eq!(released.as_ref().unwrap().incarnation(), incarnation);
        first.resume(request, &mut released).unwrap();
        assert!(released.is_none());
        assert!(matches!(
            first.resume(request, &mut released),
            Err(RealtimeSessionError::MissingReleasedState)
        ));
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
