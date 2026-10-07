//! Host publication coupled to the existing scheduler's semantic transaction.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::Duration,
};

use eredu_architectures::voicechat::{
    input::PcmInputState,
    observation::OwnedHostOutput,
    schedule::{AudioSource, Step},
    state::InterruptionFeedback,
    tts_feedback::OutputPolicy,
};
use eredu_core::{BackendFailure, scheduler::SemanticStateTransaction};
use eredu_runtime::{
    MappedRealtimeFrameHostObserver, PrepublicationRealtimeFrame, RealtimeFrameHostObserver,
};

use super::{
    VoiceChatConversation, VoiceChatInputConfig, VoiceChatInputError, VoiceChatInputPermit,
    VoiceChatInputSender, VoiceChatPublicationCapacity, VoiceChatPublicationError,
    VoiceChatPublicationLimits, VoiceChatPublicationReceiver, VoiceChatPublicationUsage,
    VoiceChatReminderPlayback, VoiceChatReminderStep, VoiceChatToolTicket, VoiceChatToolWaitAction,
    VoiceChatToolWaitMessages, VoiceChatToolWaitPolicy, VoiceChatTurnError, VoiceChatTurnResolver,
    conversation::{StagedVoiceChatConversation, VoiceChatConversationTransition},
    ingress::Ingress,
    publication_queue::{Queue, Reservation},
};

mod fork;
pub(super) mod snapshot;

/// Ordinary neural state and its bounded, committed host publications.
///
/// Register this owner with the existing shared scheduler. Host policy adoption
/// and queue delivery occur inside the same semantic commit as neural state;
/// dropping the scheduler's returned progress does not lose committed events.
/// This owner introduces no scheduler, clock, native execution or tool executor.
/// Dropping this owner closes its optional PCM sender without consuming queued input.
pub struct VoiceChatPublicationState<S: SemanticStateTransaction, D> {
    neural: S,
    reset_interruption_feedback: fn(&mut S::Branch),
    conversation: VoiceChatConversation,
    queue: Rc<RefCell<Queue<D>>>,
    input: Option<Rc<RefCell<Ingress>>>,
    identity: Rc<()>,
    epoch: Rc<Cell<u64>>,
    fork_retention: Option<eredu_runtime::execution_control::SnapshotReservation>,
    fork_neural_growth: Option<(
        eredu_runtime::execution_control::SnapshotRetainedGrowth,
        fn(&S::Branch) -> Result<u64, BackendFailure>,
    )>,
}

impl<S: SemanticStateTransaction, D> Drop for VoiceChatPublicationState<S, D> {
    fn drop(&mut self) {
        if let Some(input) = &self.input {
            Ingress::close(input);
        }
    }
}

impl<S: SemanticStateTransaction, D> VoiceChatPublicationState<S, D> {
    pub(super) fn require_input(&self) -> Result<(), VoiceChatPublicationError> {
        self.input
            .as_ref()
            .map(|_| ())
            .ok_or(VoiceChatInputError::Unavailable.into())
    }
    /// Establishes the bounded queue before any invocation is submitted.
    /// Retain the receiver outside the scheduler so committed publications remain
    /// deliverable after cancellation, failure or removal of canonical state.
    pub fn new(
        neural: S,
        conversation: VoiceChatConversation,
        limits: VoiceChatPublicationLimits,
    ) -> Result<(Self, VoiceChatPublicationReceiver<D>), VoiceChatPublicationError>
    where
        S::Branch: InterruptionFeedback,
    {
        let queue = Queue::new(limits)?;
        let receiver = VoiceChatPublicationReceiver::new(Rc::clone(&queue));
        Ok((
            Self {
                neural,
                reset_interruption_feedback: S::Branch::reset_interruption_feedback,
                conversation,
                queue,
                input: None,
                identity: Rc::new(()),
                epoch: Rc::new(Cell::new(0)),
                fork_retention: None,
                fork_neural_growth: None,
            },
            receiver,
        ))
    }

    /// Establishes bounded PCM ingress and committed host publication together.
    /// Input remains queued until the matching neural and host transition commits.
    pub fn new_with_input(
        neural: S,
        conversation: VoiceChatConversation,
        input_config: VoiceChatInputConfig,
        output_limits: VoiceChatPublicationLimits,
    ) -> Result<
        (Self, VoiceChatInputSender, VoiceChatPublicationReceiver<D>),
        VoiceChatPublicationError,
    >
    where
        S::Branch: InterruptionFeedback,
    {
        let (input, sender) = Ingress::new(input_config)?;
        let (mut state, receiver) = Self::new(neural, conversation, output_limits)?;
        state.input = Some(input);
        Ok((state, sender, receiver))
    }

    /// Reserves the next complete input frame, or explicit final partial frame.
    /// Dropping cancelled work preserves its PCM for a later admission.
    pub fn reserve_input(&self) -> Result<VoiceChatInputPermit, VoiceChatPublicationError> {
        let input = self
            .input
            .as_ref()
            .ok_or(VoiceChatInputError::Unavailable)?;
        Ok(Ingress::reserve(input)?)
    }

    /// Prepares explicit prompt/Aria work for the existing shared scheduler.
    /// The retained executor supplies its prepared prompt and conditioning assets.
    pub fn prepare_prefill(
        &mut self,
        backbone_tokens: u64,
        at: Duration,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<super::VoiceChatWork<D>, VoiceChatPublicationError> {
        let output = self.reserve(capacity)?;
        let transition = self
            .conversation
            .prepare_prefill(backbone_tokens, at, capacity)?;
        Ok(super::VoiceChatWork::new(
            super::VoiceChatWorkKind::Prefill { backbone_tokens },
            at,
            transition,
            output,
            None,
        ))
    }

    /// Selects and reserves one invocation using committed policy and explicit time.
    /// Queue backpressure and blocked selection consume neither PCM nor wait policy.
    /// Ready work freezes external host mutations until it commits or is discarded.
    pub fn prepare_next(
        &mut self,
        advance: super::VoiceChatAdvance,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<super::VoiceChatPreparation<D>, VoiceChatPublicationError> {
        let output = self.reserve(capacity)?;
        let input = match self.reserve_input() {
            Ok(input) => Some(input),
            Err(VoiceChatPublicationError::Input(
                VoiceChatInputError::NeedsInput | VoiceChatInputError::Drained,
            )) => None,
            Err(error) => return Err(error),
        };
        let fresh = input.as_ref().is_some_and(|input| !input.is_finish_only());
        let finished = input
            .as_ref()
            .is_some_and(VoiceChatInputPermit::is_finish_only)
            || self
                .input
                .as_ref()
                .is_some_and(|input| input.borrow().status().finish_committed);
        match self
            .conversation
            .prepare_next(advance, fresh, finished, capacity)?
        {
            Ok((kind, transition)) => {
                let retains_input = matches!(
                    kind.step(),
                    Step::LiveAudio
                        | Step::PerceptionOnly
                        | Step::Tool {
                            audio: AudioSource::Fresh
                        }
                        | Step::Drain
                );
                Ok(super::VoiceChatPreparation::Ready(
                    super::VoiceChatWork::new(
                        kind,
                        advance.at,
                        transition,
                        output,
                        if retains_input { input } else { None },
                    ),
                ))
            }
            Err(blocked) => Ok(super::VoiceChatPreparation::Blocked(blocked)),
        }
    }

    /// Borrows the canonical neural transaction without changing its frontier.
    pub fn neural_state(&self) -> &S {
        &self.neural
    }

    /// Borrows committed host policy without exposing separate admissions.
    pub fn conversation(&self) -> &VoiceChatConversation {
        &self.conversation
    }

    /// Reserves publication credit before work enters the shared scheduler.
    /// Keep the move-only permit with queued work; dropping rejected or cancelled
    /// work returns its credit without silently dropping an accepted frame.
    pub fn reserve(
        &self,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<VoiceChatPublicationPermit<D>, VoiceChatPublicationError> {
        Ok(VoiceChatPublicationPermit {
            reservation: Queue::reserve(&self.queue, capacity)?,
            capacity,
        })
    }

    /// Submits an external tool result between admitted invocations.
    pub fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        text: &str,
    ) -> Result<(), VoiceChatPublicationError> {
        Ok(self.conversation.submit_response(ticket, text)?)
    }

    /// Invalidates conversational work after outstanding execution retires.
    pub fn interrupt(&mut self) -> Result<(), VoiceChatPublicationError> {
        Ok(self.conversation.interrupt()?)
    }

    /// Establishes recorded-time waiting for a committed tool batch.
    pub fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<(), VoiceChatPublicationError> {
        Ok(self.conversation.start_wait(ticket, at, policy, messages)?)
    }

    /// Selects waiting work from explicit recorded time.
    pub fn advance_wait(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatToolWaitAction, VoiceChatPublicationError> {
        Ok(self.conversation.advance_wait(at)?)
    }

    /// Records actual reminder presentation completion.
    pub fn finish_playback(
        &mut self,
        receipt: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatPublicationError> {
        Ok(self.conversation.finish_playback(receipt, at)?)
    }

    /// Reports committed storage and outstanding reservations.
    pub fn usage(&self) -> VoiceChatPublicationUsage {
        self.queue.borrow().usage()
    }
}

/// Move-only queue credit retained with work before scheduler submission.
/// Dropping this permit returns the entire reservation to its originating queue.
#[must_use = "retain this reservation with work until admission or cancellation"]
pub struct VoiceChatPublicationPermit<D> {
    reservation: Reservation<D>,
    capacity: VoiceChatPublicationCapacity,
}

impl<D> VoiceChatPublicationPermit<D> {
    pub(super) fn encode_descriptor(&self, output: &mut Vec<u32>) {
        for value in [
            self.capacity.speech_samples,
            self.capacity.transcription_tokens,
            self.capacity.agent_text_bytes,
            self.capacity.transcription_text_bytes,
            self.capacity.audio_codes,
            self.capacity.tool_events,
            self.capacity.tool_bytes,
            self.capacity.diagnostics,
        ] {
            super::driver::encode_u64(output, value as u64);
        }
    }
}

/// Unpublished neural branch with exclusive, bounded host publication authority.
/// No host successor is adopted before this branch commits through its owner.
pub struct VoiceChatPublicationBranch<B, D> {
    neural: Option<B>,
    conversation: VoiceChatConversation,
    queue: Rc<RefCell<Queue<D>>>,
    input: Option<Rc<RefCell<Ingress>>>,
    input_permit: Option<VoiceChatInputPermit>,
    identity: Rc<()>,
    epoch: Rc<Cell<u64>>,
    predecessor: u64,
    _fork_retention: Option<eredu_runtime::execution_control::SnapshotReservation>,
    reservation: Option<Reservation<D>>,
    staged: Rc<RefCell<Option<StagedVoiceChatConversation<D>>>>,
}

impl<B, D> VoiceChatPublicationBranch<B, D> {
    pub(super) fn admit_prepared(
        &mut self,
        step: Step,
        transition: VoiceChatConversationTransition,
        output: VoiceChatPublicationPermit<D>,
        input: Option<VoiceChatInputPermit>,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError>
    where
        B: PcmInputState,
    {
        if let Some(input) = &input {
            if !self
                .input
                .as_ref()
                .is_some_and(|owner| input.belongs_to(owner))
            {
                return Err(VoiceChatInputError::ForeignPermit.into());
            }
        } else if step == Step::Drain {
            Ingress::validate_drain(
                self.input
                    .as_ref()
                    .ok_or(VoiceChatInputError::Unavailable)?,
            )?;
        }
        let admission = self.admit_inner(output, |_, _| Ok(transition))?;
        if let Some(input) = input {
            input.stage(step, self.neural_state_mut())?;
            self.input_permit = Some(input);
        }
        Ok(admission)
    }
    /// Borrows the ordinary unpublished neural branch for family execution.
    pub fn neural_state_mut(&mut self) -> &mut B {
        self.neural.as_mut().expect("live publication branch")
    }

    /// Consumes reserved output storage and admits one ordinary family invocation.
    pub fn admit(
        &mut self,
        step: Step,
        permit: VoiceChatPublicationPermit<D>,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError> {
        if let Some(input) = &self.input {
            if matches!(
                step,
                Step::LiveAudio
                    | Step::PerceptionOnly
                    | Step::Tool {
                        audio: AudioSource::Fresh
                    }
            ) {
                return Err(VoiceChatInputError::Required.into());
            }
            if step == Step::Drain {
                Ingress::validate_drain(input)?;
            }
        }
        self.admit_inner(permit, |conversation, capacity| {
            conversation.admit(step, capacity)
        })
    }

    /// Stages reserved PCM in the ordinary unpublished neural input state.
    /// Both permits remain owned by this transition until commit or retirement.
    /// Any admission failure requires discarding the branch before retrying.
    pub fn admit_input(
        &mut self,
        step: Step,
        output_permit: VoiceChatPublicationPermit<D>,
        input_permit: VoiceChatInputPermit,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError>
    where
        B: PcmInputState,
    {
        let input = self
            .input
            .as_ref()
            .ok_or(VoiceChatInputError::Unavailable)?;
        if !input_permit.belongs_to(input) {
            return Err(VoiceChatInputError::ForeignPermit.into());
        }
        let admission = self.admit_inner(output_permit, |conversation, capacity| {
            conversation.admit(step, capacity)
        })?;
        input_permit.stage(step, self.neural_state_mut())?;
        self.input_permit = Some(input_permit);
        Ok(admission)
    }

    /// Consumes reserved output storage for an exactly attributed reminder invocation.
    pub fn admit_reminder(
        &mut self,
        receipt: VoiceChatReminderStep,
        at: Duration,
        permit: VoiceChatPublicationPermit<D>,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError> {
        self.admit_inner(permit, |conversation, capacity| {
            conversation.admit_reminder(receipt, at, capacity)
        })
    }

    fn admit_inner(
        &mut self,
        permit: VoiceChatPublicationPermit<D>,
        admit: impl FnOnce(
            &mut VoiceChatConversation,
            VoiceChatPublicationCapacity,
        )
            -> Result<VoiceChatConversationTransition, super::VoiceChatConversationError>,
    ) -> Result<VoiceChatPublicationAdmission<D>, VoiceChatPublicationError> {
        if self.predecessor != self.epoch.get() {
            return Err(VoiceChatPublicationError::Transaction("stale branch"));
        }
        if self.reservation.is_some() {
            return Err(VoiceChatPublicationError::Transaction(
                "branch already admitted",
            ));
        }
        if !permit.reservation.belongs_to(&self.queue) {
            return Err(VoiceChatPublicationError::Transaction(
                "foreign publication permit",
            ));
        }
        let transition = admit(&mut self.conversation, permit.capacity)?;
        self.reservation = Some(permit.reservation);
        Ok(VoiceChatPublicationAdmission {
            transition,
            staged: Rc::clone(&self.staged),
        })
    }
}

/// One host admission attached to the matching native completion gate.
/// The observer stages its validated successor in the owning semantic branch;
/// the scheduler's commit adopts it and enqueues output automatically.
pub struct VoiceChatPublicationAdmission<D> {
    transition: VoiceChatConversationTransition,
    staged: Rc<RefCell<Option<StagedVoiceChatConversation<D>>>>,
}

impl<D> VoiceChatPublicationAdmission<D> {
    pub(super) fn tool_output_policy(&self, force_pad: bool) -> OutputPolicy<'_> {
        self.transition.tool_output_policy(force_pad)
    }

    /// Resolves live decisions against this exclusively admitted host candidate.
    pub fn resolver(&self) -> Result<VoiceChatTurnResolver<'_>, VoiceChatTurnError> {
        self.transition.resolver()
    }

    /// Adds mandatory host validation to the existing completion envelope.
    /// The gate contains no independently deliverable host output.
    pub fn attach<T, C, H>(
        self,
        gate: PrepublicationRealtimeFrame<T, C, H>,
    ) -> PrepublicationRealtimeFrame<
        T,
        C,
        MappedRealtimeFrameHostObserver<
            H,
            impl FnOnce(OwnedHostOutput<D>) -> Result<(), VoiceChatPublicationError>,
        >,
    >
    where
        H: RealtimeFrameHostObserver<T, Output = OwnedHostOutput<D>>,
    {
        let staged = self.staged;
        gate.map_host_output(move |frame| {
            let output = self.transition.stage(frame)?;
            let mut slot = staged.borrow_mut();
            if slot.is_some() {
                return Err(VoiceChatPublicationError::Transaction(
                    "output already staged",
                ));
            }
            *slot = Some(output);
            Ok(())
        })
    }
}

impl<S, D> SemanticStateTransaction for VoiceChatPublicationState<S, D>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
{
    type Branch = VoiceChatPublicationBranch<S::Branch, D>;
    type Error = VoiceChatPublicationError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(VoiceChatPublicationBranch {
            neural: Some(self.neural.branch().map_err(|error| {
                VoiceChatPublicationError::Backend(BackendFailure::from_error(error))
            })?),
            conversation: self.conversation.shared_owner(),
            queue: Rc::clone(&self.queue),
            input: self.input.as_ref().map(Rc::clone),
            input_permit: None,
            identity: Rc::clone(&self.identity),
            epoch: Rc::clone(&self.epoch),
            predecessor: self.epoch.get(),
            _fork_retention: self.fork_retention.clone(),
            reservation: None,
            staged: Rc::new(RefCell::new(None)),
        })
    }

    fn commit_branch(&mut self, mut branch: Self::Branch) -> Result<(), Self::Error> {
        let invalid = if !Rc::ptr_eq(&self.identity, &branch.identity) {
            Some("foreign branch")
        } else if branch.predecessor != self.epoch.get() {
            Some("stale branch")
        } else if branch.reservation.is_none() || branch.staged.borrow().is_none() {
            Some("branch has no completed host publication")
        } else if self.epoch.get() == u64::MAX {
            Some("publication frontier exhausted")
        } else {
            None
        };
        if let Some(reason) = invalid {
            Self::discard_branch(branch)?;
            return Err(VoiceChatPublicationError::Transaction(reason));
        }
        let charge = {
            let staged = branch.staged.borrow();
            let publication = staged
                .as_ref()
                .expect("validated host publication")
                .publication();
            let input_validation = branch
                .input_permit
                .as_ref()
                .map(|permit| permit.validate_output(&publication.frame))
                .transpose()
                .map_err(VoiceChatPublicationError::from);
            input_validation.and_then(|_| {
                branch
                    .reservation
                    .as_ref()
                    .expect("validated publication reservation")
                    .prepare(publication)
            })
        };
        let charge = match charge {
            Ok(charge) => charge,
            Err(error) => {
                Self::discard_branch(branch)?;
                return Err(error);
            }
        };
        if let Some((growth, estimate)) = &self.fork_neural_growth {
            let admission = estimate(branch.neural.as_ref().expect("live publication branch"))
                .and_then(|bytes| growth.admit(bytes).map_err(BackendFailure::from_error));
            if let Err(error) = admission {
                Self::discard_branch(branch)?;
                return Err(VoiceChatPublicationError::Backend(error));
            }
        }
        if branch
            .staged
            .borrow()
            .as_ref()
            .expect("validated host publication")
            .publication()
            .turn
            == Some(super::VoiceChatTurnEvent::Interrupted)
        {
            (self.reset_interruption_feedback)(
                branch.neural.as_mut().expect("live publication branch"),
            );
        }
        let next_epoch = self.epoch.get() + 1;
        let staged = branch
            .staged
            .borrow_mut()
            .take()
            .expect("validated host publication");
        let reservation = branch
            .reservation
            .take()
            .expect("validated publication reservation");
        let neural = branch.neural.take().expect("live publication branch");
        self.neural.commit_branch(neural).map_err(|error| {
            VoiceChatPublicationError::Backend(BackendFailure::from_error(error))
        })?;
        self.epoch.set(next_epoch);
        if let Some(input) = branch.input_permit.take() {
            input.commit();
        }
        reservation.commit(staged.commit(), charge);
        Ok(())
    }

    fn discard_branch(mut branch: Self::Branch) -> Result<(), Self::Error> {
        let neural = branch.neural.take().expect("live publication branch");
        S::discard_branch(neural)
            .map_err(|error| VoiceChatPublicationError::Backend(BackendFailure::from_error(error)))
    }
}

#[cfg(test)]
pub(crate) mod tests;
