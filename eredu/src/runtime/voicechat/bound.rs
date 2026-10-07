//! Prepared execution bound to the ordinary controlled session and its endpoints.

use std::time::Duration;

use eredu_core::scheduler::{RequestId, SemanticStateTransaction};
use eredu_runtime::execution_control::{ForkStateEstimate, IsolatedStateFork, SnapshotValue};

use super::{
    VoiceChatAdvance, VoiceChatConversation, VoiceChatExecutionError, VoiceChatInputConfig,
    VoiceChatInputSender, VoiceChatPublicationBranch, VoiceChatPublicationLimits,
    VoiceChatPublicationReceiver, VoiceChatPublicationState, VoiceChatReminderPlayback,
    VoiceChatSession, VoiceChatSessionConfig, VoiceChatSessionError, VoiceChatSessionProgress,
    VoiceChatSessionStatus, VoiceChatSubmittedWork, VoiceChatToolTicket, VoiceChatWork,
};

/// Prepared submission mechanism for the shared VoiceChat scheduler.
///
/// Implementations retain immutable prompt inputs and execution resources. All
/// conversational, neural and random state belongs to the supplied transaction;
/// keeping that state in this mechanism would bypass snapshot and fork accounting.
/// Submission must retain native resources through exact completion, including
/// failures after submission. This contract introduces no scheduling policy.
pub trait VoiceChatExecutor<S: SemanticStateTransaction, D> {
    /// Whether another invocation retains these shared execution resources.
    /// A busy mechanism is polled before new work is admitted, never submitted
    /// as an execution failure. Stateless mechanisms need no availability fence.
    fn is_busy(&self) -> bool {
        false
    }

    /// Submits the admitted work against its unpublished transaction branch.
    fn submit(
        &mut self,
        work: &VoiceChatWork<D>,
        branch: &mut VoiceChatPublicationBranch<S::Branch, D>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError>;
}

impl<S, D, F> VoiceChatExecutor<S, D> for F
where
    S: SemanticStateTransaction,
    F: FnMut(
        &VoiceChatWork<D>,
        &mut VoiceChatPublicationBranch<S::Branch, D>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError>,
{
    fn submit(
        &mut self,
        work: &VoiceChatWork<D>,
        branch: &mut VoiceChatPublicationBranch<S::Branch, D>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
        self(work, branch)
    }
}

/// A prepared executor, the existing session, and its bounded input/output queues.
///
/// Controlled and iterator-driven execution advance the same scheduler. Input
/// remains available while work is pending; committed output remains available
/// after cancellation or failure. This owner performs no hidden clock reads.
pub struct BoundVoiceChatSession<S, D, E>
where
    S: SemanticStateTransaction,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
{
    session: VoiceChatSession<S, D>,
    executor: E,
    input: VoiceChatInputSender,
    output: VoiceChatPublicationReceiver<D>,
}

impl<S, D, E> BoundVoiceChatSession<S, D, E>
where
    S: SemanticStateTransaction,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
    E: VoiceChatExecutor<S, D>,
{
    /// Registers prepared neural and host state and queues explicit prefill.
    /// No neural execution occurs until the first advancement.
    pub fn start(
        neural: S,
        conversation: VoiceChatConversation,
        input_config: VoiceChatInputConfig,
        output_limits: VoiceChatPublicationLimits,
        config: VoiceChatSessionConfig,
        executor: E,
    ) -> Result<Self, VoiceChatSessionError>
    where
        S::Branch: eredu_architectures::voicechat::state::InterruptionFeedback,
    {
        let (state, input, output) = VoiceChatPublicationState::new_with_input(
            neural,
            conversation,
            input_config,
            output_limits,
        )?;
        let session = VoiceChatSession::start(state, config)?;
        Ok(Self::from_parts(session, executor, input, output))
    }

    pub(super) fn from_parts(
        session: VoiceChatSession<S, D>,
        executor: E,
        input: VoiceChatInputSender,
        output: VoiceChatPublicationReceiver<D>,
    ) -> Self {
        Self {
            session,
            executor,
            input,
            output,
        }
    }

    /// Ordinary session status and resource-accounting access.
    pub fn session(&self) -> &VoiceChatSession<S, D> {
        &self.session
    }

    /// Existing tool, interruption, snapshot and restore operations.
    /// These operations retain the ordinary session's admission and replay fences.
    pub fn session_mut(&mut self) -> &mut VoiceChatSession<S, D> {
        &mut self.session
    }

    /// Prepared execution resources and their admitted reports.
    pub fn executor(&self) -> &E {
        &self.executor
    }

    /// Current PCM admission and finish status.
    pub fn input(&self) -> &VoiceChatInputSender {
        &self.input
    }

    /// Submits PCM or declares finish using the bounded ingress contract.
    pub fn input_mut(&mut self) -> &mut VoiceChatInputSender {
        &mut self.input
    }

    /// Current committed-output queue usage.
    pub fn output(&self) -> &VoiceChatPublicationReceiver<D> {
        &self.output
    }

    /// Delivers committed events, including after cancellation or failure.
    pub fn output_mut(&mut self) -> &mut VoiceChatPublicationReceiver<D> {
        &mut self.output
    }

    /// Current ordinary session lifecycle.
    pub fn status(&self) -> VoiceChatSessionStatus {
        self.session.status()
    }

    /// Advances at most one invocation using the retained execution mechanism.
    pub fn advance(
        &mut self,
        advance: VoiceChatAdvance,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        advance_bound(&mut self.session, &mut self.executor, advance)
    }

    /// Repeatedly advances this same session with explicit recorded ticks.
    /// Exhausting ticks retains pending work; dropping the iterator cancels nothing.
    pub fn drive<'a, I>(
        &'a mut self,
        ticks: I,
    ) -> impl Iterator<Item = Result<VoiceChatSessionProgress, VoiceChatSessionError>> + 'a
    where
        I: IntoIterator<Item = VoiceChatAdvance>,
        I::IntoIter: 'a,
        S: 'a,
        D: 'a,
        E: 'a,
    {
        drive_bound(&mut self.session, &mut self.executor, ticks)
    }

    /// Iterates while leaving PCM submission and committed-event delivery available.
    /// Drain output between yields to release bounded queue capacity. Endpoint
    /// operations keep their ordinary admission and replay-frontier checks.
    pub fn drive_with_endpoints<'a, I>(
        &'a mut self,
        ticks: I,
    ) -> (
        impl Iterator<Item = Result<VoiceChatSessionProgress, VoiceChatSessionError>> + 'a,
        &'a mut VoiceChatInputSender,
        &'a mut VoiceChatPublicationReceiver<D>,
    )
    where
        I: IntoIterator<Item = VoiceChatAdvance>,
        I::IntoIter: 'a,
        S: 'a,
        D: 'a,
        E: 'a,
    {
        let drive = drive_bound(&mut self.session, &mut self.executor, ticks);
        (drive, &mut self.input, &mut self.output)
    }

    /// Polls exact completion without submitting another invocation.
    pub fn poll(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        self.session.poll(at)
    }

    /// Cancels while preserving pending completion and committed output ownership.
    pub fn cancel(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatSessionStatus, VoiceChatSessionError> {
        self.session.cancel(at)
    }

    /// Separates ownership without dropping pending work or undelivered events.
    pub fn into_parts(
        self,
    ) -> (
        VoiceChatSession<S, D>,
        E,
        VoiceChatInputSender,
        VoiceChatPublicationReceiver<D>,
    ) {
        (self.session, self.executor, self.input, self.output)
    }
}

/// An independently owned continuation with explicit external-action mappings.
pub struct BoundVoiceChatFork<S, D, E>
where
    S: SemanticStateTransaction,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
{
    /// Child using the same bound advancement and endpoint API.
    pub session: BoundVoiceChatSession<S, D, E>,
    /// Source request whose settled frontier was copied.
    pub source_request: RequestId,
    /// Fresh identity used by the child.
    pub request: RequestId,
    /// Pending tool ticket mapping; this does not authorize duplicate tool execution.
    pub pending_tool: Option<(VoiceChatToolTicket, VoiceChatToolTicket)>,
    /// Pending playback mapping, without reemitting already presented audio.
    pub pending_playback: Option<(VoiceChatReminderPlayback, VoiceChatReminderPlayback)>,
}

impl<S, D, E> BoundVoiceChatSession<S, D, E>
where
    S: SemanticStateTransaction + IsolatedStateFork + ForkStateEstimate,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
    D: SnapshotValue,
    E: VoiceChatExecutor<S, D>,
{
    /// Forks retained state using explicitly supplied compatible execution resources.
    ///
    /// The executor must use the same prepared model and immutable prompt inputs.
    /// Shared native executors must serialize submissions and retain completion
    /// ownership. No executor is cloned or assumed to contain replayable state;
    /// the existing session accounts for all copied mutable state and endpoints.
    pub fn fork_with_executor<F>(
        &mut self,
        request: RequestId,
        executor: F,
    ) -> Result<BoundVoiceChatFork<S, D, F>, VoiceChatSessionError>
    where
        F: VoiceChatExecutor<S, D>,
    {
        let fork = self.session.fork(request)?;
        Ok(BoundVoiceChatFork {
            session: BoundVoiceChatSession::from_parts(
                fork.session,
                executor,
                fork.input,
                fork.output,
            ),
            source_request: fork.source_request,
            request: fork.request,
            pending_tool: fork.pending_tool,
            pending_playback: fork.pending_playback,
        })
    }
}

// Availability is a mechanism fence around ordinary advancement. It cannot
// prepare work, choose policy, adopt state, or publish a frame independently.
fn advance_bound<S, D, E>(
    session: &mut VoiceChatSession<S, D>,
    executor: &mut E,
    advance: VoiceChatAdvance,
) -> Result<VoiceChatSessionProgress, VoiceChatSessionError>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
    E: VoiceChatExecutor<S, D>,
{
    if executor.is_busy() {
        let progress = session.poll(advance.at)?;
        if progress.committed != 0 || progress.status != VoiceChatSessionStatus::Ready {
            return Ok(progress);
        }
        if executor.is_busy() {
            return Err(VoiceChatSessionError::Busy);
        }
    }
    session.advance(advance, |work, branch| executor.submit(work, branch))
}

fn drive_bound<'a, S, D, E, I>(
    session: &'a mut VoiceChatSession<S, D>,
    executor: &'a mut E,
    ticks: I,
) -> impl Iterator<Item = Result<VoiceChatSessionProgress, VoiceChatSessionError>> + 'a
where
    S: SemanticStateTransaction + 'a,
    S::Error: Send + Sync + 'static,
    D: 'a,
    E: VoiceChatExecutor<S, D> + 'a,
    I: IntoIterator<Item = VoiceChatAdvance>,
    I::IntoIter: 'a,
{
    let mut ticks = ticks.into_iter();
    std::iter::from_fn(move || {
        if session.status().settled() {
            return None;
        }
        Some(advance_bound(session, executor, ticks.next()?))
    })
}
