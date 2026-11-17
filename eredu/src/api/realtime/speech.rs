//! Host-facing endpoints for a bound PCM conversation.

use std::{any::Any, time::Duration};

use eredu_core::{scheduler::SemanticStateTransaction, RealtimeDecisionDiagnostics};
use eredu_runtime::execution_control::{ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork};

use super::*;

/// Opaque idle PCM session snapshot, charged to its session's copy budget.
pub struct SpeechRealtimeSnapshot(Box<dyn Any>);

/// A forked PCM session and the mappings for already pending external work.
pub struct SpeechRealtimeFork {
    /// Independent child using the same host protocol.
    pub session: Box<dyn SpeechRealtimeSession>,
    /// Request whose idle state was copied.
    pub source_request: RequestId,
    /// Fresh request identity of the child.
    pub request: RequestId,
    /// Source and child tool tickets, when a result is outstanding.
    pub pending_tool: Option<(VoiceChatToolTicket, VoiceChatToolTicket)>,
    /// Source and child playback receipts, when completion is outstanding.
    pub pending_playback: Option<(VoiceChatReminderPlayback, VoiceChatReminderPlayback)>,
}

/// Shared execution resources used when forking a bound PCM session.
pub trait ShareSpeechExecution {
    /// Shares immutable inputs and serializes work using the existing executor.
    fn share_for_fork(&self) -> Self;
}

/// Controlled PCM conversation using portable endpoints and completion.
pub trait SpeechRealtimeSession {
    /// The PCM input endpoint.
    fn input(&mut self) -> &mut VoiceChatInputSender;
    /// The committed output endpoint.
    fn output(&mut self) -> &mut VoiceChatPublicationReceiver<RealtimeDecisionDiagnostics>;
    /// Current lifecycle.
    fn status(&self) -> VoiceChatSessionStatus;
    /// Whether admitted tool work is outstanding.
    fn has_tool_work(&self) -> bool;
    /// Submits at most one invocation.
    fn advance(
        &mut self,
        tick: VoiceChatAdvance,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError>;
    /// Polls existing work without submitting another invocation.
    fn poll(&mut self, at: Duration) -> Result<VoiceChatSessionProgress, VoiceChatSessionError>;
    /// Cancels while retaining pending completion and committed output.
    fn cancel(&mut self, at: Duration) -> Result<VoiceChatSessionStatus, VoiceChatSessionError>;
    /// Submits text for a committed tool call.
    fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        text: &str,
    ) -> Result<(), VoiceChatSessionError>;
    /// Begins recorded tool waiting.
    fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<(), VoiceChatSessionError>;
    /// Records completion of a published reminder's playback.
    fn finish_playback(
        &mut self,
        receipt: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatSessionError>;
    /// Invalidates tool and turn work after pending execution retires.
    fn interrupt(&mut self) -> Result<(), VoiceChatSessionError>;
    /// Configures the persistent snapshot and copy limits.
    fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), VoiceChatSessionError>;
    /// Returns the persistent snapshot and copy accounting.
    fn snapshot_usage(&self) -> Option<SnapshotUsage>;
    /// Captures the complete idle session.
    fn snapshot(&mut self) -> Result<SpeechRealtimeSnapshot, VoiceChatSessionError>;
    /// Restores an idle session before any irreversible host effects.
    fn restore(&mut self, snapshot: &SpeechRealtimeSnapshot) -> Result<(), VoiceChatSessionError>;
    /// Forks mutable state while retaining the same immutable execution resources.
    fn fork(&mut self, request: RequestId) -> Result<SpeechRealtimeFork, VoiceChatSessionError>;
}

impl<S, E> SpeechRealtimeSession for BoundVoiceChatSession<S, RealtimeDecisionDiagnostics, E>
where
    S: SemanticStateTransaction
        + IsolatedStateCopy
        + IsolatedStateFork
        + ForkStateEstimate
        + 'static,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
    E: VoiceChatExecutor<S, RealtimeDecisionDiagnostics> + ShareSpeechExecution + 'static,
{
    fn input(&mut self) -> &mut VoiceChatInputSender {
        self.input_mut()
    }
    fn output(&mut self) -> &mut VoiceChatPublicationReceiver<RealtimeDecisionDiagnostics> {
        self.output_mut()
    }
    fn status(&self) -> VoiceChatSessionStatus {
        self.status()
    }
    fn has_tool_work(&self) -> bool {
        self.session().has_tool_work()
    }
    fn advance(
        &mut self,
        tick: VoiceChatAdvance,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        self.advance(tick)
    }
    fn poll(&mut self, at: Duration) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        self.poll(at)
    }
    fn cancel(&mut self, at: Duration) -> Result<VoiceChatSessionStatus, VoiceChatSessionError> {
        self.cancel(at)
    }
    fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        text: &str,
    ) -> Result<(), VoiceChatSessionError> {
        self.session_mut().submit_response(ticket, text)
    }
    fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<(), VoiceChatSessionError> {
        self.session_mut().start_wait(ticket, at, policy, messages)
    }
    fn finish_playback(
        &mut self,
        receipt: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatSessionError> {
        self.session_mut().finish_playback(receipt, at)
    }
    fn interrupt(&mut self) -> Result<(), VoiceChatSessionError> {
        self.session_mut().interrupt()
    }

    fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), VoiceChatSessionError> {
        self.session_mut().enable_snapshots(limits)
    }

    fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.session().snapshot_usage()
    }

    fn snapshot(&mut self) -> Result<SpeechRealtimeSnapshot, VoiceChatSessionError> {
        Ok(SpeechRealtimeSnapshot(Box::new(
            self.session_mut().snapshot()?,
        )))
    }

    fn restore(&mut self, snapshot: &SpeechRealtimeSnapshot) -> Result<(), VoiceChatSessionError> {
        let snapshot = snapshot
            .0
            .downcast_ref()
            .ok_or(VoiceChatSnapshotError::Foreign)?;
        self.session_mut().restore(snapshot)
    }

    fn fork(&mut self, request: RequestId) -> Result<SpeechRealtimeFork, VoiceChatSessionError> {
        let executor = self.executor().share_for_fork();
        let fork = self.fork_with_executor(request, executor)?;
        Ok(SpeechRealtimeFork {
            session: Box::new(fork.session),
            source_request: fork.source_request,
            request: fork.request,
            pending_tool: fork.pending_tool,
            pending_playback: fork.pending_playback,
        })
    }
}

impl<'session> dyn SpeechRealtimeSession + 'session {
    /// Drives the same controlled session with recorded ticks.
    /// Dropping the iterator preserves pending work.
    pub fn drive<'a, I>(
        &'a mut self,
        ticks: I,
    ) -> impl Iterator<Item = Result<VoiceChatSessionProgress, VoiceChatSessionError>>
           + 'a
           + use<'a, 'session, I>
    where
        I: IntoIterator<Item = VoiceChatAdvance>,
        I::IntoIter: 'a,
        'session: 'a,
    {
        ticks.into_iter().map(move |tick| self.advance(tick))
    }
}
