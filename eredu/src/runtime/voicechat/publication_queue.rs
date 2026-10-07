//! Reserved storage for committed speech and conversation events.

use std::{
    cell::RefCell,
    collections::{TryReserveError, VecDeque},
    rc::Rc,
};

use eredu_core::BackendFailure;

use super::{
    VoiceChatConversationError, VoiceChatConversationOutput, VoiceChatPublicationCapacity,
};

/// Total retained queue and outstanding-reservation limits for one conversation.
/// Counts bound the host payloads, not neural/native allocations. Diagnostic
/// contents retain their separate observation budget in addition to record count.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatPublicationLimits {
    /// Positive bound on queued publications plus outstanding permits.
    pub max_publications: usize,
    /// Aggregate payload allowance shared by queued output and all permits.
    pub capacity: VoiceChatPublicationCapacity,
}

/// Exact queued payload and reserved allowances, excluding already delivered output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoiceChatPublicationUsage {
    /// Committed publications awaiting delivery.
    pub queued_publications: usize,
    /// Permits held by queued or in-flight work.
    pub reserved_publications: usize,
    /// Actual payload retained in committed publications.
    pub queued: VoiceChatPublicationCapacity,
    /// Maximum payload admitted by outstanding permits.
    pub reserved: VoiceChatPublicationCapacity,
}

/// Portable publication admission, semantic transaction or storage failure.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatPublicationError {
    /// An isolated copy temporarily retains the output queue.
    #[error("VoiceChat publication queue is retained by a snapshot operation")]
    SnapshotBusy,
    /// PCM admission or committed source-audio attribution was rejected.
    #[error(transparent)]
    Input(#[from] super::VoiceChatInputError),
    /// Queue and permits would exceed the explicit bound. No input is consumed.
    #[error("VoiceChat publication {0} capacity exhausted")]
    Capacity(&'static str),
    /// Host policy rejected admission or observation.
    #[error(transparent)]
    Conversation(#[from] VoiceChatConversationError),
    /// A branch, permit or completed output does not belong to this transaction.
    #[error("invalid VoiceChat publication transaction: {0}")]
    Transaction(&'static str),
    /// Queue storage cannot be established before execution.
    #[error("VoiceChat publication storage allocation failed: {0}")]
    Allocation(#[source] TryReserveError),
    /// Neural transaction failure with the original backend source retained.
    #[error("VoiceChat publication neural transaction failed: {0}")]
    Backend(#[source] BackendFailure),
}

mod fork;
mod snapshot;

pub(super) struct PublicationCharge(VoiceChatPublicationCapacity);

struct Queued<D> {
    output: VoiceChatConversationOutput<D>,
    charge: PublicationCharge,
}

pub(super) struct Queue<D> {
    limits: VoiceChatPublicationLimits,
    output: VecDeque<Queued<D>>,
    usage: VoiceChatPublicationUsage,
    delivered: Rc<()>,
    fork_retention: Option<eredu_runtime::execution_control::SnapshotReservation>,
    retained_growth: Option<eredu_runtime::execution_control::SnapshotRetainedGrowth>,
    retained_estimator: Option<RetainedEstimator<D>>,
}

type RetainedEstimator<D> =
    fn(&Queue<D>, Option<&VoiceChatConversationOutput<D>>) -> Result<u64, BackendFailure>;

/// Single delivery owner for committed output, independent of scheduler lifetime.
/// Cancellation and request failure cannot erase already committed publications.
/// Taking an output releases its queue charge; delivered events are never replayed
/// by this owner. Retain the receiver until all desired publications are drained.
pub struct VoiceChatPublicationReceiver<D> {
    queue: Rc<RefCell<Queue<D>>>,
}

impl<D> VoiceChatPublicationReceiver<D> {
    pub(super) fn new(queue: Rc<RefCell<Queue<D>>>) -> Self {
        Self { queue }
    }

    /// Removes the oldest committed output, including after scheduler cancellation.
    /// Returns `None` while an isolated snapshot copy retains the queue. Use
    /// `try_pop_output` to distinguish this temporary fence from an empty queue.
    pub fn pop_output(&mut self) -> Option<VoiceChatConversationOutput<D>> {
        self.try_pop_output().ok().flatten()
    }

    /// Delivers one committed publication, or reports an active snapshot fence.
    pub fn try_pop_output(
        &mut self,
    ) -> Result<Option<VoiceChatConversationOutput<D>>, VoiceChatPublicationError> {
        Queue::pop(&self.queue)
    }

    /// Reports queued payload and credit still owned by pending work.
    pub fn usage(&self) -> VoiceChatPublicationUsage {
        self.queue.borrow().usage()
    }
}

impl<D> Queue<D> {
    pub(super) fn new(
        limits: VoiceChatPublicationLimits,
    ) -> Result<Rc<RefCell<Self>>, VoiceChatPublicationError> {
        if limits.max_publications == 0 {
            return Err(VoiceChatPublicationError::Capacity("publication slots"));
        }
        // Metadata capacity is fixed before any native work: successful commit
        // only moves validated payloads into an already allocated queue slot.
        let mut output = VecDeque::new();
        output
            .try_reserve_exact(limits.max_publications)
            .map_err(VoiceChatPublicationError::Allocation)?;
        Ok(Rc::new(RefCell::new(Self {
            limits,
            output,
            usage: VoiceChatPublicationUsage::default(),
            delivered: Rc::new(()),
            fork_retention: None,
            retained_growth: None,
            retained_estimator: None,
        })))
    }

    pub(super) fn reserve(
        owner: &Rc<RefCell<Self>>,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<Reservation<D>, VoiceChatPublicationError> {
        let mut queue = owner.borrow_mut();
        if queue.usage.reserved_publications
            >= queue.limits.max_publications - queue.usage.queued_publications
        {
            return Err(VoiceChatPublicationError::Capacity("publication slots"));
        }
        let available = queue
            .limits
            .capacity
            .subtract(queue.usage.queued)
            .subtract(queue.usage.reserved);
        for ((requested, name), (available, _)) in
            capacity.fields().into_iter().zip(available.fields())
        {
            if requested > available {
                return Err(VoiceChatPublicationError::Capacity(name));
            }
        }
        queue.usage.reserved = queue.usage.reserved.add(capacity);
        queue.usage.reserved_publications += 1;
        Ok(Reservation {
            queue: Rc::clone(owner),
            capacity,
            active: true,
        })
    }

    pub(super) fn pop(
        owner: &Rc<RefCell<Self>>,
    ) -> Result<Option<VoiceChatConversationOutput<D>>, VoiceChatPublicationError> {
        let mut queue = owner
            .try_borrow_mut()
            .map_err(|_| VoiceChatPublicationError::SnapshotBusy)?;
        let Some(item) = queue.output.pop_front() else {
            return Ok(None);
        };
        queue.delivered = Rc::new(());
        queue.usage.queued = queue.usage.queued.subtract(item.charge.0);
        queue.usage.queued_publications -= 1;
        Ok(Some(item.output))
    }

    pub(super) fn usage(&self) -> VoiceChatPublicationUsage {
        self.usage
    }
}

/// Credits remain reserved while queued work, a native branch or a staged output
/// can still produce a publication. Ownership is never duplicated publicly.
pub(super) struct Reservation<D> {
    queue: Rc<RefCell<Queue<D>>>,
    capacity: VoiceChatPublicationCapacity,
    active: bool,
}

impl<D> Reservation<D> {
    pub(super) fn belongs_to(&self, queue: &Rc<RefCell<Queue<D>>>) -> bool {
        Rc::ptr_eq(&self.queue, queue)
    }

    pub(super) fn prepare(
        &self,
        output: &VoiceChatConversationOutput<D>,
    ) -> Result<PublicationCharge, VoiceChatPublicationError> {
        self.capacity.validate_frame(&output.frame)?;
        let tool_bytes = self.capacity.validate_tools(output.tools.as_ref())?;
        self.capacity
            .validate_text(output.agent_text.as_ref(), output.transcription.as_ref())?;
        self.queue.borrow().validate_publication_retention(output)?;
        Ok(PublicationCharge(VoiceChatPublicationCapacity {
            speech_samples: output
                .frame
                .speech
                .as_ref()
                .map_or(0, |chunk| chunk.samples.len()),
            transcription_tokens: output
                .frame
                .transcription
                .as_ref()
                .map_or(0, |frame| frame.emitted.len()),
            agent_text_bytes: output
                .agent_text
                .as_ref()
                .map_or(0, |event| event.text.len()),
            transcription_text_bytes: output
                .transcription
                .as_ref()
                .map_or(0, |event| event.text.len()),
            audio_codes: output.frame.generated_codes.as_ref().map_or(0, Vec::len),
            tool_events: output.tools.as_ref().map_or(0, |batch| batch.events.len()),
            tool_bytes,
            diagnostics: output.frame.diagnostics.len(),
        }))
    }

    /// Uses the charge checked before neural commit. No parsing, validation or
    /// allocation occurs between neural publication and queued host delivery.
    pub(super) fn commit(
        mut self,
        output: VoiceChatConversationOutput<D>,
        charge: PublicationCharge,
    ) {
        let mut queue = self.queue.borrow_mut();
        queue.usage.reserved_publications -= 1;
        queue.usage.reserved = queue.usage.reserved.subtract(self.capacity);
        queue.usage.queued_publications += 1;
        queue.usage.queued = queue.usage.queued.add(charge.0);
        queue.output.push_back(Queued { output, charge });
        self.active = false;
    }
}

impl<D> Drop for Reservation<D> {
    fn drop(&mut self) {
        if self.active {
            let mut queue = self.queue.borrow_mut();
            queue.usage.reserved_publications -= 1;
            queue.usage.reserved = queue.usage.reserved.subtract(self.capacity);
        }
    }
}

impl VoiceChatPublicationCapacity {
    fn fields(self) -> [(usize, &'static str); 8] {
        [
            (self.speech_samples, "speech samples"),
            (self.transcription_tokens, "transcription tokens"),
            (self.agent_text_bytes, "agent text bytes"),
            (self.transcription_text_bytes, "transcription text bytes"),
            (self.audio_codes, "audio codes"),
            (self.tool_events, "tool events"),
            (self.tool_bytes, "tool bytes"),
            (self.diagnostics, "diagnostics"),
        ]
    }

    // Only called after admission/charge validation establishes component-wise
    // bounds. Integer overflow cannot exceed the configured usize limit.
    fn add(self, other: Self) -> Self {
        Self {
            speech_samples: self.speech_samples + other.speech_samples,
            transcription_tokens: self.transcription_tokens + other.transcription_tokens,
            agent_text_bytes: self.agent_text_bytes + other.agent_text_bytes,
            transcription_text_bytes: self.transcription_text_bytes
                + other.transcription_text_bytes,
            audio_codes: self.audio_codes + other.audio_codes,
            tool_events: self.tool_events + other.tool_events,
            tool_bytes: self.tool_bytes + other.tool_bytes,
            diagnostics: self.diagnostics + other.diagnostics,
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            speech_samples: self.speech_samples - other.speech_samples,
            transcription_tokens: self.transcription_tokens - other.transcription_tokens,
            agent_text_bytes: self.agent_text_bytes - other.agent_text_bytes,
            transcription_text_bytes: self.transcription_text_bytes
                - other.transcription_text_bytes,
            audio_codes: self.audio_codes - other.audio_codes,
            tool_events: self.tool_events - other.tool_events,
            tool_bytes: self.tool_bytes - other.tool_bytes,
            diagnostics: self.diagnostics - other.diagnostics,
        }
    }
}
