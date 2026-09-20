//! Paid pending bytes and escaped events over the ordinary channel transitions.
//! Decoder/stop orchestration remains in the shared speculative semantic owner.
use super::{DependencyMemoryPolicy, OriginalSemanticChannelSource};
use eredu_core::{
    HostPreparationAuthority, SpeculativeBuffer, SharedBackendFailure, BackendFailure, BackendFailureKind,
    generation::{SemanticEvent, SemanticText},
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_text::semantic_channels::{self as channels, ChannelKind, Next, State};
use std::mem::{size_of, size_of_val};
use channels::JsonFrame;
use channels::tagged::{TaggedCall, TaggedFrame};
mod tagged;
use tool_call::Call;
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Tool(#[from] SharedBackendFailure),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
    #[error(transparent)]
    Text(#[from] eredu_core::generation::SemanticTextAllocationError),
    #[error("prepared semantic byte/event extent overflow")]
    Overflow,
    #[error("prepared semantic byte/event destination is full")]
    Capacity,
    #[error("prepared semantic parser has finished or failed")]
    Ended,
    #[error("semantic tool-payload state requires its own prepared producer")]
    ToolPayload,
}
/// Fixed parser failure retaining its actual metadata payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalSemanticChannelParserError {
    #[source]
    cause: Cause,
    funding: HostMetadataFunding,
}
impl OriginalSemanticChannelParserError {
    pub(super) fn into_output(self) -> eredu_core::SpeculativeOutputError {
        match self.cause {
            Cause::Tool(error) => eredu_core::SpeculativeOutputError::Retained(error),
            Cause::Funding(error) => eredu_core::SpeculativeOutputError::HostFunding(error.into()),
            Cause::Overflow => eredu_core::SpeculativeOutputError::HostFunding(
                eredu_core::HostMetadataFundingError::Overflow,
            ),
            Cause::Capacity => eredu_core::SpeculativeOutputError::Storage(
                "semantic channel destination exhausted",
            ),
            Cause::Ended => eredu_core::SpeculativeOutputError::Storage(
                "semantic channel parser has finished or failed",
            ),
            Cause::ToolPayload => eredu_core::SpeculativeOutputError::Storage(
                "semantic tool-payload state requires its own prepared producer",
            ),
            Cause::Buffer(_) | Cause::Text(_) => eredu_core::SpeculativeOutputError::Storage(
                "semantic channel destination allocation failed",
            ),
        }
    }
}
/// Closed mutable pending/event destinations. Data retires before actual source
/// and H; every emitted SemanticText independently retains its own H on escape.
#[derive(Debug)]
pub struct OriginalSemanticChannelParser {
    pending: SpeculativeBuffer<u8>,
    events: SpeculativeBuffer<SemanticEvent>,
    call: Option<Call>,
    tagged: Option<TaggedCall>,
    tagged_authority: HostPreparationAuthority,
    tool_index: usize,
    used: usize,
    fed: usize,
    limit: usize,
    dependency_memory: DependencyMemoryPolicy,
    state: Cursor,
    ended: bool,
    failed: bool,
    source: OriginalSemanticChannelSource,
    funding: HostMetadataFunding,
}
#[derive(Debug, Clone, Copy)]
enum Cursor {
    Tool(JsonFrame),
    Tagged(TaggedFrame),
    Prefilled(ChannelKind),
    Outside,
    Channel(ChannelKind),
}
impl Cursor {
    fn from_state(state: State<'_>) -> Self {
        match state {
            State::Prefilled { kind, .. } => Self::Prefilled(kind),
            State::Outside => Self::Outside,
            State::Channel { kind, .. } => Self::Channel(kind),
        }
    }
    fn kind(self) -> Option<ChannelKind> {
        match self {
            Self::Tool(_) | Self::Tagged(_) => None,
            Self::Outside => Some(ChannelKind::Text),
            Self::Prefilled(k) | Self::Channel(k) => Some(k),
        }
    }
    fn state<'a>(self, program: &channels::ChannelProgram<'a>) -> State<'a> {
        let suffix = |kind| match kind {
            ChannelKind::Reasoning => {
                program
                    .reasoning_channel
                    .expect("retained reasoning channel")
                    .suffix
            }
            ChannelKind::Text => program.text_channel.expect("retained text channel").suffix,
        };
        match self {
            Self::Tool(_) | Self::Tagged(_) => unreachable!("tool state uses the shared JSON worker"),
            Self::Outside => State::Outside,
            Self::Prefilled(kind) => State::Prefilled {
                kind,
                suffix: suffix(kind),
            },
            Self::Channel(kind) => State::Channel {
                kind,
                suffix: suffix(kind),
            },
        }
    }
}
impl OriginalSemanticChannelParser {
    fn failure(&self, cause: Cause) -> OriginalSemanticChannelParserError {
        OriginalSemanticChannelParserError {
            cause,
            funding: self.funding.clone(),
        }
    }
    fn frames() -> Option<usize> {
        let parts = [
            channels::control_bytes()?,
            channels::json_frame_control_bytes()?,
            tool::control_bytes()?,
            tagged::controls()?,
            size_of::<Option<Call>>(), size_of::<JsonFrame>(),
            OriginalSemanticChannelSource::control_bytes()?,
            size_of::<Self>(),
            size_of::<OriginalSemanticChannelParserError>(),
            size_of::<Cause>(),
            size_of::<Result<Self, OriginalSemanticChannelParserError>>(),
            size_of::<Result<(), OriginalSemanticChannelParserError>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<(
                &OriginalSemanticChannelSource,
                usize,
                &HostMetadataFunding,
            )>(),
            size_of::<(&mut Self, &str)>(),
            size_of::<std::str::Utf8Error>(),
            size_of::<Result<&str, std::str::Utf8Error>>(),
            size_of::<Result<SpeculativeBuffer<SemanticEvent>, OriginalSemanticChannelParserError>>(
            ),
            size_of::<Result<SemanticText, eredu_core::generation::SemanticTextAllocationError>>(),
            size_of::<Result<(), eredu_core::generation::GenerationError>>(),
            size_of::<std::iter::Take<std::iter::Repeat<u8>>>(),
            size_of::<std::iter::Cloned<std::slice::Iter<'_, SemanticEvent>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn buffer<T>(
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<SpeculativeBuffer<T>, Cause> {
        let bytes = SpeculativeBuffer::<T>::retained_control_bytes(capacity)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .and_then(|n| n.checked_add(size_of::<Result<SpeculativeBuffer<T>, Cause>>()))
            .and_then(|n| n.checked_add(size_of::<(usize, &HostMetadataFunding)>()))
            .ok_or(Cause::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(SpeculativeBuffer::try_new_retained(
            capacity,
            HostPreparationAuthority::retain(funding.clone()),
        )?)
    }
    /// Builds finite pending/event storage after the original program exists.
    /// Event capacity follows the byte extent: every nonempty emitted fragment
    /// consumes at least one input byte, plus one terminal event.
    pub fn prepare(
        source: &OriginalSemanticChannelSource,
        input_bytes: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalSemanticChannelParserError> {
        Self::prepare_with_dependency_memory(source, input_bytes, funding, DependencyMemoryPolicy::default())
    }
    /// Uses configurable host-dependency headroom in addition to exact pending
    /// and event-buffer admission. Tagged JSON work receives one estimate for
    /// the full input limit; each independently mutable copy reserves its own.
    pub fn prepare_with_dependency_memory(
        source: &OriginalSemanticChannelSource,
        input_bytes: usize,
        funding: &HostMetadataFunding,
        dependency_memory: DependencyMemoryPolicy,
    ) -> Result<Self, OriginalSemanticChannelParserError> {
        let retain = |cause| OriginalSemanticChannelParserError {
            cause,
            funding: funding.clone(),
        };
        funding
            .reserve_metadata(Self::frames().ok_or_else(|| retain(Cause::Overflow))?)
            .map_err(|cause| retain(cause.into()))?;
        Self::create(source, input_bytes, funding, dependency_memory).map_err(retain)
    }
    fn create(
        source: &OriginalSemanticChannelSource,
        input_bytes: usize,
        funding: &HostMetadataFunding,
        dependency_memory: DependencyMemoryPolicy,
    ) -> Result<Self, Cause> {
        if source.tagged_tools().is_some() {
            funding.reserve_metadata(dependency_memory.estimate(input_bytes).ok_or(Cause::Overflow)?)?;
        }
        let mut pending = Self::buffer(input_bytes, funding)?;
        pending
            .try_extend(std::iter::repeat(0).take(input_bytes))
            .map_err(|_| Cause::Capacity)?;
        let events = Self::buffer(input_bytes.checked_add(1).ok_or(Cause::Overflow)?, funding)?;
        Ok(Self {
            pending,
            events,
            call: None,
            tagged: None,
            tagged_authority: HostPreparationAuthority::unmanaged(),
            tool_index: 0,
            used: 0,
            fed: 0,
            limit: input_bytes,
            dependency_memory,
            state: Cursor::from_state(State::initial(&source.program())),
            ended: false,
            failed: false,
            source: source.clone(),
            funding: funding.clone(),
        })
    }
    fn prepare_events(&mut self) -> Result<(), OriginalSemanticChannelParserError> {
        self.funding
            .reserve_metadata(Self::frames().ok_or_else(|| self.failure(Cause::Overflow))?)
            .map_err(|cause| self.failure(cause.into()))?;
        if self.events.capacity() == 0 {
            self.events = Self::buffer(
                self.limit
                    .checked_add(1)
                    .ok_or_else(|| self.failure(Cause::Overflow))?,
                &self.funding,
            )
            .map_err(|cause| self.failure(cause))?;
        }
        Ok(())
    }
    fn emit(&mut self, kind: ChannelKind, count: usize) -> Result<(), Cause> {
        if count == 0 {
            return Ok(());
        }
        if self.events.len() == self.events.capacity() {
            return Err(Cause::Capacity);
        }
        let bytes = SemanticText::retained_control_bytes(count)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .ok_or(Cause::Overflow)?;
        self.funding.reserve_metadata(bytes)?;
        let text =
            std::str::from_utf8(&self.pending[..count]).expect("shared UTF-8 channel boundary");
        let text = SemanticText::try_copy_retained(
            text,
            HostPreparationAuthority::retain(self.funding.clone()),
        )?;
        self.events
            .try_push(match kind {
                ChannelKind::Reasoning => SemanticEvent::ReasoningDelta(text),
                ChannelKind::Text => SemanticEvent::TextDelta(text),
            })
            .map_err(|_| Cause::Capacity)
    }
    fn process(&mut self) -> Result<(), Cause> {
        loop {
            if let Cursor::Tool(frame) = self.state {
                if self.process_tool(frame)? { return Ok(()); }
                continue;
            }
            if let Cursor::Tagged(frame) = self.state {
                if self.process_tagged(frame)? { return Ok(()); }
                continue;
            }
            let pending = std::str::from_utf8(&self.pending[..self.used])
                .expect("UTF-8 appends and shared ranges");
            let program = self.source.program();
            let step = channels::step(&program, self.state.state(&program), pending);
            // Drop the source borrow before mutating destinations or events.
            let (kind, emit, consume, next, wait) = (
                step.kind,
                step.emit,
                step.consume,
                step.next.map(|next| match next {
                    Next::Channel(state) => Ok(Cursor::from_state(state)),
                    Next::Tool => Err(()),
                }),
                step.wait,
            );
            self.emit(kind, emit)?;
            self.pending.copy_within(consume..self.used, 0);
            self.used -= consume;
            if let Some(next) = next {
                match next {
                    Ok(state) => self.state = state,
                    Err(()) => {
                        if self.source.tool_validation().is_none() { return Err(Cause::ToolPayload); }
                        self.state = if let Some(tools) = self.source.json_tools() { Cursor::Tool(JsonFrame::after_channel(tools)) }
                        else if let Some(tools) = self.source.tagged_tools() { Cursor::Tagged(TaggedFrame::after_channel(tools)) }
                        else { return Err(Cause::ToolPayload); };
                    },
                }
            }
            if wait {
                return Ok(());
            }
        }
    }
    /// Appends within the single admitted input extent and runs the same worker.
    /// A failing transition seals this destination while retaining its prefix.
    pub fn push(&mut self, text: &str) -> Result<(), OriginalSemanticChannelParserError> {
        if self.ended {
            return Err(self.failure(Cause::Ended));
        }
        self.prepare_events()?;
        let result = (|| {
            let fed = self.fed.checked_add(text.len()).ok_or(Cause::Overflow)?;
            let used = self.used.checked_add(text.len()).ok_or(Cause::Overflow)?;
            if fed > self.limit || used > self.limit {
                return Err(Cause::Capacity);
            }
            self.pending[self.used..used].copy_from_slice(text.as_bytes());
            self.fed = fed;
            self.used = used;
            self.process()
        })();
        if result.is_err() {
            self.ended = true;
            self.failed = true;
        }
        result.map_err(|cause| self.failure(cause))
    }
    /// Flushes only channel lookbehind. The shared outer semantic stream owns
    /// stop matching, Finished events and cancellation policy.
    pub fn finish(&mut self) -> Result<(), OriginalSemanticChannelParserError> {
        if self.failed {
            return Err(self.failure(Cause::Ended));
        }
        if self.ended {
            return Ok(());
        }
        self.prepare_events()?;
        let result = self
            .process()
            .and_then(|()| match self.state.kind() {
                Some(kind) => self.emit(kind, self.used),
                // Same ordinary finish semantics: an incomplete JSON envelope
                // produces no fabricated ToolCallEnd or visible text.
                None => {
                    if matches!(self.state, Cursor::Tagged(TaggedFrame::Payload | TaggedFrame::AfterPayload))
                        || (matches!(self.state, Cursor::Tagged(TaggedFrame::AfterEnvelope)) && self.used != 0) {
                        Err(self.tool_failure(tool::ToolCause::Incomplete))
                    } else { Ok(()) }
                },
            });
        self.ended = true;
        self.failed = result.is_err();
        if result.is_ok() {
            self.used = 0;
        }
        result.map_err(|cause| self.failure(cause))
    }
    /// Cancellation discards pending bytes without manufacturing closure output.
    pub fn cancel(&mut self) {
        self.ended = true;
        self.used = 0;
    }
    /// Independent fixed buffers plus configured tagged JSON/session headroom.
    /// Dependency internals are estimated, not measured by this quote.
    pub fn copy_bytes(&self) -> Option<usize> {
        let parts = [
            Self::frames()?,
            self.call.as_ref().map_or(Some(0), Call::copy_bytes)?,
            if self.source.tagged_tools().is_some() { self.dependency_memory.estimate(self.limit)? } else { 0 },
            SpeculativeBuffer::<u8>::retained_control_bytes(self.limit)?,
            SpeculativeBuffer::<SemanticEvent>::retained_control_bytes(self.limit.checked_add(1)?)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Independent mutable state, charged to the supplied original metadata account.
    pub fn copy(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalSemanticChannelParserError> {
        let retain = |cause| OriginalSemanticChannelParserError {
            cause,
            funding: funding.clone(),
        };
        let bytes = self
            .copy_bytes()
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .ok_or_else(|| retain(Cause::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| retain(cause.into()))?;
        let mut copied =
            self.copy_prepaid(HostPreparationAuthority::retain(funding.clone()), true)?;
        copied.funding = funding.clone();
        if let Some(call) = &mut copied.call { call.rebind_funding(funding); }
        Ok(copied)
    }
    pub(super) fn copy_prepaid(
        &self,
        host: HostPreparationAuthority,
        keep_events: bool,
    ) -> Result<Self, OriginalSemanticChannelParserError> {
        let result = (|| -> Result<Self, Cause> {
            let mut pending = SpeculativeBuffer::try_new_retained(self.limit, host.clone())?;
            pending
                .try_extend(self.pending.iter().copied())
                .map_err(|_| Cause::Capacity)?;
            let mut events = SpeculativeBuffer::try_new_retained(
                self.limit.checked_add(1).ok_or(Cause::Overflow)?,
                host.clone(),
            )?;
            if keep_events {
                events
                    .try_extend(self.events.iter().cloned())
                    .map_err(|_| Cause::Capacity)?;
            }
            let call = self.call.as_ref().map(|call| call.copy_prepaid(host.clone(), &self.funding))
                .transpose().map_err(|cause| tool::retained(tool::ToolCause::Call(cause),
                    SpeculativeBuffer::default(), None, None, HostPreparationAuthority::unmanaged(), &self.source, &self.funding))?;
            let tagged = self.tagged.clone();
            Ok(Self {
                pending,
                events,
                call,
                tagged,
                tagged_authority: host,
                tool_index: self.tool_index,
                used: self.used,
                fed: self.fed,
                limit: self.limit,
                dependency_memory: self.dependency_memory,
                state: self.state,
                ended: self.ended,
                failed: self.failed,
                source: self.source.clone(),
                funding: self.funding.clone(),
            })
        })();
        result.map_err(|cause| self.failure(cause))
    }
    pub(super) fn source(&self) -> &OriginalSemanticChannelSource {
        &self.source
    }
    pub(super) fn finish_reason(
        &mut self,
        reason: eredu_core::FinishReason,
    ) -> Result<(), OriginalSemanticChannelParserError> {
        if self.failed {
            return Err(self.failure(Cause::Ended));
        }
        if self.ended {
            return Ok(());
        }
        self.finish()?;
        self.events
            .try_push(SemanticEvent::Finished { reason })
            .map_err(|_| self.failure(Cause::Capacity))
    }
    pub(super) fn cancel_reason(&mut self) -> Result<(), OriginalSemanticChannelParserError> {
        if self.failed {
            return Err(self.failure(Cause::Ended));
        }
        if self.ended {
            return Ok(());
        }
        self.prepare_events()?;
        self.cancel();
        self.events
            .try_push(SemanticEvent::Finished {
                reason: eredu_core::FinishReason::Cancelled,
            })
            .map_err(|_| self.failure(Cause::Capacity))
    }
    pub(super) fn is_finished(&self) -> bool {
        self.ended
    }
    pub(super) fn publish(&mut self, emit: &mut dyn FnMut(SemanticEvent)) {
        for event in self.events.drain() {
            emit(event);
        }
    }
    pub(super) fn drain_owner(&mut self) -> SpeculativeBuffer<SemanticEvent> {
        std::mem::take(&mut self.events)
    }
    /// Pays replacement storage before moving out the original event owner.
    pub fn take_events(
        &mut self,
    ) -> Result<SpeculativeBuffer<SemanticEvent>, OriginalSemanticChannelParserError> {
        let replacement = Self::buffer(
            self.limit
                .checked_add(1)
                .ok_or_else(|| self.failure(Cause::Overflow))?,
            &self.funding,
        )
        .map_err(|cause| self.failure(cause))?;
        Ok(std::mem::replace(&mut self.events, replacement))
    }
}

pub(super) mod tool_call;

mod tool;
