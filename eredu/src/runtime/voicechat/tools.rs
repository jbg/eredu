//! Bounded function-channel collection and result injection after native commit.
//!
//! Wire normalization follows the pinned Speech streaming/offline helpers:
//! optional TOOLCALL tags, one object or a list, and JSON-encoded arguments.
//! Schema validation and semantic events use the ordinary facade tool parser.

use std::sync::Arc;

use eredu_architectures::voicechat::{
    channel_decision::ChannelDecisionPolicy, channels::TEXT_VOCABULARY_SIZE,
};
use eredu_core::{
    generation::{FinishReason, SemanticEvent},
    scheduler::RequestId,
    BackendFailure,
};
use serde_json::Value;

use super::{PreparedVoiceChatToolResponse, VoiceChatTextError, VoiceChatTextPreparation};
use crate::runtime::{
    chat::{
        dialect::{
            DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters, ExactEnvelope,
            FormatDialect, GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout,
            DECLARATIVE_DIALECT,
        },
        tool_schema::ToolSchemas,
    },
    generation::streaming::ToolRuntimeParser,
};

const JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    envelope: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
};
const DIALECT: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: false,
    output: ExactEnvelope {
        prefix: "<TOOLCALL>",
        suffix: "</TOOLCALL>",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    json_function: Some(&JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ",",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("<TOOLCALL>"),
    required_structural_tokens: &[],
    stop_sequences: &[],
};

/// Bounds on one incomplete function-channel payload and its completed call batch.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatToolLimits {
    /// Maximum retained payload token IDs, excluding SOTC, EOTC and PAD.
    pub max_payload_tokens: usize,
    /// Maximum decoded payload bytes admitted to JSON and schema parsing.
    pub max_payload_bytes: usize,
    /// Maximum complete calls returned in one committed batch.
    pub max_calls: usize,
}

/// Identity of a published tool batch; external results must carry this exact ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatToolTicket {
    request: RequestId,
    turn: u64,
    batch: u64,
}

/// Request/turn stamp captured when admitting work, before a native invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatTurnIdentity {
    request: RequestId,
    turn: u64,
}
impl VoiceChatTurnIdentity {
    pub(super) const fn with_request(self, request: RequestId) -> Self {
        Self { request, ..self }
    }

    /// Request owning this conversation turn.
    pub const fn request(&self) -> RequestId {
        self.request
    }
    /// Monotonic conversation turn, advanced by interruption.
    pub const fn turn(&self) -> u64 {
        self.turn
    }
}
impl VoiceChatToolTicket {
    pub(super) const fn with_request(self, request: RequestId) -> Self {
        Self { request, ..self }
    }

    /// Request that owns this batch.
    pub const fn request(&self) -> RequestId {
        self.request
    }
    /// Conversation turn, advanced explicitly by interruption.
    pub const fn turn(&self) -> u64 {
        self.turn
    }
    /// Monotonic batch identity within this channel owner.
    pub const fn batch(&self) -> u64 {
        self.batch
    }
}

/// Complete validated calls, published only after the EOTC transition commits.
#[derive(Debug)]
pub struct VoiceChatToolBatch {
    /// Ticket required when submitting the corresponding externally executed results.
    pub ticket: VoiceChatToolTicket,
    /// Semantic start/argument/completion events; no partial or malformed calls.
    pub events: Vec<SemanticEvent>,
}

impl VoiceChatToolBatch {
    /// Additional fork cost beyond copying the source event strings. Rewriting
    /// allocates each complete new ID while the copied old ID is still retained.
    pub(super) fn fork_id_estimate(
        &self,
        request: RequestId,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        let ticket = self.ticket.with_request(request);
        let copy_bytes = self.events.iter().try_fold(0_u64, |total, event| {
            let bytes = match event {
                SemanticEvent::ToolCallStart { index, .. }
                | SemanticEvent::ToolCallCompleted { index, .. } => {
                    tool_call_id_len(ticket, *index) as u64
                }
                _ => 0,
            };
            total.checked_add(bytes).ok_or_else(|| {
                BackendFailure::from_error(
                    eredu_core::execution_control::ExecutionControlError::Overflow,
                )
            })
        })?;
        Ok(eredu_core::execution_control::SnapshotEstimate {
            // The ordinary copy estimate already prices the old strings. Each
            // new allocation temporarily coexists with its copied predecessor.
            retained_bytes: copy_bytes,
            copy_bytes,
        })
    }

    /// Additional retained string bytes required by a fork's request identity.
    /// The queue separately admits changed publication-byte capacity before copying.
    pub(super) fn fork_id_growth(&self, request: RequestId) -> Result<u64, BackendFailure> {
        let ticket = self.ticket.with_request(request);
        self.events.iter().try_fold(0_u64, |total, event| {
            let growth = match event {
                SemanticEvent::ToolCallStart { id, index, .. }
                | SemanticEvent::ToolCallCompleted { id, index, .. } => {
                    tool_call_id_len(ticket, *index).saturating_sub(id.len()) as u64
                }
                _ => 0,
            };
            total.checked_add(growth).ok_or_else(|| {
                BackendFailure::from_error(
                    eredu_core::execution_control::ExecutionControlError::Overflow,
                )
            })
        })
    }

    /// Reattributes only an unpublished, independently copied batch. Delivered
    /// calls are inherited through ticket mappings and must never be emitted again.
    pub(super) fn rekey_for_fork(
        &mut self,
        request: RequestId,
    ) -> Result<(VoiceChatToolTicket, VoiceChatToolTicket), BackendFailure> {
        use std::fmt::Write;
        let source = self.ticket;
        let ticket = self.ticket.with_request(request);
        for event in &mut self.events {
            if let SemanticEvent::ToolCallStart { index, id, .. }
            | SemanticEvent::ToolCallCompleted { index, id, .. } = event
            {
                let mut replacement = String::new();
                replacement
                    .try_reserve_exact(tool_call_id_len(ticket, *index))
                    .map_err(BackendFailure::from_error)?;
                write!(
                    &mut replacement,
                    "voicechat-{}-{}-{}-{}",
                    ticket.request.value(),
                    ticket.turn,
                    ticket.batch,
                    index
                )
                .expect("writing an admitted tool identity to a String cannot fail");
                *id = replacement;
            }
        }
        self.ticket = ticket;
        Ok((source, ticket))
    }
}

fn tool_call_id_len(ticket: VoiceChatToolTicket, index: usize) -> usize {
    fn digits(value: u64) -> usize {
        value.checked_ilog10().map_or(1, |power| power as usize + 1)
    }
    "voicechat-".len()
        + 3
        + digits(ticket.request.value())
        + digits(ticket.turn)
        + digits(ticket.batch)
        + index.checked_ilog10().map_or(1, |power| power as usize + 1)
}

/// Portable tool policy rejection. Rejected input leaves retained policy unchanged.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatToolError {
    /// All caller-supplied bounds must be positive.
    #[error("VoiceChat tool limits must be positive")]
    InvalidLimits,
    /// Declared tool names or schemas are invalid.
    #[error("invalid VoiceChat tools: {0}")]
    Definitions(String),
    /// Retaining or decoding more payload would exceed an explicit bound.
    #[error("VoiceChat tool {resource} capacity {capacity} exhausted")]
    Capacity {
        /// Bounded resource.
        resource: &'static str,
        /// Admitted maximum.
        capacity: usize,
    },
    /// A function-channel token does not belong to the retained vocabulary.
    #[error("invalid VoiceChat function token {0}")]
    Token(u32),
    /// Nested calls, mismatched response feedback, or invalid call framing.
    #[error("invalid VoiceChat tool protocol: {0}")]
    Protocol(String),
    /// A previous published batch still awaits an external result or its injection.
    #[error("VoiceChat tool channel is waiting for the pending result")]
    Waiting,
    /// Result belongs to a completed, interrupted, or different request/turn.
    #[error("stale VoiceChat tool result")]
    StaleResult,
    /// A committed-output callback belongs to an interrupted or different request.
    #[error("stale VoiceChat function-channel observation")]
    StaleObservation,
    /// A function call is incomplete at an explicitly requested boundary.
    #[error("VoiceChat function payload is incomplete")]
    Incomplete,
    /// Identity counters cannot advance without reuse.
    #[error("VoiceChat tool identity exhausted")]
    IdentityExhausted,
    /// Text preparation retains typed ASCII/capacity/tokenization failures.
    #[error(transparent)]
    Text(#[from] VoiceChatTextError),
    /// Host decoding failed, preserving its original source.
    #[error("VoiceChat function decoding failed: {0}")]
    Decode(#[source] BackendFailure),
}

#[derive(Clone)]
enum Phase {
    Idle,
    Collecting(Vec<u32>),
    Waiting(VoiceChatToolTicket),
    Injecting {
        ticket: VoiceChatToolTicket,
        response: Arc<PreparedVoiceChatToolResponse>,
        position: usize,
    },
    AwaitingResponseEnd(VoiceChatToolTicket),
}

/// Host policy for already committed function tokens and explicit external results.
///
/// This owns no neural state, scheduler, wall clock or tool executor. Call
/// `observe_committed_function` only after the shared transition commits. During
/// each turn, the caller delivers committed observations exactly once and in order.
/// Request IDs must be unique across independently created channel owners. During
/// result injection, read `response_policy` before executing a candidate and
/// advance it only by observing the matching committed effective function token.
/// Failed native work therefore consumes no response token. Speech ingestion and
/// perception-only work remain available to the enclosing driver while waiting.
pub struct VoiceChatToolChannel {
    text: Arc<VoiceChatTextPreparation>,
    tools: Arc<[Value]>,
    limits: VoiceChatToolLimits,
    request: RequestId,
    turn: u64,
    next_batch: u64,
    next_index: usize,
    phase: Phase,
}
impl VoiceChatToolChannel {
    /// The enclosing fork admission has already reserved a fresh request ID.
    pub(super) fn rekey_for_fork(&mut self, request: RequestId) {
        self.request = request;
        match &mut self.phase {
            Phase::Waiting(ticket)
            | Phase::Injecting { ticket, .. }
            | Phase::AwaitingResponseEnd(ticket) => {
                *ticket = ticket.with_request(request);
            }
            Phase::Idle | Phase::Collecting(_) => {}
        }
    }

    pub(super) fn estimate_snapshot(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        use eredu_runtime::execution_control::snapshot_value_estimate;
        let phase = match &self.phase {
            Phase::Collecting(tokens) => super::conversation::host_storage::<Vec<u32>>(
                tokens.capacity(),
                std::mem::size_of::<u32>(),
            )?,
            Phase::Injecting { response, .. } => response.snapshot_retention()?,
            _ => super::conversation::host_storage::<Phase>(0, 0)?,
        };
        // Tokenizer and tool schema tables are immutable for the session's entire
        // lifetime. Injected response storage can survive its live phase, so the
        // snapshot separately charges that retained Arc payload.
        snapshot_value_estimate::<Self>([Ok(phase)])
    }

    pub(super) fn copy_snapshot(&self) -> Result<Self, BackendFailure> {
        let phase = if let Phase::Collecting(tokens) = &self.phase {
            let mut copied = Vec::new();
            copied
                .try_reserve_exact(tokens.len())
                .map_err(BackendFailure::from_error)?;
            copied.extend_from_slice(tokens);
            Phase::Collecting(copied)
        } else {
            self.phase.clone()
        };
        Ok(Self {
            text: Arc::clone(&self.text),
            tools: Arc::clone(&self.tools),
            limits: self.limits,
            request: self.request,
            turn: self.turn,
            next_batch: self.next_batch,
            next_index: self.next_index,
            phase,
        })
    }

    /// Stages unpublished policy under the same identity. The candidate is not
    /// an independent channel owner and must remain inside its shared transaction.
    pub(super) fn branch_candidate(&self) -> Self {
        Self {
            text: Arc::clone(&self.text),
            tools: Arc::clone(&self.tools),
            limits: self.limits,
            request: self.request,
            turn: self.turn,
            next_batch: self.next_batch,
            next_index: self.next_index,
            phase: self.phase.clone(),
        }
    }

    /// Retains exact prepared text policy and the ordinary JSON Schema tool definitions.
    pub fn new(
        text: VoiceChatTextPreparation,
        request: RequestId,
        tools: Vec<Value>,
        limits: VoiceChatToolLimits,
    ) -> Result<Self, VoiceChatToolError> {
        if limits.max_payload_tokens == 0 || limits.max_payload_bytes == 0 || limits.max_calls == 0
        {
            return Err(VoiceChatToolError::InvalidLimits);
        }
        ToolSchemas::new(&tools).map_err(VoiceChatToolError::Definitions)?;
        Ok(Self {
            text: Arc::new(text),
            tools: tools.into(),
            limits,
            request,
            turn: 0,
            next_batch: 0,
            next_index: 0,
            phase: Phase::Idle,
        })
    }

    /// Prepares the system prompt with the same definitions used to validate calls.
    pub fn prepare_system_prompt(
        &self,
        system: &str,
        policy: eredu_architectures::voicechat::channels::PromptPolicy,
    ) -> Result<super::PreparedVoiceChatPrompt, VoiceChatTextError> {
        self.text
            .prepare_system_prompt_with_tools(system, &self.tools, policy)
    }

    /// Current pending external batch, including a result being injected.
    pub fn pending(&self) -> Option<VoiceChatToolTicket> {
        match self.phase {
            Phase::Waiting(ticket)
            | Phase::Injecting { ticket, .. }
            | Phase::AwaitingResponseEnd(ticket) => Some(ticket),
            _ => None,
        }
    }

    /// Whether serialization, external waiting or response decoding occupies the
    /// function channel. Collection is active before a complete batch exists.
    pub fn is_active(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }

    /// Captures the identity that must accompany the eventual committed output.
    pub fn turn_identity(&self) -> VoiceChatTurnIdentity {
        VoiceChatTurnIdentity {
            request: self.request,
            turn: self.turn,
        }
    }

    /// Records one committed effective function token. Raw overridden decisions
    /// must not be passed here. PAD is ignored while collecting or waiting.
    pub fn observe_committed_function(
        &mut self,
        identity: VoiceChatTurnIdentity,
        token: u32,
    ) -> Result<Option<VoiceChatToolBatch>, VoiceChatToolError> {
        if identity != self.turn_identity() {
            return Err(VoiceChatToolError::StaleObservation);
        }
        if token as usize >= TEXT_VOCABULARY_SIZE
            || self.text.tokenizer().id_to_token(token).is_none()
        {
            return Err(VoiceChatToolError::Token(token));
        }
        let controls = self.text.controls();
        if let Phase::Injecting {
            ticket,
            response,
            position,
        } = &mut self.phase
        {
            if response.tokens()[*position] as u32 != token {
                return Err(VoiceChatToolError::Protocol(
                    "committed response token differs from prepared injection".into(),
                ));
            }
            *position += 1;
            if *position == response.tokens().len() {
                self.phase = Phase::AwaitingResponseEnd(*ticket);
            }
            return Ok(None);
        }
        if token == controls.pad {
            return Ok(None);
        }
        match &mut self.phase {
            Phase::Idle => {
                if token == controls.tool_call_start {
                    self.phase = Phase::Collecting(Vec::new());
                }
                Ok(None)
            }
            Phase::Waiting(_) => Err(VoiceChatToolError::Waiting),
            Phase::AwaitingResponseEnd(_) => {
                if token == controls.tool_call_start {
                    return Err(VoiceChatToolError::Protocol(
                        "tool call started before response end".into(),
                    ));
                }
                if token == controls.tool_response_end {
                    self.phase = Phase::Idle;
                }
                Ok(None)
            }
            Phase::Collecting(tokens) => {
                if token == controls.tool_call_start {
                    return Err(VoiceChatToolError::Protocol(
                        "nested start-of-tool-call".into(),
                    ));
                }
                if token == controls.tool_call_end {
                    let decoded = self
                        .text
                        .tokenizer()
                        .decode(tokens, false)
                        .map_err(|error| {
                            #[derive(Debug, thiserror::Error)]
                            #[error("{0}")]
                            struct Decode(#[source] tokenizers::Error);
                            VoiceChatToolError::Decode(BackendFailure::from_error(Decode(error)))
                        })?;
                    if decoded.len() > self.limits.max_payload_bytes {
                        return Err(VoiceChatToolError::Capacity {
                            resource: "decoded bytes",
                            capacity: self.limits.max_payload_bytes,
                        });
                    }
                    let ticket = VoiceChatToolTicket {
                        request: self.request,
                        turn: self.turn,
                        batch: self.next_batch,
                    };
                    let next_batch = self
                        .next_batch
                        .checked_add(1)
                        .ok_or(VoiceChatToolError::IdentityExhausted)?;
                    let (events, calls) = parse_calls(
                        &decoded,
                        &self.tools,
                        self.limits.max_calls,
                        self.next_index,
                        ticket,
                    )?;
                    let next_index = self
                        .next_index
                        .checked_add(calls)
                        .ok_or(VoiceChatToolError::IdentityExhausted)?;
                    self.phase = Phase::Waiting(ticket);
                    self.next_batch = next_batch;
                    self.next_index = next_index;
                    return Ok(Some(VoiceChatToolBatch { ticket, events }));
                }
                if tokens.len() == self.limits.max_payload_tokens {
                    return Err(VoiceChatToolError::Capacity {
                        resource: "payload tokens",
                        capacity: self.limits.max_payload_tokens,
                    });
                }
                tokens.push(token);
                Ok(None)
            }
            Phase::Injecting { .. } => unreachable!(),
        }
    }

    /// Accepts an already formatted ASCII result for the exact pending batch.
    /// Tokenizes as supplied, without appending BOS/EOS/EOTR or converting numbers.
    pub fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        response: &str,
    ) -> Result<(), VoiceChatToolError> {
        if !matches!(self.phase, Phase::Waiting(expected) if expected == ticket) {
            return Err(VoiceChatToolError::StaleResult);
        }
        let response = self.text.prepare_tool_response(response)?;
        self.submit_prepared_response(ticket, response)
    }

    pub(super) fn text_preparation(&self) -> &VoiceChatTextPreparation {
        &self.text
    }

    pub(super) fn submit_prepared_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        response: PreparedVoiceChatToolResponse,
    ) -> Result<(), VoiceChatToolError> {
        if !matches!(self.phase, Phase::Waiting(expected) if expected == ticket) {
            return Err(VoiceChatToolError::StaleResult);
        }
        self.phase = Phase::Injecting {
            ticket,
            response: Arc::new(response),
            position: 0,
        };
        Ok(())
    }

    /// Peeks at response decoding policy; reading never consumes a token.
    /// After supplied tokens are exhausted, retains tool decoding with no forced
    /// function token until the model commits its response-end control token.
    /// Use with the existing `Invocation::Tool` and the selected audio source.
    pub fn response_policy(&self) -> Option<ChannelDecisionPolicy> {
        match &self.phase {
            Phase::Injecting {
                response, position, ..
            } => Some(ChannelDecisionPolicy::Tool {
                function_feedback: Some(response.tokens()[*position] as u32),
                spoken_text: None,
            }),
            Phase::AwaitingResponseEnd(_) => Some(ChannelDecisionPolicy::Tool {
                function_feedback: None,
                spoken_text: None,
            }),
            _ => None,
        }
    }

    /// Invalidates incomplete calls and outstanding results for an interrupted turn.
    /// Delivered events and externally executed tools are not undone.
    pub fn interrupt(&mut self) -> Result<u64, VoiceChatToolError> {
        let turn = self
            .turn
            .checked_add(1)
            .ok_or(VoiceChatToolError::IdentityExhausted)?;
        self.turn = turn;
        self.next_index = 0;
        self.phase = Phase::Idle;
        Ok(turn)
    }

    /// Rejects an incomplete function payload without silently resetting it.
    pub fn check_complete_payload(&self) -> Result<(), VoiceChatToolError> {
        if matches!(self.phase, Phase::Collecting(_)) {
            Err(VoiceChatToolError::Incomplete)
        } else {
            Ok(())
        }
    }
}

fn parse_calls(
    text: &str,
    tools: &[Value],
    max_calls: usize,
    index: usize,
    ticket: VoiceChatToolTicket,
) -> Result<(Vec<SemanticEvent>, usize), VoiceChatToolError> {
    let protocol = |message: String| VoiceChatToolError::Protocol(message);
    let text = text.trim();
    let text = if let Some(inner) = text.strip_prefix("<TOOLCALL>") {
        inner
            .strip_suffix("</TOOLCALL>")
            .ok_or_else(|| protocol("missing textual TOOLCALL suffix".into()))?
            .trim()
    } else {
        text
    };
    let value: Value = serde_json::from_str(text).map_err(|error| protocol(error.to_string()))?;
    let mut calls = match value {
        Value::Array(calls) => calls,
        value @ Value::Object(_) => vec![value],
        _ => {
            return Err(protocol(
                "tool payload must be a JSON object or list".into(),
            ))
        }
    };
    if calls.is_empty() {
        return Err(protocol("empty tool-call batch".into()));
    }
    if calls.len() > max_calls {
        return Err(VoiceChatToolError::Capacity {
            resource: "calls",
            capacity: max_calls,
        });
    }
    index
        .checked_add(calls.len())
        .ok_or(VoiceChatToolError::IdentityExhausted)?;
    for call in &mut calls {
        let object = call
            .as_object_mut()
            .ok_or_else(|| protocol("tool call must be an object".into()))?;
        if object.contains_key("arguments") && object.contains_key("parameters") {
            return Err(protocol("ambiguous tool arguments".into()));
        }
        if !object.contains_key("arguments") {
            if let Some(arguments) = object.remove("parameters") {
                object.insert("arguments".into(), arguments);
            }
        }
        if let Some(Value::String(encoded)) = object.get("arguments") {
            let arguments = serde_json::from_str(encoded)
                .map_err(|error| protocol(format!("invalid encoded arguments: {error}")))?;
            object.insert("arguments".into(), arguments);
        }
    }
    let count = calls.len();
    let parser = DECLARATIVE_DIALECT
        .incremental_parser_state(DialectParameters::Declarative(&DIALECT))
        .map_err(protocol)?;
    let mut parser = ToolRuntimeParser::new_with_structural_stops(parser, [], [], [])
        .with_tool_schemas(tools)
        .map_err(protocol)?;
    parser
        .push(&format!(
            "<TOOLCALL>{}</TOOLCALL>",
            serde_json::to_string(&calls).map_err(|error| protocol(error.to_string()))?
        ))
        .map_err(protocol)?;
    parser.finish(FinishReason::Eos).map_err(protocol)?;
    let mut events = parser.take_events();
    events.retain(|event| !matches!(event, SemanticEvent::Finished { .. }));
    for event in &mut events {
        match event {
            SemanticEvent::ToolCallStart {
                index: call_index,
                id,
                ..
            }
            | SemanticEvent::ToolCallCompleted {
                index: call_index,
                id,
                ..
            } => {
                *call_index += index;
                *id = format!(
                    "voicechat-{}-{}-{}-{}",
                    ticket.request.value(),
                    ticket.turn,
                    ticket.batch,
                    *call_index
                );
            }
            SemanticEvent::ToolArgumentsDelta {
                index: call_index, ..
            } => *call_index += index,
            _ => {}
        }
    }
    Ok((events, count))
}
