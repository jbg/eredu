//! Bounded host text preparation over the exact retained VoiceChat tokenizer.

use eredu_architectures::voicechat::{
    channels::{
        ChannelError, PromptPolicy, TEXT_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID, TEXT_VOCABULARY_SIZE,
    },
    host::{PreparedHostConditioning, TextControls},
};
use eredu_core::BackendFailure;
use eredu_text::tokenizer::Tokenizer;

/// Host text whose ASCII and capacity requirements are being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatTextKind {
    /// Optional text used during the explicit prompt-prefill invocation.
    SystemPrompt,
    /// Already formatted text injected into the function channel.
    ToolResponse,
}

/// Explicit bounds for each submitted text value, before native allocation.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatTextLimits {
    /// Maximum input bytes, checked before tokenization.
    pub max_bytes: usize,
    /// Maximum unframed token IDs retained from one encoding.
    pub max_tokens: usize,
}

/// Invalid text or host tokenization; all errors are backend independent.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatTextError {
    /// Input and encoded capacities must both be positive.
    #[error("VoiceChat text {0} limit must be positive")]
    InvalidLimit(&'static str),
    /// The released text/character path accepts ASCII input only.
    #[error("VoiceChat {kind:?} contains non-ASCII text at byte {byte_offset}")]
    NonAscii {
        /// Input being validated.
        kind: VoiceChatTextKind,
        /// Offset of the first non-ASCII byte in the original input.
        byte_offset: usize,
    },
    /// Input is too large to pass to the host tokenizer.
    #[error("VoiceChat {kind:?} requires {required} bytes, capacity is {capacity}")]
    ByteCapacity {
        /// Input being validated.
        kind: VoiceChatTextKind,
        /// Original input byte count.
        required: usize,
        /// Configured input limit.
        capacity: usize,
    },
    /// The bounded system/tool renderer cannot fit the complete prompt.
    #[error("VoiceChat rendered system/tool prompt exceeds {capacity} bytes")]
    RenderedPromptCapacity {
        /// Maximum rendered bytes; rendering stops at the first overflow.
        capacity: usize,
    },
    /// Tool definitions cannot satisfy the shared schema or released template.
    #[error("invalid VoiceChat tool definitions: {0}")]
    Tools(String),
    /// Tokenization exceeds the caller's unframed retention bound.
    #[error("VoiceChat {kind:?} requires {required} tokens, capacity is {capacity}")]
    TokenCapacity {
        /// Input being validated.
        kind: VoiceChatTextKind,
        /// Complete encoding size without truncation.
        required: usize,
        /// Configured token retention limit.
        capacity: usize,
    },
    /// Empty or whitespace-only tool results have no channel payload.
    #[error("VoiceChat tool response must contain text")]
    EmptyToolResponse,
    /// An emitted ID cannot be consumed by the shared agent/function embeddings.
    #[error("VoiceChat {kind:?} emits out-of-domain token {token}")]
    TokenDomain {
        /// Input being validated.
        kind: VoiceChatTextKind,
        /// Out-of-domain token ID.
        token: u32,
    },
    /// Retained controls must agree with the released framing and vocabulary.
    #[error("invalid VoiceChat tokenizer control {0}")]
    Control(&'static str),
    /// Architecture-owned prompt framing or repetition cannot be admitted.
    #[error(transparent)]
    Prompt(#[from] ChannelError),
    /// Host tokenizer failure with its original source preserved.
    #[error("VoiceChat tokenization failed: {0}")]
    Tokenizer(#[source] BackendFailure),
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TokenizerFailure(#[source] tokenizers::Error);

fn tokenizer_failure(error: tokenizers::Error) -> VoiceChatTextError {
    VoiceChatTextError::Tokenizer(
        BackendFailure::from_error(TokenizerFailure(error))
            .with_operation("VoiceChat text preparation"),
    )
}

/// Immutable prompt IDs ready for the architecture's explicit prefill invocation.
#[derive(Debug, Clone)]
pub struct PreparedVoiceChatPrompt {
    tokens: Vec<i32>,
    policy: Option<PromptPolicy>,
}

impl PreparedVoiceChatPrompt {
    /// Unframed text IDs; the architecture adds BOS/EOS and repetition once.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Input for `Invocation::Prefill`; whitespace-only prompts omit text prefill.
    pub fn invocation_prompt(&self) -> Option<(&[i32], PromptPolicy)> {
        self.policy.map(|policy| (self.tokens.as_slice(), policy))
    }
}

/// Already formatted function-channel response, encoded without added markers.
#[derive(Debug, Clone)]
pub struct PreparedVoiceChatToolResponse {
    tokens: Vec<i32>,
}

impl PreparedVoiceChatToolResponse {
    pub(super) fn snapshot_retention(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        // Two Arc counters accompany the immutable allocation retained by saved
        // waits/responses after the live channel has advanced past it.
        let mut estimate = super::conversation::host_storage::<(Self, [usize; 2])>(
            self.tokens.capacity(),
            std::mem::size_of::<i32>(),
        )?;
        estimate.copy_bytes = 0;
        Ok(estimate)
    }

    /// Exact response IDs, with no synthetic BOS, EOS, or response-end token.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }
}

/// Host preparation over an owned copy of the exact retained tokenizer.
///
/// Padding and truncation are disabled on this copy. Bounds produce errors
/// instead of silently modifying text. No artifact or network access occurs.
pub struct VoiceChatTextPreparation {
    tokenizer: Tokenizer,
    controls: TextControls,
    limits: VoiceChatTextLimits,
}

impl VoiceChatTextPreparation {
    /// Clones admitted host tokenization and validates explicit input limits.
    pub fn new(
        host: &PreparedHostConditioning,
        limits: VoiceChatTextLimits,
    ) -> Result<Self, VoiceChatTextError> {
        Self::from_tokenizer(&host.tokenizer, host.tokens, limits)
    }

    pub(crate) fn from_tokenizer(
        tokenizer: &Tokenizer,
        controls: TextControls,
        limits: VoiceChatTextLimits,
    ) -> Result<Self, VoiceChatTextError> {
        if limits.max_bytes == 0 {
            return Err(VoiceChatTextError::InvalidLimit("byte"));
        }
        if limits.max_tokens == 0 {
            return Err(VoiceChatTextError::InvalidLimit("token"));
        }
        for (name, actual, expected) in [
            ("BOS", controls.bos, TEXT_BOS_ID as u32),
            ("EOS", controls.eos, TEXT_EOS_ID as u32),
            ("PAD", controls.pad, TEXT_PAD_ID as u32),
        ] {
            if actual != expected {
                return Err(VoiceChatTextError::Control(name));
            }
        }
        for (name, id) in [
            ("BOS", controls.bos),
            ("EOS", controls.eos),
            ("PAD", controls.pad),
            ("listening", controls.listening),
            ("user BOS", controls.user_bos),
            ("user EOS", controls.user_eos),
            ("tool call start", controls.tool_call_start),
            ("tool call end", controls.tool_call_end),
            ("tool response end", controls.tool_response_end),
        ] {
            if id as usize >= TEXT_VOCABULARY_SIZE
                || tokenizer
                    .id_to_token(id)
                    .and_then(|token| tokenizer.token_to_id(&token))
                    != Some(id)
            {
                return Err(VoiceChatTextError::Control(name));
            }
        }
        let mut tokenizer = Tokenizer::from_tokenizer((**tokenizer).clone());
        tokenizer.with_truncation(None).map_err(tokenizer_failure)?;
        tokenizer.with_padding(None);
        Ok(Self {
            tokenizer,
            controls,
            limits,
        })
    }

    /// Exact admitted controls used by function-channel orchestration.
    pub const fn controls(&self) -> TextControls {
        self.controls
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encodes unframed prompt text and admits architecture-owned framing.
    ///
    /// Whitespace-only input omits text prefill; fixed speaker startup remains
    /// the responsibility of the ordinary architecture prefill invocation.
    pub fn prepare_system_prompt(
        &self,
        text: &str,
        policy: PromptPolicy,
    ) -> Result<PreparedVoiceChatPrompt, VoiceChatTextError> {
        self.validate_input(text, VoiceChatTextKind::SystemPrompt)?;
        if text.trim().is_empty() {
            return Ok(PreparedVoiceChatPrompt {
                tokens: Vec::new(),
                policy: None,
            });
        }
        let tokens = self.encode(text, VoiceChatTextKind::SystemPrompt)?;
        policy.framed_length(&tokens)?;
        Ok(PreparedVoiceChatPrompt {
            tokens,
            policy: Some(policy),
        })
    }

    /// Renders the released system/tool template before bounded tokenization.
    ///
    /// Definitions use the shared JSON-schema tool contract. The system text
    /// must be ASCII, and the complete rendered prompt obeys the same byte,
    /// encoded-token and architecture-framing limits as ordinary prompt text.
    pub fn prepare_system_prompt_with_tools(
        &self,
        system: &str,
        tools: &[serde_json::Value],
        policy: PromptPolicy,
    ) -> Result<PreparedVoiceChatPrompt, VoiceChatTextError> {
        self.validate_input(system, VoiceChatTextKind::SystemPrompt)?;
        let rendered = super::prompt::render_system_prompt(system, tools, self.limits.max_bytes)
            .map_err(|error| match error {
                super::prompt::PromptError::Capacity { capacity } => {
                    VoiceChatTextError::RenderedPromptCapacity { capacity }
                }
                super::prompt::PromptError::Tools(message) => VoiceChatTextError::Tools(message),
            })?;
        self.prepare_system_prompt(&rendered, policy)
    }

    /// Encodes an already formatted `<TOOL_RESPONSE>...</TOOL_RESPONSE>` string.
    ///
    /// Formatting, escaping and number conversion belong to the caller. This
    /// method preserves the supplied text and adds no special or end markers.
    pub fn prepare_tool_response(
        &self,
        text: &str,
    ) -> Result<PreparedVoiceChatToolResponse, VoiceChatTextError> {
        self.validate_input(text, VoiceChatTextKind::ToolResponse)?;
        if text.trim().is_empty() {
            return Err(VoiceChatTextError::EmptyToolResponse);
        }
        let tokens = self.encode(text, VoiceChatTextKind::ToolResponse)?;
        if tokens.is_empty() {
            return Err(VoiceChatTextError::EmptyToolResponse);
        }
        Ok(PreparedVoiceChatToolResponse { tokens })
    }

    fn validate_input(
        &self,
        text: &str,
        kind: VoiceChatTextKind,
    ) -> Result<(), VoiceChatTextError> {
        if text.len() > self.limits.max_bytes {
            return Err(VoiceChatTextError::ByteCapacity {
                kind,
                required: text.len(),
                capacity: self.limits.max_bytes,
            });
        }
        if let Some(byte_offset) = text.bytes().position(|byte| !byte.is_ascii()) {
            return Err(VoiceChatTextError::NonAscii { kind, byte_offset });
        }
        Ok(())
    }

    fn encode(&self, text: &str, kind: VoiceChatTextKind) -> Result<Vec<i32>, VoiceChatTextError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_failure)?;
        let ids = encoding.get_ids();
        if ids.len() > self.limits.max_tokens {
            return Err(VoiceChatTextError::TokenCapacity {
                kind,
                required: ids.len(),
                capacity: self.limits.max_tokens,
            });
        }
        if let Some(&token) = ids
            .iter()
            .find(|&&token| token as usize >= TEXT_VOCABULARY_SIZE)
        {
            return Err(VoiceChatTextError::TokenDomain { kind, token });
        }
        Ok(ids.iter().map(|&id| id as i32).collect())
    }
}

#[cfg(test)]
mod tests;
