//! Portable generation lifecycle and semantic output events.

use serde::{Deserialize, Serialize};

/// Why generation reached a terminal state.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model emitted an end token.
    EndToken,
    /// Caller token limit was reached.
    Length,
    /// A caller stop sequence matched.
    StopSequence,
    /// Generation was cancelled.
    Cancelled,
    /// Backend execution failed.
    Failed,
}

/// Cancellation state independent of any async runtime.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct Cancellation {
    cancelled: bool,
    reason: Option<String>,
}

impl Cancellation {
    /// Requests cancellation with optional caller context.
    pub fn cancel(&mut self, reason: impl Into<String>) {
        self.cancelled = true;
        self.reason = Some(reason.into());
    }
    /// Returns whether cancellation was requested.
    pub const fn is_cancelled(&self) -> bool {
        self.cancelled
    }
    /// Returns caller context.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

/// Backend-independent phase of one autoregressive session.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPhase {
    /// Session exists but prompt work has not been submitted.
    Created,
    /// Prompt prefill is in flight.
    Prefilling,
    /// Session is ready for cached decode.
    Decoding,
    /// Session has terminated.
    Finished,
}

/// Semantic event emitted by generation orchestration.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputEvent {
    /// Prompt state became ready for decoding.
    PrefillCompleted {
        /// Number of prompt tokens consumed.
        prompt_tokens: usize,
    },
    /// One generated token committed.
    Token {
        /// Token id.
        token_id: u32,
        /// Zero-based generated position.
        position: usize,
    },
    /// Generation terminated.
    Finished {
        /// Terminal reason.
        reason: FinishReason,
    },
}
