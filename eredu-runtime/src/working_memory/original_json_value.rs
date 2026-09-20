//! Source-bound JSON validation with stock serde and retained host admission.
use super::DependencyMemoryPolicy;
use eredu_core::{HostPreparationAuthority, SemanticText, SemanticTextAllocationError};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use serde::Deserialize;
use std::mem::size_of;

pub use eredu_text::json_fragments::JsonValueKind as OriginalJsonValueKind;
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Syntax(#[from] serde_json::Error),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Text(#[from] SemanticTextAllocationError),
    #[error("JSON validation admission estimate overflow")]
    Overflow,
}

#[cfg(test)]
#[path = "original_json_value/tests.rs"]
mod tests;
/// A syntax, destination or funding refusal retaining a decoded root string.
/// A trailing-input error does not discard that string's original custody.
#[derive(Debug)]
pub struct OriginalJsonValueError {
    cause: Cause,
    partial: Option<SemanticText>,
    funding: HostMetadataFunding,
}
impl std::fmt::Display for OriginalJsonValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalJsonValueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Funding(error) => error,
            Cause::Syntax(error) => error,
            Cause::Text(error) => error,
            Cause::Overflow => &self.cause,
        })
    }
}
/// Validated root and exact borrowed input. The certificate cannot be detached
/// from the original JSON bytes, and its decoded root string retains its payer.
#[derive(Debug)]
pub struct OriginalJsonValue<'a> {
    input: &'a str,
    kind: OriginalJsonValueKind,
    text: Option<SemanticText>,
    funding: HostMetadataFunding,
}
impl<'a> OriginalJsonValue<'a> {
    /// Validates all input through stock serde_json under default dependency
    /// headroom. A root string additionally pays for its first-party destination.
    pub fn parse(
        input: &'a str,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonValueError> {
        Self::parse_with_memory_policy(input, funding, DependencyMemoryPolicy::default())
    }
    /// Sets upstream scratch headroom. This is an estimate, not a bound on the
    /// temporary JSON value's allocation or a process-wide memory ceiling.
    pub fn parse_with_memory_policy(
        input: &'a str,
        funding: &HostMetadataFunding,
        policy: DependencyMemoryPolicy,
    ) -> Result<Self, OriginalJsonValueError> {
        let mut text = None;
        let result = (|| -> Result<OriginalJsonValueKind, Cause> {
            let bytes = policy
                .estimate(input.len())
                .and_then(|n| {
                    n.checked_add(size_of::<Self>() + size_of::<OriginalJsonValueError>())
                })
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(bytes)?;
            let mut parser = serde_json::Deserializer::from_str(input);
            let value = serde_json::Value::deserialize(&mut parser)?;
            let kind = OriginalJsonValueKind::of(&value);
            if let Some(value) = value.as_str() {
                let bytes = SemanticText::retained_control_bytes(value.len())
                    .and_then(|n| {
                        n.checked_add(HostPreparationAuthority::retention_bytes::<
                            HostMetadataFunding,
                        >()?)
                    })
                    .ok_or(Cause::Overflow)?;
                funding.reserve_metadata(bytes)?;
                text = Some(SemanticText::try_copy_retained(
                    value,
                    HostPreparationAuthority::retain(funding.clone()),
                )?);
            }
            parser.end()?;
            Ok(kind)
        })();
        match result {
            Ok(kind) => Ok(Self {
                input,
                kind,
                text,
                funding: funding.clone(),
            }),
            Err(cause) => Err(OriginalJsonValueError {
                cause,
                partial: text,
                funding: funding.clone(),
            }),
        }
    }
    /// Exact bytes whose complete syntax was validated.
    pub fn source(&self) -> &'a str {
        self.input
    }
    /// Root shape from the actual parser.
    pub fn kind(&self) -> OriginalJsonValueKind {
        self.kind
    }
    /// Decoded root string, absent for every other JSON type.
    pub fn string(&self) -> Option<&str> {
        self.text.as_ref().map(SemanticText::as_str)
    }
    /// Transfers the same paid root-string owner to a semantic field or event.
    pub fn into_string(self) -> Option<SemanticText> {
        self.text
    }
}
