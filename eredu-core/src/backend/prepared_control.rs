//! Source-derived diagnostic attribution for an opaque ordinary controlled input.

use super::{BackendFailure, HostPreparationAuthority, ModelRuntime, TextGenerationBackend};
use crate::{InputModality, InputPayloadKind, PreparedInputIdentity};
use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};

/// Version of source-position and canonical-token attribution.
pub const PREPARED_PROMPT_ATTRIBUTION_VERSION: u32 = 1;

/// Canonical tokenizer values exist only for a real token-ID payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptTokenAttribution {
    /// Half-open range into the attribution's single canonical ID buffer.
    Canonical { range: [u64; 2] },
    /// Media or projected values have no canonical tokenizer sequence.
    NotTokenized,
}

/// Architecture-derived position projection for one actual admitted part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPromptSegmentPlan {
    pub source_part: u64,
    pub modality: InputModality,
    pub payload: InputPayloadKind,
    /// Positions relative to the opening decoder frontier.
    pub decoder_range: [u64; 2],
}

/// One ordered part's physical decoder attribution and optional canonical IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPromptSegment {
    pub plan: PreparedPromptSegmentPlan,
    pub tokens: PromptTokenAttribution,
}

/// Serializable diagnostic data. This value never grants execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPromptAttribution {
    pub schema_version: u32,
    pub prepared: PreparedInputIdentity,
    pub semantic_content_identity: String,
    pub opening_position: u64,
    pub decoder_positions: u64,
    pub batch: u64,
    pub segments: Vec<PreparedPromptSegment>,
    pub canonical_token_ids: Vec<u32>,
}

/// Typed failure before a prepared source can enter the ordinary driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedControlInputError {
    #[error("prepared controlled input requires unknown original source/control bounds")]
    UnknownBound,
    #[error("prepared-input capture/intervention attribution is not implemented")]
    InstrumentationUnavailable,
    #[error("selected executable has no prepared-input attribution contract")]
    Unsupported,
    #[error("prepared controlled input requires its actual semantic source identity")]
    MissingSourceIdentity,
    #[error("actual selected decoder frontier is unknown")]
    UnknownFrontier,
    #[error("prepared controlled input belongs to a different or changed source/session")]
    SourceMismatch,
    #[error("invalid prepared prompt attribution")]
    InvalidAttribution,
    #[error("prepared prompt attribution arithmetic overflow")]
    Overflow,
}

impl PreparedPromptAttribution {
    /// Validates the exact ordered coverage without executing or certifying a source.
    pub fn validate(&self) -> Result<(), PreparedControlInputError> {
        use PreparedControlInputError as E;
        if self.schema_version != PREPARED_PROMPT_ATTRIBUTION_VERSION
            || self.batch != 1
            || self.decoder_positions == 0
            || self.semantic_content_identity.trim().is_empty()
            || self.segments.len() != self.prepared.parts().len()
        {
            return Err(E::InvalidAttribution);
        }
        self.opening_position
            .checked_add(self.decoder_positions)
            .ok_or(E::Overflow)?;
        let (mut position, mut token) = (0, 0);
        for (index, (segment, actual)) in
            self.segments.iter().zip(self.prepared.parts()).enumerate()
        {
            let plan = &segment.plan;
            if plan.source_part != u64::try_from(index).map_err(|_| E::Overflow)?
                || plan.modality != actual.modality()
                || plan.payload != actual.payload_kind()
                || plan.decoder_range[0] != position
                || plan.decoder_range[1] < position
            {
                return Err(E::InvalidAttribution);
            }
            match &segment.tokens {
                PromptTokenAttribution::Canonical { range } => {
                    if plan.payload != InputPayloadKind::TokenIds
                        || range[0] != token
                        || range[1].checked_sub(token)
                            != plan.decoder_range[1].checked_sub(position)
                    {
                        return Err(E::InvalidAttribution);
                    }
                    token = range[1];
                }
                PromptTokenAttribution::NotTokenized
                    if plan.payload == InputPayloadKind::TokenIds =>
                {
                    return Err(E::InvalidAttribution)
                }
                PromptTokenAttribution::NotTokenized => {}
            }
            position = plan.decoder_range[1];
        }
        if position != self.decoder_positions || token != self.canonical_token_ids.len() as u64 {
            return Err(E::InvalidAttribution);
        }
        Ok(())
    }

    /// A complete canonical prompt exists only when every actual part is tokenized.
    pub fn complete_token_ids(&self) -> Option<&[u32]> {
        self.segments
            .iter()
            .all(|s| matches!(s.tokens, PromptTokenAttribution::Canonical { .. }))
            .then_some(&self.canonical_token_ids)
    }

    /// Exact source coordinates for one ordinary committed prediction.
    pub fn input_range(&self, prediction: u64) -> Result<[u64; 2], PreparedControlInputError> {
        let end = self
            .opening_position
            .checked_add(self.decoder_positions)
            .ok_or(PreparedControlInputError::Overflow)?;
        if prediction == 0 {
            return Ok([self.opening_position, end]);
        }
        let next = end
            .checked_add(prediction)
            .ok_or(PreparedControlInputError::Overflow)?;
        Ok([next - 1, next])
    }

    /// Logical retained metadata; no construction-peak or original bound claim.
    pub fn logical_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64)
            .checked_add(
                self.prepared
                    .logical_metadata_bytes()?
                    .checked_sub(std::mem::size_of::<PreparedInputIdentity>() as u64)?,
            )?
            .checked_add(self.semantic_content_identity.len() as u64)?
            .checked_add(
                (self.segments.len() as u64)
                    .checked_mul(std::mem::size_of::<PreparedPromptSegment>() as u64)?,
            )?
            .checked_add((self.canonical_token_ids.len() as u64).checked_mul(4)?)
    }
}

struct AttributionOwner {
    value: PreparedPromptAttribution,
    // Moves out of the final Arc before payload destruction; retires last.
    _host: HostPreparationAuthority,
}

/// Immutable source metadata retaining the provider's ordinary preparation custody.
/// No raw Arc/Weak, mutable payload or execution authority is exposed.
pub struct SharedPromptAttribution(Option<Arc<AttributionOwner>>);
impl SharedPromptAttribution {
    /// Provider publication after acquiring custody before constructing the payload.
    /// This checks diagnostic structure only; the provider separately authenticates
    /// its closed input and actual session/selection/revision before consumption.
    #[doc(hidden)]
    pub fn from_prepared(
        value: PreparedPromptAttribution,
        host: HostPreparationAuthority,
    ) -> Result<Self, PreparedControlInputError> {
        let owner = AttributionOwner { value, _host: host };
        owner.value.validate()?;
        Ok(Self(Some(Arc::new(owner))))
    }
    pub fn attribution(&self) -> &PreparedPromptAttribution {
        &self.0.as_ref().expect("live attribution").value
    }
    /// Known logical metadata and this concrete shared shell. This does not size
    /// the provider's erased host token or certify a construction peak.
    pub fn logical_storage_bytes(&self) -> Option<u64> {
        let layout = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<AttributionOwner>())
            .ok()?
            .0
            .pad_to_align();
        self.attribution()
            .logical_storage_bytes()?
            .checked_sub(std::mem::size_of::<PreparedPromptAttribution>() as u64)?
            .checked_add(u64::try_from(layout.size()).ok()?)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }
}
impl Clone for SharedPromptAttribution {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live attribution"))))
    }
}
impl Drop for SharedPromptAttribution {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for SharedPromptAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.attribution().fmt(f)
    }
}
impl PartialEq for SharedPromptAttribution {
    fn eq(&self, other: &Self) -> bool {
        self.attribution() == other.attribution()
    }
}
impl Eq for SharedPromptAttribution {}
impl Serialize for SharedPromptAttribution {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.attribution().serialize(s)
    }
}

/// Provider-owned one-use input. Implementations expose only borrowed diagnostics.
pub trait PreparedControlInput: Sized {
    type Prompt;
    fn attribution(&self) -> &PreparedPromptAttribution;
    /// Borrows the closed metadata owner for safe ordinary error retention.
    /// Cloning this value grants no source, execution or resource authority.
    fn shared_attribution(&self) -> &SharedPromptAttribution;
}

/// Optional ordinary prepared-input startup; existing text backend APIs are unchanged.
/// An opaque preparation cannot be consumed twice.
/// ```compile_fail
/// use eredu_core::{ModelRuntime, PreparedControlInputBackend};
/// fn twice<B: PreparedControlInputBackend>(runtime: &ModelRuntime<B>, input: B::ControlInput) {
///     let _ = B::consume_control_input(runtime, input);
///     let _ = B::consume_control_input(runtime, input);
/// }
/// ```
pub trait PreparedControlInputBackend: TextGenerationBackend {
    type ControlInput: PreparedControlInput<Prompt = Self::Prompt>;
    /// Acquire real ordinary host/native custody before new preparation allocations.
    fn prepare_control_input(
        runtime: &ModelRuntime<Self>,
        prompt: Self::Prompt,
    ) -> Result<Self::ControlInput, BackendFailure>;
    /// Bind one ordinarily admitted shared capture source to this exact input.
    /// Default rejection does not promote it to original managed authority.
    fn bind_control_input_capture(
        _runtime: &ModelRuntime<Self>,
        _input: Self::ControlInput,
        _config: super::TextGenerationConfig,
        _source: crate::capture::SharedCapturePlan,
    ) -> Result<Self::ControlInput, BackendFailure> {
        Err(PreparedControlInputError::InstrumentationUnavailable.into_backend_failure())
    }
    /// Authenticate exact source, selected executable and current state revision,
    /// then move the same input and immutable attribution into the shared driver.
    fn consume_control_input(
        runtime: &ModelRuntime<Self>,
        input: Self::ControlInput,
    ) -> Result<(Self::Prompt, SharedPromptAttribution), BackendFailure>;
}

impl PreparedControlInputError {
    /// Fixed typed rejection before ordinary custody or source construction.
    pub fn into_backend_failure(self) -> BackendFailure {
        super::failure::prepared_control_rejection(self)
    }
}

#[cfg(test)]
mod tests;
