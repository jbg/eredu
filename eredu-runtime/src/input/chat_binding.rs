//! Exact original-render association over the existing host-input compiler.
use super::{
    OriginalModelInput, OriginalModelInputBackend,
    host::{
        HostInputPart, HostInputPartView, HostInputPlanError, HostTensorValues,
        PreparedHostInputPlan,
    },
};
use crate::working_memory::{
    BoundCompositeSemanticStorage, CompositeChatProjection, CompositeSemanticPartRecord,
    OriginalChatBackend, OriginalRenderedChat, OriginalTextSourceError, PreparedSemanticSource,
};
use eredu_core::{
    BackendFailure, HostMetadataFunding, HostMetadataFundingError, InputModality, InputPayloadKind,
    ModelRuntime, TokenInputRejection,
};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

/// Source-derived text coordinates for one ordered host input part. Media token
/// expansion and decoder positions remain owned by architecture input admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatInputPartCoordinate {
    part: usize,
    text_start: usize,
    text_length: usize,
    rendered_range: [usize; 2],
    decoder_range: Option<[u64; 2]>,
    modality: InputModality,
    kind: InputPayloadKind,
}
impl ChatInputPartCoordinate {
    /// Index in the exact original host source.
    pub fn part(&self) -> usize {
        self.part
    }
    /// Number of rendered text tokens preceding this part.
    pub fn text_start(&self) -> usize {
        self.text_start
    }
    /// Exact text-token count; media parts insert at a text boundary.
    pub fn text_length(&self) -> usize {
        self.text_length
    }
    /// Exact range consumed from the actual rendered encoding, including markers.
    pub fn rendered_range(&self) -> [usize; 2] {
        self.rendered_range
    }
    /// Actual architecture decoder positions, when this input retains media semantics.
    pub fn decoder_range(&self) -> Option<[u64; 2]> {
        self.decoder_range
    }
    /// Modality of the original part.
    pub fn modality(&self) -> InputModality {
        self.modality
    }
    /// Role of the original payload.
    pub fn kind(&self) -> InputPayloadKind {
        self.kind
    }
}

/// Fixed refusal before model-input compilation or native submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedChatInputRejection {
    /// The actual input has no applicable architecture render projection.
    #[error("prepared chat part {part} has no authenticated render projection")]
    ProjectionUnavailable {
        /// Original part index.
        part: usize,
    },
    /// A token differs, is missing, or is additional in the ordered text parts.
    #[error("prepared chat text tokens differ at rendered token {token}")]
    TextMismatch {
        /// Offset in the actual rendered encoding.
        token: usize,
    },
    /// Projected text values cannot prove their original token content.
    #[error("prepared chat part {part} does not retain text-token identity")]
    TextIdentityUnavailable {
        /// Original part index.
        part: usize,
    },
}

/// Immutable actual render/header association retained with the native prompt.
/// Its digest describes content; source authority comes from the retained owners.
pub struct PreparedChatInputBinding {
    coordinates: Vec<ChatInputPartCoordinate>,
    content_digest: [u8; 32],
    generation_prompt: bool,
    semantics: Option<BoundCompositeSemanticStorage>,
    render: OriginalRenderedChat,
    preparation: PreparedSemanticSource,
}
impl eredu_core::SharedStorageRetirement for PreparedChatInputBinding {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
impl PreparedChatInputBinding {
    /// Exact render and semantic preparation, including original source/account
    /// identity. Another equal-text render does not satisfy this comparison.
    pub fn matches(
        &self,
        render: &OriginalRenderedChat,
        preparation: &PreparedSemanticSource,
        generation_prompt: bool,
    ) -> bool {
        self.render.same_render(render)
            && self.preparation.same_preparation(preparation)
            && self.generation_prompt == generation_prompt
    }
    /// The exact semantic preparation retained during input construction.
    /// Borrowing this owner grants no new allowance or native permission.
    pub fn preparation(&self) -> &PreparedSemanticSource {
        &self.preparation
    }
    /// Immutable coordinates derived while comparing all source text values.
    pub fn coordinates(&self) -> &[ChatInputPartCoordinate] {
        &self.coordinates
    }
    /// Same immutable original host source and all architecture coordinates.
    pub fn semantics(&self) -> Option<&BoundCompositeSemanticStorage> {
        self.semantics.as_ref()
    }
    /// Canonical ordered host-content digest; confers no execution authority.
    pub fn content_digest(&self) -> &[u8; 32] {
        &self.content_digest
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    #[error(transparent)]
    Host(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Policy(#[from] crate::working_memory::WorkingMemoryError),
    #[error(transparent)]
    Plan(#[from] HostInputPlanError),
    #[error(transparent)]
    Encoding(#[from] OriginalTextSourceError),
    #[error(transparent)]
    Association(#[from] PreparedChatInputRejection),
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}
/// An association or compiler failure retains the source-preparation payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedChatInputError {
    #[source]
    cause: Cause,
    funding: HostMetadataFunding,
}
impl PreparedChatInputError {
    /// Exact fixed token-association rejection, when comparison failed.
    pub fn association_rejection(&self) -> Option<PreparedChatInputRejection> {
        match self.cause {
            Cause::Association(cause) => Some(cause),
            _ => None,
        }
    }
}

impl std::fmt::Debug for PreparedChatInputBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedChatInputBinding")
            .field("coordinates", &self.coordinates)
            .field("content_digest", &self.content_digest)
            .field("generation_prompt", &self.generation_prompt)
            .field("render", &self.render)
            .field("preparation", &self.preparation)
            .field("has_semantics", &self.semantics.is_some())
            .finish()
    }
}

fn value_at(values: HostTensorValues<'_>, index: usize) -> Option<u32> {
    match values {
        HostTensorValues::U32(values) => values.get(index).copied(),
        HostTensorValues::I32(values) => values.get(index).and_then(|v| u32::try_from(*v).ok()),
        _ => None,
    }
}
fn values_match(values: HostTensorValues<'_>, expected: &[u32]) -> bool {
    values.len() <= expected.len()
        && (0..values.len()).all(|i| value_at(values, i) == Some(expected[i]))
}
fn visit_values(
    values: HostTensorValues<'_>,
    token: &mut impl FnMut(Option<u32>) -> Result<(), PreparedChatInputRejection>,
    part: usize,
) -> Result<(), PreparedChatInputRejection> {
    if !matches!(values, HostTensorValues::U32(_) | HostTensorValues::I32(_)) {
        return Err(PreparedChatInputRejection::TextIdentityUnavailable { part });
    }
    for index in 0..values.len() {
        token(value_at(values, index))?;
    }
    Ok(())
}
fn video_coordinates(
    origin: eredu_core::InputExtent,
    part: usize,
) -> Result<(usize, usize, usize), PreparedChatInputRejection> {
    match origin {
        eredu_core::InputExtent::VideoFrame {
            group,
            index,
            count,
            ..
        } if count > 0 && index < count => Ok((group, index, count)),
        _ => Err(PreparedChatInputRejection::ProjectionUnavailable { part }),
    }
}

fn verify_generated_parts<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    preparation: &PreparedSemanticSource,
    semantics: &BoundCompositeSemanticStorage,
    funding: &HostMetadataFunding,
) -> Result<(), Cause> {
    for record in semantics.records() {
        let part = record.source_part;
        let source = semantics
            .source()
            .part(part)
            .ok_or(PreparedChatInputRejection::ProjectionUnavailable { part })?;
        let values = source.payload_view().values;
        match record.chat_projection {
            CompositeChatProjection::VideoPrefix { text, framing, .. } => {
                let timestamp = funding.metadata_string(format_args!("{text}"))?;
                let encoded = B::encode_original_text_ids(
                    runtime,
                    preparation.tokenizer(),
                    &timestamp,
                    false,
                )?;
                if !encoded.matches_source(preparation.tokenizer())
                    || values.len() != encoded.ids().len() + 1
                    || !(0..encoded.ids().len())
                        .all(|i| value_at(values, i) == Some(encoded.ids()[i]))
                    || value_at(values, values.len() - 1) != Some(framing)
                {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part }.into());
                }
            }
            CompositeChatProjection::VideoSuffix { framing, .. } => {
                if values.len() != 1 || value_at(values, 0) != Some(framing) {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part }.into());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn compare_parts<P: HostInputPartView>(
    parts: impl ExactSizeIterator<Item = P>,
    expected: &[u32],
    records: Option<&[CompositeSemanticPartRecord]>,
    mut token_id: impl FnMut(&str) -> Option<u32>,
    mut accept: impl FnMut(ChatInputPartCoordinate),
) -> Result<(), PreparedChatInputRejection> {
    if records.is_some_and(|r| r.len() != parts.len()) {
        return Err(PreparedChatInputRejection::ProjectionUnavailable { part: 0 });
    }
    let mut offset = 0;
    let mut text_offset = 0;
    let mut decoder_offset = 0;
    let mut video: Option<(usize, usize, usize, bool)> = None;
    for (part, input) in parts.enumerate() {
        let start = offset;
        let text_start = text_offset;
        let record = records.map(|r| &r[part]);
        if record.is_some_and(|r| {
            r.source_part != part
                || r.modality != input.modality()
                || r.start != decoder_offset
                || r.end < r.start
        }) {
            return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
        }
        if let Some(r) = record {
            decoder_offset = r.end;
        }
        let projection = record.map(|r| r.chat_projection).unwrap_or(
            if input.modality() == InputModality::Text {
                CompositeChatProjection::SourceTokens
            } else {
                CompositeChatProjection::Unavailable
            },
        );
        let mut token = |value: Option<u32>| {
            if value.is_none() || expected.get(offset).copied() != value {
                return Err(PreparedChatInputRejection::TextMismatch { token: offset });
            }
            offset += 1; // Bounded by expected.len().
            Ok(())
        };
        match projection {
            CompositeChatProjection::SourceTokens => {
                if input.modality() != InputModality::Text
                    || input.kind() != InputPayloadKind::TokenIds
                {
                    return Err(PreparedChatInputRejection::TextIdentityUnavailable { part });
                }
                let values = input.payload().values;
                if record
                    .is_some_and(|r| usize::try_from(r.end - r.start).ok() != Some(values.len()))
                {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                }
                match values {
                    HostTensorValues::U32(values) => {
                        for &value in values {
                            token(Some(value))?;
                        }
                    }
                    HostTensorValues::I32(values) => {
                        for &value in values {
                            token(u32::try_from(value).ok())?;
                        }
                    }
                    _ => return Err(PreparedChatInputRejection::TextIdentityUnavailable { part }),
                }
                text_offset += offset - start;
            }
            CompositeChatProjection::Markers(markers) => {
                for &marker in markers {
                    token(token_id(marker))?;
                }
            }
            CompositeChatProjection::VideoPrefix { origin, .. } => {
                let (group, index, count) = video_coordinates(origin, part)?;
                let expanded = if index == 0 {
                    if video.is_some_and(|(prior_group, prior_index, prior_count, _)| {
                        group <= prior_group || prior_index + 1 != prior_count
                    }) {
                        return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                    }
                    values_match(input.payload().values, &expected[start..])
                } else {
                    let Some((prior_group, prior_index, prior_count, expanded)) = video else {
                        return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                    };
                    if group != prior_group || index != prior_index + 1 || count != prior_count {
                        return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                    }
                    expanded
                };
                if expanded {
                    visit_values(input.payload().values, &mut token, part)?;
                }
                video = Some((group, index, count, expanded));
            }
            CompositeChatProjection::VideoMarker {
                origin,
                compact,
                expanded,
            } => {
                let (group, index, count) = video_coordinates(origin, part)?;
                let Some((active_group, active_index, active_count, is_expanded)) = video else {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                };
                if (group, index, count) != (active_group, active_index, active_count) {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                }
                for &marker in if is_expanded {
                    expanded
                } else if index == 0 {
                    compact
                } else {
                    &[]
                } {
                    token(token_id(marker))?;
                }
            }
            CompositeChatProjection::VideoSuffix { origin, .. } => {
                let (group, index, count) = video_coordinates(origin, part)?;
                let Some((active_group, active_index, active_count, expanded)) = video else {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                };
                if (group, index, count) != (active_group, active_index, active_count) {
                    return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
                }
                if expanded {
                    visit_values(input.payload().values, &mut token, part)?;
                }
            }
            CompositeChatProjection::Unavailable => {
                return Err(PreparedChatInputRejection::ProjectionUnavailable { part });
            }
        }
        accept(ChatInputPartCoordinate {
            part,
            text_start,
            text_length: text_offset - text_start,
            rendered_range: [start, offset],
            decoder_range: record.map(|r| [r.start, r.end]),
            modality: input.modality(),
            kind: input.kind(),
        });
    }
    if video.is_some_and(|(_, index, count, _)| index + 1 != count) {
        return Err(PreparedChatInputRejection::ProjectionUnavailable { part: 0 });
    }
    if offset != expected.len() {
        return Err(PreparedChatInputRejection::TextMismatch { token: offset });
    }
    Ok(())
}

/// Encodes the actual original render and consumes a value-checked host plan
/// through the existing model-input compiler. Its actual architecture declaration
/// accounts for every rendered token, including media markers and generated
/// framing; this entry never accepts caller-declared skipped ranges.
pub fn prepare_original_chat_model_input<B: OriginalChatBackend + OriginalModelInputBackend>(
    runtime: &ModelRuntime<B>,
    preparation: &PreparedSemanticSource,
    render: &OriginalRenderedChat,
    generation_prompt: bool,
    parts: &[HostInputPart<'_>],
) -> Result<OriginalModelInput<B::Prompt>, PreparedChatInputError> {
    let funding = preparation.metadata_funding().clone();
    let result = (|| -> Result<_, Cause> {
        if !render
            .tokenizer_source()
            .same_source(preparation.tokenizer())
        {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        B::validate_semantic_source(runtime, preparation)?;
        B::validate_original_chat_render(runtime, render)?;
        let controls = [
            size_of::<PreparedChatInputBinding>(),
            size_of::<PreparedChatInputError>(),
            size_of::<Cause>(),
            size_of::<OriginalModelInput<B::Prompt>>(),
            size_of::<Result<OriginalModelInput<B::Prompt>, PreparedChatInputError>>(),
            size_of::<Result<OriginalModelInput<B::Prompt>, BackendFailure>>(),
            size_of::<PreparedHostInputPlan<'_>>(),
            size_of::<Result<PreparedHostInputPlan<'_>, HostInputPlanError>>(),
            size_of::<crate::working_memory::OriginalEncodedTokenIds>(),
            size_of::<
                Result<crate::working_memory::OriginalEncodedTokenIds, OriginalTextSourceError>,
            >(),
            size_of::<ChatInputPartCoordinate>(),
            size_of::<Vec<ChatInputPartCoordinate>>(),
            size_of::<Arc<PreparedChatInputBinding>>(),
            size_of::<Option<PreparedChatInputBinding>>(),
            size_of::<eredu_core::SharedStorageOwner<PreparedChatInputBinding>>(),
            size_of::<Option<Arc<PreparedChatInputBinding>>>(),
            size_of::<Option<BoundCompositeSemanticStorage>>(),
            size_of::<CompositeSemanticPartRecord>(),
            size_of::<CompositeChatProjection>(),
            size_of::<Option<(usize, usize, usize, bool)>>(),
            size_of::<crate::working_memory::CompositeGeneratedText>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let bytes = bytes
            .checked_add(
                usize::try_from(crate::working_memory::qualified_shared_bytes::<
                    PreparedChatInputBinding,
                >()?)
                .map_err(|_| HostMetadataFundingError::Overflow)?,
            )
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        let plan = PreparedHostInputPlan::prepare(parts)?;
        let encoded = B::encode_original_text_ids(
            runtime,
            preparation.tokenizer(),
            render.prompt(generation_prompt),
            false,
        )?;
        if !encoded.matches_source(preparation.tokenizer()) {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        let content_digest = *plan.content_digest();
        // The existing compiler retains the actual source and architecture
        // records. All partial native construction remains under its own custody.
        let input = B::prepare_original_model_input(runtime, plan, preparation.capacity_bytes())?;
        let semantics = B::original_model_input_semantics(input.prompt());
        let mut coordinates = funding.metadata_vec(parts.len())?;
        if let Some(semantics) = semantics {
            // Content comparison supplements the actual owner borrowed above;
            // it does not adopt an independently supplied semantic source.
            if semantics.source().content_digest() != &content_digest {
                return Err(TokenInputRejection::IdentityMismatch.into());
            }
            verify_generated_parts(runtime, preparation, semantics, &funding)?;
            compare_parts(
                semantics.source().parts(),
                encoded.ids(),
                Some(semantics.records()),
                |marker| preparation.tokenizer().added_token_id(marker),
                |coordinate| coordinates.push(coordinate),
            )?;
            if coordinates
                .last()
                .and_then(|c| c.decoder_range)
                .map(|r| r[1])
                != u64::try_from(semantics.layout().positions()).ok()
            {
                return Err(TokenInputRejection::IdentityMismatch.into());
            }
        } else {
            compare_parts(
                parts.iter().copied(),
                encoded.ids(),
                None,
                |_| None,
                |coordinate| coordinates.push(coordinate),
            )?;
        }
        let binding = eredu_core::SharedStorageOwner::new(PreparedChatInputBinding {
            coordinates,
            content_digest,
            generation_prompt,
            render: render.clone(),
            preparation: preparation.clone(),
            semantics: semantics.cloned(),
        });
        Ok(input.with_chat_binding(binding))
    })();
    result.map_err(|cause| PreparedChatInputError { cause, funding })
}

#[cfg(test)]
mod tests;
