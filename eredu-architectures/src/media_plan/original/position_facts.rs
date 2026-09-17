//! Borrowed attribution of the already compiled original parts. No re-admission.
use super::*;
use eredu_core::{InputPayloadKind, InputTokenCount, ObservationKind, PreparedPromptSegmentPlan};
use std::num::NonZeroU8;

/// One original source part's exact model rows and optional canonical-ID range.
/// The IDs remain in the actual original host source; no token buffer is copied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginalPromptSegment {
    /// Existing versioned attribution geometry, without any execution authority.
    pub plan: PreparedPromptSegmentPlan,
    /// Range in the concatenated actual token-ID payloads, if this part is tokens.
    pub canonical_range: Option<[u64; 2]>,
}

/// Checked scalar attribution borrowing the exact immutable original source.
/// Logical media estimates remain separate from native backing/workspace bounds.
#[derive(Clone, Copy)]
pub struct PreparedMediaPositionFacts<'a> {
    source: &'a OriginalPreparedHostInput,
    records: &'a [CompositeSemanticPartRecord],
    canonical: u64,
    projected_text: u64,
    media: u64,
    decoder: u64,
    legacy_media_input_bytes: u64,
    legacy_media_workspace_scalars: u64,
    has_encoded: bool,
}
fn add(a: u64, b: u64) -> Result<u64, MediaSemanticError> {
    a.checked_add(b)
        .ok_or(MediaSemanticError::overflow("original prompt attribution"))
}
fn value_bytes(values: HostTensorValues<'_>) -> Result<u64, MediaSemanticError> {
    let width = match values {
        HostTensorValues::Bool(_) => 1,
        _ => 4,
    };
    u64::try_from(values.len())
        .ok()
        .and_then(|n| n.checked_mul(width))
        .ok_or(MediaSemanticError::overflow("original media input bytes"))
}
impl<'a> PreparedMediaPositionFacts<'a> {
    pub(super) fn from_source(
        source: &'a OriginalPreparedHostInput,
        records: &'a [CompositeSemanticPartRecord],
        positions: usize,
    ) -> Result<Self, MediaSemanticError> {
        let invalid = || {
            MediaSemanticError::input("original prompt attribution differs from compiled source")
        };
        if source.parts().len() != records.len() {
            return Err(invalid());
        }
        let mut facts = Self {
            source,
            records,
            canonical: 0,
            projected_text: 0,
            media: 0,
            decoder: 0,
            legacy_media_input_bytes: 0,
            legacy_media_workspace_scalars: 0,
            has_encoded: false,
        };
        for (index, (record, part)) in records.iter().zip(source.parts()).enumerate() {
            if record.source_part != index
                || record.start != facts.decoder
                || record.end < record.start
                || record.modality != part.modality()
            {
                return Err(invalid());
            }
            u64::try_from(record.source_part)
                .map_err(|_| MediaSemanticError::overflow("original source part index"))?;
            let rows = record.end - record.start;
            match (record.role, part.modality(), part.kind()) {
                (
                    CompositeSemanticRole::Tokens,
                    InputModality::Text,
                    InputPayloadKind::TokenIds,
                ) => {
                    let payload = part.payload_view();
                    if !matches!(
                        payload.values,
                        HostTensorValues::U32(_) | HostTensorValues::I32(_)
                    ) || u64::try_from(payload.values.len()).ok() != Some(rows)
                    {
                        return Err(invalid());
                    }
                    if matches!(payload.values, HostTensorValues::I32(values) if values.iter().any(|value| *value < 0))
                    {
                        return Err(invalid());
                    }
                    facts.canonical = add(facts.canonical, rows)?;
                }
                (
                    CompositeSemanticRole::Projected,
                    InputModality::Text,
                    InputPayloadKind::Embeddings,
                ) => {
                    facts.projected_text = add(facts.projected_text, rows)?;
                }
                (
                    CompositeSemanticRole::Projected,
                    InputModality::Image | InputModality::Video | InputModality::Audio,
                    InputPayloadKind::Embeddings,
                ) => {
                    facts.media = add(facts.media, rows)?;
                }
                (
                    CompositeSemanticRole::Encoded,
                    InputModality::Image | InputModality::Video | InputModality::Audio,
                    InputPayloadKind::Tensor,
                ) => {
                    facts.media = add(facts.media, rows)?;
                    facts.has_encoded = true;
                    // Same numerical slots used by legacy media accounting.
                    let mut bytes = value_bytes(part.payload_view().values)?;
                    for key in [
                        InputMetadataKey::PatchGrid,
                        InputMetadataKey::PatchPositions,
                        InputMetadataKey::AudioMask,
                    ] {
                        if let Some((_, value)) = part.metadata_view(key) {
                            bytes = add(bytes, value_bytes(value.values)?)?;
                        }
                    }
                    facts.legacy_media_input_bytes = add(facts.legacy_media_input_bytes, bytes)?;
                    facts.legacy_media_workspace_scalars = add(
                        facts.legacy_media_workspace_scalars,
                        record.workspace_scalars,
                    )?;
                }
                _ => return Err(invalid()),
            }
            facts.decoder = record.end;
        }
        if u64::try_from(positions).ok() != Some(facts.decoder)
            || add(add(facts.canonical, facts.projected_text)?, facts.media)? != facts.decoder
        {
            return Err(invalid());
        }
        Ok(facts)
    }
    /// Actual original source, borrowed rather than replaced by content identity.
    pub const fn source(&self) -> &'a OriginalPreparedHostInput {
        self.source
    }
    /// Number of actual canonical tokenizer IDs.
    pub const fn canonical_tokens(&self) -> u64 {
        self.canonical
    }
    /// Text embedding rows with no canonical tokenizer IDs.
    pub const fn non_tokenized_text_positions(&self) -> u64 {
        self.projected_text
    }
    /// Media positions, including architecture-admitted projected media.
    pub const fn media_positions(&self) -> u64 {
        self.media
    }
    /// Complete model-position extent relative to the bound opening frontier.
    pub const fn decoder_positions(&self) -> u64 {
        self.decoder
    }
    /// Existing logical media equation with the selected scalar width. This
    /// preserves source payload/metadata and compiled graph-scalar populations;
    /// it is not a native primitive or first-interval workspace certificate.
    pub fn legacy_input_accounting(
        &self,
        media_scalar_bytes: NonZeroU8,
    ) -> Result<InputTokenCount, MediaSemanticError> {
        let workspace = self
            .legacy_media_workspace_scalars
            .checked_mul(u64::from(media_scalar_bytes.get()))
            .ok_or(MediaSemanticError::overflow(
                "original media logical workspace bytes",
            ))?;
        Ok(InputTokenCount::prepared(
            self.canonical,
            self.media,
            self.decoder,
            add(self.legacy_media_input_bytes, workspace)?,
            if self.has_encoded {
                ObservationKind::Conservative
            } else {
                ObservationKind::Exact
            },
        ))
    }
    /// Visits original parts in order without grid, token or descriptor copies.
    pub fn segments(&self) -> impl Iterator<Item = OriginalPromptSegment> + 'a {
        let mut canonical = 0;
        self.records.iter().map(move |record| {
            let rows = record.end - record.start;
            let (payload, canonical_range) = match record.role {
                CompositeSemanticRole::Tokens => {
                    let start = canonical;
                    canonical += rows; // Checked complete total in from_source.
                    (InputPayloadKind::TokenIds, Some([start, canonical]))
                }
                CompositeSemanticRole::Projected => (InputPayloadKind::Embeddings, None),
                CompositeSemanticRole::Encoded => (InputPayloadKind::Tensor, None),
            };
            OriginalPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: record.source_part as u64,
                    modality: record.modality,
                    payload,
                    decoder_range: [record.start, record.end],
                },
                canonical_range,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::input::host::{HostInputPart, HostTensorView, PreparedHostInputPlan};

    #[test]
    fn original_position_projection_validates_source_ranges_and_preserves_typed_tokens() {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let ids = [7i32, 11];
        let projected = [0.5f32; 12];
        let image = [1.5f32; 4];
        let parts = [
            HostInputPart {
                modality: InputModality::Text,
                kind: InputPayloadKind::TokenIds,
                payload: HostTensorView {
                    shape: &[1, 2],
                    values: HostTensorValues::I32(&ids),
                },
                metadata: &[],
                extents: &[],
            },
            HostInputPart {
                modality: InputModality::Text,
                kind: InputPayloadKind::Embeddings,
                payload: HostTensorView {
                    shape: &[1, 3, 4],
                    values: HostTensorValues::F32(&projected),
                },
                metadata: &[],
                extents: &[],
            },
            HostInputPart {
                modality: InputModality::Image,
                kind: InputPayloadKind::Tensor,
                payload: HostTensorView {
                    shape: &[4, 1],
                    values: HostTensorValues::F32(&image),
                },
                metadata: &[],
                extents: &[],
            },
        ];
        let source = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
            .unwrap();
        let used = pool.used_bytes().unwrap();
        let records = [
            CompositeSemanticPartRecord {
                source_part: 0,
                start: 0,
                end: 2,
                role: CompositeSemanticRole::Tokens,
                modality: InputModality::Text,
                ..Default::default()
            },
            CompositeSemanticPartRecord {
                source_part: 1,
                start: 2,
                end: 5,
                role: CompositeSemanticRole::Projected,
                modality: InputModality::Text,
                ..Default::default()
            },
            CompositeSemanticPartRecord {
                source_part: 2,
                start: 5,
                end: 6,
                role: CompositeSemanticRole::Encoded,
                modality: InputModality::Image,
                workspace_scalars: 19,
                ..Default::default()
            },
        ];
        let facts = PreparedMediaPositionFacts::from_source(&source, &records, 6).unwrap();
        assert!(std::ptr::eq(facts.source(), &source));
        assert_eq!(
            (
                facts.canonical_tokens(),
                facts.non_tokenized_text_positions(),
                facts.media_positions(),
                facts.decoder_positions()
            ),
            (2, 3, 1, 6)
        );
        let spans = facts.segments().collect::<Vec<_>>();
        assert_eq!(
            spans
                .iter()
                .map(|s| s.plan.decoder_range)
                .collect::<Vec<_>>(),
            [[0, 2], [2, 5], [5, 6]]
        );
        assert_eq!(
            spans.iter().map(|s| s.canonical_range).collect::<Vec<_>>(),
            [Some([0, 2]), None, None]
        );
        let input = facts
            .legacy_input_accounting(NonZeroU8::new(4).unwrap())
            .unwrap();
        assert_eq!(input.media_execution_workspace_bytes(), 4 * 4 + 19 * 4);
        assert_eq!(
            input.media_execution_workspace_kind(),
            ObservationKind::Conservative
        );
        for changed in 0..5 {
            let mut bad = records;
            match changed {
                0 => bad[1].source_part = 0,
                1 => bad[1].start = 1,
                2 => bad[1].modality = InputModality::Video,
                3 => bad[1].role = CompositeSemanticRole::Tokens,
                _ => bad[0].end = 1,
            }
            assert!(PreparedMediaPositionFacts::from_source(&source, &bad, 6).is_err());
        }
        assert!(PreparedMediaPositionFacts::from_source(&source, &records, 7).is_err());
        let mut overflow = records;
        overflow[2].workspace_scalars = u64::MAX;
        let facts = PreparedMediaPositionFacts::from_source(&source, &overflow, 6).unwrap();
        assert!(facts
            .legacy_input_accounting(NonZeroU8::new(4).unwrap())
            .err()
            .unwrap()
            .is_overflow());
        assert_eq!(pool.used_bytes().unwrap(), used);
        let values = source.part(0).unwrap().payload_view().values;
        assert!(matches!(values,HostTensorValues::I32(values) if values == [7,11]));
        let invalid_ids = [-1i32, 11];
        let invalid_parts = [HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::I32(&invalid_ids),
            },
            metadata: &[],
            extents: &[],
        }];
        let negative = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&invalid_parts).unwrap())
            .unwrap();
        assert!(PreparedMediaPositionFacts::from_source(&negative, &records[..1], 2).is_err());
        drop(negative);
        assert_eq!(pool.used_bytes().unwrap(), used);
    }
}
