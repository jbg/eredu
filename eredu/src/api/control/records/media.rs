//! Actual host parts and architecture coordinates produce diagnostic attribution.
use super::construction::{self, RecordConstructionCause as Cause, RecordConstructionError};
use super::*;
use eredu_runtime::input::{
    PreparedChatInputBinding,
    host::{HostInputPartView, HostTensorValues, HostTensorView},
};
use std::{alloc::Layout, mem::size_of};

fn tensor(
    source: HostTensorView<'_>,
    funding: &HostMetadataFunding,
) -> construction::Result<InputTensorIdentity> {
    let mut shape = construction::vector(source.shape.len(), funding)?;
    shape.extend_from_slice(source.shape);
    let dtype = match source.values {
        HostTensorValues::U32(_) => eredu_core::checkpoint::TensorDtype::U32,
        HostTensorValues::I32(_) => eredu_core::checkpoint::TensorDtype::I32,
        HostTensorValues::F32(_) => eredu_core::checkpoint::TensorDtype::F32,
        HostTensorValues::Bool(_) => eredu_core::checkpoint::TensorDtype::Bool,
    };
    InputTensorIdentity::new(dtype, shape).map_err(|_| Cause::Attribution)
}

fn map<K: Ord, V>(
    rows: Vec<(K, V)>,
    funding: &HostMetadataFunding,
) -> construction::Result<InputIdentityMap<K, V>> {
    construction::controls(
        funding,
        &[
            size_of::<Vec<(K, V)>>(),
            size_of::<Box<[(K, V)]>>(),
            size_of::<InputIdentityMap<K, V>>(),
            size_of::<Result<InputIdentityMap<K, V>, Box<[(K, V)]>>>(),
        ],
    )?;
    if rows.capacity() != rows.len() {
        funding.reserve_metadata(
            Layout::array::<(K, V)>(rows.len())
                .map_err(|_| HostMetadataFundingError::Overflow)?
                .size(),
        )?;
    }
    InputIdentityMap::from_sorted_entries(rows.into_boxed_slice()).map_err(|_| Cause::Attribution)
}

impl PromptRecord {
    pub(crate) fn from_media(
        binding: &PreparedChatInputBinding,
        funding: &HostMetadataFunding,
    ) -> Result<Self, RecordConstructionError> {
        let result = (|| {
            construction::controls(
                funding,
                &[
                    size_of::<Self>(),
                    size_of::<PreparedPromptAttribution>(),
                    size_of::<PreparedInputIdentity>(),
                    size_of::<InputPartDescriptor>(),
                    size_of::<InputTensorIdentity>(),
                    size_of::<PreparedPromptSegment>(),
                    size_of::<PreparedPromptSegmentPlan>(),
                    size_of::<PromptTokenAttribution>(),
                    size_of::<Result<Self, RecordConstructionError>>(),
                    size_of::<construction::Result<Self>>(),
                    size_of::<eredu_core::PreparedInputError>(),
                    size_of::<HostPreparationAuthority>(),
                    size_of::<(usize, u64, u64, [u64; 2])>(),
                    size_of::<HostTensorView<'_>>(),
                    size_of::<eredu_runtime::working_memory::BoundCompositeSemanticStorage>(),
                ],
            )?;
            let semantics = binding.semantics().ok_or(Cause::Attribution)?;
            let source = semantics.source();
            if source.parts().len() != binding.coordinates().len()
                || source.parts().len() != semantics.records().len()
                || source.content_digest() != binding.content_digest()
            {
                return Err(Cause::Attribution);
            }
            let tokens = source.parts().try_fold(0usize, |count, part| {
                if part.kind() != InputPayloadKind::TokenIds {
                    return Ok(count);
                }
                let HostTensorValues::U32(ids) = part.payload().values else {
                    return Err(Cause::Attribution);
                };
                count
                    .checked_add(ids.len())
                    .ok_or_else(|| HostMetadataFundingError::Overflow.into())
            })?;
            let mut descriptors = construction::vector(source.parts().len(), funding)?;
            let mut segments = construction::vector(source.parts().len(), funding)?;
            let mut canonical_token_ids = construction::vector(tokens, funding)?;
            for (index, part) in source.parts().enumerate() {
                let coordinate = &binding.coordinates()[index];
                let semantic = &semantics.records()[index];
                let decoder_range = coordinate.decoder_range().ok_or(Cause::Attribution)?;
                if coordinate.part() != index
                    || semantic.source_part != index
                    || coordinate.modality() != part.modality()
                    || coordinate.kind() != part.kind()
                    || decoder_range != [semantic.start, semantic.end]
                {
                    return Err(Cause::Attribution);
                }
                let payload = tensor(part.payload(), funding)?;
                let mut metadata = construction::vector(part.metadata().len(), funding)?;
                for (key, value) in part.metadata() {
                    metadata.push((key, tensor(value, funding)?));
                }
                let mut extents = construction::vector(part.extents().len(), funding)?;
                for &extent in part.extents() {
                    extents.push((extent.identity_key(), extent));
                }
                extents.sort_unstable_by_key(|row| row.0);
                descriptors.push(
                    InputPartDescriptor::from_fixed_entries(
                        part.modality(),
                        part.kind(),
                        payload,
                        map(metadata, funding)?,
                        map(extents, funding)?,
                    )
                    .map_err(|_| Cause::Attribution)?,
                );
                let tokens = if part.kind() == InputPayloadKind::TokenIds {
                    let HostTensorValues::U32(ids) = part.payload().values else {
                        return Err(Cause::Attribution);
                    };
                    let start = u64::try_from(canonical_token_ids.len())
                        .map_err(|_| HostMetadataFundingError::Overflow)?;
                    canonical_token_ids.extend_from_slice(ids);
                    PromptTokenAttribution::Canonical {
                        range: [
                            start,
                            u64::try_from(canonical_token_ids.len())
                                .map_err(|_| HostMetadataFundingError::Overflow)?,
                        ],
                    }
                } else {
                    PromptTokenAttribution::NotTokenized
                };
                segments.push(PreparedPromptSegment {
                    plan: PreparedPromptSegmentPlan {
                        source_part: u64::try_from(index)
                            .map_err(|_| HostMetadataFundingError::Overflow)?,
                        modality: part.modality(),
                        payload: part.kind(),
                        decoder_range,
                    },
                    tokens,
                });
            }
            let prepared =
                PreparedInputIdentity::new(descriptors).map_err(|_| Cause::Attribution)?;
            let mut semantic_content_identity = construction::string_capacity(64, funding)?;
            const HEX: &[u8; 16] = b"0123456789abcdef";
            for &byte in source.content_digest() {
                semantic_content_identity.push(HEX[usize::from(byte >> 4)] as char);
                semantic_content_identity.push(HEX[usize::from(byte & 15)] as char);
            }
            funding.reserve_metadata(
                HostPreparationAuthority::retention_bytes::<(
                    HostMetadataFunding,
                    eredu_runtime::working_memory::BoundCompositeSemanticStorage,
                )>()
                .ok_or(HostMetadataFundingError::Overflow)?,
            )?;
            let host = HostPreparationAuthority::retain((funding.clone(), semantics.clone()));
            funding.reserve_metadata(
                SharedPromptAttribution::construction_bytes()
                    .ok_or(HostMetadataFundingError::Overflow)?,
            )?;
            SharedPromptAttribution::from_prepared(
                PreparedPromptAttribution {
                    schema_version: PREPARED_PROMPT_ATTRIBUTION_VERSION,
                    prepared,
                    semantic_content_identity,
                    opening_position: semantics.binding().frontier(),
                    decoder_positions: u64::try_from(semantics.layout().positions())
                        .map_err(|_| HostMetadataFundingError::Overflow)?,
                    batch: 1,
                    segments,
                    canonical_token_ids,
                },
                host,
            )
            .map(Self)
            .map_err(|_| Cause::Attribution)
        })();
        result.map_err(|cause| RecordConstructionError::retain(cause, funding))
    }
}
