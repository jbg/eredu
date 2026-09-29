//! Host-only admission; the proof borrows the exact request rather than matching shapes.
use super::{MediaInputError, PreparedMediaInput};
use crate::{
    composite_execution::PreparedCompositeInput,
    media_plan::{AdmittedCompositeInput, QwenInputPartPlan},
    qwen::{
        ingress::{self, ProjectedTokenPolicy},
        vision::VisionConfig,
        vl::{self, PositionPart},
    },
    qwen4_exp::config::MediaTokens,
};
use eredu_core::{checkpoint::TensorDtype, InputMetadataKey as Key, InputModality};
use eredu_nn::{Error, Tensor};
use eredu_runtime::{PreparedInputInspector, PreparedInputPayload, PreparedModelInput};

/// Source-independent processor policy. No checkpoint, device or native context is retained.
#[derive(Clone)]
pub struct MediaAdmissionConfig {
    pub(super) identity: String,
    pub(super) vision: VisionConfig,
    pub(super) media: MediaTokens,
    pub(super) hidden: i32,
    pub(super) vocabulary: i32,
    /// Complete single-lane request, independently of one target invocation.
    pub(super) max_tokens: usize,
    pub(super) maximum_chunk_tokens: usize,
    pub(super) rotary_width: i32,
    pub(super) theta: f32,
    pub(super) sections: [i32; 3],
}
impl MediaAdmissionConfig {
    /// Complete request ceiling derived from the retained sequence-history limit.
    pub fn maximum_request_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Per-call single-lane target and assembly ceiling; lookup limits use this extent.
    pub fn maximum_chunk_tokens(&self) -> usize {
        self.maximum_chunk_tokens
    }

    pub(super) fn validate_request_bounds(
        &self,
        positions: u64,
        maximum_chunk_tokens: usize,
    ) -> Result<(), MediaInputError> {
        if positions > self.max_tokens as u64 {
            return Err(MediaInputError::Geometry(
                "decoder extent exceeds request bound",
            ));
        }
        if maximum_chunk_tokens > self.maximum_chunk_tokens {
            return Err(MediaInputError::Geometry(
                "media proof exceeds target chunk bound",
            ));
        }
        Ok(())
    }

    /// Admits bounded metadata before reading IDs, then constructs exact placeholder
    /// identity and rotary positions. Video parts represent individual timestamped
    /// frames, matching the released processor's ordered text/frame contract.
    pub fn admit<'a, T>(
        &self,
        input: &'a PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<AdmittedMediaInput<'a, T>, MediaInputError> {
        // Bound metadata reads before the shared inspector evaluates grid tensors.
        if input.len() > self.max_tokens {
            return Err(MediaInputError::Geometry(
                "part count exceeds request bound",
            ));
        }
        let mut grid_rows = 0usize;
        for part in input.identity().parts() {
            for key in part.metadata().keys() {
                let allowed = match key {
                    Key::PatchGrid => {
                        matches!(part.modality(), InputModality::Image | InputModality::Video)
                    }
                    Key::OriginalTokenIds => {
                        part.payload_kind() == eredu_core::InputPayloadKind::Embeddings
                    }
                    _ => false,
                };
                if !allowed {
                    return Err(MediaInputError::Geometry(
                        "metadata has no role in this input contract",
                    ));
                }
            }
            if let Some(grid) = part.metadata_value(Key::PatchGrid) {
                if grid.shape().len() != 2
                    || grid.shape()[1] != 3
                    || grid.shape()[0] > self.max_tokens
                    || *grid.dtype() != TensorDtype::I32
                {
                    return Err(MediaInputError::Geometry(
                        "patch-grid metadata exceeds its bound or has invalid geometry/type",
                    ));
                }
                grid_rows = grid_rows
                    .checked_add(grid.shape()[0])
                    .filter(|rows| *rows <= self.max_tokens)
                    .ok_or(MediaInputError::Geometry(
                        "total patch-grid metadata exceeds request bound",
                    ))?;
            }
        }
        grid_rows
            .checked_mul(3 * std::mem::size_of::<i32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(MediaInputError::Geometry(
                "patch-grid metadata exceeds host addressability",
            ))?;
        let media = &self.media;
        let admitted = crate::media_plan::admit_composite_input(input, inspector, |part| {
            crate::media_plan::qwen_input_part_with_policy(
                "qwen4_exp",
                self.hidden,
                Some(&self.vision),
                Some(media.image as i32),
                Some(media.video as i32),
                part,
                inspector,
            )
        })?;
        let tokens = usize::try_from(admitted.decoder_positions())
            .map_err(|_| MediaInputError::Geometry("decoder extent overflow"))?;
        if tokens > self.max_tokens || tokens > i32::MAX as usize {
            return Err(MediaInputError::Geometry(
                "decoder extent exceeds request bound",
            ));
        }
        tokens
            .checked_mul(3 * std::mem::size_of::<i32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(MediaInputError::Geometry(
                "rotary positions exceed host addressability",
            ))?;
        let mut grids = Vec::with_capacity(input.len());
        let mut original_ids = Vec::with_capacity(tokens);
        for (index, ((part, descriptor), plan)) in input
            .parts()
            .iter()
            .zip(input.identity().parts())
            .zip(admitted.parts())
            .enumerate()
        {
            let positions = match plan {
                QwenInputPartPlan::TextTokens { positions }
                | QwenInputPartPlan::Projected { positions, .. } => *positions,
                QwenInputPartPlan::Media { shape, .. } => shape.decoder_positions,
            } as usize;
            let ids = match part.payload() {
                PreparedInputPayload::TokenIds(ids) => Some((ids, descriptor.payload())),
                PreparedInputPayload::Embeddings(_) => {
                    let identity = descriptor.require_metadata(index, Key::OriginalTokenIds)?;
                    Some((
                        part.metadata_value(Key::OriginalTokenIds)
                            .expect("descriptor retains metadata"),
                        identity,
                    ))
                }
                _ => None,
            };
            if let Some((ids, identity)) = ids {
                if identity.shape() != [1, positions] {
                    return Err(eredu_runtime::TokenInputError::Geometry.into());
                }
                if !matches!(identity.dtype(), TensorDtype::I32 | TensorDtype::U32) {
                    return Err(eredu_runtime::TokenInputError::ScalarType.into());
                }
                let ids = inspector.i32_values(ids)?;
                if ids.len() != positions {
                    return Err(eredu_runtime::TokenInputError::Geometry.into());
                }
                if ids.iter().any(|id| *id < 0 || *id >= self.vocabulary) {
                    return Err(eredu_runtime::TokenInputError::Vocabulary.into());
                }
                if part.modality() != InputModality::Text {
                    let expected = if part.modality() == InputModality::Image {
                        media.image
                    } else {
                        media.video
                    };
                    if ids.iter().any(|id| *id as u32 != expected) {
                        return Err(MediaInputError::Geometry(
                            "projected media IDs differ from declared placeholders",
                        ));
                    }
                }
                original_ids.extend(ids);
            } else if let QwenInputPartPlan::Media { ingress, .. } = plan {
                original_ids.extend(std::iter::repeat_n(
                    ingress.placeholder_token_id as i32,
                    positions,
                ));
            }
            let grid = match plan {
                QwenInputPartPlan::Media { ingress, .. } => Some(ingress.patch_grid.clone()),
                QwenInputPartPlan::Projected { modality, .. }
                    if *modality != InputModality::Text =>
                {
                    descriptor.require_metadata(index, Key::PatchGrid)?;
                    let values = inspector
                        .i32_values(part.metadata_value(Key::PatchGrid).expect("required grid"))?;
                    if values.len()
                        != descriptor.metadata_value(Key::PatchGrid).unwrap().shape()[0] * 3
                    {
                        return Err(MediaInputError::Geometry(
                            "patch-grid value count differs from descriptor",
                        ));
                    }
                    Some(values.chunks_exact(3).map(|v| (v[0], v[1], v[2])).collect())
                }
                _ => None,
            };
            if let Some(grid) = &grid {
                if grid.len() != 1 || (part.modality() == InputModality::Video && grid[0].0 != 1) {
                    return Err(MediaInputError::Geometry(
                        "each image/video part must represent one image or timestamped video frame",
                    ));
                }
                // Adjacent parts of the same modality form one reference token-type run.
                if index > 0 && input.parts()[index - 1].modality() == part.modality() {
                    return Err(MediaInputError::Geometry(
                        "distinct same-modality media spans require intervening text tokens",
                    ));
                }
                vl::multimodal_position_ids(
                    &[PositionPart::Media(grid)],
                    self.vision.spatial_merge_size,
                    positions as i32,
                )
                .map_err(|_| {
                    MediaInputError::Geometry("projected media span differs from its grid")
                })?;
            }
            grids.push(grid);
        }
        let position_parts: Vec<_> = grids
            .iter()
            .zip(admitted.parts())
            .map(|(grid, part)| match grid {
                Some(grid) => PositionPart::Media(grid),
                None => PositionPart::Text(match part {
                    QwenInputPartPlan::TextTokens { positions }
                    | QwenInputPartPlan::Projected { positions, .. } => *positions as i32,
                    _ => unreachable!(),
                }),
            })
            .collect();
        let (positions, delta) = vl::multimodal_position_ids(
            &position_parts,
            self.vision.spatial_merge_size,
            tokens as i32,
        )
        .map_err(|_| MediaInputError::Geometry("invalid media positions"))?;
        Ok(AdmittedMediaInput {
            input,
            data: MediaAdmissionData {
                policy: self.clone(),

                admitted,
                original_ids,
                positions,
                delta,
            },
        })
    }
}

/// Validated host IDs, grids and positions tied to one immutable prepared request.
/// Admission has no tensor bound and performs no backend tensor construction.
pub struct AdmittedMediaInput<'a, T> {
    input: &'a PreparedModelInput<T>,
    data: MediaAdmissionData,
}
struct MediaAdmissionData {
    policy: MediaAdmissionConfig,
    admitted: AdmittedCompositeInput<QwenInputPartPlan>,
    original_ids: Vec<i32>,
    positions: [Vec<i32>; 3],
    delta: i32,
}
impl<T> AdmittedMediaInput<'_, T> {
    pub(crate) fn validate_policy(
        &self,
        policy: &MediaAdmissionConfig,
    ) -> Result<(), MediaInputError> {
        policy.validate_request_bounds(
            self.data.admitted.decoder_positions(),
            self.data.policy.maximum_chunk_tokens,
        )?;

        if self.data.policy.identity != policy.identity {
            return Err(MediaInputError::Geometry(
                "media proof belongs to a different policy",
            ));
        }
        Ok(())
    }

    /// Exact rotary preparation geometry using the same policy as native preparation.
    /// This describes products for the entire admitted request, before chunk assembly.
    pub fn rotary_invocation(
        &self,
    ) -> Result<eredu_nn::mechanism_memory::MechanismInvocation, MediaInputError> {
        Ok(vl::mrope_spec(
            self.data.policy.rotary_width,
            self.data.policy.theta,
            &self.data.policy.sections,
        )?
        .memory_invocation(
            &[self.data.admitted.decoder_positions(), 3],
            eredu_nn::TensorElementType::I32,
        )?)
    }

    /// Shared processor accounting for the exact ordered request.
    pub fn admission(&self) -> &AdmittedCompositeInput<QwenInputPartPlan> {
        &self.data.admitted
    }
    /// Validated original identities, including exact media placeholders.
    pub fn token_ids(&self) -> &[i32] {
        &self.data.original_ids
    }
    /// Absolute temporal, height and width rotary positions before tensor allocation.
    pub fn position_ids(&self) -> &[Vec<i32>; 3] {
        &self.data.positions
    }
    /// Signed media shift used by subsequent cached decode.
    pub fn rotary_delta(&self) -> i32 {
        self.data.delta
    }
}
impl<T: Tensor> AdmittedMediaInput<'_, T> {
    /// Allocates request tensors from the retained proof without inspecting IDs or
    /// metadata again. The request cannot be replaced by another with equal shapes.
    pub fn prepare(&self, context: &T::Context) -> Result<PreparedMediaInput<T>, MediaInputError> {
        self.data.prepare(self.input, context)
    }
}
impl MediaAdmissionData {
    fn prepare<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        context: &T::Context,
    ) -> Result<PreparedMediaInput<T>, MediaInputError> {
        let tokens = self.original_ids.len() as i32;
        let positions = vl::position_ids_tensor::<T>(&self.positions, context)?;
        let (cosine, sine) = vl::mrope_embeddings(
            &positions,
            self.policy.rotary_width,
            self.policy.theta,
            &self.policy.sections,
            context,
        )?;
        let cosine = cosine.reshape(&[1, tokens, self.policy.rotary_width], context)?;
        let sine = sine.reshape(&[1, tokens, self.policy.rotary_width], context)?;
        let prepared = ingress::prepare_input(
            PreparedCompositeInput::new(input, &self.admitted).map_err(Error::backend)?,
            ProjectedTokenPolicy::Required,
            context,
        )?;
        Ok(PreparedMediaInput {
            policy: self.policy.identity.clone(),
            prepared,
            admitted: self.admitted.clone(),
            ids: T::from_i32_slice(&self.original_ids, &[1, tokens], context)?,
            cosine,
            sine,
            delta: self.delta,
            hidden: self.policy.hidden,

            maximum_chunk_tokens: self.policy.maximum_chunk_tokens,
        })
    }
}

/// One part of an owned composite proof. All parts share one retained request and
/// validated host record, rather than duplicating request tensors or ID histories.
pub struct MediaInputPartPlan<T> {
    index: usize,
    request: std::sync::Arc<OwnedMediaAdmission<T>>,
    chunk: Option<std::sync::Arc<MediaPrefillChunk<T>>>,
}
pub(crate) struct MediaPrefillChunk<T> {
    pub(crate) prepared: std::sync::Arc<PreparedMediaInput<T>>,
    pub(crate) range: std::ops::Range<i32>,
    pub(crate) projected: Option<T>,
}
impl<T> Clone for MediaInputPartPlan<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            request: self.request.clone(),
            chunk: self.chunk.clone(),
        }
    }
}
impl<T> From<MediaInputPartPlan<T>> for crate::media_plan::PreparedInputPartPlan {
    fn from(part: MediaInputPartPlan<T>) -> Self {
        part.shared().clone().into()
    }
}
impl<T> MediaInputPartPlan<T> {
    /// A fresh request keeps the original admission proof, not a prior chunk's
    /// prepared products or encoder continuation.
    pub(crate) fn without_chunk(mut self) -> Self {
        self.chunk = None;
        self
    }
    pub(crate) fn with_chunk(mut self, chunk: std::sync::Arc<MediaPrefillChunk<T>>) -> Self {
        self.chunk = Some(chunk);
        self
    }
    pub(crate) fn chunk(parts: &[Self]) -> Option<&MediaPrefillChunk<T>> {
        parts.first()?.chunk.as_deref()
    }
    /// Shared Qwen geometry for this exact ordered part.
    pub fn shared(&self) -> &QwenInputPartPlan {
        &self.request.data.admitted.parts()[self.index]
    }
    pub(crate) fn request(parts: &[Self]) -> Result<&OwnedMediaAdmission<T>, MediaInputError> {
        let first = parts
            .first()
            .ok_or(MediaInputError::Geometry("empty composite media proof"))?;
        if parts.len() != first.request.data.admitted.parts().len()
            || parts.iter().enumerate().any(|(index, part)| {
                part.index != index
                    || !std::sync::Arc::ptr_eq(&first.request, &part.request)
                    || match (&first.chunk, &part.chunk) {
                        (None, None) => false,
                        (Some(a), Some(b)) => !std::sync::Arc::ptr_eq(a, b),
                        _ => true,
                    }
            })
        {
            return Err(MediaInputError::Geometry(
                "composite media proof mixes requests or part order",
            ));
        }
        Ok(&first.request)
    }
}
pub(crate) struct OwnedMediaAdmission<T> {
    input: PreparedModelInput<T>,
    data: MediaAdmissionData,
}
impl<T: Tensor> OwnedMediaAdmission<T> {
    pub(crate) fn input(&self) -> &PreparedModelInput<T> {
        &self.input
    }
    pub(crate) fn maximum_chunk_tokens(&self) -> usize {
        self.data.policy.maximum_chunk_tokens
    }
    pub(crate) fn validate_policy(
        &self,
        policy: &MediaAdmissionConfig,
    ) -> Result<(), MediaInputError> {
        policy.validate_request_bounds(
            self.data.admitted.decoder_positions(),
            self.data.policy.maximum_chunk_tokens,
        )?;

        if self.data.policy.identity != policy.identity {
            return Err(MediaInputError::Geometry(
                "composite media proof belongs to a different policy",
            ));
        }
        Ok(())
    }
    pub(crate) fn prepare(
        &self,
        context: &T::Context,
    ) -> Result<PreparedMediaInput<T>, MediaInputError> {
        self.data.prepare(&self.input, context)
    }
    pub(crate) fn token_ids(&self, context: &T::Context) -> Result<T, Error> {
        T::from_i32_slice(
            &self.data.original_ids,
            &[1, self.data.original_ids.len() as i32],
            context,
        )
    }
}
impl<T: Clone> AdmittedMediaInput<'_, T> {
    /// Retains the admitted request for the shared composite adapter. Cloning
    /// tensor handles happens once for the request, never once for each part.
    pub fn into_composite(self) -> AdmittedCompositeInput<MediaInputPartPlan<T>> {
        let admitted = self.data.admitted.clone();
        let request = std::sync::Arc::new(OwnedMediaAdmission {
            input: self.input.clone(),
            data: self.data,
        });
        admitted.map_parts(|index, _| MediaInputPartPlan {
            index,
            request: request.clone(),
            chunk: None,
        })
    }
}
