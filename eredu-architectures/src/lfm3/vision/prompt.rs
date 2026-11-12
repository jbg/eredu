//! Projected media ingress for the shared LFM3 decoder.

use eredu_core::{CapabilityError, InputMetadataKey, InputModality};
use eredu_nn::{Error, Tensor};
use eredu_runtime::{
    PreparedInputCacheIdentity, PreparedInputInspector, PreparedInputPart, PreparedInputPayload,
    PreparedModelInput,
};

use crate::media_plan::{
    admit_composite_input, AdmittedCompositeInput, CompositePartPlan, MediaShapePlan,
    PreparedInputPartPlan,
};

use super::{
    super::projected_prompt::{ProjectedPrompt, Span},
    MediaTokens,
};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Source {
    Image,
    Video,
}

/// A batch-one prompt with preprojected image and video rows in marker order.
///
/// The processor expands media markers before creating the prompt. Other tokens,
/// including delimiters and video timestamps, retain their language embeddings.
pub struct VisionPrompt<'a, T>(ProjectedPrompt<'a, T, 2>);

impl<'a, T: Tensor> VisionPrompt<'a, T> {
    /// Checks marker counts and projects contiguous runs onto their source rows.
    pub fn new(
        ids: &[i32],
        markers: MediaTokens,
        width: i32,
        image: Option<&'a T>,
        video: Option<&'a T>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        ProjectedPrompt::new(
            ids,
            [markers.image_token_id, markers.video_token_id],
            width,
            [image, video],
            context,
        )
        .map(Self)
    }

    pub(crate) fn tokens(&self) -> &T {
        self.0.tokens()
    }

    /// Borrows one contiguous prompt range for incremental prefill.
    ///
    /// Text indices are local to the range. Media indices still address the
    /// original projected rows, even when a range starts inside a media span.
    pub fn slice(
        &self,
        range: std::ops::Range<i32>,
        context: &T::Context,
    ) -> Result<VisionPrompt<'_, T>, Error> {
        self.0.slice(range, context).map(VisionPrompt)
    }

    pub(in crate::lfm3) fn update_pipeline_digest(
        &self,
        digest: &mut sha2::Sha256,
        context: &T::Context,
    ) -> Result<(), Error> {
        self.0.update_pipeline_digest(digest, context)
    }

    pub(in crate::lfm3) fn projected_roots(&self) -> impl Iterator<Item = &T> {
        self.0.projected_roots()
    }

    pub(crate) fn replace(&self, text: &T, context: &T::Context) -> Result<T, Error> {
        self.0.replace(text, context)
    }
}

#[derive(Clone)]
struct AdmittedMedia<T> {
    source: Source,
    pixels: T,
    grid: Vec<(i32, i32, i32)>,
    rows: i32,
}

/// Owns admitted prompt geometry and completed projections across media invocations.
#[derive(Clone)]
pub struct VisionPromptPreparation<T> {
    identity: Option<PreparedInputCacheIdentity>,
    ids: Vec<i32>,
    markers: MediaTokens,
    width: i32,
    pending: std::collections::VecDeque<AdmittedMedia<T>>,
    images: Vec<T>,
    videos: Vec<T>,
}

/// Ordered VL ingress facts retained after shared prepared-input admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VlInputPartPlan {
    /// Validated vocabulary IDs for one text part.
    Text(Vec<i32>),
    /// Validated patch metadata and its decoder span.
    Media {
        /// Image or video marker policy.
        modality: InputModality,
        /// Ordered temporal groups and spatial patch dimensions.
        grid: Vec<(i32, i32, i32)>,
        /// Decoder extent and vision workspace.
        shape: MediaShapePlan,
    },
}

impl CompositePartPlan for VlInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::Text(ids) => ids.len() as u64,
            Self::Media { shape, .. } => shape.decoder_positions,
        }
    }
}

impl From<VlInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: VlInputPartPlan) -> Self {
        match plan {
            VlInputPartPlan::Text(ids) => Self::Text {
                positions: ids.len() as u64,
            },
            VlInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

fn unsupported(reason: impl Into<String>) -> CapabilityError {
    CapabilityError::UnsupportedInput {
        architecture: "lfm3_vl".into(),
        reason: reason.into(),
    }
}

/// Checkpoint geometry needed to admit prepared input, independent of its weight source.
#[derive(Clone)]
pub struct VisionPromptAdmission {
    markers: MediaTokens,
    merge: i32,
    patch_width: i32,
    vocab_size: i32,
    width: i32,
    max_positions: u64,
    vision: super::VisionConfig,
    projector_width: u64,
}

impl VisionPromptAdmission {
    /// Retains the request geometry from the admitted checkpoint.
    pub fn new(artifact: &super::PreparedVisionLanguage) -> Result<Self, Error> {
        Self::from_declarations(artifact.tokens, &artifact.vision, artifact.language.args())
    }

    pub(super) fn from_declarations(
        markers: MediaTokens,
        vision: &super::Declaration,
        language: &crate::lfm3::ModelArgs,
    ) -> Result<Self, Error> {
        let projector_width = vision.projector_config().hidden_size as u64;
        let vision = vision.config();
        Ok(Self {
            markers,
            merge: vision.spatial_merge_size,
            patch_width: vision.patch_width()?,
            vocab_size: language.vocab_size,
            width: language.hidden_size,
            max_positions: language.max_position_embeddings as u64,
            vision: vision.clone(),
            projector_width,
        })
    }

    /// Checks the entire request and expands its token IDs before any projection.
    pub fn admit<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<VisionPromptPreparation<T>, Error> {
        self.admit_with_identity(input, inspector, None)
    }

    /// Checks prepared geometry and preserves its exact semantic cache identity.
    pub fn admit_with_identity<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<VisionPromptPreparation<T>, Error> {
        if identity.is_some_and(|identity| identity.prepared() != input.identity()) {
            return Err(Error::backend(
                "VL cache identity differs from the prepared input",
            ));
        }
        let admitted = self
            .admit_composite(input, inspector)
            .map_err(Error::backend_source)?;
        self.prepare_admitted(input, &admitted, identity)
    }

    /// Validates exact tensor roles, token values, patch grids and the complete decoder extent.
    pub fn admit_composite<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<AdmittedCompositeInput<VlInputPartPlan>, CapabilityError> {
        let mut positions = 0u64;
        admit_composite_input(input, inspector, |part| {
            let plan = self.admit_part(part, inspector)?;
            positions = positions
                .checked_add(plan.decoder_positions())
                .filter(|&total| total <= self.max_positions)
                .ok_or_else(|| unsupported("VL prompt exceeds decoder context"))?;
            Ok(plan)
        })
    }

    fn admit_part<T: Tensor>(
        &self,
        part: &PreparedInputPart<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<VlInputPartPlan, CapabilityError> {
        match (part.modality(), part.payload()) {
            (InputModality::Text, PreparedInputPayload::TokenIds(tokens)) => {
                let (&count, _) = tokens
                    .shape()
                    .split_last()
                    .ok_or_else(|| unsupported("missing token axis"))?;
                if !matches!(tokens.shape(), [_] | [1, _])
                    || count < 0
                    || count as u64 > self.max_positions
                {
                    return Err(unsupported("invalid batch-one token extent"));
                }
                let ids = inspector.i32_values(tokens)?;
                if ids.len() != count as usize
                    || ids.iter().any(|&id| {
                        id < 0
                            || id >= self.vocab_size
                            || id == self.markers.image_token_id
                            || id == self.markers.video_token_id
                    })
                {
                    return Err(unsupported("invalid VL text token IDs"));
                }
                Ok(VlInputPartPlan::Text(ids))
            }
            (
                modality @ (InputModality::Image | InputModality::Video),
                PreparedInputPayload::Tensor(pixels),
            ) => {
                let grid = part
                    .metadata()
                    .get(&InputMetadataKey::PatchGrid)
                    .ok_or_else(|| unsupported("missing VL patch grid"))?;
                let [items, 3] = grid.shape() else {
                    return Err(unsupported("VL patch grid must have shape [items, 3]"));
                };
                if *items <= 0 || *items as u64 > self.max_positions {
                    return Err(unsupported("invalid VL patch grid extent"));
                }
                let values = inspector.i32_values(grid)?;
                let (triples, remainder) = values.as_chunks::<3>();
                if !remainder.is_empty() || triples.len() != *items as usize {
                    return Err(unsupported(
                        "VL patch grid value count differs from its shape",
                    ));
                }
                let grid = triples
                    .iter()
                    .map(|&[t, h, w]| (t, h, w))
                    .collect::<Vec<_>>();
                let overflow = || CapabilityError::ArithmeticOverflow {
                    operation: "LFM3 VL vision workspace",
                };
                let mut patches = 0u64;
                let mut attention_pairs = 0u64;
                for &(time, height, width) in &grid {
                    if time <= 0
                        || height <= 0
                        || width <= 0
                        || height % self.merge != 0
                        || width % self.merge != 0
                    {
                        return Err(unsupported("invalid VL temporal or merge grid"));
                    }
                    let spatial = (height as u64) * (width as u64);
                    let extent = spatial.checked_mul(time as u64).ok_or_else(overflow)?;
                    patches = patches.checked_add(extent).ok_or_else(overflow)?;
                    attention_pairs = attention_pairs
                        .checked_add(extent.checked_mul(spatial).ok_or_else(overflow)?)
                        .ok_or_else(overflow)?;
                }
                let patch_rows = i32::try_from(patches)
                    .map_err(|_| unsupported("VL patch count exceeds tensor extent"))?;
                if pixels.shape() != [patch_rows, self.patch_width] {
                    return Err(unsupported(
                        "VL pixels do not match the admitted patch grid",
                    ));
                }
                let rows = patches / (self.merge as u64).pow(2);
                // Bound the largest frame-local attention workspace and the
                // linear, normalization, convolution and merger temporaries.
                let attention = attention_pairs
                    .checked_mul(self.vision.num_attention_heads as u64 * 2)
                    .ok_or_else(overflow)?;
                let hidden = self.vision.hidden_size as u64;
                let linear_width = self.patch_width as u64
                    + 16 * hidden
                    + 4 * self.vision.intermediate_size as u64;
                let workspace = patches
                    .checked_mul(linear_width)
                    .and_then(|n| n.checked_add(attention))
                    .and_then(|n| {
                        n.checked_add(
                            rows.checked_mul(self.projector_width * 3 + self.width as u64)?,
                        )
                    })
                    .ok_or_else(overflow)?;
                Ok(VlInputPartPlan::Media {
                    modality,
                    grid,
                    shape: MediaShapePlan {
                        capture_media: None,
                        decoder_positions: rows,
                        execution_workspace_scalars: workspace,
                    },
                })
            }
            _ => Err(unsupported(
                "VL requires text tokens or image/video patches",
            )),
        }
    }

    fn prepare_admitted<T: Tensor>(
        &self,
        input: &PreparedModelInput<T>,
        admitted: &AdmittedCompositeInput<VlInputPartPlan>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<VisionPromptPreparation<T>, Error> {
        let paired = crate::composite_execution::PreparedCompositeInput::new(input, admitted)
            .map_err(Error::backend)?;
        let length =
            usize::try_from(admitted.decoder_positions()).map_err(Error::backend_source)?;
        let mut ids = Vec::with_capacity(length);
        let mut pending = std::collections::VecDeque::new();
        for (part, plan) in input.parts().iter().zip(paired.admitted().parts()) {
            match plan {
                VlInputPartPlan::Text(text) => ids.extend(text),
                VlInputPartPlan::Media {
                    modality,
                    grid,
                    shape,
                } => {
                    let PreparedInputPayload::Tensor(pixels) = part.payload() else {
                        return Err(Error::backend("admitted VL media lost its patch tensor"));
                    };
                    let (source, marker) = match modality {
                        InputModality::Image => (Source::Image, self.markers.image_token_id),
                        InputModality::Video => (Source::Video, self.markers.video_token_id),
                        _ => {
                            return Err(Error::backend("admitted VL media has an invalid modality"))
                        }
                    };
                    let rows =
                        i32::try_from(shape.decoder_positions).map_err(Error::backend_source)?;
                    ids.resize(ids.len() + rows as usize, marker);
                    pending.push_back(AdmittedMedia {
                        source,
                        pixels: pixels.clone(),
                        grid: grid.clone(),
                        rows,
                    });
                }
            }
        }
        Ok(VisionPromptPreparation {
            identity: identity.cloned(),
            ids,
            markers: self.markers,
            width: self.width,
            pending,
            images: Vec::new(),
            videos: Vec::new(),
        })
    }
}

impl<T: Tensor> VisionPromptPreparation<T> {
    /// Semantic cache identity carried by the admitted request.
    pub fn cache_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.identity.as_ref()
    }

    /// Expanded decoder IDs, available before any media weights are materialized.
    pub fn token_ids(&self) -> &[i32] {
        &self.ids
    }

    /// Describes request ingress before any media projection or decoder execution.
    pub fn prefill_descriptor(&self, chunk: std::num::NonZeroUsize) -> [u64; 8] {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        for token in &self.ids {
            digest.update(token.to_le_bytes());
        }
        if let Some(identity) = &self.identity {
            digest.update(identity.prefix_content_fingerprint().as_bytes());
        }
        let mut descriptor = [
            0x564c5052464f504e,
            chunk.get() as u64,
            self.ids.len() as u64,
            self.pending.len() as u64,
            0,
            0,
            0,
            0,
        ];
        for (word, bytes) in descriptor[4..]
            .iter_mut()
            .zip(digest.finalize().chunks_exact(8))
        {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight digest bytes"));
        }
        descriptor
    }

    pub(super) fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.pending
            .iter()
            .map(|media| &media.pixels)
            .chain(self.images.iter())
            .chain(self.videos.iter())
    }

    pub(super) fn retained_metadata_bytes(&self) -> Option<usize> {
        use std::mem::size_of;
        let mut bytes = self.ids.capacity().checked_mul(size_of::<i32>())?;
        bytes = bytes.checked_add(
            self.pending
                .capacity()
                .checked_mul(size_of::<AdmittedMedia<T>>())?,
        )?;
        bytes = bytes.checked_add(
            self.images
                .capacity()
                .checked_add(self.videos.capacity())?
                .checked_mul(size_of::<T>())?,
        )?;
        for media in &self.pending {
            bytes = bytes.checked_add(media.grid.capacity().checked_mul(size_of::<(
                i32,
                i32,
                i32,
            )>())?)?;
        }
        if let Some(identity) = &self.identity {
            bytes = bytes.checked_add(usize::try_from(identity.logical_metadata_bytes()?).ok()?)?;
        }
        Some(bytes)
    }

    /// Borrows the next visual invocation without advancing the cursor.
    pub fn next_media(&self) -> Option<super::VisionInput<'_, T>> {
        self.pending.front().map(|media| super::VisionInput {
            pixels: &media.pixels,
            grid: &media.grid,
        })
    }

    /// Commits one completed projection. A rejected result leaves the cursor unchanged.
    pub fn accept_projection(&mut self, rows: T) -> Result<(), Error> {
        let media = self
            .pending
            .front()
            .ok_or_else(|| Error::backend("no pending LFM3 VL media projection"))?;
        if rows.shape() != [media.rows, self.width] {
            return Err(Error::backend(
                "VL projection differs from admitted media geometry",
            ));
        }
        match media.source {
            Source::Image => self.images.push(rows),
            Source::Video => self.videos.push(rows),
        }
        self.pending.pop_front();
        Ok(())
    }

    /// Publishes decoder ingress once every visual invocation has completed.
    pub fn finish(self, context: &T::Context) -> Result<ProjectedVisionPrompt<T>, Error> {
        if !self.pending.is_empty() {
            return Err(Error::backend("LFM3 VL prompt has unprojected media"));
        }
        let concatenate = |mut rows: Vec<T>| match rows.len() {
            0 | 1 => Ok(rows.pop()),
            _ => T::concatenate(&rows, 0, context).map(Some),
        };
        let image = concatenate(self.images)?;
        let video = concatenate(self.videos)?;
        let prompt = VisionPrompt::new(
            &self.ids,
            self.markers,
            self.width,
            image.as_ref(),
            video.as_ref(),
            context,
        )?;
        Ok(ProjectedVisionPrompt {
            tokens: prompt.0.tokens,
            spans: prompt.0.spans,
            ids: self.ids,
            width: self.width,
            image,
            video,
        })
    }
}

/// Decoder-ready VL input retaining expanded IDs and projected media rows.
#[derive(Clone)]
pub struct ProjectedVisionPrompt<T> {
    tokens: T,
    spans: std::sync::Arc<[Span]>,
    ids: Vec<i32>,
    width: i32,
    image: Option<T>,
    video: Option<T>,
}

impl<T: Tensor> ProjectedVisionPrompt<T> {
    pub(super) fn retained_values(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.tokens)
            .chain(self.image.iter())
            .chain(self.video.iter())
    }

    pub(super) fn retained_metadata_bytes(&self) -> Option<usize> {
        self.ids
            .capacity()
            .checked_mul(std::mem::size_of::<i32>())?
            .checked_add(self.spans.len().checked_mul(std::mem::size_of::<Span>())?)
    }

    /// Borrows the exact media-conditioned decoder input.
    pub fn prompt(&self) -> VisionPrompt<'_, T> {
        VisionPrompt(ProjectedPrompt {
            tokens: self.tokens.clone(),
            rows: [self.image.as_ref(), self.video.as_ref()],
            spans: self.spans.clone(),
            width: self.width,
            length: self.ids.len() as i32,
        })
    }

    /// Expanded decoder token IDs in prompt order.
    pub fn token_ids(&self) -> &[i32] {
        &self.ids
    }
}
