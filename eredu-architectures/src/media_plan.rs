//! Backend-neutral prepared-media admission and workspace plans.
//!
//! Concrete backends extract shapes and small metadata values from their
//! native arrays. Architecture policy validates those values and returns the
//! decoder positions and scalar workspace implied by the family equations.

use eredu_core::{
    CapabilityError, InputExtent, InputMetadataKey, InputModalities, InputModality,
    InputPartDescriptor, InputPayloadKind, InputTensorIdentity, PreparedInputIdentity,
};
use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedModelInput};

use crate::qwen::{
    hybrid::{HybridConfig, ParsedHybridConfig},
    vision::{VisionAttentionPolicy, VisionConfig},
    vl::ModelArgs as QwenVlModelArgs,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct MetadataValues<T> {
    shape: Vec<u64>,
    values: Vec<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MediaAdmissionInput {
    descriptor: InputPartDescriptor,
    payload_shape: Vec<u64>,
    patch_grid: Option<MetadataValues<i32>>,
    patch_positions: Option<MetadataValues<i32>>,
    audio_mask: Option<MetadataValues<bool>>,
}

impl MediaAdmissionInput {
    fn modality(&self) -> InputModality {
        self.descriptor.modality()
    }

    fn patch_extent(&self) -> Option<[usize; 3]> {
        self.descriptor.extents().find_map(|extent| match extent {
            InputExtent::PatchGrid {
                time,
                height,
                width,
            } => Some([time, height, width]),
            InputExtent::AudioValidFrames(_) => None,
            _ => None,
        })
    }

    fn audio_valid_frames(&self) -> Option<usize> {
        self.descriptor.extents().find_map(|extent| match extent {
            InputExtent::AudioValidFrames(frames) => Some(frames),
            InputExtent::PatchGrid { .. } => None,
            _ => None,
        })
    }
}

fn payload_name(kind: InputPayloadKind) -> &'static str {
    match kind {
        InputPayloadKind::TokenIds => "token-ID",
        InputPayloadKind::Tensor => "tensor",
        InputPayloadKind::Embeddings => "embedding",
        _ => unreachable!("unsupported input payload kind requires explicit admission"),
    }
}

fn u64_shape(identity: &InputTensorIdentity) -> Result<Vec<u64>, CapabilityError> {
    identity
        .shape()
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "prepared-input tensor dimension",
            })
        })
        .collect()
}

fn inspect_part<Tensor>(
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<MediaAdmissionInput, CapabilityError> {
    let descriptor = input
        .descriptor(&|tensor| inspector.identity(tensor))
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let inspect_metadata = descriptor.payload_kind() == InputPayloadKind::Tensor
        && descriptor.modality() != InputModality::Text;

    let metadata_shape = |key| descriptor.metadata_value(key).map(u64_shape).transpose();
    let i32_metadata = |key| -> Result<Option<MetadataValues<i32>>, CapabilityError> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: inspector.i32_values(tensor)?,
                })
            })
            .transpose()
    };
    let bool_metadata = |key| -> Result<Option<MetadataValues<bool>>, CapabilityError> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: inspector.bool_values(tensor)?,
                })
            })
            .transpose()
    };
    let payload_shape = u64_shape(descriptor.payload())?;
    let patch_grid = i32_metadata(InputMetadataKey::PatchGrid)?;
    let patch_positions = i32_metadata(InputMetadataKey::PatchPositions)?;
    let audio_mask = bool_metadata(InputMetadataKey::AudioMask)?;
    let media = MediaAdmissionInput {
        descriptor,
        payload_shape,
        patch_grid,
        patch_positions,
        audio_mask,
    };
    Ok(media)
}

/// Architecture-owned admission and accounting plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedInputPartPlan {
    /// Text decoder positions supplied as token IDs or accepted embeddings.
    Text {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained by decoder input assembly.
        modality: InputModality,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native media requiring tower execution.
    Media {
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl PreparedInputPartPlan {
    /// Exact decoder positions after architecture-owned media preparation.
    pub const fn decoder_positions(&self) -> u64 {
        match self {
            Self::Text { positions } | Self::Projected { positions, .. } => *positions,
            Self::Media { shape } => shape.decoder_positions,
        }
    }
}

/// Architecture-derived geometry and conservative execution workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaShapePlan {
    /// Exact packed capture axes for towers with declared internal hooks.
    pub capture_media: Option<eredu_core::capture::CaptureMediaShape>,
    /// Decoder positions occupied by the projected media.
    pub decoder_positions: u64,
    /// Conservative count of temporary execution scalars.
    pub execution_workspace_scalars: u64,
}

/// Resolves independently packed tower axes before observation transaction admission.
/// Projected caller inputs contribute no tower invocation.
pub fn capture_media_shape(
    plans: &[PreparedInputPartPlan],
) -> Result<Option<eredu_core::capture::CaptureMediaShape>, CapabilityError> {
    let mut result: Option<eredu_core::capture::CaptureMediaShape> = None;
    for plan in plans {
        if let PreparedInputPartPlan::Media { shape } = plan {
            if let Some(media) = shape.capture_media {
                let current = result.get_or_insert(eredu_core::capture::CaptureMediaShape {
                    patches: 0,
                    projected: 0,
                });
                current.patches = current.patches.checked_add(media.patches).ok_or(
                    CapabilityError::ArithmeticOverflow {
                        operation: "capture media patches",
                    },
                )?;
                current.projected = current.projected.checked_add(media.projected).ok_or(
                    CapabilityError::ArithmeticOverflow {
                        operation: "capture projected media",
                    },
                )?;
            }
        }
    }
    Ok(result)
}

/// Architecture-owned Qwen image/video ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenVisionIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after spatial merging.
    pub placeholder_count: u64,
    /// Validated `(time, height, width)` rows consumed by Qwen position policy.
    pub patch_grid: Vec<(i32, i32, i32)>,
}

/// Qwen3-VL admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenVlInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width text embeddings supplied directly by the caller.
    ProjectedText {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Media {
        /// Validated placeholder and patch-grid ingress.
        ingress: QwenVisionIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

/// Shared Qwen text/projected/image/video admission plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained for capability accounting.
        modality: InputModality,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Media {
        /// Validated placeholder and patch-grid ingress.
        ingress: QwenVisionIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

/// Gemma 4 admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4InputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained by the decoder input.
        modality: InputModality,
        /// Architecture-selected placeholder token repeated under the embeddings.
        placeholder_token_id: u32,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Vision {
        /// Architecture-selected placeholder token.
        placeholder_token_id: u32,
        /// Validated execution ingress geometry.
        ingress: crate::gemma4::VisionIngressPartPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
    /// Model-native audio input.
    Audio {
        /// Architecture-selected placeholder token.
        placeholder_token_id: u32,
        /// Validated execution ingress geometry.
        ingress: crate::gemma4::AudioIngressPartPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<Gemma4InputPartPlan> for PreparedInputPartPlan {
    fn from(plan: Gemma4InputPartPlan) -> Self {
        match plan {
            Gemma4InputPartPlan::TextTokens { positions } => Self::Text { positions },
            Gemma4InputPartPlan::Projected {
                modality,
                positions,
                ..
            } => Self::Projected {
                modality,
                positions,
            },
            Gemma4InputPartPlan::Vision { shape, .. }
            | Gemma4InputPartPlan::Audio { shape, .. } => Self::Media { shape },
        }
    }
}

impl From<QwenVlInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: QwenVlInputPartPlan) -> Self {
        match plan {
            QwenVlInputPartPlan::TextTokens { positions }
            | QwenVlInputPartPlan::ProjectedText { positions } => Self::Text { positions },
            QwenVlInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

impl From<QwenInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: QwenInputPartPlan) -> Self {
        match plan {
            QwenInputPartPlan::TextTokens { positions } => Self::Text { positions },
            QwenInputPartPlan::Projected {
                modality,
                positions,
            } => Self::Projected {
                modality,
                positions,
            },
            QwenInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

/// Inkling admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InklingInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width image or audio embeddings supplied by the caller.
    Projected {
        /// Semantic modality retained by decoder input assembly.
        modality: InputModality,
        /// Architecture-selected placeholder token repeated under the embeddings.
        placeholder_token_id: u32,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or audio input.
    Media {
        /// Semantic modality consumed by the tower.
        modality: InputModality,
        /// Architecture-selected placeholder and span.
        ingress: InklingIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<InklingInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: InklingInputPartPlan) -> Self {
        match plan {
            InklingInputPartPlan::TextTokens { positions } => Self::Text { positions },
            InklingInputPartPlan::Projected {
                modality,
                positions,
                ..
            } => Self::Projected {
                modality,
                positions,
            },
            InklingInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

/// Muse-Glimmer admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuseInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Vision {
        /// Semantic modality consumed by the vision tower.
        modality: InputModality,
        /// Architecture-selected placeholder span and patch grid.
        ingress: MuseVisionIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
    /// Mono waveform input with independently padded clip boundaries.
    Audio {
        /// Exact waveform-to-decoder geometry and placeholder identity.
        ingress: MuseAudioIngressPlan,
        /// Decoder and activation accounting for the audio pyramid.
        shape: MediaShapePlan,
    },
}

/// Raw waveform geometry admitted before native construction or execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseAudioIngressPlan {
    /// Unpadded PCM samples in this independent attachment.
    pub valid_samples: u64,
    /// Right-padded samples, exactly divisible by the tower cadence.
    pub padded_samples: u64,
    /// Placeholder token from the selected decoder configuration.
    pub placeholder_token_id: u32,
    /// Audio positions emitted by the complete convolutional pyramid.
    pub placeholder_count: u64,
}

impl From<MuseInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: MuseInputPartPlan) -> Self {
        match plan {
            MuseInputPartPlan::TextTokens { positions } => Self::Text { positions },
            MuseInputPartPlan::Vision { shape, .. } | MuseInputPartPlan::Audio { shape, .. } => {
                Self::Media { shape }
            }
        }
    }
}

/// Architecture-owned Inkling ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after architecture-specific projection.
    pub placeholder_count: u64,
}

/// Architecture-owned Muse-Glimmer ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseVisionIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after spatial merging.
    pub placeholder_count: u64,
    /// Validated `(time, height, width)` rows consumed by vision execution.
    pub patch_grid: Vec<(i32, i32, i32)>,
}

/// Whole-request architecture admission coupled to exact prepared tensor identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedCompositeInput<P> {
    identity: PreparedInputIdentity,
    parts: Vec<P>,
    decoder_positions: u64,
    active_modalities: InputModalities,
}

impl<P> AdmittedCompositeInput<P> {
    /// Attaches architecture-owned execution context while preserving the admitted
    /// request identity and neutral accounting. Only architecture constructors may
    /// replace these private plans.
    pub(crate) fn map_parts<Q>(
        self,
        mut map: impl FnMut(usize, P) -> Q,
    ) -> AdmittedCompositeInput<Q> {
        AdmittedCompositeInput {
            identity: self.identity,
            parts: self
                .parts
                .into_iter()
                .enumerate()
                .map(|(index, part)| map(index, part))
                .collect(),
            decoder_positions: self.decoder_positions,
            active_modalities: self.active_modalities,
        }
    }

    /// Identity recomputed from the exact tensor handles admitted by the architecture.
    pub const fn identity(&self) -> &PreparedInputIdentity {
        &self.identity
    }

    /// Ordered architecture-owned plan for every prepared input part.
    pub fn parts(&self) -> &[P] {
        &self.parts
    }

    /// Admitted decoder batch and sequence dimensions after assembling all parts.
    /// Current composite ingress contracts validate one sequence per request.
    pub const fn decoder_shape(&self) -> [u64; 2] {
        [1, self.decoder_positions]
    }

    /// Total decoder positions occupied by all ordered parts.
    pub const fn decoder_positions(&self) -> u64 {
        self.decoder_positions
    }

    /// Modalities that activate request-optional roots for this exact input.
    pub const fn active_modalities(&self) -> InputModalities {
        self.active_modalities
    }
}

pub(crate) trait CompositePartPlan {
    fn decoder_positions(&self) -> u64;
}

impl CompositePartPlan for QwenVlInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::ProjectedText { positions } => *positions,
            Self::Media { shape, .. } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for QwenInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::Projected { positions, .. } => *positions,
            Self::Media { shape, .. } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for Gemma4InputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::Projected { positions, .. } => *positions,
            Self::Vision { shape, .. } | Self::Audio { shape, .. } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for InklingInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::Projected { positions, .. } => *positions,
            Self::Media { shape, .. } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for MuseInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } => *positions,
            Self::Vision { shape, .. } | Self::Audio { shape, .. } => shape.decoder_positions,
        }
    }
}

pub(crate) fn admit_composite_input<Tensor, P>(
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
    mut admit_part: impl FnMut(&PreparedInputPart<Tensor>) -> Result<P, CapabilityError>,
) -> Result<AdmittedCompositeInput<P>, CapabilityError>
where
    P: CompositePartPlan,
{
    let descriptors = input
        .parts()
        .iter()
        .map(|part| part.descriptor(&|tensor| inspector.identity(tensor)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let identity = PreparedInputIdentity::new(descriptors)
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    if &identity != input.identity() {
        return Err(CapabilityError::Observation(
            "prepared-input tensor identity changed before architecture admission".into(),
        ));
    }
    let mut active_modalities = InputModalities {
        text: false,
        image: false,
        audio: false,
        video: false,
    };
    let mut decoder_positions = 0u64;
    let mut parts = Vec::with_capacity(input.len());
    for part in input.parts() {
        let plan = admit_part(part)?;
        decoder_positions = decoder_positions
            .checked_add(plan.decoder_positions())
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "composite decoder-position total",
            })?;
        match part.modality() {
            InputModality::Text => active_modalities.text = true,
            InputModality::Image => active_modalities.image = true,
            InputModality::Audio => active_modalities.audio = true,
            InputModality::Video => active_modalities.video = true,
            _ => {}
        }
        parts.push(plan);
    }
    if decoder_positions == 0 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: "replicated composite".into(),
            reason: "prepared input occupies no decoder positions".into(),
        });
    }
    Ok(AdmittedCompositeInput {
        identity,
        parts,
        decoder_positions,
        active_modalities,
    })
}

/// Admits a complete Qwen3-VL request before composite graph execution.
pub fn admit_qwen_vl_input<Tensor>(
    args: &QwenVlModelArgs,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<QwenVlInputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        qwen_vl_input_part(args, part, inspector)
    })
}

/// Admits a complete conditional Qwen request before composite graph execution.
pub fn admit_qwen_hybrid_input<Tensor>(
    args: &ParsedHybridConfig,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<QwenInputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        qwen_hybrid_input_part(args, part, inspector)
    })
}

/// Admits a complete Gemma 4 request before composite graph execution.
pub fn admit_gemma4_input<Tensor>(
    args: &crate::gemma4::FamilyConfig,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<Gemma4InputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        gemma4_input_part(args, part, inspector)
    })
}

/// Admits a complete Inkling request before composite graph execution.
pub fn admit_inkling_input<Tensor>(
    args: &crate::inkling::ModelArgs,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<InklingInputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        inkling_input_part(args, part, inspector)
    })
}

/// Admits a complete Muse-Glimmer request before composite graph execution.
pub fn admit_muse_input<Tensor>(
    args: &crate::muse::DecoderConfig,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<MuseInputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        muse_input_part(args, part, inspector)
    })
}

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

fn nonzero_positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    let value = positive(value, field)?;
    if value == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field,
            detail: "expected a positive value, got zero".into(),
        });
    }
    Ok(value)
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn dimension(shape: &[u64], axis: usize, field: &'static str) -> Result<u64, CapabilityError> {
    shape
        .get(axis)
        .copied()
        .ok_or(CapabilityError::InvalidConfiguration {
            field,
            detail: format!("shape {shape:?} has no axis {axis}"),
        })
}

fn unsupported(architecture: &str, reason: impl Into<String>) -> CapabilityError {
    CapabilityError::UnsupportedInput {
        architecture: architecture.into(),
        reason: reason.into(),
    }
}

fn qwen_attention_chunk_squares(
    grid: &[(i32, i32, i32)],
    merge: u64,
    window_size: i32,
    patch_size: i32,
) -> Result<(u64, u64), CapabilityError> {
    let patch = nonzero_positive(patch_size, "Qwen vision patch size")?;
    let window_pixels = nonzero_positive(window_size, "Qwen vision window size")?;
    let merger_window = window_pixels / merge / patch;
    if merger_window == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "window_size",
            detail: format!(
                "Qwen vision window {window_pixels} is too small for merge {merge} and patch {patch}"
            ),
        });
    }
    let merge_area = checked_mul(merge, merge, "Qwen attention merge area")?;
    let merge_area_square =
        checked_mul(merge_area, merge_area, "Qwen attention merge-area square")?;
    let mut full_squares = 0u64;
    let mut window_squares = 0u64;
    for (time, height, width) in grid {
        let time = nonzero_positive(*time, "Qwen grid time")?;
        let height = nonzero_positive(*height, "Qwen grid height")?;
        let width = nonzero_positive(*width, "Qwen grid width")?;
        if height % merge != 0 || width % merge != 0 {
            return Err(CapabilityError::InvalidConfiguration {
                field: "patch_grid",
                detail: format!(
                    "Qwen grid ({height}, {width}) is not divisible by spatial merge {merge}"
                ),
            });
        }
        let full_length = checked_mul(height, width, "Qwen full-attention chunk length")?;
        full_squares = checked_add(
            full_squares,
            checked_mul(
                time,
                checked_mul(full_length, full_length, "Qwen full-attention chunk square")?,
                "Qwen full-attention temporal chunks",
            )?,
            "Qwen full-attention chunk-square total",
        )?;

        let merged_height = height / merge;
        let merged_width = width / merge;
        let height_full = merged_height / merger_window;
        let height_remainder = merged_height % merger_window;
        let width_full = merged_width / merger_window;
        let width_remainder = merged_width % merger_window;
        let window_square = checked_mul(merger_window, merger_window, "Qwen merger-window square")?;
        let height_square_sum = checked_add(
            checked_mul(
                height_full,
                window_square,
                "Qwen full height-window squares",
            )?,
            checked_mul(
                height_remainder,
                height_remainder,
                "Qwen remainder height-window square",
            )?,
            "Qwen height-window square sum",
        )?;
        let width_square_sum = checked_add(
            checked_mul(width_full, window_square, "Qwen full width-window squares")?,
            checked_mul(
                width_remainder,
                width_remainder,
                "Qwen remainder width-window square",
            )?,
            "Qwen width-window square sum",
        )?;
        let item_window_squares = checked_mul(
            checked_mul(
                height_square_sum,
                width_square_sum,
                "Qwen merged window-area squares",
            )?,
            merge_area_square,
            "Qwen patch window-area squares",
        )?;
        window_squares = checked_add(
            window_squares,
            checked_mul(time, item_window_squares, "Qwen temporal window chunks")?,
            "Qwen window-attention chunk-square total",
        )?;
    }
    Ok((full_squares, window_squares))
}

/// Derives prepared Qwen image/video geometry from normalized vision policy.
fn qwen_vision(
    config: &VisionConfig,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if !matches!(
        input.modality(),
        InputModality::Image | InputModality::Video
    ) {
        return Err(unsupported(
            architecture,
            format!(
                "{} is not a Qwen vision modality",
                input.modality().as_str()
            ),
        ));
    }
    if input.payload_shape.len() != 2 {
        return Err(unsupported(
            architecture,
            format!(
                "Qwen prepared vision tensor must be [patches, patch_dims], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let patches = dimension(&input.payload_shape, 0, "Qwen prepared patch count")?;
    let merge = nonzero_positive(config.spatial_merge_size, "spatial_merge_size")?;
    let patch = nonzero_positive(config.patch_size, "Qwen vision patch size")?;
    let expected_patch_dims = checked_mul(
        checked_mul(
            nonzero_positive(config.in_channels, "Qwen vision input channels")?,
            nonzero_positive(
                config.temporal_patch_size,
                "Qwen vision temporal patch size",
            )?,
            "Qwen temporal input channels",
        )?,
        checked_mul(patch, patch, "Qwen vision patch area")?,
        "Qwen vision patch dimensions",
    )?;
    if dimension(&input.payload_shape, 1, "Qwen prepared patch dimensions")? != expected_patch_dims
    {
        return Err(unsupported(
            architecture,
            format!(
                "Qwen prepared patches have width {}, expected {expected_patch_dims}",
                input.payload_shape[1]
            ),
        ));
    }
    let merge_area = checked_mul(merge, merge, "Qwen spatial merge area")?;
    if patches % merge_area != 0 {
        return Err(unsupported(
            architecture,
            format!("Qwen patch count {patches} is not divisible by {merge_area}"),
        ));
    }
    let positions = patches / merge_area;
    let metadata = input
        .patch_grid
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Qwen media has no grid_thw metadata"))?;
    if metadata.shape.len() != 2 || metadata.shape[1] != 3 || metadata.shape[0] == 0 {
        return Err(unsupported(
            architecture,
            format!(
                "patch grid must be shaped [items, 3], got {:?}",
                metadata.shape
            ),
        ));
    }
    let expected_values = checked_mul(metadata.shape[0], 3, "Qwen patch-grid scalar count")?;
    if u64::try_from(metadata.values.len()).ok() != Some(expected_values) {
        return Err(unsupported(
            architecture,
            "Qwen patch grid has an incomplete row",
        ));
    }
    let grid = metadata
        .values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect::<Vec<_>>();
    let described_patches = grid.iter().try_fold(0u64, |total, (time, height, width)| {
        let item_patches = checked_mul(
            checked_mul(
                nonzero_positive(*time, "Qwen grid time")?,
                nonzero_positive(*height, "Qwen grid height")?,
                "Qwen grid time-height",
            )?,
            nonzero_positive(*width, "Qwen grid width")?,
            "Qwen grid item patches",
        )?;
        checked_add(total, item_patches, "Qwen described patch total")
    })?;
    if described_patches != patches {
        return Err(unsupported(
            architecture,
            format!("Qwen grid describes {described_patches} patches but payload has {patches}"),
        ));
    }
    let (full_chunk_squares, window_chunk_squares) =
        qwen_attention_chunk_squares(&grid, merge, config.window_size, config.patch_size)?;
    let depth =
        u64::try_from(config.layer_count()).map_err(|_| CapabilityError::ArithmeticOverflow {
            operation: "Qwen vision depth",
        })?;
    let full_blocks = u64::try_from(
        config
            .layer_schedule
            .iter()
            .filter(|policy| matches!(policy.attention, VisionAttentionPolicy::Full))
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Qwen full-attention block count",
    })?;
    let window_blocks = depth - full_blocks;
    let heads = positive(config.num_heads, "Qwen vision heads")?;
    let hidden = positive(config.hidden_size, "Qwen vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Qwen vision intermediate size")?;
    let out_hidden = positive(config.out_hidden_size, "Qwen vision output size")?;
    let patch_hidden = checked_mul(patches, hidden, "Qwen patch hidden elements")?;
    let patch_intermediate =
        checked_mul(patches, intermediate, "Qwen patch intermediate elements")?;
    let per_block = checked_add(
        checked_mul(32, patch_hidden, "Qwen block hidden workspace")?,
        checked_mul(6, patch_intermediate, "Qwen block intermediate workspace")?,
        "Qwen block workspace",
    )?;
    let block_workspace = checked_mul(depth, per_block, "Qwen all-block workspace")?;
    let full_attention = checked_mul(
        checked_mul(
            checked_mul(
                full_blocks,
                full_chunk_squares,
                "Qwen full-attention blocks",
            )?,
            heads,
            "Qwen full-attention heads",
        )?,
        2,
        "Qwen full-attention score/probability bound",
    )?;
    let window_attention = checked_mul(
        checked_mul(
            checked_mul(
                window_blocks,
                window_chunk_squares,
                "Qwen window-attention blocks",
            )?,
            heads,
            "Qwen window-attention heads",
        )?,
        2,
        "Qwen window-attention score/probability bound",
    )?;
    let merge_width = checked_mul(hidden, merge_area, "Qwen merger width")?;
    let merger_output = checked_mul(
        positions,
        checked_add(
            checked_mul(12, merge_width, "Qwen merger hidden workspace")?,
            checked_mul(6, out_hidden, "Qwen merger output workspace")?,
            "Qwen merger per-position workspace",
        )?,
        "Qwen merger workspace",
    )?;
    let mergers = checked_add(
        1,
        u64::try_from(config.deepstack_layer_count()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "Qwen deepstack merger count",
            }
        })?,
        "Qwen merger count",
    )?;
    let graph_scalars = checked_add(
        checked_add(
            checked_mul(16, patch_hidden, "Qwen vision setup workspace")?,
            block_workspace,
            "Qwen setup plus blocks",
        )?,
        checked_add(
            checked_add(full_attention, window_attention, "Qwen attention workspace")?,
            checked_mul(mergers, merger_output, "Qwen all-merger workspace")?,
            "Qwen attention plus mergers",
        )?,
        "Qwen vision graph workspace",
    )?;
    Ok(MediaShapePlan {
        capture_media: None,
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

/// Derives Qwen media geometry when the normalized hybrid policy may omit its
/// vision tower.
fn qwen_hybrid_vision(
    config: Option<&VisionConfig>,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    config
        .ok_or_else(|| unsupported(architecture, "loaded model has no vision configuration"))
        .and_then(|config| qwen_vision(config, input, architecture))
}

fn qwen_vision_ingress_with_shape(
    config: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<(QwenVisionIngressPlan, MediaShapePlan), CapabilityError> {
    let shape = qwen_hybrid_vision(config, input, architecture)?;
    let token_id = match input.modality() {
        InputModality::Image => image_token_id,
        InputModality::Video => video_token_id,
        InputModality::Audio | InputModality::Text => None,
        _ => None,
    }
    .ok_or_else(|| unsupported(architecture, "prepared media placeholder token is absent"))?;
    let placeholder_token_id =
        u32::try_from(token_id).map_err(|_| CapabilityError::InvalidConfiguration {
            field: "Qwen media placeholder token",
            detail: format!("expected a non-negative token ID, got {token_id}"),
        })?;
    let metadata = input
        .patch_grid
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Qwen media has no grid_thw metadata"))?;
    let patch_grid = metadata
        .values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    Ok((
        QwenVisionIngressPlan {
            placeholder_token_id,
            placeholder_count: shape.decoder_positions,
            patch_grid,
        },
        shape,
    ))
}

fn batch_one_sequence(
    shape: &[u64],
    rank: usize,
    name: &str,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if shape.len() != rank || shape.first() != Some(&1) || shape.get(1).copied().unwrap_or(0) == 0 {
        return Err(unsupported(
            architecture,
            format!("prepared {name} must be batch-one with rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape[1])
}

/// Validates one prepared Qwen3-VL part and derives the exact prefill and
/// capability-accounting plan consumed by a concrete backend.
pub fn qwen_vl_input_part<Tensor>(
    args: &QwenVlModelArgs,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenVlInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(QwenVlInputPartPlan::TextTokens {
            positions: batch_one_sequence(&shape, 2, "text token IDs", &args.model_type)?,
        }),
        (InputModality::Text, InputPayloadKind::Embeddings) => {
            let positions = batch_one_sequence(&shape, 3, "text embeddings", &args.model_type)?;
            let hidden = positive(args.text.hidden_size, "Qwen3-VL hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared text embeddings must have hidden width {hidden}, got {:?}",
                        shape
                    ),
                ));
            }
            Ok(QwenVlInputPartPlan::ProjectedText { positions })
        }
        (InputModality::Image | InputModality::Video, InputPayloadKind::Tensor) => {
            let (ingress, shape) = qwen_vision_ingress_with_shape(
                Some(&args.vision),
                Some(args.image_token_id),
                Some(args.video_token_id),
                &inspected,
                &args.model_type,
            )?;
            Ok(QwenVlInputPartPlan::Media { ingress, shape })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Qwen3-VL does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

pub(crate) fn qwen_input_part_with_policy<Tensor>(
    model_type: &str,
    hidden_size: i32,
    vision: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(QwenInputPartPlan::TextTokens {
            positions: batch_one_sequence(&shape, 2, "text token IDs", model_type)?,
        }),
        (
            modality @ (InputModality::Text | InputModality::Image | InputModality::Video),
            InputPayloadKind::Embeddings,
        ) => {
            let positions = batch_one_sequence(
                &shape,
                3,
                &format!("{} embeddings", modality.as_str()),
                model_type,
            )?;
            let hidden = positive(hidden_size, "Qwen hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    model_type,
                    format!(
                        "prepared {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            Ok(QwenInputPartPlan::Projected {
                modality,
                positions,
            })
        }
        (InputModality::Image | InputModality::Video, InputPayloadKind::Tensor) => {
            let (ingress, shape) = qwen_vision_ingress_with_shape(
                vision,
                image_token_id,
                video_token_id,
                &inspected,
                model_type,
            )?;
            Ok(QwenInputPartPlan::Media { ingress, shape })
        }
        (modality, payload) => Err(unsupported(
            model_type,
            format!(
                "Qwen does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

/// Validates one prepared conditional Qwen3.5 input part and derives the exact
/// prefill and capability-accounting plan consumed by a concrete backend.
pub fn qwen_hybrid_input_part<Tensor>(
    args: &ParsedHybridConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenInputPartPlan, CapabilityError> {
    qwen_input_part_with_policy(
        &args.text.model_type,
        args.text.hidden_size,
        args.vision.as_ref(),
        args.image_token_id,
        args.video_token_id,
        input,
        inspector,
    )
}

/// Validates one prepared text-only Qwen3.5/Qwen3-Next input part.
pub fn qwen_hybrid_text_input_part<Tensor>(
    args: &HybridConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(QwenInputPartPlan::TextTokens {
            positions: batch_one_sequence(&shape, 2, "text token IDs", &args.model_type)?,
        }),
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "text-only Qwen hybrid does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

fn gemma_batch_one_sequence(
    shape: &[u64],
    rank: usize,
    name: &str,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if shape.len() != rank || shape.first() != Some(&1) || shape.get(1).copied().unwrap_or(0) == 0 {
        return Err(unsupported(
            architecture,
            format!("prepared Gemma {name} must be batch-one with rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape[1])
}

fn gemma_placeholder_token(
    args: &crate::gemma4::FamilyConfig,
    modality: InputModality,
) -> Result<u32, CapabilityError> {
    let token = match modality {
        InputModality::Text => Some(args.text.pad_token_id),
        InputModality::Image => args.image_token_id,
        InputModality::Video => args.video_token_id,
        InputModality::Audio => args.audio_token_id,
        _ => None,
    };
    token
        .and_then(|token| u32::try_from(token).ok())
        .ok_or_else(|| {
            unsupported(
                &args.model_type,
                format!("Gemma 4 has no valid {} placeholder", modality.as_str()),
            )
        })
}

/// Validates one prepared Gemma 4 part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn gemma4_input_part<Tensor>(
    args: &crate::gemma4::FamilyConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<Gemma4InputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(Gemma4InputPartPlan::TextTokens {
            positions: gemma_batch_one_sequence(&shape, 2, "text token IDs", &args.model_type)?,
        }),
        (modality, InputPayloadKind::Embeddings) => {
            let positions = gemma_batch_one_sequence(
                &shape,
                3,
                &format!("{} embeddings", modality.as_str()),
                &args.model_type,
            )?;
            let hidden = positive(args.text.hidden_size, "Gemma text hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            Ok(Gemma4InputPartPlan::Projected {
                modality,
                placeholder_token_id: gemma_placeholder_token(args, modality)?,
                positions,
            })
        }
        (modality @ (InputModality::Image | InputModality::Video), InputPayloadKind::Tensor) => {
            if inspected.patch_grid.is_none() {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} input has no patch grid",
                        modality.as_str()
                    ),
                ));
            }
            let extent = inspected.patch_extent().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} input has no host-known patch extent",
                        modality.as_str()
                    ),
                )
            })?;
            let grid = inspected.patch_grid.as_ref().expect("checked above");
            let extent_i32 = extent.map(|value| {
                i32::try_from(value).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma patch extent",
                })
            });
            let [time, height, width] = extent_i32;
            let extent_i32 = [time?, height?, width?];
            if grid.shape != [1, 3]
                || grid.values.as_slice() != extent_i32
                || grid.values.len() != 3
            {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} patch grid must be one row matching extent {extent:?}",
                        modality.as_str()
                    ),
                ));
            }
            let vision = args.vision.as_ref().ok_or_else(|| {
                unsupported(&args.model_type, "loaded Gemma model has no vision tower")
            })?;
            let padded_patches =
                i32::try_from(shape[1]).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma padded vision patch count",
                })?;
            let ingress =
                crate::gemma4::VisionIngressPartPlan::new(vision, extent_i32, padded_patches)
                    .map_err(|error| unsupported(&args.model_type, error.to_string()))?;
            let media_shape = gemma4(args, &inspected)?;
            let valid_patches = gemma_valid_patch_count(
                inspected
                    .patch_positions
                    .as_ref()
                    .expect("Gemma media shape validation requires patch positions"),
                &args.model_type,
            )?;
            if valid_patches != ingress.valid_patches as u64
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma patch extent and position metadata disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Vision {
                placeholder_token_id: gemma_placeholder_token(args, modality)?,
                ingress,
                shape: media_shape,
            })
        }
        (InputModality::Audio, InputPayloadKind::Tensor) => {
            let valid_frames = inspected.audio_valid_frames().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "prepared Gemma audio has no host-known valid-frame extent",
                )
            })?;
            let padded_frames =
                i32::try_from(shape[1]).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma padded audio frame count",
                })?;
            let valid_frames_i32 =
                i32::try_from(valid_frames).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma valid audio frame count",
                })?;
            let ingress = crate::gemma4::AudioIngressPartPlan::new(valid_frames_i32, padded_frames)
                .map_err(|error| unsupported(&args.model_type, error.to_string()))?;
            let media_shape = gemma4(args, &inspected)?;
            let valid_mask_frames = inspected
                .audio_mask
                .as_ref()
                .expect("Gemma media shape validation requires an audio mask")
                .values
                .iter()
                .filter(|value| **value)
                .count();
            if valid_mask_frames != valid_frames
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma audio valid-frame extent and mask disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Audio {
                placeholder_token_id: gemma_placeholder_token(args, InputModality::Audio)?,
                ingress,
                shape: media_shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Gemma 4 does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

fn gemma_valid_patch_count(
    positions: &MetadataValues<i32>,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if positions.shape.len() != 3 || positions.shape[0] != 1 || positions.shape[2] != 2 {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma patch positions must be [1, patches, 2], got {:?}",
                positions.shape
            ),
        ));
    }
    let expected = checked_mul(positions.shape[1], 2, "Gemma patch-position scalar count")?;
    if u64::try_from(positions.values.len()).ok() != Some(expected) {
        return Err(unsupported(
            architecture,
            "Gemma patch positions do not match their declared shape",
        ));
    }
    u64::try_from(
        positions
            .values
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|pair| pair[0] >= 0 && pair[1] >= 0)
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Gemma valid patch count",
    })
}

fn gemma_vision(
    config: &crate::gemma4::VisionConfig,
    text_hidden: u64,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.payload_shape.len() != 3 || input.payload_shape[0] != 1 {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared vision tensor must be [1, patches, patch_dims], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let patch = nonzero_positive(config.patch_size, "Gemma vision patch size")?;
    let expected_patch_dims = checked_mul(
        3,
        checked_mul(patch, patch, "Gemma vision patch area")?,
        "Gemma vision patch dimensions",
    )?;
    if input.payload_shape[2] != expected_patch_dims {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared patches have width {}, expected {expected_patch_dims}",
                input.payload_shape[2]
            ),
        ));
    }
    let position_ids = input
        .patch_positions
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Gemma media has no patch positions"))?;
    let valid_patches = gemma_valid_patch_count(position_ids, architecture)?;
    let pool = nonzero_positive(config.pooling_kernel_size, "Gemma pooling kernel")?;
    let pool_area = checked_mul(pool, pool, "Gemma pooling area")?;
    if valid_patches % pool_area != 0 {
        return Err(unsupported(
            architecture,
            format!("Gemma valid patch count {valid_patches} is not divisible by {pool_area}"),
        ));
    }
    let positions = valid_patches / pool_area;
    let padded_patches = input.payload_shape[1];
    if position_ids.shape[1] != padded_patches {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma patch positions {:?} do not match prepared vision payload {:?}",
                position_ids.shape, input.payload_shape
            ),
        ));
    }
    let hidden = positive(config.hidden_size, "Gemma vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Gemma vision intermediate size")?;
    let depth = positive(config.num_hidden_layers, "Gemma vision depth")?;
    let patch_hidden = checked_mul(padded_patches, hidden, "Gemma vision patch hidden elements")?;
    let per_layer = checked_add(
        checked_mul(48, patch_hidden, "Gemma vision layer hidden workspace")?,
        checked_mul(
            8,
            checked_mul(
                padded_patches,
                intermediate,
                "Gemma vision intermediate elements",
            )?,
            "Gemma vision MLP workspace",
        )?,
        "Gemma vision layer workspace",
    )?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, hidden, "Gemma pooled vision workspace")?,
            checked_mul(8, text_hidden, "Gemma projected vision workspace")?,
            "Gemma vision output workspace per position",
        )?,
        "Gemma vision output workspace",
    )?;
    let graph_scalars = checked_add(
        checked_mul(20, patch_hidden, "Gemma vision setup workspace")?,
        checked_add(
            checked_mul(depth, per_layer, "Gemma all vision layers")?,
            output_workspace,
            "Gemma layers plus output workspace",
        )?,
        "Gemma vision graph workspace",
    )?;
    Ok(MediaShapePlan {
        capture_media: None,
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

fn gemma_audio(
    config: &crate::gemma4::AudioConfig,
    text_hidden: u64,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.payload_shape.len() != 3
        || input.payload_shape[0] != 1
        || input.payload_shape[2] != 128
    {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared audio tensor must be [1, frames, 128], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let mask = input
        .audio_mask
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Gemma audio has no frame mask"))?;
    let frames = input.payload_shape[1];
    if mask.shape != [1, frames] || u64::try_from(mask.values.len()).ok() != Some(frames) {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma audio mask must be [1, {frames}], got {:?}",
                mask.shape
            ),
        ));
    }
    let valid_frames =
        u64::try_from(mask.values.iter().filter(|value| **value).count()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "Gemma valid audio frame count",
            }
        })?;
    let positions = valid_frames.div_ceil(4);
    let sequence = frames.div_ceil(4);
    let hidden = positive(config.hidden_size, "Gemma audio hidden size")?;
    let depth = positive(config.num_hidden_layers, "Gemma audio depth")?;
    let heads = positive(config.num_attention_heads, "Gemma audio heads")?;
    let chunk = nonzero_positive(config.attention_chunk_size, "Gemma audio attention chunk")?;
    let past = nonzero_positive(
        config.attention_context_left.checked_sub(1).ok_or(
            CapabilityError::ArithmeticOverflow {
                operation: "Gemma audio left context",
            },
        )?,
        "Gemma audio left context",
    )?;
    let padded_sequence = checked_mul(
        sequence.div_ceil(chunk),
        chunk,
        "Gemma padded audio sequence",
    )?;
    let chunks = padded_sequence / chunk;
    let attention_elements = checked_mul(
        checked_mul(
            checked_mul(
                checked_mul(chunks, heads, "Gemma audio attention chunk heads")?,
                chunk,
                "Gemma audio attention queries",
            )?,
            checked_add(chunk, past, "Gemma audio attention key bound")?,
            "Gemma audio attention scores",
        )?,
        4,
        "Gemma audio logits/relative/mask/probability workspace",
    )?;
    let layer_workspace = checked_add(
        checked_mul(
            80,
            checked_mul(sequence, hidden, "Gemma audio hidden elements")?,
            "Gemma audio layer hidden workspace",
        )?,
        attention_elements,
        "Gemma audio layer workspace",
    )?;
    let first_frames = frames.div_ceil(2);
    let first_channels = positive(
        *config.subsampling_conv_channels.first().ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no first convolution channel count".into(),
            }
        })?,
        "Gemma audio first convolution channels",
    )?;
    let second_channels = positive(
        *config.subsampling_conv_channels.get(1).ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no second convolution channel count".into(),
            }
        })?,
        "Gemma audio second convolution channels",
    )?;
    let conv_workspace = checked_add(
        checked_mul(
            6,
            checked_mul(
                checked_mul(first_frames, 64, "Gemma first convolution grid")?,
                first_channels,
                "Gemma first convolution elements",
            )?,
            "Gemma first convolution workspace",
        )?,
        checked_mul(
            6,
            checked_mul(
                checked_mul(sequence, 32, "Gemma second convolution grid")?,
                second_channels,
                "Gemma second convolution elements",
            )?,
            "Gemma second convolution workspace",
        )?,
        "Gemma convolution workspace",
    )?;
    let output = positive(config.output_proj_dims, "Gemma audio output size")?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, output, "Gemma audio output workspace")?,
            checked_mul(8, text_hidden, "Gemma audio text projection workspace")?,
            "Gemma audio output workspace per position",
        )?,
        "Gemma audio projected output workspace",
    )?;
    let graph_scalars = checked_add(
        conv_workspace,
        checked_add(
            checked_mul(depth, layer_workspace, "Gemma all audio layers")?,
            output_workspace,
            "Gemma audio layers plus output",
        )?,
        "Gemma audio graph workspace",
    )?;
    Ok(MediaShapePlan {
        capture_media: None,
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

/// Derives prepared Gemma 4 media geometry from normalized family policy.
fn gemma4(
    args: &crate::gemma4::FamilyConfig,
    input: &MediaAdmissionInput,
) -> Result<MediaShapePlan, CapabilityError> {
    let text_hidden = positive(args.text.hidden_size, "Gemma text hidden size")?;
    match input.modality() {
        InputModality::Image | InputModality::Video => args
            .vision
            .as_ref()
            .ok_or_else(|| unsupported(&args.model_type, "loaded model has no vision tower"))
            .and_then(|config| gemma_vision(config, text_hidden, input, &args.model_type)),
        InputModality::Audio => args
            .audio
            .as_ref()
            .ok_or_else(|| unsupported(&args.model_type, "loaded model has no audio tower"))
            .and_then(|config| gemma_audio(config, text_hidden, input, &args.model_type)),
        InputModality::Text => Err(unsupported(
            &args.model_type,
            "text is not a Gemma media modality",
        )),
        _ => Err(unsupported(
            &args.model_type,
            "unknown modality is not supported by Gemma",
        )),
    }
}

/// Derives prepared Inkling media geometry from normalized family policy.
fn inkling(
    args: &crate::inkling::ModelArgs,
    input: &MediaAdmissionInput,
) -> Result<MediaShapePlan, CapabilityError> {
    match input.modality() {
        InputModality::Image => {
            let config = args.vision_config.as_ref().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "loaded Inkling model has no vision configuration",
                )
            })?;
            if input.payload_shape.len() != 5 || input.payload_shape[1..] != [2, 40, 40, 3] {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                        input.payload_shape
                    ),
                ));
            }
            let patches = input.payload_shape[0];
            let text_hidden = positive(config.text_hidden_size, "Inkling vision output size")?;
            let layer_outputs = [
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        8 * 8,
                        "Inkling vision grid",
                    )?,
                    128,
                    "Inkling vision layer 1",
                )?,
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        4 * 4,
                        "Inkling vision grid",
                    )?,
                    512,
                    "Inkling vision layer 2",
                )?,
                checked_mul(
                    checked_mul(patches, 2, "Inkling vision time")?,
                    4_800,
                    "Inkling vision layer 3",
                )?,
                checked_mul(patches, text_hidden, "Inkling vision layer 4")?,
            ];
            let graph_scalars = layer_outputs.iter().try_fold(0u64, |total, value| {
                checked_add(
                    total,
                    checked_mul(12, *value, "Inkling vision layer workspace")?,
                    "Inkling vision graph workspace",
                )
            })?;
            Ok(MediaShapePlan {
                capture_media: None,
                decoder_positions: patches,
                execution_workspace_scalars: graph_scalars,
            })
        }
        InputModality::Audio => {
            let config = args.audio_config.as_ref().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "loaded Inkling model has no audio configuration",
                )
            })?;
            let [1, padded_frames, payload_codebooks] = input.payload_shape.as_slice() else {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling audio tokens must be [1, frames, codebooks], got {:?}",
                        input.payload_shape
                    ),
                ));
            };
            let (padded_frames, payload_codebooks) = (*padded_frames, *payload_codebooks);
            let codebooks = positive(config.num_codebooks, "Inkling audio codebooks")?;
            if payload_codebooks != codebooks {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling audio payload has {payload_codebooks} codebooks, expected {codebooks}"
                    ),
                ));
            }
            let frames = if let Some(mask) = &input.audio_mask {
                if mask.shape != [1, padded_frames]
                    || u64::try_from(mask.values.len()).ok() != Some(padded_frames)
                {
                    return Err(unsupported(
                        &args.model_type,
                        format!(
                            "Inkling audio mask must be [1, {padded_frames}], got {:?}",
                            mask.shape
                        ),
                    ));
                }
                let frames = mask.values.iter().take_while(|value| **value).count();
                if mask.values[frames..].iter().any(|value| *value) {
                    return Err(unsupported(
                        &args.model_type,
                        "Inkling audio mask must describe one valid prefix",
                    ));
                }
                u64::try_from(frames).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Inkling valid audio frame count",
                })?
            } else {
                padded_frames
            };
            let hidden = positive(config.text_hidden_size, "Inkling audio hidden size")?;
            let embedded = checked_mul(
                checked_mul(padded_frames, codebooks, "Inkling audio frame codebooks")?,
                hidden,
                "Inkling audio embedding elements",
            )?;
            let reduced = checked_mul(padded_frames, hidden, "Inkling audio reduced elements")?;
            let graph_scalars = checked_add(
                checked_mul(4, embedded, "Inkling audio embedding workspace")?,
                checked_mul(12, reduced, "Inkling audio reduction/norm workspace")?,
                "Inkling audio graph workspace",
            )?;
            Ok(MediaShapePlan {
                capture_media: None,
                decoder_positions: frames,
                execution_workspace_scalars: graph_scalars,
            })
        }
        InputModality::Video => Err(unsupported(
            &args.model_type,
            "video is not a supported Inkling modality",
        )),
        InputModality::Text => Err(unsupported(
            &args.model_type,
            "text is not an Inkling media modality",
        )),
        _ => Err(unsupported(
            &args.model_type,
            "unknown modality is not supported by Inkling",
        )),
    }
}

/// Validates one prepared Inkling part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn inkling_input_part<Tensor>(
    args: &crate::inkling::ModelArgs,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<InklingInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(InklingInputPartPlan::TextTokens {
            positions: batch_one_sequence(&shape, 2, "Inkling text token IDs", &args.model_type)?,
        }),
        (
            modality @ (InputModality::Image | InputModality::Audio),
            InputPayloadKind::Embeddings,
        ) => {
            let positions = batch_one_sequence(
                &shape,
                3,
                &format!("Inkling {} embeddings", modality.as_str()),
                &args.model_type,
            )?;
            let hidden = positive(args.text_config.hidden_size, "Inkling text hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Inkling {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            let placeholder_token_id = match modality {
                InputModality::Image => args.image_token_id,
                InputModality::Audio => args.audio_token_id,
                InputModality::Text | InputModality::Video => unreachable!(),
                _ => {
                    return Err(unsupported(
                        &args.model_type,
                        "unknown projected Inkling modality",
                    ));
                }
            };
            Ok(InklingInputPartPlan::Projected {
                modality,
                placeholder_token_id,
                positions,
            })
        }
        (modality @ (InputModality::Image | InputModality::Audio), InputPayloadKind::Tensor) => {
            let shape = inkling(args, &inspected)?;
            let ingress = InklingIngressPlan {
                placeholder_token_id: if modality == InputModality::Image {
                    args.image_token_id
                } else {
                    args.audio_token_id
                },
                placeholder_count: shape.decoder_positions,
            };
            Ok(InklingInputPartPlan::Media {
                modality,
                ingress,
                shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Inkling does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

/// Derives prepared Muse vision geometry and artifact modality policy.
fn muse_vision(
    args: &crate::muse::DecoderConfig,
    input: &MediaAdmissionInput,
) -> Result<MediaShapePlan, CapabilityError> {
    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| CapabilityError::MissingCompanion {
            architecture: args.identity.text_model_type().into(),
            modality: input.modality(),
            component: "vision encoder and projector",
        })?;
    if input.modality() == InputModality::Audio
        || (input.modality() == InputModality::Video
            && vision.input_policy == crate::muse::VisionInputPolicy::Images)
    {
        return Err(unsupported(
            args.identity.text_model_type(),
            format!(
                "loaded Muse-Glimmer artifact does not support {}",
                input.modality().as_str()
            ),
        ));
    }
    let grid = input.patch_grid.as_ref().ok_or_else(|| {
        unsupported(
            args.identity.text_model_type(),
            "Muse-Glimmer media requires patch_grid metadata",
        )
    })?;
    if grid.shape.len() != 2 || grid.shape[0] == 0 || grid.shape[1] != 3 {
        return Err(unsupported(
            args.identity.text_model_type(),
            format!(
                "Muse-Glimmer patch_grid must be [items, 3], got {:?}",
                grid.shape
            ),
        ));
    }
    let expected_values = checked_mul(grid.shape[0], 3, "Muse patch-grid scalar count")?;
    if u64::try_from(grid.values.len()).ok() != Some(expected_values) {
        return Err(unsupported(
            args.identity.text_model_type(),
            "Muse-Glimmer patch_grid has an incomplete row",
        ));
    }
    let merge = nonzero_positive(vision.merge_size, "Muse vision merge size")?;
    let mut patches = 0u64;
    let mut positions = 0u64;
    for entry in grid.values.as_chunks::<3>().0 {
        if entry.iter().any(|value| *value <= 0)
            || u64::try_from(entry[1]).unwrap_or_default() % merge != 0
            || u64::try_from(entry[2]).unwrap_or_default() % merge != 0
        {
            return Err(unsupported(
                args.identity.text_model_type(),
                "Muse-Glimmer vision grids must be positive and merge-divisible",
            ));
        }
        let t = entry[0] as u64;
        let h = entry[1] as u64;
        let w = entry[2] as u64;
        patches = checked_add(
            patches,
            checked_mul(
                checked_mul(t, h, "Muse vision t*h")?,
                w,
                "Muse vision patches",
            )?,
            "Muse vision patch total",
        )?;
        positions = checked_add(
            positions,
            checked_mul(
                checked_mul(t, h / merge, "Muse merged t*h")?,
                w / merge,
                "Muse merged positions",
            )?,
            "Muse merged position total",
        )?;
    }
    if input.payload_shape.first().copied() != Some(patches) {
        return Err(unsupported(
            args.identity.text_model_type(),
            format!(
                "Muse-Glimmer payload has {} patches but metadata describes {patches}",
                input.payload_shape.first().copied().unwrap_or_default()
            ),
        ));
    }
    let graph = checked_mul(
        patches,
        positive(vision.hidden_size, "Muse vision hidden size")?,
        "Muse vision activation scalars",
    )?;
    Ok(MediaShapePlan {
        capture_media: None,
        decoder_positions: positions,
        execution_workspace_scalars: checked_mul(graph, 8, "Muse vision graph multiplier")?,
    })
}

/// Validates one prepared Muse-Glimmer part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn muse_input_part<Tensor>(
    args: &crate::muse::DecoderConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<MuseInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(MuseInputPartPlan::TextTokens {
            positions: batch_one_sequence(
                &shape,
                2,
                "Muse-Glimmer text token IDs",
                args.identity.text_model_type(),
            )?,
        }),
        (modality @ (InputModality::Image | InputModality::Video), InputPayloadKind::Tensor) => {
            let shape = muse_vision(args, &inspected)?;
            let placeholder_token_id = if modality == InputModality::Image {
                args.image_token_id
            } else {
                args.video_token_id
            };
            let patch_grid = inspected
                .patch_grid
                .as_ref()
                .expect("Muse-Glimmer shape validation requires a patch grid")
                .values
                .as_chunks::<3>()
                .0
                .iter()
                .map(|entry| (entry[0], entry[1], entry[2]))
                .collect();
            Ok(MuseInputPartPlan::Vision {
                modality,
                ingress: MuseVisionIngressPlan {
                    placeholder_token_id,
                    placeholder_count: shape.decoder_positions,
                    patch_grid,
                },
                shape,
            })
        }
        (InputModality::Audio, InputPayloadKind::Tensor)
            if args.identity == crate::muse::MuseIdentity::Spark && args.audio_config.is_none() =>
        {
            Err(CapabilityError::MissingCompanion {
                architecture: args.identity.text_model_type().into(),
                modality: InputModality::Audio,
                component: "waveform encoder and perception projection",
            })
        }
        (InputModality::Audio, InputPayloadKind::Tensor) if args.audio_config.is_some() => {
            let audio = args.audio_config.as_ref().expect("admitted audio geometry");
            let samples = batch_one_sequence(
                &shape,
                2,
                "Muse mono waveform",
                args.identity.text_model_type(),
            )?;
            let cadence = nonzero_positive(
                audio.samples_per_token().map_err(|error| {
                    unsupported(args.identity.text_model_type(), error.to_string())
                })?,
                "Muse waveform cadence",
            )?;
            let valid = descriptor
                .extents()
                .find_map(|extent| match extent {
                    InputExtent::AudioValidSamples(samples) => u64::try_from(samples).ok(),
                    _ => None,
                })
                .ok_or_else(|| {
                    unsupported(
                        args.identity.text_model_type(),
                        "Muse waveform requires valid-sample extent",
                    )
                })?;
            if samples == 0
                || samples % cadence != 0
                || valid == 0
                || valid > samples
                || valid.div_ceil(cadence) != samples / cadence
                || descriptor.extents().count() != 1
                || !descriptor.metadata().is_empty()
                || !matches!(
                    descriptor.payload().dtype(),
                    eredu_core::checkpoint::TensorDtype::F32
                        | eredu_core::checkpoint::TensorDtype::F16
                        | eredu_core::checkpoint::TensorDtype::Bf16
                )
            {
                return Err(unsupported(args.identity.text_model_type(), "Muse waveform must be floating mono PCM padded only to its final token cadence"));
            }
            let positions = samples / cadence;
            let mut frames = samples;
            let mut workspace = samples;
            for (&patch, &width) in audio.patch_sizes.iter().zip(&audio.hidden_sizes) {
                frames /= nonzero_positive(patch, "Muse audio patch")?;
                workspace = workspace.max(checked_mul(
                    checked_mul(
                        frames,
                        positive(width, "Muse audio width")?,
                        "Muse audio frame scalars",
                    )?,
                    8,
                    "Muse audio workspace",
                )?);
            }
            Ok(MuseInputPartPlan::Audio {
                ingress: MuseAudioIngressPlan {
                    valid_samples: valid,
                    padded_samples: samples,
                    placeholder_token_id: args
                        .audio_token_id
                        .expect("validated audio token identity"),
                    placeholder_count: positions,
                },
                shape: MediaShapePlan {
                    decoder_positions: positions,
                    execution_workspace_scalars: workspace,
                    capture_media: None,
                },
            })
        }
        (modality, payload) => Err(unsupported(
            args.identity.text_model_type(),
            format!(
                "Muse-Glimmer does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

/// Validates one prepared input part for a text-only architecture.
pub fn text_only_input_part<Tensor>(
    architecture: &str,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<PreparedInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(PreparedInputPartPlan::Text {
            positions: batch_one_sequence(&shape, 2, "text token IDs", architecture)?,
        }),
        (modality, payload) => Err(unsupported(
            architecture,
            format!(
                "text-only architecture does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}
