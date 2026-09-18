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

pub(crate) mod admission;
mod original;
pub(crate) mod qwen;
pub use original::{
    validate_selected_part, BoundPreparedMediaSemantics, OriginalPreparedMediaSemantics,
    OriginalPromptSegment, PreparedMediaEncoderTablePlan, PreparedMediaPositionFacts,
    PreparedMediaSemanticCompile,
};
pub use qwen::MediaSemanticError;

use crate::qwen::{
    hybrid::{HybridConfig, ParsedHybridConfig},
    vision::VisionConfig,
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
    fn qwen_ref(&self) -> qwen::InspectedPartRef<'_> {
        qwen::InspectedPartRef {
            modality: self.descriptor.modality(),
            kind: self.descriptor.payload_kind(),
            shape: qwen::ShapeRef::Legacy(&self.payload_shape),
            grid: self.patch_grid.as_ref().map(|grid| qwen::GridRef {
                shape: qwen::ShapeRef::Legacy(&grid.shape),
                values: &grid.values,
            }),
        }
    }

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
    admission::inspect_shared(
        input,
        |part| part.descriptor(&|tensor| inspector.identity(tensor))
            .map_err(|error| CapabilityError::Observation(error.to_string())),
        u64_shape,
        |tensor| inspector.i32_values(tensor),
        |tensor| inspector.bool_values(tensor),
    )
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

/// Architecture-derived geometry and conservative execution workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaShapePlan {
    /// Decoder positions occupied by the projected media.
    pub decoder_positions: u64,
    /// Conservative count of temporary execution scalars.
    pub execution_workspace_scalars: u64,
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

/// Qwen3.5/Qwen3-Next admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenHybridInputPartPlan {
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

impl From<QwenHybridInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: QwenHybridInputPartPlan) -> Self {
        match plan {
            QwenHybridInputPartPlan::TextTokens { positions } => Self::Text { positions },
            QwenHybridInputPartPlan::Projected {
                modality,
                positions,
            } => Self::Projected {
                modality,
                positions,
            },
            QwenHybridInputPartPlan::Media { shape, .. } => Self::Media { shape },
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
pub enum MuseGlimmerInputPartPlan {
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
        ingress: MuseGlimmerIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<MuseGlimmerInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: MuseGlimmerInputPartPlan) -> Self {
        match plan {
            MuseGlimmerInputPartPlan::TextTokens { positions } => Self::Text { positions },
            MuseGlimmerInputPartPlan::Vision { shape, .. } => Self::Media { shape },
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
pub struct MuseGlimmerIngressPlan {
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

impl<P: CompositePartPlan> AdmittedCompositeInput<P> {
    /// Visits exact ordered decoder extents from the already admitted family plans.
    /// Token attribution remains tied to the actual payload descriptor: projected
    /// text and media placeholders never become canonical tokenizer IDs.
    pub fn visit_prompt_segments(
        &self,
        mut visit: impl FnMut(eredu_core::PreparedPromptSegmentPlan),
    ) -> Result<(), eredu_core::PreparedControlInputError> {
        use eredu_core::PreparedControlInputError as E;
        if self.parts.len() != self.identity.parts().len() {
            return Err(E::InvalidAttribution);
        }
        let mut position = 0u64;
        for (index, (part, descriptor)) in self.parts.iter().zip(self.identity.parts()).enumerate()
        {
            let extent = part.decoder_positions();
            let end = position.checked_add(extent).ok_or(E::Overflow)?;
            visit(eredu_core::PreparedPromptSegmentPlan {
                source_part: u64::try_from(index).map_err(|_| E::Overflow)?,
                modality: descriptor.modality(),
                payload: descriptor.payload_kind(),
                decoder_range: [position, end],
            });
            position = end;
        }
        if position != self.decoder_positions {
            return Err(E::InvalidAttribution);
        }
        Ok(())
    }
}

/// Borrowed family-owned decoder extent of one already admitted part.
/// This geometry declaration grants no source, memory or execution authority.
pub trait CompositePartPlan {
    /// Exact decoder positions; no plan/grid clone or tensor inspection occurs.
    fn decoder_positions(&self) -> u64;
}

impl CompositePartPlan for PreparedInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::Text { positions } | Self::Projected { positions, .. } => *positions,
            Self::Media { shape } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for QwenVlInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::ProjectedText { positions } => *positions,
            Self::Media { shape, .. } => shape.decoder_positions,
        }
    }
}

impl CompositePartPlan for QwenHybridInputPartPlan {
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

impl CompositePartPlan for MuseGlimmerInputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } => *positions,
            Self::Vision { shape, .. } => shape.decoder_positions,
        }
    }
}

fn admit_composite_input<Tensor, P>(
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
    admission::complete_shared(
        input, identity, &mut admit_part,
        |count| Ok(Vec::with_capacity(count)),
        admission::Rejection::ordinary,
    )
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
) -> Result<AdmittedCompositeInput<QwenHybridInputPartPlan>, CapabilityError> {
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
pub fn admit_muse_glimmer_input<Tensor>(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &PreparedModelInput<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<AdmittedCompositeInput<MuseGlimmerInputPartPlan>, CapabilityError> {
    admit_composite_input(input, inspector, |part| {
        muse_glimmer_input_part(args, part, inspector)
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

/// Legacy inspection adapter for the same borrowed Qwen equation kernel.
fn qwen_vision(
    config: &VisionConfig,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    qwen::qwen_vision(config, input.qwen_ref()).map_err(|error| error.legacy(architecture))
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
    let view = qwen::qwen_part(
        qwen::QwenPolicy {
            hidden: 0,
            vision: config,
            image: image_token_id,
            video: video_token_id,
            projected_media: false,
        },
        input.qwen_ref(),
    )
    .map_err(|error| error.legacy(architecture))?;
    Ok((
        QwenVisionIngressPlan {
            placeholder_token_id: view.placeholder,
            placeholder_count: view.positions,
            patch_grid: view
                .grid
                .iter()
                .map(|row| (row[0], row[1], row[2]))
                .collect(),
        },
        MediaShapePlan {
            decoder_positions: view.positions,
            execution_workspace_scalars: view.workspace_scalars,
        },
    ))
}

#[cfg(test)]
fn qwen_vision_ingress(
    config: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &MediaAdmissionInput,
    architecture: &str,
) -> Result<QwenVisionIngressPlan, CapabilityError> {
    qwen_vision_ingress_with_shape(config, image_token_id, video_token_id, input, architecture)
        .map(|(ingress, _)| ingress)
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
    let view = qwen::qwen_part(
        admission::vl_policy(args),
        inspected.qwen_ref(),
    )
    .map_err(|error| error.legacy(&args.model_type))?;
    admission::vl_plan(view, |rows| {
        Ok(rows.iter().map(|row| (row[0], row[1], row[2])).collect())
    })
}

fn qwen_hybrid_input_part_with_policy<Tensor>(
    text: &HybridConfig,
    vision: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let view = qwen::qwen_part(
        admission::hybrid_policy(text, vision, image_token_id, video_token_id),
        inspected.qwen_ref(),
    )
    .map_err(|error| error.legacy(&text.model_type))?;
    admission::hybrid_plan(view, |rows| {
        Ok(rows.iter().map(|row| (row[0], row[1], row[2])).collect())
    })
}

/// Validates one prepared conditional Qwen3.5 input part and derives the exact
/// prefill and capability-accounting plan consumed by a concrete backend.
pub fn qwen_hybrid_input_part<Tensor>(
    args: &ParsedHybridConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    qwen_hybrid_input_part_with_policy(
        &args.text,
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
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => {
            Ok(QwenHybridInputPartPlan::TextTokens {
                positions: batch_one_sequence(&shape, 2, "text token IDs", &args.model_type)?,
            })
        }
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

/// Validates one prepared Gemma 4 part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn gemma4_input_part<Tensor>(
    args: &crate::gemma4::FamilyConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<Gemma4InputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    admission::gemma::raw::part(args, &inspected, admission::inkling::Ordinary)
}

#[cfg(test)]
fn gemma_valid_patch_count(positions: &MetadataValues<i32>, architecture: &str) -> Result<u64, CapabilityError> {
    admission::gemma::raw::gemma_valid_patch_count(positions, architecture, admission::inkling::Ordinary)
}
#[cfg(test)]
fn gemma_vision(config: &crate::gemma4::VisionConfig, text_hidden: u64, input: &MediaAdmissionInput, architecture: &str) -> Result<MediaShapePlan, CapabilityError> {
    admission::gemma::raw::gemma_vision(config, text_hidden, input, architecture, admission::inkling::Ordinary)
}
#[cfg(test)]
fn gemma_audio(config: &crate::gemma4::AudioConfig, text_hidden: u64, input: &MediaAdmissionInput, architecture: &str) -> Result<MediaShapePlan, CapabilityError> {
    admission::gemma::raw::gemma_audio(config, text_hidden, input, architecture, admission::inkling::Ordinary)
}
/// Derives prepared Gemma 4 media geometry from normalized family policy.
fn gemma4(args: &crate::gemma4::FamilyConfig, input: &MediaAdmissionInput) -> Result<MediaShapePlan, CapabilityError> {
    admission::gemma::raw::gemma4(args, input, admission::inkling::Ordinary)
}

/// Derives prepared Inkling media geometry from normalized family policy.
fn inkling(
    args: &crate::inkling::ModelArgs,
    input: &MediaAdmissionInput,
) -> Result<MediaShapePlan, CapabilityError> {
    admission::inkling::media(args, input, admission::inkling::Ordinary)
}

/// Validates one prepared Inkling part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn inkling_input_part<Tensor>(
    args: &crate::inkling::ModelArgs,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<InklingInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    admission::inkling::part(args, &inspected, admission::inkling::Ordinary)
}

/// Derives prepared Muse-Glimmer media geometry and artifact modality policy.
fn muse_glimmer(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &MediaAdmissionInput,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.modality() == InputModality::Audio
        || (input.modality() == InputModality::Video
            && args.weight_convention == crate::muse_glimmer::WeightConvention::Gguf)
    {
        return Err(unsupported(
            &args.model_type,
            format!(
                "loaded Muse-Glimmer artifact does not support {}",
                input.modality().as_str()
            ),
        ));
    }
    let vision = args.vision_config.as_ref().ok_or_else(|| {
        unsupported(
            &args.model_type,
            "loaded Muse-Glimmer artifact has no vision projector",
        )
    })?;
    let grid = input.patch_grid.as_ref().ok_or_else(|| {
        unsupported(
            &args.model_type,
            "Muse-Glimmer media requires patch_grid metadata",
        )
    })?;
    if grid.shape.len() != 2 || grid.shape[0] == 0 || grid.shape[1] != 3 {
        return Err(unsupported(
            &args.model_type,
            format!(
                "Muse-Glimmer patch_grid must be [items, 3], got {:?}",
                grid.shape
            ),
        ));
    }
    let expected_values = checked_mul(grid.shape[0], 3, "Muse patch-grid scalar count")?;
    if u64::try_from(grid.values.len()).ok() != Some(expected_values) {
        return Err(unsupported(
            &args.model_type,
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
                &args.model_type,
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
            &args.model_type,
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
        decoder_positions: positions,
        execution_workspace_scalars: checked_mul(graph, 8, "Muse vision graph multiplier")?,
    })
}

/// Validates one prepared Muse-Glimmer part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn muse_glimmer_input_part<Tensor>(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<MuseGlimmerInputPartPlan, CapabilityError> {
    let inspected = inspect_part(input, inspector)?;
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = u64_shape(descriptor.payload())?;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => {
            Ok(MuseGlimmerInputPartPlan::TextTokens {
                positions: batch_one_sequence(
                    &shape,
                    2,
                    "Muse-Glimmer text token IDs",
                    &args.model_type,
                )?,
            })
        }
        (modality @ (InputModality::Image | InputModality::Video), InputPayloadKind::Tensor) => {
            let shape = muse_glimmer(args, &inspected)?;
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
            Ok(MuseGlimmerInputPartPlan::Vision {
                modality,
                ingress: MuseGlimmerIngressPlan {
                    placeholder_token_id,
                    placeholder_count: shape.decoder_positions,
                    patch_grid,
                },
                shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::vision::VisionAttentionPolicy;
    use eredu_core::{checkpoint::TensorDtype, PreparedInputError};
    use eredu_runtime::PreparedInputPayload as RuntimePayload;
    use serde_json::json;

    type MediaMetadata<T> = MetadataValues<T>;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestPayload {
        TokenIds(Vec<u64>),
        Embeddings(Vec<u64>),
        Tensor {
            shape: Vec<u64>,
            media: Option<MediaAdmissionInput>,
        },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestInputPart {
        modality: InputModality,
        payload: TestPayload,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestTensorValues {
        None,
        I32(Vec<i32>),
        Bool(Vec<bool>),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestTensor {
        shape: Vec<usize>,
        dtype: TensorDtype,
        values: TestTensorValues,
    }

    struct TestInspector;

    impl PreparedInputInspector<TestTensor> for TestInspector {
        fn identity(&self, tensor: &TestTensor) -> Result<InputTensorIdentity, PreparedInputError> {
            InputTensorIdentity::new(tensor.dtype.clone(), tensor.shape.clone())
        }

        fn i32_values(&self, tensor: &TestTensor) -> Result<Vec<i32>, CapabilityError> {
            match &tensor.values {
                TestTensorValues::I32(values) => Ok(values.clone()),
                _ => Err(CapabilityError::Observation(
                    "test tensor does not contain i32 values".into(),
                )),
            }
        }

        fn bool_values(&self, tensor: &TestTensor) -> Result<Vec<bool>, CapabilityError> {
            match &tensor.values {
                TestTensorValues::Bool(values) => Ok(values.clone()),
                _ => Err(CapabilityError::Observation(
                    "test tensor does not contain bool values".into(),
                )),
            }
        }
    }

    struct ShapeChangingInspector;

    impl PreparedInputInspector<TestTensor> for ShapeChangingInspector {
        fn identity(&self, tensor: &TestTensor) -> Result<InputTensorIdentity, PreparedInputError> {
            let mut shape = tensor.shape.clone();
            *shape
                .last_mut()
                .expect("test tensors have non-empty shapes") += 1;
            InputTensorIdentity::new(tensor.dtype.clone(), shape)
        }

        fn i32_values(&self, tensor: &TestTensor) -> Result<Vec<i32>, CapabilityError> {
            TestInspector.i32_values(tensor)
        }

        fn bool_values(&self, tensor: &TestTensor) -> Result<Vec<bool>, CapabilityError> {
            TestInspector.bool_values(tensor)
        }
    }

    fn usize_shape(shape: &[u64]) -> Vec<usize> {
        shape
            .iter()
            .map(|value| usize::try_from(*value).unwrap())
            .collect()
    }

    fn tensor(shape: &[u64], dtype: TensorDtype, values: TestTensorValues) -> TestTensor {
        TestTensor {
            shape: usize_shape(shape),
            dtype,
            values,
        }
    }

    fn runtime_part(input: &TestInputPart) -> eredu_runtime::PreparedInputPart<TestTensor> {
        let (payload, media) = match &input.payload {
            TestPayload::TokenIds(shape) => (
                RuntimePayload::TokenIds(tensor(shape, TensorDtype::U32, TestTensorValues::None)),
                None,
            ),
            TestPayload::Embeddings(shape) => (
                RuntimePayload::Embeddings(tensor(shape, TensorDtype::F32, TestTensorValues::None)),
                None,
            ),
            TestPayload::Tensor { shape, media } => (
                RuntimePayload::Tensor(tensor(shape, TensorDtype::F32, TestTensorValues::None)),
                media.as_ref(),
            ),
        };
        let metadata = media
            .into_iter()
            .flat_map(|media| {
                [
                    media.patch_grid.as_ref().map(|metadata| {
                        (
                            InputMetadataKey::PatchGrid,
                            tensor(
                                &metadata.shape,
                                TensorDtype::I32,
                                TestTensorValues::I32(metadata.values.clone()),
                            ),
                        )
                    }),
                    media.patch_positions.as_ref().map(|metadata| {
                        (
                            InputMetadataKey::PatchPositions,
                            tensor(
                                &metadata.shape,
                                TensorDtype::I32,
                                TestTensorValues::I32(metadata.values.clone()),
                            ),
                        )
                    }),
                    media.audio_mask.as_ref().map(|metadata| {
                        (
                            InputMetadataKey::AudioMask,
                            tensor(
                                &metadata.shape,
                                TensorDtype::Bool,
                                TestTensorValues::Bool(metadata.values.clone()),
                            ),
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            })
            .collect::<Vec<_>>();
        let extents = media
            .into_iter()
            .flat_map(|media| media.descriptor.extents())
            .collect::<Vec<_>>();
        eredu_runtime::PreparedInputPart::new_with_extents(
            input.modality,
            payload,
            metadata,
            extents,
        )
        .unwrap()
    }

    fn prepared_input(
        parts: impl IntoIterator<Item = TestInputPart>,
    ) -> PreparedModelInput<TestTensor> {
        PreparedModelInput::new(
            parts.into_iter().map(|part| runtime_part(&part)).collect(),
            |tensor| TestInspector.identity(tensor),
        )
        .unwrap()
    }

    macro_rules! input_part_adapter {
        ($name:ident, $args:ty, $result:ty) => {
            fn $name(args: &$args, input: &TestInputPart) -> Result<$result, CapabilityError> {
                super::$name(args, &runtime_part(input), &TestInspector)
            }
        };
    }

    input_part_adapter!(qwen_vl_input_part, QwenVlModelArgs, QwenVlInputPartPlan);
    input_part_adapter!(
        qwen_hybrid_input_part,
        ParsedHybridConfig,
        QwenHybridInputPartPlan
    );
    input_part_adapter!(
        qwen_hybrid_text_input_part,
        HybridConfig,
        QwenHybridInputPartPlan
    );
    input_part_adapter!(
        gemma4_input_part,
        crate::gemma4::FamilyConfig,
        Gemma4InputPartPlan
    );
    input_part_adapter!(
        inkling_input_part,
        crate::inkling::ModelArgs,
        InklingInputPartPlan
    );
    input_part_adapter!(
        muse_glimmer_input_part,
        crate::muse_glimmer::DecoderConfig,
        MuseGlimmerInputPartPlan
    );

    fn text_only_input_part(
        architecture: &str,
        input: &TestInputPart,
    ) -> Result<PreparedInputPartPlan, CapabilityError> {
        super::text_only_input_part(architecture, &runtime_part(input), &TestInspector)
    }

    fn input(modality: InputModality, payload_shape: &[u64]) -> MediaAdmissionInput {
        let payload =
            InputTensorIdentity::new(TensorDtype::F32, usize_shape(payload_shape)).unwrap();
        MediaAdmissionInput {
            descriptor: InputPartDescriptor::new(modality, InputPayloadKind::Tensor, payload, [])
                .unwrap(),
            payload_shape: payload_shape.to_vec(),
            patch_grid: None,
            patch_positions: None,
            audio_mask: None,
        }
    }

    fn set_extents(
        input: &mut MediaAdmissionInput,
        extents: impl IntoIterator<Item = InputExtent>,
    ) {
        input.descriptor = InputPartDescriptor::new_with_extents(
            input.descriptor.modality(),
            input.descriptor.payload_kind(),
            input.descriptor.payload().clone(),
            input
                .descriptor
                .metadata()
                .iter()
                .map(|(key, identity)| (*key, identity.clone())),
            extents,
        )
        .unwrap();
    }

    fn set_modality(input: &mut MediaAdmissionInput, modality: InputModality) {
        input.descriptor = InputPartDescriptor::new_with_extents(
            modality,
            input.descriptor.payload_kind(),
            input.descriptor.payload().clone(),
            input
                .descriptor
                .metadata()
                .iter()
                .map(|(key, identity)| (*key, identity.clone())),
            input.descriptor.extents(),
        )
        .unwrap();
    }

    pub(super) fn qwen_vl_args() -> QwenVlModelArgs {
        crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap()
    }

    pub(super) fn qwen_hybrid_args() -> ParsedHybridConfig {
        crate::qwen::hybrid::model_args_from_config_value(&json!({
            "model_type":"qwen3_5","image_token_id":30,"video_token_id":31,
            "text_config":{
                "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":16,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
                "linear_conv_kernel_dim":2,"linear_key_head_dim":4,
                "linear_value_head_dim":4,"linear_num_key_heads":2,
                "linear_num_value_heads":4,
                "layer_types":["linear_attention","full_attention"],
                "tie_word_embeddings":false
            },
            "vision_config":{
                "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
                "num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16
            }
        }))
        .unwrap()
    }

    fn gemma_family() -> crate::gemma4::FamilyConfig {
        crate::gemma4::FamilyConfig::from_hf_json(
            &serde_json::to_vec(&json!({
                "model_type":"gemma4_unified",
                "tie_word_embeddings":false,
                "image_token_id":60,
                "audio_token_id":61,
                "text_config":{
                    "model_type":"gemma4_text","hidden_size":16,"num_hidden_layers":2,
                    "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                    "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":64,
                    "max_position_embeddings":128,"layer_types":["full_attention","full_attention"]
                },
                "vision_config":{
                    "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                    "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
                    "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                    "rms_norm_eps":0.000001
                },
                "audio_config":{
                    "hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                    "output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,
                    "attention_context_left":5,"attention_context_right":0,
                    "attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,
                    "residual_weight":0.5,"rms_norm_eps":0.000001,
                    "subsampling_conv_channels":[4,8]
                }
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn qwen_vl_admission_accepts_only_text_projected_embeddings() {
        let args = qwen_vl_args();
        let projected = TestInputPart {
            modality: InputModality::Text,
            payload: TestPayload::Embeddings(vec![1, 4, 32]),
        };
        assert_eq!(
            qwen_vl_input_part(&args, &projected).unwrap(),
            QwenVlInputPartPlan::ProjectedText { positions: 4 }
        );

        let non_text = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Embeddings(vec![1, 4, 32]),
        };
        assert!(matches!(
            qwen_vl_input_part(&args, &non_text),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn qwen_hybrid_admission_accepts_projected_text_image_and_video() {
        let args = qwen_hybrid_args();
        for modality in [
            InputModality::Text,
            InputModality::Image,
            InputModality::Video,
        ] {
            let projected = TestInputPart {
                modality,
                payload: TestPayload::Embeddings(vec![1, 4, 16]),
            };
            assert_eq!(
                qwen_hybrid_input_part(&args, &projected).unwrap(),
                QwenHybridInputPartPlan::Projected {
                    modality,
                    positions: 4,
                }
            );
        }

        let wrong_width = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Embeddings(vec![1, 4, 8]),
        };
        assert!(matches!(
            qwen_hybrid_input_part(&args, &wrong_width),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let audio = TestInputPart {
            modality: InputModality::Audio,
            payload: TestPayload::Embeddings(vec![1, 4, 16]),
        };
        assert!(matches!(
            qwen_hybrid_input_part(&args, &audio),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let projected_text = TestInputPart {
            modality: InputModality::Text,
            payload: TestPayload::Embeddings(vec![1, 4, 16]),
        };
        assert!(matches!(
            qwen_hybrid_text_input_part(&args.text, &projected_text),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn whole_input_admission_preserves_identity_order_and_total_geometry() {
        let args = qwen_hybrid_args();
        let input = prepared_input([
            TestInputPart {
                modality: InputModality::Text,
                payload: TestPayload::TokenIds(vec![1, 3]),
            },
            TestInputPart {
                modality: InputModality::Image,
                payload: TestPayload::Embeddings(vec![1, 4, 16]),
            },
            TestInputPart {
                modality: InputModality::Video,
                payload: TestPayload::Embeddings(vec![1, 2, 16]),
            },
        ]);

        let admitted = admit_qwen_hybrid_input(&args, &input, &TestInspector).unwrap();

        assert_eq!(admitted.identity(), input.identity());
        assert_eq!(admitted.decoder_positions(), 9);
        assert_eq!(
            admitted.active_modalities(),
            InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            }
        );
        assert_eq!(
            admitted.parts(),
            [
                QwenHybridInputPartPlan::TextTokens { positions: 3 },
                QwenHybridInputPartPlan::Projected {
                    modality: InputModality::Image,
                    positions: 4,
                },
                QwenHybridInputPartPlan::Projected {
                    modality: InputModality::Video,
                    positions: 2,
                },
            ]
        );
    }

    #[test]
    fn whole_input_admission_rejects_changed_tensor_identity() {
        let args = qwen_hybrid_args();
        let input = prepared_input([TestInputPart {
            modality: InputModality::Text,
            payload: TestPayload::TokenIds(vec![1, 3]),
        }]);

        assert!(matches!(
            admit_qwen_hybrid_input(&args, &input, &ShapeChangingInspector),
            Err(CapabilityError::Observation(_))
        ));
    }

    #[test]
    fn whole_input_admission_rejects_an_invalid_later_part() {
        let args = qwen_hybrid_args();
        let input = prepared_input([
            TestInputPart {
                modality: InputModality::Text,
                payload: TestPayload::TokenIds(vec![1, 3]),
            },
            TestInputPart {
                modality: InputModality::Audio,
                payload: TestPayload::Embeddings(vec![1, 2, 16]),
            },
        ]);

        assert!(matches!(
            admit_qwen_hybrid_input(&args, &input, &TestInspector),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn text_only_input_plan_rejects_every_non_token_payload() {
        let tokens = TestInputPart {
            modality: InputModality::Text,
            payload: TestPayload::TokenIds(vec![1, 4]),
        };
        assert_eq!(
            text_only_input_part("llama", &tokens).unwrap(),
            PreparedInputPartPlan::Text { positions: 4 }
        );

        for input in [
            TestInputPart {
                modality: InputModality::Text,
                payload: TestPayload::Embeddings(vec![1, 4, 16]),
            },
            TestInputPart {
                modality: InputModality::Image,
                payload: TestPayload::Embeddings(vec![1, 4, 16]),
            },
        ] {
            assert!(matches!(
                text_only_input_part("llama", &input),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
        }
    }

    #[test]
    fn gemma_patch_positions_ignore_padding() {
        let positions = MediaMetadata {
            shape: vec![1, 5, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1, -1, -1],
        };
        assert_eq!(gemma_valid_patch_count(&positions, "gemma4").unwrap(), 4);
    }

    #[test]
    fn gemma_input_plan_owns_payload_placeholder_and_ingress_policy() {
        let args = gemma_family();
        assert_eq!(
            crate::capability::gemma4(&args)
                .unwrap()
                .speculative_draft_source(),
            Some(eredu_core::SpeculativeDraftSource::Separate)
        );

        let projected = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Embeddings(vec![1, 3, 16]),
        };
        assert_eq!(
            gemma4_input_part(&args, &projected).unwrap(),
            Gemma4InputPartPlan::Projected {
                modality: InputModality::Image,
                placeholder_token_id: 60,
                positions: 3,
            }
        );

        let mut vision = input(InputModality::Image, &[1, 4, 48]);
        vision.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        vision.patch_positions = Some(MediaMetadata {
            shape: vec![1, 4, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1],
        });
        set_extents(
            &mut vision,
            [InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        );
        let vision = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Tensor {
                shape: vision.payload_shape.clone(),
                media: Some(vision),
            },
        };
        assert!(matches!(
            gemma4_input_part(&args, &vision).unwrap(),
            Gemma4InputPartPlan::Vision {
                placeholder_token_id: 60,
                ingress: crate::gemma4::VisionIngressPartPlan {
                    decoder_positions: 1,
                    ..
                },
                ..
            }
        ));
        let mut inconsistent_vision = vision.clone();
        let TestPayload::Tensor {
            media: Some(media), ..
        } = &mut inconsistent_vision.payload
        else {
            unreachable!()
        };
        set_extents(
            media,
            [InputExtent::PatchGrid {
                time: 1,
                height: 4,
                width: 1,
            }],
        );
        assert!(matches!(
            gemma4_input_part(&args, &inconsistent_vision),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let mut audio = input(InputModality::Audio, &[1, 8, 128]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 8],
            values: vec![true, true, true, true, true, true, false, false],
        });
        set_extents(&mut audio, [InputExtent::AudioValidFrames(6)]);
        let audio = TestInputPart {
            modality: InputModality::Audio,
            payload: TestPayload::Tensor {
                shape: audio.payload_shape.clone(),
                media: Some(audio),
            },
        };
        assert!(matches!(
            gemma4_input_part(&args, &audio).unwrap(),
            Gemma4InputPartPlan::Audio {
                placeholder_token_id: 61,
                ingress: crate::gemma4::AudioIngressPartPlan {
                    decoder_positions: 2,
                    ..
                },
                ..
            }
        ));
        let mut inconsistent_audio = audio.clone();
        let TestPayload::Tensor {
            media: Some(media), ..
        } = &mut inconsistent_audio.payload
        else {
            unreachable!()
        };
        set_extents(media, [InputExtent::AudioValidFrames(5)]);
        assert!(matches!(
            gemma4_input_part(&args, &inconsistent_audio),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let video = TestInputPart {
            modality: InputModality::Video,
            payload: TestPayload::Embeddings(vec![1, 1, 16]),
        };
        assert!(matches!(
            gemma4_input_part(&args, &video),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn capability_estimate_derives_embedded_mtp_from_exact_configuration() {
        let mut args = qwen_hybrid_args();
        assert_eq!(
            crate::capability::qwen_hybrid(&args)
                .unwrap()
                .speculative_draft_source(),
            None
        );
        args.text.mtp_num_hidden_layers = 2;
        assert_eq!(
            crate::capability::qwen_hybrid(&args)
                .unwrap()
                .speculative_draft_source(),
            Some(eredu_core::SpeculativeDraftSource::Embedded)
        );
    }

    #[test]
    fn qwen_plan_owns_window_geometry_and_grid_validation() {
        let config = VisionConfig {
            mode: crate::qwen::vision::VisionMode::WindowScheduled,
            layer_schedule: eredu_core::attention::LayerSchedule::new(
                2,
                vec![
                    crate::qwen::vision::VisionLayerPolicy {
                        attention: VisionAttentionPolicy::Windowed,
                        deepstack_merger: Some(0),
                    },
                    crate::qwen::vision::VisionLayerPolicy {
                        attention: VisionAttentionPolicy::Full,
                        deepstack_merger: None,
                    },
                ],
            )
            .unwrap(),
            hidden_size: 8,
            hidden_act: "silu".into(),
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 4,
            in_channels: 3,
            patch_size: 1,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            window_size: 4,
            out_hidden_size: 8,
            linear_formats: Default::default(),
        };
        let mut prepared = input(InputModality::Image, &[4, 3]);
        prepared.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        let plan = qwen_vision(&config, &prepared, "qwen3_vl").unwrap();
        assert_eq!(plan.decoder_positions, 1);
        assert!(plan.execution_workspace_scalars > 4 * 8);

        let ingress =
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl").unwrap();
        assert_eq!(ingress.placeholder_token_id, 22);
        assert_eq!(ingress.placeholder_count, 1);
        assert_eq!(ingress.patch_grid, vec![(1, 2, 2)]);

        set_modality(&mut prepared, InputModality::Video);
        assert_eq!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",)
                .unwrap()
                .placeholder_token_id,
            23
        );

        prepared.patch_grid.as_mut().unwrap().values = vec![1, 2, 4];
        assert!(matches!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        prepared.patch_grid.as_mut().unwrap().values = vec![i32::MAX, i32::MAX, i32::MAX];
        assert!(matches!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",),
            Err(CapabilityError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn gemma_plan_owns_patch_pooling_and_audio_mask_geometry() {
        let vision_config = crate::gemma4::VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 1,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-5,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        let mut vision = input(InputModality::Image, &[1, 5, 3]);
        vision.patch_positions = Some(MediaMetadata {
            shape: vec![1, 5, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1, -1, -1],
        });
        let vision_plan = gemma_vision(&vision_config, 8, &vision, "gemma4").unwrap();
        assert_eq!(vision_plan.decoder_positions, 1);

        let audio_config = crate::gemma4::AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            output_proj_dims: 8,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 3,
            attention_context_right: 0,
            attention_invalid_logits_value: -1e9,
            attention_logit_cap: 50.0,
            residual_weight: 1.0,
            rms_norm_eps: 1e-5,
            subsampling_conv_channels: vec![4, 8],
            output_projection_bias: true,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        let mut audio = input(InputModality::Audio, &[1, 8, 128]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 8],
            values: vec![true, true, true, true, true, true, false, false],
        });
        let audio_plan = gemma_audio(&audio_config, 8, &audio, "gemma4").unwrap();
        assert_eq!(audio_plan.decoder_positions, 2);
    }

    fn tiny_inkling() -> crate::inkling::ModelArgs {
        crate::inkling::ModelArgs::from_hf_json(
            &serde_json::to_vec(&json!({
                "model_type":"inkling_mm_model",
                "text_config":{
                    "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                    "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                    "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                    "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                    "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                    "intermediate_size":24,"dense_intermediate_size":48,
                    "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                    "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                    "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                    "use_global_scale":true,"gate_activation":"sigmoid"
                },
                "audio_config":{"decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16},
                "vision_config":{"decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                    "n_channels":3,"n_layers":4}
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn inkling_plan_owns_hmlp_and_dmel_shapes() {
        let args = tiny_inkling();
        let image = input(InputModality::Image, &[1, 2, 40, 40, 3]);
        let image_plan = inkling(&args, &image).unwrap();
        assert_eq!(image_plan.decoder_positions, 1);
        let image_part = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Tensor {
                shape: image.payload_shape.clone(),
                media: Some(image),
            },
        };
        assert!(matches!(
            inkling_input_part(&args, &image_part).unwrap(),
            InklingInputPartPlan::Media {
                ingress: InklingIngressPlan {
                    placeholder_token_id,
                    placeholder_count: 1,
                },
                ..
            } if placeholder_token_id == args.image_token_id
        ));

        let mut audio = input(InputModality::Audio, &[1, 3, 80]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![true, true, false],
        });
        let audio_plan = inkling(&args, &audio).unwrap();
        assert_eq!(audio_plan.decoder_positions, 2);
        let audio_part = |media: MediaAdmissionInput| TestInputPart {
            modality: InputModality::Audio,
            payload: TestPayload::Tensor {
                shape: media.payload_shape.clone(),
                media: Some(media),
            },
        };
        assert!(matches!(
            inkling_input_part(&args, &audio_part(audio.clone())).unwrap(),
            InklingInputPartPlan::Media {
                ingress: InklingIngressPlan {
                    placeholder_token_id,
                    placeholder_count: 2,
                },
                ..
            } if placeholder_token_id == args.audio_token_id
        ));

        audio.audio_mask.as_mut().unwrap().values = vec![true, false, true];
        assert!(matches!(
            inkling_input_part(&args, &audio_part(audio)),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
        assert!(matches!(
            inkling(&args, &input(InputModality::Audio, &[3, 80])),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
        assert!(matches!(
            inkling(&args, &input(InputModality::Video, &[1, 2])),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn inkling_input_plan_matches_projected_execution_policy() {
        let args = tiny_inkling();
        let projected = TestInputPart {
            modality: InputModality::Image,
            payload: TestPayload::Embeddings(vec![1, 3, 32]),
        };
        assert_eq!(
            inkling_input_part(&args, &projected).unwrap(),
            InklingInputPartPlan::Projected {
                modality: InputModality::Image,
                placeholder_token_id: args.image_token_id,
                positions: 3,
            }
        );

        for modality in [InputModality::Text, InputModality::Video] {
            let rejected = TestInputPart {
                modality,
                payload: TestPayload::Embeddings(vec![1, 3, 32]),
            };
            assert!(matches!(
                inkling_input_part(&args, &rejected),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
        }
    }

    fn tiny_muse() -> crate::muse_glimmer::DecoderConfig {
        crate::muse_glimmer::DecoderConfig::from_hf_value(&json!({
          "architectures":["MuseGlimmerForConditionalGeneration"],
          "model_type":"muse_glimmer",
          "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
          "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":1,
            "intermediate_size":0,"moe_intermediate_size":12,"num_experts":4,"num_experts_per_tok":2,
            "norm_topk_prob":true,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
            "rope_theta":10000.0,"layer_types":["sliding_attention"],"layer_rope_theta":[10000.0],
            "sliding_window":8,"tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
            "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
          "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
            "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,"merge_size":2,
            "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        }))
        .unwrap()
    }

    #[test]
    fn muse_plan_owns_grid_geometry_and_artifact_video_policy() {
        let mut prepared = input(InputModality::Video, &[4, 12]);
        prepared.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        let mut args = tiny_muse();
        let plan = muse_glimmer(&args, &prepared).unwrap();
        assert_eq!(plan.decoder_positions, 1);
        let prepared_part = |media: MediaAdmissionInput| TestInputPart {
            modality: InputModality::Video,
            payload: TestPayload::Tensor {
                shape: media.payload_shape.clone(),
                media: Some(media),
            },
        };
        assert!(matches!(
            muse_glimmer_input_part(&args, &prepared_part(prepared.clone())).unwrap(),
            MuseGlimmerInputPartPlan::Vision {
                ingress: MuseGlimmerIngressPlan {
                    placeholder_token_id: 23,
                    placeholder_count: 1,
                    patch_grid,
                },
                ..
            } if patch_grid == vec![(1, 2, 2)]
        ));

        args.weight_convention = crate::muse_glimmer::WeightConvention::Gguf;
        assert!(matches!(
            muse_glimmer_input_part(&args, &prepared_part(prepared)),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn muse_input_plan_rejects_projected_and_audio_payloads() {
        let args = tiny_muse();
        for input in [
            TestInputPart {
                modality: InputModality::Image,
                payload: TestPayload::Embeddings(vec![1, 1, 16]),
            },
            TestInputPart {
                modality: InputModality::Audio,
                payload: TestPayload::Tensor {
                    shape: vec![1, 2, 3],
                    media: Some(input(InputModality::Audio, &[1, 2, 3])),
                },
            },
        ] {
            assert!(matches!(
                muse_glimmer_input_part(&args, &input),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
        }
    }

    #[test]
    fn admitted_prompt_segments_borrow_extent_without_clone_or_owned_conversion() {
        struct Trap(u64);
        impl Clone for Trap {
            fn clone(&self) -> Self {
                panic!("segment visitor cloned admitted part")
            }
        }
        impl From<Trap> for PreparedInputPartPlan {
            fn from(_: Trap) -> Self {
                panic!("segment visitor converted owned part")
            }
        }
        impl CompositePartPlan for Trap {
            fn decoder_positions(&self) -> u64 {
                self.0
            }
        }
        let input = prepared_input([
            TestInputPart {
                modality: InputModality::Text,
                payload: TestPayload::TokenIds(vec![1, 2]),
            },
            TestInputPart {
                modality: InputModality::Text,
                payload: TestPayload::Embeddings(vec![1, 3, 4]),
            },
            TestInputPart {
                modality: InputModality::Image,
                payload: TestPayload::Embeddings(vec![1, 1, 4]),
            },
        ]);
        let mut extents = [2, 3, 1].into_iter();
        let admitted = admit_composite_input(&input, &TestInspector, |_| {
            Ok(Trap(extents.next().unwrap()))
        })
        .unwrap();
        let mut plans = Vec::new();
        admitted
            .visit_prompt_segments(|part| plans.push(part))
            .unwrap();
        assert_eq!(
            plans.iter().map(|p| p.decoder_range).collect::<Vec<_>>(),
            [[0, 2], [2, 5], [5, 6]]
        );
        assert_eq!(
            plans.iter().map(|p| p.payload).collect::<Vec<_>>(),
            [
                InputPayloadKind::TokenIds,
                InputPayloadKind::Embeddings,
                InputPayloadKind::Embeddings
            ]
        );
        assert_eq!(plans[2].modality, InputModality::Image);
        assert_eq!(admitted.decoder_positions(), 6);
    }
}

#[cfg(test)]
mod original_semantic_kernel_tests {
    use super::*;
    use eredu_runtime::input::host::{
        HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
    };
    use eredu_runtime::working_memory::WorkingMemoryPool;
    fn raw(policy: qwen::QwenPolicy<'_>, modality: InputModality, grid: &[i32]) {
        let vision = policy.vision.unwrap();
        let width = (vision.in_channels
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size) as usize;
        let count = grid
            .chunks_exact(3)
            .map(|g| (g[0] * g[1] * g[2]) as usize)
            .sum::<usize>();
        let values = (0..count * width)
            .map(|n| n as f32 / 137.)
            .collect::<Vec<_>>();
        let shape = [count, width];
        let grid_shape = [grid.len() / 3, 3];
        let metadata = [(
            InputMetadataKey::PatchGrid,
            HostTensorView {
                shape: &grid_shape,
                values: HostTensorValues::I32(grid),
            },
        )];
        let parts = [HostInputPart {
            modality,
            kind: eredu_core::InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &shape,
                values: HostTensorValues::F32(&values),
            },
            metadata: &metadata,
            extents: &[],
        }];
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let source = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
            .unwrap();
        let original = qwen::qwen_part(
            policy,
            qwen::InspectedPartRef::original(source.part(0).unwrap()).unwrap(),
        )
        .unwrap();
        let shape64 = shape.map(|n| n as u64);
        let grid64 = grid_shape.map(|n| n as u64);
        let ordinary = qwen::qwen_part(
            policy,
            qwen::InspectedPartRef {
                modality,
                kind: eredu_core::InputPayloadKind::Tensor,
                shape: qwen::ShapeRef::Legacy(&shape64),
                grid: Some(qwen::GridRef {
                    shape: qwen::ShapeRef::Legacy(&grid64),
                    values: grid,
                }),
            },
        )
        .unwrap();
        assert_eq!(original.positions, ordinary.positions);
        assert_eq!(original.placeholder, ordinary.placeholder);
        assert_eq!(original.grid, ordinary.grid);
        assert_eq!(original.workspace_scalars, ordinary.workspace_scalars);
        assert_eq!(
            original.positions,
            count as u64 / (vision.spatial_merge_size * vision.spatial_merge_size) as u64
        );
        assert!(original.workspace_scalars > values.len() as u64);
        assert_eq!(
            original.placeholder,
            match modality {
                InputModality::Image => policy.image.unwrap(),
                InputModality::Video => policy.video.unwrap(),
                _ => unreachable!(),
            } as u32
        );
    }
    #[test]
    fn original_and_legacy_grid_views_share_nonzero_geometry_and_workspace_equations() {
        let vl = super::tests::qwen_vl_args();
        let hybrid = super::tests::qwen_hybrid_args();
        for policy in [
            qwen::QwenPolicy {
                hidden: vl.text.hidden_size,
                vision: Some(&vl.vision),
                image: Some(vl.image_token_id),
                video: Some(vl.video_token_id),
                projected_media: false,
            },
            qwen::QwenPolicy {
                hidden: hybrid.text.hidden_size,
                vision: hybrid.vision.as_ref(),
                image: hybrid.image_token_id,
                video: hybrid.video_token_id,
                projected_media: true,
            },
        ] {
            for modality in [InputModality::Image, InputModality::Video] {
                raw(policy, modality, &[1, 4, 4, 5, 2, 2]);
            }
        }
    }
    #[test]
    fn borrowed_projection_and_malformed_grid_preserve_family_semantic_distinctions() {
        let vl = super::tests::qwen_vl_args();
        for projected_media in [false, true] {
            let policy = qwen::QwenPolicy {
                hidden: vl.text.hidden_size,
                vision: Some(&vl.vision),
                image: Some(vl.image_token_id),
                video: Some(vl.video_token_id),
                projected_media,
            };
            for modality in [
                InputModality::Text,
                InputModality::Image,
                InputModality::Video,
                InputModality::Audio,
            ] {
                let view = qwen::InspectedPartRef {
                    modality,
                    kind: eredu_core::InputPayloadKind::Embeddings,
                    shape: qwen::ShapeRef::Host(&[1, 3, 32]),
                    grid: None,
                };
                let result = qwen::qwen_part(policy, view);
                let allowed = modality == InputModality::Text
                    || (projected_media
                        && matches!(modality, InputModality::Image | InputModality::Video));
                assert_eq!(result.is_ok(), allowed);
                if let Ok(result) = result {
                    assert_eq!(result.positions, 3);
                    assert_eq!(result.placeholder, 0);
                }
            }
        }
        for (shape, values) in [
            (&[1usize, 3][..], &[1, 2][..]),
            (&[1usize, 2][..], &[1, 2][..]),
            (&[0usize, 3][..], &[][..]),
        ] {
            assert!(qwen::GridRef {
                shape: qwen::ShapeRef::Host(shape),
                values
            }
            .rows()
            .is_err());
        }
        let policy = qwen::QwenPolicy {
            hidden: 32,
            vision: Some(&vl.vision),
            image: Some(61),
            video: Some(62),
            projected_media: false,
        };
        for values in [[0, 2, 2], [1, 3, 2], [1, -2, 2]] {
            assert!(qwen::qwen_part(
                policy,
                qwen::InspectedPartRef {
                    modality: InputModality::Image,
                    kind: eredu_core::InputPayloadKind::Tensor,
                    shape: qwen::ShapeRef::Host(&[4, 24]),
                    grid: Some(qwen::GridRef {
                        shape: qwen::ShapeRef::Host(&[1, 3]),
                        values: &values
                    })
                }
            )
            .is_err());
        }
    }
    #[test]
    fn original_metadata_reads_match_raw_only_legacy_inspection_scope() {
        let vl = super::tests::qwen_vl_args();
        let policy = qwen::QwenPolicy {
            hidden: 32,
            vision: Some(&vl.vision),
            image: Some(61),
            video: Some(62),
            projected_media: true,
        };
        let values = [0.25_f32; 96];
        let grid = [1_i32, 2, 2];
        let wrong = [0.5_f32; 8];
        let metadata = [
            (
                InputMetadataKey::PatchGrid,
                HostTensorView {
                    shape: &[1, 3],
                    values: HostTensorValues::I32(&grid),
                },
            ),
            (
                InputMetadataKey::PatchPositions,
                HostTensorView {
                    shape: &[4, 2],
                    values: HostTensorValues::F32(&wrong),
                },
            ),
        ];
        let raw = [HostInputPart {
            modality: InputModality::Image,
            kind: eredu_core::InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[4, 24],
                values: HostTensorValues::F32(&values),
            },
            metadata: &metadata,
            extents: &[],
        }];
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let source = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&raw).unwrap())
            .unwrap();
        let error = qwen::InspectedPartRef::original(source.part(0).unwrap())
            .err()
            .expect("raw metadata is actually read");
        assert_eq!(error.key, Some(InputMetadataKey::PatchPositions));
        let projected = [HostInputPart {
            modality: InputModality::Image,
            kind: eredu_core::InputPayloadKind::Embeddings,
            payload: HostTensorView {
                shape: &[1, 3, 32],
                values: HostTensorValues::F32(&values),
            },
            metadata: &metadata,
            extents: &[],
        }];
        let source = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&projected).unwrap())
            .unwrap();
        let projected = qwen::InspectedPartRef::original(source.part(0).unwrap()).unwrap();
        assert!(projected.grid.is_none());
        assert_eq!(qwen::qwen_part(policy, projected).unwrap().positions, 3);
    }
}
