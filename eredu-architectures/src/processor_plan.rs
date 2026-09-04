//! Backend-neutral multimodal preprocessing policy.
//!
//! Family configuration, defaults, framing, sampling, resize geometry, and
//! patch/audio feature policy live here. A concrete backend executes the
//! declared pixel or signal transforms and constructs its native tensors.

use std::collections::{BTreeMap, HashMap};

use eredu_core::VideoSampling;
use eredu_gguf::{MetadataArray, MetadataValue};
use serde::Deserialize;

use crate::{configuration::SafetensorsArchitecturePlan, GgufArchitecture, ModelKind};

/// Conventional Hugging Face still-image processor configuration filename.
pub const PROCESSOR_CONFIG_FILENAME: &str = "preprocessor_config.json";
/// Conventional Hugging Face video processor configuration filename.
pub const VIDEO_PROCESSOR_CONFIG_FILENAME: &str = "video_preprocessor_config.json";
/// Muse-Glimmer's nested Hugging Face processor configuration filename.
pub const MUSE_PROCESSOR_CONFIG_FILENAME: &str = "processor_config.json";

/// Invalid family processor configuration or input geometry.
#[derive(Debug, thiserror::Error)]
pub enum ProcessorPlanError {
    /// Processor JSON could not be decoded.
    #[error("invalid processor JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Family preprocessing policy could not be satisfied.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ProcessorPlanError {
    ProcessorPlanError::Invalid(message.into())
}

/// Token IDs placed immediately around one prepared media tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaFraming {
    /// Opening family protocol token.
    pub start_token_id: u32,
    /// Closing family protocol token.
    pub end_token_id: u32,
}

/// Facade-owned tokenizer identities required to finish one GGUF media plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufSpecialTokenKind {
    /// Qwen image/video placeholders and media framing tokens.
    Qwen,
    /// Inkling image/audio content framing tokens.
    Inkling,
}

/// Typed token IDs resolved from strings only after tokenizer reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufSpecialTokenIds {
    /// Qwen tokenizer protocol IDs.
    Qwen {
        /// Image tensor placeholder.
        image_token_id: u32,
        /// Video tensor placeholder.
        video_token_id: u32,
        /// Opening vision framing token.
        vision_start_token_id: u32,
        /// Closing vision framing token.
        vision_end_token_id: u32,
    },
    /// Inkling tokenizer protocol IDs.
    Inkling {
        /// Opening image-content token.
        image_bos_token_id: u32,
        /// Opening audio-content token.
        audio_bos_token_id: u32,
    },
}

/// Backend-executed RGB transform selected by architecture policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbTransformPlan {
    /// Target image height.
    pub height: usize,
    /// Target image width.
    pub width: usize,
    /// Pixel interpolation selected by the family processor.
    pub resample: RgbResample,
    /// Scalar applied before channel normalization.
    pub rescale_factor: f32,
    /// Per-channel normalization mean.
    pub mean: [f32; 3],
    /// Per-channel normalization standard deviation.
    pub std: [f32; 3],
}

/// Backend-neutral pixel interpolation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbResample {
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos interpolation with a three-lobe kernel.
    Lanczos3,
}

#[derive(Debug, Clone, Deserialize)]
struct QwenProcessorSize {
    shortest_edge: u64,
    longest_edge: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct QwenVisualSource {
    size: QwenProcessorSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    #[serde(default = "default_true")]
    do_resize: bool,
    #[serde(default = "default_true")]
    do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    rescale_factor: f32,
    #[serde(default = "default_true")]
    do_normalize: bool,
    #[serde(default = "default_bicubic_resample")]
    resample: u8,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default = "default_qwen_video_fps")]
    fps: f64,
    #[serde(default = "default_qwen_min_frames")]
    min_frames: usize,
    #[serde(default = "default_qwen_max_frames")]
    max_frames: usize,
    #[serde(default = "default_true")]
    do_sample_frames: bool,
}

#[derive(Debug, Deserialize)]
struct QwenModelSource {
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
    #[serde(default)]
    text_config: Option<QwenTextSource>,
}

#[derive(Debug, Deserialize)]
struct QwenTextSource {
    vision_start_token_id: Option<u32>,
    vision_end_token_id: Option<u32>,
}

const fn default_true() -> bool {
    true
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

const fn default_bicubic_resample() -> u8 {
    3
}

fn default_qwen_video_fps() -> f64 {
    2.0
}

const fn default_qwen_min_frames() -> usize {
    4
}

const fn default_qwen_max_frames() -> usize {
    768
}

/// Qwen patch packing geometry consumed by a concrete backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenPatchPlan {
    /// Spatial patch side length.
    pub patch_size: usize,
    /// Frames folded into each patch row.
    pub temporal_patch_size: usize,
    /// Spatial merge side length used for patch ordering.
    pub merge_size: usize,
}

/// Complete Qwen still-image preprocessing plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenImagePlan {
    /// Family framing token IDs.
    pub framing: MediaFraming,
    /// RGB resize and normalization.
    pub transform: RgbTransformPlan,
    /// Patch packing geometry.
    pub patches: QwenPatchPlan,
}

/// One framed temporal group in a Qwen video plan.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVideoGroupPlan {
    /// Source frame indices, padded to the temporal patch size.
    pub source_indices: Vec<usize>,
    /// Family timestamp text to tokenize before the opening token.
    pub timestamp_text: String,
}

/// Complete Qwen video preprocessing plan.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVideoPlan {
    /// Family framing token IDs used for every temporal group.
    pub framing: MediaFraming,
    /// RGB resize and normalization shared by all selected frames.
    pub transform: RgbTransformPlan,
    /// Patch packing geometry.
    pub patches: QwenPatchPlan,
    /// Ordered framed temporal groups.
    pub groups: Vec<QwenVideoGroupPlan>,
}

/// Normalized Qwen image/video processor policy.
#[derive(Debug, Clone)]
pub struct QwenProcessorPlan {
    image: Option<QwenVisualSource>,
    video: Option<QwenVisualSource>,
    framing: Option<MediaFraming>,
}

impl QwenProcessorPlan {
    /// Parses Hugging Face model and optional visual processor JSON.
    pub fn from_hf_json(
        model: &[u8],
        image: Option<&[u8]>,
        video: Option<&[u8]>,
    ) -> Result<Option<Self>, ProcessorPlanError> {
        let image = image.map(parse_qwen_visual).transpose()?;
        let video = video.map(parse_qwen_visual).transpose()?;
        if image.is_none() && video.is_none() {
            return Ok(None);
        }
        let model: QwenModelSource = serde_json::from_slice(model)?;
        let text = model.text_config.as_ref();
        let start_token_id = model
            .vision_start_token_id
            .or_else(|| text.and_then(|config| config.vision_start_token_id))
            .ok_or_else(|| {
                invalid("Qwen processor requires vision_start_token_id in config.json")
            })?;
        let end_token_id = model
            .vision_end_token_id
            .or_else(|| text.and_then(|config| config.vision_end_token_id))
            .ok_or_else(|| invalid("Qwen processor requires vision_end_token_id in config.json"))?;
        Ok(Some(Self {
            image,
            video,
            framing: Some(MediaFraming {
                start_token_id,
                end_token_id,
            }),
        }))
    }

    /// Derives Qwen preprocessing geometry from the admitted GGUF projector.
    fn from_gguf_metadata(
        projector: &BTreeMap<String, MetadataValue>,
    ) -> Result<Self, ProcessorPlanError> {
        let patch_size = required_btree_usize(projector, "clip.vision.patch_size")?;
        let merge_size =
            optional_btree_usize(projector, "clip.vision.spatial_merge_size")?.unwrap_or(2);
        let factor = patch_size
            .checked_mul(merge_size)
            .ok_or_else(|| invalid("Qwen GGUF resize factor overflow"))?;
        let pixels_per_token = factor
            .checked_mul(factor)
            .ok_or_else(|| invalid("Qwen GGUF pixel geometry overflow"))?;
        let min_pixels = optional_btree_usize(projector, "clip.vision.image_min_pixels")?
            .unwrap_or(
                8usize
                    .checked_mul(pixels_per_token)
                    .ok_or_else(|| invalid("Qwen GGUF minimum pixel geometry overflow"))?,
            );
        let max_pixels = optional_btree_usize(projector, "clip.vision.image_max_pixels")?
            .unwrap_or(
                4096usize
                    .checked_mul(pixels_per_token)
                    .ok_or_else(|| invalid("Qwen GGUF maximum pixel geometry overflow"))?,
            );
        let visual = QwenVisualSource {
            size: QwenProcessorSize {
                shortest_edge: min_pixels as u64,
                longest_edge: max_pixels as u64,
            },
            patch_size,
            temporal_patch_size: 2,
            merge_size,
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_rescale_factor(),
            do_normalize: true,
            resample: default_bicubic_resample(),
            image_mean: required_btree_rgb(projector, "clip.vision.image_mean")?,
            image_std: required_btree_rgb(projector, "clip.vision.image_std")?,
            fps: default_qwen_video_fps(),
            min_frames: default_qwen_min_frames(),
            max_frames: default_qwen_max_frames(),
            do_sample_frames: true,
        };
        parse_qwen_visual_source(&visual)?;
        Ok(Self {
            image: Some(visual.clone()),
            video: Some(visual),
            framing: None,
        })
    }

    fn bind_framing(&mut self, framing: MediaFraming) {
        self.framing = Some(framing);
    }

    const fn has_framing(&self) -> bool {
        self.framing.is_some()
    }

    /// Derives one still-image transform and patch plan.
    pub fn image(&self, height: usize, width: usize) -> Result<QwenImagePlan, ProcessorPlanError> {
        let source = self
            .image
            .as_ref()
            .ok_or_else(|| invalid("Qwen model directory has no image processor config"))?;
        let factor = source
            .patch_size
            .checked_mul(source.merge_size)
            .ok_or_else(|| invalid("Qwen image resize factor overflow"))?;
        let (height, width) = if source.do_resize {
            qwen_smart_resize(
                height,
                width,
                factor,
                source.size.shortest_edge as usize,
                source.size.longest_edge as usize,
            )?
        } else {
            (height, width)
        };
        Ok(QwenImagePlan {
            framing: self
                .framing
                .ok_or_else(|| invalid("Qwen GGUF media framing token IDs are unresolved"))?,
            transform: qwen_transform(source, height, width),
            patches: qwen_patches(source),
        })
    }

    /// Derives sampling, framing, RGB transform, and patch geometry for a video.
    pub fn video(
        &self,
        total_frames: usize,
        height: usize,
        width: usize,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Result<QwenVideoPlan, ProcessorPlanError> {
        let source = self
            .video
            .as_ref()
            .ok_or_else(|| invalid("Qwen model directory has no video processor config"))?;
        let source_fps = source_fps.unwrap_or(24.0);
        validate_fps(source_fps, "video source FPS")?;
        let sample_count = match sampling {
            VideoSampling::ProcessorDefault if source.do_sample_frames => sampled_frame_count(
                total_frames,
                source_fps,
                source.fps,
                source.min_frames,
                source.max_frames,
            )?,
            VideoSampling::ProcessorDefault | VideoSampling::All => total_frames,
            VideoSampling::Fps(target_fps) => sampled_frame_count(
                total_frames,
                source_fps,
                target_fps,
                source.min_frames,
                source.max_frames,
            )?,
            VideoSampling::FrameCount(count) => count.clamp(1, total_frames),
        };
        let mut indices = uniform_sample_indices(total_frames, sample_count)?;
        let factor = source
            .patch_size
            .checked_mul(source.merge_size)
            .ok_or_else(|| invalid("Qwen video resize factor overflow"))?;
        let (height, width) = if source.do_resize {
            qwen_smart_resize_video(
                indices.len(),
                height,
                width,
                source.temporal_patch_size,
                factor,
                source.size.shortest_edge as usize,
                source.size.longest_edge as usize,
            )?
        } else {
            (height, width)
        };
        pad_frame_indices(&mut indices, source.temporal_patch_size)?;
        let groups = indices
            .chunks_exact(source.temporal_patch_size)
            .map(|chunk| {
                let first = chunk[0] as f64 / source_fps;
                let last = chunk[chunk.len() - 1] as f64 / source_fps;
                QwenVideoGroupPlan {
                    source_indices: chunk.to_vec(),
                    timestamp_text: format!("<{:.1} seconds>", (first + last) / 2.0),
                }
            })
            .collect();
        Ok(QwenVideoPlan {
            framing: self
                .framing
                .ok_or_else(|| invalid("Qwen GGUF media framing token IDs are unresolved"))?,
            transform: qwen_transform(source, height, width),
            patches: qwen_patches(source),
            groups,
        })
    }
}

fn parse_qwen_visual(bytes: &[u8]) -> Result<QwenVisualSource, ProcessorPlanError> {
    let source: QwenVisualSource = serde_json::from_slice(bytes)?;
    parse_qwen_visual_source(&source)?;
    Ok(source)
}

fn parse_qwen_visual_source(source: &QwenVisualSource) -> Result<(), ProcessorPlanError> {
    if source.patch_size == 0 || source.temporal_patch_size == 0 || source.merge_size == 0 {
        return Err(invalid(
            "Qwen patch_size, temporal_patch_size, and merge_size must be positive",
        ));
    }
    if source.size.shortest_edge == 0
        || source.size.longest_edge == 0
        || source.size.shortest_edge > source.size.longest_edge
    {
        return Err(invalid(format!(
            "invalid Qwen image size constraints: {}..{} pixels",
            source.size.shortest_edge, source.size.longest_edge
        )));
    }
    if source.resample != default_bicubic_resample() {
        return Err(invalid(format!(
            "Qwen visual resample mode {} is unsupported; expected bicubic mode 3",
            source.resample
        )));
    }
    if !source.fps.is_finite()
        || source.fps <= 0.0
        || source.min_frames == 0
        || source.max_frames < source.min_frames
    {
        return Err(invalid(format!(
            "invalid Qwen video sampling defaults: fps {}, frames {}..{}",
            source.fps, source.min_frames, source.max_frames
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum ArtifactFamilyPlan {
    Safetensors(SafetensorsArchitecturePlan),
    Gguf(crate::configuration::GgufArchitecturePlan),
}

#[derive(Debug, Clone)]
enum NormalizedProcessorPlan {
    Gemma4(Gemma4ProcessorPlan),
    Inkling(InklingProcessorPlan),
    Muse(MuseProcessorPlan),
    Qwen(QwenProcessorPlan),
}

/// Authoritative normalized family and processor state retained by inspection.
///
/// Every artifact admitted by the architecture registry contains a concrete
/// typed family plan before processor enrichment or materialization.
#[derive(Debug, Clone)]
pub struct ArtifactArchitecturePlan {
    family: ArtifactFamilyPlan,
    media_projector: Option<crate::gguf_companion::GgufMediaProjectorPlan>,
    processor: Option<NormalizedProcessorPlan>,
    prediction_extension: Option<crate::configuration::PredictionExtensionPlan>,
}

impl ArtifactArchitecturePlan {
    /// Retains the typed SafeTensors architecture and checkpoint plan from resolution.
    pub(crate) fn from_safetensors_architecture(architecture: SafetensorsArchitecturePlan) -> Self {
        Self {
            family: ArtifactFamilyPlan::Safetensors(architecture),
            media_projector: None,
            processor: None,
            prediction_extension: None,
        }
    }

    /// Retains the exact GGUF architecture from resolution.
    pub(crate) fn from_gguf_architecture(
        architecture: crate::configuration::GgufArchitecturePlan,
    ) -> Self {
        Self {
            family: ArtifactFamilyPlan::Gguf(architecture),
            media_projector: None,
            processor: None,
            prediction_extension: None,
        }
    }

    pub(crate) fn with_gguf_media_projector(
        mut self,
        media_projector: Option<crate::gguf_companion::GgufMediaProjectorPlan>,
    ) -> Self {
        self.media_projector = media_projector;
        self
    }

    /// Separates an admitted embedded-prediction artifact from its ordinary target plan.
    pub fn prediction_target_projection(
        &self,
    ) -> Result<
        Option<(Self, crate::configuration::PredictionExtensionPlan)>,
        eredu_core::artifact::ArtifactError,
    > {
        let ArtifactFamilyPlan::Safetensors(family) = &self.family else {
            return Ok(None);
        };
        let Some((family, extension)) = family.prediction_target_projection()? else {
            return Ok(None);
        };
        Ok(Some((
            Self {
                family: ArtifactFamilyPlan::Safetensors(family),
                media_projector: self.media_projector.clone(),
                processor: self.processor.clone(),
                prediction_extension: Some(extension.clone()),
            },
            extension,
        )))
    }

    /// Typed additive prediction extension retained by a target projection.
    pub const fn prediction_extension(
        &self,
    ) -> Option<&crate::configuration::PredictionExtensionPlan> {
        self.prediction_extension.as_ref()
    }

    /// Finalizes catalog-dependent SafeTensors admission before processor enrichment.
    pub(crate) fn admit_safetensors_catalog(
        mut self,
        tensors: &eredu_core::checkpoint::TensorCatalog,
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        match &mut self.family {
            ArtifactFamilyPlan::Safetensors(plan) => plan.admit_catalog(tensors)?,
            ArtifactFamilyPlan::Gguf(_) => {
                return Err(
                    eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(
                        "SafeTensors catalog admission requires a SafeTensors architecture plan"
                            .into(),
                    ),
                );
            }
        }
        Ok(self)
    }

    /// Normalizes one SafeTensors family and its processor sidecars.
    pub(crate) fn with_safetensors_processors(
        mut self,
        model: &[u8],
        image: Option<&[u8]>,
        video: Option<&[u8]>,
        muse: Option<&[u8]>,
    ) -> Result<Self, ProcessorPlanError> {
        let architecture = self.safetensors_architecture().ok_or_else(|| {
            invalid("SafeTensors processor planning requires a validated architecture plan")
        })?;
        let kind = architecture.model_kind();
        let processor = match kind {
            ModelKind::Gemma4 => Gemma4ProcessorPlan::from_hf_json(model, image, video)?
                .map(NormalizedProcessorPlan::Gemma4),
            ModelKind::Inkling => {
                InklingProcessorPlan::from_hf_json(model)?.map(NormalizedProcessorPlan::Inkling)
            }
            ModelKind::MuseGlimmer => muse
                .map(MuseProcessorPlan::from_hf_json)
                .transpose()?
                .map(NormalizedProcessorPlan::Muse),
            ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe | ModelKind::Qwen35 => {
                QwenProcessorPlan::from_hf_json(model, image, video)?
                    .map(NormalizedProcessorPlan::Qwen)
            }
            _ => None,
        };
        self.processor = processor;
        Ok(self)
    }

    /// Normalizes one GGUF architecture and its admitted media projector.
    pub(crate) fn with_gguf_processors(
        mut self,
        model: &BTreeMap<String, MetadataValue>,
        projector: Option<&BTreeMap<String, MetadataValue>>,
    ) -> Result<Self, ProcessorPlanError> {
        let architecture = self
            .gguf_architecture()
            .ok_or_else(|| invalid("GGUF processor planning requires a resolved architecture"))?;
        let as_hash_map = |metadata: &BTreeMap<String, MetadataValue>| {
            metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<HashMap<_, _>>()
        };
        let processor = match architecture {
            GgufArchitecture::Gemma4 => projector
                .map(|projector| {
                    Gemma4ProcessorPlan::from_gguf_metadata(
                        &as_hash_map(model),
                        &as_hash_map(projector),
                    )
                })
                .transpose()?
                .map(NormalizedProcessorPlan::Gemma4),
            GgufArchitecture::Inkling if projector.is_some() => Some(
                NormalizedProcessorPlan::Inkling(InklingProcessorPlan::from_gguf_metadata()?),
            ),
            GgufArchitecture::MuseGlimmer => projector
                .map(|projector| MuseProcessorPlan::from_gguf_metadata(&as_hash_map(projector)))
                .transpose()?
                .map(NormalizedProcessorPlan::Muse),
            GgufArchitecture::Qwen3Vl
            | GgufArchitecture::Qwen3VlMoe
            | GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe => projector
                .map(QwenProcessorPlan::from_gguf_metadata)
                .transpose()?
                .map(NormalizedProcessorPlan::Qwen),
            _ => None,
        };
        self.processor = processor;
        Ok(self)
    }

    /// Returns the normalized model family retained during inspection.
    pub const fn model_kind(&self) -> ModelKind {
        match &self.family {
            ArtifactFamilyPlan::Safetensors(plan) => plan.model_kind(),
            ArtifactFamilyPlan::Gguf(plan) => plan.model_kind(),
        }
    }

    /// Exact grouped mechanisms required by this architecture under `topology`.
    pub fn grouped_operation_requirements(
        &self,
        topology: Option<eredu_core::ParallelTopology>,
    ) -> Vec<eredu_runtime::GroupedOperationRequirement> {
        let routed = match &self.family {
            ArtifactFamilyPlan::Safetensors(plan) => plan.model().uses_grouped_routed_experts(),
            ArtifactFamilyPlan::Gguf(plan) => plan.model().uses_grouped_routed_experts(),
        };
        if !routed {
            return Vec::new();
        }
        let mut required = vec![eredu_runtime::GroupedOperationRequirement::GatedProduct];
        if topology.is_some_and(|topology| topology.tensor() > 1) {
            required.push(
                eredu_runtime::GroupedOperationRequirement::GatedProductTensorParallelPartial,
            );
        }
        required
    }

    /// Returns the exact normalized GGUF architecture, when applicable.
    pub const fn gguf_architecture(&self) -> Option<GgufArchitecture> {
        match &self.family {
            ArtifactFamilyPlan::Gguf(plan) => Some(plan.architecture()),
            ArtifactFamilyPlan::Safetensors(_) => None,
        }
    }

    /// Returns the validated SafeTensors architecture and checkpoint plan.
    pub const fn safetensors_architecture(&self) -> Option<&SafetensorsArchitecturePlan> {
        match &self.family {
            ArtifactFamilyPlan::Safetensors(plan) => Some(plan),
            ArtifactFamilyPlan::Gguf(_) => None,
        }
    }

    /// Returns the validated GGUF architecture geometry and checkpoint plan.
    pub const fn gguf_plan(&self) -> Option<&crate::configuration::GgufArchitecturePlan> {
        match &self.family {
            ArtifactFamilyPlan::Gguf(plan) => Some(plan),
            ArtifactFamilyPlan::Safetensors(_) => None,
        }
    }

    /// Returns the typed, structurally validated GGUF media-projector plan.
    pub const fn gguf_media_projector(
        &self,
    ) -> Option<&crate::gguf_companion::GgufMediaProjectorPlan> {
        self.media_projector.as_ref()
    }

    /// Whether authoritative artifact inspection admitted a media processor.
    pub const fn has_processor(&self) -> bool {
        self.processor.is_some()
    }

    /// Tokenizer protocol required to make an admitted GGUF media plan executable.
    pub const fn required_gguf_special_tokens(&self) -> Option<GgufSpecialTokenKind> {
        match &self.processor {
            Some(NormalizedProcessorPlan::Qwen(plan)) if !plan.has_framing() => {
                Some(GgufSpecialTokenKind::Qwen)
            }
            Some(NormalizedProcessorPlan::Inkling(plan)) if !plan.has_token_ids() => {
                Some(GgufSpecialTokenKind::Inkling)
            }
            _ => None,
        }
    }

    /// Binds facade-resolved GGUF tokenizer IDs after structural admission.
    pub fn bind_gguf_special_token_ids(
        &mut self,
        ids: GgufSpecialTokenIds,
    ) -> Result<(), ProcessorPlanError> {
        match ids {
            GgufSpecialTokenIds::Qwen {
                image_token_id,
                video_token_id,
                vision_start_token_id,
                vision_end_token_id,
            } => {
                if !matches!(&self.processor, Some(NormalizedProcessorPlan::Qwen(_))) {
                    return Err(invalid("Qwen GGUF special tokens require a processor plan"));
                }
                let projector = self.media_projector.as_mut().ok_or_else(|| {
                    invalid("Qwen GGUF special tokens require an admitted media projector")
                })?;
                projector
                    .bind_qwen_token_ids(image_token_id, video_token_id)
                    .map_err(invalid)?;
                let Some(NormalizedProcessorPlan::Qwen(plan)) = self.processor.as_mut() else {
                    unreachable!("Qwen processor presence checked before projector mutation")
                };
                plan.bind_framing(MediaFraming {
                    start_token_id: vision_start_token_id,
                    end_token_id: vision_end_token_id,
                });
            }
            GgufSpecialTokenIds::Inkling {
                image_bos_token_id,
                audio_bos_token_id,
            } => {
                let Some(NormalizedProcessorPlan::Inkling(plan)) = self.processor.as_mut() else {
                    return Err(invalid(
                        "Inkling GGUF special tokens require a processor plan",
                    ));
                };
                plan.bind_token_ids(image_bos_token_id, audio_bos_token_id);
            }
        }
        Ok(())
    }

    /// Returns the retained Gemma 4 processor plan, when admitted.
    pub const fn gemma4(&self) -> Option<&Gemma4ProcessorPlan> {
        match &self.processor {
            Some(NormalizedProcessorPlan::Gemma4(plan)) => Some(plan),
            _ => None,
        }
    }

    /// Returns the retained Inkling processor plan, when admitted.
    pub const fn inkling(&self) -> Option<&InklingProcessorPlan> {
        match &self.processor {
            Some(NormalizedProcessorPlan::Inkling(plan)) if plan.has_token_ids() => Some(plan),
            _ => None,
        }
    }

    /// Returns the retained Muse-Glimmer processor plan, when admitted.
    pub const fn muse(&self) -> Option<&MuseProcessorPlan> {
        match &self.processor {
            Some(NormalizedProcessorPlan::Muse(plan)) => Some(plan),
            _ => None,
        }
    }

    /// Returns the retained Qwen processor plan, when admitted.
    pub const fn qwen(&self) -> Option<&QwenProcessorPlan> {
        match &self.processor {
            Some(NormalizedProcessorPlan::Qwen(plan)) if plan.has_framing() => Some(plan),
            _ => None,
        }
    }
}

fn qwen_patches(source: &QwenVisualSource) -> QwenPatchPlan {
    QwenPatchPlan {
        patch_size: source.patch_size,
        temporal_patch_size: source.temporal_patch_size,
        merge_size: source.merge_size,
    }
}

fn qwen_transform(source: &QwenVisualSource, height: usize, width: usize) -> RgbTransformPlan {
    RgbTransformPlan {
        height,
        width,
        resample: RgbResample::Bicubic,
        rescale_factor: if source.do_rescale {
            source.rescale_factor
        } else {
            1.0
        },
        mean: if source.do_normalize {
            source.image_mean
        } else {
            [0.0; 3]
        },
        std: if source.do_normalize {
            source.image_std
        } else {
            [1.0; 3]
        },
    }
}

fn qwen_smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), ProcessorPlanError> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(invalid(format!(
            "smart resize requires positive dimensions and factor, got {width}x{height}, factor {factor}"
        )));
    }
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > 200.0 {
        return Err(invalid(format!(
            "absolute image aspect ratio must be at most 200, got {ratio}"
        )));
    }
    let round_to_factor =
        |value: usize| ((value as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut resized_height = round_to_factor(height).max(factor);
    let mut resized_width = round_to_factor(width).max(factor);
    let area = resized_height.saturating_mul(resized_width);
    if area > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        resized_height =
            ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        resized_width =
            ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

fn qwen_smart_resize_video(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), ProcessorPlanError> {
    if num_frames == 0 || temporal_factor == 0 || factor == 0 {
        return Err(invalid(
            "video smart resize requires frames and positive factors",
        ));
    }
    if height < factor || width < factor {
        return Err(invalid(format!(
            "video dimensions {width}x{height} must be at least resize factor {factor}"
        )));
    }
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    if ratio > 200.0 {
        return Err(invalid(format!(
            "absolute video aspect ratio must be at most 200, got {ratio}"
        )));
    }
    let round_to_factor =
        |value: usize| ((value as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut resized_height = round_to_factor(height).max(factor);
    let mut resized_width = round_to_factor(width).max(factor);
    let padded_frames = num_frames.div_ceil(temporal_factor) * temporal_factor;
    let volume = padded_frames
        .saturating_mul(resized_height)
        .saturating_mul(resized_width);
    if volume > max_pixels {
        let beta = ((num_frames * height * width) as f64 / max_pixels as f64).sqrt();
        resized_height =
            ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        resized_width =
            ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if volume < min_pixels {
        let beta = (min_pixels as f64 / (num_frames * height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4ModelSource {
    boi_token_id: Option<u32>,
    eoi_token_id: Option<u32>,
    boa_token_id: Option<u32>,
    eoa_token_id: Option<u32>,
    #[serde(default = "default_gemma_soft_tokens")]
    vision_soft_tokens_per_image: usize,
    vision_config: Option<Gemma4VisionSource>,
    audio_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4VisionSource {
    #[serde(default = "default_gemma_patch_size")]
    patch_size: usize,
    #[serde(default = "default_gemma_pooling_kernel_size")]
    pooling_kernel_size: usize,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Gemma4ImageProcessorSource {
    patch_size: Option<usize>,
    pooling_kernel_size: Option<usize>,
    max_soft_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4VideoProcessorSource {
    #[serde(default = "default_gemma_patch_size")]
    patch_size: usize,
    #[serde(default = "default_gemma_pooling_kernel_size")]
    pooling_kernel_size: usize,
    #[serde(default = "default_gemma_video_soft_tokens")]
    max_soft_tokens: usize,
    #[serde(default = "default_gemma_video_frames")]
    num_frames: usize,
}

impl Default for Gemma4VideoProcessorSource {
    fn default() -> Self {
        Self {
            patch_size: default_gemma_patch_size(),
            pooling_kernel_size: default_gemma_pooling_kernel_size(),
            max_soft_tokens: default_gemma_video_soft_tokens(),
            num_frames: default_gemma_video_frames(),
        }
    }
}

const fn default_gemma_patch_size() -> usize {
    16
}

const fn default_gemma_pooling_kernel_size() -> usize {
    3
}

const fn default_gemma_soft_tokens() -> usize {
    280
}

const fn default_gemma_video_soft_tokens() -> usize {
    70
}

const fn default_gemma_video_frames() -> usize {
    32
}

#[derive(Debug, Clone)]
struct Gemma4VisualPolicy {
    patch_size: usize,
    pooling_kernel_size: usize,
    max_soft_tokens: usize,
}

/// Complete Gemma 4 image preprocessing plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4ImagePlan {
    /// Family framing token IDs.
    pub framing: MediaFraming,
    /// RGB resize and normalization.
    pub transform: RgbTransformPlan,
    /// Spatial patch side length.
    pub patch_size: usize,
    /// Padded patch count emitted by the backend.
    pub max_patches: usize,
}

/// One selected and framed Gemma 4 video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4VideoFramePlan {
    /// Source frame index.
    pub source_index: usize,
    /// Family timestamp text to tokenize before the opening token.
    pub timestamp_text: String,
}

/// Complete Gemma 4 video preprocessing plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VideoPlan {
    /// Family framing token IDs used for every selected frame.
    pub framing: MediaFraming,
    /// RGB resize and normalization shared by all selected frames.
    pub transform: RgbTransformPlan,
    /// Spatial patch side length.
    pub patch_size: usize,
    /// Padded patch count emitted for every frame.
    pub max_patches: usize,
    /// Ordered selected source frames and timestamp text.
    pub frames: Vec<Gemma4VideoFramePlan>,
}

/// Backend-executed Gemma 4 log-mel feature policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4AudioPlan {
    /// Family framing token IDs.
    pub framing: MediaFraming,
    /// Expected waveform sample rate.
    pub sample_rate: u32,
    /// Analysis frame length in samples.
    pub frame_length: usize,
    /// Analysis hop length in samples.
    pub hop_length: usize,
    /// FFT length.
    pub fft_length: usize,
    /// Number of mel bands.
    pub mel_bins: usize,
    /// Lower mel filter frequency.
    pub min_frequency: f32,
    /// Upper mel filter frequency.
    pub max_frequency: f32,
    /// Feature floor before logarithm.
    pub mel_floor: f32,
    /// Maximum waveform sample count.
    pub max_samples: usize,
    /// Feature frames are padded to this multiple.
    pub pad_to_multiple: usize,
}

/// Normalized Gemma 4 image, video, and audio processor policy.
#[derive(Debug, Clone)]
pub struct Gemma4ProcessorPlan {
    image: Option<Gemma4VisualPolicy>,
    video: Option<(Gemma4VisualPolicy, usize)>,
    image_framing: Option<MediaFraming>,
    audio_framing: Option<MediaFraming>,
    has_audio: bool,
}

impl Gemma4ProcessorPlan {
    /// Parses Hugging Face model and optional visual processor JSON.
    pub fn from_hf_json(
        model: &[u8],
        image: Option<&[u8]>,
        video: Option<&[u8]>,
    ) -> Result<Option<Self>, ProcessorPlanError> {
        let model: Gemma4ModelSource = serde_json::from_slice(model)?;
        let image_source: Gemma4ImageProcessorSource = image
            .map(serde_json::from_slice)
            .transpose()?
            .unwrap_or_default();
        let video_source: Gemma4VideoProcessorSource = video
            .map(serde_json::from_slice)
            .transpose()?
            .unwrap_or_default();
        let image_policy = model
            .vision_config
            .as_ref()
            .map(|vision| Gemma4VisualPolicy {
                patch_size: image_source.patch_size.unwrap_or(vision.patch_size),
                pooling_kernel_size: image_source
                    .pooling_kernel_size
                    .unwrap_or(vision.pooling_kernel_size),
                max_soft_tokens: image_source
                    .max_soft_tokens
                    .unwrap_or(model.vision_soft_tokens_per_image),
            });
        let video_policy = model.vision_config.as_ref().map(|_| {
            (
                Gemma4VisualPolicy {
                    patch_size: video_source.patch_size,
                    pooling_kernel_size: video_source.pooling_kernel_size,
                    max_soft_tokens: video_source.max_soft_tokens,
                },
                video_source.num_frames,
            )
        });
        let plan = Self {
            image: image_policy,
            video: video_policy,
            image_framing: optional_framing(
                model.boi_token_id,
                model.eoi_token_id,
                "Gemma 4 visual",
            )?,
            audio_framing: optional_framing(
                model.boa_token_id,
                model.eoa_token_id,
                "Gemma 4 audio",
            )?,
            has_audio: model.audio_config.is_some(),
        };
        plan.validate()?;
        if plan.image.is_none() && !plan.has_audio {
            Ok(None)
        } else {
            Ok(Some(plan))
        }
    }

    /// Parses processor policy from portable GGUF metadata.
    pub fn from_gguf_metadata(
        model: &HashMap<String, MetadataValue>,
        projector: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ProcessorPlanError> {
        let has_vision = projector
            .get("clip.has_vision_encoder")
            .and_then(MetadataValue::as_bool)
            .unwrap_or_else(|| projector.keys().any(|key| key.starts_with("clip.vision.")));
        let patch_size = optional_metadata_usize(projector, "clip.vision.patch_size")?
            .unwrap_or(default_gemma_patch_size());
        let pooling_kernel_size =
            optional_metadata_usize(projector, "clip.vision.pooling_kernel_size")?
                .unwrap_or(default_gemma_pooling_kernel_size());
        let max_soft_tokens = optional_metadata_usize(projector, "clip.vision.max_soft_tokens")?
            .unwrap_or(default_gemma_soft_tokens());
        let video_max_soft_tokens =
            optional_metadata_usize(projector, "clip.vision.video.max_soft_tokens")?
                .unwrap_or(default_gemma_video_soft_tokens());
        let video_num_frames = optional_metadata_usize(projector, "clip.vision.video.frame_count")?
            .unwrap_or(default_gemma_video_frames());
        let plan = Self {
            image: has_vision.then_some(Gemma4VisualPolicy {
                patch_size,
                pooling_kernel_size,
                max_soft_tokens,
            }),
            video: has_vision.then_some((
                Gemma4VisualPolicy {
                    patch_size: optional_metadata_usize(projector, "clip.vision.video.patch_size")?
                        .unwrap_or(patch_size),
                    pooling_kernel_size: optional_metadata_usize(
                        projector,
                        "clip.vision.video.pooling_kernel_size",
                    )?
                    .unwrap_or(pooling_kernel_size),
                    max_soft_tokens: video_max_soft_tokens,
                },
                video_num_frames,
            )),
            image_framing: optional_framing(
                optional_metadata_u32(model, "gemma4.boi_token_id")?,
                optional_metadata_u32(model, "gemma4.eoi_token_id")?,
                "Gemma 4 visual",
            )?,
            audio_framing: optional_framing(
                optional_metadata_u32(model, "gemma4.boa_token_id")?,
                optional_metadata_u32(model, "gemma4.eoa_token_id")?,
                "Gemma 4 audio",
            )?,
            has_audio: projector
                .get("clip.has_audio_encoder")
                .and_then(MetadataValue::as_bool)
                .unwrap_or(false),
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Whether the model declares a visual processor.
    pub const fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Whether the model declares an audio processor.
    pub const fn has_audio(&self) -> bool {
        self.has_audio
    }

    /// Derives one still-image transform and padded patch plan.
    pub fn image(
        &self,
        height: usize,
        width: usize,
    ) -> Result<Gemma4ImagePlan, ProcessorPlanError> {
        let policy = self
            .image
            .as_ref()
            .ok_or_else(|| invalid("Gemma 4 model has no image processor"))?;
        let framing = self.image_framing.ok_or_else(|| {
            invalid("Gemma 4 image processor requires boi_token_id and eoi_token_id")
        })?;
        let (transform, max_patches) = gemma_visual_plan(policy, height, width)?;
        Ok(Gemma4ImagePlan {
            framing,
            transform,
            patch_size: policy.patch_size,
            max_patches,
        })
    }

    /// Derives sampling, framing, RGB transform, and patches for a video.
    pub fn video(
        &self,
        total_frames: usize,
        height: usize,
        width: usize,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Result<Gemma4VideoPlan, ProcessorPlanError> {
        let (policy, default_frames) = self
            .video
            .as_ref()
            .ok_or_else(|| invalid("Gemma 4 model has no video processor"))?;
        let framing = self.image_framing.ok_or_else(|| {
            invalid("Gemma 4 video processor requires boi_token_id and eoi_token_id")
        })?;
        let source_fps = source_fps.unwrap_or(24.0);
        validate_fps(source_fps, "video source FPS")?;
        let sample_count = match sampling {
            VideoSampling::ProcessorDefault => (*default_frames).min(total_frames),
            VideoSampling::All => total_frames,
            VideoSampling::FrameCount(count) => count.clamp(1, total_frames),
            VideoSampling::Fps(target_fps) => {
                sampled_frame_count(total_frames, source_fps, target_fps, 1, *default_frames)?
            }
        };
        let indices = uniform_sample_indices(total_frames, sample_count)?;
        let frames = indices
            .into_iter()
            .enumerate()
            .map(|(index, source_index)| Gemma4VideoFramePlan {
                source_index,
                timestamp_text: gemma_timestamp(source_index, source_fps, index == 0),
            })
            .collect();
        let (transform, max_patches) = gemma_visual_plan(policy, height, width)?;
        Ok(Gemma4VideoPlan {
            framing,
            transform,
            patch_size: policy.patch_size,
            max_patches,
            frames,
        })
    }

    /// Returns the architecture-declared audio framing and feature policy.
    pub fn audio(&self) -> Result<Gemma4AudioPlan, ProcessorPlanError> {
        if !self.has_audio {
            return Err(invalid("Gemma 4 model has no audio processor"));
        }
        let framing = self.audio_framing.ok_or_else(|| {
            invalid("Gemma 4 audio processor requires boa_token_id and eoa_token_id")
        })?;
        Ok(Gemma4AudioPlan {
            framing,
            sample_rate: 16_000,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            mel_bins: 128,
            min_frequency: 0.0,
            max_frequency: 8_000.0,
            mel_floor: 1e-3,
            max_samples: 480_000,
            pad_to_multiple: 128,
        })
    }

    fn validate(&self) -> Result<(), ProcessorPlanError> {
        for (kind, policy) in self
            .image
            .iter()
            .map(|policy| ("image", policy))
            .chain(self.video.iter().map(|(policy, _)| ("video", policy)))
        {
            if policy.patch_size == 0 || policy.pooling_kernel_size == 0 {
                return Err(invalid(format!(
                    "Gemma 4 {kind} patch and pooling sizes must be positive"
                )));
            }
            if !matches!(policy.max_soft_tokens, 70 | 140 | 280 | 560 | 1120) {
                return Err(invalid(format!(
                    "Gemma 4 {kind} max_soft_tokens must be one of 70, 140, 280, 560, or 1120, got {}",
                    policy.max_soft_tokens
                )));
            }
        }
        if self.video.as_ref().is_some_and(|(_, frames)| *frames == 0) {
            return Err(invalid(
                "Gemma 4 video processor requires a positive frame count",
            ));
        }
        Ok(())
    }
}

fn optional_framing(
    start: Option<u32>,
    end: Option<u32>,
    family: &str,
) -> Result<Option<MediaFraming>, ProcessorPlanError> {
    match (start, end) {
        (Some(start_token_id), Some(end_token_id)) => Ok(Some(MediaFraming {
            start_token_id,
            end_token_id,
        })),
        (None, None) => Ok(None),
        _ => Err(invalid(format!(
            "{family} framing must declare both opening and closing token IDs"
        ))),
    }
}

fn gemma_visual_plan(
    policy: &Gemma4VisualPolicy,
    height: usize,
    width: usize,
) -> Result<(RgbTransformPlan, usize), ProcessorPlanError> {
    let pool_area = policy
        .pooling_kernel_size
        .checked_mul(policy.pooling_kernel_size)
        .ok_or_else(|| invalid("Gemma 4 pooling area overflow"))?;
    let max_patches = policy
        .max_soft_tokens
        .checked_mul(pool_area)
        .ok_or_else(|| invalid("Gemma 4 patch budget overflow"))?;
    let (height, width) = gemma_aspect_ratio_preserving_size(
        height,
        width,
        policy.patch_size,
        max_patches,
        policy.pooling_kernel_size,
    )?;
    Ok((
        RgbTransformPlan {
            height,
            width,
            resample: RgbResample::Bicubic,
            rescale_factor: 1.0 / 255.0,
            mean: [0.0; 3],
            std: [1.0; 3],
        },
        max_patches,
    ))
}

fn gemma_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), ProcessorPlanError> {
    if height == 0 || width == 0 || patch_size == 0 || pooling_kernel_size == 0 || max_patches == 0
    {
        return Err(invalid(
            "Gemma 4 image processor dimensions must be positive",
        ));
    }
    let target_pixels = max_patches as f64 * (patch_size * patch_size) as f64;
    let factor = (target_pixels / (height * width) as f64).sqrt();
    let side_multiple = patch_size
        .checked_mul(pooling_kernel_size)
        .ok_or_else(|| invalid("Gemma 4 resize multiple overflow"))?;
    let mut target_height =
        ((factor * height as f64).floor() as usize / side_multiple) * side_multiple;
    let mut target_width =
        ((factor * width as f64).floor() as usize / side_multiple) * side_multiple;
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_multiple;
    if target_height == 0 && target_width == 0 {
        return Err(invalid(format!(
            "Gemma 4 image is too small for resize multiple {side_multiple}"
        )));
    }
    if target_height == 0 {
        target_height = side_multiple;
        target_width = (width / height).saturating_mul(side_multiple).min(max_side);
    } else if target_width == 0 {
        target_width = side_multiple;
        target_height = (height / width).saturating_mul(side_multiple).min(max_side);
    }
    if target_height * target_width > max_patches * patch_size * patch_size {
        return Err(invalid(format!(
            "Gemma 4 resize {target_height}x{target_width} exceeds the {max_patches}-patch budget"
        )));
    }
    Ok((target_height, target_width))
}

fn gemma_timestamp(source_index: usize, source_fps: f64, first: bool) -> String {
    let seconds = (source_index as f64 / source_fps).floor() as u64;
    let timestamp = format!("{:02}:{:02}", seconds / 60, seconds % 60);
    if first {
        format!("{timestamp} ")
    } else {
        format!(" {timestamp} ")
    }
}

fn optional_metadata_usize(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<usize>, ProcessorPlanError> {
    optional_metadata_u64(metadata, key)?
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| invalid(format!("processor GGUF metadata key {key:?} exceeds usize")))
        })
        .transpose()
}

fn optional_metadata_u32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<u32>, ProcessorPlanError> {
    optional_metadata_u64(metadata, key)?
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| invalid(format!("processor GGUF metadata key {key:?} must fit u32")))
        })
        .transpose()
}

fn optional_metadata_u64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<u64>, ProcessorPlanError> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let value = match value {
        MetadataValue::Uint8(value) => u64::from(*value),
        MetadataValue::Uint16(value) => u64::from(*value),
        MetadataValue::Uint32(value) => u64::from(*value),
        MetadataValue::Uint64(value) => *value,
        MetadataValue::Int8(value) => u64::try_from(*value).map_err(|_| invalid_metadata(key))?,
        MetadataValue::Int16(value) => u64::try_from(*value).map_err(|_| invalid_metadata(key))?,
        MetadataValue::Int32(value) => u64::try_from(*value).map_err(|_| invalid_metadata(key))?,
        MetadataValue::Int64(value) => u64::try_from(*value).map_err(|_| invalid_metadata(key))?,
        _ => return Err(invalid_metadata(key)),
    };
    Ok(Some(value))
}

fn optional_btree_usize(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<usize>, ProcessorPlanError> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let value = value
        .to_u32_vec()
        .and_then(|values| {
            values
                .as_slice()
                .first()
                .copied()
                .filter(|_| values.len() == 1)
        })
        .ok_or_else(|| invalid_metadata(key))?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| invalid(format!("processor GGUF metadata key {key:?} exceeds usize")))
}

fn required_btree_usize(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<usize, ProcessorPlanError> {
    optional_btree_usize(metadata, key)?.ok_or_else(|| {
        invalid(format!(
            "Qwen projector is missing GGUF metadata key {key:?}"
        ))
    })
}

fn required_btree_rgb(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<[f32; 3], ProcessorPlanError> {
    let values = metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .and_then(MetadataArray::to_f32_vec)
        .ok_or_else(|| {
            invalid(format!(
                "Qwen projector is missing float RGB metadata {key:?}"
            ))
        })?;
    values.try_into().map_err(|values: Vec<f32>| {
        invalid(format!(
            "Qwen projector metadata {key:?} must contain 3 floats, got {}",
            values.len()
        ))
    })
}

fn invalid_metadata(key: &str) -> ProcessorPlanError {
    invalid(format!(
        "processor GGUF metadata key {key:?} must be a non-negative integer"
    ))
}

fn validate_fps(value: f64, field: &str) -> Result<(), ProcessorPlanError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!(
            "{field} must be finite and positive, got {value}"
        )));
    }
    Ok(())
}

fn uniform_sample_indices(
    total_frames: usize,
    sample_count: usize,
) -> Result<Vec<usize>, ProcessorPlanError> {
    if total_frames == 0 || sample_count == 0 {
        return Err(invalid(format!(
            "video sampling requires positive frame counts, got {total_frames} source and {sample_count} requested"
        )));
    }
    let sample_count = sample_count.min(total_frames);
    if sample_count == 1 {
        return Ok(vec![0]);
    }
    let last = (total_frames - 1) as f64;
    let denominator = (sample_count - 1) as f64;
    Ok((0..sample_count)
        .map(|index| (index as f64 * last / denominator).round_ties_even() as usize)
        .collect())
}

fn sampled_frame_count(
    total_frames: usize,
    source_fps: f64,
    target_fps: f64,
    min_frames: usize,
    max_frames: usize,
) -> Result<usize, ProcessorPlanError> {
    validate_fps(source_fps, "video source FPS")?;
    validate_fps(target_fps, "video target FPS")?;
    if min_frames == 0 || max_frames < min_frames {
        return Err(invalid(format!(
            "video frame limits must be positive and ordered, got {min_frames}..{max_frames}"
        )));
    }
    let requested = (total_frames as f64 / source_fps * target_fps) as usize;
    Ok(requested
        .max(min_frames)
        .min(max_frames)
        .min(total_frames)
        .max(1))
}

fn pad_frame_indices(
    indices: &mut Vec<usize>,
    temporal_factor: usize,
) -> Result<(), ProcessorPlanError> {
    if indices.is_empty() || temporal_factor == 0 {
        return Err(invalid(
            "temporal frame padding requires frames and a positive factor",
        ));
    }
    let remainder = indices.len() % temporal_factor;
    if remainder != 0 {
        let last = *indices.last().expect("indices are non-empty");
        indices.resize(indices.len() + temporal_factor - remainder, last);
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct InklingSource {
    model_type: String,
    #[serde(default = "default_inkling_image_bos")]
    image_bos_token_id: u32,
    #[serde(default = "default_inkling_audio_bos")]
    audio_bos_token_id: u32,
    #[serde(default)]
    audio_config: Option<InklingAudioSource>,
}

#[derive(Debug, Deserialize)]
struct InklingAudioSource {
    #[serde(default = "default_inkling_dmel_bins")]
    mel_vocab_size: usize,
    #[serde(default = "default_inkling_dmel_min")]
    dmel_min_value: f32,
    #[serde(default = "default_inkling_dmel_max")]
    dmel_max_value: f32,
}

const fn default_inkling_image_bos() -> u32 {
    200_005
}

const fn default_inkling_audio_bos() -> u32 {
    200_020
}

const fn default_inkling_dmel_bins() -> usize {
    16
}

fn default_inkling_dmel_min() -> f32 {
    -7.0
}

fn default_inkling_dmel_max() -> f32 {
    2.0
}

/// Inkling image patch and normalization policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InklingImagePlan {
    /// Token placed before the image tensor.
    pub start_token_id: u32,
    /// Spatial patch side length.
    pub patch_size: usize,
    /// Duplicated temporal extent.
    pub temporal_patch_size: usize,
    /// Patch rows, including a final partial row.
    pub patch_rows: usize,
    /// Patch columns, including the released extra-column behavior.
    pub patch_columns: usize,
    /// Pixel rescaling factor.
    pub rescale_factor: f32,
    /// Channel normalization mean.
    pub mean: [f32; 3],
    /// Channel normalization standard deviation.
    pub std: [f32; 3],
    /// Raw byte-domain value used outside the source image.
    pub padding_value: f32,
}

/// Inkling audio feature extraction and dMel quantization policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InklingAudioPlan {
    /// Token placed before the audio tensor.
    pub start_token_id: u32,
    /// Required waveform sample rate.
    pub sample_rate: u32,
    /// FFT and analysis-window length.
    pub fft_length: usize,
    /// Analysis hop length.
    pub hop_length: usize,
    /// Analysis-window function.
    pub window: AudioWindow,
    /// Waveform padding and output-frame-count convention.
    pub framing: AudioFramingPlan,
    /// Number of continuous mel bands.
    pub mel_bins: usize,
    /// Lower mel filter frequency.
    pub min_frequency: f32,
    /// Upper mel filter frequency.
    pub max_frequency: f32,
    /// Frequency-to-mel conversion convention.
    pub mel_scale: MelScale,
    /// Mel filter-bank normalization convention.
    pub mel_normalization: MelNormalization,
    /// Spectrum value accumulated through the mel filters.
    pub spectrum: SpectrumValue,
    /// Logarithm applied after flooring the filtered spectrum.
    pub logarithm: Logarithm,
    /// Number of discrete mel vocabulary bins.
    pub dmel_bins: usize,
    /// Minimum quantized log-mel value.
    pub dmel_min: f32,
    /// Maximum quantized log-mel value.
    pub dmel_max: f32,
    /// Energy floor before base-ten logarithm.
    pub energy_floor: f32,
}

/// Backend-neutral analysis-window function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioWindow {
    /// Periodic Hann: `0.5 - 0.5*cos(2*pi*n/N)` for `0 <= n < N`.
    PeriodicHann,
}

/// Backend-neutral waveform framing and padding convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioFramingPlan {
    /// Zero-valued samples prepended before the first analysis frame.
    pub leading_zeros: usize,
    /// Rule used to derive the number of emitted frames.
    pub frame_count: AudioFrameCount,
    /// Samples beyond the waveform are filled with this value.
    pub trailing_padding_value: f32,
}

/// Backend-neutral output-frame-count convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFrameCount {
    /// Emit `ceil(input_samples / hop_length)` frames.
    InputDivHopCeil,
}

/// Backend-neutral frequency-to-mel conversion convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// Slaney's piecewise linear/logarithmic mel scale.
    Slaney,
}

/// Backend-neutral mel filter-bank normalization convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNormalization {
    /// Slaney area normalization, scaling each triangle by `2 / (right - left)`.
    SlaneyArea,
}

/// Backend-neutral spectrum value accumulated through a filter bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumValue {
    /// Complex magnitude, equivalent to a spectrum power of one.
    Magnitude,
}

/// Backend-neutral logarithm convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logarithm {
    /// Base-ten logarithm.
    Base10,
}

/// Normalized Inkling multimodal processor policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InklingProcessorPlan {
    image_bos_token_id: Option<u32>,
    audio_bos_token_id: Option<u32>,
    dmel_bins: usize,
    dmel_min: f32,
    dmel_max: f32,
}

impl InklingProcessorPlan {
    /// Parses the Inkling family model configuration.
    pub fn from_hf_json(model: &[u8]) -> Result<Option<Self>, ProcessorPlanError> {
        let source: InklingSource = serde_json::from_slice(model)?;
        if source.model_type != "inkling_mm_model" {
            return Ok(None);
        }
        let audio = source.audio_config.unwrap_or(InklingAudioSource {
            mel_vocab_size: default_inkling_dmel_bins(),
            dmel_min_value: default_inkling_dmel_min(),
            dmel_max_value: default_inkling_dmel_max(),
        });
        Self::new(
            source.image_bos_token_id,
            source.audio_bos_token_id,
            audio.mel_vocab_size,
            audio.dmel_min_value,
            audio.dmel_max_value,
        )
        .map(Some)
    }

    /// Builds structurally admitted Inkling GGUF processing awaiting token IDs.
    fn from_gguf_metadata() -> Result<Self, ProcessorPlanError> {
        Self::new_unresolved(
            default_inkling_dmel_bins(),
            default_inkling_dmel_min(),
            default_inkling_dmel_max(),
        )
    }

    /// Builds Inkling GGUF processing from facade-resolved tokenizer IDs.
    pub fn from_gguf_token_ids(
        image_bos_token_id: u32,
        audio_bos_token_id: u32,
    ) -> Result<Self, ProcessorPlanError> {
        Self::new(
            image_bos_token_id,
            audio_bos_token_id,
            default_inkling_dmel_bins(),
            default_inkling_dmel_min(),
            default_inkling_dmel_max(),
        )
    }

    fn new(
        image_bos_token_id: u32,
        audio_bos_token_id: u32,
        dmel_bins: usize,
        dmel_min: f32,
        dmel_max: f32,
    ) -> Result<Self, ProcessorPlanError> {
        if dmel_bins < 2 || !dmel_min.is_finite() || !dmel_max.is_finite() || dmel_max <= dmel_min {
            return Err(invalid("invalid Inkling dMel bin configuration"));
        }
        Ok(Self {
            image_bos_token_id: Some(image_bos_token_id),
            audio_bos_token_id: Some(audio_bos_token_id),
            dmel_bins,
            dmel_min,
            dmel_max,
        })
    }

    fn new_unresolved(
        dmel_bins: usize,
        dmel_min: f32,
        dmel_max: f32,
    ) -> Result<Self, ProcessorPlanError> {
        let mut plan = Self::new(0, 1, dmel_bins, dmel_min, dmel_max)?;
        plan.image_bos_token_id = None;
        plan.audio_bos_token_id = None;
        Ok(plan)
    }

    fn bind_token_ids(&mut self, image_bos_token_id: u32, audio_bos_token_id: u32) {
        self.image_bos_token_id = Some(image_bos_token_id);
        self.audio_bos_token_id = Some(audio_bos_token_id);
    }

    const fn has_token_ids(&self) -> bool {
        self.image_bos_token_id.is_some() && self.audio_bos_token_id.is_some()
    }

    /// Derives the released image grid and normalization policy.
    pub fn image(
        &self,
        height: usize,
        width: usize,
    ) -> Result<InklingImagePlan, ProcessorPlanError> {
        if height == 0 || width == 0 {
            return Err(invalid("Inkling image dimensions must be positive"));
        }
        Ok(InklingImagePlan {
            start_token_id: self
                .image_bos_token_id
                .ok_or_else(|| invalid("Inkling GGUF image framing token ID is unresolved"))?,
            patch_size: 40,
            temporal_patch_size: 2,
            patch_rows: height.div_ceil(40),
            patch_columns: width / 40 + 1,
            rescale_factor: 1.0 / 255.0,
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            padding_value: -1.0,
        })
    }

    /// Returns the released audio framing, analysis, and quantization policy.
    pub fn audio(&self) -> Result<InklingAudioPlan, ProcessorPlanError> {
        Ok(InklingAudioPlan {
            start_token_id: self
                .audio_bos_token_id
                .ok_or_else(|| invalid("Inkling GGUF audio framing token ID is unresolved"))?,
            sample_rate: 16_000,
            fft_length: 1_600,
            hop_length: 800,
            window: AudioWindow::PeriodicHann,
            framing: AudioFramingPlan {
                leading_zeros: 800,
                frame_count: AudioFrameCount::InputDivHopCeil,
                trailing_padding_value: 0.0,
            },
            mel_bins: 80,
            min_frequency: 0.0,
            max_frequency: 8_000.0,
            mel_scale: MelScale::Slaney,
            mel_normalization: MelNormalization::SlaneyArea,
            spectrum: SpectrumValue::Magnitude,
            logarithm: Logarithm::Base10,
            dmel_bins: self.dmel_bins,
            dmel_min: self.dmel_min,
            dmel_max: self.dmel_max,
            energy_floor: 1e-10,
        })
    }
}

fn default_muse_true() -> bool {
    true
}

fn default_muse_rescale() -> f32 {
    1.0 / 255.0
}

fn default_muse_mean_std() -> [f32; 3] {
    [0.5; 3]
}

const fn default_muse_patch() -> usize {
    14
}

const fn default_muse_temporal() -> usize {
    2
}

const fn default_muse_merge() -> usize {
    2
}

const fn default_muse_image_tokens() -> usize {
    4096
}

const fn default_muse_video_tokens() -> usize {
    144
}

const fn default_muse_video_frames() -> usize {
    96
}

fn default_muse_video_fps() -> f64 {
    2.0
}

const fn default_muse_lanczos() -> u8 {
    1
}

#[derive(Debug, Clone, Deserialize)]
struct MuseVisualSource {
    #[serde(default = "default_muse_true")]
    do_resize: bool,
    #[serde(default = "default_muse_true")]
    do_rescale: bool,
    #[serde(default = "default_muse_rescale")]
    rescale_factor: f32,
    #[serde(default = "default_muse_true")]
    do_normalize: bool,
    #[serde(default = "default_muse_mean_std")]
    image_mean: [f32; 3],
    #[serde(default = "default_muse_mean_std")]
    image_std: [f32; 3],
    #[serde(default = "default_muse_patch")]
    patch_size: usize,
    #[serde(default = "default_muse_temporal")]
    temporal_patch_size: usize,
    #[serde(default = "default_muse_merge")]
    merge_size: usize,
    #[serde(default = "default_muse_image_tokens")]
    max_image_tokens: usize,
    #[serde(default = "default_muse_video_tokens")]
    max_video_frame_tokens: usize,
    #[serde(default = "default_muse_video_frames")]
    num_frames: usize,
    #[serde(default = "default_muse_video_fps")]
    fps: f64,
    #[serde(default = "default_muse_true")]
    do_sample_frames: bool,
    #[serde(default = "default_muse_lanczos")]
    resample: u8,
}

#[derive(Debug, Deserialize)]
struct MuseProcessorSource {
    image_processor: MuseVisualSource,
    video_processor: MuseVisualSource,
}

/// Muse-Glimmer patch packing geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusePatchPlan {
    /// Spatial patch side length.
    pub patch_size: usize,
    /// Frames folded into every patch row.
    pub temporal_patch_size: usize,
    /// Spatial merge side length used by the family.
    pub merge_size: usize,
}

/// Complete Muse-Glimmer image processor plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuseImagePlan {
    /// Opening framing text to tokenize.
    pub start_text: &'static str,
    /// Closing framing text to tokenize.
    pub end_text: &'static str,
    /// Backend-executed RGB transform.
    pub transform: RgbTransformPlan,
    /// Patch packing geometry.
    pub patches: MusePatchPlan,
}

/// One timestamped Muse-Glimmer video patch group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseVideoGroupPlan {
    /// Source frame indices, padded to the temporal patch extent.
    pub source_indices: Vec<usize>,
    /// Timestamp framing text to tokenize before the tensor.
    pub timestamp_text: String,
    /// Separator or final framing text to tokenize after the tensor.
    pub boundary_text: &'static str,
}

/// Complete Muse-Glimmer video processor plan.
#[derive(Debug, Clone, PartialEq)]
pub struct MuseVideoPlan {
    /// Opening framing text to tokenize once.
    pub start_text: &'static str,
    /// Backend-executed RGB transform.
    pub transform: RgbTransformPlan,
    /// Patch packing geometry.
    pub patches: MusePatchPlan,
    /// Ordered timestamped temporal groups.
    pub groups: Vec<MuseVideoGroupPlan>,
}

/// Normalized Muse-Glimmer image and video processor policy.
#[derive(Debug, Clone)]
pub struct MuseProcessorPlan {
    image: MuseVisualSource,
    video: MuseVisualSource,
    image_only: bool,
}

impl MuseProcessorPlan {
    /// Parses the release's nested Hugging Face processor configuration.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ProcessorPlanError> {
        let source: MuseProcessorSource = serde_json::from_slice(bytes)?;
        validate_muse(&source.image_processor, "image")?;
        validate_muse(&source.video_processor, "video")?;
        Ok(Self {
            image: source.image_processor,
            video: source.video_processor,
            image_only: false,
        })
    }

    /// Parses the official image-only projector policy from portable GGUF metadata.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ProcessorPlanError> {
        let image = MuseVisualSource {
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_muse_rescale(),
            do_normalize: true,
            image_mean: metadata_rgb(metadata, "clip.vision.image_mean")?,
            image_std: metadata_rgb(metadata, "clip.vision.image_std")?,
            patch_size: required_metadata_usize(
                metadata,
                "clip.vision.patch_size",
                "Muse-Glimmer",
            )?,
            temporal_patch_size: 1,
            merge_size: required_metadata_usize(
                metadata,
                "clip.vision.spatial_merge_size",
                "Muse-Glimmer",
            )?,
            max_image_tokens: default_muse_image_tokens(),
            max_video_frame_tokens: default_muse_video_tokens(),
            num_frames: default_muse_video_frames(),
            fps: default_muse_video_fps(),
            do_sample_frames: true,
            resample: default_muse_lanczos(),
        };
        validate_muse(&image, "image")?;
        Ok(Self {
            video: image.clone(),
            image,
            image_only: true,
        })
    }

    /// Derives image framing, transform, and patch policy.
    pub fn image(&self, height: usize, width: usize) -> Result<MuseImagePlan, ProcessorPlanError> {
        let transform = muse_transform(&self.image, height, width, self.image.max_image_tokens)?;
        Ok(MuseImagePlan {
            start_text: "<|image_start|>",
            end_text: "<|image_end|>",
            transform,
            patches: muse_patches(&self.image),
        })
    }

    /// Derives video sampling, framing, transform, and patch policy.
    pub fn video(
        &self,
        total_frames: usize,
        height: usize,
        width: usize,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Result<MuseVideoPlan, ProcessorPlanError> {
        if self.image_only {
            return Err(invalid(
                "the official Muse-Glimmer GGUF projector is image-only because its temporal patch weights are collapsed",
            ));
        }
        let source_fps = source_fps.unwrap_or(24.0);
        validate_fps(source_fps, "Muse-Glimmer video source FPS")?;
        let requested = match sampling {
            VideoSampling::ProcessorDefault if self.video.do_sample_frames => {
                ((total_frames as f64 * self.video.fps / source_fps) as usize)
                    .min(self.video.num_frames)
                    .min(total_frames)
            }
            VideoSampling::ProcessorDefault | VideoSampling::All => total_frames,
            VideoSampling::Fps(fps) => {
                validate_fps(fps, "Muse-Glimmer target FPS")?;
                ((total_frames as f64 * fps / source_fps) as usize)
                    .min(self.video.num_frames)
                    .min(total_frames)
            }
            VideoSampling::FrameCount(count) => count.min(total_frames),
        };
        let requested = requested.max(self.video.temporal_patch_size)
            / self.video.temporal_patch_size
            * self.video.temporal_patch_size;
        let mut indices = uniform_sample_indices(total_frames, requested.min(total_frames).max(1))?;
        let unpadded = indices.clone();
        pad_frame_indices(&mut indices, self.video.temporal_patch_size)?;
        let group_count = indices.len() / self.video.temporal_patch_size;
        let groups = indices
            .chunks_exact(self.video.temporal_patch_size)
            .enumerate()
            .map(|(group, chunk)| {
                let source_index = unpadded
                    .get(group * self.video.temporal_patch_size)
                    .copied()
                    .or_else(|| unpadded.last().copied())
                    .unwrap_or(0);
                MuseVideoGroupPlan {
                    source_indices: chunk.to_vec(),
                    timestamp_text: format!("Time: {:.1}s", source_index as f64 / source_fps),
                    boundary_text: if group + 1 == group_count {
                        "<|vid_end|>"
                    } else {
                        "<|vid_frame_separator|>"
                    },
                }
            })
            .collect();
        Ok(MuseVideoPlan {
            start_text: "<|vid_start|>",
            transform: muse_transform(
                &self.video,
                height,
                width,
                self.video.max_video_frame_tokens,
            )?,
            patches: muse_patches(&self.video),
            groups,
        })
    }
}

fn validate_muse(source: &MuseVisualSource, kind: &str) -> Result<(), ProcessorPlanError> {
    if source.patch_size == 0
        || source.temporal_patch_size == 0
        || source.merge_size == 0
        || source.max_image_tokens == 0
        || source.max_video_frame_tokens == 0
        || source.num_frames == 0
        || !source.fps.is_finite()
        || source.fps <= 0.0
    {
        return Err(invalid(format!(
            "Muse-Glimmer {kind} processor dimensions, rates, and token limits must be positive"
        )));
    }
    if source.resample != default_muse_lanczos() {
        return Err(invalid(format!(
            "Muse-Glimmer {kind} processor requires Lanczos resample mode 1, got {}",
            source.resample
        )));
    }
    Ok(())
}

fn muse_patches(source: &MuseVisualSource) -> MusePatchPlan {
    MusePatchPlan {
        patch_size: source.patch_size,
        temporal_patch_size: source.temporal_patch_size,
        merge_size: source.merge_size,
    }
}

fn muse_transform(
    source: &MuseVisualSource,
    height: usize,
    width: usize,
    max_tokens: usize,
) -> Result<RgbTransformPlan, ProcessorPlanError> {
    let patch_multiple = source
        .patch_size
        .checked_mul(source.merge_size)
        .ok_or_else(|| invalid("Muse-Glimmer resize multiple overflow"))?;
    let (target_height, target_width) =
        muse_smart_resize(height, width, patch_multiple, max_tokens)?;
    Ok(RgbTransformPlan {
        height: if source.do_resize {
            target_height
        } else {
            height
        },
        width: if source.do_resize {
            target_width
        } else {
            width
        },
        resample: RgbResample::Lanczos3,
        rescale_factor: if source.do_rescale {
            source.rescale_factor
        } else {
            1.0
        },
        mean: if source.do_normalize {
            source.image_mean
        } else {
            [0.0; 3]
        },
        std: if source.do_normalize {
            source.image_std
        } else {
            [1.0; 3]
        },
    })
}

fn muse_smart_resize(
    height: usize,
    width: usize,
    patch_size: usize,
    max_tokens: usize,
) -> Result<(usize, usize), ProcessorPlanError> {
    if height == 0 || width == 0 || patch_size == 0 || max_tokens == 0 {
        return Err(invalid(
            "Muse-Glimmer smart resize requires positive dimensions",
        ));
    }
    let mut ideal_h = height as f64 / patch_size as f64;
    let mut ideal_w = width as f64 / patch_size as f64;
    let ratio = ideal_w / ideal_h;
    if ideal_h * ideal_w > max_tokens as f64 {
        ideal_h = (max_tokens as f64 / ratio).sqrt();
        ideal_w = ideal_h * ratio;
    }
    let mut candidates = Vec::new();
    for h in [ideal_h.floor() as usize, ideal_h.ceil() as usize] {
        for w in [ideal_w.floor() as usize, ideal_w.ceil() as usize] {
            if h > 0 && w > 0 && h.saturating_mul(w) <= max_tokens && !candidates.contains(&(h, w))
            {
                candidates.push((h, w));
            }
        }
    }
    if candidates.is_empty() {
        candidates.push((
            ideal_h.round().max(1.0) as usize,
            ideal_w.round().max(1.0) as usize,
        ));
    }
    let source_ratio = height as f64 / width as f64;
    let (grid_h, grid_w) = candidates
        .into_iter()
        .min_by(|left, right| {
            let left_error = (left.0 as f64 / left.1 as f64 - source_ratio).abs();
            let right_error = (right.0 as f64 / right.1 as f64 - source_ratio).abs();
            left_error.total_cmp(&right_error)
        })
        .expect("candidate list is non-empty");
    Ok((grid_h * patch_size, grid_w * patch_size))
}

fn required_metadata_usize(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
    family: &str,
) -> Result<usize, ProcessorPlanError> {
    optional_metadata_usize(metadata, key)?
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            invalid(format!(
                "{family} projector GGUF requires positive integer metadata {key:?}"
            ))
        })
}

fn metadata_rgb(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<[f32; 3], ProcessorPlanError> {
    let values = match metadata.get(key) {
        Some(MetadataValue::Array(MetadataArray::Float32(values))) => values,
        _ => {
            return Err(invalid(format!(
                "Muse-Glimmer projector GGUF requires three Float32 values in {key:?}"
            )))
        }
    };
    values.as_slice().try_into().map_err(|_| {
        invalid(format!(
            "Muse-Glimmer projector GGUF requires three Float32 values in {key:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactArchitecturePlan, AudioFrameCount, AudioWindow, Gemma4ProcessorPlan,
        GgufSpecialTokenKind, InklingProcessorPlan, Logarithm, MediaFraming, MelNormalization,
        MelScale, MuseProcessorPlan, QwenProcessorPlan, SpectrumValue,
    };
    use crate::{GgufArchitecture, ModelKind};
    use eredu_core::VideoSampling;
    use eredu_gguf::{MetadataArray, MetadataValue};
    use std::collections::BTreeMap;

    fn gguf_artifact_plan(architecture: GgufArchitecture) -> ArtifactArchitecturePlan {
        let args = crate::llama::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        }))
        .unwrap();
        let checkpoint = crate::llama::gguf_plan(&args).unwrap();
        ArtifactArchitecturePlan::from_gguf_architecture(
            crate::configuration::GgufArchitecturePlan::new(
                architecture,
                crate::configuration::GgufModelConfig::Llama(args),
                checkpoint,
                Vec::new(),
            ),
        )
    }

    fn qwen_visual() -> Vec<u8> {
        br#"{
            "size":{"shortest_edge":16,"longest_edge":16},
            "patch_size":2,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
            "min_frames":1,"max_frames":8
        }"#
        .to_vec()
    }

    #[test]
    fn artifact_plan_retains_validated_safetensors_plan_and_exact_gguf_identity() {
        let architecture = crate::configuration::resolve_model_config(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        }))
        .unwrap()
        .architecture;
        let plan = ArtifactArchitecturePlan::from_safetensors_architecture(architecture);
        assert_eq!(plan.model_kind(), ModelKind::Llama);
        assert!(plan.safetensors_architecture().is_some());
        assert_eq!(plan.gguf_architecture(), None);

        for architecture in [
            GgufArchitecture::Llama,
            GgufArchitecture::Mistral,
            GgufArchitecture::DeepSeek4,
            GgufArchitecture::Gemma4,
            GgufArchitecture::MuseGlimmer,
            GgufArchitecture::Qwen35Moe,
        ] {
            let plan = gguf_artifact_plan(architecture)
                .with_gguf_processors(&BTreeMap::new(), None)
                .unwrap();
            assert_eq!(plan.model_kind(), architecture.model_kind());
            assert_eq!(plan.gguf_architecture(), Some(architecture));
        }
    }

    #[test]
    fn artifact_plan_retains_every_normalized_processor_variant() {
        let gemma_model = br#"{
            "boi_token_id":43,"eoi_token_id":44,
            "vision_config":{"patch_size":2,"pooling_kernel_size":1}
        }"#;
        assert!(Gemma4ProcessorPlan::from_hf_json(gemma_model, None, None)
            .unwrap()
            .is_some());

        assert!(
            InklingProcessorPlan::from_hf_json(br#"{"model_type":"inkling_mm_model"}"#)
                .unwrap()
                .is_some()
        );

        assert!(
            MuseProcessorPlan::from_hf_json(br#"{"image_processor":{},"video_processor":{}}"#)
                .is_ok()
        );

        let visual = qwen_visual();
        let plan = QwenProcessorPlan::from_hf_json(
            br#"{"vision_start_token_id":44,"vision_end_token_id":45}"#,
            Some(&visual),
            None,
        )
        .unwrap();
        assert!(plan.is_some());

        let gemma_model = BTreeMap::from([
            ("gemma4.boi_token_id".into(), MetadataValue::Uint32(43)),
            ("gemma4.eoi_token_id".into(), MetadataValue::Uint32(44)),
        ]);
        let gemma_projector =
            BTreeMap::from([("clip.vision.patch_size".into(), MetadataValue::Uint32(2))]);
        assert!(gguf_artifact_plan(GgufArchitecture::Gemma4)
            .with_gguf_processors(&gemma_model, Some(&gemma_projector))
            .unwrap()
            .gemma4()
            .is_some());
        let inkling = gguf_artifact_plan(GgufArchitecture::Inkling)
            .with_gguf_processors(&BTreeMap::new(), Some(&BTreeMap::new()))
            .unwrap();
        assert_eq!(
            inkling.required_gguf_special_tokens(),
            Some(GgufSpecialTokenKind::Inkling)
        );
        assert!(inkling.inkling().is_none());

        let muse_projector = BTreeMap::from([
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.spatial_merge_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.image_mean".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![0.0; 3])),
            ),
            (
                "clip.vision.image_std".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![1.0; 3])),
            ),
        ]);
        assert!(gguf_artifact_plan(GgufArchitecture::MuseGlimmer)
            .with_gguf_processors(&BTreeMap::new(), Some(&muse_projector))
            .unwrap()
            .muse()
            .is_some());

        let qwen = gguf_artifact_plan(GgufArchitecture::Qwen3Vl)
            .with_gguf_processors(&BTreeMap::new(), Some(&muse_projector))
            .unwrap();
        assert_eq!(
            qwen.required_gguf_special_tokens(),
            Some(GgufSpecialTokenKind::Qwen)
        );
        assert!(qwen.qwen().is_none());
    }

    #[test]
    fn qwen_plan_owns_resize_sampling_and_framing() {
        let model = br#"{"vision_start_token_id":44,"vision_end_token_id":45}"#;
        let visual = qwen_visual();
        let plan = QwenProcessorPlan::from_hf_json(model, Some(&visual), Some(&visual))
            .unwrap()
            .unwrap();
        let image = plan.image(4, 4).unwrap();
        assert_eq!((image.transform.height, image.transform.width), (4, 4));
        assert_eq!(image.framing.start_token_id, 44);
        let video = plan.video(4, 4, 4, Some(2.0), VideoSampling::All).unwrap();
        assert_eq!(video.groups.len(), 2);
        assert_eq!(video.groups[0].timestamp_text, "<0.2 seconds>");
        assert_eq!(video.groups[1].timestamp_text, "<1.2 seconds>");
    }

    #[test]
    fn qwen_gguf_plan_binds_typed_framing_to_projector_policy() {
        let mut projector = BTreeMap::new();
        projector.insert("clip.vision.patch_size".into(), MetadataValue::Uint32(2));
        projector.insert(
            "clip.vision.spatial_merge_size".into(),
            MetadataValue::Uint32(2),
        );
        projector.insert(
            "clip.vision.image_min_pixels".into(),
            MetadataValue::Uint32(16),
        );
        projector.insert(
            "clip.vision.image_max_pixels".into(),
            MetadataValue::Uint32(64),
        );
        projector.insert(
            "clip.vision.image_mean".into(),
            MetadataValue::Array(MetadataArray::Float32(vec![0.1, 0.2, 0.3])),
        );
        projector.insert(
            "clip.vision.image_std".into(),
            MetadataValue::Array(MetadataArray::Float32(vec![0.4, 0.5, 0.6])),
        );

        let mut plan = QwenProcessorPlan::from_gguf_metadata(&projector).unwrap();
        assert!(plan.image(8, 8).is_err());
        plan.bind_framing(MediaFraming {
            start_token_id: 1,
            end_token_id: 2,
        });
        let image = plan.image(8, 8).unwrap();
        assert_eq!(image.framing.start_token_id, 1);
        assert_eq!(image.framing.end_token_id, 2);
        assert_eq!((image.transform.height, image.transform.width), (8, 8));
        assert_eq!(image.transform.mean, [0.1, 0.2, 0.3]);
        assert_eq!(image.patches.patch_size, 2);
        assert_eq!(image.patches.merge_size, 2);
    }

    #[test]
    fn gemma_plan_owns_defaults_resize_and_timestamp_framing() {
        let model = br#"{
            "boi_token_id":43,"eoi_token_id":44,
            "vision_soft_tokens_per_image":280,
            "vision_config":{"patch_size":16,"pooling_kernel_size":3}
        }"#;
        let plan = Gemma4ProcessorPlan::from_hf_json(model, None, None)
            .unwrap()
            .unwrap();
        let image = plan.image(320, 480).unwrap();
        assert_eq!((image.transform.height, image.transform.width), (624, 960));
        assert_eq!(image.max_patches, 2520);
        let video = plan
            .video(2, 320, 480, Some(1.0), VideoSampling::ProcessorDefault)
            .unwrap();
        assert_eq!(video.frames[0].timestamp_text, "00:00 ");
        assert_eq!(video.frames[1].timestamp_text, " 00:01 ");
    }

    #[test]
    fn inkling_plan_owns_markers_patch_grid_and_dmel_policy() {
        let plan = InklingProcessorPlan::from_hf_json(
            br#"{
                "model_type":"inkling_mm_model",
                "image_bos_token_id":7,"audio_bos_token_id":8,
                "audio_config":{"mel_vocab_size":32,"dmel_min_value":-8.0,"dmel_max_value":3.0}
            }"#,
        )
        .unwrap()
        .unwrap();
        let image = plan.image(40, 40).unwrap();
        assert_eq!(image.start_token_id, 7);
        assert_eq!((image.patch_rows, image.patch_columns), (1, 2));
        let audio = plan.audio().unwrap();
        assert_eq!(audio.start_token_id, 8);
        assert_eq!((audio.mel_bins, audio.dmel_bins), (80, 32));
        assert_eq!(audio.window, AudioWindow::PeriodicHann);
        assert_eq!(audio.framing.leading_zeros, 800);
        assert_eq!(audio.framing.frame_count, AudioFrameCount::InputDivHopCeil);
        assert_eq!(audio.framing.trailing_padding_value, 0.0);
        assert_eq!((audio.min_frequency, audio.max_frequency), (0.0, 8_000.0));
        assert_eq!(audio.mel_scale, MelScale::Slaney);
        assert_eq!(audio.mel_normalization, MelNormalization::SlaneyArea);
        assert_eq!(audio.spectrum, SpectrumValue::Magnitude);
        assert_eq!(audio.logarithm, Logarithm::Base10);
    }

    #[test]
    fn muse_plan_owns_resize_video_groups_and_text_framing() {
        let plan =
            MuseProcessorPlan::from_hf_json(br#"{"image_processor":{},"video_processor":{}}"#)
                .unwrap();
        for (height, width) in [(1200, 800), (800, 1200), (28, 4096), (4096, 28)] {
            let image = plan.image(height, width).unwrap();
            assert_eq!(image.transform.height % 28, 0);
            assert_eq!(image.transform.width % 28, 0);
            assert!(image.transform.height / 28 * (image.transform.width / 28) <= 4096);
        }
        let video = plan
            .video(4, 800, 1200, Some(2.0), VideoSampling::All)
            .unwrap();
        assert_eq!(video.start_text, "<|vid_start|>");
        assert_eq!(video.groups.len(), 2);
        assert_eq!(video.groups[0].timestamp_text, "Time: 0.0s");
        assert_eq!(video.groups[0].boundary_text, "<|vid_frame_separator|>");
        assert_eq!(video.groups[1].boundary_text, "<|vid_end|>");
    }
}
