//! Architecture-owned execution of retained media processor plans.

use eredu_core::{
    checkpoint::TensorDtype, InputExtent, InputMetadataKey, InputModality, InputTensorIdentity,
    Media, PreparedInputError, RgbImage, TokenizedMultimodalRequest, TokenizedMultimodalSegment,
};
use eredu_media::audio::{AudioWaveform, LeadingSlaneyLogMelConfig, LogMelConfig};
use eredu_media::{audio::LogMelFeatures, image::NormalizedImage};
use eredu_runtime::{
    observe_and_intervene, ActivationObserver, PreparedInputInspector, PreparedInputPart,
    PreparedInputPayload, PreparedModelInput,
};

use crate::processor_plan::RgbResample;
use crate::processor_plan::{
    ArtifactArchitecturePlan, Gemma4AudioPlan, Gemma4ImagePlan, Gemma4ProcessorPlan,
    Gemma4VideoPlan, InklingAudioPlan, InklingImagePlan, InklingProcessorPlan, MusePatchPlan,
    MuseProcessorPlan, MuseVideoPlan, ProcessorPlanError, QwenImagePlan, QwenPatchPlan,
    QwenProcessorPlan, QwenVideoPlan, RgbTransformPlan,
};

/// Fully specified model-independent log-mel operation.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioFeatureRequest {
    /// Centered periodic-Hann magnitude features with HTK filters and natural logarithm.
    SemicausalHtk {
        /// Required sampling rate.
        sample_rate: u32,
        /// Analysis window length.
        frame_length: usize,
        /// Frame stride.
        hop_length: usize,
        /// Transform length.
        fft_length: usize,
        /// Mel filter count.
        mel_bins: usize,
        /// Lowest filter frequency.
        min_frequency: f32,
        /// Highest filter frequency.
        max_frequency: f32,
        /// Additive floor before natural logarithm.
        mel_floor: f32,
        /// Maximum retained input samples.
        max_samples: usize,
        /// Input padding multiple.
        pad_to_multiple: usize,
    },
    /// Leading-zero periodic-Hann magnitude features with area-normalized Slaney filters.
    LeadingSlaney {
        /// Required sampling rate.
        sample_rate: u32,
        /// Transform and window length.
        fft_length: usize,
        /// Frame stride.
        hop_length: usize,
        /// Leading zero count.
        leading_zeros: usize,
        /// Mel filter count.
        mel_bins: usize,
        /// Lowest filter frequency.
        min_frequency: f32,
        /// Highest filter frequency.
        max_frequency: f32,
        /// Energy floor before base-ten logarithm.
        energy_floor: f32,
    },
}

/// Optional primitive failure distinguishing capability denial from execution failure.
#[derive(Debug, thiserror::Error)]
pub enum OptionalProcessorMechanism<E: std::fmt::Display> {
    /// The backend did not implement this additive primitive.
    #[error("processor mechanism {0} is unavailable")]
    Unavailable(&'static str),
    /// The implemented primitive failed.
    #[error("processor mechanism failed: {0}")]
    Backend(E),
}

/// Backend-native tensor mechanisms used after portable processing succeeds.
pub trait ProcessorMechanisms: PreparedInputInspector<Self::Tensor> {
    /// Native tensor handle.
    type Tensor;
    /// Mechanism failure.
    type Error: std::fmt::Display;

    /// Constructs a native unsigned token tensor.
    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native floating-point tensor.
    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native signed metadata tensor.
    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Constructs a native Boolean metadata tensor when implemented.
    fn tensor_bool(
        &mut self,
        _values: &[bool],
        _shape: &[usize],
    ) -> Result<Self::Tensor, OptionalProcessorMechanism<Self::Error>> {
        Err(OptionalProcessorMechanism::Unavailable("Boolean tensor"))
    }
}

trait ProcessorOperations: PreparedInputInspector<Self::Tensor> {
    type Tensor;
    type Error: std::fmt::Display;

    fn normalize_rgb(
        &mut self,
        image: &RgbImage,
        plan: RgbTransformPlan,
    ) -> Result<NormalizedImage, Self::Error>;
    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn audio_features(
        &mut self,
        audio: &eredu_core::Audio,
        request: &AudioFeatureRequest,
    ) -> Result<LogMelFeatures, OptionalProcessorMechanism<Self::Error>>;
    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<Self::Tensor, OptionalProcessorMechanism<Self::Error>>;
}

#[derive(Debug, Clone, PartialEq)]
enum HostTensor {
    U32 {
        values: Vec<u32>,
        shape: Vec<usize>,
    },
    F32 {
        values: Vec<f32>,
        shape: Vec<usize>,
    },
    I32 {
        values: Vec<i32>,
        shape: Vec<usize>,
    },
    Bool {
        values: Vec<bool>,
        shape: Vec<usize>,
    },
}

impl HostTensor {
    fn identity(&self) -> Result<InputTensorIdentity, PreparedInputError> {
        let (dtype, shape) = match self {
            Self::U32 { shape, .. } => (TensorDtype::U32, shape),
            Self::F32 { shape, .. } => (TensorDtype::F32, shape),
            Self::I32 { shape, .. } => (TensorDtype::I32, shape),
            Self::Bool { shape, .. } => (TensorDtype::Bool, shape),
        };
        InputTensorIdentity::new(dtype, shape.clone())
    }
}

#[derive(Default)]
struct HostProcessorMechanisms;

impl PreparedInputInspector<HostTensor> for HostProcessorMechanisms {
    fn identity(&self, tensor: &HostTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        tensor.identity()
    }

    fn i32_values(&self, tensor: &HostTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        match tensor {
            HostTensor::I32 { values, .. } => Ok(values.clone()),
            _ => Err(eredu_core::CapabilityError::Observation(
                "processor metadata is not an i32 buffer".into(),
            )),
        }
    }

    fn bool_values(&self, tensor: &HostTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        match tensor {
            HostTensor::Bool { values, .. } => Ok(values.clone()),
            _ => Err(eredu_core::CapabilityError::Observation(
                "processor metadata is not a Boolean buffer".into(),
            )),
        }
    }
}

fn host_tensor_len(shape: &[usize]) -> Result<usize, eredu_media::MediaError> {
    shape.iter().try_fold(1usize, |size, dimension| {
        size.checked_mul(*dimension).ok_or_else(|| {
            eredu_media::MediaError::Invalid("processor buffer geometry overflowed".into())
        })
    })
}

impl ProcessorOperations for HostProcessorMechanisms {
    type Tensor = HostTensor;
    type Error = eredu_media::MediaError;

    fn normalize_rgb(
        &mut self,
        image: &RgbImage,
        plan: RgbTransformPlan,
    ) -> Result<NormalizedImage, Self::Error> {
        use eredu_media::image::{
            rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3, RgbImageView,
        };
        let source = RgbImageView::packed(image.pixels(), image.width(), image.height())?;
        let width = u32::try_from(plan.width)
            .map_err(|_| eredu_media::MediaError::Invalid("RGB target width exceeds u32".into()))?;
        let height = u32::try_from(plan.height).map_err(|_| {
            eredu_media::MediaError::Invalid("RGB target height exceeds u32".into())
        })?;
        let resized = match plan.resample {
            RgbResample::Bicubic => resize_rgb8_bicubic(source, width, height)?,
            RgbResample::Lanczos3 => resize_rgb8_lanczos3(source, width, height)?,
        };
        rescale_and_normalize_rgb8(resized.as_view(), plan.rescale_factor, plan.mean, plan.std)
    }

    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<HostTensor, Self::Error> {
        if host_tensor_len(shape)? != values.len() {
            return Err(eredu_media::MediaError::Invalid(
                "u32 processor buffer does not match its shape".into(),
            ));
        }
        Ok(HostTensor::U32 {
            values: values.to_vec(),
            shape: shape.to_vec(),
        })
    }

    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<HostTensor, Self::Error> {
        if host_tensor_len(shape)? != values.len() || values.iter().any(|value| !value.is_finite())
        {
            return Err(eredu_media::MediaError::Invalid(
                "f32 processor buffer does not match its finite shape".into(),
            ));
        }
        Ok(HostTensor::F32 {
            values: values.to_vec(),
            shape: shape.to_vec(),
        })
    }

    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<HostTensor, Self::Error> {
        if host_tensor_len(shape)? != values.len() {
            return Err(eredu_media::MediaError::Invalid(
                "i32 processor buffer does not match its shape".into(),
            ));
        }
        Ok(HostTensor::I32 {
            values: values.to_vec(),
            shape: shape.to_vec(),
        })
    }

    fn audio_features(
        &mut self,
        audio: &eredu_core::Audio,
        request: &AudioFeatureRequest,
    ) -> Result<LogMelFeatures, OptionalProcessorMechanism<Self::Error>> {
        use eredu_media::audio::{extract_leading_slaney_log_mel, extract_log_mel};
        let waveform = AudioWaveform::new(audio.samples(), audio.sample_rate())
            .map_err(OptionalProcessorMechanism::Backend)?;
        match request {
            AudioFeatureRequest::SemicausalHtk {
                sample_rate,
                frame_length,
                hop_length,
                fft_length,
                mel_bins,
                min_frequency,
                max_frequency,
                mel_floor,
                max_samples,
                pad_to_multiple,
            } => extract_log_mel(
                waveform,
                &LogMelConfig {
                    sample_rate: *sample_rate,
                    frame_length: *frame_length,
                    hop_length: *hop_length,
                    fft_length: *fft_length,
                    mel_bins: *mel_bins,
                    min_frequency: *min_frequency,
                    max_frequency: *max_frequency,
                    mel_floor: *mel_floor,
                    max_samples: *max_samples,
                    pad_to_multiple: *pad_to_multiple,
                },
            ),
            AudioFeatureRequest::LeadingSlaney {
                sample_rate,
                fft_length,
                hop_length,
                leading_zeros,
                mel_bins,
                min_frequency,
                max_frequency,
                energy_floor,
            } => extract_leading_slaney_log_mel(
                waveform,
                &LeadingSlaneyLogMelConfig {
                    sample_rate: *sample_rate,
                    fft_length: *fft_length,
                    hop_length: *hop_length,
                    leading_zeros: *leading_zeros,
                    mel_bins: *mel_bins,
                    min_frequency: *min_frequency,
                    max_frequency: *max_frequency,
                    energy_floor: *energy_floor,
                },
            ),
        }
        .map_err(OptionalProcessorMechanism::Backend)
    }

    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<HostTensor, OptionalProcessorMechanism<Self::Error>> {
        if host_tensor_len(shape).map_err(OptionalProcessorMechanism::Backend)? != values.len() {
            return Err(OptionalProcessorMechanism::Backend(
                eredu_media::MediaError::Invalid(
                    "Boolean processor buffer does not match its shape".into(),
                ),
            ));
        }
        Ok(HostTensor::Bool {
            values: values.to_vec(),
            shape: shape.to_vec(),
        })
    }
}

/// Architecture processor selected from the retained admitted sidecar snapshot.
#[derive(Debug, Clone)]
pub struct PreparedProcessor {
    kind: ProcessorKind,
}

#[derive(Debug, Clone)]
enum ProcessorKind {
    Gemma4(Gemma4ProcessorPlan),
    Inkling(InklingProcessorPlan),
    Muse(MuseProcessorPlan),
    Qwen(QwenProcessorPlan),
}

impl PreparedProcessor {
    /// Selects architecture-owned processor semantics without backend inspection.
    pub fn from_artifact(plan: &ArtifactArchitecturePlan) -> Option<Self> {
        if let Some(plan) = plan.qwen().cloned() {
            return Some(Self {
                kind: ProcessorKind::Qwen(plan),
            });
        }
        if let Some(plan) = plan.muse().cloned() {
            return Some(Self {
                kind: ProcessorKind::Muse(plan),
            });
        }
        if let Some(plan) = plan.gemma4().cloned() {
            return Some(Self {
                kind: ProcessorKind::Gemma4(plan),
            });
        }
        plan.inkling().cloned().map(|plan| Self {
            kind: ProcessorKind::Inkling(plan),
        })
    }

    /// Executes the retained processor plan into one identity-coupled prepared input.
    pub fn prepare<M, E>(
        &self,
        request: &TokenizedMultimodalRequest,
        mechanisms: &mut M,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
    where
        M: ProcessorMechanisms,
        E: std::fmt::Display,
    {
        let mut host_mechanisms = HostProcessorMechanisms;
        let host = match &self.kind {
            ProcessorKind::Gemma4(plan) => {
                prepare_gemma4(plan, request, &mut host_mechanisms, encode_text)
            }
            ProcessorKind::Inkling(plan) => prepare_inkling(plan, request, &mut host_mechanisms),
            ProcessorKind::Muse(plan) => {
                prepare_muse(plan, request, &mut host_mechanisms, encode_text)
            }
            ProcessorKind::Qwen(plan) => {
                prepare_qwen(plan, request, &mut host_mechanisms, encode_text)
            }
        }
        .map_err(host_execution_error)?;
        lower_host_input(host, mechanisms)
    }

    /// Executes processor semantics and exposes every final payload and metadata tensor before
    /// the identity-coupled prepared input enters architecture admission.
    pub fn prepare_with_observer<M, E, O>(
        &self,
        request: &TokenizedMultimodalRequest,
        mechanisms: &mut M,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        observer: &mut O,
    ) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
    where
        M: ProcessorMechanisms,
        M::Tensor: Clone,
        E: std::fmt::Display,
        O: ActivationObserver<M::Tensor, ProcessorExecutionError<E, M::Error>> + ?Sized,
    {
        let prepared = self.prepare(request, mechanisms, encode_text)?;
        let mut parts = Vec::with_capacity(prepared.len());
        for (index, part) in prepared.parts().iter().enumerate() {
            let path = format!(
                "{}.{}",
                eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH,
                index
            );
            let payload_value = observe_and_intervene(observer, &path, part.payload().value())?;
            let payload = match part.payload() {
                PreparedInputPayload::TokenIds(_) => PreparedInputPayload::TokenIds(payload_value),
                PreparedInputPayload::Tensor(_) => PreparedInputPayload::Tensor(payload_value),
                PreparedInputPayload::Embeddings(_) => {
                    PreparedInputPayload::Embeddings(payload_value)
                }
                _ => {
                    return Err(ProcessorExecutionError::Plan(
                        "processor emitted an unsupported prepared payload kind".into(),
                    ));
                }
            };
            let metadata = part
                .metadata()
                .iter()
                .map(|(key, value)| {
                    let path = format!(
                        "{}.{}.metadata.{key:?}",
                        eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH,
                        index
                    );
                    observe_and_intervene(observer, &path, value).map(|value| (*key, value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            parts.push(
                PreparedInputPart::new_with_extents(
                    part.modality(),
                    payload,
                    metadata,
                    part.extents().iter().copied(),
                )
                .map_err(ProcessorExecutionError::Prepared)?,
            );
        }
        PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
            .map_err(ProcessorExecutionError::Prepared)
    }
}

fn host_execution_error<E: std::fmt::Display, M: std::fmt::Display>(
    error: ProcessorExecutionError<E, eredu_media::MediaError>,
) -> ProcessorExecutionError<E, M> {
    match error {
        ProcessorExecutionError::Plan(error) => ProcessorExecutionError::Plan(error),
        ProcessorExecutionError::Text(error) => ProcessorExecutionError::Text(error),
        ProcessorExecutionError::Mechanism(error) => {
            ProcessorExecutionError::Plan(error.to_string())
        }
        ProcessorExecutionError::Prepared(error) => ProcessorExecutionError::Prepared(error),
    }
}

fn lower_host_tensor<M, E>(
    tensor: &HostTensor,
    mechanisms: &mut M,
) -> Result<M::Tensor, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms,
    E: std::fmt::Display,
{
    match tensor {
        HostTensor::U32 { values, shape } => mechanisms
            .tensor_u32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::F32 { values, shape } => mechanisms
            .tensor_f32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::I32 { values, shape } => mechanisms
            .tensor_i32(values, shape)
            .map_err(ProcessorExecutionError::Mechanism),
        HostTensor::Bool { values, shape } => mechanisms
            .tensor_bool(values, shape)
            .map_err(optional_mechanism_error),
    }
}

fn lower_host_input<M, E>(
    host: PreparedModelInput<HostTensor>,
    mechanisms: &mut M,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorMechanisms,
    E: std::fmt::Display,
{
    if host.parts().iter().any(|part| {
        !matches!(
            part.payload(),
            PreparedInputPayload::TokenIds(_)
                | PreparedInputPayload::Tensor(_)
                | PreparedInputPayload::Embeddings(_)
        )
    }) {
        return Err(ProcessorExecutionError::Plan(
            "host processor emitted an unsupported payload kind".into(),
        ));
    }
    let mut parts = Vec::with_capacity(host.len());
    for part in host.parts() {
        let payload = match part.payload() {
            PreparedInputPayload::TokenIds(value) => {
                PreparedInputPayload::TokenIds(lower_host_tensor(value, mechanisms)?)
            }
            PreparedInputPayload::Tensor(value) => {
                PreparedInputPayload::Tensor(lower_host_tensor(value, mechanisms)?)
            }
            PreparedInputPayload::Embeddings(value) => {
                PreparedInputPayload::Embeddings(lower_host_tensor(value, mechanisms)?)
            }
            _ => unreachable!("payload kinds were preflighted before native lowering"),
        };
        let metadata = part
            .metadata()
            .iter()
            .map(|(key, tensor)| lower_host_tensor(tensor, mechanisms).map(|tensor| (*key, tensor)))
            .collect::<Result<Vec<_>, _>>()?;
        parts.push(
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                metadata,
                part.extents().iter().copied(),
            )
            .map_err(ProcessorExecutionError::Prepared)?,
        );
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

/// Failure from architecture processor semantics, text encoding, or a mechanism.
#[derive(Debug, thiserror::Error)]
pub enum ProcessorExecutionError<E, M>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    /// Architecture processor policy rejected the request.
    #[error("processor plan rejected input: {0}")]
    Plan(String),
    /// Facade tokenizer callback failed.
    #[error("processor text encoding failed: {0}")]
    Text(E),
    /// Generic host-media or tensor mechanism failed.
    #[error("processor mechanism failed: {0}")]
    Mechanism(M),
    /// Prepared-part or identity construction failed.
    #[error("prepared input is invalid: {0}")]
    Prepared(PreparedInputError),
}

fn plan_error<E: std::fmt::Display, M: std::fmt::Display>(
    error: ProcessorPlanError,
) -> ProcessorExecutionError<E, M> {
    ProcessorExecutionError::Plan(error.to_string())
}

fn mechanism_error<E: std::fmt::Display, M: std::fmt::Display>(
    error: M,
) -> ProcessorExecutionError<E, M> {
    ProcessorExecutionError::Mechanism(error)
}

fn optional_mechanism_error<E: std::fmt::Display, M: std::fmt::Display>(
    error: OptionalProcessorMechanism<M>,
) -> ProcessorExecutionError<E, M> {
    match error {
        OptionalProcessorMechanism::Unavailable(name) => {
            ProcessorExecutionError::Plan(format!("required {name} mechanism is unavailable"))
        }
        OptionalProcessorMechanism::Backend(error) => ProcessorExecutionError::Mechanism(error),
    }
}

fn push_tokens<M, E>(
    parts: &mut Vec<PreparedInputPart<M::Tensor>>,
    ids: &[u32],
    mechanisms: &mut M,
) -> Result<(), ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    if ids.is_empty() {
        return Ok(());
    }
    let tensor = mechanisms
        .tensor_u32(ids, &[1, ids.len()])
        .map_err(mechanism_error)?;
    parts.push(
        PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(tensor),
            [],
        )
        .map_err(ProcessorExecutionError::Prepared)?,
    );
    Ok(())
}

fn prepare_gemma4<M, E>(
    plan: &Gemma4ProcessorPlan,
    request: &TokenizedMultimodalRequest,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let mut parts = Vec::new();
    for segment in request.segments() {
        match segment {
            TokenizedMultimodalSegment::TokenIds(ids) => {
                push_tokens(&mut parts, ids, mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                let image_plan = plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(plan_error)?;
                push_tokens(&mut parts, &[image_plan.framing.start_token_id], mechanisms)?;
                parts.push(gemma4_image(
                    InputModality::Image,
                    image,
                    &image_plan,
                    mechanisms,
                )?);
                push_tokens(&mut parts, &[image_plan.framing.end_token_id], mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                let first = consistent_video_frame(video)?;
                let video_plan = plan
                    .video(
                        video.frames().len(),
                        first.height() as usize,
                        first.width() as usize,
                        video.source_fps(),
                        video.sampling(),
                    )
                    .map_err(plan_error)?;
                push_gemma4_video(&mut parts, video, &video_plan, mechanisms, encode_text)?;
            }
            TokenizedMultimodalSegment::Media(Media::Audio(audio)) => {
                let audio_plan = plan.audio().map_err(plan_error)?;
                push_tokens(&mut parts, &[audio_plan.framing.start_token_id], mechanisms)?;
                parts.push(gemma4_audio(audio, &audio_plan, mechanisms)?);
                push_tokens(&mut parts, &[audio_plan.framing.end_token_id], mechanisms)?;
            }
        }
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

fn gemma4_audio<M, E>(
    audio: &eredu_core::Audio,
    plan: &Gemma4AudioPlan,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let request = AudioFeatureRequest::SemicausalHtk {
        sample_rate: plan.sample_rate,
        frame_length: plan.frame_length,
        hop_length: plan.hop_length,
        fft_length: plan.fft_length,
        mel_bins: plan.mel_bins,
        min_frequency: plan.min_frequency,
        max_frequency: plan.max_frequency,
        mel_floor: plan.mel_floor,
        max_samples: plan.max_samples,
        pad_to_multiple: plan.pad_to_multiple,
    };
    let features = mechanisms
        .audio_features(audio, &request)
        .map_err(optional_mechanism_error)?;
    if features.values.len() != features.frames.saturating_mul(features.mel_bins)
        || features.mask.len() != features.frames
        || features.mel_bins != plan.mel_bins
    {
        return Err(ProcessorExecutionError::Plan(
            "audio mechanism returned inconsistent feature geometry".into(),
        ));
    }
    let valid_frames = features.mask.iter().filter(|valid| **valid).count();
    let payload = mechanisms
        .tensor_f32(&features.values, &[1, features.frames, features.mel_bins])
        .map_err(mechanism_error)?;
    let mask = mechanisms
        .tensor_bool(&features.mask, &[1, features.frames])
        .map_err(optional_mechanism_error)?;
    PreparedInputPart::new_with_extents(
        InputModality::Audio,
        PreparedInputPayload::Tensor(payload),
        [(InputMetadataKey::AudioMask, mask)],
        [InputExtent::AudioValidFrames(valid_frames)],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn consistent_video_frame<E, M>(
    video: &eredu_core::Video,
) -> Result<&RgbImage, ProcessorExecutionError<E, M>>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    let first = video.frames().first().expect("decoded video is non-empty");
    eredu_media::video::validate_video(video)
        .map_err(|error| ProcessorExecutionError::Plan(error.to_string()))?;
    Ok(first)
}

fn gemma4_image<M, E>(
    modality: InputModality,
    image: &RgbImage,
    plan: &Gemma4ImagePlan,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let image = mechanisms
        .normalize_rgb(image, plan.transform)
        .map_err(mechanism_error)?;
    let (values, positions, grid, extent, shape) =
        pack_gemma4(&image, plan.patch_size, plan.max_patches)?;
    let payload = mechanisms
        .tensor_f32(&values, &shape)
        .map_err(mechanism_error)?;
    let positions = mechanisms
        .tensor_i32(&positions, &[1, plan.max_patches, 2])
        .map_err(mechanism_error)?;
    let grid_tensor = mechanisms
        .tensor_i32(&grid, &[1, 3])
        .map_err(mechanism_error)?;
    PreparedInputPart::new_with_extents(
        modality,
        PreparedInputPayload::Tensor(payload),
        [
            (InputMetadataKey::PatchGrid, grid_tensor),
            (InputMetadataKey::PatchPositions, positions),
        ],
        [InputExtent::PatchGrid {
            time: extent[0],
            height: extent[1],
            width: extent[2],
        }],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn push_gemma4_video<M, E>(
    parts: &mut Vec<PreparedInputPart<M::Tensor>>,
    video: &eredu_core::Video,
    plan: &Gemma4VideoPlan,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<(), ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    for frame in &plan.frames {
        let mut prefix =
            encode_text(&frame.timestamp_text).map_err(ProcessorExecutionError::Text)?;
        prefix.push(plan.framing.start_token_id);
        push_tokens(parts, &prefix, mechanisms)?;
        let image_plan = Gemma4ImagePlan {
            framing: plan.framing,
            transform: plan.transform,
            patch_size: plan.patch_size,
            max_patches: plan.max_patches,
        };
        parts.push(gemma4_image(
            InputModality::Video,
            &video.frames()[frame.source_index],
            &image_plan,
            mechanisms,
        )?);
        push_tokens(parts, &[plan.framing.end_token_id], mechanisms)?;
    }
    Ok(())
}

type Gemma4Packed = (Vec<f32>, Vec<i32>, [i32; 3], [usize; 3], Vec<usize>);

fn pack_gemma4<E, M>(
    image: &NormalizedImage,
    patch_size: usize,
    max_patches: usize,
) -> Result<Gemma4Packed, ProcessorExecutionError<E, M>>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    if !image.height().is_multiple_of(patch_size) || !image.width().is_multiple_of(patch_size) {
        return Err(ProcessorExecutionError::Plan(
            "processed Gemma visual geometry is not divisible by its patch size".into(),
        ));
    }
    let rows = image.height() / patch_size;
    let columns = image.width() / patch_size;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| ProcessorExecutionError::Plan("Gemma patch count overflowed".into()))?;
    if count > max_patches {
        return Err(ProcessorExecutionError::Plan(format!(
            "Gemma visual input produced {count} patches, exceeding {max_patches}"
        )));
    }
    let width = 3usize
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or_else(|| ProcessorExecutionError::Plan("Gemma patch width overflowed".into()))?;
    let mut values = vec![0.0; max_patches * width];
    let mut positions = vec![-1; max_patches * 2];
    for patch_y in 0..rows {
        for patch_x in 0..columns {
            let index = patch_y * columns + patch_x;
            positions[index * 2] = i32::try_from(patch_x)
                .map_err(|_| ProcessorExecutionError::Plan("patch column exceeds i32".into()))?;
            positions[index * 2 + 1] = i32::try_from(patch_y)
                .map_err(|_| ProcessorExecutionError::Plan("patch row exceeds i32".into()))?;
            let mut output = index * width;
            for y in 0..patch_size {
                for x in 0..patch_size {
                    for channel in 0..3 {
                        values[output] =
                            image.get(channel, patch_y * patch_size + y, patch_x * patch_size + x);
                        output += 1;
                    }
                }
            }
        }
    }
    let grid = [
        1,
        i32::try_from(rows)
            .map_err(|_| ProcessorExecutionError::Plan("patch rows exceed i32".into()))?,
        i32::try_from(columns)
            .map_err(|_| ProcessorExecutionError::Plan("patch columns exceed i32".into()))?,
    ];
    Ok((
        values,
        positions,
        grid,
        [1, rows, columns],
        vec![1, max_patches, width],
    ))
}

fn prepare_inkling<M, E>(
    plan: &InklingProcessorPlan,
    request: &TokenizedMultimodalRequest,
    mechanisms: &mut M,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let mut parts = Vec::new();
    for segment in request.segments() {
        match segment {
            TokenizedMultimodalSegment::TokenIds(ids) => {
                push_tokens(&mut parts, ids, mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                let image_plan = plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(plan_error)?;
                push_tokens(&mut parts, &[image_plan.start_token_id], mechanisms)?;
                parts.push(inkling_image(image, image_plan, mechanisms)?);
            }
            TokenizedMultimodalSegment::Media(Media::Audio(audio)) => {
                let audio_plan = plan.audio().map_err(plan_error)?;
                push_tokens(&mut parts, &[audio_plan.start_token_id], mechanisms)?;
                parts.push(inkling_audio(audio, audio_plan, mechanisms)?);
            }
            TokenizedMultimodalSegment::Media(Media::Video(_)) => {
                return Err(ProcessorExecutionError::Plan(
                    "Inkling processor does not accept video".into(),
                ));
            }
        }
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

fn inkling_image<M, E>(
    image: &RgbImage,
    plan: InklingImagePlan,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let width = image.width() as usize;
    let height = image.height() as usize;
    let patch = plan.patch_size;
    let patch_count = plan
        .patch_rows
        .checked_mul(plan.patch_columns)
        .ok_or_else(|| ProcessorExecutionError::Plan("Inkling patch count overflowed".into()))?;
    let mut output = Vec::with_capacity(
        patch_count
            .checked_mul(plan.temporal_patch_size)
            .and_then(|value| value.checked_mul(patch))
            .and_then(|value| value.checked_mul(patch))
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                ProcessorExecutionError::Plan("Inkling patch storage overflowed".into())
            })?,
    );
    for row in 0..plan.patch_rows {
        for column in 0..plan.patch_columns {
            for _time in 0..plan.temporal_patch_size {
                for y in 0..patch {
                    for x in 0..patch {
                        let source_y = row * patch + y;
                        let source_x = column * patch + x;
                        for channel in 0..3 {
                            let raw = if source_y < height && source_x < width {
                                image.pixels()[(source_y * width + source_x) * 3 + channel] as f32
                            } else {
                                plan.padding_value
                            };
                            output.push(
                                (raw * plan.rescale_factor - plan.mean[channel])
                                    / plan.std[channel],
                            );
                        }
                    }
                }
            }
        }
    }
    let payload = mechanisms
        .tensor_f32(
            &output,
            &[patch_count, plan.temporal_patch_size, patch, patch, 3],
        )
        .map_err(mechanism_error)?;
    PreparedInputPart::new(
        InputModality::Image,
        PreparedInputPayload::Tensor(payload),
        [],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn inkling_audio<M, E>(
    audio: &eredu_core::Audio,
    plan: InklingAudioPlan,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let request = AudioFeatureRequest::LeadingSlaney {
        sample_rate: plan.sample_rate,
        fft_length: plan.fft_length,
        hop_length: plan.hop_length,
        leading_zeros: plan.framing.leading_zeros,
        mel_bins: plan.mel_bins,
        min_frequency: plan.min_frequency,
        max_frequency: plan.max_frequency,
        energy_floor: plan.energy_floor,
    };
    let features = mechanisms
        .audio_features(audio, &request)
        .map_err(optional_mechanism_error)?;
    if features.values.len() != features.frames.saturating_mul(features.mel_bins)
        || features.mel_bins != plan.mel_bins
    {
        return Err(ProcessorExecutionError::Plan(
            "audio mechanism returned inconsistent Inkling feature geometry".into(),
        ));
    }
    let span = f64::from(plan.dmel_max - plan.dmel_min);
    let centers = (0..plan.dmel_bins)
        .map(|index| f64::from(plan.dmel_min) + span * index as f64 / (plan.dmel_bins - 1) as f64)
        .collect::<Vec<_>>();
    let ids = features
        .values
        .iter()
        .map(|value| {
            let value = f64::from(*value).clamp(f64::from(plan.dmel_min), f64::from(plan.dmel_max));
            centers
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (value - **left).abs().total_cmp(&(value - **right).abs())
                })
                .map_or(0, |(index, _)| i32::try_from(index).unwrap_or(i32::MAX))
        })
        .collect::<Vec<_>>();
    let payload = mechanisms
        .tensor_i32(&ids, &[1, features.frames, features.mel_bins])
        .map_err(mechanism_error)?;
    let mask_values = vec![true; features.frames];
    let mask = mechanisms
        .tensor_bool(&mask_values, &[1, features.frames])
        .map_err(optional_mechanism_error)?;
    PreparedInputPart::new_with_extents(
        InputModality::Audio,
        PreparedInputPayload::Tensor(payload),
        [(InputMetadataKey::AudioMask, mask)],
        [InputExtent::AudioValidFrames(features.frames)],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn prepare_muse<M, E>(
    plan: &MuseProcessorPlan,
    request: &TokenizedMultimodalRequest,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let mut parts = Vec::new();
    for segment in request.segments() {
        match segment {
            TokenizedMultimodalSegment::TokenIds(ids) => {
                push_tokens(&mut parts, ids, mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                let image_plan = plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(plan_error)?;
                push_tokens(
                    &mut parts,
                    &encode_text(image_plan.start_text).map_err(ProcessorExecutionError::Text)?,
                    mechanisms,
                )?;
                parts.push(muse_image(
                    InputModality::Image,
                    &[image],
                    image_plan.transform,
                    image_plan.patches,
                    true,
                    mechanisms,
                )?);
                push_tokens(
                    &mut parts,
                    &encode_text(image_plan.end_text).map_err(ProcessorExecutionError::Text)?,
                    mechanisms,
                )?;
            }
            TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                let first = consistent_video_frame(video)?;
                let video_plan = plan
                    .video(
                        video.frames().len(),
                        first.height() as usize,
                        first.width() as usize,
                        video.source_fps(),
                        video.sampling(),
                    )
                    .map_err(plan_error)?;
                push_muse_video(&mut parts, video, &video_plan, mechanisms, encode_text)?;
            }
            TokenizedMultimodalSegment::Media(Media::Audio(_)) => {
                return Err(ProcessorExecutionError::Plan(
                    "Muse visual processor does not accept audio".into(),
                ));
            }
        }
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

fn push_muse_video<M, E>(
    parts: &mut Vec<PreparedInputPart<M::Tensor>>,
    video: &eredu_core::Video,
    plan: &MuseVideoPlan,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<(), ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    push_tokens(
        parts,
        &encode_text(plan.start_text).map_err(ProcessorExecutionError::Text)?,
        mechanisms,
    )?;
    for group in &plan.groups {
        push_tokens(
            parts,
            &encode_text(&group.timestamp_text).map_err(ProcessorExecutionError::Text)?,
            mechanisms,
        )?;
        let frames = group
            .source_indices
            .iter()
            .map(|index| &video.frames()[*index])
            .collect::<Vec<_>>();
        parts.push(muse_image(
            InputModality::Video,
            &frames,
            plan.transform,
            plan.patches,
            false,
            mechanisms,
        )?);
        push_tokens(
            parts,
            &encode_text(group.boundary_text).map_err(ProcessorExecutionError::Text)?,
            mechanisms,
        )?;
    }
    Ok(())
}

fn muse_image<M, E>(
    modality: InputModality,
    frames: &[&RgbImage],
    transform: RgbTransformPlan,
    patches: MusePatchPlan,
    duplicate_image: bool,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let frames = frames
        .iter()
        .map(|frame| {
            mechanisms
                .normalize_rgb(frame, transform)
                .map_err(mechanism_error)
        })
        .collect::<Result<Vec<_>, ProcessorExecutionError<E, M::Error>>>()?;
    let (values, shape, grid) = pack_muse(&frames, patches, duplicate_image)?;
    let payload = mechanisms
        .tensor_f32(&values, &shape)
        .map_err(mechanism_error)?;
    let grid = mechanisms
        .tensor_i32(&grid, &[1, 3])
        .map_err(mechanism_error)?;
    PreparedInputPart::new(
        modality,
        PreparedInputPayload::Tensor(payload),
        [(InputMetadataKey::PatchGrid, grid)],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn pack_muse<E, M>(
    frames: &[NormalizedImage],
    config: MusePatchPlan,
    duplicate_image: bool,
) -> Result<(Vec<f32>, Vec<usize>, [i32; 3]), ProcessorExecutionError<E, M>>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    let first = frames
        .first()
        .ok_or_else(|| ProcessorExecutionError::Plan("Muse patch input is empty".into()))?;
    if !first.height().is_multiple_of(config.patch_size)
        || !first.width().is_multiple_of(config.patch_size)
    {
        return Err(ProcessorExecutionError::Plan(
            "processed Muse visual geometry is not divisible by its patch size".into(),
        ));
    }
    let temporal = config.temporal_patch_size;
    if !duplicate_image && frames.len() != temporal {
        return Err(ProcessorExecutionError::Plan(format!(
            "Muse video group has {} frames, expected {temporal}",
            frames.len()
        )));
    }
    let rows = first.height() / config.patch_size;
    let columns = first.width() / config.patch_size;
    let width = temporal * 3 * config.patch_size * config.patch_size;
    let mut values = Vec::with_capacity(rows * columns * width);
    let selected_frames = if duplicate_image {
        vec![first; temporal]
    } else {
        frames.iter().take(temporal).collect::<Vec<_>>()
    };
    for patch_y in 0..rows {
        for patch_x in 0..columns {
            for frame in &selected_frames {
                for channel in 0..3 {
                    for y in 0..config.patch_size {
                        for x in 0..config.patch_size {
                            values.push(frame.get(
                                channel,
                                patch_y * config.patch_size + y,
                                patch_x * config.patch_size + x,
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok((
        values,
        vec![rows * columns, width],
        [
            1,
            i32::try_from(rows)
                .map_err(|_| ProcessorExecutionError::Plan("patch rows exceed i32".into()))?,
            i32::try_from(columns)
                .map_err(|_| ProcessorExecutionError::Plan("patch columns exceed i32".into()))?,
        ],
    ))
}

fn prepare_qwen<M, E>(
    plan: &QwenProcessorPlan,
    request: &TokenizedMultimodalRequest,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let mut parts = Vec::new();
    for segment in request.segments() {
        match segment {
            TokenizedMultimodalSegment::TokenIds(ids) => {
                push_tokens(&mut parts, ids, mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                let image_plan = plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(plan_error)?;
                push_tokens(&mut parts, &[image_plan.framing.start_token_id], mechanisms)?;
                parts.push(qwen_image(image, &image_plan, mechanisms)?);
                push_tokens(&mut parts, &[image_plan.framing.end_token_id], mechanisms)?;
            }
            TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                let first = consistent_video_frame(video)?;
                let video_plan = plan
                    .video(
                        video.frames().len(),
                        first.height() as usize,
                        first.width() as usize,
                        video.source_fps(),
                        video.sampling(),
                    )
                    .map_err(plan_error)?;
                push_qwen_video(&mut parts, video, &video_plan, mechanisms, encode_text)?;
            }
            TokenizedMultimodalSegment::Media(Media::Audio(_)) => {
                return Err(ProcessorExecutionError::Plan(
                    "Qwen visual processor does not accept audio".into(),
                ));
            }
        }
    }
    PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
        .map_err(ProcessorExecutionError::Prepared)
}

fn qwen_image<M, E>(
    image: &RgbImage,
    plan: &QwenImagePlan,
    mechanisms: &mut M,
) -> Result<PreparedInputPart<M::Tensor>, ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    let image = mechanisms
        .normalize_rgb(image, plan.transform)
        .map_err(mechanism_error)?;
    let (values, shape, grid) = pack_qwen(&[image], plan.patches, true)?;
    let payload = mechanisms
        .tensor_f32(&values, &shape)
        .map_err(mechanism_error)?;
    let metadata = mechanisms
        .tensor_i32(&grid, &[1, 3])
        .map_err(mechanism_error)?;
    PreparedInputPart::new(
        InputModality::Image,
        PreparedInputPayload::Tensor(payload),
        [(InputMetadataKey::PatchGrid, metadata)],
    )
    .map_err(ProcessorExecutionError::Prepared)
}

fn push_qwen_video<M, E>(
    parts: &mut Vec<PreparedInputPart<M::Tensor>>,
    video: &eredu_core::Video,
    plan: &QwenVideoPlan,
    mechanisms: &mut M,
    encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
) -> Result<(), ProcessorExecutionError<E, M::Error>>
where
    M: ProcessorOperations,
    E: std::fmt::Display,
{
    for group in &plan.groups {
        let mut prefix =
            encode_text(&group.timestamp_text).map_err(ProcessorExecutionError::Text)?;
        prefix.push(plan.framing.start_token_id);
        push_tokens(parts, &prefix, mechanisms)?;
        let frames = group
            .source_indices
            .iter()
            .map(|index| {
                mechanisms
                    .normalize_rgb(&video.frames()[*index], plan.transform)
                    .map_err(mechanism_error)
            })
            .collect::<Result<Vec<_>, ProcessorExecutionError<E, M::Error>>>()?;
        let (values, shape, grid) = pack_qwen(&frames, plan.patches, false)?;
        let payload = mechanisms
            .tensor_f32(&values, &shape)
            .map_err(mechanism_error)?;
        let metadata = mechanisms
            .tensor_i32(&grid, &[1, 3])
            .map_err(mechanism_error)?;
        parts.push(
            PreparedInputPart::new(
                InputModality::Video,
                PreparedInputPayload::Tensor(payload),
                [(InputMetadataKey::PatchGrid, metadata)],
            )
            .map_err(ProcessorExecutionError::Prepared)?,
        );
        push_tokens(parts, &[plan.framing.end_token_id], mechanisms)?;
    }
    Ok(())
}

fn pack_qwen<E, M>(
    frames: &[NormalizedImage],
    config: QwenPatchPlan,
    duplicate_image: bool,
) -> Result<(Vec<f32>, Vec<usize>, [i32; 3]), ProcessorExecutionError<E, M>>
where
    E: std::fmt::Display,
    M: std::fmt::Display,
{
    let first = frames
        .first()
        .ok_or_else(|| ProcessorExecutionError::Plan("visual patch input is empty".into()))?;
    if first.height() % config.patch_size != 0 || first.width() % config.patch_size != 0 {
        return Err(ProcessorExecutionError::Plan(format!(
            "processed visual geometry {}x{} is not divisible by patch size {}",
            first.width(),
            first.height(),
            config.patch_size
        )));
    }
    let temporal = config.temporal_patch_size;
    if !duplicate_image && !frames.len().is_multiple_of(temporal) {
        return Err(ProcessorExecutionError::Plan(format!(
            "processed frame count {} is not divisible by temporal patch size {temporal}",
            frames.len()
        )));
    }
    if frames
        .iter()
        .any(|frame| frame.width() != first.width() || frame.height() != first.height())
    {
        return Err(ProcessorExecutionError::Plan(
            "processed visual frames have inconsistent geometry".into(),
        ));
    }
    let grid_t = if duplicate_image {
        1
    } else {
        frames.len() / temporal
    };
    let grid_h = first.height() / config.patch_size;
    let grid_w = first.width() / config.patch_size;
    if !grid_h.is_multiple_of(config.merge_size) || !grid_w.is_multiple_of(config.merge_size) {
        return Err(ProcessorExecutionError::Plan(format!(
            "visual patch grid {grid_h}x{grid_w} is not divisible by merge size {}",
            config.merge_size
        )));
    }
    let patch_count = grid_t
        .checked_mul(grid_h)
        .and_then(|value| value.checked_mul(grid_w))
        .ok_or_else(|| ProcessorExecutionError::Plan("visual patch count overflowed".into()))?;
    let patch_width = 3usize
        .checked_mul(temporal)
        .and_then(|value| value.checked_mul(config.patch_size))
        .and_then(|value| value.checked_mul(config.patch_size))
        .ok_or_else(|| ProcessorExecutionError::Plan("visual patch width overflowed".into()))?;
    let mut values =
        Vec::with_capacity(patch_count.checked_mul(patch_width).ok_or_else(|| {
            ProcessorExecutionError::Plan("visual patch storage overflowed".into())
        })?);
    for temporal_group in 0..grid_t {
        for block_y in 0..grid_h / config.merge_size {
            for block_x in 0..grid_w / config.merge_size {
                for merge_y in 0..config.merge_size {
                    for merge_x in 0..config.merge_size {
                        let patch_y = (block_y * config.merge_size + merge_y) * config.patch_size;
                        let patch_x = (block_x * config.merge_size + merge_x) * config.patch_size;
                        for channel in 0..3 {
                            for time in 0..temporal {
                                let frame = if duplicate_image {
                                    first
                                } else {
                                    &frames[temporal_group * temporal + time]
                                };
                                for y in 0..config.patch_size {
                                    for x in 0..config.patch_size {
                                        values.push(frame.get(channel, patch_y + y, patch_x + x));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let grid = [
        i32::try_from(grid_t)
            .map_err(|_| ProcessorExecutionError::Plan("temporal grid exceeds i32".into()))?,
        i32::try_from(grid_h)
            .map_err(|_| ProcessorExecutionError::Plan("patch rows exceed i32".into()))?,
        i32::try_from(grid_w)
            .map_err(|_| ProcessorExecutionError::Plan("patch columns exceed i32".into()))?,
    ];
    Ok((values, vec![patch_count, patch_width], grid))
}

#[cfg(all(test, feature = "image", feature = "audio"))]
mod tests {
    use std::{cell::Cell, convert::Infallible};

    use eredu_core::{
        checkpoint::TensorDtype, InputTensorIdentity, MultimodalRequest, MultimodalSegment, Video,
        VideoSampling,
    };

    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct TestTensor {
        dtype: TensorDtype,
        shape: Vec<usize>,
        f32_values: Vec<f32>,
        i32_values: Vec<i32>,
        bool_values: Vec<bool>,
    }

    #[derive(Default)]
    struct TestMechanisms {
        tensors: Cell<usize>,
    }

    impl PreparedInputInspector<TestTensor> for TestMechanisms {
        fn identity(&self, tensor: &TestTensor) -> Result<InputTensorIdentity, PreparedInputError> {
            InputTensorIdentity::new(tensor.dtype.clone(), tensor.shape.clone())
        }

        fn i32_values(&self, tensor: &TestTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
            Ok(tensor.i32_values.clone())
        }

        fn bool_values(
            &self,
            tensor: &TestTensor,
        ) -> Result<Vec<bool>, eredu_core::CapabilityError> {
            Ok(tensor.bool_values.clone())
        }
    }

    impl ProcessorMechanisms for TestMechanisms {
        type Tensor = TestTensor;
        type Error = &'static str;

        fn tensor_u32(
            &mut self,
            values: &[u32],
            shape: &[usize],
        ) -> Result<Self::Tensor, Self::Error> {
            self.tensors.set(self.tensors.get() + 1);
            Ok(TestTensor {
                dtype: TensorDtype::U32,
                shape: shape.to_vec(),
                f32_values: Vec::new(),
                i32_values: values
                    .iter()
                    .map(|value| i32::try_from(*value).unwrap())
                    .collect(),
                bool_values: Vec::new(),
            })
        }

        fn tensor_f32(
            &mut self,
            values: &[f32],
            shape: &[usize],
        ) -> Result<Self::Tensor, Self::Error> {
            self.tensors.set(self.tensors.get() + 1);
            Ok(TestTensor {
                dtype: TensorDtype::F32,
                shape: shape.to_vec(),
                f32_values: values.to_vec(),
                i32_values: Vec::new(),
                bool_values: Vec::new(),
            })
        }

        fn tensor_i32(
            &mut self,
            values: &[i32],
            shape: &[usize],
        ) -> Result<Self::Tensor, Self::Error> {
            self.tensors.set(self.tensors.get() + 1);
            Ok(TestTensor {
                dtype: TensorDtype::I32,
                shape: shape.to_vec(),
                f32_values: Vec::new(),
                i32_values: values.to_vec(),
                bool_values: Vec::new(),
            })
        }

        fn tensor_bool(
            &mut self,
            values: &[bool],
            shape: &[usize],
        ) -> Result<Self::Tensor, OptionalProcessorMechanism<Self::Error>> {
            self.tensors.set(self.tensors.get() + 1);
            Ok(TestTensor {
                dtype: TensorDtype::Bool,
                shape: shape.to_vec(),
                f32_values: Vec::new(),
                i32_values: Vec::new(),
                bool_values: values.to_vec(),
            })
        }
    }

    fn qwen_processor() -> PreparedProcessor {
        let model = br#"{"vision_start_token_id":44,"vision_end_token_id":45}"#;
        let visual = br#"{
            "size":{"shortest_edge":4,"longest_edge":4},
            "patch_size":2,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
            "min_frames":1,"max_frames":8
        }"#;
        PreparedProcessor {
            kind: ProcessorKind::Qwen(
                QwenProcessorPlan::from_hf_json(model, Some(visual), Some(visual))
                    .unwrap()
                    .unwrap(),
            ),
        }
    }

    fn image(value: u8) -> RgbImage {
        RgbImage::new(vec![value; 4 * 4 * 3], 4, 4).unwrap()
    }

    #[test]
    fn later_plan_failure_precedes_all_native_tensor_work() {
        let audio = eredu_core::Audio::new(vec![0.0; 320], 16_000).unwrap();
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Audio(audio)),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!())
        .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let error = qwen_processor()
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap_err();

        assert!(matches!(error, ProcessorExecutionError::Plan(_)));
        assert_eq!(mechanisms.tensors.get(), 0);
    }

    #[test]
    fn later_host_media_failure_precedes_all_native_tensor_work() {
        let model = br#"{
            "boa_token_id":43,"eoa_token_id":44,"audio_config":{}
        }"#;
        let processor = PreparedProcessor {
            kind: ProcessorKind::Gemma4(
                Gemma4ProcessorPlan::from_hf_json(model, None, None)
                    .unwrap()
                    .unwrap(),
            ),
        };
        let audio = eredu_core::Audio::new(vec![0.0; 320], 8_000).unwrap();
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Audio(audio)),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!())
        .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let error = processor
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap_err();

        assert!(matches!(error, ProcessorExecutionError::Plan(_)));
        assert_eq!(mechanisms.tensors.get(), 0);
    }

    #[test]
    fn qwen_image_execution_owns_framing_patch_order_and_identity() {
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Image(image(8))),
            MultimodalSegment::TokenIds(vec![9]),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!())
        .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let prepared = qwen_processor()
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap();

        assert_eq!(prepared.len(), 5);
        assert_eq!(prepared.parts()[2].modality(), InputModality::Image);
        assert_eq!(prepared.parts()[2].payload().value().shape, [4, 24]);
        assert_eq!(
            prepared.parts()[2]
                .metadata_value(InputMetadataKey::PatchGrid)
                .unwrap()
                .i32_values,
            [1, 2, 2]
        );
        assert_eq!(prepared.identity().len(), prepared.len());
        assert_eq!(mechanisms.tensors.get(), 6);
    }

    #[test]
    fn processor_output_intervention_changes_the_admitted_tensor() {
        struct ReplaceFirstPayload {
            paths: Vec<String>,
        }

        impl ActivationObserver<TestTensor, ProcessorExecutionError<Infallible, &'static str>>
            for ReplaceFirstPayload
        {
            fn observe(
                &mut self,
                path: &str,
                _: &TestTensor,
            ) -> Result<(), ProcessorExecutionError<Infallible, &'static str>> {
                self.paths.push(path.to_owned());
                Ok(())
            }

            fn intervene(
                &mut self,
                path: &str,
                value: &TestTensor,
            ) -> Result<Option<TestTensor>, ProcessorExecutionError<Infallible, &'static str>>
            {
                if path == format!("{}.0", eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH) {
                    let mut replacement = value.clone();
                    replacement.i32_values = vec![99];
                    return Ok(Some(replacement));
                }
                Ok(None)
            }
        }

        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Image(image(8))),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!())
        .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let mut observer = ReplaceFirstPayload { paths: Vec::new() };
        let prepared = qwen_processor()
            .prepare_with_observer(
                &request,
                &mut mechanisms,
                &mut |_| Ok::<_, Infallible>(Vec::new()),
                &mut observer,
            )
            .unwrap();

        assert_eq!(prepared.parts()[0].payload().value().i32_values, [99]);
        assert!(observer.paths.iter().any(|path| {
            path == &format!("{}.2", eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH)
        }));
        assert!(observer
            .paths
            .iter()
            .any(|path| path.contains("metadata.PatchGrid")));
    }

    #[test]
    fn qwen_video_execution_owns_sampling_timestamps_and_temporal_packing() {
        let video = Video::new(vec![image(1), image(2)], Some(1.0), VideoSampling::All).unwrap();
        let request = MultimodalRequest::new(vec![MultimodalSegment::Media(Media::Video(video))])
            .unwrap()
            .tokenize::<Infallible>(|_| unreachable!())
            .unwrap();
        let mut timestamps = Vec::new();
        let mut mechanisms = TestMechanisms::default();
        let prepared = qwen_processor()
            .prepare(&request, &mut mechanisms, &mut |text| {
                timestamps.push(text.to_owned());
                Ok::<_, Infallible>(vec![90])
            })
            .unwrap();

        assert_eq!(timestamps, ["<0.5 seconds>"]);
        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared.parts()[1].modality(), InputModality::Video);
        assert_eq!(prepared.parts()[1].payload().value().shape, [4, 24]);
    }

    #[test]
    fn gemma_image_execution_owns_padding_positions_and_extents() {
        let model = br#"{
            "boi_token_id":43,"eoi_token_id":44,"vision_soft_tokens_per_image":70,
            "vision_config":{"patch_size":2,"pooling_kernel_size":1}
        }"#;
        let plan = Gemma4ProcessorPlan::from_hf_json(model, None, None)
            .unwrap()
            .unwrap();
        let processor = PreparedProcessor {
            kind: ProcessorKind::Gemma4(plan),
        };
        let request =
            MultimodalRequest::new(vec![MultimodalSegment::Media(Media::Image(image(8)))])
                .unwrap()
                .tokenize::<Infallible>(|_| unreachable!())
                .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let prepared = processor
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap();

        assert_eq!(prepared.len(), 3);
        let media = &prepared.parts()[1];
        assert_eq!(media.payload().value().shape, [1, 70, 12]);
        assert_eq!(
            media
                .metadata_value(InputMetadataKey::PatchPositions)
                .unwrap()
                .shape,
            [1, 70, 2]
        );
        assert_eq!(
            media.extents(),
            [InputExtent::PatchGrid {
                time: 1,
                height: 8,
                width: 8,
            }]
        );
    }

    #[test]
    fn gemma_audio_execution_owns_framing_mask_and_valid_extent() {
        let model = br#"{
            "boa_token_id":43,"eoa_token_id":44,"audio_config":{}
        }"#;
        let processor = PreparedProcessor {
            kind: ProcessorKind::Gemma4(
                Gemma4ProcessorPlan::from_hf_json(model, None, None)
                    .unwrap()
                    .unwrap(),
            ),
        };
        let audio = eredu_core::Audio::new(vec![0.0; 320], 16_000).unwrap();
        let request = MultimodalRequest::new(vec![MultimodalSegment::Media(Media::Audio(audio))])
            .unwrap()
            .tokenize::<Infallible>(|_| unreachable!())
            .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let prepared = processor
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap();

        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared.parts()[0].payload().value().i32_values, [43]);
        let audio = &prepared.parts()[1];
        assert_eq!(audio.payload().value().shape, [1, 2, 128]);
        assert_eq!(
            audio
                .metadata_value(InputMetadataKey::AudioMask)
                .unwrap()
                .bool_values,
            [true, false]
        );
        assert_eq!(audio.extents(), [InputExtent::AudioValidFrames(1)]);
        assert_eq!(prepared.parts()[2].payload().value().i32_values, [44]);
    }

    #[test]
    fn inkling_execution_owns_patch_padding_audio_quantization_and_markers() {
        let processor = PreparedProcessor {
            kind: ProcessorKind::Inkling(
                InklingProcessorPlan::from_hf_json(
                    br#"{
                        "model_type":"inkling_mm_model",
                        "image_bos_token_id":7,"audio_bos_token_id":8,
                        "audio_config":{
                            "mel_vocab_size":4,
                            "dmel_min_value":-8.0,
                            "dmel_max_value":3.0
                        }
                    }"#,
                )
                .unwrap()
                .unwrap(),
            ),
        };
        let audio = eredu_core::Audio::new(vec![0.0; 801], 16_000).unwrap();
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::Media(Media::Image(image(8))),
            MultimodalSegment::Media(Media::Audio(audio)),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!())
        .unwrap();
        let mut mechanisms = TestMechanisms::default();
        let prepared = processor
            .prepare(&request, &mut mechanisms, &mut |_| {
                Ok::<_, Infallible>(Vec::new())
            })
            .unwrap();

        assert_eq!(prepared.len(), 4);
        assert_eq!(prepared.parts()[0].payload().value().i32_values, [7]);
        assert_eq!(
            prepared.parts()[1].payload().value().shape,
            [1, 2, 40, 40, 3]
        );
        assert_eq!(prepared.parts()[2].payload().value().i32_values, [8]);
        let audio = &prepared.parts()[3];
        assert_eq!(audio.payload().value().shape, [1, 2, 80]);
        assert_eq!(audio.payload().value().i32_values[0], 0);
        assert_eq!(audio.payload().value().i32_values[80], 0);
        assert_eq!(audio.extents(), [InputExtent::AudioValidFrames(2)]);
    }

    #[test]
    fn muse_image_execution_owns_text_framing_and_temporal_duplication() {
        let visual = r#"{
            "do_resize":true,"do_rescale":true,"do_normalize":true,
            "rescale_factor":0.0039215686,"image_mean":[0.0,0.0,0.0],
            "image_std":[1.0,1.0,1.0],"patch_size":2,
            "temporal_patch_size":2,"merge_size":1,"max_image_tokens":4,
            "max_video_frame_tokens":4,"num_frames":2,"fps":1.0,
            "do_sample_frames":true,"resample":1
        }"#;
        let config = format!("{{\"image_processor\":{visual},\"video_processor\":{visual}}}");
        let processor = PreparedProcessor {
            kind: ProcessorKind::Muse(MuseProcessorPlan::from_hf_json(config.as_bytes()).unwrap()),
        };
        let request =
            MultimodalRequest::new(vec![MultimodalSegment::Media(Media::Image(image(8)))])
                .unwrap()
                .tokenize::<Infallible>(|_| unreachable!())
                .unwrap();
        let mut framing = Vec::new();
        let mut mechanisms = TestMechanisms::default();
        let prepared = processor
            .prepare(&request, &mut mechanisms, &mut |text| {
                framing.push(text.to_owned());
                Ok::<_, Infallible>(vec![90])
            })
            .unwrap();

        assert_eq!(framing, ["<|image_start|>", "<|image_end|>"]);
        assert_eq!(prepared.parts()[1].payload().value().shape, [4, 24]);
        assert_eq!(
            prepared.parts()[1]
                .metadata_value(InputMetadataKey::PatchGrid)
                .unwrap()
                .i32_values,
            [1, 2, 2]
        );
    }
}
