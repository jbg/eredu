//! Architecture selection for MLX media preprocessing.

#[cfg(any(feature = "image", feature = "audio"))]
use eredu_architectures::processor_execution::{
    AudioFeatureRequest, AudioFeatures, NormalizedRgb, OptionalProcessorMechanism,
    PreparedProcessor, ProcessorExecutionError, ProcessorMechanisms,
};
#[cfg(feature = "image")]
use eredu_architectures::processor_plan::RgbResample;
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_architectures::processor_plan::RgbTransformPlan;
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{InputTensorIdentity, PreparedInputError, TokenizedMultimodalRequest};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_runtime::PreparedInputInspector;
#[cfg(any(feature = "image", feature = "audio"))]
use safemlx::Array;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::error::Error;
#[cfg(feature = "image")]
use crate::backend::runtime::media::RgbImageView;
#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::runtime::media::{PreparedModelInput, ProcessorPreparationError};

/// Architecture-erased media processor selected during model composition.
#[derive(Debug, Clone)]
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) struct ModelProcessor {
    processor: PreparedProcessor,
}

#[cfg(any(feature = "image", feature = "audio"))]
struct MlxProcessorMechanisms;

pub(crate) fn capabilities() -> eredu_runtime::MediaPrimitiveCapabilities {
    use eredu_core::InputModality;
    use eredu_runtime::ProcessorPrimitive;

    #[allow(unused_mut)]
    let mut raw_modalities = Vec::new();
    #[allow(unused_mut)]
    let mut primitives = vec![
        ProcessorPrimitive::TensorU32,
        ProcessorPrimitive::TensorF32,
        ProcessorPrimitive::TensorI32,
        ProcessorPrimitive::TensorBool,
        ProcessorPrimitive::MetadataInspection,
    ];
    #[cfg(feature = "image")]
    {
        raw_modalities.extend([InputModality::Image, InputModality::Video]);
        primitives.extend([
            ProcessorPrimitive::RgbResizeBicubic,
            ProcessorPrimitive::RgbResizeLanczos3,
            ProcessorPrimitive::RgbNormalize,
            ProcessorPrimitive::VideoSampling,
        ]);
    }
    #[cfg(feature = "audio")]
    {
        raw_modalities.push(InputModality::Audio);
        primitives.extend([
            ProcessorPrimitive::AudioWindow,
            ProcessorPrimitive::AudioSpectrum,
            ProcessorPrimitive::AudioMelFilter,
            ProcessorPrimitive::AudioLogarithm,
        ]);
    }
    eredu_runtime::MediaPrimitiveCapabilities::new(
        raw_modalities,
        [
            InputModality::Text,
            InputModality::Image,
            InputModality::Video,
            InputModality::Audio,
        ],
        [
            InputModality::Text,
            InputModality::Image,
            InputModality::Video,
            InputModality::Audio,
        ],
        primitives,
        i32::MAX as u64,
    )
}

#[cfg(any(feature = "image", feature = "audio"))]
impl PreparedInputInspector<Array> for MlxProcessorMechanisms {
    fn identity(&self, tensor: &Array) -> Result<InputTensorIdentity, PreparedInputError> {
        crate::backend::runtime::media::input::MlxInputInspector.identity(tensor)
    }

    fn i32_values(&self, tensor: &Array) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        crate::backend::runtime::media::input::MlxInputInspector.i32_values(tensor)
    }

    fn bool_values(&self, tensor: &Array) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        crate::backend::runtime::media::input::MlxInputInspector.bool_values(tensor)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl ProcessorMechanisms for MlxProcessorMechanisms {
    type Tensor = Array;
    type Error = Error;

    fn normalize_rgb(
        &mut self,
        image: &eredu_core::RgbImage,
        plan: RgbTransformPlan,
    ) -> Result<NormalizedRgb, Self::Error> {
        #[cfg(feature = "image")]
        {
            use crate::backend::runtime::media::image::{
                rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3,
            };
            let image = RgbImageView::packed(image.pixels(), image.width(), image.height())?;
            let width = u32::try_from(plan.width)
                .map_err(|_| Error::Processor("RGB target width exceeds u32".into()))?;
            let height = u32::try_from(plan.height)
                .map_err(|_| Error::Processor("RGB target height exceeds u32".into()))?;
            let resized = match plan.resample {
                RgbResample::Bicubic => resize_rgb8_bicubic(image, width, height)?,
                RgbResample::Lanczos3 => resize_rgb8_lanczos3(image, width, height)?,
            };
            let normalized = rescale_and_normalize_rgb8(
                resized.as_view(),
                plan.rescale_factor,
                plan.mean,
                plan.std,
            )?;
            NormalizedRgb::new(
                normalized.data().to_vec(),
                normalized.width(),
                normalized.height(),
            )
            .map_err(Error::Processor)
        }
        #[cfg(not(feature = "image"))]
        {
            let _ = (image, plan);
            Err(missing_media_feature("image and video", "image"))
        }
    }

    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(Array::from_slice(values, &mlx_shape(shape)?))
    }

    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(Array::from_slice(values, &mlx_shape(shape)?))
    }

    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(Array::from_slice(values, &mlx_shape(shape)?))
    }

    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<Array, OptionalProcessorMechanism<Self::Error>> {
        Ok(Array::from_slice(
            values,
            &mlx_shape(shape).map_err(OptionalProcessorMechanism::Backend)?,
        ))
    }

    fn audio_features(
        &mut self,
        audio: &eredu_core::Audio,
        request: &AudioFeatureRequest,
    ) -> Result<AudioFeatures, OptionalProcessorMechanism<Self::Error>> {
        #[cfg(feature = "audio")]
        {
            use crate::backend::runtime::media::audio::{
                extract_leading_slaney_log_mel, extract_log_mel, AudioWaveform,
                LeadingSlaneyLogMelConfig, LogMelConfig,
            };
            let waveform = AudioWaveform::new(audio.samples(), audio.sample_rate())
                .map_err(OptionalProcessorMechanism::Backend)?;
            let features = match request {
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
            .map_err(OptionalProcessorMechanism::Backend)?;
            Ok(AudioFeatures {
                values: features.values,
                mask: features.mask,
                frames: features.frames,
                bins: features.mel_bins,
            })
        }
        #[cfg(not(feature = "audio"))]
        {
            let _ = (audio, request);
            Err(OptionalProcessorMechanism::Backend(missing_media_feature(
                "audio", "audio",
            )))
        }
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
fn mlx_shape(shape: &[usize]) -> Result<Vec<i32>, Error> {
    shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension)
                .map_err(|_| Error::Processor(format!("tensor dimension {dimension} exceeds i32")))
        })
        .collect()
}

#[cfg(all(
    any(feature = "image", feature = "audio"),
    any(not(feature = "image"), not(feature = "audio"))
))]
fn missing_media_feature(modality: &str, feature: &str) -> Error {
    Error::Processor(format!(
        "MLX {modality} preparation requires feature `{feature}` on `eredu-backend-mlx` \
         (or feature `{feature}` on the `eredu` facade)"
    ))
}

#[cfg(any(feature = "image", feature = "audio"))]
impl ModelProcessor {
    /// Lowers the authoritative architecture-owned processor plan to MLX execution.
    pub fn from_plan(
        plan: &eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    ) -> Option<Self> {
        PreparedProcessor::from_artifact(plan).map(|processor| Self { processor })
    }

    /// Instantiates raw preparation only when it belongs to the selected realization.
    pub fn from_selected(
        plan: &eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        selected: &eredu_runtime::SelectedProcessorExecution,
    ) -> Result<Option<Self>, Error> {
        if !selected.raw_media() {
            return Ok(None);
        }
        Self::from_plan(plan).map(Some).ok_or_else(|| {
            Error::Processor(
                "selected raw-media execution has no retained architecture processor".into(),
            )
        })
    }

    /// Converts a portable ordered request into owned MLX model input.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn prepare_portable_input<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>>
    where
        E: std::fmt::Display,
    {
        self.prepare_portable_input_inner(
            request,
            encode_text,
            &mut eredu_runtime::NoopObserver,
            true,
        )
    }

    /// Converts a portable request while exposing processor output before admission.
    pub fn prepare_portable_input_with_observer<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>>
    where
        E: std::fmt::Display,
    {
        self.prepare_portable_input_inner(request, encode_text, observer, false)
    }

    fn prepare_portable_input_inner<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
        retain_semantic_identity: bool,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>>
    where
        E: std::fmt::Display,
    {
        let mut observer = ProcessorObserver::<E> {
            inner: observer,
            marker: std::marker::PhantomData,
        };
        self.processor
            .prepare_with_observer(
                request,
                &mut MlxProcessorMechanisms,
                encode_text,
                &mut observer,
            )
            .and_then(|prepared| {
                if retain_semantic_identity {
                    PreparedModelInput::from_runtime_with_semantic_content(
                        prepared,
                        request.semantic_content_fingerprint(),
                    )
                    .map_err(ProcessorExecutionError::Mechanism)
                } else {
                    Ok(PreparedModelInput::from_observed_runtime(prepared))
                }
            })
            .map_err(|error| match error {
                ProcessorExecutionError::Text(error) => ProcessorPreparationError::Text(error),
                ProcessorExecutionError::Plan(error) => {
                    ProcessorPreparationError::Backend(Error::Processor(error))
                }
                ProcessorExecutionError::Mechanism(error) => {
                    ProcessorPreparationError::Backend(error)
                }
                ProcessorExecutionError::Prepared(error) => {
                    ProcessorPreparationError::Backend(Error::Processor(error.to_string()))
                }
            })
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
struct ProcessorObserver<'a, E> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    marker: std::marker::PhantomData<E>,
}

#[cfg(any(feature = "image", feature = "audio"))]
impl<E> eredu_runtime::ActivationObserver<Array, ProcessorExecutionError<E, Error>>
    for ProcessorObserver<'_, E>
where
    E: std::fmt::Display,
{
    fn observe(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<(), ProcessorExecutionError<E, Error>> {
        self.inner.observe(path, value).map_err(|error| {
            ProcessorExecutionError::Mechanism(Error::Processor(error.to_string()))
        })
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, ProcessorExecutionError<E, Error>> {
        self.inner.intervene(path, value).map_err(|error| {
            ProcessorExecutionError::Mechanism(Error::Processor(error.to_string()))
        })
    }
}

#[cfg(all(
    test,
    any(feature = "image", feature = "audio"),
    any(not(feature = "image"), not(feature = "audio"))
))]
mod feature_diagnostic_tests {
    #[cfg(not(feature = "audio"))]
    use eredu_architectures::processor_execution::AudioFeatureRequest;
    use eredu_architectures::processor_execution::ProcessorMechanisms;
    #[cfg(not(feature = "image"))]
    use eredu_architectures::processor_plan::{RgbResample, RgbTransformPlan};
    #[cfg(not(feature = "audio"))]
    use eredu_core::Audio;
    #[cfg(not(feature = "image"))]
    use eredu_core::RgbImage;

    use super::MlxProcessorMechanisms;

    #[cfg(not(feature = "image"))]
    #[test]
    fn missing_image_diagnostic_names_backend_and_facade_features() {
        let image = MlxProcessorMechanisms
            .normalize_rgb(
                &RgbImage::new(vec![0, 0, 0], 1, 1).unwrap(),
                RgbTransformPlan {
                    width: 1,
                    height: 1,
                    resample: RgbResample::Bicubic,
                    rescale_factor: 1.0,
                    mean: [0.0; 3],
                    std: [1.0; 3],
                },
            )
            .unwrap_err()
            .to_string();
        assert!(image.contains("feature `image` on `eredu-backend-mlx`"));
        assert!(image.contains("feature `image` on the `eredu` facade"));
    }

    #[cfg(not(feature = "audio"))]
    #[test]
    fn missing_audio_diagnostic_names_backend_and_facade_features() {
        let audio = MlxProcessorMechanisms
            .audio_features(
                &Audio::new(vec![0.0], 16_000).unwrap(),
                &AudioFeatureRequest::LeadingSlaney {
                    sample_rate: 16_000,
                    fft_length: 1_600,
                    hop_length: 800,
                    leading_zeros: 800,
                    mel_bins: 80,
                    min_frequency: 0.0,
                    max_frequency: 8_000.0,
                    energy_floor: 1e-10,
                },
            )
            .unwrap_err()
            .to_string();
        assert!(audio.contains("feature `audio` on `eredu-backend-mlx`"));
        assert!(audio.contains("feature `audio` on the `eredu` facade"));
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use eredu_architectures::{
        processor_execution::ProcessorMechanisms,
        processor_plan::{RgbResample, RgbTransformPlan},
    };
    use eredu_core::RgbImage;

    use super::MlxProcessorMechanisms;

    #[test]
    fn mlx_rgb_mechanism_executes_the_neutral_transform_request() {
        let normalized = MlxProcessorMechanisms
            .normalize_rgb(
                &RgbImage::new(vec![128; 4 * 4 * 3], 4, 4).unwrap(),
                RgbTransformPlan {
                    width: 4,
                    height: 4,
                    resample: RgbResample::Bicubic,
                    rescale_factor: 1.0 / 255.0,
                    mean: [0.0; 3],
                    std: [1.0; 3],
                },
            )
            .unwrap();
        assert_eq!((normalized.width(), normalized.height()), (4, 4));
        assert!((normalized.get(0, 0, 0) - 128.0 / 255.0).abs() < 1e-6);
    }
}
