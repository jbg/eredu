//! MLX tensor conversion for architecture-selected portable media processing.

#[cfg(any(feature = "image", feature = "audio"))]
use eredu_architectures::processor_execution::{
    OptionalProcessorMechanism, PreparedProcessor, ProcessorExecutionError, ProcessorMechanisms,
};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{InputTensorIdentity, PreparedInputError, TokenizedMultimodalRequest};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_runtime::PreparedInputInspector;
#[cfg(any(feature = "image", feature = "audio"))]
use safemlx::Array;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::error::Error;
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
