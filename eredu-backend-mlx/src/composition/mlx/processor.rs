//! MLX tensor conversion for architecture-selected portable media processing.

#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::{
    error::Error,
    submission_recovery::{finish_detached_preparation, Recovery},
};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::input::PreparedInputInspector;
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::input::{
    MediaInputLowering, OptionalProcessorMechanism, ProcessorMechanisms, ProcessorPreparationScope,
};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{InputTensorIdentity, PreparedInputError};
#[cfg(any(feature = "image", feature = "audio"))]
use safemlx::Array;

#[cfg(any(feature = "image", feature = "audio"))]
#[cfg_attr(test, derive(Default))]
struct MlxProcessorMechanisms {
    roots: ProcessorRoots,
}

/// Native constructors are synchronous, but later inspection and observers may
/// submit work using any of their results. Keep all roots in the preparation
/// recovery scope, including roots replaced by an intervention or abandoned on
/// a partial failure. A caller error is not evidence of native completion.
#[cfg(any(feature = "image", feature = "audio"))]
#[derive(Clone, Default)]
struct ProcessorRoots(std::rc::Rc<std::cell::RefCell<Vec<Array>>>);

#[cfg(any(feature = "image", feature = "audio"))]
impl ProcessorRoots {
    fn retain(&self, tensor: Array) -> Array {
        self.0.borrow_mut().push(tensor.clone());
        tensor
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl crate::backend::submission_recovery::Retention for ProcessorRoots {
    fn observe(&self, _: crate::backend::submission_recovery::Status) {}
}

pub(crate) fn capabilities() -> eredu_core::media::MediaPrimitiveCapabilities {
    use eredu_core::media::ProcessorPrimitive;
    use eredu_core::InputModality;

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
    eredu_core::media::MediaPrimitiveCapabilities::new(
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

    fn inspector(&self) -> &dyn PreparedInputInspector<Array> {
        self
    }

    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(self
            .roots
            .retain(Array::from_slice(values, &mlx_shape(shape)?)))
    }

    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(self
            .roots
            .retain(Array::from_slice(values, &mlx_shape(shape)?)))
    }

    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Array, Self::Error> {
        Ok(self
            .roots
            .retain(Array::from_slice(values, &mlx_shape(shape)?)))
    }

    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<Array, OptionalProcessorMechanism<Self::Error>> {
        Ok(self.roots.retain(Array::from_slice(
            values,
            &mlx_shape(shape).map_err(OptionalProcessorMechanism::Backend)?,
        )))
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
fn mlx_shape(shape: &[usize]) -> Result<Vec<i32>, Error> {
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| Error::Processor("tensor element count overflow".into()))?;
    if elements > i32::MAX as usize {
        return Err(Error::Processor(
            "processor tensor element count exceeds i32".into(),
        ));
    }
    shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension)
                .map_err(|_| Error::Processor(format!("tensor dimension {dimension} exceeds i32")))
        })
        .collect()
}

/// MLX tensor conversion and preparation recovery for a selected portable media program.
#[cfg(any(feature = "image", feature = "audio"))]
pub struct MlxProcessorScope {
    mechanisms: MlxProcessorMechanisms,
    recovery: Recovery<ProcessorRoots>,
}

#[cfg(any(feature = "image", feature = "audio"))]
impl MlxProcessorScope {
    pub(crate) fn begin() -> Result<Self, Error> {
        let roots = ProcessorRoots::default();
        let recovery = Recovery::begin(roots.clone())?;
        Ok(Self {
            mechanisms: MlxProcessorMechanisms { roots },
            recovery,
        })
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl ProcessorMechanisms for MlxProcessorScope {
    type Tensor = Array;
    type Error = Error;

    fn inspector(&self) -> &dyn PreparedInputInspector<Array> {
        self.mechanisms.inspector()
    }

    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<Array, Error> {
        self.mechanisms.tensor_u32(values, shape)
    }

    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<Array, Error> {
        self.mechanisms.tensor_f32(values, shape)
    }

    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<Array, Error> {
        self.mechanisms.tensor_i32(values, shape)
    }

    fn tensor_bool(
        &mut self,
        values: &[bool],
        shape: &[usize],
    ) -> Result<Array, OptionalProcessorMechanism<Error>> {
        self.mechanisms.tensor_bool(values, shape)
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl ProcessorPreparationScope for MlxProcessorScope {
    fn finish(self) -> Result<(), Error> {
        finish_detached_preparation(self.recovery, Ok(()), |error| error)
    }
}

/// Supplies native media tensor conversions for an already selected graph.
#[cfg(any(feature = "image", feature = "audio"))]
pub struct MlxMediaInputLowering;

#[cfg(any(feature = "image", feature = "audio"))]
impl MediaInputLowering<crate::MlxTensor> for MlxMediaInputLowering {
    type ProcessorTensor = Array;
    type ProcessorError = Error;
    type Scope = MlxProcessorScope;

    fn begin(&self) -> Result<Self::Scope, Error> {
        MlxProcessorScope::begin()
    }

    fn inspector(&self) -> &impl PreparedInputInspector<crate::MlxTensor> {
        &crate::backend::runtime::media::input::MlxTensorInputInspector
    }

    fn convert_tensor(&self, tensor: &Array) -> crate::MlxTensor {
        crate::MlxTensor::from_array(tensor.clone())
    }
}
