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
#[derive(Default)]
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

#[cfg(all(test, feature = "image"))]
mod processor_tests {
    use super::*;
    use eredu_runtime::processor_resources::ProcessorResourceError;

    struct Admit;
    impl eredu_architectures::processor_execution::ProcessorInputAdmission for Admit {
        fn validate<T>(
            &self,
            _: &eredu_runtime::PreparedModelInput<T>,
            _: &impl PreparedInputInspector<T>,
        ) -> Result<(), eredu_architectures::processor_execution::ProcessorInputAdmissionError>
        {
            Ok(())
        }
    }

    #[test]
    fn optional_coarse_budget_preserves_exact_integer_copy() {
        use eredu_runtime::processor_resources::ProcessorRequestBudget;
        let processor = PreparedProcessor::from_qwen(
            eredu_architectures::processor_plan::QwenProcessorPlan::tokens_only(),
        );
        let budget = ProcessorRequestBudget {
            decoded_input_bytes: 12,
            host_buffer_bytes: 1024,
            output_tensor_bytes: 12,
            planning_items: 128,
            decoder_positions: 3,
        };
        let constructed = std::cell::Cell::new(0);
        let prepare = |budget| {
            let request =
                eredu_core::MultimodalRequest::new(vec![eredu_core::MultimodalSegment::TokenIds(
                    vec![3, 16_777_217, i32::MAX as u32],
                )])
                .unwrap()
                .tokenize::<String>(|_| unreachable!())
                .unwrap();
            let mut mechanisms = MlxProcessorMechanisms::default();
            let result = processor.prepare_with_admission(
                &request,
                &mut mechanisms,
                &mut |_| Ok::<_, String>(vec![]),
                &Admit,
                budget,
            );
            constructed.set(mechanisms.roots.0.borrow().len());
            result
        };
        let prepared = prepare(budget).unwrap();
        assert_eq!(constructed.get(), 1);
        assert_eq!(prepared.resources().output_tensor_bytes, 12);
        assert_eq!(
            prepared.parts()[0]
                .payload()
                .value()
                .evaluated()
                .unwrap()
                .as_slice::<u32>(),
            &[3, 16_777_217, i32::MAX as u32]
        );
        let mut denied = budget;
        denied.output_tensor_bytes = 11;
        assert!(matches!(
            prepare(denied),
            Err(ProcessorExecutionError::Resources(
                ProcessorResourceError::Exhausted { .. }
            ))
        ));
        assert_eq!(
            constructed.get(),
            0,
            "coarse output rejection precedes native constructors"
        );
    }

    #[test]
    fn partial_preparation_keeps_constructed_and_replaced_roots_until_terminal() {
        use crate::backend::submission_recovery::{Probe, Recovery, Status};
        use std::{cell::Cell, rc::Rc};

        struct Pending(Rc<Cell<Status>>);
        impl Probe for Pending {
            fn seal(&mut self) {}
            fn progress(&self) -> Status {
                self.0.get()
            }
        }
        struct Replace;
        impl eredu_runtime::ActivationObserver<Array, safemlx::error::Exception> for Replace {
            fn observe(&mut self, _: &str, _: &Array) -> Result<(), safemlx::error::Exception> {
                Ok(())
            }
            fn intervene(
                &mut self,
                _: &str,
                _: &Array,
            ) -> Result<Option<Array>, safemlx::error::Exception> {
                Ok(Some(Array::from_slice(&[41_u32, 42], &[1, 2])))
            }
        }

        // Both an error result and an unwind can abandon a partially prepared
        // request while submitted native work is still unobservable.
        for unwind in [false, true] {
            let roots = ProcessorRoots::default();
            let retained = Rc::downgrade(&roots.0);
            let status = Rc::new(Cell::new(Status {
                settled: false,
                failed: !unwind,
                blocked: unwind,
            }));
            let recovery = Recovery::with_probe(roots.clone(), Pending(Rc::clone(&status)));
            let operation = || {
                let _recovery = recovery;
                let mut mechanisms = MlxProcessorMechanisms {
                    roots: roots.clone(),
                };
                let source = mechanisms.tensor_u32(&[16_777_217, 7], &[1, 2]).unwrap();
                let mut observer = ProcessorObserver::<String> {
                    inner: &mut Replace,
                    roots: roots.clone(),
                    marker: std::marker::PhantomData,
                };
                use eredu_runtime::ActivationObserver;
                let replacement = observer
                    .intervene("processor.test", &source)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    replacement.evaluated().unwrap().as_slice::<u32>(),
                    &[41, 42]
                );
                assert_eq!(roots.0.borrow().len(), 2);
                if unwind {
                    panic!("injected partial preparation unwind");
                }
                Err::<(), _>("injected later preparation failure")
            };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
            if unwind {
                assert!(outcome.is_err());
            } else {
                assert_eq!(outcome.unwrap(), Err("injected later preparation failure"));
            }
            drop(roots);
            let values = retained
                .upgrade()
                .expect("unresolved work retains source roots");
            assert_eq!(values.borrow().len(), 2);
            assert_eq!(
                values.borrow()[0].evaluated().unwrap().as_slice::<u32>(),
                &[16_777_217, 7]
            );
            drop(values);
            status.set(Status {
                settled: true,
                failed: true,
                blocked: false,
            });
            crate::backend::submission_recovery::wait_for_retirement(|| {
                retained.upgrade().is_none()
            });
        }
    }

    #[test]
    fn constructor_geometry_rejects_overflow_without_allocating() {
        assert_eq!(mlx_shape(&[2, 3]).unwrap(), [2, 3]);
        assert!(mlx_shape(&[65536, 65536]).is_err());
        assert!(mlx_shape(&[usize::MAX, 2]).is_err());
        assert!(mlx_shape(&[0, usize::MAX]).is_err());
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl ModelProcessor {
    /// Wraps a processor already prepared by the architecture construction driver.
    pub(crate) fn from_prepared(processor: PreparedProcessor) -> Self {
        Self { processor }
    }

    /// Lowers the authoritative architecture-owned processor plan to MLX execution.
    #[cfg(test)]
    pub fn from_plan(
        plan: &eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    ) -> Option<Self> {
        PreparedProcessor::from_artifact(plan).map(|processor| Self { processor })
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
        let roots = ProcessorRoots::default();
        crate::backend::submission_recovery::detached_preparation(
            roots.clone(),
            move || {
                let mut mechanisms = MlxProcessorMechanisms {
                    roots: roots.clone(),
                };
                let mut observer = ProcessorObserver::<E> {
                    inner: observer,
                    roots: roots.clone(),
                    marker: std::marker::PhantomData,
                };
                self.processor
                    .prepare_with_observer(request, &mut mechanisms, encode_text, &mut observer)
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
                        ProcessorExecutionError::Text(error) => {
                            ProcessorPreparationError::Text(error)
                        }
                        ProcessorExecutionError::Resources(error) => {
                            ProcessorPreparationError::Backend(Error::ProcessorResources(error))
                        }
                        ProcessorExecutionError::Admission(error) => {
                            ProcessorPreparationError::Backend(Error::ProcessorAdmission(error))
                        }
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
            },
            ProcessorPreparationError::Backend,
        )
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
struct ProcessorObserver<'a, E> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    roots: ProcessorRoots,
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
        self.inner
            .observe(path, value)
            .map_err(|error| ProcessorExecutionError::Mechanism(Error::from(error)))
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, ProcessorExecutionError<E, Error>> {
        self.inner
            .intervene(path, value)
            .map(|replacement| replacement.map(|tensor| self.roots.retain(tensor)))
            .map_err(|error| ProcessorExecutionError::Mechanism(Error::from(error)))
    }
}
