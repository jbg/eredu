//! Generated observations enter the ordinary reservation/transform driver lazily.
use super::*;

/// Converts neural source facts into capture metadata without materializing a
/// tensor or inferring precision from a geometry-only prototype.
pub fn generated_capture_source(
    source: &eredu_nn::GeneratedTensorSource,
) -> GeneratedCaptureSource {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_nn::TensorElementType;
    GeneratedCaptureSource {
        creation_bytes: source.creation_bytes,
        source_dtype: source.element_type.map(|element_type| match element_type {
            TensorElementType::Bool => TensorDtype::Bool,
            TensorElementType::F16 => TensorDtype::F16,
            TensorElementType::Bf16 => TensorDtype::Bf16,
            TensorElementType::F32 => TensorDtype::F32,
            TensorElementType::F64 => TensorDtype::F64,
            TensorElementType::I8 => TensorDtype::I8,
            TensorElementType::I16 => TensorDtype::I16,
            TensorElementType::I32 => TensorDtype::I32,
            TensorElementType::I64 => TensorDtype::I64,
            TensorElementType::U8 => TensorDtype::U8,
            TensorElementType::U16 => TensorDtype::U16,
            TensorElementType::U32 => TensorDtype::U32,
            TensorElementType::U64 => TensorDtype::U64,
            TensorElementType::Complex64 => TensorDtype::Complex64,
        }),
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum GeneratedFailure<E: std::error::Error + 'static> {
    #[error(transparent)]
    Backend(E),
    #[error("observation generation failed")]
    Generation,
    #[error("generated observation changed its declared geometry")]
    Geometry,
    #[error("generated observation changed its declared element type")]
    Precision,
    #[error(transparent)]
    Admission(CaptureError),
}

pub(super) struct GeneratedCapture<'a, B: CaptureBackend> {
    backend: &'a mut B,
    generate: &'a mut dyn FnMut() -> Result<B::Tensor, GeneratedFailure<B::Error>>,
    source: &'a GeneratedCaptureSource,
    value: Option<B::Tensor>,
}

impl<B: CaptureBackend> CaptureBackend for GeneratedCapture<'_, B> {
    type Tensor = B::Tensor;
    type Error = GeneratedFailure<B::Error>;
    fn source_dtype(&self, _: &Self::Tensor) -> Option<eredu_core::checkpoint::TensorDtype> {
        match &self.value {
            Some(value) => self.backend.source_dtype(value),
            None => self.source.source_dtype.clone(),
        }
    }
    fn shape(&self, prototype: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
        self.backend
            .shape(prototype)
            .map_err(GeneratedFailure::Backend)
    }
    fn estimate(
        &self,
        prototype: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        self.backend
            .estimate_generated(prototype, self.source, selection, slice)?
            .checked_add(CaptureUsage {
                retained_bytes: self.source.creation_bytes,
                ..Default::default()
            })
    }
    fn transform(
        &mut self,
        prototype: &Self::Tensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        if elements(&slice.shape).map_err(GeneratedFailure::Admission)? == 0 {
            return super::empty_payload(
                &selection.transform,
                &slice.shape,
                self.source_dtype(prototype).as_ref(),
            )
            .map_err(GeneratedFailure::Admission);
        }
        // CaptureSession calls transform only after reservation. The factory and
        // its native graph are never created for absent/skipped/over-budget points.
        if self.value.is_none() {
            self.value = Some((self.generate)()?);
            let value = self.value.as_ref().expect("generated value");
            if self
                .backend
                .shape(value)
                .map_err(GeneratedFailure::Backend)?
                != self
                    .backend
                    .shape(prototype)
                    .map_err(GeneratedFailure::Backend)?
            {
                return Err(GeneratedFailure::Geometry);
            }
            let valid_dtype =
                self.source.source_dtype.as_ref().is_none_or(|declared| {
                    self.backend.source_dtype(value).as_ref() == Some(declared)
                });
            if !valid_dtype {
                return Err(GeneratedFailure::Precision);
            }
        }
        self.backend
            .transform(
                self.value.as_ref().expect("generated value"),
                selection,
                slice,
            )
            .map_err(GeneratedFailure::Backend)
    }
}

impl CaptureSession {
    /// Uses the same scheduled record, ledger and synchronous transformation path
    /// as an existing activation. Factory errors retain the observer's error type.
    /// Temporary native values are dropped before returning to the forward driver.
    pub(crate) fn observe_generated<B: CaptureBackend, E>(
        &mut self,
        backend: &mut B,
        path: &str,
        prototype: &B::Tensor,
        source: &GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
        map_error: &impl Fn(CaptureExecutionError<B::Error>) -> E,
    ) -> Result<(), E> {
        capture_generated(backend, source, generate, map_error, |generated| {
            self.observe_classified(generated, path, prototype, classify_generated_failure)
        })
    }
}

/// Keeps deferred construction and its original error in one shared adapter.
/// Both ordinary observations and partition fragments enter the same reservation
/// path before invoking the factory; the prototype supplies geometry only.
pub(super) fn capture_generated<B: CaptureBackend, E, T>(
    backend: &mut B,
    source: &GeneratedCaptureSource,
    generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    map_error: &impl Fn(CaptureExecutionError<B::Error>) -> E,
    capture: impl FnOnce(
        &mut GeneratedCapture<'_, B>,
    ) -> Result<T, CaptureExecutionError<GeneratedFailure<B::Error>>>,
) -> Result<T, E> {
    let mut generation_failure = None;
    let result = capture(&mut GeneratedCapture {
        backend,
        generate: &mut || {
            generate().map_err(|error| {
                generation_failure = Some(error);
                GeneratedFailure::Generation
            })
        },
        source,
        value: None,
    });
    if let Some(error) = generation_failure {
        return Err(error);
    }
    result.map_err(|error| {
        map_error(match classify_generated_failure(error) {
            CaptureExecutionError::Admission(error) => error.into(),
            CaptureExecutionError::Backend(GeneratedFailure::Backend(error)) => {
                CaptureExecutionError::Backend(error)
            }
            CaptureExecutionError::Backend(
                GeneratedFailure::Geometry
                | GeneratedFailure::Precision
                | GeneratedFailure::Admission(_),
            ) => unreachable!("portable generated failure classified above"),
            CaptureExecutionError::Backend(GeneratedFailure::Generation) => {
                unreachable!("factory error retained above")
            }
        })
    })
}

// The host record and returned error must describe the same failure category.
fn classify_generated_failure<E: std::error::Error + 'static>(
    error: CaptureExecutionError<GeneratedFailure<E>>,
) -> CaptureExecutionError<GeneratedFailure<E>> {
    match error {
        CaptureExecutionError::Backend(GeneratedFailure::Geometry) => {
            CaptureError::Invalid("generated observation changed its declared geometry".into())
                .into()
        }
        CaptureExecutionError::Backend(GeneratedFailure::Precision) => {
            CaptureError::Invalid("generated observation changed its declared element type".into())
                .into()
        }
        CaptureExecutionError::Backend(GeneratedFailure::Admission(error)) => error.into(),
        error => error,
    }
}
