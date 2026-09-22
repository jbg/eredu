//! The existing static selection/flatten worker followed by finite scalar summary.
use super::*;
use crate::backend::array_copy::{CaptureNativePopulation, SummaryProgram};
use eredu_core::capture::{CaptureSummary, CaptureSummaryGeometry};

pub(crate) struct PreparedCaptureSummary {
    selection: Selection,
    numerical: SummaryProgram,
    elements: i32,
}
impl PreparedCaptureSummary {
    pub(crate) fn from_geometry(
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<Self, CaptureTensorNativeError> {
        let mut selection = Selection::reduction(
            geometry.source_shape(),
            geometry.shape(),
            geometry.starts(),
            geometry.ends(),
            geometry.strides(),
        )?;
        selection.read_unsigned = geometry.admission().points()[geometry.selection_index()].dtype
            == eredu_core::ObservationDtype::Integer;
        let elements = i32::try_from(geometry.elements())
            .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
        Ok(Self {
            selection,
            numerical: SummaryProgram::new(elements)?,
            elements,
        })
    }
    pub(crate) fn validate_source(
        &self,
        source: &Array,
    ) -> Result<Dtype, CaptureTensorNativeError> {
        PreparedCaptureTensor::validate_mechanism()?;
        let dtype = source.dtype();
        if !(if self.selection.read_unsigned {
            matches!(dtype, Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32)
        } else {
            matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        }) {
            return Err(CaptureTensorNativeError::UnsupportedDtype(dtype));
        }
        if source.shape() != &self.selection.source_shape[..self.selection.rank] {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        Ok(dtype)
    }
    pub(crate) fn validate_workspace_source(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), CaptureTensorNativeError> {
        self.selection.validate_workspace_source(source, context)
    }
    pub(crate) fn trace(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(), CaptureTensorNativeError> {
        self.validate_workspace_source(source, context)?;
        let mut trace = Trace {
            context,
            retained: Some(retained),
        };
        let chosen = selected(&self.selection, &mut trace, source.clone())?;
        let flat = trace.reshape(chosen, self.elements)?;
        if self.selection.read_unsigned {
            // Exact unsigned host iteration uses the same completed selection;
            // it introduces no reduction graph or tensor payload.
            Ok(())
        } else {
            self.numerical.trace(&flat, context, retained)
        }
    }
    pub(crate) fn execute(
        &self,
        source: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<CaptureSummary, CaptureTensorNativeError> {
        self.validate_source(source)?;
        PreparedCaptureTensor::validate_stream(stream)?;
        self.execute_shared(source, stream, completion, roots, retain)
    }
    /// The enclosing CPU numerical recipe already owns every scalar frontier.
    /// Borrow its exact stream/observer; no fresh completion or source is issued.
    pub(crate) fn execute_cpu(
        &self,
        source: &Array,
        loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<CaptureSummary, CaptureTensorNativeError> {
        self.validate_source(source)?;
        loan.validate()?;
        self.execute_shared(
            source,
            loan.stream(),
            CaptureCompletion::Original(loan.observer()),
            roots,
            retain,
        )
    }
    fn execute_shared(
        &self,
        source: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<CaptureSummary, CaptureTensorNativeError> {
        completion.validate()?;
        // The caller reserves the complete population and holds the source pin.
        let mut native = Native {
            stream,
            roots,
            completion,
        };
        let chosen = selected(
            &self.selection,
            &mut native,
            completion.clone_array(source)?,
        )?;
        let flat = native.reshape(chosen, self.elements)?;
        let evaluated = completion.settle(&flat, stream)?;
        if self.selection.read_unsigned {
            macro_rules! summarize {
                ($ty:ty) => {
                    unsigned_summary(
                        evaluated
                            .try_iter::<$ty>()
                            .map_err(CaptureTensorNativeError::HostRead)?
                            .map(u32::from),
                    )
                };
            }
            return match flat.dtype() {
                Dtype::Uint8 => summarize!(u8),
                Dtype::Uint16 => summarize!(u16),
                Dtype::Uint32 => summarize!(u32),
                other => Err(CaptureTensorNativeError::UnsupportedDtype(other)),
            };
        }
        drop(evaluated);
        self.numerical
            .execute(&flat, stream, completion, retain, &mut |_| {})
    }
    pub(crate) fn population(&self) -> Option<CaptureNativePopulation> {
        if self.selection.read_unsigned {
            return Some(CaptureNativePopulation {
                publications: 1,
                completions: 2,
                retained_roots: 3,
                controls: super::capture_original_control_bytes()?
                    .checked_add(self.control_bytes()?)?,
            });
        }
        let numerical = self.numerical.population().ok()?;
        Some(CaptureNativePopulation {
            publications: 1,
            completions: 2usize.checked_add(numerical.scalar_completions)?,
            // Caller retains source; shared selection and flatten each retain one.
            retained_roots: 3usize.checked_add(numerical.retained_outputs)?,
            controls: super::capture_original_control_bytes()?
                .checked_add(self.control_bytes()?)?,
        })
    }
    pub(crate) fn control_bytes(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<CaptureSummary>(),
            size_of::<(u64, u32, f64, f64, f64)>(),
            safemlx::EvaluatedArray::iteration_control_bytes::<u8>()?
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<u16>()?)
                .max(safemlx::EvaluatedArray::iteration_control_bytes::<u32>()?),
            crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
            size_of::<(
                &Self,
                &Array,
                &Stream,
                CaptureCompletion<'static>,
                &RefCell<Vec<Array>>,
            )>() * 2,
            size_of::<(
                &Self,
                &Array,
                &crate::backend::nn::workspace::CpuCaptureLoan<'static>,
                &RefCell<Vec<Array>>,
            )>(),
            size_of::<Result<CaptureSummary, CaptureTensorNativeError>>(),
            size_of::<Native<'static>>(),
            size_of::<(Array, Array)>(),
            size_of::<Result<CaptureSummary, CaptureTensorNativeError>>(),
            size_of::<Result<Array, CaptureTensorNativeError>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>>(),
        ];
        let clones = safemlx::PreparedArrayClone::control_bytes()?
            .checked_add(Array::inspection_clone_handle_bytes())?;
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(self.numerical.control_bytes()?)?
            .checked_add(Selection::reduction_control_bytes()?)?
            // Selected/flat recovery aliases plus the input by-value handle.
            .checked_add(clones.checked_mul(3)?)?
            .checked_add(
                safemlx::OperationEvent::nested_completion_control_bytes::<1>()?.checked_mul(2)?,
            )
    }
}

/// Every supported unsigned input is exactly representable in f64. Statistics
/// use those actual values directly; expert identities never pass through f32.
fn unsigned_summary(
    values: impl ExactSizeIterator<Item = u32>,
) -> Result<CaptureSummary, CaptureTensorNativeError> {
    let elements =
        u64::try_from(values.len()).map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
    let mut result = CaptureSummary {
        elements,
        finite: elements,
        non_finite: 0,
        nan: 0,
        positive_infinity: 0,
        negative_infinity: 0,
        min: None,
        max: None,
        mean: None,
        rms: None,
    };
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    for value in values {
        let value = f64::from(value);
        result.min = Some(result.min.map_or(value, |old| old.min(value)));
        result.max = Some(result.max.map_or(value, |old| old.max(value)));
        sum += value;
        squares += value * value;
    }
    if elements != 0 {
        result.mean = Some(sum / elements as f64);
        result.rms = Some((squares / elements as f64).sqrt());
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    #[test]
    fn unsigned_summary_preserves_large_expert_identifiers() {
        let result = super::unsigned_summary([16_777_217, u32::MAX - 1].into_iter()).unwrap();
        assert_eq!(result.elements, 2);
        assert_eq!(result.finite, 2);
        assert_eq!(result.min, Some(16_777_217.0));
        assert_eq!(result.max, Some(4_294_967_294.0));
        assert_eq!(result.mean, Some((16_777_217.0 + 4_294_967_294.0) / 2.0));
        assert_eq!(super::unsigned_summary([].into_iter()).unwrap().mean, None);
    }
}
