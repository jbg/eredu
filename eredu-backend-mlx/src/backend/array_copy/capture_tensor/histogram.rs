//! The existing static selection/flatten worker followed by fixed-edge histogram counts.
use super::*;
use crate::backend::array_copy::HistogramTotals;
use crate::backend::array_copy::{CaptureNativePopulation, HistogramProgram};
use eredu_core::capture::CaptureHistogramGeometry;

pub(crate) struct PreparedCaptureHistogram<'a> {
    selection: Selection,
    numerical: HistogramProgram<'a>,
    elements: i32,
}
impl<'a> PreparedCaptureHistogram<'a> {
    pub(crate) fn from_geometry(
        geometry: &CaptureHistogramGeometry<'a>,
    ) -> Result<Self, CaptureTensorNativeError> {
        let selection = Selection::reduction(
            geometry.source_shape(),
            geometry.shape(),
            geometry.starts(),
            geometry.ends(),
            geometry.strides(),
        )?;
        let elements = i32::try_from(geometry.elements())
            .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
        Ok(Self {
            selection,
            numerical: HistogramProgram::new(elements, geometry.edges())?,
            elements,
        })
    }
    pub(crate) fn validate_source(
        &self,
        source: &Array,
    ) -> Result<Dtype, CaptureTensorNativeError> {
        PreparedCaptureTensor::validate_mechanism()?;
        let dtype = source.dtype();
        if !matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16) {
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
        self.numerical.trace(&flat, context, retained)
    }
    pub(crate) fn execute(
        &self,
        source: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
        record: &mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<HistogramTotals, CaptureTensorNativeError> {
        self.validate_source(source)?;
        PreparedCaptureTensor::validate_stream(stream)?;
        self.execute_shared(source, stream, completion, roots, retain, record)
    }
    /// The exact numerical recipe owns this stream, edge source and each read.
    pub(crate) fn execute_cpu(
        &self, source: &Array, loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
        record: &mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<HistogramTotals, CaptureTensorNativeError> {
        self.validate_source(source)?;
        loan.validate()?;
        self.execute_shared(source, loan.stream(), CaptureCompletion::Original(loan.observer()), roots, retain, record)
    }
    fn execute_shared(
        &self, source: &Array, stream: &Stream, completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        retain: &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
        record: &mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>,
    ) -> Result<HistogramTotals, CaptureTensorNativeError> {
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
        drop(completion.settle(&flat, stream)?);
        self.numerical
            .execute(&flat, stream, completion, retain, &mut |_| {}, record)
    }
    pub(crate) fn population(&self) -> Option<CaptureNativePopulation> {
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
            crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
            size_of::<(&Self, &Array, &Stream, CaptureCompletion<'static>, &RefCell<Vec<Array>>)>() * 2,
            size_of::<(&Self, &Array, &crate::backend::nn::workspace::CpuCaptureLoan<'static>, &RefCell<Vec<Array>>)>(),
            size_of::<Result<HistogramTotals, CaptureTensorNativeError>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>>() * 2,
            size_of::<&mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>>() * 2,
            size_of::<Native<'static>>(),
            size_of::<(Array, Array)>(),
            size_of::<Result<HistogramTotals, CaptureTensorNativeError>>(),
            size_of::<Result<Array, CaptureTensorNativeError>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>>(),
            size_of::<&mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>>(),
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
