//! Logical workspace envelope for eager tensor observations.

use std::num::NonZeroU64;

use eredu_core::{
    capability::{NativeExecutionMemory, NativeWorkspaceEstimate},
    capture::{CaptureWorkspace, CaptureWorkspaceModel},
    BackendFailure,
};

use crate::CandleContext;
pub(crate) mod routed;

impl CaptureWorkspace for CandleContext {
    fn capture_workspace() -> CaptureWorkspaceModel {
        CaptureWorkspaceModel {
            max_index: i32::MAX as u64,
            reduction_chunk: NonZeroU64::new(8192).unwrap(),
            source_bytes_per_element: 24,
            selected_bytes_per_element: 16,
            workspace_base_bytes: 4096,
            workspace_bytes_per_element: 96,
            routed: Some(RoutedCaptureWorkspace {
                base_bytes: 4096,
                source_bytes_per_element: 24,
                bytes_per_route: 256,
            }),
        }
    }

    fn estimate_capture(
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate_local_capture(source, selection, slice)
    }
}

impl NativeExecutionMemory<Self> for CandleContext {
    fn refinement_workspace() -> NativeWorkspaceEstimate {
        NativeWorkspaceEstimate {
            lower: 0,
            upper: 16,
            evidence: "native position scores, FP32 readback and soft-feedback upload",
        }
    }

    fn sampling_workspace() -> NativeWorkspaceEstimate {
        NativeWorkspaceEstimate {
            lower: 4,
            upper: 64,
            evidence: "eager FP32 score extraction and host sampling rows, sorting indices, counts, and residual",
        }
    }

    fn placement_memory(
        backend: &Self,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, BackendFailure> {
        eredu_core::capability::NativePlacementMemory::placement_memory(backend)
    }

    fn capture_completes_per_step() -> bool {
        true
    }
}

impl eredu_core::capability::NativePlacementMemory for CandleContext {
    fn placement_memory(
        &self,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, BackendFailure> {
        Ok(crate::memory::placement_memory(self))
    }

    fn scratch_memory_upper_bound(&self) -> Option<u64> {
        None
    }
}

use crate::{text_native::CandleInputInspector, CandleTensor};
use candle_core::{DType, Tensor as NativeTensor};
use eredu_core::{
    capture::{
        host::{estimate_host_transform, transform_f32, transform_scores},
        CaptureBackend, CaptureCapabilities, CaptureError, CapturePayload, CaptureSelection,
        CaptureTokenDomain, CaptureTransform, CaptureTransformKind, CaptureUsage,
        ResolvedCaptureSlice,
    },
    capture::{
        RoutedCaptureWorkspace, RoutedUnitCapture, RoutedUnitCaptureSource, RoutedUnitGeometry,
    },
    checkpoint::TensorDtype,
    input::PreparedInputInspector,
    TensorObservation, TensorObservationData,
};

pub(crate) fn capabilities() -> CaptureCapabilities {
    use CaptureTransformKind::*;
    CaptureCapabilities {
        transformations: vec![
            Preview,
            Slice,
            FullTensor,
            Summary,
            Histogram,
            TopCandidates,
            TokenScores,
            RoutedUnits,
        ],
        max_histogram_bins: 128,
        physical_native_limit: false,
        conditions: vec![
            "eager local capture reserves logical host staging before materialization".into(),
        ],
    }
}

/// Exact local capture tensor selection and bounded host readout.
pub struct CandleCapture<'a> {
    pub(crate) context: &'a CandleContext,
    domain: Option<CaptureTokenDomain<'a>>,
}

impl<'a> CandleCapture<'a> {
    pub(crate) fn new(context: &'a CandleContext, domain: Option<CaptureTokenDomain<'a>>) -> Self {
        Self { context, domain }
    }
}

pub(crate) fn estimate_local_capture(
    source: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    let mut usage =
        estimate_host_transform(CandleContext::capture_workspace(), source, selection, slice)?;
    let strided_indices = slice
        .shape
        .iter()
        .zip(&slice.strides)
        .filter_map(|(width, step)| (*step > 1).then_some(*width))
        .max()
        .unwrap_or(0)
        .checked_mul(4)
        .ok_or(CaptureError::Overflow)?;
    usage.host_bytes = usage
        .host_bytes
        .checked_add(strided_indices)
        .ok_or(CaptureError::Overflow)?;
    Ok(usage)
}

impl CaptureBackend for CandleCapture<'_> {
    type Tensor = CandleTensor;
    type Error = BackendFailure;

    fn estimate_routed_units(
        &self,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        CandleContext::capture_workspace().estimate(shape, selection, slice)
    }

    fn capture_routed_units(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, CandleTensor>,
        geometry: RoutedUnitGeometry,
        slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, BackendFailure>> {
        Some(eredu_core::capture::capture_routed_units(
            &routed::NativeRoutedReadout(self.context),
            source,
            geometry,
            slice,
            None,
        ))
    }

    fn shape(&self, value: &CandleTensor) -> Result<Vec<u64>, BackendFailure> {
        Ok(value.value.dims().iter().map(|size| *size as u64).collect())
    }

    fn source_dtype(&self, value: &CandleTensor) -> Option<TensorDtype> {
        CandleInputInspector
            .identity(value)
            .ok()
            .map(|identity| identity.dtype().clone())
    }

    fn estimate(
        &self,
        tensor: &CandleTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let shape = self
            .shape(tensor)
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        estimate_local_capture(&shape, selection, slice)
    }

    fn transform(
        &mut self,
        tensor: &CandleTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
        score_row: Option<u64>,
    ) -> Result<CapturePayload, BackendFailure> {
        let map = BackendFailure::from_error;
        if matches!(
            selection.transform,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        ) {
            let row = score_row
                .and_then(|row| usize::try_from(row).ok())
                .ok_or_else(|| map(CaptureError::Invalid("no admitted score row".into())))?;
            let vocabulary = *tensor
                .value
                .dims()
                .last()
                .filter(|width| **width != 0)
                .ok_or_else(|| {
                    map(CaptureError::Invalid(
                        "expected logits with a vocabulary axis".into(),
                    ))
                })?;
            let rows = tensor.value.elem_count() / vocabulary;
            let data = tensor
                .value
                .reshape((rows, vocabulary))
                .and_then(|value| value.get(row))
                .and_then(|value| value.to_dtype(DType::F32))
                .and_then(|value| value.to_vec1::<f32>())
                .map_err(BackendFailure::from_error)?;
            return transform_scores(&data, &selection.transform, self.domain)
                .map_err(BackendFailure::from_error);
        }
        let mut selected =
            select(&tensor.value, slice, self.context).map_err(BackendFailure::from_error)?;
        match &selection.transform {
            CaptureTransform::Summary | CaptureTransform::Histogram { .. } => {
                let values = selected
                    .flatten_all()
                    .and_then(|value| value.to_dtype(DType::F32))
                    .and_then(|value| value.to_vec1::<f32>())
                    .map_err(BackendFailure::from_error)?;
                transform_f32(values, &slice.shape, &selection.transform)
                    .map_err(BackendFailure::from_error)
            }
            CaptureTransform::Preview { .. }
            | CaptureTransform::Slice
            | CaptureTransform::FullTensor => {
                let shape = if let CaptureTransform::Preview { max_elements } = selection.transform
                {
                    let len = selected
                        .elem_count()
                        .min(usize::try_from(max_elements).unwrap_or(usize::MAX));
                    selected = selected
                        .flatten_all()
                        .and_then(|value| value.narrow(0, 0, len))
                        .map_err(BackendFailure::from_error)?;
                    vec![len]
                } else {
                    selected.dims().to_vec()
                };
                let data =
                    tensor_data(&selected, tensor.boolean).map_err(BackendFailure::from_error)?;
                TensorObservation::new(shape, data)
                    .map(CapturePayload::Tensor)
                    .map_err(BackendFailure::from_error)
            }
            _ => Err(map(CaptureError::Unsupported(
                "capture transform requires native routed evidence".into(),
            ))),
        }
    }
}

pub(crate) fn select(
    value: &NativeTensor,
    slice: &ResolvedCaptureSlice,
    context: &CandleContext,
) -> candle_core::Result<NativeTensor> {
    let mut value = value.clone();
    for axis in 0..slice.shape.len() {
        let start = slice.starts[axis] as usize;
        let count = (slice.ends[axis] - slice.starts[axis]) as usize;
        if slice.strides[axis] == 1 || count == 0 {
            value = value.narrow(axis, start, count)?;
        } else {
            let indices: Vec<_> = (slice.starts[axis]..slice.ends[axis])
                .step_by(slice.strides[axis] as usize)
                .map(|index| index as u32)
                .collect();
            let shape = indices.len();
            let indices = NativeTensor::from_vec(indices, shape, &context.device)?;
            value = value.index_select(&indices, axis)?;
        }
    }
    Ok(value)
}

fn tensor_data(value: &NativeTensor, boolean: bool) -> candle_core::Result<TensorObservationData> {
    let flat = value.flatten_all()?;
    Ok(match flat.dtype() {
        DType::U8 if boolean => {
            TensorObservationData::Bool(flat.to_vec1::<u8>()?.into_iter().map(|v| v != 0).collect())
        }
        DType::U8 => {
            TensorObservationData::U64(flat.to_vec1::<u8>()?.into_iter().map(u64::from).collect())
        }
        DType::U32 => {
            TensorObservationData::U64(flat.to_vec1::<u32>()?.into_iter().map(u64::from).collect())
        }
        DType::I16 | DType::I32 | DType::I64 => TensorObservationData::I64(
            flat.to_device(&candle_core::Device::Cpu)?
                .to_dtype(DType::I64)?
                .to_vec1::<i64>()?,
        ),
        _ => TensorObservationData::F32(flat.to_dtype(DType::F32)?.to_vec1::<f32>()?),
    })
}
