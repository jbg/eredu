//! Exact eager activation mutation and its admission costs.

use candle_core::{DType, Tensor as NativeTensor};
use eredu_core::{
    backend::local_text::NoTextTransport,
    capture::{
        add, mul, CaptureError, CaptureSelection, CaptureUsage, CaptureWorkspace,
        CaptureWorkspaceModel, ResolvedCaptureSlice, RoutedUnitCaptureSource, RoutedUnitGeometry,
        SpeculativeCapturePrimitives,
    },
    intervention::{
        InterventionAction, InterventionBackend, InterventionDtype, InterventionEstimator,
        InterventionKind, InterventionMechanisms, InterventionRoutingPolicy, InterventionTensor,
        InterventionValues, RoutedUnitLocations, RoutingScoreStage,
    },
    BackendFailure,
};
use eredu_nn::Tensor;

use crate::{capture::CandleCapture, text_native::CandleTextValues, CandleContext, CandleTensor};

fn invalid(detail: &str) -> BackendFailure {
    BackendFailure::from_error(CaptureError::Invalid(detail.into()))
}

fn dtype(dtype: InterventionDtype) -> DType {
    match dtype {
        InterventionDtype::Float32 => DType::F32,
        InterventionDtype::Float16 => DType::F16,
        InterventionDtype::Bfloat16 => DType::BF16,
    }
}

fn shape(shape: &[u64]) -> Result<Vec<usize>, BackendFailure> {
    shape
        .iter()
        .map(|&length| usize::try_from(length).map_err(BackendFailure::from_error))
        .collect()
}

fn wrap(value: candle_core::Result<NativeTensor>) -> Result<CandleTensor, BackendFailure> {
    CandleTensor::wrap(value).map_err(BackendFailure::from_error)
}

impl InterventionBackend for CandleCapture<'_> {
    fn routed_unit_locations(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, CandleTensor>,
        geometry: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, BackendFailure>> {
        Some(eredu_core::capture::routed_unit_locations(
            &crate::capture::routed::NativeRoutedReadout(self.context),
            source,
            geometry,
            None,
        ))
    }

    fn select_elements(
        &mut self,
        value: &CandleTensor,
        indices: &[u64],
    ) -> Option<Result<CandleTensor, BackendFailure>> {
        Some(
            crate::capture::routed::select_elements(value, indices, self.context)
                .and_then(|value| wrap(Ok(value))),
        )
    }

    fn update_elements(
        &mut self,
        value: &CandleTensor,
        indices: &[u64],
        replacement: &CandleTensor,
    ) -> Option<Result<CandleTensor, BackendFailure>> {
        Some((|| {
            if replacement.value.dims() != [indices.len()]
                || value.value.dtype() != replacement.value.dtype()
            {
                return Err(invalid("activation replacement geometry or dtype differs"));
            }
            let indices = crate::capture::routed::native_indices(indices, self.context)?;
            wrap(
                value
                    .value
                    .flatten_all()
                    .and_then(|flat| flat.scatter(&indices, &replacement.value, 0))
                    .and_then(|flat| flat.reshape(value.value.shape())),
            )
        })())
    }

    fn intervention_dtype(
        &self,
        tensor: &CandleTensor,
    ) -> Result<InterventionDtype, BackendFailure> {
        match tensor.value.dtype() {
            DType::F32 => Ok(InterventionDtype::Float32),
            DType::F16 => Ok(InterventionDtype::Float16),
            DType::BF16 => Ok(InterventionDtype::Bfloat16),
            _ => Err(invalid("activation editing requires a floating tensor")),
        }
    }

    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        InterventionCosts.validate_geometry(source, slice)
    }

    fn select_region(
        &mut self,
        value: &CandleTensor,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CandleTensor, BackendFailure> {
        wrap(crate::capture::select(&value.value, slice, self.context))
    }

    fn update_region(
        &mut self,
        value: &CandleTensor,
        slice: &ResolvedCaptureSlice,
        replacement: &CandleTensor,
    ) -> Result<CandleTensor, BackendFailure> {
        let extents = shape(&slice.shape)?;
        let source = value.value.dims();
        let geometry = source
            .iter()
            .map(|&extent| extent as u64)
            .collect::<Vec<_>>();
        InterventionCosts
            .validate_geometry(&geometry, slice)
            .map_err(BackendFailure::from_error)?;
        if replacement.value.dims() != extents || value.value.dtype() != replacement.value.dtype() {
            return Err(invalid("activation replacement geometry or dtype differs"));
        }
        let index = (0..replacement.value.elem_count())
            .map(|mut linear| {
                let mut index = 0_usize;
                let mut stride = 1_usize;
                for axis in (0..source.len()).rev() {
                    let coordinate = linear % extents[axis];
                    linear /= extents[axis];
                    index += (slice.starts[axis] as usize
                        + coordinate * slice.strides[axis] as usize)
                        * stride;
                    stride *= source[axis];
                }
                index as u32
            })
            .collect::<Vec<_>>();
        wrap((|| {
            let indices = NativeTensor::from_slice(&index, index.len(), &self.context.device)?;
            value
                .value
                .flatten_all()?
                .scatter(&indices, &replacement.value.flatten_all()?, 0)?
                .reshape(source)
        })())
    }

    fn zeros(
        &mut self,
        extents: &[u64],
        precision: InterventionDtype,
    ) -> Result<CandleTensor, BackendFailure> {
        wrap(NativeTensor::zeros(
            shape(extents)?,
            dtype(precision),
            &self.context.device,
        ))
    }

    fn scale(&mut self, value: &CandleTensor, factor: f32) -> Result<CandleTensor, BackendFailure> {
        let scalar = wrap(
            NativeTensor::new(factor, &self.context.device)
                .and_then(|scalar| scalar.to_dtype(value.value.dtype())),
        )?;
        value
            .multiply(&scalar, self.context)
            .map_err(BackendFailure::from_error)
    }

    fn fill_masked(
        &mut self,
        value: &CandleTensor,
        keep: &[bool],
        fill: f32,
    ) -> Result<CandleTensor, BackendFailure> {
        if keep.len() != value.value.elem_count() {
            return Err(invalid("activation mask extent differs"));
        }
        wrap((|| {
            let mask: Vec<_> = keep.iter().map(|&keep| u8::from(keep)).collect();
            let mask = NativeTensor::from_vec(mask, value.value.shape(), &self.context.device)?;
            let fill = NativeTensor::new(fill, &self.context.device)?
                .to_dtype(value.value.dtype())?
                .broadcast_as(value.value.shape())?;
            mask.where_cond(&value.value, &fill)
        })())
    }

    fn mask_components(
        &mut self,
        value: &CandleTensor,
        indices: &[u32],
        keep_selected: bool,
    ) -> Result<CandleTensor, BackendFailure> {
        fill_components(value, indices, keep_selected, 0., self.context)
    }

    fn realize_tensor(
        &mut self,
        tensor: &InterventionTensor,
    ) -> Result<CandleTensor, BackendFailure> {
        let shape = shape(&tensor.shape)?;
        let device = &self.context.device;
        wrap(match &tensor.values {
            InterventionValues::Float32(values) => NativeTensor::from_slice(values, shape, device),
            InterventionValues::Float16(values) => NativeTensor::from_vec(
                values.iter().copied().map(half::f16::from_bits).collect(),
                shape,
                device,
            ),
            InterventionValues::Bfloat16(values) => NativeTensor::from_vec(
                values.iter().copied().map(half::bf16::from_bits).collect(),
                shape,
                device,
            ),
        })
    }

    fn add(
        &mut self,
        left: &CandleTensor,
        right: &CandleTensor,
    ) -> Result<CandleTensor, BackendFailure> {
        if left.value.shape() != right.value.shape() || left.value.dtype() != right.value.dtype() {
            return Err(invalid(
                "activation addition requires identical shape and dtype",
            ));
        }
        left.add(right, self.context)
            .map_err(BackendFailure::from_error)
    }

    fn fill_columns(
        &mut self,
        value: &CandleTensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<CandleTensor, BackendFailure> {
        fill_components(value, ids, false, fill, self.context)
    }
}

fn fill_components(
    value: &CandleTensor,
    indices: &[u32],
    keep_selected: bool,
    fill: f32,
    context: &CandleContext,
) -> Result<CandleTensor, BackendFailure> {
    let width = *value
        .value
        .dims()
        .last()
        .ok_or_else(|| invalid("missing component axis"))?;
    let mut mask = vec![u8::from(!keep_selected); width];
    for &index in indices {
        *mask
            .get_mut(index as usize)
            .ok_or_else(|| invalid("component outside activation"))? = u8::from(keep_selected);
    }
    wrap((|| {
        let mask = NativeTensor::from_vec(mask, width, &context.device)?
            .broadcast_as(value.value.shape())?;
        let fill = NativeTensor::new(fill, &context.device)?
            .to_dtype(value.value.dtype())?
            .broadcast_as(value.value.shape())?;
        mask.where_cond(&value.value, &fill)
    })())
}

struct InterventionCosts;

fn elements(shape: &[u64]) -> Result<u64, CaptureError> {
    shape.iter().try_fold(1_u64, |elements, extent| {
        elements.checked_mul(*extent).ok_or(CaptureError::Overflow)
    })
}

impl InterventionEstimator for InterventionCosts {
    fn routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        action: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        eredu_core::intervention::estimate_local_routed_unit_edit(geometry, source, slice, action)
    }

    fn validate_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        if elements(source)? > i32::MAX as u64
            || source.len() != slice.shape.len()
            || source.iter().any(|&extent| extent > i32::MAX as u64)
        {
            return Err(CaptureError::Unsupported(
                "activation exceeds eager native index extent".into(),
            ));
        }
        if [slice.starts.len(), slice.ends.len(), slice.strides.len()]
            .into_iter()
            .any(|rank| rank != source.len())
            || (0..source.len()).any(|axis| {
                let start = slice.starts[axis];
                let end = slice.ends[axis];
                let stride = slice.strides[axis];
                stride == 0
                    || start > end
                    || end > source[axis]
                    || (end - start).div_ceil(stride) != slice.shape[axis]
            })
        {
            return Err(CaptureError::Invalid(
                "activation region differs from source geometry".into(),
            ));
        }
        Ok(())
    }

    fn activation_usage(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        self.validate_geometry(source, slice)?;
        let source = elements(source)?;
        let region = elements(&slice.shape)?;
        let retained_bytes = source
            .checked_mul(24)
            .and_then(|bytes| bytes.checked_add(region.checked_mul(32)?))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or(CaptureError::Overflow)?;
        let host_bytes = region.checked_mul(16).ok_or(CaptureError::Overflow)?;
        Ok(CaptureUsage {
            retained_bytes,
            host_bytes,
            ..CaptureUsage::default()
        })
    }

    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::capture::estimate_local_capture(source, selection, slice)
    }

    fn original_route_usage(
        &self,
        policy: &InterventionRoutingPolicy,
        rows: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        let scores = mul(rows, u64::from(policy.expert_count))?;
        let routes = mul(rows, u64::from(policy.top_k))?;
        if scores > i32::MAX as u64 {
            return Err(CaptureError::Unsupported(
                "routing scores exceed eager native index extent".into(),
            ));
        }
        Ok(CaptureUsage {
            retained_bytes: add(4096, add(mul(scores, 40)?, mul(routes, 64)?)?)?,
            host_bytes: add(4096, add(mul(scores, 64)?, mul(routes, 128)?)?)?,
            ..CaptureUsage::default()
        })
    }
}

impl CaptureWorkspace for CandleTextValues {
    fn capture_workspace() -> CaptureWorkspaceModel {
        CandleContext::capture_workspace()
    }

    fn estimate_capture(
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::capture::estimate_local_capture(source, selection, slice)
    }
}

impl SpeculativeCapturePrimitives for CandleTextValues {
    type Queues = CandleContext;
    type Transport = NoTextTransport;
    type InvocationTensor = CandleTensor;
    type ObserverError = BackendFailure;
    type Scores = CandleTensor;
    type Tensor = CandleTensor;
    type Error = BackendFailure;
    type InvocationError = BackendFailure;
    type Backend<'a> = CandleCapture<'a>;

    fn capture_backend<'a>(
        queue: &'a CandleContext,
        _: Option<&'a NoTextTransport>,
    ) -> Self::Backend<'a> {
        CandleCapture::new(queue, None)
    }

    fn intervention_mechanisms() -> InterventionMechanisms {
        use InterventionDtype as D;
        use InterventionKind as K;
        InterventionMechanisms {
            operations: vec![
                K::Zero,
                K::Scale,
                K::Mask,
                K::MaskComponents,
                K::Replace,
                K::Add,
                K::MaskLogits,
                K::ExcludeExperts,
                K::ZeroExpertContribution,
                K::BiasRoutingScores,
                K::ForceExperts,
            ],
            dtypes: vec![D::Float32, D::Float16, D::Bfloat16],
            routed_units: true,
            score_stages: vec![
                RoutingScoreStage::RawLogits,
                RoutingScoreStage::TransformedScores,
                RoutingScoreStage::RankingScores,
            ],
        }
    }

    fn intervention_estimator() -> std::sync::Arc<dyn InterventionEstimator> {
        std::sync::Arc::new(InterventionCosts)
    }

    fn generated_creation_bytes(shape: &[u64]) -> Result<u64, CaptureError> {
        elements(shape)?
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or(CaptureError::Overflow)
    }

    fn observation_failure(error: BackendFailure) -> BackendFailure {
        error
    }

    fn invocation_failure(error: &dyn std::error::Error) -> BackendFailure {
        BackendFailure::from_error(std::io::Error::other(error.to_string()))
    }
}
