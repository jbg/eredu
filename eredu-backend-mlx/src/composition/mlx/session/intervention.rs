//! Native realization of validated host-only activation plans.

use super::bounded_capture::NativeCapture;
use super::*;
use eredu_core::{capture::*, intervention::*};
use safemlx::ops::indexing::{ArrayIndex, IntoStrideBy, TryIndexMutOp};

mod routed;
#[cfg(test)]
mod tests;

pub(crate) fn mechanisms() -> InterventionMechanisms {
    InterventionMechanisms {
        routed_units: true,
        operations: vec![
            InterventionKind::Zero,
            InterventionKind::Scale,
            InterventionKind::Mask,
            InterventionKind::MaskComponents,
            InterventionKind::Replace,
            InterventionKind::Add,
            InterventionKind::MaskLogits,
            InterventionKind::ExcludeExperts,
            InterventionKind::ZeroExpertContribution,
            InterventionKind::BiasRoutingScores,
            InterventionKind::ForceExperts,
        ],
        dtypes: vec![
            InterventionDtype::Float32,
            InterventionDtype::Float16,
            InterventionDtype::Bfloat16,
        ],
        score_stages: vec![
            RoutingScoreStage::RawLogits,
            RoutingScoreStage::TransformedScores,
            RoutingScoreStage::RankingScores,
        ],
    }
}

/// Cold MLX bounds: the same estimator is used at admission and before execution.
pub(crate) struct NativeInterventionEstimator;
impl InterventionEstimator for NativeInterventionEstimator {
    fn partition_routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        source_tokens: u64,
        ownership: &RoutedUnitCaptureOwnership,
        slice: &ResolvedCaptureSlice,
        action: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        ownership.validate(geometry)?;
        action.validate_activation_region(
            action
                .dtype()
                .ok_or_else(|| CaptureError::Invalid("sparse routing action".into()))?,
            &slice.shape,
        )?;
        let rows = ownership.maximum_source_rows(source_tokens, geometry.routes_per_token)?;
        let routes = mul(
            rows,
            if ownership.source_peer.is_some() {
                1
            } else {
                geometry.routes_per_token
            },
        )?;
        let units = ownership.coordinates.units().local_count() as u64;
        if rows > i32::MAX as u64 || mul(routes, units)? > i32::MAX as u64 {
            return Err(CaptureError::Unsupported(
                "MLX sparse partition edit exceeds signed indexing".into(),
            ));
        }
        routed::usage(routes, units, action)
    }
    fn routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        action: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        if source.len() != 2 || source[1] != geometry.components()? {
            return Err(CaptureError::Invalid(
                "invalid sparse edit virtual shape".into(),
            ));
        }
        // The virtual expert axis is never allocated. Native indexing applies to
        // participating values; lowering and exact payload gathers are host work.
        let routes = mul(source[0], geometry.routes_per_token)?;
        let values = mul(routes, geometry.units_per_expert)?;
        if values > i32::MAX as u64 || source[0] > i32::MAX as u64 {
            return Err(CaptureError::Unsupported(
                "MLX sparse edit indexing exceeds i32".into(),
            ));
        }
        action.validate_activation_region(
            action
                .dtype()
                .ok_or_else(|| CaptureError::Invalid("sparse routing action".into()))?,
            &slice.shape,
        )?;
        routed::usage(routes, geometry.units_per_expert, action)
    }

    fn validate_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        if elements(source)? > i32::MAX as u64
            || source
                .iter()
                .chain(&slice.starts)
                .chain(&slice.ends)
                .chain(&slice.strides)
                .any(|v| *v > i32::MAX as u64)
        {
            return Err(CaptureError::Unsupported(
                "MLX intervention geometry exceeds signed 32-bit indexing".into(),
            ));
        }
        Ok(())
    }
    fn activation_usage(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        action: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        // Source/backing retention plus two full update arrays; selected views,
        // index/update/where arithmetic and possible dtype upload temporaries.
        let source_elements = elements(source)?;
        let selected = elements(&slice.shape)?;
        let width = *source
            .last()
            .ok_or_else(|| CaptureError::Invalid("scalar activation edit".into()))?;
        let payload = match action {
            InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
                mul(tensor.values.len() as u64, 4)?
            }
            InterventionAction::Mask { keep, .. } => keep.len() as u64,
            InterventionAction::MaskComponents { .. } | InterventionAction::MaskLogits { .. } => {
                width
            }
            _ => 0,
        };
        Ok(CaptureUsage {
            captures: 0,
            retained_bytes: add(
                4096,
                add(
                    mul(source_elements, 16)?,
                    add(mul(selected, 32)?, mul(payload, 2)?)?,
                )?,
            )?,
            host_bytes: add(payload, mul(source.len() as u64, 32)?)?,
            encoded_bytes: 0,
        })
    }
    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        super::bounded_capture::estimate_shape(source, selection, slice)
    }
    fn original_route_usage(
        &self,
        policy: &InterventionRoutingPolicy,
        rows: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        // Upper bounds for the logical arrays in topk_indices/weights_for_indices.
        // Original score/ranking retention and partition backing cost at most
        // twelve bytes per expert; selected-score, softmax,
        // gather and weight temporaries use at most 48 bytes per selected entry,
        // plus two F32 row reductions. Ordinary projection and subsequent evidence
        // transforms are excluded. Score storage is an additional allowance for
        // original evidence when raw-logit bias replaces the ordinary score path.
        let experts = u64::from(policy.expert_count);
        let entries = mul(rows, experts)?;
        let mut retained = add(
            4096,
            add(
                mul(entries, 12)?,
                mul(rows, add(mul(u64::from(policy.top_k), 48)?, 8)?)?,
            )?,
        )?;
        let mut host = 0;
        if policy.groups > 1 {
            // Group selection also forms [rows, selected_groups, experts] bool
            // and I32 masks. This dimension must not be hidden in a fixed
            // per-expert multiplier. Include grouped top-k/reductions, masked
            // ranking and the native/host partition-ID lookup.
            let group_masks = mul(mul(entries, u64::from(policy.selected_groups))?, 5)?;
            host = mul(experts, 4)?;
            retained = add(
                retained,
                add(
                    add(mul(entries, 20)?, group_masks)?,
                    add(mul(mul(rows, u64::from(policy.groups))?, 16)?, host)?,
                )?,
            )?;
        }
        Ok(CaptureUsage {
            captures: 0,
            retained_bytes: retained,
            host_bytes: host,
            encoded_bytes: 0,
        })
    }
}

fn native_shape(shape: &[u64]) -> Result<Vec<i32>, Exception> {
    shape
        .iter()
        .map(|n| i32::try_from(*n).map_err(|_| Exception::custom("intervention index exceeds i32")))
        .collect()
}
fn indices(
    slice: &ResolvedCaptureSlice,
) -> Result<Vec<safemlx::ops::indexing::ArrayIndexOp<'static>>, Exception> {
    let starts = native_shape(&slice.starts)?;
    let ends = native_shape(&slice.ends)?;
    let strides = native_shape(&slice.strides)?;
    Ok(starts
        .iter()
        .zip(ends)
        .zip(strides)
        .map(|((start, end), stride)| (*start..end).stride_by(stride).index_op())
        .collect())
}
fn native_dtype(dtype: InterventionDtype) -> Dtype {
    match dtype {
        InterventionDtype::Float32 => Dtype::Float32,
        InterventionDtype::Float16 => Dtype::Float16,
        InterventionDtype::Bfloat16 => Dtype::Bfloat16,
    }
}

impl InterventionBackend for NativeCapture<'_> {
    fn partition_routed_unit_locations(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
        geometry: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, Error>> {
        Some(routed::partition_locations(source, geometry, self.stream))
    }
    fn routed_unit_locations(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, MlxTensor>,
        geometry: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, Error>> {
        Some(routed::locations(source, geometry, self.stream))
    }
    fn select_elements(
        &mut self,
        source: &MlxTensor,
        indices: &[u64],
    ) -> Option<Result<MlxTensor, Error>> {
        Some((|| {
            let indices = routed::indices(indices)?;
            Ok(MlxTensor::from_array(
                source
                    .as_array()
                    .reshape(&[-1], self.stream)?
                    .take(&indices, self.stream)?,
            ))
        })())
    }
    fn update_elements(
        &mut self,
        source: &MlxTensor,
        indices: &[u64],
        replacement: &MlxTensor,
    ) -> Option<Result<MlxTensor, Error>> {
        Some((|| {
            let indices = routed::indices(indices)?;
            let mut flat = source.as_array().reshape(&[-1], self.stream)?;
            flat.try_index_mut_device(indices, replacement.as_array(), self.stream)?;
            Ok(MlxTensor::from_array(
                flat.reshape(source.shape(), self.stream)?,
            ))
        })())
    }
    fn intervention_dtype(&self, tensor: &MlxTensor) -> Result<InterventionDtype, Error> {
        match tensor.as_array().dtype() {
            Dtype::Float32 => Ok(InterventionDtype::Float32),
            Dtype::Float16 => Ok(InterventionDtype::Float16),
            Dtype::Bfloat16 => Ok(InterventionDtype::Bfloat16),
            _ => {
                Err(Exception::custom("activation intervention requires f32, f16, or bf16").into())
            }
        }
    }
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        NativeInterventionEstimator.validate_geometry(source, slice)
    }
    fn select_region(
        &mut self,
        tensor: &MlxTensor,
        slice: &ResolvedCaptureSlice,
    ) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(tensor.as_array().try_index_device(
            indices(slice)?.as_slice(),
            self.stream,
        )?))
    }
    fn update_region(
        &mut self,
        tensor: &MlxTensor,
        slice: &ResolvedCaptureSlice,
        replacement: &MlxTensor,
    ) -> Result<MlxTensor, Error> {
        let mut output = tensor.as_array().clone();
        output.try_index_mut_device(
            indices(slice)?.as_slice(),
            replacement.as_array(),
            self.stream,
        )?;
        Ok(MlxTensor::from_array(output))
    }
    fn zeros(&mut self, shape: &[u64], dtype: InterventionDtype) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(safemlx::ops::zeros_dtype(
            &native_shape(shape)?,
            native_dtype(dtype),
            self.stream,
        )?))
    }
    fn scale(&mut self, value: &MlxTensor, factor: f32) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(value.as_array().multiply(
            Array::from(factor).as_dtype(value.as_array().dtype(), self.stream)?,
            self.stream,
        )?))
    }
    fn fill_masked(
        &mut self,
        value: &MlxTensor,
        keep: &[bool],
        fill: f32,
    ) -> Result<MlxTensor, Error> {
        let mask = Array::from_slice(keep, value.shape());
        Ok(MlxTensor::from_array(safemlx::ops::r#where(
            mask,
            value.as_array(),
            Array::from(fill).as_dtype(value.as_array().dtype(), self.stream)?,
            self.stream,
        )?))
    }
    fn mask_components(
        &mut self,
        value: &MlxTensor,
        ids: &[u32],
        keep_selected: bool,
    ) -> Result<MlxTensor, Error> {
        let width = *value
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("component mask needs a final axis"))?;
        let mut keep = vec![!keep_selected; width as usize];
        for id in ids {
            keep[*id as usize] = keep_selected;
        }
        let mask = safemlx::ops::broadcast_to(
            Array::from_slice(&keep, &[width]),
            value.shape(),
            self.stream,
        )?;
        Ok(MlxTensor::from_array(safemlx::ops::r#where(
            mask,
            value.as_array(),
            Array::from(0.0f32).as_dtype(value.as_array().dtype(), self.stream)?,
            self.stream,
        )?))
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<MlxTensor, Error> {
        let shape = native_shape(&tensor.shape)?;
        let array = match &tensor.values {
            InterventionValues::Float32(values) => Array::from_slice(values, &shape),
            InterventionValues::Float16(bits) => Array::from_slice(
                &bits
                    .iter()
                    .copied()
                    .map(half::f16::from_bits)
                    .collect::<Vec<_>>(),
                &shape,
            ),
            InterventionValues::Bfloat16(bits) => Array::from_slice(
                &bits
                    .iter()
                    .copied()
                    .map(half::bf16::from_bits)
                    .collect::<Vec<_>>(),
                &shape,
            ),
        };
        Ok(MlxTensor::from_array(array))
    }
    fn add(&mut self, left: &MlxTensor, right: &MlxTensor) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(
            left.as_array().add(right.as_array(), self.stream)?,
        ))
    }
    fn fill_columns(
        &mut self,
        value: &MlxTensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<MlxTensor, Error> {
        let vocabulary = *value
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("column fill requires non-scalar value"))?;
        let mut keep = vec![true; vocabulary as usize];
        for id in ids {
            keep[*id as usize] = false;
        }
        let mask = safemlx::ops::broadcast_to(
            Array::from_slice(&keep, &[vocabulary]),
            value.shape(),
            self.stream,
        )?;
        Ok(MlxTensor::from_array(safemlx::ops::r#where(
            mask,
            value.as_array(),
            Array::from(fill).as_dtype(value.as_array().dtype(), self.stream)?,
            self.stream,
        )?))
    }
}
