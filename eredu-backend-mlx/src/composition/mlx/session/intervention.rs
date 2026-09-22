//! Native realization of validated host-only activation plans.

use super::bounded_capture::NativeCapture;
use super::*;
use eredu_core::{capture::*, intervention::*};

pub(in crate::composition::mlx::session) mod model;
pub(in crate::composition::mlx::session) mod partition;
mod routed;
mod static_activation;
pub(crate) use partition::PreparedPartitionModelIntervention;
mod text;
pub(crate) use model::PreparedModelInterventions;
pub(crate) use static_activation::{
    PreparedSparseActivation, PreparedStaticActivation, SparseActivationFailure,
    StaticActivationPopulation,
};
pub(in crate::composition::mlx) use text::TextInterventionQuote;
pub(crate) use text::{PreparedTextInterventions, PreparedTextInterventionsOwner};
#[cfg(test)]
mod tests;

const OPERATIONS: &[InterventionKind] = &[
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
];
const DTYPES: &[InterventionDtype] = &[
    InterventionDtype::Float32,
    InterventionDtype::Float16,
    InterventionDtype::Bfloat16,
];
const SCORE_STAGES: &[RoutingScoreStage] = &[
    RoutingScoreStage::RawLogits,
    RoutingScoreStage::TransformedScores,
    RoutingScoreStage::RankingScores,
];
pub(crate) fn mechanism_facts() -> InterventionMechanismFacts<'static> {
    InterventionMechanismFacts {
        routed_units: true,
        operations: OPERATIONS,
        dtypes: DTYPES,
        score_stages: SCORE_STAGES,
    }
}
pub(crate) fn mechanisms() -> InterventionMechanisms {
    let facts = mechanism_facts();
    InterventionMechanisms {
        routed_units: facts.routed_units,
        operations: facts.operations.to_vec(),
        dtypes: facts.dtypes.to_vec(),
        score_stages: facts.score_stages.to_vec(),
    }
}

/// Cold MLX bounds: the same estimator is used at admission and before execution.
pub(crate) struct NativeInterventionEstimator;
impl NativeInterventionEstimator {
    /// Shared dense/sparse preflight controls and bounded diagnostic alternatives.
    /// The virtual sparse expert axis is never a native allocation.
    pub(crate) fn prepared_preflight_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let messages = [
            "MLX intervention geometry exceeds signed 32-bit indexing",
            "scalar activation edit",
            "MLX capture shape/index exceeds signed 32-bit indexing",
            "selected score IDs exceed the runtime vocabulary or count limit",
            "candidate count exceeds the runtime vocabulary",
            "histogram bin count",
            "invalid sparse edit virtual shape",
            "MLX sparse edit indexing exceeds i32",
            "sparse routing action",
            "sparse edit estimate includes record encoding",
        ];
        let frames = [
            size_of::<(&Self, &[u64], &ResolvedCaptureSlice, &InterventionAction)>(),
            size_of::<(&[u64], &CaptureSelection, &[u64], &[u64])>(),
            size_of::<[CaptureUsage; 2]>(),
            size_of::<Result<CaptureUsage, CaptureError>>(),
            // Four activation census values and eight dense capture census values.
            size_of::<[u64; 4]>(),
            size_of::<[u64; 8]>(),
            size_of::<std::slice::Iter<'_, u64>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<(
                RoutedUnitGeometry,
                &[u64],
                &ResolvedCaptureSlice,
                &InterventionAction,
            )>(),
            size_of::<(u64, u64, u64)>(),
            size_of::<Option<InterventionDtype>>(),
        ];
        frames
            .into_iter()
            .try_fold(
                size_of_val(&frames).checked_add(size_of_val(&messages))?,
                usize::checked_add,
            )?
            .checked_add(messages.iter().map(|text| text.len()).max().unwrap_or(0))
    }
}

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

    fn window_routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        _logical_source: &[u64],
        physical_source: &[u64],
        slice: &ResolvedCaptureSlice,
        action: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        // The existing worker counts native participating rows separately from
        // validation of the immutable logical action payload.
        self.routed_unit_usage(geometry, physical_source, slice, action)
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

fn native_dtype(dtype: InterventionDtype) -> Dtype {
    match dtype {
        InterventionDtype::Float32 => Dtype::Float32,
        InterventionDtype::Float16 => Dtype::Float16,
        InterventionDtype::Bfloat16 => Dtype::Bfloat16,
    }
}

impl InterventionBackend for NativeCapture<'_> {
    fn matches_intervention_shape(
        &self,
        tensor: &MlxTensor,
        expected: &[u64],
    ) -> Result<bool, Error> {
        Ok(tensor.shape().len() == expected.len()
            && tensor
                .shape()
                .iter()
                .zip(expected)
                .all(|(a, b)| u64::try_from(*a).ok() == Some(*b)))
    }

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
            let flat = source.as_array().reshape(&[-1], self.stream)?;
            static_activation::ordinary(self.stream, |worker| {
                worker.indexed_select(&flat, &indices)
            })
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
            let flat = source.as_array().reshape(&[-1], self.stream)?;
            let output = static_activation::ordinary(self.stream, |worker| {
                worker.indexed_update(&flat, &indices, replacement.as_array())
            })?;
            Ok(MlxTensor::from_array(
                output.as_array().reshape(source.shape(), self.stream)?,
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
        static_activation::ordinary(self.stream, |worker| {
            worker.select_region(tensor.as_array(), slice)
        })
    }
    fn update_region(
        &mut self,
        tensor: &MlxTensor,
        slice: &ResolvedCaptureSlice,
        replacement: &MlxTensor,
    ) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| {
            worker.update_region(tensor.as_array(), slice, replacement.as_array())
        })
    }
    fn zeros(&mut self, shape: &[u64], dtype: InterventionDtype) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| worker.zeros(shape, dtype))
    }
    fn scale(&mut self, value: &MlxTensor, factor: f32) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| worker.scale(value.as_array(), factor))
    }
    fn fill_masked(
        &mut self,
        value: &MlxTensor,
        keep: &[bool],
        fill: f32,
    ) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| {
            worker.fill_masked(value.as_array(), keep, fill)
        })
    }
    fn mask_components(
        &mut self,
        value: &MlxTensor,
        ids: &[u32],
        keep_selected: bool,
    ) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| {
            worker.mask_components(value.as_array(), ids, keep_selected)
        })
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| worker.realize_tensor(tensor))
    }
    fn add(&mut self, left: &MlxTensor, right: &MlxTensor) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| {
            worker.add(left.as_array(), right.as_array())
        })
    }
    fn fill_columns(
        &mut self,
        value: &MlxTensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<MlxTensor, Error> {
        static_activation::ordinary(self.stream, |worker| {
            worker.fill_columns(value.as_array(), ids, fill)
        })
    }
}
