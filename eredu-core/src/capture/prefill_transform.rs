//! One ordinary selected transform bound to its actual source and chunk schedule.
use super::*;
use crate::{InferenceGeometry, ObservationDtype, ObservationValueType, OutputDemand};
use std::ops::Range;
fn invalid() -> CaptureError {
    CaptureError::Invalid("ordinary capture transform source, axes or schedule differ".into())
}

/// Exact original projection/temporal source failure, preserving its typed cause.
#[derive(Debug, thiserror::Error)]
pub enum CapturePrefillPartitionError {
    /// The existing immutable request/window constructor rejected this source.
    #[error(transparent)]
    Original(#[from] CaptureError),
    /// Actual partition tensor geometry differs from the original admission.
    #[error(transparent)]
    Geometry(#[from] CaptureTensorGeometryError),
    /// Actual spatial source cannot share the retained temporal window.
    #[error(transparent)]
    Window(#[from] CaptureWindowError),
    /// The existing raw temporal scatter rejected the source.
    #[error(transparent)]
    Rows(#[from] CapturePrefillGeometryError),
}

/// Borrowed ordinary Summary/Histogram/Preview geometry. This supplies neither
/// causal hook support nor native execution, quota, allocation or completion.
#[derive(Debug)]
pub struct CapturePrefillTransformPlan<'a> {
    source: &'a AdmittedCapturePlan,
    index: usize,
    inference: InferenceGeometry,
    window: CaptureWindowGeometry,
}
impl<'a> CapturePrefillTransformPlan<'a> {
    /// Bind the actual immutable selection and exact ordinary request.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        index: usize,
        inference: InferenceGeometry,
    ) -> Result<Self, CaptureError> {
        let selection = source.plan().selections.get(index).ok_or_else(invalid)?;
        let point = source.points().get(index).ok_or_else(invalid)?;
        if source.invocation_bounds().is_some()
            || source.text_origin().map(|o| o.cached_positions) != Some(inference.cached_positions)
            || source.request().batch != inference.batch_size
            || source.request().prompt_tokens != inference.input_positions
            || inference.max_output_tokens > source.request().max_predictions
            || inference.batch_size == 0
            || inference.input_positions == 0
            || inference.max_output_tokens == 0
            || inference.prefill_chunk_positions == 0
            || inference.prefill_chunk_positions > inference.input_positions
            || !point.prefill
            || !selection.schedule.includes(CapturePhase::Prefill, 0)
            || !(point.dtype == ObservationDtype::Floating
                || point.dtype == ObservationDtype::Integer
                    && matches!(
                        selection.transform,
                        CaptureTransform::Preview { .. } | CaptureTransform::Summary
                    ))
            || point.value_type != ObservationValueType::Tensor
            || !matches!(
                selection.transform,
                CaptureTransform::Summary
                    | CaptureTransform::Histogram { .. }
                    | CaptureTransform::Preview { .. }
            )
        {
            return Err(invalid());
        }
        inference
            .cached_positions
            .checked_add(inference.input_positions)
            .and_then(|v| v.checked_add(inference.max_output_tokens))
            .ok_or(CaptureError::Overflow)?;
        let axes = point.axes.as_ref().ok_or_else(invalid)?;
        if axes
            .iter()
            .filter(|a| {
                a.dimension == SymbolicDimension::Sequence
                    || a.dimension == SymbolicDimension::TokenRows && inference.batch_size == 1
            })
            .count()
            != 1
            || axes.iter().any(|a| {
                !matches!(
                    a.dimension,
                    SymbolicDimension::Sequence
                        | SymbolicDimension::TokenRows
                        | SymbolicDimension::Batch
                        | SymbolicDimension::Known(_)
                )
            })
        {
            return Err(invalid());
        }
        let geometry = source.geometry_at(CapturePhase::Prefill, 0, None)?;
        geometry.validate()?;
        geometry.validate_slices(point, &selection.slices)?;
        let window = CaptureWindowGeometry::new(
            point,
            selection,
            inference.batch_size,
            inference.input_positions,
        )?;
        Ok(Self {
            source,
            index,
            inference,
            window,
        })
    }
    /// Project one actual spatial term while retaining the same original
    /// transform, request and temporal worker. Sum terms remain raw at native
    /// execution; this plan describes only their selected source geometry.
    pub fn prepare_partition(
        source: &'a AdmittedCapturePlan,
        index: usize,
        inference: InferenceGeometry,
        projection: &CaptureSlicePartition,
        fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, CapturePrefillPartitionError> {
        let mut plan = Self::prepare(source, index, inference)?;
        if combination == PartitionCaptureCombination::SumF64ToF32 {
            CaptureTensorGeometry::prepare_partition(
                source,
                index,
                CapturePhase::Prefill,
                0,
                None,
                projection,
                fragment,
                combination,
            )?;
        } else {
            match plan.selection().transform {
                CaptureTransform::Summary => {
                    CaptureSummaryGeometry::prepare_partition(
                        source,
                        index,
                        CapturePhase::Prefill,
                        0,
                        None,
                        projection,
                        fragment,
                    )?;
                }
                CaptureTransform::Histogram { .. } => {
                    CaptureHistogramGeometry::prepare_partition(
                        source,
                        index,
                        CapturePhase::Prefill,
                        0,
                        None,
                        projection,
                        fragment,
                    )?;
                }
                CaptureTransform::Preview { .. } => {
                    CaptureTensorGeometry::prepare_partition(
                        source,
                        index,
                        CapturePhase::Prefill,
                        0,
                        None,
                        projection,
                        fragment,
                        combination,
                    )?;
                }
                _ => return Err(CaptureTensorGeometryError::Unsupported.into()),
            }
        }
        plan.window = plan.window.project_partition(projection, fragment)?;
        Ok(plan)
    }
    /// Fixed source, typed geometry and result transports for this adapter.
    pub fn partition_preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>() * 3,
            size_of::<CaptureWindowGeometry>() * 3,
            size_of::<CapturePrefillPartitionError>(),
            size_of::<Result<Self, CapturePrefillPartitionError>>(),
            size_of::<(
                &AdmittedCapturePlan,
                usize,
                InferenceGeometry,
                &CaptureSlicePartition,
                usize,
                PartitionCaptureCombination,
            )>(),
            size_of::<CaptureSummaryGeometry<'_>>(),
            size_of::<CaptureHistogramGeometry<'_>>(),
            size_of::<(usize, &ResolvedCaptureSlice)>(),
            size_of::<Result<CaptureWindowGeometry, CaptureWindowError>>(),
            CaptureTensorGeometry::partition_preparation_control_bytes()?,
            CaptureSummaryGeometry::preparation_control_bytes()?,
            CaptureHistogramGeometry::preparation_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Exact original semantic admission, never an allocation pin.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.source
    }
    /// Actual selected ordinal.
    pub fn selection_index(&self) -> usize {
        self.index
    }
    /// Actual candidate geometry, without upgrading its readout.
    pub fn inference_geometry(&self) -> InferenceGeometry {
        self.inference
    }
    /// Fixed selected window; its extents do not establish hook equivalence.
    pub fn window(&self) -> &CaptureWindowGeometry {
        &self.window
    }
    /// Actual selected transform and spatial slices.
    pub fn selection(&self) -> &'a CaptureSelection {
        &self.source.plan().selections[self.index]
    }
    /// Number of canonical nonempty spans.
    pub fn chunk_count(&self) -> u64 {
        self.inference
            .input_positions
            .div_ceil(self.inference.prefill_chunk_positions)
    }
    /// Derive physical selection without allocating an index or shape vector.
    pub fn fragment(
        &self,
        index: u64,
    ) -> Result<CapturePrefillTransformFragment<'_, 'a>, CaptureError> {
        if index >= self.chunk_count() {
            return Err(invalid());
        }
        let start = mul(index, self.inference.prefill_chunk_positions)?;
        let end =
            add(start, self.inference.prefill_chunk_positions)?.min(self.inference.input_positions);
        let position = add(self.inference.cached_positions, start)?;
        let physical = self.window.fragment(
            &self.source.points()[self.index],
            self.selection(),
            start,
            end,
        )?;
        Ok(CapturePrefillTransformFragment {
            plan: self,
            index,
            input: start..end,
            position,
            source: physical.source,
            selected: physical.selected,
            starts: physical.starts,
            ends: physical.ends,
            strides: physical.strides,
            row_offset: physical.row_offset,
        })
    }
}

/// Source-bound physical transform view, with no independently supplied offset.
#[derive(Debug)]
pub struct CapturePrefillTransformFragment<'p, 'a> {
    plan: &'p CapturePrefillTransformPlan<'a>,
    index: u64,
    input: Range<u64>,
    position: u64,
    source: [u64; 32],
    selected: [u64; 32],
    starts: [u64; 32],
    ends: [u64; 32],
    strides: [u64; 32],
    row_offset: u64,
}
impl<'p, 'a> CapturePrefillTransformFragment<'p, 'a> {
    /// Actual bound plan.
    pub fn plan(&self) -> &'p CapturePrefillTransformPlan<'a> {
        self.plan
    }
    /// Canonical chunk index.
    pub fn chunk_index(&self) -> u64 {
        self.index
    }
    /// New-request coordinates, not encoder/patch coordinates.
    pub fn input(&self) -> &Range<u64> {
        &self.input
    }
    /// Absolute decoder position.
    pub fn position(&self) -> u64 {
        self.position
    }
    /// Bound physical readout demand.
    pub fn output_demand(&self) -> OutputDemand {
        self.plan
            .inference
            .output
            .for_chunk(self.input.end == self.plan.inference.input_positions)
    }
    /// Compare actual driver coordinates without creating execution authority.
    pub fn matches_chunk(&self, input: &Range<u64>, position: u64, output: OutputDemand) -> bool {
        input == &self.input && position == self.position && output == self.output_demand()
    }
    /// Actual physical source shape.
    pub fn source_shape(&self) -> &[u64] {
        &self.source[..self.plan.window.rank()]
    }
    /// Rectangular physical selection, before Preview.
    pub fn selected_shape(&self) -> &[u64] {
        &self.selected[..self.plan.window.rank()]
    }
    /// Local inclusive starts after globally anchored stride translation.
    pub fn starts(&self) -> &[u64] {
        &self.starts[..self.plan.window.rank()]
    }
    /// Local exclusive ends.
    pub fn ends(&self) -> &[u64] {
        &self.ends[..self.plan.window.rank()]
    }
    /// Exact positive strides.
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.plan.window.rank()]
    }
    /// Selected scalar count; construction already checked all products.
    pub fn selected_elements(&self) -> u64 {
        elements(self.selected_shape()).expect("checked fragment")
    }
    /// Global selected-row origin for existing positive-stride prefix mapping.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }
}
