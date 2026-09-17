//! Allocation-free output geometry for one admitted floating tensor selection.
use super::*;
use crate::{ObservationDtype, ObservationValueType};
mod partition;

/// Rejection by the current closed F32 observation construction geometry.
#[derive(Debug, thiserror::Error)]
pub enum CaptureTensorGeometryError {
    /// The indexed selection is not part of this immutable admission.
    #[error("capture selection {index} is absent")]
    SelectionMissing {
        /// Requested selection index.
        index: usize,
    },
    /// The selected point is not scheduled at the supplied coordinate.
    #[error("capture tensor selection is inactive")]
    Inactive,
    /// Only raw floating tensor outputs are implemented by this primitive.
    #[error("capture tensor construction requires a floating raw tensor transform")]
    Unsupported,
    /// A fragment is not the exact selected global projection or additive term.
    #[error("capture partition differs from its admitted global projection")]
    Partition,
    /// Geometry remains unknown; no destination size can be proved.
    #[error("capture tensor shape is unknown")]
    UnknownShape,
    /// A checked implementation limit, never truncation of source geometry.
    #[error("capture tensor rank {rank} exceeds construction limit {maximum}")]
    RankExceeded {
        /// Actual declared rank.
        rank: usize,
        /// Maximum rank supported by this closed constructor.
        maximum: usize,
    },
    /// Host dimensions or allocation geometry cannot be represented.
    #[error("capture tensor geometry overflow")]
    Overflow,
    /// The sparse bank or selected axes do not match the declared routed geometry.
    #[error(transparent)]
    Routed(#[from] RoutedUnitValidationError),
    /// The original admitted invocation/slice validation failed.
    #[error(transparent)]
    Capture(#[from] CaptureError),
}

/// Borrowed immutable output geometry for Preview, Slice or FullTensor F32 data.
///
/// This proves only the selected output extent. It does not establish actual
/// tensor values, source backing, native conversion/transfer funding, capture
/// quota or session provenance. Those remain the enclosing capture worker's
/// contracts. No source/shape/data allocation is performed during preparation.
/// Cloning copies only these fixed coordinates and the borrowed admission, so
/// native consumers can preserve an existing partition or physical window. It
/// creates no destination, source pin, quota, or permission to capture again.
#[derive(Debug, Clone)]
pub struct CaptureTensorGeometry<'a> {
    source: &'a AdmittedCapturePlan,
    selection: usize,
    phase: CapturePhase,
    prediction: u64,
    source_shape: [usize; 32],
    source_rank: usize,
    starts: [u64; 32],
    ends: [u64; 32],
    strides: [u64; 32],
    shape: [usize; 32],
    rank: usize,
    elements: usize,
    additive_term: bool,
}
#[derive(Clone, Copy)]
enum GeometryTransform {
    Tensor,
    AdditiveTensor,
    Summary,
    Histogram,
    RoutedUnits,
}
impl<'a> CaptureTensorGeometry<'a> {
    /// Derives exact known output axes from one actual admitted selection.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureTensorGeometryError> {
        Self::prepare_kind(
            source,
            selection,
            phase,
            prediction,
            invocation,
            GeometryTransform::Tensor,
        )
    }
    /// Project the same immutable global selection onto one actual input window.
    /// Both physical and logical extents are validated against this source; no
    /// source pin, callback, quota or native construction is authorized here.
    pub fn prepare_window(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let logical = window.validate(physical)?;
        source.geometry_at(phase, prediction, Some(physical))?;
        Self::prepare_kind(
            source,
            selection,
            phase,
            prediction,
            Some(logical),
            GeometryTransform::Tensor,
        )?
        .apply_window(physical, window)
    }
    fn apply_window(
        self,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let selection = &self.source.plan().selections[self.selection];
        let point = &self.source.points()[self.selection];
        let selected = CaptureWindowGeometry::from_resolved(
            point,
            selection,
            physical.batch,
            window.logical_sequence,
            self.source_shape(),
        )?;
        self.apply_selected_window(physical, window, selected)
    }
    // Spatial and complete sources share the exact same row intersection.
    // `selected` has already been authenticated against this original plan.
    fn apply_selected_window(
        mut self,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
        selected: CaptureWindowGeometry,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let selection = &self.source.plan().selections[self.selection];
        let point = &self.source.points()[self.selection];
        let end = window
            .start
            .checked_add(physical.sequence)
            .ok_or(CaptureTensorGeometryError::Overflow)?;
        let local = selected.fragment(point, selection, window.start, end)?;
        let mut elements = 1usize;
        for axis in 0..self.source_rank {
            self.source_shape[axis] = usize::try_from(local.source[axis])
                .map_err(|_| CaptureTensorGeometryError::Overflow)?;
            self.shape[axis] = usize::try_from(local.selected[axis])
                .map_err(|_| CaptureTensorGeometryError::Overflow)?;
            self.starts[axis] = local.starts[axis];
            self.ends[axis] = local.ends[axis];
            self.strides[axis] = local.strides[axis];
            elements = elements
                .checked_mul(self.shape[axis])
                .ok_or(CaptureTensorGeometryError::Overflow)?;
        }
        self.rank = self.source_rank;
        self.elements = elements;
        if let CaptureTransform::Preview { max_elements } = selection.transform {
            self.elements = elements.min(
                usize::try_from(max_elements).map_err(|_| CaptureTensorGeometryError::Overflow)?,
            );
            self.shape = [0; 32];
            self.shape[0] = self.elements;
            self.rank = 1;
        }
        Ok(self)
    }
    /// Fixed preparation/physical-window frames, excluding any caller allocation.
    pub fn preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            // Both the window caller and shared physical-selection adapter
            // retain their moved arguments/results until the return frontier.
            size_of::<Self>() * 2,
            size_of::<Result<Self, CaptureTensorGeometryError>>() * 2,
            size_of::<CaptureWindowGeometry>() * 2,
            size_of::<(CaptureInvocationShape, CaptureInvocationWindow, CaptureWindowGeometry)>(),
            size_of::<super::window_geometry::WindowSelection>(),
            size_of::<Result<super::window_geometry::WindowSelection, CaptureError>>(),
            size_of::<(
                CaptureInvocationShape,
                CaptureInvocationWindow,
                GeometryTransform,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn prepare_kind(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        transform: GeometryTransform,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let selected = source
            .plan
            .selections
            .get(selection)
            .ok_or(CaptureTensorGeometryError::SelectionMissing { index: selection })?;
        let point = source
            .points
            .get(selection)
            .ok_or(CaptureTensorGeometryError::SelectionMissing { index: selection })?;
        if prediction >= source.request.max_predictions
            || !selected.schedule.includes(phase, prediction)
            || !match phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Err(CaptureTensorGeometryError::Inactive);
        }
        if point.dtype != ObservationDtype::Floating
            || !match transform {
                GeometryTransform::RoutedUnits => matches!(point.value_type, ObservationValueType::RoutedUnits { .. }),
                _ => point.value_type == ObservationValueType::Tensor,
            }
            || !match transform {
                GeometryTransform::RoutedUnits => matches!(selected.transform, CaptureTransform::RoutedUnits),
                GeometryTransform::Summary => {
                    matches!(selected.transform, CaptureTransform::Summary)
                }
                GeometryTransform::Histogram => {
                    matches!(selected.transform, CaptureTransform::Histogram { .. })
                }
                GeometryTransform::AdditiveTensor => matches!(selected.transform,
                    CaptureTransform::FullTensor | CaptureTransform::Slice | CaptureTransform::Preview { .. }
                        | CaptureTransform::Summary | CaptureTransform::Histogram { .. }),
                GeometryTransform::Tensor => matches!(
                    selected.transform,
                    CaptureTransform::Preview { .. }
                        | CaptureTransform::Slice
                        | CaptureTransform::FullTensor
                ),
            }
        {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        let geometry = source.geometry_at(phase, prediction, invocation)?;
        geometry.validate()?;
        geometry.validate_slices(point, &selected.slices)?;
        let axes = point
            .axes
            .as_ref()
            .ok_or(CaptureTensorGeometryError::UnknownShape)?;
        if axes.len() > 32 {
            return Err(CaptureTensorGeometryError::RankExceeded {
                rank: axes.len(),
                maximum: 32,
            });
        }
        let mut source_shape = [0usize; 32];
        let mut shape = [0usize; 32];
        let mut starts = [0; 32];
        let mut ends = [0; 32];
        let mut strides = [1; 32];
        let mut elements = 1usize;
        for (index, axis) in axes.iter().enumerate() {
            let source_extent = geometry
                .extent(&axis.dimension)?
                .ok_or(CaptureTensorGeometryError::UnknownShape)?;
            source_shape[index] =
                usize::try_from(source_extent).map_err(|_| CaptureTensorGeometryError::Overflow)?;
            let extent =
                if let Some(slice) = selected.slices.iter().find(|slice| slice.axis == axis.name) {
                    if slice.stride == 0 || slice.start > slice.end || slice.end > source_extent {
                        return Err(CaptureError::Invalid(
                            "capture tensor slice exceeds admitted extent".into(),
                        )
                        .into());
                    }
                    starts[index] = slice.start;
                    ends[index] = slice.end;
                    strides[index] = slice.stride;
                    (slice.end - slice.start).div_ceil(slice.stride)
                } else {
                    ends[index] = source_extent;
                    source_extent
                };
            shape[index] =
                usize::try_from(extent).map_err(|_| CaptureTensorGeometryError::Overflow)?;
            elements = elements
                .checked_mul(shape[index])
                .ok_or(CaptureTensorGeometryError::Overflow)?;
        }
        let mut rank = axes.len();
        if let CaptureTransform::Preview { max_elements } = selected.transform {
            elements = elements.min(
                usize::try_from(max_elements).map_err(|_| CaptureTensorGeometryError::Overflow)?,
            );
            shape = [0; 32];
            shape[0] = elements;
            rank = 1;
        }
        Ok(Self {
            source,
            selection,
            phase,
            prediction,
            source_shape,
            source_rank: axes.len(),
            starts,
            ends,
            strides,
            shape,
            rank,
            elements,
            additive_term: matches!(transform, GeometryTransform::AdditiveTensor),
        })
    }
    /// Exact native transform for this geometry. Additive contributions retain
    /// the original global admission while producing raw terms before reduction.
    pub fn native_transform(&self) -> &CaptureTransform {
        let transform = &self.source.plan().selections[self.selection].transform;
        if self.additive_term && !matches!(transform, CaptureTransform::Preview { .. }) {
            &CaptureTransform::Slice
        } else { transform }
    }
    /// Original immutable semantic admission; no clone or native source pin.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.source
    }
    /// Selected point's index within the immutable admission.
    pub fn selection_index(&self) -> usize {
        self.selection
    }
    /// Phase used to resolve the source axes.
    pub fn phase(&self) -> CapturePhase {
        self.phase
    }
    /// Admitted scheduling coordinate, not permission to execute it again.
    pub fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Resolved source axes before selection or Preview flattening. These are
    /// immutable geometry only, never evidence of actual numerical backing.
    pub fn source_shape(&self) -> &[usize] {
        &self.source_shape[..self.source_rank]
    }
    /// Source-relative inclusive starts; global window strides keep their origin.
    pub fn starts(&self) -> &[u64] {
        &self.starts[..self.source_rank]
    }
    /// Source-relative exclusive ends.
    pub fn ends(&self) -> &[u64] {
        &self.ends[..self.source_rank]
    }
    /// Positive source-relative selection strides.
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.source_rank]
    }
    /// Actual output shape; Preview is the existing flattened-prefix output.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }
    /// Number of F32 destination scalars.
    pub fn elements(&self) -> usize {
        self.elements
    }
}

/// Exact selected source geometry for the existing finite-only Summary policy.
/// This grants no tensor destination, native allocation, provenance or completion.
/// Its private raw shape worker is never exposed as a raw-tensor claim.
#[derive(Debug)]
struct CaptureReductionGeometry<'a> {
    selected: CaptureTensorGeometry<'a>,
    starts: [u64; 32],
    ends: [u64; 32],
    strides: [u64; 32],
    fragment: Option<(
        crate::InferenceGeometry,
        std::ops::Range<u64>,
        u64,
        crate::OutputDemand,
    )>,
}
impl<'a> CaptureReductionGeometry<'a> {
    /// Derive the actual admitted summary axes without allocation or source cloning.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        transform: GeometryTransform,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let selected = CaptureTensorGeometry::prepare_kind(
            source, selection, phase, prediction, invocation, transform,
        )?;
        let mut value = Self {
            selected,
            starts: [0; 32],
            ends: [0; 32],
            strides: [1; 32],
            fragment: None,
        };
        value.starts = value.selected.starts;
        value.ends = value.selected.ends;
        value.strides = value.selected.strides;
        Ok(value)
    }
    fn prepare_window(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
        transform: GeometryTransform,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let logical = window.validate(physical)?;
        source.geometry_at(phase, prediction, Some(physical))?;
        let mut value = Self::prepare(
            source,
            selection,
            phase,
            prediction,
            Some(logical),
            transform,
        )?;
        value.selected = value.selected.apply_window(physical, window)?;
        value.starts = value.selected.starts;
        value.ends = value.selected.ends;
        value.strides = value.selected.strides;
        Ok(value)
    }
    /// Bind this same selection to one core-derived canonical source fragment.
    /// This is geometry only; the caller still authenticates its source channel.
    pub fn fragment(
        mut self,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<Self, CaptureTensorGeometryError> {
        if !std::ptr::eq(self.admission(), fragment.plan().admission())
            || self.selection_index() != fragment.plan().selection_index()
            || self.phase() != CapturePhase::Prefill
            || self.prediction() != 0
            || !matches!(
                fragment.plan().selection().transform,
                CaptureTransform::Summary | CaptureTransform::Histogram { .. }
            )
        {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        let rank = fragment.source_shape().len();
        if rank != self.selected.source_rank || rank > 32 {
            return Err(CaptureTensorGeometryError::UnknownShape);
        }
        for axis in 0..rank {
            self.starts[axis] = fragment.starts()[axis];
            self.ends[axis] = fragment.ends()[axis];
            self.strides[axis] = fragment.strides()[axis];
            self.selected.source_shape[axis] = usize::try_from(fragment.source_shape()[axis])
                .map_err(|_| CaptureTensorGeometryError::Overflow)?;
            self.selected.shape[axis] = usize::try_from(fragment.selected_shape()[axis])
                .map_err(|_| CaptureTensorGeometryError::Overflow)?;
        }
        self.fragment = Some((
            fragment.plan().inference_geometry(),
            fragment.input().clone(),
            fragment.position(),
            fragment.output_demand(),
        ));
        self.selected.rank = rank;
        self.selected.elements = usize::try_from(fragment.selected_elements())
            .map_err(|_| CaptureTensorGeometryError::Overflow)?;
        Ok(self)
    }
    /// Compare the core-derived fragment with an actual driver coordinate.
    /// Equality is placement only and supplies no source/account authority.
    pub fn matches_prefill(
        &self,
        inference: crate::InferenceGeometry,
        input: &std::ops::Range<u64>,
        position: u64,
        output: crate::OutputDemand,
    ) -> bool {
        self.fragment.as_ref().is_some_and(|(g, range, p, d)| {
            *g == inference && range == input && *p == position && *d == output
        })
    }
    /// Immutable original admission, never a backing registration.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.selected.admission()
    }
    /// Original selection ordinal.
    pub fn selection_index(&self) -> usize {
        self.selected.selection_index()
    }
    /// Original observation phase.
    pub fn phase(&self) -> CapturePhase {
        self.selected.phase()
    }
    /// Original prediction coordinate.
    pub fn prediction(&self) -> u64 {
        self.selected.prediction()
    }
    /// Actual source axes before the admitted slice.
    pub fn source_shape(&self) -> &[usize] {
        self.selected.source_shape()
    }
    /// Selected axes whose elements contribute to the summary.
    pub fn shape(&self) -> &[usize] {
        self.selected.shape()
    }
    /// Source-relative starts from the original selection or canonical fragment.
    pub fn starts(&self) -> &[u64] {
        &self.starts[..self.selected.source_rank]
    }
    /// Source-relative exclusive ends, with no reconstructed global offsets.
    pub fn ends(&self) -> &[u64] {
        &self.ends[..self.selected.source_rank]
    }
    /// Actual positive selection strides.
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.selected.source_rank]
    }
    /// Exact number of selected values, including an empty selection.
    pub fn elements(&self) -> usize {
        self.selected.elements()
    }
}

// Both typed wrappers retain the same private static reduction selection worker.
// A Histogram can never be turned into a Summary or raw-tensor claim.
macro_rules! reduction_geometry {
    ($name:ident, $kind:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Debug)]
        pub struct $name<'a> {
            inner: CaptureReductionGeometry<'a>,
        }
        impl<'a> $name<'a> {
            /// Fixed shared preparation/fragment representations; no allocation or authority.
            pub fn preparation_control_bytes() -> Option<usize> {
                use std::mem::{size_of, size_of_val};
                let frames = [
                    size_of::<Self>(),
                    size_of::<Result<Self, CaptureTensorGeometryError>>(),
                    size_of::<CaptureReductionGeometry<'a>>(),
                    size_of::<Result<CaptureReductionGeometry<'a>, CaptureTensorGeometryError>>(),
                    CaptureTensorGeometry::preparation_control_bytes()?,
                    size_of::<GeometryTransform>(),
                    size_of::<(
                        &AdmittedCapturePlan,
                        usize,
                        CapturePhase,
                        u64,
                        Option<CaptureInvocationShape>,
                    )>(),
                ];
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
            }

            /// Derive exact axes from this transform's immutable admission.
            pub fn prepare(
                source: &'a AdmittedCapturePlan,
                selection: usize,
                phase: CapturePhase,
                prediction: u64,
                invocation: Option<CaptureInvocationShape>,
            ) -> Result<Self, CaptureTensorGeometryError> {
                Ok(Self {
                    inner: CaptureReductionGeometry::prepare(
                        source,
                        selection,
                        phase,
                        prediction,
                        invocation,
                        GeometryTransform::$kind,
                    )?,
                })
            }
            /// Exact disjoint fragment of the original global reduction.
            /// Additive receipts use raw tensor terms and reduce after assembly.
            pub fn prepare_partition(
                source: &'a AdmittedCapturePlan, selection: usize, phase: CapturePhase,
                prediction: u64, invocation: Option<CaptureInvocationShape>,
                projection: &CaptureSlicePartition, fragment: usize,
            ) -> Result<Self, CaptureTensorGeometryError> {
                let selected = CaptureTensorGeometry::prepare_kind(source, selection, phase,
                    prediction, invocation, GeometryTransform::$kind)?
                    .apply_partition(projection, fragment, PartitionCaptureCombination::Disjoint)?;
                let starts = selected.starts; let ends = selected.ends; let strides = selected.strides;
                Ok(Self { inner: CaptureReductionGeometry { selected, starts, ends, strides, fragment: None } })
            }
            /// Compose the original disjoint spatial source with one physical
            /// row window; additive terms use the raw tensor source instead.
            pub fn prepare_partition_window(
                source: &'a AdmittedCapturePlan, selection: usize, phase: CapturePhase,
                prediction: u64, physical: CaptureInvocationShape, window: CaptureInvocationWindow,
                projection: &CaptureSlicePartition, fragment: usize,
            ) -> Result<Self, CaptureTensorGeometryError> {
                let selected = CaptureTensorGeometry::prepare_partition_window_kind(
                    source, selection, phase, prediction, physical, window, projection,
                    fragment, PartitionCaptureCombination::Disjoint, GeometryTransform::$kind)?;
                let starts = selected.starts; let ends = selected.ends; let strides = selected.strides;
                Ok(Self { inner: CaptureReductionGeometry { selected, starts, ends, strides, fragment: None } })
            }
            /// Bind the same globally selected reduction to one actual invocation window.
            pub fn prepare_window(
                source: &'a AdmittedCapturePlan,
                selection: usize,
                phase: CapturePhase,
                prediction: u64,
                physical: CaptureInvocationShape,
                window: CaptureInvocationWindow,
            ) -> Result<Self, CaptureTensorGeometryError> {
                Ok(Self {
                    inner: CaptureReductionGeometry::prepare_window(
                        source,
                        selection,
                        phase,
                        prediction,
                        physical,
                        window,
                        GeometryTransform::$kind,
                    )?,
                })
            }
            /// Bind only the same admission/selection's canonical physical fragment.
            pub fn fragment(
                self,
                fragment: &CapturePrefillTransformFragment<'_, '_>,
            ) -> Result<Self, CaptureTensorGeometryError> {
                Ok(Self {
                    inner: self.inner.fragment(fragment)?,
                })
            }
            /// Check placement against the actual driver, without granting authority.
            pub fn matches_prefill(
                &self,
                inference: crate::InferenceGeometry,
                input: &std::ops::Range<u64>,
                position: u64,
                output: crate::OutputDemand,
            ) -> bool {
                self.inner
                    .matches_prefill(inference, input, position, output)
            }
            /// Original immutable admission; no source backing is implied.
            pub fn admission(&self) -> &'a AdmittedCapturePlan {
                self.inner.admission()
            }
            /// Original selection ordinal.
            pub fn selection_index(&self) -> usize {
                self.inner.selection_index()
            }
            /// Original observation phase.
            pub fn phase(&self) -> CapturePhase {
                self.inner.phase()
            }
            /// Original scheduling coordinate.
            pub fn prediction(&self) -> u64 {
                self.inner.prediction()
            }
            /// Physical source axes.
            pub fn source_shape(&self) -> &[usize] {
                self.inner.source_shape()
            }
            /// Exact selected axes.
            pub fn shape(&self) -> &[usize] {
                self.inner.shape()
            }
            /// Source-relative starts.
            pub fn starts(&self) -> &[u64] {
                self.inner.starts()
            }
            /// Source-relative exclusive ends.
            pub fn ends(&self) -> &[u64] {
                self.inner.ends()
            }
            /// Exact positive strides.
            pub fn strides(&self) -> &[u64] {
                self.inner.strides()
            }
            /// Selected value count, including an empty selection.
            pub fn elements(&self) -> usize {
                self.inner.elements()
            }
        }
    };
}
reduction_geometry!(
    CaptureSummaryGeometry,
    Summary,
    "Exact selected geometry for the finite-only Summary policy; no native authority."
);
reduction_geometry!(
    CaptureHistogramGeometry,
    Histogram,
    "Exact selected geometry and borrowed fixed edges for Histogram; no native authority."
);
impl<'a> CaptureHistogramGeometry<'a> {
    /// Edges borrowed from the original admission, not this temporary geometry.
    pub fn edges(&self) -> &'a [f32] {
        match &self.admission().plan().selections[self.selection_index()].transform {
            CaptureTransform::Histogram { edges } => edges,
            _ => unreachable!("closed histogram geometry"),
        }
    }
}

mod routed;
pub use routed::{CaptureRoutedUnitsGeometry, CaptureRoutedPrefillPlan, CaptureRoutedPrefillFragment, CaptureRoutedTokenWindow};
