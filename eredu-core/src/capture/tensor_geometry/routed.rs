//! Exact sparse route geometry using the ordinary admitted axis resolver.
use super::*;
mod prefill;
pub use prefill::*;

/// Borrowed sparse bank and selected token/route/unit axes. Construction performs
/// no source read or allocation and grants no host or native execution authority.
#[derive(Debug)]
pub struct CaptureRoutedUnitsGeometry<'a> {
    inner: CaptureTensorGeometry<'a>,
    bank: RoutedUnitGeometry,
}
impl<'a> CaptureRoutedUnitsGeometry<'a> {
    /// Resolve an active sparse selection through the shared invocation geometry.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let inner = CaptureTensorGeometry::prepare_kind(
            source,
            selection,
            phase,
            prediction,
            invocation,
            GeometryTransform::RoutedUnits,
        )?;
        Self::from_inner(inner)
    }

    fn from_inner(inner: CaptureTensorGeometry<'a>) -> Result<Self, CaptureTensorGeometryError> {
        let ObservationValueType::RoutedUnits { geometry: bank, .. } =
            inner.admission().points()[inner.selection_index()].value_type
        else {
            return Err(CaptureTensorGeometryError::Unsupported);
        };
        if inner.source_shape().len() != 3
            || inner.source_shape()[1] as u64 != bank.routes_per_token
            || inner.source_shape()[2] as u64 != bank.units_per_expert
        {
            return Err(RoutedUnitValidationError::SliceExtent.into());
        }
        let shape = [
            inner.shape()[0] as u64,
            inner.shape()[1] as u64,
            inner.shape()[2] as u64,
        ];
        bank.validate_axes(inner.starts(), inner.ends(), inner.strides(), &shape)?;
        Ok(Self { inner, bank })
    }

    /// Project an explicitly bounded invocation into its original logical window.
    /// Flattened token axes retain the shared batch and sequence interpretation.
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
        let inner = CaptureTensorGeometry::prepare_kind(
            source,
            selection,
            phase,
            prediction,
            Some(logical),
            GeometryTransform::RoutedUnits,
        )?
        .apply_window(physical, window)?;
        Self::from_inner(inner)
    }

    /// Exact global coordinates of one original sparse unit fragment. The four
    /// fixed rows are starts, ends, strides and shape. Ordinary and paid workers
    /// use this same mapping; it creates no owning slice or native source.
    pub fn fragment_axes(projection: &CaptureSlicePartition, fragment: usize)
        -> Result<[[u64; 3]; 4], RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        let selected = projection.global_slice();
        let destination = projection.fragments().get(fragment).ok_or(E::Selection)?.destination();
        if [selected.starts.len(), selected.strides.len(), destination.starts.len(),
            destination.strides.len(), destination.shape.len()].iter().any(|n| *n != 3) {
            return Err(E::SliceExtent);
        }
        let mut axes = [[0; 3]; 4];
        for axis in 0..3 {
            let start = destination.starts[axis].checked_mul(selected.strides[axis])
                .and_then(|n| selected.starts[axis].checked_add(n)).ok_or(E::Overflow)?;
            let stride = destination.strides[axis].checked_mul(selected.strides[axis]).ok_or(E::Overflow)?;
            let count = destination.shape[axis];
            let end = if count == 0 { start } else {
                (count - 1).checked_mul(stride).and_then(|n| start.checked_add(n))
                    .and_then(|n| n.checked_add(1)).ok_or(E::Overflow)?
            };
            axes[0][axis] = start; axes[1][axis] = end;
            axes[2][axis] = stride; axes[3][axis] = count;
        }
        Ok(axes)
    }
    /// Borrow one receipt fragment while preserving the original global bank
    /// and token/route identities. Local native columns remain a separate source.
    pub fn prepare_partition(source: &'a AdmittedCapturePlan, selection: usize,
        phase: CapturePhase, prediction: u64, invocation: Option<CaptureInvocationShape>,
        projection: &CaptureSlicePartition, fragment: usize) -> Result<Self, CaptureTensorGeometryError> {
        if projection.axis() != 2 { return Err(CaptureTensorGeometryError::Partition); }
        let mut inner = CaptureTensorGeometry::prepare_kind(source, selection, phase,
            prediction, invocation, GeometryTransform::RoutedUnits)?;
        // Authenticate the exact global selection and local projection using the
        // existing geometry worker before translating its selected coordinates.
        inner.clone().apply_partition(projection, fragment, PartitionCaptureCombination::Disjoint)?;
        let axes = Self::fragment_axes(projection, fragment)?;
        inner.elements = 1;
        for axis in 0..3 {
            inner.starts[axis] = axes[0][axis]; inner.ends[axis] = axes[1][axis];
            inner.strides[axis] = axes[2][axis];
            inner.shape[axis] = usize::try_from(axes[3][axis]).map_err(|_| CaptureTensorGeometryError::Overflow)?;
            inner.elements = inner.elements.checked_mul(inner.shape[axis]).ok_or(CaptureTensorGeometryError::Overflow)?;
        }
        Self::from_inner(inner)
    }
    /// Fixed fragment mapping/projection frames, excluding all caller buffers.
    pub fn partition_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [Self::preparation_control_bytes()?, CaptureTensorGeometry::partition_preparation_control_bytes()?,
            size_of::<[[u64; 3]; 4]>() * 2, size_of::<Result<[[u64; 3]; 4], RoutedUnitValidationError>>(),
            size_of::<Self>() * 2, size_of::<Result<Self, CaptureTensorGeometryError>>(),
            size_of::<(&AdmittedCapturePlan, usize, CapturePhase, u64, Option<CaptureInvocationShape>, &CaptureSlicePartition, usize)>(),
            size_of::<(&CaptureSlicePartition, usize)>(), size_of::<(usize, u64, u64, u64, u64)>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Fixed constructor and validation control, excluding callers' owned storage.
    pub fn preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, CaptureTensorGeometryError>>(),
            size_of::<[u64; 3]>(),
            CaptureTensorGeometry::preparation_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// The immutable admission that owns this selection.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.inner.admission()
    }
    /// Original selected point ordinal.
    pub fn selection_index(&self) -> usize {
        self.inner.selection_index()
    }
    /// Original forward phase.
    pub fn phase(&self) -> CapturePhase {
        self.inner.phase()
    }
    /// Original prediction coordinate.
    pub fn prediction(&self) -> u64 {
        self.inner.prediction()
    }
    /// Global bank geometry; expert count never expands the sparse row extent.
    pub fn bank(&self) -> RoutedUnitGeometry {
        self.bank
    }
    /// Declared routing path borrowed from the immutable admission.
    pub fn routing(&self) -> &'a str {
        match &self.admission().points()[self.selection_index()].value_type {
            ObservationValueType::RoutedUnits { routing, .. } => routing,
            _ => unreachable!("closed routed geometry"),
        }
    }
    /// Actual source token/route/unit extents.
    pub fn source_shape(&self) -> &[usize] {
        self.inner.source_shape()
    }
    /// Selected token/route/unit extents.
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
    /// Source-relative first coordinates.
    pub fn starts(&self) -> &[u64] {
        self.inner.starts()
    }
    /// Source-relative exclusive ends.
    pub fn ends(&self) -> &[u64] {
        self.inner.ends()
    }
    /// Original positive strides.
    pub fn strides(&self) -> &[u64] {
        self.inner.strides()
    }
    /// Exact selected sparse row capacity, excluding expert-dense expansion.
    pub fn rows(&self) -> usize {
        self.inner.shape()[0] * self.inner.shape()[1]
    }
    /// Exact selected F32 scalar capacity.
    pub fn elements(&self) -> usize {
        self.inner.elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    pub(super) fn admission(units_axis: usize) -> AdmittedCapturePlan {
        let point = ObservationPoint {
            path: "routed.units".into(),
            node_id: "layer".into(),
            meaning: "selected sparse units".into(),
            value_type: ObservationValueType::RoutedUnits {
                routing: "routed.dispatch".into(),
                geometry: RoutedUnitGeometry {
                    experts: 4096,
                    units_per_expert: 7,
                    routes_per_token: 3,
                },
            },
            dtype: ObservationDtype::Floating,
            axes: Some(vec![
                TensorAxis {
                    name: "token".into(),
                    dimension: SymbolicDimension::TokenRows,
                },
                TensorAxis {
                    name: "route".into(),
                    dimension: SymbolicDimension::Known(3),
                },
                TensorAxis {
                    name: "unit".into(),
                    dimension: SymbolicDimension::Known(units_axis),
                },
            ]),
            prefill: true,
            decode: true,
            requirements: vec![ObservationRequirement::ActivationHooks],
            position: ObservationPosition::BeforeIntervention,
            retained_bytes: None,
            host_bytes: None,
        };
        let catalog = ObservationCatalog {
            schema_version: 1,
            points: vec![point],
            completeness: DescriptionCompleteness::Complete,
        };
        let capabilities = CaptureCapabilities {
            transformations: vec![CaptureTransformKind::RoutedUnits],
            ..Default::default()
        };
        let support = ObservationSupportReport {
            schema_version: 1,
            capture: capabilities.clone(),
            points: vec![ObservationSupport {
                path: "routed.units".into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        };
        let usage = CaptureUsage {
            captures: 10,
            retained_bytes: u64::MAX,
            host_bytes: u64::MAX,
            encoded_bytes: u64::MAX,
        };
        CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "sparse".into(),
                path: "routed.units".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![
                    CaptureSlice {
                        axis: "token".into(),
                        start: 1,
                        end: 10,
                        stride: 3,
                    },
                    CaptureSlice {
                        axis: "route".into(),
                        start: 0,
                        end: 3,
                        stride: 2,
                    },
                    CaptureSlice {
                        axis: "unit".into(),
                        start: 1,
                        end: 7,
                        stride: 2,
                    },
                ],
                transform: CaptureTransform::RoutedUnits,
            }],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &capabilities,
            CaptureRequestShape {
                batch: 2,
                prompt_tokens: 5,
                max_predictions: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn routed_geometry_retains_sparse_bank_and_strided_token_coordinates() {
        let source = admission(7);
        let selected =
            CaptureRoutedUnitsGeometry::prepare(&source, 0, CapturePhase::Prefill, 0, None)
                .unwrap();
        assert_eq!(selected.source_shape(), [10, 3, 7]);
        assert_eq!(selected.shape(), [3, 2, 3]);
        assert_eq!(selected.starts(), [1, 0, 1]);
        assert_eq!(selected.strides(), [3, 2, 2]);
        assert_eq!(selected.rows(), 6);
        assert_eq!(selected.elements(), 18);
        assert_eq!(selected.bank().experts, 4096);
        assert_eq!(selected.routing(), "routed.dispatch");
        assert!(std::ptr::eq(selected.admission(), &source));
        let ordinary = source
            .geometry_at(CapturePhase::Prefill, 0, None)
            .unwrap()
            .resolve(&source.points()[0])
            .unwrap()
            .unwrap();
        assert_eq!(
            ordinary,
            selected
                .source_shape()
                .iter()
                .map(|n| *n as u64)
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            CaptureTensorGeometry::prepare(&source, 0, CapturePhase::Prefill, 0, None),
            Err(CaptureTensorGeometryError::Unsupported)
        ));
        assert!(matches!(
            CaptureRoutedUnitsGeometry::prepare(&source, 0, CapturePhase::Decode, 1, None),
            Err(CaptureTensorGeometryError::Inactive)
        ));
        let changed = admission(8);
        assert!(matches!(
            CaptureRoutedUnitsGeometry::prepare(&changed, 0, CapturePhase::Prefill, 0, None),
            Err(CaptureTensorGeometryError::Routed(
                RoutedUnitValidationError::SliceExtent
            ))
        ));
    }
}

#[cfg(test)]
mod prefill_tests {
    use super::*;
    #[test]
    fn sparse_prefill_maps_uneven_batched_tokens_without_inventing_source_coverage(){
        let source=super::tests::admission(7);
        let inference=crate::InferenceGeometry {batch_size:2,cached_positions:0,input_positions:5,
            max_output_tokens:3,prefill_chunk_positions:2,output:crate::OutputDemand::LastPosition};
        let plan=CaptureRoutedPrefillPlan::prepare(&source,0,inference).unwrap();
        let expected=[vec![0,1,5,6],vec![2,3,7,8],vec![4,9]];
        let mut selected=Vec::new();let mut ranges=Vec::new();
        for (index,tokens) in expected.iter().enumerate(){
            let fragment=plan.fragment(index as u64).unwrap();
            assert_eq!((0..fragment.source_tokens()).map(|i|fragment.logical_token(i).unwrap()).collect::<Vec<_>>(),*tokens);
            assert_eq!(fragment.logical_token(fragment.source_tokens()),None);
            for i in 0..fragment.source_tokens(){if fragment.selects(i,0){selected.push(fragment.logical_token(i).unwrap());}}
            ranges.extend(fragment.source_ranges(0,fragment.source_tokens()).unwrap());
            assert!(fragment.source_ranges(0,fragment.source_tokens()+1).is_none());
        }
        assert_eq!(ranges,[[0,2],[5,7],[2,4],[7,9],[4,5],[9,10]]);
        selected.sort_unstable();assert_eq!(selected,[1,4,7]);
        assert!(plan.fragment(3).is_err());
    }
}
