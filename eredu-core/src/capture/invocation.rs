//! Physical geometry is independent of the prediction being inspected.
use super::*;
mod window_source;
pub use window_source::CaptureWindowSourceError;

/// Exact host-known tensor axes for one forward invocation. A cached forward may
/// contain multiple sequence rows. Context is optional only when no selected axis
/// needs it; absence never means zero or the generated prediction position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureInvocationShape {
    /// Number of independent sequences.
    pub batch: u64,
    /// Physical input rows per sequence in this invocation.
    pub sequence: u64,
    /// Actual context extent when known to the execution owner.
    pub context: Option<u64>,
}

/// Allocation-free refusal while checking an actual invocation against borrowed axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureAxisError {
    /// The independent logical prediction allowance is empty.
    #[error("empty invocation prediction range")]
    EmptyPredictionRange,
    /// The physical source or logical coordinate exceeds admitted bounds.
    #[error("invocation exceeds admitted geometry or prediction range")]
    InvocationBounds,
    /// The actual invocation has an empty required extent.
    #[error("empty invocation geometry")]
    Empty,
    /// An extent or product overflows.
    #[error("capture geometry overflow")]
    Overflow,
    /// An undeclared source exceeds the fixed metadata rank bound.
    #[error("capture rank exceeds the 32-axis metadata bound")]
    RankBound,
    /// The actual and declared ranks differ.
    #[error("runtime rank differs from the catalog")]
    Rank,
    /// A declared context axis lacks an exact source extent.
    #[error("selected context axis requires exact invocation context")]
    Context,
    /// An actual declared axis differs from its resolved extent.
    #[error("runtime axis {index} extent {actual} differs from expected {expected}")]
    Extent {
        /// Index in the declared axis order.
        index: usize,
        /// Actual source extent.
        actual: u64,
        /// Exact resolved extent.
        expected: u64,
    },
}
impl CaptureAxisError {
    fn legacy(self, point: Option<&ObservationPoint>) -> CaptureError {
        self.legacy_parts(point.and_then(|point| {
            point
                .axes
                .as_deref()
                .map(|axes| (point.path.as_str(), axes))
        }))
    }
    pub(crate) fn legacy_parts(self, point: Option<(&str, &[crate::TensorAxis])>) -> CaptureError {
        match self {
            Self::EmptyPredictionRange => {
                CaptureError::Invalid("empty invocation prediction range".into())
            }
            Self::InvocationBounds => CaptureError::Invalid(
                "invocation exceeds admitted geometry or prediction range".into(),
            ),
            Self::Empty => CaptureError::Invalid("empty invocation geometry".into()),
            Self::Overflow => CaptureError::Overflow,
            Self::RankBound => {
                CaptureError::Unsupported("capture rank exceeds the 32-axis metadata bound".into())
            }
            Self::Rank => CaptureError::Invalid("runtime rank differs from the catalog".into()),
            Self::Context => CaptureError::Unsupported(
                "selected context axis requires exact invocation context".into(),
            ),
            Self::Extent {
                index,
                actual,
                expected,
            } => {
                let (path, axes) = point.expect("extent diagnostic has declaration");
                CaptureError::Invalid(format!(
                    "runtime extent for {}/{} is {actual}, expected {expected} from catalog/request",
                    path, axes[index].name
                ))
            }
        }
    }
}
impl CaptureInvocationShape {
    /// Validates positive extents and checked products without touching tensors.
    pub fn validate(self) -> Result<(), CaptureError> {
        self.validate_fixed().map_err(|cause| cause.legacy(None))
    }
    /// Allocation-free validation for an already retained original descriptor.
    pub fn validate_fixed(self) -> Result<(), CaptureAxisError> {
        if self.batch == 0 || self.sequence == 0 || self.context == Some(0) {
            return Err(CaptureAxisError::Empty);
        }
        self.batch
            .checked_mul(self.sequence)
            .ok_or(CaptureAxisError::Overflow)?;
        if let Some(context) = self.context {
            self.batch
                .checked_mul(context)
                .ok_or(CaptureAxisError::Overflow)?;
        }
        Ok(())
    }
    fn extent_fixed(self, dimension: &SymbolicDimension) -> Result<Option<u64>, CaptureAxisError> {
        use CaptureAxisError as E;
        Ok(match dimension {
            SymbolicDimension::Known(n) => Some(u64::try_from(*n).map_err(|_| E::Overflow)?),
            SymbolicDimension::Batch => Some(self.batch),
            SymbolicDimension::Sequence => Some(self.sequence),
            SymbolicDimension::TokenRows => {
                Some(self.batch.checked_mul(self.sequence).ok_or(E::Overflow)?)
            }
            SymbolicDimension::Context => Some(self.context.ok_or(E::Context)?),
            SymbolicDimension::MediaPositions | SymbolicDimension::Unknown => None,
        })
    }
    pub(super) fn extent(self, dimension: &SymbolicDimension) -> Result<Option<u64>, CaptureError> {
        self.extent_fixed(dimension)
            .map_err(|cause| cause.legacy(None))
    }
    /// Checks borrowed source dimensions without constructing declaration clones
    /// or diagnostic storage. Ordinary and prepared geometry use this same loop.
    pub fn validate_actual_axes(
        self,
        axes: Option<&[crate::TensorAxis]>,
        shape: &[u64],
    ) -> Result<(), CaptureAxisError> {
        use CaptureAxisError as E;
        if self.batch == 0 || self.sequence == 0 || self.context == Some(0) {
            return Err(E::Empty);
        }
        self.batch.checked_mul(self.sequence).ok_or(E::Overflow)?;
        if let Some(n) = self.context {
            self.batch.checked_mul(n).ok_or(E::Overflow)?;
        }
        if let Some(axes) = axes {
            if axes.len() != shape.len() {
                return Err(E::Rank);
            }
            for (index, (axis, actual)) in axes.iter().zip(shape).enumerate() {
                if let Some(expected) = self.extent_fixed(&axis.dimension)? {
                    if expected != *actual {
                        return Err(E::Extent {
                            index,
                            actual: *actual,
                            expected,
                        });
                    }
                }
            }
        } else if shape.len() > 32 {
            return Err(E::RankBound);
        }
        elements(shape).map_err(|_| E::Overflow)?;
        Ok(())
    }
    /// Resolve borrowed symbolic axes into an existing exact-rank destination.
    /// Returns false for an unknown axis after checking every other declaration.
    pub fn resolve_axes_into(
        self,
        axes: &[crate::TensorAxis],
        output: &mut [u64],
    ) -> Result<bool, CaptureAxisError> {
        if axes.len() != output.len() {
            return Err(CaptureAxisError::Rank);
        }
        if self.batch == 0 || self.sequence == 0 || self.context == Some(0) {
            return Err(CaptureAxisError::Empty);
        }
        self.batch
            .checked_mul(self.sequence)
            .ok_or(CaptureAxisError::Overflow)?;
        if let Some(context) = self.context {
            self.batch
                .checked_mul(context)
                .ok_or(CaptureAxisError::Overflow)?;
        }
        let mut known = true;
        for (axis, out) in axes.iter().zip(output.iter_mut()) {
            match self.extent_fixed(&axis.dimension)? {
                Some(n) => *out = n,
                None => {
                    *out = 0;
                    known = false;
                }
            }
        }
        // Unknown placeholders do not conceal overflow among known axes.
        elements(output).map_err(|_| CaptureAxisError::Overflow)?;
        Ok(known)
    }
    /// Checks every declared axis even when other axes remain runtime-dependent.
    pub fn validate_actual(
        self,
        point: &ObservationPoint,
        shape: &[u64],
    ) -> Result<(), CaptureError> {
        self.validate_actual_axes(point.axes.as_deref(), shape)
            .map_err(|cause| cause.legacy(Some(point)))
    }
    /// Checks all known selected axes before execution, even in a partially
    /// declared tensor. Runtime still validates unknown extents before capture.
    pub fn validate_slices(
        self,
        point: &ObservationPoint,
        slices: &[CaptureSlice],
    ) -> Result<(), CaptureError> {
        self.validate()?;
        for slice in slices {
            let axis = point
                .axes
                .as_ref()
                .and_then(|axes| axes.iter().find(|axis| axis.name == slice.axis))
                .ok_or_else(|| CaptureError::Invalid(format!("unknown axis {}", slice.axis)))?;
            if self
                .extent(&axis.dimension)?
                .is_some_and(|extent| slice.end > extent)
            {
                return Err(CaptureError::Invalid(format!(
                    "slice {} exceeds invocation extent",
                    slice.axis
                )));
            }
        }
        Ok(())
    }
    /// Resolves declared axes without materializing a source. Unknown axes stay unknown.
    pub fn resolve(self, point: &ObservationPoint) -> Result<Option<Vec<u64>>, CaptureError> {
        self.validate()?;
        let Some(axes) = &point.axes else {
            return Ok(None);
        };
        let mut shape = vec![0; axes.len()];
        let known = self
            .resolve_axes_into(axes, &mut shape)
            .map_err(|cause| cause.legacy(Some(point)))?;
        Ok(known.then_some(shape))
    }
}

/// Logical row placement of an explicitly split invocation. This is geometry,
/// not source, quota, execution or completion authority. Prediction schedules
/// remain independent of this prompt-relative origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureInvocationWindow {
    /// Total logical rows in this phase (shifted seed phases can be shorter).
    pub logical_sequence: u64,
    /// Logical coordinate of the first physical row.
    pub start: u64,
}
impl CaptureInvocationWindow {
    /// Checks the complete physical span before any source work.
    pub fn validate(
        self,
        physical: CaptureInvocationShape,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        self.validate_fixed(physical).map_err(|cause|cause.legacy(None))
    }
    /// Resolves one actual tensor and a contiguous row map using declared axes.
    /// Multiple/unknown row interpretations require their own composition contract.
    pub fn source_axes(
        self,
        physical: CaptureInvocationShape,
        point: &ObservationPoint,
        actual: &[u64],
    ) -> Result<(Vec<u64>, usize, crate::component::ComponentCoordinateMap), CaptureError> {
        let plan=self.source_axes_plan(physical,point.axes.as_deref(),actual).map_err(|cause|cause.legacy(Some(point)))?;
        let logical=plan.logical;
        let axis=plan.axis;
        let mut global=actual.to_vec();
        global[axis]=logical.sequence;
        logical.validate_actual(point,&global)?;
        let host = |value| usize::try_from(value).map_err(|_| CaptureError::Overflow);
        let coordinates = crate::component::ComponentCoordinateMap::range(
            host(logical.sequence)?,
            host(self.start)?..host(self.start + physical.sequence)?,
        )
        .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        Ok((global, axis, coordinates))
    }
    /// Projects the original global selection; stride origins never reset at a
    /// span boundary. A no-overlap window has no fragment, not a fabricated value.
    pub fn project(
        self,
        physical: CaptureInvocationShape,
        point: &ObservationPoint,
        selection: &CaptureSelection,
        actual: &[u64],
    ) -> Result<CaptureSlicePartition, CaptureError> {
        let (global, axis, coordinates) = self.source_axes(physical, point, actual)?;
        let selected = resolve_slice(point, selection, &global)?;
        CaptureSlicePartition::new(&global, &selected, axis, &coordinates, 1)
    }
}

/// Admission bounds for independently shaped invocations sharing one capture
/// ledger. Prediction schedules retain their own coordinates and do not determine
/// physical sequence or context extents. Repeated invocations spend cumulative
/// limits even when their prediction coordinate repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureInvocationBounds {
    /// Exact batch extent for this run.
    pub batch: u64,
    /// Largest permitted physical sequence extent.
    pub max_sequence: u64,
    /// Largest context extent, or no admission for context-dependent axes.
    pub max_context: Option<u64>,
    /// Exclusive upper bound for the independent prediction schedule coordinate.
    pub max_predictions: u64,
}
impl CaptureInvocationBounds {
    /// Conservative geometry for static shape and budget checks.
    pub fn maximum(self) -> Result<CaptureInvocationShape, CaptureError> {
        self.maximum_fixed().map_err(|cause| cause.legacy(None))
    }
    fn maximum_fixed(self) -> Result<CaptureInvocationShape, CaptureAxisError> {
        if self.max_predictions == 0 {
            return Err(CaptureAxisError::EmptyPredictionRange);
        }
        let shape = CaptureInvocationShape {
            batch: self.batch,
            sequence: self.max_sequence,
            context: self.max_context,
        };
        shape.validate_fixed()?;
        Ok(shape)
    }
    /// Checks caller-supplied actual geometry before any observed execution.
    pub fn validate(
        self,
        shape: CaptureInvocationShape,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        self.validate_fixed(shape, prediction)
            .map_err(|cause| cause.legacy(None))
    }
    /// The identical bounds/overflow ordering with a fixed typed failure. This
    /// read-only check creates no source, invocation or execution authority.
    pub fn validate_fixed(
        self,
        shape: CaptureInvocationShape,
        prediction: u64,
    ) -> Result<(), CaptureAxisError> {
        self.maximum_fixed()?;
        shape.validate_fixed()?;
        if shape.batch != self.batch
            || shape.sequence > self.max_sequence
            || prediction >= self.max_predictions
            || match (shape.context, self.max_context) {
                (Some(actual), Some(maximum)) => actual > maximum,
                (Some(_), None) => true,
                _ => false,
            }
        {
            return Err(CaptureAxisError::InvocationBounds);
        }
        Ok(())
    }
    pub(super) fn request(self) -> CaptureRequestShape {
        CaptureRequestShape {
            batch: self.batch,
            prompt_tokens: self.max_sequence,
            max_predictions: self.max_predictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    fn point(dimensions: &[SymbolicDimension]) -> ObservationPoint {
        ObservationPoint {
            path: "fixture.value".into(),
            node_id: "fixture".into(),
            meaning: "geometry".into(),
            value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: Some(
                dimensions
                    .iter()
                    .enumerate()
                    .map(|(i, dimension)| TensorAxis {
                        name: format!("axis{i}"),
                        dimension: dimension.clone(),
                    })
                    .collect(),
            ),
            prefill: true,
            decode: true,
            requirements: vec![],
            position: ObservationPosition::BeforeIntervention,
            retained_bytes: None,
            host_bytes: None,
        }
    }
    #[test]
    fn physical_sequence_context_and_prediction_coordinates_are_independent() {
        let bounds = CaptureInvocationBounds {
            batch: 2,
            max_sequence: 4,
            max_context: Some(17),
            max_predictions: 6,
        };
        let shape = CaptureInvocationShape {
            batch: 2,
            sequence: 3,
            context: Some(11),
        };
        let point = point(&[
            SymbolicDimension::Batch,
            SymbolicDimension::Sequence,
            SymbolicDimension::TokenRows,
            SymbolicDimension::Context,
        ]);
        bounds.validate(shape, 0).unwrap();
        bounds.validate(shape, 5).unwrap();
        assert!(bounds.validate(shape, 6).is_err());
        assert_eq!(shape.resolve(&point).unwrap(), Some(vec![2, 3, 6, 11]));
        shape.validate_actual(&point, &[2, 3, 6, 11]).unwrap();
        assert!(shape.validate_actual(&point, &[2, 1, 2, 11]).is_err());
        assert!(shape.validate_actual(&point, &[2, 3, 6, 5]).is_err());
        assert_eq!(
            serde_json::from_value::<CaptureInvocationShape>(serde_json::to_value(shape).unwrap())
                .unwrap(),
            shape
        );
        let ordinary = CaptureRequestShape {
            batch: 2,
            prompt_tokens: 4,
            max_predictions: 6,
        };
        assert_eq!(
            ordinary.resolve(&point, CapturePhase::Decode, 5).unwrap(),
            Some(vec![2, 1, 2, 9])
        );
    }
    #[test]
    fn unknown_axes_do_not_hide_context_requirements_or_invalid_known_axes() {
        let unknown_context = point(&[SymbolicDimension::Unknown, SymbolicDimension::Context]);
        let shape = CaptureInvocationShape {
            batch: 1,
            sequence: 3,
            context: None,
        };
        assert!(matches!(
            shape.resolve(&unknown_context),
            Err(CaptureError::Unsupported(_))
        ));
        assert!(matches!(
            shape.validate_actual(&unknown_context, &[8, 9]),
            Err(CaptureError::Unsupported(_))
        ));
        let partly_known = point(&[SymbolicDimension::Unknown, SymbolicDimension::Sequence]);
        assert_eq!(shape.resolve(&partly_known).unwrap(), None);
        assert!(shape.validate_actual(&partly_known, &[8, 1]).is_err());
        assert!(
            shape
                .validate_slices(
                    &partly_known,
                    &[CaptureSlice {
                        axis: "axis1".into(),
                        start: 0,
                        end: 4,
                        stride: 1
                    }]
                )
                .is_err()
        );
        assert!(
            CaptureInvocationShape {
                batch: u64::MAX,
                sequence: 2,
                context: None
            }
            .validate()
            .is_err()
        );
        let schedule = CaptureSchedule {
            first_prediction: 3,
            every: 2,
            ..Default::default()
        };
        assert_eq!(
            schedule
                .count_coordinates(CapturePhase::Prefill, 0, 8)
                .unwrap(),
            Some((3, 7))
        );
        assert_eq!(
            schedule.count_and_last(CapturePhase::Prefill, 8).unwrap(),
            None
        );
        assert_eq!(
            CaptureSchedule::default()
                .count_coordinates(CapturePhase::Decode, 0, 1)
                .unwrap(),
            Some((1, 0))
        );
    }
}
