//! Physical geometry is independent of the prediction being inspected.
use super::*;

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

impl CaptureInvocationShape {
    /// Validates positive extents and checked products without touching tensors.
    pub fn validate(self) -> Result<(), CaptureError> {
        if self.batch == 0 || self.sequence == 0 || self.context == Some(0) {
            return Err(CaptureError::Invalid("empty invocation geometry".into()));
        }
        mul(self.batch, self.sequence)?;
        if let Some(context) = self.context {
            mul(self.batch, context)?;
        }
        Ok(())
    }
    pub(super) fn extent(self, dimension: &SymbolicDimension) -> Result<Option<u64>, CaptureError> {
        Ok(match dimension {
            SymbolicDimension::Known(n) => {
                Some(u64::try_from(*n).map_err(|_| CaptureError::Overflow)?)
            }
            SymbolicDimension::Batch => Some(self.batch),
            SymbolicDimension::Sequence => Some(self.sequence),
            SymbolicDimension::TokenRows => Some(mul(self.batch, self.sequence)?),
            SymbolicDimension::Context => Some(self.context.ok_or_else(|| {
                CaptureError::Unsupported(
                    "selected context axis requires exact invocation context".into(),
                )
            })?),
            SymbolicDimension::MediaPositions | SymbolicDimension::Unknown => None,
        })
    }
    /// Checks every declared axis even when other axes remain runtime-dependent.
    pub fn validate_actual(
        self,
        point: &ObservationPoint,
        shape: &[u64],
    ) -> Result<(), CaptureError> {
        self.validate()?;
        let Some(axes) = &point.axes else {
            if shape.len() > 32 {
                return Err(CaptureError::Unsupported(
                    "capture rank exceeds the 32-axis metadata bound".into(),
                ));
            }
            return elements(shape).map(|_| ());
        };
        if axes.len() != shape.len() {
            return Err(CaptureError::Invalid(
                "runtime rank differs from the catalog".into(),
            ));
        }
        for (axis, actual) in axes.iter().zip(shape) {
            if let Some(expected) = self.extent(&axis.dimension)? {
                if expected != *actual {
                    return Err(CaptureError::Invalid(format!(
                        "runtime extent for {}/{} is {actual}, expected {expected} from catalog/request",
                        point.path, axis.name
                    )));
                }
            }
        }
        elements(shape).map(|_| ())
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
        let mut shape = Vec::with_capacity(axes.len());
        // Do not return at the first unknown: later context and overflow checks
        // remain mandatory for a partially declared shape.
        let mut known = true;
        for axis in axes {
            match self.extent(&axis.dimension)? {
                Some(n) => shape.push(n),
                None => known = false,
            }
        }
        elements(&shape)?;
        Ok(known.then_some(shape))
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
        if self.max_predictions == 0 {
            return Err(CaptureError::Invalid(
                "empty invocation prediction range".into(),
            ));
        }
        let shape = CaptureInvocationShape {
            batch: self.batch,
            sequence: self.max_sequence,
            context: self.max_context,
        };
        shape.validate()?;
        Ok(shape)
    }
    /// Checks caller-supplied actual geometry before any observed execution.
    pub fn validate(
        self,
        shape: CaptureInvocationShape,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        self.maximum()?;
        shape.validate()?;
        if shape.batch != self.batch
            || shape.sequence > self.max_sequence
            || prediction >= self.max_predictions
            || match (shape.context, self.max_context) {
                (Some(actual), Some(maximum)) => actual > maximum,
                (Some(_), None) => true,
                _ => false,
            }
        {
            return Err(CaptureError::Invalid(
                "invocation exceeds admitted geometry or prediction range".into(),
            ));
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
        assert!(shape
            .validate_slices(
                &partly_known,
                &[CaptureSlice {
                    axis: "axis1".into(),
                    start: 0,
                    end: 4,
                    stride: 1
                }]
            )
            .is_err());
        assert!(CaptureInvocationShape {
            batch: u64::MAX,
            sequence: 2,
            context: None
        }
        .validate()
        .is_err());
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
