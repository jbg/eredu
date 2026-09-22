//! Metadata realization of the shared effective-parameter value workers.
use super::*;
use crate::{
    Error, Index, Tensor,
    workspace::{WorkspaceContext, WorkspaceFloatingType, WorkspaceMetadataError, WorkspaceTensor},
};
use eredu_core::parameters::{ParameterProjection, ParameterRegion};

/// A borrowed, already projected parameter and its selected host request.
/// The source keeps its actual backing/placement in the supplied trace context.
pub struct WorkspaceParameterValues<'a> {
    source: &'a WorkspaceTensor,
    region: &'a ParameterRegion,
    projection: Option<&'a ParameterProjection>,
    context: &'a WorkspaceContext,
    coefficients: Option<fn(&[i32], &WorkspaceContext) -> Result<WorkspaceTensor, Error>>,
}
impl<'a> WorkspaceParameterValues<'a> {
    /// Prepare metadata sequencing for the selected rectangular read.
    pub fn read(
        source: &'a WorkspaceTensor,
        region: &'a ParameterRegion,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>)>())?;
        context.validate_values([source])?;
        Ok(Self {
            source,
            region,
            projection: None,
            context,
            coefficients: None,
        })
    }
    /// Prepare metadata sequencing for the actual F32 coefficient source.
    /// The native producer must independently authenticate that eager source.
    pub fn projection(
        source: &'a WorkspaceTensor,
        projection: &'a ParameterProjection,
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        let mut value = Self::read(source, &projection.region, context)?;
        value.projection = Some(projection);
        Ok(value)
    }
    /// Trace the actual eager coefficient constructor selected by the backend.
    /// The callback describes storage only; the numerical producer retains the
    /// borrowed coefficient payload and establishes its own execution authority.
    pub fn projection_with_coefficients(
        source: &'a WorkspaceTensor,
        projection: &'a ParameterProjection,
        context: &'a WorkspaceContext,
        coefficients: fn(&[i32], &WorkspaceContext) -> Result<WorkspaceTensor, Error>,
    ) -> Result<Self, Error> {
        let mut value = Self::projection(source, projection, context)?;
        value.coefficients = Some(coefficients);
        Ok(value)
    }
    fn projection_request(&self) -> Result<&ParameterProjection, Error> {
        self.projection
            .ok_or_else(|| WorkspaceMetadataError::Unqualified.into())
    }
    fn extent(value: u64) -> Result<i32, Error> {
        i32::try_from(value).map_err(|_| WorkspaceMetadataError::Overflow.into())
    }
}
impl ParameterValueMechanism for WorkspaceParameterValues<'_> {
    type Value = WorkspaceTensor;
    type Output = ();
    type Error = Error;
    fn select(&self) -> Result<WorkspaceTensor, Error> {
        if self.region.starts.len() != self.region.shape.len()
            || self.region.shape.len() != self.source.shape().len()
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut indexes = self.context.metadata_vec(self.region.shape.len())?;
        for (&start, &count) in self.region.starts.iter().zip(&self.region.shape) {
            indexes.push(Index::Range(
                Self::extent(start)?,
                Self::extent(
                    start
                        .checked_add(count)
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )?,
            ));
        }
        self.source.index(&indexes, self.context)
    }
    fn cast_f32(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.cast_floating(WorkspaceFloatingType::Float32, self.context)
    }
    fn contiguous(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.contiguous(self.context)
    }
    fn complete_and_read(&self, value: WorkspaceTensor) -> Result<(), Error> {
        self.context.complete_values(&[&value])
    }
}
impl ParameterProjectionMechanism for WorkspaceParameterValues<'_> {
    fn transpose_weights(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        let projection = self.projection_request()?;
        let rank = self.region.shape.len();
        if projection.axis >= rank {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut axes = self.context.metadata_vec(rank)?;
        for axis in 0..rank {
            if axis != projection.axis {
                axes.push(i32::try_from(axis).map_err(|_| WorkspaceMetadataError::Overflow)?);
            }
        }
        axes.push(i32::try_from(projection.axis).map_err(|_| WorkspaceMetadataError::Overflow)?);
        value.transpose_axes(&axes, self.context)
    }
    fn reshape_weights(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        let projection = self.projection_request()?;
        let width = *self
            .region
            .shape
            .get(projection.axis)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        value.reshape(&[-1, Self::extent(width)?], self.context)
    }
    fn coefficients(&self) -> Result<WorkspaceTensor, Error> {
        let projection = self.projection_request()?;
        let width = *self
            .region
            .shape
            .get(projection.axis)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let count = width
            .checked_mul(projection.directions)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if usize::try_from(count).ok() != Some(projection.coefficients.len()) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let shape = [Self::extent(projection.directions)?, Self::extent(width)?];
        if let Some(construct) = self.coefficients {
            construct(&shape, self.context)
        } else {
            WorkspaceTensor::from_f32_slice(&projection.coefficients, &shape, self.context)
        }
    }
    fn transpose_coefficients(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.transpose(self.context)
    }
    fn matmul(
        &self,
        weights: &WorkspaceTensor,
        coefficients: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        WorkspaceTensor::matmul(weights, coefficients, self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::*;

    #[derive(Debug)]
    struct LayoutFacts;
    impl WorkspaceMechanisms for LayoutFacts {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            let alias = matches!(
                operation.kind,
                WorkspaceOperationKind::View(_)
                    | WorkspaceOperationKind::Transpose(_)
                    | WorkspaceOperationKind::StaticSlice { .. }
                    | WorkspaceOperationKind::CastFloating(_)
            );
            Ok(Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|layout| {
                        if alias {
                            Ok(WorkspaceOutputStorage::AliasInput(0))
                        } else {
                            layout.bytes().map(WorkspaceOutputStorage::Allocate)
                        }
                    })
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 0,
                assumptions:
                    "fixture metadata views alias; numerical outputs own their logical storage"
                        .into(),
            }))
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "fixture has no external host workspace".into(),
            }))
        }
    }

    #[test]
    fn contraction_axis_controls_output_geometry_and_completes_the_selected_result() {
        for axis in 0..3 {
            let context = WorkspaceContext::new(LayoutFacts);
            let source = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[2, 3, 4], WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap();
            let region = ParameterRegion {
                starts: vec![0, 1, 1],
                shape: vec![2, 2, 3],
            };
            let width = region.shape[axis];
            let projection = ParameterProjection {
                region,
                axis,
                directions: 2,
                coefficients: (0..width * 2).map(|i| i as f32 - 2.).collect(),
            };
            project_parameter(
                WorkspaceParameterValues::projection(&source, &projection, &context).unwrap(),
            )
            .unwrap();
            let report = context.report(&[source]).unwrap();
            let completion = report.operations.last().unwrap();
            assert!(matches!(
                completion.kind,
                WorkspaceOperationKind::ValueCompletion
            ));
            assert_eq!(completion.inputs[0].shape(), [12 / width as i32, 2]);
            assert!(completion.outputs.is_empty());
            let uploaded = report
                .operations
                .iter()
                .find(|operation| {
                    matches!(
                        operation.kind,
                        WorkspaceOperationKind::Elementwise("from_f32_slice")
                    )
                })
                .unwrap();
            assert_eq!(uploaded.outputs[0].shape(), [2, width as i32]);
            assert!(uploaded.inputs.is_empty());
            assert_eq!(report.closing_storage.maximum_allocations, 1);
            assert_eq!(report.closing_storage.bytes, Some(2 * 3 * 4 * 4));
        }
    }
}
