//! The existing native workers for one source-bound projected invocation.
use super::*;
use eredu_runtime::capture::partition::{
    PartitionInvocationCaptureGeometry, PartitionInvocationCaptureKind,
};

pub(in crate::composition::mlx::session) fn estimate_geometry(
    source: &PartitionInvocationCaptureGeometry<'_>,
) -> std::result::Result<PartitionCaptureNativeEstimate, CaptureError> {
    let capture = match source.kind() {
        PartitionInvocationCaptureKind::Tensor(geometry) => {
            super::super::super::bounded_capture::estimate_tensor_geometry(geometry)
        }
        PartitionInvocationCaptureKind::Summary(geometry) => {
            super::super::super::bounded_capture::estimate_summary(geometry)
        }
        PartitionInvocationCaptureKind::Histogram(geometry) => {
            super::super::super::bounded_capture::estimate_histogram(geometry)
        }
    }?;
    Ok(PartitionCaptureNativeEstimate {
        capture,
        generated_creation_bytes: 0,
    })
}

/// Descriptive source shared with the actual paid Host fragment plan. Its
/// scalar still comes from the observed local value and the existing Source vote.
pub(super) struct PartitionInvocationEquation<'a> {
    source: PartitionInvocationCaptureGeometry<'a>,
    producer: usize,
    estimate: PartitionCaptureNativeEstimate,
}
impl<'a> PartitionInvocationEquation<'a> {
    pub(super) fn prepare(
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
        context: &WorkspaceContext,
    ) -> Result<Self> {
        let metadata = Metadata::new(context)?;
        context.charge_metadata(
            Self::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        // Independent model callbacks use the same enclosing numerical scope.
        // Their retained axes change geometry, not the completion worker.
        let source = PartitionInvocationCaptureGeometry::from_receipt(receipt, producer, fragment)
            .map_err(|cause| metadata.error(cause))?;
        Self::from_geometry(source, producer, context)
    }
    pub(super) fn from_geometry(
        source: PartitionInvocationCaptureGeometry<'a>,
        producer: usize,
        context: &WorkspaceContext,
    ) -> Result<Self> {
        let metadata = Metadata::new(context)?;
        context.charge_metadata(
            Self::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        let estimate = estimate_geometry(&source).map_err(|cause| metadata.error(cause))?;
        Ok(Self {
            source,
            producer,
            estimate,
        })
    }

    pub(super) fn native_source(&self, dtype: TensorDtype) -> PartitionCaptureFragmentSource<'_> {
        let projection = self.source.projection();
        PartitionCaptureFragmentSource {
            producer: self.producer,
            fragment: self.source.fragment_index(),
            local_shape: projection.local_shape(),
            local_slice: projection.fragments()[self.source.fragment_index()].local(),
            transform: self.source.native_transform(),
            dtype,
            estimate: self.estimate,
        }
    }
    /// Trace the same raw selection or typed reducer used by ordinary capture.
    /// These are ordinary per-invocation completion populations. A retained
    /// numerical child uses its own explicit nested-completion source instead.
    pub(super) fn trace(
        &self,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
        roots: &mut Vec<WorkspaceTensor>,
    ) -> Result<(WorkspaceFloatingType, CaptureNativePopulation)> {
        let metadata = Metadata::new(context)?;
        context.charge_metadata(
            Self::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        context.validate_values([value])?;
        let dtype = value
            .layout()
            .representation()
            .map(|r| r.dtype())
            .ok_or_else(|| metadata.coordinate())?;
        let population = match self.source.kind() {
            PartitionInvocationCaptureKind::Tensor(geometry) => {
                let program = CaptureTensorSelection::from_geometry(geometry)
                    .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, context)
                    .map_err(|cause| metadata.error(cause))?;
                metadata.reserve(roots, 1)?;
                roots.push(value.clone());
                program
                    .trace_retained_within(value, context, roots)
                    .map_err(|cause| metadata.error(cause))?;
                CaptureNativePopulation::raw(1)
            }
            PartitionInvocationCaptureKind::Summary(geometry) => {
                let program = PreparedCaptureSummary::from_geometry(geometry)
                    .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, context)
                    .map_err(|cause| metadata.error(cause))?;
                metadata.reserve(roots, 1)?;
                roots.push(value.clone());
                program
                    .trace(value, context, roots)
                    .map_err(|cause| metadata.error(cause))?;
                program.population()
            }
            PartitionInvocationCaptureKind::Histogram(geometry) => {
                let program = PreparedCaptureHistogram::from_geometry(geometry)
                    .map_err(|cause| metadata.error(cause))?;
                program
                    .validate_workspace_source(value, context)
                    .map_err(|cause| metadata.error(cause))?;
                metadata.reserve(roots, 1)?;
                roots.push(value.clone());
                program
                    .trace(value, context, roots)
                    .map_err(|cause| metadata.error(cause))?;
                program.population()
            }
        }
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        Ok((dtype, population))
    }
    fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<Result<Self>>(),
            size_of::<PartitionInvocationCaptureGeometry<'_>>(),
            size_of::<PartitionCaptureNativeEstimate>() * 2,
            size_of::<CaptureUsage>(),
            size_of::<(
                &PartitionCaptureReceiptPlan,
                usize,
                usize,
                &WorkspaceContext,
            )>(),
            size_of::<(
                PartitionInvocationCaptureGeometry<'_>,
                usize,
                &WorkspaceContext,
            )>(),
            size_of::<(
                &Self,
                &WorkspaceTensor,
                &WorkspaceContext,
                &mut Vec<WorkspaceTensor>,
            )>(),
            size_of::<PartitionCaptureFragmentSource<'_>>(),
            size_of::<TensorDtype>(),
            size_of::<Result<(WorkspaceFloatingType, CaptureNativePopulation)>>(),
            size_of::<WorkspaceFloatingType>(),
            size_of::<CaptureNativePopulation>(),
            size_of::<CaptureTensorSelection>(),
            size_of::<PreparedCaptureSummary>(),
            size_of::<PreparedCaptureHistogram<'_>>(),
            PartitionInvocationCaptureGeometry::control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

#[cfg(test)]
mod tests;

/// Same raw selection or reducer as the ordinary projected invocation. Actual
/// activation shape, scalar and representation are validated by that worker.
pub(in crate::composition::mlx::session) fn trace_geometry(
    source: PartitionInvocationCaptureGeometry<'_>,
    producer: usize,
    value: &WorkspaceTensor,
    context: &WorkspaceContext,
    roots: &mut Vec<WorkspaceTensor>,
) -> Result<CaptureNativePopulation> {
    let parts = [
        size_of::<(
            PartitionInvocationCaptureGeometry<'_>,
            usize,
            &WorkspaceTensor,
            &WorkspaceContext,
            &mut Vec<WorkspaceTensor>,
        )>(),
        size_of::<PartitionInvocationEquation<'_>>(),
        size_of::<Result<PartitionInvocationEquation<'_>>>(),
        size_of::<Result<(WorkspaceFloatingType, CaptureNativePopulation)>>(),
        size_of::<Result<CaptureNativePopulation>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let equation = PartitionInvocationEquation::from_geometry(source, producer, context)?;
    let (_, population) = equation.trace(value, context, roots)?;
    Ok(population)
}
