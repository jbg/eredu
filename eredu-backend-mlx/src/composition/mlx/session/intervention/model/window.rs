//! Logical-to-physical slice and payload projection for one actual model hook.
use super::*;
use eredu_runtime::intervention::{
    PartitionInterventionProjectionCost, PreparedWindowInterventionPayload,
    validate_partition_column_region,
};
pub(super) struct Geometry {
    pub slice: ResolvedCaptureSlice,
    pub payload: Option<PreparedWindowInterventionPayload>,
    pub overlap: bool,
    pub projection: [CaptureUsage; 2],
}
fn vector(rank: usize, context: &WorkspaceContext) -> Result<Vec<u64>, eredu_nn::Error> {
    let mut row = context.metadata_vec(rank)?;
    row.resize(rank, 0);
    Ok(row)
}
pub(super) fn slice(rank: usize, context: &WorkspaceContext) -> Result<ResolvedCaptureSlice, eredu_nn::Error> {
    Ok(ResolvedCaptureSlice {
        starts: vector(rank, context)?,
        ends: vector(rank, context)?,
        strides: vector(rank, context)?,
        shape: vector(rank, context)?,
    })
}
pub(super) fn prepare(
    plan: &AdmittedInterventionPlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    physical: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
    span: Option<eredu_runtime::intervention::InterventionPrefillWindow>,
    shape: &[u64],
    dtype: InterventionDtype,
    metadata: CaptureUsage,
    context: &WorkspaceContext,
    ledger: &mut CaptureLedger,
) -> Result<Geometry, eredu_nn::Error> {
    let Some(window) = window else {
        let mut local = slice(shape.len(), context)?;
        match physical {
            Some(physical) => plan.resolve_prepared_invocation_at(
                index, phase, prediction, physical, shape, dtype, &mut local,
            ),
            None => plan.resolve_prepared_at(index, phase, prediction, shape, dtype, &mut local),
        }.map_err(|cause| context.metadata_source(cause))?;
        return Ok(Geometry {
            slice: local,
            payload: None,
            overlap: true,
            projection: [CaptureUsage::default(); 2],
        });
    };
    let physical = physical.ok_or_else(|| context.metadata_source(CaptureProtocolError::Invocation))?;
    let action = &plan.plan().operations[index].action;
    let point = &plan.points()[index];
    let mut global = vector(shape.len(), context)?;
    let mut selected = slice(shape.len(), context)?;
    let mut local = slice(shape.len(), context)?;
    let mut destination = slice(shape.len(), context)?;
    let (axis, overlap) = if let Some(span) = span {
        if phase != CapturePhase::Prefill || prediction != 0
            || span.physical() != physical || span.window() != window {
            return Err(context.metadata_source(CaptureProtocolError::PrefillAttribution));
        }
        span.resolve_projection_into(plan, index, shape, dtype, &mut global,
            &mut selected, &mut local, &mut destination)
            .map_err(|cause| context.metadata_source(cause))?
    } else {
        let (logical, axis) = window
            .source_axes_into(physical, Some(&point.axes), shape, &mut global)
            .map_err(|cause| context.metadata_source(cause))?;
        match plan.invocation_bounds() {
            Some(_) => plan.resolve_prepared_invocation_at(index, phase, prediction,
                logical, &global, dtype, &mut selected),
            None => plan.resolve_prepared_at(index, phase, prediction, &global, dtype, &mut selected),
        }.map_err(|cause| context.metadata_source(cause))?;
        let end = window.start.checked_add(physical.sequence).ok_or(WorkspaceMetadataError::Overflow)?;
        let overlap = CaptureSlicePartition::contiguous_fragment_into(
            &global, &selected, axis, window.start..end, &mut local, &mut destination,
        ).map_err(|cause| context.metadata_source(cause))?;
        (axis, overlap)
    };
    let column = validate_partition_column_region(action, &global, &selected)
        .map_err(|cause| context.metadata_source(cause))?;
    // Compact component edits still require their existing component producer.
    // A Context axis cannot be silently interpreted as component coordinates.
    if column && (axis + 1 == shape.len()
        || (span.is_some() && point.axes.last().is_some_and(|axis| axis.dimension == eredu_core::SymbolicDimension::Context))) {
        return Err(context.metadata_source(CaptureProtocolError::PrefillAttribution));
    }
    let mut cost = PartitionInterventionProjectionCost::new(plan, shape.len())
        .map_err(|cause| context.metadata_source(cause))?;
    if overlap {
        NativeInterventionEstimator
            .validate_geometry(shape, &local)
            .map_err(|cause| context.metadata_source(cause))?;
        cost.include(action, &local.shape)
            .map_err(|cause| context.metadata_source(cause))?;
    }
    let host = cost.usage();
    reserve_envelope(ledger, host).map_err(|cause| context.metadata_source(cause))?;
    let payload = if overlap {
        Some(
            PreparedWindowInterventionPayload::prepare(
                action,
                &destination,
                context
                    .metadata_funding()
                    .ok_or(WorkspaceMetadataError::Unqualified)?,
            )
            .map_err(|cause| context.metadata_source(cause))?,
        )
    } else {
        None
    };
    Ok(Geometry {
        slice: local,
        payload,
        overlap,
        projection: [metadata, host],
    })
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Geometry>(),
        size_of::<Result<Geometry, eredu_nn::Error>>(),
        size_of::<[ResolvedCaptureSlice; 3]>(),
        size_of::<Vec<u64>>(),
        size_of::<Result<Vec<u64>, eredu_nn::Error>>(),
        size_of::<(usize, &WorkspaceContext)>(),
        size_of::<Result<ResolvedCaptureSlice, eredu_nn::Error>>(),
        size_of::<
            Result<
                PreparedWindowInterventionPayload,
                eredu_runtime::intervention::PreparedWindowInterventionPayloadError,
            >,
        >(),
        size_of::<Result<(), eredu_core::intervention::InterventionGeometryError>>(),
        size_of::<Result<(), CaptureError>>(),
        size_of::<[CaptureUsage; 2]>(),
        size_of::<Option<PreparedWindowInterventionPayload>>(),
        size_of::<(
            &AdmittedInterventionPlan,
            usize,
            CapturePhase,
            u64,
            Option<CaptureInvocationShape>,
            Option<CaptureInvocationWindow>,
            Option<eredu_runtime::intervention::InterventionPrefillWindow>,
            &[u64],
            InterventionDtype,
            CaptureUsage,
            &WorkspaceContext,
            &mut CaptureLedger,
        )>(),
        size_of::<(&[u64], usize, std::ops::Range<u64>)>(),
        size_of::<[CaptureInvocationShape; 2]>(),
        size_of::<(bool, usize, u64)>(),
        PartitionInterventionProjectionCost::control_bytes()?,
        CaptureInvocationWindow::source_axes_control_bytes()?,
        CaptureSlicePartition::contiguous_projection_control_bytes()?,
        eredu_runtime::intervention::InterventionPrefillWindow::projection_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
