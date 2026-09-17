//! Cold facts from the retained sparse architecture source and actual provider.
use super::*;
use crate::backend::array_copy::{
    CompletedPartitionRoutedCaptureSource, PartitionRoutedCaptureLayout,
};
use eredu_architectures::component_partition::{
    ComponentPartitionLayouts, RoutedPartitionCaptureRank, RoutedPartitionCaptureSource,
};
use std::mem::{size_of, size_of_val};

pub(in super::super) fn local<'a>(
    source: &'a SharedCapturePlan,
    placement: (&'a ComponentPartitionLayouts, usize),
    index: usize,
    context: &WorkspaceContext,
) -> Result<Option<RoutedPartitionCaptureRank<'a>>> {
    let metadata = routed_metadata(context)?;
    context.charge_metadata(
        RoutedPartitionCaptureSource::control_bytes()
            .and_then(|n| {
                n.checked_add(size_of::<(
                    &SharedCapturePlan,
                    (&ComponentPartitionLayouts, usize),
                    usize,
                    &WorkspaceContext,
                    Option<RoutedPartitionCaptureRank<'_>>,
                )>())
            })
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let selected = source
        .admission()
        .plan()
        .selections
        .get(index)
        .ok_or_else(|| metadata.coordinate())?;
    let declaration = placement
        .0
        .routed_capture_source(&selected.path)
        .map_err(|e| metadata.error(e))?;
    let eredu_core::ObservationValueType::RoutedUnits {
        geometry, routing, ..
    } = &source.admission().points()[index].value_type
    else {
        return Err(metadata.coordinate());
    };
    if *geometry != declaration.geometry() || routing != declaration.routing() {
        return Err(metadata.coordinate());
    }
    Ok(declaration.rank(placement.1))
}
pub(super) fn begin(
    source: &SharedCapturePlan,
    placement: (&ComponentPartitionLayouts, usize),
    index: usize,
    input: &RoutedUnitInvocation<'_, WorkspaceTensor>,
    source_tokens: u64,
    context: &WorkspaceContext,
) -> Result<u64> {
    let metadata = routed_metadata(context)?;
    let local = local(source, placement, index, context)?.ok_or_else(|| metadata.coordinate())?;
    let geometry = match source.admission().points()[index].value_type {
        eredu_core::ObservationValueType::RoutedUnits { geometry, .. } => geometry,
        _ => return Err(metadata.coordinate()),
    };
    let shape = input.input.shape();
    context.validate_values([input.input])?;
    let rows = PartitionRoutedUnitCaptureLayout::input_rows(shape)
        .map_err(|cause| metadata.error(cause))?;
    let layout = PartitionRoutedUnitCaptureLayout {
        geometry,
        source_tokens,
        ownership: local.ownership,
    };
    layout
        .validate_input_invocation(
            rows,
            input.unit_coordinates,
            input.origins.map(|origin| origin.capture_coordinates()),
        )
        .map_err(|e| metadata.error(e))?;
    context.charge_metadata(size_of::<(
        PartitionRoutedUnitCaptureLayout<'_>,
        &RoutedUnitInvocation<'_, WorkspaceTensor>,
        &[i32],
        std::slice::Iter<'_, i32>,
        u64,
        std::result::Result<u64, eredu_core::capture::RoutedUnitValidationError>,
        Result<u64>,
    )>())?;
    Ok(rows)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn validate(
    source: &SharedCapturePlan,
    placement: (&ComponentPartitionLayouts, usize),
    index: usize,
    shapes: [&[i32]; 5],
    dtypes: Option<[safemlx::Dtype; 5]>,
    offset: u64,
    batch: &RoutedUnitBatch<'_, WorkspaceTensor>,
    geometry: RoutedUnitGeometry,
    source_tokens: u64,
    actual_rows: Option<u64>,
    context: &WorkspaceContext,
) -> Result<(usize, u64)> {
    let metadata = routed_metadata(context)?;
    let local = local(source, placement, index, context)?.ok_or_else(|| metadata.coordinate())?;
    let units = batch
        .unit_coordinates
        .ok_or_else(|| metadata.coordinate())?;
    context.charge_metadata(
        CompletedPartitionRoutedCaptureSource::control_bytes()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let result = CompletedPartitionRoutedCaptureSource::validate_source_layouts(
        shapes,
        dtypes,
        offset,
        PartitionRoutedCaptureLayout {
            geometry,
            source_tokens,
            ownership: local.ownership,
            origins: batch.origins.map(|origin| origin.capture_coordinates()),
            units,
        },
    )
    .map_err(|e| metadata.error(e))?;
    if shapes[4].first().and_then(|n| u64::try_from(*n).ok()) != actual_rows {
        return Err(metadata.coordinate());
    }
    Ok(result)
}
/// Count the exact ordinary coordinate projection; no projection is constructed.
pub(super) fn fragments(
    source: &SharedCapturePlan,
    placement: (&ComponentPartitionLayouts, usize),
    index: usize,
    geometry: &CaptureRoutedUnitsGeometry<'_>,
    context: &WorkspaceContext,
) -> Result<usize> {
    let metadata = routed_metadata(context)?;
    let local = local(source, placement, index, context)?.ok_or_else(|| metadata.coordinate())?;
    if !local.produces {
        return Ok(0);
    }
    let parts = [
        size_of::<ResolvedCaptureSlice>(),
        size_of::<[u64; 3]>(),
        size_of::<CaptureCoordinateProjectionPlan<'_>>(),
        size_of::<[Vec<u64>; 4]>(),
        size_of::<(
            &SharedCapturePlan,
            (&ComponentPartitionLayouts, usize),
            usize,
            &CaptureRoutedUnitsGeometry<'_>,
            &WorkspaceContext,
        )>(),
        size_of::<Result<usize>>(),
        size_of::<std::iter::Zip<std::slice::IterMut<'_, u64>, std::slice::Iter<'_, usize>>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let mut global = [0u64; 3];
    for (out, &n) in global.iter_mut().zip(geometry.source_shape()) {
        *out = u64::try_from(n).map_err(|e| metadata.error(e))?;
    }
    let mut slice = ResolvedCaptureSlice {
        starts: context.metadata_vec(3)?,
        ends: context.metadata_vec(3)?,
        strides: context.metadata_vec(3)?,
        shape: context.metadata_vec(3)?,
    };
    slice.starts.extend_from_slice(geometry.starts());
    slice.ends.extend_from_slice(geometry.ends());
    slice.strides.extend_from_slice(geometry.strides());
    for &n in geometry.shape() {
        slice
            .shape
            .push(u64::try_from(n).map_err(|e| metadata.error(e))?);
    }
    let plan = CaptureCoordinateProjectionPlan::prepare(
        &global,
        &slice,
        2,
        local.ownership.coordinates.units(),
        local.ownership.coordinates.units().local_count().max(1),
    )
    .map_err(|e| metadata.error(e))?;
    context.charge_metadata(plan.control_bytes())?;
    Ok(plan.fragments())
}
