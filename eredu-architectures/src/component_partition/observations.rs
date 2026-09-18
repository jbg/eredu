//! Original observation registration with prospective backing destinations.
use super::*;

pub(super) fn insert(
    observations: &mut SourceMap<String, PartitionedObservation>,
    path: &str,
    placement: PartitionedObservation,
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &str,
        PartitionedObservation,
    )>()?;
    match observations.get(path) {
        Some(previous) if previous == &placement => Ok(()),
        Some(_) => Err(allocation.duplicate(path)),
        None => {
            allocation.insert(observations, allocation.text(path)?, placement)?;
            Ok(())
        }
    }
}

pub(super) fn replicated(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    path: &str,
    axis: &str,
    local: bool,
    site: ObservationHookSite,
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &ArchitectureDescriptor,
        &str,
        &str,
        bool,
        ObservationHookSite,
        PartitionedObservation,
        String,
    )>()?;
    let placement = self::placement(descriptor, path, axis, local, site, allocation)?;
    insert(
        observations,
        path,
        allocation.observation(&placement)?,
        allocation,
    )?;
    let effective = allocation.format(format_args!("{path}.effective"))?;
    if descriptor
        .observations
        .points
        .iter()
        .any(|point| point.path == effective)
    {
        insert(observations, &effective, placement, allocation)?;
    }
    Ok(())
}

// Project exactly one declared point. Callers choose whether an explicit
// original/effective convention or exact node bindings establish other points.
pub(super) fn placement(
    descriptor: &ArchitectureDescriptor,
    path: &str,
    axis: &str,
    local: bool,
    site: ObservationHookSite,
    allocation: Destination<'_>,
) -> Result<PartitionedObservation, ComponentPartitionError> {
    use eredu_core::SymbolicDimension;
    allocation.controls::<(
        &ArchitectureDescriptor,
        &str,
        &str,
        bool,
        ObservationHookSite,
        PartitionedObservation,
        Option<ComponentCoordinateMap>,
        &eredu_core::TensorAxis,
    )>()?;
    let point = descriptor
        .observations
        .points
        .iter()
        .find(|point| point.path == path)
        .ok_or_else(|| allocation.capture_missing(path))?;
    let axes = point.axes.as_ref().ok_or_else(|| {
        allocation.capture_unsupported(format_args!("replicated observation has no axes"))
    })?;
    let mut matching = axes.iter().filter(|candidate| candidate.name == axis);
    let Some(eredu_core::TensorAxis {
        dimension: SymbolicDimension::Known(width),
        ..
    }) = matching.next()
    else {
        return Err(
            allocation.capture_invalid(format_args!("replicated observation width is unresolved"))
        );
    };
    if matching.next().is_some() {
        return Err(
            allocation.capture_invalid(format_args!("replicated observation axis is repeated"))
        );
    }
    let coordinates = local
        .then(|| ComponentCoordinateMap::range(*width, 0..*width))
        .transpose()?;
    Ok(PartitionedObservation {
        axis: allocation.text(axis)?,
        coordinates,
        exports: local,
        site,
        combination: PartitionCaptureCombination::Disjoint,
    })
}
