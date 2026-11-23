//! Native coordinate materialization only; portable runtime owns edit semantics.
use super::*;

pub(super) fn partition_locations(
    source: &PartitionRoutedUnitCaptureSource<'_, MlxTensor>,
    geometry: RoutedUnitGeometry,
    stream: &Stream,
) -> Result<RoutedUnitLocations, Error> {
    let local = RoutedUnitGeometry {
        units_per_expert: source.unit_coordinates.local_count() as u64,
        routes_per_token: if source.origins.is_some() {
            1
        } else {
            geometry.routes_per_token
        },
        ..geometry
    };
    routed_unit_locations(
        &super::super::bounded_capture::routed::NativeRoutedReadout(stream),
        &source.source,
        local,
        source.origins,
    )
}
