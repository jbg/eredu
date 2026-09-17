//! Original sparse coordinates layered on the same five completed native loans.
use super::*;
use eredu_core::capture::{
    PartitionRoutedUnitCaptureRequest, PartitionRoutedUnitCaptureSource,
    RoutedUnitCaptureOwnership, RoutedUnitOrigins,
};
use eredu_core::component::ComponentCoordinateMap;
use eredu_runtime::working_memory::CapturePartitionRoutedTransfer;

/// Actual local placement and global equation shared by cold and completed
/// layout validation. Selection slices and native work authority are separate.
#[derive(Clone, Copy)]
pub(crate) struct PartitionRoutedCaptureLayout<'a> {
    pub(crate) geometry: RoutedUnitGeometry,
    pub(crate) source_tokens: u64,
    pub(crate) ownership: &'a RoutedUnitCaptureOwnership,
    pub(crate) origins: Option<RoutedUnitOrigins<'a>>,
    pub(crate) units: &'a ComponentCoordinateMap,
}

/// Physical receive-order rows remain separate from original peer/token/slot
/// coordinates. These borrowed tags add no native array or completion authority.
pub(crate) struct CompletedPartitionRoutedCaptureSource<'a> {
    source: CompletedRoutedCaptureSource<'a>,
    origins: Option<RoutedUnitOrigins<'a>>,
    units: &'a ComponentCoordinateMap,
    ownership: &'a RoutedUnitCaptureOwnership,
    geometry: RoutedUnitGeometry,
    source_tokens: u64,
    logical_source_tokens: u64,
}
impl<'a> CompletedPartitionRoutedCaptureSource<'a> {
    pub(crate) fn new(
        source: &PartitionRoutedUnitCaptureSource<'a, Array>,
        inputs: [EvaluatedArray<'a>; 5],
        request: PartitionRoutedUnitCaptureRequest<'a>,
        invocation_source_tokens: u64,
    ) -> Result<Self, Error> {
        Self::bind_layout(&source.source, inputs, PartitionRoutedCaptureLayout {
            geometry: request.geometry, source_tokens: invocation_source_tokens,
            ownership: request.ownership, origins: source.origins, units: source.unit_coordinates,
        }, request.source_tokens)
    }
    /// Bind receipt-lifetime ownership independently of the destination slice.
    /// The actual five arrays and original host tags remain borrowed in place.
    pub(crate) fn bind_layout(
        source: &RoutedUnitCaptureSource<'a, Array>, inputs: [EvaluatedArray<'a>; 5],
        layout: PartitionRoutedCaptureLayout<'a>, logical_source_tokens: u64,
    ) -> Result<Self, Error> {
        let shapes = [source.values.shape(), source.token_indices.shape(),
            source.selection_indices.shape(), source.coefficients.shape(), source.source_groups.shape()];
        let dtypes = [source.values.dtype(), source.token_indices.dtype(),
            source.selection_indices.dtype(), source.coefficients.dtype(), source.source_groups.dtype()];
        let geometry = Self::validate_source_layouts(shapes, Some(dtypes), source.token_offset, layout)?;
        let native_rows = u64::try_from(shapes[4][0]).map_err(|_| Error::ShapeMismatch)?;
        let physical = Self::physical_bank(layout)?;
        let source = CompletedRoutedCaptureSource::bind(source, inputs, physical, native_rows, geometry)?;
        Ok(Self { source, origins: layout.origins, units: layout.units, ownership: layout.ownership,
            geometry: layout.geometry, source_tokens: layout.source_tokens, logical_source_tokens })
    }

    /// The cold observer and completed reader use identical physical layout and
    /// borrowed placement checks. None means preliminary logical geometry only.
    pub(crate) fn validate_layouts(
        shapes: [&[i32]; 5],
        dtypes: Option<[Dtype; 5]>,
        token_offset: u64,
        origins: Option<RoutedUnitOrigins<'_>>,
        units: &ComponentCoordinateMap,
        request: PartitionRoutedUnitCaptureRequest<'_>,
        invocation_source_tokens: u64,
    ) -> Result<(usize, u64), Error> {
        Self::validate_source_layouts(
            shapes,
            dtypes,
            token_offset,
            PartitionRoutedCaptureLayout {
                geometry: request.geometry,
                source_tokens: invocation_source_tokens,
                ownership: request.ownership,
                origins,
                units,
            },
        )
    }
    /// No selection slice is needed to authenticate the native source layout.
    pub(crate) fn validate_source_layouts(
        shapes: [&[i32]; 5],
        dtypes: Option<[Dtype; 5]>,
        token_offset: u64,
        layout: PartitionRoutedCaptureLayout<'_>,
    ) -> Result<(usize, u64), Error> {
        let origins = layout.origins;
        let units = layout.units;
        layout
            .ownership
            .validate_geometry(layout.geometry)
            .map_err(|_| Error::ClaimMismatch)?;
        if units != layout.ownership.coordinates.units() || shapes[4].len() != 2 {
            return Err(Error::ClaimMismatch);
        }
        let physical = Self::physical_bank(layout)?;
        let native_rows = u64::try_from(shapes[4][0]).map_err(|_| Error::ShapeMismatch)?;
        let maximum = layout
            .ownership
            .maximum_source_rows_checked(layout.source_tokens, layout.geometry.routes_per_token)
            .map_err(|_| Error::GeometryOverflow)?;
        if native_rows > maximum {
            return Err(Error::ShapeMismatch);
        }
        match (layout.ownership.source_peer, origins) {
            (None, None) if native_rows == layout.source_tokens => {}
            (Some(_), Some(origins))
                if origins.row_count() as u64 == native_rows
                    && origins.peer_count() as u64 == layout.ownership.source_peers
                    && origins.routes_per_token() as u64 == layout.geometry.routes_per_token => {}
            _ => return Err(Error::SourceChanged),
        }
        let geometry = CompletedRoutedCaptureSource::validate_physical_geometry(
            shapes,
            token_offset,
            physical,
            native_rows,
        )?;
        if let Some(dtypes) = dtypes {
            CompletedRoutedCaptureSource::validate_dtypes(dtypes)?;
        }
        Ok(geometry)
    }
    fn physical_bank(layout: PartitionRoutedCaptureLayout<'_>) -> Result<RoutedUnitGeometry, Error> {
        Ok(RoutedUnitGeometry { experts: layout.geometry.experts,
            units_per_expert: u64::try_from(layout.units.local_count()).map_err(|_| Error::GeometryOverflow)?,
            routes_per_token: if layout.ownership.source_peer.is_some() { 1 }
                else { layout.geometry.routes_per_token } })
    }
    fn route(&self, index: usize) -> Result<(Option<u64>, Route), Error> {
        let mut route = self.source.route(index)?;
        let peer = if let Some(origins) = self.origins {
            let origin = origins
                .resolve(usize::try_from(route.token).map_err(|_| Error::GeometryOverflow)?)
                .ok_or(Error::ShapeMismatch)?;
            route.token = u64::try_from(origin.token).map_err(|_| Error::GeometryOverflow)?;
            route.slot = u64::try_from(origin.slot).map_err(|_| Error::GeometryOverflow)?;
            origin
                .source_peer
                .map(u64::try_from)
                .transpose()
                .map_err(|_| Error::GeometryOverflow)?
        } else {
            None
        };
        if route.token >= self.source_tokens
            || route.slot >= self.geometry.routes_per_token
            || usize::try_from(route.expert)
                .ok()
                .and_then(|expert| self.ownership.coordinates.experts().global_to_local(expert))
                .is_none()
        {
            return Err(Error::ShapeMismatch);
        }
        Ok((peer, route))
    }

    /// Decode every original route before filtering its publication peer. Values
    /// go directly into the paid sparse destination in global selected-unit order.
    pub(crate) fn copy_routed(
        self,
        mut writer: CapturePartitionRoutedTransfer<'_, '_, '_, Identity>,
    ) -> Result<(), Error> {
        writer.validate()?;
        let request = writer.request();
        if request.geometry != self.geometry
            || request.source_tokens != self.logical_source_tokens
            || writer.invocation_source_tokens() != self.source_tokens
            || request.ownership != self.ownership
            || writer.native_rows() != self.source.source_tokens
        {
            return Err(Error::ClaimMismatch);
        }
        let starts: [u64; 3] = request
            .slice
            .starts
            .as_slice()
            .try_into()
            .map_err(|_| Error::ClaimMismatch)?;
        let ends: [u64; 3] = request
            .slice
            .ends
            .as_slice()
            .try_into()
            .map_err(|_| Error::ClaimMismatch)?;
        let strides: [u64; 3] = request
            .slice
            .strides
            .as_slice()
            .try_into()
            .map_err(|_| Error::ClaimMismatch)?;
        let counts: [u64; 3] = request
            .slice
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::ClaimMismatch)?;
        if strides.contains(&0) {
            return Err(Error::ClaimMismatch);
        }
        for index in 0..self.source.rows {
            let (peer, mut route) = self.route(index)?;
            route.token = writer
                .logical_token(route.token)
                .ok_or(Error::ShapeMismatch)?;
            if peer != self.ownership.source_peer
                || counts[2] == 0
                || route.token < starts[0]
                || route.token >= ends[0]
                || !(route.token - starts[0]).is_multiple_of(strides[0])
                || route.slot < starts[1]
                || route.slot >= ends[1]
                || !(route.slot - starts[1]).is_multiple_of(strides[1])
            {
                continue;
            }
            writer
                .begin_row(
                    peer,
                    route.token,
                    route.slot,
                    route.expert,
                    route.coefficient,
                )
                .map_err(CaptureRunHostError::from)?;
            let base = index
                .checked_mul(self.units.local_count())
                .ok_or(Error::GeometryOverflow)?;
            for unit in 0..counts[2] {
                let global = unit
                    .checked_mul(strides[2])
                    .and_then(|n| starts[2].checked_add(n))
                    .ok_or(Error::GeometryOverflow)?;
                let local = self
                    .units
                    .global_to_local(usize::try_from(global).map_err(|_| Error::GeometryOverflow)?)
                    .ok_or(Error::ClaimMismatch)?;
                let value = self
                    .source
                    .floating(0, base.checked_add(local).ok_or(Error::GeometryOverflow)?)?;
                writer.push_f32(value).map_err(CaptureRunHostError::from)?;
            }
            writer.finish_row().map_err(CaptureRunHostError::from)?;
        }
        let end = self
            .source
            .token_offset
            .checked_add(self.source.chunk_tokens)
            .ok_or(Error::GeometryOverflow)?;
        // An idle EP source has a real empty acknowledgment and no range.
        if end != self.source.token_offset {
            writer
                .source_chunk(self.source.token_offset, end)
                .map_err(CaptureRunHostError::from)?;
        }
        writer.finish().map_err(CaptureRunHostError::from)?;
        Ok(())
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            CompletedRoutedCaptureSource::control_bytes()?,
            size_of::<PartitionRoutedCaptureLayout<'_>>() * 2,
            size_of::<Self>() * 2,
            size_of::<Result<Self, Error>>(),
            size_of::<PartitionRoutedUnitCaptureRequest<'_>>() * 3,
            size_of::<PartitionRoutedUnitCaptureSource<'_, Array>>(),
            size_of::<Option<RoutedUnitOrigins<'_>>>() * 2,
            size_of::<(&ComponentCoordinateMap, &RoutedUnitCaptureOwnership)>(),
            size_of::<(
                [&[i32]; 5],
                Option<[Dtype; 5]>,
                u64,
                Option<RoutedUnitOrigins<'_>>,
                &ComponentCoordinateMap,
                PartitionRoutedUnitCaptureRequest<'_>,
            )>(),
            size_of::<(Option<u64>, Route)>(),
            size_of::<Result<(Option<u64>, Route), Error>>(),
            size_of::<eredu_core::capture::RoutedUnitOrigin>(),
            size_of::<CapturePartitionRoutedTransfer<'_, '_, '_, Identity>>() * 2,
            size_of::<[u64; 3]>() * 4,
            size_of::<usize>() * 4,
            size_of::<u64>() * 4,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::capture::ResolvedCaptureSlice;
    use eredu_core::component::RoutedComponentCoordinateMap;

    #[test]
    fn completed_partition_routes_preserve_receive_origins_and_permuted_units() {
        let values = Array::from_slice(
            &[
                half::f16::from_f32(0.5),
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.5),
                half::f16::from_f32(2.0),
                half::f16::from_f32(2.5),
                half::f16::from_f32(3.0),
            ],
            &[3, 2],
        );
        let tokens = Array::from_slice(&[2u32, 0, 1], &[3]);
        let selected = Array::from_slice(&[2i32, 0, 1], &[3]);
        let coefficients = Array::from_slice(&[0.25f32, 0.5, 0.75], &[3, 1]);
        let groups = Array::from_slice(&[0u32, 1, 0], &[3, 1]);
        let units = ComponentCoordinateMap::indices(4, vec![3, 1]).unwrap();
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::indices(5, vec![3, 1]).unwrap(),
                units.clone(),
            ),
            source_peer: Some(2),
            source_peers: 3,
        };
        let slice = ResolvedCaptureSlice {
            starts: vec![0, 0, 1],
            ends: vec![2, 2, 4],
            strides: vec![1, 1, 2],
            shape: vec![2, 2, 2],
        };
        let request = PartitionRoutedUnitCaptureRequest {
            geometry: RoutedUnitGeometry {
                experts: 5,
                units_per_expert: 4,
                routes_per_token: 2,
            },
            source_tokens: 2,
            ownership: &ownership,
            slice: &slice,
        };
        let source = PartitionRoutedUnitCaptureSource {
            source: RoutedUnitCaptureSource {
                values: &values,
                token_indices: &tokens,
                selection_indices: &selected,
                coefficients: &coefficients,
                source_groups: &groups,
                token_offset: 0,
                global_groups: Some(&[3, 1]),
            },
            origins: Some(RoutedUnitOrigins::new(&[1, 0, 2], &[1, 3, 0], 2).unwrap()),
            unit_coordinates: &units,
        };
        let loans = || {
            [
                values.evaluated().unwrap(),
                tokens.evaluated().unwrap(),
                selected.evaluated().unwrap(),
                coefficients.evaluated().unwrap(),
                groups.evaluated().unwrap(),
            ]
        };
        let reader =
            CompletedPartitionRoutedCaptureSource::new(&source, loans(), request, 2).unwrap();
        assert_eq!(
            reader.route(0).unwrap(),
            (
                Some(2),
                Route {
                    token: 0,
                    slot: 0,
                    expert: 3,
                    coefficient: 0.75
                }
            )
        );
        assert_eq!(
            reader.route(1).unwrap(),
            (
                Some(0),
                Route {
                    token: 0,
                    slot: 1,
                    expert: 3,
                    coefficient: 0.25
                }
            )
        );
        assert_eq!(
            reader.route(2).unwrap(),
            (
                Some(2),
                Route {
                    token: 1,
                    slot: 1,
                    expert: 1,
                    coefficient: 0.5
                }
            )
        );
        // Global selected unit order [1,3] reverses this physical [3,1] row.
        assert_eq!(
            [
                reader
                    .source
                    .floating(0, reader.units.global_to_local(1).unwrap())
                    .unwrap(),
                reader
                    .source
                    .floating(0, reader.units.global_to_local(3).unwrap())
                    .unwrap()
            ],
            [1.0, 0.5]
        );
        let wrong = ComponentCoordinateMap::indices(4, vec![1, 3]).unwrap();
        let substituted = PartitionRoutedUnitCaptureSource {
            source: source.source,
            origins: source.origins,
            unit_coordinates: &wrong,
        };
        assert!(matches!(
            CompletedPartitionRoutedCaptureSource::new(&substituted, loans(), request, 2),
            Err(Error::ClaimMismatch)
        ));
    }

    #[test]
    fn partition_native_layout_accepts_real_idle_sources_but_rejects_fabricated_origins() {
        let units = ComponentCoordinateMap::range(4, 0..0).unwrap();
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(5, 0..0).unwrap(),
                units.clone(),
            ),
            source_peer: Some(1),
            source_peers: 2,
        };
        let slice = ResolvedCaptureSlice {
            starts: vec![0, 0, 0],
            ends: vec![2, 2, 0],
            strides: vec![1, 1, 1],
            shape: vec![2, 2, 0],
        };
        let request = PartitionRoutedUnitCaptureRequest {
            geometry: RoutedUnitGeometry {
                experts: 5,
                units_per_expert: 4,
                routes_per_token: 2,
            },
            source_tokens: 2,
            ownership: &ownership,
            slice: &slice,
        };
        let shapes: [&[i32]; 5] = [&[0, 0], &[0], &[0], &[0, 1], &[0, 1]];
        let origins = RoutedUnitOrigins::new(&[0, 0], &[], 2).unwrap();
        let dtypes = [
            Dtype::Float16,
            Dtype::Uint32,
            Dtype::Int32,
            Dtype::Float32,
            Dtype::Uint32,
        ];
        assert_eq!(
            CompletedPartitionRoutedCaptureSource::validate_layouts(
                shapes,
                Some(dtypes),
                0,
                Some(origins),
                &units,
                request,
                2
            )
            .unwrap(),
            (0, 0)
        );
        assert!(CompletedPartitionRoutedCaptureSource::validate_layouts(
            shapes,
            Some(dtypes),
            0,
            None,
            &units,
            request,
            2
        )
        .is_err());
        let wrong = RoutedUnitOrigins::new(&[0, 0], &[], 1).unwrap();
        assert!(CompletedPartitionRoutedCaptureSource::validate_layouts(
            shapes,
            Some(dtypes),
            0,
            Some(wrong),
            &units,
            request,
            2
        )
        .is_err());
        // Empty local units do not make a serial global bank valid.
        assert!(CompletedRoutedCaptureSource::validate_geometry(
            shapes,
            0,
            RoutedUnitGeometry {
                experts: 5,
                units_per_expert: 0,
                routes_per_token: 1
            },
            0
        )
        .is_err());
    }
}
