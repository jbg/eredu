use super::*;

/// Actual borrowed provider inputs for a partitioned sparse collector. All
/// coordinates come from the running provider, separately from expected ownership.
pub struct PartitionRoutedUnitCaptureSource<'a, T> {
    /// Actual sorted values and original pre-compaction group/route arrays.
    pub source: RoutedUnitCaptureSource<'a, T>,
    /// Original source peer/token/slot tags, when input rows were exchanged.
    pub origins: Option<RoutedUnitOrigins<'a>>,
    /// Actual local scalar-column order supplied by the prepared provider.
    pub unit_coordinates: &'a crate::component::ComponentCoordinateMap,
}

/// Cold recipe for one original-coordinate sparse fragment. It grants no native
/// work authority: the caller must reserve the collector's bound before use.
#[derive(Debug, Clone, Copy)]
pub struct PartitionRoutedUnitCaptureRequest<'a> {
    /// Global bank equation and original route width.
    pub geometry: RoutedUnitGeometry,
    /// Original input token rows per source peer, before exchange expands routes.
    pub source_tokens: u64,
    /// Retained expected expert/unit placement and authoritative publication peer.
    pub ownership: &'a RoutedUnitCaptureOwnership,
    /// Original global token/slot/unit coordinates, not local receive order.
    pub slice: &'a ResolvedCaptureSlice,
}

impl PartitionRoutedUnitCaptureRequest<'_> {
    /// Checks the fragment against retained ownership without tensor work.
    pub fn validate(&self) -> Result<(), CaptureError> {
        self.ownership.validate(self.geometry)?;
        self.geometry.validate_slice(self.slice)?;
        if self.slice.ends[0] > self.source_tokens {
            return Err(CaptureError::Invalid(
                "routed fragment exceeds original input".into(),
            ));
        }
        let units = self.ownership.coordinates.units();
        let count = self.slice.shape[2];
        if count > units.local_count() as u64 {
            return Err(CaptureError::Invalid(
                "routed fragment exceeds local scalar count".into(),
            ));
        }
        if let Some(range) = units.contiguous_range() {
            if count > 0
                && (self.slice.starts[2] < range.start as u64
                    || add(self.slice.starts[2], mul(count - 1, self.slice.strides[2])?)?
                        >= range.end as u64)
            {
                return Err(CaptureError::Invalid(
                    "routed fragment addresses an unowned scalar column".into(),
                ));
            }
        } else {
            for index in 0..count {
                let global = add(self.slice.starts[2], mul(index, self.slice.strides[2])?)?;
                if usize::try_from(global)
                    .ok()
                    .and_then(|global| self.ownership.coordinates.units().global_to_local(global))
                    .is_none()
                {
                    return Err(CaptureError::Invalid(
                        "routed fragment addresses an unowned scalar column".into(),
                    ));
                }
            }
        }
        self.native_shape()?;
        Ok(())
    }

    /// Worst-case native source shape. EP receives one input row per original
    /// route; the bound includes nonexporting source peers and only local units.
    pub fn native_shape(&self) -> Result<[u64; 3], CaptureError> {
        Ok([
            self.ownership
                .maximum_source_rows(self.source_tokens, self.geometry.routes_per_token)?,
            if self.ownership.source_peer.is_some() {
                1
            } else {
                self.geometry.routes_per_token
            },
            self.ownership.coordinates.units().local_count() as u64,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{ComponentCoordinateMap, RoutedComponentCoordinateMap};

    #[test]
    fn contiguous_capture_recipe_validates_without_expanding_global_units() {
        let extent = usize::MAX;
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(1, 0..1).unwrap(),
                ComponentCoordinateMap::range(extent, 0..extent).unwrap(),
            ),
            source_peer: None,
            source_peers: 1,
        };
        let slice = ResolvedCaptureSlice {
            starts: vec![0, 0, 0],
            ends: vec![1, 1, extent as u64],
            strides: vec![1, 1, 1],
            shape: vec![1, 1, extent as u64],
        };
        let request = PartitionRoutedUnitCaptureRequest {
            geometry: RoutedUnitGeometry {
                experts: 1,
                units_per_expert: extent as u64,
                routes_per_token: 1,
            },
            source_tokens: 1,
            ownership: &ownership,
            slice: &slice,
        };
        request.validate().unwrap();
        assert_eq!(request.native_shape().unwrap(), [1, 1, extent as u64]);
    }
}
