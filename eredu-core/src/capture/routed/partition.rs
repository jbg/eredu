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

/// Actual provider layout independently of any selected sparse fragment.
/// Empty overlaps and nonexporting replicas retain these source facts too.
#[derive(Debug, Clone, Copy)]
pub struct PartitionRoutedUnitCaptureLayout<'a> {
    /// Global bank equation and original route width.
    pub geometry: RoutedUnitGeometry,
    /// Original token rows per peer for this provider invocation, not the prompt.
    pub source_tokens: u64,
    /// Exact retained expert/unit placement and publication peer.
    pub ownership: &'a RoutedUnitCaptureOwnership,
}
impl PartitionRoutedUnitCaptureLayout<'_> {
    /// Provider hooks see either flattened rows or the original batch/sequence
    /// prefix. Count that prefix without reshaping or allocating the source.
    pub fn input_rows(shape: &[i32]) -> Result<u64, RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        if shape.len() < 2 || shape.last().copied().unwrap_or(0) <= 0 {
            return Err(E::InvocationShape);
        }
        let prefix = &shape[..shape.len() - 1];
        if prefix.iter().any(|&axis| axis < 0) {
            return Err(E::InvocationShape);
        }
        if prefix.contains(&0) { return Ok(0); }
        prefix.iter().try_fold(1_u64, |rows, &axis|
            rows.checked_mul(axis as u64).ok_or(E::Overflow))
    }
    /// Shared exact receive-order coverage rule for an actual provider batch.
    pub fn advance_native_rows(rows:u64,completed:u64,offset:u64,count:u64)
        ->Result<u64,RoutedUnitValidationError> {
        let next=completed.checked_add(count).ok_or(RoutedUnitValidationError::Overflow)?;
        if offset!=completed || next<=completed || next>rows {return Err(RoutedUnitValidationError::Chunks);}
        Ok(next)
    }
    /// Validates retained source geometry without inventing a capture slice.
    pub fn validate(&self) -> Result<(), CaptureError> {
        self.ownership.validate(self.geometry)?;
        self.native_shape()?;
        Ok(())
    }
    /// Maximum input rows and actual local route/unit dimensions.
    pub fn native_shape(&self) -> Result<[u64; 3], CaptureError> {
        Ok([
            self.ownership.maximum_source_rows(self.source_tokens, self.geometry.routes_per_token)?,
            if self.ownership.source_peer.is_some() { 1 } else { self.geometry.routes_per_token },
            self.ownership.coordinates.units().local_count() as u64,
        ])
    }
    /// The ordinary partition invocation check, shared with original-account
    /// collectors. Actual native rows remain separate from the maximum bound.
    pub fn validate_invocation(&self, rows: u64,
        units: &crate::component::ComponentCoordinateMap,
        origins: Option<RoutedUnitOrigins<'_>>) -> Result<(), CaptureError> {
        self.validate_input_invocation(rows, Some(units), origins)
    }
    /// An outer exchange scope may precede bank binding. A missing actual map
    /// remains absent; input extent and original tags are still checked, and
    /// every subsequent batch must provide its actual scalar-column map.
    pub fn validate_input_invocation(&self, rows: u64,
        units: Option<&crate::component::ComponentCoordinateMap>,
        origins: Option<RoutedUnitOrigins<'_>>) -> Result<(), CaptureError> {
        self.validate()?;
        if units.is_some_and(|units| units != self.ownership.coordinates.units()) || rows > self.native_shape()?[0]
            || match origins {
                Some(origins) => self.ownership.source_peer.is_none()
                    || origins.row_count() as u64 != rows
                    || origins.peer_count() as u64 != self.ownership.source_peers
                    || origins.routes_per_token() as u64 != self.geometry.routes_per_token,
                None => self.ownership.source_peer.is_some() || rows != self.source_tokens,
            } {
            return Err(CaptureError::Invalid("actual sparse invocation differs from retained ownership".into()));
        }
        Ok(())
    }
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

    /// This fragment's source facts, before any selected coordinate projection.
    pub fn layout(&self) -> PartitionRoutedUnitCaptureLayout<'_> {
        PartitionRoutedUnitCaptureLayout { geometry: self.geometry, source_tokens: self.source_tokens, ownership: self.ownership }
    }
    /// Worst-case native source shape; never the actual received row count.
    pub fn native_shape(&self) -> Result<[u64; 3], CaptureError> { self.layout().native_shape() }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{ComponentCoordinateMap, RoutedComponentCoordinateMap};

    #[test]
    fn routed_invocation_rows_preserve_unflattened_and_idle_provider_inputs() {
        let rows = PartitionRoutedUnitCaptureLayout::input_rows;
        assert_eq!(rows(&[6, 5]), Ok(6));
        assert_eq!(rows(&[2, 3, 5]), Ok(6));
        assert_eq!(rows(&[1, 2, 3, 5]), Ok(6));
        assert_eq!(rows(&[0, 5]), Ok(0));
        assert_eq!(rows(&[2, 0, 5]), Ok(0));
        for shape in [&[][..], &[5], &[1, 0], &[-1, 5], &[0, -1, 5]] {
            assert_eq!(rows(shape), Err(RoutedUnitValidationError::InvocationShape));
        }
        assert_eq!(rows(&[i32::MAX, i32::MAX, i32::MAX, 5]),
            Err(RoutedUnitValidationError::Overflow));
        assert_eq!(rows(&[i32::MAX, i32::MAX, i32::MAX, 0, 5]), Ok(0));
    }

    #[test]
    fn routed_native_coverage_preserves_offsets_bounds_and_checked_progress() {
        let next=PartitionRoutedUnitCaptureLayout::advance_native_rows;
        assert_eq!(next(7,0,0,3).unwrap(),3);
        assert_eq!(next(7,3,3,4).unwrap(),7);
        assert_eq!(next(7,3,0,4),Err(RoutedUnitValidationError::Chunks));
        assert_eq!(next(7,3,3,0),Err(RoutedUnitValidationError::Chunks));
        assert_eq!(next(7,3,3,5),Err(RoutedUnitValidationError::Chunks));
        assert_eq!(next(u64::MAX,u64::MAX,u64::MAX,1),Err(RoutedUnitValidationError::Overflow));
    }

    #[test]
    fn invocation_layout_preserves_actual_rows_empty_owners_and_unit_order() {
        let geometry = RoutedUnitGeometry { experts: 2, units_per_expert: 4, routes_per_token: 2 };
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(2, 0..1).unwrap(),
                ComponentCoordinateMap::indices(4, vec![3, 1]).unwrap()),
            source_peer: Some(0), source_peers: 2,
        };
        let layout = PartitionRoutedUnitCaptureLayout { geometry, source_tokens: 3, ownership: &ownership };
        assert_eq!(layout.native_shape().unwrap(), [12, 1, 2]);
        let origins = RoutedUnitOrigins::new(&[1, 1], &[0, 5], 2).unwrap();
        layout.validate_invocation(2, ownership.coordinates.units(), Some(origins)).unwrap();
        layout.validate_input_invocation(2, None, Some(origins)).unwrap();
        assert!(layout.validate_input_invocation(3, None, Some(origins)).is_err());
        assert!(layout.validate_input_invocation(2, Some(&ComponentCoordinateMap::indices(4, vec![1, 3]).unwrap()), Some(origins)).is_err());
        assert!(layout.validate_invocation(12, ownership.coordinates.units(), Some(origins)).is_err());
        assert!(layout.validate_invocation(2, &ComponentCoordinateMap::indices(4, vec![1, 3]).unwrap(), Some(origins)).is_err());
        assert!(layout.validate_invocation(3, ownership.coordinates.units(), None).is_err());
        let empty = RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(2, 0..0).unwrap(), ComponentCoordinateMap::range(4, 0..4).unwrap()),
            source_peer: Some(1), source_peers: 2,
        };
        PartitionRoutedUnitCaptureLayout { ownership: &empty, ..layout }.validate_invocation(
            0, empty.coordinates.units(), Some(RoutedUnitOrigins::new(&[0, 0], &[], 2).unwrap())).unwrap();
        let local = RoutedUnitCaptureOwnership { source_peer: None, source_peers: 1, ..ownership.clone() };
        let local_layout = PartitionRoutedUnitCaptureLayout { ownership: &local, ..layout };
        local_layout.validate_invocation(3, local.coordinates.units(), None).unwrap();
        assert!(local_layout.validate_invocation(2, local.coordinates.units(), None).is_err());
    }

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
