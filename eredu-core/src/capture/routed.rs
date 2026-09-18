//! Sparse selected-unit coordinates, bounded host evidence and checked receipts.
use super::*;

mod origins;
mod partition;
mod validation;
pub use validation::{RoutedUnitAssemblyError,RoutedUnitRowIdentity, RoutedUnitValidationError};
pub use origins::{RoutedUnitOrigin, RoutedUnitOrigins};
pub use partition::{PartitionRoutedUnitCaptureLayout, PartitionRoutedUnitCaptureRequest, PartitionRoutedUnitCaptureSource};

/// Retained placement for one sparse producer. These host facts describe
/// ownership; creating or deserializing them grants no execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedUnitCaptureOwnership {
    /// Selected checkpoint-global experts and within-expert scalar columns.
    pub coordinates: crate::component::RoutedComponentCoordinateMap,
    /// Authoritative source peer for the logical input; absent without exchange.
    pub source_peer: Option<u64>,
    /// Number of source peers whose routes may reach this native producer.
    pub source_peers: u64,
}
impl RoutedUnitCaptureOwnership {
    /// Checks placement against global capture geometry, without native work.
    pub fn validate(&self, geometry: RoutedUnitGeometry) -> Result<(), CaptureError> {
        geometry.components()?;
        if !self.matches_geometry(geometry)
        {
            return Err(CaptureError::Invalid(
                "invalid sparse producer ownership".into(),
            ));
        }
        Ok(())
    }
    fn matches_geometry(&self, geometry: RoutedUnitGeometry) -> bool {
        self.coordinates.experts().global_count() as u64 == geometry.experts
            && self.coordinates.units().global_count() as u64 == geometry.units_per_expert
            && self.source_peers > 0
            && match self.source_peer {
                Some(peer) => peer < self.source_peers,
                None => self.source_peers == 1,
            }
    }
    /// The same ownership predicate with an allocation-free typed cause.
    pub fn validate_geometry(&self, geometry: RoutedUnitGeometry) -> Result<(), RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        if geometry.experts == 0 || geometry.units_per_expert == 0 || geometry.routes_per_token == 0 {
            return Err(E::Empty);
        }
        geometry.experts.checked_mul(geometry.units_per_expert).ok_or(E::Overflow)?;
        if !self.matches_geometry(geometry) { return Err(E::Ownership); }
        Ok(())
    }
    /// Exact same conservative row bound, without allocating an error string.
    pub fn maximum_source_rows_checked(&self, source_tokens: u64, routes: u64)
        -> Result<u64, RoutedUnitValidationError> {
        if self.source_peer.is_some() {
            source_tokens.checked_mul(routes).and_then(|n| n.checked_mul(self.source_peers))
                .ok_or(RoutedUnitValidationError::Overflow)
        } else { Ok(source_tokens) }
    }
    /// Conservative receive-order token-row bound, including unexported source
    /// peers. Exchange expands each original route to one native input row.
    pub fn maximum_source_rows(
        &self,
        source_tokens: u64,
        routes: u64,
    ) -> Result<u64, CaptureError> {
        self.maximum_source_rows_checked(source_tokens, routes).map_err(Into::into)
    }
}

/// Native chunk evidence retained with a distributed producer's contribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedUnitCaptureProvenance {
    /// Admission's expected expert, unit and source-peer placement.
    pub ownership: RoutedUnitCaptureOwnership,
    /// Actual receive-order native chunks, separate from logical token positions.
    pub source_token_ranges: Vec<[u64; 2]>,
}

/// Global bank geometry, independent of compact native bank indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedUnitGeometry {
    /// Checkpoint-global expert count.
    pub experts: u64,
    /// Scalar units per expert.
    pub units_per_expert: u64,
    /// Original selected route slots per token; duplicate experts retain their slots.
    pub routes_per_token: u64,
}
impl RoutedUnitGeometry {
    /// Number of virtual components. This does not allocate an expert-dense tensor.
    pub fn components(self) -> Result<u64, CaptureError> {
        self.components_with(super::admission::allocation::Allocation(None))
    }
    /// Validates the same geometry and funds any semantic diagnostic before
    /// construction. The enclosing source retains this account with the result.
    pub fn components_with_metadata(self,funding:&crate::HostMetadataFunding)->Result<u64,CaptureError>{
        funding.reserve_metadata(std::mem::size_of::<(Self,Result<u64,CaptureError>)>())
            .map_err(super::admission::allocation::CaptureAdmissionStorageError::from)?;
        self.components_with(super::admission::allocation::Allocation(Some(funding)))
    }
    pub(crate) fn components_with(self, allocation: super::admission::allocation::Allocation<'_>) -> Result<u64, CaptureError> {
        if self.experts == 0 || self.units_per_expert == 0 || self.routes_per_token == 0 {
            return Err(CaptureError::Invalid(allocation.text("empty routed-unit geometry")?));
        }
        mul(self.experts, self.units_per_expert)
    }

    /// Validate borrowed sparse axes without allocating a slice descriptor.
    /// This is geometry validation only and grants no construction authority.
    pub fn validate_axes(self, starts: &[u64], ends: &[u64], strides: &[u64], shape: &[u64])
        -> Result<(), RoutedUnitValidationError> {
        validation::validate_axes(self, starts, ends, strides, shape)
    }

    /// Selection in original `[token, route, component]` order. Native sorting
    /// and chunking never change these application coordinates.
    pub fn validate_slice(self, slice: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        validation::validate_slice(self, slice).map_err(Into::into)
    }
}

/// One selected route. Expert identity is retained alongside its selected units;
/// no row is fabricated for an expert that did not participate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedUnitCaptureRow {
    /// Source peer in the retained expert-exchange group, absent for ordinary work.
    pub source_peer: Option<u64>,
    /// Original flattened operator token row, separate from prediction index.
    pub token: u64,
    /// Original selected route slot; duplicates remain separate.
    pub slot: u64,
    /// Checkpoint-global expert ordinal.
    pub expert: u64,
    /// Exact route coefficient before expert reduction.
    pub coefficient: f32,
    /// First captured unit within the expert.
    pub unit_start: u64,
    /// Positive stride between captured units.
    pub unit_stride: u64,
    /// One-dimensional F32 host values, preserving non-finite values on the wire.
    #[serde(with = "super::tensor_wire")]
    pub values: TensorObservation,
}

/// Complete sparse evidence for the selected token rows. Rows are ordered by
/// `(source_peer, token, slot)` after receipt validation, not native sorting order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedUnitCapture {
    /// Actual global expert/unit/route geometry.
    pub geometry: RoutedUnitGeometry,
    /// Half-open original token ranges consumed by native chunks, including
    /// tokens excluded by the selection. Receipt completion rejects gaps/overlap.
    /// An assembled distributed payload leaves this empty and retains each
    /// producer's receive-order chunks in `RoutedUnitCaptureProvenance` instead.
    pub source_token_ranges: Vec<[u64; 2]>,
    /// Original route rows; unselected experts have no row.
    pub rows: Vec<RoutedUnitCaptureRow>,
}

/// Borrowed native inputs needed to resolve sorted/chunked routes. Only an
/// admitted collector may evaluate or copy these arrays. No array is retained.
pub struct RoutedUnitCaptureSource<'a, T> {
    /// Actual `[selected_routes, local_units]` values.
    pub values: &'a T,
    /// Native chunk-relative token index for each sorted row.
    pub token_indices: &'a T,
    /// Flattened selection index into the chunk's route coefficients.
    pub selection_indices: &'a T,
    /// Original `[chunk_tokens, top_k]` coefficients.
    pub coefficients: &'a T,
    /// Original pre-compaction `[provider_tokens, top_k]` expert IDs.
    pub source_groups: &'a T,
    /// Native plus provider chunk start in source_groups.
    pub token_offset: u64,
    /// Source group ID to global expert ID; absent means already global.
    pub global_groups: Option<&'a [usize]>,
}
