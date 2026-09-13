//! Sparse selected-unit coordinates, bounded host evidence and checked receipts.
use super::*;

mod origins;
mod partition;
pub use origins::{RoutedUnitOrigin, RoutedUnitOrigins};
pub use partition::{PartitionRoutedUnitCaptureRequest, PartitionRoutedUnitCaptureSource};

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
        if self.coordinates.experts().global_count() as u64 != geometry.experts
            || self.coordinates.units().global_count() as u64 != geometry.units_per_expert
            || self.source_peers == 0
            || match self.source_peer {
                Some(peer) => peer >= self.source_peers,
                None => self.source_peers != 1,
            }
        {
            return Err(CaptureError::Invalid(
                "invalid sparse producer ownership".into(),
            ));
        }
        Ok(())
    }
    /// Conservative receive-order token-row bound, including unexported source
    /// peers. Exchange expands each original route to one native input row.
    pub fn maximum_source_rows(
        &self,
        source_tokens: u64,
        routes: u64,
    ) -> Result<u64, CaptureError> {
        if self.source_peer.is_some() {
            mul(mul(source_tokens, routes)?, self.source_peers)
        } else {
            Ok(source_tokens)
        }
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
        if self.experts == 0 || self.units_per_expert == 0 || self.routes_per_token == 0 {
            return Err(CaptureError::Invalid("empty routed-unit geometry".into()));
        }
        mul(self.experts, self.units_per_expert)
    }

    /// Selection in original `[token, route, component]` order. Native sorting
    /// and chunking never change these application coordinates.
    pub fn validate_slice(self, slice: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        self.components()?;
        if slice.starts.len() != 3
            || slice.ends.len() != 3
            || slice.strides.len() != 3
            || slice.shape.len() != 3
            || slice.ends[1] > self.routes_per_token
            || slice.ends[2] > self.units_per_expert
        {
            return Err(CaptureError::Invalid(
                "invalid routed-unit slice rank or extent".into(),
            ));
        }
        for axis in 0..3 {
            if slice.strides[axis] == 0
                || slice.starts[axis] > slice.ends[axis]
                || slice.shape[axis]
                    != (slice.ends[axis] - slice.starts[axis]).div_ceil(slice.strides[axis])
            {
                return Err(CaptureError::Invalid(
                    "invalid routed-unit slice geometry".into(),
                ));
            }
        }
        Ok(())
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

impl RoutedUnitCapture {
    /// Checks selected-unit geometry and route uniqueness. Completeness is checked
    /// separately so independently bounded native chunks can be accumulated.
    pub fn validate_rows(&self, slice: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        self.geometry.validate_slice(slice)?;
        let mut seen = std::collections::BTreeSet::new();
        for row in &self.rows {
            if !row.coefficient.is_finite()
                || row.token < slice.starts[0]
                || row.token >= slice.ends[0]
                || !(row.token - slice.starts[0]).is_multiple_of(slice.strides[0])
                || row.expert >= self.geometry.experts
                || row.slot < slice.starts[1]
                || row.slot >= slice.ends[1]
                || !(row.slot - slice.starts[1]).is_multiple_of(slice.strides[1])
                || !seen.insert((row.source_peer, row.token, row.slot))
            {
                return Err(CaptureError::Invalid(
                    "invalid or duplicate routed-unit receipt".into(),
                ));
            }
            let (start, stride, count) = (slice.starts[2], slice.strides[2], slice.shape[2]);
            if row.unit_start != start
                || row.unit_stride != stride
                || row.values.shape()
                    != [usize::try_from(count).map_err(|_| CaptureError::Overflow)?]
                || !matches!(row.values.data(), crate::TensorObservationData::F32(_))
            {
                return Err(CaptureError::Invalid(
                    "routed-unit receipt differs from selection".into(),
                ));
            }
        }
        Ok(())
    }

    /// Finishes an ordinary invocation only after every selected token/route has
    /// arrived. Exchanged rows require their separate distributed ownership proof.
    pub fn finish_ordinary(
        &mut self,
        slice: &ResolvedCaptureSlice,
        source_tokens: u64,
    ) -> Result<(), CaptureError> {
        self.validate_rows(slice)?;
        self.source_token_ranges.sort_unstable();
        let mut end = 0;
        for range in &self.source_token_ranges {
            if range[0] != end || range[1] <= range[0] || range[1] > source_tokens {
                return Err(CaptureError::Invalid(
                    "duplicate or incomplete routed-unit chunks".into(),
                ));
            }
            end = range[1];
        }
        if self.rows.iter().any(|row| row.source_peer.is_some())
            || end != source_tokens
            || self.rows.len() as u64 != mul(slice.shape[0], slice.shape[1])?
        {
            return Err(CaptureError::Invalid(
                "incomplete ordinary routed-unit capture".into(),
            ));
        }
        self.rows.sort_by_key(|row| (row.token, row.slot));
        Ok(())
    }
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
