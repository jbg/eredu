//! Prospective grouped callback geometry, distinct from completed native rows.
use super::{WorkspaceGroupedObservationSchedule, WorkspaceMetadataError};
use std::ops::Range;

/// A finite source envelope for one selected callback-schedule branch. It owns
/// no tensor, route values, origin tags, native completion or execution grant.
/// If a mechanism changes schedule by row count, it must retain each applicable
/// branch rather than treating this descriptor as proof of another schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceGroupedObservationEnvelope {
    maximum_provider_rows: usize,
    routes_per_row: usize,
    unit_columns: usize,
    schedule: WorkspaceGroupedObservationSchedule,
    empty_batch: bool,
}
impl WorkspaceGroupedObservationEnvelope {
    pub fn new(maximum_provider_rows: usize, routes_per_row: usize, unit_columns: usize,
        schedule: WorkspaceGroupedObservationSchedule, empty_batch: bool)
        -> Result<Self, WorkspaceMetadataError> {
        if routes_per_row == 0 || unit_columns == 0 { return Err(WorkspaceMetadataError::Unqualified); }
        maximum_provider_rows.checked_mul(routes_per_row)
            .and_then(|rows| rows.checked_mul(unit_columns)).ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(Self { maximum_provider_rows, routes_per_row, unit_columns, schedule, empty_batch })
    }
    pub fn maximum_provider_rows(self) -> usize { self.maximum_provider_rows }
    pub fn routes_per_row(self) -> usize { self.routes_per_row }
    pub fn unit_columns(self) -> usize { self.unit_columns }
    pub fn schedule(self) -> WorkspaceGroupedObservationSchedule { self.schedule }
    pub fn maximum_selected_rows(self) -> usize { self.maximum_provider_rows * self.routes_per_row }
    pub fn maximum_unit_elements(self) -> usize { self.maximum_selected_rows() * self.unit_columns }
    pub fn maximum_callbacks(self) -> usize { self.extent(self.maximum_provider_rows).callbacks() }
    /// Refines cardinality only. The native consumer must separately establish
    /// completed rows and authenticate the same source and selected schedule.
    pub fn refine(self, provider_rows: usize) -> Result<WorkspaceGroupedObservationExtent, WorkspaceMetadataError> {
        if provider_rows > self.maximum_provider_rows { return Err(WorkspaceMetadataError::Unqualified); }
        Ok(self.extent(provider_rows))
    }
    fn extent(self, provider_rows: usize) -> WorkspaceGroupedObservationExtent {
        WorkspaceGroupedObservationExtent { source: self, provider_rows }
    }
}

/// Source-checked arithmetic for a supplied row count. This is deliberately
/// separate from both the prospective envelope and a concrete native batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceGroupedObservationExtent {
    source: WorkspaceGroupedObservationEnvelope,
    provider_rows: usize,
}
impl WorkspaceGroupedObservationExtent {
    pub fn source(self) -> WorkspaceGroupedObservationEnvelope { self.source }
    pub fn provider_rows(self) -> usize { self.provider_rows }
    pub fn selected_rows(self) -> usize { self.provider_rows * self.source.routes_per_row }
    pub fn unit_elements(self) -> usize { self.selected_rows() * self.source.unit_columns }
    pub fn callbacks(self) -> usize {
        if self.provider_rows == 0 { return usize::from(self.source.empty_batch); }
        match self.source.schedule {
            WorkspaceGroupedObservationSchedule::WholeBatch => 1,
            WorkspaceGroupedObservationSchedule::TokenChunks(chunk) => self.provider_rows.div_ceil(chunk.get() as usize),
        }
    }
    /// Provider-token ranges only; selected rows stay independently sorted
    /// inside each callback and this supplies no route or origin values.
    pub fn provider_range(self, callback: usize) -> Option<Range<usize>> {
        if callback >= self.callbacks() { return None; }
        match self.source.schedule {
            WorkspaceGroupedObservationSchedule::WholeBatch => Some(0..self.provider_rows),
            WorkspaceGroupedObservationSchedule::TokenChunks(chunk) => {
                let start=callback.checked_mul(chunk.get() as usize)?;
                Some(start..start.saturating_add(chunk.get() as usize).min(self.provider_rows))
            }
        }
    }
    pub fn selected_rows_in(self, callback: usize) -> Option<usize> {
        self.provider_range(callback)?.len().checked_mul(self.source.routes_per_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;
    #[test]
    fn prospective_grouped_rows_refine_without_inventing_received_routes() {
        let source=WorkspaceGroupedObservationEnvelope::new(7,2,3,
            WorkspaceGroupedObservationSchedule::TokenChunks(NonZeroU32::new(3).unwrap()),false).unwrap();
        assert_eq!((source.maximum_provider_rows(),source.maximum_selected_rows(),source.maximum_unit_elements(),source.maximum_callbacks()),(7,14,42,3));
        let actual=source.refine(4).unwrap();
        assert_eq!((actual.provider_rows(),actual.selected_rows(),actual.unit_elements(),actual.callbacks()),(4,8,24,2));
        assert_eq!(actual.provider_range(0),Some(0..3));
        assert_eq!(actual.provider_range(1),Some(3..4));
        assert_eq!(actual.selected_rows_in(1),Some(2));
        assert_eq!(actual.provider_range(2),None);
        assert_eq!(source.refine(0).unwrap().callbacks(),0);
        assert!(source.refine(8).is_err());
        let grouped_empty=WorkspaceGroupedObservationEnvelope::new(0,2,3,WorkspaceGroupedObservationSchedule::WholeBatch,true).unwrap();
        assert_eq!(grouped_empty.refine(0).unwrap().provider_range(0),Some(0..0));
        assert!(WorkspaceGroupedObservationEnvelope::new(usize::MAX,2,3,WorkspaceGroupedObservationSchedule::WholeBatch,false).is_err());
    }
}
