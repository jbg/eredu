//! One row sink for the shared rotary equation. Original rows consume the
//! admitted native Graph bank; ordinary rows retain their existing local Vec.
use super::*;
use safemlx::ops::{OriginalArrayRows, OriginalArrayRowsLayout};

pub(super) enum Rows<'a> {
    Ordinary(ArrayOwners),
    Original(OriginalArrayRows<'a>),
}
impl<'a> Rows<'a> {
    pub(super) fn new(
        capacity: usize,
        rank: usize,
        execution: &'a original::Execution,
    ) -> Result<Self, Error> {
        match execution.observer() {
            None => Ok(Self::Ordinary(ArrayOwners::with_capacity(capacity))),
            Some(observer) => {
                let layout = OriginalArrayRowsLayout::inspect(capacity, rank)
                    .ok_or_else(|| execution.profile_error())?;
                OriginalArrayRows::new(layout, observer)
                    .map(Self::Original)
                    .map_err(|cause| execution.source(cause))
            }
        }
    }
    pub(super) fn push(
        &mut self,
        array: Array,
        execution: &original::Execution,
    ) -> Result<(), Error> {
        match self {
            Self::Ordinary(rows) => {
                rows.push(array);
                Ok(())
            }
            Self::Original(rows) => rows.push(&array).map_err(|cause| execution.source(cause)),
        }
    }
    pub(super) fn concatenate(
        self,
        stream: &Stream,
        execution: &original::Execution,
    ) -> Result<Array, Error> {
        execution.native(match self {
            Self::Ordinary(rows) => concatenate_axis(&rows, -1, stream),
            Self::Original(rows) => rows.concatenate(-1, stream),
        })
    }
}
