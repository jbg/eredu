//! A binding output and its actual encoded source occurrences are independent.
use super::*;
use eredu_checkpoint::recipe::{EncodedRecipeReadView, RecipeMetadata};

pub(in crate::backend::runtime::residency::manager::construction) enum ReadValue {
    Direct(DirectRecipeRead),
    Materialized(PreparedMaterializedRead),
}

impl ReadValue {
    pub(in crate::backend::runtime::residency::manager::construction) fn direct(
        &self,
    ) -> Option<&DirectRecipeRead> {
        match self {
            Self::Direct(read) => Some(read),
            Self::Materialized(_) => None,
        }
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn shape(&self) -> &[i32] {
        match self {
            Self::Direct(read) => read.shape(),
            Self::Materialized(read) => &read.shape,
        }
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn dtype(
        &self,
    ) -> safemlx::Dtype {
        match self {
            Self::Direct(read) => read.dtype(),
            Self::Materialized(read) => read.dtype,
        }
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn output(
        &self,
    ) -> &RecipeMetadata {
        match self {
            Self::Direct(read) => read.encoded().output(),
            Self::Materialized(read) => &read.output,
        }
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn leaf_count(
        &self,
    ) -> usize {
        match self {
            Self::Direct(_) => 1,
            Self::Materialized(read) => read.leaves.len(),
        }
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn leaf(
        &self,
        ordinal: usize,
    ) -> Option<EncodedRecipeReadView<'_>> {
        match self {
            Self::Direct(read) => (ordinal == 0).then(|| read.encoded().borrowed()),
            Self::Materialized(read) => read
                .leaves
                .get(ordinal)
                .map(|leaf| leaf.read.encoded().borrowed()),
        }
    }
}

/// Exact-size flattening retains repeated leaf occurrences. Its remaining
/// count comes from checked source geometry, never a guessed output count.
#[derive(Clone)]
pub(in crate::backend::runtime::residency::manager::construction) struct EncodedLeaves<'a> {
    rows: &'a [PlannedRead],
    row: usize,
    leaf: usize,
    remaining: usize,
}
impl<'a> EncodedLeaves<'a> {
    pub(in crate::backend::runtime::residency::manager::construction) fn new(
        rows: &'a [PlannedRead],
    ) -> Option<Self> {
        Some(Self {
            rows,
            row: 0,
            leaf: 0,
            remaining: rows.iter().try_fold(0usize, |count, row| {
                count.checked_add(row.read.leaf_count())
            })?,
        })
    }
}
impl<'a> Iterator for EncodedLeaves<'a> {
    type Item = EncodedRecipeReadView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let row = self.rows.get(self.row)?;
            if let Some(value) = row.read.leaf(self.leaf) {
                self.leaf += 1;
                self.remaining -= 1;
                return Some(value);
            }
            self.row += 1;
            self.leaf = 0;
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for EncodedLeaves<'_> {}
