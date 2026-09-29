//! Compact row-addressable sources. A table is one architecture-authored recipe,
//! not one catalog entry per row. Preparation never reads table payloads.
use std::{collections::BTreeMap, ops::Range, path::PathBuf, sync::Arc};

use crate::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError, RecipeMetadata},
    store::{
        CheckpointLease, CheckpointSource, EncodedReadBatch, PreparedCheckpointSource,
        PreparedTensorSource, RestrictedCheckpointSource, SharedCheckpointSource, StoreError,
        TensorMetadata, TensorReadRequest, TensorSelection, TensorSourceProvenance,
        WeightStoreDiagnostics,
    },
};

/// Hard host planning limits supplied by the consuming residency mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowReadLimits {
    /// Maximum request count, including repeated rows.
    pub requests: usize,
    /// Maximum distinct rows materialized by one read recipe.
    pub rows_per_read: usize,
}

/// Admission failures before any table payload is read.
#[derive(Debug, thiserror::Error)]
pub enum RowReadError {
    /// Underlying exact source failed validation.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A recipe cannot supply bounded matrix rows.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The logical source is not a nonempty matrix.
    #[error("row source requires a nonempty matrix, got {shape:?}")]
    Geometry {
        /// Invalid logical shape.
        shape: Vec<usize>,
    },
    /// Both limits must be positive.
    #[error("row-read request and per-read limits must be positive")]
    InvalidLimits,
    /// Request metadata exceeds admitted capacity.
    #[error("row request count {requested} exceeds limit {limit}")]
    RequestLimit {
        /// Actual request count.
        requested: usize,
        /// Admitted request count.
        limit: usize,
    },
    /// An integer row is outside the logical matrix.
    #[error("row {row} is outside 0..{rows}")]
    OutOfRange {
        /// Requested row.
        row: u64,
        /// Matrix row count.
        rows: usize,
    },
}

/// Retained exact source and one compact table recipe, independent of backend.
///
/// Dynamic row recipes deliberately bypass the immutable recipe-inference cache:
/// retaining every historic row selection would make repeated lookup unbounded.
pub struct PreparedRowSource {
    source: SharedCheckpointSource,
    recipe: DerivedWeightRecipe,
    metadata: RecipeMetadata,
}

impl std::fmt::Debug for PreparedRowSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRowSource")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl PreparedRowSource {
    /// Pins only the source keys used by this table, without acquiring payloads.
    /// Concatenated shard recipes remain compact until a bounded read is planned.
    pub fn new(
        source: SharedCheckpointSource,
        recipe: DerivedWeightRecipe,
    ) -> Result<Arc<Self>, RowReadError> {
        let allowed = recipe
            .source_keys()
            .into_iter()
            .map(str::to_owned)
            .collect();
        let restricted = Arc::new(RestrictedCheckpointSource::including(
            source,
            "row-source",
            allowed,
        )?);
        let catalog = restricted
            .source_keys()
            .into_iter()
            .map(|key| {
                let metadata = restricted.source_metadata(&key)?;
                let provenance = restricted.source_provenance(&key)?;
                Ok((
                    key,
                    PreparedTensorSource {
                        metadata,
                        provenance,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, StoreError>>()?;
        let source: SharedCheckpointSource =
            Arc::new(PreparedCheckpointSource::new(restricted, catalog)?);
        let metadata = recipe.infer(source.as_ref())?;
        if metadata.shape.len() != 2 || metadata.shape.contains(&0) {
            return Err(RowReadError::Geometry {
                shape: metadata.shape,
            });
        }
        Ok(Arc::new(Self {
            source,
            recipe,
            metadata,
        }))
    }

    /// Complete recipe geometry; packed encodings can expose an encoded row width.
    /// Does not imply a resident table allocation.
    pub fn metadata(&self) -> &RecipeMetadata {
        &self.metadata
    }

    /// Deduplicates, sorts and coalesces adjacent rows into bounded recipes.
    /// The returned permutation restores the original order and multiplicity.
    pub fn plan(
        self: &Arc<Self>,
        rows: &[u64],
        limits: RowReadLimits,
    ) -> Result<RowReadPlan, RowReadError> {
        if limits.requests == 0 || limits.rows_per_read == 0 {
            return Err(RowReadError::InvalidLimits);
        }
        if rows.len() > limits.requests {
            return Err(RowReadError::RequestLimit {
                requested: rows.len(),
                limit: limits.requests,
            });
        }
        for &row in rows {
            if usize::try_from(row)
                .ok()
                .is_none_or(|row| row >= self.metadata.shape[0])
            {
                return Err(RowReadError::OutOfRange {
                    row,
                    rows: self.metadata.shape[0],
                });
            }
        }
        let mut unique = rows.iter().map(|row| *row as usize).collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        let restore = rows
            .iter()
            .map(|row| {
                unique
                    .binary_search(&(*row as usize))
                    .expect("validated requested row")
            })
            .collect();
        let mut reads = Vec::new();
        let mut start = 0;
        while start < unique.len() {
            let mut end = start + 1;
            while end < unique.len()
                && end - start < limits.rows_per_read
                && unique[end] == unique[end - 1] + 1
            {
                end += 1;
            }
            let recipe = self.recipe.select_bounded(
                self.as_ref(),
                TensorSelection::Range {
                    axis: 0,
                    start: unique[start],
                    end: unique[end - 1] + 1,
                },
            )?;
            reads.push(RowRead {
                destinations: start..end,
                recipe,
            });
            start = end;
        }
        Ok(RowReadPlan {
            source: Arc::clone(self),
            unique,
            restore,
            reads,
        })
    }
}

impl RecipeCatalog for PreparedRowSource {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
}

// A retained restricted source with no cache of transient selected recipes.
impl CheckpointSource for PreparedRowSource {
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.source.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.source.source_provenance(key)
    }
    fn prepare_encoded_read(
        &self,
        keys: &[TensorReadRequest],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        self.source.prepare_encoded_read(keys)
    }
    fn materialized_source_keys(&self) -> Vec<String> {
        self.source.materialized_source_keys()
    }
    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.source.materialized_source_shards()
    }
    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source.unclaimed_checkpoint_keys()
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.source.is_authoritative_materialized_key(key)
    }
    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }
}

/// One bounded materialization over selected source leaves.
#[derive(Debug, Clone)]
pub struct RowRead {
    destinations: Range<usize>,
    recipe: DerivedWeightRecipe,
}
impl RowRead {
    /// Rows written in the compact unique output, in ascending logical order.
    pub fn destinations(&self) -> Range<usize> {
        self.destinations.clone()
    }
    /// Bounded recipe; only its selected source leaves may be acquired.
    pub fn recipe(&self) -> &DerivedWeightRecipe {
        &self.recipe
    }
}

/// Lookup plan retaining source provenance through delayed/native consumption.
pub struct RowReadPlan {
    source: Arc<PreparedRowSource>,
    unique: Vec<usize>,
    restore: Vec<usize>,
    reads: Vec<RowRead>,
}
impl RowReadPlan {
    /// Exact restricted source, with transient recipe caching disabled.
    pub fn source(&self) -> &dyn CheckpointSource {
        self.source.as_ref()
    }
    /// Sorted unique logical rows.
    pub fn unique_rows(&self) -> &[usize] {
        &self.unique
    }
    /// Compact row index for each original request, including duplicates.
    pub fn restore_order(&self) -> &[usize] {
        &self.restore
    }
    /// Coalesced reads; execute and retire each before the next under tight budgets.
    pub fn reads(&self) -> &[RowRead] {
        &self.reads
    }
}

#[cfg(test)]
mod tests;
