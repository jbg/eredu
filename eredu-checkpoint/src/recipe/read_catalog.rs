//! Sized lookup indices over the exact metadata retained by an encoded batch.
use super::{RecipeCatalog, RecipeError, StoreError, TensorMetadata};
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

/// Sizes an index over an encoded batch's immutable metadata without copying it.
/// The batch and its source custody remain separate prerequisites.
pub struct ReadBatchCatalogPlan<'a> {
    tensors: &'a [TensorMetadata],
    indices: Layout,
}
impl<'a> ReadBatchCatalogPlan<'a> {
    /// Sizes all source occurrences, including duplicate names.
    pub fn new(tensors: &'a [TensorMetadata]) -> Result<Self, RecipeError> {
        let indices = Layout::array::<usize>(tensors.len())
            .map_err(|_| RecipeError::ArithmeticOverflow("encoded read catalog indices"))?;
        Ok(Self { tensors, indices })
    }
    /// Requested index backing and fixed constructor/result controls. Borrowed
    /// metadata, stack and allocator bookkeeping are not included. Sorting uses
    /// the index in place; owning metadata lookup has separate allocation costs.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<ReadBatchCatalog<'a, C>>(),
            size_of::<ReadBatchCatalogBuildError<C>>(),
            size_of::<Result<ReadBatchCatalog<'a, C>, ReadBatchCatalogBuildError<C>>>(),
            size_of::<Result<(), TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(self.indices.size(), usize::checked_add)
    }

    /// Constructs the single index allocation and retains its caller custody.
    /// Allocation refusal occurs before any metadata is copied or indexed.
    pub fn construct<C>(
        self,
        custody: C,
    ) -> Result<ReadBatchCatalog<'a, C>, ReadBatchCatalogBuildError<C>> {
        let mut rows = Vec::new();
        if let Err(cause) = rows.try_reserve_exact(self.tensors.len()) {
            return Err(ReadBatchCatalogBuildError {
                cause,
                _custody: custody,
            });
        }
        rows.extend(0..self.tensors.len());
        rows.sort_unstable_by(|left, right| {
            self.tensors[*left]
                .name
                .cmp(&self.tensors[*right].name)
                .then(left.cmp(right))
        });
        // Duplicate keys retain the last source occurrence, as an owning map does.
        rows.dedup_by(|later, earlier| {
            if self.tensors[*later].name == self.tensors[*earlier].name {
                *earlier = *later;
                true
            } else {
                false
            }
        });
        Ok(ReadBatchCatalog {
            tensors: self.tensors,
            rows,
            _custody: custody,
        })
    }
}

/// Move-only lookup index borrowing the batch's actual tensor records.
/// Index storage retires before the caller's custody. Owning lookups clone only
/// the selected record; borrowed lookups retain the batch's lifetime.
#[derive(Debug)]
pub struct ReadBatchCatalog<'a, C> {
    tensors: &'a [TensorMetadata],
    rows: Vec<usize>,
    _custody: C,
}
impl<C> ReadBatchCatalog<'_, C> {
    fn lookup(&self, key: &str) -> Option<&TensorMetadata> {
        self.rows
            .binary_search_by(|index| self.tensors[*index].name.as_str().cmp(key))
            .ok()
            .map(|row| &self.tensors[self.rows[row]])
    }
}
impl<C> RecipeCatalog for ReadBatchCatalog<'_, C> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.lookup(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn tensor_metadata_borrowed(&self, key: &str) -> Option<&TensorMetadata> {
        self.lookup(key)
    }
}

/// Index allocation refusal retaining the original custody. The allocation
/// starts from an empty vector, so refusal leaves no allocated index prefix.
#[derive(Debug, Clone)]
pub struct ReadBatchCatalogBuildError<C> {
    cause: TryReserveError,
    _custody: C,
}
impl<C> fmt::Display for ReadBatchCatalogBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C: fmt::Debug> std::error::Error for ReadBatchCatalogBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(test)]
mod tests;
