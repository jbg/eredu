//! Sized lookup indices over the exact metadata retained by an encoded batch.
use super::{RecipeCatalog, RecipeError, StoreError, TensorMetadata};
use std::alloc::Layout;

pub(super) struct ReadBatchCatalogPlan<'a> {
    tensors: &'a [TensorMetadata],
    indices: Layout,
}
impl<'a> ReadBatchCatalogPlan<'a> {
    pub(super) fn new(tensors: &'a [TensorMetadata]) -> Result<Self, RecipeError> {
        let indices = Layout::array::<usize>(tensors.len())
            .map_err(|_| RecipeError::ArithmeticOverflow("encoded read catalog indices"))?;
        Ok(Self { tensors, indices })
    }
    pub(super) fn build(self) -> ReadBatchCatalog<'a> {
        let mut rows = Vec::with_capacity(self.indices.size() / std::mem::size_of::<usize>());
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
        ReadBatchCatalog {
            tensors: self.tensors,
            rows,
        }
    }
}

pub(super) struct ReadBatchCatalog<'a> {
    tensors: &'a [TensorMetadata],
    rows: Vec<usize>,
}
impl ReadBatchCatalog<'_> {
    fn lookup(&self, key: &str) -> Option<&TensorMetadata> {
        self.rows
            .binary_search_by(|index| self.tensors[*index].name.as_str().cmp(key))
            .ok()
            .map(|row| &self.tensors[self.rows[row]])
    }
}
impl RecipeCatalog for ReadBatchCatalog<'_> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.lookup(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn tensor_metadata_borrowed(&self, key: &str) -> Option<&TensorMetadata> {
        self.lookup(key)
    }
}

#[cfg(test)]
mod tests;
