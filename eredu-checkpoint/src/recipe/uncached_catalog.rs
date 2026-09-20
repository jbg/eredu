//! Borrowed catalog view for temporary inference and selection work.
use super::{RecipeCatalog, StoreError, TensorMetadata};

/// Borrows source metadata while leaving all inference results with the caller.
///
/// Recursive recipe inference, selection pushdown and peak sizing use their
/// ordinary implementations without consulting or populating the catalog's
/// persistent cache. Repeated calls recompute metadata. The view itself allocates
/// nothing; inference and rewriting still allocate temporary metadata and recipes,
/// whose admission and lifetime remain the caller's responsibility.
#[derive(Debug)]
pub struct UncachedRecipeCatalog<'a, C: ?Sized> {
    catalog: &'a C,
}

impl<'a, C: RecipeCatalog + ?Sized> UncachedRecipeCatalog<'a, C> {
    /// Borrows an existing catalog without accessing its inference cache.
    pub fn new(catalog: &'a C) -> Self {
        Self { catalog }
    }
}

impl<C: RecipeCatalog + ?Sized> RecipeCatalog for UncachedRecipeCatalog<'_, C> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.catalog.tensor_metadata(key)
    }

    fn tensor_metadata_borrowed(&self, key: &str) -> Option<&TensorMetadata> {
        self.catalog.tensor_metadata_borrowed(key)
    }
}

#[cfg(test)]
mod tests;
