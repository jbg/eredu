//! Borrowed metadata from an already authenticated encoded read.
use super::{EncodedRecipeRead, RecipeMetadata};
use crate::store::{DetachedEncodedReadPlan, EncodedReadBatch, EncodedReadLayout, TensorMetadata};

/// Immutable view of one admitted read, independent of its owner's custody
/// type. Only an existing `EncodedRecipeRead` can create this view; it borrows
/// that owner's exact output and source records without copying or releasing
/// their custody. The view grants no new source admission or payload storage.
#[derive(Clone, Copy)]
pub struct EncodedRecipeReadView<'a> {
    output: &'a RecipeMetadata,
    batch: &'a EncodedReadBatch,
}

impl<C> EncodedRecipeRead<C> {
    /// Borrows the authenticated metadata while retaining this read's complete
    /// owner. Mixed custody types can share the ordinary detached-read worker.
    pub fn borrowed(&self) -> EncodedRecipeReadView<'_> {
        EncodedRecipeReadView {
            output: &self.output,
            batch: &self.batch,
        }
    }
}

impl<'a> EncodedRecipeReadView<'a> {
    /// Exact inferred output metadata from the retained read.
    pub fn output(self) -> &'a RecipeMetadata {
        self.output
    }

    /// Exact encoded sources in destination order.
    pub fn sources(self) -> &'a [TensorMetadata] {
        self.batch.tensors()
    }

    pub(crate) fn admitted_batch(self) -> &'a EncodedReadBatch {
        self.batch
    }

    /// The same detached metadata constructor over borrowed authenticated
    /// reads. It allocates only after the caller supplies its source custody.
    pub fn prepare_detached<I>(reads: I) -> Option<DetachedEncodedReadPlan<'a, I>>
    where
        I: Iterator<Item = Self> + Clone + ExactSizeIterator,
    {
        DetachedEncodedReadPlan::inspect(reads)
    }

    /// Synchronous source scratch, without cloning metadata or custody.
    pub fn borrowed_read_layout(
        reads: impl IntoIterator<Item = Self>,
    ) -> Option<EncodedReadLayout> {
        EncodedReadLayout::inspect(reads.into_iter().map(|read| read.batch))
    }
}

#[cfg(test)]
mod tests;
