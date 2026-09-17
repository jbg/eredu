//! Closed compressed-copy projection: full source inventory, two logical copies.

use super::PreparedCompressedCopy;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_nn::Error;
use eredu_runtime::working_memory::WorkspaceCompressedCache;
use safemlx::Array;
use std::num::NonZeroU32;

impl<'a> PreparedCompressedCopy<'a> {
    /// Complete retained source descriptors, including independently restored
    /// capacity stores. This is deliberately distinct from the two operands.
    pub(crate) fn visit_retained_arrays(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for array in self
            .source
            .latent_storage
            .iter()
            .chain(&self.source.rotary_key_storage)
            .chain(&self.source.latent)
            .chain(&self.source.rotary_key)
        {
            visitor(array);
        }
    }

    /// Uses the existing portable compact-copy equations after validating the
    /// actual native source geometry. The two copied logical arrays each become
    /// both the new store and its logical view; padding is not copied.
    ///
    /// The caller must import every retained source into this same projection
    /// before opening the aggregate copy span, and retain that complete source
    /// inventory through admission and native completion. No credit, execution
    /// grant, native evaluation or completion is supplied by this projection.
    pub(crate) fn project_copied_workspace(
        &self,
        batch: NonZeroU32,
        latent_width: NonZeroU32,
        rotary_width: NonZeroU32,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceCompressedCache, Error> {
        let source =
            self.source
                .project_workspace_state(batch, latent_width, rotary_width, projection)?;
        source.isolated_snapshot(projection.context())
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
