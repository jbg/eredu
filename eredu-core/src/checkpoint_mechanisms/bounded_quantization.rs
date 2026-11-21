//! Native operations for one portable bounded checkpoint conversion.

use std::num::NonZeroUsize;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::CheckpointSource,
    StoredDtype, WeightQuantization,
};

use crate::BackendFailure;

/// Hardware limits consumed by the portable tile scheduler.
pub struct QuantizationTileLimits {
    /// Maximum simultaneous submissions on separate native queues.
    pub slots: NonZeroUsize,
    /// Maximum dense scalar elements in one native submission.
    pub elements: NonZeroUsize,
}

/// Native context for packing exact checkpoint recipe tiles.
pub trait BoundedQuantizationBackend {
    /// Versioned identity of the encoder, platform, and execution device.
    fn cache_identity(&self) -> Result<String, BackendFailure>;
    /// Reports supported native tile geometry and queue count.
    fn limits(&self) -> QuantizationTileLimits;
    /// Opens exactly the requested number of native submission slots.
    fn begin(&mut self, slots: usize, working_set_bytes: u64) -> Result<(), BackendFailure>;
    /// Releases unused native allocator storage using the remaining byte headroom.
    fn trim(&mut self, available_bytes: u64) -> Result<(), BackendFailure>;
    /// Submits a preflighted input tile on one free slot and retains its source leases.
    fn submit(
        &mut self,
        slot: usize,
        input: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        encoding: WeightQuantization,
        companion_dtype: RecipeDtype,
    ) -> Result<Box<dyn QuantizationTileCompletion>, BackendFailure>;
    /// Reclaims unused native resources after all submissions have completed.
    fn finish(&mut self) -> Result<(), BackendFailure>;
}

/// Native submission ownership through exact completion and readback.
pub trait QuantizationTileCompletion {
    /// Waits for this submission and returns its packed weight and companions in order.
    fn complete(self: Box<Self>) -> Result<Box<dyn EncodedQuantizationTile>, BackendFailure>;
}

/// Host-readable encoded outputs retained through portable checkpoint writes.
pub trait EncodedQuantizationTile {
    /// Number of physical outputs.
    fn output_count(&self) -> usize;
    /// Borrows one output's little-endian bytes and physical scalar encoding.
    fn output(&self, index: usize) -> Result<(StoredDtype, &[u8]), BackendFailure>;
}
