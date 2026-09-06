//! Bounded load-time weight quantization into resident memory.

//!
//! A [`BoundedQuantizedWeightStore`](crate::backend::runtime::checkpoint::bounded_quantization::BoundedQuantizedWeightStore)
//! overlays packed tensors on an existing
//! checkpoint store. Source matrices are selected in row tiles, quantized on
//! explicit conversion streams, and written directly into final in-memory
//! encoded tensors. A fixed two-slot completion window spans tensor boundaries and
//! overlaps the next tile with the prior tile's host copy when both slots fit
//! the admitted working set, and otherwise falls back to one slot. The
//! process never requires a complete dense matrix in active memory: tile size is
//! admitted against an explicit byte bound, while final packed storage remains
//! resident for the subsequent model load. The
//! ordinary residency machinery subsequently sees only the final packed tensor
//! geometry.

use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, MemoryWeightStore, StoreError, TensorReadRequest,
    TensorSelection, WeightStoreDiagnostics,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    WeightQuantization,
};
use eredu_runtime::WeightMaterializationReport;

use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use safemlx::{memory, Array, Dtype, Stream};
use safetensors::tensor::Dtype as SafeDtype;

use crate::{
    backend::error::Error,
    backend::runtime::checkpoint::{
        quantization::quantize_tensor,
        recipe::MlxWeightRecipeExt,
        store::{MlxParameterMaterializationContext, WeightMaterialization},
    },
};

const BOUNDED_QUANTIZATION_TILE_BUFFERS: usize = 2;
const BOUNDED_QUANTIZATION_MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_QUANTIZATION_SUBMISSION_ELEMENTS: usize = i32::MAX as usize;

mod layout;
mod pipeline;
mod plan;
mod preflight;

pub use pipeline::BoundedQuantizedWeightStore;
pub use plan::{BoundedQuantizationPlan, BoundedQuantizationTarget};

#[cfg(test)]
use layout::allocator_cache_requires_clear;

#[cfg(test)]
mod tests;
