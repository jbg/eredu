//! Bounded load-time weight quantization into resident memory.

//!
//! A [`QuantizedCheckpoint`](crate::backend::runtime::checkpoint::bounded_quantization::QuantizedCheckpoint)
//! owns a neutral source overlaying packed tensors on an existing
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
    CheckpointSource, MemoryWeightStore, TensorSelection,
};
#[cfg(test)]
use eredu_checkpoint::store::{CheckpointLease, StoreError, TensorReadRequest, WeightStoreDiagnostics};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    WeightQuantization,
};
use eredu_runtime::WeightMaterializationReport;

use std::{
    collections::BTreeSet,
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
mod handoff;
mod pipeline;
mod plan;
mod preflight;
mod preparation;
mod workspace;
pub(crate) use workspace::cpu_quantization_temporary_row_bytes;

pub use pipeline::QuantizedCheckpoint;
pub(crate) use pipeline::submit_original_affine_tile;
pub(crate) use handoff::ConvertedQuantization;
pub(crate) use preparation::{ColdQuantization, PreparedQuantization};
pub use plan::{BoundedQuantizationPlan, BoundedQuantizationTarget};

#[cfg(test)]
use layout::allocator_cache_requires_clear;

#[cfg(test)]
mod tests;

pub(crate) use pipeline::{ColdConversion, CpuTileResources};
