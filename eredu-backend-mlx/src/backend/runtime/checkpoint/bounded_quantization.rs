//! Bounded load-time quantization into persisted prepared-checkpoint storage.
//!
//! Native conversion uses explicit streams and a two-slot completion window.
//! The neutral checkpoint cache owns bounded encoded writes, content identity,
//! atomic publication, eviction and source leases. Runtime materialization sees
//! ordinary file-backed packed tensors without permanent host payload copies.

use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, StoreError, TensorReadRequest, TensorSelection,
    WeightStoreDiagnostics,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    WeightQuantization,
};
use eredu_runtime::WeightMaterializationReport;

use eredu_checkpoint::prepared_cache::{
    prepared_checkpoint_recipe_key, PreparedCheckpointEntry, PreparedCheckpointWriter,
    PreparedTensorLayout,
};

use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
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
mod tests;
