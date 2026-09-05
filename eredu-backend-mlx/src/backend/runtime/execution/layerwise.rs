//! Architecture-independent execution of decoder models from resident layers.
//!
//! Reusable MLX materialization, residency-transfer, sharding, and pipeline
//! quantization capabilities used by the backend-neutral layered runtime.

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
    WeightQuantization,
};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, DenseDiskStreamReport, DenseStreamTelemetry, DenseTransferSchedule,
    RealtimeMaterializationTask, ReplicatedTextMaterializationTask, WeightBinding,
    WeightLoweringKind, DENSE_TRANSFER_WINDOW,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use safemlx::{Dtype, Stream};

#[cfg(test)]
use crate::backend::runtime::checkpoint::binding::build_module_bindings;
use crate::{
    backend::error::Error,
    backend::runtime::checkpoint::binding::ModuleBindingError,
    backend::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::runtime::residency::dense_stream::BackgroundLayerPrefetch,
    backend::runtime::residency::manager::{
        ResidencyError, ResidencyManager, ResidentTransfer, ResidentUnitLease,
    },
};
use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitId, ResidencyLedgerError};

use eredu_nn::{LinearCompanionRole, ParameterMetadata, ParameterVisitor, Parameterized};
use eredu_runtime::WeightMaterializationReport;

mod dense_stream;
pub use dense_stream::{
    open_safetensors_weight_store, DensePreparedTransfer, DenseStreamController,
    DenseStreamForwardGuard, DenseStreamGroupGuard, DenseTransferWindow,
};

mod quantization;
#[allow(unused_imports)]
pub(crate) use quantization::{packed_weight_companions, PackedWeightCompanions};
pub use quantization::{
    quantize_exact_realtime_tasks, quantize_exact_replicated_text_tasks,
    quantize_module_store_with_bindings,
};
mod sharding;
pub use sharding::{shard_addressable_member_bindings, shard_layer_bindings};
mod validation;
pub use validation::{
    validate_device_budget, validate_host_budget, validate_unused, LayerwiseModelError,
};
