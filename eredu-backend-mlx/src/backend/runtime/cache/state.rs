//! Reusable MLX realization of architecture-declared key/value state.

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut, Range},
    path::Path,
};

#[cfg(test)]
use eredu_core::cache::StateTensorDimension;
use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheOptions, StateComponentRole, StateTensorOwner, StateTensorRole,
};
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_nn::{
    AttentionCache, AttentionRequest, CompressedAttentionBlock, CompressedAttentionCache,
    CompressedAttentionScan, CompressedAttentionState, CompressedAttentionView,
    Error as ComputeError, PoolingAttentionCache, PoolingOverlap, PoolingWindows,
};
use eredu_runtime::{
    CacheResidencyReport, DeviceState, LayerRuntimeState, ResettableRuntimeLayerState,
    ResettableRuntimeState, RuntimeLayerState, RuntimeState, RuntimeStateComponents,
    SelectedStateComponentRealization, SelectedStateRealization, StateComponentPlacement,
    StateError, StateLayout, StateSegmentId, StateSegmentSpec,
};
use safemlx::{
    Array, Stream,
    error::Exception,
    ops::{
        indexing::{NewAxis, TryIndexOp},
        zeros_dtype,
    },
};

use crate::backend::{
    nn::shared::MlxNeuralBackend,
    runtime::cache::{
        kv::{
            CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, LiveKeyValueCache,
            PagedKeyValueCache, PagedKeyValueTransactionCheckpoint, PoolingCache,
            PoolingCacheState, RetainedArrayIter,
        },
        residency::{CacheResidencyManager, LoadedPromptCacheStateTensor, PromptCacheStateArray},
    },
};
use eredu_core::cache::CacheRankIdentity;
use ref_cast::RefCast;

use crate::MlxTensor;

type RetainedArrayVecIter<'a> =
    std::iter::Map<std::vec::IntoIter<&'a Array>, fn(&'a Array) -> &'a MlxTensor>;

fn retained_tensor(array: &Array) -> &MlxTensor {
    MlxTensor::ref_cast(array)
}

/// Includes every selected manager, since restored layers can retain distinct
/// physical owners. Native and host aliases are deduplicated by the inventory.
pub(crate) fn retained_state_storage<'a>(
    arrays: impl IntoIterator<Item = &'a Array>,
    managers: impl IntoIterator<Item = &'a CacheResidencyManager>,
) -> Result<
    crate::backend::runtime::residency::storage::RetainedStorage,
    crate::backend::runtime::residency::manager::ResidencyError,
> {
    let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
    for array in arrays {
        storage.include_array(array)?;
    }
    let mut inspected = std::collections::BTreeSet::new();
    for manager in managers {
        if inspected.insert(manager.session_id()) {
            storage.merge(manager.retained_storage()?)?;
        }
    }
    Ok(storage)
}

mod slot_bounds;
pub(crate) mod snapshot_estimate;
pub(crate) use slot_bounds::NativeStateSlotCounts;

mod paged_reset;
mod key_value;
pub(crate) use key_value::{
    RealtimeKvBranchPlan,
    CompleteStateProjectionFailure, PreparedResidentKvCopy, PublishedDenseResidentKvState,
    ResidentKvCopyError, ResidentKvPreparationError, SavedResidentKvCopy,
};
pub use key_value::{MlxKeyValueLayerState, MlxKeyValueState, MlxKeyValueTransactionBranch};
use key_value::{common_selected_placement, selected_state_layers};

mod pooling;
pub use pooling::{
    MlxPoolingAttentionCache, MlxPoolingAttentionState, MlxPoolingAttentionStateFactory,
};
pub(crate) use pooling::{PreparedPoolingAttentionCopy, ResidentPoolingPreparationError};

mod hybrid;
pub use hybrid::{MlxHybridLayerState, MlxHybridState};

mod resident_copy;
pub(crate) use resident_copy::ResidentDecoderPreparationError;
pub(crate) use resident_copy::{
    InitializedResidentDecoderCopy, PreparedResidentDecoderCopy, PreparedResidentDenseCopy,
    PublishedResidentDecoderState, SavedResidentDecoderCopy,
};

pub(crate) use resident_copy::OriginalResidentState;

pub(crate) use resident_copy::{
    SnapshotArraySources, SnapshotOperand, SnapshotProjectionCause, SnapshotProjectionPlan,
};

pub(crate) use resident_copy::{
    CompletedResidentSource, OriginalResidentSourceBinding, bind_completed_resident_source_priors,
    bind_completed_resident_sources, bind_unselected_completed_resident_source_priors,
    copy_completed_resident_state, copy_original_resident_state,
};
