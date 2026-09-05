//! Reusable MLX realization of architecture-declared key/value state.

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut, Range},
    path::Path,
};

use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheOptions, StateComponentRole, StateTensorDimension, StateTensorOwner,
    StateTensorPresence, StateTensorRole,
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
    error::Exception,
    ops::{
        indexing::{NewAxis, TryIndexOp},
        zeros_dtype,
    },
    Array, Stream,
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

mod key_value;
use key_value::{common_selected_placement, selected_state_layers};
pub use key_value::{MlxKeyValueLayerState, MlxKeyValueState, MlxKeyValueTransactionBranch};

mod pooling;
pub use pooling::{
    MlxPoolingAttentionCache, MlxPoolingAttentionState, MlxPoolingAttentionStateFactory,
};

mod hybrid;
pub use hybrid::{MlxHybridLayerState, MlxHybridState};
