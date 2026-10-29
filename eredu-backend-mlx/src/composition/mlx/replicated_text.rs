//! MLX mechanisms and generic binding for replicated text composition.

use std::{collections::BTreeMap, marker::PhantomData, path::Path, sync::Arc};

use eredu_checkpoint::{store::CheckpointSource, LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
};
use eredu_nn::{NeuralBackend, Parameterized, PoolingAttentionCache};
use eredu_runtime::{
    BackendMechanismCapabilities, CacheResidencyPolicy, CacheResidencyReport,
    DenseDiskStreamReport, GroupedOperationRequirement, LayerRuntimeState, LayeredArchitecture,
    ReplicatedTextArchitecture, ReplicatedTextMaterializationTask, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ReplicatedTextSession, ReplicatedTextSessionMechanisms,
    ResidencyReport, RuntimeState, SelectedReplicatedTextRealization, SelectedStateRealization,
    StateComponentPlacement, TransactionalPromptCacheMechanisms, WeightBinding,
    WeightLoweringDescriptor, WeightResidencyMechanism,
};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        nn::tensor::{active_token_validation_arrays, validate_active_token_validations},
        runtime::{
            cache::{
                residency::{
                    load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
                },
                state::{
                    MlxHybridState, MlxKeyValueState, MlxPoolingAttentionCache,
                    MlxPoolingAttentionState, MlxPoolingAttentionStateFactory,
                },
            },
            execution::generic::{MlxLayerwisePolicy, MlxSelectiveUnitPopulator},
            media::input,
        },
    },
    native_quantization::NativeQuantizationFormat,
    MlxTensor,
};

use crate::backend::runtime::execution::{
    generic::prepare_layerwise_policy_with_supplementary_bindings,
    layerwise::materialize_replicated_text_transform_groups,
};
use crate::backend::{
    nn::shared::neutral_parameter_refs,
    runtime::checkpoint::binding::build_mlx_exact_replicated_text_bindings,
};

use super::MlxModelInput;
use crate::composition::mlx::{distributed, prepared_speculative};
use eredu_architectures::composite_execution::{
    CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation, PreparedCompositeInput,
};
use eredu_architectures::replicated_text::{
    CompositeTextArchitectureVisitor, PreparedCompositeTextArchitecture,
    PreparedReplicatedTextArchitecture, PreparedRoutedCompositeTextArchitecture,
    ReplicatedTextArchitectureVisitor, ReplicatedTextStateProfiles,
};

mod capability;
mod lowering;
mod partitioned;
mod prediction;
mod session;
mod state;

pub(crate) use capability::state_capabilities;
pub(super) use capability::*;
pub(super) use lowering::*;
use partitioned::*;
pub(crate) use prediction::ErasedPrefillCursor;
pub(super) use prediction::*;
pub(super) use session::binding::*;
use session::*;
pub(super) use state::*;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod reference_acceptance;
