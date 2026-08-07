//! Language-model loading and generation utilities built on `safemlx`.
//!
//! [`api`] provides high-level loading and generation. Model-family
//! implementations live under [`architectures`], reusable neural components
//! under [`nn`], and architecture-independent execution infrastructure under
//! [`runtime`].

#![warn(missing_docs)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Model-family implementations and architecture-specific adapters.
pub mod architectures;
/// Error types returned by the language-model runtime.
pub mod error;
/// Architecture-neutral neural-network building blocks.
pub mod nn;
/// Architecture-independent model execution infrastructure.
pub mod runtime;
#[cfg(test)]
mod test_utils;

pub use api::realtime::{
    load_model as load_realtime_model, load_model_with_options as load_realtime_model_with_options,
    LoadedRealtimeModel, RealtimeCompletedStep, RealtimeInferenceScheduler, RealtimeModelKind,
    RealtimeSampling, RealtimeSession, RealtimeSpeechConfig, RealtimeStepInput, RealtimeStepOutput,
};
pub use api::{
    inspect_model, ArtifactKind, ArtifactModality, ArtifactTensorEncoding, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionOptions, ModelInspectionReport, ModelLoadOptions,
};
pub use architectures::llama::layerwise::{
    load_llama_model, LlamaCache, LlamaLoadOptions, LlamaModel,
};
pub use runtime::attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use runtime::cache::residency::{
    inspect_prompt_cache, CacheBlockId, CacheBlockLifecycle, CacheLayerResidencyReport,
    CacheLayerResidencyStats, CacheRankIdentity, CacheRepresentation, CacheResidencyError,
    CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, CacheTier, LayerCachePolicy,
    LiveCacheDiskPolicy, PagedCacheOptions, PromptCacheBlock, PromptCacheDescriptor,
    PromptCacheManifest, PromptCacheOptions, PromptCacheStateTensor, PromptCacheTopology,
    StateTensorDimension, StateTensorDtype, StateTensorOwner, StateTensorPolicy, StateTensorRole,
    CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
};
pub use runtime::cache::PagedKeyValueCache;
pub use runtime::distributed::cartesian::CartesianExecution;
pub use runtime::distributed::parallel::{
    sample_and_synchronize, LocalModelLayout, LocalTensorLayout, MemberSharding,
    ParallelBuildContext, ParallelExecutionContext, ParallelPlanBuilder, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole, ShardingPolicy, SynchronizedToken,
};
pub use runtime::distributed::topology::{
    DeviceAssignment, ParallelAxis, ParallelCommunicators, ParallelCoordinates, ParallelTopology,
    PlacementPlan, RankPartition, SubgroupMembership, TensorPlacement, TopologyPreflightReport,
};
pub use runtime::execution::layerwise::{
    load_layerwise_model, ArchitectureAdapter, DenseCacheMetrics, DenseDiskStreamReport,
    DenseExecutionGroupReport, DensePassReport, DenseTierResidencyReport, ExecutionResidency,
    ExpertWeightResidency, LayerWeightResidency, LayerwiseForwardState, LayerwiseLoadOptions,
    LayerwiseModel, LayerwiseModelMetadata, NonExpertWeightResidency, ParallelModelInfo,
    SharedWeightStore, WeightResidency,
};
pub use runtime::residency::dense_stream::{
    BackgroundPrefetchReport, DenseDiskStreamLoadOptions, DenseStreamError,
};
pub use runtime::residency::expert_cache::{ExpertCacheLoadOptions, ExpertCacheReport};

use safemlx::Array;

use crate::architectures::qwen::dense as resident_dense_qwen;

/// Builder passed to [`ModelInput`] implementations during generic generation.
pub struct ModelInputBuilder<'a, C, T> {
    /// Token ids or prompt ids for the current model step.
    pub y: &'a Array,
    /// Mutable per-layer cache used by the model implementation.
    pub cache: &'a mut Vec<Option<C>>,
    /// Caller-owned generation state carried across steps.
    pub state: &'a mut T,
}

/// Converts generic generation state into a model-specific input value.
pub trait ModelInput<'a, C, T> {
    /// Builds the concrete model input expected by a [`safemlx::module::Module`].
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

impl<'a, C> ModelInput<'a, C, Option<Array>> for resident_dense_qwen::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

/// Output type that exposes logits for token sampling.
pub trait ModelOutput {
    /// Returns the logits tensor for the current generation step.
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
