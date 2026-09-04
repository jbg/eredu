//! Architecture-owned admission for replicated text execution.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use eredu_checkpoint::{
    AffineQuantization, LinearFormat, SourceTensorEncoding, StoredDtype, WeightQuantization,
};
use eredu_core::{
    cache::PromptCacheModelIdentity,
    checkpoint::{TensorCatalog, TensorDtype},
    ArtifactInspection, InputModalities, InputModality,
};
use eredu_nn::{AttentionCache, NeuralBackend, NeuralOperatorCapabilities, Tensor};
use eredu_runtime::{
    LayerRuntimeState, ParameterTransformConstraint, ReplicatedTextArchitecture,
    ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    ReplicatedTextParameterRequirement, ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
    ReplicatedTextRequirements, ReplicatedTextStateAccess, RuntimeState,
    SelectedReplicatedTextRealization,
};

use crate::{
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    processor_plan::ArtifactArchitecturePlan,
    replicated_model::{
        AttentionState, CompressedReplicatedFamily, CompressedReplicatedModel,
        FixedReplicatedFamily, FixedReplicatedModel, FixedState, MixedState,
        ReplicatedForwardContext, Stateless,
    },
    GgufArchitecture,
};

struct Lfm2Replicated;
struct KimiLinearReplicated;
struct NemotronHReplicated;
struct QwenHybridReplicated;

#[derive(Debug)]
struct InspectionCheckpointSource {
    metadata: BTreeMap<String, eredu_checkpoint::store::TensorMetadata>,
    backend: eredu_checkpoint::store::WeightStoreBackend,
}

impl eredu_checkpoint::store::CheckpointSource for InspectionCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.metadata.get(key).cloned().ok_or_else(|| {
            eredu_checkpoint::store::StoreError::UnknownTensor {
                key: key.to_owned(),
            }
        })
    }

    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        Err(eredu_checkpoint::store::StoreError::UnknownTensor { key: request.key })
    }

    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        Ok(eredu_checkpoint::store::WeightStoreDiagnostics {
            backend: self.backend,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: Vec::new(),
            payload_shard_paths: Vec::new(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        })
    }
}

impl<B: NeuralBackend> FixedReplicatedFamily<B> for Lfm2Replicated {
    type Config = crate::lfm2::ModelArgs;
    type Unit = crate::lfm2::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config.has_sparse_moe_layers() {
            return Err(eredu_nn::Error::backend(
                "replicated LFM2 configuration contains routed layers",
            ));
        }
        Ok(())
    }
    fn layer_count(config: &Self::Config) -> Result<usize, eredu_nn::Error> {
        usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        let embedding = "model.embed_tokens.weight";
        crate::decoder::StaticModuleSpec {
            embedding_weight: embedding.into(),
            normalization_weight: "model.embedding_norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.norm_eps,
            normalization_offset: 0.0,
            embedding_quantization: config.weight_quantization_for(embedding),
            head_format: config.weight_quantization_for("lm_head.weight").into(),
            tied_head: config.tie_word_embeddings,
        }
    }
    fn state_layout(config: &Self::Config) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        crate::lfm2::state_layout(config).map_err(eredu_nn::Error::backend)
    }
    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::lfm2::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::lfm2::block::ReplicatedBlock::new(config, index, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config.layer_schedule.iter().position(|policy| {
            matches!(
                policy.operator,
                crate::lfm2::OperatorPolicy::SelfAttention(_)
            )
        })
    }
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        unit.forward(hidden, forward.mask.as_ref(), state, context)
    }
}

impl<B: eredu_nn::BlockwiseAttentionBackend> CompressedReplicatedFamily<B>
    for KimiLinearReplicated
{
    type Config = crate::kimi_linear::ModelArgs;
    type Unit = crate::kimi_linear::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config
            .layer_schedule
            .iter()
            .any(|policy| policy.feed_forward != crate::kimi_linear::FeedForwardPolicy::Dense)
        {
            return Err(eredu_nn::Error::backend(
                "replicated Kimi Linear configuration contains routed layers",
            ));
        }
        Ok(())
    }
    fn layer_count(config: &Self::Config) -> Result<usize, eredu_nn::Error> {
        usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        let embedding = "model.embed_tokens.weight";
        crate::decoder::StaticModuleSpec {
            embedding_weight: embedding.into(),
            normalization_weight: "model.norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.rms_norm_eps,
            normalization_offset: 0.0,
            embedding_quantization: config.weight_quantization_for(embedding),
            head_format: config.weight_quantization_for("lm_head.weight").into(),
            tied_head: config.tie_word_embeddings,
        }
    }
    fn state_layout(config: &Self::Config) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        crate::kimi_linear::state_layout(config).map_err(eredu_nn::Error::backend)
    }
    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::kimi_linear::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::kimi_linear::block::ReplicatedBlock::new(config, index, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config
            .layer_schedule
            .iter()
            .position(|policy| policy.attention == crate::kimi_linear::AttentionKind::Mla)
    }
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        unit.forward(hidden, forward.mask.as_ref(), state, context)
    }
}

impl<B: NeuralBackend> FixedReplicatedFamily<B> for KimiLinearReplicated {
    type Config = crate::kimi_linear::ModelArgs;
    type Unit = crate::kimi_linear::block::KdaReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config.layer_schedule.iter().any(|policy| {
            policy.feed_forward != crate::kimi_linear::FeedForwardPolicy::Dense
                || policy.attention != crate::kimi_linear::AttentionKind::Kda
        }) {
            return Err(eredu_nn::Error::backend(
                "fixed-state Kimi configuration requires dense KDA layers",
            ));
        }
        Ok(())
    }
    fn layer_count(config: &Self::Config) -> Result<usize, eredu_nn::Error> {
        usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        let embedding = "model.embed_tokens.weight";
        crate::decoder::StaticModuleSpec {
            embedding_weight: embedding.into(),
            normalization_weight: "model.norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.rms_norm_eps,
            normalization_offset: 0.0,
            embedding_quantization: config.weight_quantization_for(embedding),
            head_format: config.weight_quantization_for("lm_head.weight").into(),
            tied_head: config.tie_word_embeddings,
        }
    }
    fn state_layout(config: &Self::Config) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        crate::kimi_linear::state_layout(config).map_err(eredu_nn::Error::backend)
    }
    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::kimi_linear::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::kimi_linear::block::KdaReplicatedBlock::new(config, index, context)
    }
    fn mask_layer(_config: &Self::Config) -> Option<usize> {
        None
    }
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        _forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        unit.forward(hidden, state, context)
    }
}

impl<B: NeuralBackend> FixedReplicatedFamily<B> for NemotronHReplicated {
    type Config = crate::nemotron_h::ModelArgs;
    type Unit = crate::nemotron_h::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config.num_nextn_predict_layers != 0
            || config
                .layer_schedule
                .iter()
                .any(|policy| *policy == crate::nemotron_h::LayerPolicy::SparseMoe)
        {
            return Err(eredu_nn::Error::backend(
                "replicated Nemotron-H configuration contains routed or prediction layers",
            ));
        }
        Ok(())
    }
    fn layer_count(config: &Self::Config) -> Result<usize, eredu_nn::Error> {
        usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        let embedding = "model.embeddings.weight";
        crate::decoder::StaticModuleSpec {
            embedding_weight: embedding.into(),
            normalization_weight: "model.norm_f.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.layer_norm_epsilon,
            normalization_offset: 0.0,
            embedding_quantization: config.weight_quantization_for(embedding),
            head_format: config.weight_quantization_for("lm_head.weight").into(),
            tied_head: config.tie_word_embeddings,
        }
    }
    fn state_layout(config: &Self::Config) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        crate::nemotron_h::state_layout(config).map_err(eredu_nn::Error::backend)
    }
    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::nemotron_h::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::nemotron_h::block::ReplicatedBlock::new(config, index, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config
            .layer_schedule
            .iter()
            .position(|policy| matches!(policy, crate::nemotron_h::LayerPolicy::SelfAttention(_)))
    }
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        unit.forward(hidden, forward.mask.as_ref(), state, context)
    }
}

impl<B: NeuralBackend> FixedReplicatedFamily<B> for QwenHybridReplicated {
    type Config = crate::qwen::hybrid::HybridConfig;
    type Unit = crate::qwen::hybrid::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config.is_moe() || config.mtp_num_hidden_layers != 0 {
            return Err(eredu_nn::Error::backend(
                "replicated Qwen hybrid configuration contains routed or prediction layers",
            ));
        }
        Ok(())
    }
    fn layer_count(config: &Self::Config) -> Result<usize, eredu_nn::Error> {
        usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::decoder::StaticModuleSpec {
            embedding_weight: "model.embed_tokens.weight".into(),
            normalization_weight: "model.norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.rms_norm_eps,
            normalization_offset: 1.0,
            embedding_quantization: config.quantization,
            head_format: config.linear_format("lm_head.weight"),
            tied_head: config.tie_word_embeddings,
        }
    }
    fn state_layout(config: &Self::Config) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        crate::qwen::hybrid::state_layout(config).map_err(eredu_nn::Error::backend)
    }
    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::qwen::hybrid::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::qwen::hybrid::block::ReplicatedBlock::new(config, index, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config.layer_schedule.iter().position(|policy| {
            matches!(
                policy,
                crate::qwen::hybrid::HybridLayerPolicy::SelfAttention(_)
            )
        })
    }
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        unit.forward(hidden, forward.mask.as_ref(), state, context)
    }
}

/// Architecture-owned reason that an admitted artifact cannot use replicated text composition.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextIneligibility {
    /// The architecture uses routed computation.
    #[error("architecture requires routed execution")]
    Routed,
    /// The architecture uses heterogeneous recurrent or convolutional state.
    #[error("architecture requires hybrid state execution")]
    HybridState,
    /// The architecture owns composite or multimodal ingress.
    #[error("architecture requires composite input execution")]
    CompositeInput,
    /// The architecture owns embedded prediction groups.
    #[error("architecture requires embedded prediction execution")]
    EmbeddedPrediction,
    /// The architecture uses a frame-oriented realtime lifecycle.
    #[error("architecture requires realtime execution")]
    Realtime,
    /// The architecture is outside the replicated text contract.
    #[error("architecture is not an ordinary replicated text decoder")]
    Unrelated,
}

/// Checked architecture value passed across the architecture/backend boundary.
pub struct PreparedReplicatedTextArchitecture<A> {
    architecture: A,
    source_architecture: Option<A>,
    requirements: ReplicatedTextRequirements,
    contract: eredu_runtime::PreparedReplicatedTextContract,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

impl<A> PreparedReplicatedTextArchitecture<A> {
    /// Returns the exact architecture and artifact requirements.
    pub const fn requirements(&self) -> &ReplicatedTextRequirements {
        &self.requirements
    }

    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        self.contract.selected()
    }

    /// Returns the architecture-derived identity for persistent prompt state.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        self.contract.prompt_cache_identity()
    }

    /// Returns the architecture capability estimate presented by the session.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Returns the normalized model-type label presented by the session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into opaque architecture module ownership.
    pub fn into_modules(self) -> PreparedReplicatedTextModules<A> {
        PreparedReplicatedTextModules {
            architecture: Some(self.architecture),
            source_architecture: self.source_architecture,
            contract: Some(self.contract),
        }
    }
}

/// Opaque ownership of selected-format and optional source-format modules.
pub struct PreparedReplicatedTextModules<A> {
    architecture: Option<A>,
    source_architecture: Option<A>,
    contract: Option<eredu_runtime::PreparedReplicatedTextContract>,
}

impl<A> PreparedReplicatedTextModules<A> {
    /// Takes the selected-format architecture exactly once.
    pub fn take_architecture(&mut self) -> A {
        self.architecture
            .take()
            .expect("prepared architecture was already taken")
    }

    /// Takes the source-format architecture used by a selected transform.
    pub fn take_source_architecture(&mut self) -> Option<A> {
        self.source_architecture.take()
    }

    /// Takes the validated neutral construction contract exactly once.
    pub fn take_contract(&mut self) -> eredu_runtime::PreparedReplicatedTextContract {
        self.contract
            .take()
            .expect("prepared construction contract was already taken")
    }
}

/// Backend-generic visitor for one admitted replicated text architecture.
pub trait ReplicatedTextArchitectureVisitor<B, S>: Sized
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Visitor output.
    type Output;
    /// Backend binding failure.
    type Error;
    /// Records that validated dispatch is about to invoke architecture constructors.
    fn construction_started(&mut self);
    /// Binds one opaque selected-format-aware architecture to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Failure before or during the checked architecture/backend handoff.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextDispatchError<E> {
    /// The normalized architecture belongs to a different execution class.
    #[error(transparent)]
    Ineligible(#[from] ReplicatedTextIneligibility),
    /// Selected formats could not construct the normalized architecture.
    #[error("replicated text architecture construction failed: {0}")]
    Architecture(String),
    /// The backend visitor rejected the checked architecture.
    #[error("replicated text backend binding failed: {0}")]
    Backend(E),
}

#[allow(clippy::too_many_arguments)]
fn prepare_architecture_handoff<B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    requirements: ReplicatedTextRequirements,
    selected: SelectedReplicatedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextArchitecture<A>, String>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    prepare_architecture_handoff_with_addressable::<B, S, A>(
        architecture,
        source_architecture,
        requirements,
        selected,
        capability_estimate,
        effective_model_type,
        prompt_cache_architecture_identity,
        std::iter::empty::<&str>(),
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_architecture_handoff_with_addressable<'a, B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    requirements: ReplicatedTextRequirements,
    selected: SelectedReplicatedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextArchitecture<A>, String>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let contract =
        eredu_runtime::prepare_layered_text_contract_with_addressable_parameters::<A, B, S>(
            &architecture,
            source_architecture.as_ref(),
            selected,
            &prompt_cache_architecture_identity,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            addressable_parameters,
            context,
        )?;
    Ok(PreparedReplicatedTextArchitecture {
        architecture,
        source_architecture,
        requirements,
        contract,
        capability_estimate,
        effective_model_type,
    })
}

/// Constructs and visits the architecture using one authoritative realization.
pub fn visit_replicated_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    let requirements = selected.requirements().clone();
    let eligible = ordinary_eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref())
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    match eligible {
        EligibleConfig::Llama(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::llama(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| crate::llama::LayeredModel::<B>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_llama_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::llama::prompt_cache_architecture_fingerprint(&args);
            let architecture = crate::llama::LayeredModel::<B>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::Qwen(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::qwen(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| crate::qwen::LayeredModel::<B>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::prompt_cache_architecture_fingerprint(&args);
            let architecture = crate::qwen::LayeredModel::<B>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::Lfm2(_)
        | EligibleConfig::KimiLinear(_)
        | EligibleConfig::NemotronH(_)
        | EligibleConfig::QwenHybrid(_)
        | EligibleConfig::GptOss(_)
        | EligibleConfig::DeepSeekV3(_)
        | EligibleConfig::DeepSeekV4(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::MuseGlimmer(_)
        | EligibleConfig::QwenVl(_)
        | EligibleConfig::QwenCompositeHybrid(_) => {
            unreachable!("ordinary replicated eligibility returned heterogeneous state")
        }
    }
}

pub(crate) fn selected_uses_transform(selected: &SelectedReplicatedTextRealization) -> bool {
    selected.parameters().iter().any(|parameter| {
        matches!(
            parameter.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        )
    })
}

fn validate_plan_identity(
    requirements: &ReplicatedTextRequirements,
    config: &EligibleConfig<'_>,
) -> Result<(), String> {
    if requirements.architecture_identity() != config.architecture_identity() {
        return Err("selected realization belongs to a different normalized architecture".into());
    }
    let graph = eredu_runtime::ExecutionGraph::chain([config.execution_group()])
        .map_err(|error| error.to_string())?;
    let units = eredu_runtime::ExecutionUnitLayout::new(&graph, [config.unit_count()?])
        .map_err(|error| error.to_string())?;
    let state_layout = config.state_layout()?;
    if requirements.operators() != config.operators()
        || requirements.execution_graph() != &graph
        || requirements.execution_units() != &units
        || requirements.group_transports() != [config.group_transport()]
        || requirements.state_layout() != &state_layout
        || requirements.state_access() != config.state_access()
    {
        return Err(
            "selected realization structure differs from the normalized architecture".into(),
        );
    }
    Ok(())
}

pub(crate) fn validate_store_handoff(
    requirements: &ReplicatedTextRequirements,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<(), String> {
    let mut provenance = BTreeMap::new();
    for key in store.source_keys() {
        let source = store.source_provenance(&key).map_err(|error| {
            format!("checkpoint provenance for {key:?} is unavailable: {error}")
        })?;
        let identity = (
            source.physical_tensor.clone(),
            source.backing_shard.clone(),
            source.output.clone(),
        );
        if provenance
            .insert(identity, (source.catalog_key, source.source_encoding))
            .is_some()
        {
            return Err("checkpoint catalog contains ambiguous physical output provenance".into());
        }
    }
    let mut checked = BTreeSet::new();
    for parameter in requirements.parameters() {
        let derived = matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::Derived { .. }
        );
        let mut lowering_key = None;
        let mut admitted_keys = BTreeSet::new();
        for source in parameter.physical_sources() {
            let identity = (
                source.tensor().to_owned(),
                Some(source.shard().to_path_buf()),
                source.output().to_owned(),
            );
            let (key, encoding) = provenance.get(&identity).ok_or_else(|| {
                format!(
                    "selected physical output {:?}/{:?} in {:?} is unavailable",
                    source.tensor(),
                    source.output(),
                    source.shard()
                )
            })?;
            if !derived
                && parameter
                    .source_encoding()
                    .is_some_and(|expected| expected != encoding)
            {
                return Err(format!(
                    "selected physical encoding for {:?} differs from the admitted store",
                    parameter.name()
                ));
            }
            lowering_key.get_or_insert(key);
            admitted_keys.insert(key.as_str());
            if checked.insert(identity) {
                store.source_metadata(key).map_err(|error| {
                    format!("selected source catalog key {key:?} is unavailable: {error}")
                })?;
            }
        }
        let selected_keys = parameter
            .sources()
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if parameter.has_lowering_source() && selected_keys != admitted_keys {
            return Err(format!(
                "selected source catalog for {:?} differs from admitted physical provenance",
                parameter.name()
            ));
        }
        if !derived {
            if let Some(physical_shape) = parameter.physical_shape() {
                let key = lowering_key.ok_or_else(|| {
                    format!(
                        "selected physical facts for {:?} have no provenance",
                        parameter.name()
                    )
                })?;
                let metadata = store
                    .source_metadata(key)
                    .map_err(|error| error.to_string())?;
                if physical_shape != metadata.physical_shape.as_slice() {
                    return Err(format!(
                        "selected physical geometry for {:?} differs from the admitted store",
                        parameter.name()
                    ));
                }
            }
        }
    }
    for (target, recipe) in requirements.derived_recipes() {
        let parameter = requirements
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == target)
            .ok_or_else(|| format!("selected recipe target {target:?} is undeclared"))?;
        if !matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::Derived { .. }
        ) {
            return Err(format!(
                "selected recipe target {target:?} is not declared as derived"
            ));
        }
        let inferred = recipe
            .infer(store)
            .map_err(|error| format!("selected recipe for {target:?} is invalid: {error}"))?;
        let admitted_output = requirements
            .derived_recipe_outputs()
            .get(target)
            .expect("validated recipe contract has one output per target");
        if &inferred != admitted_output {
            return Err(format!(
                "selected recipe output for {target:?} differs from admitted metadata"
            ));
        }
        let admitted = parameter
            .physical_sources()
            .iter()
            .map(|source| {
                (
                    source.tensor().to_owned(),
                    Some(source.shard().to_path_buf()),
                    source.output().to_owned(),
                )
            })
            .collect::<BTreeSet<_>>();
        let mut consumed = BTreeSet::new();
        for source_key in recipe.source_keys() {
            let source = store.source_provenance(source_key).map_err(|error| {
                format!("selected recipe source {source_key:?} is unavailable: {error}")
            })?;
            consumed.insert((source.physical_tensor, source.backing_shard, source.output));
        }
        if consumed != admitted {
            return Err(format!(
                "selected recipe sources for {target:?} differ from admitted provenance"
            ));
        }
    }
    Ok(())
}

pub(crate) fn selected_formats(
    selected: &SelectedReplicatedTextRealization,
) -> HashMap<String, WeightQuantization> {
    selected
        .parameters()
        .iter()
        .filter_map(|parameter| {
            parameter
                .executable()
                .weight_quantization()
                .map(|format| (parameter.name().to_owned(), format))
        })
        .collect()
}

fn requirement_formats(
    requirements: &ReplicatedTextRequirements,
) -> HashMap<String, WeightQuantization> {
    requirements
        .parameters()
        .iter()
        .filter_map(|parameter| {
            parameter
                .native_executable()
                .weight_quantization()
                .map(|format| (parameter.name().to_owned(), format))
        })
        .collect()
}

pub(crate) fn selected_linear_formats(
    requirements: &ReplicatedTextRequirements,
    selected: &SelectedReplicatedTextRealization,
) -> HashMap<String, LinearFormat> {
    let linear_weights = requirements
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.role(), ReplicatedTextParameterRole::LinearWeight))
        .map(ReplicatedTextParameterRequirement::name)
        .collect::<BTreeSet<_>>();
    selected
        .parameters()
        .iter()
        .filter(|parameter| linear_weights.contains(parameter.name()))
        .map(|parameter| (parameter.name().to_owned(), parameter.executable()))
        .collect()
}

fn requirement_linear_formats(
    requirements: &ReplicatedTextRequirements,
) -> HashMap<String, LinearFormat> {
    requirements
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.role(), ReplicatedTextParameterRole::LinearWeight))
        .map(|parameter| (parameter.name().to_owned(), parameter.native_executable()))
        .collect()
}

pub(crate) fn selected_llama_args(
    args: &crate::llama::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::llama::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::llama::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn source_llama_args(
    args: &crate::llama::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::llama::ModelArgs, String> {
    let formats = requirement_formats(selected.requirements());
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::llama::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_qwen_args(
    args: &crate::qwen::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::qwen::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::qwen::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn source_qwen_args(
    args: &crate::qwen::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::qwen::ModelArgs, String> {
    let formats = requirement_formats(selected.requirements());
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::qwen::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_gpt_oss_args(
    args: &crate::gpt_oss::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::gpt_oss::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::gpt_oss::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_lfm2_args(
    args: &crate::lfm2::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::lfm2::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::lfm2::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_kimi_linear_args(
    args: &crate::kimi_linear::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::kimi_linear::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::kimi_linear::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_nemotron_h_args(
    args: &crate::nemotron_h::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::nemotron_h::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::nemotron_h::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_qwen_hybrid_args(
    args: &crate::qwen::hybrid::HybridConfig,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::qwen::hybrid::HybridConfig, String> {
    let formats = selected_formats(selected);
    let mut target = args.clone();
    if !formats.is_empty() {
        target.linear_formats = formats
            .into_iter()
            .map(|(name, format)| (name, format.into()))
            .collect();
    }
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

pub(crate) fn selected_deepseek_v3_args(
    args: &crate::deepseek::V3Args,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::deepseek::V3Args, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::deepseek::v3_with_checkpoint_formats(
            args,
            formats
                .into_iter()
                .map(|(name, format)| (name, format.into()))
                .collect(),
        )
    }
}

pub(crate) fn selected_deepseek_v4_args(
    args: &crate::deepseek::V4Args,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::deepseek::V4Args, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::deepseek::v4_with_checkpoint_formats(
            args,
            formats
                .into_iter()
                .map(|(name, format)| (name, format.into()))
                .collect(),
        )
    }
}

trait FixedProfile<B, S, F>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    type Model: ReplicatedTextArchitecture<
            B,
            S,
            Error = eredu_nn::Error,
            StaticModules = crate::decoder::StaticModules<B>,
        > + 'static;

    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error>;
}

impl<B, S, F> FixedProfile<B, S, F> for MixedState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    type Model = FixedReplicatedModel<B, F, MixedState>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedModel::new(config, context)
    }
}

impl<B, S, F> FixedProfile<B, S, F> for AttentionState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    F: FixedReplicatedFamily<B>,
{
    type Model = FixedReplicatedModel<B, F, AttentionState>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedModel::new(config, context)
    }
}

impl<B, S, F> FixedProfile<B, S, F> for FixedState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_runtime::RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    type Model = FixedReplicatedModel<B, F, FixedState>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedModel::new(config, context)
    }
}

impl<B, S, F> FixedProfile<B, S, F> for Stateless
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    type Model = FixedReplicatedModel<B, F, Stateless>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedModel::new(config, context)
    }
}

fn visit_fixed_profile<B, S, V, P>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
    P: FixedProfile<B, S, Lfm2Replicated>
        + FixedProfile<B, S, KimiLinearReplicated>
        + FixedProfile<B, S, NemotronHReplicated>
        + FixedProfile<B, S, QwenHybridReplicated>,
{
    let requirements = selected.requirements().clone();
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref())
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    match eligible {
        EligibleConfig::Lfm2(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::lfm2(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| <P as FixedProfile<B, S, Lfm2Replicated>>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_lfm2_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::lfm2::prompt_cache_architecture_fingerprint(&args);
            let architecture = <P as FixedProfile<B, S, Lfm2Replicated>>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::NemotronH(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::nemotron_h(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| <P as FixedProfile<B, S, NemotronHReplicated>>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_nemotron_h_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::nemotron_h::prompt_cache_architecture_fingerprint(&args);
            let architecture = <P as FixedProfile<B, S, NemotronHReplicated>>::new(args, context)
                .map_err(|error| {
                ReplicatedTextDispatchError::Architecture(error.to_string())
            })?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::QwenHybrid(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::qwen_hybrid_text(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    <P as FixedProfile<B, S, QwenHybridReplicated>>::new(args.clone(), context)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_hybrid_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(&args);
            let architecture = <P as FixedProfile<B, S, QwenHybridReplicated>>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::KimiLinear(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::kimi_linear(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    <P as FixedProfile<B, S, KimiLinearReplicated>>::new(args.clone(), context)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_kimi_linear_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::kimi_linear::prompt_cache_architecture_fingerprint(&args);
            let architecture = <P as FixedProfile<B, S, KimiLinearReplicated>>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let prepared = prepare_architecture_handoff::<B, S, _>(
                architecture,
                source_architecture,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
            )
            .map_err(ReplicatedTextDispatchError::Architecture)?;
            visitor
                .visit(prepared, store)
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::Llama(_)
        | EligibleConfig::Qwen(_)
        | EligibleConfig::GptOss(_)
        | EligibleConfig::DeepSeekV3(_)
        | EligibleConfig::DeepSeekV4(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::MuseGlimmer(_)
        | EligibleConfig::QwenVl(_)
        | EligibleConfig::QwenCompositeHybrid(_) => {
            Err(ReplicatedTextIneligibility::Unrelated.into())
        }
    }
}

/// Constructs a replicated architecture using key/value attention and fixed components.
pub fn visit_replicated_fixed_state_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_fixed_profile::<B, S, V, MixedState>(plan, selected, store, context, visitor)
}

/// Constructs a replicated architecture using only key/value attention state.
pub fn visit_replicated_attention_state_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_fixed_profile::<B, S, V, AttentionState>(plan, selected, store, context, visitor)
}

/// Constructs a replicated architecture using only recurrent or convolutional state.
pub fn visit_replicated_component_state_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_runtime::RuntimeStateComponents<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_fixed_profile::<B, S, V, FixedState>(plan, selected, store, context, visitor)
}

/// Constructs a replicated architecture whose execution units retain no token state.
pub fn visit_replicated_stateless_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_fixed_profile::<B, S, V, Stateless>(plan, selected, store, context, visitor)
}

trait CompressedProfile<B, S>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
{
    type Model: ReplicatedTextArchitecture<
            B,
            S,
            Error = eredu_nn::Error,
            StaticModules = crate::decoder::StaticModules<B>,
        > + 'static;
    fn new(
        config: crate::kimi_linear::ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error>;
}

impl<B, S> CompressedProfile<B, S> for crate::replicated_model::MixedCompressedState
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Model = CompressedReplicatedModel<B, KimiLinearReplicated>;
    fn new(
        config: crate::kimi_linear::ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedModel::new(config, context)
    }
}

impl<B, S> CompressedProfile<B, S> for crate::replicated_model::CompressedState
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Model = CompressedReplicatedModel<
        B,
        KimiLinearReplicated,
        crate::replicated_model::CompressedState,
    >;
    fn new(
        config: crate::kimi_linear::ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedModel::new(config, context)
    }
}

fn visit_compressed_profile<B, S, V, P>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
    P: CompressedProfile<B, S>,
{
    let requirements = selected.requirements().clone();
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref())
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let EligibleConfig::KimiLinear(args) = eligible else {
        return Err(ReplicatedTextIneligibility::Unrelated.into());
    };
    visitor.construction_started();
    let capability_estimate = crate::capability::kimi_linear(args)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let effective_model_type = args.model_type.clone();
    let source_architecture = selected_uses_transform(&selected)
        .then(|| <P as CompressedProfile<B, S>>::new(args.clone(), context))
        .transpose()
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let args = selected_kimi_linear_args(args, &selected)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let prompt_cache_architecture_identity =
        crate::kimi_linear::prompt_cache_architecture_fingerprint(&args);
    let architecture = <P as CompressedProfile<B, S>>::new(args, context)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_architecture_handoff::<B, S, _>(
        architecture,
        source_architecture,
        requirements,
        selected,
        capability_estimate,
        effective_model_type,
        prompt_cache_architecture_identity,
        context,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor
        .visit(prepared, store)
        .map_err(ReplicatedTextDispatchError::Backend)
}

/// Constructs a replicated architecture using compressed attention and fixed components.
pub fn visit_replicated_compressed_state_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_compressed_profile::<B, S, V, crate::replicated_model::MixedCompressedState>(
        plan, selected, store, context, visitor,
    )
}

/// Constructs a replicated architecture using only compressed attention state.
pub fn visit_replicated_compressed_only_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    visit_compressed_profile::<B, S, V, crate::replicated_model::CompressedState>(
        plan, selected, store, context, visitor,
    )
}

/// Supplies typed construction visitors while architecture dispatch selects the exact state profile.
///
/// Backends supporting only ordinary key/value text can continue implementing
/// [`ReplicatedTextArchitectureVisitor`] and calling
/// [`visit_replicated_text_architecture`]. A backend supporting every admitted
/// replicated profile implements this additive dispatcher instead of matching
/// semantic state kinds itself.
pub trait ReplicatedTextProfileDispatcher<B>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;
    /// State representation used when no token state is retained.
    type StatelessState: LayerRuntimeState<B>;
    /// State representation used by ordinary key/value attention.
    type AttentionState: LayerRuntimeState<B>;
    /// State representation used by recurrent or convolutional components.
    type ComponentState: LayerRuntimeState<B>;
    /// State representation used by key/value attention with fixed components.
    type AttentionComponentState: LayerRuntimeState<B>;
    /// State representation used by compressed attention.
    type CompressedState: LayerRuntimeState<B>;
    /// State representation used by compressed attention with fixed components.
    type CompressedComponentState: LayerRuntimeState<B>;

    /// Visitor for a stateless architecture.
    type StatelessVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::StatelessState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for an ordinary key/value architecture.
    type AttentionVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for a fixed-component architecture.
    type ComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::ComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for key/value attention with fixed components.
    type AttentionComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for compressed attention.
    type CompressedVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::CompressedState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for compressed attention with fixed components.
    type CompressedComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::CompressedComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the adapter into its stateless visitor.
    fn into_stateless_visitor(self) -> Self::StatelessVisitor;
    /// Consumes the adapter into its ordinary attention visitor.
    fn into_attention_visitor(self) -> Self::AttentionVisitor;
    /// Consumes the adapter into its fixed-component visitor.
    fn into_component_visitor(self) -> Self::ComponentVisitor;
    /// Consumes the adapter into its attention-with-components visitor.
    fn into_attention_component_visitor(self) -> Self::AttentionComponentVisitor;
    /// Consumes the adapter into its compressed-attention visitor.
    fn into_compressed_visitor(self) -> Self::CompressedVisitor;
    /// Consumes the adapter into its compressed-attention-with-components visitor.
    fn into_compressed_component_visitor(self) -> Self::CompressedComponentVisitor;
}

/// Dispatches one selected replicated-text architecture through its exact typed state profile.
pub fn dispatch_replicated_text_architecture<B, D>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    dispatcher: D,
) -> Result<D::Output, ReplicatedTextDispatchError<D::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    D: ReplicatedTextProfileDispatcher<B>,
    <D::AttentionState as LayerRuntimeState<B>>::LayerState: AttentionCache<B::Tensor>,
    <D::ComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_runtime::RuntimeStateComponents<B>,
    <D::AttentionComponentState as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    <D::CompressedState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor>,
    <D::CompressedComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let eligible = eligible_config(plan)?;
    let ordinary = matches!(eligible, EligibleConfig::Llama(_) | EligibleConfig::Qwen(_));
    match (ordinary, selected.state().access()) {
        (true, ReplicatedTextStateAccess::KeyValue) => visit_replicated_text_architecture(
            plan,
            selected,
            store,
            context,
            dispatcher.into_attention_visitor(),
        ),
        (false, ReplicatedTextStateAccess::Stateless) => {
            visit_replicated_stateless_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_stateless_visitor(),
            )
        }
        (false, ReplicatedTextStateAccess::KeyValue) => {
            visit_replicated_attention_state_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_attention_visitor(),
            )
        }
        (false, ReplicatedTextStateAccess::Fixed) => {
            visit_replicated_component_state_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_component_visitor(),
            )
        }
        (false, ReplicatedTextStateAccess::AttentionWithFixed) => {
            visit_replicated_fixed_state_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_attention_component_visitor(),
            )
        }
        (false, ReplicatedTextStateAccess::CompressedAttention) => {
            visit_replicated_compressed_only_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_compressed_visitor(),
            )
        }
        (false, ReplicatedTextStateAccess::CompressedAttentionWithFixed) => {
            visit_replicated_compressed_state_text_architecture(
                plan,
                selected,
                store,
                context,
                dispatcher.into_compressed_component_visitor(),
            )
        }
        (_, profile) => Err(ReplicatedTextDispatchError::Architecture(format!(
            "selected state profile {profile:?} does not match the dispatched architecture"
        ))),
    }
}

enum EligibleConfig<'a> {
    Llama(&'a crate::llama::ModelArgs),
    Qwen(&'a crate::qwen::ModelArgs),
    Lfm2(&'a crate::lfm2::ModelArgs),
    KimiLinear(&'a crate::kimi_linear::ModelArgs),
    NemotronH(&'a crate::nemotron_h::ModelArgs),
    QwenHybrid(&'a crate::qwen::hybrid::HybridConfig),
    GptOss(&'a crate::gpt_oss::ModelArgs),
    DeepSeekV3(&'a crate::deepseek::V3Args),
    DeepSeekV4(&'a crate::deepseek::V4Args),
    Gemma4(&'a crate::gemma4::FamilyConfig),
    Inkling(&'a crate::inkling::ModelArgs),
    MuseGlimmer(&'a crate::muse_glimmer::DecoderConfig),
    QwenVl(&'a crate::qwen::vl::ModelArgs),
    QwenCompositeHybrid(&'a crate::qwen::hybrid::ParsedHybridConfig),
}

impl EligibleConfig<'_> {
    fn partitioned_boundary_schema(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<eredu_runtime::BoundaryWireSchema, String> {
        use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDimension as Dim};

        let standard = |hidden| {
            eredu_runtime::NoAuxiliaryBoundarySchema::new(hidden)
                .wire_schema()
                .map_err(|error| error.to_string())
        };
        match self {
            Self::Llama(args) => standard(args.hidden_size),
            Self::Qwen(args) => standard(args.hidden_size),
            Self::Lfm2(args) => standard(args.hidden_size),
            Self::KimiLinear(args) => standard(args.hidden_size),
            Self::QwenHybrid(args) => standard(args.hidden_size),
            Self::GptOss(args) => standard(args.hidden_size),
            Self::Inkling(args) => standard(args.text_config.hidden_size),
            Self::MuseGlimmer(args) => standard(args.hidden_size),
            Self::NemotronH(args) => crate::nemotron_h::TargetBoundarySchema::from_args(args)
                .wire_schema()
                .map_err(|error| error.to_string()),
            Self::DeepSeekV3(args) => crate::deepseek::v3::TargetBoundarySchema::from_args(args)
                .wire_schema()
                .map_err(|error| error.to_string()),
            Self::DeepSeekV4(args) => crate::deepseek::v4::TargetBoundarySchema::from_args(args)
                .map_err(|error| error.to_string())?
                .wire_schema()
                .map_err(|error| error.to_string()),
            Self::QwenVl(args) => crate::qwen::vl::PipelineBoundarySchema::from_args(args)
                .wire_schema()
                .map_err(|error| error.to_string()),
            Self::Gemma4(args) => {
                let auxiliary =
                    if args.text.hidden_size_per_layer_input > 0 {
                        let width = usize::try_from(args.text.hidden_size_per_layer_input)
                            .map_err(|_| "Gemma 4 per-layer input width is not positive")?;
                        let range = eredu_core::balanced_contiguous_range(
                            width,
                            topology.tensor_parallel_size(),
                            topology.tensor_parallel_rank(),
                            false,
                        )
                        .map_err(|error| error.to_string())?;
                        vec![eredu_runtime::BoundaryTensorSpec::new(
                            "per_layer_input",
                            [
                                Dim::Batch,
                                Dim::Sequence,
                                Dim::Fixed(args.text.num_hidden_layers() as i32),
                                Dim::Fixed(i32::try_from(range.len()).map_err(|_| {
                                    "Gemma 4 local per-layer input width exceeds i32"
                                })?),
                            ],
                            eredu_runtime::BoundaryTensorDtype::Activation,
                        )]
                    } else {
                        Vec::new()
                    };
                eredu_runtime::BoundaryWireSchema::new(
                    "gemma4.text",
                    eredu_runtime::BoundaryTensorSpec::primary_activation(args.text.hidden_size),
                    auxiliary,
                )
                .map_err(|error| error.to_string())
            }
            Self::QwenCompositeHybrid(args) => {
                let deepstack = args
                    .vision
                    .as_ref()
                    .ok_or_else(|| "conditional Qwen composite omitted vision geometry".to_owned())?
                    .deepstack_layer_count();
                eredu_runtime::BoundaryWireSchema::new(
                    "qwen_conditional.decoder",
                    eredu_runtime::BoundaryTensorSpec::primary_activation(args.text.hidden_size),
                    (0..deepstack).map(|index| {
                        eredu_runtime::BoundaryTensorSpec::new(
                            format!("deepstack.{index}"),
                            [Dim::Batch, Dim::Sequence, Dim::Fixed(args.text.hidden_size)],
                            eredu_runtime::BoundaryTensorDtype::Activation,
                        )
                    }),
                )
                .map_err(|error| error.to_string())
            }
        }
    }

    fn derived_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String> {
        let mut recipes = BTreeMap::new();
        let mut extend = |next: BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>| {
            for (target, recipe) in next {
                if let Some(previous) = recipes.insert(target.clone(), recipe.clone()) {
                    if previous != recipe {
                        return Err(format!(
                            "architecture declared conflicting recipes for {target:?}"
                        ));
                    }
                }
            }
            Ok(())
        };
        match self {
            Self::Llama(_) => {}
            Self::Qwen(args) if args.is_moe() => {
                for layer in 0..self.unit_count()? {
                    let expert = crate::qwen::expert_recipes(source, args, layer)?;
                    extend(BTreeMap::from([
                        (expert.target_gate_up, expert.gate_up),
                        (expert.target_down, expert.down),
                    ]))?;
                }
            }
            Self::Qwen(_) => {}
            Self::Lfm2(args) => {
                for layer in 0..self.unit_count()? {
                    extend(crate::lfm2::unit_recipes(source, args, layer)?)?;
                }
            }
            Self::KimiLinear(args) => {
                for layer in 0..self.unit_count()? {
                    extend(crate::kimi_linear::unit_recipes(source, args, layer, true)?)?;
                }
            }
            Self::NemotronH(args) => {
                extend(crate::nemotron_h::static_recipes(source, args, None)?)?;
                for layer in 0..self.unit_count()? {
                    extend(crate::nemotron_h::unit_recipes(
                        source, args, 0, layer, true,
                    )?)?;
                }
            }
            Self::QwenHybrid(args) => {
                extend(crate::qwen::hybrid::static_recipes(source)?)?;
                for layer in 0..self.unit_count()? {
                    extend(crate::qwen::hybrid::unit_recipes(source, args, layer)?)?;
                }
            }
            Self::GptOss(args) => {
                for layer in 0..self.unit_count()? {
                    extend(
                        crate::gpt_oss::expert_recipes(source, args, layer)?
                            .into_outputs()
                            .into_outputs(),
                    )?;
                }
            }
            Self::DeepSeekV3(args) => {
                for layer in 0..self.unit_count()? {
                    extend(crate::deepseek::v3_unit_recipes(source, args, layer, true)?)?;
                }
            }
            Self::DeepSeekV4(args) => {
                for layer in 0..self.unit_count()? {
                    let expert = crate::deepseek::v4_expert_recipes(source, args, layer)?;
                    extend(BTreeMap::from([
                        (expert.target_gate_up, expert.gate_up),
                        (expert.target_down, expert.down),
                    ]))?;
                }
            }
            Self::Gemma4(args) => {
                for ordinal in 0..self.unit_count()? {
                    extend(crate::gemma4::unit_recipes(source, args, ordinal)?)?;
                }
            }
            Self::Inkling(args) => {
                extend(crate::inkling::static_safetensors_recipes(args, source)?)?;
                let counts = [
                    args.vision_config
                        .as_ref()
                        .map_or(0, |vision| vision.num_hidden_layers as usize),
                    0,
                    usize::try_from(args.text_config.num_hidden_layers)
                        .map_err(|_| "invalid Inkling text layer count".to_owned())?,
                ];
                for (group, count) in counts.into_iter().enumerate() {
                    for index in 0..count {
                        extend(crate::inkling::unit_safetensors_recipes(
                            args, source, group, index,
                        )?)?;
                    }
                }
            }
            Self::MuseGlimmer(args) => {
                extend(crate::muse_glimmer::static_safetensors_recipes(
                    args, source,
                )?)?;
                let counts = [
                    args.vision_config
                        .as_ref()
                        .map_or(0, |vision| vision.layer_count()),
                    usize::try_from(args.num_hidden_layers)
                        .map_err(|_| "invalid Muse-Glimmer text layer count".to_owned())?,
                ];
                for (group, count) in counts.into_iter().enumerate() {
                    for index in 0..count {
                        extend(crate::muse_glimmer::unit_safetensors_recipes(
                            args, source, group, index,
                        )?)?;
                    }
                }
            }
            Self::QwenVl(args) => {
                extend(crate::qwen::vl::static_recipes(source))?;
                for ordinal in 0..self.unit_count()? {
                    extend(crate::qwen::vl::unit_recipes(source, args, ordinal)?)?;
                }
            }
            Self::QwenCompositeHybrid(args) => {
                extend(crate::qwen::hybrid::static_recipes(source)?)?;
                for ordinal in 0..self.unit_count()? {
                    extend(crate::qwen::hybrid::conditional_unit_recipes(
                        source, args, ordinal,
                    )?)?;
                }
            }
        }
        Ok(recipes)
    }

    fn architecture_identity(&self) -> String {
        match self {
            Self::Llama(args) => crate::llama::prompt_cache_architecture_fingerprint(args),
            Self::Qwen(args) => crate::qwen::prompt_cache_architecture_fingerprint(args),
            Self::Lfm2(args) => crate::lfm2::prompt_cache_architecture_fingerprint(args),
            Self::KimiLinear(args) => {
                crate::kimi_linear::prompt_cache_architecture_fingerprint(args)
            }
            Self::NemotronH(args) => crate::nemotron_h::prompt_cache_architecture_fingerprint(args),
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(args)
            }
            Self::GptOss(args) => crate::gpt_oss::prompt_cache_architecture_fingerprint(args),
            Self::DeepSeekV3(args) => crate::deepseek::v3_architecture_fingerprint(args),
            Self::DeepSeekV4(args) => crate::deepseek::v4_architecture_fingerprint(args),
            Self::Gemma4(args) => args.architecture_fingerprint(),
            Self::Inkling(args) => args.architecture_fingerprint(),
            Self::MuseGlimmer(args) => args.architecture_fingerprint(),
            Self::QwenVl(args) => crate::qwen::vl::prompt_cache_architecture_fingerprint(args),
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(args)
            }
        }
    }

    fn canonical_parameter_name(&self, name: &str, aliases: &[String]) -> String {
        match self {
            Self::Inkling(_) => aliases.first().cloned().unwrap_or_else(|| name.to_owned()),
            Self::NemotronH(_) => aliases
                .iter()
                .find(|alias| alias.starts_with("model.") || alias.starts_with("lm_head."))
                .cloned()
                .unwrap_or_else(|| name.to_owned()),
            Self::GptOss(_) => name
                .strip_suffix("_blocks")
                .map_or_else(|| name.to_owned(), str::to_owned),
            _ => name.to_owned(),
        }
    }

    fn logical_parameter_shape(&self, name: &str, discovered: Vec<usize>) -> Vec<usize> {
        if let Self::KimiLinear(args) = self {
            if name.ends_with(".self_attn.A_log") {
                return vec![
                    1,
                    1,
                    usize::try_from(args.kda_config.num_heads).unwrap_or_default(),
                    1,
                ];
            }
        }
        if discovered.len() != 2 {
            return discovered;
        }
        match self {
            Self::Lfm2(args) if name.ends_with(".conv.conv.weight") => vec![
                usize::try_from(args.hidden_size).unwrap_or_default(),
                1,
                usize::try_from(args.conv_l_cache).unwrap_or_default(),
            ],
            Self::KimiLinear(args)
                if name.ends_with(".q_conv1d.weight")
                    || name.ends_with(".k_conv1d.weight")
                    || name.ends_with(".v_conv1d.weight") =>
            {
                vec![
                    usize::try_from(args.kda_config.num_heads * args.kda_config.head_dim)
                        .unwrap_or_default(),
                    1,
                    usize::try_from(args.kda_config.short_conv_kernel_size).unwrap_or_default(),
                ]
            }
            Self::NemotronH(args) if name.ends_with(".mamba.conv1d.weight") => vec![
                usize::try_from(
                    args.mamba_num_heads * args.mamba_head_dim
                        + 2 * args.n_groups * args.ssm_state_size,
                )
                .unwrap_or_default(),
                1,
                usize::try_from(args.conv_kernel).unwrap_or_default(),
            ],
            Self::QwenHybrid(_) if name.ends_with(".linear_attn.conv1d.weight") => {
                vec![discovered[0], 1, discovered[1]]
            }
            _ => discovered,
        }
    }

    fn parameter_requires_shape_recipe(&self, name: &str) -> bool {
        match self {
            Self::Lfm2(_) => name.ends_with(".conv.conv.weight"),
            Self::KimiLinear(_) => {
                name.ends_with(".self_attn.A_log")
                    || name.ends_with(".q_conv1d.weight")
                    || name.ends_with(".k_conv1d.weight")
                    || name.ends_with(".v_conv1d.weight")
            }
            Self::NemotronH(_) => name.ends_with(".mamba.conv1d.weight"),
            Self::QwenHybrid(_) => name.ends_with(".linear_attn.conv1d.weight"),
            Self::QwenCompositeHybrid(_) => name.ends_with(".linear_attn.conv1d.weight"),
            Self::Llama(_)
            | Self::Qwen(_)
            | Self::GptOss(_)
            | Self::DeepSeekV3(_)
            | Self::DeepSeekV4(_)
            | Self::Gemma4(_)
            | Self::Inkling(_)
            | Self::MuseGlimmer(_)
            | Self::QwenVl(_) => false,
        }
    }

    fn state_access(&self) -> ReplicatedTextStateAccess {
        let layout = self
            .state_layout()
            .expect("eligible configuration has a valid state layout");
        let roles = (0..layout.len()).flat_map(|layer| {
            layout
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .map(eredu_core::cache::StateComponentPolicy::role)
        });
        let mut ordinary = false;
        let mut compressed = false;
        let mut fixed = false;
        for role in roles {
            match role {
                eredu_core::cache::StateComponentRole::AttentionKeys
                | eredu_core::cache::StateComponentRole::AttentionValues => ordinary = true,
                eredu_core::cache::StateComponentRole::CompressedLatent
                | eredu_core::cache::StateComponentRole::RotaryKeys => compressed = true,
                eredu_core::cache::StateComponentRole::Fixed(_) => fixed = true,
            }
        }
        match (ordinary, compressed, fixed) {
            (false, false, false) => ReplicatedTextStateAccess::Stateless,
            (true, false, false) => ReplicatedTextStateAccess::KeyValue,
            (false, false, true) => ReplicatedTextStateAccess::Fixed,
            (true, false, true) => ReplicatedTextStateAccess::AttentionWithFixed,
            (false, true, false) => ReplicatedTextStateAccess::CompressedAttention,
            (false, true, true) => ReplicatedTextStateAccess::CompressedAttentionWithFixed,
            (true, true, _) => unreachable!("eligible replicated layout mixes attention encodings"),
        }
    }

    fn operators(&self) -> NeuralOperatorCapabilities {
        match self {
            Self::Llama(_) | Self::Qwen(_) | Self::Lfm2(_) | Self::GptOss(_) => {
                NeuralOperatorCapabilities::NONE
            }
            Self::KimiLinear(args)
                if args
                    .layer_schedule
                    .iter()
                    .any(|policy| policy.attention == crate::kimi_linear::AttentionKind::Kda) =>
            {
                crate::operator_requirements::KIMI_LINEAR
            }
            Self::NemotronH(args)
                if args
                    .layer_schedule
                    .iter()
                    .any(|policy| *policy == crate::nemotron_h::LayerPolicy::Mamba) =>
            {
                crate::operator_requirements::NEMOTRON_H
            }
            Self::QwenHybrid(args)
                if args.layer_schedule.iter().any(|policy| {
                    *policy == crate::qwen::hybrid::HybridLayerPolicy::LinearAttention
                }) =>
            {
                crate::operator_requirements::QWEN_HYBRID
            }
            Self::KimiLinear(_) | Self::NemotronH(_) | Self::QwenHybrid(_) => {
                NeuralOperatorCapabilities::NONE
            }
            Self::DeepSeekV3(_) => crate::operator_requirements::DEEPSEEK_V3,
            Self::DeepSeekV4(_) => crate::operator_requirements::DEEPSEEK_V4,
            Self::Gemma4(_) => crate::operator_requirements::GEMMA4,
            Self::Inkling(_) => crate::operator_requirements::INKLING,
            Self::MuseGlimmer(_) => crate::operator_requirements::MUSE_GLIMMER,
            Self::QwenVl(_) => crate::operator_requirements::QWEN_VL,
            Self::QwenCompositeHybrid(_) => crate::operator_requirements::QWEN_HYBRID
                .union(crate::operator_requirements::QWEN_VISION),
        }
    }

    fn execution_group(&self) -> &'static str {
        match self {
            Self::Llama(_) | Self::Qwen(_) | Self::GptOss(_) => {
                crate::decoder::TEXT_DECODER_EXECUTION_GROUP
            }
            Self::Lfm2(_) | Self::KimiLinear(_) | Self::NemotronH(_) | Self::QwenHybrid(_) => {
                crate::decoder::TARGET_EXECUTION_GROUP
            }
            Self::DeepSeekV3(_) => crate::decoder::TARGET_EXECUTION_GROUP,
            Self::DeepSeekV4(_) => crate::decoder::TARGET_EXECUTION_GROUP,
            Self::Gemma4(_) => crate::gemma4::model::TEXT_EXECUTION_GROUP,
            Self::Inkling(_) => crate::inkling::model::TEXT_EXECUTION_GROUP,
            Self::MuseGlimmer(_) => crate::muse_glimmer::model::TEXT_EXECUTION_GROUP,
            Self::QwenVl(_) => crate::qwen::vl::TEXT_EXECUTION_GROUP,
            Self::QwenCompositeHybrid(_) => crate::qwen::hybrid::VISION_EXECUTION_GROUP,
        }
    }

    fn group_transport(&self) -> eredu_runtime::ArchitectureGroupTransport {
        match self {
            Self::DeepSeekV4(_) => crate::deepseek::v4::target_group_transport(),
            _ => crate::transport::decoder(),
        }
    }

    fn unit_count(&self) -> Result<usize, String> {
        let count = match self {
            Self::Llama(args) => args.num_hidden_layers,
            Self::Qwen(args) => args.num_hidden_layers,
            Self::Lfm2(args) => args.num_hidden_layers,
            Self::KimiLinear(args) => args.num_hidden_layers,
            Self::NemotronH(args) => args.num_hidden_layers,
            Self::QwenHybrid(args) => args.num_hidden_layers,
            Self::GptOss(args) => args.num_hidden_layers,
            Self::DeepSeekV3(args) => args.num_hidden_layers,
            Self::DeepSeekV4(args) => args.num_hidden_layers,
            Self::Gemma4(args) => {
                return [
                    args.vision
                        .as_ref()
                        .map_or(0, |vision| vision.num_hidden_layers as usize),
                    args.audio
                        .as_ref()
                        .map_or(0, |audio| audio.num_hidden_layers as usize),
                    args.text.num_hidden_layers(),
                ]
                .into_iter()
                .try_fold(0usize, |total, count| total.checked_add(count))
                .ok_or_else(|| "Gemma 4 execution unit count overflowed".to_owned());
            }
            Self::Inkling(args) => {
                return args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.num_hidden_layers as usize)
                    .checked_add(
                        usize::try_from(args.text_config.num_hidden_layers)
                            .map_err(|_| "invalid Inkling text layer count".to_owned())?,
                    )
                    .ok_or_else(|| "Inkling execution unit count overflowed".to_owned());
            }
            Self::MuseGlimmer(args) => {
                return args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.layer_count())
                    .checked_add(
                        usize::try_from(args.num_hidden_layers)
                            .map_err(|_| "invalid Muse-Glimmer text layer count".to_owned())?,
                    )
                    .ok_or_else(|| "Muse-Glimmer execution unit count overflowed".to_owned());
            }
            Self::QwenVl(args) => {
                return args
                    .vision
                    .layer_count()
                    .checked_add(args.text.num_hidden_layers as usize)
                    .ok_or_else(|| "Qwen3-VL execution unit count overflowed".to_owned());
            }
            Self::QwenCompositeHybrid(args) => {
                return args
                    .vision
                    .as_ref()
                    .map_or(0, |vision| vision.layer_count())
                    .checked_add(args.text.num_hidden_layers as usize)
                    .ok_or_else(|| "Qwen composite execution unit count overflowed".to_owned());
            }
        };
        usize::try_from(count).map_err(|_| format!("invalid replicated layer count {count}"))
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, String> {
        match self {
            Self::Llama(args) => {
                crate::llama::state_layout(*args).map_err(|error| error.to_string())
            }
            Self::Qwen(args) => crate::qwen::state_layout(*args).map_err(|error| error.to_string()),
            Self::Lfm2(args) => crate::lfm2::state_layout(args).map_err(|error| error.to_string()),
            Self::KimiLinear(args) => {
                crate::kimi_linear::state_layout(args).map_err(|error| error.to_string())
            }
            Self::NemotronH(args) => {
                crate::nemotron_h::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::state_layout(args).map_err(|error| error.to_string())
            }
            Self::GptOss(args) => {
                crate::gpt_oss::state_layout(args).map_err(|error| error.to_string())
            }
            Self::DeepSeekV3(args) => {
                crate::deepseek::v3::state_layout(args).map_err(|error| error.to_string())
            }
            Self::DeepSeekV4(args) => {
                crate::deepseek::v4::state_layout(args).map_err(|error| error.to_string())
            }
            Self::Gemma4(args) => {
                crate::gemma4::state_layout(&args.text).map_err(|error| error.to_string())
            }
            Self::Inkling(args) => {
                let target =
                    crate::inkling::state_layout(args).map_err(|error| error.to_string())?;
                let prediction =
                    crate::inkling::mtp_state_layout(args).map_err(|error| error.to_string())?;
                crate::inkling::composite_state_layout(&target, prediction.as_ref())
                    .map_err(|error| error.to_string())
            }
            Self::MuseGlimmer(args) => {
                crate::muse_glimmer::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenVl(args) => {
                crate::qwen::vl::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::state_layout(&args.text).map_err(|error| error.to_string())
            }
        }
    }

    fn native_format(&self, name: &str) -> LinearFormat {
        match self {
            Self::Llama(args) => args.weight_quantization_for(name),
            Self::Qwen(args) => args.weight_quantization_for(name),
            Self::Lfm2(args) => args.weight_quantization_for(name),
            Self::KimiLinear(args) => args.weight_quantization_for(name),
            Self::NemotronH(args) => args.weight_quantization_for(name),
            Self::QwenHybrid(args) => return args.linear_format(name),
            Self::GptOss(_)
                if name.contains(".mlp.experts.")
                    && (name.ends_with("gate_up_proj") || name.ends_with("down_proj")) =>
            {
                Some(eredu_checkpoint::WeightQuantization::MxFp4)
            }
            Self::GptOss(args) => args.weight_quantization_for(name),
            Self::DeepSeekV3(args) => return args.linear_format_for(name),
            Self::DeepSeekV4(args) => {
                if name.contains(".switch_mlp.") {
                    return match args.expert_format {
                        crate::deepseek::ExpertFormat::Dense => LinearFormat::Dense,
                        crate::deepseek::ExpertFormat::MxFp4 => LinearFormat::MxFp4,
                        crate::deepseek::ExpertFormat::BlockFp8 => args.linear_format,
                    };
                }
                return args.linear_format_for(name);
            }
            Self::Gemma4(args) => {
                if name.starts_with("model.vision_") {
                    return args.vision.as_ref().map_or(LinearFormat::Dense, |config| {
                        configured_linear_format(
                            config.weight_quantization,
                            config.quantized_weights.as_ref(),
                            config.quantized_weight_configs.as_ref(),
                            name,
                        )
                    });
                }
                if name.starts_with("model.audio_") {
                    return args.audio.as_ref().map_or(LinearFormat::Dense, |config| {
                        configured_linear_format(
                            config.weight_quantization,
                            config.quantized_weights.as_ref(),
                            config.quantized_weight_configs.as_ref(),
                            name,
                        )
                    });
                }
                return args.text.linear_format_for(name);
            }
            Self::Inkling(args) => {
                if name.starts_with("visual.") {
                    return args
                        .vision_config
                        .as_ref()
                        .map_or(LinearFormat::Dense, |config| config.linear_format_for(name));
                }
                if name.starts_with("audio.") {
                    return args
                        .audio_config
                        .as_ref()
                        .map_or(LinearFormat::Dense, |config| config.linear_format_for(name));
                }
                return args.text_config.linear_format_for(name);
            }
            Self::MuseGlimmer(args) => {
                if name.starts_with("model.vision_") {
                    return args
                        .vision_config
                        .as_ref()
                        .map_or(LinearFormat::Dense, |vision| vision.linear_format_for(name));
                }
                return args.linear_format_for(name);
            }
            Self::QwenVl(args) => {
                if name.starts_with("model.visual.") {
                    return args
                        .vision
                        .linear_formats
                        .get(name)
                        .copied()
                        .unwrap_or(LinearFormat::Dense);
                }
                return args.text.weight_quantization_for(name).into();
            }
            Self::QwenCompositeHybrid(args) => {
                if name.starts_with("model.visual.") {
                    return args
                        .vision
                        .as_ref()
                        .and_then(|vision| vision.linear_formats.get(name))
                        .copied()
                        .unwrap_or(LinearFormat::Dense);
                }
                return args.text.linear_format(name);
            }
        }
        .map_or(LinearFormat::Dense, LinearFormat::from)
    }

    fn linear_parameter_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>, String> {
        match self {
            Self::Llama(args) => decoder_linear_parameter_shapes(*args),
            Self::Qwen(args) => {
                let mut shapes = decoder_linear_parameter_shapes(*args)?;
                if args.num_experts > 0 {
                    let experts = positive(args.num_experts, "Qwen expert count")?;
                    let hidden = positive(args.hidden_size, "Qwen hidden size")?;
                    let intermediate =
                        positive(args.moe_intermediate_size, "Qwen expert intermediate size")?;
                    let fused = intermediate
                        .checked_mul(2)
                        .ok_or_else(|| "Qwen fused expert width overflowed".to_string())?;
                    for layer in 0..self.unit_count()? {
                        let root = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
                        shapes.insert(format!("{root}.gate_up_proj"), vec![experts, fused, hidden]);
                        shapes.insert(
                            format!("{root}.down_proj"),
                            vec![experts, hidden, intermediate],
                        );
                    }
                }
                Ok(shapes)
            }
            Self::GptOss(args) => {
                let mut shapes = decoder_linear_parameter_shapes(*args)?;
                let experts = positive(args.num_local_experts, "GPT-OSS expert count")?;
                let hidden = positive(args.hidden_size, "GPT-OSS hidden size")?;
                let intermediate = positive(args.intermediate_size, "GPT-OSS intermediate size")?;
                let fused = intermediate
                    .checked_mul(2)
                    .ok_or_else(|| "GPT-OSS fused expert width overflowed".to_string())?;
                for layer in 0..self.unit_count()? {
                    let root = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
                    shapes.insert(format!("{root}.gate_up_proj"), vec![experts, fused, hidden]);
                    shapes.insert(
                        format!("{root}.down_proj"),
                        vec![experts, hidden, intermediate],
                    );
                }
                Ok(shapes)
            }
            Self::Lfm2(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::lfm2::safetensors_plan(args, true).map_err(|error| error.to_string())?,
                    |name| args.weight_quantization_for(name).map(LinearFormat::from),
                    "model.embed_tokens.weight",
                )?;
                if args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::lfm2::replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::KimiLinear(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::kimi_linear::safetensors_plan(args)
                        .map_err(|error| error.to_string())?,
                    |name| args.weight_quantization_for(name).map(LinearFormat::from),
                    "model.embed_tokens.weight",
                )?;
                if args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::kimi_linear::replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::NemotronH(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::nemotron_h::safetensors_plan(args).map_err(|error| error.to_string())?,
                    |name| args.weight_quantization_for(name).map(LinearFormat::from),
                    "model.embeddings.weight",
                )?;
                if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 {
                    insert_grouped_relu2_linear_shapes(
                        &mut shapes,
                        &crate::nemotron_h::replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::QwenHybrid(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::qwen::hybrid::safetensors_plan(args)
                        .map_err(|error| error.to_string())?,
                    |name| Some(args.linear_format(name)),
                    "model.embed_tokens.weight",
                )?;
                if args.num_experts > 0 && args.num_experts_per_tok > 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::qwen::hybrid::replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::DeepSeekV3(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::deepseek::v3_safetensors_plan(args, true)
                        .map_err(|error| error.to_string())?,
                    |name| Some(args.linear_format_for(name)),
                    "model.embed_tokens.weight",
                )?;
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::deepseek::v3_replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::DeepSeekV4(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::deepseek::v4_safetensors_plan(args)
                        .map_err(|error| error.to_string())?,
                    |name| Some(args.linear_format_for(name)),
                    "embed.weight",
                )?;
                if args.num_nextn_predict_layers == 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::deepseek::v4_replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::Gemma4(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::gemma4::safetensors_plan(args).map_err(|error| error.to_string())?,
                    |name| Some(self.native_format(name)),
                    "model.language_model.embed_tokens.weight",
                )?;
                if args.text.num_experts.unwrap_or_default() > 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &gemma4_replicated_expert_realization_plan(args)?,
                    )?;
                }
                Ok(shapes)
            }
            Self::Inkling(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::inkling::safetensors_plan(args).map_err(|error| error.to_string())?,
                    |name| Some(self.native_format(name)),
                    "model.embed_tokens.weight",
                )?;
                if args.text_config.has_sparse_moe_layers() {
                    let (plan, _) = inkling_replicated_expert_realization_plan(args)?;
                    insert_grouped_gated_linear_shapes(&mut shapes, &plan)?;
                }
                Ok(shapes)
            }
            Self::MuseGlimmer(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::muse_glimmer::safetensors_plan(args)
                        .map_err(|error| error.to_string())?,
                    |name| Some(self.native_format(name)),
                    "model.embed_tokens.weight",
                )?;
                if args.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &muse_replicated_expert_realization_plan(args)?,
                    )?;
                }
                Ok(shapes)
            }
            Self::QwenVl(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::qwen::vl::safetensors_plan(args).map_err(|error| error.to_string())?,
                    |name| Some(self.native_format(name)),
                    "model.language_model.embed_tokens.weight",
                )?;
                if args.text.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::qwen::replicated_expert_realization_plan(&args.text)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::QwenCompositeHybrid(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::qwen::hybrid::composite_safetensors_plan(args)
                        .map_err(|error| error.to_string())?,
                    |name| Some(self.native_format(name)),
                    "model.embed_tokens.weight",
                )?;
                if args.text.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::qwen::hybrid::replicated_expert_realization_plan(&args.text)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
        }
    }

    fn parameter_root(&self) -> &str {
        match self {
            Self::Llama(args) => crate::decoder::Config::parameter_root(*args),
            Self::Qwen(args) => crate::decoder::Config::parameter_root(*args),
            Self::Lfm2(_) | Self::KimiLinear(_) | Self::QwenHybrid(_) => "model",
            Self::NemotronH(_) => "model",
            Self::GptOss(args) => crate::decoder::Config::parameter_root(*args),
            Self::DeepSeekV3(_) => "model",
            Self::DeepSeekV4(_) => "",
            Self::Gemma4(_) => "model.language_model",
            Self::Inkling(_) | Self::MuseGlimmer(_) | Self::QwenCompositeHybrid(_) => "model",
            Self::QwenVl(args) => &args.text.parameter_root,
        }
    }

    fn tied_embeddings(&self) -> bool {
        match self {
            Self::Llama(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Qwen(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Lfm2(args) => args.tie_word_embeddings,
            Self::KimiLinear(args) => args.tie_word_embeddings,
            Self::NemotronH(args) => args.tie_word_embeddings,
            Self::QwenHybrid(args) => args.tie_word_embeddings,
            Self::GptOss(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::DeepSeekV3(args) => args.tie_word_embeddings,
            Self::DeepSeekV4(args) => args.tie_word_embeddings,
            Self::Gemma4(args) => args.text.tie_word_embeddings,
            Self::Inkling(_) => false,
            Self::MuseGlimmer(args) => args.tie_word_embeddings,
            Self::QwenVl(args) => args.text.tie_word_embeddings,
            Self::QwenCompositeHybrid(args) => args.text.tie_word_embeddings,
        }
    }

    fn embedding_shape(&self) -> Result<Vec<usize>, String> {
        let (vocabulary, hidden) = match self {
            Self::Llama(args) => (args.vocab_size, args.hidden_size),
            Self::Qwen(args) => (args.vocab_size, args.hidden_size),
            Self::Lfm2(args) => (args.vocab_size, args.hidden_size),
            Self::KimiLinear(args) => (args.vocab_size, args.hidden_size),
            Self::NemotronH(args) => (args.vocab_size, args.hidden_size),
            Self::QwenHybrid(args) => (args.vocab_size, args.hidden_size),
            Self::GptOss(args) => (args.vocab_size, args.hidden_size),
            Self::DeepSeekV3(args) => (args.vocab_size, args.hidden_size),
            Self::DeepSeekV4(args) => (args.vocab_size, args.hidden_size),
            Self::Gemma4(args) => (args.text.vocab_size, args.text.hidden_size),
            Self::Inkling(args) => (args.text_config.vocab_size, args.text_config.hidden_size),
            Self::MuseGlimmer(args) => (args.vocab_size, args.hidden_size),
            Self::QwenVl(args) => (args.text.vocab_size, args.text.hidden_size),
            Self::QwenCompositeHybrid(args) => (args.text.vocab_size, args.text.hidden_size),
        };
        Ok(vec![
            positive(vocabulary, "vocabulary size")?,
            positive(hidden, "hidden size")?,
        ])
    }

    fn parameter_role(
        &self,
        name: &str,
        companion: bool,
        linear_shapes: &BTreeMap<String, Vec<usize>>,
    ) -> ReplicatedTextParameterRole {
        match self {
            Self::Llama(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Qwen(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::GptOss(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Lfm2(_)
            | Self::KimiLinear(_)
            | Self::NemotronH(_)
            | Self::QwenHybrid(_)
            | Self::DeepSeekV3(_)
            | Self::DeepSeekV4(_)
            | Self::Gemma4(_)
            | Self::Inkling(_)
            | Self::MuseGlimmer(_)
            | Self::QwenVl(_)
            | Self::QwenCompositeHybrid(_) => {
                family_parameter_role(self, name, companion, linear_shapes)
            }
        }
    }

    fn embedding_name(&self) -> String {
        match self {
            Self::NemotronH(_) => "model.embeddings.weight".into(),
            Self::Llama(_)
            | Self::Qwen(_)
            | Self::Lfm2(_)
            | Self::KimiLinear(_)
            | Self::QwenHybrid(_) => {
                format!("{}.embed_tokens.weight", self.parameter_root())
            }
            Self::GptOss(_) => format!("{}.embed_tokens.weight", self.parameter_root()),
            Self::DeepSeekV3(_) => "model.embed_tokens.weight".into(),
            Self::DeepSeekV4(_) => "embed.weight".into(),
            Self::Gemma4(_) => "model.language_model.embed_tokens.weight".into(),
            Self::Inkling(_) | Self::MuseGlimmer(_) | Self::QwenCompositeHybrid(_) => {
                "model.embed_tokens.weight".into()
            }
            Self::QwenVl(_) => "model.language_model.embed_tokens.weight".into(),
        }
    }
}

fn family_linear_parameter_shapes(
    plan: eredu_checkpoint::schema::SafetensorsCheckpointPlan,
    native_format: impl Fn(&str) -> Option<LinearFormat>,
    embedding: &str,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    constraints.extend(
        plan.layout_groups
            .iter()
            .flat_map(|group| group.variants.iter())
            .flat_map(|variant| variant.tensors.iter()),
    );
    let mut result = BTreeMap::new();
    for constraint in constraints {
        if constraint.role == eredu_checkpoint::schema::TensorRole::Companion
            || constraint.key == embedding
            || constraint.shape.len() != 2
        {
            continue;
        }
        let mut shape = constraint.shape.clone();
        match native_format(&constraint.key).unwrap_or(LinearFormat::Dense) {
            LinearFormat::Affine(format) => {
                let bits = usize::try_from(format.bits)
                    .map_err(|_| format!("invalid affine bits for {:?}", constraint.key))?;
                let packed = shape.last_mut().expect("rank checked above");
                *packed = packed
                    .checked_mul(32)
                    .and_then(|bits_total| bits_total.checked_div(bits))
                    .ok_or_else(|| format!("invalid affine geometry for {:?}", constraint.key))?;
            }
            LinearFormat::MxFp4 => {
                let packed = shape.last_mut().expect("rank checked above");
                *packed = packed
                    .checked_mul(8)
                    .ok_or_else(|| format!("invalid MXFP4 geometry for {:?}", constraint.key))?;
            }
            _ => {}
        }
        result.entry(constraint.key.clone()).or_insert(shape);
    }
    Ok(result)
}

fn configured_linear_format(
    default: Option<WeightQuantization>,
    selected: Option<&std::collections::HashSet<String>>,
    formats: Option<&HashMap<String, WeightQuantization>>,
    name: &str,
) -> LinearFormat {
    formats
        .and_then(|formats| formats.get(name))
        .copied()
        .or_else(|| default.filter(|_| selected.is_none_or(|names| names.contains(name))))
        .map_or(LinearFormat::Dense, LinearFormat::from)
}

fn insert_grouped_gated_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    plan: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<(), String> {
    for spec in plan.unit_specs().values() {
        let groups = positive(spec.group_count(), "grouped expert count")?;
        let input = positive(spec.input_dimensions(), "grouped expert input")?;
        let intermediate = positive(
            spec.intermediate_dimensions(),
            "grouped expert intermediate",
        )?;
        let output = positive(spec.output_dimensions(), "grouped expert output")?;
        let weight = |projection: &eredu_nn::GroupedProjectionSpec| {
            projection.parameters()[0].id.as_str().to_owned()
        };
        match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
                shapes.insert(
                    weight(gate_up),
                    vec![
                        groups,
                        intermediate
                            .checked_mul(2)
                            .ok_or_else(|| "grouped expert fused width overflowed".to_owned())?,
                        input,
                    ],
                );
                shapes.insert(weight(down), vec![groups, output, intermediate]);
            }
            eredu_nn::GatedProductGroupLayout::Independent(members) => {
                for member in members {
                    shapes.insert(weight(member.gate()), vec![intermediate, input]);
                    shapes.insert(weight(member.up()), vec![intermediate, input]);
                    shapes.insert(weight(member.down()), vec![output, intermediate]);
                }
            }
            _ => return Err("unsupported grouped expert parameter layout".into()),
        }
    }
    Ok(())
}

fn insert_grouped_relu2_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    plan: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
) -> Result<(), String> {
    for spec in plan.unit_specs().values() {
        let groups = positive(spec.group_count(), "ReLU2 expert count")?;
        let hidden = positive(spec.hidden_dimensions(), "ReLU2 expert hidden width")?;
        let intermediate = positive(
            spec.intermediate_dimensions(),
            "ReLU2 expert intermediate width",
        )?;
        shapes.insert(
            spec.up().parameters()[0].id.as_str().to_owned(),
            vec![groups, intermediate, hidden],
        );
        shapes.insert(
            spec.down().parameters()[0].id.as_str().to_owned(),
            vec![groups, hidden, intermediate],
        );
    }
    Ok(())
}

fn family_parameter_role(
    config: &EligibleConfig<'_>,
    name: &str,
    companion: bool,
    linear_shapes: &BTreeMap<String, Vec<usize>>,
) -> ReplicatedTextParameterRole {
    if companion {
        ReplicatedTextParameterRole::FormatCompanion
    } else if name == config.embedding_name()
        || matches!(config, EligibleConfig::Inkling(_)) && name == "model.llm.embed.weight"
    {
        ReplicatedTextParameterRole::Embedding
    } else if linear_shapes.contains_key(name)
        || (config.tied_embeddings() && name == "lm_head.weight")
    {
        ReplicatedTextParameterRole::LinearWeight
    } else if linear_shapes.keys().any(|weight| {
        weight
            .strip_suffix(".weight")
            .is_some_and(|prefix| name == format!("{prefix}.bias"))
            || name == format!("{weight}_bias")
    }) {
        ReplicatedTextParameterRole::LinearBias
    } else {
        ReplicatedTextParameterRole::Other
    }
}

fn positive(value: i32, label: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("decoder {label} must be positive"))
}

/// Returns the exact affine-projection identities and logical geometry declared
/// by a normalized decoder. Artifact rank and name suffixes do not determine
/// transform eligibility.
fn decoder_linear_parameter_shapes<C: crate::decoder::Config>(
    config: &C,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    let root = config.parameter_root();
    let fields = config.block_parameter_fields();
    let hidden = positive(config.hidden_size(), "hidden size")?;
    let intermediate = positive(config.intermediate_size(), "intermediate size")?;
    let head = positive(config.head_dim(), "head dimension")?;
    let query = positive(config.num_attention_heads(), "query-head count")?
        .checked_mul(head)
        .ok_or_else(|| "decoder query width overflowed".to_string())?;
    let key_value = positive(config.num_key_value_heads(), "key/value-head count")?
        .checked_mul(head)
        .ok_or_else(|| "decoder key/value width overflowed".to_string())?;
    let layers = positive(config.num_hidden_layers(), "layer count")?;
    let mut result = BTreeMap::new();
    for layer in 0..layers {
        let layer_prefix = format!("{root}.layers.{layer}");
        let attention = format!("{layer_prefix}.{}", fields.attention);
        match config.attention_projection_layout() {
            crate::decoder::AttentionProjectionLayout::Split => {
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_query),
                    vec![query, hidden],
                );
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_key),
                    vec![key_value, hidden],
                );
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_value),
                    vec![key_value, hidden],
                );
            }
            crate::decoder::AttentionProjectionLayout::Fused { field } => {
                let output =
                    query
                        .checked_add(key_value.checked_mul(2).ok_or_else(|| {
                            "decoder fused attention width overflowed".to_string()
                        })?)
                        .ok_or_else(|| "decoder fused attention width overflowed".to_string())?;
                result.insert(format!("{attention}.{field}.weight"), vec![output, hidden]);
            }
        }
        result.insert(
            format!("{attention}.{}.weight", fields.attention_output),
            vec![hidden, query],
        );
        let feed_forward = format!("{layer_prefix}.{}", fields.feed_forward);
        match config.gated_projection_layout() {
            crate::decoder::GatedProjectionLayout::Split => {
                result.insert(
                    format!("{feed_forward}.{}.weight", fields.feed_forward_gate),
                    vec![intermediate, hidden],
                );
                result.insert(
                    format!("{feed_forward}.{}.weight", fields.feed_forward_up),
                    vec![intermediate, hidden],
                );
            }
            crate::decoder::GatedProjectionLayout::Fused { field } => {
                result.insert(
                    format!("{feed_forward}.{field}.weight"),
                    vec![
                        intermediate.checked_mul(2).ok_or_else(|| {
                            "decoder fused feed-forward width overflowed".to_string()
                        })?,
                        hidden,
                    ],
                );
            }
        }
        result.insert(
            format!("{feed_forward}.{}.weight", fields.feed_forward_output),
            vec![hidden, intermediate],
        );
    }
    if !config.tie_word_embeddings() {
        result.insert(
            "lm_head.weight".into(),
            vec![
                positive(config.vocabulary_size(), "vocabulary size")?,
                hidden,
            ],
        );
    }
    Ok(result)
}

fn decoder_parameter_role<C: crate::decoder::Config>(
    config: &C,
    name: &str,
    companion: bool,
    linear_shapes: &BTreeMap<String, Vec<usize>>,
) -> ReplicatedTextParameterRole {
    if companion {
        return ReplicatedTextParameterRole::FormatCompanion;
    }
    if linear_shapes.contains_key(name)
        || (config.tie_word_embeddings() && name == "lm_head.weight")
    {
        return ReplicatedTextParameterRole::LinearWeight;
    }
    if name == format!("{}.embed_tokens.weight", config.parameter_root()) {
        return ReplicatedTextParameterRole::Embedding;
    }
    if linear_shapes.keys().any(|weight| {
        weight
            .strip_suffix(".weight")
            .is_some_and(|prefix| name == format!("{prefix}.bias"))
            || name == format!("{weight}_bias")
    }) {
        return ReplicatedTextParameterRole::LinearBias;
    }
    let fields = config.block_parameter_fields();
    if name == format!("{}.norm.weight", config.parameter_root())
        || (0..usize::try_from(config.num_hidden_layers()).unwrap_or_default()).any(|layer| {
            let prefix = format!("{}.layers.{layer}", config.parameter_root());
            name == format!("{prefix}.{}.weight", fields.input_norm)
                || name == format!("{prefix}.{}.weight", fields.post_attention_norm)
                || name
                    == format!(
                        "{prefix}.{}.{}.weight",
                        fields.attention, fields.attention_query_norm
                    )
                || name
                    == format!(
                        "{prefix}.{}.{}.weight",
                        fields.attention, fields.attention_key_norm
                    )
        })
    {
        return ReplicatedTextParameterRole::Normalization;
    }
    ReplicatedTextParameterRole::Other
}

/// Derives exact replicated text requirements from an admitted artifact.
pub(crate) fn inspection_recipe_source(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<eredu_checkpoint::store::SharedCheckpointSource, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    match (
        plan.safetensors_architecture(),
        plan.gguf_plan(),
        inspection.gguf_checkpoint(),
    ) {
        (Some(architecture), None, None) => {
            let selected = architecture.checkpoint_resolution().ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArtifact(
                    "SafeTensors architecture omitted exact catalog admission".into(),
                )
            })?;
            let shards = inspection.safetensors_shards().ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArtifact(
                    "SafeTensors inspection omitted its admitted shard set".into(),
                )
            })?;
            let mut metadata = BTreeMap::new();
            for key in selected.source_keys() {
                let descriptor = inspection.tensors().get(key).ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "admitted SafeTensors source {key:?} is absent from its catalog"
                    ))
                })?;
                let backing_shard = shards
                    .tensor_locations()
                    .and_then(|locations| locations.get(key))
                    .cloned()
                    .or_else(|| {
                        (shards.payload_paths().len() == 1)
                            .then(|| shards.payload_paths()[0].clone())
                    });
                metadata.insert(
                    key.clone(),
                    eredu_checkpoint::store::TensorMetadata {
                        name: key.clone(),
                        logical_shape: descriptor.shape.clone(),
                        physical_shape: descriptor.shape.clone(),
                        stored_dtype: stored_dtype(&descriptor.dtype)?,
                        encoded_byte_len: descriptor
                            .storage
                            .as_ref()
                            .map_or(0, |storage| storage.length),
                        backing_shard,
                    },
                );
            }
            Ok(Arc::new(InspectionCheckpointSource {
                metadata,
                backend: eredu_checkpoint::store::WeightStoreBackend::Safetensors,
            }))
        }
        (None, Some(architecture), Some(checkpoint)) => {
            let primary_mapping = plan
                .gguf_media_projector()
                .map_or(architecture.tensor_mapping(), |projector| {
                    projector.primary_tensor_mapping()
                });
            let mut builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
                .add_checkpoint(
                    checkpoint.clone(),
                    architecture.checkpoint(),
                    primary_mapping,
                )
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                })?;
            if let Some(projector_plan) = plan.gguf_media_projector() {
                let companion = inspection
                    .validated_gguf()
                    .and_then(|validated| {
                        validated
                            .companion(&eredu_core::artifact::GgufCompanionRole::MediaProjector)
                    })
                    .ok_or_else(|| {
                        ReplicatedTextRequirementsError::InvalidArtifact(
                            "admitted GGUF media-projector plan omitted its exact companion".into(),
                        )
                    })?;
                builder = builder
                    .add_checkpoint(
                        companion.checkpoint().clone(),
                        projector_plan.checkpoint(),
                        projector_plan.tensor_mapping(),
                    )
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                    })?;
            }
            builder
                .build()
                .map(|source| Arc::new(source) as eredu_checkpoint::store::SharedCheckpointSource)
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                })
        }
        _ => Err(ReplicatedTextRequirementsError::InvalidArtifact(
            "artifact container and admitted architecture plan disagree".into(),
        )),
    }
}

/// Derives the complete replicated execution and checkpoint contract.
pub fn replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    let config = eligible_config(plan)?;
    replicated_text_requirements_for_config(inspection, config)
}

pub(crate) fn qwen_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::qwen::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::Qwen(args))
}

pub(crate) fn gpt_oss_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::gpt_oss::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::GptOss(args))
}

pub(crate) fn nemotron_h_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::nemotron_h::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::NemotronH(args))
}

pub(crate) fn lfm2_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::lfm2::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::Lfm2(args))
}

pub(crate) fn kimi_linear_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::kimi_linear::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::KimiLinear(args))
}

pub(crate) fn qwen_hybrid_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::qwen::hybrid::HybridConfig,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::QwenHybrid(args))
}

pub(crate) fn deepseek_v3_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::deepseek::V3Args,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::DeepSeekV3(args))
}

pub(crate) fn deepseek_v4_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::deepseek::V4Args,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::DeepSeekV4(args))
}

fn replicated_text_requirements_for_config(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: EligibleConfig<'_>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_structure(inspection, config, None)
}

type ReplicatedExecutionStructure = (
    eredu_runtime::ExecutionGraph,
    eredu_runtime::ExecutionUnitLayout,
    Vec<eredu_runtime::ArchitectureGroupTransport>,
    eredu_runtime::StateLayout,
);

fn replicated_text_requirements_for_structure(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: EligibleConfig<'_>,
    structure: Option<ReplicatedExecutionStructure>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    let mut parameters = match (
        plan.safetensors_architecture(),
        plan.gguf_plan(),
        inspection.gguf_checkpoint(),
    ) {
        (Some(architecture), None, None) => safetensors_parameters(
            architecture,
            inspection.tensors(),
            inspection.safetensors_shards().ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArtifact(
                    "SafeTensors inspection omitted its admitted shard set".into(),
                )
            })?,
            &config,
        )?,
        (None, Some(architecture), Some(checkpoint)) => {
            let primary_mapping = plan
                .gguf_media_projector()
                .map_or(architecture.tensor_mapping(), |projector| {
                    projector.primary_tensor_mapping()
                });
            let mut parameters = gguf_parameters(primary_mapping, checkpoint, &config)?;
            if let Some(projector_plan) = plan.gguf_media_projector() {
                let companion = inspection
                    .validated_gguf()
                    .and_then(|validated| {
                        validated
                            .companion(&eredu_core::artifact::GgufCompanionRole::MediaProjector)
                    })
                    .ok_or_else(|| {
                        ReplicatedTextRequirementsError::InvalidArtifact(
                            "admitted GGUF media-projector plan omitted its exact companion".into(),
                        )
                    })?;
                parameters.extend(
                    gguf_parameters(
                        projector_plan.tensor_mapping(),
                        companion.checkpoint(),
                        &config,
                    )?
                    .into_iter()
                    .filter(|parameter| parameter.presence().has_physical_source()),
                );
                parameters = finish_parameters(parameters)?;
            }
            parameters
        }
        _ => {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                "artifact container and admitted architecture plan disagree".into(),
            ))
        }
    };
    let (execution_graph, execution_units, group_transports, state_layout) = match structure {
        Some(structure) => structure,
        None => {
            let graph = eredu_runtime::ExecutionGraph::chain([config.execution_group()]).map_err(
                |error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()),
            )?;
            let units = eredu_runtime::ExecutionUnitLayout::new(
                &graph,
                [config
                    .unit_count()
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?],
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let state = config
                .state_layout()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (graph, units, vec![config.group_transport()], state)
        }
    };
    let recipe_source = inspection_recipe_source(inspection)?;
    let derived_recipes = config
        .derived_recipes(recipe_source.as_ref())
        .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?;
    let derived_recipe_outputs = derived_recipes
        .iter()
        .map(|(target, recipe)| {
            recipe
                .infer(recipe_source.as_ref())
                .map(|metadata| (target.clone(), metadata))
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "derived recipe for {target:?} is invalid: {error}"
                    ))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let derived_target_role = |target: &str| {
        if target.ends_with("_scales")
            || target.ends_with("_biases")
            || target.ends_with("_scale_inv")
        {
            ReplicatedTextParameterRole::FormatCompanion
        } else {
            config.parameter_role(target, false, &linear_shapes)
        }
    };
    let recipe_targets = derived_recipes.keys().cloned().collect::<BTreeSet<_>>();
    let consumed_sources = derived_recipes
        .values()
        .flat_map(|recipe| recipe.source_keys().into_iter())
        .collect::<BTreeSet<_>>();
    for (target, recipe) in &derived_recipes {
        let output = derived_recipe_outputs
            .get(target)
            .expect("one inferred output exists for every derived recipe");
        let physical_sources = recipe
            .source_keys()
            .iter()
            .map(|source_key| {
                let provenance = recipe_source
                    .source_provenance(source_key)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "derived recipe source {source_key:?} has no provenance: {error}"
                        ))
                    })?;
                let shard = provenance.backing_shard.ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "derived recipe source {source_key:?} has no backing shard"
                    ))
                })?;
                ReplicatedTextPhysicalSource::new(
                    provenance.physical_tensor,
                    shard,
                    provenance.output,
                )
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let existing = parameters
            .iter()
            .position(|parameter| parameter.name() == target);
        let replacement = if let Some(index) = existing {
            let parameter = &parameters[index];
            let role = if matches!(config, EligibleConfig::GptOss(_))
                && target.contains(".mlp.experts.")
                && target.ends_with("_proj_bias")
            {
                ReplicatedTextParameterRole::LinearBias
            } else {
                parameter.role()
            };
            ReplicatedTextParameterRequirement::new(
                target.clone(),
                Vec::new(),
                physical_sources,
                parameter.aliases().to_vec(),
                Some(SourceTensorEncoding::Safetensors(recipe_stored_dtype(
                    &output.dtype,
                )?)),
                Some(output.shape.clone()),
                parameter.logical_shape().to_vec(),
                parameter.native_executable(),
                role,
                parameter.owner().clone(),
                ReplicatedTextParameterPresence::Derived {
                    recipe: format!("architecture-output:{target}"),
                },
                parameter.transform_constraint(),
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
        } else {
            let role = derived_target_role(target);
            parameter_requirement(
                target.clone(),
                Vec::new(),
                physical_sources,
                Vec::new(),
                Some(SourceTensorEncoding::Safetensors(recipe_stored_dtype(
                    &output.dtype,
                )?)),
                Some(output.shape.clone()),
                output.shape.clone(),
                config.native_format(target),
                role == ReplicatedTextParameterRole::LinearWeight,
                role,
                parameter_owner(&config, target),
                ReplicatedTextParameterPresence::Derived {
                    recipe: format!("architecture-output:{target}"),
                },
            )?
        };
        if let Some(index) = existing {
            parameters[index] = replacement;
        } else {
            parameters.push(replacement);
        }
    }
    parameters.retain(|parameter| {
        recipe_targets.contains(parameter.name())
            || !std::iter::once(parameter.name())
                .chain(parameter.sources().iter().map(String::as_str))
                .chain(parameter.aliases().iter().map(String::as_str))
                .any(|identity| consumed_sources.contains(identity))
    });
    if matches!(config, EligibleConfig::GptOss(_)) {
        for parameter in &mut parameters {
            if parameter.name().contains(".mlp.experts.")
                && parameter.role() == ReplicatedTextParameterRole::LinearWeight
            {
                *parameter = ReplicatedTextParameterRequirement::new(
                    parameter.name().to_owned(),
                    parameter.sources().to_vec(),
                    parameter.physical_sources().to_vec(),
                    parameter.aliases().to_vec(),
                    parameter.source_encoding().cloned(),
                    parameter.physical_shape().map(<[usize]>::to_vec),
                    parameter.logical_shape().to_vec(),
                    parameter.native_executable(),
                    parameter.role(),
                    parameter.owner().clone(),
                    parameter.presence().clone(),
                    ParameterTransformConstraint::None,
                )
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
            }
        }
    }
    ReplicatedTextRequirements::new(
        config.architecture_identity(),
        config.operators(),
        execution_graph,
        execution_units,
        group_transports,
        state_layout,
        config.state_access(),
        parameters,
    )
    .and_then(|requirements| {
        requirements.with_derived_recipes(derived_recipes, derived_recipe_outputs)
    })
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

fn recipe_stored_dtype(
    dtype: &eredu_checkpoint::recipe::RecipeDtype,
) -> Result<StoredDtype, ReplicatedTextRequirementsError> {
    use eredu_checkpoint::recipe::RecipeDtype;
    Ok(match dtype {
        RecipeDtype::Bool => StoredDtype::Bool,
        RecipeDtype::U8 => StoredDtype::U8,
        RecipeDtype::I8 => StoredDtype::I8,
        RecipeDtype::I16 => StoredDtype::I16,
        RecipeDtype::U16 => StoredDtype::U16,
        RecipeDtype::F16 => StoredDtype::F16,
        RecipeDtype::BF16 => StoredDtype::BF16,
        RecipeDtype::I32 => StoredDtype::I32,
        RecipeDtype::U32 => StoredDtype::U32,
        RecipeDtype::F32 => StoredDtype::F32,
        RecipeDtype::F64 => StoredDtype::F64,
        RecipeDtype::I64 => StoredDtype::I64,
        RecipeDtype::U64 => StoredDtype::U64,
        RecipeDtype::C64 => StoredDtype::C64,
        RecipeDtype::F8E4M3 => StoredDtype::F8E4M3,
        RecipeDtype::F8E5M2 => StoredDtype::F8E5M2,
        RecipeDtype::F4 => StoredDtype::F4,
        RecipeDtype::F8E8M0 => StoredDtype::F8E8M0,
        RecipeDtype::Other(name) => StoredDtype::Other(name.clone()),
        _ => {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                "recipe output uses an unsupported scalar representation".into(),
            ))
        }
    })
}

/// Classifies a normalized SafeTensors architecture for replicated text binding.
pub fn safetensors_replicated_text_eligibility(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
) -> Result<ReplicatedTextStateAccess, ReplicatedTextIneligibility> {
    safetensors_eligible_config(architecture).map(|config| config.state_access())
}

fn safetensors_eligible_config(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
) -> Result<EligibleConfig<'_>, ReplicatedTextIneligibility> {
    match architecture.model() {
        SafetensorsModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
        SafetensorsModelConfig::Qwen(args) if !args.is_moe() => Ok(EligibleConfig::Qwen(args)),
        SafetensorsModelConfig::Qwen(_) => Err(ReplicatedTextIneligibility::Routed),
        SafetensorsModelConfig::QwenHybrid(args) if args.vision.is_some() => {
            Err(ReplicatedTextIneligibility::CompositeInput)
        }
        SafetensorsModelConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        SafetensorsModelConfig::QwenHybrid(args) if args.text.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::QwenHybrid(args) => Ok(EligibleConfig::QwenHybrid(&args.text)),
        SafetensorsModelConfig::KimiLinear(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::KimiLinear(args) => Ok(EligibleConfig::KimiLinear(args)),
        SafetensorsModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::Lfm2(args) => Ok(EligibleConfig::Lfm2(args)),
        SafetensorsModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        SafetensorsModelConfig::NemotronH(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::NemotronH(args) => Ok(EligibleConfig::NemotronH(args)),
        SafetensorsModelConfig::Gemma4(_)
        | SafetensorsModelConfig::Inkling(_)
        | SafetensorsModelConfig::MuseGlimmer(_)
        | SafetensorsModelConfig::QwenVl(_) => Err(ReplicatedTextIneligibility::CompositeInput),
        SafetensorsModelConfig::Moshi(_) => Err(ReplicatedTextIneligibility::Realtime),
        SafetensorsModelConfig::DeepSeekV3(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        SafetensorsModelConfig::DeepSeekV4(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        SafetensorsModelConfig::DeepSeekV3(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::DeepSeekV4(_) => Err(ReplicatedTextIneligibility::Routed),
        SafetensorsModelConfig::GptOss(_) => Err(ReplicatedTextIneligibility::Routed),
        SafetensorsModelConfig::DeepSeekV3(_) => Err(ReplicatedTextIneligibility::Unrelated),
    }
}

/// Classifies a normalized GGUF architecture for replicated text binding.
pub fn gguf_replicated_text_eligibility(
    architecture: &crate::configuration::GgufArchitecturePlan,
) -> Result<ReplicatedTextStateAccess, ReplicatedTextIneligibility> {
    gguf_eligible_config(architecture).map(|config| config.state_access())
}

fn gguf_eligible_config(
    architecture: &crate::configuration::GgufArchitecturePlan,
) -> Result<EligibleConfig<'_>, ReplicatedTextIneligibility> {
    match architecture.model() {
        GgufModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
        GgufModelConfig::Qwen(args)
            if matches!(
                architecture.architecture(),
                GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3
            ) && !args.is_moe() =>
        {
            Ok(EligibleConfig::Qwen(args))
        }
        GgufModelConfig::Qwen(_) => Err(ReplicatedTextIneligibility::Routed),
        GgufModelConfig::QwenHybrid(args) if args.vision.is_some() => {
            Err(ReplicatedTextIneligibility::CompositeInput)
        }
        GgufModelConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        GgufModelConfig::QwenHybrid(args) if args.text.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::QwenHybrid(args) => Ok(EligibleConfig::QwenHybrid(&args.text)),
        GgufModelConfig::KimiLinear(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::KimiLinear(args) => Ok(EligibleConfig::KimiLinear(args)),
        GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::Lfm2(args) => Ok(EligibleConfig::Lfm2(args)),
        GgufModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        GgufModelConfig::NemotronH(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::NemotronH(args) => Ok(EligibleConfig::NemotronH(args)),
        GgufModelConfig::Gemma4(_)
        | GgufModelConfig::Inkling(_)
        | GgufModelConfig::MuseGlimmer(_) => Err(ReplicatedTextIneligibility::CompositeInput),
        GgufModelConfig::DeepSeekV3(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        GgufModelConfig::DeepSeekV4(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        GgufModelConfig::DeepSeekV3(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::DeepSeekV4(_) => Err(ReplicatedTextIneligibility::Routed),
        GgufModelConfig::GptOss(_) => Err(ReplicatedTextIneligibility::Routed),
        GgufModelConfig::DeepSeekV3(_) => Err(ReplicatedTextIneligibility::Unrelated),
    }
}

fn eligible_config(
    plan: &ArtifactArchitecturePlan,
) -> Result<EligibleConfig<'_>, ReplicatedTextIneligibility> {
    if plan.has_processor() || plan.gguf_media_projector().is_some() {
        return Err(ReplicatedTextIneligibility::CompositeInput);
    }
    match (plan.safetensors_architecture(), plan.gguf_plan()) {
        (Some(architecture), None) => safetensors_eligible_config(architecture),
        (None, Some(architecture)) => gguf_eligible_config(architecture),
        _ => Err(ReplicatedTextIneligibility::Unrelated),
    }
}

fn ordinary_eligible_config(
    plan: &ArtifactArchitecturePlan,
) -> Result<EligibleConfig<'_>, ReplicatedTextIneligibility> {
    match eligible_config(plan)? {
        config @ (EligibleConfig::Llama(_) | EligibleConfig::Qwen(_)) => Ok(config),
        EligibleConfig::Lfm2(_)
        | EligibleConfig::KimiLinear(_)
        | EligibleConfig::NemotronH(_)
        | EligibleConfig::QwenHybrid(_)
        | EligibleConfig::GptOss(_)
        | EligibleConfig::DeepSeekV3(_)
        | EligibleConfig::DeepSeekV4(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::MuseGlimmer(_)
        | EligibleConfig::QwenVl(_)
        | EligibleConfig::QwenCompositeHybrid(_) => Err(ReplicatedTextIneligibility::HybridState),
    }
}

fn qwen_next_fused_targets(
    config: &crate::qwen::hybrid::HybridConfig,
    name: &str,
    physical_shape: &[usize],
) -> Result<Option<Vec<(String, Vec<usize>)>>, ReplicatedTextRequirementsError> {
    if config.variant != crate::qwen::hybrid::HybridVariant::Qwen3Next {
        return Ok(None);
    }
    let (base, suffix, qkvz) = if let Some(base) = name.strip_suffix("in_proj_qkvz.weight") {
        (base, "weight", true)
    } else if let Some(base) = name.strip_suffix("in_proj_qkvz.scales") {
        (base, "scales", true)
    } else if let Some(base) = name.strip_suffix("in_proj_qkvz.biases") {
        (base, "biases", true)
    } else if let Some(base) = name.strip_suffix("in_proj_qkvz.weight_scale_inv") {
        (base, "weight_scale_inv", true)
    } else if let Some(base) = name.strip_suffix("in_proj_ba.weight") {
        (base, "weight", false)
    } else if let Some(base) = name.strip_suffix("in_proj_ba.scales") {
        (base, "scales", false)
    } else if let Some(base) = name.strip_suffix("in_proj_ba.biases") {
        (base, "biases", false)
    } else {
        return Ok(None);
    };
    let (widths, ba_width) = crate::qwen::hybrid::fused_projection_widths(config)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    let groups = usize::try_from(config.linear_num_key_heads).map_err(|_| {
        ReplicatedTextRequirementsError::InvalidArchitecture(
            "Qwen3-Next grouped projection count exceeds usize".into(),
        )
    })?;
    let widths = if suffix == "weight_scale_inv" {
        crate::qwen::hybrid::fp8_block_row_widths(&widths)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
            .try_into()
            .map_err(|_| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "Qwen3-Next FP8 projection width count changed".into(),
                )
            })?
    } else {
        widths
    };
    let shape = |rows: usize| {
        let mut shape = physical_shape.to_vec();
        if let Some(first) = shape.first_mut() {
            *first = rows;
        }
        shape
    };
    let grouped_rows = |width: i32| {
        usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(groups))
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "Qwen3-Next grouped projection dimension exceeds usize".into(),
                )
            })
    };
    let targets = if qkvz {
        let widths = [
            grouped_rows(widths[0])?,
            grouped_rows(widths[1])?,
            grouped_rows(widths[2])?,
            grouped_rows(widths[3])?,
        ];
        let qkv_rows = widths[0]
            .checked_add(widths[1])
            .and_then(|rows| rows.checked_add(widths[2]))
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "Qwen3-Next grouped QKV dimension exceeds usize".into(),
                )
            })?;
        vec![
            (format!("{base}in_proj_qkv.{suffix}"), shape(qkv_rows)),
            (format!("{base}in_proj_z.{suffix}"), shape(widths[3])),
        ]
    } else {
        let rows = grouped_rows(ba_width)?;
        vec![
            (format!("{base}in_proj_b.{suffix}"), shape(rows)),
            (format!("{base}in_proj_a.{suffix}"), shape(rows)),
        ]
    };
    Ok(Some(targets))
}

fn safetensors_parameters(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    catalog: &TensorCatalog,
    shards: &eredu_checkpoint::safetensors::SafetensorsShards,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let source_linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let selected = architecture.checkpoint_resolution().ok_or_else(|| {
        ReplicatedTextRequirementsError::InvalidArtifact(
            "SafeTensors architecture omitted exact catalog admission".into(),
        )
    })?;
    let plan = architecture.checkpoint();
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    for group in &plan.layout_groups {
        for variant in &group.variants {
            if variant.tensors.iter().all(|constraint| {
                selected.source_keys().contains(&constraint.key)
                    || constraint
                        .aliases
                        .iter()
                        .any(|alias| selected.source_keys().contains(alias))
                    || constraint.requirement
                        == eredu_checkpoint::schema::TensorRequirement::Optional
            }) {
                constraints.extend(variant.tensors.iter());
                break;
            }
        }
    }
    let linear_shapes = source_linear_shapes
        .into_iter()
        .map(|(name, shape)| {
            let canonical = constraints
                .iter()
                .find(|constraint| constraint.key == name)
                .map_or(name, |constraint| {
                    config.canonical_parameter_name(&constraint.key, &constraint.aliases)
                });
            (canonical, shape)
        })
        .collect::<BTreeMap<_, _>>();
    let mut parameters = Vec::new();
    for constraint in constraints {
        let canonical = config.canonical_parameter_name(&constraint.key, &constraint.aliases);
        let source = std::iter::once(&constraint.key)
            .chain(constraint.aliases.iter())
            .find(|name| selected.source_keys().contains(*name))
            .cloned();
        let descriptor = source
            .as_deref()
            .map(|source| {
                catalog.get(source).ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "admitted SafeTensors source {source:?} is absent from its catalog"
                    ))
                })
            })
            .transpose()?;
        let presence = match (constraint.requirement, source.is_some()) {
            (eredu_checkpoint::schema::TensorRequirement::Required, true) => {
                ReplicatedTextParameterPresence::Required
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, true) => {
                ReplicatedTextParameterPresence::OptionalPresent
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, false) => {
                ReplicatedTextParameterPresence::OptionalAbsent
            }
            (eredu_checkpoint::schema::TensorRequirement::Required, false) => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                    "required admitted SafeTensors parameter {:?} has no source",
                    constraint.key
                )))
            }
        };
        let role = config.parameter_role(
            &canonical,
            constraint.role == eredu_checkpoint::schema::TensorRole::Companion,
            &linear_shapes,
        );
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            linear_shapes
                .get(&canonical)
                .cloned()
                .unwrap_or_else(|| constraint.shape.clone())
        };
        let aliases = std::iter::once(constraint.key.as_str())
            .chain(constraint.aliases.iter().map(String::as_str))
            .filter(|name| *name != canonical)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let physical_sources = source
            .as_ref()
            .map(|source| {
                let shard = shards
                    .tensor_locations()
                    .and_then(|locations| locations.get(source))
                    .or_else(|| {
                        (shards.payload_paths().len() == 1).then(|| &shards.payload_paths()[0])
                    })
                    .ok_or_else(|| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "admitted SafeTensors source {source:?} has no exact shard membership"
                        ))
                    })?;
                ReplicatedTextPhysicalSource::new(source, shard, source).map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                })
            })
            .transpose()?
            .into_iter()
            .collect::<Vec<_>>();
        let source_encoding = descriptor
            .map(|descriptor| stored_dtype(&descriptor.dtype))
            .transpose()?
            .map(SourceTensorEncoding::Safetensors);
        let physical_shape = descriptor.map(|descriptor| descriptor.shape.clone());
        if let EligibleConfig::QwenHybrid(args) = config {
            if let Some(targets) = qwen_next_fused_targets(args, &canonical, &constraint.shape)? {
                for (target, shape) in targets {
                    let role = if constraint.role == eredu_checkpoint::schema::TensorRole::Companion
                    {
                        ReplicatedTextParameterRole::FormatCompanion
                    } else {
                        ReplicatedTextParameterRole::LinearWeight
                    };
                    let selects_lowering = role == ReplicatedTextParameterRole::LinearWeight;
                    parameters.push(parameter_requirement(
                        target.clone(),
                        if selects_lowering {
                            source.clone().into_iter().collect()
                        } else {
                            Vec::new()
                        },
                        physical_sources.clone(),
                        aliases.clone(),
                        source_encoding.clone(),
                        physical_shape.clone(),
                        shape,
                        config.native_format(&target),
                        false,
                        role,
                        parameter_owner(config, &target),
                        ReplicatedTextParameterPresence::Derived {
                            recipe: "qwen3_next.grouped_projection_split".into(),
                        },
                    )?);
                }
                continue;
            }
        }
        let native_executable = if matches!(config, EligibleConfig::GptOss(_))
            && canonical.contains(".mlp.experts.")
            && (canonical.ends_with("gate_up_proj") || canonical.ends_with("down_proj"))
        {
            config.native_format(&canonical)
        } else {
            match role {
                ReplicatedTextParameterRole::LinearWeight
                | ReplicatedTextParameterRole::Embedding => config.native_format(&canonical),
                ReplicatedTextParameterRole::FormatCompanion
                | ReplicatedTextParameterRole::Normalization
                | ReplicatedTextParameterRole::LinearBias
                | ReplicatedTextParameterRole::Other => LinearFormat::Dense,
                _ => LinearFormat::Dense,
            }
        };
        parameters.push(parameter_requirement(
            canonical.clone(),
            source.clone().into_iter().collect(),
            physical_sources,
            aliases,
            source_encoding,
            physical_shape,
            logical_shape,
            native_executable,
            linear_shapes.contains_key(&canonical),
            role,
            parameter_owner(config, &canonical),
            presence,
        )?);
    }
    finish_parameters_with_tied_output(parameters, config)
}

fn gguf_parameters(
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    checkpoint: &eredu_gguf::Checkpoint,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let mut physical = BTreeMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            physical.insert(
                tensor.descriptor().name.as_str(),
                (
                    tensor,
                    shard.path(),
                    SourceTensorEncoding::Gguf {
                        ggml_type: tensor.descriptor().ggml_type,
                        endian: shard.endian(),
                    },
                ),
            );
        }
    }
    let mut parameters = Vec::new();
    for mapping in tensor_mapping {
        let Some((tensor, shard, source_encoding)) = physical.get(mapping.physical_name.as_str())
        else {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                "admitted GGUF mapping references absent tensor {:?}",
                mapping.physical_name
            )));
        };
        let native = if let Some((bits, group)) = tensor.affine() {
            LinearFormat::Affine(AffineQuantization::new(
                i32::try_from(group).map_err(|_| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "GGUF group size {group} exceeds i32"
                    ))
                })?,
                i32::from(bits),
            )?)
        } else if tensor.is_mxfp4() {
            LinearFormat::MxFp4
        } else if tensor.descriptor().ggml_type.block_and_bytes().is_ok()
            && !matches!(
                tensor.descriptor().ggml_type,
                eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
            )
        {
            let SourceTensorEncoding::Gguf { ggml_type, endian } = source_encoding else {
                unreachable!("GGUF catalog produces GGUF source encodings")
            };
            LinearFormat::GgufIQuant {
                ggml_type: *ggml_type,
                endian: *endian,
            }
        } else {
            LinearFormat::Dense
        };
        let companion = mapping.original_name != mapping.physical_name;
        let role = config.parameter_role(&mapping.layout.name, companion, &linear_shapes);
        let translated_shape = mapping
            .layout
            .shape
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "GGUF logical shape for {:?} exceeds usize",
                        mapping.layout.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let EligibleConfig::QwenHybrid(args) = config {
            if let Some(targets) =
                qwen_next_fused_targets(args, &mapping.layout.name, &translated_shape)?
            {
                let physical_shape = tensor
                    .descriptor()
                    .row_major_shape()
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "GGUF physical shape for {:?} exceeds usize",
                            mapping.physical_name
                        ))
                    })?;
                let provenance = ReplicatedTextPhysicalSource::new(
                    mapping.physical_name.clone(),
                    *shard,
                    mapping.original_name.clone(),
                )
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                })?;
                for (target, shape) in targets {
                    let role = if companion {
                        ReplicatedTextParameterRole::FormatCompanion
                    } else {
                        ReplicatedTextParameterRole::LinearWeight
                    };
                    let selects_lowering = !companion;
                    let logical_shape = if selects_lowering {
                        linear_shapes.get(&target).cloned().unwrap_or(shape)
                    } else {
                        shape
                    };
                    parameters.push(parameter_requirement(
                        target.clone(),
                        selects_lowering
                            .then(|| mapping.layout.name.clone())
                            .into_iter()
                            .collect(),
                        vec![provenance.clone()],
                        vec![mapping.original_name.clone()],
                        Some(source_encoding.clone()),
                        Some(physical_shape.clone()),
                        logical_shape,
                        if selects_lowering {
                            native
                        } else {
                            LinearFormat::Dense
                        },
                        false,
                        role,
                        parameter_owner(config, &target),
                        ReplicatedTextParameterPresence::Derived {
                            recipe: "qwen3_next.grouped_projection_split".into(),
                        },
                    )?);
                }
                continue;
            }
        }
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            let discovered = linear_shapes
                .get(&mapping.layout.name)
                .cloned()
                .unwrap_or(translated_shape.clone());
            config.logical_parameter_shape(&mapping.layout.name, discovered)
        };
        let derived = companion
            || (config.parameter_requires_shape_recipe(&mapping.layout.name)
                && logical_shape != translated_shape);
        let presence = if derived {
            ReplicatedTextParameterPresence::Derived {
                recipe: format!(
                    "gguf-output:{}:{}",
                    mapping.physical_name, mapping.original_name
                ),
            }
        } else {
            ReplicatedTextParameterPresence::Required
        };
        parameters.push(parameter_requirement(
            mapping.layout.name.clone(),
            (!derived)
                .then(|| mapping.layout.name.clone())
                .into_iter()
                .collect(),
            vec![ReplicatedTextPhysicalSource::new(
                mapping.physical_name.clone(),
                *shard,
                mapping.original_name.clone(),
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
            })?],
            vec![mapping.original_name.clone()],
            Some(source_encoding.clone()),
            Some({
                tensor
                    .descriptor()
                    .row_major_shape()
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "GGUF physical shape for {:?} exceeds usize",
                            mapping.physical_name
                        ))
                    })
            }?),
            logical_shape,
            if derived { LinearFormat::Dense } else { native },
            !derived && linear_shapes.contains_key(&mapping.layout.name),
            role,
            parameter_owner(config, &mapping.layout.name),
            presence,
        )?);
    }
    finish_parameters_with_tied_output(parameters, config)
}

fn finish_parameters_with_tied_output(
    mut parameters: Vec<ReplicatedTextParameterRequirement>,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    if config.tied_embeddings() {
        parameters.retain(|parameter| parameter.name() != "lm_head.weight");
        parameters.push(parameter_requirement(
            "lm_head.weight".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            LinearFormat::Dense,
            true,
            ReplicatedTextParameterRole::LinearWeight,
            ReplicatedTextParameterOwner::StaticRole("output".into()),
            ReplicatedTextParameterPresence::Tied {
                target: config.embedding_name(),
            },
        )?);
    }
    finish_parameters(parameters)
}

fn parameter_requirement(
    name: String,
    sources: Vec<String>,
    physical_sources: Vec<ReplicatedTextPhysicalSource>,
    aliases: Vec<String>,
    source_encoding: Option<SourceTensorEncoding>,
    physical_shape: Option<Vec<usize>>,
    logical_shape: Vec<usize>,
    native_executable: LinearFormat,
    transformable: bool,
    role: ReplicatedTextParameterRole,
    owner: ReplicatedTextParameterOwner,
    presence: ReplicatedTextParameterPresence,
) -> Result<ReplicatedTextParameterRequirement, ReplicatedTextRequirementsError> {
    let transform = if transformable {
        ParameterTransformConstraint::Linear {
            packed_axis: logical_shape.len().checked_sub(1).ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "linear parameter has a scalar logical shape".into(),
                )
            })?,
        }
    } else {
        ParameterTransformConstraint::None
    };
    ReplicatedTextParameterRequirement::new(
        name,
        sources,
        physical_sources,
        aliases,
        source_encoding,
        physical_shape,
        logical_shape,
        native_executable,
        role,
        owner,
        presence,
        transform,
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

fn parameter_owner(config: &EligibleConfig<'_>, name: &str) -> ReplicatedTextParameterOwner {
    let execution_unit = |prefix: &str, group: &str| {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.split('.').next())
            .and_then(|layer| layer.parse::<usize>().ok())
            .map(|unit| ReplicatedTextParameterOwner::ExecutionUnit {
                group: group.into(),
                unit,
            })
    };
    let composite = match config {
        EligibleConfig::Gemma4(_) => execution_unit(
            "model.vision_tower.encoder.layers.",
            crate::gemma4::model::VISION_EXECUTION_GROUP,
        )
        .or_else(|| {
            execution_unit(
                "model.audio_tower.layers.",
                crate::gemma4::model::AUDIO_EXECUTION_GROUP,
            )
        })
        .or_else(|| {
            execution_unit(
                "model.language_model.layers.",
                crate::gemma4::model::TEXT_EXECUTION_GROUP,
            )
        }),
        EligibleConfig::Inkling(_) => execution_unit(
            "visual.layers.",
            crate::inkling::model::VISION_EXECUTION_GROUP,
        )
        .or_else(|| execution_unit("model.layers.", crate::inkling::model::TEXT_EXECUTION_GROUP)),
        EligibleConfig::MuseGlimmer(_) => execution_unit(
            "model.vision_tower.layers.",
            crate::muse_glimmer::model::VISION_EXECUTION_GROUP,
        )
        .or_else(|| {
            execution_unit(
                "model.layers.",
                crate::muse_glimmer::model::TEXT_EXECUTION_GROUP,
            )
        }),
        EligibleConfig::QwenVl(_) => execution_unit(
            "model.visual.blocks.",
            crate::qwen::vl::VISION_EXECUTION_GROUP,
        )
        .or_else(|| {
            execution_unit(
                "model.language_model.layers.",
                crate::qwen::vl::TEXT_EXECUTION_GROUP,
            )
        }),
        EligibleConfig::QwenCompositeHybrid(_) => execution_unit(
            "model.visual.blocks.",
            crate::qwen::hybrid::VISION_EXECUTION_GROUP,
        )
        .or_else(|| execution_unit("model.layers.", crate::decoder::TARGET_EXECUTION_GROUP)),
        _ => None,
    };
    if let Some(owner) = composite {
        return owner;
    }
    let layer_prefix = if matches!(config, EligibleConfig::DeepSeekV4(_)) {
        "layers.".into()
    } else {
        format!("{}.layers.", config.parameter_root())
    };
    if let Some(rest) = name.strip_prefix(&layer_prefix) {
        if let Some(layer) = rest
            .split('.')
            .next()
            .and_then(|layer| layer.parse::<usize>().ok())
        {
            return ReplicatedTextParameterOwner::ExecutionUnit {
                group: config.execution_group().into(),
                unit: layer,
            };
        }
    }
    let embedding = config.embedding_name();
    let embedding_prefix = embedding.strip_suffix("weight").unwrap_or(&embedding);
    let role = if matches!(config, EligibleConfig::Gemma4(_)) {
        if name.starts_with("model.vision_tower.") {
            "vision"
        } else if name.starts_with("model.embed_vision.") {
            "vision_projection"
        } else if name.starts_with("model.audio_tower.") {
            "audio"
        } else if name.starts_with("model.embed_audio.") {
            "audio_projection"
        } else if name == embedding || name.starts_with(embedding_prefix) {
            "embedding"
        } else if name.starts_with("model.language_model.embed_tokens_per_layer.") {
            "per_layer_embedding"
        } else if name.starts_with("model.language_model.per_layer_model_projection.") {
            "per_layer_projection"
        } else if name.starts_with("model.language_model.per_layer_projection_norm.") {
            "per_layer_norm"
        } else if name.starts_with("lm_head.") {
            "output"
        } else {
            "norm"
        }
    } else if matches!(
        config,
        EligibleConfig::QwenVl(_) | EligibleConfig::QwenCompositeHybrid(_)
    ) && name.starts_with("model.visual.")
    {
        "vision"
    } else if matches!(config, EligibleConfig::Inkling(_)) {
        if name.starts_with("audio.") {
            "audio"
        } else if name.starts_with("visual.") {
            "vision"
        } else if name.starts_with("model.embed_norm.") {
            "embedding_norm"
        } else if name == embedding || name.starts_with(embedding_prefix) {
            "embedding"
        } else if name.starts_with("lm_head.") {
            "output"
        } else {
            "norm"
        }
    } else if matches!(config, EligibleConfig::MuseGlimmer(_)) && name.starts_with("model.vision_")
    {
        "vision"
    } else if matches!(config, EligibleConfig::DeepSeekV4(_)) && name.starts_with("hc_head_") {
        "hyper_head"
    } else if name == embedding || name.starts_with(embedding_prefix) {
        "embedding"
    } else if name == "lm_head.weight"
        || name == "lm_head.bias"
        || name.starts_with("lm_head.")
        || (matches!(config, EligibleConfig::DeepSeekV4(_)) && name.starts_with("head."))
    {
        "output"
    } else {
        "norm"
    };
    ReplicatedTextParameterOwner::StaticRole(role.into())
}

fn finish_parameters(
    mut parameters: Vec<ReplicatedTextParameterRequirement>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    parameters.sort_by(|left, right| left.name().cmp(right.name()));
    let mut names = BTreeSet::new();
    if let Some(duplicate) = parameters
        .iter()
        .find(|parameter| !names.insert(parameter.name()))
    {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "logical parameter {:?} is mapped more than once",
            duplicate.name()
        )));
    }
    if parameters.is_empty() {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(
            "replicated text artifact contains no admitted linear parameters".into(),
        ));
    }
    Ok(parameters)
}

fn stored_dtype(dtype: &TensorDtype) -> Result<StoredDtype, ReplicatedTextRequirementsError> {
    Ok(match dtype {
        TensorDtype::Bool => StoredDtype::Bool,
        TensorDtype::F32 => StoredDtype::F32,
        TensorDtype::F16 => StoredDtype::F16,
        TensorDtype::Bf16 => StoredDtype::BF16,
        TensorDtype::I8 => StoredDtype::I8,
        TensorDtype::U8 => StoredDtype::U8,
        TensorDtype::U16 => StoredDtype::U16,
        TensorDtype::U32 => StoredDtype::U32,
        TensorDtype::U64 => StoredDtype::U64,
        TensorDtype::I16 => StoredDtype::I16,
        TensorDtype::I32 => StoredDtype::I32,
        TensorDtype::I64 => StoredDtype::I64,
        TensorDtype::F64 => StoredDtype::F64,
        TensorDtype::Complex64 => StoredDtype::C64,
        TensorDtype::Encoded(name) => match name.as_str() {
            "F8_E4M3" => StoredDtype::F8E4M3,
            "F8_E5M2" => StoredDtype::F8E5M2,
            "F4" => StoredDtype::F4,
            "F8_E8M0" => StoredDtype::F8E8M0,
            _ => StoredDtype::Other(name.clone()),
        },
    })
}

/// Failure while deriving replicated text requirements from an admitted artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextRequirementsError {
    /// The architecture belongs to a different execution class.
    #[error(transparent)]
    Ineligible(#[from] ReplicatedTextIneligibility),
    /// Admitted artifact facts are internally inconsistent.
    #[error("invalid replicated text artifact: {0}")]
    InvalidArtifact(String),
    /// Architecture geometry or a requested transform is invalid.
    #[error("invalid replicated text architecture: {0}")]
    InvalidArchitecture(String),
}

/// Decoder strategy required after replicated composite ingress.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompositeTextDecoderStrategy {
    /// Ordinary layered decoder units.
    Direct,
    /// Architecture-routed decoder units using the shared grouped provider.
    Routed,
}

/// Exact admitted graph and artifact facts for replicated composite execution.
///
/// The complete inspection is retained privately so typed architecture dispatch
/// can consume its normalized configuration, checkpoint schemas, physical
/// provenance, companion artifacts, and processor snapshot without rediscovery.
/// Backend selection sees only family-neutral requirements.
#[derive(Debug, Clone)]
pub struct CompositeTextRequirements {
    architecture_identity: String,
    execution_graph: eredu_runtime::ExecutionGraph,
    execution_units: eredu_runtime::ExecutionUnitLayout,
    group_transports: Vec<eredu_runtime::ArchitectureGroupTransport>,
    state_layout: eredu_runtime::StateLayout,
    input_modalities: InputModalities,
    decoder: CompositeTextDecoderStrategy,
    raw_processor: bool,
    media_projector: bool,
    processor: eredu_runtime::ProcessorExecutionRequirements,
    execution: ReplicatedTextRequirements,
    routed: Option<crate::RoutedTextRequirements>,
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
}

impl PartialEq for CompositeTextRequirements {
    fn eq(&self, other: &Self) -> bool {
        self.architecture_identity == other.architecture_identity
            && self.execution_graph == other.execution_graph
            && self.execution_units == other.execution_units
            && self.group_transports == other.group_transports
            && self.state_layout == other.state_layout
            && self.input_modalities == other.input_modalities
            && self.decoder == other.decoder
            && self.raw_processor == other.raw_processor
            && self.media_projector == other.media_projector
            && self.processor == other.processor
            && self.execution == other.execution
            && self.routed == other.routed
            && self.inspection.format() == other.inspection.format()
            && self.inspection.tensors() == other.inspection.tensors()
            && self.inspection.safetensors_shards() == other.inspection.safetensors_shards()
    }
}

impl CompositeTextRequirements {
    /// The normalized artifact inspection that authoritatively selected these
    /// requirements.
    ///
    /// Architecture-owned typed dispatch consumes this snapshot directly so
    /// it never reparses artifacts or performs a second family selection.
    pub const fn inspection(&self) -> &ArtifactInspection<ArtifactArchitecturePlan> {
        &self.inspection
    }

    /// Stable architecture/cache identity derived from normalized graph facts.
    pub fn architecture_identity(&self) -> &str {
        &self.architecture_identity
    }

    /// Canonical composite execution graph.
    pub const fn execution_graph(&self) -> &eredu_runtime::ExecutionGraph {
        &self.execution_graph
    }

    /// Exact unit geometry in canonical group order.
    pub const fn execution_units(&self) -> &eredu_runtime::ExecutionUnitLayout {
        &self.execution_units
    }

    /// Architecture-owned transport, kind, and optional-root declarations.
    pub fn group_transports(&self) -> &[eredu_runtime::ArchitectureGroupTransport] {
        &self.group_transports
    }

    /// Complete target mutable-state geometry.
    pub const fn state_layout(&self) -> &eredu_runtime::StateLayout {
        &self.state_layout
    }

    /// Modalities admitted by the normalized architecture and artifact.
    pub const fn input_modalities(&self) -> InputModalities {
        self.input_modalities
    }

    /// Direct or routed target-decoder strategy.
    pub const fn decoder(&self) -> CompositeTextDecoderStrategy {
        self.decoder
    }

    /// Whether retained sidecars admit raw decoded-media preparation.
    pub const fn has_raw_processor(&self) -> bool {
        self.raw_processor
    }

    /// Whether a separately admitted GGUF projector is present.
    pub const fn has_media_projector(&self) -> bool {
        self.media_projector
    }

    /// Exact modality, representation, primitive, and native-bound requirements.
    pub const fn processor_execution(&self) -> &eredu_runtime::ProcessorExecutionRequirements {
        &self.processor
    }

    /// Complete parameter, format, operator, state, and graph requirements.
    pub const fn execution(&self) -> &ReplicatedTextRequirements {
        &self.execution
    }

    /// Exact grouped-bank requirements when the target decoder is routed.
    pub const fn routed_execution(&self) -> Option<&crate::RoutedTextRequirements> {
        self.routed.as_ref()
    }
}

/// Authoritative direct or routed execution paired with one processor selection.
#[derive(Debug, Clone)]
pub enum SelectedCompositeTextRealization {
    /// Direct decoder execution through the shared replicated session.
    Direct(eredu_runtime::SelectedCompositeRealization),
    /// Routed decoder execution through the shared planned provider.
    Routed {
        /// Exact routed text, state, materialization, and bank realization.
        execution: crate::SelectedRoutedTextRealization,
        /// Exact admitted input representations and processor mechanisms.
        processor: eredu_runtime::SelectedProcessorExecution,
    },
}

impl SelectedCompositeTextRealization {
    /// Selected shared text realization independent of decoder strategy.
    pub const fn execution(&self) -> &SelectedReplicatedTextRealization {
        match self {
            Self::Direct(selected) => selected.execution(),
            Self::Routed { execution, .. } => execution.text(),
        }
    }

    /// Selected processor realization.
    pub const fn processor(&self) -> &eredu_runtime::SelectedProcessorExecution {
        match self {
            Self::Direct(selected) => selected.processor(),
            Self::Routed { processor, .. } => processor,
        }
    }
}

/// Complete fail-closed diagnostic for composite decoder and input selection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("replicated composite realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct CompositeTextSelectionError {
    issues: Vec<String>,
}

impl CompositeTextSelectionError {
    /// Every missing execution, bank, or processor mechanism in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Selects direct/routed execution and input mechanisms before construction.
pub fn select_composite_text_realization(
    requirements: &CompositeTextRequirements,
    execution_request: &eredu_runtime::ReplicatedTextSelectionRequest,
    weight_residency: eredu_runtime::WeightResidency,
    processor_request: &eredu_runtime::ProcessorSelectionRequest,
    execution_capabilities: &eredu_runtime::BackendMechanismCapabilities,
    processor_capabilities: &eredu_runtime::MediaPrimitiveCapabilities,
) -> Result<SelectedCompositeTextRealization, CompositeTextSelectionError> {
    let processor = eredu_runtime::select_processor_execution(
        requirements.processor_execution(),
        processor_request,
        processor_capabilities,
    );
    if let Some(routed) = requirements.routed_execution() {
        let routed_request =
            crate::RoutedTextSelectionRequest::new(execution_request.clone(), weight_residency)
                .map_err(|error| CompositeTextSelectionError {
                    issues: error.issues().to_vec(),
                })?;
        let execution =
            crate::select_routed_text_realization(routed, &routed_request, execution_capabilities);
        let mut issues = execution
            .as_ref()
            .err()
            .map(|error| error.issues().to_vec())
            .unwrap_or_default();
        issues.extend(
            processor
                .as_ref()
                .err()
                .map(|error| error.issues().to_vec())
                .unwrap_or_default(),
        );
        if !issues.is_empty() {
            return Err(CompositeTextSelectionError { issues });
        }
        return Ok(SelectedCompositeTextRealization::Routed {
            execution: execution.expect("empty diagnostics imply routed selection"),
            processor: processor.expect("empty diagnostics imply processor selection"),
        });
    }
    let execution = eredu_runtime::select_replicated_text_realization(
        requirements.execution(),
        execution_request,
        execution_capabilities,
    );
    let mut issues = execution
        .as_ref()
        .err()
        .map(|error| error.issues().to_vec())
        .unwrap_or_default();
    issues.extend(
        processor
            .as_ref()
            .err()
            .map(|error| error.issues().to_vec())
            .unwrap_or_default(),
    );
    if !issues.is_empty() {
        return Err(CompositeTextSelectionError { issues });
    }
    Ok(SelectedCompositeTextRealization::Direct(
        eredu_runtime::SelectedCompositeRealization::from_parts(
            execution.expect("empty diagnostics imply direct selection"),
            processor.expect("empty diagnostics imply processor selection"),
        ),
    ))
}

/// Selects composite decoder execution around one previously selected processor proof.
pub fn select_composite_text_realization_with_processor(
    requirements: &CompositeTextRequirements,
    execution_request: &eredu_runtime::ReplicatedTextSelectionRequest,
    weight_residency: eredu_runtime::WeightResidency,
    execution_capabilities: &eredu_runtime::BackendMechanismCapabilities,
    processor: eredu_runtime::SelectedProcessorExecution,
) -> Result<SelectedCompositeTextRealization, CompositeTextSelectionError> {
    if processor.requirements() != requirements.processor_execution() {
        return Err(CompositeTextSelectionError {
            issues: vec!["selected processor requirements do not match the composite".into()],
        });
    }
    if let Some(routed) = requirements.routed_execution() {
        let request =
            crate::RoutedTextSelectionRequest::new(execution_request.clone(), weight_residency)
                .map_err(|error| CompositeTextSelectionError {
                    issues: error.issues().to_vec(),
                })?;
        let execution =
            crate::select_routed_text_realization(routed, &request, execution_capabilities)
                .map_err(|error| CompositeTextSelectionError {
                    issues: error.issues().to_vec(),
                })?;
        return Ok(SelectedCompositeTextRealization::Routed {
            execution,
            processor,
        });
    }
    let execution = eredu_runtime::select_replicated_text_realization(
        requirements.execution(),
        execution_request,
        execution_capabilities,
    )
    .map_err(|error| CompositeTextSelectionError {
        issues: error.issues().to_vec(),
    })?;
    Ok(SelectedCompositeTextRealization::Direct(
        eredu_runtime::SelectedCompositeRealization::from_parts(execution, processor),
    ))
}

/// Checked composite architecture passed to a backend-generic session constructor.
pub struct PreparedCompositeTextArchitecture<A, C> {
    architecture: crate::composite_execution::PreparedCompositeArchitecture<A>,
    source_architecture: Option<crate::composite_execution::PreparedCompositeArchitecture<A>>,
    requirements: CompositeTextRequirements,
    contract: eredu_runtime::PreparedReplicatedTextContract,
    processor: eredu_runtime::SelectedProcessorExecution,
    admission: C,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

/// Checked composite architecture whose target decoder uses grouped providers.
pub struct PreparedRoutedCompositeTextArchitecture<A, C> {
    routed: crate::PreparedRoutedTextArchitecture<
        crate::composite_execution::PreparedCompositeArchitecture<A>,
    >,
    requirements: CompositeTextRequirements,
    processor: eredu_runtime::SelectedProcessorExecution,
    admission: C,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

impl<A, C> PreparedRoutedCompositeTextArchitecture<A, C> {
    /// Exact architecture, artifact, processor, and grouped-bank requirements.
    pub const fn requirements(&self) -> &CompositeTextRequirements {
        &self.requirements
    }

    /// Authoritative processor mechanism selection.
    pub const fn processor(&self) -> &eredu_runtime::SelectedProcessorExecution {
        &self.processor
    }

    /// Selected routed text, state, weight, and bank realization.
    pub const fn routed(
        &self,
    ) -> &crate::PreparedRoutedTextArchitecture<
        crate::composite_execution::PreparedCompositeArchitecture<A>,
    > {
        &self.routed
    }

    /// Architecture capability estimate presented by the shared session.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Normalized model-type label presented by the shared session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into routed session-construction inputs.
    pub fn into_parts(
        self,
    ) -> (
        crate::PreparedRoutedTextArchitecture<
            crate::composite_execution::PreparedCompositeArchitecture<A>,
        >,
        eredu_runtime::SelectedProcessorExecution,
        C,
    ) {
        (self.routed, self.processor, self.admission)
    }
}

impl<A, C> PreparedCompositeTextArchitecture<A, C> {
    /// Exact architecture, artifact, and processor requirements.
    pub const fn requirements(&self) -> &CompositeTextRequirements {
        &self.requirements
    }

    /// Authoritative processor mechanism selection.
    pub const fn processor(&self) -> &eredu_runtime::SelectedProcessorExecution {
        &self.processor
    }

    /// Architecture-derived identity coupled to the shared text session.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        self.contract.prompt_cache_identity()
    }

    /// Exact selected text execution realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        self.contract.selected()
    }

    /// Architecture capability estimate presented by the shared session.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Normalized model-type label presented by the shared session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into generic session-construction inputs.
    pub fn into_parts(
        self,
    ) -> (
        crate::composite_execution::PreparedCompositeArchitecture<A>,
        Option<crate::composite_execution::PreparedCompositeArchitecture<A>>,
        eredu_runtime::PreparedReplicatedTextContract,
        eredu_runtime::SelectedProcessorExecution,
        C,
    ) {
        (
            self.architecture,
            self.source_architecture,
            self.contract,
            self.processor,
            self.admission,
        )
    }
}

/// Backend-generic visitor over one exact replicated composite architecture.
pub trait CompositeTextArchitectureVisitor<B, S>: Sized
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Records that validated dispatch is about to construct architecture modules.
    fn construction_started(&mut self);

    /// Binds one statically known composite architecture to generic mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;

    /// Binds one routed composite architecture to the existing planned provider.
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

pub(crate) enum CompositeConfig<'a> {
    Gemma4(&'a crate::gemma4::FamilyConfig),
    Inkling(&'a crate::inkling::ModelArgs),
    Muse(&'a crate::muse_glimmer::DecoderConfig),
    QwenVl(&'a crate::qwen::vl::ModelArgs),
    QwenHybrid(&'a crate::qwen::hybrid::ParsedHybridConfig),
}

impl CompositeConfig<'_> {
    fn requirement_config(&self) -> EligibleConfig<'_> {
        match self {
            Self::Gemma4(args) => EligibleConfig::Gemma4(args),
            Self::Inkling(args) => EligibleConfig::Inkling(args),
            Self::Muse(args) => EligibleConfig::MuseGlimmer(args),
            Self::QwenVl(args) => EligibleConfig::QwenVl(args),
            Self::QwenHybrid(args) => EligibleConfig::QwenCompositeHybrid(args),
        }
    }

    fn architecture_identity(&self) -> String {
        match self {
            Self::Gemma4(args) => args.architecture_fingerprint(),
            Self::Inkling(args) => args.architecture_fingerprint(),
            Self::Muse(args) => args.architecture_fingerprint(),
            Self::QwenVl(args) => crate::qwen::vl::prompt_cache_architecture_fingerprint(args),
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(args)
            }
        }
    }

    fn input_modalities(&self) -> InputModalities {
        match self {
            Self::Gemma4(args) => args.input_modalities(),
            Self::Inkling(args) => args.input_modalities(),
            Self::Muse(args) => InputModalities {
                text: true,
                image: args.vision_config.is_some(),
                audio: false,
                video: args.vision_config.is_some(),
            },
            Self::QwenVl(_) | Self::QwenHybrid(_) => InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            },
        }
    }

    fn routed(&self) -> bool {
        match self {
            Self::Gemma4(args) => args.text.num_experts.is_some_and(|count| count > 0),
            Self::Inkling(args) => args.text_config.has_sparse_moe_layers(),
            Self::Muse(args) => args.num_experts > 0,
            Self::QwenVl(args) => args.text.is_moe(),
            Self::QwenHybrid(args) => args.text.is_moe(),
        }
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, String> {
        match self {
            Self::Gemma4(args) => {
                crate::gemma4::state_layout(&args.text).map_err(|error| error.to_string())
            }
            Self::Inkling(args) => {
                let target =
                    crate::inkling::state_layout(args).map_err(|error| error.to_string())?;
                let prediction =
                    crate::inkling::mtp_state_layout(args).map_err(|error| error.to_string())?;
                crate::inkling::composite_state_layout(&target, prediction.as_ref())
                    .map_err(|error| error.to_string())
            }
            Self::Muse(args) => {
                crate::muse_glimmer::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenVl(args) => {
                crate::qwen::vl::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::state_layout(&args.text).map_err(|error| error.to_string())
            }
        }
    }

    fn graph_and_units(
        &self,
    ) -> Result<
        (
            eredu_runtime::ExecutionGraph,
            eredu_runtime::ExecutionUnitLayout,
            Vec<eredu_runtime::ArchitectureGroupTransport>,
        ),
        String,
    > {
        use eredu_runtime::{
            ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
            ArchitectureMergeDestination, ArchitectureParallelSubgroup, ExecutionGraph,
            ExecutionGroupSpec, ExecutionUnitLayout,
        };
        let media = |kind, roles: Vec<&str>| ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind,
            first_owner_static_roles: roles.into_iter().map(str::to_owned).collect(),
            last_owner_static_roles: Vec::new(),
            merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
            parallel_subgroup: Some(ArchitectureParallelSubgroup::TensorSharded),
            request_optional: true,
        };
        let (graph, counts, transports) = match self {
            Self::Gemma4(args) => {
                let graph = ExecutionGraph::new(
                    vec![
                        ExecutionGroupSpec::root(crate::gemma4::model::VISION_EXECUTION_GROUP),
                        ExecutionGroupSpec::root(crate::gemma4::model::AUDIO_EXECUTION_GROUP),
                        ExecutionGroupSpec::with_dependencies(
                            crate::gemma4::model::TEXT_EXECUTION_GROUP,
                            [
                                crate::gemma4::model::VISION_EXECUTION_GROUP,
                                crate::gemma4::model::AUDIO_EXECUTION_GROUP,
                            ],
                        ),
                    ],
                    crate::gemma4::model::TEXT_EXECUTION_GROUP,
                )
                .map_err(|error| error.to_string())?;
                let counts = vec![
                    args.vision
                        .as_ref()
                        .map_or(0, |vision| vision.num_hidden_layers as usize),
                    args.audio
                        .as_ref()
                        .map_or(0, |audio| audio.num_hidden_layers as usize),
                    args.text.num_hidden_layers(),
                ];
                let decoder = ArchitectureGroupTransport {
                    placement: ArchitectureGroupPlacement::Pipeline,
                    kind: ArchitectureGroupKind::Decoder,
                    first_owner_static_roles: vec![
                        "embedding".into(),
                        "per_layer_embedding".into(),
                        "per_layer_projection".into(),
                        "per_layer_norm".into(),
                    ],
                    last_owner_static_roles: if args.text.tie_word_embeddings {
                        vec!["norm".into(), "embedding".into()]
                    } else {
                        vec!["norm".into(), "output".into()]
                    },
                    merge_destination: ArchitectureMergeDestination::LastOwner,
                    parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
                    request_optional: false,
                };
                (
                    graph,
                    counts,
                    vec![
                        media(
                            ArchitectureGroupKind::VisionEncoder,
                            args.vision
                                .as_ref()
                                .map_or_else(Vec::new, |_| vec!["vision", "vision_projection"]),
                        ),
                        media(
                            ArchitectureGroupKind::AudioEncoder,
                            args.audio
                                .as_ref()
                                .map_or_else(Vec::new, |_| vec!["audio", "audio_projection"]),
                        ),
                        decoder,
                    ],
                )
            }
            Self::Inkling(args) => {
                let graph = ExecutionGraph::new(
                    vec![
                        ExecutionGroupSpec::root(crate::inkling::model::VISION_EXECUTION_GROUP),
                        ExecutionGroupSpec::root(crate::inkling::model::AUDIO_EXECUTION_GROUP),
                        ExecutionGroupSpec::with_dependencies(
                            crate::inkling::model::TEXT_EXECUTION_GROUP,
                            [
                                crate::inkling::model::VISION_EXECUTION_GROUP,
                                crate::inkling::model::AUDIO_EXECUTION_GROUP,
                            ],
                        ),
                    ],
                    crate::inkling::model::TEXT_EXECUTION_GROUP,
                )
                .map_err(|error| error.to_string())?;
                let decoder = ArchitectureGroupTransport {
                    placement: ArchitectureGroupPlacement::Pipeline,
                    kind: ArchitectureGroupKind::Decoder,
                    first_owner_static_roles: vec!["embedding".into(), "embedding_norm".into()],
                    last_owner_static_roles: vec![
                        "norm".into(),
                        "output".into(),
                        crate::inkling::model::MTP_STATIC_ROLE.into(),
                    ],
                    merge_destination: ArchitectureMergeDestination::LastOwner,
                    parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
                    request_optional: false,
                };
                (
                    graph,
                    vec![
                        args.vision_config
                            .as_ref()
                            .map_or(0, |vision| vision.num_hidden_layers as usize),
                        0,
                        args.text_config.num_hidden_layers as usize,
                    ],
                    vec![
                        media(ArchitectureGroupKind::VisionEncoder, vec!["vision"]),
                        ArchitectureGroupTransport {
                            placement: ArchitectureGroupPlacement::Pipeline,
                            kind: ArchitectureGroupKind::AudioEncoder,
                            first_owner_static_roles: vec!["audio".into()],
                            last_owner_static_roles: Vec::new(),
                            merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
                            parallel_subgroup: None,
                            request_optional: true,
                        },
                        decoder,
                    ],
                )
            }
            Self::Muse(args) => {
                let graph = ExecutionGraph::chain([
                    crate::muse_glimmer::model::VISION_EXECUTION_GROUP,
                    crate::muse_glimmer::model::TEXT_EXECUTION_GROUP,
                ])
                .map_err(|error| error.to_string())?;
                (
                    graph,
                    vec![
                        args.vision_config
                            .as_ref()
                            .map_or(0, |vision| vision.layer_count()),
                        args.num_hidden_layers as usize,
                    ],
                    vec![
                        media(ArchitectureGroupKind::VisionEncoder, vec!["vision"]),
                        crate::transport::decoder(),
                    ],
                )
            }
            Self::QwenVl(args) => {
                let graph = ExecutionGraph::new(
                    vec![
                        ExecutionGroupSpec::root(crate::qwen::vl::VISION_EXECUTION_GROUP),
                        ExecutionGroupSpec::with_dependencies(
                            crate::qwen::vl::TEXT_EXECUTION_GROUP,
                            [crate::qwen::vl::VISION_EXECUTION_GROUP],
                        ),
                    ],
                    crate::qwen::vl::TEXT_EXECUTION_GROUP,
                )
                .map_err(|error| error.to_string())?;
                (
                    graph,
                    vec![
                        args.vision.layer_count(),
                        args.text.num_hidden_layers as usize,
                    ],
                    vec![
                        media(ArchitectureGroupKind::VisionEncoder, vec!["vision"]),
                        crate::transport::decoder(),
                    ],
                )
            }
            Self::QwenHybrid(args) => {
                let vision = args.vision.as_ref().ok_or_else(|| {
                    "conditional Qwen composite omitted vision geometry".to_owned()
                })?;
                let graph = ExecutionGraph::new(
                    vec![
                        ExecutionGroupSpec::root(crate::qwen::hybrid::VISION_EXECUTION_GROUP),
                        ExecutionGroupSpec::with_dependencies(
                            crate::decoder::TARGET_EXECUTION_GROUP,
                            [crate::qwen::hybrid::VISION_EXECUTION_GROUP],
                        ),
                    ],
                    crate::decoder::TARGET_EXECUTION_GROUP,
                )
                .map_err(|error| error.to_string())?;
                (
                    graph,
                    vec![vision.layer_count(), args.text.num_hidden_layers as usize],
                    vec![
                        media(ArchitectureGroupKind::VisionEncoder, vec!["vision"]),
                        crate::transport::decoder(),
                    ],
                )
            }
        };
        let units = ExecutionUnitLayout::new(&graph, counts).map_err(|error| error.to_string())?;
        Ok((graph, units, transports))
    }
}

pub(crate) fn composite_config(
    plan: &ArtifactArchitecturePlan,
) -> Result<Option<CompositeConfig<'_>>, ReplicatedTextRequirementsError> {
    use crate::gguf_companion::GgufMediaProjectorConfig;
    if let Some(projector) = plan.gguf_media_projector() {
        return Ok(Some(match projector.model() {
            GgufMediaProjectorConfig::Gemma4(args) => CompositeConfig::Gemma4(args),
            GgufMediaProjectorConfig::Inkling(args) => CompositeConfig::Inkling(args),
            GgufMediaProjectorConfig::MuseGlimmer(args) => CompositeConfig::Muse(args),
            GgufMediaProjectorConfig::Qwen3Vl(args) => CompositeConfig::QwenVl(args),
            GgufMediaProjectorConfig::Qwen35(args) => CompositeConfig::QwenHybrid(args),
            GgufMediaProjectorConfig::Qwen3VlPending(_)
            | GgufMediaProjectorConfig::Qwen35Pending(_) => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                    "composite GGUF media token identities are unresolved".into(),
                ))
            }
        }));
    }
    let config = match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(SafetensorsModelConfig::Gemma4(args)), None) => Some(CompositeConfig::Gemma4(args)),
        (Some(SafetensorsModelConfig::Inkling(args)), None) => Some(CompositeConfig::Inkling(args)),
        (Some(SafetensorsModelConfig::MuseGlimmer(args)), None) => {
            Some(CompositeConfig::Muse(args))
        }
        (Some(SafetensorsModelConfig::QwenVl(args)), None) => Some(CompositeConfig::QwenVl(args)),
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None) if args.vision.is_some() => {
            Some(CompositeConfig::QwenHybrid(args))
        }
        (None, Some(GgufModelConfig::Gemma4(args))) => Some(CompositeConfig::Gemma4(args)),
        (None, Some(GgufModelConfig::Inkling(args))) => Some(CompositeConfig::Inkling(args)),
        (None, Some(GgufModelConfig::MuseGlimmer(args))) => Some(CompositeConfig::Muse(args)),
        (None, Some(GgufModelConfig::QwenHybrid(args))) if args.vision.is_some() => {
            Some(CompositeConfig::QwenHybrid(args))
        }
        _ => None,
    };
    Ok(config)
}

pub(crate) fn partitioned_boundary_schema(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<eredu_runtime::BoundaryWireSchema, ReplicatedTextRequirementsError> {
    if let Some(config) = composite_config(inspection.architecture_plan())? {
        return config
            .requirement_config()
            .partitioned_boundary_schema(topology)
            .map_err(ReplicatedTextRequirementsError::InvalidArchitecture);
    }
    let plan = inspection.architecture_plan();
    let config = match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(SafetensorsModelConfig::Qwen(args)), None) if args.is_moe() => {
            EligibleConfig::Qwen(args)
        }
        (Some(SafetensorsModelConfig::Lfm2(args)), None) if args.has_sparse_moe_layers() => {
            EligibleConfig::Lfm2(args)
        }
        (None, Some(GgufModelConfig::Lfm2(args))) if args.has_sparse_moe_layers() => {
            EligibleConfig::Lfm2(args)
        }
        (Some(SafetensorsModelConfig::KimiLinear(args)), None)
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::KimiLinear(args)
        }
        (None, Some(GgufModelConfig::KimiLinear(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::KimiLinear(args)
        }
        (Some(SafetensorsModelConfig::NemotronH(args)), None)
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::NemotronH(args)
        }
        (None, Some(GgufModelConfig::NemotronH(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::NemotronH(args)
        }
        (Some(SafetensorsModelConfig::GptOss(args)), None) => EligibleConfig::GptOss(args),
        (Some(SafetensorsModelConfig::DeepSeekV3(args)), None) => EligibleConfig::DeepSeekV3(args),
        (Some(SafetensorsModelConfig::DeepSeekV4(args)), None) => EligibleConfig::DeepSeekV4(args),
        (None, Some(GgufModelConfig::Qwen(args))) if args.is_moe() => EligibleConfig::Qwen(args),
        (None, Some(GgufModelConfig::GptOss(args))) => EligibleConfig::GptOss(args),
        (None, Some(GgufModelConfig::DeepSeekV3(args))) => EligibleConfig::DeepSeekV3(args),
        (None, Some(GgufModelConfig::DeepSeekV4(args))) => EligibleConfig::DeepSeekV4(args),
        _ => eligible_config(plan)?,
    };
    config
        .partitioned_boundary_schema(topology)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)
}

fn composite_processor_requirements(
    plan: &ArtifactArchitecturePlan,
    config: &CompositeConfig<'_>,
) -> Result<eredu_runtime::ProcessorExecutionRequirements, ReplicatedTextRequirementsError> {
    use eredu_runtime::{ModalityProcessorRequirements, ProcessorPrimitive as Primitive};

    let modalities = config.input_modalities();
    let maximum_dimension = u64::from(u32::try_from(i32::MAX).expect("i32::MAX fits u32"));
    let text = ModalityProcessorRequirements::new(
        InputModality::Text,
        [Primitive::TensorU32],
        true,
        !matches!(config, CompositeConfig::Muse(_)),
        maximum_dimension,
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    let mut requirements = vec![text];
    let raw = plan.has_processor();
    if modalities.image {
        let primitives = raw.then_some(match config {
            CompositeConfig::Inkling(_) => vec![Primitive::RgbNormalize, Primitive::TensorF32],
            CompositeConfig::Muse(_) => vec![
                Primitive::RgbResizeLanczos3,
                Primitive::RgbNormalize,
                Primitive::TensorF32,
                Primitive::TensorI32,
            ],
            CompositeConfig::Gemma4(_) | CompositeConfig::QwenVl(_) => vec![
                Primitive::RgbResizeBicubic,
                Primitive::RgbNormalize,
                Primitive::TensorF32,
                Primitive::TensorI32,
            ],
            CompositeConfig::QwenHybrid(_) => vec![
                Primitive::RgbResizeBicubic,
                Primitive::RgbNormalize,
                Primitive::TensorF32,
                Primitive::TensorI32,
            ],
        });
        requirements.push(
            ModalityProcessorRequirements::new(
                InputModality::Image,
                primitives.unwrap_or_default(),
                true,
                !matches!(
                    config,
                    CompositeConfig::QwenVl(_) | CompositeConfig::Muse(_)
                ),
                maximum_dimension,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?,
        );
    }
    if modalities.video {
        let primitives = raw.then_some(vec![
            Primitive::VideoSampling,
            if matches!(config, CompositeConfig::Muse(_)) {
                Primitive::RgbResizeLanczos3
            } else {
                Primitive::RgbResizeBicubic
            },
            Primitive::RgbNormalize,
            Primitive::TensorF32,
            Primitive::TensorI32,
        ]);
        requirements.push(
            ModalityProcessorRequirements::new(
                InputModality::Video,
                primitives.unwrap_or_default(),
                true,
                matches!(config, CompositeConfig::QwenHybrid(_)),
                maximum_dimension,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?,
        );
    }
    if modalities.audio {
        let mut primitives = if raw {
            vec![
                Primitive::AudioWindow,
                Primitive::AudioSpectrum,
                Primitive::AudioMelFilter,
                Primitive::AudioLogarithm,
                Primitive::TensorBool,
            ]
        } else {
            Vec::new()
        };
        if raw {
            primitives.push(if matches!(config, CompositeConfig::Inkling(_)) {
                Primitive::TensorI32
            } else {
                Primitive::TensorF32
            });
        }
        requirements.push(
            ModalityProcessorRequirements::new(
                InputModality::Audio,
                primitives,
                true,
                matches!(
                    config,
                    CompositeConfig::Gemma4(_) | CompositeConfig::Inkling(_)
                ),
                maximum_dimension,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?,
        );
    }
    eredu_runtime::ProcessorExecutionRequirements::new(requirements)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

/// Derives processor requirements for any admitted composite artifact, including
/// execution classes that use embedded prediction or distributed composition.
pub fn composite_processor_execution_requirements(
    plan: &ArtifactArchitecturePlan,
) -> Result<Option<eredu_runtime::ProcessorExecutionRequirements>, ReplicatedTextRequirementsError>
{
    let Some(config) = composite_config(plan)? else {
        return Ok(None);
    };
    composite_processor_requirements(plan, &config).map(Some)
}

fn replicated_expert_topology() -> Result<eredu_core::ParallelRankTopology, String> {
    eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|error| error.to_string())?,
        0,
    )
    .map_err(|error| error.to_string())
}

fn gemma4_replicated_expert_realization_plan(
    args: &crate::gemma4::FamilyConfig,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, String> {
    let global_experts = usize::try_from(args.text.num_experts.unwrap_or_default())
        .map_err(|error| error.to_string())?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new(crate::gemma4::model::TEXT_EXECUTION_GROUP)
            .map_err(|error| error.to_string())?;
    let mut specs = BTreeMap::new();
    for (layer, policy) in args.text.layer_schedule.iter().enumerate() {
        if policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe {
            specs.insert(
                (owner_group.clone(), layer),
                crate::gemma4::text::expert_bank_spec(&args.text, layer)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    crate::ExpertRealizationPlan::balanced(global_experts, replicated_expert_topology()?, specs)
        .map_err(|error| error.to_string())
}

fn muse_replicated_expert_realization_plan(
    args: &crate::muse_glimmer::DecoderConfig,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, String> {
    let global_experts = usize::try_from(args.num_experts).map_err(|error| error.to_string())?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new(crate::muse_glimmer::model::TEXT_EXECUTION_GROUP)
            .map_err(|error| error.to_string())?;
    let mut specs = BTreeMap::new();
    for layer in 0..usize::try_from(args.num_hidden_layers).map_err(|error| error.to_string())? {
        specs.insert(
            (owner_group.clone(), layer),
            crate::muse_glimmer::text::expert_bank_spec(args, layer)
                .map_err(|error| error.to_string())?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, replicated_expert_topology()?, specs)
        .map_err(|error| error.to_string())
}

fn inkling_replicated_expert_realization_plan(
    args: &crate::inkling::ModelArgs,
) -> Result<
    (
        crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        BTreeMap<usize, usize>,
    ),
    String,
> {
    let global_experts =
        usize::try_from(args.text_config.n_routed_experts).map_err(|error| error.to_string())?;
    let routed_routes =
        usize::try_from(args.text_config.num_experts_per_tok).map_err(|error| error.to_string())?;
    let shared_routes =
        usize::try_from(args.text_config.n_shared_experts).map_err(|error| error.to_string())?;
    let layers =
        usize::try_from(args.text_config.num_hidden_layers).map_err(|error| error.to_string())?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new(crate::inkling::model::TEXT_EXECUTION_GROUP)
            .map_err(|error| error.to_string())?;
    let mut specs = BTreeMap::new();
    let mut routes = BTreeMap::new();
    for (layer, policy) in args.text_config.layer_schedule.iter().enumerate() {
        if policy.feed_forward != crate::inkling::FeedForwardPolicy::SparseMoe {
            continue;
        }
        let (routed, shared) = crate::inkling::text::localized_expert_bank_specs(
            args,
            layer,
            &args.text_config,
            args.text_config.n_routed_experts,
        )
        .map_err(|error| error.to_string())?;
        let shared_unit = layers
            .checked_add(layer)
            .ok_or_else(|| "Inkling shared expert unit overflowed".to_owned())?;
        specs.insert((owner_group.clone(), layer), routed);
        specs.insert((owner_group.clone(), shared_unit), shared);
        routes.insert(layer, routed_routes);
        routes.insert(shared_unit, shared_routes);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, replicated_expert_topology()?, specs)
        .map(|plan| (plan, routes))
        .map_err(|error| error.to_string())
}

fn composite_routed_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: &CompositeConfig<'_>,
    execution: ReplicatedTextRequirements,
) -> Result<Option<crate::RoutedTextRequirements>, ReplicatedTextRequirementsError> {
    if !config.routed() {
        return Ok(None);
    }
    let recipe_source = inspection_recipe_source(inspection)?;
    let (owner_group, plan, catalog, routes_per_token) = match config {
        CompositeConfig::QwenVl(args) => {
            let plan =
                crate::qwen::replicated_expert_realization_plan(&args.text).map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
            let catalog = crate::qwen::expert_residency_catalog(recipe_source.as_ref(), &args.text)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (
                eredu_runtime::ExecutionGroupId::new(crate::qwen::vl::TEXT_EXECUTION_GROUP)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?,
                plan,
                catalog,
                usize::try_from(args.text.num_experts_per_tok).unwrap_or_default(),
            )
        }
        CompositeConfig::QwenHybrid(args) => {
            let plan = crate::qwen::hybrid::replicated_expert_realization_plan(&args.text)
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
            let catalog =
                crate::qwen::hybrid::expert_residency_catalog(recipe_source.as_ref(), &args.text)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (
                eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?,
                plan,
                catalog,
                usize::try_from(args.text.num_experts_per_tok).unwrap_or_default(),
            )
        }
        CompositeConfig::Gemma4(args) => {
            let owner_group =
                eredu_runtime::ExecutionGroupId::new(crate::gemma4::model::TEXT_EXECUTION_GROUP)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?;
            let plan = gemma4_replicated_expert_realization_plan(args).map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let catalog =
                crate::gemma4::expert_residency_catalog(recipe_source.as_ref(), &args.text)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (
                owner_group,
                plan,
                catalog,
                usize::try_from(args.text.top_k_experts.unwrap_or_default()).unwrap_or_default(),
            )
        }
        CompositeConfig::Muse(args) => {
            let owner_group = eredu_runtime::ExecutionGroupId::new(
                crate::muse_glimmer::model::TEXT_EXECUTION_GROUP,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let plan = muse_replicated_expert_realization_plan(args).map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let catalog =
                crate::muse_glimmer::expert_residency_catalog(recipe_source.as_ref(), args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (
                owner_group,
                plan,
                catalog,
                usize::try_from(args.num_experts_per_tok).unwrap_or_default(),
            )
        }
        CompositeConfig::Inkling(args) => {
            let owner_group =
                eredu_runtime::ExecutionGroupId::new(crate::inkling::model::TEXT_EXECUTION_GROUP)
                    .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
            let (plan, routes_by_unit) = inkling_replicated_expert_realization_plan(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            let catalog = crate::inkling::expert_residency_catalog(args, recipe_source.as_ref())
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            return crate::routed_text::gated_routed_text_requirements_with_routes(
                execution,
                owner_group,
                plan,
                catalog,
                routes_by_unit,
                recipe_source.as_ref(),
            )
            .map(Some)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            });
        }
    };
    crate::routed_text::gated_routed_text_requirements(
        execution,
        owner_group,
        plan,
        catalog,
        routes_per_token,
        recipe_source.as_ref(),
    )
    .map(Some)
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

/// Derives exact replicated composite requirements from an admitted artifact.
pub fn composite_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<CompositeTextRequirements, ReplicatedTextRequirementsError> {
    let config = composite_config(inspection.architecture_plan())?.ok_or({
        ReplicatedTextRequirementsError::Ineligible(ReplicatedTextIneligibility::Unrelated)
    })?;
    match &config {
        CompositeConfig::Inkling(args)
            if args
                .mtp_config
                .as_ref()
                .is_some_and(|mtp| mtp.num_nextn_predict_layers > 0) =>
        {
            return Err(ReplicatedTextIneligibility::EmbeddedPrediction.into())
        }
        CompositeConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
            return Err(ReplicatedTextIneligibility::EmbeddedPrediction.into())
        }
        _ => {}
    }
    let (execution_graph, execution_units, group_transports) = config
        .graph_and_units()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let state_layout = config
        .state_layout()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let processor = composite_processor_requirements(inspection.architecture_plan(), &config)?;
    let mut execution = replicated_text_requirements_for_structure(
        inspection,
        config.requirement_config(),
        Some((
            execution_graph.clone(),
            execution_units.clone(),
            group_transports.clone(),
            state_layout.clone(),
        )),
    )?;
    let routed = composite_routed_requirements(inspection, &config, execution.clone())?;
    if let Some(routed) = &routed {
        execution = routed.text().clone();
    }
    Ok(CompositeTextRequirements {
        architecture_identity: config.architecture_identity(),
        execution_graph,
        execution_units,
        group_transports,
        state_layout,
        input_modalities: config.input_modalities(),
        decoder: if config.routed() {
            CompositeTextDecoderStrategy::Routed
        } else {
            CompositeTextDecoderStrategy::Direct
        },
        raw_processor: inspection.architecture_plan().has_processor(),
        media_projector: inspection
            .architecture_plan()
            .gguf_media_projector()
            .is_some(),
        processor,
        execution,
        routed,
        inspection: inspection.clone(),
    })
}

pub(crate) fn qwen_vl_with_formats(
    args: &crate::qwen::vl::ModelArgs,
    formats: HashMap<String, LinearFormat>,
) -> Result<crate::qwen::vl::ModelArgs, String> {
    let (vision, text): (HashMap<_, _>, HashMap<_, _>) = formats
        .into_iter()
        .partition(|(name, _)| name.starts_with("model.visual."));
    let text = text
        .into_iter()
        .filter_map(|(name, format)| format.weight_quantization().map(|format| (name, format)))
        .collect();
    let mut target = crate::qwen::vl::with_checkpoint_formats(args, text, HashMap::new())?;
    target.vision.linear_formats = vision;
    target
        .vision
        .validate_for(crate::qwen::vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    Ok(target)
}

pub(crate) fn qwen_hybrid_composite_with_formats(
    args: &crate::qwen::hybrid::ParsedHybridConfig,
    formats: HashMap<String, LinearFormat>,
) -> Result<crate::qwen::hybrid::ParsedHybridConfig, String> {
    let (vision, text): (HashMap<_, _>, HashMap<_, _>) = formats
        .into_iter()
        .partition(|(name, _)| name.starts_with("model.visual."));
    let mut target = args.clone();
    target.text.quantization = None;
    target.text.fp8 = None;
    target.text.linear_formats = text;
    target.text.validate().map_err(|error| error.to_string())?;
    let target_vision = target.vision.as_mut().ok_or_else(|| {
        "conditional Qwen projector formats have no vision configuration".to_owned()
    })?;
    target_vision.linear_formats = vision;
    target_vision
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(|error| error.to_string())?;
    Ok(target)
}

#[allow(clippy::too_many_arguments)]
fn prepare_composite_architecture_handoff<B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    requirements: CompositeTextRequirements,
    selected: SelectedReplicatedTextRealization,
    processor: eredu_runtime::SelectedProcessorExecution,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    PreparedCompositeTextArchitecture<
        A,
        <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
    >,
    String,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    if selected.requirements() != requirements.execution() {
        return Err("selected composite execution differs from exact requirements".into());
    }
    if processor.requirements() != requirements.processor_execution() {
        return Err("selected composite processor differs from exact requirements".into());
    }
    let admission = architecture.admission_config();
    let architecture = crate::composite_execution::PreparedCompositeArchitecture::new(architecture);
    let source_architecture =
        source_architecture.map(crate::composite_execution::PreparedCompositeArchitecture::new);
    let contract = eredu_runtime::prepare_layered_text_contract::<_, B, S>(
        &architecture,
        source_architecture.as_ref(),
        selected,
        &prompt_cache_architecture_identity,
        eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
        context,
    )?;
    Ok(PreparedCompositeTextArchitecture {
        architecture,
        source_architecture,
        requirements,
        contract,
        processor,
        admission,
        capability_estimate,
        effective_model_type,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_routed_composite_architecture_handoff<B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    requirements: CompositeTextRequirements,
    selected: crate::SelectedRoutedTextRealization,
    processor: eredu_runtime::SelectedProcessorExecution,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    PreparedRoutedCompositeTextArchitecture<
        A,
        <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
    >,
    String,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + 'static,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
{
    if processor.requirements() != requirements.processor_execution() {
        return Err("selected composite processor differs from exact requirements".into());
    }
    let routed_requirements = requirements
        .routed_execution()
        .cloned()
        .ok_or_else(|| "routed composite selection has no grouped-bank requirements".to_owned())?;
    let admission = architecture.admission_config();
    let architecture = crate::composite_execution::PreparedCompositeArchitecture::new(architecture);
    let source_architecture =
        source_architecture.map(crate::composite_execution::PreparedCompositeArchitecture::new);
    let routed = crate::routed_text::prepare_gated_routed_architecture_handoff::<B, S, _>(
        architecture,
        source_architecture,
        routed_requirements,
        selected,
        capability_estimate.clone(),
        effective_model_type.clone(),
        prompt_cache_architecture_identity,
        context,
    )?;
    Ok(PreparedRoutedCompositeTextArchitecture {
        routed,
        requirements,
        processor,
        admission,
        capability_estimate,
        effective_model_type,
    })
}

/// Constructs and visits one selected replicated composite architecture.
pub fn visit_composite_text_architecture<B, S, V>(
    requirements: CompositeTextRequirements,
    selected: SelectedCompositeTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    V: CompositeTextArchitectureVisitor<B, S>,
{
    validate_store_handoff(requirements.execution(), store.as_ref())
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let retained = requirements.inspection.clone();
    let config = composite_config(retained.architecture_plan())
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let source_linear_formats = requirement_linear_formats(requirements.execution());
    let target_linear_formats =
        selected_linear_formats(requirements.execution(), selected.execution());
    let source_formats = requirement_formats(requirements.execution());
    let target_formats = selected_formats(selected.execution());
    let has_transform = selected_uses_transform(selected.execution());
    visitor.construction_started();
    macro_rules! visit_constructed {
        ($architecture:expr, $source:expr, $capability:expr, $model_type:expr, $cache_identity:expr) => {{
            match selected {
                SelectedCompositeTextRealization::Direct(selected) => {
                    let (execution, processor) = selected.into_parts();
                    let prepared = prepare_composite_architecture_handoff::<B, S, _>(
                        $architecture,
                        $source,
                        requirements,
                        execution,
                        processor,
                        $capability,
                        $model_type,
                        $cache_identity,
                        context,
                    )
                    .map_err(ReplicatedTextDispatchError::Architecture)?;
                    visitor
                        .visit(prepared, store)
                        .map_err(ReplicatedTextDispatchError::Backend)
                }
                SelectedCompositeTextRealization::Routed {
                    execution,
                    processor,
                } => {
                    let prepared = prepare_routed_composite_architecture_handoff::<B, S, _>(
                        $architecture,
                        $source,
                        requirements,
                        execution,
                        processor,
                        $capability,
                        $model_type,
                        $cache_identity,
                        context,
                    )
                    .map_err(ReplicatedTextDispatchError::Architecture)?;
                    visitor
                        .visit_routed(prepared, store)
                        .map_err(ReplicatedTextDispatchError::Backend)
                }
            }
        }};
    }
    match config {
        CompositeConfig::QwenVl(args) => {
            let capability = crate::capability::qwen_vl(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| qwen_vl_with_formats(args, source_linear_formats.clone()))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::qwen::vl::LayeredModel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = qwen_vl_with_formats(args, target_linear_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.effective_model_type().to_owned();
            let cache_identity = crate::qwen::vl::prompt_cache_architecture_fingerprint(&target);
            let architecture = crate::qwen::vl::LayeredModel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
        CompositeConfig::Gemma4(args) => {
            let output_projection_bias = requirements
                .execution()
                .parameters()
                .iter()
                .find(|parameter| parameter.name() == "model.audio_tower.output_proj.bias")
                .is_some_and(|parameter| {
                    !matches!(
                        parameter.presence(),
                        ReplicatedTextParameterPresence::OptionalAbsent
                    )
                });
            let mut exact_args = args.clone();
            if let Some(audio) = exact_args.audio.as_mut() {
                audio.output_projection_bias = output_projection_bias;
            }
            let capability = crate::capability::gemma4(&exact_args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| {
                    crate::gemma4::with_checkpoint_formats(&exact_args, source_formats.clone())
                })
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::gemma4::LayeredModel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target =
                crate::gemma4::with_checkpoint_formats(&exact_args, target_formats.clone())
                    .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.effective_model_type().to_owned();
            let cache_identity = target.architecture_fingerprint();
            let architecture = crate::gemma4::LayeredModel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
        CompositeConfig::Muse(args) => {
            let capability = crate::capability::muse_glimmer(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| crate::muse_glimmer::with_checkpoint_formats(args, source_formats.clone()))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::muse_glimmer::LayeredModel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = crate::muse_glimmer::with_checkpoint_formats(args, target_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.model_type.clone();
            let cache_identity = target.architecture_fingerprint();
            let architecture = crate::muse_glimmer::LayeredModel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
        CompositeConfig::Inkling(args) => {
            let capability = crate::capability::inkling(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| crate::inkling::with_checkpoint_formats(args, source_formats.clone()))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::inkling::LayeredModel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = crate::inkling::with_checkpoint_formats(args, target_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.model_type.clone();
            let cache_identity = target.architecture_fingerprint();
            let architecture = crate::inkling::LayeredModel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
        CompositeConfig::QwenHybrid(args) => {
            let capability = crate::capability::qwen_hybrid(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| qwen_hybrid_composite_with_formats(args, source_linear_formats.clone()))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = qwen_hybrid_composite_with_formats(args, target_linear_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.text.model_type.clone();
            let cache_identity =
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(&target);
            let architecture = crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(
                target, context,
            )
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
    }
}

/// Architecture-owned selection of replicated text or another execution class.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReplicatedTextExecutionClass {
    /// The admitted artifact uses the complete replicated-text lifecycle.
    Replicated(ReplicatedTextRequirements),
    /// The admitted artifact uses replicated text with architecture-routed units.
    Routed(crate::RoutedTextRequirements),
    /// The admitted artifact uses replicated text with composite ingress.
    Composite(CompositeTextRequirements),
    /// The admitted artifact uses a distinct architecture execution class.
    Other(NonReplicatedTextArchitecture),
}

/// Supplies backend policy selection without exposing the semantic branch to
/// the backend adapter.
pub trait ReplicatedTextExecutionClassDispatcher: Sized {
    /// Backend-private policy retained for replicated-text construction.
    type Replicated;
    /// Backend-private policy retained for replicated routed construction.
    type Routed;
    /// Backend-private policy retained for replicated composite construction.
    type Composite;
    /// Backend-private policy retained for an excluded construction path.
    type Other;
    /// Backend policy-selection failure.
    type Error;

    /// Selects mechanisms for exact replicated-text requirements.
    fn replicated(
        self,
        requirements: ReplicatedTextRequirements,
    ) -> Result<Self::Replicated, Self::Error>;

    /// Selects low-level mechanisms for exact replicated routed requirements.
    fn routed(
        self,
        requirements: crate::RoutedTextRequirements,
    ) -> Result<Self::Routed, Self::Error>;

    /// Selects mechanisms for exact replicated composite requirements.
    fn composite(
        self,
        requirements: CompositeTextRequirements,
    ) -> Result<Self::Composite, Self::Error>;

    /// Selects mechanisms for an architecture-owned exclusion. The semantic
    /// reason remains private to architecture dispatch.
    fn other(self) -> Result<Self::Other, Self::Error>;
}

enum SelectedReplicatedTextExecutionKind<R, T, C, O> {
    Replicated(R),
    Routed(T),
    Composite(C),
    Other(O),
}

/// Opaque architecture-owned semantic selection carrying backend-private
/// construction policy.
pub struct SelectedReplicatedTextExecution<R, T, C, O> {
    kind: SelectedReplicatedTextExecutionKind<R, T, C, O>,
}

impl<R, T, C, O> std::fmt::Debug for SelectedReplicatedTextExecution<R, T, C, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedReplicatedTextExecution")
            .finish_non_exhaustive()
    }
}

impl<R: Clone, T: Clone, C: Clone, O: Clone> Clone for SelectedReplicatedTextExecution<R, T, C, O> {
    fn clone(&self) -> Self {
        Self {
            kind: match &self.kind {
                SelectedReplicatedTextExecutionKind::Replicated(value) => {
                    SelectedReplicatedTextExecutionKind::Replicated(value.clone())
                }
                SelectedReplicatedTextExecutionKind::Routed(value) => {
                    SelectedReplicatedTextExecutionKind::Routed(value.clone())
                }
                SelectedReplicatedTextExecutionKind::Composite(value) => {
                    SelectedReplicatedTextExecutionKind::Composite(value.clone())
                }
                SelectedReplicatedTextExecutionKind::Other(value) => {
                    SelectedReplicatedTextExecutionKind::Other(value.clone())
                }
            },
        }
    }
}

/// Backend adapter invoked by an opaque selected execution value.
pub trait SelectedReplicatedTextExecutionDispatcher<R, T, C, O>: Sized {
    /// Completed adapter output.
    type Output;
    /// Adapter failure.
    type Error;

    /// Enters the generic replicated-text adapter.
    fn replicated(self, selected: R) -> Result<Self::Output, Self::Error>;

    /// Enters generic replicated routed composition.
    fn routed(self, selected: T) -> Result<Self::Output, Self::Error>;

    /// Enters generic replicated composite composition.
    fn composite(self, selected: C) -> Result<Self::Output, Self::Error>;

    /// Enters the existing excluded execution adapter.
    fn other(self, selected: O) -> Result<Self::Output, Self::Error>;
}

impl<R, T, C, O> SelectedReplicatedTextExecution<R, T, C, O> {
    /// Invokes exactly one backend adapter while keeping the semantic branch
    /// inside architecture-owned code.
    pub fn dispatch<D>(self, dispatcher: D) -> Result<D::Output, D::Error>
    where
        D: SelectedReplicatedTextExecutionDispatcher<R, T, C, O>,
    {
        match self.kind {
            SelectedReplicatedTextExecutionKind::Replicated(selected) => {
                dispatcher.replicated(selected)
            }
            SelectedReplicatedTextExecutionKind::Routed(selected) => dispatcher.routed(selected),
            SelectedReplicatedTextExecutionKind::Composite(selected) => {
                dispatcher.composite(selected)
            }
            SelectedReplicatedTextExecutionKind::Other(selected) => dispatcher.other(selected),
        }
    }
}

/// Selects the semantic execution class and lets a backend choose only the
/// mechanisms appropriate to the architecture-owned result.
pub fn dispatch_replicated_text_execution_class<D>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    topology: Option<eredu_core::topology::ParallelTopology>,
    dispatcher: D,
) -> Result<
    SelectedReplicatedTextExecution<D::Replicated, D::Routed, D::Composite, D::Other>,
    ReplicatedTextDispatchError<D::Error>,
>
where
    D: ReplicatedTextExecutionClassDispatcher,
{
    if topology.is_some_and(|topology| !topology.is_replicated()) {
        return dispatcher
            .other()
            .map(|selected| SelectedReplicatedTextExecution {
                kind: SelectedReplicatedTextExecutionKind::Other(selected),
            })
            .map_err(ReplicatedTextDispatchError::Backend);
    }
    match replicated_text_execution_class(inspection)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
    {
        ReplicatedTextExecutionClass::Replicated(requirements) => dispatcher
            .replicated(requirements)
            .map(|selected| SelectedReplicatedTextExecution {
                kind: SelectedReplicatedTextExecutionKind::Replicated(selected),
            })
            .map_err(ReplicatedTextDispatchError::Backend),
        ReplicatedTextExecutionClass::Routed(requirements) => dispatcher
            .routed(requirements)
            .map(|selected| SelectedReplicatedTextExecution {
                kind: SelectedReplicatedTextExecutionKind::Routed(selected),
            })
            .map_err(ReplicatedTextDispatchError::Backend),
        ReplicatedTextExecutionClass::Composite(requirements) => dispatcher
            .composite(requirements)
            .map(|selected| SelectedReplicatedTextExecution {
                kind: SelectedReplicatedTextExecutionKind::Composite(selected),
            })
            .map_err(ReplicatedTextDispatchError::Backend),
        ReplicatedTextExecutionClass::Other(_) => dispatcher
            .other()
            .map(|selected| SelectedReplicatedTextExecution {
                kind: SelectedReplicatedTextExecutionKind::Other(selected),
            })
            .map_err(ReplicatedTextDispatchError::Backend),
    }
}

/// Proof that an admitted artifact must not enter replicated-text construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NonReplicatedTextArchitecture {
    reason: ReplicatedTextIneligibility,
}

impl NonReplicatedTextArchitecture {
    /// Returns why the architecture registry selected another execution class.
    pub const fn reason(&self) -> &ReplicatedTextIneligibility {
        &self.reason
    }
}

/// Selects the semantic execution class and derives replicated requirements when applicable.
pub fn replicated_text_execution_class(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextExecutionClass, ReplicatedTextRequirementsError> {
    match replicated_text_requirements(inspection) {
        Ok(requirements) => Ok(ReplicatedTextExecutionClass::Replicated(requirements)),
        Err(ReplicatedTextRequirementsError::Ineligible(reason)) => {
            if reason == ReplicatedTextIneligibility::CompositeInput {
                match composite_text_requirements(inspection) {
                    Ok(requirements) => {
                        return Ok(ReplicatedTextExecutionClass::Composite(requirements))
                    }
                    Err(ReplicatedTextRequirementsError::Ineligible(
                        ReplicatedTextIneligibility::EmbeddedPrediction,
                    )) => {
                        return Ok(ReplicatedTextExecutionClass::Other(
                            NonReplicatedTextArchitecture {
                                reason: ReplicatedTextIneligibility::EmbeddedPrediction,
                            },
                        ))
                    }
                    Err(error) => return Err(error),
                }
            }
            if reason == ReplicatedTextIneligibility::Routed {
                match crate::routed_text_requirements(inspection) {
                    Ok(requirements) => {
                        return Ok(ReplicatedTextExecutionClass::Routed(requirements))
                    }
                    Err(crate::RoutedTextRequirementsError::Invalid(detail)) => {
                        return Err(ReplicatedTextRequirementsError::InvalidArchitecture(detail))
                    }
                    Err(crate::RoutedTextRequirementsError::Ineligible) => {}
                }
            }
            Ok(ReplicatedTextExecutionClass::Other(
                NonReplicatedTextArchitecture { reason },
            ))
        }
        Err(error) => Err(error),
    }
}

impl From<eredu_checkpoint::Error> for ReplicatedTextRequirementsError {
    fn from(error: eredu_checkpoint::Error) -> Self {
        Self::InvalidArchitecture(error.to_string())
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::lfm2::LayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::decoder::LayeredInput { tokens, mask }
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::kimi_linear::LayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::decoder::LayeredInput { tokens, mask }
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::nemotron_h::LayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::nemotron_h::EmbeddedInput::Target { tokens, mask }
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::qwen::hybrid::LayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::qwen::hybrid::EmbeddedInput::Target { tokens, mask }
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::deepseek::v3::Model<B>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::deepseek::mtp::EmbeddedInput::Target { tokens, mask }
    }
}

impl<B, S> ReplicatedTextArchitecture<B, S> for crate::deepseek::v4::Model<B>
where
    B: eredu_nn::HyperNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        crate::deepseek::mtp::EmbeddedInput::Target { tokens, mask }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    fn config(model_type: &str) -> serde_json::Value {
        let architecture = match model_type {
            "llama" => "LlamaForCausalLM",
            "mistral" => "MistralForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            "gpt_oss" => "GptOssForCausalLM",
            _ => unreachable!("test model type"),
        };
        serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })
    }

    fn gemma4_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"gemma4", "tie_word_embeddings":false,
            "text_config": {
                "model_type":"gemma4_text", "hidden_size":16,
                "num_hidden_layers":2, "intermediate_size":32,
                "num_attention_heads":4, "num_key_value_heads":2,
                "head_dim":4, "rms_norm_eps":0.000001, "vocab_size":64,
                "max_position_embeddings":128,
                "layer_types":["full_attention","full_attention"],
                "enable_moe_block":true, "num_experts":2,
                "top_k_experts":1, "moe_intermediate_size":8
            }
        })
    }

    #[test]
    fn standalone_gemma4_remains_a_composite_graph() {
        let config = gemma4_config();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let architecture = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap();
        assert_eq!(
            safetensors_replicated_text_eligibility(architecture),
            Err(ReplicatedTextIneligibility::CompositeInput)
        );
    }

    fn inspected(
        model_type: &str,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        inspected_config(config(model_type))
    }

    fn inspected_config(
        config: serde_json::Value,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let architecture = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap();
        let plan = architecture.checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                let dtype = match &constraint.dtype {
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U8,
                    ) => Dtype::U8,
                    _ => Dtype::F32,
                };
                let element_bytes = if dtype == Dtype::U8 { 1 } else { 4 };
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    vec![0_u8; elements * element_bytes],
                )
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
        let inspection = crate::configuration::inspect_artifact(root.path()).unwrap();
        (root, inspection)
    }

    fn heterogeneous_config(model_type: &str) -> serde_json::Value {
        match model_type {
            "lfm2" => serde_json::json!({
                "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
                "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
                "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
            }),
            "kimi_linear" => serde_json::json!({
                "model_type":"kimi_linear","vocab_size":64,"hidden_size":12,
                "num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,
                "intermediate_size":16,"head_dim":4,"model_max_length":64,
                "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],
                    "num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
                "num_experts":2,"moe_intermediate_size":8,"kv_lora_rank":6,
                "qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
                "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
                "routed_scaling_factor":1.0,"first_k_dense_replace":2,
                "num_expert_group":1,"topk_group":1
            }),
            "nemotron_h" => serde_json::json!({
                "model_type":"nemotron_h", "vocab_size":64, "hidden_size":16,
                "intermediate_size":24, "num_hidden_layers":4,
                "hybrid_override_pattern":"M*-M", "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
                "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
                "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
                "moe_intermediate_size":8,"moe_shared_expert_intermediate_size":8,
                "num_experts_per_tok":2,"n_group":2,"topk_group":1,
                "num_nextn_predict_layers":0
            }),
            "qwen3_5_text" => serde_json::json!({
                "model_type":"qwen3_5_text", "vocab_size":64, "hidden_size":32,
                "num_hidden_layers":2, "mtp_num_hidden_layers":0,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "max_position_embeddings":128,"linear_conv_kernel_dim":4,
                "linear_key_head_dim":8,"linear_value_head_dim":8,
                "linear_num_key_heads":2,"linear_num_value_heads":4,
                "intermediate_size":48,"moe_intermediate_size":16,
                "shared_expert_intermediate_size":24,"num_experts_per_tok":0,
                "num_experts":0,"layer_types":["linear_attention","full_attention"]
            }),
            _ => unreachable!("heterogeneous test model type"),
        }
    }

    #[test]
    fn conditional_qwen_preserves_exact_block_fp8_component_formats() {
        let args = crate::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_5",
            "image_token_id": 60,
            "video_token_id": 61,
            "text_config": {
                "model_type": "qwen3_5_text", "vocab_size": 64, "hidden_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8,
                "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "intermediate_size": 64,
                "layer_types": ["linear_attention", "full_attention"]
            },
            "vision_config": {
                "depth": 2, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 4, "num_position_embeddings": 16,
                "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "out_hidden_size": 32
            }
        }))
        .unwrap();
        let fp8 = LinearFormat::E4M3BlockFp8(
            eredu_checkpoint::BlockFp8Format::new(
                128,
                128,
                eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
            )
            .unwrap(),
        );
        let text = "model.layers.1.self_attn.q_proj.weight";
        let vision = "model.visual.blocks.0.attn.qkv.weight";
        let target = qwen_hybrid_composite_with_formats(
            &args,
            HashMap::from([(text.into(), fp8), (vision.into(), fp8)]),
        )
        .unwrap();

        assert_eq!(target.text.linear_format(text), fp8);
        assert_eq!(target.vision.unwrap().linear_format(vision), fp8);
    }

    #[test]
    fn exact_state_access_follows_each_valid_schedule_shape() {
        let mut cases = Vec::new();
        let mut lfm = heterogeneous_config("lfm2");
        lfm["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((lfm, ReplicatedTextStateAccess::KeyValue));
        let mut lfm = heterogeneous_config("lfm2");
        lfm["layer_types"] = serde_json::json!(["conv", "conv"]);
        cases.push((lfm, ReplicatedTextStateAccess::Fixed));

        let mut kimi = heterogeneous_config("kimi_linear");
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([1, 2]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([]);
        cases.push((kimi, ReplicatedTextStateAccess::Fixed));
        let mut kimi = heterogeneous_config("kimi_linear");
        kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
        kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
        cases.push((kimi, ReplicatedTextStateAccess::CompressedAttention));

        for (pattern, expected) in [
            ("****", ReplicatedTextStateAccess::KeyValue),
            ("MMMM", ReplicatedTextStateAccess::Fixed),
            ("----", ReplicatedTextStateAccess::Stateless),
        ] {
            let mut nemo = heterogeneous_config("nemotron_h");
            nemo["hybrid_override_pattern"] = pattern.into();
            cases.push((nemo, expected));
        }
        let mut qwen = heterogeneous_config("qwen3_5_text");
        qwen["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        cases.push((qwen, ReplicatedTextStateAccess::KeyValue));
        let mut qwen = heterogeneous_config("qwen3_5_text");
        qwen["layer_types"] = serde_json::json!(["linear_attention", "linear_attention"]);
        cases.push((qwen, ReplicatedTextStateAccess::Fixed));

        for (config, expected) in cases {
            let (_root, inspection) = inspected_config(config);
            let requirements = replicated_text_requirements(&inspection).unwrap();
            assert_eq!(requirements.state_access(), expected);
        }
    }

    #[test]
    fn gpt_oss_is_admitted_as_gated_routed_text() {
        let (_root, inspection) = inspected_config(serde_json::json!({
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "layer_types": ["sliding_attention"],
            "quantization_config": {"quant_method": "mxfp4"},
            "swiglu_limit": 7.0
        }));
        let ReplicatedTextExecutionClass::Routed(requirements) =
            replicated_text_execution_class(&inspection).unwrap()
        else {
            panic!("GPT-OSS must select routed text");
        };
        assert_eq!(
            requirements.text().grouped_operations(),
            &[eredu_runtime::GroupedOperationRequirement::GatedProduct]
        );
        assert_eq!(requirements.plan().global_group_count(), 2);
        for name in [
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_bias",
        ] {
            let parameter = requirements
                .text()
                .parameters()
                .iter()
                .find(|parameter| parameter.name() == name)
                .unwrap_or_else(|| panic!("missing routed GPT-OSS bias {name}"));
            assert_eq!(
                parameter.role(),
                ReplicatedTextParameterRole::LinearBias,
                "wrong routed GPT-OSS topology role for {name}"
            );
        }
    }

    #[test]
    fn exact_llama_and_dense_qwen_artifacts_derive_policy_independent_requirements() {
        for model_type in ["llama", "mistral", "qwen2", "qwen3"] {
            let (_root, inspection) = inspected(model_type);
            let requirements = replicated_text_requirements(&inspection).unwrap();
            let requests = [
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::FullyResident,
                    eredu_runtime::CacheResidencyPolicy::Device,
                ),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(
                        eredu_runtime::LayerwiseLoadOptions::default(),
                    ),
                    eredu_runtime::CacheResidencyPolicy::Device,
                )
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                }),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(
                        eredu_runtime::DenseDiskStreamLoadOptions::default(),
                    ),
                    eredu_runtime::CacheResidencyPolicy::Paged(
                        eredu_runtime::PagedCacheOptions::new(4, 4096, 4096, 1).unwrap(),
                    ),
                )
                .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap())
                .with_quantization(eredu_core::QuantizationRequest::MxFp4)
                .with_session(eredu_core::SessionCapabilities::new(true, true, true))
                .with_prompt_cache(true)
                .with_exact_completion(true),
            ];
            for request in requests {
                assert_eq!(
                    requirements,
                    replicated_text_requirements(&inspection).unwrap()
                );
                assert!(matches!(
                    request.residency(),
                    eredu_runtime::LayerWeightResidency::FullyResident
                        | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                ));
            }
            assert_eq!(requirements.execution_graph().groups().len(), 1);
            assert_eq!(requirements.state_layout().len(), 1);
            assert!(!requirements.parameters().is_empty());
            assert!(requirements.parameters().iter().all(|parameter| matches!(
                parameter.source_encoding(),
                Some(SourceTensorEncoding::Safetensors(StoredDtype::F32)) | None
            )));
            assert!(requirements.parameters().iter().any(|parameter| parameter
                .logical_shape()
                .len()
                == 1
                && parameter
                    .transform_target(eredu_core::QuantizationRequest::Affine {
                        group_size: 16,
                        bits: 4,
                    })
                    .unwrap()
                    .is_none()));
        }
    }

    #[test]
    fn exact_parameter_topology_preserves_geometry_ownership_aliases_and_ties() {
        let (_root, inspection) = inspected("qwen3");
        let requirements = replicated_text_requirements(&inspection).unwrap();
        let mut identities = BTreeSet::new();
        for parameter in requirements.parameters() {
            assert!(identities.insert(parameter.name()));
            assert!(!parameter.logical_shape().is_empty());
            match parameter.presence() {
                ReplicatedTextParameterPresence::Required
                | ReplicatedTextParameterPresence::OptionalPresent => {
                    let source = parameter.sources().first().unwrap();
                    let descriptor = inspection.tensors().get(source).unwrap();
                    assert_eq!(
                        parameter.physical_shape(),
                        Some(descriptor.shape.as_slice())
                    );
                    assert!(parameter.source_encoding().is_some());
                    let physical = parameter.physical_sources();
                    assert_eq!(physical.len(), 1);
                    assert_eq!(physical[0].tensor(), source);
                    assert_eq!(physical[0].output(), source);
                    let shards = inspection.safetensors_shards().unwrap();
                    let expected_shard = shards
                        .tensor_locations()
                        .and_then(|locations| locations.get(source))
                        .unwrap_or(&shards.payload_paths()[0]);
                    assert_eq!(physical[0].shard(), expected_shard);
                }
                ReplicatedTextParameterPresence::OptionalAbsent => {
                    assert!(parameter.sources().is_empty());
                    assert!(parameter.physical_sources().is_empty());
                    assert!(parameter.source_encoding().is_none());
                }
                ReplicatedTextParameterPresence::Tied { .. }
                | ReplicatedTextParameterPresence::Derived { .. } => {}
                _ => unreachable!("test covers every current presence category"),
            }
            assert_eq!(
                matches!(
                    parameter.transform_constraint(),
                    ParameterTransformConstraint::Linear { .. }
                ),
                parameter.role() == ReplicatedTextParameterRole::LinearWeight
            );
            if let ReplicatedTextParameterOwner::ExecutionUnit { group, unit } = parameter.owner() {
                assert_eq!(group, crate::decoder::TEXT_DECODER_EXECUTION_GROUP);
                assert!(*unit < requirements.execution_units().len());
            }
        }
        assert!(requirements.parameters().iter().any(|parameter| {
            parameter.logical_shape().len() == 1
                && parameter.role() == ReplicatedTextParameterRole::Normalization
                && parameter.transform_constraint() == ParameterTransformConstraint::None
        }));
        let mut tied_config = config("llama");
        tied_config["tie_word_embeddings"] = serde_json::json!(true);
        let (_root, tied_inspection) = inspected_config(tied_config);
        let tied = replicated_text_requirements(&tied_inspection).unwrap();
        let output = tied
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == "lm_head.weight")
            .unwrap();
        assert!(matches!(
            output.presence(),
            ReplicatedTextParameterPresence::Tied { target }
                if target == "model.embed_tokens.weight"
        ));
        assert!(output.sources().is_empty());
        assert!(output.physical_sources().is_empty());
    }

    #[test]
    fn routed_qwen_selects_the_distinct_routed_execution_class() {
        let mut config = config("qwen3_moe");
        config["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
        config["num_experts"] = serde_json::json!(2);
        config["num_experts_per_tok"] = serde_json::json!(2);
        config["moe_intermediate_size"] = serde_json::json!(8);
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert!(matches!(
            eligible_config(resolved.architecture_plan()),
            Err(ReplicatedTextIneligibility::Routed)
        ));

        let (_root, inspection) = inspected_config(config);
        let requirements = crate::routed_text_requirements(&inspection).unwrap();
        for request in [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        ] {
            assert_eq!(
                requirements,
                crate::routed_text_requirements(&inspection).unwrap()
            );
            assert!(matches!(
                request.parameter_banks(),
                eredu_runtime::ParameterBankResidency::WithLayer
                    | eredu_runtime::ParameterBankResidency::IndependentCache(_)
            ));
        }
        assert_eq!(requirements.owner_group().as_str(), "text_decoder");
        let plan = requirements.plan().gated().unwrap();
        assert_eq!(plan.global_expert_count(), 2);
        assert_eq!(plan.unit_specs().len(), 1);
        assert_eq!(requirements.catalog().units().len(), 2);
        assert_eq!(requirements.routes_per_token(), 2);
        assert_eq!(
            requirements.text().grouped_operations(),
            [eredu_runtime::GroupedOperationRequirement::GatedProduct]
        );

        let lowerings = requirements
            .text()
            .parameters()
            .iter()
            .filter(|parameter| parameter.has_lowering_source())
            .map(|parameter| {
                eredu_runtime::WeightLoweringCapability::new(
                    parameter
                        .lowering_descriptor(parameter.native_executable())
                        .unwrap(),
                    eredu_runtime::WeightLoweringKind::Direct,
                )
            })
            .collect();
        let state: Vec<_> = requirements
            .text()
            .state_layout()
            .layers()
            .iter()
            .enumerate()
            .flat_map(|(layer, policy)| {
                policy.components().into_iter().map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
            })
            .collect();
        let base_capabilities = eredu_runtime::BackendMechanismCapabilities::new(
            requirements.text().operators(),
            lowerings,
            vec![eredu_runtime::WeightResidencyMechanism::Resident],
            eredu_runtime::StateMechanismCapabilities::new(state)
                .with_transactions(true, true)
                .with_reset(true),
        )
        .with_grouped_operations([eredu_runtime::GroupedOperationRequirement::GatedProduct]);
        let text_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        );
        let resident_request = crate::RoutedTextSelectionRequest::new(
            text_request.clone(),
            eredu_runtime::WeightResidency::fully_resident(),
        )
        .unwrap();
        let resident = crate::select_routed_text_realization(
            &requirements,
            &resident_request,
            &base_capabilities,
        )
        .unwrap();
        assert_eq!(
            resident.bank_residency(),
            eredu_runtime::ParameterBankResidency::WithLayer
        );

        let addressable_request = crate::RoutedTextSelectionRequest::new(
            text_request,
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        )
        .unwrap();
        let denied = crate::select_routed_text_realization(
            &requirements,
            &addressable_request,
            &base_capabilities,
        )
        .unwrap_err();
        assert_eq!(
            denied.issues(),
            [
                "indexed selection and movement",
                "independently addressable storage"
            ]
        );
        let addressable_capabilities = base_capabilities
            .with_indexed_movement(true)
            .with_addressable_storage(eredu_runtime::AddressableStorageCapabilities::new(
                true,
                true,
                true,
                u64::MAX,
            ));
        let addressable = crate::select_routed_text_realization(
            &requirements,
            &addressable_request,
            &addressable_capabilities,
        )
        .unwrap();
        assert!(matches!(
            addressable.bank_residency(),
            eredu_runtime::ParameterBankResidency::IndependentCache(_)
        ));
        let member_bytes = requirements
            .catalog()
            .units()
            .iter()
            .filter_map(crate::ExpertResidencyUnit::byte_len)
            .max()
            .unwrap();
        let too_small = eredu_runtime::ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::default(),
            member_bytes,
            member_bytes,
        )
        .unwrap();
        let bounded_request = crate::RoutedTextSelectionRequest::new(
            resident_request.text().clone(),
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                too_small,
            ),
        )
        .unwrap();
        let error = crate::select_routed_text_realization(
            &requirements,
            &bounded_request,
            &addressable_capabilities,
        )
        .expect_err("undersized compact bank was admitted");
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("one routed token row") && issue.contains("2 routes")));
    }

    #[test]
    fn embedded_prediction_is_a_distinct_ineligibility() {
        let config = serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 2,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "num_nextn_predict_layers": 1
        });
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert!(matches!(
            eligible_config(resolved.architecture_plan()),
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        ));

        let (_root, inspection) = inspected_config(config);
        let request = crate::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            0,
            1,
            32,
            eredu_runtime::PipelineActivationDtype::Float32,
        )
        .unwrap()
        .with_completion_policy(partitioned_test_completion_policy());
        let error = crate::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            request,
            CollectPartitionedAdmission,
        )
        .unwrap_err();
        assert!(error.to_string().contains("embedded prediction"));
    }

    #[test]
    fn partitioned_topology_remains_a_caller_selection_choice() {
        let (_root, inspection) = inspected("llama");
        let requirements = replicated_text_requirements(&inspection).unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        )
        .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());
        assert!(!request.topology().unwrap().is_replicated());
        assert_eq!(
            requirements,
            replicated_text_requirements(&inspection).unwrap()
        );
    }

    #[derive(Debug)]
    enum CollectedPartitionedAdmission {
        Direct(crate::partitioned_execution::DirectPartitionedAdmission),
        Routed(crate::partitioned_execution::RoutedPartitionedAdmission),
        Composite(crate::partitioned_execution::CompositePartitionedAdmission),
    }

    struct CollectPartitionedAdmission;

    fn partitioned_test_completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    impl crate::partitioned_execution::PartitionedAdmissionDispatcher for CollectPartitionedAdmission {
        type Output = CollectedPartitionedAdmission;
        type Error = std::convert::Infallible;

        fn direct(
            self,
            requirements: crate::partitioned_execution::DirectPartitionedAdmission,
        ) -> Result<Self::Output, Self::Error> {
            Ok(CollectedPartitionedAdmission::Direct(requirements))
        }

        fn routed(
            self,
            requirements: crate::partitioned_execution::RoutedPartitionedAdmission,
        ) -> Result<Self::Output, Self::Error> {
            Ok(CollectedPartitionedAdmission::Routed(requirements))
        }

        fn composite(
            self,
            requirements: crate::partitioned_execution::CompositePartitionedAdmission,
        ) -> Result<Self::Output, Self::Error> {
            Ok(CollectedPartitionedAdmission::Composite(requirements))
        }
    }

    #[test]
    fn partitioned_admission_project_exact_rank_and_opaque_communication() {
        let mut llama = config("llama");
        llama["num_key_value_heads"] = serde_json::json!(2);
        let (_root, inspection) = inspected_config(llama);
        let request = crate::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
            1,
            32,
            eredu_runtime::PipelineActivationDtype::Float32,
        )
        .unwrap()
        .with_completion_policy(partitioned_test_completion_policy());
        let selected = crate::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            request,
            CollectPartitionedAdmission,
        )
        .unwrap();
        let CollectedPartitionedAdmission::Direct(requirements) = selected else {
            panic!("dense Llama did not select direct partitioned execution");
        };
        assert_eq!(requirements.topology().global_rank(), 1);
        assert_eq!(requirements.communication().rank(), 1);
        assert_eq!(requirements.communication().groups().len(), 1);
        assert_eq!(
            requirements.communication().groups()[0].local_index(),
            Some(1)
        );
        assert_eq!(requirements.boundary().primary().shape(), &[1, 32, 8]);
        let operations = requirements.communication().groups()[0]
            .requirements()
            .operations();
        assert_eq!(operations.len(), 4);
        assert_eq!(
            operations[0].operation(),
            eredu_runtime::CommunicationOperation::AllReduceSum
        );
        assert_eq!(
            operations[1].operation(),
            eredu_runtime::CommunicationOperation::AllGatherUneven
        );
        assert_eq!(
            operations[2].operation(),
            eredu_runtime::CommunicationOperation::Broadcast
        );
        assert_eq!(
            operations[3].operation(),
            eredu_runtime::CommunicationOperation::FailureAgreement
        );
        let incomplete = eredu_runtime::CommunicationCapabilities::new([operations[0].clone()])
            .unwrap()
            .with_completion_capabilities(
                eredu_runtime::CommunicationCompletionCapabilities::new([
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                ])
                .unwrap(),
            );
        assert!(incomplete
            .validate_manifest(requirements.communication())
            .unwrap_err()
            .to_string()
            .contains("AllGather"));
        assert_eq!(
            requirements.activation_dtype(),
            eredu_runtime::PipelineActivationDtype::Float32
        );
    }

    #[test]
    fn partitioned_admission_reject_data_before_dispatch() {
        let (_root, inspection) = inspected("llama");
        let request = crate::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 2).unwrap(),
            0,
            1,
            32,
            eredu_runtime::PipelineActivationDtype::Float32,
        )
        .unwrap()
        .with_completion_policy(partitioned_test_completion_policy());
        let error = crate::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            request,
            CollectPartitionedAdmission,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("data-parallel execution is not supported"));
    }

    #[test]
    fn routed_partition_admits_expert_axis_with_exact_exchange_contract() {
        let mut config = config("qwen3_moe");
        config["num_experts"] = serde_json::json!(2);
        config["num_experts_per_tok"] = serde_json::json!(1);
        config["moe_intermediate_size"] = serde_json::json!(8);
        let (_root, inspection) = inspected_config(config);
        let request = crate::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(1, 1, 2, 1).unwrap(),
            0,
            1,
            16,
            eredu_runtime::PipelineActivationDtype::Float16,
        )
        .unwrap()
        .with_completion_policy(partitioned_test_completion_policy());
        let selected = crate::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            request,
            CollectPartitionedAdmission,
        )
        .unwrap();
        let CollectedPartitionedAdmission::Routed(requirements) = selected else {
            panic!("Qwen3-MoE EP must retain routed admission");
        };
        let expert_group = requirements
            .expert_group()
            .expect("EP admission must select an opaque expert group");
        let descriptor = requirements
            .communication()
            .groups()
            .iter()
            .find(|group| group.id() == expert_group)
            .expect("selected expert group must be present");
        assert_eq!(descriptor.members(), [0, 1]);
        assert_eq!(descriptor.local_index(), Some(0));
        assert_eq!(
            descriptor
                .requirements()
                .operations()
                .iter()
                .map(eredu_runtime::CommunicationOperationRequirement::operation)
                .collect::<Vec<_>>(),
            [
                eredu_runtime::CommunicationOperation::AllGatherEven,
                eredu_runtime::CommunicationOperation::VariableAllToAll,
            ]
        );
    }

    #[test]
    fn composite_partition_uses_architecture_boundary_and_pipeline_routes() {
        let (_root, inspection) = inspected_config(gemma4_config());
        let request = crate::partitioned_execution::PartitionedSelectionRequest::new(
            eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap(),
            0,
            2,
            12,
            eredu_runtime::PipelineActivationDtype::Bfloat16,
        )
        .unwrap()
        .with_completion_policy(partitioned_test_completion_policy());
        let selected = crate::partitioned_execution::dispatch_partitioned_admission(
            &inspection,
            request,
            CollectPartitionedAdmission,
        )
        .unwrap();
        let CollectedPartitionedAdmission::Composite(requirements) = selected else {
            panic!("Gemma 4 did not select composite partitioned execution");
        };
        assert_eq!(requirements.boundary().identity(), "gemma4.text");
        assert_eq!(requirements.boundary().primary().shape(), &[2, 12, 16]);
        assert!(!requirements.communication().routes().is_empty());
        assert_eq!(
            requirements.boundary_routes().len(),
            requirements.communication().routes().len()
        );
        for (selected, descriptor) in requirements
            .boundary_routes()
            .iter()
            .zip(requirements.communication().routes())
        {
            assert_eq!(selected.route().route, descriptor.id());
            assert_eq!(selected.route().source_rank, descriptor.source());
            assert_eq!(selected.route().destination_rank, descriptor.destination());
            assert_eq!(selected.schema().identity(), "gemma4.text");
            assert_eq!(
                descriptor.boundary_contract().unwrap().schema(),
                selected.schema().identity()
            );
        }
        assert!(requirements
            .communication()
            .routes()
            .iter()
            .all(|route| route.requirement().operation()
                == eredu_runtime::CommunicationOperation::SendReceive));
    }

    #[test]
    fn composite_partition_rejects_route_endpoint_schema_and_cardinality_drift() {
        let (_root, inspection) = inspected_config(gemma4_config());
        let request = || {
            crate::partitioned_execution::PartitionedSelectionRequest::new(
                eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap(),
                0,
                2,
                12,
                eredu_runtime::PipelineActivationDtype::Bfloat16,
            )
            .unwrap()
            .with_completion_policy(partitioned_test_completion_policy())
        };
        let admission = || {
            let selected = crate::partitioned_execution::dispatch_partitioned_admission(
                &inspection,
                request(),
                CollectPartitionedAdmission,
            )
            .unwrap();
            let CollectedPartitionedAdmission::Composite(requirements) = selected else {
                panic!("Gemma 4 did not select composite partitioned execution");
            };
            requirements
        };

        let mut cardinality = admission();
        cardinality.test_boundary_routes_mut().pop().unwrap();
        assert!(
            crate::partitioned_execution::validate_selected_boundary_routes(&cardinality)
                .unwrap_err()
                .contains("cardinality")
        );

        let mut rank = admission();
        rank.test_boundary_routes_mut()[0]
            .test_route_mut()
            .source_rank = usize::MAX;
        assert!(
            crate::partitioned_execution::validate_selected_boundary_routes(&rank)
                .unwrap_err()
                .contains("endpoints")
        );

        let mut group = admission();
        group.test_boundary_routes_mut()[0]
            .test_route_mut()
            .source_group = usize::MAX;
        assert!(
            crate::partitioned_execution::validate_selected_boundary_routes(&group)
                .unwrap_err()
                .contains("execution-graph edge")
        );

        let mut schema = admission();
        *schema.test_boundary_routes_mut()[0].test_schema_mut() =
            eredu_runtime::ArchitectureBoundary::wire_schema(
                &eredu_runtime::NoAuxiliaryBoundarySchema::new(1),
            )
            .unwrap()
            .resolve(2, 12)
            .unwrap();
        assert!(
            crate::partitioned_execution::validate_selected_boundary_routes(&schema)
                .unwrap_err()
                .contains("schema/cardinality")
        );
    }

    struct CountingClassDispatcher {
        replicated: std::rc::Rc<std::cell::Cell<usize>>,
        routed: std::rc::Rc<std::cell::Cell<usize>>,
        composite: std::rc::Rc<std::cell::Cell<usize>>,
        other: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl ReplicatedTextExecutionClassDispatcher for CountingClassDispatcher {
        type Replicated = &'static str;
        type Routed = &'static str;
        type Composite = &'static str;
        type Other = &'static str;
        type Error = String;

        fn replicated(
            self,
            _requirements: ReplicatedTextRequirements,
        ) -> Result<Self::Replicated, Self::Error> {
            self.replicated.set(self.replicated.get() + 1);
            Ok("replicated")
        }

        fn routed(
            self,
            _requirements: crate::RoutedTextRequirements,
        ) -> Result<Self::Routed, Self::Error> {
            self.routed.set(self.routed.get() + 1);
            Ok("routed")
        }

        fn composite(
            self,
            _requirements: CompositeTextRequirements,
        ) -> Result<Self::Composite, Self::Error> {
            self.composite.set(self.composite.get() + 1);
            Ok("composite")
        }

        fn other(self) -> Result<Self::Other, Self::Error> {
            self.other.set(self.other.get() + 1);
            Ok("other")
        }
    }

    struct CountingSelectedDispatcher {
        replicated: std::rc::Rc<std::cell::Cell<usize>>,
        routed: std::rc::Rc<std::cell::Cell<usize>>,
        composite: std::rc::Rc<std::cell::Cell<usize>>,
        other: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl
        SelectedReplicatedTextExecutionDispatcher<
            &'static str,
            &'static str,
            &'static str,
            &'static str,
        > for CountingSelectedDispatcher
    {
        type Output = &'static str;
        type Error = String;

        fn replicated(self, selected: &'static str) -> Result<Self::Output, Self::Error> {
            self.replicated.set(self.replicated.get() + 1);
            Ok(selected)
        }

        fn routed(self, selected: &'static str) -> Result<Self::Output, Self::Error> {
            self.routed.set(self.routed.get() + 1);
            Ok(selected)
        }

        fn composite(self, selected: &'static str) -> Result<Self::Output, Self::Error> {
            self.composite.set(self.composite.get() + 1);
            Ok(selected)
        }

        fn other(self, selected: &'static str) -> Result<Self::Output, Self::Error> {
            self.other.set(self.other.get() + 1);
            Ok(selected)
        }
    }

    #[test]
    fn architecture_dispatch_separates_ordinary_routed_and_composite_classes() {
        let mut routed = config("qwen3_moe");
        routed["num_experts"] = serde_json::json!(2);
        routed["num_experts_per_tok"] = serde_json::json!(1);
        routed["moe_intermediate_size"] = serde_json::json!(8);
        let (_root, excluded) = inspected_config(routed);
        let (_root, included) = inspected("llama");
        let (_root, composite) = inspected_config(gemma4_config());
        let composite_requirements = composite_text_requirements(&composite).unwrap();
        assert_eq!(composite_requirements.execution_graph().groups().len(), 3);
        assert_eq!(composite_requirements.execution_units().group_count(), 3);
        assert_eq!(composite_requirements.state_layout().len(), 2);
        assert_eq!(
            composite_requirements.decoder(),
            CompositeTextDecoderStrategy::Routed
        );
        assert_eq!(
            composite_requirements.input_modalities(),
            InputModalities::TEXT
        );
        assert!(!composite_requirements.has_raw_processor());
        assert_eq!(
            composite_requirements,
            composite_text_requirements(&composite).unwrap()
        );

        for (inspection, expected) in [
            (&excluded, "routed"),
            (&included, "replicated"),
            (&composite, "composite"),
        ] {
            let selected_replicated = std::rc::Rc::new(std::cell::Cell::new(0));
            let selected_routed = std::rc::Rc::new(std::cell::Cell::new(0));
            let selected_composite = std::rc::Rc::new(std::cell::Cell::new(0));
            let selected_other = std::rc::Rc::new(std::cell::Cell::new(0));
            let constructed_replicated = std::rc::Rc::new(std::cell::Cell::new(0));
            let constructed_routed = std::rc::Rc::new(std::cell::Cell::new(0));
            let constructed_composite = std::rc::Rc::new(std::cell::Cell::new(0));
            let constructed_other = std::rc::Rc::new(std::cell::Cell::new(0));
            let selected = dispatch_replicated_text_execution_class(
                inspection,
                None,
                CountingClassDispatcher {
                    replicated: selected_replicated.clone(),
                    routed: selected_routed.clone(),
                    composite: selected_composite.clone(),
                    other: selected_other.clone(),
                },
            )
            .unwrap();
            let actual = selected
                .dispatch(CountingSelectedDispatcher {
                    replicated: constructed_replicated.clone(),
                    routed: constructed_routed.clone(),
                    composite: constructed_composite.clone(),
                    other: constructed_other.clone(),
                })
                .unwrap();

            assert_eq!(actual, expected);
            if expected == "other" {
                assert_eq!(selected_replicated.get(), 0);
                assert_eq!(selected_routed.get(), 0);
                assert_eq!(selected_composite.get(), 0);
                assert_eq!(selected_other.get(), 1);
                assert_eq!(constructed_replicated.get(), 0);
                assert_eq!(constructed_routed.get(), 0);
                assert_eq!(constructed_composite.get(), 0);
                assert_eq!(constructed_other.get(), 1);
            } else if expected == "routed" {
                assert_eq!(selected_replicated.get(), 0);
                assert_eq!(selected_routed.get(), 1);
                assert_eq!(selected_composite.get(), 0);
                assert_eq!(selected_other.get(), 0);
                assert_eq!(constructed_replicated.get(), 0);
                assert_eq!(constructed_routed.get(), 1);
                assert_eq!(constructed_composite.get(), 0);
                assert_eq!(constructed_other.get(), 0);
            } else if expected == "replicated" {
                assert_eq!(selected_replicated.get(), 1);
                assert_eq!(selected_routed.get(), 0);
                assert_eq!(selected_composite.get(), 0);
                assert_eq!(selected_other.get(), 0);
                assert_eq!(constructed_replicated.get(), 1);
                assert_eq!(constructed_routed.get(), 0);
                assert_eq!(constructed_composite.get(), 0);
                assert_eq!(constructed_other.get(), 0);
            } else {
                assert_eq!(selected_replicated.get(), 0);
                assert_eq!(selected_routed.get(), 0);
                assert_eq!(selected_composite.get(), 1);
                assert_eq!(selected_other.get(), 0);
                assert_eq!(constructed_replicated.get(), 0);
                assert_eq!(constructed_routed.get(), 0);
                assert_eq!(constructed_composite.get(), 1);
                assert_eq!(constructed_other.get(), 0);
            }
        }
    }
}
