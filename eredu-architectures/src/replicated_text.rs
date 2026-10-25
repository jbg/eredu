//! Architecture-owned admission for replicated text execution.

pub mod artifact;
use artifact::{safetensors_inspection_recipe_source, safetensors_parameters};
mod profiles;
pub use profiles::{
    IdentityReplicatedTextVisitor, ReplicatedTextStateProfiles, ReplicatedTextVisitorConversion,
    SharedReplicatedTextVisitor,
};

use crate::decoder::Config as _;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype, WeightQuantization};
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
        AttentionState, CompressedReplicatedFamily, CompressedReplicatedKernel,
        CompressedReplicatedModel, FixedReplicatedFamily, FixedReplicatedKernel,
        FixedReplicatedModel, FixedState, MixedState, ReplicatedForwardContext, Stateless,
    },
    GgufArchitecture,
};

struct Lfm2Replicated;
struct KimiLinearReplicated;
struct DeepSeekV3DenseReplicated;
pub(crate) struct NemotronHReplicated;
pub(crate) struct QwenHybridReplicated;

impl<B: NeuralBackend> FixedReplicatedFamily<B> for Lfm2Replicated {
    const CHUNKED_PREFILL: bool = crate::lfm2::CHUNKED_TEXT_PREFILL;
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
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::lfm2::component_definition(config)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::lfm2::static_spec(config)
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
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            forward.mask.as_ref(),
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
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
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::kimi_linear::component_definition(config)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::kimi_linear::static_spec(config)
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
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            forward.mask.as_ref(),
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
    }
}

impl<B: eredu_nn::BlockwiseAttentionBackend> CompressedReplicatedFamily<B>
    for DeepSeekV3DenseReplicated
{
    type Config = crate::deepseek::V3Args;
    type Unit = crate::deepseek::block::DenseV3Block<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config.validate().map_err(eredu_nn::Error::backend)?;
        if config.num_nextn_predict_layers != 0
            || config
                .layer_schedule
                .iter()
                .any(|policy| *policy == crate::deepseek::LayerPolicy::SparseMoe)
        {
            return Err(eredu_nn::Error::backend(
                "replicated DeepSeek-V3 configuration contains routed or prediction layers",
            ));
        }
        Ok(())
    }
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::deepseek::v3::component_definition(config, None)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::deepseek::v3::static_spec(config)
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::deepseek::v3::state_identity(config, layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::deepseek::block::DenseV3Block::new(config, index, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        (!config.layer_schedule.is_empty()).then_some(0)
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
        unit.forward(hidden, forward.mask.as_ref(), Some(state), context)
    }

    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            forward.mask.as_ref(),
            Some(state),
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
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
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::kimi_linear::component_definition(config)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::kimi_linear::static_spec(config)
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
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        _forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
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
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::nemotron_h::component_definition(
            crate::nemotron_h::parameter_description(config).map_err(eredu_nn::Error::backend)?,
            crate::nemotron_h::state_layout(config).map_err(eredu_nn::Error::backend)?,
        )
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        let embedding = "model.embeddings.weight";
        crate::decoder::StaticModuleSpec {
            normalization_groups: None,
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
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            forward.mask.as_ref(),
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
    }
}

impl<B: NeuralBackend> FixedReplicatedFamily<B> for QwenHybridReplicated {
    const CHUNKED_PREFILL: bool = crate::qwen::hybrid::CHUNKED_TEXT_PREFILL;
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
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        crate::qwen::hybrid::component_definition(config, None)
    }
    fn static_spec(config: &Self::Config) -> crate::decoder::StaticModuleSpec {
        crate::decoder::StaticModuleSpec {
            normalization_groups: None,
            embedding_weight: "model.embed_tokens.weight".into(),
            normalization_weight: "model.norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocab_size,
            hidden_size: config.hidden_size,
            normalization_epsilon: config.rms_norm_eps,
            normalization_offset: 1.0,
            embedding_quantization: config
                .linear_format("model.embed_tokens.weight")
                .weight_quantization(),
            head_format: config.linear_format("lm_head.weight"),
            tied_head: config.tie_word_embeddings,
        }
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
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented(
            hidden,
            forward.mask.as_ref(),
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
    }
}

/// Architecture-owned reason that an admitted artifact cannot use replicated text composition.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextIneligibility {
    /// The architecture alternates authoritative context and tentative refinement.
    #[error("architecture requires iterative block-refinement execution")]
    BlockRefinement,
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

/// Family-blind backend visitor for an exact replicated prediction target.
pub trait ReplicatedPredictionTargetVisitor<B, S, M>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives an exact target only after its materialized extension was paired.
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Pairs one materialized extension with its exact replicated target before backend erasure.
pub fn visit_replicated_prediction_target_architecture<B, S, M, A, V>(
    prepared: PreparedReplicatedTextArchitecture<A>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    V: ReplicatedPredictionTargetVisitor<B, S, M>,
{
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension, store)
        .map_err(ReplicatedTextDispatchError::Backend)
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

/// Visits an additional typed architecture through the ordinary production
/// selection, materialization, and session boundary without extending a
/// concrete backend family registry.
#[allow(clippy::too_many_arguments)]
pub fn visit_replicated_text_extension_architecture<B, S, A, V>(
    architecture: A,
    source_architecture: Option<A>,
    selected: SelectedReplicatedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: impl Into<String>,
    prompt_cache_architecture_identity: impl Into<String>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    let requirements = selected.requirements().clone();
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor.construction_started();
    let prepared = prepare_architecture_handoff::<B, S, A>(
        architecture,
        source_architecture,
        requirements,
        selected,
        capability_estimate,
        effective_model_type.into(),
        prompt_cache_architecture_identity.into(),
        context,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor
        .visit(prepared, store)
        .map_err(ReplicatedTextDispatchError::Backend)
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
    let output = if requirements.block_refinement().is_some() {
        eredu_runtime::ReplicatedTextOutputSelection::AllSequencePositions
    } else {
        eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition
    };
    let contract =
        eredu_runtime::prepare_layered_text_contract_with_addressable_parameters::<A, B, S>(
            &architecture,
            source_architecture.as_ref(),
            selected,
            &prompt_cache_architecture_identity,
            output,
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
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    match eligible {
        EligibleConfig::Gemma2(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::gemma2(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = "gemma2".to_owned();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    crate::decoder::CompleteDecoderKernel::<B, crate::gemma2::ModelArgs>::new(
                        args.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_gemma2_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::gemma2::prompt_cache_architecture_fingerprint(&args);
            let architecture =
                crate::decoder::CompleteDecoderKernel::<B, crate::gemma2::ModelArgs>::new(
                    args, context,
                )
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
        EligibleConfig::Llama(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::llama(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    crate::decoder::CompleteDecoderKernel::<B, crate::llama::ModelArgs>::new(
                        args.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_llama_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::llama::prompt_cache_architecture_fingerprint(&args);
            let architecture =
                crate::decoder::CompleteDecoderKernel::<B, crate::llama::ModelArgs>::new(
                    args, context,
                )
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
        EligibleConfig::Nanbeige(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::nanbeige(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = "nanbeige".to_owned();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    crate::decoder::CompleteDecoderKernel::<B, crate::nanbeige::ModelArgs>::new(
                        args.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_nanbeige_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::nanbeige::prompt_cache_architecture_fingerprint(&args);
            let architecture =
                crate::decoder::CompleteDecoderKernel::<B, crate::nanbeige::ModelArgs>::new(
                    args, context,
                )
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
        EligibleConfig::K2Horizon(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::k2_horizon(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    crate::decoder::CompleteDecoderKernel::<
                        B,
                        crate::k2_horizon::ModelArgs,
                        crate::k2_horizon::DenseBlockFactory,
                    >::new(args.clone(), context)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_k2_horizon_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::k2_horizon::prompt_cache_architecture_fingerprint(&args);
            let architecture = crate::decoder::CompleteDecoderKernel::<
                B,
                crate::k2_horizon::ModelArgs,
                crate::k2_horizon::DenseBlockFactory,
            >::new(args, context)
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
                .then(|| {
                    crate::decoder::CompleteDecoderKernel::<
                        B,
                        crate::qwen::ModelArgs,
                        crate::qwen::QwenBlockFactory,
                    >::new(args.clone(), context)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::prompt_cache_architecture_fingerprint(&args);
            let architecture = crate::decoder::CompleteDecoderKernel::<
                B,
                crate::qwen::ModelArgs,
                crate::qwen::QwenBlockFactory,
            >::new(args, context)
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
        | EligibleConfig::DiffusionGemma(_)
        | EligibleConfig::NemotronLabsDiffusion(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::Muse(_)
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
    let definition = config
        .component_definition()
        .map_err(|error| error.to_string())?;
    if requirements.operators() != config.operators()
        || requirements.component_definition() != Some(&definition)
        || requirements.state_layout() != definition.state_layout()
        || requirements.state_access()
            != ReplicatedTextStateAccess::from_layout(definition.state_layout())
                .map_err(|error| error.to_string())?
    {
        return Err(
            "selected realization structure differs from the normalized architecture".into(),
        );
    }
    Ok(())
}

/// Source authority being validated; restricted primary views must not acquire auxiliaries.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum StoreHandoffScope {
    Primary,
    Complete,
}

/// Projects an already admitted artifact onto the exact selected parameter role.
/// Other roles retain their own source leases; native binding must not gain their
/// catalog authority merely because they share an underlying checkpoint file.
pub(crate) fn restrict_store_handoff(
    requirements: &ReplicatedTextRequirements,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    scope: StoreHandoffScope,
) -> Result<eredu_checkpoint::store::SharedCheckpointSource, String> {
    validate_store_handoff(requirements, store.as_ref(), scope)?;
    let keys = requirements
        .parameters()
        .iter()
        .chain(
            requirements
                .auxiliary_parameters()
                .iter()
                .filter(|_| scope == StoreHandoffScope::Complete),
        )
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    Ok(Arc::new(
        eredu_checkpoint::store::RestrictedCheckpointSource::including(
            store,
            "declared-execution-parameters",
            keys,
        )
        .map_err(|error| error.to_string())?,
    ))
}

pub(crate) fn validate_store_handoff(
    requirements: &ReplicatedTextRequirements,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    scope: StoreHandoffScope,
) -> Result<(), String> {
    for parameter in requirements.parameters().iter().chain(
        requirements
            .auxiliary_parameters()
            .iter()
            .filter(|_| scope == StoreHandoffScope::Complete),
    ) {
        let derived = matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::Derived { .. }
        );
        let mut lowering_key = None;
        let mut admitted_keys = BTreeSet::new();
        for source in parameter.physical_sources() {
            let key = source.catalog_key();
            let actual = store.source_provenance(key).map_err(|error| {
                format!("selected source catalog key {key:?} is unavailable: {error}")
            })?;
            let metadata = store.source_metadata(key).map_err(|error| {
                format!("selected source metadata for {key:?} is unavailable: {error}")
            })?;
            if actual.catalog_key != key
                || actual.physical_tensor != source.tensor()
                || actual.backing_shard.as_deref() != Some(source.shard())
                || actual.output != source.output()
                || &actual.source_encoding != source.source_encoding()
                || metadata.encoded_byte_len != source.encoded_byte_len()
            {
                return Err(format!(
                    "selected physical provenance for catalog key {key:?} differs from the admitted store"
                ));
            }
            if !derived
                && parameter
                    .source_encoding()
                    .is_some_and(|expected| expected != source.source_encoding())
            {
                return Err(format!(
                    "selected physical encoding for {:?} differs from the admitted store",
                    parameter.name()
                ));
            }
            lowering_key.get_or_insert(key);
            admitted_keys.insert(key);
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
        if scope == StoreHandoffScope::Primary
            && requirements
                .auxiliary_parameters()
                .iter()
                .any(|parameter| parameter.name() == target)
        {
            continue;
        }
        let parameter = requirements
            .parameters()
            .iter()
            .chain(requirements.auxiliary_parameters())
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
            .map(ReplicatedTextPhysicalSource::catalog_key)
            .collect::<BTreeSet<_>>();
        let consumed = recipe.source_keys().into_iter().collect::<BTreeSet<_>>();
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

pub(crate) fn requirement_formats(
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

pub(crate) fn selected_matrix_formats(
    requirements: &ReplicatedTextRequirements,
    selected: &SelectedReplicatedTextRealization,
) -> HashMap<String, LinearFormat> {
    let matrices = requirement_matrix_formats(requirements);
    selected
        .parameters()
        .iter()
        .filter(|parameter| matrices.contains_key(parameter.name()))
        .map(|parameter| (parameter.name().to_owned(), parameter.executable()))
        .collect()
}

fn requirement_matrix_formats(
    requirements: &ReplicatedTextRequirements,
) -> HashMap<String, LinearFormat> {
    requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            matches!(
                parameter.role(),
                ReplicatedTextParameterRole::LinearWeight | ReplicatedTextParameterRole::Embedding
            )
        })
        .map(|parameter| (parameter.name().to_owned(), parameter.native_executable()))
        .collect()
}

pub(crate) fn selected_gemma2_args(
    args: &crate::gemma2::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::gemma2::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::gemma2::with_checkpoint_formats(args, formats)
    }
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

pub(crate) fn selected_nanbeige_args(
    args: &crate::nanbeige::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::nanbeige::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::nanbeige::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn source_nanbeige_args(
    args: &crate::nanbeige::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::nanbeige::ModelArgs, String> {
    let formats = requirement_formats(selected.requirements());
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::nanbeige::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn source_gemma2_args(
    args: &crate::gemma2::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::gemma2::ModelArgs, String> {
    let formats = requirement_formats(selected.requirements());
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::gemma2::with_checkpoint_formats(args, formats)
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

pub(crate) fn selected_k2_horizon_args(
    args: &crate::k2_horizon::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::k2_horizon::ModelArgs, String> {
    crate::k2_horizon::with_checkpoint_formats(
        args,
        selected
            .parameters()
            .iter()
            .map(|p| (p.name().to_owned(), p.executable())),
    )
}

pub(crate) fn source_k2_horizon_args(
    args: &crate::k2_horizon::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::k2_horizon::ModelArgs, String> {
    crate::k2_horizon::with_checkpoint_formats(
        args,
        selected
            .requirements()
            .parameters()
            .iter()
            .map(|parameter| (parameter.name().to_owned(), parameter.native_executable())),
    )
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
    crate::deepseek::v3_with_checkpoint_formats(
        args,
        selected
            .parameters()
            .iter()
            .map(|parameter| (parameter.name().to_owned(), parameter.executable()))
            .collect(),
    )
}

pub(crate) fn source_deepseek_v3_args(
    args: &crate::deepseek::V3Args,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::deepseek::V3Args, String> {
    crate::deepseek::v3_with_checkpoint_formats(
        args,
        selected
            .requirements()
            .parameters()
            .iter()
            .map(|parameter| (parameter.name().to_owned(), parameter.native_executable()))
            .collect(),
    )
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

pub(crate) fn source_deepseek_v4_args(
    args: &crate::deepseek::V4Args,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::deepseek::V4Args, String> {
    crate::deepseek::v4_with_checkpoint_formats(
        args,
        selected
            .requirements()
            .parameters()
            .iter()
            .map(|parameter| (parameter.name().to_owned(), parameter.native_executable()))
            .collect(),
    )
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
        FixedReplicatedKernel::new(config, context)?.into_model()
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
        FixedReplicatedKernel::new(config, context)?.into_model()
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
        FixedReplicatedKernel::new(config, context)?.into_model()
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
        FixedReplicatedKernel::new(config, context)?.into_model()
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
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
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
        EligibleConfig::Gemma2(_)
        | EligibleConfig::Llama(_)
        | EligibleConfig::Nanbeige(_)
        | EligibleConfig::K2Horizon(_)
        | EligibleConfig::Qwen(_)
        | EligibleConfig::GptOss(_)
        | EligibleConfig::DeepSeekV3(_)
        | EligibleConfig::DeepSeekV4(_)
        | EligibleConfig::DiffusionGemma(_)
        | EligibleConfig::NemotronLabsDiffusion(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::Muse(_)
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

trait CompressedProfile<B, S, F>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    F: CompressedReplicatedFamily<B>,
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

impl<B, S, F> CompressedProfile<B, S, F> for crate::replicated_model::MixedCompressedState
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    F: CompressedReplicatedFamily<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Model = CompressedReplicatedModel<B, F>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedKernel::new(config, context)?.into_model()
    }
}

impl<B, S, F> CompressedProfile<B, S, F> for crate::replicated_model::CompressedState
where
    B: eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    F: CompressedReplicatedFamily<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Model = CompressedReplicatedModel<B, F, crate::replicated_model::CompressedState>;
    fn new(
        config: F::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedKernel::new(config, context)?.into_model()
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
    P: CompressedProfile<B, S, KimiLinearReplicated>
        + CompressedProfile<B, S, DeepSeekV3DenseReplicated>,
{
    let requirements = selected.requirements().clone();
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    macro_rules! visit_family {
        ($family:ty, $args:expr, $capability:path, $select:path, $identity:path) => {{
            visitor.construction_started();
            let args = $args;
            let capability_estimate = $capability(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| <P as CompressedProfile<B, S, $family>>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args =
                $select(args, &selected).map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity = $identity(&args);
            let architecture = <P as CompressedProfile<B, S, $family>>::new(args, context)
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
        }};
    }
    match eligible {
        EligibleConfig::KimiLinear(args) => visit_family!(
            KimiLinearReplicated,
            args,
            crate::capability::kimi_linear,
            selected_kimi_linear_args,
            crate::kimi_linear::prompt_cache_architecture_fingerprint
        ),
        EligibleConfig::DeepSeekV3(args) => visit_family!(
            DeepSeekV3DenseReplicated,
            args,
            crate::capability::deepseek_v3,
            selected_deepseek_v3_args,
            crate::deepseek::v3_architecture_fingerprint
        ),
        _ => Err(ReplicatedTextIneligibility::Unrelated.into()),
    }
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
    /// State exposing the combined attention, fixed and named-stream contract.
    type AttentionStreamState: LayerRuntimeState<
        B,
        LayerState: AttentionCache<B::Tensor>
                        + eredu_runtime::RuntimeStateComponents<B>
                        + eredu_runtime::RuntimeAppendStreams<B>,
    >;
    /// Visitor preserving the concrete combined-state access bounds.
    type AttentionStreamVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionStreamState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Consumes this adapter into its attention-with-streams visitor.
    fn into_attention_stream_visitor(self) -> Self::AttentionStreamVisitor;
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

/// Dispatches an ordinary key/value selection without requiring unrelated backend mechanisms.
pub(crate) fn dispatch_replicated_key_value_text_architecture<B, S, V>(
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
    if selected.state().access() != ReplicatedTextStateAccess::KeyValue {
        return Err(ReplicatedTextDispatchError::Architecture(format!(
            "selected state profile {:?} is not ordinary key/value attention",
            selected.state().access()
        )));
    }
    let eligible = eligible_config(plan)?;
    if matches!(
        eligible,
        EligibleConfig::Gemma2(_)
            | EligibleConfig::Llama(_)
            | EligibleConfig::Nanbeige(_)
            | EligibleConfig::K2Horizon(_)
            | EligibleConfig::Qwen(_)
    ) {
        visit_replicated_text_architecture(plan, selected, store, context, visitor)
    } else {
        visit_replicated_attention_state_text_architecture(plan, selected, store, context, visitor)
    }
}

/// Constructs an iterative stack under its independently selected read-only
/// context mechanism, preserving ordinary parameter and completion ownership.
fn visit_replicated_block_refinement_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    if let Some(config) = plan
        .safetensors_architecture()
        .and_then(|plan| match plan.model() {
            SafetensorsModelConfig::NemotronLabsDiffusion(config) => Some(config),
            _ => None,
        })
        .or_else(|| {
            plan.gguf_plan().and_then(|plan| match plan.model() {
                GgufModelConfig::NemotronLabsDiffusion(plan) => Some(plan.text_config()),
                _ => None,
            })
        })
    {
        let requirements = selected.requirements().clone();
        validate_plan_identity(
            &requirements,
            &EligibleConfig::NemotronLabsDiffusion(config),
        )
        .map_err(ReplicatedTextDispatchError::Architecture)?;
        validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        visitor.construction_started();
        let source = selected_uses_transform(&selected)
            .then(|| {
                crate::nemotron_labs_diffusion::model::TextKernel::<B>::new(config.clone(), context)
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = config
            .with_checkpoint_formats(selected_formats(&selected))
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let identity = target.architecture_fingerprint();
        let capability = crate::capability::nemotron_labs_diffusion(&target, false)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let effective_model_type = target.model_identity().to_owned();
        let architecture =
            crate::nemotron_labs_diffusion::model::TextKernel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let prepared = prepare_architecture_handoff::<B, S, _>(
            architecture,
            source,
            requirements,
            selected,
            capability,
            effective_model_type,
            identity,
            context,
        )
        .map_err(ReplicatedTextDispatchError::Architecture)?;
        return visitor
            .visit(prepared, store)
            .map_err(ReplicatedTextDispatchError::Backend);
    }
    let config = match (plan.safetensors_architecture(), plan.gguf_plan()) {
        (Some(plan), None) => match plan.model() {
            SafetensorsModelConfig::DiffusionGemma(config) => config,
            _ => return Err(ReplicatedTextIneligibility::Unrelated.into()),
        },
        (None, Some(plan)) => match plan.model() {
            GgufModelConfig::DiffusionGemma(config) => config,
            _ => return Err(ReplicatedTextIneligibility::Unrelated.into()),
        },
        _ => return Err(ReplicatedTextIneligibility::Unrelated.into()),
    };
    let requirements = selected.requirements().clone();
    validate_plan_identity(&requirements, &EligibleConfig::DiffusionGemma(config))
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor.construction_started();
    let source = selected_uses_transform(&selected)
        .then(|| crate::diffusion_gemma::model::TextKernel::<B>::new(config.clone(), context))
        .transpose()
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let mut target = config.clone();
    target.text.weight_quantization = None;
    target.text.quantized_weights = None;
    let formats = selected_formats(&selected);
    target.text.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
    let identity = target.architecture_fingerprint();
    let capability = crate::capability::diffusion_gemma(&target)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let architecture = crate::diffusion_gemma::model::TextKernel::<B>::new(target, context)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_architecture_handoff::<B, S, _>(
        architecture,
        source,
        requirements,
        selected,
        capability,
        "diffusion_gemma".into(),
        identity,
        context,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor
        .visit(prepared, store)
        .map_err(ReplicatedTextDispatchError::Backend)
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
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend,
    D: ReplicatedTextProfileDispatcher<B>,
    <D::AttentionState as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    <D::ComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_runtime::RuntimeStateComponents<B>,
    <D::AttentionComponentState as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    <D::CompressedState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor>,
    <D::CompressedComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    if selected.requirements().block_refinement().is_some() {
        return visit_replicated_block_refinement_architecture::<
            B,
            D::AttentionState,
            D::AttentionVisitor,
        >(
            plan,
            selected,
            store,
            context,
            dispatcher.into_attention_visitor(),
        );
    }
    let eligible = eligible_config(plan)?;
    let ordinary = matches!(
        eligible,
        EligibleConfig::Gemma2(_)
            | EligibleConfig::Llama(_)
            | EligibleConfig::Nanbeige(_)
            | EligibleConfig::K2Horizon(_)
            | EligibleConfig::Qwen(_)
    );
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

/// Supplies prediction-aware visitors while architecture dispatch selects a replicated state profile.
pub trait ReplicatedPredictionProfileDispatcher<B, M>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;
    /// State representation selected for the target.
    type State: LayerRuntimeState<B>;
    /// Exact prediction-aware visitor for that representation.
    type Visitor: ReplicatedPredictionTargetVisitor<
        B,
        Self::State,
        M,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the adapter into its exact typed visitor.
    fn into_visitor(self) -> Self::Visitor;
}

/// Constructs a replicated prediction target and pairs its extension before backend erasure.
pub fn dispatch_replicated_prediction_target_architecture<B, M, D>(
    plan: &ArtifactArchitecturePlan,
    selected: SelectedReplicatedTextRealization,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    dispatcher: D,
) -> Result<D::Output, ReplicatedTextDispatchError<D::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    D: ReplicatedPredictionProfileDispatcher<B, M>,
    <D::State as LayerRuntimeState<B>>::LayerState:
        AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let requirements = selected.requirements().clone();
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    validate_store_handoff(&requirements, store.as_ref(), StoreHandoffScope::Primary)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let args = match eligible {
        EligibleConfig::QwenHybrid(args) => {
            let capability_estimate = crate::capability::qwen_hybrid_text(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| {
                    <MixedState as FixedProfile<B, D::State, QwenHybridReplicated>>::new(
                        args.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_hybrid_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(&args);
            let mut architecture =
                <MixedState as FixedProfile<B, D::State, QwenHybridReplicated>>::new(args, context)
                    .map_err(|error| {
                        ReplicatedTextDispatchError::Architecture(error.to_string())
                    })?;
            architecture.kernel_mut().enable_prediction_capture();
            let prepared = prepare_architecture_handoff::<B, D::State, _>(
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
            return visit_replicated_prediction_target_architecture(
                prepared,
                extension,
                store,
                dispatcher.into_visitor(),
            );
        }
        EligibleConfig::NemotronH(args) => args,
        _ => return Err(ReplicatedTextIneligibility::Unrelated.into()),
    };
    let capability_estimate = crate::capability::nemotron_h(args)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let effective_model_type = args.model_type.clone();
    let source_architecture = selected_uses_transform(&selected)
        .then(|| {
            <MixedState as FixedProfile<B, D::State, NemotronHReplicated>>::new(
                args.clone(),
                context,
            )
        })
        .transpose()
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let args = selected_nemotron_h_args(args, &selected)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let prompt_cache_architecture_identity =
        crate::nemotron_h::prompt_cache_architecture_fingerprint(&args);
    let mut architecture =
        <MixedState as FixedProfile<B, D::State, NemotronHReplicated>>::new(args, context)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    architecture.kernel_mut().enable_prediction_capture();
    let prepared = prepare_architecture_handoff::<B, D::State, _>(
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
    visit_replicated_prediction_target_architecture(
        prepared,
        extension,
        store,
        dispatcher.into_visitor(),
    )
}

enum EligibleConfig<'a> {
    NemotronLabsDiffusion(&'a crate::nemotron_labs_diffusion::FamilyConfig),
    K2Horizon(&'a crate::k2_horizon::ModelArgs),
    Llama(&'a crate::llama::ModelArgs),
    Gemma2(&'a crate::gemma2::ModelArgs),
    Nanbeige(&'a crate::nanbeige::ModelArgs),
    Qwen(&'a crate::qwen::ModelArgs),
    Lfm2(&'a crate::lfm2::ModelArgs),
    KimiLinear(&'a crate::kimi_linear::ModelArgs),
    NemotronH(&'a crate::nemotron_h::ModelArgs),
    QwenHybrid(&'a crate::qwen::hybrid::HybridConfig),
    GptOss(&'a crate::gpt_oss::ModelArgs),
    DeepSeekV3(&'a crate::deepseek::V3Args),
    DeepSeekV4(&'a crate::deepseek::V4Args),
    DiffusionGemma(&'a crate::diffusion_gemma::FamilyConfig),
    Gemma4(&'a crate::gemma4::FamilyConfig),
    Inkling(&'a crate::inkling::ModelArgs),
    Muse(&'a crate::muse::DecoderConfig),
    QwenVl(&'a crate::qwen::vl::ModelArgs),
    QwenCompositeHybrid(&'a crate::qwen::hybrid::ParsedHybridConfig),
}

impl EligibleConfig<'_> {
    fn execution_topology(
        &self,
    ) -> Result<Option<eredu_runtime::execution_topology::TextExecutionTopology>, eredu_nn::Error>
    {
        let topology = match self {
            Self::NemotronLabsDiffusion(args) => crate::decoder::topology::text(*args)?,
            Self::DiffusionGemma(args) => crate::diffusion_gemma::topology::text(args)?,
            Self::Llama(args) => crate::decoder::topology::text(*args)?,
            Self::Nanbeige(args) => crate::decoder::topology::text(*args)?,
            Self::Gemma2(args) => crate::decoder::topology::text(*args)?,
            Self::K2Horizon(args) => crate::decoder::topology::text(*args)?,
            Self::Qwen(args) => crate::qwen::execution_topology(args)?,
            Self::Lfm2(args) => crate::lfm2::execution_topology(args)?,
            Self::QwenHybrid(args) => crate::qwen::hybrid::topology::target(args)?,
            Self::QwenCompositeHybrid(args) => crate::qwen::hybrid::topology::target(&args.text)?,
            _ => return Ok(None),
        };
        Ok(Some(topology))
    }

    fn prefill_input_retention(&self) -> eredu_runtime::memory_forecast::PrefillInputRetention {
        use eredu_runtime::memory_forecast::PrefillInputRetention;
        match self {
            Self::Muse(_) => PrefillInputRetention::PreparedProducts,
            _ => PrefillInputRetention::TokenIds,
        }
    }

    fn supports_chunked_prefill(&self) -> bool {
        match self {
            Self::DiffusionGemma(args) => {
                args.encoder_attention
                    == crate::diffusion_gemma::EncoderAttention::CausalWithVisionSpans
            }
            Self::Llama(_)
            | Self::NemotronLabsDiffusion(_)
            | Self::Nanbeige(_)
            | Self::Qwen(_)
            | Self::Gemma2(_)
            | Self::K2Horizon(_)
            | Self::GptOss(_) => crate::decoder::CHUNKED_TEXT_PREFILL,
            Self::Muse(_) => crate::muse::model::RETAINED_PREFILL,
            Self::Lfm2(_) => crate::lfm2::CHUNKED_TEXT_PREFILL,
            Self::QwenHybrid(_) | Self::QwenCompositeHybrid(_) => {
                crate::qwen::hybrid::CHUNKED_TEXT_PREFILL
            }
            _ => false,
        }
    }

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
            Self::NemotronLabsDiffusion(args) => {
                crate::nemotron_labs_diffusion::partitioned::TextBoundarySchema::new(
                    args.hidden_size(),
                )
                .wire_schema()
                .map_err(|error| error.to_string())
            }
            Self::Nanbeige(args) => standard(args.dense_config().hidden_size),
            Self::Gemma2(args) => standard(args.dense_config().hidden_size),
            Self::Llama(args) => standard(args.hidden_size),
            Self::K2Horizon(args) => standard(args.hidden_size),
            Self::Qwen(args) => standard(args.hidden_size),
            Self::Lfm2(args) => standard(args.hidden_size),
            Self::KimiLinear(args) => standard(args.hidden_size),
            Self::QwenHybrid(args) => standard(args.hidden_size),
            Self::GptOss(args) => standard(args.hidden_size),
            Self::Inkling(args) => standard(args.text_config.hidden_size),
            Self::Muse(args) => standard(args.hidden_size),
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
            Self::DiffusionGemma(args) => if args.vision.is_some() {
                crate::diffusion_gemma::composite::BoundarySchema::new(args.text.hidden_size)
                    .wire_schema()
            } else {
                crate::diffusion_gemma::boundary::TextBoundarySchema::new(args.text.hidden_size)
                    .wire_schema()
            }
            .map_err(|error| error.to_string()),
            Self::Gemma4(args) => {
                let width = if args.text.hidden_size_per_layer_input > 0 {
                    eredu_core::balanced_contiguous_range(
                        args.text.hidden_size_per_layer_input as usize,
                        topology.tensor_parallel_size(),
                        topology.tensor_parallel_rank(),
                        false,
                    )
                    .map_err(|error| error.to_string())?
                    .len() as i32
                } else {
                    0
                };
                let state = crate::gemma4::pipeline::tensor_state_layout(&args.text, topology)?;
                crate::gemma4::TextBoundarySchema::from_state_layout(&args.text, width, &state)
                    .wire_schema()
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
        source: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
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
            Self::NemotronLabsDiffusion(_) => {
                extend(crate::nemotron_labs_diffusion::source_recipes(source)?)?
            }
            Self::Gemma2(_) | Self::Llama(_) => {}
            Self::Nanbeige(args) => extend(crate::decoder::repeated::recipes(
                source,
                "model",
                args.physical_layer_count(),
                args.num_loops(),
                !args.skip_loop_final_norm(),
                |layer| crate::decoder::Config::block_parameter_owner(*args, layer),
            )?)?,
            Self::K2Horizon(args) => {
                extend(crate::k2_horizon::linear_companion_recipes(source, args)?)?;
                for layer in 0..self.unit_count()? {
                    for (bank, count, enabled) in [
                        (
                            crate::k2_horizon::ExpertBank::FeedForward,
                            args.num_experts,
                            args.is_sparse_layer(layer),
                        ),
                        (
                            crate::k2_horizon::ExpertBank::AttentionValue,
                            args.mova_num_experts,
                            args.is_mova_layer(layer),
                        ),
                    ] {
                        if enabled {
                            extend(crate::k2_horizon::expert_recipes(
                                source,
                                args,
                                layer,
                                bank,
                                &(0..count as usize).collect::<Vec<_>>(),
                            )?)?;
                        }
                    }
                }
            }
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
                    extend(
                        eredu_checkpoint::expert::gated_product_materialization_recipes(
                            source, &expert,
                        )?,
                    )?;
                }
            }
            Self::DiffusionGemma(args) => {
                for ordinal in 0..args.text.num_hidden_layers() {
                    extend(crate::diffusion_gemma::unit_recipes(source, args, ordinal)?)?;
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
            Self::Muse(args) => {
                if args.weight_convention == crate::muse::WeightConvention::Gguf {
                    let vision_source = source
                        .source_provenance(
                            "model.vision_tower.patch_embedder.patch_embedding.weight",
                        )
                        .ok();
                    if !vision_source.as_ref().is_some_and(|source| {
                        matches!(source.source_encoding, SourceTensorEncoding::Safetensors(_))
                    }) {
                        extend(crate::muse::checkpoint::projector_gguf_recipes(
                            args, source,
                        )?)?;
                    }
                    if args.is_moe() {
                        let text_layers = usize::try_from(args.num_hidden_layers)
                            .map_err(|_| "invalid Muse-Glimmer text layer count".to_owned())?;
                        for layer in 0..text_layers {
                            let expert = crate::muse::expert_recipes(source, layer)?;
                            extend(BTreeMap::from([
                                (expert.target_gate_up, expert.gate_up),
                                (expert.target_down, expert.down),
                            ]))?;
                        }
                    }
                    return Ok(recipes);
                }
                extend(crate::muse::static_safetensors_recipes(args, source)?)?;
                let mut counts = vec![
                    args.vision_config
                        .as_ref()
                        .map_or(0, |vision| vision.layer_count()),
                    usize::try_from(args.num_hidden_layers)
                        .map_err(|_| "invalid Muse-Glimmer text layer count".to_owned())?,
                ];
                if let Some(audio) = &args.audio_config {
                    counts.push(
                        audio
                            .execution_units()
                            .map_err(|error| error.to_string())?
                            .len(),
                    );
                }
                for (group, count) in counts.into_iter().enumerate() {
                    for index in 0..count {
                        extend(crate::muse::unit_safetensors_recipes(
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

    /// Declares the fused physical inputs which this architecture deliberately
    /// projects into multiple logical parameters. Other architectures retain
    /// exclusive source ownership even when malformed recipes happen to repeat.
    fn shared_derived_source_keys(
        &self,
        recipes: &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> BTreeSet<String> {
        if !matches!(self, Self::QwenHybrid(_) | Self::Inkling(_)) {
            return BTreeSet::new();
        }
        let mut claims = BTreeMap::<String, usize>::new();
        for recipe in recipes.values() {
            for source in recipe.source_keys().into_iter().collect::<BTreeSet<_>>() {
                *claims.entry(source.to_owned()).or_default() += 1;
            }
        }
        claims
            .into_iter()
            .filter_map(|(source, count)| (count > 1).then_some(source))
            .collect()
    }

    fn architecture_identity(&self) -> String {
        match self {
            Self::NemotronLabsDiffusion(args) => args.architecture_fingerprint(),
            Self::Nanbeige(args) => crate::nanbeige::prompt_cache_architecture_fingerprint(args),
            Self::Gemma2(args) => crate::gemma2::prompt_cache_architecture_fingerprint(args),
            Self::Llama(args) => crate::llama::prompt_cache_architecture_fingerprint(args),
            Self::K2Horizon(args) => crate::k2_horizon::prompt_cache_architecture_fingerprint(args),
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
            Self::DiffusionGemma(args) => args.architecture_fingerprint(),
            Self::Gemma4(args) => args.architecture_fingerprint(),
            Self::Inkling(args) => args.architecture_fingerprint(),
            Self::Muse(args) => args.architecture_fingerprint(),
            Self::QwenVl(args) => crate::qwen::vl::prompt_cache_architecture_fingerprint(args),
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(args)
            }
        }
    }

    fn canonical_parameter_name(&self, name: &str, aliases: &[String]) -> String {
        if matches!(self, Self::NemotronLabsDiffusion(_)) {
            return crate::nemotron_labs_diffusion::canonical_parameter_name(name);
        }
        if matches!(self, Self::DiffusionGemma(_)) {
            return crate::diffusion_gemma::canonical_safetensors_target(name);
        }
        if matches!(self, Self::Muse(_)) {
            return crate::muse::checkpoint::canonical_safetensors_target(name);
        }
        if name.contains(".experts.") {
            if let Some(prefix) = name.strip_suffix(".scales") {
                return format!("{prefix}_scales");
            }
            if let Some(prefix) = name.strip_suffix(".biases") {
                return format!("{prefix}_biases");
            }
        }
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
            Self::NemotronLabsDiffusion(_) => false,
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
            Self::Gemma2(_)
            | Self::Llama(_)
            | Self::Nanbeige(_)
            | Self::K2Horizon(_)
            | Self::Qwen(_)
            | Self::GptOss(_)
            | Self::DeepSeekV3(_)
            | Self::DeepSeekV4(_)
            | Self::DiffusionGemma(_)
            | Self::Gemma4(_)
            | Self::Inkling(_)
            | Self::Muse(_)
            | Self::QwenVl(_) => false,
        }
    }

    fn state_access(&self) -> ReplicatedTextStateAccess {
        ReplicatedTextStateAccess::from_layout(
            &self.state_layout().expect("validated family state"),
        )
        .expect("validated family state access")
    }

    fn operators(&self) -> NeuralOperatorCapabilities {
        match self {
            Self::NemotronLabsDiffusion(args) => crate::decoder::operator_requirements(*args)
                .union(NeuralOperatorCapabilities::CAST_FLOAT),
            Self::K2Horizon(args) => crate::decoder::operator_requirements(*args),
            Self::Gemma2(args) => crate::decoder::operator_requirements(*args),
            Self::Llama(_)
            | Self::Nanbeige(_)
            | Self::Qwen(_)
            | Self::Lfm2(_)
            | Self::GptOss(_) => NeuralOperatorCapabilities::NONE,
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
            Self::DiffusionGemma(_) => crate::operator_requirements::GEMMA4
                .union(NeuralOperatorCapabilities::WEIGHTED_EMBEDDING),
            Self::Gemma4(_) => crate::operator_requirements::GEMMA4,
            Self::Inkling(_) => crate::operator_requirements::INKLING,
            Self::Muse(args) => crate::operator_requirements::muse(args),
            Self::QwenVl(_) => crate::operator_requirements::QWEN_VL,
            Self::QwenCompositeHybrid(_) => crate::operator_requirements::QWEN_HYBRID
                .union(crate::operator_requirements::QWEN_VISION),
        }
    }

    fn execution_group(&self) -> &'static str {
        match self {
            Self::NemotronLabsDiffusion(_) => crate::decoder::TEXT_DECODER_EXECUTION_GROUP,
            Self::Gemma2(_)
            | Self::Llama(_)
            | Self::Nanbeige(_)
            | Self::K2Horizon(_)
            | Self::Qwen(_)
            | Self::GptOss(_) => crate::decoder::TEXT_DECODER_EXECUTION_GROUP,
            Self::Lfm2(_) | Self::KimiLinear(_) | Self::NemotronH(_) | Self::QwenHybrid(_) => {
                crate::decoder::TARGET_EXECUTION_GROUP
            }
            Self::DeepSeekV3(_) => crate::decoder::TARGET_EXECUTION_GROUP,
            Self::DeepSeekV4(_) => crate::decoder::TARGET_EXECUTION_GROUP,
            Self::DiffusionGemma(_) => crate::diffusion_gemma::model::TEXT_EXECUTION_GROUP,
            Self::Gemma4(_) => crate::gemma4::model::TEXT_EXECUTION_GROUP,
            Self::Inkling(_) => crate::inkling::model::TEXT_EXECUTION_GROUP,
            Self::Muse(_) => crate::muse::model::TEXT_EXECUTION_GROUP,
            Self::QwenVl(_) => crate::qwen::vl::TEXT_EXECUTION_GROUP,
            Self::QwenCompositeHybrid(_) => crate::qwen::hybrid::VISION_EXECUTION_GROUP,
        }
    }

    fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        fn decoder<C: crate::decoder::Config>(
            args: &C,
            parameters: Result<
                eredu_runtime::ArchitectureParameterDescription,
                eredu_runtime::ParallelPlanError,
            >,
        ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
            crate::decoder::decoder_component_definition(
                args,
                parameters.map_err(eredu_nn::Error::backend_source)?,
                crate::decoder::state_layout(args)?,
            )
        }
        match self {
            Self::NemotronLabsDiffusion(args) => {
                crate::nemotron_labs_diffusion::definition::text(args, None)
            }
            Self::Llama(args) => decoder(*args, crate::decoder::dense_parameter_description(*args)),
            Self::Nanbeige(args) => {
                decoder(*args, crate::decoder::dense_parameter_description(*args))
            }
            Self::Gemma2(args) => {
                decoder(*args, crate::decoder::dense_parameter_description(*args))
            }
            Self::K2Horizon(args) => decoder(*args, crate::k2_horizon::parameter_description(args)),
            Self::Qwen(args) => decoder(*args, crate::qwen::parameter_description(args)),
            Self::GptOss(args) => decoder(*args, crate::gpt_oss::parameter_description(args)),
            Self::Lfm2(args) => crate::lfm2::component_definition(args),
            Self::KimiLinear(args) => crate::kimi_linear::component_definition(args),
            Self::NemotronH(args) => crate::nemotron_h::component_definition(
                crate::nemotron_h::parameter_description(args)
                    .map_err(eredu_nn::Error::backend_source)?,
                crate::nemotron_h::state_layout(args).map_err(eredu_nn::Error::backend_source)?,
            ),
            Self::QwenHybrid(args) => crate::qwen::hybrid::component_definition(args, None),
            Self::DeepSeekV3(args) => crate::deepseek::v3::component_definition(args, None),
            Self::DeepSeekV4(args) => crate::deepseek::v4::component_definition(args, None),
            Self::DiffusionGemma(args) => {
                crate::diffusion_gemma::model::component_definition(args, None)
            }
            Self::Gemma4(args) => crate::gemma4::model::component_definition(args, None),
            Self::Inkling(args) => crate::inkling::model::component_definition(args, None),
            Self::Muse(args) => crate::muse::model::component_definition(args, None),
            Self::QwenVl(args) => crate::qwen::vl::component_definition(args, None),
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::conditional_component_definition(args, None)
            }
        }
    }

    fn unit_count(&self) -> Result<usize, String> {
        let count = match self {
            Self::NemotronLabsDiffusion(args) => args.num_hidden_layers(),
            Self::Nanbeige(args) => {
                return crate::decoder::DecoderSchedule::new(*args)
                    .map(|schedule| schedule.len())
                    .map_err(|error| error.to_string())
            }
            Self::Gemma2(args) => args.dense_config().num_hidden_layers,
            Self::Llama(args) => args.num_hidden_layers,
            Self::K2Horizon(args) => args.num_hidden_layers,
            Self::Qwen(args) => args.num_hidden_layers,
            Self::Lfm2(args) => args.num_hidden_layers,
            Self::KimiLinear(args) => args.num_hidden_layers,
            Self::NemotronH(args) => args.num_hidden_layers,
            Self::QwenHybrid(args) => args.num_hidden_layers,
            Self::GptOss(args) => args.num_hidden_layers,
            Self::DeepSeekV3(args) => args.num_hidden_layers,
            Self::DeepSeekV4(args) => args.num_hidden_layers,
            Self::DiffusionGemma(args) => return Ok(args.text.num_hidden_layers()),
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
            Self::Muse(args) => {
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
            Self::NemotronLabsDiffusion(args) => {
                crate::decoder::state_layout(*args).map_err(|error| error.to_string())
            }
            Self::Nanbeige(args) => {
                crate::nanbeige::state_layout(args).map_err(|error| error.to_string())
            }
            Self::Gemma2(args) => {
                crate::gemma2::state_layout(*args).map_err(|error| error.to_string())
            }
            Self::Llama(args) => {
                crate::llama::state_layout(*args).map_err(|error| error.to_string())
            }
            Self::K2Horizon(args) => {
                crate::decoder::state_layout(*args).map_err(|error| error.to_string())
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
            Self::DiffusionGemma(args) => {
                crate::diffusion_gemma::state_layout(args).map_err(|error| error.to_string())
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
            Self::Muse(args) => crate::muse::state_layout(args).map_err(|error| error.to_string()),
            Self::QwenVl(args) => {
                crate::qwen::vl::state_layout(args).map_err(|error| error.to_string())
            }
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::state_layout(&args.text).map_err(|error| error.to_string())
            }
        }
    }

    fn native_format_for_shape(&self, name: &str, shape: &[usize]) -> LinearFormat {
        if let Self::Gemma4(args) = self {
            let input = shape
                .last()
                .and_then(|&size| i32::try_from(size).ok())
                .unwrap_or(0);
            if name.starts_with("model.vision_") || name.starts_with("model.embed_vision.") {
                return args.vision.as_ref().map_or(LinearFormat::Dense, |config| {
                    config.linear_format_for(name, input)
                });
            }
            if name.starts_with("model.audio_") || name.starts_with("model.embed_audio.") {
                return args.audio.as_ref().map_or(LinearFormat::Dense, |config| {
                    config.linear_format_for(name, input)
                });
            }
        }
        self.native_format(name)
    }

    fn native_format(&self, name: &str) -> LinearFormat {
        match self {
            Self::NemotronLabsDiffusion(args) => {
                return crate::decoder::Config::linear_format(*args, name)
            }
            Self::Nanbeige(args) => args.weight_quantization_for(name),
            Self::Gemma2(args) => args.weight_quantization_for(name),
            Self::Llama(args) => args.weight_quantization_for(name),
            Self::K2Horizon(args) => return args.linear_format_for(name),
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
            Self::DeepSeekV4(args) => return args.linear_format_for(name),
            Self::DiffusionGemma(args) => return args.text.linear_format_for(name),
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
            Self::Muse(args) => {
                if name.starts_with("model.vision_") {
                    return args
                        .vision_config
                        .as_ref()
                        .map_or(LinearFormat::Dense, |vision| vision.linear_format_for(name));
                }
                if name.starts_with("model.audio_tower.") {
                    return args
                        .audio_config
                        .as_ref()
                        .map_or(LinearFormat::Dense, |audio| audio.linear_format_for(name));
                }
                return args.linear_format_for(name);
            }
            Self::QwenVl(args) => {
                if name.starts_with("model.visual.") {
                    return args.vision.linear_format(name);
                }
                return args.text.weight_quantization_for(name).into();
            }
            Self::QwenCompositeHybrid(args) => {
                if name.starts_with("model.visual.") {
                    return args
                        .vision
                        .as_ref()
                        .map_or(LinearFormat::Dense, |vision| vision.linear_format(name));
                }
                return args.text.linear_format(name);
            }
        }
        .map_or(LinearFormat::Dense, LinearFormat::from)
    }

    fn linear_parameter_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>, String> {
        match self {
            Self::NemotronLabsDiffusion(args) => decoder_linear_parameter_shapes(*args),
            Self::Nanbeige(args) => decoder_linear_parameter_shapes(*args),
            Self::Gemma2(args) => decoder_linear_parameter_shapes(*args),
            Self::Llama(args) => decoder_linear_parameter_shapes(*args),
            Self::K2Horizon(args) => {
                let mut shapes = family_linear_parameter_shapes(
                    crate::k2_horizon::safetensors_plan(args)?,
                    |name| Some(args.linear_format_for(name)),
                    "model.embed_tokens.weight",
                )?;
                let rank = eredu_core::ParallelRankTopology::new(
                    eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|e| e.to_string())?,
                    0,
                )
                .map_err(|e| e.to_string())?;
                for bank in crate::k2_horizon::expert_realization_plans(args, rank, None)
                    .map_err(|e| e.to_string())?
                    .values()
                {
                    match bank {
                        crate::RoutedGroupedPlan::Gated(plan) => {
                            insert_grouped_gated_linear_shapes(&mut shapes, plan)?
                        }
                        crate::RoutedGroupedPlan::Linear(plan) => {
                            for spec in plan.unit_specs().values() {
                                shapes.insert(
                                    spec.projection().weight().id.to_string(),
                                    vec![
                                        spec.group_count() as usize,
                                        spec.output_dimensions() as usize,
                                        spec.input_dimensions() as usize,
                                    ],
                                );
                            }
                        }
                        crate::RoutedGroupedPlan::Relu2(_) => {
                            unreachable!("family specifies SwiGLU")
                        }
                    }
                }
                Ok(shapes)
            }
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
                let mut shapes = crate::deepseek::checkpoint::v4_linear_parameter_shapes(args)?;
                if args.num_nextn_predict_layers == 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::deepseek::v4_replicated_expert_realization_plan(args)
                            .map_err(|error| error.to_string())?,
                    )?;
                }
                Ok(shapes)
            }
            Self::DiffusionGemma(args) => {
                // Logical matrix geometry is independent of its packed storage.
                let dense = args.with_checkpoint_formats(HashMap::new());
                let mut groups = crate::diffusion_gemma::parallel::static_parameter_groups(&dense)
                    .map_err(|error| error.to_string())?;
                for layer in 0..args.text.num_hidden_layers() {
                    groups.extend(
                        crate::diffusion_gemma::parallel::layer_parameter_groups(&dense, layer)
                            .map_err(|error| error.to_string())?,
                    );
                }
                Ok(groups
                    .into_iter()
                    .flat_map(|group| group.members().to_vec())
                    .filter(|member| {
                        let name = member.target();
                        member.global_shape().len() >= 2
                            && name != "model.decoder.embed_tokens.weight"
                            && (name.ends_with(".weight")
                                || name.ends_with(".gate_up_proj")
                                || name.ends_with(".down_proj"))
                    })
                    .map(|member| (member.target().to_owned(), member.global_shape().to_vec()))
                    .collect())
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
                let plan =
                    crate::inkling::safetensors_plan(args).map_err(|error| error.to_string())?;
                let canonical = plan
                    .common_tensors
                    .iter()
                    .chain(
                        plan.layout_groups
                            .iter()
                            .flat_map(|group| &group.variants)
                            .flat_map(|variant| &variant.tensors),
                    )
                    .map(|tensor| {
                        (
                            tensor.key.clone(),
                            self.canonical_parameter_name(&tensor.key, &tensor.aliases),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                // Recipe finalization uses canonical targets, so released aliases
                // must retain the same semantic role and format at both stages.
                let mut shapes = family_linear_parameter_shapes(
                    plan,
                    |name| {
                        Some(self.native_format(canonical.get(name).map_or(name, String::as_str)))
                    },
                    "model.llm.embed.weight",
                )?
                .into_iter()
                .map(|(name, shape)| (canonical.get(&name).cloned().unwrap_or(name), shape))
                .collect::<BTreeMap<_, _>>();
                // The relative table and joint-selection router are raw tensor
                // operators, not backend Linear modules. Their rank alone does
                // not authorize conversion; both remain floating parameters.
                shapes.retain(|name, _| {
                    name != "model.embed_tokens.weight"
                        && !name.ends_with(".self_attn.rel_proj")
                        && !name.ends_with(".moe.router.weight")
                });
                if args.text_config.has_sparse_moe_layers() {
                    let (plan, _) = inkling_replicated_expert_realization_plan(args)?;
                    insert_grouped_gated_linear_shapes(&mut shapes, &plan)?;
                }
                Ok(shapes)
            }
            Self::Muse(args) => {
                let mut shapes = match args.weight_convention {
                    crate::muse::WeightConvention::HuggingFace => {
                        let plan = crate::muse::safetensors_plan(args)?;
                        let canonical = plan
                            .common_tensors
                            .iter()
                            .chain(
                                plan.layout_groups
                                    .iter()
                                    .flat_map(|group| &group.variants)
                                    .flat_map(|variant| &variant.tensors),
                            )
                            .map(|tensor| {
                                (
                                    tensor.key.clone(),
                                    self.canonical_parameter_name(&tensor.key, &tensor.aliases),
                                )
                            })
                            .collect::<BTreeMap<_, _>>();
                        family_linear_parameter_shapes(
                            plan,
                            |name| {
                                Some(self.native_format(
                                    canonical.get(name).map_or(name, String::as_str),
                                ))
                            },
                            "model.language_model.embed_tokens.weight",
                        )?
                        .into_iter()
                        .map(|(name, shape)| (canonical.get(&name).cloned().unwrap_or(name), shape))
                        .collect()
                    }
                    crate::muse::WeightConvention::Gguf => {
                        // GGUF text admission has no required vision tower. Its
                        // declared shapes are logical, independent of encoded
                        // source blocks, and its names have their own translation.
                        let mut shapes = crate::muse::gguf_plan(args)?
                            .common_tensors
                            .into_iter()
                            .filter(|tensor| {
                                tensor.shape.len() == 2 && tensor.key != "token_embd.weight"
                            })
                            .map(|tensor| {
                                (
                                    crate::muse::translate_text_gguf_name(&tensor.key),
                                    tensor.shape,
                                )
                            })
                            .collect::<BTreeMap<_, _>>();
                        if args.vision_config.is_some() {
                            shapes.extend(
                                crate::muse::projector_gguf_plan(args)?
                                    .common_tensors
                                    .into_iter()
                                    .filter(|tensor| tensor.shape.len() == 2)
                                    .map(|tensor| {
                                        (
                                            crate::muse::translate_projector_gguf_name(
                                                &tensor.key,
                                                args.audio_config.as_ref(),
                                            ),
                                            tensor.shape,
                                        )
                                    }),
                            );
                            let patch = crate::muse::projector_gguf_plan(args)?
                                .common_tensors
                                .into_iter()
                                .find(|tensor| tensor.key == "v.patch_embd.weight")
                                .expect("projector schema declares its patch kernel");
                            let width =
                                patch.shape[1..]
                                    .iter()
                                    .try_fold(1usize, |total, dimension| {
                                        total.checked_mul(*dimension).ok_or_else(|| {
                                            "Muse-Glimmer patch width overflowed".to_owned()
                                        })
                                    })?;
                            shapes.insert(
                                crate::muse::translate_projector_gguf_name(
                                    &patch.key,
                                    args.audio_config.as_ref(),
                                ),
                                vec![patch.shape[0], width],
                            );
                        }
                        shapes
                    }
                };
                if args.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &crate::muse::expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or_else(|| {
                            "Muse routed parameters have no expert realization".to_owned()
                        })?,
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
                // The static position table uses a fixed dense embedding lookup.
                shapes.remove("model.visual.pos_embed.weight");
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
                // The static position table uses a fixed dense embedding lookup.
                shapes.remove("model.visual.pos_embed.weight");
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
            Self::NemotronLabsDiffusion(args) => crate::decoder::Config::parameter_root(*args),
            Self::Nanbeige(args) => crate::decoder::Config::parameter_root(*args),
            Self::Gemma2(args) => crate::decoder::Config::parameter_root(*args),
            Self::Llama(args) => crate::decoder::Config::parameter_root(*args),
            Self::K2Horizon(args) => crate::decoder::Config::parameter_root(*args),
            Self::Qwen(args) => crate::decoder::Config::parameter_root(*args),
            Self::Lfm2(_) | Self::KimiLinear(_) | Self::QwenHybrid(_) => "model",
            Self::NemotronH(_) => "model",
            Self::GptOss(args) => crate::decoder::Config::parameter_root(*args),
            Self::DeepSeekV3(_) => "model",
            Self::DeepSeekV4(_) => "",
            Self::DiffusionGemma(_) => "model.decoder",
            Self::Gemma4(_) => "model.language_model",
            Self::Inkling(_) | Self::Muse(_) | Self::QwenCompositeHybrid(_) => "model",
            Self::QwenVl(args) => &args.text.parameter_root,
        }
    }

    fn tied_embeddings(&self) -> bool {
        match self {
            Self::NemotronLabsDiffusion(_) => false,
            Self::Nanbeige(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Gemma2(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Llama(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::K2Horizon(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Qwen(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Lfm2(args) => args.tie_word_embeddings,
            Self::KimiLinear(args) => args.tie_word_embeddings,
            Self::NemotronH(args) => args.tie_word_embeddings,
            Self::QwenHybrid(args) => args.tie_word_embeddings,
            Self::GptOss(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::DeepSeekV3(args) => args.tie_word_embeddings,
            Self::DeepSeekV4(args) => args.tie_word_embeddings,
            Self::DiffusionGemma(args) => args.text.tie_word_embeddings,
            Self::Gemma4(args) => args.text.tie_word_embeddings,
            Self::Inkling(_) => false,
            Self::Muse(args) => args.tie_word_embeddings,
            Self::QwenVl(args) => args.text.tie_word_embeddings,
            Self::QwenCompositeHybrid(args) => args.text.tie_word_embeddings,
        }
    }

    fn embedding_shape(&self) -> Result<Vec<usize>, String> {
        let (vocabulary, hidden) = match self {
            Self::NemotronLabsDiffusion(args) => (args.vocabulary_size(), args.hidden_size()),
            Self::Nanbeige(args) => (
                args.dense_config().vocab_size,
                args.dense_config().hidden_size,
            ),
            Self::Gemma2(args) => (
                args.dense_config().vocab_size,
                args.dense_config().hidden_size,
            ),
            Self::Llama(args) => (args.vocab_size, args.hidden_size),
            Self::K2Horizon(args) => (args.vocab_size, args.hidden_size),
            Self::Qwen(args) => (args.vocab_size, args.hidden_size),
            Self::Lfm2(args) => (args.vocab_size, args.hidden_size),
            Self::KimiLinear(args) => (args.vocab_size, args.hidden_size),
            Self::NemotronH(args) => (args.vocab_size, args.hidden_size),
            Self::QwenHybrid(args) => (args.vocab_size, args.hidden_size),
            Self::GptOss(args) => (args.vocab_size, args.hidden_size),
            Self::DeepSeekV3(args) => (args.vocab_size, args.hidden_size),
            Self::DeepSeekV4(args) => (args.vocab_size, args.hidden_size),
            Self::DiffusionGemma(args) => (args.text.vocab_size, args.text.hidden_size),
            Self::Gemma4(args) => (args.text.vocab_size, args.text.hidden_size),
            Self::Inkling(args) => (args.text_config.vocab_size, args.text_config.hidden_size),
            Self::Muse(args) => (args.vocab_size, args.hidden_size),
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
            Self::NemotronLabsDiffusion(args) => {
                decoder_parameter_role(*args, name, companion, linear_shapes)
            }
            Self::Nanbeige(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Gemma2(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Llama(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::K2Horizon(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Qwen(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::GptOss(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Lfm2(_)
            | Self::KimiLinear(_)
            | Self::NemotronH(_)
            | Self::QwenHybrid(_)
            | Self::DeepSeekV3(_)
            | Self::DeepSeekV4(_)
            | Self::DiffusionGemma(_)
            | Self::Gemma4(_)
            | Self::Inkling(_)
            | Self::Muse(_)
            | Self::QwenVl(_)
            | Self::QwenCompositeHybrid(_) => {
                family_parameter_role(self, name, companion, linear_shapes)
            }
        }
    }

    fn embedding_name(&self) -> String {
        match self {
            Self::NemotronLabsDiffusion(_) => "model.embed_tokens.weight".into(),
            Self::NemotronH(_) => "model.embeddings.weight".into(),
            Self::Gemma2(_)
            | Self::Llama(_)
            | Self::Nanbeige(_)
            | Self::K2Horizon(_)
            | Self::Qwen(_)
            | Self::Lfm2(_)
            | Self::KimiLinear(_)
            | Self::QwenHybrid(_) => {
                format!("{}.embed_tokens.weight", self.parameter_root())
            }
            Self::GptOss(_) => format!("{}.embed_tokens.weight", self.parameter_root()),
            Self::DeepSeekV3(_) => "model.embed_tokens.weight".into(),
            Self::DeepSeekV4(_) => "embed.weight".into(),
            Self::DiffusionGemma(_) => "model.decoder.embed_tokens.weight".into(),
            Self::Gemma4(_) => "model.language_model.embed_tokens.weight".into(),
            Self::Inkling(_) | Self::Muse(_) | Self::QwenCompositeHybrid(_) => {
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
        // An architecture may retain an unaligned matrix in floating storage
        // under a family-wide quantization request. Only encoded schema tensors
        // need their packed dimension expanded into logical coordinates.
        let format =
            if constraint.dtype == eredu_checkpoint::schema::StoredDtypeConstraint::Floating {
                LinearFormat::Dense
            } else {
                native_format(&constraint.key).unwrap_or(LinearFormat::Dense)
            };
        match format {
            LinearFormat::Affine(format) => {
                let bits = usize::try_from(format.bits)
                    .map_err(|_| format!("invalid affine bits for {:?}", constraint.key))?;
                let packed = shape.last_mut().expect("rank checked above");
                *packed = packed
                    .checked_mul(32)
                    .and_then(|bits_total| bits_total.checked_div(bits))
                    .ok_or_else(|| format!("invalid affine geometry for {:?}", constraint.key))?;
            }
            LinearFormat::MxFp4 | LinearFormat::NvFp4 => {
                let packed = shape.last_mut().expect("rank checked above");
                *packed = packed
                    .checked_mul(8)
                    .ok_or_else(|| format!("invalid MXFP4 geometry for {:?}", constraint.key))?;
            }
            LinearFormat::GgufIQuant { ggml_type, .. } => {
                let invalid = || format!("invalid GGUF block geometry for {:?}", constraint.key);
                let (values, bytes) = ggml_type.block_and_bytes().map_err(|_| invalid())?;
                let values = usize::try_from(values).map_err(|_| invalid())?;
                let bytes = usize::try_from(bytes).map_err(|_| invalid())?;
                let packed = shape.last_mut().expect("rank checked above");
                if bytes == 0 || !packed.is_multiple_of(bytes) {
                    return Err(invalid());
                }
                *packed = (*packed / bytes).checked_mul(values).ok_or_else(invalid)?;
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
        if config.block_parameter_owner(layer) != layer {
            continue;
        }
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

/// Derives the complete replicated execution and checkpoint contract.
pub fn replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    inspection
        .architecture_plan()
        .validation(inspection.admission_token())
        .replicated
        .get_or_init(|| derive_replicated_text_requirements(inspection))
        .clone()
}

fn derive_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    if let Some(config) = crate::routed_text::diffusion_args(inspection) {
        return if config.has_sparse_moe_layers() || config.vision.is_some() {
            Err(ReplicatedTextIneligibility::BlockRefinement.into())
        } else {
            diffusion_text_requirements(inspection, config)
        };
    }
    let plan = inspection.architecture_plan();
    let config = eligible_config(plan)?;
    let geometry = match &config {
        EligibleConfig::NemotronLabsDiffusion(config) => Some(config.generation_geometry()),
        _ => None,
    };
    let requirements = replicated_text_requirements_for_config(inspection, config)?;
    Ok(match geometry {
        Some(geometry) => requirements.with_block_refinement(geometry),
        None => requirements,
    })
}

pub(crate) fn diffusion_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: &crate::diffusion_gemma::FamilyConfig,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let geometry = eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
        canvas_length: config.canvas_length,
        vocabulary_size: std::num::NonZeroU32::new(config.text.vocab_size as u32)
            .expect("validated vocabulary"),
    };
    replicated_text_requirements_for_config(inspection, EligibleConfig::DiffusionGemma(config))
        .map(|requirements| requirements.with_block_refinement(geometry))
}

pub(crate) fn diffusion_execution_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextExecutionClass, ReplicatedTextRequirementsError> {
    let artifact = inspection.architecture_plan();
    let config = match (artifact.safetensors_architecture(), artifact.gguf_plan()) {
        (Some(plan), _) => match plan.model() {
            SafetensorsModelConfig::DiffusionGemma(config) => config,
            _ => {
                return Err(ReplicatedTextRequirementsError::Ineligible(
                    ReplicatedTextIneligibility::BlockRefinement,
                ))
            }
        },
        (_, Some(plan)) => match plan.model() {
            GgufModelConfig::DiffusionGemma(config) => config,
            _ => {
                return Err(ReplicatedTextRequirementsError::Ineligible(
                    ReplicatedTextIneligibility::BlockRefinement,
                ))
            }
        },
        _ => {
            return Err(ReplicatedTextRequirementsError::Ineligible(
                ReplicatedTextIneligibility::BlockRefinement,
            ))
        }
    };
    if config.vision.is_some() {
        return composite_text_requirements(inspection)
            .map(ReplicatedTextExecutionClass::Composite);
    }
    let text = diffusion_text_requirements(inspection, config)?;
    if !config.has_sparse_moe_layers() {
        return Ok(ReplicatedTextExecutionClass::Replicated(text));
    }
    let invalid = |error: String| ReplicatedTextRequirementsError::InvalidArchitecture(error);
    let source = artifact::target_catalog(inspection)?;
    let owner =
        eredu_runtime::ExecutionGroupId::new(crate::diffusion_gemma::model::TEXT_EXECUTION_GROUP)
            .map_err(|error| invalid(error.to_string()))?;
    let plan = crate::diffusion_gemma::model::expert_realization_plan(
        config,
        None,
        replicated_expert_topology().map_err(invalid)?,
    )
    .map_err(|error| invalid(error.to_string()))?
    .ok_or_else(|| invalid("selected shared stack has no routed bank".into()))?;
    let catalog = crate::diffusion_gemma::expert_residency_catalog(source.as_ref(), config)
        .map_err(invalid)?;
    crate::routed_text::gated_routed_text_requirements(
        text,
        owner,
        plan,
        catalog,
        config.text.top_k_experts.unwrap_or_default() as usize,
        source.as_ref(),
    )
    .map(ReplicatedTextExecutionClass::Routed)
    .map_err(|error| invalid(error.to_string()))
}

pub(crate) fn k2_horizon_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::k2_horizon::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(inspection, EligibleConfig::K2Horizon(args))
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

fn replicated_text_requirements_for_structure(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: EligibleConfig<'_>,
    component: Option<eredu_runtime::ComponentDefinition>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    let normalized = artifact::normalize(inspection, &config)?;
    let component = match component {
        Some(component) => component,
        None => config.component_definition().map_err(|error| {
            ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
        })?,
    };
    let execution_graph = component.parameters().graph().clone();
    let execution_units = component.parameters().unit_layout().clone();
    let group_transports = (0..execution_graph.groups().len())
        .map(|group| {
            component
                .group_transport(group)
                .expect("declared transport")
                .clone()
        })
        .collect();
    let state_layout = component.state_layout().clone();
    let state_access = ReplicatedTextStateAccess::from_layout(&state_layout)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    let (parameters, derived_recipes, derived_recipe_outputs) = normalized.parameters.clone();
    let shared_source_keys = normalized.shared_source_keys.clone();
    let auxiliary = normalized.prediction.clone();
    let floating_source = crate::preparation::prepared_floating_state_dtype_source(inspection)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArtifact(error.to_string()))?;
    let mut requirements = ReplicatedTextRequirements::new(
        config.architecture_identity(),
        config.operators(),
        execution_graph,
        execution_units,
        group_transports,
        state_layout,
        state_access,
        parameters,
    )
    .and_then(|requirements| {
        requirements
            .with_chunked_prefill(config.supports_chunked_prefill())
            .with_prefill_input_retention(config.prefill_input_retention())
            .with_floating_state_source(floating_source.dtype().clone())
            .with_derived_recipes_and_shared_sources(
                derived_recipes,
                derived_recipe_outputs,
                shared_source_keys,
            )
    })
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    requirements = requirements
        .with_component_definition(component)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    requirements =
        requirements.with_execution_topology(config.execution_topology().map_err(|error| {
            ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
        })?);
    if let Some(extension) = plan.prediction_extension() {
        use crate::configuration::PredictionExtensionKind;
        let roles: &[&str] = match extension.kind() {
            PredictionExtensionKind::DeepSeekV3Mtp => &["embedding"],
            // These extensions consume the target vocabulary head.
            // Every independently executing prediction replica needs its selected
            // embedding and head shard, including non-output pipeline stages.
            PredictionExtensionKind::DeepSeekV4Embedded
            | PredictionExtensionKind::QwenHybridMtp
            | PredictionExtensionKind::NemotronHMtp => &["embedding", "output"],
            PredictionExtensionKind::InklingMtp => &["embedding", "embedding_norm", "output"],
        };
        requirements = requirements
            .with_replicated_static_roles(roles.iter().copied())
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
    }
    if let Some((parameters, recipes, outputs)) = auxiliary {
        requirements = requirements
            .with_auxiliary_parameters(parameters, recipes, outputs)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
    }
    requirements
        .with_nonparameter_sources(artifact::nonparameter_sources(inspection, &normalized))
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

fn prediction_extension_materialization_parameters(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    extension: &crate::configuration::PredictionExtensionPlan,
) -> Result<FinalizedMaterializationParameters, ReplicatedTextRequirementsError> {
    let complete = extension.complete_architecture();
    let recipe_source = safetensors_inspection_recipe_source(inspection, complete)?;
    let config = match complete.model() {
        SafetensorsModelConfig::DeepSeekV3(args) => EligibleConfig::DeepSeekV3(args),
        SafetensorsModelConfig::DeepSeekV4(args) => EligibleConfig::DeepSeekV4(args),
        SafetensorsModelConfig::Inkling(args) => EligibleConfig::Inkling(args),
        SafetensorsModelConfig::QwenHybrid(args) => EligibleConfig::QwenHybrid(&args.text),
        SafetensorsModelConfig::NemotronH(args) => EligibleConfig::NemotronH(args),
        _ => {
            return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                "prediction extension has no supported materialization contract".into(),
            ));
        }
    };
    let parameters = safetensors_parameters(
        complete,
        inspection.tensors(),
        &config,
        recipe_source.as_ref(),
    )?;
    let extra_recipes = prediction_extension_recipes(extension, recipe_source.as_ref())?;
    let extra_targets = extra_recipes.keys().cloned().collect::<BTreeSet<_>>();
    let (mut parameters, mut recipes, mut outputs) = finalize_materialization_parameters(
        &config,
        parameters,
        recipe_source.as_ref(),
        extra_recipes,
    )?;
    let missing_recipe_parameters = extra_targets
        .iter()
        .filter(|target| {
            !parameters
                .iter()
                .any(|parameter| parameter.name() == *target)
        })
        .collect::<Vec<_>>();
    if !missing_recipe_parameters.is_empty() {
        return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
            format!(
                "prediction recipes have no parameter requirements: {missing_recipe_parameters:?}"
            ),
        ));
    }
    let target = inspection
        .architecture_plan()
        .safetensors_architecture()
        .ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArtifact(
                "prediction target is not a SafeTensors architecture".into(),
            )
        })?;
    let source_keys = extension
        .source_keys(target)
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArtifact(error.to_string()))?;
    let target_config = match target.model() {
        SafetensorsModelConfig::DeepSeekV3(args) => EligibleConfig::DeepSeekV3(args),
        SafetensorsModelConfig::DeepSeekV4(args) => EligibleConfig::DeepSeekV4(args),
        SafetensorsModelConfig::Inkling(args) => EligibleConfig::Inkling(args),
        SafetensorsModelConfig::QwenHybrid(args) => EligibleConfig::QwenHybrid(&args.text),
        SafetensorsModelConfig::NemotronH(args) => EligibleConfig::NemotronH(args),
        _ => {
            return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                "prediction target has no supported materialization contract".into(),
            ));
        }
    };
    let target_units = target_config
        .unit_count()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    parameters.retain(|parameter| {
        extra_targets.contains(parameter.name())
            || parameter
                .physical_sources()
                .iter()
                .any(|source| source_keys.contains(source.catalog_key()))
            || matches!(
                parameter.owner(),
                ReplicatedTextParameterOwner::ExecutionUnit { unit, .. }
                    if *unit >= target_units
            )
    });
    let names = parameters
        .iter()
        .map(|parameter| parameter.name().to_owned())
        .collect::<BTreeSet<_>>();
    recipes.retain(|target, _| names.contains(target));
    outputs.retain(|target, _| names.contains(target));
    if parameters.is_empty() {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(
            "prediction extension produced no exact materialization parameters".into(),
        ));
    }
    let parameters = parameters
        .into_iter()
        .map(|parameter| {
            let residency = crate::prediction_extension::residency::parameter_residency(
                extension,
                parameter.name(),
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            Ok(parameter.with_auxiliary_residency(residency))
        })
        .collect::<Result<_, ReplicatedTextRequirementsError>>()?;
    Ok((parameters, recipes, outputs))
}

fn prediction_extension_recipes(
    extension: &crate::configuration::PredictionExtensionPlan,
    source: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<
    BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ReplicatedTextRequirementsError,
> {
    let mut recipes = BTreeMap::new();
    let mut extend = |next: BTreeMap<_, _>| -> Result<(), ReplicatedTextRequirementsError> {
        for (target, recipe) in next {
            if let Some(previous) = recipes.get(&target) {
                if previous != &recipe {
                    return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                        format!("prediction extension conflicts at recipe target {target:?}"),
                    ));
                }
            } else {
                recipes.insert(target, recipe);
            }
        }
        Ok(())
    };
    match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            let target = usize::try_from(args.num_hidden_layers).map_err(|_| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "DeepSeek-V3 target depth exceeds usize".into(),
                )
            })?;
            for depth in 0..extension.depth() {
                extend(
                    crate::deepseek::v3_unit_recipes(source, args, target + depth, true)
                        .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?,
                )?;
            }
        }
        SafetensorsModelConfig::DeepSeekV4(args) => {
            let target = usize::try_from(args.num_hidden_layers).map_err(|_| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "DeepSeek-V4 target depth exceeds usize".into(),
                )
            })?;
            for depth in 0..extension.depth() {
                let expert = crate::deepseek::v4_expert_recipes(source, args, target + depth)
                    .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?;
                extend(
                    eredu_checkpoint::expert::gated_product_materialization_recipes(
                        source, &expert,
                    )
                    .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?,
                )?;
            }
        }
        SafetensorsModelConfig::Inkling(args) => {
            extend(
                crate::inkling::mtp_safetensors_recipes(args, source)
                    .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?,
            )?;
        }
        SafetensorsModelConfig::QwenHybrid(args) => {
            let target = usize::try_from(args.text.num_hidden_layers).map_err(|_| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "Qwen hybrid target depth exceeds usize".into(),
                )
            })?;
            for depth in 0..extension.depth() {
                extend(
                    crate::qwen::hybrid::unit_recipes(source, &args.text, target + depth)
                        .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?,
                )?;
            }
        }
        SafetensorsModelConfig::NemotronH(args) => {
            let policies = args.mtp_policies().map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let pattern = policies
                .len()
                .checked_div(extension.depth())
                .ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(
                        "Nemotron-H prediction pattern is empty".into(),
                    )
                })?;
            for prediction in 0..extension.depth() {
                for relative in 0..pattern {
                    extend(
                        crate::nemotron_h::unit_recipes(
                            source,
                            args,
                            prediction + 1,
                            relative,
                            true,
                        )
                        .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?,
                    )?;
                }
            }
        }
        _ => {}
    }
    Ok(recipes)
}

type FinalizedMaterializationParameters = (
    Vec<ReplicatedTextParameterRequirement>,
    BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
);

fn finalize_materialization_parameters(
    config: &EligibleConfig<'_>,
    mut parameters: Vec<ReplicatedTextParameterRequirement>,
    recipe_source: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    extra_recipes: BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
) -> Result<FinalizedMaterializationParameters, ReplicatedTextRequirementsError> {
    let mut derived_recipes = config
        .derived_recipes(recipe_source)
        .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?;
    for (target, recipe) in extra_recipes {
        if let Some(previous) = derived_recipes.get(&target) {
            if previous != &recipe {
                return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                    format!("prediction extension conflicts at derived target {target:?}"),
                ));
            }
        } else {
            derived_recipes.insert(target, recipe);
        }
    }
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let derived_target_role = |target: &str| {
        if target.ends_with("_scales")
            || target.ends_with("_biases")
            || target.ends_with("_scale_inv")
            || target.ends_with(".scales")
            || target.ends_with(".biases")
            || target.ends_with(".weight_scale_inv")
        {
            ReplicatedTextParameterRole::FormatCompanion
        } else {
            config.parameter_role(target, false, &linear_shapes)
        }
    };
    // Multiple logical consumers of an unchanged physical tensor keep its
    // original encoding and provenance. A source alias is not an encoded-byte
    // tensor operation (notably for native GGUF blocks).
    let original_parameters = parameters.clone();
    let mut direct_alias_targets = BTreeSet::new();
    let mut source_consumers = BTreeMap::<&str, usize>::new();
    for recipe in derived_recipes.values() {
        if let eredu_checkpoint::recipe::DerivedWeightRecipe::Source {
            key,
            selection: eredu_checkpoint::store::TensorSelection::Full,
        } = recipe
        {
            *source_consumers.entry(key).or_default() += 1;
        }
    }
    for (target, recipe) in &derived_recipes {
        let eredu_checkpoint::recipe::DerivedWeightRecipe::Source {
            key,
            selection: eredu_checkpoint::store::TensorSelection::Full,
        } = recipe
        else {
            continue;
        };
        if target == key
            && source_consumers
                .get(key.as_str())
                .copied()
                .unwrap_or_default()
                < 2
        {
            continue;
        }
        let Some(source) = original_parameters
            .iter()
            .find(|p| p.name() == key && p.presence().has_physical_source())
        else {
            continue;
        };
        // Renaming can also change semantic role (for example a source-only
        // routed tensor becoming an executable projection). Such recipes still
        // need the ordinary derived-parameter classification below.
        if source.role() != derived_target_role(target) {
            continue;
        }
        // A schema alias already names the same executable parameter. Keep its
        // canonical-name reconciliation in the existing recipe path; it does
        // not introduce an independently owned logical consumer.
        if original_parameters.iter().any(|parameter| {
            parameter.name() != target && parameter.aliases().iter().any(|alias| alias == target)
        }) {
            continue;
        }
        let replacement = if target == key {
            source.clone()
        } else {
            parameter_requirement(
                target.clone(),
                source.sources().to_vec(),
                source.physical_sources().to_vec(),
                Vec::new(),
                source.source_encoding().cloned(),
                source.physical_shape().map(<[usize]>::to_vec),
                source.logical_shape().to_vec(),
                source.native_executable(),
                source.transform_companions().is_some(),
                source.role(),
                parameter_owner(config, target),
                source.presence().clone(),
            )?
            .with_permitted_native_source_dtypes(source.permitted_native_source_dtypes().to_vec())
        };
        if let Some(index) = parameters.iter().position(|p| p.name() == target) {
            parameters[index] = replacement;
        } else {
            parameters.push(replacement);
        }
        direct_alias_targets.insert(target.clone());
    }
    for target in &direct_alias_targets {
        derived_recipes.remove(target);
    }
    let derived_recipe_outputs = derived_recipes
        .iter()
        .map(|(target, recipe)| {
            recipe
                .infer(recipe_source)
                .map(|metadata| (target.clone(), metadata))
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "derived recipe for {target:?} is invalid: {error}"
                    ))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
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
            .map(|source_key| exact_physical_source(recipe_source, source_key))
            .collect::<Result<Vec<_>, _>>()?;
        let existing = parameters.iter().position(|parameter| {
            parameter.name() == target || parameter.aliases().iter().any(|alias| alias == target)
        });
        // Remaining source recipes inherit logical geometry and native format;
        // independently owned physical aliases were lowered above.
        let alias_source = match recipe {
            eredu_checkpoint::recipe::DerivedWeightRecipe::Source {
                key,
                selection: eredu_checkpoint::store::TensorSelection::Full,
            } => parameters.iter().find(|parameter| parameter.name() == key),
            _ => None,
        };
        let replacement = if let Some(index) = existing {
            let parameter = &parameters[index];
            let role = if parameter.linear_companion().is_some() {
                ReplicatedTextParameterRole::FormatCompanion
            } else if matches!(config, EligibleConfig::GptOss(_))
                && target.contains(".mlp.experts.")
                && target.ends_with("_proj_bias")
            {
                ReplicatedTextParameterRole::LinearBias
            } else {
                derived_target_role(target)
            };
            let owner = parameter_owner(config, target);
            let native = parameter.native_executable();
            let mut logical_shape = parameter.logical_shape().to_vec();
            if output.dtype == eredu_checkpoint::recipe::RecipeDtype::U32
                && logical_shape == output.shape
            {
                let packed_bits = match native {
                    LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                    LinearFormat::MxFp4 | LinearFormat::NvFp4 => Some(4),
                    _ => None,
                };
                if let (Some(bits), Some(extent)) = (packed_bits, logical_shape.last_mut()) {
                    *extent = extent
                        .checked_mul(32)
                        .and_then(|value| value.checked_div(bits))
                        .ok_or_else(|| {
                            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                                "derived packed geometry for {target:?} overflowed"
                            ))
                        })?;
                }
            }
            let mut replacement = ReplicatedTextParameterRequirement::new(
                target.clone(),
                recipe
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                physical_sources,
                parameter.aliases().to_vec(),
                Some(SourceTensorEncoding::RecipeOutput(recipe_stored_dtype(
                    &output.dtype,
                )?)),
                Some(output.shape.clone()),
                logical_shape,
                native,
                role,
                owner,
                ReplicatedTextParameterPresence::Derived {
                    recipe: format!("architecture-output:{target}"),
                },
                parameter.transform_constraint(),
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            if let Some((scale, bias)) = parameter.transform_companions() {
                replacement =
                    replacement
                        .with_transform_companions(scale, bias)
                        .map_err(|error| {
                            ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                        })?;
            }
            if let Some((role, primary)) = parameter.linear_companion() {
                replacement = replacement
                    .with_linear_companion(role, primary.to_owned())
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?;
            }
            replacement.with_permitted_native_source_dtypes(
                parameter.permitted_native_source_dtypes().to_vec(),
            )
        } else {
            let role = derived_target_role(target);
            let mut logical_shape = linear_shapes
                .get(target)
                .cloned()
                .or_else(|| alias_source.map(|source| source.logical_shape().to_vec()))
                .unwrap_or_else(|| output.shape.clone());
            let native = if role == ReplicatedTextParameterRole::FormatCompanion {
                LinearFormat::Dense
            } else {
                alias_source.map_or_else(
                    || config.native_format_for_shape(target, &logical_shape),
                    |source| source.native_executable(),
                )
            };
            if output.dtype == eredu_checkpoint::recipe::RecipeDtype::U32
                && logical_shape == output.shape
            {
                let packed_bits = match native {
                    LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                    LinearFormat::MxFp4 | LinearFormat::NvFp4 => Some(4),
                    _ => None,
                };
                if let (Some(bits), Some(extent)) = (packed_bits, logical_shape.last_mut()) {
                    *extent = extent
                        .checked_mul(32)
                        .and_then(|value| value.checked_div(bits))
                        .ok_or_else(|| {
                            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                                "derived packed geometry for {target:?} overflowed"
                            ))
                        })?;
                }
            }
            parameter_requirement(
                target.clone(),
                recipe
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                physical_sources,
                Vec::new(),
                Some(SourceTensorEncoding::RecipeOutput(recipe_stored_dtype(
                    &output.dtype,
                )?)),
                Some(output.shape.clone()),
                logical_shape,
                native,
                role == ReplicatedTextParameterRole::LinearWeight,
                role,
                parameter_owner(config, target),
                ReplicatedTextParameterPresence::Derived {
                    recipe: format!("architecture-output:{target}"),
                },
            )?
            .with_permitted_native_source_dtypes(
                alias_source
                    .map(|source| source.permitted_native_source_dtypes().to_vec())
                    .unwrap_or_default(),
            )
        };
        if let Some(index) = existing {
            parameters[index] = replacement;
        } else {
            parameters.push(replacement);
        }
    }
    parameters.retain(|parameter| {
        direct_alias_targets.contains(parameter.name())
            || recipe_targets.contains(parameter.name())
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
    for parameter in &mut parameters {
        let optional_media_transform = match config {
            EligibleConfig::Gemma4(_) => {
                parameter.name().starts_with("model.vision_")
                    || parameter.name().starts_with("model.audio_")
                    || parameter.name().starts_with("model.embed_vision.")
                    || parameter.name().starts_with("model.embed_audio.")
            }
            EligibleConfig::Muse(_) => {
                parameter.name().starts_with("model.vision_")
                    || parameter.name().starts_with("model.audio_tower.")
            }
            EligibleConfig::QwenVl(_) => parameter.name().starts_with("model.visual."),
            _ => false,
        };
        if optional_media_transform
            && matches!(
                parameter.transform_constraint(),
                ParameterTransformConstraint::Linear { .. }
            )
        {
            *parameter = parameter
                .clone()
                .with_optional_transform_alignment(32)
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
        }
    }
    Ok((
        finish_parameters(parameters)?,
        derived_recipes,
        derived_recipe_outputs,
    ))
}

pub(crate) fn recipe_stored_dtype(
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
            ));
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
        SafetensorsModelConfig::NemotronLabsDiffusion(args) if args.vision.is_some() => {
            Err(ReplicatedTextIneligibility::CompositeInput)
        }
        SafetensorsModelConfig::NemotronLabsDiffusion(args) => {
            Ok(EligibleConfig::NemotronLabsDiffusion(args))
        }
        SafetensorsModelConfig::DiffusionGemma(_) => {
            Err(ReplicatedTextIneligibility::BlockRefinement)
        }
        SafetensorsModelConfig::Qwen4Exp(_, _) => Err(ReplicatedTextIneligibility::Routed),
        SafetensorsModelConfig::Nanbeige(args) => Ok(EligibleConfig::Nanbeige(args)),
        SafetensorsModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
        SafetensorsModelConfig::Gemma2(args) => Ok(EligibleConfig::Gemma2(args)),
        SafetensorsModelConfig::K2Horizon(args) if args.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::K2Horizon(args) => Ok(EligibleConfig::K2Horizon(args)),
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
        | SafetensorsModelConfig::Muse(_)
        | SafetensorsModelConfig::QwenVl(_) => Err(ReplicatedTextIneligibility::CompositeInput),
        SafetensorsModelConfig::Moshi(_) | SafetensorsModelConfig::VoiceChat(_) => {
            Err(ReplicatedTextIneligibility::Realtime)
        }
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
        SafetensorsModelConfig::DeepSeekV3(args) => Ok(EligibleConfig::DeepSeekV3(args)),
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
        GgufModelConfig::NemotronLabsDiffusion(plan) => {
            Ok(EligibleConfig::NemotronLabsDiffusion(plan.text_config()))
        }
        GgufModelConfig::DiffusionGemma(_) => Err(ReplicatedTextIneligibility::BlockRefinement),
        GgufModelConfig::VoiceChat(_) => Err(ReplicatedTextIneligibility::Realtime),
        GgufModelConfig::Qwen4Exp(_) => Err(ReplicatedTextIneligibility::Routed),
        GgufModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
        GgufModelConfig::K2Horizon(args) if args.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::K2Horizon(args) => Ok(EligibleConfig::K2Horizon(args)),
        GgufModelConfig::Gemma2(args) => Ok(EligibleConfig::Gemma2(args)),
        GgufModelConfig::Nanbeige(args) => Ok(EligibleConfig::Nanbeige(args)),
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
        GgufModelConfig::Gemma4(_) | GgufModelConfig::Inkling(_) | GgufModelConfig::Muse(_) => {
            Err(ReplicatedTextIneligibility::CompositeInput)
        }
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
        GgufModelConfig::DeepSeekV3(args) => Ok(EligibleConfig::DeepSeekV3(args)),
    }
}

pub(crate) fn execution_topology(
    plan: &ArtifactArchitecturePlan,
) -> Result<
    Option<eredu_runtime::execution_topology::TextExecutionTopology>,
    ReplicatedTextRequirementsError,
> {
    let Ok(config) = eligible_config(plan) else {
        return Ok(None);
    };
    config
        .execution_topology()
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
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
        config @ (EligibleConfig::Gemma2(_)
        | EligibleConfig::Llama(_)
        | EligibleConfig::Nanbeige(_)
        | EligibleConfig::K2Horizon(_)
        | EligibleConfig::Qwen(_)) => Ok(config),
        EligibleConfig::Lfm2(_)
        | EligibleConfig::KimiLinear(_)
        | EligibleConfig::NemotronH(_)
        | EligibleConfig::QwenHybrid(_)
        | EligibleConfig::GptOss(_)
        | EligibleConfig::DeepSeekV3(_)
        | EligibleConfig::DeepSeekV4(_)
        | EligibleConfig::DiffusionGemma(_)
        | EligibleConfig::NemotronLabsDiffusion(_)
        | EligibleConfig::Gemma4(_)
        | EligibleConfig::Inkling(_)
        | EligibleConfig::Muse(_)
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

pub(crate) fn exact_physical_source(
    catalog: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    key: &str,
) -> Result<ReplicatedTextPhysicalSource, ReplicatedTextRequirementsError> {
    let provenance = catalog.source_provenance(key).map_err(|error| {
        ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "admitted source {key:?} has no exact provenance: {error}"
        ))
    })?;
    let metadata = catalog.source_metadata(key).map_err(|error| {
        ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "admitted source {key:?} has no exact metadata: {error}"
        ))
    })?;
    if matches!(
        &provenance.source_encoding,
        SourceTensorEncoding::RecipeOutput(_)
    ) {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "admitted physical source {key:?} has a recipe-output encoding"
        )));
    }
    let shard = provenance.backing_shard.ok_or_else(|| {
        ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "admitted source {key:?} has no backing shard"
        ))
    })?;
    ReplicatedTextPhysicalSource::new(
        provenance.catalog_key,
        provenance.physical_tensor,
        shard,
        provenance.output,
        provenance.source_encoding,
        metadata.encoded_byte_len,
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArtifact(error.to_string()))
}

fn finish_parameters_with_tied_output(
    mut parameters: Vec<ReplicatedTextParameterRequirement>,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    if config.tied_embeddings() {
        // A redundant checkpoint head is admitted for source validation, but
        // execution uses the embedding and its own declared companions.
        let redundant = |parameter: &ReplicatedTextParameterRequirement| {
            parameter.name() == "lm_head.weight"
                || parameter
                    .linear_companion()
                    .is_some_and(|(_, primary)| primary == "lm_head.weight")
        };
        let redundant_sources = parameters
            .iter()
            .filter(|parameter| redundant(parameter))
            .flat_map(|parameter| parameter.sources().iter().cloned())
            .collect();
        parameters.retain(|parameter| !redundant(parameter));
        parameters.push(
            parameter_requirement(
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
            )?
            .with_admitted_redundant_sources(redundant_sources)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?,
        );
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
    let owner = if role == ReplicatedTextParameterRole::Embedding {
        ReplicatedTextParameterOwner::StaticRole("embedding".into())
    } else {
        owner
    };
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
    let requirement = ReplicatedTextParameterRequirement::new(
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
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    if transformable {
        let name = requirement.name();
        let grouped = !name.ends_with(".weight")
            && (name.contains(".experts.")
                || name.ends_with(".gate_up_proj")
                || name.ends_with(".down_proj"));
        let (scale, bias) = if grouped {
            (format!("{name}_scales"), format!("{name}_biases"))
        } else {
            let prefix = name.strip_suffix(".weight").unwrap_or(name);
            (format!("{prefix}.scales"), format!("{prefix}.biases"))
        };
        requirement
            .with_transform_companions(scale, bias)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })
    } else {
        Ok(requirement)
    }
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
    if let EligibleConfig::Nanbeige(args) = config {
        if let Some((layer, suffix)) = name
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once('.'))
        {
            if let Ok(layer) = layer.parse::<usize>() {
                let schedule =
                    crate::decoder::DecoderSchedule::new(*args).expect("admitted decoder schedule");
                let operation = schedule.body_operation(layer)
                    + usize::from(suffix.starts_with("output_norm."));
                return ReplicatedTextParameterOwner::ExecutionUnit {
                    group: crate::decoder::TEXT_DECODER_EXECUTION_GROUP.into(),
                    unit: schedule.owner(operation),
                };
            }
        }
    }
    if matches!(config, EligibleConfig::NemotronLabsDiffusion(_)) {
        if let Some(owner) = execution_unit(
            "draft_adapter.layers.",
            crate::decoder::TEXT_DECODER_EXECUTION_GROUP,
        ) {
            return owner;
        }
        if let Some(owner) = execution_unit("model.vision_tower.transformer.layers.", "vision") {
            return owner;
        }
        if name.starts_with("model.multi_modal_projector.") {
            return ReplicatedTextParameterOwner::ExecutionUnit {
                group: "vision_projection".into(),
                unit: 0,
            };
        }
        if name.starts_with("model.vision_tower.") {
            return ReplicatedTextParameterOwner::StaticRole("vision_embedding".into());
        }
    }
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
        EligibleConfig::Muse(args) => args
            .audio_config
            .as_ref()
            .and_then(|audio| audio.execution_units().ok())
            .and_then(|units| units.into_iter().position(|unit| unit.owns_parameter(name)))
            .map(|unit| ReplicatedTextParameterOwner::ExecutionUnit {
                group: crate::muse::audio::execution::AUDIO_EXECUTION_GROUP.into(),
                unit,
            })
            .or_else(|| {
                execution_unit(
                    "model.vision_tower.layers.",
                    crate::muse::model::VISION_EXECUTION_GROUP,
                )
            })
            .or_else(|| execution_unit("model.layers.", crate::muse::model::TEXT_EXECUTION_GROUP)),
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
    if matches!(config, EligibleConfig::DiffusionGemma(_)) {
        if let Some(owner) = execution_unit(
            "model.encoder.vision_tower.encoder.layers.",
            crate::diffusion_gemma::composite::VISION_EXECUTION_GROUP,
        ) {
            return owner;
        }
        if let Some(rest) = name.strip_prefix("model.encoder.language_model.layers.") {
            if let Some(unit) = rest
                .split('.')
                .next()
                .and_then(|value| value.parse::<usize>().ok())
            {
                return ReplicatedTextParameterOwner::ExecutionUnit {
                    group: config.execution_group().into(),
                    unit,
                };
            }
        }
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
    if let EligibleConfig::Muse(args) = config {
        if let Some(consumers) = crate::muse::parallel::vision_static_consumer_units(args, name) {
            return ReplicatedTextParameterOwner::StaticUnitConsumers {
                role: "vision".into(),
                consumers: consumers
                    .map(|unit| (crate::muse::model::VISION_EXECUTION_GROUP.into(), unit))
                    .collect(),
            };
        }
    }
    let vision = match config {
        EligibleConfig::QwenVl(args) => {
            Some((&args.vision, crate::qwen::vl::VISION_EXECUTION_GROUP))
        }
        EligibleConfig::QwenCompositeHybrid(args) => args
            .vision
            .as_ref()
            .map(|vision| (vision, crate::qwen::hybrid::VISION_EXECUTION_GROUP)),
        _ => None,
    };
    if let Some((vision, group)) = vision {
        let consumer = if name.starts_with("model.visual.merger.") {
            vision.layer_count().checked_sub(1)
        } else if let Some(rest) = name.strip_prefix("model.visual.deepstack_merger_list.") {
            rest.split('.')
                .next()
                .and_then(|index| index.parse::<usize>().ok())
                .and_then(|index| vision.deepstack_layers().get(index).copied())
                .and_then(|unit| usize::try_from(unit).ok())
        } else {
            None
        };
        if let Some(unit) = consumer {
            return ReplicatedTextParameterOwner::StaticUnitConsumers {
                role: "vision".into(),
                consumers: vec![(group.into(), unit)],
            };
        }
    }
    let embedding = config.embedding_name();
    let embedding_prefix = embedding.strip_suffix("weight").unwrap_or(&embedding);
    let role = if matches!(config, EligibleConfig::DiffusionGemma(_)) {
        if name.starts_with("model.decoder.self_conditioning.") {
            "conditioning"
        } else if name.starts_with("model.encoder.vision_tower.") {
            "vision"
        } else if name.starts_with("model.encoder.embed_vision.") {
            "vision_projection"
        } else if name == embedding || name.starts_with(embedding_prefix) {
            "embedding"
        } else if name.starts_with("lm_head.") {
            "output"
        } else {
            "norm"
        }
    } else if matches!(config, EligibleConfig::Gemma4(_)) {
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
    } else if matches!(config, EligibleConfig::Muse(_)) && name.starts_with("model.vision_") {
        "vision"
    } else if matches!(config, EligibleConfig::Muse(_)) && name == "model.embed_tokens.weight" {
        "embedding"
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
    let primaries = parameters
        .iter()
        .filter(|parameter| {
            matches!(
                parameter.role(),
                ReplicatedTextParameterRole::LinearWeight | ReplicatedTextParameterRole::Embedding
            )
        })
        .map(|parameter| parameter.name().to_owned())
        .collect::<Vec<_>>();
    let mut companions = BTreeMap::<String, Vec<_>>::new();
    for primary in &primaries {
        let prefix = primary.strip_suffix(".weight").unwrap_or(primary);
        let names = [
            (
                format!("{prefix}.scales"),
                eredu_nn::LinearCompanionRole::Scale,
            ),
            (
                format!("{primary}_scales"),
                eredu_nn::LinearCompanionRole::Scale,
            ),
            (
                format!("{prefix}.weight_scale_inv"),
                eredu_nn::LinearCompanionRole::Scale,
            ),
            (
                format!("{primary}_scale_inv"),
                eredu_nn::LinearCompanionRole::Scale,
            ),
            (
                format!("{prefix}.biases"),
                eredu_nn::LinearCompanionRole::AffineBias,
            ),
            (
                format!("{primary}_biases"),
                eredu_nn::LinearCompanionRole::AffineBias,
            ),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
        for (name, role) in names {
            companions
                .entry(name)
                .or_default()
                .push((role, primary.clone()));
        }
    }
    for parameter in &mut parameters {
        if parameter.role() != ReplicatedTextParameterRole::FormatCompanion {
            continue;
        }
        if parameter.linear_companion().is_some() {
            continue;
        }
        let candidates = companions
            .get(parameter.name())
            .map(Vec::as_slice)
            .unwrap_or_default();
        match candidates {
            [(role, primary)] => {
                *parameter = parameter
                    .clone()
                    .with_linear_companion(*role, primary.clone())
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?;
            }
            [] => {}
            _ => {
                return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                    format!(
                        "format companion {:?} ambiguously names multiple linear weights",
                        parameter.name()
                    ),
                ));
            }
        }
    }
    Ok(parameters)
}

pub(crate) fn stored_dtype(
    dtype: &TensorDtype,
) -> Result<StoredDtype, ReplicatedTextRequirementsError> {
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
#[derive(Debug, Clone, thiserror::Error)]
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
    input_modalities: InputModalities,
    decoder: CompositeTextDecoderStrategy,
    raw_processor: bool,
    media_projector: bool,
    processor: eredu_runtime::ProcessorExecutionRequirements,
    execution: ReplicatedTextRequirements,
    routed: Option<crate::RoutedTextRequirements>,
    partition_boundaries: crate::composite_partitioned::CompositePartitionBoundaries,
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
}

impl PartialEq for CompositeTextRequirements {
    fn eq(&self, other: &Self) -> bool {
        self.architecture_identity == other.architecture_identity
            && self.input_modalities == other.input_modalities
            && self.decoder == other.decoder
            && self.raw_processor == other.raw_processor
            && self.media_projector == other.media_projector
            && self.processor == other.processor
            && self.execution == other.execution
            && self.routed == other.routed
            && self.partition_boundaries == other.partition_boundaries
            && self.inspection.format() == other.inspection.format()
            && self.inspection.tensors() == other.inspection.tensors()
            && self.inspection.safetensors_shards() == other.inspection.safetensors_shards()
    }
}

impl CompositeTextRequirements {
    /// Combines already-authored routed execution and input requirements without
    /// reconstructing family configuration or checkpoint recipes.
    pub(crate) fn from_routed(
        inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
        architecture_identity: String,
        routed: crate::RoutedTextRequirements,
        processor: eredu_runtime::ProcessorExecutionRequirements,
        input_modalities: InputModalities,
        partition_boundaries: crate::composite_partitioned::CompositePartitionBoundaries,
    ) -> Result<Self, String> {
        let execution = routed.text().clone();
        partition_boundaries.validate(execution.execution_graph(), execution.execution_units())?;
        Ok(Self {
            architecture_identity,
            input_modalities,
            decoder: CompositeTextDecoderStrategy::Routed,
            raw_processor: inspection.architecture_plan().has_processor(),
            media_projector: inspection
                .architecture_plan()
                .gguf_media_projector()
                .is_some(),
            processor,
            execution,
            routed: Some(routed),
            partition_boundaries,
            inspection: inspection
                .clone()
                .map_architecture_plan(ArtifactArchitecturePlan::without_validation),
        })
    }

    /// Exact retained non-decoder continuation and cross-group wire contracts.
    pub fn partition_boundaries(
        &self,
    ) -> &crate::composite_partitioned::CompositePartitionBoundaries {
        &self.partition_boundaries
    }

    pub(crate) fn with_state_layout(
        mut self,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self, eredu_runtime::ReplicatedTextContractError> {
        self.execution = self.execution.with_state_layout(layout.clone())?;
        self.routed = self
            .routed
            .map(|routed| routed.with_state_layout(layout.clone()))
            .transpose()?;
        Ok(self)
    }

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
        self.execution.execution_graph()
    }

    /// Exact unit geometry in canonical group order.
    pub const fn execution_units(&self) -> &eredu_runtime::ExecutionUnitLayout {
        self.execution.execution_units()
    }

    /// Architecture-owned transport, kind, and optional-root declarations.
    pub fn group_transports(&self) -> &[eredu_runtime::ArchitectureGroupTransport] {
        self.execution.group_transports()
    }

    /// Complete target mutable-state geometry.
    pub const fn state_layout(&self) -> &eredu_runtime::StateLayout {
        self.execution.state_layout()
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
    architecture: A,
    source_architecture: Option<A>,
    requirements: CompositeTextRequirements,
    contract: eredu_runtime::PreparedReplicatedTextContract,
    processor: eredu_runtime::SelectedProcessorExecution,
    admission: C,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

/// Source-independent requirements retained by a prepared routed composite.
/// Artifact inspection remains with cold admission and typed dispatch, rather
/// than being captured by a constructed session or its processor policy.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedCompositeRequirements {
    execution: crate::RoutedTextRequirements,
    processor: eredu_runtime::ProcessorExecutionRequirements,
}
impl RoutedCompositeRequirements {
    /// Exact target/encoder parameters, grouped banks, row lookups and state.
    pub const fn execution(&self) -> &crate::RoutedTextRequirements {
        &self.execution
    }
    /// Exact raw, prepared and projected input mechanism requirements.
    pub const fn processor(&self) -> &eredu_runtime::ProcessorExecutionRequirements {
        &self.processor
    }
}

/// Checked composite architecture whose target decoder uses grouped providers.
pub struct PreparedRoutedCompositeTextArchitecture<A, C> {
    routed: crate::PreparedRoutedTextArchitecture<A>,
    requirements: RoutedCompositeRequirements,
    processor: eredu_runtime::SelectedProcessorExecution,
    admission: C,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

impl<A, C> PreparedRoutedCompositeTextArchitecture<A, C> {
    /// Pairs already checked modules with their exact retained composite policy.
    /// The selected authority is supplied explicitly so every bank, row provider,
    /// state and materialization choice must agree before a visitor receives it.
    pub(crate) fn from_routed_declared(
        routed: crate::PreparedRoutedTextArchitecture<A>,
        requirements: crate::RoutedTextRequirements,
        selected: &crate::SelectedRoutedTextRealization,
        processor: eredu_runtime::SelectedProcessorExecution,
        processor_requirements: eredu_runtime::ProcessorExecutionRequirements,
        admission: C,
    ) -> Result<Self, String> {
        crate::routed_text::validate_selected_routed_handoff(&requirements, selected)
            .map_err(|error| error.to_string())?;
        if processor.requirements() != &processor_requirements {
            return Err("selected composite processor differs from exact requirements".into());
        }
        if routed.text().selected() != selected.text()
            || routed.bank_residency() != selected.bank_residency()
            || routed.banks() != selected.banks()
            || routed.row_lookups().map(|rows| rows.plan()) != selected.row_lookups()
        {
            return Err("prepared composite modules differ from selected execution".into());
        }
        let capability_estimate = routed.text().capability_estimate().clone();
        let effective_model_type = routed.text().effective_model_type().to_owned();
        Ok(Self {
            routed,
            requirements: RoutedCompositeRequirements {
                execution: requirements,
                processor: processor_requirements,
            },
            processor,
            admission,
            capability_estimate,
            effective_model_type,
        })
    }

    /// Exact architecture, artifact, processor, and grouped-bank requirements.
    pub const fn requirements(&self) -> &RoutedCompositeRequirements {
        &self.requirements
    }

    /// Authoritative processor mechanism selection.
    pub const fn processor(&self) -> &eredu_runtime::SelectedProcessorExecution {
        &self.processor
    }

    /// Selected routed text, state, weight, and bank realization.
    pub const fn routed(&self) -> &crate::PreparedRoutedTextArchitecture<A> {
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
        crate::PreparedRoutedTextArchitecture<A>,
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
        A,
        Option<A>,
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

    /// Binds a routed composite target and its independently prepared predictor.
    /// Native adaptation retains processor policy and uses the shared speculative driver.
    fn visit_prediction<A, W>(
        self,
        _prepared: PreparedRoutedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
        >,
        _prediction: W,
        _target_source: eredu_checkpoint::store::SharedCheckpointSource,
        _provider_source: eredu_checkpoint::store::SharedCheckpointSource,
        _binding: crate::prepared_execution::PredictionBinding,
    ) -> Result<Self::Output, crate::prepared_execution::PreparedExecutionError<Self::Error>>
    where
        B: eredu_nn::BlockwiseAttentionBackend + eredu_nn::HyperNeuralBackend,
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        W: crate::prediction_extension::PreparedRoutedPrediction<B, A>,
    {
        Err(crate::prepared_execution::PreparedExecutionError::UnavailablePrediction)
    }

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

/// Family-blind backend visitor for an exact composite prediction target.
pub trait CompositePredictionTargetVisitor<B, S, M>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Records that validated dispatch is about to construct architecture modules.
    fn construction_started(&mut self) {}

    /// Receives a direct composite target after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A: crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;

    /// Receives a routed composite target after exact extension pairing.
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A: crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Pairs one materialized extension with an exact direct composite target.
pub fn visit_composite_prediction_target_architecture<B, S, M, A, V>(
    prepared: PreparedCompositeTextArchitecture<
        A,
        <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + 'static,
    A: crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    V: CompositePredictionTargetVisitor<B, S, M>,
{
    type Target<A> = A;
    let extension = <Target<A> as crate::prediction_extension::MaterializedPredictionTarget<
        B,
    >>::pair_prediction_extension(extension)
    .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension, store)
        .map_err(ReplicatedTextDispatchError::Backend)
}

/// Pairs one materialized extension with an exact routed composite target.
pub fn visit_routed_composite_prediction_target_architecture<B, S, M, A, V>(
    prepared: PreparedRoutedCompositeTextArchitecture<
        A,
        <A as crate::composite_execution::CompositeArchitecture<B, S>>::AdmissionConfig,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + 'static,
    A: crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    V: CompositePredictionTargetVisitor<B, S, M>,
{
    type Target<A> = A;
    let extension = <Target<A> as crate::prediction_extension::MaterializedPredictionTarget<
        B,
    >>::pair_prediction_extension(extension)
    .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit_routed(prepared, extension, store)
        .map_err(ReplicatedTextDispatchError::Backend)
}

pub(crate) enum CompositeConfig<'a> {
    NemotronLabsDiffusion(&'a crate::nemotron_labs_diffusion::FamilyConfig),
    Diffusion(&'a crate::diffusion_gemma::FamilyConfig),
    Gemma4(&'a crate::gemma4::FamilyConfig),
    Inkling(&'a crate::inkling::ModelArgs),
    Muse(&'a crate::muse::DecoderConfig),
    QwenVl(&'a crate::qwen::vl::ModelArgs),
    QwenHybrid(&'a crate::qwen::hybrid::ParsedHybridConfig),
}

impl CompositeConfig<'_> {
    fn requirement_config(&self) -> EligibleConfig<'_> {
        match self {
            Self::NemotronLabsDiffusion(args) => EligibleConfig::NemotronLabsDiffusion(args),
            Self::Diffusion(args) => EligibleConfig::DiffusionGemma(args),
            Self::Gemma4(args) => EligibleConfig::Gemma4(args),
            Self::Inkling(args) => EligibleConfig::Inkling(args),
            Self::Muse(args) => EligibleConfig::Muse(args),
            Self::QwenVl(args) => EligibleConfig::QwenVl(args),
            Self::QwenHybrid(args) => EligibleConfig::QwenCompositeHybrid(args),
        }
    }

    fn architecture_identity(&self) -> String {
        match self {
            Self::NemotronLabsDiffusion(args) => args.architecture_fingerprint(),
            Self::Diffusion(args) => args.architecture_fingerprint(),
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
            Self::NemotronLabsDiffusion(_) => InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            },
            Self::Diffusion(args) => args.input_modalities(),
            Self::Gemma4(args) => args.input_modalities(),
            Self::Inkling(args) => args.input_modalities(),
            Self::Muse(args) => InputModalities {
                text: true,
                image: args.vision_config.is_some(),
                audio: args.audio_config.is_some(),
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
            Self::NemotronLabsDiffusion(_) => false,
            Self::Diffusion(args) => args.has_sparse_moe_layers(),
            Self::Gemma4(args) => args.text.num_experts.is_some_and(|count| count > 0),
            Self::Inkling(args) => args.text_config.has_sparse_moe_layers(),
            Self::Muse(args) => args.num_experts > 0,
            Self::QwenVl(args) => args.text.is_moe(),
            Self::QwenHybrid(args) => args.text.is_moe(),
        }
    }

    fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        match self {
            Self::NemotronLabsDiffusion(args) => {
                crate::nemotron_labs_diffusion::definition::composite(args, None)
            }
            Self::Diffusion(args) => {
                crate::diffusion_gemma::composite::component_definition(args, None)
            }
            Self::Gemma4(args) => crate::gemma4::model::component_definition(args, None),
            Self::Inkling(args) => crate::inkling::model::component_definition(args, None),
            Self::Muse(args) => crate::muse::model::component_definition(args, None),
            Self::QwenVl(args) => crate::qwen::vl::component_definition(args, None),
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::conditional_component_definition(args, None)
            }
        }
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
            GgufMediaProjectorConfig::Muse(args) => CompositeConfig::Muse(args),
            GgufMediaProjectorConfig::NemotronLabsDiffusion { config, .. } => {
                CompositeConfig::NemotronLabsDiffusion(config)
            }
            GgufMediaProjectorConfig::Qwen3Vl(args) => CompositeConfig::QwenVl(args),
            GgufMediaProjectorConfig::Qwen35(args) => CompositeConfig::QwenHybrid(args),
            GgufMediaProjectorConfig::Qwen4Exp(_) => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                    "Flash-Next composite requires its selected routed architecture declaration"
                        .into(),
                ));
            }
            GgufMediaProjectorConfig::Qwen3VlPending(_)
            | GgufMediaProjectorConfig::Qwen35Pending(_)
            | GgufMediaProjectorConfig::Qwen4ExpPending { .. } => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                    "composite GGUF media token identities are unresolved".into(),
                ));
            }
        }));
    }
    let config = match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(SafetensorsModelConfig::NemotronLabsDiffusion(args)), None)
            if args.vision.is_some() =>
        {
            Some(CompositeConfig::NemotronLabsDiffusion(args))
        }
        (Some(SafetensorsModelConfig::DiffusionGemma(args)), None) if args.vision.is_some() => {
            Some(CompositeConfig::Diffusion(args))
        }
        (Some(SafetensorsModelConfig::Gemma4(args)), None) => Some(CompositeConfig::Gemma4(args)),
        (Some(SafetensorsModelConfig::Inkling(args)), None) => Some(CompositeConfig::Inkling(args)),
        (Some(SafetensorsModelConfig::Muse(args)), None) => Some(CompositeConfig::Muse(args)),
        (Some(SafetensorsModelConfig::QwenVl(args)), None) => Some(CompositeConfig::QwenVl(args)),
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None) if args.vision.is_some() => {
            Some(CompositeConfig::QwenHybrid(args))
        }
        (None, Some(GgufModelConfig::Gemma4(args))) => Some(CompositeConfig::Gemma4(args)),
        (None, Some(GgufModelConfig::Inkling(args))) => Some(CompositeConfig::Inkling(args)),
        (None, Some(GgufModelConfig::Muse(args))) => Some(CompositeConfig::Muse(args)),
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
    if let Some(config) = crate::routed_text::diffusion_args(inspection) {
        return EligibleConfig::DiffusionGemma(config)
            .partitioned_boundary_schema(topology)
            .map_err(ReplicatedTextRequirementsError::InvalidArchitecture);
    }
    let plan = inspection.architecture_plan();
    let config = match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(SafetensorsModelConfig::K2Horizon(args)), None)
        | (None, Some(GgufModelConfig::K2Horizon(args))) => EligibleConfig::K2Horizon(args),
        (Some(SafetensorsModelConfig::Qwen(args)), None) if args.is_moe() => {
            EligibleConfig::Qwen(args)
        }
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() && args.text.mtp_num_hidden_layers == 0 =>
        {
            EligibleConfig::QwenHybrid(&args.text)
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
            CompositeConfig::NemotronLabsDiffusion(_)
            | CompositeConfig::Diffusion(_)
            | CompositeConfig::Gemma4(_)
            | CompositeConfig::QwenVl(_) => vec![
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
    let recipe_source = artifact::target_catalog(inspection)?;
    let (owner_group, plan, catalog, routes_per_token) = match config {
        CompositeConfig::NemotronLabsDiffusion(_) => return Ok(None),
        CompositeConfig::Diffusion(args) => {
            let owner = eredu_runtime::ExecutionGroupId::new(
                crate::diffusion_gemma::model::TEXT_EXECUTION_GROUP,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            let plan = crate::diffusion_gemma::model::expert_realization_plan(
                args,
                None,
                replicated_expert_topology()
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "selected shared stack has no routed bank".into(),
                )
            })?;
            let catalog =
                crate::diffusion_gemma::expert_residency_catalog(recipe_source.as_ref(), args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
            (
                owner,
                plan,
                catalog,
                args.text.top_k_experts.unwrap() as usize,
            )
        }

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
            let owner_group =
                eredu_runtime::ExecutionGroupId::new(crate::muse::model::TEXT_EXECUTION_GROUP)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?;
            let plan = crate::muse::expert_realization_plan(
                args,
                None,
                replicated_expert_topology()
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "Muse routed requirements have no expert realization".into(),
                )
            })?;
            let catalog = crate::muse::expert_residency_catalog(recipe_source.as_ref(), args)
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
    inspection
        .architecture_plan()
        .validation(inspection.admission_token())
        .composite
        .get_or_init(|| derive_composite_text_requirements(inspection))
        .clone()
}

fn derive_composite_text_requirements(
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
            return Err(ReplicatedTextIneligibility::EmbeddedPrediction.into());
        }
        CompositeConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
            return Err(ReplicatedTextIneligibility::EmbeddedPrediction.into());
        }
        _ => {}
    }
    let definition = config
        .component_definition()
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    let processor = composite_processor_requirements(inspection.architecture_plan(), &config)?;
    let mut execution = replicated_text_requirements_for_structure(
        inspection,
        config.requirement_config(),
        Some(definition),
    )?;
    if let CompositeConfig::NemotronLabsDiffusion(args) = &config {
        execution = execution.with_block_refinement(args.generation_geometry());
    }
    if let CompositeConfig::Diffusion(args) = &config {
        execution = execution.with_block_refinement(
            eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
                canvas_length: args.canvas_length,
                vocabulary_size: std::num::NonZeroU32::new(args.text.vocab_size as u32).unwrap(),
            },
        );
    }
    let routed = composite_routed_requirements(inspection, &config, execution.clone())?;
    let partition_boundaries =
        crate::composite_partitioned::composite_partition_boundaries(&config)
            .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    if let Some(routed) = routed {
        return CompositeTextRequirements::from_routed(
            inspection,
            config.architecture_identity(),
            routed,
            processor,
            config.input_modalities(),
            partition_boundaries,
        )
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture);
    }
    partition_boundaries
        .validate(execution.execution_graph(), execution.execution_units())
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    Ok(CompositeTextRequirements {
        partition_boundaries,
        architecture_identity: config.architecture_identity(),
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
        routed: None,
        inspection: inspection
            .clone()
            .map_architecture_plan(ArtifactArchitecturePlan::without_validation),
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
    let contract = eredu_runtime::prepare_layered_text_contract::<_, B, S>(
        &architecture,
        source_architecture.as_ref(),
        selected,
        &prompt_cache_architecture_identity,
        if A::block_refinement_geometry(&admission).is_some() {
            eredu_runtime::ReplicatedTextOutputSelection::AllSequencePositions
        } else {
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition
        },
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
    let routed = crate::routed_text::prepare_routed_architecture_handoff::<B, S, _>(
        architecture,
        source_architecture,
        routed_requirements.clone(),
        selected.clone(),
        None,
        capability_estimate.clone(),
        effective_model_type.clone(),
        prompt_cache_architecture_identity,
        context,
    )?;
    PreparedRoutedCompositeTextArchitecture::from_routed_declared(
        routed,
        routed_requirements,
        &selected,
        processor,
        requirements.processor_execution().clone(),
        admission,
    )
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
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    V: CompositeTextArchitectureVisitor<B, S>,
{
    validate_store_handoff(
        requirements.execution(),
        store.as_ref(),
        StoreHandoffScope::Primary,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    let retained = requirements.inspection.clone();
    let config = composite_config(retained.architecture_plan())
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let source_linear_formats = requirement_matrix_formats(requirements.execution());
    let target_linear_formats =
        selected_matrix_formats(requirements.execution(), selected.execution());
    let source_formats = requirement_formats(requirements.execution());
    let target_formats = selected_formats(selected.execution());
    let has_transform = selected_uses_transform(selected.execution());
    visitor.construction_started();
    let construction = CompositeTextConstruction::<B, S, V> {
        requirements,
        selected,
        store,
        context,
        visitor,
        source_linear_formats,
        target_linear_formats,
        source_formats,
        target_formats,
        has_transform,
        state: std::marker::PhantomData,
    };
    match config {
        CompositeConfig::NemotronLabsDiffusion(args) => construction.nemotron_labs_diffusion(args),
        CompositeConfig::Diffusion(args) => construction.diffusion(args),
        CompositeConfig::QwenVl(args) => construction.qwen_vl(args),
        CompositeConfig::Gemma4(args) => construction.gemma4(args),
        CompositeConfig::Muse(args) => construction.muse(args),
        CompositeConfig::Inkling(args) => construction.inkling(args),
        CompositeConfig::QwenHybrid(args) => construction.qwen_hybrid(args),
    }
}

// Keep cold family construction in separate stack frames: one debug dispatch frame
// containing every family's source/target and direct/routed temporaries can exceed
// the default thread stack before construction reaches its first layer.
struct CompositeTextConstruction<'a, B: eredu_nn::NeuralBackend, S, V> {
    requirements: CompositeTextRequirements,
    selected: SelectedCompositeTextRealization,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &'a <B::Tensor as Tensor>::Context,
    visitor: V,
    source_linear_formats: HashMap<String, LinearFormat>,
    target_linear_formats: HashMap<String, LinearFormat>,
    source_formats: HashMap<String, WeightQuantization>,
    target_formats: HashMap<String, WeightQuantization>,
    has_transform: bool,
    state: std::marker::PhantomData<fn() -> S>,
}

impl<B, S, V> CompositeTextConstruction<'_, B, S, V>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    V: CompositeTextArchitectureVisitor<B, S>,
{
    #[inline(never)]
    fn visit<A>(
        self,
        architecture: A,
        source: Option<A>,
        capability: crate::capability::CapabilityEstimate,
        model_type: String,
        cache_identity: String,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        match self.selected {
            SelectedCompositeTextRealization::Direct(selected) => {
                let (execution, processor) = selected.into_parts();
                let prepared = prepare_composite_architecture_handoff::<B, S, _>(
                    architecture,
                    source,
                    self.requirements,
                    execution,
                    processor,
                    capability,
                    model_type,
                    cache_identity,
                    self.context,
                )
                .map_err(ReplicatedTextDispatchError::Architecture)?;
                self.visitor
                    .visit(prepared, self.store)
                    .map_err(ReplicatedTextDispatchError::Backend)
            }
            SelectedCompositeTextRealization::Routed {
                execution,
                processor,
            } => {
                let prepared = prepare_routed_composite_architecture_handoff::<B, S, _>(
                    architecture,
                    source,
                    self.requirements,
                    execution,
                    processor,
                    capability,
                    model_type,
                    cache_identity,
                    self.context,
                )
                .map_err(ReplicatedTextDispatchError::Architecture)?;
                self.visitor
                    .visit_routed(prepared, self.store)
                    .map_err(ReplicatedTextDispatchError::Backend)
            }
        }
    }

    #[inline(never)]
    fn qwen_vl(
        self,
        args: &crate::qwen::vl::ModelArgs,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::qwen_vl(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| qwen_vl_with_formats(args, self.source_linear_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| crate::qwen::vl::VlKernel::<B>::new(args, self.context))
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = qwen_vl_with_formats(args, self.target_linear_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.effective_model_type().to_owned();
        let cache_identity = crate::qwen::vl::prompt_cache_architecture_fingerprint(&target);
        let architecture = crate::qwen::vl::VlKernel::<B>::new(target, self.context)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            effective_model_type,
            cache_identity,
        )
    }

    #[inline(never)]
    fn nemotron_labs_diffusion(
        self,
        args: &crate::nemotron_labs_diffusion::FamilyConfig,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::nemotron_labs_diffusion(args, true)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| {
                args.with_checkpoint_formats(self.source_formats.clone())
                    .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))
                    .and_then(|config| {
                        crate::nemotron_labs_diffusion::composite::CompositeKernel::<B>::new(
                            config,
                            self.context,
                        )
                        .map_err(|error| {
                            ReplicatedTextDispatchError::Architecture(error.to_string())
                        })
                    })
            })
            .transpose()?;
        let target = args
            .with_checkpoint_formats(self.target_formats.clone())
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let identity = target.architecture_fingerprint();
        let architecture = crate::nemotron_labs_diffusion::composite::CompositeKernel::<B>::new(
            target,
            self.context,
        )
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            "nemotron_labs_diffusion_vlm".into(),
            identity,
        )
    }

    #[inline(never)]
    fn diffusion(
        self,
        args: &crate::diffusion_gemma::FamilyConfig,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
    where
        S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    {
        let capability = crate::capability::diffusion_gemma(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| {
                crate::diffusion_gemma::composite::CompositeKernel::<B>::new(
                    args.with_checkpoint_formats(self.source_formats.clone()),
                    self.context,
                )
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = args.with_checkpoint_formats(self.target_formats.clone());
        let identity = target.architecture_fingerprint();
        let architecture =
            crate::diffusion_gemma::composite::CompositeKernel::<B>::new(target, self.context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            "diffusion_gemma".into(),
            identity,
        )
    }

    #[inline(never)]
    fn gemma4(
        self,
        args: &crate::gemma4::FamilyConfig,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let output_projection_bias = self
            .requirements
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
        let source = self
            .has_transform
            .then(|| {
                crate::gemma4::with_checkpoint_formats(&exact_args, self.source_formats.clone())
            })
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                crate::gemma4::GemmaKernel::<B>::new(args, self.context)
                    .and_then(|kernel| kernel.into_model())
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target =
            crate::gemma4::with_checkpoint_formats(&exact_args, self.target_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.effective_model_type().to_owned();
        let cache_identity = target.architecture_fingerprint();
        let architecture = crate::gemma4::GemmaKernel::<B>::new(target, self.context)
            .and_then(|kernel| kernel.into_model())
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            effective_model_type,
            cache_identity,
        )
    }

    #[inline(never)]
    fn muse(
        self,
        args: &crate::muse::DecoderConfig,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::muse(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| crate::muse::with_checkpoint_formats(args, self.source_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| crate::muse::MuseKernel::<B>::new(args, self.context))
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = crate::muse::with_checkpoint_formats(args, self.target_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.identity.text_model_type().to_owned();
        let cache_identity = target.architecture_fingerprint();
        let architecture = crate::muse::MuseKernel::<B>::new(target, self.context)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            effective_model_type,
            cache_identity,
        )
    }

    #[inline(never)]
    fn inkling(
        self,
        args: &crate::inkling::ModelArgs,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::inkling(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| crate::inkling::with_checkpoint_formats(args, self.source_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                crate::inkling::InklingKernel::<B>::new(args, self.context)
                    .and_then(crate::inkling::InklingKernel::into_model)
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = crate::inkling::with_checkpoint_formats(args, self.target_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.model_type.clone();
        let cache_identity = target.architecture_fingerprint();
        let architecture = crate::inkling::InklingKernel::<B>::new(target, self.context)
            .and_then(crate::inkling::InklingKernel::into_model)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            effective_model_type,
            cache_identity,
        )
    }

    #[inline(never)]
    fn qwen_hybrid(
        self,
        args: &crate::qwen::hybrid::ParsedHybridConfig,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::qwen_hybrid(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let source = self
            .has_transform
            .then(|| qwen_hybrid_composite_with_formats(args, self.source_linear_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| crate::qwen::hybrid::ConditionalKernel::<B>::new(args, self.context))
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = qwen_hybrid_composite_with_formats(args, self.target_linear_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.text.model_type.clone();
        let cache_identity =
            crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(&target);
        let architecture =
            crate::qwen::hybrid::ConditionalKernel::<B>::new(target, self.context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        self.visit(
            architecture,
            source,
            capability,
            effective_model_type,
            cache_identity,
        )
    }
}

/// Constructs an admitted embedded-prediction composite target and pairs it before backend erasure.
pub fn visit_composite_prediction_target_text_architecture<B, S, M, V>(
    requirements: CompositeTextRequirements,
    selected: SelectedCompositeTextRealization,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: CompositePredictionTargetVisitor<B, S, M>,
{
    validate_store_handoff(
        requirements.execution(),
        store.as_ref(),
        StoreHandoffScope::Primary,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    let retained = requirements.inspection.clone();
    let config = composite_config(retained.architecture_plan())
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let source_linear_formats = requirement_matrix_formats(requirements.execution());
    let target_linear_formats =
        selected_matrix_formats(requirements.execution(), selected.execution());
    let source_formats = requirement_formats(requirements.execution());
    let target_formats = selected_formats(selected.execution());
    let has_transform = selected_uses_transform(selected.execution());
    visitor.construction_started();
    macro_rules! visit_constructed_prediction_target {
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
                    visit_composite_prediction_target_architecture(
                        prepared, extension, store, visitor,
                    )
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
                    visit_routed_composite_prediction_target_architecture(
                        prepared, extension, store, visitor,
                    )
                }
            }
        }};
    }
    match config {
        CompositeConfig::Inkling(args) => {
            let capability = crate::capability::inkling(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let source = has_transform
                .then(|| crate::inkling::with_checkpoint_formats(args, source_formats))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| {
                    crate::inkling::InklingKernel::<B>::new(args, context)
                        .and_then(crate::inkling::InklingKernel::into_model)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = crate::inkling::with_checkpoint_formats(args, target_formats)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.model_type.clone();
            let cache_identity = target.architecture_fingerprint();
            let architecture = crate::inkling::InklingKernel::<B>::new(target, context)
                .and_then(crate::inkling::InklingKernel::into_model)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed_prediction_target!(
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
                .then(|| qwen_hybrid_composite_with_formats(args, source_linear_formats))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| crate::qwen::hybrid::ConditionalKernel::<B>::new(args, context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = qwen_hybrid_composite_with_formats(args, target_linear_formats)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.text.model_type.clone();
            let cache_identity =
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(&target);
            let architecture = crate::qwen::hybrid::ConditionalKernel::<B>::new(target, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visit_constructed_prediction_target!(
                architecture,
                source,
                capability,
                effective_model_type,
                cache_identity
            )
        }
        CompositeConfig::NemotronLabsDiffusion(_)
        | CompositeConfig::Diffusion(_)
        | CompositeConfig::Gemma4(_)
        | CompositeConfig::Muse(_)
        | CompositeConfig::QwenVl(_) => Err(ReplicatedTextDispatchError::Architecture(
            "composite architecture does not admit an embedded prediction extension".into(),
        )),
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
}

enum SelectedReplicatedTextExecutionKind<R, T, C> {
    Replicated(R),
    Routed(T),
    Composite(C),
}

/// Opaque architecture-owned semantic selection carrying backend-private
/// construction policy.
pub struct SelectedReplicatedTextExecution<R, T, C> {
    kind: SelectedReplicatedTextExecutionKind<R, T, C>,
}

impl<R, T, C> std::fmt::Debug for SelectedReplicatedTextExecution<R, T, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedReplicatedTextExecution")
            .finish_non_exhaustive()
    }
}

impl<R: Clone, T: Clone, C: Clone> Clone for SelectedReplicatedTextExecution<R, T, C> {
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
            },
        }
    }
}

/// Backend adapter invoked by an opaque selected execution value.
pub trait SelectedReplicatedTextExecutionDispatcher<R, T, C>: Sized {
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
}

impl<R, T, C> SelectedReplicatedTextExecution<R, T, C> {
    /// Invokes exactly one backend adapter while keeping the semantic branch
    /// inside architecture-owned code.
    pub fn dispatch<D>(self, dispatcher: D) -> Result<D::Output, D::Error>
    where
        D: SelectedReplicatedTextExecutionDispatcher<R, T, C>,
    {
        match self.kind {
            SelectedReplicatedTextExecutionKind::Replicated(selected) => {
                dispatcher.replicated(selected)
            }
            SelectedReplicatedTextExecutionKind::Routed(selected) => dispatcher.routed(selected),
            SelectedReplicatedTextExecutionKind::Composite(selected) => {
                dispatcher.composite(selected)
            }
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
    SelectedReplicatedTextExecution<D::Replicated, D::Routed, D::Composite>,
    ReplicatedTextDispatchError<D::Error>,
>
where
    D: ReplicatedTextExecutionClassDispatcher,
{
    if topology.is_some_and(|topology| !topology.is_replicated()) {
        return Err(ReplicatedTextDispatchError::Architecture(
            "non-replicated topology requires partitioned admission".into(),
        ));
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
    }
}

/// Selects the semantic execution class and derives replicated requirements when applicable.
pub fn replicated_text_execution_class(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextExecutionClass, ReplicatedTextRequirementsError> {
    match replicated_text_requirements(inspection) {
        Ok(requirements) => Ok(ReplicatedTextExecutionClass::Replicated(requirements)),
        Err(ReplicatedTextRequirementsError::Ineligible(reason)) => {
            if reason == ReplicatedTextIneligibility::BlockRefinement {
                return diffusion_execution_requirements(inspection);
            }
            if reason == ReplicatedTextIneligibility::CompositeInput {
                match composite_text_requirements(inspection) {
                    Ok(requirements) => {
                        return Ok(ReplicatedTextExecutionClass::Composite(requirements));
                    }
                    Err(
                        error @ ReplicatedTextRequirementsError::Ineligible(
                            ReplicatedTextIneligibility::EmbeddedPrediction,
                        ),
                    ) => return Err(error),
                    Err(error) => return Err(error),
                }
            }
            if reason == ReplicatedTextIneligibility::Routed {
                match crate::routed_text_requirements(inspection) {
                    Ok(requirements) => {
                        return Ok(ReplicatedTextExecutionClass::Routed(requirements));
                    }
                    Err(crate::RoutedTextRequirementsError::Invalid(detail)) => {
                        return Err(ReplicatedTextRequirementsError::InvalidArchitecture(detail));
                    }
                    Err(crate::RoutedTextRequirementsError::Ineligible) => {}
                }
            }
            Err(ReplicatedTextRequirementsError::Ineligible(reason))
        }
        Err(error) => Err(error),
    }
}

impl From<eredu_checkpoint::Error> for ReplicatedTextRequirementsError {
    fn from(error: eredu_checkpoint::Error) -> Self {
        Self::InvalidArchitecture(error.to_string())
    }
}
