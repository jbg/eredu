//! Architecture-owned admission for replicated text execution.

pub mod artifact;
mod lfm3_vl;
use artifact::{safetensors_inspection_recipe_source, safetensors_parameters};
pub use lfm3_vl::{lfm3_vl_requirements, Lfm3VlRequirements};
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
struct Lfm3Replicated;
struct KimiLinearReplicated;
struct DeepSeekV3DenseReplicated;
pub(crate) struct NemotronHReplicated;
pub(crate) struct QwenHybridReplicated;

impl<B: NeuralBackend> FixedReplicatedFamily<B> for Lfm2Replicated {
    const CHUNKED_PREFILL: bool = crate::lfm2::CHUNKED_TEXT_PREFILL;
    type Config = Arc<crate::lfm2::Declaration>;
    type Unit = crate::lfm2::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        if config.args().has_sparse_moe_layers() {
            return Err(eredu_nn::Error::backend(
                "replicated LFM2 configuration contains routed layers",
            ));
        }
        Ok(())
    }
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        Ok(config.definition().as_ref().clone())
    }

    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        let s = &config.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(s.normalization_spec().clone(), context)?;
        crate::decoder::StaticModules::from_boundary(
            s.embedding_spec().clone(),
            s.head_spec().cloned(),
            boundary,
            context,
        )
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::lfm2::state_identity(config.args(), layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        let parameters = config
            .blocks
            .get(index)
            .cloned()
            .ok_or_else(|| eredu_nn::Error::backend("retained LFM2 scope omits target unit"))?;
        crate::lfm2::block::ReplicatedBlock::from_parameters(parameters, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config.args().layer_schedule.iter().position(|policy| {
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
impl<B: NeuralBackend> FixedReplicatedFamily<B> for Lfm3Replicated {
    const CHUNKED_PREFILL: bool = crate::lfm3::CHUNKED_TEXT_PREFILL;
    type Config = Arc<crate::lfm3::Declaration>;
    type Unit = crate::lfm3::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        config
            .args()
            .validate()
            .map_err(eredu_nn::Error::backend_source)
    }
    fn definition(
        config: &Self::Config,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        Ok(config.definition().as_ref().clone())
    }

    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        let s = &config.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(s.normalization_spec().clone(), context)?;
        crate::decoder::StaticModules::from_boundary(
            s.embedding_spec().clone(),
            s.head_spec().cloned(),
            boundary,
            context,
        )
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::lfm3::state_identity(config.args(), layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        let parameters = config
            .blocks
            .get(index)
            .cloned()
            .ok_or_else(|| eredu_nn::Error::backend("retained LFM3 scope omits target unit"))?;
        crate::lfm3::block::ReplicatedBlock::from_parameters(parameters, context)
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config.args().layer_schedule.iter().position(|policy| {
            matches!(
                policy.operator,
                crate::lfm3::OperatorPolicy::SelfAttention(_)
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
    type Config = Arc<crate::kimi_linear::Declaration>;
    type Unit = crate::kimi_linear::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        let config = config.args();
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
        Ok(config.definition().as_ref().clone())
    }
    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        let s = &config.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(s.normalization_spec().clone(), context)?;
        crate::decoder::StaticModules::from_boundary(
            s.embedding_spec().clone(),
            s.head_spec().cloned(),
            boundary,
            context,
        )
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::kimi_linear::state_identity(config.args(), layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::kimi_linear::block::ReplicatedBlock::from_parameters(
            config
                .blocks
                .get(index)
                .cloned()
                .ok_or_else(|| eredu_nn::Error::backend("Kimi scope omits unit"))?,
            context,
        )
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config
            .args()
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
    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        crate::decoder::StaticModules::from_construction(
            config.parameters().statics.clone(),
            context,
        )
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
    type Config = Arc<crate::kimi_linear::Declaration>;
    type Unit = crate::kimi_linear::block::KdaReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        let config = config.args();
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
        Ok(config.definition().as_ref().clone())
    }
    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        let s = &config.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(s.normalization_spec().clone(), context)?;
        crate::decoder::StaticModules::from_boundary(
            s.embedding_spec().clone(),
            s.head_spec().cloned(),
            boundary,
            context,
        )
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::kimi_linear::state_identity(config.args(), layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        crate::kimi_linear::block::KdaReplicatedBlock::from_parameters(
            config
                .blocks
                .get(index)
                .cloned()
                .ok_or_else(|| eredu_nn::Error::backend("Kimi scope omits unit"))?,
            context,
        )
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
    type Config = Arc<crate::nemotron_h::Declaration>;
    type Unit = crate::nemotron_h::block::ReplicatedBlock<B>;

    fn validate(config: &Self::Config) -> Result<(), eredu_nn::Error> {
        let config = config.args();
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
        Ok(config.definition().as_ref().clone())
    }

    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        let scopes = &config.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(scopes.normalization_spec().clone(), context)?;
        crate::decoder::StaticModules::from_boundary(
            scopes.embedding_spec().clone(),
            scopes.head_spec().cloned(),
            boundary,
            context,
        )
    }

    fn state_identity(
        config: &Self::Config,
        layout: &eredu_runtime::StateLayout,
        global_layer_start: usize,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
        crate::nemotron_h::state_identity(config.args(), layout, global_layer_start, topology)
    }
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, eredu_nn::Error> {
        let parameters =
            config.blocks.get(index).cloned().ok_or_else(|| {
                eredu_nn::Error::backend("retained Nemotron scope omits target unit")
            })?;
        crate::nemotron_h::block::ReplicatedBlock::from_parameters(
            config.args(),
            index,
            parameters,
            context,
        )
    }
    fn mask_layer(config: &Self::Config) -> Option<usize> {
        config
            .args()
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
    fn construct_statics(
        config: &Self::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::decoder::StaticModules<B>, eredu_nn::Error> {
        crate::decoder::StaticModules::from_construction(config.statics().clone(), context)
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
        crate::qwen::hybrid::block::ReplicatedBlock::from_parameters(
            crate::qwen::hybrid::BlockParameters::target(config, index)?,
            context,
        )
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
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
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
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    visitor.construction_started();
    match eligible {
        EligibleConfig::Gemma2(args, declaration) => {
            let capability_estimate = crate::capability::gemma2(args, &declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = "gemma2".to_owned();
            let selected_args = selected_gemma2_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let decoder_scopes = DecoderConstructionScopes {
                selected: Arc::new(
                    crate::gemma2::project_declaration_formats(&declaration, args, &selected_args)
                        .map_err(|error| {
                            ReplicatedTextDispatchError::Architecture(error.to_string())
                        })?,
                ),
                source: declaration,
            };
            let prompt_cache_architecture_identity =
                crate::gemma2::prompt_cache_architecture_fingerprint(&selected_args);
            visit_prepared_decoder::<B, S, _, crate::decoder::DenseBlockFactory, _>(
                args,
                selected_args,
                decoder_scopes,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
                visitor,
                store,
            )
        }
        EligibleConfig::Llama(args, declaration) => {
            let capability_estimate = crate::capability::llama(args, &declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let selected_args = selected_llama_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let decoder_scopes = DecoderConstructionScopes {
                selected: Arc::new(
                    crate::llama::project_declaration_formats(&declaration, args, &selected_args)
                        .map_err(|error| {
                        ReplicatedTextDispatchError::Architecture(error.to_string())
                    })?,
                ),
                source: declaration,
            };
            let prompt_cache_architecture_identity =
                crate::llama::prompt_cache_architecture_fingerprint(&selected_args);
            visit_prepared_decoder::<B, S, _, crate::decoder::DenseBlockFactory, _>(
                args,
                selected_args,
                decoder_scopes,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
                visitor,
                store,
            )
        }
        EligibleConfig::Nanbeige(args, declaration) => {
            let capability_estimate = crate::capability::nanbeige(args, &declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = "nanbeige".to_owned();
            let selected_args = selected_nanbeige_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let decoder_scopes = DecoderConstructionScopes {
                selected: Arc::new(
                    crate::nanbeige::project_declaration_formats(
                        &declaration,
                        args,
                        &selected_args,
                    )
                    .map_err(|error| {
                        ReplicatedTextDispatchError::Architecture(error.to_string())
                    })?,
                ),
                source: declaration,
            };
            let prompt_cache_architecture_identity =
                crate::nanbeige::prompt_cache_architecture_fingerprint(&selected_args);
            visit_prepared_decoder::<B, S, _, crate::decoder::DenseBlockFactory, _>(
                args,
                selected_args,
                decoder_scopes,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
                visitor,
                store,
            )
        }
        EligibleConfig::K2Horizon(args, declaration) => {
            let capability_estimate = crate::capability::k2_horizon(&declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let selected_args = selected_k2_horizon_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let target = declaration
                .with_formats(selected_args.clone())
                .map_err(|e| ReplicatedTextDispatchError::Architecture(e.to_string()))?;
            let decoder_scopes = DecoderConstructionScopes {
                source: declaration.common().clone(),
                selected: target.common().clone(),
            };
            let prompt_cache_architecture_identity =
                crate::k2_horizon::prompt_cache_architecture_fingerprint(&selected_args);
            visit_prepared_decoder::<B, S, _, crate::k2_horizon::DenseBlockFactory, _>(
                args,
                selected_args,
                decoder_scopes,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
                visitor,
                store,
            )
        }
        EligibleConfig::Qwen(args, declaration) => {
            let capability_estimate = crate::capability::qwen(&declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let selected_args = selected_qwen_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let selected_declaration = declaration
                .with_formats(selected_args.clone())
                .map_err(|e| ReplicatedTextDispatchError::Architecture(e.to_string()))?;
            let decoder_scopes = DecoderConstructionScopes {
                source: declaration.common().clone(),
                selected: selected_declaration.common().clone(),
            };
            let prompt_cache_architecture_identity =
                crate::qwen::prompt_cache_architecture_fingerprint(&selected_args);
            visit_prepared_decoder::<B, S, _, crate::qwen::QwenBlockFactory, _>(
                args,
                selected_args,
                decoder_scopes,
                requirements,
                selected,
                capability_estimate,
                effective_model_type,
                prompt_cache_architecture_identity,
                context,
                visitor,
                store,
            )
        }
        EligibleConfig::Lfm2(_)
        | EligibleConfig::Lfm3(_)
        | EligibleConfig::KimiLinear(_)
        | EligibleConfig::NemotronH(_)
        | EligibleConfig::QwenHybrid(_)
        | EligibleConfig::GptOss(_, _)
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

pub(crate) struct DecoderConstructionScopes {
    pub(crate) source: Arc<crate::decoder::DecoderDeclaration>,
    pub(crate) selected: Arc<crate::decoder::DecoderDeclaration>,
}

fn visit_prepared_decoder<B, S, C, P, V>(
    source_args: &C,
    args: C,
    decoder_scopes: DecoderConstructionScopes,
    requirements: ReplicatedTextRequirements,
    selected: SelectedReplicatedTextRealization,
    capability: crate::capability::CapabilityEstimate,
    model_type: String,
    identity: String,
    context: &<B::Tensor as Tensor>::Context,
    visitor: V,
    store: eredu_checkpoint::store::SharedCheckpointSource,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    C: crate::decoder::Config + Clone,
    P: crate::decoder::BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    let (architecture, source) = materialize_decoder::<B, C, P>(
        source_args,
        args,
        &decoder_scopes,
        None,
        &selected,
        context,
    )
    .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_architecture_handoff::<B, S, _>(
        architecture,
        source,
        requirements,
        selected,
        capability,
        model_type,
        identity,
        context,
    )
    .map_err(ReplicatedTextDispatchError::Architecture)?;
    visitor
        .visit(prepared, store)
        .map_err(ReplicatedTextDispatchError::Backend)
}

/// Materializes the selected resource pair from retained declarations through one typed boundary.
pub(crate) fn materialize_decoder<B, C, P>(
    source_args: &C,
    args: C,
    decoder_scopes: &DecoderConstructionScopes,
    projections: Option<(
        &crate::decoder::ProjectionDeclarations<P::Parameters>,
        &crate::decoder::ProjectionDeclarations<P::Parameters>,
    )>,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (
        crate::decoder::LayeredModel<B, C, P>,
        Option<crate::decoder::LayeredModel<B, C, P>>,
    ),
    eredu_nn::Error,
>
where
    B: NeuralBackend,
    C: crate::decoder::Config + Clone,
    P: crate::decoder::BlockFactory<B, C>,
{
    let retained = selected
        .requirements()
        .shared_component_definition()
        .ok_or_else(|| eredu_nn::Error::backend("decoder selection has no retained definition"))?;
    let source = selected
        .uses_weight_transform()
        .then(|| {
            crate::decoder::CompleteDecoderKernel::<B, C, P>::from_definition(
                source_args.clone(),
                retained,
                decoder_scopes.source.as_ref(),
                projections.map(|scopes| scopes.0),
                context,
            )
        })
        .transpose()?;
    let definition = executable_component(selected).map_err(eredu_nn::Error::backend)?;
    let architecture = crate::decoder::CompleteDecoderKernel::<B, C, P>::from_definition(
        args,
        definition,
        decoder_scopes.selected.as_ref(),
        projections.map(|scopes| scopes.1),
        context,
    )?;
    Ok((architecture, source))
}

/// Retains executable component projections for cold rank reporting and construction.
pub(crate) fn retain_executable_component(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedReplicatedTextRealization,
) -> Result<SelectedReplicatedTextRealization, String> {
    let plan = inspection.architecture_plan();
    let retained = selected
        .requirements()
        .shared_component_definition()
        .ok_or("selected execution has no retained source definition")?;
    let definition = match composite_config(plan).map_err(|error| error.to_string())? {
        Some(CompositeConfig::Inkling(config)) => {
            let target =
                crate::inkling::with_checkpoint_formats(config, selected.quantized_formats())?;
            crate::inkling::model::project_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::QwenHybrid(config)) => {
            let target = qwen_hybrid_composite_with_formats(
                config,
                selected.matrix_formats_for(selected.requirements()),
            )?;
            crate::qwen::hybrid::project_conditional_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::QwenVl(config)) => {
            let target = qwen_vl_with_formats(
                &config,
                selected.matrix_formats_for(selected.requirements()),
            )?;
            crate::qwen::vl::project_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::Gemma4(config)) => {
            let exact = config.with_audio_output_projection_bias(
                crate::gemma4::checkpoint::audio_output_projection_bias(inspection),
            );
            let target =
                crate::gemma4::with_checkpoint_formats(&exact, selected.quantized_formats())?;
            crate::gemma4::model::project_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::Diffusion(config)) => {
            let target = config
                .with_checkpoint_formats(selected.quantized_formats())
                .map_err(|e| e.to_string())?;
            crate::diffusion_gemma::composite::project_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::Muse(config)) => {
            let target =
                crate::muse::with_checkpoint_formats(config, selected.quantized_formats())?;
            crate::muse::model::project_parameters(&target, &retained)
                .map_err(|error| error.to_string())?
        }
        Some(CompositeConfig::NemotronLabsDiffusion(config)) => {
            let target = config
                .with_checkpoint_formats(selected.quantized_formats())
                .map_err(|error| error.to_string())?;
            crate::nemotron_labs_diffusion::definition::project_composite_parameters(
                &target, &retained,
            )
            .map_err(|error| error.to_string())?
        }
        None => match (
            plan.safetensors_architecture().map(|p| p.model()),
            plan.gguf_plan().map(|p| p.model()),
        ) {
            (Some(SafetensorsModelConfig::QwenHybrid(config)), None)
            | (None, Some(GgufModelConfig::QwenHybrid(config)))
                if config.vision.is_none() =>
            {
                let target = selected_qwen_hybrid_args(&config.text, &selected)?;
                crate::qwen::hybrid::project_parameters(&target, &retained)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::DiffusionGemma(config)), None)
            | (None, Some(GgufModelConfig::DiffusionGemma(config))) => {
                let target = config
                    .with_checkpoint_formats(selected.quantized_formats())
                    .map_err(|e| e.to_string())?;
                crate::diffusion_gemma::model::project_parameters(&target, &retained)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::DeepSeekV4(args)), None)
            | (None, Some(GgufModelConfig::DeepSeekV4(args))) => {
                let target = selected_deepseek_v4_args(args, &selected)?;
                crate::deepseek::v4::project_parameters(&target, &retained)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::DeepSeekV3(args)), None)
            | (None, Some(GgufModelConfig::DeepSeekV3(args))) => {
                let target = selected_deepseek_v3_args(args, &selected)?;
                crate::deepseek::v3::project_parameters(&target, &retained)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Lfm2(args)), None)
            | (None, Some(GgufModelConfig::Lfm2(args))) => {
                let target = selected_lfm2_args(args, &selected)?;
                let declaration = plan.lfm2_declaration(&target)?;
                retained
                    .project_parameters(declaration.definition().parameters().clone())
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Lfm3(args)), None)
            | (None, Some(GgufModelConfig::Lfm3(args))) => {
                let target = selected_lfm3_args(args, &selected)?;
                let declaration = plan.lfm3_declaration(&target)?;
                retained
                    .project_parameters(declaration.definition().parameters().clone())
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::KimiLinear(args)), None)
            | (None, Some(GgufModelConfig::KimiLinear(args))) => {
                let target = selected_kimi_linear_args(args, &selected)?;
                let declaration = plan.kimi_declaration(&target)?;
                retained
                    .project_parameters(declaration.definition().parameters().clone())
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::NemotronH(args)), None)
            | (None, Some(GgufModelConfig::NemotronH(args))) => {
                let target = selected_nemotron_h_args(args, &selected)?;
                let declaration = plan.nemotron_declaration(&target)?;
                retained
                    .project_parameters(declaration.definition().parameters().clone())
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Llama(args)), None)
            | (None, Some(GgufModelConfig::Llama(args))) => {
                let target = selected_llama_args(args, &selected)?;
                let source = plan
                    .dense_declaration()
                    .ok_or_else(|| "missing admitted dense scopes".to_string())?;
                let declaration = crate::llama::project_declaration_formats(source, args, &target)
                    .map_err(|error| error.to_string())?;
                let parameters = declaration
                    .parameter_description(&target, crate::decoder::ParameterEncoding::Stored)
                    .map_err(|error| error.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Gemma2(args)), None)
            | (None, Some(GgufModelConfig::Gemma2(args))) => {
                let target = selected_gemma2_args(args, &selected)?;
                let source = plan
                    .dense_declaration()
                    .ok_or_else(|| "missing admitted dense scopes".to_string())?;
                let declaration = crate::gemma2::project_declaration_formats(source, args, &target)
                    .map_err(|error| error.to_string())?;
                let parameters = declaration
                    .parameter_description(&target, crate::decoder::ParameterEncoding::Stored)
                    .map_err(|error| error.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Nanbeige(args)), None)
            | (None, Some(GgufModelConfig::Nanbeige(args))) => {
                let target = selected_nanbeige_args(args, &selected)?;
                let source = plan
                    .dense_declaration()
                    .ok_or_else(|| "missing admitted dense scopes".to_string())?;
                let declaration =
                    crate::nanbeige::project_declaration_formats(source, args, &target)
                        .map_err(|error| error.to_string())?;
                let parameters = declaration
                    .parameter_description(&target, crate::decoder::ParameterEncoding::Stored)
                    .map_err(|error| error.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::Qwen(args)), None)
            | (None, Some(GgufModelConfig::Qwen(args))) => {
                let target = selected_qwen_args(args, &selected)?;
                let declaration = plan.qwen_declaration(&target)?;
                let parameters = crate::qwen::parameter_description(
                    &declaration,
                    crate::decoder::ParameterEncoding::Stored,
                )
                .map_err(|e| e.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::GptOss(args)), None)
            | (None, Some(GgufModelConfig::GptOss(args))) => {
                let target = selected_gpt_oss_args(args, &selected)?;
                let parameters = crate::gpt_oss::parameter_description(
                    plan.gpt_oss_declaration(&target)?.as_ref(),
                    crate::decoder::ParameterEncoding::Stored,
                )
                .map_err(|error| error.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::K2Horizon(args)), None)
            | (None, Some(GgufModelConfig::K2Horizon(args))) => {
                let target = selected_k2_horizon_args(args, &selected)?;
                let declaration = plan.k2_declaration(&target)?;
                let parameters = crate::k2_horizon::parameter_description(&declaration)
                    .map_err(|e| e.to_string())?;
                retained
                    .project_parameters_with_unit_paths(parameters)
                    .map_err(|error| error.to_string())?
            }
            (Some(SafetensorsModelConfig::NemotronLabsDiffusion(config)), None) => {
                let target = config
                    .with_checkpoint_formats(selected.quantized_formats())
                    .map_err(|error| error.to_string())?;
                crate::nemotron_labs_diffusion::definition::project_text_parameters(
                    &target, &retained,
                )
                .map_err(|error| error.to_string())?
            }
            (None, Some(GgufModelConfig::NemotronLabsDiffusion(plan))) => {
                let target = plan
                    .text_config()
                    .with_checkpoint_formats(selected.quantized_formats())
                    .map_err(|error| error.to_string())?;
                crate::nemotron_labs_diffusion::definition::project_text_parameters(
                    &target, &retained,
                )
                .map_err(|error| error.to_string())?
            }
            _ => {
                return Err("selected architecture has no executable component projection".into());
            }
        },
    };
    selected
        .with_executable_component(definition)
        .map_err(|error| error.to_string())
}

pub(crate) fn executable_component(
    selected: &SelectedReplicatedTextRealization,
) -> Result<Arc<eredu_runtime::ComponentDefinition>, String> {
    selected
        .executable_component()
        .cloned()
        .ok_or_else(|| "selected execution has no retained executable definition".into())
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

pub(crate) fn selected_gemma2_args(
    args: &crate::gemma2::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::gemma2::ModelArgs, String> {
    let formats = selected.quantized_formats();
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
    let formats = selected.quantized_formats();
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
    let formats = selected.quantized_formats();
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
    let formats = selected.requirements().native_quantized_formats();
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
    let formats = selected.requirements().native_quantized_formats();
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
    let formats = selected.requirements().native_quantized_formats();
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
    let formats = selected.quantized_formats();
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
    let formats = selected.requirements().native_quantized_formats();
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
    let formats = selected.quantized_formats();
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
    let formats = selected.quantized_formats();
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::lfm2::with_checkpoint_formats(args, formats)
    }
}
pub(crate) fn selected_lfm3_args(
    args: &crate::lfm3::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::lfm3::ModelArgs, String> {
    let formats = selected.quantized_formats();
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::lfm3::with_checkpoint_formats(args, formats)
    }
}

pub(crate) fn selected_kimi_linear_args(
    args: &crate::kimi_linear::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::kimi_linear::ModelArgs, String> {
    let formats = selected.quantized_formats();
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
    let formats = selected.quantized_formats();
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
    let formats = selected.quantized_formats();
    let mut target = args.policy().clone();
    if !formats.is_empty() {
        target.linear_formats = formats
            .into_iter()
            .map(|(name, format)| (name, format.into()))
            .collect();
    }
    target.validate().map_err(|error| error.to_string())?;
    args.with_formats(target).map_err(|e| e.to_string())
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
    let formats = selected.quantized_formats();
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

    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedKernel::from_definition(config, definition, context)?.into_model()
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedKernel::from_definition(config, definition, context)?.into_model()
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedKernel::from_definition(config, definition, context)?.into_model()
    }
}

impl<B, S, F> FixedProfile<B, S, F> for Stateless
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    type Model = FixedReplicatedModel<B, F, Stateless>;
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        FixedReplicatedKernel::from_definition(config, definition, context)?.into_model()
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
    P: FixedProfile<B, S, Lfm3Replicated>
        + FixedProfile<B, S, Lfm2Replicated>
        + FixedProfile<B, S, KimiLinearReplicated>
        + FixedProfile<B, S, NemotronHReplicated>
        + FixedProfile<B, S, QwenHybridReplicated>,
{
    let requirements = selected.requirements().clone();
    let retained = requirements.shared_component_definition().ok_or_else(|| {
        ReplicatedTextDispatchError::Architecture(
            "replicated selection has no retained definition".into(),
        )
    })?;
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    match eligible {
        EligibleConfig::Lfm2(declaration) => {
            let args = declaration.args();
            visitor.construction_started();
            let capability_estimate = crate::capability::lfm2(&declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as FixedProfile<B, S, Lfm2Replicated>>::from_definition(
                        Arc::clone(&declaration),
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_lfm2_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::lfm2::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as FixedProfile<B, S, Lfm2Replicated>>::from_definition(
                plan.lfm2_declaration(&args)
                    .map_err(ReplicatedTextDispatchError::Architecture)?,
                definition,
                context,
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
        EligibleConfig::Lfm3(declaration) => {
            let args = declaration.args();
            visitor.construction_started();
            let capability_estimate = crate::capability::lfm3(&declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as FixedProfile<B, S, Lfm3Replicated>>::from_definition(
                        Arc::clone(&declaration),
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_lfm3_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::lfm3::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as FixedProfile<B, S, Lfm3Replicated>>::from_definition(
                plan.lfm3_declaration(&args)
                    .map_err(ReplicatedTextDispatchError::Architecture)?,
                definition,
                context,
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
        EligibleConfig::NemotronH(declaration) => {
            let args = declaration.args();
            visitor.construction_started();
            let capability_estimate = crate::capability::nemotron_h(&declaration)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as FixedProfile<B, S, NemotronHReplicated>>::from_definition(
                        plan.nemotron_declaration(args)
                            .map_err(eredu_nn::Error::backend)?,
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_nemotron_h_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::nemotron_h::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as FixedProfile<B, S, NemotronHReplicated>>::from_definition(
                plan.nemotron_declaration(&args)
                    .map_err(ReplicatedTextDispatchError::Architecture)?,
                definition,
                context,
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
        EligibleConfig::QwenHybrid(args) => {
            visitor.construction_started();
            let capability_estimate = crate::capability::qwen_hybrid_text(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as FixedProfile<B, S, QwenHybridReplicated>>::from_definition(
                        args.clone(),
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_hybrid_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as FixedProfile<B, S, QwenHybridReplicated>>::from_definition(
                args, definition, context,
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
        EligibleConfig::KimiLinear(declaration) => {
            let args = declaration.args();
            visitor.construction_started();
            let capability_estimate = crate::capability::kimi_linear(
                plan.kimi_declaration(args)
                    .map_err(ReplicatedTextDispatchError::Architecture)?
                    .as_ref(),
            )
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as FixedProfile<B, S, KimiLinearReplicated>>::from_definition(
                        plan.kimi_declaration(args)
                            .map_err(eredu_nn::Error::backend)?,
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_kimi_linear_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::kimi_linear::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as FixedProfile<B, S, KimiLinearReplicated>>::from_definition(
                plan.kimi_declaration(&args)
                    .map_err(ReplicatedTextDispatchError::Architecture)?,
                definition,
                context,
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
        EligibleConfig::Gemma2(_, _)
        | EligibleConfig::Llama(_, _)
        | EligibleConfig::Nanbeige(_, _)
        | EligibleConfig::K2Horizon(_, _)
        | EligibleConfig::Qwen(_, _)
        | EligibleConfig::GptOss(_, _)
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedKernel::from_definition(config, definition, context)?.into_model()
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
    fn from_definition(
        config: F::Config,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Model, eredu_nn::Error> {
        CompressedReplicatedKernel::from_definition(config, definition, context)?.into_model()
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
    let retained = requirements.shared_component_definition().ok_or_else(|| {
        ReplicatedTextDispatchError::Architecture(
            "replicated selection has no retained definition".into(),
        )
    })?;
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    macro_rules! visit_family {
        ($family:ty, $args:expr, $capability:expr, $select:path, $identity:path, $prepare:expr) => {{
            visitor.construction_started();
            let args = $args;
            let prepare = $prepare;
            let capability_estimate = ($capability)(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected
                .uses_weight_transform()
                .then(|| {
                    <P as CompressedProfile<B, S, $family>>::from_definition(
                        prepare(args).map_err(eredu_nn::Error::backend)?,
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args =
                $select(args, &selected).map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity = $identity(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = <P as CompressedProfile<B, S, $family>>::from_definition(
                prepare(&args).map_err(ReplicatedTextDispatchError::Architecture)?,
                definition,
                context,
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
        }};
    }
    match eligible {
        EligibleConfig::KimiLinear(declaration) => {
            let args = declaration.args();
            visit_family!(
                KimiLinearReplicated,
                args,
                |_args: &crate::kimi_linear::ModelArgs| crate::capability::kimi_linear(
                    &declaration
                ),
                selected_kimi_linear_args,
                crate::kimi_linear::prompt_cache_architecture_fingerprint,
                |args: &crate::kimi_linear::ModelArgs| plan.kimi_declaration(args)
            )
        }
        EligibleConfig::DeepSeekV3(args) => visit_family!(
            DeepSeekV3DenseReplicated,
            args,
            crate::capability::deepseek_v3,
            selected_deepseek_v3_args,
            crate::deepseek::v3_architecture_fingerprint,
            |args: &crate::deepseek::V3Args| Ok::<_, String>(args.clone())
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
        EligibleConfig::Gemma2(_, _)
            | EligibleConfig::Llama(_, _)
            | EligibleConfig::Nanbeige(_, _)
            | EligibleConfig::K2Horizon(_, _)
            | EligibleConfig::Qwen(_, _)
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
        eredu_runtime::validate_checkpoint_handoff(
            &requirements,
            store.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
        visitor.construction_started();
        let retained = requirements.shared_component_definition().ok_or_else(|| {
            ReplicatedTextDispatchError::Architecture(
                "shared-backbone selection has no retained definition".into(),
            )
        })?;
        let source = selected
            .uses_weight_transform()
            .then(|| {
                crate::nemotron_labs_diffusion::model::TextKernel::<B>::from_definition(
                    config.clone(),
                    retained.clone(),
                    context,
                )
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = config
            .with_checkpoint_formats(selected.quantized_formats())
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let identity = target.architecture_fingerprint();
        let capability = crate::capability::nemotron_labs_diffusion(&target, false)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let effective_model_type = target.model_identity().to_owned();
        let definition =
            executable_component(&selected).map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture = crate::nemotron_labs_diffusion::model::TextKernel::<B>::from_definition(
            target, definition, context,
        )
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
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    visitor.construction_started();
    let retained = requirements.shared_component_definition().ok_or_else(|| {
        ReplicatedTextDispatchError::Architecture(
            "shared iterative selection has no retained definition".into(),
        )
    })?;
    let source = selected
        .uses_weight_transform()
        .then(|| {
            crate::diffusion_gemma::model::TextKernel::<B>::from_definition(
                config.clone(),
                retained.clone(),
                context,
            )
        })
        .transpose()
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let target = config
        .with_text_formats(selected.quantized_formats())
        .map_err(|e| ReplicatedTextDispatchError::Architecture(e.to_string()))?;
    let identity = target.architecture_fingerprint();
    let capability = crate::capability::diffusion_gemma(&target)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let definition =
        executable_component(&selected).map_err(ReplicatedTextDispatchError::Architecture)?;
    let architecture = crate::diffusion_gemma::model::TextKernel::<B>::from_definition(
        target, definition, context,
    )
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
        EligibleConfig::Gemma2(_, _)
            | EligibleConfig::Llama(_, _)
            | EligibleConfig::Nanbeige(_, _)
            | EligibleConfig::K2Horizon(_, _)
            | EligibleConfig::Qwen(_, _)
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
    let retained = requirements.shared_component_definition().ok_or_else(|| {
        ReplicatedTextDispatchError::Architecture(
            "replicated selection has no retained definition".into(),
        )
    })?;
    let eligible = eligible_config(plan)?;
    validate_plan_identity(&requirements, &eligible)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    eredu_runtime::validate_checkpoint_handoff(
        &requirements,
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    let declaration = match eligible {
        EligibleConfig::QwenHybrid(args) => {
            let capability_estimate = crate::capability::qwen_hybrid_text(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected.uses_weight_transform()
                .then(|| {
                    <MixedState as FixedProfile<B, D::State, QwenHybridReplicated>>::from_definition(
                        args.clone(),
                        retained.clone(),
                        context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_hybrid_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let prompt_cache_architecture_identity =
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(&args);
            let definition = executable_component(&selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let mut architecture =
                <MixedState as FixedProfile<B, D::State, QwenHybridReplicated>>::from_definition(
                    args, definition, context,
                )
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
            return visit_replicated_prediction_target_architecture(
                prepared,
                extension,
                store,
                dispatcher.into_visitor(),
            );
        }
        EligibleConfig::NemotronH(declaration) => declaration,
        _ => return Err(ReplicatedTextIneligibility::Unrelated.into()),
    };
    let args = declaration.args();
    let capability_estimate = crate::capability::nemotron_h(&declaration)
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let effective_model_type = args.model_type.clone();
    let source_architecture = selected
        .uses_weight_transform()
        .then(|| {
            <MixedState as FixedProfile<B, D::State, NemotronHReplicated>>::from_definition(
                plan.nemotron_declaration(args)
                    .map_err(eredu_nn::Error::backend)?,
                retained.clone(),
                context,
            )
        })
        .transpose()
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
    let args = selected_nemotron_h_args(args, &selected)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let prompt_cache_architecture_identity =
        crate::nemotron_h::prompt_cache_architecture_fingerprint(&args);
    let definition =
        executable_component(&selected).map_err(ReplicatedTextDispatchError::Architecture)?;
    let mut architecture =
        <MixedState as FixedProfile<B, D::State, NemotronHReplicated>>::from_definition(
            plan.nemotron_declaration(&args)
                .map_err(ReplicatedTextDispatchError::Architecture)?,
            definition,
            context,
        )
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
    K2Horizon(
        &'a crate::k2_horizon::ModelArgs,
        Arc<crate::k2_horizon::Declaration>,
    ),
    Llama(
        &'a crate::llama::ModelArgs,
        Arc<crate::decoder::DecoderDeclaration>,
    ),
    Gemma2(
        &'a crate::gemma2::ModelArgs,
        Arc<crate::decoder::DecoderDeclaration>,
    ),
    Nanbeige(
        &'a crate::nanbeige::ModelArgs,
        Arc<crate::decoder::DecoderDeclaration>,
    ),
    Qwen(&'a crate::qwen::ModelArgs, Arc<crate::qwen::Declaration>),
    Lfm2(Arc<crate::lfm2::Declaration>),
    Lfm3(Arc<crate::lfm3::Declaration>),
    KimiLinear(Arc<crate::kimi_linear::Declaration>),
    NemotronH(Arc<crate::nemotron_h::Declaration>),
    QwenHybrid(&'a crate::qwen::hybrid::HybridConfig),
    GptOss(
        &'a crate::gpt_oss::ModelArgs,
        Arc<crate::gpt_oss::Declaration>,
    ),
    DeepSeekV3(&'a crate::deepseek::V3Args),
    DeepSeekV4(&'a crate::deepseek::V4Args),
    DiffusionGemma(&'a crate::diffusion_gemma::FamilyConfig),
    Gemma4(&'a crate::gemma4::FamilyConfig),
    Inkling(&'a crate::inkling::ModelArgs),
    Muse(&'a crate::muse::DecoderConfig),
    QwenVl(Arc<crate::qwen::vl::Declaration>),
    QwenCompositeHybrid(&'a crate::qwen::hybrid::ParsedHybridConfig),
}

impl EligibleConfig<'_> {
    fn execution_topology(
        &self,
    ) -> Result<Option<eredu_runtime::execution_topology::TextExecutionTopology>, eredu_nn::Error>
    {
        let topology = match self {
            Self::NemotronLabsDiffusion(args) => {
                args.text_declaration().execution_topology(*args)?
            }
            Self::DiffusionGemma(args) => crate::diffusion_gemma::topology::text(args)?,
            Self::Llama(args, declaration) => declaration.execution_topology(*args)?,
            Self::Nanbeige(args, declaration) => declaration.execution_topology(*args)?,
            Self::Gemma2(args, declaration) => declaration.execution_topology(*args)?,
            Self::K2Horizon(_, declaration) => declaration.execution_topology()?,
            Self::Qwen(args, declaration) => {
                if args.is_moe() {
                    crate::qwen::execution_topology(&declaration)?
                } else {
                    declaration.common().execution_topology(*args)?
                }
            }
            Self::Lfm2(declaration) => crate::lfm2::execution_topology(declaration)?,
            Self::Lfm3(declaration) => crate::lfm3::execution_topology(declaration)?,
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
            Self::Llama(_, _)
            | Self::NemotronLabsDiffusion(_)
            | Self::Nanbeige(_, _)
            | Self::Qwen(_, _)
            | Self::Gemma2(_, _)
            | Self::K2Horizon(_, _)
            | Self::GptOss(_, _) => crate::decoder::CHUNKED_TEXT_PREFILL,
            Self::Muse(_) => crate::muse::model::RETAINED_PREFILL,
            Self::Lfm2(_) => crate::lfm2::CHUNKED_TEXT_PREFILL,
            Self::Lfm3(_) => crate::lfm3::CHUNKED_TEXT_PREFILL,
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
            Self::Nanbeige(_, declaration) => standard(
                declaration
                    .static_construction()
                    .embedding_spec()
                    .dimensions,
            ),
            Self::Gemma2(_, declaration) | Self::Llama(_, declaration) => standard(
                declaration
                    .static_construction()
                    .embedding_spec()
                    .dimensions,
            ),
            Self::K2Horizon(_, declaration) => standard(
                declaration
                    .common()
                    .static_construction()
                    .embedding_spec()
                    .dimensions,
            ),
            Self::Qwen(_, declaration) => standard(
                declaration
                    .common()
                    .static_construction()
                    .embedding_spec()
                    .dimensions,
            ),
            Self::Lfm2(declaration) => standard(declaration.statics.embedding_spec().dimensions),
            Self::Lfm3(declaration) => standard(declaration.statics.embedding_spec().dimensions),
            Self::KimiLinear(declaration) => {
                standard(declaration.statics.embedding_spec().dimensions)
            }
            Self::QwenHybrid(args) => standard(args.hidden_size),
            Self::GptOss(_, declaration) => standard(
                declaration
                    .common()
                    .static_construction()
                    .embedding_spec()
                    .dimensions,
            ),
            Self::Inkling(args) => standard(args.text_config.hidden_size),
            Self::Muse(args) => standard(args.hidden_size),
            Self::NemotronH(declaration) => {
                crate::nemotron_h::TargetBoundarySchema::from_declaration(declaration)
                    .wire_schema()
                    .map_err(|error| error.to_string())
            }
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
                let state = crate::gemma4::graph::tensor_state_layout(args, topology)
                    .map_err(|error| error.to_string())?;
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
            Self::NemotronLabsDiffusion(config) => extend(
                crate::nemotron_labs_diffusion::source_recipes(config, source)?,
            )?,
            Self::Gemma2(_, _) | Self::Llama(_, _) => {}
            Self::Nanbeige(args, declaration) => extend(crate::decoder::repeated::recipes(
                &declaration
                    .parameter_description(*args, crate::decoder::ParameterEncoding::Stored)
                    .map_err(|error| error.to_string())?,
            ))?,
            Self::K2Horizon(args, declaration) => {
                extend(crate::k2_horizon::linear_companion_recipes(
                    source,
                    declaration,
                )?)?;
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
                                declaration,
                                layer,
                                bank,
                                &(0..count as usize).collect::<Vec<_>>(),
                            )?)?;
                        }
                    }
                }
            }
            Self::Qwen(args, declaration) if args.is_moe() => {
                for layer in 0..self.unit_count()? {
                    let expert = crate::qwen::expert_recipes(source, declaration, layer)?;
                    extend(
                        eredu_checkpoint::expert::gated_product_materialization_recipes(
                            source, &expert,
                        )?,
                    )?;
                }
            }
            Self::Qwen(_, _) => {}
            Self::Lfm2(declaration) => {
                for layer in 0..declaration.definition().parameters().unit_layout().len() {
                    extend(crate::lfm2::unit_recipes(source, &declaration, layer)?)?;
                }
            }
            Self::Lfm3(declaration) => {
                for layer in 0..declaration.definition().parameters().unit_layout().len() {
                    extend(crate::lfm3::unit_recipes(source, &declaration, layer)?)?;
                }
            }
            Self::KimiLinear(declaration) => {
                for layer in 0..declaration.definition().parameters().unit_layout().len() {
                    extend(crate::kimi_linear::unit_recipes(
                        source,
                        &declaration,
                        layer,
                        true,
                    )?)?;
                }
            }
            Self::NemotronH(declaration) => {
                let args = declaration.args();
                extend(crate::nemotron_h::static_recipes(source, args, None)?)?;
                let parameters = declaration.definition().parameters();
                for ordinal in 0..parameters.unit_layout().len() {
                    extend(crate::nemotron_h::unit_recipes_flat(
                        source,
                        &declaration,
                        ordinal,
                        true,
                    )?)?;
                }
            }
            Self::QwenHybrid(args) => {
                extend(crate::qwen::hybrid::static_recipes(source)?)?;
                let parameters = self.parameter_description()?;
                for ordinal in 0..parameters.unit_layout().len() {
                    extend(crate::qwen::hybrid::unit_recipes(source, args, ordinal)?)?;
                }
            }
            Self::GptOss(_, declaration) => {
                for layer in declaration.projections().owner_layers() {
                    extend(
                        crate::gpt_oss::expert_recipes(source, declaration, layer)?
                            .into_outputs()
                            .into_outputs(),
                    )?;
                }
            }
            Self::DeepSeekV3(args) => {
                let parameters = self.parameter_description()?;
                for ordinal in 0..parameters.unit_layout().len() {
                    extend(crate::deepseek::v3_unit_recipes(
                        source, args, ordinal, true,
                    )?)?;
                }
            }
            Self::DeepSeekV4(args) => {
                let parameters = self.parameter_description()?;
                for ordinal in 0..parameters.unit_layout().len() {
                    let expert = crate::deepseek::v4_expert_recipes(source, args, ordinal)?;
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
                extend(crate::inkling::mtp_safetensors_recipes(args, source)?)?;
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
                            let expert = crate::muse::expert_recipes(source, args, layer)?;
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
                let parameters = self.parameter_description()?;
                for ordinal in 0..parameters.unit_layout().len() {
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
            Self::Nanbeige(args, _) => crate::nanbeige::prompt_cache_architecture_fingerprint(args),
            Self::Gemma2(args, _) => crate::gemma2::prompt_cache_architecture_fingerprint(args),
            Self::Llama(args, _) => crate::llama::prompt_cache_architecture_fingerprint(args),
            Self::K2Horizon(args, _) => {
                crate::k2_horizon::prompt_cache_architecture_fingerprint(args)
            }
            Self::Qwen(args, _) => crate::qwen::prompt_cache_architecture_fingerprint(args),
            Self::Lfm2(declaration) => {
                crate::lfm2::prompt_cache_architecture_fingerprint(declaration.args())
            }
            Self::Lfm3(declaration) => {
                crate::lfm3::prompt_cache_architecture_fingerprint(declaration.args())
            }
            Self::KimiLinear(declaration) => {
                crate::kimi_linear::prompt_cache_architecture_fingerprint(declaration.args())
            }
            Self::NemotronH(declaration) => {
                crate::nemotron_h::prompt_cache_architecture_fingerprint(declaration.args())
            }
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint(args)
            }
            Self::GptOss(args, _) => crate::gpt_oss::prompt_cache_architecture_fingerprint(args),
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
        if matches!(self, Self::NemotronH(_)) {
            let canonical = aliases
                .iter()
                .find(|alias| alias.starts_with("model.") || alias.starts_with("lm_head."))
                .map_or(name, String::as_str);
            if canonical.contains(".experts.") {
                for suffix in ["scales", "biases"] {
                    if let Some(prefix) = canonical.strip_suffix(&format!(".{suffix}")) {
                        return format!("{prefix}_{suffix}");
                    }
                }
            }
            return canonical.to_owned();
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
            Self::GptOss(_, _) => name
                .strip_suffix("_blocks")
                .map_or_else(|| name.to_owned(), str::to_owned),
            _ => name.to_owned(),
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
            Self::K2Horizon(args, _) => crate::decoder::operator_requirements(*args),
            Self::Gemma2(args, _) => crate::decoder::operator_requirements(*args),
            Self::Llama(_, _)
            | Self::Nanbeige(_, _)
            | Self::Qwen(_, _)
            | Self::Lfm2(_)
            | Self::Lfm3(_)
            | Self::GptOss(_, _) => NeuralOperatorCapabilities::NONE,
            Self::KimiLinear(declaration)
                if declaration.blocks.iter().any(|scope| {
                    matches!(
                        scope.mixer,
                        crate::kimi_linear::declarations::MixerParameters::Kda(_)
                    )
                }) =>
            {
                crate::operator_requirements::KIMI_LINEAR
            }
            Self::NemotronH(declaration)
                if declaration.blocks.iter().any(|scope| {
                    matches!(
                        scope.operator,
                        crate::nemotron_h::declarations::TargetOperatorParameters::Mamba(_)
                    )
                }) =>
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

    fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        match self {
            Self::NemotronLabsDiffusion(args) => {
                crate::nemotron_labs_diffusion::definition::text(args, None)
            }
            Self::Llama(args, declaration) => declaration.component_definition(*args),
            Self::Nanbeige(args, declaration) => declaration.component_definition(*args),
            Self::Gemma2(args, declaration) => declaration.component_definition(*args),
            Self::K2Horizon(_, declaration) => declaration.component_definition(),
            Self::Qwen(_, declaration) => declaration.component_definition(),
            Self::GptOss(_, declaration) => declaration.component_definition(),
            Self::Lfm2(declaration) => Ok(declaration.definition().as_ref().clone()),
            Self::Lfm3(declaration) => Ok(declaration.definition().as_ref().clone()),
            Self::KimiLinear(declaration) => Ok(declaration.definition().as_ref().clone()),
            Self::NemotronH(declaration) => Ok(declaration.definition().as_ref().clone()),
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
            Self::Nanbeige(args, declaration) => {
                return declaration
                    .component_definition(*args)
                    .map(|definition| definition.parameters().unit_layout().len())
                    .map_err(|error| error.to_string());
            }
            Self::Gemma2(_, declaration) | Self::Llama(_, declaration) => {
                return Ok(declaration.state_layout().len());
            }
            Self::K2Horizon(_, declaration) => {
                return Ok(declaration.common().state_layout().len());
            }
            Self::Qwen(_, declaration) => return Ok(declaration.common().state_layout().len()),
            Self::Lfm2(declaration) => return Ok(declaration.blocks.len()),
            Self::Lfm3(declaration) => return Ok(declaration.blocks.len()),
            Self::KimiLinear(declaration) => return Ok(declaration.blocks.len()),
            Self::NemotronH(declaration) => return Ok(declaration.blocks.len()),
            Self::QwenHybrid(args) => args.num_hidden_layers,
            Self::GptOss(args, _) => args.num_hidden_layers,
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
            Self::NemotronLabsDiffusion(args) => Ok(args.text_declaration().state_layout().clone()),
            Self::Nanbeige(_, declaration) => Ok(declaration.state_layout().clone()),
            Self::Gemma2(_, declaration) | Self::Llama(_, declaration) => {
                Ok(declaration.state_layout().clone())
            }
            Self::K2Horizon(_, declaration) => Ok(declaration.common().state_layout().clone()),
            Self::Qwen(_, declaration) => Ok(declaration.common().state_layout().clone()),
            Self::Lfm2(declaration) => Ok(declaration.definition().state_layout().clone()),
            Self::Lfm3(declaration) => Ok(declaration.definition().state_layout().clone()),
            Self::KimiLinear(declaration) => Ok(declaration.definition().state_layout().clone()),
            Self::NemotronH(declaration) => Ok(declaration.definition().state_layout().clone()),
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::state_layout(args).map_err(|error| error.to_string())
            }
            Self::GptOss(_, declaration) => Ok(declaration.common().state_layout().clone()),
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
                crate::gemma4::state_layout(args).map_err(|error| error.to_string())
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
            Self::QwenVl(args) => Ok(args.state_layout().clone()),
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::state_layout(&args.text).map_err(|error| error.to_string())
            }
        }
    }

    fn linear_parameter_declarations(&self) -> Result<LinearParameterDeclarations, String> {
        let mut bank_companions = BTreeMap::new();
        let shapes = match self {
            Self::NemotronLabsDiffusion(args) => {
                let mut shapes = BTreeMap::new();
                insert_declared_linear_shapes(&mut shapes, args.text_declaration().as_ref())?;
                Ok::<_, String>(shapes)
            }
            Self::Nanbeige(_, declaration) => {
                let sources =
                    eredu_nn::DeclareLogicalParameters::logical_parameters(declaration.as_ref())
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .filter(|slot| slot.parameter.alias_of.is_none())
                        .collect::<Vec<_>>();
                let mut shapes = BTreeMap::new();
                insert_declared_linear_shapes(&mut shapes, &sources)?;
                Ok(shapes)
            }
            Self::Gemma2(_, declaration) | Self::Llama(_, declaration) => {
                let mut shapes = BTreeMap::new();
                insert_declared_linear_shapes(&mut shapes, declaration.as_ref())?;
                Ok(shapes)
            }
            Self::K2Horizon(_, declaration) => {
                let mut shapes = crate::k2_horizon::source_parameters(
                    declaration,
                    crate::k2_horizon::SourceLayout::Safetensors,
                )?
                .into_iter()
                .filter(|p| {
                    p.role == eredu_nn::LogicalParameterRole::LinearWeight && p.shape.len() == 2
                })
                .map(|p| (p.name, p.shape))
                .collect::<BTreeMap<_, _>>();
                let rank = eredu_core::ParallelRankTopology::new(
                    eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|e| e.to_string())?,
                    0,
                )
                .map_err(|e| e.to_string())?;
                for bank in crate::k2_horizon::expert_realization_plans(declaration, rank, None)
                    .map_err(|e| e.to_string())?
                    .values()
                {
                    match bank {
                        crate::RoutedGroupedPlan::Gated(plan) => {
                            insert_grouped_gated_linear_shapes(
                                &mut shapes,
                                &mut bank_companions,
                                plan,
                            )?
                        }
                        crate::RoutedGroupedPlan::Linear(plan) => {
                            for spec in plan.unit_specs().values() {
                                insert_declared_linear_shapes(&mut shapes, spec)?;
                            }
                        }
                        crate::RoutedGroupedPlan::Relu2(_) => {
                            unreachable!("family specifies SwiGLU")
                        }
                    }
                }
                Ok(shapes)
            }
            Self::Qwen(_, declaration) => {
                let mut shapes = BTreeMap::new();
                insert_declared_linear_shapes(&mut shapes, declaration.common().as_ref())?;
                for bank in declaration
                    .expert_banks()
                    .map_err(|e| e.to_string())?
                    .values()
                {
                    insert_gated_bank_linear_shapes(&mut shapes, &mut bank_companions, bank)?;
                }
                Ok(shapes)
            }
            Self::GptOss(_, declaration) => {
                let mut shapes = BTreeMap::new();
                insert_declared_linear_shapes(&mut shapes, declaration.common().as_ref())?;
                for layer in 0..self.unit_count()? {
                    let bank = declaration
                        .expert_bank(layer)
                        .map_err(|error| error.to_string())?;
                    insert_declared_linear_shapes(&mut shapes, bank)?;
                    // Published expert storage remains native MXFP4, independently
                    // of the ordinary matrix conversion selected for this family.
                    use eredu_nn::DeclareLogicalParameters;
                    for parameter in bank
                        .logical_parameters()
                        .map_err(|error| error.to_string())?
                    {
                        if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                            register_linear_conversion_policy(
                                &mut bank_companions,
                                parameter.parameter.id.to_string(),
                                LinearConversionPolicy::NativeOnly,
                            )?;
                        }
                    }
                }
                Ok(shapes)
            }
            Self::Lfm2(declaration) => {
                let args = declaration.args();
                let mut shapes = crate::lfm2::checkpoint::source_linear_shapes(&declaration)?;
                if args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::lfm2::expert_realization_plan(
                            &declaration,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("LFM2 routed parameters have no declared bank")?,
                    )?;
                }
                for name in shapes
                    .keys()
                    .filter(|name| args.preserves_linear_source(name))
                {
                    register_linear_conversion_policy(
                        &mut bank_companions,
                        name.clone(),
                        LinearConversionPolicy::NativeOnly,
                    )?;
                }
                Ok(shapes)
            }
            Self::Lfm3(declaration) => crate::lfm3::checkpoint::source_linear_shapes(&declaration),
            Self::KimiLinear(declaration) => {
                let args = declaration.args();
                let mut shapes =
                    crate::kimi_linear::checkpoint::source_linear_shapes(&declaration)?;
                if args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::kimi_linear::expert_realization_plan(
                            &declaration,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("Kimi Linear routed parameters have no declared bank")?,
                    )?;
                }
                Ok(shapes)
            }
            Self::NemotronH(declaration) => {
                let args = declaration.args();
                let mut shapes = crate::nemotron_h::checkpoint::source_linear_shapes(&declaration)?;
                if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 {
                    insert_grouped_relu2_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::nemotron_h::expert_realization_plan(
                            &declaration,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("Nemotron-H routed parameters have no declared bank")?,
                    )?;
                }
                if args.num_nextn_predict_layers > 0 {
                    for spec in declaration.expert_banks().values() {
                        register_relu2_conversion_companions(&mut bank_companions, spec)?;
                    }
                }
                for name in shapes.keys() {
                    let canonical =
                        crate::nemotron_h::checkpoint::normalized_parameter_key(name, args)?;
                    if args.preserves_linear_source(&canonical) {
                        for identity in [name, &canonical] {
                            register_linear_conversion_policy(
                                &mut bank_companions,
                                identity.clone(),
                                LinearConversionPolicy::NativeOnly,
                            )?;
                        }
                    }
                }
                Ok(shapes)
            }
            Self::QwenHybrid(args) => {
                let mut shapes = crate::qwen::hybrid::source_linear_shapes(args)?;
                if args.num_experts > 0 && args.num_experts_per_tok > 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::qwen::hybrid::expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("routed Qwen hybrid has no declared bank")?,
                    )?;
                }
                Ok(shapes)
            }
            Self::DeepSeekV3(args) => {
                let mut shapes = crate::deepseek::checkpoint::v3_source_linear_shapes(args)?;
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::deepseek::v3_expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("V3 routed parameters have no expert realization")?,
                    )?;
                }
                if args.num_nextn_predict_layers > 0 {
                    for spec in crate::deepseek::v3::expert_bank_specs(args)
                        .map_err(|error| error.to_string())?
                        .values()
                    {
                        register_gated_conversion_companions(&mut bank_companions, spec)?;
                    }
                }
                Ok(shapes)
            }
            Self::DeepSeekV4(args) => {
                let mut shapes = crate::deepseek::checkpoint::v4_source_linear_shapes(args)?;
                if args.num_nextn_predict_layers == 0 {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::deepseek::v4_expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?,
                    )?;
                }
                if args.num_nextn_predict_layers > 0 {
                    for spec in crate::deepseek::v4::expert_bank_specs(args)
                        .map_err(|error| error.to_string())?
                        .values()
                    {
                        register_gated_conversion_companions(&mut bank_companions, spec)?;
                    }
                }
                Ok(shapes)
            }
            Self::DiffusionGemma(args) => {
                let declaration = crate::diffusion_gemma::parallel::parameter_description(args)
                    .map_err(|error| error.to_string())?;
                for (_, declaration) in args.routed_layers() {
                    register_gated_conversion_companions(
                        &mut bank_companions,
                        &declaration.routed.experts,
                    )?;
                }
                Ok(declaration
                    .groups()
                    .iter()
                    .flat_map(|group| group.members())
                    .filter(|member| {
                        member.logical_role() == Some(eredu_nn::LogicalParameterRole::LinearWeight)
                    })
                    .map(|member| {
                        (
                            member.target().to_owned(),
                            member
                                .logical_shape()
                                .expect("logical role retains declared geometry")
                                .to_vec(),
                        )
                    })
                    .collect())
            }
            Self::Gemma4(args) => {
                let mut shapes = crate::gemma4::checkpoint::source_linear_shapes(args)?;
                if args.text.num_experts.unwrap_or_default() > 0 {
                    let plan = crate::gemma4::expert_realization_plan(
                        args,
                        None,
                        replicated_expert_topology()?,
                    )
                    .map_err(|error| error.to_string())?
                    .ok_or("Gemma 4 expert realization requires a sparse invocation")?;
                    insert_grouped_gated_linear_shapes(&mut shapes, &mut bank_companions, &plan)?;
                }
                Ok(shapes)
            }
            Self::Inkling(args) => {
                let mut shapes = crate::inkling::checkpoint::source_linear_shapes(args)?;
                if args.text_config.has_sparse_moe_layers() {
                    let (plan, _) = crate::inkling::replicated_expert_realization_plan(args)
                        .map_err(|error| error.to_string())?;
                    insert_grouped_gated_linear_shapes(&mut shapes, &mut bank_companions, &plan)?;
                }
                Ok(shapes)
            }
            Self::Muse(args) => {
                let mut shapes = crate::muse::checkpoint::source_linear_shapes(args)?;
                if args.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
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
                let mut shapes = crate::qwen::vl::source_linear_shapes(args)?;
                if args.text.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::qwen::vl::expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("Qwen3-VL routed parameters have no declared bank")?,
                    )?;
                }
                Ok(shapes)
            }
            Self::QwenCompositeHybrid(args) => {
                let mut shapes = crate::qwen::hybrid::conditional_source_linear_shapes(args)?;
                if args.text.is_moe() {
                    insert_grouped_gated_linear_shapes(
                        &mut shapes,
                        &mut bank_companions,
                        &crate::qwen::hybrid::conditional_expert_realization_plan(
                            args,
                            None,
                            replicated_expert_topology()?,
                        )
                        .map_err(|error| error.to_string())?
                        .ok_or("conditional routed Qwen has no declared bank")?,
                    )?;
                }
                Ok(shapes)
            }
        }?;
        let mut companions = shapes
            .keys()
            .map(|name| (name.clone(), LinearConversionPolicy::Ordinary))
            .collect::<BTreeMap<_, _>>();
        companions.extend(bank_companions);
        Ok(LinearParameterDeclarations { shapes, companions })
    }

    /// Source conversion policy references declared component owners, independently
    /// of physical checkpoint spelling and parameter geometry.
    fn optional_transform_alignment(&self, owner: &ReplicatedTextParameterOwner) -> Option<usize> {
        let (statics, groups): (&[&str], &[&str]) = match self {
            Self::Gemma4(_) => (
                &["vision", "vision_projection", "audio", "audio_projection"],
                &[
                    crate::gemma4::VISION_EXECUTION_GROUP,
                    crate::gemma4::AUDIO_EXECUTION_GROUP,
                ],
            ),
            Self::Muse(_) => (
                &["vision"],
                &[
                    crate::muse::VISION_EXECUTION_GROUP,
                    crate::muse::audio::execution::AUDIO_EXECUTION_GROUP,
                ],
            ),
            Self::QwenVl(_) => (&["vision"], &[crate::qwen::vl::VISION_EXECUTION_GROUP]),
            _ => return None,
        };
        let media = match owner {
            ReplicatedTextParameterOwner::StaticRole(role)
            | ReplicatedTextParameterOwner::StaticUnitConsumers { role, .. } => {
                statics.contains(&role.as_str())
            }
            ReplicatedTextParameterOwner::ExecutionUnit { group, .. } => {
                groups.contains(&group.as_str())
            }
            _ => false,
        };
        media.then_some(32)
    }

    /// Borrows family facts through the same validated pinned construction used
    /// by materialization. Source aliases remain checkpoint extensions.
    fn static_construction(&self) -> Result<crate::decoder::StaticModuleConstruction, String> {
        let construction = match self {
            Self::NemotronLabsDiffusion(args) => {
                return Ok(args.text_declaration().static_construction().clone());
            }
            Self::Nanbeige(_, declaration) => return Ok(declaration.static_construction().clone()),
            Self::Gemma2(_, declaration) | Self::Llama(_, declaration) => {
                return Ok(declaration.static_construction().clone());
            }
            Self::K2Horizon(_, declaration) => {
                return Ok(declaration.common().static_construction().clone());
            }
            Self::Qwen(_, declaration) => {
                return Ok(declaration.common().static_construction().clone());
            }
            Self::GptOss(_, declaration) => {
                return Ok(declaration.common().static_construction().clone());
            }
            Self::Lfm2(declaration) => return Ok(declaration.statics.clone()),
            Self::Lfm3(declaration) => return Ok(declaration.statics.clone()),
            Self::KimiLinear(declaration) => return Ok(declaration.statics.clone()),
            Self::NemotronH(declaration) => return Ok(declaration.statics.clone()),
            Self::QwenHybrid(args) => return Ok(args.statics().clone()),
            Self::DeepSeekV3(args) => args.parameters().statics.clone(),
            Self::DeepSeekV4(args) => args.parameters().statics.clone(),
            Self::DiffusionGemma(args) => {
                return Ok(args.pinned().ordinary.clone());
            }
            Self::Gemma4(args) => {
                return Ok(args.pinned().ordinary.clone());
            }
            Self::Inkling(args) => return Ok(args.pinned().ordinary.clone()),
            Self::Muse(args) => return Ok(args.pinned().clone()),
            Self::QwenVl(args) => {
                return Ok(args
                    .text_declaration()
                    .common()
                    .static_construction()
                    .clone());
            }
            Self::QwenCompositeHybrid(args) => return Ok(args.text.statics().clone()),
        };
        Ok(construction)
    }

    fn parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, String> {
        use crate::decoder::ParameterEncoding;
        match self {
            Self::NemotronLabsDiffusion(args) => {
                if args.vision.is_some() {
                    crate::nemotron_labs_diffusion::parallel::composite_parameter_description(args)
                } else {
                    crate::nemotron_labs_diffusion::parallel::parameter_description(args)
                }
            }
            Self::Nanbeige(args, declaration) => {
                declaration.parameter_description(*args, ParameterEncoding::Logical)
            }
            Self::Gemma2(args, declaration) => {
                declaration.parameter_description(*args, ParameterEncoding::Logical)
            }
            Self::Llama(args, declaration) => {
                declaration.parameter_description(*args, ParameterEncoding::Logical)
            }
            Self::K2Horizon(_, declaration) => {
                crate::k2_horizon::parameter_description(declaration)
            }
            Self::Qwen(_, declaration) => {
                crate::qwen::parameter_description(declaration, ParameterEncoding::Logical)
            }
            Self::GptOss(_, declaration) => {
                crate::gpt_oss::parameter_description(declaration, ParameterEncoding::Logical)
            }
            Self::Lfm2(declaration) => return Ok(declaration.definition().parameters().clone()),
            Self::Lfm3(declaration) => return Ok(declaration.definition().parameters().clone()),
            Self::KimiLinear(declaration) => {
                return Ok(declaration.definition().parameters().clone());
            }
            Self::NemotronH(declaration) => {
                return Ok(declaration.definition().parameters().clone());
            }
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::parameter_description(args, ParameterEncoding::Logical)
            }
            Self::DeepSeekV3(args) => crate::deepseek::parallel::v3_parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::DeepSeekV4(args) => crate::deepseek::parallel::v4_parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::DiffusionGemma(args) => {
                if args.vision.is_some() {
                    crate::diffusion_gemma::composite::parameter_description(args).map_err(
                        |error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string()),
                    )
                } else {
                    crate::diffusion_gemma::parallel::parameter_description(args)
                }
            }
            Self::Gemma4(args) => crate::gemma4::model::parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::Inkling(args) => crate::inkling::model::parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::Muse(args) => crate::muse::model::parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::QwenVl(args) => crate::qwen::vl::parameter_description(args)
                .map_err(|error| eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())),
            Self::QwenCompositeHybrid(args) => {
                crate::qwen::hybrid::conditional_parameter_description(args).map_err(|error| {
                    eredu_runtime::ParallelPlanError::InvalidGroup(error.to_string())
                })
            }
        }
        .map_err(|error| error.to_string())
    }

    fn declared_parameters(
        &self,
        linear_shapes: &BTreeMap<String, Vec<usize>>,
    ) -> Result<BTreeMap<String, DeclaredParameter>, String> {
        let construction = self.static_construction()?;
        declared_parameters(
            &self.parameter_description()?,
            linear_shapes,
            construction.embedding_spec().weight.id.as_str(),
            construction.head_spec().is_none(),
        )
    }
}

fn insert_declared_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    spec: &impl eredu_nn::DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        if declaration.role == eredu_nn::LogicalParameterRole::LinearWeight {
            shapes.insert(
                declaration.parameter.id.as_str().to_owned(),
                declaration.shape,
            );
        }
    }
    Ok(())
}

/// Storage-output conventions projected alongside the source matrix geometry.
struct LinearParameterDeclarations {
    shapes: BTreeMap<String, Vec<usize>>,
    companions: BTreeMap<String, LinearConversionPolicy>,
}

/// Conversion policy selected by the typed ordinary or packed bank scope.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LinearConversionPolicy {
    Ordinary,
    PackedBank,
    NativeOnly,
}
impl LinearConversionPolicy {
    fn for_parameter(self, name: &str) -> Option<(String, String)> {
        match self {
            Self::Ordinary => Some(ordinary_transform_companions(name)),
            Self::PackedBank => Some((format!("{name}_scales"), format!("{name}_biases"))),
            Self::NativeOnly => None,
        }
    }
}

fn ordinary_transform_companions(name: &str) -> (String, String) {
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    (format!("{prefix}.scales"), format!("{prefix}.biases"))
}

fn register_linear_conversion_policy(
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    name: String,
    convention: LinearConversionPolicy,
) -> Result<(), String> {
    if companions
        .insert(name.clone(), convention)
        .is_some_and(|old| old != convention)
    {
        return Err(format!(
            "conflicting conversion outputs for declared parameter {name:?}"
        ));
    }
    Ok(())
}

fn insert_gated_bank_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    spec: &eredu_nn::GroupedGatedProductSpec,
) -> Result<(), String> {
    insert_declared_linear_shapes(shapes, spec)?;
    register_gated_conversion_companions(companions, spec)
}

fn register_gated_conversion_companions(
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    spec: &eredu_nn::GroupedGatedProductSpec,
) -> Result<(), String> {
    match spec.layout() {
        eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
            for projection in [gate_up, down] {
                let name = projection.weight().id.to_string();
                register_linear_conversion_policy(
                    companions,
                    name,
                    LinearConversionPolicy::PackedBank,
                )?;
            }
        }
        eredu_nn::GatedProductGroupLayout::Independent(groups) => {
            for group in groups {
                for projection in [group.gate(), group.up(), group.down()] {
                    let name = projection.weight().id.to_string();
                    register_linear_conversion_policy(
                        companions,
                        name,
                        LinearConversionPolicy::Ordinary,
                    )?;
                }
            }
        }
        _ => return Err("unsupported declared gated-product bank layout".into()),
    }
    Ok(())
}

fn insert_grouped_gated_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    plan: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<(), String> {
    for spec in plan.unit_specs().values() {
        insert_gated_bank_linear_shapes(shapes, companions, spec)?;
    }
    Ok(())
}

fn insert_grouped_relu2_linear_shapes(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    plan: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
) -> Result<(), String> {
    for spec in plan.unit_specs().values() {
        insert_declared_linear_shapes(shapes, spec)?;
        register_relu2_conversion_companions(companions, spec)?;
    }
    Ok(())
}

fn register_relu2_conversion_companions(
    companions: &mut BTreeMap<String, LinearConversionPolicy>,
    spec: &eredu_nn::GroupedRelu2Spec,
) -> Result<(), String> {
    for projection in [spec.up(), spec.down()] {
        let name = projection.weight().id.to_string();
        register_linear_conversion_policy(companions, name, LinearConversionPolicy::PackedBank)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct DeclaredParameter {
    role: ReplicatedTextParameterRole,
    owner: Option<ReplicatedTextParameterOwner>,
    // Source-only companion references may lack geometry until admission.
    // Some(empty) denotes an admitted or expanded scalar.
    shape: Option<Vec<usize>>,
    native: Option<LinearFormat>,
    // Construction-owned encoding and explicit source-only matrix extensions
    // remain distinct from raw tensor geometry.
    linear_format: Option<eredu_nn::LinearFormatSpec>,
    source_matrix: bool,
    companion: Option<(eredu_nn::LinearCompanionRole, eredu_nn::ParameterId)>,
}

impl DeclaredParameter {
    fn has_linear_encoding(&self) -> bool {
        self.linear_format.is_some() || self.source_matrix
    }
}

fn declared_parameter_role(
    name: &str,
    companion: bool,
    declarations: &BTreeMap<String, DeclaredParameter>,
) -> ReplicatedTextParameterRole {
    if companion {
        ReplicatedTextParameterRole::FormatCompanion
    } else {
        declarations
            .get(name)
            .map_or(ReplicatedTextParameterRole::Other, |parameter| {
                parameter.role
            })
    }
}

fn register_declared_companion(
    declarations: &mut BTreeMap<String, DeclaredParameter>,
    name: &str,
    kind: eredu_nn::LinearCompanionRole,
    primary: eredu_nn::ParameterId,
) -> Result<(), String> {
    if name == primary.as_str() {
        return Err(format!("format companion {name:?} refers to itself"));
    }
    let target = declarations
        .get(primary.as_str())
        .ok_or_else(|| format!("format companion {name:?} has unresolved primary {primary:?}"))?;
    if target.role == ReplicatedTextParameterRole::FormatCompanion {
        return Err(format!(
            "format companion {name:?} refers to another companion"
        ));
    }
    let owner = target.owner.clone();
    let binding = (kind, primary);
    if let Some(parameter) = declarations.get_mut(name) {
        if parameter.role != ReplicatedTextParameterRole::FormatCompanion
            || parameter
                .companion
                .as_ref()
                .is_some_and(|previous| previous != &binding)
        {
            return Err(format!("conflicting declared format companion {name:?}"));
        }
        if parameter.owner.is_some() && owner.is_some() && parameter.owner != owner {
            return Err(format!(
                "format companion {name:?} changes its primary owner"
            ));
        }
        parameter.owner = owner;
        parameter.companion = Some(binding);
    } else {
        declarations.insert(
            name.to_owned(),
            DeclaredParameter {
                role: ReplicatedTextParameterRole::FormatCompanion,
                owner,
                shape: None,
                native: Some(LinearFormat::Dense),
                linear_format: None,
                source_matrix: false,
                companion: Some(binding),
            },
        );
    }
    Ok(())
}

/// Projects parameter roles without inferring operator meaning from names.
fn declared_parameters(
    description: &eredu_runtime::ArchitectureParameterDescription,
    source_matrices: &BTreeMap<String, Vec<usize>>,
    embedding: &str,
    tied_readout_source: bool,
) -> Result<BTreeMap<String, DeclaredParameter>, String> {
    use eredu_nn::LogicalParameterRole;
    let mut roles = BTreeMap::new();
    for tagged in description.groups() {
        let owner = ReplicatedTextParameterOwner::from_parameter_group_owner(tagged.owner())
            .map_err(|error| error.to_string())?;
        for member in tagged.members() {
            let role = match member.logical_role() {
                Some(LogicalParameterRole::LinearWeight) => {
                    ReplicatedTextParameterRole::LinearWeight
                }
                Some(LogicalParameterRole::LinearBias) => ReplicatedTextParameterRole::LinearBias,
                Some(LogicalParameterRole::Embedding) if member.target() == embedding => {
                    ReplicatedTextParameterRole::Embedding
                }
                Some(LogicalParameterRole::Embedding | LogicalParameterRole::PositionEmbedding) => {
                    ReplicatedTextParameterRole::LookupTable
                }
                Some(
                    LogicalParameterRole::Normalization | LogicalParameterRole::NormalizationBias,
                ) => ReplicatedTextParameterRole::Normalization,
                _ if member.linear_companion().is_some() => {
                    ReplicatedTextParameterRole::FormatCompanion
                }
                _ => ReplicatedTextParameterRole::Other,
            };
            if let Some(previous) = roles.insert(
                member.target().to_owned(),
                DeclaredParameter {
                    role,
                    owner: Some(owner.clone()),
                    native: Some(
                        member
                            .logical_linear_format()
                            .map_or(LinearFormat::Dense, |format| format.encoding()),
                    ),
                    shape: Some(
                        member
                            .logical_shape()
                            .unwrap_or(member.global_shape())
                            .to_vec(),
                    ),
                    linear_format: member.logical_linear_format().cloned(),
                    source_matrix: false,
                    companion: member
                        .linear_companion()
                        .zip(member.linear_companion_of().cloned()),
                },
            ) {
                if previous.role != role
                    || previous.has_linear_encoding() != member.logical_linear_format().is_some()
                    || previous.owner.as_ref() != Some(&owner)
                    || previous.companion
                        != member
                            .linear_companion()
                            .zip(member.linear_companion_of().cloned())
                    || previous.shape.as_deref()
                        != Some(member.logical_shape().unwrap_or(member.global_shape()))
                    || previous.native
                        != Some(
                            member
                                .logical_linear_format()
                                .map_or(LinearFormat::Dense, |format| format.encoding()),
                        )
                {
                    return Err(format!(
                        "conflicting declared decoder roles for {:?}",
                        member.target()
                    ));
                }
            }
        }
    }
    for member in description
        .groups()
        .iter()
        .flat_map(|group| group.members())
    {
        if let Some(format) = member.logical_linear_format() {
            let primary = eredu_runtime::ParameterMemberSpec::new(
                member.target(),
                member
                    .logical_shape()
                    .ok_or("encoded primary has no logical geometry")?
                    .to_vec(),
                eredu_runtime::MemberSharding::Replicated,
            );
            for companion in eredu_runtime::expand_linear_format_parameter(&primary, format)
                .map_err(|error| error.to_string())?
            {
                let Some((kind, primary)) = companion
                    .linear_companion()
                    .zip(companion.linear_companion_of().cloned())
                else {
                    continue;
                };
                register_declared_companion(&mut roles, companion.target(), kind, primary)?;
                let declaration = roles
                    .get_mut(companion.target())
                    .expect("registered companion");
                let shape = companion.global_shape();
                if declaration
                    .shape
                    .as_deref()
                    .is_some_and(|previous| previous != shape)
                {
                    return Err(format!(
                        "format companion {:?} conflicts with its encoding geometry",
                        companion.target()
                    ));
                }
                declaration.shape = Some(shape.to_vec());
            }
        }
    }
    // Source-only split/fused layouts are explicit family extensions referencing
    // construction slots; they do not establish another parameter owner.
    for (source, shape) in source_matrices {
        if let Some(parameter) = roles.get(source) {
            let table = matches!(
                parameter.role,
                ReplicatedTextParameterRole::Embedding | ReplicatedTextParameterRole::LookupTable
            );
            if (!table && parameter.role != ReplicatedTextParameterRole::LinearWeight)
                || parameter.shape.as_deref() != Some(shape.as_slice())
            {
                return Err(format!(
                    "decoder source matrix {source:?} conflicts with its construction slot"
                ));
            }
        } else {
            roles.insert(
                source.clone(),
                DeclaredParameter {
                    role: ReplicatedTextParameterRole::LinearWeight,
                    owner: None,
                    shape: Some(shape.clone()),
                    native: None,
                    linear_format: None,
                    source_matrix: true,
                    companion: None,
                },
            );
        }
    }
    if tied_readout_source {
        let embedding = roles
            .get(embedding)
            .filter(|parameter| parameter.role == ReplicatedTextParameterRole::Embedding)
            .ok_or("tied decoder readout has no declared embedding")?;
        let readout = DeclaredParameter {
            role: ReplicatedTextParameterRole::LinearWeight,
            owner: Some(ReplicatedTextParameterOwner::StaticRole("output".into())),
            shape: embedding.shape.clone(),
            // An optional redundant readout source has its own storage encoding;
            // it does not materialize another owner of the tied embedding.
            native: None,
            linear_format: None,
            source_matrix: true,
            companion: None,
        };
        if let Some(previous) = roles.get("lm_head.weight") {
            if previous.role != readout.role || previous.shape != readout.shape {
                return Err("tied decoder readout source conflicts with its embedding".into());
            }
        }
        roles.insert("lm_head.weight".into(), readout);
    }
    Ok(roles)
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
                ));
            }
        },
        (_, Some(plan)) => match plan.model() {
            GgufModelConfig::DiffusionGemma(config) => config,
            _ => {
                return Err(ReplicatedTextRequirementsError::Ineligible(
                    ReplicatedTextIneligibility::BlockRefinement,
                ));
            }
        },
        _ => {
            return Err(ReplicatedTextRequirementsError::Ineligible(
                ReplicatedTextIneligibility::BlockRefinement,
            ));
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
    replicated_text_requirements_for_config(
        inspection,
        EligibleConfig::K2Horizon(
            args,
            inspection
                .architecture_plan()
                .k2_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
    )
}

pub(crate) fn qwen_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::qwen::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(
        inspection,
        EligibleConfig::Qwen(
            args,
            inspection
                .architecture_plan()
                .qwen_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
    )
}

pub(crate) fn gpt_oss_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::gpt_oss::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    replicated_text_requirements_for_config(
        inspection,
        EligibleConfig::GptOss(
            args,
            inspection
                .architecture_plan()
                .gpt_oss_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
    )
}

pub(crate) fn nemotron_h_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::nemotron_h::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let declaration = inspection
        .architecture_plan()
        .nemotron_declaration(args)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    replicated_text_requirements_for_config(inspection, EligibleConfig::NemotronH(declaration))
}

pub(crate) fn lfm2_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::lfm2::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let declaration = inspection
        .architecture_plan()
        .lfm2_declaration(args)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    replicated_text_requirements_for_config(inspection, EligibleConfig::Lfm2(declaration))
}

pub(crate) fn kimi_linear_replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    args: &crate::kimi_linear::ModelArgs,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let declaration = inspection
        .architecture_plan()
        .kimi_declaration(args)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    replicated_text_requirements_for_config(inspection, EligibleConfig::KimiLinear(declaration))
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
        SafetensorsModelConfig::QwenHybrid(args) if args.vision.is_some() => {
            EligibleConfig::QwenCompositeHybrid(args)
        }
        SafetensorsModelConfig::QwenHybrid(args) => EligibleConfig::QwenHybrid(&args.text),
        SafetensorsModelConfig::NemotronH(_) => EligibleConfig::NemotronH(
            extension
                .nemotron_declaration()
                .expect("declared prediction scopes")
                .clone(),
        ),
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
    // Format recipes may publish intermediate source aliases. Executable
    // bindings are exactly the canonical prediction declaration's members.
    let residencies = extension.parameter_residencies();
    parameters.retain(|parameter| residencies.contains_key(parameter.name()));
    let names = parameters
        .iter()
        .map(|parameter| parameter.name().to_owned())
        .collect::<BTreeSet<_>>();
    let missing = residencies
        .keys()
        .filter(|name| !names.contains(*name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
            format!(
                "declared prediction parameters have no exact materialization requirements: {missing:?}",
            ),
        ));
    }
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
            let residency = residencies.get(parameter.name()).cloned().ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                    "prediction parameter {:?} has no declared physical module owner",
                    parameter.name(),
                ))
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
                            extension
                                .nemotron_declaration()
                                .expect("typed Nemotron extension"),
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

/// Joins a recipe output to its declared slot, preserving encoded storage as a
/// separate projection rather than inferring logical geometry from packed widths.
fn derived_parameter_shape(
    target: &str,
    declaration: &DeclaredParameter,
    output: &eredu_checkpoint::recipe::RecipeMetadata,
    native: LinearFormat,
    alias_source: Option<&ReplicatedTextParameterRequirement>,
) -> Result<Vec<usize>, ReplicatedTextRequirementsError> {
    use eredu_checkpoint::recipe::RecipeDtype;
    let shape = declaration
        .shape
        .clone()
        .or_else(|| alias_source.map(|source| source.logical_shape().to_vec()));
    let Some(shape) = shape else {
        return if declaration.companion.is_some() {
            Ok(output.shape.clone())
        } else {
            Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                format!("derived parameter {target:?} has no declared logical geometry"),
            ))
        };
    };
    // A source-only alias carries the exact admitted source encoding even when
    // it does not introduce another construction slot.
    let encoded = declaration.has_linear_encoding()
        || (declaration.shape.is_none() && alias_source.is_some());
    let expected = if declaration.companion.is_some() {
        shape.clone()
    } else if encoded && native == LinearFormat::NvFp4 && output.dtype == RecipeDtype::U8 {
        // ModelOpt bytes are an explicit source representation, distinct from
        // the construction's canonical packed-word storage projection.
        eredu_checkpoint::schema::matrix_for_nvfp4(
            target,
            Vec::<String>::new(),
            shape.clone(),
            1,
            eredu_checkpoint::schema::NvFp4Storage::ModelOptBytes,
        )
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?
        .into_iter()
        .next()
        .expect("NVFP4 source projection contains its primary")
        .shape
    } else if encoded
        && ((output.dtype == RecipeDtype::U32
            && matches!(
                native,
                LinearFormat::Affine(_) | LinearFormat::MxFp4 | LinearFormat::NvFp4
            ))
            || (output.dtype == RecipeDtype::U8
                && matches!(native, LinearFormat::GgufIQuant { .. })))
    {
        let format = declaration
            .linear_format
            .as_ref()
            .filter(|format| format.encoding() == native);
        let unscaled;
        let format = if let Some(format) = format {
            Some(format)
        } else if matches!(native, LinearFormat::GgufIQuant { .. }) {
            unscaled = eredu_nn::LinearFormatSpec::unscaled(native).map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            Some(&unscaled)
        } else {
            None
        };
        if let Some(format) = format {
            let primary = eredu_runtime::ParameterMemberSpec::new(
                target,
                shape.clone(),
                eredu_runtime::MemberSharding::Replicated,
            );
            eredu_runtime::expand_linear_format_parameter(&primary, format)
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?
                .into_iter()
                .next()
                .expect("encoding projection contains its primary")
                .global_shape()
                .to_vec()
        } else {
            // Admitted source-format extensions supply geometry through their
            // existing checkpoint contract.
            eredu_checkpoint::schema::matrix_for_linear_format(
                target,
                Vec::<String>::new(),
                shape.clone(),
                native,
                None,
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
            .into_iter()
            .next()
            .expect("source matrix projection contains its primary")
            .shape
        }
    } else if (declaration.companion.is_none()
        && (!encoded
            || native == LinearFormat::Dense
            || matches!(
                output.dtype,
                RecipeDtype::F16
                    | RecipeDtype::BF16
                    | RecipeDtype::F32
                    | RecipeDtype::F64
                    | RecipeDtype::F8E4M3
                    | RecipeDtype::F8E5M2
            )))
        || output.dtype == RecipeDtype::F4
    {
        shape.clone()
    } else {
        return Ok(shape);
    };
    if output.shape != expected {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "derived parameter {target:?} has output geometry {:?}, expected {expected:?} for declared logical geometry {shape:?}",
            output.shape
        )));
    }
    Ok(shape)
}

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
    let LinearParameterDeclarations {
        shapes: linear_shapes,
        mut companions,
    } = config
        .linear_parameter_declarations()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let mut declarations = config
        .declared_parameters(&linear_shapes)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let component_companions = declarations
        .iter()
        .filter(|(_, declaration)| declaration.role == ReplicatedTextParameterRole::FormatCompanion)
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    for name in declarations.keys() {
        companions
            .entry(name.clone())
            .or_insert(LinearConversionPolicy::Ordinary);
    }
    project_source_parameter_owners(&mut declarations, &derived_recipes)
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    for parameter in &parameters {
        if let Some((_, primary)) = parameter.linear_companion() {
            if !declarations.contains_key(primary) {
                let source = parameters
                    .iter()
                    .find(|source| source.name() == primary)
                    .ok_or_else(|| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                            "source companion {:?} has unresolved primary {primary:?}",
                            parameter.name()
                        ))
                    })?;
                declarations.insert(
                    primary.to_owned(),
                    DeclaredParameter {
                        role: source.role(),
                        owner: Some(source.owner().clone()),
                        shape: Some(source.logical_shape().to_vec()),
                        native: Some(source.native_executable()),
                        linear_format: None,
                        source_matrix: true,
                        companion: None,
                    },
                );
            }
        }
    }
    for parameter in &parameters {
        if let Some((kind, primary)) = parameter.linear_companion() {
            let primary = eredu_nn::ParameterId::new(primary).map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
            register_declared_companion(&mut declarations, parameter.name(), kind, primary)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
        }
    }
    let derived_target_role = |target: &str| declared_parameter_role(target, false, &declarations);
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
            let declaration = declarations.get(target).ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                    "source alias {target:?} has no declaration"
                ))
            })?;
            let output = recipe.infer(recipe_source).map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
            })?;
            let shape = derived_parameter_shape(
                target,
                declaration,
                &output,
                source.native_executable(),
                Some(source),
            )?;
            parameter_requirement(
                target.clone(),
                source.sources().to_vec(),
                source.physical_sources().to_vec(),
                Vec::new(),
                source.source_encoding().cloned(),
                source.physical_shape().map(<[usize]>::to_vec),
                shape,
                source.native_executable(),
                source
                    .transform_companions()
                    .map(|_| {
                        companions
                            .get(target)
                            .map(|convention| convention.for_parameter(target))
                            .ok_or_else(|| {
                                ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                                    "alias {target:?} has no declared conversion outputs"
                                ))
                            })
                    })
                    .transpose()?
                    .flatten(),
                source.role(),
                declared_parameter_owner(&declarations, target)?,
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
    let direct_alias_sources = direct_alias_targets
        .iter()
        .filter_map(|target| derived_recipes.get(target))
        .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
        .collect::<BTreeSet<_>>();
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
        let declaration = declarations.get(target).ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                "derived parameter {target:?} has no declaration"
            ))
        })?;
        let replacement = if let Some(index) = existing {
            let parameter = &parameters[index];
            let role = if parameter.linear_companion().is_some() {
                ReplicatedTextParameterRole::FormatCompanion
            } else {
                derived_target_role(target)
            };
            let owner = declared_parameter_owner(&declarations, target)?;
            let native = parameter.native_executable();
            let logical_shape =
                derived_parameter_shape(target, declaration, output, native, alias_source)?;
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
            let native = if role == ReplicatedTextParameterRole::FormatCompanion {
                LinearFormat::Dense
            } else {
                alias_source.map_or_else(
                    || {
                        declaration.native.ok_or_else(|| {
                            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                                "derived matrix {target:?} has no declared encoding"
                            ))
                        })
                    },
                    |source| Ok(source.native_executable()),
                )?
            };
            let logical_shape =
                derived_parameter_shape(target, declaration, output, native, alias_source)?;
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
                // Storage extensions establish conversion eligibility; a declared
                // weight role also covers raw convolution and recurrent kernels.
                linear_shapes
                    .contains_key(target)
                    .then(|| companions[target].for_parameter(target))
                    .flatten(),
                role,
                declared_parameter_owner(&declarations, target)?,
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
        if direct_alias_targets.contains(parameter.name())
            || recipe_targets.contains(parameter.name())
        {
            return true;
        }
        let identities = || {
            std::iter::once(parameter.name())
                .chain(parameter.sources().iter().map(String::as_str))
                .chain(parameter.aliases().iter().map(String::as_str))
        };
        // A source-only companion remains provenance of its canonical output;
        // it is not a second executable companion after a direct rename.
        if parameter.linear_companion().is_some()
            && !component_companions.contains(parameter.name())
            && identities().any(|identity| direct_alias_sources.contains(identity))
        {
            return false;
        }
        !identities().any(|identity| consumed_sources.contains(identity))
    });
    for parameter in &mut parameters {
        if let Some(alignment) = config.optional_transform_alignment(parameter.owner()) {
            if matches!(
                parameter.transform_constraint(),
                ParameterTransformConstraint::Linear { .. }
            ) {
                *parameter = parameter
                    .clone()
                    .with_optional_transform_alignment(alignment)
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                    })?;
            }
        }
    }
    for parameter in &mut parameters {
        if let Some((kind, primary)) = declarations
            .get(parameter.name())
            .and_then(|declaration| declaration.companion.as_ref())
        {
            if parameter
                .linear_companion()
                .is_some_and(|(actual_kind, actual_primary)| {
                    actual_kind != *kind || actual_primary != primary.as_str()
                })
            {
                return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                    format!(
                        "derived format companion {:?} conflicts with its declared primary",
                        parameter.name()
                    ),
                ));
            }
            *parameter = parameter
                .clone()
                .with_linear_companion(*kind, primary.as_str())
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
        SafetensorsModelConfig::Nanbeige(args) => Ok(EligibleConfig::Nanbeige(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        SafetensorsModelConfig::Llama(args) => Ok(EligibleConfig::Llama(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        SafetensorsModelConfig::Gemma2(args) => Ok(EligibleConfig::Gemma2(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        SafetensorsModelConfig::K2Horizon(args) if args.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::K2Horizon(args) => Ok(EligibleConfig::K2Horizon(
            args,
            architecture.k2.clone().expect("admitted K2 scopes"),
        )),
        SafetensorsModelConfig::Qwen(args) if !args.is_moe() => Ok(EligibleConfig::Qwen(
            args,
            architecture.qwen.clone().expect("admitted Qwen scopes"),
        )),
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
        SafetensorsModelConfig::KimiLinear(_) => Ok(EligibleConfig::KimiLinear(
            architecture
                .kimi
                .clone()
                .expect("admitted Kimi declaration"),
        )),
        SafetensorsModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }

        SafetensorsModelConfig::Lfm2(_) => Ok(EligibleConfig::Lfm2(
            architecture
                .lfm2
                .clone()
                .expect("admitted LFM2 declaration"),
        )),
        SafetensorsModelConfig::Lfm3(_) => Ok(EligibleConfig::Lfm3(
            architecture
                .lfm3
                .clone()
                .expect("admitted LFM3 declaration"),
        )),
        SafetensorsModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        SafetensorsModelConfig::NemotronH(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        SafetensorsModelConfig::NemotronH(_) => Ok(EligibleConfig::NemotronH(
            architecture
                .nemotron
                .clone()
                .expect("admitted Nemotron declaration"),
        )),
        SafetensorsModelConfig::Gemma4(_)
        | SafetensorsModelConfig::Inkling(_)
        | SafetensorsModelConfig::Muse(_)
        | SafetensorsModelConfig::Lfm3Vl(_)
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
        GgufModelConfig::Llama(args) => Ok(EligibleConfig::Llama(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        GgufModelConfig::K2Horizon(args) if args.is_moe() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::K2Horizon(args) => Ok(EligibleConfig::K2Horizon(
            args,
            architecture.k2.clone().expect("admitted K2 scopes"),
        )),
        GgufModelConfig::Gemma2(args) => Ok(EligibleConfig::Gemma2(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        GgufModelConfig::Nanbeige(args) => Ok(EligibleConfig::Nanbeige(
            args,
            architecture.dense.clone().expect("admitted dense scopes"),
        )),
        GgufModelConfig::Qwen(args)
            if matches!(
                architecture.architecture(),
                GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3
            ) && !args.is_moe() =>
        {
            Ok(EligibleConfig::Qwen(
                args,
                architecture.qwen.clone().expect("admitted Qwen scopes"),
            ))
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
        GgufModelConfig::KimiLinear(_) => Ok(EligibleConfig::KimiLinear(
            architecture
                .kimi
                .clone()
                .expect("admitted Kimi declaration"),
        )),
        GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }

        GgufModelConfig::Lfm2(_) => Ok(EligibleConfig::Lfm2(
            architecture
                .lfm2
                .clone()
                .expect("admitted LFM2 declaration"),
        )),
        GgufModelConfig::Lfm3(_) => Ok(EligibleConfig::Lfm3(
            architecture
                .lfm3
                .clone()
                .expect("admitted LFM3 declaration"),
        )),
        GgufModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        }
        GgufModelConfig::NemotronH(args) if args.has_sparse_moe_layers() => {
            Err(ReplicatedTextIneligibility::Routed)
        }
        GgufModelConfig::NemotronH(_) => Ok(EligibleConfig::NemotronH(
            architecture
                .nemotron
                .clone()
                .expect("admitted Nemotron declaration"),
        )),
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
        config @ (EligibleConfig::Gemma2(_, _)
        | EligibleConfig::Llama(_, _)
        | EligibleConfig::Nanbeige(_, _)
        | EligibleConfig::K2Horizon(_, _)
        | EligibleConfig::Qwen(_, _)) => Ok(config),
        EligibleConfig::Lfm2(_)
        | EligibleConfig::Lfm3(_)
        | EligibleConfig::KimiLinear(_)
        | EligibleConfig::NemotronH(_)
        | EligibleConfig::QwenHybrid(_)
        | EligibleConfig::GptOss(_, _)
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
    if physical_shape.is_empty() {
        if let Some((root, _)) = name.split_once(".linear_attn.") {
            for (source, targets) in crate::qwen::hybrid::qwen3_next_global_sources(config, root)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
            {
                if source == name {
                    return Ok(Some(
                        targets
                            .into_iter()
                            .map(|target| (target.as_str().to_owned(), Vec::new()))
                            .collect(),
                    ));
                }
            }
        }
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
    let construction = config
        .static_construction()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    if construction.head_spec().is_none() {
        use eredu_nn::DeclareLogicalParameters;
        let embedding = construction.embedding_spec();
        let declaration = embedding
            .logical_parameters()
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?
            .into_iter()
            .find(|declaration| declaration.parameter.id == embedding.weight.id)
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "tied readout has no declared embedding weight".into(),
                )
            })?;
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
                declaration.shape,
                LinearFormat::Dense,
                Some(ordinary_transform_companions("lm_head.weight")),
                ReplicatedTextParameterRole::LinearWeight,
                ReplicatedTextParameterOwner::StaticRole("output".into()),
                ReplicatedTextParameterPresence::Tied {
                    target: declaration.parameter.id.as_str().to_owned(),
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
    transform_companions: Option<(String, String)>,
    role: ReplicatedTextParameterRole,
    owner: ReplicatedTextParameterOwner,
    presence: ReplicatedTextParameterPresence,
) -> Result<ReplicatedTextParameterRequirement, ReplicatedTextRequirementsError> {
    let transform = if transform_companions.is_some() {
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
    if let Some((scale, bias)) = transform_companions {
        requirement
            .with_transform_companions(scale, bias)
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })
    } else {
        Ok(requirement)
    }
}

fn declared_parameter_format(
    declarations: &BTreeMap<String, DeclaredParameter>,
    name: &str,
) -> Result<LinearFormat, ReplicatedTextRequirementsError> {
    declarations
        .get(name)
        .and_then(|parameter| parameter.native)
        .ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                "matrix {name:?} has no declared encoding"
            ))
        })
}

fn declared_parameter_owner(
    declarations: &BTreeMap<String, DeclaredParameter>,
    name: &str,
) -> Result<ReplicatedTextParameterOwner, ReplicatedTextRequirementsError> {
    declarations
        .get(name)
        .and_then(|parameter| parameter.owner.clone())
        .ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                "parameter {name:?} has no declared storage owner"
            ))
        })
}

/// Source-format tensors inherit storage from their declared materialized outputs.
/// This never changes a construction slot's authoritative owner.
fn project_source_parameter_owners(
    declarations: &mut BTreeMap<String, DeclaredParameter>,
    recipes: &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
) -> Result<BTreeMap<String, ReplicatedTextParameterOwner>, String> {
    let mut sources = BTreeMap::new();
    for (target, recipe) in recipes {
        let Some(owner) = declarations
            .get(target)
            .and_then(|parameter| parameter.owner.as_ref())
        else {
            continue;
        };
        for source in recipe.source_keys() {
            if declarations
                .get(source)
                .is_some_and(|parameter| parameter.owner.is_some())
            {
                continue;
            }
            if sources
                .insert(source.to_owned(), owner.clone())
                .is_some_and(|previous| previous != *owner)
            {
                return Err(format!(
                    "source tensor {source:?} feeds conflicting declared storage owners"
                ));
            }
        }
    }
    for (source, owner) in &sources {
        if let Some(parameter) = declarations.get_mut(source) {
            parameter.owner = Some(owner.clone());
        }
    }
    Ok(sources)
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
    for parameter in &parameters {
        if parameter.role() != ReplicatedTextParameterRole::FormatCompanion {
            continue;
        }
        let (_, primary) = parameter.linear_companion().ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                "format companion {:?} has no declared primary",
                parameter.name()
            ))
        })?;
        let primary = parameters
            .binary_search_by(|candidate| candidate.name().cmp(primary))
            .ok()
            .map(|index| &parameters[index])
            .ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(format!(
                    "format companion {:?} refers to absent primary {primary:?}",
                    parameter.name()
                ))
            })?;
        if primary.role() == ReplicatedTextParameterRole::FormatCompanion {
            return Err(ReplicatedTextRequirementsError::InvalidArchitecture(
                format!(
                    "format companion {:?} refers to another companion",
                    parameter.name()
                ),
            ));
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

    fn retain_executable_component(
        self,
        inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    ) -> Result<Self, CompositeTextSelectionError> {
        let retain = |selected| {
            retain_executable_component(inspection, selected).map_err(|error| {
                CompositeTextSelectionError {
                    issues: vec![error],
                }
            })
        };
        Ok(match self {
            Self::Direct(selected) => {
                let (execution, processor) = selected.into_parts();
                Self::Direct(eredu_runtime::SelectedCompositeRealization::from_parts(
                    retain(execution)?,
                    processor,
                ))
            }
            Self::Routed {
                mut execution,
                processor,
            } => {
                execution.text = retain(execution.text)?;
                Self::Routed {
                    execution,
                    processor,
                }
            }
        })
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
        return SelectedCompositeTextRealization::Routed {
            execution: execution.expect("empty diagnostics imply routed selection"),
            processor: processor.expect("empty diagnostics imply processor selection"),
        }
        .retain_executable_component(requirements.inspection());
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
    SelectedCompositeTextRealization::Direct(
        eredu_runtime::SelectedCompositeRealization::from_parts(
            execution.expect("empty diagnostics imply direct selection"),
            processor.expect("empty diagnostics imply processor selection"),
        ),
    )
    .retain_executable_component(requirements.inspection())
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
        return SelectedCompositeTextRealization::Routed {
            execution,
            processor,
        }
        .retain_executable_component(requirements.inspection());
    }
    let execution = eredu_runtime::select_replicated_text_realization(
        requirements.execution(),
        execution_request,
        execution_capabilities,
    )
    .map_err(|error| CompositeTextSelectionError {
        issues: error.issues().to_vec(),
    })?;
    SelectedCompositeTextRealization::Direct(
        eredu_runtime::SelectedCompositeRealization::from_parts(execution, processor),
    )
    .retain_executable_component(requirements.inspection())
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

    /// Binds the retained dense vision-language graph and its cursor adapter.
    fn visit_vl(
        self,
        _prepared: crate::lfm3::vision::PreparedVlGraph,
    ) -> Result<Self::Output, crate::prepared_execution::PreparedExecutionError<Self::Error>> {
        Err(crate::prepared_execution::PreparedExecutionError::UnavailableExecution)
    }

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
    QwenVl(Arc<crate::qwen::vl::Declaration>),
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
            Self::QwenVl(args) => EligibleConfig::QwenVl(args.clone()),
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

    fn component_definition(
        &self,
        inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    ) -> Result<eredu_runtime::ComponentDefinition, eredu_nn::Error> {
        match self {
            Self::NemotronLabsDiffusion(args) => {
                crate::nemotron_labs_diffusion::definition::composite(args, None)
            }
            Self::Diffusion(args) => {
                crate::diffusion_gemma::composite::component_definition(args, None)
            }
            Self::Gemma4(args) => crate::gemma4::model::component_definition(
                &args.with_audio_output_projection_bias(
                    crate::gemma4::checkpoint::audio_output_projection_bias(inspection),
                ),
                None,
            ),
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
            GgufMediaProjectorConfig::Qwen3Vl(args) => CompositeConfig::QwenVl(
                plan.vl_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            ),
            GgufMediaProjectorConfig::Qwen35(args) => CompositeConfig::QwenHybrid(args),
            GgufMediaProjectorConfig::Lfm3Vl { .. } => return Ok(None),
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
        (Some(SafetensorsModelConfig::QwenVl(args)), None) => Some(CompositeConfig::QwenVl(
            plan.vl_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        )),
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
        | (None, Some(GgufModelConfig::K2Horizon(args))) => EligibleConfig::K2Horizon(
            args,
            plan.k2_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
        (Some(SafetensorsModelConfig::Qwen(args)), None) if args.is_moe() => EligibleConfig::Qwen(
            args,
            plan.qwen_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
        (Some(SafetensorsModelConfig::QwenHybrid(args)), None)
        | (None, Some(GgufModelConfig::QwenHybrid(args)))
            if args.vision.is_none() && args.text.mtp_num_hidden_layers == 0 =>
        {
            EligibleConfig::QwenHybrid(&args.text)
        }
        (Some(SafetensorsModelConfig::Lfm2(args)), None) if args.has_sparse_moe_layers() => {
            EligibleConfig::Lfm2(
                plan.lfm2_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }

        (None, Some(GgufModelConfig::Lfm2(args))) if args.has_sparse_moe_layers() => {
            EligibleConfig::Lfm2(
                plan.lfm2_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }
        (Some(SafetensorsModelConfig::KimiLinear(args)), None)
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::KimiLinear(
                plan.kimi_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }
        (None, Some(GgufModelConfig::KimiLinear(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::KimiLinear(
                plan.kimi_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }
        (Some(SafetensorsModelConfig::NemotronH(args)), None)
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::NemotronH(
                plan.nemotron_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }
        (None, Some(GgufModelConfig::NemotronH(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            EligibleConfig::NemotronH(
                plan.nemotron_declaration(args)
                    .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            )
        }
        (Some(SafetensorsModelConfig::GptOss(args)), None) => EligibleConfig::GptOss(
            args,
            plan.gpt_oss_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
        (Some(SafetensorsModelConfig::DeepSeekV3(args)), None) => EligibleConfig::DeepSeekV3(args),
        (Some(SafetensorsModelConfig::DeepSeekV4(args)), None) => EligibleConfig::DeepSeekV4(args),
        (None, Some(GgufModelConfig::Qwen(args))) if args.is_moe() => EligibleConfig::Qwen(
            args,
            plan.qwen_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
        (None, Some(GgufModelConfig::GptOss(args))) => EligibleConfig::GptOss(
            args,
            plan.gpt_oss_declaration(args)
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        ),
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
    if plan.model_kind() == crate::configuration::ModelKind::Lfm3Vl {
        return lfm3_vl::processor_requirements(plan).map(Some);
    }
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
            let plan = crate::qwen::vl::expert_realization_plan(
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
                    "Qwen3-VL routed parameters have no declared bank".into(),
                )
            })?;
            let catalog = crate::qwen::expert_residency_catalog(
                recipe_source.as_ref(),
                args.text_declaration(),
            )
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
            let plan = crate::qwen::hybrid::conditional_expert_realization_plan(
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
                    "conditional routed Qwen has no declared bank".into(),
                )
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
            let plan = crate::gemma4::expert_realization_plan(
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
                    "Gemma 4 expert realization requires a sparse invocation".into(),
                )
            })?;
            let catalog = crate::gemma4::expert_residency_catalog(recipe_source.as_ref(), args)
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
            let (plan, routes_by_unit) = crate::inkling::replicated_expert_realization_plan(args)
                .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
            })?;
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
        .component_definition(inspection)
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
    declaration: &crate::qwen::vl::Declaration,
    formats: HashMap<String, LinearFormat>,
) -> Result<Arc<crate::qwen::vl::Declaration>, String> {
    let args = declaration.args();
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
    declaration
        .with_formats(target)
        .map(Arc::new)
        .map_err(|e| e.to_string())
}

pub(crate) fn qwen_hybrid_composite_with_formats(
    args: &crate::qwen::hybrid::ParsedHybridConfig,
    formats: HashMap<String, LinearFormat>,
) -> Result<crate::qwen::hybrid::ParsedHybridConfig, String> {
    let (vision, text): (HashMap<_, _>, HashMap<_, _>) = formats
        .into_iter()
        .partition(|(name, _)| name.starts_with("model.visual."));
    let mut target = args.clone();
    let mut policy = target.text.policy().clone();
    policy.quantization = None;
    policy.fp8 = None;
    policy.linear_formats = text;
    target.text = target
        .text
        .with_formats(policy)
        .map_err(|e| e.to_string())?;
    target.text.validate().map_err(|error| error.to_string())?;
    let target_vision = target.vision.as_mut().ok_or_else(|| {
        "conditional Qwen projector formats have no vision configuration".to_owned()
    })?;
    let mut vision_config = target_vision.config().clone();
    vision_config.linear_formats = vision;
    *target_vision = Arc::new(
        target_vision
            .with_formats(vision_config)
            .map_err(|e| e.to_string())?,
    );
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
    eredu_runtime::validate_checkpoint_handoff(
        requirements.execution(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    let retained = requirements.inspection.clone();
    let config = composite_config(retained.architecture_plan())
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let source_linear_formats = requirements.execution().native_matrix_formats();
    let target_linear_formats = selected
        .execution()
        .matrix_formats_for(requirements.execution());
    let source_formats = requirements.execution().native_quantized_formats();
    let target_formats = selected.execution().quantized_formats();
    let has_transform = selected.execution().uses_weight_transform();
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
        CompositeConfig::QwenVl(args) => construction.qwen_vl(&args),
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
        args: &crate::qwen::vl::Declaration,
    ) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>> {
        let capability = crate::capability::qwen_vl(args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| qwen_vl_with_formats(args, self.source_linear_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                let definition = crate::qwen::vl::project_parameters(&args, &retained)?;
                crate::qwen::vl::VlKernel::<B>::from_definition(args, definition, self.context)
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = qwen_vl_with_formats(args, self.target_linear_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.effective_model_type().to_owned();
        let cache_identity = crate::qwen::vl::prompt_cache_architecture_fingerprint(&target);
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::qwen::vl::VlKernel::<B>::from_definition(target, definition, self.context)
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
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| {
                args.with_checkpoint_formats(self.source_formats.clone())
                    .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))
                    .and_then(|config| {
                        let definition = retained.clone();
                        crate::nemotron_labs_diffusion::composite::CompositeKernel::<B>::from_definition(
                            config, definition, self.context,
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
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::nemotron_labs_diffusion::composite::CompositeKernel::<B>::from_definition(
                target,
                definition,
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
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| {
                let config = args
                    .with_checkpoint_formats(self.source_formats.clone())
                    .map_err(eredu_nn::Error::backend_source)?;
                let definition =
                    crate::diffusion_gemma::composite::project_parameters(&config, &retained)?;
                crate::diffusion_gemma::composite::CompositeKernel::<B>::from_definition(
                    config,
                    definition,
                    self.context,
                )
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = args
            .with_checkpoint_formats(self.target_formats.clone())
            .map_err(|e| ReplicatedTextDispatchError::Architecture(e.to_string()))?;
        let identity = target.architecture_fingerprint();
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::diffusion_gemma::composite::CompositeKernel::<B>::from_definition(
                target,
                definition,
                self.context,
            )
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
        let exact_args = args.with_audio_output_projection_bias(output_projection_bias);
        let capability = crate::capability::gemma4(&exact_args)
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| {
                crate::gemma4::with_checkpoint_formats(&exact_args, self.source_formats.clone())
            })
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                let definition = crate::gemma4::model::project_parameters(&args, &retained)?;
                crate::gemma4::GemmaKernel::<B>::from_definition(args, definition, self.context)
                    .and_then(|kernel| kernel.into_model())
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target =
            crate::gemma4::with_checkpoint_formats(&exact_args, self.target_formats.clone())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.effective_model_type().to_owned();
        let cache_identity = target.architecture_fingerprint();
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::gemma4::GemmaKernel::<B>::from_definition(target, definition, self.context)
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
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| crate::muse::with_checkpoint_formats(args, self.source_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                crate::muse::MuseKernel::<B>::from_definition(args, retained.clone(), self.context)
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = crate::muse::with_checkpoint_formats(args, self.target_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.identity.text_model_type().to_owned();
        let cache_identity = target.architecture_fingerprint();
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::muse::MuseKernel::<B>::from_definition(target, definition, self.context)
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
        let parent_definition = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| crate::inkling::with_checkpoint_formats(args, self.source_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                let definition =
                    crate::inkling::model::project_parameters(&args, &parent_definition)?;
                crate::inkling::InklingKernel::<B>::from_definition(args, definition, self.context)
                    .and_then(crate::inkling::InklingKernel::into_model)
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = crate::inkling::with_checkpoint_formats(args, self.target_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.model_type.clone();
        let cache_identity = target.architecture_fingerprint();
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture =
            crate::inkling::InklingKernel::<B>::from_definition(target, definition, self.context)
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
        let retained = self
            .requirements
            .execution()
            .shared_component_definition()
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Architecture(
                    "composite selection has no retained definition".into(),
                )
            })?;
        let source = self
            .has_transform
            .then(|| qwen_hybrid_composite_with_formats(args, self.source_linear_formats.clone()))
            .transpose()
            .map_err(ReplicatedTextDispatchError::Architecture)?
            .map(|args| {
                let definition =
                    crate::qwen::hybrid::project_conditional_parameters(&args, &retained)?;
                crate::qwen::hybrid::ConditionalKernel::<B>::from_definition(
                    args,
                    definition,
                    self.context,
                )
            })
            .transpose()
            .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
        let target = qwen_hybrid_composite_with_formats(args, self.target_linear_formats.clone())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let effective_model_type = target.text.model_type.clone();
        let cache_identity =
            crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(&target);
        let definition = executable_component(self.selected.execution())
            .map_err(ReplicatedTextDispatchError::Architecture)?;
        let architecture = crate::qwen::hybrid::ConditionalKernel::<B>::from_definition(
            target,
            definition,
            self.context,
        )
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
    eredu_runtime::validate_checkpoint_handoff(
        requirements.execution(),
        store.as_ref(),
        eredu_runtime::SourceHandoffScope::Primary,
    )
    .map_err(|handoff| ReplicatedTextDispatchError::Architecture(handoff.to_string()))?;
    let retained = requirements.inspection.clone();
    let config = composite_config(retained.architecture_plan())
        .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?
        .ok_or(ReplicatedTextIneligibility::Unrelated)?;
    let source_linear_formats = requirements.execution().native_matrix_formats();
    let target_linear_formats = selected
        .execution()
        .matrix_formats_for(requirements.execution());
    let source_formats = requirements.execution().native_quantized_formats();
    let target_formats = selected.execution().quantized_formats();
    let has_transform = selected.execution().uses_weight_transform();
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
            let parent_definition = requirements
                .execution()
                .shared_component_definition()
                .ok_or_else(|| {
                    ReplicatedTextDispatchError::Architecture(
                        "composite selection has no retained definition".into(),
                    )
                })?;
            let source = has_transform
                .then(|| crate::inkling::with_checkpoint_formats(args, source_formats))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| {
                    let definition =
                        crate::inkling::model::project_parameters(&args, &parent_definition)?;
                    crate::inkling::InklingKernel::<B>::from_definition(args, definition, context)
                        .and_then(crate::inkling::InklingKernel::into_model)
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = crate::inkling::with_checkpoint_formats(args, target_formats)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.model_type.clone();
            let cache_identity = target.architecture_fingerprint();
            let definition = executable_component(selected.execution())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture =
                crate::inkling::InklingKernel::<B>::from_definition(target, definition, context)
                    .and_then(crate::inkling::InklingKernel::into_model)
                    .map_err(|error| {
                        ReplicatedTextDispatchError::Architecture(error.to_string())
                    })?;
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
            let retained = requirements
                .execution()
                .shared_component_definition()
                .ok_or_else(|| {
                    ReplicatedTextDispatchError::Architecture(
                        "composite selection has no retained definition".into(),
                    )
                })?;
            let source = has_transform
                .then(|| qwen_hybrid_composite_with_formats(args, source_linear_formats))
                .transpose()
                .map_err(ReplicatedTextDispatchError::Architecture)?
                .map(|args| {
                    let definition =
                        crate::qwen::hybrid::project_conditional_parameters(&args, &retained)?;
                    crate::qwen::hybrid::ConditionalKernel::<B>::from_definition(
                        args, definition, context,
                    )
                })
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let target = qwen_hybrid_composite_with_formats(args, target_linear_formats)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let effective_model_type = target.text.model_type.clone();
            let cache_identity =
                crate::qwen::hybrid::conditional_prompt_cache_architecture_fingerprint(&target);
            let definition = executable_component(selected.execution())
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = crate::qwen::hybrid::ConditionalKernel::<B>::from_definition(
                target, definition, context,
            )
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
