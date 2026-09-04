//! Prepared neutral construction for embedded prediction extensions.

use std::{collections::BTreeMap, num::NonZeroUsize};

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_core::{cache::LayerCachePolicy, ParallelRankTopology, ParallelTopology};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::RuntimeStateComponents;
use eredu_runtime::{ArchitectureParameters, LocalModelLayout, StateLayout};
use eredu_runtime::{
    SpeculativeArchitectureCompatibilityProof, SpeculativeCaptureEntry, SpeculativeCaptureSchema,
    SpeculativeIdentity, SpeculativeMechanism, SpeculativeMechanismRequirements,
    SpeculativePlacementRequest, SpeculativeRealizationRequirements, SpeculativeSelectionRequest,
    SpeculativeStateCacheIdentityIngredients, SpeculativeStrategyClass,
    SpeculativeStrategyRequirements,
};

use crate::configuration::{
    PredictionExtensionKind, PredictionExtensionPlan, SafetensorsModelConfig,
};

/// Construction-time neutral identities and bounds for one embedded prediction realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EmbeddedSpeculativeContractRequest {
    target: SpeculativeIdentity,
    artifact: SpeculativeIdentity,
    format: SpeculativeIdentity,
    topology: ParallelRankTopology,
    processor: SpeculativeIdentity,
    maximum_batch_size: NonZeroUsize,
    maximum_sequence_length: NonZeroUsize,
    maximum_draft_tokens: NonZeroUsize,
}

impl EmbeddedSpeculativeContractRequest {
    /// Creates exact construction-time inputs for architecture-owned embedded selection.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        target: SpeculativeIdentity,
        artifact: SpeculativeIdentity,
        format: SpeculativeIdentity,
        topology: ParallelRankTopology,
        processor: SpeculativeIdentity,
        maximum_batch_size: NonZeroUsize,
        maximum_sequence_length: NonZeroUsize,
        maximum_draft_tokens: NonZeroUsize,
    ) -> Self {
        Self {
            target,
            artifact,
            format,
            topology,
            processor,
            maximum_batch_size,
            maximum_sequence_length,
            maximum_draft_tokens,
        }
    }
}

/// Exact architecture-owned requirements and compatibility proof for an embedded extension.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EmbeddedSpeculativeContract {
    requirements: SpeculativeRealizationRequirements,
    architecture_proof: SpeculativeArchitectureCompatibilityProof,
    target_capture: SpeculativeCaptureSchema,
}

impl EmbeddedSpeculativeContract {
    /// Returns the complete requirements consumed by neutral speculative selection.
    pub const fn requirements(&self) -> &SpeculativeRealizationRequirements {
        &self.requirements
    }

    /// Returns the architecture proof paired with the exact target and capture schema.
    pub const fn architecture_proof(&self) -> &SpeculativeArchitectureCompatibilityProof {
        &self.architecture_proof
    }

    /// Returns the exact ordered capture schema advertised by the ordinary target.
    pub const fn target_capture(&self) -> &SpeculativeCaptureSchema {
        &self.target_capture
    }

    /// Creates the exact neutral selection request without reconstructing architecture proof.
    pub fn selection_request(
        &self,
        placement: SpeculativePlacementRequest,
    ) -> SpeculativeSelectionRequest {
        SpeculativeSelectionRequest::new(placement, self.target_capture.clone())
            .with_architecture_proof(self.architecture_proof.clone())
    }

    /// Consumes the contract into selection inputs.
    pub fn into_parts(
        self,
    ) -> (
        SpeculativeRealizationRequirements,
        SpeculativeArchitectureCompatibilityProof,
        SpeculativeCaptureSchema,
    ) {
        (
            self.requirements,
            self.architecture_proof,
            self.target_capture,
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct EmbeddedFamilyContract {
    family: &'static str,
    class: SpeculativeStrategyClass,
    architecture_capacity: NonZeroUsize,
    capture: EmbeddedCaptureContract,
    state_components: Vec<String>,
    additional_mechanisms: Vec<SpeculativeMechanism>,
    strategy_detail: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum EmbeddedCaptureContract {
    Single {
        path: String,
        shape: Vec<usize>,
        observation: String,
    },
    Dspark {
        layers: Vec<usize>,
        batch: usize,
        sequence: usize,
        hidden: usize,
    },
}

fn speculative_identity(
    value: impl Into<String>,
) -> Result<SpeculativeIdentity, eredu_core::artifact::ArtifactError> {
    SpeculativeIdentity::new(value).map_err(|error| invalid(error.to_string()))
}

fn positive(value: i32, field: &str) -> Result<usize, eredu_core::artifact::ArtifactError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("embedded prediction {field} must be positive")))
}

fn indexed_state(prefix: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|index| format!("{prefix}.{index}"))
        .collect()
}

fn embedded_family_contract(
    extension: &PredictionExtensionPlan,
    batch: usize,
    sequence: usize,
) -> Result<EmbeddedFamilyContract, eredu_core::artifact::ArtifactError> {
    validate_extension_contract(extension)?;
    let depth = NonZeroUsize::new(extension.depth())
        .ok_or_else(|| invalid("embedded prediction depth must be positive"))?;
    let model = extension.complete_architecture().model();
    let grouped = model
        .uses_grouped_routed_experts()
        .then_some(SpeculativeMechanism::GroupedNeuralOperations);
    let contract = match (extension.kind(), model) {
        (PredictionExtensionKind::DeepSeekV3Mtp, SafetensorsModelConfig::DeepSeekV3(args)) => {
            let hidden = positive(args.hidden_size, "hidden size")?;
            EmbeddedFamilyContract {
                family: "deepseek-v3-mtp",
                class: SpeculativeStrategyClass::EmbeddedSequential,
                architecture_capacity: depth,
                capture: EmbeddedCaptureContract::Single {
                    path: "target.final_hidden".into(),
                    shape: vec![batch, sequence, hidden],
                    observation: "prediction.target_capture".into(),
                },
                state_components: indexed_state("prediction.compressed_latent", depth.get()),
                additional_mechanisms: grouped.into_iter().collect(),
                strategy_detail: format!("depth={};hidden={hidden}", depth.get()),
            }
        }
        (PredictionExtensionKind::DeepSeekV4Embedded, SafetensorsModelConfig::DeepSeekV4(args))
            if args.dspark.is_some() =>
        {
            let strategy = DsparkPredictionStrategy::from_args(args)?;
            let hidden = positive(args.hidden_size, "hidden size")?;
            let architecture_capacity = NonZeroUsize::new(strategy.proposal_capacity())
                .ok_or_else(|| invalid("DSpark proposal capacity must be positive"))?;
            let layers = strategy.target_layer_ids().to_vec();
            let config = strategy.config();
            EmbeddedFamilyContract {
                family: "deepseek-v4-dspark",
                class: SpeculativeStrategyClass::EmbeddedFused,
                architecture_capacity,
                capture: EmbeddedCaptureContract::Dspark {
                    layers: layers.clone(),
                    batch,
                    sequence,
                    hidden,
                },
                state_components: indexed_state("prediction.dspark.pooling", depth.get()),
                additional_mechanisms: grouped
                    .into_iter()
                    .chain([SpeculativeMechanism::HyperNeuralOperations])
                    .collect(),
                strategy_detail: format!(
                    "blocks={};block_size={};noise={};markov={};layers={layers:?};hidden={hidden}",
                    depth.get(),
                    config.block_size,
                    config.noise_token_id,
                    config.markov_rank
                ),
            }
        }
        (PredictionExtensionKind::DeepSeekV4Embedded, SafetensorsModelConfig::DeepSeekV4(args)) => {
            let hidden = positive(args.hidden_size, "hidden size")?;
            let streams = positive(args.hc_mult, "hyper-connection stream count")?;
            EmbeddedFamilyContract {
                family: "deepseek-v4-mtp",
                class: SpeculativeStrategyClass::EmbeddedSequential,
                architecture_capacity: depth,
                capture: EmbeddedCaptureContract::Single {
                    path: "target.final_hyper_hidden".into(),
                    shape: vec![batch, sequence, streams, hidden],
                    observation: "prediction.target_capture".into(),
                },
                state_components: indexed_state("prediction.pooling", depth.get()),
                additional_mechanisms: grouped
                    .into_iter()
                    .chain([SpeculativeMechanism::HyperNeuralOperations])
                    .collect(),
                strategy_detail: format!("depth={};streams={streams};hidden={hidden}", depth.get()),
            }
        }
        (PredictionExtensionKind::InklingMtp, SafetensorsModelConfig::Inkling(args)) => {
            let hidden = positive(args.text_config.hidden_size, "hidden size")?;
            EmbeddedFamilyContract {
                family: "inkling-mtp",
                class: SpeculativeStrategyClass::EmbeddedSequential,
                architecture_capacity: depth,
                capture: EmbeddedCaptureContract::Single {
                    path: "target.final_hidden".into(),
                    shape: vec![batch, sequence, hidden],
                    observation: "prediction.target_capture".into(),
                },
                state_components: indexed_state("prediction.inkling", depth.get()),
                additional_mechanisms: grouped.into_iter().collect(),
                strategy_detail: format!("depth={};hidden={hidden}", depth.get()),
            }
        }
        (PredictionExtensionKind::QwenHybridMtp, SafetensorsModelConfig::QwenHybrid(args)) => {
            let hidden = positive(args.text.hidden_size, "hidden size")?;
            EmbeddedFamilyContract {
                family: "qwen-hybrid-mtp",
                class: SpeculativeStrategyClass::EmbeddedSequential,
                architecture_capacity: depth,
                capture: EmbeddedCaptureContract::Single {
                    path: "target.final_hidden".into(),
                    shape: vec![batch, sequence, hidden],
                    observation: "prediction.target_capture".into(),
                },
                state_components: indexed_state("prediction.qwen_hybrid", depth.get()),
                additional_mechanisms: grouped.into_iter().collect(),
                strategy_detail: format!("depth={};hidden={hidden}", depth.get()),
            }
        }
        (PredictionExtensionKind::NemotronHMtp, SafetensorsModelConfig::NemotronH(args)) => {
            let hidden = positive(args.hidden_size, "hidden size")?;
            let policies = args
                .mtp_policies()
                .map_err(|error| invalid(error.to_string()))?;
            EmbeddedFamilyContract {
                family: "nemotron-h-mtp",
                class: SpeculativeStrategyClass::EmbeddedSequential,
                architecture_capacity: depth,
                capture: EmbeddedCaptureContract::Single {
                    path: "target.final_hidden".into(),
                    shape: vec![batch, sequence, hidden],
                    observation: "prediction.target_capture".into(),
                },
                state_components: indexed_state("prediction.nemotron_h", policies.len()),
                additional_mechanisms: grouped.into_iter().collect(),
                strategy_detail: format!(
                    "depth={};physical_units={};hidden={hidden}",
                    depth.get(),
                    policies.len()
                ),
            }
        }
        _ => {
            return Err(invalid(
                "selected prediction extension has no embedded speculative contract",
            ));
        }
    };
    Ok(contract)
}

/// Returns the maximum proposal width owned by an embedded prediction architecture.
///
/// This differs from physical extension depth for fused strategies such as DSpark, whose
/// architecture-selected block may publish multiple proposal rows from one materialized unit.
pub fn embedded_prediction_capacity(
    extension: &PredictionExtensionPlan,
) -> Result<NonZeroUsize, eredu_core::artifact::ArtifactError> {
    embedded_family_contract(extension, 1, 1).map(|contract| contract.architecture_capacity)
}

fn topology_identity(
    topology: ParallelRankTopology,
) -> Result<SpeculativeIdentity, eredu_core::artifact::ArtifactError> {
    speculative_identity(format!(
        "tp={}:{};pp={}:{};ep={}:{};dp={}:{}",
        topology.tensor_parallel_size(),
        topology.tensor_parallel_rank(),
        topology.pipeline_parallel_size(),
        topology.pipeline_parallel_rank(),
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        topology.data_parallel_size(),
        topology.data_parallel_rank(),
    ))
}

fn capture_schema(
    family: &EmbeddedFamilyContract,
    owner: SpeculativeIdentity,
) -> Result<SpeculativeCaptureSchema, eredu_core::artifact::ArtifactError> {
    let (detail, entries) = match &family.capture {
        EmbeddedCaptureContract::Single {
            path,
            shape,
            observation,
        } => (
            format!("{path}:{shape:?}:{observation}"),
            vec![SpeculativeCaptureEntry::new(
                speculative_identity(path)?,
                shape.clone(),
                owner,
                speculative_identity(observation)?,
            )
            .and_then(|entry| entry.with_bounded_dimension(0))
            .and_then(|entry| entry.with_bounded_dimension(1))
            .map_err(|error| invalid(error.to_string()))?],
        ),
        EmbeddedCaptureContract::Dspark {
            layers,
            batch,
            sequence,
            hidden,
        } => {
            let entries = layers
                .iter()
                .copied()
                .enumerate()
                .map(|(position, layer)| {
                    SpeculativeCaptureEntry::new(
                        speculative_identity(format!("layers.{layer}.output"))?,
                        vec![*batch, *sequence, *hidden],
                        owner.clone(),
                        speculative_identity(format!("dspark.target_captures.{position}"))?,
                    )
                    .and_then(|entry| entry.with_bounded_dimension(0))
                    .and_then(|entry| entry.with_bounded_dimension(1))
                    .map_err(|error| invalid(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            (
                format!("layers={layers:?};shape=[{batch},{sequence},{hidden}]"),
                entries,
            )
        }
    };
    SpeculativeCaptureSchema::new(
        speculative_identity(format!("{}/capture/{detail}", family.family))?,
        entries,
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Derives the exact embedded strategy, capture, state, and mechanism contract before native work.
pub fn embedded_speculative_contract(
    extension: &PredictionExtensionPlan,
    request: EmbeddedSpeculativeContractRequest,
) -> Result<EmbeddedSpeculativeContract, eredu_core::artifact::ArtifactError> {
    let batch = request.maximum_batch_size.get();
    let sequence = request.maximum_sequence_length.get();
    let family = embedded_family_contract(extension, batch, sequence)?;
    finish_embedded_speculative_contract(family, request)
}

fn finish_embedded_speculative_contract(
    family: EmbeddedFamilyContract,
    request: EmbeddedSpeculativeContractRequest,
) -> Result<EmbeddedSpeculativeContract, eredu_core::artifact::ArtifactError> {
    if request.maximum_draft_tokens > family.architecture_capacity {
        return Err(invalid(format!(
            "requested embedded draft capacity {} exceeds architecture capacity {}",
            request.maximum_draft_tokens, family.architecture_capacity
        )));
    }
    let topology = topology_identity(request.topology)?;
    let owner = speculative_identity(format!(
        "{}/owner/rank/{}",
        topology.as_str(),
        request.topology.global_rank()
    ))?;
    let capture = capture_schema(&family, owner)?;
    let strategy_identity =
        speculative_identity(format!("{}/{}", family.family, family.strategy_detail))?;
    let strategy = SpeculativeStrategyRequirements::embedded(
        family.class,
        strategy_identity.clone(),
        request.maximum_draft_tokens,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let state_components = family
        .state_components
        .iter()
        .map(|component| speculative_identity(format!("{}/{component}", family.family)))
        .collect::<Result<Vec<_>, _>>()?;
    let state = SpeculativeStateCacheIdentityIngredients::new(
        request.target.clone(),
        strategy_identity.clone(),
        None,
        None,
        request.artifact,
        request.format,
        topology,
        request.topology.global_rank(),
        request.processor,
        state_components,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let mechanisms =
        SpeculativeMechanismRequirements::new(family.additional_mechanisms.into_iter().chain(
            (request.topology.world_size() > 1).then_some(SpeculativeMechanism::Communication),
        ));
    let requirements = SpeculativeRealizationRequirements::new(
        request.target.clone(),
        strategy,
        capture.clone(),
        mechanisms,
        state,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let architecture_proof = SpeculativeArchitectureCompatibilityProof::new(
        request.target,
        strategy_identity,
        capture.identity().clone(),
    );
    Ok(EmbeddedSpeculativeContract {
        requirements,
        architecture_proof,
        target_capture: capture,
    })
}

/// Exact fused DSpark behavior retained independently of an ordinary V4 target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkPredictionStrategy {
    config: crate::deepseek::DsparkConfig,
    capture_policy: crate::deepseek::config::V4TargetCapturePolicy,
    hidden_size: i32,
}

impl DsparkPredictionStrategy {
    /// Constructs the fused strategy from one complete DSpark configuration.
    pub fn from_args(
        args: &crate::deepseek::V4Args,
    ) -> Result<Self, eredu_core::artifact::ArtifactError> {
        args.validate()
            .map_err(|error| invalid(error.to_string()))?;
        let config = args
            .dspark
            .clone()
            .ok_or_else(|| invalid("DeepSeek-V4 configuration has no DSpark extension"))?;
        let capture_policy = args
            .target_capture_policy
            .clone()
            .ok_or_else(|| invalid("DSpark prediction has no target capture policy"))?;
        Ok(Self {
            config,
            capture_policy,
            hidden_size: args.hidden_size,
        })
    }

    /// Returns fused proposal capacity declared by the checkpoint.
    pub fn proposal_capacity(&self) -> usize {
        usize::try_from(self.config.block_size).expect("validated DSpark block size is positive")
    }

    /// Returns the exact target-layer identities in capture concatenation order.
    pub fn target_layer_ids(&self) -> &[usize] {
        self.capture_policy.layer_ids()
    }

    /// Returns the fused DSpark configuration retained outside the ordinary target.
    pub const fn config(&self) -> &crate::deepseek::DsparkConfig {
        &self.config
    }

    /// Returns the exact final dimension of the concatenated target capture.
    pub fn target_capture_width(&self) -> Result<i32, eredu_core::artifact::ArtifactError> {
        i32::try_from(self.capture_policy.len())
            .ok()
            .and_then(|count| count.checked_mul(self.hidden_size))
            .ok_or_else(|| invalid("DSpark target capture width overflowed"))
    }

    /// Validates one captured target value before it reaches fused projection.
    pub fn validate_target_capture_shape(
        &self,
        shape: &[i32],
    ) -> Result<(), eredu_core::artifact::ArtifactError> {
        let width = self.target_capture_width()?;
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != width {
            return Err(invalid(format!(
                "DSpark target capture must have shape [batch, sequence, {width}], got {shape:?}"
            )));
        }
        Ok(())
    }

    /// Validates one requested fused proposal width.
    pub fn validate_proposal_capacity(
        &self,
        capacity: usize,
    ) -> Result<(), eredu_core::artifact::ArtifactError> {
        let maximum = self.proposal_capacity();
        if capacity == 0 || capacity > maximum {
            return Err(invalid(format!(
                "DSpark proposal capacity must be between 1 and {maximum}, got {capacity}"
            )));
        }
        Ok(())
    }
}

/// One architecture-constructed neutral module and its exact checkpoint recipes.
pub struct PreparedPredictionUnit<M> {
    source: M,
    local: M,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
}

/// Architecture-constructed fused DSpark modules and immutable strategy.
pub struct PreparedDsparkPredictionExtension<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    strategy: DsparkPredictionStrategy,
    static_modules: PreparedPredictionUnit<crate::deepseek::v4::DsparkStatic<B>>,
}

impl<B> PreparedDsparkPredictionExtension<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Borrows the exact fused strategy selected from configuration.
    pub const fn strategy(&self) -> &DsparkPredictionStrategy {
        &self.strategy
    }

    /// Consumes the handoff into fused strategy and checkpoint-global/rank-local statics.
    pub fn into_parts(
        self,
    ) -> (
        DsparkPredictionStrategy,
        PreparedPredictionUnit<crate::deepseek::v4::DsparkStatic<B>>,
    ) {
        (self.strategy, self.static_modules)
    }
}

impl<M> PreparedPredictionUnit<M> {
    fn new(source: M, local: M, recipes: BTreeMap<String, DerivedWeightRecipe>) -> Self {
        Self {
            source,
            local,
            recipes,
        }
    }

    /// Consumes the handoff into the checkpoint-global module, rank-local module, and recipes.
    pub fn into_parts(self) -> (M, M, BTreeMap<String, DerivedWeightRecipe>) {
        (self.source, self.local, self.recipes)
    }
}

/// Architecture-selected neutral extension construction for one execution rank.
pub enum PreparedPredictionExtension<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// DeepSeek-V3 sequential MTP units.
    DeepSeekV3 {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: LocalModelLayout,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v3::Unit<B>>>,
    },
    /// DeepSeek-V4 sequential MTP units and their immutable cache policies.
    DeepSeekV4 {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: LocalModelLayout,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>,
        /// Ordered rank-local cache policy for every prediction unit.
        state: Vec<(usize, LayerCachePolicy)>,
    },
    /// DeepSeek-V4 fused DSpark blocks, pinned modules, and immutable cache policies.
    DeepSeekV4Dspark {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: LocalModelLayout,
        /// Fused strategy and its checkpoint-global/rank-local pinned modules.
        extension: PreparedDsparkPredictionExtension<B>,
        /// Ordered checkpoint-global/rank-local ordinary DSpark block pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>,
        /// Ordered rank-local cache policy for every DSpark block.
        state: Vec<(usize, LayerCachePolicy)>,
    },
    /// Inkling sequential MTP module.
    Inkling {
        /// Checkpoint-global and execution-local module pair.
        model: PreparedPredictionUnit<crate::inkling::MtpModel<B>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
    /// Dense Qwen hybrid MTP units.
    QwenHybrid {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: LocalModelLayout,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::qwen::hybrid::PredictionUnit<B>>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
    /// Nemotron-H patterned MTP groups.
    NemotronH {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: LocalModelLayout,
        /// Prediction-step groups in architecture execution order.
        groups: Vec<Vec<PreparedPredictionUnit<crate::nemotron_h::PredictionUnit<B>>>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
}

/// Family-blind backend mechanisms used to materialize an architecture-owned
/// prediction extension.
///
/// The architecture retains variant selection, parameter topology, state
/// geometry, and construction order. A concrete backend only supplies the
/// generic operations required to turn one prepared module or neutral state
/// description into its native representation.
pub trait PredictionExtensionMaterializer<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Backend materialization failure.
    type Error;
    /// Native wrapper around any architecture-constructed module.
    type Module<M>: AsMut<M>;
    /// Native cache for one pooling-attention prediction layer.
    type PoolingState: Clone + eredu_nn::PoolingAttentionCache<B::Tensor> + 'static;
    /// Native cache for one sequential compressed-latent prediction layer.
    type SequentialState: Clone + eredu_nn::CompressedAttentionCache<B::Tensor> + 'static;
    /// Native realization of an architecture-declared state layout.
    type ModelState: PredictionModelState<B> + 'static;
    /// Borrowed resources used during one materialization pass.
    type Context<'a>;

    /// Materializes one arbitrary architecture module and its derived recipes.
    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: PreparedPredictionUnit<M>,
        layout: Option<&LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<B::Tensor>;

    /// Realizes one architecture-selected pooling-attention cache policy.
    fn pooling_state(
        context: &mut Self::Context<'_>,
        ordinal: usize,
        policy: LayerCachePolicy,
    ) -> Result<Self::PoolingState, Self::Error>;

    /// Realizes one complete architecture-selected prediction state layout.
    fn model_state(
        context: &mut Self::Context<'_>,
        layout: StateLayout,
    ) -> Result<Self::ModelState, Self::Error>;

    /// Constructs an empty sequential prediction-layer cache.
    fn sequential_state() -> Self::SequentialState;
}

/// Mutable prediction-only state with contiguous architecture-ordered layers.
///
/// This is deliberately narrower than a backend's complete model-state API:
/// embedded extensions only need the ordered layer slice selected by the
/// architecture when it constructed the extension.
pub trait PredictionModelState<B>: Clone + eredu_runtime::RuntimeState<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Concrete state consumed by an embedded prediction unit.
    type LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>;

    /// Borrows prediction layers in the architecture-declared execution order.
    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState];
}

/// Family-blind mechanism for invoking an architecture-owned operation against
/// the singular typed target session.
pub trait PredictionOperationInvoker<A, B, S>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
{
    /// Backend/session invocation failure.
    type Error;

    /// Invokes one typed architecture operation without erasing its target.
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Self::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, B, S>;

    /// Constructs a mechanism error for a rejected architecture contract.
    fn invalid(message: String) -> Self::Error;
}

mod executor_sealed {
    pub trait Sealed<A> {}
}

fn validate_selected_capture_shapes(
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    lane: &eredu_runtime::SpeculativeLaneIdentity,
    shapes: Vec<Vec<usize>>,
) -> Result<(), eredu_runtime::SpeculativeCaptureError> {
    let actual = selected.requirements().capture().instantiate(shapes)?;
    let values = vec![(); actual.entries().len()];
    let envelope = eredu_runtime::SpeculativeCaptureEnvelope::new(
        eredu_runtime::SpeculativeCaptureMetadata::new(actual, lane.capture_generation()),
        values,
    )?;
    selected.validate_capture(lane, &envelope)
}

/// Architecture-owned executable strategy for one exactly paired materialized
/// prediction extension.
///
/// Implementations own proposal depth, lane-state geometry, prefill/advance
/// ordering, sequential versus fused execution, and every family-specific
/// operation. A backend supplies only typed operation invocation and storage.
pub trait MaterializedPredictionExecutor<A, B, M>: executor_sealed::Sealed<A>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Exact mutable state retained for one speculative lane.
    type LaneState: Clone + 'static;

    /// Returns the proposal depth exposed to the scheduler.
    fn depth(&self) -> usize;

    /// Returns whether extension prefill consumes only the final target token.
    fn prefill_single_token(&self) -> bool {
        false
    }

    /// Clones the architecture-materialized lane-state prototype.
    fn new_state(&self) -> Self::LaneState;

    /// Maps one physical target capture into the ordered logical entry shapes
    /// declared by the selected architecture contract.
    fn logical_capture_shapes(
        &self,
        physical_shape: &[i32],
    ) -> Result<Vec<Vec<usize>>, eredu_runtime::SpeculativeCaptureError> {
        let shape = physical_shape
            .iter()
            .copied()
            .map(|extent| {
                usize::try_from(extent)
                    .ok()
                    .filter(|extent| *extent > 0)
                    .ok_or(eredu_runtime::SpeculativeCaptureError::ShapeMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![shape])
    }

    /// Validates a physical target capture against the exact selected lane
    /// before any extension state can be mutated.
    fn validate_capture(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        lane: &eredu_runtime::SpeculativeLaneIdentity,
        physical_shape: &[i32],
    ) -> Result<(), eredu_runtime::SpeculativeCaptureError> {
        validate_selected_capture_shapes(
            selected,
            lane,
            self.logical_capture_shapes(physical_shape)?,
        )
    }

    /// Advances extension state across a target-prefill observation.
    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>;

    /// Executes exactly one sequential proposal depth.
    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>;

    /// Executes a fused proposal when selected by the architecture.
    fn fused_logits<S, I>(
        &mut self,
        _invoker: &mut I,
        _anchor: &B::Tensor,
        _capacity: usize,
        _lane: &Self::LaneState,
    ) -> Result<Option<B::Tensor>, I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        Ok(None)
    }

    /// Advances extension state after committed target execution.
    fn advance<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill::<S, I>(invoker, hidden, hidden, tokens, lane)
    }
}

/// Architecture-owned prediction topology after generic backend
/// materialization.
pub enum MaterializedPredictionExtension<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// DeepSeek-V3 sequential MTP units.
    DeepSeekV3 {
        /// Units in proposal order.
        units: Vec<M::Module<crate::deepseek::v3::Unit<B>>>,
    },
    /// DeepSeek-V4 sequential MTP units.
    DeepSeekV4 {
        /// Units in proposal order.
        units: Vec<M::Module<crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in unit order.
        state: Vec<M::PoolingState>,
    },
    /// DeepSeek-V4 fused DSpark execution.
    DeepSeekV4Dspark {
        /// Exact fused strategy retained from configuration.
        strategy: DsparkPredictionStrategy,
        /// Checkpoint-global fused projection modules.
        static_modules: M::Module<crate::deepseek::v4::DsparkStatic<B>>,
        /// Physical blocks in fused execution order.
        units: Vec<M::Module<crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in block order.
        state: Vec<M::PoolingState>,
    },
    /// Inkling sequential prediction module.
    Inkling {
        /// Complete MTP module.
        model: M::Module<crate::inkling::MtpModel<B>>,
        /// Number of sequential proposals exposed by the module.
        depth: usize,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
    /// Dense Qwen hybrid MTP units.
    QwenHybrid {
        /// Units in proposal order.
        units: Vec<M::Module<crate::qwen::hybrid::PredictionUnit<B>>>,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
    /// Nemotron-H patterned MTP groups.
    NemotronH {
        /// Groups in proposal order and units in physical execution order.
        groups: Vec<Vec<M::Module<crate::nemotron_h::PredictionUnit<B>>>>,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
}

/// Marker stored by targets that do not admit an embedded prediction extension.
pub struct NoMaterializedPrediction;

/// Materialized DeepSeek-V3 prediction units paired with their exact target.
pub struct MaterializedDeepSeekV3Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Units in proposal order.
    pub units: Vec<M::Module<crate::deepseek::v3::Unit<B>>>,
}

/// Materialized DeepSeek-V4 prediction execution paired with its exact target.
pub enum MaterializedDeepSeekV4Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Sequential MTP units and their cache prototypes.
    Sequential {
        /// Units in proposal order.
        units: Vec<M::Module<crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in unit order.
        state: Vec<M::PoolingState>,
    },
    /// Fused DSpark modules and their cache prototypes.
    Dspark {
        /// Exact fused proposal strategy.
        strategy: DsparkPredictionStrategy,
        /// Checkpoint-global fused projection modules.
        static_modules: M::Module<crate::deepseek::v4::DsparkStatic<B>>,
        /// Physical blocks in fused execution order.
        units: Vec<M::Module<crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in block order.
        state: Vec<M::PoolingState>,
    },
}

/// Materialized Inkling prediction module paired with its exact target.
pub struct MaterializedInklingPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Complete MTP module.
    pub model: M::Module<crate::inkling::MtpModel<B>>,
    /// Number of sequential proposals exposed by the module.
    pub depth: usize,
    /// Immutable prediction-state prototype.
    pub state: M::ModelState,
}

/// Materialized Qwen-hybrid prediction units paired with their exact target.
pub struct MaterializedQwenHybridPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Units in proposal order.
    pub units: Vec<M::Module<crate::qwen::hybrid::PredictionUnit<B>>>,
    /// Immutable prediction-state prototype.
    pub state: M::ModelState,
}

/// Materialized Nemotron-H prediction groups paired with their exact target.
pub struct MaterializedNemotronHPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Groups in proposal order and units in physical execution order.
    pub groups: Vec<Vec<M::Module<crate::nemotron_h::PredictionUnit<B>>>>,
    /// Immutable prediction-state prototype.
    pub state: M::ModelState,
}

pub(crate) trait NemotronHPredictionTarget<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;

    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;
}

impl<B> NemotronHPredictionTarget<B> for crate::nemotron_h::PartitionedLayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        match parallel {
            Some(parallel) => self.embed_prediction_parallel(tokens, parallel, context),
            None => self.embed_prediction(tokens, context),
        }
    }

    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        match parallel {
            Some(parallel) => self.project_prediction_parallel(hidden, parallel, context),
            None => self.project_prediction(hidden, context),
        }
    }
}

impl<B> NemotronHPredictionTarget<B> for crate::nemotron_h::LayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        use eredu_nn::EmbeddingOperator;
        match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                &mut self.static_modules_mut().embeddings,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => self
                .static_modules_mut()
                .embeddings
                .forward(tokens, context),
        }
    }

    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        use eredu_nn::{EmbeddingOperator, LinearOperator};
        let modules = self.static_modules_mut();
        match (parallel, modules.lm_head.as_mut()) {
            (Some(parallel), Some(head)) => {
                B::vocabulary_parallel_project(head, hidden, parallel, context)
            }
            (None, Some(head)) => head.forward(hidden, context),
            (Some(parallel), None) => B::vocabulary_parallel_embedding_project(
                &mut modules.embeddings,
                hidden,
                parallel,
                context,
            ),
            (None, None) => modules.embeddings.as_linear(hidden, context),
        }
    }
}

struct V3PredictionOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut M::Module<crate::deepseek::v3::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut M::SequentialState,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v3::Model<B>, B, S>
    for V3PredictionOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v3::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    M::SequentialState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v3::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        match parallel {
            Some(parallel) => architecture.pipeline_forward_prediction_neutral_parallel(
                self.unit.as_mut(),
                self.hidden,
                self.tokens,
                self.cache,
                parallel,
                context,
            ),
            None => architecture.pipeline_forward_prediction(
                self.unit.as_mut(),
                self.hidden,
                self.tokens,
                self.cache,
                context,
            ),
        }
        .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

struct V4PredictionOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut M::Module<crate::deepseek::v4::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut M::PoolingState,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4PredictionOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        let hidden = architecture.begin_partition_prediction_hidden(self.hidden, context)?;
        let output = match parallel {
            Some(parallel) => architecture.pipeline_forward_prediction_neutral_parallel(
                self.unit.as_mut(),
                &hidden,
                self.tokens,
                self.cache,
                parallel,
                context,
            ),
            None => architecture.pipeline_forward_prediction(
                self.unit.as_mut(),
                &hidden,
                self.tokens,
                self.cache,
                context,
            ),
        }?;
        architecture.finish_partition_prediction_output(output, context)
    }
}

trait InklingPredictionTarget<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        model: &mut crate::inkling::MtpModel<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;
}

impl<B> InklingPredictionTarget<B> for crate::inkling::LayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        model: &mut crate::inkling::MtpModel<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        let embeddings = match parallel {
            Some(parallel) => self.mtp_token_embeddings_parallel(tokens, parallel, context)?,
            None => self.mtp_token_embeddings(tokens, context)?,
        };
        let output = model.forward_step(
            hidden,
            &embeddings,
            tokens,
            depth,
            state.prediction_layers_mut(),
            context,
        )?;
        let logits = match parallel {
            Some(parallel) => {
                self.project_mtp_logits_parallel(&output.hidden, parallel, context)?
            }
            None => self.project_mtp_logits(&output.hidden, context)?,
        };
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: output.hidden,
            tokens: output.tokens,
        })
    }
}

impl<B> InklingPredictionTarget<B>
    for crate::composite_execution::PreparedCompositeArchitecture<crate::inkling::LayeredModel<B>>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        model: &mut crate::inkling::MtpModel<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        self.inner_mut()
            .prediction_step::<M>(model, hidden, tokens, depth, state, parallel, context)
    }
}

struct InklingPredictionOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    model: &'a mut M::Module<crate::inkling::MtpModel<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for InklingPredictionOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + InklingPredictionTarget<B>,
    M: PredictionExtensionMaterializer<B>,
{
    type Output = crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        architecture.prediction_step::<M>(
            self.model.as_mut(),
            self.hidden,
            self.tokens,
            self.depth,
            self.state,
            parallel,
            context,
        )
    }
}

trait QwenHybridPredictionTarget<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::qwen::hybrid::PredictionUnit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>;
}

impl<B> QwenHybridPredictionTarget<B> for crate::qwen::hybrid::ConditionalLayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::qwen::hybrid::PredictionUnit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        let embedded = match parallel {
            Some(parallel) => {
                self.begin_partition_prediction_embedding_parallel(tokens, parallel, context)?
            }
            None => self.begin_partition_prediction_embedding(tokens, context)?,
        };
        let layer = state
            .prediction_layers_mut()
            .get_mut(depth)
            .ok_or_else(|| eredu_nn::Error::backend("Qwen prediction state is too shallow"))?;
        let sequence = tokens.dim(1);
        let mask = if sequence > 1 {
            Some(B::causal_mask(sequence, layer.position(), None, context)?)
        } else {
            None
        };
        let hidden = match parallel {
            Some(parallel) => unit.forward_parallel(
                hidden,
                &embedded,
                mask.as_ref(),
                layer,
                parallel,
                context,
                &mut eredu_runtime::ResidentExpertProvider,
            )?,
            None => unit.forward(hidden, &embedded, mask.as_ref(), layer, context)?,
        };
        let logits = match parallel {
            Some(parallel) => {
                self.finish_partition_prediction_parallel(&hidden, parallel, context)?
            }
            None => self.finish_partition_prediction(&hidden, context)?,
        };
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: hidden,
            tokens: tokens.clone(),
        })
    }
}

impl<B> QwenHybridPredictionTarget<B>
    for crate::composite_execution::PreparedCompositeArchitecture<
        crate::qwen::hybrid::ConditionalLayeredModel<B>,
    >
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::qwen::hybrid::PredictionUnit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        self.inner_mut()
            .prediction_step::<M>(unit, hidden, tokens, depth, state, parallel, context)
    }
}

struct QwenHybridPredictionOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut M::Module<crate::qwen::hybrid::PredictionUnit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for QwenHybridPredictionOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + QwenHybridPredictionTarget<B>,
    M: PredictionExtensionMaterializer<B>,
{
    type Output = crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        architecture.prediction_step::<M>(
            self.unit.as_mut(),
            self.hidden,
            self.tokens,
            self.depth,
            self.state,
            parallel,
            context,
        )
    }
}

struct NemotronHPredictionOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    units: &'a mut [M::Module<crate::nemotron_h::PredictionUnit<B>>],
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for NemotronHPredictionOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + NemotronHPredictionTarget<B>,
    M: PredictionExtensionMaterializer<B>,
{
    type Output = crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        if self.units.is_empty() {
            return Err(eredu_nn::Error::backend(
                "Nemotron-H prediction group has no physical units",
            ));
        }
        let embedded = architecture.embed_prediction(self.tokens, parallel, context)?;
        let state_start = self
            .depth
            .checked_mul(self.units.len())
            .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H MTP state index overflowed"))?;
        let states = self.state.prediction_layers_mut();
        let sequence = self.tokens.dim(1);
        let position = states
            .get(state_start)
            .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H prediction state is too shallow"))?
            .position();
        let mask = if sequence > 1 {
            Some(B::causal_mask(sequence, position, None, context)?)
        } else {
            None
        };
        let mut hidden = self.hidden.clone();
        for (relative, unit) in self.units.iter_mut().enumerate() {
            let state = states.get_mut(state_start + relative).ok_or_else(|| {
                eredu_nn::Error::backend("Nemotron-H prediction state is too shallow")
            })?;
            hidden = match parallel {
                Some(parallel) => unit.as_mut().forward_parallel_with_provider(
                    &hidden,
                    &embedded,
                    mask.as_ref(),
                    state,
                    parallel,
                    context,
                    &mut eredu_runtime::ResidentExpertProvider,
                )?,
                None => unit
                    .as_mut()
                    .forward(&hidden, &embedded, mask.as_ref(), state, context)?,
            };
        }
        let logits = architecture.project_prediction(&hidden, parallel, context)?;
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: hidden,
            tokens: self.tokens.clone(),
        })
    }
}

impl<B, M> executor_sealed::Sealed<crate::deepseek::v3::Model<B>>
    for MaterializedDeepSeekV3Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
}

impl<B, M> MaterializedPredictionExecutor<crate::deepseek::v3::Model<B>, B, M>
    for MaterializedDeepSeekV3Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    M::SequentialState: 'static,
{
    type LaneState = Vec<M::SequentialState>;

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        (0..self.units.len())
            .map(|_| M::sequential_state())
            .collect()
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v3::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v3::Model<B>, B, S>,
    {
        if self.units.len() != lane.len() {
            return Err(I::invalid(
                "DeepSeek-V3 prediction units and lane state differ".into(),
            ));
        }
        self.units
            .iter_mut()
            .zip(lane)
            .try_for_each(|(unit, cache)| {
                invoker
                    .invoke(V3PredictionOperation::<B, M> {
                        unit,
                        hidden,
                        tokens,
                        cache,
                    })
                    .map(|_| ())
            })
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v3::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v3::Model<B>, B, S>,
    {
        let count = self.units.len();
        let unit = self.units.get_mut(draft_index).ok_or_else(|| {
            I::invalid(format!(
                "prediction depth {draft_index} exceeds {count} units"
            ))
        })?;
        let cache = lane
            .get_mut(draft_index)
            .ok_or_else(|| I::invalid("DeepSeek-V3 prediction lane is too shallow".into()))?;
        invoker
            .invoke(V3PredictionOperation::<B, M> {
                unit,
                hidden,
                tokens: token,
                cache,
            })
            .map(|output| (output.logits, output.hidden))
    }
}

impl<B, M> executor_sealed::Sealed<crate::deepseek::v4::Model<B>>
    for MaterializedDeepSeekV4Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
}

impl<B, M> MaterializedPredictionExecutor<crate::deepseek::v4::Model<B>, B, M>
    for MaterializedDeepSeekV4Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    M::PoolingState: 'static,
{
    type LaneState = Vec<M::PoolingState>;

    fn depth(&self) -> usize {
        match self {
            Self::Sequential { units, .. } | Self::Dspark { units, .. } => units.len(),
        }
    }

    fn prefill_single_token(&self) -> bool {
        matches!(self, Self::Dspark { .. })
    }

    fn new_state(&self) -> Self::LaneState {
        match self {
            Self::Sequential { state, .. } | Self::Dspark { state, .. } => state.clone(),
        }
    }

    fn logical_capture_shapes(
        &self,
        physical_shape: &[i32],
    ) -> Result<Vec<Vec<usize>>, eredu_runtime::SpeculativeCaptureError> {
        match self {
            Self::Sequential { .. } => {
                let shape = physical_shape
                    .iter()
                    .copied()
                    .map(|extent| {
                        usize::try_from(extent)
                            .ok()
                            .filter(|extent| *extent > 0)
                            .ok_or(eredu_runtime::SpeculativeCaptureError::ShapeMismatch)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![shape])
            }
            Self::Dspark { strategy, .. } => {
                strategy
                    .validate_target_capture_shape(physical_shape)
                    .map_err(|_| eredu_runtime::SpeculativeCaptureError::ShapeMismatch)?;
                let [batch, sequence, width] = physical_shape else {
                    return Err(eredu_runtime::SpeculativeCaptureError::ShapeMismatch);
                };
                let count = strategy.target_layer_ids().len();
                let width = usize::try_from(*width)
                    .ok()
                    .filter(|width| count > 0 && width % count == 0)
                    .ok_or(eredu_runtime::SpeculativeCaptureError::ShapeMismatch)?;
                let batch = usize::try_from(*batch)
                    .ok()
                    .filter(|extent| *extent > 0)
                    .ok_or(eredu_runtime::SpeculativeCaptureError::ShapeMismatch)?;
                let sequence = usize::try_from(*sequence)
                    .ok()
                    .filter(|extent| *extent > 0)
                    .ok_or(eredu_runtime::SpeculativeCaptureError::ShapeMismatch)?;
                Ok((0..count)
                    .map(|_| vec![batch, sequence, width / count])
                    .collect())
            }
        }
    }

    fn validate_capture(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        lane: &eredu_runtime::SpeculativeLaneIdentity,
        physical_shape: &[i32],
    ) -> Result<(), eredu_runtime::SpeculativeCaptureError> {
        if let Self::Dspark { strategy, .. } = self {
            let entries = selected.requirements().capture().entries();
            if entries.len() != strategy.target_layer_ids().len()
                || entries
                    .iter()
                    .zip(strategy.target_layer_ids())
                    .any(|(entry, layer)| entry.path().as_str() != format!("layers.{layer}.output"))
            {
                return Err(eredu_runtime::SpeculativeCaptureError::SchemaMismatch);
            }
        }
        validate_selected_capture_shapes(
            selected,
            lane,
            self.logical_capture_shapes(physical_shape)?,
        )
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v4::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v4::Model<B>, B, S>,
    {
        match self {
            Self::Sequential { units, .. } => {
                if units.len() != lane.len() {
                    return Err(I::invalid(
                        "DeepSeek-V4 prediction units and lane state differ".into(),
                    ));
                }
                units.iter_mut().zip(lane).try_for_each(|(unit, cache)| {
                    invoker
                        .invoke(V4PredictionOperation::<B, M> {
                            unit,
                            hidden,
                            tokens,
                            cache,
                        })
                        .map(|_| ())
                })
            }
            Self::Dspark {
                strategy,
                static_modules,
                units,
                ..
            } => {
                if units.len() != lane.len() {
                    return Err(I::invalid(
                        "DSpark prediction units and lane state differ".into(),
                    ));
                }
                invoker.invoke(V4DsparkContextOperation::<B, M> {
                    strategy,
                    static_modules,
                    units,
                    captures: target_capture,
                    caches: lane,
                })
            }
        }
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v4::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v4::Model<B>, B, S>,
    {
        match self {
            Self::Sequential { units, .. } => {
                let count = units.len();
                let unit = units.get_mut(draft_index).ok_or_else(|| {
                    I::invalid(format!(
                        "prediction depth {draft_index} exceeds {count} units"
                    ))
                })?;
                let cache = lane.get_mut(draft_index).ok_or_else(|| {
                    I::invalid("DeepSeek-V4 prediction lane is too shallow".into())
                })?;
                invoker
                    .invoke(V4PredictionOperation::<B, M> {
                        unit,
                        hidden,
                        tokens: token,
                        cache,
                    })
                    .map(|output| (output.logits, output.hidden))
            }
            Self::Dspark { .. } => Err(I::invalid("DSpark uses fused prediction proposals".into())),
        }
    }

    fn fused_logits<S, I>(
        &mut self,
        invoker: &mut I,
        anchor: &B::Tensor,
        capacity: usize,
        lane: &Self::LaneState,
    ) -> Result<Option<B::Tensor>, I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v4::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v4::Model<B>, B, S>,
    {
        match self {
            Self::Dspark {
                strategy,
                static_modules,
                units,
                ..
            } => {
                if units.len() != lane.len() {
                    return Err(I::invalid(
                        "DSpark prediction units and lane state differ".into(),
                    ));
                }
                let mut proposal_state = lane.clone();
                invoker
                    .invoke(V4DsparkProposalOperation::<B, M> {
                        strategy,
                        static_modules,
                        units,
                        anchor,
                        capacity,
                        caches: &mut proposal_state,
                    })
                    .map(Some)
            }
            Self::Sequential { .. } => Ok(None),
        }
    }
}

impl<A, B, M> executor_sealed::Sealed<A> for MaterializedInklingPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    A: InklingPredictionTarget<B>,
{
}

impl<A, B, M> MaterializedPredictionExecutor<A, B, M> for MaterializedInklingPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    M::ModelState: 'static,
    A: InklingPredictionTarget<B>,
{
    type LaneState = M::ModelState;

    fn depth(&self) -> usize {
        self.depth
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        (0..self.depth).try_for_each(|depth| {
            invoker
                .invoke(InklingPredictionOperation::<B, M> {
                    model: &mut self.model,
                    hidden,
                    tokens,
                    depth,
                    state: lane,
                })
                .map(|_| ())
        })
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if draft_index >= self.depth {
            return Err(I::invalid(format!(
                "prediction depth {draft_index} exceeds {} units",
                self.depth
            )));
        }
        invoker
            .invoke(InklingPredictionOperation::<B, M> {
                model: &mut self.model,
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
            })
            .map(|output| (output.logits, output.capture))
    }
}

impl<A, B, M> executor_sealed::Sealed<A> for MaterializedQwenHybridPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    A: QwenHybridPredictionTarget<B>,
{
}

impl<A, B, M> MaterializedPredictionExecutor<A, B, M> for MaterializedQwenHybridPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    M::ModelState: 'static,
    A: QwenHybridPredictionTarget<B>,
{
    type LaneState = M::ModelState;

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if self.units.len() != lane.prediction_layers_mut().len() {
            return Err(I::invalid(
                "Qwen hybrid prediction units and state differ".into(),
            ));
        }
        self.units
            .iter_mut()
            .enumerate()
            .try_for_each(|(depth, unit)| {
                invoker
                    .invoke(QwenHybridPredictionOperation::<B, M> {
                        unit,
                        hidden,
                        tokens,
                        depth,
                        state: lane,
                    })
                    .map(|_| ())
            })
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        let count = self.units.len();
        let unit = self.units.get_mut(draft_index).ok_or_else(|| {
            I::invalid(format!(
                "prediction depth {draft_index} exceeds {count} units"
            ))
        })?;
        invoker
            .invoke(QwenHybridPredictionOperation::<B, M> {
                unit,
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
            })
            .map(|output| (output.logits, output.capture))
    }
}

impl<A, B, M> executor_sealed::Sealed<A> for MaterializedNemotronHPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    A: NemotronHPredictionTarget<B>,
{
}

impl<A, B, M> MaterializedPredictionExecutor<A, B, M> for MaterializedNemotronHPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    M::ModelState: 'static,
    A: NemotronHPredictionTarget<B>,
{
    type LaneState = M::ModelState;

    fn depth(&self) -> usize {
        self.groups.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.groups
            .iter_mut()
            .enumerate()
            .try_for_each(|(depth, units)| {
                invoker
                    .invoke(NemotronHPredictionOperation::<B, M> {
                        units,
                        hidden,
                        tokens,
                        depth,
                        state: lane,
                    })
                    .map(|_| ())
            })
    }

    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        let count = self.groups.len();
        let units = self.groups.get_mut(draft_index).ok_or_else(|| {
            I::invalid(format!(
                "prediction depth {draft_index} exceeds {count} groups"
            ))
        })?;
        invoker
            .invoke(NemotronHPredictionOperation::<B, M> {
                units,
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
            })
            .map(|output| (output.logits, output.capture))
    }
}

struct V4DsparkContextOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    strategy: &'a DsparkPredictionStrategy,
    static_modules: &'a mut M::Module<crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [M::Module<crate::deepseek::v4::Unit<B>>],
    captures: &'a B::Tensor,
    caches: &'a mut [M::PoolingState],
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4DsparkContextOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = ();

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        architecture.pipeline_prefill_dspark_extension_context(
            self.strategy,
            self.static_modules.as_mut(),
            self.units,
            self.captures,
            self.caches,
            context,
        )
    }
}

struct V4DsparkProposalOperation<'a, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    strategy: &'a DsparkPredictionStrategy,
    static_modules: &'a mut M::Module<crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [M::Module<crate::deepseek::v4::Unit<B>>],
    anchor: &'a B::Tensor,
    capacity: usize,
    caches: &'a mut [M::PoolingState],
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4DsparkProposalOperation<'_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = B::Tensor;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        match parallel {
            Some(parallel) => architecture.pipeline_dspark_extension_proposal_neutral_parallel(
                self.strategy,
                self.static_modules.as_mut(),
                self.units,
                self.anchor,
                self.capacity,
                self.caches,
                parallel,
                context,
            ),
            None => architecture.pipeline_dspark_extension_proposal(
                self.strategy,
                self.static_modules.as_mut(),
                self.units,
                self.anchor,
                self.capacity,
                self.caches,
                context,
            ),
        }
    }
}

mod pairing_sealed {
    pub trait Sealed {}
}

/// Architecture-owned proof that one materialized extension belongs to one
/// exact ordinary target architecture.
pub trait MaterializedPredictionTarget<B>: pairing_sealed::Sealed
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Target-specific extension shape retained after pairing.
    type Extension<M>: MaterializedPredictionExecutor<Self, B, M>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;

    /// Consumes and validates the architecture-owned materialized extension.
    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;
}

macro_rules! sealed_prediction_target {
    ($target:ty) => {
        impl<B> pairing_sealed::Sealed for $target where
            B: BlockwiseAttentionBackend
                + DistributedNeuralBackend
                + GroupedNeuralBackend
                + HyperNeuralBackend
        {
        }
    };
}

sealed_prediction_target!(crate::deepseek::v3::Model<B>);
sealed_prediction_target!(crate::deepseek::v4::Model<B>);
sealed_prediction_target!(crate::nemotron_h::PartitionedLayeredModel<B>);
sealed_prediction_target!(crate::nemotron_h::LayeredModel<B>);
sealed_prediction_target!(
    crate::composite_execution::PreparedCompositeArchitecture<crate::inkling::LayeredModel<B>>
);
sealed_prediction_target!(crate::inkling::LayeredModel<B>);
sealed_prediction_target!(
    crate::composite_execution::PreparedCompositeArchitecture<
        crate::qwen::hybrid::ConditionalLayeredModel<B>,
    >
);
sealed_prediction_target!(crate::qwen::hybrid::ConditionalLayeredModel<B>);

impl<B, P> pairing_sealed::Sealed
    for crate::replicated_model::FixedReplicatedModel<
        B,
        crate::replicated_text::NemotronHReplicated,
        P,
    >
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
}

impl<B> MaterializedPredictionTarget<B> for crate::deepseek::v3::Model<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedDeepSeekV3Prediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::DeepSeekV3 { units } => {
                Ok(MaterializedDeepSeekV3Prediction { units })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the DeepSeek-V3 target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::deepseek::v4::Model<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedDeepSeekV4Prediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::DeepSeekV4 { units, state } => {
                Ok(MaterializedDeepSeekV4Prediction::Sequential { units, state })
            }
            MaterializedPredictionExtension::DeepSeekV4Dspark {
                strategy,
                static_modules,
                units,
                state,
            } => Ok(MaterializedDeepSeekV4Prediction::Dspark {
                strategy,
                static_modules,
                units,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the DeepSeek-V4 target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::nemotron_h::PartitionedLayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedNemotronHPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::NemotronH { groups, state } => {
                Ok(MaterializedNemotronHPrediction { groups, state })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the Nemotron-H target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::nemotron_h::LayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedNemotronHPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::NemotronH { groups, state } => {
                Ok(MaterializedNemotronHPrediction { groups, state })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the Nemotron-H target",
            )),
        }
    }
}

impl<B, P> MaterializedPredictionTarget<B>
    for crate::replicated_model::FixedReplicatedModel<
        B,
        crate::replicated_text::NemotronHReplicated,
        P,
    >
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedNemotronHPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::NemotronH { groups, state } => {
                Ok(MaterializedNemotronHPrediction { groups, state })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the replicated Nemotron-H target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B>
    for crate::composite_execution::PreparedCompositeArchitecture<crate::inkling::LayeredModel<B>>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedInklingPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::Inkling {
                model,
                depth,
                state,
            } => Ok(MaterializedInklingPrediction {
                model,
                depth,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the Inkling target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::inkling::LayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedInklingPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::Inkling {
                model,
                depth,
                state,
            } => Ok(MaterializedInklingPrediction {
                model,
                depth,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the Inkling target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B>
    for crate::composite_execution::PreparedCompositeArchitecture<
        crate::qwen::hybrid::ConditionalLayeredModel<B>,
    >
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedQwenHybridPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::QwenHybrid { units, state } => {
                Ok(MaterializedQwenHybridPrediction { units, state })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the Qwen-hybrid target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::qwen::hybrid::ConditionalLayeredModel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
{
    type Extension<M>
        = MaterializedQwenHybridPrediction<B, M>
    where
        M: PredictionExtensionMaterializer<B>;

    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        match extension {
            MaterializedPredictionExtension::QwenHybrid { units, state } => {
                Ok(MaterializedQwenHybridPrediction { units, state })
            }
            _ => Err(invalid(
                "materialized prediction extension does not match the conditional Qwen target",
            )),
        }
    }
}

/// Architecture-owned mutable lane-state geometry for one materialized
/// prediction extension.
pub enum MaterializedPredictionExtensionState<C, P, S> {
    /// DeepSeek-V3 compressed-latent caches in unit order.
    DeepSeekV3(Vec<C>),
    /// DeepSeek-V4 pooling-attention caches in unit order.
    DeepSeekV4(Vec<P>),
    /// DeepSeek-V4 DSpark pooling-attention caches in block order.
    DeepSeekV4Dspark(Vec<P>),
    /// Inkling prediction state.
    Inkling(S),
    /// Dense Qwen hybrid prediction state.
    QwenHybrid(S),
    /// Nemotron-H prediction state.
    NemotronH(S),
}

impl<C: Clone, P: Clone, S: Clone> Clone for MaterializedPredictionExtensionState<C, P, S> {
    fn clone(&self) -> Self {
        match self {
            Self::DeepSeekV3(state) => Self::DeepSeekV3(state.clone()),
            Self::DeepSeekV4(state) => Self::DeepSeekV4(state.clone()),
            Self::DeepSeekV4Dspark(state) => Self::DeepSeekV4Dspark(state.clone()),
            Self::Inkling(state) => Self::Inkling(state.clone()),
            Self::QwenHybrid(state) => Self::QwenHybrid(state.clone()),
            Self::NemotronH(state) => Self::NemotronH(state.clone()),
        }
    }
}

impl<B, M> MaterializedPredictionExtension<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Returns the exact number of proposals exposed by this extension.
    pub fn depth(&self) -> usize {
        match self {
            Self::DeepSeekV3 { units } => units.len(),
            Self::DeepSeekV4 { units, .. } => units.len(),
            Self::DeepSeekV4Dspark { strategy, .. } => strategy.proposal_capacity(),
            Self::Inkling { depth, .. } => *depth,
            Self::QwenHybrid { units, .. } => units.len(),
            Self::NemotronH { groups, .. } => groups.len(),
        }
    }

    /// Creates an independent mutable lane with exactly this extension's state
    /// geometry.
    pub fn new_state(
        &self,
    ) -> MaterializedPredictionExtensionState<M::SequentialState, M::PoolingState, M::ModelState>
    {
        match self {
            Self::DeepSeekV3 { units } => MaterializedPredictionExtensionState::DeepSeekV3(
                (0..units.len()).map(|_| M::sequential_state()).collect(),
            ),
            Self::DeepSeekV4 { state, .. } => {
                MaterializedPredictionExtensionState::DeepSeekV4(state.clone())
            }
            Self::DeepSeekV4Dspark { state, .. } => {
                MaterializedPredictionExtensionState::DeepSeekV4Dspark(state.clone())
            }
            Self::Inkling { state, .. } => {
                MaterializedPredictionExtensionState::Inkling(state.clone())
            }
            Self::QwenHybrid { state, .. } => {
                MaterializedPredictionExtensionState::QwenHybrid(state.clone())
            }
            Self::NemotronH { state, .. } => {
                MaterializedPredictionExtensionState::NemotronH(state.clone())
            }
        }
    }
}

impl<B> PreparedPredictionExtension<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Materializes this extension through family-blind backend mechanisms.
    pub fn materialize<M>(
        self,
        context: &mut M::Context<'_>,
    ) -> Result<MaterializedPredictionExtension<B, M>, M::Error>
    where
        M: PredictionExtensionMaterializer<B>,
        crate::deepseek::v3::Unit<B>: Parameterized<B::Tensor>,
        crate::deepseek::v4::Unit<B>: Parameterized<B::Tensor>,
        crate::deepseek::v4::DsparkStatic<B>: Parameterized<B::Tensor>,
        crate::inkling::MtpModel<B>: Parameterized<B::Tensor>,
        crate::qwen::hybrid::PredictionUnit<B>: Parameterized<B::Tensor>,
        crate::nemotron_h::PredictionUnit<B>: Parameterized<B::Tensor>,
    {
        match self {
            Self::DeepSeekV3 { layout, units } => Ok(MaterializedPredictionExtension::DeepSeekV3 {
                units: units
                    .into_iter()
                    .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                    .collect::<Result<_, _>>()?,
            }),
            Self::DeepSeekV4 {
                layout,
                units,
                state,
            } => Ok(MaterializedPredictionExtension::DeepSeekV4 {
                units: units
                    .into_iter()
                    .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                    .collect::<Result<_, _>>()?,
                state: state
                    .into_iter()
                    .map(|(ordinal, policy)| M::pooling_state(context, ordinal, policy))
                    .collect::<Result<_, _>>()?,
            }),
            Self::DeepSeekV4Dspark {
                layout,
                extension,
                units,
                state,
            } => {
                let (strategy, static_modules) = extension.into_parts();
                Ok(MaterializedPredictionExtension::DeepSeekV4Dspark {
                    strategy,
                    static_modules: M::materialize_module(context, static_modules, Some(&layout))?,
                    units: units
                        .into_iter()
                        .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                        .collect::<Result<_, _>>()?,
                    state: state
                        .into_iter()
                        .map(|(ordinal, policy)| M::pooling_state(context, ordinal, policy))
                        .collect::<Result<_, _>>()?,
                })
            }
            Self::Inkling { model, state } => {
                let depth = model.source.len();
                Ok(MaterializedPredictionExtension::Inkling {
                    model: M::materialize_module(context, model, None)?,
                    depth,
                    state: M::model_state(context, state)?,
                })
            }
            Self::QwenHybrid {
                layout,
                units,
                state,
            } => Ok(MaterializedPredictionExtension::QwenHybrid {
                units: units
                    .into_iter()
                    .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                    .collect::<Result<_, _>>()?,
                state: M::model_state(context, state)?,
            }),
            Self::NemotronH {
                layout,
                groups,
                state,
            } => Ok(MaterializedPredictionExtension::NemotronH {
                groups: groups
                    .into_iter()
                    .map(|units| {
                        units
                            .into_iter()
                            .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                            .collect::<Result<_, _>>()
                    })
                    .collect::<Result<_, _>>()?,
                state: M::model_state(context, state)?,
            }),
        }
    }
}

fn invalid(message: impl Into<String>) -> eredu_core::artifact::ArtifactError {
    eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(message.into())
}

fn tensor_rank(
    topology: ParallelRankTopology,
) -> Result<ParallelRankTopology, eredu_core::artifact::ArtifactError> {
    let tensor = ParallelTopology::new(topology.tensor_parallel_size(), 1, 1, 1)
        .map_err(|error| invalid(error.to_string()))?;
    ParallelRankTopology::new(tensor, topology.tensor_parallel_rank())
        .map_err(|error| invalid(error.to_string()))
}

pub(crate) fn validate_extension_contract(
    extension: &PredictionExtensionPlan,
) -> Result<(), eredu_core::artifact::ArtifactError> {
    let depth = match (extension.kind(), extension.complete_architecture().model()) {
        (PredictionExtensionKind::DeepSeekV3Mtp, SafetensorsModelConfig::DeepSeekV3(args)) => {
            usize::try_from(args.num_nextn_predict_layers)
        }
        (PredictionExtensionKind::DeepSeekV4Embedded, SafetensorsModelConfig::DeepSeekV4(args)) => {
            usize::try_from(args.num_nextn_predict_layers)
        }
        (PredictionExtensionKind::InklingMtp, SafetensorsModelConfig::Inkling(args)) => {
            usize::try_from(
                args.mtp_config
                    .as_ref()
                    .map_or(0, |mtp| mtp.num_nextn_predict_layers),
            )
        }
        (PredictionExtensionKind::QwenHybridMtp, SafetensorsModelConfig::QwenHybrid(args)) => {
            usize::try_from(args.text.mtp_num_hidden_layers)
        }
        (PredictionExtensionKind::NemotronHMtp, SafetensorsModelConfig::NemotronH(args)) => {
            usize::try_from(args.num_nextn_predict_layers)
        }
        _ => {
            return Err(invalid(
                "prediction extension identity does not match its admitted architecture",
            ));
        }
    }
    .map_err(|_| invalid("prediction extension depth exceeds usize"))?;
    if depth == 0 || depth != extension.depth() {
        return Err(invalid(format!(
            "prediction extension depth {} differs from admitted architecture depth {depth}",
            extension.depth()
        )));
    }
    Ok(())
}

/// Returns the capability estimate of the complete architecture that owns this extension.
pub fn prediction_extension_capability(
    extension: &PredictionExtensionPlan,
) -> Result<crate::capability::CapabilityEstimate, eredu_core::artifact::ArtifactError> {
    validate_extension_contract(extension)?;
    match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => crate::capability::deepseek_v3(args),
        SafetensorsModelConfig::DeepSeekV4(args) => crate::capability::deepseek_v4(args),
        SafetensorsModelConfig::Inkling(args) => crate::capability::inkling(args),
        SafetensorsModelConfig::QwenHybrid(args) => crate::capability::qwen_hybrid(args),
        SafetensorsModelConfig::NemotronH(args) => crate::capability::nemotron_h(args),
        _ => return Err(invalid("prediction extension has no capability estimate")),
    }
    .map_err(|error| invalid(error.to_string()))
}

fn prediction_topology(
    extension: &PredictionExtensionPlan,
    topology: ParallelRankTopology,
) -> Result<ParallelRankTopology, eredu_core::artifact::ArtifactError> {
    validate_partitioned_prediction_extension(extension, topology)?;
    tensor_rank(topology)
}

/// Validates topology restrictions owned by an excluded prediction extension.
///
/// Composition calls this before opening payload sources so an unsupported
/// extension cannot fall through to backend family policy or fail after I/O.
pub fn validate_partitioned_prediction_extension(
    extension: &PredictionExtensionPlan,
    topology: ParallelRankTopology,
) -> Result<(), eredu_core::artifact::ArtifactError> {
    validate_extension_contract(extension)?;
    if extension.kind() == PredictionExtensionKind::NemotronHMtp
        && (topology.pipeline_parallel_size() != 1 || topology.expert_parallel_size() != 1)
    {
        return Err(invalid(
            "Nemotron-H prediction extension requires pipeline=1 and expert=1",
        ));
    }
    Ok(())
}

/// Prepares an extension for one already admitted partitioned rank.
pub fn prepare_partitioned_prediction_extension<B, R, Q>(
    extension: &PredictionExtensionPlan,
    selected: &crate::partitioned_execution::SelectedPartitionedAdmission<R, Q>,
    store: &dyn CheckpointSource,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    prepare(
        extension,
        selected.requirements().topology(),
        store,
        source_context,
        execution_context,
    )
}

/// Prepares an extension for a single-rank replicated target.
pub fn prepare_replicated_prediction_extension<B>(
    extension: &PredictionExtensionPlan,
    store: &dyn CheckpointSource,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    let topology = ParallelTopology::new(1, 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, 0))
        .map_err(|error| invalid(error.to_string()))?;
    prepare(
        extension,
        topology,
        store,
        source_context,
        execution_context,
    )
}

fn prepare<B>(
    extension: &PredictionExtensionPlan,
    topology: ParallelRankTopology,
    store: &dyn CheckpointSource,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    let tensor_rank = prediction_topology(extension, topology)?;
    match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            let parameters = crate::deepseek::parallel::v3_parameter_description(args)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::deepseek::parallel::v3_local_geometry(args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source = crate::deepseek::v3::Model::<B>::new(args.clone(), source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v3::Model::<B>::new_parallel(
                args.clone(),
                geometry,
                execution_context,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("DeepSeek-V3 target count exceeds usize"))?;
            let mut units = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let ordinal = target + depth;
                let source_unit = source
                    .construct_unit(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let local_unit = local
                    .construct_unit(depth + 1, 0, execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let recipes = crate::deepseek::v3_unit_recipes(store, args, ordinal, true)
                    .map_err(invalid)?;
                units.push(PreparedPredictionUnit::new(
                    source_unit,
                    local_unit,
                    recipes,
                ));
            }
            Ok(PreparedPredictionExtension::DeepSeekV3 { layout, units })
        }
        SafetensorsModelConfig::DeepSeekV4(args) => {
            let parameters = crate::deepseek::parallel::v4_parameter_description(args)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::deepseek::parallel::v4_local_geometry(args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let state_layout = crate::deepseek::v4::state_layout(geometry.args())
                .map_err(|error| invalid(error.to_string()))?;
            let source = crate::deepseek::v4::Model::<B>::new(args.clone(), source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v4::Model::<B>::new_parallel(
                args.clone(),
                geometry,
                execution_context,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("DeepSeek-V4 target count exceeds usize"))?;
            let mut units = Vec::with_capacity(extension.depth());
            let mut state = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let ordinal = target + depth;
                let source_unit = source
                    .construct_unit(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let local_unit = local
                    .construct_unit(depth + 1, 0, execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let expert =
                    crate::deepseek::v4_expert_recipes(store, args, ordinal).map_err(invalid)?;
                let recipes = BTreeMap::from([
                    (expert.target_gate_up, expert.gate_up),
                    (expert.target_down, expert.down),
                ]);
                let policy = state_layout.layer(ordinal).cloned().ok_or_else(|| {
                    invalid(format!(
                        "DeepSeek-V4 prediction depth {depth} has no state policy"
                    ))
                })?;
                units.push(PreparedPredictionUnit::new(
                    source_unit,
                    local_unit,
                    recipes,
                ));
                state.push((ordinal, policy));
            }
            if args.dspark.is_some() {
                let strategy = DsparkPredictionStrategy::from_args(args)?;
                let source_static =
                    source.static_modules().dspark.clone().ok_or_else(|| {
                        invalid("source DSpark model has no fused pinned modules")
                    })?;
                let local_static = local
                    .static_modules()
                    .dspark
                    .clone()
                    .ok_or_else(|| invalid("local DSpark model has no fused pinned modules"))?;
                Ok(PreparedPredictionExtension::DeepSeekV4Dspark {
                    layout,
                    extension: PreparedDsparkPredictionExtension {
                        strategy,
                        static_modules: PreparedPredictionUnit::new(
                            source_static,
                            local_static,
                            BTreeMap::new(),
                        ),
                    },
                    units,
                    state,
                })
            } else {
                Ok(PreparedPredictionExtension::DeepSeekV4 {
                    layout,
                    units,
                    state,
                })
            }
        }
        SafetensorsModelConfig::Inkling(args) => {
            let source = crate::inkling::MtpModel::<B>::new(args, source_context)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
            let local = crate::inkling::MtpModel::<B>::new(args, execution_context)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
            let recipes = crate::inkling::mtp_safetensors_recipes(args, store).map_err(invalid)?;
            let state = crate::inkling::mtp_state_layout(args)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no state layout"))?;
            Ok(PreparedPredictionExtension::Inkling {
                model: PreparedPredictionUnit::new(source, local, recipes),
                state,
            })
        }
        SafetensorsModelConfig::QwenHybrid(args) => {
            if args.text.is_moe() {
                return Err(invalid(
                    "Qwen hybrid routed prediction requires an extension expert provider",
                ));
            }
            let source_architecture = crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(
                args.clone(),
                source_context,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let description = source_architecture
                .parameter_description(source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &description,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::qwen::hybrid::conditional_local_geometry(args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.text.num_hidden_layers)
                .map_err(|_| invalid("Qwen hybrid target count exceeds usize"))?;
            let mut units = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let source = crate::qwen::hybrid::PredictionUnit::<B>::new(
                    &args.text,
                    depth,
                    source_context,
                )
                .map_err(|error| invalid(error.to_string()))?;
                let local_config = geometry.text().prediction(depth).ok_or_else(|| {
                    invalid(format!(
                        "Qwen hybrid prediction depth {depth} has no local geometry"
                    ))
                })?;
                let local = crate::qwen::hybrid::PredictionUnit::<B>::new(
                    local_config,
                    depth,
                    execution_context,
                )
                .map_err(|error| invalid(error.to_string()))?;
                let recipes = crate::qwen::hybrid::unit_recipes(store, &args.text, target + depth)
                    .map_err(invalid)?;
                units.push(PreparedPredictionUnit::new(source, local, recipes));
            }
            let state = geometry
                .state_layout()
                .slice(target..target + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            Ok(PreparedPredictionExtension::QwenHybrid {
                layout,
                units,
                state,
            })
        }
        SafetensorsModelConfig::NemotronH(args) => {
            let source_architecture =
                crate::nemotron_h::LayeredModel::<B>::new(args.clone(), source_context)
                    .map_err(|error| invalid(error.to_string()))?;
            let description = source_architecture
                .parameter_description(source_context)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &description,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::nemotron_h::local_geometry(args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let policies = args
                .mtp_policies()
                .map_err(|error| invalid(error.to_string()))?;
            let pattern = policies
                .len()
                .checked_div(extension.depth())
                .filter(|pattern| *pattern > 0)
                .ok_or_else(|| invalid("Nemotron-H MTP pattern is empty"))?;
            let mut groups = Vec::with_capacity(extension.depth());
            for prediction in 0..extension.depth() {
                let mut units = Vec::with_capacity(pattern);
                for relative in 0..pattern {
                    let physical = prediction * pattern + relative;
                    let source = crate::nemotron_h::PredictionUnit::<B>::new(
                        args,
                        prediction,
                        relative,
                        source_context,
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    let local_geometry =
                        geometry.prediction_unit(physical).copied().ok_or_else(|| {
                            invalid(format!(
                                "Nemotron-H prediction unit {physical} has no local geometry"
                            ))
                        })?;
                    let local = crate::nemotron_h::PredictionUnit::<B>::new_with_geometry(
                        args,
                        prediction,
                        relative,
                        policies[physical],
                        local_geometry,
                        execution_context,
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    let recipes = crate::nemotron_h::unit_recipes(
                        store,
                        args,
                        prediction + 1,
                        relative,
                        true,
                    )
                    .map_err(invalid)?;
                    units.push(PreparedPredictionUnit::new(source, local, recipes));
                }
                groups.push(units);
            }
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("Nemotron-H target depth exceeds usize"))?;
            let state = geometry
                .state_layout()
                .slice(target..target + policies.len())
                .map_err(|error| invalid(error.to_string()))?;
            Ok(PreparedPredictionExtension::NemotronH {
                layout,
                groups,
                state,
            })
        }
        _ => Err(invalid(
            "selected prediction extension has no neutral preparation",
        )),
    }
}

#[cfg(test)]
mod speculative_contract_tests {
    use super::*;
    use eredu_core::ParallelTopology;

    fn id(value: &str) -> SpeculativeIdentity {
        SpeculativeIdentity::new(value).unwrap()
    }

    fn request(
        topology: ParallelRankTopology,
        capacity: usize,
    ) -> EmbeddedSpeculativeContractRequest {
        EmbeddedSpeculativeContractRequest::new(
            id("target-architecture-v1"),
            id("artifact-sha256-v1"),
            id("safetensors-layout-v1"),
            topology,
            id("processor-contract-v1"),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(7).unwrap(),
            NonZeroUsize::new(capacity).unwrap(),
        )
    }

    fn replicated() -> ParallelRankTopology {
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap()
    }

    fn distributed_rank() -> ParallelRankTopology {
        ParallelRankTopology::new(ParallelTopology::new(2, 2, 1, 1).unwrap(), 3).unwrap()
    }

    fn sequential(
        family: &'static str,
        state_prefix: &str,
        state_count: usize,
        mechanisms: Vec<SpeculativeMechanism>,
        shape: Vec<usize>,
    ) -> EmbeddedFamilyContract {
        EmbeddedFamilyContract {
            family,
            class: SpeculativeStrategyClass::EmbeddedSequential,
            architecture_capacity: NonZeroUsize::new(3).unwrap(),
            capture: EmbeddedCaptureContract::Single {
                path: "target.final_hidden".into(),
                shape,
                observation: "prediction.target_capture".into(),
            },
            state_components: indexed_state(state_prefix, state_count),
            additional_mechanisms: mechanisms,
            strategy_detail: format!("depth=3;state={state_count}"),
        }
    }

    #[test]
    fn sequential_family_contracts_preserve_capacity_state_and_exact_mechanisms() {
        let grouped = SpeculativeMechanism::GroupedNeuralOperations;
        let hyper = SpeculativeMechanism::HyperNeuralOperations;
        let cases = [
            sequential(
                "deepseek-v3-mtp",
                "prediction.compressed_latent",
                3,
                vec![grouped],
                vec![2, 7, 16],
            ),
            sequential(
                "deepseek-v4-mtp",
                "prediction.pooling",
                3,
                vec![grouped, hyper],
                vec![2, 7, 4, 16],
            ),
            sequential(
                "inkling-mtp",
                "prediction.inkling",
                3,
                vec![grouped],
                vec![2, 7, 24],
            ),
            sequential(
                "qwen-hybrid-mtp",
                "prediction.qwen_hybrid",
                3,
                vec![],
                vec![2, 7, 32],
            ),
            sequential(
                "nemotron-h-mtp",
                "prediction.nemotron_h",
                6,
                vec![grouped],
                vec![2, 7, 40],
            ),
        ];

        for family in cases {
            let expected_state = family.state_components.len();
            let expected_mechanisms = family.additional_mechanisms.clone();
            let contract =
                finish_embedded_speculative_contract(family, request(replicated(), 2)).unwrap();
            assert_eq!(
                contract.requirements().strategy().class(),
                SpeculativeStrategyClass::EmbeddedSequential
            );
            assert_eq!(
                contract.requirements().strategy().proposal_capacity().get(),
                2
            );
            assert_eq!(
                contract.requirements().state().state_components().len(),
                expected_state
            );
            for mechanism in expected_mechanisms {
                assert!(contract
                    .requirements()
                    .mechanisms()
                    .mechanisms()
                    .contains(&mechanism));
            }
            assert!(!contract
                .requirements()
                .mechanisms()
                .mechanisms()
                .contains(&SpeculativeMechanism::Communication));
            assert_eq!(contract.target_capture().entries().len(), 1);
        }
    }

    #[test]
    fn dspark_contract_preserves_ordered_intermediate_layers_and_fused_capacity() {
        let family = EmbeddedFamilyContract {
            family: "deepseek-v4-dspark",
            class: SpeculativeStrategyClass::EmbeddedFused,
            architecture_capacity: NonZeroUsize::new(8).unwrap(),
            capture: EmbeddedCaptureContract::Dspark {
                layers: vec![5, 1, 9],
                batch: 2,
                sequence: 7,
                hidden: 16,
            },
            state_components: indexed_state("prediction.dspark.pooling", 2),
            additional_mechanisms: vec![
                SpeculativeMechanism::GroupedNeuralOperations,
                SpeculativeMechanism::HyperNeuralOperations,
            ],
            strategy_detail: "blocks=2;block_size=8;noise=127;markov=4;layers=[5, 1, 9];hidden=16"
                .into(),
        };
        let reordered = EmbeddedFamilyContract {
            capture: EmbeddedCaptureContract::Dspark {
                layers: vec![1, 5, 9],
                batch: 2,
                sequence: 7,
                hidden: 16,
            },
            ..family.clone()
        };
        let contract =
            finish_embedded_speculative_contract(family, request(distributed_rank(), 4)).unwrap();
        let reordered =
            finish_embedded_speculative_contract(reordered, request(distributed_rank(), 4))
                .unwrap();

        assert_eq!(
            contract.requirements().strategy().class(),
            SpeculativeStrategyClass::EmbeddedFused
        );
        assert_eq!(
            contract.requirements().strategy().proposal_capacity().get(),
            4
        );
        assert_eq!(
            contract
                .target_capture()
                .entries()
                .iter()
                .map(|entry| entry.path().as_str())
                .collect::<Vec<_>>(),
            ["layers.5.output", "layers.1.output", "layers.9.output"]
        );
        assert!(contract
            .target_capture()
            .entries()
            .iter()
            .all(|entry| entry.shape() == [2, 7, 16]));
        assert!(contract
            .target_capture()
            .instantiate(vec![vec![1, 6, 16], vec![1, 6, 16], vec![1, 6, 16]])
            .is_ok());
        assert_eq!(
            contract
                .target_capture()
                .instantiate(vec![vec![1, 8, 16], vec![1, 8, 16], vec![1, 8, 16]])
                .unwrap_err(),
            eredu_runtime::SpeculativeCaptureError::ShapeMismatch
        );
        assert_ne!(contract.target_capture(), reordered.target_capture());
        assert!(contract
            .requirements()
            .mechanisms()
            .mechanisms()
            .contains(&SpeculativeMechanism::Communication));
        assert_eq!(contract.requirements().state().rank(), 3);
    }

    #[test]
    fn capacity_and_construction_identity_fail_closed_before_selection() {
        let family = sequential(
            "qwen-hybrid-mtp",
            "prediction.qwen_hybrid",
            3,
            vec![],
            vec![2, 7, 32],
        );
        let error = finish_embedded_speculative_contract(family.clone(), request(replicated(), 4))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds architecture capacity 3"));

        let baseline =
            finish_embedded_speculative_contract(family.clone(), request(replicated(), 2)).unwrap();
        let changed_processor = EmbeddedSpeculativeContractRequest::new(
            id("target-architecture-v1"),
            id("artifact-sha256-v1"),
            id("safetensors-layout-v1"),
            replicated(),
            id("different-processor"),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(7).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        let changed = finish_embedded_speculative_contract(family, changed_processor).unwrap();
        assert_ne!(baseline.requirements(), changed.requirements());
    }
}
