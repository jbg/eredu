//! Prepared neutral construction for embedded prediction extensions.

use eredu_runtime::speculative::validate_selected_capture_shapes;
use eredu_runtime::speculative::{materialize_prediction_module, materialize_prediction_units};
use eredu_runtime::speculative::{
    PredictionModule, PredictionPoolingState, SequentialPredictionCache,
};

mod selection;
pub use selection::SelectedPredictionParameters;

mod bindings;
use bindings::{PredictionBindingIndex, PreparedPredictionBindings};
mod qwen4_exp;
pub use eredu_runtime::speculative::{PredictionConstructionError, PreparedRoutedPrediction};
pub(crate) use qwen4_exp::speculative_contract as qwen4_speculative_contract;
pub use qwen4_exp::{
    MaterializedQwen4Prediction, PreparedQwen4PredictionWeights, Qwen4PredictionTarget,
};

use std::num::NonZeroUsize;

pub(crate) mod invocation;
mod placement;
pub(crate) mod residency;
mod resources;
pub use eredu_runtime::speculative::{
    MaterializedPredictionExecutor, ModelPredictionStateProfile, PoolingPredictionStateProfile,
    PredictionStateLifecycle, SequentialPredictionStateProfile,
};
pub use eredu_runtime::speculative::{
    PredictionExtensionMaterializer, PredictionInvocation, PredictionModuleRole,
    PredictionModuleVisitor, PredictionOperationInvoker, PreparedPredictionUnit,
};
use invocation::prediction_invocation;
pub(crate) use placement::PredictionPlacementSlot;
pub use placement::PreparedPredictionPlacement;

use eredu_core::{cache::LayerCachePolicy, ParallelRankTopology, ParallelTopology};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::{state::PortableTextState, LocalModelLayout, StateLayout};
use eredu_runtime::{
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion, RuntimeStateComponents,
};
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

/// Simultaneous weight ownership within an embedded prediction invocation.
fn prepare_prediction_unit<T: 'static, M: Parameterized<T>>(
    source: M,
    local: M,
    bindings: PreparedPredictionBindings,
) -> Result<PreparedPredictionUnit<M>, eredu_core::artifact::ArtifactError> {
    PreparedPredictionUnit::new(source, local, bindings.tasks, bindings.role)
        .map_err(|error| invalid(error.to_string()))
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
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration used by the same placement compiler.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v3::Unit<B>>>,
        /// Exact prediction-only local state geometry retained before binding.
        state: StateLayout,
    },
    /// DeepSeek-V4 sequential MTP units and their immutable cache policies.
    DeepSeekV4 {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration used by the same placement compiler.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>,
        /// Ordered rank-local cache policy for every prediction unit.
        state: Vec<(usize, LayerCachePolicy)>,
        /// Exact local layout preserving ordinary state frontier offsets.
        state_layout: StateLayout,
    },
    /// DeepSeek-V4 fused DSpark blocks, pinned modules, and immutable cache policies.
    DeepSeekV4Dspark {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration used by the same placement compiler.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Fused strategy and its checkpoint-global/rank-local pinned modules.
        extension: PreparedDsparkPredictionExtension<B>,
        /// Ordered checkpoint-global/rank-local ordinary DSpark block pairs.
        units: Vec<PreparedPredictionUnit<crate::deepseek::v4::Unit<B>>>,
        /// Ordered rank-local cache policy for every DSpark block.
        state: Vec<(usize, LayerCachePolicy)>,
        /// Exact local layout preserving ordinary state frontier offsets.
        state_layout: StateLayout,
    },
    /// Inkling sequential MTP units and optional shared normalization.
    Inkling {
        /// Exact layout used to bind replicated depths and shared target shards.
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration retaining static prediction and target ownership.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Ordered checkpoint-global/executable depth pairs.
        units: Vec<PreparedPredictionUnit<crate::inkling::MtpDepth<B>>>,
        /// One canonical chain normalization, when configured.
        shared: Option<PreparedPredictionUnit<crate::inkling::MtpShared<B>>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
    /// Dense Qwen hybrid MTP units.
    QwenHybrid {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration used by the same placement compiler.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Ordered checkpoint-global/rank-local unit pairs.
        units: Vec<PreparedPredictionUnit<crate::qwen::hybrid::PredictionUnit<B>>>,
        /// One canonical shared fusion/normalization module per replica.
        shared: PreparedPredictionUnit<crate::qwen::hybrid::PredictionShared<B>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
    /// Nemotron-H patterned MTP groups.
    NemotronH {
        /// Exact tensor-placement layout used to lower global bindings.
        layout: std::sync::Arc<LocalModelLayout>,
        /// Global declaration used by the same placement compiler.
        parameters: std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>,
        /// Prediction-step groups in architecture execution order.
        groups: Vec<Vec<PreparedPredictionUnit<crate::nemotron_h::PredictionUnit<B>>>>,
        /// Exact prediction-only state layout.
        state: StateLayout,
    },
}

pub use eredu_runtime::PredictionModelState;

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
        units: Vec<PredictionModule<B, M, crate::deepseek::v3::Unit<B>>>,
    },
    /// DeepSeek-V4 sequential MTP units.
    DeepSeekV4 {
        /// Units in proposal order.
        units: Vec<PredictionModule<B, M, crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in unit order.
        state: Vec<PredictionPoolingState<B, M>>,
    },
    /// DeepSeek-V4 fused DSpark execution.
    DeepSeekV4Dspark {
        /// Exact fused strategy retained from configuration.
        strategy: DsparkPredictionStrategy,
        /// Checkpoint-global fused projection modules.
        static_modules: PredictionModule<B, M, crate::deepseek::v4::DsparkStatic<B>>,
        /// Physical blocks in fused execution order.
        units: Vec<PredictionModule<B, M, crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in block order.
        state: Vec<PredictionPoolingState<B, M>>,
    },
    /// Inkling sequential prediction units.
    Inkling {
        /// Units in proposal order.
        units: Vec<PredictionModule<B, M, crate::inkling::MtpDepth<B>>>,
        /// Optional canonical chain normalization.
        shared: Option<PredictionModule<B, M, crate::inkling::MtpShared<B>>>,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
    /// Dense Qwen hybrid MTP units.
    QwenHybrid {
        /// Units in proposal order.
        units: Vec<PredictionModule<B, M, crate::qwen::hybrid::PredictionUnit<B>>>,
        /// Canonical shared fusion/normalization.
        shared: PredictionModule<B, M, crate::qwen::hybrid::PredictionShared<B>>,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
    /// Nemotron-H patterned MTP groups.
    NemotronH {
        /// Groups in proposal order and units in physical execution order.
        groups: Vec<Vec<PredictionModule<B, M, crate::nemotron_h::PredictionUnit<B>>>>,
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
    pub units: Vec<PredictionModule<B, M, crate::deepseek::v3::Unit<B>>>,
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
        units: Vec<PredictionModule<B, M, crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in unit order.
        state: Vec<PredictionPoolingState<B, M>>,
    },
    /// Fused DSpark modules and their cache prototypes.
    Dspark {
        /// Exact fused proposal strategy.
        strategy: DsparkPredictionStrategy,
        /// Checkpoint-global fused projection modules.
        static_modules: PredictionModule<B, M, crate::deepseek::v4::DsparkStatic<B>>,
        /// Physical blocks in fused execution order.
        units: Vec<PredictionModule<B, M, crate::deepseek::v4::Unit<B>>>,
        /// Immutable lane-state prototypes in block order.
        state: Vec<PredictionPoolingState<B, M>>,
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
    /// Units in proposal order.
    pub units: Vec<PredictionModule<B, M, crate::inkling::MtpDepth<B>>>,
    /// Optional canonical chain normalization.
    pub shared: Option<PredictionModule<B, M, crate::inkling::MtpShared<B>>>,
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
    pub units: Vec<PredictionModule<B, M, crate::qwen::hybrid::PredictionUnit<B>>>,
    /// Canonical shared fusion/normalization, visited once for parameter control.
    pub shared: PredictionModule<B, M, crate::qwen::hybrid::PredictionShared<B>>,
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
    pub groups: Vec<Vec<PredictionModule<B, M, crate::nemotron_h::PredictionUnit<B>>>>,
    /// Immutable prediction-state prototype.
    pub state: M::ModelState,
}

impl<B, M> TryFrom<MaterializedPredictionExtension<B, M>> for MaterializedDeepSeekV3Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type Error = MaterializedPredictionExtension<B, M>;

    fn try_from(extension: MaterializedPredictionExtension<B, M>) -> Result<Self, Self::Error> {
        match extension {
            MaterializedPredictionExtension::DeepSeekV3 { units } => {
                Ok(MaterializedDeepSeekV3Prediction { units })
            }
            extension => Err(extension),
        }
    }
}

impl<B, M> TryFrom<MaterializedPredictionExtension<B, M>> for MaterializedDeepSeekV4Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type Error = MaterializedPredictionExtension<B, M>;

    fn try_from(extension: MaterializedPredictionExtension<B, M>) -> Result<Self, Self::Error> {
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
            extension => Err(extension),
        }
    }
}

impl<B, M> TryFrom<MaterializedPredictionExtension<B, M>> for MaterializedInklingPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type Error = MaterializedPredictionExtension<B, M>;

    fn try_from(extension: MaterializedPredictionExtension<B, M>) -> Result<Self, Self::Error> {
        match extension {
            MaterializedPredictionExtension::Inkling {
                units,
                shared,
                state,
            } => Ok(MaterializedInklingPrediction {
                units,
                shared,
                state,
            }),
            extension => Err(extension),
        }
    }
}

impl<B, M> TryFrom<MaterializedPredictionExtension<B, M>> for MaterializedQwenHybridPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type Error = MaterializedPredictionExtension<B, M>;

    fn try_from(extension: MaterializedPredictionExtension<B, M>) -> Result<Self, Self::Error> {
        match extension {
            MaterializedPredictionExtension::QwenHybrid {
                units,
                shared,
                state,
            } => Ok(MaterializedQwenHybridPrediction {
                units,
                shared,
                state,
            }),
            extension => Err(extension),
        }
    }
}

impl<B, M> TryFrom<MaterializedPredictionExtension<B, M>> for MaterializedNemotronHPrediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type Error = MaterializedPredictionExtension<B, M>;

    fn try_from(extension: MaterializedPredictionExtension<B, M>) -> Result<Self, Self::Error> {
        match extension {
            MaterializedPredictionExtension::NemotronH { groups, state } => {
                Ok(MaterializedNemotronHPrediction { groups, state })
            }
            extension => Err(extension),
        }
    }
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
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
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
            Some(parallel) => self
                .kernel_mut()
                .embed_prediction_parallel(tokens, parallel, context),
            None => self.kernel_mut().embed_prediction(tokens, context),
        }
    }

    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.kernel_mut().project_prediction_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
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
                &mut self.kernel_mut().static_modules_mut().embeddings,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => self
                .kernel_mut()
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
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.kernel_mut().static_modules_mut().project_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }
}

struct V3PredictionOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut PredictionModule<B, M, crate::deepseek::v3::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut SequentialPredictionCache<B, M>,
    depth: usize,
    pass: eredu_runtime::ExpertPass,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v3::Model<B>, B, S>
    for V3PredictionOperation<'_, '_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v3::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v3::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.unit, context, |unit| {
            let outcome = (|| {
                if let Some(observer) = self.observer {
                    return architecture
                        .kernel_mut()
                        .pipeline_forward_prediction_observed(
                            unit,
                            self.depth,
                            self.hidden,
                            self.tokens,
                            self.cache,
                            self.pass,
                            parallel,
                            context,
                            observer,
                        );
                }
                match parallel {
                    Some(parallel) => architecture
                        .kernel_mut()
                        .pipeline_forward_prediction_neutral_parallel(
                            unit,
                            self.hidden,
                            self.tokens,
                            self.cache,
                            parallel,
                            context,
                        ),
                    None => architecture.kernel_mut().pipeline_forward_prediction(
                        unit,
                        self.hidden,
                        self.tokens,
                        self.cache,
                        context,
                    ),
                }
            })();
            prediction_invocation(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |output| [&output.logits, &output.hidden, &output.tokens],
            )
        })
    }
}

struct V4PredictionOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut PredictionModule<B, M, crate::deepseek::v4::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut PredictionPoolingState<B, M>,
    depth: usize,
    pass: eredu_runtime::ExpertPass,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4PredictionOperation<'_, '_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    PredictionPoolingState<B, M>: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.unit, context, |unit| {
            let outcome = (|| {
                let hidden = architecture
                    .kernel_mut()
                    .begin_partition_prediction_hidden(self.hidden, context)?;
                let output = if let Some(observer) = self.observer {
                    architecture
                        .kernel_mut()
                        .pipeline_forward_prediction_observed(
                            unit,
                            self.depth,
                            &hidden,
                            self.tokens,
                            self.cache,
                            self.pass,
                            parallel,
                            context,
                            observer,
                        )
                } else {
                    match parallel {
                        Some(parallel) => architecture
                            .kernel_mut()
                            .pipeline_forward_prediction_neutral_parallel(
                                unit,
                                &hidden,
                                self.tokens,
                                self.cache,
                                parallel,
                                context,
                            ),
                        None => architecture.kernel_mut().pipeline_forward_prediction(
                            unit,
                            &hidden,
                            self.tokens,
                            self.cache,
                            context,
                        ),
                    }
                }?;
                architecture
                    .kernel_mut()
                    .finish_partition_prediction_output(output, context)
            })();
            prediction_invocation(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |output| [&output.logits, &output.hidden, &output.tokens],
            )
        })
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
        unit: &mut crate::inkling::MtpDepth<B>,
        shared: Option<&mut crate::inkling::MtpShared<B>>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;
}

impl<B> InklingPredictionTarget<B> for crate::inkling::InklingKernel<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::inkling::MtpDepth<B>,
        shared: Option<&mut crate::inkling::MtpShared<B>>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        let state = state
            .prediction_layers_mut()
            .get_mut(depth)
            .ok_or_else(|| {
                eredu_nn::Error::backend("Inkling MTP state has no selected prediction depth")
            })?;
        let embeddings = match parallel {
            Some(parallel) => self.mtp_token_embeddings_parallel(tokens, parallel, context)?,
            None => self.mtp_token_embeddings(tokens, context)?,
        };
        let path = observer
            .as_ref()
            .map(|_| format!("model.mtp.layers.{depth}"));
        let mut instrumentation = match observer {
            Some(observer) => crate::decoder::ComponentInstrumentation::new(
                path.as_deref().expect("observed prediction"),
                observer,
            ),
            None => crate::decoder::ComponentInstrumentation::disabled(),
        };
        let output = unit.forward_step_instrumented(
            hidden,
            &embeddings,
            tokens,
            state,
            shared.map(|shared| &mut shared.chain_norm),
            context,
            &mut instrumentation,
        )?;
        let logits = instrumentation.with_scope("prediction.readout", |instrumentation| {
            self.project_mtp_logits_instrumented(&output.hidden, parallel, context, instrumentation)
        })?;
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: output.hidden,
            tokens: output.tokens,
        })
    }
}

impl<B, D, K, R> InklingPredictionTarget<B> for eredu_runtime::ComponentAdapter<D, K, R>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    D: std::borrow::Borrow<eredu_runtime::ComponentDefinition>,
    K: InklingPredictionTarget<B>,
{
    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::inkling::MtpDepth<B>,
        shared: Option<&mut crate::inkling::MtpShared<B>>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        self.kernel_mut().prediction_step::<M>(
            unit, shared, hidden, tokens, depth, state, observer, parallel, context,
        )
    }
}

struct InklingPredictionOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut PredictionModule<B, M, crate::inkling::MtpDepth<B>>,
    shared: Option<&'a mut PredictionModule<B, M, crate::inkling::MtpShared<B>>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for InklingPredictionOperation<'_, '_, B, M>
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
        M::invoke_module_with_shared(self.unit, self.shared, context, |unit, shared| {
            let outcome = architecture.prediction_step::<M>(
                unit,
                shared,
                self.hidden,
                self.tokens,
                self.depth,
                self.state,
                self.observer,
                parallel,
                context,
            );
            prediction_invocation(
                outcome,
                self.state
                    .prediction_layers_mut()
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |output| [&output.logits, &output.capture, &output.tokens],
            )
        })
    }
}

pub(crate) trait QwenHybridPredictionTarget<B>
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
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error>;

    fn prediction_step<M>(
        &mut self,
        unit: &mut crate::qwen::hybrid::PredictionUnit<B>,
        shared: &mut crate::qwen::hybrid::PredictionShared<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        let embedded = self.embed_prediction(tokens, parallel, context)?;
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
        let path = observer.as_ref().map(|_| format!("mtp.layers.{depth}"));
        let hidden = match (parallel, observer.as_deref_mut()) {
            (Some(parallel), Some(observer)) => unit.forward_parallel_observed_with_provider(
                shared,
                path.as_deref().expect("observed prediction"),
                unit.observation_points(depth),
                hidden,
                &embedded,
                mask.as_ref(),
                layer,
                parallel,
                context,
                &mut eredu_runtime::ResidentExpertProvider,
                observer,
            )?,
            (None, Some(observer)) => unit.forward_observed_with_provider(
                shared,
                path.as_deref().expect("observed prediction"),
                unit.observation_points(depth),
                hidden,
                &embedded,
                mask.as_ref(),
                layer,
                context,
                &mut eredu_runtime::ResidentExpertProvider,
                observer,
            )?,
            (Some(parallel), None) => unit.forward_parallel(
                shared,
                hidden,
                &embedded,
                mask.as_ref(),
                layer,
                parallel,
                context,
                &mut eredu_runtime::ResidentExpertProvider,
            )?,
            (None, None) => {
                unit.forward(shared, hidden, &embedded, mask.as_ref(), layer, context)?
            }
        };
        let logits = match observer {
            Some(observer) => {
                let root = format!("{}.prediction.readout", path.expect("observed prediction"));
                self.project_prediction(
                    &hidden,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::new(&root, observer),
                )?
            }
            None => self.project_prediction(
                &hidden,
                parallel,
                context,
                &mut crate::decoder::ComponentInstrumentation::disabled(),
            )?,
        };
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: hidden,
            tokens: tokens.clone(),
        })
    }
}

impl<B> QwenHybridPredictionTarget<B> for crate::qwen::hybrid::LayeredModel<B>
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
            Some(parallel) => self
                .kernel_mut()
                .pipeline_embed_parallel(tokens, parallel, context),
            None => self.kernel_mut().begin_partition_embedding(tokens, context),
        }
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        crate::decoder::DecoderStaticResources::decoder_resources_mut(
            self.kernel_mut().static_modules_mut(),
        )
        .project(hidden, parallel, context, instrumentation)
    }
}

impl<B> QwenHybridPredictionTarget<B> for crate::qwen::hybrid::ConditionalLayeredModel<B>
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
            Some(parallel) => self
                .kernel_mut()
                .pipeline_embed_parallel(tokens, parallel, context),
            None => self
                .kernel_mut()
                .begin_partition_prediction_embedding(tokens, context),
        }
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.kernel_mut().project_prediction_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }
}

struct QwenHybridPredictionOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    unit: &'a mut PredictionModule<B, M, crate::qwen::hybrid::PredictionUnit<B>>,
    shared: &'a mut PredictionModule<B, M, crate::qwen::hybrid::PredictionShared<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for QwenHybridPredictionOperation<'_, '_, B, M>
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
        M::invoke_module_with_shared(self.unit, Some(self.shared), context, |unit, shared| {
            let outcome = architecture.prediction_step::<M>(
                unit,
                shared.expect("Qwen prediction declares a shared owner"),
                self.hidden,
                self.tokens,
                self.depth,
                self.state,
                self.observer,
                parallel,
                context,
            );
            prediction_invocation(
                outcome,
                self.state
                    .prediction_layers_mut()
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |output| [&output.logits, &output.capture, &output.tokens],
            )
        })
    }
}

struct NemotronHPredictionOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    units: &'a mut [PredictionModule<B, M, crate::nemotron_h::PredictionUnit<B>>],
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<A, B, S, M> eredu_runtime::PredictionTargetOperation<A, B, S>
    for NemotronHPredictionOperation<'_, '_, B, M>
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
        mut self,
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
            hidden = M::invoke_module(unit, context, |unit| {
                let expert_count = unit.expert_count();
                let outcome = match (parallel, self.observer.as_deref_mut()) {
                    (Some(parallel), Some(observer)) => unit
                        .forward_parallel_observed_with_provider(
                            &format!("model.mtp.layers.{}", state_start + relative),
                            expert_count,
                            &hidden,
                            &embedded,
                            mask.as_ref(),
                            state,
                            parallel,
                            context,
                            &mut eredu_runtime::ResidentExpertProvider,
                            observer,
                        ),
                    (None, Some(observer)) => unit.forward_observed_with_provider(
                        &format!("model.mtp.layers.{}", state_start + relative),
                        expert_count,
                        &hidden,
                        &embedded,
                        mask.as_ref(),
                        state,
                        context,
                        &mut eredu_runtime::ResidentExpertProvider,
                        observer,
                    ),
                    (Some(parallel), None) => unit.forward_parallel_with_provider(
                        &hidden,
                        &embedded,
                        mask.as_ref(),
                        state,
                        parallel,
                        context,
                        &mut eredu_runtime::ResidentExpertProvider,
                    ),
                    (None, None) => unit.forward(&hidden, &embedded, mask.as_ref(), state, context),
                };
                prediction_invocation(
                    outcome,
                    eredu_runtime::RuntimeLayerState::<B>::retained_values(state),
                    |output| [output],
                )
            })?;
        }
        let logits = match self.observer {
            Some(observer) => {
                let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
                let path = format!(
                    "model.mtp.layers.{}.prediction.readout",
                    state_start + self.units.len() - 1
                );
                architecture.project_prediction(
                    &hidden,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed),
                )?
            }
            None => architecture.project_prediction(
                &hidden,
                parallel,
                context,
                &mut crate::decoder::ComponentInstrumentation::disabled(),
            )?,
        };
        Ok(crate::speculative_execution::EmbeddedPredictionOutput {
            logits,
            capture: hidden,
            tokens: self.tokens.clone(),
        })
    }
}

impl<B, M> MaterializedPredictionExecutor<crate::deepseek::v3::Model<B>, B, M>
    for MaterializedDeepSeekV3Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    type LaneState = Vec<SequentialPredictionCache<B, M>>;
    type StateProfile = SequentialPredictionStateProfile;

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit_units::<crate::deepseek::v3::Unit<B>>(0, self.units.iter_mut())
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        (0..self.units.len())
            .map(|_| SequentialPredictionCache::<B, M>::new())
            .collect()
    }

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
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
        for (depth, (unit, cache)) in self.units.iter_mut().zip(lane).enumerate() {
            invoker.invoke(V3PredictionOperation::<B, M> {
                unit,
                hidden,
                tokens,
                cache,
                depth,
                pass: eredu_runtime::ExpertPass::Prefill,
                observer: observer.as_mut().map(|observer| {
                    &mut **observer
                        as &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>
                }),
            })?;
        }
        Ok(())
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
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
                depth: draft_index,
                pass: eredu_runtime::ExpertPass::Decode,
                observer,
            })
            .map(|output| (output.logits, output.hidden))
    }
}

impl<B, M> MaterializedPredictionExecutor<crate::deepseek::v4::Model<B>, B, M>
    for MaterializedDeepSeekV4Prediction<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    PredictionPoolingState<B, M>: 'static,
{
    type LaneState = Vec<PredictionPoolingState<B, M>>;
    type StateProfile = PoolingPredictionStateProfile;

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        let (offset, units) = match self {
            Self::Sequential { units, .. } => (0, units),
            Self::Dspark {
                static_modules,
                units,
                ..
            } => {
                visitor.visit::<crate::deepseek::v4::DsparkStatic<B>>(0, static_modules)?;
                (1, units)
            }
        };
        visitor.visit_units::<crate::deepseek::v4::Unit<B>>(offset, units.iter_mut())
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn depth(&self) -> usize {
        match self {
            Self::Sequential { units, .. } | Self::Dspark { units, .. } => units.len(),
        }
    }

    fn prefill_sequence_len(&self, target_sequence: usize) -> usize {
        match self {
            Self::Dspark { .. } => {
                eredu_runtime::prediction_resources::PredictionExecutionMode::Fused
                    .prefill_sequence_len(target_sequence)
            }
            Self::Sequential { .. } => {
                eredu_runtime::prediction_resources::PredictionExecutionMode::Sequential
                    .prefill_sequence_len(target_sequence)
            }
        }
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

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
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
                for (depth, (unit, cache)) in units.iter_mut().zip(lane).enumerate() {
                    invoker.invoke(V4PredictionOperation::<B, M> {
                        unit,
                        hidden,
                        tokens,
                        cache,
                        depth,
                        pass: eredu_runtime::ExpertPass::Prefill,
                        observer: observer.as_mut().map(|o| {
                            &mut **o
                                as &mut dyn eredu_runtime::ActivationObserver<
                                    B::Tensor,
                                    eredu_nn::Error,
                                >
                        }),
                    })?;
                }
                Ok(())
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
                    observer,
                })
            }
        }
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
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
                        depth: draft_index,
                        pass: eredu_runtime::ExpertPass::Decode,
                        observer,
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
        let mut proposal_state = lane.clone();
        self.fused_logits_observed::<S, I>(invoker, anchor, capacity, &mut proposal_state, None)
    }

    fn fused_logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        anchor: &B::Tensor,
        capacity: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
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
                invoker
                    .invoke(V4DsparkProposalOperation::<B, M> {
                        strategy,
                        static_modules,
                        units,
                        anchor,
                        capacity,
                        caches: lane,
                        observer,
                    })
                    .map(Some)
            }
            Self::Sequential { .. } => Ok(None),
        }
    }
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
    type StateProfile = ModelPredictionStateProfile;

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(shared) = self.shared.as_mut() {
            visitor.visit::<crate::inkling::MtpShared<B>>(0, shared)?;
        }
        visitor.visit_units::<crate::inkling::MtpDepth<B>>(1, self.units.iter_mut())
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if lane.prediction_layers_mut().len() != self.units.len() {
            return Err(I::invalid(
                "Inkling MTP state does not match prediction depths".into(),
            ));
        }
        self.units
            .iter_mut()
            .enumerate()
            .try_for_each(|(depth, unit)| {
                invoker
                    .invoke(InklingPredictionOperation::<B, M> {
                        unit,
                        shared: self.shared.as_mut(),
                        hidden,
                        tokens,
                        depth,
                        state: lane,
                        observer: observer.as_mut().map(|observer| {
                            &mut **observer
                                as &mut dyn eredu_runtime::ActivationObserver<
                                    B::Tensor,
                                    eredu_nn::Error,
                                >
                        }),
                    })
                    .map(|_| ())
            })
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if lane.prediction_layers_mut().len() != self.units.len() {
            return Err(I::invalid(
                "Inkling MTP state does not match prediction depths".into(),
            ));
        }
        if draft_index >= self.units.len() {
            return Err(I::invalid(format!(
                "prediction depth {draft_index} exceeds {} units",
                self.units.len()
            )));
        }
        invoker
            .invoke(InklingPredictionOperation::<B, M> {
                unit: &mut self.units[draft_index],
                shared: self.shared.as_mut(),
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
                observer,
            })
            .map(|output| (output.logits, output.capture))
    }
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
    type StateProfile = ModelPredictionStateProfile;

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit::<crate::qwen::hybrid::PredictionShared<B>>(0, &mut self.shared)?;
        visitor.visit_units::<crate::qwen::hybrid::PredictionUnit<B>>(1, self.units.iter_mut())
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
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
                        shared: &mut self.shared,
                        hidden,
                        tokens,
                        depth,
                        state: lane,
                        observer: observer.as_mut().map(|observer| {
                            &mut **observer
                                as &mut dyn eredu_runtime::ActivationObserver<
                                    B::Tensor,
                                    eredu_nn::Error,
                                >
                        }),
                    })
                    .map(|_| ())
            })
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
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
                shared: &mut self.shared,
                hidden,
                tokens: token,
                depth: draft_index,
                state: lane,
                observer,
            })
            .map(|output| (output.logits, output.capture))
    }
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
    type StateProfile = ModelPredictionStateProfile;

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit_units::<crate::nemotron_h::PredictionUnit<B>>(
            0,
            self.groups.iter_mut().flatten(),
        )
    }

    fn depth(&self) -> usize {
        self.groups.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        _target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
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
                        observer: observer.as_mut().map(|observer| {
                            &mut **observer
                                as &mut dyn eredu_runtime::ActivationObserver<
                                    B::Tensor,
                                    eredu_nn::Error,
                                >
                        }),
                    })
                    .map(|_| ())
            })
    }

    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
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
                observer,
            })
            .map(|output| (output.logits, output.capture))
    }
}

struct V4DsparkContextOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    strategy: &'a DsparkPredictionStrategy,
    static_modules: &'a mut PredictionModule<B, M, crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [PredictionModule<B, M, crate::deepseek::v4::Unit<B>>],
    captures: &'a B::Tensor,
    caches: &'a mut [PredictionPoolingState<B, M>],
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4DsparkContextOperation<'_, '_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    PredictionPoolingState<B, M>: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = ();

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.static_modules, context, |static_modules| {
            let outcome = architecture.kernel_mut().pipeline_prefill_dspark_extension_context_observed_with_modules::<_, invocation::MaterializedPredictionModules<B, M>>(
            self.strategy,
            static_modules,
            self.units,
            self.captures,
            self.caches,
            context,
            self.observer,
        );
            prediction_invocation(
                outcome,
                self.caches
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |_output| [],
            )
        })
    }
}

struct V4DsparkProposalOperation<'a, 'o, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    strategy: &'a DsparkPredictionStrategy,
    static_modules: &'a mut PredictionModule<B, M, crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [PredictionModule<B, M, crate::deepseek::v4::Unit<B>>],
    anchor: &'a B::Tensor,
    capacity: usize,
    caches: &'a mut [PredictionPoolingState<B, M>],
    observer: Option<&'o mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
}

impl<B, S, M> eredu_runtime::PredictionTargetOperation<crate::deepseek::v4::Model<B>, B, S>
    for V4DsparkProposalOperation<'_, '_, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    crate::deepseek::v4::Model<B>:
        eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    M: PredictionExtensionMaterializer<B>,
    PredictionPoolingState<B, M>: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = B::Tensor;

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.static_modules, context, |static_modules| {
            let outcome = architecture.kernel_mut().pipeline_dspark_extension_proposal_observed_with_modules::<_, invocation::MaterializedPredictionModules<B, M>>(
            self.strategy,
            static_modules,
            self.units,
            self.anchor,
            self.capacity,
            self.caches,
            parallel,
            context,
            self.observer,
        );
            prediction_invocation(
                outcome,
                self.caches
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |output| [output],
            )
        })
    }
}

mod pairing_sealed {
    pub trait Sealed {}
}

/// Architecture-owned proof that one materialized extension belongs to one
/// exact ordinary target architecture.
pub trait MaterializedPredictionTarget<B>:
    pairing_sealed::Sealed + eredu_runtime::speculative::PredictionTarget<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Exact target context reported when the materialized family does not match.
    const PAIRING_MISMATCH: &'static str;

    /// Consumes and validates the architecture-owned materialized extension.
    fn pair_prediction_extension<M>(
        extension: MaterializedPredictionExtension<B, M>,
    ) -> Result<Self::Extension<M>, eredu_core::artifact::ArtifactError>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;
}

macro_rules! prediction_pairing {
    (
        $backend:ident $(, $policy:ident)?; $materializer:ident;
        $kernel:ty => $target:ty;
        $extension:ty; $grouped:path;
        $mismatch:literal
    ) => {
        impl<$backend $(, $policy: 'static)?> pairing_sealed::Sealed for $target
        where
            $backend: BlockwiseAttentionBackend + DistributedNeuralBackend + $grouped + HyperNeuralBackend,
        {}

        impl<$backend $(, $policy: 'static)?> eredu_runtime::speculative::PredictionKernel<$backend, $target> for $kernel
        where
            $backend: BlockwiseAttentionBackend + DistributedNeuralBackend + $grouped + HyperNeuralBackend,
        {
            type Extension<$materializer> = $extension
            where
                $materializer: PredictionExtensionMaterializer<$backend>;
        }

        impl<$backend $(, $policy: 'static)?> MaterializedPredictionTarget<$backend> for $target
        where
            $backend: BlockwiseAttentionBackend + DistributedNeuralBackend + $grouped + HyperNeuralBackend,
        {
            const PAIRING_MISMATCH: &'static str = $mismatch;

            fn pair_prediction_extension<$materializer>(
                extension: MaterializedPredictionExtension<$backend, $materializer>,
            ) -> Result<Self::Extension<$materializer>, eredu_core::artifact::ArtifactError>
            where
                $materializer: PredictionExtensionMaterializer<$backend>,
            {
                Self::Extension::<$materializer>::try_from(extension)
                    .map_err(|_| invalid(Self::PAIRING_MISMATCH))
            }
        }
    };
}

prediction_pairing!(
    B; M;
    crate::deepseek::v3::V3Kernel<B> => crate::deepseek::v3::Model<B>;
    MaterializedDeepSeekV3Prediction<B, M>; GroupedNeuralBackend;
    "materialized prediction extension does not match the DeepSeek-V3 target"
);

prediction_pairing!(
    B; M;
    crate::deepseek::v4::V4Kernel<B> => crate::deepseek::v4::Model<B>;
    MaterializedDeepSeekV4Prediction<B, M>; GroupedNeuralBackend;
    "materialized prediction extension does not match the DeepSeek-V4 target"
);

prediction_pairing!(
    B; M;
    crate::nemotron_h::NemotronKernel<B> => crate::nemotron_h::PartitionedLayeredModel<B>;
    MaterializedNemotronHPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Nemotron-H target"
);

prediction_pairing!(
    B; M;
    crate::nemotron_h::CompleteNemotronKernel<B> => crate::nemotron_h::LayeredModel<B>;
    MaterializedNemotronHPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Nemotron-H target"
);

prediction_pairing!(
    B, P; M;
    crate::replicated_model::FixedReplicatedKernel<B, crate::replicated_text::NemotronHReplicated, P> => crate::replicated_model::FixedReplicatedModel<B, crate::replicated_text::NemotronHReplicated, P>;
    MaterializedNemotronHPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Nemotron-H target"
);

prediction_pairing!(
    B; M;
    crate::inkling::InklingKernel<B> => crate::inkling::LayeredModel<B>;
    MaterializedInklingPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Inkling target"
);

prediction_pairing!(
    B; M;
    crate::qwen::hybrid::ConditionalKernel<B> => crate::qwen::hybrid::ConditionalLayeredModel<B>;
    MaterializedQwenHybridPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the conditional Qwen target"
);

prediction_pairing!(
    B; M;
    crate::qwen::hybrid::QwenKernel<B> => crate::qwen::hybrid::LayeredModel<B>;
    MaterializedQwenHybridPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Qwen text target"
);

prediction_pairing!(
    B, P; M;
    crate::replicated_model::FixedReplicatedKernel<B, crate::replicated_text::QwenHybridReplicated, P> => crate::replicated_model::FixedReplicatedModel<B, crate::replicated_text::QwenHybridReplicated, P>;
    MaterializedQwenHybridPrediction<B, M>; eredu_nn::TensorParallelGroupedNeuralBackend;
    "materialized prediction extension does not match the Qwen text target"
);

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
            Self::Inkling { units, .. } => units.len(),
            Self::QwenHybrid { units, .. } => units.len(),
            Self::NemotronH { groups, .. } => groups.len(),
        }
    }

    /// Creates an independent mutable lane with exactly this extension's state
    /// geometry.
    pub fn new_state(
        &self,
    ) -> MaterializedPredictionExtensionState<
        SequentialPredictionCache<B, M>,
        PredictionPoolingState<B, M>,
        M::ModelState,
    > {
        match self {
            Self::DeepSeekV3 { units } => MaterializedPredictionExtensionState::DeepSeekV3(
                (0..units.len())
                    .map(|_| SequentialPredictionCache::<B, M>::new())
                    .collect(),
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
    /// Applies the retained execution policy to every physical prediction owner.
    /// Shared modules occur once here even when several logical depths use them.
    pub fn with_residency(mut self, residency: eredu_runtime::LayerWeightResidency) -> Self {
        match &mut self {
            Self::DeepSeekV3 { units, .. } => {
                for unit in units {
                    unit.set_residency(residency);
                }
            }
            Self::DeepSeekV4 { units, .. } => {
                for unit in units {
                    unit.set_residency(residency);
                }
            }
            Self::DeepSeekV4Dspark {
                extension, units, ..
            } => {
                extension.static_modules.set_residency(residency);
                for unit in units {
                    unit.set_residency(residency);
                }
            }
            Self::Inkling { units, shared, .. } => {
                for unit in units {
                    unit.set_residency(residency);
                }
                if let Some(shared) = shared {
                    shared.set_residency(residency);
                }
            }
            Self::QwenHybrid { units, shared, .. } => {
                for unit in units {
                    unit.set_residency(residency);
                }
                shared.set_residency(residency);
            }
            Self::NemotronH { groups, .. } => {
                for unit in groups.iter_mut().flatten() {
                    unit.set_residency(residency);
                }
            }
        }
        self
    }

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
        crate::inkling::MtpDepth<B>: Parameterized<B::Tensor>,
        crate::inkling::MtpShared<B>: Parameterized<B::Tensor>,
        crate::qwen::hybrid::PredictionUnit<B>: Parameterized<B::Tensor>,
        crate::nemotron_h::PredictionUnit<B>: Parameterized<B::Tensor>,
    {
        match self {
            Self::DeepSeekV3 { layout, units, .. } => {
                Ok(MaterializedPredictionExtension::DeepSeekV3 {
                    units: materialize_prediction_units::<B, M, _>(context, units, Some(&layout))?,
                })
            }
            Self::DeepSeekV4 {
                layout,
                units,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::DeepSeekV4 {
                units: materialize_prediction_units::<B, M, _>(context, units, Some(&layout))?,
                state: state
                    .into_iter()
                    .map(|(ordinal, policy)| {
                        PredictionPoolingState::<B, M>::resident_from_policy(ordinal, &policy)
                            .map_err(eredu_core::BackendFailure::from_error)
                            .map_err(M::Error::from)
                    })
                    .collect::<Result<_, _>>()?,
            }),
            Self::DeepSeekV4Dspark {
                layout,
                extension,
                units,
                state,
                ..
            } => {
                let (strategy, static_modules) = extension.into_parts();
                Ok(MaterializedPredictionExtension::DeepSeekV4Dspark {
                    strategy,
                    static_modules: materialize_prediction_module::<B, M, _>(
                        context,
                        static_modules,
                        Some(&layout),
                    )?,
                    units: materialize_prediction_units::<B, M, _>(context, units, Some(&layout))?,
                    state: state
                        .into_iter()
                        .map(|(ordinal, policy)| {
                            PredictionPoolingState::<B, M>::resident_from_policy(ordinal, &policy)
                                .map_err(eredu_core::BackendFailure::from_error)
                                .map_err(M::Error::from)
                        })
                        .collect::<Result<_, _>>()?,
                })
            }
            Self::Inkling {
                layout,
                units,
                shared,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::Inkling {
                shared: shared
                    .map(|shared| {
                        materialize_prediction_module::<B, M, _>(context, shared, Some(&layout))
                    })
                    .transpose()?,
                units: materialize_prediction_units::<B, M, _>(context, units, Some(&layout))?,
                state: M::ModelState::device(state).map_err(M::Error::from)?,
            }),
            Self::QwenHybrid {
                layout,
                units,
                shared,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::QwenHybrid {
                shared: materialize_prediction_module::<B, M, _>(context, shared, Some(&layout))?,
                units: materialize_prediction_units::<B, M, _>(context, units, Some(&layout))?,
                state: M::ModelState::device(state).map_err(M::Error::from)?,
            }),
            Self::NemotronH {
                layout,
                groups,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::NemotronH {
                groups: groups
                    .into_iter()
                    .map(|units| {
                        materialize_prediction_units::<B, M, _>(context, units, Some(&layout))
                    })
                    .collect::<Result<_, _>>()?,
                state: M::ModelState::device(state).map_err(M::Error::from)?,
            }),
        }
    }
}

fn invalid(message: impl Into<String>) -> eredu_core::artifact::ArtifactError {
    eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(message.into())
}

pub(crate) fn tensor_rank(
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
        SafetensorsModelConfig::NemotronH(_) => crate::capability::nemotron_h(
            extension
                .nemotron_declaration()
                .ok_or_else(|| invalid("prediction extension has no Nemotron declaration"))?,
        ),
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

/// Validates the excluded prediction extension before opening payload sources.
///
/// Prediction units use their tensor-rank layout and replicate across target
/// pipeline and expert coordinates. Target topology does not change the physical
/// prediction schedule or its state; the retained tasks account for these replicas.
pub fn validate_partitioned_prediction_extension(
    extension: &PredictionExtensionPlan,
    _topology: ParallelRankTopology,
) -> Result<(), eredu_core::artifact::ArtifactError> {
    validate_extension_contract(extension)
}

pub(crate) fn prepare<B>(
    selected: &SelectedPredictionParameters,
    topology: ParallelRankTopology,
    source_context: &<B::Tensor as Tensor>::Context,
    execution_context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedPredictionExtension<B>, eredu_core::artifact::ArtifactError>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    let extension = &selected.extension;
    let tasks = selected.tasks();
    let tensor_rank = prediction_topology(extension, topology)?;
    if tensor_rank != selected.rank {
        return Err(invalid(
            "prediction construction rank differs from retained placement",
        ));
    }
    let definition = selected.definition().clone();
    let parameters = definition.parameters().clone();
    let layout = selected.layout().as_ref().clone();
    let mut bindings = PredictionBindingIndex::for_extension(extension, tasks)?;
    let prepared = match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            let SafetensorsModelConfig::DeepSeekV3(target_args) = &selected.model else {
                return Err(invalid("prediction source and executable families differ"));
            };
            let geometry = crate::deepseek::parallel::v3_local_geometry(&target_args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = eredu_runtime::prepare_transform_source_layout(
                tasks.iter().map(|task| task.lowering()),
                || {
                    let source_parameters = extension.definition().parameters();
                    eredu_runtime::derive_partitioned_transform_source_layout(
                        source_parameters,
                        &parameters,
                        tensor_rank,
                    )
                    .map_err(|error| invalid(error.to_string()))
                },
            )?;
            let source_geometry = source_layout
                .as_ref()
                .map(|layout| {
                    crate::deepseek::parallel::v3_local_geometry(args, layout)
                        .map_err(|error| invalid(error.to_string()))
                })
                .transpose()?;
            let source = crate::deepseek::v3::V3UnitFactory::from_definition(
                args.clone(),
                extension.definition().clone(),
                source_geometry,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let target_layers = usize::try_from(args.num_hidden_layers)
                .map_err(|error| invalid(error.to_string()))?;
            let state = geometry
                .state_layout()
                .slice(target_layers..target_layers + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v3::V3UnitFactory::from_definition(
                target_args.clone(),
                definition,
                Some(geometry),
            )
            .map_err(|error| invalid(error.to_string()))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::prediction_source(extension)
                    .architecture_descriptor();
            let mut units = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let source_unit = source
                    .construct_unit::<B>(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut local_unit = local
                    .construct_unit::<B>(depth + 1, 0, execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
                if let crate::deepseek::v3::Unit::Prediction(prediction) = &mut local_unit {
                    if let crate::deepseek::block::V3FeedForward::Routed(moe) =
                        &mut prediction.decoder.feed_forward
                    {
                        let scope = descriptor.component_scopes.iter()
                            .find(|scope| matches!(scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth))
                            .ok_or_else(|| invalid("prepared prediction has no component scope"))?;
                        let [component] = scope.routed_components.as_slice() else {
                            return Err(invalid(
                                "V3 prediction must declare one resident routed bank",
                            ));
                        };
                        let coordinates =
                            crate::component_partition::derive_coordinates_for_experts(
                                component,
                                &layout,
                                &(0..component.expert_count).collect::<Vec<_>>(),
                            )
                            .map_err(|error| invalid(error.to_string()))?;
                        moe.bind_resident_unit_coordinates(
                            coordinates.units().clone(),
                            topology.topology().world_size() > 1,
                        );
                    }
                }
                units.push(
                    prepare_prediction_unit(source_unit, local_unit, bindings.take(Some(depth))?)?
                        .with_source_layout(source_layout.clone()),
                );
            }
            Ok(PreparedPredictionExtension::DeepSeekV3 {
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(parameters),
                units,
                state,
            })
        }
        SafetensorsModelConfig::DeepSeekV4(args) => {
            let SafetensorsModelConfig::DeepSeekV4(target_args) = &selected.model else {
                return Err(invalid("prediction source and executable families differ"));
            };
            let geometry = crate::deepseek::parallel::v4_local_geometry(&target_args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let state_layout = crate::deepseek::v4::state_layout(geometry.args())
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = eredu_runtime::prepare_transform_source_layout(
                tasks.iter().map(|task| task.lowering()),
                || {
                    let source_parameters = extension.definition().parameters();
                    eredu_runtime::derive_partitioned_transform_source_layout(
                        source_parameters,
                        &parameters,
                        tensor_rank,
                    )
                    .map_err(|error| invalid(error.to_string()))
                },
            )?;
            let source_geometry = source_layout
                .as_ref()
                .map(|layout| {
                    crate::deepseek::parallel::v4_local_geometry(args, layout)
                        .map_err(|error| invalid(error.to_string()))
                })
                .transpose()?;
            let source = crate::deepseek::v4::V4UnitFactory::from_definition(
                args.clone(),
                extension.definition().clone(),
                source_geometry,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v4::V4UnitFactory::from_definition(
                target_args.clone(),
                definition,
                Some(geometry),
            )
            .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("DeepSeek-V4 target count exceeds usize"))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::prediction_source(extension)
                    .architecture_descriptor();
            let mut units = Vec::with_capacity(extension.depth());
            let mut state = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let ordinal = target + depth;
                let source_unit = source
                    .construct_unit::<B>(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut local_unit = local
                    .construct_unit::<B>(depth + 1, 0, execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let policy = state_layout.layer(ordinal).cloned().ok_or_else(|| {
                    invalid(format!(
                        "DeepSeek-V4 prediction depth {depth} has no state policy"
                    ))
                })?;
                let routed = match &mut local_unit {
                    crate::deepseek::v4::Unit::Prediction(prediction) => {
                        &mut prediction.decoder.feed_forward
                    }
                    crate::deepseek::v4::Unit::Dspark(block) => &mut block.feed_forward,
                    crate::deepseek::v4::Unit::Target(_) => {
                        return Err(invalid("prediction construction returned a target unit"));
                    }
                };
                {
                    let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                        scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth
                    ) || matches!(scope.kind, eredu_core::component::ComponentExecutionScopeKind::FusedPrediction)).ok_or_else(|| invalid("prepared V4 prediction has no component scope"))?;
                    let mut components = scope
                        .routed_components
                        .iter()
                        .filter(|component| component.layer_index == ordinal);
                    let component = components.next().ok_or_else(|| {
                        invalid("V4 prediction must declare its resident routed bank")
                    })?;
                    if components.next().is_some() {
                        return Err(invalid(
                            "V4 prediction declares multiple resident banks for one block",
                        ));
                    }
                    let coordinates = crate::component_partition::derive_coordinates_for_experts(
                        component,
                        &layout,
                        &(0..component.expert_count).collect::<Vec<_>>(),
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    routed.bind_resident_unit_coordinates(
                        coordinates.units().clone(),
                        topology.topology().world_size() > 1,
                    );
                }
                units.push(
                    prepare_prediction_unit(source_unit, local_unit, bindings.take(Some(depth))?)?
                        .with_source_layout(source_layout.clone()),
                );
                state.push((ordinal, policy));
            }
            let state_layout = state_layout
                .slice(target..target + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            if args.dspark.is_some() {
                let strategy = DsparkPredictionStrategy::from_args(args)?;
                let source_static = source
                    .construct_dspark::<B>(source_context)
                    .map_err(|error| invalid(error.to_string()))?
                    .ok_or_else(|| {
                        invalid("source DSpark declaration has no fused pinned modules")
                    })?;
                let local_static = local
                    .construct_dspark::<B>(execution_context)
                    .map_err(|error| invalid(error.to_string()))?
                    .ok_or_else(|| {
                        invalid("local DSpark declaration has no fused pinned modules")
                    })?;
                Ok(PreparedPredictionExtension::DeepSeekV4Dspark {
                    layout: std::sync::Arc::new(layout),
                    parameters: std::sync::Arc::new(parameters),
                    extension: PreparedDsparkPredictionExtension {
                        strategy,
                        static_modules: prepare_prediction_unit(
                            source_static,
                            local_static,
                            bindings.take(None)?,
                        )?
                        .with_source_layout(source_layout.clone()),
                    },
                    units,
                    state,
                    state_layout,
                })
            } else {
                Ok(PreparedPredictionExtension::DeepSeekV4 {
                    layout: std::sync::Arc::new(layout),
                    parameters: std::sync::Arc::new(parameters),
                    units,
                    state,
                    state_layout,
                })
            }
        }
        SafetensorsModelConfig::Inkling(args) => {
            let SafetensorsModelConfig::Inkling(target_args) = &selected.model else {
                return Err(invalid("prediction source and executable families differ"));
            };
            let source = crate::inkling::MtpModel::<B>::new(args, source_context)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
            let local = crate::inkling::MtpModel::<B>::new(&target_args, execution_context)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no configured depth"))?;
            let state = crate::inkling::mtp_state_layout(args)
                .map_err(|error| invalid(error.to_string()))?
                .ok_or_else(|| invalid("Inkling prediction extension has no state layout"))?;
            let shared = match (source.chain_norm, local.chain_norm) {
                (Some(source), Some(local)) => Some(prepare_prediction_unit(
                    crate::inkling::MtpShared { chain_norm: source },
                    crate::inkling::MtpShared { chain_norm: local },
                    bindings.take(None)?,
                )?),
                (None, None) => None,
                _ => {
                    return Err(invalid(
                        "Inkling source and executable chain norms disagree",
                    ));
                }
            };
            let units = source
                .layers
                .into_iter()
                .zip(local.layers)
                .enumerate()
                .map(|(depth, (source, local))| {
                    prepare_prediction_unit(source, local, bindings.take(Some(depth))?)
                })
                .collect::<Result<_, _>>()?;
            Ok(PreparedPredictionExtension::Inkling {
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(parameters),
                units,
                shared,
                state,
            })
        }
        SafetensorsModelConfig::QwenHybrid(args) => {
            let SafetensorsModelConfig::QwenHybrid(target_args) = &selected.model else {
                return Err(invalid("prediction source and executable families differ"));
            };
            let description = parameters.clone();
            let geometry = crate::qwen::hybrid::local_geometry(&target_args.text, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = eredu_runtime::prepare_transform_source_layout(
                tasks.iter().map(|task| task.lowering()),
                || {
                    eredu_runtime::derive_partitioned_transform_source_layout(
                        extension.definition().parameters(),
                        &description,
                        tensor_rank,
                    )
                    .map_err(|error| invalid(error.to_string()))
                },
            )?;
            let source_geometry = source_layout
                .as_ref()
                .map(|layout| crate::qwen::hybrid::local_geometry(&args.text, layout))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.text.num_hidden_layers)
                .map_err(|_| invalid("Qwen hybrid target count exceeds usize"))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::prediction_source(extension)
                    .architecture_descriptor();
            let unit_parameters = (0..extension.depth())
                .map(|depth| {
                    let source_config = source_geometry
                        .as_ref()
                        .and_then(|g| g.prediction(depth))
                        .unwrap_or(&args.text);
                    let local_config = geometry
                        .prediction(depth)
                        .ok_or_else(|| invalid("Qwen hybrid prediction has no local geometry"))?;
                    Ok((
                        crate::qwen::hybrid::BlockParameters::prediction(source_config, depth)
                            .map_err(|e| invalid(e.to_string()))?,
                        crate::qwen::hybrid::BlockParameters::prediction(local_config, depth)
                            .map_err(|e| invalid(e.to_string()))?,
                    ))
                })
                .collect::<Result<Vec<_>, eredu_core::artifact::ArtifactError>>()?;
            let source_shared = args
                .text
                .prediction_shared()
                .cloned()
                .ok_or_else(|| invalid("undeclared prediction shared scope"))?;
            let local_shared = target_args
                .text
                .prediction_shared()
                .cloned()
                .ok_or_else(|| invalid("undeclared prediction shared scope"))?;
            let mut units = Vec::with_capacity(extension.depth());
            for (depth, (source_parameters, local_parameters)) in
                unit_parameters.into_iter().enumerate()
            {
                let source = crate::qwen::hybrid::PredictionUnit::<B>::from_parameters(
                    source_parameters,
                    target + depth,
                    source_context,
                )
                .map_err(|e| invalid(e.to_string()))?;
                let mut local = crate::qwen::hybrid::PredictionUnit::<B>::from_parameters(
                    local_parameters,
                    target + depth,
                    execution_context,
                )
                .map_err(|e| invalid(e.to_string()))?;
                if let crate::qwen::hybrid::FeedForward::Routed(moe) = &mut local.block.feed_forward
                {
                    let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                        scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth: selected } if selected == depth
                    )).ok_or_else(|| invalid("prepared Qwen prediction has no component scope"))?;
                    let [component] = scope.routed_components.as_slice() else {
                        return Err(invalid(
                            "Qwen prediction must declare one resident routed bank",
                        ));
                    };
                    let coordinates = crate::component_partition::derive_coordinates_for_experts(
                        component,
                        &layout,
                        &(0..component.expert_count).collect::<Vec<_>>(),
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    moe.bind_resident_unit_coordinates(
                        coordinates.units().clone(),
                        topology.topology().world_size() > 1,
                    );
                }
                units.push(
                    prepare_prediction_unit(source, local, bindings.take(Some(depth))?)?
                        .with_source_layout(source_layout.clone()),
                );
            }
            let state = geometry
                .state_layout()
                .slice(target..target + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            let shared = prepare_prediction_unit(
                crate::qwen::hybrid::PredictionShared::<B>::from_parameters(
                    source_shared,
                    source_context,
                )
                .map_err(|error| invalid(error.to_string()))?,
                crate::qwen::hybrid::PredictionShared::<B>::from_parameters(
                    local_shared,
                    execution_context,
                )
                .map_err(|error| invalid(error.to_string()))?,
                bindings.take(None)?,
            )?
            .with_source_layout(source_layout.clone());
            Ok(PreparedPredictionExtension::QwenHybrid {
                shared,
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(description),
                units,
                state,
            })
        }
        SafetensorsModelConfig::NemotronH(_) => {
            let SafetensorsModelConfig::NemotronH(_) = &selected.model else {
                return Err(invalid("prediction source and executable families differ"));
            };
            let source_declaration = extension
                .nemotron_declaration()
                .ok_or_else(|| invalid("Nemotron prediction has no retained typed scope"))?;
            let crate::configuration::PredictionScopes::Nemotron(local_declaration) =
                &selected.scopes
            else {
                return Err(invalid("selected Nemotron prediction has no typed scope"));
            };
            let args = source_declaration.args();
            let target_args = local_declaration.args();
            let source_description = extension.definition().parameters();
            let description = parameters.clone();
            let geometry = crate::nemotron_h::local_geometry(local_declaration, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = eredu_runtime::prepare_transform_source_layout(
                tasks.iter().map(|task| task.lowering()),
                || {
                    eredu_runtime::derive_partitioned_transform_source_layout(
                        &source_description,
                        &description,
                        tensor_rank,
                    )
                    .map_err(|error| invalid(error.to_string()))
                },
            )?;
            let source_geometry = source_layout
                .as_ref()
                .map(|layout| crate::nemotron_h::local_geometry(source_declaration, layout))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?;
            let pattern = source_declaration
                .predictions
                .len()
                .checked_div(extension.depth())
                .filter(|pattern| *pattern > 0)
                .ok_or_else(|| invalid("Nemotron-H MTP pattern is empty"))?;
            if source_declaration.predictions.len() != local_declaration.predictions.len()
                || source_declaration
                    .predictions
                    .iter()
                    .zip(&local_declaration.predictions)
                    .any(|(source, local)| source.policy != local.policy)
            {
                return Err(invalid(
                    "Nemotron-H prediction source and local invocations differ",
                ));
            }
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::prediction_source(extension)
                    .architecture_descriptor();
            let mut groups = Vec::with_capacity(extension.depth());
            for prediction in 0..extension.depth() {
                let mut units = Vec::with_capacity(pattern);
                for relative in 0..pattern {
                    let physical = prediction * pattern + relative;
                    let source_unit_geometry = source_geometry
                        .as_ref()
                        .map(|geometry| {
                            geometry.prediction_unit(physical).copied().ok_or_else(|| {
                                invalid(format!(
                                    "Nemotron-H prediction source unit {physical} has no local geometry"
                                ))
                            })
                        })
                        .transpose()?;
                    let source = crate::nemotron_h::PredictionUnit::<B>::from_parameters(
                        args,
                        source_declaration.predictions[physical].clone(),
                        source_unit_geometry,
                        source_context,
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    let local_geometry =
                        geometry.prediction_unit(physical).copied().ok_or_else(|| {
                            invalid(format!(
                                "Nemotron-H prediction unit {physical} has no local geometry"
                            ))
                        })?;
                    let mut local = crate::nemotron_h::PredictionUnit::<B>::from_parameters(
                        target_args,
                        local_declaration.predictions[physical].clone(),
                        Some(local_geometry),
                        execution_context,
                    )
                    .map_err(|error| invalid(error.to_string()))?;
                    if let crate::nemotron_h::Operator::Sparse(moe) = &mut local.block.operator {
                        let scope = descriptor.component_scopes.iter().find(|scope| matches!(
                            scope.kind, eredu_core::component::ComponentExecutionScopeKind::Prediction { depth } if depth == prediction
                        )).ok_or_else(|| invalid("prepared Nemotron prediction has no component scope"))?;
                        let component = scope
                            .routed_components
                            .iter()
                            .find(|component| {
                                component.layer_index == args.num_hidden_layers as usize + physical
                            })
                            .ok_or_else(|| {
                                invalid("prepared Nemotron prediction has no declared routed bank")
                            })?;
                        let coordinates =
                            crate::component_partition::derive_coordinates_for_experts(
                                component,
                                &layout,
                                &(0..component.expert_count).collect::<Vec<_>>(),
                            )
                            .map_err(|error| invalid(error.to_string()))?;
                        moe.bind_resident_unit_coordinates(
                            coordinates.units().clone(),
                            topology.topology().world_size() > 1,
                        );
                    }
                    units.push(
                        prepare_prediction_unit(source, local, bindings.take(Some(physical))?)?
                            .with_source_layout(source_layout.clone()),
                    );
                }
                groups.push(units);
            }
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("Nemotron-H target depth exceeds usize"))?;
            let state = geometry
                .state_layout()
                .slice(target..target + source_declaration.predictions.len())
                .map_err(|error| invalid(error.to_string()))?;
            Ok(PreparedPredictionExtension::NemotronH {
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(description),
                groups,
                state,
            })
        }
        _ => Err(invalid(
            "selected prediction extension has no neutral preparation",
        )),
    }?;
    bindings.finish()?;
    Ok(prepared)
}
