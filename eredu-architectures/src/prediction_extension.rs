//! Prepared neutral construction for embedded prediction extensions.

use eredu_nn::EmbeddingOperator as _;
use std::num::NonZeroUsize;

pub mod equation;
pub(crate) mod invocation;
mod placement;
pub(crate) mod residency;
mod snapshot;
/// Source-bound metadata materialization of selected prediction equations.
pub mod workspace;
use invocation::prediction_invocation_optional;
pub use invocation::{PredictionInvocation, PreparedPredictionInvocationRoots};
pub(crate) use placement::PredictionPlacementSlot;
pub use placement::PreparedPredictionPlacement;
mod construction;
pub use construction::PreparedPredictionConstruction;
pub(crate) use construction::prepare_retained;

/// Visits the actual materialized prediction modules in architecture-declared
/// ownership order. A module ordinal identifies a physical parameter owner; it
/// is independent of proposal depth and mutable lane-state position.
pub trait PredictionModuleVisitor<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Traversal failure retained by the consumer.
    type Error;
    /// Borrows one complete typed module without rebuilding or loading parameters.
    fn visit<U: Parameterized<B::Tensor>>(
        &mut self,
        ordinal: usize,
        module: &mut M::Module<U>,
    ) -> Result<(), Self::Error>;
}

/// Borrows retained prediction owners for cold storage inspection. Architectures
/// enumerate physical modules and immutable state prototypes; materializers
/// inspect their own storage without loading parameters or cloning state.
pub trait PredictionResourceVisitor<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Inspection failure retained by the consumer.
    type Error;

    /// Borrows a complete module, using the same physical ordinal as mutable
    /// parameter traversal. Shared target parameters remain with the target.
    fn module<U: Parameterized<B::Tensor>>(
        &mut self,
        ordinal: usize,
        module: &M::Module<U>,
    ) -> Result<(), Self::Error>;

    /// Borrows a retained pooling-state prototype, including its storage owner.
    fn pooling_state(&mut self, state: &M::PoolingState) -> Result<(), Self::Error>;

    /// Borrows a retained model-state prototype, including its storage owners.
    fn model_state(&mut self, state: &M::ModelState) -> Result<(), Self::Error>;
}

use eredu_core::{ParallelRankTopology, ParallelTopology, cache::LayerCachePolicy};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::{ArchitectureParameters, LocalModelLayout, StateLayout};
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
            vec![
                SpeculativeCaptureEntry::new(
                    speculative_identity(path)?,
                    shape.clone(),
                    owner,
                    speculative_identity(observation)?,
                )
                .and_then(|entry| entry.with_bounded_dimension(0))
                .and_then(|entry| entry.with_bounded_dimension(1))
                .map_err(|error| invalid(error.to_string()))?,
            ],
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
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PredictionModuleRole {
    /// One physical unit; ordinary units execute sequentially.
    Unit,
    /// A shared module retained while ordinary units execute. Its bytes overlap
    /// the largest unit instead of requiring every unit to stay resident.
    Shared,
}

/// One architecture-constructed neutral module and its exact checkpoint recipes.
pub struct PreparedPredictionUnit<M> {
    source: M,
    local: M,
    tasks: std::sync::Arc<Vec<ReplicatedTextMaterializationTask>>,
    source_layout: Option<std::sync::Arc<LocalModelLayout>>,
    residency: eredu_runtime::LayerWeightResidency,
    role: PredictionModuleRole,
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
    fn new<T: 'static>(
        source: M,
        local: M,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<Self, eredu_core::artifact::ArtifactError>
    where
        M: Parameterized<T>,
    {
        Self::new_with_role(source, local, tasks, PredictionModuleRole::Unit)
    }

    fn new_shared<T: 'static>(
        source: M,
        local: M,
        tasks: &[ReplicatedTextMaterializationTask],
    ) -> Result<Self, eredu_core::artifact::ArtifactError>
    where
        M: Parameterized<T>,
    {
        Self::new_with_role(source, local, tasks, PredictionModuleRole::Shared)
    }

    fn new_with_role<T: 'static>(
        source: M,
        local: M,
        tasks: &[ReplicatedTextMaterializationTask],
        role: PredictionModuleRole,
    ) -> Result<Self, eredu_core::artifact::ArtifactError>
    where
        M: Parameterized<T>,
    {
        let names = eredu_nn::validate_parameter_topology(&source)
            .map_err(|error| invalid(error.to_string()))?
            .into_iter()
            .map(|metadata| metadata.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let all_task_names = tasks
            .iter()
            .map(ReplicatedTextMaterializationTask::name)
            .collect::<Vec<_>>();
        let tasks = tasks
            .iter()
            .filter(|task| names.contains(task.name()))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(owner) = tasks.iter().find_map(|task| task.auxiliary_residency()) {
            if owner.shared() != (role == PredictionModuleRole::Shared)
                || tasks
                    .iter()
                    .any(|task| task.auxiliary_residency() != Some(owner))
            {
                return Err(invalid(
                    "cold prediction residency grouping differs from the constructed physical module",
                ));
            }
        }
        let selected_names = tasks
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name()).chain(
                    task.output_companions()
                        .iter()
                        .map(ReplicatedTextOutputCompanion::name),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        let missing = names
            .iter()
            .filter(|name| !selected_names.contains(name.as_str()))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(invalid(format!(
                "prediction module parameters have no exact pre-resource materialization tasks: {missing:?}; local={:?}; all={all_task_names:?}",
                tasks
                    .iter()
                    .map(ReplicatedTextMaterializationTask::name)
                    .collect::<Vec<_>>()
            )));
        }
        Ok(Self {
            source,
            local,
            tasks: std::sync::Arc::new(tasks),
            source_layout: None,
            residency: eredu_runtime::LayerWeightResidency::FullyResident,
            role,
        })
    }

    fn with_source_layout(mut self, layout: Option<std::sync::Arc<LocalModelLayout>>) -> Self {
        self.source_layout = layout;
        self
    }

    /// Source placement used by admitted local transforms, when present. The
    /// source module already has this rank's floating geometry; recipes remain
    /// global and are selected through this layout before bounded conversion.
    pub fn source_layout(&self) -> Option<&std::sync::Arc<LocalModelLayout>> {
        self.source_layout.as_ref()
    }

    /// Retained immutable-weight policy selected for the complete execution.
    /// A native module owner must account for this module under the same bounds
    /// as its target, including overlapping shared modules and target statics.
    pub const fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }

    /// Architecture-declared overlap with other physical prediction owners.
    pub const fn role(&self) -> PredictionModuleRole {
        self.role
    }

    /// Consumes source/target modules and exact tasks. A source layout marks a
    /// rank-local conversion source; otherwise the source module is global.
    pub fn into_parts(self) -> (M, M, Vec<ReplicatedTextMaterializationTask>) {
        (
            self.source,
            self.local,
            std::sync::Arc::unwrap_or_clone(self.tasks),
        )
    }

    /// Consumes the modules while retaining the exact immutable task row.
    /// No materialization-task payload is copied by this source loan.
    pub fn into_shared_parts(
        self,
    ) -> (M, M, std::sync::Arc<Vec<ReplicatedTextMaterializationTask>>) {
        (self.source, self.local, self.tasks)
    }
}

/// Borrows the exact architecture-selected mutable-state layout for a prediction lane.
/// Member ordinals identify architecture state layers, not proposal iterations or
/// physical parameter-module membership. Native projection validates actual
/// current state against these declarations without reconstructing family policy.
#[derive(Clone, Copy, Debug)]
pub enum PredictionStateSourceLayout<'a> {
    /// Ordered compressed-cache members and their complete rank-local policies.
    Sequential(&'a [(usize, LayerCachePolicy)]),
    /// Ordered pooling-cache members and their complete rank-local policies.
    Pooling(&'a [(usize, LayerCachePolicy)]),
    /// Exact model-state layout for the selected prediction profile.
    Model(&'a StateLayout),
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
        /// Ordered rank-local cache policy for every prediction unit.
        state: Vec<(usize, LayerCachePolicy)>,
        /// Immutable inputs retained from the successful ordinary constructor.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
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
        /// Immutable inputs retained from the successful ordinary constructor.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
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
        /// Immutable inputs retained from the successful ordinary constructor.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
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
        /// Exact immutable declarations from successful source preparation.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
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
        /// Immutable successful source declarations for retained reconstruction.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
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
        /// Immutable successful declarations, grouped by physical prediction units.
        construction: std::sync::Arc<construction::PreparedPredictionConstruction>,
    },
}

/// Destination for a fresh lane from the exact architecture-owned state source.
///
/// The selected extension supplies membership and prototypes. The destination
/// supplies allocation/copy authority and a closed result owner; returning a
/// prepared state does not authorize invocation, checkpointing or ordinary clone.
pub trait PredictionStateStartupFactory<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Concrete preparation refusal, retaining the destination's authority.
    type Error;
    /// Closed destination ownership, independent of the source's lifetime.
    type Prepared<T>;

    /// Constructs exactly the selected number of initially empty sequential slots.
    fn sequential(
        &mut self,
        count: usize,
    ) -> Result<Self::Prepared<Vec<M::SequentialState>>, Self::Error>;

    /// Copies actual pooling prototypes, preserving each policy and current state.
    fn pooling(
        &mut self,
        source: &[M::PoolingState],
    ) -> Result<Self::Prepared<Vec<M::PoolingState>>, Self::Error>;

    /// Copies the actual architecture-materialized model-state prototype.
    fn model(
        &mut self,
        source: &M::ModelState,
    ) -> Result<Self::Prepared<M::ModelState>, Self::Error>;
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
    type PoolingState: Clone
        + eredu_nn::PoolingAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeLayerState<B>
        + 'static;
    /// Native cache for one sequential compressed-latent prediction layer.
    type SequentialState: Clone
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeLayerState<B>
        + 'static;
    /// Native realization of an architecture-declared state layout.
    type ModelState: PredictionModelState<B> + 'static;
    /// Borrowed resources used during one materialization pass.
    type Context<'a>;
    /// Resource authority used to create independently retained state snapshots.
    /// This may carry ownership beyond the tensor queue or materialization pass.
    type SnapshotContext<'a>: Copy;

    /// Retains the concrete changed-state/output roots of one physical call.
    /// The default preserves ordinary native retirement. Metadata materializers
    /// override the optional form to pay their actual root destination before fill.
    fn retain_prediction_invocation<'a, O, const N: usize>(
        outcome: Result<O, eredu_nn::Error>,
        state: impl IntoIterator<Item = &'a B::Tensor>,
        outputs: impl FnOnce(&O) -> [&B::Tensor; N],
        context: &<B::Tensor as Tensor>::Context,
    ) -> PredictionInvocation<B::Tensor, O>
    where
        B::Tensor: 'a,
    {
        Self::retain_prediction_invocation_optional(
            outcome,
            state,
            |output| Some(outputs(output)),
            context,
        )
    }

    /// Same exact root retention with an optional semantic readout.
    fn retain_prediction_invocation_optional<'a, O, const N: usize>(
        outcome: Result<O, eredu_nn::Error>,
        state: impl IntoIterator<Item = &'a B::Tensor>,
        outputs: impl FnOnce(&O) -> Option<[&B::Tensor; N]>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> PredictionInvocation<B::Tensor, O>
    where
        B::Tensor: 'a,
    {
        prediction_invocation_optional(outcome, state, outputs)
    }

    /// Retains the same roots through an explicit actual module-call destination.
    /// Ordinary/metadata adapters preserve their existing retention implementation.
    fn retain_prediction_invocation_from_source<'a, O, const N: usize>(
        outcome: Result<O, eredu_nn::Error>,
        state: impl IntoIterator<Item = &'a B::Tensor>,
        outputs: impl FnOnce(&O) -> [&B::Tensor; N],
        source: Option<&mut dyn PreparedPredictionInvocationRoots<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> PredictionInvocation<B::Tensor, O>
    where
        B::Tensor: 'a,
    {
        Self::retain_prediction_invocation_optional_from_source(
            outcome,
            state,
            |output| Some(outputs(output)),
            source,
            context,
        )
    }
    fn retain_prediction_invocation_optional_from_source<'a, O, const N: usize>(
        outcome: Result<O, eredu_nn::Error>,
        state: impl IntoIterator<Item = &'a B::Tensor>,
        outputs: impl FnOnce(&O) -> Option<[&B::Tensor; N]>,
        source: Option<&mut dyn PreparedPredictionInvocationRoots<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> PredictionInvocation<B::Tensor, O>
    where
        B::Tensor: 'a,
    {
        match source {
            Some(source) => {
                invocation::prediction_invocation_prepared(outcome, state, outputs, source)
            }
            None => Self::retain_prediction_invocation_optional(outcome, state, outputs, context),
        }
    }

    /// Adds a source-explicit root destination without changing ordinary module
    /// scheduling. A prepared native materializer lends its exact once-only slot.
    fn invoke_module_with_roots<U, O>(
        module: &mut Self::Module<U>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(
            &mut U,
            Option<&mut dyn PreparedPredictionInvocationRoots<B::Tensor>>,
        ) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
    {
        Self::invoke_module(module, context, |module| operation(module, None))
    }
    /// Same nested shared-module worker with its actual root destination loan.
    fn invoke_module_with_shared_roots<U, V, O>(
        module: &mut Self::Module<U>,
        shared: Option<&mut Self::Module<V>>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(
            &mut U,
            Option<&mut V>,
            Option<&mut dyn PreparedPredictionInvocationRoots<B::Tensor>>,
        ) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
        V: Parameterized<B::Tensor>,
    {
        Self::invoke_module_with_shared(module, shared, context, |module, shared| {
            operation(module, shared, None)
        })
    }

    /// Invokes one typed module while its immutable parameters remain resident.
    /// The result always carries changed state dependencies, including on an
    /// equation failure. A paged owner must settle these dependencies before
    /// unloading parameters or releasing their residency reservation.
    fn invoke_module<U, O>(
        module: &mut Self::Module<U>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(&mut U) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
    {
        let invocation = operation(module.as_mut());
        Self::complete_prediction_values(invocation.retained_values(), context)
            .map_err(eredu_nn::Error::backend_source)?;
        invocation.into_outcome()
    }

    /// Invokes a unit with a distinct shared parameter owner. Both reservations
    /// remain live through their common output and state dependencies.
    fn invoke_module_with_shared<U, V, O>(
        module: &mut Self::Module<U>,
        shared: Option<&mut Self::Module<V>>,
        context: &<B::Tensor as Tensor>::Context,
        operation: impl FnOnce(&mut U, Option<&mut V>) -> PredictionInvocation<B::Tensor, O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
        V: Parameterized<B::Tensor>,
    {
        Self::invoke_module(module, context, |module| match shared {
            None => operation(module, None),
            Some(shared) => {
                let mut retained = Vec::new();
                let outcome = Self::invoke_module(shared, context, |shared| {
                    let invocation = operation(module, Some(shared));
                    retained.extend(invocation.retained_values().cloned());
                    invocation
                });
                PredictionInvocation::new(outcome, retained)
            }
        })
    }

    /// Settles the exact retained prediction state and output dependencies before
    /// observation publication. Native implementations include deferred validation
    /// dependencies and retain the original error source on failure.
    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure>
    where
        B::Tensor: 'a;

    /// Reserves one concrete shared materialization frame before construction.
    /// Ordinary materializers use their existing caller-owned host environment;
    /// counted metadata implementations charge their supplied context here.
    fn materialization_controls<T>(_: &mut Self::Context<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Creates a final exact-membership container before filling any child.
    /// Nested module/state payloads remain their actual producer's obligation.
    fn materialization_vector<T>(
        _: &mut Self::Context<'_>,
        count: usize,
    ) -> Result<Vec<T>, Self::Error> {
        Ok(Vec::with_capacity(count))
    }

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

    /// Complete durable-copy bound for one sequential cache, including native
    /// storage and any auxiliary ownership. Unknown support is never zero cost.
    fn sequential_snapshot_estimate(
        _state: &Self::SequentialState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies one sequential cache into independently restorable storage after
    /// reservation. Native failures retain their original source.
    fn sequential_snapshot<'a>(
        _state: &Self::SequentialState,
        _context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::SequentialState>, eredu_core::BackendFailure> {
        Ok(None)
    }
    /// Complete durable-copy bound for one pooling cache.
    fn pooling_snapshot_estimate(
        _state: &Self::PoolingState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies all pooling state and settles native copying before returning.
    fn pooling_snapshot<'a>(
        _state: &Self::PoolingState,
        _context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::PoolingState>, eredu_core::BackendFailure> {
        Ok(None)
    }
    /// Complete durable-copy bound for the declared model-state profile.
    fn model_snapshot_estimate(
        _state: &Self::ModelState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies the complete declared model-state profile after reservation.
    fn model_snapshot<'a>(
        _state: &Self::ModelState,
        _context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::ModelState>, eredu_core::BackendFailure> {
        Ok(None)
    }
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

    /// Borrows a deterministic diagnostic before choosing its paid destination.
    fn invalid_arguments(&self, arguments: std::fmt::Arguments<'_>) -> Self::Error {
        Self::invalid(arguments.to_string())
    }
}

/// Typed source visitor for an already prepared prediction lane.
/// Destinations may copy or project the supplied current state; membership and
/// profile dispatch stay in the selected architecture, never in a backend.
pub trait PredictionStateSourceFactory<B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Source-bound preparation, projection or native-copy refusal.
    type Error;
    /// Destination selected by the source profile; it may retain payload custody.
    /// Projection destinations may ignore the source type parameter.
    type Prepared<T>;
    /// Visits each actual sequential member in declared order.
    fn sequential(
        &mut self,
        source: &[M::SequentialState],
    ) -> Result<Self::Prepared<Vec<M::SequentialState>>, Self::Error>;
    /// Visits each actual pooling member in declared order.
    fn pooling(
        &mut self,
        source: &[M::PoolingState],
    ) -> Result<Self::Prepared<Vec<M::PoolingState>>, Self::Error>;
    /// Visits the actual complete decoder representation.
    fn model(
        &mut self,
        source: &M::ModelState,
    ) -> Result<Self::Prepared<M::ModelState>, Self::Error>;
}

/// Compatibility name for the same source visitor used by copy destinations.
pub use PredictionStateSourceFactory as PredictionStateCopyFactory;

mod executor_sealed {
    pub trait Sealed<A> {}
}

fn validate_selected_capture_shapes(
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    lane: &(impl eredu_runtime::SpeculativeLaneIdentityView + ?Sized),
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

    /// Reads the common actual source frontier of every selected lane member.
    /// A mismatched/empty lane cannot certify a captured span's seed placement.
    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error>;

    /// Reads only the stateful members consumed by this actual equation. A
    /// sequential proposal selects its architecture depth/group; earlier depths
    /// may already have advanced. Prefill, fused execution and replay consume
    /// the full lane. This allocation-free read grants no source authority.
    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError>;

    /// Completes every retained lane value plus the actual phase outputs. The
    /// architecture supplies state membership; the backend settles native work.
    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        self.with_state_values(state, |values| {
            M::complete_prediction_values(values.chain(outputs.iter().copied()), context)
        })
    }

    /// Lends the same architecture-selected lane roots to an exact completion
    /// mechanism without cloning values or allocating an intermediate list.
    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R;

    /// Visits every materialized prediction parameter owner exactly once in
    /// stable ordinal order. Shared target parameters remain with the target.
    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error>;

    /// Visits all modules and state prototypes owned by this extension without
    /// mutation, materialization, completion polling, or state cloning. Active
    /// lane state and outputs are separate owners and are not included here.
    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error>;

    /// Complete prediction-lane copy bound with architecture-owned membership.
    fn snapshot_estimate(
        &self,
        _state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies every prediction state member into durable independent storage.
    fn snapshot<'a>(
        &self,
        _state: &Self::LaneState,
        _context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        Ok(None)
    }

    /// Returns the proposal depth exposed to the scheduler.
    fn depth(&self) -> usize;

    /// Iteration/proposal depths supplied to the finite occurrence planner.
    /// A depth may contain several modules; shared modules may be outside those
    /// depths. Physical source membership uses the actual module/resource visitors.
    /// This describes the actual extension, independently of requested capacity.
    fn occurrence_shape(
        &self,
    ) -> Option<eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape> {
        Some(
            eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape::Sequential {
                depth: std::num::NonZeroUsize::new(self.depth())?,
            },
        )
    }

    /// Whether this exact extension implements internal prefill, proposal and
    /// replay hooks. This does not establish public collector or transport support.
    fn supports_internal_observations(&self) -> bool {
        false
    }

    /// Resolves complete phase hooks against the retained strategy. Backend
    /// binding forwards this fact without interpreting a family or phase name.
    fn activation_execution(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
    ) -> Option<crate::speculative_execution::SpeculativeActivationExecution> {
        crate::speculative_execution::SpeculativeActivationExecution::selected(
            self.supports_internal_observations(),
            self.depth(),
            selected,
        )
    }

    /// Physical rows consumed while seeding prediction state. Sequential
    /// prediction pairs each target hidden row with the following token; fused
    /// context builders may instead consume the complete accepted prefix.
    fn prefill_sequence_len(&self, target_sequence: usize) -> usize {
        self.prefill_alignment()
            .sequence_len(target_sequence as u64) as usize
    }

    /// Architecture-declared pairing consumed by execution and neutral planning.
    /// Fused scheduling alone does not determine hidden/token alignment.
    fn prefill_alignment(&self) -> eredu_core::speculative::PredictionPrefillAlignment {
        eredu_core::speculative::PredictionPrefillAlignment::NextToken
    }

    /// Clones the architecture-materialized lane-state prototype.
    fn new_state(&self) -> Self::LaneState;

    /// Sends the same state source as `new_state` to a qualified destination.
    /// Membership and family dispatch remain in this selected extension.
    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error>;

    /// Visits current lane membership through the same typed source contract.
    /// Static dispatch retains no model/module borrow in a saved copy provider.
    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error>;

    /// Copies through the same source dispatch used by workspace projection.
    fn prepare_copy_state<F: PredictionStateCopyFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        Self::prepare_state(source, destination)
    }

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
        lane: &(impl eredu_runtime::SpeculativeLaneIdentityView + ?Sized),
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

    /// Prefills with explicitly installed internal hooks. An absent observer
    /// preserves the ordinary path without creating diagnostic tensors.
    #[allow(clippy::too_many_arguments)]
    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if observer.is_some() {
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "selected prediction prefill has no internal observation path"
            )));
        }
        self.prefill::<S, I>(invoker, target_capture, hidden, tokens, lane)
    }

    /// Executes a proposal depth with admitted internal hooks. Outer proposal
    /// observers alone do not imply support for this operation.
    #[allow(clippy::too_many_arguments)]
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
        if observer.is_some() {
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "selected prediction proposal has no internal observation path"
            )));
        }
        self.logits::<S, I>(invoker, hidden, token, draft_index, lane)
    }

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

    /// Executes a fused proposal using the caller's temporary lane. The caller
    /// retains this lane through completion and discards it after the phase;
    /// speculative cache writes are never committed by this operation.
    #[allow(clippy::too_many_arguments)]
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
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if observer.is_some() {
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "selected fused proposal has no internal observation path"
            )));
        }
        self.fused_logits::<S, I>(invoker, anchor, capacity, lane)
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

    /// Replays retained target inputs with internal hooks when the selected
    /// extension supplies them. The caller retains replay/proposal attribution.
    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if observer.is_some() {
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "selected prediction replay has no internal observation path"
            )));
        }
        self.advance::<S, I>(invoker, hidden, tokens, lane)
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
    /// Inkling sequential prediction units.
    Inkling {
        /// Units in proposal order.
        units: Vec<M::Module<crate::inkling::MtpDepth<B>>>,
        /// Optional canonical chain normalization.
        shared: Option<M::Module<crate::inkling::MtpShared<B>>>,
        /// Immutable prediction-state prototype.
        state: M::ModelState,
    },
    /// Dense Qwen hybrid MTP units.
    QwenHybrid {
        /// Units in proposal order.
        units: Vec<M::Module<crate::qwen::hybrid::PredictionUnit<B>>>,
        /// Canonical shared fusion/normalization.
        shared: M::Module<crate::qwen::hybrid::PredictionShared<B>>,
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
    /// Units in proposal order.
    pub units: Vec<M::Module<crate::inkling::MtpDepth<B>>>,
    /// Optional canonical chain normalization.
    pub shared: Option<M::Module<crate::inkling::MtpShared<B>>>,
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
    /// Canonical shared fusion/normalization, visited once for parameter control.
    pub shared: M::Module<crate::qwen::hybrid::PredictionShared<B>>,
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
            Some(parallel) => self.embed_prediction_parallel(tokens, parallel, context),
            None => self.embed_prediction(tokens, context),
        }
    }

    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.project_prediction_instrumented(hidden, parallel, context, instrumentation)
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
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.static_modules_mut()
            .project_instrumented(hidden, parallel, context, instrumentation)
    }
}

struct V3PredictionSeedOperation<'a, B, M>
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
    for V3PredictionSeedOperation<'_, B, M>
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
    type Output = ();
    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v3::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_nn::Error> {
        M::invoke_module_with_roots(self.unit, context, |unit, source| {
            let outcome = architecture.pipeline_seed_prediction(
                unit,
                self.hidden,
                self.tokens,
                self.cache,
                parallel,
                context,
            );
            M::retain_prediction_invocation_from_source(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |_| [],
                source,
                context,
            )
        })
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
    unit: &'a mut M::Module<crate::deepseek::v3::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut M::SequentialState,
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
    M::SequentialState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v3::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_roots(self.unit, context, |unit, source| {
            let outcome = (|| {
                if let Some(observer) = self.observer {
                    return architecture.pipeline_forward_prediction_observed(
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
                    Some(parallel) => architecture.pipeline_forward_prediction_neutral_parallel(
                        unit,
                        self.hidden,
                        self.tokens,
                        self.cache,
                        parallel,
                        context,
                    ),
                    None => architecture.pipeline_forward_prediction(
                        unit,
                        self.hidden,
                        self.tokens,
                        self.cache,
                        context,
                    ),
                }
            })();
            M::retain_prediction_invocation_from_source(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |output| [&output.logits, &output.hidden, &output.tokens],
                source,
                context,
            )
        })
    }
}

struct V4PredictionSeedOperation<'a, B, M>
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
    for V4PredictionSeedOperation<'_, B, M>
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
    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_nn::Error> {
        M::invoke_module_with_roots(self.unit, context, |unit, source| {
            let outcome = architecture.pipeline_seed_prediction(
                unit,
                self.hidden,
                self.tokens,
                self.cache,
                parallel,
                context,
            );
            M::retain_prediction_invocation_from_source(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |_| [],
                source,
                context,
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
    unit: &'a mut M::Module<crate::deepseek::v4::Unit<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    cache: &'a mut M::PoolingState,
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
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = crate::deepseek::mtp::PredictionOutput<B::Tensor>;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_roots(self.unit, context, |unit, source| {
            let outcome = (|| {
                let hidden =
                    architecture.begin_partition_prediction_hidden(self.hidden, context)?;
                let output = if let Some(observer) = self.observer {
                    architecture.pipeline_forward_prediction_observed(
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
                            .pipeline_forward_prediction_neutral_parallel(
                                unit,
                                &hidden,
                                self.tokens,
                                self.cache,
                                parallel,
                                context,
                            ),
                        None => architecture.pipeline_forward_prediction(
                            unit,
                            &hidden,
                            self.tokens,
                            self.cache,
                            context,
                        ),
                    }
                }?;
                architecture.finish_partition_prediction_output(output, context)
            })();
            M::retain_prediction_invocation_from_source(
                outcome,
                eredu_runtime::RuntimeLayerState::<B>::retained_values(self.cache),
                |output| [&output.logits, &output.hidden, &output.tokens],
                source,
                context,
            )
        })
    }
}

fn model_equation_frontier<B, M>(
    state: &mut M,
    selected: std::ops::Range<usize>,
    expected: usize,
) -> Result<u64, equation::PredictionFrontierError>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionModelState<B>,
{
    if state.layout().len() != expected || state.prediction_layers_mut().len() != expected {
        return Err(equation::PredictionFrontierError::Membership);
    }
    // The declaration owns which physical members are stateless. Read each
    // selected actual member without cloning the layout or its backing arrays.
    equation::source_frontier(
        (0..expected).map(|index| {
            if matches!(state.layout().layer(index), Some(LayerCachePolicy::NoState)) {
                None
            } else {
                Some(state.prediction_layers_mut()[index].position())
            }
        }),
        selected,
        expected,
    )
}

fn model_prefill_frontier<B, M>(state: &mut M) -> Result<u64, eredu_nn::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionModelState<B>,
{
    let count = state.prediction_layers_mut().len();
    if state.layout().len() != count {
        return Err(eredu_nn::Error::backend(
            "prediction layout differs from state members",
        ));
    }
    let mut frontier = None;
    for index in 0..count {
        // Stateless physical modules participate in execution, but have no token
        // frontier. The retained architecture layout owns this distinction.
        if matches!(state.layout().layer(index), Some(LayerCachePolicy::NoState)) {
            continue;
        }
        let actual = state.prediction_layers_mut()[index].position();
        if frontier.is_some_and(|prior| prior != actual) {
            return Err(eredu_nn::Error::backend(
                "prediction state frontiers differ",
            ));
        }
        frontier = Some(actual);
    }
    u64::try_from(
        frontier
            .ok_or_else(|| eredu_nn::Error::backend("prediction lane has no stateful frontier"))?,
    )
    .map_err(eredu_nn::Error::backend_source)
}

fn common_prefill_frontier(
    mut positions: impl Iterator<Item = i32>,
) -> Result<u64, eredu_nn::Error> {
    let first = positions
        .next()
        .ok_or_else(|| eredu_nn::Error::backend("prediction seed has no state members"))?;
    if positions.any(|position| position != first) {
        return Err(eredu_nn::Error::backend(
            "prediction seed state members have different frontiers",
        ));
    }
    u64::try_from(first).map_err(eredu_nn::Error::backend_source)
}

/// Seed state has no score value; proposal and observed sequence forwards do.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PredictionReadout {
    StateOnly,
    Sequence,
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
        readout: PredictionReadout,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>,
        eredu_nn::Error,
    >
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
        unit: &mut crate::inkling::MtpDepth<B>,
        shared: Option<&mut crate::inkling::MtpShared<B>>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        readout: PredictionReadout,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>,
        eredu_nn::Error,
    >
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
        if readout == PredictionReadout::StateOnly {
            return Ok(None);
        }
        let logits = instrumentation.with_scope("prediction.readout", |instrumentation| {
            self.project_mtp_logits_instrumented(&output.hidden, parallel, context, instrumentation)
        })?;
        Ok(Some(
            crate::speculative_execution::EmbeddedPredictionOutput {
                logits,
                capture: output.hidden,
                tokens: crate::speculative_execution::EmbeddedPredictionTensor::ordinary(output.tokens),
            },
        ))
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
        unit: &mut crate::inkling::MtpDepth<B>,
        shared: Option<&mut crate::inkling::MtpShared<B>>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut M::ModelState,
        readout: PredictionReadout,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>,
        eredu_nn::Error,
    >
    where
        M: PredictionExtensionMaterializer<B>,
    {
        self.inner_mut().prediction_step::<M>(
            unit, shared, hidden, tokens, depth, state, readout, observer, parallel, context,
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
    unit: &'a mut M::Module<crate::inkling::MtpDepth<B>>,
    shared: Option<&'a mut M::Module<crate::inkling::MtpShared<B>>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    readout: PredictionReadout,
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
    type Output = Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_shared_roots(
            self.unit,
            self.shared,
            context,
            |unit, shared, source| {
                let outcome = architecture.prediction_step::<M>(
                    unit,
                    shared,
                    self.hidden,
                    self.tokens,
                    self.depth,
                    self.state,
                    self.readout,
                    self.observer,
                    parallel,
                    context,
                );
                M::retain_prediction_invocation_optional_from_source(
                    outcome,
                    self.state
                        .prediction_layers_mut()
                        .iter()
                        .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                    |output| {
                        output
                            .as_ref()
                            .map(|output| [&output.logits, &output.capture, &output.tokens])
                    },
                    source,
                    context,
                )
            },
        )
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
        readout: PredictionReadout,
        mut observer: Option<
            &mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>,
        >,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>,
        eredu_nn::Error,
    >
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
        if readout == PredictionReadout::StateOnly {
            return Ok(None);
        }
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
        Ok(Some(
            crate::speculative_execution::EmbeddedPredictionOutput {
                logits,
                capture: hidden,
                tokens: crate::speculative_execution::EmbeddedPredictionTensor::ordinary(tokens.clone()),
            },
        ))
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
            Some(parallel) => self.pipeline_embed_parallel(tokens, parallel, context),
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
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.static_modules_mut()
            .project_instrumented(hidden, parallel, context, instrumentation)
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
            Some(parallel) => self.pipeline_embed_parallel(tokens, parallel, context),
            None => self.begin_partition_prediction_embedding(tokens, context),
        }
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.project_prediction_instrumented(hidden, parallel, context, instrumentation)
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
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.inner_mut().embed_prediction(tokens, parallel, context)
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.inner_mut()
            .project_prediction(hidden, parallel, context, instrumentation)
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
    unit: &'a mut M::Module<crate::qwen::hybrid::PredictionUnit<B>>,
    shared: &'a mut M::Module<crate::qwen::hybrid::PredictionShared<B>>,
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    readout: PredictionReadout,
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
    type Output = Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut A,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_shared_roots(
            self.unit,
            Some(self.shared),
            context,
            |unit, shared, source| {
                let outcome = architecture.prediction_step::<M>(
                    unit,
                    shared.expect("Qwen prediction declares a shared owner"),
                    self.hidden,
                    self.tokens,
                    self.depth,
                    self.state,
                    self.readout,
                    self.observer,
                    parallel,
                    context,
                );
                M::retain_prediction_invocation_optional_from_source(
                    outcome,
                    self.state
                        .prediction_layers_mut()
                        .iter()
                        .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                    |output| {
                        output
                            .as_ref()
                            .map(|output| [&output.logits, &output.capture, &output.tokens])
                    },
                    source,
                    context,
                )
            },
        )
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
    units: &'a mut [M::Module<crate::nemotron_h::PredictionUnit<B>>],
    hidden: &'a B::Tensor,
    tokens: &'a B::Tensor,
    depth: usize,
    state: &'a mut M::ModelState,
    readout: PredictionReadout,
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
    type Output = Option<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>>;

    fn preserves_architecture_declarations(&self) -> bool { true }

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
            hidden = M::invoke_module_with_roots(unit, context, |unit, source| {
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
                M::retain_prediction_invocation_from_source(
                    outcome,
                    eredu_runtime::RuntimeLayerState::<B>::retained_values(state),
                    |output| [output],
                    source,
                    context,
                )
            })?;
        }
        if self.readout == PredictionReadout::StateOnly {
            return Ok(None);
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
        Ok(Some(
            crate::speculative_execution::EmbeddedPredictionOutput {
                logits,
                capture: hidden,
                tokens: crate::speculative_execution::EmbeddedPredictionTensor::ordinary(self.tokens.clone()),
            },
        ))
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

    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError> {
        let count = self.units.len();
        let selected = equation::selected_members(equation, count)?;
        equation::source_frontier(
            state.iter().map(|value| {
                Some(eredu_nn::CompressedAttentionCache::<B::Tensor>::offset(
                    value,
                ))
            }),
            selected,
            count,
        )
    }

    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error> {
        common_prefill_frontier(
            state
                .iter()
                .map(eredu_nn::CompressedAttentionCache::<B::Tensor>::offset),
        )
    }

    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R {
        use_values(
            &mut state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
        )
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        for (ordinal, module) in self.units.iter_mut().enumerate() {
            visitor.visit::<crate::deepseek::v3::Unit<B>>(ordinal, module)?;
        }
        Ok(())
    }

    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        for (ordinal, module) in self.units.iter().enumerate() {
            visitor.module::<crate::deepseek::v3::Unit<B>>(ordinal, module)?;
        }
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::sequential_snapshot_estimate)
    }
    fn snapshot<'a>(
        &self,
        state: &Self::LaneState,
        context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        snapshot::sequence_copy(state, |state| M::sequential_snapshot(state, context))
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        (0..self.units.len())
            .map(|_| M::sequential_state())
            .collect()
    }

    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.sequential(self.units.len())
    }

    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.sequential(source)
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
        crate::deepseek::v3::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v3::Model<B>, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
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
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "DeepSeek-V3 prediction units and lane state differ"
            )));
        }
        for (depth, (unit, cache)) in self.units.iter_mut().zip(lane).enumerate() {
            if observer.is_none() {
                invoker.invoke(V3PredictionSeedOperation::<B, M> {
                    unit,
                    hidden,
                    tokens,
                    cache,
                })?;
                continue;
            }
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
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
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
            invoker.invalid_arguments(format_args!(
                "prediction depth {draft_index} exceeds {count} units"
            ))
        })?;
        let cache = lane.get_mut(draft_index).ok_or_else(|| {
            invoker.invalid_arguments(format_args!(
                "{}",
                "DeepSeek-V3 prediction lane is too shallow"
            ))
        })?;
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

    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v3::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v3::Model<B>, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
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

    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError> {
        let count = match self {
            Self::Sequential { units, .. } | Self::Dspark { units, .. } => units.len(),
        };
        let selected = equation::selected_members(equation, count)?;
        equation::source_frontier(
            state
                .iter()
                .map(|value| Some(eredu_nn::PoolingAttentionCache::<B::Tensor>::offset(value))),
            selected,
            count,
        )
    }

    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error> {
        common_prefill_frontier(
            state
                .iter()
                .map(eredu_nn::PoolingAttentionCache::<B::Tensor>::offset),
        )
    }

    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R {
        use_values(
            &mut state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
        )
    }

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
        for (ordinal, module) in units.iter_mut().enumerate() {
            visitor.visit::<crate::deepseek::v4::Unit<B>>(offset + ordinal, module)?;
        }
        Ok(())
    }

    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        let (offset, units) = match self {
            Self::Sequential { units, .. } => (0, units),
            Self::Dspark {
                static_modules,
                units,
                ..
            } => {
                visitor.module::<crate::deepseek::v4::DsparkStatic<B>>(0, static_modules)?;
                (1, units)
            }
        };
        for (ordinal, module) in units.iter().enumerate() {
            visitor.module::<crate::deepseek::v4::Unit<B>>(offset + ordinal, module)?;
        }
        let state = match self {
            Self::Sequential { state, .. } | Self::Dspark { state, .. } => state,
        };
        for state in state {
            visitor.pooling_state(state)?;
        }
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::pooling_snapshot_estimate)
    }
    fn snapshot<'a>(
        &self,
        state: &Self::LaneState,
        context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        snapshot::sequence_copy(state, |state| M::pooling_snapshot(state, context))
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn depth(&self) -> usize {
        match self {
            Self::Sequential { units, .. } | Self::Dspark { units, .. } => units.len(),
        }
    }

    fn occurrence_shape(
        &self,
    ) -> Option<eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape> {
        use eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape;
        let depth = std::num::NonZeroUsize::new(self.depth())?;
        Some(match self {
            Self::Sequential { .. } => EmbeddedPredictionShape::Sequential { depth },
            Self::Dspark { strategy, .. } => EmbeddedPredictionShape::Fused {
                depth,
                maximum_proposals: std::num::NonZeroUsize::new(strategy.proposal_capacity())?,
            },
        })
    }

    fn prefill_alignment(&self) -> eredu_core::speculative::PredictionPrefillAlignment {
        use eredu_core::speculative::PredictionPrefillAlignment;
        match self {
            Self::Dspark { .. } => PredictionPrefillAlignment::Aligned,
            Self::Sequential { .. } => PredictionPrefillAlignment::NextToken,
        }
    }

    fn new_state(&self) -> Self::LaneState {
        match self {
            Self::Sequential { state, .. } | Self::Dspark { state, .. } => state.clone(),
        }
    }

    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        match self {
            Self::Sequential { state, .. } | Self::Dspark { state, .. } => {
                destination.pooling(state)
            }
        }
    }

    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.pooling(source)
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
        lane: &(impl eredu_runtime::SpeculativeLaneIdentityView + ?Sized),
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
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
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
                    return Err(invoker.invalid_arguments(format_args!(
                        "{}",
                        "DeepSeek-V4 prediction units and lane state differ"
                    )));
                }
                for (depth, (unit, cache)) in units.iter_mut().zip(lane).enumerate() {
                    if observer.is_none() {
                        invoker.invoke(V4PredictionSeedOperation::<B, M> {
                            unit,
                            hidden,
                            tokens,
                            cache,
                        })?;
                        continue;
                    }
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
                    return Err(invoker.invalid_arguments(format_args!(
                        "{}",
                        "DSpark prediction units and lane state differ"
                    )));
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
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
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
                    invoker.invalid_arguments(format_args!(
                        "prediction depth {draft_index} exceeds {count} units"
                    ))
                })?;
                let cache = lane.get_mut(draft_index).ok_or_else(|| {
                    invoker.invalid_arguments(format_args!(
                        "{}",
                        "DeepSeek-V4 prediction lane is too shallow"
                    ))
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
            Self::Dspark { .. } => Err(invoker
                .invalid_arguments(format_args!("{}", "DSpark uses fused prediction proposals"))),
        }
    }

    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        crate::deepseek::v4::Model<B>:
            eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<crate::deepseek::v4::Model<B>, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
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
                    return Err(invoker.invalid_arguments(format_args!(
                        "{}",
                        "DSpark prediction units and lane state differ"
                    )));
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

    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError> {
        let count = self.units.len();
        let selected = equation::selected_members(equation, count)?;
        model_equation_frontier::<B, M::ModelState>(state, selected, count)
    }

    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error> {
        model_prefill_frontier::<B, M::ModelState>(state)
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R {
        use_values(
            &mut state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
        )
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(shared) = self.shared.as_mut() {
            visitor.visit::<crate::inkling::MtpShared<B>>(0, shared)?;
        }
        for (depth, unit) in self.units.iter_mut().enumerate() {
            visitor.visit::<crate::inkling::MtpDepth<B>>(depth + 1, unit)?;
        }
        Ok(())
    }

    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(shared) = self.shared.as_ref() {
            visitor.module::<crate::inkling::MtpShared<B>>(0, shared)?;
        }
        for (depth, unit) in self.units.iter().enumerate() {
            visitor.module::<crate::inkling::MtpDepth<B>>(depth + 1, unit)?;
        }
        visitor.model_state(&self.state)?;
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot<'a>(
        &self,
        state: &Self::LaneState,
        context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(&self.state)
    }

    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(source)
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
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
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
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "Inkling MTP state does not match prediction depths"
            )));
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
                        readout: if observer.is_some() {
                            PredictionReadout::Sequence
                        } else {
                            PredictionReadout::StateOnly
                        },
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
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
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
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "Inkling MTP state does not match prediction depths"
            )));
        }
        if draft_index >= self.units.len() {
            return Err(invoker.invalid_arguments(format_args!(
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
                readout: PredictionReadout::Sequence,
                observer,
            })
            .and_then(|output| {
                output
                    .map(|output| (output.logits, output.capture))
                    .ok_or_else(|| {
                        invoker.invalid_arguments(format_args!(
                            "{}",
                            "proposal omitted its requested sequence scores"
                        ))
                    })
            })
    }

    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
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

    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError> {
        let count = self.units.len();
        let selected = equation::selected_members(equation, count)?;
        model_equation_frontier::<B, M::ModelState>(state, selected, count)
    }

    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error> {
        model_prefill_frontier::<B, M::ModelState>(state)
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R {
        use_values(
            &mut state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
        )
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.visit::<crate::qwen::hybrid::PredictionShared<B>>(0, &mut self.shared)?;
        for (ordinal, module) in self.units.iter_mut().enumerate() {
            visitor.visit::<crate::qwen::hybrid::PredictionUnit<B>>(ordinal + 1, module)?;
        }
        Ok(())
    }

    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.module::<crate::qwen::hybrid::PredictionShared<B>>(0, &self.shared)?;
        for (ordinal, module) in self.units.iter().enumerate() {
            visitor.module::<crate::qwen::hybrid::PredictionUnit<B>>(ordinal + 1, module)?;
        }
        visitor.model_state(&self.state)?;
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot<'a>(
        &self,
        state: &Self::LaneState,
        context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(&self.state)
    }

    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(source)
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
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
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
            return Err(invoker.invalid_arguments(format_args!(
                "{}",
                "Qwen hybrid prediction units and state differ"
            )));
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
                        readout: if observer.is_some() {
                            PredictionReadout::Sequence
                        } else {
                            PredictionReadout::StateOnly
                        },
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
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
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
            invoker.invalid_arguments(format_args!(
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
                readout: PredictionReadout::Sequence,
                observer,
            })
            .and_then(|output| {
                output
                    .map(|output| (output.logits, output.capture))
                    .ok_or_else(|| {
                        invoker.invalid_arguments(format_args!(
                            "{}",
                            "proposal omitted its requested sequence scores"
                        ))
                    })
            })
    }
    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
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

    fn equation_frontier(
        &self,
        state: &mut Self::LaneState,
        equation: &equation::PredictionEquation<&B::Tensor>,
    ) -> Result<u64, equation::PredictionFrontierError> {
        let count = self
            .groups
            .iter()
            .try_fold(0usize, |count, group| count.checked_add(group.len()))
            .ok_or(equation::PredictionFrontierError::Membership)?;
        let selected = match equation {
            equation::PredictionEquation::Sequential { depth, .. } => {
                // Match the existing operation's actual physical state indexing.
                let group = self
                    .groups
                    .get(*depth)
                    .ok_or(equation::PredictionFrontierError::Membership)?;
                let start = depth
                    .checked_mul(group.len())
                    .ok_or(equation::PredictionFrontierError::Membership)?;
                let end = start
                    .checked_add(group.len())
                    .ok_or(equation::PredictionFrontierError::Membership)?;
                start..end
            }
            _ => 0..count,
        };
        model_equation_frontier::<B, M::ModelState>(state, selected, count)
    }

    fn prefill_frontier(&self, state: &mut Self::LaneState) -> Result<u64, eredu_nn::Error> {
        model_prefill_frontier::<B, M::ModelState>(state)
    }

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn with_state_values<'a, R>(
        &self,
        state: &'a mut Self::LaneState,
        use_values: impl FnOnce(&mut dyn Iterator<Item = &'a B::Tensor>) -> R,
    ) -> R {
        use_values(
            &mut state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
        )
    }

    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        for (ordinal, module) in self.groups.iter_mut().flatten().enumerate() {
            visitor.visit::<crate::nemotron_h::PredictionUnit<B>>(ordinal, module)?;
        }
        Ok(())
    }

    fn visit_retained_resources<V: PredictionResourceVisitor<B, M>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        for (ordinal, module) in self.groups.iter().flatten().enumerate() {
            visitor.module::<crate::nemotron_h::PredictionUnit<B>>(ordinal, module)?;
        }
        visitor.model_state(&self.state)?;
        Ok(())
    }

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot<'a>(
        &self,
        state: &Self::LaneState,
        context: M::SnapshotContext<'a>,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.groups.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
    }

    fn prepare_new_state<F: PredictionStateStartupFactory<B, M>>(
        &self,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(&self.state)
    }

    fn prepare_state<F: PredictionStateSourceFactory<B, M>>(
        source: &Self::LaneState,
        destination: &mut F,
    ) -> Result<F::Prepared<Self::LaneState>, F::Error> {
        destination.model(source)
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
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
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
                        readout: if observer.is_some() {
                            PredictionReadout::Sequence
                        } else {
                            PredictionReadout::StateOnly
                        },
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
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
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
            invoker.invalid_arguments(format_args!(
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
                readout: PredictionReadout::Sequence,
                observer,
            })
            .and_then(|output| {
                output
                    .map(|output| (output.logits, output.capture))
                    .ok_or_else(|| {
                        invoker.invalid_arguments(format_args!(
                            "{}",
                            "proposal omitted its requested sequence scores"
                        ))
                    })
            })
    }

    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
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
    static_modules: &'a mut M::Module<crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [M::Module<crate::deepseek::v4::Unit<B>>],
    captures: &'a B::Tensor,
    caches: &'a mut [M::PoolingState],
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
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = ();

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_roots(self.static_modules, context, |static_modules, source| {
            let outcome = architecture.pipeline_prefill_dspark_extension_context_observed_with_modules::<_, invocation::MaterializedPredictionModules<B, M>>(
            self.strategy,
            static_modules,
            self.units,
            self.captures,
            self.caches,
            context,
            self.observer,
        );
            M::retain_prediction_invocation_from_source(
                outcome,
                self.caches
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |_output| [],
                source,
                context,
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
    static_modules: &'a mut M::Module<crate::deepseek::v4::DsparkStatic<B>>,
    units: &'a mut [M::Module<crate::deepseek::v4::Unit<B>>],
    anchor: &'a B::Tensor,
    capacity: usize,
    caches: &'a mut [M::PoolingState],
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
    M::PoolingState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    type Output = B::Tensor;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module_with_roots(self.static_modules, context, |static_modules, source| {
            let outcome = architecture.pipeline_dspark_extension_proposal_observed_with_modules::<_, invocation::MaterializedPredictionModules<B, M>>(
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
            M::retain_prediction_invocation_from_source(
                outcome,
                self.caches
                    .iter()
                    .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values),
                |output| [output],
                source,
                context,
            )
        })
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
sealed_prediction_target!(crate::qwen::hybrid::LayeredModel<B>);

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

impl<B, P> pairing_sealed::Sealed
    for crate::replicated_model::FixedReplicatedModel<
        B,
        crate::replicated_text::QwenHybridReplicated,
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
                units,
                shared,
                state,
            } => Ok(MaterializedInklingPrediction {
                units,
                shared,
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
                units,
                shared,
                state,
            } => Ok(MaterializedInklingPrediction {
                units,
                shared,
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
            MaterializedPredictionExtension::QwenHybrid {
                units,
                shared,
                state,
            } => Ok(MaterializedQwenHybridPrediction {
                units,
                shared,
                state,
            }),
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
            MaterializedPredictionExtension::QwenHybrid {
                units,
                shared,
                state,
            } => Ok(MaterializedQwenHybridPrediction {
                units,
                shared,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the conditional Qwen target",
            )),
        }
    }
}

impl<B> MaterializedPredictionTarget<B> for crate::qwen::hybrid::LayeredModel<B>
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
            MaterializedPredictionExtension::QwenHybrid {
                units,
                shared,
                state,
            } => Ok(MaterializedQwenHybridPrediction {
                units,
                shared,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the Qwen text target",
            )),
        }
    }
}

impl<B, P> MaterializedPredictionTarget<B>
    for crate::replicated_model::FixedReplicatedModel<
        B,
        crate::replicated_text::QwenHybridReplicated,
        P,
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
            MaterializedPredictionExtension::QwenHybrid {
                units,
                shared,
                state,
            } => Ok(MaterializedQwenHybridPrediction {
                units,
                shared,
                state,
            }),
            _ => Err(invalid(
                "materialized prediction extension does not match the Qwen text target",
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
            Self::Inkling { units, .. } => units.len(),
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
    /// Borrows the exact state policies retained by this selected construction.
    /// This descriptor supplies geometry only; it grants no copy, mutation or
    /// native completion authority over the caller's current state.
    pub fn state_source_layout(&self) -> PredictionStateSourceLayout<'_> {
        match self {
            Self::DeepSeekV3 { state, .. } => PredictionStateSourceLayout::Sequential(state),
            Self::DeepSeekV4 { state, .. } | Self::DeepSeekV4Dspark { state, .. } => {
                PredictionStateSourceLayout::Pooling(state)
            }
            Self::Inkling { state, .. }
            | Self::QwenHybrid { state, .. }
            | Self::NemotronH { state, .. } => PredictionStateSourceLayout::Model(state),
        }
    }

    /// Applies the retained execution policy to every physical prediction owner.
    /// Shared modules occur once here even when several logical depths use them.
    pub fn with_residency(mut self, residency: eredu_runtime::LayerWeightResidency) -> Self {
        match &mut self {
            Self::DeepSeekV3 { units, .. } => {
                for unit in units {
                    unit.residency = residency;
                }
            }
            Self::DeepSeekV4 { units, .. } => {
                for unit in units {
                    unit.residency = residency;
                }
            }
            Self::DeepSeekV4Dspark {
                extension, units, ..
            } => {
                extension.static_modules.residency = residency;
                for unit in units {
                    unit.residency = residency;
                }
            }
            Self::Inkling { units, shared, .. } => {
                for unit in units {
                    unit.residency = residency;
                }
                if let Some(shared) = shared {
                    shared.residency = residency;
                }
            }
            Self::QwenHybrid { units, shared, .. } => {
                for unit in units {
                    unit.residency = residency;
                }
                shared.residency = residency;
            }
            Self::NemotronH { groups, .. } => {
                for unit in groups.iter_mut().flatten() {
                    unit.residency = residency;
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
        M::materialization_controls::<(
            Self,
            MaterializedPredictionExtension<B, M>,
            Result<MaterializedPredictionExtension<B, M>, M::Error>,
        )>(context)?;
        match self {
            Self::DeepSeekV3 { layout, units, .. } => {
                Ok(MaterializedPredictionExtension::DeepSeekV3 {
                    units: materialize_units::<B, M, _>(units, &layout, context)?,
                })
            }
            Self::DeepSeekV4 {
                layout,
                units,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::DeepSeekV4 {
                units: materialize_units::<B, M, _>(units, &layout, context)?,
                state: materialize_pooling::<B, M>(state, context)?,
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
                    static_modules: M::materialize_module(context, static_modules, Some(&layout))?,
                    units: materialize_units::<B, M, _>(units, &layout, context)?,
                    state: materialize_pooling::<B, M>(state, context)?,
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
                    .map(|shared| M::materialize_module(context, shared, Some(&layout)))
                    .transpose()?,
                units: materialize_units::<B, M, _>(units, &layout, context)?,
                state: M::model_state(context, state)?,
            }),
            Self::QwenHybrid {
                layout,
                units,
                shared,
                state,
                ..
            } => Ok(MaterializedPredictionExtension::QwenHybrid {
                shared: M::materialize_module(context, shared, Some(&layout))?,
                units: materialize_units::<B, M, _>(units, &layout, context)?,
                state: M::model_state(context, state)?,
            }),
            Self::NemotronH {
                layout,
                groups,
                state,
                ..
            } => {
                M::materialization_controls::<(
                    Vec<Vec<PreparedPredictionUnit<crate::nemotron_h::PredictionUnit<B>>>>,
                    Vec<Vec<M::Module<crate::nemotron_h::PredictionUnit<B>>>>,
                )>(context)?;
                let mut output = M::materialization_vector(context, groups.len())?;
                for units in groups {
                    output.push(materialize_units::<B, M, _>(units, &layout, context)?);
                }
                Ok(MaterializedPredictionExtension::NemotronH {
                    groups: output,
                    state: M::model_state(context, state)?,
                })
            }
        }
    }
}

fn materialize_units<B, M, U>(
    units: Vec<PreparedPredictionUnit<U>>,
    layout: &LocalModelLayout,
    context: &mut M::Context<'_>,
) -> Result<Vec<M::Module<U>>, M::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    U: Parameterized<B::Tensor>,
{
    M::materialization_controls::<(
        Vec<PreparedPredictionUnit<U>>,
        std::vec::IntoIter<PreparedPredictionUnit<U>>,
        Vec<M::Module<U>>,
        Result<Vec<M::Module<U>>, M::Error>,
    )>(context)?;
    let mut output = M::materialization_vector(context, units.len())?;
    for unit in units {
        output.push(M::materialize_module(context, unit, Some(layout))?);
    }
    Ok(output)
}

fn materialize_pooling<B, M>(
    state: Vec<(usize, LayerCachePolicy)>,
    context: &mut M::Context<'_>,
) -> Result<Vec<M::PoolingState>, M::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    M::materialization_controls::<(
        Vec<(usize, LayerCachePolicy)>,
        std::vec::IntoIter<(usize, LayerCachePolicy)>,
        Vec<M::PoolingState>,
        Result<Vec<M::PoolingState>, M::Error>,
    )>(context)?;
    let mut output = M::materialization_vector(context, state.len())?;
    for (ordinal, policy) in state {
        output.push(M::pooling_state(context, ordinal, policy)?);
    }
    Ok(output)
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

/// Prepares an extension for one already admitted partitioned rank.
pub fn prepare_partitioned_prediction_extension<B, R, Q>(
    extension: &PredictionExtensionPlan,
    selected: &crate::partitioned_execution::SelectedPartitionedAdmission<R, Q>,
    tasks: &[ReplicatedTextMaterializationTask],
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
        tasks,
        source_context,
        execution_context,
    )
}

/// Prepares an extension for a single-rank replicated target.
pub fn prepare_replicated_prediction_extension<B>(
    extension: &PredictionExtensionPlan,
    tasks: &[ReplicatedTextMaterializationTask],
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
        tasks,
        source_context,
        execution_context,
    )
}

mod ordinary_construction;

pub(crate) fn prepare<B>(
    extension: &PredictionExtensionPlan,
    topology: ParallelRankTopology,
    tasks: &[ReplicatedTextMaterializationTask],
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
    let construction = ordinary_construction::Construction::<B> {
        extension, topology, tensor_rank, tasks, source_context, execution_context,
    };
    match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => construction.v3(args),
        SafetensorsModelConfig::DeepSeekV4(args) => construction.v4(args),
        SafetensorsModelConfig::Inkling(args) => construction.inkling(args),

        SafetensorsModelConfig::QwenHybrid(args) => construction.qwen(args),

        SafetensorsModelConfig::NemotronH(args) => construction.nemotron(args),
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
                assert!(
                    contract
                        .requirements()
                        .mechanisms()
                        .mechanisms()
                        .contains(&mechanism)
                );
            }
            assert!(
                !contract
                    .requirements()
                    .mechanisms()
                    .mechanisms()
                    .contains(&SpeculativeMechanism::Communication)
            );
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
        assert!(
            contract
                .target_capture()
                .entries()
                .iter()
                .all(|entry| entry.shape() == [2, 7, 16])
        );
        assert!(
            contract
                .target_capture()
                .instantiate(vec![vec![1, 6, 16], vec![1, 6, 16], vec![1, 6, 16]])
                .is_ok()
        );
        assert_eq!(
            contract
                .target_capture()
                .instantiate(vec![vec![1, 8, 16], vec![1, 8, 16], vec![1, 8, 16]])
                .unwrap_err(),
            eredu_runtime::SpeculativeCaptureError::ShapeMismatch
        );
        assert_ne!(contract.target_capture(), reordered.target_capture());
        assert!(
            contract
                .requirements()
                .mechanisms()
                .mechanisms()
                .contains(&SpeculativeMechanism::Communication)
        );
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
