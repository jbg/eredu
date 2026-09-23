//! Prepared neutral construction for embedded prediction extensions.

use eredu_nn::EmbeddingOperator as _;
use std::num::NonZeroUsize;

pub(crate) mod invocation;
mod placement;
pub(crate) mod residency;
mod resources;
mod snapshot;
use invocation::prediction_invocation;
pub use invocation::PredictionInvocation;
pub(crate) use placement::PredictionPlacementSlot;
pub use placement::PreparedPredictionPlacement;

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

use eredu_core::{cache::LayerCachePolicy, ParallelRankTopology, ParallelTopology};
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
    tasks: Vec<ReplicatedTextMaterializationTask>,
    source_layout: Option<std::sync::Arc<LocalModelLayout>>,
    residency: eredu_runtime::LayerWeightResidency,
    role: PredictionModuleRole,
    local_parameters: Vec<eredu_nn::ParameterMetadata>,
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
                return Err(invalid("cold prediction residency grouping differs from the constructed physical module"));
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
                tasks.iter().map(ReplicatedTextMaterializationTask::name).collect::<Vec<_>>()
            )));
        }
        let local_parameters = eredu_nn::validate_parameter_topology(&local)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            source,
            local,
            tasks,
            source_layout: None,
            residency: eredu_runtime::LayerWeightResidency::FullyResident,
            role,
            local_parameters,
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
        (self.source, self.local, self.tasks)
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
    fn sequential_snapshot(
        _state: &Self::SequentialState,
        _context: &<B::Tensor as Tensor>::Context,
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
    fn pooling_snapshot(
        _state: &Self::PoolingState,
        _context: &<B::Tensor as Tensor>::Context,
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
    fn model_snapshot(
        _state: &Self::ModelState,
        _context: &<B::Tensor as Tensor>::Context,
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

    /// Completes every retained lane value plus the actual phase outputs. The
    /// architecture supplies state membership; the backend settles native work.
    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure>;

    /// Visits every materialized prediction parameter owner exactly once in
    /// stable ordinal order. Shared target parameters remain with the target.
    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
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
    fn snapshot(
        &self,
        _state: &Self::LaneState,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        Ok(None)
    }

    /// Returns the proposal depth exposed to the scheduler.
    fn depth(&self) -> usize;

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
        eredu_runtime::prediction_resources::PredictionExecutionMode::Sequential
            .prefill_sequence_len(target_sequence)
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
            return Err(I::invalid(
                "selected prediction prefill has no internal observation path".into(),
            ));
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
            return Err(I::invalid(
                "selected prediction proposal has no internal observation path".into(),
            ));
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
            return Err(I::invalid(
                "selected fused proposal has no internal observation path".into(),
            ));
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
            return Err(I::invalid(
                "selected prediction replay has no internal observation path".into(),
            ));
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

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.unit, context, |unit| {
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
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::speculative_execution::EmbeddedPredictionOutput<B::Tensor>, eredu_nn::Error>
    where
        M: PredictionExtensionMaterializer<B>,
    {
        self.inner_mut().prediction_step::<M>(
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
    unit: &'a mut M::Module<crate::inkling::MtpDepth<B>>,
    shared: Option<&'a mut M::Module<crate::inkling::MtpShared<B>>>,
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
    units: &'a mut [M::Module<crate::nemotron_h::PredictionUnit<B>>],
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

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
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

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::sequential_snapshot_estimate)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
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

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
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

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sequence_estimate(state, M::pooling_snapshot_estimate)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
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

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
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

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
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

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
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

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.units.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
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

    fn supports_internal_observations(&self) -> bool {
        true
    }

    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        M::complete_prediction_values(
            state
                .prediction_layers_mut()
                .iter()
                .flat_map(eredu_runtime::RuntimeLayerState::<B>::retained_values)
                .chain(outputs.iter().copied()),
            context,
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

    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::model_snapshot_estimate(state)
    }
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        M::model_snapshot(state, context)
    }

    fn depth(&self) -> usize {
        self.groups.len()
    }

    fn new_state(&self) -> Self::LaneState {
        self.state.clone()
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

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.static_modules, context, |static_modules| {
            let outcome = architecture.pipeline_prefill_dspark_extension_context_observed_with_modules::<_, invocation::MaterializedPredictionModules<B, M>>(
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

    fn apply(
        self,
        architecture: &mut crate::deepseek::v4::Model<B>,
        _state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, eredu_nn::Error> {
        M::invoke_module(self.static_modules, context, |static_modules| {
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
        match self {
            Self::DeepSeekV3 { layout, units, .. } => {
                Ok(MaterializedPredictionExtension::DeepSeekV3 {
                    units: units
                        .into_iter()
                        .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                        .collect::<Result<_, _>>()?,
                })
            }
            Self::DeepSeekV4 {
                layout,
                units,
                state,
                ..
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
                ..
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
                units: units
                    .into_iter()
                    .map(|unit| M::materialize_module(context, unit, Some(&layout)))
                    .collect::<Result<_, _>>()?,
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
                ..
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
    match extension.complete_architecture().model() {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            let mut formats = args.linear_formats.clone();
            formats.extend(
                tasks
                    .iter()
                    .filter(|task| {
                        matches!(
                            task.role(),
                            eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                        )
                    })
                    .map(|task| (task.name().to_owned(), task.executable())),
            );
            let target_args =
                crate::deepseek::v3_with_checkpoint_formats(args, formats).map_err(invalid)?;
            let parameters = crate::deepseek::parallel::v3_parameter_description(&target_args)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::deepseek::parallel::v3_local_geometry(&target_args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = if tasks.iter().any(|task| {
                matches!(
                    task.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                )
            }) {
                let source_parameters = crate::deepseek::parallel::v3_parameter_description(args)
                    .map_err(|error| invalid(error.to_string()))?;
                Some(std::sync::Arc::new(
                    crate::partitioned_execution::derive_partitioned_transform_source_layout(
                        &source_parameters,
                        &parameters,
                        tensor_rank,
                    )
                    .map_err(invalid)?,
                ))
            } else {
                None
            };
            let source = match &source_layout {
                Some(layout) => crate::deepseek::v3::Model::<B>::new_parallel(
                    args.clone(),
                    crate::deepseek::parallel::v3_local_geometry(args, layout)
                        .map_err(|error| invalid(error.to_string()))?,
                    source_context,
                ),
                None => crate::deepseek::v3::Model::<B>::new(args.clone(), source_context),
            }
            .map_err(|error| invalid(error.to_string()))?;
            let target_layers = usize::try_from(args.num_hidden_layers)
                .map_err(|error| invalid(error.to_string()))?;
            let state = geometry
                .state_layout()
                .slice(target_layers..target_layers + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v3::Model::<B>::new_parallel(
                target_args,
                geometry,
                execution_context,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                    extension.complete_architecture().clone(),
                )
                .architecture_descriptor();
            let mut units = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let source_unit = source
                    .construct_unit(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut local_unit = local
                    .construct_unit(depth + 1, 0, execution_context)
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
                    PreparedPredictionUnit::new(source_unit, local_unit, tasks)?
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
            let mut formats = args.linear_formats.clone();
            formats.extend(
                tasks
                    .iter()
                    .filter(|task| {
                        matches!(
                            task.role(),
                            eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                        )
                    })
                    .map(|task| (task.name().to_owned(), task.executable())),
            );
            let target_args =
                crate::deepseek::v4_with_checkpoint_formats(args, formats).map_err(invalid)?;
            let parameters = crate::deepseek::parallel::v4_parameter_description(&target_args)
                .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::deepseek::parallel::v4_local_geometry(&target_args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let state_layout = crate::deepseek::v4::state_layout(geometry.args())
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = if tasks.iter().any(|task| {
                matches!(
                    task.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                )
            }) {
                let source_parameters = crate::deepseek::parallel::v4_parameter_description(args)
                    .map_err(|error| invalid(error.to_string()))?;
                Some(std::sync::Arc::new(
                    crate::partitioned_execution::derive_partitioned_transform_source_layout(
                        &source_parameters,
                        &parameters,
                        tensor_rank,
                    )
                    .map_err(invalid)?,
                ))
            } else {
                None
            };
            let source = match &source_layout {
                Some(layout) => crate::deepseek::v4::Model::<B>::new_parallel(
                    args.clone(),
                    crate::deepseek::parallel::v4_local_geometry(args, layout)
                        .map_err(|error| invalid(error.to_string()))?,
                    source_context,
                ),
                None => crate::deepseek::v4::Model::<B>::new(args.clone(), source_context),
            }
            .map_err(|error| invalid(error.to_string()))?;
            let local = crate::deepseek::v4::Model::<B>::new_parallel(
                target_args,
                geometry,
                execution_context,
            )
            .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.num_hidden_layers)
                .map_err(|_| invalid("DeepSeek-V4 target count exceeds usize"))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                    extension.complete_architecture().clone(),
                )
                .architecture_descriptor();
            let mut units = Vec::with_capacity(extension.depth());
            let mut state = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let ordinal = target + depth;
                let source_unit = source
                    .construct_unit(depth + 1, 0, source_context)
                    .map_err(|error| invalid(error.to_string()))?;
                let mut local_unit = local
                    .construct_unit(depth + 1, 0, execution_context)
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
                        return Err(invalid("prediction construction returned a target unit"))
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
                    PreparedPredictionUnit::new(source_unit, local_unit, tasks)?
                        .with_source_layout(source_layout.clone()),
                );
                state.push((ordinal, policy));
            }
            let state_layout = state_layout
                .slice(target..target + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
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
                    layout: std::sync::Arc::new(layout),
                    parameters: std::sync::Arc::new(parameters),
                    extension: PreparedDsparkPredictionExtension {
                        strategy,
                        static_modules: PreparedPredictionUnit::new_shared(
                            source_static,
                            local_static,
                            tasks,
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
            let mut formats = args
                .text_config
                .quantized_weight_configs
                .clone()
                .unwrap_or_default();
            for task in tasks {
                match task.executable().weight_quantization() {
                    Some(format) => {
                        formats.insert(task.name().to_owned(), format);
                    }
                    None => {
                        formats.remove(task.name());
                    }
                }
            }
            let target_args =
                crate::inkling::with_checkpoint_formats(args, formats).map_err(invalid)?;
            let parameters =
                crate::inkling::LayeredModel::<B>::new(target_args.clone(), execution_context)
                    .and_then(|model| model.parameter_description(execution_context))
                    .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                tensor_rank,
            )
            .map_err(invalid)?;
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
                (Some(source), Some(local)) => Some(PreparedPredictionUnit::new_shared(
                    crate::inkling::MtpShared { chain_norm: source },
                    crate::inkling::MtpShared { chain_norm: local },
                    tasks,
                )?),
                (None, None) => None,
                _ => {
                    return Err(invalid(
                        "Inkling source and executable chain norms disagree",
                    ))
                }
            };
            let units = source
                .layers
                .into_iter()
                .zip(local.layers)
                .map(|(source, local)| PreparedPredictionUnit::new(source, local, tasks))
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
            // Prediction units carry their own resident expert banks, populated
            // by the same exact parameter tasks as their attention and shared
            // projections. Execution uses ResidentExpertProvider, independently
            // of the target's expert residency policy.
            let description = if args.vision.is_some() {
                crate::qwen::hybrid::ConditionalLayeredModel::<B>::new(args.clone(), source_context)
                    .and_then(|model| model.parameter_description(source_context))
            } else {
                crate::qwen::hybrid::LayeredModel::<B>::new(args.text.clone(), source_context)
                    .and_then(|model| model.parameter_description(source_context))
            }
            .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &description,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::qwen::hybrid::local_geometry(&args.text, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let target = usize::try_from(args.text.num_hidden_layers)
                .map_err(|_| invalid("Qwen hybrid target count exceeds usize"))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                    extension.complete_architecture().clone(),
                )
                .architecture_descriptor();
            let mut units = Vec::with_capacity(extension.depth());
            for depth in 0..extension.depth() {
                let source = crate::qwen::hybrid::PredictionUnit::<B>::new(
                    &args.text,
                    depth,
                    source_context,
                )
                .map_err(|error| invalid(error.to_string()))?;
                let local_config = geometry.prediction(depth).ok_or_else(|| {
                    invalid(format!(
                        "Qwen hybrid prediction depth {depth} has no local geometry"
                    ))
                })?;
                let mut local = crate::qwen::hybrid::PredictionUnit::<B>::new(
                    local_config,
                    depth,
                    execution_context,
                )
                .map_err(|error| invalid(error.to_string()))?;
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
                units.push(PreparedPredictionUnit::new(source, local, tasks)?);
            }
            let state = geometry
                .state_layout()
                .slice(target..target + extension.depth())
                .map_err(|error| invalid(error.to_string()))?;
            let shared = PreparedPredictionUnit::new_shared(
                crate::qwen::hybrid::PredictionShared::<B>::new(&args.text, source_context)
                    .map_err(|error| invalid(error.to_string()))?,
                crate::qwen::hybrid::PredictionShared::<B>::new(&args.text, execution_context)
                    .map_err(|error| invalid(error.to_string()))?,
                tasks,
            )?;
            Ok(PreparedPredictionExtension::QwenHybrid {
                shared,
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(description),
                units,
                state,
            })
        }
        SafetensorsModelConfig::NemotronH(args) => {
            let source_architecture =
                crate::nemotron_h::LayeredModel::<B>::new(args.clone(), source_context)
                    .map_err(|error| invalid(error.to_string()))?;
            let source_description = source_architecture
                .parameter_description(source_context)
                .map_err(|error| invalid(error.to_string()))?;
            // Auxiliary tasks can lower prediction matrices independently of the
            // target. Preserve source formats elsewhere and construct executable
            // modules from the exact selected formats, including companions.
            let mut formats = source_description
                .groups()
                .iter()
                .flat_map(|group| group.group().members())
                .filter_map(|member| {
                    args.weight_quantization_for(member.target())
                        .map(|format| (member.target().to_owned(), format))
                })
                .collect::<std::collections::HashMap<_, _>>();
            for task in tasks {
                if let Some(format) = task.executable().weight_quantization() {
                    formats.insert(task.name().to_owned(), format);
                } else {
                    formats.remove(task.name());
                }
            }
            let target_args =
                crate::nemotron_h::with_checkpoint_formats(args, formats).map_err(invalid)?;
            let description =
                crate::nemotron_h::LayeredModel::<B>::new(target_args.clone(), execution_context)
                    .map_err(|error| invalid(error.to_string()))?
                    .parameter_description(execution_context)
                    .map_err(|error| invalid(error.to_string()))?;
            let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                &description,
                tensor_rank,
            )
            .map_err(invalid)?;
            let geometry = crate::nemotron_h::local_geometry(&target_args, &layout)
                .map_err(|error| invalid(error.to_string()))?;
            let source_layout = if tasks.iter().any(|task| {
                matches!(
                    task.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                )
            }) {
                Some(std::sync::Arc::new(
                    crate::partitioned_execution::derive_partitioned_transform_source_layout(
                        &source_description,
                        &description,
                        tensor_rank,
                    )
                    .map_err(invalid)?,
                ))
            } else {
                None
            };
            let source_geometry = source_layout
                .as_ref()
                .map(|layout| crate::nemotron_h::local_geometry(args, layout))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?;
            let policies = args
                .mtp_policies()
                .map_err(|error| invalid(error.to_string()))?;
            let pattern = policies
                .len()
                .checked_div(extension.depth())
                .filter(|pattern| *pattern > 0)
                .ok_or_else(|| invalid("Nemotron-H MTP pattern is empty"))?;
            let descriptor =
                crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                    extension.complete_architecture().clone(),
                )
                .architecture_descriptor();
            let mut groups = Vec::with_capacity(extension.depth());
            for prediction in 0..extension.depth() {
                let mut units = Vec::with_capacity(pattern);
                for relative in 0..pattern {
                    let physical = prediction * pattern + relative;
                    let source = match &source_geometry {
                        Some(geometry) => {
                            let geometry = geometry.prediction_unit(physical).copied()
                                .ok_or_else(|| invalid(format!(
                                    "Nemotron-H prediction source unit {physical} has no local geometry"
                                )))?;
                            crate::nemotron_h::PredictionUnit::<B>::new_with_geometry(
                                args, prediction, relative, policies[physical], geometry, source_context,
                            )
                        }
                        None => crate::nemotron_h::PredictionUnit::<B>::new(
                            args, prediction, relative, source_context,
                        ),
                    }
                    .map_err(|error| invalid(error.to_string()))?;
                    let local_geometry =
                        geometry.prediction_unit(physical).copied().ok_or_else(|| {
                            invalid(format!(
                                "Nemotron-H prediction unit {physical} has no local geometry"
                            ))
                        })?;
                    let mut local = crate::nemotron_h::PredictionUnit::<B>::new_with_geometry(
                        &target_args,
                        prediction,
                        relative,
                        policies[physical],
                        local_geometry,
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
                        PreparedPredictionUnit::new(source, local, tasks)?
                            .with_source_layout(source_layout.clone()),
                    );
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
                layout: std::sync::Arc::new(layout),
                parameters: std::sync::Arc::new(description),
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
