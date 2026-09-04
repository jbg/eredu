//! Prepared neutral construction for embedded prediction extensions.

use std::collections::BTreeMap;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_core::{cache::LayerCachePolicy, ParallelRankTopology, ParallelTopology};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Tensor,
};
use eredu_runtime::{ArchitectureParameters, LocalModelLayout, StateLayout};

use crate::configuration::{
    PredictionExtensionKind, PredictionExtensionPlan, SafetensorsModelConfig,
};

/// One architecture-constructed neutral module and its exact checkpoint recipes.
pub struct PreparedPredictionUnit<M> {
    source: M,
    local: M,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
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
            if args.dspark.is_some() {
                return Err(invalid(
                    "DSpark requires a dedicated fused prediction extension",
                ));
            }
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
            Ok(PreparedPredictionExtension::DeepSeekV4 {
                layout,
                units,
                state,
            })
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
