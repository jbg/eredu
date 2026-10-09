//! Cold exact binding plans for independently leased TTS transformer units.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    bindings_from_recipe_set, place_weight_bindings, BindingPlacementError, ExecutionUnitLayout,
    LocalModelLayout, ParameterGroupOwner, RecipeBindingError, WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{execution_parameter, SpeechParameterError},
    tts_transformer::TransformerConfig,
    tts_transformer_layered::tts_transformer_parameter_description,
    tts_transformer_layered::TransformerError,
};

/// Selects one independently executed EAR-TTS transformer from retained artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsTransformerKind {
    /// Causal speech backbone.
    Causal,
    /// Bidirectional character encoder.
    CharacterEncoder,
}

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum TtsTransformerPreparationError {
    /// Invalid floating execution profile or attempted precision reselection.
    #[error("invalid speech precision: {0}")]
    Precision(&'static str),
    /// Transformer geometry cannot be constructed.
    #[error(transparent)]
    Transformer(#[from] TransformerError),
    /// A required parameter is missing from the selected component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// The exact rank-local recipe placement is invalid.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Shared partition planning rejected the selected rank topology.
    #[error("invalid TTS transformer tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS transformer parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for each decoder block and final normalization.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedTtsTransformerBindings {
    config: TransformerConfig,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedTtsTransformerBindings {
    /// Selects exact narrow projections; all norms and cache storage remain F32.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, TtsTransformerPreparationError> {
        if self.config.precision() != Default::default() && self.config.precision() != precision {
            return Err(TtsTransformerPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(TtsTransformerPreparationError::Precision)?;
        precision.bindings(&self.source, &mut self.units)?;
        self.config = self.config.with_precision(precision);
        Ok(self)
    }

    /// Projects exact transformer bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
        kind: TtsTransformerKind,
    ) -> Result<Self, TtsTransformerPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            TransformerConfig::new(match kind {
                TtsTransformerKind::Causal => &artifacts.plan().config().tts.backbone,
                TtsTransformerKind::CharacterEncoder => {
                    &artifacts.plan().config().tts.character_encoder
                }
            })
            .map_err(TransformerError::from)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        config: TransformerConfig,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, TtsTransformerPreparationError> {
        // This selected path executes ordinary F32 projections after bounded
        // recipe conversion, independently of the checkpoint storage format.
        let config = config.with_formats(BTreeMap::new());
        let description = tts_transformer_parameter_description(&config)?;
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(TtsTransformerPreparationError::Geometry(
                    "unexpected static transformer owner".into(),
                ));
            };
            let group = description
                .graph()
                .group_index(group.as_str())
                .expect("validated parameter ownership");
            let ordinal = layout
                .ordinal(group, *global_unit)
                .expect("validated parameter unit");
            for member in owned.members() {
                let name = member.target();
                let parameter = parameters
                    .get(name)
                    .filter(|p| p.component == Component::Tts)
                    .ok_or_else(|| SpeechParameterError::MissingParameter {
                        component: Component::Tts,
                        name: name.into(),
                    })?;
                if parameter.output.shape != member.global_shape()
                    || parameter.recipe.infer(source.as_ref())? != parameter.output
                {
                    return Err(TtsTransformerPreparationError::Geometry(name.into()));
                }
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(TtsTransformerPreparationError::Geometry(name.into()));
                }
                unit_recipes[ordinal].insert(name.into(), parameter.recipe);
            }
        }
        let inputs = unit_recipes
            .iter()
            .flat_map(|unit| unit.values())
            .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
            .collect();
        let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
            source,
            "voicechat/tts_transformer/leased-units",
            inputs,
        )?);
        let units = unit_recipes
            .into_iter()
            .map(|recipes| {
                let canonical_targets = recipes
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>();
                let recipes = AtomicRecipeSet::new(source.as_ref(), recipes)?;
                bindings_from_recipe_set(source.as_ref(), recipes)?
                    .into_iter()
                    .map(|binding| {
                        let target = binding.name().to_owned();
                        if !canonical_targets.contains(&target) {
                            return Err(TtsTransformerPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(TtsTransformerPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, TtsTransformerPreparationError>>()?;
        Ok(Self {
            config,
            layout,
            source,
            units,
            local_layout: None,
        })
    }

    /// Selects complete-stack tensor placement using the shared physical planner.
    /// Component construction owns no pipeline transport or expert/data replicas;
    /// those dimensions require an enclosing architecture partition selection.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), TtsTransformerPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(TtsTransformerPreparationError::Parallel(
                "complete-stack component selection requires a tensor-only topology".into(),
            ));
        }
        let description = tts_transformer_parameter_description(&self.config)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(TtsTransformerPreparationError::Parallel)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Applies an already selected rank-local layout to the exact retained recipes.
    /// Shape validation precedes recipe lowering and source authority is preserved.
    /// This consumes global bindings and rejects a second partition selection.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, TtsTransformerPreparationError> {
        if self.local_layout.is_some() {
            return Err(TtsTransformerPreparationError::Parallel(
                "transformer bindings already have a selected tensor partition".into(),
            ));
        }
        super::tts_transformer_layered::transformer_local_configs(&self.config, layout)?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.local_layout = Some(layout.clone());
        Ok(self)
    }

    /// Narrows retained source authority and exact bindings to one selected stage.
    /// Unit addresses remain canonical; this result is not a zero-based complete
    /// provider layout and must be consumed by a partition traversal.
    pub fn into_partition<G>(
        self,
        partition: &eredu_runtime::ArchitecturePartition<
            G,
            super::tts_transformer_layered::TransformerBoundarySchema,
        >,
    ) -> Result<PreparedTtsTransformerPartitionBindings, TtsTransformerPreparationError> {
        super::tts_transformer_layered::validate_transformer_partition(
            &self.config,
            self.local_layout.as_ref(),
            partition,
        )?;
        let selected = partition
            .units()
            .map(|address| {
                self.layout
                    .ordinal(address.group(), address.index())
                    .map(|ordinal| (ordinal, address))
                    .ok_or_else(|| {
                        TtsTransformerPreparationError::Geometry(
                            "partition unit outside transformer".into(),
                        )
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let units = self
            .units
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, bindings)| {
                selected
                    .get(&ordinal)
                    .copied()
                    .map(|address| (address, bindings))
            })
            .collect::<Vec<_>>();
        let inputs = units
            .iter()
            .flat_map(|(_, bindings)| bindings)
            .flat_map(|binding| {
                binding
                    .source_recipe()
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        let source = Arc::new(RestrictedCheckpointSource::including(
            self.source,
            "voicechat/tts_transformer/partition-units",
            inputs,
        )?);
        Ok(PreparedTtsTransformerPartitionBindings {
            config: self.config,
            local_layout: self.local_layout,
            source,
            units,
            state_layout: partition.state().map(|state| state.layout().clone()),
        })
    }

    /// Validated transformer geometry; no other speech stack is selected.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.units
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        TransformerConfig,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.config, self.layout, self.source, self.units)
    }
}

/// Exact canonical units and restricted retained sources for a pipeline stage.
/// A normalization-only stage has no state and owns only its normalization recipe.
pub struct PreparedTtsTransformerPartitionBindings {
    config: TransformerConfig,
    local_layout: Option<LocalModelLayout>,
    source: SharedCheckpointSource,
    units: Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    state_layout: Option<eredu_runtime::StateLayout>,
}
impl PreparedTtsTransformerPartitionBindings {
    /// Global family equations; physical TP geometry is retained separately.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
    /// Exact selected physical layout when tensor parallelism is also selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }
    /// Source authority limited to the actual selected unit recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Canonical addresses and their exact resident/bounded materialization recipes.
    pub fn unit_bindings(&self) -> &[(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)] {
        &self.units
    }
    /// Selected block cache schedule; absent for a standalone normalization stage.
    pub fn state_layout(&self) -> Option<&eredu_runtime::StateLayout> {
        self.state_layout.as_ref()
    }
    /// Consumes exact stage authority without rebasing canonical unit addresses.
    pub fn into_parts(
        self,
    ) -> (
        TransformerConfig,
        Option<LocalModelLayout>,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
        Option<eredu_runtime::StateLayout>,
    ) {
        (
            self.config,
            self.local_layout,
            self.source,
            self.units,
            self.state_layout,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voicechat::{
        artifact::tests::official_parameters, checkpoint, config::VoiceChatConfig,
    };
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    };
    use eredu_core::{ParallelRankTopology, ParallelTopology};

    struct HeaderSource(BTreeMap<String, TensorMetadata>);
    impl CheckpointSource for HeaderSource {
        fn source_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("cold transformer partitioning must not open weight payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    #[test]
    fn released_transformer_tp_preparation_retains_source_and_partitions_recipes() {
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let prepared = PreparedTtsTransformerBindings::from_parameters(
            TransformerConfig::new(&config.tts.backbone).unwrap(),
            source,
            &official_parameters(),
        )
        .unwrap();
        let exact_source = prepared.source().clone();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
        let (selected, layout) = prepared.select_parallel(topology).unwrap();
        assert!(Arc::ptr_eq(&exact_source, selected.source()));
        // Released geometry is hidden 1152, Q/K/V 16 heads of width 72,
        // and intermediate 4608. TP2 shards projections and replicates gains.
        for (suffix, shape) in [
            ("layers.0.self_attn.q_proj.weight", vec![576, 1152]),
            ("layers.0.self_attn.k_proj.weight", vec![576, 1152]),
            ("layers.0.mlp.down_proj.weight", vec![1152, 2304]),
            ("layers.0.input_layernorm.weight", vec![1152]),
            ("norm.weight", vec![1152]),
        ] {
            let name = format!("tts_model.tts_model.backbone.{suffix}");
            let binding = selected
                .unit_bindings()
                .iter()
                .flatten()
                .find(|b| b.name() == name)
                .unwrap();
            assert_eq!(binding.logical_target(), Some(name.as_str()));
            let output = binding
                .source_recipe()
                .infer(selected.source().as_ref())
                .unwrap();
            assert_eq!(output.shape, shape);
            assert_eq!(output.dtype, RecipeDtype::F32);
            assert_eq!(
                binding.expected_bytes(),
                shape.iter().product::<usize>() as u64 * 4
            );
        }
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(TtsTransformerPreparationError::Parallel(_))
        ));
    }
    #[test]
    fn released_transformer_partition_restricts_canonical_units_and_norm_has_no_state() {
        use super::super::tts_transformer_layered::TransformerBoundarySchema;
        use crate::decoder::Config;
        use eredu_runtime::{
            ArchitecturePartition, ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule,
            PartitionOwnership,
        };
        let released =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let config = TransformerConfig::new(&released.tts.backbone).unwrap();
        let parameters = official_parameters();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let description = tts_transformer_parameter_description(&config).unwrap();
        let state = crate::decoder::state_layout(&config).unwrap();
        let state_plan =
            ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..config.geometry().num_hidden_layers,
            )]);
        for (group, range, output) in [(0, 2..3, false), (1, 0..1, true)] {
            let selected = ArchitecturePartition::from_description(
                &description,
                [(
                    description.unit_layout().group_id(group).unwrap().as_str(),
                    range.clone(),
                )],
                PartitionOwnership::new(false, output, std::iter::empty::<&str>()).unwrap(),
                &state,
                &state_plan,
                (),
                TransformerBoundarySchema::new(&config),
            )
            .unwrap();
            let prepared = PreparedTtsTransformerBindings::from_parameters(
                config.clone(),
                source.clone(),
                &parameters,
            )
            .unwrap()
            .into_partition(&selected)
            .unwrap();
            assert_eq!(prepared.unit_bindings().len(), 1);
            let (address, bindings) = &prepared.unit_bindings()[0];
            assert_eq!((address.group(), address.index()), (group, range.start));
            if output {
                assert!(prepared.state_layout().is_none());
                assert_eq!(
                    prepared.source().source_keys(),
                    vec![format!("{}.norm.weight", config.parameter_root())]
                );
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].expected_bytes(), 1152 * 4);
            } else {
                assert_eq!(
                    prepared.state_layout().unwrap(),
                    &state.slice(2..3).unwrap()
                );
                assert!(prepared
                    .source()
                    .source_keys()
                    .iter()
                    .all(|key| key.starts_with("tts_model.tts_model.backbone.layers.2.")));
                assert!(matches!(
                    prepared
                        .source()
                        .source_metadata("tts_model.tts_model.backbone.norm.weight"),
                    Err(StoreError::UnauthorizedTensor { .. })
                ));
            }
        }
    }
}
