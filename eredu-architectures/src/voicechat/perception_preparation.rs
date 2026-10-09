//! Cold exact binding plans for independently leased perception units.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeError},
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
    perception::{PerceptionDimensions, PerceptionError},
    perception_layered::perception_parameter_description,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionPreparationError {
    /// Invalid floating execution profile or attempted precision reselection.
    #[error("invalid speech precision: {0}")]
    Precision(&'static str),
    /// Perception geometry cannot be constructed.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
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
    /// Exact recipes cannot realize the selected tensor placement.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// The shared planner rejected the selected tensor topology.
    #[error("invalid perception tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat perception parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for subsampling, each Conformer block, and final projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedPerceptionBindings {
    precision: super::precision::SpeechPrecision,
    dimensions: PerceptionDimensions,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedPerceptionBindings {
    /// Selects portable execution precision and exact converted bindings before materialization.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, PerceptionPreparationError> {
        if self.precision != Default::default() && self.precision != precision {
            return Err(PerceptionPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(PerceptionPreparationError::Precision)?;
        precision.bindings(&self.source, &mut self.units)?;
        self.precision = precision;
        Ok(self)
    }
    /// Retained floating storage and arithmetic policy.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }

    /// Projects complete perception bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, PerceptionPreparationError> {
        let source = artifacts.component_source(Component::Perception).ok_or(
            SpeechParameterError::MissingComponent(Component::Perception),
        )?;
        Self::from_parameters(
            PerceptionDimensions::from_config(&artifacts.plan().config().perception)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component and preprocessing payloads are excluded.
    pub fn from_parameters(
        dimensions: PerceptionDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, PerceptionPreparationError> {
        let description = perception_parameter_description(dimensions)?;
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(PerceptionPreparationError::Geometry(
                    "unexpected static perception owner".into(),
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
                    .filter(|p| p.component == Component::Perception)
                    .ok_or_else(|| SpeechParameterError::MissingParameter {
                        component: Component::Perception,
                        name: name.into(),
                    })?;
                if parameter.output.shape != member.global_shape()
                    || parameter.recipe.infer(source.as_ref())? != parameter.output
                {
                    return Err(PerceptionPreparationError::Geometry(name.into()));
                }
                let parameter = execution_parameter(parameter, &source)?;
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
            "voicechat/perception/leased-units",
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
                            return Err(PerceptionPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(PerceptionPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, PerceptionPreparationError>>()?;
        Ok(Self {
            precision: Default::default(),
            dimensions,
            layout,
            source,
            units,
            local_layout: None,
        })
    }

    /// Selects complete-stack tensor placement through the shared physical planner.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), PerceptionPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(PerceptionPreparationError::Parallel(
                "complete-stack perception selection requires a tensor-only topology".into(),
            ));
        }
        let description = perception_parameter_description(self.dimensions)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(PerceptionPreparationError::Parallel)?;
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies exact selected tensor placements without changing retained source authority.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, PerceptionPreparationError> {
        if self.local_layout.is_some() {
            return Err(PerceptionPreparationError::Parallel(
                "perception bindings already have a tensor partition".into(),
            ));
        }
        super::perception_layered::perception_local_dimensions(self.dimensions, layout)?;
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
            super::perception_layered::PerceptionBoundarySchema,
        >,
    ) -> Result<PreparedPerceptionPartitionBindings, PerceptionPreparationError> {
        super::perception_layered::validate_perception_partition(
            self.dimensions,
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
                        PerceptionPreparationError::Geometry(
                            "partition unit outside perception".into(),
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
            "voicechat/perception/partition-units",
            inputs,
        )?);
        Ok(PreparedPerceptionPartitionBindings {
            precision: self.precision,
            dimensions: self.dimensions,
            local_layout: self.local_layout,
            source,
            units,
            state_layout: partition.state().map(|state| state.layout().clone()),
        })
    }

    /// Exact retained tensor placement, present only after parallel selection.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }

    /// Validated perception geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
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
        PerceptionDimensions,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.dimensions, self.layout, self.source, self.units)
    }
}

/// Exact canonical units and restricted retained sources for a pipeline stage.
/// A parameter-only stage has no state and owns only its selected recipes.
pub struct PreparedPerceptionPartitionBindings {
    precision: super::precision::SpeechPrecision,
    dimensions: PerceptionDimensions,
    local_layout: Option<LocalModelLayout>,
    source: SharedCheckpointSource,
    units: Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    state_layout: Option<eredu_runtime::StateLayout>,
}
impl PreparedPerceptionPartitionBindings {
    /// Retained floating policy for partition construction.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }
    /// Global family equations; physical TP geometry is retained separately.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
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
    /// Selected block cache schedule; absent for a parameter-only stage.
    pub fn state_layout(&self) -> Option<&eredu_runtime::StateLayout> {
        self.state_layout.as_ref()
    }
    /// Consumes exact stage authority without rebasing canonical unit addresses.
    pub fn into_parts(
        self,
    ) -> (
        PerceptionDimensions,
        Option<LocalModelLayout>,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
        Option<eredu_runtime::StateLayout>,
    ) {
        (
            self.dimensions,
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
    use eredu_checkpoint::{
        recipe::RecipeDtype,
        store::{
            CheckpointLease, CheckpointSource, TensorMetadata, TensorReadRequest,
            WeightStoreDiagnostics,
        },
    };
    use eredu_core::{ParallelRankTopology, ParallelTopology};

    #[test]
    fn perception_precision_keeps_norms_and_history_f32_and_accounts_narrow_bindings() {
        use crate::voicechat::precision::{SpeechFloat, SpeechPrecision};
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = PerceptionDimensions::from_config(&config.perception).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let parameters = official_parameters();
        for compute in [SpeechFloat::F32, SpeechFloat::F16] {
            let policy = SpeechPrecision {
                weights: SpeechFloat::F16,
                compute,
            };
            let prepared = PreparedPerceptionBindings::from_parameters(
                dimensions,
                source.clone(),
                &parameters,
            )
            .unwrap()
            .with_precision(policy)
            .unwrap();
            for (name, dtype) in [
                (
                    "stt_model.perception.encoder.layers.0.feed_forward1.linear1.weight",
                    RecipeDtype::F16,
                ),
                (
                    "stt_model.perception.encoder.layers.0.norm_feed_forward1.weight",
                    RecipeDtype::F32,
                ),
                (
                    "stt_model.perception.encoder.layers.0.conv.pointwise_conv1.weight",
                    if compute == SpeechFloat::F32 {
                        RecipeDtype::F32
                    } else {
                        RecipeDtype::F16
                    },
                ),
            ] {
                let binding = prepared
                    .units
                    .iter()
                    .flatten()
                    .find(|b| b.name() == name)
                    .unwrap();
                let output = binding.source_recipe().infer(source.as_ref()).unwrap();
                assert_eq!(output.dtype, dtype);
                assert_eq!(output.byte_len(), binding.expected_bytes());
            }
            let resources =
                crate::voicechat::provider_residency::SpeechProviderResources::from_bindings(
                    prepared.source(),
                    prepared.unit_bindings(),
                    &parameters,
                )
                .unwrap();
            assert!(resources.parameter_bytes < 2_436_566_528);
            let topology =
                ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
            let (selected, _) = prepared.select_parallel(topology).unwrap();
            let name = "stt_model.perception.encoder.layers.0.feed_forward1.linear1.weight";
            let binding = selected
                .unit_bindings()
                .iter()
                .flatten()
                .find(|binding| binding.logical_target() == Some(name))
                .unwrap();
            let output = binding.source_recipe().infer(source.as_ref()).unwrap();
            assert_eq!(output.shape, [2048, 1024]);
            assert_eq!(output.dtype, RecipeDtype::F16);
            assert_eq!(binding.expected_bytes(), 2048 * 1024 * 2);
        }
        assert!(SpeechPrecision {
            weights: SpeechFloat::F16,
            compute: SpeechFloat::Bf16
        }
        .validate()
        .is_err());
    }

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
            panic!("cold perception selection must not acquire payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    #[test]
    fn released_perception_pipeline_preparation_restricts_stage_sources_and_requires_ingress() {
        use super::super::{
            perception_distribution::{
                PerceptionPipelinePreparationError, PreparedPerceptionPipelineBindings,
            },
            perception_layered::{perception_state_layout, PerceptionBoundarySchema},
        };
        use eredu_core::{CollectiveGroupId, CompletionCancellationMode};
        use eredu_runtime::{
            ArchitectureGroupKind, ArchitecturePartition, ArchitectureStatePartitionPlan,
            ArchitectureStatePartitionRule, CommunicationCompletionPolicy,
            CommunicationGroupDescriptor, CommunicationGroupRequirements, CommunicationManifest,
            CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
            LayeredPartitionDriver, PartitionOutputPublication, PartitionOwnership,
            PartitionedExecutionPlan, PipelineActivationDtype, PipelineWireContract,
        };
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = PerceptionDimensions::from_config(&config.perception).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let description = perception_parameter_description(dimensions).unwrap();
        let state = perception_state_layout(dimensions).unwrap();
        for (group, range, prefix) in [
            (0, 0..1, "stt_model.perception.encoder.pre_encode."),
            (1, 5..6, "stt_model.perception.encoder.layers.5."),
            (2, 0..1, "stt_model.perception.proj."),
        ] {
            let partition = ArchitecturePartition::from_description(
                &description,
                [(description.graph().groups()[group].id(), range.clone())],
                PartitionOwnership::new(group == 0, group == 2, std::iter::empty::<&str>())
                    .unwrap(),
                &state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(1, 0..dimensions.layers),
                ]),
                (),
                PerceptionBoundarySchema::new(dimensions),
            )
            .unwrap();
            let prepared = PreparedPerceptionBindings::from_parameters(
                dimensions,
                source.clone(),
                &official_parameters(),
            )
            .unwrap()
            .into_partition(&partition)
            .unwrap();
            assert_eq!(prepared.unit_bindings().len(), 1);
            let address = prepared.unit_bindings()[0].0;
            assert_eq!((address.group(), address.index()), (group, range.start));
            assert!(prepared
                .source()
                .source_keys()
                .iter()
                .all(|key| key.starts_with(prefix)));
            if group == 1 {
                assert_eq!(prepared.state_layout(), Some(&state.slice(5..6).unwrap()));
            } else {
                assert!(prepared.state_layout().is_none());
            }
            if group != 2 {
                assert!(matches!(
                    prepared
                        .source()
                        .source_metadata("stt_model.perception.proj.weight"),
                    Err(StoreError::UnauthorizedTensor { .. })
                ));
                continue;
            }
            let mut drivers = vec![None, None, None];
            drivers[2] = Some(
                LayeredPartitionDriver::new_with_state_ownership(&partition, 2, 0..1, false)
                    .unwrap(),
            );
            let session = CollectiveGroupId::new(1);
            let manifest = CommunicationManifest::new(
                2,
                1,
                vec![CommunicationGroupDescriptor::new(
                    session,
                    0,
                    vec![0, 1],
                    Some(1),
                    CommunicationGroupRequirements::new([
                        CommunicationOperationRequirement::failure_agreement(true),
                        CommunicationOperationRequirement::tensors(
                            CommunicationOperation::Broadcast,
                            vec![eredu_core::checkpoint::TensorDtype::F32],
                            CommunicationTensorLimits::new(1, 3, 1 << 20, None).unwrap(),
                            true,
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )
                .unwrap()],
                vec![],
            )
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
            let plan = PartitionedExecutionPlan::new(
                description.graph().clone(),
                vec![
                    (ArchitectureGroupKind::AudioEncoder, false),
                    (ArchitectureGroupKind::AudioEncoder, false),
                    (ArchitectureGroupKind::Projector, false),
                ],
                drivers,
                vec![],
                Some(PartitionOutputPublication {
                    group: session,
                    owner_rank: 1,
                }),
                Some(session),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            assert!(matches!(
                PreparedPerceptionPipelineBindings::new(
                    prepared,
                    partition,
                    manifest,
                    plan,
                    None,
                    PipelineActivationDtype::Float32
                ),
                Err(PerceptionPipelinePreparationError::Contract(
                    "routes differ from the stage's single ingress and egress"
                ))
            ));
        }
    }

    #[test]
    fn released_perception_tp_preparation_retains_exact_sources_and_full_replicas() {
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = PerceptionDimensions::from_config(&config.perception).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let prepared =
            PreparedPerceptionBindings::from_parameters(dimensions, source, &official_parameters())
                .unwrap();
        let retained = prepared.source().clone();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
        let (selected, layout) = prepared.select_parallel(topology).unwrap();
        assert!(Arc::ptr_eq(&retained, selected.source()));
        // Pinned encoder: width 1024, eight heads, FFN 4096. Convolutions,
        // normalization, subsampling and language projection retain full replicas.
        for (name, shape) in [
            (
                "encoder.layers.0.self_attn.linear_q.weight",
                vec![512, 1024],
            ),
            (
                "encoder.layers.0.self_attn.linear_pos.weight",
                vec![512, 1024],
            ),
            (
                "encoder.layers.0.self_attn.linear_out.weight",
                vec![1024, 512],
            ),
            ("encoder.layers.0.self_attn.pos_bias_u", vec![4, 128]),
            (
                "encoder.layers.0.feed_forward1.linear1.weight",
                vec![2048, 1024],
            ),
            (
                "encoder.layers.0.feed_forward2.linear2.weight",
                vec![1024, 2048],
            ),
            (
                "encoder.layers.0.conv.pointwise_conv1.weight",
                vec![2048, 1024, 1],
            ),
            ("encoder.layers.0.norm_self_att.weight", vec![1024]),
            ("encoder.pre_encode.conv.0.weight", vec![256, 1, 3, 3]),
            ("proj.weight", vec![4480, 1024]),
        ] {
            let name = format!("stt_model.perception.{name}");
            let binding = selected
                .unit_bindings()
                .iter()
                .flatten()
                .find(|binding| binding.name() == name)
                .unwrap();
            assert_eq!(binding.logical_target(), Some(name.as_str()));
            let output = binding
                .source_recipe()
                .infer(selected.source().as_ref())
                .unwrap();
            assert_eq!(output.shape, shape, "{name}");
            assert_eq!(output.dtype, RecipeDtype::F32);
            assert_eq!(
                binding.expected_bytes(),
                shape.iter().product::<usize>() as u64 * 4
            );
        }
        assert!(matches!(
            selected
                .source()
                .source_metadata("tts_model.tts_model.backbone.norm.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        // A same-sized relative-bias shard from the other rank is not correlated
        // with this rank's query, key, value and position projection slices.
        let mut mismatched = layout.clone();
        let target = "stt_model.perception.encoder.layers.0.self_attn.pos_bias_u";
        let original = layout.tensor(target).unwrap();
        mismatched.insert(
            target.into(),
            eredu_runtime::LocalTensorLayout::new(
                original.logical_name(),
                original.role(),
                original.global_shape().to_vec(),
                original.local_shape().to_vec(),
                eredu_runtime::TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 4,
                },
                Some(8),
                Some(0..4),
                false,
            ),
        );
        assert!(
            super::super::perception_layered::perception_local_dimensions(dimensions, &mismatched)
                .is_err()
        );
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(PerceptionPreparationError::Parallel(_))
        ));
    }
    #[test]
    fn coupled_pipeline_preparation_preserves_sources_and_rejects_incompatible_authority() {
        use crate::voicechat::{
            perception_distribution::PreparedPerceptionPipelineBindings,
            perception_layered::{perception_state_layout, PerceptionBoundarySchema},
            perception_rnnt_preparation::{
                PerceptionRnntPreparationError, PreparedPerceptionRnntPipelineBindings,
            },
            rnnt::RnntDimensions,
            rnnt_predictor::{rnnt_predictor_parameter_description, rnnt_predictor_state_layout},
            rnnt_predictor_distribution::PreparedRnntPredictorPipelineBindings,
            rnnt_predictor_partition::{RnntPredictorBoundarySchema, RnntPredictorPipelineCuts},
            rnnt_predictor_preparation::PreparedRnntPredictorBindings,
            rnnt_preparation::PreparedRnntBindings,
            rnnt_projection_preparation::PreparedRnntProjectionBindings,
        };
        use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
        use eredu_runtime::*;
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let perception_dimensions = PerceptionDimensions::from_config(&config.perception).unwrap();
        let rnnt_dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let parameters = official_parameters();
        let session = CollectiveGroupId::new(77);
        let manifest = |world, decision_dtype| {
            CommunicationManifest::new(
                world,
                0,
                vec![CommunicationGroupDescriptor::new(
                    session,
                    0,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([
                        CommunicationOperationRequirement::failure_agreement(true),
                        CommunicationOperationRequirement::tensors(
                            CommunicationOperation::Broadcast,
                            [TensorDtype::F32, decision_dtype],
                            CommunicationTensorLimits::new(1, 3, 1 << 20, None).unwrap(),
                            true,
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )
                .unwrap()],
                vec![],
            )
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
        };
        let prepared = || {
            let description = perception_parameter_description(perception_dimensions).unwrap();
            let partition = ArchitecturePartition::from_description(
                &description,
                [
                    (description.graph().groups()[0].id(), 0..1),
                    (
                        description.graph().groups()[1].id(),
                        0..perception_dimensions.layers,
                    ),
                    (description.graph().groups()[2].id(), 0..1),
                ],
                PartitionOwnership::new(true, true, std::iter::empty::<&str>()).unwrap(),
                &perception_state_layout(perception_dimensions).unwrap(),
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(1, 0..perception_dimensions.layers),
                ]),
                (),
                PerceptionBoundarySchema::new(perception_dimensions),
            )
            .unwrap();
            let plan = PartitionedExecutionPlan::new(
                description.graph().clone(),
                vec![
                    (ArchitectureGroupKind::AudioEncoder, false),
                    (ArchitectureGroupKind::AudioEncoder, false),
                    (ArchitectureGroupKind::Projector, false),
                ],
                (0..3)
                    .map(|group| {
                        Some(
                            LayeredPartitionDriver::new_with_state_ownership(
                                &partition,
                                group,
                                if group == 1 {
                                    0..perception_dimensions.layers
                                } else {
                                    0..1
                                },
                                group == 1,
                            )
                            .unwrap(),
                        )
                    })
                    .collect(),
                vec![],
                Some(PartitionOutputPublication {
                    group: session,
                    owner_rank: 0,
                }),
                Some(session),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let perception = PreparedPerceptionPipelineBindings::new(
                PreparedPerceptionBindings::from_parameters(
                    perception_dimensions,
                    source.clone(),
                    &parameters,
                )
                .unwrap()
                .into_partition(&partition)
                .unwrap(),
                partition,
                manifest(1, TensorDtype::I32),
                plan,
                None,
                PipelineActivationDtype::Float32,
            )
            .unwrap();
            let description = rnnt_predictor_parameter_description(rnnt_dimensions).unwrap();
            let partition = ArchitecturePartition::from_description(
                &description,
                [(description.graph().groups()[0].id(), 0..2)],
                PartitionOwnership::new(true, true, std::iter::empty::<&str>()).unwrap(),
                &rnnt_predictor_state_layout(rnnt_dimensions).unwrap(),
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..2),
                ]),
                (),
                RnntPredictorBoundarySchema::new(rnnt_dimensions).unwrap(),
            )
            .unwrap();
            let plan = PartitionedExecutionPlan::new(
                description.graph().clone(),
                vec![(ArchitectureGroupKind::Recurrent, false)],
                vec![Some(
                    LayeredPartitionDriver::new_with_state_ownership(&partition, 0, 0..2, true)
                        .unwrap(),
                )],
                vec![],
                Some(PartitionOutputPublication {
                    group: session,
                    owner_rank: 0,
                }),
                Some(session),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let rnnt = || {
                PreparedRnntBindings::from_parameters(rnnt_dimensions, source.clone(), &parameters)
                    .unwrap()
            };
            let predictor = PreparedRnntPredictorPipelineBindings::new(
                PreparedRnntPredictorBindings::from_rnnt(rnnt(), &partition).unwrap(),
                partition,
                RnntPredictorPipelineCuts::new(rnnt_dimensions, []).unwrap(),
                manifest(1, TensorDtype::I32),
                plan,
                None,
                PipelineActivationDtype::Float32,
            )
            .unwrap();
            (
                perception,
                predictor,
                PreparedRnntProjectionBindings::from_rnnt(rnnt()).unwrap(),
            )
        };
        let (perception, predictor, projections) = prepared();
        let retained = (
            perception.bindings().source().clone(),
            predictor.bindings().source().clone(),
            projections.source().clone(),
        );
        let aggregate = PreparedPerceptionRnntPipelineBindings::new(
            perception,
            predictor,
            projections,
            manifest(1, TensorDtype::I32),
            session,
            0,
            None,
        )
        .unwrap();
        assert_eq!(aggregate.resources().predictor_parameter_bytes, 26_255_360);
        assert_eq!(aggregate.resources().projection_parameter_bytes, 9_517_060);
        assert_eq!(aggregate.resources().perception_cache_bytes, 7_667_712);
        assert_eq!(aggregate.resources().prediction_cache_bytes, 10_240);
        let (perception, predictor, projections, ..) = aggregate.into_parts();
        assert!(Arc::ptr_eq(perception.bindings().source(), &retained.0));
        assert!(Arc::ptr_eq(predictor.bindings().source(), &retained.1));
        assert!(Arc::ptr_eq(projections.source(), &retained.2));
        for (source, denied) in [
            (
                perception.bindings().source(),
                "stt_model.rnnt_joint.enc.weight",
            ),
            (
                predictor.bindings().source(),
                "stt_model.rnnt_decoder.prediction.embed.weight",
            ),
            (
                projections.source(),
                "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l0",
            ),
        ] {
            assert!(matches!(
                source.source_metadata(denied),
                Err(StoreError::UnauthorizedTensor { .. })
            ));
        }
        for (world, dtype) in [(2, TensorDtype::I32), (1, TensorDtype::F16)] {
            let (perception, predictor, projections) = prepared();
            let error = PreparedPerceptionRnntPipelineBindings::new(
                perception,
                predictor,
                projections,
                manifest(world, dtype),
                session,
                0,
                None,
            )
            .err()
            .unwrap();
            assert!(matches!(
                error,
                PerceptionRnntPreparationError::Contract(_)
                    | PerceptionRnntPreparationError::Capability(_)
            ));
        }
        // Neural payload support alone cannot realize the projection cohort's
        // coordinator: its scalar control publication must also be exact I32.
        let cohort = CollectiveGroupId::new(78);
        for dtype in [TensorDtype::F16, TensorDtype::I32] {
            let base = manifest(1, TensorDtype::I32);
            let mut groups = base.groups().to_vec();
            groups.push(
                CommunicationGroupDescriptor::new(
                    cohort,
                    1,
                    vec![0],
                    Some(0),
                    CommunicationGroupRequirements::new([
                        CommunicationOperationRequirement::failure_agreement(true),
                        CommunicationOperationRequirement::tensors(
                            CommunicationOperation::AllReduceSum,
                            [TensorDtype::F32],
                            CommunicationTensorLimits::new(1, 2, 16384, None).unwrap(),
                            true,
                        )
                        .unwrap(),
                        CommunicationOperationRequirement::tensors(
                            CommunicationOperation::AllGatherUneven,
                            [TensorDtype::F32],
                            CommunicationTensorLimits::new(1, 2, 16384, None)
                                .unwrap()
                                .with_output_tensor_elements(16384)
                                .unwrap(),
                            true,
                        )
                        .unwrap(),
                        CommunicationOperationRequirement::tensors(
                            CommunicationOperation::Broadcast,
                            [dtype.clone()],
                            CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
                            true,
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                )
                .unwrap(),
            );
            let coordination = CommunicationManifest::new(1, 0, groups, vec![])
                .unwrap()
                .with_completion_policy(base.completion_policy().unwrap());
            let (perception, predictor, projections) = prepared();
            let result = PreparedPerceptionRnntPipelineBindings::new(
                perception,
                predictor,
                projections,
                coordination,
                session,
                0,
                Some(cohort),
            );
            if dtype == TensorDtype::I32 {
                assert_eq!(result.unwrap().projection_owner(), Some(0));
            } else {
                assert!(matches!(
                    result,
                    Err(PerceptionRnntPreparationError::Capability(_))
                ));
            }
        }
    }
}
