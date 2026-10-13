//! Cold exact binding plans for independently leased TTS RVQ units.

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
    tts_mog::TtsError,
    tts_rvq::RvqDimensions,
    tts_rvq_layered::rvq_parameter_description,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum RvqPreparationError {
    /// Selected communication lacks membership, bounded completion or required operations.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// An exact communication requirement is malformed.
    #[error(transparent)]
    Communication(#[from] eredu_runtime::CommunicationManifestError),
    /// RVQ geometry cannot be constructed.
    #[error(transparent)]
    Rvq(#[from] TtsError),
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
    /// Shared physical placement cannot be applied to an exact recipe.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid RVQ tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS RVQ parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for the stacked feedback codebook bank and shared latent projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedRvqBindings {
    dimensions: RvqDimensions,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedRvqBindings {
    /// Projects exact RVQ bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, RvqPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            RvqDimensions::from_config(&artifacts.plan().config().tts)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        dimensions: RvqDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, RvqPreparationError> {
        let description = rvq_parameter_description(dimensions)?;
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(RvqPreparationError::Geometry(
                    "unexpected static RVQ owner".into(),
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
                    return Err(RvqPreparationError::Geometry(name.into()));
                }
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(RvqPreparationError::Geometry(name.into()));
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
            "voicechat/tts_rvq/leased-units",
            inputs,
        )?);
        let units = unit_recipes
            .into_iter()
            .map(|recipes| {
                let targets = recipes
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>();
                let recipes = AtomicRecipeSet::new(source.as_ref(), recipes)?;
                bindings_from_recipe_set(source.as_ref(), recipes)?
                    .into_iter()
                    .map(|binding| {
                        let target = binding.name().to_owned();
                        if !targets.contains(&target) {
                            return Err(RvqPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(RvqPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, RvqPreparationError>>()?;
        Ok(Self {
            dimensions,
            layout,
            source,
            units,
            local_layout: None,
        })
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RvqPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(RvqPreparationError::Parallel(
                "RVQ component selection requires a tensor-only topology".into(),
            ));
        }
        let description = rvq_parameter_description(self.dimensions)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(RvqPreparationError::Parallel)?;
        if let Some(selected) = self.local_layout.as_ref() {
            if selected != &layout {
                return Err(RvqPreparationError::Parallel(
                    "retained RVQ placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(mut self, layout: &LocalModelLayout) -> Result<Self, RvqPreparationError> {
        if self.local_layout.is_some() {
            return Err(RvqPreparationError::Parallel(
                "RVQ bindings already have a tensor partition".into(),
            ));
        }
        super::tts_rvq_layered::rvq_local_dimensions(self.dimensions, layout)?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.local_layout = Some(layout.clone());
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }

    /// Validated RVQ geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> RvqDimensions {
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
        RvqDimensions,
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        (
            self.dimensions,
            self.local_layout,
            self.layout,
            self.source,
            self.units,
        )
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::voicechat::{
        artifact::tests::official_parameters, checkpoint, config::VoiceChatConfig,
        provider_residency::SpeechProviderResources,
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
            panic!("cold RVQ selection must not acquire payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    #[test]
    fn released_rvq_tp_preparation_retains_coherent_latent_slices_and_conversion_bounds() {
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = RvqDimensions::from_config(&config.tts).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let parameters = official_parameters();
        let prepare = || {
            PreparedRvqBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap()
        };
        let prepared = prepare();
        let retained = prepared.source().clone();
        let global = SpeechProviderResources::from_bindings(
            prepared.source(),
            prepared.unit_bindings(),
            &parameters,
        )
        .unwrap();
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
        let (selected, layout) = prepared.select_parallel(topology).unwrap();
        assert!(Arc::ptr_eq(&retained, selected.source()));
        assert!(matches!(
            selected
                .source()
                .source_metadata("tts_model.tts_model.mog_head.low_mat"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        for (name, shape) in [
            ("tts_model.tts_model.rvq_embs", vec![31, 1024, 256]),
            ("tts_model.tts_model.embed_code.weight", vec![1152, 256]),
        ] {
            let binding = selected
                .unit_bindings()
                .iter()
                .flatten()
                .find(|b| b.name() == name)
                .unwrap();
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
        let resources = SpeechProviderResources::from_bindings(
            selected.source(),
            selected.unit_bindings(),
            &parameters,
        )
        .unwrap();
        assert_eq!(
            resources.parameter_bytes,
            (31 * 1024 * 256 + 1152 * 256) * 4
        );
        assert_eq!(2 * resources.parameter_bytes, global.parameter_bytes);
        assert!(resources.materialization_peak_bytes > resources.largest_unit_bytes);
        let (selected, same_layout) = selected.select_parallel(topology).unwrap();
        assert_eq!(same_layout, layout);
        assert!(Arc::ptr_eq(&retained, selected.source()));
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(RvqPreparationError::Parallel(_))
        ));

        let opposite =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let (_, first) = prepare().select_parallel(opposite).unwrap();
        let mut mismatched = layout;
        let projection = "tts_model.tts_model.embed_code.weight";
        mismatched.insert(projection.into(), first.tensor(projection).unwrap().clone());
        assert!(prepare().into_parallel(&mismatched).is_err());
    }
}

/// Validates the exact operations needed by guarded latent-sharded feedback.
/// Actual batch/time limits are checked against retained authority at execution.
pub fn validate_parallel_communication(
    dimensions: RvqDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), RvqPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    dimensions.validate()?;
    let selected = eredu_runtime::replicated_execution::validate_collective_coordination(
        manifest,
        group,
        [CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(
                1,
                3,
                dimensions
                    .cardinality
                    .max(dimensions.latent)
                    .max(dimensions.hidden) as usize,
                None,
            )?,
            true,
        )?],
    )?;
    if selected.members().len() > dimensions.latent as usize {
        return Err(RvqPreparationError::Parallel(
            "tensor degree exceeds latent width".into(),
        ));
    }
    Ok(())
}
