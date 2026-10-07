//! Cold exact binding plans for independently leased TTS MoG units.

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
    tts_mog::{MogDimensions, TtsError},
    tts_mog_layered::mog_parameter_description,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum MogPreparationError {
    /// Selected communication lacks required bounded mechanisms.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// Communication requirements cannot be represented.
    #[error(transparent)]
    Communication(#[from] eredu_runtime::CommunicationManifestError),
    /// Mog geometry cannot be constructed.
    #[error(transparent)]
    Mog(#[from] TtsError),
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
    #[error("invalid MoG tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS MoG parameter {0} has inconsistent geometry")]
    Geometry(String),
}

#[cfg(test)]
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
            panic!("cold MoG selection must not acquire payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    #[test]
    fn released_mog_tp_preparation_retains_component_slices_and_replica_accounting() {
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = MogDimensions::from_config(&config.tts).unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let parameters = official_parameters();
        let prepare = || {
            PreparedMogBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap()
        };
        let prepared = prepare();
        let retained = prepared.source().clone();
        let global_resources = SpeechProviderResources::from_bindings(
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
                .source_metadata("tts_model.tts_model.backbone.norm.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        for (suffix, shape) in [
            ("mlp_stack.0.mlp.gate_proj.weight", vec![2304, 1152]),
            ("mlp_stack.0.mlp.down_proj.weight", vec![1152, 2304]),
            ("proj_mus.weight", vec![32768, 1152]),
            ("low_mat", vec![1024, 512, 32]),
            ("proj_logits.weight", vec![1024, 1152]),
            ("proj_else.weight", vec![512, 1152]),
            ("proj_logs.weight", vec![1, 1152]),
        ] {
            let name = format!("tts_model.tts_model.mog_head.{suffix}");
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
        // Pinned norms (two per residual block plus prediction), mixture logits,
        // latent residual and log-scale projections remain full per-rank replicas.
        let replicas = (2 * 3 * 1152 + 1152 + 1024 * 1152 + 512 * 1152 + 1152) * 4;
        assert_eq!(
            2 * resources.parameter_bytes,
            global_resources.parameter_bytes + replicas
        );
        assert!(resources.materialization_peak_bytes > resources.largest_unit_bytes);
        let (selected, same_layout) = selected.select_parallel(topology).unwrap();
        assert_eq!(same_layout, layout);
        assert!(Arc::ptr_eq(&retained, selected.source()));
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(MogPreparationError::Parallel(_))
        ));

        let opposite =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let (_, first) = prepare().select_parallel(opposite).unwrap();
        let mut mismatched = layout;
        let low = "tts_model.tts_model.mog_head.low_mat";
        mismatched.insert(low.into(), first.tensor(low).unwrap().clone());
        assert!(prepare().into_parallel(&mismatched).is_err());
    }
}

/// Cold bindings for each residual block, the prediction head, and selected bank.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedMogBindings {
    dimensions: MogDimensions,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedMogBindings {
    /// Projects exact MoG bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, MogPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            MogDimensions::from_config(&artifacts.plan().config().tts)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        dimensions: MogDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, MogPreparationError> {
        let description = mog_parameter_description(dimensions)?;
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(MogPreparationError::Geometry(
                    "unexpected static MoG owner".into(),
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
                    return Err(MogPreparationError::Geometry(name.into()));
                }
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(MogPreparationError::Geometry(name.into()));
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
            "voicechat/tts_mog/leased-units",
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
                            return Err(MogPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(MogPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, MogPreparationError>>()?;
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
    ) -> Result<(Self, LocalModelLayout), MogPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(MogPreparationError::Parallel(
                "MoG component selection requires a tensor-only topology".into(),
            ));
        }
        let description = mog_parameter_description(self.dimensions)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(MogPreparationError::Parallel)?;
        if let Some(selected) = self.local_layout.as_ref() {
            if selected != &layout {
                return Err(MogPreparationError::Parallel(
                    "retained MoG placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(mut self, layout: &LocalModelLayout) -> Result<Self, MogPreparationError> {
        if self.local_layout.is_some() {
            return Err(MogPreparationError::Parallel(
                "MoG bindings already have a tensor partition".into(),
            ));
        }
        super::tts_mog_layered::mog_local_dimensions(self.dimensions, layout)?;
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

    /// Validated MoG geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> MogDimensions {
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
        MogDimensions,
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

/// Validates bounded agreement and sum authority before native TP construction.
pub fn validate_parallel_communication(
    dimensions: MogDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), MogPreparationError> {
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
                dimensions.hidden.max(dimensions.latent) as usize,
                None,
            )?,
            true,
        )?],
    )?;
    if selected.members().len() > dimensions.intermediate.min(dimensions.rank) as usize {
        return Err(MogPreparationError::Parallel(
            "MoG tensor degree exceeds partitioned widths".into(),
        ));
    }
    Ok(())
}
