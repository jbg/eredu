//! Cold exact binding plans for independently leased function head units.

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
    channels::{ChannelError, TEXT_VOCABULARY_SIZE},
    checkpoint::Component,
    function_head_layered::{function_head_parameter_description, FunctionHeadDimensions},
    parameters::SpeechParameterError,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum FunctionHeadPreparationError {
    /// Function head geometry cannot be constructed.
    #[error(transparent)]
    FunctionHead(#[from] ChannelError),
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
    /// Tensor topology or a repeated selection is invalid.
    #[error("invalid function-head tensor partition: {0}")]
    Parallel(String),
    /// Selected communication cannot supply bounded failure agreement or gathering.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// An exact tensor communication requirement is malformed.
    #[error(transparent)]
    Manifest(#[from] eredu_runtime::CommunicationManifestError),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat function head parameter {0} has inconsistent geometry")]
    Geometry(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voicechat::{artifact::tests::official_parameters, checkpoint};
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
            panic!("cold function-head selection must not acquire payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    #[test]
    fn function_head_collective_admission_requires_failure_and_exact_f32_gather_bounds() {
        use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
        use eredu_runtime::{
            CommunicationCompletionPolicy, CommunicationGroupDescriptor,
            CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
            CommunicationOperationRequirement, CommunicationTensorLimits,
        };
        let id = CollectiveGroupId::new(71);
        let manifest = |failure: bool, gather: Option<(usize, usize, TensorDtype)>| {
            let mut operations = Vec::new();
            if failure {
                operations.push(CommunicationOperationRequirement::failure_agreement(true));
            }
            if let Some((local, output, dtype)) = gather {
                operations.push(
                    CommunicationOperationRequirement::tensors(
                        CommunicationOperation::AllGatherUneven,
                        [dtype],
                        CommunicationTensorLimits::new(1, 3, local, None)
                            .unwrap()
                            .with_output_tensor_elements(output)
                            .unwrap(),
                        true,
                    )
                    .unwrap(),
                );
            }
            CommunicationManifest::new(
                2,
                0,
                vec![CommunicationGroupDescriptor::new(
                    id,
                    0,
                    vec![0, 1],
                    Some(0),
                    CommunicationGroupRequirements::new(operations).unwrap(),
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
        let dimensions = FunctionHeadDimensions::new(4, 23).unwrap();
        let accepted = manifest(true, Some((12, 23, TensorDtype::F32)));
        validate_parallel_communication(dimensions, &accepted, id).unwrap();
        for (label, rejected) in [
            (
                "missing failure agreement",
                manifest(false, Some((12, 23, TensorDtype::F32))),
            ),
            ("missing gather", manifest(true, None)),
            (
                "short local shard",
                manifest(true, Some((11, 23, TensorDtype::F32))),
            ),
            (
                "short gathered output",
                manifest(true, Some((12, 22, TensorDtype::F32))),
            ),
            (
                "wrong execution precision",
                manifest(true, Some((12, 23, TensorDtype::F16))),
            ),
        ] {
            assert!(
                matches!(
                    validate_parallel_communication(dimensions, &rejected, id),
                    Err(FunctionHeadPreparationError::Coordination(_))
                ),
                "{label}"
            );
        }
        assert!(matches!(
            validate_parallel_communication(
                FunctionHeadDimensions::new(4, 1).unwrap(),
                &accepted,
                id
            ),
            Err(FunctionHeadPreparationError::Parallel(_))
        ));
    }

    #[test]
    fn released_function_head_tp_preparation_retains_exact_source_and_conversion_bounds() {
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let parameters = official_parameters();
        let prepare = || {
            PreparedFunctionHeadBindings::from_parameters(
                FunctionHeadDimensions::new(4480, 131072).unwrap(),
                source.clone(),
                &parameters,
            )
            .unwrap()
        };
        let prepared = prepare();
        let exact_source = prepared.source().clone();
        let stored = prepared.parameter().stored_bytes;
        let workspace = prepared.parameter().conversion_workspace_bytes;
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
        let (selected, layout) = prepared.select_parallel(topology).unwrap();
        assert!(Arc::ptr_eq(&exact_source, selected.source()));
        assert_eq!(
            selected.source().source_keys(),
            ["stt_model.function_head.weight"]
        );
        assert!(matches!(
            selected
                .source()
                .source_metadata("stt_model.perception.proj.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let binding = &selected.unit_bindings()[0][0];
        let output = binding
            .source_recipe()
            .infer(selected.source().as_ref())
            .unwrap();
        assert_eq!(output.shape, [65536, 4480]);
        assert_eq!(output.dtype, RecipeDtype::F32);
        assert_eq!(binding.expected_bytes(), 65536 * 4480 * 4);
        assert_eq!(selected.parameter().output, output);
        assert_eq!(selected.parameter().stored_bytes, stored);
        assert_eq!(selected.parameter().conversion_workspace_bytes, workspace);
        assert!(
            selected.parameter().materialization_peak_bytes
                >= stored + workspace + binding.expected_bytes()
        );
        assert!(matches!(
            selected.into_parallel(&layout),
            Err(FunctionHeadPreparationError::Parallel(_))
        ));

        let wrong_description = function_head_parameter_description(
            FunctionHeadDimensions::new(4481, 131072).unwrap(),
            &eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
        )
        .unwrap();
        let wrong_layout =
            eredu_runtime::derive_partitioned_local_layout(&wrong_description, topology).unwrap();
        assert!(prepare().into_parallel(&wrong_layout).is_err());
        let pipeline =
            ParallelRankTopology::new(ParallelTopology::new(1, 2, 1, 1).unwrap(), 0).unwrap();
        assert!(matches!(
            prepare().select_parallel(pipeline),
            Err(FunctionHeadPreparationError::Parallel(_))
        ));
    }
}

/// Cold bindings for the independently materializable function vocabulary projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedFunctionHeadBindings {
    dimensions: FunctionHeadDimensions,
    format: eredu_nn::LinearFormatSpec,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    parameter: ParameterAdmission,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedFunctionHeadBindings {
    /// Projects exact function head bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, FunctionHeadPreparationError> {
        let source = artifacts.component_source(Component::FunctionHead).ok_or(
            SpeechParameterError::MissingComponent(Component::FunctionHead),
        )?;
        Self::from_parameters(
            FunctionHeadDimensions::new(
                i32::try_from(artifacts.plan().config().perception.output_dim).map_err(|_| {
                    ChannelError::Configuration("function head width exceeds native indexing")
                })?,
                TEXT_VOCABULARY_SIZE as i32,
            )?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_parameters(
        dimensions: FunctionHeadDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, FunctionHeadPreparationError> {
        let weight = "stt_model.function_head.weight";
        let primary = parameters
            .get(weight)
            .filter(|p| p.component == Component::FunctionHead)
            .ok_or_else(|| SpeechParameterError::MissingParameter {
                component: Component::FunctionHead,
                name: weight.into(),
            })?;
        let format = crate::linear_format::standard_linear_format(weight, primary.linear_format)
            .map_err(ChannelError::from)?;
        let description = function_head_parameter_description(dimensions, &format)?;
        let layout = description.unit_layout().clone();
        let mut selected_parameter = None;
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(FunctionHeadPreparationError::Geometry(
                    "unexpected static function head owner".into(),
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
                    .filter(|p| p.component == Component::FunctionHead)
                    .ok_or_else(|| SpeechParameterError::MissingParameter {
                        component: Component::FunctionHead,
                        name: name.into(),
                    })?;
                if parameter.output.shape != member.global_shape()
                    || parameter.recipe.infer(source.as_ref())? != parameter.output
                {
                    return Err(FunctionHeadPreparationError::Geometry(name.into()));
                }
                let mut parameter = parameter.clone();
                // Floating parameters use F32 arithmetic; packed matrix storage
                // retains its admitted encoding and ordinary companion bindings.
                if matches!(parameter.output.dtype, RecipeDtype::F16 | RecipeDtype::BF16) {
                    parameter.recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::Cast {
                        input: Box::new(parameter.recipe),
                        dtype: RecipeDtype::F32,
                    };
                    parameter.output = parameter.recipe.infer(source.as_ref())?;
                }
                parameter.materialization_peak_bytes = parameter
                    .recipe
                    .peak_materialization_bytes(source.as_ref())?
                    .checked_add(parameter.conversion_workspace_bytes)
                    .and_then(|bytes| bytes.checked_add(parameter.stored_bytes))
                    .ok_or(RecipeError::ArithmeticOverflow(
                        "function head execution peak bytes",
                    ))?;
                if name == weight {
                    selected_parameter = Some(parameter.clone());
                }
                if !matches!(parameter.output.dtype, RecipeDtype::F32 | RecipeDtype::U32) {
                    return Err(FunctionHeadPreparationError::Geometry(name.into()));
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
            "voicechat/function_head/leased-units",
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
                            return Err(FunctionHeadPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(FunctionHeadPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, FunctionHeadPreparationError>>()?;
        Ok(Self {
            dimensions,
            format,
            layout,
            source,
            units,
            parameter: selected_parameter.expect("one declared function head parameter"),
            local_layout: None,
        })
    }

    /// Selects vocabulary rows through the shared physical parameter planner,
    /// or validates an already retained selection against the exact topology.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), FunctionHeadPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(FunctionHeadPreparationError::Parallel(
                "function-head component selection requires a tensor-only topology".into(),
            ));
        }
        let format = self.format.clone();
        let description = function_head_parameter_description(self.dimensions, &format)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(FunctionHeadPreparationError::Parallel)?;
        if let Some(selected) = self.local_layout.as_ref() {
            if selected != &layout {
                return Err(FunctionHeadPreparationError::Parallel(
                    "retained function-head placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Applies exact rank-local recipes while preserving the retained source.
    /// Source conversion overhead is retained even when the output rows are sliced.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, FunctionHeadPreparationError> {
        if self.local_layout.is_some() {
            return Err(FunctionHeadPreparationError::Parallel(
                "function-head bindings already have a tensor partition".into(),
            ));
        }
        let format = self.format.clone();
        super::function_head_layered::function_head_parallel_range(
            self.dimensions,
            &format,
            layout,
        )?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.parameter.recipe = self.units[0]
            .iter()
            .find(|b| b.name() == "stt_model.function_head.weight")
            .expect("validated primary function weight")
            .source_recipe();
        self.parameter.output = self.parameter.recipe.infer(self.source.as_ref())?;
        self.parameter.materialization_peak_bytes = self
            .parameter
            .recipe
            .peak_materialization_bytes(self.source.as_ref())?
            .checked_add(self.parameter.conversion_workspace_bytes)
            .and_then(|bytes| bytes.checked_add(self.parameter.stored_bytes))
            .ok_or(RecipeError::ArithmeticOverflow(
                "function head execution peak bytes",
            ))?;
        self.local_layout = Some(layout.clone());
        Ok(self)
    }

    /// Exact physical tensor placement retained by this preparation.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }

    /// Exact execution recipe and conservative physical storage, converter
    /// workspace and recipe-temporary accounting for the selected head.
    pub fn parameter(&self) -> &ParameterAdmission {
        &self.parameter
    }

    /// Validated function head geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> FunctionHeadDimensions {
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
        FunctionHeadDimensions,
        eredu_nn::LinearFormatSpec,
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        (
            self.dimensions,
            self.format,
            self.local_layout,
            self.layout,
            self.source,
            self.units,
        )
    }
}

/// Checks bounded failure agreement and the minimum one-position F32 vocabulary gather.
/// Actual batch/sequence extents remain bounded by the retained communication authority.
pub fn validate_parallel_communication(
    dimensions: FunctionHeadDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), FunctionHeadPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        replicated_execution::validate_collective_coordination, CommunicationOperation,
        CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    dimensions.validate()?;
    let selected = validate_collective_coordination(manifest, group, [])?;
    let vocabulary = dimensions.vocabulary as usize;
    let degree = selected.members().len();
    if degree > vocabulary {
        return Err(FunctionHeadPreparationError::Parallel(
            "function-head tensor degree exceeds vocabulary rows".into(),
        ));
    }
    validate_collective_coordination(
        manifest,
        group,
        [CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherUneven,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, vocabulary.div_ceil(degree), None)?
                .with_output_tensor_elements(vocabulary)?,
            true,
        )?],
    )?;
    Ok(())
}
