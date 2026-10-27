//! Cold exact binding plans for independently leased function head units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{ComponentPreparationError, LocalModelLayout, PreparedComponent, WeightBinding};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    channels::{ChannelError, TEXT_VOCABULARY_SIZE},
    checkpoint::Component,
    function_head_layered::{function_head_component_definition, FunctionHeadDimensions},
    parameters::{admitted_member, SpeechParameterError},
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum FunctionHeadPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
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

/// Cold bindings for the independently materializable function vocabulary projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedFunctionHeadBindings {
    dimensions: FunctionHeadDimensions,
    format: eredu_nn::LinearFormatSpec,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
    parameter: ParameterAdmission,
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
        let definition = function_head_component_definition(dimensions, &format)?;
        let mut selected_parameter = None;
        let prepared = PreparedComponent::prepare(
            "voicechat/function_head/leased-units",
            definition,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter =
                    admitted_member(parameters, &source, Component::FunctionHead, member)?;
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
                if !matches!(parameter.output.dtype, RecipeDtype::F32 | RecipeDtype::U32)
                    && !(parameter.output.dtype == RecipeDtype::U8
                        && super::parameters::is_byte_scale(parameters, name))
                {
                    return Err(FunctionHeadPreparationError::Geometry(name.into()));
                }
                Ok::<_, FunctionHeadPreparationError>(parameter.recipe)
            },
        )?;
        Ok(Self {
            dimensions,
            format,
            prepared,
            parameter: selected_parameter.expect("one declared function head parameter"),
        })
    }

    /// Selects vocabulary rows through the shared physical parameter planner,
    /// or validates an already retained selection against the exact topology.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), FunctionHeadPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Applies exact rank-local recipes while preserving the retained source.
    /// Source conversion overhead is retained even when the output rows are sliced.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, FunctionHeadPreparationError> {
        let format = self.format.clone();
        super::function_head_layered::function_head_parallel_range(
            self.dimensions,
            &format,
            layout,
        )?;
        self.prepared = self.prepared.select(layout)?;
        self.parameter.recipe = self.prepared.unit_bindings()[0]
            .iter()
            .find(|b| b.name() == "stt_model.function_head.weight")
            .expect("validated primary function weight")
            .source_recipe();
        self.parameter.output = self
            .parameter
            .recipe
            .infer(self.prepared.source().as_ref())?;
        self.parameter.materialization_peak_bytes = self
            .parameter
            .recipe
            .peak_materialization_bytes(self.prepared.source().as_ref())?
            .checked_add(self.parameter.conversion_workspace_bytes)
            .and_then(|bytes| bytes.checked_add(self.parameter.stored_bytes))
            .ok_or(RecipeError::ArithmeticOverflow(
                "function head execution peak bytes",
            ))?;
        Ok(self)
    }

    /// Exact physical tensor placement retained by this preparation.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
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
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Complete semantic declaration retained through precision and placement selection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        FunctionHeadDimensions,
        eredu_nn::LinearFormatSpec,
        eredu_runtime::ComponentDefinition,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("function head declares only independently leased units");
        (
            self.dimensions,
            self.format,
            definition,
            local,
            layout,
            source,
            units,
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
