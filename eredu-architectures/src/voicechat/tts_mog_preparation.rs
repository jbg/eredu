//! Cold exact binding plans for independently leased TTS MoG units.

use eredu_core::SubmissionBackend;

use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{LayeredTensorCohort, ParameterUnitMechanisms};
use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{RecipeDtype, RecipeError},
    store::SharedCheckpointSource,
};
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_runtime::{
    ComponentPreparationError, LocalModelLayout, PreparedComponent, RecipeBindingError,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{admitted_member, execution_parameter, SpeechParameterError},
    tts_mog::{MogDimensions, TtsError},
    tts_mog_layered::{BoundMog, MogKernel, MogUnit},
    tts_mog_specs::{released_declaration, MogDeclaration},
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum MogPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Invalid floating execution profile or attempted precision reselection.
    #[error("invalid speech precision: {0}")]
    Precision(&'static str),
    /// Selected communication lacks required bounded mechanisms.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// Communication requirements cannot be represented.
    #[error(transparent)]
    Communication(#[from] eredu_core::communication::CommunicationManifestError),
    /// Mog geometry cannot be constructed.
    #[error(transparent)]
    Mog(#[from] TtsError),
    /// A required parameter is missing from the selected component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid MoG tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS MoG parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for each residual block, the prediction head, and selected bank.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedMogBindings {
    precision: super::precision::SpeechPrecision,
    prepared: PreparedComponent<MogDeclaration>,
}

impl PreparedMogBindings {
    /// Selects portable execution precision and exact converted bindings before materialization.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, MogPreparationError> {
        if self.precision != Default::default() && self.precision != precision {
            return Err(MogPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(MogPreparationError::Precision)?;
        let declaration = super::precision::SpeechPrecisionDeclaration::from_groups(
            self.prepared.declaration().definition().parameters().groups().iter().filter(|owned| matches!(
                owned.owner(), eredu_runtime::ParameterGroupOwner::ExecutionUnit { group, .. }
                    if group.as_str() == super::tts_mog_layered::BLOCKS
            )),
        );
        self.prepared = self.prepared.transform_units(|source, units| {
            precision
                .bindings(source, units, &declaration)
                .map_err(MogPreparationError::from)
        })?;
        self.precision = precision;
        Ok(self)
    }
    /// Retained floating storage and arithmetic policy.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }

    /// Projects exact MoG bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, MogPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            released_declaration()?.clone(),
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        declaration: MogDeclaration,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, MogPreparationError> {
        let prepared = PreparedComponent::prepare(
            "voicechat/tts_mog/leased-units",
            declaration,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = admitted_member(parameters, &source, Component::Tts, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(MogPreparationError::Geometry(name.into()));
                }
                Ok::<_, MogPreparationError>(parameter.recipe)
            },
        )?;
        Ok(Self {
            precision: Default::default(),
            prepared,
        })
    }

    /// Binds prepared units locally or to an explicitly selected tensor cohort.
    pub fn bind<B, H, C>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
        mechanisms: &H,
        cohort: Option<C>,
    ) -> Result<
        (
            BoundMog<B, H::UnitPolicy<MogUnit<B>>, C>,
            eredu_runtime::LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend
            + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
            + 'static,
        H: ParameterUnitMechanisms<B>,
        C: LayeredTensorCohort<B>,
    {
        let selected = if let Some(cohort) = &cohort {
            let (manifest, group) = cohort.selection();
            validate_parallel_communication(self.dimensions(), manifest, group)
                .map_err(BackendFailure::from_error)?;
            self.select_parallel(cohort.tensor_topology()?)
                .map_err(BackendFailure::from_error)?
                .0
        } else {
            if self.local_layout().is_some() {
                return Err(BackendFailure::from_error(TtsError::Configuration(
                    "tensor-local parameter placement requires its selected cohort",
                )));
            }
            self
        };
        let precision = selected.precision();
        let (declaration, local, invocations, source, units) = selected.into_parts();
        let definition = declaration.definition().clone();
        let kernel = match local.as_ref() {
            Some(layout) => MogKernel::<B>::new_parallel(declaration, layout),
            None => MogKernel::<B>::new(declaration),
        }
        .and_then(|kernel| kernel.with_precision(precision))
        .map_err(BackendFailure::from_error)?;
        let architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
            .map_err(BackendFailure::from_error)?;
        BoundMog::bind(
            architecture,
            source,
            invocations,
            units,
            residency,
            context,
            mechanisms,
            cohort,
        )
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), MogPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(mut self, layout: &LocalModelLayout) -> Result<Self, MogPreparationError> {
        self.prepared.declaration().local_dimensions(layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated MoG geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> MogDimensions {
        self.prepared.declaration().dimensions()
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted source used for cold residency accounting.
    pub(super) fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    fn into_parts(
        self,
    ) -> (
        MogDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (declaration, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (declaration, local, layout, source, units)
    }
}

/// Validates bounded agreement and sum authority before native TP construction.
pub fn validate_parallel_communication(
    dimensions: MogDimensions,
    manifest: &eredu_core::communication::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), MogPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_core::communication::CommunicationOperation;
    use eredu_core::communication::{CommunicationOperationRequirement, CommunicationTensorLimits};
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
