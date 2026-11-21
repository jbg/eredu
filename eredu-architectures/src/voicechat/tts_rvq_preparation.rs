//! Cold exact binding plans for independently leased TTS RVQ units.

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
use eredu_runtime::{ComponentPreparationError, LocalModelLayout, PreparedComponent};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{admitted_member, execution_parameter, SpeechParameterError},
    tts_mog::TtsError,
    tts_rvq::RvqDimensions,
    tts_rvq_layered::{BoundRvq, RvqKernel, RvqUnit},
    tts_rvq_specs::{released_declaration, RvqDeclaration},
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum RvqPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Selected communication lacks membership, bounded completion or required operations.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// An exact communication requirement is malformed.
    #[error(transparent)]
    Communication(#[from] eredu_core::communication::CommunicationManifestError),
    /// RVQ geometry cannot be constructed.
    #[error(transparent)]
    Rvq(#[from] TtsError),
    /// A required parameter is missing from the selected component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
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
    prepared: PreparedComponent<RvqDeclaration>,
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
            released_declaration()?.clone(),
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        declaration: RvqDeclaration,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, RvqPreparationError> {
        let prepared = PreparedComponent::prepare(
            "voicechat/tts_rvq/leased-units",
            declaration,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = admitted_member(parameters, &source, Component::Tts, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(RvqPreparationError::Geometry(name.into()));
                }
                Ok(parameter.recipe)
            },
        )?;
        Ok(Self { prepared })
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
            BoundRvq<B, H::UnitPolicy<RvqUnit<B>>, C>,
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
        let (declaration, local, invocations, source, units) = selected.into_parts();
        let definition = declaration.definition().clone();
        let kernel = match local.as_ref() {
            Some(layout) => RvqKernel::<B>::new_parallel(declaration, layout),
            None => RvqKernel::<B>::new(declaration),
        }
        .map_err(BackendFailure::from_error)?;
        let architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
            .map_err(BackendFailure::from_error)?;
        BoundRvq::bind(
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
    ) -> Result<(Self, LocalModelLayout), RvqPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(mut self, layout: &LocalModelLayout) -> Result<Self, RvqPreparationError> {
        self.prepared.declaration().local_dimensions(layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated RVQ geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> RvqDimensions {
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
        RvqDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (declaration, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("RVQ declares only independently leased units");
        (declaration, local, layout, source, units)
    }
}

/// Validates the exact operations needed by guarded latent-sharded feedback.
/// Actual batch/time limits are checked against retained authority at execution.
pub fn validate_parallel_communication(
    dimensions: RvqDimensions,
    manifest: &eredu_core::communication::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), RvqPreparationError> {
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
