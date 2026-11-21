//! Cold exact binding plans for independently leased PRVQ codebook units.

use eredu_core::SubmissionBackend;

use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{LayeredTensorCohort, ParameterUnitMechanisms};

use eredu_checkpoint::{recipe::RecipeDtype, store::SharedCheckpointSource};
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_runtime::{
    ComponentPreparationError, ComponentRecipeConversion, LocalModelLayout, PreparedComponent,
};

use crate::{
    prvq::{Codebook, Geometry},
    prvq_layered::{BoundQuantizer, QuantizerKernel},
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum QuantizerPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// A retained communicator lacks required membership, completion or operations.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// A communication requirement is malformed.
    #[error(transparent)]
    Communication(#[from] eredu_core::communication::CommunicationManifestError),

    /// Quantizer geometry cannot be constructed.
    #[error(transparent)]
    Quantizer(#[from] crate::Error),
    /// A required parameter is missing from the selected component.
    #[error("PRVQ parameter {0} is missing")]
    MissingParameter(String),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid PRVQ tensor partition: {0}")]
    Parallel(String),
}

/// Cold bindings for each canonical codebook mean table and retained variance.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no waveform or conversational execution capability.
pub struct PreparedQuantizerBindings {
    prepared: PreparedComponent<crate::declarations::QuantizerDeclaration>,
}

impl PreparedQuantizerBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes<'a>(
        declaration: crate::declarations::QuantizerDeclaration,
        source: SharedCheckpointSource,
        parameters: impl FnMut(&str) -> Option<&'a eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, QuantizerPreparationError> {
        let prepared = PreparedComponent::prepare_physical_recipes(
            "prvq/codebooks/leased-units",
            declaration,
            source,
            parameters,
            Some(ComponentRecipeConversion {
                dtype: RecipeDtype::F32,
                from: &[RecipeDtype::F16, RecipeDtype::BF16],
            }),
            |name| QuantizerPreparationError::MissingParameter(name.into()),
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
            BoundQuantizer<B, H::UnitPolicy<Codebook<B::Tensor>>, C>,
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
            validate_parallel_communication(self.geometry(), manifest, group)
                .map_err(BackendFailure::from_error)?;
            self.select_parallel(cohort.tensor_topology()?)
                .map_err(BackendFailure::from_error)?
                .0
        } else {
            if self.local_layout().is_some() {
                return Err(BackendFailure::from_error(crate::Error::InvalidShape(
                    "tensor-local parameter placement requires its selected cohort".into(),
                )));
            }
            self
        };
        let (declaration, local, invocations, source, units) = selected.into_parts();
        let definition = declaration.definition().clone();
        let kernel = match local.as_ref() {
            Some(layout) => QuantizerKernel::<B>::new_parallel(declaration, layout),
            None => QuantizerKernel::<B>::new(declaration),
        }
        .map_err(BackendFailure::from_error)?;
        let architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
            .map_err(BackendFailure::from_error)?;
        let (units, metadata) = eredu_runtime::BoundStatelessComponent::bind(
            architecture,
            source,
            invocations,
            units,
            residency,
            context,
            mechanisms,
            cohort,
        )?;
        Ok((BoundQuantizer::new(units), metadata))
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), QuantizerPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, QuantizerPreparationError> {
        crate::prvq_layered::declared_local_dimensions(self.prepared.declaration().spec(), layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated neural quantizer geometry.
    pub fn geometry(&self) -> Geometry {
        *self.prepared.declaration().config()
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    fn into_parts(
        self,
    ) -> (
        crate::declarations::QuantizerDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (definition, local, layout, source, units)
    }
}

/// Admits exact PRVQ sums and local failure agreement before native parameter reads.
/// One-row limits are the minimum viable component; the retained communicator
/// validates every requested batch/frame extent before each actual collective.
pub fn validate_parallel_communication(
    geometry: Geometry,
    manifest: &eredu_core::communication::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), QuantizerPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_core::communication::CommunicationOperation;
    use eredu_core::communication::{CommunicationOperationRequirement, CommunicationTensorLimits};
    use eredu_runtime::replicated_execution::validate_collective_coordination;
    geometry.validate()?;
    let selected = validate_collective_coordination(
        manifest,
        group,
        [CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(
                1,
                3,
                geometry.channels.max(geometry.cardinality) as usize,
                None,
            )?,
            true,
        )?],
    )?;
    if selected.members().len() > geometry.channels as usize {
        return Err(QuantizerPreparationError::Parallel(
            "PRVQ tensor degree exceeds channels".into(),
        ));
    }
    Ok(())
}
