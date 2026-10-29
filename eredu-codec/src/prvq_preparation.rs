//! Cold exact binding plans for independently leased PRVQ codebook units.

use eredu_checkpoint::{recipe::RecipeDtype, store::SharedCheckpointSource};
use eredu_runtime::{
    ComponentPreparationError, ComponentRecipeConversion, LocalModelLayout, PreparedComponent,
    WeightBinding,
};

use crate::{prvq::Geometry, prvq_layered::quantizer_component_definition};

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
    Communication(#[from] eredu_runtime::CommunicationManifestError),

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
    geometry: Geometry,
    prefix: String,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
}

impl PreparedQuantizerBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes<'a>(
        geometry: Geometry,
        prefix: &str,
        source: SharedCheckpointSource,
        parameters: impl FnMut(&str) -> Option<&'a eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, QuantizerPreparationError> {
        let definition = quantizer_component_definition(geometry, prefix)?;
        let prepared = PreparedComponent::prepare_physical_recipes(
            "prvq/codebooks/leased-units",
            definition,
            source,
            parameters,
            Some(ComponentRecipeConversion {
                dtype: RecipeDtype::F32,
                from: &[RecipeDtype::F16, RecipeDtype::BF16],
            }),
            |name| QuantizerPreparationError::MissingParameter(name.into()),
        )?;
        Ok(Self {
            geometry,
            prefix: prefix.into(),
            prepared,
        })
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), QuantizerPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, QuantizerPreparationError> {
        crate::prvq_layered::quantizer_local_dimensions(self.geometry, &self.prefix, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated neural quantizer geometry.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
    /// Canonical parameter namespace retained from preparation.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Validated semantic definition retained through source and placement selection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        Geometry,
        String,
        eredu_runtime::ComponentDefinition,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (
            self.geometry,
            self.prefix,
            definition,
            local,
            layout,
            source,
            units,
        )
    }
}

/// Admits exact PRVQ sums and local failure agreement before native parameter reads.
/// One-row limits are the minimum viable component; the retained communicator
/// validates every requested batch/frame extent before each actual collective.
pub fn validate_parallel_communication(
    geometry: Geometry,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), QuantizerPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        replicated_execution::validate_collective_coordination, CommunicationOperation,
        CommunicationOperationRequirement, CommunicationTensorLimits,
    };
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
