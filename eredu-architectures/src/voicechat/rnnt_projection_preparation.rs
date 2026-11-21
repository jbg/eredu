//! Restricted RNNT embedding and joint recipes without recurrent allocation.

use super::{
    rnnt::{RnntDimensions, RnntError},
    rnnt_layered::{RnntKernel, RnntUnit},
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_projection_layered::RnntProjection,
    rnnt_specs::RnntDeclaration,
};
use eredu_core::communication::CommunicationManifest;
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::SubmissionBackend;
use eredu_core::{BackendFailure, CollectiveGroupId, ParallelRankTopology, ParallelTopology};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    ComponentAdapter, LayerWeightResidency, LayeredTensorCohort, LayerwiseModelMetadata,
    LocalModelLayout, ParameterUnitMechanisms, PreparedComponent,
};

/// Exact four-unit source authority, including vocabulary placement when selected.
/// Recurrent recipes and replica declarations are excluded from this owner.
pub struct PreparedRnntProjectionBindings {
    prepared: PreparedComponent<RnntDeclaration<false>>,
    selection: Option<(CommunicationManifest, CollectiveGroupId)>,
}
impl PreparedRnntProjectionBindings {
    /// Projects existing retained RNNT recipes without reading parameter payloads.
    /// Existing tensor selection is retained exactly; recipes are never sliced twice.
    pub fn from_rnnt(bindings: PreparedRnntBindings) -> Result<Self, RnntPreparationError> {
        let declaration = bindings.declaration().projections()?;
        let prepared = bindings
            .prepared
            .project("voicechat/rnnt/projection-units", declaration)?;
        if let Some(layout) = prepared.local_layout() {
            prepared.declaration().local_dimensions(layout)?;
        }
        Ok(Self {
            prepared,
            selection: None,
        })
    }

    /// Selects vocabulary shards and exact encoder/prediction replicas for one cohort.
    pub fn select_parallel(
        mut self,
        manifest: &CommunicationManifest,
        group: CollectiveGroupId,
    ) -> Result<Self, RnntPreparationError> {
        super::rnnt_preparation::validate_neural_communication(
            self.dimensions(),
            manifest,
            group,
            true,
        )?;
        let cohort = manifest
            .groups()
            .iter()
            .find(|cohort| cohort.id() == group)
            .expect("validated projection cohort");
        let topology = ParallelTopology::new(cohort.members().len(), 1, 1, 1)
            .and_then(|topology| {
                ParallelRankTopology::new(
                    topology,
                    cohort.local_index().expect("validated local membership"),
                )
            })
            .map_err(|error| RnntPreparationError::Parallel(error.to_string()))?;
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        self.prepared.declaration().local_dimensions(&layout)?;
        self.prepared = self.prepared.select(&layout)?;
        self.selection = Some((manifest.clone(), group));
        Ok(self)
    }
    /// Global geometry used by projection equations and the shared greedy driver.
    pub fn dimensions(&self) -> RnntDimensions {
        self.prepared.declaration().dimensions()
    }
    /// Exact optional vocabulary placement with no recurrent replica declarations.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Architecture-ordered embedding, encoder, prediction and output bindings.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }
    /// Binds the retained recipes with the shared parameter unit mechanism.
    pub fn bind<B, H, C>(
        self,
        residency: LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
        mechanisms: &H,
        cohort: Option<C>,
    ) -> Result<
        (
            RnntProjection<B, H::UnitPolicy<RnntUnit<B>>, C>,
            LayerwiseModelMetadata,
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
        let selection = self
            .selection
            .as_ref()
            .map(|(manifest, group)| (manifest, *group));
        if selection != cohort.as_ref().map(LayeredTensorCohort::selection)
            || self.prepared.local_layout().is_some() != selection.is_some()
        {
            return Err(BackendFailure::from_error(RnntError::InvalidConfiguration(
                "RNNT projection parameter placement differs from its tensor cohort",
            )));
        }
        let (declaration, local, invocations, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .map_err(BackendFailure::from_error)?;
        let definition = declaration.definition().clone();
        let kernel = match local.as_ref() {
            Some(layout) => RnntKernel::<B, false>::new_parallel(declaration, layout),
            None => RnntKernel::<B, false>::new(declaration),
        }
        .map_err(BackendFailure::from_error)?;
        let architecture =
            ComponentAdapter::new(definition, kernel).map_err(BackendFailure::from_error)?;
        RnntProjection::bind(
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
}
