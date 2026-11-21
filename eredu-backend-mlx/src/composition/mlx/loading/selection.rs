use super::*;

/// Native rank and world placement retained during cold loading.
#[derive(Debug, Clone, Copy)]
pub struct MlxModelPlacement {
    pub(crate) rank_context: Option<crate::backend::MlxRankContext>,
    pub(crate) world_backend: Option<safemlx::distributed::Backend>,
}

impl MlxModelPlacement {
    pub(crate) const fn direct(rank_context: Option<crate::backend::MlxRankContext>) -> Self {
        Self {
            rank_context,
            world_backend: None,
        }
    }
}

/// Validates the exact rank assignment without inspecting the model request.
pub(crate) fn direct_rank(
    topology: Option<eredu_core::ParallelRankTopology>,
    device: Option<crate::backend::DeviceAssignment>,
) -> Result<Option<crate::backend::MlxRankContext>, Error> {
    if !cfg!(feature = "distributed") && topology.is_some() {
        return Err(Error::DistributedModelExecutionDisabled);
    }
    match (topology, device) {
        (Some(topology), Some(device)) => crate::backend::MlxRankContext::new(
            topology.world_size(),
            topology.global_rank(),
            device,
        )
        .map(Some),
        (None, None) => Ok(None),
        (Some(_), None) => Err(Error::Parallel(
            "parallel execution has no MLX device assignment".into(),
        )),
        (None, Some(_)) => Err(Error::Parallel(
            "MLX device assignment requires a parallel topology".into(),
        )),
    }
}

/// Side-effect-free MLX facts for portable preparation selection.
pub struct MlxPreparationMechanisms;

impl eredu_core::preparation::NativePreparation for MlxPreparationMechanisms {
    fn row_lookup_support(
        &self,
    ) -> Option<&dyn eredu_core::preparation::rows::NativeRowLookupSupport> {
        Some(&crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport)
    }

    fn input_score_attention_workspace(
        &self,
    ) -> Option<eredu_core::attention::InputScoreAttentionMechanism> {
        Some(crate::backend::nn::attention::INPUT_SCORE_WORKSPACE)
    }

    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &C,
    ) -> Result<u64, String> {
        crate::backend::runtime::checkpoint::recipe::native_recipe_workspace(recipe, source)
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: cfg!(feature = "tensor-capture"),
            routing_tensors: cfg!(feature = "tensor-capture"),
            routed_unit_tensors: cfg!(feature = "tensor-capture"),
            floating_to_f32: cfg!(feature = "tensor-capture"),
        }
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        super::super::session::capture_capabilities()
    }

    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::super::structural::preparation_mechanism_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_nn::GroupedOperationRequirement,
    ) -> bool {
        super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES.contains(&requirement)
    }

    fn replicated_text_support(
        &self,
    ) -> Option<&dyn eredu_core::preparation::ReplicatedTextMechanismSupport> {
        Some(&super::super::replicated_text::MlxReplicatedTextSupport)
    }

    fn processor_capabilities(&self) -> eredu_core::media::MediaPrimitiveCapabilities {
        super::super::processor::capabilities()
    }

    fn speculative_capabilities(
        &self,
    ) -> eredu_core::speculative::SpeculativeMechanismCapabilities {
        super::super::speculative::speculative_mechanism_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_core::communication::CommunicationCapabilities {
        crate::backend::runtime::distributed::topology::mlx_communication_capabilities()
    }
}
