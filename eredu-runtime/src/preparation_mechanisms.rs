//! Portable mechanism selection and admission from native preparation observations.

use eredu_core::preparation::NativePreparation;
pub use eredu_core::preparation::{BackendInspection, PlanMechanismFacts};

use eredu_core::PreparationMechanismCapabilities;

use crate::{
    BackendMechanismCapabilities, GroupedOperationRequirement, MediaPrimitiveCapabilities,
    NormalizedLoadRequest, ReplicatedTextRequirements, ReplicatedTextSelectionRequest,
    SpeculativeMechanismCapabilities,
};
use eredu_core::communication::CommunicationCapabilities;

/// Preparation queries used by portable selectors and retained inspection.
///
/// Native facts are adapted here to whole-candidate reports. Inspection can
/// record these queries and their results for deterministic revalidation.
pub trait PreparationMechanismProvider {
    /// Floating activation and state representation produced by materializing the
    /// declared embedding dtype. This is an immutable native mechanism fact.
    fn floating_state_dtype(
        &self,
        _source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<crate::StateStorageDtype> {
        None
    }

    /// Storage tiers and completion ownership for addressable table rows.
    fn row_lookup_storage(&self) -> Option<crate::AddressableStorageCapabilities> {
        None
    }

    /// Exact row decoder and scalar-recipe support, using immutable headers only.
    fn row_lookup_workspace(
        &self,
        _descriptor: &crate::RowLookupDescriptor,
    ) -> Result<Option<crate::RowLookupWorkspace>, crate::RowLookupError> {
        Ok(None)
    }

    /// Native row decoder storage and retention facts for one bounded acquisition.
    fn row_lookup_decode_memory(
        &self,
        descriptor: &crate::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, crate::RowLookupError> {
        descriptor.decode_memory()
    }

    /// Native temporary geometry for explicit input-score attention. No family
    /// identity, tensor, stream or device is consulted.
    fn input_score_attention_workspace(
        &self,
    ) -> Option<crate::memory_estimation::InputScoreAttentionMechanism> {
        None
    }

    /// Conservative live buffers for one physical recipe, including native
    /// copies and index buffers. This must inspect metadata only.
    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        _recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        _source: &C,
    ) -> Result<u64, String> {
        Err("backend did not report native recipe workspace".into())
    }

    /// Host observation collector facts, independent of model family and devices.
    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms::default()
    }

    /// Bounded native transformations, without allocating or selecting a device.
    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        Default::default()
    }

    /// Portable artifact, residency, topology, modality, completion, and session mechanisms.
    fn preparation_capabilities(&self) -> PreparationMechanismCapabilities;

    /// Whether one exact grouped-compute semantic is implemented.
    fn supports_grouped_operation(&self, requirement: GroupedOperationRequirement) -> bool;

    /// Exact operator and state support for portable text admission, when implemented.
    fn replicated_text_support(&self) -> Option<&dyn crate::ReplicatedTextMechanismSupport> {
        None
    }

    /// Mechanisms applicable to one exact neutral text requirement and request.
    fn replicated_text_capabilities(
        &self,
        requirements: &ReplicatedTextRequirements,
        request: &ReplicatedTextSelectionRequest,
    ) -> BackendMechanismCapabilities {
        match self.replicated_text_support() {
            Some(support) => {
                crate::synthesize_replicated_text_capabilities(requirements, request, support)
            }
            None => BackendMechanismCapabilities::new(
                eredu_nn::NeuralOperatorCapabilities::NONE,
                Vec::new(),
                Vec::new(),
                crate::StateMechanismCapabilities::new([]),
            ),
        }
    }

    /// Mechanisms for an independently owned auxiliary state role.
    fn state_capabilities(
        &self,
        requirements: &crate::StateRealizationRequirements,
        policy: &crate::CacheResidencyPolicy,
    ) -> crate::StateMechanismCapabilities {
        match self.replicated_text_support() {
            Some(support) => crate::synthesize_state_capabilities(requirements, policy, support),
            None => crate::StateMechanismCapabilities::new([]),
        }
    }

    /// Portable media and prepared-tensor primitives.
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities;

    /// Portable speculative execution mechanisms.
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities;

    /// Portable communication operations, bounds, framing, and completion support.
    fn communication_capabilities(&self) -> CommunicationCapabilities;
}

impl<N: NativePreparation> PreparationMechanismProvider for N {
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<crate::StateStorageDtype> {
        NativePreparation::floating_state_dtype(self, source)
    }

    fn row_lookup_storage(&self) -> Option<crate::AddressableStorageCapabilities> {
        self.row_lookup_support()?.row_storage()
    }

    fn row_lookup_workspace(
        &self,
        descriptor: &crate::RowLookupDescriptor,
    ) -> Result<Option<crate::RowLookupWorkspace>, crate::RowLookupError> {
        match self.row_lookup_support() {
            Some(support) => crate::RowLookupMechanismSupport::workspace(support, descriptor),
            None => Ok(None),
        }
    }

    fn row_lookup_decode_memory(
        &self,
        descriptor: &crate::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, crate::RowLookupError> {
        match self.row_lookup_support() {
            Some(support) => crate::RowLookupMechanismSupport::decode_memory(support, descriptor),
            None => descriptor.decode_memory(),
        }
    }

    fn input_score_attention_workspace(
        &self,
    ) -> Option<eredu_core::attention::InputScoreAttentionMechanism> {
        NativePreparation::input_score_attention_workspace(self)
    }

    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &C,
    ) -> Result<u64, String> {
        NativePreparation::recipe_materialization_workspace(self, recipe, source)
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        NativePreparation::observation_mechanisms(self)
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        NativePreparation::capture_capabilities(self)
    }

    fn preparation_capabilities(&self) -> PreparationMechanismCapabilities {
        NativePreparation::preparation_capabilities(self)
    }

    fn supports_grouped_operation(&self, requirement: GroupedOperationRequirement) -> bool {
        NativePreparation::supports_grouped_operation(self, requirement)
    }

    fn replicated_text_support(&self) -> Option<&dyn crate::ReplicatedTextMechanismSupport> {
        NativePreparation::replicated_text_support(self)
    }

    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
        NativePreparation::processor_capabilities(self)
    }

    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
        NativePreparation::speculative_capabilities(self)
    }

    fn communication_capabilities(&self) -> CommunicationCapabilities {
        NativePreparation::communication_capabilities(self)
    }
}

/// Backend mechanism facts for direct portable model admission.
pub struct ModelLoadFacts<M> {
    /// Normalized request used by the portable selector.
    pub request: NormalizedLoadRequest,
    /// Exact mechanism predicates supplied by the backend.
    pub mechanisms: M,
}

/// Chooses supported original GGUF blocks when affine storage is unavailable
/// or the native integer dot prefers original blocks under automatic selection.
pub fn select_packed_gguf_tensors(
    mechanisms: &impl PreparationMechanismProvider,
    checkpoint: &eredu_gguf::Checkpoint,
    allow_native_preference: bool,
) -> Vec<String> {
    use eredu_checkpoint::{
        lowering::WeightLoweringDescriptor, AffineQuantization, LinearFormat, SourceTensorEncoding,
    };

    let Some(support) = mechanisms.replicated_text_support() else {
        return Vec::new();
    };
    checkpoint
        .shards()
        .iter()
        .flat_map(|shard| {
            shard.tensors().iter().filter_map(|tensor| {
                let (bits, group) = tensor.affine()?;
                let affine =
                    AffineQuantization::new(i32::try_from(group).ok()?, bits.into()).ok()?;
                let tensor = tensor.descriptor();
                let shape = tensor
                    .dimensions
                    .iter()
                    .rev()
                    .map(|&dim| usize::try_from(dim))
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                let axis = Some(shape.len().checked_sub(1)?);
                let endian = shard.endian();
                let ggml_type = tensor.ggml_type;
                let descriptor = |format| {
                    WeightLoweringDescriptor::new(
                        SourceTensorEncoding::Gguf { ggml_type, endian },
                        format,
                        shape.clone(),
                        shape.clone(),
                        axis,
                    )
                };
                let packed = descriptor(LinearFormat::GgufIQuant { ggml_type, endian }).ok()?;
                let affine = descriptor(LinearFormat::Affine(affine))
                    .is_ok_and(|descriptor| support.supports_direct(&descriptor));
                (support.supports_direct(&packed)
                    && (!affine
                        || allow_native_preference && support.prefers_original_encoding(&packed)))
                .then(|| tensor.name.clone())
            })
        })
        .collect()
}

/// Native mechanism and weight-lowering facts used to admit an external draft.
pub struct ExternalDraftFacts<M> {
    /// Exact backend mechanism predicates.
    pub mechanisms: M,
}

/// Backend-supplied mechanism and load-request facts for portable model planning.
pub struct AutomaticModelFacts<M> {
    /// Inputs for normalizing and checking a concrete candidate.
    pub preparation: crate::execution_plan::ExecutionPlanPreparation,
    /// Side-effect-free support for the exact required mechanisms.
    pub mechanisms: M,
    /// Host ingress paths compiled into the selected backend.
    pub media: eredu_core::MediaFeatureAvailability,
}

impl<M> eredu_core::backend::DirectLoadFacts for ModelLoadFacts<M> {
    type Request = NormalizedLoadRequest;
    type Mechanisms = M;

    fn rank(request: &NormalizedLoadRequest) -> Option<eredu_core::ParallelRankTopology> {
        request.parallel_topology()
    }

    fn from_request(request: NormalizedLoadRequest, mechanisms: M) -> Self {
        Self {
            request,
            mechanisms,
        }
    }
}

impl<M> eredu_core::backend::AdmissionFacts for ExternalDraftFacts<M> {
    type Inputs = M;
    fn from_inputs(mechanisms: Self::Inputs) -> Self {
        Self { mechanisms }
    }
}

impl<M> eredu_core::backend::AdmissionFacts for AutomaticModelFacts<M> {
    type Inputs = (
        crate::execution_plan::ExecutionPlanPreparation,
        M,
        eredu_core::MediaFeatureAvailability,
    );
    fn from_inputs((preparation, mechanisms, media): Self::Inputs) -> Self {
        Self {
            preparation,
            mechanisms,
            media,
        }
    }
}
