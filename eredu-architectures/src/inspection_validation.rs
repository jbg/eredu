//! Reusable proofs for one immutable architecture and exact artifact admission.

use std::sync::{Arc, Mutex, OnceLock};

use eredu_core::artifact::ArtifactAdmissionToken;
use eredu_runtime::{NormalizedLoadRequest, ReplicatedTextRequirements};

use crate::{
    configuration::PredictionExtensionPlan,
    processor_plan::ArtifactArchitecturePlan,
    replicated_text::{CompositeTextRequirements, ReplicatedTextRequirementsError},
    routed_text::{RoutedTextRequirements, RoutedTextRequirementsError},
};

/// Clones of an architecture share proofs; semantic changes start a new set.
#[derive(Debug, Default)]
pub(crate) struct InspectionValidation {
    pub projection:
        OnceLock<Result<Option<(ArtifactArchitecturePlan, PredictionExtensionPlan)>, String>>,
    admissions: Mutex<Vec<(ArtifactAdmissionToken, Arc<AdmittedValidation>)>>,
}

impl InspectionValidation {
    /// Retains immutable artifact declarations without execution requirements or
    /// selections that may own an inspection projection.
    pub fn declarations_only(&self) -> Arc<Self> {
        let admissions = self
            .admissions
            .lock()
            .expect("inspection validation poisoned");
        let retained = admissions
            .iter()
            .map(|(token, facts)| {
                let copy = AdmittedValidation::default();
                if let Some(value) = facts.catalog.get() {
                    let _ = copy.catalog.set(value.clone());
                }
                if let Some(value) = facts.sources.get() {
                    let _ = copy.sources.set(value.clone());
                }
                if let Some(value) = facts.normalized.get() {
                    let _ = copy.normalized.set(value.clone());
                }
                if let Some(value) = facts.qwen4.get() {
                    let _ = copy.qwen4.set(value.clone());
                }
                (token.clone(), Arc::new(copy))
            })
            .collect();
        Arc::new(Self {
            projection: OnceLock::new(),
            admissions: Mutex::new(retained),
        })
    }

    pub fn for_admission(&self, token: ArtifactAdmissionToken) -> Arc<AdmittedValidation> {
        let mut admissions = self
            .admissions
            .lock()
            .expect("inspection validation poisoned");
        if let Some((_, facts)) = admissions
            .iter()
            .find(|(origin, _)| origin.same_admission(&token))
        {
            return Arc::clone(facts);
        }
        let facts = Arc::new(AdmittedValidation::default());
        admissions.push((token, Arc::clone(&facts)));
        facts
    }
}

#[derive(Debug, Default)]
pub(crate) struct AdmittedValidation {
    #[cfg(test)]
    pub selection_runs: std::sync::atomic::AtomicUsize,
    pub qwen4: OnceLock<
        Result<
            Arc<crate::preparation_selection::qwen4::AdmittedTarget>,
            crate::PreparationSelectionError,
        >,
    >,
    pub sources: OnceLock<
        Result<
            Arc<crate::artifact_preparation::ArtifactSourceDeclarations>,
            ReplicatedTextRequirementsError,
        >,
    >,
    pub catalog: OnceLock<
        Result<
            Arc<crate::replicated_text::artifact::ArtifactRecipeCatalog>,
            ReplicatedTextRequirementsError,
        >,
    >,
    pub normalized: OnceLock<
        Result<
            Arc<crate::replicated_text::artifact::NormalizedArtifactPreparation>,
            ReplicatedTextRequirementsError,
        >,
    >,
    pub replicated: OnceLock<Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError>>,
    pub routed: OnceLock<Result<RoutedTextRequirements, RoutedTextRequirementsError>>,
    pub composite: OnceLock<Result<CompositeTextRequirements, ReplicatedTextRequirementsError>>,
    pub selections: Mutex<Vec<ValidatedSelection>>,
}

#[derive(Debug)]
pub(crate) struct ValidatedSelection {
    pub request: NormalizedLoadRequest,
    pub mechanisms: Vec<MechanismObservation>,
    pub selected: Result<crate::SelectedPreparation, crate::PreparationSelectionError>,
}

/// Retain the actual neutral facts consulted by selection, rather than relying
/// on a backend name or object address as evidence that its support is unchanged.
#[derive(Debug)]
pub(crate) enum MechanismObservation {
    FloatingStateDtype(
        eredu_core::checkpoint::TensorDtype,
        Option<eredu_runtime::StateStorageDtype>,
    ),
    RowStorage(Option<eredu_runtime::AddressableStorageCapabilities>),
    RowWorkspace(
        eredu_runtime::RowLookupDescriptor,
        Option<Option<eredu_runtime::RowLookupWorkspace>>,
    ),
    RowDecodeMemory(
        eredu_runtime::RowLookupDescriptor,
        Option<eredu_nn::mechanism_memory::MechanismMemoryContract>,
    ),
    InputScoreWorkspace(Option<eredu_runtime::memory_estimation::InputScoreAttentionMechanism>),
    Observation(eredu_core::ObservationMechanisms),
    Capture(eredu_core::capture::CaptureCapabilities),
    Preparation(eredu_core::PreparationMechanismCapabilities),
    Grouped(eredu_runtime::GroupedOperationRequirement, bool),
    Text(
        ReplicatedTextRequirements,
        eredu_runtime::ReplicatedTextSelectionRequest,
        eredu_runtime::BackendMechanismCapabilities,
    ),
    Processor(eredu_runtime::MediaPrimitiveCapabilities),
    State(
        eredu_runtime::StateRealizationRequirements,
        eredu_runtime::CacheResidencyPolicy,
        eredu_runtime::StateMechanismCapabilities,
    ),
    Speculative(eredu_runtime::SpeculativeMechanismCapabilities),
    Communication(eredu_runtime::CommunicationCapabilities),
}

impl MechanismObservation {
    pub fn matches(&self, provider: &impl crate::PreparationMechanismProvider) -> bool {
        match self {
            Self::FloatingStateDtype(source, facts) => {
                *facts == provider.floating_state_dtype(source)
            }
            Self::RowStorage(facts) => *facts == provider.row_lookup_storage(),
            Self::RowWorkspace(descriptor, facts) => facts.as_ref().is_some_and(|facts| {
                provider
                    .row_lookup_workspace(descriptor)
                    .is_ok_and(|current| *facts == current)
            }),
            Self::RowDecodeMemory(descriptor, facts) => facts.as_ref().is_some_and(|facts| {
                provider
                    .row_lookup_decode_memory(descriptor)
                    .is_ok_and(|current| *facts == current)
            }),
            Self::InputScoreWorkspace(facts) => {
                *facts == provider.input_score_attention_workspace()
            }
            Self::Observation(facts) => *facts == provider.observation_mechanisms(),
            Self::Capture(facts) => *facts == provider.capture_capabilities(),
            Self::Preparation(facts) => *facts == provider.preparation_capabilities(),
            Self::Grouped(requirement, supported) => {
                *supported == provider.supports_grouped_operation(*requirement)
            }
            Self::Text(requirements, request, facts) => {
                *facts == provider.replicated_text_capabilities(requirements, request)
            }
            Self::Processor(facts) => *facts == provider.processor_capabilities(),
            Self::State(requirements, policy, facts) => {
                *facts == provider.state_capabilities(requirements, policy)
            }
            Self::Speculative(facts) => *facts == provider.speculative_capabilities(),
            Self::Communication(facts) => *facts == provider.communication_capabilities(),
        }
    }
}

pub(crate) struct RecordingMechanisms<'a, P> {
    pub provider: &'a P,
    pub observations: std::cell::RefCell<Vec<MechanismObservation>>,
}

impl<P: crate::PreparationMechanismProvider> crate::PreparationMechanismProvider
    for RecordingMechanisms<'_, P>
{
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_runtime::StateStorageDtype> {
        let facts = self.provider.floating_state_dtype(source);
        self.observations
            .borrow_mut()
            .push(MechanismObservation::FloatingStateDtype(
                source.clone(),
                facts,
            ));
        facts
    }

    fn row_lookup_storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        let facts = self.provider.row_lookup_storage();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::RowStorage(facts));
        facts
    }

    fn row_lookup_workspace(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        let facts = self.provider.row_lookup_workspace(descriptor);
        // Failures carry native causes without semantic equality. Re-evaluate a
        // failed selection rather than treating equal diagnostic text as a proof.
        self.observations
            .borrow_mut()
            .push(MechanismObservation::RowWorkspace(
                descriptor.clone(),
                facts.as_ref().ok().copied(),
            ));
        facts
    }

    fn row_lookup_decode_memory(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        let facts = self.provider.row_lookup_decode_memory(descriptor);
        self.observations
            .borrow_mut()
            .push(MechanismObservation::RowDecodeMemory(
                descriptor.clone(),
                facts.as_ref().ok().cloned(),
            ));
        facts
    }

    fn input_score_attention_workspace(
        &self,
    ) -> Option<eredu_runtime::memory_estimation::InputScoreAttentionMechanism> {
        let facts = self.provider.input_score_attention_workspace();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::InputScoreWorkspace(facts));
        facts
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        let facts = self.provider.observation_mechanisms();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Observation(facts));
        facts
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        let facts = self.provider.capture_capabilities();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Capture(facts.clone()));
        facts
    }

    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        let facts = self.provider.preparation_capabilities();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Preparation(facts));
        facts
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        let supported = self.provider.supports_grouped_operation(requirement);
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Grouped(requirement, supported));
        supported
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        let facts = self
            .provider
            .replicated_text_capabilities(requirements, request);
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Text(
                requirements.clone(),
                request.clone(),
                facts.clone(),
            ));
        facts
    }

    fn state_capabilities(
        &self,
        requirements: &eredu_runtime::StateRealizationRequirements,
        policy: &eredu_runtime::CacheResidencyPolicy,
    ) -> eredu_runtime::StateMechanismCapabilities {
        let facts = self.provider.state_capabilities(requirements, policy);
        self.observations
            .borrow_mut()
            .push(MechanismObservation::State(
                requirements.clone(),
                policy.clone(),
                facts.clone(),
            ));
        facts
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        let facts = self.provider.processor_capabilities();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Processor(facts.clone()));
        facts
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        let facts = self.provider.speculative_capabilities();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Speculative(facts.clone()));
        facts
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        let facts = self.provider.communication_capabilities();
        self.observations
            .borrow_mut()
            .push(MechanismObservation::Communication(facts.clone()));
        facts
    }
}

#[cfg(test)]
mod row_tests;
