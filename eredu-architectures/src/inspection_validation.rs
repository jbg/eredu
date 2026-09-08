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
    Speculative(eredu_runtime::SpeculativeMechanismCapabilities),
    Communication(eredu_runtime::CommunicationCapabilities),
}

impl MechanismObservation {
    pub fn matches(&self, provider: &impl crate::PreparationMechanismProvider) -> bool {
        match self {
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
