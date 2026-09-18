//! Loaded declaration and account-only control source; no native/session owner.
use super::*;
use crate::composition::mlx::session::{OriginalInterventionDeclaration, intervention::NativeInterventionEstimator};
use eredu_core::{HostPreparationAuthority, intervention::PreparedInterventionPlanCopy,
    speculative::AdmittedSpeculativeActivations};
use eredu_runtime::{intervention::StaticInterventionPreflight,
    working_memory::WorkingMemoryPool};

pub(super) struct Source {
    declaration: OriginalInterventionDeclaration,
    pool: WorkingMemoryPool,
    funding: HostMetadataFunding,
}
impl Source {
    pub(super) fn prepare(sources: &OriginalSpeculativeNumericalSources) -> Result<Self, Error> {
        let funding = sources.metadata_funding();
        funding.reserve_metadata(size_of::<(Self, Result<Self, Error>, &OriginalSpeculativeNumericalSources)>())
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self { declaration: sources.activation_control_declaration()?,
            pool: sources.pool().clone(), funding: funding.clone() })
    }
    pub(super) fn pool(&self) -> &WorkingMemoryPool { &self.pool }
    pub(super) fn validate(&self, state: &OriginalSpeculativeCapture, plan: &AdmittedSpeculativeActivations) -> Result<(), Error> {
        let bytes = OriginalInterventionDeclaration::activation_validation_control_bytes()
            .and_then(|n| n.checked_add(size_of::<(&Self, &OriginalSpeculativeCapture, &AdmittedSpeculativeActivations, Result<(), Error>)>()))
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        self.funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
        state.validate_control_plan(plan).map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        self.declaration.validate_activations(plan)
    }
    pub(super) fn candidate(&self, state: &OriginalSpeculativeCapture, plan: &AdmittedSpeculativeActivations)
        -> Result<(Option<OriginalInterventionSource>, eredu_core::capture::CaptureUsage), Error>
    {
        let frames = [
            size_of::<(&Self, &OriginalSpeculativeCapture, &AdmittedSpeculativeActivations)>(),
            size_of::<(Option<OriginalInterventionSource>, eredu_core::capture::CaptureUsage)>(),
            size_of::<Result<(Option<OriginalInterventionSource>, eredu_core::capture::CaptureUsage), Error>>(),
            size_of::<Result<OriginalInterventionSource, eredu_runtime::working_memory::OriginalInterventionSourceError>>(),
            size_of::<Result<StaticInterventionPreflight, eredu_runtime::intervention::StaticInterventionScratchError>>(),
            size_of::<Result<(), eredu_runtime::intervention::StaticInterventionPreflightError>>(),
            PreparedInterventionPlanCopy::inspection_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            OriginalInterventionDeclaration::activation_validation_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            NativeInterventionEstimator::prepared_preflight_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            StaticInterventionPreflight::required_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            OriginalCaptureSource::validation_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            OriginalInterventionSource::validation_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        ];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let retain = |cause| retain_planning_error(cause, self.funding.clone());
        self.validate(state, plan)?;
        let inherited = state.control_usage().map_err(retain)?;
        let source = if plan.interventions().is_empty() { None } else {
            let copy = PreparedInterventionPlanCopy::inspect(plan.interventions())
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
            Some(self.pool.compile_intervention_source(copy)
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?)
        };
        let host = HostPreparationAuthority::retain(self.funding.clone());
        let mut scratch = StaticInterventionPreflight::prepare(host)
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let result = match &source {
            Some(source) => scratch.run_invocation_with_source(state.source().plan().admission(), source.plan(), &NativeInterventionEstimator, inherited),
            None => scratch.run_invocation_without_evidence(state.source().plan().admission(), plan.interventions(), &NativeInterventionEstimator, inherited),
        };
        result.map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        Ok((source, inherited))
    }
}
