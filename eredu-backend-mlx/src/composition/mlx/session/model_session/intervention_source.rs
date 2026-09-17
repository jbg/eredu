//! Actual loaded declaration loan with a freshly paid session identity.
use super::*;
use eredu_architectures::prepared_sources::PreparedModelDiscovery;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::{
    replicated_session::ReplicatedTextControlOrigin, working_memory::WorkingMemoryError,
};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

/// Only this loaded-session constructor can create the source. The selected
/// declaration Arc already belongs to the retained model; no caller DTO or
/// native transport is adopted. Fresh String/control births use actual source H.
pub(crate) struct OriginalInterventionDeclaration {
    prepared: Arc<PreparedModelDiscovery>,
    partition: Option<Arc<partition_capture::LoadedPartitionCapture>>,
    session: String,
    origin: ReplicatedTextControlOrigin,
    activation_execution: Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>,
    active_overlay: Option<String>,
    _funding: WorkspaceMetadataFunding,
}
impl OriginalInterventionDeclaration {
    /// Paid projection of the actual loaded declaration for the outer control
    /// observer. Catalog and origin remain exact aliases; only identity text is
    /// copied. No session, transport, native root or callback is retained.
    pub(crate) fn copy_for_control(&self, funding: &WorkspaceMetadataFunding) -> Result<Self, Error> {
        let parts = [size_of::<Self>(), size_of::<Result<Self, Error>>(),
            size_of::<(&Self, &WorkspaceMetadataFunding)>(),
            size_of::<Option<String>>(), size_of::<Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>>()];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let session = copy_identity_text(&self.session, funding)?;
        let active_overlay = self.active_overlay.as_deref().map(|text| copy_identity_text(text, funding)).transpose()?;
        Ok(Self { prepared: self.prepared.clone(), partition: self.partition.clone(), session, origin: self.origin.clone(),
            activation_execution: self.activation_execution.clone(), active_overlay, _funding: funding.clone() })
    }
    pub(crate) fn matches_origin(&self, origin: &ReplicatedTextControlOrigin) -> bool {
        self.origin.same_origin(origin)
    }
    pub(crate) fn validation_control_bytes() -> Option<usize> {
        PreparedModelDiscovery::partitioned_intervention_validation_control_bytes()?
            .checked_add(size_of::<&Self>())?
            .checked_add(size_of::<Result<(), Error>>())
    }
    pub(crate) fn activation_validation_control_bytes() -> Option<usize> {
        PreparedModelDiscovery::original_speculative_activation_validation_control_bytes()?
            .checked_add(size_of::<&Self>())?
            .checked_add(size_of::<Result<(), Error>>())
    }
    pub(crate) fn validate_activations(
        &self,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), Error> {
        let execution = self.activation_execution.as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        self.prepared.validate_original_speculative_activations_with_routed_interventions(
            plan, execution, &self.session, self.active_overlay.as_deref(),
            super::super::intervention::mechanism_facts(),
            super::super::intervention::PreparedModelInterventions::routed_mechanism_facts(),
        ).map_err(|cause| crate::composition::mlx::model::retain_planning_error(
            cause, self._funding.clone(),
        ))
    }
    pub(crate) fn validate(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), Error> {
        let facts=super::super::intervention::mechanism_facts();
        let result=match &self.partition {
            Some(partition)=>{
                let (_,execution,_)=partition.source_labels();
                self.prepared.validate_partitioned_static_intervention(plan,facts,&self.session,
                    partition.layouts(),&partition.discovery,execution)
            },
            None=>self.prepared.validate_static_intervention(plan,facts,&self.session),
        };
        result.map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, self._funding.clone())
            })
    }
}
impl MlxModelSession {
    pub(crate) fn original_intervention_declaration(
        &self,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Option<OriginalInterventionDeclaration>, Error> {
        let frames = [
            size_of::<OriginalInterventionDeclaration>(),
            size_of::<Option<Arc<partition_capture::LoadedPartitionCapture>>>(),
            size_of::<Option<&Arc<partition_capture::LoadedPartitionCapture>>>(),
            size_of::<(&str,&str,eredu_runtime::CommunicationSessionIdentity)>(),
            size_of::<Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>>(),
            size_of::<Option<String>>(),
            size_of::<Option<OriginalInterventionDeclaration>>(),
            size_of::<Result<Option<OriginalInterventionDeclaration>, Error>>(),
            size_of::<String>(),
            size_of::<Vec<u8>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<String, std::string::FromUtf8Error>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                std::collections::TryReserveError,
            >()
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                eredu_runtime::replicated_session::PreparedControlBindingError,
            >()
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?,
            size_of::<
                Result<
                    ReplicatedTextControlOrigin,
                    eredu_runtime::replicated_session::PreparedControlBindingError,
                >,
            >(),
            size_of::<
                Option<
                    Result<
                        ReplicatedTextControlOrigin,
                        eredu_runtime::replicated_session::PreparedControlBindingError,
                    >,
                >,
            >(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<Error>().ok_or(
                Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow),
            )?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let err =
            |cause| crate::composition::mlx::model::retain_planning_error(cause, funding.clone());
        let Some(prepared) = &self.capture_discovery else {
            return Ok(None);
        };
        // Public admission already resolved this exact loaded source. Keep its
        // immutable layouts/support owner; never rebuild discovery under quote.
        let partition=match &self.payload.distributed {
            None=>None,
            Some(transport)=>{
                let loaded=self.partition_capture.get().and_then(|source|source.as_ref().ok())
                    .ok_or_else(||err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)))?;
                let (_,execution,setup)=loaded.source_labels();
                if setup!=transport.session_identity() || setup.participant_count()!=loaded.layouts().topology().world_size()
                    || execution!=prepared.execution_identity() {
                    return Err(err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)));
                }
                Some(loaded.clone())
            },
        };
        let model = self
            .original_model_source()
            .map_err(|cause| err(Error::PrefillControl(cause)))?;
        let origin = model
            .erased()
            .resident_control_origin_fixed()
            .ok_or_else(|| err(Error::PrefillControl(WorkingMemoryError::UnknownBound)))?
            .map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
            })?;
        let session = copy_identity_text(&self.intervention_session_identity, funding)?;
        let active_overlay = self.payload.parameter_state.active.as_deref()
            .map(|text| copy_identity_text(text, funding)).transpose()?;
        let activation_execution = model.erased().speculative_activation_execution();
        Ok(Some(OriginalInterventionDeclaration {
            prepared: prepared.clone(),
            partition,
            session,
            origin,
            activation_execution,
            active_overlay,
            _funding: funding.clone(),
        }))
    }
}

fn copy_identity_text(source: &str, funding: &WorkspaceMetadataFunding) -> Result<String, Error> {
    let frames = [
        size_of::<Vec<u8>>(), size_of::<String>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<Result<String, std::string::FromUtf8Error>>(),
        size_of::<Result<String, Error>>(),
        size_of::<(&str, &WorkspaceMetadataFunding)>(),
    ];
    let bytes = frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|n| n.checked_add(source.len()))
        .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
    funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
    let mut text = Vec::new();
    text.try_reserve_exact(source.len()).map_err(|cause|
        crate::composition::mlx::model::retain_planning_error(cause, funding.clone()))?;
    if text.capacity() != source.len() {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
    }
    text.extend_from_slice(source.as_bytes());
    Ok(String::from_utf8(text).expect("copied valid UTF-8"))
}
