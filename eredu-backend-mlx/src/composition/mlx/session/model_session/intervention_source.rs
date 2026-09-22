//! Actual loaded declaration loan with a freshly paid session identity.
use super::*;
use eredu_architectures::prepared_sources::PreparedModelDiscovery;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
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
    partition: Option<partition_capture::LoadedPartitionCapture>,
    session: String,
    origin: ReplicatedTextControlOrigin,
    activation_execution:
        Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>,
    active_overlay: Option<String>,
    _funding: HostMetadataFunding,
}
impl OriginalInterventionDeclaration {
    pub(crate) fn with_autoregressive_execution(
        mut self,
        execution: eredu_architectures::speculative_execution::SpeculativeActivationExecution,
    ) -> Self {
        self.activation_execution = Some(execution);
        self
    }
    #[cfg(test)]
    pub(crate) fn activation_discovery(
        &self,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        let execution = self.activation_execution.as_ref().ok_or_else(|| {
            eredu_core::capture::CaptureError::Unsupported(
                "missing selected activation traversal".into(),
            )
        })?;
        match &self.partition {
            None => self.prepared.speculative_activations(
                execution,
                &super::super::intervention::mechanisms(),
                &self.session,
                self.active_overlay.as_deref(),
            ),
            Some(partition) => {
                let mut report = self
                    .prepared
                    .speculative_activations_with_partition_support(
                        execution,
                        &super::super::intervention::mechanisms(),
                        &self.session,
                        self.active_overlay.as_deref(),
                        partition.layouts(),
                        |_| {
                            eredu_core::ObservationSupportStatus::Unverified(
                                "retained phase report follows".into(),
                            )
                        },
                    )?;
                report.captures.support = partition.discovery.support.clone();
                for point in &mut report.interventions.points {
                    let status = partition
                        .discovery
                        .support
                        .points
                        .iter()
                        .find(|row| row.path == point.path)
                        .ok_or_else(|| {
                            eredu_core::capture::CaptureError::Unsupported(
                                "loaded partition support is absent".into(),
                            )
                        })?;
                    point.prefill = status.prefill.clone();
                    point.decode = status.decode.clone();
                }
                Ok(report)
            }
        }
    }

    /// Paid projection of the actual loaded declaration for the outer control
    /// observer. Catalog and origin remain exact aliases; only identity text is
    /// copied. No session, transport, native root or callback is retained.
    pub(crate) fn copy_for_control(&self, funding: &HostMetadataFunding) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(&Self, &HostMetadataFunding)>(),
            size_of::<Option<String>>(),
            size_of::<
                Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>,
            >(),
        ];
        funding
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let session = copy_identity_text(&self.session, funding)?;
        let active_overlay = self
            .active_overlay
            .as_deref()
            .map(|text| copy_identity_text(text, funding))
            .transpose()?;
        Ok(Self {
            prepared: self.prepared.clone(),
            partition: self.partition.clone(),
            session,
            origin: self.origin.clone(),
            activation_execution: self.activation_execution.clone(),
            active_overlay,
            _funding: funding.clone(),
        })
    }
    /// Admit a fresh logical declaration from the loaded source and the
    /// actual absolute capture geometry. The intermediate and its diagnostics
    /// remain under this declaration's actual planning account until the
    /// immutable source compiler has produced its independent retained owner.
    pub(crate) fn compile_source(
        &self,
        raw: &eredu_core::intervention::InterventionPlan,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        session_id: &str,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<eredu_runtime::working_memory::OriginalInterventionSource, Error> {
        use eredu_core::{
            HostPreparationAuthority,
            intervention::{
                AdmittedInterventionPlan, InterventionAdmissionError,
                PreparedInterventionAdmission, PreparedInterventionPlanCopy,
            },
        };
        use eredu_runtime::{
            inspection::FundedInterventionDiscovery,
            working_memory::{OriginalInterventionSource, OriginalInterventionSourceError},
        };
        let funding = &self._funding;
        let err =
            |cause| crate::composition::mlx::model::retain_planning_error(cause, funding.clone());
        let controls = [
            PreparedInterventionAdmission::inspection_control_bytes(),
            PreparedInterventionPlanCopy::inspection_control_bytes(),
            Self::validation_control_bytes(),
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>(),
            Some(size_of::<(
                FundedInterventionDiscovery,
                HostPreparationAuthority,
                AdmittedInterventionPlan,
                PreparedInterventionPlanCopy<'_>,
                OriginalInterventionSource,
                Result<OriginalInterventionSource, Error>,
            )>()),
            eredu_core::BackendFailure::source_retention_peak_bytes::<InterventionAdmissionError>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                OriginalInterventionSourceError,
            >(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), |sum, part| sum.checked_add(part?))
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let facts = super::super::intervention::mechanism_facts();
        let discovery = match &self.partition {
            Some(partition) => {
                let (_, execution, _) = partition.source_labels();
                self.prepared.prepare_partitioned_intervention_discovery(
                    raw,
                    facts,
                    &self.session,
                    partition.layouts(),
                    &partition.discovery,
                    execution,
                    funding,
                )
            }
            None => {
                self.prepared
                    .prepare_intervention_discovery(raw, facts, &self.session, funding)
            }
        }
        .map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        let plan = PreparedInterventionAdmission::inspect(
            raw,
            discovery.discovery(),
            capture.request(),
            capture.invocation_bounds(),
            capture.text_origin().unwrap_or_default(),
            session_id,
        )
        .map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        funding
            .reserve_metadata(plan.required_bytes())
            .map_err(Error::WorkspacePlanning)?;
        let host = HostPreparationAuthority::retain(funding.clone());
        let admitted = plan.construct(&host).map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        self.validate(&admitted)?;
        let copy = PreparedInterventionPlanCopy::inspect(&admitted).map_err(err)?;
        pool.compile_intervention_source(copy).map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })
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
        let execution = self
            .activation_execution
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let facts = super::super::intervention::mechanism_facts();
        let routed =
            super::super::intervention::PreparedModelInterventions::routed_mechanism_facts();
        let result = match &self.partition {
            None => self
                .prepared
                .validate_original_speculative_activations_with_routed_interventions(
                    plan,
                    execution,
                    &self.session,
                    self.active_overlay.as_deref(),
                    facts,
                    routed,
                ),
            Some(partition) => self
                .prepared
                .validate_original_partitioned_autoregressive_activations(
                    plan,
                    execution,
                    &self.session,
                    self.active_overlay.as_deref(),
                    facts,
                    routed,
                    partition.layouts(),
                    &partition.discovery,
                    partition.source_labels().1,
                ),
        };
        result.map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, self._funding.clone())
        })
    }
    pub(crate) fn validate(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), Error> {
        let facts = super::super::intervention::mechanism_facts();
        let result = match &self.partition {
            Some(partition) => {
                let (_, execution, _) = partition.source_labels();
                self.prepared.validate_partitioned_prepared_intervention(
                    plan,
                    facts,
                    &self.session,
                    partition.layouts(),
                    &partition.discovery,
                    execution,
                )
            }
            None => self
                .prepared
                .validate_prepared_intervention(plan, facts, &self.session),
        };
        result.map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, self._funding.clone())
        })
    }
}
impl MlxModelSession {
    pub(crate) fn original_intervention_declaration(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Option<OriginalInterventionDeclaration>, Error> {
        let frames = [
            size_of::<OriginalInterventionDeclaration>(),
            size_of::<Option<partition_capture::LoadedPartitionCapture>>(),
            size_of::<Option<&partition_capture::LoadedPartitionCapture>>(),
            size_of::<(&str, &str, eredu_runtime::CommunicationSessionIdentity)>(),
            size_of::<
                Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution>,
            >(),
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
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                eredu_runtime::replicated_session::PreparedControlBindingError,
            >()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
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
            eredu_core::BackendFailure::source_retention_peak_bytes::<Error>()
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let err =
            |cause| crate::composition::mlx::model::retain_planning_error(cause, funding.clone());
        let Some(prepared) = &self.capture_discovery else {
            return Ok(None);
        };
        // Compile this exact loaded source under its original account if it is
        // still cold, then retain the immutable layouts/support owner for quote.
        let partition = match &self.payload.distributed {
            None => None,
            Some(transport) => {
                let loaded = self.original_partition_capture(funding)?.ok_or_else(|| {
                    err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
                })?;
                let (_, execution, setup) = loaded.source_labels();
                if setup != transport.session_identity()
                    || setup.participant_count() != loaded.layouts().topology().world_size()
                    || execution != prepared.execution_identity()
                {
                    return Err(err(Error::PrefillControl(
                        WorkingMemoryError::IdentityMismatch,
                    )));
                }
                Some(loaded.clone())
            }
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
        let active_overlay = self
            .payload
            .parameter_state
            .active
            .as_deref()
            .map(|text| copy_identity_text(text, funding))
            .transpose()?;
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

fn copy_identity_text(source: &str, funding: &HostMetadataFunding) -> Result<String, Error> {
    let frames = [
        size_of::<Vec<u8>>(),
        size_of::<String>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<Result<String, std::string::FromUtf8Error>>(),
        size_of::<Result<String, Error>>(),
        size_of::<(&str, &HostMetadataFunding)>(),
    ];
    let bytes = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|n| n.checked_add(source.len()))
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    let mut text = Vec::new();
    text.try_reserve_exact(source.len()).map_err(|cause| {
        crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
    })?;
    if text.capacity() != source.len() {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
    }
    text.extend_from_slice(source.as_bytes());
    Ok(String::from_utf8(text).expect("copied valid UTF-8"))
}
