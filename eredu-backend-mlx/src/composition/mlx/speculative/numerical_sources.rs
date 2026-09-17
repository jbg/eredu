//! Common source prerequisites for original speculative numerical programs.
//! Model occurrence schedules remain owned by their exact AR/Embedded binding.
use super::*;
use crate::{
    backend::{OriginalCopyEnvironment, nn::workspace::MlxMetalWorkspaceMechanisms},
    composition::mlx::{
        model::{Executable, retain_planning_error},
        session::OriginalInterventionDeclaration,
    },
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::{
    replicated_session::ReplicatedTextControlOrigin,
    working_memory::{OriginalSpeculativeRequest, WorkingMemoryError, WorkingMemoryPool},
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum SourceError {
    #[error("original numerical prerequisites require the exact retained target source")]
    Source,
}
/// Closed request ownership shared by sampling, capture, random programs and
/// controller snapshots. The request retains its private schedule tag and exact
/// issuance account; this owner neither relabels it nor grants a model role.
/// Native facts are borrowed from the selected executable's existing runtime.
pub(crate) struct OriginalSpeculativeNumericalSources {
    target: ReplicatedTextControlOrigin,
    blueprint: eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
    roots: safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    indexed: Vec<(u32,crate::backend::runtime::residency::parameter_bank::IndexedBankSource)>,
    intervention_declaration: Option<OriginalInterventionDeclaration>,
    request: OriginalSpeculativeRequest,
    pool: WorkingMemoryPool,
    funding: WorkspaceMetadataFunding,
}
/// Immutable prerequisites captured from the actual executable before its
/// disjoint selected strategy and mutable session are lent to an executor.
/// This move-only source grants no request, occurrence, or native authority.
pub(crate) struct OriginalSpeculativeNumericalPreparation {
    target: ReplicatedTextControlOrigin,
    blueprint: eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
    roots: safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    indexed: Vec<(u32,crate::backend::runtime::residency::parameter_bank::IndexedBankSource)>,
    execution: eredu_runtime::working_memory::InferenceExecutionIdentity,
    pool: WorkingMemoryPool,
    funding: WorkspaceMetadataFunding,
}
impl OriginalSpeculativeNumericalPreparation {
    pub(crate) fn prepare(
        target: &Executable,
        pool: &WorkingMemoryPool,
        funding: WorkspaceMetadataFunding,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<eredu_runtime::working_memory::InferenceExecutionIdentity>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(
                &Executable,
                &WorkingMemoryPool,
                WorkspaceMetadataFunding,
            )>(),
            size_of::<safemlx::PrefillRootsRuntime>(),
            size_of::<Result<safemlx::PrefillRootsRuntime, Error>>(),
            size_of::<Option<MlxMetalWorkspaceMechanisms>>(),
            size_of::<ReplicatedTextControlOrigin>(),
            size_of::<
                Option<
                    Result<
                        ReplicatedTextControlOrigin,
                        eredu_runtime::replicated_session::PreparedControlBindingError,
                    >,
                >,
            >(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<SourceError>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<SourceError>().ok_or(
                Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow),
            )?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<WorkingMemoryError>().ok_or(
                Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow),
            )?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<
                eredu_runtime::replicated_session::PreparedControlBindingError,
            >()
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Error>().ok_or(
                Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow),
            )?,
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let origin = target
            .erased()
            .resident_control_origin_fixed()
            .ok_or_else(|| retain_planning_error(SourceError::Source, funding.clone()))?
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        // Cloning the blueprint aliases the already retained source graph and
        // exact selection. No artifact, configuration or task map is reopened.
        let blueprint = target.inference_blueprint()
            .ok_or_else(|| retain_planning_error(SourceError::Source, funding.clone()))?
            .clone();
        let roots = target
            .erased()
            .prefill_roots_runtime()
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let mechanisms = target
            .workspace_mechanisms()
            .ok_or_else(|| retain_planning_error(SourceError::Source, funding.clone()))?;
        let banks=target.erased().indexed_bank_sources();
        let mut indexed=funding.metadata_vec(banks.map_or(0,|banks|banks.len()))
            .map_err(|cause|retain_planning_error(cause,funding.clone()))?;
        if let Some(banks)=banks {
            for (bank,source) in banks {indexed.push((bank.value(),source.clone()));}
        }
        Ok(Self {
            target: origin,
            blueprint,
            roots,
            mechanisms,
            indexed,
            execution: target.erased().inference_execution_identity().clone(),
            pool: pool.clone(),
            funding,
        })
    }
    pub(crate) fn execution_identity(&self) -> &eredu_runtime::working_memory::InferenceExecutionIdentity {
        &self.execution
    }
    pub(crate) fn pool(&self) -> &WorkingMemoryPool { &self.pool }
    pub(crate) fn metadata_funding(&self) -> &WorkspaceMetadataFunding { &self.funding }

    /// Consumes the source only after the selected schedule has opened its exact
    /// request. Equal geometry cannot replace its pool or execution identity.
    pub(crate) fn bind(self, request: OriginalSpeculativeRequest) -> Result<OriginalSpeculativeNumericalSources, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<OriginalSpeculativeRequest>(),
            size_of::<OriginalSpeculativeNumericalSources>(),
            size_of::<Result<OriginalSpeculativeNumericalSources, Error>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        request.validate_pool(&self.pool)
            .and_then(|()| request.validate_execution(&self.execution))
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        Ok(OriginalSpeculativeNumericalSources {
            target: self.target, blueprint: self.blueprint, roots: self.roots,
            mechanisms: self.mechanisms, indexed:self.indexed, intervention_declaration: None,
            request, pool: self.pool, funding: self.funding,
        })
    }
}
impl OriginalSpeculativeNumericalSources {
    /// One invocation's quote metadata has its own lifetime. The same pool,
    /// execution and request ceiling still admit every constructor. Outputs,
    /// completion recovery and errors retain this account through their usual
    /// funding aliases; the selected-source owner does not retain it.
    pub(crate) fn prepare_phase_metadata(&self) -> Result<WorkspaceMetadataFunding, Error> {
        let parts = [
            size_of::<&Self>(),
            size_of::<WorkspaceMetadataFunding>(),
            size_of::<Result<WorkspaceMetadataFunding, WorkspaceMetadataFundingError>>(),
            size_of::<Result<WorkspaceMetadataFunding, Error>>(),
        ];
        self.funding.reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        self.pool.prepare_workspace_metadata(
            self.request.execution_identity(), self.request.capacity_bytes(),
        ).map_err(Error::WorkspacePlanning)
    }
    /// Keeps existing callers on the same prerequisite capture and request bind.
    pub(crate) fn prepare(
        target: &Executable,
        request: OriginalSpeculativeRequest,
        pool: &WorkingMemoryPool,
        funding: WorkspaceMetadataFunding,
    ) -> Result<Self, Error> {
        OriginalSpeculativeNumericalPreparation::prepare(target, pool, funding)?.bind(request)
    }
    /// Quote the actual retained target bank sources in this phase's account.
    /// The directory carries no source allowance or mutable invocation state.
    pub(crate) fn target_addressable_sources(&self,
        mechanism:crate::backend::nn::workspace::ResidentExecutionMechanisms,
        environment:&OriginalCopyEnvironment<'_>,funding:&WorkspaceMetadataFunding,
    )->Result<Option<crate::backend::nn::workspace::AddressableSources>,Error> {
        self.validate_environment(environment)?;
        if self.indexed.is_empty(){return Ok(None);}
        funding.reserve_metadata(size_of::<(Option<crate::backend::nn::workspace::AddressableSources>,
            Result<Option<crate::backend::nn::workspace::AddressableSources>,Error>,
            &Self,&OriginalCopyEnvironment<'_>,&WorkspaceMetadataFunding)>())
            .map_err(Error::WorkspacePlanning)?;
        let runtime=environment.input_runtime().map_err(|cause|retain_planning_error(cause,funding.clone()))?;
        crate::backend::nn::workspace::AddressableSources::new(
            self.indexed.iter().map(|(bank,source)|(*bank,source)),mechanism,&runtime,Some(&self.pool),funding,
        ).map(Some).map_err(|cause|retain_planning_error(cause,funding.clone()))
    }
    pub(crate) fn target_origin(&self) -> &ReplicatedTextControlOrigin {
        &self.target
    }
    pub(crate) fn target_blueprint(
        &self,
    ) -> &eredu_architectures::prepared_execution::PreparedInferenceBlueprint {
        &self.blueprint
    }
    pub(crate) fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }
    pub(crate) fn request(&self) -> &OriginalSpeculativeRequest {
        &self.request
    }
    pub(crate) fn metadata_funding(&self) -> &WorkspaceMetadataFunding {
        &self.funding
    }
    pub(crate) fn numerical_prerequisites(
        &self,
    ) -> (&safemlx::PrefillRootsRuntime, MlxMetalWorkspaceMechanisms) {
        (&self.roots, self.mechanisms)
    }
    /// Exact pool comparison only; actual consumers retain their existing source,
    /// stream, scope and completion checks before any numerical operation.
    pub(crate) fn validate_environment(
        &self,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<(), Error> {
        self.request
            .validate_pool(environment.pool())
            .map_err(|cause| self.retain_startup_error(cause))
    }
    pub(crate) fn with_intervention_declaration(
        mut self,
        declaration: Option<OriginalInterventionDeclaration>,
    ) -> Result<Self, Error> {
        if declaration
            .as_ref()
            .is_some_and(|source| !source.matches_origin(&self.target))
        {
            return Err(self.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        self.intervention_declaration = declaration;
        Ok(self)
    }
    pub(crate) fn validate_intervention(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), Error> {
        let controls = OriginalInterventionDeclaration::validation_control_bytes().ok_or(
            Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow),
        )?;
        self.funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let declaration = self
            .intervention_declaration
            .as_ref()
            .ok_or_else(|| self.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        if !declaration.matches_origin(&self.target) {
            return Err(self.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        declaration.validate(plan)
    }
    pub(crate) fn activation_control_declaration(&self) -> Result<OriginalInterventionDeclaration, Error> {
        let declaration = self.intervention_declaration.as_ref()
            .ok_or_else(|| self.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        if !declaration.matches_origin(&self.target) {
            return Err(self.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        declaration.copy_for_control(&self.funding)
    }
    pub(crate) fn validate_activations(
        &self, plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), Error> {
        let controls = OriginalInterventionDeclaration::activation_validation_control_bytes()
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
        self.funding.reserve_metadata(controls).map_err(Error::WorkspacePlanning)?;
        let declaration = self.intervention_declaration.as_ref()
            .ok_or_else(|| self.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        if !declaration.matches_origin(&self.target) {
            return Err(self.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        declaration.validate_activations(plan)
    }
    #[track_caller]
    pub(crate) fn retain_error(&self, cause: Error) -> Error {
        if std::env::var_os("EREDU_EXTERNAL_FLOW_TRACE").is_some(){eprintln!("EXTERNAL_RETAIN_ERROR {}: {}",std::panic::Location::caller(),cause);}
        match cause.take_retained_backend_failure() {
            Ok(cause) => Error::StorageSource(cause),
            Err(cause) => retain_planning_error(cause, self.funding.clone()),
        }
    }
    #[track_caller]
    pub(crate) fn retain_startup_error<E: std::error::Error + Send + Sync + 'static>(
        &self,
        cause: E,
    ) -> Error {
        if std::env::var_os("EREDU_EXTERNAL_FLOW_TRACE").is_some(){eprintln!("EXTERNAL_RETAIN_STARTUP {}: {}",std::panic::Location::caller(),cause);}
        retain_planning_error(cause, self.funding.clone())
    }
}
