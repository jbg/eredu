//! Cold, request-bound evidence for ordinary token-ID generation.
//!
//! Construction does not enter native work or change the loading authority.
//! A reservation alone cannot construct this proof. Its caller must additionally
//! bind the original core run before preparing inputs and retain each issued
//! step through commitment, completion recovery and escaped token storage.

use super::*;
use eredu_core::{
    AdmissionRequest, CapabilityError, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, OutputDemand, RuntimeStateEstimate, TextControllerContract,
    TextControllerWorkspace, TextPreparationInput, TokenFilterController, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    ControllerStorageContract, ControllerStorageError, ControllerWorkspaceContribution,
    IncrementalInferenceQuote, InferenceRequest, InferenceTextPreparation,
    RegisteredControllerStorage, WorkingMemoryError, WorkingMemoryPool,
};

mod candidates;
mod capture;
pub(super) use capture::OriginalPartitionCaptureFrame;
mod installation;
mod observed;
pub(in crate::composition::mlx::session::model_session) mod original_prepared;
pub(super) use observed::quote_completed_input;
#[cfg(test)]
pub(super) use original_prepared::take_original_cold_facts;
mod prediction;
mod sampling_revision;
mod control;
mod branch;
pub(in crate::composition::mlx::session::model_session) use branch::exchange as exchange_branch;
mod preparation;
pub(crate) fn prediction_scope_facts()
-> Result<eredu_runtime::working_memory::TextPredictionScopeFacts, Error> {
    prediction::facts()
}
mod sequence;
pub(super) mod token_input;
pub(crate) fn preparation_scope_facts()
-> Result<eredu_runtime::working_memory::TextPreparationScopeFacts, Error> {
    preparation::facts()
}
use sequence::{AdmissionFailure, SequenceQuotation};
#[cfg(test)]
pub(super) mod opening_rows_fixture;
#[cfg(test)]
pub(super) use capture::funding_probe::{CaptureFundingProbe, CapturedFundingQuote};
pub(super) use capture::CaptureAdmission;
use capture::CaptureQuotation;
mod coordinates;
mod graph;
mod opening;
mod original_table;
mod owner;
mod pipeline_cache;
mod tracking;
pub(in crate::composition::mlx::session) use owner::TextExecutionQuoteOwner;
mod resume;
pub(in crate::composition::mlx::session::model_session) use candidates::TextWorkspaceCandidate;
use candidates::plan_candidates_with_handoff_retained;
use coordinates::PredictionCoordinates;
use opening::OpeningSeal;
pub(in crate::composition::mlx::session) use resume::{
    PendingSavedTextAdmission, admit_original_saved, admit_saved,
    original_resume_admission_control_bytes, seal_saved_native_quote,
};

/// The concrete quote allocation, consuming retirement and common original
/// source-staging controls. This enters capture/sequence Q once, or the legacy
/// unheld diagnostic once, including branches with no decoder source.
pub(super) fn owner_control_bytes() -> Result<u64, Error> {
    owner::control_bytes()
        .and_then(|n| n.checked_add(installation::control_bytes()?))
        .and_then(|n| n.checked_add(original_prepared::prompt_binding_control_bytes()?))
        .and_then(|n| n.checked_add(sequence::decoder_staging_control_bytes()?))
        .and_then(|n| n.checked_add(prediction::bank_transfer_bytes()?))
        .and_then(|n| n.checked_add(token_input::control_bytes()?))
        .and_then(|n| n.checked_add(original_table::admission_controls()?))
        .and_then(|n| n.checked_add(crate::backend::runtime::execution::generic::OriginalOperationRegistration::control_bytes()?))
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

pub(super) fn original_table_work_control_bytes() -> Option<u64> {
    original_table::work_controls()
}

/// Exactly one layerwise identity owner. Native installation retains the full
/// workspace; legacy execution moves only its identity and drops cold sources.
#[derive(Debug)]
enum LayerwiseQuoteSources {
    Identity(crate::backend::runtime::execution::generic::LayerwiseWorkspaceIdentity),
    Retained(crate::backend::runtime::execution::generic::LayerwiseWorkspace),
}
impl LayerwiseQuoteSources {
    fn identity(&self) -> &crate::backend::runtime::execution::generic::LayerwiseWorkspaceIdentity {
        match self {
            Self::Identity(identity) => identity,
            Self::Retained(workspace) => workspace.identity(),
        }
    }
    fn retained_workspace(
        &self,
    ) -> Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace> {
        match self {
            Self::Identity(_) => None,
            Self::Retained(workspace) => Some(workspace),
        }
    }
}

/// Immutable evidence created only after the complete native quote reserves its
/// actual domain. Clones of the containing Rc preserve this exact request; they
/// do not create new step or output allowances.
#[derive(Debug)]
pub(in crate::composition::mlx::session) struct TextExecutionQuote {
    config: TextGenerationConfig,
    coordinates: PredictionCoordinates,
    controller: TextControllerContract,
    storage_contract: ControllerStorageContract,
    session: Rc<Cell<bool>>,
    model_pool: WorkingMemoryPool,
    context_pool: WorkingMemoryPool,
    parameter_epoch: u64,
    layerwise: Option<LayerwiseQuoteSources>,
    _host_sources: Option<crate::backend::runtime::residency::manager::HostCopySourcePins>,
    disk: Option<crate::backend::runtime::execution::generic::DiskLayerwiseReceipt>,
    source_capacity_bytes: u64,
    request: InferenceRequest,
    opening: OpeningSeal,
    capture: Option<CaptureQuotation>,
    funding: RefCell<Option<eredu_runtime::working_memory::WorkingMemoryFundingRun>>,
    sequence: Option<SequenceQuotation>,
    pub(super) preparation_scopes: Option<preparation::PreparationScopes>,
    prediction_scopes: Option<prediction::PredictionScopes>,
    operation_host_destinations:
        RefCell<Option<eredu_runtime::working_memory::OriginalHostDestinationBank>>,
    operation_banks:
        RefCell<Option<crate::backend::runtime::execution::generic::OriginalOperationBankOwner>>,
    operation_registration:
        Option<crate::backend::runtime::execution::generic::OriginalOperationRegistration>,
    prefill_scopes: Option<RefCell<eredu_runtime::working_memory::OriginalTextPrefillScopes>>,
    record_quota: Option<safemlx::SubmissionRecordQuota>,
    graph_quota: Option<safemlx::SubmissionGraphQuota>,
    sampling_revision: RefCell<Option<sampling_revision::SamplingRevisionOwner>>,
    native_recipe: Option<crate::backend::nn::workspace::ResidentNativeRecipe>,
    native_storage: Option<crate::backend::runtime::residency::storage::native_storage::BankOwner>,
    parallel_control: Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlOwner>,
    // Accepted canonical page pins move into the native bank at first use.
    // Keeping them here is not permission to mutate or execute a paged source.
    paged_sources: RefCell<Option<crate::backend::nn::workspace::ProjectedPagedSources>>,
    // The accepted source program remains request-owned across prefill/decode;
    // root owners borrow only their exact finite invocation rows.
    addressable: RefCell<Option<crate::backend::submission_recovery::addressable::AddressableRequestOwner>>,
    // The actual completed B account/root metadata; contains no native payload.
    prepared_source: Option<original_prepared::PreparedMediaQuoteSource>,
    // Exact saved cache residence for post-commit publication only. No B roots,
    // media ingress or previous inference request is retained by this witness.
    saved_cache_source: Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    // A resumed composite token has no completed B; retain only its newly
    // funded constructor Context, separately from source credit/binding.
    continuation_metadata: Option<eredu_nn::workspace::WorkspaceContext>,
    // Last: the source token retires before its historical generation Q.
    original_table: Option<original_table::Owned>,
    // The cold quote and any shared plan aliases retain their own host account.
    planning_metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}

impl TextExecutionQuote {
    fn publication_source(&self) -> Option<eredu_runtime::input::OriginalPreparedWorkspaceSource> {
        self.prepared_source
            .as_ref()
            .map(original_prepared::PreparedMediaQuoteSource::publication_source)
            .or_else(|| self.saved_cache_source.clone())
    }

    fn layerwise_identity(
        &self,
    ) -> Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspaceIdentity> {
        self.layerwise.as_ref().map(LayerwiseQuoteSources::identity)
    }
    pub(super) fn install_parallel_control(&self,step:&eredu_runtime::working_memory::InferenceTextStep)
        ->Result<Option<(crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlInstallation,
    crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection)>,Error>{
        self.request
            .validate_same_request(step.request())
            .map_err(Error::PrefillControl)?;
        self.parallel_control
            .as_ref()
            .map(|owner| owner.install())
            .transpose()
    }

    pub(super) fn native_storage_bank(
        &self,
    ) -> Option<crate::backend::runtime::residency::storage::native_storage::BankOwner> {
        self.native_storage.clone()
    }

    #[cfg(test)]
    pub(super) fn test_original_table(
        &self,
    ) -> Option<&eredu_runtime::working_memory::OriginalResidentResetSource> {
        self.original_table.as_ref().map(|slot| slot.source())
    }
    pub(super) fn original_token_input(&self) -> Option<&token_input::InputQuotation> {
        self.sequence.as_ref().and_then(|s| s.input.as_ref())
    }
    pub(super) fn completion_output_ingress(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<super::completion_roots::CompletionOutputIngress, Error> {
        if self.native_storage.is_none() {
            return Ok(Default::default());
        }
        self.request
            .validate_same_request(step.request())
            .map_err(Error::PrefillControl)?;
        let controls = self
            .original_controls()
            .ok_or(Error::PrefillScopeUnavailable)?;
        controls
            .validate_reservation(
                step.request()
                    .memory_reservation()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,
            )
            .map_err(Error::PrefillControl)?;
        super::completion_roots::CompletionOutputIngress::prepare(&controls)
    }

    pub(super) fn token_validation_ingress(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        prefill: bool,
    ) -> Result<crate::backend::nn::tensor::TokenValidationIngress, Error> {
        if self.native_storage.is_none() {
            return Ok(Default::default());
        }
        self.request
            .validate_same_request(step.request())
            .map_err(Error::PrefillControl)?;
        let recipe = self
            .native_recipe
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let controls = self
            .original_controls()
            .ok_or(Error::PrefillScopeUnavailable)?;
        crate::backend::nn::tensor::TokenValidationIngress::prepare(
            recipe, step, prefill, &controls,
        )
    }

    pub(super) fn activate_operation_bank(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<Option<crate::backend::runtime::execution::generic::OriginalOperationActivation>, Error> {
        self.request.validate_same_request(step.request()).map_err(memory)?;
        self.operation_banks.try_borrow().map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref().map(|bank| bank.activate()).transpose()
    }

    pub(super) fn model_execution_preparation(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        session: &MlxModelSession,
    ) -> Result<
        Option<crate::backend::submission_recovery::prefill::ModelExecutionPreparation>,
        Error,
    > {
        if self.prediction_scopes.is_none()
            || (self.record_quota.is_none()
                && self.graph_quota.is_none()
                && self.native_storage.is_none())
        {
            return Ok(None);
        }
        self.request
            .validate_same_request(step.request())
            .map_err(Error::PrefillControl)?;
        let facts = self
            .prefill_scopes
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .facts()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let runtime = session.payload.model.erased().prefill_roots_runtime()?;
        let controls = self
            .original_controls()
            .ok_or(Error::PrefillScopeUnavailable)?;
        crate::backend::submission_recovery::prefill::ModelExecutionPreparation::new(
            step.request(),
            facts,
            runtime,
            controls,
        )
        .and_then(|prepared| prepared.with_resident_recipe(self.native_recipe.as_ref(), step))
        .and_then(|prepared| {
            let owner = self.addressable.try_borrow().map_err(|_| Error::PrefillScopeReentrant)?.clone();
            prepared.with_addressable_source(self.native_recipe.as_ref(), owner.as_ref(), step)
        })
        .map(Some)
    }
    pub(super) fn claim_prediction_scopes(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionSet>, Error> {
        self.request
            .validate_same_request(step.request())
            .map_err(Error::PrefillControl)?;
        let revision = self.sampling_revision.try_borrow().map_err(|_| Error::PredictionScopeReentrant)?;
        let sampling = if revision.is_some() { None } else {
            self.native_recipe.as_ref().map(|recipe| recipe.completion_for_sampling_step(step)).transpose()?
        };
        let paged = self
            .paged_sources
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .clone();

        self.prediction_scopes
            .as_ref()
            .map(|bank| {
                let set = bank.claim(step)?;
                let source = self.parallel_control.as_ref()
                    .map(|owner| owner.sampling_source(step)).transpose()?;
                let set = set.with_operations(self.operation_registration.clone(), step.request().clone())
                    .with_sampling(sampling)
                    .with_paged_sources(paged, step.attempt());
                match revision.as_ref() {
                    Some(revision) => set.with_sampling_replacement(revision.claim(step, source)?).map_err(memory),
                    None => Ok(set.with_sampling_source(source)),
                }
            })
            .transpose()
    }

    pub(super) fn claim_prefill_scopes(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        session: &MlxModelSession,
    ) -> Result<
        Option<(
            crate::backend::submission_recovery::prefill::PrefillBankOwner,
            crate::backend::submission_recovery::prefill::PrefillBankProjection,
        )>,
        Error,
    > {
        let Some(bank) = &self.prefill_scopes else {
            return Ok(None);
        };
        let mut original = {
            let mut bank = bank
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            bank.claim(step).map_err(Error::PrefillControl)?
        };
        // The quote/source loan ends before native controls or recovery nodes are
        // constructed. The genuine one-shot step is already consumed on failure.
        let runtime = session.payload.model.erased().prefill_roots_runtime()?;
        let graph = self
            .graph_quota
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let controls = self
            .original_controls()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let registration = self
            .operation_registration
            .clone()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let mut host_destinations = {
            let mut bank = self
                .operation_host_destinations
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            bank.take()
        };
        let paged = self
            .paged_sources
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .clone();
        let program = self.native_recipe.as_ref()
            .map(|recipe| recipe.take_addressable_source_program()).transpose()?.flatten();
        let mut prepared_paged = None;
        let has_program = program.is_some();
        if let Some(program) = program {
            let root = original.take_source_component(&mut host_destinations, program.facts())
                .map_err(Error::PrefillControl)?;
            let sources = program.accept(root)?;
            if let Some(facts) = sources.target_facts() {
                let mut target = sources.take_target()?;
                original.restore_source_component(&mut host_destinations, &mut target, facts)
                    .map_err(Error::PrefillControl)?;
            }
            prepared_paged = sources.take_paged()?;
            if !sources.occurrences().is_empty() {
                let funding = self.planning_metadata.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
                let owner = crate::backend::submission_recovery::addressable::AddressableRequestOwner::new(
                    sources, step.request(),
                    self.native_storage.as_ref().ok_or(Error::PrefillScopeUnavailable)?.clone(),
                    controls.clone(), registration.clone(), None, funding,
                )?;
                let mut slot = self.addressable.try_borrow_mut().map_err(|_| Error::PrefillScopeReentrant)?;
                if slot.is_some() { return Err(Error::PrefillScopeUnavailable); }
                *slot = Some(owner);
            }
        }
        if let Some(paged) = &paged {
            let context = self.planning_metadata.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            let sources = if has_program {
                prepared_paged.take()
            } else {
                paged.host_source_facts()
                    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
                    .map(|facts| original.take_source_component(&mut host_destinations, facts))
                    .transpose().map_err(Error::PrefillControl)?
            };
            paged.construct_host_program(sources, &controls.clone().into())
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        } else if prepared_paged.is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let operations = session
            .payload
            .model
            .erased()
            .prepare_original_operation_banks(
                &session.payload.memory_pool,
                &original,
                step,
                registration.clone(),
                controls.clone(),
                host_destinations,
                self.layerwise
                    .as_ref()
                    .and_then(LayerwiseQuoteSources::retained_workspace),
                self.native_recipe.as_ref(),
                self.planning_metadata.as_ref(),
            )?;
        // The independent quote owner covers decode and cancellation as well as
        // the first submission. Replacing/dropping a bank never occurs under a
        // policy or inspector loan.
        let old = std::mem::replace(
            &mut *self
                .operation_banks
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?,
            operations,
        );
        drop(old);
        let paged = self
            .paged_sources
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .clone();
        if let Some(paged) = &paged {
            let plan = self
                .native_recipe
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .plan();
            paged.bind_request(
                &original,
                step.request(),
                plan,
                registration.clone(),
                controls.clone(),
            )?;
        }
        let addressable = self.addressable.try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?.clone();
        crate::backend::submission_recovery::prefill::PrefillBankOwner::new_with_resident_sources(
            original,
            step.request(),
            &runtime,
            self.record_quota.as_ref(),
            graph,
            controls,
            self.native_recipe.as_ref(),
            addressable.as_ref(),
        )
        .map(|(owner, projection)| {
            Some((
                owner
                    .with_operation_registration(registration)
                    .with_native_storage(self.native_storage.clone())
                    .with_paged_sources(paged),
                projection,
            ))
        })
    }

    pub(in crate::composition::mlx::session) fn saved_sampling_input(&self)
        -> Option<eredu_runtime::working_memory::SamplingWorkspaceInputPlan> {
        self.native_recipe.as_ref()?.sampling_program().input()
    }

    pub(in crate::composition::mlx::session) fn has_capture(&self) -> bool {
        self.capture.is_some()
    }

    /// Validate the original immutable geometry and actual retained plan/path
    /// sources. Current account/origin health is checked by the installed owner.
    pub(in crate::composition::mlx::session) fn validate_capture_source(
        &self,
        session: &MlxModelSession,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<(), Error> {
        self.capture
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .validate(session, source, self.request.geometry())
    }

    pub(super) fn take_capture_installation(
        &self,
        session: &MlxModelSession,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<super::text_capture::InstalledCapture, Error> {
        self.capture
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_installed(session, source, self.request.geometry())
    }

    pub(super) fn take_capture_installation_with_error_allowance(
        &self,
        session: &MlxModelSession,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<
        (
            super::text_capture::InstalledCapture,
            super::text_error::OriginalErrorAllowance,
        ),
        Error,
    > {
        self.capture
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_installed_with_error_allowance(session, source, self.request.geometry())
    }

    /// Copying capture/result storage needs its own destination admission. The
    /// current saved component cannot preserve either original bank.
    pub(in crate::composition::mlx::session) fn validate_capture_copy(&self) -> Result<(), Error> {
        if self.has_capture() || self.sequence.is_some() {
            return Err(memory(WorkingMemoryError::UnknownBound));
        }
        Ok(())
    }

    /// Sampling-only callers cannot omit a live capture ledger.
    pub(in crate::composition::mlx::session) fn validate_paired_capture_copy(
        &self,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_paired_capture_copy_with_source(None)
    }

    /// The complete generation copy supplies the exact installed capture source.
    /// This checks association only: its checkpoint destination is separately
    /// priced and constructed under the enclosing accepted host copy.
    pub(in crate::composition::mlx::session) fn validate_paired_capture_copy_with_source(
        &self,
        source: Option<&eredu_core::capture::SharedCapturePlan>,
    ) -> Result<(), WorkingMemoryError> {
        match (&self.capture, source) {
            (None, None) => {}
            (Some(capture), Some(source)) => {
                capture.validate_checkpoint_source(source, self.request.geometry())?;
            }
            (Some(_), None) => return Err(WorkingMemoryError::UnknownBound),
            (None, Some(_)) => return Err(WorkingMemoryError::IdentityMismatch),
        }
        if let Some(sequence) = &self.sequence {
            sequence.validate_saved_host_handoff()?;
        }
        Ok(())
    }

    pub(super) fn activate_disk_route(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::manager::DiskRouteGuard>, Error> {
        self.opening.require_sealed().map_err(|error| memory(error))?;
        self.disk
            .as_ref()
            .map(|receipt| receipt.activate())
            .transpose()
    }

    pub(super) fn funding_scope(
        &self,
    ) -> Result<eredu_runtime::working_memory::WorkingMemoryFundingScope, Error> {
        self.funding
            .borrow()
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .scope()
            .map_err(|error| memory(error))
    }

    /// Prepare the observed eager seed constructor inside its current original Scope.
    pub(super) fn prepare_sampling_graph(
        &self,
    ) -> Result<Option<safemlx::PreparedResidentGraph>, Error> {
        let Some(recipe) = &self.native_recipe else {
            return Ok(None);
        };
        let Some(layout) = recipe.sampling_preparation_graph()? else {
            return Ok(None);
        };
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        safemlx::OperationEvent::prepare_resident_graph(layout, &observer)
            .map(Some)
            .map_err(Error::from)
    }

    /// The guard comes only from this quote's consuming original promotion.
    /// A retained quote keeps Q protected even after its pending bank is taken.
    pub(super) fn original_controls(
        &self,
    ) -> Option<eredu_runtime::working_memory::OriginalTextControlGuard> {
        self.capture
            .as_ref()
            .map(CaptureQuotation::control_guard)
            .or_else(|| self.sequence.as_ref().map(SequenceQuotation::control_guard))
    }

    pub(super) fn funded_work(
        &self,
        scope: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    ) -> Result<super::text_funding::FundedWorkOwner, Error> {
        let controls = self.original_controls();
        if let Some(controls) = &controls {
            controls
                .validate_reservation(self.request.memory_reservation().ok_or_else(unknown)?)
                .map_err(|error| memory(error))?;
        }
        let original_table = self
            .original_table
            .as_ref()
            .map(|source| source.model_source(&scope))
            .transpose()?;
        super::text_funding::FundedWork::new_model_with_native(
            scope,
            controls,
            self.capture
                .as_ref()
                .and_then(CaptureQuotation::opening_rows),
            self.capture.as_ref().and_then(CaptureQuotation::text_interventions),
            original_table,
            self.publication_source(),
            self.native_storage.clone(),
        )
    }

    /// The original pair withholds native funding access until successful
    /// native begin. The ordinary/saved active-scope path below is unchanged.
    pub(super) fn prepared_preparation_work(
        &self,
    ) -> Result<super::text_funding::PreparedFundedWork, Error> {
        let controls = self.original_controls().ok_or_else(unknown)?;
        controls
            .validate_reservation(self.request.memory_reservation().ok_or_else(unknown)?)
            .map_err(|error| memory(error))?;
        let scope = {
            self.funding
                .borrow()
                .as_ref()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
                .prepare_scope()
        }
        .map_err(|error| memory(error))?;
        super::text_funding::PreparedFundedWork::new_with_native(
            scope,
            controls,
            self.capture
                .as_ref()
                .and_then(CaptureQuotation::opening_rows),
            self.publication_source(),
            self.native_storage.clone(),
        )
    }

    pub(super) fn preparation_work(&self) -> Result<super::text_funding::FundedWorkOwner, Error> {
        // Preparation publishes only its own roots, never the model table.
        super::text_funding::FundedWork::new_model_with_native(
            self.funding_scope()?,
            self.original_controls(),
            self.capture
                .as_ref()
                .and_then(CaptureQuotation::opening_rows),
            None,
            None,
            self.publication_source(),
            self.native_storage.clone(),
        )
    }

    pub(super) fn sampler_scope(
        &self,
    ) -> Result<eredu_runtime::working_memory::WorkingMemorySamplerScope, Error> {
        self.funding
            .borrow()
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .sampler_scope()
            .map_err(|error| memory(error))
    }

    pub(super) fn take_funding_run(
        &self,
    ) -> Result<eredu_runtime::working_memory::WorkingMemoryFundingRun, Error> {
        self.funding
            .borrow_mut()
            .take()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))
    }
    pub(in crate::composition::mlx::session) fn config(&self) -> TextGenerationConfig {
        self.config
    }

    pub(super) fn parameter_epoch(&self) -> u64 {
        self.parameter_epoch
    }

    pub(in crate::composition::mlx::session) fn contract(&self) -> &TextControllerContract {
        &self.controller
    }

    pub(super) fn storage_contract(&self) -> &ControllerStorageContract {
        &self.storage_contract
    }

    pub(in crate::composition::mlx::session) fn request(&self) -> &InferenceRequest {
        &self.request
    }

    pub(super) fn predecessor(&self) -> Result<Option<&InferenceRequest>, Error> {
        self.opening.predecessor().map_err(|error| memory(error))
    }

    /// The cold quote belongs to this precise opening branch, even if a later
    /// restoration returns to identical request identity and position.
    pub(super) fn validate_opening(
        &self,
        retained: &eredu_runtime::working_memory::InferenceRetention,
    ) -> Result<(), Error> {
        self.opening
            .validate(retained, self.request.geometry().cached_positions)
            .map_err(|error| memory(error))
    }

    /// Private future installer endpoint; no production draft constructor is
    /// exposed by this prerequisite. The closed installer must first publish
    /// and exchange its exact admitted copied state. This re-reads that actual
    /// target after settlement and seals once, never accepts a raw revision.
    /// It does not establish copied-state provenance or perform installation.
    fn seal_installed_opening(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        self.opening.require_pending().map_err(|error| memory(error))?;
        self.validate_binding(runtime, &self.request)?;
        let session = runtime.session();
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::Other(Box::new(error)))?;
        session
            .payload
            .model
            .erased()
            .validate_text_frontier(self.request.geometry().cached_positions)?;
        let retained = session
            .payload
            .model
            .erased()
            .retained_inference_authority()?;
        self.opening.publish_installed(&retained).map_err(|error| memory(error))
    }

    /// Rechecks the actual owned vector before any preparation stage or native
    /// allocation. Equal length alone does not account for spare host capacity.
    pub(in crate::composition::mlx::session) fn validate_prompt(
        &self,
        ids: &Vec<u32>,
    ) -> Result<(), Error> {
        validate_prompt_storage(
            ids,
            self.request.geometry().input_positions,
            self.source_capacity_bytes,
        )
    }

    /// Request-local completed prediction count. Absolute sampling positions
    /// remain stable for observations and saved data; this private mapping
    /// does not mint a context, receipt or another output allowance.
    pub(in crate::composition::mlx::session) fn local_prediction(
        &self,
        absolute_prediction: u64,
    ) -> Result<u64, Error> {
        self.local_prediction_fixed(absolute_prediction)
            .map_err(|error| memory(error))
    }

    pub(in crate::composition::mlx::session) fn local_prediction_fixed(
        &self,
        absolute_prediction: u64,
    ) -> Result<u64, WorkingMemoryError> {
        self.coordinates.local(absolute_prediction)
    }

    /// Expected decoder frontier for this absolute sampling position.
    pub(in crate::composition::mlx::session) fn prediction_frontier(
        &self,
        absolute_prediction: u64,
    ) -> Result<u64, Error> {
        self.coordinates
            .frontier(absolute_prediction)
            .map_err(|error| memory(error))
    }

    /// Validates the actual decoder before the indicated absolute prediction.
    /// The initial prediction consumes the complete new prompt; later ones
    /// consume the preceding sampled token. Returns the checked logical
    /// frontier even when the selected decoder has no stateful tensor layers.
    pub(in crate::composition::mlx::session) fn validate_frontier(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        absolute_prediction: u64,
    ) -> Result<u64, Error> {
        self.validate(runtime, &self.request)?;
        let expected = self.prediction_frontier(absolute_prediction)?;
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(expected)?;
        Ok(expected)
    }

    /// Checks immutable binding without acquiring authority, reaping work or
    /// changing state. Core receipts separately enforce attempt order and quota;
    /// the native entry also checks the sampler's Rc identity and capture absence.
    pub(in crate::composition::mlx::session) fn validate(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        request: &InferenceRequest,
    ) -> Result<(), Error> {
        self.opening.require_sealed().map_err(|error| memory(error))?;
        self.validate_binding(runtime, request)
    }

    // Cold immutable binding is also needed before the future installer can
    // seal a draft. Calling this alone never authorizes ordinary text work.
    fn validate_binding(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        request: &InferenceRequest,
    ) -> Result<(), Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session.ensure_healthy()?;
        if !Rc::ptr_eq(&self.session, &session.poison)
            || !self.model_pool.same_domain(&session.payload.memory_pool)
            || !self
                .context_pool
                .same_domain(runtime.backend().memory_pool())
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        session.validate_parameter_epoch(&mut Some(self.parameter_epoch))?;
        let layerwise = session.payload.model.layerwise_workspace()?;
        if layerwise.as_ref().map(|workspace| workspace.identity()) != self.layerwise_identity() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        if let Some(receipt) = &self.disk {
            receipt.validate()?;
        }
        self.request
            .validate_same_request(request)
            .map_err(|error| memory(error))?;
        request
            .validate(
                session
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                self.request.geometry(),
            )
            .map_err(|error| memory(error))
    }
}

/// Admits resident or selected bounded token-ID requests whose resources are proved
/// below. Existing state charges remain live across quoted prefix continuation.
/// Completed original media sources use the same candidate and installation
/// workers. Prediction, communication and unknown selected primitive/host facts
/// remain explicit admission gaps.
pub(super) fn admit<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure> {
    admit_inner(runtime, input, config, controller, None, false, None, None)
        .map_err(AdmissionFailure::into_backend)
}

/// Private original-admission route; public options stay gated until the
/// corresponding installer, execution and shared drain are connected.
pub(super) fn admit_with_capture<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
    source: &eredu_core::capture::SharedCapturePlan,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure> {
    admit_inner(
        runtime,
        input,
        config,
        controller,
        Some(source),
        false,
        None,
        None,
    )
    .map_err(AdmissionFailure::into_backend)
}

/// Private joined row mechanism; public gateway selection remains unchanged.
pub(super) fn admit_with_capture_opening_rows<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
    source: &eredu_core::capture::SharedCapturePlan,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure> {
    admit_inner(runtime, input, config, controller, Some(source), true, None, None)
        .map_err(AdmissionFailure::into_backend)
}

pub(super) fn admit_with_sequence<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
    source: Option<&eredu_core::capture::SharedCapturePlan>,
    interventions: Option<&eredu_core::intervention::SharedInterventionPlan>,
    claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure> {
    #[cfg(test)]
    let opening_rows = !matches!(input, TextPreparationInput::OriginalPrepared(_))
        && sequence::fixture::opening_rows(runtime, source);
    #[cfg(not(test))]
    let opening_rows = false;
    admit_inner(
        runtime,
        input,
        config,
        controller,
        source,
        opening_rows,
        Some(claim),
        interventions,
    )
    .map_err(AdmissionFailure::into_backend)
}

pub(super) fn prepare_generation_sequence(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &MlxTextPreparation,
    claim: eredu_core::GenerationSequencePreparation<'_, '_>,
) -> Result<eredu_core::RetainedGenerationSequence, BackendFailure> {
    #[cfg(test)]
    return sequence::fixture::prepare(runtime, preparation, claim);
    #[cfg(not(test))]
    sequence::prepare(runtime, preparation, claim)
}

#[cfg(test)]
pub(super) fn intercept_claimed_work(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    operation: &super::text_step::TextOperation<'_>,
) -> Result<(), Error> {
    sequence::fixture::intercept_claimed_work(runtime, state, operation)
}
#[cfg(test)]
pub(super) fn intercept_consumed_row_installation(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
    rows: &crate::composition::mlx::replicated_text::NativeOpeningRowsOwner,
) -> Result<(), Error> {
    sequence::fixture::intercept_consumed_row_installation(runtime, preparation, source, rows)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(super) fn capture_only_sampling_for_test(
    runtime: &mut ModelRuntime<MlxBackend<'static>>,
    source: &eredu_core::capture::SharedCapturePlan,
    config: TextGenerationConfig,
) -> (MlxTextPreparation, MlxTextGenerationState) {
    sequence::fixture::capture_only_sampling(runtime, source, config)
}

#[cfg(test)]
pub(super) fn record_capture_sampling_admission(
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
) {
    sequence::fixture::record_capture_sampling(preparation, source);
}

#[cfg(test)]
pub(super) fn record_sequence_admission(preparation: &MlxTextPreparation) {
    sequence::fixture::record(preparation);
}

#[cfg(test)]
pub(super) fn intercept_sequence_sampling(
    preparation: &MlxTextPreparation,
    state: MlxTextGenerationState,
) -> Result<MlxTextGenerationState, Error> {
    sequence::fixture::intercept_sampling(preparation, state)
}

#[cfg(test)]
pub(super) fn record_sequence_installation(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
) {
    sequence::fixture::record_installation(runtime, state, preparation, source);
}

fn admit_inner<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
    capture_source: Option<&eredu_core::capture::SharedCapturePlan>,
    opening_rows: bool,
    sequence_claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
    intervention_source: Option<&eredu_core::intervention::SharedInterventionPlan>,
) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), AdmissionFailure> {
    let mut planning_metadata = None;
    let mut capture_metadata = None;
    let result: Result<_, AdmissionFailure> = (|| {
        // Token and authenticated media inputs use the same original row
        // owner. Legacy entries have no matching native source installation.
        if intervention_source.is_some()
            && (capture_source.is_none() || sequence_claim.is_none()
                || (!matches!(input, TextPreparationInput::OriginalPrepared(_))
                    && sequence_claim.and_then(|claim| claim.request().token_input()).is_none())) {
            return Err(unknown().into());
        }
        let prepared_input = if let TextPreparationInput::OriginalPrepared(prompt) = input {
            use eredu_core::PreparedRequestRejection as R;
            let refusal = original_prepared::preflight(
                runtime,
                prompt,
                config,
                capture_source,
                sequence_claim,
            );
            if refusal != R::MissingInspectionStorage {
                return Err(AdmissionFailure::Sequence(refusal.into_backend_failure()));
            }
            Some(
                original_prepared::admission_input(prompt)
                    .map_err(|cause| AdmissionFailure::Sequence(cause.into_backend_failure()))?,
            )
        } else {
            None
        };
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session.ensure_healthy()?;
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let policy = config.inference_policy();
        policy
            .validate(config.sampling().max_new_tokens)
            .map_err(capability)?;
        let capacity = policy.managed_memory_capacity_bytes.ok_or_else(unknown)?;
        let max_output_tokens = config
            .sampling()
            .max_new_tokens
            .and_then(|count| u64::try_from(count).ok())
            .ok_or_else(unknown)?;
        let (positions, capacity_bytes) = match input {
            TextPreparationInput::TokenIds {
                positions,
                capacity_bytes,
            } => (*positions, *capacity_bytes),
            TextPreparationInput::OriginalTokenIds(plan) => {
                let accepted = sequence_claim
                    .and_then(|claim| claim.request().token_input())
                    .ok_or_else(unknown)?;
                if !std::ptr::eq(accepted, *plan) {
                    return Err(memory(WorkingMemoryError::IdentityMismatch).into());
                }
                i32::try_from(plan.tokens().len())
                    .map_err(|_| memory(WorkingMemoryError::Overflow))?;
                (plan.tokens().len() as u64, plan.destination_bytes())
            }
            TextPreparationInput::Prepared(_) => return Err(unknown().into()),
            TextPreparationInput::OriginalPrepared(_) => (
                prepared_input
                    .expect("authenticated completed input")
                    .model_positions,
                0, // Completed B has no new token input destination.
            ),
        };
        // Original token and completed-media sources were authenticated above.
        // Ordinary Vec inputs retain component-only None semantics; both original
        // sources require full native populations without an auxiliary ceiling.
        let original_native = matches!(
            input,
            TextPreparationInput::OriginalTokenIds(_) | TextPreparationInput::OriginalPrepared(_)
        );
        if original_native
            || policy.submission_tracking_capacity_bytes.is_some()
            || policy.graph_metadata_capacity_bytes.is_some()
        {
            runtime.backend().validate_original_stream_owners()?;
        }
        let opening = session
            .payload
            .model
            .erased()
            .retained_inference_authority()?;
        let predecessor = opening
            .admission()
            .map(|admission| admission.request().clone());
        let cached_positions = opening
            .admission()
            .map_or(0, |admission| admission.position());
        let mut geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions,
            input_positions: positions,
            max_output_tokens,
            prefill_chunk_positions: policy
                .prefill_chunk_positions
                .map(|chunk| chunk.get().min(positions))
                .unwrap_or(positions),
            output: OutputDemand::LastPosition,
        };
        if let TextPreparationInput::OriginalPrepared(prompt) = input {
            if geometry.cached_positions != 0 {
                return Err(memory(WorkingMemoryError::IdentityMismatch).into());
            }
            if let Some(chunk) = prompt.prefill_chunk_positions {
                geometry.prefill_chunk_positions =
                    geometry.prefill_chunk_positions.min(chunk.get());
            }
        }
        geometry.validate().map_err(capability)?;

        let model = &session.payload.model;
        if let Some(predecessor) = &predecessor {
            predecessor
                .validate(
                    model.erased().inference_execution_identity(),
                    predecessor.geometry(),
                )
                .map_err(|error| memory(error))?;
            predecessor
                .memory_reservation()
                .ok_or_else(unknown)?
                .validate_domain(&session.payload.memory_pool)
                .map_err(|error| memory(error))?;
        }
        let blueprint = model.inference_blueprint().ok_or_else(unknown)?;
        let selected = blueprint.selected();
        // Distributed original entry proceeds only with the same complete
        // retained session/world/selection tuple. The candidate below must
        // still compile its actual direct parallel source and total recipe.
        match (
            &session.payload.distributed,
            session.payload.target.has_retained_world(),
            selected.communication_manifest(),
        ) {
            (None, false, None) | (Some(_), true, Some(_)) => {}
            _ => return Err(unknown().into()),
        }
        if !session
            .payload
            .memory_pool
            .same_domain(runtime.backend().memory_pool())
            || !model.has_published_idle_storage()
            || !model.has_workspace_mechanisms()
            || session.payload._memory_owner.is_some()
            || session.payload.parameter_state.active.is_some()
            || !matches!(
                selected.text_realization().residency(),
                eredu_runtime::LayerWeightResidency::FullyResident
                    | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            )
            || !crate::backend::runtime::cache::kv::PagedKeyValueCache::original_generation_residency_supported(
                selected.text_realization().state().policy(),
                model
                    .erased()
                    .capability_estimate()
                    .state_layout()
                    .layer_layout(),
                session.floating_state_dtype_bytes,
            )
        {
            return Err(unknown().into());
        }
        if !selected.text_realization().exact_completion_available() {
            return Err(memory(WorkingMemoryError::CompletionUnavailable).into());
        }
        // Plain generation does not invoke a retained prediction extension.
        // Its modules, prototypes, sources and residency owners still join this
        // complete idle inventory; target quotation authenticates the retained
        // selection without rebuilding the extension.
        // This includes prepared sources, operator helpers, overlays, processors and
        // erased observers. Unknown owners stay unknown. Existing physical aliases
        // were registered together at initial publication; never charge their byte
        // total again as incremental request storage.
        let idle = session.payload.retained_idle_storage()?;
        let original_table = original_table::Plan::prepare(session)?;
        idle.validate_original_table(
            &session.payload.memory_pool,
            original_table.as_ref().map(original_table::Plan::source),
        )?;
        if idle.nonstate_bytes()?.is_none()
            || idle.decoder_state_bytes()?.is_none()
            || (predecessor.is_none() && !idle.has_empty_decoder_storage()?)
        {
            return Err(unknown().into());
        }
        model.erased().validate_text_frontier(cached_positions)?;
        if original_native && capture_source.is_some() {
            // A precompiled declaration does not run the raw declaration hook.
            // Qualify the same retained source here before either input form
            // revalidates it or a cold candidate borrows its partition layouts.
            let funding = session.payload.memory_pool.prepare_workspace_metadata(
                model.erased().inference_execution_identity(), capacity,
            ).map_err(Error::WorkspacePlanning)?;
            capture_metadata = Some(funding);
            session.original_partition_capture(capture_metadata.as_ref().expect("created account"))?;
        }
        let capture_destination = capture_metadata.as_ref().map_or(
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
            eredu_runtime::working_memory::WorkspaceReportMetadata::with_funding,
        );
        let capture = capture_source
            .map(|source| {
                let prepared = if matches!(input, TextPreparationInput::OriginalPrepared(_)) {
                    CaptureAdmission::new_media(session, geometry, source, capture_destination)
                } else {
                    CaptureAdmission::new(session, geometry, source, capture_destination)
                };
                prepared.and_then(|capture| {
                    let capture = if let Some(interventions) = intervention_source {
                        capture.with_intervention_source(interventions, config.inference_policy()
                            .managed_memory_capacity_bytes.ok_or_else(unknown)?)?
                    } else {
                        capture
                    };
                    Ok(if opening_rows { capture.with_opening_rows() } else { capture })
                })
            })
            .transpose()?;
        if let Some(capture) = &capture {
            geometry.output = capture.physical_output(geometry.output);
        }
        // Physical readout is fixed before candidate quotation and admission.
        let coordinates = PredictionCoordinates::new(0, geometry).map_err(|error| memory(error))?;
        let mut epoch = None;
        session.validate_parameter_epoch(&mut epoch)?;
        let parameter_epoch = epoch.ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
        let layerwise_workspace = model.layerwise_workspace()?;
        let workspace = controller
            .inference_workspace(max_output_tokens)
            .ok_or_else(unknown)?;
        // A completed original input has no ordinary unpriced controller-source
        // adoption path. Original C and a declared empty inventory use the same
        // existing validation below; neither creates map nodes during inspection.
        if prepared_input.is_some()
            && controller
                .inference_storage()
                .original_token_domain()
                .is_none()
            && !matches!(controller.inference_storage(), eredu_core::TextControllerStorage::RunOwnedWithPreparedSemantic { .. })
            && !controller
                .inference_storage()
                .shared_sources()
                .is_some_and(|mut sources| sources.next().is_none())
        {
            return Err(AdmissionFailure::Sequence(
                eredu_core::PreparedRequestRejection::MissingController.into_backend_failure(),
            ));
        }
        let storage_contract = if controller
            .inference_storage()
            .original_token_domain()
            .is_some()
            || matches!(controller.inference_storage(), eredu_core::TextControllerStorage::RunOwnedWithPreparedSemantic { .. })
        {
            let claim = sequence_claim.ok_or_else(|| {
                AdmissionFailure::Sequence(
                    eredu_core::GenerationSequenceBankRejection::IdentityMismatch
                        .into_backend_failure(),
                )
            })?;
            ControllerStorageContract::inspect_original_sequence(
                controller,
                workspace,
                runtime.backend().memory_pool(),
                model.erased().inference_execution_identity(),
                claim,
            )
            .map_err(AdmissionFailure::Sequence)?
        } else {
            ControllerStorageContract::inspect(controller)
                .map_err(|error| Error::Other(Box::new(error)))?
        };
        if !storage_contract.has_original_domain() {
            // Original inspection already checked this exact immutable mask borrow
            // through its fixed pre-claim rejection path above.
            storage_contract
                .validate_workspace(workspace)
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        let capabilities = session.payload.model.erased().capability_estimate();
        let admission_request = AdmissionRequest {
            input: prepared_input.unwrap_or_else(|| {
                InputTokenCount::text(geometry.cached_positions + geometry.input_positions)
            }),
            max_output_tokens,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        if let Some(rejection) =
            eredu_core::check_admission_context(capabilities.capabilities(), admission_request)
                .map_err(capability)?
        {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::PrefillPlanningError::Admission(rejection),
            ))
            .into());
        }
        // Existing-only pins grant credit for exact immutable source identities.
        // Custom sources not yet registered retain the full quote and are adopted
        // after reservation. No failed pin registers missing storage.
        let registered_controller = if storage_contract.has_original_domain() {
            // Original C remains in its own account. No registration or credit path.
            None
        } else {
            match storage_contract.pin_registered(controller, runtime.backend().memory_pool()) {
                Ok(registered) => Some(registered),
                Err(ControllerStorageError::Storage(WorkingMemoryError::IdentityMismatch)) => None,
                Err(error) => return Err(Error::Other(Box::new(error)).into()),
            }
        };
        // Only this session's unique funding owners delegate succession. Tokens
        // retain no payload or account lifetime; completed accounts are pruned by
        // metadata alone, and the runtime atomically rechecks every live account.
        {
            let mut handoffs = session.capacity_handoffs.borrow_mut();
            let mut index = 0;
            while index < handoffs.len() {
                if handoffs[index].is_retired().map_err(|error| memory(error))? {
                    handoffs.swap_remove(index);
                } else {
                    index += 1;
                }
            }
        }
        let handoffs = session.capacity_handoffs.borrow();
        // Unique compiled source leaves the actual synchronous input ONCE. Every
        // candidate subsequently borrows its unchanged original descriptor only.
        let mut decoder_source = sequence_claim
        .map(|claim| {
            eredu_runtime::working_memory::OriginalGenerationDecoderSource::take_original(
                claim,
                runtime.backend().memory_pool(),
            )
        })
        .transpose()
        .map_err(AdmissionFailure::Sequence)?
        .flatten();
        #[cfg(test)]
        sequence::fixture::decoder_taken(runtime, capture_source, &decoder_source);
        let (
            reservation,
            controller_contract,
            accepted,
            native_recipe,
            prepared_source,
            paged_sources,
            candidate_planning,
        ) = plan_candidates_with_handoff_retained(
            model.erased().inference_execution_identity(),
            runtime.backend().memory_pool(),
            capabilities.capabilities(),
            admission_request,
            geometry,
            capacity,
            workspace,
            &handoffs,
            |candidate| {
                if let TextPreparationInput::OriginalPrepared(prompt) = input {
                    return original_prepared::quote_candidate(
                        runtime,
                        prompt,
                        candidate,
                        config,
                        workspace,
                        &storage_contract,
                        sequence_claim.expect("authenticated original prepared sequence"),
                        capacity,
                        original_table.is_some(),
                        layerwise_workspace.as_ref(),
                        capture.as_ref(),
                    );
                }
                quote_incremental_with_sequence(
                    session,
                    candidate,
                    capacity_bytes,
                    config,
                    workspace,
                    &storage_contract,
                    registered_controller.as_ref(),
                    predecessor.is_some(),
                    capture.as_ref(),
                    sequence_claim,
                    original_table.is_some(),
                    layerwise_workspace.as_ref(),
                )
            },
        )?;
        planning_metadata = candidate_planning;
        // Rebind after acceptance so this paid snapshot retires before the actual
        // reservation on every subsequent failure, including pin preparation.
        let layerwise_workspace = layerwise_workspace;
        // Immediate, infallible installation after acceptance: on errors/unwind the
        // source must retire before this actual reservation. No guard is minted.
        let decoder_staging = sequence::DecoderStaging::new(&mut decoder_source);
        drop(handoffs);
        #[cfg(test)]
        sequence::fixture::decoder_checkpoint(runtime, false)
            .map_err(AdmissionFailure::Sequence)?;
        // Both aggregate-control routes consume this exact accepted proof. The
        // ordinary no-control route keeps its existing diagnostic retirement.
        let accepted = if capture.is_some()
            || sequence_claim.is_some()
            || policy.submission_tracking_capacity_bytes.is_some()
            || policy.graph_metadata_capacity_bytes.is_some()
            || original_table.is_some()
        {
            Some(accepted)
        } else {
            drop(accepted);
            None
        };
        // The reservation intrinsically retains every excluded host/decoder charge.
        // Diagnostic candidates and temporary host pins must not extend their lives.
        drop(registered_controller);
        installation::AcceptedInstallation {
            config,
            coordinates,
            controller_contract,
            storage_contract,
            opening,
            predecessor,
            parameter_epoch,
            capacity_bytes,
            original_native,
            has_sequence: sequence_claim.is_some(),
            capture,
            layerwise_workspace,
            original_table,
            accepted,
            native_recipe,
            prepared_source,
            paged_sources,
            reservation,
            planning_metadata: planning_metadata.clone(),
        }
        .install(runtime, controller, decoder_staging)
    })();
    result.map_err(|cause| match (cause, planning_metadata.or(capture_metadata)) {
        (AdmissionFailure::Native(cause), Some(funding)) => AdmissionFailure::Native(
            crate::composition::mlx::model::retain_planning_error(cause, funding),
        ),
        (AdmissionFailure::Sequence(cause), Some(funding)) => AdmissionFailure::Native(
            crate::composition::mlx::model::retain_planning_failure(cause, funding),
        ),
        (cause, None) => cause,
    })
}

#[cfg(test)]
pub(super) fn quote(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    config: TextGenerationConfig,
    controller: TextControllerWorkspace<'_>,
) -> Result<(RuntimeStateEstimate, usize), Error> {
    let generation = session
        .payload
        .model
        .quote_replicated_resident_text_with_sampling(geometry, config, controller.filter)?;
    let (state, prompt, outside) = enclosing_components(session, geometry, source_capacity)?;
    let outside = generation
        .enclosing_preparation_workspace(&prompt, controller, outside)
        .map_err(capability)?;
    let state = generation
        .equations
        .refine_state_backing(state)
        .map_err(capability)?;
    generation
        .equations
        .compose(state, outside)
        .map(|estimate| (estimate, generation.sampling.output_width))
        .map_err(capability)
}

fn quote_incremental(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    config: TextGenerationConfig,
    controller: TextControllerWorkspace<'_>,
    storage_contract: &ControllerStorageContract,
    registered_controller: Option<&RegisteredControllerStorage>,
    registered_decoder: bool,
    capture: Option<&CaptureAdmission<'_>>,
) -> Result<TextWorkspaceCandidate, Error> {
    quote_incremental_with_sequence(
        session,
        geometry,
        source_capacity,
        config,
        controller,
        storage_contract,
        registered_controller,
        registered_decoder,
        capture,
        None,
        false,
        None,
    )
}

fn quote_incremental_with_sequence(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    config: TextGenerationConfig,
    controller: TextControllerWorkspace<'_>,
    storage_contract: &ControllerStorageContract,
    registered_controller: Option<&RegisteredControllerStorage>,
    registered_decoder: bool,
    capture: Option<&CaptureAdmission<'_>>,
    sequence_claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
    original_table: bool,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
) -> Result<TextWorkspaceCandidate, Error> {
    let mut planning_metadata = None;
    let mut planning_context = None;
    let mut paged_sources = None;
    let result: Result<_, Error> = (|| {
        // Only the genuine original input claim requests whole-native automatic
        // planning. Merely supplying a sequence/decoder does not change legacy mode.
        let original_native = sequence_claim
            .and_then(|claim| claim.request().token_input())
            .is_some();
        let (generation, decoder, mut native_recipe) =
            if original_native && session.payload.distributed.is_some() {
                // The same candidate/recipe worker consumes the exact initial
                // partition and native setup. Complete world/control admission is
                // still required by the outer entry; this is no alternate driver.
                let selected = session
                    .payload
                    .model
                    .inference_blueprint()
                    .ok_or_else(unknown)?
                    .selected();
                let manifest = selected.communication_manifest().ok_or_else(unknown)?;
                let capture_layouts = if capture.is_some() {
                    Some(session.partition_capture_source().ok_or_else(unknown)?)
                } else { None };
                let funded = session
                    .payload
                    .model
                    .quote_registered_direct_parallel_text_with_sampling_recipe_funded(
                        geometry,
                        config,
                        controller.filter,
                        &session.payload.memory_pool,
                        config
                            .inference_policy()
                            .managed_memory_capacity_bytes
                            .ok_or_else(unknown)?,
                        capture.map(CaptureAdmission::prepared_selection),
                        capture.and_then(CaptureAdmission::intervention_quote),
                        capture_layouts.as_ref().map(|loaded| (loaded.layouts(), manifest.rank())),
                        retained_sources,
                        |funding| session.original_workspace_parallel_source(funding)?.ok_or_else(unknown),
                    )?;
                let (generation, storage, recipe, sources, context, funding) = funded.into_parts();
                paged_sources = sources;
                planning_metadata = Some(funding);
                planning_context = Some(context);
                (generation, Some(storage), Some(recipe))
            } else if let Some(capture) = capture.filter(|_| original_native) {
                let funded = session
                    .payload
                    .model
                    .quote_registered_resident_text_with_sampling_capture_recipe_funded(
                        geometry,
                        config,
                        controller.filter,
                        &session.payload.memory_pool,
                        config
                            .inference_policy()
                            .managed_memory_capacity_bytes
                            .ok_or_else(unknown)?,
                        capture.prepared_selection(),
                        capture.intervention_quote(),
                        retained_sources,
                    )?;
                let (generation, storage, recipe, sources, context, funding) = funded.into_parts();
                paged_sources = sources;
                planning_metadata = Some(funding);
                planning_context = Some(context);
                (generation, Some(storage), Some(recipe))
            } else if let Some(capture) = capture {
                if registered_decoder {
                    let (generation, storage, host) = session
                        .payload
                        .model
                        .quote_registered_resident_text_with_sampling_and_prefill_capture(
                            geometry,
                            config,
                            controller.filter,
                            &session.payload.memory_pool,
                            capture.bind_geometry(geometry)?,
                        )?;
                    capture.validate_host(&host)?;
                    (generation, Some(storage), None)
                } else {
                    let (generation, host) = session
                        .payload
                        .model
                        .quote_replicated_resident_text_with_sampling_and_prefill_capture(
                            geometry,
                            config,
                            controller.filter,
                            capture.bind_geometry(geometry)?,
                        )?;
                    capture.validate_host(&host)?;
                    (generation, None, None)
                }
            } else if original_native
                || (config
                    .inference_policy()
                    .submission_tracking_capacity_bytes
                    .is_some()
                    && config
                        .inference_policy()
                        .graph_metadata_capacity_bytes
                        .is_some())
            {
                // Original recipes always bind the actual opening decoder roots. A
                // first request has an empty root set; predecessor presence is not a
                // storage certificate. Nonempty roots still require canonical rows.
                let funded = session
                    .payload
                    .model
                    .quote_registered_resident_text_with_sampling_recipe_funded(
                        geometry,
                        config,
                        controller.filter,
                        &session.payload.memory_pool,
                        config
                            .inference_policy()
                            .managed_memory_capacity_bytes
                            .ok_or_else(unknown)?,
                        retained_sources,
                    )?;
                let (generation, storage, recipe, sources, context, funding) = funded.into_parts();
                paged_sources = sources;
                planning_metadata = Some(funding);
                planning_context = Some(context);
                (generation, Some(storage), Some(recipe))
            } else if registered_decoder {
                let (generation, storage) = session
                    .payload
                    .model
                    .quote_registered_resident_text_with_sampling(
                        geometry,
                        config,
                        controller.filter,
                        &session.payload.memory_pool,
                    )?;
                (generation, Some(storage), None)
            } else {
                (
                    session
                        .payload
                        .model
                        .quote_replicated_resident_text_with_sampling(
                            geometry,
                            config,
                            controller.filter,
                        )?,
                    None,
                    None,
                )
            };
        let quote = observed::quote_observed_with_sequence(
            session,
            geometry,
            source_capacity,
            config,
            controller,
            storage_contract,
            registered_controller,
            observed::ObservedWorkspace {
                equations: &generation.equations,
                sampling: &generation.sampling,
                storage: decoder.as_ref().map_or(
                    observed::ObservedStorage::Unregistered,
                    observed::ObservedStorage::Registered,
                ),
                state_input: InputTokenCount::text(
                    geometry.cached_positions + geometry.input_positions,
                ),
            },
            &mut native_recipe,
            capture,
            sequence_claim,
            original_table,
            original_native,
            retained_sources,
            planning_context.as_ref(),
            paged_sources.as_ref(),
        )?;
        if let Some(context) = &planning_context {
            context
                .charge_metadata(std::mem::size_of::<(
                    TextWorkspaceCandidate,
                    Result<TextWorkspaceCandidate, Error>,
                )>())
                .map_err(|cause| Error::Neural(cause.into()))?;
        }
        Ok(TextWorkspaceCandidate {
            quote,
            output_width: generation.sampling.output_width,
            native_recipe,
            prepared_source: None,
            paged_sources: paged_sources.take(),
            planning_metadata: planning_metadata.clone(),
        })
    })();
    // The context's trace and provider shells retire before the independently
    // retained result/error owner. No native execution authority is created.
    drop(planning_context);
    result.map_err(|cause| match cause {
        // These scalar refusals retain no planning payload. Keep the shared
        // planner's retry classification visible after the candidate retires.
        error @ Error::PrefillControl(
            WorkingMemoryError::SubmissionTrackingCapacity { .. }
            | WorkingMemoryError::GraphMetadataCapacity { .. },
        ) => error,
        cause => match planning_metadata {
            Some(funding) => crate::composition::mlx::model::retain_planning_error(cause, funding),
            None => cause,
        },
    })
}

fn enclosing_components(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
) -> Result<
    (
        RuntimeStateEstimate,
        eredu_runtime::working_memory::TextPromptWorkspaceReport,
        ExecutionWorkspaceEstimate,
    ),
    Error,
> {
    enclosing_components_for_controls(session, geometry, source_capacity, true)
}

fn enclosing_components_for_controls(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    legacy_work_controls: bool,
) -> Result<
    (
        RuntimeStateEstimate,
        eredu_runtime::working_memory::TextPromptWorkspaceReport,
        ExecutionWorkspaceEstimate,
    ),
    Error,
> {
    enclosing_components_with_input(
        session,
        geometry,
        source_capacity,
        legacy_work_controls,
        None,
        None,
        false,
        None,
    )
}

fn enclosing_components_with_input(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    legacy_work_controls: bool,
    original_input: Option<&eredu_runtime::working_memory::OriginalTokenInputLayout>,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    retain_source_workspace: bool,
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<
    (
        RuntimeStateEstimate,
        eredu_runtime::working_memory::TextPromptWorkspaceReport,
        ExecutionWorkspaceEstimate,
    ),
    Error,
> {
    let (state, prompt, outside) = enclosing_observed_components(
        session,
        geometry,
        source_capacity,
        legacy_work_controls,
        original_input,
        retained_sources,
        retain_source_workspace,
        context,
        None,
        InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
    )?;
    Ok((
        state,
        prompt.expect("text input preparation was requested"),
        outside,
    ))
}

fn enclosing_observed_components(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    legacy_work_controls: bool,
    original_input: Option<&eredu_runtime::working_memory::OriginalTokenInputLayout>,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    retain_source_workspace: bool,
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
    completed_source: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    state_input: InputTokenCount,
) -> Result<
    (
        RuntimeStateEstimate,
        Option<eredu_runtime::working_memory::TextPromptWorkspaceReport>,
        ExecutionWorkspaceEstimate,
    ),
    Error,
> {
    let model = &session.payload.model;
    let metadata = context.map_or_else(
        eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary,
        eredu_runtime::working_memory::WorkspaceReportMetadata::new,
    );
    let report_error = |cause| Error::Neural(metadata.error(cause));
    let bounded = |bytes, text: &str| {
        metadata
            .bounded(bytes, format_args!("{text}"))
            .map_err(report_error)
    };

    // The selected quote already owns this snapshot. Diagnostic-only callers
    // retain their existing local construction and never clone its identity.
    let temporary_sources = if retained_sources.is_none() {
        model.layerwise_workspace()?
    } else {
        None
    };
    let layerwise = retained_sources.or(temporary_sources.as_ref());
    let retained_source_controls = layerwise
        .map(|workspace| {
            workspace
                .known_retained_control_bytes(retain_source_workspace)
                .ok_or_else(unknown)
        })
        .transpose()?
        .unwrap_or(0);
    let prompt = if let Some(source) = completed_source {
        if original_input.is_some()
            || source.borrowed_storage().is_none()
            || !source.pool().same_domain(&session.payload.memory_pool)
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        // The closed original source already owns leaves, canonical parts and
        // cache identity. This path requests no second input construction.
        None
    } else {
        Some(match (original_input, context) {
            (Some(input), Some(context)) => {
                eredu_runtime::working_memory::quote_original_token_prompt_workspace(
                    geometry, input, context,
                )
                .map_err(Error::Neural)?
            }
            (None, Some(context)) => eredu_runtime::working_memory::quote_text_prompt_workspace(
                geometry,
                Some(source_capacity),
                context,
            )
            .map_err(Error::Neural)?,
            (Some(input), None) => model.quote_original_token_prompt_workspace(geometry, input)?,
            (None, None) => model.quote_text_prompt_workspace(geometry, Some(source_capacity))?,
        })
    };
    let state = eredu_core::estimate_runtime_state_facts(
        model.erased().capability_estimate().state_layout(),
        state_input,
        geometry.max_output_tokens,
        geometry.batch_size,
        session.floating_state_dtype_bytes,
    )
    .map_err(|cause| report_error(cause.into()))?;
    let state = metadata.state_from_facts(state).map_err(report_error)?;
    // Each zero names an absent or already-composed owner in the unobserved
    // baseline. The closed capture extension adds cumulative H and new source C
    // below the same original enclosing proof; its native transforms belong to
    // the observed equation spans. Speculation, state copy and arbitrary external
    // observers still require their own complete composition.
    metadata
        .admit::<ExecutionWorkspaceEstimate>()
        .map_err(report_error)?;
    let outside = ExecutionWorkspaceEstimate {
        geometry,
        activations: bounded(
            0,
            "all layer activations belong to the prepared equation trace; full prompt storage is composed separately",
        )?,
        attention: bounded(
            0,
            "selected attention scratch is already included in the equation trace",
        )?,
        vocabulary: bounded(
            0,
            "equations price score storage; cumulative configured sampling is composed separately",
        )?,
        state_update: bounded(
            0,
            "selected state growth, transaction checkpoints and rollback overlap belong to the equation trace",
        )?,
        materialization: match layerwise {
            Some(workspace) => metadata
                .clone_bound(workspace.materialization())
                .map_err(report_error)?,
            None => bounded(
                0,
                "already published fully resident model/source storage; no weight materialization, transfer or collective in this selected ordinary path",
            )?,
        },
        retained: bounded(
            if legacy_work_controls {
                super::text_funding::text_work_control_bytes(geometry.max_output_tokens)?
                    .checked_add(owner_control_bytes()?)
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?
            } else {
                0
            }
            .checked_add(retained_source_controls)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            "known retained layerwise snapshot/identity/layout payload and one initial original host pin (direct metadata and cold scratch remain separate); concrete original quote Rc and finite Prompt, Sampling and inference work Rc controls including consuming retirement; controller payload and capture bank/active controls remain separate; legacy diagnostics alone do not establish escaped-alias host custody",
        )?,
    };
    Ok((state, prompt, outside))
}

fn capability(error: CapabilityError) -> Error {
    Error::Other(Box::new(error))
}

fn validate_prompt_storage(
    ids: &Vec<u32>,
    positions: u64,
    source_capacity_bytes: u64,
) -> Result<(), Error> {
    let overflow = || {
        capability(CapabilityError::ArithmeticOverflow {
            operation: "quoted text prompt capacity",
        })
    };
    let actual_positions = u64::try_from(ids.len()).map_err(|_| overflow())?;
    let actual_capacity = u64::try_from(ids.capacity())
        .ok()
        .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(overflow)?;
    if actual_positions != positions || actual_capacity > source_capacity_bytes {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    Ok(())
}

#[cfg(test)]
mod prompt_storage_tests {
    use super::validate_prompt_storage;

    #[test]
    fn same_token_count_does_not_hide_larger_owned_host_capacity() {
        let ids = vec![3_u32, 5, 7];
        let capacity = ids.capacity() as u64 * 4;
        validate_prompt_storage(&ids, 3, capacity).unwrap();
        validate_prompt_storage(&ids, 3, capacity + 64).unwrap();
        assert!(validate_prompt_storage(&ids, 2, capacity).is_err());

        let mut oversized = Vec::with_capacity(ids.capacity() + 1);
        oversized.extend_from_slice(&ids);
        assert!(validate_prompt_storage(&oversized, 3, capacity).is_err());
    }
}
#[track_caller]
fn memory(error: WorkingMemoryError) -> Error {
    Error::text_admission(error)
}
#[track_caller]
fn unknown() -> Error {
    memory(WorkingMemoryError::UnknownBound)
}

#[cfg(test)]
pub(super) fn record_prediction_context_for_test(
    preparation: &MlxTextPreparation,
    context: &eredu_core::TextStepContext,
) {
    sequence::fixture::record_prediction_context(preparation, context);
}

// Exact session-bound existing probe; no configuration, fact or claim is changed.
#[cfg(test)]
pub(super) fn record_tracking_admission(preparation: &MlxTextPreparation) -> bool {
    if preparation
        .quote
        .as_ref()
        .is_some_and(|quote| quote.record_quota.is_some() || quote.graph_quota.is_some())
    {
        sequence::fixture::record(preparation);
        true
    } else {
        false
    }
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
#[path = "text_quote/prefill_tests.rs"]
mod prefill_tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
