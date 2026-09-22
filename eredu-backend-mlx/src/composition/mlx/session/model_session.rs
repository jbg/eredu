pub(crate) mod control_slot;
mod intervention_source;
pub(crate) use intervention_source::OriginalInterventionDeclaration;

use super::*;
use eredu_core::TextPreparationInput;
mod original_host_input;
pub(crate) use original_host_input::CompletedOriginalModelInput;
pub use original_host_input::{
    MlxHostInputUploadError, MlxOriginalPreparedModelInput, MlxOriginalPreparedNativeInput,
    MlxPreparedInputMaterializer, MlxPreparedModelInputBindError, MlxPreparedModelInputError,
    MlxPreparedModelInputPlan, MlxPreparedNativeInputError, MlxPreparedNativeInputPlan,
};

mod completion_roots;
pub(super) use completion_roots::{CompletionRootsOwner, ObservationRoots};
mod cache_control;
mod idle_storage;
mod original_operation;
mod parallel_workspace;
pub(in crate::composition::mlx) mod partition_capture;
mod payload_owner;
mod prediction_startup;
mod resident_reset;
pub(super) mod saved_array_copy;
mod speculative_capture;
mod submission_owner;
pub(super) mod text_capture;
mod text_error;
mod text_execution;
pub(super) mod text_funding;
pub(in crate::composition::mlx) mod text_quote;
mod text_step;
pub(super) use payload_owner::SessionPayloadOwner;
pub(in crate::composition::mlx) use speculative_capture::{
    OriginalModelPartitionPreparation, OriginalModelPartitionSource, SpeculativePartitionBinding,
};
pub(super) use submission_owner::SubmissionResourcesOwner;
pub use text_step::MlxTextStepPermit;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod capacity_handoff_tests;
#[cfg(test)]
mod controller_credit_tests;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod disk_layerwise_tests;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod funded_sampler_copy_tests;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod host_layerwise_tests;
#[cfg(test)]
mod host_preparation_tests;
#[cfg(test)]
mod initial_publication_tests;
#[cfg(test)]
mod layout_publication_tests;
#[cfg(test)]
mod loaded_helper_tests;
#[cfg(test)]
mod managed_admission_tests;
#[cfg(test)]
mod observation_paths_tests;
#[cfg(test)]
mod operation_memory_tests;
#[cfg(test)]
mod optional_filter_tests;
#[cfg(test)]
mod output_retention_tests;
#[cfg(test)]
mod partition_capture_tests;
#[cfg(test)]
mod preparation_authority_tests;
#[cfg(test)]
mod preparation_memory_tests;
#[cfg(test)]
mod shared_controller_tests;
#[cfg(test)]
mod text_quote_tests;
#[cfg(test)]
mod text_reuse_tests;
#[cfg(test)]
mod text_step_tests;
use eredu_core::{BackendFailure, BackendFailureKind};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use super::recovery::{Probe, Recovery, Retention, Status};
use crate::backend::managed_memory::{NativeMemoryOwner, NativeMemoryRetention};
use crate::backend::ordinary_retirement;

#[path = "parameters.rs"]
mod parameters;

pub(super) struct SessionPayload {
    model: Executable,
    parameter_state: parameters::NativeParameterState,
    target: crate::backend::MlxPreparedTarget,
    distributed: Option<MlxDistributedSession>,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
    #[cfg(test)]
    _retirement_probe: Option<Box<dyn std::any::Any>>,
    // Payload retirement releases every native owner before domain authority.
    _memory_owner: Option<crate::backend::managed_memory::NativeMemoryOwner>,
    // Independent snapshot authority follows the state installed by exchange.
    state_memory: NativeMemoryRetention,
    // The selected model's domain remains identifiable independently of its
    // loading authority. Operations may also be invoked through another native
    // context, whose allocation authority must be acquired before entry.
    memory_ledger: eredu_runtime::working_memory::MemoryLedger,
    // Mutating operations can leave caches, parameter replacements or other
    // payloads installed after their submission completes. Retain their domain
    // authority with the payload, including on failure.
    operation_memory: RefCell<NativeMemoryRetention>,
    // Keep source-owned nonstate publications with their model. Escaped token
    // owners retain only their decoder/output publications and physical charges.
    nonstate_publication:
        RefCell<Option<crate::backend::runtime::residency::storage::RetainedStoragePublication>>,
}

/// One submission owns both the entire executable and its neutral authority.
/// Scope tickets keep this owner alive even if the public session is dropped.
enum SubmissionPurpose {
    Model,
    OriginalModel,
    // Sources and intermediate descriptors live through the same recovery
    // tickets as ordinary submissions. Only final destinations are published.
    SavedArrayCopy(Rc<RefCell<Vec<Array>>>),
    // Same completion engine, with independent actual decoder source custody.
    SavedComponentsCopy(saved_array_copy::decoder::CopyRecovery),
}

impl SubmissionPurpose {
    fn is_saved_copy(&self) -> bool {
        matches!(self, Self::SavedArrayCopy(_) | Self::SavedComponentsCopy(_))
    }
}

#[cfg(test)]
#[path = "model_session/saved_copy_completion_tests.rs"]
mod saved_copy_completion_tests;

pub(super) struct SubmissionResources {
    // The first real text submission owns all prefill controls. Installed views
    // are weak and cannot keep these nodes alive after independent settlement.
    prefill_scopes: RefCell<Option<crate::backend::submission_recovery::prefill::PrefillBankOwner>>,
    // Cached decode borrows this exact configured ModelExecution scope. Its
    // prepared root owner survives all early errors in the outer Recovery.
    model_execution:
        RefCell<Option<crate::backend::submission_recovery::prefill::ModelExecutionOwner>>,
    // Lexical control installation closes at shared callback exit, independently
    // of the request cursor and any pending native recovery source retention.
    parallel_control: RefCell<Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlInstallation>>,
    operation_bank: RefCell<Option<crate::backend::runtime::execution::generic::OriginalOperationActivation>>,
    // Same issued text step's loaded capture source; owns no session/model loan.
    partition_capture: RefCell<Option<text_quote::OriginalPartitionCaptureFrame>>,
    // Only TextOperation installs a genuine original set. Copies never refill it.
    prediction_scopes:
        RefCell<Option<crate::backend::submission_recovery::prediction::PredictionSet>>,
    payload: RefCell<Option<SessionPayloadOwner>>,
    lease: RefCell<Option<SubmissionLease>>,
    poison: Rc<Cell<bool>>,
    scopes: Cell<usize>,
    release_requested: Cell<bool>,
    // Token handles retain this owner after the native session lease is released.
    inference_retention: RefCell<eredu_runtime::working_memory::InferenceRetention>,
    // Resulting branch identity, captured once by execution finalization. This
    // stamp retains no native payload or reservation and survives owner cleanup.
    state_revision: RefCell<Option<eredu_runtime::working_memory::InferenceStateRevision>>,
    funding_retired: Cell<bool>,
    purpose: SubmissionPurpose,
    // An operation's disk mechanism restriction follows exact recovery, then
    // retires even when an emitted token keeps this owner alive.
    direct_route: RefCell<Option<crate::backend::runtime::residency::manager::DiskRouteGuard>>,
    // Opaque token handles can outlive the releasable session payload.
    memory_owner: RefCell<Option<crate::backend::managed_memory::NativeMemoryOwner>>,
    memory_retention: RefCell<NativeMemoryRetention>,
    completion_output: RefCell<completion_roots::CompletionOutputIngress>,
    // Last: original work custody survives every other concrete payload field.
    funding: RefCell<Option<text_funding::FundedWorkOwner>>,
    sampling_funding: RefCell<Option<text_funding::FundedWorkOwner>>,
    sampling_finalized: Cell<bool>,
    // Last: enclosing original operation controls outlive every native/payload field.
    original_operation_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

impl SubmissionResources {
    fn completion_failure(&self, status: Status, ordinary_message: &'static str) -> Error {
        if matches!(self.purpose, SubmissionPurpose::OriginalModel) {
            Error::OriginalOperationCompletion {
                settled: status.settled,
                failed: status.failed,
                blocked: status.blocked,
            }
        } else if self.purpose.is_saved_copy() {
            Error::SavedCopyCompletion {
                settled: status.settled,
                failed: status.failed,
                blocked: status.blocked,
            }
        } else {
            Error::ArchitectureModel(ordinary_message.into())
        }
    }

    pub(super) fn new(lease: SubmissionLease, poison: Rc<Cell<bool>>) -> SubmissionResourcesOwner {
        Self::with_purpose(lease, poison, SubmissionPurpose::Model)
    }

    fn with_purpose(
        lease: SubmissionLease,
        poison: Rc<Cell<bool>>,
        purpose: SubmissionPurpose,
    ) -> SubmissionResourcesOwner {
        Self::with_operation_funding(lease, poison, purpose, None)
    }

    fn with_operation_funding(
        lease: SubmissionLease,
        poison: Rc<Cell<bool>>,
        purpose: SubmissionPurpose,
        original_operation_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
    ) -> SubmissionResourcesOwner {
        SubmissionResourcesOwner::new(Self {
            prediction_scopes: RefCell::new(None),
            prefill_scopes: RefCell::new(None),
            model_execution: RefCell::new(None),
            parallel_control: RefCell::new(None),
            operation_bank: RefCell::new(None),
            partition_capture: RefCell::new(None),
            completion_output: RefCell::new(Default::default()),
            payload: RefCell::new(None),
            lease: RefCell::new(Some(lease)),
            poison,
            scopes: Cell::new(0),
            release_requested: Cell::new(false),
            inference_retention: RefCell::new(Default::default()),
            state_revision: RefCell::new(None),
            funding: RefCell::new(None),
            sampling_funding: RefCell::new(None),
            sampling_finalized: Cell::new(false),
            funding_retired: Cell::new(false),
            purpose,
            direct_route: RefCell::new(None),
            memory_owner: RefCell::new(None),
            memory_retention: RefCell::new(Default::default()),
            original_operation_funding,
        })
    }

    fn retain_memory_owner(
        &self,
        owner: Option<&crate::backend::managed_memory::NativeMemoryOwner>,
    ) {
        *self.memory_owner.borrow_mut() = owner.cloned();
    }

    pub(super) fn retain_memory(&self, retention: &NativeMemoryRetention) {
        self.memory_retention.borrow_mut().extend_from(retention);
    }

    fn allocation_memory(&self) -> NativeMemoryRetention {
        let mut memory = self.memory_retention.borrow().clone();
        if let Some(owner) = self.memory_owner.borrow().as_ref() {
            memory.retain(owner);
        }
        memory
    }

    fn observer_allocation_authority(
        &self,
    ) -> super::observation::ArrayObserverAllocationAuthority {
        super::observation::ArrayObserverAllocationAuthority::new(
            self.allocation_memory(),
            &self.inference_retention.borrow(),
        )
    }

    pub(super) fn retain_funded_array(&self, output: &Array) {
        let sampling = self.sampling_funding.borrow();
        let original = self.funding.borrow();
        if let Some(funding) = sampling.as_ref().or(original.as_ref()) {
            funding.retain(output);
        }
    }

    fn retain_output_allocation(&self, output: &Array) -> Result<(), Error> {
        if let Some(funding) = self.funding.borrow().as_ref() {
            funding.retain(output);
            return Ok(());
        }
        if output
            .allocation_info()?
            .is_some_and(|info| info.bytes() == 0)
        {
            return Ok(());
        }
        let retention = self.memory_retention.borrow();
        let owner = self.memory_owner.borrow();
        let requests = self.inference_retention.borrow();
        let owner_count = retention
            .owners()
            .len()
            .checked_add(usize::from(owner.is_some()))
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let request_count = requests.requests().count();
        if owner_count == 0 && request_count == 0 {
            return Ok(());
        }
        let pool = self
            .payload
            .borrow()
            .as_ref()
            .map(|payload| payload.memory_ledger.clone())
            .or_else(|| owner.as_ref().map(|owner| owner.pool().clone()))
            .or_else(|| retention.owners().first().map(|owner| owner.pool().clone()))
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?;
        for request in requests.requests() {
            request
                .memory_reservation()
                .validate_ledger(&pool)
                .map_err(Error::PrefillControl)?;
        }
        let bytes = owner_count
            .checked_mul(std::mem::size_of::<NativeMemoryOwner>())
            .and_then(|n| {
                request_count
                    .checked_mul(std::mem::size_of::<
                        eredu_runtime::working_memory::WorkingMemoryReservation,
                    >())
                    .and_then(|b| n.checked_add(b))
            })
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let prepared = crate::backend::managed_memory::OrdinaryArrayAttachment::<(
            NativeMemoryRetention,
            Vec<eredu_runtime::working_memory::WorkingMemoryReservation>,
        )>::prepare(&pool, bytes)?;
        // Constructor permission precedes both exact-capacity ownership vectors.
        let mut owners = Vec::with_capacity(owner_count);
        owners.extend(retention.owners().iter().cloned());
        owners.extend(owner.iter().cloned());
        let memory = NativeMemoryRetention::from_prepared_owners(owners);
        let mut reservations = Vec::with_capacity(request_count);
        reservations.extend(
            requests
                .requests()
                .map(|request| request.memory_reservation().clone()),
        );
        prepared.attach(output, (memory, reservations))
    }

    pub(super) fn retain_inference(
        &self,
        retention: &eredu_runtime::working_memory::InferenceRetention,
    ) {
        self.inference_retention.borrow_mut().extend_from(retention);
    }

    fn retain_request(&self, request: &eredu_runtime::working_memory::InferenceRequest) {
        self.inference_retention.borrow_mut().retain(request);
    }

    fn retain_input(&self, input: &MlxModelInput) {
        if let Some(request) = &input.inference_request {
            self.retain_request(request);
        }
        if let Some(memory) = &input.memory_owner {
            self.memory_retention.borrow_mut().retain(memory);
        }
    }

    pub(super) fn inference_retention(&self) -> eredu_runtime::working_memory::InferenceRetention {
        self.inference_retention.borrow().clone()
    }

    pub(super) fn state_revision(
        &self,
    ) -> Option<eredu_runtime::working_memory::InferenceStateRevision> {
        self.state_revision.borrow().clone()
    }

    pub(super) fn poison_on_unwind(&self) -> ObservationUnwind<'_> {
        ObservationUnwind(self)
    }

    pub(super) fn resources_releasable(&self) -> bool {
        self.scopes.get() == 0
    }

    pub(super) fn model_retirement_observer(
        &self,
    ) -> Result<Option<safemlx::OriginalScopeObserver>, Error> {
        // Only clone a fixed existing alias under this loan. Record destruction
        // runs later, after every RefCell/source borrow has ended.
        self.model_execution.try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref()
            .map(crate::backend::submission_recovery::prefill::ModelExecutionOwner::retirement_observer)
            .transpose()
    }

    fn close_parallel_control(&self) {
        if let Some(installation) = self.parallel_control.borrow().as_ref() {
            installation.close();
        }
    }

    pub(super) fn request_release(&self) {
        self.close_parallel_control();
        self.release_requested.set(true);
        self.release_if_settled();
    }

    fn release_if_settled(&self) {
        if self.release_requested.get() && self.scopes.get() == 0 {
            // Saved array copies have an explicit fallible destination-only
            // finalizer. They never publish or charge the source model here.
            if !self.poison.get() && matches!(&self.purpose, SubmissionPurpose::Model) {
                let funding = self.funding.borrow().clone();
                let payload = self.payload.borrow().clone();
                if let Some(funding) = funding {
                    if let Some(payload) = payload {
                        // Only an independently settled operation reaches this
                        // point. Unknown backing keeps the funding uncertified.
                        // Failure retains the remaining bound. Cold inspection
                        // cannot manufacture the missing completion evidence.
                        let publication = (|| {
                            let mut nonstate = funding.prepare_inventory()?;
                            let mut decoder = funding.prepare_inventory()?;
                            payload.collect_retained_idle_storage(&mut nonstate, &mut decoder)?;
                            if let Some(publication) = funding.publish_model(nonstate, decoder)? {
                                payload.nonstate_publication.replace(Some(publication));
                            }
                            Ok::<_, Error>(())
                        })();
                        if let Err(cause) = publication {
                            funding.retain_callback_failure(cause);
                        }
                    }
                    if self.funding_retired.get() {
                        if let Err(cause) = funding.certify() {
                            funding.retain_callback_failure(cause);
                        }
                    }
                }
            }
            // Replacement sampling has its own publication account, but the
            // same actual completion cut governs both accounts. It publishes
            // only its token/RNG roots, never the model or capture inventory.
            if !self.poison.get()
                && self.funding_retired.get()
                && !self.sampling_finalized.replace(true)
            {
                if let Some(funding) = self.sampling_funding.borrow().as_ref() {
                    if let Err(cause) = funding
                        .prepare_inventory()
                        .and_then(|inventory| funding.publish(inventory))
                        .and_then(|()| funding.certify())
                    {
                        funding.retain_callback_failure(cause);
                    }
                }
            }
            // A live old completion must not keep Rc::get_mut unavailable after
            // its authority is released for a newer submission.
            let partition_capture = self.partition_capture.borrow_mut().take();
            let prefill_scopes = self.prefill_scopes.borrow_mut().take();
            let model_execution = self.model_execution.borrow_mut().take();
            let direct_route = self.direct_route.borrow_mut().take();
            let operation_bank = self.operation_bank.borrow_mut().take();
            let payload = self.payload.borrow_mut().take();
            let lease = self.lease.borrow_mut().take();
            // All loans end before native/control/Q destruction. Outstanding
            // active collectors have their own Recovery custody; only this
            // operation's independent scope evidence permits this path.
            drop((
                partition_capture,
                prefill_scopes,
                model_execution,
                direct_route,
                operation_bank,
                payload,
                lease,
            ));
        }
    }

    pub(super) fn reject_unresolved(&self) {
        self.poison.set(true);
    }

    pub(super) fn is_healthy(&self) -> bool {
        !self.poison.get()
    }

    pub(super) fn ensure_healthy(&self) -> Result<(), Error> {
        // Settlement callbacks cannot return their publication failure. Move
        // that exact cause into the next fallible observation before a generic
        // session fence can mask it. The funding owner remains fenced afterward.
        let failure = self
            .funding
            .borrow()
            .as_ref()
            .and_then(|funding| funding.take_collection_failure())
            .or_else(|| {
                self.sampling_funding
                    .borrow()
                    .as_ref()
                    .and_then(|funding| funding.take_collection_failure())
            });
        if let Some(cause) = failure {
            return Err(cause);
        }
        if !self.is_healthy() {
            Err(Error::ArchitectureModel(
                "native session is poisoned by unresolved or failed work".into(),
            ))
        } else {
            Ok(())
        }
    }
}

struct ReleaseSubmission(SubmissionResourcesOwner);
impl Drop for ReleaseSubmission {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.reject_unresolved();
        }
        self.0.request_release();
    }
}

#[derive(Clone, Copy)]
pub(super) enum ScopeFunding {
    Model,
    Sampling,
}

pub(super) struct ScopeRetention {
    owner: SubmissionResourcesOwner,
    funding: ScopeFunding,
    paged: Option<crate::backend::nn::workspace::PagedScopeRetention>,
    // Last: closed Recovery allocation custody, independent of native Scope.
    _original: Option<eredu_runtime::working_memory::OriginalPredictionRecoveryCustody>,
}

pub(super) struct ObservationRetention {
    _roots: ObservationRoots,
    ticket: ScopeRetention,
}

impl crate::backend::submission_recovery::prediction::PredictionRetention for ScopeRetention {
    fn install_paged_scope(
        &mut self,
        scope: crate::backend::nn::workspace::PagedScopeRetention,
    ) -> Result<(), Error> {
        if self.paged.is_some() {
            return Err(Error::PrefillScopeUnavailable);
        }
        self.paged = Some(scope);
        Ok(())
    }

    fn install_prediction_custody(
        &mut self,
        custody: eredu_runtime::working_memory::OriginalPredictionRecoveryCustody,
    ) {
        self._original = Some(custody);
    }
}
impl crate::backend::submission_recovery::prediction::PredictionRetention for ObservationRetention {
    fn install_prediction_custody(
        &mut self,
        custody: eredu_runtime::working_memory::OriginalPredictionRecoveryCustody,
    ) {
        self.ticket._original = Some(custody);
    }
}

impl Retention for ObservationRetention {
    fn configure_ordinary_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
    ) -> Result<(), safemlx::error::Exception> {
        self.ticket.configure_ordinary_scope(scope)
    }
    fn observe(&self, status: Status) {
        self.ticket.observe(status);
    }
}

pub(super) struct ObservationUnwind<'a>(&'a SubmissionResources);
impl Drop for ObservationUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.reject_unresolved();
        }
    }
}

impl Retention for ScopeRetention {
    fn configure_ordinary_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
    ) -> Result<(), safemlx::error::Exception> {
        let sampling = self.owner.sampling_funding.borrow();
        let model = self.owner.funding.borrow();
        let work = match self.funding {
            ScopeFunding::Model => model.as_ref(),
            // Initial sampling shares its model Work; an admitted revision
            // retains its separate sampling Work through the same owner.
            ScopeFunding::Sampling => sampling.as_ref().or(model.as_ref()),
        };
        if let Some(work) = work {
            work.configure_ordinary_scope(scope)?;
        }
        Ok(())
    }
    fn observe(&self, status: Status) {
        if let Some(paged) = &self.paged {
            paged.observe(status);
        }
        if status.failed || status.blocked {
            self.owner.poison.set(true);
        }
    }
}

impl Drop for ScopeRetention {
    fn drop(&mut self) {
        self.owner.scopes.set(self.owner.scopes.get() - 1);
        self.owner.release_if_settled();
    }
}

pub(super) struct ResourceOperation<P: Probe = safemlx::SubmissionScope> {
    sampling: Option<crate::backend::nn::workspace::ResidentCompletionRecipe>,
    owner: SubmissionResourcesOwner,
    recovery: Option<Recovery<ScopeRetention, P>>,
}

impl ResourceOperation {
    pub(super) fn sampling_context<'a>(
        &self,
        stream: &'a Stream,
        logits: Option<&MlxTensor>,
        synchronization: Option<(
            &crate::backend::runtime::distributed::Group,
            &eredu_runtime::PartitionCommunicationAuthority,
            usize,
        )>,
    ) -> Result<Option<crate::backend::runtime::generation::OriginalSamplingContext<'a>>, Error>
    {
        let Some(recipe) = self.sampling else {
            return Ok(None);
        };
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        if !crate::backend::submission_recovery::prediction::owns_observer(
            self.recovery
                .as_ref()
                .ok_or(Error::PredictionScopeUnavailable)?,
            &observer,
        ) {
            return Err(Error::PredictionScopeUnavailable);
        }
        let mut role = self
            .owner
            .take_sampling_event_scope()?
            .ok_or(Error::PredictionScopeUnavailable)?;
        let synchronization = match (role.take_sampling_source(), synchronization) {
            (Some(source), Some((group, authority, root))) => {
                Some(source.bind(group, authority, root)?)
            }
            (None, Some(_)) => return Err(Error::PredictionScopeUnavailable),
            (_, None) => None,
        };
        crate::backend::runtime::generation::OriginalSamplingContext::new_with_source(
            recipe,
            role,
            observer,
            stream,
            logits.map(MlxTensor::as_array),
            synchronization,
        )
        .map(Some)
    }
    pub(super) fn begin_prediction(owner: &SubmissionResourcesOwner) -> Result<Self, Error> {
        owner.ensure_healthy()?;
        let role = owner.take_sampling_scope()?;
        let sampling = role.as_ref().and_then(|role| role.sampling_recipe());
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(1, role.is_some());
        Ok(Self {
            owner: owner.clone(),
            recovery: Some(owner.recovery_with_prediction(role)?),
            sampling,
        })
    }

    pub(super) fn begin(owner: &SubmissionResourcesOwner) -> Result<Self, Error> {
        owner.ensure_healthy()?;
        Ok(Self {
            owner: owner.clone(),
            recovery: Some(owner.recovery()?),
            sampling: None,
        })
    }
}

impl<P: Probe> ResourceOperation<P> {
    #[cfg(test)]
    pub(super) fn with_probe(owner: &SubmissionResourcesOwner, probe: P) -> Self {
        Self {
            owner: owner.clone(),
            recovery: Some(Recovery::with_probe(owner.ticket(), probe)),
            sampling: None,
        }
    }

    pub(super) fn finish<T>(
        mut self,
        result: Result<T, Error>,
    ) -> Result<(T, Recovery<ScopeRetention, P>), Error> {
        let recovery = self.recovery.as_mut().expect("live resource operation");
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            if let Err(error) = result {
                return Err(error);
            }
            return Err(Error::ArchitectureModel(
                "native resource operation failed or is unobservable".into(),
            ));
        }
        Ok((result?, self.recovery.take().unwrap()))
    }
}

impl<P: Probe> Drop for ResourceOperation<P> {
    fn drop(&mut self) {
        if let Some(recovery) = self.recovery.as_mut() {
            // No rollback proof accompanies an abandoned operation, even if
            // all native children happened to finish successfully.
            self.owner.reject_unresolved();
            let status = recovery.observe_abandonment();
            if status.is_none_or(|status| !status.settled || status.failed || status.blocked) {
                self.owner.reject_unresolved();
            }
        }
    }
}

struct SessionOperation<'a, P: Probe = safemlx::SubmissionScope> {
    session: &'a mut MlxModelSession,
    owner: SubmissionResourcesOwner,
    recovery: Option<Recovery<ScopeRetention, P>>,
    handed_off: bool,
    token_validations: crate::backend::nn::tensor::TokenValidationIngress,
}

pub(super) fn restored_error_permits_recovery(status: Status, error: &Error) -> bool {
    !status.failed && !status.blocked && error.model_state_preserved()
}

pub(super) fn complete_model_operation<T, P: Probe>(
    value: T,
    owner: SubmissionResourcesOwner,
    recovery: Recovery<ScopeRetention, P>,
) -> Result<T, Error> {
    owner.request_release();
    let status = recovery.finish().map_err(|cause| {
        owner.reject_unresolved();
        cause.into_error()
    })?;
    if !status.settled || status.failed || status.blocked {
        owner.reject_unresolved();
        return Err(owner.completion_failure(
            status,
            "native operation failed or returned without proven completion; session is poisoned",
        ));
    }
    Ok(value)
}

impl<P: Probe> SessionOperation<'_, P> {
    fn begin_token_validation(&mut self) -> Result<TokenValidationScope, Error> {
        self.token_validations.begin()
    }

    fn model(&mut self) -> &mut Executable {
        &mut self
            .session
            .payload
            .get_mut()
            .expect("idle session has exclusive native payload ownership")
            .model
    }

    fn finish<T>(
        self,
        result: Result<T, Error>,
    ) -> Result<(T, SubmissionResourcesOwner, Recovery<ScopeRetention, P>), Error> {
        self.finish_with_preservation(result, false)
    }

    fn finish_execution<T>(
        self,
        result: Result<T, Error>,
    ) -> Result<(T, SubmissionResourcesOwner, Recovery<ScopeRetention, P>), Error> {
        let result = result.and_then(|value| {
            // Direct model callers have no sampler to transfer request handles
            // into submission ownership. Read the completed transaction's
            // retained authority before its output can escape the session.
            let retention = self
                .session
                .payload
                .model
                .erased()
                .retained_inference_authority()?;
            self.owner
                .state_revision
                .replace(Some(retention.revision().clone()));
            self.owner.retain_inference(&retention);
            Ok(value)
        });
        self.finish_with_preservation(result, true)
    }

    fn finish_with_preservation<T>(
        mut self,
        result: Result<T, Error>,
        allow_preservation: bool,
    ) -> Result<(T, SubmissionResourcesOwner, Recovery<ScopeRetention, P>), Error> {
        self.owner.close_parallel_control();
        self.owner
            .payload
            .replace(Some(self.session.payload.clone()));
        let recovery = self.recovery.as_mut().expect("live operation scope");
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            if let Err(error) = result {
                if matches!(self.owner.purpose, SubmissionPurpose::Model) {
                    self.session.record_failure(&error);
                }
                return Err(error);
            }
            return Err(self.owner.completion_failure(
                status,
                "native session execution failed or became unobservable; session is poisoned",
            ));
        }
        let value = match result {
            Err(error) if allow_preservation && restored_error_permits_recovery(status, &error) => {
                // Only a direct model call can supply this evidence. A larger
                // speculative/cache operation may have other mutated state.
                // Pending cleanup keeps the restored executable and its lease
                // in recovery without permanently poisoning the session. The
                // lease still excludes reuse until every scope retires, and a
                // later failed/blocked observation still poisons its owner.
                self.handed_off = true;
                self.owner.request_release();
                self.recovery.take();
                return Err(error);
            }
            Err(error) => {
                // The saved-copy finalizer returns the exact original error.
                // Do not allocate an additional diagnostic copy during its
                // bounded completion/failure cleanup.
                if matches!(self.owner.purpose, SubmissionPurpose::Model) {
                    self.session.record_failure(&error);
                }
                return Err(error);
            }
            Ok(value) => value,
        };
        self.handed_off = true;
        Ok((value, self.owner.clone(), self.recovery.take().unwrap()))
    }
}

impl<P: Probe> Drop for SessionOperation<'_, P> {
    fn drop(&mut self) {
        if self.handed_off {
            return;
        }
        self.owner.reject_unresolved();
        self.owner
            .payload
            .replace(Some(self.session.payload.clone()));
        if let Some(recovery) = self.recovery.as_mut() {
            let status = recovery.observe_abandonment();
            if status.is_none_or(|status| !status.settled || status.failed || status.blocked) {
                self.owner.reject_unresolved();
            }
        }
        self.owner.request_release();
        // The preallocated node either releases the scope ticket or retains it
        // together with the executable and lease in nonblocking quarantine.
        self.recovery.take();
    }
}

mod pending_prompt;

/// MLX-owned prefill input.
///
/// Ordinary conversion clones array handles, not tensor storage. Closed pending
/// token inputs share their actual part and host custody. Both remain independent
/// of the caller's temporary `ModelInput` view.
#[derive(Debug, Clone)]
pub struct MlxModelInput {
    parts: pending_prompt::ModelInputParts,
    controlled_attribution: Option<eredu_core::SharedPromptAttribution>,
    original_media: Option<input::OriginalMediaPacket>,
    placement_semantics: Option<eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
    cache_identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    prefill_chunk_positions: Option<std::num::NonZeroU64>,
    inference_request: Option<eredu_runtime::working_memory::InferenceRequest>,
    memory_owner: Option<NativeMemoryOwner>,
    // Only the checked token-ID preparation worker can publish this marker.
    // Numerical aliases preserve it; independently built/copied inputs do not.
    quote: Option<text_quote::TextExecutionQuoteOwner>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: pending_prompt::ModelInputParts::Owned(input.parts.to_vec()),
            controlled_attribution: None,
            // Public ModelInput exposes a replaceable parts slice. Rebuilding
            // from that view never transfers the private completed-upload proof.
            original_media: None,
            placement_semantics: None,
            cache_identity: input.shared_cache_identity().cloned().or_else(|| {
                input
                    .cache_identity()
                    .cloned()
                    .map(eredu_runtime::SharedPreparedInputCacheIdentity::new)
            }),
            prefill_chunk_positions: input.prefill_chunk_positions(),
            inference_request: input.inference_request().cloned(),
            memory_owner: input.memory_owner().cloned(),
            quote: None,
        }
    }
}

impl MlxModelInput {
    /// Borrow the exact single plain-text token source without rebuilding input
    /// parts or changing its policy. Completion/shape/dtype are checked by the
    /// caller's source-bound preparation worker.
    pub(crate) fn plain_token_array(&self) -> Option<&Array> {
        let [part] = &*self.parts else {
            return None;
        };
        if part.modality() != eredu_core::InputModality::Text {
            return None;
        }
        let input::InputPayload::TokenIds(tokens) = part.payload() else {
            return None;
        };
        Some(tokens)
    }

    /// Source-authenticated extent for the ordinary pending-copy follow-on.
    /// Raw constructors and semantic relabeling carry no such proof.
    pub(crate) fn controlled_decoder_positions(&self) -> Option<u64> {
        self.controlled_attribution
            .as_ref()
            .map(|value| value.attribution().decoder_positions)
    }

    /// Logical shared metadata retained by ordinary snapshot copies. This does
    /// not certify native storage, map-node capacity, or construction peak.
    pub(super) fn copied_metadata_bytes(&self) -> Option<u64> {
        let cache = self
            .cache_identity
            .as_ref()
            .map_or(Some(0), |v| v.as_ref().logical_metadata_bytes())?;
        let attribution = self
            .controlled_attribution
            .as_ref()
            .map_or(Some(0), |v| v.logical_storage_bytes())?;
        cache.checked_add(attribution)
    }

    /// Only the ordinary per-slot copy worker supplies these reconstructed
    /// parts. Policy/attribution stay coupled to the original immutable source;
    /// moving this Vec avoids cloning each copied metadata map again.
    pub(super) fn from_copied_parts(
        parts: Vec<input::InputPart>,
        source: &Self,
        owner: NativeMemoryOwner,
    ) -> Self {
        Self {
            parts: pending_prompt::ModelInputParts::Owned(parts),
            controlled_attribution: source.controlled_attribution.clone(),
            // Independent copies use their actual new slots through ordinary admission.
            original_media: None,
            placement_semantics: None,
            cache_identity: source.cache_identity.clone(),
            prefill_chunk_positions: source.prefill_chunk_positions,
            inference_request: source.inference_request.clone(),
            memory_owner: Some(owner),
            quote: None,
        }
    }

    pub(super) fn has_original_input_custody(&self) -> bool {
        matches!(&self.parts, pending_prompt::ModelInputParts::Original(_))
            || self
                .cache_identity
                .as_ref()
                .is_some_and(|identity| identity.original_source().is_some())
            || self.quote.is_some()
            || self.inference_request.as_ref().is_some()
    }

    pub(super) fn with_memory_owner(mut self, owner: NativeMemoryOwner) -> Self {
        self.memory_owner = Some(owner);
        self.quote = None;
        self
    }

    /// Retains an already admitted request through prompt borrowing and native
    /// execution. The runtime validates its target, geometry and one-use start.
    pub(crate) fn with_inference_request(
        mut self,
        request: eredu_runtime::working_memory::InferenceRequest,
    ) -> Self {
        self.prefill_chunk_positions =
            std::num::NonZeroU64::new(request.geometry().prefill_chunk_positions);
        self.inference_request = Some(request);
        self.quote = None;
        self
    }

    /// Copies request policy alongside an independently copied input payload.
    /// Retention preserves the original start authority; it never grants another.
    pub(super) fn inherit_request_policy(mut self, source: &Self) -> Self {
        self.prefill_chunk_positions = source.prefill_chunk_positions;
        self.inference_request = source.inference_request.clone();
        self.controlled_attribution = source.controlled_attribution.clone();
        self.original_media = None;
        self.placement_semantics = None;
        self.quote = None;
        self
    }

    /// Selects the maximum positions per ordinary prefill span. Selected media
    /// adapters retain their architecture-owned ingress across decoder spans.
    /// This schedules work; it does not establish a media memory budget.
    pub fn with_prefill_chunk_positions(mut self, positions: std::num::NonZeroU64) -> Self {
        self.prefill_chunk_positions = Some(positions);
        self.quote = None;
        self
    }

    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        let mut owned = input.with_model_input(|borrowed| Self::from(borrowed));
        owned.cache_identity = input.shared_cache_identity().cloned();
        owned
    }

    /// Returns the exact semantic identity carried by processor-produced input.
    pub fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cache_identity.as_ref().map(AsRef::as_ref)
    }

    /// Borrows the physical identity owner shared by prompt/session snapshots.
    pub fn shared_cache_identity(
        &self,
    ) -> Option<&eredu_runtime::SharedPreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Couples manually prepared tensors to caller-owned semantic content.
    pub fn with_semantic_content_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Result<Self, Error> {
        if self.inference_request.as_ref().is_some() {
            // Replacing a funded identity needs a new construction plan. Losing
            // quote provenance alone cannot authorize unpriced host allocation.
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )));
        }
        self.quote = None;
        self.controlled_attribution = None;
        self.original_media = None;
        self.placement_semantics = None;
        let prepared = eredu_runtime::PreparedModelInput::new(self.parts.to_vec(), |array| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, array)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let identity = eredu_runtime::SharedPreparedInputCacheIdentity::new(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        if let Some(memory) = &self.memory_owner {
            memory.retain_metadata(&eredu_runtime::SharedHostMetadata::Input(identity.clone()))?;
        }
        self.cache_identity = Some(identity);
        Ok(self)
    }

    /// Borrows the owned input parts as a model-input view for one operation.
    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        let input = match self.cache_identity.as_ref() {
            Some(identity) => input::ModelInput::with_shared_cache_identity(&self.parts, identity),
            None => input::ModelInput::new(&self.parts),
        };
        let input = match &self.memory_owner {
            Some(owner) => input.with_memory_owner(owner),
            None => input,
        };
        let input = match self.prefill_chunk_positions {
            Some(positions) => input.with_prefill_chunk_positions(positions),
            None => input,
        };
        let input = match &self.inference_request {
            Some(request) => input.with_inference_request(request),
            None => input,
        };
        let input = match &self.original_media {
            Some(packet) => input.with_original_media(packet),
            None => input,
        };
        let input = match self
            .quote
            .as_ref()
            .and_then(|quote| quote.execution_metadata())
        {
            Some(metadata) => input.with_original_media_metadata(metadata),
            None => input,
        };
        let input = match self.placement_semantics.as_ref().or_else(|| {
            self.quote
                .as_ref()
                .and_then(|quote| quote.copied_media_semantics())
        }) {
            Some(semantics) => input.with_copied_media_semantics(semantics),
            None => input,
        };
        execute(input)
    }
}

/// Opaque independently writable model state for one exact loaded executable.
/// This covers native persistent state only; complete generation snapshots also
/// compose sampling, pending input, facade semantics and capture admissions.
pub struct MlxNativeTextState {
    state: Box<dyn std::any::Any>,
    displaced_placement: Option<eredu_runtime::replicated_session::ControlBranchPlacement>,
    memory_retention: NativeMemoryRetention,
    // The erased Box deallocates before its independent constructor account.
    // Exchange may replace its contents, but never its allocation provenance.
    host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl eredu_core::execution_control::NativeTextStateBackend for MlxBackend<'_> {
    type NativeTextState = MlxNativeTextState;

    fn native_text_state_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport<&'static str> {
        use eredu_core::execution_control::ControlSupport;
        let session = runtime.session();
        if session.poison.get() {
            return ControlSupport::Unsupported {
                reason: "native session is fenced by unresolved or failed work",
            };
        }
        if !runtime
            .backend()
            .matches_prepared_target(&session.payload.target)
        {
            return ControlSupport::Unsupported {
                reason: "prepared execution and supplied context use different native targets",
            };
        }
        session.payload.model.erased().native_control_support()
    }

    fn estimate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: Option<&MlxNativeTextState>,
    ) -> Result<Option<eredu_core::execution_control::SnapshotEstimate>, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        // Reading estimates performs no reaping, native scopes or native allocation.
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if !matches!(
            Self::native_text_state_support(runtime),
            eredu_core::execution_control::ControlSupport::Supported
        ) {
            return Ok(None);
        }
        let estimate = session
            .payload
            .model
            .erased()
            .estimate_native_control_state(saved.map(|saved| saved.state.as_ref()))?;
        Ok(estimate.and_then(|mut estimate| {
            let metadata = saved
                .map(|saved| saved.memory_retention.logical_metadata_bytes())
                .unwrap_or(Some(0))?
                .max(NativeMemoryRetention::singleton_metadata_bytes())
                .checked_add(std::mem::size_of::<MlxNativeTextState>() as u64)?;
            estimate.retained_bytes = estimate.retained_bytes.checked_add(metadata)?;
            estimate.copy_bytes = estimate.copy_bytes.checked_add(metadata)?;
            Some(estimate)
        }))
    }

    fn estimate_native_text_growth(
        runtime: &ModelRuntime<Self>,
        saved: &MlxNativeTextState,
        additional: u64,
    ) -> Result<Option<u64>, Error> {
        Self::validate_native_text_state(runtime, saved)?;
        runtime
            .session()
            .payload
            .model
            .erased()
            .estimate_native_control_growth(saved.state.as_ref(), additional)
    }

    fn validate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: &MlxNativeTextState,
    ) -> Result<(), Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if let eredu_core::execution_control::ControlSupport::Unsupported { reason } =
            Self::native_text_state_support(runtime)
        {
            return Err(Error::ArchitectureModel(reason.into()));
        }
        session
            .payload
            .model
            .erased()
            .validate_native_control_state(saved.state.as_ref())
    }

    fn exchange_text_branch(
        runtime: &mut ModelRuntime<Self>,
        installed: eredu_core::TextBranchSource<'_, Self>,
        incoming: eredu_core::TextBranchSource<'_, Self>,
        slot: &mut MlxNativeTextState,
    ) -> Result<(), Error> {
        text_quote::exchange_branch(runtime, installed, incoming, slot)
    }

    fn exchange_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        slot: &mut MlxNativeTextState,
    ) -> Result<(), Error> {
        Self::validate_native_text_state(runtime, slot)?;
        // Partitioned exchanges complete their bounded all-rank preparation
        // before the checked host move. Do not open a subsequent native scope
        // whose failure could follow that move.
        let session = runtime.session_mut();
        session.ensure_no_submission_in_flight()?;
        let payload = session
            .payload
            .get_mut()
            .ok_or_else(|| Error::ArchitectureModel("native payload is still retained".into()))?;
        // The initial installed state belongs to the model. Later exchanges
        // also retain the independent authority which created that state.
        if let Some(memory) = payload._memory_owner.clone() {
            payload.state_memory.retain(&memory);
        }
        // Ordinary work may have materialized this state through another
        // context domain. Its authority must follow displaced state even after
        // the enclosing session and its operation history retire.
        {
            let payload: &mut SessionPayload = payload;
            payload
                .state_memory
                .extend_from(payload.operation_memory.get_mut());
        }
        payload
            .model
            .erased_mut()
            .exchange_native_control_state(slot.state.as_mut())?;
        std::mem::swap(&mut payload.state_memory, &mut slot.memory_retention);
        Ok(())
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession {
    payload: SessionPayloadOwner,
    poison: Rc<Cell<bool>>,
    failure: RefCell<Option<String>>,
    authority: RefCell<SessionAuthority>,
    // Capacity succession is private session policy metadata. Escaped outputs
    // retain physical charges, never the ability to authorize a later ceiling.
    capacity_handoffs: RefCell<Vec<eredu_runtime::working_memory::WorkingMemoryCapacityHandoff>>,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    capabilities: eredu_core::SessionCapabilities,
    capture_discovery:
        Option<std::sync::Arc<eredu_architectures::prepared_sources::PreparedModelDiscovery>>,
    speculative_capture_layouts: speculative_capture::LayoutsCell,
    partition_capture:
        RefCell<Option<partition_capture::Publication<partition_capture::PartitionCaptureData>>>,
    intervention_session_identity: String,
    state_residency: CacheResidencyPolicy,
}

impl MlxModelSession {
    #[cfg(test)]
    pub(in crate::composition::mlx) fn fixture_device_unit_window(
        &self,
        units: usize,
    ) -> Option<usize> {
        Some(
            self.payload
                .model
                .inference_blueprint()?
                .selected()
                .text_realization()
                .residency()
                .device_depth(units),
        )
    }
    #[cfg(test)]
    pub(in crate::composition::mlx) fn fixture_selected_parameter(
        &self,
        name: &str,
    ) -> Option<&eredu_runtime::SelectedParameterRealization> {
        self.payload
            .model
            .inference_blueprint()?
            .selected()
            .text_realization()
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == name)
    }

    #[cfg(test)]
    pub(in crate::composition::mlx) fn fixture_detached_payload_read_bytes(&self) -> Option<u64> {
        let workspace = self.payload.model.layerwise_workspace().ok()??;
        let mut bytes = workspace.detached_physical_read_bytes(0)?;
        let mut ordinal = 1;
        while let Some(read) = workspace.detached_physical_read_bytes(ordinal) {
            bytes = bytes.checked_add(read)?;
            ordinal = ordinal.checked_add(1)?;
        }
        Some(bytes)
    }

    #[cfg(test)]
    pub(crate) fn fixture_execution_identity(
        &self,
    ) -> eredu_runtime::working_memory::InferenceExecutionIdentity {
        self.payload
            .model
            .erased()
            .inference_execution_identity()
            .clone()
    }
    #[cfg(test)]
    pub(super) fn test_state_presence(
        &self,
    ) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.payload.model.erased().state_snapshot()
    }

    #[cfg(test)]
    pub(super) fn test_failed_operation(
        &mut self,
        probe: impl Probe,
        error: Error,
        allow_preservation: bool,
    ) -> Result<(), Error> {
        self.ensure_no_submission_in_flight()?;
        let lease = self.authority.borrow_mut().begin_submission().unwrap();
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        owner.retain_memory_owner(self.payload._memory_owner.as_ref());
        owner.retain_memory(&self.payload.state_memory);
        let recovery = Recovery::with_probe(owner.ticket(), probe);
        SessionOperation {
            session: self,
            owner,
            recovery: Some(recovery),
            handed_off: false,
            token_validations: Default::default(),
        }
        .finish_with_preservation::<()>(Err(error), allow_preservation)
        .map(|_| ())
    }

    #[cfg(test)]
    pub(super) fn test_payload_owner(&self) -> SessionPayloadOwner {
        self.payload.clone()
    }
    #[cfg(test)]
    pub(crate) fn test_payload_owner_count(&self) -> usize {
        self.payload.active_owner_count()
    }
    #[cfg(test)]
    pub(crate) fn test_payload_retirement_probe(&self) -> impl Fn() -> bool + 'static {
        self.payload.retirement_probe()
    }
    #[cfg(test)]
    pub(super) fn set_retirement_probe(&mut self, probe: Box<dyn std::any::Any>) {
        self.payload.get_mut().unwrap()._retirement_probe = Some(probe);
    }
    /// Creates a session retaining the model's exact native target.
    /// The backend provider checks that target before this method resets state.
    pub(crate) fn from_model(
        mut model: MlxModel,
        admitted_capabilities: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        ordinary_retirement::reclaim();
        let realized_capabilities = eredu_core::SessionCapabilities::new(true, true, true);
        SessionAdmission::new(admitted_capabilities)
            .validate(realized_capabilities)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let floating_state_dtype_bytes = model.floating_state_dtype_bytes();
        let memory_owner = model.memory_owner().cloned();
        let memory_ledger = memory_owner
            .as_ref()
            .map(|owner| owner.pool().clone())
            .unwrap_or_else(crate::backend::managed_memory::ledger);
        let state_residency = model.state_residency().clone();
        #[cfg(any(feature = "image", feature = "audio"))]
        let processor = model.take_processor();
        let distributed = model.take_distributed();
        let mut parameter_state = parameters::NativeParameterState::default();
        parameter_state.model_identity = distributed
            .as_ref()
            .map(MlxDistributedSession::register_parameter_model)
            .transpose()?
            .flatten();
        let capture_discovery = model.take_capture_discovery().map(std::sync::Arc::new);
        let intervention_session_identity =
            eredu_core::intervention::new_intervention_session_identity();
        let (executable, target) = model.into_execution_parts();
        #[cfg(test)]
        crate::tests::support::path_instrumentation::session_reset_attempt();
        let mut session = Self {
            // The closed owner may retire its final Rc under the native runtime
            // lock. It then queues the same preallocated semantic owner;
            // resident-manager leases and observers drop only at an ordinary
            // unlocked host boundary.
            payload: SessionPayloadOwner::new(SessionPayload {
                model: executable,
                parameter_state,
                target,
                distributed,
                #[cfg(any(feature = "image", feature = "audio"))]
                processor,
                #[cfg(test)]
                _retirement_probe: None,
                _memory_owner: memory_owner,
                state_memory: Default::default(),
                memory_ledger,
                operation_memory: Default::default(),
                nonstate_publication: Default::default(),
            }),
            poison: Rc::new(Cell::new(false)),
            failure: RefCell::new(None),
            authority: RefCell::new(SessionAuthority::new()),
            capacity_handoffs: RefCell::new(Vec::new()),
            floating_state_dtype_bytes,
            capabilities: realized_capabilities,
            capture_discovery,
            speculative_capture_layouts: Default::default(),
            partition_capture: RefCell::new(None),
            intervention_session_identity,
            state_residency,
        };
        session.reset()?;
        session.publish_initial_idle_storage()?;
        Ok(session)
    }

    pub(in crate::composition::mlx) fn floating_state_dtype_bytes(&self) -> std::num::NonZeroU8 {
        self.payload
            .parameter_state
            .floating_state_dtype_bytes
            .map_or(self.floating_state_dtype_bytes, |edited| {
                edited.max(self.floating_state_dtype_bytes)
            })
    }

    fn reject_unadmitted_text<T>(&self, backend: &MlxBackend<'_>) -> Result<T, Error> {
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        Err(Error::before_model_mutation(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }

    /// A quoted operation enters only with the move-only evidence returned by
    /// the current text permit. Historical request retention never selects it.

    fn begin_text_submission(
        &mut self,
        backend: &MlxBackend<'_>,
        operation: Option<text_step::TextOperation<'_>>,
    ) -> Result<SessionOperation<'_>, Error> {
        let Some(operation) = operation else {
            return self.reject_unadmitted_text(backend);
        };
        operation
            .validate(self, backend)
            .map_err(|error| error.at_original_stage("text submission operation validation"))?;

        let token_validations = operation
            .token_validation_ingress()
            .map_err(|error| error.at_original_stage("text submission token ingress"))?;

        let completion_output = operation
            .completion_output_ingress()
            .map_err(|error| error.at_original_stage("text submission completion ingress"))?;

        self.ensure_no_submission_in_flight()?;

        let inference = self.payload.model.erased().retained_inference_authority()?;

        let lease = self
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;

        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        let previous = owner.completion_output.replace(completion_output);
        drop(previous);
        owner.retain_request(operation.request());
        owner.retain_inference(&inference);
        operation
            .handoff(self, &inference)
            .map_err(|error| error.at_original_stage("text submission inference handoff"))?;

        let scope = operation.take_funding();
        if let Some(scope) = scope {
            owner
                .funding
                .replace(Some(operation.funded_work(scope).map_err(|error| {
                    error.at_original_stage("text submission work construction")
                })?));
        }

        let roles = operation
            .prediction_scopes()
            .map_err(|error| error.at_original_stage("text submission prediction scopes"))?;
        *owner.prediction_scopes.borrow_mut() = roles;
        *owner.sampling_funding.borrow_mut() = operation
            .sampling_work()
            .map_err(|error| error.at_original_stage("text submission sampling work"))?;

        let prefill = operation
            .prefill_scopes(self)
            .map_err(|error| error.at_original_stage("text submission prefill scopes"))?;
        let projection = prefill.map(|(bank, projection)| {
            *owner.prefill_scopes.borrow_mut() = Some(bank);
            projection
        });

        // The quote owns the finite bank across branch exchanges. This actual
        // submission alone activates its weak policy view, before native roles.
        *owner.operation_bank.borrow_mut() = operation
            .activate_operation_bank()
            .map_err(|error| error.at_original_stage("text submission residency activation"))?;

        let model_preparation = operation.model_execution_preparation(self)?;

        let role = owner.take_model_execution_scope()?;
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(0, role.is_some());

        let (recovery, model_execution) =
            owner.recovery_with_model_prediction(role, model_preparation)?;

        let projection = match model_execution.as_ref() {
            Some(model) => Some(
                crate::backend::submission_recovery::prefill::PrefillControlProjection::Model(
                    model.projection(),
                ),
            ),
            None => projection.map(
                crate::backend::submission_recovery::prefill::PrefillControlProjection::Prefill,
            ),
        };
        // Strong ownership precedes fallible weak installation. Scope/root
        // preparation and same-carrier binding precede all input/model work.
        owner.model_execution.replace(model_execution);
        self.payload
            .model
            .erased()
            .install_prefill_controls(projection)?;

        let (installation, projection) = match operation.parallel_control_installation()? {
            Some((installation, projection)) => (Some(installation), Some(projection)),
            None => (None, None),
        };
        // Until installation succeeds, the local guard closes any provisional
        // weak view on error. No native phase can execute inside this setter.
        self.payload
            .model
            .erased()
            .install_parallel_control(projection)?;
        *owner.parallel_control.borrow_mut() = installation;
        // Retain the exact source before lending an observer or running model
        // work. Construction refusal still follows this operation's Recovery.
        *owner.partition_capture.borrow_mut() = operation
            .partition_capture_frame(self)
            .map_err(|error| error.at_original_stage("text submission partition capture"))?;

        let submission = SessionOperation {
            session: self,
            owner,
            recovery: Some(recovery),
            handed_off: false,
            token_validations,
        };
        // Activation can fail, so establish ordinary recovery ownership first.
        // No input conversion, constructor or model work precedes this guard.
        submission.owner.direct_route.replace(
            operation.activate_disk_route().map_err(|error| {
                error.at_original_stage("text submission disk route activation")
            })?,
        );

        Ok(submission)
    }

    /// A residency handle alone never grants permission in a different ledger.
    /// All retained owners here are immutable unquoted authorities: no inventory
    /// publication can revoke the lease shared by an existing descendant.
    fn operation_memory(
        &self,
        context_pool: Option<&eredu_runtime::working_memory::MemoryLedger>,
    ) -> Result<NativeMemoryRetention, Error> {
        let mut memory = self.payload.operation_memory.borrow().clone();
        if let Some(owner) = self.payload._memory_owner.as_ref() {
            memory.retain(owner);
        }
        for pool in std::iter::once(&self.payload.memory_ledger).chain(context_pool) {
            if !memory.covers_pool(pool) {
                memory.retain(&NativeMemoryOwner::acquire(pool)?);
            }
        }
        // Do not publish partial acquisition if a later ledger rejects. Once
        // entry is authorized, preserve it before any fallible native work.
        self.payload
            .operation_memory
            .borrow_mut()
            .extend_from(&memory);
        // State-exchange ownership follows its state rather than becoming a
        // permanent history on the enclosing executable. This submission still
        // retains it through completion and any escaped output aliases.
        memory.extend_from(&self.payload.state_memory);
        Ok(memory)
    }

    fn begin_operation_in(
        &mut self,
        context_pool: Option<&eredu_runtime::working_memory::MemoryLedger>,
    ) -> Result<SessionOperation<'_>, Error> {
        self.ensure_no_submission_in_flight()?;
        self.payload.model.erased().install_parallel_control(None)?;
        let inference = self.payload.model.erased().retained_inference_authority()?;
        let memory = self.operation_memory(context_pool)?;
        let lease = self
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        owner.retain_memory_owner(self.payload._memory_owner.as_ref());
        owner.retain_memory(&self.payload.state_memory);
        owner.retain_memory(&memory);
        owner.retain_inference(&inference);
        let recovery = owner.recovery()?;
        Ok(SessionOperation {
            session: self,
            owner,
            recovery: Some(recovery),
            handed_off: false,
            token_validations: Default::default(),
        })
    }

    pub(in crate::composition::mlx) fn with_model_operation<T>(
        &mut self,
        operation: impl FnOnce(&mut Executable) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.with_model_operation_in(None, operation)
    }

    fn with_model_operation_in<T>(
        &mut self,
        context_pool: Option<&eredu_runtime::working_memory::MemoryLedger>,
        operation: impl FnOnce(&mut Executable) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut guard = self.begin_operation_in(context_pool)?;
        let result = operation(guard.model());
        let (value, owner, recovery) = guard.finish(result)?;
        complete_model_operation(value, owner, recovery)
    }

    fn with_shared_operation<T>(
        &self,
        context_pool: Option<&eredu_runtime::working_memory::MemoryLedger>,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.ensure_no_submission_in_flight()?;
        let memory = self.operation_memory(context_pool)?;
        let lease = self
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        owner.retain_memory_owner(self.payload._memory_owner.as_ref());
        owner.retain_memory(&self.payload.state_memory);
        owner.retain_memory(&memory);
        owner.payload.replace(Some(self.payload.clone()));
        let mut recovery = owner.recovery()?;
        let _release = ReleaseSubmission(owner.clone());
        let result = operation();
        if let Err(error) = &result {
            self.record_failure(error);
        }
        recovery.seal();
        // Successful host reads can leave completion bookkeeping pending. Wait
        // for that work before reusing the session; errors remain nonblocking.
        let status = if result.is_ok() {
            recovery.finish().map_err(|cause| {
                owner.reject_unresolved();
                cause.into_error()
            })?
        } else {
            recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            owner.reject_unresolved();
            return Err(result.err().unwrap_or_else(|| {
                Error::ArchitectureModel(
                    "native observation or sampling failed or remains unresolved".into(),
                )
            }));
        }
        if result.is_err() {
            owner.reject_unresolved();
        }
        result
    }

    /// Logical live-state estimate before host admission. Every rejection is a
    /// fixed unknown; ordinary validation retains its formatted diagnostics.
    pub(super) fn original_snapshot_estimate(
        &self,
        backend: &MlxBackend<'_>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if !self.matches_healthy_backend(backend) {
            return None;
        }
        self.authority.try_borrow().ok()?.require_idle().ok()?;
        let mut estimate = self
            .payload
            .model
            .erased()
            .estimate_original_native_control_state()?;
        let metadata = NativeMemoryRetention::singleton_metadata_bytes()
            .checked_add(std::mem::size_of::<MlxNativeTextState>() as u64)?;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(metadata)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(metadata)?;
        Some(estimate)
    }

    // Same source/health predicates for fixed pre-admission rejection. Ordinary
    // validation below keeps its existing diagnostic and source behavior.
    fn matches_healthy_backend(&self, backend: &MlxBackend<'_>) -> bool {
        !self.poison.get() && backend.matches_prepared_target(&self.payload.target)
    }

    pub(crate) fn validate_backend(&self, backend: &MlxBackend<'_>) -> Result<(), Error> {
        self.ensure_healthy()?;
        backend.validate_prepared_target(&self.payload.target)
    }

    pub(in crate::composition::mlx) fn ensure_no_submission_in_flight(&self) -> Result<(), Error> {
        super::recovery::reap();
        ordinary_retirement::reclaim();
        self.ensure_healthy()?;
        self.authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        // Existing explicit cleanup boundary, never a cold quotation/estimate.
        // The authority loan ended before an expired weak can release custody.
        self.payload.model.erased().retire_expired_opening_rows()
    }

    fn lifecycle_failure(&self, error: Error) -> BackendFailure {
        let kind = if self.poison.get() {
            BackendFailureKind::InvalidSession
        } else if self.authority.borrow().require_idle().is_err() {
            BackendFailureKind::Busy
        } else {
            BackendFailureKind::Other
        };
        BackendFailure::new(kind, error)
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        if self.poison.get() {
            let message = self.failure.borrow().as_ref().map_or_else(
                || "native session is fenced by unresolved or failed work".to_owned(),
                |cause| format!("native session is fenced after prior operation failure: {cause}"),
            );
            return Err(Error::ArchitectureModel(message));
        }
        Ok(())
    }

    fn record_failure(&self, error: &Error) {
        let mut failure = self.failure.borrow_mut();
        if failure.is_none() {
            *failure = Some(error.to_string());
        }
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn processor(&self) -> Option<&ModelProcessor> {
        self.payload.processor.as_ref()
    }

    pub(crate) fn effective_model_type(&self) -> &str {
        self.payload.model.effective_model_type()
    }

    /// Reports how the session-owned model exposes speculative weights.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        self.payload.model.speculative_capability()
    }

    pub(in crate::composition::mlx) fn supports_external_capture_spans(&self) -> bool {
        self.payload
            .model
            .erased()
            .supports_external_capture_spans()
    }

    pub(in crate::composition::mlx) fn speculative_activation_discovery(
        &self,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;
        if let Some(binding) = self.speculative_partition_binding() {
            return binding.discovery();
        }
        let prepared = self.capture_discovery.as_ref().ok_or_else(|| {
            CaptureError::Unsupported("session has no retained activation catalog".into())
        })?;
        let Some(execution) = self
            .payload
            .model
            .erased()
            .speculative_activation_execution()
        else {
            return prepared.autoregressive_activations(
                &super::intervention::mechanisms(),
                &self.intervention_session_identity,
                self.payload.parameter_state.active.as_deref(),
            );
        };
        prepared.speculative_activations(
            &execution,
            &super::intervention::mechanisms(),
            &self.intervention_session_identity,
            self.payload.parameter_state.active.as_deref(),
        )
    }

    /// Installs causal observers on this session's selected embedded-prediction executor.
    pub fn install_embedded_prediction_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: RuntimeActivationObserver<MlxTensor, Error> + 'static,
        LogitsObserver: RuntimeActivationObserver<Array, Error> + 'static,
    {
        self.install_embedded_prediction_observer_set(
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                tensors, logits,
            ),
        )
    }

    /// Installs outer observers and optional phase-aware internal observation authority.
    /// This native binding consumes the architecture-owned observer set; portable
    /// capture admission and record delivery remain the caller's responsibility.
    pub fn install_embedded_prediction_observer_set(
        &mut self,
        observers: eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
            MlxTensor,
            Array,
            Error,
        >,
    ) -> Result<(), Error> {
        self.ensure_no_submission_in_flight()?;
        if !self.payload.model.erased().has_embedded_prediction() {
            return Err(Error::ArchitectureModel(
                "session has no selected embedded-prediction executor".into(),
            ));
        }
        self.with_model_operation(|model| {
            if model.install_embedded_prediction_observers(observers) {
                Ok(())
            } else {
                Err(Error::ArchitectureModel(
                    "session has no selected embedded-prediction executor".into(),
                ))
            }
        })
    }

    /// Moves one already charged internal speculative record after execution.
    /// This performs no native work and does not imply proposal acceptance.
    /// A live completion retaining the payload returns a neutral busy failure.
    pub fn take_speculative_activation_capture(
        &mut self,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeActivationCapture>,
        eredu_core::BackendFailure,
    > {
        let payload = self.payload.get_mut().ok_or_else(|| {
            eredu_core::BackendFailure::from_error(eredu_core::SessionAuthorityError::Busy)
        })?;
        Ok(payload
            .model
            .erased_mut()
            .take_speculative_activation_capture())
    }

    /// Drains an original portable capture or native transformation failure.
    /// Requires exclusive payload ownership even when the session is poisoned;
    /// reading this host evidence does not settle or recover native work.
    pub fn take_speculative_activation_error(
        &mut self,
    ) -> Result<Option<eredu_core::speculative::SpeculativeControlError>, eredu_core::BackendFailure>
    {
        let payload = self.payload.get_mut().ok_or_else(|| {
            eredu_core::BackendFailure::from_error(eredu_core::SessionAuthorityError::Busy)
        })?;
        Ok(payload
            .model
            .erased_mut()
            .take_speculative_activation_error())
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.payload.model.residency_report()
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.payload.model.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        self.payload.model.parameter_bank_report()
    }

    /// Returns the complete model-derived identity for a reusable prompt cache.
    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        self.ensure_parameter_cache_compatible()?;
        self.payload
            .model
            .prompt_cache_model_identity()
            .map_err(Into::into)
    }

    pub(in crate::composition::mlx) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        self.payload.model.architecture_capability_estimate()
    }

    pub(in crate::composition::mlx) fn prepared_input_plans(
        &self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
    ) -> Result<
        Vec<eredu_architectures::media_plan::PreparedInputPartPlan>,
        eredu_core::CapabilityError,
    > {
        self.payload.model.prepared_input_plans(input)
    }

    #[cfg(test)]
    pub(in crate::composition::mlx) fn speculative_model_mut(
        &mut self,
    ) -> Result<&mut Executable, Error> {
        self.ensure_no_submission_in_flight()?;
        // Fixture mutations may allocate after initial storage publication has
        // released the loading owner. Preserve their own unquoted authority.
        self.operation_memory(None)?;
        Ok(&mut self.payload.get_mut().expect("idle payload").model)
    }

    #[cfg(test)]
    pub(crate) fn neutral_prediction_target_mut(
        &mut self,
    ) -> Result<&mut dyn super::super::replicated_text::ErasedReplicatedTextExecutable, Error> {
        self.ensure_no_submission_in_flight()?;
        self.operation_memory(None)?;
        Ok(self
            .payload
            .get_mut()
            .expect("idle payload")
            .model
            .erased_mut())
    }

    /// Clears all MLX cache state under the authoritative selected policy.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.reset_in(None)
    }

    fn reset_in(
        &mut self,
        context_pool: Option<&eredu_runtime::working_memory::MemoryLedger>,
    ) -> Result<(), Error> {
        let policy = self.state_residency.clone();
        self.with_model_operation_in(context_pool, |model| {
            if model.has_neutral_partitioned_control() {
                model.reset_cache_distributed().map_err(Into::into)
            } else {
                model.reset_cache_with_options(policy).map_err(Into::into)
            }
        })
    }

    /// Submits one cached decode position from a portable token id.
    pub fn submit_token_decode(
        &mut self,
        backend: &MlxBackend<'_>,
        token_id: u32,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        self.submit_decode_input(backend, || {
            #[cfg(test)]
            crate::tests::support::path_instrumentation::session_input_creation_attempt();
            let token_ids = [token_id];
            Array::try_from_slice(token_ids.as_slice(), &[1])?
                .try_index_device(NewAxis, backend.stream())
                .map_err(Into::into)
        })
    }

    fn submit_decode_input(
        &mut self,
        backend: &MlxBackend<'_>,
        input: impl FnOnce() -> Result<Array, Error>,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        self.submit_decode_input_with_permission(backend, input, None)
    }

    fn submit_decode_input_with_permission(
        &mut self,
        backend: &MlxBackend<'_>,
        input: impl FnOnce() -> Result<Array, Error>,
        permission: Option<text_step::TextOperation<'_>>,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        self.submit_decode_input_with_capture_permission(backend, input, permission, None)
    }

    /// Submit using the capture bank protected by the original text request.
    /// Required one-use permission prevents this path from opening unquoted work.
    fn submit_decode_input_with_funded_capture(
        &mut self,
        backend: &MlxBackend<'_>,
        input: impl FnOnce() -> Result<Array, Error>,
        permission: text_step::TextOperation<'_>,
        capture: &mut eredu_runtime::capture::FundedCaptureSession,
        prediction: u64,
        domain: Option<eredu_core::capture::CaptureTokenDomain<'_>>,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        self.submit_decode_input_with_capture_permission(
            backend,
            input,
            Some(permission),
            Some((capture, prediction, domain)),
        )
    }

    fn submit_decode_input_with_capture_permission(
        &mut self,
        backend: &MlxBackend<'_>,
        input: impl FnOnce() -> Result<Array, Error>,
        permission: Option<text_step::TextOperation<'_>>,
        capture: Option<(
            &mut eredu_runtime::capture::FundedCaptureSession,
            u64,
            Option<eredu_core::capture::CaptureTokenDomain<'_>>,
        )>,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        let metadata = permission
            .as_ref()
            .and_then(text_step::TextOperation::execution_metadata);
        let result = (|| {
            let mut operation = self.begin_text_submission(backend, permission)?;

            let token_validation_scope = operation.begin_token_validation()?;

            let input = input();

            let input = match input {
                Ok(ref input) => Ok(input),
                Err(error) => Err(error),
            };
            let execute =
                |model: &mut Executable,
                 observer: &mut dyn RuntimeActivationObserver<Array, Error>| {
                    model.erased_mut().decode_result_with_observer_and_metadata(
                        input,
                        backend.stream(),
                        observer,
                        metadata.as_ref(),
                    )
                };

            let output = match capture {
                Some((capture, prediction, domain)) => {
                    operation.with_funded_capture(backend, capture, prediction, domain, execute)
                }
                None => execute(operation.model(), &mut eredu_runtime::NoopObserver),
            };

            let public_output = operation.model().erased().partition_public_output();

            let (output, owner, recovery) = operation.finish_execution(output)?;

            owned_model_submission(
                output,
                token_validation_scope.finish(),
                public_output,
                owner,
                recovery,
            )
        })();
        result.map_err(|cause| {
            match metadata
                .as_ref()
                .and_then(|context| context.metadata_funding())
            {
                Some(funding) => {
                    crate::composition::mlx::model::retain_planning_error(cause, funding)
                }
                None => cause,
            }
        })
    }

    /// Returns aggregate cache-residency telemetry for this session.
    pub fn cache_residency_report(
        &self,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        self.payload
            .model
            .cache_residency_report()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Atomically persists the completed prefix owned by this session.
    pub fn save_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<eredu_core::cache::SharedPromptCacheManifest, Error> {
        self.ensure_parameter_cache_compatible()?;
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        let funding = self.prepare_cache_persistence()?;
        let control = self.prepare_cache_control(
            eredu_runtime::replicated_session::SessionCacheControlOperation::Save,
            funding.context(),
        )?;
        let host = funding
            .context()
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        self.with_model_operation_funded(host, |model| {
            if model.has_neutral_partitioned_control() {
                model
                    .save_prompt_cache_distributed(
                        &funding,
                        control.as_ref(),
                        root,
                        descriptor,
                        prefix_token_ids,
                        options,
                    )?
                    .ok_or_else(|| {
                        Error::PromptCache(eredu_core::cache::PromptCacheError::RankHasNoState)
                    })
            } else {
                model
                    .save_prompt_cache(&funding, root, descriptor, prefix_token_ids, options)
                    .map_err(Into::into)
            }
        })
    }

    /// Opens a compatible persisted prefix and replaces this session's cache.
    pub fn load_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<eredu_core::cache::SharedPromptCacheManifest, Error> {
        self.ensure_parameter_cache_compatible()?;
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        let CacheResidencyPolicy::Paged(_) = &self.state_residency else {
            return Err(Error::PromptCache(
                eredu_core::cache::PromptCacheError::PagedStateRequired,
            ));
        };
        let funding = self.prepare_cache_persistence()?;
        let materialization = self.prepare_cache_materialization(backend, &funding)?;
        let control = self.prepare_cache_control(
            eredu_runtime::replicated_session::SessionCacheControlOperation::Load,
            funding.context(),
        )?;
        let host = funding
            .context()
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        self.with_model_operation_funded(host, |model| {
            let manifest = if model.has_neutral_partitioned_control() {
                model
                    .load_prompt_cache_distributed(
                        &funding,
                        &materialization,
                        control.as_ref(),
                        root,
                        expected,
                        prefix_token_ids,
                    )?
                    .ok_or_else(|| {
                        Error::PromptCache(eredu_core::cache::PromptCacheError::RankHasNoState)
                    })?
            } else {
                model.load_prompt_cache(
                    &funding,
                    &materialization,
                    root,
                    expected,
                    prefix_token_ids,
                )?
            };
            Ok(manifest)
        })
    }

    /// Opens a persisted prefix only when it matches an exact prepared input.
    pub fn load_prompt_cache_for_input(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input: &MlxModelInput,
    ) -> Result<eredu_core::cache::SharedPromptCacheManifest, Error> {
        self.ensure_parameter_cache_compatible()?;
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let identity = input.cache_identity.clone().ok_or_else(|| {
            Error::PromptCache(eredu_core::cache::PromptCacheError::PreparedInputIdentityRequired)
        })?;
        let CacheResidencyPolicy::Paged(_) = &self.state_residency else {
            return Err(Error::PromptCache(
                eredu_core::cache::PromptCacheError::PagedStateRequired,
            ));
        };
        let root = root.as_ref();
        let funding = self.prepare_cache_persistence()?;
        let materialization = self.prepare_cache_materialization(backend, &funding)?;
        let control = self.prepare_cache_control(
            eredu_runtime::replicated_session::SessionCacheControlOperation::Load,
            funding.context(),
        )?;
        let host = funding
            .context()
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        self.with_model_operation_funded(host, |model| {
            if model.has_neutral_partitioned_control() {
                model
                    .load_prompt_cache_for_input_distributed(
                        &funding,
                        &materialization,
                        control.as_ref(),
                        root,
                        expected,
                        prefix_token_ids,
                        identity,
                    )?
                    .ok_or_else(|| {
                        Error::PromptCache(eredu_core::cache::PromptCacheError::RankHasNoState)
                    })
            } else {
                model
                    .load_prompt_cache_for_input(
                        &funding,
                        &materialization,
                        root,
                        expected,
                        prefix_token_ids,
                        identity,
                    )
                    .map_err(Into::into)
            }
        })
    }

    /// Returns communication when this is a distributed session.
    pub fn distributed(&self) -> Option<&MlxDistributedSession> {
        self.payload.distributed.as_ref()
    }

    pub(super) fn original_sampling_selection(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        usize,
    )> {
        self.payload
            .model
            .erased()
            .partition_sampling_context()
            .map(|(group, authority, _stream, rank)| (group, authority, rank))
    }

    pub(super) fn synchronizes_sampling(&self) -> bool {
        self.payload
            .model
            .erased()
            .partition_sampling_context()
            .is_some()
    }

    /// Samples on the canonical rank and synchronizes the result for this
    /// distributed model session.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        self.with_shared_operation(None, || {
            self.sample_under_submission(
                logits,
                batch_size,
                sampler,
                temperature,
                prng_state,
                finished,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample_under_submission<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        self.ensure_healthy()?;
        let executable = self.payload.model.erased();
        let (group, authority, stream, sampling_rank) =
            executable.partition_sampling_context().ok_or_else(|| {
                Error::Parallel(
                    "sampling synchronization requires a distributed model session".into(),
                )
            })?;
        crate::backend::runtime::distributed::parallel::sample_and_synchronize_bounded(
            logits,
            batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            group,
            authority,
            stream,
        )
    }

    /// Submits instrumented prefill through this selected MLX session.
    pub fn submit_prefill_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: MlxModelInput,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        self.submit_prefill_result_with_observer(backend, Ok(input), None, observer)
    }

    fn submit_prefill_result_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        self.submit_prefill_cancellable_result_with_observer(
            backend,
            input,
            capture_geometry,
            &eredu_core::GenerationCancellationToken::new(),
            observer,
        )?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "uncancellable caller participated in cancelled prefill".into(),
            )
        })
    }

    fn submit_prefill_cancellable_result_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Option<Submission<Array, MlxSessionCompletion>>, Error> {
        self.submit_prefill_cancellable_result(
            backend,
            input,
            capture_geometry,
            cancellation,
            true,
            observer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_prefill_cancellable_result(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        retain_observer_allocations: bool,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Option<Submission<Array, MlxSessionCompletion>>, Error> {
        self.submit_prefill_cancellable_with_permission(
            backend,
            input,
            capture_geometry,
            cancellation,
            retain_observer_allocations,
            observer,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_prefill_cancellable_with_permission(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        retain_observer_allocations: bool,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
        permission: Option<text_step::TextOperation<'_>>,
    ) -> Result<Option<Submission<Array, MlxSessionCompletion>>, Error> {
        self.submit_prefill_cancellable_with_capture_permission(
            backend,
            input,
            capture_geometry,
            cancellation,
            retain_observer_allocations,
            observer,
            permission,
            None,
        )
    }

    /// Submit using the original protected bank and its actual request geometry.
    /// Completion and delivery remain separate: this entry does not drain a frame.
    fn submit_prefill_cancellable_with_funded_capture(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        permission: text_step::TextOperation<'_>,
        capture: &mut eredu_runtime::capture::FundedCaptureSession,
        bound: &text_capture::InstalledPrefillCapture<'_>,
        domain: Option<eredu_core::capture::CaptureTokenDomain<'_>>,
    ) -> Result<Option<Submission<Array, MlxSessionCompletion>>, Error> {
        let capture_geometry = match bound {
            text_capture::InstalledPrefillCapture::Prompt(_) => {
                Some(capture.source().admission().request())
            }
            // Its source origin remains absolute; physical input is the fresh
            // request's single token, authenticated by the admitted view below.
            text_capture::InstalledPrefillCapture::Continuation(_) => None,
        };
        self.submit_prefill_cancellable_with_capture_permission(
            backend,
            input,
            capture_geometry,
            cancellation,
            false,
            &mut eredu_runtime::NoopObserver,
            Some(permission),
            Some((capture, bound, domain)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_prefill_cancellable_with_capture_permission(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Result<MlxModelInput, Error>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        retain_observer_allocations: bool,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
        permission: Option<text_step::TextOperation<'_>>,
        capture: Option<(
            &mut eredu_runtime::capture::FundedCaptureSession,
            &text_capture::InstalledPrefillCapture<'_>,
            Option<eredu_core::capture::CaptureTokenDomain<'_>>,
        )>,
    ) -> Result<Option<Submission<Array, MlxSessionCompletion>>, Error> {
        let mut operation = self.begin_text_submission(backend, permission)?;
        if let Ok(input) = &input {
            operation.owner.retain_input(input);
        }
        // Funded capture retains its arrays through the exact existing work.
        // Only the legacy observer may request its separate unquoted authority.
        let mut allocation_authority = (capture.is_none() && retain_observer_allocations)
            .then(|| operation.owner.observer_allocation_authority());
        let token_validation_scope = operation.begin_token_validation()?;
        let input_succeeded = input.is_ok();
        let execute =
            |model: &mut Executable, observer: &mut dyn RuntimeActivationObserver<Array, Error>| {
                match input {
                    Ok(input) => input.with_borrowed(|input| {
                        model.erased_mut().prefill_cancellable_result_with_observer(
                            Ok(input),
                            None,
                            capture_geometry,
                            cancellation,
                            backend.stream(),
                            observer,
                        )
                    }),
                    Err(error) => model.erased_mut().prefill_cancellable_result_with_observer(
                        Err(error),
                        None,
                        capture_geometry,
                        cancellation,
                        backend.stream(),
                        observer,
                    ),
                }
            };
        let output = match capture {
            Some((capture, text_capture::InstalledPrefillCapture::Prompt(bound), domain)) => {
                operation.with_funded_prefill_capture(backend, capture, bound, domain, execute)
            }
            Some((capture, text_capture::InstalledPrefillCapture::Continuation(bound), domain)) => {
                operation.with_funded_continuation_capture(backend, capture, bound, domain, execute)
            }
            None => {
                let output = execute(
                    operation.model(),
                    &mut ArrayObserverAdapter {
                        inner: observer,
                        routed_path: None,
                        routed_invocation_active: false,
                        // Preserve the legacy success path's authority lifetime
                        // through whole-operation finalization; errors move it.
                        allocation_authority: if input_succeeded {
                            allocation_authority.clone()
                        } else {
                            allocation_authority.take()
                        },
                    },
                );
                output.and_then(|output| {
                    observer.finish()?;
                    Ok(output)
                })
            }
        };
        let (output, owner, recovery) = operation.finish_execution(output)?;
        let validations = token_validation_scope.finish();
        match output {
            Some(output) => model_array_submission(output, validations, owner, recovery).map(Some),
            None => {
                model_completion(None, validations, owner, recovery)?.wait()?;
                Ok(None)
            }
        }
    }

    fn submit_decode_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        _input: impl FnOnce() -> Result<Array, Error>,
        _observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        self.reject_unadmitted_text(backend)
    }
}

impl<'a> BackendSession<MlxBackend<'a>> for MlxModelSession {
    type PrefillInput = MlxModelInput;
    type DecodeInput = Array;
    type Output = MlxModelOutput;
    type Completion = MlxSessionCompletion;

    fn capabilities(&self) -> eredu_core::SessionCapabilities {
        self.capabilities
    }

    fn prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        _input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        self.reject_unadmitted_text(backend)
    }

    fn prefill_cancellable(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
        cancellation: &eredu_core::GenerationCancellationToken,
    ) -> Result<Option<Submission<Self::Output, Self::Completion>>, Error> {
        let public_output = self.payload.model.erased().partition_public_output();
        self.submit_prefill_cancellable_result(
            backend,
            Ok(input),
            None,
            cancellation,
            false,
            &mut eredu_runtime::NoopObserver,
        )
        .map(|submission| {
            submission.map(|submission| Submission {
                output: MlxModelOutput::new(
                    public_output.then(|| MlxTensor::from_array(submission.output)),
                ),
                completion: submission.completion,
            })
        })
    }

    fn decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        self.submit_decode_input(backend, || Ok(input))
    }

    fn observe_output(
        &self,
        backend: &MlxBackend<'a>,
        output: &Self::Output,
    ) -> Result<ObservationSet, Error> {
        self.validate_backend(backend)?;
        self.with_shared_operation(Some(backend.memory_ledger()), || {
            let mut observations = ObservationSet::new();
            if let Some(logits) = output.logits() {
                observations
                    .insert(
                        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                        ObservationValue::Tensor(
                            super::observation::readback::observe_tensor_retained(
                                logits,
                                backend.stream(),
                            )?,
                        ),
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            }
            Ok(observations)
        })
    }
}

impl<'a> InspectableBackendSession<MlxBackend<'a>> for MlxModelSession {
    fn inspect_prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let submission = self.submit_prefill_with_observer(backend, input, &mut collector)?;
        let logits = submission.wait()?;
        let output = if self.payload.model.erased().partition_public_output() {
            MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
        } else {
            MlxModelOutput::new(None)
        };
        let observations = self.with_shared_operation(Some(backend.memory_ledger()), || {
            collector.materialize(backend.stream())
        })?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }

    fn inspect_decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let submission = self.submit_decode_with_observer(backend, || Ok(input), &mut collector)?;
        let logits = submission.wait()?;
        let output = if self.payload.model.erased().partition_public_output() {
            MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
        } else {
            MlxModelOutput::new(None)
        };
        let observations = self.with_shared_operation(Some(backend.memory_ledger()), || {
            collector.materialize(backend.stream())
        })?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }
}

struct TextPreparationRetention {
    _request: eredu_runtime::working_memory::InferenceRequest,
    funding: Option<text_funding::FundedWorkOwner>,
}
impl super::recovery::Retention for TextPreparationRetention {
    fn configure_ordinary_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
    ) -> Result<(), safemlx::error::Exception> {
        if let Some(work) = &self.funding {
            work.configure_ordinary_scope(scope)?;
        }
        Ok(())
    }
    fn observe(&self, status: super::recovery::Status) {
        if let Some(funding) = &self.funding {
            funding.observe_preparation(status);
        }
    }
}

/// Prepared request policy retained by ordinary and controlled text generation.
#[derive(Debug, Clone)]
pub struct MlxTextPreparation {
    request: Option<eredu_runtime::working_memory::InferenceTextPreparation>,
    chunk: Option<std::num::NonZeroU64>,
    quote: Option<text_quote::TextExecutionQuoteOwner>,
}

fn validate_preparation_funding(preparation: &MlxTextPreparation) -> Result<(), Error> {
    if preparation.quote.is_none() || preparation.request.is_none() {
        return Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )));
    }
    Ok(())
}

impl eredu_runtime::working_memory::LoadedDecodeSourceBackend for MlxBackend<'_> {
    fn compile_loaded_decode_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::decoder_storage::DecodeCompilePlan<'_>,
    ) -> Result<eredu_runtime::working_memory::LoadedDecodeSource, eredu_core::BackendFailure> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_decode_source(plan)
            .map_err(|error| {
                // The actual partial compiler/source owner enters the closed core
                // envelope directly; no native Error::Other allocation intervenes.
                let kind = match error.accounting_failure() {
                    Some(WorkingMemoryError::Poisoned | WorkingMemoryError::IdentityMismatch) => {
                        eredu_core::BackendFailureKind::InvalidSession
                    }
                    Some(WorkingMemoryError::UnknownBound) => {
                        eredu_core::BackendFailureKind::Unsupported
                    }
                    _ => eredu_core::BackendFailureKind::ResourceExhausted,
                };
                eredu_core::BackendFailure::new(kind, error)
            })
    }
}

mod speculative_prompt;

impl eredu_runtime::input::OriginalModelInputBackend for MlxBackend<'_> {
    fn original_model_input_semantics(
        input: &Self::Prompt,
    ) -> Option<&eredu_runtime::working_memory::BoundCompositeSemanticStorage> {
        match input.original_media.as_ref()? {
            input::OriginalMediaPacket::Original(completed) => {
                Some(completed.borrowed_semantics().storage())
            }
            input::OriginalMediaPacket::Ordinary(_) => None,
        }
    }
    fn prepare_original_model_input(
        runtime: &ModelRuntime<Self>,
        plan: eredu_runtime::input::host::PreparedHostInputPlan<'_>,
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<eredu_runtime::input::OriginalModelInput<Self::Prompt>, eredu_core::BackendFailure>
    {
        let limits = limits
            .resolve(runtime.backend().memory_ledger().topology())
            .map_err(eredu_core::BackendFailure::from_error)?;
        text_quote::prepare_parameter_observation_paths(runtime.session(), &limits)
            .map_err(eredu_core::BackendFailure::from_error)?;
        original_host_input::prepare_model_input(runtime, plan, limits)
    }
}

impl eredu_runtime::working_memory::OriginalTokenizerBackend for MlxBackend<'_> {
    fn validate_semantic_source(
        runtime: &ModelRuntime<Self>,
        preparation: &eredu_runtime::working_memory::PreparedSemanticSource,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        if !runtime.session().matches_healthy_backend(runtime.backend()) {
            return Err(eredu_core::TokenInputRejection::IdentityMismatch);
        }
        let model = runtime
            .session()
            .original_model_source()
            .map_err(|_| eredu_core::TokenInputRejection::Busy)?;
        preparation
            .validate(
                runtime.backend().memory_ledger(),
                model.erased().inference_execution_identity(),
            )
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)
    }

    fn validate_original_prepared_input_domain(
        runtime: &ModelRuntime<Self>,
        prompt: &Self::Prompt,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        original_host_input::validate_text_domain(runtime, prompt, source)
    }

    fn prepare_semantic_prompt(
        runtime: &ModelRuntime<Self>,
        preparation: &eredu_runtime::working_memory::PreparedSemanticSource,
        input: &eredu_core::TokenIdsInputPlan<'_>,
        chunk: Option<std::num::NonZeroU64>,
    ) -> Result<Self::Prompt, eredu_core::BackendFailure> {
        speculative_prompt::prepare(runtime, preparation, input, chunk)
    }

    fn prepare_semantic_source(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<
        eredu_runtime::working_memory::PreparedSemanticSource,
        eredu_core::SpeculativeOutputError,
    > {
        if !runtime.session().matches_healthy_backend(runtime.backend()) {
            return Err(eredu_core::SpeculativeOutputError::Storage(
                "speculative semantic session mismatch",
            ));
        }
        source
            .validate_pool(runtime.backend().memory_ledger())
            .map_err(|_| {
                eredu_core::SpeculativeOutputError::Storage(
                    "speculative semantic source pool mismatch",
                )
            })?;
        eredu_runtime::working_memory::PreparedSemanticSource::new(
            source,
            runtime
                .session()
                .payload
                .model
                .erased()
                .inference_execution_identity(),
            limits.resolve(runtime.backend().memory_ledger().topology())?,
        )
    }

    fn prepare_original_text_source_budget(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<
        eredu_runtime::working_memory::OriginalTextSourceBudget,
        eredu_runtime::working_memory::OriginalTextSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        source
            .validate_pool(runtime.backend().memory_ledger())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        source
            .prepare_text_source_budget(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                limits.resolve(runtime.backend().memory_ledger().topology())?,
            )
            .map_err(Into::into)
    }
    fn validate_original_tokenizer_source(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
    ) -> Result<(), eredu_core::BackendFailure> {
        // Source/account authentication is scalar and happens before S/E work.
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| {
                eredu_core::GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure()
            })?;
        source
            .validate_pool(runtime.backend().memory_ledger())
            .map_err(|_| {
                eredu_core::GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure()
            })
    }
    fn compile_original_text_stop_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<
        eredu_runtime::working_memory::OriginalStopSource,
        eredu_runtime::working_memory::OriginalTextSourceError,
    > {
        if !runtime.session().matches_healthy_backend(runtime.backend()) {
            return Err(
                eredu_runtime::working_memory::OriginalTextSourceError::Domain(
                    eredu_core::TokenInputRejection::IdentityMismatch,
                ),
            );
        }
        runtime
            .backend()
            .memory_ledger()
            .compile_stop_source(plan)
            .map_err(Into::into)
    }
    fn encode_original_text_ids(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<
        eredu_runtime::working_memory::OriginalEncodedTokenIds,
        eredu_runtime::working_memory::OriginalTextSourceError,
    > {
        if !runtime.session().matches_healthy_backend(runtime.backend()) {
            return Err(
                eredu_runtime::working_memory::OriginalTextSourceError::Domain(
                    eredu_core::TokenInputRejection::IdentityMismatch,
                ),
            );
        }
        runtime
            .backend()
            .memory_ledger()
            .encode_tokenizer_ids(source, input, add_special_tokens)
            .map_err(Into::into)
    }
    fn compile_original_tokenizer_source_for_generation(
        runtime: &ModelRuntime<Self>,
        input: eredu_runtime::working_memory::OriginalTokenizerInput<'_>,
    ) -> Result<
        eredu_runtime::working_memory::OriginalTokenizer,
        eredu_runtime::working_memory::OriginalTokenizerSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| {
                eredu_runtime::working_memory::OriginalTokenizerSourceError::Domain(
                    eredu_core::TokenInputRejection::IdentityMismatch,
                )
            })?;
        runtime
            .backend()
            .memory_ledger()
            .compile_tokenizer_source_for_generation(input)
            .map_err(eredu_runtime::working_memory::OriginalTokenizerSourceError::from)
    }

    fn encode_original_tokenizer_ids(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<eredu_runtime::working_memory::OriginalEncodedTokenIds, eredu_core::BackendFailure>
    {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .encode_tokenizer_ids(source, input, add_special_tokens)
            .map_err(
                eredu_runtime::working_memory::OriginalTokenizerEncodeError::into_backend_failure,
            )
    }
    fn compile_original_tokenizer_file(
        runtime: &ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
    ) -> Result<eredu_runtime::working_memory::OriginalTokenizer, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_tokenizer_file(read)
            .map_err(
                eredu_runtime::working_memory::OriginalTokenizerInputError::into_backend_failure,
            )
    }
    fn compile_original_tokenizer(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::tokenizer_storage::TokenizerPlan<'_>,
    ) -> Result<eredu_runtime::working_memory::OriginalTokenizer, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_tokenizer(plan)
            .map_err(eredu_runtime::working_memory::OriginalTokenizerError::into_backend_failure)
    }
}

impl eredu_runtime::working_memory::OriginalStopSourceBackend for MlxBackend<'_> {
    fn compile_original_stop_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<eredu_runtime::working_memory::OriginalStopSource, eredu_core::BackendFailure> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_stop_source(plan)
            .map_err(|error| {
                // The actual partial compiler/source owner enters the closed core
                // envelope directly; no native Error::Other allocation intervenes.
                let kind = match error.accounting_failure() {
                    Some(WorkingMemoryError::Poisoned | WorkingMemoryError::IdentityMismatch) => {
                        eredu_core::BackendFailureKind::InvalidSession
                    }
                    Some(WorkingMemoryError::UnknownBound) => {
                        eredu_core::BackendFailureKind::Unsupported
                    }
                    _ => eredu_core::BackendFailureKind::ResourceExhausted,
                };
                eredu_core::BackendFailure::new(kind, error)
            })
    }
}

impl<'a> TextGenerationBackend for MlxBackend<'a> {
    type TextPreparation = MlxTextPreparation;
    type TextPreparationControl = crate::backend::distributed::MlxTextPreparationControl;
    type TextStepPermit = MlxTextStepPermit;

    fn text_preparation_report(
        preparation: &Self::TextPreparation,
    ) -> Option<eredu_core::TextPreparationReport<'_>> {
        let request = preparation.request.as_ref()?.request();
        let reservation = request.memory_reservation();
        Some(eredu_core::TextPreparationReport {
            geometry: request.geometry(),
            admission: reservation.admission(),
        })
    }

    fn acquire_host_preparation(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::HostPreparationAuthority, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        let lease = runtime
            .backend()
            .memory_ledger()
            .acquire_unquoted()
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(eredu_core::HostPreparationAuthority::retain(lease))
    }

    fn prepare_shared_token_filter(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> eredu_core::TokenFilter,
    ) -> Result<eredu_core::SharedTokenFilter, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .prepare_shared_token_filter(factory)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn prepare_shared_controller_bytes(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Vec<u8>,
    ) -> Result<eredu_core::SharedControllerBytes, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .prepare_shared_controller_bytes(factory)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn prepare_shared_controller_declaration<T: eredu_core::ControllerDeclarationData>(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Result<T, eredu_core::BackendFailure>,
    ) -> Result<eredu_core::SharedControllerDeclaration, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        runtime
            .backend()
            .memory_ledger()
            .prepare_shared_controller_declaration(factory)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn begin_text_step<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        state: &Self::TextGenerationState,
        controller: &C,
        input: eredu_core::PendingTextInput<&Self::Prompt, &Self::Token>,
        context: &eredu_core::TextStepContext,
    ) -> Result<Self::TextStepPermit, Error> {
        MlxTextStepPermit::begin(runtime, preparation, state, controller, input, context)
    }

    fn finish_text_step(permit: Self::TextStepPermit) -> Result<(), Error> {
        permit.finish()
    }

    fn bind_text_preparation_run<C: eredu_core::TokenFilterController>(
        _: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        controller: &C,
        context: &eredu_core::TextStepContext,
    ) -> Result<(), eredu_core::BackendFailure> {
        if let Some(quote) = &preparation.quote {
            quote
                .storage_contract()
                .validate(controller)
                .map_err(eredu_core::BackendFailure::from_error)?;
        }
        if let Some(preparation) = &preparation.request {
            preparation
                .bind_run(context)
                .map_err(eredu_core::BackendFailure::from_error)?;
        }
        #[cfg(test)]
        text_quote::record_prediction_context_for_test(preparation, context);
        Ok(())
    }

    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        cancellation: &eredu_core::GenerationCancellationToken,
        permit: &mut Self::TextStepPermit,
    ) -> Result<Option<Submission<Self::Token, Self::TextCompletion>>, Error> {
        let operation = permit.claim(
            runtime,
            state,
            eredu_core::PendingTextInput::Prefill(&prompt),
            decision,
        )?;
        let mut operation = operation;
        let allowance = operation
            .as_mut()
            .and_then(text_step::TextOperation::take_error_allowance);
        // Keep original custody outside the entire operation, including failed
        // Work construction, existing recovery finalization and sampling.
        let result = (|| {
            #[cfg(test)]
            if let Some(operation) = &operation {
                text_quote::intercept_claimed_work(runtime, state, operation)?;
            }
            text_execution::prefill(runtime, prompt, decision, state, cancellation, operation)
        })();
        // Execution has consumed TextOperation and ended its permit loan. Keep
        // the same allowance outside receipt attachment and its failure too.
        let result = match result {
            Ok(Some(mut submission)) => match permit.attach_receipt(&mut submission.output) {
                Ok(()) => Ok(Some(submission)),
                Err(error) => Err(error),
            },
            Ok(None) => Ok(None),
            Err(error) => Err(error),
        };
        text_error::finish(allowance, result)
    }

    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        permit: &mut Self::TextStepPermit,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let operation = permit.claim(
            runtime,
            state,
            eredu_core::PendingTextInput::Decode(&token),
            decision,
        )?;
        let mut operation = operation;
        let allowance = operation
            .as_mut()
            .and_then(text_step::TextOperation::take_error_allowance);
        let result = (|| {
            #[cfg(test)]
            if let Some(operation) = &operation {
                text_quote::intercept_claimed_work(runtime, state, operation)?;
            }
            text_execution::decode(runtime, token, decision, state, operation)
        })();
        let result = match result {
            Ok(mut submission) => match permit.attach_receipt(&mut submission.output) {
                Ok(()) => Ok(submission),
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        text_error::finish(allowance, result)
    }

    fn admit_text_preparation<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
    ) -> Result<Self::TextPreparation, eredu_core::BackendFailure> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(eredu_core::BackendFailure::from_error)?;
        let policy = config.inference_policy();
        policy
            .validate(config.sampling().max_new_tokens)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let (request, quote) = text_quote::admit(runtime, input, config, controller)?;
        let chunk = std::num::NonZeroU64::new(request.request().geometry().prefill_chunk_positions);
        let preparation = MlxTextPreparation {
            request: Some(request),
            chunk,
            quote: Some(quote),
        };
        #[cfg(test)]
        text_quote::record_tracking_admission(&preparation);
        Ok(preparation)
    }

    fn admit_text_preparation_with_options<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: &eredu_core::TextPreparationOptions,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        if options.interventions.is_some() {
            return Err(eredu_core::BackendFailure::from_error(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ));
        }
        let Some(source) = &options.capture else {
            return Self::admit_text_preparation(runtime, input, config, controller);
        };
        // The closed route proves finite native equations, actual source storage
        // and cumulative H before the shared driver prepares prompt or sampler.
        #[cfg(test)]
        if let Some(result) = text_quote::opening_rows_fixture::admit_if_selected(
            runtime,
            input,
            config.clone(),
            controller,
            source,
        ) {
            let (request, quote) = result?;
            let chunk =
                std::num::NonZeroU64::new(request.request().geometry().prefill_chunk_positions);
            return Ok(MlxTextPreparation {
                request: Some(request),
                chunk,
                quote: Some(quote),
            });
        }
        let (request, quote) =
            text_quote::admit_with_capture(runtime, input, config, controller, source)?;
        let chunk = std::num::NonZeroU64::new(request.request().geometry().prefill_chunk_positions);
        let preparation = MlxTextPreparation {
            request: Some(request),
            chunk,
            quote: Some(quote),
        };
        #[cfg(test)]
        if !text_quote::record_tracking_admission(&preparation) {
            text_quote::record_capture_sampling_admission(&preparation, source);
        }
        Ok(preparation)
    }

    fn admit_text_preparation_with_original_prepared<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        // The shared quote dispatch handles this source before ordinary work.
        // Do not reinterpret missing contributions as an unbudgeted request.
        if !matches!(input, eredu_core::TextPreparationInput::OriginalPrepared(_)) {
            return Err(
                eredu_core::PreparedRequestRejection::SourceUnavailable.into_backend_failure()
            );
        }
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }

    fn admit_text_preparation_with_sequence<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        let (request, quote) = text_quote::admit_with_sequence(
            runtime,
            input,
            config,
            controller,
            options.and_then(|options| options.capture.as_ref()),
            options.and_then(|options| options.interventions.as_ref()),
            claim,
        )?;
        let chunk = std::num::NonZeroU64::new(request.request().geometry().prefill_chunk_positions);
        let preparation = MlxTextPreparation {
            request: Some(request),
            chunk,
            quote: Some(quote),
        };
        #[cfg(test)]
        text_quote::record_sequence_admission(&preparation);
        Ok(preparation)
    }

    fn admit_text_preparation_with_sequence_consumer<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        // Every existing original producer converges on SequenceBinding, which
        // measures and seals the actual consumer before accepting the quote.
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }

    fn admit_text_preparation_with_decoder<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        // Explicit concrete source opt-in. Shared admission takes it once before
        // candidate retries and binds it into the same accepted original bank.
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }

    fn admit_text_preparation_with_plain_text<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        // Explicit combined-source opt-in. The runtime seals actual literal
        // stops, decoder geometry and identity in the same original bank.
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }

    fn prepare_generation_sequence_admitted(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        claim: eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<eredu_core::RetainedGenerationSequence, BackendFailure> {
        text_quote::prepare_generation_sequence(runtime, preparation, claim)
    }

    fn install_text_capture_admitted(
        runtime: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        preparation: &Self::TextPreparation,
        source: &eredu_core::capture::SharedCapturePlan,
        context: &eredu_core::TextStepContext,
    ) -> Result<(), BackendFailure> {
        text_capture::install(runtime, state, preparation, source, context)?;
        #[cfg(test)]
        if let Some(cause) = TEST_INSTRUMENTATION_FAILURE.with(|slot| slot.borrow_mut().take()) {
            return Err(BackendFailure::from_error(cause));
        }
        #[cfg(test)]
        text_quote::record_sequence_installation(runtime, state, preparation, source);
        Ok(())
    }

    fn admit_text_preparation_with_token_input<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &eredu_core::TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }

    fn prepare_original_text_prompt_admitted(
        backend: &Self,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, BackendFailure> {
        use eredu_core::TokenInputRejection as R;
        let quote = preparation
            .quote
            .as_ref()
            .ok_or_else(|| R::Unavailable.into_backend_failure())?;
        let request = preparation
            .request
            .as_ref()
            .ok_or_else(|| R::Unavailable.into_backend_failure())?;
        request
            .request()
            .memory_reservation()
            .validate_ledger(backend.memory_ledger())
            .map_err(|_| R::IdentityMismatch.into_backend_failure())?;
        quote
            .validate_original_prompt_preflight(backend, request)
            .map_err(R::into_backend_failure)?;
        #[cfg(test)]
        if let Some(error) = TEST_PROMPT_FAILURE.with(|slot| slot.borrow_mut().take()) {
            return Err(BackendFailure::from_error(error));
        }
        let input = quote
            .original_token_input()
            .ok_or_else(|| R::Unavailable.into_backend_failure())?;
        let scopes = quote
            .preparation_scopes
            .as_ref()
            .ok_or_else(|| R::Unavailable.into_backend_failure())?;
        // A pending Prompt owns the prior take; retry consumes no new input.
        let ids = if scopes
            .has_pending_prompt()
            .map_err(R::into_backend_failure)?
        {
            None
        } else {
            Some(text_quote::token_input::PromptIds::Original(input.take()?))
        };
        let mut prompt = scopes
            .prompt(quote, backend, ids, request)
            .map_err(BackendFailure::from_error)?;
        prompt.quote = Some(quote.clone());
        Ok(prompt)
    }

    fn prepare_text_prompt_admitted(
        backend: &Self,
        ids: Vec<u32>,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Error> {
        if preparation
            .quote
            .as_ref()
            .is_some_and(|q| q.original_token_input().is_some())
        {
            return Err(Error::PreparationScopeInputMismatch);
        }
        validate_preparation_funding(preparation)?;
        let quote = preparation.quote.clone();
        if let Some(quote) = &preparation.quote {
            quote.validate_prompt(&ids)?;
            quote
                .request()
                .memory_reservation()
                .validate_ledger(backend.memory_ledger())
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        match &preparation.request {
            Some(preparation) => {
                let mut prompt = if let Some(scopes) =
                    quote.as_ref().and_then(|q| q.native_preparation_scopes())
                {
                    scopes.prompt(
                        quote.as_ref().expect("joined original quote"),
                        backend,
                        Some(text_quote::token_input::PromptIds::Legacy(ids)),
                        preparation,
                    )?
                } else {
                    let stage = preparation
                        .claim_prompt()
                        .map_err(|error| Error::Other(Box::new(error)))?;
                    let funding = quote
                        .as_ref()
                        .map(|quote| quote.preparation_work())
                        .transpose()?;
                    let prompt = super::recovery::detached_retained(
                        TextPreparationRetention {
                            _request: preparation.request().clone(),
                            funding: funding.clone(),
                        },
                        || {
                            let request = preparation.request();
                            prepare_text_prompt(
                                backend,
                                ids,
                                None,
                                Some(request),
                                funding.as_deref(),
                            )
                        },
                    )?;
                    stage
                        .finish()
                        .map_err(|error| Error::Other(Box::new(error)))?;
                    prompt
                };
                prompt.quote = quote;
                Ok(prompt)
            }
            None => Self::prepare_text_prompt(backend, ids),
        }
    }

    fn bind_text_prompt_preparation(
        backend: &Self,
        prompt: Self::Prompt,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Error> {
        if let Some(quote) = &preparation.quote {
            if quote.has_completed_input() {
                let request = preparation.request.as_ref().ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))?;
                return quote.bind_completed_input_prompt(backend, request, prompt);
            }
            text_step::validate_prompt_binding(&prompt, quote)?;
            preparation
                .request
                .as_ref()
                .ok_or_else(|| {
                    Error::Other(Box::new(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ))
                })?
                .bind_prompt()
                .map_err(|error| Error::Other(Box::new(error)))?;
            return Ok(prompt);
        }
        match &preparation.request {
            Some(preparation) => {
                let prompt = prompt.with_inference_request(preparation.request().clone());
                preparation
                    .bind_prompt()
                    .map_err(|error| Error::Other(Box::new(error)))?;
                Ok(prompt)
            }
            None => Ok(match preparation.chunk {
                Some(chunk) => prompt.with_prefill_chunk_positions(chunk),
                None => prompt,
            }),
        }
    }

    fn start_text_generation_admitted(
        backend: &Self,
        config: TextGenerationConfig,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::TextGenerationState, Error> {
        validate_preparation_funding(preparation)?;
        let quote = preparation
            .quote
            .as_ref()
            .expect("validated complete preparation");
        let request = preparation
            .request
            .as_ref()
            .expect("validated complete preparation");
        if quote.config() != config {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
        }
        quote
            .request()
            .memory_reservation()
            .validate_ledger(backend.memory_ledger())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let mut state = if let Some(scopes) = quote.native_preparation_scopes() {
            scopes.sampling(quote, config, request)?
        } else {
            let stage = request
                .claim_sampling(config.clone())
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            let (sampler, completion) = stage
                .construct_sampler(quote.sampler_scope()?)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            let funding = quote.preparation_work()?;
            let mut state = super::recovery::detached_retained(
                TextPreparationRetention {
                    _request: request.request().clone(),
                    funding: Some(funding.clone()),
                },
                || {
                    let state = start_text_generation_with_sampler(
                        config,
                        Some(super::generation::MlxOrdinarySampler::Funded(sampler)),
                    )?;
                    if let Some(random) = &state.sampling.prng {
                        funding.retain(random.as_array());
                    }
                    Ok(state)
                },
            )?;
            state.sampling.inference_retention.retain(request.request());
            completion
                .finish()
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            state
        };
        state.funding = Some(quote.take_funding_run()?);
        state.sampling.parameter_epoch = Some(quote.parameter_epoch());
        state.sampling.quote = Some(quote.clone());
        #[cfg(test)]
        let state = text_quote::intercept_sequence_sampling(preparation, state)?;
        Ok(state)
    }
    fn requires_original_preparation_control(runtime: &ModelRuntime<Self>) -> bool {
        runtime.session().payload.distributed.is_some()
    }
    fn prepare_text_preparation_control(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        sequence: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Option<Self::TextPreparationControl>, BackendFailure> {
        preparation_control::prepare(runtime, input, config, sequence)
    }
    fn agree_text_preparation(
        runtime: &ModelRuntime<Self>,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, BackendFailure> {
        preparation_control::agree(runtime, None, stage, status)
    }
    fn agree_text_preparation_with_control(
        runtime: &ModelRuntime<Self>,
        control: Option<&Self::TextPreparationControl>,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, BackendFailure> {
        preparation_control::agree(runtime, control, stage, status)
    }

    fn text_preparation_usage(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::run_preparation::TextPreparationUsage, BackendFailure> {
        match &runtime.session().payload.distributed {
            Some(transport) => transport.text_preparation_usage(),
            None => Ok(Default::default()),
        }
    }

    fn reset_session(
        backend: &Self,
        session: &mut Self::Session,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        session
            .validate_backend(backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        claim
            .validate_session(session)
            .map_err(BackendFailure::from_error)?;
        let quoted = session.payload._memory_owner.is_none()
            && session
                .payload
                .operation_memory
                .borrow()
                .owners()
                .is_empty()
            && session.payload.state_memory.owners().is_empty();
        if quoted
            && session
                .payload
                .model
                .erased()
                .resident_reset_profile()
                .is_some()
        {
            session.publish_prepared_resident_reset(backend.memory_ledger(), claim)
        } else {
            session
                .reset_in(Some(backend.memory_ledger()))
                .map_err(|error| session.lifecycle_failure(error))
        }
    }

    // The production provider keeps the default rejection. This scoped fixture
    // receives the unchanged genuine core claim after explicit test settlement.
    #[cfg(all(
        test,
        target_vendor = "apple",
        feature = "metal",
        not(feature = "cuda")
    ))]
    fn reset_session_admitted(
        backend: &Self,
        session: &mut Self::Session,
        claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        resident_reset::tests::admitted(backend, session, claim)
    }

    fn synchronize_session(backend: &Self, session: &Self::Session) -> Result<(), BackendFailure> {
        session
            .validate_backend(backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        // A failed collective may still own quarantined native work. Do not
        // enter an unbounded queue wait after its communication owner is fenced.
        if let Some(transport) = &session.payload.distributed {
            eredu_runtime::run_preparation::TextPreparationTransport::ensure_preparation_active(
                transport,
            )?;
        }
        backend
            .weights_stream()
            .synchronize()
            .map_err(|error| BackendFailure::new(BackendFailureKind::Other, error))?;
        backend
            .synchronize()
            .map_err(|error| BackendFailure::new(BackendFailureKind::Other, error))?;
        // Queue completion alone cannot release session authority or prove that
        // recovery after a failed/cancelled operation has safely retired.
        session
            .ensure_no_submission_in_flight()
            .map_err(|error| session.lifecycle_failure(error))?;
        // Explicit synchronous housekeeping finishes nested retirement callbacks
        // after queue completion and the session's independent idle proof.
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        // Deferred state/output destruction can retire the last strong prefill
        // owner after the earlier idle check. Remove only newly expired weak
        // installations now, so their priced control headers do not retain the
        // completed request's entire host account. Live owners remain intact.
        session
            .ensure_no_submission_in_flight()
            .map_err(|error| session.lifecycle_failure(error))?;
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        Ok(())
    }

    fn text_sampling_control_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport<&'static str> {
        Self::text_execution_control_support(runtime)
    }
    type Prompt = MlxModelInput;
    type Token = MlxTextToken;
    type TextGenerationState = MlxTextGenerationState;
    type TextCompletion = MlxTextCompletion;

    fn text_execution_control_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport<&'static str> {
        <Self as eredu_core::execution_control::NativeTextStateBackend>::native_text_state_support(
            runtime,
        )
    }

    fn intervention_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        let session = runtime.session();
        let prepared = session.capture_discovery.as_ref().ok_or_else(|| {
            eredu_core::capture::CaptureError::Unsupported(
                "session has no retained intervention catalog".into(),
            )
        })?;
        let mechanisms = super::intervention::mechanisms();
        let mut discovery = prepared.intervention(&mechanisms)?;
        if let Some(partition) = session.loaded_partition_capture()? {
            discovery = eredu_runtime::inspection::intervention_support(
                discovery.points,
                &partition.discovery,
                &mechanisms,
            );
        }
        discovery.session_identity = Some(session.intervention_session_identity.clone());
        Ok(discovery)
    }

    fn active_parameter_overlay(runtime: &ModelRuntime<Self>) -> Option<&str> {
        runtime.session().payload.parameter_state.active.as_deref()
    }

    fn validate_text_interventions(
        runtime: &ModelRuntime<Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        Self::validate_text_capture(runtime, capture)?;
        if plan.is_empty() {
            return Ok(());
        }
        eredu_runtime::intervention::validate_session(
            capture,
            plan,
            &Self::intervention_discovery(runtime)?,
            &super::intervention::NativeInterventionEstimator,
        )
    }

    fn configure_text_interventions(
        runtime: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        capture: eredu_core::capture::AdmittedCapturePlan,
        plan: eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        text_step::validate_instrumentation(state)?;
        Self::validate_text_interventions(runtime, &capture, &plan)?;
        eredu_runtime::intervention::install_session(
            &mut state.capture,
            capture,
            Some((
                plan,
                std::sync::Arc::new(super::intervention::NativeInterventionEstimator),
            )),
        )?;
        if let (Some(capture), Some(partition)) = (
            state.capture.as_mut(),
            runtime.session().loaded_partition_capture()?,
        ) {
            capture.ensure_partition_capture(
                partition.identity(runtime.session().payload.parameter_state.active.as_deref())?,
            )?;
        }
        Ok(())
    }

    fn prepared_artifact_identity(
        runtime: &ModelRuntime<Self>,
    ) -> Option<eredu_core::artifact::ArtifactIdentity> {
        runtime
            .session()
            .capture_discovery
            .as_ref()?
            .resolved_artifact_identity()
    }

    fn capture_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
        if let Some(partition) = runtime.session().loaded_partition_capture()? {
            return Ok(partition.discovery.clone());
        }
        runtime
            .session()
            .capture_discovery
            .as_ref()
            .ok_or_else(|| {
                eredu_core::capture::CaptureError::Unsupported(
                    "session has no retained capture catalog".into(),
                )
            })?
            .capture()
    }

    fn configure_text_capture(
        runtime: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        plan: eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        text_step::validate_instrumentation(state)?;
        Self::validate_text_capture(runtime, &plan)?;
        eredu_runtime::intervention::install_session(&mut state.capture, plan, None)?;
        if let (Some(capture), Some(partition)) = (
            &mut state.capture,
            runtime.session().loaded_partition_capture()?,
        ) {
            capture.ensure_partition_capture(
                partition.identity(Self::active_parameter_overlay(runtime))?,
            )?;
        }
        Ok(())
    }

    fn validate_text_capture(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        use eredu_core::capture::*;
        if plan.is_empty() {
            return Ok(());
        }
        if plan.request().batch != 1 {
            return Err(CaptureError::Unsupported(
                "bounded capture requires single-sequence text generation".into(),
            ));
        }
        eredu_runtime::capture::validate_session(
            plan,
            &Self::capture_discovery(runtime)?,
            super::bounded_capture::estimate_shape,
        )
    }

    fn try_take_text_capture(
        state: &mut Self::TextGenerationState,
    ) -> Result<Option<eredu_core::capture::SharedCapturedStep>, Error> {
        if let Some(capture) = &mut state.funded_capture {
            if state.capture.is_some() {
                return Err(Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )));
            }
            // Core resolves its retained exact completions before this hook.
            // The host drain retains its pending owner on failure and does not
            // certify the original native operation or release recovery roots.
            return capture
                .collector_mut()
                .take_shared_step()
                .map_err(|error| Error::Other(Box::new(error)));
        }
        Ok(state
            .capture
            .as_mut()
            .and_then(|capture| capture.take_shared_step()))
    }

    fn text_capture_pending(state: &Self::TextGenerationState) -> bool {
        state
            .funded_capture
            .as_ref()
            .is_some_and(|capture| capture.collector().has_pending_step())
            || state
                .capture
                .as_ref()
                .is_some_and(|capture| capture.has_pending_step())
    }

    fn start_text_generation(
        backend: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Error> {
        let memory = NativeMemoryOwner::acquire(backend.memory_ledger())?;
        super::recovery::detached_retained(memory.clone(), || {
            let mut state = start_text_generation(config)?;
            state.sampling.memory_retention = NativeMemoryRetention::from_owner(&memory);
            Ok(state)
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        let memory = NativeMemoryOwner::acquire(backend.memory_ledger())?;
        super::recovery::detached_retained(memory.clone(), || {
            prepare_text_prompt(backend, prompt_token_ids, Some(memory), None, None)
        })
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        Self::submit_text_prefill_decision(
            runtime,
            prompt,
            &eredu_core::TokenSamplingDecision::new(filter.clone()),
            state,
        )
    }

    fn submit_text_prefill_decision(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        Self::submit_text_prefill_cancellable_decision(
            runtime,
            prompt,
            decision,
            state,
            &eredu_core::GenerationCancellationToken::new(),
        )?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "uncancellable caller participated in cancelled prefill".into(),
            )
        })
    }

    fn submit_text_prefill_cancellable_decision(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        cancellation: &eredu_core::GenerationCancellationToken,
    ) -> Result<Option<Submission<Self::Token, Self::TextCompletion>>, Error> {
        text_execution::prefill(runtime, prompt, decision, state, cancellation, None)
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        Self::submit_text_decode_decision(
            runtime,
            token,
            &eredu_core::TokenSamplingDecision::new(filter.clone()),
            state,
        )
    }

    fn submit_text_decode_decision(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        text_execution::decode(runtime, token, decision, state, None)
    }
}

#[cfg(test)]
std::thread_local! {
    static TEST_SAMPLING_FAILURE: std::cell::RefCell<Option<Error>> = const { std::cell::RefCell::new(None) };
    static TEST_PROMPT_FAILURE: std::cell::RefCell<Option<Error>> = const { std::cell::RefCell::new(None) };
    static TEST_INSTRUMENTATION_FAILURE: std::cell::RefCell<Option<eredu_core::capture::CaptureError>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
pub(crate) struct TestInstrumentationFailure;
#[cfg(test)]
impl Drop for TestInstrumentationFailure {
    fn drop(&mut self) {
        TEST_INSTRUMENTATION_FAILURE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
#[cfg(test)]
impl MlxBackend<'_> {
    pub(crate) fn fail_next_instrumentation_for_test(
        error: eredu_core::capture::CaptureError,
    ) -> TestInstrumentationFailure {
        TEST_INSTRUMENTATION_FAILURE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(error);
        });
        TestInstrumentationFailure
    }
}
#[cfg(test)]
pub(crate) struct TestPromptFailure;
#[cfg(test)]
impl Drop for TestPromptFailure {
    fn drop(&mut self) {
        TEST_PROMPT_FAILURE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
#[cfg(test)]
impl MlxBackend<'_> {
    pub(crate) fn fail_next_prompt_for_test(error: Error) -> TestPromptFailure {
        TEST_PROMPT_FAILURE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(error);
        });
        TestPromptFailure
    }
}
#[cfg(test)]
pub(crate) struct TestSamplingFailure;
#[cfg(test)]
impl Drop for TestSamplingFailure {
    fn drop(&mut self) {
        TEST_SAMPLING_FAILURE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
#[cfg(test)]
impl MlxBackend<'_> {
    pub(crate) fn fail_next_sampling_for_test(error: Error) -> TestSamplingFailure {
        TEST_SAMPLING_FAILURE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(error);
        });
        TestSamplingFailure
    }
}

fn model_array_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    owner: SubmissionResourcesOwner,
    recovery: Recovery<ScopeRetention>,
) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
    let completion = model_completion(Some(&output), token_validations, owner.clone(), recovery)?;
    owner.retain_output_allocation(&output)?;
    Ok(Submission { output, completion })
}

fn model_completion(
    output: Option<&Array>,
    token_validations: TokenValidationBatch,
    owner: SubmissionResourcesOwner,
    recovery: Recovery<ScopeRetention>,
) -> Result<MlxSessionCompletion, Error> {
    let release_on_failure = completion_roots::ConstructionRelease::new(&owner);
    let ingress = {
        let mut slot = owner
            .completion_output
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        slot.take()
    };
    let output = ingress.finish(output)?;
    let roots = CompletionRootsOwner::new(output, token_validations, owner.clone());
    let funding_retirement = owner.funding.borrow().as_ref().map(|funding| {
        for array in roots.arrays() {
            funding.retain(array);
        }
        text_funding::RetireFundedCompletion(owner.clone())
    });
    let completion = MlxSessionCompletion {
        inner: MlxSessionCompletionKind::Model {
            roots,
            owner: owner.clone(),
            recovery: RefCell::new(Some(recovery)),
            observation: super::output_completion::Observation::new(),
            _funding_retirement: funding_retirement,
        },
    };
    release_on_failure.disarm();
    Ok(completion)
}

fn owned_model_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    public_output: bool,
    owner: SubmissionResourcesOwner,
    recovery: Recovery<ScopeRetention>,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let submission = model_array_submission(output, token_validations, owner, recovery)?;
    Ok(Submission {
        output: MlxModelOutput::new(
            public_output.then(|| MlxTensor::from_array(submission.output)),
        ),
        completion: submission.completion,
    })
}

#[cfg(test)]
pub(super) fn model_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    public_output: bool,
    submission_lease: SubmissionLease,
) -> Submission<MlxModelOutput, MlxSessionCompletion> {
    let owner = SubmissionResources::new(submission_lease, Rc::new(Cell::new(false)));
    let mut recovery = owner.recovery().unwrap();
    recovery.seal();
    owned_model_submission(output, token_validations, public_output, owner, recovery).unwrap()
}

/// Allocation worker shared by admitted and independently guarded preparation.
fn start_text_generation(config: TextGenerationConfig) -> Result<MlxTextGenerationState, Error> {
    start_text_generation_with_sampler(config, None)
}

fn start_text_generation_with_sampler(
    config: TextGenerationConfig,
    sampler: Option<super::generation::MlxOrdinarySampler>,
) -> Result<MlxTextGenerationState, Error> {
    #[cfg(test)]
    if let Some(error) = TEST_SAMPLING_FAILURE.with(|slot| slot.borrow_mut().take()) {
        return Err(error);
    }
    let sampling = config.sampling();
    let prng = if sampling.temperature == 0.0 {
        None
    } else {
        Some(RandomState::from_key(safemlx::random::key(config.seed())?))
    };
    let sampler = match sampler {
        Some(sampler) => sampler,
        None => super::generation::MlxOrdinarySampler::Unquoted(
            MlxTextSampler::from_config(config).map_err(|error| {
                eredu_core::BackendError::Execution {
                    session: "text-generation".into(),
                    operation: "configure Mirostat V2".into(),
                    message: error.to_string(),
                }
            })?,
        ),
    };
    Ok(MlxTextGenerationState {
        sampling: super::generation::MlxTextSamplingState {
            temperature: sampling.temperature,
            prng,
            sampler,
            next_prediction: 0,
            parameter_epoch: None,
            inference_retention: Default::default(),
            memory_retention: Default::default(),
            quote: None,
        },
        capture: None,
        funded_capture: None,
        funding: None,
    })
}

/// The final-shaped eager source receives its owner before the input escapes.
fn prepare_text_prompt(
    backend: &MlxBackend<'_>,
    prompt_token_ids: Vec<u32>,
    memory: Option<NativeMemoryOwner>,
    request: Option<&eredu_runtime::working_memory::InferenceRequest>,
    funding: Option<&text_funding::FundedWork>,
) -> Result<MlxModelInput, Error> {
    prepare_text_prompt_slice(backend, &prompt_token_ids, memory, request, funding, false)
}
fn prepare_text_prompt_input(
    backend: &MlxBackend<'_>,
    ids: text_quote::token_input::PromptIds,
    request: &eredu_runtime::working_memory::InferenceRequest,
    funding: &text_funding::FundedWork,
    original_native: bool,
) -> Result<MlxModelInput, Error> {
    prepare_text_prompt_slice(
        backend,
        ids.tokens(),
        None,
        Some(request),
        Some(funding),
        original_native,
    )
}
fn prepare_text_prompt_slice(
    _backend: &MlxBackend<'_>,
    prompt_token_ids: &[u32],
    memory: Option<NativeMemoryOwner>,
    request: Option<&eredu_runtime::working_memory::InferenceRequest>,
    funding: Option<&text_funding::FundedWork>,
    original_native: bool,
) -> Result<MlxModelInput, Error> {
    if prompt_token_ids.is_empty() {
        return Err(Error::ArchitectureModel(
            "text generation requires at least one prompt token".into(),
        ));
    }
    let identity_plan = eredu_runtime::input::TextInputIdentityPlan::new(
        1,
        u64::try_from(prompt_token_ids.len())
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    )
    .and_then(|plan| plan.bind(&prompt_token_ids))
    .map_err(|error| Error::Other(Box::new(error)))?;
    // Upload and semantic hashing consume the same immutable borrowed source.
    let width = i32::try_from(identity_plan.tokens().len())
        .map_err(|_| Error::ArchitectureModel("prompt width exceeds int32".into()))?;
    let tokens = if original_native {
        Array::try_from_original_prompt_ids(identity_plan.tokens())?
    } else {
        Array::try_from_slice(identity_plan.tokens(), &[1, width])?
    };
    if let Some(funding) = funding {
        funding.retain(&tokens);
    }
    if let Some(memory) = &memory {
        memory.retain_array(&tokens)?;
    }
    if let Some(reservation) = request
        .map(|request| request.memory_reservation())
        .filter(|_| funding.is_none())
    {
        // Request authority contains no native roots. Escaped prompt backing
        // retains the charge even before the separate prompt-bind stage.
        reservation
            .validate_ledger(_backend.memory_ledger())
            .map_err(Error::PrefillControl)?;
        crate::backend::managed_memory::OrdinaryArrayAttachment::prepare(
            _backend.memory_ledger(),
            0,
        )?
        .attach(&tokens, reservation.clone())?;
    }
    let part = input::input_part(
        InputModality::Text,
        input::InputPayload::TokenIds(tokens),
        [],
        [],
    )?;
    let identity = match funding.and_then(|work| work.original_metadata_custody()) {
        Some(custody) => identity_plan.construct_original(custody),
        None => identity_plan.construct(),
    }
    .map_err(|error| Error::Other(Box::new(error)))?;
    if let Some(funding) = funding {
        funding.retain_metadata(eredu_runtime::SharedHostMetadata::Input(identity.clone()));
    } else if let Some(memory) = &memory {
        memory.retain_metadata(&eredu_runtime::SharedHostMetadata::Input(identity.clone()))?;
    } else if let Some(reservation) = request.map(|request| request.memory_reservation()) {
        // Legacy whole-request reservations remain authoritative independently
        // of native input aliases. Use a separate namespace from exact physical
        // source registration: the two kinds of custody are not interchangeable.
        identity
            .try_attach(&eredu_core::SharedStorageAccountingId::default(), || {
                Ok::<Box<dyn Send + Sync>, std::convert::Infallible>(Box::new(reservation.clone()))
            })
            .map_err(|error| Error::Other(Box::new(error)))?;
    }
    // Move the sole eager handle into its final part container. The public
    // borrowed ModelInput conversion remains unchanged for ordinary callers.
    let mut prompt = MlxModelInput {
        parts: pending_prompt::ModelInputParts::Owned(vec![part]),
        controlled_attribution: None,
        original_media: None,
        placement_semantics: None,
        cache_identity: Some(identity),
        prefill_chunk_positions: None,
        inference_request: None,
        memory_owner: None,
        quote: None,
    };
    if let Some(memory) = memory {
        prompt = prompt.with_memory_owner(memory);
    }
    if let Some(request) = request {
        prompt = prompt.with_inference_request(request.clone());
    }
    Ok(prompt)
}

// The concrete one-part owner and worker transports. Token backing and native
// descriptor/handle allocations come from OriginalPromptInputFacts; semantic
// identity backing is already included by the neutral prompt report.
fn text_prompt_input_control_bytes() -> Option<usize> {
    use std::{alloc::Layout, mem::size_of};
    let parts = Layout::array::<input::InputPart>(1).ok()?.size();
    [
        size_of::<input::InputPart>(),
        size_of::<Vec<input::InputPart>>(),
        size_of::<MlxModelInput>(),
        size_of::<Result<MlxModelInput, Error>>(),
        size_of::<Array>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<safemlx::error::Exception>(),
        size_of::<&[u32]>(),
        size_of::<i32>(),
        size_of::<bool>(),
    ]
    .into_iter()
    .try_fold(parts, usize::checked_add)
}

impl eredu_runtime::working_memory::OriginalChatBackend for MlxBackend<'_> {
    fn compile_original_capture_declaration(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::capture::CapturePlan,
        request: eredu_core::capture::CaptureRequestShape,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<
        eredu_runtime::working_memory::OriginalCaptureSource,
        eredu_runtime::working_memory::OriginalCaptureSourceError,
    > {
        use eredu_runtime::working_memory::{
            OriginalCaptureSourceError as Failure, WorkingMemoryError,
        };
        let controls = std::mem::size_of::<Result<(), Error>>()
            .checked_add(
                eredu_core::BackendFailure::source_retention_peak_bytes::<Error>()
                    .ok_or_else(|| Failure::rejected(WorkingMemoryError::Overflow))?,
            )
            .ok_or_else(|| Failure::rejected(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(controls)
            .map_err(|error| Failure::backend(error.into_backend_failure(), funding))?;
        let fail = |error: Error| Failure::backend(error.into_backend_failure(), funding);
        let session = runtime.session();
        session.validate_backend(runtime.backend()).map_err(fail)?;
        let opening = session
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .map_err(fail)?;
        let origin = eredu_core::capture::CaptureTextOrigin {
            cached_positions: text_quote::actual_frontier(
                session.payload.model.erased(),
                &opening,
                funding,
            )
            .map_err(fail)?,
        };
        let partition = session.original_partition_capture(funding).map_err(fail)?;
        let (catalog, support) = if let Some(partition) = &partition {
            (&partition.discovery.catalog, &partition.discovery.support)
        } else {
            session
                .capture_discovery
                .as_ref()
                .ok_or_else(|| Failure::rejected(WorkingMemoryError::UnknownBound))?
                .capture_parts()
        };
        runtime
            .backend()
            .memory_ledger()
            .compile_capture_declaration(plan, catalog, support, request, origin, funding)
    }

    fn compile_original_intervention_declaration(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::intervention::InterventionPlan,
        capture: &eredu_core::capture::SharedCapturePlan,
        session_id: &str,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<eredu_runtime::working_memory::OriginalInterventionSource, eredu_core::BackendFailure>
    {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(Error::into_backend_failure)?;
        let declaration = runtime
            .session()
            .original_intervention_declaration(funding)
            .map_err(Error::into_backend_failure)?
            .ok_or_else(|| eredu_core::TokenInputRejection::Unsupported.into_backend_failure())?;
        declaration
            .compile_source(
                plan,
                capture.admission(),
                session_id,
                runtime.backend().memory_ledger(),
            )
            .map_err(Error::into_backend_failure)
    }

    fn compile_original_forbidden_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
    ) -> Result<
        eredu_runtime::working_memory::OriginalForbiddenSource,
        eredu_runtime::working_memory::OriginalForbiddenSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| {
                eredu_runtime::working_memory::OriginalForbiddenSourceError::rejected(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?;
        runtime
            .backend()
            .memory_ledger()
            .compile_forbidden_source(plan)
    }

    fn compile_original_forbidden_tokenizer_source(
        runtime: &ModelRuntime<Self>,
        tokenizer: &eredu_runtime::working_memory::OriginalTokenizer,
        trigger: &[u8],
    ) -> Result<
        eredu_runtime::working_memory::OriginalForbiddenSource,
        eredu_runtime::working_memory::OriginalForbiddenSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| {
                eredu_runtime::working_memory::OriginalForbiddenSourceError::rejected(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?;
        runtime
            .backend()
            .memory_ledger()
            .compile_forbidden_tokenizer_source(tokenizer, trigger)
    }

    fn prepare_original_chat_profile(
        runtime: &ModelRuntime<Self>,
        template: &eredu_runtime::working_memory::OriginalChatTemplate,
        tokenizer: &eredu_runtime::working_memory::OriginalTokenizer,
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<
        eredu_runtime::working_memory::OriginalChatProfilePreparation,
        eredu_runtime::working_memory::OriginalChatProfileError,
    > {
        <Self as eredu_runtime::working_memory::OriginalChatBackend>::validate_original_chat_sources(runtime, template, tokenizer)?;
        eredu_runtime::working_memory::OriginalChatProfilePreparation::new(
            template,
            tokenizer,
            runtime
                .session()
                .payload
                .model
                .erased()
                .inference_execution_identity(),
            limits.resolve(runtime.backend().memory_ledger().topology())?,
        )
    }

    fn validate_original_chat_sources(
        runtime: &ModelRuntime<Self>,
        template: &eredu_runtime::working_memory::OriginalChatTemplate,
        tokenizer: &eredu_runtime::working_memory::OriginalTokenizer,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        let pool = runtime.backend().memory_ledger();
        template
            .validate_pool(pool)
            .and_then(|()| tokenizer.validate_pool(pool))
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)
    }
    fn validate_original_chat_render(
        runtime: &ModelRuntime<Self>,
        render: &eredu_runtime::working_memory::OriginalRenderedChat,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        render
            .validate_pool(runtime.backend().memory_ledger())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)
    }
    fn compile_original_chat_template(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::chat_storage::ChatTemplatePlan<'_>,
    ) -> Result<
        eredu_runtime::working_memory::OriginalChatTemplate,
        eredu_runtime::working_memory::OriginalChatSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_chat_template(plan)
            .map_err(Into::into)
    }
    fn compile_original_chat_template_file(
        runtime: &ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
        model_id: &str,
        has_tools: bool,
    ) -> Result<
        eredu_runtime::working_memory::OriginalChatTemplate,
        eredu_runtime::working_memory::OriginalChatSourceError,
    > {
        runtime
            .session()
            .validate_backend(runtime.backend())
            .map_err(|_| eredu_core::TokenInputRejection::IdentityMismatch)?;
        runtime
            .backend()
            .memory_ledger()
            .compile_chat_template_file(read, model_id, has_tools)
            .map_err(Into::into)
    }
    fn render_original_chat(
        runtime: &ModelRuntime<Self>,
        template: &eredu_runtime::working_memory::OriginalChatTemplate,
        tokenizer: &eredu_runtime::working_memory::OriginalTokenizer,
        context: eredu_text::chat_storage::ChatRenderContext<'_>,
    ) -> Result<
        eredu_runtime::working_memory::OriginalRenderedChat,
        eredu_runtime::working_memory::OriginalChatRenderOperationError,
    > {
        <Self as eredu_runtime::working_memory::OriginalChatBackend>::validate_original_chat_sources(runtime,template,tokenizer)?;
        runtime
            .backend()
            .memory_ledger()
            .render_original_chat(template, tokenizer, context)
            .map_err(Into::into)
    }
}

mod preparation_control;
