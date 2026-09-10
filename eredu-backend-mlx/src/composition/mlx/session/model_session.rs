use super::*;
use eredu_core::{BackendFailure, BackendFailureKind};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use super::recovery::{Probe, Recovery, Retention, Status};
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};

pub(super) struct SessionPayload {
    model: Executable,
    target: crate::backend::MlxPreparedTarget,
    distributed: Option<MlxDistributedSession>,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
    #[cfg(test)]
    _retirement_probe: Option<Box<dyn std::any::Any>>,
}

/// One submission owns both the entire executable and its neutral authority.
/// Scope tickets keep this owner alive even if the public session is dropped.
pub(super) struct SubmissionResources {
    payload: RefCell<Option<Rc<OrdinaryRetirement<SessionPayload>>>>,
    lease: RefCell<Option<SubmissionLease>>,
    poison: Rc<Cell<bool>>,
    scopes: Cell<usize>,
    release_requested: Cell<bool>,
}

impl SubmissionResources {
    pub(super) fn new(lease: SubmissionLease, poison: Rc<Cell<bool>>) -> Rc<Self> {
        Rc::new(Self {
            payload: RefCell::new(None),
            lease: RefCell::new(Some(lease)),
            poison,
            scopes: Cell::new(0),
            release_requested: Cell::new(false),
        })
    }

    pub(super) fn recovery(self: &Rc<Self>) -> Result<Recovery<ScopeRetention>, Error> {
        Recovery::begin(self.ticket()).map_err(Into::into)
    }

    pub(super) fn observation_recovery(
        self: &Rc<Self>,
        roots: Vec<Array>,
    ) -> Result<Recovery<ObservationRetention>, Error> {
        Recovery::begin(ObservationRetention {
            _roots: roots,
            ticket: self.ticket(),
        })
        .map_err(Into::into)
    }

    pub(super) fn poison_on_unwind(&self) -> ObservationUnwind<'_> {
        ObservationUnwind(self)
    }

    pub(super) fn ticket(self: &Rc<Self>) -> ScopeRetention {
        self.scopes.set(self.scopes.get() + 1);
        ScopeRetention(Rc::clone(self))
    }

    pub(super) fn resources_releasable(&self) -> bool {
        self.scopes.get() == 0
    }

    pub(super) fn request_release(&self) {
        self.release_requested.set(true);
        self.release_if_settled();
    }

    fn release_if_settled(&self) {
        if self.release_requested.get() && self.scopes.get() == 0 {
            // A live old completion must not keep Rc::get_mut unavailable after
            // its authority is released for a newer submission.
            self.payload.borrow_mut().take();
            self.lease.borrow_mut().take();
        }
    }

    pub(super) fn reject_unresolved(&self) {
        self.poison.set(true);
    }

    pub(super) fn ensure_healthy(&self) -> Result<(), Error> {
        if self.poison.get() {
            Err(Error::ArchitectureModel(
                "native session is poisoned by unresolved or failed work".into(),
            ))
        } else {
            Ok(())
        }
    }
}

struct ReleaseSubmission(Rc<SubmissionResources>);
impl Drop for ReleaseSubmission {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.reject_unresolved();
        }
        self.0.request_release();
    }
}

pub(super) struct ScopeRetention(Rc<SubmissionResources>);

pub(super) struct ObservationRetention {
    _roots: Vec<Array>,
    ticket: ScopeRetention,
}

impl Retention for ObservationRetention {
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
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.0.poison.set(true);
        }
    }
}

impl Drop for ScopeRetention {
    fn drop(&mut self) {
        self.0.scopes.set(self.0.scopes.get() - 1);
        self.0.release_if_settled();
    }
}

pub(super) struct ResourceOperation<P: Probe = safemlx::SubmissionScope> {
    owner: Rc<SubmissionResources>,
    recovery: Option<Recovery<ScopeRetention, P>>,
}

impl ResourceOperation {
    pub(super) fn begin(owner: &Rc<SubmissionResources>) -> Result<Self, Error> {
        owner.ensure_healthy()?;
        Ok(Self {
            owner: Rc::clone(owner),
            recovery: Some(owner.recovery()?),
        })
    }
}

impl<P: Probe> ResourceOperation<P> {
    #[cfg(test)]
    pub(super) fn with_probe(owner: &Rc<SubmissionResources>, probe: P) -> Self {
        Self {
            owner: Rc::clone(owner),
            recovery: Some(Recovery::with_probe(owner.ticket(), probe)),
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
            recovery.seal();
            let status = recovery.progress();
            if !status.settled || status.failed || status.blocked || std::thread::panicking() {
                self.owner.reject_unresolved();
            }
        }
    }
}

struct SessionOperation<'a, P: Probe = safemlx::SubmissionScope> {
    session: &'a mut MlxModelSession,
    owner: Rc<SubmissionResources>,
    recovery: Option<Recovery<ScopeRetention, P>>,
    handed_off: bool,
}

pub(super) fn restored_error_permits_recovery(status: Status, error: &Error) -> bool {
    !status.failed && !status.blocked && error.model_state_preserved()
}

pub(super) fn complete_model_operation<T, P: Probe>(
    value: T,
    owner: Rc<SubmissionResources>,
    recovery: Recovery<ScopeRetention, P>,
) -> Result<T, Error> {
    owner.request_release();
    let status = recovery.finish();
    if !status.settled || status.failed || status.blocked {
        owner.reject_unresolved();
        return Err(Error::ArchitectureModel(
            "native operation failed or returned without proven completion; session is poisoned"
                .into(),
        ));
    }
    Ok(value)
}

impl<P: Probe> SessionOperation<'_, P> {
    fn model(&mut self) -> &mut Executable {
        &mut Rc::get_mut(&mut self.session.payload)
            .expect("idle session has exclusive native payload ownership")
            .model
    }

    fn finish<T>(
        self,
        result: Result<T, Error>,
    ) -> Result<(T, Rc<SubmissionResources>, Recovery<ScopeRetention, P>), Error> {
        self.finish_with_preservation(result, false)
    }

    fn finish_execution<T>(
        self,
        result: Result<T, Error>,
    ) -> Result<(T, Rc<SubmissionResources>, Recovery<ScopeRetention, P>), Error> {
        self.finish_with_preservation(result, true)
    }

    fn finish_with_preservation<T>(
        mut self,
        result: Result<T, Error>,
        allow_preservation: bool,
    ) -> Result<(T, Rc<SubmissionResources>, Recovery<ScopeRetention, P>), Error> {
        self.owner
            .payload
            .replace(Some(Rc::clone(&self.session.payload)));
        let recovery = self.recovery.as_mut().expect("live operation scope");
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            if let Err(error) = &result {
                self.session.record_failure(error);
            }
            return Err(Error::ArchitectureModel(
                "native session execution failed or became unobservable; session is poisoned"
                    .into(),
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
                self.session.record_failure(&error);
                return Err(error);
            }
            Ok(value) => value,
        };
        self.handed_off = true;
        Ok((value, Rc::clone(&self.owner), self.recovery.take().unwrap()))
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
            .replace(Some(Rc::clone(&self.session.payload)));
        if let Some(recovery) = self.recovery.as_mut() {
            recovery.seal();
            let status = recovery.progress();
            if !status.settled || status.failed || status.blocked || std::thread::panicking() {
                self.owner.reject_unresolved();
            }
        }
        self.owner.request_release();
        // The preallocated node either releases the scope ticket or retains it
        // together with the executable and lease in nonblocking quarantine.
        self.recovery.take();
    }
}

/// MLX-owned prefill input.
///
/// Arrays are cloned handles, not copied tensor storage. Owning the handles
/// makes submission independent of the caller's temporary `ModelInput` view.
#[derive(Debug, Clone)]
pub struct MlxModelInput {
    parts: Vec<input::InputPart>,
    cache_identity: Option<eredu_runtime::PreparedInputCacheIdentity>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: input.parts.to_vec(),
            cache_identity: input.cache_identity().cloned(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        let mut owned = input.with_model_input(|borrowed| Self::from(borrowed));
        owned.cache_identity = input.cache_identity().cloned();
        owned
    }

    /// Returns the exact semantic identity carried by processor-produced input.
    pub const fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Couples manually prepared tensors to caller-owned semantic content.
    pub fn with_semantic_content_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Result<Self, Error> {
        let prepared = eredu_runtime::PreparedModelInput::new(self.parts.clone(), |array| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, array)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.cache_identity = Some(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Ok(self)
    }

    /// Borrows the owned input parts as a model-input view for one operation.
    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        let input = match self.cache_identity.as_ref() {
            Some(identity) => input::ModelInput::with_cache_identity(&self.parts, identity),
            None => input::ModelInput::new(&self.parts),
        };
        execute(input)
    }
}

/// Opaque independently writable model state for one exact loaded executable.
/// This covers native persistent state only; complete generation snapshots also
/// compose sampling, pending input, facade semantics and capture admissions.
pub struct MlxNativeTextState {
    state: Box<dyn std::any::Any>,
}

impl eredu_core::execution_control::NativeTextStateBackend for MlxBackend<'_> {
    type NativeTextState = MlxNativeTextState;

    fn native_text_state_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport {
        use eredu_core::execution_control::ControlSupport;
        let session = runtime.session();
        if let Err(error) = session.validate_backend(runtime.backend()) {
            return ControlSupport::Unsupported {
                reason: error.to_string(),
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
        session
            .payload
            .model
            .erased()
            .estimate_native_control_state(saved.map(|saved| saved.state.as_ref()))
    }

    fn capture_native_text_state(
        runtime: &mut ModelRuntime<Self>,
    ) -> Result<MlxNativeTextState, Error> {
        let estimate = Self::estimate_native_text_state(runtime, None)?;
        if estimate.is_none() {
            return Err(Error::ArchitectureModel(
                "native text snapshot estimate is unavailable".into(),
            ));
        }
        // The inner Result is deliberate: the copy never changes installed
        // state. Settle its native work before returning a recoverable copy
        // error; a failed/unobservable scope still fences the shared engine.
        runtime
            .session_mut()
            .with_model_operation(|model| Ok(model.erased_mut().capture_native_control_state()))?
            .map(|state| MlxNativeTextState { state })
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

    fn copy_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &MlxNativeTextState,
    ) -> Result<MlxNativeTextState, Error> {
        Self::validate_native_text_state(runtime, saved)?;
        if Self::estimate_native_text_state(runtime, Some(saved))?.is_none() {
            return Err(Error::ArchitectureModel(
                "native text snapshot estimate is unavailable".into(),
            ));
        }
        runtime
            .session_mut()
            .with_model_operation(|model| {
                Ok(model
                    .erased_mut()
                    .copy_native_control_state(saved.state.as_ref()))
            })?
            .map(|state| MlxNativeTextState { state })
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
            return Err(Error::ArchitectureModel(reason));
        }
        session
            .payload
            .model
            .erased()
            .validate_native_control_state(saved.state.as_ref())
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
        let payload = Rc::get_mut(&mut session.payload)
            .ok_or_else(|| Error::ArchitectureModel("native payload is still retained".into()))?;
        payload
            .model
            .erased_mut()
            .exchange_native_control_state(slot.state.as_mut())
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession {
    payload: Rc<OrdinaryRetirement<SessionPayload>>,
    poison: Rc<Cell<bool>>,
    failure: RefCell<Option<String>>,
    authority: RefCell<SessionAuthority>,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    capabilities: eredu_core::SessionCapabilities,
    capture_discovery: Option<eredu_architectures::prepared_sources::PreparedModelDiscovery>,
    intervention_session_identity: String,
    state_residency: CacheResidencyPolicy,
}

impl MlxModelSession {
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
        let recovery = Recovery::with_probe(owner.ticket(), probe);
        SessionOperation {
            session: self,
            owner,
            recovery: Some(recovery),
            handed_off: false,
        }
        .finish_with_preservation::<()>(Err(error), allow_preservation)
        .map(|_| ())
    }

    #[cfg(test)]
    pub(super) fn test_payload_weak(&self) -> std::rc::Weak<OrdinaryRetirement<SessionPayload>> {
        Rc::downgrade(&self.payload)
    }
    #[cfg(test)]
    pub(crate) fn test_payload_retirement_probe(&self) -> impl Fn() -> bool + 'static {
        let payload = Rc::downgrade(&self.payload);
        move || payload.upgrade().is_none()
    }
    #[cfg(test)]
    pub(super) fn set_retirement_probe(&mut self, probe: Box<dyn std::any::Any>) {
        Rc::get_mut(&mut self.payload).unwrap()._retirement_probe = Some(probe);
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
        let state_residency = model.state_residency().clone();
        #[cfg(any(feature = "image", feature = "audio"))]
        let processor = model.take_processor();
        let distributed = model.take_distributed();
        let capture_discovery = model.take_capture_discovery();
        let intervention_session_identity =
            eredu_core::intervention::new_intervention_session_identity();
        let (executable, target) = model.into_execution_parts();
        #[cfg(test)]
        crate::tests::support::path_instrumentation::session_reset_attempt();
        let mut session = Self {
            // The final Rc may be released by a native recovery reaper while
            // it holds the runtime lock. Stage the complete semantic owner;
            // resident-manager leases and observers drop only at an ordinary
            // unlocked host boundary.
            payload: Rc::new(OrdinaryRetirement::new(SessionPayload {
                model: executable,
                target,
                distributed,
                #[cfg(any(feature = "image", feature = "audio"))]
                processor,
                #[cfg(test)]
                _retirement_probe: None,
            })),
            poison: Rc::new(Cell::new(false)),
            failure: RefCell::new(None),
            authority: RefCell::new(SessionAuthority::new()),
            floating_state_dtype_bytes,
            capabilities: realized_capabilities,
            capture_discovery,
            intervention_session_identity,
            state_residency,
        };
        session.reset()?;
        Ok(session)
    }

    pub(in crate::composition::mlx) const fn floating_state_dtype_bytes(
        &self,
    ) -> std::num::NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    fn begin_submission(
        &mut self,
        backend: &MlxBackend<'_>,
    ) -> Result<SessionOperation<'_>, Error> {
        self.validate_backend(backend)?;
        self.begin_operation()
    }

    fn begin_operation(&mut self) -> Result<SessionOperation<'_>, Error> {
        self.ensure_no_submission_in_flight()?;
        let lease = self
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        let recovery = owner.recovery()?;
        Ok(SessionOperation {
            session: self,
            owner,
            recovery: Some(recovery),
            handed_off: false,
        })
    }

    pub(in crate::composition::mlx) fn with_model_operation<T>(
        &mut self,
        operation: impl FnOnce(&mut Executable) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut guard = self.begin_operation()?;
        let result = operation(guard.model());
        let (value, owner, recovery) = guard.finish(result)?;
        complete_model_operation(value, owner, recovery)
    }

    fn with_shared_operation<T>(
        &self,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.ensure_no_submission_in_flight()?;
        let lease = self
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        owner.payload.replace(Some(Rc::clone(&self.payload)));
        let mut recovery = owner.recovery()?;
        let _release = ReleaseSubmission(Rc::clone(&owner));
        let result = operation();
        if let Err(error) = &result {
            self.record_failure(error);
        }
        recovery.seal();
        // Successful host reads can leave completion bookkeeping pending. Wait
        // for that work before reusing the session; errors remain nonblocking.
        let status = if result.is_ok() {
            recovery.finish()
        } else {
            recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            owner.reject_unresolved();
            return Err(Error::ArchitectureModel(
                "native observation or sampling failed or remains unresolved".into(),
            ));
        }
        if result.is_err() {
            owner.reject_unresolved();
        }
        result
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
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
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

    /// Installs causal observers on this session's selected embedded-prediction executor.
    pub fn install_embedded_prediction_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: RuntimeActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: RuntimeActivationObserver<Array, Exception> + 'static,
    {
        self.ensure_no_submission_in_flight()?;
        if !self.payload.model.erased().has_embedded_prediction() {
            return Err(Error::ArchitectureModel(
                "session has no selected embedded-prediction executor".into(),
            ));
        }
        let observers =
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                tensors, logits,
            );
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

    pub(in crate::composition::mlx) fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        self.payload.model.prepared_input_part_plan(input)
    }

    #[cfg(test)]
    pub(in crate::composition::mlx) fn speculative_model_mut(
        &mut self,
    ) -> Result<&mut Executable, Error> {
        self.ensure_no_submission_in_flight()?;
        Ok(&mut Rc::get_mut(&mut self.payload).expect("idle payload").model)
    }

    #[cfg(test)]
    pub(crate) fn neutral_prediction_target_mut(
        &mut self,
    ) -> Result<&mut dyn super::super::replicated_text::ErasedReplicatedTextExecutable, Error> {
        self.ensure_no_submission_in_flight()?;
        Ok(Rc::get_mut(&mut self.payload)
            .expect("idle payload")
            .model
            .erased_mut())
    }

    /// Clears all MLX cache state under the authoritative selected policy.
    pub fn reset(&mut self) -> Result<(), Error> {
        let policy = self.state_residency.clone();
        self.with_model_operation(|model| {
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
            Array::from(token_ids.as_slice())
                .try_index_device(NewAxis, backend.stream())
                .map_err(Into::into)
        })
    }

    fn submit_decode_input(
        &mut self,
        backend: &MlxBackend<'_>,
        input: impl FnOnce() -> Result<Array, Error>,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        let mut operation = self.begin_submission(backend)?;
        let token_validation_scope = TokenValidationScope::begin()?;
        let output = input()
            .map_err(Error::before_model_mutation)
            .and_then(|input| decode_model(operation.model(), &input, backend.stream()));
        let public_output = operation.model().erased().partition_public_output();
        let (output, owner, recovery) = operation.finish_execution(output)?;
        Ok(owned_model_submission(
            output,
            token_validation_scope.finish(),
            public_output,
            owner,
            recovery,
        ))
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
    ) -> Result<PromptCacheManifest, Error> {
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        self.with_model_operation(|model| {
            if model.has_neutral_partitioned_control() {
                model
                    .save_prompt_cache_distributed(root, descriptor, prefix_token_ids, options)?
                    .ok_or_else(|| {
                        Error::ArchitectureModel(
                            "this partition rank owns no prompt-cache state".into(),
                        )
                    })
            } else {
                model
                    .save_prompt_cache(
                        root,
                        descriptor,
                        prefix_token_ids,
                        options,
                        backend.stream(),
                    )
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
    ) -> Result<PromptCacheManifest, Error> {
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let options = options.clone();
        self.with_model_operation(|model| {
            let manifest = if model.has_neutral_partitioned_control() {
                model
                    .load_prompt_cache_distributed(root, expected, prefix_token_ids)?
                    .ok_or_else(|| {
                        Error::ArchitectureModel(
                            "this partition rank owns no prompt-cache state".into(),
                        )
                    })?
            } else {
                model.load_prompt_cache(
                    root,
                    expected,
                    prefix_token_ids,
                    options,
                    backend.stream(),
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
    ) -> Result<PromptCacheManifest, Error> {
        self.validate_backend(backend)?;
        self.ensure_no_submission_in_flight()?;
        let identity = input.cache_identity.clone().ok_or_else(|| {
            Error::ArchitectureModel(
                "prompt-cache loading requires prepared-input semantic identity".into(),
            )
        })?;
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let root = root.as_ref();
        let options = options.clone();
        self.with_model_operation(|model| {
            if model.has_neutral_partitioned_control() {
                model
                    .load_prompt_cache_for_input_distributed(
                        root,
                        expected,
                        prefix_token_ids,
                        identity,
                    )?
                    .ok_or_else(|| {
                        Error::ArchitectureModel(
                            "this partition rank owns no prompt-cache state".into(),
                        )
                    })
            } else {
                model
                    .load_prompt_cache_for_input(
                        root,
                        expected,
                        prefix_token_ids,
                        identity,
                        options.clone(),
                        backend.stream(),
                    )
                    .map_err(Into::into)
            }
        })
    }

    /// Returns communication when this is a distributed session.
    pub fn distributed(&self) -> Option<&MlxDistributedSession> {
        self.payload.distributed.as_ref()
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
        self.with_shared_operation(|| {
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
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let mut operation = self.begin_submission(backend)?;
        let token_validation_scope = TokenValidationScope::begin()?;
        let output = input.with_borrowed(|input| {
            operation.model().erased_mut().prefill_with_observer(
                input,
                None,
                backend.stream(),
                &mut ArrayObserverAdapter { inner: observer },
            )
        });
        let output = output.and_then(|output| {
            observer.finish()?;
            Ok(output)
        });
        let (output, owner, recovery) = operation.finish_execution(output)?;
        Ok(model_array_submission(
            output,
            token_validation_scope.finish(),
            owner,
            recovery,
        ))
    }

    fn submit_decode_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Array,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let mut operation = self.begin_submission(backend)?;
        let token_validation_scope = TokenValidationScope::begin()?;
        let output = operation.model().erased_mut().decode_with_observer(
            &input,
            backend.stream(),
            &mut ArrayObserverAdapter { inner: observer },
        );
        let output = output.and_then(|output| {
            observer.finish()?;
            Ok(output)
        });
        let (output, owner, recovery) = operation.finish_execution(output)?;
        Ok(model_array_submission(
            output,
            token_validation_scope.finish(),
            owner,
            recovery,
        ))
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
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        let mut operation = self.begin_submission(backend)?;
        let token_validation_scope = TokenValidationScope::begin()?;
        let output =
            input.with_borrowed(|input| prefill_model(operation.model(), input, backend.stream()));
        let public_output = operation.model().erased().partition_public_output();
        let (output, owner, recovery) = operation.finish_execution(output)?;
        Ok(owned_model_submission(
            output,
            token_validation_scope.finish(),
            public_output,
            owner,
            recovery,
        ))
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
        self.with_shared_operation(|| {
            let mut observations = ObservationSet::new();
            if let Some(logits) = output.logits() {
                observations
                    .insert(
                        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                        ObservationValue::Tensor(observe_tensor(logits, backend.stream())?),
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
        let observations =
            self.with_shared_operation(|| collector.materialize(backend.stream()))?;
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
        let submission = self.submit_decode_with_observer(backend, input, &mut collector)?;
        let logits = submission.wait()?;
        let output = if self.payload.model.erased().partition_public_output() {
            MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
        } else {
            MlxModelOutput::new(None)
        };
        let observations =
            self.with_shared_operation(|| collector.materialize(backend.stream()))?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }
}

impl<'a> TextGenerationBackend for MlxBackend<'a> {
    fn reset_session(backend: &Self, session: &mut Self::Session) -> Result<(), BackendFailure> {
        session
            .validate_backend(backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        session
            .reset()
            .map_err(|error| session.lifecycle_failure(error))
    }

    fn synchronize_session(backend: &Self, session: &Self::Session) -> Result<(), BackendFailure> {
        session
            .validate_backend(backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
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
            .map_err(|error| session.lifecycle_failure(error))
    }

    fn text_sampling_control_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport {
        Self::text_execution_control_support(runtime)
    }
    type Prompt = MlxModelInput;
    type Token = MlxTextToken;
    type TextGenerationState = MlxTextGenerationState;
    type TextCompletion = MlxTextCompletion;

    fn text_execution_control_support(
        runtime: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport {
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
        let mut discovery = prepared.intervention(&super::intervention::mechanisms())?;
        discovery.session_identity = Some(session.intervention_session_identity.clone());
        Ok(discovery)
    }

    fn validate_text_interventions(
        runtime: &ModelRuntime<Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        use eredu_core::capture::*;
        Self::validate_text_capture(runtime, capture)?;
        if plan.is_empty() {
            return Ok(());
        }
        if runtime.session().payload.distributed.is_some() {
            return Err(CaptureError::Unsupported(
                "interventions require ordinary single-rank text generation".into(),
            ));
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
        Self::validate_text_interventions(runtime, &capture, &plan)?;
        eredu_runtime::intervention::install_session(
            &mut state.capture,
            capture,
            Some((
                plan,
                std::sync::Arc::new(super::intervention::NativeInterventionEstimator),
            )),
        )
    }

    fn capture_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
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
        Self::validate_text_capture(runtime, &plan)?;
        eredu_runtime::intervention::install_session(&mut state.capture, plan, None)
    }

    fn validate_text_capture(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        use eredu_core::capture::*;
        if plan.is_empty() {
            return Ok(());
        }
        if runtime.session().payload.distributed.is_some() || plan.request().batch != 1 {
            return Err(CaptureError::Unsupported(
                "bounded capture requires ordinary single-rank, single-sequence text generation"
                    .into(),
            ));
        }
        eredu_runtime::capture::validate_session(
            plan,
            &Self::capture_discovery(runtime)?,
            super::bounded_capture::estimate_shape,
        )
    }

    fn take_text_capture(
        state: &mut Self::TextGenerationState,
    ) -> Option<eredu_core::capture::CapturedStep> {
        state
            .capture
            .as_mut()
            .and_then(eredu_runtime::capture::CaptureSession::take_step)
    }

    fn start_text_generation(
        _: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Error> {
        super::recovery::detached(Vec::new(), || {
            let sampling = config.sampling();
            let prng = if sampling.temperature == 0.0 {
                None
            } else {
                Some(RandomState::from_key(safemlx::random::key(config.seed())?))
            };
            let sampler = MlxTextSampler::from_config(config).map_err(|error| {
                eredu_core::BackendError::Execution {
                    session: "text-generation".into(),
                    operation: "configure Mirostat V2".into(),
                    message: error.to_string(),
                }
            })?;
            Ok(MlxTextGenerationState {
                sampling: super::generation::MlxTextSamplingState {
                    temperature: sampling.temperature,
                    prng,
                    sampler,
                    next_prediction: 0,
                },
                capture: None,
            })
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::ArchitectureModel(
                "text generation requires at least one prompt token".into(),
            ));
        }
        super::recovery::detached(Vec::new(), || {
            let tokens = Array::from(prompt_token_ids.as_slice())
                .try_index_device(NewAxis, backend.stream())?;
            let parts = [input::input_part(
                InputModality::Text,
                input::InputPayload::TokenIds(tokens),
                [],
                [],
            )?];
            MlxModelInput::from(input::ModelInput::new(&parts)).with_semantic_content_fingerprint(
                eredu_core::cache::prompt_cache_token_fingerprint(&prompt_token_ids),
            )
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
        let filter = decision.filter();
        let stream = runtime.backend().stream().clone();
        if let Some(capture) = &mut state.capture {
            if prompt.with_borrowed(|input| {
                input
                    .parts
                    .iter()
                    .any(|part| part.modality() != InputModality::Text)
            }) {
                return Err(Error::ArchitectureModel("capture/interventions require ordinary text input; media execution is unsupported".into()));
            }
            let request = capture.plan().request();
            let aligned = prompt.with_borrowed(|input| {
                if input.parts.len() != 1 {
                    return false;
                }
                let input::InputPayload::TokenIds(tokens) = input.parts[0].payload() else {
                    return false;
                };
                let shape = tokens.shape();
                shape.len() == 2
                    && u64::try_from(shape[0]) == Ok(request.batch)
                    && u64::try_from(shape[1]) == Ok(request.prompt_tokens)
            });
            if !aligned {
                return Err(Error::ArchitectureModel("capture/interventions require one token-ID prompt matching admitted batch and prompt length".into()));
            }
            capture
                .begin_step(eredu_core::capture::CapturePhase::Prefill, 0)
                .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
            let (backend, session) = runtime.parts_mut();
            let submission = session.submit_prefill_with_observer(
                backend,
                prompt,
                &mut super::bounded_capture::observer(capture, &stream, decision.capture_domain()),
            )?;
            let submission = Submission {
                output: MlxModelOutput::new(Some(MlxTensor::from_array(submission.output))),
                completion: submission.completion,
            };
            return sample_text_submission(runtime.session(), submission, filter, state, stream);
        }
        let submission = runtime.prefill(prompt)?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
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
        let filter = decision.filter();
        let stream = runtime.backend().stream().clone();
        if let Some(capture) = &mut state.capture {
            capture
                .begin_step(
                    eredu_core::capture::CapturePhase::Decode,
                    state.sampling.next_prediction,
                )
                .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
            let (backend, session) = runtime.parts_mut();
            // Input framing belongs to the existing session recovery boundary.
            let input = session.with_shared_operation(|| {
                token
                    .value
                    .try_index_device((.., NewAxis), &stream)
                    .map_err(Into::into)
            })?;
            let submission = session.submit_decode_with_observer(
                backend,
                input,
                &mut super::bounded_capture::observer(capture, &stream, decision.capture_domain()),
            )?;
            let submission = Submission {
                output: MlxModelOutput::new(Some(MlxTensor::from_array(submission.output))),
                completion: submission.completion,
            };
            return sample_text_submission(runtime.session(), submission, filter, state, stream);
        }
        let (backend, session) = runtime.parts_mut();
        let submission = session.submit_decode_input(backend, || {
            #[cfg(test)]
            crate::tests::support::path_instrumentation::session_input_creation_attempt();
            token
                .value
                .try_index_device((.., NewAxis), &stream)
                .map_err(Into::into)
        })?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
    }
}

fn model_array_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    owner: Rc<SubmissionResources>,
    recovery: Recovery<ScopeRetention>,
) -> Submission<Array, MlxSessionCompletion> {
    let mut retained = Vec::with_capacity(1 + token_validations.arrays().count());
    retained.push(output.clone());
    retained.extend(token_validations.arrays().cloned());
    Submission {
        output,
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model {
                token_validations,
                _retained: retained,
                owner,
                recovery: RefCell::new(Some(recovery)),
                observation_error: RefCell::new(None),
            },
        },
    }
}

fn owned_model_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    public_output: bool,
    owner: Rc<SubmissionResources>,
    recovery: Recovery<ScopeRetention>,
) -> Submission<MlxModelOutput, MlxSessionCompletion> {
    let submission = model_array_submission(output, token_validations, owner, recovery);
    Submission {
        output: MlxModelOutput::new(
            public_output.then(|| MlxTensor::from_array(submission.output)),
        ),
        completion: submission.completion,
    }
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
    owned_model_submission(output, token_validations, public_output, owner, recovery)
}
