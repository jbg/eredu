use super::*;
use std::{
    cell::{Cell, RefCell},
    error::Error as _,
    rc::Rc,
};

#[derive(Default)]
struct Audit {
    events: RefCell<Vec<&'static str>>,
    capabilities: Cell<SessionCapabilities>,
    busy: Cell<bool>,
    poisoned: Cell<bool>,
    fail_sync: Cell<bool>,
    change_during_preparation: Cell<bool>,
    foreign_claim_probe: Cell<bool>,
}
impl Audit {
    fn note(&self, event: &'static str) {
        self.events.borrow_mut().push(event);
    }
    fn take(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.events.borrow_mut())
    }
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("ordinary synchronization failed")]
    Synchronization,
    #[error("session is poisoned")]
    Poisoned,
}
fn busy() -> BackendFailure {
    BackendFailure::new(BackendFailureKind::Busy, crate::SessionResetRejection::Busy)
}
fn health(audit: &Audit) -> Result<(), BackendFailure> {
    if audit.poisoned.get() {
        return Err(BackendFailure::new(
            BackendFailureKind::InvalidSession,
            Failure::Poisoned,
        ));
    }
    if audit.busy.get() {
        return Err(busy());
    }
    Ok(())
}

struct ResetBackend<const MODE: u8>(Rc<Audit>);
struct ResetSession {
    audit: Rc<Audit>,
    state: u32,
}
impl<const MODE: u8> BackendProvider for ResetBackend<MODE> {
    type ModelConfig = u32;
    type Model = u32;
    type Session = ResetSession;
    type Error = Infallible;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("reset-readiness", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Infallible> {
        Ok(Vec::new())
    }
    fn prepare_model(&self, state: u32) -> Result<PreparedModel<u32>, Infallible> {
        Ok(PreparedModel::new(state, self.0.capabilities.get()))
    }
    fn create_session(&self, model: PreparedModel<u32>) -> Result<ResetSession, Infallible> {
        Ok(ResetSession {
            audit: Rc::clone(&self.0),
            state: model.into_inner(),
        })
    }
}
impl<const MODE: u8> BackendSession<ResetBackend<MODE>> for ResetSession {
    type PrefillInput = Vec<u32>;
    type DecodeInput = u32;
    type Output = u32;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        self.audit.capabilities.get()
    }
    fn prefill(
        &mut self,
        _: &ResetBackend<MODE>,
        input: Vec<u32>,
    ) -> Result<Submission<u32, Done>, Infallible> {
        self.state += input.into_iter().sum::<u32>();
        Ok(Submission {
            output: self.state,
            completion: Done,
        })
    }
    fn decode(
        &mut self,
        _: &ResetBackend<MODE>,
        input: u32,
    ) -> Result<Submission<u32, Done>, Infallible> {
        self.state += input;
        Ok(Submission {
            output: self.state,
            completion: Done,
        })
    }
    fn observe_output(
        &self,
        _: &ResetBackend<MODE>,
        _: &u32,
    ) -> Result<ObservationSet, Infallible> {
        Ok(ObservationSet::new())
    }
}
impl<const MODE: u8> TextGenerationBackend for ResetBackend<MODE> {
    type TextPreparation = ();
    type TextPreparationControl = ();
    type TextStepPermit = ();
    type Prompt = Vec<u32>;
    type Token = u32;
    type TextGenerationState = ();
    type TextCompletion = Done;
    fn begin_text_step<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &(),
        _: &(),
        _: &C,
        _: PendingTextInput<&Self::Prompt, &u32>,
        _: &TextStepContext,
    ) -> Result<(), Infallible> {
        Ok(())
    }
    fn finish_text_step(_: ()) -> Result<(), Infallible> {
        Ok(())
    }
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Self::Prompt>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn reset_session(
        backend: &Self,
        session: &mut ResetSession,
        claim: crate::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        backend.0.note("ordinary_reset");
        claim.validate_session(session).unwrap();
        claim
            .validate_capabilities(<ResetSession as BackendSession<Self>>::capabilities(
                session,
            ))
            .unwrap();
        assert_eq!(claim.limits(), &crate::SessionResetLimits::default());
        session.state = 0;
        Ok(())
    }
    fn reset_session_admitted(
        backend: &Self,
        session: &mut ResetSession,
        claim: crate::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        backend.0.note("direct");
        apply(session, claim)
    }
    fn synchronize_session(backend: &Self, _: &ResetSession) -> Result<(), BackendFailure> {
        backend.0.note("synchronize");
        if backend.0.fail_sync.get() {
            return Err(BackendFailure::new(
                BackendFailureKind::Other,
                Failure::Synchronization,
            ));
        }
        Ok(())
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), Infallible> {
        Ok(())
    }
    fn prepare_text_prompt(_: &Self, ids: Vec<u32>) -> Result<Vec<u32>, Infallible> {
        Ok(ids)
    }
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Vec<u32>,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<u32, Done>, Infallible> {
        runtime.prefill(prompt)
    }
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: u32,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<u32, Done>, Infallible> {
        runtime.decode(token)
    }
}

// This test provider's fixed reset demand is deliberately a core comparison
// fixture, not a runtime account or native allocation inventory.
const REQUIRED: u64 = 64;
fn apply(
    session: &mut ResetSession,
    claim: crate::SessionResetClaim<'_>,
) -> Result<(), BackendFailure> {
    health(&session.audit)?;
    claim
        .validate_session(session)
        .map_err(|e| BackendFailure::new(BackendFailureKind::InvalidSession, e))?;
    claim.validate_capabilities(session.audit.capabilities.get())?;
    if session.audit.foreign_claim_probe.get() {
        let foreign = ResetSession {
            audit: Rc::clone(&session.audit),
            state: session.state,
        };
        let error = claim.validate_session(&foreign).unwrap_err();
        return Err(BackendFailure::new(
            BackendFailureKind::InvalidSession,
            error,
        ));
    }
    session.audit.note("compare");
    let topology = crate::MemoryTopology::new(vec![crate::MemoryDomainDescription {
        name: "host".into(),
        locations: vec![crate::MemoryLocation::Host],
    }])
    .unwrap();
    let mut requirements = crate::DomainMemoryRequirements::zero(&topology);
    requirements
        .add_allocation(
            REQUIRED,
            &crate::MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap(),
        )
        .unwrap();
    let accepted = claim
        .compare(&topology, requirements)
        .map_err(|e| BackendFailure::new(BackendFailureKind::Other, e))?;
    assert!(
        accepted
            .requirements()
            .get(topology.host_domain())
            .unwrap()
            .total()
            .unwrap()
            >= REQUIRED
    );
    session.audit.note("publish");
    session.state = 0;
    Ok(())
}

struct Ready<'s>(&'s mut ResetSession);
impl SessionResetReadiness for Ready<'_> {
    fn capabilities(&self) -> SessionCapabilities {
        self.0.audit.capabilities.get()
    }
}
impl Drop for Ready<'_> {
    fn drop(&mut self) {
        self.0.audit.note("drop_ready");
    }
}
fn prepare(session: &mut ResetSession) -> Result<Ready<'_>, BackendFailure> {
    session.audit.note("prepare");
    health(&session.audit)?;
    if session.audit.change_during_preparation.get() {
        session.audit.capabilities.set(changed());
    }
    Ok(Ready(session))
}
impl SessionResetPreparationBackend for ResetBackend<0> {
    type Readiness<'s>
        = Ready<'s>
    where
        Self: 's;
}
impl SessionResetPreparationBackend for ResetBackend<1> {
    type Readiness<'s>
        = Ready<'s>
    where
        Self: 's;
    fn prepare_session_reset_ordinary<'s>(
        _: &'s Self,
        session: &'s mut ResetSession,
    ) -> Result<Ready<'s>, BackendFailure>
    where
        Self: 's,
    {
        prepare(session)
    }
    fn reset_ready_session_admitted<'s>(
        backend: &'s Self,
        ready: Ready<'s>,
        claim: crate::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure>
    where
        Self: 's,
    {
        backend.0.note("ready_reset");
        apply(&mut *ready.0, claim)
    }
}
impl SessionResetPreparationBackend for ResetBackend<2> {
    type Readiness<'s>
        = Ready<'s>
    where
        Self: 's;
    fn prepare_session_reset_ordinary<'s>(
        _: &'s Self,
        session: &'s mut ResetSession,
    ) -> Result<Ready<'s>, BackendFailure>
    where
        Self: 's,
    {
        prepare(session)
    }
}
fn fixture<const MODE: u8>() -> (Rc<Audit>, ModelRuntime<ResetBackend<MODE>>) {
    let audit = Rc::new(Audit::default());
    let runtime = ModelRuntime::prepare(ResetBackend(Rc::clone(&audit)), 42).unwrap();
    (audit, runtime)
}
fn limits(bytes: u64) -> crate::SessionResetLimits {
    crate::SessionResetLimits {
        memory_limits: crate::MemoryLimitDeclarations::new([(
            "host".into(),
            crate::MemoryLimit::Finite(bytes),
        )]),
        additional_headroom: Default::default(),
    }
}
fn changed() -> SessionCapabilities {
    SessionCapabilities::default().with_activation_inspection(true)
}

#[test]
fn admitted_entry_does_not_invoke_failing_ordinary_synchronization() {
    let (audit, mut runtime) = fixture::<1>();
    audit.fail_sync.set(true);
    runtime.reset_admitted(limits(REQUIRED)).unwrap();
    assert_eq!(audit.take(), ["direct", "compare", "publish"]);
    assert_eq!(runtime.session().state, 0);
    runtime.decode(9).unwrap();
    let error = runtime.reset().unwrap_err();
    assert!(matches!(
        error.source().unwrap().downcast_ref::<Failure>(),
        Some(Failure::Synchronization)
    ));
    assert_eq!(audit.take(), ["synchronize"]);
    assert_eq!(runtime.session().state, 9);
    audit.fail_sync.set(false);
    runtime.reset().unwrap();
    assert_eq!(audit.take(), ["synchronize", "ordinary_reset"]);
    runtime.synchronize().unwrap();
    assert_eq!(audit.take(), ["synchronize"]);
}

#[test]
fn unsupported_preparation_and_ready_hook_preserve_state_without_fallback() {
    let (audit, mut runtime) = fixture::<0>();
    let error = match runtime.prepare_reset_ordinary() {
        Err(error) => error,
        Ok(_) => panic!("default preparation must reject"),
    };
    assert_eq!(error.kind(), BackendFailureKind::Unsupported);
    assert!(audit.take().is_empty());
    assert_eq!(runtime.session().state, 42);

    let (audit, mut runtime) = fixture::<2>();
    let ready = runtime.prepare_reset_ordinary().unwrap();
    let error = ready.reset_admitted(limits(REQUIRED)).unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Unsupported);
    assert_eq!(audit.take(), ["prepare", "drop_ready"]);
    assert_eq!(runtime.session().state, 42);
}

#[test]
fn real_borrowed_ready_owner_carries_genuine_exact_and_short_claims() {
    let (audit, mut runtime) = fixture::<1>();
    let error = runtime
        .prepare_reset_ordinary()
        .unwrap()
        .reset_admitted(limits(REQUIRED - 1))
        .unwrap_err();
    assert!(
        matches!(error.source().unwrap().downcast_ref::<crate::SessionResetRejection>(),
        Some(crate::SessionResetRejection::MemoryDomain(crate::MemoryDomainError::BudgetExceeded { requested_bytes: REQUIRED, limit_bytes, .. })) if *limit_bytes == REQUIRED - 1)
    );
    assert_eq!(
        audit.take(),
        ["prepare", "ready_reset", "compare", "drop_ready"]
    );
    assert_eq!(runtime.session().state, 42);
    runtime
        .prepare_reset_ordinary()
        .unwrap()
        .reset_admitted(limits(REQUIRED))
        .unwrap();
    assert_eq!(
        audit.take(),
        ["prepare", "ready_reset", "compare", "publish", "drop_ready"]
    );
    assert_eq!(runtime.session().state, 0);
    assert_eq!(runtime.decode(7).unwrap().output, 7);
}

#[test]
fn initial_and_changed_ready_capabilities_reject_before_claim_consumption() {
    let (audit, mut runtime) = fixture::<1>();
    audit.capabilities.set(changed());
    assert!(runtime.reset_admitted(limits(REQUIRED)).is_err());
    assert!(runtime.prepare_reset_ordinary().is_err());
    assert!(audit.take().is_empty());
    audit.capabilities.set(SessionCapabilities::default());
    audit.change_during_preparation.set(true);
    assert!(runtime.prepare_reset_ordinary().is_err());
    assert_eq!(audit.take(), ["prepare", "drop_ready"]);
    audit.change_during_preparation.set(false);
    audit.capabilities.set(SessionCapabilities::default());
    let ready = runtime.prepare_reset_ordinary().unwrap();
    audit.capabilities.set(changed());
    let error = ready.reset_admitted(limits(REQUIRED)).unwrap_err();
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<crate::SessionAdmissionError>()
        .is_some());
    assert_eq!(audit.take(), ["prepare", "drop_ready"]);
    assert_eq!(runtime.session().state, 42);
}

#[test]
fn busy_poison_and_foreign_claim_keep_original_state_and_end_the_ready_borrow() {
    let (audit, mut runtime) = fixture::<1>();
    audit.busy.set(true);
    let error = match runtime.prepare_reset_ordinary() {
        Err(error) => error,
        Ok(_) => panic!("busy preparation"),
    };
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    assert_eq!(audit.take(), ["prepare"]);
    let error = runtime.reset_admitted(limits(REQUIRED)).unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    assert_eq!(audit.take(), ["direct"]);
    audit.busy.set(false);
    let ready = runtime.prepare_reset_ordinary().unwrap();
    audit.busy.set(true);
    let error = ready.reset_admitted(limits(REQUIRED)).unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<crate::SessionResetRejection>(),
        Some(crate::SessionResetRejection::Busy)
    ));
    assert_eq!(audit.take(), ["prepare", "ready_reset", "drop_ready"]);
    assert_eq!(runtime.session().state, 42);
    audit.busy.set(false);
    let ready = runtime.prepare_reset_ordinary().unwrap();
    audit.poisoned.set(true);
    assert_eq!(
        ready.reset_admitted(limits(REQUIRED)).unwrap_err().kind(),
        BackendFailureKind::InvalidSession
    );
    assert_eq!(audit.take(), ["prepare", "ready_reset", "drop_ready"]);
    audit.poisoned.set(false);
    audit.foreign_claim_probe.set(true);
    let error = runtime
        .prepare_reset_ordinary()
        .unwrap()
        .reset_admitted(limits(REQUIRED))
        .unwrap_err();
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<crate::SessionResetRejection>(),
        Some(crate::SessionResetRejection::ForeignSession)
    ));
    assert_eq!(audit.take(), ["prepare", "ready_reset", "drop_ready"]);
    assert_eq!(runtime.session().state, 42);
}

#[test]
fn abandoning_readiness_only_releases_borrow_and_never_resets_or_waits() {
    let (audit, mut runtime) = fixture::<1>();
    audit.fail_sync.set(true);
    drop(runtime.prepare_reset_ordinary().unwrap());
    assert_eq!(audit.take(), ["prepare", "drop_ready"]);
    assert_eq!(runtime.decode(3).unwrap().output, 45);
    assert!(audit.take().is_empty());
}
