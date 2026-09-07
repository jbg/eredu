//! Detached continuations for the ordinary text-generation machine.

use super::*;
use std::sync::Arc;

/// Unforgeable identity of one continuation, distinct from its shared driver.
#[derive(Clone)]
pub struct TextContinuationIdentity(Arc<()>);

impl PartialEq for TextContinuationIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TextContinuationIdentity {}

/// Unforgeable identity of the exclusive driver shared by one serial branch tree.
/// A later generation on the same loaded executable has a different identity.
#[derive(Clone)]
pub struct TextDriverIdentity(Arc<()>);
impl PartialEq for TextDriverIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TextDriverIdentity {}

/// Failure to advance or settle a detached ordinary continuation.
#[derive(Debug, thiserror::Error)]
pub enum TextContinuationError<B, C>
where
    B: std::error::Error + 'static,
    C: std::error::Error + 'static,
{
    /// The continuation was created by another exclusive runtime driver.
    #[error("text continuation belongs to a different runtime driver")]
    IncompatibleDriver,
    /// A previous operation failed; only cleanup and record draining remain valid.
    #[error("text continuation failed and cannot advance")]
    Failed,
    /// Native completion and portable record draining have not both succeeded.
    #[error("text continuation has no completed, drained boundary")]
    NotQuiescent,
    /// Ordinary backend execution or portable constraint control failed.
    #[error(transparent)]
    Generation(#[from] ControlledTextGenerationError<B, C>),
}

/// Pending input, sampling state and constraint state of the ordinary machine.
///
/// This is neither a full generation snapshot nor a cloneable native state.
/// After commitment, the last token remains pending decode input. Completion
/// handles stay here until settling or drop. The owning facade also retains
/// native model state, semantic decoding, termination and output delivery.
pub struct TextGenerationContinuation<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    owner: Arc<()>,
    identity: TextContinuationIdentity,
    inner: TextGenerationMachine<B, C>,
    failed: bool,
    records_drained: bool,
}

impl<B, C> TextGenerationContinuation<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    /// Borrows the canonical constraint state without advancing it.
    pub fn controller(&self) -> &C {
        &self.inner.controller
    }

    /// Borrows the canonical constraint state for the ordinary semantic checks.
    pub fn controller_mut(&mut self) -> &mut C {
        &mut self.inner.controller
    }

    /// Remaining ordinary token allowance, independent of facade EOS policy.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.inner.remaining_tokens
    }

    /// Whether the next model input is the original prompt rather than a token.
    pub fn is_prefill_pending(&self) -> bool {
        matches!(self.inner.step, Some(PendingTextInput::Prefill(_)))
    }

    /// Requires exact completion and delivery before snapshot composition.
    /// This does not establish the facade's semantic or lifecycle boundary.
    pub fn require_quiescent(&self) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        if self.failed {
            return Err(TextContinuationError::Failed);
        }
        if !self.inner.completions.is_empty() || !self.records_drained {
            return Err(TextContinuationError::NotQuiescent);
        }
        Ok(())
    }
}

/// Exclusive execution owner for detached ordinary continuations.
///
/// The same machine drives the existing borrowed iterators. This owner allows a
/// facade to retain separate continuations while serially switching independently
/// saved native model states. It does not switch those native states itself: the
/// facade must compose the backend's validated state-exchange mechanism. Native
/// objects need not implement `Send` or `Sync`.
pub struct TextGenerationDriver<'a, B: TextGenerationBackend> {
    runtime: &'a mut ModelRuntime<B>,
    owner: Arc<()>,
}

impl<'a, B: TextGenerationBackend> TextGenerationDriver<'a, B> {
    /// Exclusively borrows the runtime while this driver exists. A continuation
    /// retained after driver drop can be cleaned up but cannot attach to another
    /// driver, even one borrowing the same runtime.
    pub fn new(runtime: &'a mut ModelRuntime<B>) -> Self {
        Self {
            runtime,
            owner: Arc::new(()),
        }
    }

    /// Read-only access for exact loaded-session discovery and native estimates.
    pub fn runtime(&self) -> &ModelRuntime<B> {
        self.runtime
    }

    /// Starts the existing ordinary machine with an opaque prepared prompt.
    /// Neither model execution nor token sampling occurs here.
    pub fn start<C: TokenFilterController>(
        &mut self,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        Ok(TextGenerationContinuation {
            owner: Arc::clone(&self.owner),
            identity: TextContinuationIdentity(Arc::new(())),
            inner: TextGenerationMachine::new(self.runtime, prompt, config, controller)?,
            failed: false,
            records_drained: true,
        })
    }

    fn validate<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        if !Arc::ptr_eq(&self.owner, &state.owner) {
            return Err(TextContinuationError::IncompatibleDriver);
        }
        Ok(())
    }

    /// Lends the completed ordinary state to portable snapshot composition.
    /// All native work and the preceding bounded record batch must be settled.
    /// The guard holds both mutable borrows, preventing advancement while native
    /// copies, compatibility validation and host-state installation are composed.
    pub fn quiescent<'d, 's, C: TokenFilterController>(
        &'d mut self,
        state: &'s mut TextGenerationContinuation<B, C>,
    ) -> Result<TextContinuationBoundary<'d, 's, B, C>, TextContinuationError<B::Error, C::Error>>
    {
        self.validate(state)?;
        state.require_quiescent()?;
        Ok(TextContinuationBoundary {
            runtime: self.runtime,
            state,
        })
    }

    /// Advances at most one constraint-committed canonical token using the
    /// ordinary sampler and pending input. Call `take_completed_step` before
    /// advancing again or composing a pause/snapshot boundary.
    #[allow(clippy::type_complexity)]
    pub fn advance<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
    ) -> Result<Option<ControlledToken<B::Token>>, TextContinuationError<B::Error, C::Error>> {
        self.validate(state)?;
        state.require_quiescent()?;
        // A caught unwind must not make a partially advanced machine resumable.
        state.failed = true;
        state.records_drained = false;
        match state.inner.next_committed(self.runtime) {
            None => {
                state.failed = false;
                state.records_drained = true;
                Ok(None)
            }
            Some(Ok(token)) => {
                state.failed = false;
                Ok(Some(token))
            }
            Some(Err(error)) => Err(TextContinuationError::Generation(error)),
        }
    }

    /// Settles exact native completion and moves out the single bounded record
    /// batch. It remains available after failure for attributed error evidence;
    /// draining never turns a failed continuation into a resumable one.
    pub fn take_completed_step<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
    ) -> Result<Option<crate::capture::CapturedStep>, TextContinuationError<B::Error, C::Error>>
    {
        self.validate(state)?;
        let was_failed = state.failed;
        state.failed = true;
        if let Err(error) = state.inner.resolve_completions_before_decode() {
            return Err(ControlledTextGenerationError::Backend(error).into());
        }
        let records = B::take_text_capture(&mut state.inner.backend_state);
        state.records_drained = true;
        state.failed = was_failed;
        Ok(records)
    }

    /// Installs observations before the first model prediction.
    pub fn enable_capture<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        self.validate_installation(state)?;
        B::configure_text_capture(self.runtime, &mut state.inner.backend_state, plan)
    }

    /// Installs immutable observation/intervention admissions before execution.
    pub fn enable_interventions<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        self.validate_installation(state)?;
        B::configure_text_interventions(self.runtime, &mut state.inner.backend_state, capture, plan)
    }

    fn validate_installation<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<(), crate::capture::CaptureError> {
        self.validate(state)
            .and_then(|()| state.require_quiescent())
            .map_err(|error| crate::capture::CaptureError::Invalid(error.to_string()))?;
        if !state.is_prefill_pending() {
            return Err(crate::capture::CaptureError::Invalid(
                "capture and interventions must be configured before generation".into(),
            ));
        }
        Ok(())
    }
}

/// Exclusive access for composing native mechanisms at an already settled
/// continuation boundary. This is a backend/runtime-author surface, not a full
/// snapshot API. Operations must validate before mutation and preserve or fence
/// state on failure. Catching an unwind fences this continuation automatically.
pub struct TextContinuationBoundary<'d, 's, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    runtime: &'d mut ModelRuntime<B>,
    state: &'s mut TextGenerationContinuation<B, C>,
}

impl<B: TextGenerationBackend, C: TokenFilterController> TextContinuationBoundary<'_, '_, B, C> {
    /// Same-run identity used before restoring any state component.
    pub fn identity(&self) -> TextContinuationIdentity {
        self.state.identity.clone()
    }

    /// Exclusive source driver, common to this run and its isolated descendants.
    pub fn driver_identity(&self) -> TextDriverIdentity {
        TextDriverIdentity(Arc::clone(&self.state.owner))
    }

    /// Constraint state for an independently staged host checkpoint.
    pub fn controller(&self) -> &C {
        &self.state.inner.controller
    }

    /// Remaining model-prediction allowance, including any pending input.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.state.inner.remaining_tokens
    }

    /// Read-only mechanism inputs. No host or native state advances here.
    #[allow(clippy::type_complexity)]
    pub fn parts(
        &self,
    ) -> (
        &ModelRuntime<B>,
        &B::TextGenerationState,
        Option<PendingTextInput<&B::Prompt, &B::Token>>,
    ) {
        (
            self.runtime,
            &self.state.inner.backend_state,
            self.state.inner.step.as_ref().map(PendingTextInput::as_ref),
        )
    }

    /// Native mechanism access while the pending input remains read-only.
    /// Callers retain the existing completion owner and must not submit ordinary
    /// model predictions through this snapshot-composition borrow.
    #[allow(clippy::type_complexity)]
    pub fn mechanism_parts(
        &mut self,
    ) -> (
        &mut ModelRuntime<B>,
        &mut B::TextGenerationState,
        Option<PendingTextInput<&B::Prompt, &B::Token>>,
    ) {
        (
            self.runtime,
            &mut self.state.inner.backend_state,
            self.state.inner.step.as_ref().map(PendingTextInput::as_ref),
        )
    }

    /// Installs independently prepared host state after all fallible validation,
    /// copying and native installation have succeeded. This performs no work.
    pub fn install_host_state(
        &mut self,
        controller: C,
        pending: Option<PendingTextInput<B::Prompt, B::Token>>,
        remaining_tokens: Option<usize>,
    ) {
        self.state.inner.controller = controller;
        self.state.inner.step = pending;
        self.state.inner.remaining_tokens = remaining_tokens;
    }

    /// Constructs a child continuation from independently prepared components.
    /// Its native model state must be installed before advancing it. A child has
    /// a fresh run identity and shares only this driver's execution authority.
    pub fn fork_host_state(
        &self,
        backend_state: B::TextGenerationState,
        controller: C,
        pending: Option<PendingTextInput<B::Prompt, B::Token>>,
        remaining_tokens: Option<usize>,
    ) -> TextGenerationContinuation<B, C> {
        TextGenerationContinuation {
            owner: Arc::clone(&self.state.owner),
            identity: TextContinuationIdentity(Arc::new(())),
            inner: TextGenerationMachine {
                backend_state,
                controller,
                step: pending,
                completions: Vec::new(),
                remaining_tokens,
            },
            failed: false,
            records_drained: true,
        }
    }

    /// Fences a continuation when a composed operation cannot preserve it.
    pub fn fail(&mut self) {
        self.state.failed = true;
    }
}

impl<B: TextGenerationBackend, C: TokenFilterController> Drop
    for TextContinuationBoundary<'_, '_, B, C>
{
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.state.failed = true;
        }
    }
}

impl<B: crate::execution_control::NativeTextStateBackend> TextGenerationDriver<'_, B> {
    fn validate_boundary<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        self.validate(state)?;
        state.require_quiescent()
    }

    /// Estimates installed or saved native state at a completed continuation
    /// boundary. Portable composition adds all host-state estimates and reserves
    /// resources before calling either copying operation below.
    pub fn estimate_native_state<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
        saved: Option<&B::NativeTextState>,
    ) -> Result<
        Option<crate::execution_control::SnapshotEstimate>,
        TextContinuationError<B::Error, C::Error>,
    > {
        self.validate_boundary(state)?;
        B::estimate_native_text_state(self.runtime, saved)
            .map_err(|error| ControlledTextGenerationError::Backend(error).into())
    }

    /// Copies the installed model state through its existing completion owner.
    /// Caller must first reserve the complete snapshot's resource estimate.
    pub fn capture_native_state<C: TokenFilterController>(
        &mut self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<B::NativeTextState, TextContinuationError<B::Error, C::Error>> {
        self.validate_boundary(state)?;
        B::capture_native_text_state(self.runtime)
            .map_err(|error| ControlledTextGenerationError::Backend(error).into())
    }

    /// Independently copies a compatible saved slot after resource reservation.
    /// The currently installed continuation remains unchanged.
    pub fn copy_native_state<C: TokenFilterController>(
        &mut self,
        state: &TextGenerationContinuation<B, C>,
        saved: &B::NativeTextState,
    ) -> Result<B::NativeTextState, TextContinuationError<B::Error, C::Error>> {
        self.validate_boundary(state)?;
        B::copy_native_text_state(self.runtime, saved)
            .map_err(|error| ControlledTextGenerationError::Backend(error).into())
    }

    /// Exchanges the installed native state with a previously copied slot.
    /// The facade pairs this with the corresponding detached continuation and
    /// semantic state before advancing again. No token or sampler advances here.
    pub fn exchange_native_state<C: TokenFilterController>(
        &mut self,
        state: &TextGenerationContinuation<B, C>,
        slot: &mut B::NativeTextState,
    ) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        self.validate_boundary(state)?;
        B::exchange_native_text_state(self.runtime, slot)
            .map_err(|error| ControlledTextGenerationError::Backend(error).into())
    }
}

impl<B: crate::execution_control::NativeTextStateBackend, C: TokenFilterController>
    TextContinuationBoundary<'_, '_, B, C>
{
    /// Exchanges a quiescent child and its model-state slot with the installed
    /// continuation. All host checks precede native exchange; after its success
    /// the complete ordinary machine is moved infallibly. The facade then moves
    /// the matching semantic/lifecycle state before another prediction.
    pub fn exchange_branch(
        &mut self,
        other: &mut TextGenerationContinuation<B, C>,
        native: &mut B::NativeTextState,
    ) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        if !Arc::ptr_eq(&self.state.owner, &other.owner) {
            return Err(TextContinuationError::IncompatibleDriver);
        }
        other.require_quiescent()?;
        B::validate_native_text_state(self.runtime, native)
            .map_err(ControlledTextGenerationError::Backend)?;
        B::exchange_native_text_state(self.runtime, native)
            .map_err(ControlledTextGenerationError::Backend)?;
        std::mem::swap(self.state, other);
        Ok(())
    }
}
