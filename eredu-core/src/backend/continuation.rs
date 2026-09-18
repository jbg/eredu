//! Detached continuations for the ordinary text-generation machine.

use super::*;

/// Unforgeable identity of one continuation, distinct from its shared driver.
/// This is the already issued immutable run identity; clones allocate nothing
/// and cannot keep an admitted request/control allocation alive after retirement.
#[derive(Clone)]
pub struct TextContinuationIdentity(TextRunIdentity);

impl PartialEq for TextContinuationIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for TextContinuationIdentity {}

/// Unforgeable identity of the exclusive driver shared by one serial branch tree.
/// A later generation on the same loaded executable has a different identity.
#[derive(Clone, PartialEq, Eq)]
pub struct TextDriverIdentity(TextRunIdentity);
impl TextDriverIdentity {
    fn validate(&self) -> Result<(), BackendFailure> { self.0.validate() }
    pub(super) fn for_branch(identity: &TextRunIdentity) -> Self { Self(identity.clone()) }
}

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
    owner: TextDriverIdentity,
    identity: TextContinuationIdentity,
    inner: TextGenerationMachine<B, C>,
    failed: bool,
    records_drained: bool,
    // Final custody retires after every machine-owned host/native payload.
    host_preparation: HostPreparationAuthority,
}

impl<B, C> TextGenerationContinuation<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    #[cfg(test)]
    pub(super) fn context_for_test(&self) -> &TextStepContext {
        &self.inner.step_context
    }
    #[cfg(test)]
    pub(super) fn context_mut_for_test(&mut self) -> &mut TextStepContext {
        &mut self.inner.step_context
    }
    #[cfg(test)]
    pub(super) fn identity_for_test(&self) -> TextContinuationIdentity {
        self.identity.clone()
    }

    /// Retains host-copy custody until this continuation's payloads retire.
    ///
    /// This combines existing and incoming ownership without admitting work,
    /// proving a byte bound, or acquiring authority for another copy. Callers
    /// must independently acquire the backend's authority before allocation and
    /// retain custody on any separately escaping payloads or aliases.
    pub fn retain_host_preparation(&mut self, authority: HostPreparationAuthority) {
        let combined = HostPreparationAuthority::retain((self.host_preparation.clone(), authority));
        self.host_preparation = combined;
    }

    /// Borrows the canonical constraint state without advancing it.
    pub fn controller(&self) -> &C {
        &self.inner.controller
    }

    /// Queries the existing controller's termination condition without exposing
    /// mutable policy or revising this run's bound policy identity.
    ///
    /// This preserves `TokenFilterController::is_complete` semantics, including
    /// internal query/cache mutation and its exact error. It grants no step or
    /// completion authority and does not make the continuation quiescent.
    pub fn controller_is_complete(&mut self) -> Result<bool, C::Error> {
        self.inner.controller.is_complete()
    }

    /// Mutably borrows the canonical constraint state and revises policy.
    pub fn controller_mut(&mut self) -> &mut C {
        self.inner.step_context.revise_policy();
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
        self.inner
            .step_context
            .validate()
            .map_err(ControlledTextGenerationError::Preparation)?;
        if self.failed {
            return Err(TextContinuationError::Failed);
        }
        if !self.inner.completions.is_empty()
            || !self.records_drained
            || self.inner.capture_pending()
        {
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
    owner: TextDriverIdentity,
}

impl<'a, B: TextGenerationBackend> TextGenerationDriver<'a, B> {
    /// Exclusively borrows the runtime while this driver exists. A continuation
    /// retained after driver drop can be cleaned up but cannot attach to another
    /// driver, even one borrowing the same runtime.
    pub fn new(runtime: &'a mut ModelRuntime<B>) -> Self {
        Self {
            runtime,
            owner: TextDriverIdentity(TextStepContext::new().run_identity().clone()),
        }
    }

    /// Read-only access for exact loaded-session discovery and native estimates.
    pub fn runtime(&self) -> &ModelRuntime<B> {
        self.runtime
    }

    /// Agrees delivery using this continuation's exact retained readiness source.
    /// Validation failure preserves the original local error when one exists.
    pub fn finish_text_preparation_cancellable<C:TokenFilterController,T,E>(
        &self,state:&TextGenerationContinuation<B,C>,
        stage:crate::run_preparation::TextPreparationStage,local:Result<Option<T>,E>,
        map_backend:impl FnOnce(BackendFailure)->E,
    )->Result<Option<T>,E> {
        let valid=if self.owner != state.owner || state.failed {
            Err(PreparedRequestRejection::RequestMismatch.into_backend_failure())
        } else {state.inner.step_context.validate()};
        if let Err(error)=valid {
            return match local {Err(local)=>Err(local),Ok(value)=>{drop(value);Err(map_backend(error))}};
        }
        self.runtime.finish_text_preparation_control_cancellable(state.inner.preparation_control.as_ref(),
            stage,local,map_backend)
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
        self.start_input(TextGenerationInput::Prepared(prompt), config, controller)
    }

    /// Starts detached ordinary generation with admission before native input.
    pub fn start_input<C: TokenFilterController>(
        &mut self,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.owner.validate().map_err(ControlledTextGenerationError::Preparation)?;
        let inner = TextGenerationMachine::new(self.runtime, input, config, controller)?;
        Ok(TextGenerationContinuation {
            owner: self.owner.clone(),
            identity: TextContinuationIdentity(inner.step_context.run_identity().clone()),
            inner,
            failed: false,
            records_drained: true,
            host_preparation: HostPreparationAuthority::unmanaged(),
        })
    }

    /// Starts a detached machine from prepared input and original-admission options.
    pub fn start_with_options<C: TokenFilterController>(
        &mut self,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
        options: TextPreparationOptions,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.start_input_with_options(
            TextGenerationInput::Prepared(prompt),
            config,
            controller,
            options,
        )
    }

    /// Includes owned sources in the original admission and completes the single
    /// Instrumentation readiness stage before exposing the continuation.
    pub fn start_input_with_options<C: TokenFilterController>(
        &mut self,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: TextPreparationOptions,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.owner.validate().map_err(ControlledTextGenerationError::Preparation)?;
        // Obtain the validated fresh run before exposing its continuation identity.
        let inner = TextGenerationMachine::new_preparation(
            self.runtime,
            input,
            config,
            controller,
            Some(options),
        )?;
        Ok(TextGenerationContinuation {
            owner: self.owner.clone(),
            identity: TextContinuationIdentity(inner.step_context.run_identity().clone()),
            inner,
            failed: false,
            records_drained: true,
            host_preparation: HostPreparationAuthority::unmanaged(),
        })
    }

    /// Prepares a detached original request with borrowed EOS/max policy.
    /// Sequence-only preparation does not add Instrumentation readiness.
    pub fn start_input_with_sequence<C: TokenFilterController>(
        &mut self,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.owner.validate().map_err(ControlledTextGenerationError::Preparation)?;
        let inner = TextGenerationMachine::new_preparation_with_sequence(
            self.runtime,
            input,
            config,
            controller,
            options,
            Some(sequence),
        )?;
        Ok(TextGenerationContinuation {
            owner: self.owner.clone(),
            identity: TextContinuationIdentity(inner.step_context.run_identity().clone()),
            inner,
            failed: false,
            records_drained: true,
            host_preparation: HostPreparationAuthority::unmanaged(),
        })
    }

    /// Validates this driver's exact ownership before taking the sequence once.
    /// The caller must materialize before its existing first finish_step vote.
    pub fn take_prepared_sequence<C: TokenFilterController>(
        &self,
        state: &mut TextGenerationContinuation<B, C>,
    ) -> Result<
        Option<crate::generation::RetainedGenerationSequence>,
        TextContinuationError<B::Error, C::Error>,
    > {
        self.validate(state)?;
        if state.failed {
            return Err(TextContinuationError::Failed);
        }
        Ok(state.inner.prepared_sequence.take())
    }

    /// Starts a fresh admitted machine from immutable saved data and the final
    /// owned controller. No old preparation/context is cloned. Zero output or
    /// cancellation returns no continuation and performs no saved-source copy.
    /// Backends must complete installation and Sampling readiness before this
    /// exposes an advanceable continuation.
    pub fn resume_saved<C: TokenFilterController>(
        &mut self,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<
        Option<TextGenerationContinuation<B, C>>,
        ControlledTextGenerationError<B::Error, C::Error>,
    >
    where
        B: TextResumeBackend,
    {
        self.owner.validate().map_err(ControlledTextGenerationError::Preparation)?;
        let inner = TextGenerationMachine::from_resume(
            self.runtime,
            saved,
            config,
            controller,
            cancellation,
        )?;
        Ok(inner.map(|inner| TextGenerationContinuation {
            owner: self.owner.clone(),
            identity: TextContinuationIdentity(inner.step_context.run_identity().clone()),
            inner,
            failed: false,
            records_drained: true,
            host_preparation: HostPreparationAuthority::unmanaged(),
        }))
    }

    fn validate<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<(), TextContinuationError<B::Error, C::Error>> {
        self.owner.validate().map_err(ControlledTextGenerationError::Preparation)?;
        if self.owner != state.owner {
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
        if state.inner.prepared_sequence.requires_copy_admission() {
            return Err(
                ControlledTextGenerationError::Preparation(BackendFailure::new(
                    BackendFailureKind::Unsupported,
                    GenerationSequenceAdmissionError::CopyNotAdmitted,
                ))
                .into(),
            );
        }
        Ok(TextContinuationBoundary {
            runtime: self.runtime,
            state,
        })
    }

    /// Advances at most one constraint-committed canonical token using the
    /// ordinary sampler and pending input. Call `take_completed_delivery` before
    /// advancing again or composing a pause/snapshot boundary.
    #[allow(clippy::type_complexity)]
    pub fn advance<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
    ) -> Result<Option<ControlledToken<B::Token>>, TextContinuationError<B::Error, C::Error>> {
        self.advance_cancellable(state, &crate::GenerationCancellationToken::new())
    }

    /// Uses the same advancement with a live, non-rewindable cancellation token.
    /// A cancelled prefill settles native work and commits no sampled token.
    #[allow(clippy::type_complexity)]
    pub fn advance_cancellable<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<ControlledToken<B::Token>>, TextContinuationError<B::Error, C::Error>> {
        self.validate(state)?;
        state.require_quiescent()?;
        // A caught unwind must not make a partially advanced machine resumable.
        state.failed = true;
        state.records_drained = false;
        match state.inner.next_committed(self.runtime, cancellation) {
            None => {
                state.failed = false;
                state.records_drained = !state.inner.capture_pending();
                Ok(None)
            }
            Some(Ok(token)) => {
                state.failed = false;
                Ok(Some(token))
            }
            Some(Err(error)) => Err(TextContinuationError::Generation(error)),
        }
    }

    /// Retained delivery after exact completion. A ready shared frame moves
    /// without copying its payload or detaching custody. None is quiescent only
    /// when `capture_pending` is false. Drain failures keep the continuation
    /// undrained and remain retryable for delivery; they never clear failure.
    pub fn take_completed_delivery<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
    ) -> Result<
        Option<crate::capture::SharedCapturedStep>,
        TextContinuationError<B::Error, C::Error>,
    > {
        self.validate(state)?;
        let was_failed = state.failed;
        state.failed = true;
        state.records_drained = false;
        let records = state
            .inner
            .take_capture_delivery()
            .map_err(ControlledTextGenerationError::Backend)?;
        state.records_drained = !state.inner.capture_pending();
        state.failed = was_failed;
        Ok(records)
    }

    /// Read-only pending evidence; no completion or delivery is performed.
    pub fn capture_pending<C: TokenFilterController>(
        &self,
        state: &TextGenerationContinuation<B, C>,
    ) -> Result<bool, TextContinuationError<B::Error, C::Error>> {
        self.validate(state)?;
        Ok(state.inner.capture_pending())
    }

    /// Installs observations before the first model prediction.
    pub fn enable_capture<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.validate_installation(state)
            .map_err(crate::run_preparation::TextCaptureSetupError::Capture)?;
        state.inner.step_context.revise_policy();
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = {
            // Failed backend setup can still leave a changed instrumentation state.
            B::configure_text_capture(self.runtime, &mut state.inner.backend_state, plan)
        };
        self.runtime.finish_text_preparation_control(state.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Installs ordinary capture using the actual pending source and the shared
    /// Instrumentation agreement. No separate startup engine or original grant.
    pub fn enable_prepared_capture<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        source: crate::capture::SharedCapturePlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = self
            .validate_installation(state)
            .map_err(crate::run_preparation::TextCaptureSetupError::Capture)
            .and_then(|()| {
                state
                    .inner
                    .configure_prepared_capture(self.runtime, source)
                    .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)
            });
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.runtime.finish_text_preparation_control(state.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local,
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Installs immutable observation/intervention admissions before execution.
    pub fn enable_interventions<C: TokenFilterController>(
        &mut self,
        state: &mut TextGenerationContinuation<B, C>,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.validate_installation(state)
            .map_err(crate::run_preparation::TextCaptureSetupError::Capture)?;
        state.inner.step_context.revise_policy();
        state
            .inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = {
            B::configure_text_interventions(
                self.runtime,
                &mut state.inner.backend_state,
                capture,
                plan,
            )
        };
        self.runtime.finish_text_preparation_control(state.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
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

/// Read-only, completed generation state for an independently admitted snapshot.
/// Borrowed and detached sessions use this same machine view. It cannot install
/// state, issue a prediction, or manufacture a detached driver identity.
pub struct TextSnapshotSource<'a, B: TextGenerationBackend, C: TokenFilterController> {
    runtime: &'a mut ModelRuntime<B>,
    inner: &'a TextGenerationMachine<B, C>,
    driver: Option<TextDriverIdentity>,
}
impl<B: TextGenerationBackend, C: TokenFilterController> TextSnapshotSource<'_, B, C> {
    /// Existing immutable run identity; cloning it allocates nothing.
    pub fn identity(&self) -> TextContinuationIdentity {
        TextContinuationIdentity(self.inner.step_context.run_identity().clone())
    }
    /// Only detached continuations possess a branch driver identity.
    pub fn driver_identity(&self) -> Option<TextDriverIdentity> {
        self.driver.clone()
    }
    /// Existing immutable controller state.
    pub fn controller(&self) -> &C {
        &self.inner.controller
    }
    /// Remaining model predictions at this boundary.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.inner.remaining_tokens
    }
    /// Borrows unchanged native/sampling/input source facts.
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
            &self.inner.backend_state,
            self.inner.step.as_ref().map(PendingTextInput::as_ref),
        )
    }
    /// Lends only runtime mechanism access; source state and policy stay immutable.
    /// The copy mechanism must separately admit destination storage and native work.
    #[allow(clippy::type_complexity)]
    pub fn copy_mechanism_parts(
        &mut self,
    ) -> (
        &mut ModelRuntime<B>,
        &B::TextGenerationState,
        Option<PendingTextInput<&B::Prompt, &B::Token>>,
    ) {
        (
            self.runtime,
            &self.inner.backend_state,
            self.inner.step.as_ref().map(PendingTextInput::as_ref),
        )
    }
}
impl<B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Borrows an already completed and drained source. This performs no wait,
    /// record drain, allocation, policy revision or new generation admission.
    pub fn snapshot_source(
        &mut self,
    ) -> Result<TextSnapshotSource<'_, B, C>, TextContinuationError<B::Error, C::Error>> {
        if self.inner.branch_fenced { return Err(TextContinuationError::Failed) }
        self.inner
            .step_context
            .validate()
            .map_err(ControlledTextGenerationError::Preparation)?;
        if !self.inner.completions.is_empty() || self.inner.capture_pending() {
            return Err(TextContinuationError::NotQuiescent);
        }
        Ok(TextSnapshotSource {
            runtime: self.runtime,
            inner: &self.inner,
            driver: None,
        })
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

impl<B: crate::execution_control::TextSamplingControlBackend, C: TokenFilterController>
    TextContinuationBoundary<'_, '_, B, C>
{
    /// Lends only the admitted prospective sampler update at this already
    /// completed boundary. Model, controller and source policy remain private.
    pub fn sampling_boundary(&mut self) -> TextSamplingBoundary<'_, B> {
        TextSamplingBoundary {
            runtime: self.runtime,
            state: &mut self.state.inner.backend_state,
            context: &self.state.inner.step_context,
        }
    }
}

impl<B: TextGenerationBackend, C: TokenFilterController> TextContinuationBoundary<'_, '_, B, C> {
    /// Lends the same immutable capture view as a borrowed generation machine.
    pub fn snapshot_source(&mut self) -> TextSnapshotSource<'_, B, C> {
        let driver = Some(self.driver_identity());
        TextSnapshotSource {
            runtime: self.runtime,
            inner: &self.state.inner,
            driver,
        }
    }
    /// Same-run identity used before restoring any state component.
    pub fn identity(&self) -> TextContinuationIdentity {
        self.state.identity.clone()
    }

    /// Exclusive source driver, common to this run and its isolated descendants.
    pub fn driver_identity(&self) -> TextDriverIdentity {
        self.state.owner.clone()
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

    /// Lends the runtime to an independently admitted copy mechanism while
    /// keeping generation state and pending input read-only. Unlike mutable
    /// mechanism access, this does not revise the source run's policy.
    ///
    /// The caller must establish destination funding and the backend's exact
    /// source/submission guards before copying. This borrow grants no model
    /// prediction, source mutation, or future continuation authority.
    #[allow(clippy::type_complexity)]
    pub fn copy_mechanism_parts(
        &mut self,
    ) -> (
        &mut ModelRuntime<B>,
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
        self.state.inner.step_context.revise_policy();
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
        self.state.inner.step_context.revise_policy();
        self.state.inner.controller = controller;
        self.state.inner.step = pending;
        self.state.inner.remaining_tokens = remaining_tokens;
    }

    /// Retains independently acquired host-copy custody with the installed
    /// continuation. This is lifetime retention only, not copy admission or
    /// settlement. Retain before installation so partial installation on unwind
    /// remains protected; conservative custody may then survive a failed copy.
    pub fn retain_host_preparation(&mut self, authority: HostPreparationAuthority) {
        self.state.retain_host_preparation(authority);
    }

    /// Constructs a child continuation from independently prepared components.
    /// Its native model state must be installed before advancing it. A child has
    /// a fresh run identity and shares only this driver's execution authority.
    /// Inherited host custody protects shared source aliases; the caller must
    /// separately acquire and retain authority for the destination copy.
    pub fn fork_host_state(
        &self,
        backend_state: B::TextGenerationState,
        controller: C,
        pending: Option<PendingTextInput<B::Prompt, B::Token>>,
        remaining_tokens: Option<usize>,
    ) -> TextGenerationContinuation<B, C> {
        // Infallible compatibility API: failed issuance creates a terminal
        // context. No continuation boundary or execution hook can expose it as
        // a valid new identity; restore cannot clear that terminal condition.
        let step_context = TextStepContext::new();
        TextGenerationContinuation {
            owner: self.state.owner.clone(),
            identity: TextContinuationIdentity(step_context.run_identity().clone()),
            inner: TextGenerationMachine {
                capture_source: self.state.inner.capture_source.clone(),
                intervention_source: self.state.inner.intervention_source.clone(),
                resume_host: self.state.inner.resume_host.clone(),
                prepared_sequence: preparation::PreparedSequence::Ordinary,
                preparation: self.state.inner.preparation.clone(),
                preparation_control: self.state.inner.preparation_control.clone(),
                backend_state,
                controller,
                step: pending,
                completions: Vec::new(),
                    remaining_tokens,
                branch_owner: self.state.inner.branch_owner.clone(),
                branch_fenced: false,
                step_context,
            },
            failed: false,
            records_drained: true,
            host_preparation: self.state.host_preparation.clone(),
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
        if self.state.owner != other.owner {
            return Err(TextContinuationError::IncompatibleDriver);
        }
        other.require_quiescent()?;
        B::exchange_text_branch(self.runtime,
            TextBranchSource::from_machine(&mut self.state.inner),
            TextBranchSource::from_machine(&mut other.inner), native)
            .map_err(ControlledTextGenerationError::Backend)?;
        std::mem::swap(self.state, other);
        Ok(())
    }
}
