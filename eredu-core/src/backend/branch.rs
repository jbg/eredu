//! Loans of exact ordinary machines for an atomic native branch placement.
use super::*;

/// A failed placement cannot authorize another prediction or state copy.
#[derive(Debug, thiserror::Error)]
#[error("text branch placement failed and fenced this machine")]
pub struct TextBranchFenced;

/// Inactive ordinary machine paired with the actual displaced native slot.
/// Only a completed machine can produce this owner; no public constructor or
/// mutable state extraction can substitute a run or native revision.
pub struct TextGenerationBranch<B: crate::execution_control::NativeTextStateBackend, C: TokenFilterController> {
    inner: TextGenerationMachine<B, C>,
    native: B::NativeTextState,
}

impl<B: crate::execution_control::NativeTextStateBackend, C: TokenFilterController> TextGenerationBranch<B, C> {
    /// Read-only prospective constraint state.
    pub fn controller(&self) -> &C { &self.inner.controller }
    /// Remaining output allowance of this exact branch.
    pub fn remaining_tokens(&self) -> Option<usize> { self.inner.remaining_tokens }
    /// Borrows the actual admitted child sources and installed sampler facts.
    pub fn resume_facts(&self) -> TextResumeFacts<'_> where B: TextResumeBackend {
        B::text_resume_facts(&self.inner.backend_state)
    }
}

impl<B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Borrows cold facts from the exclusively held selected execution.
    pub fn runtime(&self) -> &ModelRuntime<B> { self.runtime }
    /// Exclusive serial branch tree; later fresh runs on the same runtime differ.
    pub fn driver_identity(&self) -> TextDriverIdentity { TextDriverIdentity::for_branch(&self.inner.branch_owner) }
    /// A serial placement failure permanently prevents this machine advancing.
    pub fn branch_is_fenced(&self) -> bool { self.inner.branch_fenced }
    /// Borrows the actual admitted sources and installed sampler facts.
    pub fn resume_facts(&self) -> TextResumeFacts<'_> where B: TextResumeBackend {
        B::text_resume_facts(&self.inner.backend_state)
    }

    /// Installs a separately admitted replacement while retaining the original
    /// exclusive runtime borrow. The producer must finish native installation
    /// before returning its ordinary machine; host installation is infallible.
    pub fn replace_completed<T, E>(
        &mut self,
        prepare: impl for<'r> FnOnce(&'r mut ModelRuntime<B>) -> Result<Option<(ControlledTextGeneration<'r, B, C>, T)>, E>,
        map: impl Fn(TextContinuationError<B::Error, C::Error>) -> E,
    ) -> Result<Option<T>, E> {
        self.snapshot_source().map_err(&map)?;
        let Some((replacement, copied)) = prepare(self.runtime)? else { return Ok(None) };
        let mut inner = replacement.inner;
        inner.branch_owner = self.inner.branch_owner.clone();
        self.inner = inner;
        Ok(Some(copied))
    }
}

impl<B: crate::execution_control::NativeTextStateBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Builds a child through original resume and places the parent back through
    /// the same atomic exchange worker. No prompt replay, model copy, sampler draw
    /// or delivery occurs during exchange. A placement failure fences the parent
    /// even if it happened before the native worker's first mutation.
    pub fn fork_completed<T, E>(
        &mut self,
        prepare: impl for<'r> FnOnce(&'r mut ModelRuntime<B>) -> Result<Option<(ControlledTextGeneration<'r, B, C>, B::NativeTextState, T)>, E>,
        map: impl Fn(TextContinuationError<B::Error, C::Error>) -> E,
    ) -> Result<Option<(TextGenerationBranch<B, C>, T)>, E> {
        self.snapshot_source().map_err(&map)?;
        let Some((replacement, mut native, copied)) = prepare(self.runtime)? else { return Ok(None) };
        let ControlledTextGeneration { runtime, mut inner } = replacement;
        inner.branch_owner = self.inner.branch_owner.clone();
        self.inner.branch_fenced = true;
        B::exchange_text_branch(runtime, TextBranchSource::from_machine(&mut inner),
            TextBranchSource::from_machine(&mut self.inner), &mut native)
            .map_err(|cause| map(ControlledTextGenerationError::Backend(cause).into()))?;
        self.inner.branch_fenced = false;
        Ok(Some((TextGenerationBranch { inner, native }, copied)))
    }

    /// Exchanges both complete machines after source identity and quiescence
    /// validation. Native placement success precedes the infallible host move.
    pub fn exchange_branch(&mut self, branch: &mut TextGenerationBranch<B, C>)
        -> Result<(), TextContinuationError<B::Error, C::Error>> {
        self.snapshot_source()?;
        if self.inner.branch_owner != branch.inner.branch_owner {
            return Err(TextContinuationError::IncompatibleDriver);
        }
        if branch.inner.branch_fenced { return Err(TextContinuationError::Failed) }
        branch.inner.step_context.validate().map_err(ControlledTextGenerationError::Preparation)?;
        if !branch.inner.completions.is_empty() || branch.inner.capture_pending() {
            return Err(TextContinuationError::NotQuiescent);
        }
        B::exchange_text_branch(self.runtime, TextBranchSource::from_machine(&mut self.inner),
            TextBranchSource::from_machine(&mut branch.inner), &mut branch.native)
            .map_err(ControlledTextGenerationError::Backend)?;
        std::mem::swap(&mut self.inner, &mut branch.inner);
        Ok(())
    }
}

/// Closed completed-source loan supplied by the ordinary machine's exchange
/// guard. Applications cannot construct a source from counters or native handles.
/// The backend admits each placement before changing revisions, then rebinds
/// actual pending input using the receipt from that same native exchange.
pub struct TextBranchSource<'a, B: TextGenerationBackend> {
    context: &'a TextStepContext,
    state: &'a mut B::TextGenerationState,
    pending: Option<PendingTextInput<&'a mut B::Prompt, &'a mut B::Token>>,
}
impl<'a, B: TextGenerationBackend> TextBranchSource<'a, B> {
    pub(super) fn from_machine<C: TokenFilterController>(machine: &'a mut TextGenerationMachine<B, C>) -> Self {
        Self {
            context: &machine.step_context,
            state: &mut machine.backend_state,
            pending: machine.step.as_mut().map(|input| match input {
                PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(prompt),
                PendingTextInput::Decode(token) => PendingTextInput::Decode(token),
            }),
        }
    }
    /// Actual core-issued run, policy and next-attempt evidence.
    pub fn context(&self) -> &TextStepContext { self.context }
    /// Installed sampling and observation owner, without mutation.
    pub fn state(&self) -> &B::TextGenerationState { self.state }
    /// Exact pending prompt or committed token, without advancing it.
    pub fn pending(&self) -> Option<PendingTextInput<&B::Prompt, &B::Token>> {
        self.pending.as_ref().map(|input| match input {
            PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(&**prompt),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(&**token),
        })
    }
    /// Consumes the closed loan for backend-only placement rebinding. The
    /// mutable references are the actual machine values, not copied handles.
    /// No controller, run/policy revision setter or new attempt is exposed.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (
        &'a TextStepContext,
        &'a mut B::TextGenerationState,
        Option<PendingTextInput<&'a mut B::Prompt, &'a mut B::Token>>,
    ) {
        (self.context, self.state, self.pending)
    }
}
