//! Fresh preparation from an opaque immutable saved source.

use super::*;

/// Cumulative-observation semantics of a fresh original saved-state admission.
/// Both modes require new physical destination/native admission. Restoration
/// preserves current live spending; an independent branch inherits saved usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginalTextResumeKind {
    /// Rewind state while retaining the same run's current cumulative budgets.
    Restore,
    /// Create an independently admitted child with the saved usage as a minimum.
    Branch,
}

/// Prospective child policy borrowed before the same original resume admission.
/// Omitted replacements preserve the actual saved sources. A terminal restore
/// copies complete state but authorizes no subsequent prediction.
#[derive(Clone, Copy, Debug)]
pub struct OriginalTextResumeOptions<'a> {
    /// Restore current cumulative spending or create an independent child.
    pub kind: OriginalTextResumeKind,
    /// Restore terminal state with zero future predictions.
    pub terminal: bool,
    /// Actual destination journal identity when deriving an intervention source.
    pub session_id: Option<&'a str>,
    /// Replacement capture limits, validated against inherited spending.
    pub capture_limits: Option<&'a crate::capture::CaptureLimits>,
    /// Borrowed replacement intervention declarations, including explicit empty.
    pub intervention: Option<&'a crate::intervention::InterventionPlan>,
    /// Prospective sampler policy applied before the child is exposed.
    pub sampling: Option<crate::execution_control::SamplingOverride>,
}
impl OriginalTextResumeOptions<'_> {
    /// Preserves every saved policy under the selected cumulative-budget intent.
    pub const fn new(kind: OriginalTextResumeKind) -> Self {
        Self {
            kind,
            terminal: false,
            session_id: None,
            capture_limits: None,
            intervention: None,
            sampling: None,
        }
    }
}

/// Scalar lineage read from the actual immutable saved sampler and capture
/// checkpoint. These facts provide no copy, admission or execution authority.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextResumeSourceFacts {
    /// Actual capture consumption retained by the saved source.
    pub inherited_capture_usage: crate::capture::CaptureUsage,
    /// Sampler configuration and random-stream facts at the saved boundary.
    pub sampling_before: crate::execution_control::SamplingStateFacts,
}

/// Read-only projection of the actual installed original resume destination.
/// Identity strings borrow its retained immutable sources without allocating.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextResumeFacts<'a> {
    /// Actual sampler after the prospective child policy has been installed.
    pub sampling_after: crate::execution_control::SamplingStateFacts,
    /// Actual capture source identity, absent for an unobserved destination.
    pub capture_plan_id: Option<&'a str>,
    /// Actual intervention source identity, including an explicit empty source.
    pub intervention_plan_id: Option<&'a str>,
    /// Whether the installed source contains any intervention operations.
    pub has_interventions: bool,
}

/// Backend extension for a fresh ordinary run from an immutable saved pair.
///
/// This is separate from snapshot restoration: no old preparation, receipt,
/// attempt or copy-account authority can authorize the new run. Implementations
/// must reject incomplete source, copy, construction or future-work coverage
/// before allocation under an enforced policy. No backend support is supplied by
/// default. The existing ordinary machine owns all prediction/commit behavior.
pub trait TextResumeBackend: TextGenerationBackend {
    /// Closed immutable decoder/sampler/pending-input source and its custody.
    /// This is saved data, not a run or a replayable allocation permission.
    type ResumeSource;

    /// Move-only preparation owning fresh authority, exact source custody and
    /// provisional state. Payload must retire before its authority on every
    /// failure/unwind. Keep no active submission lease across readiness: every
    /// hook must settle or hand off its native work before returning success.
    type ResumePreparation;

    /// Move-only displaced execution state returned by the successful resume
    /// transaction. Ordinary replacement retires it; serial branching keeps it
    /// with the exact outgoing machine for a later authenticated placement.
    type DisplacedState;

    /// Borrows the saved source's exact scalar lineage without copying it or
    /// invoking a planner. None means this projection has not been implemented.
    fn saved_text_resume_facts(_saved: &Self::ResumeSource) -> Option<TextResumeSourceFacts> {
        None
    }

    /// Borrows the successfully installed source identities and sampler facts.
    /// Implementations must not reconstruct sources or allocate diagnostics.
    fn text_resume_facts(state: &Self::TextGenerationState) -> TextResumeFacts<'_>;

    /// Coldly validates the actual source and final owned controller, quotes the
    /// complete new request, and binds fresh authority to this core-issued run
    /// at attempt zero. No controller decisions or numerical copies are allowed.
    /// Return no independently cloned source history or old run authority.
    fn admit_text_resume<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Self::ResumeSource,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<Self::ResumePreparation, BackendFailure>;

    /// Prepare a fresh explicit readiness source for an originally resumed
    /// request. The source shares the loaded session's monotone protocol owner;
    /// saved state supplies no transport or model admission. Failure may not
    /// fall back to ordinary consensus and retains unresolved native custody.
    fn prepare_text_resume_control(
        _runtime: &ModelRuntime<Self>,
        _saved: &Self::ResumeSource,
        _config: TextGenerationConfig,
        _context: &TextStepContext,
        _host: &HostPreparationAuthority,
        _options: &OriginalTextResumeOptions<'_>,
    ) -> Result<Option<Self::TextPreparationControl>, BackendFailure> {
        Ok(None)
    }

    /// Fresh original preparation under an independently admitted host-copy
    /// constructor account. H is lifetime custody only: the backend must admit
    /// the complete new request and its native roles against the issued context.
    /// Saved copy accounts and historical requests cannot supply that authority.
    /// The default refuses without invoking the ordinary allocating planner.
    fn admit_original_text_resume<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _saved: &Self::ResumeSource,
        _config: TextGenerationConfig,
        _controller: &C,
        _context: &TextStepContext,
        _host: &HostPreparationAuthority,
        _options: &OriginalTextResumeOptions<'_>,
    ) -> Result<Self::ResumePreparation, BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }

    /// Actual installed immutable capture source, if resume constructed a fresh
    /// collector. The shared machine retains only this source alias for delivery
    /// and subsequent checkpoints; it supplies no observation/native authority.
    fn text_resume_capture_source(
        _state: &Self::TextGenerationState,
    ) -> Option<&crate::capture::SharedCapturePlan> {
        None
    }

    /// Actual installed immutable intervention source of a resumed machine.
    /// A returned alias retains the same original source; it grants no edit.
    fn text_resume_intervention_source(
        _state: &Self::TextGenerationState,
    ) -> Option<&crate::intervention::SharedInterventionPlan> {
        None
    }

    /// Constructs the new prompt and provisional decoder under the admitted
    /// prompt stage, without exchanging installed state. Copies and publication
    /// use actual source owners; successful payload stays in `prepared`/return
    /// value through Prompt readiness. Complete native work before returning.
    fn prepare_text_resume_prompt(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
    ) -> Result<Self::Prompt, Self::Error>;

    /// Constructs fresh runnable sampling state from actual saved history,
    /// adaptive policy and RNG under this request's consumed sampling stage.
    /// Preserve absolute observation position separately from the new run-local
    /// attempt zero. Historical grants must not be copied into the result.
    fn prepare_text_resume_sampling(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
    ) -> Result<Self::TextGenerationState, Self::Error>;

    /// Revalidates source origin, target, final controller and new run binding,
    /// then installs provisional decoder state through the existing exchange.
    /// Bind the final quote to the actual installed revision, not an old one.
    /// This occurs inside the Sampling local result, before its agreement.
    ///
    /// Before any installation mutation, arm failure custody in `prepared`.
    /// It must remain armed through Sampling readiness: failure, cancellation,
    /// peer rejection or unwind after mutation must retain/recover or fence the
    /// exact session. Drop must not perform unproved completion or collectives.
    /// All fallible publication and native settlement must finish in this hook.
    fn install_text_resume<C: TokenFilterController>(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
        sampling: &mut Self::TextGenerationState,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), Self::Error>;

    /// Finish-only extraction after Sampling Ready, called exactly once by the
    /// shared driver. Disarms successful installation and yields the fresh
    /// ordinary preparation; it must not allocate, copy, submit or fail.
    /// Borrowing keeps staged custody in the ordered core owner if this hook
    /// unwinds. Keep custody there until returned preparation/state owns it;
    /// never drop the last authority before their payloads on unwind.
    fn finish_text_resume(
        prepared: &mut Self::ResumePreparation,
    ) -> (Self::TextPreparation, Self::DisplacedState);
}

impl<B: TextResumeBackend, C: TokenFilterController> TextGenerationMachine<B, C> {
    pub(super) fn from_resume(
        runtime: &mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Self>, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_resume_with_host(
            runtime,
            saved,
            config,
            controller,
            cancellation,
            None,
            &OriginalTextResumeOptions::new(OriginalTextResumeKind::Restore),
        )
        .map(|result| {
            result.map(|(machine, displaced)| {
                drop(displaced);
                machine
            })
        })
    }

    fn from_resume_with_host(
        runtime: &mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
        host: Option<&HostPreparationAuthority>,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<Option<(Self, B::DisplacedState)>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        let step_context = TextStepContext::new();
        step_context
            .validate()
            .map_err(ControlledTextGenerationError::Preparation)?;
        let Some((
            (
                controller,
                prompt,
                backend_state,
                prepared_sequence,
                preparation,
                preparation_control,
            ),
            displaced,
        )) = preparation::prepare_resume(
            runtime,
            saved,
            config.clone(),
            controller,
            &step_context,
            cancellation,
            host,
            options,
        )?
        else {
            return Ok(None);
        };
        Ok(Some((
            Self {
                capture_source: B::text_resume_capture_source(&backend_state).cloned(),
                intervention_source: B::text_resume_intervention_source(&backend_state).cloned(),
                resume_host: host.cloned(),
                prepared_sequence,
                controller,
                step: if options.terminal {
                    drop(prompt);
                    None
                } else {
                    Some(PendingTextInput::Prefill(prompt))
                },
                completions: Vec::new(),
                remaining_tokens: config.sampling().max_new_tokens,
                branch_owner: step_context.run_identity().clone(),
                branch_fenced: false,
                step_context,
                backend_state,
                preparation,
                preparation_control,
            },
            displaced,
        )))
    }
}

impl<'a, B: TextResumeBackend, C: TokenFilterController> ControlledTextGeneration<'a, B, C> {
    /// Prepares a fresh ordinary machine using this final owned controller.
    /// Zero output or cancellation returns no source; neither copies nor installs
    /// saved state. Positive requests require complete backend resume support.
    pub fn resume_saved(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Self>, ControlledTextGenerationError<B::Error, C::Error>> {
        let inner =
            TextGenerationMachine::from_resume(runtime, saved, config, controller, cancellation)?;
        Ok(inner.map(|inner| Self { runtime, inner }))
    }
}

impl<'a, B: TextResumeBackend, C: TokenFilterController> ControlledTextGeneration<'a, B, C> {
    /// Starts the same fresh driver through the backend's original admission
    /// hook. Host custody must come from the independently accepted destination
    /// constructor plan; it grants no native work or source-copy permission.
    /// Classification stays original even when the facade owns the copied cursor.
    pub fn resume_saved_original(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
        host: &HostPreparationAuthority,
    ) -> Result<Option<Self>, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::resume_saved_original_with_options(
            runtime,
            saved,
            config,
            controller,
            cancellation,
            host,
            &OriginalTextResumeOptions::new(OriginalTextResumeKind::Restore),
        )
    }

    /// Resume with explicit restoration or independently admitted branch
    /// semantics, preserving the same preparation and generation worker.
    pub fn resume_saved_original_with_options(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
        host: &HostPreparationAuthority,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<Option<Self>, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::resume_saved_original_with_displaced(
            runtime,
            saved,
            config,
            controller,
            cancellation,
            host,
            options,
        )
        .map(|result| {
            result.map(|(generation, displaced)| {
                drop(displaced);
                generation
            })
        })
    }

    /// Same admitted resume, retaining the actual displaced native owner for
    /// serial branch composition. No state copy or second admission is added.
    pub fn resume_saved_original_with_displaced(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        controller: C,
        cancellation: &crate::GenerationCancellationToken,
        host: &HostPreparationAuthority,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<Option<(Self, B::DisplacedState)>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        let inner = TextGenerationMachine::from_resume_with_host(
            runtime,
            saved,
            config,
            controller,
            cancellation,
            Some(host),
            options,
        )?;
        Ok(inner.map(|(inner, displaced)| (Self { runtime, inner }, displaced)))
    }
}

/// Fixed original resume constructor/readiness transports, excluding backend
/// payloads, controller allocations and native completion-vector backing.
pub fn text_resume_control_bytes<B: TextResumeBackend, C: TokenFilterController>() -> Option<usize>
{
    use std::mem::size_of;
    [
        preparation::resume_control_bytes::<B, C>()?,
        size_of::<TextGenerationMachine<B, C>>(),
        size_of::<ControlledTextGeneration<'_, B, C>>(),
        size_of::<TextStepContext>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<OriginalTextResumeOptions<'_>>(),
        size_of::<Option<crate::capture::SharedCapturePlan>>(),
        size_of::<Option<crate::intervention::SharedInterventionPlan>>(),
        size_of::<Option<HostPreparationAuthority>>(),
        size_of::<
            Result<
                Option<ControlledTextGeneration<'_, B, C>>,
                ControlledTextGenerationError<B::Error, C::Error>,
            >,
        >(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

impl<'a, B: TextResumeBackend> TextGeneration<'a, B> {
    /// Resumes through the same fresh preparation and machine as controlled
    /// generation. Zero output/cancellation performs no saved-source copy.
    pub fn resume_saved(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Self>, BackendFailure> {
        Self::resume_saved_with_token_filter(runtime, saved, config, TokenFilter::All, cancellation)
    }

    /// As above, owning the final fixed validity filter before cold admission.
    pub fn resume_saved_with_token_filter(
        runtime: &'a mut ModelRuntime<B>,
        saved: &B::ResumeSource,
        config: TextGenerationConfig,
        filter: TokenFilter,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Self>, BackendFailure> {
        let inner = TextGenerationMachine::from_resume(
            runtime,
            saved,
            config,
            FixedTokenFilter(filter),
            cancellation,
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(inner.map(|inner| Self { runtime, inner }))
    }
}
