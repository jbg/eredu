//! Ownership and submission of an erased, already-bound text executable.

use crate::executable_binding::ErasedTextExecutable;
use eredu_core::backend::{TextExecutableTypes, TextSessionNative};
use std::{
    any::Any,
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    parameter_operations::{
        ParameterModelIdentity, ParameterOperationBinding, ParameterOperationCoordinator,
    },
    prefill::ErasedPrefillCursor,
    run_preparation::{
        TextPreparationAgreementError, TextPreparationCoordinator, TEXT_PREPARATION_WORDS,
    },
    ActivationObserver,
};
use std::sync::Arc;

/// Portable loaded capability and memory accounting.
pub mod capability;
/// Admission and retained observer setup for text capture.
pub mod capture;
/// Portable causal submissions and observed input admission.
pub mod causal;
mod causal_generation;
/// Loaded ordinary, continuation, and speculative memory forecasts.
pub mod forecast;
mod input;
mod mode;
pub use input::{prepare_plain_text_prompt, PreparedTextPrompt, TextPromptView};
mod erased_state;
/// Read-only session telemetry and distributed sampling mechanisms.
pub mod operations;
/// Loaded parameter access and native tensor mechanism contracts.
pub mod parameters;
mod prompt_cache;
mod shared_backbone;
pub use erased_state::{
    ErasedTextState, ErasedTextStateAccess, InstalledTextState, TextSnapshotStorage,
};
mod speculative_observation;
mod state;
pub use speculative_observation::SpeculativeObservationModel;
mod request;
mod snapshot;
pub use request::{TextRequestFor, TextRequestState, TextSamplingFor, TextSamplingState};
pub use snapshot::{SnapshotPendingGeometry, SnapshotRefinementFacts, TextSnapshotMechanisms};
/// Dispatch for installed non-causal text modes.
pub mod specialized;
use eredu_core::backend::{TextSubmissionScope, TextTransactionError, TextTransactionMechanisms};
use eredu_core::capture::CaptureRequestShape;
use eredu_core::observation::ObserverTensorMapping;
use eredu_core::parameters::ParameterError;
use eredu_core::run_preparation::{
    TextPreparationOutcome, TextPreparationStage, TextPreparationStatus, TextPreparationUsage,
};
use eredu_core::session_authority::SessionRetirement;
use eredu_core::TextSessionLifecycle;
use eredu_core::{
    BackendFailure, BackendFailureKind, BackendProvider, BackendSession, Completion,
    ObservationSet, PreparedModel, SessionAdmission, SessionAuthority, SessionAuthorityError,
    SessionCapabilities, Submission, SubmissionLease,
};
pub use mode::TextModeState;

/// Selected mutable-state placement and realized precision for an installed text model.
pub struct TextSessionLayout {
    residency: crate::CacheResidencyPolicy,
    floating_state_dtype_bytes: std::num::NonZeroU8,
}

impl TextSessionLayout {
    /// Adopts placement and precision facts from the prepared executable.
    pub fn new(
        residency: crate::CacheResidencyPolicy,
        floating_state_dtype_bytes: std::num::NonZeroU8,
    ) -> Self {
        Self {
            residency,
            floating_state_dtype_bytes,
        }
    }

    /// Returns the admitted lifetime policy for mutable model state.
    pub const fn residency(&self) -> &crate::CacheResidencyPolicy {
        &self.residency
    }

    /// Returns the unmodified floating state precision.
    pub const fn floating_state_dtype_bytes(&self) -> std::num::NonZeroU8 {
        self.floating_state_dtype_bytes
    }
}

/// The selected text session. It owns a single executable for all execution
/// and state operations.
pub struct TextSessionDriver<R, N, G = eredu_core::backend::AdvancedTextGeneration> {
    generation: std::marker::PhantomData<fn() -> G>,
    /// The resource root shared only with outstanding native work.
    payload: Rc<R>,
    /// Native submission mechanisms and session metadata.
    native: N,
    capture: capture::TextCaptureCatalog,
    capabilities: SessionCapabilities,
    authority: RefCell<SessionAuthority>,
    preparation: Option<Arc<TextPreparationCoordinator>>,
    parameter_control: Option<(Arc<ParameterOperationCoordinator>, ParameterModelIdentity)>,
    parameter_usage: Rc<std::cell::Cell<eredu_core::capture::CaptureUsage>>,
    parameter_epoch: u64,
    parameter_overlay: Option<parameters::ActiveParameterOverlay>,
    layout: TextSessionLayout,
    speculative_capture_layouts: crate::capture::SpeculativePartitionLayouts,
    intervention_identity: String,
}

impl<R: 'static, N, G> TextSessionDriver<R, N, G> {
    /// Retains the same monotone agreement owner for speculative execution.
    pub fn preparation_coordinator(&self) -> Option<&Arc<TextPreparationCoordinator>> {
        self.preparation.as_ref()
    }

    fn parameter_control(
        &self,
    ) -> Result<&(Arc<ParameterOperationCoordinator>, ParameterModelIdentity), ParameterError> {
        self.parameter_control.as_ref().ok_or_else(|| {
            ParameterError::Unsupported(
                "parameter operations require selected bounded completion".into(),
            )
        })
    }

    /// Returns the coordinator and identity for exchanging loaded parameter metadata.
    fn parameter_catalog_operation(
        &self,
        epoch: u64,
    ) -> Result<
        (
            Arc<ParameterOperationCoordinator>,
            ParameterOperationBinding,
        ),
        ParameterError,
    > {
        let (owner, identity) = self.parameter_control()?;
        Ok((Arc::clone(owner), identity.catalog_binding(epoch)))
    }

    /// Binds a parameter read or edit to the current loaded model and parameter version.
    fn parameter_operation(
        &self,
        epoch: u64,
    ) -> Result<
        (
            Arc<ParameterOperationCoordinator>,
            ParameterOperationBinding,
        ),
        ParameterError,
    > {
        let (owner, identity) = self.parameter_control()?;
        let prepared = self
            .model_discovery()
            .ok_or_else(|| ParameterError::Unsupported("no retained prepared identity".into()))?;
        let binding = identity.binding(
            &prepared.capture()?.artifact_identity,
            prepared.execution_identity(),
            epoch,
        )?;
        Ok((Arc::clone(owner), binding))
    }

    /// Stable identity of this installed model session for intervention admission.
    pub fn intervention_identity(&self) -> &str {
        &self.intervention_identity
    }

    /// Layout retained for state cache, control, and memory admission.
    pub const fn state_layout(&self) -> &TextSessionLayout {
        &self.layout
    }

    /// Retained lazy activation placements for the installed speculative routes.
    pub fn speculative_capture_layouts(&self) -> &crate::capture::SpeculativePartitionLayouts {
        &self.speculative_capture_layouts
    }

    /// Retained architecture discovery for this installed model.
    pub fn model_discovery(
        &self,
    ) -> Option<&std::sync::Arc<dyn crate::executable_binding::discovery::LoadedModelDiscovery>>
    {
        self.capture.discovery.as_ref()
    }

    /// Inspects the resources retained for this exact session.
    pub fn resources(&self) -> &R {
        &self.payload
    }

    /// Borrows resources exclusively at a settled submission boundary.
    pub fn exclusive_resources(&mut self) -> Option<&mut R> {
        self.authority.get_mut().require_idle().ok()?;
        Rc::get_mut(&mut self.payload)
    }

    /// Shares the resource lifetime without granting model access to recovery.
    pub fn retention_root(&self) -> Rc<dyn Any> {
        self.payload.clone()
    }

    /// Borrows the native mechanism state for this session.
    pub fn native(&self) -> &N {
        &self.native
    }
}

/// Prompt storage capable of retaining an independently prepared cursor.
pub trait TextPrefillInput {
    /// Borrowed prepared input accepted by the erased executable.
    type Borrowed<'a>
    where
        Self: 'a;

    /// Borrows the prepared payload for a synchronous invocation.
    fn borrowed(&self) -> Self::Borrowed<'_>;

    /// Exact prepared prefix identity retained by the input.
    fn prompt_cache_identity(&self) -> Option<&crate::PreparedInputCacheIdentity> {
        None
    }

    /// Borrows the cursor if this prompt already owns one.
    fn prefill_cursor(&self) -> Option<&ErasedPrefillCursor>;

    /// Borrows the retained cursor to advance one admitted step.
    fn prefill_cursor_mut(&mut self) -> Option<&mut ErasedPrefillCursor>;

    /// Transfers ownership from the original ingress to the retained cursor.
    fn install_prefill_cursor(&mut self, cursor: ErasedPrefillCursor);
}

/// Native storage operations for a plain token prompt that can split without a retained cursor.
pub trait PlainTextPrefill<B: BackendProvider>: Sized {
    /// Token count for a plain prompt, when it can be split.
    fn plain_token_count(&self) -> Option<usize>;

    /// Creates a prefix and suffix. The suffix retains the original prompt identity.
    fn split_plain_prompt(
        &self,
        backend: &B,
        at: std::num::NonZeroUsize,
    ) -> Result<(Self, Self), B::Error>;
}

use eredu_core::speculative::{
    SpeculativeCapability, SpeculativeDraftSource, SpeculativeTargetFacts,
};

/// The erased model entry points used by the portable text driver.
pub trait TextExecutable<M: eredu_core::backend::TextExecutableTypes> {
    /// Prepared cached token positions.
    type Decode;
    /// Tensor values exposed to portable observers.
    type ObservationTensor;
    /// Converts invocation tensor handles into the types exposed to observers.
    type ObserverMapping: ObserverTensorMapping<
        ObserverTensor = Self::ObservationTensor,
        InvocationError = Self::Error,
        ObserverError = Self::Error,
    >;
    /// Score value.
    type Scores;
    /// Tensor execution context.
    type Context;
    /// Invocation failure.
    type Error: From<BackendFailure>;
    /// Transaction-local attention state used by tentative execution.
    type AttentionState;

    /// Exact capabilities of the installed executable.
    fn capability_estimate(&self) -> &crate::CapabilityEstimate;

    /// Architecture identity retained at installation.
    fn effective_model_type(&self) -> &str;

    /// Internal activation execution realized by the selected prediction route.
    fn partition_activation_execution(
        &self,
    ) -> Option<crate::speculative::SpeculativeActivationExecution> {
        None
    }

    /// Returns decoder spans and media geometry for the retained input route.
    fn prepared_input_plans(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
    ) -> Result<
        Vec<crate::executable_binding::retained_session::PreparedInputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::executable_binding::erased_text::text_input_plans::<M>(
            self.effective_model_type(),
            input,
        )
    }

    /// Counts admitted model positions in an owned prepared prompt.
    fn prepared_input_positions(&self, _input: TextPromptView<'_, M::InputTensor>) -> Option<u64> {
        None
    }

    /// Whether an embedded prediction extension is installed.
    fn has_embedded_prediction(&self) -> bool {
        false
    }

    /// Exact geometry of the installed block refinement program.
    fn block_refinement_geometry(
        &self,
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
        None
    }

    /// Whether the installed executable accepts the final-position prefill invocation.
    fn accepts_final_position_prefill(&self) -> bool {
        false
    }

    /// Whether the selected route projects only the last prefill position.
    fn projects_final_prefill_position(&self) -> bool {
        false
    }

    /// Executes the final-position prefill invocation.
    fn prefill_final_position(
        &mut self,
        _input: TextPromptView<'_, M::InputTensor>,
        _context: &Self::Context,
    ) -> Result<Self::Scores, Self::Error> {
        Err(unsupported_text_invocation("final-position prefill").into())
    }

    /// Executes prefill with full observation geometry or a local preparation rejection.
    fn prefill_result_with_observer(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, Self::Error>,
        mask: Option<&<Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor>,
        capture: Option<CaptureRequestShape>,
        context: &Self::Context,
        observer: &mut dyn ActivationObserver<
            <Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
            Self::Error,
        >,
    ) -> Result<Self::Scores, Self::Error>;

    /// Retained cache identity for the state owned by this executable.
    fn prompt_cache_model_identity(&self) -> Option<&eredu_core::cache::PromptCacheModelIdentity> {
        None
    }

    /// Borrows the portable persistence driver for the installed state.
    fn prompt_cache<'a>(
        &'a mut self,
        _context: &'a Self::Context,
    ) -> Option<crate::replicated_session::TextPromptCache<'a, Self::Context>> {
        None
    }

    /// Derives the persistence identity for a plain token prefix.
    fn prompt_cache_prefix_identity(
        &self,
        prefix: &[u32],
        input: Option<&crate::PreparedInputCacheIdentity>,
    ) -> String {
        input.map_or_else(
            || eredu_core::cache::prompt_cache_token_fingerprint(prefix),
            |identity| identity.prefix_content_fingerprint().to_owned(),
        )
    }

    /// Exportable byte extent, when the executable can enumerate it directly.
    fn prompt_cache_storage_bytes(&self) -> Result<Option<u64>, BackendFailure> {
        Ok(None)
    }

    /// Whether this executable can retain a cursor for the prepared input.
    fn retains_prefill_cursor(&self) -> bool {
        false
    }

    /// Whether the installed executable accepts split plain-token input.
    fn accepts_split_prefill(&self) -> bool {
        false
    }

    /// Opens a retained cursor over the complete input request.
    fn start_prefill_cursor(
        &mut self,
        _input: TextPromptView<'_, M::InputTensor>,
        _max_tokens: std::num::NonZeroUsize,
        _capture: Option<CaptureRequestShape>,
        _context: &Self::Context,
    ) -> Result<Option<ErasedPrefillCursor>, Self::Error> {
        Ok(None)
    }

    /// Prepares ingress in the cursor without committing any decoder positions.
    fn advance_prefill_ingress(
        &mut self,
        _cursor: &mut ErasedPrefillCursor,
        _context: &Self::Context,
        _observer: &mut dyn ActivationObserver<
            <Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
            Self::Error,
        >,
    ) -> Result<(), Self::Error> {
        Err(unsupported_text_invocation("retained prefill ingress").into())
    }

    /// Advances one admitted decoder phase of a retained prefill.
    fn advance_prefill_cursor(
        &mut self,
        _cursor: &mut ErasedPrefillCursor,
        _context: &Self::Context,
        _observer: &mut dyn ActivationObserver<
            <Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
            Self::Error,
        >,
    ) -> Result<Self::Scores, Self::Error> {
        Err(unsupported_text_invocation("retained prefill decode").into())
    }

    /// Executes cached decode on the installed model.
    fn decode_result_with_observer(
        &mut self,
        input: Result<&Self::Decode, Self::Error>,
        context: &Self::Context,
        observer: &mut dyn ActivationObserver<
            <Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
            Self::Error,
        >,
    ) -> Result<Self::Scores, Self::Error>;

    /// Creates an independent causal state for the installed executable.
    fn prepare_autoregressive_cache(&mut self) -> Result<M::PredictionState, M::Error> {
        Err(unsupported_text_invocation("ordinary prediction state").into())
    }

    /// Exchanges the installed causal state with an independent, settled state.
    fn exchange_autoregressive_cache(
        &mut self,
        _cache: &mut M::PredictionState,
        _context: &M::Context,
    ) -> Result<(), M::Error> {
        Err(unsupported_text_invocation("autoregressive prediction exchange").into())
    }

    /// Encodes a complete prepared context in the supplied causal state.
    fn prepared_context_forward(
        &mut self,
        _input: TextPromptView<'_, M::InputTensor>,
        _cache: &mut M::PredictionState,
        _context: &M::Context,
        _observer: &mut dyn ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<(M::Scores, usize), M::Error> {
        Err(unsupported_text_invocation("prepared causal context").into())
    }

    /// Evaluates the tentative canvas without advancing committed model state.
    fn attention_forward_with_observer(
        &mut self,
        _canvas: &<Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
        _feedback: Option<&<Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor>,
        _invocation: eredu_core::decoding::AttentionInvocation,
        _state: Option<&mut Self::AttentionState>,
        _prefill: bool,
        _context: &Self::Context,
        _observer: &mut dyn ActivationObserver<
            <Self::ObserverMapping as ObserverTensorMapping>::InvocationTensor,
            Self::Error,
        >,
    ) -> Result<Self::Scores, Self::Error> {
        Err(unsupported_text_invocation("attention invocation").into())
    }

    /// Whether reset must reach the installed partition cohort.
    fn has_partition_control(&self) -> bool {
        false
    }

    /// Resets locally owned request state while retaining installed parameters.
    fn reset_local(&mut self) -> Result<(), Self::Error>;

    /// Resets request state through the selected partition protocol.
    fn reset_partitioned(&mut self) -> Result<(), Self::Error>;

    /// Whether this rank publishes scores to the caller.
    fn partition_public_output(&self) -> bool {
        true
    }
}

fn unsupported_text_invocation(operation: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        std::io::Error::new(std::io::ErrorKind::Unsupported, operation),
    )
    .with_operation(operation)
}

impl<R, N, G> TextSessionDriver<R, N, G> {
    /// Observes submission exclusion without progressing native work.
    pub fn submission_idle(&self) -> Result<(), SessionAuthorityError> {
        self.authority.borrow().require_idle()
    }
}

impl<R, N: SessionRetirement, G> TextSessionDriver<R, N, G> {
    /// Progresses retirement and verifies that the session is available.
    pub fn ensure_no_submission_in_flight(&self) -> Result<(), N::Error> {
        self.native.refresh_native_session()?;
        self.submission_idle()
            .map_err(N::submission_authority_error)
    }

    /// Grants the exact lease that native completion must retain until retirement.
    pub fn acquire_submission(&self) -> Result<SubmissionLease, N::Error> {
        self.native.refresh_native_session()?;
        self.authority
            .borrow_mut()
            .begin_submission()
            .map_err(N::submission_authority_error)
    }

    /// Reports native lifecycle failure without releasing submission authority.
    pub fn lifecycle_failure(&self, error: N::Error) -> BackendFailure {
        let kind = if self.native.native_session_fenced() {
            BackendFailureKind::InvalidSession
        } else if self.submission_idle().is_err() {
            BackendFailureKind::Busy
        } else {
            return BackendFailure::from_error(error);
        };
        BackendFailure::new(kind, error)
    }
}

impl<R, N, M, T, V, Processor, G> TextSessionDriver<R, N, G>
where
    R: DerefMut<Target = TextSessionResources<M, T, V, Processor>> + 'static,
    N: TextTransactionMechanisms,
{
    /// Rejects stale generation state before an operation can mutate the model.
    pub fn validate_parameter_epoch(&self, saved: &mut Option<u64>) -> Result<(), N::Error>
    where
        N::Error: From<eredu_core::BackendError>,
    {
        let current = self.parameter_epoch;
        if saved.is_some_and(|epoch| epoch != current) {
            return Err(eredu_core::BackendError::Execution {
                session: "text-generation".into(),
                operation: "validate parameter version".into(),
                message: "generation state belongs to an earlier parameter version; prepare a fresh exact-token prefill".into(),
            }.into());
        }
        *saved = Some(current);
        Ok(())
    }

    /// Advances a plain prefix and commits the suffix only after exact completion.
    fn advance_plain_prefill_prefix<B, P>(
        runtime: &mut eredu_core::ModelRuntime<B, Self>,
        input: &mut P,
        max_tokens: std::num::NonZeroUsize,
        epoch: &mut Option<u64>,
    ) -> Result<bool, B::Error>
    where
        B: BackendProvider<Error = N::Error>,
        Self: BackendSession<B, PrefillInput = P>,
        P: PlainTextPrefill<B>,
        B::Error: From<eredu_core::BackendError>,
    {
        if !input
            .plain_token_count()
            .is_some_and(|count| count > max_tokens.get())
        {
            return Ok(false);
        }
        runtime.session().validate_parameter_epoch(epoch)?;
        let (prefix, suffix) = input.split_plain_prompt(runtime.backend(), max_tokens)?;
        runtime.prefill(prefix)?.completion.wait()?;
        *input = suffix;
        Ok(true)
    }

    /// Runs one settled transaction on the installed model.
    pub fn with_model_transaction<U>(
        &mut self,
        operation: impl FnOnce(&mut M) -> Result<U, N::Error>,
    ) -> Result<U, N::Error> {
        self.with_prepared_model_transaction(|local| local, |model, ()| operation(model))
    }

    /// Agrees local acquisition before any rank can mutate the model. A
    /// rejected, unused scope still establishes exact terminal evidence.
    pub fn with_prepared_model_transaction<U, A>(
        &mut self,
        prepare: impl FnOnce(Result<(), N::Error>) -> Result<A, N::Error>,
        operation: impl FnOnce(&mut M, A) -> Result<U, N::Error>,
    ) -> Result<U, N::Error> {
        let acquired = self
            .acquire_submission()
            .and_then(|lease| self.native.open_transaction(lease))
            .and_then(|scope| {
                if Rc::get_mut(&mut self.payload).is_none() {
                    let error = N::transaction_error(TextTransactionError::Busy);
                    return Err(scope.settle(self.payload.clone(), Ok(error))?);
                }
                Ok(ActiveTextInvocation::<_, _, N::Scores, N::Error> {
                    root: &mut self.payload,
                    scope: Some(scope),
                    submission: std::marker::PhantomData,
                })
            });
        let (guard, local) = match acquired {
            Ok(guard) => (Some(guard), Ok(())),
            Err(error) => (None, Err(error)),
        };
        let prepared = prepare(local);
        let Some(guard) = guard else {
            return prepared
                .and_then(|_| Err(N::transaction_error(TextTransactionError::AbsentAuthority)));
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Err(guard.settle(Ok(error))?),
        };
        let model = &mut Rc::get_mut(guard.root)
            .expect("exclusive resource root")
            .model;
        let result = operation(model, prepared);
        guard.settle(result)
    }
}

/// Selected input for a cached decode, including local admission failure.
pub enum TextDecodeInput<Prepared, Sampled, E> {
    /// Already prepared positions or a local rejection the executable must observe.
    Prepared(Result<Prepared, E>),
    /// One explicit committed token ID.
    TokenId(u32),
    /// A native sample requiring a cached decode axis after admission.
    Sampled(Result<Sampled, E>),
}

/// Portable prepared prompt over the backend's native input tensor.
pub type TextPromptFor<B> =
    PreparedTextPrompt<<B as eredu_core::TextGenerationBackend>::InputTensor>;

/// The resident executable and the native resources that must retire with it.
pub struct TextSessionResources<E, N, V, P> {
    /// The installed executable, including its model state.
    pub model: E,
    /// Selected portable processor program.
    pub processor: P,
    /// Portable overlay ownership, retained with the executable until native retirement.
    pub parameter_overlay: Option<InstalledTextParameterOverlay<V>>,
    /// Native resources whose lifetime is tied to the loaded executable.
    pub retained: N,
}

/// Reversible parameter handles retained by the installed text driver.
pub enum InstalledTextParameterOverlay<V> {
    /// Local edit, whose replacement is already owned by the installed model.
    Local {
        /// Immutable originals needed to undo the edit.
        originals: std::collections::BTreeMap<String, V>,
    },
    /// Rank-local portion of a collectively published edit.
    Partitioned {
        /// Immutable originals for a coordinated removal.
        originals: std::collections::BTreeMap<String, V>,
        /// Published values needed to roll back an interrupted removal.
        published: std::collections::BTreeMap<String, V>,
        /// Admitted resource bound for the replacement execution state.
        reset_estimate: eredu_core::execution_control::SnapshotEstimate,
    },
}

impl<V> InstalledTextParameterOverlay<V> {
    /// Original tensor handles for either kind of installed overlay.
    pub fn originals(&self) -> &std::collections::BTreeMap<String, V> {
        match self {
            Self::Local { originals } | Self::Partitioned { originals, .. } => originals,
        }
    }
}

/// Portable executable profile derived from an installed native session's values.
pub type SessionTextTypes<B, N> =
    crate::executable_binding::PortableTextTypes<<N as TextSessionNative<B>>::TextValues>;

/// Resource root assembled by portable installation for one backend and mechanism set.
pub type TextSessionResourcesFor<B, N> = TextSessionResources<
    <N as TextSessionMechanisms<B>>::Model,
    <N as eredu_core::backend::TextSessionNative<B>>::Retained,
    <N as eredu_core::backend::TextSessionNative<B>>::ParameterValue,
    crate::executable_binding::PreparedTextProcessor<SessionTextTypes<B, N>>,
>;

/// Portable model resource types derived from the selected native contracts.
pub trait TextSessionMechanisms<B: BackendProvider>: TextSessionNative<B> {
    /// Owned prompt tracked by portable prefill.
    type Prefill: for<'a> TextPrefillInput<
            Borrowed<'a> = TextPromptView<
                'a,
                <SessionTextTypes<B, Self> as TextExecutableTypes>::InputTensor,
            >,
        > + 'static;
    /// Erased executable and its portable reporting operations.
    type Executable: crate::executable_binding::ErasedTextExecutable<SessionTextTypes<B, Self>>
        + ?Sized
        + 'static;
    /// Owning executable handle.
    type Model: AsRef<Self::Executable>
        + AsMut<Self::Executable>
        + capture::TextCaptureGeometry
        + 'static;
    /// One installed portable payload protected by native retirement.
    type Resources: DerefMut<Target = TextSessionResourcesFor<B, Self>> + 'static;

    /// Preserves a portable preparation failure at the operation boundary.
    fn preparation_error(error: BackendFailure) -> B::Error;

    /// Retains a fully composed portable model payload through native retirement.
    fn retain_model_resources(
        resources: TextSessionResourcesFor<B, Self>,
    ) -> <Self as TextSessionMechanisms<B>>::Resources;
}

impl<B, N> TextSessionMechanisms<B> for N
where
    B: BackendProvider<Error: From<BackendFailure>>,
    N: TextSessionNative<B>,
    N::Retained: 'static,
{
    type Prefill = PreparedTextPrompt<<SessionTextTypes<B, N> as TextExecutableTypes>::InputTensor>;
    type Executable = dyn crate::executable_binding::ErasedTextExecutable<SessionTextTypes<B, N>>;
    type Model = crate::executable_binding::InstalledTextExecutable<SessionTextTypes<B, N>>;
    type Resources = N::Retired<TextSessionResourcesFor<B, Self>>;

    fn preparation_error(error: BackendFailure) -> B::Error {
        error.into()
    }

    fn retain_model_resources(
        resources: TextSessionResourcesFor<B, Self>,
    ) -> <Self as TextSessionMechanisms<B>>::Resources {
        N::retain_resources(resources)
    }
}

/// Portable text driver instantiated with mechanism types chosen at composition.
pub type PreparedTextDriver<B, M> = TextSessionDriver<
    <M as TextSessionMechanisms<B>>::Resources,
    M,
    <M as eredu_core::backend::TextSessionNative<B>>::Generation,
>;

struct ActiveTextInvocation<'a, R, S, T, E>
where
    R: 'static,
    S: TextSubmissionScope<T, E>,
{
    root: &'a mut Rc<R>,
    scope: Option<S>,
    submission: std::marker::PhantomData<fn() -> (T, E)>,
}

impl<R, S, T, E> Drop for ActiveTextInvocation<'_, R, S, T, E>
where
    R: 'static,
    S: TextSubmissionScope<T, E>,
{
    fn drop(&mut self) {
        if let Some(scope) = self.scope.take() {
            scope.abandon(self.root.clone());
        }
    }
}

type DriverError<B> = <B as BackendProvider>::Error;
type ExecutableFor<B, N> = <N as TextSessionMechanisms<B>>::Executable;
type TextDecodeValue<B, N> =
    <ExecutableFor<B, N> as TextExecutable<SessionTextTypes<B, N>>>::Decode;
type TextScores<B, N> = <ExecutableFor<B, N> as TextExecutable<SessionTextTypes<B, N>>>::Scores;
type TextObserver<'a, B, N> = dyn ActivationObserver<
        <ExecutableFor<B, N> as TextExecutable<SessionTextTypes<B, N>>>::ObservationTensor,
        DriverError<B>,
    > + 'a;

fn invocation_observer<
    'a,
    M: eredu_core::backend::TextExecutableTypes,
    E: TextExecutable<M> + ?Sized,
>(
    observer: &'a mut dyn ActivationObserver<E::ObservationTensor, E::Error>,
) -> crate::inspection::MappedActivationObserver<
    'a,
    dyn ActivationObserver<E::ObservationTensor, E::Error> + 'a,
    E::ObserverMapping,
> {
    crate::inspection::MappedActivationObserver::new(observer)
}
type RawTextSubmission<B, N> =
    Submission<TextScores<B, N>, <N as eredu_core::backend::TextSessionNative<B>>::Completion>;

impl<R: 'static, N, G> TextSessionDriver<R, N, G> {
    fn shared_backbone_policy<B>(
        &self,
    ) -> Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: Deref<Target = TextSessionResourcesFor<B, N>>,
    {
        self.payload
            .model
            .as_ref()
            .capability_estimate()
            .shared_backbone_draft()
    }

    fn prefill_chunking_support<B>(&self) -> Result<(), &'static str>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: Deref<Target = TextSessionResourcesFor<B, N>>,
    {
        let resources = self.resources();
        let executable = resources.model.as_ref();
        if executable.retains_prefill_cursor() {
            return Ok(());
        }
        if N::control_transport(&resources.retained).is_some() {
            return Err("distributed execution retains a complete prefill pass");
        }
        if executable.accepts_split_prefill() {
            Ok(())
        } else {
            Err("selected architecture or prediction extension retains a complete prefill pass")
        }
    }

    /// Resets the executable and waits for the native scope to settle.
    pub fn reset<B>(&mut self, backend: &B) -> Result<(), B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let invocation = self.begin::<B>(backend, false)?;
        let executable = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let result = if executable.has_partition_control() {
            executable.reset_partitioned()
        } else {
            executable.reset_local()
        };
        invocation.settle(result)
    }

    /// Agrees a portable decoding command through this session's retained transport.
    fn coordinate_decoding_control<B>(
        &self,
        local: eredu_core::decoding::DecodingCoordination,
    ) -> Result<eredu_core::decoding::DecodingCoordination, TextPreparationAgreementError>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let Some(transport) = N::control_transport(&self.payload.retained) else {
            return Ok(local);
        };
        TextPreparationCoordinator::require(self.preparation.as_deref())?
            .coordinate_decoding_control(transport, local)
    }

    /// Submits prefill, including a local rejection that the model must observe.
    pub fn advance_prefill<B>(
        &mut self,
        backend: &B,
        input: Result<N::Prefill, B::Error>,
    ) -> Result<Submission<N::Output, N::Completion>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let (submission, publish) = self.run_prefill(backend, input, None, None)?;
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }

    /// Submits cached decode under the same authority as input preparation.
    pub fn advance_decode<B>(
        &mut self,
        backend: &B,
        input: TextDecodeInput<TextDecodeValue<B, N>, N::SampledDecode, B::Error>,
    ) -> Result<Submission<N::Output, N::Completion>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let (submission, publish) = self.run_decode(backend, input, None)?;
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }

    /// Submits a prompt and a bounded observation on the same transaction.
    pub fn advance_observed_prefill<B>(
        &mut self,
        backend: &B,
        input: Result<N::Prefill, B::Error>,
        capture: Option<CaptureRequestShape>,
        observer: &mut TextObserver<'_, B, N>,
    ) -> Result<RawTextSubmission<B, N>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.run_prefill(backend, input, capture, Some(observer))
            .map(|(submission, _)| submission)
    }

    /// Prepares the next non-final retained step. The final decoder span runs
    /// through the ordinary score-producing prefill path.
    fn prepare_prefill_prefix<B>(
        &mut self,
        backend: &B,
        input: &mut N::Prefill,
        max_tokens: std::num::NonZeroUsize,
        capture: Option<CaptureRequestShape>,
    ) -> Result<Option<crate::prefill::PrefillStep>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        use crate::prefill::PrefillStep;

        if input.prefill_cursor().is_none() {
            let mut invocation = self.begin::<B>(backend, true)?;
            let executable = Rc::get_mut(invocation.root)
                .expect("exclusive resource root")
                .model
                .as_mut();
            let cursor = executable
                .start_prefill_cursor(input.borrowed(), max_tokens, capture, N::context(backend))
                .and_then(|cursor| {
                    invocation.validate_tokens()?;
                    Ok(cursor)
                });
            if let Some(cursor) = invocation.settle(cursor)? {
                input.install_prefill_cursor(cursor);
            }
        }
        let Some(cursor) = input.prefill_cursor() else {
            return Ok(None);
        };
        match cursor
            .next_step()
            .map_err(|error| N::preparation_error(BackendFailure::from(error)))?
        {
            PrefillStep::Complete => Ok(None),
            PrefillStep::Decoder(ref range) if range.end == cursor.token_count() => Ok(None),
            step => Ok(Some(step)),
        }
    }

    /// Runs the selected non-final ingress or decoder step to exact completion.
    fn advance_prefill_prefix<B>(
        &mut self,
        backend: &B,
        cursor: &mut ErasedPrefillCursor,
        step: &crate::prefill::PrefillStep,
        observer: &mut TextObserver<'_, B, N>,
    ) -> Result<(), B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        use crate::prefill::PrefillStep;

        match step {
            PrefillStep::PrepareInput => {
                let mut invocation = self.begin::<B>(backend, true)?;
                let executable = Rc::get_mut(invocation.root)
                    .expect("exclusive resource root")
                    .model
                    .as_mut();
                let result = executable
                    .advance_prefill_ingress(
                        cursor,
                        N::context(backend),
                        &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
                    )
                    .and_then(|()| invocation.validate_tokens())
                    .and_then(|()| observer.finish());
                invocation.settle(result)
            }
            PrefillStep::Decoder(_) => self
                .advance_prefill_step(backend, cursor, observer)?
                .completion
                .wait(),
            PrefillStep::Complete => Ok(()),
        }
    }

    /// Submits one decoder phase for a retained prefill cursor.
    fn advance_prefill_step<B>(
        &mut self,
        backend: &B,
        cursor: &mut ErasedPrefillCursor,
        observer: &mut TextObserver<'_, B, N>,
    ) -> Result<RawTextSubmission<B, N>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let invocation = self.begin::<B>(backend, true)?;
        let executable = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let output = executable.advance_prefill_cursor(
            cursor,
            N::context(backend),
            &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
        );
        let output = output.and_then(|output| {
            observer.finish()?;
            Ok(output)
        });
        invocation.finish(output)
    }

    /// Submits cached decode with an observation on the same transaction.
    pub fn advance_observed_decode<B>(
        &mut self,
        backend: &B,
        input: TextDecodeInput<TextDecodeValue<B, N>, N::SampledDecode, B::Error>,
        observer: &mut TextObserver<'_, B, N>,
    ) -> Result<RawTextSubmission<B, N>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.run_decode(backend, input, Some(observer))
            .map(|(submission, _)| submission)
    }

    /// Submits one attention evaluation of the tentative refinement canvas.
    pub fn advance_refinement<B>(
        &mut self,
        backend: &B,
        canvas: &<N::Executable as TextExecutable<SessionTextTypes<B, N>>>::ObservationTensor,
        feedback: Option<
            &<N::Executable as TextExecutable<SessionTextTypes<B, N>>>::ObservationTensor,
        >,
        observer: &mut TextObserver<'_, B, N>,
    ) -> Result<RawTextSubmission<B, N>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let invocation = self.begin::<B>(backend, true)?;
        let executable = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        type Mapping<B, N> =
            <ExecutableFor<B, N> as TextExecutable<SessionTextTypes<B, N>>>::ObserverMapping;
        let output = executable.attention_forward_with_observer(
            Mapping::<B, N>::borrow_invocation_tensor(canvas),
            feedback.map(Mapping::<B, N>::borrow_invocation_tensor),
            eredu_core::decoding::AttentionInvocation::DIFFUSION,
            None,
            false,
            N::context(backend),
            &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
        );
        let output = output.and_then(|output| {
            observer.finish()?;
            Ok(output)
        });
        invocation.finish(output)
    }

    /// Reserves a submission scope to consume an already settled score value.
    pub fn retain_scores<B>(
        &mut self,
        backend: &B,
        scores: TextScores<B, N>,
    ) -> Result<RawTextSubmission<B, N>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.begin::<B>(backend, true)?.finish(Ok(scores))
    }

    fn run_prefill<B>(
        &mut self,
        backend: &B,
        input: Result<N::Prefill, B::Error>,
        capture: Option<CaptureRequestShape>,
        observer: Option<&mut TextObserver<'_, B, N>>,
    ) -> Result<(RawTextSubmission<B, N>, bool), B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let invocation = self.begin::<B>(backend, true)?;
        let executable = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let mut input = input;
        let context = N::context(backend);
        let observed = observer.is_some();
        let mut noop = crate::NoopObserver;
        let observer = observer.unwrap_or(&mut noop);
        let output = if let Some(cursor) = input
            .as_mut()
            .ok()
            .and_then(TextPrefillInput::prefill_cursor_mut)
        {
            cursor
                .final_decoder_range()
                .map_err(|error| N::preparation_error(BackendFailure::from(error)))
                .and_then(|_| {
                    executable.advance_prefill_cursor(
                        cursor,
                        context,
                        &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
                    )
                })
        } else {
            let borrowed = match input {
                Ok(ref prompt) => Ok(prompt),
                Err(error) => Err(error),
            };
            match (borrowed, capture, observed) {
                (Ok(prompt), None, false) if executable.accepts_final_position_prefill() => {
                    executable.prefill_final_position(prompt.borrowed(), context)
                }
                (input, capture, _) => executable.prefill_result_with_observer(
                    input.map(TextPrefillInput::borrowed),
                    None,
                    capture,
                    context,
                    &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
                ),
            }
        }
        .and_then(|output| {
            observer.finish()?;
            Ok(output)
        });
        let publish = executable.partition_public_output();
        Ok((invocation.finish(output)?, publish))
    }

    fn run_decode<B>(
        &mut self,
        backend: &B,
        input: TextDecodeInput<TextDecodeValue<B, N>, N::SampledDecode, B::Error>,
        observer: Option<&mut TextObserver<'_, B, N>>,
    ) -> Result<(RawTextSubmission<B, N>, bool), B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let invocation = self.begin::<B>(backend, true)?;
        let input = match input {
            TextDecodeInput::Prepared(input) => input,
            TextDecodeInput::TokenId(token) => N::decode_token_id(token, backend),
            TextDecodeInput::Sampled(input) => {
                input.and_then(|token| N::decode_sample(token, backend))
            }
        };
        let executable = Rc::get_mut(invocation.root)
            .expect("exclusive resource root")
            .model
            .as_mut();
        let mut noop = crate::NoopObserver;
        let observer = observer.unwrap_or(&mut noop);
        let output = executable
            .decode_result_with_observer(
                match input {
                    Ok(ref tokens) => Ok(tokens),
                    Err(error) => Err(error),
                },
                N::context(backend),
                &mut invocation_observer::<SessionTextTypes<B, N>, N::Executable>(observer),
            )
            .and_then(|output| {
                observer.finish()?;
                Ok(output)
            });
        let publish = executable.partition_public_output();
        Ok((invocation.finish(output)?, publish))
    }

    fn begin<B>(
        &mut self,
        backend: &B,
        token_validation: bool,
    ) -> Result<ActiveTextInvocation<'_, R, N::Scope<'_>, TextScores<B, N>, B::Error>, B::Error>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        let scope = self
            .native
            .begin(self.acquire_submission()?, token_validation)?;
        let invocation = ActiveTextInvocation {
            root: &mut self.payload,
            scope: Some(scope),
            submission: std::marker::PhantomData,
        };
        if Rc::get_mut(invocation.root).is_none() {
            return Err(N::busy());
        }
        Ok(invocation)
    }
}

impl<R: 'static, S, T, E> ActiveTextInvocation<'_, R, S, T, E>
where
    S: TextSubmissionScope<T, E>,
{
    fn validate_tokens(&mut self) -> Result<(), E> {
        self.scope
            .as_mut()
            .expect("active native scope")
            .validate_tokens()
    }

    fn settle<Value>(mut self, result: Result<Value, E>) -> Result<Value, E> {
        self.scope
            .take()
            .expect("active native scope")
            .settle(self.root.clone(), result)
    }

    fn finish(mut self, output: Result<T, E>) -> Result<Submission<T, S::Completion>, E> {
        self.scope
            .take()
            .expect("active native scope")
            .finish(self.root.clone(), output)
    }
}

impl<B, N, G> eredu_core::backend::ModelSessionInstaller<B>
    for TextSessionDriver<N::Resources, N, G>
where
    B: BackendProvider,
    N: TextSessionMechanisms<B, Generation = G>,
{
    type Model = crate::executable_binding::LoadedTextExecutable<
        N::Model,
        N::Communication,
        crate::executable_binding::PreparedTextProcessor<SessionTextTypes<B, N>>,
        N::Target,
    >;
    type Session = Self;

    fn install_session(
        backend: &B,
        prepared: PreparedModel<Self::Model>,
    ) -> Result<Self, B::Error> {
        N::validate_target(backend, &prepared.target)?;
        let admitted = prepared.capabilities();
        let realized = prepared.executable.as_ref().realized_session_capabilities();
        SessionAdmission::new(admitted)
            .validate(realized)
            .map_err(|_| backend.session_capability_mismatch(admitted, realized))?;
        let crate::executable_binding::LoadedTextExecutable {
            executable,
            communication,
            processor,
            layout,
            discovery,
            target,
        } = prepared.into_inner();
        let (retained, native) = N::attach_native(target, communication)?;
        let resources = N::retain_model_resources(TextSessionResources {
            model: executable,
            processor,
            retained,
            parameter_overlay: None,
        });
        let transport = N::control_transport(&resources.retained);
        let preparation = match transport {
            Some(transport) => N::preparation_admission(transport, TEXT_PREPARATION_WORDS)?
                .map(TextPreparationCoordinator::from_admission)
                .transpose()
                .map_err(|error| N::preparation_error(BackendFailure::from_error(error)))?
                .map(Arc::new),
            None => None,
        };
        let parameter_control = transport
            .filter(|transport| N::parameter_control_supported(transport))
            .map(|transport| {
                let owner = ParameterOperationCoordinator::new(transport)?;
                let identity = owner.register_model()?;
                Ok::<_, ParameterError>((Arc::new(owner), identity))
            })
            .transpose()
            .map_err(|error| N::preparation_error(BackendFailure::from_error(error)))?;
        let mut session = Self {
            generation: std::marker::PhantomData,
            payload: Rc::new(resources),
            native,
            capture: capture::TextCaptureCatalog::new(Some(discovery)),
            capabilities: realized,
            authority: RefCell::new(SessionAuthority::new()),
            preparation,
            parameter_control,
            parameter_usage: Rc::default(),
            parameter_epoch: 0,
            parameter_overlay: None,
            layout,
            speculative_capture_layouts: Default::default(),
            intervention_identity: eredu_core::intervention::new_intervention_session_identity(),
        };
        session.reset(backend)?;
        Ok(session)
    }
}

impl<B, N, G> BackendSession<B> for TextSessionDriver<N::Resources, N, G>
where
    B: BackendProvider,
    N: TextSessionMechanisms<B>,
{
    type PrefillInput = N::Prefill;
    type DecodeInput = TextDecodeValue<B, N>;
    type Output = N::Output;
    type Completion = N::Completion;

    fn speculative_capability(&self) -> SpeculativeCapability {
        use eredu_core::refinement::PredictionScoreKind;
        let executable = self.resources().model.as_ref();
        let capability = executable.capability_estimate();
        let facts = SpeculativeTargetFacts {
            shared_backbone_ready: capability.shared_backbone_draft().is_some(),
            embedded_ready: executable.has_embedded_prediction(),
            score_kind: if executable.block_refinement_geometry().is_some() {
                PredictionScoreKind::RefinementPosition
            } else {
                PredictionScoreKind::CausalConditional
            },
            declared_source: capability.speculative_draft_source(),
            architecture: executable.effective_model_type(),
        };
        match (
            facts.shared_backbone_ready,
            facts.score_kind,
            facts.embedded_ready,
        ) {
            (true, _, _) => SpeculativeCapability::Ready {
                draft_source: SpeculativeDraftSource::SharedBackbone,
            },
            (
                _,
                score_kind @ (eredu_core::refinement::PredictionScoreKind::RefinementPosition
                | eredu_core::refinement::PredictionScoreKind::ConstrainedFinalization),
                _,
            ) => SpeculativeCapability::IncompatibleScores { score_kind },
            (_, _, true) => SpeculativeCapability::Ready {
                draft_source: SpeculativeDraftSource::Embedded,
            },
            _ => match facts.declared_source {
                Some(draft_source @ SpeculativeDraftSource::Separate) => {
                    SpeculativeCapability::Declared { draft_source }
                }
                Some(draft_source) => SpeculativeCapability::Unsupported {
                    draft_source,
                    architecture: facts.architecture.to_owned(),
                },
                None => SpeculativeCapability::Unavailable,
            },
        }
    }

    fn draft_target_identity(&self) -> Option<&str> {
        Some(self.intervention_identity())
    }

    fn capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }

    fn prefill(
        &mut self,
        backend: &B,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error> {
        self.advance_prefill(backend, Ok(input))
    }

    fn decode(
        &mut self,
        backend: &B,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error> {
        self.advance_decode(backend, TextDecodeInput::Prepared(Ok(input)))
    }

    fn observe_output(
        &self,
        backend: &B,
        output: &Self::Output,
    ) -> Result<ObservationSet, B::Error> {
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        let lease = self.acquire_submission()?;
        let tensors = N::output_scores(output).cloned().into_iter().collect();
        let values =
            self.native
                .read_observation_tensors(lease, self.retention_root(), backend, tensors)?;
        let mut observations = ObservationSet::new();
        if let Some(value) = values.into_iter().next() {
            observations
                .insert(
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                    eredu_core::ObservationValue::Tensor(value),
                )
                .map_err(|error| N::preparation_error(BackendFailure::from_error(error)))?;
        }
        Ok(observations)
    }
}

impl<B, N, G> TextSessionLifecycle<B> for TextSessionDriver<N::Resources, N, G>
where
    B: BackendProvider,
    N: TextSessionMechanisms<B>,
{
    fn reset_session(&mut self, backend: &B) -> Result<(), BackendFailure> {
        self.native
            .validate_native_target(&self.payload.retained, backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        self.reset(backend)
            .map_err(|error| self.lifecycle_failure(error))
    }

    fn synchronize_session(&self, backend: &B) -> Result<(), BackendFailure> {
        self.native
            .validate_native_target(&self.payload.retained, backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        N::check_transport_active(&self.payload.retained)?;
        N::synchronize_queues(backend)
            .map_err(|error| BackendFailure::new(BackendFailureKind::Other, error))?;
        self.ensure_no_submission_in_flight()
            .map_err(|error| self.lifecycle_failure(error))
    }

    fn agree_text_preparation(
        &self,
        backend: &B,
        stage: TextPreparationStage,
        status: TextPreparationStatus,
    ) -> Result<TextPreparationOutcome, BackendFailure> {
        let valid = self
            .native
            .validate_native_target(&self.payload.retained, backend)
            .and_then(|()| self.ensure_no_submission_in_flight());
        let status = if valid.is_err() {
            TextPreparationStatus::Failed
        } else {
            status
        };
        let agreed = match N::control_transport(&self.payload.retained) {
            Some(transport) => TextPreparationCoordinator::require(self.preparation.as_deref())
                .map_err(|error| BackendFailure::new(BackendFailureKind::Unsupported, error))
                .and_then(|coordinator| {
                    coordinator
                        .agree(transport, stage, status)
                        .map_err(BackendFailure::from_error)
                }),
            None => Ok(status.local_outcome()),
        };
        valid.map_err(|error| self.lifecycle_failure(error))?;
        agreed
    }

    fn text_preparation_usage(&self, _backend: &B) -> Result<TextPreparationUsage, BackendFailure> {
        if N::control_transport(&self.payload.retained).is_none() {
            return Ok(TextPreparationUsage::default());
        }
        TextPreparationCoordinator::require(self.preparation.as_deref())
            .map_err(|error| BackendFailure::new(BackendFailureKind::Unsupported, error))?
            .usage()
            .map_err(BackendFailure::from_error)
    }
}

/// Physical route for the next advancement of a portable text generation run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextAdvanceRoute {
    /// Ordinary causal prefill, cached decode, or retained score continuation.
    Causal,
    /// A block-refinement step in its installed phase.
    Block(eredu_core::refinement::GenerationPhase),
    /// Propose and verify a tentative shared-backbone suffix.
    SharedProposal,
    /// Select the next target token from retained verification.
    SharedTarget,
}

impl TextAdvanceRoute {
    /// Chooses the exact next route from the installed mode and retained work.
    pub fn select<P, T>(
        mode: eredu_core::decoding::DecodingMode,
        phase: Option<eredu_core::refinement::GenerationPhase>,
        verification_pending: bool,
        input: &eredu_core::PendingTextInput<P, T>,
    ) -> Self {
        use eredu_core::{decoding::DecodingMode, refinement::GenerationPhase, PendingTextInput};

        match mode {
            DecodingMode::Diffusion { .. } => {
                Self::Block(phase.unwrap_or(GenerationPhase::Context))
            }
            DecodingMode::SelfSpeculation(_) if verification_pending => Self::SharedTarget,
            DecodingMode::SelfSpeculation(_) if matches!(input, PendingTextInput::Decode(_)) => {
                Self::SharedProposal
            }
            _ => Self::Causal,
        }
    }

    /// Declares the controller input required by this route.
    pub const fn operation_kind(self) -> eredu_core::TextOperationKind {
        use eredu_core::{
            refinement::{GenerationPhase, PredictionScoreKind},
            TextOperationKind,
        };

        match self {
            Self::Causal => TextOperationKind::CausalDecision,
            Self::Block(GenerationPhase::Finalization) => {
                TextOperationKind::Decision(PredictionScoreKind::ConstrainedFinalization)
            }
            Self::Block(phase) => TextOperationKind::Progress(phase),
            Self::SharedProposal => TextOperationKind::Progress(GenerationPhase::Refinement),
            Self::SharedTarget => {
                TextOperationKind::Decision(PredictionScoreKind::CausalConditional)
            }
        }
    }
}

impl<B, N, G> eredu_core::DistributedModelSession<B> for TextSessionDriver<N::Resources, N, G>
where
    B: BackendProvider,
    N: TextSessionMechanisms<B>,
    N::ControlTransport: eredu_core::DistributedSession<Error = B::Error>,
    N::Model: 'static,
    N::Retained: 'static,
    Self: BackendSession<B>,
{
    type DistributedSession = N::ControlTransport;

    fn distributed_session(&self) -> Option<&Self::DistributedSession> {
        N::control_transport(&self.resources().retained)
    }
}

impl<B, N, T, G> eredu_core::MultimodalPreparationSession<B>
    for TextSessionDriver<N::Resources, N, G>
where
    B: eredu_core::TextGenerationBackend<InputTensor = T>,
    B::Error: From<BackendFailure>,
    N: TextSessionMechanisms<B, TextValues: eredu_core::backend::NativeTextValues<InputTensor = T>>
        + eredu_core::backend::TextMediaNative<B>,
    Self: eredu_core::backend::TextOperationDriver<
        B,
        Prompt = crate::text_session_driver::TextPromptFor<B>,
    >,
{
    fn prepare_multimodal_input<E>(
        &self,
        request: &eredu_core::TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<
        crate::text_session_driver::TextPromptFor<B>,
        eredu_core::MultimodalPreparationFailure<B::Error, E>,
    >
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        use eredu_core::MultimodalPreparationFailure as Failure;
        let resources = self.resources();
        let processor = resources.processor.as_deref();
        resources
            .model
            .as_ref()
            .capability_estimate()
            .admit_raw_media_request(request, processor.is_some())?;
        let processor = processor.ok_or_else(|| {
            Failure::Backend(unsupported_text_invocation("multimodal preparation").into())
        })?;
        let scope = self.native().begin_processor().map_err(Failure::Backend)?;
        let (input, identity) =
            crate::processor_binding::prepare_media(processor, request, scope, encode_text)?;
        Ok(PreparedTextPrompt::from_prepared_media(input, identity))
    }
}

/// Retains the portable capture binding for the exact installed distributed session.
pub fn speculative_partition_binding<B, N>(
    session: &TextSessionDriver<N::Resources, N>,
) -> Option<crate::capture::SpeculativePartitionBinding<N::ControlTransport>>
where
    B: BackendProvider,
    N: TextSessionMechanisms<B>,
{
    let resources = session.resources();
    let transport = N::control_transport(&resources.retained)?;
    Some(crate::capture::SpeculativePartitionBinding::new(
        session
            .model_discovery()
            .cloned()
            .map(|prepared| prepared as Arc<_>),
        transport.clone(),
        N::control_transport_identity(transport),
        session.preparation_coordinator().cloned(),
        resources.model.as_ref().partition_activation_execution(),
        session.intervention_identity().to_owned(),
        session.parameter_overlay_identity().map(str::to_owned),
        session.speculative_capture_layouts().clone(),
    ))
}
