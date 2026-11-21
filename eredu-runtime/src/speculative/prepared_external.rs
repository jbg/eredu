//! Source and mechanism handoff for external draft construction.

use crate::SelectedSpeculativeRealization;
use eredu_core::speculative::assistant::ExternalAssistantBinder;
use eredu_core::TokenizerCompatibilityProof;

use super::{
    BoundAssistantRefVisitor, BoundAssistantVisitor, ExternalAssistantConstruction,
    InstalledExternalAssistant, SelectedAssistantExecutionVisitor, SelectedAssistantExecutor,
};

/// Prepared conditioned assistant that can bind through native parameter mechanisms.
pub trait PreparedAssistantExecution<B: ExternalAssistantBinder> {
    /// The bound assistant with its retained selection and execution authority.
    type Output;

    /// Binds the selected assistant using its exact prepared inputs.
    fn bind_assistant(self, binder: B) -> Result<Self::Output, B::Error>;
}

/// Heterogeneous retained assistant selected by portable preparation.
pub enum AssistantDispatch<L, R> {
    /// The selected assistant in the first variant.
    First(L),
    /// The selected assistant in the second variant.
    Second(R),
}

/// Materialized assistant paired with authority for its typed executor.
pub struct TypedAssistant<B: ExternalAssistantBinder, A: ExternalAssistantConstruction, D> {
    assistant: InstalledExternalAssistant<B, A>,
    executor: D,
}

impl<
        B: ExternalAssistantBinder,
        A: ExternalAssistantConstruction,
        D: SelectedAssistantExecutor<A>,
    > TypedAssistant<B, A, D>
{
    /// Keeps the selected family executor alongside its materialized storage.
    pub fn new(assistant: InstalledExternalAssistant<B, A>, executor: D) -> Self {
        Self {
            assistant,
            executor,
        }
    }
}

/// Typed visitation of a materialized assistant selected by portable code.
pub trait BoundAssistantDispatch<B: ExternalAssistantBinder> {
    /// Borrows the selected native storage for observation.
    fn visit_ref<V: BoundAssistantRefVisitor<B>>(&self, visitor: V) -> V::Output;
    /// Borrows the selected native storage exclusively.
    fn visit<V: BoundAssistantVisitor<B>>(&mut self, visitor: V) -> V::Output;
    /// Borrows selected storage and supplies authority for its executor.
    fn visit_executor<V: SelectedAssistantExecutionVisitor<B>>(&mut self, visitor: V) -> V::Output;
}

impl<B: ExternalAssistantBinder> BoundAssistantDispatch<B> for std::convert::Infallible {
    fn visit_ref<V: BoundAssistantRefVisitor<B>>(&self, _: V) -> V::Output {
        match *self {}
    }

    fn visit<V: BoundAssistantVisitor<B>>(&mut self, _: V) -> V::Output {
        match *self {}
    }

    fn visit_executor<V: SelectedAssistantExecutionVisitor<B>>(&mut self, _: V) -> V::Output {
        match *self {}
    }
}

impl<B, A, D> BoundAssistantDispatch<B> for TypedAssistant<B, A, D>
where
    B: ExternalAssistantBinder,
    A: ExternalAssistantConstruction,
    D: SelectedAssistantExecutor<A> + Copy,
{
    fn visit_ref<V: BoundAssistantRefVisitor<B>>(&self, visitor: V) -> V::Output {
        visitor.visit::<A>(&self.assistant)
    }

    fn visit<V: BoundAssistantVisitor<B>>(&mut self, visitor: V) -> V::Output {
        visitor.visit::<A>(&mut self.assistant)
    }

    fn visit_executor<V: SelectedAssistantExecutionVisitor<B>>(&mut self, visitor: V) -> V::Output {
        visitor.visit(&mut self.assistant, self.executor)
    }
}

impl<B, L, R> BoundAssistantDispatch<B> for AssistantDispatch<L, R>
where
    B: ExternalAssistantBinder,
    L: BoundAssistantDispatch<B>,
    R: BoundAssistantDispatch<B>,
{
    fn visit_ref<V: BoundAssistantRefVisitor<B>>(&self, visitor: V) -> V::Output {
        match self {
            Self::First(assistant) => assistant.visit_ref(visitor),
            Self::Second(assistant) => assistant.visit_ref(visitor),
        }
    }

    fn visit<V: BoundAssistantVisitor<B>>(&mut self, visitor: V) -> V::Output {
        match self {
            Self::First(assistant) => assistant.visit(visitor),
            Self::Second(assistant) => assistant.visit(visitor),
        }
    }

    fn visit_executor<V: SelectedAssistantExecutionVisitor<B>>(&mut self, visitor: V) -> V::Output {
        match self {
            Self::First(assistant) => assistant.visit_executor(visitor),
            Self::Second(assistant) => assistant.visit_executor(visitor),
        }
    }
}

/// Admitted execution facts retained with the selected assistant dispatcher.
pub struct BoundAssistant<D> {
    dispatch: D,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
    capture: super::ExternalPredictionCaptureRequest,
    residency: crate::LayerWeightResidency,
}

impl<D> BoundAssistant<D> {
    /// Pairs the bound dispatcher with the exact portable selection and source policy.
    pub fn new(
        dispatch: D,
        selected: SelectedSpeculativeRealization,
        tokenizer: TokenizerCompatibilityProof,
        capture: super::ExternalPredictionCaptureRequest,
        residency: crate::LayerWeightResidency,
    ) -> Self {
        Self {
            dispatch,
            selected,
            tokenizer,
            capture,
            residency,
        }
    }

    /// Borrows the retained typed dispatch for observation.
    pub fn dispatch(&self) -> &D {
        &self.dispatch
    }

    /// Borrows the retained typed dispatch under exclusive operation ownership.
    pub fn dispatch_mut(&mut self) -> &mut D {
        &mut self.dispatch
    }

    /// Exact speculative selection retained through binding.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        &self.selected
    }

    /// Exact tokenizer proof retained through binding.
    pub const fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.tokenizer
    }

    /// Target feature capture admitted for this assistant.
    pub const fn capture(&self) -> &super::ExternalPredictionCaptureRequest {
        &self.capture
    }

    /// Weight residency selected for the assistant independently of the target.
    pub const fn residency(&self) -> crate::LayerWeightResidency {
        self.residency
    }
}

/// Exact prepared sources paired with independent-draft selection.
pub struct PreparedAutoregressiveDraft<S> {
    sources: S,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
}
impl<S> PreparedAutoregressiveDraft<S> {
    /// Retains checkpoint sources with their speculative realization and tokenizer proof.
    pub fn new(
        sources: S,
        selected: SelectedSpeculativeRealization,
        tokenizer: TokenizerCompatibilityProof,
    ) -> Self {
        Self {
            sources,
            selected,
            tokenizer,
        }
    }

    /// Consumes the inseparable native materialization handoff.
    pub fn into_parts(
        self,
    ) -> (
        S,
        SelectedSpeculativeRealization,
        TokenizerCompatibilityProof,
    ) {
        (self.sources, self.selected, self.tokenizer)
    }
}
/// Portable handoff of the selected external execution mechanism.
pub enum PreparedExternalDraft<A, S> {
    /// Target-feature-conditioned assistant.
    Assistant(A),
    /// Two ordinary decoders with isolated caches.
    Autoregressive(PreparedAutoregressiveDraft<S>),
}

/// Independent decoder and discovery retained from portable model construction.
pub trait IndependentDraftModel<E: eredu_core::backend::TextExecutableTypes> {
    /// Borrows the decoder for storage observation.
    fn executable(&self) -> &(dyn crate::executable_binding::ErasedTextExecutable<E> + 'static);

    /// Borrows the decoder under exclusive draft authority.
    fn executable_mut(
        &mut self,
    ) -> &mut (dyn crate::executable_binding::ErasedTextExecutable<E> + 'static);

    /// Projects the loaded model's geometry using its current storage.
    fn memory_geometry(
        &self,
        residency: Option<&crate::ResidencyReport>,
    ) -> Result<crate::memory_forecast::LoadedMemoryGeometry, eredu_core::CapabilityError>;
}

impl<E: eredu_core::backend::TextExecutableTypes, C, P, T> IndependentDraftModel<E>
    for crate::executable_binding::LoadedTextExecutable<
        crate::executable_binding::InstalledTextExecutable<E>,
        C,
        P,
        T,
    >
{
    fn executable(&self) -> &(dyn crate::executable_binding::ErasedTextExecutable<E> + 'static) {
        self.executable.as_ref()
    }

    fn executable_mut(
        &mut self,
    ) -> &mut (dyn crate::executable_binding::ErasedTextExecutable<E> + 'static) {
        self.executable.as_mut()
    }

    fn memory_geometry(
        &self,
        residency: Option<&crate::ResidencyReport>,
    ) -> Result<crate::memory_forecast::LoadedMemoryGeometry, eredu_core::CapabilityError> {
        self.memory_geometry(residency)
    }
}

/// Selected installation retained alongside the backend's completion resources.
pub struct InstalledExternalDraft<D, E: eredu_core::backend::TextExecutableTypes> {
    execution: ExternalDraftExecution<D, E>,
    target_identity: String,
}

/// Constructed external executor paired with its retained portable selection.
pub enum ExternalDraftExecution<D, E: eredu_core::backend::TextExecutableTypes> {
    /// Target-feature-conditioned assistant.
    Assistant(BoundAssistant<D>),
    /// Independently placed ordinary draft model.
    Autoregressive {
        /// Installed model and native target resources.
        model: Box<dyn IndependentDraftModel<E>>,
        /// Selected speculative realization retained for requests.
        selected: SelectedSpeculativeRealization,
        /// Exact tokenizer compatibility checked at admission.
        tokenizer: TokenizerCompatibilityProof,
    },
}

impl<D, E: eredu_core::backend::TextExecutableTypes> InstalledExternalDraft<D, E> {
    /// Exact installed target session admitted for this draft.
    pub fn target_identity(&self) -> &str {
        &self.target_identity
    }

    /// Borrows the constructed draft and its retained selection.
    pub fn execution(&self) -> &ExternalDraftExecution<D, E> {
        &self.execution
    }

    /// Borrows the draft under its native retirement and mutation scope.
    pub fn execution_mut(&mut self) -> &mut ExternalDraftExecution<D, E> {
        &mut self.execution
    }

    /// Tokenizer proof retained by either selected execution path.
    pub fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        match &self.execution {
            ExternalDraftExecution::Assistant(draft) => draft.tokenizer_compatibility(),
            ExternalDraftExecution::Autoregressive { tokenizer, .. } => *tokenizer,
        }
    }

    /// Exact realization retained by either selected execution path.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        match &self.execution {
            ExternalDraftExecution::Assistant(draft) => draft.selected(),
            ExternalDraftExecution::Autoregressive { selected, .. } => selected,
        }
    }

    /// Retained feature capture required by a conditioned assistant.
    pub fn conditioned_capture(&self) -> Option<&super::ExternalPredictionCaptureRequest> {
        match &self.execution {
            ExternalDraftExecution::Assistant(draft) => Some(draft.capture()),
            ExternalDraftExecution::Autoregressive { .. } => None,
        }
    }
}

struct ObserveAssistantConversions;

impl<B: ExternalAssistantBinder> BoundAssistantRefVisitor<B> for ObserveAssistantConversions {
    type Output = Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    >;

    fn visit<A: ExternalAssistantConstruction>(
        self,
        assistant: &InstalledExternalAssistant<B, A>,
    ) -> Self::Output {
        assistant.parameter_conversion_retention()
    }
}

struct TrimAssistantConversions;

impl<B: ExternalAssistantBinder> BoundAssistantVisitor<B> for TrimAssistantConversions {
    type Output = Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    >;

    fn visit<A: ExternalAssistantConstruction>(
        self,
        assistant: &mut InstalledExternalAssistant<B, A>,
    ) -> Self::Output {
        assistant.trim_parameter_conversions()
    }
}

struct ObserveAssistantMemory<N> {
    host_execution: bool,
    residency: crate::LayerWeightResidency,
    scratch: crate::memory_estimation::MemoryBytes,
    native: std::marker::PhantomData<N>,
}

impl<B: ExternalAssistantBinder, N: eredu_core::capability::NativeAllocatorObservations>
    BoundAssistantRefVisitor<B> for ObserveAssistantMemory<N>
{
    type Output = Result<
        Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>,
        eredu_core::BackendFailure,
    >;

    fn visit<A: ExternalAssistantConstruction>(
        self,
        assistant: &InstalledExternalAssistant<B, A>,
    ) -> Self::Output {
        assistant.memory_profile(
            self.host_execution,
            self.residency,
            || assistant.parameter_memory_observation::<N>(),
            self.scratch,
        )
    }
}

impl<D, E: eredu_core::backend::TextExecutableTypes> InstalledExternalDraft<D, E> {
    /// Reports the admitted participant's conversion budget and current allocation.
    pub fn parameter_conversion_retention<B: ExternalAssistantBinder>(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    >
    where
        D: BoundAssistantDispatch<B>,
    {
        match self.execution() {
            ExternalDraftExecution::Autoregressive { model, .. } => {
                model.executable().parameter_conversion_retention()
            }
            ExternalDraftExecution::Assistant(assistant) => {
                assistant.dispatch().visit_ref(ObserveAssistantConversions)
            }
        }
    }

    /// Releases optional conversions after the native scope acquires exclusive authority.
    pub fn trim_parameter_conversions<B: ExternalAssistantBinder>(
        &mut self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    >
    where
        D: BoundAssistantDispatch<B>,
    {
        match self.execution_mut() {
            ExternalDraftExecution::Autoregressive { model, .. } => {
                model.executable_mut().trim_parameter_conversions()
            }
            ExternalDraftExecution::Assistant(assistant) => {
                assistant.dispatch_mut().visit(TrimAssistantConversions)
            }
        }
    }

    /// Projects memory for a selected conditioned assistant.
    pub fn conditioned_memory_profile<
        B: ExternalAssistantBinder,
        N: eredu_core::capability::NativeAllocatorObservations,
    >(
        &self,
        host_execution: Result<bool, eredu_core::BackendFailure>,
        scratch: crate::memory_estimation::MemoryBytes,
    ) -> Result<
        Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>,
        crate::memory_forecast::GenerationForecastError,
    >
    where
        D: BoundAssistantDispatch<B>,
    {
        let ExternalDraftExecution::Assistant(assistant) = self.execution() else {
            return Ok(None);
        };
        Ok(assistant
            .dispatch()
            .visit_ref(ObserveAssistantMemory::<N> {
                host_execution: host_execution?,
                residency: assistant.residency(),
                scratch,
                native: std::marker::PhantomData,
            })?)
    }

    /// Projects the selected independent decoder using its installed parameter heads.
    pub fn autoregressive_memory_facts(
        &self,
        host_execution: Result<bool, eredu_core::BackendFailure>,
        native: eredu_core::capability::NativeMemoryFacts,
    ) -> Result<
        Option<crate::text_session_driver::forecast::IndependentDraftMemoryFacts>,
        crate::memory_forecast::GenerationForecastError,
    > {
        let ExternalDraftExecution::Autoregressive { model, .. } = self.execution() else {
            return Ok(None);
        };
        let residency = model
            .executable()
            .residency_report()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let geometry = model.memory_geometry(residency.as_ref())?;
        let banks = crate::text_session_driver::capability::bank_memory_facts(
            model
                .executable()
                .parameter_bank_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
        );
        Ok(Some(
            crate::text_session_driver::forecast::IndependentDraftMemoryFacts {
                geometry,
                residency,
                banks,
                native,
                host_execution: host_execution?,
            },
        ))
    }
}

impl<A, S> PreparedExternalDraft<A, S> {
    /// Constructs the admitted draft from its retained sources under the native scope.
    pub fn construct<M, D, E>(
        self,
        mechanisms: M,
        target_identity: &str,
    ) -> Result<
        super::InstalledDraftSession<M::AssistantBinder, M::Resources>,
        eredu_core::AutomaticPlanningError,
    >
    where
        M: eredu_core::automatic::DraftBindingResources<
            Payload = InstalledExternalDraft<D, E>,
            AssistantBinder: ExternalAssistantBinder<
                Error = <M::Backend as eredu_core::BackendProvider>::Error,
            >,
        >,
        D: 'static,
        E: eredu_core::backend::TextExecutableTypes,
        A: PreparedAssistantExecution<M::AssistantBinder, Output = BoundAssistant<D>>,
        S: eredu_core::BindModelSources<M::Backend, Model: IndependentDraftModel<E> + 'static>,
    {
        let construction_error = |error: <M::Backend as eredu_core::BackendProvider>::Error| {
            eredu_core::AutomaticPlanningError::Backend {
                operation: "realize_external_drafter",
                message: error.to_string(),
            }
        };
        let execution = match self {
            Self::Assistant(prepared) => ExternalDraftExecution::Assistant(
                prepared
                    .bind_assistant(mechanisms.assistant_binder())
                    .map_err(construction_error)?,
            ),
            Self::Autoregressive(prepared) => {
                let (sources, selected, tokenizer) = prepared.into_parts();
                let (backend, placement) = mechanisms.model_binding();
                let model = sources
                    .construct_model(backend, placement)
                    .map_err(|error| eredu_core::AutomaticPlanningError::Backend {
                        operation: "realize_external_drafter",
                        message: error.to_string(),
                    })?;
                ExternalDraftExecution::Autoregressive {
                    model: Box::new(model.into_inner()),
                    selected,
                    tokenizer,
                }
            }
        };
        let resources = mechanisms.finish_binding(InstalledExternalDraft {
            execution,
            target_identity: target_identity.to_owned(),
        })?;
        Ok(super::InstalledDraftSession::new(resources))
    }
}
