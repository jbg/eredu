//! Portable draft execution, observation, and selection retained through native work.

use eredu_core::speculative::assistant::ExternalAssistantBinder;
use eredu_core::{
    capability::{NativeAllocatorObservations, NativePlacementMemory},
    residency::{
        ParameterConversionRetentionObserver, ParameterConversionRetentionReport,
        ParameterConversionRetentionTrimReport, ParameterConversionRetentionTrimmer,
        ParameterConversionTrimError,
    },
    session_authority::{RetainedExecutionPayload, RetainedPayloadOperation},
    BackendFailure, Observed, SpeculativeExecutionTopology, TokenizerCompatibilityProof,
};

use super::{BoundAssistantDispatch, InstalledExternalDraft};
use crate::{
    memory_estimation::MemoryBytes,
    memory_forecast::{ConditionedAssistantMemoryProfile, GenerationForecastError},
    text_session_driver::forecast::{DraftForecastFacts, IndependentDraftMemoryFacts},
    SelectedSpeculativeRealization,
};
use eredu_core::backend::TextExecutableTypes;

/// Owns the selected portable draft alongside its exact native retirement resources.
pub struct InstalledDraftSession<B, N> {
    native: N,
    binder: std::marker::PhantomData<fn() -> B>,
}

impl<B, N> InstalledDraftSession<B, N> {
    pub(super) fn new(native: N) -> Self {
        Self {
            native,
            binder: std::marker::PhantomData,
        }
    }

    /// Observes the retirement resources paired with the installed draft.
    pub fn resources(&self) -> &N {
        &self.native
    }
}

impl<B, N, D: 'static, E> InstalledDraftSession<B, N>
where
    N: RetainedExecutionPayload<Payload = InstalledExternalDraft<D, E>>,
    E: TextExecutableTypes,
{
    /// Borrows the portable draft protected by native resources.
    pub fn installed(&self) -> &InstalledExternalDraft<D, E> {
        self.native.payload()
    }

    /// Exact execution queue of this draft.
    pub fn retain_queue(&self) -> N::Queue {
        self.native.retain_queue()
    }

    /// Acquires the mutation scope before portable speculative preparation.
    pub fn begin_operation(&mut self) -> Result<N::Operation<'_>, N::Error> {
        self.native.begin_operation()
    }

    /// The admitted draft execution and placement.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        self.installed().selected()
    }

    /// The admitted topology shared with the target execution.
    pub fn topology(&self) -> SpeculativeExecutionTopology {
        self.selected().placement().topology()
    }

    /// Tokenizer compatibility proved at portable admission.
    pub fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.installed().tokenizer_compatibility()
    }
}

impl<B, N, D: 'static, E> DraftForecastFacts for InstalledDraftSession<B, N>
where
    N: RetainedExecutionPayload<Payload = InstalledExternalDraft<D, E>> + NativePlacementMemory,
    D: BoundAssistantDispatch<B>,
    B: ExternalAssistantBinder,
    E: TextExecutableTypes,
{
    fn forecast_topology(&self) -> SpeculativeExecutionTopology {
        self.topology()
    }

    fn forecast_proposal_capacity(&self) -> u64 {
        self.selected()
            .requirements()
            .strategy()
            .proposal_capacity()
            .get() as u64
    }

    fn conditioned_forecast(
        &self,
    ) -> Result<Option<ConditionedAssistantMemoryProfile>, GenerationForecastError> {
        let scratch = self.native.scratch_memory_upper_bound().map_or_else(
            || {
                MemoryBytes::unknown(
                    "native conditioned projection and operator scratch bound is unavailable",
                )
            },
            |upper| MemoryBytes::estimated(0, upper, "native conditioned operator scratch bound"),
        );
        self.installed()
            .conditioned_memory_profile::<B, N::Memory>(self.native.host_execution(), scratch)
    }

    fn independent_forecast(
        &self,
    ) -> Result<Option<IndependentDraftMemoryFacts>, GenerationForecastError> {
        self.installed().autoregressive_memory_facts(
            self.native.host_execution(),
            N::Memory::native_memory_facts(),
        )
    }
}

impl<B, N, D: 'static, E> ParameterConversionRetentionObserver for InstalledDraftSession<B, N>
where
    N: RetainedExecutionPayload<Payload = InstalledExternalDraft<D, E>>,
    D: BoundAssistantDispatch<B>,
    B: ExternalAssistantBinder,
    E: TextExecutableTypes,
{
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        self.installed().parameter_conversion_retention::<B>()
    }
}

impl<B, N, D: 'static, E> ParameterConversionRetentionTrimmer for InstalledDraftSession<B, N>
where
    N: RetainedExecutionPayload<Payload = InstalledExternalDraft<D, E>>,
    N::Error: std::error::Error + Send + Sync + 'static,
    D: BoundAssistantDispatch<B>,
    B: ExternalAssistantBinder,
    E: TextExecutableTypes,
{
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        let mut operation = self.begin_operation().map_err(BackendFailure::from_error)?;
        let result = operation.payload_mut().trim_parameter_conversions::<B>();
        operation
            .settle(Ok(result))
            .map_err(BackendFailure::from_error)?
    }
}
