//! Source-explicit plain speculation through the existing ordinary/controlled drivers.
use super::speculative_semantic::{
    OriginalSpeculativeStops, PreparedPlainSpeculativeSemanticError,
};
use super::{LoadedModel, ManagedPlainTextRequest, ManagedPlainTextSource};
use crate::{
    api::{PreparedChatSpeculativeConstraint, PreparedChatSpeculativeGenerationOptions},
    runtime::chat::constraints::ConstraintController,
};
use eredu_core::speculative::SpeculativeControlError;
use eredu_core::{
    BackendFailure, GenerationCancellationToken, GenerationError, HostMetadataFundingError,
    HostPreparationAuthority, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationOutput, SpeculativeGenerationVisitor, TokenInputRejection,
    generation::SemanticEvent,
};
use eredu_nn::workspace::WorkspaceMetadataFunding;
use eredu_runtime::{
    speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    },
    working_memory::{OriginalTextSourceError, OriginalTokenizerBackend},
};
use std::mem::size_of;

mod chat;
mod preparation;
pub use chat::{ManagedPreparedChatSpeculativeError, ManagedPreparedChatSpeculativeRequest};
mod batch;
pub use batch::{ManagedPlainTextSpeculativeBatchLane, ManagedPlainTextSpeculativeBatchRequest};

/// One original plain prompt and explicit target/draft request. Caller-owned
/// input, stop strings and closure payload stay borrowed/moved; framework-created
/// prompt, semantic, controller, EOS and callback storage use their real accounts.
pub struct ManagedPlainTextSpeculativeRequest<'a, D, F> {
    /// Literal prompt, generation policy and literal stopping configuration.
    pub text: ManagedPlainTextRequest<'a>,
    /// The actually selected assistant; unavailable strategies refuse normally.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Existing proposal width and fair scheduler controls.
    pub options: PreparedChatSpeculativeGenerationOptions,
    /// Existing cooperative cancellation token, shared with the driver.
    pub cancellation: GenerationCancellationToken,
    /// Receives committed semantic events only, after cache commitment.
    pub on_event: F,
}
impl<D, F> std::fmt::Debug for ManagedPlainTextSpeculativeRequest<'_, D, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedPlainTextSpeculativeRequest")
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Input(#[from] TokenInputRejection),
    #[error("{0}")]
    Generation(#[from] GenerationError),
    #[error("{0}")]
    Host(#[from] HostMetadataFundingError),
    #[error("{0}")]
    Preparation(#[from] PreparedPlainSpeculativeSemanticError),
    #[error("{0}")]
    Source(#[from] OriginalTextSourceError),
    #[error("{0}")]
    Backend(#[from] BackendFailure),
    #[error("{0}")]
    Control(#[from] SpeculativeControlError),
    #[error("{0}")]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
    #[error("expected one speculative output, received {0}")]
    Cardinality(usize),
}
/// Backend-independent preparation/advancement failure. Native causes are kept
/// beneath BackendFailure; the actual enclosing host account retires last.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ManagedPlainTextSpeculativeError {
    #[source]
    cause: Cause,
    funding: Option<WorkspaceMetadataFunding>,
}
impl ManagedPlainTextSpeculativeError {
    /// Fixed source, policy or unsupported-profile rejection when applicable.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match self.cause {
            Cause::Input(error) => Some(error),
            _ => None,
        }
    }
    /// The selected backend failure with its original retained source chain.
    pub fn backend_failure(&self) -> Option<&BackendFailure> {
        match &self.cause {
            Cause::Backend(error) => Some(error),
            _ => None,
        }
    }
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(std::mem::size_of_val(parts), usize::checked_add)
}
fn reserve(funding: &WorkspaceMetadataFunding, bytes: Option<usize>) -> Result<(), Cause> {
    funding.reserve_metadata(bytes.ok_or(HostMetadataFundingError::Overflow)?)?;
    Ok(())
}
impl<B: OriginalTokenizerBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    /// Generates plain text from an original tokenizer/input source through the
    /// existing speculative scheduler. The request must name a managed capacity;
    /// missing selected native producers refuse without an ordinary fallback.
    pub fn generate_managed_plain_text_speculative<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationOutput, ManagedPlainTextSpeculativeError> {
        let driver = eredu_runtime::RunSpeculativeGeneration::new(request.options.scheduler);
        let mut funding = None;
        self.run_managed_plain_speculative(source, request, driver, &mut funding)
            .map_err(|cause| ManagedPlainTextSpeculativeError { cause, funding })
    }

    /// Lends the same request to the existing controlled driver. Step, rollback,
    /// snapshots, cancellation and commitment use the shared worker. model.logits
    /// raw captures, score/candidate readouts, Summary and Histogram use originally
    /// funded sources. Static logits interventions are installed through the
    /// session's ordinary control API and support bounded Preview/Summary evidence.
    /// Capture spending is shared across snapshots and branches; saved immutable
    /// edit sources survive clearing or replacement of the current plan. Internal
    /// activation requests use the same shared configuration hook; selected native
    /// mechanisms authenticate the loaded source and admit each actual phase.
    pub fn with_controlled_managed_plain_text_speculative<'a, F, D>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ManagedPlainTextSpeculativeError>
    where
        F: FnMut(SemanticEvent) + 'a,
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        if !source.original().matches_configuration(&self.tokenizer) {
            return Err(ManagedPlainTextSpeculativeError {
                cause: TokenInputRejection::IdentityMismatch.into(),
                funding: None,
            });
        }
        // The original source lends IDs; no ordinary tokenizer vocabulary map is cloned.
        let vocabulary = source
            .original()
            .ids()
            .max()
            .map_or(Some(0), |id| usize::try_from(id).ok()?.checked_add(1))
            .ok_or_else(|| ManagedPlainTextSpeculativeError {
                cause: HostMetadataFundingError::Overflow.into(),
                funding: None,
            })?;
        let mut failure = None;
        let driver = DriveControlledSpeculation::new(
            request.options.scheduler,
            options,
            drive,
            &mut failure,
        )
        .with_vocabulary(vocabulary);
        let mut funding = None;
        let result = self.run_managed_plain_speculative(source, request, driver, &mut funding);
        if let Some(error) = failure {
            return Err(ManagedPlainTextSpeculativeError {
                cause: error.into(),
                funding,
            });
        }
        result.map_err(|cause| ManagedPlainTextSpeculativeError { cause, funding })
    }

    /// Continuously drives the same controlled session, delivering tentative
    /// and committed attribution through its ordinary bounded step records.
    /// Breaking the callback cancels through the existing shared driver.
    pub fn generate_observed_managed_plain_text_speculative<'a, F, D>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        on_step: D,
    ) -> Result<SpeculativeGenerationOutput, ManagedPlainTextSpeculativeError>
    where
        F: FnMut(SemanticEvent) + 'a,
        D: FnMut(
            eredu_runtime::speculative::ControlledSpeculativeStep,
        ) -> std::ops::ControlFlow<()>,
    {
        self.with_controlled_managed_plain_text_speculative(
            source,
            request,
            options,
            crate::api::controlled_speculative::continuous_steps(on_step),
        )
    }

    fn run_managed_plain_speculative<'a, F, V>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>,
        driver: V,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<SpeculativeGenerationOutput, Cause>
    where
        F: FnMut(SemanticEvent) + 'a,
        V: SpeculativeGenerationVisitor,
    {
        // Retire tokenizer, semantic and lane-construction temporaries before
        // entering the shared driver and its model workspace constructors.
        let request = self.prepare_managed_plain_speculative_request(source, request, retained)?;
        self.execute_managed_speculative_request(request, driver, retained)
    }

    fn execute_managed_speculative_request<'a, V: SpeculativeGenerationVisitor>(
        &mut self,
        request: eredu_core::SpeculativeGenerationBatchRequest<
            'a,
            B,
            B::Drafter,
            PreparedChatSpeculativeConstraint,
        >,
        driver: V,
        retained: &Option<WorkspaceMetadataFunding>,
    ) -> Result<SpeculativeGenerationOutput, Cause> {
        let funding = retained
            .as_ref()
            .expect("prepared request retains its host source");
        reserve(
            funding,
            sum(&[
                size_of::<V>(),
                size_of::<Result<SpeculativeGenerationOutput, Cause>>(),
                size_of::<Result<SpeculativeGenerationOutput, ManagedPlainTextSpeculativeError>>(),
            ]),
        )?;
        let output = B::with_speculative_execution(&mut self.runtime, request, driver)
            .map_err(|error| Cause::Backend(B::into_backend_failure(error)))?;
        let mut requests = output.into_requests();
        if requests.len() != 1 {
            return Err(Cause::Cardinality(requests.len()));
        }
        Ok(requests.pop().expect("checked single speculative output"))
    }

    #[inline(never)]
    fn prepare_managed_plain_speculative_request<'a, F>(
        &self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<
        eredu_core::SpeculativeGenerationBatchRequest<
            'a,
            B,
            B::Drafter,
            PreparedChatSpeculativeConstraint,
        >,
        Cause,
    >
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        let ManagedPlainTextSpeculativeRequest {
            text,
            drafting,
            options,
            cancellation,
            on_event,
        } = request;
        options.scheduler.validate()?;
        let prepared = self.prepare_managed_speculative_host(
            source,
            text,
            options.max_draft_tokens,
            cancellation,
            on_event,
            retained,
        )?;
        let funding = prepared.funding().clone();
        reserve(
            &funding,
            sum(&[
                size_of::<ManagedPlainTextSpeculativeRequest<'a, B::Drafter, F>>(),
                size_of::<
                    eredu_core::SpeculativeGenerationBatchRequest<
                        'a,
                        B,
                        B::Drafter,
                        PreparedChatSpeculativeConstraint,
                    >,
                >(),
                size_of::<
                    Result<
                        eredu_core::SpeculativeGenerationBatchRequest<
                            'a,
                            B,
                            B::Drafter,
                            PreparedChatSpeculativeConstraint,
                        >,
                        Cause,
                    >,
                >(),
                size_of::<(
                    &Self,
                    &ManagedPlainTextSource,
                    &mut Option<WorkspaceMetadataFunding>,
                )>(),
            ]),
        )?;
        let lane = prepared.finish(self)?;
        let mut lanes = preparation::buffer(1, &funding)?;
        lanes.try_push(lane)?;
        let request = eredu_core::SpeculativeGenerationBatchRequest::new(
            drafting,
            lanes,
            self.tokenizer_fingerprint,
        );
        Ok(request)
    }
}
