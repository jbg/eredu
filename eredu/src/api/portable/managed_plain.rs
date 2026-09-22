//! Public borrowed plain-text composition over the original C/S/E/I/R driver.
mod prepared;
use super::original_token_input::plain::{
    start_original_plain_string_with_options_for, OriginalDomainError, OriginalPlainSession,
    OriginalPlainStartError,
};
use super::LoadedModel;
use crate::api::PreparedChatGenerationSettings;
use eredu_core::{
    BackendFailure, ControlledTextGenerationError, FinishReason, GenerationCancellationToken,
    GenerationPlainTextEvent, GenerationPlainTextOutput, TextGenerationBackend,
    TokenInputRejection,
};
use eredu_runtime::working_memory::{
    OriginalTextSourceBudget, OriginalTextSourceError, OriginalTokenizer, OriginalTokenizerBackend,
    OriginalTokenizerSourceError,
};
pub use prepared::{ManagedModelInputError, ManagedPreparedInputRequest};

/// Actual input to fresh tokenizer source preparation.
#[derive(Debug)]
pub enum TokenizerSourceInput {
    /// Read a consumed exact tokenizer JSON file under its original allowance.
    File(std::fs::File),
    /// Serialize this loaded model's complete retained tokenizer configuration.
    RetainedConfiguration,
}
impl From<std::fs::File> for TokenizerSourceInput {
    fn from(file: std::fs::File) -> Self {
        Self::File(file)
    }
}

/// An originally compiled tokenizer and decoder, authenticated against loaded metadata.
/// Clones retain the same source account; no tokenizer, byte allowance or mutable handle escapes.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct ManagedPlainTextSource(OriginalTokenizer);

/// A tokenizer source could not be compiled or matched to loaded metadata.
/// Original compiler failures retain their complete source preparation custody.
#[derive(Debug, thiserror::Error)]
pub enum ManagedPlainTextSourceError {
    /// Fixed loaded-model/source identity mismatch.
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    /// File preparation or original source compilation failed.
    #[error(transparent)]
    Source(#[from] OriginalTokenizerSourceError),
}
impl ManagedPlainTextSourceError {
    /// Fixed source refusal, when construction or matching rejected its identity.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match self {
            Self::Input(cause) | Self::Source(OriginalTokenizerSourceError::Domain(cause)) => {
                Some(*cause)
            }
            _ => None,
        }
    }
}
impl ManagedPlainTextSource {
    pub(super) fn original(&self) -> &OriginalTokenizer {
        &self.0
    }
}

/// Borrowed input and policy for the managed plain-text driver.
/// Chat rendering, tools and owned callback events use their existing separate APIs.
#[derive(Debug, Clone)]
pub struct ManagedPlainTextRequest<'a> {
    /// Complete caller-owned plain prompt, borrowed only during startup.
    pub input: &'a str,
    /// Existing checkpoint overrides, sampler, seed and enforced inference policy.
    pub settings: PreparedChatGenerationSettings,
    /// Literal stop strings in source order, borrowed only during startup.
    pub stop_sequences: &'a [&'a str],
    /// Apply the selected tokenizer's special-token postprocessor to the prompt.
    pub add_special_tokens: bool,
    /// Omit special-token spellings from visible output text.
    pub skip_special_tokens: bool,
}
impl<'a> ManagedPlainTextRequest<'a> {
    /// Uses no literal stops, applies prompt special tokens, and hides output special tokens.
    pub fn new(input: &'a str, settings: PreparedChatGenerationSettings) -> Self {
        Self {
            input,
            settings,
            stop_sequences: &[],
            add_special_tokens: true,
            skip_special_tokens: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
    #[error(transparent)]
    Stop(#[from] eredu_text::stop_storage::StopSourceError),
    #[error(transparent)]
    Source(#[from] OriginalTextSourceError),
    #[error(transparent)]
    Domain(#[from] OriginalDomainError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    #[error(transparent)]
    Snapshot(#[from] eredu_runtime::execution_control::TextSnapshotError<BackendFailure>),
}
/// Backend-independent managed startup or advancement failure.
/// Fixed and original-source failures remain by value, preserving failed-prefix custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ManagedPlainTextError {
    #[source]
    cause: Cause,
    // Retain a failed source prefix's ceiling until its owning cause retires.
    source_budget: Option<OriginalTextSourceBudget>,
    // Raw input construction controls remain paid through a rejected handoff.
    input_preparation: Option<eredu_runtime::input::OriginalModelInputCustody>,
}
impl ManagedPlainTextError {
    pub(crate) fn from_step(
        cause: BackendFailure,
        input_preparation: Option<eredu_runtime::input::OriginalModelInputCustody>,
    ) -> Self {
        Self {
            cause: Cause::Backend(cause),
            source_budget: None,
            input_preparation,
        }
    }
    fn new(cause: Cause) -> Self {
        Self {
            cause,
            source_budget: None,
            input_preparation: None,
        }
    }
    /// Borrows a fixed input/identity refusal without allocating an error wrapper.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match self.cause {
            Cause::Input(cause) => Some(cause),
            _ => None,
        }
    }
    /// Borrows the classified backend failure, preserving its original source chain.
    pub fn backend_failure(&self) -> Option<&BackendFailure> {
        match &self.cause {
            Cause::Backend(cause) => Some(cause),
            _ => None,
        }
    }
    /// Borrows a classified snapshot/resume failure and its retained cause.
    pub fn snapshot_error(
        &self,
    ) -> Option<&eredu_runtime::execution_control::TextSnapshotError<BackendFailure>> {
        match &self.cause {
            Cause::Snapshot(cause) => Some(cause),
            _ => None,
        }
    }
    fn startup<B: TextGenerationBackend>(error: OriginalPlainStartError<B::Error>) -> Self {
        Self::new(match error {
            OriginalPlainStartError::Input(e) => Cause::Input(e),
            OriginalPlainStartError::Generation(e) => Cause::Generation(e),
            OriginalPlainStartError::Stop(e) => Cause::Stop(e),
            OriginalPlainStartError::Source(e) => Cause::Source(e),
            OriginalPlainStartError::Backend(e) => Cause::Backend(e),
            OriginalPlainStartError::Startup(ControlledTextGenerationError::Preparation(e)) => {
                Cause::Backend(e)
            }
            OriginalPlainStartError::Startup(ControlledTextGenerationError::Backend(e)) => {
                Cause::Backend(B::into_backend_failure(e))
            }
            OriginalPlainStartError::Startup(ControlledTextGenerationError::Controller(e)) => {
                Cause::Domain(e)
            }
        })
    }
}

/// Move-only controlled advancement of the same managed driver used by uninterrupted generation.
/// Events borrow retained text; terminal output keeps that same funded storage alive.
#[repr(transparent)]
pub struct ManagedPlainTextSession<'a, B: TextGenerationBackend>(
    pub(super) OriginalPlainSession<'a, B>,
);
impl<'a, B: TextGenerationBackend> ManagedPlainTextSession<'a, B> {
    /// Borrows the actual accepted request's selected chunk and admission bound.
    /// The report is identical whether this session is run or advanced manually;
    /// its memory requirement is historical, not a measured high-water mark.
    pub fn preparation_report(&self) -> Option<eredu_core::TextPreparationReport<'_>> {
        self.0.preparation_report()
    }

    /// Borrows the canonical committed token prefix.
    pub fn token_ids(&self) -> &[u32] {
        self.0.token_ids()
    }
    /// Returns the terminal reason once generation finishes.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.0.finish_reason()
    }
    /// Advances one shared generation step, lending output fragments to the callback.
    /// Cancellation uses the same first-terminal-reason rule as uninterrupted generation.
    pub fn advance(
        self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Self, ManagedPlainTextError> {
        self.0.advance(cancellation, emit).map(Self)
    }
    /// Freezes terminal text and IDs without copying. Returns this session if still active.
    pub fn into_output(self) -> Result<GenerationPlainTextOutput, Self> {
        self.0.into_output().map_err(Self)
    }
    /// Runs the same consuming advancement loop through termination.
    pub fn run(
        self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<GenerationPlainTextOutput, ManagedPlainTextError> {
        self.0.run(cancellation, emit)
    }
}

impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Compiles actual file bytes or this model's retained complete configuration
    /// under the backend's existing source account.
    /// Opening the file and constructing the already-loaded model remain separate operations.
    /// The complete supported tokenizer configuration must match this loaded model;
    /// equal vocabulary IDs alone do not authorize a different merge or preprocessing policy.
    pub fn compile_managed_plain_text_source(
        &self,
        input: impl Into<TokenizerSourceInput>,
    ) -> Result<ManagedPlainTextSource, ManagedPlainTextSourceError> {
        let source = match input.into() {
            TokenizerSourceInput::File(file) => {
                crate::api::tokenizer::compile_original_text_tokenizer_file(&self.runtime, file)?
            }
            TokenizerSourceInput::RetainedConfiguration => {
                B::compile_original_tokenizer_source_for_generation(
                    &self.runtime,
                    eredu_runtime::working_memory::OriginalTokenizerInput::Configuration(
                        &self.tokenizer,
                    ),
                )?
            }
        };
        if !source.matches_configuration(&self.tokenizer) {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        Ok(ManagedPlainTextSource(source))
    }

    /// Starts an originally admitted plain-text request using the existing controlled driver.
    /// Missing complete backend bounds still
    /// refuse before prompt/native preparation; this entry supplies no admission fallback.
    /// Pre-start cancellation returns None before source validation or request construction.
    pub fn start_managed_plain_text<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_plain_text_with_options(source, request, cancellation, None)
    }

    // All source validation, C/S/E/I/R construction and public error custody is
    // shared by unobserved and source-explicit observed startup.
    fn start_managed_plain_text_with_options<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        cancellation: &GenerationCancellationToken,
        options: Option<eredu_core::TextPreparationOptions>,
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        if !source.0.matches_configuration(&self.tokenizer) {
            return Err(ManagedPlainTextError::new(Cause::Input(
                TokenInputRejection::IdentityMismatch,
            )));
        }
        let (config, _) = self
            .resolve_text_generation_settings(request.settings.clone())
            .map_err(|e| ManagedPlainTextError::new(Cause::Generation(e)))?;
        let source_budget = B::prepare_original_text_source_budget(
            &self.runtime,
            &source.0,
            &request.settings.inference.memory_limits,
        )
        .map_err(|error| ManagedPlainTextError::new(Cause::Source(error)))?;
        // The exact public request/error plus internal startup return share the genuine
        // terminal consumer's fixed control envelope. The session/source wrappers are transparent.
        start_original_plain_string_with_options_for::<
            B,
            (
                ManagedPlainTextRequest<'_>,
                ManagedPlainTextError,
                OriginalPlainStartError<B::Error>,
            ),
        >(
            &mut self.runtime,
            &source.0,
            request.input,
            config,
            &self.eos_token_ids,
            request.stop_sequences,
            request.add_special_tokens,
            request.skip_special_tokens,
            cancellation,
            options,
        )
        .map(|session| session.map(ManagedPlainTextSession))
        .map_err(|error| {
            let mut error = ManagedPlainTextError::startup::<B>(error);
            error.source_budget = Some(source_budget);
            error
        })
    }

    /// Generates borrowed plain-text events and retained final output using the same
    /// managed session as explicit advancement. None means cancellation before startup.
    pub fn generate_managed_plain_text(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Option<GenerationPlainTextOutput>, ManagedPlainTextError> {
        self.start_managed_plain_text(source, request, cancellation)?
            .map(|session| session.run(cancellation, emit))
            .transpose()
    }
}

/// Immutable saved managed text state. It owns independent native and host
/// copies; it is not a resumed request or authority to restore or fork a session.
#[repr(transparent)]
pub struct ManagedPlainTextSnapshot<B: eredu_runtime::execution_control::TextSnapshotBackend>(
    super::original_token_input::plain::OriginalPlainSnapshot<B>,
);
impl<B: eredu_runtime::execution_control::TextSnapshotBackend> ManagedPlainTextSnapshot<B> {
    /// Canonical committed token prefix at the captured boundary.
    pub fn token_ids(&self) -> &[u32] {
        self.0.token_ids()
    }
    /// Terminal reason when the source was already finished.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.0.finish_reason()
    }
    /// Complete logical snapshot storage retained by its SnapshotBudget lease.
    pub fn retained_bytes(&self) -> u64 {
        self.0.retained_bytes()
    }
    /// Absolute next model prediction represented by the saved native sampler.
    pub fn next_prediction(&self) -> u64 {
        self.0.next_prediction()
    }
}

/// Neutral snapshot error, including the original backend source and any
/// independently retained partial-copy custody. No native error type escapes.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct GenerationSnapshotError(
    #[from] pub(crate) eredu_runtime::execution_control::TextSnapshotError<BackendFailure>,
);
impl GenerationSnapshotError {
    /// Borrows the shared classified snapshot failure without copying its source.
    pub fn cause(&self) -> &eredu_runtime::execution_control::TextSnapshotError<BackendFailure> {
        &self.0
    }
}
impl<B: eredu_runtime::execution_control::TextSnapshotBackend> ManagedPlainTextSession<'_, B> {
    /// Captures a completed token boundary through the shared snapshot driver.
    /// Logical limits reserve before separate host and native physical admission.
    /// Backend qualification includes the actual pending-input/state geometry;
    /// original observation checkpoints still require their own admitted copier.
    /// failure preserves the live source and consumes any attempted copy budget.
    /// This method does not enable restoration, branching or a new prediction.
    pub fn snapshot(
        &mut self,
        budget: &eredu_runtime::execution_control::SnapshotBudget,
        host_memory_limits: eredu_core::MemoryLimitDeclarations,
        native: eredu_runtime::working_memory::WorkspaceCopyLimits,
    ) -> Result<ManagedPlainTextSnapshot<B>, GenerationSnapshotError> {
        self.0
            .snapshot(budget, host_memory_limits, native)
            .map(ManagedPlainTextSnapshot)
            .map_err(GenerationSnapshotError)
    }
}

impl<B> ManagedPlainTextSnapshot<B>
where
    B: eredu_runtime::execution_control::TextSnapshotBackend + eredu_core::TextResumeBackend<ResumeSource = <B as eredu_runtime::execution_control::TextSnapshotBackend>::SavedTextComponents>,
{
    /// Remaining output allowance at the saved frontier, including no new credit.
    pub fn remaining_tokens(&self) -> Option<usize> { self.0.remaining_tokens() }
}

impl<B> LoadedModel<B>
where
    B: eredu_runtime::execution_control::TextSnapshotBackend + eredu_core::TextResumeBackend<ResumeSource = <B as eredu_runtime::execution_control::TextSnapshotBackend>::SavedTextComponents>,
{
    /// Restores an immutable managed snapshot through a freshly admitted run.
    /// The saved decoder, tokenizer/stop progress, IDs and terminal state remain
    /// untouched. Missing original native resume qualification returns a typed
    /// refusal; this is an unfinished integration rather than a model limit.
    /// Omitted output count uses the saved remaining allowance; an override may
    /// shorten it. Other sampling settings must satisfy the saved-source contract.
    /// Cancellation or an already terminal snapshot returns None without copying.
    pub fn restore_managed_plain_text<'a>(
        &'a mut self, snapshot: &ManagedPlainTextSnapshot<B>,
        settings: PreparedChatGenerationSettings, host_memory_limits: eredu_core::MemoryLimitDeclarations,
        cancellation:&GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a,B>>,ManagedPlainTextError> {
        self.resume_managed_plain_snapshot(snapshot,settings,host_memory_limits,false,cancellation)
    }

    /// Starts an independent branch at the same saved frontier. This uses the
    /// same controlled/plain driver and fresh admission as restoration, while
    /// also consuming the snapshot budget's branch slot. All attempts share its
    /// cumulative copy budget; dropping a branch never refunds completed copies.
    pub fn fork_managed_plain_text<'a>(
        &'a mut self, snapshot:&ManagedPlainTextSnapshot<B>,
        settings:PreparedChatGenerationSettings,host_memory_limits: eredu_core::MemoryLimitDeclarations,
        cancellation:&GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a,B>>,ManagedPlainTextError> {
        self.resume_managed_plain_snapshot(snapshot,settings,host_memory_limits,true,cancellation)
    }

    fn resume_managed_plain_snapshot<'a>(
        &'a mut self,snapshot:&ManagedPlainTextSnapshot<B>,
        mut settings:PreparedChatGenerationSettings,host_memory_limits: eredu_core::MemoryLimitDeclarations,branch:bool,
        cancellation:&GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a,B>>,ManagedPlainTextError> {
        if cancellation.is_cancelled() || snapshot.finish_reason().is_some() { return Ok(None); }
        if !snapshot.0.source().matches_configuration(&self.tokenizer) {
            return Err(ManagedPlainTextError::new(Cause::Input(TokenInputRejection::IdentityMismatch)));
        }
        if settings.overrides.max_new_tokens.is_none() { settings.overrides.max_new_tokens = snapshot.0.remaining_tokens(); }
        if settings.overrides.max_new_tokens == Some(0) { return Ok(None); }
        let (config,_) = self.resolve_text_generation_settings(settings).map_err(|e| ManagedPlainTextError::new(Cause::Generation(e)))?;
        snapshot.0.resume(&mut self.runtime,config,host_memory_limits,branch,cancellation)
            .map(|session| session.map(ManagedPlainTextSession))
            .map_err(|e| ManagedPlainTextError::new(Cause::Snapshot(e)))
    }
}

mod observed;
