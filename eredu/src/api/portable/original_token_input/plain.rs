//! Private original C → E → I/R plain-string composition. This supplies no
//! public default, chat rendering or owned live event. Snapshot capture uses the
//! shared completed-state driver; restore/resume remain separately qualified.
use super::*;
use crate::api::request::BackendGenerationTokenSource;
use crate::runtime::generation::streaming::RetainedConsumerCursor;
use eredu_core::{BackendFailure, GenerationDecoderError, GenerationSequenceRequest,
    TokenIdsInputPlan, TokenInputRejection};
use eredu_core::{
    BackendFailureKind, ControlledTextGenerationError, GenerationCancellationToken,
    GenerationPlainTextEvent, GenerationPlainTextOutput, GenerationSequenceConsumerLayout,
    GenerationTiming, ModelRuntime, OriginalSourceWitness, TextControllerStorage,
    TextControllerWorkspace, TextPreparationOptions, TokenFilter, TokenFilterController,
    TokenSamplingDecision,
};
use eredu_runtime::working_memory::{
    AggregateGenerationDecoderInput, OriginalTextSourceError, OriginalTokenizer,
    OriginalTokenizerBackend,
};
use eredu_text::stop_storage::{StopCompilePlan, StopSourceError};
use std::time::{Duration, Instant};
use crate::api::ManagedPlainTextError;

/// No history or callback-owned state. Its one immutable C alias is authenticated
/// by runtime before admission and before/after each actual decision callback.
struct OriginalDomainController {
    source: OriginalTokenizer,
}
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("token {0} is outside the original canonical tokenizer domain")]
pub(crate) struct OriginalDomainError(u32);
impl OriginalDomainController {
    fn domain(&self) -> &TokenFilter {
        self.source
            .generation_domain()
            .expect("selected original generation source")
    }
    fn witness(&self) -> OriginalSourceWitness<'_> {
        OriginalSourceWitness::new(&self.source)
    }
}
impl TokenFilterController for OriginalDomainController {
    type Error = OriginalDomainError;
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self.domain().into(),
            // Only the final cloned mask exists. Sampling/Work prices it; there
            // is no owned pre-override mask, history or second C source charge.
            additional_host_bytes: 0,
        })
    }
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithOriginalTokenDomain(self.witness())
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(self.domain().clone())
    }
    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        Ok(TokenSamplingDecision::new(self.current_filter()?)
            .with_original_tokenizer_validity(self.domain(), self.witness())
            .with_controller_storage(self.inference_storage()))
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        if self.domain().allows(token) {
            Ok(())
        } else {
            Err(OriginalDomainError(token))
        }
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
type PlainSource<'a, B> = BackendGenerationTokenSource<'a, B, OriginalDomainController>;
type PlainError = ControlledTextGenerationError<BackendFailure, OriginalDomainError>;
type PlainCursor =
    RetainedConsumerCursor<PlainError, GenerationDecoderError, false, true, true>;

/// Startup failures stay by value: a fixed rejection cannot allocate a new error
/// Box before I/R admission. Existing backend errors retain their actual owners.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalPlainStartError<B: std::error::Error + 'static> {
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
    #[error(transparent)]
    Stop(#[from] StopSourceError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    #[error(transparent)]
    Source(#[from] OriginalTextSourceError),
    #[error(transparent)]
    Startup(ControlledTextGenerationError<B, OriginalDomainError>),
}

/// A source plus the existing consuming cursor. Both ordinary looping and manual
/// advancement call advance_plain; there is no second generation/decoding loop.
/// Source/native/controller controls retire before the last cursor R owner.
pub(crate) struct OriginalPlainSession<'a, B: TextGenerationBackend> {
    source: PlainSource<'a, B>,
    active: Duration,
    cursor: PlainCursor,
    input_custody: Option<eredu_runtime::input::OriginalModelInputCustody>,
}
impl<'a, B: TextGenerationBackend> OriginalPlainSession<'a, B> {
    pub(crate) fn retain_input_custody(
        &mut self,
        custody: Option<eredu_runtime::input::OriginalModelInputCustody>,
    ) {
        self.input_custody = custody;
    }
    /// Only changes the borrowed delivery callback. Source admission, capture
    /// ledger and native completion remain owned by the existing generator.
    pub(crate) fn set_capture_observer(
        &mut self,
        observer: &'a mut dyn FnMut(
            Option<u32>,
            Option<eredu_core::capture::SharedCapturedStep>,
            f64,
        ),
    ) {
        self.source.on_token = Some(observer);
    }

    pub(crate) fn preparation_report(&self) -> Option<eredu_core::TextPreparationReport<'_>> {
        self.source.generator.preparation_report()
    }
    pub(crate) fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    pub(crate) fn finish_reason(&self) -> Option<eredu_core::FinishReason> {
        self.cursor.finish_reason()
    }
    pub(crate) fn advance(
        self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Self, ManagedPlainTextError> {
        let Self {
            mut source,
            active,
            cursor,
            input_custody,
        } = self;
        let started = Instant::now();
        // Count prior active preparation/advancement, excluding caller pauses.
        source.generation_started = started
            .checked_sub(active)
            .expect("active intervals are contained in this process clock history");

        match cursor.advance_plain(&mut source, cancellation, emit) {
            Ok(cursor) => Ok(Self {
                source,
                active: active + started.elapsed(),
                cursor,
                input_custody,
            }),
            Err(failure) => {
                use crate::runtime::generation::streaming::CommittedGenerationError;
                let (kind,operation) = match failure.cause() {
                    CommittedGenerationError::Source(
                        ControlledTextGenerationError::Preparation(error)|ControlledTextGenerationError::Backend(error),
                    ) => (error.kind(),error.operation()),
                    _ => (BackendFailureKind::Other,"plain text advancement"),
                };
                drop(source);
                Err(ManagedPlainTextError::from_step(failure.into_backend_failure(kind).with_operation(operation), input_custody))
            }
        }
    }
    pub(crate) fn into_output(self) -> Result<GenerationPlainTextOutput, Self> {
        if self.finish_reason().is_none() {
            return Err(self);
        }
        let Self {
            source,
            active: _,
            cursor,
            input_custody: _,
        } = self;
        let timing = GenerationTiming::new(source.time_to_first_token);
        drop(source);
        Ok(cursor
            .into_text_output(timing)
            .unwrap_or_else(|_| unreachable!("terminal checked above")))
    }
    pub(crate) fn run(
        mut self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<GenerationPlainTextOutput, ManagedPlainTextError> {
        while self.finish_reason().is_none() {
            self = self.advance(cancellation, emit)?;
        }
        Ok(self
            .into_output()
            .unwrap_or_else(|_| unreachable!("same terminal cursor")))
    }
}

// Prepared input is lent by its owning public request until the core consumes it.
// The enum itself never clones a native prompt or allocates a wrapper.
enum PlainInput<'a, B: TextGenerationBackend> {
    Text { input: &'a str, add_special_tokens: bool },
    Prepared(&'a mut Option<B::Prompt>),
}
type StartupControls<'a, B: TextGenerationBackend, E> = (
    E, Option<TextPreparationOptions>, PlainInput<'a, B>,
    Option<eredu_runtime::working_memory::OriginalEncodedTokenIds>,
    ManagedPlainTextError,
    Result<OriginalPlainSession<'a, B>, ManagedPlainTextError>,
);

/// Concrete fixed consumer envelope shared by plain/chat construction. Lifetimes
/// remain ordinary borrows; this descriptor confers no source/account authority.
pub(super) fn plain_consumer_layout<B: TextGenerationBackend, E>()
-> Option<GenerationSequenceConsumerLayout> {
    PlainCursor::terminal_layout::<
        PlainSource<'_, B>,
        OriginalPlainSession<'_, B>,
        StartupControls<'_, B, E>,
    >()
}

/// Same original startup, consuming the exact immutable source options through
/// core admission. No ordinary capture configuration is installed afterward.
pub(in crate::api::portable) fn start_original_plain_string_with_options_for<'a, B, E>(
    runtime: &'a mut ModelRuntime<B>,
    source: &OriginalTokenizer,
    input: &str,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    add_special_tokens: bool,
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
    options: Option<TextPreparationOptions>,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalPlainStartError<B::Error>>
where
    B: OriginalTokenizerBackend,
{
    start_original_plain_input_with_options_for::<B, E>(
        runtime, source, PlainInput::Text { input, add_special_tokens }, config,
        eos, stops, skip_special_tokens, cancellation, options,
    )
}

pub(in crate::api::portable) fn start_original_prepared_input_with_options_for<'a, B, E>(
    runtime: &'a mut ModelRuntime<B>,
    source: &OriginalTokenizer,
    prompt: B::Prompt,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
    options: Option<TextPreparationOptions>,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalPlainStartError<B::Error>>
where B: OriginalTokenizerBackend,
{
    let mut prompt = Some(prompt);
    start_original_plain_input_with_options_for::<B, E>(
        runtime, source, PlainInput::Prepared(&mut prompt), config,
        eos, stops, skip_special_tokens, cancellation, options,
    )
}

fn start_original_plain_input_with_options_for<'a, B, E>(
    runtime: &'a mut ModelRuntime<B>,
    source: &OriginalTokenizer,
    input: PlainInput<'_, B>,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
    options: Option<TextPreparationOptions>,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalPlainStartError<B::Error>>
where B: OriginalTokenizerBackend,
{
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let started = Instant::now();
    let maximum = config
        .sampling()
        .max_new_tokens
        .ok_or(TokenInputRejection::Unsupported)?;
    if maximum == 0 {
        return Err(eredu_core::generation::GenerationError::ZeroTokenBudget.into());
    }
    let domain = source
        .generation_domain()
        .ok_or(TokenInputRejection::Unsupported)?;
    if !matches!(domain, TokenFilter::Allowed(mask) if mask.iter().any(|bit| *bit))
        || eos.iter().any(|&id| !domain.allows(id))
    {
        return Err(TokenInputRejection::InvalidToken.into());
    }
    B::validate_original_tokenizer_source(runtime, source)?;
    let consumer = plain_consumer_layout::<B, E>().ok_or(TokenInputRejection::Overflow)?;
    let stops =
        B::compile_original_text_stop_source(runtime, StopCompilePlan::prepare_refs(stops)?)?;
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let encoded = match &input {
        PlainInput::Text { input, add_special_tokens } => {
            let encoded = B::encode_original_text_ids(runtime, source, input, *add_special_tokens)?;
            if !encoded.matches_source(source) || encoded.ids().iter().any(|&id| !domain.allows(id)) {
                return Err(TokenInputRejection::IdentityMismatch.into());
            }
            Some(encoded)
        }
        PlainInput::Prepared(prompt) => {
            B::validate_original_prepared_input_domain(runtime,
                prompt.as_ref().ok_or(TokenInputRejection::IdentityMismatch)?, source)?;
            None
        }
    };
    if cancellation.is_cancelled() { return Ok(None); }
    let header = AggregateGenerationDecoderInput::new_plain_text(
        source,
        &stops,
        maximum,
        skip_special_tokens,
    )
    .map_err(|_| TokenInputRejection::IdentityMismatch)?;
    let request = GenerationSequenceRequest::new(maximum, eos)
        .with_decoder(&header)
        .with_consumer(&consumer);
    let controller = OriginalDomainController {
        source: source.clone(),
    };
    let mut generator = match input {
        PlainInput::Text { .. } => {
            let encoded = encoded.as_ref().ok_or(TokenInputRejection::IdentityMismatch)?;
            let plan = TokenIdsInputPlan::new(encoded.ids())?;
            eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
                runtime, plan, config, controller, options, request,
            )
        }
        PlainInput::Prepared(prompt) => eredu_core::ControlledTextGeneration::from_input_with_sequence(
            runtime, eredu_core::TextGenerationInput::OriginalPrepared(
                prompt.take().ok_or(TokenInputRejection::IdentityMismatch)?),
            config, controller, options, request,
        ),
    }.map_err(OriginalPlainStartError::Startup)?;
    // Synchronous core construction has consumed the genuine I claim and copied
    // the IDs; it does not retain the E source borrow. Never adopt E's Vec.
    drop(encoded);
    let sequence = generator
        .take_prepared_sequence()
        .expect("genuine original I/R path");
    let cursor = PlainCursor::from_terminal_sequence::<
        PlainSource<'_, B>,
        OriginalPlainSession<'_, B>,
        StartupControls<'_, B, E>,
    >(sequence)
    .unwrap_or_else(|_| unreachable!("core checked this concrete consumer"));
    Ok(Some(OriginalPlainSession {
        input_custody: None,
        source: BackendGenerationTokenSource {
            generator,
            on_token: None,
            delivery_failure: None,
            generation_started: started,
            time_to_first_token: None,
        },
        active: started.elapsed(),
        cursor,
    }))
}

mod snapshot;
pub(crate) use snapshot::OriginalPlainSnapshot;
