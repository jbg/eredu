//! Borrowed original sequence request and the shared machine's one-use claim.

use super::*;
use crate::generation::RetainedGenerationSequence;
mod consumer;
mod decoder;
pub use consumer::GenerationSequenceConsumerLayout;
pub use decoder::{
    GenerationDecoderError, GenerationDecoderInput, GenerationDecoderOutput, GenerationPlainText,
    GenerationPlainTextEvent, GenerationPlainTextEvents, GenerationPlainTextProjection,
};

/// Immutable EOS/max input to original synchronous text admission.
///
/// This borrows the caller's actual EOS storage without sorting or copying it.
/// Order and duplicates are permitted; the admitted provider prepares its own
/// immutable sorted policy. This request grants no storage or execution budget.
#[derive(Debug)]
pub struct GenerationSequenceRequest<'e> {
    max_new_tokens: usize,
    eos_token_ids: &'e [u32],
    consumer: Option<&'e GenerationSequenceConsumerLayout>,
    decoder: Option<&'e dyn GenerationDecoderInput>,
    input: Option<&'e TokenIdsInputPlan<'e>>,
}

impl<'e> GenerationSequenceRequest<'e> {
    /// Borrows resolved policy; the shared machine checks the exact config limit.
    pub const fn new(max_new_tokens: usize, eos_token_ids: &'e [u32]) -> Self {
        Self {
            max_new_tokens,
            eos_token_ids,
            consumer: None,
            decoder: None,
            input: None,
        }
    }
    /// Couples the actual borrowed prompt to the same original admission.
    /// Must be paired with `TextGenerationInput::OriginalTokenIds`.
    pub fn with_token_input<'i>(
        self,
        input: &'i TokenIdsInputPlan<'i>,
    ) -> GenerationSequenceRequest<'i>
    where
        'e: 'i,
    {
        GenerationSequenceRequest {
            max_new_tokens: self.max_new_tokens,
            eos_token_ids: self.eos_token_ids,
            consumer: self.consumer,
            decoder: self.decoder,
            input: Some(input),
        }
    }
    /// Exact source borrowed by this synchronous original request.
    pub const fn token_input(&self) -> Option<&'e TokenIdsInputPlan<'e>> {
        self.input
    }
    /// Borrows a completed concrete consumer contribution before original admission.
    /// This diagnostic grants neither storage nor work; the backend must explicitly
    /// opt in and seal it with the same request before returning a matching provider.
    pub fn with_consumer(mut self, consumer: &'e GenerationSequenceConsumerLayout) -> Self {
        self.consumer = Some(consumer);
        self
    }
    /// Borrows one concrete source-bearing decoder input for original admission.
    /// The backend must explicitly accept this input and move its unique source
    /// before adaptive candidate retries. Core owns neither tokenizer policy nor
    /// source compilation. An existing sequence/consumer hook cannot accept it.
    pub fn with_decoder(mut self, decoder: &'e dyn GenerationDecoderInput) -> Self {
        self.decoder = Some(decoder);
        self
    }
    /// Exact synchronous decoder input; no byte allowance or mutable owner exit.
    pub const fn decoder_input(&self) -> Option<&'e dyn GenerationDecoderInput> {
        self.decoder
    }
    /// Concrete immutable consumer contribution, if requested originally.
    pub const fn consumer_layout(&self) -> Option<&'e GenerationSequenceConsumerLayout> {
        self.consumer
    }
    /// Finite output allowance supplied to original admission.
    pub const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }
    /// Original immutable EOS storage, including its order and duplicates.
    pub const fn eos_token_ids(&self) -> &'e [u32] {
        self.eos_token_ids
    }
}

/// Move-only claim issued inside the existing original Admission operation.
///
/// Admission first borrows this claim; after original run binding, extraction
/// consumes that same claim. Neither its request nor context can be replaced.
/// The borrow ends with synchronous preparation: retained runtime ownership must
/// be the actual accepted plan/bank, not an address retained from this claim.
/// Context clones are identity evidence only, never allocation permission.
///
/// ```compile_fail
/// fn duplicate(claim: eredu_core::GenerationSequencePreparation<'_, '_>) {
///     let second = claim.clone();
/// }
/// ```
#[derive(Debug)]
pub struct GenerationSequencePreparation<'r, 'e> {
    request: &'r GenerationSequenceRequest<'e>,
    context: &'r TextStepContext,
}

impl<'r, 'e> GenerationSequencePreparation<'r, 'e> {
    pub(super) fn new(
        request: &'r GenerationSequenceRequest<'e>,
        context: &'r TextStepContext,
    ) -> Self {
        Self { request, context }
    }
    /// The exact borrowed request supplied before original admission.
    pub const fn request(&self) -> &'r GenerationSequenceRequest<'e> {
        self.request
    }
    /// Genuine original run/policy context at attempt zero.
    pub const fn context(&self) -> &'r TextStepContext {
        self.context
    }
}

/// Neutral rejection at the original sequence preparation boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum GenerationSequenceAdmissionError {
    /// The backend has no closed original-admission sequence provider.
    #[error("backend has no original-admission generation sequence provider")]
    Unsupported,
    /// Existing sequence support alone cannot accept a consumer contribution.
    #[error("backend has no original-admission generation sequence consumer support")]
    ConsumerUnsupported,
    /// Existing sequence/consumer support does not admit a decoder source.
    #[error("backend has no original-admission decoder source support")]
    DecoderUnsupported,
    /// Suffix-only support cannot silently omit an original stop projection.
    #[error("backend has no original-admission plain text projection support")]
    PlainTextUnsupported,
    /// The request differs from the resolved finite generation allowance.
    #[error("generation sequence allowance differs from the original finite config")]
    RequestMismatch,
    /// The returned sequence does not match the original request or initial state.
    #[error("admitted generation sequence differs from its original request")]
    InvalidSequence,
    /// Mutable retained sequence copying has no separately admitted destination.
    #[error("retained generation sequence copy has no admitted destination")]
    CopyNotAdmitted,
}

impl GenerationSequenceAdmissionError {
    /// Requested storage and core error controls for rejecting one returned
    /// retained sequence whose policy differs from its genuine original claim.
    ///
    /// This measures the actual private rejection envelope and its concrete
    /// BackendFailure source retirement. The provider's token/EOS buffers and
    /// ownership are separate original facts. No allocation, rejection, funding
    /// or completion authority is created. `None` means checked size overflow.
    pub fn rejected_sequence_retention_peak_bytes() -> Option<usize> {
        BackendFailure::source_retention_peak_bytes::<RejectedSequence>()
    }
}

#[derive(Debug)]
pub(in crate::backend) enum PreparedSequence {
    Legacy,
    // None means already taken, not permission to fall back to legacy copying.
    Retained(Option<RetainedGenerationSequence>),
}
impl PreparedSequence {
    pub(in crate::backend) fn take(&mut self) -> Option<RetainedGenerationSequence> {
        match self {
            Self::Legacy => None,
            Self::Retained(value) => value.take(),
        }
    }
    pub(in crate::backend) fn requires_copy_admission(&self) -> bool {
        matches!(self, Self::Retained(_))
    }
    pub(in crate::backend) fn install(
        sequence: RetainedGenerationSequence,
        request: &GenerationSequenceRequest<'_>,
    ) -> Result<Self, BackendFailure> {
        if !sequence.matches_preparation(request.max_new_tokens, request.eos_token_ids)
            || sequence.consumer_layout() != request.consumer_layout()
            || !sequence.matches_decoder_input(request.decoder_input())
            || sequence.decoder_output()
                != request.decoder_input().map_or(
                    GenerationDecoderOutput::Suffix,
                    GenerationDecoderInput::output_kind,
                )
        {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidInput,
                RejectedSequence {
                    cause: GenerationSequenceAdmissionError::InvalidSequence,
                    _sequence: sequence,
                },
            ));
        }
        Ok(Self::Retained(Some(sequence)))
    }
}

// Keep the rejected provider's payload/custody alive with the escaped cause.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RejectedSequence {
    #[source]
    cause: GenerationSequenceAdmissionError,
    _sequence: RetainedGenerationSequence,
}

impl<'a, B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'a, B, C> {
    /// Original synchronous preparation with a borrowed finite sequence request.
    /// Existing options retain their exact Instrumentation behavior; None plus
    /// a sequence alone does not add that stage. No prediction is performed.
    pub fn from_input_with_sequence(
        runtime: &'a mut ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let inner = TextGenerationMachine::new_preparation_with_sequence(
            runtime,
            input,
            config,
            controller,
            options,
            Some(sequence),
        )?;
        Ok(Self { runtime, inner })
    }
    /// Moves out the original retained sequence at most once. Before prediction,
    /// its caller must use prepare_storage's local result in the existing first
    /// finish_step readiness; this accessor performs neither work nor a vote.
    pub fn take_prepared_sequence(&mut self) -> Option<RetainedGenerationSequence> {
        self.inner.prepared_sequence.take()
    }
}
impl<'a, B: TextGenerationBackend> TextGeneration<'a, B> {
    /// Ordinary fixed-filter counterpart of the controlled original sequence entry.
    pub fn from_input_with_sequence(
        runtime: &'a mut ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        filter: TokenFilter,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<Self, BackendFailure> {
        let inner = TextGenerationMachine::new_preparation_with_sequence(
            runtime,
            input,
            config,
            FixedTokenFilter(filter),
            options,
            Some(sequence),
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(Self { runtime, inner })
    }
    /// Takes the original sequence once; materialization/readiness stay with its caller.
    pub fn take_prepared_sequence(&mut self) -> Option<RetainedGenerationSequence> {
        self.inner.prepared_sequence.take()
    }
}
