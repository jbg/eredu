//! Borrowed source and thin adapters for original host input construction.
use super::*;

/// A checked immutable U32 source. This plan allocates nothing and grants no
/// memory or execution authority. Its caller backing remains caller-owned.
#[derive(Debug)]
pub struct TokenIdsInputPlan<'a> {
    tokens: &'a [u32],
}
impl<'a> TokenIdsInputPlan<'a> {
    /// Checks a positive representable destination before original admission.
    pub fn new(tokens: &'a [u32]) -> Result<Self, TokenInputRejection> {
        if tokens.is_empty() {
            return Err(TokenInputRejection::Empty);
        }
        std::alloc::Layout::array::<u32>(tokens.len())
            .map_err(|_| TokenInputRejection::Overflow)?;
        u64::try_from(tokens.len())
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or(TokenInputRejection::Overflow)?;
        Ok(Self { tokens })
    }
    /// Actual immutable source; no mutable or owned storage export.
    pub const fn tokens(&self) -> &'a [u32] {
        self.tokens
    }
    /// Exact requested U32 allocation, excluding caller capacity and controls.
    pub fn destination_bytes(&self) -> u64 {
        self.tokens.len() as u64 * 4
    }
}

impl<'a, B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'a, B, C> {
    /// Starts the same machine with one originally constructed input destination.
    /// The source borrow ends with startup, not with the returned generator.
    pub fn from_token_ids_with_sequence(
        runtime: &'a mut ModelRuntime<B>,
        input: TokenIdsInputPlan<'_>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_input_with_sequence(
            runtime,
            TextGenerationInput::OriginalTokenIds,
            config,
            controller,
            options,
            sequence.with_token_input(&input),
        )
    }
}
impl<'a, B: TextGenerationBackend> TextGeneration<'a, B> {
    /// Fixed-filter counterpart of the original borrowed input entry.
    pub fn from_token_ids_with_sequence(
        runtime: &'a mut ModelRuntime<B>,
        input: TokenIdsInputPlan<'_>,
        config: TextGenerationConfig,
        filter: TokenFilter,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<Self, BackendFailure> {
        Self::from_input_with_sequence(
            runtime,
            TextGenerationInput::OriginalTokenIds,
            config,
            filter,
            options,
            sequence.with_token_input(&input),
        )
    }
}
impl<B: TextGenerationBackend> TextGenerationDriver<'_, B> {
    /// Detached startup through the same original preparation and machine.
    pub fn start_token_ids_with_sequence<C: TokenFilterController>(
        &mut self,
        input: TokenIdsInputPlan<'_>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
        sequence: GenerationSequenceRequest<'_>,
    ) -> Result<TextGenerationContinuation<B, C>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.start_input_with_sequence(
            TextGenerationInput::OriginalTokenIds,
            config,
            controller,
            options,
            sequence.with_token_input(&input),
        )
    }
}
