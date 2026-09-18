//! Original active grammar state consuming the same ordinary terminal policy.
use super::{
    OriginalGrammarForcingError, OriginalGrammarSlicer, OriginalGrammarTokenParser,
    OriginalGrammarTokenParserCopyError, OriginalGrammarTokenParserError,
};
use crate::runtime::chat::constraints::grammar_policy::{self, Context};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use llguidance::{api::StopReason, toktrie::SimpleVob};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::runtime::chat::constraints) enum Cause {
    #[error("original grammar state source is unavailable")]
    Source,
    #[error("original grammar state control extent overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Policy(#[from] grammar_policy::Failure),
    #[error(transparent)]
    Parser(#[from] OriginalGrammarTokenParserError),
    #[error(transparent)]
    Forcing(#[from] OriginalGrammarForcingError),
    #[error(transparent)]
    Tokenization(#[from] super::super::OriginalGrammarTokenizationError),
    #[error(transparent)]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
    #[error("grammar tokenizer could not represent activation bytes exactly")]
    ActivationEncoding,
}
#[derive(Debug, thiserror::Error)]
enum OperationControl {
    #[error("original grammar state control extent overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
}
#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error(transparent)]
    Operation(#[from] Cause),
    #[error(transparent)]
    Copy(#[from] OriginalGrammarTokenParserCopyError),
}
// The fixed cell holds both the successful parser and an owning failed prefix.
// It is paid before allocation; an operation moves into its existing failure
// slot instead of multiplying the full parser through every Result transport.
#[derive(Debug)]
struct Storage {
    parser: Option<OriginalGrammarTokenParser>,
    failure: Option<Failure>,
}
/// Actual active semantic state. Its paid cell retires before the funding;
/// either the parser or the failed worker retains the immutable source chain.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarState {
    storage: Box<Storage>,
    terminal_eos_alias_committed: bool,
    funding: HostMetadataFunding,
}
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarStateError {
    prefix: OriginalGrammarState,
}
impl std::fmt::Display for OriginalGrammarStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(
            self.prefix
                .storage
                .failure
                .as_ref()
                .expect("failed active grammar"),
            f,
        )
    }
}
impl std::error::Error for OriginalGrammarStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(
            self.prefix
                .storage
                .failure
                .as_ref()
                .expect("failed active grammar"),
        )
    }
}
#[derive(Debug, thiserror::Error)]
enum ConstructionCause {
    #[error("original active grammar state constructor extent overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
}
/// Before the paid cell exists, a refusal retains the actual unboxed parser.
/// This cold constructor failure does not enlarge ordinary operation errors.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarStateConstructionError {
    #[source]
    cause: ConstructionCause,
    prefix: OriginalGrammarTokenParser,
    funding: HostMetadataFunding,
}
impl OriginalGrammarTokenParser {
    #[inline(never)]
    pub(in crate::runtime::chat::constraints) fn into_active(
        self,
    ) -> Result<OriginalGrammarState, OriginalGrammarStateConstructionError> {
        let funding = self.lexer.funding.clone();
        let result = (|| -> Result<(), ConstructionCause> {
            let parts = [
                size_of::<Self>(),
                size_of::<Storage>(),
                size_of::<Box<Storage>>(),
                size_of::<OriginalGrammarState>(),
                size_of::<ConstructionCause>(),
                size_of::<OriginalGrammarStateConstructionError>(),
                size_of::<Result<OriginalGrammarState, OriginalGrammarStateConstructionError>>(),
                size_of::<Result<(), ConstructionCause>>(),
                size_of::<HostMetadataFunding>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(ConstructionCause::Overflow)?,
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(OriginalGrammarState {
                storage: Box::new(Storage {
                    parser: Some(self),
                    failure: None,
                }),
                terminal_eos_alias_committed: false,
                funding,
            }),
            Err(cause) => Err(OriginalGrammarStateConstructionError {
                cause,
                prefix: self,
                funding,
            }),
        }
    }
}
impl OriginalGrammarState {
    pub(in crate::runtime::chat::constraints) fn copy_required_bytes(&self) -> Option<usize> {
        if self.storage.failure.is_some() {
            return None;
        }
        Self::copy_controls()?.checked_add(self.storage.parser.as_ref()?.copy_required_bytes()?)
    }
    fn copy_controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<CopyCause>(),
            size_of::<OriginalGrammarStateCopyError>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Arc<OriginalGrammarSlicer>>(),
            size_of::<Storage>(),
            size_of::<Box<Storage>>(),
            size_of::<(&Self, &HostMetadataFunding)>(),
            size_of::<Result<Self, OriginalGrammarStateCopyError>>(),
            size_of::<Option<OriginalGrammarState>>(),
            size_of::<Result<(), CopyCause>>(),
            size_of::<Option<Failure>>(),
            size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserCopyError>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn controller_funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(super) fn is_initial_controller(&self) -> bool {
        !self.terminal_eos_alias_committed && self.parser().parser().tokens().is_empty()
    }
    pub(in crate::runtime::chat::constraints) fn into_controller(
        self,
        capacity: usize,
        validity: eredu_core::SharedTokenFilter,
    ) -> Result<
        super::OriginalPreparedGrammarController,
        super::OriginalPreparedGrammarControllerError,
    > {
        super::OriginalPreparedGrammarController::new(self, capacity, validity)
    }

    pub(in crate::runtime::chat::constraints) fn into_auto_controller(
        self,
        capacity: usize,
        validity: eredu_core::SharedTokenFilter,
    ) -> Result<
        super::OriginalPreparedGrammarController,
        super::OriginalPreparedGrammarControllerError,
    > {
        super::OriginalPreparedGrammarController::new_auto(self, capacity, validity)
    }

    /// Runs the ordinary activation trial with the actual original tokenizer.
    /// Whole-token activation preserves its ID; cross-token activation uses the
    /// same special-aware byte tokenizer and exact ordinary decoded round trip.
    pub(super) fn try_activation(
        self,
        token: u32,
        found: eredu_core::speculative::byte_trigger::TriggerMatch<'_>,
    ) -> Result<(Self, bool), OriginalGrammarStateError> {
        if found.starts_at_token_boundary {
            return self.try_commit(token);
        }
        self.operation(|owner| {
            use super::super::{
                GrammarTokenizationMode, OriginalGrammarTokenIds, OriginalGrammarTokenizationError,
            };
            use eredu_core::{HostPreparationAuthority, SpeculativeBuffer};
            let capacity = found
                .prefix
                .len()
                .checked_add(found.tail.len())
                .ok_or(Cause::Overflow)?;
            let parts = [
                SpeculativeBuffer::<u8>::retained_control_bytes(capacity).ok_or(Cause::Overflow)?,
                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                    .ok_or(Cause::Overflow)?,
                llguidance::toktrie::TokTrie::decoded_tokens_match_control_bytes()
                    .ok_or(Cause::Overflow)?,
                size_of::<eredu_core::speculative::byte_trigger::TriggerMatch<'_>>(),
                size_of::<OriginalGrammarTokenIds>(),
                size_of::<Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>>(),
                size_of::<SpeculativeBuffer<u8>>(),
                size_of::<
                    Result<SpeculativeBuffer<u8>, eredu_core::SpeculativeBufferAllocationError>,
                >(),
                size_of::<Result<(), eredu_core::GenerationError>>(),
                size_of::<HostPreparationAuthority>(),
                size_of::<(usize, u32)>(),
                size_of::<
                    std::iter::Copied<
                        std::iter::Chain<std::slice::Iter<'_, u8>, std::slice::Iter<'_, u8>>,
                    >,
                >(),
            ];
            owner.funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let mut bytes = SpeculativeBuffer::try_new_retained(
                capacity,
                HostPreparationAuthority::retain(owner.funding.clone()),
            )?;
            bytes
                .try_extend(found.prefix.iter().chain(found.tail).copied())
                .map_err(|_| Cause::Source)?;
            let vocabulary = owner.parser().vocabulary();
            let tokens = vocabulary.tokenize_bytes_with_funding(
                &bytes,
                GrammarTokenizationMode::Special,
                &owner.funding,
            )?;
            if !vocabulary
                .trie_source()
                .trie()
                .decoded_tokens_match(tokens.ids(), &bytes)
            {
                return Err(Cause::ActivationEncoding);
            }
            let parser = owner.storage.parser.take().ok_or(Cause::Source)?;
            let (parser, consumed) = parser.try_consume_tokens(tokens.ids())?;
            owner.storage.parser = Some(parser);
            Ok(consumed == tokens.ids().len())
        })
    }

    #[inline(never)]
    fn operation<T, F>(mut self, run: F) -> Result<(Self, T), OriginalGrammarStateError>
    where
        F: FnOnce(&mut Self) -> Result<T, Cause>,
    {
        if let Err(cause) = self.prepare_operation::<T, F>() {
            self.record_control(cause);
            return Err(OriginalGrammarStateError { prefix: self });
        }
        match self.run_operation(run) {
            Ok(value) => Ok((self, value)),
            Err(()) => Err(OriginalGrammarStateError { prefix: self }),
        }
    }
    // The complete Cause stays in this bounded callback frame and is moved
    // directly into the cell paid when this active state was constructed.
    #[inline(never)]
    fn run_operation<T, F>(&mut self, run: F) -> Result<T, ()>
    where
        F: FnOnce(&mut Self) -> Result<T, Cause>,
    {
        match run(self) {
            Ok(value) => Ok(value),
            Err(cause) => {
                self.storage.failure = Some(cause.into());
                Err(())
            }
        }
    }
    #[inline(never)]
    fn record_control(&mut self, cause: OperationControl) {
        let cause = match cause {
            OperationControl::Overflow => Cause::Overflow,
            OperationControl::Funding(cause) => Cause::Funding(cause),
        };
        self.storage.failure = Some(cause.into());
    }
    #[inline(never)]
    fn prepare_operation<T, F>(&self) -> Result<(), OperationControl>
    where
        F: FnOnce(&mut Self) -> Result<T, Cause>,
    {
        let parts = [
            grammar_policy::control_bytes::<Self>().ok_or(OperationControl::Overflow)?,
            eredu_core::PackedTokenFilter::control_bytes(),
            size_of::<(
                &SimpleVob,
                &[u32],
                Option<&[u32]>,
                usize,
                std::ops::RangeTo<usize>,
            )>(),
            size_of::<Self>(),
            size_of::<Cause>(),
            size_of::<OperationControl>(),
            size_of::<Result<(), OperationControl>>(),
            size_of::<Result<T, ()>>(),
            size_of::<(&mut Self, F)>(),
            size_of::<(&mut Self, OperationControl)>(),
            size_of::<Failure>(),
            size_of::<OriginalGrammarStateError>(),
            size_of::<F>(),
            size_of::<T>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<(Self, T), OriginalGrammarStateError>>(),
            size_of::<Result<T, Cause>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<Result<(OriginalGrammarTokenParser, usize), OriginalGrammarTokenParserError>>(
            ),
            size_of::<Result<(OriginalGrammarTokenParser, bool), OriginalGrammarTokenParserError>>(
            ),
            size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarForcingError>>(),
            size_of::<Option<OriginalGrammarTokenParser>>(),
            size_of::<[u32; 1]>(),
            size_of::<(&mut Self, u32)>(),
        ];
        self.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(OperationControl::Overflow)?,
        )?;
        Ok(())
    }
    fn mask_inner(&mut self) -> Result<(), Cause> {
        let parser = self.storage.parser.take().ok_or(Cause::Source)?;
        self.storage.parser = Some(parser.compute_mask()?);
        Ok(())
    }
    pub(in crate::runtime::chat::constraints) fn compute_mask(
        self,
    ) -> Result<Self, OriginalGrammarStateError> {
        self.operation(Self::mask_inner).map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn token_mask(&self) -> Option<&SimpleVob> {
        self.storage.parser.as_ref()?.parser().token_mask()
    }
    /// Loans the exact completed paid parser mask; mutation cannot overlap this
    /// borrow. Source/account authentication remains the caller's separate join.
    pub(in crate::runtime::chat::constraints) fn packed_filter<'a>(
        &'a self,
        validity: &'a eredu_core::SharedTokenFilter,
    ) -> Result<eredu_core::PackedTokenFilter<'a>, eredu_core::PackedTokenFilterError> {
        let mask = self
            .token_mask()
            .ok_or(eredu_core::PackedTokenFilterError::Geometry)?;
        // TokTrie retains an extra traversal sentinel bit. A vocabulary ending
        // on a word boundary therefore owns one more backing word than its
        // sampling domain; loan the logical prefix without changing custody.
        let words = mask
            .as_slice()
            .get(..mask.len().div_ceil(u32::BITS as usize))
            .ok_or(eredu_core::PackedTokenFilterError::Geometry)?;
        eredu_core::PackedTokenFilter::new(words, mask.len(), validity)
    }
    pub(in crate::runtime::chat::constraints) fn parser(&self) -> &OriginalGrammarTokenParser {
        self.storage
            .parser
            .as_ref()
            .expect("complete active grammar state")
    }
    pub(in crate::runtime::chat::constraints) fn commit(
        self,
        token: u32,
    ) -> Result<Self, OriginalGrammarStateError> {
        self.operation(|state| grammar_policy::commit(state, token))
            .map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn try_commit(
        self,
        token: u32,
    ) -> Result<(Self, bool), OriginalGrammarStateError> {
        self.operation(|state| state.try_token(token))
    }
    pub(in crate::runtime::chat::constraints) fn is_complete(
        self,
    ) -> Result<(Self, bool), OriginalGrammarStateError> {
        self.operation(grammar_policy::complete)
    }
    pub(in crate::runtime::chat::constraints) fn is_terminal(
        self,
    ) -> Result<(Self, bool), OriginalGrammarStateError> {
        self.operation(grammar_policy::terminal)
    }
}
impl Context for OriginalGrammarState {
    type Error = Cause;
    fn alias_committed(&self) -> bool {
        self.terminal_eos_alias_committed
    }
    fn mark_alias(&mut self) {
        self.terminal_eos_alias_committed = true;
    }
    fn try_token(&mut self, token: u32) -> Result<bool, Cause> {
        let parser = self.storage.parser.take().ok_or(Cause::Source)?;
        let (parser, consumed) = parser.try_consume_tokens(&[token])?;
        self.storage.parser = Some(parser);
        Ok(consumed == 1)
    }
    fn is_eos(&self, token: u32) -> Result<bool, Cause> {
        Ok(self
            .storage
            .parser
            .as_ref()
            .ok_or(Cause::Source)?
            .vocabulary()
            .trie_source()
            .trie()
            .eos_tokens()
            .contains(&token))
    }
    fn mask_allows(&mut self, token: u32) -> Result<bool, Cause> {
        self.mask_inner()?;
        Ok(self.token_mask().ok_or(Cause::Source)?.is_allowed(token))
    }
    fn accepting(&mut self) -> Result<bool, Cause> {
        let parser = self.storage.parser.take().ok_or(Cause::Source)?;
        let (parser, accepting) = parser.is_accepting()?;
        self.storage.parser = Some(parser);
        Ok(accepting)
    }
    fn stopped(&self) -> bool {
        self.storage
            .parser
            .as_ref()
            .is_none_or(|parser| parser.parser().stop_reason() != StopReason::NotStopped)
    }
    fn refusal(&self, cause: grammar_policy::Failure) -> Cause {
        cause.into()
    }
}
#[derive(Debug, thiserror::Error)]
enum CopyCause {
    #[error("original active grammar copy extent overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error("original active grammar parser copy failed")]
    Parser,
}
/// The failed parser copy stays in the cell paid before copying began. The
/// outer error can therefore traverse sampler and native publication frames
/// without carrying the complete lexer/chart failure inline at each boundary.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarStateCopyError {
    cause: CopyCause,
    prefix: Option<OriginalGrammarState>,
    source: Arc<OriginalGrammarSlicer>,
    funding: HostMetadataFunding,
}
impl OriginalGrammarStateCopyError {
    fn source_cause(&self) -> &(dyn std::error::Error + 'static) {
        match self.prefix.as_ref() {
            Some(prefix) => prefix
                .storage
                .failure
                .as_ref()
                .expect("failed copied parser"),
            None => &self.cause,
        }
    }
}
impl std::fmt::Display for OriginalGrammarStateCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.source_cause(), f)
    }
}
impl std::error::Error for OriginalGrammarStateCopyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source_cause())
    }
}
impl OriginalGrammarState {
    #[inline(never)]
    pub(in crate::runtime::chat::constraints) fn try_copy(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalGrammarStateCopyError> {
        let original = self.parser();
        let reservation = Self::copy_controls()
            .ok_or(CopyCause::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes).map_err(CopyCause::Funding));
        if let Err(cause) = reservation {
            return Err(OriginalGrammarStateCopyError {
                cause,
                prefix: None,
                source: Arc::clone(&original.lexer.source),
                funding: funding.clone(),
            });
        }
        // No nested constructor can run before its failure destination exists.
        // Successful and failed copies retain this same paid cell; no allocation
        // is needed after the underlying constructor refuses funding.
        let mut copied = Self {
            storage: Box::new(Storage {
                parser: None,
                failure: None,
            }),
            terminal_eos_alias_committed: self.terminal_eos_alias_committed,
            funding: funding.clone(),
        };
        match original.try_copy(funding) {
            Ok(parser) => {
                copied.storage.parser = Some(parser);
                Ok(copied)
            }
            Err(cause) => {
                copied.storage.failure = Some(cause.into());
                Err(OriginalGrammarStateCopyError {
                    cause: CopyCause::Parser,
                    prefix: Some(copied),
                    source: Arc::clone(&original.lexer.source),
                    funding: funding.clone(),
                })
            }
        }
    }
}
