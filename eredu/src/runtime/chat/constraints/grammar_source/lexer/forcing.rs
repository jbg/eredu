//! Canonical original-tokenizer output feeding the shared ordinary forcing driver.
use super::{OriginalGrammarTokenParser, OriginalGrammarTokenParserError};
use crate::runtime::chat::constraints::grammar_source::{
    GrammarTokenizationMode, OriginalGrammarTokenIds, OriginalGrammarTokenizationError,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use llguidance::{ForcedTokenContext, ForcedTokenSelection};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original forced-token source differs")]
    Source,
    #[error("original forced-token destination extent overflow")]
    Overflow,
    #[error("original forced-token destination differs")]
    Capacity,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Parser(#[from] OriginalGrammarTokenParserError),
    #[error(transparent)]
    Tokenization(#[from] OriginalGrammarTokenizationError),
}
#[derive(Debug)]
struct Context {
    bytes: Vec<u8>,
    encoded: Option<OriginalGrammarTokenIds>,
    parser: Option<OriginalGrammarTokenParser>,
    canonical: Option<bool>,
    funding: HostMetadataFunding,
}
/// Forced IDs and prefix bytes retain the actual tokenization, mutable parser,
/// immutable source and funding until their consumer finishes.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarForcedTokens {
    selected: ForcedTokenSelection,
    context: Context,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarForcingError {
    #[source]
    cause: Cause,
    context: Context,
}
fn reserve(values: &mut Vec<u8>, total: usize, funding: &HostMetadataFunding) -> Result<(), Cause> {
    if total <= values.capacity() {
        return Ok(());
    }
    funding.reserve_metadata(
        Layout::array::<u8>(total)
            .map_err(|_| Cause::Overflow)?
            .size(),
    )?;
    values.try_reserve_exact(total - values.len())?;
    if values.capacity() != total {
        return Err(Cause::Capacity);
    }
    Ok(())
}
impl Context {
    fn collect(&mut self) -> Result<(Option<u32>, usize), Cause> {
        let source = self.parser.as_ref().ok_or(Cause::Source)?;
        let token = source.parser.tokens().last().copied();
        if let Some(token) = token {
            let trie = source.vocabulary().trie_source().trie();
            let total = trie.raw_token_bytes_len(&[token]).ok_or(Cause::Overflow)?;
            reserve(&mut self.bytes, total, &self.funding)?;
            self.bytes.extend(trie.raw_token_bytes(&[token]));
        }
        let existing = self.bytes.len();
        let can_force = self.canonical.ok_or(Cause::Source)?
            && !source
                .vocabulary()
                .compiled_declaration()
                .lexer_spec()
                .no_forcing;
        if can_force {
            let parser = self.parser.take().ok_or(Cause::Source)?;
            self.parser = Some(parser.force_bytes()?);
        }
        let pending = self
            .parser
            .as_ref()
            .ok_or(Cause::Source)?
            .parser
            .chart()
            .pending_token_bytes();
        let total = self
            .bytes
            .len()
            .checked_add(pending.len())
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.bytes, total, &self.funding)?;
        self.bytes.extend_from_slice(pending);
        Ok((token, existing))
    }
    fn canonical(&self) -> bool {
        self.canonical.expect("qualified tokenizer")
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn tokenize(&mut self, offset: usize) -> Result<(), Cause> {
        let source = self.parser.as_ref().ok_or(Cause::Source)?.vocabulary();
        let bytes = self.bytes.get(offset..).ok_or(Cause::Source)?;
        self.encoded = Some(source.tokenize_bytes_with_funding(
            bytes,
            GrammarTokenizationMode::Marker,
            &self.funding,
        )?);
        Ok(())
    }
    fn tokens(&self) -> &[u32] {
        self.encoded.as_ref().map_or(&[], |source| source.ids())
    }
    fn fixed_tokens(&self) -> usize {
        self.encoded
            .as_ref()
            .map_or(0, |source| source.fixed_tokens())
    }
    fn chop(&mut self, fixed: usize) -> Result<(usize, usize), Cause> {
        let tokens = self
            .encoded
            .as_ref()
            .ok_or(Cause::Source)?
            .ids()
            .get(fixed..)
            .ok_or(Cause::Source)?;
        let source = self.parser.take().ok_or(Cause::Source)?;
        let (source, count, bytes) = source.chop_tokens(tokens)?;
        self.parser = Some(source);
        Ok((count, bytes))
    }
}
// The shared selector returns only this fixed marker. The original error is
// moved once into the caller's paid slot, beside its still-live Context.
#[derive(Clone, Copy, Debug)]
enum DriverFailure {
    Source,
    Stored,
}
#[derive(Clone, Copy, Debug)]
enum SelectionMode {
    Stopped,
    Prefix,
    Tokens,
}
struct Driver<'a> {
    context: &'a mut Context,
    failure: &'a mut Option<Cause>,
}
impl Driver<'_> {
    fn record<T>(&mut self, result: Result<T, Cause>) -> Result<T, DriverFailure> {
        result.map_err(|cause| {
            *self.failure = Some(cause);
            DriverFailure::Stored
        })
    }
}
impl ForcedTokenContext for Driver<'_> {
    type Error = DriverFailure;
    fn collect(&mut self) -> Result<(Option<u32>, usize), DriverFailure> {
        let result = self.context.collect();
        self.record(result)
    }
    fn canonical(&self) -> bool {
        self.context.canonical()
    }
    fn bytes(&self) -> &[u8] {
        self.context.bytes()
    }
    fn tokenize(&mut self, offset: usize) -> Result<(), DriverFailure> {
        let result = self.context.tokenize(offset);
        self.record(result)
    }
    fn tokens(&self) -> &[u32] {
        self.context.tokens()
    }
    fn fixed_tokens(&self) -> usize {
        self.context.fixed_tokens()
    }
    fn chop(&mut self, fixed: usize) -> Result<(usize, usize), DriverFailure> {
        let result = self.context.chop(fixed);
        self.record(result)
    }
    fn invalid_source(&self) -> DriverFailure {
        DriverFailure::Source
    }
}
impl OriginalGrammarForcedTokens {
    fn finish_mask(mut self) -> Result<OriginalGrammarTokenParser, OriginalGrammarForcingError> {
        let result = (|| -> Result<OriginalGrammarTokenParser, Cause> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarForcingError>(),
                size_of::<Cause>(),
                size_of::<OriginalGrammarTokenParser>(),
                size_of::<Option<u32>>(),
                size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserError>>(),
                size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarForcingError>>(),
                size_of::<Result<OriginalGrammarTokenParser, Cause>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<std::ops::Range<usize>>(),
                size_of::<(&[u8], Option<u32>)>(),
            ];
            self.context.funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let first = self.tokens().first().copied();
            let prefix = &self.context.bytes[self.selected.prefix_range()];
            let parser = self.context.parser.take().ok_or(Cause::Source)?;
            Ok(parser.mask_from_prefix(first, prefix)?)
        })();
        match result {
            Ok(parser) => Ok(parser),
            Err(cause) => Err(OriginalGrammarForcingError {
                cause,
                context: self.context,
            }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn tokens(&self) -> &[u32] {
        let tokens = self
            .context
            .encoded
            .as_ref()
            .map_or(&[][..], |source| source.ids());
        &tokens[self.selected.token_range()]
    }
    pub(in crate::runtime::chat::constraints) fn prefix(&self) -> &[u8] {
        &self.context.bytes[self.selected.prefix_range()]
    }
    pub(in crate::runtime::chat::constraints) fn into_parser(
        mut self,
    ) -> OriginalGrammarTokenParser {
        self.context
            .parser
            .take()
            .expect("completed original forcing")
    }
}
impl OriginalGrammarTokenParser {
    fn force_bytes(self) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser
                .force_bytes(lexer, trie, limits, &|bytes| {
                    funding.reserve_metadata(bytes)
                })
                .map(|parser| (parser, ()))
        })
        .map(|(owner, ())| owner)
    }
    fn chop_tokens(
        self,
        tokens: &[u32],
    ) -> Result<(Self, usize, usize), OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser
                .chop_tokens(lexer, trie, limits, tokens, &|bytes| {
                    funding.reserve_metadata(bytes)
                })
                .map(|(parser, count, bytes)| (parser, (count, bytes)))
        })
        .map(|(owner, (count, bytes))| (owner, count, bytes))
    }
    pub(in crate::runtime::chat::constraints) fn force_tokens(
        self,
    ) -> Result<OriginalGrammarForcedTokens, OriginalGrammarForcingError> {
        self.forcing(false)
    }
    pub(in crate::runtime::chat::constraints) fn compute_mask(
        self,
    ) -> Result<Self, OriginalGrammarForcingError> {
        self.forcing(true)?.finish_mask()
    }
    fn mask_from_prefix(
        self,
        first: Option<u32>,
        prefix: &[u8],
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(|parser, lexer, trie, limits, funding| {
            parser
                .compute_mask(lexer, trie, limits, first, prefix, &|bytes| {
                    funding.reserve_metadata(bytes)
                })
                .map(|parser| (parser, ()))
        })
        .map(|(owner, ())| owner)
    }
    fn forcing(
        self,
        for_mask: bool,
    ) -> Result<OriginalGrammarForcedTokens, OriginalGrammarForcingError> {
        let funding = self.lexer.funding.clone();
        let mut context = Context {
            bytes: Vec::new(),
            encoded: None,
            parser: Some(self),
            canonical: None,
            funding,
        };
        let mut failure = None;
        let result = (|| -> Result<ForcedTokenSelection, DriverFailure> {
            let prepared = prepare_selection(&mut context, for_mask);
            let mut driver = Driver {
                context: &mut context,
                failure: &mut failure,
            };
            match driver.record(prepared)? {
                SelectionMode::Stopped => Ok(ForcedTokenSelection::empty()),
                SelectionMode::Prefix => llguidance::select_forced_prefix(&mut driver),
                SelectionMode::Tokens => llguidance::select_forced_tokens(&mut driver),
            }
        })();
        match result {
            Ok(selected) => Ok(OriginalGrammarForcedTokens { selected, context }),
            Err(marker) => Err(OriginalGrammarForcingError {
                cause: match marker {
                    DriverFailure::Source => Cause::Source,
                    DriverFailure::Stored => failure.expect("recorded original forcing failure"),
                },
                context,
            }),
        }
    }
}

// This completes before entering the selector or any parser callback, so its
// owning preflight Result does not remain on the shared traversal's stack.
#[inline(never)]
fn prepare_selection(context: &mut Context, for_mask: bool) -> Result<SelectionMode, Cause> {
    let parts = [
        llguidance::forced_token_driver_control_bytes::<Driver<'_>>().ok_or(Cause::Overflow)?,
        size_of::<OriginalGrammarTokenParser>(),
        size_of::<Context>(),
        size_of::<Cause>(),
        size_of::<OriginalGrammarForcedTokens>(),
        size_of::<OriginalGrammarForcingError>(),
        size_of::<ForcedTokenSelection>(),
        size_of::<Option<OriginalGrammarTokenIds>>(),
        size_of::<Result<OriginalGrammarForcedTokens, OriginalGrammarForcingError>>(),
        size_of::<Result<ForcedTokenSelection, DriverFailure>>(),
        size_of::<Option<Cause>>(),
        size_of::<Driver<'_>>(),
        size_of::<DriverFailure>(),
        size_of::<SelectionMode>(),
        size_of::<Result<SelectionMode, Cause>>(),
        size_of::<Result<SelectionMode, DriverFailure>>(),
        size_of::<Result<(), DriverFailure>>(),
        size_of::<Result<(Option<u32>, usize), Cause>>(),
        size_of::<Result<(usize, usize), Cause>>(),
        size_of::<(&mut Context, &mut Option<Cause>, bool)>(),
        size_of::<Result<(), Cause>>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<Layout, std::alloc::LayoutError>>(),
        size_of::<Layout>(),
        size_of::<llguidance::toktrie::RawTokenBytes<'_>>(),
        size_of::<Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>>(),
        size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserError>>(),
        size_of::<
            Result<(OriginalGrammarTokenParser, usize, usize), OriginalGrammarTokenParserError>,
        >(),
        size_of::<(&mut Vec<u8>, usize, &HostMetadataFunding)>(),
        size_of::<(usize, usize, Option<u32>, bool, bool)>(),
        size_of::<[u32; 1]>(),
    ];
    context.funding.reserve_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Cause::Overflow)?,
    )?;
    context.canonical = Some(
        context
            .parser
            .as_ref()
            .ok_or(Cause::Source)?
            .vocabulary()
            .recipe
            .grammar_tokenizer_is_canonical()
            .ok_or(Cause::Source)?,
    );
    if for_mask {
        let parser = context.parser.as_ref().ok_or(Cause::Source)?;
        if parser.parser.stop_reason() != llguidance::api::StopReason::NotStopped {
            return Ok(SelectionMode::Stopped);
        }
        if !context.canonical.ok_or(Cause::Source)?
            || parser
                .vocabulary()
                .compiled_declaration()
                .lexer_spec()
                .no_forcing
        {
            return Ok(SelectionMode::Prefix);
        }
    }
    Ok(SelectionMode::Tokens)
}
