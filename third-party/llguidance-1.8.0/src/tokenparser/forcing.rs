//! Shared canonical forced-token selection; each owner supplies its actual storage.
use crate::{Instant, TokenParser};
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};
use toktrie::TokenId;
/// Result ranges refer to the context's actual retained tokenization and bytes.
#[derive(Clone, Debug)]
pub struct ForcedTokenSelection {
    tokens: Range<usize>,
    prefix: Range<usize>,
}
impl ForcedTokenSelection {
    /// An empty forcing selection for a session already stopped before work.
    pub fn empty() -> Self {
        Self {
            tokens: 0..0,
            prefix: 0..0,
        }
    }

    /// Forced IDs in the completed selected encoding.
    pub fn token_range(&self) -> Range<usize> {
        self.tokens.clone()
    }
    /// Bytes which must prefix the next token when no complete token is forced.
    pub fn prefix_range(&self) -> Range<usize> {
        self.prefix.clone()
    }
}
/// Storage and tokenizer callbacks for the ordinary forced-token algorithm.
/// This trait supplies no admission authority; original composition uses only
/// its retained tokenizer/recipe owner and reserves each reached destination.
pub trait ForcedTokenContext {
    /// Concrete failure retaining the caller's partially constructed source.
    type Error;
    /// Collects last-token raw bytes plus pending forced bytes and returns the
    /// last token's ID/raw byte extent (or None/zero for empty history).
    fn collect(&mut self) -> Result<(Option<TokenId>, usize), Self::Error>;
    /// Canonicality of this exact tokenizer, independent of grammar forcing policy.
    fn canonical(&self) -> bool;
    /// Actual collected raw bytes.
    fn bytes(&self) -> &[u8];
    /// Replaces the selected encoding only after the exact suffix is tokenized.
    fn tokenize(&mut self, offset: usize) -> Result<(), Self::Error>;
    /// Actual selected canonical IDs, empty before any encoding.
    fn tokens(&self) -> &[TokenId];
    /// Stable-token prefix reported by the actual marked-token tokenizer.
    fn fixed_tokens(&self) -> usize;
    /// Runs the same chop worker on the selected ID suffix after `fixed`.
    fn chop(&mut self, fixed: usize) -> Result<(usize, usize), Self::Error>;
    /// A typed source/destination mismatch.
    fn invalid_source(&self) -> Self::Error;
}
/// Fixed local frames for this shared driver, excluding the context's actual
/// buffers, tokenization, parser work and their individual reservation frames.
pub fn forced_token_driver_control_bytes<C: ForcedTokenContext>() -> Option<usize> {
    let parts = [
        size_of::<&mut C>(),
        size_of::<ForcedTokenSelection>(),
        size_of::<Result<ForcedTokenSelection, C::Error>>(),
        size_of::<Result<(), C::Error>>(),
        size_of::<Result<(usize, usize), C::Error>>(),
        size_of::<Result<(Option<TokenId>, usize), C::Error>>(),
        size_of::<Option<TokenId>>(),
        size_of::<(usize, usize, usize, usize, usize, bool)>(),
        size_of::<Range<usize>>() * 2,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
/// Selects canonical forced tokens using the same last-token retry, marked-token
/// stability and suffix chop decisions for ordinary and prepared owners.
pub fn select_forced_tokens<C: ForcedTokenContext>(
    context: &mut C,
) -> Result<ForcedTokenSelection, C::Error> {
    let (existing, existing_bytes) = context.collect()?;
    let total = context.bytes().len();
    if existing_bytes > total {
        return Err(context.invalid_source());
    }
    if total > existing_bytes && context.canonical() {
        context.tokenize(0)?;
        let mut existing_count = usize::from(existing.is_some());
        let mut fixed = context.fixed_tokens();
        if existing.is_some_and(|token| context.tokens().first() != Some(&token)) {
            context.tokenize(existing_bytes)?;
            fixed = context.fixed_tokens();
            existing_count = 0;
        } else {
            fixed = fixed.max(existing_count);
        }
        if fixed > context.tokens().len() {
            return Err(context.invalid_source());
        }
        let (chop_tokens, chop_bytes) = context.chop(fixed)?;
        let end = context
            .tokens()
            .len()
            .checked_sub(chop_tokens)
            .ok_or_else(|| context.invalid_source())?;
        let prefix = total
            .checked_sub(chop_bytes)
            .ok_or_else(|| context.invalid_source())?;
        if end < existing_count
            || (existing_count != 0 && context.tokens().first().copied() != existing)
        {
            return Err(context.invalid_source());
        }
        Ok(ForcedTokenSelection {
            tokens: existing_count..end,
            prefix: prefix..total,
        })
    } else {
        Ok(ForcedTokenSelection {
            tokens: 0..0,
            prefix: existing_bytes..total,
        })
    }
}
/// Collects pending bytes when the exact grammar/tokenizer policy disables
/// canonical token forcing. The next mask still receives that raw prefix.
pub fn select_forced_prefix<C: ForcedTokenContext>(
    context: &mut C,
) -> Result<ForcedTokenSelection, C::Error> {
    let (_, existing) = context.collect()?;
    let total = context.bytes().len();
    if existing > total {
        return Err(context.invalid_source());
    }
    Ok(ForcedTokenSelection {
        tokens: 0..0,
        prefix: existing..total,
    })
}

struct Ordinary<'a> {
    parser: &'a mut TokenParser,
    bytes: Vec<u8>,
    encoded: Option<(Vec<TokenId>, usize)>,
    start: Option<Instant>,
}
impl ForcedTokenContext for Ordinary<'_> {
    type Error = &'static str;
    fn collect(&mut self) -> Result<(Option<TokenId>, usize), Self::Error> {
        let token = self.parser.llm_tokens.last().copied();
        if let Some(token) = token {
            self.bytes
                .extend(self.parser.token_env.tok_trie().raw_token_bytes(&[token]));
        }
        let existing = self.bytes.len();
        self.parser.compute_ff_bytes_to(&mut self.bytes);
        Ok((token, existing))
    }
    fn canonical(&self) -> bool {
        self.parser.token_env.tokenize_is_canonical()
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn tokenize(&mut self, offset: usize) -> Result<(), Self::Error> {
        self.start.get_or_insert_with(Instant::now);
        self.encoded = Some(
            self.parser
                .token_env
                .tokenize_bytes_marker(&self.bytes[offset..]),
        );
        Ok(())
    }
    fn tokens(&self) -> &[TokenId] {
        self.encoded.as_ref().map_or(&[], |e| e.0.as_slice())
    }
    fn fixed_tokens(&self) -> usize {
        self.encoded.as_ref().map_or(0, |e| e.1)
    }
    fn chop(&mut self, fixed: usize) -> Result<(usize, usize), Self::Error> {
        let tokens = &self.encoded.as_ref().expect("encoded tokens").0[fixed..];
        let trie = self.parser.token_env.tok_trie();
        Ok(self
            .parser
            .parser
            .with_recognizer(|recognizer| trie.chop_tokens(recognizer, tokens)))
    }
    fn invalid_source(&self) -> Self::Error {
        "forced-token source differs"
    }
}
pub(super) fn ordinary(parser: &mut TokenParser) -> (Vec<TokenId>, Vec<u8>) {
    let mut context = Ordinary {
        parser,
        bytes: Vec::new(),
        encoded: None,
        start: None,
    };
    let selected = select_forced_tokens(&mut context).expect("ordinary canonical token source");
    if let Some(start) = context.start {
        context
            .parser
            .parser
            .perf_counters()
            .tokenize_ff
            .record(start.elapsed());
    }
    let mut tokens = context.encoded.take().map_or_else(Vec::new, |e| e.0);
    tokens.truncate(selected.tokens.end);
    tokens.drain(..selected.tokens.start);
    let prefix = context.bytes[selected.prefix].to_vec();
    (tokens, prefix)
}
