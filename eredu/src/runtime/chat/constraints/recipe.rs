//! Closed immutable reconstruction data for one constraint configuration.

use super::super::tokenizer_env::recipe::FrozenGrammarTokenizer;
use eredu_core::SharedControllerBytes;
use llguidance::{
    api::TopLevelGrammar,
    earley::{SlicerSource, SlicerSourceDescriptor, SlicerSourceView},
    toktrie::TokRxInfo,
};
use serde_json::Value;
use std::fmt;
use eredu_core::HostPreparationAuthority;
use llguidance::derivre::ParserAllocationFunding;
mod serialization;
use serialization::Cause;

/// Serialization failures retain their original source-construction payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct RecipeBuildError {
    #[source]
    cause: Cause,
    funding: ParserAllocationFunding,
}

// Version 4 also retains exact ordinary slicer recognition records.
// Version 3 also retains the exact prefix-normalized tokenizer source.
// Version 2 retained actual prepared trie metadata; version 1 fixed
// ParserFactory::new_simple, quiet/default general slices, and
// the existing tokenizer prefix-normalization policy. Changing reconstruction
// settings requires a new recipe version, not an implicit reinterpretation.
const MAGIC: &[u8; 8] = b"EREDUCR\0";
const VERSION: u32 = 4;
const HEADER_BYTES: usize = MAGIC.len() + std::mem::size_of::<u32>();
const OFFSET_BYTES: usize = std::mem::size_of::<u64>();
const TOKEN_BYTES: usize = std::mem::size_of::<u32>();

#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn bytes(self, bytes: &[u8]) -> &[u8] {
        &bytes[self.start..self.end]
    }

    fn write(self, bytes: &mut [u8], value: &[u8]) {
        bytes[self.start..self.end].copy_from_slice(value);
    }
}

#[derive(Clone, Copy)]
struct Strings {
    offsets: Span,
    data: Span,
    count: usize,
}

impl Strings {
    fn plan(cursor: &mut usize, strings: &[String]) -> Result<Self, Cause> {
        let count = strings.len();
        let directory = count
            .checked_add(1)
            .and_then(|count| count.checked_mul(OFFSET_BYTES))
            .ok_or_else(overflow)?;
        let payload = strings.iter().try_fold(0usize, |total, value| {
            total.checked_add(value.len()).ok_or_else(overflow)
        })?;
        Ok(Self {
            offsets: take_span(cursor, directory)?,
            data: take_span(cursor, payload)?,
            count,
        })
    }

    fn write(self, bytes: &mut [u8], strings: &[String]) {
        let mut cursor = self.data.start;
        for (index, value) in strings.iter().enumerate() {
            write_offset(bytes, self.offsets.start + index * OFFSET_BYTES, cursor);
            let end = cursor + value.len();
            bytes[cursor..end].copy_from_slice(value.as_bytes());
            cursor = end;
        }
        write_offset(
            bytes,
            self.offsets.start + self.count * OFFSET_BYTES,
            cursor,
        );
        debug_assert_eq!(cursor, self.data.end);
    }

    fn iter(self, bytes: &[u8]) -> impl ExactSizeIterator<Item = &str> {
        (0..self.count).map(move |index| {
            let position = self.offsets.start + index * OFFSET_BYTES;
            let start = read_offset(bytes, position);
            let end = read_offset(bytes, position + OFFSET_BYTES);
            // Only checked construction writes these immutable sections, and
            // their sources are String values. No untrusted decoding API exists.
            std::str::from_utf8(&bytes[start..end]).expect("validated recipe string")
        })
    }
}

#[derive(Clone, Copy)]
struct GrammarTokenizer {
    root: Span,
    object: Span,
    encode_special_tokens: bool,
    canonical: bool,
}

#[derive(Clone, Copy)]
struct SlicerLayout {
    bytes: Span,
    descriptor: SlicerSourceDescriptor,
}

#[derive(Clone, Copy)]
struct Layout {
    tokenizer: Option<Span>,
    tokenizer_object: Option<Span>,
    trie_info: Option<Span>,
    grammar_tokenizer: Option<GrammarTokenizer>,
    slicer: Option<SlicerLayout>,
    grammar: Span,
    max_tokens: Option<usize>,
    tools: Span,
    eos: Span,
    structural_ids: Span,
    structural_spellings: Strings,
    stops: Strings,
    trigger: Option<Span>,
}

fn overflow() -> Cause { Cause::Overflow }

fn take_span(cursor: &mut usize, size: usize) -> Result<Span, Cause> {
    let end = cursor.checked_add(size).ok_or_else(overflow)?;
    u64::try_from(end).map_err(|_| overflow())?;
    if end > isize::MAX as usize {
        return Err(overflow());
    }
    let span = Span {
        start: *cursor,
        end,
    };
    *cursor = end;
    Ok(span)
}

fn token_bytes(count: usize) -> Result<usize, Cause> {
    count.checked_mul(TOKEN_BYTES).ok_or_else(overflow)
}

fn write_offset(bytes: &mut [u8], position: usize, value: usize) {
    // take_span checked the entire allocation against u64 and isize limits.
    bytes[position..position + OFFSET_BYTES].copy_from_slice(&(value as u64).to_le_bytes());
}

fn read_offset(bytes: &[u8], position: usize) -> usize {
    u64::from_le_bytes(bytes[position..position + OFFSET_BYTES].try_into().unwrap()) as usize
}

fn write_tokens(span: Span, bytes: &mut [u8], tokens: &[u32]) {
    for (target, value) in bytes[span.start..span.end]
        .chunks_exact_mut(TOKEN_BYTES)
        .zip(tokens)
    {
        target.copy_from_slice(&value.to_le_bytes());
    }
}

fn tokens(span: Span, bytes: &[u8]) -> impl ExactSizeIterator<Item = u32> + '_ {
    span.bytes(bytes)
        .chunks_exact(TOKEN_BYTES)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
}

/// All variable retained reconstruction data lives in one immutable byte owner.
/// Fixed-size ranges/counts refer only to that owner's checked format.
#[derive(Clone)]
pub(crate) struct ConstraintRecipe {
    bytes: SharedControllerBytes,
    layout: Layout,
    original_trie: Option<eredu_runtime::working_memory::OriginalTokenTrieSource>,
}

impl ConstraintRecipe {
    pub(super) fn bind_original_trie(
        &mut self,
        source: Option<&eredu_runtime::working_memory::OriginalTokenTrieSource>,
    ) {
        self.original_trie = source.cloned();
    }

    pub(super) fn original_trie(&self) -> Option<&eredu_runtime::working_memory::OriginalTokenTrieSource> {
        self.original_trie.as_ref()
    }

    /// Synthetic fixtures use the same serializer with explicit unenforced accounting.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tokenizer_json: Option<&[u8]>,
        grammar: &TopLevelGrammar,
        tools: &[Value],
        eos: &[u32],
        structural_spellings: &[String],
        structural_ids: &[u32],
        stops: &[String],
        trigger: Option<&str>,
    ) -> Result<Self, RecipeBuildError> {
        Self::build(
            tokenizer_json,
            grammar,
            tools,
            eos,
            structural_spellings,
            structural_ids,
            stops,
            trigger,
            None,
            None,
            None,
            &ParserAllocationFunding::unenforced(),
            &HostPreparationAuthority::unmanaged(),
        )
    }

    /// Retains metadata produced by the actual shared tokenizer environment.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_trie_info(
        tokenizer_json: Option<&[u8]>,
        grammar: &TopLevelGrammar,
        tools: &[Value],
        eos: &[u32],
        structural_spellings: &[String],
        structural_ids: &[u32],
        stops: &[String],
        trigger: Option<&str>,
        trie_info: TokRxInfo,
        grammar_tokenizer: Option<&FrozenGrammarTokenizer>,
        slicer_source: Option<&SlicerSource>,
        funding: &ParserAllocationFunding,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, RecipeBuildError> {
        Self::build(
            tokenizer_json,
            grammar,
            tools,
            eos,
            structural_spellings,
            structural_ids,
            stops,
            trigger,
            Some(trie_info),
            grammar_tokenizer,
            slicer_source,
            funding,
            authority,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        tokenizer_json: Option<&[u8]>,
        grammar: &TopLevelGrammar,
        tools: &[Value],
        eos: &[u32],
        structural_spellings: &[String],
        structural_ids: &[u32],
        stops: &[String],
        trigger: Option<&str>,
        trie_info: Option<TokRxInfo>,
        grammar_tokenizer: Option<&FrozenGrammarTokenizer>,
        slicer_source: Option<&SlicerSource>,
        funding: &ParserAllocationFunding,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, RecipeBuildError> {
        let result = (|| -> Result<Self, Cause> {
        let controls = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Layout>(),
            std::mem::size_of::<RecipeBuildError>(),
            std::mem::size_of::<Result<Self, RecipeBuildError>>(),
            std::mem::size_of::<serialization::Count>(),
            SharedControllerBytes::source_shell_bytes().ok_or(Cause::Overflow)?,
        ];
        funding.reserve(controls.into_iter().try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(Cause::Overflow)?)?;
        if structural_spellings.len() != structural_ids.len() {
            return Err(Cause::Structural);
        }
        let mut grammar_json = serialization::Count::default();
        serialization::value(&mut grammar_json, grammar, funding)?;
        let mut tools_json = serialization::Count::default();
        serialization::tools(&mut tools_json, tools, funding)?;
        let mut total = HEADER_BYTES;
        let mut layout = Layout {
            tokenizer: tokenizer_json
                .map(|value| take_span(&mut total, value.len()))
                .transpose()?,
            tokenizer_object: None,
            grammar_tokenizer: grammar_tokenizer
                .map(|source| {
                    let root = take_span(&mut total, source.bytes().len())?;
                    let range = source.object_range();
                    Ok::<_, Cause>(GrammarTokenizer {
                        root,
                        object: Span {
                            start: root.start.checked_add(range.start).ok_or_else(overflow)?,
                            end: root.start.checked_add(range.end).ok_or_else(overflow)?,
                        },
                        encode_special_tokens: source.encode_special_tokens(),
                        canonical: source.canonical(),
                    })
                })
                .transpose()?,
            slicer: slicer_source
                .map(|source| {
                    Ok::<_, Cause>(SlicerLayout {
                        bytes: take_span(&mut total, source.as_bytes().len())?,
                        descriptor: source.descriptor(),
                    })
                })
                .transpose()?,
            trie_info: trie_info
                .map(|_| take_span(&mut total, 10 * TOKEN_BYTES))
                .transpose()?,
            grammar: take_span(&mut total, grammar_json.bytes())?,
            max_tokens: grammar.max_tokens,
            tools: take_span(&mut total, tools_json.bytes())?,
            eos: take_span(&mut total, token_bytes(eos.len())?)?,
            structural_ids: take_span(&mut total, token_bytes(structural_ids.len())?)?,
            structural_spellings: Strings::plan(&mut total, structural_spellings)?,
            stops: Strings::plan(&mut total, stops)?,
            trigger: trigger
                .map(|value| take_span(&mut total, value.len()))
                .transpose()?,
        };
        if let (Some(root), Some(json)) = (layout.tokenizer, tokenizer_json) {
            if let Some(span) = super::super::tokenizer_env::recipe::tokenizer_span(json) {
                layout.tokenizer_object = Some(Span {
                    start: root.start.checked_add(span.start).ok_or_else(overflow)?,
                    end: root.start.checked_add(span.end).ok_or_else(overflow)?,
                });
            }
        }
        let mut bytes = Vec::new();
        funding.try_grow_vec(&mut bytes, total)?;
        bytes.resize(total, 0);
        bytes[..MAGIC.len()].copy_from_slice(MAGIC);
        bytes[MAGIC.len()..HEADER_BYTES].copy_from_slice(&VERSION.to_le_bytes());
        if let (Some(span), Some(json)) = (layout.tokenizer, tokenizer_json) {
            span.write(&mut bytes, json);
        }
        if let (Some(layout), Some(source)) = (layout.grammar_tokenizer, grammar_tokenizer) {
            layout.root.write(&mut bytes, source.bytes());
        }
        if let (Some(layout), Some(source)) = (layout.slicer, slicer_source) {
            layout.bytes.write(&mut bytes, source.as_bytes());
        }
        if let (Some(span), Some(info)) = (layout.trie_info, trie_info) {
            let option = |id: Option<u32>| [u32::from(id.is_some()), id.unwrap_or(0)];
            let bos = option(info.tok_bos);
            let pad = option(info.tok_pad);
            let unk = option(info.tok_unk);
            let eot = option(info.tok_end_of_turn);
            write_tokens(
                span,
                &mut bytes,
                &[
                    info.vocab_size,
                    info.tok_eos,
                    bos[0],
                    bos[1],
                    pad[0],
                    pad[1],
                    unk[0],
                    unk[1],
                    eot[0],
                    eot[1],
                ],
            );
        }
        let mut grammar_destination = &mut bytes[layout.grammar.start..layout.grammar.end];
        serialization::value(&mut grammar_destination, grammar, funding)?;
        if !grammar_destination.is_empty() { return Err(Cause::Destination); }
        let mut tools_destination = &mut bytes[layout.tools.start..layout.tools.end];
        serialization::tools(&mut tools_destination, tools, funding)?;
        if !tools_destination.is_empty() { return Err(Cause::Destination); }
        write_tokens(layout.eos, &mut bytes, eos);
        write_tokens(layout.structural_ids, &mut bytes, structural_ids);
        layout
            .structural_spellings
            .write(&mut bytes, structural_spellings);
        layout.stops.write(&mut bytes, stops);
        if let (Some(span), Some(trigger)) = (layout.trigger, trigger) {
            span.write(&mut bytes, trigger.as_bytes());
        }
        Ok(Self {
            bytes: SharedControllerBytes::new(bytes, authority.clone()),
            layout,
            original_trie: None,
        })
        })();
        result.map_err(|cause| RecipeBuildError { cause, funding: funding.clone() })
    }

    pub(super) fn fingerprint(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"[");
        digest.update(self.layout.grammar.bytes(self.bytes.as_ref()));
        digest.update(b",");
        digest.update(self.layout.tools.bytes(self.bytes.as_ref()));
        digest.update(b"]");
        digest.finalize().into()
    }

    pub(crate) fn tokenizer_json(&self) -> Option<&[u8]> {
        self.layout
            .tokenizer
            .map(|span| span.bytes(self.bytes.as_ref()))
    }

    /// Exact raw object inside the validated frozen envelope; its sealed range
    /// survives independent registered copies of these same immutable bytes.
    pub(crate) fn tokenizer_object_json(&self) -> Option<&[u8]> {
        self.layout
            .tokenizer_object
            .map(|span| span.bytes(self.bytes.as_ref()))
    }

    /// Named immutable source-view frames, charged before original source use.
    pub(crate) fn grammar_source_control_bytes(&self) -> Option<usize> {
        let mut words = self.eos_token_ids();
        let parts = [
            self.layout
                .slicer
                .map(|layout| layout.descriptor.control_bytes())
                .unwrap_or(Some(0))?,
            std::mem::size_of::<Option<SlicerSourceView<'_>>>(),
            std::mem::size_of::<Result<SlicerSourceView<'_>, llguidance::earley::SlicerSourceError>>(
            ),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Layout>(),
            std::mem::size_of::<Span>(),
            std::mem::size_of::<GrammarTokenizer>(),
            std::mem::size_of::<Option<GrammarTokenizer>>(),
            std::mem::size_of::<TokRxInfo>(),
            std::mem::size_of::<Option<TokRxInfo>>(),
            std::mem::size_of::<Option<&[u8]>>(),
            std::mem::size_of::<Option<bool>>(),
            std::mem::size_of::<Option<u32>>(),
            std::mem::size_of::<[u8; 4]>(),
            std::mem::size_of::<(u32, u32)>(),
            std::mem::size_of_val(&words),
            std::mem::size_of_val(&(&mut words)),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Complete historical normalized HF envelope for the ordinary grammar path.
    pub(crate) fn grammar_tokenizer_json(&self) -> Option<&[u8]> {
        self.layout
            .grammar_tokenizer
            .map(|source| source.root.bytes(self.bytes.as_ref()))
    }
    /// Source object after the shared prefix worker, borrowed without JSON parsing.
    pub(crate) fn grammar_tokenizer_object_json(&self) -> Option<&[u8]> {
        self.layout
            .grammar_tokenizer
            .map(|source| source.object.bytes(self.bytes.as_ref()))
    }
    /// Actual non-serde HF special-splitting policy retained by `freeze`.
    pub(crate) fn grammar_tokenizer_is_canonical(&self) -> Option<bool> {
        self.original_trie.as_ref().map(|source| source.tokenization_is_canonical())
            .or_else(|| self.layout.grammar_tokenizer.map(|source| source.canonical))
    }

    pub(crate) fn grammar_encode_special_tokens(&self) -> Option<bool> {
        self.layout
            .grammar_tokenizer
            .map(|source| source.encode_special_tokens)
    }

    /// Copies only fixed metadata from the actual immutable recipe bytes.
    /// It does not infer token roles from spellings or reconstruct a vocabulary.
    pub(crate) fn trie_info(&self) -> Option<TokRxInfo> {
        let span = self.layout.trie_info?;
        let mut values = tokens(span, self.bytes.as_ref());
        let vocab_size = values.next().expect("written trie metadata");
        let tok_eos = values.next().expect("written trie metadata");
        let mut option = || {
            let present = values.next().expect("written optional tag");
            let id = values.next().expect("written optional ID");
            (present != 0).then_some(id)
        };
        Some(TokRxInfo {
            vocab_size,
            tok_eos,
            tok_bos: option(),
            tok_pad: option(),
            tok_unk: option(),
            tok_end_of_turn: option(),
        })
    }

    /// Exact scalar from the same TopLevelGrammar serialized into this recipe.
    /// It is retained by checked construction, so startup need not parse JSON.
    pub(crate) fn grammar_max_tokens(&self) -> Option<usize> {
        self.layout.max_tokens
    }

    pub(crate) fn grammar(&self) -> Result<TopLevelGrammar, String> {
        serde_json::from_slice(self.layout.grammar.bytes(self.bytes.as_ref()))
            .map_err(|error| error.to_string())
    }

    pub(crate) fn tools(&self) -> Result<Vec<Value>, String> {
        serde_json::from_slice(self.layout.tools.bytes(self.bytes.as_ref()))
            .map_err(|error| error.to_string())
    }

    pub(crate) fn eos_token_ids(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        tokens(self.layout.eos, self.bytes.as_ref())
    }

    pub(crate) fn structural_tokens(&self) -> impl ExactSizeIterator<Item = (u32, &str)> + '_ {
        tokens(self.layout.structural_ids, self.bytes.as_ref())
            .zip(self.layout.structural_spellings.iter(self.bytes.as_ref()))
    }

    pub(crate) fn stop_sequences(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.layout.stops.iter(self.bytes.as_ref())
    }

    pub(crate) fn trigger(&self) -> Option<&str> {
        self.layout.trigger.map(|span| {
            std::str::from_utf8(span.bytes(self.bytes.as_ref())).expect("validated recipe trigger")
        })
    }

    /// Exact closed layout into this recipe's same registered byte owner.
    pub(crate) fn slicer_source(&self) -> Option<SlicerSourceView<'_>> {
        let layout = self.layout.slicer?;
        layout
            .descriptor
            .checked_view(layout.bytes.bytes(self.bytes.as_ref()))
            .ok()
    }

    pub(crate) fn source(&self) -> &SharedControllerBytes {
        &self.bytes
    }

    /// Matches the semantic plan's tools, resolved structural tokens, and stops.
    /// Grammar, tokenizer, EOS and activation trigger belong to other plan
    /// comparisons. Borrowed packed comparisons allocate no decoded payload.
    pub(crate) fn semantic_eq(&self, other: &Self) -> bool {
        self.layout.tools.bytes(self.bytes.as_ref())
            == other.layout.tools.bytes(other.bytes.as_ref())
            && self.structural_tokens().eq(other.structural_tokens())
            && self.stop_sequences().eq(other.stop_sequences())
    }
}

impl fmt::Debug for ConstraintRecipe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConstraintRecipe")
            .field("version", &VERSION)
            .field("bytes", &self.bytes.as_ref().len())
            .field("has_tokenizer", &self.layout.tokenizer.is_some())
            .field("structural_tokens", &self.layout.structural_spellings.count)
            .field("stops", &self.layout.stops.count)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
