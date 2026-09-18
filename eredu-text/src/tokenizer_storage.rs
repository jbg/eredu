//! Fresh HF and compiled-decode aggregate, without an account or raw HF escape.
mod encode;
mod prefix;
pub use encode::{
    EncodeIdsError, EncodeIdsFailure, EncodeIdsPlan, EncodedTokenIds, NormalizationBuffer,
};
pub use prefix::{InputPrefixFailure, InputPrefixPlan};

/// Complete retained-configuration serialization through the canonical HF workers.
pub use tokenizers::source_serialization::{
    Error as TokenizerSerializationError, Failure as TokenizerSerializationFailure,
    serialize as serialize_configuration,
};

use crate::decoder_storage::{
    DecodeCompileFailure, DecodeCompilePlan, DecodeCompileRequirements, DecodeSourceError,
    PreparedDecodeSource, compiler,
};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};
use tokenizers::{TokenizerCompileError, TokenizerCompileFailure, TokenizerCompilePlan};

/// Fixed root/profile error returned before any destination reserve.
#[derive(Debug)]
pub enum TokenizerSourceError {
    /// Actual root or component planning diagnostic.
    Root(TokenizerCompileError),
    /// Fixed decoder envelope or layout validation failure.
    Decoder(DecodeSourceError),
}
impl fmt::Display for TokenizerSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root(e) => fmt::Display::fmt(e, f),
            Self::Decoder(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for TokenizerSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Root(e) => Some(e),
            Self::Decoder(e) => Some(e),
        }
    }
}
impl TokenizerSourceError {
    fn overflow() -> Self {
        Self::Decoder(DecodeSourceError::Overflow)
    }
}
/// Checked fresh aggregate compiler buffers and concrete controls.
#[derive(Debug, Clone, Copy)]
pub struct TokenizerRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl TokenizerRequirements {
    /// Simultaneous HF and decoder compiler destination bytes.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named construction, partial, error and return representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked total; no budget or existing-owner adoption is implied.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// One root source; no constructed component or caller byte capacity is accepted.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::TokenizerPlan;
/// fn repeat(plan:TokenizerPlan<'_>){let _=plan.compile();let _=plan.compile();}
/// ```
#[derive(Debug)]
pub struct TokenizerPlan<'a> {
    root: TokenizerCompilePlan<'a>,
    decoder: DecodeCompileRequirements,
    requirements: TokenizerRequirements,
    generation_domain: Option<usize>,
    #[cfg(feature = "tokenizer-compiler-test-support")]
    decode_failure: Option<usize>,
}
impl<'a> TokenizerPlan<'a> {
    /// Plans one immutable root borrow before any destination allocation.
    pub fn prepare_json(input: &'a [u8]) -> Result<Self, TokenizerSourceError> {
        let root = TokenizerCompilePlan::prepare_json(input).map_err(TokenizerSourceError::Root)?;
        let facts = root.requirements();
        // The admitted ByteLevel/fallback forms preserve or shrink each stored
        // token piece. Fallback byte-run expansion belongs to the existing decode
        // destination bound. Canonical added IDs occur at most twice in this visit.
        let decoder = compiler::requirements(facts.decoder_id_slots(), facts.decoder_piece_bytes())
            .map_err(TokenizerSourceError::Decoder)?;
        let controls = [
            facts.control_bytes(),
            tokenizers::Tokenizer::configuration_comparison_control_bytes()
                .ok_or_else(TokenizerSourceError::overflow)?,
            decoder.control_bytes(),
            size_of::<Root>(),
            size_of::<Arc<Root>>(),
            size_of::<Self>(),
            size_of::<(Self, bool)>(),
            size_of::<Result<Self, TokenizerSourceError>>(),
            size_of::<TokenizerRequirements>(),
            size_of::<PreparedTokenizer>(),
            size_of::<TokenizerCompileError>(),
            size_of::<TokenizerSourceError>(),
            size_of::<TokenizerConstructionFailure>(),
            size_of::<Cause>(),
            size_of::<Result<PreparedTokenizer, TokenizerConstructionFailure>>(),
            size_of::<Option<tokenizers::Tokenizer>>(),
            size_of::<Result<DecodeCompilePlan<'_>, DecodeSourceError>>(),
            size_of::<Result<tokenizers::Tokenizer, TokenizerCompileFailure>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(TokenizerSourceError::overflow)?;
        let buffers = facts
            .buffer_bytes()
            .checked_add(decoder.buffer_bytes())
            .and_then(|n| n.checked_add(root_shell_bytes()?))
            .ok_or_else(TokenizerSourceError::overflow)?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(TokenizerSourceError::overflow)?;
        Ok(Self {
            root,
            decoder,
            generation_domain: None,
            requirements: TokenizerRequirements {
                buffers,
                controls,
                total,
            },
            #[cfg(feature = "tokenizer-compiler-test-support")]
            decode_failure: None,
        })
    }
    /// Selects the source's actual non-serde HF special-token splitting flag
    /// before original construction. No completed source can be modified.
    pub fn with_encode_special_tokens(mut self, enabled: bool) -> Self {
        self.root = self.root.with_encode_special_tokens(enabled);
        self
    }
    /// Returns actual sealed construction requirements.
    pub fn requirements(&self) -> TokenizerRequirements {
        self.requirements
    }
    /// Selects a canonical generation domain before fresh source admission.
    /// Its dense allocation belongs to the runtime's same original C owner.
    /// Existing decoder-only plans do not request this allocation, including
    /// sparse high-ID sources. An already constructed source cannot be upgraded.
    pub fn with_generation_domain(mut self) -> Result<Self, TokenizerSourceError> {
        let extent = usize::try_from(self.root.requirements().domain_extent())
            .map_err(|_| TokenizerSourceError::overflow())?;
        std::alloc::Layout::array::<bool>(extent).map_err(|_| TokenizerSourceError::overflow())?;
        self.generation_domain = Some(extent);
        Ok(self)
    }
    /// Checked source-derived dense capacity, present only for generation mode.
    pub fn generation_domain_extent(&self) -> Option<usize> {
        self.generation_domain
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only capacity overflow on the actual selected reserve.
    pub fn fail_model_reservation(mut self, stage: usize) -> Self {
        self.root = self.root.fail_model_reservation(stage);
        self
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only capacity overflow on the actual selected reserve.
    pub fn fail_added_reservation(mut self, stage: usize) -> Self {
        self.root = self.root.fail_added_reservation(stage);
        self
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only overflow on one actual packed-template reserve target.
    pub fn fail_template_reservation(mut self, stage: usize) -> Self {
        self.root = self.root.fail_template_reservation(stage);
        self
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only failure on the actual selected regex constructor target.
    pub fn fail_regex_construction(mut self, target: RegexConstructionFailure) -> Self {
        self.root = self.root.fail_regex_construction(target);
        self
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only failure in an actual ordered source regex constructor.
    pub fn fail_regex_construction_at(
        mut self,
        ordinal: usize,
        target: RegexConstructionFailure,
    ) -> Self {
        self.root = self.root.fail_regex_construction_at(ordinal, target);
        self
    }
    /// Constructs fresh HF and its decoder program, preserving all partial failures.
    pub fn compile(self) -> Result<PreparedTokenizer, TokenizerConstructionFailure> {
        let tokenizer = self
            .root
            .compile()
            .map_err(|error| TokenizerConstructionFailure {
                cause: Cause::Root(error),
                tokenizer: None,
            })?;
        let result = (|| {
            let plan = DecodeCompilePlan::prepare_hf(&tokenizer).map_err(Cause::Validation)?;
            let facts = plan.requirements();
            if facts.id_slots() > self.decoder.id_slots()
                || facts.piece_bytes() > self.decoder.piece_bytes()
            {
                return Err(Cause::Validation(DecodeSourceError::Overflow));
            }
            #[cfg(feature = "tokenizer-compiler-test-support")]
            let plan = if let Some(stage) = self.decode_failure {
                plan.fail_reservation(stage)
            } else {
                plan
            };
            plan.compile().map_err(Cause::Decode)
        })();
        match result {
            Ok(decoder) => Ok(PreparedTokenizer {
                root: Some(Arc::new(Root {
                    tokenizer,
                    decoder,
                    envelope: self.decoder,
                })),
                projection: None,
            }),
            Err(cause) => Err(TokenizerConstructionFailure {
                cause,
                tokenizer: Some(tokenizer),
            }),
        }
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only capacity overflow on the actual selected reserve.
    pub fn fail_pipeline_reservation(mut self, stage: usize) -> Self {
        self.root = self.root.fail_reservation(stage);
        self
    }
    /// Development-only refusal at the normalizer's actual source destination.
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    pub fn fail_normalizer_reservation(mut self, stage: usize) -> Self {
        self.root = self.root.fail_normalizer_reservation(stage);
        self
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only capacity overflow on the actual selected reserve.
    pub fn fail_decode_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 3);
        self.decode_failure = Some(stage);
        self
    }
}
/// Opaque immutable aggregate; its operations are readonly projections.
/// No raw HF, clone, serde, configuration or unpriced encoding API is exposed.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::PreparedTokenizer;
/// fn raw(value:PreparedTokenizer){let _=value.tokenizer;}
/// ```
/// ```compile_fail
/// use eredu_text::tokenizer_storage::PreparedTokenizer;
/// fn copy(value:PreparedTokenizer){let _=value.clone();}
/// ```
pub struct PreparedTokenizer {
    projection: Option<Box<prefix::Projection>>,
    root: Option<Arc<Root>>,
}
#[derive(Debug)]
struct Root {
    tokenizer: tokenizers::Tokenizer,
    decoder: PreparedDecodeSource,
    envelope: DecodeCompileRequirements,
}
fn root_shell_bytes() -> Option<usize> {
    Some(
        Layout::array::<AtomicUsize>(2)
            .ok()?
            .extend(Layout::new::<Root>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}
impl Drop for PreparedTokenizer {
    fn drop(&mut self) {
        drop(self.projection.take());
        if let Some(root) = self.root.take() {
            drop(Arc::into_inner(root));
        }
    }
}
impl PreparedTokenizer {
    fn root(&self) -> &Root {
        self.root.as_deref().expect("live tokenizer root")
    }
    fn input_view(&self) -> tokenizers::tokenizer::TokenizerInput<'_> {
        let root = self.root();
        if let Some(projection) = &self.projection {
            root.tokenizer.input_prefix_view(
                projection
                    .added
                    .as_ref()
                    .unwrap_or_else(|| root.tokenizer.get_added_vocabulary()),
            )
        } else {
            (&root.tokenizer).into()
        }
    }
    /// Plans the exact lexical token bytes used by semantic token constraints.
    /// This borrows the source, creates no trie, and grants no allocation credit.
    pub fn token_byte_vocabulary(
        &self,
    ) -> Result<crate::token_bytes::PackedTokenBytePlan<'_>, crate::token_bytes::TokenByteError>
    {
        crate::token_bytes::PackedTokenBytePlan::prepare(
            self,
            crate::token_bytes::TokenByteEncoding::inspect(self.root().tokenizer.get_decoder())?,
        )
    }
    /// Plans actual trie-input bytes, preserving special-token markers and
    /// lexical fallback bytes. No environment, trie or source account is created.
    pub fn token_trie_vocabulary(
        &self,
    ) -> Result<crate::token_bytes::PackedTokenBytePlan<'_>, crate::token_bytes::TokenByteError>
    {
        crate::token_bytes::PackedTokenBytePlan::prepare_trie(
            self,
            crate::token_bytes::TokenByteEncoding::inspect(self.root().tokenizer.get_decoder())?,
        )
    }
    /// Authenticates the actual compiled configuration against an immutable loaded
    /// tokenizer by borrow. This creates no tokenizer, map, string or cache.
    pub fn matches_configuration(&self, selected: &crate::tokenizer::Tokenizer) -> bool {
        self.input_view().matches_compiled_configuration(selected)
    }

    /// Checks a closed projection's exact retained root, or full configuration
    /// equality for independently compiled sources where removal is the identity.
    /// No source identity or account authority is created by this comparison.
    pub fn is_input_prefix_derivative_of(&self, original: &Self) -> bool {
        if self.projection.is_some()
            && Arc::ptr_eq(
                self.root.as_ref().expect("live source"),
                original.root.as_ref().expect("live original"),
            )
        {
            return true;
        }
        self.input_prefix_removal_is_identity()
            && original.input_prefix_removal_is_identity()
            && self
                .input_view()
                .matches_compiled_configuration(&original.root().tokenizer)
    }

    /// Checks the concrete component profile on which input-prefix removal is
    /// the identity. New component support must extend this explicit check.
    pub fn input_prefix_removal_is_identity(&self) -> bool {
        if self.projection.is_some() {
            return true;
        }
        use tokenizers::{NormalizerWrapper as N, PreTokenizerWrapper as P};
        fn leaf(value: &P) -> bool {
            matches!(
                value,
                P::ByteLevel(_) | P::Digits(_) | P::Split(_) | P::Whitespace(_)
            ) || matches!(value, P::Metaspace(meta) if meta.get_prepend_scheme() == tokenizers::pre_tokenizers::metaspace::PrependScheme::Never)
        }
        matches!(
            self.root().tokenizer.get_normalizer(),
            None | Some(N::NFC(_)) | Some(N::Replace(_))
        ) && match self.root().tokenizer.get_pre_tokenizer() {
            None => true,
            Some(P::Sequence(value)) => value.as_ref().iter().all(leaf),
            Some(value) => leaf(value),
        }
    }

    /// The actual model produces deterministic canonical BPE tokenizations;
    /// stochastic dropout and other future model mechanisms do not gain this fact.
    pub fn tokenization_is_canonical(&self) -> bool {
        match self.root().tokenizer.get_model() {
            tokenizers::ModelWrapper::BPE(model) => model.dropout.is_none(),
            tokenizers::ModelWrapper::WordLevel(_) => true,
            tokenizers::ModelWrapper::Unigram(model) => {
                model.alpha.is_none() || model.alpha == Some(0.0)
            }
            _ => false,
        }
    }

    /// Looks up an exact spelling without allocating.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.input_view().token_to_id(token)
    }
    /// Exact added-token membership without allocating or inferring special status.
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.input_view()
            .added_vocabulary()
            .get_vocab()
            .get(token)
            .copied()
    }
    /// Borrows the canonical added-first decoding spelling.
    pub fn spelling(&self, id: u32) -> Option<&str> {
        self.input_view().decode_vocabulary().id_to_token(id)
    }
    /// Borrows model-plus-added ID visits, retaining duplicate visits.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.input_view().decode_vocabulary().ids()
    }
    /// Tests actual special membership by spelling.
    pub fn is_special(&self, token: &str) -> bool {
        self.input_view().added_vocabulary().is_special_token(token)
    }
    /// Borrows the fresh immutable program for the existing fixed decoder.
    pub fn decode_source(&self) -> &PreparedDecodeSource {
        self.projection
            .as_ref()
            .and_then(|p| p.decoder.as_ref())
            .unwrap_or(&self.root().decoder)
    }
}
impl fmt::Debug for PreparedTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTokenizer").finish_non_exhaustive()
    }
}
#[derive(Debug)]
enum Cause {
    Root(TokenizerCompileFailure),
    Validation(DecodeSourceError),
    Decode(DecodeCompileFailure),
}
/// The failing compiler prefix and every earlier completed aggregate part.
pub struct TokenizerConstructionFailure {
    cause: Cause,
    tokenizer: Option<tokenizers::Tokenizer>,
}
impl TokenizerConstructionFailure {
    /// Whether this failure retains the completed HF aggregate.
    pub fn completed_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }
    /// Borrows the actual root compiler failure, if construction stopped there.
    pub fn root_failure(&self) -> Option<&TokenizerCompileFailure> {
        match &self.cause {
            Cause::Root(e) => Some(e),
            _ => None,
        }
    }
    /// Borrows the actual decoder compiler failure after HF construction.
    pub fn decode_failure(&self) -> Option<&DecodeCompileFailure> {
        match &self.cause {
            Cause::Decode(e) => Some(e),
            _ => None,
        }
    }
}
impl fmt::Debug for TokenizerConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenizerConstructionFailure")
            .field("cause", &self.cause)
            .field("completed_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}
impl fmt::Display for TokenizerConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Root(e) => fmt::Display::fmt(e, f),
            Cause::Validation(e) => fmt::Display::fmt(e, f),
            Cause::Decode(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for TokenizerConstructionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Root(e) => Some(e),
            Cause::Validation(e) => Some(e),
            Cause::Decode(e) => Some(e),
        }
    }
}
#[cfg(test)]
mod tests;

/// Fixed dependency-development selectors; these supply no capacities or account authority.
#[cfg(feature = "tokenizer-compiler-test-support")]
pub use tokenizers::{
    TokenizerRegexBuffer as RegexBuffer,
    TokenizerRegexConstructionFailure as RegexConstructionFailure,
    TokenizerRegexDelegateBuffer as RegexDelegateBuffer,
    TokenizerRegexWorkspaceFailure as RegexWorkspaceFailure,
};
