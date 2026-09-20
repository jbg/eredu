//! Immutable stock-HF tokenizer and Eredu's fixed-buffer decode program.
mod encode;
mod geometry;
mod prefix;
pub use encode::{EncodeIdsError, EncodeIdsFailure, EncodeIdsPlan, EncodedTokenIds};
pub use prefix::{InputPrefixFailure, InputPrefixPlan, remove_input_prefixes};

use crate::{
    decoder_storage::{
        DecodeCompileFailure, DecodeCompilePlan, DecodeSourceError, PreparedDecodeSource,
    },
    tokenizer::{ModelCachePolicy, metadata::SnapshotMetadata},
};
use std::{fmt, sync::Arc};

/// Source inspection or upstream construction failure.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerSourceError {
    /// Stock tokenizer parsing or construction error.
    #[error("tokenizer construction failed: {0}")]
    Root(#[source] tokenizers::Error),
    /// Eredu's decoder could not compile this configuration.
    #[error(transparent)]
    Decoder(DecodeSourceError),
}
impl TokenizerSourceError {
    fn overflow() -> Self {
        Self::Decoder(DecodeSourceError::Overflow)
    }
}

/// Configurable estimate for dependency construction and encoding work.
/// The input size is measured; dependency allocations are not. These parameters
/// provide admission headroom, never an enforceable heap or process ceiling.
#[derive(Debug, Clone, Copy)]
pub struct TokenizerMemoryEstimate {
    /// Fixed headroom for each independent construction or encoding operation.
    pub fixed_bytes: usize,
    /// Construction headroom per serialized tokenizer byte, including metadata,
    /// transient model copies for cache configuration, and the decoder program.
    pub bytes_per_source_byte: usize,
    /// Encoding headroom per UTF-8 input byte, including upstream Encoding output.
    pub bytes_per_text_byte: usize,
}
impl Default for TokenizerMemoryEstimate {
    fn default() -> Self {
        Self {
            fixed_bytes: 64 * 1024,
            bytes_per_source_byte: 128,
            bytes_per_text_byte: 512,
        }
    }
}
impl TokenizerMemoryEstimate {
    fn construction(self, bytes: usize) -> Option<usize> {
        bytes
            .checked_mul(self.bytes_per_source_byte)?
            .checked_add(self.fixed_bytes)
    }
    fn encoding(self, bytes: usize) -> Option<usize> {
        bytes
            .checked_mul(self.bytes_per_text_byte)?
            .checked_add(self.fixed_bytes)
    }
}

/// Admission estimate for fresh construction, including dependency headroom.
#[derive(Debug, Clone, Copy)]
pub struct TokenizerRequirements {
    estimated_bytes: usize,
}
impl TokenizerRequirements {
    /// Estimated construction footprint, not a measured allocation count or bound.
    pub fn required_bytes(&self) -> usize {
        self.estimated_bytes
    }
}

/// One borrowed construction input, consumed by one admitted attempt.
/// Planning computes an estimate without constructing HF. Malformed tokenizer
/// configurations are diagnosed during construction through upstream's parser.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::TokenizerPlan;
/// fn repeat(plan:TokenizerPlan<'_>){let _=plan.compile();let _=plan.compile();}
/// ```
#[derive(Debug)]
pub struct TokenizerPlan<'a> {
    input: &'a [u8],
    estimate: TokenizerMemoryEstimate,
    requirements: TokenizerRequirements,
    encode_special_tokens: bool,
    generation_domain: Option<usize>,
    #[cfg(feature = "tokenizer-compiler-test-support")]
    decode_failure: Option<usize>,
}
impl<'a> TokenizerPlan<'a> {
    /// Computes default headroom from the borrowed serialized input length.
    pub fn prepare_json(input: &'a [u8]) -> Result<Self, TokenizerSourceError> {
        let estimate = TokenizerMemoryEstimate::default();
        let estimated_bytes = estimate
            .construction(input.len())
            .ok_or_else(TokenizerSourceError::overflow)?;
        Ok(Self {
            input,
            estimate,
            requirements: TokenizerRequirements { estimated_bytes },
            encode_special_tokens: false,
            generation_domain: None,
            #[cfg(feature = "tokenizer-compiler-test-support")]
            decode_failure: None,
        })
    }
    /// Overrides planning headroom; downstream operations retain this choice.
    pub fn with_memory_estimate(
        mut self,
        estimate: TokenizerMemoryEstimate,
    ) -> Result<Self, TokenizerSourceError> {
        self.requirements.estimated_bytes = estimate
            .construction(self.input.len())
            .ok_or_else(TokenizerSourceError::overflow)?;
        self.estimate = estimate;
        Ok(self)
    }
    /// Selects HF's non-serde special-token splitting flag before construction.
    pub fn with_encode_special_tokens(mut self, enabled: bool) -> Self {
        self.encode_special_tokens = enabled;
        self
    }
    /// Returns the estimate for this construction attempt.
    pub fn requirements(&self) -> TokenizerRequirements {
        self.requirements
    }
    /// Inspects only vocabulary geometry in the JSON input. The streaming serde
    /// reader may allocate scratch for escaped strings; no model is constructed.
    pub fn with_generation_domain(mut self) -> Result<Self, TokenizerSourceError> {
        let extent = geometry::domain_extent(self.input)
            .map_err(|error| TokenizerSourceError::Root(Box::new(error)))?;
        std::alloc::Layout::array::<bool>(extent).map_err(|_| TokenizerSourceError::overflow())?;
        self.generation_domain = Some(extent);
        Ok(self)
    }
    /// Upper bound for the dense canonical-ID mask; absent for decoder-only use.
    pub fn generation_domain_extent(&self) -> Option<usize> {
        self.generation_domain
    }
    /// Constructs stock HF with its model cache disabled, then owns the cold
    /// vocabulary index and the fixed-buffer decode program. The admitting owner
    /// must retain its reservation through completion or failure retirement.
    pub fn compile(self) -> Result<PreparedTokenizer, TokenizerConstructionFailure> {
        let mut tokenizer = ModelCachePolicy::disabled()
            .from_bytes(self.input)
            .map_err(|error| TokenizerConstructionFailure {
                cause: Cause::Root(error),
                tokenizer: None,
                _metadata: None,
            })?;
        tokenizer.set_encode_special_tokens(self.encode_special_tokens);
        build(
            tokenizer,
            self.input.len(),
            self.estimate,
            None,
            #[cfg(feature = "tokenizer-compiler-test-support")]
            self.decode_failure,
        )
    }
    /// Development-only failure at one of Eredu's three actual decoder reserves.
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    pub fn fail_decode_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 3);
        self.decode_failure = Some(stage);
        self
    }
}

/// Opaque immutable aggregate. Raw HF and mutable source access remain private.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::PreparedTokenizer;
/// fn raw(value:PreparedTokenizer){let _=value.tokenizer;}
/// ```
/// ```compile_fail
/// use eredu_text::tokenizer_storage::PreparedTokenizer;
/// fn copy(value:PreparedTokenizer){let _=value.clone();}
/// ```
pub struct PreparedTokenizer {
    root: Arc<Root>,
}
#[derive(Debug)]
struct Root {
    tokenizer: tokenizers::Tokenizer,
    metadata: SnapshotMetadata,
    decoder: PreparedDecodeSource,
    source_bytes: usize,
    estimate: TokenizerMemoryEstimate,
    origin: Option<Arc<Root>>,
}
fn build(
    tokenizer: tokenizers::Tokenizer,
    source_bytes: usize,
    estimate: TokenizerMemoryEstimate,
    origin: Option<Arc<Root>>,
    #[cfg(feature = "tokenizer-compiler-test-support")] fail_decode: Option<usize>,
) -> Result<PreparedTokenizer, TokenizerConstructionFailure> {
    let metadata = SnapshotMetadata::new(&tokenizer);
    let result = (|| {
        let plan = DecodeCompilePlan::prepare_metadata(&metadata).map_err(Cause::Validation)?;
        #[cfg(feature = "tokenizer-compiler-test-support")]
        let plan = if let Some(stage) = fail_decode {
            plan.fail_reservation(stage)
        } else {
            plan
        };
        plan.compile().map_err(Cause::Decode)
    })();
    match result {
        Ok(decoder) => Ok(PreparedTokenizer {
            root: Arc::new(Root {
                tokenizer,
                metadata,
                decoder,
                source_bytes,
                estimate,
                origin,
            }),
        }),
        Err(cause) => Err(TokenizerConstructionFailure {
            cause,
            tokenizer: Some(tokenizer),
            _metadata: Some(metadata),
        }),
    }
}
fn same_configuration(a: &tokenizers::Tokenizer, b: &tokenizers::Tokenizer) -> bool {
    let non_serde_model = match (a.get_model(), b.get_model()) {
        (tokenizers::ModelWrapper::Unigram(a), tokenizers::ModelWrapper::Unigram(b)) => {
            a.alpha == b.alpha && a.nbest_size == b.nbest_size && a.min_score == b.min_score
        }
        _ => true,
    };
    non_serde_model
        && a.get_encode_special_tokens() == b.get_encode_special_tokens()
        && match (serde_json::to_value(a), serde_json::to_value(b)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
}
impl PreparedTokenizer {
    fn root(&self) -> &Root {
        &self.root
    }
    fn input_view(&self) -> &tokenizers::Tokenizer {
        &self.root.tokenizer
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
    /// Compares public serialized configuration and the non-serde special-token
    /// flag. This cold check allocates temporary JSON values; it grants no funding.
    pub fn matches_configuration(&self, selected: &crate::tokenizer::Tokenizer) -> bool {
        same_configuration(self.input_view(), selected)
    }

    /// Checks the retained origin of a prefix-normalized copy, or equal unchanged
    /// configurations. Model copies remain independently resident and charged.
    pub fn is_input_prefix_derivative_of(&self, original: &Self) -> bool {
        self.root()
            .origin
            .as_ref()
            .is_some_and(|origin| Arc::ptr_eq(origin, &original.root))
            || (self.input_prefix_removal_is_identity()
                && original.input_prefix_removal_is_identity()
                && same_configuration(self.input_view(), original.input_view()))
    }

    /// Whether removing Prepend normalizers and Metaspace input prefixes changes
    /// any selected component. Decoder prefixes are deliberately unaffected.
    pub fn input_prefix_removal_is_identity(&self) -> bool {
        prefix::removal_is_identity(self.input_view())
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
            .get_added_vocabulary()
            .get_vocab()
            .get(token)
            .copied()
    }
    /// Borrows the canonical added-first decoding spelling.
    pub fn spelling(&self, id: u32) -> Option<&str> {
        self.root().metadata.vocabulary.id_to_token(id)
    }
    /// Borrows model-plus-added ID visits, retaining duplicate visits.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.root().metadata.vocabulary.ids()
    }
    /// Tests actual special membership by spelling.
    pub fn is_special(&self, token: &str) -> bool {
        self.input_view()
            .get_added_vocabulary()
            .is_special_token(token)
    }
    /// Borrows the fresh immutable program for the existing fixed decoder.
    pub fn decode_source(&self) -> &PreparedDecodeSource {
        &self.root().decoder
    }
}
impl fmt::Debug for PreparedTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTokenizer").finish_non_exhaustive()
    }
}
#[derive(Debug)]
enum Cause {
    Root(tokenizers::Error),
    Validation(DecodeSourceError),
    Decode(DecodeCompileFailure),
}
/// Construction error retaining completed HF and metadata when decoder
/// construction fails. Upstream owns and retires its own partial parser state.
#[derive(Debug)]
pub struct TokenizerConstructionFailure {
    cause: Cause,
    tokenizer: Option<tokenizers::Tokenizer>,
    _metadata: Option<SnapshotMetadata>,
}
impl TokenizerConstructionFailure {
    /// Whether this failure retains the completed HF aggregate.
    pub fn completed_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }
    /// Original upstream construction cause, if HF construction failed.
    pub fn root_failure(&self) -> Option<&(dyn std::error::Error + Send + Sync)> {
        match &self.cause {
            Cause::Root(error) => Some(error.as_ref()),
            _ => None,
        }
    }
    /// Actual decoder failure, including successfully reserved buffer prefixes.
    pub fn decode_failure(&self) -> Option<&DecodeCompileFailure> {
        match &self.cause {
            Cause::Decode(error) => Some(error),
            _ => None,
        }
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
        Some(match &self.cause {
            Cause::Root(e) => e.as_ref(),
            Cause::Validation(e) => e,
            Cause::Decode(e) => e,
        })
    }
}
#[cfg(test)]
mod tests;
