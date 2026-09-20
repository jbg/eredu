use eredu_text::tokenizer::Tokenizer;

/// Borrowed vocabulary metadata for one loaded tokenizer configuration.
///
/// Every query borrows existing storage or copies a scalar. The view exposes no
/// HF object, owning snapshot, serializer or allocating text operation. The
/// fingerprint is the value already computed when the model was constructed.
/// This access boundary does not fund HF construction, caches or text operations.
///
/// Raw HF access is unavailable:
/// ```compile_fail
/// fn raw<B: eredu_core::TextGenerationBackend>(model: &eredu::api::LoadedModel<B>) {
///     let _: &tokenizers::Tokenizer = &*model.tokenizer();
/// }
/// ```
///
/// An owning tokenizer snapshot cannot escape through the view:
/// ```compile_fail
/// fn snapshot<B: eredu_core::TextGenerationBackend>(model: &eredu::api::LoadedModel<B>) {
///     let _ = model.tokenizer().snapshot();
/// }
/// ```
///
/// Iteration retains the resident borrow:
/// ```compile_fail
/// fn retire<B: eredu_core::TextGenerationBackend>(model: eredu::api::LoadedModel<B>) {
///     let view = model.tokenizer();
///     let mut ids = view.decode_ids();
///     drop(model);
///     let _ = ids.next();
/// }
/// ```
pub struct LoadedTokenizerView<'a> {
    tokenizer: &'a Tokenizer,
    vocabulary: eredu_text::tokenizer::TokenizerVocabulary<'a>,
    fingerprint: &'a [u8; 32],
}

impl<'a> LoadedTokenizerView<'a> {
    pub(super) fn new(tokenizer: &'a Tokenizer, fingerprint: &'a [u8; 32]) -> Self {
        Self {
            vocabulary: tokenizer.vocabulary(),
            tokenizer,
            fingerprint,
        }
    }

    /// Looks up an exact spelling using the tokenizer's existing added-first map.
    /// This does not normalize or tokenize arbitrary input text.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Borrows the canonical decoding spelling, including normalized added tokens.
    /// Unknown IDs return `None`; this lookup does not allocate a spelling.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocabulary.id_to_token(id)
    }

    /// Tests actual added-vocabulary special membership by exact spelling.
    /// A normalized decoding spelling need not itself be marked special.
    pub fn is_special_token(&self, token: &str) -> bool {
        self.vocabulary.is_special_token(token)
    }

    /// Iterates populated model and added reverse-map IDs.
    ///
    /// Order is unspecified and overlapping IDs can repeat. Sparse IDs are not
    /// filled in. This is the decoding vocabulary population, not a claim that
    /// every entry is eligible for generation. The iterator directly borrows
    /// existing index and invokes no user callback or allocating HF operation.
    pub fn decode_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.vocabulary.ids()
    }

    /// Borrows the existing loaded-model fingerprint without recomputing it.
    pub fn fingerprint(&self) -> &[u8; 32] {
        self.fingerprint
    }
}
