//! Borrowed input/vocabulary projection consumed by the same encoding workers.
use super::{AddedVocabulary, DecodeVocabulary, Tokenizer};

/// Actual immutable model and input vocabulary. This is a mechanism view, not
/// source identity, allocation authority, or permission to adopt a tokenizer.
#[derive(Clone, Copy)]
pub struct TokenizerInput<'a> {
    pub(super) root: &'a Tokenizer,
    pub(super) added: &'a AddedVocabulary,
    pub(super) remove_input_prefixes: bool,
}
impl<'a> From<&'a Tokenizer> for TokenizerInput<'a> {
    fn from(root: &'a Tokenizer) -> Self { Self { root, added: &root.added_vocabulary, remove_input_prefixes: false } }
}
impl Tokenizer {
    /// Project input-prefix removal with the actual refreshed added vocabulary.
    /// The enclosing source owner retains and authenticates both sources; this
    /// borrowed projection allocates nothing and shares model/decoder storage.
    pub fn input_prefix_view<'a>(&'a self, added: &'a AddedVocabulary) -> TokenizerInput<'a> {
        TokenizerInput { root: self, added, remove_input_prefixes: true }
    }
}
impl<'a> TokenizerInput<'a> {
    /// Actual projected added-first decoding vocabulary.
    pub fn decode_vocabulary(self) -> DecodeVocabulary<'a> { DecodeVocabulary { model: &self.root.model, added: self.added } }
    /// Borrow the unchanged decoder configuration.
    pub fn decoder(self) -> Option<&'a crate::DecoderWrapper> { self.root.get_decoder() }
    /// Actual projected vocabulary spelling lookup.
    pub fn token_to_id(self, spelling: &str) -> Option<u32> { self.added.token_to_id(spelling, &self.root.model) }
    /// Borrow the actual projected added vocabulary.
    pub fn added_vocabulary(self) -> &'a AddedVocabulary { self.added }
}
