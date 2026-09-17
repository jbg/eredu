//! Borrowed vocabulary access; no raw model, allocating lookup or mutation.
use super::storage::Storage;

/// Readonly view of the existing BPE's concrete vocabulary storage.
///
/// The view does not own a model and never initializes its tokenization cache.
/// ID order is unspecified; ambiguous Legacy IDs remain duplicated in iteration.
pub struct BpeVocabulary<'a> {
    storage: &'a Storage,
}
impl<'a> BpeVocabulary<'a> {
    pub(super) fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }
    /// Number of canonical forward vocabulary spellings.
    pub fn len(&self) -> usize {
        self.storage.len()
    }
    /// Whether the forward vocabulary contains no spellings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Looks up a spelling without copying it or changing model state.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.storage.id(token).copied()
    }
    /// Borrows the canonical reverse spelling without allocating a String.
    pub fn spelling(&self, id: u32) -> Option<&'a str> {
        self.storage.token(id)
    }
    /// Borrows forward IDs directly, without scanning the sparse ID extent.
    pub fn ids(self) -> impl Iterator<Item = u32> + 'a {
        self.storage.vocab().map(|(_, id)| *id)
    }
}
impl std::fmt::Debug for BpeVocabulary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BpeVocabulary")
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BpeCompilePlan, BPE};
    use crate::Model;
    #[test]
    fn borrowed_vocabulary_preserves_packed_and_legacy_canonical_queries() {
        let json = r#"{"vocab":{"h":2,"i":5,"hi":90},"merges":[["h","i"]]}"#;
        let packed = BpeCompilePlan::prepare_model_json(json.as_bytes())
            .unwrap()
            .compile()
            .unwrap();
        let legacy: BPE = serde_json::from_str(json).unwrap();
        for model in [&packed, &legacy] {
            let view = model.vocabulary();
            assert_eq!(view.len(), 3);
            assert!(!view.is_empty());
            assert_eq!(view.token_id("hi"), Some(90));
            assert_eq!(view.spelling(90), Some("hi"));
            assert_eq!(view.spelling(3), None);
            let mut ids: Vec<_> = view.ids().collect();
            ids.sort_unstable();
            assert_eq!(ids, [2, 5, 90]);
            assert_eq!(model.tokenize("hi").unwrap()[0].id, 90);
        }
    }
}
