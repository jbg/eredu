use super::*;
use std::convert::TryFrom;

/// Borrowed properties of one actual added token; no owned String or map.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct AddedTokenRef<'a> {
    pub content: &'a str,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}
impl<'a> From<&'a AddedToken> for AddedTokenRef<'a> {
    fn from(t: &'a AddedToken) -> Self {
        Self {
            content: &t.content,
            single_word: t.single_word,
            lstrip: t.lstrip,
            rstrip: t.rstrip,
            normalized: t.normalized,
            special: t.special,
        }
    }
}
impl AddedTokenRef<'_> {
    pub fn to_owned(self) -> AddedToken {
        AddedToken {
            content: self.content.to_owned(),
            single_word: self.single_word,
            lstrip: self.lstrip,
            rstrip: self.rstrip,
            normalized: self.normalized,
            special: self.special,
        }
    }
}
impl AddedVocabulary {
    pub fn new() -> Self {
        Self {
            storage: packed::Packed::default(),
        }
    }
    pub fn len(&self) -> usize {
        self.storage.spellings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrowed canonical spellings; allocation-free for every constructor.
    pub fn vocabulary(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.storage
            .spellings
            .iter()
            .map(move |&i| (self.storage.content(i), self.storage.entries[i].id))
    }
    pub fn tokens(&self) -> impl Iterator<Item = (u32, AddedTokenRef<'_>)> + '_ {
        self.storage
            .ids
            .iter()
            .map(move |&i| (self.storage.entries[i].id, self.storage.token(i)))
    }
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.tokens().map(|(id, _)| id)
    }
    /// Explicit owned copy. Use `vocabulary` for borrowed traversal.
    pub fn get_vocab(&self) -> AHashMap<String, u32> {
        self.vocabulary()
            .map(|(s, id)| (s.to_owned(), id))
            .collect()
    }
    /// Explicit owned copy. Use `tokens` for borrowed traversal.
    pub fn get_added_tokens_decoder(&self) -> AHashMap<u32, AddedToken> {
        self.tokens().map(|(id, t)| (id, t.to_owned())).collect()
    }
    pub fn token_ref(&self, id: u32) -> Option<AddedTokenRef<'_>> {
        self.storage.id_index(id).map(|i| self.storage.token(i))
    }
    pub fn token_to_id(&self, token: &str, model: &impl Model) -> Option<u32> {
        self.storage
            .spelling_index(token)
            .map(|i| self.storage.entries[i].id)
            .or_else(|| model.token_to_id(token))
    }
    pub(crate) fn decode_token_ref(&self, id: u32) -> Option<&str> {
        self.storage.id_index(id).map(|i| self.storage.pattern(i))
    }
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.decode_token_ref(id).map(str::to_owned)
    }
    pub fn is_special_token(&self, token: &str) -> bool {
        self.storage
            .spelling_index(token)
            .is_some_and(|i| self.storage.entries[i].special_seen)
    }
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.storage.encode_special_tokens = value
    }
    pub fn get_encode_special_tokens(&self) -> bool {
        self.storage.encode_special_tokens
    }
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        let mut values: Vec<_> = self
            .tokens()
            .map(|(id, token)| (id, token.to_owned(), self.is_special_token(token.content)))
            .collect();
        // Temporary mutation index; the retained execution representation is packed.
        let mut index: AHashMap<_, _> = values
            .iter()
            .enumerate()
            .map(|(i, (_, token, _))| (token.content.clone(), i))
            .collect();
        let mut next_id = values
            .iter()
            .map(|(id, _, _)| u64::from(*id) + 1)
            .max()
            .unwrap_or(0)
            .max(model.get_vocab_size() as u64);
        let mut changed = 0;
        for token in tokens {
            if token.content.is_empty() {
                continue;
            }
            if let Some(&i) = index.get(&token.content) {
                if values[i].1 == token {
                    continue;
                }
                values[i].2 |= token.special;
                values[i].1 = token;
            } else {
                let id = if let Some(id) = model.token_to_id(&token.content) {
                    id
                } else {
                    let id = u32::try_from(next_id).map_err(|_| "added token ID overflow")?;
                    next_id += 1;
                    id
                };
                index.insert(token.content.clone(), values.len());
                let special = token.special;
                values.push((id, token, special));
            }
            changed += 1;
        }
        let mut storage = packed::Packed::from_tokens(values, normalizer)?;
        storage.encode_special_tokens = self.storage.encode_special_tokens;
        self.storage = storage;
        Ok(changed)
    }
    pub fn add_special_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        self.add_tokens(tokens, model, normalizer)
    }
    pub fn refresh_normalized_tokens<N: Normalizer>(
        &mut self,
        normalizer: Option<&N>,
    ) -> Result<()> {
        let values = self
            .tokens()
            .map(|(id, token)| (id, token.to_owned(), self.is_special_token(token.content)))
            .collect();
        let mut storage = packed::Packed::from_tokens(values, normalizer)?;
        storage.encode_special_tokens = self.storage.encode_special_tokens;
        self.storage = storage;
        Ok(())
    }
    pub(super) fn raw_matches<'a>(
        &'a self,
        sentence: &'a str,
        normalized: bool,
    ) -> impl Iterator<Item = (u32, Offsets)> + 'a {
        self.storage
            .matcher
            .matches(sentence, normalized, move |index| {
                self.storage.pattern_len(index as usize)
            })
            .map(move |(index, offsets)| (self.storage.entries[index as usize].id, offsets))
    }
}
impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Debug for AddedVocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddedVocabulary")
            .field("tokens", &self.len())
            .field("encode_special_tokens", &self.storage.encode_special_tokens)
            .finish()
    }
}
impl Serialize for AddedVocabulary {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Entry<'a> {
            id: u32,
            #[serde(flatten)]
            token: AddedTokenRef<'a>,
        }
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for (id, token) in self.tokens() {
            seq.serialize_element(&Entry { id, token })?
        }
        seq.end()
    }
}
