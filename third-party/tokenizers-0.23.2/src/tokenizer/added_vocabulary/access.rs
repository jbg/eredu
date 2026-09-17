use super::*;
use std::borrow::Cow;

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
    /// Ordinary construction retains the legacy defaults and DAAC matcher.
    pub fn new() -> Self {
        Self {
            storage: Storage::Legacy(LegacyAddedVocabulary::new()),
        }
    }
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Legacy(v) => v.len(),
            Storage::Packed(v) => v.spellings.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrowed canonical forward spellings and IDs for both storage modes.
    pub fn vocabulary(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        let legacy = match &self.storage {
            Storage::Legacy(v) => Some(v.added_tokens_map.iter()),
            _ => None,
        };
        let packed = match &self.storage {
            Storage::Packed(v) => Some(v),
            _ => None,
        };
        legacy
            .into_iter()
            .flatten()
            .map(|(s, id)| (s.as_str(), *id))
            .chain(packed.into_iter().flat_map(|v| {
                v.spellings
                    .iter()
                    .map(move |&i| (v.content(i), v.entries[i].id))
            }))
    }
    /// Borrowed canonical reverse population; ordering is unspecified.
    pub fn tokens(&self) -> impl Iterator<Item = (u32, AddedTokenRef<'_>)> + '_ {
        let legacy = match &self.storage {
            Storage::Legacy(v) => Some(v.added_tokens_map_r.iter()),
            _ => None,
        };
        let packed = match &self.storage {
            Storage::Packed(v) => Some(v),
            _ => None,
        };
        legacy
            .into_iter()
            .flatten()
            .map(|(id, t)| (*id, t.into()))
            .chain(
                packed
                    .into_iter()
                    .flat_map(|v| v.ids.iter().map(move |&i| (v.entries[i].id, v.token(i)))),
            )
    }
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.tokens().map(|(id, _)| id)
    }
    /// Legacy borrows its existing map; Packed explicitly materializes an unmanaged copy.
    /// Use `vocabulary` for allocation-free access. The return is intentionally Cow,
    /// since a concrete borrowed map cannot describe packed storage.
    pub fn get_vocab(&self) -> Cow<'_, AHashMap<String, u32>> {
        match &self.storage {
            Storage::Legacy(v) => Cow::Borrowed(v.get_vocab()),
            Storage::Packed(_) => Cow::Owned(
                self.vocabulary()
                    .map(|(s, id)| (s.to_owned(), id))
                    .collect(),
            ),
        }
    }
    /// Explicit map compatibility; Packed copies strings into an unmanaged map.
    pub fn get_added_tokens_decoder(&self) -> Cow<'_, AHashMap<u32, AddedToken>> {
        match &self.storage {
            Storage::Legacy(v) => Cow::Borrowed(v.get_added_tokens_decoder()),
            Storage::Packed(_) => {
                Cow::Owned(self.tokens().map(|(id, t)| (id, t.to_owned())).collect())
            }
        }
    }
    pub fn token_ref(&self, id: u32) -> Option<AddedTokenRef<'_>> {
        match &self.storage {
            Storage::Legacy(v) => v.added_tokens_map_r.get(&id).map(Into::into),
            Storage::Packed(v) => v.id_index(id).map(|i| v.token(i)),
        }
    }
    pub fn token_to_id(&self, token: &str, model: &impl Model) -> Option<u32> {
        match &self.storage {
            Storage::Legacy(v) => v.added_tokens_map.get(token).copied(),
            Storage::Packed(v) => v.spelling_index(token).map(|i| v.entries[i].id),
        }
        .or_else(|| model.token_to_id(token))
    }
    #[deprecated(
        since = "0.19.0",
        note = "use simple_id_to_token followed by model.id_to_token"
    )]
    pub fn id_to_token(&self, id: u32, model: &impl Model) -> Option<String> {
        self.simple_id_to_token(id)
            .or_else(|| model.id_to_token(id))
    }
    pub(crate) fn decode_token_ref(&self, id: u32) -> Option<&str> {
        match &self.storage {
            Storage::Legacy(v) => v.decode_token_ref(id),
            Storage::Packed(v) => v.id_index(id).map(|i| v.content(i)),
        }
    }
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.decode_token_ref(id).map(str::to_owned)
    }
    pub fn is_special_token(&self, token: &str) -> bool {
        match &self.storage {
            Storage::Legacy(v) => v.is_special_token(token),
            Storage::Packed(v) => v
                .spelling_index(token)
                .is_some_and(|i| v.entries[i].special_seen),
        }
    }
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        match &mut self.storage {
            Storage::Legacy(v) => v.set_encode_special_tokens(value),
            Storage::Packed(v) => v.encode_special_tokens = value,
        }
    }
    pub fn get_encode_special_tokens(&self) -> bool {
        match &self.storage {
            Storage::Legacy(v) => v.get_encode_special_tokens(),
            Storage::Packed(v) => v.encode_special_tokens,
        }
    }
    /// Explicit unmanaged mutation materializes a complete legacy representation
    /// before replacement. A matcher-construction error preserves the packed source.
    fn legacy_mut(&mut self) -> Result<&mut LegacyAddedVocabulary> {
        if let Storage::Packed(p) = &self.storage {
            let mut v = LegacyAddedVocabulary {
                added_tokens_map: self
                    .vocabulary()
                    .map(|(s, id)| (s.to_owned(), id))
                    .collect(),
                added_tokens_map_r: self.tokens().map(|(id, t)| (id, t.to_owned())).collect(),
                special_tokens_set: p
                    .spellings
                    .iter()
                    .filter(|&&i| p.entries[i].special_seen)
                    .map(|&i| p.content(i).to_owned())
                    .collect(),
                normalized_cache: AHashMap::new(),
                split_trie: None,
                split_normalized_trie: None,
                encode_special_tokens: p.encode_special_tokens,
            };
            v.refresh_added_tokens()?;
            self.storage = Storage::Legacy(v);
        }
        match &mut self.storage {
            Storage::Legacy(v) => Ok(v),
            Storage::Packed(_) => unreachable!(),
        }
    }
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        self.legacy_mut()?.add_tokens(tokens, model, normalizer)
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
        self.legacy_mut()?.refresh_normalized_tokens(normalizer)
    }
    pub(super) fn raw_matches<'a>(
        &'a self,
        sentence: &'a str,
        normalized: bool,
    ) -> impl Iterator<Item = (u32, Offsets)> + 'a {
        let legacy = match &self.storage {
            Storage::Legacy(v) => {
                if normalized {
                    v.split_normalized_trie.as_ref()
                } else {
                    v.split_trie.as_ref()
                }
            }
            _ => None,
        };
        let packed = match &self.storage {
            Storage::Packed(v) => Some(packed::Matches::new(v, sentence, normalized)),
            _ => None,
        };
        legacy
            .into_iter()
            .flat_map(move |trie| trie.leftmost_find_iter(sentence))
            .map(|m| (m.value(), (m.start(), m.end())))
            .chain(packed.into_iter().flatten())
    }
}
impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Debug for AddedVocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.storage {
            Storage::Legacy(v) => std::fmt::Debug::fmt(v, f),
            Storage::Packed(v) => f
                .debug_struct("PackedAddedVocabulary")
                .field("tokens", &v.ids.len())
                .field("encode_special_tokens", &v.encode_special_tokens)
                .finish(),
        }
    }
}
impl Serialize for AddedVocabulary {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match &self.storage {
            Storage::Legacy(v) => v.serialize(serializer),
            Storage::Packed(v) => {
                #[derive(Serialize)]
                struct Entry<'a> {
                    id: u32,
                    #[serde(flatten)]
                    token: AddedTokenRef<'a>,
                }
                let mut seq = serializer.serialize_seq(Some(v.ids.len()))?;
                for &i in &v.ids {
                    seq.serialize_element(&Entry {
                        id: v.entries[i].id,
                        token: v.token(i),
                    })?;
                }
                seq.end()
            }
        }
    }
}
