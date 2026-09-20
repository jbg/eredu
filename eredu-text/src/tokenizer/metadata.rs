//! Owned cold-load facts used by allocation-free decoder planning.

use crate::decoder_storage::Mode;

#[derive(Debug)]
struct Token {
    id: u32,
    spelling: String,
    special: bool,
}

#[derive(Debug)]
pub(crate) struct SnapshotMetadata {
    pub(crate) vocabulary: SnapshotVocabulary,
    pub(crate) mode: Option<Mode>,
}

impl SnapshotMetadata {
    pub(crate) fn new(tokenizer: &tokenizers::Tokenizer) -> Self {
        Self {
            vocabulary: SnapshotVocabulary::new(tokenizer),
            mode: Mode::inspect(tokenizer.get_decoder()),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SnapshotVocabulary {
    ids: Vec<u32>,
    duplicate_model_id: Option<u32>,
    tokens: Vec<Token>,
}

impl SnapshotVocabulary {
    fn new(tokenizer: &tokenizers::Tokenizer) -> Self {
        // Keep model and added IDs separate: merging by spelling can hide a
        // model ID when an added token shadows the same spelling.
        let mut ids: Vec<_> = tokenizer.get_vocab(false).into_values().collect();
        ids.sort_unstable();
        let duplicate_model_id = ids
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .map(|pair| pair[0]);
        ids.extend(
            tokenizer
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .keys()
                .copied(),
        );
        ids.sort_unstable();
        let mut tokens = Vec::new();
        let mut previous = None;
        for &id in &ids {
            if previous == Some(id) {
                continue;
            }
            previous = Some(id);
            if let Some(spelling) = tokenizer.id_to_token(id) {
                let special = tokenizer.get_added_vocabulary().is_special_token(&spelling);
                tokens.push(Token {
                    id,
                    spelling,
                    special,
                });
            }
        }
        Self {
            ids,
            tokens,
            duplicate_model_id,
        }
    }

    pub(crate) fn duplicate_model_id(&self) -> Option<u32> {
        self.duplicate_model_id
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.ids.iter().copied()
    }

    fn token(&self, id: u32) -> Option<&Token> {
        self.tokens
            .binary_search_by_key(&id, |token| token.id)
            .ok()
            .map(|index| &self.tokens[index])
    }

    pub(crate) fn id_to_token(&self, id: u32) -> Option<&str> {
        self.token(id).map(|token| token.spelling.as_str())
    }

    pub(crate) fn is_special(&self, id: u32) -> bool {
        self.token(id).is_some_and(|token| token.special)
    }
}
