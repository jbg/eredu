use self::serialization::OrderedVocabulary;
use crate::tokenizer::{Model, Result, Token};
use ahash::AHashMap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

mod serialization;
pub(crate) use serialization::Serialization as WordLevelSerialization;
mod compile;
pub use compile::{WordLevelCompileError, WordLevelCompileFailure, WordLevelCompilePlan, WordLevelCompileRequirements};
mod trainer;

// Re-export
pub use trainer::*;

type Vocab = AHashMap<String, u32>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordLevel error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
    #[error("Bad vocabulary json file")]
    BadVocabulary,
}

struct Config {
    files: Option<String>,
    vocab: AHashMap<String, u32>,
    unk_token: String,
}

/// A `WordLevelBuilder` can be used to create a `WordLevel`
/// model with a custom configuration.
pub struct WordLevelBuilder {
    config: Config,
}

impl Default for WordLevelBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: AHashMap::new(),
                unk_token: String::from("<unk>"),
            },
        }
    }
}

impl WordLevelBuilder {
    /// Construct a new `WordLevelBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    #[must_use]
    pub fn vocab(mut self, vocab: AHashMap<String, u32>) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Constructs a `WordLevel` model that uses the `WordLevelBuilder`'s configuration.
    pub fn build(mut self) -> Result<WordLevel> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordLevel::read_file(&vocab)?;
        }

        let vocab: Vocabulary = self.config.vocab.into_iter().collect();
        let mut vocab_r = ReverseVocabulary::new();
        vocab_r.try_reserve(vocab.len())?;
        reverse_vocabulary(&vocab, &mut vocab_r, &mut String::new(), |target, length| target.try_reserve_exact(length))?;

        Ok(WordLevel {
            vocab,
            vocab_r,
            unk_token: self.config.unk_token,
        })
    }
}

type Vocabulary = hashbrown::HashMap<String, u32>;
type ReverseVocabulary = hashbrown::HashMap<u32, String>;

#[derive(PartialEq, Clone, Eq)]
pub struct WordLevel {
    vocab: Vocabulary,
    vocab_r: ReverseVocabulary,
    pub unk_token: String,
}

impl std::fmt::Debug for WordLevel {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordLevel")
            .field("unk_token", &self.unk_token)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

fn reverse_vocabulary<E>(vocab: &Vocabulary, reverse: &mut ReverseVocabulary, pending: &mut String,
    mut reserve: impl FnMut(&mut String, usize) -> std::result::Result<(), E>) -> std::result::Result<(), E> {
    for (spelling, &id) in vocab {
        reserve(pending, spelling.len())?;
        pending.push_str(spelling);
        reverse.insert(id, std::mem::take(pending));
    }
    Ok(())
}

impl WordLevel {
    pub(crate) fn same_configuration(&self,other:&Self)->bool {
        self.unk_token==other.unk_token && self.vocab==other.vocab && self.vocab_r==other.vocab_r
    }
    pub(crate) fn configuration_comparison_control_bytes()->Option<usize> {
        use std::mem::size_of;
        [size_of::<[&Self;2]>(),size_of::<hashbrown::hash_map::Iter<'_,String,u32>>(),
         size_of::<hashbrown::hash_map::Iter<'_,u32,String>>(),size_of::<(&String,&u32)>(),
         size_of::<(&u32,&String)>(),size_of::<[usize;2]>(),size_of::<bool>()]
         .iter().copied().try_fold(0usize,usize::checked_add)
    }

    /// The original whole-span model lookup, shared by token and ID emission.
    pub(crate) fn lookup<'a>(&'a self, token: &'a str) -> std::result::Result<(u32, &'a str), Error> {
        if let Some(&id) = self.vocab.get(token) {
            Ok((id, token))
        } else if let Some(&id) = self.vocab.get(&self.unk_token) {
            Ok((id, &self.unk_token))
        } else {
            Err(Error::MissingUnkToken)
        }
    }

    pub(crate) fn decode_ids(
        &self,
    ) -> std::iter::Copied<hashbrown::hash_map::Values<'_, String, u32>> {
        self.vocab.values().copied()
    }

    pub(crate) fn visit_decode_ids(&self, visitor: &mut dyn FnMut(u32)) {
        for id in self.vocab.values() {
            visitor(*id);
        }
    }

    pub(crate) fn decode_token_ref(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(String::as_str)
    }

    pub fn builder() -> WordLevelBuilder {
        WordLevelBuilder::new()
    }

    pub fn read_file(vocab_path: &str) -> Result<Vocab> {
        let vocab_file = File::open(vocab_path)?;
        let mut vocab_file = BufReader::new(vocab_file);
        let mut buffer = String::new();
        let mut vocab = AHashMap::new();

        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;

        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };
        Ok(vocab)
    }

    /// Initialize a WordLevel model from vocab and merges file.
    pub fn from_file(vocab_path: &str, unk_token: String) -> Result<WordLevel> {
        let vocab = WordLevel::read_file(vocab_path)?;
        Self::builder().vocab(vocab).unk_token(unk_token).build()
    }
}

impl Default for WordLevel {
    fn default() -> Self {
        Self {
            vocab: Vocabulary::new(),
            vocab_r: ReverseVocabulary::new(),
            unk_token: String::from("<unk>"),
        }
    }
}

impl Model for WordLevel {
    type Trainer = WordLevelTrainer;

    fn tokenize(&self, token: &str) -> Result<Vec<Token>> {
        let (id, value) = self.lookup(token)?;
        Ok(vec![Token { id, value: value.to_owned(), offsets: (0, token.len()) }])
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.iter().map(|(spelling, &id)| (spelling.clone(), id)).collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.keys().len()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabulary(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(serialized.as_bytes())?;

        Ok(vec![vocab_path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        WordLevelTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_unk() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let wordlevel = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = wordlevel.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = wordlevel.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "a".into(), (0, 1)),]);
    }

    #[test]
    fn test_tokenize_missing_unk_token() {
        let vocab: Vocab = [("a".into(), 0), ("b".into(), 1)].iter().cloned().collect();
        let wordlevel = WordLevelBuilder::default().vocab(vocab).build().unwrap();
        let tokens = wordlevel.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "a".into(), (0, 1)),]);

        let error = wordlevel.tokenize("c").err().unwrap();
        assert!(error.is::<Error>());
    }
}
