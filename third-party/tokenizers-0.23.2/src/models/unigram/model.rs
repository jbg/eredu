use super::{lattice::Lattice, trainer::UnigramTrainer, trie::Trie};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::{Cache, MAX_LENGTH};
use std::collections::HashMap;

use std::convert::TryInto;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

pub(super) type TokenMap = hashbrown::HashMap<String, u32>;
pub(super) type Vocab = Vec<(String, f64)>;

/// A `Unigram` model to encode sentences.
pub struct Unigram {
    pub(super) token_to_ids: TokenMap,
    pub(crate) vocab: Vocab,
    pub(super) cache: Option<Cache<String, Vec<String>>>,
    pub(super) trie: Trie<u8>,
    pub min_score: f64,
    pub(super) unk_id: Option<usize>,
    pub(super) bos_id: usize,
    pub(super) eos_id: usize,

    pub(super) fuse_unk: bool,
    pub(super) is_optimized: bool,
    pub(super) byte_fallback: bool,

    pub alpha: Option<f64>,
    pub nbest_size: Option<usize>,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id
            && self.vocab == other.vocab
            && self.alpha == other.alpha
            && self.nbest_size == other.nbest_size
    }
}

impl Clone for Unigram {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new Unigram will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.as_ref().map(Cache::fresh);
        Self {
            vocab: self.vocab.clone(),
            cache: fresh_cache,
            token_to_ids: self.token_to_ids.clone(),
            trie: self.trie.clone(),
            min_score: self.min_score,
            unk_id: self.unk_id,
            bos_id: self.bos_id,
            eos_id: self.eos_id,
            fuse_unk: self.fuse_unk,
            is_optimized: self.is_optimized,
            byte_fallback: self.byte_fallback,
            alpha: self.alpha,
            nbest_size: self.nbest_size,
        }
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

pub(super) static K_UNK_PENALTY: f64 = 10.0;

#[derive(thiserror::Error, Debug)]
pub enum UnigramError {
    #[error("The vocabulary is empty but at least <unk> is needed")]
    EmptyVocabulary,
    #[error("The `unk_id` is larger than vocabulary size")]
    UnkIdNotInVocabulary,
    #[error("Encountered an unknown token but `unk_id` is missing")]
    MissingUnkId,
}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, Some(0), false).unwrap()
    }
}

impl Unigram {
    pub(crate) fn cache_capacity(&self) -> usize {
        self.cache.as_ref().map_or(0, |cache| cache.capacity)
    }

    pub(crate) fn decode_ids(
        &self,
    ) -> std::iter::Copied<hashbrown::hash_map::Values<'_, String, u32>> {
        self.token_to_ids.values().copied()
    }

    pub(crate) fn visit_decode_ids(&self, visitor: &mut dyn FnMut(u32)) {
        for id in self.token_to_ids.values() {
            visitor(*id);
        }
    }

    pub(crate) fn decode_token_ref(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|(token, _)| token.as_str())
    }

    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self> {
        Self::from_with_cache_policy(
            vocab,
            unk_id,
            byte_fallback,
            crate::ModelCachePolicy::default(),
        )
    }

    /// Constructs without model-cache storage when explicitly selected. Other
    /// model/trie construction and tokenization allocations are unchanged.
    pub fn from_with_cache_policy(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
        policy: crate::ModelCachePolicy,
    ) -> Result<Self> {
        let n = vocab.len();
        let mut token_to_ids = TokenMap::new();
        let mut trie = Trie::default();

        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(Box::new(UnigramError::EmptyVocabulary));
            }
            if unk_id >= vocab.len() {
                return Err(Box::new(UnigramError::UnkIdNotInVocabulary));
            }
        }
        let bos_id = n + 1;
        let eos_id = n + 2;

        let mut pending = String::new();
        index_vocabulary(
            &vocab,
            &mut token_to_ids,
            &mut trie,
            &mut pending,
            |out, size| out.try_reserve_exact(size),
        )?;
        let min_score = vocab.iter().fold(
            f64::INFINITY,
            |min, (_, score)| if *score < min { *score } else { min },
        );
        let fuse_unk = true;
        let is_optimized = true;

        Ok(Self {
            vocab,
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
            cache: (policy.capacity != 0).then(|| Cache::new(policy.capacity)),
            is_optimized,
            byte_fallback,
            alpha: None,
            nbest_size: None,
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
        self.cache = self.cache.as_ref().map(Cache::fresh);
    }

    #[cfg(test)]
    pub(super) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
    }
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }
    pub(super) fn len(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;

        let len = lattice.len();

        let mut begin_pos = 0;
        while begin_pos < len {
            let mblen = lattice.sentence[begin_pos..]
                .chars()
                .next()
                .unwrap()
                .len_utf8();

            let mut has_single_node = false;

            for n in self
                .trie
                .common_prefix_lengths(lattice.sentence[begin_pos..].bytes())
            {
                let tok = &lattice.sentence[begin_pos..begin_pos + n];
                let id = *self.token_to_ids.get(tok).unwrap();

                let item = &self.vocab[id as usize];
                assert_eq!(item.0, tok);
                let score: f64 = item.1;
                lattice.insert(begin_pos, n, score, id.try_into().unwrap());
                if !has_single_node && n == mblen {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                if let Some(unk_id) = self.unk_id {
                    lattice.insert(begin_pos, mblen, unk_score, unk_id);
                }
            }
            begin_pos += mblen
        }
    }

    /// This functions take a String, and will encode it in a Vec of Strings,
    /// of the best tokenization available to the current model.
    /// ```
    /// use tokenizers::models::unigram::Unigram;
    ///
    /// let pieces = vec![
    ///     ("<unk>".to_string(), 0.0),
    ///     ("a".to_string(), 0.0),
    ///     ("b".to_string(), 0.0),
    ///     ("c".to_string(), 0.0),
    ///     ("d".to_string(), 0.0),
    ///     ("cd".to_string(), 1.0),
    ///     ("ab".to_string(), 2.0),
    ///     ("abc".to_string(), 5.0),
    ///     ("abcd".to_string(), 10.0),
    /// ];
    /// let model = Unigram::from(pieces, Some(0), false).unwrap();
    /// let result = model.encode("abcdacdxx").unwrap();
    /// assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
    /// ```
    pub fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }
        if self.alpha.is_none() || self.alpha == Some(0.0) {
            if let Some(result) = self.cache.as_ref().and_then(|cache| cache.get(sentence)) {
                Ok(result.to_vec())
            } else {
                let result = if self.is_optimized {
                    self.encode_optimized(sentence)?
                } else {
                    self.encode_unoptimized(sentence)?
                };
                if sentence.len() < MAX_LENGTH {
                    if let Some(cache) = &self.cache {
                        cache.set(sentence.to_owned(), result.clone());
                    }
                }
                Ok(result)
            }
        } else {
            let result = self.encode_unoptimized(sentence)?;
            Ok(result)
        }
    }

    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut scratch = super::encode::UnigramScratch::default();
        scratch.reserve_nodes(sentence.len() + 1)?;
        scratch.reserve_spans(sentence.len())?;
        scratch.encode(self, sentence)?;
        Ok(scratch
            .spans()
            .iter()
            .map(|&(start, end)| sentence[start..end].to_owned())
            .collect())
    }

    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        let path = match (self.nbest_size, self.alpha) {
            (Some(n), Some(alpha)) if n > 0 => lattice.sample_nbest(n, alpha),
            (_, Some(alpha)) => lattice.sample(alpha),
            _ => lattice.viterbi(),
        };
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in path.iter() {
                let item = lattice.piece(&node.borrow());
                if node.borrow().id == self.unk_id.ok_or(UnigramError::MissingUnkId)? {
                    token.push_str(&item);
                } else {
                    if !token.is_empty() {
                        results.push(token);
                        token = String::new();
                    }
                    results.push(item);
                }
            }
            if !token.is_empty() {
                results.push(token);
            }
            Ok(results)
        } else {
            let results: Vec<String> = path
                .iter()
                .map(|node| lattice.piece(&node.borrow()))
                .collect();
            Ok(results)
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub fn iter(&self) -> UnigramIterator<'_> {
        UnigramIterator { model: self, i: 0 }
    }

    /// Loads a SentencePiece output model after being trained by tokenizers.
    /// After that you can use the model with tokenizers library.
    /// ```no_run
    /// use tokenizers::models::unigram::Unigram;
    /// use std::path::Path;
    ///
    /// let model = Unigram::load("mymodel-unigram.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Unigram> {
        let string = read_to_string(path)?;
        Ok(serde_json::from_str(&string)?)
    }

    pub(crate) fn cache_is_enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// Clears the internal cache
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }

    /// Resize the cache
    pub fn resize_cache(&mut self, capacity: usize) {
        if capacity == 0 {
            self.cache = None;
            return;
        }
        if let Some(cache) = &mut self.cache {
            cache.resize(capacity);
        }
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.model.len() {
            let r = Some(&self.model.vocab[i]);
            self.i += 1;
            r
        } else {
            None
        }
    }
}

impl Model for Unigram {
    type Trainer = UnigramTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
        let str_tokens = self.encode(sentence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(str_tokens.len());
        for string in str_tokens {
            let len = string.len();
            let offsets = (offset, offset + len);
            self.visit_piece_ids(&string, |id, text| {
                tokens.push(Token::new(id, text.to_owned(), offsets))
            })?;
            offset += len;
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|item| item.0.clone())
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let name = match name {
            Some(name) => format!("{name}-unigram.json"),
            None => "unigram.json".to_string(),
        };
        let mut fullpath = PathBuf::new();
        fullpath.push(folder);
        fullpath.push(name);
        let string = serde_json::to_string_pretty(self)?;
        std::fs::write(&fullpath, string)?;
        Ok(vec![fullpath])
    }

    fn get_trainer(&self) -> Self::Trainer {
        UnigramTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_populate_nodes_unk() {
        let pieces = vec![("<unk>".to_string(), 0.0)];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 1);
        assert_eq!(lattice.begin_nodes[1].len(), 1);
        assert_eq!(lattice.begin_nodes[2].len(), 1);
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 4);
    }

    #[test]
    fn test_populate_nodes() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.2),
            ("ab".to_string(), 0.3),
            ("bc".to_string(), 0.4),
        ];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 2); // a, ab
        assert_eq!(lattice.begin_nodes[1].len(), 2); // b, bc
        assert_eq!(lattice.begin_nodes[2].len(), 1); // c(unk)

        // Id is the vocabulary id from Unigram model
        // node_id is simply the rank of the given node in the lattice.
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 1);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 2);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, 4);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[0][1].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 4);
        assert_eq!(lattice.begin_nodes[1][1].borrow().node_id, 5);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 6);
    }

    #[test]
    fn test_encode() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
            ("cd".to_string(), 1.0),
            ("ab".to_string(), 2.0),
            ("abc".to_string(), 5.0),
            ("abcd".to_string(), 10.0),
        ];

        let model = Unigram::from(sentencepieces, Some(0), false).unwrap();
        let result = model.encode("abcd").unwrap();
        assert_eq!(result, vec!["abcd"]);
    }

    #[test]
    fn test_encode2() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("ab".to_string(), 0.0),
            ("cd".to_string(), -0.1),
            ("abc".to_string(), -0.2),
            ("a".to_string(), -0.3),
            ("b".to_string(), -0.4),
            ("c".to_string(), -0.5),
            ("ABC".to_string(), -0.5),
            ("abcdabcd".to_string(), 20.0), // User defined just max the scores.
            ("q".to_string(), 20.5),
            ("r".to_string(), 20.5),
            ("qr".to_string(), -0.5),
        ];

        let mut model = Unigram::from(sentencepieces, Some(0), false).unwrap();

        for is_optimized in &[true, false] {
            model.set_optimized(*is_optimized);
            println!("IsOptimized {is_optimized:?}");
            assert_eq!(model.encode("abc").unwrap(), vec!["abc"]);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            model.set_fuse_unk(false);
            assert_eq!(model.encode("AB").unwrap(), vec!["A", "B"]);
            model.set_fuse_unk(true);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            assert_eq!(model.encode("abcd").unwrap(), vec!["ab", "cd"]);
            assert_eq!(model.encode("abcc").unwrap(), vec!["abc", "c"]);
            assert_eq!(
                model.encode("xabcabaabcdd").unwrap(),
                vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
            );
            model.set_fuse_unk(false);
            assert_eq!(
                model.encode("xyz東京").unwrap(),
                vec!["x", "y", "z", "東", "京"]
            );
            model.set_fuse_unk(true);
            assert_eq!(model.encode("xyz東京").unwrap(), vec!["xyz東京"]);

            // User encoded in original version
            assert_eq!(model.encode("ABC").unwrap(), vec!["ABC"]);
            assert_eq!(model.encode("abABCcd").unwrap(), vec!["ab", "ABC", "cd"]);
            assert_eq!(
                model.encode("ababcdabcdcd").unwrap(),
                vec!["ab", "abcdabcd", "cd"]
            );
            assert_eq!(model.encode("abqrcd").unwrap(), vec!["ab", "q", "r", "cd"]);
        }
    }

    #[test]
    fn test_unigram_bytefallback() {
        // In [97]: processor.encode_as_pieces("⅐⅛⅑ ")
        // Out[97]: ['▁', '<0xE2>', '<0x85>', '<0x90>', '⅛', '<0xE2>', '<0x85>', '<0x91>', '▁']
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("<0xC3>".to_string(), -0.01),
            ("<0xA9>".to_string(), -0.03),
        ];
        let unigram = Unigram::from(sentencepieces, Some(0), true).unwrap();
        let tokens: Vec<Token> = unigram.tokenize("é").unwrap();
        assert_eq!(
            tokens,
            [
                Token {
                    id: 1,
                    value: "<0xC3>".to_string(),
                    offsets: (0, 2)
                },
                Token {
                    id: 2,
                    value: "<0xA9>".to_string(),
                    offsets: (0, 2)
                }
            ]
        );

        let tokens = unigram.tokenize("?é").unwrap();
        assert_eq!(tokens[0].id, 0);
    }
}

pub(super) fn index_vocabulary(
    vocab: &Vocab,
    index: &mut TokenMap,
    trie: &mut Trie<u8>,
    pending: &mut String,
    mut reserve: impl FnMut(
        &mut String,
        usize,
    ) -> std::result::Result<(), std::collections::TryReserveError>,
) -> std::result::Result<(), std::collections::TryReserveError> {
    for (id, (token, _)) in vocab.iter().enumerate() {
        reserve(pending, token.len())?;
        pending.push_str(token);
        index.insert(std::mem::take(pending), id as u32);
        trie.push(token.as_bytes());
    }
    Ok(())
}
impl Unigram {
    pub(crate) fn same_configuration(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id
            && self.bos_id == other.bos_id
            && self.eos_id == other.eos_id
            && self.fuse_unk == other.fuse_unk
            && self.is_optimized == other.is_optimized
            && self.byte_fallback == other.byte_fallback
            && self.min_score.to_bits() == other.min_score.to_bits()
            && self.alpha.map(f64::to_bits) == other.alpha.map(f64::to_bits)
            && self.nbest_size == other.nbest_size
            && self.token_to_ids == other.token_to_ids
            && self.vocab.len() == other.vocab.len()
            && self
                .vocab
                .iter()
                .zip(&other.vocab)
                .all(|((a, x), (b, y))| a == b && x.to_bits() == y.to_bits())
    }
    pub(crate) fn configuration_comparison_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<[&Self; 2]>(),
            size_of::<[usize; 2]>(),
            size_of::<[u64; 2]>(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'_, (String, f64)>,
                    std::slice::Iter<'_, (String, f64)>,
                >,
            >(),
            size_of::<hashbrown::hash_map::Iter<'_, String, u32>>(),
            size_of::<(&String, &u32)>(),
            size_of::<bool>(),
        ]
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
    }
}
