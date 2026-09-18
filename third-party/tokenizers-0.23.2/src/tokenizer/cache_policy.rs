//! Constructor policy for model-local tokenization caches, not allocation admission.
use super::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer,
};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Deserializer, Serialize};

/// Selects model-cache construction before BPE/Unigram are built.
/// Regex/global state and tokenization itself may still allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCachePolicy {
    /// Maximum entries in each model cache. Zero constructs no cache.
    /// This limits entry count, not allocator bytes or process-wide memory.
    pub capacity: usize,
}
impl Default for ModelCachePolicy {
    fn default() -> Self {
        Self {
            capacity: crate::utils::cache::DEFAULT_CACHE_CAPACITY,
        }
    }
}
impl ModelCachePolicy {
    pub const fn disabled() -> Self {
        Self { capacity: 0 }
    }
}

/// Concrete tokenizer deserialization with an explicit model-cache policy.
/// This uses the ordinary tokenizer visitor and is not a bounded parser.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerSeed(pub ModelCachePolicy);

impl<'de> DeserializeSeed<'de> for TokenizerSeed {
    type Value = Tokenizer;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Tokenizer, D::Error> {
        super::serialization::deserialize_with_model_seed::<
            D,
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
            _,
        >(deserializer, crate::models::ModelSeed(self.0))
        .map(Tokenizer)
    }
}

impl Tokenizer {
    /// Reads and constructs with the selected model-cache policy. File input and
    /// deserialization still allocate; this establishes no residency bound.
    pub fn from_file_with_cache_policy<P: AsRef<std::path::Path>>(
        file: P,
        policy: ModelCachePolicy,
    ) -> super::Result<Self> {
        let content = std::fs::read_to_string(file)?;
        Self::from_bytes_with_cache_policy(content.as_bytes(), policy)
    }

    /// Constructs from JSON with policy selected before model construction.
    pub fn from_bytes_with_cache_policy<P: AsRef<[u8]>>(
        bytes: P,
        policy: ModelCachePolicy,
    ) -> super::Result<Self> {
        let mut deserializer = serde_json::Deserializer::from_slice(bytes.as_ref());
        let tokenizer = TokenizerSeed(policy).deserialize(&mut deserializer)?;
        deserializer.end()?;
        Ok(tokenizer)
    }

    /// Reports current model-cache capacity, not construction provenance or
    /// allocation authority. Thread-local storage from retired cache generations
    /// is outside this model-local configuration report. Models without a cache report zero capacity.
    /// HF serialization omits this choice; reconstruction must supply it again.
    pub fn model_cache_policy(&self) -> ModelCachePolicy {
        let capacity = match self.get_model() {
            ModelWrapper::BPE(model) => model.cache_capacity(),
            ModelWrapper::Unigram(model) => model.cache_capacity(),
            _ => 0,
        };
        ModelCachePolicy { capacity }
    }
}

#[cfg(test)]
#[path = "cache_policy_tests.rs"]
mod tests;
