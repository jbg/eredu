//! Model cache controls using the public Hugging Face tokenizer API.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Requested entry limit for BPE and Unigram tokenization caches.
///
/// BPE caches are per model and per thread. This limit is not a byte limit or a
/// bound on process memory. Upstream may retain thread-local allocations from
/// retired BPE cache generations. Other models have no model cache to configure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCachePolicy {
    /// Maximum entries per model cache; zero prevents new cached entries.
    pub capacity: usize,
}

impl Default for ModelCachePolicy {
    fn default() -> Self {
        Self { capacity: 10_000 }
    }
}

impl ModelCachePolicy {
    /// Prevents new model-cache entries during tokenization.
    pub const fn disabled() -> Self {
        Self { capacity: 0 }
    }

    /// Invalidates existing entries and applies the entry limit through the
    /// upstream model controls. This does not reclaim retired BPE generations
    /// from other threads, or enable a BPE cache constructed without one.
    /// The public tokenizer API requires cloning and replacing the model to
    /// change its cache; both model copies coexist during this operation.
    pub fn apply(self, tokenizer: &mut tokenizers::Tokenizer) {
        if !matches!(
            tokenizer.get_model(),
            tokenizers::models::ModelWrapper::BPE(_) | tokenizers::models::ModelWrapper::Unigram(_)
        ) {
            return;
        }
        let mut model = tokenizer.get_model().clone();
        model.clear_cache();
        model.resize_cache(self.capacity);
        tokenizer.with_model(model);
    }

    /// Deserializes a tokenizer and sets its cache limit before returning it.
    /// JSON parsing and upstream construction use their ordinary allocations;
    /// this policy does not bound that transient memory.
    pub fn from_bytes(self, bytes: impl AsRef<[u8]>) -> tokenizers::Result<tokenizers::Tokenizer> {
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes)?;
        // Stock deserialization creates the default cache. Replacing that
        // newly constructed model is only needed for a different entry limit.
        if self != Self::default() {
            self.apply(&mut tokenizer);
        }
        Ok(tokenizer)
    }

    /// Reads tokenizer JSON and applies the cache limit before first use.
    pub fn from_file(self, path: impl AsRef<Path>) -> tokenizers::Result<tokenizers::Tokenizer> {
        self.from_bytes(std::fs::read(path)?)
    }
}
