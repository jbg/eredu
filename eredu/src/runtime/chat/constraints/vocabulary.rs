//! One immutable allocation for token offsets and decoded token bytes.

use super::ConstraintError;
use eredu_core::{BackendFailure, ModelRuntime, SharedControllerBytes, TextGenerationBackend};
use llguidance::toktrie::{TokEnv, TokTrie};

#[derive(Clone, Copy)]
pub(super) struct VocabularyLayout {
    tokens: usize,
    table_bytes: usize,
    total_bytes: usize,
    max_token_bytes: usize,
}

/// An immutable trie borrow and checked metadata. Preflight allocates no token
/// payload and never creates or advances a grammar state.
pub(super) struct VocabularyPlan {
    environment: TokEnv,
    layout: VocabularyLayout,
}

impl VocabularyPlan {
    pub(super) fn new(environment: TokEnv) -> Result<Self, ConstraintError> {
        let trie = environment.tok_trie();
        let tokens = trie.vocab_size();
        let overflow =
            || ConstraintError::new("packed token vocabulary exceeds host storage limits");
        let table_bytes = tokens
            .checked_add(1)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(overflow)?;
        let mut total_bytes = table_bytes;
        let mut max_token_bytes = 0;
        for token in 0..tokens {
            let token = u32::try_from(token).map_err(|_| overflow())?;
            let len = token_bytes(trie, token).len();
            total_bytes = total_bytes.checked_add(len).ok_or_else(overflow)?;
            max_token_bytes = max_token_bytes.max(len);
        }
        // Vec and the portable offset representation must both hold the entire
        // allocation. These checks precede the allocating backend factory.
        u64::try_from(total_bytes).map_err(|_| overflow())?;
        if total_bytes > isize::MAX as usize {
            return Err(overflow());
        }
        Ok(Self {
            environment,
            layout: VocabularyLayout {
                tokens,
                table_bytes,
                total_bytes,
                max_token_bytes,
            },
        })
    }

    pub(super) fn prepare<B: TextGenerationBackend>(
        self,
        runtime: &ModelRuntime<B>,
    ) -> Result<SharedTokenVocabulary, BackendFailure> {
        let bytes = B::prepare_shared_controller_bytes(runtime, || self.pack());
        bytes.map(|bytes| SharedTokenVocabulary {
            bytes,
            layout: self.layout,
        })
    }

    /// Standalone semantic fixtures have no managed runtime/domain. Production
    /// setup always uses `prepare` and its backend-owned construction authority.
    #[cfg(test)]
    pub(super) fn unregistered(self) -> SharedTokenVocabulary {
        SharedTokenVocabulary {
            bytes: SharedControllerBytes::new(self.pack()),
            layout: self.layout,
        }
    }

    fn pack(&self) -> Vec<u8> {
        let trie = self.environment.tok_trie();
        let mut bytes = Vec::with_capacity(self.layout.total_bytes);
        let mut offset = self.layout.table_bytes;
        bytes.extend_from_slice(&(offset as u64).to_le_bytes());
        for token in 0..self.layout.tokens {
            offset += token_bytes(trie, token as u32).len();
            bytes.extend_from_slice(&(offset as u64).to_le_bytes());
        }
        for token in 0..self.layout.tokens {
            bytes.extend_from_slice(token_bytes(trie, token as u32));
        }
        debug_assert_eq!(bytes.len(), self.layout.total_bytes);
        bytes
    }
}

pub(super) fn token_bytes(trie: &TokTrie, token: u32) -> &[u8] {
    let bytes = trie.token(token);
    // Structural markers are trie metadata, not decoded output. Absent token
    // slots remain empty, preserving the previous nested vocabulary semantics.
    bytes
        .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
        .unwrap_or(bytes)
}

#[derive(Clone)]
pub(super) struct SharedTokenVocabulary {
    bytes: SharedControllerBytes,
    layout: VocabularyLayout,
}

impl SharedTokenVocabulary {
    pub(super) fn len(&self) -> usize {
        self.layout.tokens
    }

    pub(super) fn get(&self, token: usize) -> Option<&[u8]> {
        self.layout.get(self.bytes.as_ref(), token)
    }

    /// Exact immutable source; this loan grants no accounting credit.
    pub(super) fn source(&self) -> (VocabularyLayout, &[u8]) {
        (self.layout, self.bytes.as_ref())
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        (0..self.len()).map(|token| {
            self.get(token)
                .expect("checked immutable vocabulary offsets")
        })
    }
}

impl VocabularyLayout {
    pub(super) fn len(self) -> usize {
        self.tokens
    }
    pub(super) fn maximum(self) -> usize {
        self.max_token_bytes
    }
    pub(super) fn get(self, bytes: &[u8], token: usize) -> Option<&[u8]> {
        eredu_core::speculative::packed_controller_token_bytes(
            bytes,
            self.tokens,
            self.max_token_bytes,
            token,
        )
    }
}

#[cfg(test)]
mod tests;
