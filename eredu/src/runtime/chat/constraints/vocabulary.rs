//! Decoded output bytes from the canonical token trie.
use llguidance::toktrie::TokTrie;

pub(super) fn token_bytes(trie: &TokTrie, token: u32) -> &[u8] {
    let bytes = trie.token(token);
    // Structural markers are trie metadata, not decoded output. Absent token
    // slots remain empty, preserving the previous nested vocabulary semantics.
    bytes
        .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
        .unwrap_or(bytes)
}

#[cfg(test)]
mod tests;
