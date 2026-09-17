//! Explicit history growth makes retained and replacement payloads priceable.

#[derive(Debug, Default)]
pub(super) struct TokenHistory {
    storage: Box<[u32]>,
    len: usize,
}

impl Clone for TokenHistory {
    fn clone(&self) -> Self {
        self.prepare_copy()
            .expect("existing token history has a representable payload extent")
            .copy()
    }
}

/// Borrows the actual fixed-size box, including unused history slots.
#[derive(Debug)]
pub(super) struct TokenHistoryCopy<'a> {
    source: &'a TokenHistory,
    bytes: u64,
}

impl TokenHistoryCopy<'_> {
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(super) fn copy(self) -> TokenHistory {
        #[cfg(test)]
        PAYLOAD_COPIES.with(|count| count.set(count.get() + 1));
        TokenHistory {
            storage: self.source.storage.clone(),
            len: self.source.len,
        }
    }
}

impl TokenHistory {
    pub(super) fn prepare_copy(&self) -> Option<TokenHistoryCopy<'_>> {
        Some(TokenHistoryCopy {
            source: self,
            bytes: copy_payload_bytes(self.capacity())?,
        })
    }

    pub(super) fn from_tokens(tokens: impl IntoIterator<Item = u32>) -> Self {
        let storage = tokens.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let len = storage.len();
        Self { storage, len }
    }

    pub(super) fn as_slice(&self) -> &[u32] {
        &self.storage[..self.len]
    }

    pub(super) fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Exact additional destination used by the same push worker, if full.
    pub(super) fn push_payload_bytes(&self) -> Option<u64> {
        if self.len < self.capacity() { Some(0) }
        else { copy_payload_bytes(next_history_capacity(self.capacity())?) }
    }

    pub(super) fn push(&mut self, token: u32) {
        if self.len == self.capacity() {
            let capacity = next_history_capacity(self.capacity())
                .expect("accepted-token history capacity overflow");
            let mut next = vec![0; capacity].into_boxed_slice();
            next[..self.len].copy_from_slice(self.as_slice());
            self.storage = next;
        }
        self.storage[self.len] = token;
        self.len += 1;
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
    }
}

fn copy_payload_bytes(capacity: usize) -> Option<u64> {
    let bytes = capacity.checked_mul(std::mem::size_of::<u32>())?;
    if bytes > isize::MAX as usize {
        return None;
    }
    u64::try_from(bytes).ok()
}

pub(crate) fn next_history_capacity(capacity: usize) -> Option<usize> {
    capacity.checked_mul(2).map(|next| next.max(4))
}

#[cfg(test)]
mod copy_tests {
    use super::*;

    #[test]
    fn copy_payload_checks_the_host_allocation_extent_without_allocating() {
        let maximum = isize::MAX as usize / std::mem::size_of::<u32>();
        assert_eq!(copy_payload_bytes(0), Some(0));
        assert_eq!(copy_payload_bytes(8), Some(32));
        assert_eq!(copy_payload_bytes(maximum), Some((maximum as u64) * 4));
        assert_eq!(copy_payload_bytes(maximum + 1), None);
        assert_eq!(copy_payload_bytes(usize::MAX), None);
    }
}

// Counts the actual payload-copy worker, not descriptor clones. Thread-local
// storage keeps parallel conformance tests independent without allocator hooks.
#[cfg(test)]
thread_local! {
    static PAYLOAD_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn payload_copy_count() -> usize {
    PAYLOAD_COPIES.with(std::cell::Cell::get)
}
