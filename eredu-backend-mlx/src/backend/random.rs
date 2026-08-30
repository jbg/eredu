//! Stateful random-key ownership for MLX backend sessions.

use safemlx::{error::Result, ops::indexing::TryIndexOp, random, Array, Stream};

pub(crate) fn split_key_at(key: &Array, index: usize, stream: &Stream) -> Result<Array> {
    let count = index
        .checked_add(1)
        .and_then(|count| i32::try_from(count).ok())
        .ok_or_else(|| safemlx::error::Exception::custom("random subkey index exceeds i32"))?;
    let keys = safemlx::random::split_n(key, count, stream)?;
    keys.try_index_device(index as i32, stream)
}

/// A backend-owned MLX PRNG key that advances through native key splitting.
#[derive(Debug, Clone)]
pub struct RandomState {
    state: Array,
}

impl RandomState {
    /// Creates reproducible state from a seed.
    pub fn with_seed(seed: u64) -> Result<Self> {
        Ok(Self {
            state: random::key(seed)?,
        })
    }

    /// Takes ownership of an existing native PRNG key.
    pub fn from_key(key: Array) -> Self {
        Self { state: key }
    }

    /// Advances the state and returns the next key.
    pub fn next_key(&mut self, stream: &Stream) -> Result<Array> {
        let next = random::split_n(&self.state, 2, stream)?;
        self.state = next.try_index_device(0, stream)?;
        next.try_index_device(1, stream)
    }

    /// Replaces the current state from a seed.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.state = random::key(seed)?;
        Ok(())
    }

    /// Borrows the current native key.
    pub fn as_array(&self) -> &Array {
        &self.state
    }

    /// Mutably borrows the current native key.
    pub fn as_array_mut(&mut self) -> &mut Array {
        &mut self.state
    }
}
