//! MLX collective adapter for backend-neutral scheduler consensus.

use safemlx::{
    distributed::{self, Group},
    error::Exception,
    transforms::async_eval_with_event,
    Array, Stream,
};
use safemlx_lm_core::consensus::ConsensusTransport;

/// MLX transport failure before conversion into a neutral consensus error.
#[derive(Debug, thiserror::Error)]
pub enum MlxConsensusError {
    /// Portable words cannot be represented as one MLX vector dimension.
    #[error("metadata exceeds i32")]
    MetadataOverflow,
    /// MLX collective or exact-completion failure.
    #[error(transparent)]
    Runtime(#[from] Exception),
}

/// Topology-scoped MLX implementation of scheduler metadata consensus.
pub struct MlxConsensusTransport<'a> {
    group: &'a Group,
    stream: &'a Stream,
}

impl<'a> MlxConsensusTransport<'a> {
    /// Binds consensus to one distributed group and execution stream.
    pub const fn new(group: &'a Group, stream: &'a Stream) -> Self {
        Self { group, stream }
    }
}

impl ConsensusTransport for MlxConsensusTransport<'_> {
    type Error = MlxConsensusError;

    fn participant_count(&self) -> usize {
        self.group.size()
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let length = i32::try_from(local.len()).map_err(|_| MlxConsensusError::MetadataOverflow)?;
        let local = Array::from_slice(local, &[length]);
        let gathered = distributed::all_gather(&local, self.group, self.stream)?;
        async_eval_with_event([&gathered])?.synchronize()?;
        Ok(gathered.evaluated()?.as_slice::<u32>().to_vec())
    }
}
