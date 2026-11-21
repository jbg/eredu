//! Attention storage and lifecycle shared by heterogeneous model states.

use eredu_core::cache::mechanisms::{
    CompressedCacheStorage, PagedKeyValueStorage, ResidentKeyValueStorage,
};
use eredu_nn::AttentionCache;

use super::key_value::{invalid_state, KeyValueLayerState, KeyValueStateError};
use crate::cache::{live::CacheResidencyManager, CompressedLatentCache};

/// Attention storage associated with one mixed attention and fixed-state layer.
#[derive(Debug, Clone)]
pub enum HybridAttentionState<
    M: ResidentKeyValueStorage + PagedKeyValueStorage + CompressedCacheStorage,
> {
    /// Ordinary or key-only key/value history.
    KeyValue(KeyValueLayerState<M>),
    /// Latent and rotary-key history.
    Compressed(CompressedLatentCache<M>),
}

impl<M: ResidentKeyValueStorage + PagedKeyValueStorage + CompressedCacheStorage>
    HybridAttentionState<M>
{
    /// Checkpoints the current frontier for rollback in the same paging session.
    pub fn deep_clone_state(&self) -> Result<Self, KeyValueStateError> {
        match self {
            Self::KeyValue(cache) => cache.deep_clone_state().map(Self::KeyValue),
            Self::Compressed(cache) => Ok(Self::Compressed(cache.deep_clone_state()?)),
        }
    }

    /// Returns the current token frontier.
    pub fn offset(&self) -> i32 {
        match self {
            Self::KeyValue(cache) => cache.offset(),
            Self::Compressed(cache) => cache.offset(),
        }
    }

    /// Clears this layer's retained attention state.
    pub fn clear(&mut self) -> Result<(), KeyValueStateError> {
        match self {
            Self::KeyValue(cache) => cache.clear(),
            Self::Compressed(cache) => Ok(cache.clear()?),
        }
    }

    /// Borrows the roots required by a native submission.
    pub fn retained_values(&self) -> Vec<&M::Value> {
        match self {
            Self::KeyValue(cache) => cache.retained_values().collect(),
            Self::Compressed(cache) => cache
                .retained_arrays()
                .into_iter()
                .map(M::borrow_value)
                .collect(),
        }
    }

    /// Borrows the page catalog of a block-addressable attention cache.
    pub fn manager(&self) -> Option<&CacheResidencyManager<M>> {
        match self {
            Self::KeyValue(KeyValueLayerState::Paged(cache)) => Some(cache.manager()),
            Self::Compressed(cache) => cache.residency_manager(),
            Self::KeyValue(_) => None,
        }
    }

    /// Rebinds paged attention after the shared catalog has been forked.
    pub fn rebind_paging_manager(&mut self, manager: CacheResidencyManager<M>) {
        match self {
            Self::KeyValue(KeyValueLayerState::Paged(cache)) => {
                cache.rebind_paging_manager(manager)
            }
            Self::Compressed(cache) => cache.rebind_paging_manager(manager),
            Self::KeyValue(_) => {}
        }
    }

    /// Copies a layer into the already forked catalog. A transaction candidate
    /// shares immutable resident key/value roots unless isolation was requested.
    pub fn copy_candidate(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        queue: &M::Queue,
        isolated: bool,
    ) -> Result<Self, KeyValueStateError> {
        match self {
            Self::KeyValue(cache) => cache
                .copy_candidate(manager, queue, isolated)
                .map(Self::KeyValue),
            Self::Compressed(cache) => {
                let mut copy = cache.isolated_snapshot(queue)?;
                if copy.is_paged() {
                    let manager = manager.ok_or_else(|| {
                        invalid_state("copied paged attention requires a paging catalog")
                    })?;
                    copy.rebind_paging_manager(manager.clone());
                }
                Ok(Self::Compressed(copy))
            }
        }
    }

    /// Seals the current paged tail before prompt-cache publication.
    pub fn finalize(&mut self) -> Result<(), KeyValueStateError> {
        match self {
            Self::KeyValue(KeyValueLayerState::Paged(cache)) => Ok(cache.finalize()?),
            Self::Compressed(cache) if cache.is_paged() => Ok(cache.finalize()?),
            _ => Err(invalid_state(
                "prompt-cache persistence requires paged attention state",
            )),
        }
    }

    /// Restores a checkpoint made with the same retained attention representation.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        queue: &M::Queue,
    ) -> Result<(), KeyValueStateError> {
        match (self, checkpoint) {
            (Self::KeyValue(current), Self::KeyValue(previous)) => {
                current.restore_checkpoint(previous, queue)
            }
            (Self::Compressed(current), Self::Compressed(previous)) => {
                Ok(current.restore_checkpoint(previous, queue)?)
            }
            _ => Err(invalid_state(
                "hybrid attention checkpoint representation changed",
            )),
        }
    }
}
