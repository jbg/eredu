use super::MlxPoolingAttentionCache;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::cache::kv::{ConcatKeyValueCache, LiveKeyValueCache, PoolingCache};
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

/// Borrowed native cache-representation dispatch. Every variant preserves the
/// exact resident local cache and optional pooling streams; no host container
/// or native descriptor is allocated during preparation. The caller retains
/// exclusive settled source access and custody through completion/recovery.
/// This component does not admit allocation or prove a whole state copy.
pub(crate) struct PreparedPoolingAttentionCopy<'a> {
    storage: PoolingCopyStorage<'a>,
}

enum PoolingCopyStorage<'a> {
    Local(&'a ConcatKeyValueCache),
    Compressed {
        local: &'a ConcatKeyValueCache,
        pool: &'a PoolingCache,
    },
    Sparse {
        local: &'a ConcatKeyValueCache,
        pool: &'a PoolingCache,
        index_pool: &'a PoolingCache,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("paged pooling local state requires an independent manager-copy mechanism")]
pub(crate) struct PagedPoolingCopy;

impl<'a> PreparedPoolingAttentionCopy<'a> {
    pub(crate) fn preparation_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        size_of::<Self>()
            .checked_add(size_of::<Result<Self, PagedPoolingCopy>>())?
            .checked_add(size_of::<&MlxPoolingAttentionCache>())?
            .checked_add(size_of::<&ConcatKeyValueCache>())?
            .checked_add(size_of::<PoolingCopyStorage<'a>>())?
            .checked_add(size_of::<&LiveKeyValueCache>())?
            .checked_add(size_of::<[&PoolingCache; 2]>())
    }

    pub(super) fn new(source: &'a MlxPoolingAttentionCache) -> Result<Self, Exception> {
        Self::new_fixed(source).map_err(Exception::from_source)
    }

    pub(super) fn new_fixed(
        source: &'a MlxPoolingAttentionCache,
    ) -> Result<Self, PagedPoolingCopy> {
        let local = match source.local() {
            LiveKeyValueCache::Resident(local) => local,
            LiveKeyValueCache::Paged(_) => {
                return Err(PagedPoolingCopy);
            }
        };
        let storage = match source {
            MlxPoolingAttentionCache::Local(_) => PoolingCopyStorage::Local(local),
            MlxPoolingAttentionCache::Compressed { pool, .. } => {
                PoolingCopyStorage::Compressed { local, pool }
            }
            MlxPoolingAttentionCache::Sparse {
                pool, index_pool, ..
            } => PoolingCopyStorage::Sparse {
                local,
                pool,
                index_pool,
            },
        };
        Ok(Self { storage })
    }

    /// Visits local key/value operands first, then the primary and index
    /// pooling streams in each leaf's stable five-slot order. Source aliases
    /// are not deduplicated because each slot requires its own destination.
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        match &self.storage {
            PoolingCopyStorage::Local(local) => {
                local.prepare_isolated_copy().visit_operands(visitor)
            }
            PoolingCopyStorage::Compressed { local, pool } => {
                local.prepare_isolated_copy().visit_operands(visitor);
                pool.prepare_isolated_copy().visit_operands(visitor);
            }
            PoolingCopyStorage::Sparse {
                local,
                pool,
                index_pool,
            } => {
                local.prepare_isolated_copy().visit_operands(visitor);
                pool.prepare_isolated_copy().visit_operands(visitor);
                index_pool.prepare_isolated_copy().visit_operands(visitor);
            }
        }
    }

    /// The existing copy worker is host-only exactly when every leaf is empty.
    /// This query preserves source policy/scalars and never resets populated state.
    pub(crate) fn empty_copy_control_bytes(&self) -> Option<usize> {
        let mut populated = false;
        self.visit_operands(&mut |_| populated = true);
        (!populated)
            .then(|| self.copy_control_bytes::<Exception>())
            .flatten()
    }

    pub(crate) fn copy_control_bytes<E>(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<&Self>(),
            size_of::<Result<Self, PagedPoolingCopy>>(),
            size_of::<MlxPoolingAttentionCache>(),
            size_of::<Result<MlxPoolingAttentionCache, E>>(),
            size_of::<Result<LiveKeyValueCache, E>>(),
            size_of::<Option<&RefCell<Vec<Array>>>>(),
            size_of::<&Stream>(),
            size_of::<Option<usize>>(),
            size_of::<&mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>>(),
            size_of::<(
                &ConcatKeyValueCache,
                Option<&PoolingCache>,
                Option<&PoolingCache>,
            )>(),
        ];
        let mut bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?;
        let (local, pool, index) = match &self.storage {
            PoolingCopyStorage::Local(local) => (*local, None, None),
            PoolingCopyStorage::Compressed { local, pool } => (*local, Some(*pool), None),
            PoolingCopyStorage::Sparse {
                local,
                pool,
                index_pool,
            } => (*local, Some(*pool), Some(*index_pool)),
        };
        bytes = bytes.checked_add(local.prepare_isolated_copy().control_bytes::<E>()?)?;
        for pool in [pool, index].into_iter().flatten() {
            bytes = bytes.checked_add(pool.prepare_isolated_copy().control_bytes::<E>()?)?;
        }
        Some(bytes)
    }

    pub(crate) fn copy(self, stream: &Stream) -> Result<MlxPoolingAttentionCache, Exception> {
        self.copy_with(&mut |copy| copy.copy(stream))
    }

    /// Earlier completed leaves remain in the caller's recovery collector if
    /// a later copy fails. The collector is not a publication inventory.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxPoolingAttentionCache, Exception> {
        self.copy_with(&mut |copy| copy.copy_retained(stream, roots))
    }

    pub(crate) fn copy_with<E>(
        self,
        copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
    ) -> Result<MlxPoolingAttentionCache, E> {
        fn local<'a, E>(
            source: &'a ConcatKeyValueCache,
            copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
        ) -> Result<LiveKeyValueCache, E> {
            source
                .prepare_isolated_copy()
                .copy_with(copy)
                .map(LiveKeyValueCache::Resident)
        }
        fn pool<'a, E>(
            source: &'a PoolingCache,
            copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
        ) -> Result<PoolingCache, E> {
            source.prepare_isolated_copy().copy_with(copy)
        }
        Ok(match self.storage {
            PoolingCopyStorage::Local(source) => {
                MlxPoolingAttentionCache::Local(local(source, copy)?)
            }
            PoolingCopyStorage::Compressed {
                local: source,
                pool: primary,
            } => MlxPoolingAttentionCache::Compressed {
                local: local(source, copy)?,
                pool: pool(primary, copy)?,
            },
            PoolingCopyStorage::Sparse {
                local: source,
                pool: primary,
                index_pool,
            } => MlxPoolingAttentionCache::Sparse {
                local: local(source, copy)?,
                pool: pool(primary, copy)?,
                index_pool: pool(index_pool, copy)?,
            },
        })
    }
}

#[cfg(test)]
mod tests;
