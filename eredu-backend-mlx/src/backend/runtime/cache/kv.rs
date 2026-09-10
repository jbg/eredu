use safemlx::{
    error::Exception,
    ops::{
        broadcast_to, concatenate_axis,
        indexing::{TryIndexMutOp, TryIndexOp},
        matmul, maximum, r#where, sum_axis, zeros_dtype,
    },
    Array, Dtype, Stream,
};

use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, Error as ComputeError,
};

use crate::{
    backend::nn::shared::MlxNeuralBackend,
    backend::runtime::cache::residency::{CacheBlockArrays, CacheResidencyManager},
};
use eredu_core::cache::{CacheBlockId, CacheRankIdentity, CacheRepresentation};
use eredu_runtime::{CacheResidencyReport, PagedCacheOptions};
use ref_cast::RefCast;

use crate::MlxTensor;

pub type RetainedArrayIter<'a> = std::iter::Map<
    std::iter::Chain<std::option::Iter<'a, Array>, std::option::Iter<'a, Array>>,
    fn(&'a Array) -> &'a MlxTensor,
>;
type RetainedArrayVecIter<'a> =
    std::iter::Map<std::vec::IntoIter<&'a Array>, fn(&'a Array) -> &'a MlxTensor>;

fn retained_tensor(array: &Array) -> &MlxTensor {
    MlxTensor::ref_cast(array)
}

/// A per-layer attention key/value cache.
pub trait KeyValueCache {
    /// Returns the current sequence offset represented by the cache.
    fn offset(&self) -> i32;

    /// Returns the maximum retained sequence length for sliding-window caches.
    fn max_size(&self) -> Option<i32>;

    /// Returns retained cache arrays that must be materialized before weights
    /// used to produce them can be released.
    fn retained_arrays(&self) -> Vec<&Array> {
        Vec::new()
    }

    /// Returns whether attention must consume ordered cache blocks directly.
    fn is_paged(&self) -> bool {
        false
    }

    /// Runs exact attention from ordered cache blocks when this is a paged cache.
    ///
    /// Ordinary caches return `None` and continue through the existing
    /// contiguous attention kernel.
    fn paged_attention(
        &mut self,
        _queries: &Array,
        _scale: f32,
        _mask: Option<&Array>,
        _sinks: Option<&Array>,
        _softcap: Option<f32>,
        _arithmetic: eredu_nn::AttentionArithmetic,
        _stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        Ok(None)
    }

    /// Adds keys and values for an immediate attention operation.
    ///
    /// Paged implementations return only the submitted arrays because the
    /// subsequent attention call scans history blockwise. Ordinary caches use
    /// the contiguous history returned by [`KeyValueCache::update_and_fetch`].
    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        self.update_and_fetch(keys, values, stream)
    }

    /// Adds the newest keys and values and returns the full keys and values to attend over.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception>;
}

mod pooling;
#[allow(unused_imports)]
pub use pooling::{PoolingCache, PoolingCacheState, PoolingWindows};

mod compressed;
pub use compressed::CompressedLatentCache;

mod paged;
pub use paged::{
    LiveKeyValueCache, PagedKeyValueCache, PagedKeyValueTransactionCheckpoint,
    PagedLatentAttentionBlock,
};

mod attention;
#[cfg(test)]
use attention::absolute_attention_mask;
pub use attention::{BlockwiseAttentionAccumulator, KeyValueAttentionBlock};

fn cache_residency_exception(error: impl std::fmt::Display) -> Exception {
    Exception::custom(error.to_string())
}

mod contiguous;
pub use contiguous::ConcatKeyValueCache;

#[cfg(test)]
#[path = "kv/tests.rs"]
mod tests;
