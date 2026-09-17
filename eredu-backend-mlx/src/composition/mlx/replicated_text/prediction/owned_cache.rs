//! Prediction-cache payloads and the authorities retained by their descendants.

use crate::{
    backend::{
        managed_memory::{NativeMemoryOwner, NativeMemoryRetention},
        nn::shared::MlxNeuralBackend,
    },
    MlxTensor,
};
use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, Error, PoolingAttentionCache,
    PoolingOverlap, PoolingWindows,
};
use eredu_runtime::RuntimeLayerState;
use safemlx::Stream;

/// Opaque cache ownership across speculative copies, checkpoints, and rollback.
/// Native completion and allocation attachments remain the enclosing worker's
/// responsibility; this wrapper does not independently authorize native work.
#[derive(Debug)]
pub(crate) struct OwnedPredictionCache<C> {
    // Payload retirement must precede releasing its allocation authority.
    value: C,
    memory: NativeMemoryRetention,
}

impl<C> OwnedPredictionCache<C> {
    pub(crate) fn new(value: C, memory: NativeMemoryRetention) -> Self {
        Self { value, memory }
    }

    /// Borrows cache-specific inspection and copying mechanisms.
    pub(crate) fn inner(&self) -> &C {
        &self.value
    }

    pub(crate) fn memory(&self) -> &NativeMemoryRetention {
        &self.memory
    }

    /// Retain this before copying, including in detached native recovery.
    /// The new authority covers independent allocations; source authorities
    /// remain necessary for shared backing and failure paths.
    pub(crate) fn memory_for_copy(&self, owner: &NativeMemoryOwner) -> NativeMemoryRetention {
        self.memory().with_added_owner(owner)
    }

    /// Current out-of-line retention storage, excluding the inline wrapper.
    /// An independent copy must additionally price its new authority handle.
    #[cfg(test)]
    pub(crate) fn ownership_metadata_bytes(&self) -> Option<u64> {
        self.memory.logical_metadata_bytes()
    }

    /// Out-of-line retention capacity for `memory_for_copy`, including the new
    /// owner. The enclosing estimator separately includes the wrapper itself.
    pub(crate) fn copy_ownership_metadata_bytes(&self) -> Option<u64> {
        self.memory.metadata_bytes_with_added_owner()
    }
}

impl<C: Clone> Clone for OwnedPredictionCache<C> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            memory: self.memory.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        // Payload clone_from can partially replace state before unwinding.
        // Keep both old and incoming authorities before invoking that code.
        self.memory.extend_from(&source.memory);
        self.value.clone_from(&source.value);
    }
}

impl<C: RuntimeLayerState<MlxNeuralBackend>> RuntimeLayerState<MlxNeuralBackend>
    for OwnedPredictionCache<C>
{
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.value.retained_values()
    }
}

impl<C: CompressedAttentionCache<MlxTensor>> CompressedAttentionCache<MlxTensor>
    for OwnedPredictionCache<C>
{
    type Checkpoint = OwnedPredictionCache<C::Checkpoint>;

    fn offset(&self) -> i32 {
        self.value.offset()
    }

    fn is_paged(&self) -> bool {
        self.value.is_paged()
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<MlxTensor>,
        context: &Stream,
    ) -> Result<CompressedAttentionView<MlxTensor>, Error> {
        self.value.append(state, context)
    }

    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &Stream,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<MlxTensor>) -> Result<u64, Error>,
    {
        self.value.visit_blocks(query_tokens, context, visitor)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        OwnedPredictionCache::new(self.value.checkpoint(), self.memory.clone())
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, context: &Stream) -> Result<(), Error> {
        // Restore may mutate native state and then fail; neither side can be
        // refunded until the resulting installed state and recovery retire.
        self.memory.extend_from(&checkpoint.memory);
        self.value.restore(&checkpoint.value, context)
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.value.finalize()
    }

    fn clear(&mut self) -> Result<(), Error> {
        // Clearing logical contents does not prove all backing/aliases retired.
        self.value.clear()
    }
}

impl<C: PoolingAttentionCache<MlxTensor>> PoolingAttentionCache<MlxTensor>
    for OwnedPredictionCache<C>
{
    type Checkpoint = OwnedPredictionCache<C::Checkpoint>;

    fn offset(&self) -> i32 {
        self.value.offset()
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.value.pooling_ratio(stream)
    }

    fn append_local(&mut self, keys: MlxTensor, context: &Stream) -> Result<MlxTensor, Error> {
        self.value.append_local(keys, context)
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        self.value.local_mask(query_tokens, offset, context)
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: MlxTensor,
        gates: MlxTensor,
        absolute_offset: i32,
        context: &Stream,
    ) -> Result<PoolingWindows<MlxTensor>, Error> {
        self.value
            .accumulate_pooling_windows(stream, values, gates, absolute_offset, context)
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: MlxTensor,
        gates: MlxTensor,
    ) -> Result<PoolingOverlap<MlxTensor>, Error> {
        self.value.replace_pooling_overlap(stream, values, gates)
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, Error> {
        self.value.append_pooled(stream, values, context)
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<Option<MlxTensor>, Error> {
        self.value
            .pooling_mask(stream, query_tokens, offset, context)
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint, Error> {
        Ok(OwnedPredictionCache::new(
            self.value.checkpoint()?,
            self.memory.clone(),
        ))
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, context: &Stream) -> Result<(), Error> {
        self.memory.extend_from(&checkpoint.memory);
        self.value.restore(&checkpoint.value, context)
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.value.finalize()
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.value.clear()
    }
}

#[cfg(test)]
mod tests;
