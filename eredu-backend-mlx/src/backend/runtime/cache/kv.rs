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

// TODO: somehow move quantized methods to a separate trait?
/// A per-layer attention key/value cache.
pub trait KeyValueCache {
    /// Returns whether this cache stores quantized keys and values.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

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

/// Append-only compressed-token cache with an incomplete pooling window.
///
/// Values and gate logits are accumulated until `ratio` source tokens are
/// available. Complete windows are returned to the architecture's pooling
/// operator, while pooled outputs grow independently at the compressed token
/// rate. This representation is shared by compressed-attention and indexer
/// streams and preserves partial decode state across calls.
#[derive(Debug, Clone)]
pub struct PoolingCache {
    ratio: i32,
    pending_values: Option<Array>,
    pending_gates: Option<Array>,
    pooled: Option<Array>,
    overlap_values: Option<Array>,
    overlap_gates: Option<Array>,
    processed_tokens: i32,
}

/// Exact materialized state of an append-only pooling stream.
#[derive(Debug, Clone, Default)]
pub struct PoolingCacheState {
    pub pending_values: Option<Array>,
    pub pending_gates: Option<Array>,
    pub pooled: Option<Array>,
    pub overlap_values: Option<Array>,
    pub overlap_gates: Option<Array>,
}

/// Complete source windows emitted by [`PoolingCache::accumulate_windows`].
pub struct PoolingWindows {
    /// Source values shaped `[batch, complete_source_tokens, width]`.
    pub values: Array,
    /// Gate logits with the same source-token extent.
    pub gates: Array,
    /// Absolute source-token position of the first returned value.
    pub base_position: i32,
}

impl PoolingCache {
    /// Number of source tokens represented by one pooled token.
    pub const fn ratio(&self) -> i32 {
        self.ratio
    }

    /// Creates an empty pooling stream.
    pub fn new(ratio: i32) -> Result<Self, Exception> {
        if ratio <= 0 {
            return Err(Exception::custom("pooling ratio must be positive"));
        }
        Ok(Self {
            ratio,
            pending_values: None,
            pending_gates: None,
            pooled: None,
            overlap_values: None,
            overlap_gates: None,
            processed_tokens: 0,
        })
    }

    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let clone = |array: &Option<Array>| {
            array
                .as_ref()
                .map(|array| array.clone().deep_clone())
                .transpose()
        };
        Ok(Self {
            ratio: self.ratio,
            pending_values: clone(&self.pending_values)?,
            pending_gates: clone(&self.pending_gates)?,
            pooled: clone(&self.pooled)?,
            overlap_values: clone(&self.overlap_values)?,
            overlap_gates: clone(&self.overlap_gates)?,
            processed_tokens: self.processed_tokens,
        })
    }

    /// Source tokens represented by complete and incomplete windows.
    pub const fn processed_tokens(&self) -> i32 {
        self.processed_tokens
    }

    /// Number of compressed tokens currently retained.
    pub fn pooled_tokens(&self) -> i32 {
        self.pooled.as_ref().map_or(0, |pooled| pooled.dim(1))
    }

    /// Returns every array whose lifetime is part of this cache state.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.pending_values
            .iter()
            .chain(self.pending_gates.iter())
            .chain(self.pooled.iter())
            .chain(self.overlap_values.iter())
            .chain(self.overlap_gates.iter())
    }

    /// Borrows the exact array components used by prompt-cache persistence.
    pub fn state_arrays(&self) -> [Option<&Array>; 5] {
        [
            self.pending_values.as_ref(),
            self.pending_gates.as_ref(),
            self.pooled.as_ref(),
            self.overlap_values.as_ref(),
            self.overlap_gates.as_ref(),
        ]
    }

    /// Restores an exactly validated pooling frontier.
    pub fn restore_state(
        &mut self,
        state: PoolingCacheState,
        processed_tokens: i32,
    ) -> Result<(), Exception> {
        if processed_tokens < 0 {
            return Err(Exception::custom(
                "restored pooling source offset must be non-negative",
            ));
        }
        if state.pending_values.is_some() != state.pending_gates.is_some()
            || state.overlap_values.is_some() != state.overlap_gates.is_some()
        {
            return Err(Exception::custom(
                "restored pooling value/gate components are incomplete",
            ));
        }
        let pending = processed_tokens % self.ratio;
        if state
            .pending_values
            .as_ref()
            .map_or(0, |array| array.dim(1))
            != pending
            || state.pooled.as_ref().map_or(0, |array| array.dim(1))
                != processed_tokens / self.ratio
        {
            return Err(Exception::custom(
                "restored pooling state does not match its source-token frontier",
            ));
        }
        self.pending_values = state.pending_values;
        self.pending_gates = state.pending_gates;
        self.pooled = state.pooled;
        self.overlap_values = state.overlap_values;
        self.overlap_gates = state.overlap_gates;
        self.processed_tokens = processed_tokens;
        Ok(())
    }

    /// Replaces the overlap carried between adjacent complete windows and
    /// returns the previous pair. This is used by stride-one compressed
    /// streams whose logical window spans the previous and current groups.
    pub fn replace_overlap(
        &mut self,
        values: Array,
        gates: Array,
    ) -> (Option<Array>, Option<Array>) {
        let previous_values = self.overlap_values.replace(values);
        let previous_gates = self.overlap_gates.replace(gates);
        (previous_values, previous_gates)
    }

    /// Adds source values and returns only complete pooling windows.
    pub fn accumulate_windows(
        &mut self,
        values: Array,
        gates: Array,
        absolute_offset: i32,
        stream: &Stream,
    ) -> Result<PoolingWindows, Exception> {
        if values.ndim() != 3
            || gates.ndim() != 3
            || values.dim(0) != gates.dim(0)
            || values.dim(1) != gates.dim(1)
        {
            return Err(Exception::custom(
                "pooling values and gates must be rank-3 with matching batch/token dimensions",
            ));
        }
        if absolute_offset != self.processed_tokens {
            return Err(Exception::custom(format!(
                "pooling cache expected source offset {}, got {absolute_offset}",
                self.processed_tokens
            )));
        }
        let previous_pending = self.pending_values.as_ref().map_or(0, |value| value.dim(1));
        let values = match self.pending_values.take() {
            Some(previous) => concatenate_axis(&[previous, values], 1, stream)?,
            None => values,
        };
        let gates = match self.pending_gates.take() {
            Some(previous) => concatenate_axis(&[previous, gates], 1, stream)?,
            None => gates,
        };
        let total = values.dim(1);
        let usable = total / self.ratio * self.ratio;
        self.processed_tokens = self
            .processed_tokens
            .checked_add(total - previous_pending)
            .ok_or_else(|| Exception::custom("pooling source offset overflowed"))?;
        let ready_values = values.try_index_device((.., ..usable, ..), stream)?;
        let ready_gates = gates.try_index_device((.., ..usable, ..), stream)?;
        if usable < total {
            self.pending_values = Some(values.try_index_device((.., usable.., ..), stream)?);
            self.pending_gates = Some(gates.try_index_device((.., usable.., ..), stream)?);
        }
        Ok(PoolingWindows {
            values: ready_values,
            gates: ready_gates,
            base_position: absolute_offset - previous_pending,
        })
    }

    /// Appends newly compressed values and returns the complete compressed history.
    pub fn update_and_fetch(
        &mut self,
        new_pooled: Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if new_pooled.ndim() != 3 {
            return Err(Exception::custom(
                "pooled cache values must have shape [batch, tokens, width]",
            ));
        }
        let empty_shape = [new_pooled.dim(0), 0, new_pooled.dim(-1)];
        let dtype = new_pooled.dtype();
        if new_pooled.dim(1) > 0 {
            self.pooled = Some(match self.pooled.take() {
                Some(previous) => concatenate_axis(&[previous, new_pooled], 1, stream)?,
                None => new_pooled,
            });
        }
        match &self.pooled {
            Some(pooled) => Ok(pooled.clone()),
            None => zeros_dtype(&empty_shape, dtype, stream),
        }
    }

    /// Builds the causal validity mask for compressed positions.
    ///
    /// Query `offset + j` may consume pooled token `i` only after its complete
    /// source window exists: `i < (offset + j + 1) / ratio`.
    pub fn make_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let pooled_tokens = self.pooled_tokens();
        if pooled_tokens == 0 || query_tokens == 1 {
            return Ok(None);
        }
        let pooled = Array::arange::<i32, i32>(Some(0), pooled_tokens, None, stream)?;
        let visible =
            Array::arange::<i32, i32>(Some(offset + 1), offset + query_tokens + 1, None, stream)?
                .floor_divide(Array::from_int(self.ratio), stream)?
                .reshape(&[query_tokens, 1], stream)?;
        Ok(Some(
            pooled
                .reshape(&[1, pooled_tokens], stream)?
                .lt(visible, stream)?,
        ))
    }

    /// Clears all complete and partial compressed state.
    pub fn clear(&mut self) {
        self.pending_values = None;
        self.pending_gates = None;
        self.pooled = None;
        self.overlap_values = None;
        self.overlap_gates = None;
        self.processed_tokens = 0;
    }
}

const COMPRESSED_LATENT_CACHE_STEP: i32 = 256;

/// Compressed attention cache that stores one latent KV vector and one rotary
/// key vector per token, independent of the number of attention heads.
///
/// This representation is used by Multi-head Latent Attention (MLA). Arrays
/// have shape `[batch, sequence, dimension]`; head-specific keys and values are
/// reconstructed transiently by the attention implementation.
#[derive(Debug, Clone)]
pub struct CompressedLatentCache {
    latent_storage: Option<Array>,
    rotary_key_storage: Option<Array>,
    latent: Option<Array>,
    rotary_key: Option<Array>,
    offset: i32,
    length: i32,
    capacity: i32,
    step: i32,
    paged: Option<Box<PagedCompressedLatentCache>>,
}

impl Default for CompressedLatentCache {
    fn default() -> Self {
        Self {
            latent_storage: None,
            rotary_key_storage: None,
            latent: None,
            rotary_key: None,
            offset: 0,
            length: 0,
            capacity: 0,
            step: COMPRESSED_LATENT_CACHE_STEP,
            paged: None,
        }
    }
}

impl CompressedLatentCache {
    /// Creates an empty compressed latent cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forks mutable cache state while retaining the immutable paging catalog.
    ///
    /// Resident storage and a paged mutable tail receive independent MLX
    /// buffers. Sealed paged blocks remain shared through the residency
    /// manager because they are immutable and addressed by token range.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let clone_array = |array: &Option<Array>| {
            array
                .as_ref()
                .map(|array| array.clone().deep_clone())
                .transpose()
        };
        let mut clone = self.clone();
        clone.latent_storage = clone_array(&self.latent_storage)?;
        clone.rotary_key_storage = clone_array(&self.rotary_key_storage)?;
        clone.latent = clone_array(&self.latent)?;
        clone.rotary_key = clone_array(&self.rotary_key)?;
        if let (Some(clone), Some(source)) = (clone.paged.as_deref_mut(), self.paged.as_deref()) {
            clone.tail_latent = clone_array(&source.tail_latent)?;
            clone.tail_rotary = clone_array(&source.tail_rotary)?;
        }
        Ok(clone)
    }

    /// Clears local arrays after the shared paging manager has been cleared.
    pub fn reset_local_after_manager_clear(&mut self) {
        if let Some(paged) = self.paged.as_deref_mut() {
            paged.tail_latent = None;
            paged.tail_rotary = None;
            paged.tail_start = 0;
            paged.offset = 0;
        } else {
            *self = Self::default();
        }
    }

    /// Creates compressed-latent paging under a shared model-wide manager.
    pub fn new_paged(
        manager: CacheResidencyManager,
        global_layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        if !manager.options().full_attention_enabled() {
            return Err(Exception::custom(
                "paged compressed-latent attention requires explicit blockwise full-attention enablement",
            ));
        }
        Ok(Self {
            paged: Some(Box::new(PagedCompressedLatentCache::new(
                manager,
                global_layer,
                rank,
            )?)),
            ..Self::default()
        })
    }

    /// Returns whether this cache uses block-addressable compressed state.
    pub const fn is_paged(&self) -> bool {
        self.paged.is_some()
    }

    /// Returns the shared manager for a paged compressed cache.
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.paged.as_deref().map(|paged| &paged.manager)
    }

    #[cfg(test)]
    fn paged_block_ids(&self) -> Result<Option<Vec<CacheBlockId>>, Exception> {
        self.paged
            .as_deref()
            .map(PagedCompressedLatentCache::block_ids)
            .transpose()
    }

    #[cfg(test)]
    fn paged_tail_block(&self) -> Option<PagedLatentAttentionBlock> {
        self.paged
            .as_deref()
            .and_then(PagedCompressedLatentCache::tail_block)
    }

    /// Seals a partial compressed tail for safe persistence.
    pub fn finalize(&mut self) -> Result<(), Exception> {
        match self.paged.as_deref_mut() {
            Some(paged) => paged.seal_tail(),
            None => Ok(()),
        }
    }

    /// Returns the number of cached tokens.
    pub fn offset(&self) -> i32 {
        self.paged.as_deref().map_or(self.offset, |paged| {
            i32::try_from(paged.offset).unwrap_or(i32::MAX)
        })
    }

    /// Returns the allocated token capacity of the backing arrays.
    pub fn capacity(&self) -> i32 {
        self.paged
            .as_deref()
            .map_or(self.capacity, |paged| paged.tail_len())
    }

    /// Returns the retained latent and rotary-key arrays, when initialized.
    pub fn arrays(&self) -> Option<(&Array, &Array)> {
        if self.paged.is_some() {
            return None;
        }
        Some((self.latent.as_ref()?, self.rotary_key.as_ref()?))
    }

    pub fn retained_arrays(&self) -> Vec<&Array> {
        match self.paged.as_deref() {
            Some(paged) => paged
                .tail_latent
                .iter()
                .chain(paged.tail_rotary.iter())
                .collect(),
            None => self.latent.iter().chain(self.rotary_key.iter()).collect(),
        }
    }

    #[cfg(test)]
    pub fn restore_resident(
        &mut self,
        latent: Array,
        rotary_key: Array,
        end: i32,
    ) -> Result<(), Exception> {
        if latent.dim(1) != rotary_key.dim(1) || latent.dim(1) != end {
            return Err(Exception::custom(
                "restored compressed cache arrays do not match their token range",
            ));
        }
        self.paged = None;
        self.latent_storage = Some(latent.clone());
        self.rotary_key_storage = Some(rotary_key.clone());
        self.latent = Some(latent);
        self.rotary_key = Some(rotary_key);
        self.offset = end;
        self.length = end;
        self.capacity = end;
        Ok(())
    }

    /// Clears all retained state.
    pub fn clear(&mut self) -> Result<(), Exception> {
        if let Some(paged) = self.paged.as_deref_mut() {
            return paged.clear();
        }
        self.latent_storage = None;
        self.rotary_key_storage = None;
        self.latent = None;
        self.rotary_key = None;
        self.offset = 0;
        self.length = 0;
        self.capacity = 0;
        Ok(())
    }

    /// Restores an earlier speculative frontier without leaving paged blocks
    /// written after that frontier visible to the shared residency manager.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        match (&mut self.paged, &checkpoint.paged) {
            (Some(current), Some(previous)) => {
                if current.global_layer != previous.global_layer
                    || current.manager.session_id() != previous.manager.session_id()
                {
                    return Err(Exception::custom(
                        "compressed cache checkpoint does not belong to the same paged layer",
                    ));
                }
                current.truncate(previous.offset, stream)?;
            }
            (None, None) => {}
            _ => {
                return Err(Exception::custom(
                    "compressed cache checkpoint changed residency mode",
                ));
            }
        }
        self.clone_from(checkpoint);
        Ok(())
    }

    fn grown_capacity(&self, required: i32) -> i32 {
        let chunks = (required + self.step - 1) / self.step;
        chunks * self.step
    }

    fn padded(array: &Array, capacity: i32, stream: &Stream) -> Result<Array, Exception> {
        let mut shape = array.shape().to_vec();
        shape[1] = capacity;
        zeros_dtype(&shape, array.dtype(), stream)
    }

    fn refresh_logical_arrays(&mut self, stream: &Stream) -> Result<(), Exception> {
        self.latent = Some(
            self.latent_storage
                .as_ref()
                .expect("latent cache storage initialized")
                .try_index_device((.., ..self.length, ..), stream)?,
        );
        self.rotary_key = Some(
            self.rotary_key_storage
                .as_ref()
                .expect("rotary-key cache storage initialized")
                .try_index_device((.., ..self.length, ..), stream)?,
        );
        Ok(())
    }

    /// Appends compressed states and returns the full states to attend over.
    pub fn update_and_fetch(
        &mut self,
        latent: Array,
        rotary_key: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        if let Some(paged) = self.paged.as_deref_mut() {
            let returned = (latent.clone(), rotary_key.clone());
            paged.append(latent, rotary_key, stream)?;
            return Ok(returned);
        }
        if latent.ndim() != 3 || rotary_key.ndim() != 3 {
            return Err(Exception::custom(
                "compressed latent cache expects rank-3 [batch, sequence, dimension] arrays",
            ));
        }
        if latent.dim(0) != rotary_key.dim(0) || latent.dim(1) != rotary_key.dim(1) {
            return Err(Exception::custom(
                "compressed latent and rotary-key cache updates must share batch and sequence dimensions",
            ));
        }
        if let (Some(previous_latent), Some(previous_rotary)) =
            (&self.latent_storage, &self.rotary_key_storage)
        {
            if previous_latent.dim(0) != latent.dim(0)
                || previous_latent.dim(2) != latent.dim(2)
                || previous_rotary.dim(0) != rotary_key.dim(0)
                || previous_rotary.dim(2) != rotary_key.dim(2)
            {
                return Err(Exception::custom(
                    "compressed latent cache update dimensions do not match retained state",
                ));
            }
        }

        let new_tokens = latent.dim(1);
        let required = self.length + new_tokens;
        if self.latent_storage.is_none() {
            self.capacity = self.grown_capacity(required);
            if self.capacity == required {
                self.latent_storage = Some(latent);
                self.rotary_key_storage = Some(rotary_key);
                self.length = required;
                self.offset += new_tokens;
                self.refresh_logical_arrays(stream)?;
                return Ok((
                    self.latent
                        .as_ref()
                        .expect("latent cache initialized")
                        .clone(),
                    self.rotary_key
                        .as_ref()
                        .expect("rotary-key cache initialized")
                        .clone(),
                ));
            }
            self.latent_storage = Some(Self::padded(&latent, self.capacity, stream)?);
            self.rotary_key_storage = Some(Self::padded(&rotary_key, self.capacity, stream)?);
        } else if required > self.capacity {
            let new_capacity = self.grown_capacity(required);
            let padding = new_capacity - self.capacity;
            let latent_padding = Self::padded(&latent, padding, stream)?;
            let rotary_padding = Self::padded(&rotary_key, padding, stream)?;
            self.latent_storage = Some(concatenate_axis(
                &[
                    self.latent_storage
                        .take()
                        .expect("latent cache storage initialized"),
                    latent_padding,
                ],
                1,
                stream,
            )?);
            self.rotary_key_storage = Some(concatenate_axis(
                &[
                    self.rotary_key_storage
                        .take()
                        .expect("rotary-key cache storage initialized"),
                    rotary_padding,
                ],
                1,
                stream,
            )?);
            self.capacity = new_capacity;
        }

        self.latent_storage
            .as_mut()
            .expect("latent cache storage initialized")
            .try_index_mut_device((.., self.length..required, ..), &latent, stream)?;
        self.rotary_key_storage
            .as_mut()
            .expect("rotary-key cache storage initialized")
            .try_index_mut_device((.., self.length..required, ..), &rotary_key, stream)?;
        self.length = required;
        self.offset += new_tokens;
        self.refresh_logical_arrays(stream)?;
        Ok((
            self.latent
                .as_ref()
                .expect("latent cache initialized")
                .clone(),
            self.rotary_key
                .as_ref()
                .expect("rotary-key cache initialized")
                .clone(),
        ))
    }
}

impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for CompressedLatentCache {
    type RetainedValues<'a> = RetainedArrayVecIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_arrays().into_iter().map(retained_tensor)
    }
}

impl CompressedAttentionCache<MlxTensor> for CompressedLatentCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        CompressedLatentCache::offset(self)
    }

    fn is_paged(&self) -> bool {
        CompressedLatentCache::is_paged(self)
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<MlxTensor>,
        context: &Stream,
    ) -> Result<CompressedAttentionView<MlxTensor>, ComputeError> {
        let appended = state.clone();
        let (latent, rotary) = self
            .update_and_fetch(
                state.latent.into_array(),
                state.rotary.into_array(),
                context,
            )
            .map_err(ComputeError::backend)?;
        if self.is_paged() {
            Ok(CompressedAttentionView::Paged { appended })
        } else {
            Ok(CompressedAttentionView::Resident(
                CompressedAttentionState {
                    latent: MlxTensor::from_array(latent),
                    rotary: MlxTensor::from_array(rotary),
                },
            ))
        }
    }

    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &Stream,
        mut visitor: F,
    ) -> Result<CompressedAttentionScan, ComputeError>
    where
        F: FnMut(CompressedAttentionBlock<MlxTensor>) -> Result<u64, ComputeError>,
    {
        let paged = self
            .paged
            .as_deref_mut()
            .ok_or_else(|| ComputeError::backend("compressed block scan requires paged state"))?;
        let block_ids = paged.block_ids().map_err(ComputeError::backend)?;
        let manager = paged.manager.clone();
        let global_layer = paged.global_layer;
        let tail = paged.tail_block();
        let mut scan = CompressedAttentionScan::default();
        let mut blocks = manager
            .prefetch_blocks(block_ids, context)
            .map_err(ComputeError::backend)?;
        while let Some(lease) = blocks.next_block().map_err(ComputeError::backend)? {
            let id = lease.id();
            let state = match lease.arrays() {
                CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                    CompressedAttentionState {
                        latent: MlxTensor::from_array(latent.clone()),
                        rotary: MlxTensor::from_array(rotary_key.clone()),
                    }
                }
                _ => {
                    return Err(ComputeError::backend(
                        "paged compressed cache found an incompatible block representation",
                    ));
                }
            };
            scan.reconstruction_scratch_bytes =
                scan.reconstruction_scratch_bytes
                    .max(visitor(CompressedAttentionBlock {
                        start: id.start,
                        end: id.end,
                        state,
                    })?);
            scan.blocks += 1;
            scan.bytes += lease.bytes();
            drop(lease);
        }
        if let Some(tail) = tail {
            scan.reconstruction_scratch_bytes =
                scan.reconstruction_scratch_bytes
                    .max(visitor(CompressedAttentionBlock {
                        start: tail.start,
                        end: tail.end,
                        state: CompressedAttentionState {
                            latent: MlxTensor::from_array(tail.latent),
                            rotary: MlxTensor::from_array(tail.rotary_key),
                        },
                    })?);
            scan.blocks += 1;
            scan.bytes += tail.bytes;
        }
        manager
            .record_attention_scan(
                global_layer,
                query_tokens > 1,
                scan.blocks,
                scan.bytes,
                scan.reconstruction_scratch_bytes,
            )
            .map_err(ComputeError::backend)?;
        Ok(scan)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &Stream,
    ) -> Result<(), ComputeError> {
        self.restore_checkpoint(checkpoint, context)
            .map_err(ComputeError::backend)
    }

    fn finalize(&mut self) -> Result<(), ComputeError> {
        CompressedLatentCache::finalize(self).map_err(ComputeError::backend)
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        CompressedLatentCache::clear(self).map_err(ComputeError::backend)
    }
}

#[derive(Debug, Clone)]
struct PagedCompressedLatentCache {
    manager: CacheResidencyManager,
    global_layer: usize,
    rank: Option<CacheRankIdentity>,
    tail_latent: Option<Array>,
    tail_rotary: Option<Array>,
    tail_start: i64,
    offset: i64,
}

impl PagedCompressedLatentCache {
    fn new(
        manager: CacheResidencyManager,
        global_layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let offset = manager
            .layer_end(global_layer, CacheRepresentation::CompressedLatentRotary)
            .map_err(cache_residency_exception)?;
        Ok(Self {
            manager,
            global_layer,
            rank,
            tail_latent: None,
            tail_rotary: None,
            tail_start: offset,
            offset,
        })
    }

    fn tail_len(&self) -> i32 {
        self.tail_latent.as_ref().map_or(0, |latent| latent.dim(1))
    }

    fn tail_bytes(&self) -> u64 {
        self.tail_latent
            .iter()
            .chain(self.tail_rotary.iter())
            .map(|array| array.nbytes() as u64)
            .sum()
    }

    fn append(&mut self, latent: Array, rotary: Array, stream: &Stream) -> Result<(), Exception> {
        self.manager
            .bind_transfer_device(stream)
            .map_err(cache_residency_exception)?;
        if latent.ndim() != 3
            || rotary.ndim() != 3
            || latent.dim(0) != rotary.dim(0)
            || latent.dim(1) != rotary.dim(1)
            || latent.dtype() != rotary.dtype()
            || latent.dim(1) <= 0
        {
            return Err(Exception::custom(
                "paged compressed cache expects same-dtype rank-3 arrays with matching batch and sequence dimensions",
            ));
        }
        if let Some(tail) = &self.tail_latent {
            if tail.dim(0) != latent.dim(0)
                || tail.dim(2) != latent.dim(2)
                || tail.dtype() != latent.dtype()
                || self.tail_rotary.as_ref().is_none_or(|old| {
                    old.dim(0) != rotary.dim(0)
                        || old.dim(2) != rotary.dim(2)
                        || old.dtype() != rotary.dtype()
                })
            {
                return Err(Exception::custom(
                    "paged compressed cache update dimensions do not match the retained tail",
                ));
            }
        }
        let previous_tail_latent = self.tail_latent.clone();
        let previous_tail_rotary = self.tail_rotary.clone();
        let previous_tail_start = self.tail_start;
        let previous_offset = self.offset;
        let previous_blocks = self.block_ids()?;
        let result = self.append_inner(latent, rotary, stream);
        if let Err(error) = result {
            let rollback = self.rollback_append(
                previous_tail_latent,
                previous_tail_rotary,
                previous_tail_start,
                previous_offset,
                &previous_blocks,
            );
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback) => Err(Exception::custom(format!(
                    "{error}; additionally failed to roll back compressed cache append: {rollback}"
                ))),
            };
        }
        Ok(())
    }

    fn append_inner(
        &mut self,
        latent: Array,
        rotary: Array,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let block_size = self.manager.options().block_size_tokens();
        let input_len = latent.dim(1);
        let mut input_start = 0;
        while input_start < input_len {
            let candidate_tail_start = if self.tail_latent.is_none() {
                self.offset + input_start as i64
            } else {
                self.tail_start
            };
            let take = (block_size - self.tail_len()).min(input_len - input_start);
            let input_end = input_start + take;
            let latent_part = latent.try_index_device((.., input_start..input_end, ..), stream)?;
            let rotary_part = rotary.try_index_device((.., input_start..input_end, ..), stream)?;
            let candidate_latent = match &self.tail_latent {
                Some(previous) => concatenate_axis(&[previous.clone(), latent_part], 1, stream)?,
                None => latent_part,
            };
            let candidate_rotary = match &self.tail_rotary {
                Some(previous) => concatenate_axis(&[previous.clone(), rotary_part], 1, stream)?,
                None => rotary_part,
            };
            let candidate_bytes =
                candidate_latent.nbytes() as u64 + candidate_rotary.nbytes() as u64;
            let candidate_end = candidate_tail_start + candidate_latent.dim(1) as i64;
            self.manager
                .set_tail_state(self.global_layer, candidate_bytes, candidate_end)
                .map_err(cache_residency_exception)?;
            self.tail_start = candidate_tail_start;
            self.tail_latent = Some(candidate_latent);
            self.tail_rotary = Some(candidate_rotary);
            input_start = input_end;
            if self.tail_len() == block_size {
                self.seal_tail()?;
            }
        }
        self.offset += input_len as i64;
        Ok(())
    }

    fn rollback_append(
        &mut self,
        previous_tail_latent: Option<Array>,
        previous_tail_rotary: Option<Array>,
        previous_tail_start: i64,
        previous_offset: i64,
        previous_blocks: &[CacheBlockId],
    ) -> Result<(), Exception> {
        self.tail_latent = previous_tail_latent;
        self.tail_rotary = previous_tail_rotary;
        self.tail_start = previous_tail_start;
        self.offset = previous_offset;
        let current_blocks = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::CompressedLatentRotary,
                0,
                i64::MAX,
                0,
            )
            .map_err(cache_residency_exception)?;
        let mut rollback_error = None;
        for id in current_blocks
            .into_iter()
            .rev()
            .filter(|id| !previous_blocks.contains(id))
        {
            if let Err(error) = self.manager.remove_block(&id) {
                rollback_error.get_or_insert_with(|| cache_residency_exception(error));
            }
        }
        if let Err(error) =
            self.manager
                .set_tail_state(self.global_layer, self.tail_bytes(), previous_offset)
        {
            rollback_error.get_or_insert_with(|| cache_residency_exception(error));
        }
        match rollback_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn truncate(&mut self, len: i64, stream: &Stream) -> Result<(), Exception> {
        if len < 0 || len > self.offset {
            return Err(Exception::custom(format!(
                "paged compressed cache truncate length {len} is outside 0..{}",
                self.offset
            )));
        }
        if len >= self.tail_start {
            let retained = i32::try_from(len - self.tail_start)
                .map_err(|_| Exception::custom("paged compressed cache truncate overflow"))?;
            let candidate_latent = self
                .tail_latent
                .as_ref()
                .map(|latent| latent.try_index_device((.., ..retained, ..), stream))
                .transpose()?;
            let candidate_rotary = self
                .tail_rotary
                .as_ref()
                .map(|rotary| rotary.try_index_device((.., ..retained, ..), stream))
                .transpose()?;
            let (candidate_latent, candidate_rotary) = if retained == 0 {
                (None, None)
            } else {
                (candidate_latent, candidate_rotary)
            };
            let candidate_bytes = candidate_latent
                .iter()
                .chain(candidate_rotary.iter())
                .map(|array| array.nbytes() as u64)
                .sum();
            self.manager
                .set_tail_state(self.global_layer, candidate_bytes, len)
                .map_err(cache_residency_exception)?;
            self.tail_latent = candidate_latent;
            self.tail_rotary = candidate_rotary;
            self.offset = len;
            return Ok(());
        }

        let ids = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::CompressedLatentRotary,
                0,
                self.offset,
                0,
            )
            .map_err(cache_residency_exception)?;
        let crossing = ids.into_iter().find(|id| id.start < len && id.end > len);
        let replacement = if let Some(id) = crossing {
            let lease = self
                .manager
                .lease_block(&id, stream)
                .map_err(cache_residency_exception)?;
            let retained = i32::try_from(len - id.start)
                .map_err(|_| Exception::custom("paged compressed cache truncate overflow"))?;
            let (latent, rotary_key) = match lease.arrays() {
                CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => (
                    latent.try_index_device((.., ..retained, ..), stream)?,
                    rotary_key.try_index_device((.., ..retained, ..), stream)?,
                ),
                _ => {
                    return Err(Exception::custom(
                        "paged compressed cache found an incompatible block representation",
                    ));
                }
            };
            safemlx::transforms::async_eval_with_event([&latent, &rotary_key])?.synchronize()?;
            let latent = latent.deep_clone()?;
            let rotary_key = rotary_key.deep_clone()?;
            Some((
                lease,
                CacheBlockArrays::CompressedLatentRotary { latent, rotary_key },
            ))
        } else {
            None
        };
        self.manager
            .truncate_layer_transaction(
                self.global_layer,
                CacheRepresentation::CompressedLatentRotary,
                len,
                replacement,
                0,
            )
            .map_err(cache_residency_exception)?;
        self.tail_latent = None;
        self.tail_rotary = None;
        self.offset = len;
        self.tail_start = len;
        Ok(())
    }

    fn seal_tail(&mut self) -> Result<(), Exception> {
        let Some(latent) = self.tail_latent.take() else {
            return Ok(());
        };
        let rotary_key = self
            .tail_rotary
            .take()
            .expect("compressed latent and rotary tails are initialized atomically");
        let end = self.tail_start + latent.dim(1) as i64;
        if let Err(error) = self.manager.set_tail_state(self.global_layer, 0, end) {
            self.tail_latent = Some(latent);
            self.tail_rotary = Some(rotary_key);
            return Err(cache_residency_exception(error));
        }
        if let Err(error) = self.manager.seal_block(
            self.global_layer,
            self.tail_start,
            end,
            self.rank,
            CacheBlockArrays::CompressedLatentRotary {
                latent: latent.clone(),
                rotary_key: rotary_key.clone(),
            },
            false,
        ) {
            self.tail_latent = Some(latent);
            self.tail_rotary = Some(rotary_key);
            self.manager
                .set_tail_state(self.global_layer, self.tail_bytes(), end)
                .map_err(cache_residency_exception)?;
            return Err(cache_residency_exception(error));
        }
        self.tail_start = end;
        Ok(())
    }

    fn block_ids(&self) -> Result<Vec<CacheBlockId>, Exception> {
        self.manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::CompressedLatentRotary,
                0,
                self.offset,
                0,
            )
            .map_err(cache_residency_exception)
    }

    fn tail_block(&self) -> Option<PagedLatentAttentionBlock> {
        if let (Some(latent), Some(rotary_key)) = (&self.tail_latent, &self.tail_rotary) {
            Some(PagedLatentAttentionBlock {
                start: self.tail_start,
                end: self.offset,
                latent: latent.clone(),
                rotary_key: rotary_key.clone(),
                bytes: self.tail_bytes(),
            })
        } else {
            None
        }
    }

    fn clear(&mut self) -> Result<(), Exception> {
        self.manager
            .truncate_layer_transaction(
                self.global_layer,
                CacheRepresentation::CompressedLatentRotary,
                0,
                None,
                0,
            )
            .map_err(cache_residency_exception)?;
        self.tail_latent = None;
        self.tail_rotary = None;
        self.tail_start = 0;
        self.offset = 0;
        Ok(())
    }
}

pub struct PagedLatentAttentionBlock {
    pub start: i64,
    pub end: i64,
    pub latent: Array,
    pub rotary_key: Array,
    pub bytes: u64,
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache + ?Sized,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        T::retained_arrays(self)
    }

    fn is_paged(&self) -> bool {
        T::is_paged(self)
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        T::paged_attention(self, queries, scale, mask, sinks, stream)
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        T::update_for_attention(self, keys, values, stream)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values, stream)
    }
}

/// Block-addressable key/value cache sharing one global residency manager.
///
/// Sealed blocks are immutable. Appends modify only the layer-local tail, and
/// exact full attention scans blocks without constructing a whole-history
/// device array. Sliding attention discards state that no query can observe.
#[derive(Debug, Clone)]
pub struct PagedKeyValueCache {
    manager: CacheResidencyManager,
    global_layer: usize,
    rank: Option<CacheRankIdentity>,
    sliding_window: Option<i32>,
    key_only: bool,
    prefix_tokens: i32,
    tail_keys: Option<Array>,
    tail_values: Option<Array>,
    tail_start: i64,
    offset: i64,
}

/// Rollback metadata for one sequential semantic branch.
///
/// The local tail arrays are already deep-cloned into the branch. This value
/// records only the shared-manager state that a discarded append must restore.
#[derive(Debug)]
pub struct PagedKeyValueTransactionCheckpoint {
    session_id: u64,
    global_layer: usize,
    offset: i64,
    tail_bytes: u64,
    block_ids: Vec<CacheBlockId>,
}

/// A live key/value cache whose residency is selected independently from the
/// model's parameter residency.
///
/// Hybrid architecture caches use this value beside their fixed-size
/// convolution or recurrent state. Call sites continue to consume the shared
/// [`KeyValueCache`] contract and therefore do not need architecture-specific
/// paged-attention dispatch.
#[derive(Debug, Clone)]
pub enum LiveKeyValueCache {
    /// Chunked key/value arrays retained on the execution device.
    Resident(ConcatKeyValueCache),
    /// Block-addressable key/value arrays under explicit finite budgets.
    Paged(PagedKeyValueCache),
}

impl LiveKeyValueCache {
    /// Wraps an architecture-selected resident cache growth policy.
    pub const fn resident(cache: ConcatKeyValueCache) -> Self {
        Self::Resident(cache)
    }

    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        match self {
            Self::Resident(cache) => cache.deep_clone_state().map(Self::Resident),
            Self::Paged(cache) => Ok(Self::Paged(cache.clone())),
        }
    }

    /// Creates a block-addressable cache with exact global-layer and rank
    /// identity.
    pub fn paged(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        PagedKeyValueCache::new_with_layout(
            manager,
            global_layer,
            sliding_window,
            prefix_tokens,
            rank,
        )
        .map(Self::Paged)
    }

    /// Creates a block-addressable key-only cache. The pager stores a
    /// canonical one-channel sentinel beside each key block so the common
    /// prompt-cache representation remains self-describing.
    pub fn paged_key_only(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        PagedKeyValueCache::new_key_only_with_layout(
            manager,
            global_layer,
            sliding_window,
            prefix_tokens,
            rank,
        )
        .map(Self::Paged)
    }

    /// Returns a current paging report, or `None` for resident state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::Resident(_) => Ok(None),
            Self::Paged(cache) => cache.report().map(Some),
        }
    }

    /// Snapshots resident state. Paged state is persisted through its manager
    /// and deliberately has no monolithic array snapshot.
    pub fn snapshot_arrays(&self, stream: &Stream) -> Result<Option<(Array, Array)>, Exception> {
        match self {
            Self::Resident(cache) => cache.snapshot_arrays(stream),
            Self::Paged(_) => Ok(None),
        }
    }

    /// Seals a paged mutable tail before persistence. Resident state requires
    /// no preparation.
    pub fn finalize(&mut self) -> Result<(), Exception> {
        match self {
            Self::Resident(_) => Ok(()),
            Self::Paged(cache) => cache.finalize(),
        }
    }

    /// Restores an earlier transactional frontier, removing paged deltas.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        match (self, checkpoint) {
            (Self::Resident(cache), Self::Resident(previous)) => {
                cache.clone_from(previous);
                Ok(())
            }
            (Self::Paged(cache), Self::Paged(previous)) => {
                cache.restore_checkpoint(previous, stream)
            }
            _ => Err(Exception::custom(
                "live key/value checkpoint changed residency mode",
            )),
        }
    }

    /// Clears this layer after a model-wide paging manager clear.
    pub fn reset_local_after_manager_clear(&mut self) {
        match self {
            Self::Resident(cache) => cache.clear(),
            Self::Paged(cache) => cache.reset_local_after_manager_clear(),
        }
    }

    /// Returns the paging manager when paging is active.
    pub const fn manager(&self) -> Option<&CacheResidencyManager> {
        match self {
            Self::Resident(_) => None,
            Self::Paged(cache) => Some(cache.manager()),
        }
    }

    /// Restores a contiguous snapshot into resident state.
    pub fn restore_resident(
        &mut self,
        keys: Array,
        values: Array,
        offset: i32,
    ) -> Result<(), Exception> {
        match self {
            Self::Resident(cache) => cache.restore_resident(keys, values, offset),
            Self::Paged(_) => Err(Exception::custom(
                "cannot restore a monolithic snapshot into a paged live cache",
            )),
        }
    }
}

impl KeyValueCache for LiveKeyValueCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Resident(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Resident(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Resident(cache) => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Resident(_) => Ok(None),
            Self::Paged(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Resident(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Resident(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }
}

impl PagedKeyValueCache {
    /// Creates one layer cache attached to a model-wide manager.
    pub fn new(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
    ) -> Result<Self, Exception> {
        Self::new_with_layout(manager, global_layer, sliding_window, 0, None)
    }

    /// Forks the mutable tail while sharing immutable sealed blocks.
    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let clone_array = |array: &Option<Array>| {
            array
                .as_ref()
                .map(|array| array.clone().deep_clone())
                .transpose()
        };
        let mut clone = self.clone();
        clone.tail_keys = clone_array(&self.tail_keys)?;
        clone.tail_values = clone_array(&self.tail_values)?;
        Ok(clone)
    }

    /// Captures the shared-manager frontier required to discard one
    /// sequential speculative branch.
    ///
    /// Sliding paging is excluded because advancing it may irreversibly
    /// discard immutable blocks before the branch is committed. Full-attention
    /// paging is append-only, so rollback removes only newly sealed blocks and
    /// restores the canonical mutable-tail accounting.
    pub fn transaction_checkpoint(&self) -> Result<PagedKeyValueTransactionCheckpoint, Exception> {
        if self.sliding_window.is_some() {
            return Err(Exception::custom(
                "transactional branching is unsupported for paged sliding attention",
            ));
        }
        let block_ids = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                i64::MAX,
                0,
            )
            .map_err(cache_residency_exception)?;
        Ok(PagedKeyValueTransactionCheckpoint {
            session_id: self.manager.session_id(),
            global_layer: self.global_layer,
            offset: self.offset,
            tail_bytes: self.tail_bytes(),
            block_ids,
        })
    }

    /// Removes all manager-visible state appended by a discarded sequential
    /// branch. Existing immutable blocks are retained even if their physical
    /// residency changed while the branch was executing.
    pub fn rollback_transaction(
        &mut self,
        checkpoint: &PagedKeyValueTransactionCheckpoint,
    ) -> Result<(), Exception> {
        if checkpoint.session_id != self.manager.session_id()
            || checkpoint.global_layer != self.global_layer
        {
            return Err(Exception::custom(
                "paged transaction checkpoint does not belong to this cache layer",
            ));
        }
        let current = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                i64::MAX,
                0,
            )
            .map_err(cache_residency_exception)?;
        if checkpoint
            .block_ids
            .iter()
            .any(|expected| !current.contains(expected))
        {
            return Err(Exception::custom(
                "paged transaction removed canonical immutable blocks",
            ));
        }
        let mut first_error = None;
        for id in current
            .into_iter()
            .rev()
            .filter(|id| !checkpoint.block_ids.contains(id))
        {
            if let Err(error) = self.manager.remove_block(&id) {
                first_error.get_or_insert_with(|| cache_residency_exception(error));
            }
        }
        if let Err(error) =
            self.manager
                .set_tail_state(self.global_layer, checkpoint.tail_bytes, checkpoint.offset)
        {
            first_error.get_or_insert_with(|| cache_residency_exception(error));
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Creates one layer cache with pinned-prefix and rank-local identity.
    pub fn new_with_layout(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::new_with_layout_and_mode(
            manager,
            global_layer,
            sliding_window,
            prefix_tokens,
            rank,
            false,
        )
    }

    /// Creates one key-only layer cache with pinned-prefix and rank identity.
    pub fn new_key_only_with_layout(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        Self::new_with_layout_and_mode(
            manager,
            global_layer,
            sliding_window,
            prefix_tokens,
            rank,
            true,
        )
    }

    fn new_with_layout_and_mode(
        manager: CacheResidencyManager,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
        key_only: bool,
    ) -> Result<Self, Exception> {
        if sliding_window.is_some_and(|window| window <= 0) {
            return Err(Exception::custom(
                "paged sliding attention window must be positive",
            ));
        }
        if sliding_window.is_none() && !manager.options().full_attention_enabled() {
            return Err(Exception::custom(
                "paged full attention requires explicit blockwise full-attention enablement",
            ));
        }
        if prefix_tokens < 0 {
            return Err(Exception::custom(
                "paged attention prefix token count must not be negative",
            ));
        }
        let offset = manager
            .layer_end(global_layer, CacheRepresentation::KeyValue)
            .map_err(cache_residency_exception)?;
        Ok(Self {
            manager,
            global_layer,
            rank,
            sliding_window,
            key_only,
            prefix_tokens,
            tail_keys: None,
            tail_values: None,
            tail_start: offset,
            offset,
        })
    }

    /// Returns the shared model-wide residency manager.
    pub const fn manager(&self) -> &CacheResidencyManager {
        &self.manager
    }

    /// Returns whether another cache has the same immutable transaction
    /// identity. Speculative branches may advance their tails independently,
    /// but must not switch residency sessions, ranks, layers, or geometry when
    /// they are published back into canonical state.
    pub fn has_same_transaction_identity(&self, other: &Self) -> bool {
        self.manager.session_id() == other.manager.session_id()
            && self.global_layer == other.global_layer
            && self.rank == other.rank
            && self.sliding_window == other.sliding_window
            && self.key_only == other.key_only
            && self.prefix_tokens == other.prefix_tokens
    }

    /// Returns the global layer identity of this cache.
    pub const fn global_layer(&self) -> usize {
        self.global_layer
    }

    /// Returns the exact visible attention window, or `None` for full attention.
    pub const fn attention_window(&self) -> Option<i32> {
        self.sliding_window
    }

    /// Returns whether values are represented by the canonical one-channel
    /// persistence sentinel instead of an attention payload.
    pub const fn is_key_only(&self) -> bool {
        self.key_only
    }

    /// Returns a current aggregate manager report.
    pub fn report(&self) -> Result<CacheResidencyReport, Exception> {
        self.manager.report().map_err(cache_residency_exception)
    }

    pub fn reset_local_after_manager_clear(&mut self) {
        self.tail_keys = None;
        self.tail_values = None;
        self.tail_start = 0;
        self.offset = 0;
    }

    /// Seals a partially filled tail so it is safe to persist.
    pub fn finalize(&mut self) -> Result<(), Exception> {
        self.seal_tail()
    }

    /// Truncates this layer to an absolute token length.
    pub fn truncate(&mut self, len: i64, stream: &Stream) -> Result<(), Exception> {
        if len < 0 || len > self.offset {
            return Err(Exception::custom(format!(
                "paged cache truncate length {len} is outside 0..{}",
                self.offset
            )));
        }
        if len >= self.tail_start {
            let retained = i32::try_from(len - self.tail_start)
                .map_err(|_| Exception::custom("paged cache truncate length overflow"))?;
            let candidate_keys = self
                .tail_keys
                .as_ref()
                .map(|keys| keys.try_index_device((.., .., ..retained, ..), stream))
                .transpose()?;
            let candidate_values = self
                .tail_values
                .as_ref()
                .map(|values| values.try_index_device((.., .., ..retained, ..), stream))
                .transpose()?;
            let (candidate_keys, candidate_values) = if retained == 0 {
                (None, None)
            } else {
                (candidate_keys, candidate_values)
            };
            let candidate_bytes = candidate_keys
                .iter()
                .chain(candidate_values.iter())
                .map(|array| array.nbytes() as u64)
                .sum();
            self.manager
                .set_tail_state(self.global_layer, candidate_bytes, len)
                .map_err(cache_residency_exception)?;
            self.tail_keys = candidate_keys;
            self.tail_values = candidate_values;
            self.offset = len;
            return Ok(());
        }

        let ids = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                self.offset,
                self.prefix_tokens as i64,
            )
            .map_err(cache_residency_exception)?;
        let crossing = ids.into_iter().find(|id| id.start < len && id.end > len);
        let replacement = if let Some(id) = crossing {
            let lease = self
                .manager
                .lease_block(&id, stream)
                .map_err(cache_residency_exception)?;
            let retained = i32::try_from(len - id.start)
                .map_err(|_| Exception::custom("paged cache truncate length overflow"))?;
            let (keys, values) = match lease.arrays() {
                CacheBlockArrays::KeyValue { keys, values } => (
                    keys.try_index_device((.., .., ..retained, ..), stream)?,
                    values.try_index_device((.., .., ..retained, ..), stream)?,
                ),
                _ => {
                    return Err(Exception::custom(
                        "paged key/value cache found an incompatible block representation",
                    ))
                }
            };
            safemlx::transforms::async_eval_with_event([&keys, &values])?.synchronize()?;
            let keys = keys.deep_clone()?;
            let values = values.deep_clone()?;
            Some((lease, CacheBlockArrays::KeyValue { keys, values }))
        } else {
            None
        };
        self.manager
            .truncate_layer_transaction(
                self.global_layer,
                CacheRepresentation::KeyValue,
                len,
                replacement,
                self.prefix_tokens as i64,
            )
            .map_err(cache_residency_exception)?;
        self.tail_keys = None;
        self.tail_values = None;
        self.offset = len;
        self.tail_start = len;
        Ok(())
    }

    /// Restores an earlier speculative frontier and atomically removes blocks
    /// sealed after it.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.global_layer != checkpoint.global_layer
            || self.manager.session_id() != checkpoint.manager.session_id()
            || self.key_only != checkpoint.key_only
        {
            return Err(Exception::custom(
                "key/value cache checkpoint does not belong to the same paged layer",
            ));
        }
        self.truncate(checkpoint.offset, stream)?;
        self.clone_from(checkpoint);
        Ok(())
    }

    /// Clears live state while preserving paging configuration.
    pub fn clear(&mut self) -> Result<(), Exception> {
        self.manager
            .truncate_layer_transaction(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                None,
                0,
            )
            .map_err(cache_residency_exception)?;
        self.tail_keys = None;
        self.tail_values = None;
        self.tail_start = 0;
        self.offset = 0;
        Ok(())
    }

    fn tail_len(&self) -> i32 {
        self.tail_keys.as_ref().map_or(0, |keys| keys.dim(-2))
    }

    fn tail_bytes(&self) -> u64 {
        self.tail_keys
            .iter()
            .chain(self.tail_values.iter())
            .map(|array| array.nbytes() as u64)
            .sum()
    }

    fn seal_tail(&mut self) -> Result<(), Exception> {
        let Some(keys) = self.tail_keys.take() else {
            return Ok(());
        };
        let values = self
            .tail_values
            .take()
            .expect("paged keys and values tails are initialized atomically");
        let end = self.tail_start + keys.dim(-2) as i64;
        if let Err(error) = self.manager.set_tail_state(self.global_layer, 0, end) {
            self.tail_keys = Some(keys);
            self.tail_values = Some(values);
            return Err(cache_residency_exception(error));
        }
        if let Err(error) = self.manager.seal_block(
            self.global_layer,
            self.tail_start,
            end,
            self.rank,
            CacheBlockArrays::KeyValue {
                keys: keys.clone(),
                values: values.clone(),
            },
            end <= self.prefix_tokens as i64,
        ) {
            self.tail_keys = Some(keys);
            self.tail_values = Some(values);
            self.manager
                .set_tail_state(self.global_layer, self.tail_bytes(), end)
                .map_err(cache_residency_exception)?;
            return Err(cache_residency_exception(error));
        }
        self.tail_start = end;
        Ok(())
    }

    fn normalize_update_values(
        &self,
        keys: &Array,
        values: Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if !self.key_only {
            return Ok(values);
        }
        if keys.ndim() != 4
            || values.ndim() != 4
            || keys.shape()[..3] != values.shape()[..3]
            || values.dim(-1) != 0
            || keys.dtype() != values.dtype()
        {
            return Err(Exception::custom(
                "paged key-only cache expects matching rank-4 keys and zero-width values",
            ));
        }
        let mut shape = keys.shape().to_vec();
        *shape.last_mut().expect("rank-4 key-only cache") = 1;
        zeros_dtype(&shape, keys.dtype(), stream)
    }

    fn validate_update(&self, keys: &Array, values: &Array) -> Result<(), Exception> {
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(Exception::custom(
                "paged key/value cache expects rank-4 [batch, heads, sequence, dimension] arrays",
            ));
        }
        let geometry_matches = if self.key_only {
            keys.shape()[..3] == values.shape()[..3] && values.dim(-1) == 1
        } else {
            keys.shape() == values.shape()
        };
        if !geometry_matches || keys.dtype() != values.dtype() {
            return Err(Exception::custom(
                "paged key/value cache updates must have identical key and value shapes and dtypes",
            ));
        }
        if keys.dim(-2) <= 0 {
            return Err(Exception::custom(
                "paged key/value cache update must contain at least one token",
            ));
        }
        Ok(())
    }

    fn append(&mut self, keys: Array, values: Array, stream: &Stream) -> Result<(), Exception> {
        self.manager
            .bind_transfer_device(stream)
            .map_err(cache_residency_exception)?;
        let values = self.normalize_update_values(&keys, values, stream)?;
        self.validate_update(&keys, &values)?;
        if let Some(tail) = &self.tail_keys {
            if tail.dim(0) != keys.dim(0)
                || tail.dim(1) != keys.dim(1)
                || tail.dim(3) != keys.dim(3)
                || tail.dtype() != keys.dtype()
            {
                return Err(Exception::custom(
                    "paged key/value cache update does not match the retained tail",
                ));
            }
        }
        let previous_tail_keys = self.tail_keys.clone();
        let previous_tail_values = self.tail_values.clone();
        let previous_tail_start = self.tail_start;
        let previous_offset = self.offset;
        let previous_blocks = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                i64::MAX,
                0,
            )
            .map_err(cache_residency_exception)?;
        let result = self.append_inner(keys, values, stream);
        if let Err(error) = result {
            let rollback = self.rollback_append(
                previous_tail_keys,
                previous_tail_values,
                previous_tail_start,
                previous_offset,
                &previous_blocks,
            );
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback) => Err(Exception::custom(format!(
                    "{error}; additionally failed to roll back key/value cache append: {rollback}"
                ))),
            };
        }
        Ok(())
    }

    fn append_inner(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let block_size = self.manager.options().block_size_tokens();
        let mut input_start = 0;
        let input_len = keys.dim(-2);
        while input_start < input_len {
            let candidate_tail_start = if self.tail_keys.is_none() {
                self.offset + input_start as i64
            } else {
                self.tail_start
            };
            let available = block_size - self.tail_len();
            let take = available.min(input_len - input_start);
            let input_end = input_start + take;
            let key_part = keys.try_index_device((.., .., input_start..input_end, ..), stream)?;
            let value_part =
                values.try_index_device((.., .., input_start..input_end, ..), stream)?;
            let candidate_keys = match &self.tail_keys {
                Some(previous) => concatenate_axis(&[previous.clone(), key_part], -2, stream)?,
                None => key_part,
            };
            let candidate_values = match &self.tail_values {
                Some(previous) => concatenate_axis(&[previous.clone(), value_part], -2, stream)?,
                None => value_part,
            };
            let candidate_bytes = candidate_keys.nbytes() as u64 + candidate_values.nbytes() as u64;
            let candidate_end = candidate_tail_start + candidate_keys.dim(-2) as i64;
            self.manager
                .set_tail_state(self.global_layer, candidate_bytes, candidate_end)
                .map_err(cache_residency_exception)?;
            self.tail_start = candidate_tail_start;
            self.tail_keys = Some(candidate_keys);
            self.tail_values = Some(candidate_values);
            input_start = input_end;
            if self.tail_len() == block_size {
                self.seal_tail()?;
            }
        }
        self.offset += input_len as i64;
        if let Some(window) = self.sliding_window {
            let visible_start = (self.offset - window as i64).max(self.prefix_tokens as i64);
            self.manager
                .discard_before(
                    self.global_layer,
                    CacheRepresentation::KeyValue,
                    visible_start,
                    self.prefix_tokens as i64,
                )
                .map_err(cache_residency_exception)?;
        }
        Ok(())
    }

    fn rollback_append(
        &mut self,
        previous_tail_keys: Option<Array>,
        previous_tail_values: Option<Array>,
        previous_tail_start: i64,
        previous_offset: i64,
        previous_blocks: &[CacheBlockId],
    ) -> Result<(), Exception> {
        self.tail_keys = previous_tail_keys;
        self.tail_values = previous_tail_values;
        self.tail_start = previous_tail_start;
        self.offset = previous_offset;
        let current_blocks = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                0,
                i64::MAX,
                0,
            )
            .map_err(cache_residency_exception)?;
        let mut rollback_error = None;
        for id in current_blocks
            .into_iter()
            .rev()
            .filter(|id| !previous_blocks.contains(id))
        {
            if let Err(error) = self.manager.remove_block(&id) {
                rollback_error.get_or_insert_with(|| cache_residency_exception(error));
            }
        }
        if let Err(error) =
            self.manager
                .set_tail_state(self.global_layer, self.tail_bytes(), previous_offset)
        {
            rollback_error.get_or_insert_with(|| cache_residency_exception(error));
        }
        match rollback_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn contiguous_visible(
        &self,
        start: i64,
        end: i64,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let ids = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                start,
                end,
                0,
            )
            .map_err(cache_residency_exception)?;
        let mut key_parts = Vec::new();
        let mut value_parts = Vec::new();
        let mut blocks = self
            .manager
            .prefetch_blocks(ids, stream)
            .map_err(cache_residency_exception)?;
        while let Some(lease) = blocks.next_block().map_err(cache_residency_exception)? {
            let id = lease.id();
            let slice_start = i32::try_from(start.max(id.start) - id.start)
                .map_err(|_| Exception::custom("paged cache visible range overflow"))?;
            let slice_end = i32::try_from(end.min(id.end) - id.start)
                .map_err(|_| Exception::custom("paged cache visible range overflow"))?;
            match lease.arrays() {
                CacheBlockArrays::KeyValue { keys, values } => {
                    key_parts
                        .push(keys.try_index_device((.., .., slice_start..slice_end, ..), stream)?);
                    value_parts.push(
                        values.try_index_device((.., .., slice_start..slice_end, ..), stream)?,
                    );
                }
                _ => {
                    return Err(Exception::custom(
                        "paged key/value cache found an incompatible block representation",
                    ))
                }
            }
        }
        if let (Some(keys), Some(values)) = (&self.tail_keys, &self.tail_values) {
            if self.tail_start < end && self.offset > start {
                let slice_start = i32::try_from(start.max(self.tail_start) - self.tail_start)
                    .map_err(|_| Exception::custom("paged cache visible range overflow"))?;
                let slice_end = i32::try_from(end.min(self.offset) - self.tail_start)
                    .map_err(|_| Exception::custom("paged cache visible range overflow"))?;
                key_parts
                    .push(keys.try_index_device((.., .., slice_start..slice_end, ..), stream)?);
                value_parts
                    .push(values.try_index_device((.., .., slice_start..slice_end, ..), stream)?);
            }
        }
        let key_refs = key_parts.iter().collect::<Vec<_>>();
        let value_refs = value_parts.iter().collect::<Vec<_>>();
        if key_refs.is_empty() {
            return Err(Exception::custom("paged cache visible range is empty"));
        }
        let keys = if key_refs.len() == 1 {
            key_refs[0].clone()
        } else {
            concatenate_axis(&key_refs, -2, stream)?
        };
        let values = if value_refs.len() == 1 {
            value_refs[0].clone()
        } else {
            concatenate_axis(&value_refs, -2, stream)?
        };
        Ok((keys, values))
    }
}

impl Default for PagedKeyValueCache {
    fn default() -> Self {
        let options = PagedCacheOptions::new(1, 1, 0, 1)
            .expect("internal paged cache placeholder options are finite");
        let manager = CacheResidencyManager::new(options)
            .expect("internal paged cache placeholder manager can be created");
        Self {
            manager,
            global_layer: 0,
            rank: None,
            sliding_window: Some(1),
            key_only: false,
            prefix_tokens: 0,
            tail_keys: None,
            tail_values: None,
            tail_start: 0,
            offset: 0,
        }
    }
}

impl KeyValueCache for PagedKeyValueCache {
    fn offset(&self) -> i32 {
        i32::try_from(self.offset).unwrap_or(i32::MAX)
    }

    fn max_size(&self) -> Option<i32> {
        self.sliding_window
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.tail_keys
            .iter()
            .chain(self.tail_values.iter())
            .collect()
    }

    fn is_paged(&self) -> bool {
        true
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        if self.key_only {
            return Err(Exception::custom(
                "key-only paged caches require architecture-owned attention",
            ));
        }
        let query_len = queries.dim(-2) as i64;
        let query_start = self.offset - query_len;
        let visible_start = self
            .sliding_window
            .map_or(0, |window| (query_start - (window - 1) as i64).max(0));
        let ids = self
            .manager
            .layer_block_ids(
                self.global_layer,
                CacheRepresentation::KeyValue,
                visible_start,
                self.offset,
                self.prefix_tokens as i64,
            )
            .map_err(cache_residency_exception)?;
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            queries,
            scale,
            mask,
            query_start,
            self.sliding_window,
            self.prefix_tokens as i64,
            sinks,
            self.offset,
            stream,
        )?;
        let mut scanned_blocks = 0u64;
        let mut scanned_bytes = 0u64;
        let mut scratch = 0u64;
        let mut blocks = self
            .manager
            .prefetch_blocks(ids, stream)
            .map_err(cache_residency_exception)?;
        while let Some(lease) = blocks.next_block().map_err(cache_residency_exception)? {
            let id = lease.id();
            let (keys, values) = match lease.arrays() {
                CacheBlockArrays::KeyValue { keys, values } => (keys.clone(), values.clone()),
                _ => {
                    return Err(Exception::custom(
                        "paged key/value cache found an incompatible block representation",
                    ))
                }
            };
            let block = KeyValueAttentionBlock::unleased(id.start, id.end, keys, values);
            scratch = scratch.max(
                queries.dim(0) as u64
                    * queries.dim(1) as u64
                    * query_len as u64
                    * (id.end - id.start) as u64
                    * 4,
            );
            scanned_blocks += 1;
            scanned_bytes += lease.bytes();
            accumulator.accumulate(&block, stream)?;
            accumulator.submit()?;
            drop(lease);
        }
        if let (Some(keys), Some(values)) = (&self.tail_keys, &self.tail_values) {
            let block = KeyValueAttentionBlock::unleased(
                self.tail_start,
                self.offset,
                keys.clone(),
                values.clone(),
            );
            scratch = scratch.max(
                queries.dim(0) as u64
                    * queries.dim(1) as u64
                    * query_len as u64
                    * (self.offset - self.tail_start) as u64
                    * 4,
            );
            scanned_blocks += 1;
            scanned_bytes += block.bytes;
            accumulator.accumulate(&block, stream)?;
        }
        let output = accumulator.finish(stream)?;
        safemlx::transforms::eval([&output])?;
        self.manager
            .record_attention_scan(
                self.global_layer,
                query_len > 1,
                scanned_blocks,
                scanned_bytes,
                scratch,
            )
            .map_err(cache_residency_exception)?;
        Ok(Some(output))
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let submitted = (keys.clone(), values.clone());
        self.append(keys, values, stream)?;
        Ok(submitted)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let previous_offset = self.offset;
        let update_len = keys.dim(-2) as i64;
        let fallback_keys = keys.clone();
        let fallback_values = values.clone();
        self.append(keys, values, stream)?;
        if let Some(window) = self.sliding_window {
            let start = (previous_offset - (window - 1) as i64).max(0);
            self.contiguous_visible(start, self.offset, stream)
        } else if update_len == self.offset - previous_offset {
            Ok((fallback_keys, fallback_values))
        } else {
            Err(Exception::custom("paged cache offset changed unexpectedly"))
        }
    }
}

impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for PagedKeyValueCache {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.tail_keys
            .iter()
            .chain(self.tail_values.iter())
            .map(retained_tensor)
    }
}

pub struct KeyValueAttentionBlock {
    pub start: i64,
    pub end: i64,
    pub keys: Array,
    pub values: Array,
    pub bytes: u64,
}

impl KeyValueAttentionBlock {
    pub fn unleased(start: i64, end: i64, keys: Array, values: Array) -> Self {
        let bytes = keys.nbytes() as u64 + values.nbytes() as u64;
        Self {
            start,
            end,
            keys,
            values,
            bytes,
        }
    }
}

/// MLX online-softmax recurrence used behind the neutral blockwise-attention
/// backend contract.
pub struct BlockwiseAttentionAccumulator {
    queries: Array,
    output_dtype: Dtype,
    scale: f32,
    explicit_mask: Option<Array>,
    query_start: i64,
    sliding_window: Option<i32>,
    prefix_tokens: i64,
    sinks: Option<Array>,
    mask_origin: i64,
    batch: i32,
    query_heads: i32,
    query_len: i32,
    head_dim: i32,
    running_max: Option<Array>,
    running_sum: Option<Array>,
    accumulator: Option<Array>,
}

impl BlockwiseAttentionAccumulator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queries: &Array,
        scale: f32,
        explicit_mask: Option<&Array>,
        query_start: i64,
        sliding_window: Option<i32>,
        prefix_tokens: i64,
        sinks: Option<&Array>,
        context_end: i64,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if queries.ndim() != 4 {
            return Err(Exception::custom(
                "blockwise attention requires rank-4 queries",
            ));
        }
        if let Some(mask) = explicit_mask {
            if mask.ndim() != 2 {
                return Err(Exception::custom(
                    "paged attention supports only rank-2 explicit attention masks",
                ));
            }
            if mask.dim(0) != queries.dim(-2) {
                return Err(Exception::custom(
                    "paged attention mask query dimension does not match the active query",
                ));
            }
        }
        let batch = queries.dim(0);
        let query_heads = queries.dim(1);
        let query_len = queries.dim(2);
        let head_dim = queries.dim(3);
        Ok(Self {
            queries: queries.as_dtype(Dtype::Float32, stream)?,
            output_dtype: queries.dtype(),
            scale,
            explicit_mask: explicit_mask.cloned(),
            query_start,
            sliding_window,
            prefix_tokens,
            sinks: sinks.cloned(),
            mask_origin: explicit_mask.map_or(0, |mask| context_end - mask.dim(1) as i64),
            batch,
            query_heads,
            query_len,
            head_dim,
            running_max: None,
            running_sum: None,
            accumulator: None,
        })
    }

    pub fn accumulate(
        &mut self,
        block: &KeyValueAttentionBlock,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.accumulate_with_bias(block, None, stream)
    }

    /// Submits the current recurrence before the next cache-block dependency
    /// is inserted, allowing the dedicated transfer stream to prefetch the
    /// following block while this block's attention executes.
    pub fn submit(&self) -> Result<(), Exception> {
        safemlx::transforms::async_eval(
            self.running_max
                .iter()
                .chain(self.running_sum.iter())
                .chain(self.accumulator.iter()),
        )
    }

    pub fn accumulate_with_bias(
        &mut self,
        block: &KeyValueAttentionBlock,
        additive_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let block_start = block.start;
        let block_end = block.end;
        let keys = &block.keys;
        let values = &block.values;
        if keys.ndim() != 4
            || values.ndim() != 4
            || values.dim(0) != keys.dim(0)
            || values.dim(1) != keys.dim(1)
            || values.dim(2) != keys.dim(2)
        {
            return Err(Exception::custom(
                "paged key/value block shapes are inconsistent",
            ));
        }
        let key_heads = keys.dim(1);
        if self.query_heads % key_heads != 0 {
            return Err(Exception::custom(
                "query attention heads are not divisible by paged key/value heads",
            ));
        }
        let key_len = keys.dim(2);
        let value_dim = values.dim(3);
        let repeats = self.query_heads / key_heads;
        let (keys, values) = if repeats == 1 {
            (keys.clone(), values.clone())
        } else {
            let keys = broadcast_to(
                &keys.reshape(&[self.batch, key_heads, 1, key_len, self.head_dim], stream)?,
                &[self.batch, key_heads, repeats, key_len, self.head_dim],
                stream,
            )?
            .reshape(
                &[self.batch, self.query_heads, key_len, self.head_dim],
                stream,
            )?;
            let values = broadcast_to(
                &values.reshape(&[self.batch, key_heads, 1, key_len, value_dim], stream)?,
                &[self.batch, key_heads, repeats, key_len, value_dim],
                stream,
            )?
            .reshape(&[self.batch, self.query_heads, key_len, value_dim], stream)?;
            (keys, values)
        };
        let keys = keys.as_dtype(Dtype::Float32, stream)?;
        let values = values.as_dtype(Dtype::Float32, stream)?;
        let mut scores = matmul(
            &self.queries.multiply(Array::from_f32(self.scale), stream)?,
            &keys.swap_axes(-1, -2, stream)?,
            stream,
        )?;
        if let Some(bias) = additive_bias {
            scores = scores.add(bias, stream)?;
        }
        let allowed = absolute_attention_mask(
            self.query_start,
            self.query_len,
            block_start,
            block_end,
            self.sliding_window,
            self.prefix_tokens,
        );
        let allowed = Array::from_slice(&allowed, &[self.query_len, key_len]);
        let effective_mask = if let Some(mask) = &self.explicit_mask {
            let relative_start = block_start - self.mask_origin;
            let relative_end = block_end - self.mask_origin;
            if relative_start < 0 || relative_end > mask.dim(1) as i64 {
                return Err(Exception::custom(
                    "paged attention mask does not cover every visible cache block",
                ));
            }
            let mask =
                mask.try_index_device((.., relative_start as i32..relative_end as i32), stream)?;
            if mask.dtype() == Dtype::Bool {
                let combined = allowed.logical_and(&mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            } else {
                let combined =
                    allowed.logical_and(&mask.is_neg_inf(stream)?.logical_not(stream)?, stream)?;
                scores = scores.add(mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            }
        } else {
            scores = r#where(&allowed, scores, Array::from_f32(f32::MIN), stream)?;
            allowed
        };
        let block_max = scores.max_axis(-1, true, stream)?;
        let mut weights = scores.subtract(&block_max, stream)?.exp(stream)?;
        weights = weights.multiply(effective_mask.as_dtype(Dtype::Float32, stream)?, stream)?;
        let block_sum = sum_axis(&weights, -1, true, stream)?;
        let block_accumulator = matmul(&weights, &values, stream)?;
        match (&self.running_max, &self.running_sum, &self.accumulator) {
            (Some(old_max), Some(old_sum), Some(old_accumulator)) => {
                let new_max = maximum(old_max, &block_max, stream)?;
                let old_scale = old_max.subtract(&new_max, stream)?.exp(stream)?;
                let block_scale = block_max.subtract(&new_max, stream)?.exp(stream)?;
                self.running_sum = Some(
                    old_sum
                        .multiply(&old_scale, stream)?
                        .add(block_sum.multiply(&block_scale, stream)?, stream)?,
                );
                self.accumulator = Some(
                    old_accumulator
                        .multiply(&old_scale, stream)?
                        .add(block_accumulator.multiply(&block_scale, stream)?, stream)?,
                );
                self.running_max = Some(new_max);
            }
            _ => {
                if let Some(sinks) = &self.sinks {
                    if sinks.ndim() != 1 || sinks.dim(0) != self.query_heads {
                        return Err(Exception::custom(
                            "paged attention sinks must have one value per query head",
                        ));
                    }
                    let sink = sinks
                        .as_dtype(Dtype::Float32, stream)?
                        .reshape(&[1, self.query_heads, 1, 1], stream)?;
                    let sink = broadcast_to(
                        &sink,
                        &[self.batch, self.query_heads, self.query_len, 1],
                        stream,
                    )?;
                    let new_max = maximum(&sink, &block_max, stream)?;
                    let sink_sum = sink.subtract(&new_max, stream)?.exp(stream)?;
                    let block_scale = block_max.subtract(&new_max, stream)?.exp(stream)?;
                    self.running_max = Some(new_max);
                    self.running_sum =
                        Some(sink_sum.add(block_sum.multiply(&block_scale, stream)?, stream)?);
                    self.accumulator = Some(block_accumulator.multiply(block_scale, stream)?);
                } else {
                    self.running_max = Some(block_max);
                    self.running_sum = Some(block_sum);
                    self.accumulator = Some(block_accumulator);
                }
            }
        }
        safemlx::transforms::eval([
            self.running_max
                .as_ref()
                .expect("blockwise attention initialized row maximum"),
            self.running_sum
                .as_ref()
                .expect("blockwise attention initialized normalization"),
            self.accumulator
                .as_ref()
                .expect("blockwise attention initialized accumulator"),
        ])?;
        Ok(())
    }

    pub fn finish(self, stream: &Stream) -> Result<Array, Exception> {
        let accumulator = self
            .accumulator
            .ok_or_else(|| Exception::custom("blockwise attention received no cache blocks"))?;
        let running_sum = self
            .running_sum
            .ok_or_else(|| Exception::custom("blockwise attention normalization is empty"))?;
        let nonzero = running_sum.gt(Array::from_f32(0.0), stream)?;
        let safe_sum = r#where(&nonzero, &running_sum, Array::from_f32(1.0), stream)?;
        let output = accumulator.divide(safe_sum, stream)?;
        output.as_dtype(self.output_dtype, stream)
    }
}

fn absolute_attention_mask(
    query_start: i64,
    query_len: i32,
    key_start: i64,
    key_end: i64,
    sliding_window: Option<i32>,
    prefix_tokens: i64,
) -> Vec<bool> {
    let mut mask = Vec::with_capacity(query_len as usize * (key_end - key_start) as usize);
    for query in query_start..query_start + query_len as i64 {
        for key in key_start..key_end {
            let causal = key <= query;
            let visible = sliding_window
                .is_none_or(|window| key < prefix_tokens || key >= query - (window - 1) as i64);
            mask.push(causal && visible);
        }
    }
    mask
}

fn cache_residency_exception(error: impl std::fmt::Display) -> Exception {
    Exception::custom(error.to_string())
}

/// A cache that appends all key/value states along the sequence axis.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    key_only: bool,
    offset: i32,
    length: i32,
    capacity: i32,
    step: i32,
    max_size: Option<i32>,
    /// Attention window whose retained past is one token shorter because the
    /// current token is included in the window. Updates return the full
    /// submitted span before applying this retention bound.
    attention_window: Option<i32>,
}

/// A bounded key/value cache for causal sliding-window attention.
///
/// The cache keeps only the newest `max_size` states between calls while
/// preserving an absolute token offset for positional encodings. During an
/// update it returns the retained prefix together with every newly submitted
/// state so multi-token attention can still compute all queries correctly;
/// callers must apply the matching sliding-window mask.
#[derive(Debug, Clone)]
pub struct SlidingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    max_size: i32,
}

impl Default for SlidingKeyValueCache {
    fn default() -> Self {
        // Model-aware high-level callers construct this cache with the actual
        // window. The effectively unbounded default keeps generic direct
        // callers correct until they provide a configured cache explicitly.
        Self::new(i32::MAX)
    }
}

impl SlidingKeyValueCache {
    /// Creates an empty cache retaining at most `max_size` states.
    pub fn new(max_size: i32) -> Self {
        assert!(max_size > 0, "sliding KV cache size must be positive");
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
        }
    }

    /// Returns the arrays retained for the next attention call.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(self.values.iter())
    }

    /// Clears cached arrays while preserving the configured window size.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }
}

impl KeyValueCache for SlidingKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.max_size)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.keys.iter().chain(self.values.iter()).collect()
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let new_tokens = keys.dim(-2);
        let combined_keys = match self.keys.take() {
            Some(previous) => concatenate_axis(&[previous, keys], -2, stream)?,
            None => keys,
        };
        let combined_values = match self.values.take() {
            Some(previous) => concatenate_axis(&[previous, values], -2, stream)?,
            None => values,
        };
        self.offset += new_tokens;

        let combined_len = combined_keys.dim(-2);
        let retained_start = (combined_len - self.max_size).max(0);
        self.keys = Some(combined_keys.try_index_device((.., .., retained_start.., ..), stream)?);
        self.values =
            Some(combined_values.try_index_device((.., .., retained_start.., ..), stream)?);

        Ok((combined_keys, combined_values))
    }
}

impl ConcatKeyValueCache {
    /// Creates an empty concatenating key/value cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty cache that persists keys and reuses them as values.
    pub fn new_key_only() -> Self {
        Self {
            key_only: true,
            ..Self::default()
        }
    }

    pub fn deep_clone_state(&self) -> Result<Self, Exception> {
        let mut cache = self.clone();
        cache.keys = self
            .keys
            .as_ref()
            .map(|array| array.clone().deep_clone())
            .transpose()?;
        cache.values = self
            .values
            .as_ref()
            .map(|array| array.clone().deep_clone())
            .transpose()?;
        Ok(cache)
    }

    /// Creates an empty concatenating cache that retains at most `max_size` tokens.
    pub fn new_with_max_size(max_size: i32) -> Self {
        Self {
            max_size: Some(max_size),
            ..Self::default()
        }
    }

    /// Creates a cache for exact causal sliding-window attention.
    ///
    /// A window of `N` includes the current token, so the cache retains at most
    /// `N - 1` past states between calls while returning every newly submitted
    /// state for multi-token prefill attention.
    pub fn new_for_sliding_attention(window: i32) -> Self {
        assert!(window > 0, "sliding attention window must be positive");
        Self {
            attention_window: Some(window),
            ..Self::default()
        }
    }

    /// Creates an exact sliding-window cache that persists keys only.
    pub fn new_key_only_for_sliding_attention(window: i32) -> Self {
        assert!(window > 0, "sliding attention window must be positive");
        Self {
            key_only: true,
            attention_window: Some(window),
            ..Self::default()
        }
    }

    /// Creates an unbounded cache whose backing arrays grow in `step`-token chunks.
    #[cfg(test)]
    pub fn new_with_step(step: i32) -> Self {
        Self {
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Creates a bounded cache whose backing arrays grow in `step`-token chunks.
    #[cfg(test)]
    pub fn new_with_max_size_and_step(max_size: i32, step: i32) -> Self {
        Self {
            max_size: Some(max_size),
            step: step.max(1),
            ..Self::default()
        }
    }

    /// Truncates the cache to `len` tokens.
    pub fn truncate(&mut self, len: i32, stream: &Stream) -> Result<(), Exception> {
        if self.attention_window.is_some() {
            if len < 0 || len > self.offset {
                return Err(Exception::custom(format!(
                    "sliding cache truncate position {len} is outside 0..{}",
                    self.offset
                )));
            }
            if len == 0 {
                self.clear();
                return Ok(());
            }
            let retained_origin = self.offset - self.length;
            if len < retained_origin {
                return Err(Exception::custom(format!(
                    "cannot roll a sliding cache back to absolute position {len}; retained state starts at {retained_origin}"
                )));
            }
            let retained = len - retained_origin;
            if let Some(keys) = self.keys.take() {
                self.keys = Some(keys.try_index_device((.., .., ..retained, ..), stream)?);
            }
            if let Some(values) = self.values.take() {
                self.values = Some(values.try_index_device((.., .., ..retained, ..), stream)?);
            }
            self.offset = len;
            self.length = retained;
            self.capacity = retained;
            return Ok(());
        }
        if len < 0 || len > self.length {
            return Err(Exception::custom(format!(
                "concatenating cache truncate length {len} is outside 0..{}",
                self.length
            )));
        }
        if self.step > 1 {
            // Chunked caches treat the array shape as backing capacity and
            // expose only `length` through logical_arrays. Rolling the logical
            // frontier back is sufficient: the next update overwrites the
            // abandoned speculative range in place without forcing capacity
            // regrowth.
            self.offset = len;
            self.length = len;
            return Ok(());
        }
        if let Some(keys) = self.keys.take() {
            self.keys = Some(keys.try_index_device((.., .., ..len, ..), stream)?);
        }
        if let Some(values) = self.values.take() {
            self.values = Some(values.try_index_device((.., .., ..len, ..), stream)?);
        }
        self.offset = len;
        self.length = len;
        self.capacity = len;
        Ok(())
    }

    /// Returns the arrays currently retained by the cache.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(self.values.iter())
    }

    /// Clears cached arrays while preserving cache configuration.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.length = 0;
        self.capacity = 0;
    }
}

impl ConcatKeyValueCache {
    fn grown_capacity(&self, required: i32) -> i32 {
        let step = self.step.max(1);
        let chunks = (required + step - 1) / step;
        let capacity = chunks * step;
        self.max_size
            .map_or(capacity, |max_size| capacity.min(max_size))
    }

    fn padded(array: &Array, capacity: i32, stream: &Stream) -> Result<Array, Exception> {
        let mut shape = array.shape().to_vec();
        let sequence_axis = shape.len() - 2;
        shape[sequence_axis] = capacity;
        zeros_dtype(&shape, array.dtype(), stream)
    }

    fn logical_arrays(&self, stream: &Stream) -> Result<(Array, Array), Exception> {
        let keys = self
            .keys
            .as_ref()
            .expect("Keys cannot be None")
            .try_index_device((.., .., ..self.length, ..), stream)?;
        let values = if self.key_only {
            keys.clone()
        } else {
            self.values
                .as_ref()
                .expect("Values cannot be None")
                .try_index_device((.., .., ..self.length, ..), stream)?
        };
        Ok((keys, values))
    }

    pub fn snapshot_arrays(&self, stream: &Stream) -> Result<Option<(Array, Array)>, Exception> {
        if self.keys.is_none() {
            return Ok(None);
        }
        self.logical_arrays(stream).map(Some)
    }

    pub fn restore_resident(
        &mut self,
        keys: Array,
        values: Array,
        end: i32,
    ) -> Result<(), Exception> {
        let length = keys.dim(-2);
        let compatible = keys.ndim() == values.ndim()
            && keys.ndim() >= 2
            && keys.shape()[..keys.ndim() - 2] == values.shape()[..values.ndim() - 2]
            && values.dim(-2) == length;
        if !compatible || end < length {
            return Err(Exception::custom(
                "restored key/value cache arrays do not match their retained range",
            ));
        }
        if self.attention_window.is_none() && end != length {
            return Err(Exception::custom(
                "restored full-attention cache must begin at token zero",
            ));
        }
        if let Some(window) = self.attention_window {
            if length > window.saturating_sub(1) {
                return Err(Exception::custom(
                    "restored sliding cache exceeds its retained history",
                ));
            }
        }
        self.keys = Some(keys);
        self.values = (!self.key_only).then_some(values);
        self.offset = end;
        self.length = length;
        self.capacity = length;
        Ok(())
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        self.attention_window.or(self.max_size)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.keys.iter().chain(self.values.iter()).collect()
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let new_tokens = keys.dim(-2);
        self.offset += new_tokens;

        if let Some(window) = self.attention_window {
            let combined_keys = match self.keys.take() {
                Some(previous) => concatenate_axis(&[previous, keys], -2, stream)?,
                None => keys,
            };
            let combined_values = if self.key_only {
                combined_keys.clone()
            } else {
                match self.values.take() {
                    Some(previous) => concatenate_axis(&[previous, values], -2, stream)?,
                    None => values,
                }
            };
            let retained = (window - 1).min(combined_keys.dim(-2));
            if retained == 0 {
                self.keys = None;
                self.values = None;
            } else {
                let start = combined_keys.dim(-2) - retained;
                self.keys = Some(combined_keys.try_index_device((.., .., start.., ..), stream)?);
                self.values = if self.key_only {
                    None
                } else {
                    Some(combined_values.try_index_device((.., .., start.., ..), stream)?)
                };
            }
            self.length = retained;
            self.capacity = retained;
            return Ok((combined_keys, combined_values));
        }

        if self.step <= 1 {
            if self.key_only {
                self.keys = Some(match self.keys.take() {
                    Some(previous) => concatenate_axis(&[previous, keys], -2, stream)?,
                    None => keys,
                });
            } else {
                match (self.keys.take(), self.values.take()) {
                    (Some(k), Some(v)) => {
                        self.keys = Some(concatenate_axis(&[k, keys], -2, stream)?);
                        self.values = Some(concatenate_axis(&[v, values], -2, stream)?);
                    }
                    _ => {
                        self.keys = Some(keys);
                        self.values = Some(values);
                    }
                }
            }
            if let Some(max_size) = self.max_size {
                let length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
                if length > max_size {
                    let start = length - max_size;
                    self.keys = Some(
                        self.keys
                            .take()
                            .expect("Keys cannot be None")
                            .try_index_device((.., .., start.., ..), stream)?,
                    );
                    if !self.key_only {
                        self.values = Some(
                            self.values
                                .take()
                                .expect("Values cannot be None")
                                .try_index_device((.., .., start.., ..), stream)?,
                        );
                    }
                }
            }
            self.length = self.keys.as_ref().expect("Keys cannot be None").dim(-2);
            self.capacity = self.length;
            return Ok((
                self.keys.clone().expect("Keys cannot be None"),
                if self.key_only {
                    self.keys.clone().expect("Keys cannot be None")
                } else {
                    self.values.clone().expect("Values cannot be None")
                },
            ));
        }

        let required = self.length + new_tokens;

        if let Some(max_size) = self.max_size {
            if required > max_size {
                if self.keys.is_none() {
                    let start = new_tokens - max_size;
                    self.keys = Some(keys.try_index_device((.., .., start.., ..), stream)?);
                    if !self.key_only {
                        self.values = Some(values.try_index_device((.., .., start.., ..), stream)?);
                    }
                    self.length = max_size;
                    self.capacity = max_size;
                    return self.logical_arrays(stream);
                }
                let (old_keys, old_values) = self.logical_arrays(stream)?;
                let combined_keys = concatenate_axis(&[old_keys, keys], -2, stream)?;
                let combined_values = concatenate_axis(&[old_values, values], -2, stream)?;
                let start = required - max_size;
                self.keys = Some(combined_keys.try_index_device((.., .., start.., ..), stream)?);
                if !self.key_only {
                    self.values =
                        Some(combined_values.try_index_device((.., .., start.., ..), stream)?);
                }
                self.length = max_size;
                self.capacity = max_size;
                return self.logical_arrays(stream);
            }
        }

        if self.keys.is_none() {
            self.capacity = self.grown_capacity(required);
            if self.capacity == required {
                self.keys = Some(keys);
                if !self.key_only {
                    self.values = Some(values);
                }
                self.length = required;
                return self.logical_arrays(stream);
            }
            self.keys = Some(Self::padded(&keys, self.capacity, stream)?);
            if !self.key_only {
                self.values = Some(Self::padded(&values, self.capacity, stream)?);
            }
        } else if required > self.capacity {
            let new_capacity = self.grown_capacity(required);
            let padding = new_capacity - self.capacity;
            let key_padding = Self::padded(&keys, padding, stream)?;
            let value_padding = (!self.key_only)
                .then(|| Self::padded(&values, padding, stream))
                .transpose()?;
            self.keys = Some(concatenate_axis(
                &[self.keys.take().expect("Keys cannot be None"), key_padding],
                -2,
                stream,
            )?);
            if let Some(value_padding) = value_padding {
                self.values = Some(concatenate_axis(
                    &[
                        self.values.take().expect("Values cannot be None"),
                        value_padding,
                    ],
                    -2,
                    stream,
                )?);
            }
            self.capacity = new_capacity;
        }

        self.keys
            .as_mut()
            .expect("Keys cannot be None")
            .try_index_mut_device((.., .., self.length..required, ..), &keys, stream)?;
        if !self.key_only {
            self.values
                .as_mut()
                .expect("Values cannot be None")
                .try_index_mut_device((.., .., self.length..required, ..), &values, stream)?;
        }
        self.length = required;
        self.logical_arrays(stream)
    }
}

impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for ConcatKeyValueCache {
    type RetainedValues<'a> = RetainedArrayIter<'a>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.keys
            .iter()
            .chain(self.values.iter())
            .map(retained_tensor)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        BlockwiseAttentionAccumulator, CompressedLatentCache, ConcatKeyValueCache,
        KeyValueAttentionBlock, KeyValueCache, PagedKeyValueCache, PoolingCache,
        SlidingKeyValueCache,
    };
    use crate::backend::runtime::cache::residency::{
        open_prompt_cache, CacheBlockArrays, CacheResidencyManager,
    };
    use eredu_core::cache::{
        CacheRankIdentity, CacheRepresentation, PromptCacheDescriptor, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheTopology,
    };
    use eredu_runtime::{inspect_prompt_cache, resolve_prompt_cache_root, PagedCacheOptions};
    use safemlx::{
        fast::ScaledDotProductAttentionMask,
        ops::{indexing::TryIndexOp, zeros_dtype},
        transforms::eval,
        Array, Device, DeviceType, Dtype, ExecutionContext,
    };

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_key_only_cache_uses_canonical_sentinel_blocks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(2, 4096, 4096, 1).unwrap()).unwrap();
        let mut cache =
            PagedKeyValueCache::new_key_only_with_layout(manager.clone(), 0, Some(8), 0, None)
                .unwrap();
        let keys = Array::from_slice(
            &(0..20).map(|v| v as f32).collect::<Vec<_>>(),
            &[1, 1, 5, 4],
        );
        let values = zeros_dtype(&[1, 1, 5, 0], Dtype::Float32, stream).unwrap();
        let (visible_keys, visible_values) = cache.update_and_fetch(keys, values, stream).unwrap();
        assert_eq!(visible_keys.shape(), &[1, 1, 5, 4]);
        assert_eq!(visible_values.shape(), &[1, 1, 5, 1]);
        cache.finalize().unwrap();
        let ids = manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 5, 0)
            .unwrap();
        assert_eq!(ids.len(), 3);
        for id in ids {
            let lease = manager.lease_block(&id, stream).unwrap();
            let CacheBlockArrays::KeyValue { keys, values } = lease.arrays() else {
                panic!("key-only cache emitted the wrong block representation");
            };
            assert_eq!(keys.dim(-1), 4);
            assert_eq!(values.dim(-1), 1);
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn cache_checkpoint_clone_shares_kv_array_handles() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = ConcatKeyValueCache::new_with_step(8);
        let keys = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        cache.update_and_fetch(keys.clone(), keys, stream).unwrap();
        let checkpoint = cache.clone();
        let cache_arrays = cache.arrays().collect::<Vec<_>>();
        let checkpoint_arrays = checkpoint.arrays().collect::<Vec<_>>();

        assert_eq!(cache_arrays.len(), 2);
        assert_eq!(checkpoint_arrays.len(), 2);
        for (current, saved) in cache_arrays.into_iter().zip(checkpoint_arrays) {
            eval([current, saved]).unwrap();
            let current = current.evaluated().unwrap();
            let saved = saved.evaluated().unwrap();
            assert_eq!(
                current.as_slice::<f32>().as_ptr(),
                saved.as_slice::<f32>().as_ptr()
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn pooling_cache_is_invariant_to_decode_chunking() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = PoolingCache::new(4).unwrap();
        let first = cache
            .accumulate_windows(
                Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1]),
                Array::from_slice(&[0.0f32; 3], &[1, 3, 1]),
                0,
                stream,
            )
            .unwrap();
        assert_eq!(first.values.dim(1), 0);
        let second = cache
            .accumulate_windows(
                Array::from_slice(&[4.0f32, 5.0], &[1, 2, 1]),
                Array::from_slice(&[0.0f32; 2], &[1, 2, 1]),
                3,
                stream,
            )
            .unwrap();
        assert_eq!(second.base_position, 0);
        assert_eq!(second.values.shape(), [1, 4, 1]);
        assert_eq!(cache.processed_tokens(), 5);
        assert_eq!(
            second.values.evaluated().unwrap().as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn chunked_cache_truncate_preserves_backing_capacity_and_overwrites_rollback() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = ConcatKeyValueCache::new_with_step(8);
        let initial = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3, 1]);
        cache
            .update_and_fetch(initial.clone(), initial, stream)
            .unwrap();
        eval(cache.arrays()).unwrap();
        let pointers_before = cache
            .arrays()
            .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr())
            .collect::<Vec<_>>();

        cache.truncate(1, stream).unwrap();

        assert_eq!(cache.length, 1);
        assert_eq!(cache.capacity, 8);
        assert!(cache.arrays().all(|array| array.dim(-2) == 8));
        let pointers_after = cache
            .arrays()
            .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr())
            .collect::<Vec<_>>();
        assert_eq!(pointers_after, pointers_before);

        let replacement = Array::from_slice(&[7.0f32, 8.0], &[1, 1, 2, 1]);
        let (keys, values) = cache
            .update_and_fetch(replacement.clone(), replacement, stream)
            .unwrap();
        eval([&keys, &values]).unwrap();
        assert_eq!(
            keys.evaluated().unwrap().as_slice::<f32>(),
            &[1.0, 7.0, 8.0]
        );
        assert_eq!(
            values.evaluated().unwrap().as_slice::<f32>(),
            &[1.0, 7.0, 8.0]
        );
        assert_eq!(cache.capacity, 8);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn chunked_cache_grows_by_steps_and_preserves_sliding_values() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = ConcatKeyValueCache::new_with_max_size_and_step(8, 4);

        let mut fetched = None;
        for value in 0..10 {
            let keys =
                Array::full::<f32>(&[1, 1, 1, 2], Array::from_f32(value as f32), stream).unwrap();
            let values = keys.clone();
            fetched = Some(cache.update_and_fetch(keys, values, stream).unwrap().0);
            if value == 0 {
                assert_eq!(cache.capacity, 4);
            } else if value == 4 {
                assert_eq!(cache.capacity, 8);
            }
        }

        let fetched = fetched.unwrap();
        assert_eq!(cache.offset(), 10);
        assert_eq!(fetched.shape(), &[1, 1, 8, 2]);
        assert_eq!(
            fetched
                .try_index_device((0, 0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            2.0
        );
        assert_eq!(
            fetched
                .try_index_device((0, 0, -1, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            9.0
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn sliding_cache_returns_transient_context_and_retains_only_its_window() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = SlidingKeyValueCache::new(3);

        let keys = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 4, 1]);
        let fetched = cache
            .update_and_fetch(keys.clone(), keys, stream)
            .unwrap()
            .0;
        assert_eq!(fetched.shape(), &[1, 1, 4, 1]);
        assert_eq!(cache.offset(), 4);
        assert_eq!(cache.arrays().next().unwrap().shape(), &[1, 1, 3, 1]);

        let next = Array::from_slice(&[4.0f32], &[1, 1, 1, 1]);
        let fetched = cache
            .update_and_fetch(next.clone(), next, stream)
            .unwrap()
            .0;
        assert_eq!(fetched.shape(), &[1, 1, 4, 1]);
        assert_eq!(
            fetched
                .try_index_device((0, 0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            1.0
        );
        let retained = cache.arrays().next().unwrap();
        assert_eq!(retained.shape(), &[1, 1, 3, 1]);
        assert_eq!(
            retained
                .try_index_device((0, 0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            2.0
        );
        assert_eq!(cache.offset(), 5);
    }

    fn paged_options(full_attention: bool) -> PagedCacheOptions {
        PagedCacheOptions::new(2, 96, 4096, 1)
            .unwrap()
            .with_full_attention(full_attention)
    }

    fn fully_masked_blockwise_output(mask: &Array, stream: &safemlx::Stream) -> f32 {
        let queries = Array::zeros::<f32>(&[1, 1, 1, 1], stream).unwrap();
        let first = KeyValueAttentionBlock::unleased(
            0,
            2,
            Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap(),
            Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]),
        );
        let second = KeyValueAttentionBlock::unleased(
            2,
            4,
            Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap(),
            Array::from_slice(&[3.0f32, 4.0], &[1, 1, 2, 1]),
        );
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            &queries,
            1.0,
            Some(mask),
            3,
            None,
            0,
            None,
            4,
            stream,
        )
        .unwrap();
        accumulator.accumulate(&first, stream).unwrap();
        accumulator.accumulate(&second, stream).unwrap();
        let output = accumulator.finish(stream).unwrap();
        eval([&output]).unwrap();
        output.evaluated().unwrap().as_slice::<f32>()[0]
    }

    #[derive(Clone, Copy, Debug)]
    struct BlockwiseAttentionCase {
        batch: i32,
        query_heads: i32,
        key_value_heads: i32,
        query_len: i32,
        context_len: i32,
        head_dim: i32,
        block_size: i32,
        dtype: Dtype,
        logit_magnitude: f32,
        tolerance: f64,
    }

    fn patterned_values(length: usize, stride: usize, modulus: usize, scale: f32) -> Vec<f32> {
        (0..length)
            .map(|index| {
                let centered =
                    ((index * stride + stride / 2) % modulus) as f32 - (modulus / 2) as f32;
                centered * scale
            })
            .collect()
    }

    fn assert_blockwise_attention_matches_reference(
        case: BlockwiseAttentionCase,
        stream: &safemlx::Stream,
    ) -> Array {
        assert!(case.query_len <= case.context_len);
        assert_eq!(case.query_heads % case.key_value_heads, 0);
        let query_elements =
            (case.batch * case.query_heads * case.query_len * case.head_dim) as usize;
        let key_elements =
            (case.batch * case.key_value_heads * case.context_len * case.head_dim) as usize;
        let query = Array::from_slice(
            &patterned_values(query_elements, 7, 23, 0.125 * case.logit_magnitude),
            &[case.batch, case.query_heads, case.query_len, case.head_dim],
        )
        .as_dtype(case.dtype, stream)
        .unwrap();
        let keys = Array::from_slice(
            &patterned_values(key_elements, 11, 29, 0.1 * case.logit_magnitude),
            &[
                case.batch,
                case.key_value_heads,
                case.context_len,
                case.head_dim,
            ],
        )
        .as_dtype(case.dtype, stream)
        .unwrap();
        let values = Array::from_slice(
            &patterned_values(key_elements, 5, 19, 0.05),
            &[
                case.batch,
                case.key_value_heads,
                case.context_len,
                case.head_dim,
            ],
        )
        .as_dtype(case.dtype, stream)
        .unwrap();
        let scale = (case.head_dim as f32).sqrt().recip();
        let query_start = (case.context_len - case.query_len) as i64;
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            &query,
            scale,
            None,
            query_start,
            None,
            0,
            None,
            case.context_len as i64,
            stream,
        )
        .unwrap();
        let mut block_start = 0;
        while block_start < case.context_len {
            let block_end = (block_start + case.block_size).min(case.context_len);
            let block = KeyValueAttentionBlock::unleased(
                block_start as i64,
                block_end as i64,
                keys.try_index_device((.., .., block_start..block_end, ..), stream)
                    .unwrap(),
                values
                    .try_index_device((.., .., block_start..block_end, ..), stream)
                    .unwrap(),
            );
            accumulator.accumulate(&block, stream).unwrap();
            block_start = block_end;
        }
        let actual = accumulator.finish(stream).unwrap();
        assert_eq!(actual.dtype(), case.dtype);

        // Compare in FP32 after quantizing the inputs to the requested runtime
        // dtype. This isolates the online reduction from input quantization and
        // verifies that lower-precision caches still accumulate in FP32.
        let query_reference = query.as_dtype(Dtype::Float32, stream).unwrap();
        let key_reference = keys.as_dtype(Dtype::Float32, stream).unwrap();
        let value_reference = values.as_dtype(Dtype::Float32, stream).unwrap();
        let allowed = super::absolute_attention_mask(
            query_start,
            case.query_len,
            0,
            case.context_len as i64,
            None,
            0,
        );
        let mask = Array::from_slice(&allowed, &[case.query_len, case.context_len]);
        let reference = safemlx::fast::scaled_dot_product_attention(
            query_reference,
            key_reference,
            value_reference,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )
        .unwrap();
        let actual = actual.as_dtype(Dtype::Float32, stream).unwrap();
        eval([&actual, &reference]).unwrap();
        assert!(
            actual
                .all_close(&reference, case.tolerance, case.tolerance, None, stream)
                .unwrap()
                .item::<bool>(stream),
            "blockwise attention case did not match its contiguous reference: {case:?}"
        );
        actual
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_returns_zero_for_fully_false_boolean_mask() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mask = Array::from_slice(&[false, false, false, false], &[1, 4]);
        let output = fully_masked_blockwise_output(&mask, context.stream());
        assert_eq!(output, 0.0);
        assert!(output.is_finite());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_returns_zero_for_all_negative_infinity_mask() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mask = Array::from_slice(&[f32::NEG_INFINITY; 4], &[1, 4]);
        let output = fully_masked_blockwise_output(&mask, context.stream());
        assert_eq!(output, 0.0);
        assert!(output.is_finite());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_matches_multiple_block_sizes_and_batched_queries() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        for block_size in [1, 2, 3, 5, 8] {
            assert_blockwise_attention_matches_reference(
                BlockwiseAttentionCase {
                    batch: 2,
                    query_heads: 4,
                    key_value_heads: 4,
                    query_len: 3,
                    context_len: 8,
                    head_dim: 4,
                    block_size,
                    dtype: Dtype::Float32,
                    logit_magnitude: 1.0,
                    tolerance: 1e-5,
                },
                context.stream(),
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_matches_grouped_and_multi_query_attention() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        for key_value_heads in [2, 1] {
            assert_blockwise_attention_matches_reference(
                BlockwiseAttentionCase {
                    batch: 2,
                    query_heads: 4,
                    key_value_heads,
                    query_len: 2,
                    context_len: 7,
                    head_dim: 4,
                    block_size: 3,
                    dtype: Dtype::Float32,
                    logit_magnitude: 1.0,
                    tolerance: 1e-5,
                },
                context.stream(),
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_accumulates_lower_precision_inputs_in_float32() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        for (dtype, tolerance) in [(Dtype::Float16, 2e-3), (Dtype::Bfloat16, 1e-2)] {
            assert_blockwise_attention_matches_reference(
                BlockwiseAttentionCase {
                    batch: 2,
                    query_heads: 4,
                    key_value_heads: 2,
                    query_len: 3,
                    context_len: 7,
                    head_dim: 4,
                    block_size: 3,
                    dtype,
                    logit_magnitude: 1.0,
                    tolerance,
                },
                context.stream(),
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn blockwise_attention_remains_finite_with_large_logits() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let output = assert_blockwise_attention_matches_reference(
            BlockwiseAttentionCase {
                batch: 2,
                query_heads: 4,
                key_value_heads: 1,
                query_len: 3,
                context_len: 9,
                head_dim: 8,
                block_size: 4,
                dtype: Dtype::Float32,
                logit_magnitude: 1_000.0,
                tolerance: 1e-5,
            },
            context.stream(),
        );
        assert!(output
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_full_attention_matches_contiguous_causal_attention() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let queries = Array::from_slice(
            &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            &[1, 1, 5, 2],
        );
        let keys = Array::from_slice(
            &[1.0f32, 0.0, 0.8, 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8],
            &[1, 1, 5, 2],
        );
        let values = Array::from_slice(
            &[0.2f32, 1.0, 0.4, 0.8, 0.6, 0.6, 0.8, 0.4, 1.0, 0.2],
            &[1, 1, 5, 2],
        );
        cache
            .update_and_fetch(keys.clone(), values.clone(), stream)
            .unwrap();
        let paged = cache
            .paged_attention(&queries, 2.0f32.sqrt().recip(), None, None, stream)
            .unwrap()
            .unwrap();
        let reference = safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            2.0f32.sqrt().recip(),
            Some(ScaledDotProductAttentionMask::Causal),
            None,
            stream,
        )
        .unwrap();
        eval([&paged, &reference]).unwrap();
        assert!(paged
            .all_close(&reference, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
        let report = manager.report().unwrap();
        assert_eq!(report.logical_cached_tokens, 5);
        assert_eq!(report.block_seals, 2);
        assert!(report.peak_device_bytes <= manager.options().device_budget_bytes());
        assert_eq!(report.prefill_full_attention_blocks, 3);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_attention_preserves_learned_sink_normalization() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
        let mut cache = PagedKeyValueCache::new(manager, 0, None).unwrap();
        let queries = Array::from_slice(&[0.25f32, 0.5, 0.75, 1.0], &[1, 2, 2, 1]);
        let keys = Array::from_slice(&[0.2f32, 0.4, 0.6, 0.8], &[1, 2, 2, 1]);
        let values = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let sinks = Array::from_slice(&[0.1f32, -0.2], &[2]);
        cache
            .update_and_fetch(keys.clone(), values.clone(), stream)
            .unwrap();
        let paged = cache
            .paged_attention(&queries, 1.0, None, Some(&sinks), stream)
            .unwrap()
            .unwrap();
        let reference = safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            1.0,
            Some(ScaledDotProductAttentionMask::Causal),
            Some(&sinks),
            stream,
        )
        .unwrap();
        eval([&paged, &reference]).unwrap();
        assert!(paged
            .all_close(&reference, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_cache_enforces_one_budget_across_layers() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        // The finite device budget covers one protected block per layer plus
        // the active mutable tail.
        // Host-transfer buffers are page-aligned, so one demoted logical
        // block consumes one 32 KiB physical host allocation.
        let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
            .unwrap()
            .with_full_attention(true);
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut first = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let mut second = PagedKeyValueCache::new(manager.clone(), 1, None).unwrap();
        for cache in [&mut first, &mut second] {
            let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 4, 1]);
            cache
                .update_and_fetch(states.clone(), states, stream)
                .unwrap();
        }
        let report = manager.report().unwrap();
        assert_eq!(report.key_value_blocks, 4);
        assert_eq!(report.device_blocks, 3);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 48);
        assert_eq!(report.current_host_bytes, 32 * 1024);
        assert!(report.peak_device_bytes <= manager.options().device_budget_bytes());
        assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn rejected_key_value_append_restores_tail_and_newly_sealed_blocks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let options = PagedCacheOptions::new(2, 20, 4096, 1)
            .unwrap()
            .with_full_attention(true);
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        cache
            .append(
                Array::from_slice(&[1.0f32], &[1, 1, 1, 1]),
                Array::from_slice(&[10.0f32], &[1, 1, 1, 1]),
                stream,
            )
            .unwrap();

        let error = cache
            .append(
                Array::from_slice(&[2.0f32, 3.0], &[1, 1, 2, 1]),
                Array::from_slice(&[20.0f32, 30.0], &[1, 1, 2, 1]),
                stream,
            )
            .expect_err("the protected sealed block plus tail must exceed the device budget");
        assert!(error.what().contains("budget exceeded"));
        assert_eq!(cache.offset, 1);
        assert_eq!(cache.tail_start, 0);
        assert_eq!(cache.tail_len(), 1);
        assert!(manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .is_empty());
        let report = manager.report().unwrap();
        assert_eq!(report.mutable_tail_bytes, 8);
        assert_eq!(report.current_device_bytes, 8);
        assert_eq!(report.logical_cached_tokens, 1);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn rejected_compressed_append_restores_tail_and_newly_sealed_blocks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let options = PagedCacheOptions::new(2, 20, 4096, 1)
            .unwrap()
            .with_full_attention(true);
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
        cache
            .update_and_fetch(
                Array::from_slice(&[1.0f32], &[1, 1, 1]),
                Array::from_slice(&[10.0f32], &[1, 1, 1]),
                stream,
            )
            .unwrap();

        let error = cache
            .update_and_fetch(
                Array::from_slice(&[2.0f32, 3.0], &[1, 2, 1]),
                Array::from_slice(&[20.0f32, 30.0], &[1, 2, 1]),
                stream,
            )
            .expect_err("the protected sealed block plus tail must exceed the device budget");
        assert!(error.what().contains("budget exceeded"));
        assert_eq!(cache.offset(), 1);
        let tail = cache
            .paged_tail_block()
            .expect("the original compressed tail must be restored");
        assert_eq!((tail.start, tail.end), (0, 1));
        assert!(cache.paged_block_ids().unwrap().unwrap().is_empty());
        let report = manager.report().unwrap();
        assert_eq!(report.mutable_tail_bytes, 8);
        assert_eq!(report.current_device_bytes, 8);
        assert_eq!(report.logical_cached_tokens, 1);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_cache_truncates_at_and_inside_sealed_blocks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();

        cache.truncate(3, stream).unwrap();
        assert_eq!(cache.offset(), 3);
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 3);
        let suffix = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
        cache
            .update_and_fetch(suffix.clone(), suffix, stream)
            .unwrap();
        assert_eq!(cache.offset(), 4);

        cache.truncate(2, stream).unwrap();
        assert_eq!(cache.offset(), 2);
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 2);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn rejected_sealed_truncation_preserves_local_and_manager_state() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 7, 1]);
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();
        let before_ids = manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap();
        let leased_id = before_ids.iter().find(|id| id.start == 2).unwrap().clone();
        let suffix_lease = manager.lease_block(&leased_id, stream).unwrap();

        let error = cache
            .truncate(1, stream)
            .expect_err("a leased middle suffix block must reject truncation");
        assert!(error.what().contains("has lease count 1, expected 0"));
        assert_eq!(cache.offset(), 7);
        assert_eq!(cache.tail_start, 6);
        assert_eq!(cache.tail_len(), 1);
        assert_eq!(
            cache
                .tail_keys
                .as_ref()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[6.0]
        );
        assert_eq!(
            manager
                .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
                .unwrap(),
            before_ids
        );
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 7);

        drop(suffix_lease);
        cache.truncate(1, stream).unwrap();
        assert_eq!(cache.offset(), 1);
        assert_eq!(manager.report().unwrap().logical_cached_tokens, 1);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn compressed_latent_paging_seals_atomic_block_pairs() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
        let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
        let latent = Array::from_slice(
            &[0.0f32, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
            &[1, 5, 2],
        );
        let rotary = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 5, 1]);
        cache.update_and_fetch(latent, rotary, stream).unwrap();
        assert_eq!(cache.offset(), 5);
        let before = manager.report().unwrap();
        assert_eq!(before.compressed_latent_blocks, 2);
        assert!(before.mutable_tail_bytes > 0);
        cache.finalize().unwrap();
        let after = manager.report().unwrap();
        assert_eq!(after.compressed_latent_blocks, 3);
        assert_eq!(after.mutable_tail_bytes, 0);
        assert_eq!(after.logical_cached_tokens, 5);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn compressed_latent_host_demotion_and_rehydration_preserve_atomic_pairs() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        // Each two-token latent/rotary pair occupies 24 logical bytes. Two
        // blocks fit on the device and one page-aligned host allocation fits
        // in the finite host tier.
        let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
            .unwrap()
            .with_full_attention(true);
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
        let latent_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let rotary_values = (100..106).map(|value| value as f32).collect::<Vec<_>>();
        cache
            .update_and_fetch(
                Array::from_slice(&latent_values, &[1, 6, 2]),
                Array::from_slice(&rotary_values, &[1, 6, 1]),
                stream,
            )
            .unwrap();

        let report = manager.report().unwrap();
        assert_eq!(report.compressed_latent_blocks, 3);
        assert_eq!(report.device_blocks, 2);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 48);
        assert_eq!(report.current_host_bytes, 32 * 1024);
        assert_eq!(report.host_demotions, 1);
        let first = cache.paged_block_ids().unwrap().unwrap().remove(0);
        assert_eq!(
            first.representation,
            CacheRepresentation::CompressedLatentRotary
        );

        let lease = manager.lease_block(&first, stream).unwrap();
        match lease.arrays() {
            CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                eval([latent, rotary_key]).unwrap();
                assert_eq!(
                    latent.evaluated().unwrap().as_slice::<f32>(),
                    &latent_values[..4]
                );
                assert_eq!(
                    rotary_key.evaluated().unwrap().as_slice::<f32>(),
                    &rotary_values[..2]
                );
            }
            CacheBlockArrays::KeyValue { .. } => {
                panic!("compressed-latent block was rehydrated as key/value state")
            }
        }
        drop(lease);
        let report = manager.report().unwrap();
        assert_eq!(report.host_promotions, 1);
        assert!(report.host_demotions >= 2);
        assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn compressed_latent_live_disk_demotion_and_rehydration_preserve_atomic_pairs() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let directory = tempfile::tempdir().unwrap();
        let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
            .unwrap()
            .with_full_attention(true)
            .with_live_disk(directory.path(), 4096, 2)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
        let latent_values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let rotary_values = (100..108).map(|value| value as f32).collect::<Vec<_>>();
        cache
            .update_and_fetch(
                Array::from_slice(&latent_values, &[1, 8, 2]),
                Array::from_slice(&rotary_values, &[1, 8, 1]),
                stream,
            )
            .unwrap();

        let mut report = manager.report().unwrap();
        for _ in 0..100 {
            if report.disk_blocks >= 1 && report.in_flight_write_blocks == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            report = manager.report().unwrap();
        }
        assert_eq!(report.compressed_latent_blocks, 4);
        assert!(report.disk_blocks >= 1);
        assert!(report.disk_demotions >= 1);
        assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
        let first = cache.paged_block_ids().unwrap().unwrap().remove(0);

        let lease = manager.lease_block(&first, stream).unwrap();
        match lease.arrays() {
            CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                eval([latent, rotary_key]).unwrap();
                assert_eq!(
                    latent.evaluated().unwrap().as_slice::<f32>(),
                    &latent_values[..4]
                );
                assert_eq!(
                    rotary_key.evaluated().unwrap().as_slice::<f32>(),
                    &rotary_values[..2]
                );
            }
            CacheBlockArrays::KeyValue { .. } => {
                panic!("compressed-latent disk block was rehydrated as key/value state")
            }
        }
        drop(lease);
        let report = manager.report().unwrap();
        assert!(report.disk_promotions >= 1);
        assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());

        drop(cache);
        drop(manager);
        assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn paged_sliding_cache_discards_invisible_blocks_and_preserves_offsets() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let manager = CacheResidencyManager::new(paged_options(false)).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, Some(3)).unwrap();
        let prefix = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache
            .update_and_fetch(prefix.clone(), prefix, stream)
            .unwrap();
        let next = Array::from_slice(&[5.0f32], &[1, 1, 1, 1]);
        let visible = cache
            .update_and_fetch(next.clone(), next, stream)
            .unwrap()
            .0;
        assert_eq!(cache.offset(), 6);
        assert_eq!(visible.shape(), &[1, 1, 3, 1]);
        assert_eq!(
            visible
                .try_index_device((0, 0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            3.0
        );
        assert_eq!(manager.report().unwrap().discarded_sliding_blocks, 1);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn prompt_cache_is_atomic_inspectable_and_reopens_lazily() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let options = paged_options(true);
        let manager = CacheResidencyManager::new(options.clone()).unwrap();
        let rank = CacheRankIdentity {
            pipeline_rank: Some(1),
            tensor_parallel_rank: None,
            expert_parallel_rank: None,
        };
        let mut cache =
            PagedKeyValueCache::new_with_layout(manager.clone(), 0, None, 0, Some(rank)).unwrap();
        let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();
        cache.finalize().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        let descriptor = PromptCacheDescriptor {
            model_family: "decoder".into(),
            effective_model_type: "decoder".into(),
            checkpoint_fingerprint: "sha256:test-checkpoint".into(),
            prefix_content_fingerprint: "tokens:11,12,13,14,15".into(),
            architecture_fingerprint: "sha256:test-architecture".into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            batch_size: 1,
            layer_prefix_offsets: vec![0],
            state_segments: vec![
                eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap(),
            ],
            layer_layout: PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
            sink_tokens: 0,
            topology: PromptCacheTopology {
                pipeline: Some((2, 1)),
                ..PromptCacheTopology::default()
            },
        };
        let tokens = [11u32, 12, 13, 14, 15];
        let invalid_destination = directory.path().join("invalid-prompt-cache");
        let mut invalid_descriptor = descriptor.clone();
        invalid_descriptor.topology.pipeline = Some((2, 0));
        assert!(manager
            .save_prompt_cache(
                &invalid_destination,
                invalid_descriptor,
                &tokens,
                &[],
                &PromptCacheOptions::default(),
            )
            .is_err());
        assert!(!invalid_destination.exists());
        manager
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &tokens,
                &[],
                &PromptCacheOptions::default(),
            )
            .unwrap();
        let inspected = inspect_prompt_cache(&destination).unwrap();
        assert_eq!(inspected.total_prefix_tokens, 5);
        assert!(inspected
            .blocks
            .iter()
            .all(|block| block.rank == Some(rank)));
        drop(cache);
        drop(manager);
        assert!(resolve_prompt_cache_root(&destination)
            .unwrap()
            .join("manifest.json")
            .is_file());

        let mut incompatible = descriptor.clone();
        incompatible.topology.pipeline = Some((2, 0));
        let identity = PromptCacheModelIdentity {
            model_family: descriptor.model_family.clone(),
            effective_model_type: descriptor.effective_model_type.clone(),
            architecture_fingerprint: descriptor.architecture_fingerprint.clone(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            state_segments: descriptor.state_segments.clone(),
            topology: descriptor.topology.clone(),
            layer_layout: descriptor.layer_layout.clone(),
        };
        assert!(open_prompt_cache(
            &destination,
            &incompatible,
            &identity,
            &tokens,
            options.clone()
        )
        .is_err());

        let (loaded_manager, loaded_manifest) =
            open_prompt_cache(&destination, &descriptor, &identity, &tokens, options).unwrap();
        assert_eq!(loaded_manifest.blocks.len(), 3);
        let mut restored =
            PagedKeyValueCache::new_with_layout(loaded_manager.clone(), 0, None, 0, Some(rank))
                .unwrap();
        assert_eq!(restored.offset(), 5);
        let suffix = Array::from_slice(&[5.0f32], &[1, 1, 1, 1]);
        restored
            .update_and_fetch(suffix.clone(), suffix, stream)
            .unwrap();
        assert_eq!(restored.offset(), 6);
        assert_eq!(loaded_manager.report().unwrap().prompt_cache_loads, 1);

        let manifest_path = resolve_prompt_cache_root(&destination)
            .unwrap()
            .join("manifest.json");
        let mut corrupted: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        corrupted["blocks"][0]["logical_bytes"] = serde_json::json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&corrupted).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(&destination).is_err());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn distinct_window_schedule_save_reopen_continue_preserves_ranges() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true)
            .with_persistence_retention(true);
        let manager = CacheResidencyManager::new(options.clone()).unwrap();
        let windows = [None, Some(3), Some(5), Some(2)];
        let mut caches = windows
            .iter()
            .enumerate()
            .map(|(layer, window)| {
                PagedKeyValueCache::new(manager.clone(), layer, *window).unwrap()
            })
            .collect::<Vec<_>>();
        let prefix = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 6, 1]);
        for cache in &mut caches {
            cache
                .update_and_fetch(prefix.clone(), prefix.clone(), stream)
                .unwrap();
            cache.finalize().unwrap();
        }
        let retained_before = windows
            .iter()
            .enumerate()
            .map(|(layer, _)| {
                manager
                    .layer_block_ids(layer, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
                    .unwrap()
                    .into_iter()
                    .map(|id| (id.start, id.end))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(retained_before, vec![vec![(0, 2), (2, 4), (4, 6)]; 4]);
        let layer_layout = PromptCacheModelIdentity::key_value_layouts(windows, 1, 1).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "schedule-fixture".into(),
            effective_model_type: "schedule-fixture".into(),
            checkpoint_fingerprint: "checkpoint".into(),
            prefix_content_fingerprint: "tokens:0..6".into(),
            architecture_fingerprint: "architecture".into(),
            layer_count: windows.len(),
            global_layer_start: 0,
            global_layer_end: windows.len(),
            batch_size: 1,
            layer_prefix_offsets: vec![0; windows.len()],
            state_segments: vec![eredu_core::cache::PromptCacheStateSegment::new(
                "state",
                0..windows.len(),
            )
            .unwrap()],
            layer_layout: layer_layout.clone(),
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let tokens = [10, 11, 12, 13, 14, 15];
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("distinct-windows");
        manager
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &tokens,
                &[],
                &PromptCacheOptions::default(),
            )
            .unwrap();

        let query = Array::from_slice(&[1.0f32], &[1, 1, 1, 1]);
        let suffix = Array::from_slice(&[6.0f32], &[1, 1, 1, 1]);
        let mut uninterrupted = Vec::new();
        for (cache, window) in caches.iter_mut().zip(windows) {
            cache
                .update_for_attention(suffix.clone(), suffix.clone(), stream)
                .unwrap();
            let output = cache
                .paged_attention(&query, 1.0, None, None, stream)
                .unwrap()
                .unwrap()
                .evaluated()
                .unwrap()
                .item::<f32>();
            let expected_start = window.map_or(0, |window| 6 - i64::from(window - 1));
            let normalizer = (expected_start..=6)
                .map(|value| ((value - 6) as f32).exp())
                .sum::<f32>();
            let expected = (expected_start..=6)
                .map(|value| value as f32 * ((value - 6) as f32).exp())
                .sum::<f32>()
                / normalizer;
            assert!((output - expected).abs() < 1e-5);
            uninterrupted.push(output);
        }
        drop(caches);
        drop(manager);

        let identity = PromptCacheModelIdentity {
            model_family: descriptor.model_family.clone(),
            effective_model_type: descriptor.effective_model_type.clone(),
            architecture_fingerprint: descriptor.architecture_fingerprint.clone(),
            layer_count: windows.len(),
            global_layer_start: 0,
            global_layer_end: windows.len(),
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; windows.len()],
            state_segments: descriptor.state_segments.clone(),
            topology: PromptCacheTopology::default(),
            layer_layout,
        };
        let (manager, manifest) =
            open_prompt_cache(&destination, &descriptor, &identity, &tokens, options).unwrap();
        assert_eq!(manifest.layer_layout, descriptor.layer_layout);

        for (layer, window) in windows.into_iter().enumerate() {
            let retained_after = manager
                .layer_block_ids(layer, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
                .unwrap()
                .into_iter()
                .map(|id| (id.start, id.end))
                .collect::<Vec<_>>();
            assert_eq!(retained_after, retained_before[layer]);
            let mut restored = PagedKeyValueCache::new(manager.clone(), layer, window).unwrap();
            assert_eq!(restored.offset(), 6);
            restored
                .update_for_attention(suffix.clone(), suffix.clone(), stream)
                .unwrap();
            let output = restored
                .paged_attention(&query, 1.0, None, None, stream)
                .unwrap()
                .unwrap()
                .evaluated()
                .unwrap()
                .item::<f32>();
            assert!((output - uninterrupted[layer]).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn live_disk_budget_demotes_and_drop_removes_ephemeral_blocks() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let directory = tempfile::tempdir().unwrap();
        // MLX host-transfer buffers are page-aligned; permit one physical
        // allocation while keeping the logical device budget at two blocks.
        let options = PagedCacheOptions::new(2, 64, 32 * 1024, 1)
            .unwrap()
            .with_full_attention(true)
            .with_live_disk(directory.path(), 4096, 2)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let states = Array::from_slice(
            &[
                0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
            ],
            &[1, 1, 6, 2],
        );
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();
        let mut report = manager.report().unwrap();
        for _ in 0..100 {
            if report.disk_demotions == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            report = manager.report().unwrap();
        }
        assert_eq!(report.disk_blocks, 1);
        assert_eq!(report.disk_demotions, 1);
        assert!(fs::read_dir(directory.path()).unwrap().next().is_some());
        drop(cache);
        drop(manager);
        assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
    }
}
