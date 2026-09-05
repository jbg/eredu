//! Contiguous and paged compressed-latent cache storage.

use super::*;

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

    pub(crate) fn rebind_paging_manager(&mut self, manager: CacheResidencyManager) {
        if let Some(paged) = self.paged.as_deref_mut() {
            paged.manager = manager;
        }
    }

    #[cfg(test)]
    pub(super) fn paged_block_ids(&self) -> Result<Option<Vec<CacheBlockId>>, Exception> {
        self.paged
            .as_deref()
            .map(PagedCompressedLatentCache::block_ids)
            .transpose()
    }

    #[cfg(test)]
    pub(super) fn paged_tail_block(&self) -> Option<PagedLatentAttentionBlock> {
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
        *self = checkpoint.deep_clone_state()?;
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
