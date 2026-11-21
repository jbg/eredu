//! Contiguous and paged compressed-latent cache storage.

use eredu_core::{
    cache::{mechanisms::CompressedCacheStorage, CacheRankIdentity, CacheRepresentation},
    BackendFailure,
};
use eredu_nn::{
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionHistory,
    CompressedAttentionScan, CompressedAttentionState, CompressedAttentionView,
    Error as ComputeError, NeuralBackend,
};

use super::live::{CacheBlockArrays, CacheResidencyManager, CacheStreamState, LiveCacheStream};

type Storage<M> = <M as eredu_core::cache::mechanisms::LiveCacheBackend>::Tensor;
type Queue<M> = <M as eredu_core::cache::mechanisms::LiveCacheBackend>::Queue;

fn invalid(detail: &'static str) -> BackendFailure {
    BackendFailure::from_error(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        detail,
    ))
}

const COMPRESSED_LATENT_CACHE_STEP: i32 = 256;

/// Compressed attention cache that stores one latent KV vector and one rotary
/// key vector per token, independent of the number of attention heads.
///
/// This representation is used by Multi-head Latent Attention (MLA). Arrays
/// have shape `[batch, sequence, dimension]`; head-specific keys and values are
/// reconstructed transiently by the attention implementation.
#[derive(Debug, Clone)]
pub struct CompressedLatentCache<M: CompressedCacheStorage> {
    latent_storage: Option<Storage<M>>,
    rotary_key_storage: Option<Storage<M>>,
    latent: Option<Storage<M>>,
    rotary_key: Option<Storage<M>>,
    offset: i32,
    length: i32,
    capacity: i32,
    step: i32,
    paged: Option<Box<PagedCompressedLatentCache<M>>>,
}

impl<M: CompressedCacheStorage> Default for CompressedLatentCache<M> {
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

impl<M: CompressedCacheStorage> CompressedLatentCache<M> {
    /// Creates an empty compressed latent cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bounds retained token storage after an admitted continuation, including
    /// resident allocation chunks and the paged tail's full input span.
    pub fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        if let Some(paged) = self.paged.as_deref() {
            return u64::try_from(paged.cursor.offset)
                .ok()?
                .checked_add(additional);
        }
        let required = u64::try_from(self.length).ok()?.checked_add(additional)?;
        let step = u64::try_from(self.step).ok().filter(|step| *step > 0)?;
        let padded = required
            .checked_add(step - 1)?
            .checked_div(step)?
            .checked_mul(step)?;
        Some(padded.max(u64::try_from(self.capacity).ok()?))
    }

    /// Copies logical state into independent compact buffers. Resident padding
    /// is discarded so snapshot storage matches the retained logical arrays;
    /// continuation accounting separately includes future chunk allocation.
    /// Paged callers must bind the copied tail to an independently copied manager.
    pub fn isolated_snapshot(&self, stream: &Queue<M>) -> Result<Self, BackendFailure> {
        let copy = |array: &Option<Storage<M>>| {
            array
                .as_ref()
                .map(|array| M::copy_tensor(array, stream))
                .transpose()
        };
        let mut snapshot = self.clone();
        if let Some(paged) = snapshot.paged.as_deref_mut() {
            if let Some([latent, rotary]) = &mut paged.cursor.tail {
                *latent = M::copy_tensor(latent, stream)?;
                *rotary = M::copy_tensor(rotary, stream)?;
            }
        } else {
            snapshot.latent_storage = copy(&self.latent)?;
            snapshot.rotary_key_storage = copy(&self.rotary_key)?;
            snapshot.latent = snapshot.latent_storage.clone();
            snapshot.rotary_key = snapshot.rotary_key_storage.clone();
            snapshot.capacity = snapshot.length;
        }
        Ok(snapshot)
    }

    /// Forks mutable cache state while retaining the immutable paging catalog.
    ///
    /// Resident storage and a paged mutable tail receive independent native
    /// buffers. Sealed paged blocks remain shared through the residency
    /// manager because they are immutable and addressed by token range.
    pub fn deep_clone_state(&self) -> Result<Self, BackendFailure> {
        let clone_array =
            |array: &Option<Storage<M>>| array.as_ref().map(M::fork_tensor).transpose();
        let mut clone = self.clone();
        clone.latent_storage = clone_array(&self.latent_storage)?;
        clone.rotary_key_storage = clone_array(&self.rotary_key_storage)?;
        clone.latent = clone_array(&self.latent)?;
        clone.rotary_key = clone_array(&self.rotary_key)?;
        if let (Some(clone), Some(source)) = (clone.paged.as_deref_mut(), self.paged.as_deref()) {
            clone.cursor.tail = source
                .cursor
                .tail
                .as_ref()
                .map(|[latent, rotary]| {
                    Ok::<_, BackendFailure>([M::fork_tensor(latent)?, M::fork_tensor(rotary)?])
                })
                .transpose()?;
        }
        Ok(clone)
    }

    /// Clears local arrays after the shared paging manager has been cleared.
    pub fn reset_local_after_manager_clear(&mut self) {
        if let Some(paged) = self.paged.as_deref_mut() {
            paged.cursor = CacheStreamState::at_offset(0);
        } else {
            *self = Self::default();
        }
    }

    /// Creates compressed-latent paging under a shared model-wide manager.
    pub fn new_paged(
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, BackendFailure> {
        if !manager.options().full_attention_enabled() {
            return Err(invalid(
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
    pub fn residency_manager(&self) -> Option<&CacheResidencyManager<M>> {
        self.paged.as_deref().map(|paged| &paged.manager)
    }

    /// Attaches copied page storage when publishing an isolated cache.
    pub fn rebind_paging_manager(&mut self, manager: CacheResidencyManager<M>) {
        if let Some(paged) = self.paged.as_deref_mut() {
            paged.manager = manager;
        }
    }

    /// Seals a partial compressed tail for safe persistence.
    pub fn finalize(&mut self) -> Result<(), BackendFailure> {
        match self.paged.as_deref_mut() {
            Some(paged) => paged.seal_tail(),
            None => Ok(()),
        }
    }

    /// Returns the number of cached tokens.
    pub fn offset(&self) -> i32 {
        self.paged.as_deref().map_or(self.offset, |paged| {
            i32::try_from(paged.cursor.offset).unwrap_or(i32::MAX)
        })
    }

    /// Returns the allocated token capacity of the backing arrays.
    pub fn capacity(&self) -> i32 {
        self.paged
            .as_deref()
            .map_or(self.capacity, |paged| paged.tail_len())
    }

    /// Returns the retained latent and rotary-key arrays, when initialized.
    pub fn arrays(&self) -> Option<(&Storage<M>, &Storage<M>)> {
        if self.paged.is_some() {
            return None;
        }
        Some((self.latent.as_ref()?, self.rotary_key.as_ref()?))
    }

    /// Borrows the current resident values or unsealed paged tail.
    pub fn retained_arrays(&self) -> Vec<&Storage<M>> {
        match self.paged.as_deref() {
            Some(paged) => paged.cursor.tail.iter().flatten().collect(),
            None => self.latent.iter().chain(self.rotary_key.iter()).collect(),
        }
    }

    /// Clears all retained state.
    pub fn clear(&mut self) -> Result<(), BackendFailure> {
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
        stream: &Queue<M>,
    ) -> Result<(), BackendFailure> {
        let checkpoint = checkpoint.deep_clone_state()?;
        match (&mut self.paged, &checkpoint.paged) {
            (Some(current), Some(previous)) => {
                if current.global_layer != previous.global_layer
                    || current.manager.session_id() != previous.manager.session_id()
                {
                    return Err(invalid(
                        "compressed cache checkpoint does not belong to the same paged layer",
                    ));
                }
                current
                    .manager
                    .restore_stream(
                        &current.live_stream(),
                        &mut current.cursor,
                        &previous.cursor,
                        stream,
                    )
                    .map_err(BackendFailure::from_error)?;
            }
            (None, None) => {}
            _ => {
                return Err(invalid(
                    "compressed cache checkpoint changed residency mode",
                ));
            }
        }
        *self = checkpoint;
        Ok(())
    }

    /// Appends compressed states and returns the full states to attend over.
    pub fn update_and_fetch(
        &mut self,
        latent: Storage<M>,
        rotary_key: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), BackendFailure> {
        if let Some(paged) = self.paged.as_deref_mut() {
            let returned = (latent.clone(), rotary_key.clone());
            paged.append(latent, rotary_key, stream)?;
            return Ok(returned);
        }
        if M::shape(&latent).len() != 3 || M::shape(&rotary_key).len() != 3 {
            return Err(invalid(
                "compressed latent cache expects rank-3 [batch, sequence, dimension] arrays",
            ));
        }
        if M::shape(&latent)[0] != M::shape(&rotary_key)[0]
            || M::shape(&latent)[1] != M::shape(&rotary_key)[1]
        {
            return Err(invalid(
                "compressed latent and rotary-key cache updates must share batch and sequence dimensions",
            ));
        }
        if let (Some(previous_latent), Some(previous_rotary)) =
            (&self.latent_storage, &self.rotary_key_storage)
        {
            if M::shape(previous_latent)[0] != M::shape(&latent)[0]
                || M::shape(previous_latent)[2] != M::shape(&latent)[2]
                || M::shape(previous_rotary)[0] != M::shape(&rotary_key)[0]
                || M::shape(previous_rotary)[2] != M::shape(&rotary_key)[2]
            {
                return Err(invalid(
                    "compressed latent cache update dimensions do not match retained state",
                ));
            }
        }

        let new_tokens = M::shape(&latent)[1];
        let required = self
            .length
            .checked_add(new_tokens)
            .ok_or_else(|| invalid("compressed cache length exceeds i32"))?;
        let offset = self
            .offset
            .checked_add(new_tokens)
            .ok_or_else(|| invalid("compressed cache frontier exceeds i32"))?;
        let capacity = if required > self.capacity {
            required
                .checked_add(self.step - 1)
                .and_then(|rounded| (rounded / self.step).checked_mul(self.step))
                .ok_or_else(|| invalid("compressed cache capacity exceeds i32"))?
        } else {
            self.capacity
        };
        let reserve = |current: Option<&Storage<M>>, incoming: &Storage<M>| {
            let backing = match current {
                Some(backing) if capacity == self.capacity => backing.clone(),
                Some(backing) => {
                    let padding =
                        M::allocate_tokens(incoming, 1, capacity - self.capacity, stream)?;
                    M::concatenate_tokens(&[backing, &padding], 1, stream)?
                }
                None if capacity == required => return Ok(incoming.clone()),
                None => M::allocate_tokens(incoming, 1, capacity, stream)?,
            };
            M::write_tokens(backing, 1, self.length..required, incoming, stream)
        };
        let latent_storage = reserve(self.latent_storage.as_ref(), &latent)?;
        let rotary_storage = reserve(self.rotary_key_storage.as_ref(), &rotary_key)?;
        let latent = M::slice_tokens(&latent_storage, 1, 0..required, stream)?;
        let rotary = M::slice_tokens(&rotary_storage, 1, 0..required, stream)?;
        self.latent_storage = Some(latent_storage);
        self.rotary_key_storage = Some(rotary_storage);
        self.latent = Some(latent.clone());
        self.rotary_key = Some(rotary.clone());
        self.length = required;
        self.offset = offset;
        self.capacity = capacity;
        Ok((latent, rotary))
    }
}

impl<B, M> crate::RuntimeLayerState<B> for CompressedLatentCache<M>
where
    M: CompressedCacheStorage,
    B: NeuralBackend<Tensor = M::Value>,
{
    type RetainedValues<'a> =
        std::iter::Map<std::vec::IntoIter<&'a Storage<M>>, fn(&Storage<M>) -> &M::Value>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.retained_arrays().into_iter().map(M::borrow_value)
    }
}

impl<M: CompressedCacheStorage> CompressedAttentionCache<M::Value> for CompressedLatentCache<M> {
    type Checkpoint = Self;
    type History = Self;

    fn history(&self) -> Option<&Self::History> {
        Some(self)
    }

    fn offset(&self) -> i32 {
        CompressedLatentCache::offset(self)
    }

    fn is_paged(&self) -> bool {
        CompressedLatentCache::is_paged(self)
    }

    fn append(
        &mut self,
        state: CompressedAttentionState<M::Value>,
        context: &Queue<M>,
    ) -> Result<CompressedAttentionView<M::Value>, ComputeError> {
        let appended = state.clone();
        let (latent, rotary) = self
            .update_and_fetch(
                M::into_storage(state.latent),
                M::into_storage(state.rotary),
                context,
            )
            .map_err(ComputeError::backend_source)?;
        if self.is_paged() {
            Ok(CompressedAttentionView::Paged { appended })
        } else {
            Ok(CompressedAttentionView::Resident(
                CompressedAttentionState {
                    latent: M::into_value(latent),
                    rotary: M::into_value(rotary),
                },
            ))
        }
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &Queue<M>,
    ) -> Result<(), ComputeError> {
        self.restore_checkpoint(checkpoint, context)
            .map_err(ComputeError::backend_source)
    }

    fn finalize(&mut self) -> Result<(), ComputeError> {
        CompressedLatentCache::finalize(self).map_err(ComputeError::backend_source)
    }

    fn clear(&mut self) -> Result<(), ComputeError> {
        CompressedLatentCache::clear(self).map_err(ComputeError::backend_source)
    }
}

impl<M: CompressedCacheStorage> CompressedAttentionHistory<M::Value> for CompressedLatentCache<M> {
    fn scan_blocks<F>(
        &self,
        query_tokens: i32,
        context: &Queue<M>,
        visitor: F,
    ) -> Result<CompressedAttentionScan, ComputeError>
    where
        F: FnMut(CompressedAttentionBlock<M::Value>) -> Result<u64, ComputeError>,
    {
        let paged = self
            .paged
            .as_deref()
            .ok_or_else(|| ComputeError::backend("compressed block scan requires paged state"))?;
        paged.manager.scan_compressed_history(
            &paged.live_stream(),
            &paged.cursor,
            query_tokens,
            context,
            M::into_value,
            visitor,
        )
    }
}

#[derive(Debug, Clone)]
struct PagedCompressedLatentCache<M: CompressedCacheStorage> {
    manager: CacheResidencyManager<M>,
    global_layer: usize,
    rank: Option<CacheRankIdentity>,
    cursor: CacheStreamState<Storage<M>>,
}

impl<M: CompressedCacheStorage> PagedCompressedLatentCache<M> {
    fn new(
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, BackendFailure> {
        let cursor = manager
            .open_stream(
                &LiveCacheStream {
                    global_layer,
                    representation: CacheRepresentation::CompressedLatentRotary,
                    rank,
                    protected_prefix: 0,
                },
                None,
            )
            .map_err(BackendFailure::from_error)?;
        Ok(Self {
            manager,
            global_layer,
            rank,
            cursor,
        })
    }

    fn tail_len(&self) -> i32 {
        self.cursor
            .tail
            .as_ref()
            .map_or(0, |[latent, _]| M::shape(latent)[1])
    }

    fn append(
        &mut self,
        latent: Storage<M>,
        rotary: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(), BackendFailure> {
        self.manager
            .append_pages(
                &self.live_stream(),
                &mut self.cursor,
                CacheBlockArrays::CompressedLatentRotary {
                    latent,
                    rotary_key: rotary,
                },
                None,
                stream,
            )
            .map_err(BackendFailure::from_error)
    }

    fn live_stream(&self) -> LiveCacheStream {
        LiveCacheStream {
            global_layer: self.global_layer,
            representation: CacheRepresentation::CompressedLatentRotary,
            rank: self.rank,
            protected_prefix: 0,
        }
    }

    fn seal_tail(&mut self) -> Result<(), BackendFailure> {
        self.manager
            .seal_stream(&self.live_stream(), &mut self.cursor)
            .map_err(BackendFailure::from_error)
    }

    fn clear(&mut self) -> Result<(), BackendFailure> {
        self.manager
            .clear_stream(&self.live_stream(), &mut self.cursor)
            .map_err(BackendFailure::from_error)
    }
}
