//! Block-addressable key/value cache storage and scans.

use super::*;

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
    pub(super) tail_keys: Option<Array>,
    tail_values: Option<Array>,
    pub(super) tail_start: i64,
    pub(super) offset: i64,
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

    pub(crate) fn rebind_paging_manager(&mut self, manager: CacheResidencyManager) {
        if let Self::Paged(cache) = self {
            cache.rebind_paging_manager(manager);
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
                *cache = previous.deep_clone_state()?;
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

    /// Snapshots append-only local state while retaining its exact array views.
    /// Appends replace tails rather than mutating them, so the shared views are
    /// immutable for the lifetime of the checkpoint.
    pub fn checkpoint_clone_state(&self) -> Self {
        self.clone()
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

    pub(crate) fn rebind_paging_manager(&mut self, manager: CacheResidencyManager) {
        self.manager = manager;
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
            return Err(Exception::custom(format!(
                "key/value cache checkpoint does not belong to the same paged layer: current \
                 session/layer/rank/key-only={}/{}/{:?}/{}, checkpoint={}/{}/{:?}/{}",
                self.manager.session_id(),
                self.global_layer,
                self.rank,
                self.key_only,
                checkpoint.manager.session_id(),
                checkpoint.global_layer,
                checkpoint.rank,
                checkpoint.key_only,
            )));
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

    pub(super) fn tail_len(&self) -> i32 {
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

    pub(super) fn append(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(), Exception> {
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
