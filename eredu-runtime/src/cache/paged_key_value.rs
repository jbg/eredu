//! Portable paged key/value history and attention scheduling.

use eredu_core::cache::{
    mechanisms::{
        LiveCacheBackend, PagedAttentionGeometry, PagedAttentionOperands, PagedKeyValueStorage,
    },
    CacheRankIdentity, CacheRepresentation,
};
use eredu_nn::{AttentionRequest, IndexedAttentionInput, RelativeAttentionInput, Tensor};

use super::live::{
    CacheBlockArrays, CacheResidencyError, CacheResidencyManager, CacheStreamCheckpoint,
    CacheStreamState, LiveCacheStream,
};
use crate::CacheResidencyReport;

type Storage<M> = <M as LiveCacheBackend>::Tensor;
type Queue<M> = <M as LiveCacheBackend>::Queue;

fn invalid(detail: impl Into<String>) -> CacheResidencyError {
    CacheResidencyError::ArrayMismatch(detail.into())
}

/// Owns the mutable tail and retained frontier of one paged key/value layer.
#[derive(Debug, Clone)]
pub struct PagedKeyValueCache<M: PagedKeyValueStorage> {
    manager: CacheResidencyManager<M>,
    global_layer: usize,
    rank: Option<CacheRankIdentity>,
    sliding_window: Option<i32>,
    key_only: bool,
    prefix_tokens: i32,
    cursor: CacheStreamState<Storage<M>>,
    retained_history: Option<std::sync::Arc<super::live::CacheHistoryRetention>>,
}

impl<M: PagedKeyValueStorage> PagedKeyValueCache<M> {
    /// Creates one layer cache attached to a model-wide manager.
    pub fn new(
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        sliding_window: Option<i32>,
    ) -> Result<Self, CacheResidencyError> {
        Self::new_with_layout(manager, global_layer, sliding_window, 0, None)
    }

    /// Copies every logical tail element, including strided head/token views.
    /// Immutable sealed blocks still belong to the existing manager; independent
    /// branches rebind this layer to their copied manager before advancing.
    pub fn deep_clone_state(&self, stream: &Queue<M>) -> Result<Self, CacheResidencyError> {
        let mut snapshot = self.clone();
        if let Some([keys, values]) = &self.cursor.tail {
            snapshot.cursor.tail = Some([
                M::copy_tensor(keys, stream)?,
                M::copy_tensor(values, stream)?,
            ]);
        }
        Ok(snapshot)
    }

    /// Snapshots append-only local state while retaining its exact array views.
    /// Appends replace tails rather than mutating them, so the shared views are
    /// immutable for the lifetime of the checkpoint.
    pub fn checkpoint_clone_state(&self) -> Result<Self, CacheResidencyError> {
        let mut checkpoint = self.clone();
        if let Some(window) = self.sliding_window {
            let start = (self.cursor.offset - i64::from(window)).max(0);
            if start < self.cursor.tail_start {
                checkpoint.retained_history = Some(self.manager.retain_history(
                    self.global_layer,
                    CacheRepresentation::KeyValue,
                    start,
                    self.cursor.tail_start,
                )?);
            }
        }
        Ok(checkpoint)
    }

    /// Captures the shared-manager frontier required to discard one
    /// sequential speculative branch.
    ///
    /// Sliding paging is excluded because advancing it may irreversibly
    /// discard immutable blocks before the branch is committed. Full-attention
    /// paging is append-only, so rollback removes only newly sealed blocks and
    /// restores the canonical mutable-tail accounting.
    pub fn transaction_checkpoint(&self) -> Result<CacheStreamCheckpoint, CacheResidencyError> {
        self.manager.checkpoint_stream_append(
            &self.live_stream(),
            &self.cursor,
            self.sliding_window.is_some(),
        )
    }

    /// Discards a sequential append and reinstates its original accounting.
    pub fn rollback_transaction(
        &mut self,
        checkpoint: &CacheStreamCheckpoint,
    ) -> Result<(), CacheResidencyError> {
        self.manager
            .rollback_stream_append(&self.live_stream(), checkpoint)
    }

    /// Creates one layer cache with pinned-prefix and rank-local identity.
    pub fn new_with_layout(
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, CacheResidencyError> {
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
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, CacheResidencyError> {
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
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        sliding_window: Option<i32>,
        prefix_tokens: i32,
        rank: Option<CacheRankIdentity>,
        key_only: bool,
    ) -> Result<Self, CacheResidencyError> {
        let cursor = manager.open_stream(
            &LiveCacheStream {
                global_layer,
                representation: CacheRepresentation::KeyValue,
                rank,
                protected_prefix: i64::from(prefix_tokens),
            },
            sliding_window.map(i64::from),
        )?;
        Ok(Self {
            manager,
            global_layer,
            rank,
            sliding_window,
            key_only,
            prefix_tokens,
            cursor,
            retained_history: None,
        })
    }

    /// Returns the shared model-wide residency manager.
    pub const fn manager(&self) -> &CacheResidencyManager<M> {
        &self.manager
    }

    /// Binds the tail to an independently forked page catalog.
    pub fn rebind_paging_manager(&mut self, manager: CacheResidencyManager<M>) {
        self.manager = manager;
        self.retained_history = None;
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
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        self.manager.report()
    }

    /// Resets the local cursor after a model-wide catalog clear.
    pub fn reset_local_after_manager_clear(&mut self) {
        self.cursor = CacheStreamState::at_offset(0);
    }

    /// Seals a partially filled tail so it is safe to persist.
    pub fn finalize(&mut self) -> Result<(), CacheResidencyError> {
        self.seal_tail()
    }

    /// Truncates this layer to an absolute token length.
    pub fn truncate(&mut self, len: i64, stream: &Queue<M>) -> Result<(), CacheResidencyError> {
        self.manager
            .truncate_stream(&self.live_stream(), &mut self.cursor, len, stream)
    }

    /// Restores an earlier speculative frontier and atomically removes blocks
    /// sealed after it.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        if !self.has_same_transaction_identity(checkpoint) {
            return Err(invalid(format!(
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
        self.manager.restore_stream(
            &self.live_stream(),
            &mut self.cursor,
            &checkpoint.cursor,
            stream,
        )?;
        // Only the saved frontier retains old history. The resumed live state
        // must allow it to expire when that checkpoint is released.
        self.retained_history = None;
        Ok(())
    }

    /// Clears live state while preserving paging configuration.
    pub fn clear(&mut self) -> Result<(), CacheResidencyError> {
        self.manager
            .clear_stream(&self.live_stream(), &mut self.cursor)?;
        self.retained_history = None;
        Ok(())
    }

    /// Returns the number of positions in the mutable tail.
    pub fn tail_len(&self) -> i32 {
        self.cursor
            .tail
            .as_ref()
            .map_or(0, |[keys, _]| M::shape(keys)[2])
    }

    fn live_stream(&self) -> LiveCacheStream {
        LiveCacheStream {
            global_layer: self.global_layer,
            representation: CacheRepresentation::KeyValue,
            rank: self.rank,
            protected_prefix: i64::from(self.prefix_tokens),
        }
    }

    fn seal_tail(&mut self) -> Result<(), CacheResidencyError> {
        self.manager
            .seal_stream(&self.live_stream(), &mut self.cursor)
    }

    fn normalize_update_values(
        &self,
        keys: &Storage<M>,
        values: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<Storage<M>, CacheResidencyError> {
        if !self.key_only {
            return Ok(values);
        }
        if M::shape(keys).len() != 4
            || M::shape(&values).len() != 4
            || M::shape(keys)[..3] != M::shape(&values)[..3]
            || M::shape(&values)[3] != 0
            || M::dtype(keys) != M::dtype(&values)
        {
            return Err(invalid(
                "paged key-only cache expects matching rank-4 keys and zero-width values",
            ));
        }
        let mut shape = M::shape(keys).to_vec();
        shape[3] = 1;
        Ok(M::zeroed_cache_tensor(&shape, keys, stream)?)
    }

    fn validate_update(
        &self,
        keys: &Storage<M>,
        values: &Storage<M>,
    ) -> Result<(), CacheResidencyError> {
        if M::shape(keys).len() != 4 || M::shape(values).len() != 4 {
            return Err(invalid(
                "paged key/value cache expects rank-4 [batch, heads, sequence, dimension] arrays",
            ));
        }
        let geometry_matches = if self.key_only {
            M::shape(keys)[..3] == M::shape(values)[..3] && M::shape(values)[3] == 1
        } else {
            M::shape(keys) == M::shape(values)
        };
        if !geometry_matches || M::dtype(keys) != M::dtype(values) {
            return Err(invalid(
                "paged key/value cache updates must have identical key and value shapes and dtypes",
            ));
        }
        if M::shape(keys)[2] <= 0 {
            return Err(invalid(
                "paged key/value cache update must contain at least one token",
            ));
        }
        Ok(())
    }

    /// Appends keys and values and discards sliding history past the new window.
    pub fn append(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        let values = self.normalize_update_values(&keys, values, stream)?;
        self.append_normalized(keys, values, false, stream)
    }

    fn append_normalized(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        retain_for_attention: bool,
        stream: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        self.validate_update(&keys, &values)?;
        let page = CacheBlockArrays::KeyValue { keys, values };
        self.manager.append_pages(
            &self.live_stream(),
            &mut self.cursor,
            page,
            self.sliding_window
                .filter(|_| !retain_for_attention)
                .map(i64::from),
            stream,
        )
    }

    fn discard_sliding_history(&self) -> Result<(), CacheResidencyError> {
        if let Some(window) = self.sliding_window {
            self.manager.discard_stream_history(
                &self.live_stream(),
                self.cursor.offset,
                i64::from(window),
            )?;
        }
        Ok(())
    }

    /// Returns the absolute frontier, saturated to the neural position type.
    pub fn offset(&self) -> i32 {
        i32::try_from(self.cursor.offset).unwrap_or(i32::MAX)
    }

    /// Returns the sliding window, when one is installed.
    pub fn max_size(&self) -> Option<i32> {
        self.sliding_window
    }

    /// Returns the native roots of the current mutable tail.
    pub fn retained_arrays(&self) -> Vec<&Storage<M>> {
        self.arrays().collect()
    }

    pub(crate) fn arrays(&self) -> impl Iterator<Item = &Storage<M>> {
        self.cursor.tail.iter().flatten()
    }

    /// Appends the submitted span and retains history needed by its queries.
    pub fn update_for_attention(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), CacheResidencyError> {
        let submitted = (keys.clone(), values.clone());
        let values = self.normalize_update_values(&keys, values, stream)?;
        self.append_normalized(keys, values, true, stream)?;
        Ok(submitted)
    }

    /// Appends a span and returns the visible history for a sliding cache.
    /// Full-context callers receive the submitted span and scan the cache separately.
    pub fn update_and_fetch(
        &mut self,
        keys: Storage<M>,
        values: Storage<M>,
        stream: &Queue<M>,
    ) -> Result<(Storage<M>, Storage<M>), CacheResidencyError> {
        if let Some(window) = self.sliding_window {
            let values = self.normalize_update_values(&keys, values, stream)?;
            self.validate_update(&keys, &values)?;
            let page = CacheBlockArrays::KeyValue { keys, values };
            let [keys, values] = self.manager.append_with_visible_history(
                &self.live_stream(),
                &mut self.cursor,
                page,
                i64::from(window),
                stream,
            )?;
            return Ok((keys, values));
        }
        let previous_offset = self.cursor.offset;
        let update_len = M::shape(&keys).get(2).copied().map(i64::from);
        let fallback_keys = keys.clone();
        let fallback_values = values.clone();
        self.append(keys, values, stream)?;
        if update_len == Some(self.cursor.offset - previous_offset) {
            Ok((fallback_keys, fallback_values))
        } else {
            Err(invalid("paged cache offset changed unexpectedly"))
        }
    }
}

impl<M: PagedKeyValueStorage> PagedKeyValueCache<M> {
    /// Reads retained history and the submitted transient suffix without mutation.
    pub fn read_transient(
        &self,
        request: AttentionRequest<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<Storage<M>, CacheResidencyError> {
        request
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        self.attend(
            Self::operands(&request),
            Some([
                M::storage_ref(&request.keys),
                M::storage_ref(&request.values),
            ]),
            queue,
        )
    }

    fn operands<'a>(
        request: &'a AttentionRequest<'_, M::Value>,
    ) -> PagedAttentionOperands<'a, Storage<M>> {
        PagedAttentionOperands {
            queries: M::storage_ref(&request.queries),
            scale: request.scale,
            mask: request.mask.map(M::storage_ref),
            sinks: request.sinks.map(M::storage_ref),
            softcap: request.softcap,
            arithmetic: request.arithmetic,
        }
    }

    /// Runs attention over the exact visible pages and optional transient suffix.
    pub fn attend(
        &self,
        operands: PagedAttentionOperands<'_, Storage<M>>,
        transient: Option<[&Storage<M>; 2]>,
        queue: &Queue<M>,
    ) -> Result<Storage<M>, CacheResidencyError> {
        let tokens = M::shape(operands.queries)
            .get(2)
            .copied()
            .ok_or_else(|| invalid("paged attention queries require a token axis"))?;
        let read = self.plan_read(
            tokens,
            transient.map(|[keys, _]| i64::from(M::shape(keys)[2])),
        )?;
        let geometry = self.geometry(&read, transient.is_some());
        let kernel = M::paged_attention(operands, geometry, queue)?;
        self.manager.run_attention_scan(
            &self.live_stream(),
            &self.cursor,
            read,
            transient,
            self.sliding_window.map(i64::from),
            queue,
            kernel,
        )
    }

    /// Runs relative attention using retained history and absolute query positions.
    pub fn attend_relative(
        &self,
        input: &RelativeAttentionInput<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<Storage<M>, CacheResidencyError> {
        input
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let tokens = input.queries.dim(2);
        if i64::from(input.query_offset) != self.cursor.offset - i64::from(tokens)
            || input.window != self.sliding_window
            || self.prefix_tokens != 0
        {
            return Err(invalid(
                "relative attention request differs from retained cache geometry",
            ));
        }
        let read = self.plan_read(tokens, None)?;
        let kernel = M::paged_relative_attention(input, self.geometry(&read, false), queue)?;
        self.manager.run_attention_scan(
            &self.live_stream(),
            &self.cursor,
            read,
            None,
            self.sliding_window.map(i64::from),
            queue,
            kernel,
        )
    }

    fn plan_read(
        &self,
        tokens: i32,
        transient: Option<i64>,
    ) -> Result<super::live::CacheAttentionRead, CacheResidencyError> {
        if self.key_only {
            return Err(invalid(
                "key-only paged caches require architecture-owned attention",
            ));
        }
        self.manager.plan_attention_read(
            &self.live_stream(),
            self.cursor.offset,
            i64::from(tokens),
            transient,
            self.sliding_window.map(i64::from),
        )
    }

    fn geometry(
        &self,
        read: &super::live::CacheAttentionRead,
        transient: bool,
    ) -> PagedAttentionGeometry {
        PagedAttentionGeometry {
            query_start: read.query_start,
            context_end: read.context_end,
            window: self.sliding_window,
            prefix: i64::from(self.prefix_tokens),
            transient_context_start: transient.then_some(read.visible_start),
        }
    }

    /// Reads only the unique positions requested for each query and completes
    /// that query before admitting its successor.
    pub fn attend_indexed(
        &self,
        input: &IndexedAttentionInput<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<Storage<M>, CacheResidencyError> {
        input
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let mut batches = Vec::new();
        let mut scratch = 0;
        let stream = self.live_stream();
        for batch in 0..input.queries.dim(0) {
            let mut rows = Vec::new();
            for query in 0..input.queries.dim(2) {
                let positions = M::indexed_positions(input, batch, query, queue)?;
                let (unique, slots) = eredu_nn::compact_indexed_positions(&positions);
                let gathered = if unique.is_empty() {
                    None
                } else {
                    let (gathered, copied_scratch) = self.manager.read_indexed_stream(
                        &stream,
                        &self.cursor,
                        batch,
                        &unique,
                        queue,
                    )?;
                    scratch = scratch.max(copied_scratch);
                    Some(gathered)
                };
                rows.push(M::attend_indexed_row(
                    input, batch, query, &slots, gathered, queue,
                )?);
            }
            batches.push(M::concatenate_tokens(
                &rows.iter().collect::<Vec<_>>(),
                2,
                queue,
            )?);
        }
        let output = M::concatenate_tokens(&batches.iter().collect::<Vec<_>>(), 0, queue)?;
        self.discard_sliding_history()?;
        self.manager
            .record_selected_attention(self.global_layer, 0, 0, scratch)?;
        Ok(output)
    }

    /// Exact retained frontier as an absolute token position.
    pub const fn frontier(&self) -> i64 {
        self.cursor.offset
    }

    /// Absolute start of the current mutable tail.
    pub const fn tail_start(&self) -> i64 {
        self.cursor.tail_start
    }
}

impl<B, M> crate::RuntimeLayerState<B> for PagedKeyValueCache<M>
where
    B: eredu_nn::NeuralBackend<Tensor = M::Value>,
    M: PagedKeyValueStorage,
{
    type RetainedValues<'a> = std::iter::Flatten<std::array::IntoIter<Option<&'a M::Value>, 2>>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.cursor
            .tail
            .as_ref()
            .map_or([None, None], |[keys, values]| {
                [Some(M::borrow_value(keys)), Some(M::borrow_value(values))]
            })
            .into_iter()
            .flatten()
    }
}

impl<M: PagedKeyValueStorage> eredu_nn::AttentionCache<M::Value> for PagedKeyValueCache<M> {
    fn update_memory_contract(
        &self,
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error> {
        M::describe_paged_append(invocation)
    }

    fn uses_blockwise_attention(&self) -> bool {
        true
    }

    fn offset(&self) -> i32 {
        self.offset()
    }

    fn max_size(&self) -> Option<i32> {
        self.max_size()
    }

    fn update_for_attention(
        &mut self,
        keys: M::Value,
        values: M::Value,
        queue: &Queue<M>,
    ) -> Result<(M::Value, M::Value), eredu_nn::Error> {
        let (keys, values) = self
            .update_for_attention(M::into_storage(keys), M::into_storage(values), queue)
            .map_err(eredu_nn::Error::backend_source)?;
        Ok((M::into_value(keys), M::into_value(values)))
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        request.validate()?;
        self.attend(Self::operands(&request), None, queue)
            .map(M::into_value)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn indexed_attention<B: eredu_nn::NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: IndexedAttentionInput<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        self.attend_indexed(&request, queue)
            .map(M::into_value)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn relative_attention<B: eredu_nn::NeuralBackend<Tensor = M::Value>>(
        &mut self,
        request: RelativeAttentionInput<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        self.attend_relative(&request, queue)
            .map(M::into_value)
            .map_err(eredu_nn::Error::backend_source)
    }
}

impl<M: PagedKeyValueStorage> eredu_nn::ReadOnlyAttentionCache<M::Value> for PagedKeyValueCache<M> {
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, M::Value>,
        queue: &Queue<M>,
    ) -> Result<M::Value, eredu_nn::Error> {
        self.read_transient(request, queue)
            .map(M::into_value)
            .map_err(eredu_nn::Error::backend_source)
    }
}
