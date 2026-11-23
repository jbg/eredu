//! Shared append, truncation, and tail sealing for key/value, latent, and record streams.

use super::*;
use eredu_core::cache::mechanisms::LiveCacheAppendBackend;
use std::ops::Range;

/// Address and visibility policy for one mutable stream in a live cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveCacheStream {
    /// Exact global layer owning the stream.
    pub global_layer: usize,
    /// Physical page representation and stream identity.
    pub representation: CacheRepresentation,
    /// Rank-local cache ownership, when partitioned.
    pub rank: Option<CacheRankIdentity>,
    /// Number of leading tokens protected from sliding eviction.
    pub protected_prefix: i64,
}

impl LiveCacheStream {
    /// Earliest past position required by the first query of an invocation.
    pub fn history_start(query_start: i64, sliding_window: Option<i64>) -> i64 {
        sliding_window.map_or(0, |window| {
            query_start
                .saturating_sub(window.saturating_sub(1).max(0))
                .max(0)
        })
    }
}

/// Exact source intervals for a contiguous range read.
struct CacheStreamRead {
    /// Sealed pages and their page-local slices in stream order.
    pub pages: Vec<(CacheBlockId, Range<i32>)>,
    /// Tail-local slice, when the range includes unsealed values.
    pub tail: Option<Range<i32>>,
}

/// Block-local row indices for an indexed attention read.
struct CacheIndexedRead {
    /// Sealed pages and their requested rows.
    pub pages: Vec<(CacheBlockId, Vec<i32>)>,
    /// Requested rows from the unsealed tail.
    pub tail: Vec<i32>,
}

/// Exact retained history and query positions for attention.
pub struct CacheAttentionRead {
    /// Sealed pages intersecting the retained history and protected prefix.
    pages: Vec<CacheBlockId>,
    /// Start of the first query in this invocation.
    pub query_start: i64,
    query_len: i64,
    /// Start of the retained sliding window.
    pub visible_start: i64,
    /// End including any ephemeral context supplied for this read.
    pub context_end: i64,
}

/// Runtime-owned tail and logical positions in one append stream.
#[derive(Debug, Clone)]
pub struct CacheStreamState<T> {
    /// Mutable tail tensors, present as a pair.
    pub tail: Option<[T; 2]>,
    /// Absolute start of the current tail.
    pub tail_start: i64,
    /// Absolute stream frontier.
    pub offset: i64,
}

/// Catalog and frontier retained for rollback of a sequential append.
#[derive(Debug)]
pub struct CacheStreamCheckpoint {
    session: u64,
    stream: LiveCacheStream,
    offset: i64,
    tail_bytes: u64,
    blocks: Vec<CacheBlockId>,
}

impl<T> CacheStreamState<T> {
    /// Starts a stream at a committed frontier with no unsealed tail.
    pub const fn at_offset(offset: i64) -> Self {
        Self {
            tail: None,
            tail_start: offset,
            offset,
        }
    }
}

impl<M: LiveCacheAppendBackend> CacheResidencyManager<M> {
    /// Appends native values, sealing full pages and restoring the original
    /// accounting when the append fails. Publishes the cursor
    /// after all page operations succeed. The caller owns exclusive mutation
    /// of this stream while the operation is in progress.
    pub fn append_pages(
        &self,
        stream: &LiveCacheStream,
        previous: &mut CacheStreamState<M::Tensor>,
        input: Page<M>,
        sliding_window: Option<i64>,
        queue: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        if previous.offset < 0
            || previous.tail_start < 0
            || stream.protected_prefix < 0
            || sliding_window.is_some_and(|window| window <= 0)
        {
            return Err(CacheResidencyError::ArrayMismatch(
                "cache append positions and visibility must be valid".into(),
            ));
        }
        if input.representation() != stream.representation {
            return Err(CacheResidencyError::ArrayMismatch(
                "append payload does not match the cache stream".into(),
            ));
        }
        let inputs = input.arrays();
        let axis = sequence_axis(stream.representation, M::shape(inputs[0]))?;
        let length = M::shape(inputs[0])[axis];
        if length <= 0 {
            return Err(CacheResidencyError::ArrayMismatch(
                "cache append must contain at least one position".into(),
            ));
        }
        validate_block_arrays::<M>(&input, i64::from(length))?;
        let block_size = self.options().block_size_tokens();
        if let Some(tail) = &previous.tail {
            for (retained, incoming) in tail.iter().zip(inputs) {
                let retained_shape = M::shape(retained);
                let incoming_shape = M::shape(incoming);
                if retained_shape.len() != incoming_shape.len()
                    || retained_shape.iter().zip(incoming_shape).enumerate().any(
                        |(dimension, (retained, incoming))| {
                            dimension != axis && retained != incoming
                        },
                    )
                    || M::dtype(retained) != M::dtype(incoming)
                    || !(1..block_size).contains(&retained_shape[axis])
                    || previous
                        .tail_start
                        .checked_add(i64::from(retained_shape[axis]))
                        != Some(previous.offset)
                {
                    return Err(CacheResidencyError::ArrayMismatch(
                        "cache append does not match the retained tail".into(),
                    ));
                }
            }
        }
        self.bind_transfer_device(queue)?;
        let end = previous
            .offset
            .checked_add(i64::from(length))
            .ok_or_else(|| {
                CacheResidencyError::Runtime("cache append frontier overflowed".into())
            })?;
        let mut sealed = Vec::new();
        let result = (|| {
            let mut tail = previous.tail.clone();
            let mut tail_start = previous.tail_start;
            let mut input_start = 0;
            let mut tail_len = tail.as_ref().map_or(0, |values| M::shape(&values[0])[axis]);
            while input_start < length {
                if tail.is_none() {
                    tail_start = previous.offset + i64::from(input_start);
                }
                let take = (block_size - tail_len).min(length - input_start);
                let input_end = input_start + take;
                let append = |index: usize| -> Result<M::Tensor, BackendFailure> {
                    let next = M::slice_tokens(inputs[index], axis, input_start..input_end, queue)?;
                    match &tail {
                        Some(values) => {
                            M::concatenate_tokens(&[&values[index], &next], axis, queue)
                        }
                        None => Ok(next),
                    }
                };
                let first = append(0)?;
                let first = if matches!(
                    stream.representation,
                    CacheRepresentation::AppendStream { .. }
                ) {
                    M::compact_tokens(&first, queue)?
                } else {
                    first
                };
                let second = append(1)?;
                let tail_end = tail_start + i64::from(tail_len + take);
                self.set_tail_state(
                    stream.global_layer,
                    stream.representation,
                    M::bytes(&first) + M::bytes(&second),
                    tail_end,
                )?;
                tail = if tail_len + take == block_size {
                    let block =
                        CacheBlockArrays::from_parts(stream.representation, [first, second]);
                    sealed.push(self.seal_pending_page(stream, tail_start, tail_end, block)?);
                    tail_start = tail_end;
                    tail_len = 0;
                    None
                } else {
                    tail_len += take;
                    Some([first, second])
                };
                input_start = input_end;
            }
            if let Some(window) = sliding_window {
                self.discard_stream_history(stream, end, window)?;
            }
            Ok(CacheStreamState {
                tail,
                tail_start,
                offset: end,
            })
        })();
        *previous = result.map_err(|operation| {
            let mut rollback = None;
            for id in sealed.iter().rev() {
                match self.remove_block(id) {
                    Ok(())
                    | Err(CacheResidencyError::Lifecycle(CacheLifecycleError::MissingBlock(_))) => {
                    }
                    Err(error) => {
                        rollback.get_or_insert(error);
                    }
                }
            }
            let prior_bytes = previous.tail.iter().flatten().map(M::bytes).sum();
            if let Err(error) = self.set_tail_state(
                stream.global_layer,
                stream.representation,
                prior_bytes,
                previous.offset,
            ) {
                rollback.get_or_insert(error);
            }
            match rollback {
                Some(rollback) => CacheResidencyError::Rollback {
                    operation: Box::new(operation),
                    rollback: Box::new(rollback),
                },
                None => operation,
            }
        })?;
        Ok(())
    }

    /// Reads the context required by the earliest query before publishing the
    /// append, whose final frontier can discard history that query still needs.
    pub fn append_with_visible_history(
        &self,
        stream: &LiveCacheStream,
        state: &mut CacheStreamState<M::Tensor>,
        input: Page<M>,
        window: i64,
        queue: &Queue<M>,
    ) -> Result<[M::Tensor; 2], CacheResidencyError>
    where
        M: eredu_core::cache::mechanisms::LiveCacheReadSubmissionBackend<Event: 'static>,
    {
        let [first, second] = input.arrays();
        let start = LiveCacheStream::history_start(state.offset, Some(window));
        let visible = if start < state.offset {
            let [past_first, past_second] =
                self.read_stream_range(stream, state, start..state.offset, queue)?;
            let axis = sequence_axis(stream.representation, M::shape(first))?;
            [
                M::concatenate_tokens(&[&past_first, first], axis, queue)?,
                M::concatenate_tokens(&[&past_second, second], axis, queue)?,
            ]
        } else {
            [first.clone(), second.clone()]
        };
        self.append_pages(stream, state, input, Some(window), queue)?;
        Ok(visible)
    }

    /// Reads a covered interval from sealed pages and the current tail.
    fn read_stream_range(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        requested: Range<i64>,
        queue: &Queue<M>,
    ) -> Result<[M::Tensor; 2], CacheResidencyError>
    where
        M: eredu_core::cache::mechanisms::LiveCacheReadSubmissionBackend<Event: 'static>,
    {
        use eredu_core::cache::mechanisms::CacheReadSubmission;

        let tail = state.tail.as_ref().map(|_| state.tail_start..state.offset);
        let read = self.plan_stream_read(stream, tail, requested.clone())?;
        let mut pages =
            self.prefetch_blocks(read.pages.iter().map(|(id, _)| id.clone()).collect(), queue)?;
        let mut parts = [Vec::new(), Vec::new()];
        let mut axis = None;
        let mut slice_pair = |tensors: [&M::Tensor; 2], range: Range<i32>, compact: bool| {
            let selected_axis = sequence_axis(stream.representation, M::shape(tensors[0]))?;
            axis = Some(selected_axis);
            for (parts, tensor) in parts.iter_mut().zip(tensors) {
                let slice = M::slice_tokens(tensor, selected_axis, range.clone(), queue)?;
                parts.push(if compact {
                    M::compact_tokens(&slice, queue)?
                } else {
                    slice
                });
            }
            Ok::<_, CacheResidencyError>(())
        };
        for (_, slice) in read.pages {
            let lease = pages.next_block()?.ok_or_else(|| {
                CacheResidencyError::Runtime("planned cache page is missing".into())
            })?;
            let [first, second] = lease.arrays().arrays().map(Clone::clone);
            let scope = M::begin_cache_read(queue, lease)?;
            slice_pair([&first, &second], slice, true)?;
            scope.settle()?;
        }
        if let Some(slice) = read.tail {
            let [first, second] = state.tail.as_ref().ok_or_else(|| {
                CacheResidencyError::Runtime("planned cache tail is missing".into())
            })?;
            slice_pair([first, second], slice, false)?;
        }
        let Some(axis) = axis else {
            return Err(CacheResidencyError::InvalidTokenRange {
                start: requested.start,
                end: requested.end,
            });
        };
        let join = |mut parts: Vec<M::Tensor>| -> Result<M::Tensor, BackendFailure> {
            if parts.len() == 1 {
                return Ok(parts.pop().expect("one selected slice"));
            }
            M::concatenate_tokens(&parts.iter().collect::<Vec<_>>(), axis, queue)
        };
        let [first, second] = parts;
        let pair = [join(first)?, join(second)?];
        if pair.iter().any(|value| {
            M::shape(value).get(axis).copied().map(i64::from)
                != Some(requested.end - requested.start)
        }) {
            return Err(CacheResidencyError::ArrayMismatch(
                "visible cache range is incomplete".into(),
            ));
        }
        Ok(pair)
    }

    /// Reinstalls a saved cursor after discarding pages beyond its sealed
    /// prefix. The saved mutable tail may have been sealed by a later append.
    pub fn restore_stream(
        &self,
        stream: &LiveCacheStream,
        state: &mut CacheStreamState<M::Tensor>,
        saved: &CacheStreamState<M::Tensor>,
        queue: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        self.truncate_stream(stream, state, saved.tail_start, queue)?;
        self.set_tail_state(
            stream.global_layer,
            stream.representation,
            saved.tail.iter().flatten().map(M::bytes).sum(),
            saved.offset,
        )?;
        *state = saved.clone();
        Ok(())
    }

    /// Truncates one stream, holding any crossing page's lease until the
    /// replacement is ready and the catalog transaction commits.
    pub fn truncate_stream(
        &self,
        stream: &LiveCacheStream,
        previous: &mut CacheStreamState<M::Tensor>,
        end: i64,
        queue: &Queue<M>,
    ) -> Result<(), CacheResidencyError> {
        if !(0..=previous.offset).contains(&end)
            || !(0..=previous.offset).contains(&previous.tail_start)
        {
            return Err(CacheResidencyError::InvalidTokenRange { start: 0, end });
        }
        if end >= previous.tail_start {
            let length = i32::try_from(end - previous.tail_start).map_err(|_| {
                CacheResidencyError::InvalidTokenRange {
                    start: previous.tail_start,
                    end,
                }
            })?;
            let tail = match (length, previous.tail.as_ref()) {
                (0, _) | (_, None) => None,
                (_, Some([first, second])) => {
                    let axis = sequence_axis(stream.representation, M::shape(&first))?;
                    Some([
                        M::slice_tokens(&first, axis, 0..length, queue)?,
                        M::slice_tokens(&second, axis, 0..length, queue)?,
                    ])
                }
            };
            let bytes = tail.iter().flatten().map(M::bytes).sum();
            self.set_tail_state(stream.global_layer, stream.representation, bytes, end)?;
            previous.tail = tail;
            previous.offset = end;
            return Ok(());
        }

        let crossing = self
            .layer_block_ids(
                stream.global_layer,
                stream.representation,
                0,
                previous.offset,
                stream.protected_prefix,
            )?
            .into_iter()
            .find(|block| block.start < end && block.end > end);
        let replacement = if let Some(id) = crossing {
            let lease = self.lease_block(&id, queue)?;
            let length = i32::try_from(end - id.start).map_err(|_| {
                CacheResidencyError::InvalidTokenRange {
                    start: id.start,
                    end,
                }
            })?;
            let [first, second] = lease.arrays().arrays();
            let axis = sequence_axis(stream.representation, M::shape(first))?;
            let first = M::slice_tokens(first, axis, 0..length, queue)?;
            let second = M::slice_tokens(second, axis, 0..length, queue)?;
            M::settle_tensors(&[&first, &second])?;
            let page = CacheBlockArrays::from_parts(
                stream.representation,
                [
                    M::copy_tensor(&first, queue)?,
                    M::copy_tensor(&second, queue)?,
                ],
            );
            Some((lease, page))
        } else {
            None
        };
        self.truncate_layer_transaction(
            stream.global_layer,
            stream.representation,
            end,
            replacement,
            stream.protected_prefix,
        )?;
        *previous = CacheStreamState::at_offset(end);
        Ok(())
    }
}

impl<M: LiveCacheBackend> CacheResidencyManager<M> {
    /// Opens a cursor at the frontier already installed in the page catalog.
    pub fn open_stream(
        &self,
        stream: &LiveCacheStream,
        window: Option<i64>,
    ) -> Result<CacheStreamState<M::Tensor>, CacheResidencyError> {
        if stream.protected_prefix < 0 || window.is_some_and(|window| window <= 0) {
            return Err(CacheResidencyError::ArrayMismatch(
                "invalid cache stream visibility".into(),
            ));
        }
        if stream.representation == CacheRepresentation::KeyValue
            && window.is_none()
            && !self.options().full_attention_enabled()
        {
            return Err(CacheResidencyError::Runtime(
                "paged full attention requires explicit blockwise full-attention enablement".into(),
            ));
        }
        self.layer_end(stream.global_layer, stream.representation)
            .map(CacheStreamState::at_offset)
    }

    /// Clears the source pages and resets the cursor only after publication.
    pub fn clear_stream(
        &self,
        stream: &LiveCacheStream,
        state: &mut CacheStreamState<M::Tensor>,
    ) -> Result<(), CacheResidencyError> {
        self.truncate_layer_transaction(stream.global_layer, stream.representation, 0, None, 0)?;
        *state = CacheStreamState::at_offset(0);
        Ok(())
    }

    /// Records the catalog state of a sequential, append-only transaction.
    pub fn checkpoint_stream_append(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        sliding: bool,
    ) -> Result<CacheStreamCheckpoint, CacheResidencyError> {
        if sliding {
            return Err(CacheResidencyError::Runtime(
                "transactional branching is unsupported for paged sliding attention".into(),
            ));
        }
        Ok(CacheStreamCheckpoint {
            session: self.session_id(),
            stream: *stream,
            offset: state.offset,
            tail_bytes: state.tail.iter().flatten().map(M::bytes).sum(),
            blocks: self.layer_block_ids(
                stream.global_layer,
                stream.representation,
                0,
                i64::MAX,
                0,
            )?,
        })
    }

    /// Rolls back pages published by a discarded append while preserving
    /// canonical pages and reinstating the original tail accounting.
    pub fn rollback_stream_append(
        &self,
        stream: &LiveCacheStream,
        checkpoint: &CacheStreamCheckpoint,
    ) -> Result<(), CacheResidencyError> {
        if self.session_id() != checkpoint.session || stream != &checkpoint.stream {
            return Err(CacheResidencyError::Runtime(
                "paged transaction checkpoint does not belong to this cache layer".into(),
            ));
        }
        let current =
            self.layer_block_ids(stream.global_layer, stream.representation, 0, i64::MAX, 0)?;
        if checkpoint.blocks.iter().any(|id| !current.contains(id)) {
            return Err(CacheResidencyError::Runtime(
                "paged transaction removed canonical immutable blocks".into(),
            ));
        }
        let mut failure = None;
        for id in current
            .into_iter()
            .rev()
            .filter(|id| !checkpoint.blocks.contains(id))
        {
            if let Err(error) = self.remove_block(&id) {
                failure.get_or_insert(error);
            }
        }
        if let Err(error) = self.set_tail_state(
            stream.global_layer,
            stream.representation,
            checkpoint.tail_bytes,
            checkpoint.offset,
        ) {
            failure.get_or_insert(error);
        }
        failure.map_or(Ok(()), Err)
    }

    /// Discards history no longer needed by subsequent queries.
    pub fn discard_stream_history(
        &self,
        stream: &LiveCacheStream,
        offset: i64,
        window: i64,
    ) -> Result<(), CacheResidencyError> {
        self.discard_before(
            stream.global_layer,
            stream.representation,
            offset.saturating_sub(window).max(stream.protected_prefix),
            stream.protected_prefix,
        )
    }

    /// Validates coverage and selects page-local slices before opening any leases.
    fn plan_stream_read(
        &self,
        stream: &LiveCacheStream,
        tail: Option<Range<i64>>,
        requested: Range<i64>,
    ) -> Result<CacheStreamRead, CacheResidencyError> {
        let invalid = || CacheResidencyError::InvalidTokenRange {
            start: requested.start,
            end: requested.end,
        };
        if requested.start < 0
            || requested.start >= requested.end
            || tail
                .as_ref()
                .is_some_and(|tail| tail.start < 0 || tail.end < tail.start)
        {
            return Err(invalid());
        }
        let ids = self.layer_block_ids(
            stream.global_layer,
            stream.representation,
            requested.start,
            requested.end,
            0,
        )?;
        let mut frontier = requested.start;
        let mut pages = Vec::with_capacity(ids.len());
        for id in ids {
            let start = requested.start.max(id.start);
            let end = requested.end.min(id.end);
            if start != frontier || end <= start {
                return Err(invalid());
            }
            let range = i32::try_from(start - id.start).map_err(|_| invalid())?
                ..i32::try_from(end - id.start).map_err(|_| invalid())?;
            pages.push((id, range));
            frontier = end;
        }
        let tail = if frontier == requested.end {
            None
        } else {
            let tail = tail.ok_or_else(invalid)?;
            if frontier != tail.start.max(requested.start) || requested.end > tail.end {
                return Err(invalid());
            }
            Some(
                i32::try_from(frontier - tail.start).map_err(|_| invalid())?
                    ..i32::try_from(requested.end - tail.start).map_err(|_| invalid())?,
            )
        };
        Ok(CacheStreamRead { pages, tail })
    }

    /// Validates indexed coverage and projects the selected rows onto their pages.
    fn plan_indexed_read(
        &self,
        stream: &LiveCacheStream,
        offset: i64,
        tail_start: Option<i64>,
        positions: &[i32],
    ) -> Result<CacheIndexedRead, CacheResidencyError> {
        let invalid = || CacheResidencyError::InvalidTokenRange {
            start: 0,
            end: offset,
        };
        if offset < 0
            || tail_start.is_some_and(|start| !(0..=offset).contains(&start))
            || positions.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(invalid());
        }
        if let Some(&position) = positions
            .iter()
            .find(|&&position| position < 0 || i64::from(position) >= offset)
        {
            return Err(CacheResidencyError::IndexedPositionOutOfRange { position, offset });
        }
        let mut ids =
            self.selected_layer_block_ids(stream.global_layer, stream.representation, positions)?;
        ids.sort_by_key(|id| id.start);
        let mut remaining = positions;
        let mut pages = Vec::with_capacity(ids.len());
        for id in ids {
            if remaining.first().is_some_and(|&p| i64::from(p) < id.start) {
                return Err(invalid());
            }
            let end = remaining.partition_point(|&p| i64::from(p) < id.end);
            let rows = remaining[..end]
                .iter()
                .map(|&position| {
                    i32::try_from(i64::from(position) - id.start).map_err(|_| invalid())
                })
                .collect::<Result<Vec<_>, _>>()?;
            pages.push((id, rows));
            remaining = &remaining[end..];
        }
        let tail = if remaining.is_empty() {
            Vec::new()
        } else {
            let start = tail_start.ok_or_else(invalid)?;
            remaining
                .iter()
                .map(|&position| {
                    i32::try_from(i64::from(position) - start)
                        .ok()
                        .filter(|&local| local >= 0)
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(CacheIndexedRead { pages, tail })
    }

    /// Selects the retained pages needed by an attention invocation.
    pub fn plan_attention_read(
        &self,
        stream: &LiveCacheStream,
        offset: i64,
        query_len: i64,
        transient_len: Option<i64>,
        sliding_window: Option<i64>,
    ) -> Result<CacheAttentionRead, CacheResidencyError> {
        if offset < 0
            || query_len <= 0
            || transient_len.is_some_and(|len| len < 0)
            || sliding_window.is_some_and(|window| window <= 0)
            || (transient_len.is_none() && query_len > offset)
        {
            return Err(CacheResidencyError::InvalidTokenRange {
                start: 0,
                end: query_len,
            });
        }
        let query_start = if transient_len.is_some() {
            offset
        } else {
            offset
                .checked_sub(query_len)
                .ok_or(CacheResidencyError::InvalidTokenRange {
                    start: 0,
                    end: query_len,
                })?
        };
        let context_end = offset
            .checked_add(transient_len.unwrap_or(0))
            .ok_or_else(|| {
                CacheResidencyError::Runtime("attention context extent overflowed".into())
            })?;
        let visible_start = LiveCacheStream::history_start(query_start, sliding_window);
        let pages = self.layer_block_ids(
            stream.global_layer,
            stream.representation,
            visible_start,
            offset,
            stream.protected_prefix,
        )?;
        Ok(CacheAttentionRead {
            pages,
            query_start,
            query_len,
            visible_start,
            context_end,
        })
    }

    /// Seals the current dirty tail without changing its caller-owned tensors on failure.
    pub fn seal_stream(
        &self,
        stream: &LiveCacheStream,
        state: &mut CacheStreamState<M::Tensor>,
    ) -> Result<(), CacheResidencyError> {
        if let Some(tail) = &state.tail {
            state.tail_start = self.seal_stream_tail(stream, state.tail_start, tail.clone())?;
            state.tail = None;
        }
        Ok(())
    }

    /// Seals a dirty tail without changing its caller-owned tensors on failure.
    fn seal_stream_tail(
        &self,
        stream: &LiveCacheStream,
        start: i64,
        values: [M::Tensor; 2],
    ) -> Result<i64, CacheResidencyError> {
        let shape = M::shape(&values[0]);
        let axis = sequence_axis(stream.representation, shape)?;
        let end = start
            .checked_add(i64::from(shape[axis]))
            .ok_or_else(|| CacheResidencyError::Runtime("cache tail frontier overflowed".into()))?;
        let page = CacheBlockArrays::from_parts(stream.representation, values);
        self.seal_pending_page(stream, start, end, page)?;
        Ok(end)
    }

    fn seal_pending_page(
        &self,
        stream: &LiveCacheStream,
        start: i64,
        end: i64,
        page: Page<M>,
    ) -> Result<CacheBlockId, CacheResidencyError> {
        let bytes = page_bytes::<M>(&page);
        self.set_tail_state(stream.global_layer, stream.representation, 0, end)?;
        let result = self.seal_block(
            stream.global_layer,
            start,
            end,
            stream.rank,
            page,
            end <= stream.protected_prefix,
        );
        result.map_err(|operation| {
            match self.set_tail_state(stream.global_layer, stream.representation, bytes, end) {
                Ok(()) => operation,
                Err(rollback) => CacheResidencyError::Rollback {
                    operation: Box::new(operation),
                    rollback: Box::new(rollback),
                },
            }
        })
    }
}

fn sequence_axis(
    representation: CacheRepresentation,
    shape: &[i32],
) -> Result<usize, CacheResidencyError> {
    match representation {
        CacheRepresentation::CompressedLatentRotary if shape.len() == 3 => Some(1),
        CacheRepresentation::CompressedLatentRotary => None,
        _ => shape.len().checked_sub(2),
    }
    .ok_or_else(|| CacheResidencyError::ArrayMismatch("cache tensor has no sequence axis".into()))
}

impl<M> CacheResidencyManager<M>
where
    M: eredu_core::cache::mechanisms::LiveCacheSelectedReadBackend<Event: 'static>,
{
    /// Reads precisely the selected positions. Each page is copied and retired
    /// separately so the next page can reuse its device budget.
    pub fn read_indexed_stream(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        batch: i32,
        positions: &[i32],
        queue: &Queue<M>,
    ) -> Result<([M::Tensor; 2], u64), CacheResidencyError> {
        let read = self.plan_indexed_read(
            stream,
            state.offset,
            state.tail.as_ref().map(|_| state.tail_start),
            positions,
        )?;
        let mut pages =
            self.prefetch_blocks(read.pages.iter().map(|(id, _)| id.clone()).collect(), queue)?;
        let mut parts = [Vec::new(), Vec::new()];
        let mut selected_bytes = 0u64;
        for (_, rows) in read.pages {
            let lease = pages.next_block()?.ok_or_else(|| {
                CacheResidencyError::Runtime("planned indexed cache page is missing".into())
            })?;
            self.record_selected_attention(stream.global_layer, 1, lease.bytes(), 0)?;
            let CacheBlockArrays::KeyValue { keys, values } = lease.arrays() else {
                return Err(CacheResidencyError::ArrayMismatch(
                    "selected history has incompatible representation".into(),
                ));
            };
            let [keys, values] = [keys.clone(), values.clone()];
            let copied = M::gather_token_rows([&keys, &values], batch, &rows, queue, lease)?;
            for (parts, tensor) in parts.iter_mut().zip(copied) {
                selected_bytes = selected_bytes.saturating_add(M::bytes(&tensor));
                parts.push(tensor);
            }
        }
        if !read.tail.is_empty() {
            let [keys, values] = state.tail.as_ref().ok_or_else(|| {
                CacheResidencyError::Runtime("planned indexed cache tail is missing".into())
            })?;
            let copied = M::gather_token_rows([keys, values], batch, &read.tail, queue, ())?;
            let bytes = copied.iter().map(M::bytes).sum::<u64>();
            selected_bytes = selected_bytes.saturating_add(bytes);
            self.record_selected_attention(stream.global_layer, 1, bytes, 0)?;
            for (parts, tensor) in parts.iter_mut().zip(copied) {
                parts.push(tensor);
            }
        }
        let [keys, values] = parts;
        let join = |parts: &[M::Tensor]| {
            M::concatenate_tokens(&parts.iter().collect::<Vec<_>>(), 2, queue)
        };
        Ok((
            [join(&keys)?, join(&values)?],
            selected_bytes.saturating_mul(2),
        ))
    }

    /// Copies a requested append-stream interval without pinning its entire
    /// history. Each compact copy is complete before acquiring the next page.
    pub fn read_append_stream(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        range: Range<i64>,
        queue: &Queue<M>,
    ) -> Result<M::Tensor, CacheResidencyError> {
        let tail = state.tail.as_ref().map(|_| state.tail_start..state.offset);
        let read = self.plan_stream_read(stream, tail, range)?;
        let mut parts = Vec::with_capacity(read.pages.len() + usize::from(read.tail.is_some()));
        for (id, slice) in read.pages {
            let lease = self.lease_block(&id, queue)?;
            self.record_append_stream_read(stream.global_layer, 1, lease.bytes(), 0)?;
            let CacheBlockArrays::AppendStream { records, .. } = lease.arrays() else {
                return Err(CacheResidencyError::ArrayMismatch(
                    "append stream has incompatible stored representation".into(),
                ));
            };
            let records = records.clone();
            parts.push(M::copy_token_slice(&records, 2, slice, queue, lease)?);
        }
        if let Some(slice) = read.tail {
            let [records, _] = state.tail.as_ref().ok_or_else(|| {
                CacheResidencyError::Runtime("planned append-stream tail is missing".into())
            })?;
            parts.push(M::slice_tokens(records, 2, slice, queue)?);
        }
        M::concatenate_tokens(&parts.iter().collect::<Vec<_>>(), 2, queue).map_err(Into::into)
    }
}

impl<M: LiveCacheBackend<Event: 'static>> CacheResidencyManager<M> {
    /// Drives one admitted attention scan and accounts for the exact blocks used.
    pub fn run_attention_scan(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        read: CacheAttentionRead,
        transient: Option<[&M::Tensor; 2]>,
        window: Option<i64>,
        queue: &Queue<M>,
        mut accumulator: impl eredu_core::cache::mechanisms::CacheAttentionAccumulator<
            M::Tensor,
            Queue<M>,
        >,
    ) -> Result<M::Tensor, CacheResidencyError> {
        use eredu_core::cache::mechanisms::CacheAttentionInput;

        let mut scanned_blocks = 0u64;
        let mut scanned_bytes = 0u64;
        let mut scratch = 0u64;
        for pass in 0..accumulator.scan().passes() {
            if pass > 0 {
                accumulator.begin_pass(pass, queue)?;
            }
            let mut pages = self.prefetch_blocks(read.pages.clone(), queue)?;
            while let Some(lease) = pages.next_block()? {
                let id = lease.id();
                let positions = id.start..id.end;
                let [first, second] = lease.arrays().arrays().map(Clone::clone);
                scanned_blocks += 1;
                scanned_bytes += lease.bytes();
                scratch = scratch.max(accumulator.accumulate(
                    CacheAttentionInput::SealedPage,
                    positions,
                    [&first, &second],
                    queue,
                    lease,
                )?);
            }
            if let Some([first, second]) = &state.tail {
                scanned_blocks += 1;
                scanned_bytes += M::bytes(first) + M::bytes(second);
                scratch = scratch.max(accumulator.accumulate(
                    CacheAttentionInput::RetainedTail,
                    state.tail_start..state.offset,
                    [first, second],
                    queue,
                    (),
                )?);
            }
            if let Some([first, second]) = transient {
                scanned_blocks += 1;
                scanned_bytes += M::bytes(first) + M::bytes(second);
                scratch = scratch.max(accumulator.accumulate(
                    CacheAttentionInput::Transient,
                    state.offset..read.context_end,
                    [first, second],
                    queue,
                    (),
                )?);
            }
        }
        let output = accumulator.finish(queue)?;
        if transient.is_none() {
            if let Some(window) = window {
                self.discard_stream_history(stream, state.offset, window)?;
            }
        }
        self.record_attention_scan(
            stream.global_layer,
            read.query_len > 1,
            scanned_blocks,
            scanned_bytes,
            scratch,
        )?;
        Ok(output)
    }
}

impl<M> CacheResidencyManager<M>
where
    M: eredu_core::cache::mechanisms::LiveCacheReadSubmissionBackend<Event: 'static>,
{
    /// Supplies compressed history to the portable reconstruction operator.
    /// Each source page remains leased until the reconstructed block settles.
    pub fn scan_compressed_history<T: eredu_nn::Tensor>(
        &self,
        stream: &LiveCacheStream,
        state: &CacheStreamState<M::Tensor>,
        query_tokens: i32,
        queue: &Queue<M>,
        tensor: impl Fn(M::Tensor) -> T,
        mut visit: impl FnMut(eredu_nn::CompressedAttentionBlock<T>) -> Result<u64, eredu_nn::Error>,
    ) -> Result<eredu_nn::CompressedAttentionScan, eredu_nn::Error> {
        use eredu_core::cache::mechanisms::CacheReadSubmission;
        use eredu_nn::{
            CompressedAttentionBlock, CompressedAttentionScan, CompressedAttentionState, Error,
        };

        let ids = self
            .layer_block_ids(
                stream.global_layer,
                stream.representation,
                0,
                state.offset,
                0,
            )
            .map_err(Error::backend)?;
        let mut pages = self.prefetch_blocks(ids, queue).map_err(Error::backend)?;
        let mut scan = CompressedAttentionScan::default();
        let mut accumulate = |start, end, [latent, rotary]: [&M::Tensor; 2], bytes| {
            let scratch = visit(CompressedAttentionBlock {
                start,
                end,
                state: CompressedAttentionState {
                    latent: tensor(latent.clone()),
                    rotary: tensor(rotary.clone()),
                },
            })?;
            scan.blocks += 1;
            scan.bytes += bytes;
            scan.reconstruction_scratch_bytes = scan.reconstruction_scratch_bytes.max(scratch);
            Ok::<_, Error>(())
        };
        while let Some(lease) = pages.next_block().map_err(Error::backend)? {
            let (start, end, bytes) = (lease.id().start, lease.id().end, lease.bytes());
            let [latent, rotary] = lease.arrays().arrays().map(Clone::clone);
            let scope = M::begin_cache_read(queue, lease).map_err(Error::backend)?;
            accumulate(start, end, [&latent, &rotary], bytes)?;
            scope.settle().map_err(Error::backend)?;
        }
        if let Some([latent, rotary]) = &state.tail {
            accumulate(
                state.tail_start,
                state.offset,
                [latent, rotary],
                M::bytes(latent) + M::bytes(rotary),
            )?;
        }
        self.record_attention_scan(
            stream.global_layer,
            query_tokens > 1,
            scan.blocks,
            scan.bytes,
            scan.reconstruction_scratch_bytes,
        )
        .map_err(Error::backend)?;
        Ok(scan)
    }
}
