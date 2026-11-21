//! Named record streams backed by the portable page catalog and tail ledger.

use super::live::{
    CacheBlockArrays, CacheResidencyError, CacheResidencyManager, CacheStreamState, LiveCacheStream,
};
use eredu_core::cache::append_stream::{
    AppendOnlyStream, AppendStreamError, AppendStreamLimits, AppendStreamSpec,
};
use eredu_core::cache::{
    mechanisms::{CacheTensorStorage, LiveCacheSelectedReadBackend},
    CacheRankIdentity, CacheRepresentation,
};
use eredu_nn::Tensor;
use std::ops::Range;

fn cache_error(error: CacheResidencyError) -> AppendStreamError {
    AppendStreamError::Tensor(eredu_nn::Error::backend_source(error))
}

/// Bounded lane-local append storage sharing the model's ordinary paging manager.
/// The native paired-block encoding uses a zero-width second array: only records
/// consume logical payload bytes. No attention operation is exposed by this type.
#[derive(Debug, Clone)]
pub struct PagedAppendStream<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>> {
    spec: AppendStreamSpec,
    limits: AppendStreamLimits,
    manager: CacheResidencyManager<M>,
    stream: LiveCacheStream,
    cursor: CacheStreamState<M::Tensor>,
}

impl<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>> PagedAppendStream<M> {
    /// Binds architecture geometry to already admitted shared residency limits.
    /// Catalog length is bounded by `entries`; reads and tail copies by `scratch_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: AppendStreamSpec,
        limits: AppendStreamLimits,
        manager: CacheResidencyManager<M>,
        global_layer: usize,
        lane: u32,
        rank: Option<CacheRankIdentity>,
        scratch_bytes: u64,
    ) -> Result<Self, AppendStreamError> {
        let required = limits.scratch_bytes(&spec)?;
        if required > scratch_bytes {
            return Err(AppendStreamError::Budget {
                resource: "scratch bytes",
                required,
                limit: scratch_bytes,
            });
        }
        let page_bytes = (limits.page_entries as u64)
            .checked_mul(spec.record_bytes()?)
            .ok_or(AppendStreamError::Geometry)?;
        if page_bytes > manager.options().device_budget_bytes() {
            return Err(AppendStreamError::Budget {
                resource: "device page bytes",
                required: page_bytes,
                limit: manager.options().device_budget_bytes(),
            });
        }
        if limits.page_entries != manager.options().block_size_tokens() as usize {
            return Err(AppendStreamError::Geometry);
        }
        manager
            .validate_append_stream_layout(global_layer, lane, rank, &spec, limits)
            .map_err(|error| match error {
                CacheResidencyError::ArrayMismatch(_) => AppendStreamError::Geometry,
                error => cache_error(error),
            })?;
        let stream = LiveCacheStream {
            global_layer,
            representation: CacheRepresentation::AppendStream {
                slot: spec.slot,
                lane,
            },
            rank,
            protected_prefix: 0,
        };
        let cursor = manager.open_stream(&stream, None).map_err(cache_error)?;
        if cursor.offset as usize > limits.entries {
            return Err(AppendStreamError::Budget {
                resource: "entries",
                required: cursor.offset as u64,
                limit: limits.entries as u64,
            });
        }
        Ok(Self {
            spec,
            limits,
            manager,
            stream,
            cursor,
        })
    }

    /// Native roots retained until the execution owner's completion boundary.
    pub fn retained_values(&self) -> impl Iterator<Item = &M::Value> {
        self.cursor
            .tail
            .iter()
            .map(|[records, _]| M::borrow_value(records))
    }

    /// Restores an exact frontier while leaving cumulative activity budgets live.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        queue: &M::Queue,
    ) -> Result<(), AppendStreamError> {
        if self.spec != checkpoint.spec
            || self.limits != checkpoint.limits
            || self.stream != checkpoint.stream
            || self.manager.session_id() != checkpoint.manager.session_id()
        {
            return Err(AppendStreamError::Geometry);
        }
        self.manager
            .restore_stream(&self.stream, &mut self.cursor, &checkpoint.cursor, queue)
            .map_err(cache_error)
    }

    /// Shared page owner for this stream.
    pub fn manager(&self) -> &CacheResidencyManager<M> {
        &self.manager
    }

    /// Seals a partial tail through the same persistence path as ordinary attention.
    pub fn finalize(&mut self) -> Result<(), AppendStreamError> {
        self.manager
            .seal_stream(&self.stream, &mut self.cursor)
            .map_err(cache_error)
    }

    /// Clears this stream alone; transfer and observation counters are not reset.
    pub fn clear(&mut self) -> Result<(), AppendStreamError> {
        self.manager
            .clear_stream(&self.stream, &mut self.cursor)
            .map_err(cache_error)
    }

    /// Copies local mutable roots into an already forked model-wide manager.
    /// The state owner forks the shared manager once for all its streams.
    pub fn fork_into(
        &self,
        manager: CacheResidencyManager<M>,
        queue: &M::Queue,
    ) -> Result<Self, AppendStreamError> {
        let mut snapshot = self.clone();
        if let Some([records, reserved]) = &self.cursor.tail {
            let copy =
                |tensor| M::copy_tensor(tensor, queue).map_err(eredu_nn::Error::backend_source);
            snapshot.cursor.tail = Some([copy(records)?, copy(reserved)?]);
        }
        snapshot.manager = manager;
        Ok(snapshot)
    }
}

impl<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>>
    AppendOnlyStream<M::Value> for PagedAppendStream<M>
{
    fn specification(&self) -> &AppendStreamSpec {
        &self.spec
    }
    fn len(&self) -> usize {
        self.cursor.offset as usize
    }

    fn append(
        &mut self,
        frontier: usize,
        values: M::Value,
        stream: &M::Queue,
    ) -> Result<(), AppendStreamError> {
        crate::append_stream::admit_append(
            &self.spec,
            &self.limits,
            self.len(),
            frontier,
            &values,
        )?;
        let entries = values.dim(0);
        let records = M::into_storage(values.reshape(&[1, 1, entries, self.spec.width], stream)?);
        let reserved = M::zeroed_cache_tensor(&[1, 1, entries, 0], &records, stream)
            .map_err(eredu_nn::Error::backend_source)?;
        let CacheRepresentation::AppendStream { slot, lane } = self.stream.representation else {
            return Err(AppendStreamError::Geometry);
        };
        self.manager
            .append_pages(
                &self.stream,
                &mut self.cursor,
                CacheBlockArrays::AppendStream {
                    slot,
                    lane,
                    records,
                    reserved,
                },
                None,
                stream,
            )
            .map_err(cache_error)
    }

    fn read(
        &mut self,
        range: Range<usize>,
        stream: &M::Queue,
    ) -> Result<M::Value, AppendStreamError> {
        crate::append_stream::admit_read(&self.limits, self.len(), &range)?;
        self.manager
            .record_append_stream_read(
                self.stream.global_layer,
                0,
                0,
                2 * range.len() as u64 * self.spec.record_bytes()?,
            )
            .map_err(cache_error)?;
        let start = range.start as i64;
        let end = range.end as i64;
        let result = self
            .manager
            .read_append_stream(&self.stream, &self.cursor, start..end, stream)
            .map_err(|error| match error {
                CacheResidencyError::InvalidTokenRange { .. } => AppendStreamError::Range,
                other => cache_error(other),
            })?;
        let result =
            M::into_value(result).reshape(&[range.len() as i32, self.spec.width], stream)?;
        // Ensure the returned range does not retain an oversized tail view.
        let result = result.compact(stream)?;
        if result.shape() != [range.len() as i32, self.spec.width]
            || result.element_type() != Some(self.spec.element)
        {
            return Err(AppendStreamError::Geometry);
        }
        Ok(result)
    }
}
