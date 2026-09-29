//! Named records reuse ordinary paged blocks, tail transactions, and native leases.
use super::*;
use crate::backend::runtime::cache::residency::CacheBlockLease;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use eredu_nn::Tensor;
use eredu_runtime::{AppendOnlyStream, AppendStreamError, AppendStreamLimits, AppendStreamSpec};
use std::ops::Range;

fn native(error: Exception) -> AppendStreamError {
    AppendStreamError::Tensor(eredu_nn::Error::backend(error))
}

struct ReadLease(CacheBlockLease);
impl Retention for ReadLease {
    fn observe(&self, _: Status) {}
}

/// Bounded lane-local append storage sharing the model's ordinary paging manager.
/// The native paired-block encoding uses a zero-width second array: only records
/// consume logical payload bytes. No attention operation is exposed by this type.
#[derive(Debug, Clone)]
pub struct MlxPagedAppendStream {
    spec: AppendStreamSpec,
    limits: AppendStreamLimits,
    cache: PagedKeyValueCache,
}

impl MlxPagedAppendStream {
    /// Binds architecture geometry to already admitted shared residency limits.
    /// Catalog length is bounded by `entries`; reads and tail copies by `scratch_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: AppendStreamSpec,
        limits: AppendStreamLimits,
        manager: CacheResidencyManager,
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
                crate::backend::runtime::cache::residency::CacheResidencyError::ArrayMismatch(
                    _,
                ) => AppendStreamError::Geometry,
                error => native(cache_residency_exception(error)),
            })?;
        let representation = CacheRepresentation::AppendStream {
            slot: spec.slot,
            lane,
        };
        let cache = PagedKeyValueCache::new_with_representation(
            manager,
            global_layer,
            None,
            0,
            rank,
            true,
            representation,
        )
        .map_err(native)?;
        if cache.offset as usize > limits.entries {
            return Err(AppendStreamError::Budget {
                resource: "entries",
                required: cache.offset as u64,
                limit: limits.entries as u64,
            });
        }
        Ok(Self {
            spec,
            limits,
            cache,
        })
    }

    /// Native roots retained until the execution owner's completion boundary.
    pub fn retained_values(&self) -> impl Iterator<Item = &MlxTensor> {
        self.cache.tail_keys.iter().map(retained_tensor)
    }

    /// Captures an exact append-only frontier without gathering sealed history.
    pub fn checkpoint_clone_state(&self) -> Result<Self, AppendStreamError> {
        Ok(Self {
            spec: self.spec.clone(),
            limits: self.limits,
            cache: self.cache.checkpoint_clone_state().map_err(native)?,
        })
    }

    /// Restores an exact frontier while leaving cumulative activity budgets live.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), AppendStreamError> {
        if self.spec != checkpoint.spec
            || self.limits != checkpoint.limits
            || !self.cache.has_same_transaction_identity(&checkpoint.cache)
        {
            return Err(AppendStreamError::Geometry);
        }
        self.cache
            .restore_checkpoint(&checkpoint.cache, stream)
            .map_err(native)
    }

    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        self.cache.manager()
    }

    /// Seals a partial tail through the same persistence path as ordinary attention.
    pub fn finalize(&mut self) -> Result<(), AppendStreamError> {
        self.cache.finalize().map_err(native)
    }

    /// Clears this stream alone; transfer and observation counters are not reset.
    pub fn clear(&mut self) -> Result<(), AppendStreamError> {
        self.cache.clear().map_err(native)
    }

    /// Copies local mutable roots into an already forked model-wide manager.
    /// The state owner forks the shared manager once for all its streams.
    pub(crate) fn fork_into(
        &self,
        manager: CacheResidencyManager,
        stream: &Stream,
    ) -> Result<Self, AppendStreamError> {
        let mut cache = self.cache.deep_clone_state(stream).map_err(native)?;
        cache.rebind_paging_manager(manager);
        Ok(Self {
            spec: self.spec.clone(),
            limits: self.limits,
            cache,
        })
    }
}

impl AppendOnlyStream<MlxTensor> for MlxPagedAppendStream {
    fn specification(&self) -> &AppendStreamSpec {
        &self.spec
    }
    fn len(&self) -> usize {
        self.cache.offset as usize
    }

    fn append(
        &mut self,
        frontier: usize,
        values: MlxTensor,
        stream: &Stream,
    ) -> Result<(), AppendStreamError> {
        if frontier != self.len() {
            return Err(AppendStreamError::Frontier {
                expected: frontier,
                actual: self.len(),
            });
        }
        if values.shape().len() != 2
            || values.dim(0) <= 0
            || values.dim(1) != self.spec.width
            || values.element_type() != Some(self.spec.element)
        {
            return Err(AppendStreamError::Geometry);
        }
        let next = frontier
            .checked_add(values.dim(0) as usize)
            .ok_or(AppendStreamError::Geometry)?;
        if next > self.limits.entries {
            return Err(AppendStreamError::Budget {
                resource: "entries",
                required: next as u64,
                limit: self.limits.entries as u64,
            });
        }
        let entries = values.dim(0);
        let array = values.into_array();
        let records = array
            .reshape(&[1, 1, entries, self.spec.width], stream)
            .map_err(native)?;
        let reserved = zeros_dtype(&[1, 1, entries, 0], records.dtype(), stream).map_err(native)?;
        self.cache.append(records, reserved, stream).map_err(native)
    }

    fn read(
        &mut self,
        range: Range<usize>,
        stream: &Stream,
    ) -> Result<MlxTensor, AppendStreamError> {
        if range.start >= range.end || range.end > self.len() {
            return Err(AppendStreamError::Range);
        }
        if range.len() > self.limits.read_entries {
            return Err(AppendStreamError::Budget {
                resource: "read entries",
                required: range.len() as u64,
                limit: self.limits.read_entries as u64,
            });
        }
        self.cache
            .manager
            .record_append_stream_read(
                self.cache.global_layer,
                0,
                0,
                2 * range.len() as u64 * self.spec.record_bytes()?,
            )
            .map_err(cache_residency_exception)
            .map_err(native)?;
        let start = range.start as i64;
        let end = range.end as i64;
        let ids = self
            .cache
            .manager
            .layer_block_ids(
                self.cache.global_layer,
                self.cache.representation,
                start,
                end,
                0,
            )
            .map_err(cache_residency_exception)
            .map_err(native)?;
        let mut frontier = start;
        let mut parts = Vec::new();
        // Each lease is retired only after a compact independent copy has completed.
        // The next page may then reuse its device budget. No entire-history gather.
        for id in ids {
            if id.start.max(start) != frontier {
                return Err(AppendStreamError::Range);
            }
            let lease = self
                .cache
                .manager
                .lease_block(&id, stream)
                .map_err(cache_residency_exception)
                .map_err(native)?;
            self.cache
                .manager
                .record_append_stream_read(self.cache.global_layer, 1, lease.bytes(), 0)
                .map_err(cache_residency_exception)
                .map_err(native)?;
            let mut recovery = Recovery::begin(ReadLease(lease)).map_err(native)?;
            let copy = (|| {
                let CacheBlockArrays::AppendStream { records, .. } =
                    recovery.retention().0.arrays()
                else {
                    return Err(Exception::custom(
                        "append stream has incompatible stored representation",
                    ));
                };
                let part = records.try_index_device(
                    (
                        0,
                        0,
                        (frontier - id.start) as i32..(end.min(id.end) - id.start) as i32,
                        ..,
                    ),
                    stream,
                )?;
                let part = part.contiguous(false, stream)?.deep_clone()?;
                safemlx::transforms::eval([&part])?;
                Ok::<_, Exception>(part)
            })();
            recovery.seal();
            let part = copy.map_err(native)?;
            let status = recovery.finish();
            if status.failed || status.blocked {
                return Err(native(Exception::custom(
                    "append range copy failed; unresolved source lease retained",
                )));
            }
            parts.push(part);
            frontier = end.min(id.end);
        }
        if frontier < end {
            if frontier < self.cache.tail_start {
                return Err(AppendStreamError::Range);
            }
            let records = self
                .cache
                .tail_keys
                .as_ref()
                .ok_or(AppendStreamError::Range)?;
            parts.push(
                records
                    .try_index_device(
                        (
                            0,
                            0,
                            (frontier - self.cache.tail_start) as i32
                                ..(end - self.cache.tail_start) as i32,
                            ..,
                        ),
                        stream,
                    )
                    .map_err(native)?,
            );
        }
        let result = concatenate_axis(&parts, 0, stream).map_err(native)?;
        // Ensure the returned range does not retain an oversized tail view.
        let result = MlxTensor::from_array(result).compact(stream)?;
        if result.shape() != [range.len() as i32, self.spec.width]
            || result.element_type() != Some(self.spec.element)
        {
            return Err(AppendStreamError::Geometry);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
