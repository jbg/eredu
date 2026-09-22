//! Native callbacks of the shared descriptive truncation traversal.
use super::*;
use eredu_runtime::cache::{PagedTruncateMechanisms, PagedTruncatePlan};
use std::ops::Range;

struct Native<'a> {
    cache: &'a mut PagedKeyValueCache,
    stream: &'a Stream,
    crossing: Option<CacheBlockId>,
}
impl PagedKeyValueCache {
    pub(super) fn truncate_with_plan(
        &mut self,
        len: i64,
        stream: &Stream,
    ) -> Result<(), Exception> {
        // Guarded checkpoint restoration has its own current-and-saved source
        // companion. This arbitrary-cut entry has no such numerical grant.
        if let Some(owner) = crate::backend::nn::shared::current_ordinary_execution_owner()? {
            return Err(checkpoint::unqualified_truncate(owner.host()));
        }
        let crossing = if len >= 0 && len < self.tail_start {
            self.manager
                .layer_block_ids(
                    self.global_layer,
                    CacheRepresentation::KeyValue,
                    0,
                    self.offset,
                    i64::from(self.prefix_tokens),
                )
                .map_err(cache_residency_exception)?
                .into_iter()
                .find(|id| id.start < len && id.end > len)
        } else {
            None
        };
        let plan = PagedTruncatePlan::new(
            self.offset,
            self.tail_start,
            len,
            crossing.as_ref().map(|id| id.start..id.end),
        )
        .map_err(|cause| Exception::from_retained_source(cause))?;
        plan.run(&mut Native {
            cache: self,
            stream,
            crossing,
        })
    }
}
impl PagedTruncateMechanisms for Native<'_> {
    type Source = super::super::super::residency::CacheBlockLease;
    type Pair = (Option<Array>, Option<Array>);
    type Error = Exception;
    fn slice_tail(&mut self, n: i32) -> Result<Self::Pair, Exception> {
        Ok((
            self.cache
                .tail_keys
                .as_ref()
                .map(|a| a.try_index_device((.., .., ..n, ..), self.stream))
                .transpose()?,
            self.cache
                .tail_values
                .as_ref()
                .map(|a| a.try_index_device((.., .., ..n, ..), self.stream))
                .transpose()?,
        ))
    }
    fn publish_tail(&mut self, end: i64, n: i32, pair: Self::Pair) -> Result<(), Exception> {
        let (keys, values) = if n == 0 { (None, None) } else { pair };
        let bytes = keys
            .iter()
            .chain(values.iter())
            .map(|array| array.nbytes() as u64)
            .sum();
        self.cache
            .manager
            .set_tail_state(self.cache.global_layer, bytes, end)
            .map_err(cache_residency_exception)?;
        self.cache.tail_keys = keys;
        self.cache.tail_values = values;
        self.cache.offset = end;
        Ok(())
    }
    fn acquire_block(&mut self, range: Range<i64>) -> Result<Self::Source, Exception> {
        let id = self
            .crossing
            .as_ref()
            .filter(|id| id.start == range.start && id.end == range.end)
            .ok_or_else(|| {
                Exception::custom("paged truncation source differs from inspected block")
            })?;
        self.cache
            .manager
            .lease_block(id, self.stream)
            .map_err(cache_residency_exception)
    }
    fn slice_block(&mut self, source: &Self::Source, n: i32) -> Result<Self::Pair, Exception> {
        match source.arrays() {
            CacheBlockArrays::KeyValue { keys, values } => Ok((
                Some(keys.try_index_device((.., .., ..n, ..), self.stream)?),
                Some(values.try_index_device((.., .., ..n, ..), self.stream)?),
            )),
            _ => Err(Exception::custom(
                "paged key/value cache found an incompatible block representation",
            )),
        }
    }
    fn complete_slices(&mut self, pair: &Self::Pair) -> Result<(), Exception> {
        safemlx::transforms::async_eval_with_event([
            pair.0.as_ref().expect("block keys"),
            pair.1.as_ref().expect("block values"),
        ])?
        .synchronize()
    }
    fn copy_slices(&mut self, (keys, values): Self::Pair) -> Result<Self::Pair, Exception> {
        Ok((
            Some(
                keys.expect("block keys")
                    .contiguous(false, self.stream)?
                    .deep_clone()?,
            ),
            Some(
                values
                    .expect("block values")
                    .contiguous(false, self.stream)?
                    .deep_clone()?,
            ),
        ))
    }
    fn publish_catalog(
        &mut self,
        end: i64,
        replacement: Option<(Self::Source, Self::Pair)>,
    ) -> Result<(), Exception> {
        let replacement = replacement.map(|(lease, (keys, values))| {
            (
                lease,
                CacheBlockArrays::KeyValue {
                    keys: keys.expect("block keys"),
                    values: values.expect("block values"),
                },
            )
        });
        self.cache
            .manager
            .truncate_layer_transaction(
                self.cache.global_layer,
                CacheRepresentation::KeyValue,
                end,
                replacement,
                i64::from(self.cache.prefix_tokens),
            )
            .map_err(cache_residency_exception)?;
        self.cache.tail_keys = None;
        self.cache.tail_values = None;
        self.cache.offset = end;
        self.cache.tail_start = end;
        Ok(())
    }
    fn restore_checkpoint(&mut self, _: i64, _: i64) -> Result<(), Exception> {
        Err(Exception::custom(
            "paged checkpoint requires its retained saved source",
        ))
    }
}
