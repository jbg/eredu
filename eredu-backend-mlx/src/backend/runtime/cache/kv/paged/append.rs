//! Native tensor/account adapter for the common paged append traversal.
use super::*;
use crate::backend::nn::workspace::OriginalPagedAppendClaim;
use eredu_runtime::cache::PagedAppendMechanisms;

pub(super) struct NativeAppend<'a, 'source> {
    pub(super) cache: &'a mut PagedKeyValueCache,
    pub(super) keys: Array,
    pub(super) values: Array,
    pub(super) stream: &'a Stream,
    pub(super) original: Option<&'a mut OriginalPagedAppendClaim<'source>>,
}
impl PagedAppendMechanisms for NativeAppend<'_, '_> {
    type Pair = [Array; 2];
    type Error = Exception;
    fn slice_input(&mut self, input: std::ops::Range<i32>) -> Result<Self::Pair, Exception> {
        let keys = self
            .keys
            .try_index_device((.., .., input.clone(), ..), self.stream)?;
        let keys = self.retain(keys)?;
        let values = self
            .values
            .try_index_device((.., .., input, ..), self.stream)?;
        let values = self.retain(values)?;
        Ok([keys, values])
    }

    fn join_tail(&mut self, [key_part, value_part]: Self::Pair) -> Result<Self::Pair, Exception> {
        let keys = match &self.cache.tail_keys {
            Some(previous) => {
                let joined = concatenate_axis(&[previous, &key_part], -2, self.stream);
                drop(key_part);
                joined?
            }
            None => key_part,
        };
        let keys = self.retain(keys)?;
        let values = match &self.cache.tail_values {
            Some(previous) => {
                let joined = concatenate_axis(&[previous, &value_part], -2, self.stream);
                drop(value_part);
                joined?
            }
            None => value_part,
        };
        let values = self.retain(values)?;
        Ok([keys, values])
    }
    fn publish_tail(
        &mut self,
        start: i64,
        end: i64,
        [keys, values]: Self::Pair,
    ) -> Result<(), Exception> {
        let bytes = keys.nbytes() as u64 + values.nbytes() as u64;
        if let Some(claim) = self.original.as_deref_mut() {
            claim.prepare_tail([&keys, &values])?;
            claim.rebalance_host(0, Some(bytes), None, self.stream)?;
            self.cache
                .manager
                .publish_original_tail(claim, bytes, end, false, false)?;
        } else {
            self.cache
                .manager
                .set_tail_state(self.cache.global_layer, bytes, end)
                .map_err(cache_residency_exception)?;
        }
        self.cache.tail_start = start;
        self.cache.tail_keys = Some(keys);
        self.cache.tail_values = Some(values);
        Ok(())
    }
    fn seal_tail(&mut self) -> Result<(), Exception> {
        match self.original.as_deref_mut() {
            Some(claim) => self.cache.seal_tail_original(claim, self.stream),
            None => self.cache.seal_tail(),
        }
    }
    fn finish(&mut self, end: i64, retain_for_attention: bool) -> Result<(), Exception> {
        self.cache.offset = end;
        if !retain_for_attention {
            match self.original.as_deref_mut() {
                Some(claim) => claim.discard_after_visible()?,
                None => self.cache.discard_sliding_history()?,
            }
        }
        Ok(())
    }
}

impl NativeAppend<'_, '_> {
    fn retain(&mut self, value: Array) -> Result<Array, Exception> {
        match self.original.as_deref_mut() {
            Some(claim) => claim.retain(value),
            None => Ok(value),
        }
    }
}
