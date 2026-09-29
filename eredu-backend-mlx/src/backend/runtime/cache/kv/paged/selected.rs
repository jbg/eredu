//! Selected history access shares paging, residency and completion leases with
//! ordinary attention. Only pages containing requested positions are acquired.
use super::*;
use crate::backend::runtime::cache::residency::CacheBlockLease;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use safemlx::ops::indexing::take_axis;

struct SelectedPage(CacheBlockLease);
impl Retention for SelectedPage {
    fn observe(&self, _: Status) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::submission_recovery::{reap, Probe};
    use std::{cell::Cell, rc::Rc};

    struct FailedProbe(Rc<Cell<bool>>);
    impl Probe for FailedProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: self.0.get(),
                failed: true,
                blocked: false,
            }
        }
    }

    #[test]
    fn indexed_failed_completion_retains_page_lease_until_terminal_evidence() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 128, 32768, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        cache
            .update_for_attention(array.clone(), array, &stream)
            .unwrap();
        let id = manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 2, 0)
            .unwrap()
            .remove(0);
        let lease = manager.lease_block(&id, &stream).unwrap();
        let settled = Rc::new(Cell::new(false));
        let recovery = Recovery::with_probe(SelectedPage(lease), FailedProbe(settled.clone()));
        assert!(!recovery.progress().settled);
        drop(recovery);
        reap();
        assert!(
            manager.remove_block(&id).is_err(),
            "failed, unresolved work must keep the block pinned"
        );
        settled.set(true);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            manager.remove_block(&id).is_ok()
        });
    }
}

impl PagedKeyValueCache {
    pub(super) fn selected_attention(
        &mut self,
        input: &eredu_nn::IndexedAttentionInput<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut scratch = 0u64;
        let output = crate::backend::nn::attention::indexed_attention_with_reader(
            input,
            stream,
            |batch, positions, stream| {
                if positions.iter().any(|p| i64::from(*p) >= self.offset) {
                    return Err(Exception::custom(
                        "selected position exceeds retained history",
                    ));
                }
                let mut ids = self
                    .manager
                    .selected_layer_block_ids(
                        self.global_layer,
                        CacheRepresentation::KeyValue,
                        positions,
                    )
                    .map_err(cache_residency_exception)?;
                ids.sort_by_key(|id| id.start);
                // Check coverage before acquiring pages; a sliding window may
                // already have discarded an explicitly requested old position.
                for &position in positions {
                    let position = i64::from(position);
                    if !(position >= self.tail_start
                        && position < self.offset
                        && self.tail_keys.is_some())
                        && !ids
                            .iter()
                            .any(|id| id.start <= position && position < id.end)
                    {
                        return Err(Exception::custom("selected position is no longer retained"));
                    }
                }
                let gather = |keys: &Array, values: &Array, start: i64, end: i64| {
                    let indexes = positions
                        .iter()
                        .filter_map(|p| {
                            let p = i64::from(*p);
                            (p >= start && p < end).then_some((p - start) as i32)
                        })
                        .collect::<Vec<_>>();
                    let indexes = Array::from_slice(&indexes, &[indexes.len() as i32]);
                    let keys = take_axis(
                        keys.try_index_device((batch..batch + 1, .., .., ..), stream)?,
                        &indexes,
                        2,
                        stream,
                    )?;
                    let values = take_axis(
                        values.try_index_device((batch..batch + 1, .., .., ..), stream)?,
                        &indexes,
                        2,
                        stream,
                    )?;
                    // Source leases remain live until both independent compact
                    // copies have completed, including their transfers.
                    safemlx::transforms::eval([&keys, &values])?;
                    Ok::<_, Exception>((keys, values))
                };
                let mut keys = Vec::new();
                let mut values = Vec::new();
                let mut selected_bytes = 0u64;
                let mut pages = self
                    .manager
                    .prefetch_blocks(ids, stream)
                    .map_err(cache_residency_exception)?;
                while let Some(lease) = pages.next_block().map_err(cache_residency_exception)? {
                    self.manager
                        .record_selected_attention(self.global_layer, 1, lease.bytes(), 0)
                        .map_err(cache_residency_exception)?;
                    // An evaluation error alone is not completion evidence.
                    let mut recovery = Recovery::begin(SelectedPage(lease))?;
                    let lease = &recovery.retention().0;
                    let id = lease.id();
                    let CacheBlockArrays::KeyValue {
                        keys: source_keys,
                        values: source_values,
                    } = lease.arrays()
                    else {
                        return Err(Exception::custom(
                            "selected history has incompatible representation",
                        ));
                    };
                    let gathered = gather(source_keys, source_values, id.start, id.end);
                    recovery.seal();
                    let (k, v) = gathered?;
                    let status = recovery.finish();
                    if status.failed || status.blocked {
                        return Err(Exception::custom(
                            "selected page copy failed; unresolved source lease retained",
                        ));
                    }
                    selected_bytes += (k.nbytes() + v.nbytes()) as u64;
                    keys.push(k);
                    values.push(v);
                }
                if positions
                    .last()
                    .is_some_and(|p| i64::from(*p) >= self.tail_start)
                {
                    if let (Some(k), Some(v)) = (&self.tail_keys, &self.tail_values) {
                        let (k, v) = gather(k, v, self.tail_start, self.offset)?;
                        selected_bytes += (k.nbytes() + v.nbytes()) as u64;
                        self.manager
                            .record_selected_attention(
                                self.global_layer,
                                1,
                                (k.nbytes() + v.nbytes()) as u64,
                                0,
                            )
                            .map_err(cache_residency_exception)?;
                        keys.push(k);
                        values.push(v);
                    }
                }
                scratch = scratch.max(selected_bytes.saturating_mul(2));
                Ok((
                    concatenate_axis(&keys, 2, stream)?,
                    concatenate_axis(&values, 2, stream)?,
                ))
            },
        )?;
        self.discard_sliding_history()?;
        self.manager
            .record_selected_attention(self.global_layer, 0, 0, scratch)
            .map_err(cache_residency_exception)?;
        Ok(output)
    }
}
