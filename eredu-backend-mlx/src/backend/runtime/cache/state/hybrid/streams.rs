//! Exact named-stream bindings and lifecycle within ordinary heterogeneous state.
use super::*;
use crate::backend::runtime::cache::MlxPagedAppendStream;
use eredu_runtime::{
    AppendOnlyStream, AppendStreamBinding, AppendStreamError, AppendStreamSpec,
    ResidentAppendStream, RuntimeAppendStreams,
};

fn exception(error: AppendStreamError) -> Exception {
    Exception::custom(error.to_string())
}

/// Native stream selected by an exact component placement and finite binding.
#[derive(Debug, Clone)]
pub enum MlxAppendStream {
    /// Resident immutable chunks with bounded compact reads.
    Resident(ResidentAppendStream<MlxTensor>),
    /// Pages sharing the model's ordinary cache residency manager.
    Paged(MlxPagedAppendStream),
}
impl AppendOnlyStream<MlxTensor> for MlxAppendStream {
    fn specification(&self) -> &AppendStreamSpec {
        match self {
            Self::Resident(s) => s.specification(),
            Self::Paged(s) => s.specification(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Resident(s) => s.len(),
            Self::Paged(s) => s.len(),
        }
    }
    fn append(
        &mut self,
        frontier: usize,
        values: MlxTensor,
        stream: &Stream,
    ) -> Result<(), AppendStreamError> {
        match self {
            Self::Resident(s) => s.append(frontier, values, stream),
            Self::Paged(s) => s.append(frontier, values, stream),
        }
    }
    fn read(
        &mut self,
        range: Range<usize>,
        stream: &Stream,
    ) -> Result<MlxTensor, AppendStreamError> {
        match self {
            Self::Resident(s) => s.read(range, stream),
            Self::Paged(s) => s.read(range, stream),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct LayerStreams {
    bindings: Vec<AppendStreamBinding>,
    values: Vec<(u32, u32, MlxAppendStream)>,
}
impl LayerStreams {
    pub(super) fn new(
        layer: usize,
        bindings: &[AppendStreamBinding],
        placement: impl Fn(u32) -> Result<StateComponentPlacement, Exception>,
        manager: Option<&CacheResidencyManager>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let mut values = Vec::new();
        for binding in bindings {
            let selected = placement(binding.spec.slot)?;
            for lane in 0..binding.lanes {
                let value = match selected {
                    StateComponentPlacement::Device => MlxAppendStream::Resident(
                        ResidentAppendStream::new(
                            binding.spec.clone(),
                            binding.limits,
                            binding.payload_bytes,
                            binding.scratch_bytes,
                            binding.catalog_bytes,
                        )
                        .map_err(exception)?,
                    ),
                    StateComponentPlacement::Paged => MlxAppendStream::Paged(
                        MlxPagedAppendStream::new(
                            binding.spec.clone(),
                            binding.limits,
                            manager
                                .ok_or_else(|| {
                                    Exception::custom(
                                        "paged append stream requires the shared manager",
                                    )
                                })?
                                .clone(),
                            layer,
                            lane,
                            rank,
                            binding.scratch_bytes,
                        )
                        .map_err(exception)?,
                    ),
                    _ => {
                        return Err(Exception::custom(
                            "selected append-stream placement is not implemented",
                        ))
                    }
                };
                values.push((binding.spec.slot, lane, value));
            }
        }
        Ok(Self {
            bindings: bindings.to_vec(),
            values,
        })
    }
    pub(super) fn retained_values(&self) -> Vec<&MlxTensor> {
        self.values
            .iter()
            .flat_map(|(_, _, value)| match value {
                MlxAppendStream::Resident(s) => s.retained_values().collect::<Vec<_>>(),
                MlxAppendStream::Paged(s) => s.retained_values().collect(),
            })
            .collect()
    }
    pub(super) fn checkpoint(&self) -> Result<Self, Exception> {
        Ok(Self {
            bindings: self.bindings.clone(),
            values: self
                .values
                .iter()
                .map(|(slot, lane, value)| {
                    Ok((
                        *slot,
                        *lane,
                        match value {
                            MlxAppendStream::Resident(s) => MlxAppendStream::Resident(s.clone()),
                            MlxAppendStream::Paged(s) => MlxAppendStream::Paged(
                                s.checkpoint_clone_state().map_err(exception)?,
                            ),
                        },
                    ))
                })
                .collect::<Result<_, Exception>>()?,
        })
    }
    pub(super) fn restore(&mut self, previous: &Self, stream: &Stream) -> Result<(), Exception> {
        if self.bindings != previous.bindings || self.values.len() != previous.values.len() {
            return Err(Exception::custom(
                "append-stream checkpoint bindings differ",
            ));
        }
        for ((slot, lane, current), (old_slot, old_lane, old)) in
            self.values.iter_mut().zip(&previous.values)
        {
            if slot != old_slot || lane != old_lane {
                return Err(Exception::custom(
                    "append-stream checkpoint identity differs",
                ));
            }
            match (current, old) {
                (MlxAppendStream::Resident(current), MlxAppendStream::Resident(old)) => {
                    current.clone_from(old)
                }
                (MlxAppendStream::Paged(current), MlxAppendStream::Paged(old)) => {
                    current.restore_checkpoint(old, stream).map_err(exception)?
                }
                _ => {
                    return Err(Exception::custom(
                        "append-stream checkpoint placement differs",
                    ))
                }
            }
        }
        Ok(())
    }
    pub(super) fn fork(
        &self,
        manager: Option<&CacheResidencyManager>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            bindings: self.bindings.clone(),
            values: self
                .values
                .iter()
                .map(|(slot, lane, value)| {
                    Ok((
                        *slot,
                        *lane,
                        match value {
                            MlxAppendStream::Resident(s) => {
                                MlxAppendStream::Resident(s.try_clone_with(|value| {
                                    value
                                        .as_array()
                                        .contiguous(false, stream)?
                                        .deep_clone()
                                        .map(MlxTensor::from_array)
                                })?)
                            }
                            MlxAppendStream::Paged(s) => MlxAppendStream::Paged(
                                s.fork_into(
                                    manager
                                        .ok_or_else(|| {
                                            Exception::custom(
                                                "forked append stream requires a copied manager",
                                            )
                                        })?
                                        .clone(),
                                    stream,
                                )
                                .map_err(exception)?,
                            ),
                        },
                    ))
                })
                .collect::<Result<_, Exception>>()?,
        })
    }
    pub(super) fn clear(&mut self) -> Result<(), Exception> {
        for (_, _, value) in &mut self.values {
            match value {
                MlxAppendStream::Resident(s) => s.clear(),
                MlxAppendStream::Paged(s) => s.clear().map_err(exception)?,
            }
        }
        Ok(())
    }
    pub(super) fn finalize(&mut self) -> Result<(), Exception> {
        for (_, _, value) in &mut self.values {
            match value {
                MlxAppendStream::Paged(s) => s.finalize().map_err(exception)?,
                MlxAppendStream::Resident(_) => {
                    return Err(Exception::custom(
                        "prompt persistence requires selected paged append streams",
                    ))
                }
            }
        }
        Ok(())
    }
    pub(super) fn paged_count(&self) -> usize {
        self.values
            .iter()
            .filter(|(_, _, s)| matches!(s, MlxAppendStream::Paged(_)))
            .count()
    }
    pub(super) fn metadata_bytes(&self) -> Option<u64> {
        let bindings = (self.bindings.capacity() as u64)
            .checked_mul(std::mem::size_of::<AppendStreamBinding>() as u64)?;
        let values = (self.values.capacity() as u64).checked_mul(std::mem::size_of::<(
            u32,
            u32,
            MlxAppendStream,
        )>() as u64)?;
        self.values
            .iter()
            .try_fold(bindings.checked_add(values)?, |bytes, (_, _, value)| {
                bytes.checked_add(match value {
                    MlxAppendStream::Resident(s) => s.catalog_bytes()?,
                    MlxAppendStream::Paged(_) => 0, // The shared manager charges its page catalog.
                })
            })
    }
    pub(super) fn validate_batch(&self, batch: usize) -> Result<(), Exception> {
        if batch == 0 || self.bindings.iter().any(|b| (b.lanes as usize) < batch) {
            return Err(Exception::custom(
                "prompt-cache batch exceeds append-stream lane capacity",
            ));
        }
        // Bindings reserve maximum batch capacity. Manifests describe active
        // lanes; omitting unused capacity must never discard stored history.
        if self
            .values
            .iter()
            .any(|(_, lane, stream)| (*lane as usize) >= batch && stream.len() != 0)
        {
            return Err(Exception::custom(
                "prompt-cache batch would omit nonempty append-stream history",
            ));
        }
        Ok(())
    }
    pub(super) fn consistent_manager(&self, manager: Option<&CacheResidencyManager>) -> bool {
        self.values.iter().all(|(_, _, s)| match s {
            MlxAppendStream::Resident(_) => true,
            MlxAppendStream::Paged(s) => {
                manager.is_some_and(|m| m.session_id() == s.manager().session_id())
            }
        })
    }
    // Includes every bound lane and future chunk descriptor. Ordinary facade
    // geometry estimates one text lane; this extra allowance remains conservative
    // for batched low-level callers and physically isolated stream snapshots.
    pub(super) fn snapshot_growth(&self, additional: u64) -> Option<u64> {
        self.bindings.iter().try_fold(0_u64, |bytes, binding| {
            let current = self
                .values
                .iter()
                .filter(|(slot, _, _)| *slot == binding.spec.slot)
                .map(|(_, _, value)| value.len() as u64)
                .max()
                .unwrap_or(0);
            let records = current
                .checked_add(additional)?
                .min(binding.limits.entries as u64);
            let chunks = records.div_ceil(binding.limits.page_entries as u64);
            let payload = records
                .checked_mul(binding.spec.record_bytes().ok()?)?
                .checked_mul(2)?;
            bytes.checked_add(
                payload
                    .checked_add(chunks.checked_mul(4096)?)?
                    .checked_mul(u64::from(binding.lanes))?,
            )
        })
    }
}

impl RuntimeAppendStreams<MlxNeuralBackend> for MlxHybridLayerState {
    type Stream = MlxAppendStream;
    fn append_stream(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<&mut Self::Stream, AppendStreamError> {
        self.streams
            .values
            .iter_mut()
            .find(|(s, l, _)| *s == slot && *l == lane)
            .map(|(_, _, v)| v)
            .ok_or(AppendStreamError::Geometry)
    }
    fn append_stream_pair(
        &mut self,
        first: u32,
        second: u32,
        lane: u32,
    ) -> Result<(&mut Self::Stream, &mut Self::Stream), AppendStreamError> {
        let position = |slot| {
            self.streams
                .values
                .iter()
                .position(|(s, l, _)| *s == slot && *l == lane)
                .ok_or(AppendStreamError::Geometry)
        };
        let a = position(first)?;
        let b = position(second)?;
        if a == b {
            return Err(AppendStreamError::Geometry);
        }
        if a < b {
            let (left, right) = self.streams.values.split_at_mut(b);
            Ok((&mut left[a].2, &mut right[0].2))
        } else {
            let (left, right) = self.streams.values.split_at_mut(a);
            Ok((&mut right[0].2, &mut left[b].2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::cache::{
        AppendStreamPolicy, MutableStateResidency, StateTensorDtype, StateTensorPolicy,
    };
    use eredu_core::{AttentionPolicy, LayerSchedule};
    use eredu_nn::{Tensor, TensorElementType};
    use eredu_runtime::AppendStreamLimits;

    fn fixture() -> (StateLayout, Vec<AppendStreamBinding>) {
        let streams = [5, 9]
            .map(|slot| AppendStreamPolicy::new(slot, 2, StateTensorDtype::Int32, 1).unwrap());
        let fixed = StateTensorPolicy::new(
            StateTensorRole::IntegerHistory { slot: 0 },
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(2).unwrap(),
            ],
            StateTensorDtype::Int32,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .unwrap();
        let layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value_with_state(
                    AttentionPolicy::Full,
                    1,
                    2,
                    vec![fixed],
                    streams.to_vec(),
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let bindings = [5, 9]
            .map(|slot| AppendStreamBinding {
                layer: 0,
                lanes: 2,
                spec: AppendStreamSpec {
                    slot,
                    width: 2,
                    element: TensorElementType::I32,
                },
                limits: AppendStreamLimits {
                    entries: 32,
                    page_entries: 2,
                    read_entries: 3,
                },
                payload_bytes: 256,
                scratch_bytes: 80,
                catalog_bytes: 65536,
            })
            .to_vec();
        (layout, bindings)
    }

    fn step(state: &mut MlxHybridState, value: i32, stream: &Stream) {
        step_for_batch(state, value, 2, stream);
    }
    fn step_for_batch(state: &mut MlxHybridState, value: i32, batch: usize, stream: &Stream) {
        let layer = &mut state.layers_mut()[0];
        let kv: MlxTensor = Array::from_slice(
            &[0.5f32, -0.75, 1.25, 2.0][..batch * 2],
            &[batch as i32, 1, 1, 2],
        )
        .into();
        AttentionCache::update_for_attention(layer, kv.clone(), kv, stream).unwrap();
        // Exercise reversed mutable-pair lookup and keep the second batch lane empty.
        let (b, a) = layer.append_stream_pair(9, 5, 0).unwrap();
        for (s, delta) in [(a, 0), (b, 1)] {
            let tensor: MlxTensor = Array::from_slice(&[value + delta, -value], &[1, 2]).into();
            s.append(s.len(), tensor, stream).unwrap();
        }
        *layer
            .fixed_component(StateTensorRole::IntegerHistory { slot: 0 })
            .unwrap() = Some(
            Array::from_slice(&[value, 17, 23, -value][..batch * 2], &[batch as i32, 2]).into(),
        );
    }

    fn last(state: &mut MlxHybridState, stream: &Stream) -> Vec<i32> {
        let value = state.layers_mut()[0].append_stream(5, 0).unwrap();
        let n = value.len();
        value
            .read(n - 1..n, stream)
            .unwrap()
            .to_i32_vec(stream)
            .unwrap()
    }

    #[test]
    fn combined_stream_lifecycle_preserves_checkpoints_forks_isolation_and_reset() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        exercise_lifecycle(&stream);
    }

    #[test]
    #[ignore = "requires Metal device access"]
    fn combined_stream_state_lifecycle_and_persistence_metal() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        exercise_lifecycle(&stream);
        exercise_persistence(&stream, 2);
    }

    fn exercise_lifecycle(stream: &Stream) {
        for paged in [false, true] {
            let (layout, bindings) = fixture();
            let manager = CacheResidencyManager::new(
                eredu_runtime::PagedCacheOptions::new(2, 256, 262144, 1)
                    .unwrap()
                    .with_full_attention(true)
                    .with_pool(eredu_runtime::CacheResidencyPool::new(
                        eredu_runtime::CachePoolLimits::new(1024, 1048576, 1048576, 0).unwrap(),
                    ))
                    .unwrap(),
            )
            .unwrap();
            let mut state = if paged {
                MlxHybridState::paged(layout, manager, None, &bindings).unwrap()
            } else {
                MlxHybridState::device(layout, &bindings).unwrap()
            };
            for value in [17, 23, 16_777_217] {
                step(&mut state, value, &stream);
            }
            let checkpoint = state.deep_clone_state().unwrap();
            let mut fork = state.fork_prediction_target_state(&stream).unwrap();
            let mut isolated = state.isolated_snapshot(&stream).unwrap();
            step(&mut state, 99, &stream);
            step(&mut fork, 71, &stream);
            assert_eq!(last(&mut state, &stream), [99, -99]);
            assert_eq!(last(&mut fork, &stream), [71, -71]);
            assert_eq!(last(&mut isolated, &stream), [16_777_217, -16_777_217]);
            state.restore_checkpoint(&checkpoint, &stream).unwrap();
            assert_eq!(state.offset(), 3);
            assert_eq!(last(&mut state, &stream), [16_777_217, -16_777_217]);
            assert_eq!(state.layers_mut()[0].append_stream(9, 1).unwrap().len(), 0);
            assert!(state.layers_mut()[0].append_stream_pair(5, 5, 0).is_err());
            assert!(state.layers_mut()[0].append_stream(5, 2).is_err());
            let report = state.residency_report().unwrap();
            state.clear().unwrap();
            assert_eq!(state.offset(), 0);
            assert_eq!(state.layers_mut()[0].append_stream(5, 0).unwrap().len(), 0);
            if let Some(before) = report {
                let after = state.residency_report().unwrap().unwrap();
                assert_eq!(
                    before.append_stream_read_bytes,
                    after.append_stream_read_bytes
                );
                assert_eq!(before.transfer_bytes, after.transfer_bytes);
            }
            assert_eq!(last(&mut fork, &stream), [71, -71]);
            assert_eq!(last(&mut isolated, &stream), [16_777_217, -16_777_217]);
        }
    }

    #[test]
    fn combined_construction_requires_complete_exact_bounded_stream_bindings() {
        let (layout, bindings) = fixture();
        assert!(MlxHybridState::device(layout.clone(), &[]).is_err());
        for modify in [
            (|b: &mut Vec<AppendStreamBinding>| {
                b[1].spec.slot = 5;
            }) as fn(&mut Vec<AppendStreamBinding>),
            |b| b[0].spec.element = TensorElementType::F32,
            |b| b[0].spec.width = 3,
            |b| b[0].lanes = 0,
            |b| b[0].lanes = 3,
            |b| b[0].scratch_bytes = 79,
            |b| b[0].payload_bytes = 255,
            |b| b[0].catalog_bytes = 0,
            |b| b[0].layer = 1,
        ] {
            let mut invalid = bindings.clone();
            modify(&mut invalid);
            assert!(MlxHybridState::device(layout.clone(), &invalid).is_err());
        }
    }

    #[test]
    fn combined_state_owner_finalizes_and_restores_every_persisted_stream() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        exercise_persistence(&stream, 2);
    }

    #[test]
    fn combined_state_persistence_accepts_unused_lane_capacity() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        exercise_persistence(&stream, 1);
    }

    #[test]
    fn prompt_cache_cannot_discard_nonempty_inactive_stream_lanes() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let (layout, bindings) = fixture();
        let mut state = MlxHybridState::device(layout, &bindings).unwrap();
        let layer = &mut state.layers_mut()[0];
        assert!(layer.streams.validate_batch(0).is_err());
        assert!(layer.streams.validate_batch(3).is_err());
        layer.streams.validate_batch(1).unwrap();
        layer
            .append_stream(5, 1)
            .unwrap()
            .append(0, Array::from_slice(&[19i32, -19], &[1, 2]).into(), &stream)
            .unwrap();
        assert!(layer.streams.validate_batch(1).is_err());
        layer.streams.validate_batch(2).unwrap();
    }

    fn exercise_persistence(stream: &Stream, batch: usize) {
        use crate::backend::runtime::cache::residency::{
            load_prompt_cache_state_tensors, open_prompt_cache,
        };
        use eredu_core::cache::{
            PromptCacheModelIdentity, PromptCacheStateSegment, PromptCacheTopology,
        };
        let (layout, bindings) = fixture();
        let options = eredu_runtime::PagedCacheOptions::new(2, 512, 262144, 1)
            .unwrap()
            .with_full_attention(true);
        let mut state = MlxHybridState::paged(
            layout.clone(),
            CacheResidencyManager::new(options.clone()).unwrap(),
            None,
            &bindings,
        )
        .unwrap();
        for value in [19, 71, 16_777_217] {
            step_for_batch(&mut state, value, batch, &stream);
        }
        let descriptor = PromptCacheDescriptor::new(
            "fixture",
            "fixture",
            "weights",
            "prefix",
            "layout",
            1,
            0,
            1,
            batch,
            layout.layers().clone(),
            vec![0],
            vec![PromptCacheStateSegment::new("state", 0..1).unwrap()],
            0,
            PromptCacheTopology::default(),
        )
        .unwrap();
        let identity = PromptCacheModelIdentity::new(
            "fixture",
            "fixture",
            "layout",
            1,
            0,
            1,
            0,
            PromptCacheTopology::default(),
            layout.layers().clone(),
            vec![0],
            descriptor.state_segments().to_vec(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("state");
        let saved = state
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &[1, 2, 3],
                &PromptCacheOptions::default(),
            )
            .unwrap();
        assert_eq!(
            saved
                .stream_frontiers
                .iter()
                .map(|f| f.records)
                .collect::<Vec<_>>(),
            if batch == 1 {
                vec![3, 3]
            } else {
                vec![3, 0, 3, 0]
            }
        );
        let (manager, manifest) =
            open_prompt_cache(&destination, &descriptor, &identity, &[1, 2, 3], options).unwrap();
        let arrays = load_prompt_cache_state_tensors(&destination, &manifest, &stream).unwrap();
        let mut restored = MlxHybridState::paged(layout, manager, None, &bindings).unwrap();
        restored
            .restore_prompt_cache_state(arrays, 3, &[0])
            .unwrap();
        assert_eq!(restored.offset(), 3);
        assert_eq!(last(&mut restored, &stream), [16_777_217, -16_777_217]);
        let fixed = restored.layers_mut()[0]
            .fixed_component(StateTensorRole::IntegerHistory { slot: 0 })
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(
            fixed.to_i32_vec(&stream).unwrap(),
            [16_777_217, 17, 23, -16_777_217][..batch * 2]
        );
        assert_eq!(
            restored.layers_mut()[0].append_stream(5, 1).unwrap().len(),
            0
        );
        step_for_batch(&mut restored, 101, batch, &stream);
        assert_eq!(last(&mut restored, &stream), [101, -101]);
        assert_eq!(last(&mut state, &stream), [16_777_217, -16_777_217]);
    }
}
