use super::*;
use eredu_nn::TensorElementType;
use eredu_runtime::ResidentAppendStream;

fn cpu() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}
fn spec(slot: u32) -> AppendStreamSpec {
    AppendStreamSpec {
        slot,
        width: 2,
        element: TensorElementType::I32,
    }
}
fn limits() -> AppendStreamLimits {
    AppendStreamLimits {
        entries: 256,
        page_entries: 2,
        read_entries: 3,
    }
}
fn records(values: &[i32]) -> MlxTensor {
    Array::from_slice(values, &[values.len() as i32 / 2, 2]).into()
}
fn create(manager: &CacheResidencyManager, slot: u32, lane: u32) -> MlxPagedAppendStream {
    MlxPagedAppendStream::new(spec(slot), limits(), manager.clone(), 0, lane, None, 80).unwrap()
}

#[test]
fn combined_prompt_cache_restores_ragged_streams_fixed_state_and_attention() {
    use crate::backend::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, PromptCacheStateArray,
    };
    use eredu_core::cache::{
        AppendStreamPolicy, LayerCachePolicy, MutableStateResidency, PromptCacheDescriptor,
        PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateSegment, PromptCacheTopology,
        StateTensorDimension, StateTensorDtype, StateTensorOwner, StateTensorPolicy,
        StateTensorRole,
    };
    use eredu_core::{AttentionPolicy, LayerSchedule};
    let stream = cpu();
    let options = PagedCacheOptions::new(2, 512, 262144, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let mut attention = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let kv = Array::from_slice(
        &(0..12).map(|n| n as f32 - 4.0).collect::<Vec<_>>(),
        &[2, 1, 6, 1],
    );
    attention.append(kv.clone(), kv, &stream).unwrap();
    attention.finalize().unwrap();
    let mut summaries = create(&manager, 5, 0);
    let values = [i32::MAX, i32::MIN, 16_777_217, -16_777_217, 31, -41];
    summaries.append(0, records(&values), &stream).unwrap();
    // Lane one has no complete visible blocks. Persisting it must still record
    // its zero frontier, not infer a shared frontier from the unpadded lane.
    let role = StateTensorRole::IntegerHistory { slot: 1 };
    let layout = LayerSchedule::new(
        1,
        vec![LayerCachePolicy::key_value_with_state(
            AttentionPolicy::Full,
            1,
            1,
            vec![StateTensorPolicy::new(
                role,
                vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::fixed(3).unwrap(),
                ],
                StateTensorDtype::Int32,
                MutableStateResidency::AlwaysDeviceMutable,
            )
            .unwrap()],
            vec![AppendStreamPolicy::new(5, 2, StateTensorDtype::Int32, 2).unwrap()],
        )
        .unwrap()],
    )
    .unwrap();
    let descriptor = PromptCacheDescriptor::new(
        "fixture",
        "fixture",
        "weights",
        "prefix",
        "layout",
        1,
        0,
        1,
        2,
        layout.clone(),
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
        layout,
        vec![0],
        descriptor.state_segments().to_vec(),
    )
    .unwrap();
    let fixed = Array::from_slice(&values, &[2, 3]);
    let arrays = [PromptCacheStateArray {
        owner: StateTensorOwner::Layer(0),
        role,
        array: &fixed,
    }];
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("combined");
    let tokens = [7, 8, 9, 10, 11, 12];
    assert!(
        manager
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &tokens,
                &arrays,
                &PromptCacheOptions::default()
            )
            .is_err(),
        "unsealed stream tail cannot be persisted"
    );
    summaries.finalize().unwrap();
    let saved = manager
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &tokens,
            &arrays,
            &PromptCacheOptions::default(),
        )
        .unwrap();
    assert_eq!(
        saved
            .stream_frontiers
            .iter()
            .map(|f| (f.slot, f.lane, f.records))
            .collect::<Vec<_>>(),
        [(5, 0, 3), (5, 1, 0)]
    );
    let (restored, manifest) =
        open_prompt_cache(&destination, &descriptor, &identity, &tokens, options).unwrap();
    let state = load_prompt_cache_state_tensors(&destination, &manifest, &stream).unwrap();
    assert_eq!(state.len(), 1);
    assert_eq!(
        MlxTensor::from(state[0].array.clone())
            .to_i32_vec(&stream)
            .unwrap(),
        values
    );
    let mut read = create(&restored, 5, 0);
    assert_eq!(create(&restored, 5, 1).len(), 0);
    assert_eq!(
        PagedKeyValueCache::new(restored.clone(), 0, None)
            .unwrap()
            .offset(),
        6
    );
    // Retained page sources continue to work after the artifact is unlinked.
    std::fs::remove_dir_all(&destination).unwrap();
    assert_eq!(
        read.read(0..3, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        values
    );
    let report = restored.report().unwrap();
    assert_eq!(
        report.imported_retained_shards,
        manifest.blocks.len() as u64
    );
    assert!(report.current_host_bytes <= 262144);
    assert!(report.current_device_bytes <= 512);
}

#[test]
fn append_streams_preserve_integer_rows_and_independent_tails_through_rollback() {
    let stream = cpu();
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 128, 32768, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut first = create(&manager, 0, 0);
    let mut second = create(&manager, 1, 0);
    let mut lane = create(&manager, 0, 1);
    let mut attention = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let kv = Array::from_slice(&[3.0f32], &[1, 1, 1, 1]);
    attention.append(kv.clone(), kv, &stream).unwrap();
    first
        .append(0, records(&[i32::MAX, i32::MIN]), &stream)
        .unwrap();
    second
        .append(0, records(&[16777217, -16777217]), &stream)
        .unwrap();
    lane.append(0, records(&[41, 42]), &stream).unwrap();
    assert_eq!(manager.report().unwrap().mutable_tail_bytes, 32);
    let checkpoint = first.checkpoint_clone_state().unwrap();
    first
        .append(1, records(&[3, 4, 5, 6, 7, 8]), &stream)
        .unwrap();
    second.append(1, records(&[9, 10]), &stream).unwrap();
    assert_eq!(
        first
            .read(0..3, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [i32::MAX, i32::MIN, 3, 4, 5, 6]
    );
    first.restore_checkpoint(&checkpoint, &stream).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(
        second
            .read(0..2, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [16777217, -16777217, 9, 10]
    );
    assert_eq!(
        lane.read(0..1, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [41, 42]
    );
    assert_eq!(attention.offset(), 1);
    assert_eq!(manager.report().unwrap().mutable_tail_bytes, 24);
    first.append(1, records(&[11, 12]), &stream).unwrap();
    first.finalize().unwrap();
    let mut restored = create(&manager, 0, 0);
    assert_eq!(
        restored
            .read(0..2, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [i32::MAX, i32::MIN, 11, 12]
    );
    let report = manager.report().unwrap();
    assert_eq!(report.append_stream_blocks, 2);
    assert_eq!(
        report.logical_cached_tokens, 1,
        "summary records are not attention tokens"
    );
}

#[test]
fn paged_append_ranges_match_resident_across_host_eviction_and_fork() {
    let stream = cpu();
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 32, 32768, 1).unwrap()).unwrap();
    let mut paged = create(&manager, 2, 3);
    let mut resident = ResidentAppendStream::new(spec(2), limits(), 2048, 80, 65536).unwrap();
    for (start, count) in [(0, 1), (1, 3), (4, 2), (6, 1)] {
        let values = (start * 2..(start + count) * 2)
            .map(|n| n as i32 * 19 - 31)
            .collect::<Vec<_>>();
        paged.append(start, records(&values), &stream).unwrap();
        resident.append(start, records(&values), &stream).unwrap();
    }
    assert!(manager.report().unwrap().host_blocks > 0);
    for range in [0..3, 3..6, 5..7, 1..2, 0..3] {
        let expected = resident
            .read(range.clone(), &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap();
        assert_eq!(
            paged
                .read(range, &stream)
                .unwrap()
                .to_i32_vec(&stream)
                .unwrap(),
            expected
        );
        assert!(manager.report().unwrap().current_device_bytes <= 32);
    }
    let fork_manager = manager.fork_session(&stream).unwrap();
    let mut fork = paged.fork_into(fork_manager, &stream).unwrap();
    paged.append(7, records(&[99, 100]), &stream).unwrap();
    fork.append(7, records(&[-99, -100]), &stream).unwrap();
    assert_eq!(
        paged
            .read(7..8, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [99, 100]
    );
    assert_eq!(
        fork.read(7..8, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [-99, -100]
    );
    assert!(matches!(
        paged.read(0..4, &stream),
        Err(AppendStreamError::Budget { .. })
    ));
    assert!(matches!(
        paged.append(7, records(&[1, 2]), &stream),
        Err(AppendStreamError::Frontier { .. })
    ));
}

#[test]
fn append_stream_disk_round_trip_preserves_exact_records() {
    disk_round_trip(cpu());
}

#[test]
#[ignore = "requires native Metal device access"]
fn append_stream_disk_round_trip_metal() {
    disk_round_trip(Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Gpu,
        0,
    )));
}

fn disk_round_trip(stream: Stream) {
    let directory = tempfile::tempdir().unwrap();
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 32, 131072, 1)
            .unwrap()
            .with_live_disk(directory.path(), 65536, 1)
            .unwrap(),
    )
    .unwrap();
    let mut paged = create(&manager, 7, 2);
    for row in 0..64 {
        paged
            .append(
                row,
                records(&[i32::MAX - row as i32, i32::MIN + row as i32]),
                &stream,
            )
            .unwrap();
    }
    let report = manager.report().unwrap();
    assert!(report.disk_blocks > 0);
    for start in [0, 27, 61, 0] {
        let actual = paged
            .read(start..start + 3, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap();
        let expected = (start..start + 3)
            .flat_map(|row| [i32::MAX - row as i32, i32::MIN + row as i32])
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    assert!(manager.report().unwrap().current_device_bytes <= 32);
    paged.clear().unwrap();
    assert_eq!(paged.len(), 0);
    assert_eq!(manager.report().unwrap().append_stream_blocks, 0);
}

#[test]
fn paged_qsa_scores_bounded_tiles_and_reads_only_selected_position_rows() {
    use eredu_architectures::qwen4_exp::qsa::{select_positions, QsaSelectionSpec};
    let stream = cpu();
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 48, 262144, 1).unwrap()).unwrap();
    let mut keys = MlxPagedAppendStream::new(
        AppendStreamSpec {
            slot: 0,
            width: 2,
            element: TensorElementType::F32,
        },
        limits(),
        manager.clone(),
        0,
        0,
        None,
        80,
    )
    .unwrap();
    let mut positions = create(&manager, 1, 0);
    keys.append(
        0,
        Array::from_slice(
            &[
                10.0f32, 0., 0., 4., -1., 5., 2., 2., 0., 1., 1., 0., -2., -3., 1., 1.,
            ],
            &[8, 2],
        )
        .into(),
        &stream,
    )
    .unwrap();
    positions
        .append(
            0,
            records(&[0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            &stream,
        )
        .unwrap();
    let query = Array::from_slice(&[1.0f32, 0., 0., 1.], &[2, 2]).into();
    let mut selection = QsaSelectionSpec {
        heads: 2,
        dimensions: 2,
        ratio: 2,
        token_budget: 4,
        tile_blocks: 2,
        workspace_bytes: 0,
    };
    selection.workspace_bytes = selection.required_workspace().unwrap();
    let checkpoint = keys.checkpoint_clone_state().unwrap();
    let before = manager.report().unwrap();
    assert_eq!(
        select_positions(selection, &query, &mut keys, &mut positions, &[20], &stream).unwrap(),
        [0, 2, 7, 8, 20]
    );
    let after = manager.report().unwrap();
    assert_eq!(
        after.append_stream_read_blocks - before.append_stream_read_blocks,
        6
    );
    assert_eq!(
        after.append_stream_read_bytes - before.append_stream_read_bytes,
        96
    );
    assert_eq!(after.append_stream_scratch_peak_bytes, 32);
    keys.restore_checkpoint(&checkpoint, &stream).unwrap();
    assert_eq!(
        manager.report().unwrap().append_stream_read_bytes,
        after.append_stream_read_bytes
    );
}

#[test]
fn append_stream_admission_and_failed_append_preserve_committed_state() {
    let stream = cpu();
    let too_small =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 8, 0, 1).unwrap()).unwrap();
    assert!(matches!(
        MlxPagedAppendStream::new(spec(0), limits(), too_small, 0, 0, None, 80),
        Err(AppendStreamError::Budget {
            resource: "device page bytes",
            required: 16,
            limit: 8
        })
    ));
    let manager = CacheResidencyManager::new(PagedCacheOptions::new(2, 16, 0, 1).unwrap()).unwrap();
    let mut paged = create(&manager, 0, 0);
    let mut sibling = create(&manager, 1, 0);
    sibling.append(0, records(&[31, 37]), &stream).unwrap();
    paged.append(0, records(&[17, 19]), &stream).unwrap();
    assert!(paged.append(1, records(&[23, 29]), &stream).is_err());
    assert_eq!(paged.len(), 1);
    assert_eq!(
        paged
            .read(0..1, &stream)
            .unwrap()
            .to_i32_vec(&stream)
            .unwrap(),
        [17, 19]
    );
    assert_eq!(manager.report().unwrap().mutable_tail_bytes, 16);
    assert!(matches!(
        MlxPagedAppendStream::new(spec(0), limits(), manager.clone(), 0, 0, None, 80),
        Err(AppendStreamError::Geometry)
    ));
    paged.finalize().unwrap();
    let malformed = AppendStreamSpec {
        width: 1,
        ..spec(0)
    };
    assert!(matches!(
        MlxPagedAppendStream::new(malformed, limits(), manager.clone(), 0, 0, None, 120),
        Err(AppendStreamError::Geometry)
    ));
    let malformed = AppendStreamSpec {
        element: TensorElementType::F32,
        ..spec(0)
    };
    assert!(matches!(
        MlxPagedAppendStream::new(malformed, limits(), manager, 0, 0, None, 80),
        Err(AppendStreamError::Geometry)
    ));
}

#[test]
fn append_stream_failed_read_retains_lease_until_native_completion() {
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
    let stream = cpu();
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 32, 32768, 1).unwrap()).unwrap();
    let mut paged = create(&manager, 4, 0);
    paged
        .append(0, records(&[17, 19, 23, 29]), &stream)
        .unwrap();
    let id = manager
        .layer_block_ids(
            0,
            CacheRepresentation::AppendStream { slot: 4, lane: 0 },
            0,
            2,
            0,
        )
        .unwrap()
        .remove(0);
    let lease = manager.lease_block(&id, &stream).unwrap();
    let settled = Rc::new(Cell::new(false));
    let recovery = Recovery::with_probe(ReadLease(lease), FailedProbe(settled.clone()));
    drop(recovery);
    reap();
    assert!(manager.remove_block(&id).is_err());
    settled.set(true);
    crate::backend::submission_recovery::wait_for_retirement(|| manager.remove_block(&id).is_ok());
}

#[cfg(feature = "metal")]
#[test]
fn qsa_device_selection_matches_portable_and_defers_invalid_scores() {
    use crate::backend::nn::tensor::TokenValidationScope;
    use eredu_architectures::qwen4_exp::qsa::{
        select_positions, select_positions_tensor, QsaSelectionSpec,
    };
    use eredu_nn::Tensor;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let bounds = AppendStreamLimits {
        entries: 32,
        page_entries: 8,
        read_entries: 8,
    };
    let mut keys = ResidentAppendStream::new(
        AppendStreamSpec {
            slot: 0,
            width: 2,
            element: TensorElementType::F32,
        },
        bounds,
        4096,
        4096,
        65536,
    )
    .unwrap();
    let mut positions = ResidentAppendStream::new(spec(1), bounds, 4096, 4096, 65536).unwrap();
    keys.append(
        0,
        Array::from_slice(
            &[
                10.0f32, 0., 0., 4., -1., 5., 2., 2., 0., 1., 1., 0., -2., -3., 1., 1.,
            ],
            &[8, 2],
        )
        .into(),
        &stream,
    )
    .unwrap();
    positions
        .append(
            0,
            records(&[0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            &stream,
        )
        .unwrap();
    let mut selection = QsaSelectionSpec {
        heads: 2,
        dimensions: 2,
        ratio: 2,
        token_budget: 4,
        tile_blocks: 8,
        workspace_bytes: 0,
    };
    selection.workspace_bytes = selection.required_workspace().unwrap();
    let query = Array::from_slice(&[1.0f32, 0., 0., 1.], &[2, 2]).into();
    for budget in [4, 8, 16] {
        selection.token_budget = budget;
        selection.workspace_bytes = selection.required_workspace().unwrap();
        let expected =
            select_positions(selection, &query, &mut keys, &mut positions, &[20], &stream).unwrap();
        let scope = TokenValidationScope::begin().unwrap();
        let actual = select_positions_tensor(
            selection,
            &query,
            &mut keys,
            &mut positions,
            &[20],
            21,
            &stream,
        )
        .unwrap()
        .unwrap();
        let validations = scope.finish();
        safemlx::transforms::eval(
            validations
                .arrays()
                .chain(std::iter::once(actual.as_array())),
        )
        .unwrap();
        validations.validate_completed().unwrap();
        assert_eq!(actual.to_i32_vec(&stream).unwrap(), expected);
    }
    // Invalid data is rejected at the ordinary submission completion boundary.
    let scope = TokenValidationScope::begin().unwrap();
    let scores: MlxTensor = Array::from_slice(&[f32::NAN, 1.], &[2]).into();
    let _ = scores
        .topk_rows(&records(&[0, 1, 2, 3]), 1, &stream)
        .unwrap()
        .unwrap();
    let validations = scope.finish();
    safemlx::transforms::eval(validations.arrays()).unwrap();
    assert!(validations.validate_completed().is_err());
    let scope = TokenValidationScope::begin().unwrap();
    let _ = records(&[0, 2, 2, 3])
        .sorted_unique_indices(4, &stream)
        .unwrap()
        .unwrap();
    let validations = scope.finish();
    safemlx::transforms::eval(validations.arrays()).unwrap();
    assert!(validations.validate_completed().is_err());
}
