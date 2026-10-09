//! Nonzero atomic publication and bounded copying across two native cache families.

use eredu_backend_mlx::{
    backend::runtime::cache::state::{
        JointCopyLimits, JointStateError, MlxHybridState, MlxJointState, MlxJointStateBranch,
        MlxKeyValueState,
    },
    MlxTensor,
};
use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    },
    execution_control::{SnapshotLimits, SnapshotResourceKind},
    scheduler::SemanticStateTransaction,
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::AttentionCache;
use eredu_runtime::execution_control::{
    IsolatedStateCopy, IsolatedStateError, IsolatedStateSnapshot, SnapshotBudget,
    SnapshotReservation,
};
use eredu_runtime::{LayerRuntimeState, RuntimeState, RuntimeStateComponents, StateLayout};
use safemlx::{Array, Device, DeviceType, Stream};

fn states(stream: &Stream, batch: i32) -> (MlxHybridState, MlxKeyValueState) {
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(2).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::fixed_only(vec![fixed]).unwrap()]).unwrap(),
    )
    .unwrap();
    let mut hybrid = MlxHybridState::device(layout, &[]).unwrap();
    *hybrid
        .layer(0)
        .unwrap()
        .fixed_component(StateTensorRole::Recurrent)
        .unwrap() = Some(MlxTensor::from_array(Array::from_slice(
        &[1f32, 2.],
        &[1, 2],
    )));
    hybrid.layer(0).unwrap().advance_fixed(2).unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut kv = MlxKeyValueState::device(layout).unwrap();
    let values = (1..=batch * 4).map(|n| n as f32).collect::<Vec<_>>();
    let keys = MlxTensor::from_array(Array::from_slice(&values, &[batch, 1, 2, 2]));
    let values = MlxTensor::from_array(Array::from_slice(
        &values.iter().map(|n| -*n).collect::<Vec<_>>(),
        &[batch, 1, 2, 2],
    ));
    kv.as_mut()[0]
        .update_for_attention(keys, values, stream)
        .unwrap();
    (hybrid, kv)
}

fn owner(stream: &Stream, batch: i32, limits: JointCopyLimits) -> MlxJointState {
    let (hybrid, kv) = states(stream, batch);
    MlxJointState::new(hybrid, kv, stream, limits).unwrap()
}

fn unlimited() -> JointCopyLimits {
    JointCopyLimits {
        max_branch_bytes: 1 << 20,
        max_total_copy_bytes: 1 << 24,
    }
}

fn values(arrays: Vec<&Array>) -> Vec<Vec<f32>> {
    arrays
        .into_iter()
        .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        .collect()
}

fn advance(branch: &mut MlxJointStateBranch, stream: &Stream) {
    let (hybrid, kv) = branch.parts_mut();
    *hybrid
        .layer(0)
        .unwrap()
        .fixed_component(StateTensorRole::Recurrent)
        .unwrap() = Some(MlxTensor::from_array(Array::from_slice(
        &[11f32, 12.],
        &[1, 2],
    )));
    hybrid.layer(0).unwrap().advance_fixed(1).unwrap();
    let keys = MlxTensor::from_array(Array::from_slice(
        &[101f32, 102., 103., 104.],
        &[2, 1, 1, 2],
    ));
    let values = MlxTensor::from_array(Array::from_slice(
        &[-101f32, -102., -103., -104.],
        &[2, 1, 1, 2],
    ));
    kv[0].update_for_attention(keys, values, stream).unwrap();
    // The caller establishes exact execution completion before publication.
    safemlx::transforms::async_eval_with_event(branch.retained_arrays())
        .unwrap()
        .synchronize()
        .unwrap();
}

fn copy_budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 8,
        max_branches: 8,
        retained_bytes: 1 << 24,
        cumulative_copy_bytes: 1 << 26,
    })
}

fn isolated_copy(
    state: &MlxJointState,
    budget: &SnapshotBudget,
) -> (MlxJointState, SnapshotReservation) {
    let lease = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(state.estimate_isolated_copy().unwrap()),
        )
        .unwrap();
    (state.copy_isolated(&lease).unwrap(), lease)
}

#[test]
fn isolated_joint_copy_allows_continuation_and_restores_both_nonzero_caches() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let budget = copy_budget();
    let mut canonical = owner(&stream, 2, unlimited());
    let baseline = values(canonical.retained_arrays());
    let cost = canonical.copy_estimate().unwrap().copy_bytes;
    let saved = IsolatedStateSnapshot::capture(&canonical, &budget).unwrap();
    assert_eq!(canonical.copy_bytes_consumed(), cost);
    let (mut child, _child_lease) = isolated_copy(&canonical, &budget);

    // Retaining the snapshot does not reserve the source's execution lane.
    let mut branch = canonical.branch().unwrap();
    advance(&mut branch, &stream);
    canonical.commit_branch(branch).unwrap();
    let continued = values(canonical.retained_arrays());
    assert_ne!(continued, baseline);

    let mut branch = child.branch().unwrap();
    advance(&mut branch, &stream);
    child.commit_branch(branch).unwrap();
    let mut branch = child.branch().unwrap();
    advance(&mut branch, &stream);
    child.commit_branch(branch).unwrap();
    assert_eq!(child.parts().0.offset(), 4);
    assert_eq!(AttentionCache::offset(&child.parts().1.as_ref()[0]), 4);
    assert_eq!(values(canonical.retained_arrays()), continued);
    let consumed = 2 * cost;
    assert_eq!(canonical.copy_bytes_consumed(), consumed);
    assert_eq!(child.copy_bytes_consumed(), consumed);

    saved.restore(&mut canonical).unwrap();
    assert_eq!(values(canonical.retained_arrays()), baseline);
    assert_eq!(canonical.parts().0.offset(), 2);
    assert_eq!(AttentionCache::offset(&canonical.parts().1.as_ref()[0]), 2);
    assert_eq!(canonical.copy_bytes_consumed(), consumed + cost);
    assert_eq!(child.copy_bytes_consumed(), consumed + cost);
    assert_eq!(child.parts().0.offset(), 4);
    let retained = budget.usage().retained_bytes;
    saved.restore(&mut canonical).unwrap();
    assert_eq!(values(canonical.retained_arrays()), baseline);
    assert_eq!(budget.usage().retained_bytes, retained);
    assert_eq!(canonical.copy_bytes_consumed(), consumed + 2 * cost);
}

#[test]
fn isolated_joint_restore_rejects_foreign_or_outstanding_owners() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let budget = copy_budget();
    let mut canonical = owner(&stream, 2, unlimited());
    let foreign = owner(&stream, 2, unlimited());
    assert!(matches!(
        canonical.restore_isolated(foreign),
        Err(JointStateError::Identity)
    ));
    let baseline = values(canonical.retained_arrays());
    let (saved, _lease) = isolated_copy(&canonical, &budget);
    let branch = canonical.branch().unwrap();
    assert!(matches!(
        canonical.restore_isolated(saved),
        Err(JointStateError::Busy)
    ));
    assert_eq!(values(canonical.retained_arrays()), baseline);
    drop(branch);

    let (saved, _lease) = isolated_copy(&canonical, &budget);
    let branch = saved.branch().unwrap();
    assert!(matches!(
        canonical.validate_isolated_restore(&saved),
        Err(JointStateError::Busy)
    ));
    drop(branch);
    canonical.restore_isolated(saved).unwrap();
    assert_eq!(values(canonical.retained_arrays()), baseline);
}

#[test]
fn isolated_joint_descendants_share_nonrefundable_copy_allowance() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let cost = owner(&stream, 2, unlimited())
        .copy_estimate()
        .unwrap()
        .copy_bytes;
    let canonical = owner(
        &stream,
        2,
        JointCopyLimits {
            max_branch_bytes: cost,
            max_total_copy_bytes: cost,
        },
    );
    let budget = copy_budget();
    let (saved, lease) = isolated_copy(&canonical, &budget);
    let baseline = values(saved.retained_arrays());
    assert!(matches!(
        saved.copy_isolated(&lease),
        Err(JointStateError::Budget)
    ));
    drop(canonical.branch().unwrap());
    assert_eq!(canonical.copy_bytes_consumed(), cost);
    assert_eq!(saved.copy_bytes_consumed(), cost);
    assert_eq!(values(saved.retained_arrays()), baseline);
    drop(saved);
    drop(lease);
    assert_eq!(canonical.copy_bytes_consumed(), cost);
    assert_eq!(budget.usage().retained_bytes, 0);
    assert_eq!(budget.usage().cumulative_copy_bytes, cost);
}

#[test]
fn realtime_snapshot_restores_native_draws_with_voicechat_clocks_and_history() {
    use eredu_architectures::voicechat::{
        agent_sampling::{AgentSampler, AgentSamplingPolicy},
        config::VoiceChatConfig,
        schedule::{AudioSource, ScheduleLimits, SchedulePolicy, ScheduleState, Step},
    };
    use eredu_backend_mlx::backend::{
        random::RandomState, runtime::generation::MlxSamplingBackend,
    };
    use eredu_nn::Tensor;
    use eredu_runtime::{DenseSamplingBackend, RealtimeGenerationState};

    struct Ready;
    impl eredu_core::Completion for Ready {
        type Error = std::convert::Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    type Run = RealtimeGenerationState<
        MlxJointState,
        ScheduleState,
        AgentSamplingPolicy,
        AgentSampler,
        RandomState,
        Ready,
    >;
    let config = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let policy = SchedulePolicy::from_config(
        &config,
        ScheduleLimits {
            max_audio_frames: 32,
            max_backbone_positions: 128,
            max_tts_positions: 128,
            max_codec_frames: 32,
        },
    )
    .unwrap();
    let mut schedule = ScheduleState::new(policy.clone());
    for step in [Step::Prefill { backbone_tokens: 5 }, Step::LiveAudio] {
        let plan = schedule.plan(step).unwrap();
        let expected = *plan.expected();
        schedule = plan.complete(expected).unwrap();
    }
    let baseline_clocks = *schedule.progress();
    let sampling = AgentSamplingPolicy::new(0.7, 0.9, 1.1, 32).unwrap();
    let mut sampler = AgentSampler::new(sampling);
    sampler.set_history(&[64, 65, 64]).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut random = RandomState::with_seed(231).unwrap();
    MlxSamplingBackend::uniform_f32(&[5], &mut random, &stream)
        .unwrap()
        .to_f32_vec(&stream)
        .unwrap();
    // This tests composition of copied components, not a coupled model forward.
    let mut run = Run::new(
        owner(&stream, 2, unlimited()),
        policy.generation_contract(),
        schedule,
        sampling,
        vec![sampler],
        Some(random),
    )
    .unwrap();
    let baseline = values(run.model_state().retained_arrays());
    let budget = copy_budget();
    let saved = IsolatedStateSnapshot::capture(&run, &budget).unwrap();

    let mut branch = run.branch().unwrap();
    let drawn = branch
        .draw_standard_normal::<MlxSamplingBackend>(&[9], &stream)
        .unwrap()
        .to_f32_vec(&stream)
        .unwrap();
    advance(branch.model_state_mut(), &stream);
    let plan = branch
        .schedule_state()
        .plan(Step::Tool {
            audio: AudioSource::Retained,
        })
        .unwrap();
    let expected = *plan.expected();
    *branch.schedule_state_mut() = plan.complete(expected).unwrap();
    branch.samplers_mut()[0].accept_effective(67).unwrap();
    branch.attach_submission_completion(Ready).unwrap();
    run.commit_branch(branch).unwrap();
    assert_eq!(run.samplers()[0].history(), &[64, 65, 64, 67]);
    assert_ne!(values(run.model_state().retained_arrays()), baseline);
    assert_eq!(run.schedule_state().progress().perception_frames, 1);
    assert_eq!(run.schedule_state().progress().backbone_positions, 7);

    saved.restore(&mut run).unwrap();
    assert_eq!(run.revision(), 2);
    assert_eq!(*run.schedule_state().progress(), baseline_clocks);
    assert_eq!(run.sampling(), &sampling);
    assert_eq!(run.samplers()[0].history(), &[64, 65, 64]);
    assert_eq!(values(run.model_state().retained_arrays()), baseline);
    let mut replay = run.branch().unwrap();
    assert_eq!(
        replay
            .draw_standard_normal::<MlxSamplingBackend>(&[9], &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap(),
        drawn
    );
    Run::discard_branch(replay).unwrap();
}

#[test]
fn joint_state_commits_both_candidates_and_rejects_foreign_publication() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut canonical = owner(&stream, 2, unlimited());
    let baseline = values(canonical.retained_arrays());
    let mut branch = canonical.branch().unwrap();
    assert!(matches!(canonical.branch(), Err(JointStateError::Busy)));
    advance(&mut branch, &stream);
    assert_eq!(values(canonical.retained_arrays()), baseline);
    MlxJointState::discard_branch(branch).unwrap();
    assert_eq!(values(canonical.retained_arrays()), baseline);
    let mut foreign = owner(&stream, 2, unlimited());
    let foreign_baseline = values(foreign.retained_arrays());
    let mut branch = canonical.branch().unwrap();
    advance(&mut branch, &stream);
    assert!(matches!(
        foreign.commit_branch(branch),
        Err(JointStateError::Identity)
    ));
    assert_eq!(values(canonical.retained_arrays()), baseline);
    assert_eq!(values(foreign.retained_arrays()), foreign_baseline);
    let mut branch = canonical.branch().unwrap();
    advance(&mut branch, &stream);
    canonical.commit_branch(branch).unwrap();
    assert_eq!(canonical.parts().0.offset(), 3);
    assert_eq!(AttentionCache::offset(&canonical.parts().1.as_ref()[0]), 3);
    assert_eq!(
        values(canonical.retained_arrays()),
        vec![
            vec![11., 12.],
            vec![1., 2., 3., 4., 101., 102., 5., 6., 7., 8., 103., 104.],
            vec![-1., -2., -3., -4., -101., -102., -5., -6., -7., -8., -103., -104.],
        ]
    );
}

#[test]
fn joint_state_prices_both_batch_lanes_and_reserves_copy_allowance_for_isolation() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let one = owner(&stream, 1, unlimited());
    let two = owner(&stream, 2, unlimited());
    assert_eq!(one.retained_tensor_bytes().unwrap(), 40);
    assert_eq!(two.retained_tensor_bytes().unwrap(), 72);
    assert_eq!(
        two.copy_estimate().unwrap().copy_bytes - one.copy_estimate().unwrap().copy_bytes,
        64
    );
    let cost = two.copy_estimate().unwrap().copy_bytes;
    let limited = owner(
        &stream,
        2,
        JointCopyLimits {
            max_branch_bytes: cost,
            max_total_copy_bytes: cost,
        },
    );
    let branch = limited.branch().unwrap();
    assert_eq!(limited.copy_bytes_consumed(), 0);
    MlxJointState::discard_branch(branch).unwrap();
    let budget = copy_budget();
    let (saved, lease) = isolated_copy(&limited, &budget);
    assert_eq!(limited.copy_bytes_consumed(), cost);
    assert!(matches!(
        saved.copy_isolated(&lease),
        Err(JointStateError::Budget)
    ));
    // Reaching the copy limit does not block immutable frame candidates.
    drop(limited.branch().unwrap());
    assert_eq!(limited.copy_bytes_consumed(), cost);
    let too_small = owner(
        &stream,
        2,
        JointCopyLimits {
            max_branch_bytes: cost - 1,
            max_total_copy_bytes: cost,
        },
    );
    assert!(matches!(too_small.branch(), Err(JointStateError::Budget)));
    assert_eq!(too_small.copy_bytes_consumed(), 0);
}

#[test]
fn joint_state_failed_second_copy_preserves_state_and_releases_finite_pool_admission() {
    use eredu_backend_mlx::backend::runtime::cache::residency::CacheResidencyManager;
    use eredu_core::cache::CacheRepresentation;
    use eredu_runtime::{CachePoolLimits, CacheResidencyPool, PagedCacheOptions};

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let pool = CacheResidencyPool::new(CachePoolLimits::new(192, 192, 192, 0).unwrap());
    let options = PagedCacheOptions::new(4, 192, 192, 1)
        .unwrap()
        .with_full_attention(true)
        .with_pool(pool.clone())
        .unwrap();
    let (hybrid, device) = states(&stream, 2);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let original_namespace = manager.session_id();
    let mut kv = MlxKeyValueState::paged(device.layout().clone(), manager, None).unwrap();
    kv.as_mut()[0]
        .update_for_attention(
            MlxTensor::from_array(Array::from_slice(
                &[1f32, 2., 3., 4., 5., 6., 7., 8.],
                &[2, 1, 2, 2],
            )),
            MlxTensor::from_array(Array::from_slice(
                &[-1f32, -2., -3., -4., -5., -6., -7., -8.],
                &[2, 1, 2, 2],
            )),
            &stream,
        )
        .unwrap();
    let mut canonical = MlxJointState::new(hybrid, kv, &stream, unlimited()).unwrap();
    let baseline = values(canonical.retained_arrays());
    let cost = canonical.copy_estimate().unwrap().copy_bytes;
    let blocker = CacheResidencyManager::new(options).unwrap();
    blocker
        .set_tail_state(0, CacheRepresentation::KeyValue, 128, 2)
        .unwrap();
    assert_eq!(pool.report().unwrap().current_device_bytes, 192);
    // The first, device-only hybrid copy succeeds. The second state's copied
    // manager cannot reserve its 64-byte tail in this same finite process pool.
    let budget = copy_budget();
    let failed = IsolatedStateSnapshot::capture(&canonical, &budget)
        .err()
        .expect("second copy exceeds pool");
    assert!(matches!(failed, IsolatedStateError::Backend(_)));
    assert!(failed.to_string().contains("budget"));
    assert_eq!(values(canonical.retained_arrays()), baseline);
    assert_eq!(canonical.copy_bytes_consumed(), cost);
    drop(blocker);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut branch = loop {
        match canonical.branch() {
            Ok(branch) => break branch,
            Err(JointStateError::Busy) if std::time::Instant::now() < deadline => {
                std::thread::yield_now()
            }
            Err(error) => panic!("settled failed copy prevented retry: {error}"),
        }
    };
    assert_eq!(canonical.copy_bytes_consumed(), cost * 2);
    assert_eq!(budget.usage().retained_bytes, 0);
    assert_eq!(budget.usage().snapshots, 0);
    assert_eq!(budget.usage().cumulative_copy_bytes, cost);
    assert_eq!(pool.report().unwrap().current_device_bytes, 128);
    let (_, caches) = branch.parts_mut();
    let eredu_backend_mlx::backend::runtime::cache::state::MlxKeyValueLayerState::Paged(cache) =
        &caches[0]
    else {
        panic!("paged selection changed")
    };
    assert_ne!(cache.manager().session_id(), original_namespace);
    advance(&mut branch, &stream);
    assert_eq!(values(canonical.retained_arrays()), baseline);
    canonical.commit_branch(branch).unwrap();
    assert_eq!(AttentionCache::offset(&canonical.parts().1.as_ref()[0]), 3);
    assert_eq!(pool.report().unwrap().current_device_bytes, 96);
}

#[test]
fn paged_candidates_share_sealed_roots_and_isolate_catalog_updates() {
    use eredu_backend_mlx::backend::runtime::cache::{
        residency::{CacheBlockArrays, CacheResidencyManager},
        state::MlxKeyValueLayerState,
    };
    use eredu_core::cache::CacheRepresentation;
    use eredu_runtime::PagedCacheOptions;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let (hybrid, device) = states(&stream, 2);
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 8)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut kv = MlxKeyValueState::paged(device.layout().clone(), manager.clone(), None).unwrap();
    let input = MlxTensor::from_array(Array::from_slice(
        &[1_f32, 2., 3., 4., 5., 6., 7., 8.],
        &[2, 1, 2, 2],
    ));
    kv.as_mut()[0]
        .update_for_attention(input.clone(), input, &stream)
        .unwrap();
    let mut canonical = MlxJointState::new(hybrid, kv, &stream, unlimited()).unwrap();
    let ids = manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 2, 0)
        .unwrap();
    assert_eq!(ids.len(), 1);
    let source = manager.lease_block(&ids[0], &stream).unwrap();
    let CacheBlockArrays::KeyValue { keys, .. } = source.arrays() else {
        panic!("KV fixture")
    };
    let source_values = keys.evaluated().unwrap();
    let mut branch = canonical.branch().unwrap();
    let (_, caches) = branch.parts_mut();
    let MlxKeyValueLayerState::Paged(cache) = &caches[0] else {
        panic!("paged fixture")
    };
    let candidate_manager = cache.manager().clone();
    assert_ne!(candidate_manager.session_id(), manager.session_id());
    let candidate_ids = candidate_manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 2, 0)
        .unwrap();
    let candidate = candidate_manager
        .lease_block(&candidate_ids[0], &stream)
        .unwrap();
    let CacheBlockArrays::KeyValue { keys, .. } = candidate.arrays() else {
        panic!("KV fixture")
    };
    let candidate_values = keys.evaluated().unwrap();
    assert_eq!(
        source_values.as_slice::<f32>().as_ptr(),
        candidate_values.as_slice::<f32>().as_ptr(),
        "sealed payload stays shared"
    );
    advance(&mut branch, &stream);
    assert_eq!(AttentionCache::offset(&canonical.parts().1.as_ref()[0]), 2);
    MlxJointState::discard_branch(branch).unwrap();
    assert_eq!(
        manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 4, 0)
            .unwrap(),
        ids
    );
    assert_eq!(
        source_values.as_slice::<f32>(),
        &[1., 2., 3., 4., 5., 6., 7., 8.]
    );
    let mut branch = canonical.branch().unwrap();
    advance(&mut branch, &stream);
    canonical.commit_branch(branch).unwrap();
    assert_eq!(AttentionCache::offset(&canonical.parts().1.as_ref()[0]), 3);
}
