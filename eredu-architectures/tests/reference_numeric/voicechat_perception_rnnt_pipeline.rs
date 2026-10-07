//! Paired acoustic/transcription candidates through actual component pipelines.
use super::voicechat_mog_residency::weights;
use super::voicechat_perception_distributed::PipelinePerception;
use super::voicechat_perception_residency::{assert_tensor, tensor};
use super::voicechat_rnnt::dimensions;
use super::voicechat_rnnt_coordination::coordinator_pair;
use super::voicechat_rnnt_predictor::{compare_state as compare_rnnt, PipelinePredictor};
use super::voicechat_rnnt_residency::{same_state as same_rnnt, Policy};
use super::*;
use eredu_architectures::voicechat::{
    perception_layered::PerceptionRuntimeState,
    perception_rnnt::{PerceptionRnntLimits, PerceptionRnntState},
    perception_rnnt_pipeline,
    rnnt::RnntLimits,
    rnnt_pipeline::RnntPipelineCursor,
    rnnt_projection_layered::{LayerwiseRnntProjection, RnntProjectionArchitecture},
};
use eredu_core::DistributedCommitEpoch;
use serde_json::Value;
type State = PerceptionRnntState<NumericTensor, PerceptionRuntimeState<NumericTensor>>;
fn same_state(actual: &State, expected: &State) {
    assert_eq!(actual.frames(), expected.frames());
    assert_eq!(
        actual.perception().global_blocks(),
        expected.perception().global_blocks()
    );
    for (a, b) in actual
        .perception()
        .blocks()
        .iter()
        .zip(expected.perception().blocks())
    {
        assert_eq!(a.valid_channel(), b.valid_channel());
        for (a, b) in [
            (a.channel(), b.channel()),
            (a.convolution(), b.convolution()),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.shape, b.shape);
                    assert_eq!(a.data, b.data);
                }
                (None, None) => {}
                _ => panic!("parent cache presence changed"),
            }
        }
    }
    match (actual.rnnt().predictor(), expected.rnnt().predictor()) {
        (Some(_), Some(_)) => same_rnnt(actual.rnnt(), expected.rnnt()),
        (None, None) => assert_eq!(actual.rnnt().frames(), 0),
        _ => panic!("parent predictor presence changed"),
    }
}

#[test]
fn voicechat_perception_rnnt_pipeline_preserves_paired_frontiers_and_replay() {
    let perception_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let rnnt_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_RNNT_REFERENCE_JSON).unwrap();
    let (coordinators, timeouts) = coordinator_pair();
    let perception_world = Arc::new(NumericPartitionWorld::default());
    let predictor_world = Arc::new(NumericPartitionWorld::default());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let perception_fixture = &perception_fixture;
                let rnnt_fixture = &rnnt_fixture;
                let perception_world = perception_world.clone();
                let predictor_world = predictor_world.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut perception =
                        PipelinePerception::new(perception_fixture, rank, perception_world.clone());
                    let mut predictor =
                        PipelinePredictor::new(rnnt_fixture, 1, rank, predictor_world.clone());
                    let mut projections = LayerwiseRnntProjection::new(
                        RnntProjectionArchitecture::new(dimensions(rnnt_fixture)).unwrap(),
                        Policy::bounded(weights(&rnnt_fixture["parameters"])),
                    );
                    let empty = State::from_perception(perception.initial_state()).unwrap();
                    let mut committed = empty.clone();
                    let mut cursor = RnntPipelineCursor::new(
                        1000,
                        DistributedCommitEpoch::new(100).unwrap(),
                        256,
                    );
                    let limits = PerceptionRnntLimits {
                        frame_capacity: 16,
                        rnnt: RnntLimits {
                            max_symbols_per_frame: 3,
                            hypothesis_capacity: 32,
                        },
                    };
                    let window = &perception_fixture["subsampling"]["steps"][0];
                    let input = tensor(&window["input"]);
                    let drop = window["drop"].as_u64().unwrap() as usize;
                    // Both source and cache ownership are checked before either component leases weights.
                    let foreign = PipelinePerception::new(
                        perception_fixture,
                        1 - rank,
                        Arc::new(NumericPartitionWorld::default()),
                    );
                    let foreign = State::from_perception(foreign.initial_state()).unwrap();
                    assert!(
                        perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [5, 2, 0],
                            DistributedCommitEpoch::new(1).unwrap(),
                            &mut cursor,
                            &input,
                            drop,
                            &foreign,
                            limits,
                            &context
                        )
                        .is_err()
                    );
                    assert_eq!(perception.calls, 0);
                    assert!(predictor.calls.is_empty());
                    assert!(projections.runtime().policy().acquired.is_empty());
                    let mut insufficient =
                        RnntPipelineCursor::new(1000, DistributedCommitEpoch::new(100).unwrap(), 4);
                    assert!(
                        perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [5, 2, 1],
                            DistributedCommitEpoch::new(1).unwrap(),
                            &mut insufficient,
                            &input,
                            drop,
                            &empty,
                            limits,
                            &context
                        )
                        .is_err()
                    );
                    assert_eq!(
                        perception.calls, 0,
                        "RNNT reservation failure precedes perception reads"
                    );
                    // Perception may finish physically while its paired candidate remains unpublished.
                    perception.reject_commit = rank == 1;
                    let rejected=perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [5, 2, 2],
                            DistributedCommitEpoch::new(1).unwrap(),
                            &mut cursor,
                            &input,
                            drop,
                            &empty,
                            limits,
                            &context
                        );
                    if rank==1 {
                        assert!(matches!(rejected,Err(eredu_architectures::voicechat::perception_rnnt::PerceptionRnntError::PerceptionCommit { .. })));
                    } else {assert!(rejected.is_err());}

                    perception.reject_commit = false;
                    assert_eq!(perception.calls, 1);
                    assert!(
                        predictor.calls.is_empty(),
                        "uncommitted perception cannot enter RNNT"
                    );
                    assert!(projections.runtime().policy().acquired.is_empty());
                    assert_eq!(cursor.next_step(), 1005);
                    same_state(&committed, &empty);
                    barrier.wait();
                    if rank == 0 {
                        // The first stage fails before its peer executes. Arm only
                        // rank zero so the inactive stage retains no fault for retry.
                        predictor_world.arm_fault_all(1, NumericPartitionFault::LocalExecution);
                    }
                    barrier.wait();
                    assert!(
                        perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [5, 2, 3],
                            DistributedCommitEpoch::new(2).unwrap(),
                            &mut cursor,
                            &input,
                            drop,
                            &empty,
                            limits,
                            &context
                        )
                        .is_err()
                    );
                    barrier.wait();
                    assert_eq!(perception.calls, 2);
                    assert_eq!(predictor.calls.len(),1,"the rejected phase actually entered recurrent execution");
                    assert_eq!(cursor.next_step(), 1010);
                    same_state(&committed, &empty);
                    let mut epoch = 3;
                    for (index, window) in perception_fixture["subsampling"]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let input = tensor(&window["input"]);
                        let drop = window["drop"].as_u64().unwrap() as usize;
                        let parent = committed.clone();
                        if index == 1 {
                            barrier.wait();
                            if rank == 0 {
                                perception_world.arm_fault_one_per_agreement_group(
                                    NumericPartitionFault::LocalExecution,
                                );
                            }
                            barrier.wait();
                            let recurrent_before = predictor.calls.len();
                            let projection_before = projections.runtime().policy().acquired.len();
                            assert!(
                                perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                                    &mut perception,
                                    &mut projections,
                                    &mut predictor,
                                    &mut coordinator,
                                    [5, 2, epoch],
                                    DistributedCommitEpoch::new(epoch).unwrap(),
                                    &mut cursor,
                                    &input,
                                    drop,
                                    &parent,
                                    limits,
                                    &context
                                )
                                .is_err()
                            );
                            epoch += 1;
                            barrier.wait();
                            assert_eq!(predictor.calls.len(), recurrent_before);
                            assert_eq!(
                                projections.runtime().policy().acquired.len(),
                                projection_before
                            );
                            same_state(&parent, &committed);
                        }
                        let calls_before = perception.calls;
                        let output =
                            perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                                &mut perception,
                                &mut projections,
                                &mut predictor,
                                &mut coordinator,
                                [5, 2, epoch],
                                DistributedCommitEpoch::new(epoch).unwrap(),
                                &mut cursor,
                                &input,
                                drop,
                                &parent,
                                limits,
                                &context,
                            )
                            .unwrap();
                        epoch += 1;
                        assert_eq!(
                            perception.calls,
                            calls_before + 1,
                            "one perception invocation serves both readouts"
                        );
                        let reference = &perception_fixture["perception"]["steps"][index];
                        assert_tensor(
                            &output.encoder,
                            &reference["encoder_output"],
                            perception_fixture,
                        );
                        assert_tensor(
                            &output.projected,
                            &reference["projected_output"],
                            perception_fixture,
                        );
                        assert_eq!(output.state.frames(), index as u64 + 1);
                        assert_eq!(output.state.rnnt().frames(), output.state.frames());
                        assert_eq!(
                            output.state.perception().blocks().len(),
                            usize::from(rank == 0)
                        );
                        if rank == 0 {
                            let calls=perception.calls;
                            assert!(eredu_architectures::voicechat::perception_rnnt_pipeline::PerceptionPipelineExecution::<NumericBackend>::validate_parent(&perception,output.state.perception(),2).is_err());
                            assert_eq!(perception.calls,calls,"batch mismatch validation cannot execute perception");
                            let cache = &output.state.perception().blocks()[0];
                            assert_tensor(
                                cache.channel().unwrap(),
                                &reference["channel_cache"],
                                perception_fixture,
                            );
                            let convolution = cache
                                .convolution()
                                .unwrap()
                                .transpose_axes(&[0, 2, 1], &context)
                                .unwrap();
                            assert_tensor(
                                &convolution,
                                &reference["time_cache"],
                                perception_fixture,
                            );
                        }
                        let expected = &rnnt_fixture["streaming_frames"][index];
                        assert_eq!(
                            output.emitted,
                            expected["emitted"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_u64().unwrap() as usize)
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(
                            output.first_blank,
                            expected["first_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            output.ended_blank,
                            expected["ended_blank"].as_bool().unwrap()
                        );
                        assert_eq!(
                            output.hit_symbol_limit,
                            expected["hit_symbol_limit"].as_bool().unwrap()
                        );
                        compare_rnnt(
                            output.state.rnnt().predictor().unwrap(),
                            rank,
                            expected,
                            rnnt_fixture,
                        );
                        same_state(&parent, &committed);
                        if index == 0 {
                            let replay =
                                perception_rnnt_pipeline::advance::<NumericBackend, _, _, _, _>(
                                    &mut perception,
                                    &mut projections,
                                    &mut predictor,
                                    &mut coordinator,
                                    [5, 2, epoch],
                                    DistributedCommitEpoch::new(epoch).unwrap(),
                                    &mut cursor,
                                    &input,
                                    drop,
                                    &parent,
                                    limits,
                                    &context,
                                )
                                .unwrap();
                            epoch += 1;
                            same_state(&replay.state, &output.state);
                            assert_eq!(replay.encoder.data, output.encoder.data);
                            assert_eq!(replay.projected.data, output.projected.data);
                            same_state(&parent, &committed);
                        }
                        committed = output.state;
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
