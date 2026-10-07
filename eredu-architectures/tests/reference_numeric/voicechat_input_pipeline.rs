//! Host PCM control and actual paired pipelines share one input transition.
use super::voicechat_mog_residency::weights;
use super::voicechat_perception_distributed::PipelinePerception;
use super::voicechat_perception_residency::{
    assert_tensor, dimensions as perception_dimensions, parameters,
};
use super::voicechat_rnnt::dimensions as rnnt_dimensions;
use super::voicechat_rnnt_coordination::coordinator_pair;
use super::voicechat_rnnt_predictor::PipelinePredictor;
use super::voicechat_rnnt_residency::Policy;
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    frontend::{FrontendOptions, PcmPrecision, VoiceChatFrontend},
    input::{InputExecution, InputFrame, InputState},
    input_pipeline::PipelineInput,
    perception_layered::PerceptionRuntimeState,
    perception_rnnt::{PerceptionRnnt, PerceptionRnntLimits, PerceptionRnntState},
    rnnt::RnntLimits,
    rnnt_pipeline::RnntPipelineCursor,
    rnnt_projection_layered::{LayerwiseRnntProjection, RnntProjectionArchitecture},
};
use eredu_core::DistributedCommitEpoch;
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use serde_json::Value;
type State = InputState<NumericTensor, PerceptionRuntimeState<NumericTensor>>;
fn frontend() -> VoiceChatFrontend {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    VoiceChatFrontend::new(
        &config.perception.preprocessor,
        FrontendOptions {
            pcm_precision: PcmPrecision::F32,
            buffer_frames: 3,
            queue_capacity_samples: 5000,
            pad_and_drop_preencoded: false,
        },
    )
    .unwrap()
}
fn limits() -> PerceptionRnntLimits {
    PerceptionRnntLimits {
        frame_capacity: 8,
        rnnt: RnntLimits {
            max_symbols_per_frame: 3,
            hypothesis_capacity: 32,
        },
    }
}
fn finish() -> PcmFinishPolicy {
    PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: 1,
    }
}
fn compare(
    actual: &InputFrame<NumericTensor, PerceptionRuntimeState<NumericTensor>>,
    expected: &InputFrame<NumericTensor>,
    rank: usize,
    fixture: &Value,
) {
    let value =
        |tensor: &NumericTensor| serde_json::json!({"shape":tensor.shape,"values":tensor.data});
    assert_eq!(actual.frame_index, expected.frame_index);
    assert_eq!(actual.start_sample, expected.start_sample);
    assert_eq!(actual.valid_samples, expected.valid_samples);
    assert_eq!(actual.source, expected.source);
    assert_eq!(actual.emitted, expected.emitted);
    assert_eq!(
        (
            actual.first_blank,
            actual.ended_blank,
            actual.hit_symbol_limit
        ),
        (
            expected.first_blank,
            expected.ended_blank,
            expected.hit_symbol_limit
        )
    );
    assert_tensor(&actual.encoder, &value(&expected.encoder), fixture);
    assert_tensor(&actual.projected, &value(&expected.projected), fixture);
    assert_eq!(
        actual.state.frontend().frames(),
        expected.state.frontend().frames()
    );
    assert_eq!(
        actual.state.neural().frames(),
        expected.state.neural().frames()
    );
    assert_eq!(
        actual.state.neural().rnnt().hypothesis(),
        expected.state.neural().rnnt().hypothesis()
    );
    if rank == 0 {
        let a = &actual.state.neural().perception().blocks()[0];
        let b = &expected.state.neural().perception().blocks()[0];
        assert_tensor(a.channel().unwrap(), &value(b.channel().unwrap()), fixture);
        assert_tensor(
            a.convolution().unwrap(),
            &value(b.convolution().unwrap()),
            fixture,
        );
    } else {
        assert!(actual.state.neural().perception().blocks().is_empty());
    }
    let actual = actual.state.neural().rnnt().predictor().unwrap();
    let expected = expected.state.neural().rnnt().predictor().unwrap();
    assert_eq!(actual.global_layers(), rank..rank + 1);
    assert_tensor(actual.output(), &value(expected.output()), fixture);
    assert_tensor(
        &actual.local_state().layers[0].hidden,
        &value(&expected.local_state().layers[rank].hidden),
        fixture,
    );
    assert_tensor(
        &actual.local_state().layers[0].cell,
        &value(&expected.local_state().layers[rank].cell),
        fixture,
    );
}
#[test]
fn voicechat_input_pipeline_chunks_empty_drain_and_replay_match_ordinary() {
    let perception_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TOOL_SILENCE_REFERENCE_JSON).unwrap();
    let rnnt_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_RNNT_REFERENCE_JSON).unwrap();
    let pcm: Vec<_> = (0..2877i64)
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.)
        .collect();
    let (coordinators, timeouts) = coordinator_pair();
    let perception_world = Arc::new(NumericPartitionWorld::default());
    let predictor_world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let perception_fixture = &perception_fixture;
                let rnnt_fixture = &rnnt_fixture;
                let pcm = &pcm;
                let perception_world = perception_world.clone();
                let predictor_world = predictor_world.clone();
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut ordinary = PerceptionRnnt::<NumericBackend>::new(
                        perception_dimensions(perception_fixture),
                        rnnt_dimensions(rnnt_fixture),
                        &context,
                    )
                    .unwrap();
                    let mut all = parameters(perception_fixture);
                    all.extend(weights(&rnnt_fixture["parameters"]));
                    replace_numeric_parameters(&mut ordinary, &all);
                    let mut whole =
                        InputState::from_parts(frontend(), PerceptionRnntState::default()).unwrap();
                    whole.submit(pcm, 16_000).unwrap();
                    whole.finish(finish()).unwrap();
                    let mut expected = Vec::new();
                    while let Some(frame) =
                        ordinary.advance_pcm(&whole, limits(), &context).unwrap()
                    {
                        whole = frame.state.clone();
                        expected.push(frame);
                    }
                    assert_eq!(expected.len(), 4);
                    let mut perception =
                        PipelinePerception::new(perception_fixture, rank, perception_world);
                    let mut predictor =
                        PipelinePredictor::new(rnnt_fixture, 1, rank, predictor_world);
                    let mut projections = LayerwiseRnntProjection::new(
                        RnntProjectionArchitecture::new(rnnt_dimensions(rnnt_fixture)).unwrap(),
                        Policy::bounded(weights(&rnnt_fixture["parameters"])),
                    );
                    let initial = || {
                        State::from_parts(
                            frontend(),
                            PerceptionRnntState::from_perception(perception.initial_state())
                                .unwrap(),
                        )
                        .unwrap()
                    };
                    let mut split = initial();
                    let mut mismatch = initial();
                    mismatch
                        .submit(&pcm[..if rank == 0 { 1280 } else { 1279 }], 16_000)
                        .unwrap();
                    let mut different = initial();
                    let mut differing_pcm = pcm[..1280].to_vec();
                    if rank == 1 {
                        differing_pcm[100] += 0.25;
                    }
                    different.submit(&differing_pcm, 16_000).unwrap();
                    let mut cursor = RnntPipelineCursor::new(
                        500,
                        DistributedCommitEpoch::new(100).unwrap(),
                        128,
                    );
                    {
                        let mut execution = PipelineInput::<NumericBackend, _, _, _, _>::new(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [8, 3, 0],
                            DistributedCommitEpoch::new(1).unwrap(),
                            &mut cursor,
                        );
                        assert!(execution
                            .advance_pcm(&split, limits(), &context)
                            .unwrap()
                            .is_none());
                        assert!(!execution.is_consumed());
                        assert!(execution
                            .advance_pcm(&mismatch, limits(), &context)
                            .is_err());
                        assert!(
                            !execution.is_consumed(),
                            "availability disagreement cannot consume neural authority"
                        );
                    }
                    assert_eq!(cursor.next_step(), 500);
                    assert_eq!(perception.calls, 0);
                    assert!(predictor.calls.is_empty());
                    {
                        let mut execution = PipelineInput::<NumericBackend, _, _, _, _>::new(
                            &mut perception,
                            &mut projections,
                            &mut predictor,
                            &mut coordinator,
                            [8, 3, 1],
                            DistributedCommitEpoch::new(1).unwrap(),
                            &mut cursor,
                        );
                        assert!(execution
                            .advance_pcm(&different, limits(), &context)
                            .is_err());
                        assert!(execution.is_consumed());
                    }
                    assert_eq!(cursor.next_step(), 500);
                    assert_eq!(perception.calls, 0);
                    assert!(
                        projections.runtime().policy().acquired.is_empty(),
                        "differing host features cannot enter neural execution"
                    );
                    let mut epoch = 2;
                    let mut delivered = 0;
                    let mut failed = false;
                    let mut replayed = false;
                    for chunk in pcm.chunks(157) {
                        split.submit(chunk, 16_000).unwrap();
                        loop {
                            let ready = split.frontend().has_ready_frame();
                            let parent = split.clone();
                            let queued = parent.frontend().queued_samples();
                            let before = cursor.next_step();
                            if ready && !failed {
                                perception.reject_commit = rank == 1;
                            }
                            let (result, consumed) = {
                                let mut execution =
                                    PipelineInput::<NumericBackend, _, _, _, _>::new(
                                        &mut perception,
                                        &mut projections,
                                        &mut predictor,
                                        &mut coordinator,
                                        [8, 3, epoch],
                                        DistributedCommitEpoch::new(epoch).unwrap(),
                                        &mut cursor,
                                    );
                                let result = execution.advance_pcm(&parent, limits(), &context);
                                (result, execution.is_consumed())
                            };
                            if !ready {
                                assert!(result.unwrap().is_none());
                                assert!(!consumed);
                                assert_eq!(cursor.next_step(), before);
                                break;
                            }
                            assert!(consumed);
                            epoch += 1;
                            assert_eq!(parent.frontend().queued_samples(), queued);
                            assert_eq!(parent.frontend().frames(), delivered as u64);
                            assert_eq!(parent.neural().frames(), delivered as u64);
                            if !failed {
                                assert!(result.is_err());
                                failed = true;
                                perception.reject_commit = false;
                                assert!(cursor.next_step() > before);
                                continue;
                            }
                            let frame = result.unwrap().unwrap();
                            compare(&frame, &expected[delivered], rank, perception_fixture);
                            if !replayed {
                                let before = cursor.next_step();
                                let replay = {
                                    let mut execution =
                                        PipelineInput::<NumericBackend, _, _, _, _>::new(
                                            &mut perception,
                                            &mut projections,
                                            &mut predictor,
                                            &mut coordinator,
                                            [8, 3, epoch],
                                            DistributedCommitEpoch::new(epoch).unwrap(),
                                            &mut cursor,
                                        );
                                    execution
                                        .advance_pcm(&parent, limits(), &context)
                                        .unwrap()
                                        .unwrap()
                                };
                                epoch += 1;
                                compare(&replay, &expected[delivered], rank, perception_fixture);
                                assert!(
                                    cursor.next_step() > before,
                                    "forking host/neural state cannot refund the cursor"
                                );
                                assert_eq!(parent.frontend().queued_samples(), queued);
                                replayed = true;
                            }
                            split = frame.state;
                            delivered += 1;
                        }
                    }
                    assert_eq!(delivered, 2);
                    assert_eq!(split.frontend().queued_samples(), 317);
                    split.finish(finish()).unwrap();
                    loop {
                        let before = cursor.next_step();
                        let (result, consumed) = {
                            let mut execution = PipelineInput::<NumericBackend, _, _, _, _>::new(
                                &mut perception,
                                &mut projections,
                                &mut predictor,
                                &mut coordinator,
                                [8, 3, epoch],
                                DistributedCommitEpoch::new(epoch).unwrap(),
                                &mut cursor,
                            );
                            (
                                execution.advance_pcm(&split, limits(), &context).unwrap(),
                                execution.is_consumed(),
                            )
                        };
                        let Some(frame) = result else {
                            assert!(!consumed);
                            assert_eq!(cursor.next_step(), before);
                            break;
                        };
                        epoch += 1;
                        compare(&frame, &expected[delivered], rank, perception_fixture);
                        split = frame.state;
                        delivered += 1;
                    }
                    assert_eq!(delivered, 4);
                    assert!(split.frontend().is_drained());
                    assert_eq!(split.neural().frames(), 4);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
