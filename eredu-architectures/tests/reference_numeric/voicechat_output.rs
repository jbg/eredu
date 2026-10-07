//! Output attribution uses declared invocation clocks, not live-audio frame IDs.
use super::*;
use eredu_architectures::voicechat::{
    channel_decision::{ChannelDecision, FeedbackTokens},
    config::VoiceChatConfig,
    output::{SpeechPayload, assemble},
    schedule::{
        AGENT_CHANNEL, AudioSource, OUTPUT_CHANNEL, Progress, ScheduleLimits, SchedulePolicy,
        ScheduleState, Step,
    },
};
use eredu_core::{RealtimeChannelPayload, RealtimeSlotCoordinate};
use eredu_runtime::{CompletedRealtimeFrame, RealtimeFrameChannel, RealtimeHostChannelValues};

fn host(
    frame: &CompletedRealtimeFrame<NumericTensor, NumericTensor>,
) -> CompletedRealtimeFrame<RealtimeHostChannelValues, ()> {
    CompletedRealtimeFrame::new(
        frame.channels().clone(),
        frame.batch(),
        frame
            .values()
            .iter()
            .map(|v| {
                let payload = match &frame.channels().channel(v.channel()).unwrap().payload {
                    RealtimeChannelPayload::Categorical { width, .. } => {
                        assert_eq!(v.payload().shape(), [1, *width as i32]);
                        RealtimeHostChannelValues::Categorical(
                            v.payload().data.iter().map(|x| *x as i32).collect(),
                        )
                    }
                    RealtimeChannelPayload::Dense { shape } => {
                        assert_eq!(v.payload().shape(), [1, shape[0] as i32]);
                        RealtimeHostChannelValues::Dense(
                            v.payload().data.iter().map(|x| *x as f32).collect(),
                        )
                    }
                };
                RealtimeFrameChannel::new(v.channel(), v.coordinate(), payload)
            })
            .collect(),
        vec![],
        frame.decision_channels().to_vec(),
        vec![],
    )
    .unwrap()
}

#[test]
fn voicechat_output_keeps_tool_audio_clocks_and_validates_before_publication() {
    let context = NumericContext::default();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let policy = SchedulePolicy::from_config(
        &config,
        ScheduleLimits {
            max_audio_frames: 8,
            max_backbone_positions: 128,
            max_tts_positions: 128,
            max_codec_frames: 8,
        },
    )
    .unwrap();
    let mut state = ScheduleState::new(policy);
    let decision = ChannelDecision {
        raw_agent: NumericTensor::from_i32_slice(&[13], &[1, 1, 1], &context).unwrap(),
        raw_function: NumericTensor::from_i32_slice(&[14], &[1], &context).unwrap(),
        raw_agent_id: 13,
        raw_function_id: 14,
        feedback: FeedbackTokens::new(12, 9).unwrap(),
        spoken_text: 3,
    };
    let generated =
        NumericTensor::from_i32_slice(&(0..31).collect::<Vec<_>>(), &[1, 1, 31], &context).unwrap();
    let pcm = NumericTensor::from_f32_slice(&vec![0.125; 1764], &[1, 1, 1764], &context).unwrap();
    let tail = NumericTensor::from_f32_slice(&[0.25; 8], &[1, 1, 8], &context).unwrap();
    let mut progress = Progress::default();
    for (index, step) in [
        Step::Prefill { backbone_tokens: 4 },
        Step::LiveAudio,
        Step::PerceptionOnly,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Step::Reminder,
        Step::Drain,
    ]
    .into_iter()
    .enumerate()
    {
        let plan = state.plan(step).unwrap();
        let (has_heads, has_speech) = match step {
            Step::Prefill { .. } => {
                progress.backbone_positions = 4;
                progress.tts_conditional_positions = 37;
                progress.tts_unconditional_positions = 37;
                (false, false)
            }
            Step::LiveAudio => {
                progress.audio_frames += 1;
                progress.perception_frames += 1;
                progress.rnnt_frames += 1;
                progress.backbone_positions += 1;
                (true, true)
            }
            Step::PerceptionOnly => {
                progress.audio_frames += 1;
                progress.perception_frames += 1;
                progress.rnnt_frames += 1;
                (false, false)
            }
            Step::ReplayAudio => {
                progress.backbone_positions += 1;
                (false, false)
            }
            Step::Tool { .. } => {
                progress.backbone_positions += 1;
                (true, true)
            }
            Step::Reminder => (false, true),
            Step::Drain => {
                progress.codec_finished = true;
                progress.output_samples += 8;
                (false, false)
            }
        };
        if has_speech {
            progress.tts_conditional_positions += 1;
            progress.tts_unconditional_positions += 1;
            progress.codec_frames += 1;
            progress.output_samples += 1764;
        }
        let payload = || {
            if has_speech {
                SpeechPayload::Frame {
                    generated: &generated,
                    pcm: &pcm,
                }
            } else if matches!(step, Step::Drain) {
                SpeechPayload::Tail(&tail)
            } else {
                SpeechPayload::None
            }
        };
        let heads = has_heads.then_some(&decision);
        if has_speech {
            assert!(
                assemble(
                    plan.clone(),
                    progress,
                    heads,
                    SpeechPayload::<NumericTensor>::None,
                    &context
                )
                .is_err()
            );
            let mut wrong = progress;
            wrong.audio_frames += 1;
            assert!(assemble(plan.clone(), wrong, heads, payload(), &context).is_err());
        }
        let output = assemble(plan, progress, heads, payload(), &context).unwrap();
        let observed = host(&output.frame);
        let projected = output.projection.project(&observed).unwrap();
        assert_eq!(projected.invocation_index, index as u64);
        assert_eq!(projected.after, progress);
        assert_eq!(
            projected
                .decisions
                .map(|d| (d.raw_agent, d.feedback.agent(), d.spoken_text)),
            has_heads.then_some((13, 12, 3))
        );
        if has_speech {
            let chunk = projected.speech.unwrap();
            assert_eq!(chunk.sample_rate, 22050);
            assert_eq!(chunk.start_sample, (progress.codec_frames - 1) * 1764);
            assert_eq!(chunk.delay_samples, 8);
            assert!(!chunk.is_tail);
            assert_eq!(chunk.samples, [0.125; 1764]);
            assert_eq!(
                projected.generated_codes.unwrap(),
                (0..31).collect::<Vec<_>>()
            );
        }
        if matches!(step, Step::Tool { .. } | Step::Reminder | Step::Drain) {
            assert_eq!(projected.before.audio_frames, 2);
            assert_eq!(projected.after.audio_frames, 2);
        }
        if matches!(step, Step::Drain) {
            let chunk = projected.speech.unwrap();
            assert_eq!(chunk.start_sample, 5292);
            assert_eq!(chunk.samples, [0.25; 8]);
            assert!(chunk.is_tail);
            assert_eq!(
                projected.before.backbone_positions,
                projected.after.backbone_positions
            );
        }
        if has_heads {
            // These independently corrupted observer results must fail before the
            // owner can publish the candidate schedule or semantic output.
            for corruption in 0..4 {
                let values = observed
                    .values()
                    .iter()
                    .map(|v| {
                        let mut payload = v.payload().clone();
                        let mut coordinate = v.coordinate();
                        if v.channel() == AGENT_CHANNEL {
                            if corruption == 0 {
                                payload = RealtimeHostChannelValues::Categorical(vec![15]);
                            }
                            if corruption == 1 {
                                coordinate =
                                    Some(RealtimeSlotCoordinate::new(index + 1, AGENT_CHANNEL));
                            }
                        }
                        if v.channel() == OUTPUT_CHANNEL {
                            if let RealtimeHostChannelValues::Dense(samples) = &mut payload {
                                if corruption == 2 {
                                    samples[0] = f32::NAN;
                                }
                                if corruption == 3 {
                                    samples.pop();
                                }
                            }
                        }
                        RealtimeFrameChannel::new(v.channel(), coordinate, payload)
                    })
                    .collect();
                let invalid = CompletedRealtimeFrame::<_, ()>::new(
                    observed.channels().clone(),
                    1,
                    values,
                    vec![],
                    observed.decision_channels().to_vec(),
                    vec![],
                )
                .unwrap();
                assert!(output.projection.project(&invalid).is_err());
            }
        }
        state = output.schedule;
    }
    // Draining an initialized but unused codec does not invent eight samples.
    let cold = ScheduleState::new(state.policy().clone());
    let prefill = cold.plan(Step::Prefill { backbone_tokens: 0 }).unwrap();
    let empty = Progress {
        tts_conditional_positions: 37,
        tts_unconditional_positions: 37,
        ..Progress::default()
    };
    let prefilled =
        assemble::<NumericTensor>(prefill, empty, None, SpeechPayload::None, &context).unwrap();
    let drain = prefilled.schedule.plan(Step::Drain).unwrap();
    let finished = Progress {
        codec_finished: true,
        ..empty
    };
    assert!(
        assemble::<NumericTensor>(
            drain.clone(),
            finished,
            None,
            SpeechPayload::Tail(&tail),
            &context
        )
        .is_err()
    );
    let output =
        assemble::<NumericTensor>(drain, finished, None, SpeechPayload::None, &context).unwrap();
    assert!(
        output
            .projection
            .project(&host(&output.frame))
            .unwrap()
            .speech
            .is_none()
    );
}
