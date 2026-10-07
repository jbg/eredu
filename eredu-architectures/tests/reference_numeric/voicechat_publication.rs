//! Actual coupled outputs cross the same scheduler completion/publication gate.
use super::*;
use eredu_architectures::voicechat::{
    execution::ExecutionOutput,
    observation::{OwnedHostOutput, VoiceChatHostObserver},
};
use eredu_core::scheduler::{RequestId, Scheduler, SchedulerLimits, WorkDescriptor};
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use eredu_runtime::{
    PrepublicationRealtimeFrame, RealtimeCompletionCreationError, RealtimeFrameHostObserver,
    submit_realtime_output,
};
use std::{cell::Cell, rc::Rc, time::Instant};

#[derive(Clone)]
struct TrackedNeural {
    inner: Neural,
    commits: Rc<Cell<usize>>,
    discards: Rc<Cell<usize>>,
}
impl NeuralStateParts for TrackedNeural {
    type Backbone = Backbone;
    type TtsState = [CausalCache];
    fn parts_mut(&mut self) -> (&mut Backbone, &mut [CausalCache]) {
        self.inner.parts_mut()
    }
}
impl SemanticStateTransaction for TrackedNeural {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), Infallible> {
        self.inner = branch.inner;
        self.commits.set(self.commits.get() + 1);
        Ok(())
    }
    fn discard_branch(branch: Self) -> Result<(), Infallible> {
        branch.discards.set(branch.discards.get() + 1);
        Ok(())
    }
}
#[derive(Clone)]
struct Gate {
    ready: Rc<Cell<bool>>,
    waits: Rc<Cell<usize>>,
    fail: bool,
}
impl Completion for Gate {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.ready.get())
    }
    fn wait(&self) -> Result<(), Self::Error> {
        assert!(self.ready.get(), "pending work must retain its resources");
        self.waits.set(self.waits.get() + 1);
        if self.fail {
            Err(std::io::Error::other("completion failed"))
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
    None,
    Pcm,
    Attribution,
    ReminderAttribution,
    Conversion,
    Completion,
    Submission,
    Cancel,
}
struct Observe {
    calls: Rc<Cell<usize>>,
    fault: Fault,
}
impl RealtimeFrameHostObserver<CompletedRealtimeFrame<NumericTensor, NumericTensor>> for Observe {
    type Output = CompletedRealtimeFrame<RealtimeHostChannelValues, ()>;
    type Error = std::io::Error;
    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<NumericTensor, NumericTensor>,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        if self.fault == Fault::Conversion {
            return Err(std::io::Error::other("host copy failed"));
        }
        if self.fault == Fault::Pcm {
            let values = frame
                .values()
                .iter()
                .map(|value| {
                    let mut payload = value.payload().clone();
                    if value.channel() == eredu_architectures::voicechat::schedule::OUTPUT_CHANNEL {
                        payload.data[0] = f32::NAN;
                    }
                    RealtimeFrameChannel::new(value.channel(), value.coordinate(), payload)
                })
                .collect();
            let damaged = CompletedRealtimeFrame::new(
                frame.channels().clone(),
                1,
                values,
                vec![],
                frame.decision_channels().to_vec(),
                vec![],
            )
            .unwrap();
            Ok(host(&damaged))
        } else {
            Ok(host(frame))
        }
    }
}
#[derive(Clone, Copy)]
struct Work(bool);
impl WorkDescriptor for Work {
    type Error = Infallible;
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Infallible> {
        output.push(u32::from(self.0));
        Ok(())
    }
}
type Published = PrepublicationRealtimeFrame<
    ExecutionOutput<NumericTensor>,
    Gate,
    VoiceChatHostObserver<Observe>,
>;
type Session = RealtimeGenerationState<
    State<NumericTensor, TrackedNeural>,
    ScheduleState,
    (),
    AgentSampler,
    i32,
    Gate,
>;

#[test]
fn voicechat_publication_gates_owned_speech_transcription_and_failed_candidates() {
    let now = Instant::now(); // Explicit scheduler time, identical throughout this replay.
    for fault in [
        Fault::None,
        Fault::Pcm,
        Fault::Attribution,
        Fault::ReminderAttribution,
        Fault::Conversion,
        Fault::Completion,
        Fault::Submission,
        Fault::Cancel,
    ] {
        let context = NumericContext::default();
        let fixture = fixture();
        let backbone_fixture: Value =
            serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
        let mut backbone = ResidentRuntime::new(
            super::super::voicechat_backbone_execution::model(&backbone_fixture, &context),
            &context,
        )
        .unwrap();
        let mut function =
            super::super::voicechat_backbone_execution::function(&backbone_fixture, &context);
        let fusion = ChannelFusion::from_config(&config().channels).unwrap();
        let mut perception = perception(&context);
        struct Emit;
        impl<'a> ParameterVisitorMut<'a, NumericTensor> for Emit {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
                if metadata.id.as_str() == "stt_model.rnnt_joint.joint_net.2.bias" {
                    *value = NumericTensor::new(vec![2], vec![1., 0.]);
                }
            }
        }
        // The independent constant joint boundary emits two nonblank symbols,
        // retaining the actual prediction state at its declared symbol bound.
        perception.visit_parameters_mut(&mut Emit);
        let mut tts = tts(&fixture, &context);
        let mut codec = codec(&fixture, &context);
        let controls = controls(&fixture);
        let mapping = mapping(&fixture);
        let (codes, layout) = prompt(&fixture, &context);
        let mut input = input_state();
        input.submit(&[0.125; 157], 16000).unwrap();
        input
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::ZeroPad,
                trailing_silence_frames: 0,
            })
            .unwrap();
        let commits = Rc::new(Cell::new(0));
        let discards = Rc::new(Cell::new(0));
        let neural = TrackedNeural {
            inner: Neural {
                backbone: super::super::voicechat_backbone_execution::state(),
                tts: caches(),
            },
            commits: commits.clone(),
            discards: discards.clone(),
        };
        let policy = SchedulePolicy::from_config(
            &config(),
            ScheduleLimits {
                max_audio_frames: 4,
                max_backbone_positions: 64,
                max_tts_positions: 64,
                max_codec_frames: 4,
            },
        )
        .unwrap();
        let state = Session::new(
            State::new(input, neural, FeedbackTokens::new(12, 12).unwrap()),
            policy.generation_contract(),
            ScheduleState::new(policy),
            (),
            vec![AgentSampler::new(
                AgentSamplingPolicy::streaming_defaults(8).unwrap(),
            )],
            Some(0),
        )
        .unwrap();
        let mut components = Components {
            input: &mut perception,
            backbone: eredu_architectures::voicechat::backbone_execution::OrdinaryBackbone::new(
                &mut backbone,
            ),
            function: &mut function,
            fusion: &fusion,
            speech: SpeechComponents::new(&mut tts, &controls, &mut codec).unwrap(),
            characters: &mapping,
            max_character_cells: 1024,
            input_limits: PerceptionRnntLimits {
                frame_capacity: 4,
                rnnt: RnntLimits {
                    max_symbols_per_frame: 2,
                    hypothesis_capacity: 8,
                },
            },
            vocabulary: TokenDomain::new(24),
        };
        let mut scheduler: Scheduler<Work, Session, Published> =
            Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
        let request = RequestId::new(3);
        scheduler.register(request, state).unwrap();
        let ready = Rc::new(Cell::new(true));
        let waits = Rc::new(Cell::new(0));
        let observed = Rc::new(Cell::new(0));
        for live in [false, true] {
            let active_fault = if live { fault } else { Fault::None };
            ready.set(!live || active_fault == Fault::Submission);
            scheduler.enqueue(request, Work(live)).unwrap();
            let submitted = scheduler.run_local_turn(now, |_, work, branch| {
                let invocation = if work.0 {
                    Invocation::Live {
                        decisions: ChannelDecisionPolicy::Main {
                            agent_feedback: Some(3),
                            function_feedback: Some(9),
                            spoken_text: Some(3),
                        }
                        .into(),
                        output: OutputPolicy {
                            force_silence_on_eos: true,
                            silence_on_pad: &[false],
                            waveform: WaveformSilence::PreserveGenerated,
                        },
                    }
                } else {
                    Invocation::Prefill {
                        prompt: None,
                        codes: &codes,
                        layout: &layout,
                        conditioning: None,
                    }
                };
                // Numeric mechanisms finish synchronously, including on error.
                // Native adapters must enter their recovery scope before execute.
                let mut result = components
                    .execute::<NumericBackend, _, _, _>(&invocation, branch, None, &context)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                if active_fault == Fault::Attribution {
                    result.input.as_mut().unwrap().start_sample += 1;
                }
                if active_fault == Fault::ReminderAttribution {
                    result.reminder_token = Some(3);
                }
                let gate = Gate {
                    ready: ready.clone(),
                    waits: waits.clone(),
                    fail: active_fault == Fault::Completion,
                };
                let submitted = submit_realtime_output(branch, result, |state, output| {
                    assert_eq!(
                        state.speech.as_ref().unwrap().model_position(),
                        if work.0 { 38 } else { 37 }
                    );
                    if work.0 {
                        assert!(
                            output.input.is_some()
                                && output.backbone.is_some()
                                && output.speech.is_some()
                        );
                    }
                    if active_fault == Fault::Submission {
                        return Err(RealtimeCompletionCreationError::after_submission(
                            std::io::Error::other("event creation failed"),
                            gate,
                        ));
                    }
                    Ok((gate, 1))
                })
                .map_err(|e| std::io::Error::other(e.to_string()))?;
                Ok::<_, std::io::Error>(PrepublicationRealtimeFrame::new(
                    submitted,
                    VoiceChatHostObserver::new(Observe {
                        calls: observed.clone(),
                        fault: active_fault,
                    }),
                ))
            });
            if active_fault == Fault::Submission {
                assert!(matches!(
                    submitted,
                    Err(eredu_core::scheduler::SchedulerError::Submission(_))
                ));
                assert_eq!(commits.get(), 1);
                assert_eq!(observed.get(), 1);
                assert_eq!(discards.get(), 1);
                assert!(
                    waits.get() >= 3,
                    "quarantine completion was waited before discard"
                );
                continue;
            }
            let submitted = submitted.unwrap();
            if !live {
                assert_eq!(submitted.committed.len(), 1);
                let (_, _, frame) = submitted.committed.into_iter().next().unwrap();
                let output = frame.into_host_output().unwrap();
                assert!(output.speech.is_none() && output.transcription.is_none());
                assert_eq!(commits.get(), 1);
                continue;
            }
            assert!(submitted.committed.is_empty());
            assert_eq!(
                commits.get(),
                1,
                "{fault:?}: candidate must remain unpublished"
            );
            assert_eq!(
                observed.get(),
                1,
                "{fault:?}: no conversion before successful completion"
            );
            let canonical = scheduler.request_state(request).unwrap();
            assert_eq!(
                canonical.model_state().input().frontend().queued_samples(),
                157
            );
            assert_eq!(canonical.random_state(), Some(&0));
            if active_fault == Fault::Cancel {
                scheduler.cancel(request).unwrap();
            }
            ready.set(true);
            let completed = scheduler.poll_completions(now);
            if active_fault == Fault::None {
                assert!(completed.failed.is_empty());
                assert_eq!(completed.committed.len(), 1);
                assert_eq!(commits.get(), 2);
                let (_, _, frame) = completed.committed.into_iter().next().unwrap();
                let output: OwnedHostOutput<()> = frame.into_host_output().unwrap();
                // Native result and prepublication wrapper are gone: the facade
                // receives independently owned PCM, codes and RNNT attribution.
                let speech = output.speech.unwrap();
                assert_eq!(
                    (
                        speech.start_sample,
                        speech.sample_rate,
                        speech.samples.len()
                    ),
                    (0, 22050, 1764)
                );
                assert!(speech.samples.iter().all(|v| v.is_finite()));
                assert!(speech.samples.iter().any(|v| v.abs() > 1e-5));
                let text = output.transcription.unwrap();
                assert_eq!(
                    (
                        text.frame_index,
                        text.start_sample,
                        text.sample_rate,
                        text.valid_samples
                    ),
                    (0, 0, 16000, 157)
                );
                assert_eq!(text.emitted, [0, 0]);
                assert!(!text.first_blank && !text.ended_blank && text.hit_symbol_limit);
                assert_eq!(output.generated_codes.unwrap().len(), 31);
                assert_eq!(
                    scheduler.request_state(request).unwrap().random_state(),
                    Some(&10)
                );
            } else {
                assert!(completed.committed.is_empty());
                assert_eq!(commits.get(), 1);
                assert_eq!(discards.get(), 1);
                if active_fault != Fault::Cancel {
                    assert_eq!(completed.failed.len(), 1);
                }
            }
        }
    }
}
