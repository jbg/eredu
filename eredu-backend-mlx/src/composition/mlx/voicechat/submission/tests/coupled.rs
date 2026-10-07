//! Learned compact components cross native submission and the shared scheduler.
use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_architectures::{
    decoder, nemotron_h,
    voicechat::{
        backbone_execution::OrdinaryBackbone,
        channel_decision::ChannelDecisionPolicy,
        channels::{ChannelFusion, PromptPolicy},
        execution::{Components, ToolAudio},
        observation::OwnedHostOutput,
        rnnt::RnntLimits,
        speech::SpeechComponents,
        tts_feedback::{OutputPolicy, WaveformSilence},
        tts_transformer::TransformerConfig,
    },
};
use eredu_core::RealtimeDecisionDiagnostics;
use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
use eredu_core::scheduler::{
    RequestId, Scheduler, SchedulerLimits, TransitionOutput, WorkDescriptor,
};
use eredu_evaluation::voicechat::coupled as fixture;
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};
use eredu_runtime::execution_control::{
    ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork, IsolatedStateSnapshot, SnapshotBudget,
};
use eredu_runtime::{ResidentRuntime, TokenDomain};
use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

use super::backbone;

type Runtime =
    ResidentRuntime<nemotron_h::LayeredModel<MlxNeuralBackend>, MlxNeuralBackend, MlxHybridState>;
type Host = OwnedHostOutput<RealtimeDecisionDiagnostics>;
type CoupledComponents<'a> = Components<
    'a,
    MlxNeuralBackend,
    OrdinaryBackbone<'a, MlxNeuralBackend, Runtime>,
    eredu_architectures::voicechat::perception_rnnt::PerceptionRnnt<MlxNeuralBackend>,
    eredu_architectures::voicechat::tts_transformer::TtsTransformer<MlxNeuralBackend>,
    eredu_architectures::voicechat::tts_mog::MogHead<MlxNeuralBackend>,
    eredu_architectures::voicechat::tts_rvq::RvqRefinement<MlxNeuralBackend>,
    eredu_architectures::voicechat::tts_conditioning_layered::ResidentConditioning<
        MlxNeuralBackend,
    >,
    eredu_codec::ear::AudioDecoder<MlxNeuralBackend>,
    crate::composition::mlx::voicechat::MlxFunctionHead,
>;

#[derive(Clone, Copy)]
enum InputCase {
    Partial,
    Silence { virtual_tail: bool },
}
impl InputCase {
    fn second_frame_is_live(self) -> bool {
        matches!(self, Self::Silence { .. })
    }
}

#[derive(Clone, Copy)]
struct Work(usize);
impl WorkDescriptor for Work {
    type Error = Infallible;
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Infallible> {
        output.push(self.0 as u32);
        Ok(())
    }
}

struct Record {
    host: Host,
    retained: Vec<(Vec<i32>, Vec<f32>)>,
    random: Vec<u32>,
    history: Vec<u32>,
    hypothesis: Vec<usize>,
}
fn record(host: Host, state: &MlxVoiceChatGeneration, stream: &Stream) -> Record {
    if host.invocation_index == 1 {
        use eredu_core::cache::StateTensorRole;
        let fixed = state
            .model_state()
            .neural()
            .parts()
            .0
            .fixed_numeric_snapshot()
            .unwrap();
        for role in [
            StateTensorRole::Convolution { slot: 0 },
            StateTensorRole::Recurrent,
        ] {
            let (_, _, _, values) = fixed
                .iter()
                .find(|(layer, found, _, _)| *layer == 0 && *found == role)
                .expect("live Nemotron retains convolution and recurrent state");
            assert!(
                values.iter().any(|value| *value != 0.),
                "nonzero {role:?} fixture state"
            );
        }
        let predictor = state
            .model_state()
            .input()
            .neural()
            .rnnt()
            .predictor()
            .unwrap()
            .local_state();
        assert_eq!(predictor.layers.len(), 1);
        for tensor in [&predictor.layers[0].hidden, &predictor.layers[0].cell] {
            assert!(
                tensor
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .any(|value| *value != 0.),
                "RNNT emissions retain nonzero LSTM state"
            );
        }
    }
    let mut retained = Vec::new();
    let mut add = |array: &Array| {
        // Sliding-window cache views may keep gaps between guidance lanes.
        // Compare logical elements independently of snapshot compaction.
        retained.push((
            array.shape().to_vec(),
            array
                .contiguous(false, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
        ));
    };
    for array in state.model_state().neural().retained_arrays() {
        add(array);
    }
    state.model_state().visit_component_tensors(&mut |tensor| {
        // Feedback includes integer code tensors, so use the general floating
        // conversion solely for this test's value comparison after completion.
        let array = tensor.as_array();
        if array.dtype() == safemlx::Dtype::Float32 {
            add(array);
        } else {
            let value = array.as_dtype(safemlx::Dtype::Float32, stream).unwrap();
            add(&value);
        }
    });
    Record {
        host,
        retained,
        random: state
            .random_state()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<u32>()
            .unwrap(),
        history: state.samplers()[0].history().to_vec(),
        hypothesis: state
            .model_state()
            .input()
            .neural()
            .rnnt()
            .hypothesis()
            .to_vec(),
    }
}

fn submit(
    index: usize,
    input_case: InputCase,
    components: &mut CoupledComponents<'_>,
    codes: &MlxTensor,
    layout: &eredu_architectures::voicechat::tts_ingress::PromptLayout,
    branch: &mut MlxVoiceChatBranch,
    stream: &Stream,
) -> Result<MlxVoiceChatPrepublication, VoiceChatSubmissionError> {
    let output = OutputPolicy {
        force_silence_on_eos: true,
        silence_on_pad: &[false],
        waveform: WaveformSilence::PreserveGenerated,
    };
    let invocation = match index {
        0 => Invocation::Prefill {
            prompt: Some((
                &[5, 7],
                PromptPolicy {
                    repeat: 1,
                    max_tokens: 8,
                },
            )),
            codes,
            layout,
            conditioning: None,
        },
        1 | 2 if index == 1 || input_case.second_frame_is_live() => Invocation::Live {
            decisions: ChannelDecisionPolicy::Main {
                agent_feedback: Some(3),
                function_feedback: Some(9),
                spoken_text: Some(3),
            }
            .into(),
            output,
        },
        2 => Invocation::PerceptionOnly,
        3 => Invocation::Tool {
            audio: ToolAudio::Retained,
            decisions: ChannelDecisionPolicy::Tool {
                function_feedback: Some(9),
                spoken_text: Some(3),
            }
            .into(),
            output,
        },
        4 => Invocation::Reminder { spoken: 3, output },
        5 => Invocation::Drain,
        _ => unreachable!(),
    };
    components.preflight::<MlxSamplingBackend, _, _, _>(&invocation, branch, None)?;
    let owner = Arc::new(Owner {
        _neural: branch.model_state().neural.retention(),
        poisoned: Rc::new(Cell::new(false)),
    });
    let mut roots = branch_roots(branch);
    invocation.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
    // Learned modules stay owned by run() until every completion is observed.
    // This exercises the same private submission mechanism as the public
    // selected-provider adapter, without weakening released-artifact admission.
    submit_invocation(branch, owner, roots, stream, |branch| {
        components.execute::<MlxSamplingBackend, _, _, _>(&invocation, branch, None, stream)
    })
}

fn run(controlled: bool, input_case: InputCase, stream: &Stream) -> Vec<Record> {
    assert!(!controlled || matches!(input_case, InputCase::Partial));
    let data = fixture::fixture();
    let mut backbone: Runtime = ResidentRuntime::new(backbone::model(stream), stream).unwrap();
    backbone::bind_units(backbone.units_mut(), stream);
    // Controlled replay acquires the function head through a bounded host pool;
    // the scheduled baseline keeps the identical selected provider resident.
    let mut function = backbone::function(controlled, stream);
    let fusion = ChannelFusion::from_config(&fixture::config().channels).unwrap();
    let mut perception = fixture::perception::<MlxNeuralBackend>(stream);
    // Emit two symbols per frame to exercise retained LSTM state, independently
    // of the perception result and greedy joint-network tie behavior.
    struct Emit;
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Emit {
        fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut MlxTensor) {
            if metadata.id.as_str() == "stt_model.rnnt_joint.joint_net.2.bias" {
                *tensor = MlxTensor::from_array(Array::from_slice(&[1_f32, 0.], &[2]));
            }
        }
    }
    perception.visit_parameters_mut(&mut Emit);
    let mut tts = fixture::tts::<MlxNeuralBackend>(&data, stream);
    let mut codec = fixture::codec::<MlxNeuralBackend>(&data, stream);
    let feedback = fixture::controls(&data);
    let mapping = fixture::mapping(&data);
    let (codes, layout) = fixture::prompt::<MlxTensor>(&data, stream);
    let mut input = fixture::input_state::<MlxTensor>();
    let source_samples = match input_case {
        InputCase::Partial => 1437,
        InputCase::Silence { .. } => 1280,
    };
    let mut pcm: Vec<_> = (0..source_samples)
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.)
        .collect();
    if matches!(
        input_case,
        InputCase::Silence {
            virtual_tail: false
        }
    ) {
        pcm.resize(2560, 0.0);
    }
    for chunk in pcm.chunks(157) {
        input.submit(chunk, 16000).unwrap();
    }
    input
        .finish(eredu_media::audio::streaming::PcmFinishPolicy {
            final_frame: eredu_media::audio::streaming::FinalPcmFrame::ZeroPad,
            trailing_silence_frames: u64::from(matches!(
                input_case,
                InputCase::Silence { virtual_tail: true }
            )),
        })
        .unwrap();
    let tts_config = TransformerConfig::new(&fixture::tts_backbone_config(&data)).unwrap();
    let neural = MlxJointState::new(
        MlxHybridState::device(nemotron_h::state_layout(&backbone::args()).unwrap(), &[]).unwrap(),
        MlxKeyValueState::device(decoder::state_layout(&tts_config).unwrap()).unwrap(),
        stream,
        JointCopyLimits {
            max_branch_bytes: 1 << 24,
            max_total_copy_bytes: 1 << 28,
        },
    )
    .unwrap();
    let policy = SchedulePolicy::from_config(
        &fixture::config(),
        ScheduleLimits {
            max_audio_frames: 4,
            max_backbone_positions: 64,
            max_tts_positions: 64,
            max_codec_frames: 4,
        },
    )
    .unwrap();
    let sampling = AgentSamplingPolicy::streaming_defaults(16).unwrap();
    let mut state = MlxVoiceChatGeneration::new(
        State::new(input, neural, FeedbackTokens::new(12, 12).unwrap()),
        policy.generation_contract(),
        ScheduleState::new(policy),
        sampling,
        vec![AgentSampler::new(sampling)],
        Some(RandomState::with_seed(23).unwrap()),
    )
    .unwrap();
    let mut components = Components {
        input: &mut perception,
        backbone: OrdinaryBackbone::new(&mut backbone),
        function: &mut function,
        fusion: &fusion,
        speech: SpeechComponents::new(&mut tts, &feedback, &mut codec).unwrap(),
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
    let mut records = Vec::new();
    if controlled {
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 1 << 28,
            cumulative_copy_bytes: 1 << 32,
        });
        let mut saved = None;
        let mut forked = None;
        let mut snapshot_retained_bytes = 0;
        for index in 0..6 {
            let mut branch = state.branch().unwrap();
            if index == 1 {
                let mut pending = fixture::input_state::<MlxTensor>();
                pending.submit(&pcm[..157], 16000).unwrap();
                std::mem::swap(&mut pending, &mut branch.model_state_mut().input);
                assert!(matches!(
                    submit(
                        index,
                        input_case,
                        &mut components,
                        &codes,
                        &layout,
                        &mut branch,
                        stream
                    ),
                    Err(VoiceChatSubmissionError::Execution(
                        ExecutionError::NeedsInput
                    ))
                ));
                assert!(!branch.has_submission_completion());
                std::mem::swap(&mut pending, &mut branch.model_state_mut().input);
            }
            let frame = submit(
                index,
                input_case,
                &mut components,
                &codes,
                &layout,
                &mut branch,
                stream,
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(30);
            while !frame.is_complete().unwrap() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            assert_eq!(state.revision(), index as u64);
            state.commit_branch(branch).unwrap();
            records.push(record(frame.into_host_output().unwrap(), &state, stream));
            if index == 1 {
                // Nonzero perception/RNNT, both TTS lanes and codec carry coexist
                // with 157 unconsumed final PCM samples in the saved boundary.
                assert_eq!(state.model_state().input().frontend().queued_samples(), 157);
                saved = Some(IsolatedStateSnapshot::capture(&state, &budget).unwrap());
                snapshot_retained_bytes = budget.usage().retained_bytes;
                let reservation = budget
                    .reserve(
                        SnapshotResourceKind::Branch,
                        Some(state.estimate_isolated_copy().unwrap()),
                    )
                    .unwrap();
                forked = Some(state.fork_isolated(&reservation).unwrap());
                drop(reservation);
                assert_eq!(budget.usage().branches, 1);
            }
        }
        let used = budget.usage().cumulative_copy_bytes;
        saved.as_ref().unwrap().restore(&mut state).unwrap();
        assert_eq!(state.revision(), 7);
        assert!(budget.usage().cumulative_copy_bytes > used);
        assert_eq!(state.model_state().input().frontend().queued_samples(), 157);
        assert_eq!(
            state.model_state().input().neural().rnnt().hypothesis(),
            &[0, 0]
        );
        for index in 2..6 {
            let mut branch = state.branch().unwrap();
            let frame = submit(
                index,
                input_case,
                &mut components,
                &codes,
                &layout,
                &mut branch,
                stream,
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(30);
            while !frame.is_complete().unwrap() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            state.commit_branch(branch).unwrap();
            let replay = record(frame.into_host_output().unwrap(), &state, stream);
            assert_same_record(&records[index], &replay);
        }
        let mut child = forked.take().unwrap();
        assert_same_record(
            &records[1],
            &record(records[1].host.clone(), &child, stream),
        );
        assert_eq!(child.model_state().input().frontend().queued_samples(), 157);
        for index in 2..6 {
            let mut branch = child.branch().unwrap();
            let frame = submit(
                index,
                input_case,
                &mut components,
                &codes,
                &layout,
                &mut branch,
                stream,
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(30);
            while !frame.is_complete().unwrap() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            let retained = MlxVoiceChatGeneration::estimate_fork_branch(&branch).unwrap();
            child.commit_branch(branch).unwrap();
            assert!(retained >= child.estimate_isolated_copy().unwrap().retained_bytes);
            let continuation = record(frame.into_host_output().unwrap(), &child, stream);
            assert_same_record(&records[index], &continuation);
        }
        let consumed_copy_bytes = budget.usage().cumulative_copy_bytes;
        drop(child);
        ordinary_retirement::reclaim();
        assert_eq!(budget.usage().branches, 0);
        assert_eq!(budget.usage().retained_bytes, snapshot_retained_bytes);
        assert_eq!(budget.usage().cumulative_copy_bytes, consumed_copy_bytes);
        assert_eq!(state.samplers()[0].history(), &[3, 12]);
    } else {
        let mut scheduler: Scheduler<Work, MlxVoiceChatGeneration, MlxVoiceChatPrepublication> =
            Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
        let request = RequestId::new(3);
        scheduler.register(request, state).unwrap();
        let now = Instant::now(); // One explicit scheduler time for this replay.
        for index in 0..6 {
            scheduler.enqueue(request, Work(index)).unwrap();
            let mut result = scheduler
                .run_local_turn(now, |_, work, branch| {
                    submit(
                        work.0,
                        input_case,
                        &mut components,
                        &codes,
                        &layout,
                        branch,
                        stream,
                    )
                })
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(30);
            while result.committed.is_empty() {
                assert!(result.failed.is_empty());
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
                result = scheduler.poll_completions(now);
            }
            assert!(result.failed.is_empty());
            assert_eq!(result.committed.len(), 1);
            let (_, _, frame) = result.committed.pop().unwrap();
            records.push(record(
                frame.into_host_output().unwrap(),
                scheduler.request_state(request).unwrap(),
                stream,
            ));
        }
    }
    drop(components);
    let residency = function.residency_report().unwrap();
    if controlled {
        assert_eq!(
            residency
                .offload()
                .peak_resident_bytes()
                .get(eredu_core::residency::MemoryTier::Device),
            24 * 4 * 4
        );
    }
    records
}

fn assert_same_record(a: &Record, b: &Record) {
    assert_eq!(a.host.invocation_index, b.host.invocation_index);
    assert_eq!(a.host.reminder_token, b.host.reminder_token);
    assert_eq!(a.host.before, b.host.before);
    assert_eq!(a.host.after, b.host.after);
    assert_eq!(a.host.decisions, b.host.decisions);
    assert_eq!(a.host.generated_codes, b.host.generated_codes);
    assert_eq!(a.host.speech, b.host.speech);
    assert_eq!(a.host.transcription, b.host.transcription);
    assert_eq!(a.retained.len(), b.retained.len());
    for (index, ((a_shape, a_values), (b_shape, b_values))) in
        a.retained.iter().zip(&b.retained).enumerate()
    {
        assert_eq!(
            a_shape, b_shape,
            "tensor {index} shape at invocation {}",
            a.host.invocation_index
        );
        assert_eq!(a_values.len(), b_values.len(), "tensor {index} length");
        if let Some(first) = a_values.iter().zip(b_values).position(|(a, b)| a != b) {
            let max_diff = a_values
                .iter()
                .zip(b_values)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            panic!(
                "tensor {index} shape {a_shape:?} at invocation {}: first difference {first}: {} != {}; maximum absolute difference {max_diff}",
                a.host.invocation_index, a_values[first], b_values[first]
            );
        }
    }
    assert_eq!(a.random, b.random);
    assert_eq!(a.history, b.history);
    assert_eq!(a.hypothesis, b.hypothesis);
}

#[test]
fn coupled_native_virtual_continuation_matches_explicit_silence() {
    use eredu_media::audio::streaming::PcmFrameSource;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut continuation = run(false, InputCase::Silence { virtual_tail: true }, &stream);
    let explicit = run(
        false,
        InputCase::Silence {
            virtual_tail: false,
        },
        &stream,
    );
    assert_eq!(continuation.len(), explicit.len());
    let virtual_transcription = continuation[2].host.transcription.as_mut().unwrap();
    let explicit_transcription = explicit[2].host.transcription.as_ref().unwrap();
    assert_eq!(virtual_transcription.source, PcmFrameSource::Continuation);
    assert_eq!(virtual_transcription.valid_samples, 0);
    assert_eq!(virtual_transcription.start_sample, 1280);
    assert_eq!(explicit_transcription.source, PcmFrameSource::Input);
    assert_eq!(explicit_transcription.valid_samples, 1280);
    assert_eq!(explicit_transcription.start_sample, 1280);
    // Only source attribution differs. Both invocations run the learned live
    // path, including RNNT, channel decisions, both TTS lanes and the codec.
    virtual_transcription.source = PcmFrameSource::Input;
    virtual_transcription.valid_samples = 1280;
    for (virtual_frame, explicit_frame) in continuation.iter().zip(&explicit) {
        assert_same_record(virtual_frame, explicit_frame);
    }
    assert_ne!(continuation[1].random, continuation[2].random);
    assert!(continuation[2].host.speech.is_some());
    assert_eq!(continuation[2].host.after.audio_frames, 2);
    assert_eq!(continuation[2].host.after.codec_frames, 2);
    assert!(continuation[5].host.after.codec_finished);
    assert!(continuation[5].host.speech.as_ref().unwrap().is_tail);
}

#[test]
fn coupled_native_execution_matches_explicit_and_scheduler_publication() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let controlled = run(true, InputCase::Partial, &stream);
    let scheduled = run(false, InputCase::Partial, &stream);
    let expected = [
        (0, 4, 37, 0, 0),
        (1, 5, 38, 1, 1764),
        (2, 5, 38, 1, 1764),
        (2, 6, 39, 2, 3528),
        (2, 6, 40, 3, 5292),
        (2, 6, 40, 3, 5300),
    ];
    for (index, ((a, b), clocks)) in controlled.iter().zip(&scheduled).zip(expected).enumerate() {
        assert_eq!(a.host.invocation_index, index as u64);
        assert_eq!(a.host.reminder_token, (index == 4).then_some(3));
        let p = a.host.after;
        assert_eq!(
            (
                p.audio_frames,
                p.backbone_positions,
                p.tts_conditional_positions,
                p.codec_frames,
                p.output_samples
            ),
            clocks
        );
        assert_eq!(p.perception_frames, p.audio_frames);
        assert_eq!(p.rnnt_frames, p.audio_frames);
        assert_eq!(p.tts_unconditional_positions, p.tts_conditional_positions);
        assert_same_record(a, b);
        assert!(
            a.retained
                .iter()
                .flat_map(|(_, v)| v)
                .all(|v| v.is_finite())
        );
        if let Some(speech) = &a.host.speech {
            assert_eq!(speech.samples.len(), if index == 5 { 8 } else { 1764 });
            assert!(speech.samples.iter().all(|v| v.is_finite()));
            assert!(speech.samples.iter().any(|v| v.abs() > 1e-5));
        }
    }
    assert_eq!(
        controlled[1].host.transcription.as_ref().unwrap().emitted,
        [0, 0]
    );
    assert_eq!(
        controlled[2]
            .host
            .transcription
            .as_ref()
            .unwrap()
            .valid_samples,
        157
    );
    assert_eq!(
        controlled[2]
            .host
            .transcription
            .as_ref()
            .unwrap()
            .start_sample,
        1280
    );
    assert_eq!(controlled[5].history, [3, 12]);
    assert!(controlled[5].host.after.codec_finished);
    assert!(controlled[5].host.speech.as_ref().unwrap().is_tail);
    assert_ne!(controlled[0].random, controlled[1].random);
    assert_eq!(controlled[1].random, controlled[2].random);
    assert_ne!(controlled[2].random, controlled[3].random);
    assert_ne!(controlled[3].random, controlled[4].random);
    assert_eq!(controlled[4].random, controlled[5].random);
}
