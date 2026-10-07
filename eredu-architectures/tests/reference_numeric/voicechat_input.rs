use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    conformer::ConformerDimensions,
    frontend::{FrontendOptions, PcmPrecision},
    input::{InputError, InputExecution, InputFrame, InputState},
    perception::PerceptionDimensions,
    perception_rnnt::{PerceptionRnnt, PerceptionRnntError, PerceptionRnntLimits},
    rnnt::{RnntDimensions, RnntLimits},
    subsampling::SubsamplingDimensions,
};
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};

fn state() -> InputState<NumericTensor> {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    InputState::new(
        &config.perception.preprocessor,
        FrontendOptions {
            pcm_precision: PcmPrecision::F32,
            buffer_frames: 3,
            queue_capacity_samples: 4000,
            pad_and_drop_preencoded: false,
        },
    )
    .unwrap()
}

fn model(context: &NumericContext) -> PerceptionRnnt<NumericBackend> {
    struct Bind;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut NumericTensor) {
            let id = metadata.id.as_str();
            // Deterministic nonzero perception and recurrent parameters. Joint
            // output scores explicitly select blank, independently of encoder values.
            let values = match id {
                "stt_model.rnnt_joint.joint_net.2.weight" => vec![0.0; tensor.data.len()],
                "stt_model.rnnt_joint.joint_net.2.bias" => vec![0.0, 1.0],
                _ => (0..tensor.data.len())
                    .map(|i| ((i % 11) as f32 - 4.0) * 0.015)
                    .collect(),
            };
            *tensor = NumericTensor::new(tensor.shape(), values);
        }
    }
    let mut model = PerceptionRnnt::new(
        PerceptionDimensions {
            subsampling: SubsamplingDimensions {
                features: 128,
                channels: 2,
                width: 4,
            },
            block: ConformerDimensions {
                width: 4,
                heads: 2,
                ff_hidden: 8,
                conv_kernel: 3,
                left_context: 3,
            },
            layers: 1,
            output_width: 6,
        },
        RnntDimensions {
            vocab_size: 1,
            predictor_width: 1,
            predictor_layers: 1,
            encoder_width: 4,
            joint_width: 1,
        },
        context,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind);
    model
}

fn limits(capacity: u64) -> PerceptionRnntLimits {
    PerceptionRnntLimits {
        frame_capacity: capacity,
        rnnt: RnntLimits {
            max_symbols_per_frame: 2,
            hypothesis_capacity: 4,
        },
    }
}

fn assert_frame(actual: &InputFrame<NumericTensor>, expected: &InputFrame<NumericTensor>) {
    assert_eq!(actual.frame_index, expected.frame_index);
    assert_eq!(actual.start_sample, expected.start_sample);
    assert_eq!(actual.valid_samples, expected.valid_samples);
    assert_eq!(actual.encoder.data, expected.encoder.data);
    assert_eq!(actual.projected.data, expected.projected.data);
    assert!(actual.encoder.data.iter().any(|v| v.abs() > 1.0e-5));
    assert!(actual.projected.data.iter().any(|v| v.abs() > 1.0e-5));
    assert!(actual.emitted.is_empty());
    assert!(actual.first_blank && actual.ended_blank && !actual.hit_symbol_limit);
    let actual = actual.state.neural();
    let expected = expected.state.neural();
    assert_eq!(actual.frames(), expected.frames());
    assert_eq!(actual.rnnt().frames(), expected.rnnt().frames());
    assert_eq!(actual.rnnt().hypothesis(), expected.rnnt().hypothesis());
    for (a, e) in actual
        .perception()
        .blocks()
        .iter()
        .zip(expected.perception().blocks())
    {
        assert_eq!(a.valid_channel(), e.valid_channel());
        assert_eq!(a.channel().unwrap().data, e.channel().unwrap().data);
        assert_eq!(a.convolution().unwrap().data, e.convolution().unwrap().data);
    }
    let actual = &actual.rnnt().predictor().unwrap().local_state().layers;
    let expected = &expected.rnnt().predictor().unwrap().local_state().layers;
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.hidden.data, e.hidden.data);
        assert_eq!(a.cell.data, e.cell.data);
    }
}

#[test]
fn voicechat_input_pcm_chunks_finish_and_capacity_retry_preserve_neural_frontier() {
    let context = NumericContext::default();
    let mut model = model(&context);
    let pcm: Vec<_> = (0..2877i64)
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.0)
        .collect();
    let mut whole = state();
    whole.submit(&pcm, 16_000).unwrap();
    whole
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let mut expected = Vec::new();
    while let Some(frame) = model.advance_pcm(&whole, limits(3), &context).unwrap() {
        whole = frame.state.clone();
        expected.push(frame);
    }
    assert_eq!(expected.len(), 3);
    assert!(whole.frontend().is_drained());
    for (index, frame) in expected.iter().enumerate() {
        assert_eq!(frame.frame_index, index as u64);
        assert_eq!(frame.start_sample, index as u64 * 1280);
        assert_eq!(frame.valid_samples, [1280, 1280, 317][index]);
        assert_eq!(frame.state.frontend().frames(), index as u64 + 1);
        assert_eq!(frame.state.neural().frames(), index as u64 + 1);
    }

    let mut split = state();
    let mut delivered = 0;
    for chunk in pcm.chunks(157) {
        split.submit(chunk, 16_000).unwrap();
        let queued = split.frontend().queued_samples();
        while let Some(frame) = model.advance_pcm(&split, limits(3), &context).unwrap() {
            assert_frame(&frame, &expected[delivered]);
            split = frame.state;
            delivered += 1;
        }
        if queued < 1280 {
            assert_eq!(split.frontend().queued_samples(), queued);
        }
    }
    assert_eq!(delivered, 2);
    assert_eq!(split.frontend().queued_samples(), 317);
    split
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let final_frame = model
        .advance_pcm(&split, limits(3), &context)
        .unwrap()
        .unwrap();
    assert_frame(&final_frame, &expected[2]);
    assert!(final_frame.state.frontend().is_drained());
    assert!(model
        .advance_pcm(&final_frame.state, limits(3), &context)
        .unwrap()
        .is_none());

    // A branch already contains one successful neural frame and queued PCM.
    // Capacity rejection must not consume that PCM or advance any parent cache.
    let snapshot = expected[0].state.clone();
    let queued = snapshot.frontend().queued_samples();
    let channel = snapshot.neural().perception().blocks()[0]
        .channel()
        .unwrap()
        .data
        .clone();
    assert!(matches!(
        model.advance_pcm(&snapshot, limits(1), &context),
        Err(InputError::PerceptionRnnt(PerceptionRnntError::Capacity {
            required: 2,
            capacity: 1
        }))
    ));
    assert_eq!(snapshot.frontend().queued_samples(), queued);
    assert_eq!(snapshot.frontend().frames(), 1);
    assert_eq!(snapshot.neural().frames(), 1);
    assert_eq!(
        snapshot.neural().perception().blocks()[0]
            .channel()
            .unwrap()
            .data,
        channel
    );
    let retry = model
        .advance_pcm(&snapshot, limits(3), &context)
        .unwrap()
        .unwrap();
    assert_frame(&retry, &expected[1]);
    assert_eq!(snapshot.frontend().queued_samples(), queued);
    assert_eq!(snapshot.neural().frames(), 1);
}
