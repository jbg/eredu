mod support {
    #[allow(dead_code)]
    pub mod numeric_backend;
}

use eredu_codec::streaming::{
    Conv1dSpec, ConvTranspose1dSpec, StreamPhase, StreamableConv1d, StreamableConvTranspose1d,
    TransposeTail,
};
use eredu_nn::{Index, PadMode, Tensor};
use support::numeric_backend::{activate, Context, FailAt, NumericTensor};

fn chunk(tensor: &NumericTensor, start: usize, end: usize, context: &Context) -> NumericTensor {
    tensor
        .index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(start as i32, end as i32),
            ],
            context,
        )
        .unwrap()
}

fn values(chunks: &[NumericTensor], context: &Context) -> Vec<f32> {
    NumericTensor::concatenate(chunks, 2, context)
        .unwrap()
        .to_f32_vec(context)
        .unwrap()
}

fn convolution(context: &Context, mode: PadMode) -> StreamableConv1d<NumericTensor> {
    let mut layer = StreamableConv1d::unloaded(
        Conv1dSpec {
            input_channels: 2,
            output_channels: 2,
            kernel_size: 3,
            stride: 2,
            dilation: 2,
            groups: 2,
            bias: true,
            pad_mode: mode,
        },
        context,
    )
    .unwrap();
    layer.weight.replace(NumericTensor::f32(
        vec![2., -1., 3., -2., 0.5, 1.],
        &[2, 3, 1],
    ));
    layer
        .bias
        .as_mut()
        .unwrap()
        .replace(NumericTensor::f32(vec![0.25, -0.5], &[2]));
    layer
}

// Independent scalar equation: grouped cross-correlation over a globally
// padded finite input. It does not call an operator or construct streaming carry.
fn expected_convolution(input: &[f32], length: usize, mode: PadMode) -> Vec<f32> {
    let weights = [[2., -1., 3.], [-2., 0.5, 1.]];
    let biases = [0.25, -0.5];
    let mut output = Vec::new();
    for channel in 0..2 {
        for frame in 0..length.div_ceil(2) {
            let mut sum = biases[channel];
            for (tap, weight) in weights[channel].iter().enumerate() {
                let index = 2 * frame as isize + 2 * tap as isize - 3;
                let sample = if (0..length as isize).contains(&index) {
                    input[channel * length + index as usize]
                } else if mode == PadMode::Edge {
                    input[channel * length + index.clamp(0, length as isize - 1) as usize]
                } else {
                    0.
                };
                sum += sample * weight;
            }
            output.push(sum);
        }
    }
    output
}

#[test]
fn every_chunk_partition_and_final_stride_match_grouped_dilated_equation() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let input = vec![1., 2., 3., 4., 5., 6., 7., -1., 3., 2., -2., 4., 1., -3.];
    let tensor = NumericTensor::f32(input.clone(), &[1, 2, 7]);
    for mode in [PadMode::Constant, PadMode::Edge] {
        let expected = expected_convolution(&input, 7, mode);
        let prototype = convolution(&context, mode);
        assert_eq!(
            prototype
                .forward(&tensor, &context)
                .unwrap()
                .to_f32_vec(&context)
                .unwrap(),
            expected
        );
        // All 64 partitions of seven samples, including single-sample chunks.
        for cuts in 0..64 {
            let mut layer = prototype.clone();
            let mut output = Vec::new();
            assert!(layer
                .step(&chunk(&tensor, 0, 0, &context), &context)
                .unwrap()
                .is_none());
            let mut start = 0;
            for end in 1..=7 {
                if end == 7 || cuts & (1 << (end - 1)) != 0 {
                    if let Some(y) = layer
                        .step(&chunk(&tensor, start, end, &context), &context)
                        .unwrap()
                    {
                        output.push(y);
                    }
                    assert!(layer.retained_samples() < 5);
                    start = end;
                }
            }
            output.push(
                layer
                    .finish(&context)
                    .unwrap()
                    .expect("odd length has one pending stride"),
            );
            assert_eq!(
                values(&output, &context),
                expected,
                "partition {cuts}, {mode:?}"
            );
            assert_eq!(layer.retained_samples(), 0);
            assert_eq!(layer.phase(), StreamPhase::Finished);
            assert!(layer.finish(&context).unwrap().is_none());
            assert!(matches!(
                layer.step(&tensor, &context),
                Err(eredu_codec::Error::StreamFinished)
            ));
        }
    }
}

fn transpose(context: &Context, tail: TransposeTail) -> StreamableConvTranspose1d<NumericTensor> {
    let mut layer = StreamableConvTranspose1d::unloaded(
        ConvTranspose1dSpec {
            input_channels: 2,
            output_channels: 2,
            kernel_size: 5,
            stride: 2,
            groups: 2,
            bias: true,
            tail,
        },
        context,
    )
    .unwrap();
    layer.weight.replace(NumericTensor::f32(
        vec![1., 2., -1., 0.5, 3., -2., 1., 0.25, 4., -1.],
        &[2, 5, 1],
    ));
    layer
        .bias
        .as_mut()
        .unwrap()
        .replace(NumericTensor::f32(vec![0.5, -0.25], &[2]));
    layer
}

// Independent finite transposed-convolution scatter equation, including a single
// bias contribution per output sample regardless of the number of overlaps.
fn expected_transpose(input: &[f32], length: usize, tail: TransposeTail) -> Vec<f32> {
    let weights = [[1., 2., -1., 0.5, 3.], [-2., 1., 0.25, 4., -1.]];
    let biases = [0.5, -0.25];
    let out_length = length * 2 + if tail == TransposeTail::Emit { 3 } else { 0 };
    let mut output = vec![0.; 2 * out_length];
    for channel in 0..2 {
        for position in 0..out_length {
            output[channel * out_length + position] = biases[channel];
        }
        for frame in 0..length {
            for (tap, weight) in weights[channel].iter().enumerate() {
                let position = frame * 2 + tap;
                if position < out_length {
                    output[channel * out_length + position] +=
                        input[channel * length + frame] * weight;
                }
            }
        }
    }
    output
}

#[test]
fn overlap_add_counts_bias_once_and_obeys_declared_tail_for_every_partition() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let input = vec![1., -2., 3., 4., 2., -1.];
    let tensor = NumericTensor::f32(input.clone(), &[1, 2, 3]);
    for tail in [TransposeTail::Trim, TransposeTail::Emit] {
        let expected = expected_transpose(&input, 3, tail);
        let prototype = transpose(&context, tail);
        assert_eq!(
            prototype
                .forward(&tensor, &context)
                .unwrap()
                .to_f32_vec(&context)
                .unwrap(),
            expected
        );
        for cuts in 0..4 {
            let mut layer = prototype.clone();
            let mut output = Vec::new();
            assert_eq!(
                layer
                    .step(&chunk(&tensor, 0, 0, &context), &context)
                    .unwrap()
                    .shape(),
                &[1, 2, 0]
            );
            let mut start = 0;
            for end in 1..=3 {
                if end == 3 || cuts & (1 << (end - 1)) != 0 {
                    output.push(
                        layer
                            .step(&chunk(&tensor, start, end, &context), &context)
                            .unwrap(),
                    );
                    assert_eq!(layer.retained_samples(), 3);
                    start = end;
                }
            }
            if let Some(y) = layer.finish(&context).unwrap() {
                output.push(y);
            }
            assert_eq!(
                values(&output, &context),
                expected,
                "partition {cuts}, {tail:?}"
            );
            assert!(layer.finish(&context).unwrap().is_none());
            assert!(matches!(
                layer.step(&tensor, &context),
                Err(eredu_codec::Error::StreamFinished)
            ));
        }
    }
}

#[test]
fn forked_carry_is_independent_and_failed_step_can_be_retried() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let first = NumericTensor::f32(vec![1., 2., -1., 3.], &[1, 2, 2]);
    let second = NumericTensor::f32(vec![3., 2.], &[1, 2, 1]);
    let alternative = NumericTensor::f32(vec![10., -5.], &[1, 2, 1]);
    let mut layer = convolution(&context, PadMode::Constant);
    let prefix = layer.step(&first, &context).unwrap().unwrap();
    let snapshot = layer.clone();
    context.set_failure(FailAt::Execution);
    // Complete a stride so the convolution mechanism is actually invoked.
    let two = NumericTensor::concatenate(&[second.clone(), second.clone()], 2, &context).unwrap();
    assert!(layer.step(&two, &context).is_err());
    context.set_failure(FailAt::None);
    let retried = layer.step(&two, &context).unwrap().unwrap();
    let mut replay = snapshot.clone();
    assert_eq!(
        retried.to_f32_vec(&context).unwrap(),
        replay
            .step(&two, &context)
            .unwrap()
            .unwrap()
            .to_f32_vec(&context)
            .unwrap()
    );
    layer.finish(&context).unwrap();
    let mut fork = snapshot;
    assert!(fork.step(&alternative, &context).unwrap().is_none());
    let last = fork.finish(&context).unwrap().unwrap();
    assert_eq!(
        values(&[prefix, last], &context),
        expected_convolution(&[1., 2., 10., -1., 3., -5.], 3, PadMode::Constant)
    );
    fork.reset_state();
    assert_eq!(fork.phase(), StreamPhase::AcceptingInput);
    assert_eq!(fork.retained_samples(), 0);
    assert!(fork.step(&second, &context).unwrap().is_none());
    assert_eq!(
        fork.finish(&context)
            .unwrap()
            .unwrap()
            .to_f32_vec(&context)
            .unwrap(),
        expected_convolution(&[3., 2.], 1, PadMode::Constant)
    );
}
