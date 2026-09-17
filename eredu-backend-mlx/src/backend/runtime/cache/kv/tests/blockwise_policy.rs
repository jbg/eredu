use super::*;
use crate::{MlxTensor, backend::nn::shared::MlxNeuralBackend};
use eredu_nn::{
    AttentionArithmetic, AttentionRequest, BlockwiseAttentionBackend, BlockwiseAttentionOptions,
    BlockwiseAttentionSpec, NeuralBackend,
};

#[test]
#[ignore = "requires native CPU attention"]
fn typed_blockwise_rounding_cap_and_bias_match_nonzero_contiguous_attention() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let tensor = |shape: &[i32], count: usize, phase: usize| {
        MlxTensor::from_array(Array::from_slice(
            &(0..count)
                .map(|i| ((i * 7 + phase) % 19) as f32 * 0.09 - 0.65)
                .collect::<Vec<_>>(),
            shape,
        ))
    };
    let q = tensor(&[1, 2, 3, 4], 24, 3);
    let k = tensor(&[1, 1, 5, 4], 20, 7);
    let v = tensor(&[1, 1, 5, 4], 20, 11);
    let sinks = tensor(&[2], 2, 5);
    let bias = (0..3)
        .flat_map(|query| (0..5).map(move |key| (query * 5 + key) as f32 * 0.013 - 0.08))
        .collect::<Vec<_>>();
    let complete_bias = Array::from_slice(&bias, &[3, 5]);
    let mask = Array::from_slice(
        &bias
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index % 5 <= index / 5 + 2 {
                    *value
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>(),
        &[3, 5],
    );
    let mask = MlxTensor::from_array(mask);
    for arithmetic in [AttentionArithmetic::Fused, AttentionArithmetic::InputScores] {
        let options = BlockwiseAttentionOptions {
            arithmetic,
            softcap: Some(1.75),
        };
        let expected = MlxNeuralBackend::attention_with_sinks(
            AttentionRequest {
                arithmetic,
                queries: q.clone(),
                keys: k.clone(),
                values: v.clone(),
                scale: 0.5,
                softcap: options.softcap,
                mask: Some(&mask),
                sinks: Some(&sinks),
            },
            stream,
        )
        .unwrap();
        let mut accumulator = MlxNeuralBackend::begin_blockwise_attention_with_options(
            BlockwiseAttentionSpec {
                queries: &q,
                scale: 0.5,
                mask: None,
                query_start: 2,
                context_end: 5,
                sliding_window: None,
                prefix_tokens: 0,
                sinks: Some(&sinks),
            },
            options,
            stream,
        )
        .unwrap();
        for pass in 0..options.passes() {
            if pass == 1 {
                MlxNeuralBackend::begin_blockwise_value_pass(&mut accumulator, stream).unwrap();
            }
            for (start, end) in [(0, 2), (2, 5)] {
                let keys = MlxTensor::from_array(
                    k.as_array()
                        .try_index_device((.., .., start..end, ..), stream)
                        .unwrap(),
                );
                let values = MlxTensor::from_array(
                    v.as_array()
                        .try_index_device((.., .., start..end, ..), stream)
                        .unwrap(),
                );
                let bias = MlxTensor::from_array(
                    complete_bias
                        .try_index_device((.., start..end), stream)
                        .unwrap(),
                );
                MlxNeuralBackend::accumulate_blockwise_attention_with_bias(
                    &mut accumulator,
                    start as i64,
                    end as i64,
                    keys,
                    values,
                    Some(&bias),
                    stream,
                )
                .unwrap();
            }
        }
        let actual = MlxNeuralBackend::finish_blockwise_attention(accumulator, stream).unwrap();
        let actual = actual.as_array().evaluated().unwrap();
        let expected = expected.as_array().evaluated().unwrap();
        for (actual, expected) in actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
        {
            assert!(
                (actual - expected).abs() < 3e-5,
                "{arithmetic:?}: {actual} vs {expected}"
            );
        }
    }
}
