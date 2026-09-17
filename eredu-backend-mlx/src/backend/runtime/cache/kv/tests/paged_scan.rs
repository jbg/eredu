use super::*;
use crate::{MlxTensor, backend::nn::shared::MlxNeuralBackend};
use eredu_nn::{AttentionArithmetic, AttentionRequest, NeuralBackend};

#[test]
#[ignore = "requires native CPU cache/attention producers"]
fn paged_scan_prefix_window_tail_and_two_pass_cap_match_nonzero_contiguous_attention() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let array = |shape: &[i32], count: usize, phase: usize| {
        Array::from_slice(
            &(0..count)
                .map(|i| ((i * 7 + phase) % 23) as f32 * 0.07 - 0.6)
                .collect::<Vec<_>>(),
            shape,
        )
    };
    let queries = array(&[1, 2, 3, 4], 24, 3);
    let keys = array(&[1, 1, 8, 4], 32, 7);
    let values = array(&[1, 1, 8, 4], 32, 11);
    let sinks = array(&[2], 2, 5);
    // Absolute queries 5..8, causal width3 plus absolute prefix0..1.
    let mask = Array::from_slice(
        &(0..3)
            .flat_map(|q| {
                (0..8).map(move |k| {
                    if k <= q + 5 && (k >= q + 3 || k < 1) {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>(),
        &[3, 8],
    );
    let mask_tensor = MlxTensor::from_array(mask);
    let sink_tensor = MlxTensor::from_array(sinks.clone());
    for arithmetic in [AttentionArithmetic::Fused, AttentionArithmetic::InputScores] {
        let expected = MlxNeuralBackend::attention_with_sinks(
            AttentionRequest {
                arithmetic,
                queries: MlxTensor::from_array(queries.clone()),
                keys: MlxTensor::from_array(keys.clone()),
                values: MlxTensor::from_array(values.clone()),
                scale: 0.5,
                softcap: Some(1.75),
                mask: Some(&mask_tensor),
                sinks: Some(&sink_tensor),
            },
            stream,
        )
        .unwrap();
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(3, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let mut cache =
            PagedKeyValueCache::new_with_layout(manager.clone(), 3, Some(3), 1, None).unwrap();
        cache
            .update_for_attention(keys.clone(), values.clone(), stream)
            .unwrap();
        let actual = cache
            .paged_attention(
                &queries,
                0.5,
                None,
                Some(&sinks),
                Some(1.75),
                arithmetic,
                stream,
            )
            .unwrap()
            .unwrap();
        let actual = actual.evaluated().unwrap();
        let expected = expected.as_array().evaluated().unwrap();
        for (a, b) in actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
        {
            assert!((a - b).abs() < 3e-5, "{arithmetic:?}: {a} vs {b}");
        }
        assert_eq!(cache.offset, 8);
        assert_eq!(cache.tail_start, 6);
        // The partial prefix block and final sealed window block remain.
        let ids = manager
            .layer_block_ids(3, CacheRepresentation::KeyValue, 0, 8, 1)
            .unwrap();
        assert_eq!(
            ids.iter().map(|id| id.start..id.end).collect::<Vec<_>>(),
            [0..3, 3..6]
        );
    }
}
