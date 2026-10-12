#[test]
fn read_only_attention_combines_prefix_and_transient_canvas_without_publication() {
    use crate::MlxTensor;
    use eredu_nn::{AttentionArithmetic, AttentionRequest};
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 128, 128 * 1024, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut paged = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let mut resident = ConcatKeyValueCache::new_append_only();
    let keys = Array::from_slice(&[0.2f32, -0.3, 0.7, 0.5], &[1, 1, 4, 1]);
    let values = Array::from_slice(&[2.0f32, -1.0, 3.0, 4.0], &[1, 1, 4, 1]);
    for cache in [&mut resident as &mut dyn KeyValueCache, &mut paged] {
        cache
            .update_for_attention(keys.clone(), values.clone(), stream)
            .unwrap();
    }
    let report = manager.report().unwrap();
    for changed in [0.0f32, 0.8, -0.4] {
        let queries = MlxTensor::from_array(Array::from_slice(&[0.4f32, -0.6], &[1, 1, 2, 1]));
        let transient_keys = MlxTensor::from_array(Array::from_slice(
            &[0.1f32 + changed, 0.9 - changed],
            &[1, 1, 2, 1],
        ));
        let transient_values = MlxTensor::from_array(Array::from_slice(
            &[-3.0f32 + changed, 5.0 - changed],
            &[1, 1, 2, 1],
        ));
        let request = || AttentionRequest {
            arithmetic: AttentionArithmetic::Fused,
            queries: queries.clone(),
            keys: transient_keys.clone(),
            values: transient_values.clone(),
            scale: 1.0,
            mask: None,
            softcap: None,
            sinks: None,
        };
        let read = |a: Array| a.into_evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let actual_resident = read(
            resident
                .attention_with_transient(request(), stream)
                .unwrap(),
        );
        let actual_paged = read(paged.attention_with_transient(request(), stream).unwrap());
        // Independent scalar softmax over all prefix and canvas positions.
        for (position, query) in [0.4f64, -0.6].into_iter().enumerate() {
            let keys = [
                0.2f64,
                -0.3,
                0.7,
                0.5,
                f64::from(0.1 + changed),
                f64::from(0.9 - changed),
            ];
            let values = [
                2.0f64,
                -1.0,
                3.0,
                4.0,
                f64::from(-3.0 + changed),
                f64::from(5.0 - changed),
            ];
            let weights: Vec<_> = keys.into_iter().map(|key| (query * key).exp()).collect();
            let expected = weights
                .iter()
                .zip(values)
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                / weights.iter().sum::<f64>();
            for actual in [&actual_resident, &actual_paged] {
                assert!(
                    (f64::from(actual[position]) - expected).abs() < 2e-5,
                    "{actual:?} != {expected}"
                );
            }
        }
        assert_eq!(resident.offset(), 4);
        assert_eq!(paged.offset(), 4);
        let after = manager.report().unwrap();
        assert_eq!(after.logical_cached_tokens, report.logical_cached_tokens);
        assert_eq!(after.key_value_blocks, report.key_value_blocks);
    }
    let output = resident.snapshot_arrays(stream).unwrap().unwrap();
    assert_eq!(
        output
            .0
            .into_evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        [0.2, -0.3, 0.7, 0.5]
    );
}

#[test]
fn read_only_sliding_attention_excludes_page_overhang_without_clipping_canvas() {
    use crate::MlxTensor;
    use eredu_nn::{AttentionArithmetic, AttentionRequest};
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    // The retained suffix starts inside a four-position page. A canvas longer
    // than the window must still attend to every canvas position.
    for window in [1, 3] {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(4, 128, 128 * 1024, 1).unwrap())
                .unwrap();
        let mut paged = PagedKeyValueCache::new(manager.clone(), 0, Some(window)).unwrap();
        let mut resident = ConcatKeyValueCache::new_append_only_for_sliding_attention(window);
        let keys = Array::from_slice(&[0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], &[1, 1, 7, 1]);
        let values =
            Array::from_slice(&[70.0f32, 60.0, 50.0, 40.0, 30.0, 2.0, -3.0], &[1, 1, 7, 1]);
        for cache in [&mut resident as &mut dyn KeyValueCache, &mut paged] {
            cache
                .update_for_attention(keys.clone(), values.clone(), stream)
                .unwrap();
        }
        let before = manager.report().unwrap();
        let query =
            MlxTensor::from_array(Array::from_slice(&[0.4f32, -0.6, 0.8, -0.2], &[1, 1, 4, 1]));
        let canvas_keys =
            MlxTensor::from_array(Array::from_slice(&[0.2f32, -0.1, 0.9, 0.4], &[1, 1, 4, 1]));
        let canvas_values =
            MlxTensor::from_array(Array::from_slice(&[1.0f32, 4.0, -2.0, 6.0], &[1, 1, 4, 1]));
        for cache in [&mut resident as &mut dyn KeyValueCache, &mut paged] {
            let output = cache
                .attention_with_transient(
                    AttentionRequest {
                        arithmetic: AttentionArithmetic::Fused,
                        queries: query.clone(),
                        keys: canvas_keys.clone(),
                        values: canvas_values.clone(),
                        scale: 1.0,
                        mask: None,
                        softcap: None,
                        sinks: None,
                    },
                    stream,
                )
                .unwrap()
                .into_evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let history = if window == 1 {
                vec![]
            } else {
                vec![(-0.6f64, 2.0f64), (0.7, -3.0)]
            };
            for (i, q) in [0.4f64, -0.6, 0.8, -0.2].into_iter().enumerate() {
                let context = history.iter().copied().chain([
                    (0.2, 1.0),
                    (-0.1, 4.0),
                    (0.9, -2.0),
                    (0.4, 6.0),
                ]);
                let (numerator, denominator) = context.fold((0.0, 0.0), |(n, d), (k, v)| {
                    let p = (q * k).exp();
                    (n + p * v, d + p)
                });
                assert!(
                    (f64::from(output[i]) - numerator / denominator).abs() < 2e-5,
                    "window={window}: {output:?}"
                );
            }
            assert_eq!(cache.offset(), 7);
        }
        let after = manager.report().unwrap();
        assert_eq!(before.logical_cached_tokens, after.logical_cached_tokens);
        assert_eq!(before.key_value_blocks, after.key_value_blocks);
    }
}
