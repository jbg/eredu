#[test]
fn indexed_paged_history_matches_scalar_grouped_reference_without_scanning_history() {
    use crate::MlxTensor;
    use eredu_nn::{AttentionArithmetic, IndexedAttentionInput};
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let options = PagedCacheOptions::new(2, 160, 512 * 1024, 1).unwrap().with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let key_data = patterned_values(36, 3, 17, 0.125);
    let value_data = patterned_values(36, 5, 19, 0.25);
    let query_data = patterned_values(16, 7, 11, 0.2);
    let keys = MlxTensor::from_array(Array::from_slice(&key_data, &[1,2,9,2]));
    let values = MlxTensor::from_array(Array::from_slice(&value_data, &[1,2,9,2]));
    for start in (0..9).step_by(2) {
        let end = (start + 2).min(9);
        cache.update_for_attention(
            keys.as_array().try_index_device((.., .., start..end, ..), stream).unwrap(),
            values.as_array().try_index_device((.., .., start..end, ..), stream).unwrap(), stream,
        ).unwrap();
    }
    let queries = MlxTensor::from_array(Array::from_slice(&query_data, &[1,4,2,2]));
    let positions = MlxTensor::from_array(Array::from_slice(&[8i32,0,8,-1,999, -1,-1,-1,-1,-1], &[1,2,5]));
    let validity = MlxTensor::from_array(Array::from_slice(&[true,true,true,true,false,true,true,true,true,true], &[1,2,5]));
    let request = IndexedAttentionInput {
        queries: &queries, keys: &keys, values: &values, key_position_offset: 0,
        selected_positions: &positions, validity: Some(&validity), mask: None, local: None,
        scale: 0.7, arithmetic: AttentionArithmetic::Fused, sinks: None,
    };
    let resident = crate::backend::nn::attention::indexed_sparse_attention(&request, stream).unwrap();
    for _ in 0..2 {
        let before = manager.report().unwrap();
        let paged = cache.paged_indexed_attention(&request, stream).unwrap().unwrap();
        let data = paged.evaluated().unwrap();
        for head in 0..4 {
            let kv = head / 2;
            let scores = [8,0,8].map(|position| (0..2).map(|d| query_data[head*4+d] * key_data[(kv*9+position)*2+d]).sum::<f32>() * 0.7);
            let denom = scores.iter().map(|s| s.exp()).sum::<f32>();
            for d in 0..2 {
                let expected = [8,0,8].into_iter().zip(scores).map(|(position, score)| score.exp()*value_data[(kv*9+position)*2+d]).sum::<f32>()/denom;
                assert!((data.as_slice::<f32>()[head*4+d]-expected).abs()<1e-5);
                assert_eq!(data.as_slice::<f32>()[head*4+2+d], 0.0);
            }
        }
        assert!(paged.all_close(&resident, 1e-5,1e-5,None,stream).unwrap().item::<bool>(stream));
        let report = manager.report().unwrap();
        // Exactly the first sealed page and partial tail; no intervening pages,
        // duplicate acquisition, or reads for the fully invalid second query.
        assert_eq!(report.selected_attention_blocks-before.selected_attention_blocks, 2);
        assert!(report.peak_device_bytes <= 160);
        assert!(report.peak_host_bytes <= 512*1024);
    }
    let invalid = MlxTensor::from_array(Array::from_slice(&[9i32,0,0,0,0,0,0,0,0,0], &[1,2,5]));
    let bad = IndexedAttentionInput { selected_positions: &invalid, ..request };
    assert!(cache.paged_indexed_attention(&bad, stream).unwrap_err().what().contains("exceeds retained"));
    // Failed reads leave retained state and future lookups usable.
    assert_eq!(cache.offset(), 9);
    assert!(cache.paged_indexed_attention(&request, stream).is_ok());
    let snapshot = cache.checkpoint_clone_state().unwrap();
    let next = Array::from_slice(&[1.0f32,2.0,3.0,4.0], &[1,2,1,2]);
    cache.update_for_attention(next.clone(), next, stream).unwrap();
    cache.restore_checkpoint(&snapshot, stream).unwrap();
    assert_eq!(cache.offset(), 9);
    let restored = cache.paged_indexed_attention(&request, stream).unwrap().unwrap();
    assert!(restored.all_close(&resident, 1e-5,1e-5,None,stream).unwrap().item::<bool>(stream));
}
