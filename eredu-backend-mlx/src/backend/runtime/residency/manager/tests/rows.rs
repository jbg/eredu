#[test]
fn row_ranges_share_native_budget_pins_and_bounded_catalog_retirement() {
    use eredu_checkpoint::{recipe::DerivedWeightRecipe,rows::PreparedRowSource,store::MemoryWeightStore};
    use eredu_core::residency::OffloadUnitRange;
    use eredu_runtime::RowResidencyRange;
    let source:eredu_checkpoint::store::SharedCheckpointSource=Arc::new(MemoryWeightStore::from_safetensors([
        ("table".into(),Dtype::F32,vec![128,2],(0..256).flat_map(|i| (i as f32+0.5).to_le_bytes()).collect()),
    ]).unwrap());
    let table=PreparedRowSource::new(source.clone(),DerivedWeightRecipe::source("table",TensorSelection::Full)).unwrap();
    let range=OffloadUnitRange::new(OffloadUnitId::new("table.rows").unwrap(),0,128,8,ResidencyPolicy::Cacheable).unwrap();
    let rows=RowResidencyRange::new(range.clone(),table,"row").unwrap();
    let plan=OffloadPlan::with_ranges(OffloadConfig::new(Some(16),Some(65536),1).unwrap(),[],[range.clone()]).unwrap();
    let stream=cpu_stream();
    let manager=ResidencyManager::new_shared_row_ranges(source,BTreeMap::new(),plan,[],vec![rows],stream.clone(),stream).unwrap();
    manager.initialize().unwrap();
    assert!(manager.lock().unwrap().storage.is_empty());
    let pinned=manager.acquire(&range.member_id(0).unwrap(),MemoryTier::Device).unwrap();
    for row in 1..64 {
        let id=range.member_id(row).unwrap();
        let lease=manager.acquire(&id,MemoryTier::Device).unwrap();
        let array=lease.device_value("row").unwrap();
        let evaluated=array.evaluated().unwrap();
        assert_eq!(evaluated.as_slice::<f32>(),&[row as f32*2.+0.5,row as f32*2.+1.5]);
        let state=manager.lock().unwrap();
        assert!(state.storage.len()<=2);
        assert!(state.control.units().len()<=2);
        assert_eq!(state.control.ledger().unit_reports().len(),2);
        drop(state);
        drop(lease);
        MlxNeuralBackend::reclaim_retired_resources();
    }
    let ids=(64..67).map(|row| (range.member_id(row).unwrap(),1)).collect::<Vec<_>>();
    assert!(manager.acquire_many_with_demand(&ids,MemoryTier::Device).is_err());
    let state=manager.lock().unwrap();
    assert_eq!(state.storage.len(),2,"failed admission cannot accumulate uncached row records");
    assert_eq!(state.control.units().len(),2);
    drop(state);
    assert_eq!(pinned.device_value("row").unwrap().evaluated().unwrap().as_slice::<f32>(),&[0.5,1.5]);
    drop(pinned);
    MlxNeuralBackend::reclaim_retired_resources();
    for row in [0,63] { assert!(manager.evict(&range.member_id(row).unwrap(),MemoryTier::Device).unwrap()); }
    let state=manager.lock().unwrap();
    assert!(state.storage.is_empty());
    assert_eq!(state.control.units().len(),0);
}
