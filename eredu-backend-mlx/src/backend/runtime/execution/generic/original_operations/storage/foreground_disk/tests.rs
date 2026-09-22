use super::*;
use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_runtime::residency::{OffloadUnit, WeightBinding};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::{collections::BTreeMap, sync::Arc};

#[test]
fn empty_disk_schedule_retains_source_identity_without_allocation_population() {
    if !crate::tests::support::native_process::enter("empty-disk-source") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let directory = tempfile::tempdir().unwrap();
    let bytes = 7.0f32.to_le_bytes();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::F32, vec![1], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let primary: eredu_checkpoint::store::RetainedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap()).into();
    let id = OffloadUnitId::new("unit").unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(4), Some(4096), 1).unwrap(),
        [
            OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                .unwrap(),
        ],
    )
    .unwrap();
    let units = [OffloadUnit::new(
        id,
        [WeightBinding::new("weight", "weight", TensorSelection::Full, 4).unwrap()],
    )
    .unwrap()];
    let make = || {
        crate::backend::runtime::residency::manager::prepare_foreground_disk_descriptors(
            &primary,
            &BTreeMap::new(),
            &plan,
            &units,
            &["only".into()],
            &pool,
        )
        .unwrap()
        .unwrap()
    };
    let source = make();
    let foreign = make();
    assert!(!source.same_source(&foreign));
    let request = |source| ForegroundDiskRequestPlan {
        windows: Vec::new(),
        source,
        pool: pool.clone(),
        forwards: 1,
        temporary_bytes: 0,
        nested_source_peak: None,
        background: None,
    };
    let actual = request(source.clone());
    let facts = actual.source_facts().unwrap();
    assert_eq!(
        actual.population(),
        Some(ForegroundDiskPopulation::default())
    );
    assert_eq!(actual.source_backing_capacity(), Some(0));
    assert_eq!(facts.capacity_bytes(), 0);
    assert_eq!(facts.maximum_attempts(), 0);
    assert_eq!(facts.maximum_partitions(), 0);
    assert_eq!(actual.host_facts(), request(source).host_facts());
    assert_ne!(actual.host_facts(), request(foreign).host_facts());
    assert!(
        facts.protected_bytes() > 0,
        "the zero bank still pays its control owner"
    );
}
