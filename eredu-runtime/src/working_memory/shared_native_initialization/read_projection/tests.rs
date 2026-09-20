use super::*;
use eredu_checkpoint::{
    recipe::{EncodedRecipeMapping, EncodedRecipeMappingPlan},
    store::{
        MemoryEncodedReadPlan, MemoryWeightStore, SafetensorsEncodedReadPlan,
        SafetensorsWeightStore,
    },
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

fn required<P: SharedNativeInitializer>(plan: &P) -> Option<u64> {
    let result = WorkingMemoryPool::shared_native_initialization_required_bytes(plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(error) => panic!("{error}"),
    }
}
fn mapping(pool: &WorkingMemoryPool) -> InitializedSharedNative<EncodedRecipeMapping> {
    let full = pool
        .initialize_shared_native(EncodedRecipeMappingPlan::source(0..8).unwrap())
        .unwrap();
    pool.initialize_shared_native(
        EncodedRecipeMappingPlan::selected(full.output(), &[6..8, 0..3, 6..7]).unwrap(),
    )
    .unwrap()
}
fn memory() -> ((), MemoryWeightStore) {
    (
        (),
        MemoryWeightStore::from_safetensors([
            ("a".into(), Dtype::U8, vec![4], vec![1, 2, 3, 4]),
            ("雪".into(), Dtype::U8, vec![4], vec![11, 12, 13, 14]),
        ])
        .unwrap(),
    )
}
fn file() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "a",
                TensorView::new(Dtype::U8, vec![4], &[1, 2, 3, 4]).unwrap(),
            ),
            (
                "雪",
                TensorView::new(Dtype::U8, vec![4], &[11, 12, 13, 14]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(path).unwrap();
    (dir, store)
}

// Source birth, key storage, read scratch and output are fixture prerequisites.
// The actual batch, mapping and projection constructors share one admitted pool.
macro_rules! sequence {
    ($name:ident, $fixture:ident, $read_plan:ident) => {
        #[test]
        fn $name() {
            for short in [false, true] {
                let (_dir, source) = $fixture();
                let keys = ["a".into(), "雪".into()];
                let Some(batch_bytes) = required(&$read_plan::new(&source, &keys).unwrap()) else {
                    return;
                };
                // Derive the exact typed projection quote from an independently
                // admitted fixture and retire it before the exact-budget run.
                let sizing = WorkingMemoryPool::new(1 << 20, 0).unwrap();
                let mapped = mapping(&sizing);
                let map_bytes = mapped.original_bytes();
                let base_bytes =
                    required(&EncodedRecipeMappingPlan::source(0..8).unwrap()).unwrap();
                let batch = sizing
                    .initialize_shared_native($read_plan::new(&source, &keys).unwrap())
                    .unwrap()
                    .into_owned_read();
                let plan = batch.project(mapped.output()).unwrap();
                let projection_bytes = required(&plan).unwrap();
                drop(plan);
                drop(mapped);
                assert_eq!(sizing.used_bytes().unwrap(), 0);
                let total = batch_bytes + map_bytes + projection_bytes;
                assert!(total - 1 >= base_bytes + map_bytes);
                let pool = WorkingMemoryPool::new(total - u64::from(short), 0).unwrap();
                let mapped = mapping(&pool);
                let batch = pool
                    .initialize_shared_native($read_plan::new(&source, &keys).unwrap())
                    .unwrap();
                assert_eq!(pool.used_bytes().unwrap(), batch_bytes + map_bytes);
                let batch = batch.into_owned_read();
                assert_eq!(pool.used_bytes().unwrap(), batch_bytes + map_bytes);
                let result = pool.initialize_shared_native(batch.project(mapped.output()).unwrap());
                drop((source, keys));
                if short {
                    let failure = result.unwrap_err();
                    assert!(matches!(
                        failure.accounting_failure(),
                        Some(WorkingMemoryError::BudgetExceeded { .. })
                    ));
                    assert!(failure.rejected_plan().is_some());
                    assert!(failure.constructor_failure().is_none());
                    assert_eq!(pool.used_bytes().unwrap(), batch_bytes + map_bytes);
                    drop(failure);
                    assert_eq!(pool.used_bytes().unwrap(), map_bytes);
                    drop(mapped);
                } else {
                    let projected = result.unwrap();
                    assert_eq!(pool.used_bytes().unwrap(), total);
                    drop(mapped);
                    assert_eq!(pool.used_bytes().unwrap(), batch_bytes + projection_bytes);
                    let read = projected.into_owned_read();
                    assert_eq!(pool.used_bytes().unwrap(), batch_bytes + projection_bytes);
                    assert_eq!(read.tensors().len(), 2);
                    assert_eq!(read.byte_len(), 6);
                    let mut output = [0; 6];
                    read.read_into(&mut output).unwrap();
                    assert_eq!(output, [13, 14, 1, 2, 3, 13]);
                    let mut short_output = [99; 5];
                    assert!(read.read_into(&mut short_output).is_err());
                    assert_eq!(short_output, [99; 5]);
                    drop(read);
                }
                assert_eq!(pool.used_bytes().unwrap(), 0);
            }
        }
    };
}
sequence!(
    memory_projection_retains_original_and_projected_admission,
    memory,
    MemoryEncodedReadPlan
);
sequence!(
    file_projection_retains_original_and_projected_admission,
    file,
    SafetensorsEncodedReadPlan
);
