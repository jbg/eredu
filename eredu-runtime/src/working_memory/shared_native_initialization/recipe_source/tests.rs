use super::*;
use eredu_checkpoint::store::{
    MemoryWeightStore, RestrictedCheckpointSource, SafetensorsWeightStore, TensorSelection,
    MaterializedCheckpointSource, PreparedCheckpointSource, PreparedTensorSource,
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::sync::Arc;

#[test]
fn routed_memory_and_file_recipes_keep_custody_after_source_and_key_retirement() {
    let recipe = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![
            DerivedWeightRecipe::source(
                "雪",
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![3, 0, 3]
                }
            );
            2
        ],
    };
    let probe = EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap();
    let bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&probe);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(bytes.is_ok(), "{bytes:?}");
    }
    if matches!(bytes, Err(WorkingMemoryError::UnknownBound)) {
        return;
    }
    let required = bytes.unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    serialize_to_file(
        [(
            "雪",
            TensorView::new(Dtype::U8, vec![4], &[3, 7, 11, 17]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let memory = || -> RetainedCheckpointSource { Arc::new(
        MemoryWeightStore::from_safetensors([(
            "雪".into(),
            Dtype::U8,
            vec![4],
            vec![3, 7, 11, 17],
        )])
        .unwrap(),
    )
    .into() };
    let file = || -> RetainedCheckpointSource {
        Arc::new(SafetensorsWeightStore::open(&path).unwrap()).into()
    };
    let mut sources = vec![memory(), file()];
    for source in [memory(), file()] {
        let overlay: RetainedCheckpointSource = Arc::new(MaterializedCheckpointSource::new(
            source, MemoryWeightStore::from_safetensors([]).unwrap(),
            Default::default(), Default::default(),
        )).into();
        let catalog = overlay.source_keys().into_iter().map(|key| {
            let row = PreparedTensorSource {
                metadata: overlay.source_metadata(&key).unwrap(),
                provenance: overlay.source_provenance(&key).unwrap(),
            };
            (key, row)
        }).collect();
        sources.push(Arc::new(PreparedCheckpointSource::new(overlay, catalog).unwrap()).into());
    }
    for source in sources {
        let root: RetainedCheckpointSource = Arc::new(
            RestrictedCheckpointSource::including(
                source,
                "selected",
                std::collections::BTreeSet::from(["雪".into()]),
            )
            .unwrap(),
        )
        .into();
        let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
        let Err(EncodedRecipeSourceError::Keys(error)) =
            short.prepare_encoded_recipe(&root, &recipe)
        else {
            panic!("key comparison precedes source access")
        };
        assert!(
            matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == required && *available_bytes == required - 1)
        );
        drop(error);
        assert_eq!(short.used_bytes().unwrap(), 0);
        let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
        let read = pool
            .prepare_encoded_recipe(&root, &recipe)
            .unwrap()
            .unwrap();
        assert_eq!(root.source_diagnostics().unwrap().physical_read_bytes, 0);
        drop(root);
        let mut output = [0; 6];
        EncodedRecipeRead::read_many_borrowed_into(std::iter::once(&read), &mut [&mut output])
            .unwrap();
        assert_eq!(output, [17, 3, 17, 17, 3, 17]);
        assert!(pool.used_bytes().unwrap() > 0);
        drop(read);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
