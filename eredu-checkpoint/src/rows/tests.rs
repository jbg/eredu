use super::*;
use crate::store::{EncodedTensorLease, MemoryWeightStore, ReadPolicy};
use safetensors::tensor::Dtype;

fn read_f32(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
    reads: &mut Vec<(String, u64)>,
) -> Vec<f32> {
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => {
            let lease = source
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert!(lease.bounded_read_proof().physically_bounded);
            reads.push((key.clone(), lease.bounded_read_proof().length_bytes));
            lease
                .encoded_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        DerivedWeightRecipe::Reshape { input, .. } => read_f32(input, source, reads),
        DerivedWeightRecipe::Concatenate { axis: 0, inputs } => inputs
            .iter()
            .flat_map(|input| read_f32(input, source, reads))
            .collect(),
        other => panic!("fixture requires bounded contiguous row recipes, got {other:?}"),
    }
}

#[test]
fn compact_sharded_rows_coalesce_deduplicate_and_restore_order() {
    let tensors = (0..128).map(|shard| {
        let bytes = (0..8)
            .flat_map(|i| ((shard * 8 + i) as f32 + 0.5).to_le_bytes())
            .collect::<Vec<_>>();
        (format!("shard{shard}"), Dtype::F32, vec![4, 2], bytes)
    });
    let source: SharedCheckpointSource =
        Arc::new(MemoryWeightStore::from_safetensors(tensors).unwrap());
    let recipe = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: (0..128)
            .map(|shard| {
                DerivedWeightRecipe::source(format!("shard{shard}"), TensorSelection::Full)
            })
            .collect(),
    };
    let table = PreparedRowSource::new(source, recipe).unwrap();
    assert_eq!(table.metadata().shape, [512, 2]);
    let requests = [511, 4, 3, 4, 5, 0];
    let limits = RowReadLimits {
        requests: 6,
        rows_per_read: 3,
    };
    for _ in 0..8 {
        let plan = table.plan(&requests, limits).unwrap();
        assert_eq!(plan.unique_rows(), [0, 3, 4, 5, 511]);
        assert_eq!(plan.restore_order(), [4, 2, 1, 2, 3, 0]);
        assert_eq!(plan.reads().len(), 3);
        assert!(CheckpointSource::recipe_cache(plan.source()).is_none());
        let mut reads = Vec::new();
        let mut compact = Vec::new();
        for read in plan.reads() {
            assert!(read.destinations().len() <= 3);
            assert_eq!(read.destinations().start, compact.len() / 2);
            compact.extend(read_f32(read.recipe(), plan.source(), &mut reads));
        }
        assert_eq!(
            reads.iter().map(|(_, bytes)| *bytes).sum::<u64>(),
            5 * 2 * 4
        );
        assert_eq!(
            reads.len(),
            4,
            "one contiguous request across the shard boundary uses only its two source ranges"
        );
        for (&row, &slot) in requests.iter().zip(plan.restore_order()) {
            assert_eq!(
                &compact[slot * 2..slot * 2 + 2],
                &[row as f32 * 2. + 0.5, row as f32 * 2. + 1.5]
            );
        }
    }
    assert!(matches!(
        table.plan(&[512], limits),
        Err(RowReadError::OutOfRange { .. })
    ));
    assert!(matches!(
        table.plan(&[0; 7], limits),
        Err(RowReadError::RequestLimit { .. })
    ));
    assert!(table.plan(&[], limits).unwrap().reads().is_empty());
    assert!(matches!(
        table.plan(
            &[],
            RowReadLimits {
                requests: 0,
                ..limits
            }
        ),
        Err(RowReadError::InvalidLimits)
    ));
}

#[test]
fn row_planning_rejects_malformed_shapes_and_preserves_restricted_sources() {
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            ("table".into(), Dtype::F32, vec![2, 2], vec![0; 16]),
            ("other".into(), Dtype::F32, vec![4], vec![0; 16]),
        ])
        .unwrap(),
    );
    assert!(matches!(
        PreparedRowSource::new(
            source.clone(),
            DerivedWeightRecipe::source("other", TensorSelection::Full)
        ),
        Err(RowReadError::Geometry { .. })
    ));
    assert!(PreparedRowSource::new(
        source.clone(),
        DerivedWeightRecipe::source("missing", TensorSelection::Full)
    )
    .is_err());
    let table = PreparedRowSource::new(
        source,
        DerivedWeightRecipe::source("table", TensorSelection::Full),
    )
    .unwrap();
    let plan = table
        .plan(
            &[1],
            RowReadLimits {
                requests: 1,
                rows_per_read: 1,
            },
        )
        .unwrap();
    assert_eq!(plan.source().source_keys(), ["table"]);
    assert!(plan.source().source_metadata("other").is_err());
    assert_eq!(
        plan.source()
            .source_provenance("table")
            .unwrap()
            .physical_tensor,
        "table"
    );
}

#[test]
fn disk_row_reads_are_bounded_and_preparation_does_not_read_payloads() {
    use crate::store::SafetensorsWeightStore;
    use safetensors::tensor::{serialize_to_file, TensorView};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("model.safetensors");
    let bytes = (0..16384)
        .flat_map(|i| (i as f32 + 0.25).to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "table",
            TensorView::new(Dtype::F32, vec![8192, 2], &bytes).unwrap(),
        )],
        None,
        &file,
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let before = source.source_diagnostics().unwrap();
    let table = PreparedRowSource::new(
        source.clone(),
        DerivedWeightRecipe::source("table", TensorSelection::Full),
    )
    .unwrap();
    let plan = table
        .plan(
            &[8000, 2, 3, 2],
            RowReadLimits {
                requests: 4,
                rows_per_read: 2,
            },
        )
        .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_read_bytes,
        before.physical_read_bytes
    );
    let mut reads = Vec::new();
    for read in plan.reads() {
        read_f32(read.recipe(), plan.source(), &mut reads);
    }
    let after = source.source_diagnostics().unwrap();
    assert_eq!(reads.len(), 2);
    assert_eq!(after.physical_read_bytes - before.physical_read_bytes, 24);
}
