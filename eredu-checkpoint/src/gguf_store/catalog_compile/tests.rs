use super::super::tests::test_plan;
use super::*;
use crate::{schema::CatalogPolicy, store::WeightStore, validation::resolve_gguf_plan};
use eredu_gguf::{Endian, GgmlType, MetadataValue, TensorInput, Writer, WriterOptions};
use std::{
    fs::File,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn write(
    path: &Path,
    metadata: &BTreeMap<String, MetadataValue>,
    inputs: &[TensorInput<'_>],
    endian: Endian,
) {
    Writer::new(WriterOptions {
        version: 3,
        endian,
        alignment: 32,
    })
    .unwrap()
    .write(File::create(path).unwrap(), metadata, inputs)
    .unwrap();
}
fn dense_pair(path: &Path) -> Checkpoint {
    let bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
    write(
        path,
        &BTreeMap::new(),
        &[
            TensorInput {
                name: "first",
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            },
            TensorInput {
                name: "second",
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            },
        ],
        Endian::Little,
    );
    Checkpoint::open(path).unwrap()
}
fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}

#[test]
fn immutable_catalog_multishard_dense_quantized_outputs_preserve_sorted_metadata_and_reads() {
    let dir = tempfile::tempdir().unwrap();
    for endian in [Endian::Little, Endian::Big] {
        for ty in [GgmlType::Q4_0, GgmlType::MxFp4] {
            let paths = [
                dir.path().join("weights-00001-of-00002.gguf"),
                dir.path().join("weights-00002-of-00002.gguf"),
            ];
            let dense = [1.25_f32, -3.5]
                .into_iter()
                .flat_map(|v| match endian {
                    Endian::Little => v.to_le_bytes(),
                    Endian::Big => v.to_be_bytes(),
                })
                .collect::<Vec<_>>();
            let (values, bytes) = ty.block_and_bytes().unwrap();
            let packed = (0..128 / values * bytes)
                .map(|i| (i % 31 + 1) as u8)
                .collect::<Vec<_>>();
            for (i, path) in paths.iter().enumerate() {
                let metadata = BTreeMap::from([
                    ("split.no".into(), MetadataValue::Uint16(i as u16)),
                    ("split.count".into(), MetadataValue::Uint16(2)),
                    ("split.tensors.count".into(), MetadataValue::Uint64(2)),
                ]);
                let input = if i == 0 {
                    TensorInput {
                        name: "z_dense",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &dense,
                    }
                } else {
                    TensorInput {
                        name: "a_matrix.weight",
                        dimensions: &[64, 2],
                        ggml_type: ty,
                        data: &packed,
                    }
                };
                write(path, &metadata, &[input], endian);
            }
            let checkpoint = Checkpoint::open(&paths[0]).unwrap();
            let plan = test_plan(&checkpoint);
            let resolved = resolve_gguf_plan(&checkpoint, &plan).unwrap();
            let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
            let mut expected_keys = mapping
                .iter()
                .map(|m| m.layout.name.clone())
                .collect::<Vec<_>>();
            expected_keys.sort();
            let ordinary = GgufWeightStore::builder()
                .add_resolved_checkpoint(checkpoint.clone(), &resolved, &mapping)
                .unwrap()
                .build_with_prepared_reader_buffers()
                .unwrap();
            let drops = Arc::new(AtomicUsize::new(0));
            let completed = GgufCatalogPlan::new(checkpoint, &resolved, &mapping, 1)
                .compile(Custody(drops.clone()))
                .unwrap();
            let source = completed.build_with_prepared_reader_buffers().unwrap();
            assert_eq!(source.keys(), expected_keys);
            assert_eq!(source.keys().last().unwrap(), "z_dense");
            assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
            for key in source.keys() {
                assert_eq!(
                    source.metadata(&key).unwrap(),
                    ordinary.metadata(&key).unwrap()
                );
                assert_eq!(
                    source
                        .acquire(request(&key))
                        .unwrap()
                        .materialize_portable()
                        .unwrap(),
                    ordinary
                        .acquire(request(&key))
                        .unwrap()
                        .materialize_portable()
                        .unwrap()
                );
            }
            let alias = source.clone();
            drop(source);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(alias);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn immutable_catalog_mapping_refusals_and_unclaimed_order_match_the_ordinary_worker() {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dense_pair(&dir.path().join("catalog.gguf"));
    let plan = test_plan(&checkpoint);
    let resolved = resolve_gguf_plan(&checkpoint, &plan).unwrap();
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    for scenario in 0..3 {
        let mut map = mapping.clone();
        let (rows, expected) = match scenario {
            0 => {
                map.push(map[0].clone());
                (0, "duplicate source output")
            }
            1 => {
                map.pop();
                (1, "omits a catalog output")
            }
            _ => {
                map[1].layout.name = map[0].layout.name.clone();
                (1, "collides with an existing output")
            }
        };
        let expected = format!(
            "GGUF checkpoint operation failed for tensor {:?}: {}",
            if scenario == 1 { "second" } else { "first" },
            match expected {
                "duplicate source output" =>
                    "admitted GGUF tensor mapping contains a duplicate source output",
                "omits a catalog output" => "admitted GGUF tensor mapping omits a catalog output",
                _ => "translated logical tensor collides with an existing output",
            }
        );
        let ordinary = GgufWeightStore::builder()
            .add_resolved_checkpoint(checkpoint.clone(), &resolved, &map)
            .unwrap_err();
        let original = GgufCatalogPlan::new(checkpoint.clone(), &resolved, &map, 1)
            .compile(())
            .unwrap_err();
        assert_eq!(ordinary.to_string(), expected);
        assert_eq!(original.to_string(), expected);
        assert_eq!(original.completed_rows(), rows);
    }
    let selected = crate::schema::GgufCheckpointPlan::new(
        "one selected",
        vec![plan.common_tensors[0].clone()],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let selected = resolve_gguf_plan(&checkpoint, &selected).unwrap();
    let ordinary = GgufWeightStore::builder()
        .add_resolved_checkpoint(checkpoint.clone(), &selected, &mapping)
        .unwrap()
        .build()
        .unwrap();
    let source = GgufCatalogPlan::new(checkpoint.clone(), &selected, &mapping, 1)
        .compile(())
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(source.keys(), vec!["first"]);
    assert_eq!(source.unclaimed_checkpoint_keys(), vec!["second"]);
    assert_eq!(
        source.unclaimed_checkpoint_keys(),
        ordinary.unclaimed_checkpoint_keys()
    );
    // Missing mapping of an unclaimed output is still checked, after first row.
    let failed = GgufCatalogPlan::new(checkpoint, &selected, &mapping[..1], 1)
        .compile(())
        .unwrap_err();
    assert_eq!(failed.completed_rows(), 1);
    assert!(failed.to_string().contains("omits a catalog output"));
}

#[test]
fn immutable_catalog_reached_reserve_failure_keeps_original_input_and_successful_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained.gguf");
    let checkpoint = dense_pair(&path);
    let plan = test_plan(&checkpoint);
    let resolved = resolve_gguf_plan(&checkpoint, &plan).unwrap();
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    let input_name = checkpoint.shards()[0].tensors()[0]
        .descriptor()
        .name
        .as_ptr();
    let drops = Arc::new(AtomicUsize::new(0));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FAIL_RESERVE.with(|n| n.set(None));
        }
    }
    let reset = Reset;
    // first physical + logical, second physical succeed; second logical fails.
    FAIL_RESERVE.with(|n| n.set(Some(3)));
    let error = GgufCatalogPlan::new(checkpoint, &resolved, &mapping, 1)
        .compile(Custody(drops.clone()))
        .unwrap_err();
    drop(reset);
    assert!(matches!(error.cause.issue, Issue::Reserve(_)));
    assert_eq!(error.completed_rows(), 1);
    assert_eq!(
        error.checkpoint().shards()[0].tensors()[0]
            .descriptor()
            .name
            .as_ptr(),
        input_name
    );
    assert_eq!(
        error
            .builder
            .catalog
            .get("first")
            .unwrap()
            .metadata
            .logical_shape,
        [2]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        error.completed_rows(),
        1,
        "diagnostics do not reopen the input"
    );
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
