use super::*;
use crate::store::{EncodedTensorLease, ReadPolicy, SafetensorsWeightStore, TensorReadRequest};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};

type Recipe = DerivedWeightRecipe;
fn source(key: &str, selection: TensorSelection) -> Recipe {
    Recipe::source(key, selection)
}
fn concatenate(inputs: Vec<Recipe>) -> Recipe {
    Recipe::Concatenate { axis: 1, inputs }
}
fn fixture() -> (tempfile::TempDir, SafetensorsWeightStore, Vec<u8>) {
    let directory = tempfile::tempdir().unwrap();
    let mut gate = Vec::new();
    let mut up = Vec::new();
    let mut interleaved = Vec::new();
    let mut expected = Vec::new();
    for expert in 0..2 {
        let mut halves = [Vec::new(), Vec::new()];
        for row in 0..3 {
            for side in 0..2 {
                for column in 0..4 {
                    let value = (expert * 100 + side * 40 + row * 4 + column) as f32 + 0.125;
                    halves[side].extend_from_slice(&value.to_le_bytes());
                    interleaved.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        gate.extend_from_slice(&halves[0]);
        up.extend_from_slice(&halves[1]);
        expected.extend_from_slice(&halves[0]);
        expected.extend_from_slice(&halves[1]);
    }
    let bytes = [gate, up, interleaved];
    serialize_to_file(
        ["gate", "up", "interleaved"]
            .into_iter()
            .zip(bytes.iter())
            .enumerate()
            .map(|(i, (key, bytes))| {
                (
                    key,
                    TensorView::new(Dtype::F32, vec![2, if i == 2 { 6 } else { 3 }, 4], bytes)
                        .unwrap(),
                )
            }),
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(directory.path()).unwrap();
    (directory, store, expected)
}
fn selected_rows(indices: Vec<usize>) -> Recipe {
    source("interleaved", TensorSelection::Indices { axis: 1, indices })
}
fn ordinary_rows(store: &SafetensorsWeightStore, indices: Vec<usize>) -> Vec<u8> {
    store
        .acquire_lease(TensorReadRequest {
            key: "interleaved".into(),
            selection: TensorSelection::Indices { axis: 1, indices },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap()
        .encoded_bytes()
        .unwrap()
        .to_vec()
}

#[test]
fn packed_axis_join_and_interleaved_rows_read_exact_final_outputs() {
    let (_directory, store, expected) = fixture();
    let recipes = [
        concatenate(vec![
            source("gate", TensorSelection::Full),
            source("up", TensorSelection::Full),
        ]),
        concatenate(vec![
            selected_rows(vec![0, 2, 4]),
            selected_rows(vec![1, 3, 5]),
        ]),
    ];
    let reads = recipes
        .iter()
        .map(|recipe| recipe.prepare_encoded_read(&store).unwrap().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(store.source_diagnostics().unwrap().physical_read_bytes, 0);
    for read in &reads {
        assert_eq!(read.output().shape, [2, 6, 4]);
        assert_eq!(read.output().dtype, RecipeDtype::F32);
        assert_eq!(read.output().byte_len, 192);
    }
    assert_eq!(
        reads[0]
            .sources()
            .iter()
            .map(|source| source.encoded_byte_len)
            .sum::<u64>(),
        192
    );
    // The two exact source occurrences share a file but each selects only half
    // of it. Metadata provenance is not misreported as bytes actually read.
    assert_eq!(
        reads[1]
            .sources()
            .iter()
            .map(|source| source.encoded_byte_len)
            .sum::<u64>(),
        384
    );
    let mut packed = [0; 192];
    let mut interleaved = [0; 192];
    EncodedRecipeRead::read_many_borrowed_into(reads.iter(), &mut [&mut packed, &mut interleaved])
        .unwrap();
    assert_eq!(packed.as_slice(), expected);
    assert_eq!(interleaved, packed);
    assert_eq!(store.source_diagnostics().unwrap().physical_read_bytes, 384);
    assert_eq!(
        store.source_diagnostics().unwrap().currently_cached_shards,
        0
    );
    // Cross-check the selected leaves through the ordinary bounded lease worker.
    let gate = ordinary_rows(&store, vec![0, 2, 4]);
    let up = ordinary_rows(&store, vec![1, 3, 5]);
    let ordinary = (0..2)
        .flat_map(|expert| {
            gate[expert * 48..(expert + 1) * 48]
                .iter()
                .chain(&up[expert * 48..(expert + 1) * 48])
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(packed.as_slice(), ordinary);
}

#[test]
fn repeated_partial_rows_keep_exact_detached_scratch_and_failure_custody() {
    let (directory, store, _) = fixture();
    let expected = ordinary_rows(&store, vec![5, 1, 5]);
    let read = selected_rows(vec![5, 1, 5])
        .prepare_encoded_read(&store)
        .unwrap()
        .unwrap();
    let whole = source("interleaved", TensorSelection::Full)
        .prepare_encoded_read(&store)
        .unwrap()
        .unwrap();
    assert_eq!(read.output().byte_len, 96);
    assert_eq!(read.sources()[0].encoded_byte_len, 192);
    assert!(read.clone_storage_bytes().unwrap() > whole.clone_storage_bytes().unwrap());
    let custody = Arc::new(());
    let alive = Arc::downgrade(&custody);
    let plan = EncodedRecipeRead::prepare_detached(std::iter::once(&read)).unwrap();
    let quoted = plan.read_layout::<Arc<()>>().unwrap().required_bytes();
    assert!(plan.required_bytes::<Arc<()>>().unwrap() > 0);
    let detached = plan.construct(custody).unwrap();
    assert!(quoted >= detached.read_layout().unwrap().required_bytes());
    assert!(detached.matches_read(0, &read));
    assert!(!detached.matches_read(0, &whole));
    drop((read, whole, store));
    let mut bytes = [0; 96];
    detached.read_many_into(&mut [&mut bytes]).unwrap();
    assert_eq!(bytes.as_slice(), expected);
    assert_eq!(detached.physical_read_bytes(0), Some(96));
    let mut short = [99; 95];
    let short_failure = detached.read_many_into(&mut [&mut short]).unwrap_err();
    assert_eq!(short, [99; 95]);
    assert_eq!(detached.physical_read_bytes(0), Some(96));
    std::fs::remove_file(directory.path().join("model.safetensors")).unwrap();
    let failure = detached.read_many_into(&mut [&mut bytes]).unwrap_err();
    drop(detached);
    assert!(alive.upgrade().is_some());
    drop(failure);
    assert!(alive.upgrade().is_some());
    drop(short_failure);
    assert!(alive.upgrade().is_none());
}

#[test]
fn projection_preserves_nested_geometry_and_refuses_scalar_conversion() {
    let (_directory, store, _) = fixture();
    let selected = selected_rows(vec![4, 0]);
    let expected = ordinary_rows(&store, vec![4, 0]);
    let recipe = Recipe::Stack {
        axis: 2,
        inputs: vec![selected.clone(), selected.clone()],
    };
    let read = recipe.prepare_encoded_read(&store).unwrap().unwrap();
    assert_eq!(read.output().shape, [2, 2, 2, 4]);
    let mut output = vec![0; read.output().byte_len as usize];
    read.read_into(&mut output).unwrap();
    let expected = expected
        .chunks_exact(16)
        .flat_map(|row| row.iter().chain(row).copied())
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
    let before = store.source_diagnostics().unwrap().physical_read_bytes;
    let conversion = Recipe::Cast {
        input: Box::new(selected.clone()),
        dtype: RecipeDtype::U32,
    };
    assert!(conversion.prepare_encoded_read(&store).unwrap().is_none());
    let reorder = Recipe::Transpose {
        input: Box::new(selected),
        axes: vec![2, 1, 0],
    };
    assert!(reorder.prepare_encoded_read(&store).unwrap().is_none());
    let invalid = selected_rows(vec![usize::MAX]);
    assert!(matches!(
        invalid.prepare_encoded_read(&store),
        Err(RecipeError::InvalidIndices { .. })
    ));
    assert_eq!(
        store.source_diagnostics().unwrap().physical_read_bytes,
        before
    );
}

#[test]
fn packed_row_projection_keeps_the_shared_byte_alignment_refusal() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("packed.safetensors");
    let payload = [0x12, 0x34, 0x56];
    serialize_to_file(
        [(
            "packed",
            TensorView::new(Dtype::F4, vec![2, 3], &payload).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(&path).unwrap();
    let selected = source(
        "packed",
        TensorSelection::Indices {
            axis: 1,
            indices: vec![0, 2],
        },
    );
    assert!(selected.prepare_encoded_read(&store).unwrap().is_none());
    let joined = concatenate(vec![
        source("packed", TensorSelection::Full),
        source("packed", TensorSelection::Full),
    ]);
    assert!(joined.prepare_encoded_read(&store).unwrap().is_none());
    assert_eq!(store.source_diagnostics().unwrap().physical_read_bytes, 0);
}
