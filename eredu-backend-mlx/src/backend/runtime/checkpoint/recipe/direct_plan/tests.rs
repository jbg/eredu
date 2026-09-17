use super::*;
use safemlx::{Device, DeviceType};

fn streams() -> (Stream, Stream) {
    (
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
    )
}

#[test]
fn retained_plan_prices_every_owner_output_without_rereading_or_readmitting_sources() {
    let (_directory, store) = super::super::tests::fixture();
    let source = DerivedWeightRecipe::source("left", TensorSelection::Full);
    let bindings = [
        WeightBinding::from_recipe("first", source.clone(), 16).unwrap(),
        WeightBinding::from_recipe("second", source, 16).unwrap(),
    ];
    let plan = PreparedDirectReadPlan::prepare(store.as_ref(), &bindings).unwrap();
    let prepared = store.source_diagnostics().unwrap();
    let facts = MetalAllocationFacts::current_host().unwrap();
    let cloned = plan.clone();
    assert_eq!(
        plan.workspace_bound(facts).unwrap(),
        cloned.workspace_bound(facts).unwrap()
    );
    assert_eq!(
        plan.workspace_bound(facts).unwrap().bytes(),
        Some(2 * facts.buffer_capacity(16).unwrap())
    );
    assert_eq!(plan.host_staging_bytes(), 0);
    assert_eq!(plan.outputs()[0].binding(), &bindings[0]);
    assert_eq!(plan.outputs()[0].shape(), [2, 2]);
    assert_eq!(plan.outputs()[0].dtype(), Dtype::Int32);
    assert_eq!(store.source_diagnostics().unwrap(), prepared);
    assert_eq!(prepared.physical_reads, 0);
    let (source_stream, execution_stream) = streams();
    let mut custody = Vec::new();
    let arrays = cloned
        .materialize(&source_stream, &execution_stream, |array| {
            custody.push(array.clone())
        })
        .unwrap();
    safemlx::transforms::eval(arrays.values()).unwrap();
    execution_stream.synchronize().unwrap();
    for array in arrays.values() {
        assert_eq!(
            array.evaluated().unwrap().try_as_slice::<i32>().unwrap(),
            [1, 2, 3, 4]
        );
    }
    // Same checkpoint source is read into two independent owner allocations.
    let first = arrays["first"].allocation_info().unwrap().unwrap();
    let second = arrays["second"].allocation_info().unwrap().unwrap();
    assert_ne!(first.identity(), second.identity());
    assert!(first.bytes() as u64 <= facts.buffer_capacity(16).unwrap());
    assert!(second.bytes() as u64 <= facts.buffer_capacity(16).unwrap());
    // Two direct inputs enter custody before both lazy stream copies.
    assert_eq!(custody.len(), 4);
    assert_eq!(custody[0].allocation_info().unwrap().unwrap(), first);
    assert_eq!(custody[1].allocation_info().unwrap().unwrap(), second);
}

#[test]
fn ordinary_and_alias_paths_are_unpriced_but_invalid_metadata_keeps_original_error() {
    let (_directory, store) = super::super::tests::fixture();
    let cast = WeightBinding::from_recipe(
        "cast",
        DerivedWeightRecipe::Cast {
            input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
            dtype: RecipeDtype::F32,
        },
        16,
    )
    .unwrap();
    let alias = WeightBinding::alias("alias", "owner", 16).unwrap();
    for binding in [cast, alias] {
        assert!(matches!(
            PreparedDirectReadPlan::prepare(store.as_ref(), &[binding]),
            Err(PreparedDirectReadError::Unpriced {
                source: WorkingMemoryError::UnknownBound,
                ..
            })
        ));
    }
    let invalid = WeightBinding::new("bad", "left", TensorSelection::Full, 8).unwrap();
    assert!(matches!(
        PreparedDirectReadPlan::prepare(store.as_ref(), &[invalid]),
        Err(PreparedDirectReadError::Recipe(
            WeightRecipeError::Preflight(_)
        ))
    ));
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn retained_read_rejects_replaced_admitted_file_without_publishing_partial_arrays() {
    let (directory, store) = super::super::tests::fixture();
    let binding = WeightBinding::new("left", "left", TensorSelection::Full, 16).unwrap();
    let plan = PreparedDirectReadPlan::prepare(store.as_ref(), &[binding]).unwrap();
    let file = directory.path().join("model.safetensors");
    let original = std::fs::read(&file).unwrap();
    std::fs::rename(&file, directory.path().join("original.safetensors")).unwrap();
    std::fs::write(&file, original).unwrap();
    let (source_stream, execution_stream) = streams();
    let mut retained = 0;
    assert!(matches!(
        plan.materialize(&source_stream, &execution_stream, |_| retained += 1),
        Err(PreparedDirectReadError::Recipe(
            WeightRecipeError::CheckpointStore(_)
        ))
    ));
    assert_eq!(
        retained, 0,
        "failed direct initialization exposes no partially initialized tensor"
    );
}

#[test]
fn later_batch_source_failure_keeps_earlier_stream_copies_in_caller_custody() {
    let (directory, store) = super::super::tests::fixture();
    let bindings = (0..65)
        .map(|index| {
            WeightBinding::new(format!("output{index}"), "left", TensorSelection::Full, 16).unwrap()
        })
        .collect::<Vec<_>>();
    let plan = PreparedDirectReadPlan::prepare(store.as_ref(), &bindings).unwrap();
    assert_eq!(
        plan.batches.len(),
        2,
        "portable parameter batching selects a real second read"
    );
    let file = directory.path().join("model.safetensors");
    let original = std::fs::read(&file).unwrap();
    let (source_stream, execution_stream) = streams();
    let mut retained = Vec::new();
    let result = plan.materialize(&source_stream, &execution_stream, |array| {
        retained.push(array.clone());
        if retained.len() == 65 {
            // First batch inputs and its first copy already exist. Changing the
            // source must fail the later retained read rather than readmitting.
            std::fs::rename(&file, directory.path().join("original.safetensors")).unwrap();
            std::fs::write(&file, &original).unwrap();
        }
    });
    assert!(matches!(
        result,
        Err(PreparedDirectReadError::Recipe(
            WeightRecipeError::CheckpointStore(_)
        ))
    ));
    assert_eq!(retained.len(), 128);
    safemlx::transforms::eval(retained.iter()).unwrap();
    execution_stream.synchronize().unwrap();
    for array in &retained {
        assert_eq!(
            array.evaluated().unwrap().try_as_slice::<i32>().unwrap(),
            [1, 2, 3, 4]
        );
    }
}
