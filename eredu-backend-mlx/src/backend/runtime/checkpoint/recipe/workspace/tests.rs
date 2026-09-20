use super::*;
use eredu_runtime::WeightBinding;
use safemlx::{Device, DeviceType};

#[test]
fn direct_materialization_quote_requires_exact_binding_bytes_and_preserves_ordinary_gap() {
    let (_directory, store) = super::super::tests::fixture();
    let allocation = NativeAllocationFacts::current_host().unwrap();
    let direct = WeightBinding::new("left", "left", TensorSelection::Full, 16).unwrap();
    let converted = WeightBinding::from_recipe(
        "converted",
        DerivedWeightRecipe::Cast {
            input: Box::new(direct.source_recipe()),
            dtype: RecipeDtype::F32,
        },
        16,
    )
    .unwrap();
    let wrong = WeightBinding::new("wrong", "left", TensorSelection::Full, 8).unwrap();
    let batches = plan_binding_reads(store.as_ref(), [&direct, &converted, &wrong]).unwrap();
    assert_eq!(batches.len(), 3);
    let capacity = allocation.buffer_capacity(16).unwrap();
    assert_eq!(
        batches[0].workspace_bound(allocation).unwrap().bytes(),
        Some(capacity)
    );
    assert_eq!(
        batches[1].workspace_bound(allocation).unwrap().bytes(),
        None
    );
    assert!(batches[2].workspace_bound(allocation).is_err());
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
}

#[test]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_direct_materialization_batch_peaks_include_retained_sources_and_stream_copies() {
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let execution = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let allocation = NativeAllocationFacts::current_host().unwrap();
    for positions in [37, 4097, 16385] {
        let directory = tempfile::tempdir().unwrap();
        let values = (0..positions)
            .flat_map(|value| (value as u32 + 1).to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "tokens",
                TensorView::new(SafeDtype::U32, vec![positions], &values).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store =
            eredu_checkpoint::store::SafetensorsWeightStore::open(directory.path()).unwrap();
        let direct = DerivedWeightRecipe::source("tokens", TensorSelection::Full);
        let bindings = [
            WeightBinding::from_recipe("source", direct.clone(), positions as u64 * 4).unwrap(),
            WeightBinding::from_recipe(
                "joined",
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![direct.clone(), direct],
                },
                positions as u64 * 8,
            )
            .unwrap(),
        ];
        for copies in [false, true] {
            stream.synchronize().unwrap();
            execution.synchronize().unwrap();
            let before = safemlx::memory::active_memory().unwrap();
            let reads_before = store.source_diagnostics().unwrap().physical_reads;
            let mut batches = plan_binding_reads(&store, &bindings).unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.remove(0);
            let quote = batch.workspace_bound(allocation).unwrap().bytes().unwrap();
            assert_eq!(safemlx::memory::active_memory().unwrap(), before);
            assert_eq!(
                store.source_diagnostics().unwrap().physical_reads,
                reads_before
            );
            safemlx::memory::reset_peak_memory().unwrap();
            let BindingReadBatch::Direct { reads, .. } = batch else {
                panic!("selected direct read")
            };
            let sources = DirectRecipeRead::materialize_many(reads).unwrap();
            let outputs = sources
                .iter()
                .map(|source| {
                    if copies {
                        source.copy(&execution).unwrap()
                    } else {
                        source.clone()
                    }
                })
                .collect::<Vec<_>>();
            safemlx::transforms::eval(outputs.iter()).unwrap();
            stream.synchronize().unwrap();
            execution.synchronize().unwrap();
            let observed = safemlx::memory::peak_memory()
                .unwrap()
                .saturating_sub(before) as u64;
            assert!(observed <= quote, "observed {observed} > bound {quote}");
            for (index, output) in outputs.iter().enumerate() {
                let evaluated = output.evaluated().unwrap();
                let actual = evaluated.try_as_slice::<u32>().unwrap();
                assert_eq!(actual.len(), positions * (index + 1));
                assert!(actual
                    .iter()
                    .enumerate()
                    .all(|(position, value)| *value == (position % positions + 1) as u32));
                let input_backing = sources[index].allocation_info().unwrap().unwrap();
                let output_backing = output.allocation_info().unwrap().unwrap();
                // MLX Copy transports dependency ordering but shares the full
                // backing, including on a different execution stream.
                assert_eq!(input_backing, output_backing);
            }
            eprintln!("direct materialization positions={positions} copies={copies} observed={observed} bound={quote}");
        }
    }
}
