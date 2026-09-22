use super::*;
use crate::backend::runtime::checkpoint::{
    bounded_quantization::{submit_original_affine_tile, BoundedQuantizationTarget},
    store::{
        MaterializationPayloadShape, PreparedEncodedInputPlan, PreparedWeightMaterialization,
        WeightMaterialization,
    },
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{CheckpointSource, MemoryWeightStore, SafetensorsWeightStore, TensorSelection},
    AffineQuantization,
};
use safemlx::{Dtype, OperationEvent, PreparedInputRuntime, PreparedOriginalBufferBudget};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

#[test]
fn admitted_encoded_sources_feed_original_affine_tiles_with_the_same_pool() {
    for (dtype, stored) in [
        (Dtype::Float32, SafeDtype::F32),
        (Dtype::Float16, SafeDtype::F16),
        (Dtype::Bfloat16, SafeDtype::BF16),
    ] {
        for (companion, recipe_dtype) in [
            (Dtype::Float32, RecipeDtype::F32),
            (Dtype::Float16, RecipeDtype::F16),
            (Dtype::Bfloat16, RecipeDtype::BF16),
        ] {
            for on_disk in [false, true] {
                let Some(layout) = OperationEvent::cpu_affine_quantize_submission_layout(
                    dtype, companion, 2, 2, 64, 32, 4,
                ) else {
                    assert_ne!(
                        std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref(),
                        Ok("1")
                    );
                    return;
                };
                let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
                let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
                let runtime = PreparedInputRuntime::prepare().unwrap();
                let bytes: Vec<u8> = (0..256)
                    .flat_map(|index| {
                        let value = (index % 16) as f32;
                        match dtype {
                            Dtype::Float16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
                            Dtype::Bfloat16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
                            _ => value.to_le_bytes().to_vec(),
                        }
                    })
                    .collect();
                let directory = tempfile::tempdir().unwrap();
                let source: Box<dyn CheckpointSource> = if on_disk {
                    serialize_to_file(
                        [(
                            "weight",
                            TensorView::new(stored, vec![4, 64], &bytes).unwrap(),
                        )],
                        None,
                        &directory.path().join("model.safetensors"),
                    )
                    .unwrap();
                    Box::new(SafetensorsWeightStore::open(directory.path()).unwrap())
                } else {
                    Box::new(
                        MemoryWeightStore::from_safetensors([(
                            "weight".into(),
                            stored,
                            vec![4, 64],
                            bytes,
                        )])
                        .unwrap(),
                    )
                };
                let read = DerivedWeightRecipe::source(
                    "weight",
                    TensorSelection::Indices {
                        axis: 0,
                        indices: vec![3, 1],
                    },
                )
                .prepare_encoded_read(source.as_ref())
                .unwrap()
                .unwrap();
                let input_plan =
                    PreparedEncodedInputPlan::new(&read, &runtime, &[2, 64], dtype).unwrap();
                let input_bytes = input_plan.required_bytes().unwrap();
                let shape = MaterializationPayloadShape {
                    inputs: 1,
                    outputs: 3,
                    pending_sources: 0,
                };
                let controls =
                    PreparedWeightMaterialization::bank_layout_with_payload::<(), ()>(1, shape)
                        .unwrap()
                        .prepared_slot_control_bytes;
                let budget = PreparedOriginalBufferBudget::try_new(
                    &runtime,
                    layout.physical_capacity(&runtime).unwrap(),
                    (),
                )
                .unwrap()
                .try_allocate()
                .unwrap();
                let target = BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                    .unwrap()
                    .with_affine_companion_dtype(recipe_dtype.clone())
                    .unwrap();
                // Cold metadata/runtime and transport Vec are fixture owners;
                // the real payload and read scratch receive source admission.
                let mut inputs = Vec::with_capacity(1);
                component_destinations_with_native_budget(
                    0,
                    0,
                    0,
                    None,
                    0,
                    |_| input_bytes,
                    0,
                    controls,
                    Some(&budget),
                    |controls, observer, pool, host| {
                        let before = pool.fixture_host_charge().unwrap();
                        let input = input_plan.prepare(pool).unwrap();
                        input.validate_pool(pool).unwrap();
                        inputs.push(input.output().try_prepared_source_array().unwrap());
                        drop(input);
                        assert_eq!(pool.fixture_host_charge().unwrap(), before + input_bytes);
                        let ready = PreparedWeightMaterialization::try_new(controls.clone(), shape)
                            .unwrap_or_else(|_| panic!("tile slot"));
                        let owner = WeightMaterialization::prepare_retained_impl(
                            inputs,
                            Vec::new(),
                            Some((ready, observer.clone())),
                            None,
                        )
                        .unwrap();
                        let owner = submit_original_affine_tile(
                            owner,
                            AffineQuantization::new(32, 4).unwrap(),
                            &target,
                            &stream,
                            layout,
                        )
                        .unwrap();
                        owner.wait().unwrap();
                        let mut outputs = owner.completed_outputs();
                        let weight = outputs.next().unwrap().unwrap();
                        assert_eq!(weight.as_slice::<u32>().len(), 16);
                        for (index, &word) in weight.as_slice::<u32>().iter().enumerate() {
                            assert_eq!(
                                word,
                                if index % 2 == 0 {
                                    0x89abcdef
                                } else {
                                    0x01234567
                                }
                            );
                        }
                        for expected in [-1.0, 15.0] {
                            let output = outputs.next().unwrap().unwrap();
                            for index in 0..4 {
                                let actual = match companion {
                                    Dtype::Float16 => {
                                        output.as_slice::<half::f16>()[index].to_f32()
                                    }
                                    Dtype::Bfloat16 => {
                                        output.as_slice::<half::bf16>()[index].to_f32()
                                    }
                                    _ => output.as_slice::<f32>()[index],
                                };
                                assert_eq!(actual, expected);
                            }
                        }
                        assert!(outputs.next().is_none());
                        drop((outputs, weight));
                        owner.finish().unwrap();
                        drop(host);
                    },
                );
                assert_eq!(budget.occupied_bytes(), 0);
            }
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
