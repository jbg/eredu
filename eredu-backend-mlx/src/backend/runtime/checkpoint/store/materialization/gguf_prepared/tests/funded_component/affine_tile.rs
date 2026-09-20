use super::*;
use crate::backend::runtime::checkpoint::{
    bounded_quantization::{BoundedQuantizationTarget, submit_original_affine_tile},
    store::{MaterializationPayloadShape, PreparedWeightMaterialization, WeightMaterialization},
};
use eredu_checkpoint::{AffineQuantization, recipe::RecipeDtype};
use safemlx::{
    CpuAffineQuantizeSubmissionLayout, Dtype, OperationEvent, PreparedInputRuntime,
    PreparedOriginalBufferBudget,
};

fn layout(
    dtype: Dtype,
    companion: Dtype,
    rank: usize,
) -> Option<CpuAffineQuantizeSubmissionLayout> {
    let layout =
        OperationEvent::cpu_affine_quantize_submission_layout(dtype, companion, rank, 2, 64, 32, 4);
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    layout
}

fn target(dtype: Dtype) -> BoundedQuantizationTarget {
    BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
        .unwrap()
        .with_affine_companion_dtype(match dtype {
            Dtype::Float16 => RecipeDtype::F16,
            Dtype::Bfloat16 => RecipeDtype::BF16,
            _ => RecipeDtype::F32,
        })
        .unwrap()
}

fn source(dtype: Dtype, rank: usize, strided: bool, stream: &Stream) -> Array {
    let values: Vec<f32> = (0..128)
        .map(|index| {
            let index = if strided {
                (index % 2) * 64 + index / 2
            } else {
                index
            };
            (index % 16) as f32
        })
        .collect();
    let mut shape = vec![1; rank];
    shape[0] = if strided { 64 } else { 2 };
    shape[rank - 1] = if strided { 2 } else { 64 };
    let input = match dtype {
        Dtype::Float16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            &shape,
        ),
        Dtype::Bfloat16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            &shape,
        ),
        _ => Array::from_slice(&values, &shape),
    };
    let input = if strided {
        let mut axes: Vec<i32> = (0..rank as i32).collect();
        axes.swap(0, rank - 1);
        input.transpose_axes(&axes, stream).unwrap()
    } else {
        input
    };
    input.evaluated().unwrap();
    input.evaluated().unwrap();
    input
}

fn with_owner(
    input: Array,
    layout: CpuAffineQuantizeSubmissionLayout,
    operation: impl FnOnce(WeightMaterialization, &OriginalScopeObserver),
) {
    let shape = MaterializationPayloadShape {
        inputs: 1,
        outputs: 3,
        pending_sources: 0,
    };
    let controls = PreparedWeightMaterialization::bank_layout_with_payload::<(), ()>(1, shape)
        .unwrap()
        .prepared_slot_control_bytes;
    // Component input/runtime and native capacity are independent fixture owners.
    // The source producer and whole model admission are not exercised here.
    let inputs = vec![input];
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let capacity = layout.physical_capacity(&runtime).unwrap();
    let budget = PreparedOriginalBufferBudget::try_new(&runtime, capacity, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    component_destinations_with_native_budget(
        0,
        0,
        0,
        None,
        0,
        |_| 0,
        0,
        controls,
        Some(&budget),
        |controls, observer, _, host| {
            let ready = PreparedWeightMaterialization::try_new(controls.clone(), shape)
                .unwrap_or_else(|_| panic!("prepared tile storage"));
            let owner = WeightMaterialization::prepare_retained_impl(
                inputs,
                Vec::new(),
                Some((ready, observer.clone())),
            )
            .unwrap();
            operation(owner, observer);
            assert!(budget.occupied_bytes() <= capacity);
            drop(host);
        },
    );
    assert_eq!(budget.occupied_bytes(), 0);
}

#[test]
fn original_affine_tile_uses_shared_producer_and_retained_completion() {
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for companion in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for rank in [2, 4] {
                for strided in [false, true] {
                    let Some(layout) = layout(dtype, companion, rank) else {
                        return;
                    };
                    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
                    let _runtime =
                        PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
                    let input = source(dtype, rank, strided, &stream);
                    let target = target(companion);
                    with_owner(input, layout, |owner, _| {
                        let pointer = owner.outputs().as_ptr();
                        let owner = submit_original_affine_tile(
                            owner,
                            AffineQuantization::new(32, 4).unwrap(),
                            &target,
                            &stream,
                            layout,
                        )
                        .unwrap();
                        owner.wait().unwrap();
                        assert_eq!(owner.outputs().as_ptr(), pointer);
                        assert_eq!(
                            owner
                                .outputs()
                                .iter()
                                .map(Array::nbytes)
                                .collect::<Vec<_>>(),
                            layout.output_bytes()
                        );
                        assert!(owner.outputs().iter().all(|output| output.ndim() == rank));
                        let mut outputs = owner.completed_outputs();
                        let packed = outputs.next().unwrap().unwrap();
                        for (index, &word) in packed.as_slice::<u32>().iter().enumerate() {
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
                        drop(outputs);
                        drop(packed);
                        owner.finish().unwrap();
                    });
                }
            }
        }
    }
}

#[test]
fn original_affine_tile_rejects_mismatched_quote_before_native_work() {
    let Some(layout) = layout(Dtype::Float32, Dtype::Float16, 2) else {
        return;
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let input = source(Dtype::Float32, 2, false, &stream);
    let target = target(Dtype::Float32);
    with_owner(input, layout, |owner, observer| {
        assert!(matches!(
            submit_original_affine_tile(
                owner,
                AffineQuantization::new(32, 4).unwrap(),
                &target,
                &stream,
                layout,
            ),
            Err(crate::backend::error::Error::PrefillControl(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        assert!(!observer.status().failed());
    });
}

#[test]
fn original_affine_tile_requires_an_original_owner() {
    let Some(layout) = layout(Dtype::Float32, Dtype::Float32, 2) else {
        return;
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let owner = WeightMaterialization::prepare_retained(
        vec![source(Dtype::Float32, 2, false, &stream)],
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        submit_original_affine_tile(
            owner,
            AffineQuantization::new(32, 4).unwrap(),
            &target(Dtype::Float32),
            &stream,
            layout,
        ),
        Err(crate::backend::error::Error::CheckpointMaterialization(
            CheckpointMaterializationError::OriginalOperationDomain
        ))
    ));
}

#[test]
fn original_affine_tile_refuses_a_lazy_source_without_evaluating_it() {
    let Some(layout) = layout(Dtype::Float32, Dtype::Float32, 2) else {
        return;
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let input = source(Dtype::Float32, 2, false, &stream);
    let lazy = input.add(&input, &stream).unwrap();
    with_owner(lazy, layout, |owner, observer| {
        assert!(matches!(
            submit_original_affine_tile(
                owner,
                AffineQuantization::new(32, 4).unwrap(),
                &target(Dtype::Float32),
                &stream,
                layout,
            ),
            Err(crate::backend::error::Error::CheckpointMaterialization(
                CheckpointMaterializationError::OriginalNative(_)
            ))
        ));
        assert!(!observer.status().failed());
    });
}
