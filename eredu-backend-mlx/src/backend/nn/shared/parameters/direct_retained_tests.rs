use super::*;
use eredu_checkpoint::{
    AffineQuantization, BlockFp8Format, BlockFp8ScaleEncoding, WeightQuantization,
};
use safemlx::{ops::QuantizationMode, Device, DeviceType};
use std::cell::Cell;

thread_local! { static HOUSEKEEPING_CALLS: Cell<usize> = const { Cell::new(0) }; }

fn housekeeping() {
    HOUSEKEEPING_CALLS.with(|calls| calls.set(calls.get() + 1));
}

struct HousekeepingHook;
impl Drop for HousekeepingHook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

fn cold<R>(visit: impl FnOnce() -> R) -> R {
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let hook = HousekeepingHook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let result = visit();
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    result
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn affine() -> WeightQuantization {
    WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap())
}

// Materialize actual nonzero physical fields before the measured visit. This
// setup intentionally uses the independent public parameter-tree contract; it
// is not part of the allocation-free retained visitor. Packed values are a
// storage fixture, not a claim of forward/dequantization numerical parity.
fn populate(module: &mut impl PhysicalParameters) {
    for value in module.parameters_mut().flatten().into_values() {
        let shape = value.shape().to_vec();
        let count = shape.iter().map(|&axis| axis as usize).product::<usize>();
        assert!(count > 0);
        *value = match value.dtype() {
            Dtype::Float32 => Array::from_slice(&vec![1.25_f32; count], &shape),
            Dtype::Float16 => Array::from_slice(&vec![half::f16::from_f32(1.25); count], &shape),
            Dtype::Uint32 => Array::from_slice(&vec![0x1234_5678_u32; count], &shape),
            Dtype::Uint8 => Array::from_slice(&vec![0x38_u8; count], &shape),
            dtype => panic!("unexpected fixture dtype: {dtype:?}"),
        };
        let actual = value.evaluated().unwrap();
        match value.dtype() {
            Dtype::Float32 => assert!(actual.try_to_vec::<f32>().unwrap().iter().all(|v| *v > 0.0)),
            Dtype::Float16 => assert!(actual
                .try_to_vec::<half::f16>()
                .unwrap()
                .iter()
                .all(|v| v.to_f32() > 0.0)),
            Dtype::Uint32 => assert!(actual.try_to_vec::<u32>().unwrap().iter().all(|v| *v != 0)),
            Dtype::Uint8 => assert!(actual.try_to_vec::<u8>().unwrap().iter().all(|v| *v != 0)),
            _ => unreachable!(),
        }
    }
}

fn assert_inventory(module: &impl NativeRetainedValues, expected_count: usize, complete: bool) {
    assert_borrowed_sources(module);
    // Build an independent expected set before starting the cold measurement.
    // Compare addresses of actual Array fields, not merely equal metadata.
    let mut expected: Vec<_> = module
        .parameters()
        .flatten()
        .into_values()
        .map(|value| value as *const Array as usize)
        .collect();
    expected.sort_unstable();
    assert_eq!(expected.len(), expected_count);
    let mut actual = [0_usize; 16];
    let mut count = 0;
    let result = cold(|| {
        module.visit_native_retained_values(&mut |value| {
            actual[count] = value.as_array() as *const Array as usize;
            count += 1;
        })
    });
    assert_eq!(result, complete);
    assert_eq!(count, expected_count);
    actual[..count].sort_unstable();
    assert_eq!(&actual[..count], expected.as_slice());
}

#[test]
fn direct_linear_inventory_covers_nonzero_quantization_companions() {
    let stream = stream();
    let affine = AffineQuantization::new(32, 4).unwrap();
    for (format, count) in [
        (LinearFormat::Dense, 2),
        (LinearFormat::Affine(affine), 4),
        (LinearFormat::MxFp4, 3),
        (
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
            ),
            3,
        ),
        (
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0).unwrap(),
            ),
            3,
        ),
    ] {
        let mut linear =
            common::linear::PhysicalLinear::unloaded(32, 2, true, format, &stream).unwrap();
        populate(&mut linear);
        assert_inventory(&linear, count, true);
    }
    let mut no_bias =
        common::linear::PhysicalLinear::unloaded(32, 2, false, LinearFormat::Dense, &stream)
            .unwrap();
    populate(&mut no_bias);
    assert_inventory(&no_bias, 1, true);
}

#[test]
fn direct_grouped_inventory_covers_selector_helpers_and_both_projection_banks() {
    let stream = stream();
    for (quantization, selector_count, gated_count, relu_count) in [
        (Some(affine()), 7, 8, 6),
        (Some(WeightQuantization::MxFp4), 6, 6, 4),
        (None, 5, 4, 2),
    ] {
        let config = common::grouped::TopKGroupSelectorConfig::new(
            1,
            2,
            32,
            common::grouped::TopKGroupScoring::Softmax,
            true,
            1e-6,
            1.0,
            1,
            1,
            true,
            true,
            Some(1e-5),
            false,
            true,
        )
        .unwrap();
        let mut selector = common::grouped::TopKGroupSelector::new_with_quantization(
            config,
            quantization,
            &stream,
        )
        .unwrap();
        populate(&mut selector);
        assert_inventory(&selector, selector_count, true);
        let mut gated = common::grouped::PackedGatedProductGroups::new(
            2,
            32,
            32,
            quantization,
            quantization,
            [true; 2],
            &stream,
        )
        .unwrap();
        populate(&mut gated);
        assert_inventory(&gated, gated_count, true);
        let mut relu =
            common::grouped::PackedRelu2Groups::new(2, 32, 32, [quantization; 2], &stream).unwrap();
        populate(&mut relu);
        assert_inventory(&relu, relu_count, true);
    }
}

#[test]
fn direct_normalization_and_hyper_inventory_visits_every_nonzero_field() {
    let stream = stream();
    let mut norm = nn::RmsNorm::unloaded(4, 1e-5, Dtype::Float32, &stream).unwrap();
    populate(&mut norm);
    assert_inventory(&norm, 1, true);
    let mut connection =
        common::hyper_connections::HyperConnection::unloaded(2, 4, 3, 1e-5, &stream).unwrap();
    populate(&mut connection);
    assert_inventory(&connection, 3, true);
    let mut head =
        common::hyper_connections::HyperHead::unloaded(2, 4, 1e-5, 1e-6, &stream).unwrap();
    populate(&mut head);
    assert_inventory(&head, 3, true);
}

#[test]
fn direct_embedding_inventory_includes_distinct_packed_native_storage() {
    let stream = stream();
    let mut dense = common::linear::PhysicalEmbedding::Dense(
        nn::Embedding::unloaded(2, 32, Dtype::Float32, &stream).unwrap(),
    );
    populate(&mut dense);
    assert_inventory(&dense, 1, true);
    for (mode, count) in [(QuantizationMode::Affine, 3), (QuantizationMode::MxFp4, 2)] {
        let mut physical = common::linear::PhysicalEmbedding::Quantized(
            nn::QuantizedEmbedding::unloaded_with_mode(2, 32, 32, 4, mode, &stream).unwrap(),
        );
        populate(&mut physical);
        assert_inventory(&physical, count, true);
        let common::linear::PhysicalEmbedding::Quantized(embedding) = &mut physical else {
            unreachable!()
        };
        embedding.native = Some(
            common::native_quantization::NativeQuantizedTensor::from_iq_array(
                Array::from_slice(&[3_u8; 68], &[2, 34]),
                &[2, 32],
                eredu_gguf::GgmlType::Q8_0,
                eredu_gguf::Endian::Little,
            )
            .unwrap(),
        );
        let common::linear::PhysicalEmbedding::Quantized(embedding) = &physical else {
            unreachable!()
        };
        assert_borrowed_sources(&physical);
        let packed = embedding.native.as_ref().unwrap().retained_packed_array();
        let expected = packed as *const Array as usize;
        let mut seen = [0_usize; 4];
        let mut actual = 0;
        assert!(cold(|| physical.visit_native_retained_values(
            &mut |value| {
                seen[actual] = value.as_array() as *const Array as usize;
                actual += 1;
            }
        )));
        assert_eq!(actual, count + 1);
        assert_eq!(seen[actual - 1], expected);
        assert_eq!(physical.native_retained_value_slot_bound(), Some(4));
        assert_eq!(
            packed.evaluated().unwrap().try_to_vec::<u8>().unwrap(),
            [3_u8; 68]
        );
    }
}

#[test]
fn direct_borrowed_inventory_keeps_strided_alias_backing_and_lazy_fields() {
    let stream = stream();
    let root = Array::from_slice(&(1..=64).map(|n| n as f32).collect::<Vec<_>>(), &[2, 32]);
    let tail = root.try_index_device((.., 0), &stream).unwrap();
    tail.evaluated().unwrap();
    let source_allocation = root.allocation_info().unwrap().unwrap();
    let mut linear =
        common::linear::PhysicalLinear::unloaded(32, 2, true, LinearFormat::Dense, &stream)
            .unwrap();
    linear.weight.value = root.clone();
    linear.bias.value = Some(tail.clone());
    assert_inventory(&linear, 2, true);
    assert_eq!(
        linear
            .bias
            .value
            .as_ref()
            .unwrap()
            .allocation_info()
            .unwrap(),
        Some(source_allocation)
    );
    assert!(source_allocation.bytes() > tail.size() * std::mem::size_of::<f32>());
    let surviving_alias = linear.bias.value.as_ref().unwrap().clone();
    drop((linear, root, tail));
    assert_eq!(
        surviving_alias
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        [1.0, 33.0]
    );

    let root = Array::from_slice(&[2_f32, 3.0, 4.0, 5.0], &[2, 2]);
    let mut lazy =
        common::linear::PhysicalLinear::unloaded(2, 2, false, LinearFormat::Dense, &stream)
            .unwrap();
    lazy.weight.value = root.square(&stream).unwrap();
    assert_eq!(lazy.weight.value.allocation_info().unwrap(), None);
    assert_inventory(&lazy, 1, true);
    assert_eq!(lazy.weight.value.allocation_info().unwrap(), None);
}

#[test]
fn direct_named_wrapper_propagates_callback_unwind_without_evaluation() {
    let stream = stream();
    let root = Array::from_slice(&[2_f32, 3.0], &[2]);
    let lazy = root.square(&stream).unwrap();
    let native = nn::RmsNorm {
        weight: crate::module::PhysicalParam::new(lazy),
        eps: 1e-5,
    };
    let wrapped = MlxModule::new(
        MlxNamedModule::with_exact_topology(
            native,
            [("weight", ParameterSpec::trainable("norm.weight").unwrap())],
        )
        .unwrap(),
    );
    let mut pointer = 0;
    cold(|| {
        assert!(
            wrapped.visit_retained_values(
                &mut |value| pointer = value.as_array() as *const Array as usize
            )
        )
    });
    assert_eq!(
        pointer,
        &wrapped.inner.inner.weight.value as *const Array as usize
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cold(|| wrapped.visit_retained_values(&mut |_| std::panic::panic_any(73_u32)))
    }));
    assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 73);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    assert_eq!(
        wrapped.inner.inner.weight.value.allocation_info().unwrap(),
        None
    );
}

#[test]
fn packed_embedding_aliases_share_backing_but_keep_both_physical_visit_slots() {
    let stream = stream();
    let mut physical = common::linear::PhysicalEmbedding::Quantized(
        nn::QuantizedEmbedding::unloaded_iq(
            2,
            32,
            eredu_gguf::GgmlType::Q8_0,
            eredu_gguf::Endian::Little,
            &stream,
        )
        .unwrap(),
    );
    populate(&mut physical);
    assert_eq!(physical.native_retained_value_slot_bound(), Some(4));
    assert_inventory(&physical, 1, true);
    let common::linear::PhysicalEmbedding::Quantized(embedding) = &mut physical else {
        unreachable!()
    };
    let expected = embedding
        .inner
        .weight
        .value
        .allocation_info()
        .unwrap()
        .unwrap();
    let native = common::native_quantization::NativeQuantizedTensor::from_iq_array(
        embedding.inner.weight.value.clone(),
        &[2, 32],
        eredu_gguf::GgmlType::Q8_0,
        eredu_gguf::Endian::Little,
    )
    .unwrap();
    let surviving_native = native.clone();
    assert!(std::ptr::eq(
        native.retained_packed_array(),
        surviving_native.retained_packed_array()
    ));
    embedding.native = Some(native);
    let expected_fields = [
        &embedding.inner.weight.value as *const Array as usize,
        embedding.native.as_ref().unwrap().retained_packed_array() as *const Array as usize,
    ];
    assert_borrowed_sources(&physical);
    let mut fields = [0_usize; 4];
    let mut count = 0;
    assert!(cold(|| physical.visit_native_retained_values(
        &mut |value| {
            fields[count] = value.as_array() as *const Array as usize;
            count += 1;
        }
    )));
    assert_eq!(count, 2);
    assert_eq!(fields[..count], expected_fields);
    assert_eq!(physical.native_retained_value_slot_bound(), Some(4));
    assert_eq!(
        surviving_native
            .retained_packed_array()
            .allocation_info()
            .unwrap(),
        Some(expected)
    );
    drop(physical);
    assert_eq!(
        surviving_native
            .retained_packed_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<u8>()
            .unwrap(),
        [0x38_u8; 68]
    );
}

// Independent PhysicalParameters trees are setup-only oracles. The measured
// strict bridge borrows the same actual fields and retained topology names.
fn assert_borrowed_sources(module: &impl NativeRetainedValues) {
    let physical = module.parameters().flatten();
    let trainable = module.trainable_parameters().flatten();
    let topology: BTreeMap<_, _> = physical
        .keys()
        .map(|local| {
            (
                local.to_string(),
                ParameterSpec::trainable(format!("模型.{local}")).unwrap(),
            )
        })
        .collect();
    struct Rows<'a> {
        rows: Vec<(ParameterMetadataView<'a>, &'a MlxTensor)>,
        auxiliary: usize,
    }
    impl<'a> ParameterSourceVisitor<'a, MlxTensor> for Rows<'a> {
        fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a MlxTensor) {
            self.rows.push((metadata, value));
        }
        fn retained(&mut self, _: &'a MlxTensor) {
            self.auxiliary += 1;
        }
    }
    // Preallocate the observer before measuring; no arbitrary rank/slot cap.
    let mut rows = Rows {
        rows: Vec::with_capacity(physical.len()),
        auxiliary: 0,
    };
    cold(|| visit_module_parameter_sources(module, &topology, &mut rows)).unwrap();
    assert_eq!(module.native_parameter_source_count(), Some(physical.len()));
    assert_eq!(rows.rows.len(), physical.len());
    let mut all = Vec::new();
    assert!(module.visit_native_retained_values(
        &mut |value| all.push(value.as_array() as *const Array as usize)
    ));
    assert_eq!(all.len(), physical.len() + rows.auxiliary);
    for (metadata, value) in rows.rows {
        let local = metadata.id().as_str().strip_prefix("模型.").unwrap();
        assert!(std::ptr::eq(
            value.as_array(),
            *physical.get(local).unwrap()
        ));
        assert!(std::ptr::eq(
            metadata.id(),
            &topology.get(local).unwrap().id
        ));
        assert_eq!(metadata.trainable(), trainable.contains_key(local));
    }
}
